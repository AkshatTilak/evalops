"""TraceCollector for intercepting and queueing LangGraph workflow execution events.
S5-10a: Non-blocking asynchronous event listener & trace formulation.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

from common.models.database import EvalFlowTrace

logger = logging.getLogger("evalops.trace_collector")


class TraceCollector:
    """Listens to LangGraph execution events and asynchronously processes trace records."""

    def __init__(self, db_session_factory: Optional[Callable] = None, max_queue_size: int = 1000):
        self.db_session_factory = db_session_factory
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.worker_task: Optional[asyncio.Task] = None
        self.collected_traces: List[Dict[str, Any]] = []
        self._is_running: bool = False

    async def start(self):
        """Starts the background worker to consume events from non-blocking queue."""
        if self._is_running:
            return
        self._is_running = True
        self.worker_task = asyncio.create_task(self._process_queue())
        logger.info("TraceCollector background processing worker started.")

    async def stop(self):
        """Stops the background worker gracefully."""
        self._is_running = False
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("TraceCollector stopped.")

    def emit_event(self, event_data: Dict[str, Any]) -> bool:
        """Enqueues an execution event synchronously without blocking the calling loop.
        
        Event Data Payload format:
        {
            "run_id": str (UUID),
            "workflow_id": str,
            "node_id": str,
            "node_type": str,
            "input_state": dict,
            "output_state": dict,
            "latency_ms": float,
            "timestamp": str (ISO) or datetime
        }
        """
        try:
            self.queue.put_nowait(event_data)
            return True
        except asyncio.QueueFull:
            logger.warning("TraceCollector queue full. Dropping trace event.")
            return False

    async def emit_event_async(self, event_data: Dict[str, Any]) -> bool:
        """Asynchronously enqueues an execution event."""
        try:
            await self.queue.put(event_data)
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue trace event: {e}")
            return False

    def parse_event_to_trace(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and structures event data into an EvalFlowTrace dictionary payload."""
        timestamp_val = event_data.get("timestamp")
        if isinstance(timestamp_val, str):
            try:
                timestamp_dt = datetime.fromisoformat(timestamp_val)
            except ValueError:
                timestamp_dt = datetime.utcnow()
        elif isinstance(timestamp_val, datetime):
            timestamp_dt = timestamp_val
        else:
            timestamp_dt = datetime.utcnow()

        return {
            "id": event_data.get("id", str(uuid.uuid4())),
            "run_id": event_data.get("run_id"),
            "workflow_id": event_data.get("workflow_id", "default_flow"),
            "node_id": event_data.get("node_id", "unknown_node"),
            "node_type": event_data.get("node_type", "action"),
            "input_state": event_data.get("input_state") or event_data.get("inputs") or {},
            "output_state": event_data.get("output_state") or event_data.get("outputs") or {},
            "latency_ms": float(event_data.get("latency_ms") or event_data.get("duration_ms") or 0.0),
            "timestamp": timestamp_dt
        }

    async def _process_queue(self):
        """Worker loop processing trace items asynchronously."""
        while self._is_running:
            try:
                event_data = await asyncio.wait_for(self.queue.get(), timeout=0.5)
                trace_dict = self.parse_event_to_trace(event_data)
                self.collected_traces.append(trace_dict)

                if self.db_session_factory:
                    try:
                        async with self.db_session_factory() as session:
                            db_trace = EvalFlowTrace(**trace_dict)
                            session.add(db_trace)
                            await session.commit()
                    except Exception as db_err:
                        logger.error(f"Error persisting flow trace to DB: {db_err}")

                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in TraceCollector processing queue: {e}")
