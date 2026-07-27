"""Unit test for S5-10a: LangGraph Flow Trace Collector & Event Interceptor."""

import pytest
import asyncio
import uuid
from projects.evalops.src.runner.trace_collector import TraceCollector


@pytest.mark.asyncio
async def test_trace_collector_non_blocking_queue():
    """Verify TraceCollector enqueues events asynchronously and processes step payloads."""
    collector = TraceCollector(max_queue_size=10)
    await collector.start()

    run_id = str(uuid.uuid4())
    event_payload = {
        "run_id": run_id,
        "workflow_id": "wf_test_routing",
        "node_id": "router_step_1",
        "node_type": "router",
        "input_state": {"user_intent": "billing_inquiry"},
        "output_state": {"target_agent": "billing_agent", "confidence": 0.98},
        "duration_ms": 45.2
    }

    # Non-blocking emission
    success = collector.emit_event(event_payload)
    assert success is True

    # Give background worker time to consume item
    await asyncio.sleep(0.6)
    await collector.stop()

    assert len(collector.collected_traces) == 1
    trace = collector.collected_traces[0]
    assert trace["run_id"] == run_id
    assert trace["workflow_id"] == "wf_test_routing"
    assert trace["node_id"] == "router_step_1"
    assert trace["node_type"] == "router"
    assert trace["input_state"] == {"user_intent": "billing_inquiry"}
    assert trace["output_state"] == {"target_agent": "billing_agent", "confidence": 0.98}
    assert trace["latency_ms"] == 45.2
