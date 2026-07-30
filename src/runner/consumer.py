"""Kafka-driven asynchronous evaluation runner consumer (S6-07b).

Listens to 'agent-eval-trigger' Kafka events, dispatches evaluation test cases
against target agents or workflows through runner dispatch, and updates database records.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import select

from common.config.settings import settings
from common.clients.postgres import get_sessionmaker
from common.models.database import EvalRunHistory, EvalTestCase, EvalTestSuite
from projects.evalops.src.runner.dispatch import dispatch_run

logger = logging.getLogger("evalops.runner.consumer")

EVAL_TRIGGER_TOPIC = "agent-eval-trigger"


def publish_eval_trigger_event(
    hub_id: str,
    suite_id: str,
    run_id: str,
    agent_id: Optional[str] = None,
    framework: Optional[str] = "both",
    metrics: Optional[list[str]] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> bool:
    """Publishes an evaluation trigger event to Kafka."""
    try:
        from confluent_kafka import Producer
        conf = {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "client.id": "evalops-producer",
        }
        producer = Producer(conf)
        payload = {
            "event": "eval_run_trigger",
            "hub_id": hub_id,
            "suite_id": suite_id,
            "run_id": run_id,
            "agent_id": agent_id,
            "framework": framework,
            "metrics": metrics,
            "thresholds": thresholds,
            "timestamp": datetime.utcnow().isoformat()
        }
        producer.produce(EVAL_TRIGGER_TOPIC, json.dumps(payload).encode("utf-8"))
        producer.flush(timeout=2.0)
        logger.info(f"Published eval trigger event for run {run_id} in hub {hub_id} to topic {EVAL_TRIGGER_TOPIC}")
        return True
    except Exception as e:
        logger.warning(f"Failed to publish eval trigger to Kafka ({e}). Will run evaluation directly in background fallback mode.")
        return False


async def process_eval_run(event_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Executes evaluation benchmarks for an eval suite run and updates database records."""
    hub_id = event_payload.get("hub_id")
    suite_id = event_payload.get("suite_id")
    run_id = event_payload.get("run_id")
    framework_selected = event_payload.get("framework") or "both"

    # Backwards compatibility check
    agent_id = event_payload.get("agent_id")
    if not hub_id or not suite_id or not run_id:
        if not (agent_id and run_id):
            logger.error("Invalid event payload for eval run: missing hub_id, suite_id, or run_id")
            return {"status": "error", "message": "Missing hub_id, suite_id, or run_id"}

    logger.info(f"Starting evaluation run {run_id} (Suite: {suite_id}, Hub: {hub_id})")
    SessionLocal = get_sessionmaker()

    async with SessionLocal() as db:
        suite = None
        if suite_id:
            suite_res = await db.execute(
                select(EvalTestSuite).where(
                    EvalTestSuite.id == suite_id,
                    EvalTestSuite.hub_id == hub_id,
                )
            )
            suite = suite_res.scalar_one_or_none()

        if not suite:
            if agent_id:
                suite_res = await db.execute(select(EvalTestSuite).where(EvalTestSuite.target_id == agent_id))
                suite = suite_res.scalar_one_or_none()
            if not suite:
                logger.error(f"EvalTestSuite {suite_id} not found in DB for run {run_id}")
                return {"status": "error", "message": "EvalTestSuite not found"}

        eval_hub_id = hub_id or suite.hub_id

        # Fetch test cases
        cases_res = await db.execute(select(EvalTestCase).where(EvalTestCase.suite_id == suite.id))
        test_cases = cases_res.scalars().all()

        if not test_cases:
            logger.warning(f"No test cases found for suite {suite.id}. Completing run with 0 cases.")
            run_res = await db.execute(
                select(EvalRunHistory).where(
                    EvalRunHistory.id == run_id,
                    EvalRunHistory.hub_id == eval_hub_id,
                )
            )
            history_record = run_res.scalar_one_or_none()
            if history_record:
                history_record.run_status = "completed"
                history_record.total_test_cases = 0
                history_record.passed_count = 0
                history_record.failed_count = 0
                await db.commit()
            return {"status": "completed", "run_id": run_id, "cases_evaluated": 0}

        outcomes = await dispatch_run(
            db,
            eval_hub_id=eval_hub_id,
            suite=suite,
            cases=test_cases,
            run_id=run_id,
            framework=framework_selected,
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "cases_evaluated": len(outcomes),
        }


# Alias for legacy invocations
process_agent_eval_run = process_eval_run


async def run_eval_kafka_consumer() -> None:
    """Kafka consumer loop for processing evaluation trigger events."""
    logger.info("Initializing Agent Evaluation Kafka Consumer...")
    try:
        from confluent_kafka import Consumer, KafkaError
        conf = {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "evalops-agent-eval-runner-group",
            "auto.offset.reset": "earliest",
        }
        consumer = Consumer(conf)
        consumer.subscribe([EVAL_TRIGGER_TOPIC])
    except Exception as e:
        logger.warning(f"Agent Eval Kafka consumer initialization failed: {e}. Worker will rely on direct async invocation.")
        return

    logger.info(f"Agent Eval Kafka Consumer subscribed to topic {EVAL_TRIGGER_TOPIC}.")
    try:
        while True:
            msg = await asyncio.to_thread(consumer.poll, 1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Kafka consumer error: {msg.error()}")
                    await asyncio.sleep(2.0)
                    continue

            try:
                val = json.loads(msg.value().decode("utf-8"))
                logger.info(f"Received eval trigger event for run_id: {val.get('run_id')}")
                asyncio.create_task(process_eval_run(val))
            except Exception as pe:
                logger.error(f"Failed to parse eval trigger message: {pe}")
    except asyncio.CancelledError:
        logger.info("Eval Kafka consumer loop cancelled.")
    finally:
        try:
            consumer.close()
        except Exception:
            pass
