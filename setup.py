"""EvalOps setup hook.

Called by the gateway lifespan to verify tables are initialized.
"""

import asyncio
from fastapi import FastAPI

from common.observability.logger import get_logger

logger = get_logger("evalops")


async def init_app_state(app: FastAPI, settings) -> None:
    """Initialize SQL schemas for EvalOps and launch background Kafka consumer."""
    if getattr(settings, "EVALOPS_CONSUMER_ENABLED", False):
        try:
            from projects.evalops.src.worker import run_evalops_consumer
            app.state.evalops_consumer_task = asyncio.create_task(run_evalops_consumer(app))
            logger.info("EvalOps background Kafka consumer started successfully.")
        except Exception as e:
            logger.error("Failed to start EvalOps background Kafka consumer: %s", e)
    else:
        logger.info("EvalOps background consumer disabled (EVALOPS_CONSUMER_ENABLED=false).")


async def shutdown_app_state(app: FastAPI, settings) -> None:
    """Cleanup on shutdown."""
    if hasattr(app.state, "evalops_consumer_task"):
        logger.info("Stopping EvalOps background Kafka consumer...")
        app.state.evalops_consumer_task.cancel()
        try:
            await app.state.evalops_consumer_task
        except asyncio.CancelledError:
            pass
        logger.info("EvalOps background Kafka consumer task cancelled.")
    logger.info("EvalOps dashboard shut down")
