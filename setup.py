"""EvalOps setup hook.

Called by the gateway lifespan to verify tables are initialized.
"""

from fastapi import FastAPI

from common.observability.logger import get_logger

logger = get_logger("evalops")


async def init_app_state(app: FastAPI, settings) -> None:
    """Initialize SQL schemas for EvalOps."""
    pass


async def shutdown_app_state(app: FastAPI, settings) -> None:
    """Cleanup on shutdown."""
    logger.info("EvalOps dashboard shut down")
