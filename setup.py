"""EvalOps setup hook.

Called by the gateway lifespan to verify tables are initialized.
"""

from fastapi import FastAPI

from common.observability.logger import get_logger
from projects.evalops.src.database.models import Base
from common.clients.postgres import get_engine

logger = get_logger("evalops")


async def init_app_state(app: FastAPI, settings) -> None:
    """Initialize SQL schemas for EvalOps."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("EvalOps database tables verified/created successfully.")
    except Exception as e:
        logger.error("Failed to initialize EvalOps database: %s", e)


async def shutdown_app_state(app: FastAPI, settings) -> None:
    """Cleanup on shutdown."""
    logger.info("EvalOps dashboard shut down")
