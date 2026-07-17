"""Main entrypoint for the EvalOps FastAPI application."""

import logging
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from common.config import get_settings
from projects.evalops.src.utils.logging import setup_logging

logger = logging.getLogger("evalops")

# 1. Setup Application config and logging
settings = get_settings()
setup_logging(settings.app_env)

logger.info("Initializing EvalOps application in env: %s", settings.app_env)

# 2. Initialize FastAPI app
from common.observability.tracing import setup_tracing
setup_tracing("evalops")

app = FastAPI(
    title="EvalOps Continuous AI Evaluation & Observability Harness",
    description="Evaluation Gate, Router Gateway Benchmarking, and Diagnostic Suite.",
    version="0.1.0",
)

# 3. Setup OpenTelemetry tracing
try:
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry FastAPI instrumentation initialized.")
except Exception as e:
    logger.warning("Could not initialize OpenTelemetry instrumentation: %s", e)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple API health endpoint to confirm the server is running.

    Returns:
        A dictionary showing system health status.
    """
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "database": "configured" if settings.database_url else "missing",
        "qdrant": "configured" if settings.qdrant_url else "missing",
    }


@app.post("/chat/completions")
async def mock_chat_completions(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Mock routing proxy gateway for chat completions.

    Used for benchmarking latency, cost metrics, and router classification.

    Args:
        payload: Chat completion requests parameters.

    Returns:
        A mock chat completion response structure.
    """
    if "messages" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'messages' in payload.")

    logger.debug("Received mock proxy request: %s", payload)

    # Basic mock completion response mimicking OpenAI schema
    return {
        "id": "chatcmpl-mock123",
        "object": "chat.completion",
        "model": payload.get("model", "auto"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a placeholder response from the routing gateway.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
        },
    }
