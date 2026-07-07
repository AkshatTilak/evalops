"""Unit tests for Model Registry, Qdrant Client, Inference Client, and LiteLLM Client.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from common.schemas.model_registry import ModelSpec, ModelRole, ModelMode
from common.clients.inference import InferenceClient
from common.clients.litellm import completion_with_fallback, truncate_messages as litellm_truncate


@pytest.mark.asyncio
async def test_model_registry_resolution_and_overrides(mocker):
    """Test resolution of models and env overrides."""
    # Mock settings override
    from common.config.settings import settings
    mocker.patch.object(settings, "OCR_MODEL", "gemini")

    # Mock database specs list
    from common.models.database import ModelRegistryModel
    mock_specs = [
        ModelRegistryModel(
            id=1,
            role="ocr",
            mode="local",
            provider="huggingface",
            model_id="THUDM/GLM-OCR",
            display_name="GLM-OCR",
            framework="transformers",
            is_default=True,
            is_enabled=True,
            priority=0,
        ),
        ModelRegistryModel(
            id=2,
            role="ocr",
            mode="cloud",
            provider="gemini",
            model_id="gemini/gemini-3.5-flash",
            display_name="Gemini 3.5 Flash",
            framework="litellm",
            is_default=False,
            is_enabled=True,
            priority=1,
        )
    ]

    # Mock DB execute query returning our mock specs
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_specs
    
    mock_db = AsyncMock()
    mock_db.execute.return_value = MagicMock(scalars=lambda: mock_scalars)

    from common.models.registry import get_model_spec
    
    spec = await get_model_spec("ocr", "auto", db=mock_db)
    
    # It should resolve to Gemini 3.5 Flash since OCR_MODEL is overridden to 'gemini'
    assert spec.provider == "gemini"
    assert spec.model_id == "gemini/gemini-3.5-flash"


def test_truncate_messages():
    """Verify context truncation logic."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! " * 100},
        {"role": "assistant", "content": "Hi there! " * 100},
        {"role": "user", "content": "How are you? " * 100},
    ]

    # Under 8000 tokens limit, it shouldn't truncate
    res = litellm_truncate(messages, "gemini/gemini-3.5-flash", 8000)
    assert len(res) == len(messages)

    # Let's mock token_counter to simulate truncation triggers
    with patch("litellm.token_counter") as mock_counter:
        # First check triggers truncation (> 50 tokens), then next checks fit
        mock_counter.side_effect = [100, 80, 40]
        
        truncated = litellm_truncate(messages, "gemini/gemini-2.5-flash:free", 50)
        
        # System message (messages[0]) and the last message (messages[3]) must be preserved
        assert len(truncated) < len(messages)
        assert truncated[0]["role"] == "system"
        assert truncated[-1]["content"] == messages[-1]["content"]


@pytest.mark.asyncio
async def test_inference_client_circuit_breaker(mocker):
    """Verify InferenceClient circuit breaker degradation on consecutive failures."""
    client = InferenceClient(base_url="http://mock-inference:8010")
    
    # Mock httpx AsyncClient
    mock_async_client = AsyncMock()
    # Mock connection errors
    mock_async_client.post.side_effect = httpx.ConnectError("Connection refused")
    mocker.patch.object(client, "_get_client", return_value=mock_async_client)

    # Trigger failures up to max_failures (5)
    for _ in range(5):
        with pytest.raises(RuntimeError):
            await client.classify("test prompt")

    assert client._is_degraded is True

    # Next attempt should be blocked immediately by circuit breaker
    with pytest.raises(RuntimeError, match="degraded"):
        await client.classify("another prompt")


def test_logger_level_and_formatters(mocker):
    """Test standard and JSON logging formatters and setting log levels."""
    import logging
    import json
    from common.config.settings import settings
    from common.observability.logger import get_logger, request_id_var, RequestIdFormatter, JSONFormatter
    
    # Check default log level resolution
    logger = get_logger("test-logger")
    assert logger.level == getattr(logging, settings.LOG_LEVEL.upper())
    
    # Mock contextvar request_id
    token = request_id_var.set("test-1234-uuid")
    try:
        # Test RequestIdFormatter
        formatter = RequestIdFormatter(fmt="%(request_id)s | %(message)s")
        record = logging.LogRecord("test-logger", logging.INFO, "path", 10, "hello", (), None)
        formatted = formatter.format(record)
        assert "test-1234-uuid | hello" in formatted
        
        # Test JSONFormatter
        json_formatter = JSONFormatter(datefmt="%Y-%m-%d")
        formatted_json = json_formatter.format(record)
        log_data = json.loads(formatted_json)
        assert log_data["message"] == "hello"
        assert log_data["request_id"] == "test-1234-uuid"
        assert log_data["level"] == "INFO"
    finally:
        request_id_var.reset(token)


@pytest.mark.asyncio
async def test_request_id_middleware():
    """Test RequestIdMiddleware generates or propagates request IDs."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from common.observability.logger import RequestIdMiddleware, request_id_var
    
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    
    @app.get("/test-route")
    async def test_route():
        # Read request ID inside request context to verify propagation
        return {"current_request_id": request_id_var.get()}
        
    client = TestClient(app)
    
    # 1. Without header (should generate new request ID)
    response = client.get("/test-route")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    generated_id = response.headers["X-Request-ID"]
    assert response.json()["current_request_id"] == generated_id
    
    # 2. With header (should propagate provided request ID)
    custom_id = "custom-test-id-999"
    response = client.get("/test-route", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id
    assert response.json()["current_request_id"] == custom_id
