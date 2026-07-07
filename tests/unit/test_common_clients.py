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
