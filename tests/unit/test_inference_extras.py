"""Unit tests for the new inference mock loaders, VRAMManager enhancements, and schemas."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from common.schemas import HealthResponse, ErrorResponse, PaginatedResponse, SubAgentResult
from inference.core.vram_manager import VRAMManager
from inference.models.baidu_ocr import load_baidu_ocr
from inference.models.jina_clip import load_jina_clip
from inference.models.sensevoice import load_sensevoice
from inference.models.classifier import load_classifier


@pytest.mark.asyncio
async def test_mock_loaders_return_contracts():
    """Verify that all mock loaders return the correct schemas/structures."""
    # 1. Baidu OCR
    ocr_model = await load_baidu_ocr()
    ocr_res = await ocr_model.extract_layout(b"fake_image_bytes")
    assert "text" in ocr_res
    assert "blocks" in ocr_res
    assert "tables" in ocr_res
    assert "layout" in ocr_res
    assert len(ocr_res["blocks"]) > 0

    # 2. Jina Clip
    clip_model = await load_jina_clip()
    txt_embeds = await clip_model.embed_texts(["hello", "world"])
    img_embeds = await clip_model.embed_images([b"fake_image"])
    assert len(txt_embeds) == 2
    assert len(txt_embeds[0]) == 1024
    assert len(img_embeds) == 1
    assert len(img_embeds[0]) == 1024

    # 3. SenseVoice
    sv_model = await load_sensevoice()
    asr_res = await sv_model.transcribe_audio(b"fake_audio")
    assert asr_res["text"] == "This is a mock transcription of the audio content."
    assert asr_res["emotion"] == "neutral"
    assert asr_res["audio_events"] == ["laughter"]

    # 4. Classifier
    cls_model = await load_classifier()
    cls_res = await cls_model.classify_prompt("Write a function to add two numbers")
    assert cls_res["complexity"] == "complex"
    assert "coding" in cls_res["required_agents"]


@pytest.mark.asyncio
async def test_vram_manager_latency_metrics():
    """Verify that VRAMManager records cold start and warm latency stats correctly."""
    # Create fresh instance to avoid singleton conflicts in tests
    manager = VRAMManager(budget_mb=5000, idle_timeout=100)
    
    async def slow_inference(self=None):
        await asyncio.sleep(0.02)
        return "done"

    async def mock_loader():
        await asyncio.sleep(0.01)
        return type("Mock", (), {"test_call": slow_inference})()
        
    manager.register_loader("test-model", mock_loader, vram_mb=100, max_concurrency=2)
    
    # 1. First load (Cold start)
    wrapped = await manager.ensure_loaded("test-model")
    summary = manager.get_latency_summary()
    assert "test-model" in summary
    assert summary["test-model"]["cold_start"]["count"] == 1
    assert summary["test-model"]["cold_start"]["avg_s"] > 0.0
    
    # 2. Warm calls (Inference calls)
    res = await wrapped.test_call()
    assert res == "done"
    
    summary = manager.get_latency_summary()
    assert summary["test-model"]["warm_inference"]["count"] == 1
    assert summary["test-model"]["warm_inference"]["avg_s"] > 0.0


@pytest.mark.asyncio
async def test_vram_manager_concurrency_limits():
    """Verify that VRAMManager limits concurrency per model using Semaphore."""
    manager = VRAMManager(budget_mb=5000, idle_timeout=100)
    
    # We will track active concurrent calls
    active_calls = 0
    max_active_calls = 0
    
    async def slow_call(self=None):
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        await asyncio.sleep(0.1)
        active_calls -= 1
        return "ok"
        
    async def mock_loader():
        return type("Mock", (), {"run": slow_call})()
        
    # Register with max concurrency of 2
    manager.register_loader("concurrent-model", mock_loader, vram_mb=100, max_concurrency=2)
    wrapped = await manager.ensure_loaded("concurrent-model")
    
    # Invoke 4 concurrent tasks
    await asyncio.gather(
        wrapped.run(),
        wrapped.run(),
        wrapped.run(),
        wrapped.run()
    )
    
    # Concurrency limit is 2, so max active concurrent calls should not exceed 2
    assert max_active_calls <= 2


def test_schema_modifications():
    """Verify that new and modified schemas are valid Pydantic models."""
    # 1. SubAgentResult modification
    res = SubAgentResult(
        source="coding",
        status="success",
        content="print('hello')",
        latency_ms=150.5,
        model_used="gemini-3.5-flash"
    )
    assert res.latency_ms == 150.5
    assert res.model_used == "gemini-3.5-flash"

    # 2. HealthResponse
    hr = HealthResponse(status="healthy", details={"load": 0.5})
    assert hr.status == "healthy"

    # 3. ErrorResponse
    er = ErrorResponse(error_code="NOT_FOUND", message="Resource not found")
    assert er.error_code == "NOT_FOUND"

    # 4. PaginatedResponse
    pr = PaginatedResponse[str](
        items=["a", "b"],
        total=10,
        page=1,
        size=2,
        pages=5
    )
    assert len(pr.items) == 2
    assert pr.pages == 5


@pytest.mark.asyncio
async def test_downloader_offline_and_cache(tmp_path, monkeypatch):
    """Verify downloader behavior under offline mode and cache directory configurations."""
    from inference.core.downloader import download_model_from_hub

    # Create dummy local cache path
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("MODEL_CACHE_DIR", str(cache_dir))

    # 1. Local directory check: should return immediately if path exists
    local_dir = tmp_path / "my_local_model"
    local_dir.mkdir()
    result = download_model_from_hub(str(local_dir))
    assert result == str(local_dir)

    # 2. Offline check when model is not cached
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="Offline mode.*is active but model.*is not cached"):
        download_model_from_hub("some-org/non-existent-model")

    # 3. Offline check when model IS cached
    cached_model_name = "some-org/my-model"
    repo_folder = "models--" + cached_model_name.replace("/", "--")
    cached_path = cache_dir / repo_folder
    cached_path.mkdir()

    result_cached = download_model_from_hub(cached_model_name)
    assert result_cached == str(cached_path)


@pytest.mark.asyncio
async def test_loader_device_mapping_and_quantization(monkeypatch):
    """Verify that loaders handle device auto-detection, CUDA forcing, and quantization compatibility."""
    from inference.models.classifier import load_classifier
    from inference.models.sensevoice import load_sensevoice
    from common.config.settings import settings

    # Force CPU mode
    monkeypatch.setattr(settings, "DEVICE", "cpu")
    cls_model = await load_classifier()
    assert cls_model.device == "cpu"

    # Force CUDA - should fail if CUDA not available in test env
    monkeypatch.setattr(settings, "DEVICE", "cuda")
    import torch
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA is forced.*but not available"):
            await load_classifier()

    # Reset to CPU before running other validations
    monkeypatch.setattr(settings, "DEVICE", "cpu")

    # Verify incompatible quantization check (mocking get_active_model to return a spec with bad quantization)
    from common.schemas.model_registry import ModelSpec
    mock_spec = ModelSpec(
        id=1,
        role="asr",
        mode="local",
        provider="funasr",
        model_id="FunAudioLLM/SenseVoiceSmall",
        display_name="SenseVoice",
        quantization="unsupported_quant_format",
        framework="funasr"
    )

    with patch("inference.models.sensevoice.get_active_model", return_value=mock_spec):
        with pytest.raises(ValueError, match="Incompatible quantization level"):
            await load_sensevoice()

