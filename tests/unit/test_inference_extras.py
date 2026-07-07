"""Unit tests for the new inference mock loaders, VRAMManager enhancements, and schemas."""

import asyncio
import pytest
from unittest.mock import AsyncMock

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
