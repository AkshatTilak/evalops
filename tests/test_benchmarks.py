import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from projects.evalops.scripts.bench_gguf import run_benchmark as run_gguf_bench
from projects.evalops.scripts.bench_mmlu import run_benchmark as run_mmlu_bench
from projects.evalops.scripts.bench_models import run_comparative_benchmarks

@pytest.mark.asyncio
async def test_bench_gguf(mocker):
    """Verify that bench_gguf script runs and writes results to DB."""
    mocker.patch("common.clients.inference.InferenceClient.classify", new_callable=AsyncMock, return_value={
        "complexity": "simple",
        "required_agents": []
    })
    mocker.patch("common.clients.inference.InferenceClient.close", new_callable=AsyncMock)
    
    mock_db = AsyncMock()
    mock_db.__aenter__.return_value = mock_db
    mocker.patch("projects.evalops.scripts.bench_gguf.get_sessionmaker", return_value=lambda: mock_db)
    
    await run_gguf_bench()
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()

@pytest.mark.asyncio
async def test_bench_mmlu(mocker):
    """Verify that bench_mmlu script runs and logs results to DB."""
    mocker.patch("projects.evalops.scripts.bench_mmlu.execute_orchestrator", new_callable=AsyncMock, return_value={
        "final_response": "The solution is 3."
    })
    mock_db = AsyncMock()
    mock_db.__aenter__.return_value = mock_db
    mocker.patch("projects.evalops.scripts.bench_mmlu.get_sessionmaker", return_value=lambda: mock_db)
    
    await run_mmlu_bench()
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()

@pytest.mark.asyncio
async def test_bench_models(mocker):
    """Verify that bench_models script resolves specifications and saves reports."""
    mock_db = AsyncMock()
    mock_db.__aenter__.return_value = mock_db
    mocker.patch("projects.evalops.scripts.bench_models.get_sessionmaker", return_value=lambda: mock_db)
    mocker.patch("projects.evalops.scripts.bench_models.init_model_registry", new_callable=AsyncMock)
    
    from common.models.database import ModelRegistryModel
    mock_models = [
        ModelRegistryModel(
            role="classifier",
            model_id="Arch-Router-1.5B",
            display_name="Arch-Router-1.5B GGUF",
            provider="llama-cpp",
            mode="local",
            vram_mb=2000,
            is_enabled=True,
            priority=0
        )
    ]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_models
    mock_db.execute.return_value = mock_result
    
    await run_comparative_benchmarks()
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
