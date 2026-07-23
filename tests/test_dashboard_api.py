import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.clients.postgres import get_async_db
from projects.evalops.api import router as evalops_router

app = FastAPI()
app.include_router(evalops_router, prefix="/api/evalops")

client = TestClient(app)


@pytest.mark.asyncio
async def test_get_dashboard_stats_endpoint():
    """Verify GET /api/evalops/dashboard/stats returns metric summary dict."""
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_res

    app.dependency_overrides[get_async_db] = lambda: mock_db

    response = client.get("/api/evalops/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "faithfulness" in data["metrics"]
    assert "relevance" in data["metrics"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_dashboard_trends_endpoint():
    """Verify GET /api/evalops/dashboard/trends returns time-series trends."""
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_res

    app.dependency_overrides[get_async_db] = lambda: mock_db

    response = client.get("/api/evalops/dashboard/trends?days=7")
    assert response.status_code == 200
    data = response.json()
    assert data["days"] == 7
    assert "trends" in data

    app.dependency_overrides.clear()
