"""Unit tests for S5-01d: Enhanced Eval Run Orchestration."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from projects.evalops.api import router as evalops_router, EvalRunRequest


app = FastAPI()
app.include_router(evalops_router, prefix="/api/evalops")

client = TestClient(app)


def test_eval_run_request_schema():
    """Verify EvalRunRequest model accepts framework, metrics, and thresholds."""
    req = EvalRunRequest(
        agent_id="agent-123",
        framework="deepeval",
        metrics=["hallucination", "faithfulness"],
        thresholds={"faithfulness": 0.8},
    )
    assert req.agent_id == "agent-123"
    assert req.framework == "deepeval"
    assert req.metrics == ["hallucination", "faithfulness"]
    assert req.thresholds == {"faithfulness": 0.8}


from common.clients.postgres import get_async_db

@pytest.mark.asyncio
@patch("projects.evalops.api.publish_eval_trigger_event", return_value=True)
async def test_trigger_eval_run_endpoint(mock_publish):
    """Verify POST /api/evalops/run initiates run and passes framework params."""
    mock_db = AsyncMock()
    app.dependency_overrides[get_async_db] = lambda: mock_db

    response = client.post(
        "/api/evalops/run",
        json={
            "agent_id": "agent-xyz",
            "framework": "ragas",
            "metrics": ["faithfulness", "context_recall"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "initiated"
    assert data["framework"] == "ragas"
    assert data["agent_id"] == "agent-xyz"

    app.dependency_overrides.clear()
