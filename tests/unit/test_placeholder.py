"""Placeholder unit tests to verify PyTest configuration."""

from fastapi.testclient import TestClient

from common.config import get_settings
from projects.evalops.src.main import app


def test_health_endpoint() -> None:
    """Test that the application health endpoint returns a healthy status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data


def test_settings_load() -> None:
    """Test that Settings can be loaded and validated successfully."""
    settings = get_settings()
    assert settings.app_env is not None
    assert settings.database_url is not None
