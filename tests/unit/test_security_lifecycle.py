"""Unit tests for newly implemented security audit logging, database validation, and startup sequence logic.
"""

import pytest
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from slowapi.errors import RateLimitExceeded

from common.config.settings import Settings, settings
from common.observability.logger import RequestAuditMiddleware, log_security_event
from projects.guardroute.src.agents.coding import run_code_sandbox
from gateway.api import verify_api_key
from gateway.main import custom_rate_limit_exceeded_handler


@pytest.mark.asyncio
async def test_settings_database_url_validation():
    """Ensure validate_settings raises ValueError on invalid DATABASE_URL scheme."""
    with pytest.raises(ValueError, match="DATABASE_URL must start with 'postgresql\\+asyncpg://'"):
        Settings(
            DATABASE_URL="postgresql://user:pass@localhost:5432/db",
            APP_ENV="development"
        )


@pytest.mark.asyncio
async def test_settings_qdrant_api_key_validation_in_production():
    """Ensure validate_settings raises ValueError on missing QDRANT_API_KEY in production."""
    with pytest.raises(ValueError, match="QDRANT_API_KEY is required in production environment."):
        Settings(
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/db",
            APP_ENV="production",
            QDRANT_API_KEY=None
        )


@pytest.mark.asyncio
async def test_verify_api_key_logging(mocker):
    """Ensure verify_api_key calls log_security_event on auth failures."""
    # Mock settings.AUTH_ENABLED to True to ensure logic runs
    mocker.patch.object(settings, "AUTH_ENABLED", True)
    
    mock_log_security = mocker.patch("gateway.api.log_security_event")
    
    # 1. Missing header
    with pytest.raises(HTTPException) as excinfo:
        await verify_api_key(x_api_key=None)
    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
    mock_log_security.assert_any_call("AUTH_FAILURE", {"reason": "Missing X-API-Key header"})

    # 2. Invalid API key (DB mismatch)
    mock_db = AsyncMock()
    mock_db.__aenter__.return_value = mock_db
    mock_db.execute.return_value = MagicMock(scalar_one_or_none=lambda: None)
    
    mocker.patch("gateway.api.get_sessionmaker", return_value=lambda: mock_db)
    
    with pytest.raises(HTTPException) as excinfo:
        await verify_api_key(x_api_key="sk_live_badkey")
    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
    mock_log_security.assert_any_call(
        "AUTH_FAILURE", 
        {"reason": "Invalid or inactive X-API-Key", "provided_key": "sk_live_badkey"}
    )


@pytest.mark.asyncio
async def test_rate_limit_logging(mocker):
    """Ensure custom_rate_limit_exceeded_handler logs rate limit violations."""
    mock_log_security = mocker.patch("gateway.main.log_security_event")
    
    request = MagicMock(spec=Request)
    request.url.path = "/api/test"
    request.method = "GET"
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    
    mock_limit = MagicMock()
    mock_limit.error_message = None
    mock_limit.__str__.return_value = "10/minute"
    exc = RateLimitExceeded(mock_limit)
    exc.detail = "10/minute"
    
    # Mock default handler
    mocker.patch("gateway.main._rate_limit_exceeded_handler")
    
    await custom_rate_limit_exceeded_handler(request, exc)
    
    mock_log_security.assert_called_once_with(
        "RATE_LIMIT_VIOLATION",
        {
            "client_ip": "127.0.0.1",
            "path": "/api/test",
            "method": "GET",
            "limit": "10/minute"
        }
    )


@pytest.mark.asyncio
async def test_sandbox_execution_logging(mocker):
    """Ensure run_code_sandbox logs execution attempts and outcomes."""
    mock_log_security = mocker.patch("common.observability.logger.log_security_event")
    
    # Run simple code sandbox execution
    await run_code_sandbox("x = 5\nprint(x)", timeout=1.0)
    
    # Should log ATTEMPT and RESULT (SUCCESS)
    mock_log_security.assert_any_call("SANDBOX_EXECUTION_ATTEMPT", {"code_length": 14, "timeout": 1.0})
    
    # Extract the call args for the RESULT log
    result_call = None
    for call in mock_log_security.call_args_list:
        if call[0][0] == "SANDBOX_EXECUTION_RESULT":
            result_call = call
            break
            
    assert result_call is not None
    assert result_call[0][1]["status"] == "SUCCESS"
    assert result_call[0][1]["latency_ms"] > 0.0


@pytest.mark.asyncio
async def test_request_audit_middleware_logging(mocker):
    """Ensure RequestAuditMiddleware logs request metadata and latency."""
    # Mock loggers
    mock_audit_logger = MagicMock()
    mocker.patch("common.observability.logger.get_logger", return_value=mock_audit_logger)
    
    app = FastAPI()
    app.add_middleware(RequestAuditMiddleware)
    
    @app.get("/test-audit")
    def test_route():
        return {"status": "ok"}
        
    client = TestClient(app)
    client.get("/test-audit")
    
    # Verify audit logger was called
    assert mock_audit_logger.info.called or mock_audit_logger.warning.called or mock_audit_logger.error.called
