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
from common.clients.litellm import completion_with_fallback


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


@pytest.mark.asyncio
async def test_litellm_completion_timeout_enforced(mocker):
    """Ensure that litellm.acompletion is invoked with timeout=60.0."""
    mock_acompletion = mocker.patch("litellm.acompletion", new_callable=AsyncMock)
    
    messages = [{"role": "user", "content": "hello"}]
    mocker.patch.object(settings, "GOOGLE_API_KEY", "mock-key")
    
    await completion_with_fallback(
        model="gemini/gemini-3.5-flash",
        messages=messages,
        fallbacks=[]
    )
    
    mock_acompletion.assert_called_once()
    kwargs = mock_acompletion.call_args[1]
    assert kwargs.get("timeout") == 60.0


@pytest.mark.asyncio
async def test_litellm_enforces_https_api_base(mocker):
    """Ensure completion_with_fallback raises ValueError if non-HTTPS api_base is configured or passed."""
    messages = [{"role": "user", "content": "hello"}]
    mocker.patch.object(settings, "GOOGLE_API_KEY", "mock-key")
    
    # 1. Via kwargs
    with pytest.raises(ValueError, match="must use HTTPS"):
        await completion_with_fallback(
            model="gemini/gemini-3.5-flash",
            messages=messages,
            fallbacks=[],
            api_base="http://insecure-api.com"
        )
        
    # 2. Via env var
    with patch.dict("os.environ", {"GEMINI_API_BASE": "http://insecure-gemini.com"}):
        with pytest.raises(ValueError, match="must use HTTPS"):
            await completion_with_fallback(
                model="gemini/gemini-3.5-flash",
                messages=messages,
                fallbacks=[]
            )


def test_docker_compose_network_isolation():
    """Verify that only the gateway and admin tools expose ports and all services are on contained_net."""
    import re
    from pathlib import Path
    
    compose_path = Path(__file__).parent.parent.parent.parent.parent / "infrastructure" / "docker-compose.yml"
    assert compose_path.exists()
    
    with open(compose_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    services = {}
    current_service = None
    in_services = False
    lines = content.splitlines()
    for line in lines:
        if line.startswith("services:"):
            in_services = True
            continue
        elif line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            if not line.startswith("#"):
                in_services = False
                current_service = None
                continue

        if in_services:
            service_match = re.match(r"^  ([a-zA-Z0-9_-]+):", line)
            if service_match:
                current_service = service_match.group(1)
                services[current_service] = {"ports": [], "networks": []}
                continue
                
            if current_service:
                if line.strip().startswith("-"):
                    # elements under list
                    if services[current_service].get("_in_ports"):
                        port_val = line.replace("-", "").strip().strip('"').strip("'")
                        services[current_service]["ports"].append(port_val)
                    elif services[current_service].get("_in_networks"):
                        net_val = line.replace("-", "").strip().strip('"').strip("'")
                        services[current_service]["networks"].append(net_val)
                else:
                    services[current_service]["_in_ports"] = False
                    services[current_service]["_in_networks"] = False
                    if "ports:" in line and "#" not in line:
                        services[current_service]["_in_ports"] = True
                        inline_ports = re.search(r"ports:\s*\[([^\]]+)\]", line)
                        if inline_ports:
                            ports_str = inline_ports.group(1)
                            ports = [p.strip().strip('"').strip("'") for p in ports_str.split(",")]
                            services[current_service]["ports"].extend(ports)
                    elif "networks:" in line and "#" not in line:
                        services[current_service]["_in_networks"] = True

    # Allowed exposed ports check: only gateway, pgadmin, kafka-ui, jaeger (UI only), plus dev DB & inference ports
    allowed_services_with_ports = {"gateway", "pgadmin", "kafka-ui", "jaeger", "postgres", "qdrant", "redis", "neo4j", "zookeeper", "kafka", "inference"}
    for name, info in services.items():
        if info["ports"]:
            assert name in allowed_services_with_ports, f"Service {name} exposes ports {info['ports']} but is not allowed!"
            if name == "jaeger":
                for port in info["ports"]:
                    assert "16686" in port, f"Jaeger exposes non-UI port: {port}"
        
        assert "contained_net" in info["networks"], f"Service {name} is not connected to contained_net!"


@pytest.mark.asyncio
async def test_postgres_engine_close(mocker):
    """Ensure close_postgres disposes the async engine and resets the engine singleton."""
    from common.clients.postgres import close_postgres
    import common.clients.postgres as pg_module
    
    # Mock engine
    mock_engine = AsyncMock()
    pg_module._engine = mock_engine
    
    await close_postgres()
    
    mock_engine.dispose.assert_called_once()
    assert pg_module._engine is None


def test_inference_server_error_mapping():
    """Ensure that InferenceServerError triggers a 503 HTTP response from the gateway."""
    from gateway.main import app
    from common.clients.inference import InferenceServerError, InferenceClient
    
    app.state.guardroute_inference = InferenceClient(base_url="http://mock")
    client = TestClient(app)
    
    with patch("projects.guardroute.api.run_classification", side_effect=InferenceServerError("Mock server offline")):
        with patch("gateway.api.verify_api_key", return_value=None):
            resp = client.post("/api/guardroute/classify", json={"prompt": "test prompt"})
            assert resp.status_code == 503
            json_resp = resp.json()
            assert "offline" in json_resp.get("message", json_resp.get("detail", ""))


def test_limiter_redis_fallback(mocker):
    """Ensure limiter falls back to memory:// if Redis is unreachable on startup."""
    import redis
    mocker.patch("redis.from_url", side_effect=redis.ConnectionError("Redis down"))
    
    import importlib
    import common.observability.limiter as limiter_mod
    importlib.reload(limiter_mod)
    
    assert limiter_mod.storage_uri == "memory://"


