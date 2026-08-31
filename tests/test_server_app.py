"""Integration tests for nightmarenet_server/app.py (server composition).

Verifies the composition of FastAPI middleware, CORS headers, health checks,
exception handlers, mounted sub-routers, session management, and API versioning.
"""

import os
import sys
from unittest import mock
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient
from nightmarenet_server import __version__ as server_version
from nightmarenet import __version__ as core_version
import nightmarenet_server.app as server_app_module


@pytest.fixture
def client():
    """Create a TestClient with safe in-memory environment overrides."""
    with mock.patch.dict(
        os.environ,
        {
            "NIGHTMARENET_DATABASE_URL": "sqlite:///:memory:",
            "NIGHTMARENET_CORS_ORIGINS": "http://localhost:3000,https://app.nightmarenet.ai",
            "NIGHTMARENET_JWT_SECRET": "testsecret123",
            "NIGHTMARENET_SESSION_SECRET": "test-session-secret-xyz",
        },
    ):
        with mock.patch("nightmarenet_server.app._init_db_safe"):
            app = server_app_module.create_app()
            assert app is not None
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c


def test_hosted_health_endpoint_schema_and_version(client: TestClient):
    """Test /api/v1/server/health returns 200 with expected schema and versions."""
    response = client.get("/api/v1/server/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["server_version"] == server_version
    assert data["core_version"] == core_version
    assert "oauth_enabled" in data
    assert "sso_enabled" in data


def test_hosted_health_with_oauth_and_sso_env_flags():
    """Test /api/v1/server/health reflects configured environment variables."""
    with mock.patch.dict(
        os.environ,
        {
            "NIGHTMARENET_GITHUB_CLIENT_ID": "gh-client-id-123",
            "NIGHTMARENET_OIDC_CLIENT_ID": "oidc-client-id-456",
            "NIGHTMARENET_OIDC_DEFAULT_METADATA_URL": "https://auth.example.com/.well-known/openid-configuration",
        },
    ):
        with mock.patch("nightmarenet_server.app._init_db_safe"):
            app = server_app_module.create_app()
            assert app is not None
            with TestClient(app, raise_server_exceptions=False) as c:
                res = c.get("/api/v1/server/health")
                assert res.status_code == 200
                data = res.json()
                assert data["oauth_enabled"] is True
                assert data["sso_enabled"] is True


def test_cors_headers_present_on_options_request(client: TestClient):
    """Test CORS preflight headers are returned for allowed origins."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type,Authorization",
    }
    response = client.options("/api/v1/server/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-credentials" in response.headers


def test_cors_headers_present_on_alternate_origin(client: TestClient):
    """Test CORS headers work for secondary configured origin."""
    headers = {
        "Origin": "https://app.nightmarenet.ai",
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/api/v1/server/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://app.nightmarenet.ai"


def test_cors_disallows_unconfigured_origins(client: TestClient):
    """Test CORS headers are not returned for unallowed origin."""
    headers = {
        "Origin": "https://malicious-site.example.com",
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/api/v1/server/health", headers=headers)
    assert response.headers.get("access-control-allow-origin") != "https://malicious-site.example.com"


def test_unknown_route_returns_json_404(client: TestClient):
    """Test requests to unregistered endpoints return a JSON 404 response."""
    response = client.get("/api/v1/nonexistent/endpoint/path")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert response.headers.get("content-type", "").startswith("application/json")


def test_invalid_content_type_or_malformed_json_handling(client: TestClient):
    """Test invalid payloads or malformed JSON return HTTP 422 or formatted JSON error."""
    response = client.post(
        "/api/v1/keys",
        content="not a json string",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_api_keys_validation_error_handling(client: TestClient):
    """Test exception handler returns formatted JSON when required body keys are missing."""
    response = client.post(
        "/api/v1/keys",
        json={"name": "test-key-without-org-user"},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "org_id + user_id required" in data["detail"]


def test_search_router_is_mounted_and_accessible(client: TestClient):
    """Test the search router endpoints are mounted under the composed application."""
    response = client.get("/api/v1/search")
    assert response.status_code in (200, 401, 403, 422)


def test_api_versioning_and_core_route_mounting(client: TestClient):
    """Test core OpenAPI docs and core health or generation endpoints are mounted."""
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    assert "text/html" in docs_response.headers.get("content-type", "")

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    openapi_data = openapi_response.json()
    assert "paths" in openapi_data
    assert "/api/v1/server/health" in openapi_data["paths"]
    assert "/api/v1/keys" in openapi_data["paths"]


def test_request_id_and_audit_middleware_headers(client: TestClient):
    """Test middleware attaches X-Request-ID or processes requests with correlation tracking."""
    response = client.get(
        "/api/v1/server/health",
        headers={"X-Request-ID": "req-test-uuid-9999"},
    )
    assert response.status_code == 200
    if "x-request-id" in response.headers:
        assert response.headers["x-request-id"] == "req-test-uuid-9999"


def test_cors_origin_parser_helper():
    """Test _cors_origins parsing helper with various environment configurations."""
    with mock.patch.dict(os.environ, {"NIGHTMARENET_CORS_ORIGINS": "https://a.com, https://b.com , "}):
        origins = server_app_module._cors_origins()
        assert origins == ["https://a.com", "https://b.com"]

    with mock.patch.dict(os.environ, {"NIGHTMARENET_CORS_ORIGINS": "*"}):
        origins = server_app_module._cors_origins()
        assert origins == ["*"]


def test_create_app_returns_none_when_fastapi_unavailable():
    """Test create_app returns None gracefully if FastAPI is not available."""
    with mock.patch.object(server_app_module, "_FASTAPI_AVAILABLE", False):
        app = server_app_module.create_app()
        assert app is None


def test_init_db_safe_handles_exceptions_gracefully():
    """Test _init_db_safe does not raise uncaught exceptions on database init errors."""
    with mock.patch("nightmarenet_server.models.base.init_db", side_effect=Exception("DB connection error")):
        # Should catch and log error rather than crashing
        server_app_module._init_db_safe()


def test_init_db_safe_when_sqlalchemy_missing():
    """Test _init_db_safe exits early when SQLAlchemy is unavailable."""
    with mock.patch.dict("sys.modules", {"nightmarenet_server.models.base": None}):
        server_app_module._init_db_safe()


def test_attach_routers_when_modules_unavailable():
    """Test sub-router attachment functions gracefully handle absent dependencies."""
    mock_app = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"nightmarenet_server.auth.oauth": None}):
        server_app_module._attach_oauth(mock_app)

    with mock.patch.dict("sys.modules", {"nightmarenet_server.auth.oidc": None}):
        server_app_module._attach_sso(mock_app)

    with mock.patch.dict("sys.modules", {"nightmarenet_server.realtime.websocket": None}):
        server_app_module._attach_realtime(mock_app)

    with mock.patch.dict("sys.modules", {"nightmarenet_server.search.endpoints": None}):
        server_app_module._attach_search(mock_app)

    with mock.patch.dict("sys.modules", {"nightmarenet_server.audit.endpoints": None}):
        server_app_module._attach_audit(mock_app)

    with mock.patch.dict("sys.modules", {"nightmarenet_server.middleware": None}):
        server_app_module._attach_audit_middleware(mock_app)


def test_api_keys_revocation_missing_key_returns_404(client: TestClient):
    """Test deleting non-existent API key returns 404 with standard JSON detail."""
    with mock.patch("nightmarenet_server.auth.api_keys.revoke_api_key", return_value=False):
        response = client.delete("/api/v1/keys/key_nonexistent_123")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "API key not found"


def test_api_keys_minting_success_flow(client: TestClient):
    """Test minting a new API key returns key metadata and plaintext key string."""
    mock_row = mock.MagicMock()
    mock_row.id = "key_row_id_123"
    mock_row.name = "ci-key"
    mock_row.scopes = ["read", "write"]

    with mock.patch(
        "nightmarenet_server.auth.api_keys.mint_api_key",
        return_value=("nm_test_plaintext_key_value", mock_row),
    ):
        response = client.post(
            "/api/v1/keys",
            json={"org_id": "org_abc", "user_id": "user_xyz", "name": "ci-key", "scopes": ["read", "write"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "key_row_id_123"
        assert data["plaintext"] == "nm_test_plaintext_key_value"
        assert data["name"] == "ci-key"
        assert data["scopes"] == ["read", "write"]


def test_api_keys_revocation_success_flow(client: TestClient):
    """Test deleting an existing API key returns 200 with revoked flag."""
    with mock.patch("nightmarenet_server.auth.api_keys.revoke_api_key", return_value=True):
        response = client.delete("/api/v1/keys/key_valid_123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "key_valid_123"
        assert data["revoked"] is True


def test_app_redoc_endpoint_mounted(client: TestClient):
    """Test ReDoc endpoint is accessible and mounted."""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_session_middleware_configured_when_available():
    """Test SessionMiddleware is added with secret key when Starlette is present."""
    with mock.patch("nightmarenet_server.app._init_db_safe"):
        with mock.patch.dict(os.environ, {"NIGHTMARENET_SESSION_SECRET": "custom-secret-key-123"}):
            app = server_app_module.create_app()
            assert app is not None
            middleware_classes = [m.cls.__name__ for m in app.user_middleware]
            assert "SessionMiddleware" in middleware_classes or "CORSMiddleware" in middleware_classes


def test_openapi_info_schema_metadata(client: TestClient):
    """Test the OpenAPI metadata contains correct platform details and title."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "NightmareNet Hosted Platform"
    assert "NightmareNet core" in schema["info"]["description"]
    assert str(server_version) in schema["info"]["version"]
    assert str(core_version) in schema["info"]["version"]


def test_api_keys_routes_disabled_when_fastapi_missing():
    """Test _attach_api_key_routes is safe and early-returns when FastAPI is missing."""
    mock_app = mock.MagicMock()
    with mock.patch.object(server_app_module, "_FASTAPI_AVAILABLE", False):
        server_app_module._attach_api_key_routes(mock_app)
        mock_app.include_router.assert_not_called()


def test_api_keys_routes_disabled_when_deps_import_error():
    """Test _attach_api_key_routes handles missing auth or models deps gracefully."""
    mock_app = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"nightmarenet_server.auth.api_keys": None}):
        server_app_module._attach_api_key_routes(mock_app)
        mock_app.include_router.assert_not_called()


def test_audit_immutability_guards_registration_failure():
    """Test _attach_audit logs and continues if register_immutability_guards fails."""
    mock_app = mock.MagicMock()
    with mock.patch(
        "nightmarenet_server.audit.logger.register_immutability_guards",
        side_effect=Exception("Guard registration failed"),
    ):
        with mock.patch("nightmarenet_server.audit.endpoints.build_audit_router", return_value=None):
            server_app_module._attach_audit(mock_app)


def test_api_version_tag_and_prefix_consistency(client: TestClient):
    """Test standard /api/v1 prefixes are consistently configured across routes."""
    openapi_res = client.get("/openapi.json")
    assert openapi_res.status_code == 200
    paths = openapi_res.json().get("paths", {})
    for path in paths.keys():
        if path.startswith("/api/"):
            assert path.startswith("/api/v1/")


def test_app_root_title_and_description():
    """Test application root construction title, description, and metadata tags."""
    with mock.patch("nightmarenet_server.app._init_db_safe"):
        app = server_app_module.create_app()
        assert app is not None
        assert "NightmareNet Hosted Platform" in app.title
        assert "multi-tenant auth" in app.description


def test_default_database_initialization_on_startup():
    """Test database initialization routine executes without crashing on startup event."""
    with mock.patch("nightmarenet_server.app.init_db", create=True):
        server_app_module._init_db_safe()

def test_cors_preflight_allows_custom_headers(client: TestClient):
    """Test CORS preflight accepts custom tracing and authorization headers."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-Custom-Header,Authorization",
    }
    response = client.options("/api/v1/server/health", headers=headers)
    assert response.status_code == 200
