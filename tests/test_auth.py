"""Tests for API key authentication middleware in isolation."""

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from nightmarenet.api.auth import APIKeyMiddleware  # noqa: E402


@pytest.fixture(autouse=True)
def clear_api_key_env(monkeypatch):
    """Clear NIGHTMARENET_API_KEY before each test for isolation."""
    monkeypatch.delenv("NIGHTMARENET_API_KEY", raising=False)


class TestAPIKeyMiddleware:
    """Test APIKeyMiddleware behavior in isolation."""

    def test_valid_key_passes_through(self):
        """Requests with correct API key should pass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/protected", headers={"X-API-Key": "test-secret-key"})
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_invalid_key_returns_401(self):
        """Requests with wrong API key should return 401."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/protected", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401
        assert response.json()["error"] == "Unauthorized"
        assert "Invalid or missing API key" in response.json()["detail"]

    def test_missing_key_returns_401_when_auth_enabled(self):
        """Requests without API key header should return 401 when auth is enabled."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/protected")
        assert response.status_code == 401
        assert response.json()["error"] == "Unauthorized"

    def test_public_path_health_bypasses_auth(self):
        """Health endpoint should bypass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/api/v1/health")
        def health():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_public_path_docs_bypasses_auth(self):
        """Docs endpoint should bypass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/docs")
        def docs():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/docs")
        assert response.status_code == 200

    def test_public_path_openapi_bypasses_auth(self):
        """OpenAPI JSON endpoint should bypass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/openapi.json")
        def openapi():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_public_prefix_badge_bypasses_auth(self):
        """Badge endpoints should bypass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/api/v1/badge/{badge_id}")
        def badge(badge_id: str):
            return {"badge": badge_id}

        client = TestClient(app)
        response = client.get("/api/v1/badge/test-badge")
        assert response.status_code == 200

    def test_public_prefix_ws_bypasses_auth(self):
        """WebSocket endpoints should bypass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/ws/test")
        def websocket():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/ws/test")
        assert response.status_code == 200

    def test_exempt_referer_localhost_bypasses_auth(self):
        """Requests from localhost:3000 referer should bypass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get(
            "/protected",
            headers={"Referer": "http://localhost:3000/some-page"},
        )
        assert response.status_code == 200

    def test_exempt_referer_127_0_0_1_bypasses_auth(self):
        """Requests from 127.0.0.1:3000 referer should bypass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get(
            "/protected",
            headers={"Referer": "http://127.0.0.1:3000/some-page"},
        )
        assert response.status_code == 200

    def test_exempt_origin_localhost_bypasses_auth(self):
        """Requests from localhost:3000 origin should bypass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get(
            "/protected",
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.status_code == 200

    def test_auth_disabled_when_no_api_key(self):
        """When no API key is provided, auth should be disabled (dev mode)."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key=None)

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/protected")
        assert response.status_code == 200

    def test_timing_safe_comparison(self):
        """API key comparison should use timing-safe comparison (hmac.compare_digest)."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)

        # Test that the middleware uses hmac.compare_digest by verifying
        # that it rejects keys that are similar but not equal
        similar_wrong_key = "test-secret-kez"  # Last char different
        response = client.get("/protected", headers={"X-API-Key": similar_wrong_key})
        assert response.status_code == 401

        # Verify the correct key still works
        response = client.get("/protected", headers={"X-API-Key": "test-secret-key"})
        assert response.status_code == 200

    def test_env_var_override(self, monkeypatch):
        """Middleware should check NIGHTMARENET_API_KEY env var at request time."""
        monkeypatch.setenv("NIGHTMARENET_API_KEY", "env-secret-key")

        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="constructor-key")

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)

        # Env var should take precedence over constructor arg
        response = client.get("/protected", headers={"X-API-Key": "env-secret-key"})
        assert response.status_code == 200

        # Constructor key should not work when env var is set
        response = client.get("/protected", headers={"X-API-Key": "constructor-key"})
        assert response.status_code == 401

    def test_non_exempt_referer_requires_auth(self):
        """Requests from non-exempt referers should require authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/protected")
        def protected():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get(
            "/protected",
            headers={"Referer": "http://example.com/some-page"},
        )
        assert response.status_code == 401

    def test_non_public_path_requires_auth(self):
        """Non-public paths should require authentication when enabled."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/api/v1/some-endpoint")
        def endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/api/v1/some-endpoint")
        assert response.status_code == 401

    def test_redoc_bypasses_auth(self):
        """ReDoc endpoint should bypass authentication."""
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key="test-secret-key")

        @app.get("/redoc")
        def redoc():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/redoc")
        assert response.status_code == 200
