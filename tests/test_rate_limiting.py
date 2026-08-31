import asyncio
from unittest import mock

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from nightmarenet_server.middleware.rate_limiting import (
    RateLimiter,
    RateLimitException,
    RateLimitingMiddleware,
    clear_rate_limits,
    enforce_rate_limit,
)


@pytest.fixture(autouse=True)
def clean_limits():
    clear_rate_limits()


@pytest.fixture
def test_app():
    app = FastAPI()

    # Register exception handler
    @app.exception_handler(RateLimitException)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": exc.detail,
            },
        )

    app.add_middleware(RateLimitingMiddleware)

    # ── Test routes matching mappings ──
    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/v1/auth/login")
    def login():
        return {"status": "logged_in"}

    @app.post("/api/v1/pipeline/create")
    def create_pipeline():
        return {"status": "created"}

    @app.get("/api/v1/pipeline/runs")
    def list_runs():
        return {"status": "runs_list"}

    # ── Explicitly decorated route ──
    @app.get("/api/v1/explicit-strict", dependencies=[Depends(RateLimiter("STRICT"))])
    def explicit_strict():
        return {"status": "explicit"}

    return app


def test_unlimited_endpoint_never_rate_limits(test_app):
    client = TestClient(test_app)

    # Make multiple requests to unlimited endpoint
    for _ in range(10):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        # Headers should be present and indicate unlimited
        assert response.headers["X-RateLimit-Limit"] == "999999"
        assert response.headers["X-RateLimit-Remaining"] == "999999"
        assert response.headers["X-RateLimit-Reset"] == "0"


def test_strict_endpoint_enforces_limit_and_returns_429(test_app):
    client = TestClient(test_app)

    # STRICT limit is 5 requests per minute
    for i in range(5):
        response = client.post("/api/v1/auth/login")
        assert response.status_code == 200
        assert response.headers["X-RateLimit-Limit"] == "5"
        assert int(response.headers["X-RateLimit-Remaining"]) == 4 - i
        assert int(response.headers["X-RateLimit-Reset"]) > 0

    # 6th request should fail with 429
    response = client.post("/api/v1/auth/login")
    assert response.status_code == 429
    assert response.headers["X-RateLimit-Limit"] == "5"
    assert response.headers["X-RateLimit-Remaining"] == "0"
    assert int(response.headers["X-RateLimit-Reset"]) > 0
    assert "Retry-After" in response.headers

    # Body matches RateLimitError schema
    data = response.json()
    assert data["error"] == "Rate limit exceeded"
    assert "Too many requests" in data["detail"]


def test_moderate_endpoint_enforces_limit(test_app):
    client = TestClient(test_app)

    # MODERATE limit is 30 requests per minute
    # Let's perform 30 requests successfully
    for i in range(30):
        response = client.post("/api/v1/pipeline/create")
        assert response.status_code == 200
        assert response.headers["X-RateLimit-Limit"] == "30"
        assert int(response.headers["X-RateLimit-Remaining"]) == 29 - i

    # 31st request should be rate limited
    response = client.post("/api/v1/pipeline/create")
    assert response.status_code == 429
    assert response.headers["X-RateLimit-Limit"] == "30"
    assert response.headers["X-RateLimit-Remaining"] == "0"


def test_generous_endpoint_enforces_limit(test_app):
    client = TestClient(test_app)

    # GENEROUS limit is 120 requests per minute
    # Let's just check the headers are injected correctly
    response = client.get("/api/v1/pipeline/runs")
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "120"
    assert response.headers["X-RateLimit-Remaining"] == "119"


def test_explicit_rate_limiter_dependency_overrides_default(test_app):
    client = TestClient(test_app)

    # /api/v1/explicit-strict has STRICT limit (5)
    # even though it is a GET request (default is GENEROUS)
    for _ in range(5):
        response = client.get("/api/v1/explicit-strict")
        assert response.status_code == 200
        assert response.headers["X-RateLimit-Limit"] == "5"

    response = client.get("/api/v1/explicit-strict")
    assert response.status_code == 429


def test_in_memory_fallback_when_redis_fails():
    # Mock redis connection failure or get_redis_client returning client that fails
    mock_redis = mock.MagicMock()
    mock_redis.pipeline.side_effect = Exception("Redis connection refused")

    target_path = "nightmarenet_server.middleware.rate_limiting.get_redis_client"
    with mock.patch(target_path, return_value=mock_redis):
        mock_req = mock.MagicMock(spec=Request)
        mock_req.headers = {"X-API-Key": "test_key"}
        mock_req.client = mock.MagicMock(host="127.0.0.1")

        # Enforce rate limit should fall back to in-memory and succeed
        exceeded, headers = asyncio.run(enforce_rate_limit(mock_req, "STRICT"))
        assert exceeded is False
        assert headers["X-RateLimit-Limit"] == "5"
        assert headers["X-RateLimit-Remaining"] == "4"
