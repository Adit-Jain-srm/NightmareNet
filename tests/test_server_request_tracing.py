import logging
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nightmarenet.utils.logging_config import RequestIdFilter, request_id_ctx
from nightmarenet_server.middleware.request_tracing import RequestTracingMiddleware


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(RequestTracingMiddleware)

    @app.get("/test")
    def test_route():
        # Validate context var is set during request
        req_id = request_id_ctx.get()
        assert req_id != ""
        return {"id": req_id}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_middleware_generates_request_id(client):
    response = client.get("/test")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    body = response.json()
    assert body["id"] == response.headers["X-Request-ID"]


def test_middleware_accepts_incoming_request_id(client):
    test_id = "test-correlation-id-123"
    response = client.get("/test", headers={"X-Request-ID": test_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == test_id

    body = response.json()
    assert body["id"] == test_id


def test_request_id_filter():
    filter_ = RequestIdFilter()
    record = logging.LogRecord("name", logging.INFO, "pathname", 1, "msg", (), None)

    token = request_id_ctx.set("test-filter-id")
    try:
        filter_.filter(record)
        assert record.request_id == "test-filter-id"
    finally:
        request_id_ctx.reset(token)


@mock.patch("nightmarenet_server.middleware.request_tracing.logger.info")
def test_middleware_logs_completion(mock_logger_info, client):
    test_id = "log-test-id"
    response = client.get("/test", headers={"X-Request-ID": test_id})
    assert response.status_code == 200

    mock_logger_info.assert_called_once()
    args, kwargs = mock_logger_info.call_args
    assert "Request completed:" in args[0]

    extra = kwargs.get("extra", {})
    assert extra["method"] == "GET"
    assert extra["path"] == "/test"
    assert extra["status"] == 200
    assert extra["request_id"] == test_id
    assert "duration_ms" in extra


def test_middleware_performance(client):
    import time

    start = time.perf_counter()
    for _ in range(100):
        client.get("/test")
    duration = time.perf_counter() - start

    avg_ms = (duration / 100) * 1000
    # Allow some leeway for TestClient overhead, but it should be fast
    assert avg_ms < 5.0, f"Middleware overhead too high: {avg_ms:.2f}ms per request"
