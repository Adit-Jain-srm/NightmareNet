import asyncio
from unittest import mock

import pytest

try:
    from fastapi.testclient import TestClient

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from nightmarenet_server.app import (
    create_app,
    is_shutting_down,
    reset_shutdown_state,
    set_shutting_down,
    trigger_graceful_shutdown,
)
from nightmarenet_server.realtime.websocket import RunBroker


@pytest.fixture(autouse=True)
def _cleanup_shutdown_state():
    reset_shutdown_state()
    yield
    reset_shutdown_state()


def test_shutdown_state_flag():
    assert not is_shutting_down()
    set_shutting_down(True)
    assert is_shutting_down()
    reset_shutdown_state()
    assert not is_shutting_down()


def test_trigger_graceful_shutdown_sequence():
    async def _test():
        assert not is_shutting_down()
        with mock.patch(
            "nightmarenet_server.realtime.websocket.close_all_websockets",
            new_callable=mock.AsyncMock,
        ) as mock_close_ws:
            await trigger_graceful_shutdown(grace_period=0.1)

        assert is_shutting_down()
        mock_close_ws.assert_awaited_once_with(code=1001, reason="Server shutting down")

    asyncio.run(_test())


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_health_endpoint_503_during_shutdown():
    app = create_app()
    assert app is not None

    with TestClient(app) as client:
        # Before shutdown: 200 OK
        res_before = client.get("/api/v1/server/health")
        assert res_before.status_code == 200
        assert res_before.json().get("status") == "ok"

        # Set shutdown flag
        set_shutting_down(True)

        # During shutdown: 503 Service Unavailable
        res_after = client.get("/api/v1/server/health")
        assert res_after.status_code == 503
        assert res_after.json() == {"status": "shutting_down"}

        # Reset flag before exiting TestClient context
        reset_shutdown_state()


def test_websocket_close_all():
    async def _test():
        broker = RunBroker()
        mock_ws = mock.AsyncMock()

        broker.subscribe("run_test_123", websocket=mock_ws)

        await broker.close_all(code=1001, reason="Server shutting down")

        mock_ws.close.assert_awaited_once_with(code=1001, reason="Server shutting down")

    asyncio.run(_test())
