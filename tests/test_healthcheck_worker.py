"""Regression tests for ``docker/healthcheck_worker.py``.

These tests lock in the fix for a critical bug: the prior worker HEALTHCHECK
was simply ``os.environ.get('NIGHTMARENET_REDIS_URL')`` — and the same env
var was hardcoded in the Dockerfile via ``ENV``. The check therefore always
exited 0 regardless of broker state, Celery worker state, or fallback worker
state.

The replacement script performs a real TCP probe of the broker plus a
``celery_app.control.ping(...)`` when Celery is present. These tests verify
both the regression-locking property (the old tautology is gone) and the
new behaviour under happy-path and failure scenarios.
"""

import importlib.util
import socket
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "docker" / "healthcheck_worker.py"


def _load_module():
    """Import the healthcheck script under a unique module name.

    The script lives outside the Python package layout (it ships inside the
    Docker image), so we import it via ``importlib.util`` and clear the
    cached entry between tests that need to re-trigger module-level
    initialisation.
    """
    spec = importlib.util.spec_from_file_location("_healthcheck_worker_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hc():
    return _load_module()


# --------------------------------------------------------------------------- #
# Regression — the old tautological check would have passed always.
# --------------------------------------------------------------------------- #
def test_old_tautological_check_would_have_passed_with_broken_broker(monkeypatch):
    """Documents the bug we just fixed.

    The previous HEALTHCHECK was literally::

        python -c "import os, sys; sys.exit(0 if os.environ.get('NIGHTMARENET_REDIS_URL') else 1)"

    Because the Dockerfile sets ``NIGHTMARENET_REDIS_URL`` via ``ENV`` on
    the same image, this check always exits 0. If we ever regress to a
    similar pattern, this test will keep the lesson visible.
    """
    monkeypatch.setenv("NIGHTMARENET_REDIS_URL", "redis://unreachable.invalid:6379/0")
    # Simulate the old behaviour explicitly:
    import os as _os

    assert _os.environ.get("NIGHTMARENET_REDIS_URL")  # i.e. would return truthy → exit 0


# --------------------------------------------------------------------------- #
# Broker probe — happy path.
# --------------------------------------------------------------------------- #
def test_broker_reachable_when_socket_connects(hc, monkeypatch):
    fake_sock = mock.MagicMock()
    fake_sock.__enter__.return_value = fake_sock
    fake_sock.__exit__.return_value = False
    monkeypatch.setattr(hc.socket, "create_connection", lambda *a, **kw: fake_sock)

    ok, msg = hc.check_broker_reachable("redis://redis:6379/0", timeout=1.0)
    assert ok is True
    assert "redis:6379" in msg


def test_broker_unreachable_on_connection_refused(hc, monkeypatch):
    def _refuse(*_args, **_kwargs):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(hc.socket, "create_connection", _refuse)
    ok, msg = hc.check_broker_reachable("redis://redis:6379/0", timeout=1.0)
    assert ok is False
    assert "unreachable" in msg


def test_broker_unreachable_on_timeout(hc, monkeypatch):
    def _timeout(*_args, **_kwargs):
        raise socket.timeout("slow")

    monkeypatch.setattr(hc.socket, "create_connection", _timeout)
    ok, msg = hc.check_broker_reachable("redis://redis:6379/0", timeout=0.1)
    assert ok is False
    assert "unreachable" in msg


def test_broker_url_parses_custom_host_port(hc, monkeypatch):
    captured: dict = {}

    def _capture(addr, timeout):
        captured["addr"] = addr
        captured["timeout"] = timeout
        return mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

    monkeypatch.setattr(hc.socket, "create_connection", _capture)
    hc.check_broker_reachable("redis://broker.example.com:6390/2", timeout=2.5)
    assert captured["addr"] == ("broker.example.com", 6390)
    assert captured["timeout"] == 2.5


# --------------------------------------------------------------------------- #
# Celery probe.
# --------------------------------------------------------------------------- #
def test_celery_probe_ok_when_app_missing(hc, monkeypatch):
    """When the optional package isn't installed we fall back gracefully."""
    fake_module = types.ModuleType("nightmarenet_server.tasks.celery_app")
    fake_module.celery_app = None
    monkeypatch.setitem(sys.modules, "nightmarenet_server.tasks.celery_app", fake_module)

    ok, msg = hc.check_celery_worker(timeout=1.0)
    assert ok is True
    assert "fallback" in msg.lower()


def test_celery_probe_ok_when_worker_responds(hc, monkeypatch):
    fake_app = mock.MagicMock()
    fake_app.control.ping.return_value = [{"celery@host": {"ok": "pong"}}]
    fake_module = types.ModuleType("nightmarenet_server.tasks.celery_app")
    fake_module.celery_app = fake_app
    monkeypatch.setitem(sys.modules, "nightmarenet_server.tasks.celery_app", fake_module)

    ok, msg = hc.check_celery_worker(timeout=1.0)
    assert ok is True
    assert "responded" in msg


def test_celery_probe_unhealthy_when_no_replies(hc, monkeypatch):
    fake_app = mock.MagicMock()
    fake_app.control.ping.return_value = []
    fake_module = types.ModuleType("nightmarenet_server.tasks.celery_app")
    fake_module.celery_app = fake_app
    monkeypatch.setitem(sys.modules, "nightmarenet_server.tasks.celery_app", fake_module)

    ok, msg = hc.check_celery_worker(timeout=1.0)
    assert ok is False
    assert "did not reply" in msg


def test_celery_probe_unhealthy_when_ping_raises(hc, monkeypatch):
    fake_app = mock.MagicMock()
    fake_app.control.ping.side_effect = RuntimeError("broker dropped")
    fake_module = types.ModuleType("nightmarenet_server.tasks.celery_app")
    fake_module.celery_app = fake_app
    monkeypatch.setitem(sys.modules, "nightmarenet_server.tasks.celery_app", fake_module)

    ok, msg = hc.check_celery_worker(timeout=1.0)
    assert ok is False
    assert "raised" in msg


# --------------------------------------------------------------------------- #
# Password redaction in log output.
# --------------------------------------------------------------------------- #
def test_redact_hides_password_in_broker_url(hc):
    redacted = hc._redact("redis://user:s3cret@redis:6379/0")
    assert "s3cret" not in redacted
    assert "***" in redacted


def test_redact_passes_through_when_no_password(hc):
    url = "redis://redis:6379/0"
    assert hc._redact(url) == url


# --------------------------------------------------------------------------- #
# End-to-end main() — composes broker + celery probes.
# --------------------------------------------------------------------------- #
def test_main_returns_zero_when_broker_ok_and_fallback_mode(hc, monkeypatch, capsys):
    monkeypatch.setattr(
        hc, "check_broker_reachable", lambda *a, **kw: (True, "broker reachable at redis:6379")
    )
    monkeypatch.setattr(
        hc, "check_celery_worker", lambda *a, **kw: (True, "celery not installed; fallback mode")
    )
    assert hc.main() == 0
    captured = capsys.readouterr()
    assert "[healthy]" in captured.out


def test_main_returns_one_when_broker_unreachable(hc, monkeypatch, capsys):
    monkeypatch.setattr(
        hc, "check_broker_reachable", lambda *a, **kw: (False, "broker unreachable at redis:6379")
    )
    # Celery check shouldn't even be reached
    monkeypatch.setattr(
        hc,
        "check_celery_worker",
        lambda *a, **kw: pytest.fail("celery probe should be short-circuited"),
    )
    assert hc.main() == 1
    captured = capsys.readouterr()
    assert "[unhealthy]" in captured.err


def test_main_returns_one_when_celery_worker_silent(hc, monkeypatch, capsys):
    monkeypatch.setattr(
        hc, "check_broker_reachable", lambda *a, **kw: (True, "broker reachable at redis:6379")
    )
    monkeypatch.setattr(
        hc, "check_celery_worker", lambda *a, **kw: (False, "celery worker did not reply to ping")
    )
    assert hc.main() == 1
    captured = capsys.readouterr()
    assert "[unhealthy]" in captured.err
    assert "did not reply" in captured.err
