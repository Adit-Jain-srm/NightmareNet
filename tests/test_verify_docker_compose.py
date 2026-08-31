"""Tests for scripts/verify_docker_compose.py.

These exercise the script's decision logic (compose-CLI discovery, the
service-not-running heuristic, and the API/redis/postgres/worker checks)
with subprocess and HTTP calls mocked out — no real Docker daemon or
network access is required or used.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.verify_docker_compose import (
    _find_compose_command,
    _service_not_running,
    check_api_health,
    check_redis,
    run_check,
)


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["docker", "compose", "exec"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestServiceNotRunningHeuristic:
    def test_detects_not_running_phrase(self):
        proc = _completed(returncode=1, stderr='service "redis" is not running')
        assert _service_not_running(proc) is True

    def test_detects_no_container_found(self):
        proc = _completed(returncode=1, stderr="no container found for redis_1")
        assert _service_not_running(proc) is True

    def test_does_not_flag_unrelated_errors(self):
        proc = _completed(returncode=1, stderr="permission denied")
        assert _service_not_running(proc) is False


class TestFindComposeCommand:
    @patch("scripts.verify_docker_compose.shutil.which")
    @patch("scripts.verify_docker_compose.subprocess.run")
    def test_prefers_docker_compose_v2(self, mock_run, mock_which):
        mock_which.side_effect = lambda name: f"/usr/bin/{name}" if name == "docker" else None
        mock_run.return_value = _completed(returncode=0, stdout="Docker Compose version v2.24.0")
        assert _find_compose_command() == ["docker", "compose"]

    @patch("scripts.verify_docker_compose.shutil.which")
    @patch("scripts.verify_docker_compose.subprocess.run")
    def test_falls_back_to_docker_compose_v1(self, mock_run, mock_which):
        def which(name):
            return f"/usr/bin/{name}" if name in ("docker", "docker-compose") else None

        mock_which.side_effect = which

        def run(cmd, **kwargs):
            if cmd[:2] == ["docker", "compose"]:
                return _completed(returncode=1, stderr="unknown command")
            return _completed(returncode=0, stdout="docker-compose version 1.29.2")

        mock_run.side_effect = run
        assert _find_compose_command() == ["docker-compose"]

    @patch("scripts.verify_docker_compose.shutil.which", return_value=None)
    def test_returns_none_when_nothing_available(self, mock_which):
        assert _find_compose_command() is None


class TestCheckApiHealth:
    @patch("scripts.verify_docker_compose.requests.get")
    def test_success_on_first_attempt(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"status": "ok", "version": "1.2.3"}
        )
        ok, message, skipped = check_api_health("127.0.0.1", 8000, timeout=1.0, retries=3)
        assert ok is True
        assert skipped is False
        assert "1.2.3" in message
        mock_get.assert_called_once()

    @patch("scripts.verify_docker_compose.requests.get")
    def test_unexpected_status_field_fails(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"status": "degraded", "version": "1.2.3"}
        )
        ok, message, skipped = check_api_health("127.0.0.1", 8000, timeout=1.0, retries=1)
        assert ok is False
        assert "unexpected body" in message

    @patch("scripts.verify_docker_compose.time.sleep", return_value=None)
    @patch("scripts.verify_docker_compose.requests.get")
    def test_retries_then_fails_with_clear_message(self, mock_get, _mock_sleep):
        mock_get.side_effect = requests.ConnectionError("refused")
        ok, message, skipped = check_api_health("127.0.0.1", 8000, timeout=0.1, retries=3)
        assert ok is False
        assert skipped is False
        assert mock_get.call_count == 3
        assert "unreachable after 3 attempts" in message

    @patch("scripts.verify_docker_compose.time.sleep", return_value=None)
    @patch("scripts.verify_docker_compose.requests.get")
    def test_recovers_after_transient_failure(self, mock_get, _mock_sleep):
        mock_get.side_effect = [
            requests.ConnectionError("refused"),
            MagicMock(status_code=200, json=lambda: {"status": "ok", "version": "0.9.0"}),
        ]
        ok, message, skipped = check_api_health("127.0.0.1", 8000, timeout=0.1, retries=3)
        assert ok is True
        assert "0.9.0" in message


class TestCheckRedis:
    @patch("scripts.verify_docker_compose.subprocess.run")
    def test_pass_on_pong(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="PONG\n")
        ok, message, skipped = check_redis(["docker", "compose"], "redis", 5.0, hosted=False)
        assert ok is True
        assert skipped is False
        assert "PONG" in message

    @patch("scripts.verify_docker_compose.subprocess.run")
    def test_skipped_when_not_running_and_default_profile(self, mock_run):
        mock_run.return_value = _completed(returncode=1, stderr='service "redis" is not running')
        ok, message, skipped = check_redis(["docker", "compose"], "redis", 5.0, hosted=False)
        assert ok is False
        assert skipped is True
        assert "--profile hosted" in message

    @patch("scripts.verify_docker_compose.subprocess.run")
    def test_hard_failure_when_not_running_and_hosted_profile(self, mock_run):
        mock_run.return_value = _completed(returncode=1, stderr='service "redis" is not running')
        ok, message, skipped = check_redis(["docker", "compose"], "redis", 5.0, hosted=True)
        assert ok is False
        assert skipped is False


class TestRunCheck:
    def test_wraps_timeout_as_fail(self):
        def fn():
            raise subprocess.TimeoutExpired(cmd="x", timeout=1.0)

        result = run_check("some check", fn, optional=False, timeout=1.0)
        assert result.status == "FAIL"
        assert "timed out" in result.message

    def test_wraps_unexpected_exception_as_fail(self):
        def fn():
            raise RuntimeError("boom")

        result = run_check("some check", fn, optional=False, timeout=1.0)
        assert result.status == "FAIL"
        assert "boom" in result.message

    def test_optional_failure_reported_as_skip(self):
        def fn():
            return False, "not running", False

        result = run_check("some check", fn, optional=True, timeout=1.0)
        assert result.status == "SKIP"

    def test_required_failure_reported_as_fail(self):
        def fn():
            return False, "not running", False

        result = run_check("some check", fn, optional=False, timeout=1.0)
        assert result.status == "FAIL"

    def test_explicit_skip_flag_reported_as_skip_even_when_required(self):
        def fn():
            return False, "service not running", True

        result = run_check("some check", fn, optional=False, timeout=1.0)
        assert result.status == "SKIP"


@pytest.mark.parametrize("api_ok", [True, False])
def test_main_exit_code_matches_failures(api_ok, monkeypatch):
    """Smoke test: main() exits 0 only when there are zero FAIL results."""
    from scripts.verify_docker_compose import main

    # No compose CLI in this sandbox, but redis/db/worker/reachability
    # checks are skipped entirely when compose_cmd is None, so the only
    # required check left is the API health check.
    monkeypatch.setattr("scripts.verify_docker_compose._find_compose_command", lambda: None)
    monkeypatch.setattr(
        "scripts.verify_docker_compose.check_api_health",
        lambda *a, **k: (api_ok, "mocked", False),
    )
    expected = 0 if api_ok else 1
    assert main(["--api-retries", "1", "--timeout", "0.1"]) == expected
