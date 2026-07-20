"""Tests for webhook validation, blocked internal IPs, and retry logic."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from nightmarenet.utils.webhooks import (
    _send_webhook_request,
    trigger_webhook,
    validate_webhook_url,
)


class TestValidateWebhookUrl:
    def test_rejects_http(self):
        assert validate_webhook_url("http://hooks.slack.com/services/T/B/x") is False

    def test_rejects_non_allowlisted_domain(self):
        assert validate_webhook_url("https://evil.com/hook") is False

    def test_rejects_slack_without_services_path(self):
        assert validate_webhook_url("https://hooks.slack.com/other/path") is False

    def test_accepts_slack_with_services_path(self):
        with patch("socket.getaddrinfo") as mock_res:
            mock_res.return_value = [(2, 1, 6, "", ("44.228.100.1", 0))]
            assert validate_webhook_url("https://hooks.slack.com/services/T123/B456/abc") is True

    def test_accepts_discord(self):
        with patch("socket.getaddrinfo") as mock_res:
            mock_res.return_value = [(2, 1, 6, "", ("162.159.128.1", 0))]
            assert validate_webhook_url("https://discord.com/api/webhooks/123/token") is True

    def test_rejects_internal_ip_loopback(self):
        with patch("socket.getaddrinfo") as mock_res:
            mock_res.return_value = [(2, 1, 6, "", ("127.0.0.1", 0))]
            assert validate_webhook_url("https://hooks.slack.com/services/T123/B456/abc") is False

    def test_rejects_internal_ip_private(self):
        with patch("socket.getaddrinfo") as mock_res:
            mock_res.return_value = [(2, 1, 6, "", ("192.168.1.1", 0))]
            assert validate_webhook_url("https://hooks.slack.com/services/T123/B456/abc") is False

    def test_rejects_if_any_resolved_address_is_private(self):
        with patch("socket.getaddrinfo") as mock_res:
            mock_res.return_value = [
                (2, 1, 6, "", ("44.228.100.1", 0)),
                (2, 1, 6, "", ("10.0.0.1", 0)),
            ]
            assert validate_webhook_url("https://hooks.slack.com/services/T123/B456/abc") is False

    def test_rejects_dns_failure(self):
        import socket as _socket

        with patch("socket.getaddrinfo", side_effect=_socket.gaierror("fail")):
            assert validate_webhook_url("https://hooks.slack.com/services/T123/B456/abc") is False


class TestWebhookEndpointBlocksInternalIP:
    """Regression test: the /api/v1/webhooks/test endpoint must reject
    URLs that resolve to internal IPs BEFORE dispatching."""

    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient

        from nightmarenet.api.app import app

        return TestClient(app)

    def test_rejects_internal_ip_with_400(self, client, monkeypatch):
        monkeypatch.delenv("NIGHTMARENET_API_KEY", raising=False)

        with patch("socket.getaddrinfo") as mock_res:
            mock_res.return_value = [(2, 1, 6, "", ("127.0.0.1", 0))]
            response = client.post(
                "/api/v1/notifications/test-webhook",
                json={
                    "url": "https://hooks.slack.com/services/T/B/x",
                    "event_type": "run_complete",
                },
            )

        assert response.status_code == 400
        assert "Invalid webhook URL" in response.json()["detail"]

    def test_dispatch_not_called_for_blocked_url(self, client, monkeypatch):
        monkeypatch.delenv("NIGHTMARENET_API_KEY", raising=False)

        with patch("socket.getaddrinfo") as mock_res:
            mock_res.return_value = [(2, 1, 6, "", ("10.0.0.1", 0))]
            with patch("nightmarenet.utils.webhooks.trigger_webhook") as mock_trigger:
                client.post(
                    "/api/v1/notifications/test-webhook",
                    json={
                        "url": "https://hooks.slack.com/services/T/B/x",
                        "event_type": "alert",
                    },
                )

        mock_trigger.assert_not_called()


def _make_http_error(code: int) -> urllib.error.HTTPError:
    """Helper to create an HTTPError with a given status code."""
    return urllib.error.HTTPError(
        url="https://example.com",
        code=code,
        msg=f"HTTP {code}",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )


class TestWebhookRetryLogic:
    """Tests for the exponential backoff retry behaviour in _send_webhook_request."""

    @patch("nightmarenet.utils.webhooks.time.sleep")
    @patch("nightmarenet.utils.webhooks.urllib.request.urlopen")
    def test_retries_on_500_and_succeeds(self, mock_urlopen, mock_sleep):
        """Server error on attempt 1, success on attempt 2."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            _make_http_error(500),
            mock_response,
        ]

        _send_webhook_request(
            "https://hooks.slack.com/services/T/B/x",
            "run_complete",
            "Test",
            {},
            max_retries=3,
            backoff_factor=0.01,
        )

        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once()

    @patch("nightmarenet.utils.webhooks.time.sleep")
    @patch("nightmarenet.utils.webhooks.urllib.request.urlopen")
    def test_retries_on_429_rate_limit(self, mock_urlopen, mock_sleep):
        """429 rate-limited on attempts 1 and 2, success on attempt 3."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            _make_http_error(429),
            _make_http_error(429),
            mock_response,
        ]

        _send_webhook_request(
            "https://hooks.slack.com/services/T/B/x",
            "alert",
            "Rate limited test",
            {},
            max_retries=3,
            backoff_factor=0.01,
        )

        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("nightmarenet.utils.webhooks.time.sleep")
    @patch("nightmarenet.utils.webhooks.urllib.request.urlopen")
    def test_exhausts_all_retries_on_503(self, mock_urlopen, mock_sleep):
        """503 on all 3 attempts -> raises HTTPError after exhausting retries."""
        mock_urlopen.side_effect = [
            _make_http_error(503),
            _make_http_error(503),
            _make_http_error(503),
        ]

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _send_webhook_request(
                "https://hooks.slack.com/services/T/B/x",
                "run_complete",
                "Persistent failure",
                {},
                max_retries=3,
                backoff_factor=0.01,
            )

        assert exc_info.value.code == 503
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("nightmarenet.utils.webhooks.time.sleep")
    @patch("nightmarenet.utils.webhooks.urllib.request.urlopen")
    def test_no_retry_on_400_bad_request(self, mock_urlopen, mock_sleep):
        """Non-transient 400 error -> fails immediately without retrying."""
        mock_urlopen.side_effect = _make_http_error(400)

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _send_webhook_request(
                "https://hooks.slack.com/services/T/B/x",
                "run_complete",
                "Bad request",
                {},
                max_retries=3,
                backoff_factor=0.01,
            )

        assert exc_info.value.code == 400
        assert mock_urlopen.call_count == 1
        mock_sleep.assert_not_called()

    @patch("nightmarenet.utils.webhooks.time.sleep")
    @patch("nightmarenet.utils.webhooks.urllib.request.urlopen")
    def test_no_retry_on_404(self, mock_urlopen, mock_sleep):
        """Non-transient 404 error -> fails immediately."""
        mock_urlopen.side_effect = _make_http_error(404)

        with pytest.raises(urllib.error.HTTPError):
            _send_webhook_request(
                "https://hooks.slack.com/services/T/B/x",
                "run_complete",
                "Not found",
                {},
                max_retries=3,
                backoff_factor=0.01,
            )

        assert mock_urlopen.call_count == 1
        mock_sleep.assert_not_called()

    @patch("nightmarenet.utils.webhooks.time.sleep")
    @patch("nightmarenet.utils.webhooks.urllib.request.urlopen")
    def test_retries_on_network_timeout(self, mock_urlopen, mock_sleep):
        """URLError (network timeout) on attempt 1, success on attempt 2."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection timed out"),
            mock_response,
        ]

        _send_webhook_request(
            "https://hooks.slack.com/services/T/B/x",
            "alert",
            "Timeout test",
            {},
            max_retries=3,
            backoff_factor=0.01,
        )

        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once()

    @patch("nightmarenet.utils.webhooks.time.sleep")
    @patch("nightmarenet.utils.webhooks.urllib.request.urlopen")
    def test_exponential_backoff_intervals(self, mock_urlopen, mock_sleep):
        """Verify sleep intervals follow exponential backoff pattern."""
        mock_urlopen.side_effect = [
            _make_http_error(500),
            _make_http_error(500),
            _make_http_error(500),
        ]

        with pytest.raises(urllib.error.HTTPError):
            _send_webhook_request(
                "https://hooks.slack.com/services/T/B/x",
                "run_complete",
                "Backoff test",
                {},
                max_retries=3,
                backoff_factor=1.0,
            )

        # Expect 2 sleeps: backoff_factor * 2^0 = 1.0, backoff_factor * 2^1 = 2.0
        assert mock_sleep.call_count == 2
        sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_args[0] == pytest.approx(1.0)
        assert sleep_args[1] == pytest.approx(2.0)

    @patch("nightmarenet.utils.webhooks.time.sleep")
    @patch("nightmarenet.utils.webhooks.urllib.request.urlopen")
    def test_configurable_timeout_forwarded(self, mock_urlopen, mock_sleep):
        """Verify custom timeout is forwarded to urlopen."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _send_webhook_request(
            "https://hooks.slack.com/services/T/B/x",
            "run_complete",
            "Timeout config test",
            {},
            timeout=15.0,
            max_retries=1,
        )

        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 15.0

    @patch("nightmarenet.utils.webhooks.time.sleep")
    @patch("nightmarenet.utils.webhooks.urllib.request.urlopen")
    def test_trigger_webhook_forwards_retry_params(self, mock_urlopen, mock_sleep):
        """End-to-end: trigger_webhook passes timeout and max_retries through."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            _make_http_error(503),
            mock_response,
        ]

        config = {
            "notifications": {
                "webhooks": [
                    {"url": "https://hooks.slack.com/services/T/B/x", "events": ["run_complete"]}
                ]
            }
        }

        trigger_webhook(
            config,
            "run_complete",
            "E2E retry test",
            {"run_id": "test-123"},
            timeout=10.0,
            max_retries=3,
        )

        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once()

