"""Tests for webhook validation and blocked internal IPs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nightmarenet.utils.message_builders import (
    DiscordMessageBuilder,
    SlackMessageBuilder,
    build_webhook_payload,
)
from nightmarenet.utils.webhooks import validate_webhook_url


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


class TestSlackMessageBuilder:
    def test_build_basic_message(self):
        payload = SlackMessageBuilder.build("run_complete", "Test message")
        assert "blocks" in payload
        assert "text" in payload
        assert len(payload["blocks"]) >= 2  # header + section + divider

    def test_build_with_details(self):
        details = {"run_id": "123", "model": "gpt-4"}
        payload = SlackMessageBuilder.build("run_complete", "Test", details)
        assert any("fields" in block for block in payload["blocks"])
        assert any("run_id" in str(block) for block in payload["blocks"])

    def test_build_with_dashboard_url(self):
        payload = SlackMessageBuilder.build(
            "run_complete", "Test", dashboard_url="https://example.com"
        )
        assert any(block.get("type") == "actions" for block in payload["blocks"])
        actions_block = next(b for b in payload["blocks"] if b.get("type") == "actions")
        assert actions_block["elements"][0]["url"] == "https://example.com"

    def test_emoji_selection(self):
        payload = SlackMessageBuilder.build("run_complete", "Test")
        assert "✅" in payload["blocks"][0]["text"]["text"]
        payload = SlackMessageBuilder.build("regression_detected", "Test")
        assert "⚠️" in payload["blocks"][0]["text"]["text"]
        payload = SlackMessageBuilder.build("alert", "Test")
        assert "🚨" in payload["blocks"][0]["text"]["text"]
        payload = SlackMessageBuilder.build("deploy", "Test")
        assert "🚀" in payload["blocks"][0]["text"]["text"]

    def test_color_selection(self):
        assert (
            SlackMessageBuilder._get_color("alert") == SlackMessageBuilder.COLOR_ERROR
        )
        assert (
            SlackMessageBuilder._get_color("regression_detected")
            == SlackMessageBuilder.COLOR_ERROR
        )
        assert (
            SlackMessageBuilder._get_color("run_complete")
            == SlackMessageBuilder.COLOR_SUCCESS
        )
        assert SlackMessageBuilder._get_color("deploy") == SlackMessageBuilder.COLOR_INFO


class TestDiscordMessageBuilder:
    def test_build_basic_message(self):
        payload = DiscordMessageBuilder.build("run_complete", "Test message")
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        assert payload["embeds"][0]["title"]
        assert payload["embeds"][0]["description"] == "Test message"

    def test_build_with_details(self):
        details = {"run_id": "123", "model": "gpt-4"}
        payload = DiscordMessageBuilder.build("run_complete", "Test", details)
        embed = payload["embeds"][0]
        assert "fields" in embed
        assert len(embed["fields"]) == 2
        assert embed["fields"][0]["name"] == "run_id"
        assert embed["fields"][0]["value"] == "123"

    def test_build_with_dashboard_url(self):
        payload = DiscordMessageBuilder.build(
            "run_complete", "Test", dashboard_url="https://example.com"
        )
        assert payload["embeds"][0]["url"] == "https://example.com"

    def test_emoji_selection(self):
        payload = DiscordMessageBuilder.build("run_complete", "Test")
        assert "✅" in payload["embeds"][0]["title"]
        payload = DiscordMessageBuilder.build("regression_detected", "Test")
        assert "⚠️" in payload["embeds"][0]["title"]
        payload = DiscordMessageBuilder.build("alert", "Test")
        assert "🚨" in payload["embeds"][0]["title"]
        payload = DiscordMessageBuilder.build("deploy", "Test")
        assert "🚀" in payload["embeds"][0]["title"]

    def test_color_selection(self):
        assert (
            DiscordMessageBuilder._get_color("alert") == DiscordMessageBuilder.COLOR_ERROR
        )
        assert (
            DiscordMessageBuilder._get_color("regression_detected")
            == DiscordMessageBuilder.COLOR_ERROR
        )
        assert (
            DiscordMessageBuilder._get_color("run_complete")
            == DiscordMessageBuilder.COLOR_SUCCESS
        )
        assert DiscordMessageBuilder._get_color("deploy") == DiscordMessageBuilder.COLOR_INFO

    def test_timestamp_present(self):
        payload = DiscordMessageBuilder.build("run_complete", "Test")
        assert "timestamp" in payload["embeds"][0]
        assert payload["embeds"][0]["timestamp"].endswith("Z")


class TestBuildWebhookPayload:
    def test_slack_url_uses_slack_builder(self):
        payload = build_webhook_payload(
            "https://hooks.slack.com/services/T/B/x", "run_complete", "Test"
        )
        assert "blocks" in payload
        assert payload["blocks"][0]["type"] == "header"

    def test_discord_url_uses_discord_builder(self):
        payload = build_webhook_payload(
            "https://discord.com/api/webhooks/123/token", "run_complete", "Test"
        )
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1

    def test_discordapp_url_uses_discord_builder(self):
        payload = build_webhook_payload(
            "https://discordapp.com/api/webhooks/123/token", "run_complete", "Test"
        )
        assert "embeds" in payload

    def test_office_url_uses_office_format(self):
        payload = build_webhook_payload(
            "https://example.webhook.office.com/webhook", "run_complete", "Test"
        )
        assert "@type" in payload
        assert payload["@type"] == "MessageCard"

    def test_generic_url_uses_generic_format(self):
        payload = build_webhook_payload("https://example.com/webhook", "run_complete", "Test")
        assert "event" in payload
        assert payload["event"] == "run_complete"
        assert "message" in payload

    def test_dashboard_url_passed_to_builders(self):
        payload = build_webhook_payload(
            "https://hooks.slack.com/services/T/B/x", "run_complete", "Test", dashboard_url="https://dash.com"
        )
        assert any(block.get("type") == "actions" for block in payload["blocks"])

