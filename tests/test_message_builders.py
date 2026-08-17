import unittest

from nightmarenet.utils.message_builders import (
    DiscordMessageBuilder,
    SlackMessageBuilder,
    build_webhook_payload,
)


class TestMessageBuildersUnit(unittest.TestCase):
    def test_build_slack_payload_all_fields(self):
        payload = SlackMessageBuilder.build(
            event_type="run_complete",
            message="Training completed successfully",
            details={"Accuracy": 0.95, "Loss": 0.05},
            dashboard_url="https://dashboard.example.com/run/123",
        )
        self.assertIn("text", payload)
        self.assertIn("blocks", payload)
        self.assertTrue(len(payload["blocks"]) >= 3)
        self.assertIn("attachments", payload)

    def test_build_discord_payload_missing_optional(self):
        payload = DiscordMessageBuilder.build(
            event_type="alert",
            message="Alert triggered",
        )
        self.assertIn("embeds", payload)
        embed = payload["embeds"][0]
        self.assertEqual(embed["description"], "Alert triggered")
        self.assertEqual(embed["fields"], [])

    def test_unicode_and_large_payload(self):
        large_details = {f"Key_{i}": "🔥 " * 50 for i in range(15)}
        payload = build_webhook_payload(
            url="https://hooks.slack.com/services/XXX",
            event_type="regression_detected",
            message="Regression in 🚀 model accuracy",
            details=large_details,
        )
        self.assertIn("blocks", payload)

    def test_build_webhook_payload_destinations(self):
        # Slack
        slack_p = build_webhook_payload("https://hooks.slack.com/test", "deploy", "Deployed v1")
        self.assertIn("blocks", slack_p)

        # Discord
        discord_p = build_webhook_payload(
            "https://discord.com/api/webhooks/test", "deploy", "Deployed v1"
        )
        self.assertIn("embeds", discord_p)

        # Teams
        teams_p = build_webhook_payload(
            "https://outlook.office.com/webhook/test", "deploy", "Deployed v1"
        )
        self.assertEqual(teams_p.get("@type"), "MessageCard")

        # Generic
        generic_p = build_webhook_payload(
            "https://custom.webhook.org/endpoint", "deploy", "Deployed v1"
        )
        self.assertEqual(generic_p.get("event"), "deploy")


if __name__ == "__main__":
    unittest.main()
