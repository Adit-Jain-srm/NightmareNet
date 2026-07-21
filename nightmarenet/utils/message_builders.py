"""Message builders for rich webhook notifications (Slack Block Kit, Discord embeds)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class SlackMessageBuilder:
    """Builds Slack Block Kit messages for webhook notifications."""

    # Slack color codes for sidebar
    COLOR_SUCCESS = "#36a64f"  # green
    COLOR_WARNING = "#ff9900"  # orange
    COLOR_ERROR = "#ff0000"  # red
    COLOR_INFO = "#0078d7"  # blue

    @staticmethod
    def build(
        event_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        dashboard_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a Slack Block Kit message.

        Args:
            event_type: One of 'run_complete', 'regression_detected', 'alert', 'deploy'.
            message: The headline text/message.
            details: A dictionary of key-value details to include.
            dashboard_url: Optional URL to link back to the dashboard.

        Returns:
            A Slack Block Kit payload dictionary.
        """
        blocks: List[Dict[str, Any]] = []

        # Header with emoji based on event type
        emoji = SlackMessageBuilder._get_emoji(event_type)
        blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} NightmareNet: {event_type.upper().replace('_', ' ')}",
                    "emoji": True,
                },
            }
        )

        # Main message section
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message,
                },
            }
        )

        # Add details as fields if present
        if details:
            field_items = []
            for key, value in details.items():
                field_items.append({"type": "mrkdwn", "text": f"*{key}:* {value}"})

            # Slack allows up to 10 fields per section, split if needed
            for i in range(0, len(field_items), 10):
                blocks.append(
                    {
                        "type": "section",
                        "fields": field_items[i : i + 10],
                    }
                )

        # Add divider
        blocks.append({"type": "divider"})

        # Add dashboard button if URL provided
        if dashboard_url:
            blocks.append(
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "View in Dashboard",
                                "emoji": True,
                            },
                            "url": dashboard_url,
                            "style": "primary",
                        }
                    ],
                }
            )

        # Determine color based on event type
        color = SlackMessageBuilder._get_color(event_type)

        return {
            "text": f"{emoji} {message}",
            "blocks": blocks,
            "attachments": [{"color": color, "blocks": []}] if color else [],
        }

    @staticmethod
    def _get_emoji(event_type: str) -> str:
        """Get emoji for event type."""
        emoji_map = {
            "run_complete": "✅",
            "regression_detected": "⚠️",
            "alert": "🚨",
            "deploy": "🚀",
        }
        return emoji_map.get(event_type, "ℹ️")

    @staticmethod
    def _get_color(event_type: str) -> Optional[str]:
        """Get sidebar color for event type."""
        if event_type in ("alert", "regression_detected"):
            return SlackMessageBuilder.COLOR_ERROR
        elif event_type == "run_complete":
            return SlackMessageBuilder.COLOR_SUCCESS
        elif event_type == "deploy":
            return SlackMessageBuilder.COLOR_INFO
        return None


class DiscordMessageBuilder:
    """Builds Discord rich embed messages for webhook notifications."""

    # Discord color codes (decimal)
    COLOR_SUCCESS = 5763719  # green
    COLOR_WARNING = 16750848  # orange
    COLOR_ERROR = 15548997  # red
    COLOR_INFO = 3447003  # blue

    @staticmethod
    def build(
        event_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        dashboard_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a Discord rich embed message.

        Args:
            event_type: One of 'run_complete', 'regression_detected', 'alert', 'deploy'.
            message: The headline text/message.
            details: A dictionary of key-value details to include.
            dashboard_url: Optional URL to link back to the dashboard.

        Returns:
            A Discord embed payload dictionary.
        """
        color = DiscordMessageBuilder._get_color(event_type)
        emoji = DiscordMessageBuilder._get_emoji(event_type)

        embed: Dict[str, Any] = {
            "title": f"{emoji} NightmareNet: {event_type.upper().replace('_', ' ')}",
            "description": message,
            "color": color,
            "timestamp": DiscordMessageBuilder._get_timestamp(),
            "fields": [],
        }

        # Add details as fields if present
        if details:
            for key, value in details.items():
                embed["fields"].append(
                    {"name": key, "value": str(value), "inline": True}
                )

        # Add dashboard URL if provided
        if dashboard_url:
            embed["url"] = dashboard_url

        return {"embeds": [embed]}

    @staticmethod
    def _get_emoji(event_type: str) -> str:
        """Get emoji for event type."""
        emoji_map = {
            "run_complete": "✅",
            "regression_detected": "⚠️",
            "alert": "🚨",
            "deploy": "🚀",
        }
        return emoji_map.get(event_type, "ℹ️")

    @staticmethod
    def _get_color(event_type: str) -> int:
        """Get embed color for event type."""
        if event_type in ("alert", "regression_detected"):
            return DiscordMessageBuilder.COLOR_ERROR
        elif event_type == "run_complete":
            return DiscordMessageBuilder.COLOR_SUCCESS
        elif event_type == "deploy":
            return DiscordMessageBuilder.COLOR_INFO
        return DiscordMessageBuilder.COLOR_INFO

    @staticmethod
    def _get_timestamp() -> str:
        """Get current ISO 8601 timestamp."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"


def build_webhook_payload(
    url: str,
    event_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    dashboard_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Build appropriate webhook payload based on URL destination.

    Args:
        url: The webhook URL to determine the format.
        event_type: One of 'run_complete', 'regression_detected', 'alert', 'deploy'.
        message: The headline text/message.
        details: A dictionary of key-value details to include.
        dashboard_url: Optional URL to link back to the dashboard.

    Returns:
        A payload dictionary formatted for the destination platform.
    """
    if "slack.com" in url:
        return SlackMessageBuilder.build(event_type, message, details, dashboard_url)
    elif "discord.com" in url or "discordapp.com" in url:
        return DiscordMessageBuilder.build(event_type, message, details, dashboard_url)
    elif "office.com" in url or "microsoft.com" in url or "webhook.office.com" in url:
        # Microsoft Teams/Office 365 webhook format
        theme_color = "FF0000" if event_type in ("alert", "regression_detected") else "0078D7"
        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color,
            "summary": message,
            "title": f"NightmareNet: {event_type.upper()}",
            "sections": [
                {
                    "activityTitle": message,
                    "facts": [{"name": k, "value": str(v)} for k, v in details.items()]
                    if details
                    else [],
                    "markdown": True,
                }
            ],
        }
    else:
        # Generic fallback for non-Slack/Discord webhooks
        details_str = ""
        if details:
            details_str = "\n".join(f"- **{k}**: {v}" for k, v in details.items())

        return {
            "event": event_type,
            "message": message,
            "details": details,
            "text": f"{message}\n\n{details_str}" if details_str else message,
            "content": f"**NightmareNet: {event_type.upper()}**\n{message}\n\n{details_str}"
            if details_str
            else message,
        }
