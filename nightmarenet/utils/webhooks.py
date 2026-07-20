"""Webhook notification dispatcher and GPU monitoring utilities for NightmareNet."""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def validate_webhook_url(url: str) -> bool:
    """Validate a webhook URL against the allowlist and block internal IPs.

    Args:
        url: The webhook URL to validate. Must be HTTPS and resolve to
            a public (non-private, non-loopback) IP address.

    Returns:
        True if the URL passes all validation checks, False otherwise.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Allowlist check with path restrictions
        allowed = False
        if hostname == "hooks.slack.com" and parsed.path.startswith("/services/"):
            allowed = True
        elif hostname in ("discord.com", "discordapp.com") and parsed.path.startswith(
            "/api/webhooks/"
        ):
            allowed = True
        elif hostname.endswith(".webhook.office.com"):
            allowed = True

        if not allowed:
            return False

        # Resolve all addresses and reject if any is non-global
        try:
            addr_infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        except socket.gaierror:
            return False

        if not addr_infos:
            return False

        for addr_info in addr_infos:
            ip_str = addr_info[4][0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return False

        return True
    except Exception:
        return False


def trigger_webhook(
    config: dict,
    event_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
    max_retries: int = 3,
) -> None:
    """Send webhook notifications to configured endpoints based on event_type.

    Args:
        config: Full configuration dictionary containing notifications.webhooks.
        event_type: One of 'run_complete', 'regression_detected', 'alert', 'deploy'.
        message: The headline text/message.
        details: A dictionary of key-value details to include.
        timeout: Request timeout in seconds (default 5.0).
        max_retries: Maximum attempt count for transient errors (default 3).
    """
    webhooks = config.get("notifications", {}).get("webhooks", [])
    if not webhooks:
        return

    for webhook in webhooks:
        url = webhook.get("url")
        if not url:
            continue
        events = webhook.get("events")
        # If events is not specified, default to all event types
        if events is not None and event_type not in events:
            continue

        try:
            _send_webhook_request(
                url,
                event_type,
                message,
                details or {},
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception as e:
            logger.warning("Failed to send webhook notification to %s: %s", url, e)


def _send_webhook_request(
    url: str,
    event_type: str,
    message: str,
    details: Dict[str, Any],
    timeout: float = 5.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> None:
    """Send an HTTP POST request to a webhook URL with exponential backoff retry logic.

    Args:
        url: Destination webhook URL.
        event_type: Event type identifier.
        message: Primary notification message.
        details: Additional structured event metadata.
        timeout: Timeout per request attempt in seconds.
        max_retries: Total number of attempts to make (default 3).
        backoff_factor: Multiplier for exponential backoff sleep interval.
    """
    # Build payload based on URL/destination
    payload: Dict[str, Any] = {}

    details_str = ""
    if details:
        details_str = "\n".join(f"- **{k}**: {v}" for k, v in details.items())

    if "slack.com" in url:
        payload = {
            "text": f"*{message}*",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"NightmareNet: {event_type.upper()}",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{message}\n\n{details_str}" if details_str else message,
                    },
                },
            ],
        }
    elif "discord.com" in url or "discordapp.com" in url:
        payload = {
            "embeds": [
                {
                    "title": f"NightmareNet: {event_type.upper()}",
                    "description": message,
                    "color": (
                        16738304 if event_type in ("alert", "regression_detected") else 3447003
                    ),
                    "fields": [
                        {"name": k, "value": str(v), "inline": True} for k, v in details.items()
                    ]
                    if details
                    else [],
                }
            ]
        }
    elif "office.com" in url or "microsoft.com" in url or "webhook.office.com" in url:
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": (
                "FF0000" if event_type in ("alert", "regression_detected") else "0078D7"
            ),
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
        # Generic compatibility payload
        payload = {
            "event": event_type,
            "message": message,
            "details": details,
            "text": f"{message}\n\n{details_str}" if details_str else message,
            "content": f"**NightmareNet: {event_type.upper()}**\n{message}\n\n{details_str}"
            if details_str
            else message,
        }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "NightmareNet-Webhook/0.2.0"},
    )

    attempts = max(1, max_retries)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response.read()
            return
        except urllib.error.HTTPError as e:
            is_transient = (e.code == 429) or (500 <= e.code < 600)
            if is_transient and attempt < attempts - 1:
                sleep_secs = backoff_factor * (2**attempt)
                logger.warning(
                    "Webhook request to %s failed with status %d (attempt %d/%d). Retrying in %.1f seconds...",
                    url,
                    e.code,
                    attempt + 1,
                    attempts,
                    sleep_secs,
                )
                time.sleep(sleep_secs)
                continue
            raise e
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            if attempt < attempts - 1:
                sleep_secs = backoff_factor * (2**attempt)
                logger.warning(
                    "Webhook request to %s failed with network error %s (attempt %d/%d). Retrying in %.1f seconds...",
                    url,
                    e,
                    attempt + 1,
                    attempts,
                    sleep_secs,
                )
                time.sleep(sleep_secs)
                continue
            raise e


def check_vram_pressure(device_index: int = 0, threshold: float = 0.85) -> bool:
    """Check if the GPU VRAM usage ratio exceeds a threshold.

    Args:
        device_index: Index of the CUDA device.
        threshold: Memory usage ratio threshold (default 0.85).

    Returns:
        True if the usage is above the threshold, False otherwise.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # Get free and total memory from CUDA (in bytes)
        free, total = torch.cuda.mem_get_info(device_index)
        used = total - free
        ratio = used / total
        return ratio > threshold
    except Exception:
        return False
