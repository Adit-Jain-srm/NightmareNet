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

from nightmarenet.utils.message_builders import build_webhook_payload

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
) -> None:
    """Send webhook notifications to configured endpoints based on event_type.

    Args:
        config: Full configuration dictionary containing notifications.webhooks.
        event_type: One of 'run_complete', 'regression_detected', 'alert', 'deploy'.
        message: The headline text/message.
        details: A dictionary of key-value details to include.
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
            _send_webhook_request(url, event_type, message, details or {})
        except Exception as e:
            logger.warning("Failed to send webhook notification to %s: %s", url, e)


def _send_webhook_request(url: str, event_type: str, message: str, details: Dict[str, Any]) -> None:
    # Build payload based on URL/destination using rich formatters
    dashboard_url = details.get("dashboard_url") if details else None
    # Remove dashboard_url from details before passing to builder
    details_copy = details.copy() if details else {}
    if dashboard_url and "dashboard_url" in details_copy:
        details_copy.pop("dashboard_url")
    payload = build_webhook_payload(url, event_type, message, details_copy, dashboard_url)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "NightmareNet-Webhook/0.2.0"},
    )

    max_retries = 1
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                response.read()
            return
        except urllib.error.HTTPError as e:
            if e.code == 429 or (500 <= e.code < 600):
                if attempt < max_retries:
                    logger.warning(
                        "Webhook request to %s failed with status %d. Retrying in 2 seconds...",
                        url,
                        e.code,
                    )
                    time.sleep(2)
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
