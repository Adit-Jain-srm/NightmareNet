"""Shared constants and configuration for NightmareNet API.

Centralizes values that are used across multiple API modules to avoid
duplication and ensure consistency.
"""

import os

# --- File paths ---
WEBHOOKS_FILE_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "webhooks.json",
)
