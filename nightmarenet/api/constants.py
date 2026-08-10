"""Shared constants and configuration for NightmareNet API.

Centralizes values that are used across multiple API modules to avoid
duplication and ensure consistency.
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

# --- File paths ---
WEBHOOKS_FILE_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "webhooks.json",
)

# --- Shared rate limiter ---
limiter = Limiter(key_func=get_remote_address)
