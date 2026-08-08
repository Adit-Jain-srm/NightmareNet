"""Hosted API versioning and deprecation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypeVar

from nightmarenet.api.versioning import deprecated as core_deprecated
from nightmarenet.api.versioning import get_deprecation_headers as core_get_deprecation_headers

F = TypeVar("F", bound=Callable[..., Any])

API_VERSION_HEADER = "API-Version"
API_VERSION_VALUE = "v1"


def deprecated(sunset: str, alternative: Optional[str] = None) -> Callable[[F], F]:
    """Hosted endpoint deprecation decorator, aliasing shared core logic."""
    return core_deprecated(sunset=sunset, alternative=alternative)


def get_deprecation_headers(endpoint: Optional[Any]) -> dict[str, str]:
    """Return Deprecation/Sunset headers for hosted endpoints."""
    return core_get_deprecation_headers(endpoint)
