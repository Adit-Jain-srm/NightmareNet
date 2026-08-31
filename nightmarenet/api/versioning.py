"""API versioning and deprecation helpers for NightmareNet."""

import asyncio
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

API_VERSION_HEADER = "API-Version"
API_VERSION_VALUE = "v1"


def _format_http_date(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


def _parse_sunset_date(value: str) -> str:
    try:
        if "T" in value:
            dt = datetime.fromisoformat(value)
        else:
            dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("sunset must be a date in YYYY-MM-DD or ISO format") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return _format_http_date(dt)


def deprecated(sunset: str, alternative: Optional[str] = None) -> Callable[[F], F]:
    """Mark a FastAPI endpoint as deprecated and attach versioning metadata."""

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)
        else:

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

        wrapper.__deprecated__ = True  # type: ignore[attr-defined]
        wrapper.__deprecated_sunset__ = _parse_sunset_date(sunset)  # type: ignore[attr-defined]
        if alternative:
            wrapper.__deprecated_alternative__ = alternative  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def get_deprecation_headers(endpoint: Optional[Any]) -> Dict[str, str]:
    """Return Deprecation/Sunset header values for a deprecated endpoint."""
    if endpoint is None or not getattr(endpoint, "__deprecated__", False):
        return {}

    headers: Dict[str, str] = {
        "Deprecation": "true",
        "Sunset": getattr(endpoint, "__deprecated_sunset__", ""),
    }
    alternative = getattr(endpoint, "__deprecated_alternative__", None)
    if alternative:
        headers["Link"] = f'<{alternative}>; rel="alternate"'
    return headers
