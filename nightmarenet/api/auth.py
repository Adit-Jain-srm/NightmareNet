"""API key authentication middleware for NightmareNet."""

import hmac
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import Request, Response  # type: ignore[import-untyped]
    from fastapi.responses import JSONResponse  # type: ignore[import-untyped]
    from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed. Install with: pip install nightmarenet[api]"
    ) from e

# Paths that bypass authentication
_PUBLIC_PATHS = frozenset(
    {
        "/api/v1/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)

# Public path prefixes (everything starting with one of these is exempt).
# Badges are intentionally public so they can be embedded in READMEs and
# served at CDN-friendly cache durations without leaking API keys.
_PUBLIC_PREFIXES: tuple = (
    "/api/v1/badge/",
    "/ws/",
)

# Exempt origins: requests from these origins skip auth (same-origin dev proxy).
_EXEMPT_REFERERS: tuple = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header against NIGHTMARENET_API_KEY env var.

    If the env var is not set, auth is disabled (dev mode) with a startup warning.
    Public paths (health, docs) are always exempt.
    """

    def __init__(self, app, api_key: Optional[str] = None):
        super().__init__(app)
        self.api_key = api_key or os.environ.get("NIGHTMARENET_API_KEY")
        if not self.api_key:
            logger.warning(
                "NIGHTMARENET_API_KEY not set — API authentication is DISABLED. "
                "Set the env var for production use."
            )

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path in _PUBLIC_PATHS:
            return await call_next(request)
        if any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES):
            return await call_next(request)

        # Allow same-origin requests from the dev frontend proxy
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        if any(origin.startswith(r) or referer.startswith(r) for r in _EXEMPT_REFERERS):
            return await call_next(request)

        active_key = os.environ.get("NIGHTMARENET_API_KEY") or self.api_key
        if not active_key:
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key")
        if not provided_key or not hmac.compare_digest(provided_key, active_key):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Invalid or missing API key."},
            )

        return await call_next(request)
