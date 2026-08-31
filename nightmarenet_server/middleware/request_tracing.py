"""FastAPI middleware for structured request tracing and correlation IDs.

Provides the RequestTracingMiddleware which extracts or generates an
X-Request-ID, stores it in a context variable for downstream logs and tasks,
attaches it to the response, and logs a completion summary.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable

from nightmarenet.utils.logging_config import request_id_ctx

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    _STARLETTE_AVAILABLE = True
except ImportError:
    BaseHTTPMiddleware = object  # type: ignore[assignment,misc]
    Request = Any  # type: ignore[misc,assignment]
    Response = Any  # type: ignore[misc,assignment]
    _STARLETTE_AVAILABLE = False


logger = logging.getLogger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """Middleware for injecting and logging request correlation IDs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # 1. Accept incoming X-Request-ID or generate a new one
        incoming_id = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
        request_id = incoming_id.strip() if incoming_id else str(uuid.uuid4())

        # 2. Store in context variable for downstream logs and tasks
        token = request_id_ctx.set(request_id)

        # Also store in request.state for local access if needed
        request.state.request_id = request_id

        try:
            # 3. Process the request
            response = await call_next(request)

            # 4. Attach request ID to response headers
            response.headers["X-Request-ID"] = request_id

            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            # 5. Log completion summary with duration
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Use extra to ensure request_id is available even if Filter fails,
            # but rely on Filter for other logs
            logger.info(
                "Request completed: %s %s",
                request.method,
                request.url.path,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "request_id": request_id,
                },
            )

            request_id_ctx.reset(token)
