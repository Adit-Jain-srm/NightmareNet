"""HTTP middleware: correlation request ids and mutation audit logging."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Callable, Optional, Tuple

from nightmarenet_server.audit.actions import MUTATION_METHOD_ACTIONS

logger = logging.getLogger(__name__)

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

_SKIP_PREFIXES = (
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/audit",
    "/api/v1/server/health",
    "/api/v1/health",
)


class RequestIdMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """Generate or propagate ``X-Request-ID`` into ``request.state.request_id``."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
        request_id = incoming.strip() if incoming else str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def _client_ip(request: Request) -> Optional[str]:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return None


def _resolve_actor(request: Request) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (actor_id, actor_role, org_id) from Bearer JWT when possible."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None, "anonymous", None
    token = auth.split(" ", 1)[1].strip()
    if not token or token.startswith("nm_"):
        return None, "member", None
    try:
        from nightmarenet_server.auth.jwt_helpers import decode_access_token

        payload = decode_access_token(token)
        return (
            payload.get("sub"),
            payload.get("role", "member"),
            payload.get("org_id"),
        )
    except Exception:
        return None, "anonymous", None


def _entity_from_path(path: str) -> Tuple[str, str]:
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return "root", "root"
    # Prefer /api/v1/<resource>[/<id>]
    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "v1":
        entity_type = parts[2]
        entity_id = parts[3] if len(parts) > 3 else parts[2]
        return entity_type[:64], entity_id[:128]
    entity_type = parts[0][:64]
    entity_id = parts[-1][:128]
    return entity_type, entity_id


class AuditMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """Auto-log successful POST/PUT/PATCH/DELETE requests."""

    def __init__(self, app: Any, session_factory: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(app)
        self._session_factory = session_factory

    def _get_session(self) -> Any:
        if self._session_factory is not None:
            return self._session_factory()
        from nightmarenet_server.models.base import (
            DEFAULT_DATABASE_URL,
            get_session_factory,
        )

        db_url = os.environ.get("NIGHTMARENET_DATABASE_URL", DEFAULT_DATABASE_URL)
        return get_session_factory(db_url)()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method.upper()
        path = request.url.path
        should_audit = method in MUTATION_METHOD_ACTIONS and not any(
            path == prefix or path.startswith(prefix + "/") for prefix in _SKIP_PREFIXES
        )

        response = await call_next(request)

        if not should_audit:
            return response
        if response.status_code >= 400:
            return response

        try:
            from nightmarenet_server.audit.logger import write_audit_event

            actor_id, actor_role, org_id = _resolve_actor(request)
            entity_type, entity_id = _entity_from_path(path)
            request_id = getattr(request.state, "request_id", None)
            session = self._get_session()
            try:
                write_audit_event(
                    session,
                    action=MUTATION_METHOD_ACTIONS[method],
                    entity_type=entity_type,
                    entity_id=entity_id,
                    actor_id=actor_id,
                    actor_role=actor_role,
                    org_id=org_id,
                    metadata={"method": method, "path": path, "status": response.status_code},
                    ip_address=_client_ip(request),
                    request_id=request_id,
                )
            finally:
                session.close()
        except Exception:
            logger.exception("Failed to write audit event for %s %s", method, path)

        return response
