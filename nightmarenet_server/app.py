"""Hosted NightmareNet FastAPI application.

Mounts the open-source distortion/evaluation routes from
:mod:`nightmarenet.api.app`, layers the hosted-platform routers (OAuth,
realtime WebSocket, API-key minting) on top, and bootstraps the local
SQLAlchemy schema on startup.

Per :file:`CLAUDE.md`:

* uses ``Union[X, Y]`` annotations for Python 3.9 compatibility,
* deliberately omits ``from __future__ import annotations`` because the
  upstream OSS app uses FastAPI ``Body(...)`` (Pydantic v2 + future
  annotations is incompatible),
* guards every optional dependency (FastAPI, SQLAlchemy, Authlib, Celery)
  with ``try/except ImportError`` so the OSS test-suite continues to pass
  even without the hosted extras.
"""

import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from nightmarenet import __version__ as core_version
from nightmarenet_server import __version__ as server_version

logger = logging.getLogger(__name__)

_is_shutting_down = False
_in_flight_requests = 0


def is_shutting_down() -> bool:
    """Return whether the server is currently in graceful shutdown state."""
    return _is_shutting_down


def set_shutting_down(val: bool = True) -> None:
    """Set the graceful shutdown state flag."""
    global _is_shutting_down
    _is_shutting_down = val


def reset_shutdown_state() -> None:
    """Reset graceful shutdown state (primarily for testing)."""
    global _is_shutting_down, _in_flight_requests
    _is_shutting_down = False
    _in_flight_requests = 0


async def trigger_graceful_shutdown(grace_period: float = 25.0) -> None:
    """Execute the server graceful shutdown sequence."""
    global _is_shutting_down, _in_flight_requests
    _is_shutting_down = True
    logger.info("Graceful shutdown sequence initiated (grace_period=%.1fs)...", grace_period)

    # 1. Close all active WebSocket connections with code 1001
    try:
        from nightmarenet_server.realtime.websocket import close_all_websockets

        await close_all_websockets(code=1001, reason="Server shutting down")
    except Exception:
        logger.exception("Failed to close WebSockets during shutdown")

    # 2. Wait for in-flight requests to complete up to grace_period
    start_time = time.time()
    while _in_flight_requests > 0 and (time.time() - start_time) < grace_period:
        await asyncio.sleep(0.1)

    if _in_flight_requests > 0:
        logger.warning(
            "Grace period expired with %d in-flight request(s) remaining.",
            _in_flight_requests,
        )

    # 3. Flush pending metrics/logs
    for handler in logging.getLogger().handlers:
        handler.flush()
    logger.info("Graceful shutdown complete.")


try:
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    _FASTAPI_AVAILABLE = True
except ImportError:
    Depends = None  # type: ignore[assignment]
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    CORSMiddleware = None  # type: ignore[assignment,misc]
    _FASTAPI_AVAILABLE = False

try:
    from starlette.middleware.sessions import SessionMiddleware
except ImportError:
    SessionMiddleware = None  # type: ignore[assignment,misc]

try:
    from nightmarenet_server.middleware.versioning import (
        API_VERSION_HEADER,
        API_VERSION_VALUE,
        get_deprecation_headers,
    )
except ImportError:
    API_VERSION_HEADER = "API-Version"  # type: ignore[assignment]
    API_VERSION_VALUE = "v1"  # type: ignore[assignment]

    def get_deprecation_headers(endpoint: Optional[Any]) -> Dict[str, str]:
        return {}


def _cors_origins() -> List[str]:
    """Parse ``NIGHTMARENET_CORS_ORIGINS`` into a list."""
    raw = os.environ.get("NIGHTMARENET_CORS_ORIGINS", "*")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _attach_oauth(app: Any) -> None:
    try:
        from nightmarenet_server.auth.oauth import build_oauth_router
    except ImportError:
        logger.info("OAuth router unavailable — skipping.")
        return
    router = build_oauth_router()
    if router is None:
        logger.info("OAuth router not constructed (missing optional deps).")
        return
    app.include_router(router)


def _attach_sso(app: Any) -> None:
    try:
        from nightmarenet_server.auth.oidc import build_sso_routers
    except ImportError:
        logger.info("SSO router unavailable — skipping.")
        return
    sso_router, admin_router = build_sso_routers()
    if sso_router is not None:
        app.include_router(sso_router)
    if admin_router is not None:
        app.include_router(admin_router)


def _attach_realtime(app: Any) -> None:
    try:
        from nightmarenet_server.realtime.websocket import build_realtime_router
    except ImportError:
        logger.info("Realtime router unavailable — skipping.")
        return
    router = build_realtime_router()
    if router is None:
        return
    app.include_router(router)


def _attach_api_key_routes(app: Any) -> None:
    """Mount minimal API-key minting/revocation endpoints."""
    if not _FASTAPI_AVAILABLE:
        return

    try:
        from fastapi import APIRouter

        from nightmarenet_server.auth.api_keys import (
            mint_api_key,
            require_api_key,
            revoke_api_key,
        )
        from nightmarenet_server.models.base import (
            DEFAULT_DATABASE_URL,
            get_session_factory,
        )
    except ImportError:
        logger.info("API-key routes unavailable — skipping.")
        return

    router = APIRouter(prefix="/api/v1/keys", tags=["api-keys"])
    db_url = os.environ.get("NIGHTMARENET_DATABASE_URL", DEFAULT_DATABASE_URL)
    session_factory = get_session_factory(db_url)

    def _session() -> Any:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    require_key = require_api_key()
    session_param = Depends(_session)
    require_key_param = Depends(require_key) if require_key else None

    @router.post("")
    async def mint_key(
        body: Dict[str, Any],
        db: Any = session_param,
    ) -> Dict[str, Any]:
        org_id = body.get("org_id")
        user_id = body.get("user_id")
        if not org_id or not user_id:
            raise HTTPException(status_code=400, detail="org_id + user_id required")
        plaintext, row = mint_api_key(
            db,
            org_id=org_id,
            user_id=user_id,
            name=body.get("name", "default"),
            scopes=body.get("scopes") or [],
        )
        return {
            "id": row.id,
            "plaintext": plaintext,
            "name": row.name,
            "scopes": row.scopes,
        }

    @router.delete("/{key_id}")
    async def delete_key(
        key_id: str,
        db: Any = session_param,
        _identity: Any = require_key_param,
    ) -> Dict[str, Any]:
        ok = revoke_api_key(db, key_id)
        if not ok:
            raise HTTPException(status_code=404, detail="API key not found")
        return {"id": key_id, "revoked": True}

    app.include_router(router)


def _attach_search(app: Any) -> None:
    try:
        from nightmarenet_server.search.endpoints import build_search_router
    except ImportError:
        logger.info("Search router unavailable; skipping.")
        return
    router = build_search_router()
    if router is None:
        logger.info("Search router not constructed (missing optional deps).")
        return
    app.include_router(router)


def _attach_audit(app: Any) -> None:
    try:
        from nightmarenet_server.audit.endpoints import build_audit_router
        from nightmarenet_server.audit.logger import register_immutability_guards
    except ImportError:
        logger.info("Audit router unavailable; skipping.")
        return
    try:
        register_immutability_guards()
    except Exception:
        logger.exception("Failed to register audit immutability guards")
    router = build_audit_router()
    if router is None:
        logger.info("Audit router not constructed (missing optional deps).")
        return
    app.include_router(router)


def _attach_audit_middleware(app: Any) -> None:
    """Correlation id + mutation audit (Starlette: last added runs first)."""
    try:
        from nightmarenet_server.middleware import AuditMiddleware, RequestTracingMiddleware
    except ImportError:
        logger.info("Audit middleware unavailable; skipping.")
        return
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RequestTracingMiddleware)


def _init_db_safe() -> None:
    """Best-effort init_db; never crash app startup."""
    try:
        from nightmarenet_server.models.base import (
            DEFAULT_DATABASE_URL,
            init_db,
        )
    except ImportError:
        logger.info("SQLAlchemy not installed; skipping init_db().")
        return
    db_url = os.environ.get("NIGHTMARENET_DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        init_db(db_url)
        logger.info("Hosted DB initialised at %s", db_url)
    except Exception:
        logger.exception("init_db() failed; continuing without schema.")


def create_app() -> Optional[Any]:
    """Build the hosted FastAPI application.

    Returns ``None`` if FastAPI is not installed so callers can detect and
    fail gracefully.
    """
    if not _FASTAPI_AVAILABLE:
        logger.warning("FastAPI not installed; hosted server is disabled.")
        return None

    core_app: Optional[Any] = None
    try:
        from nightmarenet.api.app import app as _core_app

        core_app = _core_app
    except ImportError:
        pass

    @asynccontextmanager
    async def lifespan(app_instance: Any):
        _init_db_safe()

        loop = asyncio.get_running_loop()
        _shutdown_task = None

        def _sig_handler(sig: int) -> None:
            nonlocal _shutdown_task
            logger.info("Received signal %d — starting graceful shutdown sequence.", sig)
            set_shutting_down(True)
            _shutdown_task = asyncio.create_task(trigger_graceful_shutdown())

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda s=sig: _sig_handler(s))
            except Exception:
                try:
                    signal.signal(sig, lambda s, _f: _sig_handler(s))
                except Exception:
                    pass

        yield

        if _shutdown_task is not None:
            await _shutdown_task
        else:
            await trigger_graceful_shutdown(grace_period=1.0)

    app = FastAPI(
        title="NightmareNet Hosted Platform",
        description=(
            "Hosted layer on top of the open-source NightmareNet core — "
            "multi-tenant auth, experiment tracking, realtime run streaming."
        ),
        version=f"{server_version}+core{core_version}",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse

        from nightmarenet_server.middleware.rate_limiting import (
            RateLimitException,
            RateLimitingMiddleware,
        )

        @app.exception_handler(RateLimitException)
        async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
            headers = getattr(request.state, "rate_limit_headers", {})
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": exc.detail,
                },
                headers=headers,
            )

        app.add_middleware(RateLimitingMiddleware)
        logger.info("Successfully registered RateLimitingMiddleware.")

        if core_app is not None and hasattr(core_app, "state"):
            core_limiter = getattr(core_app.state, "limiter", None)
            if core_limiter is not None:
                core_limiter.enabled = False
                logger.info(
                    "Disabled core slowapi limiter"
                    " — hosted tiered middleware handles rate limiting."
                )
    except ImportError as e:
        logger.warning("Could not register RateLimitingMiddleware: %s", e)

    if SessionMiddleware is not None:
        session_secret = os.environ.get(
            "NIGHTMARENET_SESSION_SECRET",
            "dev-only-change-in-production",
        )
        app.add_middleware(SessionMiddleware, secret_key=session_secret)

    @app.middleware("http")
    async def graceful_shutdown_middleware(request: Any, call_next: Any) -> Any:
        global _in_flight_requests
        if _is_shutting_down and request.url.path in (
            "/api/v1/health",
            "/api/v1/server/health",
        ):
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content={"status": "shutting_down"})

        _in_flight_requests += 1
        try:
            return await call_next(request)
        finally:
            _in_flight_requests -= 1

    @app.middleware("http")
    async def _hosted_api_version_header(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers[API_VERSION_HEADER] = API_VERSION_VALUE
        response.headers["X-API-Version"] = core_version
        for name, value in get_deprecation_headers(request.scope.get("endpoint")).items():
            response.headers[name] = value
        return response

    _attach_audit_middleware(app)
    _attach_oauth(app)
    _attach_sso(app)
    _attach_realtime(app)
    _attach_api_key_routes(app)
    _attach_search(app)
    _attach_audit(app)

    if core_app is not None:
        app.mount("/", core_app)

    @app.get("/api/v1/server/health", tags=["System"])
    async def hosted_health() -> Any:
        if _is_shutting_down:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content={"status": "shutting_down"})
        return {
            "status": "ok",
            "server_version": server_version,
            "core_version": core_version,
            "oauth_enabled": bool(
                os.environ.get("NIGHTMARENET_GITHUB_CLIENT_ID")
                or os.environ.get("NIGHTMARENET_GOOGLE_CLIENT_ID")
            ),
            "sso_enabled": bool(
                os.environ.get("NIGHTMARENET_OIDC_CLIENT_ID")
                and os.environ.get("NIGHTMARENET_OIDC_DEFAULT_METADATA_URL")
            ),
        }

    return app


app: Optional[Any] = create_app()
