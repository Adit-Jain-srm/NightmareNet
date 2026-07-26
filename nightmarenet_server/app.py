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

import logging
import os
from typing import Any, Dict, List, Optional, Union

from nightmarenet import __version__ as core_version
from nightmarenet_server import __version__ as server_version

logger = logging.getLogger(__name__)

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

    try:
        from nightmarenet.api.app import app as core_app
    except ImportError:
        core_app = None

    app = FastAPI(
        title="NightmareNet Hosted Platform",
        description=(
            "Hosted layer on top of the open-source NightmareNet core — "
            "multi-tenant auth, experiment tracking, realtime run streaming."
        ),
        version=f"{server_version}+core{core_version}",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if SessionMiddleware is not None:
        session_secret = os.environ.get(
            "NIGHTMARENET_SESSION_SECRET",
            "dev-only-change-in-production",
        )
        app.add_middleware(SessionMiddleware, secret_key=session_secret)

    _attach_oauth(app)
    _attach_realtime(app)
    _attach_api_key_routes(app)
    _attach_search(app)

    if core_app is not None:
        app.mount("/", core_app)

    @app.on_event("startup")
    async def _on_startup() -> None:
        _init_db_safe()

    @app.get("/api/v1/server/health", tags=["System"])
    async def hosted_health() -> Dict[str, Union[str, bool]]:
        return {
            "status": "ok",
            "server_version": server_version,
            "core_version": core_version,
            "oauth_enabled": bool(
                os.environ.get("NIGHTMARENET_GITHUB_CLIENT_ID")
                or os.environ.get("NIGHTMARENET_GOOGLE_CLIENT_ID")
            ),
        }

    return app


app: Optional[Any] = create_app()
