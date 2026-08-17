"""Query API for immutable audit events."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException, Query

    _FASTAPI_AVAILABLE = True
except ImportError:
    APIRouter = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    Query = None  # type: ignore[assignment,misc]
    _FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)

if _FASTAPI_AVAILABLE:
    _ACTOR_Q = Query(None, description="Filter by actor_id / user_id")
    _ENTITY_Q = Query(None, description="Filter by entity type or id")
    _AFTER_Q = Query(None, description="Only events at/after this time")
    _ACTION_Q = Query(None, description="Filter by action enum value")
    _REQUEST_ID_Q = Query(None, description="Filter by correlation request id")
    _OFFSET_Q = Query(0, ge=0)
    _LIMIT_Q = Query(50, ge=1, le=200)
else:
    _ACTOR_Q = None
    _ENTITY_Q = None
    _AFTER_Q = None
    _ACTION_Q = None
    _REQUEST_ID_Q = None
    _OFFSET_Q = 0
    _LIMIT_Q = 50


def build_audit_router() -> Optional[Any]:
    if not _FASTAPI_AVAILABLE:
        return None

    from nightmarenet_server.audit.logger import query_audit_events, serialize_audit_event
    from nightmarenet_server.models.base import DEFAULT_DATABASE_URL, get_session_factory

    router = APIRouter(prefix="/api/v1/audit", tags=["audit"])
    db_url = os.environ.get("NIGHTMARENET_DATABASE_URL", DEFAULT_DATABASE_URL)
    session_factory = get_session_factory(db_url)

    @router.get("")
    async def list_audit_events(
        actor: Optional[str] = _ACTOR_Q,
        entity: Optional[str] = _ENTITY_Q,
        after: Optional[datetime] = _AFTER_Q,
        action: Optional[str] = _ACTION_Q,
        request_id: Optional[str] = _REQUEST_ID_Q,
        offset: int = _OFFSET_Q,
        limit: int = _LIMIT_Q,
    ) -> Dict[str, Any]:
        session = session_factory()
        try:
            rows, total = query_audit_events(
                session,
                actor=actor,
                entity=entity,
                after=after,
                action=action,
                request_id=request_id,
                offset=offset,
                limit=limit,
            )
            items: List[Dict[str, Any]] = [serialize_audit_event(r) for r in rows]
            return {
                "items": items,
                "total": total,
                "offset": offset,
                "limit": limit,
            }
        except Exception as exc:
            logger.exception("audit query failed")
            raise HTTPException(status_code=500, detail="Failed to query audit events") from exc
        finally:
            session.close()

    return router
