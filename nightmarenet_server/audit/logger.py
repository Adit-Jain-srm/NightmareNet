"""Append-only audit event writer and query helpers."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from nightmarenet_server.audit.actions import AuditAction

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 365


def get_retention_days() -> int:
    """Return configured audit retention in days (default 1 year)."""
    raw = os.environ.get("AUDIT_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_RETENTION_DAYS
    return max(1, days)


def retention_cutoff(now: Optional[datetime] = None) -> datetime:
    """Timestamp before which events are eligible for retention purge."""
    current = now or datetime.now(timezone.utc)
    return current - timedelta(days=get_retention_days())


def write_audit_event(
    session: Any,
    *,
    action: Union[AuditAction, str],
    entity_type: str,
    entity_id: str,
    actor_id: Optional[str] = None,
    actor_role: Optional[str] = None,
    org_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
    commit: bool = True,
) -> Any:
    """Append one immutable audit record.

    Args:
        session: SQLAlchemy session.
        action: AuditAction or string value.
        entity_type: Resource / entity type.
        entity_id: Resource / entity id.
        actor_id: Authenticated user id (optional).
        actor_role: Actor role claim (optional).
        org_id: Tenant id (optional; stored when provided).
        metadata: Extra JSON-serializable context.
        ip_address: Client IP.
        request_id: Correlation id from request tracing.
        commit: Commit the session after insert.

    Returns:
        Persisted ``AuditLog`` row.
    """
    from nightmarenet_server.models.tables import AuditLog

    action_value = action.value if isinstance(action, AuditAction) else str(action)
    row = AuditLog(
        id=str(uuid.uuid4()),
        org_id=org_id,
        user_id=actor_id,
        actor_role=actor_role,
        action=action_value,
        resource_type=entity_type,
        resource_id=entity_id,
        metadata_json=json.dumps(metadata) if metadata is not None else None,
        ip_address=ip_address,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc),
    )
    session.add(row)
    if commit:
        session.commit()
        session.refresh(row)
    else:
        session.flush()
    return row


def query_audit_events(
    session: Any,
    *,
    actor: Optional[str] = None,
    entity: Optional[str] = None,
    after: Optional[datetime] = None,
    action: Optional[str] = None,
    request_id: Optional[str] = None,
    org_id: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
) -> Tuple[List[Any], int]:
    """Return paginated audit events with optional filters.

    ``entity`` matches ``resource_type`` or ``resource_id``.
    """
    from nightmarenet_server.models.tables import AuditLog

    query = session.query(AuditLog)
    if actor is not None:
        query = query.filter(AuditLog.user_id == actor)
    if entity is not None:
        query = query.filter((AuditLog.resource_type == entity) | (AuditLog.resource_id == entity))
    if after is not None:
        query = query.filter(AuditLog.timestamp >= after)
    if action is not None:
        query = query.filter(AuditLog.action == action)
    if request_id is not None:
        query = query.filter(AuditLog.request_id == request_id)
    if org_id is not None:
        query = query.filter(AuditLog.org_id == org_id)

    total = query.count()
    rows = (
        query.order_by(AuditLog.timestamp.desc(), AuditLog.id.desc())
        .offset(max(0, offset))
        .limit(max(1, min(limit, 200)))
        .all()
    )
    return rows, total


def serialize_audit_event(row: Any) -> Dict[str, Any]:
    """Serialize an AuditLog row for the query API."""
    metadata: Any = None
    raw = getattr(row, "metadata_json", None)
    if raw:
        try:
            metadata = json.loads(raw)
        except (TypeError, ValueError):
            metadata = {"raw": raw}
    ts = getattr(row, "timestamp", None)
    return {
        "id": row.id,
        "timestamp": ts.isoformat() if ts is not None else None,
        "actor_id": row.user_id,
        "actor_role": getattr(row, "actor_role", None),
        "action": row.action,
        "entity_type": row.resource_type,
        "entity_id": row.resource_id,
        "metadata": metadata,
        "ip_address": getattr(row, "ip_address", None),
        "request_id": getattr(row, "request_id", None),
        "org_id": row.org_id,
    }


def enforce_append_only(mapper: Any, connection: Any, target: Any) -> None:
    """SQLAlchemy before_update/before_delete guard for SQLite and app layer."""
    raise RuntimeError("audit_logs is append-only: UPDATE/DELETE are forbidden")


def register_immutability_guards() -> None:
    """Install ORM-level append-only guards (complements DB triggers on Postgres)."""
    from sqlalchemy import event

    from nightmarenet_server.models.tables import AuditLog

    if getattr(AuditLog, "_audit_immutability_registered", False):
        return
    event.listen(AuditLog, "before_update", enforce_append_only)
    event.listen(AuditLog, "before_delete", enforce_append_only)
    AuditLog._audit_immutability_registered = True  # type: ignore[attr-defined]
