"""Unit tests for immutable SOC 2 audit logging (issue #699)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from nightmarenet_server.audit.actions import MUTATION_METHOD_ACTIONS, AuditAction
from nightmarenet_server.audit.logger import (
    get_retention_days,
    query_audit_events,
    register_immutability_guards,
    serialize_audit_event,
    write_audit_event,
)
from nightmarenet_server.audit.middleware import AuditMiddleware, RequestIdMiddleware
from nightmarenet_server.models.base import Base, get_engine
from nightmarenet_server.models.tables import AuditLog


@pytest.fixture()
def session_factory(tmp_path, monkeypatch):
    db_path = tmp_path / "audit.db"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("NIGHTMARENET_DATABASE_URL", url)
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    register_immutability_guards()
    return factory


@pytest.fixture()
def session(session_factory):
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def test_all_action_types_defined():
    expected = {
        "CREATE",
        "READ",
        "UPDATE",
        "DELETE",
        "LOGIN",
        "LOGOUT",
        "EXPORT",
        "PERMISSION_CHANGE",
    }
    assert {a.value for a in AuditAction} == expected
    assert set(MUTATION_METHOD_ACTIONS.keys()) == {"POST", "PUT", "PATCH", "DELETE"}


def test_write_audit_event_persists_required_fields(session):
    row = write_audit_event(
        session,
        action=AuditAction.CREATE,
        entity_type="keys",
        entity_id="key-1",
        actor_id="user-1",
        actor_role="admin",
        org_id=None,
        metadata={"scope": "write"},
        ip_address="127.0.0.1",
        request_id="req-abc",
    )
    assert row.id
    assert row.action == "CREATE"
    assert row.user_id == "user-1"
    assert row.actor_role == "admin"
    assert row.resource_type == "keys"
    assert row.resource_id == "key-1"
    assert row.request_id == "req-abc"
    assert row.ip_address == "127.0.0.1"
    payload = serialize_audit_event(row)
    assert payload["actor_id"] == "user-1"
    assert payload["entity_type"] == "keys"
    assert payload["request_id"] == "req-abc"


def test_immutability_blocks_update_and_delete(session):
    row = write_audit_event(
        session,
        action=AuditAction.UPDATE,
        entity_type="run",
        entity_id="run-1",
        request_id="req-1",
    )
    row.action = "DELETE"
    with pytest.raises(RuntimeError, match="append-only"):
        session.commit()
    session.rollback()

    again = session.get(AuditLog, row.id)
    session.delete(again)
    with pytest.raises(RuntimeError, match="append-only"):
        session.commit()
    session.rollback()


def test_query_filters_by_actor_entity_and_after(session):
    now = datetime.now(timezone.utc)
    write_audit_event(
        session,
        action=AuditAction.CREATE,
        entity_type="keys",
        entity_id="a",
        actor_id="alice",
        request_id="r1",
    )
    write_audit_event(
        session,
        action=AuditAction.DELETE,
        entity_type="runs",
        entity_id="b",
        actor_id="bob",
        request_id="r2",
    )
    rows, total = query_audit_events(session, actor="alice")
    assert total == 1
    assert rows[0].user_id == "alice"

    rows, total = query_audit_events(session, entity="runs")
    assert total == 1
    assert rows[0].resource_id == "b"

    rows, total = query_audit_events(session, after=now - timedelta(minutes=1), limit=10)
    assert total == 2
    assert len(rows) == 2


def test_query_pagination(session):
    for i in range(5):
        write_audit_event(
            session,
            action=AuditAction.READ,
            entity_type="doc",
            entity_id=str(i),
            request_id=f"p-{i}",
        )
    page1, total = query_audit_events(session, offset=0, limit=2)
    page2, _ = query_audit_events(session, offset=2, limit=2)
    assert total == 5
    assert len(page1) == 2
    assert len(page2) == 2
    assert {r.id for r in page1}.isdisjoint({r.id for r in page2})


def test_middleware_logs_post_and_skips_get(session_factory):
    app = FastAPI()
    app.add_middleware(AuditMiddleware, session_factory=session_factory)
    app.add_middleware(RequestIdMiddleware)

    @app.post("/api/v1/keys")
    async def create_key():
        return {"ok": True}

    @app.get("/api/v1/keys")
    async def list_keys():
        return {"ok": True}

    client = TestClient(app)
    post = client.post("/api/v1/keys", headers={"X-Request-ID": "corr-1"})
    assert post.status_code == 200
    assert post.headers.get("X-Request-ID") == "corr-1"
    get = client.get("/api/v1/keys")
    assert get.status_code == 200

    db = session_factory()
    try:
        rows = db.query(AuditLog).all()
        assert len(rows) == 1
        assert rows[0].action == "CREATE"
        assert rows[0].request_id == "corr-1"
        assert rows[0].resource_type == "keys"
    finally:
        db.close()


def test_write_covers_remaining_action_types(session):
    for action in (
        AuditAction.LOGIN,
        AuditAction.LOGOUT,
        AuditAction.EXPORT,
        AuditAction.PERMISSION_CHANGE,
    ):
        row = write_audit_event(
            session,
            action=action,
            entity_type="auth",
            entity_id="session",
            request_id=f"rid-{action.value}",
        )
        assert row.action == action.value


def test_retention_days_config(monkeypatch):
    monkeypatch.setenv("AUDIT_RETENTION_DAYS", "730")
    assert get_retention_days() == 730
    monkeypatch.delenv("AUDIT_RETENTION_DAYS", raising=False)
    assert get_retention_days() == 365


def test_audit_write_throughput_benchmark(session):
    n = 1200
    start = time.perf_counter()
    for i in range(n):
        write_audit_event(
            session,
            action=AuditAction.CREATE,
            entity_type="bench",
            entity_id=str(i),
            request_id=f"b-{i}",
            commit=False,
        )
    session.commit()
    elapsed = time.perf_counter() - start
    rate = n / elapsed if elapsed > 0 else float("inf")
    assert rate >= 1000.0, f"audit write rate {rate:.1f}/s below 1000/s target"


def test_audit_query_endpoint(session_factory):
    db = session_factory()
    write_audit_event(
        db,
        action=AuditAction.CREATE,
        entity_type="keys",
        entity_id="k1",
        actor_id="u1",
        request_id="q-1",
    )
    db.close()

    from nightmarenet_server.audit.endpoints import build_audit_router

    app = FastAPI()
    router = build_audit_router()
    assert router is not None
    app.include_router(router)
    client = TestClient(app)
    resp = client.get("/api/v1/audit", params={"actor": "u1", "limit": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert body["items"][0]["actor_id"] == "u1"
    assert body["items"][0]["request_id"] == "q-1"
