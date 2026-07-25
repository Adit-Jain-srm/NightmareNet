"""Celery task wrapping :class:`nightmarenet.pipeline.Pipeline`.

The task streams pipeline events to two sinks:

1. The Postgres ``run_events`` table — one row per event, durable history.
2. The in-memory realtime broker (see
   :mod:`nightmarenet_server.realtime.websocket`) — broadcasts to any
   subscribed WebSocket client for live UI updates.

When Celery is not installed, ``run_pipeline_task`` degrades to a plain
synchronous function so the rest of the platform (and the existing 522+-test
suite) continues to work unmodified.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from nightmarenet_server.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _persist_event(
    session_factory: Any,
    run_id: str,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    """Insert a single ``run_events`` row."""
    try:
        from nightmarenet_server.models import RunEvent
    except ImportError:
        return

    session = session_factory()
    try:
        row = RunEvent(
            run_id=run_id,
            event_type=event_type,
            payload_json=json.dumps(payload, default=str),
        )
        session.add(row)
        session.commit()
    except Exception:
        logger.exception("Failed to persist run event for run %s", run_id)
        session.rollback()
    finally:
        session.close()


def _update_run_status(
    session_factory: Any,
    run_id: str,
    status: str,
    phase: str = "",
    progress_pct: float = 0.0,
    metrics: Optional[Dict[str, Any]] = None,
    completed: bool = False,
    error: Optional[str] = None,
) -> None:
    """Update a single ``runs`` row with latest progress."""
    try:
        from nightmarenet_server.models import Run
    except ImportError:
        return

    session = session_factory()
    try:
        run = session.get(Run, run_id)
        if run is None:
            logger.warning("Run %s missing — cannot update status.", run_id)
            return
        run.status = status
        run.phase = phase or run.phase
        run.progress_pct = progress_pct
        if metrics is not None:
            run.metrics_json = json.dumps(
                {**metrics, **({"error": error} if error else {})},
                default=str,
            )
        if completed and run.completed_at is None:
            run.completed_at = datetime.now(timezone.utc)
        session.commit()
    except Exception:
        logger.exception("Failed to update run %s", run_id)
        session.rollback()
    finally:
        session.close()


def _upsert_search_index(session_factory: Any, run_id: str) -> None:
    """Best-effort semantic indexing for completed runs."""
    try:
        from nightmarenet_server.models import AuditLog, Run
        from nightmarenet_server.search.embedder import ExperimentEmbedder, document_from_orm
        from nightmarenet_server.search.endpoints import get_index
    except ImportError:
        return

    session = session_factory()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return
        experiment = getattr(run, "experiment", None)
        audit_logs = []
        if experiment is not None:
            audit_logs = (
                session.query(AuditLog)
                .filter(AuditLog.resource_id.in_([run.id, experiment.id]))
                .all()
            )
        doc = document_from_orm(run, experiment=experiment, audit_logs=audit_logs)
        get_index().add(doc.run_id, ExperimentEmbedder().embed_run(doc), doc.metadata())
    except Exception:
        logger.exception("Failed to update search index for run %s", run_id)
    finally:
        session.close()


def _get_session_factory() -> Any:
    """Build a session factory using the configured DB URL."""
    from nightmarenet_server.models.base import (
        DEFAULT_DATABASE_URL,
        get_session_factory,
    )

    db_url = os.environ.get("NIGHTMARENET_DATABASE_URL", DEFAULT_DATABASE_URL)
    return get_session_factory(db_url)


def _broadcast(run_id: str, event: Dict[str, Any]) -> None:
    """Push an event into the realtime broker, if available."""
    try:
        from nightmarenet_server.realtime.websocket import publish_event

        publish_event(run_id, event)
    except Exception:
        logger.debug("Realtime broker unavailable", exc_info=True)


def execute_pipeline(run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a NightmareNet pipeline synchronously for a single run.

    Designed to be called either directly (in-process fallback) or from a
    Celery worker. Writes durable events to Postgres and broadcasts the same
    payloads through the realtime broker.
    """
    from nightmarenet.pipeline import Pipeline

    session_factory = _get_session_factory()

    def on_event(event: Dict[str, Any]) -> None:
        enriched = {
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        _persist_event(
            session_factory,
            run_id,
            event.get("type") or event.get("status") or "progress",
            enriched,
        )
        _broadcast(run_id, enriched)
        _update_run_status(
            session_factory,
            run_id,
            status=str(event.get("status", "running")),
            phase=str(event.get("phase", "")),
            progress_pct=float(event.get("progress_pct", 0.0)),
        )

    _update_run_status(
        session_factory,
        run_id,
        status="running",
        phase="ingesting",
        progress_pct=0.0,
    )

    pipeline = Pipeline(config=config, on_event=on_event)
    try:
        pipeline.run()
    except Exception as exc:
        logger.exception("Pipeline failed for run %s", run_id)
        _update_run_status(
            session_factory,
            run_id,
            status="failed",
            completed=True,
            error=str(exc),
            metrics={"final_status": "failed"},
        )
        _broadcast(
            run_id,
            {"type": "error", "run_id": run_id, "error": str(exc)},
        )
        raise

    metrics = pipeline.metrics.to_dict()
    _update_run_status(
        session_factory,
        run_id,
        status="completed",
        phase="complete",
        progress_pct=100.0,
        completed=True,
        metrics=metrics,
    )
    _broadcast(
        run_id,
        {"type": "completed", "run_id": run_id, "metrics": metrics},
    )
    _upsert_search_index(session_factory, run_id)
    return metrics


if celery_app is not None:

    @celery_app.task(
        name="nightmarenet.run_pipeline",
        bind=True,
        autoretry_for=(),
        max_retries=0,
        track_started=True,
    )
    def run_pipeline_task(self: Any, run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Celery task entrypoint — delegates to :func:`execute_pipeline`."""
        logger.info("Celery task starting for run %s (task_id=%s)", run_id, self.request.id)
        return execute_pipeline(run_id, config)

else:

    def run_pipeline_task(run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[misc]
        """Synchronous fallback used when Celery is not installed."""
        logger.info("Celery not installed — executing run %s inline.", run_id)
        return execute_pipeline(run_id, config)
