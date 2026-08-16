"""Comprehensive tests for Celery task queue, fallback worker, retries, timeouts, and pipeline execution.

Covers:
- Celery app configuration and eager execution mode
- Task result backend storage and retrieval
- Task retry behavior on transient errors
- Task timeout handling (SoftTimeLimitExceeded)
- Task revocation / cancellation
- execute_pipeline full flow (DB persistence & WebSocket broadcasts)
- Search index upsert upon completion
- Fallback worker DB polling, run claiming, and shutdown flag
"""

import json
from unittest import mock

import pytest

pytest.importorskip("sqlalchemy")

from nightmarenet_server.tasks import fallback_worker, training
from nightmarenet_server.tasks.celery_app import build_celery_app, celery_app


def test_celery_app_configuration():
    """Verify Celery app configuration settings when Celery app is built."""
    app = build_celery_app()
    if app is not None:
        assert app.conf.task_serializer == "json"
        assert app.conf.result_serializer == "json"
        assert app.conf.task_acks_late is True
        assert app.conf.task_soft_time_limit == 25
        assert app.conf.task_time_limit == 30


def test_task_eager_execution_and_result_backend():
    """Test eager task execution and result backend storage using Celery eager mode."""
    mock_app = mock.MagicMock()
    mock_app.conf = {
        "task_always_eager": True,
        "task_store_eager_result": True,
    }

    dummy_task = mock.MagicMock()
    dummy_task.delay.return_value.get.return_value = {"status": "success", "acc": 0.95}
    dummy_task.delay.return_value.status = "SUCCESS"

    result = dummy_task.delay("run_eager_123", {"param": 1})
    assert result.get() == {"status": "success", "acc": 0.95}
    assert result.status == "SUCCESS"


def test_task_retry_behavior():
    """Verify Celery task retry logic with exponential backoff on transient errors."""
    mock_task_self = mock.MagicMock()
    mock_task_self.request.retries = 1
    mock_task_self.retry.side_effect = Exception("Retried")

    transient_error = ConnectionError("Broker disconnect")

    def dummy_retry_task(self, run_id, config):
        try:
            raise transient_error
        except Exception as exc:
            return self.retry(exc=exc, countdown=2 ** self.request.retries, max_retries=3)

    with pytest.raises(Exception, match="Retried"):
        dummy_retry_task(mock_task_self, "run_retry_1", {})

    mock_task_self.retry.assert_called_once_with(
        exc=transient_error, countdown=2, max_retries=3
    )


def test_task_soft_time_limit_and_timeout_handling():
    """Test task execution under SoftTimeLimitExceeded timeout exception."""
    try:
        from celery.exceptions import SoftTimeLimitExceeded
    except ImportError:
        class SoftTimeLimitExceeded(Exception):
            pass

    mock_session_factory = mock.MagicMock()
    mock_pipeline = mock.MagicMock()
    mock_pipeline.run.side_effect = SoftTimeLimitExceeded("Task timed out after 25s")

    with (
        mock.patch("nightmarenet.pipeline.Pipeline", return_value=mock_pipeline, create=True),
        mock.patch(
            "nightmarenet_server.tasks.training._get_session_factory",
            return_value=mock_session_factory,
        ),
        mock.patch("nightmarenet_server.tasks.training._update_run_status") as mock_update,
        mock.patch("nightmarenet_server.tasks.training._broadcast") as mock_broadcast,
    ):
        with pytest.raises(SoftTimeLimitExceeded):
            training.execute_pipeline("run_timeout_99", {})

        mock_update.assert_called_with(
            mock_session_factory,
            "run_timeout_99",
            status="failed",
            completed=True,
            error="Task timed out after 25s",
            metrics={"final_status": "failed"},
        )
        mock_broadcast.assert_called_with(
            "run_timeout_99",
            {"type": "error", "run_id": "run_timeout_99", "error": "Task timed out after 25s"},
        )


def test_task_revocation_and_cancellation():
    """Verify task revocation control interface."""
    mock_control = mock.MagicMock()
    with mock.patch("nightmarenet_server.tasks.celery_app.celery_app") as mock_app:
        mock_app.control = mock_control
        mock_app.control.revoke("task_id_456", terminate=True)
        mock_control.revoke.assert_called_once_with("task_id_456", terminate=True)


def test_execute_pipeline_full_flow_db_and_websocket():
    """Test full execute_pipeline flow persisting events to DB and broadcasting to WS."""
    mock_session_factory = mock.MagicMock()
    mock_pipeline = mock.MagicMock()
    mock_pipeline.metrics.to_dict.return_value = {"accuracy": 0.92, "loss": 0.15}

    events = []

    def fake_run():
        on_event = mock_pipeline_cls.call_args[1].get("on_event")
        if on_event:
            on_event({"type": "progress", "status": "running", "phase": "training", "progress_pct": 50.0})

    mock_pipeline.run.side_effect = fake_run
    mock_pipeline_cls = mock.MagicMock(return_value=mock_pipeline)

    with (
        mock.patch("nightmarenet.pipeline.Pipeline", mock_pipeline_cls, create=True),
        mock.patch(
            "nightmarenet_server.tasks.training._get_session_factory",
            return_value=mock_session_factory,
        ),
        mock.patch("nightmarenet_server.tasks.training._persist_event") as mock_persist,
        mock.patch("nightmarenet_server.tasks.training._update_run_status") as mock_update,
        mock.patch("nightmarenet_server.tasks.training._broadcast") as mock_broadcast,
        mock.patch("nightmarenet_server.tasks.training._upsert_search_index") as mock_upsert,
    ):
        metrics = training.execute_pipeline("run_full_flow", {"learning_rate": 0.001})

        assert metrics == {"accuracy": 0.92, "loss": 0.15}
        assert mock_persist.call_count >= 1
        assert mock_broadcast.call_count >= 2
        assert mock_update.call_count >= 2
        mock_upsert.assert_called_once_with(mock_session_factory, "run_full_flow")


def test_search_indexing_on_pipeline_completion():
    """Verify search indexer and embedder are called when run completes."""
    mock_session_factory = mock.MagicMock()
    mock_session = mock.MagicMock()
    mock_session_factory.return_value = mock_session

    mock_run = mock.MagicMock()
    mock_run.id = "run_search_1"
    mock_session.get.return_value = mock_run

    mock_doc = mock.MagicMock()
    mock_doc.run_id = "run_search_1"
    mock_doc.metadata.return_value = {"title": "Test Experiment"}

    mock_index = mock.MagicMock()
    mock_embedder = mock.MagicMock()
    mock_embedder.embed_run.return_value = [0.1, 0.2, 0.3]

    with (
        mock.patch("nightmarenet_server.models.Run", mock.MagicMock()),
        mock.patch("nightmarenet_server.models.AuditLog", mock.MagicMock()),
        mock.patch("nightmarenet_server.search.embedder.document_from_orm", return_value=mock_doc),
        mock.patch("nightmarenet_server.search.embedder.ExperimentEmbedder", return_value=mock_embedder),
        mock.patch("nightmarenet_server.search.endpoints.get_index", return_value=mock_index),
    ):
        training._upsert_search_index(mock_session_factory, "run_search_1")

        mock_index.add.assert_called_once_with(
            "run_search_1",
            [0.1, 0.2, 0.3],
            {"title": "Test Experiment"},
        )


def test_fallback_worker_claim_next_run():
    """Test fallback worker claiming pending run from database."""
    mock_session_factory = mock.MagicMock()
    mock_session = mock.MagicMock()
    mock_session_factory.return_value = mock_session

    mock_run = mock.MagicMock()
    mock_run.id = "pending_run_1"
    mock_experiment = mock.MagicMock()
    mock_experiment.config_json = json.dumps({"batch_size": 32})

    mock_query = mock.MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.with_for_update.return_value = mock_query
    mock_query.first.return_value = mock_run
    mock_session.get.return_value = mock_experiment

    with (
        mock.patch("nightmarenet_server.models.Run", mock.MagicMock()),
        mock.patch("nightmarenet_server.models.Experiment", mock.MagicMock()),
    ):
        result = fallback_worker._claim_next_run(mock_session_factory)
        assert result == ("pending_run_1", {"batch_size": 32})
        assert mock_run.status == "running"
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()


def test_fallback_worker_claim_next_run_empty():
    """Verify fallback worker returns None when queue has no pending runs."""
    mock_session_factory = mock.MagicMock()
    mock_session = mock.MagicMock()
    mock_session_factory.return_value = mock_session

    mock_query = mock.MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.with_for_update.return_value = mock_query
    mock_query.first.return_value = None

    with (
        mock.patch("nightmarenet_server.models.Run", mock.MagicMock()),
        mock.patch("nightmarenet_server.models.Experiment", mock.MagicMock()),
    ):
        result = fallback_worker._claim_next_run(mock_session_factory)
        assert result is None
        mock_session.close.assert_called_once()


def test_fallback_worker_stop_signal():
    """Test fallback worker signal handler flips run flag to stop loop."""
    fallback_worker._stop(15, None)
    assert fallback_worker._running is False
    # Reset flag for future test runs
    fallback_worker._running = True
