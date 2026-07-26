# Mock Celery before importing tasks
import sys
from unittest import mock

import pytest

pytest.importorskip("sqlalchemy")

mock_celery = mock.MagicMock()
sys.modules['celery'] = mock_celery

from nightmarenet_server.tasks import training  # noqa: E402


def test_execute_pipeline_success():
    config = {"some": "config"}
    mock_pipeline = mock.MagicMock()
    mock_pipeline_cls = mock.MagicMock(return_value=mock_pipeline)
    mock_pipeline.metrics.to_dict.return_value = {"acc": 0.9}

    mock_session_factory = mock.MagicMock()

    with mock.patch("nightmarenet.pipeline.Pipeline", mock_pipeline_cls, create=True), \
         mock.patch(
             "nightmarenet_server.tasks.training._get_session_factory",
             return_value=mock_session_factory
         ), \
         mock.patch("nightmarenet_server.tasks.training._persist_event"), \
         mock.patch("nightmarenet_server.tasks.training._update_run_status") as mock_update, \
         mock.patch("nightmarenet_server.tasks.training._broadcast"), \
         mock.patch("nightmarenet_server.tasks.training._upsert_search_index") as mock_upsert:

        metrics = training.execute_pipeline("run_123", config)

        assert metrics == {"acc": 0.9}
        mock_pipeline.run.assert_called_once()
        mock_upsert.assert_called_once_with(mock_session_factory, "run_123")
        assert mock_update.call_count >= 2

def test_execute_pipeline_failure():
    config = {"some": "config"}
    mock_pipeline = mock.MagicMock()
    mock_pipeline_cls = mock.MagicMock(return_value=mock_pipeline)
    mock_pipeline.run.side_effect = ValueError("Pipeline error")

    mock_session_factory = mock.MagicMock()

    with mock.patch("nightmarenet.pipeline.Pipeline", mock_pipeline_cls, create=True), \
         mock.patch(
             "nightmarenet_server.tasks.training._get_session_factory",
             return_value=mock_session_factory
         ), \
         mock.patch("nightmarenet_server.tasks.training._persist_event"), \
         mock.patch("nightmarenet_server.tasks.training._update_run_status"), \
         mock.patch("nightmarenet_server.tasks.training._broadcast") as mock_broadcast, \
         mock.patch("nightmarenet_server.tasks.training._upsert_search_index"):

        with pytest.raises(ValueError, match="Pipeline error"):
            training.execute_pipeline("run_123", config)

        mock_broadcast.assert_called_with("run_123", {"type": "error", "run_id": "run_123", "error": "Pipeline error"})  # noqa: E501

def test_persist_event_no_models():
    mock_session_factory = mock.MagicMock()
    with mock.patch.dict('sys.modules', {'nightmarenet_server.models': None}):
        training._persist_event(mock_session_factory, "run_1", "test", {})
        mock_session_factory.assert_not_called()

def test_update_run_status_no_models():
    mock_session_factory = mock.MagicMock()
    with mock.patch.dict('sys.modules', {'nightmarenet_server.models': None}):
        training._update_run_status(mock_session_factory, "run_1", "running")
        mock_session_factory.assert_not_called()

def test_run_pipeline_task_defined():
    # Verify the task wrapper is defined
    assert callable(training.run_pipeline_task)
    assert hasattr(training.run_pipeline_task, "name") or isinstance(training.run_pipeline_task, mock.MagicMock)  # noqa: E501

def test_celery_app_mocked():
    # Verify our celery mock is in place for the test suite
    from nightmarenet_server.tasks import celery_app
    assert celery_app is not None
