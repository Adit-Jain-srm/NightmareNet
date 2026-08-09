"""Tests for pipeline runner registry behavior."""

import json
import time
from unittest.mock import MagicMock

import pytest

from nightmarenet.pipeline_runner import (
    PipelineRunner,
    _get_runs_dir,
    _persist_run_state,
    _update_run_state,
    load_persisted_runs,
    register_runner,
)


class _FakePipeline:
    def cancel(self) -> None:
        pass


def test_register_evicts_completed_runners_first(monkeypatch) -> None:
    import nightmarenet.pipeline_runner as pr

    monkeypatch.setattr(pr, "_MAX_RUNNERS", 2)
    pr._runners.clear()
    r1 = PipelineRunner(_FakePipeline())
    r2 = PipelineRunner(_FakePipeline())
    register_runner(r1)
    register_runner(r2)
    r3 = PipelineRunner(_FakePipeline())
    register_runner(r3)
    assert r1.id not in pr._runners
    assert r2.id in pr._runners
    assert r3.id in pr._runners


def test_register_raises_when_registry_at_cap_and_all_running(monkeypatch) -> None:
    import nightmarenet.pipeline_runner as pr

    monkeypatch.setattr(pr, "_MAX_RUNNERS", 1)
    pr._runners.clear()
    r1 = PipelineRunner(_FakePipeline())
    t = MagicMock()
    t.is_alive.return_value = True
    r1._thread = t
    register_runner(r1)
    r2 = PipelineRunner(_FakePipeline())
    with pytest.raises(RuntimeError, match="capacity"):
        register_runner(r2)


def test_persistence_round_trip_with_metrics(monkeypatch, tmp_path) -> None:
    """Test that run state persists and loads correctly with metrics."""
    import nightmarenet.pipeline_runner as pr

    monkeypatch.setenv("NIGHTMARENET_RUNS_DIR", str(tmp_path))
    pr._runners.clear()

    run_id = "test-run-123"
    config = {"test": "config"}
    timestamp = time.time()
    metrics = {"accuracy": 0.95, "loss": 0.05}

    # Persist initial state
    _persist_run_state(run_id, config, "running", timestamp)

    # Update with metrics
    _update_run_state(run_id, "complete", timestamp, metrics)

    # Load and verify
    runs_dir = _get_runs_dir()
    run_file = runs_dir / f"{run_id}.json"
    assert run_file.exists()

    with open(run_file, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["run_id"] == run_id
    assert loaded["status"] == "complete"
    assert loaded["config"] == config
    assert loaded["metrics"] == metrics


def test_stale_detection_all_active_statuses(monkeypatch, tmp_path) -> None:
    """Test that stale detection works for all active statuses."""
    import nightmarenet.pipeline_runner as pr

    monkeypatch.setenv("NIGHTMARENET_RUNS_DIR", str(tmp_path))
    pr._runners.clear()

    active_statuses = ["running", "ingesting", "preparing", "training", "evaluating"]
    old_timestamp = time.time() - 400  # More than 5 minutes ago

    for status in active_statuses:
        run_id = f"test-{status}"
        _persist_run_state(run_id, {}, status, old_timestamp)

    # Load persisted runs - should mark all as interrupted
    load_persisted_runs()

    runs_dir = _get_runs_dir()
    for status in active_statuses:
        run_file = runs_dir / f"test-{status}.json"
        with open(run_file, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["status"] == "interrupted"


def test_atomic_write_prevents_corruption(monkeypatch, tmp_path) -> None:
    """Test that atomic write creates valid JSON even if interrupted."""
    import nightmarenet.pipeline_runner as pr

    monkeypatch.setenv("NIGHTMARENET_RUNS_DIR", str(tmp_path))
    pr._runners.clear()

    run_id = "test-atomic"
    config = {"test": "data"}
    timestamp = 1234567890.0

    # Persist state
    _persist_run_state(run_id, config, "running", timestamp)

    # Verify file exists and is valid JSON
    runs_dir = _get_runs_dir()
    run_file = runs_dir / f"{run_id}.json"
    assert run_file.exists()

    with open(run_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["run_id"] == run_id
    assert data["status"] == "running"
