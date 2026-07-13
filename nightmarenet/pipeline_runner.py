"""Background runner for the NightmareNet pipeline.

Executes a ``Pipeline`` in a background thread with event streaming
for WebSocket / SSE integration.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from nightmarenet.pipeline import Pipeline

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs a Pipeline in a daemon thread, streaming events via callback.

    Usage::

        runner = PipelineRunner(pipeline)
        runner.start(urls=["https://..."])
        runner.status()  # -> dict
        runner.cancel()

    Args:
        pipeline: Configured Pipeline instance.
        on_event: Optional event callback ``fn(event_dict)`` for WebSocket.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.pipeline = pipeline
        self.pipeline.run_id = self.id
        self.on_event = on_event
        self._thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

        # Wire event callback into the pipeline
        if on_event:
            self.pipeline.on_event = on_event

    def start(self, **ingest_kwargs: Any) -> str:
        """Launch the pipeline in a background thread.

        Accepts the same keyword arguments as ``Pipeline.ingest()``.

        Returns:
            The run ID.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Pipeline is already running.")

        self._cancel_event.clear()
        self._start_time = time.time()
        self._last_heartbeat = self._start_time

        # Persist initial run state
        _persist_run_state(self.id, self.pipeline.config, "running", self._start_time)

        def _run() -> None:
            try:
                self._last_heartbeat = time.time()
                _update_run_state(self.id, "ingesting", self._last_heartbeat)
                self.pipeline.ingest(**ingest_kwargs)
                if self._cancel_event.is_set():
                    _update_run_state(self.id, "cancelled", time.time())
                    return
                self._last_heartbeat = time.time()
                _update_run_state(self.id, "preparing", self._last_heartbeat)
                self.pipeline.prepare()
                if self._cancel_event.is_set():
                    _update_run_state(self.id, "cancelled", time.time())
                    return
                self._last_heartbeat = time.time()
                _update_run_state(self.id, "training", self._last_heartbeat)
                self.pipeline.train()
                if self._cancel_event.is_set():
                    _update_run_state(self.id, "cancelled", time.time())
                    return
                self._last_heartbeat = time.time()
                _update_run_state(self.id, "evaluating", self._last_heartbeat)
                self.pipeline.evaluate()
                _update_run_state(self.id, "complete", time.time(), self.pipeline.metrics.to_dict())
            except Exception:
                logger.exception("Pipeline run %s failed", self.id)
                _update_run_state(self.id, "failed", time.time(), self.pipeline.metrics.to_dict())

        self._thread = threading.Thread(target=_run, daemon=True, name=f"pipeline-{self.id}")
        self._thread.start()
        logger.info("Pipeline %s started.", self.id)
        return self.id

    def cancel(self) -> None:
        """Request cancellation of a running pipeline."""
        self._cancel_event.set()
        self.pipeline.cancel()
        logger.info("Pipeline %s cancellation requested.", self.id)

    def status(self) -> dict:
        """Return the current pipeline metrics as a dict."""
        data = self.pipeline.metrics.to_dict()
        data["run_id"] = self.id
        data["is_running"] = self._thread is not None and self._thread.is_alive()
        data["start_time"] = getattr(self, "_start_time", None)
        data["last_heartbeat"] = getattr(self, "_last_heartbeat", None)
        return data

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ------------------------------------------------------------------
# Global runner registry (for multi-pipeline API support)
# ------------------------------------------------------------------

_runners: dict[str, PipelineRunner] = {}
_MAX_RUNNERS = int(os.environ.get("NIGHTMARENET_MAX_PIPELINE_RUNNERS", "64"))


def register_runner(runner: PipelineRunner) -> str:
    """Register a runner and return its ID.

    Evicts completed (not running) entries when the registry would exceed
    :envvar:`NIGHTMARENET_MAX_PIPELINE_RUNNERS` (default 64). If the cap is
    reached and every registered run is still active, raises ``RuntimeError``.
    Also evicts runs older than 30 days from disk periodically.
    """
    # Periodically evict old runs from disk
    evict_old_runs()
    while len(_runners) >= _MAX_RUNNERS:
        for rid, r in list(_runners.items()):
            if not r.is_running:
                del _runners[rid]
                logger.info("Evicted completed pipeline runner %s (registry cap).", rid)
                break
        else:
            msg = (
                f"Pipeline runner registry at capacity ({_MAX_RUNNERS}) and all "
                "registered runs are still active"
            )
            raise RuntimeError(msg) from None
    _runners[runner.id] = runner
    return runner.id


def get_runner(run_id: str) -> Optional[PipelineRunner]:
    """Retrieve a runner by ID."""
    return _runners.get(run_id)


def list_runners() -> list[dict]:
    """Return status of all registered runners."""
    return [r.status() for r in _runners.values()]


def list_all_runs(include_historical: bool = True) -> list[dict]:
    """Return status of all runs, including historical ones from disk.

    Args:
        include_historical: If True, include completed/failed runs from disk.

    Returns:
        List of run status dicts.
    """
    runs = [r.status() for r in _runners.values()]

    if include_historical:
        runs_dir = _get_runs_dir()
        if runs_dir.exists():
            for run_file in runs_dir.glob("*.json"):
                run_id = run_file.stem
                # Skip if already in active registry
                if run_id in _runners:
                    continue
                try:
                    with open(run_file, encoding="utf-8") as f:
                        data = json.load(f)
                    # Add is_running=False for historical runs
                    data["is_running"] = False
                    runs.append(data)
                except Exception:
                    logger.debug("Failed to load run state from %s", run_file)

    return runs


def _atomic_write_json(file_path: Path, data: dict) -> None:
    """Atomically write JSON data to a file.

    Uses a temporary file and os.replace to ensure atomic writes,
    preventing corruption if the process crashes mid-write.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(file_path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(file_path))
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


# ------------------------------------------------------------------
# File-based persistence
# ------------------------------------------------------------------

_RUNS_DIR_ENV = "NIGHTMARENET_RUNS_DIR"
_DEFAULT_RUNS_DIR = "runs"


def _get_runs_dir() -> Path:
    """Get the directory for storing run state files.

    Uses NIGHTMARENET_RUNS_DIR environment variable if set, otherwise
    defaults to './runs' relative to the current working directory.
    In production, set NIGHTMARENET_RUNS_DIR to a persistent location
    (e.g., the same directory as tracking.output_dir).
    """
    runs_dir = Path(os.environ.get(_RUNS_DIR_ENV, _DEFAULT_RUNS_DIR))
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _persist_run_state(run_id: str, config: dict, status: str, timestamp: float) -> None:
    """Persist initial run state to disk."""
    runs_dir = _get_runs_dir()
    run_file = runs_dir / f"{run_id}.json"

    state = {
        "run_id": run_id,
        "status": status,
        "config": config,
        "start_time": timestamp,
        "last_heartbeat": timestamp,
        "metrics": {},
    }

    _atomic_write_json(run_file, state)
    logger.debug("Persisted run state for %s to %s", run_id, run_file)


def _update_run_state(
    run_id: str,
    status: str,
    timestamp: float,
    metrics: Optional[dict] = None,
) -> None:
    """Update run state on disk."""
    runs_dir = _get_runs_dir()
    run_file = runs_dir / f"{run_id}.json"

    if not run_file.exists():
        logger.warning("Run state file %s not found for update", run_file)
        return

    try:
        with open(run_file, encoding="utf-8") as f:
            state = json.load(f)

        state["status"] = status
        state["last_heartbeat"] = timestamp
        if metrics:
            state["metrics"] = metrics

        _atomic_write_json(run_file, state)
    except Exception:
        logger.debug("Failed to update run state for %s", run_id)


def load_persisted_runs() -> None:
    """Load persisted runs from disk and recover state.

    Marks stale 'running' entries as 'interrupted'.
    Evicts runs older than 30 days.
    """
    runs_dir = _get_runs_dir()
    if not runs_dir.exists():
        return

    now = time.time()
    stale_threshold = 300  # 5 minutes in seconds
    age_threshold = 30 * 24 * 3600  # 30 days in seconds

    for run_file in runs_dir.glob("*.json"):
        try:
            with open(run_file, encoding="utf-8") as f:
                state = json.load(f)

            run_id = state.get("run_id")
            status = state.get("status")
            start_time = state.get("start_time", 0)
            last_heartbeat = state.get("last_heartbeat", 0)

            # Evict old runs
            if now - start_time > age_threshold:
                run_file.unlink()
                logger.info("Evicted old run %s (age > 30 days)", run_id)
                continue

            # Detect stale running runs
            active_statuses = {"running", "ingesting", "preparing", "training", "evaluating"}
            if status in active_statuses and (now - last_heartbeat > stale_threshold):
                state["status"] = "interrupted"
                _atomic_write_json(run_file, state)
                logger.info("Marked stale run %s as interrupted", run_id)

        except Exception:
            logger.debug("Failed to load run state from %s", run_file)

    logger.info("Loaded persisted runs from %s", runs_dir)


def evict_old_runs() -> None:
    """Evict runs older than 30 days from disk."""
    runs_dir = _get_runs_dir()
    if not runs_dir.exists():
        return

    now = time.time()
    age_threshold = 30 * 24 * 3600  # 30 days in seconds

    for run_file in runs_dir.glob("*.json"):
        try:
            with open(run_file, encoding="utf-8") as f:
                state = json.load(f)

            start_time = state.get("start_time", 0)
            if now - start_time > age_threshold:
                run_file.unlink()
                logger.info("Evicted old run %s (age > 30 days)", state.get("run_id"))
        except Exception:
            logger.debug("Failed to evict run %s", run_file)
