"""End-to-end pipeline: data -> distortion -> training -> evaluation.

Orchestrates the full NightmareNet sleep-cycle workflow by running each
phase module (nightmarenet/phases/) in order against a shared
PipelineContext. Status tracking, progress percentages, and the on_event
callback all live here, since the phase modules deliberately don't touch
them (see phases/base.py).
"""

from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from nightmarenet.exceptions import PipelinePhaseError
from nightmarenet.phases.base import PipelineContext
from nightmarenet.phases.evaluate import EvaluatePhase
from nightmarenet.phases.export import ExportPhase
from nightmarenet.phases.ingest import IngestPhase
from nightmarenet.phases.optimize import OptimizePhase
from nightmarenet.phases.prepare import PreparePhase
from nightmarenet.phases.train import TrainPhase
from nightmarenet.training.callbacks import TrainingEvent
from nightmarenet.utils.config import load_config
from nightmarenet.utils.telemetry import setup_telemetry
from nightmarenet.utils.tracking import create_tracker_from_config
from nightmarenet.utils.webhooks import trigger_webhook

logger = logging.getLogger(__name__)


class PipelineStatus(str, enum.Enum):
    """Lifecycle status of a Pipeline run."""

    IDLE = "idle"
    INGESTING = "ingesting"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetrics:
    """Snapshot of live training metrics."""

    status: PipelineStatus = PipelineStatus.IDLE
    current_cycle: int = 0
    total_cycles: int = 0
    current_phase: str = ""
    phase_loss: float = 0.0
    progress_pct: float = 0.0
    eta_seconds: float = 0.0
    history: list = field(default_factory=list)
    error: Optional[str] = None
    baseline_results: Optional[dict] = None
    trained_results: Optional[dict] = None
    comparison: Optional[dict] = None
    report_md: Optional[str] = None
    adaption_quality: Optional[dict] = None
    quality_feedback: Optional[dict] = None
    per_cycle_metrics: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "current_cycle": self.current_cycle,
            "total_cycles": self.total_cycles,
            "current_phase": self.current_phase,
            "phase_loss": self.phase_loss,
            "progress_pct": round(self.progress_pct, 2),
            "eta_seconds": round(self.eta_seconds, 1),
            "history": self.history,
            "per_cycle_metrics": self.per_cycle_metrics,
            "error": self.error,
            "has_report": self.report_md is not None,
        }


class Pipeline:
    """Orchestrates the full NightmareNet pipeline.

    Usage::

        pipe = Pipeline(config)
        pipe.ingest(urls=["https://en.wikipedia.org/wiki/Machine_learning"])
        pipe.prepare()
        pipe.train()
        pipe.evaluate()
        pipe.export("results/my_model")

    Args:
        config: Full NightmareNet YAML configuration (dict).
        on_event: Optional callback ``fn(metrics_dict)`` called after every
                  phase and status change for live dashboards.
    """

    def __init__(
        self,
        config: dict,
        on_event: Optional[Callable[[dict], None]] = None,
        run_id: Optional[str] = None,
        distributed: Optional[str] = None,
        resume_dir: Optional[str] = None,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.config = config
        self.on_event = on_event
        self.distributed = distributed
        self.resume_dir = resume_dir
        self.metrics = PipelineMetrics()

        # Initialise OTel tracing + metrics (no-op if endpoint not configured)
        setup_telemetry(config)

        self.tracker = create_tracker_from_config(config)
        self.tracker.log_config(config)

        self._context = PipelineContext(
            run_id=self.run_id,
            config=config,
            tracker=self.tracker,
            distributed=distributed,
            resume_dir=resume_dir,
        )

    # ------------------------------------------------------------------
    # Private helpers (status/progress tracking - not the phases' job)
    # ------------------------------------------------------------------

    def _on_training_event(self, event: TrainingEvent) -> None:
        logger.debug("Training event: %s (%s)", event.event_type.value, event.phase)

    def _emit(self) -> None:
        if self.on_event is not None:
            try:
                self.on_event(self.metrics.to_dict())
            except Exception:
                logger.debug("on_event callback failed", exc_info=True)

    def _set_status(self, status: PipelineStatus) -> None:
        self.metrics.status = status
        self._emit()

    def _fail(self, error: str) -> None:
        self.metrics.status = PipelineStatus.FAILED
        self.metrics.error = error
        self._emit()

        trigger_webhook(
            self.config,
            "run_complete",
            "Pipeline run failed.",
            {
                "run_id": self.run_id,
                "status": "failed",
                "error": error,
                "model": self.config.get("model", {}).get("name", "unknown"),
            },
        )

    def _on_train_progress(self, event: dict) -> None:
        self.metrics.current_cycle = event.get("cycle", self.metrics.current_cycle)
        phase = event.get("phase", "")
        if phase:
            self.metrics.current_phase = phase
        avg_loss = event.get("avg_loss")
        if avg_loss is not None:
            self.metrics.phase_loss = float(avg_loss)
        pct = event.get("progress_pct")
        if pct is not None:
            # Training occupies 15-85% of overall pipeline progress
            self.metrics.progress_pct = 15.0 + (float(pct) * 0.7)
        history = event.get("history")
        if history is not None:
            self.metrics.history = history
        cycle_metrics = event.get("cycle_metrics")
        if cycle_metrics is not None:
            self.metrics.per_cycle_metrics.append(cycle_metrics)
        self._emit()

    def cancel(self) -> None:
        """Request graceful cancellation of a running pipeline."""
        self._context.cancelled = True
        self._set_status(PipelineStatus.CANCELLED)

    # ------------------------------------------------------------------
    # Stage 1: Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        *,
        urls: Optional[list[str]] = None,
        file_path: Optional[str] = None,
        text_content: Optional[str] = None,
        hf_dataset: Optional[str] = None,
        hf_subset: Optional[str] = None,
    ) -> None:
        """Load data from one of the supported sources.

        Exactly one of *urls*, *file_path*, *text_content*, or *hf_dataset*
        must be provided.
        """
        self._set_status(PipelineStatus.INGESTING)
        self.metrics.progress_pct = 2.0
        self.metrics.current_phase = "ingest"
        self._emit()

        try:
            result = IngestPhase(
                urls=urls,
                file_path=file_path,
                text_content=text_content,
                hf_dataset=hf_dataset,
                hf_subset=hf_subset,
            ).execute(self._context)
        except ValueError as exc:
            self._fail(f"Ingestion failed: {exc}")
            raise

        if not result.success:
            self._fail(f"Ingestion failed: {result.error}")
            raise PipelinePhaseError(phase="ingest", cycle=None, details=result.error)

        self.metrics.progress_pct = 8.0
        self._emit()

    # ------------------------------------------------------------------
    # Stage 1.5: Optimize (optional - Adaption Labs)
    # ------------------------------------------------------------------

    def optimize(self) -> None:
        """Phase-aware dataset optimization via Adaption Labs (optional, no-op if disabled)."""
        self.metrics.progress_pct = 8.0
        self.metrics.current_phase = "optimize"
        self._emit()

        result = OptimizePhase().execute(self._context)

        if not result.success:
            self._fail(f"Optimization failed: {result.error}")
            raise PipelinePhaseError(phase="optimize", cycle=None, details=result.error)

        self.metrics.adaption_quality = self._context.adaption_quality
        self.metrics.progress_pct = 15.0
        self._emit()

    # ------------------------------------------------------------------
    # Stage 2: Prepare
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Generate dream/nightmare splits and tokenise all data."""
        if self._context.dataset is None:
            raise RuntimeError("Call .ingest() before .prepare()")

        self._set_status(PipelineStatus.PREPARING)
        self.metrics.progress_pct = 10.0
        self.metrics.current_phase = "prepare"
        self._emit()

        result = PreparePhase(on_training_event=self._on_training_event).execute(self._context)

        if not result.success:
            self._fail(f"Preparation failed: {result.error}")
            raise PipelinePhaseError(phase="prepare", cycle=None, details=result.error)

        self.metrics.progress_pct = 15.0
        self._emit()

    # ------------------------------------------------------------------
    # Stage 3: Train
    # ------------------------------------------------------------------

    def train(self) -> list[dict]:
        """Run the full sleep-cycle training pipeline.

        Returns:
            Training history (list of phase result dicts).
        """
        if self._context.trainer is None:
            raise RuntimeError("Call .prepare() before .train()")
        if self._context.cancelled:
            return []

        num_cycles = self.config.get("training", {}).get("num_cycles", 3)
        self._set_status(PipelineStatus.TRAINING)
        self.metrics.total_cycles = num_cycles
        self.metrics.progress_pct = 15.0
        self._emit()

        result = TrainPhase(on_progress=self._on_train_progress).execute(self._context)

        if not result.success:
            self._fail(f"Training failed: {result.error}")
            raise PipelinePhaseError(
                phase="train", cycle=self.metrics.current_cycle, details=result.error
            )

        history = result.data.get("history", [])
        if history:
            for record in history:
                self.tracker.log_phase(
                    cycle=record.get("cycle", 0),
                    phase=record.get("phase", "unknown"),
                    metrics=record,
                )
            last = history[-1]
            self.metrics.current_cycle = last.get("cycle", 0)
            self.metrics.current_phase = last.get("phase", "")
            self.metrics.phase_loss = last.get("avg_loss", 0.0)

        self.metrics.progress_pct = 85.0
        self.metrics.eta_seconds = 0.0
        self._emit()
        logger.info("Training complete.")
        return history

    # ------------------------------------------------------------------
    # Stage 4: Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Run baseline vs trained model evaluation and generate a report.

        Returns:
            Comparison dict with all metric deltas.
        """
        if self._context.trainer is None:
            raise RuntimeError("Call .train() before .evaluate()")

        self._set_status(PipelineStatus.EVALUATING)
        self.metrics.progress_pct = 88.0
        self.metrics.current_phase = "evaluate"
        self._emit()

        result = EvaluatePhase().execute(self._context)

        if not result.success:
            self._fail(f"Ingestion failed: {result.error}")
            raise PipelinePhaseError(phase="evaluate", cycle=None, details=result.error)

        self.metrics.trained_results = self._context.trained_results
        self.metrics.baseline_results = self._context.baseline_results
        self.metrics.comparison = self._context.comparison
        self.metrics.report_md = self._context.report_md
        self.metrics.quality_feedback = self._context.quality_feedback

        return self._context.comparison or {}

    def _compute_quality_feedback(self, comparison: dict) -> None:
        """Correlate Adaption quality with robustness improvement."""
        if not self.metrics.adaption_quality:
            return

        robustness_delta = comparison.get("robustness_delta")
        if robustness_delta is None:
            for key in ("robustness", "avg_robustness", "mean_robustness"):
                if key in comparison:
                    robustness_delta = comparison[key]
                    break

        feedback: dict = {
            "adaption_phases_optimized": list(self.metrics.adaption_quality.keys()),
            "robustness_delta": robustness_delta,
        }

        target_improvement = 0.10
        if robustness_delta is not None and robustness_delta < target_improvement:
            feedback["suggestions"] = [
                "Increase nightmare blueprint aggressiveness",
                "Enable reasoning_traces for stronger training signal",
                "Increase max_rows for more diverse training data",
            ]
        elif robustness_delta is not None:
            feedback["suggestions"] = []
            feedback["status"] = "on_target"

        self.metrics.quality_feedback = feedback

    # ------------------------------------------------------------------
    # Stage 5: Export
    # ------------------------------------------------------------------

    def export(self, output_dir: str) -> str:
        """Save the trained model, tokenizer, and report to disk.

        Args:
            output_dir: Directory to save artifacts.

        Returns:
            Path to the saved model directory.
        """
        result = ExportPhase(output_dir).execute(self._context)

        if not result.success:
            raise RuntimeError(f"Export failed: {result.error}")

        return result.data["output_dir"]

    # ------------------------------------------------------------------
    # Convenience: run all stages
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        urls: Optional[list[str]] = None,
        file_path: Optional[str] = None,
        text_content: Optional[str] = None,
        hf_dataset: Optional[str] = None,
        hf_subset: Optional[str] = None,
        export_dir: Optional[str] = None,
    ) -> dict:
        """Execute the full pipeline end-to-end.

        Returns:
            The evaluation comparison dict.
        """
        self.ingest(
            urls=urls,
            file_path=file_path,
            text_content=text_content,
            hf_dataset=hf_dataset,
            hf_subset=hf_subset,
        )

        self.optimize()
        self.prepare()
        self.train()
        comparison = self.evaluate()

        if export_dir:
            self.export(export_dir)

        return comparison


def create_pipeline_from_config(
    config_path: str = "configs/default.yaml",
    on_event: Optional[Callable[[dict], None]] = None,
) -> Pipeline:
    """Create a Pipeline from a YAML config file.

    Args:
        config_path: Path to the YAML configuration.
        on_event: Optional event callback for live dashboards.

    Returns:
        Configured Pipeline instance.
    """
    config = load_config(config_path)
    return Pipeline(config=config, on_event=on_event)
