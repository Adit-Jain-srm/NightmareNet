"""End-to-end pipeline: data → distortion → training → evaluation.

Orchestrates the full NightmareNet sleep-cycle workflow as a single
unit of work, with status tracking and optional callbacks for live
metric streaming.
"""

from __future__ import annotations

import copy
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nightmarenet.data.generator import create_generators_from_config
from nightmarenet.data.ingest import DataIngestor
from nightmarenet.distortions.text import apply_text_distortions
from nightmarenet.evaluation.evaluator import Evaluator
from nightmarenet.evaluation.metrics import evaluate_cycle, quick_robustness_score
from nightmarenet.training.callbacks import CallbackManager, TrainingEvent
from nightmarenet.training.trainer import Trainer, _tokenize_dataset
from nightmarenet.utils.config import load_config
from nightmarenet.utils.telemetry import record_metric, setup_telemetry, trace_phase
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
    history: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    baseline_results: Optional[dict] = None
    trained_results: Optional[dict] = None
    comparison: Optional[dict] = None
    report_md: Optional[str] = None
    adaption_quality: Optional[dict] = None
    quality_feedback: Optional[dict] = None
    per_cycle_metrics: list[dict] = field(
        default_factory=list
    )  # <-- add this, remove the duplicate `history` line

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
            "per_cycle_metrics": self.per_cycle_metrics,  # <-- expose to dashboard
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
        on_event: Optional callback ``fn(metrics_dict)`` called after
                  every phase and status change for live dashboards.
    """

    def __init__(
        self,
        config: dict,
        on_event: Optional[Callable[[dict], None]] = None,
        run_id: Optional[str] = None,
        distributed: Optional[str] = None,
        resume_dir: Optional[str] = None,
    ) -> None:
        import uuid

        self.run_id = run_id or str(uuid.uuid4())
        self.config = config
        self.on_event = on_event
        self.distributed = distributed
        self.resume_dir = resume_dir
        self.metrics = PipelineMetrics()
        self._cancelled = False

        # Initialise OTel tracing + metrics (no-op if endpoint not configured)
        setup_telemetry(config)

        # Populated by each stage
        self._dataset = None
        self._wake_dataset = None
        self._dream_base = None
        self._nightmare_base = None
        self._train_dl = None
        self._eval_dl = None
        self._dream_dl = None
        self._nightmare_dl = None
        self._val_dl = None
        self._trainer: Optional[Trainer] = None
        self._baseline_model = None
        # Held-out evaluation references (set during prepare())
        self._eval_dataset = None
        self._distortion_fn: Optional[Callable] = None
        # Adaptive cycle termination state
        self._last_robustness_score: Optional[float] = None
        self._convergence_count = 0
        self._final_convergence_delta: Optional[float] = None
        self._cycles_completed = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_training_event(self, event: TrainingEvent) -> None:
        logger.debug(
            "Training event: %s (%s)",
            event.event_type.value,
            event.phase,
        )

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

    def _handle_cycle_end(self, event: dict) -> None:
        """Handle per-cycle evaluation and adaptive convergence detection."""
        if self._trainer is not None and self._eval_dl is not None:
            try:
                metrics = evaluate_cycle(
                    model=self._trainer.model,
                    dataloader=self._eval_dl,
                    tokenizer=self._trainer.tokenizer,
                    base_dataset=self._eval_dataset,
                    distortion_fn=self._distortion_fn,
                    text_column=self.config.get("dataset", {}).get("text_column", "text"),
                    max_length=self.config.get("model", {}).get("max_length", 128),
                    batch_size=self.config.get("training", {}).get("batch_size", 8),
                    device=str(self._trainer.device),
                )
                metrics["cycle"] = event.get("cycle", 0)
                self.metrics.per_cycle_metrics.append(metrics)
                self._emit()
            except Exception:
                logger.exception("Per-cycle evaluation failed; continuing training.")

        if self._trainer is None:
            return
        training_cfg = self.config.get("training", {})
        auto_terminate = training_cfg.get("auto_terminate", False)
        threshold = training_cfg.get("convergence_threshold", 0.005)
        patience = training_cfg.get("convergence_patience", 2)
        if auto_terminate:
            score = quick_robustness_score(
                model=self._trainer.model,
                base_dataset=self._dataset,
                tokenizer=self._trainer.tokenizer,
                distortion_fn=apply_text_distortions,
                strength=0.5,
                text_column=self.config.get("dataset", {}).get("text_column", "text"),
                max_length=self.config.get("model", {}).get("max_length", 128),
                batch_size=training_cfg.get("batch_size", 8),
                device=str(self._trainer.device),
            )
            if self._last_robustness_score is None:
                self._last_robustness_score = score
                self._cycles_completed = event.get("cycle", 0) + 1
            else:
                delta = abs(score - self._last_robustness_score)
                self._final_convergence_delta = delta
                if delta < threshold:
                    self._convergence_count += 1
                else:
                    self._convergence_count = 0
                self._last_robustness_score = score
                self._cycles_completed = event.get("cycle", 0) + 1
                if self._convergence_count >= patience and self._trainer is not None:
                    logger.info(
                        "Robustness converged after %d cycles (delta=%.6f).",
                        self._cycles_completed,
                        delta,
                    )
                    self._trainer.request_stop()

    def cancel(self) -> None:
        """Request graceful cancellation of a running pipeline."""
        self._cancelled = True
        self.metrics.status = PipelineStatus.CANCELLED
        self._emit()

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

        Exactly one of *urls*, *file_path*, *text_content*, or
        *hf_dataset* must be provided.
        """
        self._set_status(PipelineStatus.INGESTING)
        self.metrics.progress_pct = 2.0
        self.metrics.current_phase = "ingest"
        self._emit()

        dataset_cfg = self.config.get("dataset", {})
        text_column = dataset_cfg.get("text_column", "text")
        max_samples = dataset_cfg.get("max_samples")
        seed = self.config.get("seed", 42)

        ingestor = DataIngestor(
            text_column=text_column,
            max_samples=max_samples,
            seed=seed,
        )

        span_attrs = {
            "source": (
                "urls"
                if urls
                else ("file" if file_path else ("text" if text_content else "huggingface"))
            ),
            "dataset.name": dataset_cfg.get("name", ""),
        }
        with trace_phase("ingest", span_attrs):
            try:
                if urls:
                    self._dataset = ingestor.from_urls(urls)
                elif file_path:
                    self._dataset = ingestor.from_file(file_path)
                elif text_content:
                    self._dataset = ingestor.from_text_content(text_content)
                elif hf_dataset:
                    self._dataset = ingestor.from_huggingface(hf_dataset, subset=hf_subset)
                else:
                    raise ValueError(
                        "Provide one of: urls, file_path, text_content, or hf_dataset."
                    )
                if self._dataset is not None:
                    logger.info("Ingestion complete: %d samples.", len(self._dataset))
                self.metrics.progress_pct = 8.0
                self._emit()
            except Exception as exc:
                self._fail(f"Ingestion failed: {exc}")
                raise

    # ------------------------------------------------------------------
    # Stage 1.5: Optimize (optional — Adaption Labs)
    # ------------------------------------------------------------------

    def optimize(self) -> None:
        """Phase-aware dataset optimization via Adaption Labs.

        Reads configuration from ``self.config["adaption"]``. Supports
        per-phase brand controls and recipe specifications. If disabled
        or the SDK is unavailable, this is a silent no-op.

        When phase-specific controls are configured, produces separate
        optimized datasets for wake, dream, and nightmare phases stored
        as ``self._wake_dataset``, ``self._dream_base``, ``self._nightmare_base``.
        """
        adaption_cfg = self.config.get("adaption", {})
        if not adaption_cfg.get("enabled", False):
            return

        import os

        try:
            from nightmarenet.data.adaption import Adaption, AdaptionOptimizer
        except ImportError:
            logger.info("adaption SDK not installed; skipping optimization.")
            return

        if Adaption is None:
            logger.info("adaption SDK not available; skipping optimization.")
            return

        if not os.environ.get("ADAPTION_API_KEY"):
            logger.info("ADAPTION_API_KEY not set; skipping optimization.")
            return

        column_mapping = adaption_cfg.get("column_mapping", {})
        max_rows = adaption_cfg.get("max_rows", 5000)

        self.metrics.progress_pct = 8.0
        self.metrics.current_phase = "optimize"
        self._emit()
        _optimize_span_ctx = trace_phase("optimize", {"has_phase_controls": str(False)})

        optimizer = AdaptionOptimizer()

        # Determine if we have per-phase controls (needed for span attr)
        has_phase_controls = any(
            adaption_cfg.get(f"{phase}_controls", {}).get("enabled", False)
            for phase in ("wake", "dream", "nightmare")
        )

        with trace_phase("optimize", {"has_phase_controls": str(has_phase_controls)}):
            # Estimate-first gating
            if adaption_cfg.get("estimate_first", False):
                try:
                    estimate = optimizer.estimate_cost(self._dataset, column_mapping)
                    if estimate:
                        max_credits = adaption_cfg.get("max_credits", 100)
                        if estimate["credits"] > max_credits:
                            logger.warning(
                                "Adaption estimated %.1f credits (budget: %.1f). "
                                "Skipping optimization.",
                                estimate["credits"],
                                max_credits,
                            )
                            return
                        logger.info(
                            "Adaption estimate: %.1f credits, ~%.1f min",
                            estimate["credits"],
                            estimate["estimated_minutes"],
                        )
                except Exception:
                    logger.warning("Estimate check failed; proceeding anyway.", exc_info=True)

            quality_results: dict = {}

            if has_phase_controls:
                self._optimize_per_phase(
                    optimizer, adaption_cfg, column_mapping, max_rows, quality_results
                )
            else:
                self._optimize_generic(
                    optimizer, adaption_cfg, column_mapping, max_rows, quality_results
                )

        self.metrics.adaption_quality = quality_results or None
        self.metrics.progress_pct = 15.0
        self._emit()

    def _optimize_generic(
        self,
        optimizer: Any,
        adaption_cfg: dict,
        column_mapping: dict,
        max_rows: int,
        quality_results: dict,
    ) -> None:
        """Single generic optimization pass (backward-compatible)."""
        brand_controls = adaption_cfg.get("brand_controls")
        recipe_specification = adaption_cfg.get("recipe_specification")

        try:
            result = optimizer.optimize_dataset(
                self._dataset,
                column_mapping,
                max_rows=max_rows,
                brand_controls=brand_controls,
                recipe_specification=recipe_specification,
            )
            if result is not None:
                optimized_dataset, quality = result
                self._dataset = optimized_dataset
                quality_results["generic"] = quality
                logger.info("Dataset optimization complete: %s", quality)
            else:
                logger.warning("Adaption optimization returned None; keeping original.")
        except Exception:
            logger.warning("Adaption optimization failed; keeping original.", exc_info=True)

    def _optimize_per_phase(
        self,
        optimizer: Any,
        adaption_cfg: dict,
        column_mapping: dict,
        max_rows: int,
        quality_results: dict,
    ) -> None:
        """Separate optimization passes per training phase."""
        phase_names = ("wake", "dream", "nightmare")

        for phase_name in phase_names:
            phase_cfg = adaption_cfg.get(f"{phase_name}_controls", {})
            if not phase_cfg.get("enabled", False):
                continue

            self.metrics.current_phase = f"optimize_{phase_name}"
            self._emit()

            brand_controls = phase_cfg.get("brand_controls")
            recipe_specification = phase_cfg.get("recipe_specification")

            try:
                result = optimizer.optimize_dataset(
                    self._dataset,
                    column_mapping,
                    max_rows=max_rows,
                    brand_controls=brand_controls,
                    recipe_specification=recipe_specification,
                )
                if result is not None:
                    optimized_dataset, quality = result
                    quality_results[phase_name] = quality

                    if phase_name == "wake":
                        self._wake_dataset = optimized_dataset
                    elif phase_name == "dream":
                        self._dream_base = optimized_dataset
                    elif phase_name == "nightmare":
                        self._nightmare_base = optimized_dataset

                    logger.info("Phase '%s' optimization complete: %s", phase_name, quality)
                else:
                    logger.warning(
                        "Phase '%s' optimization returned None; using original.",
                        phase_name,
                    )
            except Exception:
                logger.warning(
                    "Phase '%s' optimization failed; using original.",
                    phase_name,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Stage 2: Prepare
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Generate dream/nightmare splits and tokenise all data.

        Uses phase-specific optimized datasets when available from the
        Adaption optimization stage.
        """
        if self._dataset is None:
            raise RuntimeError("Call .ingest() before .prepare()")
        self._set_status(PipelineStatus.PREPARING)
        self.metrics.progress_pct = 10.0
        self.metrics.current_phase = "prepare"
        self._emit()

        model_name = self.config.get("model", {}).get("name", "unknown")
        with trace_phase("prepare", {"model.name": model_name}):
            try:
                dream_gen, nightmare_gen = create_generators_from_config(self.config)

                # ── Train / eval split ──────────────────────────────────────
                # Reserve a held-out fraction for post-training evaluation so
                # we do not measure performance on the same data we trained on.
                _min_eval_samples = 25
                eval_split_ratio = self.config.get("evaluation", {}).get("eval_split_ratio", 0.2)
                base_for_split = (
                    self._wake_dataset if self._wake_dataset is not None else self._dataset
                )
                n_total = len(base_for_split)  # type: ignore[arg-type]

                if eval_split_ratio > 0.0 and n_total >= _min_eval_samples:
                    n_eval = max(1, int(n_total * eval_split_ratio))
                    n_train = n_total - n_eval
                    train_indices = list(range(n_train))
                    eval_indices = list(range(n_train, n_total))
                    wake_data = base_for_split.select(train_indices)
                    self._eval_dataset = base_for_split.select(eval_indices)
                    logger.info(
                        "Train/eval split: %d train, %d eval (ratio=%.2f).",
                        n_train,
                        n_eval,
                        eval_split_ratio,
                    )
                else:
                    if eval_split_ratio > 0.0:
                        logger.warning(
                            "Dataset has only %d samples (minimum %d for splitting). "
                            "Using full dataset for both training and evaluation.",
                            n_total,
                            _min_eval_samples,
                        )
                    wake_data = base_for_split
                    self._eval_dataset = base_for_split

                # Save reference distortion function for robustness evaluation
                self._distortion_fn = apply_text_distortions

                dream_base = self._dream_base if self._dream_base is not None else self._dataset
                nightmare_base = (
                    self._nightmare_base if self._nightmare_base is not None else self._dataset
                )

                self._dream_generator = dream_gen
                self._nightmare_generator = nightmare_gen
                self._dream_base_dataset = dream_base
                self._nightmare_base_dataset = nightmare_base

                dream_data = dream_gen.generate(dream_base)
                nightmare_data = nightmare_gen.generate(nightmare_base)

                # Create trainer (loads model + tokenizer)
                self.callback_manager = CallbackManager()
                self._trainer = Trainer(
                    config=self.config,
                    distributed=self.distributed,
                    resume_dir=self.resume_dir,
                    callback_manager=self.callback_manager,
                )
                self.callback_manager.on_all(self._on_training_event)

                self._trainer.run_id = self.run_id

                # Snapshot baseline model weights for later evaluation
                self._baseline_model = copy.deepcopy(self._trainer.model)
                self._baseline_model.eval()

                # Tokenise
                text_column = self.config.get("dataset", {}).get("text_column", "text")
                max_length = self.config.get("model", {}).get("max_length", 128)
                batch_size = self.config.get("training", {}).get("batch_size", 8)

                self._train_dl = _tokenize_dataset(
                    wake_data,
                    self._trainer.tokenizer,
                    text_column,
                    max_length,
                    batch_size,
                )
                self._eval_dl = _tokenize_dataset(
                    self._eval_dataset,
                    self._trainer.tokenizer,
                    text_column,
                    max_length,
                    batch_size,
                )
                self._dream_dl = _tokenize_dataset(
                    dream_data,
                    self._trainer.tokenizer,
                    text_column,
                    max_length,
                    batch_size,
                )
                self._nightmare_dl = _tokenize_dataset(
                    nightmare_data,
                    self._trainer.tokenizer,
                    text_column,
                    max_length,
                    batch_size,
                )
                logger.info("Preparation complete: dataloaders ready.")
                self.metrics.progress_pct = 15.0
                self._emit()
            except Exception as exc:
                self._fail(f"Preparation failed: {exc}")
                raise

    # ------------------------------------------------------------------
    # Stage 3: Train
    # ------------------------------------------------------------------

    def train(self) -> list[dict]:
        """Run the full sleep-cycle training pipeline.

        Returns:
            Training history (list of phase result dicts).
        """
        if self._trainer is None:
            raise RuntimeError("Call .prepare() before .train()")
        if self._cancelled:
            return []

        num_cycles = self.config.get("training", {}).get("num_cycles", 3)
        self._set_status(PipelineStatus.TRAINING)
        self.metrics.total_cycles = num_cycles
        self.metrics.progress_pct = 15.0
        self._emit()

        def _on_train_progress(event: dict) -> None:
            self.metrics.current_cycle = event.get("cycle", self.metrics.current_cycle)
            phase = event.get("phase", "")
            if phase:
                self.metrics.current_phase = phase
            avg_loss = event.get("avg_loss")
            if avg_loss is not None:
                self.metrics.phase_loss = float(avg_loss)
            pct = event.get("progress_pct")
            if pct is not None:
                # Training occupies 15–85% of overall pipeline progress
                self.metrics.progress_pct = 15.0 + (float(pct) * 0.7)
            history = event.get("history")
            if history is not None:
                self.metrics.history = history
            if event.get("event") == "cycle_end":
                self._handle_cycle_end(event)
            self._emit()

        train_attrs = {
            "training.num_cycles": str(num_cycles),
            "model.name": self.config.get("model", {}).get("name", "unknown"),
        }
        start = time.time()
        with trace_phase("train", train_attrs):
            try:
                history = self._trainer.train(
                    train_dataloader=self._train_dl,
                    dream_dataloader=self._dream_dl,
                    nightmare_dataloader=self._nightmare_dl,
                    val_dataloader=self._val_dl,
                    on_progress=_on_train_progress,
                    dream_generator=getattr(self, "_dream_generator", None),
                    nightmare_generator=getattr(self, "_nightmare_generator", None),
                    dream_base_dataset=getattr(self, "_dream_base_dataset", None),
                    nightmare_base_dataset=getattr(self, "_nightmare_base_dataset", None),
                )

                # Update metrics from history
                self.metrics.history = history
                if history:
                    last = history[-1]
                    self.metrics.current_cycle = last.get("cycle", 0)
                    self.metrics.current_phase = last.get("phase", "")
                    self.metrics.phase_loss = last.get("avg_loss", 0.0)

                elapsed = time.time() - start
                self.metrics.progress_pct = 85.0
                self.metrics.eta_seconds = 0.0
                self._emit()
                logger.info("Training complete in %.1fs.", elapsed)
                return history
            except Exception as exc:
                self._fail(f"Training failed: {exc}")
                raise

    # ------------------------------------------------------------------
    # Stage 4: Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Run baseline vs trained model evaluation and generate report.

        Returns:
            Comparison dict with all metric deltas.
        """
        if self._trainer is None:
            raise RuntimeError("Call .train() before .evaluate()")

        self._set_status(PipelineStatus.EVALUATING)
        self.metrics.progress_pct = 88.0
        self.metrics.current_phase = "evaluate"
        self._emit()

        eval_attrs = {"model.name": self.config.get("model", {}).get("name", "unknown")}
        with trace_phase("evaluate", eval_attrs):
            try:
                evaluator = Evaluator(
                    model=self._trainer.model,
                    tokenizer=self._trainer.tokenizer,
                    config=self.config,
                    device=str(self._trainer.device),
                )

                # Use the held-out eval DataLoader so we never measure on
                # training data; fall back to train_dl only if eval_dl is
                # unavailable (should not happen in practice).
                clean_dl = self._eval_dl if self._eval_dl is not None else self._train_dl

                # Evaluate trained model
                trained_results = evaluator.evaluate(
                    clean_dataloader=clean_dl,
                    base_dataset=self._eval_dataset,
                    distortion_fn=self._distortion_fn,
                    label="nightmarenet-trained",
                )
                self.metrics.trained_results = trained_results

                # Evaluate baseline model (pre-training snapshot)
                baseline_evaluator = Evaluator(
                    model=self._baseline_model,
                    tokenizer=self._trainer.tokenizer,
                    config=self.config,
                    device=str(self._trainer.device),
                )
                baseline_results = baseline_evaluator.evaluate(
                    clean_dataloader=clean_dl,
                    base_dataset=self._eval_dataset,
                    distortion_fn=self._distortion_fn,
                    label="baseline",
                )
                self.metrics.baseline_results = baseline_results

                # Generate comparison
                comparison = evaluator.compare(baseline_results, trained_results)
                comparison["convergence"] = {
                    "cycles_completed": self._cycles_completed,
                    "final_delta": self._final_convergence_delta,
                    "auto_terminated": (
                        self._convergence_count
                        >= self.config.get("training", {}).get("convergence_patience", 2)
                    ),
                }
                self.metrics.comparison = comparison

                # Generate markdown report
                report = evaluator.generate_report(comparison)
                self.metrics.report_md = report

                # Save results
                results_dict = {
                    "baseline": baseline_results,
                    "trained": trained_results,
                    "comparison": comparison,
                }
                evaluator.save_results(results_dict)

                self.metrics.progress_pct = 100.0
                self.metrics.current_phase = "complete"
                self._set_status(PipelineStatus.COMPLETE)
                logger.info("Evaluation complete.")

                self._compute_quality_feedback(comparison)

                # Trigger webhook for run_complete (success)
                robustness_metric = comparison.get("metrics", {}).get("robustness", {})
                robustness_delta = robustness_metric.get("deltas", {}).get("auc_robustness")
                if robustness_delta is None:
                    robustness_delta = comparison.get("robustness_delta")

                trigger_webhook(
                    self.config,
                    "run_complete",
                    "Pipeline run completed successfully.",
                    {
                        "run_id": self.run_id,
                        "status": "complete",
                        "model": self.config.get("model", {}).get("name", "unknown"),
                        "robustness_delta": (
                            f"{robustness_delta:+.4f}"
                            if isinstance(robustness_delta, float)
                            else "N/A"
                        ),
                    },
                )

                # Trigger webhook for regression_detected
                if isinstance(robustness_delta, (int, float)) and robustness_delta < 0:
                    baseline_auc = robustness_metric.get("baseline", {}).get(
                        "auc_robustness", "N/A"
                    )
                    trained_auc = robustness_metric.get("trained", {}).get("auc_robustness", "N/A")
                    trigger_webhook(
                        self.config,
                        "regression_detected",
                        (
                            "Robustness regression detected after training! "
                            f"Drop: {robustness_delta:+.4f}"
                        ),
                        {
                            "run_id": self.run_id,
                            "model": self.config.get("model", {}).get("name", "unknown"),
                            "robustness_delta": f"{robustness_delta:+.4f}",
                            "baseline_auc": baseline_auc,
                            "trained_auc": trained_auc,
                        },
                    )

                # Export robustness delta as an OTel metric
                robustness_delta_val = comparison.get("robustness_delta")
                if robustness_delta_val is not None:
                    record_metric(
                        "robustness_score",
                        float(robustness_delta_val),
                        {"model": self.config.get("model", {}).get("name", "unknown")},
                    )

                return comparison
            except Exception as exc:
                self._fail(f"Evaluation failed: {exc}")
                raise

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
        logger.info("Quality feedback: %s", feedback)

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
        import os

        if self._trainer is None:
            raise RuntimeError("No trained model to export.")

        os.makedirs(output_dir, exist_ok=True)

        self._trainer.model.save_pretrained(output_dir)
        self._trainer.tokenizer.save_pretrained(output_dir)

        if self.metrics.report_md:
            report_path = os.path.join(output_dir, "evaluation_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(self.metrics.report_md)

        logger.info("Model exported to %s", output_dir)
        return output_dir

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
