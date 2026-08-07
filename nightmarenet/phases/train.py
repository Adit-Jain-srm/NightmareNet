"""Training phase: runs the sleep-cycle training loop with adaptive
convergence detection.

Extracted from `Pipeline.train()` and `Pipeline._handle_cycle_end()`
(nightmarenet/pipeline.py, L~560-680). Behavior unchanged: runs
`Trainer.train()`, tracks progress via an on_progress callback, and
performs per-cycle evaluation + auto-termination on robustness
convergence.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from nightmarenet.distortions.text import apply_text_distortions
from nightmarenet.evaluation.metrics import evaluate_cycle, quick_robustness_score
from nightmarenet.phases.base import Phase, PhaseResult, PipelineContext
from nightmarenet.utils.telemetry import trace_phase

logger = logging.getLogger(__name__)


class TrainPhase(Phase):
    name = "train"

    def __init__(self, on_progress: Optional[Any] = None) -> None:
        # Optional callback for orchestrator-level progress/metrics updates,
        # called with the same event dicts the old Pipeline._emit()'d on.
        self.on_progress = on_progress

    def execute(self, context: PipelineContext) -> PhaseResult:
        if context.trainer is None:
            return PhaseResult(
                success=False, phase_name=self.name, error="No trainer: run prepare before train."
            )
        if context.cancelled:
            return PhaseResult(success=True, phase_name=self.name, data={"history": []})

        num_cycles = context.config.get("training", {}).get("num_cycles", 3)

        def _on_train_progress(event: dict[str, Any]) -> None:
            if event.get("event") == "cycle_end":
                self._handle_cycle_end(context, event)
            if self.on_progress is not None:
                self.on_progress(event)

        train_attrs = {
            "training.num_cycles": str(num_cycles),
            "model.name": context.config.get("model", {}).get("name", "unknown"),
        }
        start = time.time()
        with trace_phase("train", train_attrs):
            try:
                history = context.trainer.train(
                    train_dataloader=context.train_dl,
                    dream_dataloader=context.dream_dl,
                    nightmare_dataloader=context.nightmare_dl,
                    val_dataloader=context.val_dl,
                    on_progress=_on_train_progress,
                    dream_generator=context.dream_generator,
                    nightmare_generator=context.nightmare_generator,
                    dream_base_dataset=context.dream_base_dataset,
                    nightmare_base_dataset=context.nightmare_base_dataset,
                )
                # Log training lineage for compliance reporting
                if history and context.tracker is not None:
                    for record in history:
                        context.tracker.log_phase(
                            cycle=record.get("cycle", 0),
                            phase=record.get("phase", "unknown"),
                            metrics=record,
                        )

                context.history = history
                elapsed = time.time() - start
                logger.info("Training complete in %.1fs.", elapsed)
            except (RuntimeError, ValueError) as exc:
                logger.exception("Training failed.")
                return PhaseResult(success=False, phase_name=self.name, error=str(exc))

        return PhaseResult(success=True, phase_name=self.name, data={"history": history})

    def _handle_cycle_end(self, context: PipelineContext, event: dict[str, Any]) -> None:
        """Per-cycle evaluation and adaptive convergence detection."""
        if context.trainer is not None and context.eval_dl is not None:
            try:
                metrics = evaluate_cycle(
                    model=context.trainer.model,
                    dataloader=context.eval_dl,
                    tokenizer=context.trainer.tokenizer,
                    base_dataset=context.eval_dataset,
                    distortion_fn=context.distortion_fn,
                    text_column=context.config.get("dataset", {}).get("text_column", "text"),
                    max_length=context.config.get("model", {}).get("max_length", 128),
                    batch_size=context.config.get("training", {}).get("batch_size", 8),
                    device=str(context.trainer.device),
                )
                metrics["cycle"] = event.get("cycle", 0)
                # Caller (orchestrator) reads this back off the event if needed;
                # per-cycle metrics accumulation stays orchestrator-side.
                event["cycle_metrics"] = metrics
            except (RuntimeError, ValueError):
                logger.exception("Per-cycle evaluation failed; continuing training.")

        if context.trainer is None:
            return

        training_cfg = context.config.get("training", {})
        if not training_cfg.get("auto_terminate", False):
            return

        threshold = training_cfg.get("convergence_threshold", 0.005)
        patience = training_cfg.get("convergence_patience", 2)

        score = quick_robustness_score(
            model=context.trainer.model,
            base_dataset=context.dataset,
            tokenizer=context.trainer.tokenizer,
            distortion_fn=apply_text_distortions,
            strength=0.5,
            text_column=context.config.get("dataset", {}).get("text_column", "text"),
            max_length=context.config.get("model", {}).get("max_length", 128),
            batch_size=training_cfg.get("batch_size", 8),
            device=str(context.trainer.device),
        )

        if context.last_robustness_score is None:
            context.last_robustness_score = score
            context.cycles_completed = event.get("cycle", 0) + 1
            return

        delta = abs(score - context.last_robustness_score)
        context.final_convergence_delta = delta
        if delta < threshold:
            context.convergence_count += 1
        else:
            context.convergence_count = 0
        context.last_robustness_score = score
        context.cycles_completed = event.get("cycle", 0) + 1

        if context.convergence_count >= patience:
            logger.info(
                "Robustness converged after %d cycles (delta=%.6f).",
                context.cycles_completed,
                delta,
            )
            context.trainer.request_stop()
