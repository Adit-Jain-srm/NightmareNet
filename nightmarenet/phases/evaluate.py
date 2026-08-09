"""Evaluation phase: baseline vs trained comparison, report generation,
webhooks, and optional Hub push.

Extracted from `Pipeline.evaluate()` and `Pipeline._compute_quality_feedback()`
(nightmarenet/pipeline.py, L~680-810). Behavior unchanged.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from nightmarenet.evaluation.evaluator import Evaluator
from nightmarenet.exceptions import HubUploadError
from nightmarenet.phases.base import Phase, PhaseResult, PipelineContext
from nightmarenet.utils.telemetry import record_metric, trace_phase
from nightmarenet.utils.webhooks import trigger_webhook

logger = logging.getLogger(__name__)


class EvaluatePhase(Phase):
    name = "evaluate"

    def execute(self, context: PipelineContext) -> PhaseResult:
        if context.trainer is None:
            return PhaseResult(
                success=False, phase_name=self.name, error="No trainer: run train before evaluate."
            )

        eval_attrs = {"model.name": context.config.get("model", {}).get("name", "unknown")}
        with trace_phase("evaluate", eval_attrs):
            try:
                evaluator = Evaluator(
                    model=context.trainer.model,
                    tokenizer=context.trainer.tokenizer,
                    config=context.config,
                    device=str(context.trainer.device),
                )
                clean_dl = context.eval_dl if context.eval_dl is not None else context.train_dl

                trained_results = evaluator.evaluate(
                    clean_dataloader=clean_dl,
                    base_dataset=context.eval_dataset,
                    distortion_fn=context.distortion_fn,
                    label="nightmarenet-trained",
                )
                context.trained_results = trained_results

                baseline_evaluator = Evaluator(
                    model=context.baseline_model,
                    tokenizer=context.trainer.tokenizer,
                    config=context.config,
                    device=str(context.trainer.device),
                )
                baseline_results = baseline_evaluator.evaluate(
                    clean_dataloader=clean_dl,
                    base_dataset=context.eval_dataset,
                    distortion_fn=context.distortion_fn,
                    label="baseline",
                )
                context.baseline_results = baseline_results

                comparison = evaluator.compare(baseline_results, trained_results)
                comparison["convergence"] = {
                    "cycles_completed": context.cycles_completed,
                    "final_delta": context.final_convergence_delta,
                    "auto_terminated": (
                        context.convergence_count
                        >= context.config.get("training", {}).get("convergence_patience", 2)
                    ),
                }
                context.comparison = comparison

                report = evaluator.generate_report(comparison)
                context.report_md = report

                evaluator.save_results(
                    {
                        "baseline": baseline_results,
                        "trained": trained_results,
                        "comparison": comparison,
                    }
                )

                # EU AI Act Article 15 compliance report
                tracking_cfg = context.config.get("tracking", {})
                if tracking_cfg.get("compliance_report", False):
                    from nightmarenet.compliance.report import generate_report

                    eval_cfg = context.config.get("evaluation", {})
                    output_dir = eval_cfg.get("output_dir", "results")
                    model_path = ""
                    training_cfg = context.config.get("training", {})
                    checkpoint_dir = training_cfg.get("checkpoint_dir")
                    if checkpoint_dir:
                        model_path = checkpoint_dir

                    generate_report(
                        config=context.config,
                        comparison=comparison,
                        model_path=model_path,
                        output_dir=output_dir,
                        tracker=context.tracker,
                    )
                    logger.info("Compliance report generated.")

                self._compute_quality_feedback(context, comparison)
                self._fire_webhooks(context, comparison)
                self._maybe_push_hub(context, comparison)

            except (RuntimeError, ValueError, OSError) as exc:
                logger.exception("Evaluation failed.")
                return PhaseResult(success=False, phase_name=self.name, error=str(exc))

        return PhaseResult(success=True, phase_name=self.name, data={"comparison": comparison})

    def _compute_quality_feedback(
        self, context: PipelineContext, comparison: dict[str, Any]
    ) -> None:
        if not context.adaption_quality:
            return

        robustness_delta = comparison.get("robustness_delta")
        if robustness_delta is None:
            for key in ("robustness", "avg_robustness", "mean_robustness"):
                if key in comparison:
                    robustness_delta = comparison[key]
                    break

        feedback: dict[str, Any] = {
            "adaption_phases_optimized": list(context.adaption_quality.keys()),
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

        context.quality_feedback = feedback
        logger.info("Quality feedback: %s", feedback)

    def _fire_webhooks(self, context: PipelineContext, comparison: dict[str, Any]) -> None:
        robustness_metric = comparison.get("metrics", {}).get("robustness", {})
        robustness_delta = robustness_metric.get("deltas", {}).get("auc_robustness")
        if robustness_delta is None:
            robustness_delta = comparison.get("robustness_delta")

        trigger_webhook(
            context.config,
            "run_complete",
            "Pipeline run completed successfully.",
            {
                "run_id": context.run_id,
                "status": "complete",
                "model": context.config.get("model", {}).get("name", "unknown"),
                "robustness_delta": (
                    f"{robustness_delta:+.4f}" if isinstance(robustness_delta, float) else "N/A"
                ),
            },
        )

        if isinstance(robustness_delta, (int, float)) and robustness_delta < 0:
            baseline_auc = robustness_metric.get("baseline", {}).get("auc_robustness", "N/A")
            trained_auc = robustness_metric.get("trained", {}).get("auc_robustness", "N/A")
            trigger_webhook(
                context.config,
                "regression_detected",
                f"Robustness regression detected after training! Drop: {robustness_delta:+.4f}",
                {
                    "run_id": context.run_id,
                    "model": context.config.get("model", {}).get("name", "unknown"),
                    "robustness_delta": f"{robustness_delta:+.4f}",
                    "baseline_auc": baseline_auc,
                    "trained_auc": trained_auc,
                },
            )

        robustness_delta_val = comparison.get("robustness_delta")
        if robustness_delta_val is not None:
            record_metric(
                "robustness_score",
                float(robustness_delta_val),
                {"model": context.config.get("model", {}).get("name", "unknown")},
            )

    def _maybe_push_hub(self, context: PipelineContext, comparison: dict[str, Any]) -> None:
        tracking_cfg = context.config.get("tracking", {})
        auto_push_repo = tracking_cfg.get("auto_push_hub")
        if not auto_push_repo:
            return

        import yaml

        from nightmarenet.hub.core import push_model
        from nightmarenet.phases.export import ExportPhase

        logger.info("Pushing to Hub...")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                ExportPhase(tmp_dir).execute(context)

                pipeline_metadata = {
                    "robustness_score": float(comparison.get("robustness_score", 0.0)),
                    "training_config": context.config,
                }
                metadata_file_path = os.path.join(tmp_dir, "hub_metadata.yaml")
                with open(metadata_file_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(pipeline_metadata, f)

                push_model(
                    model_dir=tmp_dir,
                    repo_id=auto_push_repo,
                    metadata_path=metadata_file_path,
                )
        except (RuntimeError, OSError, ValueError, HubUploadError) as upload_err:
            logger.error("Push failed: %s", upload_err)
