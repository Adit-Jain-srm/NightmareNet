"""Evaluation engine for running all metrics and producing comparison reports."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

from torch.utils.data import DataLoader

from nightmarenet.evaluation.glue import evaluate_glue
from nightmarenet.evaluation.metrics import (
    classification_metrics,
    generalization_score,
    hallucination_rate,
    recall_score,
    robustness_score,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs all evaluation metrics and produces comparison reports.

    Args:
        model: Language model to evaluate.
        tokenizer: Tokenizer for the model.
        config: Evaluation configuration dictionary.
        device: Device to run evaluations on.
        tracker: Optional experiment tracker.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: dict[str, Any],
        device: str = "cpu",
        tracker: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.tracker = tracker
        self.eval_config = config.get("evaluation", {})
        default_metrics = (
            ["classification", "robustness"]
            if config.get("model", {}).get("type") == "image_classification"
            else ["recall", "generalization", "robustness", "hallucination"]
        )
        self.enabled_metrics = list(self.eval_config.get("metrics", default_metrics))
        calibration_enabled = self.eval_config.get("calibration", {}).get("enabled", False)
        if calibration_enabled and "calibration" not in self.enabled_metrics:
            self.enabled_metrics.append("calibration")
        self.output_dir = self.eval_config.get("output_dir", "results")
        self.significance_alpha = self.eval_config.get("significance_alpha", 0.05)
        os.makedirs(self.output_dir, exist_ok=True)

    def _log_eval(self, prefix: str, metrics: dict[str, Any]) -> None:
        """Log evaluation metrics to the experiment tracker."""
        if self.tracker is None:
            return
        self.tracker.log_metrics(
            {f"eval/{prefix}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        )

    def evaluate(
        self,
        clean_dataloader: DataLoader,
        ood_dataloader: Optional[DataLoader] = None,
        base_dataset: Any = None,
        distortion_fn: Any = None,
        label: str = "model",
    ) -> dict[str, Any]:
        """Run all enabled evaluation metrics.

        Args:
            clean_dataloader: DataLoader for clean test data.
            ood_dataloader: Optional DataLoader for out-of-distribution data.
            base_dataset: Optional base dataset for robustness testing.
            distortion_fn: Optional distortion function for robustness testing.
            label: Label for this evaluation run (e.g., "baseline", "dreamphase").

        Returns:
            Dict mapping metric names to their results.
        """
        results: dict[str, Any] = {"label": label, "timestamp": datetime.now().isoformat()}

        if "recall" in self.enabled_metrics:
            logger.info("Evaluating: recall")
            try:
                results["recall"] = recall_score(
                    self.model, clean_dataloader, self.tokenizer, self.device
                )
                if self.tracker:
                    self._log_eval("recall", results["recall"])
            except Exception as e:
                logger.error("Failed to compute recall: %s", e)
                results["recall"] = {"error": str(e)}

        if "generalization" in self.enabled_metrics and ood_dataloader is not None:
            logger.info("Evaluating: generalization")
            try:
                results["generalization"] = generalization_score(
                    self.model, ood_dataloader, clean_dataloader, self.device
                )
                if self.tracker:
                    self._log_eval(
                        "generalization",
                        results["generalization"],
                    )
            except Exception as e:
                logger.error("Failed to compute generalization: %s", e)
                results["generalization"] = {"error": str(e)}

        if (
            "robustness" in self.enabled_metrics
            and base_dataset is not None
            and distortion_fn is not None
        ):
            logger.info("Evaluating: robustness")
            try:
                strengths = self.eval_config.get(
                    "robustness_strengths",
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                )
                dataset_config = self.config.get("dataset", {})
                model_config = self.config.get("model", {})
                results["robustness"] = robustness_score(
                    self.model,
                    base_dataset,
                    self.tokenizer,
                    distortion_fn,
                    strengths=strengths,
                    text_column=dataset_config.get("text_column", "text"),
                    max_length=model_config.get("max_length", 128),
                    batch_size=self.config.get("training", {}).get("batch_size", 8),
                    device=self.device,
                    export_failures=self.eval_config.get("export_failures", False),
                )
                if self.eval_config.get("export_failures", False):
                    failures = results["robustness"].get("per_sample_data")
                    if failures:
                        from nightmarenet.evaluation.failure_export import save_failure_report

                        format_val = self.eval_config.get("failure_export_format", "json")
                        threshold = self.eval_config.get("failure_threshold", 0.20)
                        saved_path = save_failure_report(
                            failures, self.output_dir, format_val, threshold
                        )
                        if saved_path:
                            logger.info("Saved failure report to %s", saved_path)

                if self.tracker:
                    self._log_eval(
                        "robustness",
                        results["robustness"],
                    )
            except Exception as e:
                logger.error("Failed to compute robustness: %s", e)
                results["robustness"] = {"error": str(e)}

        if "hallucination" in self.enabled_metrics:
            logger.info("Evaluating: hallucination")
            try:
                results["hallucination"] = hallucination_rate(
                    self.model, clean_dataloader, self.tokenizer, self.device
                )
                if self.tracker:
                    self._log_eval(
                        "hallucination",
                        results["hallucination"],
                    )
            except Exception as e:
                logger.error("Failed to compute hallucination: %s", e)
                results["hallucination"] = {"error": str(e)}

        if "classification" in self.enabled_metrics:
            logger.info("Evaluating: classification")
            try:
                results["classification"] = classification_metrics(
                    self.model, clean_dataloader, self.device
                )
                if self.tracker:
                    self._log_eval(
                        "classification",
                        results["classification"],
                    )
            except Exception as e:
                logger.error("Failed to compute classification: %s", e)
                results["classification"] = {"error": str(e)}

        if "glue" in self.enabled_metrics:
            logger.info("Evaluating: GLUE benchmark")
            try:
                glue_tasks = self.eval_config.get("glue_tasks", None)
                glue_max_samples = self.eval_config.get("glue_max_samples", None)
                results["glue"] = evaluate_glue(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    tasks=glue_tasks,
                    device=self.device,
                    max_length=self.config.get("model", {}).get("max_length", 128),
                    batch_size=self.config.get("training", {}).get("batch_size", 8),
                    max_samples=glue_max_samples,
                )
                avg = results["glue"].get("average", {})
                if self.tracker and isinstance(avg, dict):
                    self.tracker.log_metrics(
                        {f"eval/glue_{k}": v for k, v in avg.items() if isinstance(v, (int, float))}
                    )
            except Exception as e:
                logger.error("Failed to compute GLUE: %s", e)
                results["glue"] = {"error": str(e)}

        if "certification" in self.enabled_metrics and base_dataset is not None:
            logger.info("Evaluating: certification")
            try:
                results["certification"] = self._run_certification(base_dataset)
                if self.tracker:
                    self._log_eval("certification", results["certification"])
            except Exception as e:
                logger.error("Failed to compute certification: %s", e)
                results["certification"] = {"error": str(e)}

        if "calibration" in self.enabled_metrics:
            logger.info("Evaluating: calibration")
            try:
                results["calibration"] = self._run_calibration(clean_dataloader)
                if self.tracker:
                    self._log_eval("calibration", results["calibration"])
            except Exception as e:
                logger.error("Failed to compute calibration: %s", e)
                results["calibration"] = {"error": str(e)}

        return results

    def _run_certification(self, base_dataset: Any) -> dict[str, Any]:
        """Run certified-robustness verification (randomized smoothing) on a dataset."""
        import nightmarenet.evaluation.evaluator

        certify_fn = getattr(nightmarenet.evaluation.evaluator, "certify_dataset", None)
        from nightmarenet.evaluation.evaluator.runners import run_certification

        return run_certification(
            self.model,
            self.tokenizer,
            self.device,
            base_dataset,
            self.eval_config,
            self.config,
            certify_fn=certify_fn,
        )

    def _run_calibration(self, dataloader: DataLoader) -> dict[str, Any]:
        """Run ECE computation and temperature scaling on the dataloader."""
        from nightmarenet.evaluation.evaluator.runners import run_calibration

        return run_calibration(self.model, self.device, dataloader, self.eval_config)

    def compare(
        self,
        baseline_results: dict[str, Any],
        trained_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce a comparison between baseline and trained model results."""
        from nightmarenet.evaluation.evaluator.comparators import compare_results

        return compare_results(self, baseline_results, trained_results)

    def save_results(
        self,
        results: dict[str, Any],
        filename: str = "evaluation_results.json",
    ) -> None:
        """Save evaluation results to a JSON file."""
        from nightmarenet.evaluation.evaluator.reporters import save_results

        save_results(self, results, filename)

    def generate_report(self, comparison: dict[str, Any]) -> str:
        """Generate a markdown report from a comparison dict."""
        from nightmarenet.evaluation.evaluator.reporters import generate_report

        return generate_report(self, comparison)

    def save_report(
        self,
        comparison: dict[str, Any],
        filename: str = "evaluation_report.md",
    ) -> str:
        """Generate and save a markdown report."""
        from nightmarenet.evaluation.evaluator.reporters import save_report

        return save_report(self, comparison, filename)
