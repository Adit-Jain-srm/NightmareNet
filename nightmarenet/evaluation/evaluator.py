"""Evaluation engine for running all metrics and producing comparison reports.

Runs metrics before and after training to produce baseline vs. DreamPhase
comparison tables.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

from torch.utils.data import DataLoader

from nightmarenet.evaluation.metrics import (
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
    """

    def __init__(self, model, tokenizer, config, device="cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.eval_config = config.get("evaluation", {})
        self.enabled_metrics = self.eval_config.get(
            "metrics", ["recall", "generalization", "robustness", "hallucination"]
        )
        self.output_dir = self.eval_config.get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(
        self,
        clean_dataloader: DataLoader,
        ood_dataloader: Optional[DataLoader] = None,
        base_dataset=None,
        distortion_fn=None,
        label: str = "model",
    ) -> dict:
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
        results = {"label": label, "timestamp": datetime.now().isoformat()}

        if "recall" in self.enabled_metrics:
            logger.info("Evaluating: recall")
            try:
                results["recall"] = recall_score(
                    self.model, clean_dataloader, self.tokenizer, self.device
                )
            except Exception as e:
                logger.error("Failed to compute recall: %s", e)
                results["recall"] = {"error": str(e)}

        if "generalization" in self.enabled_metrics and ood_dataloader is not None:
            logger.info("Evaluating: generalization")
            try:
                results["generalization"] = generalization_score(
                    self.model, ood_dataloader, clean_dataloader, self.device
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
            except Exception as e:
                logger.error("Failed to compute hallucination: %s", e)
                results["hallucination"] = {"error": str(e)}

        return results

    def compare(self, baseline_results: dict, trained_results: dict) -> dict:
        """Produce a comparison between baseline and trained model results.

        Args:
            baseline_results: Evaluation results from the baseline model.
            trained_results: Evaluation results from the DreamPhase-trained model.

        Returns:
            Dict with side-by-side comparison for each metric.
        """
        comparison = {
            "baseline_label": baseline_results.get("label", "baseline"),
            "trained_label": trained_results.get("label", "dreamphase"),
            "metrics": {},
        }

        for metric_name in self.enabled_metrics:
            baseline = baseline_results.get(metric_name, {})
            trained = trained_results.get(metric_name, {})

            if not baseline and not trained:
                continue

            metric_comparison = {
                "baseline": baseline,
                "trained": trained,
            }

            # Compute deltas for key numeric fields
            deltas = {}
            for key in baseline:
                if isinstance(baseline.get(key), (int, float)) and isinstance(
                    trained.get(key), (int, float)
                ):
                    deltas[key] = trained[key] - baseline[key]
            metric_comparison["deltas"] = deltas

            comparison["metrics"][metric_name] = metric_comparison

        return comparison

    def save_results(self, results: dict, filename: str = "evaluation_results.json") -> None:
        """Save evaluation results to a JSON file.

        Args:
            results: Results dictionary to save.
            filename: Name of the output file.
        """
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info("Results saved to %s", path)
        except Exception as e:
            logger.error("Failed to save results to %s: %s", path, e)

    def generate_report(self, comparison: dict) -> str:
        """Generate a markdown report from a comparison dict.

        Args:
            comparison: Output of self.compare().

        Returns:
            Markdown-formatted comparison report.
        """
        def _fmt(val, signed: bool = False) -> str:
            """Format a metric value: floats get .4f, others pass through."""
            if isinstance(val, float):
                return f"{val:+.4f}" if signed else f"{val:.4f}"
            return str(val)

        def _metric_ok(metric_data: dict) -> bool:
            """Check a metric section has no errors in baseline or trained."""
            return (
                "error" not in metric_data.get("baseline", {})
                and "error" not in metric_data.get("trained", {})
            )

        lines = [
            "# NightmareNet Evaluation Report",
            "",
            f"**Baseline**: {comparison.get('baseline_label', 'N/A')}",
            f"**Trained**: {comparison.get('trained_label', 'N/A')}",
            "",
            "## Results",
            "",
        ]

        metrics = comparison.get("metrics", {})

        if "recall" in metrics and _metric_ok(metrics["recall"]):
            r = metrics["recall"]
            lines.extend([
                "### Recall",
                "",
                "| Metric | Baseline | Trained | Delta |",
                "|--------|----------|---------|-------|",
            ])
            for key in ["token_accuracy", "perplexity"]:
                bl = r.get("baseline", {}).get(key, "N/A")
                tr = r.get("trained", {}).get(key, "N/A")
                delta = r.get("deltas", {}).get(key, "N/A")
                lines.append(
                    f"| {key} | {_fmt(bl)} "
                    f"| {_fmt(tr)} "
                    f"| {_fmt(delta, signed=True)} |"
                )
            lines.append("")

        if "generalization" in metrics and _metric_ok(metrics["generalization"]):
            r = metrics["generalization"]
            lines.extend([
                "### Generalization",
                "",
                "| Metric | Baseline | Trained | Delta |",
                "|--------|----------|---------|-------|",
            ])
            for key in ["generalization_score", "generalization_ratio"]:
                bl = r.get("baseline", {}).get(key, "N/A")
                tr = r.get("trained", {}).get(key, "N/A")
                delta = r.get("deltas", {}).get(key, "N/A")
                lines.append(
                    f"| {key} | {_fmt(bl)} "
                    f"| {_fmt(tr)} "
                    f"| {_fmt(delta, signed=True)} |"
                )
            lines.append("")

        if "robustness" in metrics and _metric_ok(metrics["robustness"]):
            r = metrics["robustness"]
            lines.extend([
                "### Robustness",
                "",
                "| Metric | Baseline | Trained | Delta |",
                "|--------|----------|---------|-------|",
            ])
            bl_auc = r.get("baseline", {}).get("auc_robustness", "N/A")
            tr_auc = r.get("trained", {}).get("auc_robustness", "N/A")
            delta_auc = r.get("deltas", {}).get("auc_robustness", "N/A")
            lines.append(
                f"| AUC Robustness | {_fmt(bl_auc)} "
                f"| {_fmt(tr_auc)} "
                f"| {_fmt(delta_auc, signed=True)} |"
            )
            lines.append("")

        if "hallucination" in metrics and _metric_ok(metrics["hallucination"]):
            r = metrics["hallucination"]
            lines.extend([
                "### Hallucination",
                "",
                "| Metric | Baseline | Trained | Delta |",
                "|--------|----------|---------|-------|",
            ])
            for key in ["hallucination_rate", "avg_hallucination_confidence"]:
                bl = r.get("baseline", {}).get(key, "N/A")
                tr = r.get("trained", {}).get(key, "N/A")
                delta = r.get("deltas", {}).get(key, "N/A")
                lines.append(
                    f"| {key} | {_fmt(bl)} "
                    f"| {_fmt(tr)} "
                    f"| {_fmt(delta, signed=True)} |"
                )
            lines.append("")

        return "\n".join(lines)

    def save_report(self, comparison: dict, filename: str = "evaluation_report.md") -> str:
        """Generate and save a markdown report.

        Args:
            comparison: Output of self.compare().
            filename: Name of the output file.
        """
        report = self.generate_report(comparison)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(report)
        logger.info("Report saved to %s", path)
        return report
