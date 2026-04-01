"""Evaluation engine for running all metrics and producing comparison reports.

Runs metrics before and after training to produce baseline vs. DreamPhase
comparison tables.
"""

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

    def __init__(self, model, tokenizer, config, device="cpu"):
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
            results["recall"] = recall_score(
                self.model, clean_dataloader, self.tokenizer, self.device
            )

        if "generalization" in self.enabled_metrics and ood_dataloader is not None:
            logger.info("Evaluating: generalization")
            results["generalization"] = generalization_score(
                self.model, ood_dataloader, clean_dataloader, self.device
            )

        if (
            "robustness" in self.enabled_metrics
            and base_dataset is not None
            and distortion_fn is not None
        ):
            logger.info("Evaluating: robustness")
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

        if "hallucination" in self.enabled_metrics:
            logger.info("Evaluating: hallucination")
            results["hallucination"] = hallucination_rate(
                self.model, clean_dataloader, self.tokenizer, self.device
            )

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

    def save_results(self, results: dict, filename: str = "evaluation_results.json"):
        """Save evaluation results to a JSON file.

        Args:
            results: Results dictionary to save.
            filename: Name of the output file.
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", path)

    def generate_report(self, comparison: dict) -> str:
        """Generate a markdown report from a comparison dict.

        Args:
            comparison: Output of self.compare().

        Returns:
            Markdown-formatted comparison report.
        """
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

        if "recall" in metrics:
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
                    f"| {key} | {bl:.4f if isinstance(bl, float) else bl} "
                    f"| {tr:.4f if isinstance(tr, float) else tr} "
                    f"| {delta:+.4f if isinstance(delta, float) else delta} |"
                )
            lines.append("")

        if "generalization" in metrics:
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
                    f"| {key} | {bl:.4f if isinstance(bl, float) else bl} "
                    f"| {tr:.4f if isinstance(tr, float) else tr} "
                    f"| {delta:+.4f if isinstance(delta, float) else delta} |"
                )
            lines.append("")

        if "robustness" in metrics:
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
                f"| AUC Robustness | {bl_auc:.4f if isinstance(bl_auc, float) else bl_auc} "
                f"| {tr_auc:.4f if isinstance(tr_auc, float) else tr_auc} "
                f"| {delta_auc:+.4f if isinstance(delta_auc, float) else delta_auc} |"
            )
            lines.append("")

        if "hallucination" in metrics:
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
                    f"| {key} | {bl:.4f if isinstance(bl, float) else bl} "
                    f"| {tr:.4f if isinstance(tr, float) else tr} "
                    f"| {delta:+.4f if isinstance(delta, float) else delta} |"
                )
            lines.append("")

        return "\n".join(lines)

    def save_report(self, comparison: dict, filename: str = "evaluation_report.md"):
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
