"""Evaluation engine for running all metrics and producing comparison reports.

Runs metrics before and after training to produce baseline vs. DreamPhase
comparison tables.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
from scipy import stats
from torch.utils.data import DataLoader

from nightmarenet.evaluation.certification import certify_dataset
from nightmarenet.evaluation.glue import evaluate_glue
from nightmarenet.evaluation.metrics import (
    classification_metrics,
    generalization_score,
    hallucination_rate,
    recall_score,
    robustness_score,
)

logger = logging.getLogger(__name__)


def _bootstrap_ci(
    baseline: list[float],
    trained: list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for paired differences.

    Args:
        baseline: List of baseline metric values (e.g., per-strength scores).
        trained: List of trained metric values (same length as baseline).
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level for CI (default 0.05 for 95% CI).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Dict with delta_mean, ci_lower, ci_upper, p_value, and significant flag.
    """
    if len(baseline) != len(trained) or len(baseline) < 2:
        return {
            "delta_mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "significant": False,
            "method": "insufficient_data",
        }

    baseline_arr = np.array(baseline)
    trained_arr = np.array(trained)
    deltas = trained_arr - baseline_arr
    delta_mean = float(np.mean(deltas))

    # Bootstrap resampling with seeded RNG for reproducibility
    rng = np.random.default_rng(seed)
    n = len(deltas)
    bootstrap_deltas = np.array(
        [np.mean(deltas[rng.choice(n, size=n, replace=True)]) for _ in range(n_bootstrap)]
    )
    ci_lower = float(np.percentile(bootstrap_deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2)))

    # Paired t-test p-value
    try:
        _, p_value = stats.ttest_rel(trained_arr, baseline_arr)
        p_value = float(p_value)
        # Handle NaN from identical data (zero variance)
        if np.isnan(p_value) or np.isinf(p_value):
            p_value = 1.0
    except Exception:
        p_value = 1.0

    # Significant if CI doesn't include 0 and p < alpha
    significant = (ci_lower > 0 or ci_upper < 0) and p_value < alpha

    return {
        "delta_mean": delta_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": significant,
        "method": "bootstrap_ci",
        "alpha": alpha,
    }


class Evaluator:
    """Runs all evaluation metrics and produces comparison reports.

    Args:
        model: Language model to evaluate.
        tokenizer: Tokenizer for the model.
        config: Evaluation configuration dictionary.
        device: Device to run evaluations on.
    """

    def __init__(self, model, tokenizer, config, device="cpu", tracker=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.tracker = tracker
        self.eval_config = config.get("evaluation", {})
        self.enabled_metrics = self.eval_config.get(
            "metrics", ["recall", "generalization", "robustness", "hallucination"]
        )
        self.output_dir = self.eval_config.get("output_dir", "results")
        self.significance_alpha = self.eval_config.get("significance_alpha", 0.05)
        os.makedirs(self.output_dir, exist_ok=True)

    def _log_eval(self, prefix: str, metrics: dict) -> None:
        """Log evaluation metrics to the experiment tracker."""
        if self.tracker is None:
            return
        self.tracker.log_metrics(
            {f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        )

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
                )
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

        return results

    def _run_certification(self, base_dataset) -> dict:
        """Run certified-robustness verification (randomized smoothing) on a dataset.

        Opt-in metric: only invoked from evaluate() when "certification" is explicitly
        listed in evaluation.metrics. Config is read from the evaluation.certification
        namespace (see configs/default.yaml).

        Budget control: `budget` caps the total forward passes for the *estimation*
        stage across the whole run (n * subset_size). If the configured n and
        subset_size would exceed it, n is reduced proportionally (subset_size is left
        alone, since it controls how many samples get any signal at all) and a warning
        is logged. This is a coarse, evaluator-level cap on top of certify_dataset's own
        finer-grained per-sample budget splitting.

        Args:
            base_dataset: Dataset to certify (same dataset used for robustness testing).

        Returns:
            Dict with certified_radius_mean, certified_radius_median,
            certification_abstain_rate, certified_accuracy, samples_certified, and
            budget_exceeded -- the shape expected in compare()'s comparison dict.
        """
        cert_config = self.eval_config.get("certification", {})
        n = cert_config.get("n", 1000)
        n0 = cert_config.get("n0", 100)
        subset_size = cert_config.get("subset_size", 50)
        budget = cert_config.get("budget")

        effective_size = subset_size if subset_size is not None else len(base_dataset)
        budget_exceeded = False
        if budget is not None and effective_size > 0 and n * effective_size > budget:
            reduced_n = max(1, budget // effective_size)
            logger.warning(
                "Certification budget exceeded: n=%d * subset_size=%d = %d > budget=%d; "
                "reducing n to %d",
                n, effective_size, n * effective_size, budget, reduced_n,
            )
            n = reduced_n
            budget_exceeded = True

        dataset_config = self.config.get("dataset", {})
        model_config = self.config.get("model", {})
        cert_result = certify_dataset(
            self.model,
            self.tokenizer,
            base_dataset,
            text_column=dataset_config.get("text_column", "text"),
            label_column=cert_config.get("label_column", "label"),
            sigma=cert_config.get("sigma", 0.1),
            n=n,
            n0=n0,
            alpha=cert_config.get("alpha", 0.001),
            subset_size=subset_size,
            batch_size=cert_config.get("batch_size", 100),
            max_length=model_config.get("max_length", 128),
            device=self.device,
        )

        return {
            "certified_radius_mean": cert_result["certified_radius_mean"],
            "certified_radius_median": cert_result["certified_radius_median"],
            "certification_abstain_rate": cert_result["certification_abstain_rate"],
            "certified_accuracy": cert_result["certified_accuracy"],
            "samples_certified": cert_result["n_samples"],
            "budget_exceeded": budget_exceeded,
        }

    def compare(self, baseline_results: dict, trained_results: dict) -> dict:
        """Produce a comparison between baseline and trained model results.

        Args:
            baseline_results: Evaluation results from the baseline model.
            trained_results: Evaluation results from the DreamPhase-trained model.

        Returns:
            Dict with side-by-side comparison for each metric, including
            statistical significance testing where applicable.
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

            # Compute deltas for key numeric fields. bool is a subtype of int in Python,
            # so it's explicitly excluded here -- otherwise flags like budget_exceeded
            # would silently get a meaningless numeric delta (e.g. True - False == 1).
            deltas = {}
            for key in baseline:
                baseline_val = baseline.get(key)
                trained_val = trained.get(key)
                if (
                    isinstance(baseline_val, (int, float))
                    and not isinstance(baseline_val, bool)
                    and isinstance(trained_val, (int, float))
                    and not isinstance(trained_val, bool)
                ):
                    deltas[key] = trained_val - baseline_val
            metric_comparison["deltas"] = deltas

            # Add statistical significance testing for robustness (per-strength scores)
            if metric_name == "robustness":
                baseline_perplexities = baseline.get("perplexities", [])
                trained_perplexities = trained.get("perplexities", [])
                if baseline_perplexities and trained_perplexities:
                    # Use inverse perplexity as the metric (higher is better).
                    # Perplexity is lower-is-better, so 1/ppl converts it to higher-is-better
                    # for the paired statistical test to correctly interpret improvements.
                    baseline_scores = [1.0 / max(p, 1e-8) for p in baseline_perplexities]
                    trained_scores = [1.0 / max(p, 1e-8) for p in trained_perplexities]
                    significance = _bootstrap_ci(
                        baseline_scores,
                        trained_scores,
                        alpha=self.significance_alpha,
                    )
                    metric_comparison["significance"] = significance

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
            return "error" not in metric_data.get(
                "baseline", {}
            ) and "error" not in metric_data.get("trained", {})

        lines = [
            "# NightmareNet Evaluation Report",
            "",
            f"**Baseline**: {comparison.get('baseline_label', 'N/A')}",
            f"**Trained**: {comparison.get('trained_label', 'N/A')}",
            "",
            "## Results",
            "",
        ]
        convergence = comparison.get("convergence")

        if convergence:
            final_delta = convergence.get("final_delta")

            lines.extend(
                [
                    "## Training Summary",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Cycles completed | {convergence.get('cycles_completed', 'N/A')} |",
                    (
                        f"| Final robustness delta | {final_delta:.6f} |"
                        if final_delta is not None
                        else "| Final robustness delta | N/A |"
                    ),
                    f"| Adaptive termination |"
                    f"{'Yes' if convergence.get('auto_terminated') else 'No'} |",
                ]
            )
        metrics = comparison.get("metrics", {})

        if "recall" in metrics and _metric_ok(metrics["recall"]):
            r = metrics["recall"]
            lines.extend(
                [
                    "### Recall",
                    "",
                    "| Metric | Baseline | Trained | Delta |",
                    "|--------|----------|---------|-------|",
                ]
            )
            for key in ["token_accuracy", "perplexity"]:
                bl = r.get("baseline", {}).get(key, "N/A")
                tr = r.get("trained", {}).get(key, "N/A")
                delta = r.get("deltas", {}).get(key, "N/A")
                lines.append(f"| {key} | {_fmt(bl)} | {_fmt(tr)} | {_fmt(delta, signed=True)} |")
            lines.append("")

        if "generalization" in metrics and _metric_ok(metrics["generalization"]):
            r = metrics["generalization"]
            lines.extend(
                [
                    "### Generalization",
                    "",
                    "| Metric | Baseline | Trained | Delta |",
                    "|--------|----------|---------|-------|",
                ]
            )
            for key in ["generalization_score", "generalization_ratio"]:
                bl = r.get("baseline", {}).get(key, "N/A")
                tr = r.get("trained", {}).get(key, "N/A")
                delta = r.get("deltas", {}).get(key, "N/A")
                lines.append(f"| {key} | {_fmt(bl)} | {_fmt(tr)} | {_fmt(delta, signed=True)} |")
            lines.append("")

        if "robustness" in metrics and _metric_ok(metrics["robustness"]):
            r = metrics["robustness"]
            lines.extend(
                [
                    "### Robustness",
                    "",
                    "| Metric | Baseline | Trained | Delta |",
                    "|--------|----------|---------|-------|",
                ]
            )
            bl_auc = r.get("baseline", {}).get("auc_robustness", "N/A")
            tr_auc = r.get("trained", {}).get("auc_robustness", "N/A")
            delta_auc = r.get("deltas", {}).get("auc_robustness", "N/A")
            lines.append(
                f"| AUC Robustness | {_fmt(bl_auc)} "
                f"| {_fmt(tr_auc)} "
                f"| {_fmt(delta_auc, signed=True)} |"
            )

            # Add statistical significance information
            sig = r.get("significance", {})
            if sig and sig.get("method") == "bootstrap_ci":
                sig_verdict = (
                    "**statistically significant**"
                    if sig.get("significant")
                    else "not significant"
                )
                lines.extend(
                    [
                        "",
                        "**Statistical Significance (Bootstrap CI)**",
                        f"- Delta mean: {_fmt(sig.get('delta_mean', 0.0), signed=True)}",
                        f"- 95% CI: [{_fmt(sig.get('ci_lower', 0.0))}, "
                        f"{_fmt(sig.get('ci_upper', 0.0))}]",
                        f"- p-value: {sig.get('p_value', 1.0):.4f}",
                        f"- Verdict: {sig_verdict} (α={sig.get('alpha', 0.05)})",
                    ]
                )
            lines.append("")

        if "hallucination" in metrics and _metric_ok(metrics["hallucination"]):
            r = metrics["hallucination"]
            lines.extend(
                [
                    "### Hallucination",
                    "",
                    "| Metric | Baseline | Trained | Delta |",
                    "|--------|----------|---------|-------|",
                ]
            )
            for key in ["hallucination_rate", "avg_hallucination_confidence"]:
                bl = r.get("baseline", {}).get(key, "N/A")
                tr = r.get("trained", {}).get(key, "N/A")
                delta = r.get("deltas", {}).get(key, "N/A")
                lines.append(f"| {key} | {_fmt(bl)} | {_fmt(tr)} | {_fmt(delta, signed=True)} |")
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
