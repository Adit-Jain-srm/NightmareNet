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
        default_metrics = (
            ["classification", "robustness"]
            if config.get("model", {}).get("type") == "image_classification"
            else ["recall", "generalization", "robustness", "hallucination"]
        )
        self.enabled_metrics = self.eval_config.get("metrics", default_metrics)
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
                    export_failures=self.eval_config.get("export_failures", False),
                )
                if self.eval_config.get("export_failures", False):
                    failures = results["robustness"].get("per_sample_data")
                    if failures:
                        from nightmarenet.evaluation.failure_export import save_failure_report

                        format = self.eval_config.get("failure_export_format", "json")
                        threshold = self.eval_config.get("failure_threshold", 0.20)
                        saved_path = save_failure_report(
                            failures, self.output_dir, format, threshold
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
                n,
                effective_size,
                n * effective_size,
                budget,
                reduced_n,
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
                if "accuracies" in baseline:
                    baseline_scores = baseline.get("accuracies", [])
                    trained_scores = trained.get("accuracies", [])
                else:
                    baseline_perplexities = baseline.get("perplexities", [])
                    trained_perplexities = trained.get("perplexities", [])
                    baseline_scores = (
                        [1.0 / max(p, 1e-8) for p in baseline_perplexities]
                        if baseline_perplexities
                        else []
                    )
                    trained_scores = (
                        [1.0 / max(p, 1e-8) for p in trained_perplexities]
                        if trained_perplexities
                        else []
                    )
                if baseline_scores and trained_scores:
                    significance = _bootstrap_ci(
                        baseline_scores,
                        trained_scores,
                        alpha=self.significance_alpha,
                    )
                    metric_comparison["significance"] = significance

            comparison["metrics"][metric_name] = metric_comparison

            # Backward-compatible top-level robustness summary.
        robustness = comparison["metrics"].get("robustness")
        if robustness:
            trained_auc = robustness.get("trained", {}).get("auc_robustness")
            delta_auc = robustness.get("deltas", {}).get("auc_robustness")

            if trained_auc is not None:
                comparison["robustness_score"] = trained_auc

            if delta_auc is not None:
                comparison["robustness_delta"] = delta_auc

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

        if "classification" in metrics and _metric_ok(metrics["classification"]):
            r = metrics["classification"]
            lines.extend(
                [
                    "### Classification",
                    "",
                    "| Metric | Baseline | Trained | Delta |",
                    "|--------|----------|---------|-------|",
                ]
            )
            for key in ["accuracy", "f1_weighted"]:
                bl = r.get("baseline", {}).get(key, "N/A")
                tr = r.get("trained", {}).get(key, "N/A")
                delta = r.get("deltas", {}).get(key, "N/A")
                lines.append(f"| {key} | {_fmt(bl)} | {_fmt(tr)} | {_fmt(delta, signed=True)} |")
            lines.append("")

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
                    "**statistically significant**" if sig.get("significant") else "not significant"
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

            trained_robustness = r.get("trained", {})
            if "top_failures" in trained_robustness:
                lines.extend(
                    [
                        "",
                        "## Confidence Delta Analysis",
                        "",
                        "### Top 10 Most Vulnerable Samples",
                        "",
                        "| Sample | Preview | Clean | Distorted | Delta |",
                        "|--------|---------|-------|-----------|-------|",
                    ]
                )
                for f in trained_robustness["top_failures"]:
                    idx = f.get("sample_index", "N/A")
                    prev = f.get("preview", "N/A")
                    cln = f.get("clean_confidence", 0.0)
                    dst = f.get("distorted_confidence", 0.0)
                    dlt = f.get("confidence_delta", 0.0)
                    lines.append(f"| {idx} | {prev} | {_fmt(cln)} | {_fmt(dst)} | {_fmt(dlt)} |")

                dist = trained_robustness.get("delta_distribution", {})
                if dist:
                    lines.extend(
                        [
                            "",
                            "### Severity Distribution",
                            "",
                            f"- **0-10% drop**: {dist.get('0_10', 0)}",
                            f"- **10-25% drop**: {dist.get('10_25', 0)}",
                            f"- **25-50% drop**: {dist.get('25_50', 0)}",
                            f"- **50%+ drop**: {dist.get('50_plus', 0)}",
                        ]
                    )

            lines.append("")

        if "certification" in metrics and _metric_ok(metrics["certification"]):
            lines.extend(self._format_certification_section(metrics["certification"]))

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

    def _format_certification_section(self, cert_metrics: dict) -> list:
        """Formats the "Certified Robustness" markdown section (issue #162).

        Uses the same baseline/trained/delta table shape as the other sections in
        generate_report() (for consistency, and because the acceptance criteria for
        issue #162 specifically calls for a baseline-vs-trained certified-radius
        comparison), plus a callout distinguishing this formal, distribution-free
        guarantee from the empirical Robustness (AUC) metric above it, and a
        configuration line (noise sigma / smoothing sample count) read from
        `evaluation.certification` -- config values that are inputs to the run, not
        outputs, so they don't belong in the per-run baseline/trained dict itself
        (and aren't part of _run_certification's returned keys, which several
        existing tests pin exactly).

        Args:
            cert_metrics: metrics["certification"] from a compare() output, i.e.
                {"baseline": {...}, "trained": {...}, "deltas": {...}}.

        Returns:
            List of markdown lines (not yet joined).
        """

        def _fmt(val, signed: bool = False) -> str:
            if isinstance(val, float):
                return f"{val:+.4f}" if signed else f"{val:.4f}"
            if val is None:
                return "N/A"
            return str(val)

        def _pct(val) -> str:
            return f"{val * 100:.1f}%" if isinstance(val, (int, float)) else "N/A"

        def _pct_signed(val) -> str:
            return f"{val * 100:+.1f}pp" if isinstance(val, (int, float)) else "N/A"

        baseline = cert_metrics.get("baseline", {})
        trained = cert_metrics.get("trained", {})
        deltas = cert_metrics.get("deltas", {})

        cert_config = self.eval_config.get("certification", {})
        sigma = cert_config.get("sigma", 0.1)
        n = cert_config.get("n", 1000)
        n0 = cert_config.get("n0", 100)
        alpha = cert_config.get("alpha", 0.001)
        subset_size = cert_config.get("subset_size")

        def _samples_str(side: dict) -> str:
            certified = side.get("samples_certified", "N/A")
            total = subset_size if subset_size is not None else certified
            return f"{certified} / {total}"

        lines = [
            "### Certified Robustness (Randomized Smoothing)",
            "",
            "> **Formal vs. empirical**: certified radii are a formal, "
            "distribution-free guarantee (no perturbation with embedding-space L2 "
            "norm below the radius can change the prediction) -- unlike the "
            "empirical Robustness (AUC) score above, which only reflects degradation "
            "under the specific distortions actually tried. Radii are L2 distances "
            "in **embedding space**, not token/edit-distance space.",
            "",
            "| Metric | Baseline | Trained | Delta |",
            "|--------|----------|---------|-------|",
            (
                f"| Mean certified radius | {_fmt(baseline.get('certified_radius_mean'))} "
                f"| {_fmt(trained.get('certified_radius_mean'))} "
                f"| {_fmt(deltas.get('certified_radius_mean'), signed=True)} |"
            ),
            (
                f"| Median certified radius | {_fmt(baseline.get('certified_radius_median'))} "
                f"| {_fmt(trained.get('certified_radius_median'))} "
                f"| {_fmt(deltas.get('certified_radius_median'), signed=True)} |"
            ),
            (
                f"| Abstention rate | {_pct(baseline.get('certification_abstain_rate'))} "
                f"| {_pct(trained.get('certification_abstain_rate'))} "
                f"| {_pct_signed(deltas.get('certification_abstain_rate'))} |"
            ),
            (
                f"| Certified accuracy | {_fmt(baseline.get('certified_accuracy'))} "
                f"| {_fmt(trained.get('certified_accuracy'))} "
                f"| {_fmt(deltas.get('certified_accuracy'), signed=True)} |"
            ),
            (f"| Samples certified | {_samples_str(baseline)} | {_samples_str(trained)} | N/A |"),
            "",
            f"**Configuration**: noise sigma (σ) = {sigma}, smoothing samples (n) = {n}, "
            f"selection samples (n0) = {n0}, significance level (α) = {alpha}",
        ]

        if baseline.get("budget_exceeded") or trained.get("budget_exceeded"):
            lines.append(
                "> Note: the configured compute budget reduced `n` for at least one "
                "run below (see logs) -- certified radii for that run are valid but "
                "computed with fewer smoothing samples than requested."
            )

        lines.append("")
        return lines

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
