"""Result formatting, console output, and report generation/file export."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def save_results(
    evaluator: Any,
    results: dict[str, Any],
    filename: str = "evaluation_results.json",
) -> None:
    """Save evaluation results to a JSON file.

    Args:
        evaluator: The Evaluator instance.
        results: Results dictionary to save.
        filename: Name of the output file.
    """
    path = os.path.join(evaluator.output_dir, filename)
    try:
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", path)
    except Exception as e:
        logger.error("Failed to save results to %s: %s", path, e)


def _format_certification_section(evaluator: Any, cert_metrics: dict[str, Any]) -> list[str]:
    """Formats the "Certified Robustness" markdown section.

    Args:
        evaluator: The Evaluator instance.
        cert_metrics: metrics["certification"] from a compare() output.

    Returns:
        List of markdown lines.
    """

    def _fmt(val: Any, signed: bool = False) -> str:
        if isinstance(val, float):
            return f"{val:+.4f}" if signed else f"{val:.4f}"
        if val is None:
            return "N/A"
        return str(val)

    def _pct(val: Any) -> str:
        return f"{val * 100:.1f}%" if isinstance(val, (int, float)) else "N/A"

    def _pct_signed(val: Any) -> str:
        return f"{val * 100:+.1f}pp" if isinstance(val, (int, float)) else "N/A"

    baseline = cert_metrics.get("baseline", {})
    trained = cert_metrics.get("trained", {})
    deltas = cert_metrics.get("deltas", {})

    cert_config = evaluator.eval_config.get("certification", {})
    sigma = cert_config.get("sigma", 0.1)
    n = cert_config.get("n", 1000)
    n0 = cert_config.get("n0", 100)
    alpha = cert_config.get("alpha", 0.001)
    subset_size = cert_config.get("subset_size")

    def _samples_str(side: dict[str, Any]) -> str:
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


def generate_report(evaluator: Any, comparison: dict[str, Any]) -> str:
    """Generate a markdown report from a comparison dict.

    Args:
        evaluator: The Evaluator instance.
        comparison: Output of Evaluator.compare().

    Returns:
        Markdown-formatted comparison report.
    """

    def _fmt(val: Any, signed: bool = False) -> str:
        """Format a metric value: floats get .4f, others pass through."""
        if isinstance(val, float):
            return f"{val:+.4f}" if signed else f"{val:.4f}"
        return str(val)

    def _metric_ok(metric_data: dict[str, Any]) -> bool:
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
        lines.extend(_format_certification_section(evaluator, metrics["certification"]))

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

    if "calibration" in metrics and _metric_ok(metrics["calibration"]):
        r = metrics["calibration"]
        lines.extend(
            [
                "### Calibration",
                "",
                "| Metric | Baseline | Trained | Delta |",
                "|--------|----------|---------|-------|",
            ]
        )
        for key in ["ece_before", "ece_after", "optimal_temperature"]:
            bl = r.get("baseline", {}).get(key, "N/A")
            tr = r.get("trained", {}).get(key, "N/A")
            delta = r.get("deltas", {}).get(key, "N/A")
            lines.append(f"| {key} | {_fmt(bl)} | {_fmt(tr)} | {_fmt(delta, signed=True)} |")
        lines.append("")

    failure_cats = comparison.get("failure_categories")
    if failure_cats is None and "robustness" in metrics:
        failure_cats = metrics["robustness"].get("trained", {}).get("failure_categories")
        if failure_cats is None:
            failure_cats = metrics["robustness"].get("baseline", {}).get("failure_categories")

    if failure_cats is not None:
        lines.extend(["## Failure by Distortion Type", ""])
        if failure_cats and any(cat.get("count", 0) > 0 for cat in failure_cats.values()):
            lines.extend(
                [
                    "| Distortion | Failures | Failure Rate | Avg Confidence Δ |",
                    "|------------|----------|--------------|------------------|",
                ]
            )
            for dtype, cat in failure_cats.items():
                cnt = cat.get("count", 0)
                rate = cat.get("failure_rate", 0.0)
                avg_delta = cat.get("avg_confidence_delta", 0.0)
                lines.append(f"| {dtype} | {cnt} | {rate * 100:.1f}% | {avg_delta:.4f} |")
            lines.append("")
        else:
            lines.extend(["No failures detected across evaluated distortion types.", ""])

    return "\n".join(lines)


def save_report(
    evaluator: Any,
    comparison: dict[str, Any],
    filename: str = "evaluation_report.md",
) -> str:
    """Generate and save a markdown report.

    Args:
        evaluator: The Evaluator instance.
        comparison: Output of self.compare().
        filename: Name of the output file.

    Returns:
        Markdown-formatted comparison report.
    """
    report = generate_report(evaluator, comparison)
    path = os.path.join(evaluator.output_dir, filename)
    with open(path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", path)
    return report
