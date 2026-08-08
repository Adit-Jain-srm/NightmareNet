"""Model comparison logic (baseline vs. trained)."""

from __future__ import annotations

from typing import Any

from nightmarenet.evaluation.evaluator.runners import _bootstrap_ci


def compare_results(
    evaluator: Any,
    baseline_results: dict[str, Any],
    trained_results: dict[str, Any],
) -> dict[str, Any]:
    """Produce a comparison between baseline and trained model results.

    Args:
        evaluator: The Evaluator instance.
        baseline_results: Evaluation results from the baseline model.
        trained_results: Evaluation results from the DreamPhase-trained model.

    Returns:
        Dict with side-by-side comparison for each metric, including
        statistical significance testing where applicable.
    """
    comparison: dict[str, Any] = {
        "baseline_label": baseline_results.get("label", "baseline"),
        "trained_label": trained_results.get("label", "dreamphase"),
        "metrics": {},
    }

    for metric_name in evaluator.enabled_metrics:
        baseline = baseline_results.get(metric_name, {})
        trained = trained_results.get(metric_name, {})

        if not baseline and not trained:
            continue

        metric_comparison: dict[str, Any] = {
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
                    alpha=evaluator.significance_alpha,
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

    failure_categories = (
        trained_results.get("failure_categories")
        or trained_results.get("robustness", {}).get("failure_categories")
        or baseline_results.get("failure_categories")
        or baseline_results.get("robustness", {}).get("failure_categories")
    )
    if failure_categories is not None:
        comparison["failure_categories"] = failure_categories

    return comparison
