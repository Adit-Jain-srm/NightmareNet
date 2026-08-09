"""Degradation curves calculation."""

from __future__ import annotations


def calculate_degradation_curves(model_results: dict[str, dict]) -> dict[str, list[dict]]:
    """Extract degradation curves from evaluation results per model.

    Args:
        model_results: A mapping from model name to its evaluation results.
            Expected format for each model's results:
            {
                "distortion_type": {
                    "strengths": [0.1, ...],
                    "accuracies": [0.95, ...]
                }
            }

    Returns:
        A dictionary mapping model names to a list of data points representing
        the degradation curve (strength vs robustness).
    """
    curves = {}
    for model_name, results in model_results.items():
        curve = []
        for distortion, dist_results in results.items():
            strengths = dist_results.get("strengths", [])
            accuracies = dist_results.get("accuracies", [])

            for s, acc in zip(strengths, accuracies):
                curve.append({"distortion": distortion, "strength": s, "robustness": acc})
        curves[model_name] = curve
    return curves
