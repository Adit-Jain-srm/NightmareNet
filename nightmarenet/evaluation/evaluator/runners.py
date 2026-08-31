"""Batch processing, accumulation loops, inference execution, and metric runners."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from nightmarenet.evaluation.calibration import (
    TemperatureScaler,
    compute_ece,
    reliability_diagram_data,
)
from nightmarenet.evaluation.certification import certify_dataset

logger = logging.getLogger(__name__)


def _bootstrap_ci(
    baseline: list[float],
    trained: list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
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


def run_certification(
    model: Any,
    tokenizer: Any,
    device: str,
    base_dataset: Any,
    eval_config: dict[str, Any],
    config: dict[str, Any],
    certify_fn: Any = None,
) -> dict[str, Any]:
    """Run certified-robustness verification (randomized smoothing) on a dataset.

    Budget control: `budget` caps the total forward passes for the *estimation*
    stage across the whole run (n * subset_size). If the configured n and
    subset_size would exceed it, n is reduced proportionally (subset_size is left
    alone, since it controls how many samples get any signal at all) and a warning
    is logged.

    Args:
        model: Model to certify.
        tokenizer: Tokenizer for the model.
        device: Device to run certification on.
        base_dataset: Dataset to certify.
        eval_config: Evaluation configuration dictionary.
        config: Full configuration dictionary.
        certify_fn: Optional callable replacing certify_dataset (for mocking/spying).

    Returns:
        Dict with certification results.
    """
    if certify_fn is None:
        certify_fn = certify_dataset

    cert_config = eval_config.get("certification", {})
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

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    cert_result = certify_fn(
        model,
        tokenizer,
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
        device=device,
    )

    return {
        "certified_radius_mean": cert_result["certified_radius_mean"],
        "certified_radius_median": cert_result["certified_radius_median"],
        "certification_abstain_rate": cert_result["certification_abstain_rate"],
        "certified_accuracy": cert_result["certified_accuracy"],
        "samples_certified": cert_result["n_samples"],
        "budget_exceeded": budget_exceeded,
    }


def run_calibration(
    model: Any,
    device: str,
    dataloader: DataLoader,
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    """Run ECE computation and temperature scaling on the dataloader.

    Args:
        model: Model to calibrate.
        device: Device to run calibration on.
        dataloader: DataLoader for the dataset.
        eval_config: Evaluation configuration dictionary.

    Returns:
        Dict with calibration results.
    """
    model.eval()
    all_logits = []
    all_labels = []

    try:
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                if "labels" in batch:
                    labels = batch["labels"]
                elif "label" in batch:
                    labels = batch["label"]
                else:
                    continue

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
    except Exception as e:
        logger.error("Error during calibration metrics forward pass: %s", e)
        raise RuntimeError("Error during calibration metrics forward pass") from e

    if not all_logits:
        raise ValueError("No logits or labels found in dataloader for calibration.")

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    n_samples = len(all_logits)
    if n_samples < 2:
        raise ValueError(f"Too few samples for calibration split: {n_samples}")

    # Split into calibration and test set (50/50 split)
    split_idx = n_samples // 2
    calib_logits = all_logits[:split_idx]
    calib_labels = all_labels[:split_idx]
    test_logits = all_logits[split_idx:]
    test_labels = all_labels[split_idx:]

    calibration_cfg = eval_config.get("calibration", {})
    ece_bins = calibration_cfg.get("ece_bins", 15)
    use_scaling = calibration_cfg.get("temperature_scaling", True)

    # Fit TemperatureScaler if enabled
    optimal_temp = 1.0
    if use_scaling:
        scaler = TemperatureScaler()
        optimal_temp = scaler.fit(calib_logits, calib_labels)
        calib_test_logits = scaler.calibrate(test_logits)
    else:
        calib_test_logits = test_logits

    # Compute uncalibrated ECE on test split
    probs_before = torch.softmax(test_logits, dim=-1)
    conf_before, preds_before = probs_before.max(dim=-1)

    ece_before = compute_ece(
        conf_before.numpy(), preds_before.numpy(), test_labels.numpy(), n_bins=ece_bins
    )

    # Compute calibrated/final ECE and reliability data on test split
    probs_after = torch.softmax(calib_test_logits, dim=-1)
    conf_after, preds_after = probs_after.max(dim=-1)

    ece_after = compute_ece(
        conf_after.numpy(), preds_after.numpy(), test_labels.numpy(), n_bins=ece_bins
    )

    rel_data = reliability_diagram_data(
        conf_after.numpy(), preds_after.numpy(), test_labels.numpy(), n_bins=ece_bins
    )

    return {
        "ece_before": ece_before,
        "ece_after": ece_after,
        "optimal_temperature": optimal_temp,
        "bin_confidences": rel_data["bin_confidences"],
        "bin_accuracies": rel_data["bin_accuracies"],
        "bin_counts": rel_data["bin_counts"],
    }
