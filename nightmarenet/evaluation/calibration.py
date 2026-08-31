"""Model confidence calibration metrics and temperature scaling.

Implements Expected Calibration Error (ECE) and Temperature Scaling in compliance
with safety and reliability guidelines.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Weighted average of |accuracy - confidence| across M bins.

    Args:
        confidences: Max softmax probabilities of predictions, shape (N,).
        predictions: Predicted class indices, shape (N,).
        labels: Ground truth class indices, shape (N,).
        n_bins: Number of bins (default 15).

    Returns:
        ECE as a float value between 0 and 1.
    """
    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0

    # Ensure inputs are numpy arrays
    confidences = np.asarray(confidences)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = boundaries[i]
        bin_upper = boundaries[i + 1]

        # Samples falling into the current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin |= confidences == bin_lower

        bin_size = np.sum(in_bin)

        if bin_size > 0:
            # Average accuracy in this bin
            bin_acc = np.mean(predictions[in_bin] == labels[in_bin])
            # Average confidence in this bin
            bin_conf = np.mean(confidences[in_bin])

            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def reliability_diagram_data(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, list[Any]]:
    """Compute reliability diagram data for plotting.

    Args:
        confidences: Max softmax probabilities of predictions, shape (N,).
        predictions: Predicted class indices, shape (N,).
        labels: Ground truth class indices, shape (N,).
        n_bins: Number of bins (default 15).

    Returns:
        Dict with keys: "bin_confidences", "bin_accuracies", "bin_counts"
    """
    confidences = np.asarray(confidences)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = boundaries[i]
        bin_upper = boundaries[i + 1]

        # Samples falling into the current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin |= confidences == bin_lower

        bin_size = int(np.sum(in_bin))
        bin_counts.append(bin_size)

        if bin_size > 0:
            bin_accuracies.append(float(np.mean(predictions[in_bin] == labels[in_bin])))
            bin_confidences.append(float(np.mean(confidences[in_bin])))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


class TemperatureScaler:
    """Softmax temperature scaling optimizer.

    Scales logits by z / T where T is optimized via negative log-likelihood (NLL) minimization.
    """

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Find optimal temperature T on validation logits.

        Optimized using L-BFGS.

        Args:
            logits: Unscaled logit tensors of shape (N, num_classes).
            labels: Ground truth label tensors of shape (N,).

        Returns:
            The optimized temperature scalar T.
        """
        # Detach inputs and move to device
        logits = logits.detach().float()
        labels = labels.detach().long()

        # Define temperature parameter (initialized to 1.0)
        # Using a Parameter so PyTorch autograd tracks it
        temperature = nn.Parameter(torch.ones(1, device=logits.device))
        nll_criterion = nn.CrossEntropyLoss()

        # Define optimizer. L-BFGS requires a closure that re-evaluates the model
        # and returns loss.
        # Strong Wolfe line search is recommended for faster convergence
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def eval_loss() -> torch.Tensor:
            optimizer.zero_grad()
            # Enforce temperature constraint: clamp to a minimum value of 0.01
            # to prevent division by zero.
            temp = torch.clamp(temperature, min=0.01)
            loss = nll_criterion(logits / temp, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Retrieve and store the optimized temperature
        self.temperature = float(torch.clamp(temperature, min=0.01).item())
        logger.info(
            "Temperature scaling optimization complete. Optimal T: %.4f",
            self.temperature,
        )
        return self.temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrate logits using the optimized temperature parameter.

        Args:
            logits: Logit tensors of shape (N, num_classes) or (batch_size, num_classes).

        Returns:
            Logit tensors scaled by 1/T.
        """
        return logits / self.temperature
