import numpy as np
import pytest
import torch

from nightmarenet.evaluation.calibration import (
    TemperatureScaler,
    compute_ece,
    reliability_diagram_data,
)


def test_compute_ece_perfect_calibration():
    # Construct a dataset with perfect calibration
    # E.g. in bin [0.0, 0.1] we have 10 elements, 1 correct (acc=0.1, conf=0.1)
    # In bin [0.8, 0.9] we have 10 elements, 8 correct (acc=0.8, conf=0.8)
    confidences = []
    predictions = []
    labels = []

    # Bin 1: conf = 0.1, accuracy = 0.1
    for i in range(10):
        confidences.append(0.1)
        predictions.append(1 if i == 0 else 0)
        labels.append(1)

    # Bin 2: conf = 0.8, accuracy = 0.8
    for i in range(10):
        confidences.append(0.8)
        predictions.append(1 if i < 8 else 0)
        labels.append(1)

    ece = compute_ece(confidences, predictions, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_compute_ece_overconfident():
    # Construct an overconfident dataset
    # E.g. confidence is always 0.95, but accuracy is 0.5 (random guessing)
    n_samples = 100
    confidences = np.full(n_samples, 0.95)
    predictions = np.ones(n_samples, dtype=int)
    # 50% accurate
    labels = np.array([1 if i % 2 == 0 else 0 for i in range(n_samples)])

    ece = compute_ece(confidences, predictions, labels, n_bins=10)
    # Expected ECE is |accuracy - confidence| = |0.5 - 0.95| = 0.45
    assert ece == pytest.approx(0.45, abs=1e-2)


def test_reliability_diagram_data():
    confidences = np.array([0.1, 0.2, 0.5, 0.85, 0.9])
    predictions = np.array([1, 1, 1, 1, 1])
    labels = np.array([1, 0, 1, 0, 1])

    data = reliability_diagram_data(confidences, predictions, labels, n_bins=10)
    assert "bin_confidences" in data
    assert "bin_accuracies" in data
    assert "bin_counts" in data

    assert len(data["bin_confidences"]) == 10
    assert len(data["bin_accuracies"]) == 10
    assert len(data["bin_counts"]) == 10
    assert sum(data["bin_counts"]) == 5


def test_temperature_scaler_fit_overconfident():
    # Generate synthetic overconfident logits and labels
    # Model assigns high logit value (e.g. 5.0 vs 0.0) to class 1,
    # but labels are 50% class 1 and 50% class 0.
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 200
    logits = torch.zeros((n_samples, 2))
    logits[:, 1] = 4.0  # High confidence predictions for class 1
    labels = torch.randint(0, 2, (n_samples,))

    scaler = TemperatureScaler()
    optimal_t = scaler.fit(logits, labels)

    # Since logits are highly overconfident (yielding probability close to 1.0 for wrong answers),
    # the optimal temperature T should scale them down, hence T should be > 1.0.
    assert optimal_t > 1.0

    # Test calibrate function
    calibrated_logits = scaler.calibrate(logits)
    assert calibrated_logits.shape == logits.shape
    # Check that scaled logits are smaller
    assert float(calibrated_logits[0, 1].item()) < 4.0


def test_temperature_scaler_reduces_ece():
    # Create a synthetic dataset representing model output on a validation set
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 400
    # True logits are slightly informative, but overconfident
    true_classes = torch.randint(0, 2, (n_samples,))
    logits = torch.randn((n_samples, 2)) * 0.5
    # Boost correct class by 2.0 to make model somewhat accurate (confs ~ 0.88),
    # but make it overconfident compared to true accuracy of ~0.80.
    for i in range(n_samples):
        logits[i, true_classes[i]] += 2.0

    # Add a calibration split (validation set) and a test split
    calib_logits = logits[:200]
    calib_labels = true_classes[:200]
    test_logits = logits[200:]
    test_labels = true_classes[200:]

    # ECE before scaling on test split
    probs_before = torch.softmax(test_logits, dim=-1)
    conf_before, preds_before = probs_before.max(dim=-1)
    ece_before = compute_ece(
        conf_before.numpy(), preds_before.numpy(), test_labels.numpy(), n_bins=15
    )

    # Fit temperature scaler
    scaler = TemperatureScaler()
    optimal_t = scaler.fit(calib_logits, calib_labels)

    # ECE after scaling on test split
    calib_test_logits = scaler.calibrate(test_logits)
    probs_after = torch.softmax(calib_test_logits, dim=-1)
    conf_after, preds_after = probs_after.max(dim=-1)
    ece_after = compute_ece(
        conf_after.numpy(), preds_after.numpy(), test_labels.numpy(), n_bins=15
    )

    # Verify that calibration successfully reduced ECE
    assert ece_after < ece_before
    assert optimal_t > 0.0


def test_calibration_edge_cases():
    # Empty inputs
    assert compute_ece([], [], []) == 0.0

    # Single sample
    assert compute_ece([0.5], [1], [1], n_bins=1) == 0.5
