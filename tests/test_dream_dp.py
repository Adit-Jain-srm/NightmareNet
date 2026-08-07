"""Tests for Dream-phase differential privacy noise (issue #546)."""

from __future__ import annotations

import pytest

from nightmarenet.distortions.dream import (
    PrivacyAccountant,
    apply_dp_gaussian_noise,
    calibrate_gaussian_sigma,
    distort,
)


def test_sigma_decreases_as_epsilon_grows():
    s1 = calibrate_gaussian_sigma(1.0)
    s3 = calibrate_gaussian_sigma(3.0)
    s8 = calibrate_gaussian_sigma(8.0)
    assert s1 > s3 > s8 > 0


def test_calibrate_rejects_non_positive_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        calibrate_gaussian_sigma(0.0)


def test_dp_noise_is_deterministic_with_seed():
    text = "privacy preserving dream"
    a = apply_dp_gaussian_noise(text, 3.0, seed=42)
    b = apply_dp_gaussian_noise(text, 3.0, seed=42)
    c = apply_dp_gaussian_noise(text, 3.0, seed=7)
    assert a == b
    assert a != c


def test_distort_skips_dp_when_epsilon_null():
    text = "clean anchor sentence"
    baseline = distort(text, strength=0.0, seed=42, config={"dream_dp_epsilon": None})
    assert baseline == text


def test_distort_applies_dp_when_epsilon_set():
    text = "clean anchor sentence"
    out = distort(
        text,
        strength=0.0,
        seed=42,
        config={"dream_dp_epsilon": 1.0, "dream_dp_delta": 1e-5},
    )
    assert out != text


def test_privacy_accountant_accumulates_across_cycles():
    acc = PrivacyAccountant(budget=12.0)
    assert acc.spend(1.0, cycle=0, note="dream") == 1.0
    assert acc.spend(3.0, cycle=1, note="dream") == 4.0
    assert acc.spend(8.0, cycle=2, note="dream") == 12.0
    assert acc.remaining() == 0.0
    assert len(acc.events) == 3


def test_example_dp_config_loads():
    from nightmarenet.utils.config import load_config

    cfg = load_config("configs/examples/dp-training.yaml")
    assert cfg["distortion"]["dream_dp_epsilon"] == 3.0
    assert cfg["distortion"]["dream_dp_delta"] == 1e-5
