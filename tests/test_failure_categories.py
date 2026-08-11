"""Tests for failure categorization by distortion type in evaluation reports (Issue #372)."""

import json

from nightmarenet.evaluation.evaluator import Evaluator
from nightmarenet.evaluation.metrics import (
    categorize_failures_by_distortion,
)


def test_empty_input():
    """Test that empty or None input produces an empty dictionary."""
    assert categorize_failures_by_distortion([]) == {}
    assert categorize_failures_by_distortion(None) == {}


def test_grouping_and_counts():
    """Test grouping failed samples by distortion type and counting them."""
    records = [
        {"distortion_type": "gaussian_noise", "confidence_delta": 0.2, "is_failure": True},
        {"distortion_type": "gaussian_noise", "confidence_delta": 0.4, "is_failure": True},
        {"distortion_type": "contrast_drop", "confidence_delta": 0.1, "is_failure": True},
    ]
    res = categorize_failures_by_distortion(records)
    assert "gaussian_noise" in res
    assert "contrast_drop" in res
    assert res["gaussian_noise"]["count"] == 2
    assert res["contrast_drop"]["count"] == 1


def test_average_confidence_delta():
    """Test average confidence delta calculation."""
    records = [
        {"distortion_type": "blur", "confidence_delta": 0.1, "is_failure": True},
        {"distortion_type": "blur", "confidence_delta": 0.3, "is_failure": True},
        {"distortion_type": "blur", "confidence_delta": 0.5, "is_failure": True},
    ]
    res = categorize_failures_by_distortion(records)
    assert res["blur"]["avg_confidence_delta"] == 0.3


def test_failure_rate():
    """Test failure rate computation when total_samples is specified."""
    records = [
        {"distortion_type": "blur", "confidence_delta": 0.2, "is_failure": True},
        {"distortion_type": "blur", "confidence_delta": 0.4, "is_failure": True},
    ]
    totals = {"blur": 10}
    res = categorize_failures_by_distortion(records, total_samples_per_distortion=totals)
    assert res["blur"]["failure_rate"] == 0.2
    assert res["blur"]["count"] == 2


def test_deterministic_sorting():
    """Test sorting: highest failure rate, then count, then name."""
    records = [
        {
            "distortion_type": "alpha",
            "confidence_delta": 0.1,
            "is_failure": True,
            "total_samples": 2,
        },
        {
            "distortion_type": "beta",
            "confidence_delta": 0.1,
            "is_failure": True,
            "total_samples": 4,
        },
        {
            "distortion_type": "beta",
            "confidence_delta": 0.2,
            "is_failure": True,
            "total_samples": 4,
        },
        {
            "distortion_type": "gamma",
            "confidence_delta": 0.3,
            "is_failure": True,
            "total_samples": 1,
        },
        {
            "distortion_type": "delta",
            "confidence_delta": 0.1,
            "is_failure": True,
            "total_samples": 2,
        },
    ]
    res = categorize_failures_by_distortion(records)
    keys = list(res.keys())
    assert keys == ["gamma", "beta", "alpha", "delta"]


def test_text_distortions():
    """Test categorization with text distortion record aliases."""
    records = [
        {"distortion": "typo_insertion", "delta": 0.15, "failed": True},
        {"distortion_name": "synonym_swap", "confidence_drop": 0.25, "correct": False},
    ]
    res = categorize_failures_by_distortion(records)
    assert "typo_insertion" in res
    assert "synonym_swap" in res
    assert res["typo_insertion"]["count"] == 1
    assert res["synonym_swap"]["count"] == 1


def test_vision_distortions():
    """Test categorization with vision distortion records."""
    records = [
        {"type": "rotation", "confidence_delta": 0.5, "is_failure": True},
        {"type": "color_jitter", "confidence_delta": 0.1, "is_failure": False},
    ]
    res = categorize_failures_by_distortion(records)
    assert "rotation" in res
    assert "color_jitter" in res
    assert res["rotation"]["count"] == 1
    assert res["color_jitter"]["count"] == 0
    assert res["color_jitter"]["failure_rate"] == 0.0


def test_json_serialization(tmp_path):
    """Test JSON serialization of failure_categories."""
    failure_cats = {"blur": {"count": 2, "failure_rate": 0.2, "avg_confidence_delta": 0.3}}
    evaluator = Evaluator(
        model=None,
        tokenizer=None,
        config={"evaluation": {"output_dir": str(tmp_path)}},
    )
    results = {
        "label": "test_run",
        "failure_categories": failure_cats,
    }
    evaluator.save_results(results, "test_out.json")
    out_file = tmp_path / "test_out.json"
    assert out_file.exists()
    loaded = json.loads(out_file.read_text())
    assert "failure_categories" in loaded
    assert loaded["failure_categories"]["blur"]["count"] == 2


def test_markdown_rendering(tmp_path):
    """Test Markdown report generation with failure categories and empty failure state."""
    evaluator = Evaluator(
        model=None,
        tokenizer=None,
        config={"evaluation": {"output_dir": str(tmp_path)}},
    )

    comparison_with_failures = {
        "baseline_label": "base",
        "trained_label": "trained",
        "failure_categories": {
            "gaussian_noise": {"count": 10, "failure_rate": 0.8, "avg_confidence_delta": 0.25},
            "blur": {"count": 2, "failure_rate": 0.2, "avg_confidence_delta": 0.1},
        },
        "metrics": {},
    }
    report = evaluator.generate_report(comparison_with_failures)
    assert "## Failure by Distortion Type" in report
    assert "| Distortion | Failures | Failure Rate | Avg Confidence Δ |" in report
    assert "| gaussian_noise | 10 | 80.0% | 0.2500 |" in report

    comparison_no_failures = {
        "baseline_label": "base",
        "trained_label": "trained",
        "failure_categories": {},
        "metrics": {},
    }
    empty_report = evaluator.generate_report(comparison_no_failures)
    assert "## Failure by Distortion Type" in empty_report
    assert "No failures detected across evaluated distortion types." in empty_report


def test_empty_failure_categories():
    """Test that an empty dict for failure categories is preserved."""
    from nightmarenet.evaluation.evaluator import Evaluator
    evaluator = Evaluator(model=None, tokenizer=None, config={})
    trained = {"failure_categories": {}}
    baseline = {"failure_categories": {"blur": {"count": 5}}}
    result = evaluator.compare(baseline, trained)
    assert result.get("failure_categories") == {}


def test_missing_key_failure():
    """Test that records without explicit failure flags default to False."""
    from nightmarenet.evaluation.metrics import categorize_failures_by_distortion
    sample = {"distortion_type": "blur", "confidence_delta": 0.1}
    res = categorize_failures_by_distortion([sample])
    # The count should be 0 because it's not a failure
    assert res["blur"]["count"] == 0
    assert res["blur"]["failure_rate"] == 0.0


def test_auto_wiring():
    """Test that failure_categories is populated in robustness_score."""
    from unittest.mock import MagicMock, patch

    from nightmarenet.evaluation.metrics import robustness_score

    class DummyDataset:
        def map(self, *args, **kwargs): return self
        def set_format(self, *args, **kwargs): return self
        def __iter__(self): yield {"text": "test"}
        def __len__(self): return 1
        column_names = ["text"]

    class MockTokenizer:
        def __call__(self, text, **kwargs):
            import torch
            return {"input_ids": torch.tensor([[1, 2, 3]])}
        pad_token_id = 0

    mock_model = MagicMock()

    with patch("nightmarenet.evaluation.metrics.compute_perplexity") as mock_ppl, \
         patch("nightmarenet.evaluation.metrics.classification_metrics") as mock_cls:

        # Mock responses to bypass actual inference
        mock_ppl.return_value = {"perplexity": 2.0, "per_sample_ppls": [2.0]}
        # We need a failure, so orig_preds != dist_preds. Since it is called
        # twice (clean, then distorted), we use side_effect to return different predictions.
        mock_cls.side_effect = [
            {"per_sample_preds": [0], "per_sample_confs": [0.9]},  # Clean dataset
            {"per_sample_preds": [1], "per_sample_confs": [0.4]}   # Distorted dataset
        ]

        results = robustness_score(
            model=mock_model,
            base_dataset=DummyDataset(),
            tokenizer=MockTokenizer(),
            distortion_fn=lambda x, **kwargs: x,
            strengths=[0.1],
            export_failures=True,
            failure_records=None,
        )
        assert "failure_categories" in results
        # By default, metrics.py export logic uses "text_distortion" as distortion_type
        assert results["failure_categories"]["text_distortion"]["count"] == 1
