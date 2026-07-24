from nightmarenet.evaluation.evaluator import Evaluator
from nightmarenet.evaluation.metrics import (
    build_delta_distribution,
    compute_confidence_delta,
    rank_failures,
    truncate_preview,
)


def test_compute_confidence_delta():
    assert abs(compute_confidence_delta(0.9, 0.4) - 0.5) < 1e-6
    assert abs(compute_confidence_delta(0.5, 0.5) - 0.0) < 1e-6
    assert abs(compute_confidence_delta(0.4, 0.9) - (-0.5)) < 1e-6


def test_truncate_preview():
    text = (
        "This is a very long text that should definitely be "
        "truncated because it exceeds fifty characters easily."
    )
    truncated = truncate_preview(text, max_len=50)
    assert len(truncated) == 53  # 50 + len("...")
    assert truncated.endswith("...")

    short = "Short text"
    assert truncate_preview(short, max_len=50) == short


def test_rank_failures():
    failures = [
        {"sample_index": 0, "confidence_delta": 0.1},
        {"sample_index": 1, "confidence_delta": 0.9},
        {"sample_index": 2, "confidence_delta": 0.5},
        {"sample_index": 3, "confidence_delta": -0.1},
    ]
    ranked = rank_failures(failures)
    assert ranked[0]["sample_index"] == 1
    assert ranked[1]["sample_index"] == 2
    assert ranked[2]["sample_index"] == 0
    assert ranked[3]["sample_index"] == 3


def test_build_delta_distribution():
    failures = [
        {"confidence_delta": 0.05},  # 0_10
        {"confidence_delta": 0.10},  # 10_25 (boundary)
        {"confidence_delta": 0.15},  # 10_25
        {"confidence_delta": 0.25},  # 25_50 (boundary)
        {"confidence_delta": 0.40},  # 25_50
        {"confidence_delta": 0.50},  # 50_plus (boundary)
        {"confidence_delta": 0.80},  # 50_plus
        {"confidence_delta": -0.1},  # ignored
        {"confidence_delta": 0.0},  # ignored
    ]
    dist = build_delta_distribution(failures)
    assert dist["0_10"] == 1
    assert dist["10_25"] == 2
    assert dist["25_50"] == 2
    assert dist["50_plus"] == 2


def test_lm_confidence_conversion():
    clean_ppl = 2.0
    dist_ppl = 5.0
    clean_conf = 1.0 / clean_ppl
    dist_conf = 1.0 / dist_ppl
    delta = compute_confidence_delta(clean_conf, dist_conf)
    assert abs(delta - 0.3) < 1e-6


def test_markdown_report_generation():
    evaluator = Evaluator(
        model=None, tokenizer=None, config={"evaluation": {"metrics": ["robustness"]}}
    )

    comparison = {
        "baseline_label": "baseline",
        "trained_label": "trained",
        "metrics": {
            "robustness": {
                "trained": {
                    "auc_robustness": 0.85,
                    "top_failures": [
                        {
                            "sample_index": 12,
                            "preview": "Test preview",
                            "clean_confidence": 0.95,
                            "distorted_confidence": 0.35,
                            "confidence_delta": 0.60,
                        }
                    ],
                    "delta_distribution": {"0_10": 0, "10_25": 0, "25_50": 0, "50_plus": 1},
                }
            }
        },
    }

    report = evaluator.generate_report(comparison)
    assert "## Confidence Delta Analysis" in report
    assert "### Top 10 Most Vulnerable Samples" in report
    assert "| 12 | Test preview | 0.9500 | 0.3500 | 0.6000 |" in report
    assert "- **50%+ drop**: 1" in report


def test_empty_failures_report():
    evaluator = Evaluator(
        model=None, tokenizer=None, config={"evaluation": {"metrics": ["robustness"]}}
    )

    comparison = {
        "baseline_label": "baseline",
        "trained_label": "trained",
        "metrics": {
            "robustness": {
                "trained": {
                    "auc_robustness": 0.99,
                    "top_failures": [],
                    "delta_distribution": {"0_10": 0, "10_25": 0, "25_50": 0, "50_plus": 0},
                }
            }
        },
    }

    report = evaluator.generate_report(comparison)
    assert "## Confidence Delta Analysis" in report
    assert "### Top 10 Most Vulnerable Samples" in report
    assert "### Severity Distribution" in report
    assert "- **50%+ drop**: 0" in report
