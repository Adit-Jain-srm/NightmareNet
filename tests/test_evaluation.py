"""Tests for evaluation metrics and comparison."""

import logging

import pytest
import torch
import torch.nn as nn

from nightmarenet.evaluation.evaluator import Evaluator, _bootstrap_ci

# ---------------------------------------------------------------------------------------
# Minimal fakes for certification integration tests (issue #161).
#
# Mirrors the fakes in tests/test_certification.py, kept self-contained here so this
# module doesn't depend on another test module's internals.
# ---------------------------------------------------------------------------------------


class _FakeModelConfig:
    num_labels = 2


class _CertNoiseSignClassifier(nn.Module):
    """Deterministic (given a seed) stand-in for a PreTrainedModel classifier."""

    def __init__(self, vocab_size: int = 20, hidden_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.zeros_(self.embedding.weight)
        self.config = _FakeModelConfig()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None):
        embeds = self.embedding(input_ids)
        means = embeds.mean(dim=(1, 2))
        logits = torch.stack([-means, means], dim=1)
        return type("Output", (), {"logits": logits})()


class _CertFakeTokenizer:
    """Minimal HF-tokenizer-shaped callable: maps any text to a fixed token sequence."""

    def __call__(self, text, truncation=True, max_length=128, return_tensors="pt"):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


class _CertListDataset:
    """Minimal HF-Dataset-shaped wrapper around a list of dicts."""

    def __init__(self, examples):
        self._examples = examples

    def __len__(self):
        return len(self._examples)

    def __iter__(self):
        return iter(self._examples)

    def shuffle(self, seed=42):
        return self

    def select(self, indices):
        return _CertListDataset([self._examples[i] for i in indices])


def _make_cert_evaluator(tmp_path, metrics, certification_config=None):
    config = {
        "evaluation": {
            "metrics": metrics,
            "output_dir": str(tmp_path),
        },
        "dataset": {"text_column": "text"},
        "model": {"max_length": 32},
    }
    if certification_config is not None:
        config["evaluation"]["certification"] = certification_config
    return Evaluator(
        model=_CertNoiseSignClassifier(),
        tokenizer=_CertFakeTokenizer(),
        config=config,
    )


def test_bootstrap_ci_identical():
    """Test that identical data produces p > 0.05 (not significant)."""
    baseline = [0.5, 0.6, 0.7, 0.8, 0.9]
    trained = [0.5, 0.6, 0.7, 0.8, 0.9]

    result = _bootstrap_ci(baseline, trained, alpha=0.05)

    assert result["delta_mean"] == pytest.approx(0.0, abs=1e-6)
    assert result["p_value"] > 0.05
    assert result["significant"] is False
    assert result["method"] == "bootstrap_ci"


def test_bootstrap_ci_different():
    """Test that clearly different data produces p < 0.05 (significant)."""
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5]
    trained = [0.6, 0.7, 0.8, 0.9, 1.0]

    result = _bootstrap_ci(baseline, trained, alpha=0.05)

    assert result["delta_mean"] > 0
    assert result["p_value"] < 0.05
    assert result["significant"] is True
    assert result["ci_lower"] > 0  # CI should be entirely positive


def test_bootstrap_ci_insufficient_data():
    """Test that insufficient data returns not significant result."""
    baseline = [0.5]
    trained = [0.6]

    result = _bootstrap_ci(baseline, trained, alpha=0.05)

    assert result["significant"] is False
    assert result["method"] == "insufficient_data"
    assert result["p_value"] == 1.0


def test_bootstrap_ci_custom_alpha():
    """Test that custom alpha threshold is respected."""
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5]
    trained = [0.6, 0.7, 0.8, 0.9, 1.0]

    result = _bootstrap_ci(baseline, trained, alpha=0.01)

    assert result["alpha"] == 0.01
    assert result["significant"] is True


# ---------------------------------------------------------------------------------------
# Certification integration (issue #161)
# ---------------------------------------------------------------------------------------


class TestCertificationIntegration:
    def test_certification_not_run_when_not_enabled(self, tmp_path):
        """Existing behavior is unchanged when certification isn't opted into: no
        regression, and the key is simply absent rather than erroring."""
        evaluator = _make_cert_evaluator(tmp_path, metrics=["recall"])
        dataset = _CertListDataset([{"text": "a", "label": 0}])

        results = evaluator.evaluate(clean_dataloader=None, base_dataset=dataset)

        assert "certification" not in results

    def test_certification_skipped_without_base_dataset(self, tmp_path):
        """Even when enabled, certification needs a dataset to certify -- it should be
        skipped (not error) if base_dataset isn't provided, same as robustness."""
        evaluator = _make_cert_evaluator(tmp_path, metrics=["certification"])

        results = evaluator.evaluate(clean_dataloader=None, base_dataset=None)

        assert "certification" not in results

    def test_certification_produces_expected_output_keys(self, tmp_path):
        """Integration test: evaluate() with certification in metrics list produces the
        expected output keys (issue #161 acceptance criteria)."""
        evaluator = _make_cert_evaluator(
            tmp_path,
            metrics=["certification"],
            certification_config={
                "n": 20,
                "n0": 10,
                "sigma": 0.1,
                "alpha": 0.05,
                "batch_size": 10,
                "subset_size": 2,
                "budget": None,
            },
        )
        dataset = _CertListDataset(
            [
                {"text": "example one", "label": 0},
                {"text": "example two", "label": 1},
            ]
        )

        results = evaluator.evaluate(clean_dataloader=None, base_dataset=dataset)

        assert "certification" in results
        cert = results["certification"]
        assert set(cert.keys()) == {
            "certified_radius_mean",
            "certified_radius_median",
            "certification_abstain_rate",
            "certified_accuracy",
            "samples_certified",
            "budget_exceeded",
        }
        assert cert["samples_certified"] == 2
        assert 0.0 <= cert["certification_abstain_rate"] <= 1.0
        assert cert["budget_exceeded"] is False

    def test_certification_config_read_from_namespace(self, tmp_path, monkeypatch):
        """Config values must be read from evaluation.certification, not hardcoded
        defaults -- verified by checking certify_dataset receives them."""
        captured = {}
        real_certify_dataset = __import__(
            "nightmarenet.evaluation.evaluator", fromlist=["certify_dataset"]
        ).certify_dataset

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real_certify_dataset(*args, **kwargs)

        monkeypatch.setattr("nightmarenet.evaluation.evaluator.certify_dataset", spy)

        evaluator = _make_cert_evaluator(
            tmp_path,
            metrics=["certification"],
            certification_config={
                "n": 15,
                "n0": 7,
                "sigma": 0.33,
                "alpha": 0.02,
                "batch_size": 5,
                "subset_size": 1,
                "budget": None,
            },
        )
        dataset = _CertListDataset([{"text": "a", "label": 0}])

        evaluator.evaluate(clean_dataloader=None, base_dataset=dataset)

        assert captured["n"] == 15
        assert captured["n0"] == 7
        assert captured["sigma"] == 0.33
        assert captured["alpha"] == 0.02
        assert captured["batch_size"] == 5
        assert captured["subset_size"] == 1

    def test_certification_budget_reduces_n_and_flags_exceeded(self, tmp_path):
        """Budget control: when n * subset_size exceeds budget, n is reduced
        proportionally, budget_exceeded is flagged, and a warning is logged."""
        evaluator = _make_cert_evaluator(
            tmp_path,
            metrics=["certification"],
            certification_config={
                "n": 1000,
                "subset_size": 10,
                "budget": 50,  # 1000 * 10 = 10000 >> 50
                "sigma": 0.1,
                "alpha": 0.05,
                "batch_size": 10,
            },
        )
        dataset = _CertListDataset([{"text": "a", "label": 0} for _ in range(10)])

        # caplog.at_level(..., logger=name) is *not* used here: whether it attaches its
        # capture handler directly to the named logger or only adjusts that logger's
        # level (leaving the handler on the root logger) differs across pytest versions,
        # and nightmarenet/utils/logging_config.setup_logging() sets propagate=False on
        # the "nightmarenet" logger the first time it's called anywhere in the test
        # session -- so whether this test passes can depend on both the pytest version
        # resolved for a given Python version and on test execution order elsewhere in
        # the suite. Attaching a plain logging.Handler directly to the evaluator's own
        # logger sidesteps both: a logger's own handlers always fire regardless of its
        # (or an ancestor's) propagate setting, independent of caplog/pytest internals.
        target_logger = logging.getLogger("nightmarenet.evaluation.evaluator")
        records: list[logging.LogRecord] = []

        class _ListHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _ListHandler(level=logging.WARNING)
        original_level = target_logger.level
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.WARNING)
        try:
            results = evaluator.evaluate(clean_dataloader=None, base_dataset=dataset)
        finally:
            target_logger.removeHandler(handler)
            target_logger.setLevel(original_level)

        cert = results["certification"]
        assert cert["budget_exceeded"] is True
        assert any("budget" in record.getMessage().lower() for record in records)

    def test_certification_within_budget_not_flagged(self, tmp_path):
        """When n * subset_size is within budget, no reduction happens."""
        evaluator = _make_cert_evaluator(
            tmp_path,
            metrics=["certification"],
            certification_config={
                "n": 10,
                "subset_size": 2,
                "budget": 1000,  # comfortably above 10 * 2 = 20
                "sigma": 0.1,
                "alpha": 0.05,
                "batch_size": 10,
            },
        )
        dataset = _CertListDataset([{"text": "a", "label": 0}, {"text": "b", "label": 1}])

        results = evaluator.evaluate(clean_dataloader=None, base_dataset=dataset)

        assert results["certification"]["budget_exceeded"] is False

    def test_compare_includes_certification_results(self, tmp_path):
        """compare() folds certification into the comparison dict alongside other
        metrics, using the same generic baseline/trained/deltas shape."""
        evaluator = _make_cert_evaluator(tmp_path, metrics=["certification"])

        baseline_results = {
            "label": "baseline",
            "certification": {
                "certified_radius_mean": 0.1,
                "certified_radius_median": 0.1,
                "certification_abstain_rate": 0.5,
                "certified_accuracy": 0.6,
                "samples_certified": 10,
                "budget_exceeded": False,
            },
        }
        trained_results = {
            "label": "trained",
            "certification": {
                "certified_radius_mean": 0.2,
                "certified_radius_median": 0.2,
                "certification_abstain_rate": 0.3,
                "certified_accuracy": 0.8,
                "samples_certified": 10,
                "budget_exceeded": True,
            },
        }

        comparison = evaluator.compare(baseline_results, trained_results)

        assert "certification" in comparison["metrics"]
        cert_comparison = comparison["metrics"]["certification"]
        assert cert_comparison["baseline"]["certified_radius_mean"] == 0.1
        assert cert_comparison["trained"]["certified_radius_mean"] == 0.2
        assert cert_comparison["deltas"]["certified_radius_mean"] == pytest.approx(0.1)
        assert cert_comparison["deltas"]["certification_abstain_rate"] == pytest.approx(-0.2)
        # bool is a subtype of int in Python -- budget_exceeded must not sneak into the
        # numeric deltas (True - False == 1 would be a meaningless "delta").
        assert "budget_exceeded" not in cert_comparison["deltas"]
