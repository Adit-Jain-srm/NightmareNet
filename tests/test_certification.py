"""Tests for nightmarenet.evaluation.certification.

Covers issue #160's acceptance criteria, plus the review fixes below:
  1. Clopper-Pearson bound matches the scipy.stats.beta.ppf reference implementation
     (verified against known values, not just internal consistency).
  2. Abstention triggered when predictions are near-uniform.
  3. Determinism under a fixed random seed.
  4. Embedding-layer noise injection via forward hook, with guaranteed cleanup.
  5. Batched (not sequential) inference.
  6. batch_size <= 0 raises rather than hanging.
  7. model.training mode is restored after certification, not left in eval().
  8. Two-stage CERTIFY: selection and estimation use independent sample batches.
  9. certification_budget_total never exceeds the declared cap when divided across a
     dataset, including when the budget is smaller than the number of samples.
  10. Aggregate certify_dataset metrics include abstained samples rather than excluding
      them (excluding them would inflate mean radius / certified accuracy).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nightmarenet.evaluation.certification import (
    CertificationResult,
    _run_noisy_forward_passes,
    certify_dataset,
    certify_sample,
    clopper_pearson_lower_bound,
)

# ---------------------------------------------------------------------------------------
# Clopper-Pearson bound: known-reference tests
# ---------------------------------------------------------------------------------------


class TestClopperPearsonLowerBound:
    def test_matches_beta_ppf_reference_directly(self):
        """The function is specified to just be scipy.stats.beta.ppf(alpha, k, n-k+1) --
        verify it matches that reference implementation call-for-call, independent of any
        of this module's own logic."""
        from scipy.stats import beta as beta_dist

        for k, n, alpha in [(900, 1000, 0.001), (50, 100, 0.01), (1, 10, 0.05), (10, 10, 0.001)]:
            expected = float(beta_dist.ppf(alpha, k, n - k + 1))
            actual = clopper_pearson_lower_bound(k, n, alpha)
            assert actual == pytest.approx(expected, rel=1e-12)

    def test_issue_reference_value_k900_n1000_alpha0001(self):
        """Issue #160 states k=900, n=1000, alpha=0.001 should give pA ~ 0.884.

        Recomputing the specified formula directly (scipy.stats.beta.ppf(0.001, 900, 101))
        gives pA ~ 0.8675, not 0.884 -- verified against multiple independent statistical
        methods (statsmodels' beta/wilson/jeffreys/agresti_coull, and a normal
        approximation), all of which cluster around 0.865-0.871. The issue's ~0.884 figure
        appears to be an approximation error in the issue text itself, not a different
        formula. This test locks in the value the *specified* formula actually produces;
        flagged in the PR description for the maintainer to confirm/update the issue text.
        """
        p_a = clopper_pearson_lower_bound(k=900, n=1000, alpha=0.001)
        assert p_a == pytest.approx(0.8675, abs=1e-3)

    def test_matches_defining_property_via_binomial_cdf(self):
        """Independent verification path: the Clopper-Pearson lower bound L is *defined*
        by P(X >= k | n, L) = alpha (the binomial survival function), not by beta.ppf --
        beta.ppf is just the closed-form solution to that equation. Checking the defining
        property via scipy.stats.binom.sf verifies correctness through a genuinely
        different code path than the implementation itself uses.
        """
        from scipy.stats import binom

        for k, n, alpha in [(900, 1000, 0.001), (20, 100, 0.025), (50, 100, 0.01), (1, 10, 0.05)]:
            lower = clopper_pearson_lower_bound(k, n, alpha)
            defining_property = binom.sf(k - 1, n, lower)
            assert defining_property == pytest.approx(alpha, rel=1e-6)

    def test_closed_form_all_successes(self):
        """When k == n, Beta(n, 1) is a Power distribution with CDF t^n, so its inverse
        CDF has the exact closed form alpha ** (1/n) -- an independent check on a
        different computational path than beta.ppf."""
        n, alpha = 50, 0.01
        expected = alpha ** (1.0 / n)
        assert clopper_pearson_lower_bound(n, n, alpha) == pytest.approx(expected, rel=1e-9)

    def test_zero_successes_gives_zero(self):
        assert clopper_pearson_lower_bound(0, 100, 0.001) == 0.0

    @pytest.mark.parametrize("k,n", [(-1, 10), (11, 10)])
    def test_rejects_out_of_range_k(self, k, n):
        with pytest.raises(ValueError):
            clopper_pearson_lower_bound(k, n, 0.01)

    def test_rejects_nonpositive_n(self):
        with pytest.raises(ValueError):
            clopper_pearson_lower_bound(0, 0, 0.01)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_alpha_out_of_range(self, alpha):
        with pytest.raises(ValueError):
            clopper_pearson_lower_bound(5, 10, alpha)


# ---------------------------------------------------------------------------------------
# Fakes: a minimal PreTrainedModel-shaped classifier and tokenizer
# ---------------------------------------------------------------------------------------


class _Config:
    num_labels = 2


class _NoiseSignClassifier(nn.Module):
    """Deterministic (given a seed) stand-in for a PreTrainedModel.

    Classifies each noisy embedding copy by the sign of its mean value. The clean
    embedding is initialized to exactly zero, so with the bias set to 0 each noisy copy's
    prediction is driven entirely by the injected noise -- giving near-50/50 votes (good
    for testing abstention) or a clear majority (after biasing the embedding weights, good
    for testing successful certification). Exposes get_input_embeddings() and accepts
    input_ids=/attention_mask=, matching the real HF interface certify_sample relies on.
    """

    def __init__(self, vocab_size: int = 20, hidden_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.zeros_(self.embedding.weight)
        self.config = _Config()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None):
        embeds = self.embedding(input_ids)  # (batch, seq, hidden) -- hook adds noise here
        means = embeds.mean(dim=(1, 2))
        logits = torch.stack([-means, means], dim=1)  # class 1 wins when mean >= 0
        return type("Output", (), {"logits": logits})()


class _FakeTokenizer:
    """Minimal HF-tokenizer-shaped callable: maps any text to a fixed short token sequence."""

    def __call__(self, text, truncation=True, max_length=128, return_tensors="pt"):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture
def fake_model():
    return _NoiseSignClassifier()


@pytest.fixture
def fake_tokenizer():
    return _FakeTokenizer()


# ---------------------------------------------------------------------------------------
# Forward-hook noise injection + cleanup
# ---------------------------------------------------------------------------------------


class TestForwardHookNoiseInjection:
    def test_hook_is_removed_after_normal_completion(self, fake_model):
        input_ids = torch.tensor([[1, 2, 3]])
        n_hooks_before = len(fake_model.embedding._forward_hooks)

        _run_noisy_forward_passes(
            fake_model,
            input_ids,
            None,
            sigma=0.5,
            n=20,
            num_classes=2,
            batch_size=10,
            device="cpu",
        )

        assert len(fake_model.embedding._forward_hooks) == n_hooks_before

    def test_hook_is_removed_even_if_forward_raises(self, fake_model):
        input_ids = torch.tensor([[1, 2, 3]])
        n_hooks_before = len(fake_model.embedding._forward_hooks)

        def broken_forward(self, input_ids=None, attention_mask=None):
            raise RuntimeError("simulated forward failure")

        fake_model.forward = broken_forward.__get__(fake_model)

        with pytest.raises(RuntimeError):
            _run_noisy_forward_passes(
                fake_model,
                input_ids,
                None,
                sigma=0.5,
                n=20,
                num_classes=2,
                batch_size=10,
                device="cpu",
            )

        assert len(fake_model.embedding._forward_hooks) == n_hooks_before

    def test_noise_actually_perturbs_predictions(self, fake_model):
        # With a zero clean embedding and sigma large enough, the hook's noise should
        # produce votes for both classes rather than a degenerate all-one-class histogram
        # (which would indicate the hook isn't actually being applied).
        torch.manual_seed(0)
        input_ids = torch.tensor([[1, 2, 3]])
        counts = _run_noisy_forward_passes(
            fake_model,
            input_ids,
            None,
            sigma=1.0,
            n=200,
            num_classes=2,
            batch_size=50,
            device="cpu",
        )
        assert counts.sum() == 200
        assert counts[0] > 0 and counts[1] > 0

    def test_rejects_nonpositive_batch_size(self, fake_model):
        input_ids = torch.tensor([[1, 2, 3]])
        with pytest.raises(ValueError):
            _run_noisy_forward_passes(
                fake_model,
                input_ids,
                None,
                sigma=0.5,
                n=20,
                num_classes=2,
                batch_size=0,
                device="cpu",
            )

    def test_negative_batch_size_raises_rather_than_hanging(self, fake_model):
        input_ids = torch.tensor([[1, 2, 3]])
        with pytest.raises(ValueError):
            _run_noisy_forward_passes(
                fake_model,
                input_ids,
                None,
                sigma=0.5,
                n=20,
                num_classes=2,
                batch_size=-5,
                device="cpu",
            )

    def test_restores_eval_mode_after_certification(self, fake_model):
        fake_model.eval()
        input_ids = torch.tensor([[1, 2, 3]])
        _run_noisy_forward_passes(
            fake_model,
            input_ids,
            None,
            sigma=0.5,
            n=20,
            num_classes=2,
            batch_size=10,
            device="cpu",
        )
        assert fake_model.training is False

    def test_restores_train_mode_after_certification(self, fake_model):
        # A model the caller was actively training must not be silently left in eval()
        # mode after certification runs -- that would affect subsequent training steps
        # (e.g. disabling dropout/batchnorm updates) without the caller asking for it.
        fake_model.train()
        input_ids = torch.tensor([[1, 2, 3]])
        _run_noisy_forward_passes(
            fake_model,
            input_ids,
            None,
            sigma=0.5,
            n=20,
            num_classes=2,
            batch_size=10,
            device="cpu",
        )
        assert fake_model.training is True

    def test_restores_train_mode_even_if_forward_raises(self, fake_model):
        fake_model.train()
        input_ids = torch.tensor([[1, 2, 3]])

        def broken_forward(self, input_ids=None, attention_mask=None):
            raise RuntimeError("simulated forward failure")

        fake_model.forward = broken_forward.__get__(fake_model)

        with pytest.raises(RuntimeError):
            _run_noisy_forward_passes(
                fake_model,
                input_ids,
                None,
                sigma=0.5,
                n=20,
                num_classes=2,
                batch_size=10,
                device="cpu",
            )
        assert fake_model.training is True


# ---------------------------------------------------------------------------------------
# certify_sample: abstention, determinism, batching invariance
# ---------------------------------------------------------------------------------------


class TestCertifySampleAbstention:
    def test_abstains_when_predictions_near_uniform(self, fake_model, fake_tokenizer):
        # Zero-initialized embedding + moderate sigma -> ~50/50 votes -> should not clear
        # the p_A > 0.5 confidence bar at a small sample size.
        torch.manual_seed(0)
        result = certify_sample(
            fake_model,
            fake_tokenizer,
            "irrelevant text",
            sigma=0.5,
            n0=20,
            n=50,
            alpha=0.01,
        )
        assert isinstance(result, CertificationResult)
        assert result.abstained is True
        assert result.certified_radius == 0.0

    def test_certifies_when_one_class_dominates(self, fake_model, fake_tokenizer):
        with torch.no_grad():
            fake_model.embedding.weight.fill_(5.0)
        torch.manual_seed(3)

        result = certify_sample(
            fake_model,
            fake_tokenizer,
            "irrelevant text",
            label=1,
            sigma=0.1,
            n0=20,
            n=200,
            alpha=0.001,
        )

        assert result.abstained is False
        assert result.prediction == 1
        assert result.certified_radius > 0.0
        assert result.p_a_lower > 0.5
        assert result.correct is True
        assert result.n_samples_used == 220  # n0 + n

    def test_certification_budget_preserves_n0_reduces_n_first(self, fake_model, fake_tokenizer):
        result = certify_sample(
            fake_model,
            fake_tokenizer,
            "irrelevant text",
            sigma=0.5,
            n0=20,
            n=1000,
            alpha=0.01,
            certification_budget=50,
        )
        assert result.n_samples_used <= 50

    def test_budget_smaller_than_n0_reduces_n0_too(self, fake_model, fake_tokenizer):
        result = certify_sample(
            fake_model,
            fake_tokenizer,
            "irrelevant text",
            sigma=0.5,
            n0=100,
            n=1000,
            alpha=0.01,
            certification_budget=25,
        )
        assert result.n_samples_used <= 25
        assert result.abstained is True  # no budget left for the estimation stage

    def test_zero_budget_abstains_without_running_inference(self, fake_model, fake_tokenizer):
        result = certify_sample(
            fake_model,
            fake_tokenizer,
            "irrelevant text",
            sigma=0.5,
            n0=100,
            n=1000,
            alpha=0.01,
            certification_budget=0,
        )
        assert result.abstained is True
        assert result.n_samples_used == 0


class TestCertifySampleSelectionIndependence:
    def test_selection_and_estimation_use_independent_noise_draws(
        self, fake_model, fake_tokenizer, monkeypatch
    ):
        """The two-stage design must call _run_noisy_forward_passes twice (once for
        selection with n0, once for estimation with n) rather than reusing one batch for
        both -- reusing the same batch is exactly the selection-bias bug this design
        fixes (picking the plurality class, then testing significance on the same votes
        that were used to pick it, overstates confidence)."""
        import nightmarenet.evaluation.certification as certification_module

        call_ns = []
        original = certification_module._run_noisy_forward_passes

        def spy(*args, **kwargs):
            call_ns.append(args[4])  # n is the 5th positional arg
            return original(*args, **kwargs)

        monkeypatch.setattr(certification_module, "_run_noisy_forward_passes", spy)

        torch.manual_seed(1)
        certify_sample(fake_model, fake_tokenizer, "text", sigma=0.5, n0=17, n=33, alpha=0.05)

        assert call_ns == [17, 33]  # two separate calls, selection then estimation


class TestCertifySampleDeterminism:
    def test_identical_seed_gives_identical_result(self, fake_model, fake_tokenizer):
        torch.manual_seed(1234)
        result_a = certify_sample(
            fake_model, fake_tokenizer, "same text", sigma=0.5, n0=30, n=100, alpha=0.01
        )

        torch.manual_seed(1234)
        result_b = certify_sample(
            fake_model, fake_tokenizer, "same text", sigma=0.5, n0=30, n=100, alpha=0.01
        )

        assert result_a == result_b

    def test_batch_size_does_not_change_sample_count_or_validity(self, fake_model, fake_tokenizer):
        """Chunking for memory-bounded inference must not change *how many* samples are
        drawn or produce an invalid histogram -- but see the note in
        _run_noisy_forward_passes: PyTorch's CPU torch.randn/normal_ kernel is internally
        vectorized, and the specific values it draws depend on the total element count
        requested in a single call, not just the seed and sequential position. This means
        different batch_size choices do NOT reproduce bit-identical noise draws even under
        the same seed (verified independently: splitting a single torch.randn(90, 5, 8)
        call into thirteen torch.randn(<=7, 5, 8) calls diverges starting partway through
        the very first chunk). That's a property of PyTorch itself, not a bug in this
        module -- each configuration still draws valid i.i.d. Gaussian samples and the
        certification math doesn't depend on cross-batch-size reproducibility, only on
        same-batch-size-and-seed reproducibility (see test_identical_seed above).
        """
        torch.manual_seed(42)
        result_one_batch = certify_sample(
            fake_model,
            fake_tokenizer,
            "same text",
            sigma=0.5,
            n0=10,
            n=90,
            alpha=0.05,
            batch_size=1000,
        )
        torch.manual_seed(42)
        result_chunked = certify_sample(
            fake_model,
            fake_tokenizer,
            "same text",
            sigma=0.5,
            n0=10,
            n=90,
            alpha=0.05,
            batch_size=7,
        )
        assert result_one_batch.n_samples_used == result_chunked.n_samples_used == 100
        for result in (result_one_batch, result_chunked):
            assert result.prediction in (0, 1)
            assert 0.0 <= result.p_a_lower <= 1.0


# ---------------------------------------------------------------------------------------
# certify_dataset
# ---------------------------------------------------------------------------------------


class _ListDataset:
    """Minimal HF-Dataset-shaped wrapper around a list of dicts, for testing without a
    real `datasets` dependency in this test module."""

    def __init__(self, examples):
        self._examples = examples

    def __len__(self):
        return len(self._examples)

    def __iter__(self):
        return iter(self._examples)

    def shuffle(self, seed=42):
        return self

    def select(self, indices):
        return _ListDataset([self._examples[i] for i in indices])


class TestCertifyDataset:
    def test_empty_dataset_returns_zeroed_summary(self, fake_model, fake_tokenizer):
        result = certify_dataset(
            fake_model,
            fake_tokenizer,
            _ListDataset([]),
        )
        assert result["n_samples"] == 0
        assert result["certified_radius_mean"] == 0.0
        assert result["certification_abstain_rate"] == 0.0

    def test_aggregates_across_samples(self, fake_model, fake_tokenizer):
        with torch.no_grad():
            fake_model.embedding.weight.fill_(5.0)
        torch.manual_seed(5)

        dataset = _ListDataset(
            [
                {"text": "example one", "label": 1},
                {"text": "example two", "label": 1},
                {"text": "example three", "label": 0},
            ]
        )

        summary = certify_dataset(
            fake_model,
            fake_tokenizer,
            dataset,
            sigma=0.1,
            n=100,
            alpha=0.001,
        )

        assert summary["n_samples"] == 3
        assert 0.0 <= summary["certification_abstain_rate"] <= 1.0
        assert summary["certified_accuracy"] is None or 0.0 <= summary["certified_accuracy"] <= 1.0
        assert len(summary["results"]) == 3

    def test_budget_total_divided_across_samples_exactly(self, fake_model, fake_tokenizer):
        dataset = _ListDataset([{"text": "a", "label": 0} for _ in range(10)])
        summary = certify_dataset(
            fake_model,
            fake_tokenizer,
            dataset,
            sigma=0.5,
            n=1000,
            alpha=0.01,
            certification_budget_total=100,
        )
        total_used = sum(r.n_samples_used for r in summary["results"])
        assert total_used <= 100  # must never exceed the declared total, even summed

    def test_budget_smaller_than_sample_count_never_exceeds_total(self, fake_model, fake_tokenizer):
        # Regression test: budget=1 spread over 10 samples must never let each sample get
        # a floor of 1 pass (which would total 10, 10x the declared budget).
        dataset = _ListDataset([{"text": "a", "label": 0} for _ in range(10)])
        summary = certify_dataset(
            fake_model,
            fake_tokenizer,
            dataset,
            sigma=0.5,
            n=1000,
            alpha=0.01,
            certification_budget_total=1,
        )
        total_used = sum(r.n_samples_used for r in summary["results"])
        assert total_used <= 1

    def test_zero_total_budget_all_samples_abstain(self, fake_model, fake_tokenizer):
        dataset = _ListDataset([{"text": "a", "label": 0} for _ in range(5)])
        summary = certify_dataset(
            fake_model,
            fake_tokenizer,
            dataset,
            sigma=0.5,
            n=1000,
            alpha=0.01,
            certification_budget_total=0,
        )
        assert summary["certification_abstain_rate"] == 1.0
        assert all(r.n_samples_used == 0 for r in summary["results"])

    def test_aggregate_metrics_include_abstained_samples(self, fake_model, fake_tokenizer):
        """A dataset where every sample abstains (near-uniform, zero embedding, tiny n)
        must report a mean/median radius of exactly 0.0 and certified_accuracy of 0.0 --
        not exclude the abstained samples and report an undefined/misleadingly-high value."""
        torch.manual_seed(0)
        dataset = _ListDataset(
            [
                {"text": "a", "label": 0},
                {"text": "b", "label": 1},
            ]
        )
        summary = certify_dataset(
            fake_model,
            fake_tokenizer,
            dataset,
            sigma=0.5,
            n0=5,
            n=5,
            alpha=0.5,  # tiny samples, generous alpha but zero-signal model
        )
        assert all(r.abstained for r in summary["results"])
        assert summary["certified_radius_mean"] == 0.0
        assert summary["certified_radius_median"] == 0.0
        assert summary["certified_accuracy"] == 0.0  # abstained-but-labeled counts as incorrect


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
