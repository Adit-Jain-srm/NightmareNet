"""Tests for evaluation metrics."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from nightmarenet.compression.pruning import BottleneckWrapper, MagnitudePruner


class SimpleModel(nn.Module):
    """A tiny model for testing metrics."""

    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, labels=None, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.linear(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
        return type("Output", (), {"loss": loss, "logits": logits})()


class TestMagnitudePruner:
    """Test the MagnitudePruner."""

    def test_pruning_reduces_nonzero_params(self):
        model = SimpleModel()
        pruner = MagnitudePruner(pruning_ratio=0.5)

        # Count non-zero params before
        before_nonzero = sum((p != 0).sum().item() for p in model.parameters() if p.dim() >= 2)

        stats = pruner.apply(model)

        # Count non-zero params after
        after_nonzero = sum((p != 0).sum().item() for p in model.parameters() if p.dim() >= 2)

        assert after_nonzero < before_nonzero
        assert stats["pruned_params"] > 0
        assert stats["total_params"] > 0
        assert 0 < stats["sparsity"] < 1

    def test_zero_pruning_preserves_all(self):
        model = SimpleModel()
        pruner = MagnitudePruner(pruning_ratio=0.0)
        stats = pruner.apply(model)
        assert stats["pruned_params"] == 0

    def test_invalid_pruning_ratio(self):
        with pytest.raises(ValueError):
            MagnitudePruner(pruning_ratio=1.0)
        with pytest.raises(ValueError):
            MagnitudePruner(pruning_ratio=-0.1)

    def test_pruning_stats_format(self):
        model = SimpleModel()
        pruner = MagnitudePruner(pruning_ratio=0.3)
        stats = pruner.apply(model)
        assert "pruned_params" in stats
        assert "total_params" in stats
        assert "sparsity" in stats


class TestBottleneckWrapper:
    """Test the BottleneckWrapper."""

    def test_wraps_linear_layer(self):
        original = nn.Linear(32, 64)
        wrapper = BottleneckWrapper(original, rank_ratio=0.5)
        # _infer_hidden_dim gets the last dim of the weight matrix [64, 32] → 32
        assert wrapper.hidden_dim == 32
        assert wrapper.bottleneck_dim == 16

    def test_forward_pass(self):
        original = nn.Linear(32, 64)
        wrapper = BottleneckWrapper(original, rank_ratio=0.5)
        # Input dim matches hidden_dim (32), output goes through original (32→64)
        x = torch.randn(2, 10, 32)
        output = wrapper(x)
        assert output.shape == (2, 10, 64)

    def test_custom_bottleneck_dim(self):
        original = nn.Linear(32, 64)
        wrapper = BottleneckWrapper(original, bottleneck_dim=8)
        assert wrapper.bottleneck_dim == 8


class TestMetricsComputation:
    """Test metric helper computations."""

    def test_perplexity_is_finite(self):
        from nightmarenet.evaluation.metrics import compute_perplexity

        model = SimpleModel()
        # Create a simple dataloader
        input_ids = torch.randint(0, 100, (4, 16))

        class SimpleDataLoader:
            def __iter__(self):
                for i in range(0, len(input_ids), 2):
                    yield {"input_ids": input_ids[i : i + 2]}

        dataloader = SimpleDataLoader()
        ppl = compute_perplexity(model, dataloader, device="cpu")
        assert np.isfinite(ppl)
        assert ppl > 0

    def test_recall_score_returns_dict(self):
        from nightmarenet.evaluation.metrics import recall_score

        model = SimpleModel()
        input_ids = torch.randint(0, 100, (4, 16))

        class MockTokenizer:
            pad_token_id = 0

        class SimpleDataLoader:
            def __iter__(self):
                yield {"input_ids": input_ids}

        result = recall_score(model, SimpleDataLoader(), MockTokenizer(), device="cpu")
        assert "metric" in result
        assert result["metric"] == "recall"
        assert "token_accuracy" in result
        assert "perplexity" in result
        assert 0 <= result["token_accuracy"] <= 1

    def test_hallucination_rate_returns_dict(self):
        from nightmarenet.evaluation.metrics import hallucination_rate

        model = SimpleModel()
        input_ids = torch.randint(0, 100, (4, 16))

        class MockTokenizer:
            pad_token_id = 0

        class SimpleDataLoader:
            def __iter__(self):
                yield {"input_ids": input_ids}

        result = hallucination_rate(model, SimpleDataLoader(), MockTokenizer(), device="cpu")
        assert "metric" in result
        assert result["metric"] == "hallucination"
        assert "hallucination_rate" in result
        assert 0 <= result["hallucination_rate"] <= 1
