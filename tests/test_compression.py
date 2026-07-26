"""Unit tests for nightmarenet/compression/pruning.py

Covers MagnitudePruner, BottleneckWrapper, and apply_bottleneck_to_model.
Uses a tiny hand-built model fixture — no downloads, no GPU needed.
"""

import pytest
import torch
import torch.nn as nn

from nightmarenet.compression.pruning import (
    BottleneckWrapper,
    MagnitudePruner,
    apply_bottleneck_to_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """A minimal 2-layer MLP with a >=2D weight matrix, so MagnitudePruner
    actually has something to prune (it skips dim() < 2 params like biases).
    """

    def __init__(self, in_dim=16, hidden_dim=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TinyBlockWithMLP(nn.Module):
    """A minimal 'transformer-block'-shaped module whose child is literally
    named `mlp`, so apply_bottleneck_to_model's substring match ("mlp"/"attn"
    in child name) actually fires on it during the recursive walk.
    """

    def __init__(self, hidden_dim=16):
        super().__init__()
        # named child "mlp" -> apply_bottleneck_to_model will wrap this
        self.mlp = nn.Linear(hidden_dim, hidden_dim)
        # named child "other" -> should NOT be wrapped
        self.other = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.other(self.mlp(x))


@pytest.fixture
def tiny_mlp():
    torch.manual_seed(0)
    return TinyMLP()


@pytest.fixture
def tiny_block():
    torch.manual_seed(0)
    return TinyBlockWithMLP(hidden_dim=16)


# ---------------------------------------------------------------------------
# MagnitudePruner
# ---------------------------------------------------------------------------


class TestMagnitudePruner:
    def test_pruning_ratio_respected(self, tiny_mlp):
        pruner = MagnitudePruner(pruning_ratio=0.2)
        stats = pruner.apply(tiny_mlp)
        assert stats["total_params"] > 0
        # exact count can differ from k due to ties at the threshold,
        # so check approx, not exact
        assert abs(stats["sparsity"] - 0.2) < 0.05

    def test_pruning_ratio_zero_noop(self, tiny_mlp):
        before = {n: p.clone() for n, p in tiny_mlp.named_parameters()}
        pruner = MagnitudePruner(pruning_ratio=0.0)
        stats = pruner.apply(tiny_mlp)
        assert stats["pruned_params"] == 0
        for n, p in tiny_mlp.named_parameters():
            assert torch.equal(p, before[n])

    def test_pruning_ratio_high(self, tiny_mlp):
        pruner = MagnitudePruner(pruning_ratio=0.9)
        pruner.apply(tiny_mlp)
        x = torch.randn(4, 16)
        out = tiny_mlp(x)
        assert out.shape == (4, 8)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("bad_ratio", [-0.1, 1.0, 1.5])
    def test_pruning_invalid_ratio(self, bad_ratio):
        with pytest.raises(ValueError):
            MagnitudePruner(pruning_ratio=bad_ratio)

    def test_pruned_model_forward_pass(self, tiny_mlp):
        pruner = MagnitudePruner(pruning_ratio=0.5)
        pruner.apply(tiny_mlp)
        x = torch.randn(4, 16)
        out = tiny_mlp(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_pruning_determinism(self):
        torch.manual_seed(42)
        model_a = TinyMLP()
        torch.manual_seed(42)
        model_b = TinyMLP()

        MagnitudePruner(pruning_ratio=0.3).apply(model_a)
        MagnitudePruner(pruning_ratio=0.3).apply(model_b)

        for (na, pa), (_nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
            assert torch.equal(pa, pb), f"mismatch at {na}"


# ---------------------------------------------------------------------------
# BottleneckWrapper
# ---------------------------------------------------------------------------


class TestBottleneckWrapper:
    def test_bottleneck_reduces_parameters(self):
        layer = nn.Linear(16, 16)
        original_param_count = sum(p.numel() for p in layer.parameters())

        wrapped = BottleneckWrapper(layer, rank_ratio=0.25)
        bottleneck_param_count = (
            wrapped.down_project.weight.numel() + wrapped.up_project.weight.numel()
        )
        # the bottleneck path itself should be cheaper than a full hidden_dim^2 layer
        assert bottleneck_param_count < original_param_count

    def test_bottleneck_forward_shape(self):
        layer = nn.Linear(16, 16)
        wrapped = BottleneckWrapper(layer, rank_ratio=0.5)
        x = torch.randn(4, 16)
        out = wrapped(x)
        # forward feeds bottlenecked tensor into original_layer, so expected
        # shape is original_layer's own output shape, not just "== input"
        expected = layer(x)
        assert out.shape == expected.shape

    def test_bottleneck_rank_ratio(self):
        layer_a = nn.Linear(16, 16)
        layer_b = nn.Linear(16, 16)
        wrapped_low = BottleneckWrapper(layer_a, rank_ratio=0.1)
        wrapped_high = BottleneckWrapper(layer_b, rank_ratio=0.9)
        assert wrapped_low.bottleneck_dim < wrapped_high.bottleneck_dim

    @pytest.mark.parametrize("bad_ratio", [0.0, -0.1, 1.1])
    def test_bottleneck_invalid_rank_ratio(self, bad_ratio):
        with pytest.raises(ValueError):
            BottleneckWrapper(nn.Linear(16, 16), rank_ratio=bad_ratio)

    def test_apply_bottleneck_to_model(self, tiny_block):
        stats = apply_bottleneck_to_model(tiny_block, rank_ratio=0.5)
        # only the "mlp"-named child should be wrapped, not "other"
        assert stats["wrapped_count"] == 1
        assert isinstance(tiny_block.mlp, BottleneckWrapper)
        assert not isinstance(tiny_block.other, BottleneckWrapper)

        x = torch.randn(4, 16)
        out = tiny_block(x)
        assert out.shape == (4, 16)
        assert torch.isfinite(out).all()
