"""Tests for distributed training utilities."""

import torch

from nightmarenet.training.distributed import DistributedContext, is_accelerate_available


class TestDistributedContext:
    """Test DistributedContext wrapper."""

    def test_disabled_by_default(self):
        ctx = DistributedContext(enabled=False)
        assert ctx.enabled is False
        assert ctx.is_main_process is True
        assert ctx.num_processes == 1

    def test_device_fallback(self):
        ctx = DistributedContext(enabled=False)
        device = ctx.device
        assert isinstance(device, torch.device)

    def test_prepare_passthrough_when_disabled(self):
        ctx = DistributedContext(enabled=False)
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dummy_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(8, 10)),
            batch_size=4,
        )
        prepared = ctx.prepare(model, optimizer, dummy_loader)
        assert prepared[0] is model
        assert prepared[1] is optimizer
        assert prepared[2] is dummy_loader

    def test_backward_passthrough(self):
        ctx = DistributedContext(enabled=False)
        x = torch.tensor([2.0], requires_grad=True)
        loss = x * 3
        ctx.backward(loss)
        assert x.grad is not None

    def test_clip_grad_norm_passthrough(self):
        ctx = DistributedContext(enabled=False)
        model = torch.nn.Linear(4, 2)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        ctx.clip_grad_norm(model.parameters(), max_norm=1.0)
        # Should not raise

    def test_unwrap_model_passthrough(self):
        ctx = DistributedContext(enabled=False)
        model = torch.nn.Linear(4, 2)
        assert ctx.unwrap_model(model) is model

    def test_wait_for_everyone_noop(self):
        ctx = DistributedContext(enabled=False)
        ctx.wait_for_everyone()  # Should not raise

    def test_is_accelerate_available_returns_bool(self):
        result = is_accelerate_available()
        assert isinstance(result, bool)

    def test_enabled_without_accelerate_fallback(self):
        """If accelerate isn't installed, enabled=True should fall back gracefully."""
        ctx = DistributedContext(enabled=True)
        if not is_accelerate_available():
            assert ctx.enabled is False
        # Either way: should work
        assert ctx.is_main_process is True
