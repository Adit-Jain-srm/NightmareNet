"""Unit tests for nightmarenet.distributed.strategies."""

from __future__ import annotations

import unittest
from unittest import mock

import torch.nn as nn

from nightmarenet.distributed.ddp_wrapper import DDPWrapper
from nightmarenet.distributed.device_pool import DevicePool
from nightmarenet.distributed.strategies import apply_phase_strategy, unwrap_model


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)


class MockWrapperModule(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.module = inner


class TestDistributedStrategiesUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleModel()
        self.device_pool_multi = DevicePool(override_devices=[0, 1])
        self.device_pool_single = DevicePool(override_devices=[0])
        self.device_pool_empty = DevicePool(override_devices=[])

    def test_unwrap_model_unwrapped(self) -> None:
        """Test unwrap_model returns the module directly if not wrapped."""
        unwrapped = unwrap_model(self.model)
        self.assertIs(unwrapped, self.model)

    def test_unwrap_model_single_wrapper(self) -> None:
        """Test unwrap_model strips a single module wrapper layer."""
        wrapped = MockWrapperModule(self.model)
        unwrapped = unwrap_model(wrapped)
        self.assertIs(unwrapped, self.model)

    def test_unwrap_model_nested_wrapper(self) -> None:
        """Test unwrap_model recursively unwraps multiple nested wrapper layers."""
        wrapped = MockWrapperModule(MockWrapperModule(MockWrapperModule(self.model)))
        unwrapped = unwrap_model(wrapped)
        self.assertIs(unwrapped, self.model)

    def test_apply_phase_strategy_distributed_disabled(self) -> None:
        """Test apply_phase_strategy returns unwrapped model when distributed is disabled."""
        mock_ddp = mock.Mock(spec=DDPWrapper)
        result = apply_phase_strategy(
            phase="wake",
            model=self.model,
            device_pool=self.device_pool_multi,
            ddp_wrapper=mock_ddp,
            distributed_enabled=False,
        )
        self.assertIs(result, self.model)
        mock_ddp.wrap_model.assert_not_called()

    def test_apply_phase_strategy_single_device_fallback(self) -> None:
        """Test apply_phase_strategy falls back to unwrapped model with <= 1 device."""
        mock_ddp = mock.Mock(spec=DDPWrapper)
        result = apply_phase_strategy(
            phase="wake",
            model=self.model,
            device_pool=self.device_pool_single,
            ddp_wrapper=mock_ddp,
            distributed_enabled=True,
        )
        self.assertIs(result, self.model)

        result_empty = apply_phase_strategy(
            phase="nightmare",
            model=self.model,
            device_pool=self.device_pool_empty,
            ddp_wrapper=mock_ddp,
            distributed_enabled=True,
        )
        self.assertIs(result_empty, self.model)

    def test_apply_phase_strategy_wake_and_nightmare_ddp_initialized(self) -> None:
        """Test apply_phase_strategy applies DDP wrapping for wake and nightmare phases."""
        mock_ddp = mock.Mock(spec=DDPWrapper)
        mock_ddp.is_initialized = True
        wrapped_target = SimpleModel()
        mock_ddp.wrap_model.return_value = wrapped_target

        result_wake = apply_phase_strategy(
            phase="wake",
            model=self.model,
            device_pool=self.device_pool_multi,
            ddp_wrapper=mock_ddp,
            distributed_enabled=True,
        )
        self.assertIs(result_wake, wrapped_target)
        mock_ddp.wrap_model.assert_called_with(self.model)

        mock_ddp.wrap_model.reset_mock()
        result_nightmare = apply_phase_strategy(
            phase="nightmare",
            model=self.model,
            device_pool=self.device_pool_multi,
            ddp_wrapper=mock_ddp,
            distributed_enabled=True,
        )
        self.assertIs(result_nightmare, wrapped_target)
        mock_ddp.wrap_model.assert_called_with(self.model)

    def test_apply_phase_strategy_ddp_uninitialized_fallback(self) -> None:
        """Test fallback when DDP is requested in wake phase but DDPWrapper is uninitialized."""
        mock_ddp = mock.Mock(spec=DDPWrapper)
        mock_ddp.is_initialized = False

        result = apply_phase_strategy(
            phase="wake",
            model=self.model,
            device_pool=self.device_pool_multi,
            ddp_wrapper=mock_ddp,
            distributed_enabled=True,
        )
        self.assertIs(result, self.model)
        mock_ddp.wrap_model.assert_not_called()

    @mock.patch("torch.nn.DataParallel")
    def test_apply_phase_strategy_dream_dataparallel(self, mock_dp_cls: mock.MagicMock) -> None:
        """Test apply_phase_strategy wraps model with DataParallel for dream phase."""
        mock_dp_instance = mock.Mock()
        mock_dp_cls.return_value = mock_dp_instance
        mock_ddp = mock.Mock(spec=DDPWrapper)

        result = apply_phase_strategy(
            phase="dream",
            model=self.model,
            device_pool=self.device_pool_multi,
            ddp_wrapper=mock_ddp,
            distributed_enabled=True,
        )
        self.assertIs(result, mock_dp_instance)
        mock_dp_cls.assert_called_once_with(self.model, device_ids=[0, 1])

    def test_apply_phase_strategy_compress_phase_single_device(self) -> None:
        """Test compress phase always runs unwrapped on a single device."""
        mock_ddp = mock.Mock(spec=DDPWrapper)
        result = apply_phase_strategy(
            phase="compress",
            model=self.model,
            device_pool=self.device_pool_multi,
            ddp_wrapper=mock_ddp,
            distributed_enabled=True,
        )
        self.assertIs(result, self.model)

    def test_apply_phase_strategy_unknown_phase(self) -> None:
        """Test unknown phase returns the unwrapped model safely."""
        mock_ddp = mock.Mock(spec=DDPWrapper)
        result = apply_phase_strategy(
            phase="unrecognized_phase",
            model=self.model,
            device_pool=self.device_pool_multi,
            ddp_wrapper=mock_ddp,
            distributed_enabled=True,
        )
        self.assertIs(result, self.model)


if __name__ == "__main__":
    unittest.main()
