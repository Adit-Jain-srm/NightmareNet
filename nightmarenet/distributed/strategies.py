"""Strategy selection for distributed execution."""

from __future__ import annotations

import logging

import torch.nn as nn

from nightmarenet.distributed.ddp_wrapper import DDPWrapper
from nightmarenet.distributed.device_pool import DevicePool

logger = logging.getLogger(__name__)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Recursively unwrap a model from DDP or DataParallel."""
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


def apply_phase_strategy(
    phase: str, model: nn.Module, device_pool: DevicePool, ddp_wrapper: DDPWrapper
) -> nn.Module:
    """Applies the correct distributed strategy based on the phase.

    Wake/Nightmare: DDP (if available)
    Dream: DataParallel (embarrassingly parallel inference)
    Compress: Single GPU
    """
    model = unwrap_model(model)
    num_devices = device_pool.get_num_devices()

    if num_devices <= 1:
        return model

    if phase in ("wake", "nightmare"):
        if ddp_wrapper.is_initialized:
            logger.info(f"Applying DDP strategy for phase: {phase}")
            return ddp_wrapper.wrap_model(model)
        else:
            logger.warning(
                f"DDP requested for phase {phase} but not initialized. "
                "Falling back to single device."
            )
            return model

    elif phase == "dream":
        logger.info(f"Applying DataParallel strategy for phase: {phase}")
        return nn.DataParallel(model, device_ids=device_pool.available_devices)

    elif phase == "compress":
        logger.info("Compress phase must run on single device. Skipping distributed wrapper.")
        return model

    return model
