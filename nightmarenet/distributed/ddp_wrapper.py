"""Native PyTorch DDP wrappers."""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


class DDPWrapper:
    """Manages the distributed process group and model wrapping for native PyTorch DDP."""

    def __init__(self, backend: str = "nccl") -> None:
        self.backend = backend
        self.is_initialized = False

    def setup(self) -> None:
        """Initialize the distributed process group."""
        if not dist.is_available():
            logger.warning("torch.distributed is not available. Falling back.")
            return

        if not dist.is_initialized():
            # Check if launched via torchrun (environmental variables present)
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                dist.init_process_group(backend=self.backend)
                self.is_initialized = True
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                logger.info(
                    f"DDP initialized on rank {dist.get_rank()} with backend {self.backend}"
                )
            else:
                logger.warning("Not launched via torchrun (RANK not found in env). DDP disabled.")

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap the model in DistributedDataParallel if initialized."""
        if self.is_initialized and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            model = model.to(local_rank)
            return DistributedDataParallel(model, device_ids=[local_rank])
        return model

    def teardown(self) -> None:
        """Destroy the process group."""
        if self.is_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("DDP process group destroyed.")
