"""Crash recovery and checkpoint resume logic."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


class ResumeManager:
    """Manages restoring state from the latest complete checkpoint."""

    def __init__(self, resume_dir: str) -> None:
        self.resume_dir = resume_dir

    def verify_and_load(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, current_config: dict
    ) -> dict:
        """Loads state into model and optimizer and returns metadata."""
        from nightmarenet.distributed.checkpoint import (
            load_model_weights,
            validate_checkpoint_integrity,
        )

        # 1. Run the dedicated validation check (structural + checksum + version checking)
        metadata = validate_checkpoint_integrity(self.resume_dir, current_config)

        # 2. Load Model
        has_params = list(model.parameters())
        device = next(model.parameters()).device if has_params else torch.device("cpu")
        load_model_weights(model, self.resume_dir, device)

        # 3. Load Optimizer
        opt_file = os.path.join(self.resume_dir, "optimizer.pt")
        if os.path.exists(opt_file):
            optimizer.load_state_dict(torch.load(opt_file, map_location="cpu"))
            logger.info("Loaded optimizer state from checkpoint.")

        # 4. Load RNG
        rng_file = os.path.join(self.resume_dir, "rng_state.pt")
        if os.path.exists(rng_file):
            rng_states = torch.load(rng_file)
            torch.set_rng_state(rng_states["cpu"])
            if torch.cuda.is_available() and "cuda" in rng_states:
                torch.cuda.set_rng_state_all(rng_states["cuda"])
            logger.info("Loaded RNG states from checkpoint.")

        return metadata
