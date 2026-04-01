"""Training phases: Wake, Dream, Nightmare, and Compression.

Each phase encapsulates a distinct training step in the sleep-inspired
training paradigm. Phases are called by the Trainer orchestrator.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nightmarenet.utils.validation import validate_positive_int

logger = logging.getLogger(__name__)


class WakePhase:
    """Standard supervised fine-tuning on real-world data.

    Args:
        model: The language model to train.
        optimizer: Optimizer instance.
        config: Training configuration dictionary.
        device: Device to train on.
    """

    def __init__(self, model, optimizer, config, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    def run(self, dataloader: DataLoader, num_epochs: int = 1) -> dict:
        """Run the wake phase (standard training).

        Args:
            dataloader: DataLoader providing tokenized training batches.
            num_epochs: Number of epochs to train.

        Returns:
            Dict with training metrics (avg_loss, total_steps).
        """
        validate_positive_int(num_epochs, "num_epochs", allow_zero=True)
        if num_epochs == 0:
            logger.info("Wake Phase - Skipping training because num_epochs=0.")
            return {
                "phase": "wake",
                "avg_loss": 0.0,
                "total_steps": 0,
            }
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            step_count = 0

            progress = tqdm(dataloader, desc=f"Wake Phase - Epoch {epoch + 1}/{num_epochs}")
            for step, batch in enumerate(progress):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch, labels=batch.get("input_ids"))
                loss = outputs.loss / self.gradient_accumulation_steps

                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    logger.warning("Wake Phase - NaN/Inf loss at step %d, skipping.", step)
                    self.optimizer.zero_grad()
                    continue

                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                step_count += 1
                total_steps += 1

                progress.set_postfix(loss=epoch_loss / step_count)

            avg_epoch_loss = epoch_loss / max(step_count, 1)
            logger.info(
                "Wake Phase - Epoch %d/%d - Avg Loss: %.4f",
                epoch + 1,
                num_epochs,
                avg_epoch_loss,
            )
            total_loss += avg_epoch_loss

        return {
            "phase": "wake",
            "avg_loss": total_loss / max(num_epochs, 1),
            "total_steps": total_steps,
        }


class DreamPhase:
    """Training on mildly distorted data with optional KL regularization.

    Trains the model on dream data while optionally using KL divergence
    against wake-phase outputs to prevent catastrophic forgetting.

    Args:
        model: The language model to train.
        optimizer: Optimizer instance.
        config: Training configuration dictionary.
        device: Device to train on.
        reference_model: Optional frozen copy of the model from wake phase for KL regularization.
        kl_weight: Weight for the KL divergence regularization term.
    """

    def __init__(
        self,
        model,
        optimizer,
        config,
        device="cpu",
        reference_model=None,
        kl_weight: float = 0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.reference_model = reference_model
        self.kl_weight = kl_weight
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    def _compute_kl_loss(self, logits, batch):
        """Compute KL divergence between current and reference model outputs."""
        if self.reference_model is None:
            return 0.0

        with torch.no_grad():
            ref_outputs = self.reference_model(**batch, labels=batch.get("input_ids"))
            ref_logits = ref_outputs.logits

        # KL(P_ref || P_current) to keep current close to reference
        log_probs = F.log_softmax(logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        kl_loss = F.kl_div(log_probs, ref_probs, reduction="batchmean")

        return kl_loss * self.kl_weight

    def run(self, dataloader: DataLoader, num_epochs: int = 1) -> dict:
        """Run the dream phase (distorted data training with KL regularization).

        Args:
            dataloader: DataLoader providing tokenized dream data batches.
            num_epochs: Number of epochs to train.

        Returns:
            Dict with training metrics.
        """
        validate_positive_int(num_epochs, "num_epochs", allow_zero=True)
        if num_epochs == 0:
            logger.info("Dream Phase - Skipping training because num_epochs=0.")
            return {
                "phase": "dream",
                "avg_loss": 0.0,
                "avg_kl_loss": 0.0,
                "total_steps": 0,
            }
        self.model.train()
        if self.reference_model is not None:
            self.reference_model.eval()

        total_loss = 0.0
        total_kl = 0.0
        total_steps = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_kl = 0.0
            step_count = 0

            progress = tqdm(dataloader, desc=f"Dream Phase - Epoch {epoch + 1}/{num_epochs}")
            for step, batch in enumerate(progress):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch, labels=batch.get("input_ids"))
                loss = outputs.loss / self.gradient_accumulation_steps

                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    logger.warning("Dream Phase - NaN/Inf loss at step %d, skipping.", step)
                    self.optimizer.zero_grad()
                    continue

                # Add KL regularization
                kl_loss = self._compute_kl_loss(outputs.logits, batch)
                total_phase_loss = loss + kl_loss / self.gradient_accumulation_steps
                total_phase_loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                if isinstance(kl_loss, torch.Tensor):
                    epoch_kl += kl_loss.item()
                step_count += 1
                total_steps += 1

                progress.set_postfix(loss=epoch_loss / step_count)

            avg_epoch_loss = epoch_loss / max(step_count, 1)
            avg_epoch_kl = epoch_kl / max(step_count, 1)
            logger.info(
                "Dream Phase - Epoch %d/%d - Avg Loss: %.4f - KL: %.4f",
                epoch + 1,
                num_epochs,
                avg_epoch_loss,
                avg_epoch_kl,
            )
            total_loss += avg_epoch_loss
            total_kl += avg_epoch_kl

        return {
            "phase": "dream",
            "avg_loss": total_loss / max(num_epochs, 1),
            "avg_kl_loss": total_kl / max(num_epochs, 1),
            "total_steps": total_steps,
        }


class NightmarePhase:
    """Training on extreme perturbations with higher learning rate.

    Stress-tests learned representations by training on aggressively
    corrupted, contradictory, and adversarial data.

    Args:
        model: The language model to train.
        optimizer: Optimizer instance (may use higher LR).
        config: Training configuration dictionary.
        device: Device to train on.
        lr_multiplier: Factor to multiply the learning rate by during this phase.
    """

    def __init__(self, model, optimizer, config, device="cpu", lr_multiplier=2.0):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.lr_multiplier = lr_multiplier
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    def _adjust_lr(self, multiplier):
        """Temporarily adjust learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= multiplier

    def run(self, dataloader: DataLoader, num_epochs: int = 1) -> dict:
        """Run the nightmare phase (adversarial training).

        Args:
            dataloader: DataLoader providing tokenized nightmare data batches.
            num_epochs: Number of epochs to train.

        Returns:
            Dict with training metrics.
        """
        validate_positive_int(num_epochs, "num_epochs", allow_zero=True)
        if num_epochs == 0:
            logger.info("Nightmare Phase - Skipping training because num_epochs=0.")
            return {
                "phase": "nightmare",
                "avg_loss": 0.0,
                "total_steps": 0,
            }
        self.model.train()

        # Increase learning rate for nightmare phase
        self._adjust_lr(self.lr_multiplier)

        total_loss = 0.0
        total_steps = 0

        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                step_count = 0

                progress = tqdm(
                    dataloader,
                    desc=f"Nightmare Phase - Epoch {epoch + 1}/{num_epochs}",
                )
                for step, batch in enumerate(progress):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(**batch, labels=batch.get("input_ids"))
                    loss = outputs.loss / self.gradient_accumulation_steps

                    if math.isnan(loss.item()) or math.isinf(loss.item()):
                        logger.warning("Nightmare Phase - NaN/Inf loss at step %d, skipping.", step)
                        self.optimizer.zero_grad()
                        continue

                    loss.backward()

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                    step_count += 1
                    total_steps += 1

                    progress.set_postfix(loss=epoch_loss / step_count)

                avg_epoch_loss = epoch_loss / max(step_count, 1)
                logger.info(
                    "Nightmare Phase - Epoch %d/%d - Avg Loss: %.4f",
                    epoch + 1,
                    num_epochs,
                    avg_epoch_loss,
                )
                total_loss += avg_epoch_loss
        finally:
            # Restore original learning rate
            self._adjust_lr(1.0 / self.lr_multiplier)

        return {
            "phase": "nightmare",
            "avg_loss": total_loss / max(num_epochs, 1),
            "total_steps": total_steps,
        }


class CompressionPhase:
    """Apply model compression (pruning / bottleneck).

    Forces information retention efficiency by pruning weights or applying
    bottleneck layers, then optionally fine-tuning the remaining weights.

    Args:
        model: The language model to compress.
        config: Compression configuration dictionary.
        device: Device to use.
    """

    def __init__(self, model, config, device="cpu"):
        self.model = model
        self.config = config
        self.device = device

    def run(
        self,
        dataloader: Optional[DataLoader] = None,
        optimizer=None,
    ) -> dict:
        """Run the compression phase.

        Args:
            dataloader: Optional DataLoader for fine-tuning after pruning.
            optimizer: Optional optimizer for fine-tuning.

        Returns:
            Dict with compression metrics.
        """
        pruning_ratio = self.config.get("pruning_ratio", 0.2)
        method = self.config.get("pruning_method", "magnitude")

        logger.info(
            "Compression Phase - Method: %s, Ratio: %.2f", method, pruning_ratio
        )

        if method == "magnitude":
            try:
                from nightmarenet.compression.pruning import MagnitudePruner

                pruner = MagnitudePruner(pruning_ratio=pruning_ratio)
                stats = pruner.apply(self.model)
            except Exception as exc:
                logger.error("Compression Phase - Failed to apply pruning: %s", exc)
                stats = {"pruned_params": 0, "total_params": 0}
        else:
            logger.warning("Unknown pruning method '%s'; skipping.", method)
            stats = {"pruned_params": 0, "total_params": 0}

        # Optional fine-tuning after pruning
        if (
            self.config.get("finetune_after_prune", True)
            and dataloader is not None
            and optimizer is not None
        ):
            finetune_epochs = self.config.get("finetune_epochs", 1)
            logger.info(
                "Fine-tuning after compression for %d epoch(s)...", finetune_epochs
            )
            self.model.train()
            for epoch in range(finetune_epochs):
                for batch in tqdm(
                    dataloader, desc=f"Post-compression fine-tune - Epoch {epoch + 1}"
                ):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch, labels=batch.get("input_ids"))
                    loss = outputs.loss

                    if math.isnan(loss.item()) or math.isinf(loss.item()):
                        logger.warning("Compression fine-tune - NaN/Inf loss, skipping.")
                        optimizer.zero_grad()
                        continue

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        return {
            "phase": "compression",
            "method": method,
            "pruning_ratio": pruning_ratio,
            **stats,
        }
