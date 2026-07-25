"""Training phases: Wake, Dream, Nightmare, and Compression.

Each phase encapsulates a distinct training step in the sleep-inspired
training paradigm. Phases are called by the Trainer orchestrator.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from tqdm import tqdm

from nightmarenet.training.callbacks import (
    CallbackManager,
    EventType,
    TrainingEvent,
)
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

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: Union[str, torch.device] = "cpu",
        scaler: Optional[torch.amp.GradScaler] = None,
        model_type: str = "causal_lm",
        callback_manager: Optional[CallbackManager] = None,
        lr_scheduler=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.scaler = scaler
        self.model_type = model_type
        self.callback_manager = callback_manager
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm: float = config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps: int = config.get("gradient_accumulation_steps", 1)

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
        use_amp = self.scaler is not None

        for epoch in range(num_epochs):
            if self.callback_manager is not None:
                self.callback_manager.emit(
                    TrainingEvent(
                        event_type=EventType.EPOCH_START,
                        phase="wake",
                        epoch=epoch + 1,
                    )
                )
            epoch_loss = 0.0
            step_count = 0

            progress = tqdm(dataloader, desc=f"Wake Phase - Epoch {epoch + 1}/{num_epochs}")
            for step, batch in enumerate(progress):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Only causal LM needs labels=input_ids.
                # Sequence classification and image classification already have batch["labels"].
                if self.model_type not in ("seq_classification", "image_classification"):
                    batch["labels"] = batch["input_ids"]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = self.model(
                        **batch,
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps

                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    logger.warning("Wake Phase - NaN/Inf loss at step %d, skipping.", step)
                    self.optimizer.zero_grad()
                    continue

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
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

            if self.callback_manager is not None:
                self.callback_manager.emit(
                    TrainingEvent(
                        event_type=EventType.EPOCH_END,
                        phase="wake",
                        epoch=epoch + 1,
                        metrics={"avg_loss": avg_epoch_loss},
                    )
                )

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
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: Union[str, torch.device] = "cpu",
        reference_model: Optional[torch.nn.Module] = None,
        kl_weight: float = 0.1,
        scaler: Optional[torch.amp.GradScaler] = None,
        model_type: str = "causal_lm",
        callback_manager: Optional[CallbackManager] = None,
        lr_scheduler=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.reference_model = reference_model
        self.kl_weight = kl_weight
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.model_type = model_type
        self.callback_manager = callback_manager

    def _compute_kl_loss(
        self,
        logits: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> Union[torch.Tensor, float]:
        """Compute KL divergence between current and reference model outputs."""
        if self.reference_model is None:
            return 0.0

        with torch.no_grad():
            # Remove labels to avoid unnecessary loss computation in the reference model
            ref_batch = {k: v for k, v in batch.items() if k != "labels"}
            ref_outputs = self.reference_model(**ref_batch)
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
        use_amp = self.scaler is not None

        for epoch in range(num_epochs):
            if self.callback_manager is not None:
                self.callback_manager.emit(
                    TrainingEvent(
                        event_type=EventType.EPOCH_START,
                        phase="dream",
                        epoch=epoch + 1,
                    )
                )

            epoch_loss = 0.0
            epoch_kl = 0.0
            step_count = 0

            progress = tqdm(dataloader, desc=f"Dream Phase - Epoch {epoch + 1}/{num_epochs}")
            for step, batch in enumerate(progress):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.model_type not in ("seq_classification", "image_classification"):
                    batch["labels"] = batch["input_ids"]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.gradient_accumulation_steps

                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    logger.warning("Dream Phase - NaN/Inf loss at step %d, skipping.", step)
                    self.optimizer.zero_grad()
                    continue

                # Add KL regularization
                with torch.amp.autocast("cuda", enabled=use_amp):
                    kl_loss = self._compute_kl_loss(outputs.logits, batch)
                    total_phase_loss = loss + kl_loss / self.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(total_phase_loss).backward()
                else:
                    total_phase_loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
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

            if self.callback_manager is not None:
                self.callback_manager.emit(
                    TrainingEvent(
                        event_type=EventType.EPOCH_END,
                        phase="dream",
                        epoch=epoch + 1,
                        metrics={
                            "avg_loss": avg_epoch_loss,
                            "avg_kl_loss": avg_epoch_kl,
                        },
                    )
                )

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

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: Union[str, torch.device] = "cpu",
        lr_multiplier: float = 2.0,
        scaler: Optional[torch.amp.GradScaler] = None,
        model_type: str = "causal_lm",
        lr_scheduler=None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.callback_manager = callback_manager
        self.lr_scheduler = lr_scheduler
        if lr_multiplier <= 0:
            raise ValueError(f"lr_multiplier must be > 0, got {lr_multiplier}")
        self.lr_multiplier = lr_multiplier
        self.scaler = scaler
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.model_type = model_type

    def _save_lr(self) -> dict:
        """Save current learning rates and scheduler base_lrs."""
        state = {"optimizer_lrs": [pg["lr"] for pg in self.optimizer.param_groups]}
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "base_lrs"):
            state["scheduler_base_lrs"] = list(self.lr_scheduler.base_lrs)
        return state

    def _restore_lr(self, saved_state: dict) -> None:
        """Restore learning rates and scheduler base_lrs from saved state."""
        opt_lrs = saved_state["optimizer_lrs"]
        if len(opt_lrs) != len(self.optimizer.param_groups):
            logger.warning(
                "LR restore mismatch: %d saved vs %d param groups",
                len(opt_lrs),
                len(self.optimizer.param_groups),
            )
        for pg, lr in zip(self.optimizer.param_groups, opt_lrs):
            pg["lr"] = lr

        if self.lr_scheduler is not None and "scheduler_base_lrs" in saved_state:
            self.lr_scheduler.base_lrs = list(saved_state["scheduler_base_lrs"])

    def _adjust_lr(self, multiplier: float) -> None:
        """Temporarily adjust learning rate (composes with scheduler)."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= multiplier
        # Also scale scheduler base_lrs so step() doesn't override
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "base_lrs"):
            self.lr_scheduler.base_lrs = [lr * multiplier for lr in self.lr_scheduler.base_lrs]

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

        # Save and increase learning rate for nightmare phase
        saved_lr_state = self._save_lr()
        self._adjust_lr(self.lr_multiplier)

        total_loss = 0.0
        total_steps = 0
        use_amp = self.scaler is not None

        try:
            for epoch in range(num_epochs):
                if self.callback_manager is not None:
                    self.callback_manager.emit(
                        TrainingEvent(
                            event_type=EventType.EPOCH_START,
                            phase="nightmare",
                            epoch=epoch + 1,
                        )
                    )
                epoch_loss = 0.0
                step_count = 0

                progress = tqdm(
                    dataloader,
                    desc=f"Nightmare Phase - Epoch {epoch + 1}/{num_epochs}",
                )
                for step, batch in enumerate(progress):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    if self.model_type not in ("seq_classification", "image_classification"):
                        batch["labels"] = batch["input_ids"]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = self.model(**batch)
                        loss = outputs.loss / self.gradient_accumulation_steps

                    if math.isnan(loss.item()) or math.isinf(loss.item()):
                        logger.warning("Nightmare Phase - NaN/Inf loss at step %d, skipping.", step)
                        self.optimizer.zero_grad()
                        continue

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
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
                if self.callback_manager is not None:
                    self.callback_manager.emit(
                        TrainingEvent(
                            event_type=EventType.EPOCH_END,
                            phase="nightmare",
                            epoch=epoch + 1,
                            metrics={"avg_loss": avg_epoch_loss},
                        )
                    )
                total_loss += avg_epoch_loss
        finally:
            # Restore original learning rate from saved values
            self._restore_lr(saved_lr_state)

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

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        device: Union[str, torch.device] = "cpu",
        scaler: Optional[torch.amp.GradScaler] = None,
        model_type: str = "causal_lm",
        callback_manager: Optional[CallbackManager] = None,
        lr_scheduler=None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.scaler = scaler
        self.model_type = model_type
        self.callback_manager = callback_manager
        self.lr_scheduler = lr_scheduler

    def run(
        self,
        dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
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

        logger.info("Compression Phase - Method: %s, Ratio: %.2f", method, pruning_ratio)

        # Snapshot teacher BEFORE pruning (for distillation)
        distillation_enabled = self.config.get("distillation", False)
        teacher_model = None
        if distillation_enabled and dataloader is not None and optimizer is not None:
            import copy

            teacher_model = copy.deepcopy(self.model)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            logger.info("Compression Phase - Teacher snapshot taken for distillation.")

        if method == "magnitude":
            try:
                from nightmarenet.compression.pruning import MagnitudePruner

                pruner = MagnitudePruner(pruning_ratio=pruning_ratio)
                stats = pruner.apply(self.model)
                pruning_succeeded = True
            except Exception as exc:
                logger.error("Compression Phase - Failed to apply pruning: %s", exc)
                stats = {"pruned_params": 0, "total_params": 0}
                pruning_succeeded = False
        else:
            logger.warning("Unknown pruning method '%s'; skipping.", method)
            stats = {"pruned_params": 0, "total_params": 0}
            pruning_succeeded = False

        # Distillation step (RSLAD-style): pruned student learns from teacher
        distillation_stats = {}
        if teacher_model is not None and pruning_succeeded:
            from nightmarenet.compression.distillation import run_distillation

            temperature = self.config.get("distillation_temperature", 4.0)
            alpha = self.config.get("distillation_alpha", 0.7)
            distillation_epochs = self.config.get("distillation_epochs", 1)
            logger.info("Compression Phase - Running RSLAD-style distillation...")
            distillation_stats = run_distillation(
                teacher=teacher_model,
                student=self.model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=self.device,
                epochs=distillation_epochs,
                temperature=temperature,
                alpha=alpha,
                scaler=self.scaler,
            )
        # Optional fine-tuning after pruning
        if (
            self.config.get("finetune_after_prune", True)
            and dataloader is not None
            and optimizer is not None
        ):
            finetune_epochs = self.config.get("finetune_epochs", 1)
            logger.info("Fine-tuning after compression for %d epoch(s)...", finetune_epochs)
            self.model.train()
            for epoch in range(finetune_epochs):
                if self.callback_manager is not None:
                    self.callback_manager.emit(
                        TrainingEvent(
                            event_type=EventType.EPOCH_START,
                            phase="compression",
                            epoch=epoch + 1,
                        )
                    )
                for batch in tqdm(
                    dataloader, desc=f"Post-compression fine-tune - Epoch {epoch + 1}"
                ):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    use_amp = self.scaler is not None
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        if self.model_type not in ("seq_classification", "image_classification"):
                            batch["labels"] = batch["input_ids"]
                        outputs = self.model(**batch)
                        loss = outputs.loss

                    if math.isnan(loss.item()) or math.isinf(loss.item()):
                        logger.warning("Compression fine-tune - NaN/Inf loss, skipping.")
                        optimizer.zero_grad()
                        continue

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    optimizer.zero_grad()

                if self.callback_manager is not None:
                    self.callback_manager.emit(
                        TrainingEvent(
                            event_type=EventType.EPOCH_END,
                            phase="compression",
                            epoch=epoch + 1,
                        )
                    )

        return {
            "phase": "compression",
            "method": method,
            "pruning_ratio": pruning_ratio,
            **stats,
            **distillation_stats,
        }
