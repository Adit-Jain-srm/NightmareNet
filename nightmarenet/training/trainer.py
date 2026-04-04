"""Main training loop orchestrator.

Manages the full sleep-cycle training pipeline: instantiates phases,
manages checkpointing, and logs per-phase metrics.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import signal
from typing import Any, Optional

import torch
from datasets import IterableDataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from nightmarenet.training.distributed import DistributedContext
from nightmarenet.training.phases import (
    CompressionPhase,
    DreamPhase,
    NightmarePhase,
    WakePhase,
)
from nightmarenet.training.scheduler import create_scheduler_from_config
from nightmarenet.utils.tracking import create_tracker_from_config

logger = logging.getLogger(__name__)

_MODEL_TYPE_MAP = {
    "causal_lm": AutoModelForCausalLM,
    "masked_lm": AutoModelForMaskedLM,
    "seq_classification": AutoModelForSequenceClassification,
}


def _get_device(config: dict) -> torch.device:
    """Determine the training device from config."""
    device_str = config.get("model", {}).get("device", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _create_amp_scaler(use_amp: bool, device: torch.device) -> Optional[torch.amp.GradScaler]:
    """Create a GradScaler for mixed-precision training, or None if disabled."""
    if use_amp and device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def _tokenize_dataset(
    dataset: Any,
    tokenizer: Any,
    text_column: str,
    max_length: int,
    batch_size: int,
) -> DataLoader:
    """Tokenize a dataset and return a DataLoader."""

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    if isinstance(dataset, IterableDataset):
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names if dataset.column_names else [text_column],
        )
        tokenized = tokenized.with_format("torch")
        return DataLoader(tokenized, batch_size=batch_size)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    tokenized.set_format("torch")
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)


class Trainer:
    """Orchestrates the full sleep-cycle training pipeline.

    Args:
        config: Full configuration dictionary.
        model: Optional pre-loaded model. If None, loads from config.
        tokenizer: Optional pre-loaded tokenizer. If None, loads from config.
    """

    def __init__(
        self,
        config: dict,
        model=None,
        tokenizer=None,
    ):
        self.config = config
        self.device = _get_device(config)
        self.training_config = config.get("training", {})
        self.model_config = config.get("model", {})
        self.compression_config = config.get("compression", {})

        # Load model and tokenizer
        model_name = self.model_config.get("name", "gpt2")
        self.model_type = self.model_config.get("type", "causal_lm")
        if model is None:
            logger.info("Loading model: %s (type=%s)", model_name, self.model_type)
            model_cls = _MODEL_TYPE_MAP.get(self.model_type)
            if model_cls is None:
                raise ValueError(
                    f"Unknown model type '{self.model_type}'. "
                    f"Supported: {list(_MODEL_TYPE_MAP)}"
                )
            try:
                kwargs = {}
                if self.model_type == "seq_classification":
                    kwargs["num_labels"] = self.model_config.get("num_labels", 2)
                self.model = model_cls.from_pretrained(model_name, **kwargs)
            except Exception as exc:
                raise RuntimeError(f"Failed to load model '{model_name}': {exc}") from exc
        else:
            self.model = model
        self.model.to(self.device)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.get("learning_rate", 5e-5),
            weight_decay=self.training_config.get("weight_decay", 0.01),
        )

        # Mixed-precision training
        self.use_amp = self.training_config.get("use_amp", False)
        self.scaler = _create_amp_scaler(self.use_amp, self.device)
        if self.use_amp:
            if self.device.type == "cuda":
                logger.info("Mixed-precision training (AMP) enabled.")
            else:
                logger.warning("use_amp=True but device is %s; AMP disabled.", self.device)
                self.use_amp = False
                self.scaler = None

        # Gradient checkpointing
        if self.training_config.get("gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled.")
            else:
                logger.warning("Model does not support gradient checkpointing.")

        # Create scheduler
        self.scheduler = create_scheduler_from_config(config)

        # Reference model for KL regularization (created after wake phase)
        self.reference_model = None

        # Training history
        self.history: list[dict] = []

        # Interrupt flag
        self._interrupted = False

        # Checkpoint directory
        self.checkpoint_dir = self.training_config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Log directory
        self.log_dir = self.training_config.get("log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Distributed training context
        distributed_cfg = self.training_config.get("distributed", False)
        amp_mode = "fp16" if self.use_amp else "no"
        self.dist_ctx = DistributedContext(
            enabled=distributed_cfg,
            mixed_precision=amp_mode,
            gradient_accumulation_steps=self.training_config.get(
                "gradient_accumulation_steps", 1
            ),
        )

        # Experiment tracker (only on main process to avoid duplicates)
        if self.dist_ctx.is_main_process:
            self.tracker = create_tracker_from_config(config)
        else:
            self.tracker = create_tracker_from_config({})

    def _create_reference_model(self) -> None:
        """Create a frozen copy of the current model for KL regularization."""
        self.reference_model = copy.deepcopy(self.model)
        self.reference_model.eval()  # type: ignore[attr-defined]
        for param in self.reference_model.parameters():  # type: ignore[attr-defined]
            param.requires_grad = False
        logger.info("Created reference model for KL regularization.")

    def _handle_interrupt(self, signum, frame) -> None:
        """Handle SIGINT by flagging for a graceful checkpoint save."""
        logger.warning("Received interrupt signal, will save checkpoint and stop.")
        self._interrupted = True

    def _save_checkpoint(self, cycle: int, phase: str):
        """Save a model checkpoint after a phase."""
        if not self.training_config.get("save_every_phase", True):
            return
        if not self.dist_ctx.is_main_process:
            return

        path = os.path.join(self.checkpoint_dir, f"cycle{cycle}_{phase}")
        os.makedirs(path, exist_ok=True)
        if self.dist_ctx.enabled:
            self.dist_ctx.save_model(self.model, path)
        else:
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Checkpoint saved: %s", path)

    def _save_history(self):
        """Save training history to a JSON file."""
        path = os.path.join(self.log_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training history saved: %s", path)

    def train(
        self,
        train_dataloader: DataLoader,
        dream_dataloader: DataLoader,
        nightmare_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> list[dict]:
        """Run the full sleep-cycle training pipeline.

        Args:
            train_dataloader: DataLoader for wake phase (real data).
            dream_dataloader: DataLoader for dream phase (mildly distorted).
            nightmare_dataloader: DataLoader for nightmare phase (extreme perturbations).
            val_dataloader: Optional DataLoader for validation.

        Returns:
            List of phase result dicts (training history).
        """
        logger.info("Starting training with schedule:\n%s", self.scheduler.summary())  # type: ignore[union-attr]
        logger.info("Device: %s", self.device)
        self.tracker.log_config(self.config)

        # Prepare model/optimizer/dataloaders for distributed training
        if self.dist_ctx.enabled:
            loaders = [train_dataloader, dream_dataloader, nightmare_dataloader]
            if val_dataloader is not None:
                loaders.append(val_dataloader)
            prepared = self.dist_ctx.prepare(self.model, self.optimizer, *loaders)
            self.model = prepared[0]
            self.optimizer = prepared[1]
            train_dataloader = prepared[2]
            dream_dataloader = prepared[3]
            nightmare_dataloader = prepared[4]
            if val_dataloader is not None:
                val_dataloader = prepared[5]
            self.device = self.dist_ctx.device

        prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        current_cycle = 0
        current_phase = "init"
        try:
            for cycle, phase, num_epochs in self.scheduler:
                if self._interrupted:
                    break
                # Early stopping check
                if hasattr(self.scheduler, 'should_stop') and self.scheduler.should_stop:
                    logger.info("Early stopping: halting training at cycle %d.", cycle + 1)
                    break
                current_cycle = cycle
                current_phase = phase
                logger.info(
                    "=== Cycle %d - Phase: %s (%d epochs) ===",
                    cycle + 1,
                    phase,
                    num_epochs,
                )

                result: dict

                if phase == "wake":
                    wake_runner = WakePhase(
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.training_config,
                        device=self.device,
                        scaler=self.scaler,
                    )
                    result = wake_runner.run(train_dataloader, num_epochs=num_epochs)

                    # Create reference model after first wake phase
                    if cycle == 0:
                        self._create_reference_model()

                elif phase == "dream":
                    dream_runner = DreamPhase(
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.training_config,
                        device=self.device,
                        reference_model=self.reference_model,
                        kl_weight=0.1,
                        scaler=self.scaler,
                    )
                    result = dream_runner.run(dream_dataloader, num_epochs=num_epochs)

                elif phase == "nightmare":
                    lr_multiplier = self.training_config.get("nightmare_lr_multiplier", 2.0)
                    nightmare_runner = NightmarePhase(
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.training_config,
                        device=self.device,
                        lr_multiplier=lr_multiplier,
                        scaler=self.scaler,
                    )
                    result = nightmare_runner.run(nightmare_dataloader, num_epochs=num_epochs)

                elif phase == "compress":
                    compress_runner = CompressionPhase(
                        model=self.model,
                        config=self.compression_config,
                        device=self.device,
                        scaler=self.scaler,
                    )
                    result = compress_runner.run(
                        dataloader=train_dataloader,
                        optimizer=self.optimizer,
                    )
                else:
                    logger.warning("Unknown phase: %s", phase)
                    continue

                # NaN/Inf detection on phase avg_loss
                avg_loss = result.get("avg_loss")
                if avg_loss is not None and (math.isnan(avg_loss) or math.isinf(avg_loss)):
                    logger.critical(
                        "Phase '%s' in cycle %d returned NaN/Inf avg_loss: %s",
                        phase,
                        cycle,
                        avg_loss,
                    )

                result["cycle"] = cycle
                self.history.append(result)

                # Log to tracker
                self.tracker.log_phase(cycle, phase, result)

                # Update adaptive scheduler with phase loss
                avg_loss = result.get("avg_loss")
                if hasattr(self.scheduler, 'update') and avg_loss is not None:
                    self.scheduler.update(phase, avg_loss)

                # Save checkpoint
                self._save_checkpoint(cycle, phase)

                # Log metrics
                logger.info("Phase result: %s", json.dumps(result, indent=2, default=str))
        except KeyboardInterrupt:
            logger.warning("Training interrupted by KeyboardInterrupt.")
        finally:
            if self._interrupted:
                self._save_checkpoint(current_cycle, current_phase)
                logger.info("Training interrupted, checkpoint saved.")
            signal.signal(signal.SIGINT, prev_handler)

        # Save final model and history
        final_path = os.path.join(self.checkpoint_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        if self.dist_ctx.enabled:
            self.dist_ctx.wait_for_everyone()
            self.dist_ctx.save_model(self.model, final_path)
            if self.dist_ctx.is_main_process:
                self.tokenizer.save_pretrained(final_path)
                self._save_history()
        else:
            self.model.save_pretrained(final_path)
            self.tokenizer.save_pretrained(final_path)
            self._save_history()

        self.tracker.finish()
        logger.info("Training complete. Final model saved to %s", final_path)
        return self.history


def create_trainer_from_config(config: dict, model=None, tokenizer=None) -> Trainer:
    """Create a Trainer instance from a configuration dictionary.

    Args:
        config: Full configuration dictionary.
        model: Optional pre-loaded model.
        tokenizer: Optional pre-loaded tokenizer.

    Returns:
        Configured Trainer instance.
    """
    return Trainer(config=config, model=model, tokenizer=tokenizer)
