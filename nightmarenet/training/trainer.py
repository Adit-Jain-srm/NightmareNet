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
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from nightmarenet.training.phases import (
    CompressionPhase,
    DreamPhase,
    NightmarePhase,
    WakePhase,
)
from nightmarenet.training.scheduler import CyclicScheduler, create_scheduler_from_config

logger = logging.getLogger(__name__)


def _get_device(config):
    """Determine the training device from config."""
    device_str = config.get("model", {}).get("device", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _tokenize_dataset(dataset, tokenizer, text_column, max_length, batch_size):
    """Tokenize a dataset and return a DataLoader."""

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

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
        if model is None:
            logger.info("Loading model: %s", model_name)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
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

        # Create scheduler
        self.scheduler = create_scheduler_from_config(config)

        # Reference model for KL regularization (created after wake phase)
        self.reference_model = None

        # Training history
        self.history = []

        # Interrupt flag
        self._interrupted = False

        # Checkpoint directory
        self.checkpoint_dir = self.training_config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Log directory
        self.log_dir = self.training_config.get("log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def _create_reference_model(self):
        """Create a frozen copy of the current model for KL regularization."""
        self.reference_model = copy.deepcopy(self.model)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
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

        path = os.path.join(self.checkpoint_dir, f"cycle{cycle}_{phase}")
        os.makedirs(path, exist_ok=True)
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
        logger.info("Starting training with schedule:\n%s", self.scheduler.summary())
        logger.info("Device: %s", self.device)

        prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        current_cycle = 0
        current_phase = "init"
        try:
            for cycle, phase, num_epochs in self.scheduler:
                if self._interrupted:
                    break
                current_cycle = cycle
                current_phase = phase
                logger.info(
                    "=== Cycle %d - Phase: %s (%d epochs) ===",
                    cycle + 1,
                    phase,
                    num_epochs,
                )

                if phase == "wake":
                    phase_runner = WakePhase(
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.training_config,
                        device=self.device,
                    )
                    result = phase_runner.run(train_dataloader, num_epochs=num_epochs)

                    # Create reference model after first wake phase
                    if cycle == 0:
                        self._create_reference_model()

                elif phase == "dream":
                    phase_runner = DreamPhase(
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.training_config,
                        device=self.device,
                        reference_model=self.reference_model,
                        kl_weight=0.1,
                    )
                    result = phase_runner.run(dream_dataloader, num_epochs=num_epochs)

                elif phase == "nightmare":
                    lr_multiplier = self.training_config.get("nightmare_lr_multiplier", 2.0)
                    phase_runner = NightmarePhase(
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.training_config,
                        device=self.device,
                        lr_multiplier=lr_multiplier,
                    )
                    result = phase_runner.run(nightmare_dataloader, num_epochs=num_epochs)

                elif phase == "compress":
                    phase_runner = CompressionPhase(
                        model=self.model,
                        config=self.compression_config,
                        device=self.device,
                    )
                    result = phase_runner.run(
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
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        self._save_history()

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
