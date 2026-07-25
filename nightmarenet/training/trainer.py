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
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from datasets import IterableDataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from nightmarenet.distributed.checkpoint import AtomicCheckpointer
from nightmarenet.distributed.ddp_wrapper import DDPWrapper
from nightmarenet.distributed.device_pool import DevicePool
from nightmarenet.distributed.resume import ResumeManager
from nightmarenet.distributed.strategies import apply_phase_strategy
from nightmarenet.training.callbacks import (
    CallbackManager,
    EventType,
    TrainingEvent,
)
from nightmarenet.training.phases import (
    CompressionPhase,
    DreamPhase,
    NightmarePhase,
    WakePhase,
)
from nightmarenet.training.scheduler import create_scheduler_from_config
from nightmarenet.utils.tracking import create_tracker_from_config
from nightmarenet.utils.webhooks import check_vram_pressure, trigger_webhook

logger = logging.getLogger(__name__)

_MODEL_TYPE_MAP = {
    "causal_lm": AutoModelForCausalLM,
    "masked_lm": AutoModelForMaskedLM,
    "seq_classification": AutoModelForSequenceClassification,
}


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class VisionModelWrapper(torch.nn.Module):
    def __init__(self, base_model, criterion=None):
        super().__init__()
        self.base_model = base_model
        self.criterion = criterion or torch.nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None, **kwargs):
        logits = self.base_model(pixel_values)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return ModelOutput(loss=loss, logits=logits)


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


def _worker_init_fn(worker_id: int) -> None:
    """Ensure numpy and Python random are seeded correctly inside DataLoader workers."""
    import random

    import numpy as np

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _tokenize_dataset(
    dataset: Any,
    tokenizer: Any,
    text_column: str,
    max_length: int,
    batch_size: int,
    label_column: Optional[str] = None,
) -> DataLoader:
    """Tokenize a dataset and return a DataLoader."""
    if tokenizer is None or not hasattr(dataset, "map"):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, worker_init_fn=_worker_init_fn
        )

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        if label_column is not None and label_column in examples:
            tokenized["labels"] = examples[label_column]
        return tokenized

    if isinstance(dataset, IterableDataset):
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names if dataset.column_names else [text_column],
        )
        tokenized = tokenized.with_format("torch")
        return DataLoader(tokenized, batch_size=batch_size, worker_init_fn=_worker_init_fn)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    tokenized.set_format("torch")
    return DataLoader(
        tokenized, batch_size=batch_size, shuffle=True, worker_init_fn=_worker_init_fn
    )


def _set_reproducibility(seed: int) -> None:
    """Set seeds across Python, NumPy, and PyTorch for full reproducibility."""
    import random

    import numpy as np

    logger.info("Setting reproducibility seed: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        distributed: Optional[str] = None,
        resume_dir: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self.config = config
        self.device = _get_device(config)
        self.training_config = config.get("training", {})
        self.model_config = config.get("model", {})
        self.compression_config = config.get("compression", {})

        # Load model and tokenizer
        resume_from = self.training_config.get("resume_from")
        model_name = self.model_config.get("name", "gpt2")
        self.model_type = self.model_config.get("type", "causal_lm")
        if model is None:
            logger.info(
                "Loading model base architecture: %s (type=%s)",
                model_name,
                self.model_type,
            )
            if self.model_type == "image_classification":
                import torchvision.models as torchvision_models

                model_creator = None
                for name in dir(torchvision_models):
                    if name.lower() == model_name.lower():
                        model_creator = getattr(torchvision_models, name)
                        break
                if model_creator is None:
                    raise ValueError(f"Unknown torchvision model name '{model_name}'.")
                num_classes = self.model_config.get("num_labels", 10)
                try:
                    base_model = model_creator(num_classes=num_classes)
                except TypeError:
                    base_model = model_creator()
                self.model = VisionModelWrapper(base_model)
            else:
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

        # Load weights from checkpoint if resuming
        if resume_from:
            from nightmarenet.distributed.checkpoint import load_model_weights

            load_model_weights(self.model, resume_from, self.device)

        if self.model_type == "image_classification":
            self.tokenizer = None
        elif tokenizer is None:
            has_resume = resume_from and os.path.exists(resume_from)
            tokenizer_load_path = resume_from if has_resume else model_name
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
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
        self.lr_scheduler = None

        # Reference model for KL regularization (created after wake phase)
        self.reference_model: Optional[torch.nn.Module] = None

        # Training history
        self.history: list[dict] = []

        # Interrupt flag
        self._interrupted = False

        # External stop request (e.g. adaptive termination)
        self._stop_requested = False

        # Checkpoint directory
        self.checkpoint_dir = self.training_config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Log directory
        self.log_dir = self.training_config.get("log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Distributed multi-GPU setup
        override_devices = None
        if distributed and distributed != "auto":
            override_devices = [int(x.strip()) for x in distributed.split(",")]
        self.device_pool = DevicePool(override_devices=override_devices)
        self.ddp_wrapper = DDPWrapper()

        if self.device_pool.should_use_ddp():
            self.ddp_wrapper.setup()

        # Checkpointer and Resume
        self.checkpointer = AtomicCheckpointer(self.checkpoint_dir)
        self.resume_manager = ResumeManager(resume_dir) if resume_dir else None

        # Load from resume if provided
        if self.resume_manager:
            metadata = self.resume_manager.verify_and_load(self.model, self.optimizer, self.config)
            self._start_cycle = metadata.get("cycle", 0)
            self._start_phase = metadata.get("phase", None)
            logger.info(f"Resuming from cycle {self._start_cycle}, phase {self._start_phase}")
        else:
            self._start_cycle = 0
            self._start_phase = None

        # Experiment tracker (only on main process)
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            self.tracker = create_tracker_from_config(config)
        else:
            self.tracker = create_tracker_from_config({})

        self.run_id: Optional[str] = None
        self._vram_alert_sent = False
        self.callback_manager = callback_manager or CallbackManager()

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

    def request_stop(self) -> None:
        """Request graceful termination after the current phase."""
        logger.info("External stop requested.")
        self._stop_requested = True

    def _save_checkpoint(self, cycle: int, phase: str):
        """Save a model checkpoint after a phase."""
        if not self.training_config.get("save_every_phase", True):
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        run_id_to_use = getattr(self, "run_id", "default_run")
        if not run_id_to_use:
            run_id_to_use = "default_run"

        metrics = self.history[-1] if self.history else None
        devices_used = self.device_pool.available_devices if hasattr(self, "device_pool") else []

        path = self.checkpointer.save(
            run_id=run_id_to_use,
            cycle=cycle,
            phase=phase,
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            metrics=metrics,
            devices_used=devices_used,
        )

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

        # Save training state
        import time

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": (
                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            ),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "cycle": cycle,
            "phase": phase,
            "history": self.history,
            "metadata": {
                "timestamp": time.time(),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "trainer_class": self.__class__.__name__,
            },
        }
        torch.save(state, os.path.join(path, "training_state.pt"))
        logger.info("Checkpoint saved: %s (including training state)", path)

        try:
            self.tracker.log_artifact(path)
        except Exception as e:
            logger.warning("Failed to log checkpoint artifact to tracker: %s", e)

        # Post-save validation and complete file hashes update
        meta_path = os.path.join(path, "metadata.json")
        try:
            from nightmarenet.distributed.checkpoint import (
                compute_dir_hashes,
                validate_checkpoint_integrity,
            )

            file_hashes = compute_dir_hashes(path)
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    metadata = json.load(f)
            metadata["file_hashes"] = file_hashes
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Post-save validation check
            validate_checkpoint_integrity(path, self.config)
            logger.info("Post-save checkpoint integrity validation passed successfully.")
        except Exception as e:
            logger.error("Post-save checkpoint integrity validation failed: %s", e)
            raise

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
        on_progress: Optional[Callable[[dict], None]] = None,
        dream_generator: Optional[Any] = None,
        nightmare_generator: Optional[Any] = None,
        dream_base_dataset: Optional[Any] = None,
        nightmare_base_dataset: Optional[Any] = None,
    ) -> list[dict]:
        """Run the full sleep-cycle training pipeline.

        Args:
            train_dataloader: DataLoader for wake phase (real data).
            dream_dataloader: DataLoader for dream phase (mildly distorted).
            nightmare_dataloader: DataLoader for nightmare phase (extreme perturbations).
            val_dataloader: Optional DataLoader for validation.
            on_progress: Optional progress callback.
            dream_generator: Optional generator for dream dataset.
            nightmare_generator: Optional generator for nightmare dataset.
            dream_base_dataset: Optional base dataset for dream.
            nightmare_base_dataset: Optional base dataset for nightmare.

        Returns:
            List of phase result dicts (training history).
        """

        # Set the reproducibility seed before training starts (Fixes Issue #64)
        seed = self.config.get("seed", 42)
        _set_reproducibility(seed)

        logger.info("Starting training with schedule:\n%s", self.scheduler.summary())  # type: ignore[union-attr]
        logger.info("Device: %s", self.device)
        self.tracker.log_config(self.config)

        # Native distributed logic is applied per-phase via apply_phase_strategy

        warmup_steps = self.training_config.get("warmup_steps", 0)
        lr_schedule = self.training_config.get("lr_schedule", "none")

        def get_dataloader_steps(dataloader, default_steps=1000):
            try:
                return len(dataloader)
            except (TypeError, AttributeError):
                return default_steps

        self.lr_scheduler = None

        if lr_schedule == "linear_warmup_cosine":
            grad_accum = self.training_config.get("gradient_accumulation_steps", 1)
            steps_per_epoch_train = get_dataloader_steps(train_dataloader)
            steps_per_epoch_dream = get_dataloader_steps(dream_dataloader)
            steps_per_epoch_nightmare = get_dataloader_steps(nightmare_dataloader)

            wake_epochs = self.training_config.get("wake_epochs", 3)
            dream_epochs = self.training_config.get("dream_epochs", 2)
            nightmare_epochs = self.training_config.get("nightmare_epochs", 1)
            num_cycles = self.training_config.get("num_cycles", 3)

            finetune_after_prune = self.compression_config.get("finetune_after_prune", True)
            finetune_epochs = (
                self.compression_config.get("finetune_epochs", 1) if finetune_after_prune else 0
            )

            def get_opt_steps(steps, accum):
                return max(1, steps // accum)

            wake_steps = get_opt_steps(steps_per_epoch_train, grad_accum) * wake_epochs
            dream_steps = get_opt_steps(steps_per_epoch_dream, grad_accum) * dream_epochs
            nightmare_steps = (
                get_opt_steps(steps_per_epoch_nightmare, grad_accum) * nightmare_epochs
            )
            compress_steps = get_opt_steps(steps_per_epoch_train, grad_accum) * finetune_epochs

            total_steps = num_cycles * (wake_steps + dream_steps + nightmare_steps + compress_steps)

            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif lr_schedule != "none" and warmup_steps > 0:

            def warmup_lambda(current_step):
                return min(1.0, current_step / warmup_steps)

            self.lr_scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=warmup_lambda,
            )
        prev_handler = None
        if threading.current_thread() is threading.main_thread():
            prev_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_interrupt)

        current_cycle = 0
        current_phase = "init"
        total_phases = max(len(self.scheduler), 1)
        completed_phases = 0

        # Load training state if resuming
        resume_from = self.training_config.get("resume_from")
        if resume_from:
            from nightmarenet.distributed.checkpoint import validate_checkpoint_integrity

            try:
                # Explicit structural, version and checksum validation of the checkpoint folder
                validate_checkpoint_integrity(resume_from, self.config)
            except Exception as val_err:
                logger.error("Checkpoint integrity validation failed: %s", val_err)
                raise ValueError(f"Cannot resume from corrupted checkpoint: {val_err}") from val_err

            state_path = os.path.join(resume_from, "training_state.pt")
            if os.path.exists(state_path):
                logger.info("Resuming training state from %s", state_path)
                import copy
                import pickle

                state = None
                try:
                    state = torch.load(state_path, map_location=self.device)
                except (pickle.PickleError, KeyError, RuntimeError) as e:
                    logger.error(
                        "Failed to load training state from %s: %s. Continuing with fresh history.",
                        state_path,
                        e,
                    )

                if state is not None:
                    # Validate optimizer state dict compatibility
                    saved_opt = state.get("optimizer_state_dict")
                    if saved_opt and "param_groups" in saved_opt:
                        saved_groups = len(saved_opt["param_groups"])
                        current_groups = len(self.optimizer.param_groups)
                        if saved_groups != current_groups:
                            logger.warning(
                                "Optimizer param group count mismatch (saved: %d, current: %d). "
                                "Skipping loading optimizer state.",
                                saved_groups,
                                current_groups,
                            )
                        else:
                            try:
                                self.optimizer.load_state_dict(saved_opt)
                            except Exception as opt_err:
                                logger.warning("Failed to load optimizer state dict: %s", opt_err)

                    has_sched_state = state.get("lr_scheduler_state_dict") is not None
                    if self.lr_scheduler is not None and has_sched_state:
                        try:
                            self.lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
                            logger.info("Resumed learning rate scheduler state.")
                        except Exception as lr_err:
                            logger.warning(
                                "Failed to load learning rate scheduler state dict: %s",
                                lr_err,
                            )

                    if self.scaler is not None and state.get("scaler_state_dict") is not None:
                        try:
                            self.scaler.load_state_dict(state["scaler_state_dict"])
                        except Exception as scaler_err:
                            logger.warning("Failed to load scaler state dict: %s", scaler_err)

                    self.history = copy.deepcopy(state.get("history", []))
                    completed_phases = len(self.history)

                    start_cycle = state.get("cycle", 0)
                    start_phase = state.get("phase")

                    # Validate resume point
                    valid_phases = ["wake", "dream", "nightmare", "compress"]
                    if hasattr(self.scheduler, "PHASE_ORDER"):
                        valid_phases = self.scheduler.PHASE_ORDER
                    elif hasattr(self.scheduler, "base_scheduler") and hasattr(
                        self.scheduler.base_scheduler, "PHASE_ORDER"
                    ):
                        valid_phases = self.scheduler.base_scheduler.PHASE_ORDER

                    if start_phase not in valid_phases:
                        raise ValueError(
                            f"Corrupted start_phase '{start_phase}' in checkpoint state. "
                            f"Must be one of {valid_phases}"
                        )

                    if start_phase:
                        if hasattr(self.scheduler, "base_scheduler"):
                            self.scheduler.base_scheduler.start_cycle = start_cycle
                            self.scheduler.base_scheduler.start_phase = start_phase
                        else:
                            self.scheduler.start_cycle = start_cycle
                            self.scheduler.start_phase = start_phase
                        logger.info(
                            "Scheduler configured to resume after cycle %d, phase %s",
                            start_cycle,
                            start_phase,
                        )
            else:
                logger.warning(
                    "resume_from path specified but training_state.pt not found: %s",
                    state_path,
                )

        last_cycle_for_strength = -1
        try:
            for cycle, phase, num_epochs in self.scheduler:
                if cycle < getattr(self, "_start_cycle", 0):
                    logger.info(f"Skipping cycle {cycle} (resuming from {self._start_cycle})")
                    continue

                if cycle != last_cycle_for_strength:
                    last_cycle_for_strength = cycle
                    distortion_config = self.config.get("distortion", {})
                    schedule_across_cycles = distortion_config.get(
                        "schedule_across_cycles",
                        False,
                    )
                    regenerate_dream = False
                    regenerate_nightmare = False
                    cycle_strength = None

                    if schedule_across_cycles:
                        num_cycles = self.training_config.get("num_cycles", 3)
                        strength_min = distortion_config.get("strength_min", 0.2)
                        strength_max = distortion_config.get("strength_max", 0.9)

                        if num_cycles > 1:
                            difference = strength_max - strength_min
                            cycle_strength = strength_min + difference * cycle / (num_cycles - 1)
                        else:
                            cycle_strength = strength_max

                        logger.info(
                            "Cycle %d: Scheduled distortion strength = %.4f",
                            cycle + 1,
                            cycle_strength,
                        )
                        if dream_generator is not None:
                            dream_generator.strength = cycle_strength
                            regenerate_dream = True
                        if nightmare_generator is not None:
                            nightmare_generator.strength = cycle_strength
                            regenerate_nightmare = True

                    if nightmare_generator is not None:
                        if hasattr(nightmare_generator, "set_target_model"):
                            nightmare_generator.set_target_model(
                                self.model,
                                self.tokenizer,
                            )
                        if hasattr(nightmare_generator, "set_cycle"):
                            nightmare_generator.set_cycle(cycle)

                        # Cycle 0 was generated during Pipeline.prepare(). Later cycles
                        # must be regenerated with current weights for model-aware attacks.
                        if cycle > 0 and getattr(
                            nightmare_generator,
                            "uses_gradient_learned",
                            False,
                        ):
                            regenerate_nightmare = True

                    if regenerate_dream or regenerate_nightmare:
                        text_column = self.config.get("dataset", {}).get(
                            "text_column",
                            "text",
                        )
                        max_length = self.config.get("model", {}).get(
                            "max_length",
                            128,
                        )
                        batch_size = self.training_config.get("batch_size", 8)

                        if (
                            regenerate_dream
                            and dream_generator is not None
                            and dream_base_dataset is not None
                        ):
                            logger.info(
                                "Regenerating dream dataset for cycle %d%s",
                                cycle + 1,
                                (
                                    f" at strength {cycle_strength:.4f}"
                                    if cycle_strength is not None
                                    else ""
                                ),
                            )
                            dream_data = dream_generator.generate(dream_base_dataset)
                            dream_dataloader = _tokenize_dataset(
                                dream_data,
                                self.tokenizer,
                                text_column,
                                max_length,
                                batch_size,
                            )

                        if (
                            regenerate_nightmare
                            and nightmare_generator is not None
                            and nightmare_base_dataset is not None
                        ):
                            logger.info(
                                "Regenerating nightmare dataset for cycle %d%s",
                                cycle + 1,
                                (
                                    f" at strength {cycle_strength:.4f}"
                                    if cycle_strength is not None
                                    else " with current target-model gradients"
                                ),
                            )
                            nightmare_data = nightmare_generator.generate(nightmare_base_dataset)
                            nightmare_dataloader = _tokenize_dataset(
                                nightmare_data,
                                self.tokenizer,
                                text_column,
                                max_length,
                                batch_size,
                            )

                if cycle == getattr(self, "_start_cycle", 0):
                    start_phase = getattr(self, "_start_phase", None)
                    if start_phase and not getattr(self, "_resume_caught_up", False):
                        if phase != start_phase:
                            logger.info(
                                f"Skipping phase {phase} in cycle {cycle} (already completed)"
                            )
                            continue
                        else:
                            logger.info(
                                f"Skipping phase {phase} in cycle {cycle} (already completed)"
                            )
                            self._resume_caught_up = True
                            continue

                if self._interrupted:
                    break

                if self._stop_requested:
                    logger.info("Training stop requested. Terminating after cycle %d.", cycle + 1)
                    break
                # Early stopping check
                if hasattr(self.scheduler, "should_stop") and self.scheduler.should_stop:
                    logger.info("Early stopping: halting training at cycle %d.", cycle + 1)
                    break
                current_cycle = cycle
                current_phase = phase
                if on_progress is not None:
                    try:
                        on_progress(
                            {
                                "cycle": cycle,
                                "phase": phase,
                                "progress_pct": (completed_phases / total_phases) * 100.0,
                                "status": "phase_start",
                            }
                        )

                    except Exception:
                        logger.debug("on_progress callback failed", exc_info=True)
                logger.info(
                    "=== Cycle %d - Phase: %s (%d epochs) ===",
                    cycle + 1,
                    phase,
                    num_epochs,
                )

                self.callback_manager.emit(
                    TrainingEvent(
                        event_type=EventType.PHASE_START,
                        phase=phase,
                        metadata={
                            "cycle": cycle,
                        },
                    )
                )

                result: dict

                phase_model = apply_phase_strategy(
                    phase=phase,
                    model=self.model,
                    device_pool=self.device_pool,
                    ddp_wrapper=self.ddp_wrapper,
                )

                if phase == "wake":
                    wake_runner = WakePhase(
                        model=phase_model,
                        model_type=self.model_type,
                        optimizer=self.optimizer,
                        config=self.training_config,
                        device=self.device,
                        scaler=self.scaler,
                        callback_manager=self.callback_manager,
                        lr_scheduler=self.lr_scheduler,
                    )
                    result = wake_runner.run(train_dataloader, num_epochs=num_epochs)

                    # Create reference model after first wake phase
                    if cycle == 0:
                        self._create_reference_model()

                elif phase == "dream":
                    if self.reference_model is None:
                        self._create_reference_model()
                    dream_runner = DreamPhase(
                        model=phase_model,
                        optimizer=self.optimizer,
                        config=self.training_config,
                        device=self.device,
                        reference_model=self.reference_model,
                        kl_weight=0.1,
                        scaler=self.scaler,
                        model_type=self.model_type,
                        callback_manager=self.callback_manager,
                        lr_scheduler=self.lr_scheduler,
                    )
                    result = dream_runner.run(dream_dataloader, num_epochs=num_epochs)

                elif phase == "nightmare":
                    lr_multiplier = self.training_config.get("nightmare_lr_multiplier", 2.0)
                    nightmare_runner = NightmarePhase(
                        model=phase_model,
                        optimizer=self.optimizer,
                        config=self.training_config,
                        device=self.device,
                        lr_multiplier=lr_multiplier,
                        scaler=self.scaler,
                        model_type=self.model_type,
                        callback_manager=self.callback_manager,
                        lr_scheduler=self.lr_scheduler,
                    )
                    result = nightmare_runner.run(nightmare_dataloader, num_epochs=num_epochs)

                elif phase == "compress":
                    compress_runner = CompressionPhase(
                        model=phase_model,
                        config=self.compression_config,
                        device=self.device,
                        scaler=self.scaler,
                        model_type=self.model_type,
                        callback_manager=self.callback_manager,
                        lr_scheduler=self.lr_scheduler,
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
                result["phase"] = phase
                self.history.append(result)
                completed_phases += 1

                if on_progress is not None:
                    try:
                        on_progress(
                            {
                                "cycle": cycle,
                                "phase": phase,
                                "avg_loss": result.get("avg_loss"),
                                "progress_pct": (completed_phases / total_phases) * 100.0,
                                "status": "phase_end",
                                "history": list(self.history),
                            }
                        )
                    except Exception:
                        logger.debug("on_progress callback failed", exc_info=True)

                self.callback_manager.emit(
                    TrainingEvent(
                        event_type=EventType.PHASE_END,
                        phase=phase,
                        metrics={
                            "avg_loss": result.get("avg_loss", 0.0),
                        },
                        metadata={
                            "cycle": cycle,
                        },
                    )
                )

                # Log to tracker
                self.tracker.log_phase(cycle, phase, result)

                # Update adaptive scheduler with phase loss
                avg_loss = result.get("avg_loss")
                if hasattr(self.scheduler, "update") and avg_loss is not None:
                    self.scheduler.update(phase, avg_loss)

                # Save checkpoint
                self._save_checkpoint(cycle, phase)

                self.callback_manager.emit(
                    TrainingEvent(
                        event_type=EventType.CHECKPOINT,
                        phase=phase,
                        metadata={
                            "cycle": cycle,
                        },
                    )
                )

                # Check GPU VRAM pressure
                if self.device.type == "cuda":
                    device_idx = self.device.index if self.device.index is not None else 0
                    has_pressure = check_vram_pressure(device_idx, threshold=0.85)
                    if not getattr(self, "_vram_alert_sent", False) and has_pressure:
                        self._vram_alert_sent = True
                        try:
                            free, total = torch.cuda.mem_get_info(device_idx)
                            used = total - free
                            pct = (used / total) * 100.0
                            gpu_name = torch.cuda.get_device_name(device_idx)
                            trigger_webhook(
                                self.config,
                                "alert",
                                f"GPU VRAM pressure detected: {pct:.1f}% used on {gpu_name}.",
                                {
                                    "run_id": getattr(self, "run_id", "unknown"),
                                    "gpu": gpu_name,
                                    "used_vram": f"{used / (1024**2):.1f} MB",
                                    "total_vram": f"{total / (1024**2):.1f} MB",
                                    "usage_percent": f"{pct:.1f}%",
                                    "cycle": cycle + 1,
                                    "phase": phase,
                                },
                            )
                        except Exception as e:
                            logger.warning("Failed to record VRAM alert details: %s", e)

                # Log metrics
                logger.info("Phase result: %s", json.dumps(result, indent=2, default=str))

                # Notify pipeline that a full training cycle has completed
                if phase == "compress" and on_progress is not None:
                    try:
                        on_progress(
                            {
                                "event": "cycle_end",
                                "cycle": cycle,
                                "phase": phase,
                                "history": list(self.history),
                            }
                        )
                    except Exception:
                        logger.debug("on_progress callback failed", exc_info=True)
        except KeyboardInterrupt:
            logger.warning("Training interrupted by KeyboardInterrupt.")
        finally:
            if self._interrupted:
                self._save_checkpoint(current_cycle, current_phase)
                logger.info("Training interrupted, checkpoint saved.")
            if prev_handler is not None:
                signal.signal(signal.SIGINT, prev_handler)

        # Save final model and history
        final_path = os.path.join(self.checkpoint_dir, "final")
        os.makedirs(final_path, exist_ok=True)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(final_path)
            else:
                torch.save(self.model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(final_path)
            self._save_history()

        if dist.is_available() and dist.is_initialized():
            self.ddp_wrapper.teardown()

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
