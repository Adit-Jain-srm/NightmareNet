"""Data preparation phase: dream/nightmare splits, tokenization, trainer setup.

Extracted from `Pipeline.prepare()` (nightmarenet/pipeline.py, L~430-560).
Behavior unchanged: builds train/eval split, generates dream/nightmare
datasets, creates the Trainer (loading model + tokenizer), snapshots the
baseline model, and tokenizes all four dataloaders.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from nightmarenet.data.generator import create_generators_from_config
from nightmarenet.distortions.text import apply_text_distortions
from nightmarenet.phases.base import Phase, PhaseResult, PipelineContext
from nightmarenet.training.callbacks import CallbackManager
from nightmarenet.training.trainer import Trainer, _tokenize_dataset
from nightmarenet.utils.telemetry import trace_phase

logger = logging.getLogger(__name__)

_MIN_EVAL_SAMPLES = 25


class PreparePhase(Phase):
    name = "prepare"

    def __init__(self, on_training_event: Optional[Any] = None) -> None:
        # Optional callback wired into the CallbackManager, e.g. for
        # orchestrator-level logging of training events.
        self.on_training_event = on_training_event

    def execute(self, context: PipelineContext) -> PhaseResult:
        if context.dataset is None:
            return PhaseResult(
                success=False, phase_name=self.name, error="No dataset: run ingest before prepare."
            )

        model_name = context.config.get("model", {}).get("name", "unknown")
        with trace_phase("prepare", {"model.name": model_name}):
            try:
                dream_gen, nightmare_gen = create_generators_from_config(context.config)

                # Train / eval split
                eval_split_ratio = context.config.get("evaluation", {}).get("eval_split_ratio", 0.2)
                base_for_split = (
                    context.wake_dataset if context.wake_dataset is not None else context.dataset
                )
                n_total = len(base_for_split)

                model_type = context.config.get("model", {}).get("type", "")
                if model_type == "image_classification":
                    wake_data = base_for_split
                    if context.eval_dataset is None:
                        context.eval_dataset = base_for_split
                    context.distortion_fn = lambda x, strength: x
                elif eval_split_ratio > 0.0 and n_total >= _MIN_EVAL_SAMPLES:
                    n_eval = max(1, int(n_total * eval_split_ratio))
                    n_train = n_total - n_eval
                    wake_data = base_for_split.select(list(range(n_train)))
                    context.eval_dataset = base_for_split.select(list(range(n_train, n_total)))
                    logger.info(
                        "Train/eval split: %d train, %d eval (ratio=%.2f).",
                        n_train,
                        n_eval,
                        eval_split_ratio,
                    )
                    context.distortion_fn = apply_text_distortions
                else:
                    if eval_split_ratio > 0.0:
                        logger.warning(
                            "Dataset has only %d samples (minimum %d for splitting). "
                            "Using full dataset for both training and evaluation.",
                            n_total,
                            _MIN_EVAL_SAMPLES,
                        )
                    wake_data = base_for_split
                    context.eval_dataset = base_for_split
                    context.distortion_fn = apply_text_distortions

                dream_base = (
                    context.dream_base if context.dream_base is not None else context.dataset
                )
                nightmare_base = (
                    context.nightmare_base
                    if context.nightmare_base is not None
                    else context.dataset
                )

                context.dream_generator = dream_gen
                context.nightmare_generator = nightmare_gen
                context.dream_base_dataset = dream_base
                context.nightmare_base_dataset = nightmare_base

                uses_gradient_learned = nightmare_gen.uses_gradient_learned
                if uses_gradient_learned:
                    # The target model must exist before cycle-zero nightmare data
                    # is generated, so build the trainer early in this branch.
                    context.callback_manager = CallbackManager()
                    context.trainer = Trainer(
                        config=context.config,
                        distributed=context.distributed,
                        resume_dir=context.resume_dir,
                        callback_manager=context.callback_manager,
                    )
                    assert context.trainer is not None
                    if self.on_training_event is not None:
                        context.callback_manager.on_all(self.on_training_event)
                    context.trainer.run_id = context.run_id
                    nightmare_gen.set_target_model(
                        context.trainer.model,
                        context.trainer.tokenizer,
                    )
                    nightmare_gen.set_cycle(0)

                dream_data = dream_gen.generate(dream_base)
                nightmare_data = nightmare_gen.generate(nightmare_base)

                if not uses_gradient_learned:
                    context.callback_manager = CallbackManager()
                    context.trainer = Trainer(
                        config=context.config,
                        distributed=context.distributed,
                        resume_dir=context.resume_dir,
                        callback_manager=context.callback_manager,
                    )
                    assert context.trainer is not None
                    if self.on_training_event is not None:
                        context.callback_manager.on_all(self.on_training_event)
                    context.trainer.run_id = context.run_id

                # Snapshot baseline model weights for later evaluation. Gradient
                # generation uses autograd.grad and does not mutate model params.
                trainer = context.trainer
                assert trainer is not None
                context.baseline_model = copy.deepcopy(trainer.model)
                context.baseline_model.eval()

                text_column = context.config.get("dataset", {}).get("text_column", "text")
                max_length = context.config.get("model", {}).get("max_length", 128)
                batch_size = context.config.get("training", {}).get("batch_size", 8)

                context.train_dl = _tokenize_dataset(
                    wake_data, trainer.tokenizer, text_column, max_length, batch_size
                )
                context.eval_dl = _tokenize_dataset(
                    context.eval_dataset,
                    trainer.tokenizer,
                    text_column,
                    max_length,
                    batch_size,
                )
                context.dream_dl = _tokenize_dataset(
                    dream_data, trainer.tokenizer, text_column, max_length, batch_size
                )
                context.nightmare_dl = _tokenize_dataset(
                    nightmare_data, trainer.tokenizer, text_column, max_length, batch_size
                )
                logger.info("Preparation complete: dataloaders ready.")
            except (ValueError, RuntimeError, OSError) as exc:
                logger.exception("Preparation failed.")
                return PhaseResult(success=False, phase_name=self.name, error=str(exc))

        return PhaseResult(success=True, phase_name=self.name)
