"""CLI entry point: run the full sleep-cycle training pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from nightmarenet.data.generator import create_generators_from_config
from nightmarenet.data.loader import load_from_config
from nightmarenet.training.trainer import Trainer
from nightmarenet.utils.config import load_config
from nightmarenet.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="NightmareNet: Run the full sleep-cycle training pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and data but skip training. Prints config summary and exits.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        choices=["none", "wandb", "tensorboard", "mlflow"],
        help="Override tracking backend (none, wandb, tensorboard, mlflow).",
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    try:
        # Load config
        config = load_config(args.config)
        logger.info("Loaded config from %s", args.config)

        # Override tracker backend from CLI
        if args.tracker is not None:
            config.setdefault("tracking", {})["backend"] = args.tracker

        # Set seed
        seed = config.get("seed", 42)
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if args.dry_run:
            logger.info("Dry-run mode: skipping data generation and training.")
            logger.info(
                "Config summary: model=%s, cycles=%d, batch_size=%d, lr=%s",
                config["model"]["name"],
                config["training"]["num_cycles"],
                config["training"]["batch_size"],
                config["training"]["learning_rate"],
            )
            return None

        # Load dataset
        logger.info("Loading dataset...")
        dataset_wrapper = load_from_config(config)

        # Create generators and generate dream/nightmare data. The gradient
        # strategy needs the current model before cycle-zero data generation;
        # disabled and attention strategies preserve the original ordering.
        logger.info("Generating dream and nightmare data...")
        dream_gen, nightmare_gen = create_generators_from_config(config)
        trainer = None
        if nightmare_gen.uses_gradient_learned:
            trainer = Trainer(config=config)
            nightmare_gen.set_target_model(trainer.model, trainer.tokenizer)
            nightmare_gen.set_cycle(0)

        dream_data = dream_gen.generate(dataset_wrapper.train_data)
        nightmare_data = nightmare_gen.generate(dataset_wrapper.train_data)

        if trainer is None:
            trainer = Trainer(config=config)

        # Tokenize datasets
        from nightmarenet.training.trainer import _tokenize_dataset

        text_column = config.get("dataset", {}).get("text_column", "text")
        max_length = config.get("model", {}).get("max_length", 128)
        batch_size = config.get("training", {}).get("batch_size", 8)
        label_column = config["dataset"].get("label_column")
        train_dataloader = _tokenize_dataset(
            dataset_wrapper.train_data,
            trainer.tokenizer,
            text_column,
            max_length,
            batch_size,
            label_column,
        )
        dream_dataloader = _tokenize_dataset(
            dream_data, trainer.tokenizer, text_column, max_length, batch_size, label_column
        )
        nightmare_dataloader = _tokenize_dataset(
            nightmare_data, trainer.tokenizer, text_column, max_length, batch_size, label_column
        )

        # Run training
        logger.info("Starting training pipeline...")
        history = trainer.train(
            train_dataloader=train_dataloader,
            dream_dataloader=dream_dataloader,
            nightmare_dataloader=nightmare_dataloader,
            dream_generator=dream_gen,
            nightmare_generator=nightmare_gen,
            dream_base_dataset=dataset_wrapper.train_data,
            nightmare_base_dataset=dataset_wrapper.train_data,
        )

        logger.info("Training complete. %d phase results recorded.", len(history))
        return history

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
