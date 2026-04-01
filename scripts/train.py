"""CLI entry point: run the full sleep-cycle training pipeline."""

import argparse
import logging
import sys

import yaml

from nightmarenet.data.generator import create_generators_from_config
from nightmarenet.data.loader import load_from_config
from nightmarenet.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
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
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", args.config)

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

    # Load dataset
    logger.info("Loading dataset...")
    dataset_wrapper = load_from_config(config)

    # Create generators and generate dream/nightmare data
    logger.info("Generating dream and nightmare data...")
    dream_gen, nightmare_gen = create_generators_from_config(config)
    dream_data = dream_gen.generate(dataset_wrapper.train_data)
    nightmare_data = nightmare_gen.generate(dataset_wrapper.train_data)

    # Create trainer
    trainer = Trainer(config=config)

    # Tokenize datasets
    from nightmarenet.training.trainer import _tokenize_dataset

    text_column = config.get("dataset", {}).get("text_column", "text")
    max_length = config.get("model", {}).get("max_length", 128)
    batch_size = config.get("training", {}).get("batch_size", 8)

    train_dataloader = _tokenize_dataset(
        dataset_wrapper.train_data, trainer.tokenizer, text_column, max_length, batch_size
    )
    dream_dataloader = _tokenize_dataset(
        dream_data, trainer.tokenizer, text_column, max_length, batch_size
    )
    nightmare_dataloader = _tokenize_dataset(
        nightmare_data, trainer.tokenizer, text_column, max_length, batch_size
    )

    # Run training
    logger.info("Starting training pipeline...")
    history = trainer.train(
        train_dataloader=train_dataloader,
        dream_dataloader=dream_dataloader,
        nightmare_dataloader=nightmare_dataloader,
    )

    logger.info("Training complete. %d phase results recorded.", len(history))
    return history


if __name__ == "__main__":
    main()
