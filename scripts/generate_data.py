"""CLI entry point: generate dream and nightmare datasets."""

import argparse
import logging
import sys

import yaml

from nightmarenet.data.generator import create_generators_from_config
from nightmarenet.data.loader import load_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="NightmareNet: Generate dream and nightmare datasets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated/",
        help="Output directory for generated datasets.",
    )
    parser.add_argument(
        "--dream-only",
        action="store_true",
        help="Generate only dream data.",
    )
    parser.add_argument(
        "--nightmare-only",
        action="store_true",
        help="Generate only nightmare data.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", args.config)

    # Set seed
    seed = config.get("seed", 42)
    import random

    random.seed(seed)

    # Load base dataset
    logger.info("Loading base dataset...")
    dataset_wrapper = load_from_config(config)

    # Create generators
    dream_gen, nightmare_gen = create_generators_from_config(config)

    # Generate data
    if not args.nightmare_only:
        logger.info("Generating dream data...")
        dream_gen.generate_and_save(dataset_wrapper.train_data, args.output)
        logger.info("Dream data saved to %s/dream", args.output)

    if not args.dream_only:
        logger.info("Generating nightmare data...")
        nightmare_gen.generate_and_save(dataset_wrapper.train_data, args.output)
        logger.info("Nightmare data saved to %s/nightmare", args.output)

    logger.info("Data generation complete.")


if __name__ == "__main__":
    main()
