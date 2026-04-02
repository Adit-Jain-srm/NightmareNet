"""CLI entry point: generate dream and nightmare datasets."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from nightmarenet.data.generator import create_generators_from_config
from nightmarenet.data.loader import load_from_config
from nightmarenet.utils.config import load_config
from nightmarenet.utils.logging_config import setup_logging

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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dream-only",
        action="store_true",
        help="Generate only dream data.",
    )
    mode_group.add_argument(
        "--nightmare-only",
        action="store_true",
        help="Generate only nightmare data.",
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    try:
        # Load config
        config = load_config(args.config)
        logger.info("Loaded config from %s", args.config)

        # Set seed
        seed = config.get("seed", 42)
        import random

        random.seed(seed)

        # Ensure output directory exists
        os.makedirs(args.output, exist_ok=True)

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

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Data generation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error during data generation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
