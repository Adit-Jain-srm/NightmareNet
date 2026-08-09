"""Demo: end-to-end scrape → train → evaluate pipeline.

Scrapes Wikipedia articles, trains GPT-2 through one NightmareNet
sleep cycle, and produces a before/after evaluation report.

Usage:
    python scripts/demo_pipeline.py [--config configs/demo_scrape.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys

from nightmarenet.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="NightmareNet Demo: scrape → train → evaluate in one command.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/demo_scrape.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/demo_model",
        help="Directory to save the hardened model.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    args = parser.parse_args()
    setup_logging(log_level=args.log_level)

    try:
        from nightmarenet.pipeline import Pipeline
        from nightmarenet.utils.config import load_config

        # Load config
        config = load_config(args.config)
        urls = config.pop(
            "scrape_urls",
            [
                "https://en.wikipedia.org/wiki/Machine_learning",
                "https://en.wikipedia.org/wiki/Neural_network_(machine_learning)",
            ],
        )

        logger.info("=" * 60)
        logger.info("NightmareNet Demo Pipeline")
        logger.info("=" * 60)
        logger.info("Model: %s (%s)", config["model"]["name"], config["model"]["type"])
        logger.info("URLs:  %d pages to scrape", len(urls))
        logger.info("Cycles: %d", config["training"]["num_cycles"])
        logger.info("=" * 60)

        # Set seed
        import random

        import numpy as np
        import torch

        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create and run pipeline
        pipeline = Pipeline(config=config)

        logger.info("\n📡 Step 1/5: Ingesting data from %d URLs...", len(urls))
        pipeline.ingest(urls=urls)

        logger.info("\n🔧 Step 2/5: Generating dream/nightmare data + tokenizing...")
        pipeline.prepare()

        logger.info("\n🧠 Step 3/5: Running sleep-cycle training...")
        history = pipeline.train()

        logger.info("\n📊 Step 4/5: Evaluating baseline vs. trained model...")
        comparison = pipeline.evaluate()

        logger.info("\n💾 Step 5/5: Exporting model to %s...", args.output)
        pipeline.export(args.output)

        # Print report
        report = pipeline.metrics.report_md
        if report:
            logger.info("\n" + "=" * 60)
            logger.info("EVALUATION REPORT")
            logger.info("=" * 60)
            print(report)

        logger.info("\n✅ Demo complete! %d phases trained.", len(history))
        logger.info("Model saved to: %s", args.output)

        return comparison

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Demo interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Demo failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
