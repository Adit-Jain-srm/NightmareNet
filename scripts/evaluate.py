"""CLI entry point: evaluate a trained model checkpoint."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nightmarenet.data.loader import load_from_config
from nightmarenet.distortions.text import apply_text_distortions
from nightmarenet.evaluation.evaluator import Evaluator
from nightmarenet.training.trainer import _MODEL_TYPE_MAP, _tokenize_dataset
from nightmarenet.utils.config import load_config
from nightmarenet.utils.logging_config import setup_logging
from nightmarenet.utils.tracking import create_tracker_from_config

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="NightmareNet: Evaluate a trained model checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline model checkpoint (or HuggingFace model name) for comparison.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "markdown"],
        default="json",
        help="Output format for results (json or markdown).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
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
        # Validate checkpoint path
        if not os.path.exists(args.checkpoint):
            logger.error("Checkpoint path does not exist: %s", args.checkpoint)
            sys.exit(1)

        # Load config
        config = load_config(args.config)
        logger.info("Loaded config from %s", args.config)

        # Override tracker backend from CLI
        if args.tracker is not None:
            config.setdefault("tracking", {})["backend"] = args.tracker

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained model
        model_type = config.get("model", {}).get("type", "causal_lm")
        model_cls = _MODEL_TYPE_MAP.get(model_type, AutoModelForCausalLM)
        logger.info("Loading trained model from %s (type=%s)", args.checkpoint, model_type)
        trained_model = model_cls.from_pretrained(args.checkpoint).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        logger.info("Loading evaluation dataset...")
        dataset_wrapper = load_from_config(config)

        text_column = config.get("dataset", {}).get("text_column", "text")
        max_length = config.get("model", {}).get("max_length", 128)
        batch_size = config.get("training", {}).get("batch_size", 8)

        clean_dataloader = _tokenize_dataset(
            dataset_wrapper.test_data, tokenizer, text_column, max_length, batch_size
        )

        # Evaluate trained model
        tracker = create_tracker_from_config(config)
        evaluator = Evaluator(
            model=trained_model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            tracker=tracker,
        )

        trained_results = evaluator.evaluate(
            clean_dataloader=clean_dataloader,
            base_dataset=dataset_wrapper.test_data,
            distortion_fn=apply_text_distortions,
            label="dreamphase",
        )

        # If baseline provided, evaluate and compare
        if args.baseline:
            logger.info("Loading baseline model from %s", args.baseline)
            baseline_model = model_cls.from_pretrained(args.baseline).to(device)

            # Load baseline tokenizer to ensure correct encoding for the baseline model
            baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline)
            if baseline_tokenizer.pad_token is None:
                baseline_tokenizer.pad_token = baseline_tokenizer.eos_token

            baseline_dataloader = _tokenize_dataset(
                dataset_wrapper.test_data,
                baseline_tokenizer,
                text_column,
                max_length,
                batch_size,
            )

            baseline_evaluator = Evaluator(
                model=baseline_model,
                tokenizer=baseline_tokenizer,
                config=config,
                device=device,
            )

            baseline_results = baseline_evaluator.evaluate(
                clean_dataloader=baseline_dataloader,
                base_dataset=dataset_wrapper.test_data,
                distortion_fn=apply_text_distortions,
                label="baseline",
            )

            comparison = evaluator.compare(baseline_results, trained_results)

            if args.output_format == "markdown":
                report = evaluator.save_report(comparison)
                logger.info("\n%s", report)
            else:
                evaluator.save_results(comparison, "comparison_results.json")
        else:
            if args.output_format == "markdown":
                logger.info("Markdown report requires --baseline for comparison.")
            evaluator.save_results(trained_results, "evaluation_results.json")

        logger.info("Evaluation complete.")
        tracker.finish()

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
