"""CLI entry point: evaluate a trained model checkpoint."""

import argparse
import logging
import sys

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from nightmarenet.data.loader import load_from_config
from nightmarenet.distortions.text import apply_text_distortions
from nightmarenet.evaluation.evaluator import Evaluator
from nightmarenet.training.trainer import _tokenize_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
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
        help="Output format for results.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", args.config)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    logger.info("Loading trained model from %s", args.checkpoint)
    trained_model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
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
    evaluator = Evaluator(
        model=trained_model,
        tokenizer=tokenizer,
        config=config,
        device=device,
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
        baseline_model = AutoModelForCausalLM.from_pretrained(args.baseline).to(device)

        baseline_evaluator = Evaluator(
            model=baseline_model,
            tokenizer=tokenizer,
            config=config,
            device=device,
        )

        baseline_results = baseline_evaluator.evaluate(
            clean_dataloader=clean_dataloader,
            base_dataset=dataset_wrapper.test_data,
            distortion_fn=apply_text_distortions,
            label="baseline",
        )

        comparison = evaluator.compare(baseline_results, trained_results)

        evaluator.save_results(comparison, "comparison_results.json")
        report = evaluator.save_report(comparison)
        logger.info("\n%s", report)
    else:
        evaluator.save_results(trained_results, "evaluation_results.json")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
