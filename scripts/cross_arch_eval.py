"""Cross-architecture robustness evaluation.

This script trains a small model (e.g., DistilBERT) with NightmareNet and evaluates it,
then zero-shot evaluates a larger model (e.g., BERT-large) on the same distortions
to measure transferability of robustness improvements.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nightmarenet.data.generator import create_generators_from_config
from nightmarenet.data.loader import load_from_config
from nightmarenet.distortions.text import apply_text_distortions
from nightmarenet.evaluation.evaluator import Evaluator
from nightmarenet.training.trainer import Trainer, _tokenize_dataset
from nightmarenet.utils.config import load_config
from nightmarenet.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="NightmareNet: Cross-architecture robustness evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/examples/cross-arch-eval.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--finetune-large",
        action="store_true",
        help="Whether to optionally fine-tune the large model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and data but skip training. Prints config summary and exits.",
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    try:
        config = load_config(args.config)
        logger.info("Loaded config from %s", args.config)

        # Basic setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = config.get("seed", 42)
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_model_name = config.get("model", {}).get("train", "distilbert-base-uncased")
        eval_model_name = config.get("model", {}).get("eval", "bert-large-uncased")

        if args.dry_run:
            logger.info("Dry-run mode: skipping execution.")
            logger.info(
                "Config summary: train_model=%s, eval_model=%s", train_model_name, eval_model_name
            )
            return

        # 1. Load dataset
        logger.info("Loading dataset...")
        dataset_wrapper = load_from_config(config)

        text_column = config.get("dataset", {}).get("text_column", "text")
        label_column = config.get("dataset", {}).get("label_column", "label")
        max_length = config.get("model", {}).get("max_length", 128)
        batch_size = config.get("training", {}).get("batch_size", 8)
        num_labels = config.get("model", {}).get("num_labels", 2)

        # 2. Train DistilBERT + NightmareNet
        logger.info("=== Phase 1: Train %s with NightmareNet ===", train_model_name)
        distilbert_config = copy.deepcopy(config)
        distilbert_config["model"]["name"] = train_model_name

        dream_gen, nightmare_gen = create_generators_from_config(distilbert_config)

        logger.info("Generating dream and nightmare data for %s...", train_model_name)
        dream_data = dream_gen.generate(dataset_wrapper.train_data)
        nightmare_data = nightmare_gen.generate(dataset_wrapper.train_data)

        trainer_small = Trainer(config=distilbert_config)

        train_dataloader_small = _tokenize_dataset(
            dataset_wrapper.train_data,
            trainer_small.tokenizer,
            text_column,
            max_length,
            batch_size,
            label_column,
        )
        dream_dataloader_small = _tokenize_dataset(
            dream_data, trainer_small.tokenizer, text_column, max_length, batch_size, label_column
        )
        nightmare_dataloader_small = _tokenize_dataset(
            nightmare_data,
            trainer_small.tokenizer,
            text_column,
            max_length,
            batch_size,
            label_column,
        )

        logger.info("Starting training pipeline for %s...", train_model_name)
        trainer_small.train(
            train_dataloader=train_dataloader_small,
            dream_dataloader=dream_dataloader_small,
            nightmare_dataloader=nightmare_dataloader_small,
            dream_generator=dream_gen,
            nightmare_generator=nightmare_gen,
            dream_base_dataset=dataset_wrapper.train_data,
            nightmare_base_dataset=dataset_wrapper.train_data,
        )

        # Evaluate DistilBERT
        clean_dataloader_small = _tokenize_dataset(
            dataset_wrapper.test_data,
            trainer_small.tokenizer,
            text_column,
            max_length,
            batch_size,
            label_column,
        )
        evaluator_small = Evaluator(
            model=trainer_small.model,
            tokenizer=trainer_small.tokenizer,
            config=distilbert_config,
            device=device,
        )
        logger.info("Evaluating %s...", train_model_name)

        start_time = time.time()
        results_small = evaluator_small.evaluate(
            clean_dataloader=clean_dataloader_small,
            base_dataset=dataset_wrapper.test_data,
            distortion_fn=apply_text_distortions,
            label="small_robust",
        )
        inference_time_small = (time.time() - start_time) / len(dataset_wrapper.test_data)

        # 3. Load BERT-large
        logger.info("=== Phase 2: Zero-Shot Evaluate %s ===", eval_model_name)
        large_config = copy.deepcopy(config)
        large_config["model"]["name"] = eval_model_name

        tokenizer_large = AutoTokenizer.from_pretrained(eval_model_name)
        if tokenizer_large.pad_token is None:
            tokenizer_large.pad_token = tokenizer_large.eos_token

        model_large = AutoModelForSequenceClassification.from_pretrained(
            eval_model_name, num_labels=num_labels
        ).to(device)

        clean_dataloader_large = _tokenize_dataset(
            dataset_wrapper.test_data,
            tokenizer_large,
            text_column,
            max_length,
            batch_size,
            label_column,
        )

        evaluator_large = Evaluator(
            model=model_large,
            tokenizer=tokenizer_large,
            config=large_config,
            device=device,
        )
        logger.info("Evaluating %s (Zero-Shot)...", eval_model_name)

        start_time = time.time()
        results_large_zeroshot = evaluator_large.evaluate(
            clean_dataloader=clean_dataloader_large,
            base_dataset=dataset_wrapper.test_data,
            distortion_fn=apply_text_distortions,
            label="large_zeroshot",
        )
        inference_time_large_zeroshot = (time.time() - start_time) / len(dataset_wrapper.test_data)

        # (Optional) Fine-tune BERT-large
        results_large_finetuned = None
        inference_time_large_finetuned = None

        if args.finetune_large:
            logger.info("=== Phase 3: Fine-Tune %s with NightmareNet ===", eval_model_name)
            dream_gen_large, nightmare_gen_large = create_generators_from_config(large_config)
            dream_data_large = dream_gen_large.generate(dataset_wrapper.train_data)
            nightmare_data_large = nightmare_gen_large.generate(dataset_wrapper.train_data)

            trainer_large = Trainer(config=large_config)

            train_dataloader_large = _tokenize_dataset(
                dataset_wrapper.train_data,
                trainer_large.tokenizer,
                text_column,
                max_length,
                batch_size,
                label_column,
            )
            dream_dataloader_large = _tokenize_dataset(
                dream_data_large,
                trainer_large.tokenizer,
                text_column,
                max_length,
                batch_size,
                label_column,
            )
            nightmare_dataloader_large = _tokenize_dataset(
                nightmare_data_large,
                trainer_large.tokenizer,
                text_column,
                max_length,
                batch_size,
                label_column,
            )

            trainer_large.train(
                train_dataloader=train_dataloader_large,
                dream_dataloader=dream_dataloader_large,
                nightmare_dataloader=nightmare_dataloader_large,
                dream_generator=dream_gen_large,
                nightmare_generator=nightmare_gen_large,
                dream_base_dataset=dataset_wrapper.train_data,
                nightmare_base_dataset=dataset_wrapper.train_data,
            )

            evaluator_large_ft = Evaluator(
                model=trainer_large.model,
                tokenizer=trainer_large.tokenizer,
                config=large_config,
                device=device,
            )

            start_time = time.time()
            results_large_finetuned = evaluator_large_ft.evaluate(
                clean_dataloader=clean_dataloader_large,
                base_dataset=dataset_wrapper.test_data,
                distortion_fn=apply_text_distortions,
                label="large_finetuned",
            )
            inference_time_large_finetuned = (time.time() - start_time) / len(
                dataset_wrapper.test_data
            )

        # 4. Generate Comparison and Save
        logger.info("=== Aggregating Results ===")

        def extract_metrics(res, inf_time):
            # Based on standard NightmareNet evaluator outputs
            clean_acc = res.get("clean_performance", {}).get("accuracy", 0.0)
            if not clean_acc and "clean_performance" in res:
                # Fallback if metric is different
                clean_acc = res["clean_performance"].get("score", 0.0)

            # Aggregate robustness
            robustness_scores = []
            for d in res.get("distorted_performance", []):
                acc = d.get("performance", {}).get("accuracy", 0.0)
                if not acc:
                    acc = d.get("performance", {}).get("score", 0.0)
                robustness_scores.append(acc)

            avg_robustness = sum(robustness_scores) / max(len(robustness_scores), 1)

            return {
                "robustness_score": avg_robustness,
                "clean_accuracy": clean_acc,
                "inference_time": inf_time,
            }

        summary = {
            "DistilBERT + NightmareNet": extract_metrics(results_small, inference_time_small),
            "BERT-large (zero-shot)": extract_metrics(
                results_large_zeroshot, inference_time_large_zeroshot
            ),
        }

        if results_large_finetuned:
            summary["BERT-large + NightmareNet"] = extract_metrics(
                results_large_finetuned, inference_time_large_finetuned
            )

        with open("cross_arch_eval_results.json", "w") as f:
            json.dump(summary, f, indent=4)

        logger.info("Results saved to cross_arch_eval_results.json")
        for model_name, metrics in summary.items():
            logger.info(f"{model_name}: {metrics}")

    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
