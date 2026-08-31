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
from nightmarenet.evaluation.metrics import classification_metrics
from nightmarenet.training.trainer import Trainer, _tokenize_dataset
from nightmarenet.utils.config import load_config
from nightmarenet.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _release_gpu(*objects) -> None:
    """Drop references and return cached CUDA memory between model phases."""
    for obj in objects:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _time_clean_inference(model, dataloader, device, sample_count: int) -> float:
    """Seconds per sample for a single clean forward pass (no distortion sweep)."""
    model.eval()
    start = time.time()
    classification_metrics(model, dataloader, device)
    return (time.time() - start) / max(sample_count, 1)


def _evaluate_with_sweep_timing(
    evaluator,
    clean_dataloader,
    base_dataset,
    distortion_fn,
    label: str,
    sample_count: int,
):
    """Run the full evaluator sweep and return (results, eval_sweep_time)."""
    start = time.time()
    results = evaluator.evaluate(
        clean_dataloader=clean_dataloader,
        base_dataset=base_dataset,
        distortion_fn=distortion_fn,
        label=label,
    )
    eval_sweep_time = (time.time() - start) / max(sample_count, 1)
    return results, eval_sweep_time


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
        "--output",
        type=str,
        default="cross_arch_eval_results.json",
        help="Path to write the results JSON.",
    )
    parser.add_argument(
        "--large-checkpoint",
        type=str,
        default=None,
        help=(
            "Pre-finetuned checkpoint for the large model to avoid training an uninitialized head."
        ),
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

        n_test = len(dataset_wrapper.test_data)
        inference_time_small = _time_clean_inference(
            trainer_small.model, clean_dataloader_small, device, n_test
        )
        results_small, eval_sweep_time_small = _evaluate_with_sweep_timing(
            evaluator_small,
            clean_dataloader_small,
            dataset_wrapper.test_data,
            apply_text_distortions,
            "small_robust",
            n_test,
        )

        # Free DistilBERT before loading BERT-large (4GB GPU target).
        _release_gpu(trainer_small.model, trainer_small, evaluator_small)

        # 3. Load BERT-large
        eval_model_name_to_load = (
            args.large_checkpoint if args.large_checkpoint else eval_model_name
        )
        logger.info("=== Phase 2: Zero-Shot Evaluate %s ===", eval_model_name_to_load)
        large_config = copy.deepcopy(config)
        large_config["model"]["name"] = eval_model_name_to_load

        tokenizer_large = AutoTokenizer.from_pretrained(eval_model_name_to_load)
        if tokenizer_large.pad_token is None:
            tokenizer_large.pad_token = tokenizer_large.eos_token

        model_large = AutoModelForSequenceClassification.from_pretrained(
            eval_model_name_to_load, num_labels=num_labels
        ).to(device)

        if not args.large_checkpoint:
            logger.info(
                "No fine-tuned checkpoint provided. Running clean fine-tuning on %s...",
                eval_model_name,
            )
            train_dataloader_large = _tokenize_dataset(
                dataset_wrapper.train_data,
                tokenizer_large,
                text_column,
                max_length,
                batch_size,
                label_column,
            )
            optimizer = torch.optim.AdamW(model_large.parameters(), lr=2e-5)
            model_large.train()
            for batch in train_dataloader_large:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model_large(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            logger.info("Clean fine-tuning complete.")

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

        inference_time_large_zeroshot = _time_clean_inference(
            model_large, clean_dataloader_large, device, n_test
        )
        results_large_zeroshot, eval_sweep_time_large_zeroshot = _evaluate_with_sweep_timing(
            evaluator_large,
            clean_dataloader_large,
            dataset_wrapper.test_data,
            apply_text_distortions,
            "large_zeroshot",
            n_test,
        )

        # (Optional) Fine-tune BERT-large
        results_large_finetuned = None
        inference_time_large_finetuned = None
        eval_sweep_time_large_finetuned = None

        if args.finetune_large:
            _release_gpu(model_large, evaluator_large)

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

            inference_time_large_finetuned = _time_clean_inference(
                trainer_large.model, clean_dataloader_large, device, n_test
            )
            results_large_finetuned, eval_sweep_time_large_finetuned = _evaluate_with_sweep_timing(
                evaluator_large_ft,
                clean_dataloader_large,
                dataset_wrapper.test_data,
                apply_text_distortions,
                "large_finetuned",
                n_test,
            )

        # 4. Generate Comparison and Save
        logger.info("=== Aggregating Results ===")

        def extract_metrics(res, inference_time_per_sample, eval_sweep_time):
            # Based on standard NightmareNet evaluator outputs
            clean_perf = res.get("clean_performance", {})
            clean_acc = clean_perf.get("accuracy", clean_perf.get("score", 0.0))

            # Aggregate robustness
            robustness_scores = []
            for d in res.get("distorted_performance", []):
                perf = d.get("performance", {})
                acc = perf.get("accuracy", perf.get("score", 0.0))
                robustness_scores.append(acc)

            avg_robustness = sum(robustness_scores) / max(len(robustness_scores), 1)

            return {
                "robustness_score": avg_robustness,
                "clean_accuracy": clean_acc,
                "inference_time_per_sample": inference_time_per_sample,
                "eval_sweep_time": eval_sweep_time,
            }

        summary = {
            "DistilBERT + NightmareNet": extract_metrics(
                results_small, inference_time_small, eval_sweep_time_small
            ),
            "BERT-large (zero-shot)": extract_metrics(
                results_large_zeroshot,
                inference_time_large_zeroshot,
                eval_sweep_time_large_zeroshot,
            ),
        }

        if results_large_finetuned:
            summary["BERT-large + NightmareNet"] = extract_metrics(
                results_large_finetuned,
                inference_time_large_finetuned,
                eval_sweep_time_large_finetuned,
            )

        with open(args.output, "w") as f:
            json.dump(summary, f, indent=4)

        logger.info("Results saved to %s", args.output)
        for model_name, metrics in summary.items():
            logger.info("%s: %s", model_name, metrics)

    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
