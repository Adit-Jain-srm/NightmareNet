"""Orchestrator for multi-model ensemble robustness benchmarking."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nightmarenet.distortions.registry import get_registry
from nightmarenet.evaluation.metrics import classification_metrics

logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    name: str = "sst2"
    split: str = "validation"
    max_samples: Optional[int] = 100
    text_column: str = "sentence"


class DistortionConfig(BaseModel):
    type: str = "dream"
    strengths: list[float] = Field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])


class EnsembleConfig(BaseModel):
    models: list[str]
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    distortions: list[DistortionConfig] = Field(default_factory=list)


def _evaluate_model_worker(
    model_name: str,
    dataset_name: str,
    dataset_split: str,
    max_samples: int,
    text_column: str,
    distortions_data: list[dict[str, Any]],
    cache_dir: Optional[Path] = None,
    no_cache: bool = False,
) -> dict[str, Any]:
    """Worker function to evaluate a single model.

    Runs in a separate process to ensure memory is freed after execution.
    """

    def _get_cache_key(
        model: str,
        dataset: str,
        split: str,
        distortion_type: str,
        strength: float,
    ) -> str:
        """Generate cache key for a specific evaluation tuple."""
        safe_model = model.replace("/", "_").replace("-", "_")
        return f"{safe_model}_{dataset}_{split}_{distortion_type}_{strength:g}.json"

    from datasets import load_dataset
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())

    logger.info("Loading dataset %s", dataset_name)
    ds = load_dataset(dataset_name, split=dataset_split)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    registry = get_registry()

    start_time = time.time()

    total_auc = 0.0
    results_by_distortion = {}

    try:
        for dist_dict in distortions_data:
            distortion_type = dist_dict["type"]
            strengths = dist_dict["strengths"]

            accuracies = []
            for strength in strengths:
                cache_key = _get_cache_key(
                    model_name, dataset_name, dataset_split, distortion_type, strength
                )
                cache_file = cache_dir / cache_key if cache_dir else None

                if cache_file and cache_file.exists() and not no_cache:
                    try:
                        with open(cache_file, encoding="utf-8") as f:
                            cached_result = json.load(f)
                            accuracies.append(cached_result["accuracy"])
                            logger.info("Cache hit for %s at strength %.1f", model_name, strength)
                            continue
                    except (OSError, json.JSONDecodeError, KeyError) as e:
                        logger.warning("Corrupted cache file %s: %s, re-evaluating", cache_file, e)

                def distortion_fn(text, _s=strength, _dt=distortion_type):
                    return registry.apply(_dt, text, strength=_s, seed=42)

                distorted = ds.map(
                    lambda x: {text_column: distortion_fn(x[text_column])},
                    desc=f"Distorting {distortion_type} at strength {strength:.1f}",
                )

                def tokenize_fn(examples):
                    return tokenizer(
                        examples[text_column],
                        truncation=True,
                        padding="max_length",
                        max_length=128,
                        return_tensors="pt",
                    )

                tokenized = distorted.map(
                    tokenize_fn,
                    batched=True,
                    remove_columns=distorted.column_names,
                )

                # Ensure labels column exists for classification_metrics
                if "label" in tokenized.column_names and "labels" not in tokenized.column_names:
                    tokenized = tokenized.rename_column("label", "labels")

                tokenized.set_format("torch")
                dataloader = DataLoader(tokenized, batch_size=8)

                metrics = classification_metrics(model, dataloader, device=device)
                accuracy = metrics.get("accuracy", 0.0)
                accuracies.append(accuracy)

                if cache_file and not no_cache:
                    try:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump({"accuracy": accuracy}, f)
                    except OSError as e:
                        logger.warning("Failed to write cache file %s: %s", cache_file, e)

            from sklearn.metrics import auc as sklearn_auc

            auc = float(sklearn_auc(strengths, accuracies))
            total_auc += auc

            results_by_distortion[distortion_type] = {
                "strengths": strengths,
                "accuracies": accuracies,
                "auc": auc,
            }

    except Exception as e:
        logger.error("Evaluation failed for %s: %s", model_name, e)
        raise e

    latency = time.time() - start_time
    avg_auc = total_auc / max(1, len(distortions_data))

    return {
        "model": model_name,
        "robustness": avg_auc,
        "latency": latency,
        "params": params,
        "results_by_distortion": results_by_distortion,
    }


class EnsembleOrchestrator:
    """Orchestrates the evaluation of multiple models from a config."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # Validate with Pydantic
        self.config = EnsembleConfig(**raw_config)

    def run(
        self,
        timeout_seconds: int = 300,
        output_dir: str = "./results",
        no_cache: bool = False,
    ) -> dict[str, Any]:
        """Run the ensemble benchmark suite.

        Args:
            timeout_seconds: Maximum time (in seconds) to allow per model.
            output_dir: Output directory for results and cache.
            no_cache: Force re-evaluation without using cache.
        """
        models = self.config.models
        dataset_cfg = self.config.dataset
        ds_name = dataset_cfg.name
        ds_split = dataset_cfg.split
        max_samples = dataset_cfg.max_samples
        text_column = dataset_cfg.text_column

        distortions = self.config.distortions

        if not distortions:
            distortions_data = [{"type": "dream", "strengths": [0.1, 0.3, 0.5, 0.7, 0.9]}]
        else:
            distortions_data = [{"type": d.type, "strengths": d.strengths} for d in distortions]

        results = {}
        models_summary = []

        cache_dir = Path(output_dir) / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for model_name in models:
            logger.info("Starting evaluation for %s", model_name)

            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _evaluate_model_worker,
                    model_name,
                    ds_name,
                    ds_split,
                    max_samples or 100,
                    text_column,
                    distortions_data,
                    cache_dir,
                    no_cache,
                )
                try:
                    res = future.result(timeout=timeout_seconds)
                    summary = {
                        "model": res["model"],
                        "robustness": res["robustness"],
                        "latency": res["latency"],
                        "params": res["params"],
                    }
                    models_summary.append(summary)

                    model_results = res["results_by_distortion"]
                    results[model_name] = model_results

                except TimeoutError:
                    logger.error("Timeout exceeded for model %s", model_name)
                except Exception as e:
                    logger.error("Error evaluating model %s: %s", model_name, e)

        return {
            "models_summary": models_summary,
            "raw_results": results,
        }
