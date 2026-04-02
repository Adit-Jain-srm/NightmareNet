"""GLUE benchmark evaluation for NightmareNet models.

Runs standard NLP classification benchmarks (SST-2, MRPC, QNLI, RTE, etc.)
using HuggingFace ``datasets`` and reports accuracy / F1 per task.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# GLUE tasks with their dataset config, input columns, and primary metric.
GLUE_TASKS: dict[str, dict[str, Any]] = {
    "sst2": {
        "dataset": "glue",
        "subset": "sst2",
        "input_columns": ["sentence"],
        "label_column": "label",
        "num_labels": 2,
        "metric": "accuracy",
    },
    "mrpc": {
        "dataset": "glue",
        "subset": "mrpc",
        "input_columns": ["sentence1", "sentence2"],
        "label_column": "label",
        "num_labels": 2,
        "metric": "f1",
    },
    "qnli": {
        "dataset": "glue",
        "subset": "qnli",
        "input_columns": ["question", "sentence"],
        "label_column": "label",
        "num_labels": 2,
        "metric": "accuracy",
    },
    "rte": {
        "dataset": "glue",
        "subset": "rte",
        "input_columns": ["sentence1", "sentence2"],
        "label_column": "label",
        "num_labels": 2,
        "metric": "accuracy",
    },
    "cola": {
        "dataset": "glue",
        "subset": "cola",
        "input_columns": ["sentence"],
        "label_column": "label",
        "num_labels": 2,
        "metric": "matthews_correlation",
    },
}


def _tokenize_glue_task(
    dataset: Any,
    tokenizer: Any,
    input_columns: list[str],
    max_length: int,
) -> Any:
    """Tokenize a GLUE dataset split.

    Handles single-sentence and sentence-pair tasks.

    Args:
        dataset: HuggingFace dataset split.
        tokenizer: Tokenizer to use.
        input_columns: Column name(s) for tokenizer input.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset with ``torch`` format set.
    """

    def tokenize_fn(examples: dict) -> dict:
        if len(input_columns) == 1:
            return tokenizer(
                examples[input_columns[0]],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        return tokenizer(
            examples[input_columns[0]],
            examples[input_columns[1]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    remove_cols = [c for c in dataset.column_names if c not in ("label", "labels")]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    # Rename 'label' to 'labels' if needed (HF convention)
    if "label" in tokenized.column_names and "labels" not in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


def evaluate_glue_task(
    model: torch.nn.Module,
    tokenizer: Any,
    task_name: str,
    device: str | torch.device = "cpu",
    max_length: int = 128,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate a model on a single GLUE task.

    Args:
        model: Sequence-classification model.
        tokenizer: Tokenizer for the model.
        task_name: Key from :data:`GLUE_TASKS` (e.g. ``"sst2"``).
        device: Torch device.
        max_length: Max sequence length for tokenization.
        batch_size: Evaluation batch size.
        max_samples: Optional cap on validation samples (for quick runs).

    Returns:
        Dict with task name, primary metric, accuracy, and F1.
    """
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    task_cfg = GLUE_TASKS.get(task_name)
    if task_cfg is None:
        return {"task": task_name, "error": f"Unknown GLUE task '{task_name}'"}

    logger.info("GLUE: loading %s (subset=%s)", task_cfg["dataset"], task_cfg["subset"])
    try:
        raw = load_dataset(task_cfg["dataset"], task_cfg["subset"])
    except Exception as exc:
        logger.error("Failed to load GLUE dataset %s: %s", task_name, exc)
        return {"task": task_name, "error": str(exc)}

    # Use validation split (test labels are hidden on GLUE)
    split_name = "validation" if "validation" in raw else "test"
    val_ds = raw[split_name]

    if max_samples is not None and len(val_ds) > max_samples:
        val_ds = val_ds.select(range(max_samples))

    tokenized = _tokenize_glue_task(
        val_ds,
        tokenizer,
        task_cfg["input_columns"],
        max_length,
    )
    dataloader = DataLoader(tokenized, batch_size=batch_size)

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"GLUE/{task_name}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                labels = batch["labels"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)
    except Exception as exc:
        logger.error("GLUE evaluation failed for %s: %s", task_name, exc)
        return {"task": task_name, "error": str(exc)}

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    result: dict[str, Any] = {
        "task": task_name,
        "accuracy": float(accuracy),
        "f1_weighted": float(f1),
        "num_samples": len(all_labels),
    }

    if task_cfg["metric"] == "matthews_correlation":
        result["matthews_correlation"] = float(matthews_corrcoef(all_labels, all_preds))

    primary = task_cfg["metric"]
    result["primary_metric"] = primary
    result["primary_score"] = result.get(primary, result["accuracy"])

    return result


def evaluate_glue(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: Optional[list[str]] = None,
    device: str | torch.device = "cpu",
    max_length: int = 128,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate a model on multiple GLUE tasks.

    Args:
        model: Sequence-classification model.
        tokenizer: Tokenizer for the model.
        tasks: List of GLUE task names. Defaults to all supported tasks.
        device: Torch device.
        max_length: Max sequence length.
        batch_size: Evaluation batch size.
        max_samples: Per-task sample limit.

    Returns:
        Dict mapping task names to their result dicts, plus an
        ``"average"`` key with the mean primary score.
    """
    if tasks is None:
        tasks = list(GLUE_TASKS.keys())

    results: dict[str, Any] = {}
    scores: list[float] = []

    for task in tasks:
        task_result = evaluate_glue_task(
            model=model,
            tokenizer=tokenizer,
            task_name=task,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            max_samples=max_samples,
        )
        results[task] = task_result
        if "error" not in task_result:
            scores.append(task_result["primary_score"])

    results["average"] = {
        "mean_primary_score": float(np.mean(scores)) if scores else 0.0,
        "tasks_evaluated": len(scores),
        "tasks_failed": len(tasks) - len(scores),
    }

    return results
