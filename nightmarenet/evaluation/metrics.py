"""Evaluation metrics for measuring model robustness, generalization, and quality.

Implements metrics for:
- Recall / F1 on clean test data
- Generalization score on out-of-distribution data
- Robustness score under increasing distortion
- Hallucination rate
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import torch
from datasets import IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _safe_float(value: float, default: float = 0.0) -> float:
    """Return default if value is NaN or Inf."""
    if math.isnan(value) or math.isinf(value):
        logger.warning("Detected NaN/Inf metric value, using default %.4f", default)
        return default
    return float(value)


def categorize_failures_by_distortion(
    failure_records: Optional[list[dict[str, Any]]] = None,
    total_samples_per_distortion: Optional[dict[str, int]] = None,
) -> dict[str, dict[str, Any]]:
    """Group failed evaluation samples by distortion type and compute category statistics.

    Args:
        failure_records: List of per-sample evaluation records.
            Each record dictionary may contain:
            - "distortion_type" (or "distortion", "distortion_name", "type")
            - "is_failure" (or "failed", "correct")
            - "confidence_delta" (or "confidence_drop", "delta")
            - "total_samples" (optional per-record sample count)
        total_samples_per_distortion: Optional dict mapping distortion type names
            to total sample counts evaluated for that distortion type.

    Returns:
        Deterministic dict mapping distortion types to:
        {
            "count": int,
            "failure_rate": float,
            "avg_confidence_delta": float
        }
        Sorted by highest failure rate, then failure count, then name.
    """
    if not failure_records:
        return {}

    stats: dict[str, dict[str, Any]] = {}

    for rec in failure_records:
        if not isinstance(rec, dict):
            continue

        dtype = str(
            rec.get("distortion_type")
            or rec.get("distortion")
            or rec.get("distortion_name")
            or rec.get("type")
            or "unknown"
        )

        is_fail = rec.get("is_failure")
        if is_fail is None:
            is_fail = rec.get("failed")
        if is_fail is None:
            is_fail = not rec.get("correct") if "correct" in rec else True
        is_fail = bool(is_fail)

        raw_delta = rec.get("confidence_delta")
        if raw_delta is None:
            raw_delta = rec.get("confidence_drop", rec.get("delta", 0.0))
        try:
            conf_delta = _safe_float(float(raw_delta) if raw_delta is not None else 0.0)
        except (ValueError, TypeError):
            conf_delta = 0.0

        if dtype not in stats:
            stats[dtype] = {
                "count": 0,
                "total": 0,
                "confidence_deltas": [],
                "sample_total": rec.get("total_samples"),
            }

        stats[dtype]["total"] += 1
        if is_fail:
            stats[dtype]["count"] += 1
            stats[dtype]["confidence_deltas"].append(conf_delta)

    categories: dict[str, dict[str, Any]] = {}

    for dtype, s in stats.items():
        count = s["count"]
        if total_samples_per_distortion and dtype in total_samples_per_distortion:
            total = total_samples_per_distortion[dtype]
        elif s["sample_total"] is not None:
            total = int(s["sample_total"])
        else:
            total = s["total"]

        total = max(total, count)
        failure_rate = _safe_float(count / total) if total > 0 else 0.0
        avg_conf_delta = (
            _safe_float(sum(s["confidence_deltas"]) / count)
            if count > 0 and s["confidence_deltas"]
            else 0.0
        )

        categories[dtype] = {
            "count": count,
            "failure_rate": failure_rate,
            "avg_confidence_delta": avg_conf_delta,
        }

    sorted_keys = sorted(
        categories.keys(),
        key=lambda k: (-categories[k]["failure_rate"], -categories[k]["count"], k),
    )

    return {k: categories[k] for k in sorted_keys}


def compute_confidence_delta(clean_conf: float, dist_conf: float) -> float:
    """Compute confidence delta (degradation) from clean to distorted."""
    return clean_conf - dist_conf


def rank_failures(failures: list[dict]) -> list[dict]:
    """Sort failed samples by confidence_delta descending."""
    return sorted(failures, key=lambda x: x.get("confidence_delta", 0.0), reverse=True)


def build_delta_distribution(failures: list[dict]) -> dict:
    """Bucket every failure into severity buckets."""
    dist = {"0_10": 0, "10_25": 0, "25_50": 0, "50_plus": 0}
    for f in failures:
        delta = f.get("confidence_delta", 0.0)
        if delta >= 0.5:
            dist["50_plus"] += 1
        elif delta >= 0.25:
            dist["25_50"] += 1
        elif delta >= 0.10:
            dist["10_25"] += 1
        elif delta > 0.0:
            dist["0_10"] += 1
    return dist


def truncate_preview(text: str, max_len: int = 50) -> str:
    """Truncate text for preview."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def compute_perplexity(
    model, dataloader: DataLoader, device="cpu", return_per_sample: bool = False
):
    """Compute perplexity of a language model on a dataset.

    Args:
        model: Language model with a forward method returning loss.
        dataloader: DataLoader providing tokenized batches.
        device: Device to run inference on.

    Returns:
        Perplexity score (lower is better for clean data).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    per_sample_ppls = []

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                batch = {k: v.to(device) for k, v in batch.items()}

                if "attention_mask" in batch:
                    mask = batch["attention_mask"]
                else:
                    mask = torch.ones_like(batch["input_ids"])

                labels = batch["input_ids"].clone()
                labels[mask == 0] = -100

                outputs = model(**batch, labels=labels)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                loss = loss.view(shift_labels.size(0), -1)

                shift_mask = mask[..., 1:].contiguous()
                valid_tokens_per_sample = shift_mask.sum(dim=1)
                per_sample_loss = (loss * shift_mask).sum(dim=1) / torch.clamp(
                    valid_tokens_per_sample, min=1
                )

                if return_per_sample:
                    batch_ppls = (
                        torch.exp(torch.clamp(per_sample_loss, max=100)).cpu().numpy().tolist()
                    )
                    per_sample_ppls.extend(batch_ppls)

                total_loss += (per_sample_loss * valid_tokens_per_sample).sum().item()
                total_tokens += valid_tokens_per_sample.sum().item()
    except Exception as e:
        logger.warning("Error during perplexity computation: %s", e)
        return float("inf")

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 100))  # Cap to avoid overflow
    result = float(perplexity)
    if math.isnan(result) or math.isinf(result):
        logger.warning("Perplexity is NaN/Inf, returning inf")
        result = float("inf")

    if return_per_sample:
        return {"perplexity": result, "per_sample_ppls": per_sample_ppls}
    return result


def quick_robustness_score(
    model,
    base_dataset,
    tokenizer,
    distortion_fn,
    *,
    strength: float = 0.5,
    subset_size: int = 50,
    text_column: str = "text",
    max_length: int = 128,
    batch_size: int = 8,
    device="cpu",
) -> float:
    """Compute a lightweight robustness score on a fixed dataset subset.

    Intended for inexpensive per-cycle convergence checks. Evaluates a
    single distortion strength on a deterministic subset and returns a
    scalar robustness score (higher is better).
    """
    if len(base_dataset) == 0:
        return 0.0

    is_vision = tokenizer is None or not hasattr(base_dataset, "map")
    if is_vision:
        try:
            indices = list(range(min(subset_size, len(base_dataset))))
            subset = torch.utils.data.Subset(base_dataset, indices)

            class DummyGenerator:
                def __init__(self, strength, seed, config):
                    self.strength = strength
                    self.seed = seed
                    self.config = config
                    self.target_model = model

            from nightmarenet.data.generator import DistortedVisionDataset

            dummy_gen = DummyGenerator(strength, seed=42, config={})
            distorted_ds = DistortedVisionDataset(subset, dummy_gen, phase="dream")
            dataloader = DataLoader(distorted_ds, batch_size=batch_size, shuffle=False)

            metrics = classification_metrics(model, dataloader, device)
            return metrics.get("accuracy", 0.0)
        except Exception as e:
            logger.warning("Error during quick robustness computation: %s", e)
            return 0.0

    try:
        subset = base_dataset.shuffle(seed=42).select(range(min(subset_size, len(base_dataset))))
        distorted = subset.map(
            lambda example: {
                **example,
                text_column: distortion_fn(
                    example[text_column],
                    strength=strength,
                ),
            },
            desc="Quick robustness probe",
        )

        def tokenize_fn(examples):
            return tokenizer(
                examples[text_column],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        if isinstance(distorted, IterableDataset):
            tokenized = distorted.map(
                tokenize_fn,
                batched=True,
                remove_columns=(
                    distorted.column_names if distorted.column_names else [text_column]
                ),
            )
            tokenized = tokenized.with_format("torch")
            dataloader = DataLoader(tokenized, batch_size=batch_size)
        else:
            tokenized = distorted.map(
                tokenize_fn,
                batched=True,
                remove_columns=distorted.column_names,
                desc="Tokenizing",
            )
            tokenized.set_format("torch")
            dataloader = DataLoader(
                tokenized,
                batch_size=batch_size,
                shuffle=True,
            )
        perplexity = compute_perplexity(
            model=model,
            dataloader=dataloader,
            device=device,
        )
        return _safe_float(1.0 / max(perplexity, 1e-8))
    except Exception as e:
        logger.warning("Error during quick robustness computation: %s", e)
        return 0.0


def evaluate_cycle(
    model,
    dataloader: DataLoader,
    tokenizer,
    base_dataset,
    distortion_fn,
    *,
    text_column: str = "text",
    max_length: int = 128,
    batch_size: int = 8,
    device="cpu",
) -> dict:
    """Lightweight per-cycle probe: clean accuracy + robustness at 3 strengths.

    Reuses recall_score() for accuracy and quick_robustness_score() for
    robustness, keeping this cheap enough to run after every training cycle.
    """
    recall = recall_score(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=device,
    )
    accuracy = recall["token_accuracy"]

    robustness = {}
    for strength in (0.3, 0.5, 0.7):
        robustness[strength] = quick_robustness_score(
            model=model,
            base_dataset=base_dataset,
            tokenizer=tokenizer,
            distortion_fn=distortion_fn,
            strength=strength,
            text_column=text_column,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )

    return {
        "accuracy": accuracy,
        "robustness": robustness,
    }


def recall_score(
    model,
    dataloader: DataLoader,
    tokenizer,
    device="cpu",
) -> dict:
    """Compute recall-style metrics on clean test data.

    Measures the model's ability to correctly predict next tokens
    on clean, unperturbed test data.

    Args:
        model: Language model.
        dataloader: DataLoader for clean test data.
        tokenizer: Tokenizer for decoding.
        device: Device to run inference on.

    Returns:
        Dict with perplexity and token-level accuracy.
    """
    model.eval()

    if tokenizer is None:
        metrics = classification_metrics(model, dataloader, device)
        if "error" in metrics:
            return {"metric": "recall", "token_accuracy": 0.0, "perplexity": float("inf")}
        return {
            "metric": "recall",
            "token_accuracy": metrics["accuracy"],
            "perplexity": float("inf"),
        }

    if tokenizer.pad_token_id is None:
        fallback = getattr(tokenizer, "eos_token_id", None) or 0
        logger.warning("tokenizer.pad_token_id is None, falling back to %d", fallback)
        tokenizer.pad_token_id = fallback

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing recall"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch.get("input_ids"))
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            predictions = shift_logits.argmax(dim=-1)

            # Only count non-padding tokens
            mask = shift_labels != tokenizer.pad_token_id
            correct += ((predictions == shift_labels) & mask).sum().item()
            total += mask.sum().item()

    accuracy = correct / max(total, 1)
    perplexity = compute_perplexity(model, dataloader, device)

    return {
        "metric": "recall",
        "token_accuracy": _safe_float(accuracy),
        "perplexity": _safe_float(perplexity, default=float("inf")),
    }


def generalization_score(
    model,
    ood_dataloader: DataLoader,
    clean_dataloader: DataLoader,
    device="cpu",
) -> dict:
    """Compute generalization score on out-of-distribution data.

    Compares perplexity on OOD data vs clean data. A smaller ratio
    indicates better generalization.

    Args:
        model: Language model.
        ood_dataloader: DataLoader for out-of-distribution data.
        clean_dataloader: DataLoader for clean in-distribution data.
        device: Device to run inference on.

    Returns:
        Dict with OOD perplexity, clean perplexity, and generalization ratio.
    """
    ood_ppl = compute_perplexity(model, ood_dataloader, device)
    clean_ppl = compute_perplexity(model, clean_dataloader, device)

    # Ratio close to 1.0 = good generalization
    ratio = ood_ppl / max(clean_ppl, 1e-6)

    return {
        "metric": "generalization",
        "ood_perplexity": _safe_float(ood_ppl, default=float("inf")),
        "clean_perplexity": _safe_float(clean_ppl, default=float("inf")),
        "generalization_ratio": _safe_float(ratio, default=float("inf")),
        "generalization_score": _safe_float(1.0 / max(ratio, 1e-8)),
    }


def robustness_score(
    model,
    base_dataset,
    tokenizer,
    distortion_fn,
    strengths: Optional[list] = None,
    text_column: str = "text",
    max_length: int = 128,
    batch_size: int = 8,
    device="cpu",
    failure_records: Optional[list[dict[str, Any]]] = None,
    total_samples_per_distortion: Optional[dict[str, int]] = None,
    export_failures: bool = False,
) -> dict:
    """Compute robustness score under increasing distortion strengths.

    Measures how gracefully model performance degrades as distortion
    intensity increases. Reports area under the robustness curve (AUC).

    Args:
        model: Language model.
        base_dataset: Base HuggingFace Dataset to distort at various strengths.
        tokenizer: Tokenizer for encoding.
        distortion_fn: Function(text, strength) -> distorted_text.
        strengths: List of distortion strengths to evaluate at.
        text_column: Name of the text column.
        max_length: Max sequence length for tokenization.
        batch_size: Batch size for evaluation.
        device: Device to run inference on.
        failure_records: Optional list of per-sample evaluation failure records.
        total_samples_per_distortion: Optional mapping of distortion types to sample totals.

    Returns:
        Dict with per-strength perplexities, AUC robustness score, and failure categories.
    """
    if strengths is None:
        strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    failure_categories = categorize_failures_by_distortion(
        failure_records, total_samples_per_distortion
    )

    is_vision = tokenizer is None or not hasattr(base_dataset, "map")

    all_failures = []

    if is_vision:
        accuracies = []

        # Baseline collection
        class DummyGenerator0:
            def __init__(self, strength, seed, config):
                self.strength = strength
                self.seed = seed
                self.config = config
                self.target_model = model

        from nightmarenet.data.generator import DistortedVisionDataset

        dummy_gen_0 = DummyGenerator0(0.0, seed=42, config={})
        clean_ds = DistortedVisionDataset(base_dataset, dummy_gen_0, phase="dream")
        clean_dl = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)
        clean_metrics = classification_metrics(model, clean_dl, device, return_per_sample=True)
        baseline_confs = clean_metrics.get("per_sample_confs", [])
        orig_preds = clean_metrics.get("per_sample_preds", [])
        orig_confs = clean_metrics.get("per_sample_confs", [])
        failures_data = []

        for strength in strengths:

            class DummyGenerator:
                def __init__(self, strength, seed, config):
                    self.strength = strength
                    self.seed = seed
                    self.config = config
                    self.target_model = model

            from nightmarenet.data.generator import DistortedVisionDataset

            dummy_gen = DummyGenerator(strength, seed=42, config={})
            distorted_ds = DistortedVisionDataset(base_dataset, dummy_gen, phase="dream")
            dataloader = DataLoader(distorted_ds, batch_size=batch_size, shuffle=False)

            metrics = classification_metrics(model, dataloader, device, return_per_sample=True)
            acc = metrics.get("accuracy", 0.0)
            accuracies.append(acc)

            if export_failures:
                dist_preds = metrics.get("per_sample_preds", [])
                dist_confs = metrics.get("per_sample_confs", [])
                for i in range(len(orig_preds)):
                    if i >= len(dist_preds):
                        break
                    failures_data.append(
                        {
                            "sample_index": i,
                            "original_input": "<vision_input>",
                            "distorted_input": "<vision_distorted>",
                            "original_prediction": int(orig_preds[i]),
                            "distorted_prediction": int(dist_preds[i]),
                            "original_confidence": float(orig_confs[i]),
                            "distorted_confidence": float(dist_confs[i]),
                            "confidence_drop": float(orig_confs[i] - dist_confs[i]),
                            "distortion_type": "vision_distortion",
                            "distortion_strength": float(strength),
                            "seed": 42,
                        }
                    )

            logger.info("Robustness - Strength %.1f: Accuracy = %.4f", strength, acc)

            dist_confs = metrics.get("per_sample_confs", [])
            for i in range(len(baseline_confs)):
                if i >= len(dist_confs):
                    break
                delta = compute_confidence_delta(baseline_confs[i], dist_confs[i])
                if delta > 0.0:
                    all_failures.append(
                        {
                            "sample_index": i,
                            "preview": "N/A",
                            "clean_confidence": baseline_confs[i],
                            "distorted_confidence": dist_confs[i],
                            "confidence_delta": delta,
                        }
                    )

        _trapz_fn = getattr(np, "trapezoid", None)
        if _trapz_fn is None:
            _trapz_fn = np.trapz  # type: ignore[attr-defined]
        auc = float(_trapz_fn(accuracies, strengths))

        res = {
            "metric": "robustness",
            "strengths": strengths,
            "accuracies": accuracies,
            "auc_robustness": _safe_float(auc),
            "failure_categories": failure_categories,
        }
        if export_failures:
            res["per_sample_data"] = failures_data
        return res

        ranked_failures = rank_failures(all_failures)
        res["top_failures"] = ranked_failures[:10]
        res["delta_distribution"] = build_delta_distribution(ranked_failures)
        res["confidence_deltas"] = [f["confidence_delta"] for f in ranked_failures]
        return res

    perplexities = []
    failures_data = []

    orig_preds = []
    orig_confs = []
    orig_texts = []

    if export_failures:
        # Run classification on clean dataset to get original preds
        def _get_preds(dataset):
            def tok_fn(examples):
                return tokenizer(
                    examples[text_column],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )

            tok_ds = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names)
            tok_ds.set_format("torch")
            dl = DataLoader(tok_ds, batch_size=batch_size, shuffle=False)
            return classification_metrics(model, dl, device, return_per_sample=True)

        clean_metrics = _get_preds(base_dataset)
        orig_preds = clean_metrics.get("per_sample_preds", [])
        orig_confs = clean_metrics.get("per_sample_confs", [])
        orig_texts = [x[text_column] for x in base_dataset]

    def _get_ppls(dataset):
        def tok_fn(examples):
            return tokenizer(
                examples[text_column],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        tok_ds = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names)
        tok_ds.set_format("torch")
        dl = DataLoader(tok_ds, batch_size=batch_size, shuffle=False)
        res = compute_perplexity(model, dl, device, return_per_sample=True)
        if isinstance(res, dict):
            return res.get("per_sample_ppls", [])
        return []

    baseline_ppls = _get_ppls(base_dataset)
    baseline_confs = [1.0 / max(p, 1e-8) for p in baseline_ppls]
    texts = [x[text_column] for x in base_dataset]

    for strength in strengths:
        # Apply distortion at this strength
        distorted = base_dataset.map(
            lambda x, _s=strength: {text_column: distortion_fn(x[text_column], strength=_s)},
            desc=f"Distorting at strength {strength:.1f}",
        )

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples[text_column],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        tokenized = distorted.map(
            tokenize_fn,
            batched=True,
            remove_columns=distorted.column_names,
        )
        tokenized.set_format("torch")
        dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=False)

        ppl_res = compute_perplexity(model, dataloader, device, return_per_sample=True)
        if isinstance(ppl_res, dict):
            ppl = ppl_res["perplexity"]
            dist_ppls = ppl_res.get("per_sample_ppls", [])
        else:
            ppl = ppl_res
            dist_ppls = []

        dist_confs = [1.0 / max(p, 1e-8) for p in dist_ppls]
        for i in range(len(baseline_confs)):
            if i >= len(dist_confs):
                break
            delta = compute_confidence_delta(baseline_confs[i], dist_confs[i])
            if delta > 0.0:
                all_failures.append(
                    {
                        "sample_index": i,
                        "preview": truncate_preview(texts[i]),
                        "clean_confidence": baseline_confs[i],
                        "distorted_confidence": dist_confs[i],
                        "confidence_delta": delta,
                    }
                )

        perplexities.append(ppl)

        if export_failures:
            dist_metrics = classification_metrics(model, dataloader, device, return_per_sample=True)
            dist_preds = dist_metrics.get("per_sample_preds", [])
            dist_confs = dist_metrics.get("per_sample_confs", [])
            dist_texts = [x[text_column] for x in distorted]

            for i in range(len(orig_preds)):
                if i >= len(dist_preds):
                    break
                failures_data.append(
                    {
                        "sample_index": i,
                        "original_input": orig_texts[i],
                        "distorted_input": dist_texts[i],
                        "original_prediction": int(orig_preds[i]),
                        "distorted_prediction": int(dist_preds[i]),
                        "original_confidence": float(orig_confs[i]),
                        "distorted_confidence": float(dist_confs[i]),
                        "confidence_drop": float(orig_confs[i] - dist_confs[i]),
                        "distortion_type": "text_distortion",
                        "distortion_strength": float(strength),
                        "seed": 42,
                    }
                )

        logger.info("Robustness - Strength %.1f: Perplexity = %.2f", strength, ppl)

    # Compute AUC using trapezoidal rule (normalized)
    # Lower perplexity = better, so we use 1/ppl for AUC
    inv_ppls = [1.0 / max(p, 1e-8) for p in perplexities]
    _trapz_fn = getattr(np, "trapezoid", None)
    if _trapz_fn is None:
        _trapz_fn = np.trapz  # type: ignore[attr-defined]
    auc = float(_trapz_fn(inv_ppls, strengths))

    res = {
        "metric": "robustness",
        "strengths": strengths,
        "perplexities": [_safe_float(p, default=float("inf")) for p in perplexities],
        "auc_robustness": _safe_float(auc),
        "failure_categories": failure_categories,
    }
    if export_failures:
        res["per_sample_data"] = failures_data
    return res

    ranked_failures = rank_failures(all_failures)
    res["top_failures"] = ranked_failures[:10]
    res["delta_distribution"] = build_delta_distribution(ranked_failures)
    res["confidence_deltas"] = [f["confidence_delta"] for f in ranked_failures]
    return res


def hallucination_rate(
    model,
    factual_dataloader: DataLoader,
    tokenizer,
    device="cpu",
    confidence_threshold: float = 0.5,
) -> dict:
    """Estimate hallucination rate via next-token prediction confidence.

    A proxy for hallucination: measures how often the model's top prediction
    diverges significantly from the ground truth on factual data. High
    divergence on factual data suggests the model may hallucinate.

    Args:
        model: Language model.
        factual_dataloader: DataLoader for factual text data.
        tokenizer: Tokenizer for decoding.
        device: Device to run inference on.

    Returns:
        Dict with hallucination rate and confidence metrics.
    """
    model.eval()
    total_predictions = 0
    hallucinated = 0
    confidence_scores = []

    try:
        with torch.no_grad():
            for batch in tqdm(factual_dataloader, desc="Computing hallucination rate"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch, labels=batch.get("input_ids"))
                logits = outputs.logits

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch["input_ids"][:, 1:].contiguous()

                # Get top-1 predictions and their probabilities
                probs = torch.softmax(shift_logits, dim=-1)
                top_probs, top_preds = probs.max(dim=-1)

                # Only evaluate non-padding tokens
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = getattr(tokenizer, "eos_token_id", None)
                if pad_token_id is None:
                    pad_token_id = 0
                mask = shift_labels != pad_token_id

                # Count hallucinations: incorrect prediction with high confidence
                incorrect = (top_preds != shift_labels) & mask
                high_confidence = top_probs > confidence_threshold
                hallucinated += (incorrect & high_confidence).sum().item()
                total_predictions += mask.sum().item()

                # Track confidence on incorrect predictions
                if incorrect.any():
                    confidence_scores.extend(top_probs[incorrect].cpu().numpy().tolist())
    except Exception as e:
        logger.warning("Error during hallucination rate computation: %s", e)
        return {
            "metric": "hallucination",
            "hallucination_rate": 0.0,
            "total_predictions": 0,
            "hallucinated_predictions": 0,
            "avg_hallucination_confidence": 0.0,
            "error": str(e),
        }

    rate = hallucinated / max(total_predictions, 1)
    avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0

    return {
        "metric": "hallucination",
        "hallucination_rate": _safe_float(rate),
        "total_predictions": total_predictions,
        "hallucinated_predictions": hallucinated,
        "avg_hallucination_confidence": _safe_float(avg_confidence),
    }


def classification_metrics(
    model,
    dataloader: DataLoader,
    device="cpu",
    return_per_sample: bool = False,
) -> dict:
    """Compute classification metrics (accuracy, F1, per-class stats).

    Args:
        model: Sequence classification model.
        dataloader: DataLoader providing tokenized batches with 'labels' column.
        device: Device to run inference on.

    Returns:
        Dict with accuracy, weighted F1, and per-class precision/recall/F1.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    model.eval()
    all_preds = []
    all_labels = []
    all_confs = []

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing classification metrics"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                probs = torch.softmax(logits, dim=-1)
                confs, preds = probs.max(dim=-1)

                preds = preds.cpu().numpy()
                confs = confs.cpu().numpy()

                if "labels" in batch:
                    labels = batch["labels"].cpu().numpy()
                    all_labels.extend(labels.tolist())
                else:
                    # Dummy labels if not present
                    all_labels.extend([0] * len(preds))

                all_preds.extend(preds.tolist())
                if return_per_sample:
                    all_confs.extend(confs.tolist())
    except Exception as e:
        logger.warning("Error during classification metrics computation: %s", e)
        return {
            "metric": "classification",
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "error": str(e),
        }

    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, zero_division=0
    )

    res = {
        "metric": "classification",
        "accuracy": _safe_float(accuracy),
        "f1_weighted": _safe_float(f1_weighted),
        "precision_per_class": [_safe_float(p) for p in precision],
        "recall_per_class": [_safe_float(r) for r in recall],
        "f1_per_class": [_safe_float(f) for f in f1_per_class],
        "support_per_class": support.tolist(),
    }
    if return_per_sample:
        res["per_sample_preds"] = all_preds
        res["per_sample_confs"] = all_confs
    return res
