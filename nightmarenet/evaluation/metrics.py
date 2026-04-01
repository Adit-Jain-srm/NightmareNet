"""Evaluation metrics for measuring model robustness, generalization, and quality.

Implements metrics for:
- Recall / F1 on clean test data
- Generalization score on out-of-distribution data
- Robustness score under increasing distortion
- Hallucination rate
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_perplexity(model, dataloader: DataLoader, device="cpu") -> float:
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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch.get("input_ids"))
            total_loss += outputs.loss.item() * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return float(perplexity)


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
        "token_accuracy": accuracy,
        "perplexity": perplexity,
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
    ratio = ood_ppl / max(clean_ppl, 1e-8)

    return {
        "metric": "generalization",
        "ood_perplexity": ood_ppl,
        "clean_perplexity": clean_ppl,
        "generalization_ratio": ratio,
        "generalization_score": 1.0 / max(ratio, 1e-8),  # Higher is better
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

    Returns:
        Dict with per-strength perplexities and AUC robustness score.
    """
    if strengths is None:
        strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    perplexities = []

    for strength in strengths:
        # Apply distortion at this strength
        distorted = base_dataset.map(
            lambda x: {
                text_column: distortion_fn(x[text_column], strength=strength)
            },
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
        dataloader = DataLoader(tokenized, batch_size=batch_size)

        ppl = compute_perplexity(model, dataloader, device)
        perplexities.append(ppl)
        logger.info("Robustness - Strength %.1f: Perplexity = %.2f", strength, ppl)

    # Compute AUC using trapezoidal rule (normalized)
    # Lower perplexity = better, so we use 1/ppl for AUC
    inv_ppls = [1.0 / max(p, 1e-8) for p in perplexities]
    auc = float(np.trapz(inv_ppls, strengths))

    return {
        "metric": "robustness",
        "strengths": strengths,
        "perplexities": perplexities,
        "auc_robustness": auc,
    }


def hallucination_rate(
    model,
    factual_dataloader: DataLoader,
    tokenizer,
    device="cpu",
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
            mask = shift_labels != tokenizer.pad_token_id

            # Count hallucinations: incorrect prediction with high confidence
            incorrect = (top_preds != shift_labels) & mask
            high_confidence = top_probs > 0.5
            hallucinated += (incorrect & high_confidence).sum().item()
            total_predictions += mask.sum().item()

            # Track confidence on incorrect predictions
            if incorrect.any():
                confidence_scores.extend(
                    top_probs[incorrect].cpu().numpy().tolist()
                )

    rate = hallucinated / max(total_predictions, 1)
    avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0

    return {
        "metric": "hallucination",
        "hallucination_rate": rate,
        "total_predictions": total_predictions,
        "hallucinated_predictions": hallucinated,
        "avg_hallucination_confidence": avg_confidence,
    }
