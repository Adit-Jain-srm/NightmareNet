"""Causal LM robustness metrics (perplexity under named distortions).

Sequence-classification benchmarks use accuracy. GPT-2-class models need a
perplexity-based analogue: how much does PPL degrade when text is distorted?
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEFAULT_DISTORTION_TYPES = ("dream", "nightmare", "text")


def _build_distorter(name: str, strength: float) -> Callable[[str], str]:
    """Return a text→text callable for a named distortion family."""
    key = name.strip().lower()
    if key == "dream":
        from nightmarenet.distortions import dream as dream_mod

        def fn_dream(text: str) -> str:
            return dream_mod.distort(text, strength=strength, seed=42)

        return fn_dream

    if key == "nightmare":
        from nightmarenet.distortions import nightmare as nightmare_mod

        # Disable slow learned-adversarial path for eval sweeps.
        cfg = {
            "adversarial": {
                "contradiction": 0.3,
                "ambiguity": 0.3,
                "cross_domain": 0.2,
                "misleading_context": 0.2,
                "learned": 0.0,
            }
        }

        def fn_night(text: str) -> str:
            return nightmare_mod.distort(text, strength=strength, seed=42, config=cfg)

        return fn_night

    if key in ("text", "char", "surface"):
        from nightmarenet.distortions.text import apply_text_distortions

        def fn_text(text: str) -> str:
            return apply_text_distortions(text, strength=strength)

        return fn_text

    if key == "semantic":
        from nightmarenet.distortions.semantic import apply_semantic_distortions

        def fn_sem(text: str) -> str:
            return apply_semantic_distortions(text, strength=strength)

        return fn_sem

    raise ValueError(
        f"Unknown distortion type {name!r}; expected one of "
        f"dream, nightmare, text, semantic"
    )


def texts_to_dataloader(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int = 128,
    batch_size: int = 4,
) -> DataLoader:
    """Tokenize strings into a causal-LM DataLoader (input_ids + attention_mask)."""
    if not texts:
        raise ValueError("texts must be non-empty")
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])

    def _collate(batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([b[0] for b in batch])
        attention_mask = torch.stack([b[1] for b in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)


def perplexity_on_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: str = "cpu",
    max_length: int = 128,
    batch_size: int = 4,
) -> float:
    """Compute causal LM perplexity on a list of raw text strings."""
    # Lazy import: metrics.py pulls optional heavy deps (datasets) at module load.
    from nightmarenet.evaluation.metrics import compute_perplexity

    loader = texts_to_dataloader(
        tokenizer, texts, max_length=max_length, batch_size=batch_size
    )
    return float(compute_perplexity(model, loader, device=device))


def relative_ppl_degradation(clean_ppl: float, distorted_ppl: float) -> float:
    """Relative PPL increase under distortion (0 = no degradation; higher = worse)."""
    if clean_ppl <= 0 or clean_ppl == float("inf"):
        return float("inf")
    if distorted_ppl == float("inf"):
        return float("inf")
    return max(0.0, (distorted_ppl - clean_ppl) / clean_ppl)


def evaluate_causal_lm_robustness(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    distortion_types: Optional[Sequence[str]] = None,
    strength: float = 0.5,
    device: str = "cpu",
    max_length: int = 128,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """Evaluate causal LM robustness via perplexity under named distortions.

    This is the GPT-2-class counterpart to classification accuracy curves:
    lower distorted perplexity (and lower relative degradation) is better.

    Args:
        model: Causal LM (``AutoModelForCausalLM``-compatible).
        tokenizer: Matching tokenizer.
        texts: Evaluation strings (non-empty).
        distortion_types: Named families (default: dream, nightmare, text).
        strength: Distortion intensity in ``[0, 1]``.
        device: Torch device string.
        max_length: Truncation length.
        batch_size: Eval batch size.

    Returns:
        Dict with ``clean_perplexity``, per-type distorted PPL / deltas, and a
        scalar ``robustness_score`` = ``clean_ppl / mean_distorted_ppl``
        (higher is better; 1.0 means no average degradation).
    """
    types = list(distortion_types or DEFAULT_DISTORTION_TYPES)
    if len(types) < 3:
        raise ValueError("evaluate_causal_lm_robustness requires at least 3 distortion types")

    clean_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not clean_texts:
        raise ValueError("texts must contain at least one non-empty string")

    clean_ppl = perplexity_on_texts(
        model,
        tokenizer,
        clean_texts,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    per_distortion: Dict[str, Any] = {}
    distorted_ppls: List[float] = []
    for dtype in types:
        distort_fn = _build_distorter(dtype, strength)
        distorted = [distort_fn(t) for t in clean_texts]
        # Keep non-empty after distortion; fall back to original if emptied.
        distorted = [d if d.strip() else t for d, t in zip(distorted, clean_texts)]
        ppl = perplexity_on_texts(
            model,
            tokenizer,
            distorted,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )
        deg = relative_ppl_degradation(clean_ppl, ppl)
        per_distortion[dtype] = {
            "perplexity": float(ppl),
            "delta_ppl": float(ppl - clean_ppl) if ppl != float("inf") else float("inf"),
            "relative_degradation": float(deg),
        }
        distorted_ppls.append(float(ppl))
        logger.info(
            "Causal LM robustness [%s @ %.2f]: ppl=%.4f (clean=%.4f, relΔ=%.4f)",
            dtype,
            strength,
            ppl,
            clean_ppl,
            deg,
        )

    finite = [p for p in distorted_ppls if p != float("inf")]
    mean_distorted = sum(finite) / len(finite) if finite else float("inf")
    if mean_distorted == float("inf") or mean_distorted <= 0:
        robustness_score = 0.0
    else:
        robustness_score = float(clean_ppl / mean_distorted)

    return {
        "metric": "causal_lm_robustness",
        "strength": strength,
        "distortion_types": types,
        "clean_perplexity": float(clean_ppl),
        "per_distortion": per_distortion,
        "mean_distorted_perplexity": float(mean_distorted),
        "mean_relative_degradation": float(
            sum(relative_ppl_degradation(clean_ppl, p) for p in finite) / max(len(finite), 1)
            if finite
            else float("inf")
        ),
        "robustness_score": robustness_score,
    }
