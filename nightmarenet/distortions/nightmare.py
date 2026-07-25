"""Nightmare-phase distortion pipeline (text + semantic + adversarial)."""

import random
from typing import Any, Dict, Optional

from nightmarenet.distortions.adversarial import apply_adversarial_distortions
from nightmarenet.distortions.semantic import apply_semantic_distortions
from nightmarenet.distortions.text import apply_text_distortions


def distort(
    text: str,
    strength: float,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Apply aggressive nightmare distortions to text."""
    if seed is not None:
        random.seed(seed)
    text_config = config.get("text") if config else None
    semantic_config = config.get("semantic") if config else None
    adversarial_config = config.get("adversarial") if config else None

    if adversarial_config is None and strength >= 0.5:
        adversarial_config = {
            "contradiction": 0.3,
            "ambiguity": 0.3,
            "cross_domain": 0.2,
            "misleading_context": 0.2,
            "learned": min(strength, 0.4),
        }

    result = apply_text_distortions(text, strength=strength, config=text_config)
    result = apply_semantic_distortions(result, strength=strength, config=semantic_config)
    return apply_adversarial_distortions(result, strength=strength, config=adversarial_config)
