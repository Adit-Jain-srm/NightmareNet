"""Dream-phase distortion pipeline (mild text + semantic + optional DP noise)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nightmarenet.distortions.semantic import apply_semantic_distortions
from nightmarenet.distortions.text import apply_text_distortions

DEFAULT_DP_DELTA = 1e-5
DEFAULT_DP_SENSITIVITY = 1.0


def calibrate_gaussian_sigma(
    epsilon: float,
    delta: float = DEFAULT_DP_DELTA,
    sensitivity: float = DEFAULT_DP_SENSITIVITY,
) -> float:
    """Calibrate Gaussian mechanism noise scale for (ε, δ)-DP.

    Uses the analytic bound σ = Δ · √(2 ln(1.25/δ)) / ε
    (Dwork & Roth, Algorithmic Foundations of Differential Privacy).

    Args:
        epsilon: Privacy budget per mechanism call (must be > 0).
        delta: Failure probability in (0, 1).
        sensitivity: L2 sensitivity of the query (Δ₂).

    Returns:
        Noise standard deviation σ.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if sensitivity < 0:
        raise ValueError(f"sensitivity must be >= 0, got {sensitivity}")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


@dataclass
class PrivacyAccountant:
    """Tracks cumulative ε spend across Dream-phase cycles."""

    budget: Optional[float] = None
    cumulative_epsilon: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)

    def spend(
        self,
        epsilon: float,
        *,
        cycle: Optional[int] = None,
        note: str = "",
    ) -> float:
        """Record an ε spend and return the new cumulative total."""
        if epsilon < 0:
            raise ValueError(f"epsilon spend must be >= 0, got {epsilon}")
        self.cumulative_epsilon += float(epsilon)
        self.events.append(
            {
                "epsilon": float(epsilon),
                "cycle": cycle,
                "note": note,
                "cumulative": self.cumulative_epsilon,
            }
        )
        return self.cumulative_epsilon

    def remaining(self) -> Optional[float]:
        """Return remaining budget, or None if no budget was set."""
        if self.budget is None:
            return None
        return max(0.0, float(self.budget) - self.cumulative_epsilon)


def apply_dp_gaussian_noise(
    text: str,
    epsilon: float,
    *,
    delta: float = DEFAULT_DP_DELTA,
    sensitivity: float = DEFAULT_DP_SENSITIVITY,
    seed: Optional[int] = None,
) -> str:
    """Apply a discrete Gaussian mechanism approximation to text codepoints.

    Each non-space character ``c`` is treated as a scalar query ``ord(c)`` with
    sensitivity ``sensitivity``. Calibrated N(0, σ²) noise is added and the
    result is rounded back to a printable codepoint. This is a research
    approximation for Dream-phase DP noise (not a claim of end-to-end DP-SGD).

    Args:
        text: Input string.
        epsilon: Per-call privacy parameter used to set σ.
        delta: DP δ parameter.
        sensitivity: Assumed L2 sensitivity.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Noised text string.
    """
    if not text:
        return text

    sigma = calibrate_gaussian_sigma(epsilon, delta=delta, sensitivity=sensitivity)
    rng = random.Random(seed)
    chars: List[str] = []
    for ch in text:
        if ch.isspace():
            chars.append(ch)
            continue
        noisy = ord(ch) + rng.gauss(0.0, sigma)
        code = int(round(noisy))
        if 32 <= ord(ch) <= 126:
            code = max(32, min(126, code))
        else:
            code = max(0, min(0x10FFFF, code))
        try:
            chars.append(chr(code))
        except ValueError:
            chars.append(ch)
    return "".join(chars)


def distort(
    text: str,
    strength: float,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Apply mild dream distortions to text, with optional DP noise."""
    if seed is not None:
        random.seed(seed)
    text_config = config.get("text") if config else None
    semantic_config = config.get("semantic") if config else None
    result = apply_text_distortions(text, strength=strength, config=text_config)
    result = apply_semantic_distortions(result, strength=strength, config=semantic_config)

    if config is None:
        return result

    epsilon = config.get("dream_dp_epsilon")
    if epsilon is None:
        return result

    delta = float(config.get("dream_dp_delta", DEFAULT_DP_DELTA))
    sensitivity = float(config.get("dream_dp_sensitivity", DEFAULT_DP_SENSITIVITY))
    return apply_dp_gaussian_noise(
        result,
        float(epsilon),
        delta=delta,
        sensitivity=sensitivity,
        seed=seed,
    )
