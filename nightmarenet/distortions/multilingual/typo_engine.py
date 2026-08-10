"""Keyboard-layout-aware typo distortion (MulTypo-style, clean-room).

Four error classes on layout neighbors:
  replace  — nearby key
  insert   — accidental double press
  delete   — skipped key
  transpose — swap with next character

``strength`` in [0, 1] is treated as a character error rate target.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from nightmarenet.distortions.multilingual.keyboard_layouts import (
    get_layout,
    normalize_language,
    resolve_layout_name,
)

ERROR_TYPES = ("replace", "insert", "delete", "transpose")


def _pick_neighbor(neighbors: Sequence[Tuple[str, float]], rng: random.Random) -> str:
    chars, weights = zip(*neighbors)
    return rng.choices(list(chars), weights=list(weights), k=1)[0]


def _eligible_indices(chars: List[str], layout) -> List[int]:
    out = []
    for i, ch in enumerate(chars):
        key = ch.lower() if ch.isascii() and ch.isalpha() else ch
        if key in layout and layout[key]:
            out.append(i)
    return out


def distort(
    text: str,
    strength: float = 0.3,
    seed: Optional[int] = None,
    language: Optional[str] = None,
    keyboard_layout: Optional[str] = None,
    h_weight: float = 1.0,
    v_weight: float = 0.65,
) -> str:
    """Apply layout-aware typos to ``text``.

    Args:
        text: Input string.
        strength: Character error rate in [0, 1]. ``0`` is a no-op.
        seed: Optional RNG seed for determinism.
        language: english / german / french / russian / hindi / arabic (or ISO codes).
        keyboard_layout: Override layout id (qwerty, qwertz, azerty, ...).
        h_weight: Horizontal neighbor weight.
        v_weight: Vertical neighbor weight.
    """
    if not text:
        return text
    if strength <= 0.0:
        return text

    lang = normalize_language(language)
    layout_name = resolve_layout_name(lang, keyboard_layout)
    layout = get_layout(layout_name, h_weight=h_weight, v_weight=v_weight)
    rng = random.Random(seed)

    chars = list(text)
    # Cap edits so short strings still change under moderate strength.
    n_edits = max(1, int(round(len(chars) * min(strength, 1.0)))) if strength > 0 else 0
    # At very low strength, probabilistic single-pass CER is enough.
    if strength < 0.15:
        n_edits = sum(1 for _ in chars if rng.random() < strength)

    for _ in range(n_edits):
        eligible = _eligible_indices(chars, layout)
        if not eligible and len(chars) < 2:
            break
        err = rng.choice(ERROR_TYPES)

        if err == "delete" and chars:
            idx = rng.randrange(len(chars))
            del chars[idx]
            continue

        if err == "transpose" and len(chars) >= 2:
            idx = rng.randrange(len(chars) - 1)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            continue

        if not eligible:
            continue
        idx = rng.choice(eligible)
        ch = chars[idx]
        key = ch.lower() if ch.isascii() and ch.isalpha() else ch
        neighbors = layout.get(key) or []
        if not neighbors:
            continue

        replacement = _pick_neighbor(neighbors, rng)
        if ch.isascii() and ch.isupper() and replacement.isalpha():
            replacement = replacement.upper()

        if err == "insert":
            chars.insert(idx + 1, replacement)
        else:  # replace
            chars[idx] = replacement

    return "".join(chars)


def keyboard_typo(
    text: str,
    strength: float = 0.3,
    seed: Optional[int] = None,
    language: Optional[str] = None,
    keyboard_layout: Optional[str] = None,
) -> str:
    """Registry / CLI entry point for the ``keyboard_typo`` engine."""
    return distort(
        text,
        strength=strength,
        seed=seed,
        language=language,
        keyboard_layout=keyboard_layout,
    )
