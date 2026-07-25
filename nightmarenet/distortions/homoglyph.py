from __future__ import annotations

import random
from typing import Optional

from nightmarenet.distortions.text import KEYBOARD_ADJACENT
from nightmarenet.utils.validation import validate_strength

# Latin to visual homoglyph mapping (mostly Cyrillic lookalikes)
LATIN_TO_HOMOGLYPH = {
    "a": "а",  # Cyrillic small letter a
    "c": "с",  # Cyrillic small letter es
    "e": "е",  # Cyrillic small letter ie
    "i": "і",  # Cyrillic small letter byelorussian-ukrainian i
    "j": "ј",  # Cyrillic small letter je
    "o": "о",  # Cyrillic small letter o
    "p": "р",  # Cyrillic small letter er
    "s": "ѕ",  # Cyrillic small letter dze
    "x": "х",  # Cyrillic small letter ha
    "y": "у",  # Cyrillic small letter u
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "I": "І",
    "J": "Ј",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "S": "Ѕ",
    "T": "Т",
    "X": "Х",
    "Y": "У",
}


def distort(text: str, strength: float, seed: Optional[int] = None) -> str:
    """Apply homoglyph and keyboard typo distortions to text.

    Args:
        text: Input text string.
        strength: Float 0-1 controlling overall distortion probability.
        seed: Optional random seed for reproducibility.

    Returns:
        Text with character-level lookalike or adjacent-key typo distortions.
    """
    validate_strength(strength)

    if not text:
        return text

    rng = random.Random(seed)
    chars = list(text)

    for i, ch in enumerate(chars):
        if rng.random() < strength:
            options = []

            # Homoglyph option
            if ch in LATIN_TO_HOMOGLYPH:
                options.append(LATIN_TO_HOMOGLYPH[ch])

            # Adjacent key typo option
            ch_lower = ch.lower()
            if ch_lower in KEYBOARD_ADJACENT:
                typos = KEYBOARD_ADJACENT[ch_lower]
                typo_char = rng.choice(typos)
                if ch.isupper():
                    typo_char = typo_char.upper()
                options.append(typo_char)

            if options:
                chars[i] = rng.choice(options)

    return "".join(chars)
