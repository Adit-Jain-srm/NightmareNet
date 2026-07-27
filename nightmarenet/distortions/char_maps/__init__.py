"""Per-script character maps for Unicode-aware text distortions.

Each script submodule (latin, cyrillic, arabic, cjk, devanagari) exposes:

- ``TYPO_MAP``: ``Dict[str, str]`` mapping a character to a string of
  plausible mistyped/confused replacement characters, used by
  ``typo_injection``.
- ``CONFUSABLES``: ``Dict[str, List[str]]`` mapping a character to visually
  similar characters (often from a *different* script), used by
  ``homoglyph_substitution``.

``detect_script`` buckets a single character into one of the supported
script keys so the distortion functions never have to hardcode Unicode
ranges themselves. Adding a new script means adding one new module here and
one entry in the two registries below -- no changes needed in text.py.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List

from nightmarenet.distortions.char_maps import arabic, cjk, cyrillic, devanagari, latin

SUPPORTED_SCRIPTS = ("latin", "cyrillic", "arabic", "cjk", "devanagari")

_TYPO_MAPS: Dict[str, Dict[str, str]] = {
    "latin": latin.TYPO_MAP,
    "cyrillic": cyrillic.TYPO_MAP,
    "arabic": arabic.TYPO_MAP,
    "cjk": cjk.TYPO_MAP,
    "devanagari": devanagari.TYPO_MAP,
}

_CONFUSABLES: Dict[str, Dict[str, List[str]]] = {
    "latin": latin.CONFUSABLES,
    "cyrillic": cyrillic.CONFUSABLES,
    "arabic": arabic.CONFUSABLES,
    "cjk": cjk.CONFUSABLES,
    "devanagari": devanagari.CONFUSABLES,
}

# Coarse Unicode code-point ranges, only precise enough to pick the right
# typo/confusables table -- not a full script-detection implementation.
_SCRIPT_RANGES = (
    (0x0600, 0x06FF, "arabic"),  # Arabic
    (0x0750, 0x077F, "arabic"),  # Arabic Supplement
    (0xFB50, 0xFDFF, "arabic"),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF, "arabic"),  # Arabic Presentation Forms-B
    (0x0400, 0x04FF, "cyrillic"),  # Cyrillic
    (0x0500, 0x052F, "cyrillic"),  # Cyrillic Supplement
    (0x0900, 0x097F, "devanagari"),
    (0x4E00, 0x9FFF, "cjk"),  # CJK Unified Ideographs
    (0x3400, 0x4DBF, "cjk"),  # CJK Extension A
    (0x3040, 0x309F, "cjk"),  # Hiragana
    (0x30A0, 0x30FF, "cjk"),  # Katakana
    (0xAC00, 0xD7A3, "cjk"),  # Hangul syllables
    (0x0041, 0x005A, "latin"),
    (0x0061, 0x007A, "latin"),
    (0x00C0, 0x024F, "latin"),  # Latin-1 Supplement + Extended-A/B
)


def detect_script(char: str) -> str:
    """Return the script bucket for a single character/grapheme cluster.

    Looks at the first code point of ``char`` (the base character of a
    grapheme cluster, since combining marks appear after it) and falls back
    to ``"latin"`` for anything unrecognized -- punctuation, digits,
    symbols, emoji, whitespace, etc. -- so callers always get a usable map.
    """
    if not char:
        return "latin"
    cp = ord(char[0])
    for start, end, script in _SCRIPT_RANGES:
        if start <= cp <= end:
            return script
    return "latin"


def get_typo_map(script: str) -> Dict[str, str]:
    """Return the typo-adjacency map for a script, defaulting to Latin."""
    return _TYPO_MAPS.get(script, _TYPO_MAPS["latin"])


def get_confusables(script: str) -> Dict[str, List[str]]:
    """Return the homoglyph confusables map for a script, defaulting to Latin."""
    return _CONFUSABLES.get(script, _CONFUSABLES["latin"])


def is_rtl(char: str) -> bool:
    """True if ``char`` has a strong right-to-left Unicode bidi category.

    Only the base character is checked (consistent with detect_script),
    which is sufficient since combining marks inherit directionality from
    their base character under the Unicode Bidirectional Algorithm.
    """
    if not char:
        return False
    return unicodedata.bidirectional(char[0]) in ("AL", "R")
