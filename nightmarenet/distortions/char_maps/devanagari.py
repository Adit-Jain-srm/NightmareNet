"""Devanagari-script character maps.

Not required by the issue's acceptance criteria (which lists CJK, Arabic,
and Cyrillic), but included as bonus coverage since the Problem section
explicitly calls out Devanagari as an affected script and the char_maps
package is designed to make adding a script a small, self-contained file.

TYPO_MAP models two common real-world error classes: visually similar
consonants, and short/long matra (vowel sign) confusion -- the latter is
especially important to get right at the grapheme-cluster level, since a
matra is a combining mark that must stay attached to its base consonant.
"""

from __future__ import annotations

from typing import Dict, List

TYPO_MAP: Dict[str, str] = {
    "\u092c": "\u0935",
    "\u0935": "\u092c",
    "\u0921": "\u0926",
    "\u0926": "\u0921",
    "\u0927": "\u0918",
    "\u0918": "\u0927",
    "\u0925": "\u091f",
    "\u091f": "\u0925",
    "\u093f": "\u0940",
    "\u0940": "\u093f",
    "\u0941": "\u0942",
    "\u0942": "\u0941",
}

CONFUSABLES: Dict[str, List[str]] = {
    "\u092c": ["\u0935"],
    "\u0921": ["\u0926"],
    "\u093f": ["\u0940"],
}
