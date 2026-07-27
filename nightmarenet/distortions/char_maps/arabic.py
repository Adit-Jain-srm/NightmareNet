"""Arabic-script character maps.

Arabic physical keyboard layouts vary significantly by OS and region, so
instead of guessing a single layout, TYPO_MAP models the most common
real-world Arabic typo/OCR error class: confusing letters that share the
same base glyph shape (rasm) and differ only by the number or placement of
diacritical dots -- e.g. letter pairs like b/t/th, j/h/kh, d/dh, r/z,
s/sh, and f/q. This is a well-documented error class for both fast typing
and OCR, and is more representative of real Arabic noise than an assumed
keyboard layout.

Text using these maps stays right-to-left: all replacements are drawn from
the same script, and Arabic contextual shaping (initial/medial/final glyph
forms) is recomputed at render time from the base letters, so substituting
one base Arabic letter for another does not corrupt shaping or break the
paragraph's RTL direction.
"""

from __future__ import annotations

from typing import Dict, List

TYPO_MAP: Dict[str, str] = {
    "\u0628": "\u062a\u062b",
    "\u062a": "\u0628\u062b",
    "\u062b": "\u0628\u062a",
    "\u062c": "\u062d\u062e",
    "\u062d": "\u062c\u062e",
    "\u062e": "\u062c\u062d",
    "\u062f": "\u0630",
    "\u0630": "\u062f",
    "\u0631": "\u0632",
    "\u0632": "\u0631",
    "\u0633": "\u0634",
    "\u0634": "\u0633",
    "\u0635": "\u0636",
    "\u0636": "\u0635",
    "\u0637": "\u0638",
    "\u0638": "\u0637",
    "\u0639": "\u063a",
    "\u063a": "\u0639",
    "\u0641": "\u0642",
    "\u0642": "\u0641",
    "\u0647": "\u0629",
    "\u0629": "\u0647",
    "\u0648": "\u0624",
    "\u0624": "\u0648",
    "\u064a": "\u0649\u0626",
}

CONFUSABLES: Dict[str, List[str]] = {
    "\u0647": ["\u0629", "\u06be"],
    "\u064a": ["\u0649", "\u0626"],
    "\u0648": ["\u0624"],
    "\u0627": ["\u0623", "\u0625", "\u0622"],
    "\u0643": ["\u06af"],
}
