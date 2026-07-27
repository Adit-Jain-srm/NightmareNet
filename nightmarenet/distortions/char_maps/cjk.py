"""CJK-script character maps.

CJK text is entered through an IME (pinyin, zhuyin, romaji, etc.), so there
is no meaningful "adjacent key" concept the way there is for Latin/Cyrillic
keyboards. TYPO_MAP instead models visually/structurally similar Han
characters -- the dominant real-world error class for OCR and fast
handwriting/typing (e.g. mistaking one stroke-similar character for
another). Homophone confusions (same pinyin, different character) are
intentionally out of scope here since they need a pronunciation dictionary;
see docs/plugin_development.md for how to extend this map with one.

This module also covers Hiragana/Katakana and Hangul syllables for the
purposes of script bucketing (char_maps.detect_script groups them with CJK)
even though the maps below focus on Han characters, which are the most
common case flagged in the issue.
"""

from __future__ import annotations

from typing import Dict, List

TYPO_MAP: Dict[str, str] = {
    "\u5df1": "\u5df2\u5df3",
    "\u5df2": "\u5df1\u5df3",
    "\u5df3": "\u5df1\u5df2",
    "\u4eba": "\u5165\u516b",
    "\u5165": "\u4eba\u516b",
    "\u516b": "\u4eba\u5165",
    "\u65e5": "\u66f0",
    "\u66f0": "\u65e5",
    "\u672a": "\u672b",
    "\u672b": "\u672a",
    "\u571f": "\u58eb",
    "\u58eb": "\u571f",
    "\u5e72": "\u5343",
    "\u5343": "\u5e72",
    "\u5200": "\u529b",
    "\u529b": "\u5200",
    "\u738b": "\u7389",
    "\u7389": "\u738b",
    "\u53c8": "\u53c9",
    "\u53c9": "\u53c8",
    "\u620a": "\u620c\u620d",
    "\u620c": "\u620a\u620d",
    "\u620d": "\u620a\u620c",
    "\u5927": "\u592a\u72ac",
    "\u592a": "\u5927\u72ac",
    "\u72ac": "\u5927\u592a",
    "\u76ee": "\u81ea",
    "\u81ea": "\u76ee",
    "\u53e3": "\u56d7",
    "\u56d7": "\u53e3",
}

CONFUSABLES: Dict[str, List[str]] = {
    "\u4e00": ["\u4e28", "\uff5c", "1", "l"],
    "\u53e3": ["\u56d7"],
    "\u4eba": ["\u5165"],
    "\u65e5": ["\u66f0"],
    "\u58eb": ["\u571f"],
    "\u5df1": ["\u5df2", "\u5df3"],
}
