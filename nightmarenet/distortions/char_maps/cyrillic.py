"""Cyrillic-script character maps.

TYPO_MAP models the standard ЙЦУКЕН keyboard layout's physical adjacency.
CONFUSABLES maps Cyrillic letters to their visually-identical Latin/Greek
counterparts (the reverse of the pairs in char_maps.latin), plus a couple
of same-script Cyrillic confusions (е/ё, и/й) that are common typing slips.
"""

from __future__ import annotations

from typing import Dict, List

TYPO_MAP: Dict[str, str] = {
    "й": "цф",
    "ц": "йыв",
    "у": "кеа",
    "к": "уеап",
    "е": "кнпр",
    "н": "егшр",
    "г": "нш",
    "ш": "гщо",
    "щ": "шзл",
    "з": "щхд",
    "х": "зъж",
    "ъ": "хэ",
    "ф": "йыя",
    "ы": "фцвч",
    "в": "ыцас",
    "а": "вупм",
    "п": "аори",
    "р": "пенто",
    "о": "ршгть",
    "л": "олдщ",
    "д": "лжэб",
    "ж": "дъб",
    "э": "жо",
    "я": "фчс",
    "ч": "ясм",
    "с": "чмвт",
    "м": "сит",
    "и": "мтьй",
    "т": "ирбю",
    "ь": "тбю",
    "б": "ьил",
    "ю": "бт",
}

CONFUSABLES: Dict[str, List[str]] = {
    "а": ["a"],
    "е": ["e", "ё"],
    "о": ["o", "0"],
    "р": ["p"],
    "с": ["c"],
    "х": ["x"],
    "у": ["y"],
    "і": ["i", "1"],
    "ј": ["j"],
    "ѕ": ["s"],
    "һ": ["h"],
    "к": ["k"],
    "м": ["m"],
    "т": ["t"],
    "и": ["й"],
}
