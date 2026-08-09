"""Latin-script character maps.

TYPO_MAP: QWERTY keyboard-adjacency, used to simulate fat-finger typos.
CONFUSABLES: a curated subset of the Unicode confusables database --
characters from other scripts that render as visually identical (or
near-identical) to a given Latin letter. This is the classic IDN-homograph
character set (Cyrillic/Greek look-alikes) rather than a live fetch of
Unicode's confusables.txt, since this package has no network access to
unicode.org at runtime. Extend this table as new confusable pairs are
verified; see docs/plugin_development.md.
"""

from __future__ import annotations

from typing import Dict, List

TYPO_MAP: Dict[str, str] = {
    "a": "sqwz",
    "b": "vngh",
    "c": "xdfv",
    "d": "sfcxer",
    "e": "rdsw",
    "f": "dgcvrt",
    "g": "fhbvty",
    "h": "gjbnyu",
    "i": "ujko",
    "j": "hknmui",
    "k": "jlmio",
    "l": "kop",
    "m": "njk",
    "n": "bmhj",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "etdf",
    "s": "adwxez",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "xsa",
}

CONFUSABLES: Dict[str, List[str]] = {
    "a": ["\u0430", "\u0251"],  # Cyrillic а, Latin alpha ɑ
    "e": ["\u0435"],  # Cyrillic е
    "o": ["\u043e", "0"],  # Cyrillic о, digit zero
    "p": ["\u0440", "\u03c1"],  # Cyrillic р, Greek rho ρ
    "c": ["\u0441", "\u03f2"],  # Cyrillic с, Greek lunate sigma ϲ
    "x": ["\u0445", "\u03c7"],  # Cyrillic х, Greek chi χ
    "y": ["\u0443", "\u03b3"],  # Cyrillic у, Greek gamma γ
    "i": ["\u0456", "1", "l"],  # Cyrillic (Ukrainian) і, digit one, lowercase L
    "j": ["\u0458"],  # Cyrillic je ј
    "s": ["\u0455"],  # Cyrillic dze ѕ
    "h": ["\u04bb"],  # Cyrillic shha һ
    "k": ["\u043a"],  # Cyrillic к
    "m": ["\u043c"],  # Cyrillic м
    "t": ["\u0442"],  # Cyrillic т
    "n": ["\u0578"],  # Armenian n ո (visually close to Latin n)
}
