"""Builtin and custom keyboard layouts for multilingual typo noise.

Layouts store weighted neighbors (horizontal vs vertical). Inspired by
layout-aware typo models such as MulTypo (Zhao et al., ACL 2026), but
implemented clean-room without that package.
"""

from typing import Dict, List, Optional, Sequence, Tuple

NeighborList = List[Tuple[str, float]]


# Default weights: adjacent keys on the same row beat vertical neighbors.
DEFAULT_H_WEIGHT = 1.0
DEFAULT_V_WEIGHT = 0.65


def _neighbors_from_rows(
    rows: Sequence[str],
    h_weight: float = DEFAULT_H_WEIGHT,
    v_weight: float = DEFAULT_V_WEIGHT,
) -> Dict[str, NeighborList]:
    """Build adjacency from physical keyboard rows (left-to-right)."""
    grid = [list(r) for r in rows]
    coords: Dict[str, List[Tuple[int, int]]] = {}
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            coords.setdefault(ch, []).append((r, c))

    neighbors: Dict[str, NeighborList] = {}
    for ch, positions in coords.items():
        weighted: Dict[str, float] = {}
        for r, c in positions:
            for dc, w in ((-1, h_weight), (1, h_weight)):
                nc = c + dc
                if 0 <= nc < len(grid[r]):
                    other = grid[r][nc]
                    if other != ch:
                        weighted[other] = max(weighted.get(other, 0.0), w)
            for dr, w in ((-1, v_weight), (1, v_weight)):
                nr = r + dr
                if 0 <= nr < len(grid) and 0 <= c < len(grid[nr]):
                    other = grid[nr][c]
                    if other != ch:
                        weighted[other] = max(weighted.get(other, 0.0), w)
        neighbors[ch] = [(k, v) for k, v in sorted(weighted.items())]
    return neighbors


# Physical rows (lowercase / base letters only).
_LAYOUT_ROWS: Dict[str, Tuple[str, ...]] = {
    "qwerty": ("qwertyuiop", "asdfghjkl", "zxcvbnm"),
    "qwertz": ("qwertzuiop", "asdfghjkl", "yxcvbnm"),
    "azerty": ("azertyuiop", "qsdfghjklm", "wxcvbn"),
    # ЙЦУКЕН (Russian)
    "cyrillic": (
        "йцукенгшщзхъ",
        "фывапролджэ",
        "ячсмитьбю",
    ),
    # Simplified Arabic PC (common letter block)
    "arabic": (
        "ضصثقفغعهخح",
        "شسيبلاتنمك",
        "ئءؤرلاىةوزظ",
    ),
    # Simplified InScript-style Devanagari letter block
    "devanagari": (
        "कखगघङचछजझञ",
        "टठडढणतथदधन",
        "पफबभमयरलवशषसह",
    ),
}


_CUSTOM_LAYOUTS: Dict[str, Dict[str, NeighborList]] = {}


def list_layouts() -> List[str]:
    names = set(_LAYOUT_ROWS) | set(_CUSTOM_LAYOUTS)
    return sorted(names)


def register_layout(
    name: str,
    rows: Sequence[str],
    *,
    h_weight: float = DEFAULT_H_WEIGHT,
    v_weight: float = DEFAULT_V_WEIGHT,
    overwrite: bool = False,
) -> None:
    """Register a custom keyboard layout from row strings.

    Args:
        name: Layout id (e.g. ``dvorak``).
        rows: Physical key rows, left-to-right.
        h_weight: Weight for same-row neighbors.
        v_weight: Weight for vertical neighbors.
        overwrite: Replace an existing custom layout with the same name.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("layout name must be non-empty")
    if key in _LAYOUT_ROWS:
        raise ValueError(f"cannot overwrite builtin layout '{key}'")
    if key in _CUSTOM_LAYOUTS and not overwrite:
        raise ValueError(f"layout '{key}' already registered")
    _CUSTOM_LAYOUTS[key] = _neighbors_from_rows(rows, h_weight=h_weight, v_weight=v_weight)


def unregister_layout(name: str) -> None:
    """Remove a previously registered custom layout (builtins are kept)."""
    _CUSTOM_LAYOUTS.pop(name.strip().lower(), None)


def get_layout(
    name: str,
    *,
    h_weight: float = DEFAULT_H_WEIGHT,
    v_weight: float = DEFAULT_V_WEIGHT,
) -> Dict[str, NeighborList]:
    """Return neighbor map for a layout name."""
    key = name.strip().lower()
    if key in _CUSTOM_LAYOUTS:
        return _CUSTOM_LAYOUTS[key]
    if key not in _LAYOUT_ROWS:
        available = ", ".join(list_layouts())
        raise KeyError(f"Unknown keyboard layout '{name}'. Available: {available}")
    return _neighbors_from_rows(_LAYOUT_ROWS[key], h_weight=h_weight, v_weight=v_weight)


# Language code / name -> default layout
LANGUAGE_LAYOUTS: Dict[str, str] = {
    "english": "qwerty",
    "en": "qwerty",
    "german": "qwertz",
    "de": "qwertz",
    "french": "azerty",
    "fr": "azerty",
    "russian": "cyrillic",
    "ru": "cyrillic",
    "hindi": "devanagari",
    "hi": "devanagari",
    "arabic": "arabic",
    "ar": "arabic",
}


def normalize_language(language: Optional[str]) -> str:
    if not language:
        return "english"
    key = language.strip().lower()
    if key not in LANGUAGE_LAYOUTS:
        supported = ", ".join(sorted({k for k in LANGUAGE_LAYOUTS if len(k) > 2}))
        raise ValueError(f"Unsupported language '{language}'. Supported: {supported}")
    return key


def default_layout_for_language(language: Optional[str]) -> str:
    key = normalize_language(language)
    # Prefer canonical names over ISO codes for lookups that already passed normalize
    if key in LANGUAGE_LAYOUTS:
        return LANGUAGE_LAYOUTS[key]
    return "qwerty"


def resolve_layout_name(
    language: Optional[str] = None,
    keyboard_layout: Optional[str] = None,
) -> str:
    if keyboard_layout:
        return keyboard_layout.strip().lower()
    return default_layout_for_language(language)
