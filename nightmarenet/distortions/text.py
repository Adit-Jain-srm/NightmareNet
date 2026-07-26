"""Text-level corruption functions for dream and nightmare data generation.

Each function accepts a `strength` float (0–1) controlling distortion intensity.
Low strength (0.2–0.3) is used for Dream phase; high strength (0.7–0.9) for Nightmare.
"""

from __future__ import annotations

import logging
import random
import string

import regex

from nightmarenet.distortions import char_maps
from nightmarenet.utils.validation import validate_strength

logger = logging.getLogger(__name__)


# Keyboard adjacency map for simulating typos. Kept as a public alias of the
# canonical Latin map in char_maps.latin for backward compatibility --
# existing code importing KEYBOARD_ADJACENT from this module keeps working.
KEYBOARD_ADJACENT = char_maps.latin.TYPO_MAP


def _graphemes(text: str) -> list:
    """Split text into grapheme clusters (user-perceived characters).

    Unlike `list(text)`, which splits on raw Unicode code points, this keeps
    combining marks attached to their base character -- e.g. Arabic tashkeel,
    Devanagari matras, or accented Latin letters written as base + combining
    accent. Operating on graphemes instead of code points is what makes the
    distortions in this module safe to run on non-Latin scripts.
    """
    return regex.findall(r"\X", text)


def char_swap(text, strength=0.3) -> str:
    """Randomly swap adjacent grapheme clusters in the text.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of swapping each character pair.

    Returns:
        Corrupted text with some adjacent character pairs swapped.
    """
    chars = _graphemes(text)
    if len(chars) < 2:
        return text
    i = 0
    while i < len(chars) - 1:
        if random.random() < strength * 0.3:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2  # Skip next to avoid double-swap
        else:
            i += 1
    return "".join(chars)


def char_insert(text, strength=0.3) -> str:
    """Randomly insert characters into the text.

    Inserted characters are drawn from the same script as the grapheme they
    follow (falling back to Latin lowercase for scripts without a typo map),
    so CJK/Arabic/Cyrillic/Devanagari text doesn't get random Latin noise
    spliced in.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of insertion at each position.

    Returns:
        Corrupted text with random characters inserted.
    """
    if not text:
        return text
    chars = _graphemes(text)
    result = []
    for ch in chars:
        result.append(ch)
        if random.random() < strength * 0.15:
            script = char_maps.detect_script(ch)
            pool = list(char_maps.get_typo_map(script).keys()) or list(string.ascii_lowercase)
            result.append(random.choice(pool))
    return "".join(result)


def char_delete(text, strength=0.3) -> str:
    """Randomly delete grapheme clusters from the text.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of deleting each character.

    Returns:
        Corrupted text with some characters removed.
    """
    if not text:
        return text
    return "".join(ch for ch in _graphemes(text) if random.random() > strength * 0.15)


def keyboard_typo(text, strength=0.3) -> str:
    """Replace characters with QWERTY-keyboard-adjacent characters (Latin only).

    Kept as a Latin-only, backward-compatible entry point. For script-aware
    typo simulation across CJK/Arabic/Cyrillic/Devanagari text, use
    `typo_injection` instead.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of each character being replaced.

    Returns:
        Corrupted text with keyboard-adjacent character replacements.
    """
    if not text:
        return text
    chars = _graphemes(text)
    for i, ch in enumerate(chars):
        if random.random() < strength * 0.2 and ch.lower() in KEYBOARD_ADJACENT:
            adjacent = KEYBOARD_ADJACENT[ch.lower()]
            replacement = random.choice(adjacent)
            chars[i] = replacement.upper() if ch.isupper() else replacement
    return "".join(chars)


def typo_injection(text, strength=0.3) -> str:
    """Replace characters with script-appropriate confused characters.

    Unicode-aware, multi-script equivalent of `keyboard_typo`: each grapheme
    cluster is matched against the typo map for its own script (Latin
    keyboard-adjacency, Cyrillic ЙЦУКЕН-adjacency, Arabic dotted-letter
    confusion, CJK visually-similar-character confusion, Devanagari
    consonant/matra confusion), so mixed-script text gets plausible
    per-script noise instead of being skipped or corrupted.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of each character being replaced.

    Returns:
        Corrupted text with script-aware character replacements.
    """
    if not text:
        return text
    chars = _graphemes(text)
    for i, ch in enumerate(chars):
        if random.random() >= strength * 0.2:
            continue
        script = char_maps.detect_script(ch)
        typo_map = char_maps.get_typo_map(script)
        base = ch.lower() if script in ("latin", "cyrillic") else ch
        if base not in typo_map:
            continue
        replacement = random.choice(typo_map[base])
        if script in ("latin", "cyrillic") and ch.isupper():
            replacement = replacement.upper()
        chars[i] = replacement
    return "".join(chars)


def homoglyph_substitution(text, strength=0.3) -> str:
    """Replace characters with visually-similar Unicode homoglyphs.

    Uses a curated subset of the Unicode confusables database (see
    char_maps/*.py) to substitute characters with look-alikes -- often from
    a different script, e.g. Latin "a" -> Cyrillic "а". This simulates
    homograph-style corruption (the same class of confusion used in IDN
    homograph attacks) across Latin, Cyrillic, Arabic, CJK, and Devanagari
    text.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of each character being replaced.

    Returns:
        Corrupted text with homoglyph substitutions applied.
    """
    if not text:
        return text
    chars = _graphemes(text)
    for i, ch in enumerate(chars):
        if random.random() >= strength * 0.2:
            continue
        script = char_maps.detect_script(ch)
        confusables = char_maps.get_confusables(script)
        base = ch.lower() if script in ("latin", "cyrillic") else ch
        if base not in confusables or not confusables[base]:
            continue
        chars[i] = random.choice(confusables[base])
    return "".join(chars)


def word_shuffle(text, strength=0.3, window_size=5) -> str:
    """Shuffle words within a sliding window.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of shuffling each window.
        window_size: Size of the window within which words are shuffled.

    Returns:
        Text with words shuffled within windows.
    """
    words = text.split()
    if len(words) <= 1:
        return text

    # Effective window size scales with strength
    effective_window = max(2, int(window_size * strength))

    result = []
    i = 0
    while i < len(words):
        window = words[i : i + effective_window]
        if random.random() < strength and len(window) > 1:
            random.shuffle(window)
        result.extend(window)
        i += effective_window

    return " ".join(result)


def token_mask(text, strength=0.3, mask_token="[MASK]") -> str:
    """Replace random words with a mask token.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of masking each word.
        mask_token: Token to use as replacement.

    Returns:
        Text with some words replaced by the mask token.
    """
    words = text.split()
    if not words:
        return text
    return " ".join(mask_token if random.random() < strength * 0.3 else w for w in words)


def token_replace(text, strength=0.3, vocabulary=None) -> str:
    """Replace random words with random vocabulary tokens.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of replacing each word.
        vocabulary: Optional list of replacement words. Defaults to common English words.

    Returns:
        Text with some words replaced by random vocabulary tokens.
    """
    if vocabulary is None:
        vocabulary = [
            "the",
            "of",
            "and",
            "to",
            "in",
            "is",
            "it",
            "that",
            "was",
            "for",
            "on",
            "are",
            "with",
            "as",
            "his",
            "they",
            "be",
            "at",
            "one",
            "have",
            "this",
            "from",
            "by",
            "hot",
            "word",
            "but",
            "what",
            "some",
            "we",
            "can",
            "out",
            "other",
            "were",
            "all",
            "there",
            "when",
            "up",
            "use",
            "your",
            "how",
            "each",
            "she",
            "which",
            "do",
            "their",
            "time",
            "if",
            "will",
            "way",
            "about",
            "many",
            "then",
            "them",
            "would",
            "write",
            "like",
            "so",
            "these",
            "her",
            "long",
            "make",
            "thing",
            "see",
            "him",
            "two",
            "has",
            "look",
            "more",
            "day",
            "could",
            "go",
            "come",
            "did",
        ]

    words = text.split()
    if not words:
        return text
    return " ".join(
        random.choice(vocabulary) if random.random() < strength * 0.2 else w for w in words
    )


def apply_text_distortions(text, strength=0.3, config=None) -> str:
    """Apply a combination of text-level distortions based on config weights.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling overall distortion intensity.
        config: Optional dict mapping distortion names to their application probabilities.

    Returns:
        Corrupted text after applying selected distortions.
    """
    validate_strength(strength)

    if not text or not text.strip():
        return text

    try:
        default_config = {
            "char_swap": 0.3,
            "char_insert": 0.2,
            "char_delete": 0.2,
            "keyboard_typo": 0.3,
            "word_shuffle": 0.2,
            "token_mask": 0.3,
        }
        config = config or default_config

        distortion_funcs = {
            "char_swap": char_swap,
            "char_insert": char_insert,
            "char_delete": char_delete,
            "keyboard_typo": keyboard_typo,
            "typo_injection": typo_injection,
            "homoglyph_substitution": homoglyph_substitution,
            "word_shuffle": word_shuffle,
            "token_mask": token_mask,
        }

        result = text
        for name, prob in config.items():
            if name in distortion_funcs and random.random() < prob:
                result = distortion_funcs[name](result, strength=strength)

        return result
    except Exception:
        logger.warning("Text distortion failed; returning original text", exc_info=True)
        return text
