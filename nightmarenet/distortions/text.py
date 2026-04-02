"""Text-level corruption functions for dream and nightmare data generation.

Each function accepts a `strength` float (0–1) controlling distortion intensity.
Low strength (0.2–0.3) is used for Dream phase; high strength (0.7–0.9) for Nightmare.
"""

from __future__ import annotations

import logging
import random
import string

from nightmarenet.utils.validation import validate_strength

logger = logging.getLogger(__name__)


# Keyboard adjacency map for simulating typos
KEYBOARD_ADJACENT = {
    "a": "sqwz", "b": "vngh", "c": "xdfv", "d": "sfcxer",
    "e": "rdsw", "f": "dgcvrt", "g": "fhbvty", "h": "gjbnyu",
    "i": "ujko", "j": "hknmui", "k": "jlmio", "l": "kop",
    "m": "njk", "n": "bmhj", "o": "iklp", "p": "ol",
    "q": "wa", "r": "etdf", "s": "adwxez", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc",
    "y": "tghu", "z": "xsa",
}


def char_swap(text, strength=0.3) -> str:
    """Randomly swap adjacent characters in the text.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of swapping each character pair.

    Returns:
        Corrupted text with some adjacent character pairs swapped.
    """
    if len(text) < 2:
        return text
    chars = list(text)
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

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of insertion at each position.

    Returns:
        Corrupted text with random characters inserted.
    """
    if not text:
        return text
    chars = list(text)
    result = []
    for ch in chars:
        result.append(ch)
        if random.random() < strength * 0.15:
            result.append(random.choice(string.ascii_lowercase))
    return "".join(result)


def char_delete(text, strength=0.3) -> str:
    """Randomly delete characters from the text.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of deleting each character.

    Returns:
        Corrupted text with some characters removed.
    """
    if not text:
        return text
    return "".join(ch for ch in text if random.random() > strength * 0.15)


def keyboard_typo(text, strength=0.3) -> str:
    """Replace characters with keyboard-adjacent characters to simulate typos.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of each character being replaced.

    Returns:
        Corrupted text with keyboard-adjacent character replacements.
    """
    if not text:
        return text
    chars = list(text)
    for i, ch in enumerate(chars):
        if random.random() < strength * 0.2 and ch.lower() in KEYBOARD_ADJACENT:
            adjacent = KEYBOARD_ADJACENT[ch.lower()]
            replacement = random.choice(adjacent)
            chars[i] = replacement.upper() if ch.isupper() else replacement
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
    return " ".join(
        mask_token if random.random() < strength * 0.3 else w for w in words
    )


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
            "the", "of", "and", "to", "in", "is", "it", "that", "was", "for",
            "on", "are", "with", "as", "his", "they", "be", "at", "one", "have",
            "this", "from", "by", "hot", "word", "but", "what", "some", "we",
            "can", "out", "other", "were", "all", "there", "when", "up", "use",
            "your", "how", "each", "she", "which", "do", "their", "time", "if",
            "will", "way", "about", "many", "then", "them", "would", "write",
            "like", "so", "these", "her", "long", "make", "thing", "see", "him",
            "two", "has", "look", "more", "day", "could", "go", "come", "did",
        ]

    words = text.split()
    if not words:
        return text
    return " ".join(
        random.choice(vocabulary) if random.random() < strength * 0.2 else w
        for w in words
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
