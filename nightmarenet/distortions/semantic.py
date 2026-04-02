"""Semantic-level distortion functions for dream and nightmare data generation.

Meaning-level distortions that alter the semantics of text rather than just
surface characters. Each function accepts a `strength` float (0–1).
"""

from __future__ import annotations

import logging
import random

from nightmarenet.utils.validation import validate_strength

logger = logging.getLogger(__name__)


# Simple synonym dictionary for lightweight synonym replacement
SYNONYM_MAP = {
    "good": ["great", "fine", "excellent", "nice", "decent"],
    "bad": ["poor", "terrible", "awful", "horrible", "dreadful"],
    "big": ["large", "huge", "enormous", "massive", "vast"],
    "small": ["tiny", "little", "minute", "compact", "miniature"],
    "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
    "slow": ["sluggish", "gradual", "unhurried", "leisurely", "steady"],
    "happy": ["glad", "joyful", "pleased", "cheerful", "content"],
    "sad": ["unhappy", "sorrowful", "gloomy", "melancholy", "downcast"],
    "important": ["significant", "crucial", "vital", "essential", "key"],
    "old": ["ancient", "aged", "elderly", "vintage", "mature"],
    "new": ["recent", "fresh", "modern", "novel", "latest"],
    "beautiful": ["gorgeous", "lovely", "stunning", "attractive", "pretty"],
    "said": ["stated", "mentioned", "declared", "noted", "remarked"],
    "think": ["believe", "consider", "suppose", "reckon", "assume"],
    "make": ["create", "produce", "build", "construct", "form"],
    "know": ["understand", "realize", "recognize", "comprehend", "grasp"],
    "take": ["grab", "seize", "acquire", "obtain", "collect"],
    "see": ["observe", "notice", "spot", "view", "witness"],
    "come": ["arrive", "approach", "appear", "emerge", "reach"],
    "use": ["utilize", "employ", "apply", "operate", "exercise"],
    "city": ["town", "metropolis", "municipality", "urban area", "settlement"],
    "country": ["nation", "state", "land", "territory", "republic"],
    "world": ["globe", "earth", "planet", "realm", "domain"],
    "people": ["individuals", "persons", "humans", "citizens", "population"],
    "work": ["labor", "effort", "task", "job", "employment"],
    "water": ["liquid", "fluid", "aqua", "moisture", "H2O"],
    "house": ["home", "dwelling", "residence", "building", "abode"],
}

# Negation words for injection
NEGATION_WORDS = ["not", "never", "no longer", "hardly", "barely", "scarcely"]

# Domain-specific sentence fragments for topic splicing
DOMAIN_FRAGMENTS = {
    "medical": [
        "According to recent clinical trials,",
        "The patient's symptoms indicated that",
        "Medical research suggests that",
        "In terms of therapeutic outcomes,",
        "From a pharmacological perspective,",
    ],
    "legal": [
        "Under the provisions of the statute,",
        "The court's ruling established that",
        "Legal precedent suggests that",
        "In accordance with regulatory requirements,",
        "The contractual obligation implies that",
    ],
    "financial": [
        "Market analysis indicates that",
        "From a fiscal perspective,",
        "The quarterly earnings report shows that",
        "Investment strategies suggest that",
        "Economic indicators reveal that",
    ],
    "technology": [
        "The latest benchmark results show that",
        "In terms of computational efficiency,",
        "The system architecture requires that",
        "According to the technical specification,",
        "Performance metrics indicate that",
    ],
    "culinary": [
        "The recipe calls for",
        "Culinary experts recommend that",
        "For optimal flavor profile,",
        "The cooking technique requires that",
        "Nutritional analysis shows that",
    ],
}


def synonym_replace(text, strength=0.3) -> str:
    """Replace words with their synonyms based on a lightweight dictionary.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of replacing each eligible word.

    Returns:
        Text with some words replaced by synonyms.
    """
    words = text.split()
    if not words:
        return text

    result = []
    for word in words:
        if not word:
            result.append(word)
            continue
        lower_word = word.lower().strip(".,!?;:'\"")
        if lower_word in SYNONYM_MAP and random.random() < strength * 0.5:
            synonym = random.choice(SYNONYM_MAP[lower_word])
            # Preserve original capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()
            # Preserve trailing punctuation
            trailing = ""
            for ch in reversed(word):
                if ch in ".,!?;:'\"":
                    trailing = ch + trailing
                else:
                    break
            result.append(synonym + trailing)
        else:
            result.append(word)

    return " ".join(result)


def negation_inject(text, strength=0.3) -> str:
    """Inject negation words into sentences to flip their meaning.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of injecting negation.

    Returns:
        Text with negation words injected into some sentences.
    """
    sentences = text.split(". ")
    if not sentences:
        return text

    result = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) < 3 or random.random() > strength * 0.4:
            result.append(sentence)
            continue

        # Find verb-like positions (words after common auxiliaries)
        verb_indicators = {
            "is", "are", "was", "were", "has",
            "have", "will", "can", "do", "does", "did",
        }
        inserted = False
        new_words = []
        for i, word in enumerate(words):
            new_words.append(word)
            if (
                not inserted
                and word.lower() in verb_indicators
                and i + 1 < len(words)
                and words[i + 1].lower() not in {"not", "never", "no"}
            ):
                negation = random.choice(NEGATION_WORDS)
                new_words.append(negation)
                inserted = True

        if not inserted and len(words) > 2:
            # Fallback: insert negation after the second word
            insert_pos = min(2, len(new_words))
            negation = random.choice(NEGATION_WORDS)
            new_words.insert(insert_pos, negation)

        result.append(" ".join(new_words))

    return ". ".join(result)


def topic_splice(text, strength=0.3, domains=None) -> str:
    """Splice in sentences from unrelated domains to create out-of-distribution noise.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability and amount of splicing.
        domains: Optional list of domain names from DOMAIN_FRAGMENTS to use.

    Returns:
        Text with out-of-domain sentence fragments injected.
    """
    if not text or not text.strip():
        return text

    available_domains = domains or list(DOMAIN_FRAGMENTS.keys())
    sentences = text.split(". ")

    if len(sentences) < 2:
        return text

    result = []
    for sentence in sentences:
        result.append(sentence)
        if random.random() < strength * 0.3:
            domain = random.choice(available_domains)
            fragment = random.choice(DOMAIN_FRAGMENTS[domain])
            # Append a partial sentence from another domain
            result.append(fragment + " this is particularly relevant")

    return ". ".join(result)


def entity_swap(text, strength=0.3) -> str:
    """Swap named entities or key nouns between sentences.

    A lightweight approximation: swaps capitalized words between sentences.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability of performing a swap.

    Returns:
        Text with some capitalized words (likely entities) swapped between sentences.
    """
    sentences = text.split(". ")
    if len(sentences) < 2:
        return text

    # Collect capitalized words (likely entities) from each sentence
    entity_positions = []
    for s_idx, sentence in enumerate(sentences):
        words = sentence.split()
        for w_idx, word in enumerate(words):
            stripped = word.strip(".,!?;:'\"")
            if stripped and stripped[0].isupper() and w_idx > 0 and len(stripped) > 1:
                entity_positions.append((s_idx, w_idx, stripped))

    if len(entity_positions) < 2:
        return text

    # Randomly swap some entity pairs
    num_swaps = max(1, int(len(entity_positions) * strength * 0.3))
    for _ in range(num_swaps):
        if len(entity_positions) < 2:
            break
        idx1, idx2 = random.sample(range(len(entity_positions)), 2)
        pos1 = entity_positions[idx1]
        pos2 = entity_positions[idx2]

        sent_words1 = sentences[pos1[0]].split()
        sent_words2 = sentences[pos2[0]].split()

        if pos1[1] < len(sent_words1) and pos2[1] < len(sent_words2):
            # Preserve punctuation
            punct1 = ""
            punct2 = ""
            w1 = sent_words1[pos1[1]]
            w2 = sent_words2[pos2[1]]
            for ch in reversed(w1):
                if ch in ".,!?;:'\"":
                    punct1 = ch + punct1
                else:
                    break
            for ch in reversed(w2):
                if ch in ".,!?;:'\"":
                    punct2 = ch + punct2
                else:
                    break
            sent_words1[pos1[1]] = pos2[2] + punct1
            sent_words2[pos2[1]] = pos1[2] + punct2
            sentences[pos1[0]] = " ".join(sent_words1)
            sentences[pos2[0]] = " ".join(sent_words2)

    return ". ".join(sentences)


def apply_semantic_distortions(text, strength=0.3, config=None) -> str:
    """Apply a combination of semantic-level distortions based on config weights.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling overall distortion intensity.
        config: Optional dict mapping distortion names to their application probabilities.

    Returns:
        Semantically distorted text.
    """
    validate_strength(strength)

    if not text or not text.strip():
        return text

    try:
        default_config = {
            "synonym_replace": 0.4,
            "negation_inject": 0.3,
            "topic_splice": 0.3,
        }
        config = config or default_config

        distortion_funcs = {
            "synonym_replace": synonym_replace,
            "negation_inject": negation_inject,
            "topic_splice": topic_splice,
            "entity_swap": entity_swap,
        }

        result = text
        for name, prob in config.items():
            if name in distortion_funcs and random.random() < prob:
                result = distortion_funcs[name](result, strength=strength)

        return result
    except Exception:
        logger.warning("Semantic distortion failed; returning original text", exc_info=True)
        return text
