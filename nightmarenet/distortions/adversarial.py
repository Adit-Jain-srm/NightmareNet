"""Adversarial prompt construction for nightmare data generation.

Creates stress-test inputs: contradictions, ambiguous queries, cross-domain
prompts, and deliberately misleading context. Each function accepts a
`strength` float (0–1).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Optional

from nightmarenet.distortions.learned import LearnedAdversarialGenerator
from nightmarenet.utils.validation import validate_strength

logger = logging.getLogger(__name__)

# Cache generators so the fallback MLM and per-cycle examples are reused.
_LEARNED_CACHE: dict[tuple[str, str, bool, int], LearnedAdversarialGenerator] = {}


def _get_learned_generator(
    model_name: str,
    strength: float,
    *,
    strategy: str = "attention",
    cache_enabled: bool = True,
    seed: int = 42,
    target_model: Optional[Any] = None,
    target_tokenizer: Optional[Any] = None,
) -> LearnedAdversarialGenerator:
    """Return a cached learned generator for one model/strategy combination."""

    key = (model_name, strategy, cache_enabled, seed)
    generator = _LEARNED_CACHE.get(key)
    if generator is None:
        generator = LearnedAdversarialGenerator(
            model_name=model_name,
            strength=strength,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            strategy=strategy,
            cache_enabled=cache_enabled,
            seed=seed,
        )
        _LEARNED_CACHE[key] = generator
    else:
        generator.strength = strength
        generator.set_target_model(target_model, target_tokenizer)
    return generator


# Templates for contradictory premises
CONTRADICTION_TEMPLATES = [
    "Although {premise}, the opposite is actually true: {negated}.",
    "{premise}. However, this is widely disputed because {negated}.",
    "Despite evidence that {premise}, many experts argue {negated}.",
    "It was once believed that {premise}, but recent findings show {negated}.",
    "While {premise}, there is strong evidence suggesting {negated}.",
]

# Ambiguity injection templates
AMBIGUITY_TEMPLATES = [
    "What exactly do you mean by '{phrase}'? Consider the following: {text}",
    "The term '{phrase}' could refer to many things. {text}",
    "It is unclear whether {text}. This could be interpreted multiple ways.",
    "Depending on context, {text}. Or perhaps the opposite.",
    "Some say {text}, while others interpret this differently.",
]

# Cross-domain framing templates
CROSS_DOMAIN_FRAMES = {
    "legal": [
        "From a legal standpoint, ",
        "Under applicable regulations, ",
        "In the context of contractual law, ",
        "Per the relevant statute, ",
    ],
    "medical": [
        "From a clinical perspective, ",
        "In terms of patient outcomes, ",
        "According to medical literature, ",
        "The diagnostic criteria suggest that ",
    ],
    "financial": [
        "From an investment perspective, ",
        "In terms of market dynamics, ",
        "The financial implications are that ",
        "According to fiscal analysis, ",
    ],
    "philosophical": [
        "Philosophically speaking, ",
        "From an epistemological standpoint, ",
        "In the context of moral reasoning, ",
        "The ontological implications suggest that ",
    ],
    "technical": [
        "From a systems engineering viewpoint, ",
        "In terms of algorithmic complexity, ",
        "The computational requirements imply that ",
        "According to the technical specification, ",
    ],
}

# Misleading context prefixes
MISLEADING_PREFIXES = [
    "According to a well-known study (which was later retracted), ",
    "As everyone knows (though this is actually false), ",
    "It is a well-established fact (actually a common misconception) that ",
    "Research has conclusively shown (in a now-debunked paper) that ",
    "The scientific consensus (which is actually disputed) states that ",
    "Historical records clearly indicate (though they have been misinterpreted) that ",
    "Experts unanimously agree (despite significant disagreement) that ",
    "A comprehensive review confirmed (from a single flawed study) that ",
]


def inject_contradiction(text, strength=0.3) -> str:
    """Inject contradictory premises into the text.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the severity of contradiction.

    Returns:
        Text with contradictory statements injected.
    """
    sentences = text.split(". ")
    if not sentences or len(sentences) < 1:
        return text

    result = []
    for sentence in sentences:
        result.append(sentence)
        if random.random() < strength * 0.4 and len(sentence.split()) > 3:
            # Create a negated version of the sentence
            words = sentence.split()
            negated_words = list(words)

            # Simple negation: insert "not" or flip key words
            negation_targets = {
                "is",
                "are",
                "was",
                "were",
                "will",
                "can",
                "has",
                "have",
                "does",
                "do",
            }
            inserted = False
            for i, w in enumerate(negated_words):
                if w.lower() in negation_targets and i + 1 < len(negated_words):
                    negated_words.insert(i + 1, "not")
                    inserted = True
                    break

            if not inserted and len(negated_words) > 2:
                negated_words.insert(2, "not")

            negated = " ".join(negated_words)
            template = random.choice(CONTRADICTION_TEMPLATES)
            contradiction = template.format(
                premise=sentence.strip().rstrip("."),
                negated=negated.strip().rstrip("."),
            )
            result.append(contradiction)

    return ". ".join(result)


def inject_ambiguity(text, strength=0.3) -> str:
    """Wrap text in ambiguous framing to create under-specified queries.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the degree of ambiguity.

    Returns:
        Text wrapped with ambiguous framing.
    """
    if not text or not text.strip():
        return text

    if random.random() > strength * 0.5:
        return text

    words = text.split()
    # Pick a key phrase (2–3 words) from the text
    if len(words) >= 3:
        start = random.randint(0, max(0, len(words) - 3))
        phrase = " ".join(words[start : start + min(3, len(words) - start)])
    else:
        phrase = text.strip()

    template = random.choice(AMBIGUITY_TEMPLATES)
    return template.format(phrase=phrase, text=text.strip())


def cross_domain_splice(text, strength=0.3, source_domain=None, target_domain=None) -> str:
    """Reframe text through the lens of an unrelated domain.

    E.g., a medical question framed with legal terminology.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the likelihood and intensity of reframing.
        source_domain: Optional source domain to frame from.
        target_domain: Optional target domain to frame to.

    Returns:
        Text reframed through a different domain's language.
    """
    if not text or not text.strip():
        return text

    if random.random() > strength * 0.4:
        return text

    available_domains = list(CROSS_DOMAIN_FRAMES.keys())
    domain = target_domain or random.choice(available_domains)

    prefix = random.choice(CROSS_DOMAIN_FRAMES[domain])

    sentences = text.split(". ")
    if not sentences:
        return text

    # Apply cross-domain framing to some sentences
    result = []
    for sentence in sentences:
        if random.random() < strength * 0.5 and len(sentence.split()) > 2:
            # Lowercase the first character to make the splicing grammatical
            modified = sentence.strip()
            if modified and modified[0].isupper():
                modified = modified[0].lower() + modified[1:]
            result.append(prefix + modified)
        else:
            result.append(sentence)

    return ". ".join(result)


def inject_misleading_context(text, strength=0.3) -> str:
    """Prepend deliberately misleading context to the text.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the probability and amount of misleading context.

    Returns:
        Text with misleading context prepended to some sentences.
    """
    if not text or not text.strip():
        return text

    if random.random() > strength * 0.4:
        return text

    sentences = text.split(". ")
    result = []
    for sentence in sentences:
        if random.random() < strength * 0.35 and len(sentence.split()) > 3:
            prefix = random.choice(MISLEADING_PREFIXES)
            # Lowercase the first character for grammatical flow
            modified = sentence.strip()
            if modified and modified[0].isupper():
                modified = modified[0].lower() + modified[1:]
            result.append(prefix + modified)
        else:
            result.append(sentence)

    return ". ".join(result)


def construct_adversarial_prompt(text, strength=0.3) -> str:
    """Create a structured adversarial prompt with misleading instructions.

    NOTE: This is an opt-in utility not included in the default adversarial
    pipeline. Call directly when constructing custom adversarial challenges.

    Combines multiple adversarial techniques into a single challenging prompt.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling the overall adversarial intensity.

    Returns:
        An adversarial prompt constructed from the input text.
    """
    if not text or not text.strip():
        return text

    components = []

    # Add misleading context
    if random.random() < strength * 0.5:
        components.append(random.choice(MISLEADING_PREFIXES).rstrip() + ":")

    components.append(text.strip())

    # Add contradictory follow-up
    if random.random() < strength * 0.4:
        sentences = text.split(". ")
        if sentences and len(sentences[0].split()) > 3:
            components.append(
                f"Note: The above may be incorrect."
                f" Consider that {sentences[0].strip().lower()}"
                "... or perhaps not."
            )

    # Add ambiguous question
    if random.random() < strength * 0.3:
        components.append("But what does this really mean? Is the opposite equally valid?")

    return " ".join(components)


def apply_adversarial_distortions(
    text,
    strength=0.3,
    config=None,
    *,
    target_model=None,
    target_tokenizer=None,
    cycle_id: int = 0,
) -> str:
    """Apply a combination of adversarial distortions based on config weights.

    Args:
        text: Input text string.
        strength: Float 0–1 controlling overall adversarial intensity.
        config: Optional dict mapping distortion names to their application probabilities.
        target_model: Optional model being trained for gradient-based attacks.
        target_tokenizer: Tokenizer paired with ``target_model``.
        cycle_id: Training cycle used to partition learned-example cache entries.

    Returns:
        Adversarially distorted text.
    """
    validate_strength(strength)

    if not text or not text.strip():
        return text

    try:
        default_config = {
            "contradiction": 0.3,
            "ambiguity": 0.3,
            "cross_domain": 0.2,
            "misleading_context": 0.2,
        }
        config = config or default_config

        distortion_funcs = {
            "contradiction": inject_contradiction,
            "ambiguity": inject_ambiguity,
            "cross_domain": cross_domain_splice,
            "misleading_context": inject_misleading_context,
        }

        result = text
        for name, prob in config.items():
            if name in distortion_funcs and random.random() < prob:
                result = distortion_funcs[name](result, strength=strength)

        # Apply learned adversarial distortion if configured.
        learned_weight = config.get("learned", 0.0)
        if learned_weight > 0 and random.random() < learned_weight:
            try:
                generator = _get_learned_generator(
                    config.get("learned_model", "distilbert-base-uncased"),
                    strength,
                    strategy=config.get("learned_strategy", "attention"),
                    cache_enabled=config.get("learned_cache", True),
                    seed=config.get("learned_seed", 42),
                    target_model=target_model,
                    target_tokenizer=target_tokenizer,
                )
                generator.set_cycle(cycle_id)
                result = generator.generate(
                    result,
                    strength=strength,
                    cycle_id=cycle_id,
                )
            except Exception:
                logger.warning(
                    "Learned adversarial distortion failed; continuing with other distortions",
                    exc_info=True,
                )

        return result
    except Exception:
        logger.warning("Adversarial distortion failed; returning original text", exc_info=True)
        return text
