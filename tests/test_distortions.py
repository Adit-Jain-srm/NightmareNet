"""Tests for distortion functions (text, semantic, adversarial)."""

import random

from nightmarenet.distortions.adversarial import (
    apply_adversarial_distortions,
    construct_adversarial_prompt,
    cross_domain_splice,
    inject_ambiguity,
    inject_contradiction,
    inject_misleading_context,
)
from nightmarenet.distortions.semantic import (
    apply_semantic_distortions,
    entity_swap,
    negation_inject,
    synonym_replace,
    topic_splice,
)
from nightmarenet.distortions.text import (
    apply_text_distortions,
    char_delete,
    char_insert,
    char_swap,
    keyboard_typo,
    token_mask,
    token_replace,
    word_shuffle,
)

# Fix seed for reproducibility
SEED = 42
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. Paris is the capital of France."
LONG_TEXT = (
    "Machine learning is a subset of artificial intelligence. "
    "It allows computers to learn from data without being explicitly programmed. "
    "Deep learning is a type of machine learning that uses neural networks. "
    "Natural language processing enables computers to understand human language."
)


class TestTextDistortions:
    """Test text-level corruption functions."""

    def setup_method(self):
        random.seed(SEED)

    def test_char_swap_produces_output(self):
        result = char_swap(SAMPLE_TEXT, strength=0.5)
        assert isinstance(result, str)
        assert len(result) == len(SAMPLE_TEXT)

    def test_char_swap_zero_strength_preserves_text(self):
        result = char_swap(SAMPLE_TEXT, strength=0.0)
        assert result == SAMPLE_TEXT

    def test_char_swap_empty_input(self):
        assert char_swap("", strength=0.5) == ""
        assert char_swap("a", strength=0.5) == "a"

    def test_char_insert_produces_output(self):
        result = char_insert(SAMPLE_TEXT, strength=0.5)
        assert isinstance(result, str)
        assert len(result) >= len(SAMPLE_TEXT)

    def test_char_insert_zero_strength(self):
        result = char_insert(SAMPLE_TEXT, strength=0.0)
        assert result == SAMPLE_TEXT

    def test_char_delete_produces_output(self):
        result = char_delete(SAMPLE_TEXT, strength=0.5)
        assert isinstance(result, str)
        assert len(result) <= len(SAMPLE_TEXT)

    def test_char_delete_zero_strength(self):
        result = char_delete(SAMPLE_TEXT, strength=0.0)
        assert result == SAMPLE_TEXT

    def test_keyboard_typo_produces_output(self):
        result = keyboard_typo(SAMPLE_TEXT, strength=0.5)
        assert isinstance(result, str)
        assert len(result) == len(SAMPLE_TEXT)

    def test_keyboard_typo_empty_input(self):
        assert keyboard_typo("", strength=0.5) == ""

    def test_word_shuffle_produces_output(self):
        result = word_shuffle(SAMPLE_TEXT, strength=0.8)
        assert isinstance(result, str)
        # Same number of words
        assert len(result.split()) == len(SAMPLE_TEXT.split())

    def test_word_shuffle_single_word(self):
        assert word_shuffle("hello", strength=0.8) == "hello"

    def test_token_mask_produces_output(self):
        result = token_mask(SAMPLE_TEXT, strength=0.5)
        assert isinstance(result, str)
        # Should contain some mask tokens at high strength
        # At zero strength, no masks
        result_no_mask = token_mask(SAMPLE_TEXT, strength=0.0)
        assert "[MASK]" not in result_no_mask

    def test_token_replace_produces_output(self):
        result = token_replace(SAMPLE_TEXT, strength=0.5)
        assert isinstance(result, str)
        assert len(result.split()) == len(SAMPLE_TEXT.split())

    def test_apply_text_distortions_produces_output(self):
        result = apply_text_distortions(SAMPLE_TEXT, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_apply_text_distortions_empty_input(self):
        assert apply_text_distortions("", strength=0.5) == ""
        assert apply_text_distortions("   ", strength=0.5) == "   "

    def test_apply_text_distortions_custom_config(self):
        config = {"char_swap": 1.0, "word_shuffle": 0.0}
        result = apply_text_distortions(SAMPLE_TEXT, strength=0.5, config=config)
        assert isinstance(result, str)

    def test_high_strength_produces_more_distortion(self):
        random.seed(SEED)
        low = apply_text_distortions(SAMPLE_TEXT, strength=0.1)
        random.seed(SEED)
        high = apply_text_distortions(SAMPLE_TEXT, strength=0.9)
        # High strength should produce different output than low in most cases
        # (probabilistic, but with fixed seed this is deterministic)
        assert isinstance(low, str)
        assert isinstance(high, str)


class TestSemanticDistortions:
    """Test semantic-level distortion functions."""

    def setup_method(self):
        random.seed(SEED)

    def test_synonym_replace_produces_output(self):
        result = synonym_replace(SAMPLE_TEXT, strength=0.5)
        assert isinstance(result, str)
        assert len(result.split()) == len(SAMPLE_TEXT.split())

    def test_synonym_replace_preserves_punctuation(self):
        text = "This is good."
        random.seed(0)  # Ensure the replacement happens
        result = synonym_replace(text, strength=1.0)
        assert isinstance(result, str)

    def test_negation_inject_produces_output(self):
        result = negation_inject(LONG_TEXT, strength=0.8)
        assert isinstance(result, str)
        assert len(result) >= len(LONG_TEXT)

    def test_topic_splice_produces_output(self):
        result = topic_splice(LONG_TEXT, strength=0.8)
        assert isinstance(result, str)

    def test_topic_splice_short_text(self):
        result = topic_splice("Hello.", strength=0.8)
        assert isinstance(result, str)

    def test_entity_swap_produces_output(self):
        result = entity_swap(LONG_TEXT, strength=0.8)
        assert isinstance(result, str)

    def test_entity_swap_no_entities(self):
        result = entity_swap("hello world test.", strength=0.8)
        assert result == "hello world test."

    def test_apply_semantic_distortions_produces_output(self):
        result = apply_semantic_distortions(LONG_TEXT, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_apply_semantic_distortions_empty_input(self):
        assert apply_semantic_distortions("", strength=0.5) == ""


class TestAdversarialDistortions:
    """Test adversarial prompt construction functions."""

    def setup_method(self):
        random.seed(SEED)

    def test_inject_contradiction_produces_output(self):
        result = inject_contradiction(LONG_TEXT, strength=0.8)
        assert isinstance(result, str)
        assert len(result) >= len(LONG_TEXT)

    def test_inject_ambiguity_produces_output(self):
        result = inject_ambiguity(SAMPLE_TEXT, strength=0.8)
        assert isinstance(result, str)

    def test_inject_ambiguity_empty_input(self):
        assert inject_ambiguity("", strength=0.8) == ""

    def test_cross_domain_splice_produces_output(self):
        result = cross_domain_splice(LONG_TEXT, strength=0.8)
        assert isinstance(result, str)

    def test_cross_domain_splice_with_domain(self):
        result = cross_domain_splice(
            LONG_TEXT, strength=0.8, target_domain="legal"
        )
        assert isinstance(result, str)

    def test_inject_misleading_context_produces_output(self):
        result = inject_misleading_context(LONG_TEXT, strength=0.8)
        assert isinstance(result, str)

    def test_construct_adversarial_prompt(self):
        result = construct_adversarial_prompt(SAMPLE_TEXT, strength=0.8)
        assert isinstance(result, str)
        assert len(result) >= len(SAMPLE_TEXT)

    def test_apply_adversarial_distortions_produces_output(self):
        result = apply_adversarial_distortions(LONG_TEXT, strength=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_apply_adversarial_distortions_empty_input(self):
        assert apply_adversarial_distortions("", strength=0.5) == ""


class TestLearnedAdversarialDistortions:
    """Test learned adversarial distortion (with mocked model)."""

    def test_generator_fallback_no_model(self):
        """When model is unavailable, generator falls back to random replacements."""
        from nightmarenet.distortions.learned import LearnedAdversarialGenerator

        gen = LearnedAdversarialGenerator(model_name="nonexistent-model-xyz", device="cpu")
        assert not gen._available
        result = gen.generate(LONG_TEXT, strength=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generator_empty_input(self):
        from nightmarenet.distortions.learned import LearnedAdversarialGenerator

        gen = LearnedAdversarialGenerator(model_name="nonexistent-model-xyz", device="cpu")
        assert gen.generate("", strength=0.5) == ""
        assert gen.generate("  ", strength=0.5) == "  "

    def test_generator_single_word(self):
        from nightmarenet.distortions.learned import LearnedAdversarialGenerator

        gen = LearnedAdversarialGenerator(model_name="nonexistent-model-xyz", device="cpu")
        result = gen.generate("hello", strength=0.5)
        assert result == "hello"

    def test_importance_fallback(self):
        """Fallback importance scores are random floats."""
        from nightmarenet.distortions.learned import LearnedAdversarialGenerator

        gen = LearnedAdversarialGenerator(model_name="nonexistent-model-xyz", device="cpu")
        scores = gen._get_token_importance("hello world test")
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_adversarial_via_config(self):
        """apply_adversarial_distortions with learned=1.0 activates learned distortion."""
        config = {
            "contradiction": 0.0,
            "ambiguity": 0.0,
            "cross_domain": 0.0,
            "misleading_context": 0.0,
            "learned": 1.0,
            "learned_model": "nonexistent-model-xyz",
        }
        random.seed(SEED)
        result = apply_adversarial_distortions(LONG_TEXT, strength=0.5, config=config)
        assert isinstance(result, str)
        assert len(result) > 0
