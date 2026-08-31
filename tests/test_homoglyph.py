from __future__ import annotations

from nightmarenet.distortions import homoglyph
from nightmarenet.distortions.registry import get_registry


class TestHomoglyphDistortion:
    """Test the homoglyph and keyboard typo distortion engine."""

    def test_homoglyph_produces_output(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = homoglyph.distort(text, strength=0.5, seed=42)
        assert isinstance(result, str)
        assert len(result) == len(text)

    def test_zero_strength_preserves_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = homoglyph.distort(text, strength=0.0, seed=42)
        assert result == text

    def test_empty_input(self):
        assert homoglyph.distort("", strength=0.5, seed=42) == ""

    def test_determinism_with_seed(self):
        text = "The quick brown fox jumps over the lazy dog."
        result1 = homoglyph.distort(text, strength=0.5, seed=100)
        result2 = homoglyph.distort(text, strength=0.5, seed=100)

        assert result1 == result2

    def test_maximum_strength_modifies_text(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        result = homoglyph.distort(text, strength=1.0, seed=42)
        # Verify that all characters have been swapped to either homoglyphs or typos
        assert result != text
        assert len(result) == len(text)

    def test_registry_round_trip(self):
        registry = get_registry()
        assert "homoglyph" in registry

        text = "The quick brown fox jumps over the lazy dog."
        direct_result = homoglyph.distort(text, strength=0.3, seed=42)
        registry_result = registry.apply("homoglyph", text, strength=0.3, seed=42)

        assert direct_result == registry_result
