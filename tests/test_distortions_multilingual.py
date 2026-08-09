"""Tests for Unicode-aware, multi-script text distortions.

Covers the acceptance criteria in issue #310:
- char_swap works correctly on CJK strings (no broken grapheme clusters)
- typo_injection has character maps for Arabic, Cyrillic, and CJK
- homoglyph_substitution uses confusables tables for non-Latin scripts
- RTL text maintains correct base directionality after distortion
- Existing Latin-alphabet behavior is unchanged (see tests/test_distortions.py)
"""

from __future__ import annotations

import random
import unicodedata

import regex

from nightmarenet.distortions import char_maps
from nightmarenet.distortions.text import (
    char_delete,
    char_insert,
    char_swap,
    homoglyph_substitution,
    typo_injection,
)

SEED = 42

# "你好世界，这是一个测试。" (Hello world, this is a test.)
CJK_TEXT = "\u4f60\u597d\u4e16\u754c\uff0c\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u3002"
# "مرحبا بالعالم" (Hello world)
ARABIC_TEXT = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"
# "привет мир" (hello world)
CYRILLIC_TEXT = "\u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"
# "नमस्ते दुनिया" (hello world)
DEVANAGARI_TEXT = "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e"


def _base_direction(text: str) -> str:
    """First-strong-character paragraph direction per Unicode Bidi rule P2/P3."""
    for ch in text:
        bidi = unicodedata.bidirectional(ch)
        if bidi in ("L",):
            return "ltr"
        if bidi in ("AL", "R"):
            return "rtl"
    return "neutral"


class TestGraphemeAwareness:
    """char_swap (and friends) must never split a grapheme cluster."""

    def setup_method(self):
        random.seed(SEED)

    def test_char_swap_cjk_no_broken_clusters(self):
        result = char_swap(CJK_TEXT, strength=1.0)
        original_graphemes = set(regex.findall(r"\X", CJK_TEXT))
        result_graphemes = regex.findall(r"\X", result)
        assert all(g in original_graphemes for g in result_graphemes)
        assert len(result_graphemes) == len(regex.findall(r"\X", CJK_TEXT))

    def test_char_swap_combining_marks_stay_attached(self):
        text = "cafe\u0301 man\u0303ana"  # café mañana, decomposed form
        result = char_swap(text, strength=1.0)
        graphemes = regex.findall(r"\X", result)
        for g in graphemes:
            if len(g) > 1:
                assert unicodedata.combining(g[0]) == 0

    def test_char_swap_devanagari_matras_stay_attached(self):
        result = char_swap(DEVANAGARI_TEXT, strength=1.0)
        original_graphemes = set(regex.findall(r"\X", DEVANAGARI_TEXT))
        result_graphemes = regex.findall(r"\X", result)
        assert all(g in original_graphemes for g in result_graphemes)


class TestScriptDetection:
    def test_detects_cjk(self):
        assert char_maps.detect_script("\u4f60") == "cjk"

    def test_detects_arabic(self):
        assert char_maps.detect_script("\u0645") == "arabic"

    def test_detects_cyrillic(self):
        assert char_maps.detect_script("\u043f") == "cyrillic"

    def test_detects_devanagari(self):
        assert char_maps.detect_script("\u0928") == "devanagari"

    def test_detects_latin_default(self):
        assert char_maps.detect_script("a") == "latin"
        assert char_maps.detect_script("!") == "latin"

    def test_is_rtl(self):
        assert char_maps.is_rtl("\u0645") is True
        assert char_maps.is_rtl("a") is False
        assert char_maps.is_rtl("\u4f60") is False


class TestTypoInjection:
    def setup_method(self):
        random.seed(SEED)

    def test_typo_injection_produces_output_per_script(self):
        for text in (CJK_TEXT, ARABIC_TEXT, CYRILLIC_TEXT, DEVANAGARI_TEXT):
            result = typo_injection(text, strength=1.0)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_typo_injection_changes_text_at_high_strength(self):
        mappable_text = "\u4eba\u65e5\u571f\u5e72\u5200\u738b"  # 人日土干刀王
        changed = False
        for seed in range(20):
            random.seed(seed)
            if typo_injection(mappable_text, strength=1.0) != mappable_text:
                changed = True
                break
        assert changed

    def test_typo_injection_arabic_has_map_coverage(self):
        assert len(char_maps.arabic.TYPO_MAP) > 0

    def test_typo_injection_cyrillic_has_map_coverage(self):
        assert len(char_maps.cyrillic.TYPO_MAP) > 0

    def test_typo_injection_cjk_has_map_coverage(self):
        assert len(char_maps.cjk.TYPO_MAP) > 0

    def test_typo_injection_empty_input(self):
        assert typo_injection("", strength=0.5) == ""

    def test_typo_injection_zero_strength_is_noop(self):
        for text in (CJK_TEXT, ARABIC_TEXT, CYRILLIC_TEXT):
            assert typo_injection(text, strength=0.0) == text

    def test_typo_injection_preserves_grapheme_count(self):
        for text in (CJK_TEXT, ARABIC_TEXT, CYRILLIC_TEXT, DEVANAGARI_TEXT):
            result = typo_injection(text, strength=1.0)
            assert len(regex.findall(r"\X", result)) == len(regex.findall(r"\X", text))


class TestHomoglyphSubstitution:
    def setup_method(self):
        random.seed(SEED)

    def test_homoglyph_substitution_produces_output(self):
        for text in (CJK_TEXT, ARABIC_TEXT, CYRILLIC_TEXT, "apple orange banana"):
            result = homoglyph_substitution(text, strength=1.0)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_homoglyph_substitution_uses_confusables_table(self):
        random.seed(SEED)
        result = homoglyph_substitution("aeiou aeiou aeiou aeiou aeiou", strength=1.0)
        assert result != "aeiou aeiou aeiou aeiou aeiou"

    def test_homoglyph_substitution_empty_input(self):
        assert homoglyph_substitution("", strength=0.5) == ""

    def test_homoglyph_substitution_zero_strength_is_noop(self):
        for text in (CJK_TEXT, ARABIC_TEXT, CYRILLIC_TEXT):
            assert homoglyph_substitution(text, strength=0.0) == text

    def test_homoglyph_substitution_non_latin_tables_exist(self):
        assert len(char_maps.cyrillic.CONFUSABLES) > 0
        assert len(char_maps.arabic.CONFUSABLES) > 0
        assert len(char_maps.cjk.CONFUSABLES) > 0


class TestRTLDirectionality:
    """RTL text must keep its overall (base) direction after distortion."""

    def setup_method(self):
        random.seed(SEED)

    def test_arabic_base_direction_preserved_char_swap(self):
        assert _base_direction(ARABIC_TEXT) == "rtl"
        result = char_swap(ARABIC_TEXT, strength=1.0)
        assert _base_direction(result) == "rtl"

    def test_arabic_base_direction_preserved_typo_injection(self):
        result = typo_injection(ARABIC_TEXT, strength=1.0)
        assert _base_direction(result) == "rtl"

    def test_arabic_base_direction_preserved_homoglyph(self):
        result = homoglyph_substitution(ARABIC_TEXT, strength=1.0)
        assert _base_direction(result) == "rtl"

    def test_arabic_base_direction_preserved_char_insert(self):
        result = char_insert(ARABIC_TEXT, strength=1.0)
        assert _base_direction(result) == "rtl"

    def test_arabic_char_insert_does_not_inject_latin_noise(self):
        result = char_insert(ARABIC_TEXT, strength=1.0)
        inserted_chars = set(result) - set(ARABIC_TEXT)
        assert all(char_maps.detect_script(c) != "latin" for c in inserted_chars if c.isalpha())


class TestRoundtripValidation:
    """Distortions should never crash, drop to empty, or corrupt structure."""

    @staticmethod
    def _roundtrip_checks(text, fn):
        result = fn(text, strength=0.5)
        assert isinstance(result, str)
        assert "\ufffd" not in result
        original_graphemes = regex.findall(r"\X", text)
        result_graphemes = regex.findall(r"\X", result)
        return original_graphemes, result_graphemes

    def test_cjk_roundtrip_all_functions(self):
        for fn in (char_swap, typo_injection, homoglyph_substitution):
            random.seed(SEED)
            orig, result = self._roundtrip_checks(CJK_TEXT, fn)
            assert len(result) == len(orig)

    def test_arabic_roundtrip_all_functions(self):
        for fn in (char_swap, typo_injection, homoglyph_substitution):
            random.seed(SEED)
            orig, result = self._roundtrip_checks(ARABIC_TEXT, fn)
            assert len(result) == len(orig)

    def test_cyrillic_roundtrip_all_functions(self):
        for fn in (char_swap, typo_injection, homoglyph_substitution):
            random.seed(SEED)
            orig, result = self._roundtrip_checks(CYRILLIC_TEXT, fn)
            assert len(result) == len(orig)

    def test_devanagari_roundtrip_all_functions(self):
        for fn in (char_swap, typo_injection, homoglyph_substitution):
            random.seed(SEED)
            orig, result = self._roundtrip_checks(DEVANAGARI_TEXT, fn)
            assert len(result) == len(orig)

    def test_char_delete_never_crashes_non_latin(self):
        for text in (CJK_TEXT, ARABIC_TEXT, CYRILLIC_TEXT, DEVANAGARI_TEXT):
            result = char_delete(text, strength=0.5)
            assert isinstance(result, str)
