"""Tests for layout-aware multilingual keyboard typo distortion (#698)."""

from __future__ import annotations

import unicodedata

import pytest
import regex

from nightmarenet.distortions.multilingual import (
    distort,
    keyboard_typo,
    list_layouts,
    register_layout,
    unregister_layout,
)
from nightmarenet.distortions.multilingual.keyboard_layouts import (
    LANGUAGE_LAYOUTS,
    default_layout_for_language,
    resolve_layout_name,
)
from nightmarenet.distortions.registry import DistortionRegistry
from nightmarenet.distortions.validators import (
    validate_distortion_contract,
    validate_keyboard_layout,
    validate_language,
)

SAMPLES = {
    "english": "The quick brown fox jumps over the lazy dog",
    "german": "Hallo Welt das ist ein Test",
    "french": "Bonjour le monde ceci est un test",
    "russian": "привет мир это простой тест",
    "hindi": "नमस्ते दुनिया यह एक परीक्षा है",
    "arabic": "مرحبا بالعالم هذا اختبار",
}


@pytest.mark.parametrize("language", list(SAMPLES))
def test_language_produces_output(language: str) -> None:
    text = SAMPLES[language]
    out = keyboard_typo(text, strength=0.6, seed=7, language=language)
    assert isinstance(out, str)
    assert out != ""
    assert out != text
    assert len(out) >= max(1, len(text) - 5)


def test_german_cli_style_call() -> None:
    out = keyboard_typo("Hallo Welt", strength=0.5, seed=42, language="german")
    assert isinstance(out, str)
    assert out != ""


def test_strength_zero_is_noop() -> None:
    text = SAMPLES["english"]
    assert keyboard_typo(text, strength=0.0, seed=1, language="english") == text


def test_empty_input() -> None:
    assert keyboard_typo("", strength=0.8, seed=1, language="french") == ""


def test_deterministic_with_seed() -> None:
    text = SAMPLES["english"]
    a = keyboard_typo(text, strength=0.5, seed=99, language="english")
    b = keyboard_typo(text, strength=0.5, seed=99, language="english")
    assert a == b


def test_language_selects_default_layout() -> None:
    assert default_layout_for_language("german") == "qwertz"
    assert default_layout_for_language("french") == "azerty"
    assert default_layout_for_language("russian") == "cyrillic"
    # null / omitted keyboard_layout follows language
    assert resolve_layout_name("german", None) == "qwertz"
    assert resolve_layout_name("english", None) == "qwerty"


def test_validate_language_and_layout() -> None:
    assert validate_language("german") == []
    assert validate_language("zz") != []
    assert validate_keyboard_layout("qwerty") == []
    assert validate_keyboard_layout("nope") != []


def test_custom_layout_registerable() -> None:
    name = "test_dvorak_698"
    try:
        register_layout(name, ("pyfgcrl", "aoeuidhtns", "qjkxbmwvz"), overwrite=True)
        assert name in list_layouts()
        assert validate_keyboard_layout(name) == []
        out = distort("piano", strength=0.8, seed=3, keyboard_layout=name)
        assert isinstance(out, str)
    finally:
        unregister_layout(name)


def test_cannot_overwrite_builtin_layout() -> None:
    with pytest.raises(ValueError, match="cannot overwrite builtin"):
        register_layout("qwerty", ("abc",), overwrite=True)


def test_uppercase_cyrillic_is_eligible() -> None:
    text = "ПРИВЕТ МИР"
    out = keyboard_typo(text, strength=0.8, seed=3, language="russian")
    assert out != text
    # Distorted letters should stay uppercase when the source cluster was.
    assert any(ch.isupper() for ch in out)


def test_hindi_grapheme_clusters_stay_intact() -> None:
    # Includes matras (vowel signs) and a virama conjunct in नमस्ते
    text = "नमस्ते दुनिया परीक्षा"
    out = keyboard_typo(text, strength=0.9, seed=5, language="hindi")
    assert out != text
    for cluster in regex.findall(r"\X", out):
        if len(cluster) > 1:
            assert unicodedata.combining(cluster[0]) == 0


def test_contract_and_registry() -> None:
    failures = validate_distortion_contract(keyboard_typo)
    assert failures == [], failures

    registry = DistortionRegistry()
    assert "keyboard_typo" in registry
    a = registry.apply("keyboard_typo", SAMPLES["english"], strength=0.4, seed=11)
    b = registry.apply(
        "keyboard_typo",
        SAMPLES["english"],
        strength=0.4,
        seed=11,
        language="english",
    )
    assert a == b
    assert isinstance(a, str)


def test_supported_languages_cover_six() -> None:
    names = {k for k in LANGUAGE_LAYOUTS if len(k) > 2}
    for lang in ("english", "german", "french", "russian", "hindi", "arabic"):
        assert lang in names
