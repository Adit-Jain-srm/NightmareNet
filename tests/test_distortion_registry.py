"""Tests for distortion plugin registry."""

import pytest

from nightmarenet.distortions.base import BaseDistortion
from nightmarenet.distortions.registry import DistortionRegistry, get_registry


def test_builtin_engines_registered() -> None:
    registry = DistortionRegistry()
    assert "dream" in registry
    assert "nightmare" in registry
    assert len(registry) >= 2


def test_apply_dream_deterministic_with_seed() -> None:
    registry = DistortionRegistry()
    text = "The quick brown fox jumps over the lazy dog."
    a = registry.apply("dream", text, strength=0.3, seed=42)
    b = registry.apply("dream", text, strength=0.3, seed=42)
    assert a == b
    assert isinstance(a, str)
    assert len(a) > 0


def test_unknown_engine_raises() -> None:
    registry = DistortionRegistry()
    with pytest.raises(KeyError, match="Unknown distortion"):
        registry.apply("nonexistent", "hello", strength=0.5)


def test_custom_engine_registration() -> None:
    registry = DistortionRegistry()

    def upper_distort(text: str, strength: float, seed: int = None) -> str:
        return text.upper()

    registry.register("upper", upper_distort)
    assert registry.apply("upper", "hello", strength=1.0) == "HELLO"


def test_get_registry_singleton() -> None:
    a = get_registry()
    b = get_registry()
    assert a is b


def test_decorator_registration() -> None:
    registry = DistortionRegistry()

    @registry.register_decorator("reverse", phase="custom", description="Reverse text")
    def reverse_distort(text: str, strength: float, seed: int = None) -> str:
        return text[::-1]

    assert "reverse" in registry
    assert registry.apply("reverse", "hello", strength=1.0) == "olleh"
    assert registry._metadata["reverse"]["phase"] == "custom"
    assert registry._metadata["reverse"]["description"] == "Reverse text"
    assert registry._metadata["reverse"]["source"] == "custom"


def test_list_engines_by_source() -> None:
    registry = DistortionRegistry()
    engines_by_source = registry.list_engines_by_source()

    assert "builtin" in engines_by_source
    assert "plugin" in engines_by_source
    assert "custom" in engines_by_source

    # Check built-ins are in the right category
    builtin_names = [e["name"] for e in engines_by_source["builtin"]]
    assert "dream" in builtin_names
    assert "nightmare" in builtin_names

    # Check metadata
    for engine in engines_by_source["builtin"]:
        assert engine["source"] == "builtin"


def test_base_distortion_plugin() -> None:
    registry = DistortionRegistry()

    class TestPlugin(BaseDistortion):
        name = "test_plugin"
        phase = "custom"
        description = "Test plugin for validation"

        def distort(self, text: str, strength: float, seed: int = None) -> str:
            return text + " [distorted]"

    plugin = TestPlugin()
    assert plugin.validate() is True

    # Register manually
    registry.register(
        plugin.name,
        plugin.distort,
        metadata={
            "phase": plugin.phase,
            "description": plugin.description,
            "source": "custom",
        },
    )

    assert "test_plugin" in registry
    assert registry.apply("test_plugin", "hello", strength=0.5) == "hello [distorted]"


def test_builtin_metadata_source() -> None:
    registry = DistortionRegistry()

    assert registry._metadata["dream"]["source"] == "builtin"
    assert registry._metadata["nightmare"]["source"] == "builtin"
