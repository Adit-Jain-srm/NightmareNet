"""Tests for distortion plugin system (validators, loader, testing)."""

from pathlib import Path

from nightmarenet.distortions.base import BaseDistortion
from nightmarenet.distortions.loader import load_custom_engine, load_from_file
from nightmarenet.distortions.registry import DistortionRegistry
from nightmarenet.distortions.testing import (
    validate_distortion_function,
    validate_distortion_plugin,
)
from nightmarenet.distortions.validators import (
    validate_base_distortion,
    validate_distortion_contract,
)


def test_validate_distortion_contract_valid() -> None:
    """Test contract validation with a valid function."""

    def valid_distort(text: str, strength: float, seed: int = None) -> str:
        if strength == 0.0:
            return text
        return text.upper()

    failures = validate_distortion_contract(valid_distort)
    assert failures == []


def test_validate_distortion_contract_empty_input() -> None:
    """Test that empty input must return empty."""

    def invalid_distort(text: str, strength: float, seed: int = None) -> str:
        return "not empty"

    failures = validate_distortion_contract(invalid_distort)
    assert len(failures) > 0
    assert any("empty" in f.lower() for f in failures)


def test_validate_distortion_contract_strength_zero() -> None:
    """Test that strength=0.0 should be approximately no-op."""

    def invalid_distort(text: str, strength: float, seed: int = None) -> str:
        return text.upper()  # Always upper, not no-op at strength=0

    failures = validate_distortion_contract(invalid_distort)
    assert len(failures) > 0
    assert any("strength=0.0" in f for f in failures)


def test_validate_distortion_contract_determinism() -> None:
    """Test that same inputs must produce same output."""

    def invalid_distort(text: str, strength: float, seed: int = None) -> str:
        # Use a counter to ensure different outputs
        invalid_distort.counter = getattr(invalid_distort, "counter", 0) + 1
        return text + str(invalid_distort.counter)

    failures = validate_distortion_contract(invalid_distort)
    assert len(failures) > 0
    assert any("determin" in f.lower() for f in failures)


def test_validate_base_distortion_valid() -> None:
    """Test BaseDistortion validation with a valid class."""

    class ValidPlugin(BaseDistortion):
        name = "valid_plugin"
        phase = "custom"
        description = "A valid plugin"

        def distort(self, text: str, strength: float, seed: int = None) -> str:
            return text

    failures = validate_base_distortion(ValidPlugin)
    assert failures == []


def test_validate_base_distortion_missing_name() -> None:
    """Test BaseDistortion validation with missing name."""

    class InvalidPlugin(BaseDistortion):
        name = ""
        phase = "custom"
        description = "Invalid plugin"

        def distort(self, text: str, strength: float, seed: int = None) -> str:
            return text

    failures = validate_base_distortion(InvalidPlugin)
    assert len(failures) > 0
    assert any("name" in f.lower() for f in failures)


def test_validate_base_distortion_not_subclass() -> None:
    """Test BaseDistortion validation with non-subclass."""

    class NotAPlugin:
        pass

    failures = validate_base_distortion(NotAPlugin)
    assert len(failures) > 0
    assert any("inherit" in f.lower() for f in failures)


def test_validate_distortion_function() -> None:
    """Test the testing.py helper for functions."""

    def my_distort(text: str, strength: float, seed: int = None) -> str:
        return text

    failures = validate_distortion_function(my_distort)
    assert failures == []


def test_validate_distortion_plugin() -> None:
    """Test the testing.py helper for BaseDistortion classes."""

    class MyPlugin(BaseDistortion):
        name = "my_plugin"
        phase = "custom"
        description = "My plugin"

        def distort(self, text: str, strength: float, seed: int = None) -> str:
            return text

    failures = validate_distortion_plugin(MyPlugin)
    assert failures == []


def test_load_from_file_not_found(tmp_path: Path) -> None:
    """Test loading from non-existent file."""
    registry = DistortionRegistry()
    result = load_from_file("nonexistent.py", "some_func", registry)
    assert result is None


def test_load_from_file_not_python(tmp_path: Path) -> None:
    """Test loading from non-Python file."""
    registry = DistortionRegistry()
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("not python")

    result = load_from_file(str(txt_file), "some_func", registry)
    assert result is None


def test_load_from_file_function_not_found(tmp_path: Path) -> None:
    """Test loading when function doesn't exist in file."""
    registry = DistortionRegistry()
    py_file = tmp_path / "test.py"
    py_file.write_text("def other_func(): pass")

    result = load_from_file(str(py_file), "missing_func", registry)
    assert result is None


def test_load_from_file_success(tmp_path: Path) -> None:
    """Test successful loading of a function from file."""
    registry = DistortionRegistry()
    py_file = tmp_path / "test.py"
    py_file.write_text("""
def my_distort(text: str, strength: float, seed: int = None) -> str:
    return text.upper()
""")

    result = load_from_file(str(py_file), "my_distort", registry)
    assert result is not None
    assert result("hello", 0.5) == "HELLO"


def test_load_custom_engine_invalid_format() -> None:
    """Test loading custom engine with invalid reference format."""
    registry = DistortionRegistry()
    result = load_custom_engine("invalid_format", registry)
    assert result is None


def test_load_custom_engine_missing_colon(tmp_path: Path) -> None:
    """Test loading custom engine without colon separator."""
    registry = DistortionRegistry()
    result = load_custom_engine("custom:invalid_no_colon", registry)
    assert result is None


def test_load_custom_engine_success(tmp_path: Path) -> None:
    """Test successful loading of custom engine."""
    registry = DistortionRegistry()
    py_file = tmp_path / "custom.py"
    py_file.write_text("""
def custom_distort(text: str, strength: float, seed: int = None) -> str:
    return text[::-1]
""")

    result = load_custom_engine(f"custom:{py_file}:custom_distort", registry)
    assert result is not None
    assert result in registry
    assert result.startswith("custom_")  # New naming pattern with hash
    assert registry.apply(result, "hello", 0.5) == "olleh"
