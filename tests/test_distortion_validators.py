from __future__ import annotations

import unittest
from typing import Any, Optional
from unittest import mock

from nightmarenet.distortions.base import BaseDistortion
from nightmarenet.distortions.validators import (
    validate_base_distortion,
    validate_distortion_contract,
    validate_plugin_package,
)
from nightmarenet.utils.validation import (
    validate_config_keys,
    validate_strength,
    validate_text,
)


class ValidDummyDistortion(BaseDistortion):
    name = "valid_dummy"
    phase = "custom"
    description = "A valid dummy distortion for testing."

    def distort(self, text: str, strength: float, seed: Optional[int] = None) -> str:
        if not text:
            return ""
        if strength == 0.0:
            return text
        # Deterministic simple transformation
        return f"{text}_distorted_{int(strength * 100)}"

    def validate(self) -> bool:
        return True


class IncompleteDistortionNoName(BaseDistortion):
    name = ""
    phase = "dream"
    description = "Missing name"

    def distort(self, text: str, strength: float, seed: Optional[int] = None) -> str:
        return text

    def validate(self) -> bool:
        return False


class IncompleteDistortionNoDistort(BaseDistortion):  # type: ignore[abstract]
    name = "no_distort"
    phase = "nightmare"
    description = "No distort method"

    def validate(self) -> bool:
        return True


class UninstantiableDistortion(BaseDistortion):
    def __init__(self) -> None:
        super().__init__()
        raise RuntimeError("Initialization failed deliberately")

    def distort(self, text: str, strength: float, seed: Optional[int] = None) -> str:
        return text


class TestDistortionValidators(unittest.TestCase):
    def test_validate_distortion_contract_valid_function(self) -> None:
        """Test validate_distortion_contract passes with a compliant distortion function."""
        engine = ValidDummyDistortion()
        failures = validate_distortion_contract(engine.distort)
        self.assertEqual(failures, [])

    def test_validate_distortion_contract_non_callable(self) -> None:
        """Test validate_distortion_contract fails when passed a non-callable object."""
        failures = validate_distortion_contract("not_a_callable")  # type: ignore[arg-type]
        self.assertEqual(len(failures), 1)
        self.assertIn("Function is not callable", failures[0])

    def test_validate_distortion_contract_empty_input_violation(self) -> None:
        """Test contract catches failure to return empty string for empty input."""

        def bad_distort(text: str, strength: float = 0.5, seed: Optional[int] = None) -> str:
            if text == "":
                return "unexpected_output"
            return text

        failures = validate_distortion_contract(bad_distort)
        self.assertTrue(any("Empty input should return empty string" in f for f in failures))

    def test_validate_distortion_contract_strength_zero_violation(self) -> None:
        """Test validate_distortion_contract catches non-no-op behavior at strength=0.0."""

        def bad_distort(text: str, strength: float = 0.5, seed: Optional[int] = None) -> str:
            if not text:
                return ""
            return text.upper()  # completely alters text even at strength 0.0

        failures = validate_distortion_contract(bad_distort)
        self.assertTrue(any("strength=0.0 should be approximately no-op" in f for f in failures))

    def test_validate_distortion_contract_non_deterministic(self) -> None:
        """Test validate_distortion_contract catches non-deterministic output."""
        counter = {"val": 0}

        def random_distort(text: str, strength: float = 0.5, seed: Optional[int] = None) -> str:
            if not text:
                return ""
            if strength == 0.0:
                return text
            counter["val"] += 1
            return f"{text}_{counter['val']}"

        failures = validate_distortion_contract(random_distort)
        self.assertTrue(any("Non-deterministic" in f for f in failures))

    def test_validate_distortion_contract_non_string_return(self) -> None:
        """Test validate_distortion_contract catches non-string return types."""

        def non_string_distort(text: str, strength: float = 0.5, seed: Optional[int] = None) -> Any:
            if not text:
                return ""
            if strength == 0.0:
                return text
            return 12345

        failures = validate_distortion_contract(non_string_distort)
        self.assertTrue(any("Result should be str" in f or "non-str result" in f for f in failures))

    def test_validate_distortion_contract_exception_handling(self) -> None:
        """Test validate_distortion_contract gracefully captures unhandled exceptions."""

        def exploding_distort(text: str, strength: float = 0.5, seed: Optional[int] = None) -> str:
            raise ValueError("Distortion exploded")

        failures = validate_distortion_contract(exploding_distort)
        self.assertGreater(len(failures), 0)
        self.assertTrue(any("raised exception" in f for f in failures))

    def test_validate_distortion_contract_unicode_and_multilingual(self) -> None:
        """Test validate_distortion_contract works seamlessly with unicode and multilingual text."""
        engine = ValidDummyDistortion()
        multilingual_samples = [
            "こんにちは世界",
            "Привет мир, как дела?",
            "Bonjour le monde! 🌍✨",
            "مرحبا بالعالم",
            "CJK 漢字 + Emoji 🚀 + Math ∑(x^2)",
        ]
        for sample in multilingual_samples:
            failures = validate_distortion_contract(engine.distort, text=sample)
            self.assertEqual(failures, [], f"Failed on sample: {sample}")

    def test_validate_base_distortion_valid_class(self) -> None:
        """Test validate_base_distortion with a completely valid BaseDistortion subclass."""
        failures = validate_base_distortion(ValidDummyDistortion)
        self.assertEqual(failures, [])

    def test_validate_base_distortion_non_subclass(self) -> None:
        """Test validate_base_distortion catches non-BaseDistortion classes."""

        class NotADistortion:
            pass

        failures = validate_base_distortion(NotADistortion)  # type: ignore[arg-type]
        self.assertEqual(failures, ["Class must inherit from BaseDistortion"])

    def test_validate_base_distortion_missing_attributes_and_methods(self) -> None:
        """Test validate_base_distortion catches missing name, phase, and validate attributes."""
        failures = validate_base_distortion(IncompleteDistortionNoName)
        self.assertTrue(any("non-empty 'name'" in f for f in failures))

        # Test class with missing distort method
        class MissingDistort(BaseDistortion):
            name = "missing_distort"
            phase = "custom"
            description = "desc"
            distort = None  # type: ignore[assignment]

            def validate(self) -> bool:
                return True

        failures_distort = validate_base_distortion(MissingDistort)
        self.assertTrue(any("callable 'distort' method" in f for f in failures_distort))

    def test_validate_base_distortion_uninstantiable(self) -> None:
        """Test validate_base_distortion handles instantiation exceptions."""
        failures = validate_base_distortion(UninstantiableDistortion)
        self.assertTrue(any("Failed to instantiate" in f for f in failures))

    def test_validate_plugin_package_metadata_failure(self) -> None:
        """Test validate_plugin_package handles missing package metadata gracefully."""
        failures = validate_plugin_package("non_existent_package_xyz_123")
        self.assertEqual(len(failures), 1)
        self.assertIn("Failed to load package metadata", failures[0])

    def test_validate_plugin_package_entry_points(self) -> None:
        """Test validate_plugin_package checks entry points in distortions group."""
        with (
            mock.patch("importlib.metadata.metadata", return_value={"Name": "dummy_pkg"}),
            mock.patch("importlib.metadata.entry_points") as mock_eps,
        ):
            # No entry points found
            mock_eps.return_value = []
            failures = validate_plugin_package("dummy_pkg")
            self.assertTrue(any("No entry points found" in f for f in failures))

            # Entry point matches package
            mock_ep = mock.Mock()
            mock_ep.dist.name = "dummy_pkg"
            mock_eps.return_value = [mock_ep]
            valid_failures = validate_plugin_package("dummy_pkg")
            self.assertEqual(valid_failures, [])

    def test_validate_strength_parameter_ranges_and_types(self) -> None:
        """Test validate_strength for numeric validity, boundary ranges, and type errors."""
        # Valid boundary and intermediate values
        self.assertEqual(validate_strength(0.0), 0.0)
        self.assertEqual(validate_strength(0.5), 0.5)
        self.assertEqual(validate_strength(1.0), 1.0)
        self.assertEqual(validate_strength(1), 1.0)
        self.assertEqual(validate_strength(0), 0.0)

        # Invalid out-of-range values
        with self.assertRaises(ValueError) as ctx:
            validate_strength(-0.01, name="distortion_strength")
        self.assertIn("distortion_strength must be in [0, 1]", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_strength(1.01, name="distortion_strength")
        self.assertIn("distortion_strength must be in [0, 1]", str(ctx.exception))

        # Invalid types
        with self.assertRaises(TypeError) as ctx:
            validate_strength("0.5", name="distortion_strength")  # type: ignore[arg-type]
        self.assertIn("distortion_strength must be a number, got str", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:
            validate_strength(None, name="strength")  # type: ignore[arg-type]
        self.assertIn("strength must be a number, got NoneType", str(ctx.exception))

    def test_distortion_config_field_validation(self) -> None:
        """Test required vs optional field validation in distortion configurations."""
        valid_config = {
            "type": "typo",
            "strength": 0.5,
            "seed": 42,
            "mode": "keyboard_adjacent",
        }
        # Validate required fields
        validate_config_keys(valid_config, ["type", "strength"], context="distortion config")

        # Missing required field
        invalid_config = {"strength": 0.5}
        with self.assertRaises(ValueError) as ctx:
            validate_config_keys(invalid_config, ["type", "strength"], context="distortion config")
        self.assertIn("distortion config missing required keys", str(ctx.exception))
        self.assertIn("type", str(ctx.exception))

        # Non-dict config
        with self.assertRaises(TypeError) as ctx:
            validate_config_keys("not_a_dict", ["type"], context="distortion config")  # type: ignore[arg-type]
        self.assertIn("distortion config must be a dict, got str", str(ctx.exception))

    def test_composition_validation_chained_distortions(self) -> None:
        """Test composition validation when chaining multiple distortions sequentially."""
        engine1 = ValidDummyDistortion()

        class UpperDistortion(BaseDistortion):
            name = "upper_dummy"
            phase = "custom"
            description = "Upper distortion"

            def distort(self, text: str, strength: float, seed: Optional[int] = None) -> str:
                if not text or strength == 0.0:
                    return text
                return text.upper()

            def validate(self) -> bool:
                return True

        engine2 = UpperDistortion()

        def chained_pipeline(text: str, strength: float, seed: Optional[int] = None) -> str:
            # Validate strength before pipeline execution
            s = validate_strength(strength, name="pipeline_strength")
            t = validate_text(text, name="pipeline_text")
            t1 = engine1.distort(t, s, seed=seed)
            return engine2.distort(t1, s, seed=seed)

        # Chained pipeline satisfies contract
        failures = validate_distortion_contract(chained_pipeline)
        self.assertEqual(failures, [])

        # Chained pipeline handles empty and unicode
        self.assertEqual(chained_pipeline("", 0.5), "")
        self.assertIn("DISTORTED", chained_pipeline("hello", 0.5))


if __name__ == "__main__":
    unittest.main()
