"""Validation utilities for distortion plugins.

Provides contract validation to ensure plugins meet the expected
behavior and interface requirements.
"""

import logging
from typing import Any, List

from nightmarenet.distortions.base import BaseDistortion

logger = logging.getLogger(__name__)


def validate_distortion_contract(
    fn: Any,
    text: str = "The quick brown fox jumps over the lazy dog.",
) -> List[str]:
    """Validate that a distortion function meets the contract.

    Args:
        fn: The distortion function to validate
        text: Sample text for testing (default: standard pangram)

    Returns:
        List of validation failure messages (empty if valid)
    """
    failures = []

    # Check callable
    if not callable(fn):
        failures.append("Function is not callable")
        return failures

    # Test empty input
    try:
        result = fn("", strength=0.5, seed=42)
        if result != "":
            failures.append("Empty input should return empty string")
    except Exception as e:
        failures.append(f"Empty input raised exception: {e}")

    # Test strength=0.0 (should be approximately no-op)
    try:
        result = fn(text, strength=0.0, seed=42)
        # Allow for minor differences (e.g., whitespace) but require substantial similarity
        if result != text:
            # Check if result is close enough (same length, at most 1 char different)
            if len(result) != len(text) or sum(1 for a, b in zip(result, text) if a != b) > 1:
                failures.append(
                    f"strength=0.0 should be approximately no-op, "
                    f"got: '{result}' (expected: '{text}')"
                )
    except Exception as e:
        failures.append(f"strength=0.0 raised exception: {e}")

    # Test determinism
    try:
        result1 = fn(text, strength=0.5, seed=42)
        result2 = fn(text, strength=0.5, seed=42)
        if result1 != result2:
            failures.append(
                "Non-deterministic: same (text, strength, seed) should produce identical output"
            )
    except Exception as e:
        failures.append(f"Determinism test raised exception: {e}")

    # Test type correctness
    try:
        result = fn(text, strength=0.5, seed=42)
        if not isinstance(result, str):
            failures.append(f"Result should be str, got {type(result)}")
    except Exception as e:
        failures.append(f"Type correctness test raised exception: {e}")

    # Test strength in valid range
    try:
        for strength in [0.0, 0.5, 1.0]:
            result = fn(text, strength=strength, seed=42)
            if not isinstance(result, str):
                failures.append(f"strength={strength} produced non-str result")
    except Exception as e:
        failures.append(f"Strength range test raised exception: {e}")

    return failures


def validate_base_distortion(engine_cls: type) -> List[str]:
    """Validate a BaseDistortion subclass implementation.

    Args:
        engine_cls: The BaseDistortion subclass to validate

    Returns:
        List of validation failure messages (empty if valid)
    """
    failures = []

    # Check inheritance
    if not issubclass(engine_cls, BaseDistortion):
        failures.append("Class must inherit from BaseDistortion")
        return failures

    # Check required attributes
    try:
        instance = engine_cls()
    except Exception as e:
        failures.append(f"Failed to instantiate: {e}")
        return failures

    if not hasattr(instance, "name") or not instance.name:
        failures.append("Class must have a non-empty 'name' attribute")

    if not hasattr(instance, "phase"):
        failures.append("Class must have a 'phase' attribute")

    if not hasattr(instance, "description"):
        failures.append("Class must have a 'description' attribute")

    # Check distort method
    if not hasattr(instance, "distort") or not callable(instance.distort):
        failures.append("Class must have a callable 'distort' method")
    else:
        # Validate the distort method contract
        method_failures = validate_distortion_contract(instance.distort)
        failures.extend(method_failures)

    # Check validate method
    if not hasattr(instance, "validate") or not callable(instance.validate):
        failures.append("Class must have a callable 'validate' method")

    return failures


def validate_plugin_package(
    package_name: str,
) -> List[str]:
    """Validate a plugin package structure.

    Args:
        package_name: Name of the installed package to validate

    Returns:
        List of validation failure messages (empty if valid)
    """
    failures = []

    try:
        import importlib.metadata

        importlib.metadata.metadata(package_name)
    except Exception as e:
        failures.append(f"Failed to load package metadata: {e}")
        return failures

    # Check for entry points
    try:
        eps = importlib.metadata.entry_points(group="nightmarenet.distortions")
        package_eps = [
            ep for ep in eps if hasattr(ep, "dist") and ep.dist and ep.dist.name == package_name
        ]

        if not package_eps:
            failures.append("No entry points found in 'nightmarenet.distortions' group")
    except Exception as e:
        failures.append(f"Failed to check entry points: {e}")

    return failures
