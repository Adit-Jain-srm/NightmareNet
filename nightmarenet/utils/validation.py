"""Centralized input validation utilities for NightmareNet.

Provides reusable validators for common parameter types used across the codebase.
Validators raise descriptive ValueError or TypeError exceptions on failure.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def validate_strength(strength: float, name: str = "strength") -> float:
    """Validate that a distortion strength value is in [0, 1].

    Args:
        strength: The strength value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated strength value.

    Raises:
        TypeError: If strength is not a number.
        ValueError: If strength is outside [0, 1].
    """
    if not isinstance(strength, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(strength).__name__}")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {strength}")
    return float(strength)


def validate_positive_int(value: int, name: str = "value", allow_zero: bool = False) -> int:
    """Validate that a value is a positive integer.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.
        allow_zero: If True, allows zero as a valid value.

    Returns:
        The validated integer value.

    Raises:
        TypeError: If value is not an integer.
        ValueError: If value is not positive (or non-negative when allow_zero=True).
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    else:
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def validate_positive_float(value: float, name: str = "value", allow_zero: bool = False) -> float:
    """Validate that a value is a positive float.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.
        allow_zero: If True, allows zero as a valid value.

    Returns:
        The validated float value.

    Raises:
        TypeError: If value is not a number.
        ValueError: If value is not positive.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    val = float(value)
    if allow_zero:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")
    else:
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")
    return val


def validate_ratio(value: float, name: str = "ratio") -> float:
    """Validate that a value is a valid ratio in [0, 1).

    Args:
        value: The ratio value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated ratio value.

    Raises:
        TypeError: If value is not a number.
        ValueError: If value is outside [0, 1).
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not 0.0 <= float(value) < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")
    return float(value)


def validate_text(text: str, name: str = "text", allow_empty: bool = True) -> str:
    """Validate that a value is a string.

    Args:
        text: The string to validate.
        name: Parameter name for error messages.
        allow_empty: If True, allows empty strings.

    Returns:
        The validated string.

    Raises:
        TypeError: If text is not a string.
        ValueError: If text is empty and allow_empty is False.
    """
    if not isinstance(text, str):
        raise TypeError(f"{name} must be a string, got {type(text).__name__}")
    if not allow_empty and not text.strip():
        raise ValueError(f"{name} must not be empty")
    return text


def validate_dataset_columns(dataset: Any, required_columns: Sequence[str]) -> None:
    """Validate that a dataset contains required columns.

    Args:
        dataset: A HuggingFace Dataset or similar object with column_names.
        required_columns: Column names that must be present.

    Raises:
        ValueError: If any required columns are missing.
        AttributeError: If dataset doesn't have column_names attribute.
    """
    if not hasattr(dataset, "column_names"):
        raise AttributeError(
            f"Dataset object of type {type(dataset).__name__} has no 'column_names' attribute"
        )
    missing = set(required_columns) - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Available columns: {dataset.column_names}"
        )


def validate_non_empty_dataset(dataset: Any, name: str = "dataset") -> None:
    """Validate that a dataset is not empty.

    Args:
        dataset: A dataset with __len__ support.
        name: Name for error messages.

    Raises:
        ValueError: If dataset is empty.
    """
    if len(dataset) == 0:
        raise ValueError(f"{name} is empty (0 samples)")


def validate_config_keys(config: dict, required_keys: Sequence[str], context: str = "config") -> None:
    """Validate that a config dictionary contains required keys.

    Args:
        config: Configuration dictionary.
        required_keys: Keys that must be present.
        context: Context string for error messages.

    Raises:
        TypeError: If config is not a dict.
        ValueError: If any required keys are missing.
    """
    if not isinstance(config, dict):
        raise TypeError(f"{context} must be a dict, got {type(config).__name__}")
    missing = set(required_keys) - set(config.keys())
    if missing:
        raise ValueError(f"{context} missing required keys: {missing}")


def validate_dataloader(dataloader: Any, name: str = "dataloader") -> None:
    """Validate that a dataloader is usable (not None).

    Args:
        dataloader: A DataLoader instance.
        name: Name for error messages.

    Raises:
        ValueError: If dataloader is None.
    """
    if dataloader is None:
        raise ValueError(f"{name} must not be None")
