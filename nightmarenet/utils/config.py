"""YAML configuration loading and schema validation for NightmareNet.

Provides validated config loading with defaults, type checking, and
clear error messages for misconfiguration.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default configuration with all supported keys and their default values.
DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": "gpt2",
        "type": "causal_lm",
        "num_labels": 2,
        "max_length": 128,
        "device": "auto",
    },
    "dataset": {
        "name": "wikitext",
        "subset": "wikitext-2-raw-v1",
        "text_column": "text",
        "max_samples": None,
        "streaming": False,
    },
    "training": {
        "wake_epochs": 3,
        "dream_epochs": 2,
        "nightmare_epochs": 1,
        "num_cycles": 3,
        "compression_rounds": 1,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "nightmare_lr_multiplier": 2.0,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "use_amp": False,
        "gradient_checkpointing": False,
        "early_stopping": False,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 1e-4,
        "distributed": False,
        "save_every_phase": True,
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
    },
    "distortion": {
        "dream_strength": 0.25,
        "nightmare_strength": 0.8,
        "strength_schedule": "uniform",
        "schedule_across_cycles": False,
        "strength_min": 0.2,
        "strength_max": 0.9,
        "text": {
            "char_swap": 0.3,
            "char_insert": 0.2,
            "char_delete": 0.2,
            "keyboard_typo": 0.3,
            "word_shuffle": 0.2,
            "token_mask": 0.3,
        },
        "semantic": {
            "synonym_replace": 0.4,
            "negation_inject": 0.3,
            "topic_splice": 0.3,
        },
        "adversarial": {
            "contradiction": 0.3,
            "ambiguity": 0.3,
            "cross_domain": 0.2,
            "misleading_context": 0.2,
            "learned": 0.0,
            "learned_model": "distilbert-base-uncased",
        },
    },
    "compression": {
        "pruning_ratio": 0.2,
        "pruning_method": "magnitude",
        "bottleneck_rank_ratio": 0.5,
        "finetune_after_prune": True,
        "finetune_epochs": 1,
    },
    "evaluation": {
        "metrics": ["recall", "generalization", "robustness", "hallucination"],
        "robustness_strengths": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "output_dir": "results",
        "output_format": "json",
    },
    "seed": 42,
    "tracking": {
        "backend": "none",
        "project": "nightmarenet",
        "log_dir": "logs/runs",
    },
    "notifications": {
        "webhooks": [],
    },
}

# Schema for type validation: maps dotted key paths to (expected_type, min, max, required).
# Max pruning ratio just below 1.0 because MagnitudePruner rejects ratio=1.0 (would prune all).
_MAX_PRUNING_RATIO = 1.0 - 1e-4

_SCHEMA: dict[str, tuple] = {
    "model.name": (str, None, None, True),
    "model.type": (str, None, None, True),
    "model.num_labels": (int, 1, 10000, False),
    "model.max_length": (int, 1, 8192, True),
    "model.device": (str, None, None, True),
    "dataset.name": (str, None, None, True),
    "dataset.text_column": (str, None, None, True),
    "dataset.streaming": (bool, None, None, False),
    "training.wake_epochs": (int, 0, 1000, True),
    "training.dream_epochs": (int, 0, 1000, True),
    "training.nightmare_epochs": (int, 0, 1000, True),
    "training.num_cycles": (int, 1, 1000, True),
    "training.compression_rounds": (int, 0, 1000, True),
    "training.batch_size": (int, 1, 4096, True),
    "training.learning_rate": (float, 1e-10, 1.0, True),
    "training.nightmare_lr_multiplier": (float, 0.1, 100.0, True),
    "training.max_grad_norm": (float, 0.0, 1000.0, True),
    "training.gradient_accumulation_steps": (int, 1, 1024, True),
    "training.use_amp": (bool, None, None, False),
    "training.gradient_checkpointing": (bool, None, None, False),
    "training.early_stopping": (bool, None, None, False),
    "training.early_stopping_patience": (int, 1, 100, False),
    "training.early_stopping_min_delta": (float, 0.0, 1.0, False),
    "training.distributed": (bool, None, None, False),
    "tracking.backend": (str, None, None, False),
    "tracking.project": (str, None, None, False),
    "distortion.dream_strength": (float, 0.0, 1.0, True),
    "distortion.nightmare_strength": (float, 0.0, 1.0, True),
    "distortion.strength_schedule": (str, None, None, False),
    "distortion.schedule_across_cycles": (bool, None, None, False),
    "distortion.strength_min": (float, 0.0, 1.0, False),
    "distortion.strength_max": (float, 0.0, 1.0, False),
    "compression.pruning_ratio": (float, 0.0, _MAX_PRUNING_RATIO, True),
    "compression.bottleneck_rank_ratio": (float, 0.01, 1.0, True),
    "seed": (int, 0, 2**31 - 1, True),
}


def _deep_merge(base: dict, override: dict) -> dict[str, Any]:
    """Deep-merge override dict into base dict, returning a new dict.

    Args:
        base: Base dictionary with defaults.
        override: Dictionary with user overrides.

    Returns:
        Merged dictionary where override values take precedence.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _get_nested(config: dict, dotted_key: str) -> Any:
    """Get a value from a nested dict using dotted key notation.

    Args:
        config: Configuration dictionary.
        dotted_key: Key in dotted notation, e.g. "model.name".

    Returns:
        The value at the specified key path, or None if not found.
    """
    keys = dotted_key.split(".")
    current = config
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return None
        current = current[k]
    return current


def validate_config(config: dict) -> list[str]:
    """Validate a configuration dictionary against the schema.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []
    for dotted_key, (expected_type, min_val, max_val, required) in _SCHEMA.items():
        value = _get_nested(config, dotted_key)
        if value is None:
            if required:
                errors.append(f"Missing required config key: '{dotted_key}'")
            continue

        # Type check (allow int for float fields)
        if expected_type is float and isinstance(value, int):
            value = float(value)
        elif not isinstance(value, expected_type):
            errors.append(
                f"Config '{dotted_key}' must be {expected_type.__name__}, "
                f"got {type(value).__name__}: {value!r}"
            )
            continue

        # Range check for numeric types
        if isinstance(value, (int, float)):
            if min_val is not None and value < min_val:
                errors.append(
                    f"Config '{dotted_key}' must be >= {min_val}, got {value}"
                )
            if max_val is not None and value > max_val:
                errors.append(
                    f"Config '{dotted_key}' must be <= {max_val}, got {value}"
                )

    webhooks = _get_nested(config, "notifications.webhooks")
    if webhooks is not None:
        if not isinstance(webhooks, list):
            errors.append("Config 'notifications.webhooks' must be a list")
        else:
            for i, wh in enumerate(webhooks):
                if not isinstance(wh, dict):
                    errors.append(f"Config 'notifications.webhooks[{i}]' must be a mapping")
                    continue
                if "url" not in wh:
                    errors.append(
                        f"Config 'notifications.webhooks[{i}]' missing required key: 'url'"
                    )
                elif not isinstance(wh["url"], str):
                    errors.append(f"Config 'notifications.webhooks[{i}].url' must be a string")
                if "events" in wh:
                    if not isinstance(wh["events"], list):
                        errors.append(
                            f"Config 'notifications.webhooks[{i}].events' must be a list"
                        )
                    else:
                        for j, ev in enumerate(wh["events"]):
                            if not isinstance(ev, str):
                                errors.append(
                                    f"Config 'notifications.webhooks[{i}].events[{j}]' "
                                    "must be a string"
                                )
                            elif ev not in (
                                "run_complete",
                                "regression_detected",
                                "alert",
                                "deploy",
                            ):
                                errors.append(
                                    f"Config 'notifications.webhooks[{i}].events[{j}]' "
                                    "must be one of ('run_complete', 'regression_detected', "
                                    f"'alert', 'deploy'), got {ev!r}"
                                )

    return errors


def load_config(path: str) -> dict:
    """Load and validate a YAML configuration file.

    Merges user config with defaults so all keys are guaranteed to exist.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated and merged configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config has validation errors.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.info("Loading configuration from %s", path)

    with open(path) as f:
        user_config = yaml.safe_load(f)

    if user_config is None:
        logger.warning("Config file %s is empty, using defaults.", path)
        user_config = {}

    if not isinstance(user_config, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping,"
            f" got {type(user_config).__name__}"
        )

    # Merge with defaults
    config = _deep_merge(DEFAULT_CONFIG, user_config)

    # Validate
    errors = validate_config(config)
    if errors:
        error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    logger.info("Configuration loaded and validated successfully.")
    return config
