"""Test helpers for distortion plugin authors.

Provides utilities for plugin authors to validate their implementations
against the NightmareNet distortion contract.
"""

from typing import List

from nightmarenet.distortions.validators import (
    validate_base_distortion,
    validate_distortion_contract,
)


def validate_distortion_plugin(engine_cls: type) -> List[str]:
    """Run the standard validation suite against a plugin.

    Returns list of failures (empty = valid).
    Checks: determinism, strength_0_noop, empty_input, type_correctness.

    Args:
        engine_cls: The BaseDistortion subclass to validate

    Returns:
        List of validation failure messages (empty if valid)

    Example:
        import logging
        logger = logging.getLogger(__name__)
        from nightmarenet.distortions.testing import validate_distortion_plugin
        from my_plugin import MyDistortion

        failures = validate_distortion_plugin(MyDistortion)
        if failures:
            for f in failures:
                logger.warning("Validation failed: %s", f)
        else:
            logger.info("Plugin is valid!")
    """
    return validate_base_distortion(engine_cls)


def validate_distortion_function(fn) -> List[str]:
    """Validate a standalone distortion function.

    Args:
        fn: A distortion function with signature
            (text: str, strength: float, seed: int = None) -> str

    Returns:
        List of validation failure messages (empty if valid)

    Example:
        import logging
        logger = logging.getLogger(__name__)
        from nightmarenet.distortions.testing import validate_distortion_function

        def my_distortion(text: str, strength: float, seed: int = None) -> str:
            return text

        failures = validate_distortion_function(my_distortion)
        if not failures:
            logger.info("Function is valid!")
    """
    return validate_distortion_contract(fn)
