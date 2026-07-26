"""Load custom distortion engines from file paths.

Supports loading distortion functions from Python files at runtime,
enabling the 'custom:' prefix in YAML configs.
"""

import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

from nightmarenet.distortions.registry import DistortionFn

logger = logging.getLogger(__name__)


def load_from_file(
    file_path: str,
    function_name: str,
    registry,
) -> Optional[DistortionFn]:
    """Load a distortion function from a Python file.

    Args:
        file_path: Path to the Python file containing the distortion function
        function_name: Name of the function to load from the file
        registry: DistortionRegistry instance to register the loaded function

    Returns:
        The loaded distortion function, or None if loading failed
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    if not path.suffix == ".py":
        logger.error(f"File must be a Python file (.py): {file_path}")
        return None

    try:
        # Load the module from file
        # Use full path as module name to avoid sys.modules key collisions
        posix_path = path.as_posix().replace("/", "_").replace(chr(92), "_")
        module_name = f"nightmarenet_custom_{posix_path}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.error(f"Failed to load spec for {file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get the function
        fn = getattr(module, function_name, None)
        if fn is None:
            logger.error(f"Function '{function_name}' not found in {file_path}")
            return None

        if not callable(fn):
            logger.error(f"'{function_name}' is not callable in {file_path}")
            return None

        logger.info(f"Loaded distortion function '{function_name}' from {file_path}")
        return fn

    except Exception as e:
        logger.error(f"Failed to load from {file_path}: {e}")
        return None


def load_custom_engine(
    engine_ref: str,
    registry,
) -> Optional[str]:
    """Load a custom engine from a 'custom:' reference.

    Args:
        engine_ref: Reference in format 'custom:path/to/file.py:function_name'
        registry: DistortionRegistry instance to register the loaded function

    Returns:
        The registered engine name, or None if loading failed
    """
    if not engine_ref.startswith("custom:"):
        return None

    ref = engine_ref[7:]  # Remove 'custom:' prefix

    # Parse path:function_name
    if ":" not in ref:
        logger.error(f"Invalid custom engine reference: {engine_ref}")
        return None

    file_path, function_name = ref.rsplit(":", 1)

    fn = load_from_file(file_path, function_name, registry)
    if fn is None:
        return None

    # Register with a derived name (include file path to avoid collisions)
    # Use a hash of the file path to keep the name manageable
    file_hash = hashlib.md5(file_path.encode(), usedforsecurity=False).hexdigest()[:8]
    engine_name = f"custom_{file_hash}_{function_name}"
    registry.register(
        engine_name,
        fn,
        metadata={
            "phase": "custom",
            "description": f"Custom distortion from {file_path}",
            "source": "custom",
            "file_path": file_path,
        },
    )

    return engine_name
