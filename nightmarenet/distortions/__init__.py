"""Distortion functions for dream and nightmare data generation."""

from nightmarenet.distortions.adversarial import (
    apply_adversarial_distortions as apply_adversarial_distortions,
)
from nightmarenet.distortions.base import BaseDistortion as BaseDistortion
from nightmarenet.distortions.loader import (
    load_custom_engine as load_custom_engine,
)
from nightmarenet.distortions.multilingual import (
    keyboard_typo as keyboard_typo,
)
from nightmarenet.distortions.multilingual import (
    register_layout as register_layout,
)
from nightmarenet.distortions.registry import (
    DistortionRegistry as DistortionRegistry,
)
from nightmarenet.distortions.registry import (
    get_registry as get_registry,
)
from nightmarenet.distortions.semantic import (
    apply_semantic_distortions as apply_semantic_distortions,
)
from nightmarenet.distortions.testing import (
    validate_distortion_function as validate_distortion_function,
)
from nightmarenet.distortions.testing import (
    validate_distortion_plugin as validate_distortion_plugin,
)
from nightmarenet.distortions.text import (
    apply_text_distortions as apply_text_distortions,
)

__all__ = [
    "BaseDistortion",
    "DistortionRegistry",
    "get_registry",
    "validate_distortion_plugin",
    "validate_distortion_function",
    "apply_text_distortions",
    "apply_semantic_distortions",
    "apply_adversarial_distortions",
    "load_custom_engine",
    "keyboard_typo",
    "register_layout",
]
