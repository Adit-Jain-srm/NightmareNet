"""Computer Vision (image-based) distortions."""

from nightmarenet.distortions.vision.base import ImageDistortion
from nightmarenet.distortions.vision.dream import (
    ColorJitter,
    GaussianBlur,
    GeometricTransform,
    JPEGCompression,
)
from nightmarenet.distortions.vision.nightmare import (
    FGSM,
    PGD,
    AdversarialPatch,
    PixelPerturbation,
)

__all__ = [
    "ImageDistortion",
    "ColorJitter",
    "GeometricTransform",
    "GaussianBlur",
    "JPEGCompression",
    "FGSM",
    "PGD",
    "AdversarialPatch",
    "PixelPerturbation",
]
