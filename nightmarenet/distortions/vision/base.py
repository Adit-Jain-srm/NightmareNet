"""Base class for vision (image-based) distortion engines."""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class ImageDistortion(ABC):
    """Base class for all vision distortion engines.

    Plugin authors should inherit from this class to ensure their vision
    distortion engines follow the expected contract.
    """

    name: str = ""
    phase: str = "custom"  # dream, nightmare, or custom
    description: str = ""

    @abstractmethod
    def distort(
        self, image: torch.Tensor, strength: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Apply distortion to an image tensor at the given strength.

        Args:
            image: Input image tensor of shape (C, H, W) normalized to [0, 1]
            strength: Float in [0.0, 1.0] controlling distortion intensity
            seed: Optional random seed for reproducibility

        Returns:
            Distorted image tensor of shape (C, H, W) normalized to [0, 1]

        Contract:
            - strength=0.0 should be a no-op
            - strength=1.0 should produce maximum distortion
            - Same (image, strength, seed) must produce deterministic output
            - Output must be clamped to [0, 1]
        """
        ...

    def validate(self) -> bool:
        """Self-validation: returns True if the engine is properly configured."""
        return bool(self.name)
