"""Base class for distortion engines.

Provides the abstract interface that all distortion engines (built-in and plugins)
must implement for consistency and validation.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseDistortion(ABC):
    """Base class for all distortion engines (built-in and plugins).

    Plugin authors should inherit from this class to ensure their distortion
    engines follow the expected contract.
    """

    name: str = ""
    phase: str = "custom"  # dream, nightmare, or custom
    description: str = ""

    @abstractmethod
    def distort(self, text: str, strength: float, seed: Optional[int] = None) -> str:
        """Apply distortion to text at the given strength.

        Args:
            text: Input text to distort
            strength: Float in [0.0, 1.0] controlling distortion intensity
            seed: Optional random seed for reproducibility

        Returns:
            Distorted text

        Contract:
            - strength=0.0 should be approximately a no-op
            - strength=1.0 should produce maximum distortion
            - Same (text, strength, seed) must produce deterministic output
            - Empty input must return empty without raising
        """
        ...

    def validate(self) -> bool:
        """Self-validation: returns True if the engine is properly configured."""
        return bool(self.name)
