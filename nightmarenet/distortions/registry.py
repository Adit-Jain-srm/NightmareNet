"""Distortion plugin registry.

Allows registration of custom distortion engines for extensibility.
Built-in engines: dream, nightmare.

Supports:
- Entry point discovery for third-party packages
- Decorator-based registration for single-file plugins
- File-based custom engine loading

Usage:
    from nightmarenet.distortions.registry import DistortionRegistry

    registry = DistortionRegistry()
    registry.register("custom_dream", my_distortion_fn)
    result = registry.apply("custom_dream", text, strength=0.5)
"""

import importlib.metadata
import logging
from typing import Any, Callable, Dict, List, Optional

import torch

from nightmarenet.distortions.base import BaseDistortion

DistortionFn = Callable[[str, float, Optional[int]], str]
VisionDistortionFn = Callable[[torch.Tensor, float, Optional[int]], torch.Tensor]

logger = logging.getLogger(__name__)


class DistortionRegistry:
    """Plugin registry for distortion engines.

    Supports registration of custom distortion functions that follow
    the signature: (text: str, strength: float, seed: Optional[int]) -> str
    """

    def __init__(self) -> None:
        self._engines: Dict[str, DistortionFn] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._register_builtins()
        self._discover_plugins()

    def _register_builtins(self) -> None:
        from nightmarenet.distortions import dream as dream_mod
        from nightmarenet.distortions import nightmare as nightmare_mod

        self.register(
            "dream",
            dream_mod.distort,
            metadata={
                "phase": "dream",
                "description": "Mild stochastic augmentation",
                "source": "builtin",
            },
        )
        self.register(
            "nightmare",
            nightmare_mod.distort,
            metadata={
                "phase": "nightmare",
                "description": "Adversarial perturbation",
                "source": "builtin",
            },
        )

    def _discover_plugins(self) -> None:
        """Discover and load third-party distortion plugins via entry points."""
        try:
            eps = importlib.metadata.entry_points(group="nightmarenet.distortions")
        except TypeError:
            # Python < 3.10 compatibility
            try:
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get("nightmarenet.distortions", [])  # type: ignore[attr-defined, assignment]
            except Exception:
                eps = []  # type: ignore[assignment]

        for ep in eps:
            try:
                cls = ep.load()
                instance = cls()
                if isinstance(instance, BaseDistortion) and instance.validate():
                    self.register(
                        instance.name,
                        instance.distort,
                        metadata={
                            "phase": instance.phase,
                            "description": instance.description,
                            "source": "plugin",
                            "package": ep.dist.name
                            if hasattr(ep, "dist") and ep.dist
                            else "unknown",
                        },
                    )
                    logger.info(f"Loaded distortion plugin '{ep.name}' from {ep.value}")
            except Exception as e:
                logger.warning(f"Failed to load distortion plugin '{ep.name}': {e}")

    def register(
        self,
        name: str,
        fn: DistortionFn,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a distortion engine."""
        if not callable(fn):
            raise TypeError(f"Distortion function must be callable, got {type(fn)}")
        self._engines[name] = fn
        self._metadata[name] = metadata or {}

    def unregister(self, name: str) -> None:
        """Remove a registered distortion engine."""
        self._engines.pop(name, None)
        self._metadata.pop(name, None)

    def apply(
        self,
        name: str,
        text: str,
        strength: float = 0.3,
        seed: Optional[int] = None,
    ) -> str:
        """Apply a named distortion to text."""
        if name not in self._engines:
            available = ", ".join(sorted(self._engines.keys()))
            raise KeyError(f"Unknown distortion '{name}'. Available: {available}")
        return self._engines[name](text, strength, seed)

    def list_engines(self) -> List[Dict[str, Any]]:
        """List all registered distortion engines with metadata."""
        return [
            {"name": name, **self._metadata.get(name, {})} for name in sorted(self._engines.keys())
        ]

    def list_engines_by_source(self) -> Dict[str, List[Dict[str, Any]]]:
        """List engines grouped by source (builtin, plugin, custom)."""
        result: Dict[str, List[Dict[str, Any]]] = {"builtin": [], "plugin": [], "custom": []}
        for name in sorted(self._engines.keys()):
            source = self._metadata.get(name, {}).get("source", "custom")
            result.setdefault(source, []).append({"name": name, **self._metadata.get(name, {})})
        return result

    def get_engine_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific engine."""
        return self._metadata.get(name, {})

    def register_decorator(
        self,
        name: str,
        phase: str = "custom",
        description: str = "",
    ):
        """Decorator for registering distortion functions.

        Usage:
            @registry.register_decorator(
                'my_distortion', phase='nightmare', description='My custom distortion'
            )
            def my_distortion(text: str, strength: float, seed: int = None) -> str:
                return text
        """

        def decorator(fn: DistortionFn) -> DistortionFn:
            self.register(
                name,
                fn,
                metadata={
                    "phase": phase,
                    "description": description,
                    "source": "custom",
                },
            )
            return fn

        return decorator

    @property
    def engine_names(self) -> List[str]:
        return sorted(self._engines.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._engines

    def __len__(self) -> int:
        return len(self._engines)


_default_registry: Optional[DistortionRegistry] = None


def get_registry() -> DistortionRegistry:
    """Get the global distortion registry (lazy singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = DistortionRegistry()
    return _default_registry


class VisionDistortionRegistry:
    """Plugin registry for vision distortion engines.

    Supports registration of custom distortion functions that follow
    the signature: (image: torch.Tensor, strength: float, seed: Optional[int]) -> torch.Tensor
    """

    def __init__(self) -> None:
        self._engines: Dict[str, VisionDistortionFn] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._register_builtins()
        self._discover_plugins()

    def _register_builtins(self) -> None:
        from nightmarenet.distortions.vision.dream import (
            ColorJitter,
            GaussianBlur,
            GeometricTransform,
            JPEGCompression,
        )
        from nightmarenet.distortions.vision.gaussian_noise import GaussianNoise
        from nightmarenet.distortions.vision.nightmare import (
            FGSM,
            PGD,
            AdversarialPatch,
            PixelPerturbation,
        )

        noise_engine = GaussianNoise()
        self.register(
            noise_engine.name,
            noise_engine.distort,
            metadata={
                "phase": noise_engine.phase,
                "description": noise_engine.description,
                "source": "builtin",
            },
        )

        for engine in [
            ColorJitter(),
            GeometricTransform(),
            GaussianBlur(),
            JPEGCompression(),
            FGSM(),
            PGD(),
            AdversarialPatch(),
            PixelPerturbation(),
        ]:
            self.register(
                engine.name,
                engine.distort,
                metadata={
                    "phase": engine.phase,
                    "description": engine.description,
                    "source": "builtin",
                },
            )

    def _discover_plugins(self) -> None:
        """Discover and load third-party vision distortion plugins via entry points."""
        try:
            eps = importlib.metadata.entry_points(group="nightmarenet.distortions.vision")
        except TypeError:
            try:
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get("nightmarenet.distortions.vision", [])  # type: ignore[attr-defined, assignment]
            except Exception:
                eps = []  # type: ignore[assignment]

        from nightmarenet.distortions.vision.base import ImageDistortion

        for ep in eps:
            try:
                cls = ep.load()
                instance = cls()
                if isinstance(instance, ImageDistortion) and instance.validate():
                    self.register(
                        instance.name,
                        instance.distort,
                        metadata={
                            "phase": instance.phase,
                            "description": instance.description,
                            "source": "plugin",
                            "package": ep.dist.name
                            if hasattr(ep, "dist") and ep.dist
                            else "unknown",
                        },
                    )
                    logger.info("Loaded vision distortion plugin '%s' from %s", ep.name, ep.value)
            except Exception as e:
                logger.warning("Failed to load vision distortion plugin '%s': %s", ep.name, e)

    def register(
        self,
        name: str,
        fn: VisionDistortionFn,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a vision distortion engine."""
        if not callable(fn):
            raise TypeError(f"Distortion function must be callable, got {type(fn)}")
        self._engines[name] = fn
        self._metadata[name] = metadata or {}

    def unregister(self, name: str) -> None:
        """Remove a registered vision distortion engine."""
        self._engines.pop(name, None)
        self._metadata.pop(name, None)

    def apply(
        self,
        name: str,
        image: torch.Tensor,
        strength: float = 0.3,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply a named vision distortion to an image tensor."""
        if name not in self._engines:
            available = ", ".join(sorted(self._engines.keys()))
            raise KeyError(f"Unknown vision distortion '{name}'. Available: {available}")
        return self._engines[name](image, strength, seed)

    def list_engines(self) -> List[Dict[str, Any]]:
        """List all registered vision distortion engines with metadata."""
        return [
            {"name": name, **self._metadata.get(name, {})} for name in sorted(self._engines.keys())
        ]

    def list_engines_by_source(self) -> Dict[str, List[Dict[str, Any]]]:
        """List engines grouped by source (builtin, plugin, custom)."""
        result: Dict[str, List[Dict[str, Any]]] = {"builtin": [], "plugin": [], "custom": []}
        for name in sorted(self._engines.keys()):
            source = self._metadata.get(name, {}).get("source", "custom")
            result.setdefault(source, []).append({"name": name, **self._metadata.get(name, {})})
        return result

    def get_engine_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific engine."""
        return self._metadata.get(name, {})

    def register_decorator(
        self,
        name: str,
        phase: str = "custom",
        description: str = "",
    ):
        """Decorator for registering vision distortion functions."""

        def decorator(fn: VisionDistortionFn) -> VisionDistortionFn:
            self.register(
                name,
                fn,
                metadata={
                    "phase": phase,
                    "description": description,
                    "source": "custom",
                },
            )
            return fn

        return decorator

    @property
    def engine_names(self) -> List[str]:
        return sorted(self._engines.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._engines

    def __len__(self) -> int:
        return len(self._engines)


_default_vision_registry: Optional[VisionDistortionRegistry] = None


def get_vision_registry() -> VisionDistortionRegistry:
    """Get the global vision distortion registry (lazy singleton)."""
    global _default_vision_registry
    if _default_vision_registry is None:
        _default_vision_registry = VisionDistortionRegistry()
    return _default_vision_registry
