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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from typing import TypeAlias

    VisionDistortionFn = Callable[[_torch.Tensor, float, Optional[int]], _torch.Tensor]
    VisionApplyReturn: TypeAlias = _torch.Tensor
else:
    # Runtime: use Any for flexibility when torch is optional
    from typing import TypeAlias

    VisionDistortionFn = Callable[..., Any]
    VisionApplyReturn: TypeAlias = Any  # type: ignore[assignment]

DistortionFn: TypeAlias = Callable[[str, float, Optional[int]], str]

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
        """Discover and register distortion engines from entry points."""
        # Entry point: nightmarenet.distortions (group name)
        # Each registered entry point should provide a function that returns
        # (name: str, fn: Callable, metadata: dict)
        try:
            entry_points = importlib.metadata.entry_points(group="nightmarenet.distortions")  # type: ignore[assignment]
        except TypeError:
            # Python < 3.10: entry_points() takes no group argument
            eps = importlib.metadata.entry_points()  # type: ignore[assignment]
            entry_points = [ep for ep in eps if ep.group == "nightmarenet.distortions"]
        except AttributeError:
            # Fallback for older Python versions
            entry_points = []  # type: ignore[assignment]

        for ep in entry_points:
            try:
                loader = ep.load
            except AttributeError:
                continue
            try:
                name, fn, metadata = loader()
                self.register(name, fn, metadata)
            except Exception as exc:
                logger.warning(
                    "Failed to load distortion plugin '%s': %s",
                    ep.name,
                    exc,
                )

    def register(
        self,
        name: str,
        fn: Union[DistortionFn, Callable[..., Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a distortion engine.

        Args:
            name: Unique identifier for this distortion engine.
            fn: Distortion function. For text distortions:
                (text: str, strength: float, seed: Optional[int]) -> str
            metadata: Optional dict with additional info (phase, description, source).
        """
        if not callable(fn):
            raise TypeError(f"Distortion engine '{name}' must be callable, got {type(fn)}")

        if name in self._engines:
            logger.warning("Overwriting existing distortion engine '%s'", name)

        self._engines[name] = fn
        self._metadata[name] = metadata or {}
        logger.debug("Registered distortion engine '%s'", name)

    def unregister(self, name: str) -> None:
        """Unregister a distortion engine.

        Args:
            name: The name of the distortion engine to remove.
        """
        if name in self._engines:
            del self._engines[name]
        if name in self._metadata:
            del self._metadata[name]
            logger.debug("Unregistered distortion engine '%s'", name)

    def apply(
        self,
        name: str,
        image: Any,
        strength: float = 0.3,
        seed: Optional[int] = None,
    ) -> Any:
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
        """Get metadata for a specific distortion engine."""
        return self._metadata.get(name, {})

    def __contains__(self, name: str) -> bool:
        """Check if a distortion engine is registered."""
        return name in self._engines


# Global registry instances
_distortion_registry: DistortionRegistry = DistortionRegistry()


def get_registry() -> DistortionRegistry:
    """Get the global distortion registry instance.

    Returns:
        The singleton DistortionRegistry instance.
    """
    return _distortion_registry


class VisionDistortionRegistry:
    """Registry for vision-based distortion engines.

    Manages image-based distortions (color jitter, noise, FGSM, etc.)
    for computer vision models.
    """

    def __init__(self) -> None:
        self._engines: Dict[str, Callable[..., Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in vision distortion engines."""
        try:
            from nightmarenet.distortions.vision import (
                FGSM,
                ColorJitter,
                GaussianBlur,
            )

            # Register wrapper functions that match the expected signature
            def _color_jitter(image: Any, strength: float, seed: Optional[int]) -> Any:
                return ColorJitter().distort(image, strength, seed)

            def _gaussian_noise(image: Any, strength: float, seed: Optional[int]) -> Any:
                return GaussianBlur().distort(image, strength, seed)

            def _fgsm(image: Any, strength: float, seed: Optional[int]) -> Any:
                return FGSM().distort(image, strength, seed)

            self.register("color_jitter", _color_jitter)
            self.register("gaussian_noise", _gaussian_noise)
            self.register("fgsm", _fgsm)
        except ImportError as exc:
            logger.warning("Failed to register vision distortions: %s", exc)

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a vision distortion engine.

        Args:
            name: Unique identifier for this distortion engine.
            fn: Distortion function with signature (image: Tensor, **kwargs) -> Tensor.
            metadata: Optional dict with additional info.
        """
        if name in self._engines:
            logger.warning("Overwriting existing vision distortion engine '%s'", name)

        self._engines[name] = fn
        self._metadata[name] = metadata or {}

    def apply(
        self,
        name: str,
        image: Any,
        strength: float = 0.3,
        seed: Optional[int] = None,
    ) -> Any:
        """Apply a named vision distortion to an image tensor."""
        if name not in self._engines:
            available = ", ".join(sorted(self._engines.keys()))
            raise KeyError(f"Unknown vision distortion '{name}'. Available: {available}")
        return self._engines[name](image, strength, seed)

    @property
    def engine_names(self) -> List[str]:
        """List all registered engine names."""
        return sorted(self._engines.keys())

    def get_engine_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific vision distortion engine."""
        return self._metadata.get(name, {})

    def list_engines(self) -> List[Dict[str, Any]]:
        """List all registered vision distortion engines."""
        return [
            {"name": name, **self._metadata.get(name, {})} for name in sorted(self._engines.keys())
        ]


_vision_registry: VisionDistortionRegistry = VisionDistortionRegistry()


def get_vision_registry() -> VisionDistortionRegistry:
    """Get the global vision distortion registry instance.

    Returns:
        The singleton VisionDistortionRegistry instance.
    """
    return _vision_registry
