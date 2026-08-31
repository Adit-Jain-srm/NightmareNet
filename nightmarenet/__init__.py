"""NightmareNet: Autonomous AI Self-Improvement Platform."""

from __future__ import annotations

from typing import Any

__version__ = "0.5.0"  # x-release-please-version

__all__ = ["Pipeline", "Evaluator", "get_registry", "__version__"]

_LAZY_EXPORTS = frozenset({"Pipeline", "Evaluator", "get_registry"})

_INSTALL_HINT = (
    "Install NightmareNet with its dependencies, e.g. "
    "`pip install nightmarenet` (add extras such as `[api]` as needed)."
)


def __getattr__(name: str) -> Any:
    """Lazy-import public exports so missing deps raise a clear ImportError."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        if name == "Pipeline":
            from nightmarenet.pipeline import Pipeline as _Pipeline

            globals()["Pipeline"] = _Pipeline
            return _Pipeline
        if name == "Evaluator":
            from nightmarenet.evaluation.evaluator import Evaluator as _Evaluator

            globals()["Evaluator"] = _Evaluator
            return _Evaluator
        # name == "get_registry"
        from nightmarenet.distortions.registry import get_registry as _get_registry

        globals()["get_registry"] = _get_registry
        return _get_registry
    except ImportError as exc:
        raise ImportError(
            f"Cannot import {name!r} from nightmarenet: {exc}. {_INSTALL_HINT}"
        ) from exc


def __dir__() -> list[str]:
    return sorted(list(__all__))
