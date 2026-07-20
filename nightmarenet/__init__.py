"""NightmareNet: Autonomous AI Self-Improvement Platform."""

__version__ = "0.2.0"

try:
    from nightmarenet.distortions.registry import get_registry as get_registry
except ImportError:
    get_registry = None  # type: ignore[assignment]

try:
    from nightmarenet.evaluation.evaluator import Evaluator as Evaluator
except ImportError:
    Evaluator = None  # type: ignore[assignment]

try:
    from nightmarenet.pipeline import Pipeline as Pipeline
except ImportError:
    Pipeline = None  # type: ignore[assignment]

__all__ = ["Pipeline", "Evaluator", "get_registry", "__version__"]
