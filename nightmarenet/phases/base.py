"""Shared base classes for pipeline phases.

Defines the three things every phase module imports:

- ``PipelineContext``: mutable state passed between phases, replacing the
  ``self._*`` attributes that used to live directly on ``Pipeline``.
- ``PhaseResult``: the outcome of running a single phase.
- ``Phase``: abstract base class every phase implements.

Field names here match what prepare.py, train.py, evaluate.py, and
export.py already reference (context.dataset, context.trainer,
context.trained_results, etc.) - this file is deliberately written to fit
those, not the other way around.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class PipelineContext:
    """Mutable state shared across all phases of a single pipeline run."""

    # Set at construction time
    run_id: str
    config: dict[str, Any]
    tracker: Optional[Any] = None
    distributed: Optional[str] = None
    resume_dir: Optional[str] = None
    cancelled: bool = False

    # Populated by IngestPhase
    dataset: Optional[Any] = None

    # Populated by OptimizePhase
    wake_dataset: Optional[Any] = None
    dream_base: Optional[Any] = None
    nightmare_base: Optional[Any] = None
    adaption_quality: Optional[dict[str, Any]] = None

    # Populated by PreparePhase
    train_dl: Optional[Any] = None
    eval_dl: Optional[Any] = None
    dream_dl: Optional[Any] = None
    nightmare_dl: Optional[Any] = None
    val_dl: Optional[Any] = None
    eval_dataset: Optional[Any] = None
    distortion_fn: Optional[Callable[..., Any]] = None
    callback_manager: Optional[Any] = None
    dream_generator: Optional[Any] = None
    nightmare_generator: Optional[Any] = None
    dream_base_dataset: Optional[Any] = None
    nightmare_base_dataset: Optional[Any] = None
    trainer: Optional[Any] = None
    baseline_model: Optional[Any] = None

    # Populated by TrainPhase
    history: list[Any] = field(default_factory=list)
    last_robustness_score: Optional[float] = None
    convergence_count: int = 0
    final_convergence_delta: Optional[float] = None
    cycles_completed: int = 0

    # Populated by EvaluatePhase
    trained_results: Optional[dict[str, Any]] = None
    baseline_results: Optional[dict[str, Any]] = None
    comparison: Optional[dict[str, Any]] = None
    report_md: Optional[str] = None
    quality_feedback: Optional[dict[str, Any]] = None


@dataclass
class PhaseResult:
    """Outcome of running a single phase."""

    success: bool
    phase_name: str
    error: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)


class Phase(abc.ABC):
    """Base class every pipeline phase implements."""

    name: str = "phase"

    @abc.abstractmethod
    def execute(self, context: PipelineContext) -> PhaseResult:
        """Run this phase, mutating ``context`` in place, and report the result."""
        raise NotImplementedError
