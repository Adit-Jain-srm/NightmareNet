"""Optimize phase: Adaption Labs dataset optimization (optional).

Extracted from `Pipeline.optimize()` / `_optimize_generic()` /
`_optimize_per_phase()` (nightmarenet/pipeline.py). Behavior unchanged:
silent no-op if disabled/SDK unavailable/no API key; supports per-phase
brand controls writing to context.wake_dataset / dream_base / nightmare_base.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from nightmarenet.phases.base import Phase, PhaseResult, PipelineContext
from nightmarenet.utils.telemetry import trace_phase

logger = logging.getLogger(__name__)

PHASE_NAMES = ("wake", "dream", "nightmare")


class OptimizePhase(Phase):
    name = "optimize"

    def execute(self, context: PipelineContext) -> PhaseResult:
        adaption_cfg = context.config.get("adaption", {})
        if not adaption_cfg.get("enabled", False):
            return PhaseResult(success=True, phase_name=self.name, data={"skipped": "disabled"})

        try:
            from nightmarenet.data.adaption import Adaption, AdaptionOptimizer
        except ImportError:
            logger.info("adaption SDK not installed; skipping optimization.")
            return PhaseResult(success=True, phase_name=self.name, data={"skipped": "sdk_missing"})

        if Adaption is None:
            logger.info("adaption SDK not available; skipping optimization.")
            return PhaseResult(
                success=True, phase_name=self.name, data={"skipped": "sdk_unavailable"}
            )

        if not os.environ.get("ADAPTION_API_KEY"):
            logger.info("ADAPTION_API_KEY not set; skipping optimization.")
            return PhaseResult(success=True, phase_name=self.name, data={"skipped": "no_api_key"})

        column_mapping = adaption_cfg.get("column_mapping", {})
        max_rows = adaption_cfg.get("max_rows", 5000)

        optimizer = AdaptionOptimizer()

        has_phase_controls = any(
            adaption_cfg.get(f"{phase}_controls", {}).get("enabled", False) for phase in PHASE_NAMES
        )

        with trace_phase("optimize", {"has_phase_controls": str(has_phase_controls)}):
            if adaption_cfg.get("estimate_first", False):
                skip_reason = self._check_estimate(optimizer, context, adaption_cfg, column_mapping)
                if skip_reason is not None:
                    return PhaseResult(
                        success=True, phase_name=self.name, data={"skipped": skip_reason}
                    )

            quality_results: dict[str, Any] = {}

            if has_phase_controls:
                self._optimize_per_phase(
                    optimizer, context, adaption_cfg, column_mapping, max_rows, quality_results
                )
            else:
                self._optimize_generic(
                    optimizer, context, adaption_cfg, column_mapping, max_rows, quality_results
                )

        context.adaption_quality = quality_results or None

        return PhaseResult(success=True, phase_name=self.name, data={"quality": quality_results})

    def _check_estimate(
        self,
        optimizer: Any,
        context: PipelineContext,
        adaption_cfg: dict[str, Any],
        column_mapping: dict[str, Any],
    ) -> Optional[str]:
        """Return a skip reason if the estimated cost exceeds budget, else None."""
        try:
            estimate = optimizer.estimate_cost(context.dataset, column_mapping)
        except Exception:
            logger.warning("Estimate check failed; proceeding anyway.", exc_info=True)
            return None

        if not estimate:
            return None

        max_credits = adaption_cfg.get("max_credits", 100)
        if estimate["credits"] > max_credits:
            logger.warning(
                "Adaption estimated %.1f credits (budget: %.1f). Skipping optimization.",
                estimate["credits"],
                max_credits,
            )
            return "over_budget"

        logger.info(
            "Adaption estimate: %.1f credits, ~%.1f min",
            estimate["credits"],
            estimate["estimated_minutes"],
        )
        return None

    def _optimize_generic(
        self,
        optimizer: Any,
        context: PipelineContext,
        adaption_cfg: dict[str, Any],
        column_mapping: dict[str, Any],
        max_rows: int,
        quality_results: dict[str, Any],
    ) -> None:
        """Single generic optimization pass (backward-compatible)."""
        brand_controls = adaption_cfg.get("brand_controls")
        recipe_specification = adaption_cfg.get("recipe_specification")

        try:
            result = optimizer.optimize_dataset(
                context.dataset,
                column_mapping,
                max_rows=max_rows,
                brand_controls=brand_controls,
                recipe_specification=recipe_specification,
            )
            if result is not None:
                optimized_dataset, quality = result
                context.dataset = optimized_dataset
                quality_results["generic"] = quality
                logger.info("Dataset optimization complete: %s", quality)
            else:
                logger.warning("Adaption optimization returned None; keeping original.")
        except Exception:
            logger.warning("Adaption optimization failed; keeping original.", exc_info=True)

    def _optimize_per_phase(
        self,
        optimizer: Any,
        context: PipelineContext,
        adaption_cfg: dict[str, Any],
        column_mapping: dict[str, Any],
        max_rows: int,
        quality_results: dict[str, Any],
    ) -> None:
        """Separate optimization passes per training phase."""
        for phase_name in PHASE_NAMES:
            phase_cfg = adaption_cfg.get(f"{phase_name}_controls", {})
            if not phase_cfg.get("enabled", False):
                continue

            brand_controls = phase_cfg.get("brand_controls")
            recipe_specification = phase_cfg.get("recipe_specification")

            try:
                result = optimizer.optimize_dataset(
                    context.dataset,
                    column_mapping,
                    max_rows=max_rows,
                    brand_controls=brand_controls,
                    recipe_specification=recipe_specification,
                )
                if result is not None:
                    optimized_dataset, quality = result
                    quality_results[phase_name] = quality

                    if phase_name == "wake":
                        context.wake_dataset = optimized_dataset
                    elif phase_name == "dream":
                        context.dream_base = optimized_dataset
                    elif phase_name == "nightmare":
                        context.nightmare_base = optimized_dataset

                    logger.info("Phase '%s' optimization complete: %s", phase_name, quality)
                else:
                    logger.warning(
                        "Phase '%s' optimization returned None; using original.",
                        phase_name,
                    )
            except Exception:
                logger.warning(
                    "Phase '%s' optimization failed; using original.",
                    phase_name,
                    exc_info=True,
                )
