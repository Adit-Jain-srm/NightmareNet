"""Model export phase.

Extracted from `Pipeline.export()` (nightmarenet/pipeline.py, L~810-830).
Behavior unchanged: saves model, tokenizer, and evaluation report (if any)
to the given output directory.
"""

from __future__ import annotations

import logging
import os

from nightmarenet.phases.base import Phase, PhaseResult, PipelineContext

logger = logging.getLogger(__name__)


class ExportPhase(Phase):
    name = "export"

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def execute(self, context: PipelineContext) -> PhaseResult:
        if context.trainer is None:
            return PhaseResult(
                success=False, phase_name=self.name, error="No trained model to export."
            )

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            context.trainer.model.save_pretrained(self.output_dir)
            context.trainer.tokenizer.save_pretrained(self.output_dir)

            if context.report_md:
                report_path = os.path.join(self.output_dir, "evaluation_report.md")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(context.report_md)
        except OSError as exc:
            logger.exception("Export failed.")
            return PhaseResult(success=False, phase_name=self.name, error=str(exc))

        logger.info("Model exported to %s", self.output_dir)
        return PhaseResult(success=True, phase_name=self.name, data={"output_dir": self.output_dir})
