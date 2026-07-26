"""Data ingestion phase: load data from urls / file / text / HuggingFace.

Extracted from `Pipeline.ingest()` (nightmarenet/pipeline.py). Behavior
unchanged: exactly one of urls/file_path/text_content/hf_dataset must be
provided, at construction time (this Phase takes its source args in
__init__, same pattern as ExportPhase taking output_dir).
"""

from __future__ import annotations

import logging
from typing import Optional

from nightmarenet.data.ingest import DataIngestor
from nightmarenet.phases.base import Phase, PhaseResult, PipelineContext
from nightmarenet.utils.telemetry import trace_phase

logger = logging.getLogger(__name__)


class IngestPhase(Phase):
    name = "ingest"

    def __init__(
        self,
        *,
        urls: Optional[list[str]] = None,
        file_path: Optional[str] = None,
        text_content: Optional[str] = None,
        hf_dataset: Optional[str] = None,
        hf_subset: Optional[str] = None,
    ) -> None:
        self.urls = urls
        self.file_path = file_path
        self.text_content = text_content
        self.hf_dataset = hf_dataset
        self.hf_subset = hf_subset

    def execute(self, context: PipelineContext) -> PhaseResult:
        dataset_cfg = context.config.get("dataset", {})
        text_column = dataset_cfg.get("text_column", "text")
        max_samples = dataset_cfg.get("max_samples")
        seed = context.config.get("seed", 42)

        ingestor = DataIngestor(
            text_column=text_column,
            max_samples=max_samples,
            seed=seed,
        )

        span_attrs = {
            "source": (
                "urls"
                if self.urls
                else (
                    "file" if self.file_path else ("text" if self.text_content else "huggingface")
                )
            ),
            "dataset.name": dataset_cfg.get("name", ""),
        }
        with trace_phase("ingest", span_attrs):
            try:
                model_type = context.config.get("model", {}).get("type", "")
                if model_type == "image_classification":
                    from nightmarenet.data.loader import load_from_config

                    wrapper = load_from_config(context.config)
                    context.dataset = wrapper.train_data
                    context.eval_dataset = wrapper.test_data
                elif self.urls:
                    context.dataset = ingestor.from_urls(self.urls)
                elif self.file_path:
                    context.dataset = ingestor.from_file(self.file_path)
                elif self.text_content:
                    context.dataset = ingestor.from_text_content(self.text_content)
                elif self.hf_dataset:
                    context.dataset = ingestor.from_huggingface(
                        self.hf_dataset, subset=self.hf_subset
                    )
                else:
                    raise ValueError(
                        "Provide one of: urls, file_path, text_content, or hf_dataset."
                    )
                if context.dataset is not None:
                    logger.info("Ingestion complete: %d samples.", len(context.dataset))
            except ValueError:
                # Bad input (missing source, insufficient samples, etc.) is a
                # validation error the caller should see as ValueError directly,
                # not wrapped as a PhaseResult failure or PipelinePhaseError.
                raise
            except (RuntimeError, OSError) as exc:
                logger.exception("Ingestion failed.")
                return PhaseResult(success=False, phase_name=self.name, error=str(exc))

        return PhaseResult(success=True, phase_name=self.name)
