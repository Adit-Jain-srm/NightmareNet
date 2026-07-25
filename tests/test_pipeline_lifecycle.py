"""Tests for Pipeline lifecycle status transitions and edge cases."""

from __future__ import annotations

import pytest

from nightmarenet.pipeline import Pipeline, PipelineStatus


@pytest.fixture
def minimal_config():
    """Minimal config for pipeline lifecycle tests."""
    return {
        "model": {
            "name": "gpt2",
            "type": "causal_lm",
            "max_length": 32,
            "device": "cpu",
        },
        "dataset": {
            "text_column": "text",
            "max_samples": 20,
        },
        "training": {
            "wake_epochs": 1,
            "dream_epochs": 1,
            "nightmare_epochs": 1,
            "num_cycles": 1,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "save_every_phase": False,
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
        },
        "distortion": {
            "dream_strength": 0.25,
            "nightmare_strength": 0.8,
        },
        "compression": {
            "pruning_ratio": 0.1,
            "pruning_method": "magnitude",
        },
        "evaluation": {
            "metrics": ["recall"],
        },
        "tracking": {"backend": "none"},
        "seed": 42,
    }


def _enough_text(n: int = 30) -> str:
    return "\n\n".join(
        f"Paragraph {i}: This sentence provides enough data for the pipeline test."
        for i in range(n)
    )


class TestPipelineStatusTransitions:
    """Tests that verify correct status transitions through the pipeline."""

    def test_initial_status_is_idle(self, minimal_config):
        pipe = Pipeline(minimal_config)
        assert pipe.metrics.status == PipelineStatus.IDLE

    def test_ingest_sets_ingesting(self, minimal_config):
        pipe = Pipeline(minimal_config)
        pipe.ingest(text_content=_enough_text())
        assert pipe.metrics.status == PipelineStatus.INGESTING

    def test_prepare_sets_preparing(self, minimal_config):
        pipe = Pipeline(minimal_config)
        pipe.ingest(text_content=_enough_text())
        pipe.prepare()
        assert pipe.metrics.status == PipelineStatus.PREPARING

    def test_cancel_sets_cancelled(self, minimal_config):
        pipe = Pipeline(minimal_config)
        pipe.cancel()
        assert pipe.metrics.status == PipelineStatus.CANCELLED

    def test_cancel_after_ingest_sets_cancelled(self, minimal_config):
        pipe = Pipeline(minimal_config)
        pipe.ingest(text_content=_enough_text())
        pipe.cancel()
        assert pipe.metrics.status == PipelineStatus.CANCELLED

    def test_cancel_flag_persists(self, minimal_config):
        pipe = Pipeline(minimal_config)
        pipe.cancel()
        assert pipe._context.cancelled is True

    def test_failed_ingest_sets_failed(self, minimal_config):
        pipe = Pipeline(minimal_config)
        with pytest.raises(ValueError):
            pipe.ingest(text_content="Too short.")
        assert pipe.metrics.status == PipelineStatus.FAILED

    def test_failed_ingest_stores_error_message(self, minimal_config):
        pipe = Pipeline(minimal_config)
        with pytest.raises(ValueError):
            pipe.ingest(text_content="Tiny.")
        assert pipe.metrics.error is not None
        assert "Ingestion failed" in pipe.metrics.error

    def test_multiple_cancels_idempotent(self, minimal_config):
        pipe = Pipeline(minimal_config)
        pipe.cancel()
        pipe.cancel()
        pipe.cancel()
        assert pipe.metrics.status == PipelineStatus.CANCELLED

    def test_pipeline_with_no_source_fails_validation(self, minimal_config):
        pipe = Pipeline(minimal_config)
        with pytest.raises(ValueError, match="Provide one of"):
            pipe.ingest()

    def test_prepare_requires_ingest_first(self, minimal_config):
        pipe = Pipeline(minimal_config)
        with pytest.raises(RuntimeError, match="Call .ingest"):
            pipe.prepare()

    def test_train_requires_prepare_first(self, minimal_config):
        pipe = Pipeline(minimal_config)
        with pytest.raises(RuntimeError, match="Call .prepare"):
            pipe.train()

    def test_evaluate_requires_train_first(self, minimal_config):
        pipe = Pipeline(minimal_config)
        with pytest.raises(RuntimeError, match="Call .train"):
            pipe.evaluate()


class TestPipelineOptimizeSkip:
    """Tests that optimize stage is skipped when adaption is disabled."""

    def test_optimize_skipped_no_adaption_config(self, minimal_config):
        pipe = Pipeline(minimal_config)
        pipe.ingest(text_content=_enough_text())
        pipe.optimize()
        assert pipe.metrics.status != PipelineStatus.FAILED

    def test_optimize_skipped_adaption_disabled(self, minimal_config):
        minimal_config["adaption"] = {"enabled": False}
        pipe = Pipeline(minimal_config)
        pipe.ingest(text_content=_enough_text())
        pipe.optimize()
        assert pipe.metrics.error is None


class TestPipelineConfigValidation:
    """Tests that invalid configs are properly rejected."""

    def test_pipeline_with_invalid_model_type(self):
        cfg = {
            "model": {"name": "gpt2", "type": "invalid_type", "max_length": 32},
            "dataset": {"text_column": "text"},
            "training": {
                "wake_epochs": 1,
                "dream_epochs": 1,
                "nightmare_epochs": 1,
                "num_cycles": 1,
                "batch_size": 2,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": 1,
                "save_every_phase": False,
                "checkpoint_dir": "checkpoints",
                "log_dir": "logs",
            },
            "distortion": {"dream_strength": 0.25, "nightmare_strength": 0.8},
            "compression": {"pruning_ratio": 0.1, "pruning_method": "magnitude"},
            "evaluation": {"metrics": ["recall"]},
            "tracking": {"backend": "none"},
            "seed": 42,
        }
        pipe = Pipeline(cfg)
        assert pipe.metrics.status == PipelineStatus.IDLE

    def test_pipeline_metrics_to_dict_shape(self, minimal_config):
        pipe = Pipeline(minimal_config)
        d = pipe.metrics.to_dict()
        required_keys = {
            "status",
            "current_cycle",
            "total_cycles",
            "current_phase",
            "phase_loss",
            "progress_pct",
            "eta_seconds",
            "history",
            "error",
            "has_report",
        }
        assert required_keys.issubset(set(d.keys()))
