"""Tests for the Pipeline orchestrator and PipelineRunner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nightmarenet.pipeline import Pipeline, PipelineStatus


@pytest.fixture
def minimal_config():
    """Minimal config for pipeline tests."""
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


class TestPipelineStatus:
    """Tests for PipelineStatus enum."""

    def test_status_values(self):
        assert PipelineStatus.IDLE.value == "idle"
        assert PipelineStatus.TRAINING.value == "training"
        assert PipelineStatus.COMPLETE.value == "complete"
        assert PipelineStatus.FAILED.value == "failed"
        assert PipelineStatus.CANCELLED.value == "cancelled"


class TestPipelineInit:
    """Tests for Pipeline initialization."""

    def test_init(self, minimal_config):
        pipe = Pipeline(minimal_config)
        assert pipe.metrics.status == PipelineStatus.IDLE
        assert pipe._context.dataset is None
        assert pipe._context.trainer is None

    def test_init_with_callback(self, minimal_config):
        events = []
        pipe = Pipeline(minimal_config, on_event=lambda e: events.append(e))
        assert pipe.on_event is not None


class TestPipelineIngest:
    """Tests for the ingest stage."""

    def test_ingest_text_content(self, minimal_config):
        """Ingesting raw text should produce a Dataset."""
        pipe = Pipeline(minimal_config)
        content = "\n\n".join(
            [
                f"Paragraph {i}: This sentence is long enough to be valid training data."
                for i in range(30)
            ]
        )
        pipe.ingest(text_content=content)
        assert pipe._context.dataset is not None
        assert len(pipe._context.dataset) >= 10
        assert pipe.metrics.status == PipelineStatus.INGESTING

    def test_ingest_no_source_raises(self, minimal_config):
        """Calling ingest with no source should raise."""
        pipe = Pipeline(minimal_config)
        with pytest.raises(ValueError, match="Provide one of"):
            pipe.ingest()

    def test_ingest_insufficient_data(self, minimal_config):
        """Ingesting too little data should fail."""
        pipe = Pipeline(minimal_config)
        content = "Short text.\n\nAnother short one."
        with pytest.raises(ValueError, match="usable samples"):
            pipe.ingest(text_content=content)


class TestPipelineStageOrder:
    """Tests that stages must be called in order."""

    def test_prepare_before_ingest_raises(self, minimal_config):
        pipe = Pipeline(minimal_config)
        with pytest.raises(RuntimeError, match="Call .ingest"):
            pipe.prepare()

    def test_train_before_prepare_raises(self, minimal_config):
        pipe = Pipeline(minimal_config)
        with pytest.raises(RuntimeError, match="Call .prepare"):
            pipe.train()

    def test_evaluate_before_train_raises(self, minimal_config):
        pipe = Pipeline(minimal_config)
        with pytest.raises(RuntimeError, match="Call .train"):
            pipe.evaluate()


class TestPipelineCancel:
    """Tests for pipeline cancellation."""

    def test_cancel_sets_status(self, minimal_config):
        pipe = Pipeline(minimal_config)
        pipe.cancel()
        assert pipe.metrics.status == PipelineStatus.CANCELLED
        assert pipe._context.cancelled is True


class TestPipelineMetrics:
    """Tests for PipelineMetrics serialization."""

    def test_to_dict(self, minimal_config):
        pipe = Pipeline(minimal_config)
        d = pipe.metrics.to_dict()
        assert d["status"] == "idle"
        assert d["current_cycle"] == 0
        assert d["has_report"] is False
        assert isinstance(d["history"], list)

    def test_ingest_failure_raises_pipeline_phase_error(self, minimal_config, monkeypatch):
        """Unexpected (non-ValueError) ingestion errors should surface as PipelinePhaseError."""
        from nightmarenet.data.ingest import DataIngestor
        from nightmarenet.exceptions import PipelinePhaseError

        def _boom(self, *args, **kwargs):
            raise RuntimeError("simulated ingestion failure")

        monkeypatch.setattr(DataIngestor, "from_text_content", _boom)

        pipe = Pipeline(minimal_config)
        with pytest.raises(PipelinePhaseError) as exc_info:
            pipe.ingest(text_content="placeholder text content for the test")

        assert exc_info.value.phase == "ingest"

    def test_ingest_validation_error_stays_value_error(self, minimal_config):
        """The existing 'no source provided' validation error must stay a plain ValueError."""
        pipe = Pipeline(minimal_config)
        with pytest.raises(ValueError):
            pipe.ingest()


class TestPipelineEventCallback:
    """Tests that events are emitted at each stage."""

    def test_events_on_ingest(self, minimal_config):
        events = []
        pipe = Pipeline(minimal_config, on_event=lambda e: events.append(e))
        content = "\n\n".join(
            [
                f"Paragraph {i}: Long enough text for ingestion testing purposes here."
                for i in range(30)
            ]
        )
        pipe.ingest(text_content=content)
        assert len(events) >= 1
        assert events[0]["status"] == "ingesting"

    def test_event_callback_exception_doesnt_crash(self, minimal_config):
        """If callback raises, pipeline should not crash."""

        def bad_callback(e):
            raise RuntimeError("callback error")

        pipe = Pipeline(minimal_config, on_event=bad_callback)
        content = "\n\n".join(
            [f"Paragraph {i}: This is text number {i} with enough characters." for i in range(30)]
        )
        # Should not raise
        pipe.ingest(text_content=content)


class TestPipelineRunner:
    """Tests for the PipelineRunner class."""

    def test_runner_status(self, minimal_config):
        from nightmarenet.pipeline_runner import PipelineRunner

        pipe = Pipeline(minimal_config)
        runner = PipelineRunner(pipe)
        status = runner.status()
        assert "run_id" in status
        assert status["is_running"] is False

    def test_runner_cancel(self, minimal_config):
        from nightmarenet.pipeline_runner import PipelineRunner

        pipe = Pipeline(minimal_config)
        runner = PipelineRunner(pipe)
        runner.cancel()
        assert pipe._context.cancelled is True

    def test_runner_registry(self, minimal_config):
        from nightmarenet.pipeline_runner import (
            PipelineRunner,
            get_runner,
            register_runner,
        )

        pipe = Pipeline(minimal_config)
        runner = PipelineRunner(pipe)
        rid = register_runner(runner)
        assert get_runner(rid) is runner
        assert get_runner("nonexistent") is None


class TestAdaptiveTermination:
    """Tests adaptive cycle termination."""

    def test_auto_terminate_disabled(self, minimal_config):
        """Adaptive termination should do nothing when disabled."""

        pipe = Pipeline(minimal_config)

        pipe._context.trainer = MagicMock()
        pipe._context.dataset = MagicMock()

        minimal_config["training"]["auto_terminate"] = False

        event = {"cycle": 0}
        with patch("nightmarenet.phases.train.quick_robustness_score") as mock_score:
            from nightmarenet.phases.train import TrainPhase

            TrainPhase()._handle_cycle_end(pipe._context, event)

            mock_score.assert_not_called()
            pipe._context.trainer.request_stop.assert_not_called()

    def test_convergence_patience_respected(self, minimal_config):
        """Training should stop only after patience consecutive small deltas."""

        minimal_config["training"]["auto_terminate"] = True
        minimal_config["training"]["convergence_threshold"] = 0.01
        minimal_config["training"]["convergence_patience"] = 2

        pipe = Pipeline(minimal_config)
        pipe._context.trainer = MagicMock()
        pipe._context.trainer.model = MagicMock()
        pipe._context.trainer.tokenizer = MagicMock()
        pipe._context.trainer.device = "cpu"
        pipe._context.dataset = MagicMock()

        with patch("nightmarenet.phases.train.quick_robustness_score") as mock_score:
            mock_score.side_effect = [
                0.80,
                0.805,
                0.809,
            ]

            from nightmarenet.phases.train import TrainPhase

            phase = TrainPhase()
            phase._handle_cycle_end(pipe._context, {"cycle": 0})
            pipe._context.trainer.request_stop.assert_not_called()

            phase._handle_cycle_end(pipe._context, {"cycle": 1})
            pipe._context.trainer.request_stop.assert_not_called()

            phase._handle_cycle_end(pipe._context, {"cycle": 2})
            pipe._context.trainer.request_stop.assert_called_once()

    def test_hard_cap_configuration_preserved(self, minimal_config):
        """num_cycles remains configured when adaptive termination is disabled."""

        pipe = Pipeline(minimal_config)

        assert pipe.config["training"]["num_cycles"] == 1


class TestEvalSplit:
    """Tests for the train/eval data split introduced in prepare()."""

    def _make_dataset(self, n: int = 50):
        """Return a minimal HuggingFace Dataset with `n` text samples."""
        from datasets import Dataset

        return Dataset.from_dict(
            {
                "text": [
                    f"Sample {i}: This is a valid training sentence with enough text."
                    for i in range(n)
                ]
            }
        )

    def test_eval_split_creates_held_out_dataset(self, minimal_config):
        """With a large enough dataset, prepare() should split into train and eval."""
        minimal_config["evaluation"] = {"eval_split_ratio": 0.2, "metrics": ["recall"]}

        pipe = Pipeline(minimal_config)
        pipe._context.dataset = self._make_dataset(50)

        with (
            patch("nightmarenet.phases.prepare.create_generators_from_config") as mock_gen,
            patch("nightmarenet.phases.prepare.Trainer") as mock_trainer_cls,
            patch("nightmarenet.phases.prepare._tokenize_dataset") as mock_tok,
        ):
            mock_dream = MagicMock()
            mock_nightmare = MagicMock()
            mock_dream.generate.return_value = pipe._context.dataset
            mock_nightmare.generate.return_value = pipe._context.dataset
            mock_gen.return_value = (mock_dream, mock_nightmare)

            mock_trainer = MagicMock()
            mock_trainer.model = MagicMock()
            mock_trainer.tokenizer = MagicMock()
            mock_trainer_cls.return_value = mock_trainer

            mock_tok.return_value = MagicMock()

            pipe.prepare()

        assert pipe._context.eval_dataset is not None
        assert len(pipe._context.eval_dataset) == 10
        assert len(pipe._context.dataset) - len(pipe._context.eval_dataset) == 40

    def test_eval_split_small_dataset_fallback(self, minimal_config, caplog):
        """When dataset is too small, prepare() warns and uses full dataset for eval."""
        minimal_config["evaluation"] = {"eval_split_ratio": 0.2, "metrics": ["recall"]}

        pipe = Pipeline(minimal_config)
        pipe._context.dataset = self._make_dataset(20)

        with (
            patch("nightmarenet.phases.prepare.create_generators_from_config") as mock_gen,
            patch("nightmarenet.phases.prepare.Trainer") as mock_trainer_cls,
            patch("nightmarenet.phases.prepare._tokenize_dataset") as mock_tok,
        ):
            mock_dream = MagicMock()
            mock_nightmare = MagicMock()
            mock_dream.generate.return_value = pipe._context.dataset
            mock_nightmare.generate.return_value = pipe._context.dataset
            mock_gen.return_value = (mock_dream, mock_nightmare)

            mock_trainer = MagicMock()
            mock_trainer.model = MagicMock()
            mock_trainer.tokenizer = MagicMock()
            mock_trainer_cls.return_value = mock_trainer

            mock_tok.return_value = MagicMock()

            import logging as _logging

            nm_logger = _logging.getLogger("nightmarenet")
            old_propagate = nm_logger.propagate
            nm_logger.propagate = True
            try:
                with caplog.at_level("WARNING", logger="nightmarenet.phases.prepare"):
                    pipe.prepare()
            finally:
                nm_logger.propagate = old_propagate

            assert any("minimum" in r.getMessage() for r in caplog.records)

        assert pipe._context.eval_dataset is not None
        assert len(pipe._context.eval_dataset) == 20

    def test_eval_split_disabled_when_zero(self, minimal_config):
        """Setting eval_split_ratio=0.0 must skip splitting entirely."""
        minimal_config["evaluation"] = {"eval_split_ratio": 0.0, "metrics": ["recall"]}

        pipe = Pipeline(minimal_config)
        pipe._context.dataset = self._make_dataset(50)

        with (
            patch("nightmarenet.phases.prepare.create_generators_from_config") as mock_gen,
            patch("nightmarenet.phases.prepare.Trainer") as mock_trainer_cls,
            patch("nightmarenet.phases.prepare._tokenize_dataset") as mock_tok,
        ):
            mock_dream = MagicMock()
            mock_nightmare = MagicMock()
            mock_dream.generate.return_value = pipe._context.dataset
            mock_nightmare.generate.return_value = pipe._context.dataset
            mock_gen.return_value = (mock_dream, mock_nightmare)

            mock_trainer = MagicMock()
            mock_trainer.model = MagicMock()
            mock_trainer.tokenizer = MagicMock()
            mock_trainer_cls.return_value = mock_trainer

            mock_tok.return_value = MagicMock()

            pipe.prepare()

        assert pipe._context.eval_dataset is not None
        assert len(pipe._context.eval_dataset) == 50

    def test_prepare_sets_distortion_fn(self, minimal_config):
        """prepare() must set _distortion_fn to a non-None callable."""
        minimal_config["evaluation"] = {"eval_split_ratio": 0.2, "metrics": ["recall"]}

        pipe = Pipeline(minimal_config)
        pipe._context.dataset = self._make_dataset(50)

        with (
            patch("nightmarenet.phases.prepare.create_generators_from_config") as mock_gen,
            patch("nightmarenet.phases.prepare.Trainer") as mock_trainer_cls,
            patch("nightmarenet.phases.prepare._tokenize_dataset") as mock_tok,
        ):
            mock_dream = MagicMock()
            mock_nightmare = MagicMock()
            mock_dream.generate.return_value = pipe._context.dataset
            mock_nightmare.generate.return_value = pipe._context.dataset
            mock_gen.return_value = (mock_dream, mock_nightmare)

            mock_trainer = MagicMock()
            mock_trainer.model = MagicMock()
            mock_trainer.tokenizer = MagicMock()
            mock_trainer_cls.return_value = mock_trainer

            mock_tok.return_value = MagicMock()

            pipe.prepare()

        assert callable(pipe._context.distortion_fn)

    def test_evaluate_passes_base_dataset_and_distortion_fn(self, minimal_config):
        """evaluate() must pass base_dataset and distortion_fn to Evaluator.evaluate()."""
        from datasets import Dataset

        pipe = Pipeline(minimal_config)

        mock_trainer = MagicMock()
        mock_trainer.device = "cpu"
        mock_trainer.model = MagicMock()
        mock_trainer.tokenizer = MagicMock()

        pipe._context.trainer = mock_trainer
        pipe._context.baseline_model = MagicMock()
        pipe._context.eval_dl = MagicMock()
        pipe._context.train_dl = MagicMock()
        pipe._context.eval_dataset = Dataset.from_dict({"text": ["sample one", "sample two"]})
        pipe._context.distortion_fn = lambda text, strength, seed=None: text

        fake_results = {
            "label": "test",
            "recall": {
                "token_accuracy": 0.9,
                "perplexity": 1.5,
            },
        }

        with patch("nightmarenet.phases.evaluate.Evaluator") as mock_evaluator_cls:
            mock_eval_instance = MagicMock()
            mock_eval_instance.evaluate.return_value = fake_results
            mock_eval_instance.compare.return_value = {
                "metrics": {},
                "robustness_delta": None,
            }
            mock_eval_instance.generate_report.return_value = "# Report"
            mock_eval_instance.save_results.return_value = None
            mock_evaluator_cls.return_value = mock_eval_instance

            with patch("nightmarenet.phases.evaluate.trigger_webhook"):
                pipe.evaluate()

            for c in mock_eval_instance.evaluate.call_args_list:
                kwargs = c.kwargs
                assert kwargs.get("base_dataset") is pipe._context.eval_dataset
                assert kwargs.get("distortion_fn") is pipe._context.distortion_fn
                assert kwargs.get("clean_dataloader") is pipe._context.eval_dl


class TestPerCycleMetrics:
    """Tests per-cycle evaluation via evaluate_cycle()."""

    def test_per_cycle_metrics_appended(self, minimal_config):
        pipe = Pipeline(minimal_config)
        pipe._context.trainer = MagicMock()
        pipe._context.trainer.model = MagicMock()
        pipe._context.trainer.tokenizer = MagicMock()
        pipe._context.trainer.device = "cpu"
        pipe._context.dataset = MagicMock()
        pipe._context.eval_dl = MagicMock()
        pipe._context.eval_dataset = MagicMock()
        pipe._context.distortion_fn = MagicMock()

        minimal_config["training"]["auto_terminate"] = False

        with patch("nightmarenet.phases.train.evaluate_cycle") as mock_eval_cycle:
            mock_eval_cycle.return_value = {
                "accuracy": 0.85,
                "robustness": {
                    0.3: 0.8,
                    0.5: 0.7,
                    0.7: 0.6,
                },
            }

            from nightmarenet.phases.train import TrainPhase

            event = {"cycle": 0}
            TrainPhase()._handle_cycle_end(pipe._context, event)

            mock_eval_cycle.assert_called_once()
            assert event["cycle_metrics"]["cycle"] == 0
            assert event["cycle_metrics"]["accuracy"] == 0.85

    def test_per_cycle_metrics_skipped_without_eval_dl(self, minimal_config):
        """No eval_dl means the lightweight probe is skipped."""
        pipe = Pipeline(minimal_config)
        pipe._trainer = MagicMock()
        pipe._dataset = MagicMock()

        minimal_config["training"]["auto_terminate"] = False

        from nightmarenet.phases.train import TrainPhase

        TrainPhase()._handle_cycle_end(pipe._context, {"cycle": 0})

        assert pipe.metrics.per_cycle_metrics == []

    def test_per_cycle_metrics_exception_does_not_crash_training(self, minimal_config):
        """Exceptions from evaluate_cycle() should not crash training."""
        pipe = Pipeline(minimal_config)
        pipe._trainer = MagicMock()
        pipe._trainer.model = MagicMock()
        pipe._trainer.tokenizer = MagicMock()
        pipe._trainer.device = "cpu"
        pipe._dataset = MagicMock()
        pipe._eval_dl = MagicMock()
        pipe._eval_dataset = MagicMock()
        pipe._distortion_fn = MagicMock()

        minimal_config["training"]["auto_terminate"] = False

        with patch("nightmarenet.phases.train.evaluate_cycle") as mock_eval_cycle:
            mock_eval_cycle.side_effect = RuntimeError("boom")

            from nightmarenet.phases.train import TrainPhase

            TrainPhase()._handle_cycle_end(pipe._context, {"cycle": 0})

            assert pipe.metrics.per_cycle_metrics == []
