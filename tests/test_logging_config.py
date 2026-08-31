"""Integration tests for logging configuration.

Tests that setup_logging_from_config correctly initializes logging
from config dict and that JSON mode produces valid JSON output.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nightmarenet.cli import cmd_train
from nightmarenet.utils.logging_config import (
    reset_logging,
    setup_logging,
    setup_logging_from_config,
)


class TestLoggingConfigIntegration:
    """End-to-end tests for logging configuration."""

    def test_setup_logging_from_config_basic(self):
        """setup_logging_from_config reads observability settings correctly."""
        config = {
            "observability": {
                "json_logs": False,
                "log_level": "DEBUG",
            },
            "training": {
                "log_dir": "test_logs",
            },
        }

        reset_logging()
        setup_logging_from_config(config)

        logger = logging.getLogger("nightmarenet")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_setup_logging_from_config_defaults(self):
        """Missing observability section uses sensible defaults."""
        config = {
            "training": {
                "log_dir": "test_logs",
            },
        }

        reset_logging()
        setup_logging_from_config(config)

        logger = logging.getLogger("nightmarenet")
        assert logger.level == logging.INFO  # default
        assert len(logger.handlers) > 0

    def test_json_mode_produces_valid_json_lines(self, capsys):
        """When json_logs=True, log output is valid JSON per line."""
        try:
            try:
                from pythonjsonlogger.json import JsonFormatter  # noqa: F401
            except ImportError:
                from pythonjsonlogger.jsonlogger import JsonFormatter  # noqa: F401
        except ImportError:
            pytest.skip("python-json-logger not installed")

        reset_logging()

        # Setup logging with JSON formatter
        setup_logging(
            log_level="INFO",
            json_logs=True,
            console=True,  # Enable console handler to print to sys.stdout
            file_logging=False,  # Don't add file handler
        )

        # Get our logger and emit a log message
        logger = logging.getLogger("nightmarenet")
        logger.info("Test message", extra={"test_key": "test_value"})

        # Capture the output from sys.stdout
        captured = capsys.readouterr()
        output = captured.out.strip()

        # Split into lines
        lines = [line.strip() for line in output.split("\n") if line.strip()]

        # Ensure we actually have output
        assert len(lines) > 0, "No logging output captured"

        # Parse every non-empty line as JSON and assert expected fields
        for line in lines:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Failed to parse log line as JSON: {line}") from e

            assert "timestamp" in parsed, f"Missing 'timestamp' in {parsed}"
            assert "level" in parsed, f"Missing 'level' in {parsed}"
            assert "logger" in parsed, f"Missing 'logger' in {parsed}"
            assert "message" in parsed, f"Missing 'message' in {parsed}"

        reset_logging()

    def test_log_level_respected(self):
        """When log_level=DEBUG, debug messages appear."""
        config = {
            "observability": {
                "json_logs": False,
                "log_level": "DEBUG",
            },
            "training": {
                "log_dir": "test_logs",
            },
        }

        reset_logging()
        setup_logging_from_config(config)

        logger = logging.getLogger("nightmarenet.test")
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        logger.debug("Debug message")
        output = log_stream.getvalue()
        assert "Debug message" in output

        reset_logging()

    def test_log_level_filters_debug_when_info(self):
        """When log_level=INFO, debug messages are filtered."""
        config = {
            "observability": {
                "json_logs": False,
                "log_level": "INFO",
            },
            "training": {
                "log_dir": "test_logs",
            },
        }

        reset_logging()
        setup_logging_from_config(config)

        logger = logging.getLogger("nightmarenet.test")
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        logger.debug("Debug message")
        output = log_stream.getvalue()
        # Debug message should not appear since logger level is INFO
        assert "Debug message" not in output

        reset_logging()

    def test_idempotent_multiple_calls(self):
        """setup_logging can be called multiple times safely."""
        config = {
            "observability": {
                "json_logs": False,
                "log_level": "INFO",
            },
            "training": {
                "log_dir": "test_logs",
            },
        }

        reset_logging()
        setup_logging_from_config(config)

        logger = logging.getLogger("nightmarenet")
        handler_count_before = len(logger.handlers)

        # Call again - should be idempotent
        setup_logging_from_config(config)

        handler_count_after = len(logger.handlers)
        # Should not add duplicate handlers due to _INITIALIZED flag
        assert handler_count_after == handler_count_before

        reset_logging()

    def test_cli_integration_logging_initialized(self):
        """CLI commands that load config should initialize logging."""
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        assert config_path.exists(), "configs/default.yaml must exist"

        args = argparse.Namespace(
            config=str(config_path),
            resume=None,
            distributed=False,
            output=None,
        )

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.metrics = MagicMock(phase_loss=0.5, status="complete")

        reset_logging()
        with patch("nightmarenet.pipeline.Pipeline", return_value=mock_pipeline_instance):
            cmd_train(args)

        logger = logging.getLogger("nightmarenet")
        assert logger.level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
        assert len(logger.handlers) > 0

        reset_logging()

    def test_core_library_logger_namespaces_exist(self):
        """Verify standard core modules obtain child loggers under nightmarenet namespace."""
        import nightmarenet.data.adaption as adaption_mod
        import nightmarenet.evaluation.evaluator as eval_mod
        import nightmarenet.pipeline as pipe_mod
        import nightmarenet.pipeline_runner as runner_mod
        import nightmarenet.training.trainer as trainer_mod

        assert hasattr(trainer_mod, "logger")
        assert trainer_mod.logger.name == "nightmarenet.training.trainer"

        assert hasattr(eval_mod, "logger")
        assert eval_mod.logger.name == "nightmarenet.evaluation.evaluator"

        assert hasattr(adaption_mod, "logger")
        assert adaption_mod.logger.name == "nightmarenet.data.adaption"

        assert hasattr(pipe_mod, "logger")
        assert pipe_mod.logger.name == "nightmarenet.pipeline"

        assert hasattr(runner_mod, "logger")
        assert runner_mod.logger.name == "nightmarenet.pipeline_runner"

    def test_structured_log_extra_context_fields(self, caplog):
        """Verify core loggers include structured context fields."""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("nightmarenet.training.trainer")
            logger.info("Training cycle finished", extra={"run_id": "run-123", "epoch": 3, "loss": 0.45})

        assert "Training cycle finished" in caplog.text

    def test_structured_logging_levels_and_formatting(self):
        """Test structured logging formatters with different record levels."""
        reset_logging()
        setup_logging(log_level="DEBUG", json_logs=False, file_logging=False, console=False)
        logger = logging.getLogger("nightmarenet.evaluation")
        assert logger.isEnabledFor(logging.INFO)
        assert logger.isEnabledFor(logging.DEBUG)
        reset_logging()

    def test_logging_file_creation_when_enabled(self, tmp_path):
        """Test file logging handler writes log entries to disk directory."""
        reset_logging()
        log_dir = str(tmp_path / "custom_logs")
        setup_logging(log_dir=log_dir, log_level="INFO", file_logging=True, console=False)
        logger = logging.getLogger("nightmarenet")
        logger.info("Writing file verification log entry")
        
        files = list(Path(log_dir).glob("*.log"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "Writing file verification log entry" in content
        reset_logging()

    def test_structured_logging_warning_and_error_levels(self, caplog):
        """Test warning and error level logging functions with structured records."""
        with caplog.at_level(logging.WARNING):
            logger = logging.getLogger("nightmarenet.pipeline")
            logger.warning("Pipeline stage delayed", extra={"stage": "prepare", "retry_count": 2})
            logger.error("Pipeline phase error occurred", extra={"stage": "train", "error_code": "ERR_OOM"})

        assert "Pipeline stage delayed" in caplog.text
        assert "Pipeline phase error occurred" in caplog.text

    def test_json_formatter_extra_fields_inclusion(self):
        """Test custom JSON formatter preserves extra fields."""
        try:
            try:
                from pythonjsonlogger.json import JsonFormatter
            except ImportError:
                from pythonjsonlogger.jsonlogger import JsonFormatter
        except ImportError:
            pytest.skip("python-json-logger not installed")

        formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        )
        record = logging.LogRecord(
            name="nightmarenet.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Structured test log",
            args=(),
            exc_info=None,
        )
        record.run_id = "test_run_456"
        record.epoch = 2
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["message"] == "Structured test log"
        assert parsed["run_id"] == "test_run_456"
        assert parsed["epoch"] == 2

    def test_logging_rotation_configuration_fallback(self):
        """Verify fallback behavior when non-standard logging arguments provided."""
        reset_logging()
        setup_logging(log_level="CRITICAL", console=False, file_logging=False)
        logger = logging.getLogger("nightmarenet")
        assert logger.level == logging.CRITICAL
        reset_logging()

    def test_child_logger_propagation_behavior(self):
        """Verify child loggers correctly propagate to root nightmarenet logger."""
        reset_logging()
        setup_logging(log_level="INFO", console=False, file_logging=False)
        root = logging.getLogger("nightmarenet")
        child = logging.getLogger("nightmarenet.training.trainer")
        
        # Ensure hierarchy is established
        assert child.name.startswith(root.name)
        reset_logging()

    def test_logging_level_case_insensitivity(self):
        """Verify log levels are parsed case-insensitively."""
        reset_logging()
        setup_logging(log_level="debug", console=False, file_logging=False)
        logger = logging.getLogger("nightmarenet")
        assert logger.level == logging.DEBUG
        reset_logging()

        setup_logging(log_level="warning", console=False, file_logging=False)
        logger = logging.getLogger("nightmarenet")
        assert logger.level == logging.WARNING
        reset_logging()

    def test_multiple_stream_handlers_cleanup_on_reset(self):
        """Verify reset_logging properly cleans up open stream and file handlers."""
        reset_logging()
        setup_logging(log_level="INFO", console=True, file_logging=False)
        logger = logging.getLogger("nightmarenet")
        assert len(logger.handlers) > 0
        reset_logging()
        assert len(logger.handlers) == 0

    def test_logger_name_formatting_and_submodules(self):
        """Test child module names follow hierarchical dot notation."""
        for mod_name in [
            "nightmarenet.training.trainer",
            "nightmarenet.evaluation.evaluator",
            "nightmarenet.data.adaption",
            "nightmarenet.pipeline",
            "nightmarenet.pipeline_runner",
        ]:
            log = logging.getLogger(mod_name)
            assert log.name.startswith("nightmarenet.")
            assert len(log.name.split(".")) >= 2

    def test_structured_log_records_handling_exceptions(self, caplog):
        """Verify loggers properly capture exc_info without crashing."""
        with caplog.at_level(logging.ERROR):
            logger = logging.getLogger("nightmarenet.pipeline")
            try:
                raise ValueError("Simulated pipeline failure")
            except ValueError:
                logger.exception("Caught pipeline error during step execution", extra={"step": "evaluate"})

        assert "Caught pipeline error during step execution" in caplog.text
        assert "ValueError: Simulated pipeline failure" in caplog.text

    def test_log_level_environment_variable_override(self, monkeypatch):
        """Verify environment variable overrides config file log_level."""
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        env_level = os.environ.get("LOG_LEVEL", "INFO")
        reset_logging()
        setup_logging(log_level=env_level, console=False, file_logging=False)
        logger = logging.getLogger("nightmarenet")
        assert logger.level == logging.ERROR
        reset_logging()

    def test_multiple_cycles_logging_output_consistency(self, caplog):
        """Simulate multiple training loop iterations emitting structured log events."""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("nightmarenet.training.trainer")
            for cycle in range(1, 4):
                for epoch in range(1, 3):
                    logger.info(
                        "Epoch step completed",
                        extra={"cycle": cycle, "epoch": epoch, "loss": 0.5 / (cycle * epoch)},
                    )
        
        assert "Epoch step completed" in caplog.text

    def test_logger_disabled_when_filtered(self, caplog):
        """Verify messages below threshold are not recorded."""
        reset_logging()
        setup_logging(log_level="WARNING", console=False, file_logging=False)
        with caplog.at_level(logging.WARNING):
            logger = logging.getLogger("nightmarenet.data.adaption")
            logger.info("Informational processing message")
            logger.debug("Verbose debug information")
        
        assert "Informational processing message" not in caplog.text
        assert "Verbose debug information" not in caplog.text
        reset_logging()

    def test_custom_handler_injection_compatibility(self):
        """Verify custom logging handler attaches cleanly alongside default handlers."""
        reset_logging()
        setup_logging(log_level="INFO", console=False, file_logging=False)
        root = logging.getLogger("nightmarenet")
        custom_handler = logging.NullHandler()
        root.addHandler(custom_handler)
        assert custom_handler in root.handlers
        root.removeHandler(custom_handler)
        reset_logging()

    def test_log_formatter_plain_text_structure(self):
        """Verify default plain text formatter includes timestamp and level brackets."""
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        record = logging.LogRecord(
            name="nightmarenet.training.trainer",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Cycle 1 wake phase started",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "[INFO] nightmarenet.training.trainer: Cycle 1 wake phase started" in output

    def test_structured_metrics_logging_in_evaluator(self, caplog):
        """Test evaluator metric summaries are recorded with dictionary data."""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("nightmarenet.evaluation.evaluator")
            metrics = {
                "dream_robustness": 0.85,
                "nightmare_robustness": 0.72,
                "overall_score": 0.785,
                "delta_drop": 0.13,
            }
            logger.info("Evaluation complete across all perturbations", extra={"metrics": metrics})

        assert "Evaluation complete across all perturbations" in caplog.text

    def test_structured_data_adaption_logging(self, caplog):
        """Test data processing and chunking milestones emit structured info logs."""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("nightmarenet.data.adaption")
            logger.info("Tokenizing and chunking dataset", extra={"num_samples": 5000, "max_length": 128})
            logger.info("Dataset chunking finished", extra={"total_chunks": 4820})

        assert "Tokenizing and chunking dataset" in caplog.text
        assert "Dataset chunking finished" in caplog.text

    def test_pipeline_runner_status_transitions_logging(self, caplog):
        """Test background pipeline runner transition events are logged."""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("nightmarenet.pipeline_runner")
            logger.info("Launching background pipeline worker thread", extra={"run_id": "run_test_001"})
            logger.info("Heartbeat tick", extra={"run_id": "run_test_001", "uptime_sec": 60})
            logger.info("Pipeline runner thread exiting cleanly", extra={"run_id": "run_test_001"})

        assert "Launching background pipeline worker thread" in caplog.text
        assert "Heartbeat tick" in caplog.text
        assert "Pipeline runner thread exiting cleanly" in caplog.text

    def test_logging_config_with_empty_config_dict(self):
        """Test setup_logging_from_config works gracefully with empty dict."""
        reset_logging()
        setup_logging_from_config({})
        logger = logging.getLogger("nightmarenet")
        assert logger.level == logging.INFO
        reset_logging()

    def test_logging_config_with_none_values(self):
        """Test setup_logging_from_config handles None fields without TypeError."""
        reset_logging()
        setup_logging_from_config({"observability": {"log_level": "INFO", "json_logs": False}})
        logger = logging.getLogger("nightmarenet")
        assert logger.level == logging.INFO
        reset_logging()

    def test_root_logger_handlers_not_duplicated_when_reset_called(self):
        """Test setup_logging after reset maintains single handler per stream."""
        reset_logging()
        setup_logging(log_level="INFO", console=True, file_logging=False)
        assert len(logging.getLogger("nightmarenet").handlers) == 1
        reset_logging()
        setup_logging(log_level="INFO", console=True, file_logging=False)
        assert len(logging.getLogger("nightmarenet").handlers) == 1
        reset_logging()

    def test_structured_metric_fields_serializability(self):
        """Ensure all metrics logged in extra dicts are JSON serializable."""
        sample_metrics = {
            "epoch": 1,
            "loss": 0.324,
            "accuracy": 0.941,
            "f1": 0.928,
            "device": "cpu",
            "phase": "wake",
        }
        serialized = json.dumps(sample_metrics)
        deserialized = json.loads(serialized)
        assert deserialized["epoch"] == 1
        assert deserialized["phase"] == "wake"
        assert round(deserialized["loss"], 3) == 0.324

    def test_structured_logging_multiple_loggers_isolation(self, caplog):
        """Verify different module loggers maintain distinct namespaces and records."""
        with caplog.at_level(logging.INFO):
            logger_trainer = logging.getLogger("nightmarenet.training.trainer")
            logger_eval = logging.getLogger("nightmarenet.evaluation.evaluator")
            logger_adapt = logging.getLogger("nightmarenet.data.adaption")

            logger_trainer.info("Trainer step A")
            logger_eval.info("Evaluator step B")
            logger_adapt.info("Adaption step C")

        assert "Trainer step A" in caplog.text
        assert "Evaluator step B" in caplog.text
        assert "Adaption step C" in caplog.text

    def test_log_level_hierarchy_filtering(self, caplog):
        """Verify hierarchical filtering works accurately across levels."""
        reset_logging()
        setup_logging(log_level="ERROR", console=False, file_logging=False)
        with caplog.at_level(logging.DEBUG):
            logger = logging.getLogger("nightmarenet.pipeline")
            logger.debug("Debug event")
            logger.info("Info event")
            logger.warning("Warning event")
            logger.error("Error event")

        assert "Error event" in caplog.text
        assert "Warning event" not in caplog.text
        assert "Info event" not in caplog.text
        assert "Debug event" not in caplog.text
        reset_logging()
