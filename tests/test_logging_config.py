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
