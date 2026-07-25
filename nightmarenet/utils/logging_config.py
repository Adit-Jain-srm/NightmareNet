"""Structured logging configuration for NightmareNet.

Sets up consistent logging across all modules with file and console handlers,
structured formatting, and configurable log levels.

When ``json_logs=True`` and ``python-json-logger`` is installed, all handlers
emit newline-delimited JSON records suitable for ingestion by Datadog, Grafana
Loki, or any structured log pipeline. Falls back to plain-text formatting if
the package is absent so that existing environments are unaffected.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

_INITIALIZED = False

#: Fields always included in every JSON log record.
_JSON_LOG_FIELDS = ["asctime", "levelname", "name", "message"]


def _build_formatter(json_logs: bool) -> logging.Formatter:
    """Return a formatter — JSON when requested and available, else plain text.

    Args:
        json_logs: If True, attempt to use JsonFormatter from python-json-logger.

    Returns:
        A :class:`logging.Formatter` instance.
    """
    if json_logs:
        try:
            try:
                from pythonjsonlogger.json import JsonFormatter  # type: ignore[import-untyped]
            except ImportError:
                from pythonjsonlogger.jsonlogger import (
                    JsonFormatter,  # type: ignore[import-untyped]
                )

            return JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ",
                rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
            )
        except ImportError:
            logging.getLogger("nightmarenet").warning(
                "python-json-logger not installed; falling back to plain-text logging. "
                "Install it with: pip install nightmarenet[otel]"
            )

    return logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    console: bool = True,
    file_logging: bool = True,
    json_logs: bool = False,
) -> None:
    """Configure structured logging for the NightmareNet package.

    Sets up root logger for the 'nightmarenet' namespace with console and
    optional file handlers. Safe to call multiple times (idempotent).

    Args:
        log_dir: Directory for log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Whether to add a console handler.
        file_logging: Whether to add a file handler.
        json_logs: Emit JSON-structured log records (requires
            ``python-json-logger``; install via ``pip install nightmarenet[otel]``).
            Falls back to plain text if the package is unavailable.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = _build_formatter(json_logs)

    root_logger = logging.getLogger("nightmarenet")
    root_logger.setLevel(level)
    root_logger.propagate = False

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if file_logging:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"nightmarenet_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.info(
        "Logging initialized",
        extra={"log_level": log_level, "log_dir": log_dir, "json_logs": json_logs},
    )


def setup_logging_from_config(config: dict) -> None:
    """Convenience wrapper that reads ``observability`` settings from a config dict.

    Args:
        config: Full NightmareNet YAML configuration dictionary.
    """
    obs = config.get("observability", {})
    training = config.get("training", {})
    setup_logging(
        log_dir=training.get("log_dir", "logs"),
        log_level=obs.get("log_level", "INFO"),
        console=True,
        file_logging=True,
        json_logs=obs.get("json_logs", False),
    )


def reset_logging() -> None:
    """Reset logging configuration (primarily for testing)."""
    global _INITIALIZED
    root_logger = logging.getLogger("nightmarenet")
    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)
    root_logger.propagate = True
    _INITIALIZED = False
