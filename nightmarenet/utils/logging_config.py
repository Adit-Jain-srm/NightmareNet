"""Structured logging configuration for NightmareNet.

Sets up consistent logging across all modules with file and console handlers,
structured formatting, and configurable log levels.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


_INITIALIZED = False


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    console: bool = True,
    file_logging: bool = True,
) -> None:
    """Configure structured logging for the NightmareNet package.

    Sets up root logger for the 'nightmarenet' namespace with console and
    optional file handlers. Safe to call multiple times (idempotent).

    Args:
        log_dir: Directory for log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Whether to add a console handler.
        file_logging: Whether to add a file handler.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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

    root_logger.info("Logging initialized (level=%s, dir=%s)", log_level, log_dir)


def reset_logging() -> None:
    """Reset logging configuration (primarily for testing)."""
    global _INITIALIZED
    root_logger = logging.getLogger("nightmarenet")
    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)
    _INITIALIZED = False
