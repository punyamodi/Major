"""
MedAide+ Structured Logger

Provides structured, colored logging with file and console handlers.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional


class ColorFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for console output."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
) -> None:
    """
    Configure global logging for MedAide+.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If None, logs only to console.
        format_str: Log message format string.
    """
    root_logger = logging.getLogger("medaide_plus")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(ColorFormatter(format_str))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(format_str))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger under the medaide_plus namespace.

    Args:
        name: Logger name (e.g., 'm1_amqu', 'pipeline').

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(f"medaide_plus.{name}")
    if not logging.getLogger("medaide_plus").handlers:
        setup_logging()
    return logger
