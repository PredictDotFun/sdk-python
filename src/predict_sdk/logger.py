"""Logging configuration for the Predict SDK."""

from __future__ import annotations

import logging
from typing import Literal

LogLevel = Literal["ERROR", "WARN", "INFO", "DEBUG"]

LOG_LEVEL_MAP: dict[LogLevel, int] = {
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def get_logger(name: str = "predict_sdk", level: LogLevel = "INFO") -> logging.Logger:
    """
    Get a configured logger for the SDK.

    Args:
        name: The logger name.
        level: The log level.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(LOG_LEVEL_MAP[level])
    return logger


class Logger:
    """
    Simple logger class that mirrors the JS SDK Logger.

    Provides level-based filtering for console output.
    """

    def __init__(self, level: LogLevel = "INFO") -> None:
        self._logger = get_logger(level=level)
        self._level = level

    @property
    def level(self) -> LogLevel:
        """Get the current log level."""
        return self._level

    def set_level(self, level: LogLevel) -> None:
        """Set the log level."""
        self._level = level
        self._logger.setLevel(LOG_LEVEL_MAP[level])

    def error(self, message: str, *args: object) -> None:
        """Log an error message."""
        self._logger.error(message, *args)

    def warn(self, message: str, *args: object) -> None:
        """Log a warning message."""
        self._logger.warning(message, *args)

    def info(self, message: str, *args: object) -> None:
        """Log an info message."""
        self._logger.info(message, *args)

    def debug(self, message: str, *args: object) -> None:
        """Log a debug message."""
        self._logger.debug(message, *args)
