"""Structured logging setup using loguru."""

from __future__ import annotations

import sys

from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    """Configure loguru for the application.

    Call once at startup. All modules should import logger from loguru directly:
        from loguru import logger
    """
    logger.remove()  # Remove default handler

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )


def setup_file_logger(path: str = "logs/geo_spy.log", level: str = "DEBUG") -> None:
    """Add file logging (optional, call after setup_logger)."""
    logger.add(
        path,
        level=level,
        rotation="50 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )
