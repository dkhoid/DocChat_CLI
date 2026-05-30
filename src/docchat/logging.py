"""Structured logging configuration using structlog.

Provides JSON-formatted logs for production (Railway/cloud) and
human-readable colored output for local development.

Usage:
    from docchat.logging import get_logger
    logger = get_logger(__name__)
    logger.info("chunks_retrieved", count=5, latency_ms=12.3)
"""

import os
import sys

import structlog


def _is_production() -> bool:
    env = os.environ.get("DOCCHAT_ENV", "development").lower()
    return env in ("production", "prod")


def configure_logging() -> None:
    """Configure structlog once at application startup."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if _is_production():
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to the given module name."""
    return structlog.get_logger(logger_name=name)


# Auto-configure on first import
configure_logging()
