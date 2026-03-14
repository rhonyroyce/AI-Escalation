"""Centralized logging configuration for the Escalation AI pipeline.

Provides structured JSON logging for machine parsing (log aggregation tools)
alongside human-readable console output.  Also provides a ``PhaseTimer``
context manager for timing pipeline phases with automatic log messages.
"""

import logging
import logging.handlers
import json
import sys
import time
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Produces structured JSON log lines for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if hasattr(record, "phase"):
            log_data["phase"] = record.phase
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "ticket_count"):
            log_data["ticket_count"] = record.ticket_count
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with optional icons."""

    FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
    DATEFMT = "%H:%M:%S"

    LEVEL_ICONS = {
        logging.DEBUG: "  ",
        logging.INFO: "  ",
        logging.WARNING: "  ",
        logging.ERROR: "  ",
        logging.CRITICAL: "  ",
    }

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record)


class PhaseTimer:
    """Context manager that logs phase start/end with duration.

    Usage::

        with PhaseTimer("Classification", logger, phase=1):
            run_classification()
    """

    def __init__(
        self,
        phase_name: str,
        phase_logger: logging.Logger,
        phase: Optional[int] = None,
    ) -> None:
        self.phase_name = phase_name
        self.logger = phase_logger
        self.phase = phase
        self._start: float = 0.0

    def __enter__(self) -> "PhaseTimer":
        self._start = time.perf_counter()
        extra = {}
        if self.phase is not None:
            extra["phase"] = self.phase
        self.logger.info(
            "Phase started: %s", self.phase_name, extra=extra
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        duration_ms = (time.perf_counter() - self._start) * 1000
        extra: dict = {"duration_ms": round(duration_ms, 1)}
        if self.phase is not None:
            extra["phase"] = self.phase

        if exc_type is not None:
            self.logger.error(
                "Phase FAILED: %s (%.1f ms)",
                self.phase_name,
                duration_ms,
                extra=extra,
                exc_info=(exc_type, exc_val, exc_tb),
            )
        else:
            self.logger.info(
                "Phase completed: %s (%.1f ms)",
                self.phase_name,
                duration_ms,
                extra=extra,
            )
        return None  # Don't suppress exceptions


def _is_interactive() -> bool:
    """Return True if stdout is connected to a terminal (TTY)."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_logs: bool = True,
) -> Path:
    """Configure logging with both console (human) and file (JSON) handlers.

    Args:
        log_dir: Directory for log files (created if missing).
        level: Console log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Explicit log file path.  If ``None``, a timestamped file
            is created inside *log_dir*.
        json_logs: If ``True``, the file handler uses JSON formatting.
            If ``False``, uses the standard text format.

    Returns:
        Path to the log file being written.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir_path / f"escalation_ai_{timestamp}.log"

    # --- File handler: always DEBUG, JSON formatted ---
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    if json_logs:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    # --- Console handler: user-specified level, human-readable ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(ConsoleFormatter())

    # --- Configure root logger ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove pre-existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Quiet noisy third-party loggers
    for name in ("urllib3", "requests", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_path
