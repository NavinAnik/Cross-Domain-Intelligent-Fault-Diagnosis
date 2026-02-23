"""
Centralized logging configuration for the project.

Creates logs/ directory and configures root logger with file and console handlers.
All logs are written to logs/run_YYYYMMDD_HHMMSS.log.
"""

import logging
import os
from datetime import datetime


_LOGGING_INITIALIZED = False


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    include_console: bool = True,
) -> str:
    """
    Configure root logger with file handler and optional console handler.

    Args:
        log_dir: Directory for log files. Created if missing.
        log_level: Logging level (default INFO).
        include_console: If True, also log to stdout.

    Returns:
        Path to the log file.
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return _get_log_file_path(log_dir)

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    _LOGGING_INITIALIZED = True
    return log_file


def _get_log_file_path(log_dir: str = "logs") -> str:
    """Return most recent log file path in log_dir (for reference)."""
    if not os.path.isdir(log_dir):
        return os.path.join(log_dir, "run_unknown.log")
    files = [f for f in os.listdir(log_dir) if f.startswith("run_") and f.endswith(".log")]
    if not files:
        return os.path.join(log_dir, "run_unknown.log")
    files.sort(reverse=True)
    return os.path.join(log_dir, files[0])
