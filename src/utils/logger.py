"""
Centralized logging configuration for the IoT Predictive Maintenance system.

Provides structured logging with JSON output for production environments
and human-readable output for development. Supports log rotation and
configurable log levels per module.
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging in production environments."""

    def __init__(self, service_name: str = "iot-predictive-maintenance"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "device_id"):
            log_entry["device_id"] = record.device_id

        if hasattr(record, "model_name"):
            log_entry["model_name"] = record.model_name

        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms

        return json.dumps(log_entry, default=str)


class DevFormatter(logging.Formatter):
    """Human-readable formatter for development environments."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return (
            f"{color}{timestamp} [{record.levelname:8s}] "
            f"{record.module}.{record.funcName}:{record.lineno} - "
            f"{record.getMessage()}{self.RESET}"
        )


def setup_logger(
    name: str = "iot_maintenance",
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_output: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (typically module path).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output with rotation.
        json_output: If True, use JSON formatting for structured logs.
        max_bytes: Maximum log file size before rotation (default 10MB).
        backup_count: Number of rotated log files to retain.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    if json_output:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(DevFormatter())
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.

    Args:
        module_name: Fully qualified module name.

    Returns:
        Logger instance inheriting from the root IoT maintenance logger.
    """
    return logging.getLogger(f"iot_maintenance.{module_name}")
