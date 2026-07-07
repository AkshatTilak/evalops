"""Logging configuration utility for structured and development logs."""

import json
import logging
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs records as JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a structured JSON object.

        Args:
            record: The LogRecord instance to format.

        Returns:
            A string containing the formatted JSON log message.
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(app_env: str = "development") -> None:
    """Configures the root logger with environment-appropriate formatters.

    Args:
        app_env: The deployment environment name (e.g., 'production', 'development').
    """
    root_logger = logging.getLogger()
    # Avoid duplicate handlers if setup is called multiple times
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    # Set base level
    level = logging.INFO if app_env == "production" else logging.DEBUG
    root_logger.setLevel(level)

    # Console output handler
    handler = logging.StreamHandler(sys.stdout)

    if app_env == "production":
        formatter = JSONFormatter()
    else:
        # Easy-to-read developer logging format
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress verbose dependency logs in development
    if app_env != "production":
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
