"""
Structured logger implementation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .formatters import JSONFormatter, TextFormatter, ColoredTextFormatter
from .handlers import (
    RotatingFileHandlerWithCompression,
    SyslogHandler,
    BufferedHandler,
    MultiProcessSafeHandler,
)


class StructuredLogger:
    """
    Enhanced logger with structured logging support.

    Provides convenient methods for adding context and metrics to logs.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize structured logger.

        Args:
            logger: Underlying Python logger
        """
        self.logger = logger

    def _log_with_context(self, level: int, msg: str, context: Optional[dict] = None, **kwargs):
        """
        Log with context.

        Args:
            level: Log level
            msg: Message
            context: Context dictionary
            **kwargs: Additional keyword arguments for logging
        """
        extra = kwargs.get('extra', {})
        if context:
            extra['context'] = context
        kwargs['extra'] = extra
        self.logger.log(level, msg, **kwargs)

    def debug(self, msg: str, context: Optional[dict] = None, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, msg, context, **kwargs)

    def info(self, msg: str, context: Optional[dict] = None, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, msg, context, **kwargs)

    def warning(self, msg: str, context: Optional[dict] = None, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, msg, context, **kwargs)

    def error(self, msg: str, context: Optional[dict] = None, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, msg, context, **kwargs)

    def critical(self, msg: str, context: Optional[dict] = None, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, msg, context, **kwargs)

    def __getattr__(self, name: str):
        """Delegate unknown attributes to underlying logger."""
        return getattr(self.logger, name)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_format: str = "json",
    service_name: str = "quant-platform",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_syslog: bool = False,
    syslog_address: Optional[str] = None,
    max_file_size_mb: int = 100,
    backup_count: int = 10,
    include_context: bool = True,
    multi_process_safe: bool = False,
) -> StructuredLogger:
    """
    Setup structured logging with multiple handlers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_format: Format ('json' or 'text')
        service_name: Service name for logs
        enable_console: Enable console output
        enable_file: Enable file output
        enable_syslog: Enable syslog output
        syslog_address: Syslog server address
        max_file_size_mb: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        include_context: Include context in logs
        multi_process_safe: Enable multi-process safe file locking

    Returns:
        Configured StructuredLogger instance

    Example:
        logger = setup_logging(
            level="INFO",
            log_dir=Path("/var/log/quant"),
            log_format="json",
            service_name="factor-engine"
        )
        logger.info("Service started", context={"version": "1.0.0"})
    """
    # Create root logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    # Choose formatter
    if log_format == "json":
        formatter = JSONFormatter(include_context=include_context, service_name=service_name)
    else:
        formatter = TextFormatter(include_context=include_context, service_name=service_name)

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # Use colored formatter for console if text format
        if log_format == "text" and sys.stdout.isatty():
            console_formatter = ColoredTextFormatter(
                include_context=include_context,
                service_name=service_name
            )
            console_handler.setFormatter(console_formatter)
        else:
            console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    # File handler
    if enable_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{service_name}.log"
        file_handler = RotatingFileHandlerWithCompression(
            filename=log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8',
            compress_old=True,
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)

        # Wrap with multi-process safe handler if requested
        if multi_process_safe:
            file_handler = MultiProcessSafeHandler(file_handler)

        logger.addHandler(file_handler)

    # Syslog handler
    if enable_syslog and syslog_address:
        syslog_handler = SyslogHandler(address=syslog_address)
        syslog_handler.setLevel(getattr(logging, level.upper()))
        syslog_handler.setFormatter(formatter)
        logger.addHandler(syslog_handler)

    return StructuredLogger(logger)


def get_logger(name: str, parent: str = "quant-platform") -> StructuredLogger:
    """
    Get a child logger under the parent namespace.

    Args:
        name: Logger name
        parent: Parent logger name

    Returns:
        StructuredLogger instance

    Example:
        logger = get_logger("factor_engine")
        logger.info("Factor computed", context={"factor_id": "momentum"})
    """
    full_name = f"{parent}.{name}" if parent else name
    return StructuredLogger(logging.getLogger(full_name))


def configure_from_config(config) -> StructuredLogger:
    """
    Configure logging from a ProductionConfig object.

    Args:
        config: ProductionConfig instance with logging configuration

    Returns:
        Configured StructuredLogger instance
    """
    return setup_logging(
        level=config.logging.level,
        log_dir=config.logging.output_dir,
        log_format=config.logging.format,
        service_name=config.service_name,
        enable_console=config.logging.enable_console,
        enable_file=config.logging.enable_file,
        enable_syslog=config.logging.enable_syslog,
        syslog_address=config.logging.syslog_address,
        max_file_size_mb=config.logging.max_file_size_mb,
        backup_count=config.logging.backup_count,
        include_context=config.logging.include_context,
    )
