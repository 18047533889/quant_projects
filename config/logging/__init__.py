"""
Enterprise-grade structured logging system.

Features:
- JSON structured logging
- Log rotation with size and backup limits
- Sensitive data sanitization
- Performance logging
- Contextual logging
- Multiple output handlers (console, file, syslog)
"""

from .structured_logger import (
    StructuredLogger,
    get_logger,
    setup_logging,
)
from .formatters import (
    JSONFormatter,
    TextFormatter,
)
from .handlers import (
    RotatingFileHandlerWithCompression,
    SyslogHandler,
)
from .performance import (
    PerformanceLogger,
    log_performance,
    log_execution_time,
)
from .sanitization import (
    sanitize_message,
    sanitize_dict,
    add_sensitive_pattern,
)

__all__ = [
    "StructuredLogger",
    "get_logger",
    "setup_logging",
    "JSONFormatter",
    "TextFormatter",
    "RotatingFileHandlerWithCompression",
    "SyslogHandler",
    "PerformanceLogger",
    "log_performance",
    "log_execution_time",
    "sanitize_message",
    "sanitize_dict",
    "add_sensitive_pattern",
]
