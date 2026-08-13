"""
Structured logging configuration for QuantEvaluator package.

Provides:
- Structured JSON logging with context
- Performance logging with function timing
- Error context capture with stack traces
- Configurable log levels
- Sensitive data sanitization
"""

import json
import logging
import time
import functools
import re
import traceback
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from pathlib import Path


# Patterns for sensitive data that should be sanitized
SENSITIVE_PATTERNS = [
    (re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'password'),
    (re.compile(r'token["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'token'),
    (re.compile(r'api_key["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'api_key'),
    (re.compile(r'secret["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'secret'),
    (re.compile(r'authorization:\s*bearer\s+(\S+)', re.IGNORECASE), 'auth_token'),
    (re.compile(r'aws_secret_access_key["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'aws_secret'),
]


def sanitize_message(message: str) -> str:
    """Remove sensitive data from log messages."""
    sanitized = message
    for pattern, label in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(f'{label}=***REDACTED***', sanitized)
    return sanitized


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize sensitive data from dictionaries."""
    if not isinstance(data, dict):
        return data

    sanitized = {}
    sensitive_keys = {'password', 'token', 'api_key', 'secret', 'auth', 'credential', 'access_key', 'secret_key'}

    for key, value in data.items():
        lower_key = key.lower()
        if any(sk in lower_key for sk in sensitive_keys):
            sanitized[key] = '***REDACTED***'
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [sanitize_dict(v) if isinstance(v, dict) else v for v in value]
        elif isinstance(value, str):
            sanitized[key] = sanitize_message(value)
        else:
            sanitized[key] = value

    return sanitized


class StructuredFormatter(logging.Formatter):
    """JSON formatter with structured context and sanitization."""

    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': sanitize_message(record.getMessage()),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra context if available
        if self.include_context and hasattr(record, 'context'):
            log_data['context'] = sanitize_dict(record.context)

        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info),
            }

        # Add any custom fields
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'lineno', 'module', 'msecs', 'message',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'context', 'duration_ms'}:
                if not key.startswith('_'):
                    log_data[key] = value

        return json.dumps(log_data)


class PerformanceLogger:
    """Context manager and decorator for performance logging."""

    def __init__(self, logger: logging.Logger, operation: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.operation = operation
        self.context = context or {}
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000

        extra = {
            'duration_ms': round(duration_ms, 2),
            'context': {**self.context, 'operation': self.operation}
        }

        if exc_type is None:
            self.logger.info(f"Operation completed: {self.operation}", extra=extra)
        else:
            extra['context']['error'] = str(exc_val)
            self.logger.error(f"Operation failed: {self.operation}", extra=extra, exc_info=True)

        return False

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation = self.operation or f"{func.__module__}.{func.__qualname__}"
            context = self.context.copy()

            # Add function arguments to context (sanitized)
            if args:
                context['args_count'] = len(args)
            if kwargs:
                context['kwargs'] = sanitize_dict(kwargs)

            with PerformanceLogger(self.logger, operation, context):
                return func(*args, **kwargs)

        return wrapper


def log_performance(operation: Optional[str] = None, logger: Optional[logging.Logger] = None):
    """
    Decorator for logging function performance.

    Example:
        @log_performance("backtest_strategy")
        def run_backtest(strategy: Strategy):
            ...
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger, operation
        if logger is None:
            logger = logging.getLogger(func.__module__)
        if operation is None:
            operation = f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                extra = {
                    'duration_ms': round(duration_ms, 2),
                    'context': {
                        'operation': operation,
                        'success': True,
                    }
                }
                logger.info(f"Function executed: {operation}", extra=extra)
                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                extra = {
                    'duration_ms': round(duration_ms, 2),
                    'context': {
                        'operation': operation,
                        'success': False,
                        'error_type': type(e).__name__,
                    }
                }
                logger.error(f"Function failed: {operation}", extra=extra, exc_info=True)
                raise

        return wrapper
    return decorator


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = True,
    include_context: bool = True,
) -> logging.Logger:
    """
    Configure structured logging for the QuantEvaluator package.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        structured: Use JSON structured format (True) or plain text (False)
        include_context: Include context fields in structured logs

    Returns:
        Configured logger instance

    Example:
        logger = setup_logging(level="DEBUG", log_file=Path("logs/quant_evaluator.log"))
        logger.info("Evaluation started", extra={'context': {'strategy_id': 'momentum_v1'}})
    """
    logger = logging.getLogger('quant_evaluator')
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))

    if structured:
        console_handler.setFormatter(StructuredFormatter(include_context=include_context))
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )

    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))

        if structured:
            file_handler.setFormatter(StructuredFormatter(include_context=include_context))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )

        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the quant_evaluator namespace."""
    return logging.getLogger(f'quant_evaluator.{name}')
