"""
Performance logging utilities.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional


class PerformanceLogger:
    """
    Context manager and decorator for performance logging.

    Example as context manager:
        with PerformanceLogger(logger, "database_query", {"table": "factors"}):
            result = db.query(...)

    Example as decorator:
        @PerformanceLogger(logger, "compute_factor")
        def compute_factor(data):
            ...
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        log_args: bool = False,
    ):
        """
        Initialize performance logger.

        Args:
            logger: Logger instance
            operation: Operation name
            context: Additional context to log
            log_args: Whether to log function arguments
        """
        self.logger = logger
        self.operation = operation
        self.context = context or {}
        self.log_args = log_args
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        self.logger.debug(
            f"Starting operation: {self.operation}",
            extra={'context': self.context, 'operation': self.operation}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log."""
        duration_ms = (time.perf_counter() - self.start_time) * 1000

        extra = {
            'duration_ms': round(duration_ms, 2),
            'operation': self.operation,
            'context': self.context,
        }

        if exc_type is None:
            self.logger.info(
                f"Operation completed: {self.operation}",
                extra=extra
            )
        else:
            extra['context']['error_type'] = exc_type.__name__
            extra['context']['error'] = str(exc_val)
            self.logger.error(
                f"Operation failed: {self.operation}",
                extra=extra,
                exc_info=(exc_type, exc_val, exc_tb)
            )

        return False  # Don't suppress exception

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation = self.operation or f"{func.__module__}.{func.__qualname__}"
            context = self.context.copy()

            # Add function arguments to context if requested
            if self.log_args:
                if args:
                    context['args_count'] = len(args)
                    # Log first few args (sanitized)
                    context['args_preview'] = str(args[:3])
                if kwargs:
                    # Don't log sensitive kwargs
                    safe_kwargs = {
                        k: v for k, v in kwargs.items()
                        if not any(s in k.lower() for s in ['password', 'token', 'secret', 'key'])
                    }
                    context['kwargs'] = safe_kwargs

            with PerformanceLogger(self.logger, operation, context):
                return func(*args, **kwargs)

        return wrapper


def log_performance(
    operation: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    log_args: bool = False,
):
    """
    Decorator for logging function performance.

    Args:
        operation: Operation name (defaults to function qualname)
        logger: Logger instance (defaults to module logger)
        log_args: Whether to log function arguments

    Example:
        @log_performance("fetch_market_data")
        def fetch_data(symbol: str):
            ...

        @log_performance(log_args=True)
        def process_factors(factors: List[str]):
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
                    'operation': operation,
                    'context': {
                        'success': True,
                    }
                }

                if log_args:
                    if args:
                        extra['context']['args_count'] = len(args)
                    if kwargs:
                        safe_kwargs = {
                            k: v for k, v in kwargs.items()
                            if not any(s in k.lower() for s in ['password', 'token', 'secret', 'key'])
                        }
                        extra['context']['kwargs'] = safe_kwargs

                logger.info(f"Function executed: {operation}", extra=extra)
                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000

                extra = {
                    'duration_ms': round(duration_ms, 2),
                    'operation': operation,
                    'context': {
                        'success': False,
                        'error_type': type(e).__name__,
                    }
                }

                logger.error(f"Function failed: {operation}", extra=extra, exc_info=True)
                raise

        return wrapper

    return decorator


def log_execution_time(logger: logging.Logger, operation: str):
    """
    Simple decorator that only logs execution time.

    Args:
        logger: Logger instance
        operation: Operation name

    Example:
        @log_execution_time(logger, "heavy_computation")
        def compute():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.debug(
                    f"{operation} took {duration_ms:.2f}ms",
                    extra={'duration_ms': round(duration_ms, 2), 'operation': operation}
                )
        return wrapper
    return decorator


@contextmanager
def timed_block(logger: logging.Logger, block_name: str, context: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing code blocks.

    Args:
        logger: Logger instance
        block_name: Name of the code block
        context: Additional context

    Example:
        with timed_block(logger, "data_loading", {"source": "database"}):
            data = load_data()
    """
    start = time.perf_counter()
    extra_context = context or {}

    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Block '{block_name}' completed in {duration_ms:.2f}ms",
            extra={
                'duration_ms': round(duration_ms, 2),
                'operation': block_name,
                'context': extra_context,
            }
        )


class OperationTimer:
    """
    Timer for measuring operation duration with intermediate checkpoints.

    Example:
        timer = OperationTimer(logger, "complex_operation")
        timer.checkpoint("loaded_data")
        # ... do work ...
        timer.checkpoint("processed_data")
        # ... more work ...
        timer.finish()
    """

    def __init__(self, logger: logging.Logger, operation: str):
        """
        Initialize timer.

        Args:
            logger: Logger instance
            operation: Operation name
        """
        self.logger = logger
        self.operation = operation
        self.start_time = time.perf_counter()
        self.last_checkpoint = self.start_time
        self.checkpoints = []

    def checkpoint(self, name: str):
        """
        Record a checkpoint.

        Args:
            name: Checkpoint name
        """
        now = time.perf_counter()
        duration_from_start = (now - self.start_time) * 1000
        duration_from_last = (now - self.last_checkpoint) * 1000

        self.checkpoints.append({
            'name': name,
            'elapsed_ms': round(duration_from_start, 2),
            'delta_ms': round(duration_from_last, 2),
        })

        self.logger.debug(
            f"Checkpoint '{name}' in {self.operation}",
            extra={
                'operation': self.operation,
                'checkpoint': name,
                'elapsed_ms': round(duration_from_start, 2),
                'delta_ms': round(duration_from_last, 2),
            }
        )

        self.last_checkpoint = now

    def finish(self):
        """
        Finish timing and log summary.
        """
        total_duration_ms = (time.perf_counter() - self.start_time) * 1000

        self.logger.info(
            f"Operation completed: {self.operation}",
            extra={
                'operation': self.operation,
                'duration_ms': round(total_duration_ms, 2),
                'checkpoints': self.checkpoints,
            }
        )
