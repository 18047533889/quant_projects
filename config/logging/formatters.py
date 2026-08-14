"""
Log formatters for structured and text logging.
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict
from .sanitization import sanitize_message, sanitize_dict, sanitize_exception


class JSONFormatter(logging.Formatter):
    """
    JSON formatter with structured context and sanitization.

    Produces one JSON object per log line with consistent schema.
    """

    def __init__(self, include_context: bool = True, service_name: str = "quant-platform"):
        super().__init__()
        self.include_context = include_context
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log structure
        log_data: Dict[str, Any] = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': sanitize_message(record.getMessage()),
            'service': self.service_name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
        }

        # Add context if available
        if self.include_context and hasattr(record, 'context'):
            log_data['context'] = sanitize_dict(record.context)

        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms

        if hasattr(record, 'operation'):
            log_data['operation'] = record.operation

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = sanitize_exception(record.exc_info[1])

        # Add stack trace if present
        if record.stack_info:
            log_data['stack_trace'] = sanitize_message(record.stack_info)

        # Add custom fields from extra
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'lineno', 'module', 'msecs', 'message',
                'pathname', 'process', 'processName', 'relativeCreated',
                'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                'context', 'duration_ms', 'operation', 'taskName'
            }:
                if not key.startswith('_'):
                    if isinstance(value, (dict, list)):
                        log_data[key] = sanitize_dict(value)
                    elif isinstance(value, str):
                        log_data[key] = sanitize_message(value)
                    else:
                        log_data[key] = value

        try:
            return json.dumps(log_data, default=str, ensure_ascii=False)
        except Exception:
            # Fallback if JSON serialization fails
            return json.dumps({
                'timestamp': log_data['timestamp'],
                'level': record.levelname,
                'message': 'Failed to serialize log message',
                'error': 'JSON serialization error'
            })


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter with sanitization.

    Format: TIMESTAMP LEVEL [SERVICE.LOGGER] MESSAGE [CONTEXT]
    """

    def __init__(self, include_context: bool = True, service_name: str = "quant-platform"):
        super().__init__()
        self.include_context = include_context
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text."""
        # Timestamp
        timestamp = datetime.utcfromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Base message
        message = sanitize_message(record.getMessage())
        parts = [
            timestamp,
            record.levelname.ljust(8),
            f'[{self.service_name}.{record.name}]',
            message,
        ]

        # Add location
        parts.append(f'({record.module}:{record.lineno})')

        # Add duration if available
        if hasattr(record, 'duration_ms'):
            parts.append(f'[{record.duration_ms:.2f}ms]')

        # Add context if available
        if self.include_context and hasattr(record, 'context'):
            context_str = json.dumps(sanitize_dict(record.context), default=str)
            parts.append(f'context={context_str}')

        result = ' '.join(parts)

        # Add exception if present
        if record.exc_info:
            exc_text = ''.join(traceback.format_exception(*record.exc_info))
            result += '\n' + sanitize_message(exc_text)

        return result


class ColoredTextFormatter(TextFormatter):
    """
    Text formatter with ANSI color codes for terminal output.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get base formatted message
        message = super().format(record)

        # Add color based on level
        color = self.COLORS.get(record.levelname, self.RESET)
        return f'{color}{message}{self.RESET}'
