"""
Sensitive data sanitization for logs.

Prevents leaking credentials, tokens, and other sensitive information.
"""

import re
from typing import Any, Dict, List, Tuple, Pattern

# Global list of sensitive patterns
_SENSITIVE_PATTERNS: List[Tuple[Pattern, str]] = [
    (re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'password'),
    (re.compile(r'passwd["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'passwd'),
    (re.compile(r'pwd["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'pwd'),
    (re.compile(r'token["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'token'),
    (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'api_key'),
    (re.compile(r'secret["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'secret'),
    (re.compile(r'authorization:\s*bearer\s+(\S+)', re.IGNORECASE), 'auth_token'),
    (re.compile(r'authorization:\s*basic\s+(\S+)', re.IGNORECASE), 'basic_auth'),
    (re.compile(r'aws[_-]?secret[_-]?access[_-]?key["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'aws_secret'),
    (re.compile(r'private[_-]?key["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'private_key'),
    (re.compile(r'client[_-]?secret["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), 'client_secret'),
    # Credit card patterns (basic)
    (re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'), 'credit_card'),
    # SSN pattern (US)
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 'ssn'),
    # JWT tokens
    (re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'), 'jwt_token'),
]

# Sensitive keys that should be redacted in dictionaries
_SENSITIVE_KEYS = {
    'password', 'passwd', 'pwd',
    'token', 'access_token', 'refresh_token', 'id_token',
    'api_key', 'apikey', 'api-key',
    'secret', 'client_secret',
    'auth', 'authorization', 'authentication',
    'credential', 'credentials',
    'access_key', 'secret_key',
    'private_key', 'public_key',
    'certificate', 'cert',
    'session', 'session_id',
    'cookie', 'cookies',
}


def add_sensitive_pattern(pattern: str, label: str) -> None:
    """
    Add a custom sensitive pattern for sanitization.

    Args:
        pattern: Regular expression pattern
        label: Label to use when replacing (e.g., 'custom_secret')

    Example:
        add_sensitive_pattern(r'internal_id["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', 'internal_id')
    """
    _SENSITIVE_PATTERNS.append((re.compile(pattern, re.IGNORECASE), label))


def sanitize_message(message: str) -> str:
    """
    Remove sensitive data from log messages.

    Args:
        message: Original message

    Returns:
        Sanitized message with sensitive data replaced

    Example:
        >>> sanitize_message("Login with password=secret123")
        'Login with password=***REDACTED***'
    """
    if not isinstance(message, str):
        return message

    sanitized = message
    for pattern, label in _SENSITIVE_PATTERNS:
        sanitized = pattern.sub(f'{label}=***REDACTED***', sanitized)

    return sanitized


def sanitize_dict(data: Any, max_depth: int = 10, _current_depth: int = 0) -> Any:
    """
    Recursively sanitize sensitive data from dictionaries.

    Args:
        data: Data to sanitize (dict, list, or primitive)
        max_depth: Maximum recursion depth
        _current_depth: Current recursion depth (internal)

    Returns:
        Sanitized data structure

    Example:
        >>> sanitize_dict({'user': 'john', 'password': 'secret'})
        {'user': 'john', 'password': '***REDACTED***'}
    """
    if _current_depth >= max_depth:
        return '***MAX_DEPTH_EXCEEDED***'

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            lower_key = key.lower() if isinstance(key, str) else str(key).lower()

            # Check if key is sensitive
            if any(sk in lower_key for sk in _SENSITIVE_KEYS):
                sanitized[key] = '***REDACTED***'
            elif isinstance(value, dict):
                sanitized[key] = sanitize_dict(value, max_depth, _current_depth + 1)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [
                    sanitize_dict(item, max_depth, _current_depth + 1)
                    for item in value
                ]
            elif isinstance(value, str):
                sanitized[key] = sanitize_message(value)
            else:
                sanitized[key] = value

        return sanitized

    elif isinstance(data, (list, tuple)):
        sanitized_list = [
            sanitize_dict(item, max_depth, _current_depth + 1)
            for item in data
        ]
        return type(data)(sanitized_list)

    elif isinstance(data, str):
        return sanitize_message(data)

    else:
        return data


def sanitize_exception(exc: Exception) -> Dict[str, Any]:
    """
    Sanitize exception information for logging.

    Args:
        exc: Exception to sanitize

    Returns:
        Dictionary with sanitized exception info
    """
    import traceback

    return {
        'type': type(exc).__name__,
        'message': sanitize_message(str(exc)),
        'traceback': [
            sanitize_message(line)
            for line in traceback.format_tb(exc.__traceback__)
        ] if exc.__traceback__ else [],
    }


def is_sensitive_key(key: str) -> bool:
    """
    Check if a key name indicates sensitive data.

    Args:
        key: Key name to check

    Returns:
        True if key appears to contain sensitive data
    """
    lower_key = key.lower()
    return any(sk in lower_key for sk in _SENSITIVE_KEYS)
