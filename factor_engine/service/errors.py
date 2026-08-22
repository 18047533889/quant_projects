"""R21-087..090 + R21-164..166: stable error taxonomy, HTTP mapping, redaction.

Public errors expose only ``error_code`` (stable family) + sanitized message +
``error_id``.  Full tracebacks stay in internal artifacts/logs.  Secrets
(passwords, tokens, DSNs, API keys, Authorization headers, internal env paths)
are redacted before any message crosses the service boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from enum import Enum
from typing import Any

# R21-164: stable error families.
ERROR_FAMILIES = [
    "VALIDATION",
    "POLICY",
    "PIT",
    "SOURCE",
    "SCHEMA",
    "DQ",
    "RESOURCE",
    "TIMEOUT",
    "CANCEL",
    "CACHE",
    "MATERIALIZE",
    "BACKEND",
    "INTERNAL",
    "AUTH",
    "AUTHORIZATION",
    "IDEMPOTENCY",
    "CONFLICT",
    "REJECTED",
    "INTERNAL_CONTRACT_VIOLATION",
    "INTERRUPTED",
]


class ErrorFamily(str, Enum):
    VALIDATION = "VALIDATION"
    POLICY = "POLICY"
    PIT = "PIT"
    SOURCE = "SOURCE"
    SCHEMA = "SCHEMA"
    DQ = "DQ"
    RESOURCE = "RESOURCE"
    TIMEOUT = "TIMEOUT"
    CANCEL = "CANCEL"
    CACHE = "CACHE"
    MATERIALIZE = "MATERIALIZE"
    BACKEND = "BACKEND"
    INTERNAL = "INTERNAL"
    AUTH = "AUTH"
    AUTHORIZATION = "AUTHORIZATION"
    IDEMPOTENCY = "IDEMPOTENCY"
    CONFLICT = "CONFLICT"
    REJECTED = "REJECTED"
    INTERNAL_CONTRACT_VIOLATION = "INTERNAL_CONTRACT_VIOLATION"
    INTERRUPTED = "INTERRUPTED"


# R21-165: stable HTTP mapping.
HTTP_MAP = {
    "VALIDATION": 422,
    "POLICY": 403,
    "PIT": 422,
    "SOURCE": 422,
    "SCHEMA": 422,
    "DQ": 422,
    "RESOURCE": 503,
    "TIMEOUT": 503,
    "CANCEL": 409,
    "CACHE": 503,
    "MATERIALIZE": 422,
    "BACKEND": 500,
    "INTERNAL": 500,
    "AUTH": 401,
    "AUTHORIZATION": 403,
    "IDEMPOTENCY": 409,
    "CONFLICT": 409,
    "REJECTED": 429,
    "INTERNAL_CONTRACT_VIOLATION": 500,
    "INTERRUPTED": 409,
}

# R21-166: explicit named error codes (not message parsing).
NAMED_ERROR_CODES = {
    "PRODUCTION_ENDPOINT_CONFIG_POLICY_CONFLICT": "CONFLICT",
    "IDEMPOTENCY_KEY_CONFLICT": "IDEMPOTENCY",
    "SOURCE_EMPTY": "SOURCE",
    "SOURCE_STALE": "SOURCE",
    "SOURCE_SCHEMA_DRIFT": "SCHEMA",
    "EXPECTED_SPARSE": "DQ",
    "FACTOR_DEGENERATE": "DQ",
    "OUTPUT_DOMAIN_VIOLATION": "DQ",
    "JOB_QUEUE_FULL": "REJECTED",
    "JOB_DEADLINE_EXCEEDED": "TIMEOUT",
    "JOB_CANCELLED": "CANCEL",
    "JOB_INTERRUPTED": "INTERRUPTED",
    "RESOURCE_BUDGET_EXCEEDED": "RESOURCE",
    "CLICKHOUSE_BUDGET_EXCEEDED": "RESOURCE",
    "UNKNOWN_JOB": "VALIDATION",
    "AUTH_REQUIRED": "AUTH",
    "AUTH_INVALID_KEY": "AUTH",
    "AUTHORIZATION_DENIED": "AUTHORIZATION",
    "OWNER_ONLY": "AUTHORIZATION",
    "MATERIALIZE_NEEDS_HIGHER_PRIVILEGE": "AUTHORIZATION",
    "INTERNAL_CONTRACT_VIOLATION": "INTERNAL_CONTRACT_VIOLATION",
    "MALFORMED_REQUEST": "VALIDATION",
    "UNSUPPORTED_FEATURE": "VALIDATION",
}


class ServiceError(Exception):
    """Exception carrying a stable error code + optional HTTP status."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        error_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.family = NAMED_ERROR_CODES.get(code, "INTERNAL")
        self.message = message
        self.status = status if status is not None else HTTP_MAP.get(self.family, 500)
        self.error_id = error_id or uuid.uuid4().hex[:12]
        self.extra = dict(extra or {})

    def to_public(self) -> dict[str, Any]:
        return {
            "error": self.family,
            "error_code": self.code,
            "message": sanitize_message(self.message),
            "error_id": self.error_id,
            **(self.extra or {}),
        }


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"([Pp]assword\s*[=:]\s*)[^\s,;\"']+"), r"\1<redacted>"),
    (re.compile(r"([Tt]oken\s*[=:]\s*)[^\s,;\"']+"), r"\1<redacted>"),
    (re.compile(r"([Aa][Pp][Ii][_-]?[Kk]ey\s*[=:]\s*)[^\s,;\"']+"), r"\1<redacted>"),
    (re.compile(r"(?i)(postgres|mysql|duckdb|clickhouse|sqlite)[a-z0-9+]*://[^\s]+"), "<redacted-dsn>"),
    (re.compile(r"(Authorization\s*[=:]\s*)[^\r\n,;]+"), r"\1<redacted>"),
    (re.compile(r"(X-Api-Key\s*[=:]\s*)[^\r\n,;]+"), r"\1<redacted>"),
    (re.compile(r"(s3://)[^/\s]+(:[^\s@]+)?@"), r"\1<redacted>@"),
]
_SECRET_ENV_KEYS = {
    "password", "token", "secret", "api_key", "apikey", "authorization",
    "dsn", "access_key", "private_key",
}


def sanitize_message(text: Any) -> str:
    """R21-089: redact credentials/DSNs from any user-visible message."""
    if text is None:
        return ""
    out = str(text)
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def redact_config(mapping: dict[str, Any], *, depth: int = 0) -> dict[str, Any]:
    """Return a config dict with secret-shaped keys/values removed.  Never log
    the full unsanitized config (R21-174)."""
    if depth > 6:
        return {"<deep>"}
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        lowered = str(key).lower()
        if any(seg in lowered for seg in _SECRET_ENV_KEYS):
            out[key] = "<redacted>"
            continue
        if isinstance(value, dict):
            out[key] = redact_config(value, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            out[key] = [
                redact_config(v, depth=depth + 1) if isinstance(v, dict) else sanitize_message(v)
                for v in value
            ]
        else:
            out[key] = sanitize_message(value)
    return out


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]


def classify_exception(exc: BaseException, *, run_mode: str = "research") -> tuple[str, str]:
    """Map an arbitrary exception to a stable ``(error_code, family)``.

    R21-166: error codes flow into metrics; never parse exception message text
    downstream.
    """
    from runtime.resource_errors import ResourceGovernanceError, is_fail_closed_error

    if isinstance(exc, ServiceError):
        return exc.code, exc.family
    name = type(exc).__name__
    if isinstance(exc, ResourceGovernanceError):
        if "deadline" in str(exc).lower() or "timeout" in str(exc).lower():
            return "JOB_DEADLINE_EXCEEDED", "TIMEOUT"
        return "RESOURCE_BUDGET_EXCEEDED", "RESOURCE"
    if is_fail_closed_error(exc):
        if "PIT" in name or "pit" in str(exc)[:120].lower():
            return "PIT_VIOLATION", "PIT"
        return "SOURCE_EMPTY", "SOURCE"
    if "Timeout" in name or "Deadline" in name:
        return "JOB_DEADLINE_EXCEEDED", "TIMEOUT"
    if "Cancelled" in name or "CancelledError" in name:
        return "JOB_CANCELLED", "CANCEL"
    if name in {"ValueError", "TypeError", "KeyError", "DSLParseError"}:
        return "VALIDATION_FAILED", "VALIDATION"
    if "DQ" in name or "dq" in name[:4].lower():
        return "OUTPUT_DOMAIN_VIOLATION", "DQ"
    if "sql" in name.lower() or "clickhouse" in name.lower() or "duckdb" in name.lower():
        return "BACKEND_EXECUTION_FAILED", "BACKEND"
    return "INTERNAL_ERROR", "INTERNAL"


def json_dumps_redacted(payload: Any, **kwargs: Any) -> str:
    return json.dumps(redact_config(payload) if isinstance(payload, dict) else payload, **kwargs)
