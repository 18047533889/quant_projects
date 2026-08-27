"""Platform security layer — passwords, auth sessions, RBAC.

QRP-P1. Pure stdlib (``hashlib``/``hmac``/``secrets``) plus the contracts RBAC
vocabulary. No third-party auth deps.
"""

from __future__ import annotations

from .passwords import hash_password, verify_password
from .auth import (
    SessionManager,
    SessionToken,
    create_session_token,
    parse_session_token,
)
from .rbac import (
    AuthorizationError,
    PermissionDenied,
    RBAC,
    effective_permissions,
    resolve_principal_permissions,
)

__all__ = [
    "hash_password",
    "verify_password",
    "SessionManager",
    "SessionToken",
    "create_session_token",
    "parse_session_token",
    "AuthorizationError",
    "PermissionDenied",
    "RBAC",
    "effective_permissions",
    "resolve_principal_permissions",
]
