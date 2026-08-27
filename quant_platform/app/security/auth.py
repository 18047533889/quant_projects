"""Auth — session token issue/validate.

QRP-P1. Phase 1 uses a signed HTTP-only cookie token: an HMAC-SHA256 signature
over the session id + principal id + expiry, using a server secret (stdlib
``hmac``/``hashlib``). The session row lives in the ``sessions`` table (metadata
DB). Validation checks the signature, the expiry, and that the session is not
revoked.

Token format: ``<session_id>.<principal_id>.<expiry_epoch>.<hex_hmac>``
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Any

# Default session lifetime (seconds).
SESSION_TTL_SECONDS = 8 * 60 * 60  # 8h


@dataclass(frozen=True)
class SessionToken:
    """A validated session token."""

    session_id: str
    principal_id: str
    expires_at: int  # unix epoch seconds
    signature: str


def _sign(secret: bytes, session_id: str, principal_id: str, expires_at: int) -> str:
    msg = f"{session_id}.{principal_id}.{expires_at}".encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def create_session_token(
    secret: bytes,
    session_id: str,
    principal_id: str,
    expires_at: int,
) -> str:
    """Build the signed token string."""
    sig = _sign(secret, session_id, principal_id, expires_at)
    return f"{session_id}.{principal_id}.{expires_at}.{sig}"


def parse_session_token(secret: bytes, token: str) -> SessionToken | None:
    """Validate ``token``; return a ``SessionToken`` or None if invalid/expired."""
    parts = token.split(".")
    if len(parts) != 4:
        return None
    session_id, principal_id, expires_at_str, sig = parts
    try:
        expires_at = int(expires_at_str)
    except ValueError:
        return None
    expected = _sign(secret, session_id, principal_id, expires_at)
    if not hmac.compare_digest(expected, sig):
        return None
    if time.time() > expires_at:
        return None
    return SessionToken(session_id, principal_id, expires_at, sig)


class SessionManager:
    """Issue/validate/revoke sessions against the metadata DB.

    ``db`` must implement the ``Db`` Protocol (``query``/``execute``/
    ``transaction``). ``secret`` is the server HMAC secret.
    """

    def __init__(self, db: Any, secret: bytes, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        self._db = db
        self._secret = secret
        self._ttl = ttl_seconds

    def issue(self, principal_id: str) -> str:
        """Create a session row and return the signed token string."""
        session_id = secrets.token_hex(16)
        now = int(time.time())
        expires_at = now + self._ttl
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, principal_id, created_at, expires_at, revoked) "
                "VALUES (?, ?, ?, ?, 0)",
                (session_id, principal_id, now, expires_at),
            )
        return create_session_token(self._secret, session_id, principal_id, expires_at)

    def validate(self, token: str) -> SessionToken | None:
        """Return the ``SessionToken`` if the signature is valid, unexpired, and
        the session row exists and is not revoked; else None."""
        parsed = parse_session_token(self._secret, token)
        if parsed is None:
            return None
        rows = self._db.query(
            "SELECT revoked FROM sessions WHERE session_id = ?",
            (parsed.session_id,),
        )
        if not rows:
            return None
        if rows[0]["revoked"]:
            return None
        return parsed

    def revoke(self, token: str) -> None:
        """Revoke the session behind ``token`` (no-op if invalid)."""
        parsed = parse_session_token(self._secret, token)
        if parsed is None:
            return
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE sessions SET revoked = 1 WHERE session_id = ?",
                (parsed.session_id,),
            )
