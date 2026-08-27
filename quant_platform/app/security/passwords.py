"""Password hashing — scrypt with constant-time compare.

QRP-P1. Uses stdlib ``hashlib.scrypt`` (preferred over argon2, which is not
installed). Stored format::

    scrypt$n$r$p$salt$hash

where ``n``/``r``/``p`` are the scrypt cost parameters, ``salt`` is a random
16-byte salt, and ``hash`` is the 32-byte derived key — all base64-encoded
(``n``/``r``/``p`` as plain integers). Verification recomputes with the stored
parameters and compares in constant time.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

# scrypt cost parameters (OWASP-recommended defaults for interactive logins).
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32

_PREFIX = "scrypt$"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def hash_password(password: str, *, n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P) -> str:
    """Hash ``password`` with scrypt; return ``scrypt$n$r$p$salt$hash``."""
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=_KEY_BYTES,
    )
    return f"{_PREFIX}{n}${r}${p}${_b64(salt)}${_b64(dk)}"


def _parse(stored: str) -> tuple[int, int, int, bytes, bytes]:
    if not stored.startswith(_PREFIX):
        raise ValueError("unsupported password hash format (expected scrypt$...)")
    body = stored[len(_PREFIX):]
    parts = body.split("$")
    if len(parts) != 5:
        raise ValueError("malformed scrypt hash")
    n, r, p = (int(parts[0]), int(parts[1]), int(parts[2]))
    salt = _unb64(parts[3])
    dk = _unb64(parts[4])
    return n, r, p, salt, dk


def verify_password(password: str, stored: str) -> bool:
    """Return True iff ``password`` matches ``stored`` (constant-time compare)."""
    n, r, p, salt, expected = _parse(stored)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=len(expected),
    )
    return hmac.compare_digest(dk, expected)
