"""Password hashing tests — scrypt hash/verify, wrong-password fails, format."""

from __future__ import annotations

import pytest

from quant_platform.app.security.passwords import hash_password, verify_password


def test_hash_verify_roundtrip():
    h = hash_password("correct horse battery staple")
    assert h.startswith("scrypt$")
    assert verify_password("correct horse battery staple", h) is True


def test_wrong_password_fails():
    h = hash_password("right-password")
    assert verify_password("wrong-password", h) is False


def test_scrypt_format():
    h = hash_password("pw")
    # scrypt$n$r$p$salt$hash — 6 segments after splitting on '$'.
    parts = h.split("$")
    assert parts[0] == "scrypt"
    assert len(parts) == 6
    # n, r, p are integers.
    int(parts[1]), int(parts[2]), int(parts[3])
    # salt and hash are base64 (non-empty).
    assert parts[4] and parts[5]


def test_salts_are_unique():
    h1 = hash_password("same")
    h2 = hash_password("same")
    assert h1 != h2  # random salt => different hashes


def test_verify_rejects_malformed():
    with pytest.raises(ValueError):
        verify_password("pw", "not-a-scrypt-hash")
    with pytest.raises(ValueError):
        verify_password("pw", "scrypt$1$2$3$onlyfourparts")
