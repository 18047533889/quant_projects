"""
R32-P0-107: CanonicalIdentityEncoder全仓唯一.
R32-P0-108: 关键identity至少128-bit.
DA-P0-010: correctness cache identity 使用 strict + >=128 bit.
DA-P0-011: correctness identity 禁止 repr fallback.
DA-P0-012: set/frozenset typed canonicalization.
DA-P0-013: dict key typed canonicalization.
DA-P0-014: NaN/Infinity/-0.0 explicit identity tokens.

Canonical identity encoding with production fail-closed semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from data_access.core.exceptions import ValidationError


class CanonicalIdentityEncoder:
    """Canonical identity encoder - single source of truth for identity hashing.

    R32-P0-107: 统一:
    - list/tuple保序
    - set无序
    - map key sort
    - timezone-aware datetime
    - enum type+value
    - bytes deterministic
    - dataclass canonical fields

    Production禁止repr fallback.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict

    def encode(self, value: Any) -> str:
        """Encode value to canonical string representation.

        Args:
            value: Value to encode

        Returns:
            Canonical string representation

        Raises:
            ValidationError: In strict mode when repr fallback needed
        """
        canonical = self._encode_value(value)
        return json.dumps(canonical, sort_keys=True, ensure_ascii=True)

    def hash_identity(self, value: Any, *, bits: int = 256) -> str:
        """Hash value to identity digest.

        Args:
            value: Value to hash
            bits: Hash bits (64, 128, 256, 384, 512)

        Returns:
            Hex digest of specified bit length

        Raises:
            ValidationError: If bits not supported or encoding fails
        """
        if bits not in {64, 128, 256, 384, 512}:
            raise ValidationError(f"Unsupported hash bits: {bits}")

        canonical = self.encode(value)

        if bits == 64:
            # MD5 truncated to 64 bits for ephemeral cache keys
            # NOT for correctness/security identities
            full_digest = hashlib.md5(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()
            digest = full_digest[:16]  # 64 bits = 16 hex chars
        elif bits == 128:
            # R32-P0-108: 128-bit minimum for non-display identities
            # MD5 used only for cache key generation, not cryptographic security
            digest = hashlib.md5(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()
        elif bits == 256:
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        elif bits == 384:
            digest = hashlib.sha384(canonical.encode("utf-8")).hexdigest()
        else:  # 512
            digest = hashlib.sha512(canonical.encode("utf-8")).hexdigest()

        return digest

    def _encode_value(self, value: Any) -> Any:
        """Encode single value to canonical form.

        DA-P0-014: NaN/Infinity/-0.0 handled explicitly.
        DA-P0-012: set/frozenset use typed canonicalization.
        DA-P0-013: dict keys use typed canonicalization.
        """
        # None
        if value is None:
            return None

        # Boolean (before int, since bool is int subclass)
        if isinstance(value, bool):
            return value

        # Numeric types
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            # DA-P0-014: Explicit tokens for special float values
            if math.isnan(value):
                return {"__float_special__": "NaN"}
            elif math.isinf(value):
                if value > 0:
                    return {"__float_special__": "+Inf"}
                else:
                    return {"__float_special__": "-Inf"}
            elif value == 0.0:
                # Distinguish -0.0 from +0.0
                if math.copysign(1.0, value) < 0:
                    return {"__float_special__": "-0.0"}
                else:
                    return 0.0
            else:
                # Regular float: use exact representation
                return {"__float__": value}

        # String
        if isinstance(value, str):
            return value

        # Bytes - deterministic hex encoding
        if isinstance(value, bytes):
            return {"__bytes__": value.hex()}

        # Datetime - canonical ISO with timezone
        if isinstance(value, datetime):
            # Ensure timezone-aware
            if value.tzinfo is None:
                if self.strict:
                    raise ValidationError(
                        "R32-P0-107: datetime must be timezone-aware in strict mode"
                    )
                # Assume UTC for naive datetime
                value = value.replace(tzinfo=timezone.utc)
            return {"__datetime__": value.isoformat()}

        # Enum - type + value
        if isinstance(value, Enum):
            return {
                "__enum__": {
                    "type": f"{value.__class__.__module__}.{value.__class__.__name__}",
                    "value": value.value,
                }
            }

        # List/tuple - preserve order
        if isinstance(value, (list, tuple)):
            return [self._encode_value(v) for v in value]

        # Set/frozenset - sorted with type preservation
        # DA-P0-012: typed canonicalization for sets
        if isinstance(value, (set, frozenset)):
            # Encode each item with type tag, then sort by canonical bytes
            typed_items = []
            for item in value:
                encoded = self._encode_value(item)
                # Create sortable tuple: (type_tag, encoded_value)
                type_tag = type(item).__name__
                typed_items.append((type_tag, encoded))

            # Sort by JSON-encoded tuple for deterministic order
            try:
                sorted_items = sorted(typed_items, key=lambda x: json.dumps(x, sort_keys=True))
            except TypeError:
                # Fallback: stringify for comparison
                sorted_items = sorted(typed_items, key=lambda x: str(x))

            return {"__set__": sorted_items}

        # Dict/mapping - sort keys with type preservation
        # DA-P0-013: dict key typed canonicalization
        if isinstance(value, dict):
            # Encode both keys and values with type information
            typed_items = []
            for k, v in value.items():
                encoded_key = self._encode_value(k)
                encoded_value = self._encode_value(v)
                # Preserve key type
                key_type = type(k).__name__
                typed_items.append(((key_type, encoded_key), encoded_value))

            # Sort by canonical key representation
            try:
                sorted_items = sorted(typed_items, key=lambda x: json.dumps(x[0], sort_keys=True))
            except TypeError:
                sorted_items = sorted(typed_items, key=lambda x: str(x[0]))

            return {"__dict__": sorted_items}

        # Dataclass - canonical field order
        if is_dataclass(value):
            return {
                "__dataclass__": {
                    "type": f"{value.__class__.__module__}.{value.__class__.__name__}",
                    "fields": {
                        f.name: self._encode_value(getattr(value, f.name))
                        for f in fields(value)
                    },
                }
            }

        # Fallback: repr in non-strict, error in strict
        # DA-P0-011: correctness identity禁止repr fallback
        if self.strict:
            raise ValidationError(
                f"DA-P0-011: Cannot encode {type(value).__name__} in strict mode. "
                f"Correctness identity禁止repr fallback. "
                f"Supported types: None, bool, int, float, str, bytes, datetime, "
                f"Enum, list, tuple, set, frozenset, dict, dataclass."
            )

        return {"__repr__": repr(value)}


@dataclass(frozen=True)
class IdentityDigest:
    """Identity digest with minimum bit requirement enforcement.

    R32-P0-108: 关键identity至少128-bit.
    DA-P0-010: correctness cache identity也必须≥128-bit + strict.

    cache display id可短; source/experiment/security/policy/artifact/correctness identity
    应≥128 bit,最好full SHA256 internal.
    """

    digest: str
    bits: int
    identity_type: str  # "cache", "correctness", "source", "experiment", "security", "policy", "artifact"

    def __post_init__(self) -> None:
        """Validate digest meets minimum bit requirement."""
        min_bits = self._minimum_bits_for_type()

        # Validate actual digest length
        hex_chars = len(self.digest)
        actual_bits = hex_chars * 4  # Each hex char = 4 bits

        if actual_bits < min_bits:
            raise ValidationError(
                f"DA-P0-010 / R32-P0-108: {self.identity_type} identity requires ≥{min_bits} bits, "
                f"got {actual_bits} bits ({hex_chars} hex chars)"
            )

        if self.bits != actual_bits:
            raise ValidationError(
                f"Declared bits {self.bits} doesn't match digest length "
                f"{actual_bits} bits"
            )

    def _minimum_bits_for_type(self) -> int:
        """Get minimum bit requirement for identity type."""
        if self.identity_type == "cache":
            return 64  # Display IDs can be short (ephemeral performance cache)
        elif self.identity_type in {
            "correctness",  # DA-P0-010: correctness cache identity must be ≥128 bits
            "source",
            "experiment",
            "security",
            "policy",
            "artifact",
        }:
            return 128  # Critical identities must be ≥128 bits
        else:
            return 128  # Default to 128 for unknown types

    def short_display(self, length: int = 8) -> str:
        """Get short display version (first N hex chars)."""
        return self.digest[:length]


def create_identity_encoder(*, strict: bool | None = None) -> CanonicalIdentityEncoder:
    """Create identity encoder with appropriate strictness.

    Args:
        strict: If None, auto-detect from environment

    Returns:
        Configured encoder
    """
    if strict is None:
        import os

        strict = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {
            "1",
            "true",
            "yes",
        }

    return CanonicalIdentityEncoder(strict=strict)


def hash_source_identity(value: Any, *, bits: int = 256) -> IdentityDigest:
    """Hash source identity with minimum 128-bit requirement.

    R32-P0-108: Source identity must be ≥128 bits.
    """
    encoder = create_identity_encoder(strict=True)
    digest = encoder.hash_identity(value, bits=bits)

    return IdentityDigest(
        digest=digest,
        bits=bits,
        identity_type="source",
    )


def hash_experiment_identity(value: Any, *, bits: int = 256) -> IdentityDigest:
    """Hash experiment identity with minimum 128-bit requirement."""
    encoder = create_identity_encoder(strict=True)
    digest = encoder.hash_identity(value, bits=bits)

    return IdentityDigest(
        digest=digest,
        bits=bits,
        identity_type="experiment",
    )


def hash_security_identity(value: Any, *, bits: int = 256) -> IdentityDigest:
    """Hash security identity with minimum 128-bit requirement."""
    encoder = create_identity_encoder(strict=True)
    digest = encoder.hash_identity(value, bits=bits)

    return IdentityDigest(
        digest=digest,
        bits=bits,
        identity_type="security",
    )


def hash_cache_key(value: Any, *, bits: int = 64) -> str:
    """Hash cache key - can be shorter for display purposes.

    NOTE: This is for ephemeral performance cache keys only.
    For correctness cache identity, use hash_correctness_identity() instead.
    """
    encoder = create_identity_encoder(strict=False)
    return encoder.hash_identity(value, bits=max(bits, 64))


def hash_correctness_identity(value: Any, *, bits: int = 256) -> IdentityDigest:
    """Hash correctness cache identity with minimum 128-bit requirement.

    DA-P0-010: Correctness cache identity must be:
    - strict=True (no repr fallback)
    - ≥128 bits (preferably full SHA-256)

    Use this for cache that affects quantitative result correctness:
    - Factor computation result cache
    - Data transformation cache that feeds into factors
    - Any cache where incorrect reuse would produce wrong trading signals
    """
    encoder = create_identity_encoder(strict=True)
    digest = encoder.hash_identity(value, bits=bits)

    return IdentityDigest(
        digest=digest,
        bits=bits,
        identity_type="correctness",
    )


__all__ = [
    "CanonicalIdentityEncoder",
    "IdentityDigest",
    "create_identity_encoder",
    "hash_source_identity",
    "hash_experiment_identity",
    "hash_security_identity",
    "hash_cache_key",
    "hash_correctness_identity",
]
