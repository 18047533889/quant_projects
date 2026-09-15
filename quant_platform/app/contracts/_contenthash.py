"""Shared canonical content-hash helpers for the platform contracts DTO layer.

The original content_hash/canonical_str codec is legacy v1 and has structural
collisions. It remains available only for compatibility with persisted keys.
New versioned consumers should use content_hash_v2 and store their hash codec.
The historical rules below describe v1, not a collision-free serialization.

PURE stdlib only. Implements the *semantic-identity* hash rule from
``quant_platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §2 (DRAFT): sha256 over a
canonical, sorted, length-prefixed tuple of semantic fields.

IMPORTANT — this is a **strict canonical codec for semantic identity**, NOT an
object-byte hash. It must be deterministic across processes and reject inputs
that cannot be canonically serialized. Unlike object-byte hashing (which hashes
the raw bytes), this codec canonicalizes *values*; therefore:

- floats must be finite (NaN/Inf raise);
- ``-0.0`` is normalized to ``0.0``;
- datetimes must be timezone-aware and are emitted as UTC ISO-8601 ending in ``Z``
  (naive datetimes raise);
- enums are reduced to their ``.value``;
- dict keys are sorted;
- unsupported types raise (NO ``str(obj)`` fallback, NO ``repr`` that may change
  across Python versions).

This codec is distinct from the object-bytes ``content_hash`` computed by the
artifact publisher/resolver over the COS object bytes (see ``storage.py``).
"""
from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import is_dataclass, fields
from datetime import date, datetime, timezone
from typing import Any, Mapping

__all__ = ["length_prefixed", "canonical_str", "content_hash",
           "canonical_bytes_v2", "content_hash_v2"]


def _canonical_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "naive datetime not allowed in canonical content hash; "
            f"got {value!r} — provide a timezone-aware datetime"
        )
    utc = value.astimezone(timezone.utc)
    return utc.isoformat().replace("+00:00", "Z")


def _canonical_float(value: float) -> str:
    if math.isnan(value):
        raise ValueError("NaN not allowed in canonical content hash")
    if math.isinf(value):
        raise ValueError("Inf not allowed in canonical content hash")
    # Normalize negative zero to positive zero so -0.0 == 0.0 canonically.
    if value == 0.0:
        return "0.0"
    return repr(value)


def canonical_str(value: Any) -> str:
    """Deterministic canonical string form of a scalar / mapping / sequence.

    - enums → their ``.value``;
    - dicts are sorted by canonical key;
    - floats must be finite; ``-0.0`` is normalized to ``0.0``;
    - datetimes must be timezone-aware; emitted as UTC ISO-8601 ending in ``Z``;
    - naive datetimes raise;
    - sequences are ordered (tuple/list);
    - bool/int/None/str are canonical scalars;
    - any other type raises.
    """
    if isinstance(value, enum.Enum):
        return canonical_str(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        # Frozen dataclasses (the DTO layer's bread-and-butter) canonicalize via
        # their ordered fields. This keeps FeatureSetVersion/FeatureMemberRef etc.
        # hashable by value.
        return "{" + ",".join(
            f"{f.name}={canonical_str(getattr(value, f.name))}"
            for f in fields(value)
        ) + "}"
    if isinstance(value, datetime):
        return _canonical_datetime(value)
    if isinstance(value, date):
        # Plain date (not datetime) → ISO date form. datetime is handled above.
        return value.isoformat()
    if isinstance(value, Mapping):
        return "{" + ",".join(
            f"{canonical_str(k)}={canonical_str(v)}"
            for k, v in sorted(value.items(), key=lambda kv: canonical_str(kv[0]))
        ) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(canonical_str(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _canonical_float(value)
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    raise TypeError(
        f"unsupported type for canonical content hash: {type(value).__name__!r} "
        f"(value={value!r}) — no str()/repr() fallback"
    )


def length_prefixed(digest: "hashlib._Hash", value: Any) -> None:
    """Hash a field with length-prefixing so delimiters cannot collide."""
    encoded = canonical_str(value).encode("utf-8")
    digest.update(str(len(encoded)).encode("ascii"))
    digest.update(b":")
    digest.update(encoded)


def content_hash(*fields: Any) -> str:
    """Legacy v1 digest; retained for old records, not safe for new identities.

    Recursive containers and scalar types can collide in the v1 display
    encoding. Migrate each durable consumer explicitly to content_hash_v2;
    do not silently replace already-stored keys or historical validators.
    """
    digest = hashlib.sha256()
    for field in fields:
        length_prefixed(digest, field)
    return digest.hexdigest()


def _frame_v2(tag: bytes, payload: bytes) -> bytes:
    return tag + str(len(payload)).encode("ascii") + b":" + payload


def canonical_bytes_v2(value: Any, *, _depth: int = 0) -> bytes:
    """Typed, recursively framed encoding for explicitly versioned identities.

    Mapping order is irrelevant; sequences retain order and list/tuple kind.
    Enum aliases retain the platform convention of their underlying value.
    Equivalent UTC instants and signed zero normalize; nonfinite floats,
    naive timestamps, cycles/deep nesting and unsupported objects reject.
    This is an identity codec, not a JSON transport serializer.
    """
    if _depth > 64:
        raise ValueError("canonical v2 nesting exceeds 64 levels; cycle or oversized input")
    def encode(item: Any) -> bytes:
        return canonical_bytes_v2(item, _depth=_depth + 1)
    if isinstance(value, enum.Enum):
        return encode(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        payload = encode(cls.__module__ + "." + cls.__qualname__)
        payload += b"".join(encode(f.name) + encode(getattr(value, f.name)) for f in fields(value))
        return _frame_v2(b"D", payload)
    if isinstance(value, datetime):
        return _frame_v2(b"T", _canonical_datetime(value).encode("utf-8"))
    if isinstance(value, date):
        return _frame_v2(b"d", value.isoformat().encode("ascii"))
    if isinstance(value, Mapping):
        pairs = sorted((encode(k), encode(v)) for k, v in value.items())
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate canonical mapping key")
        return _frame_v2(b"M", b"".join(k + v for k, v in pairs))
    if isinstance(value, (tuple, list)):
        return _frame_v2(b"L" if isinstance(value, list) else b"Q",
                         b"".join(encode(item) for item in value))
    if isinstance(value, bool):
        return _frame_v2(b"B", b"1" if value else b"0")
    if isinstance(value, int):
        return _frame_v2(b"I", str(value).encode("ascii"))
    if isinstance(value, float):
        return _frame_v2(b"F", _canonical_float(value).encode("ascii"))
    if value is None:
        return _frame_v2(b"N", b"")
    if isinstance(value, str):
        return _frame_v2(b"S", value.encode("utf-8"))
    raise TypeError(f"unsupported canonical v2 type: {type(value).__name__}")


def content_hash_v2(*values: Any) -> str:
    """SHA-256 in the semantic-v2 domain; callers must persist the codec."""
    digest = hashlib.sha256(b"quant_platform.semantic-hash.v2\0")
    for value in values:
        digest.update(canonical_bytes_v2(value))
    return digest.hexdigest()
