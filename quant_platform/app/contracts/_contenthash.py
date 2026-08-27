"""Shared canonical content-hash helpers for the platform contracts DTO layer.

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

__all__ = ["length_prefixed", "canonical_str", "content_hash"]


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
    """sha256 over the given semantic fields (length-prefixed, canonical)."""
    digest = hashlib.sha256()
    for field in fields:
        length_prefixed(digest, field)
    return digest.hexdigest()
