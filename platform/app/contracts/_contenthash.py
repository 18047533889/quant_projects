"""Shared content-hash helpers for the platform contracts DTO layer.

PURE stdlib only. Implements the content-hash rule from
``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §2 (DRAFT): sha256 over a
canonical, sorted, length-prefixed tuple of semantic fields. The hash is
derived, never self-reported; a caller-supplied hash that does not match the
recomputed value fails closed.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

__all__ = ["length_prefixed", "canonical_str", "content_hash"]


def length_prefixed(digest: "hashlib._Hash", value: Any) -> None:
    """Hash a field with length-prefixing so delimiters cannot collide."""
    encoded = canonical_str(value).encode("utf-8")
    digest.update(str(len(encoded)).encode("ascii"))
    digest.update(b":")
    digest.update(encoded)


def canonical_str(value: Any) -> str:
    """Deterministic string form of a scalar / mapping / sequence.

    - dicts are sorted by key;
    - floats are canonicalized via ``repr`` (stable across processes);
    - datetimes are snapshotted to ISO-8601 UTC;
    - sequences are joined with a length-prefixed separator.
    """
    if isinstance(value, Mapping):
        return "{" + ",".join(
            f"{canonical_str(k)}={canonical_str(v)}" for k, v in sorted(value.items(), key=lambda kv: canonical_str(kv[0]))
        ) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(canonical_str(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if hasattr(value, "isoformat"):  # datetime / date
        return value.isoformat()
    return str(value)


def content_hash(*fields: Any) -> str:
    """sha256 over the given semantic fields (length-prefixed, canonical)."""
    digest = hashlib.sha256()
    for field in fields:
        length_prefixed(digest, field)
    return digest.hexdigest()
