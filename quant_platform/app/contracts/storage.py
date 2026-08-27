"""ObjectStore + LocalArtifactCache Protocols.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.11 (spec §22).
PURE stdlib ``typing.Protocol``. The 1.6TB local disk is not a long-term source
of truth; all durable bytes live in the ObjectStore, and the LocalArtifactCache
is disposable and rebuildable.
"""

from __future__ import annotations

import enum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ObjectStore",
    "LocalArtifactCache",
    "CacheEvictionPolicy",
    "ObjectMetadata",
]


@runtime_checkable
class ObjectMetadata(Protocol):
    """Metadata returned by ``ObjectStore.head`` (spec §22)."""

    content_hash: str
    size_bytes: int
    etag: str | None


@runtime_checkable
class ObjectStore(Protocol):
    """Durable object storage abstraction (spec §22)."""

    def put(self, uri: str, data: bytes) -> ObjectMetadata:
        """Upload bytes; two-phase publish semantics (spec §7.4)."""
        ...

    def get(self, uri: str) -> bytes:
        """Download the full object."""
        ...

    def head(self, uri: str) -> ObjectMetadata:
        """Return object metadata (content_hash / size / etag)."""
        ...

    def delete(self, uri: str) -> None:
        """Explicit deletion; never for published artifacts."""
        ...


class CacheEvictionPolicy(enum.Enum):
    """LocalArtifactCache eviction policy (spec §22)."""

    LRU = "LRU"


@runtime_checkable
class LocalArtifactCache(Protocol):
    """Local disk cache keyed by content_hash (spec §22)."""

    max_bytes: int
    eviction_policy: CacheEvictionPolicy

    def lookup(self, content_hash: str) -> bytes | None:
        """Return cached bytes for ``content_hash`` or None on miss."""
        ...

    def store(self, content_hash: str, data: bytes) -> None:
        """Store bytes, validating checksum; enforce max_bytes + LRU eviction."""
        ...

    def hit_rate(self) -> float:
        """Cache hit-rate metric (spec §22)."""
        ...
