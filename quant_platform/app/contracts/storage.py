"""Storage ports — ArtifactPublisher / ArtifactResolver / ArtifactStoragePort.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.11 (spec §22).
PURE stdlib ``typing.Protocol``.

The platform does NOT own a second ObjectStore authority. ``data_access`` owns the
mature ObjectStore. This module declares *platform-side ports* that will be
implemented later via an adapter over DataAccess's public ObjectStore — that
adapter is a separate task (QRP-P0R-C5). Only the contract types live here.

- ``ArtifactStoragePort`` — minimal durable-byte operations (put/get/head/delete).
- ``ArtifactPublisher`` — publishes bytes, computes ``content_hash``, uploads,
  HEAD-verifies, returns an ``ArtifactRef``.
- ``ArtifactResolver`` — download/open + ``content_hash`` verify.
- ``ArtifactMaterializer`` — download + verify + write to a local path.
- ``ArtifactAccessGate`` / ``FormulaAccessGuard`` / ``AuditEvent`` — artifact
  security classification + formula-endpoint authorization + audit (spec §23).
- ``LocalArtifactCache`` — local disk cache keyed by content_hash.

``ObjectStore``/``ObjectMetadata`` (the old redefinition) is kept ONLY as a
deprecated alias for backward-compat, because nothing outside this package imports
it — we check imports first (grep confirms only ``data_access`` owns an ObjectStore).
If nothing imports it, we keep the alias marked deprecated; it will be removed in
the C5 adapter task.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .artifact_ref import ArtifactRef
from .rbac import Permission, SecurityClassification

__all__ = [
    "ArtifactStoragePort",
    "ArtifactPublisher",
    "ArtifactResolver",
    "ArtifactMaterializer",
    "LocalArtifactCache",
    "CacheEvictionPolicy",
    "ArtifactAccessGate",
    "FormulaAccessGuard",
    "AuditEvent",
    "ObjectStore",
    "ObjectMetadata",
    "GCObject",
    "GCRootSnapshot",
    "GCDryRunPlan",
    "GCTombstoneClaim",
    "TombstoneClaimStatus",
    "DeletionStatus",
    "DeletionReceipt",
    "GCReferenceAuthority",
]


@dataclass(frozen=True)
class GCObject:
    object_id: str
    object_version: str
    storage_uri: str
    content_hash: str
    lifecycle_state: str
    terminal: bool
    size_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("object_id", "object_version", "storage_uri", "content_hash", "lifecycle_state"):
            if not getattr(self, name):
                raise ValueError(f"GCObject.{name} is required")
        if self.size_bytes < 0:
            raise ValueError("GCObject.size_bytes cannot be negative")

    @property
    def key(self) -> tuple[str, str]:
        return self.object_id, self.object_version


@dataclass(frozen=True)
class GCRootSnapshot:
    epoch: int
    objects: tuple[GCObject, ...]
    references: Mapping[tuple[str, str], tuple[tuple[str, str], ...]]
    production_roots: frozenset[tuple[str, str]] = frozenset()
    approved_release_roots: frozenset[tuple[str, str]] = frozenset()
    active_read_roots: frozenset[tuple[str, str]] = frozenset()
    retryable_job_roots: frozenset[tuple[str, str]] = frozenset()
    retained_research_roots: frozenset[tuple[str, str]] = frozenset()
    rollback_roots: frozenset[tuple[str, str]] = frozenset()
    pending_label_roots: frozenset[tuple[str, str]] = frozenset()

    @property
    def roots(self) -> frozenset[tuple[str, str]]:
        return frozenset().union(
            self.production_roots, self.approved_release_roots, self.active_read_roots,
            self.retryable_job_roots, self.retained_research_roots, self.rollback_roots,
            self.pending_label_roots,
        )


@dataclass(frozen=True)
class GCDryRunPlan:
    snapshot_epoch: int
    candidates: tuple[GCObject, ...]
    live_keys: frozenset[tuple[str, str]]
    estimated_reclaim_bytes: int


class TombstoneClaimStatus(str, enum.Enum):
    CLAIMED = "CLAIMED"
    ROOTED = "ROOTED"
    LEASED = "LEASED"
    EPOCH_CHANGED = "EPOCH_CHANGED"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"


@dataclass(frozen=True)
class GCTombstoneClaim:
    status: TombstoneClaimStatus
    object_id: str
    object_version: str
    claim_id: str | None = None
    claimed_epoch: int | None = None
    reason: str = ""


class DeletionStatus(str, enum.Enum):
    SKIPPED_PROTECTED = "SKIPPED_PROTECTED"
    LOGICAL_DELETED = "LOGICAL_DELETED"
    PHYSICAL_DELETED = "PHYSICAL_DELETED"
    PROVIDER_RETENTION_PENDING = "PROVIDER_RETENTION_PENDING"


@dataclass(frozen=True)
class DeletionReceipt:
    object_id: str
    object_version: str
    storage_uri: str
    content_hash: str
    status: DeletionStatus
    deleted_at: datetime
    claim_id: str | None
    cache_evicted: bool
    head_verified_absent: bool
    retained_reason: str = ""


@runtime_checkable
class GCReferenceAuthority(Protocol):
    """Existing registry transaction seam; implementations own refs and tombstones."""

    def snapshot_for_gc(self) -> GCRootSnapshot: ...

    def claim_gc_tombstone(
        self, object_id: str, object_version: str, *, expected_epoch: int
    ) -> GCTombstoneClaim:
        """Atomically recheck roots/leases/epoch and persist an idempotent tombstone."""
        ...

    def record_deletion_receipt(self, receipt: DeletionReceipt) -> DeletionReceipt:
        """Persist or return the immutable receipt for this exact object version."""
        ...

    def get_deletion_receipt(
        self, object_id: str, object_version: str
    ) -> DeletionReceipt | None: ...


@runtime_checkable
class ArtifactStoragePort(Protocol):
    """Minimal durable-byte operations (adapter over DataAccess ObjectStore)."""

    def put(self, uri: str, data: bytes) -> "ObjectMetadata":
        """Upload bytes; two-phase publish semantics (spec §7.4)."""
        ...

    def get(self, uri: str) -> bytes:
        """Download the full object."""
        ...

    def head(self, uri: str) -> "ObjectMetadata":
        """Return object metadata (content_hash / size / etag)."""
        ...

    def delete(self, uri: str) -> None:
        """Explicit deletion; never for published artifacts."""
        ...


@runtime_checkable
class ArtifactPublisher(Protocol):
    """Publishes bytes → computes content_hash → uploads → HEAD-verifies →
    returns ArtifactRef (spec §7.4)."""

    def publish(self, artifact: ArtifactRef, data: bytes) -> ArtifactRef:
        """Publish ``data`` for ``artifact``; computes/verifies content_hash.

        Returns the resolved ``ArtifactRef`` (content_hash set from actual bytes).
        """
        ...


@runtime_checkable
class ArtifactResolver(Protocol):
    """Download/open + content_hash verify (spec §22)."""

    def open(self, artifact: ArtifactRef) -> bytes:
        """Download bytes for ``artifact`` and verify content_hash; fail closed."""
        ...

    def verify(self, artifact: ArtifactRef, data: bytes) -> bool:
        """True iff ``data``'s sha256 equals ``artifact.content_hash``."""
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

    def evict(self, content_hash: str) -> bool:
        """Idempotently remove one exact content version."""
        ...


@runtime_checkable
class ObjectMetadata(Protocol):
    """Metadata returned by storage ``head`` (spec §22)."""

    content_hash: str
    size_bytes: int
    etag: str | None


# ---------------------------------------------------------------------------
# Artifact security classification + access gate (spec §5.8, §23)
# ---------------------------------------------------------------------------


@runtime_checkable
class ArtifactAccessGate(Protocol):
    """Authorization gate for artifact access.

    ``authorize`` returns True iff the principal may exercise ``permission`` on a
    resource whose ``security_classification`` is ``classification``. The RBAC
    agent wires this to the platform's permission/principal model (see
    ``quant_platform.app.security.rbac.RBAC``); a minimal pure implementation is
    provided in :mod:`quant_platform.app.storage.access_guard`.
    """

    def authorize(
        self,
        principal: Any,
        permission: Permission,
        classification: SecurityClassification,
        resource: Any = None,
    ) -> bool:
        """True iff ``principal`` may exercise ``permission`` at ``classification``."""
        ...


@dataclass(frozen=True)
class AuditEvent:
    """Audit record for a guarded access decision (spec §23)."""

    principal_id: str
    team: str | None
    role: str | None
    permission: str
    resource: str
    request_id: str | None
    security_digest: str | None
    credential_scope_id: str | None
    action: str
    result: str  # "ALLOW" | "DENY"
    timestamp: datetime


@runtime_checkable
class FormulaAccessGuard(Protocol):
    """Guard for ``GET /factors/{id}/formula`` (spec §23).

    ``guard`` returns an ``AuditEvent`` describing the decision. A denied decision
    must map to HTTP 403 (never 200 + null). The guard MUST record an audit event
    on every invocation (allow or deny).
    """

    def guard(
        self,
        *,
        principal: Any,
        factor_id: str,
        request_id: str | None = None,
        security_digest: str | None = None,
        credential_scope_id: str | None = None,
    ) -> AuditEvent:
        """Authorize formula read; returns the audit event for the decision."""
        ...


# ---------------------------------------------------------------------------
# materializer (spec §22)
# ---------------------------------------------------------------------------


@runtime_checkable
class ArtifactMaterializer(Protocol):
    """Materializes a published artifact to a local path (spec §22).

    ``materialize`` resolves the artifact, verifies its content_hash, and writes
    the bytes to ``dest``. Returns the destination path. Fails closed on any
    content_hash mismatch.
    """

    def materialize(self, artifact: ArtifactRef, dest: str) -> str:
        """Download + verify + write ``artifact`` to ``dest``; return ``dest``."""
        ...


# DEPRECATED alias: the platform must NOT become a second ObjectStore authority.
# DataAccess owns ObjectStore; this alias is retained only for any legacy import
# and will be removed in the C5 adapter task (QRP-P0R-C5).
ObjectStore = ArtifactStoragePort
