"""Cluster / Library — platform PROJECTION refs (View layer only).

P0-PLAT-001 / P0-FA-010: the platform is a projection consumer.  ``FactorAssets``
(cluster_governance / library_governance) is the **single domain authority** for
cluster / factor-library semantics — it owns the artifacts (SimilarityGraphVersion,
ClusterSetVersionArtifact, LogicalCluster, ClusterVersionArtifact,
FactorLibraryVersionArtifact), the versioning, the membership semantics and the
lifecycle.  The platform must never re-define those semantics.

What the platform is allowed to define here — and only here — is the *ref /
view* shape it carries for API projection, DB read models and frontend
responses:

- ``ClusterSetVersionRef`` / ``ClusterVersionRef`` / ``FactorLibraryVersionRef``
  — opaque refs that carry the **domain package's own digest** (content hash of
  the domain artifact) plus the slot values the domain hashed.  The platform
  validates FORMAT only; it never recomputes / re-negotiates the digest.
- ``ClusterSetVersionView`` / ``ClusterVersionView`` / ``FactorLibraryVersionView``
  — read-model projections of domain artifacts for UI rendering.  Views are
  derived from domain artifacts by a projection adapter; they are not an
  alternate authority.

Library *lifecycle* (CANDIDATE -> SHADOW -> APPROVED -> PRODUCTION -> RETIRED)
and the active pointer live in the domain (``factor_assets/library_governance``);
the platform DB stores the pointer the domain writes.  P0-PLAT-002: a library
version body is immutable — status transitions append ``LibraryLifecycleEvent``
records, they never mutate a version's status.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from .identities import sha256_hex

__all__ = [
    "ClusterConversionType",
    "ClusterSetVersionRef",
    "ClusterVersionRef",
    "FactorLibraryVersionRef",
    "ClusterSetVersionView",
    "ClusterVersionView",
    "FactorLibraryVersionView",
    "LibraryStatus",
    "LibraryLifecycleEvent",
    "LibraryActivePointer",
    "LibraryPromotionRecord",
    "promote_library_version",
    "rollback_library_version",
]


class ClusterConversionType(enum.Enum):
    """Cluster-lineage transition label — carried, never re-derived (spec §14)."""

    UNCHANGED = "UNCHANGED"
    MIGRATED = "MIGRATED"
    SPLIT = "SPLIT"
    MERGED = "MERGED"
    NEW = "NEW"
    DISSOLVED = "DISSOLVED"


# --------------------------------------------------------------------------- #
# Carried refs (domain-owned digests)
# --------------------------------------------------------------------------- #


def _require_digest(value: str, label: str) -> str:
    return sha256_hex(value, label)


@dataclass(frozen=True)
class ClusterSetVersionRef:
    """Carried ref to the domain ``ClusterSetVersionArtifact`` digest."""

    cluster_set_version_id: str
    digest: str
    similarity_graph_version_ref: str = ""
    policy_hash: str = ""

    def __post_init__(self) -> None:
        if not self.cluster_set_version_id:
            raise ValueError("cluster_set_version_id is required")
        object.__setattr__(self, "digest", _require_digest(self.digest, "cluster_set digest"))


@dataclass(frozen=True)
class ClusterVersionRef:
    """Carried ref to the domain ``ClusterVersionArtifact`` digest."""

    cluster_version_id: str
    digest: str
    logical_cluster_id: str = ""
    cluster_set_version_id: str = ""
    #: Rendered algorithm-cluster label from the domain artifact (projection).
    algorithm_cluster_label: str = ""

    def __post_init__(self) -> None:
        if not self.cluster_version_id:
            raise ValueError("cluster_version_id is required")
        object.__setattr__(self, "digest", _require_digest(self.digest, "cluster_version digest"))


@dataclass(frozen=True)
class FactorLibraryVersionRef:
    """Carried ref to the domain ``FactorLibraryVersionArtifact`` digest."""

    library_version_id: str
    digest: str
    logical_library_id: str = ""
    cluster_set_version_id: str = ""

    def __post_init__(self) -> None:
        if not self.library_version_id:
            raise ValueError("library_version_id is required")
        object.__setattr__(self, "digest", _require_digest(self.digest, "factor_library_version digest"))


# --------------------------------------------------------------------------- #
# Read-model views (projection of domain artifacts; NOT an authority)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClusterSetVersionView:
    """Read-model projection of a domain ClusterSetVersionArtifact (UI/API)."""

    cluster_set_version_id: str
    algorithm: str
    backend: str = ""
    seed: int = 0
    resolution: float | None = None
    policy_hash: str = ""
    similarity_graph_version_ref: str = ""
    created_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClusterVersionView:
    """Read-model projection of a domain ClusterVersionArtifact (UI/API)."""

    cluster_version_id: str
    logical_cluster_id: str
    cluster_set_version_id: str
    algorithm_cluster_label: str
    member_factor_ids: tuple[str, ...] = ()
    conversion_type: ClusterConversionType | None = None
    created_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorLibraryVersionView:
    """Read-model projection of a domain FactorLibraryVersionArtifact (UI/API).

    Carries the domain artifact's status for rendering only; the authoritative
    status lives with ``factor_assets/library_governance``.  Views are derived,
    never written as a second truth.
    """

    library_version_id: str
    logical_library_id: str
    cluster_set_version_id: str
    status: str = "CANDIDATE"
    member_count: int = 0
    policy_hash: str = ""
    evidence_snapshot: str = ""
    created_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Lifecycle events + active pointer (platform persistence for domain-ordered
# transitions; the DOMAIN decides, the platform records).
# --------------------------------------------------------------------------- #


class LibraryStatus(enum.Enum):
    """Library lifecycle statuses (domain lifecycle; platform records events)."""

    CANDIDATE = "CANDIDATE"
    SHADOW = "SHADOW"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"

    @classmethod
    def next(cls, step: "LibraryStatus") -> "LibraryStatus | None":
        order = [cls.CANDIDATE, cls.SHADOW, cls.APPROVED, cls.PRODUCTION]
        if step not in order:
            return None
        idx = order.index(step)
        return order[idx + 1] if idx + 1 < len(order) else None


@dataclass(frozen=True)
class LibraryLifecycleEvent:
    """Append-only record of one library version status transition.

    P0-PLAT-002: a library version BODY is immutable — promotion/rollback append
    a new :class:`LibraryLifecycleEvent` (and optionally flip the active pointer);
    they never rebuild the same ``library_version_id`` with a different status.
    """

    logical_library_id: str
    version_id: str
    event_kind: str  # PROMOTED / RETIRED / ROLLED_BACK / CREATED
    from_status: str | None = None
    to_status: str | None = None
    actor_principal_id: str = ""
    reason: str = ""
    created_at: datetime | None = None


@dataclass(frozen=True)
class LibraryActivePointer:
    """The active production pointer for a logical library (CAS-updated)."""

    logical_library_id: str
    active_version_id: str
    active_status: str = "PRODUCTION"
    updated_at: datetime | None = None


@dataclass(frozen=True)
class LibraryPromotionRecord:
    """Audit record of one promotion/rollback step (kept for compat callers)."""

    logical_library_id: str
    version_id: str
    from_status: str
    to_status: str
    actor_principal_id: str = ""
    reason: str = ""
    created_at: datetime | None = None


def promote_library_version(
    version: FactorLibraryVersionView,
    pointer: LibraryActivePointer,
    actor_principal_id: str = "",
    reason: str = "",
) -> tuple[FactorLibraryVersionView, LibraryActivePointer, LibraryPromotionRecord]:
    """Advance a library version one promotion step (domain-ordered transition).

    The version **body** stays identical — ``status`` on a
    :class:`FactorLibraryVersionView` is a *rendered* view of the domain
    artifact's lifecycle, not a mutable field on an immutable artifact.  This
    helper flips the active pointer and returns a NEW view (for rendering) plus
    the promotion record the platform persists as a :class:`LibraryLifecycleEvent`.

    Callers who need the authoritative transition must go through
    ``factor_assets/library_governance``; this platform helper exists only so
    archived plans / tests keep a narrow, ref-shaped promotion path that never
    mints a second lifecycle authority.
    """
    try:
        current = LibraryStatus(version.status)
    except ValueError:
        raise ValueError(f"unknown library status: {version.status!r}")
    nxt = LibraryStatus.next(current)
    if nxt is None:
        raise ValueError(
            f"version {version.library_version_id!r} is already at terminal status "
            f"{version.status!r}; cannot promote further"
        )
    new_view = FactorLibraryVersionView(
        library_version_id=version.library_version_id,
        logical_library_id=version.logical_library_id,
        cluster_set_version_id=version.cluster_set_version_id,
        status=nxt.value,
        member_count=version.member_count,
        policy_hash=version.policy_hash,
        evidence_snapshot=version.evidence_snapshot,
        created_at=version.created_at,
        metadata=dict(version.metadata),
    )
    new_pointer = LibraryActivePointer(
        logical_library_id=pointer.logical_library_id,
        active_version_id=version.library_version_id,
        active_status=nxt.value,
        updated_at=pointer.updated_at,
    )
    record = LibraryPromotionRecord(
        logical_library_id=version.logical_library_id,
        version_id=version.library_version_id,
        from_status=version.status,
        to_status=nxt.value,
        actor_principal_id=actor_principal_id,
        reason=reason,
        created_at=None,
    )
    return new_view, new_pointer, record


def rollback_library_version(
    pointer: LibraryActivePointer,
    target_version_id: str,
    actor_principal_id: str = "",
    reason: str = "",
) -> tuple[LibraryActivePointer, LibraryPromotionRecord]:
    """Flip the active pointer back to a previously-approved version.

    Rollback only appends a rollback record and re-points the pointer; the old
    version body is never modified or deleted (P0-PLAT-002).
    """
    record = LibraryPromotionRecord(
        logical_library_id=pointer.logical_library_id,
        version_id=target_version_id,
        from_status=pointer.active_status,
        to_status="APPROVED",
        actor_principal_id=actor_principal_id,
        reason=reason,
        created_at=None,
    )
    new_pointer = LibraryActivePointer(
        logical_library_id=pointer.logical_library_id,
        active_version_id=target_version_id,
        active_status="APPROVED",
        updated_at=None,
    )
    return new_pointer, record
