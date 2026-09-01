"""FactorLibraryVersion — platform projection of the domain library artifact.

P0-PLAT-002: a library version BODY is immutable.  ``factor_assets/library_governance``
is the **sole domain authority** for library semantics — it owns the immutable
``FactorLibraryVersionArtifact``, the lifecycle state machine and the active
pointer.  The platform's job here is persistence / projection of the domain's
decisions:

- ``LibraryMembership`` — a PURE-DTO projection of the domain membership (never
  an alternate membership semantic).
- ``FactorLibraryVersionView`` — read-model of the domain version artifact; the
  ``status`` field is a *rendered* lifecycle state, not a mutable field on an
  immutable artifact.
- ``LibraryLifecycleEvent`` / ``LibraryActivePointer`` / ``LibraryPromotionRecord``
  — the append-only event log + the pointer the platform persists.  Promotion
  NEVER rebuilds the same ``library_version_id`` with a different status (that
  would violate immutable-version semantics); it appends an event and flips the
  pointer.

``promote_library_version`` / ``rollback_library_version`` here are a narrow
platform-shaped transition helper that mirrors what the domain enforces — they
return a NEW view + pointer + record, never mutating a version body.  Callers
that need the authoritative transition go through ``factor_assets/library_governance``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "LibraryStatus",
    "LibraryPromotionStep",
    "LibraryMembership",
    "FactorLibraryVersionView",
    "LibraryActivePointer",
    "LibraryPromotionRecord",
    "LibraryLifecycleEvent",
    "promote_library_version",
    "rollback_library_version",
]


class LibraryStatus(enum.Enum):
    """Lifecycle status of a library version (domain lifecycle; platform records)."""

    CANDIDATE = "CANDIDATE"
    SHADOW = "SHADOW"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


class LibraryPromotionStep(enum.Enum):
    """Ordered promotion ladder for a library version (spec §16)."""

    CANDIDATE = "CANDIDATE"
    SHADOW = "SHADOW"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"

    @classmethod
    def next(cls, step: "LibraryPromotionStep") -> "LibraryPromotionStep | None":
        order = list(cls)
        idx = order.index(step)
        if idx + 1 >= len(order):
            return None
        return order[idx + 1]


@dataclass(frozen=True)
class LibraryMembership:
    """A library member (PURE-DTO projection) — not just a factor_id (spec §15).

    ``weight_in_model`` is deliberately ABSENT: it belongs to ``ModelVersion``.
    """

    factor_definition_id: str
    selected_treatment_id: str | None = None
    orientation: str | None = None
    cluster_id: str | None = None
    representative_of: str | None = None
    similarity_ref: str | None = None
    health_state_ref: str | None = None
    evidence_ref: str | None = None
    assembly_score: float | None = None
    selection_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.factor_definition_id:
            raise ValueError("factor_definition_id is required")


@dataclass(frozen=True)
class FactorLibraryVersionView:
    """Read-model projection of the domain ``FactorLibraryVersionArtifact``.

    The version body is immutable; ``status`` is the *rendered* lifecycle state
    the domain last ordered for this version id (projected from
    ``LibraryLifecycleEvent``).  It is NOT a second mutable copy of the domain
    status.
    """

    library_version_id: str
    logical_library_id: str
    cluster_set_version_id: str
    members: tuple[LibraryMembership, ...] = ()
    policy_hash: str = ""
    evidence_snapshot: str = ""
    status: str = "CANDIDATE"  # CANDIDATE/SHADOW/APPROVED/PRODUCTION/RETIRED
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.library_version_id:
            raise ValueError("library_version_id is required")
        if not self.logical_library_id:
            raise ValueError("logical_library_id is required")
        if not self.cluster_set_version_id:
            raise ValueError("cluster_set_version_id is required")
        if self.status not in {s.value for s in LibraryStatus}:
            raise ValueError(f"unknown library status: {self.status!r}")


@dataclass(frozen=True)
class LibraryActivePointer:
    """The active pointer the platform persists (CAS-updated, spec §53.8).

    The pointer is the ONLY mutable part of the library versioning model. It
    records which version is currently PRODUCTION (or the highest promoted step)
    for a logical library. Versions themselves are immutable and never deleted.
    """

    logical_library_id: str
    active_version_id: str
    active_status: str = "PRODUCTION"
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.logical_library_id:
            raise ValueError("logical_library_id is required")
        if not self.active_version_id:
            raise ValueError("active_version_id is required")
        if self.active_status not in {s.value for s in LibraryStatus}:
            raise ValueError(f"unknown active status: {self.active_status!r}")


@dataclass(frozen=True)
class LibraryPromotionRecord:
    """Audit record of one promotion/rollback step (spec §53.8)."""

    logical_library_id: str
    version_id: str
    from_status: str
    to_status: str
    actor_principal_id: str = ""
    reason: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.logical_library_id:
            raise ValueError("logical_library_id is required")
        if not self.version_id:
            raise ValueError("version_id is required")


@dataclass(frozen=True)
class LibraryLifecycleEvent:
    """Append-only event for one library version status transition (P0-PLAT-002)."""

    logical_library_id: str
    version_id: str
    event_kind: str  # CREATED / PROMOTED / RETIRED / ROLLED_BACK
    from_status: str | None = None
    to_status: str | None = None
    actor_principal_id: str = ""
    reason: str = ""
    created_at: datetime | None = None


def promote_library_version(
    version: FactorLibraryVersionView,
    pointer: LibraryActivePointer,
    actor_principal_id: str = "",
    reason: str = "",
) -> tuple[FactorLibraryVersionView, LibraryActivePointer, LibraryPromotionRecord]:
    """Advance a library version one promotion step and flip the active pointer.

    Returns ``(new_view, new_pointer, record)``.  The version body is NOT
    rebuilt with a different status — a NEW view is produced for rendering and
    the pointer is updated; the appended ``LibraryPromotionRecord`` is what the
    platform persists (as a ``LibraryLifecycleEvent``).  History is preserved —
    the old version object is untouched.
    """
    current = LibraryPromotionStep(version.status)
    nxt = LibraryPromotionStep.next(current)
    if nxt is None:
        raise ValueError(
            f"version {version.library_version_id!r} is already at terminal status "
            f"{version.status!r}; cannot promote further"
        )
    new_view = FactorLibraryVersionView(
        library_version_id=version.library_version_id,
        logical_library_id=version.logical_library_id,
        cluster_set_version_id=version.cluster_set_version_id,
        members=version.members,
        policy_hash=version.policy_hash,
        evidence_snapshot=version.evidence_snapshot,
        status=nxt.value,
        created_at=version.created_at,
    )
    new_pointer = LibraryActivePointer(
        logical_library_id=pointer.logical_library_id,
        active_version_id=version.library_version_id,
        active_status=nxt.value,
    )
    record = LibraryPromotionRecord(
        logical_library_id=version.logical_library_id,
        version_id=version.library_version_id,
        from_status=version.status,
        to_status=nxt.value,
        actor_principal_id=actor_principal_id,
        reason=reason,
    )
    return new_view, new_pointer, record


def rollback_library_version(
    pointer: LibraryActivePointer,
    target_version_id: str,
    actor_principal_id: str = "",
    reason: str = "",
) -> tuple[LibraryActivePointer, LibraryPromotionRecord]:
    """Roll the active pointer back to a prior version (spec §53.8).

    The target version is NOT deleted — only the pointer moves. The previously
    active version (e.g. ``v103``) remains in history.
    """
    if target_version_id == pointer.active_version_id:
        raise ValueError(
            f"target {target_version_id!r} is already the active version; "
            "nothing to roll back"
        )
    new_pointer = LibraryActivePointer(
        logical_library_id=pointer.logical_library_id,
        active_version_id=target_version_id,
        active_status="PRODUCTION",
    )
    record = LibraryPromotionRecord(
        logical_library_id=pointer.logical_library_id,
        version_id=target_version_id,
        from_status=pointer.active_status,
        to_status="PRODUCTION",
        actor_principal_id=actor_principal_id,
        reason=reason,
    )
    return new_pointer, record
