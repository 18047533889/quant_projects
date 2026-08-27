"""FactorLibraryVersion — immutable governance asset set with promotion/rollback.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.7 (spec §15,
§16, §17, §53.8). PURE stdlib frozen dataclasses + enum.

THE FOUR-LAYER VERSIONING MODEL:

    ClusterVersion -> FactorLibraryVersion -> FeatureSetVersion -> ModelVersion

A ``FactorLibraryVersion`` is the *governance asset set*: which factor
definitions, with which selected treatment and orientation, are admitted into a
logical library at a point in time. It is IMMUTABLE — never updated in place.
Promotion/rollback is expressed through an *active pointer* (e.g. ``PRODUCTION``)
that can point to ``FLV_102``; a new ``FLV_103`` moves CANDIDATE -> SHADOW ->
APPROVED -> PRODUCTION. ``v102`` history is preserved; rollback flips the pointer
back to ``v102`` and NEVER deletes ``v103``.

Members are NOT just ``factor_id`` — each is a ``LibraryMembership`` carrying
``factor_definition_id``, ``selected_treatment_id``, ``orientation``,
``cluster_id`` and health/evidence refs. ``weight_in_model`` does NOT live here;
it belongs to ``ModelVersion`` (see ``model_version.py``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "LibraryStatus",
    "LibraryPromotionStep",
    "LibraryMembership",
    "FactorLibraryVersion",
    "LibraryActivePointer",
    "LibraryPromotionRecord",
    "promote_library_version",
    "rollback_library_version",
]


class LibraryStatus(enum.Enum):
    """Lifecycle status of a FactorLibraryVersion (spec §16)."""

    CANDIDATE = "CANDIDATE"
    SHADOW = "SHADOW"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"


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
    """A library member — not just a factor_id (spec §15).

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
class FactorLibraryVersion:
    """Immutable library version (spec §15/§16/§17).

    A library spans MANY clusters, so it binds to a global clustering run
    (``cluster_set_version_id``), NOT a single ``cluster_version_id``.
    """

    library_version_id: str
    logical_library_id: str
    cluster_set_version_id: str
    members: tuple[LibraryMembership, ...] = ()
    policy_hash: str = ""
    evidence_snapshot: str = ""
    status: str = "CANDIDATE"  # CANDIDATE/SHADOW/APPROVED/PRODUCTION
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
    """The mutable *active pointer* that promotion/rollback flips (spec §53.8).

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


def promote_library_version(
    version: FactorLibraryVersion,
    pointer: LibraryActivePointer,
    actor_principal_id: str = "",
    reason: str = "",
) -> tuple[FactorLibraryVersion, LibraryActivePointer, LibraryPromotionRecord]:
    """Advance a library version one promotion step and flip the active pointer.

    Returns ``(new_version, new_pointer, record)``. The version is immutable, so
    promotion produces a NEW ``FactorLibraryVersion`` with the next status. The
    pointer is updated to point at this version once it reaches PRODUCTION (or
    whenever it is the highest promoted step). History is preserved — the old
    version object is untouched.
    """
    current = LibraryPromotionStep(version.status)
    nxt = LibraryPromotionStep.next(current)
    if nxt is None:
        raise ValueError(
            f"version {version.library_version_id!r} is already at terminal status "
            f"{version.status!r}; cannot promote further"
        )
    new_version = FactorLibraryVersion(
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
    return new_version, new_pointer, record


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
