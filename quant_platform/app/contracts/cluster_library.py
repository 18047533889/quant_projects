"""ClusterVersion + FactorLibraryVersion + LibraryMembership.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.6/§4.7 (spec
§8.5, §8.6, §14, §15, §16). PURE stdlib frozen dataclasses. ``[RECONCILE]``
against ``factor_assets/clustering/`` and library-governance types before
freeze.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "ClusterConversionType",
    "ClusterVersion",
    "FactorLibraryVersion",
    "LibraryMembership",
]


class ClusterConversionType(enum.Enum):
    """Cluster-lineage conversion type (spec §14)."""

    UNCHANGED = "UNCHANGED"
    MIGRATED = "MIGRATED"
    SPLIT = "SPLIT"
    MERGED = "MERGED"
    NEW = "NEW"
    DISSOLVED = "DISSOLVED"


@dataclass(frozen=True)
class ClusterVersion:
    """Immutable cluster version (spec §14). Never updated in place."""

    cluster_version_id: str
    logical_cluster_id: str
    algorithm_cluster_label: str
    member_factor_ids: tuple[str, ...] = ()
    policy_hash: str = ""
    conversion_type: ClusterConversionType | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.cluster_version_id:
            raise ValueError("cluster_version_id is required")
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")
        if not self.algorithm_cluster_label:
            raise ValueError("algorithm_cluster_label is required")


@dataclass(frozen=True)
class LibraryMembership:
    """A library member — not just a factor_id (spec §15)."""

    factor_definition_id: str
    selected_treatment_id: str | None = None
    orientation: str | None = None
    cluster_id: str | None = None
    representative_of: str | None = None
    similarity_ref: str | None = None
    health_state_ref: str | None = None
    assembly_score: float | None = None
    selection_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.factor_definition_id:
            raise ValueError("factor_definition_id is required")


@dataclass(frozen=True)
class FactorLibraryVersion:
    """Immutable library version (spec §15/§16)."""

    library_version_id: str
    logical_library_id: str
    cluster_version_id: str
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
        if not self.cluster_version_id:
            raise ValueError("cluster_version_id is required")
        if self.status not in {"CANDIDATE", "SHADOW", "APPROVED", "PRODUCTION"}:
            raise ValueError(f"unknown library status: {self.status!r}")
