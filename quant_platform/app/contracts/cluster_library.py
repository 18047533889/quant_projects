"""Cluster / Library — layered SimilarityGraph → ClusterSet → Cluster model.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.6/§4.7 (spec
§8.5, §8.6, §14, §15, §16). PURE stdlib frozen dataclasses. ``[RECONCILE]``
against ``factor_assets/clustering/`` and library-governance types before freeze.

THE CLUSTERSETVERSION CORRECTION (§5.6, spec §34/§35/§40):

``FactorLibraryVersion.cluster_version_id`` was WRONG — a library spans MANY
clusters, so it must bind to a **global clustering run / ClusterSetVersion**, NOT
one cluster version. The layered model:

- ``SimilarityGraphVersion`` — one global similarity-graph version.
- ``ClusterSetVersion`` — one *global clustering run* (many clusters at once),
  bound to a ``SimilarityGraphVersion``.
- ``LogicalCluster`` — stable logical id (``CL_PV_MOM_017`` style), stable across
  versions.
- ``ClusterVersion`` — one LogicalCluster's version within a ClusterSetVersion.
- ``ClusterMembership`` — factor ↔ logical-cluster binding.
- ``ClusterLineageEdge`` — old→new transition (UNCHANGED/MIGRATED/SPLIT/MERGED/
  NEW/DISSOLVED), preserving ``ClusterConversionType``.

``FactorLibraryVersion.cluster_version_id`` has been RENAMED to
``cluster_set_version_id``.
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
    "SimilarityGraphVersion",
    "ClusterSetVersion",
    "LogicalCluster",
    "ClusterMembership",
    "ClusterLineageEdge",
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
class SimilarityGraphVersion:
    """One global similarity-graph version (§34)."""

    graph_version_id: str
    graph_ref: str
    policy_hash: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.graph_version_id:
            raise ValueError("graph_version_id is required")
        if not self.graph_ref:
            raise ValueError("graph_ref is required")


@dataclass(frozen=True)
class ClusterSetVersion:
    """One *global clustering run* — many clusters at once (§35).

    ``cluster_set_version_id`` is a global id. A library binds to this (a
    cluster-set), NOT to a single ``ClusterVersion``.
    """

    cluster_set_version_id: str
    similarity_graph_version: SimilarityGraphVersion
    algorithm: str
    backend: str = ""
    seed: int = 0
    resolution: float | None = None
    policy_hash: str = ""
    clustering_run_ref: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.cluster_set_version_id:
            raise ValueError("cluster_set_version_id is required")
        if not self.algorithm:
            raise ValueError("algorithm is required")


@dataclass(frozen=True)
class LogicalCluster:
    """Stable logical cluster id (``CL_PV_MOM_017`` style) — stable across versions."""

    logical_cluster_id: str
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")


@dataclass(frozen=True)
class ClusterVersion:
    """Immutable cluster version (spec §14). Never updated in place.

    A ``ClusterVersion`` belongs to a ``ClusterSetVersion`` (the global run) and
    to a ``LogicalCluster`` (the stable id). Refreshes produce new versions and
    record ``v43 -> v44`` lineage via ``ClusterLineageEdge``.
    """

    cluster_version_id: str
    logical_cluster_id: str
    cluster_set_version_id: str
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
        if not self.cluster_set_version_id:
            raise ValueError("cluster_set_version_id is required")
        if not self.algorithm_cluster_label:
            raise ValueError("algorithm_cluster_label is required")


@dataclass(frozen=True)
class ClusterMembership:
    """A factor's membership in a logical cluster."""

    factor_definition_id: str
    logical_cluster_id: str
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.factor_definition_id:
            raise ValueError("factor_definition_id is required")
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")


@dataclass(frozen=True)
class ClusterLineageEdge:
    """Old→new cluster version transition (spec §14)."""

    old_version_id: str
    new_version_id: str
    transition: ClusterConversionType
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.old_version_id:
            raise ValueError("old_version_id is required")
        if not self.new_version_id:
            raise ValueError("new_version_id is required")


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
    """Immutable library version (spec §15/§16).

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
        if self.status not in {"CANDIDATE", "SHADOW", "APPROVED", "PRODUCTION"}:
            raise ValueError(f"unknown library status: {self.status!r}")
