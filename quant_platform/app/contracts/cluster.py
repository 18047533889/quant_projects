"""ClusterVersion label-drift + incremental clustering continuity (spec §14, §53.9).

DRAFT. PURE stdlib. Builds on the layered model in ``cluster_library.py``
(``ClusterVersion``, ``LogicalCluster``, ``ClusterLineageEdge``,
``ClusterConversionType``) WITHOUT duplicating those dataclasses.

This module adds the *label-drift* and *incremental clustering* semantics:

- ``algorithm_cluster_label`` (the raw label emitted by the clustering algorithm
  in a given run) is SEPARATE from ``logical_cluster_id`` (the stable id that
  survives across runs). The same logical cluster may carry a different
  algorithm label in a later run; a new algorithm label may map to an existing
  logical cluster.
- Lineage transitions (UNCHANGED/MIGRATED/SPLIT/MERGED/NEW/DISSOLVED) record the
  prev/next version relationship via ``ClusterLineageEdge``.
- Incremental clustering: singleton / ambiguous / split / merge cases must
  preserve logical-cluster ID continuity so downstream consumers (library,
  feature set) can rely on stable ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .cluster_library import (
    ClusterConversionType,
    ClusterVersionRef,
)

__all__ = [
    "ClusterLabelDrift",
    "ClusterLineageEdge",
    "IncrementalClusterDecision",
    "ClusterVersionPair",
    "classify_cluster_label_drift",
    "resolve_incremental_cluster",
    "build_lineage_edges",
]


@dataclass(frozen=True)
class ClusterLineageEdge:
    """Old→new cluster version transition (spec §14) — carried ref pair.

    ``old_version_id`` / ``new_version_id`` are the domain version refs'
    ``cluster_version_id`` values; ``transition`` labels the change the domain
    certified.  The platform renders lineage, never re-derives it.
    """

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
class ClusterLabelDrift:
    """Label-drift classification between two runs of one logical cluster.

    ``algorithm_cluster_label`` is the raw algorithm output; ``logical_cluster_id``
    is the stable id. Drift is detected when the algorithm label changes while the
    logical id stays the same (or vice versa).
    """

    logical_cluster_id: str
    prev_algorithm_label: str
    next_algorithm_label: str
    label_changed: bool
    transition: ClusterConversionType

    def __post_init__(self) -> None:
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")


def classify_cluster_label_drift(
    prev: ClusterVersionRef,
    next_: ClusterVersionRef,
) -> ClusterLabelDrift:
    """Classify label drift between two versions of the same logical cluster.

    The logical cluster id must match (that is what makes them the "same"
    cluster). The algorithm label may differ — that is the drift.  The refs are
    carried from the domain's ClusterVersionArtifact; the platform only
    classifies the *transition label* it renders, never re-derives cluster
    membership semantics.
    """
    if prev.logical_cluster_id != next_.logical_cluster_id:
        raise ValueError(
            "cannot classify drift across different logical clusters: "
            f"{prev.logical_cluster_id!r} vs {next_.logical_cluster_id!r}"
        )
    label_changed = _ref_label(prev) != _ref_label(next_)
    transition = (
        ClusterConversionType.MIGRATED
        if label_changed
        else ClusterConversionType.UNCHANGED
    )
    return ClusterLabelDrift(
        logical_cluster_id=prev.logical_cluster_id,
        prev_algorithm_label=_ref_label(prev),
        next_algorithm_label=_ref_label(next_),
        label_changed=label_changed,
        transition=transition,
    )


def _ref_label(ref: ClusterVersionRef) -> str:
    """Render the domain algorithm-cluster label the ref carries ('' when absent)."""
    label = getattr(ref, "algorithm_cluster_label", "") or ""
    return str(label)


@dataclass(frozen=True)
class IncrementalClusterDecision:
    """Decision for one incremental-clustering case (spec §14, §53.9).

    ``logical_cluster_id`` is the STABLE id that must be preserved for continuity.
    ``transition`` records how this run relates to the previous one.
    """

    logical_cluster_id: str
    transition: ClusterConversionType
    algorithm_cluster_label: str
    member_factor_ids: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")
        if not self.algorithm_cluster_label:
            raise ValueError("algorithm_cluster_label is required")


def resolve_incremental_cluster(
    *,
    case: str,
    logical_cluster_id: str,
    algorithm_cluster_label: str,
    member_factor_ids: tuple[str, ...] = (),
    prev_logical_cluster_ids: tuple[str, ...] = (),
) -> IncrementalClusterDecision:
    """Resolve an incremental-clustering case to a stable logical-cluster id.

    ``case`` is one of ``singleton`` / ``ambiguous`` / ``split`` / ``merge`` /
    ``new`` / ``dissolved``:

    - ``singleton``: one factor, one cluster -> UNCHANGED (or NEW if no prev).
    - ``ambiguous``: a factor is equidistant between clusters -> keep the prior
      logical id (continuity) and mark MIGRATED.
    - ``split``: one prior cluster splits into several -> each new piece keeps a
      NEW logical id, but the prior id is recorded as the source (SPLIT).
    - ``merge``: several prior clusters merge into one -> the merged cluster takes
      a NEW logical id, recording all sources (MERGED).
    - ``new``: a brand-new cluster with no prior -> NEW.
    - ``dissolved``: a prior cluster no longer exists -> DISSOLVED.
    """
    case = case.lower()
    if case == "singleton":
        transition = (
            ClusterConversionType.NEW
            if not prev_logical_cluster_ids
            else ClusterConversionType.UNCHANGED
        )
        return IncrementalClusterDecision(
            logical_cluster_id=logical_cluster_id,
            transition=transition,
            algorithm_cluster_label=algorithm_cluster_label,
            member_factor_ids=member_factor_ids,
            reason="singleton cluster",
        )
    if case == "ambiguous":
        # Preserve the prior logical id for continuity.
        return IncrementalClusterDecision(
            logical_cluster_id=logical_cluster_id,
            transition=ClusterConversionType.MIGRATED,
            algorithm_cluster_label=algorithm_cluster_label,
            member_factor_ids=member_factor_ids,
            reason="ambiguous assignment; kept prior logical id",
        )
    if case == "split":
        return IncrementalClusterDecision(
            logical_cluster_id=logical_cluster_id,
            transition=ClusterConversionType.SPLIT,
            algorithm_cluster_label=algorithm_cluster_label,
            member_factor_ids=member_factor_ids,
            reason=f"split from {prev_logical_cluster_ids}",
        )
    if case == "merge":
        return IncrementalClusterDecision(
            logical_cluster_id=logical_cluster_id,
            transition=ClusterConversionType.MERGED,
            algorithm_cluster_label=algorithm_cluster_label,
            member_factor_ids=member_factor_ids,
            reason=f"merged from {prev_logical_cluster_ids}",
        )
    if case == "new":
        return IncrementalClusterDecision(
            logical_cluster_id=logical_cluster_id,
            transition=ClusterConversionType.NEW,
            algorithm_cluster_label=algorithm_cluster_label,
            member_factor_ids=member_factor_ids,
            reason="new cluster",
        )
    if case == "dissolved":
        return IncrementalClusterDecision(
            logical_cluster_id=logical_cluster_id,
            transition=ClusterConversionType.DISSOLVED,
            algorithm_cluster_label=algorithm_cluster_label,
            member_factor_ids=member_factor_ids,
            reason="cluster dissolved",
        )
    raise ValueError(f"unknown incremental case: {case!r}")


@dataclass(frozen=True)
class ClusterVersionPair:
    """A prev/next version relationship for one logical cluster (spec §14)."""

    logical_cluster_id: str
    prev_version_id: str
    next_version_id: str
    transition: ClusterConversionType
    created_at: datetime | None = None

    def to_lineage_edge(self) -> ClusterLineageEdge:
        return ClusterLineageEdge(
            old_version_id=self.prev_version_id,
            new_version_id=self.next_version_id,
            transition=self.transition,
            created_at=self.created_at,
        )
    def __post_init__(self) -> None:
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")
        if not self.prev_version_id:
            raise ValueError("prev_version_id is required")
        if not self.next_version_id:
            raise ValueError("next_version_id is required")


def build_lineage_edges(
    pairs: tuple[ClusterVersionPair, ...],
) -> tuple[ClusterLineageEdge, ...]:
    """Build the lineage edges for a set of prev/next version pairs."""
    return tuple(p.to_lineage_edge() for p in pairs)
