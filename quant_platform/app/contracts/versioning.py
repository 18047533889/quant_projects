"""Versioning — the four-layer chain + FeatureSetDiff/retrain classification.

DRAFT. PURE stdlib. This module ties the four-layer versioning model together
and provides the *diff/trigger classification* that decides whether a change
requires model retrain (spec §18).

THE FOUR-LAYER CHAIN:

    ClusterVersion -> FactorLibraryVersion -> FeatureSetVersion -> ModelVersion

- ``ClusterVersion``: label-drift + incremental clustering (see ``cluster.py``).
- ``FactorLibraryVersion``: governance asset set, immutable, promotion/rollback
  via active pointer (see ``factor_library.py``).
- ``FeatureSetVersion``: model-input truth source, immutable, ordered feature
  identity (see ``feature_set.py``).
- ``ModelVersion``: training result, ``weight_in_model`` lives here (see
  ``model_version.py``).

``classify_feature_set_diff`` is the pure, testable entry point: given an old and
a new ``FeatureSetVersion`` it returns a ``FeatureSetDiff`` whose category drives
the retrain decision. Default: METADATA_ONLY and EVIDENCE_ONLY -> NO retrain;
MEMBERSHIP/TREATMENT/ORIENTATION/SCHEMA/LABEL/material DATA_REVISION -> emit
``ModelRetrainRequiredEvent``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .cluster import ClusterLabelDrift, ClusterVersionPair, classify_cluster_label_drift
from .cluster_library import ClusterConversionType, ClusterLineageEdge
from .factor_library import (
    FactorLibraryVersion,
    LibraryActivePointer,
    LibraryPromotionRecord,
    LibraryStatus,
    promote_library_version,
    rollback_library_version,
)
from .feature_set import (
    FeatureSetDiff,
    FeatureSetDiffCategory,
    FeatureSetVersion,
    ModelRetrainRequiredEvent,
    classify_feature_set_diff,
    compute_feature_set_content_hash,
)
from .model_version import (
    DataSnapshot,
    LabelDefinition,
    ModelVersion,
    ModelWeight,
    SplitPlan,
)

__all__ = [
    # re-exported chain types
    "ClusterLabelDrift",
    "ClusterVersionPair",
    "ClusterLineageEdge",
    "ClusterConversionType",
    "classify_cluster_label_drift",
    "FactorLibraryVersion",
    "LibraryActivePointer",
    "LibraryPromotionRecord",
    "LibraryStatus",
    "promote_library_version",
    "rollback_library_version",
    "FeatureSetVersion",
    "FeatureSetDiff",
    "FeatureSetDiffCategory",
    "ModelRetrainRequiredEvent",
    "classify_feature_set_diff",
    "compute_feature_set_content_hash",
    "ModelVersion",
    "ModelWeight",
    "LabelDefinition",
    "DataSnapshot",
    "SplitPlan",
    # versioning-specific
    "VersionChain",
    "VersionChainLink",
    "VersionChainStatus",
    "build_version_chain",
    "retrain_decision_for_diff",
]


class VersionChainStatus:
    """Status labels for the four-layer version chain."""

    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True)
class VersionChainLink:
    """One link in the four-layer version chain.

    Each link is immutable and references the next layer's version. The chain is
    the strict ordering:

        cluster_set_version -> library_version -> feature_set_version -> model_version
    """

    cluster_set_version_id: str
    library_version_id: str
    feature_set_version_id: str
    model_version_id: str | None = None
    status: str = VersionChainStatus.ACTIVE
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.cluster_set_version_id:
            raise ValueError("cluster_set_version_id is required")
        if not self.library_version_id:
            raise ValueError("library_version_id is required")
        if not self.feature_set_version_id:
            raise ValueError("feature_set_version_id is required")
        if self.status not in {
            VersionChainStatus.ACTIVE,
            VersionChainStatus.SUPERSEDED,
            VersionChainStatus.ROLLED_BACK,
        }:
            raise ValueError(f"unknown chain status: {self.status!r}")


@dataclass(frozen=True)
class VersionChain:
    """The ordered history of version-chain links for one logical library."""

    logical_library_id: str
    links: tuple[VersionChainLink, ...] = ()

    @property
    def active(self) -> VersionChainLink | None:
        for link in self.links:
            if link.status == VersionChainStatus.ACTIVE:
                return link
        return None

    def __post_init__(self) -> None:
        if not self.logical_library_id:
            raise ValueError("logical_library_id is required")


def build_version_chain(
    logical_library_id: str,
    links: tuple[VersionChainLink, ...],
) -> VersionChain:
    """Build a VersionChain, enforcing at most one ACTIVE link."""
    active_count = sum(1 for l in links if l.status == VersionChainStatus.ACTIVE)
    if active_count > 1:
        raise ValueError(
            f"version chain for {logical_library_id!r} has {active_count} ACTIVE "
            "links; exactly one is required"
        )
    return VersionChain(logical_library_id=logical_library_id, links=links)


def retrain_decision_for_diff(
    diff: FeatureSetDiff,
    feature_set_id: str,
) -> ModelRetrainRequiredEvent | None:
    """Convenience: turn a FeatureSetDiff into a retrain event (or None)."""
    return diff.to_event(feature_set_id)
