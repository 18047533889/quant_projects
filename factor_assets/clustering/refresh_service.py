"""Evidence-backed cluster refresh orchestration; scheduling remains platform-owned."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

from factor_assets.contracts.cluster_governance import (
    ClusterLineageEdge, ClusterSetVersionArtifact, ClusterVersionArtifact,
    ClusterVersionMatcher,
)


@dataclass(frozen=True)
class ClusterRefreshPolicy:
    policy_ref: str
    local_review_days: int = 30
    global_review_days: int = 90
    pending_share_trigger: float = .10
    migration_share_trigger: float = .10
    min_quality_failures: int = 1


@dataclass(frozen=True)
class ClusterRefreshEvidence:
    observed_at: str
    days_since_local_review: int
    days_since_global_review: int
    pending_share: float
    migration_share: float
    persistent_quality_failures: int
    similarity_semantics_changed: bool = False
    universe_changed: bool = False
    horizon_changed: bool = False
    fingerprint_semantics_changed: bool = False
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClusterRefreshDecision:
    kind: str
    reason_codes: tuple[str, ...]
    policy_ref: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class ClusterRefreshRun:
    decision: ClusterRefreshDecision
    candidate_cluster_set: ClusterSetVersionArtifact
    lineage_edges: tuple[ClusterLineageEdge, ...]
    registry_event_ref: str
    status: str = "RESEARCH_CANDIDATE_NOT_PRODUCTION"
    production_cluster_set_ref: str = ""


class ClusterRefreshRegistry:
    """Append-only evidence registry for refresh candidates, not publication."""
    def __init__(self) -> None:
        self._runs: list[ClusterRefreshRun] = []

    def append(self, run: ClusterRefreshRun) -> ClusterRefreshRun:
        if run.status != "RESEARCH_CANDIDATE_NOT_PRODUCTION":
            raise ValueError("refresh service cannot publish production cluster sets")
        ref = f"cluster-refresh:{len(self._runs) + 1}:{run.candidate_cluster_set.content_hash}"
        stored = ClusterRefreshRun(
            run.decision, run.candidate_cluster_set, run.lineage_edges, ref,
            run.status, run.production_cluster_set_ref,
        )
        self._runs.append(stored)
        return stored

    @property
    def runs(self) -> tuple[ClusterRefreshRun, ...]:
        return tuple(self._runs)


def evaluate_cluster_refresh(
    evidence: ClusterRefreshEvidence, policy: ClusterRefreshPolicy,
) -> ClusterRefreshDecision:
    reasons: list[str] = []
    semantic = any((evidence.similarity_semantics_changed, evidence.universe_changed,
                    evidence.horizon_changed, evidence.fingerprint_semantics_changed))
    if semantic:
        reasons.append("SEMANTIC_REBUILD_REQUIRED")
    if evidence.days_since_global_review >= policy.global_review_days:
        reasons.append("GLOBAL_REVIEW_DUE")
    if evidence.pending_share >= policy.pending_share_trigger:
        reasons.append("PENDING_SHARE_TRIGGER")
    if evidence.migration_share >= policy.migration_share_trigger:
        reasons.append("MIGRATION_SHARE_TRIGGER")
    if evidence.persistent_quality_failures >= policy.min_quality_failures:
        reasons.append("PERSISTENT_QUALITY_FAILURE")
    if reasons:
        kind = "GLOBAL_REFRESH"
    elif evidence.days_since_local_review >= policy.local_review_days:
        kind, reasons = "LOCAL_REVIEW", ["LOCAL_REVIEW_DUE"]
    else:
        kind, reasons = "NO_REFRESH", ["NO_TRIGGER"]
    return ClusterRefreshDecision(kind, tuple(reasons), policy.policy_ref,
                                  tuple(evidence.evidence_refs))


def execute_cluster_refresh(
    *, decision: ClusterRefreshDecision,
    previous_clusters: Mapping[str, ClusterVersionArtifact],
    cluster_builder: Callable[[], tuple[ClusterSetVersionArtifact, Mapping[str, ClusterVersionArtifact]]],
    registry: ClusterRefreshRegistry,
    production_cluster_set_ref: str,
    matcher: ClusterVersionMatcher | None = None,
) -> ClusterRefreshRun:
    if decision.kind != "GLOBAL_REFRESH":
        raise ValueError("only a GLOBAL_REFRESH decision can execute clustering")
    candidate, current = cluster_builder()  # existing clustering service/callback authority
    lineage = tuple((matcher or ClusterVersionMatcher()).match(previous_clusters, current))
    draft = ClusterRefreshRun(
        decision, candidate, lineage, registry_event_ref="PENDING_APPEND",
        production_cluster_set_ref=production_cluster_set_ref,
    )
    return registry.append(draft)
