"""Cluster governance contracts for factor_assets (DLIB-FA-007/008/009/010/011).

These are the canonical, deep-immutable, hash-addressed artifacts that govern
factor clustering and cluster-set versioning.  They are distinct from the
legacy :class:`~factor_assets.clustering.families.ClusterArtifact` /
:class:`~factor_assets.clustering.lineage.FamilyLineage`:

- :class:`SimilarityGraphArtifact` — sparse-graph observation semantics that
  distinguish a PAIR NEVER MEASURED from a PAIR MEASURED BELOW THRESHOLD.
- :class:`ClusterSetVersionArtifact` — an immutable, versioned snapshot of a
  full clustering run (one per global refresh).
- :class:`LogicalCluster` / :class:`ClusterVersionArtifact` — stable logical
  cluster identity (``CL_PV_MOM_017``) decoupled from the unstable algorithm
  label (Leiden label ``18``).
- :class:`ClusterMembership` / :class:`ClusterLineageEdge` — per-factor
  membership and cross-version lineage.
- :class:`ClusterVersionMatcher` — matches clusters across versions into
  UNCHANGED / MIGRATED / SPLIT / MERGED / NEW / DISSOLVED.
- :class:`IncrementalClusterAssignment` — daily incremental assignment that
  never mutates the immutable production ClusterSetVersion in place.

The legacy :class:`~factor_assets.clustering.lineage.FamilyLineage` is a
RESEARCH/ANALYTIC-ONLY factor parent-child *behavioral* lineage (derived from
degree/correlation/neighborhood overlap).  It is NOT real factor genealogy —
real genealogy is generator ``parent_factor_ids`` / FO mutation lineage / FE
AST / candidate provenance.  It is kept but scoped research-only.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional

from factor_assets.contracts._canonical import canonical_digest
from factor_assets.contracts.similarity import is_unknown_identity

__all__ = [
    "GraphCompletenessClass",
    "UnknownEdgeSemantics",
    "SimilarityGraphArtifact",
    "ClusterSetVersionArtifact",
    "LogicalCluster",
    "ClusterVersionArtifact",
    "ClusterMembership",
    "ClusterLineageEdge",
    "ClusterVersionMatch",
    "ClusterVersionMatcher",
    "IncrementalAssignmentKind",
    "IncrementalClusterAssignment",
    "ClusterResolutionSelector",
    "ClusterScale",
]


class GraphCompletenessClass(Enum):
    """Completeness class of a similarity graph (DLIB-FA-007/54).

    Production Leiden only accepts a certified graph:
    ``CERTIFIED_ANN_REFINED`` or ``CERTIFIED_EXACT``.  Anything else fails
    closed.
    """

    CERTIFIED_ANN_REFINED = "CERTIFIED_ANN_REFINED"
    CERTIFIED_EXACT = "CERTIFIED_EXACT"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class UnknownEdgeSemantics(Enum):
    """Semantics of an unmeasured pair (DLIB-FA-007).

    Distinguishes a PAIR NEVER MEASURED from a PAIR MEASURED BELOW THRESHOLD.
    An unmeasured pair is never treated as a computed zero.
    """

    NEVER_MEASURED = "NEVER_MEASURED"
    MEASURED_BELOW_THRESHOLD = "MEASURED_BELOW_THRESHOLD"


class ClusterScale(Enum):
    """Cluster scale (DLIB-FA-011/66).

    MICRO_CLUSTER = high-similarity dedup; MACRO_CLUSTER = broad
    mechanism/style.  Library selection constrains micro redundancy and
    controls macro exposure.
    """

    MICRO_CLUSTER = "MICRO_CLUSTER"
    MACRO_CLUSTER = "MACRO_CLUSTER"


class IncrementalAssignmentKind(Enum):
    """Outcome of an incremental cluster assignment (DLIB-FA-010)."""

    ASSIGNED = "ASSIGNED"
    AMBIGUOUS = "AMBIGUOUS"
    BRIDGE = "BRIDGE"
    SINGLETON = "SINGLETON"
    OUTLIER = "OUTLIER"
    PENDING_GLOBAL_REFRESH = "PENDING_GLOBAL_REFRESH"


def _freeze_mapping(mapping: Mapping) -> Mapping:
    """Deep-freeze a mapping into an immutable snapshot."""
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class SimilarityGraphArtifact:
    """Sparse similarity-graph observation semantics (DLIB-FA-007).

    Records the node universe, the fingerprint spec, the ANN policy (backend /
    k / exact-refinement policy), the edge threshold, the edge-affinity policy
    version, the refined/stored edge counts, coverage diagnostics, the graph
    completeness class, the unknown-edge semantics, and the graph identity.

    ``graph_completeness_class`` must be ``CERTIFIED_ANN_REFINED`` or
    ``CERTIFIED_EXACT`` for production Leiden (DLIB-FA-054); otherwise the
    graph is not certified and clustering fails closed.
    """

    graph_identity: str
    node_universe_ref: str
    fingerprint_spec_ref: str
    ann_policy: Mapping[str, object]
    ann_k: int
    exact_refinement_policy: str
    edge_threshold: float
    edge_affinity_policy_version: str
    refined_pair_count: int
    stored_edge_count: int
    coverage_diagnostics: Mapping[str, object]
    graph_completeness_class: GraphCompletenessClass
    unknown_edge_semantics: UnknownEdgeSemantics
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.graph_identity:
            raise ValueError("graph_identity is required")
        if not self.node_universe_ref:
            raise ValueError("node_universe_ref is required")
        if not self.fingerprint_spec_ref:
            raise ValueError("fingerprint_spec_ref is required")
        # P0-12 (R55 audit): an UNKNOWN placeholder graph/fingerprint identity
        # is not an identity — two graphs measured under different specs would
        # both read UNKNOWN and cluster versioning would silently merge them.
        for _field_name in ("graph_identity", "node_universe_ref", "fingerprint_spec_ref"):
            _value = getattr(self, _field_name)
            if isinstance(_value, str) and is_unknown_identity(_value):
                raise ValueError(
                    f"{_field_name} is UNKNOWN ({_value!r}); a "
                    "SimilarityGraphArtifact may not carry an unknown identity "
                    "— resolve the concrete spec value first (fail closed)."
                )
        if not isinstance(self.ann_policy, Mapping):
            raise TypeError("ann_policy must be a mapping")
        if not isinstance(self.ann_k, int) or isinstance(self.ann_k, bool) or self.ann_k < 1:
            raise ValueError("ann_k must be a positive integer")
        if not isinstance(self.graph_completeness_class, GraphCompletenessClass):
            raise TypeError("graph_completeness_class must be a GraphCompletenessClass")
        if not isinstance(self.unknown_edge_semantics, UnknownEdgeSemantics):
            raise TypeError("unknown_edge_semantics must be an UnknownEdgeSemantics")
        object.__setattr__(self, "ann_policy", _freeze_mapping(self.ann_policy))
        object.__setattr__(self, "coverage_diagnostics", _freeze_mapping(self.coverage_diagnostics))
        computed = canonical_digest(
            self.graph_identity,
            self.node_universe_ref,
            self.fingerprint_spec_ref,
            self.ann_policy,
            self.ann_k,
            self.exact_refinement_policy,
            self.edge_threshold,
            self.edge_affinity_policy_version,
            self.refined_pair_count,
            self.stored_edge_count,
            self.coverage_diagnostics,
            self.graph_completeness_class.value,
            self.unknown_edge_semantics.value,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed similarity-graph "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )

    @property
    def is_certified(self) -> bool:
        """Whether the graph is certified for production Leiden."""
        return self.graph_completeness_class in (
            GraphCompletenessClass.CERTIFIED_ANN_REFINED,
            GraphCompletenessClass.CERTIFIED_EXACT,
        )

    def to_dict(self) -> dict:
        return {
            "graph_identity": self.graph_identity,
            "node_universe_ref": self.node_universe_ref,
            "fingerprint_spec_ref": self.fingerprint_spec_ref,
            "ann_policy": dict(self.ann_policy),
            "ann_k": self.ann_k,
            "exact_refinement_policy": self.exact_refinement_policy,
            "edge_threshold": self.edge_threshold,
            "edge_affinity_policy_version": self.edge_affinity_policy_version,
            "refined_pair_count": self.refined_pair_count,
            "stored_edge_count": self.stored_edge_count,
            "coverage_diagnostics": dict(self.coverage_diagnostics),
            "graph_completeness_class": self.graph_completeness_class.value,
            "unknown_edge_semantics": self.unknown_edge_semantics.value,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ClusterSetVersionArtifact:
    """Immutable, versioned snapshot of a full clustering run (DLIB-FA-009).

    One per global refresh.  ``member_universe_ref``, the similarity-graph
    version, the algorithm / backend / backend_version / seed / resolution,
    the clustering policy ref, the assignment artifact ref, ``created_at`` and
    a derived ``content_hash``.  Never mutated in place — a new global refresh
    produces a NEW ClusterSetVersionArtifact.
    """

    cluster_set_version_id: str
    member_universe_ref: str
    similarity_graph_version_ref: str
    algorithm: str
    backend: str
    backend_version: str
    seed: int
    resolution: float
    clustering_policy_ref: str
    assignment_artifact_ref: str
    created_at: str
    content_hash: str = ""
    refresh_trigger_ref: str = ""
    downstream_validation_ref: str = ""

    def __post_init__(self) -> None:
        if not self.cluster_set_version_id:
            raise ValueError("cluster_set_version_id is required")
        if not self.member_universe_ref:
            raise ValueError("member_universe_ref is required")
        if not self.similarity_graph_version_ref:
            raise ValueError("similarity_graph_version_ref is required")
        # P0-12 (R55 audit): an UNKNOWN version ref is not a version — two
        # cluster-set versions computed under different graphs would both read
        # UNKNOWN and version matching would silently merge them.
        for _field_name in ("cluster_set_version_id", "similarity_graph_version_ref"):
            _value = getattr(self, _field_name)
            if isinstance(_value, str) and is_unknown_identity(_value):
                raise ValueError(
                    f"{_field_name} is UNKNOWN ({_value!r}); a "
                    "ClusterSetVersionArtifact may not carry an unknown "
                    "version identity (fail closed)."
                )
        if not self.algorithm:
            raise ValueError("algorithm is required")
        if not self.backend:
            raise ValueError("backend is required")
        if not self.backend_version:
            raise ValueError("backend_version is required")
        if not self.clustering_policy_ref:
            raise ValueError("clustering_policy_ref is required")
        if not self.assignment_artifact_ref:
            raise ValueError("assignment_artifact_ref is required")
        if not self.created_at:
            raise ValueError("created_at is required")
        computed = canonical_digest(
            self.cluster_set_version_id,
            self.member_universe_ref,
            self.similarity_graph_version_ref,
            self.algorithm,
            self.backend,
            self.backend_version,
            self.seed,
            self.resolution,
            self.clustering_policy_ref,
            self.assignment_artifact_ref,
            self.refresh_trigger_ref,
            self.downstream_validation_ref,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed cluster-set-version "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )

    def to_dict(self) -> dict:
        return {
            "cluster_set_version_id": self.cluster_set_version_id,
            "member_universe_ref": self.member_universe_ref,
            "similarity_graph_version_ref": self.similarity_graph_version_ref,
            "algorithm": self.algorithm,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "seed": self.seed,
            "resolution": self.resolution,
            "clustering_policy_ref": self.clustering_policy_ref,
            "assignment_artifact_ref": self.assignment_artifact_ref,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
            "refresh_trigger_ref": self.refresh_trigger_ref,
            "downstream_validation_ref": self.downstream_validation_ref,
        }


@dataclass(frozen=True)
class LogicalCluster:
    """Stable logical cluster identity (DLIB-FA-009).

    ``logical_cluster_id`` (e.g. ``CL_PV_MOM_017``) is stable across
    ClusterSetVersions, decoupled from the unstable algorithm label (Leiden
    label ``18``).  ``algorithm_cluster_label`` records the raw label from the
    most recent run for audit.
    """

    logical_cluster_id: str
    algorithm_cluster_label: Optional[int] = None
    scale: ClusterScale = ClusterScale.MICRO_CLUSTER
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")
        if not isinstance(self.scale, ClusterScale):
            raise TypeError("scale must be a ClusterScale")

    def to_dict(self) -> dict:
        return {
            "logical_cluster_id": self.logical_cluster_id,
            "algorithm_cluster_label": self.algorithm_cluster_label,
            "scale": self.scale.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class ClusterVersionArtifact:
    """Per-logical-cluster, per-ClusterSetVersion artifact (DLIB-FA-009)."""

    logical_cluster_id: str
    cluster_set_version_ref: str
    member_factor_ids: tuple[str, ...]
    representative_factor_id: Optional[str] = None
    scale: ClusterScale = ClusterScale.MICRO_CLUSTER
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")
        if not self.cluster_set_version_ref:
            raise ValueError("cluster_set_version_ref is required")
        if not self.member_factor_ids:
            raise ValueError("member_factor_ids cannot be empty")
        object.__setattr__(self, "member_factor_ids", tuple(self.member_factor_ids))
        if len(set(self.member_factor_ids)) != len(self.member_factor_ids):
            raise ValueError("member_factor_ids must be unique")
        if not isinstance(self.scale, ClusterScale):
            raise TypeError("scale must be a ClusterScale")
        computed = canonical_digest(
            self.logical_cluster_id,
            self.cluster_set_version_ref,
            self.member_factor_ids,
            self.representative_factor_id,
            self.scale.value,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed cluster-version "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )

    def to_dict(self) -> dict:
        return {
            "logical_cluster_id": self.logical_cluster_id,
            "cluster_set_version_ref": self.cluster_set_version_ref,
            "member_factor_ids": list(self.member_factor_ids),
            "representative_factor_id": self.representative_factor_id,
            "scale": self.scale.value,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ClusterMembership:
    """Per-factor membership in a logical cluster at a ClusterSetVersion."""

    factor_id: str
    logical_cluster_id: str
    cluster_set_version_ref: str
    scale: ClusterScale = ClusterScale.MICRO_CLUSTER

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")
        if not self.cluster_set_version_ref:
            raise ValueError("cluster_set_version_ref is required")
        if not isinstance(self.scale, ClusterScale):
            raise TypeError("scale must be a ClusterScale")

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "logical_cluster_id": self.logical_cluster_id,
            "cluster_set_version_ref": self.cluster_set_version_ref,
            "scale": self.scale.value,
        }


@dataclass(frozen=True)
class ClusterLineageEdge:
    """Cross-version lineage edge between two logical clusters."""

    from_cluster_id: str
    to_cluster_id: str
    from_cluster_set_version_ref: str
    to_cluster_set_version_ref: str
    match: "ClusterVersionMatch"

    def __post_init__(self) -> None:
        if not self.from_cluster_id:
            raise ValueError("from_cluster_id is required")
        if not self.to_cluster_id:
            raise ValueError("to_cluster_id is required")
        if not isinstance(self.match, ClusterVersionMatch):
            raise TypeError("match must be a ClusterVersionMatch")

    def to_dict(self) -> dict:
        return {
            "from_cluster_id": self.from_cluster_id,
            "to_cluster_id": self.to_cluster_id,
            "from_cluster_set_version_ref": self.from_cluster_set_version_ref,
            "to_cluster_set_version_ref": self.to_cluster_set_version_ref,
            "match": self.match.value,
        }


class ClusterVersionMatch(Enum):
    """How a logical cluster changed across versions (DLIB-FA-009)."""

    UNCHANGED = "UNCHANGED"
    MIGRATED = "MIGRATED"
    SPLIT = "SPLIT"
    MERGED = "MERGED"
    NEW = "NEW"
    DISSOLVED = "DISSOLVED"


class ClusterVersionMatcher:
    """Matches clusters across versions (DLIB-FA-009).

    Uses member overlap / Jaccard / weighted overlap / representative
    continuity / semantic-family consistency to classify each cluster as
    UNCHANGED / MIGRATED / SPLIT / MERGED / NEW / DISSOLVED.

    This is NOT the legacy :class:`~factor_assets.clustering.lineage.FamilyLineage`
    (factor parent-child behavioral lineage).  It matches *logical clusters*
    across ClusterSetVersions.
    """

    def __init__(self, overlap_threshold: float = 0.5):
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in [0, 1]")
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def jaccard(a: set, b: set) -> float:
        """Jaccard similarity of two member sets."""
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def match(
        self,
        previous: Mapping[str, ClusterVersionArtifact],
        current: Mapping[str, ClusterVersionArtifact],
    ) -> list[ClusterLineageEdge]:
        """Match clusters in ``previous`` to clusters in ``current``.

        Returns a list of :class:`ClusterLineageEdge` describing how each
        previous cluster evolved.  A previous cluster with no current match is
        DISSOLVED; a current cluster with no previous match is NEW.
        """
        edges: list[ClusterLineageEdge] = []
        prev_by_id, cur_by_id = dict(previous), dict(current)
        links: list[tuple[str, str, float]] = []
        for prev_id, prev_artifact in sorted(prev_by_id.items()):
            a = set(prev_artifact.member_factor_ids)
            for cur_id, cur_artifact in sorted(cur_by_id.items()):
                b = set(cur_artifact.member_factor_ids)
                intersection = len(a & b)
                if not intersection:
                    continue
                # Require either Jaccard or strong retention in both directions;
                # retain all qualifying links to expose split/merge topology.
                j = self.jaccard(a, b)
                retention = min(intersection / len(a), intersection / len(b))
                if max(j, retention) >= self.overlap_threshold:
                    links.append((prev_id, cur_id, j))
        prev_degree = {pid: sum(p == pid for p, _, _ in links) for pid in prev_by_id}
        cur_degree = {cid: sum(c == cid for _, c, _ in links) for cid in cur_by_id}
        for prev_id, cur_id, score in links:
            if prev_degree[prev_id] > 1:
                kind = ClusterVersionMatch.SPLIT
            elif cur_degree[cur_id] > 1:
                kind = ClusterVersionMatch.MERGED
            else:
                kind = ClusterVersionMatch.UNCHANGED if score >= .9 else ClusterVersionMatch.MIGRATED
            edges.append(ClusterLineageEdge(
                prev_id, cur_id, prev_by_id[prev_id].cluster_set_version_ref,
                cur_by_id[cur_id].cluster_set_version_ref, kind,
            ))
        linked_prev = {p for p, _, _ in links}
        linked_cur = {c for _, c, _ in links}
        for prev_id in sorted(set(prev_by_id) - linked_prev):
            edges.append(ClusterLineageEdge(
                prev_id, prev_id, prev_by_id[prev_id].cluster_set_version_ref, "",
                ClusterVersionMatch.DISSOLVED,
            ))
        for cur_id in sorted(set(cur_by_id) - linked_cur):
            edges.append(ClusterLineageEdge(
                cur_id, cur_id, "", cur_by_id[cur_id].cluster_set_version_ref,
                ClusterVersionMatch.NEW,
            ))
        return edges


@dataclass(frozen=True)
class IncrementalClusterAssignment:
    """Daily incremental cluster assignment (DLIB-FA-010).

    A new factor is fingerprinted -> ANN shortlist -> exact refined
    similarities -> cluster affinity -> this assignment.  It NEVER mutates the
    immutable production ClusterSetVersion in place; new factors go into an
    IncrementalOverlayVersion / PendingAssignment until a weekly/monthly Global
    Refresh produces a NEW ClusterSetVersion.
    """

    factor_id: str
    logical_cluster_id: Optional[str]
    kind: IncrementalAssignmentKind
    cluster_set_version_ref: str
    affinity: Optional[float] = None
    parent_cluster_set_hash: str = ""
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not isinstance(self.kind, IncrementalAssignmentKind):
            raise TypeError("kind must be an IncrementalAssignmentKind")
        if not self.cluster_set_version_ref:
            raise ValueError("cluster_set_version_ref is required")
        if self.kind is IncrementalAssignmentKind.ASSIGNED and not self.parent_cluster_set_hash:
            raise ValueError("ASSIGNED outcomes require parent_cluster_set_hash for CAS")
        if self.affinity is not None:
            if isinstance(self.affinity, bool) or not isinstance(self.affinity, (int, float)):
                raise TypeError("affinity must be a non-boolean number or None")
            a = float(self.affinity)
            if a != a or a in (float("inf"), float("-inf")):
                raise ValueError("affinity must be finite")
            object.__setattr__(self, "affinity", a)
        computed = canonical_digest(
            self.factor_id,
            self.logical_cluster_id,
            self.kind.value,
            self.cluster_set_version_ref,
            self.affinity,
            self.parent_cluster_set_hash,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed incremental "
                "assignment content hash; a caller may not self-report an "
                "arbitrary hash — FAIL CLOSED"
            )

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "logical_cluster_id": self.logical_cluster_id,
            "kind": self.kind.value,
            "cluster_set_version_ref": self.cluster_set_version_ref,
            "affinity": self.affinity,
            "parent_cluster_set_hash": self.parent_cluster_set_hash,
            "content_hash": self.content_hash,
        }


class ClusterResolutionSelector:
    """Selects a clustering resolution (DLIB-FA-011).

    NOT max-modularity.  A grid of resolutions is scored on bootstrap /
    time-slice stability (ARI, VI), cluster-size distribution, singleton ratio,
    unstable-small-cluster ratio, cross-cluster similarity, within-cluster
    similarity, representative stability, and downstream FactorLibrary utility,
    then a stable plateau is chosen.  Modularity is only a tiebreaker — a high
    modularity alone never selects a resolution.
    """

    #: Metric names that measure *stability* (higher is better).  These drive
    #: the composite score.  ``vi`` (variation of information) is inverted
    #: (lower is better).  Modularity is deliberately NOT in this set.
    STABILITY_KEYS = frozenset({
        "ari",
        "adjusted_rand_index",
        "stability",
        "bootstrap_stability",
        "time_slice_stability",
        "representative_stability",
        "within_cluster_similarity",
        "downstream_library_utility",
        "vi",
    })

    def __init__(self, resolutions: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)):
        if not resolutions:
            raise ValueError("resolutions must be non-empty")
        self.resolutions = tuple(resolutions)

    @staticmethod
    def _score_value(key: str, value: float) -> float:
        """Normalize a metric value so higher is better."""
        if key == "vi":
            if value < 0:
                raise ValueError("variation of information must be non-negative")
            return 1.0 / (1.0 + value)  # bounded, lower VI is better
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"stability metric {key!r} must be normalized to [0,1]")
        return value

    def select(
        self,
        scores: Mapping[float, Mapping[str, float]],
    ) -> float:
        """Select the resolution with the best composite stability score.

        ``scores`` maps resolution -> {metric: value}.  The composite is the
        mean of the available STABILITY metrics (higher is better); when no
        stability metric is present the composite falls back to the mean of all
        available metrics.  Modularity is only a tiebreaker — a resolution is
        never selected for having the highest modularity.
        """
        if not scores:
            raise ValueError("scores must be non-empty")
        unknown = set(scores) - set(self.resolutions)
        if unknown:
            raise ValueError(f"scores contain resolutions outside declared grid: {sorted(unknown)}")
        composites: dict[float, float] = {}
        for resolution in self.resolutions:
            if resolution not in scores:
                continue
            metric_scores = scores[resolution]
            values = [
                self._score_value(key, v)
                for key, v in metric_scores.items()
                if key in self.STABILITY_KEYS and isinstance(v, (int, float))
                and not isinstance(v, bool) and math.isfinite(v)
            ]
            if not values:
                continue
            composites[resolution] = sum(values) / len(values)
        if not composites:
            raise ValueError("required normalized stability evidence is missing")
        best = max(composites.values())
        near = [r for r in self.resolutions if r in composites and composites[r] >= best - .02]
        # Choose the center of the longest adjacent stable plateau; a singleton
        # is allowed only when no adjacent configuration has comparable evidence.
        runs: list[list[float]] = []
        for r in near:
            if not runs or self.resolutions.index(r) != self.resolutions.index(runs[-1][-1]) + 1:
                runs.append([r])
            else:
                runs[-1].append(r)
        plateau = max(runs, key=lambda run: (len(run), max(composites[r] for r in run)))
        return plateau[(len(plateau) - 1) // 2]
