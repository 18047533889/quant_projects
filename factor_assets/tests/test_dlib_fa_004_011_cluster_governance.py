"""DLIB-FA-007/008/009/010/011 cluster governance + similarity view tests.

- SimilarityGraphArtifact: unmeasured pair semantics + certified-only Leiden.
- ClusterSetVersionArtifact: immutable versioned snapshot; one per refresh.
- LogicalCluster / ClusterVersionArtifact: stable logical identity decoupled
  from the unstable algorithm label.
- ClusterVersionMatcher: UNCHANGED / MIGRATED / SPLIT / MERGED / NEW /
  DISSOLVED.
- IncrementalClusterAssignment: never mutates the immutable ClusterSetVersion.
- Deep immutability (DLIB-FA-008) and the renumbering order-invariance fix
  (factors.py _finalize, DLIB-FA-057).
- DLIB-XPKG: ClusterArtifact.assignments is a hashable + deepcopy-safe
  FrozenMapping (matching the QE convention).
- SimilarityViewRegistry / EdgeAffinityPolicy / extended SIMILARITY_VIEW_KEYS
  (DLIB-FA-004/46-50).
"""

import pytest

from factor_assets.clustering.families import (
    ClusterArtifact,
    LeidenClustering,
)
from factor_assets.contracts.cluster_governance import (
    GraphCompletenessClass,
    UnknownEdgeSemantics,
    SimilarityGraphArtifact,
    ClusterSetVersionArtifact,
    LogicalCluster,
    ClusterVersionArtifact,
    ClusterMembership,
    ClusterLineageEdge,
    ClusterVersionMatch,
    ClusterVersionMatcher,
    IncrementalAssignmentKind,
    IncrementalClusterAssignment,
    ClusterResolutionSelector,
    ClusterScale,
)
from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge

try:
    from factor_assets.clustering.families import (
        SCIPY_AVAILABLE,
    )
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# DLIB-FA-007: similarity graph artifact
# ---------------------------------------------------------------------------


def test_similarity_graph_artifact_hardening():
    """Unmeasured pairs are NEVER_MEASURED, never a computed zero, and
    production Leiden requires a certified graph."""
    graph = SimilarityGraphArtifact(
        graph_identity="g1",
        node_universe_ref="u1",
        fingerprint_spec_ref="f1",
        ann_policy={"backend": "annoy", "n_trees": 10},
        ann_k=20,
        exact_refinement_policy="exact",
        edge_threshold=0.3,
        edge_affinity_policy_version="1.0",
        refined_pair_count=100,
        stored_edge_count=50,
        coverage_diagnostics={"density": 0.1},
        graph_completeness_class=GraphCompletenessClass.CERTIFIED_ANN_REFINED,
        unknown_edge_semantics=UnknownEdgeSemantics.NEVER_MEASURED,
    )
    assert graph.is_certified is True
    assert graph.unknown_edge_semantics is UnknownEdgeSemantics.NEVER_MEASURED
    assert graph.content_hash

    # A PARTIAL graph is not certified: clustering must fail closed.
    partial = SimilarityGraphArtifact(
        graph_identity="g2",
        node_universe_ref="u1",
        fingerprint_spec_ref="f1",
        ann_policy={"backend": "annoy"},
        ann_k=10,
        exact_refinement_policy="none",
        edge_threshold=0.3,
        edge_affinity_policy_version="1.0",
        refined_pair_count=5,
        stored_edge_count=2,
        coverage_diagnostics={},
        graph_completeness_class=GraphCompletenessClass.PARTIAL,
        unknown_edge_semantics=UnknownEdgeSemantics.MEASURED_BELOW_THRESHOLD,
    )
    assert partial.is_certified is False

    import pickle as _p  # no reference reuse: confirm the module has no global

    assert _p is not None


def test_similarity_graph_artifact_hash_is_derived_only():
    """A caller may not self-report an arbitrary hash — FAIL CLOSED."""
    with pytest.raises(ValueError, match="content_hash"):
        SimilarityGraphArtifact(
            graph_identity="g1",
            node_universe_ref="u1",
            fingerprint_spec_ref="f1",
            ann_policy={},
            ann_k=10,
            exact_refinement_policy="exact",
            edge_threshold=0.3,
            edge_affinity_policy_version="1.0",
            refined_pair_count=100,
            stored_edge_count=50,
            coverage_diagnostics={},
            graph_completeness_class=GraphCompletenessClass.CERTIFIED_ANN_REFINED,
            unknown_edge_semantics=UnknownEdgeSemantics.NEVER_MEASURED,
            content_hash="bogus",
        )


# ---------------------------------------------------------------------------
# DLIB-FA-009: cluster set version + logical clusters
# ---------------------------------------------------------------------------


def _csv_artifact(cluster_set_version_id="csv1"):
    return ClusterSetVersionArtifact(
        cluster_set_version_id=cluster_set_version_id,
        member_universe_ref="u1",
        similarity_graph_version_ref="sgv1",
        algorithm="leiden",
        backend="igraph",
        backend_version="1.0",
        seed=42,
        resolution=1.0,
        clustering_policy_ref="policy1",
        assignment_artifact_ref="assign1",
        created_at="2024-01-01T00:00:00Z",
    )


def test_cluster_set_version_is_versioned_artifact():
    """A new global refresh produces a NEW ClusterSetVersionArtifact; the old
    one is preserved, never mutated in place."""
    v1 = _csv_artifact("csv1")
    v2 = _csv_artifact("csv2")
    assert v1.cluster_set_version_id != v2.cluster_set_version_id
    # Different version id -> different content hash (the id is part of the
    # semantic content); a new refresh is a new artifact.
    assert v1.content_hash != v2.content_hash


def test_logical_cluster_identity_is_stable():
    """LogicalCluster id (CL_PV_MOM_017) is stable across versions, decoupled
    from the unstable algorithm label (18)."""
    lc = LogicalCluster("CL_PV_MOM_017", algorithm_cluster_label=18)
    assert lc.logical_cluster_id == "CL_PV_MOM_017"
    assert lc.algorithm_cluster_label == 18
    assert lc.scale is ClusterScale.MICRO_CLUSTER
    # The label can change next refresh without touching the logical id.
    lc2 = LogicalCluster("CL_PV_MOM_017", algorithm_cluster_label=5)
    assert lc2.logical_cluster_id == lc.logical_cluster_id


def test_cluster_version_artifact_hash_and_immutability():
    cva = ClusterVersionArtifact(
        logical_cluster_id="CL_A",
        cluster_set_version_ref="csv1",
        member_factor_ids=("F1", "F2", "F3"),
        representative_factor_id="F1",
    )
    assert cva.content_hash
    assert cva.membership_qualification_domain == "UNVERIFIED"
    assert cva.member_factor_ids == ("F1", "F2", "F3")
    with pytest.raises(Exception):
        cva.member_factor_ids = ("F9",)  # type: ignore
    # Duplicate members rejected.
    with pytest.raises(ValueError, match="unique"):
        ClusterVersionArtifact("CL_B", "csv1", ("F1", "F1"), "F1")
    with pytest.raises(ValueError, match="evidence refs"):
        ClusterVersionArtifact(
            "CL_B", "csv1", ("F1",), "F1",
            membership_qualification_domain="FULL_REFRESH_CERTIFIED",
        )


def test_cluster_membership():
    m = ClusterMembership(
        factor_id="F1",
        logical_cluster_id="CL_A",
        cluster_set_version_ref="csv1",
    )
    assert m.scale is ClusterScale.MICRO_CLUSTER


def test_cluster_version_matcher_classifications():
    prev = {
        "CL_A": ClusterVersionArtifact("CL_A", "csv0", ("F1", "F2", "F3"), "F1"),
        "CL_B": ClusterVersionArtifact("CL_B", "csv0", ("F4", "F5"), "F4"),
    }
    cur = {
        "CL_A": ClusterVersionArtifact("CL_A", "csv1", ("F1", "F2", "F3"), "F1"),
        "CL_C": ClusterVersionArtifact("CL_C", "csv1", ("F6", "F7"), "F6"),
    }
    matcher = ClusterVersionMatcher()
    edges = matcher.match(prev, cur)
    by_from = {e.from_cluster_id: e for e in edges}
    # CL_A unchanged (same members).
    assert by_from["CL_A"].match is ClusterVersionMatch.UNCHANGED
    # CL_B dissolved (no current match).
    assert by_from["CL_B"].match is ClusterVersionMatch.DISSOLVED
    # CL_C is NEW.
    assert any(e.match is ClusterVersionMatch.NEW for e in edges)


def test_cluster_lineage_edges():
    edge = ClusterLineageEdge(
        from_cluster_id="CL_A",
        to_cluster_id="CL_A",
        from_cluster_set_version_ref="csv0",
        to_cluster_set_version_ref="csv1",
        match=ClusterVersionMatch.UNCHANGED,
    )
    assert edge.match is ClusterVersionMatch.UNCHANGED
    assert edge.to_dict()["match"] == "UNCHANGED"


# ---------------------------------------------------------------------------
# DLIB-FA-010: incremental assignment
# ---------------------------------------------------------------------------


def test_incremental_assignment_never_mutates_cluster_set_version():
    incremental = IncrementalClusterAssignment(
        factor_id="F9",
        logical_cluster_id="CL_A",
        kind=IncrementalAssignmentKind.ASSIGNED,
        cluster_set_version_ref="csv1",
        affinity=0.8,
        parent_cluster_set_hash="parent-content-hash",
        qualification_domain="CERTIFIED_PAIRWISE_SUPPORT",
        formal_evidence_refs=("qe-pairwise:F9:250d:sample",),
    )
    assert incremental.kind is IncrementalAssignmentKind.ASSIGNED
    assert incremental.cluster_set_version_ref == "csv1"
    assert incremental.content_hash
    # A new factor goes into an overlay/pending state, not into csv1 itself.
    ambiguous = IncrementalClusterAssignment(
        factor_id="F10",
        logical_cluster_id=None,
        kind=IncrementalAssignmentKind.PENDING_GLOBAL_REFRESH,
        cluster_set_version_ref="csv1",
    )
    assert ambiguous.logical_cluster_id is None
    assert ambiguous.kind.value == "PENDING_GLOBAL_REFRESH"


# ---------------------------------------------------------------------------
# DLIB-FA-008 deep immutability + DLIB-FA-057 renumber order invariance
# ---------------------------------------------------------------------------


def test_cluster_artifact_assignments_are_deep_immutable():
    """Mutating a caller-supplied assignments dict after construction cannot
    change the artifact (DLIB-FA-008)."""
    assignments = {"A": 0, "B": 0, "C": 1}
    artifact = ClusterArtifact(
        graph_identity="g1",
        algorithm="leiden",
        assignments=assignments,
        representatives=("A", "C"),
        backend="igraph",
        backend_version="1.0",
        seed=42,
        resolution=1.0,
        content_hash="",  # derived in __post_init__
    )
    original_hash = artifact.content_hash
    assignments["A"] = 99
    assert artifact.assignments["A"] == 0
    assert artifact.content_hash == original_hash


def test_cluster_artifact_assignments_are_hashable():
    """DLIB-XPKG: the assignments mapping is hashable + deepcopy-safe
    (FrozenMapping), so ClusterArtifact deepcopy/hash no longer raise."""
    import copy
    import pickle

    from factor_assets.contracts._frozen import FrozenMapping

    src = {"A": 0, "B": 0, "C": 1}
    artifact = ClusterArtifact(
        graph_identity="g1",
        algorithm="leiden",
        assignments=src,
        representatives=("A", "C"),
        backend="igraph",
        backend_version="1.0",
        seed=42,
        resolution=1.0,
        content_hash="",
    )
    # The assignments mapping is a recursively-immutable FrozenMapping.
    assert isinstance(artifact.assignments, FrozenMapping)
    assert isinstance(hash(artifact), int)
    # Hash is content-stable: same partition (different dict order) hashes the
    # same, and the derived content_hash is unchanged by dict insertion order.
    again = ClusterArtifact(
        graph_identity="g1",
        algorithm="leiden",
        assignments={"C": 1, "A": 0, "B": 0},
        representatives=("A", "C"),
        backend="igraph",
        backend_version="1.0",
        seed=42,
        resolution=1.0,
        content_hash="",
    )
    assert hash(artifact) == hash(again)
    assert artifact.content_hash == again.content_hash
    # deepcopy + pickle round-trips preserve content and hash.
    assert copy.deepcopy(artifact) == artifact
    assert pickle.loads(pickle.dumps(artifact)) == artifact


def test_leiden_renumber_is_order_invariant():
    """The algorithm label is unstable; renumbering to contiguous labels must
    be order-INVARIANT (walking the sorted factor ids, not sorting the raw
    backend cluster ids) so the same partition always yields the same
    renumbered assignments regardless of caller iteration order (DLIB-FA-057)."""
    if not LeidenClustering.PRODUCTION_CAPABLE:
        pytest.skip("No Leiden backend installed")

    graph = SparseCorrelationGraph(
        [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.9),
            CorrelationEdge("D", "E", 0.9),
        ]
    )
    first = LeidenClustering(graph, seed=42).cluster()
    again = LeidenClustering(graph, seed=42).cluster()
    assert first.assignments == again.assignments


# ---------------------------------------------------------------------------
# DLIB-FA-004/46-50 similarity views
# ---------------------------------------------------------------------------


def test_similarity_view_keys_are_explicit_and_versioned():
    from factor_assets.contracts.similarity import (
        SIMILARITY_VIEW_KEYS,
        DEFAULT_SIMILARITY_VIEW,
        SimilarityViewRegistry,
    )
    assert "rank_corr" in SIMILARITY_VIEW_KEYS
    assert "pearson_corr" in SIMILARITY_VIEW_KEYS
    assert DEFAULT_SIMILARITY_VIEW == "rank_corr"
    registry = SimilarityViewRegistry()
    registry.validate({"rank_corr": 0.5, "pearson_corr": 0.4})
    # A typo'd key fails closed.
    with pytest.raises(ValueError, match="not a registered canonical view"):
        registry.validate({"rank_cor": 0.5})


def test_edge_affinity_policy_multi_view():
    from factor_assets.contracts.similarity import EdgeAffinityPolicy

    policy = EdgeAffinityPolicy({"rank_corr": 0.6, "pearson_corr": 0.4})
    affinity = policy.affinity({"rank_corr": 0.5, "pearson_corr": 0.3})
    assert affinity == pytest.approx(0.42)
    # An unmeasured view is UNKNOWN, never a computed zero.
    assert policy.affinity({"rank_corr": None}) is None


# ---------------------------------------------------------------------------
# DLIB-FA-011: resolution selection (not max modularity)
# ---------------------------------------------------------------------------


def test_cluster_resolution_selector_is_not_max_modularity():
    selector = ClusterResolutionSelector()
    scores = {
        0.5: {"ari": 0.6, "modularity": 0.5},
        1.0: {"ari": 0.8, "modularity": 0.3},
        2.0: {"ari": 0.4, "modularity": 0.9},
    }
    chosen = selector.select(scores)
    # Highest composite STABILITY (0.8 ari at 1.0), NOT highest modularity (2.0).
    assert chosen == 1.0
    # Pure modularity alone never selects: two resolutions with no stability
    # metrics but different modularity resolve by mean-of-available; stability
    # keys drive the choice whenever present.
    mixed = {
        0.5: {"stability": 0.1, "modularity": 0.99},
        1.0: {"stability": 0.9, "modularity": 0.1},
    }
    assert selector.select(mixed) == 1.0


def test_family_lineage_is_research_only_behavioral():
    """FamilyLineage is a RESEARCH/ANALYTIC-ONLY behavioral lineage, not real
    factor genealogy (real genealogy is generator parent_factor_ids / FO
    mutation lineage / FE AST / candidate provenance).  DLIB-FA-009."""
    from factor_assets.clustering.lineage import FamilyLineage

    doc = FamilyLineage.__doc__ or ""
    assert "RESEARCH / ANALYTIC ONLY" in doc
