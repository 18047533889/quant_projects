"""R55 P0-13: production-mode certified-graph enforcement for clustering.

Covers the single gate (:func:`factor_assets.clustering.certification.
enforce_certified_graph`) and its wiring at EVERY clustering entry point:

(a) production + uncertified graph -> raises ProductionClusterViolation;
(b) production + certified graph   -> proceeds and records the certification
    id (and execution mode) on the result artifact;
(c) research + uncertified graph   -> allowed (the relaxed path is retained);
(d) production + graph whose content hash differs from the certification's
    recorded hash (stale / mutated graph) -> raises;
(e) no bypass: every production-mode clustering entry point hits the gate.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from factor_assets.clustering import (
    CertifiedGraphArtifact,
    ClusterQualityPolicy,
    ConnectedComponents,
    ExecutionMode,
    GraphCertificationStatus,
    HierarchicalClustering,
    LeidenClustering,
    ModularityClustering,
    ProductionClusterViolation,
    compute_graph_content_hash,
    enforce_certified_graph,
    graph_summary,
)
from factor_assets.clustering import certification as cert_module
from factor_assets.clustering.certification import (
    CERTIFIED_COMPLETENESS_CLASSES,
)
from factor_assets.graph.sparse import CorrelationEdge, SparseCorrelationGraph


#: A freeze window comfortably in the future relative to any test clock.
FROZEN_UNTIL = "2099-01-01T00:00:00Z"


def _two_triangles_graph():
    """Six factors in two dense triangles, weakly bridged (same fixture the
    Leiden production tests use)."""
    edges = [
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("B", "C", 0.9),
        CorrelationEdge("A", "C", 0.9),
        CorrelationEdge("D", "E", 0.9),
        CorrelationEdge("E", "F", 0.9),
        CorrelationEdge("D", "F", 0.9),
        CorrelationEdge("C", "D", 0.2),
    ]
    return SparseCorrelationGraph(edges)


def _certify(
    graph,
    *,
    certification_id="cert-1",
    completeness="CERTIFIED_ANN_REFINED",
    algorithms=("leiden",),
    status=GraphCertificationStatus.CERTIFIED,
    frozen_until=FROZEN_UNTIL,
):
    """Build a CertifiedGraphArtifact that certifies ``graph``."""
    nodes, edge_records = graph_summary(graph)
    return CertifiedGraphArtifact(
        certification_id=certification_id,
        graph_content_hash=compute_graph_content_hash(
            node_universe=nodes, edge_list=edge_records
        ),
        node_count=len(nodes),
        edge_count=len(edge_records),
        graph_completeness_class=completeness,
        allowed_algorithms=tuple(algorithms),
        status=status,
        certified_at="2026-01-01T00:00:00Z",
        frozen_until=frozen_until,
        certified_by="r55-p0-13-tests",
        evidence_refs=("evidence:graph-cert",),
        notes={"reason": "unit-test certification"},
    )


PRODUCTION = ExecutionMode.PRODUCTION
RESEARCH = ExecutionMode.RESEARCH


# ---------------------------------------------------------------------------
# (a) production + uncertified graph -> raises
# ---------------------------------------------------------------------------


def test_production_leiden_without_certification_raises():
    graph = _two_triangles_graph()
    with pytest.raises(ProductionClusterViolation, match="no graph certification"):
        LeidenClustering(graph, execution_mode=PRODUCTION)


def test_production_leiden_rejects_non_certified_object():
    graph = _two_triangles_graph()
    with pytest.raises(ProductionClusterViolation, match="CertifiedGraphArtifact"):
        LeidenClustering(graph, execution_mode=PRODUCTION, certification={"id": "x"})


def test_production_pending_or_revoked_certification_raises():
    graph = _two_triangles_graph()
    for status in (GraphCertificationStatus.PENDING, GraphCertificationStatus.REVOKED):
        cert = _certify(graph, certification_id=f"c-{status.value}", status=status)
        with pytest.raises(ProductionClusterViolation, match=status.value):
            LeidenClustering(graph, execution_mode=PRODUCTION, certification=cert)


def test_production_non_certified_completeness_class_raises():
    graph = _two_triangles_graph()
    cert = _certify(graph, completeness="PARTIAL")
    with pytest.raises(ProductionClusterViolation, match="graph_completeness_class"):
        LeidenClustering(graph, execution_mode=PRODUCTION, certification=cert)


def test_production_stale_freeze_window_raises():
    graph = _two_triangles_graph()
    cert = _certify(graph, frozen_until="2026-02-01T00:00:00Z")
    with pytest.raises(ProductionClusterViolation, match="stale"):
        LeidenClustering(
            graph,
            execution_mode=PRODUCTION,
            certification=cert,
        ).cluster(now="2026-03-01T00:00:00Z")


def test_production_algorithm_not_allowed_raises():
    graph = _two_triangles_graph()
    cert = _certify(graph, algorithms=("leiden",))
    # The toy/research algorithms are refused: they are neither production-
    # capable nor on the certification's allow-list.
    with pytest.raises(ProductionClusterViolation, match="not production-capable"):
        ModularityClustering(
            graph,
            allow_toy_algorithm=True,
            execution_mode=PRODUCTION,
            certification=cert,
        ).cluster()
    with pytest.raises(ProductionClusterViolation, match="not production-capable"):
        ConnectedComponents(
            graph, execution_mode=PRODUCTION, certification=cert
        ).find_components()
    with pytest.raises(ProductionClusterViolation, match="not production-capable"):
        HierarchicalClustering(
            graph, execution_mode=PRODUCTION, certification=cert
        ).build_dendrogram()


def test_production_leiden_not_on_certification_allow_list_raises():
    """A certification that names some OTHER algorithm cannot back a Leiden
    production run."""
    graph = _two_triangles_graph()
    cert = _certify(graph, algorithms=("louvain",))
    with pytest.raises(ProductionClusterViolation, match="allowed_algorithms"):
        LeidenClustering(graph, execution_mode=PRODUCTION, certification=cert)


def test_production_uncertified_algorithms_refused_entirely():
    """Only Leiden is production-capable; the other algorithms raise in
    production even with a valid certification that names them, because no
    certification may authorise a non-production algorithm (R55 P0-13)."""
    graph = _two_triangles_graph()
    permissive = _certify(
        graph,
        algorithms=("leiden", "connected_components", "modularity", "hierarchical"),
    )
    with pytest.raises(ProductionClusterViolation):
        ConnectedComponents(
            graph, execution_mode=PRODUCTION, certification=permissive
        ).find_components()
    with pytest.raises(ProductionClusterViolation):
        ModularityClustering(
            graph,
            allow_toy_algorithm=True,
            execution_mode=PRODUCTION,
            certification=permissive,
        ).cluster()
    with pytest.raises(ProductionClusterViolation):
        HierarchicalClustering(
            graph, execution_mode=PRODUCTION, certification=permissive
        ).build_dendrogram()


# ---------------------------------------------------------------------------
# (b) production + certified graph -> proceeds and records certification id
# ---------------------------------------------------------------------------


def test_production_leiden_with_certified_graph_proceeds():
    pytest.importorskip("igraph")
    graph = _two_triangles_graph()
    cert = _certify(graph, certification_id="cert-prod-1")
    result = LeidenClustering(
        graph, seed=42, execution_mode=PRODUCTION, certification=cert,
        quality_policy=ClusterQualityPolicy(.8, 'fixture-medoid', '1'),
    ).cluster()
    assert result.num_clusters == 2
    assert result.certification_id == "cert-prod-1"
    assert result.execution_mode == "PRODUCTION"
    assert result.graph_content_hash == cert.graph_content_hash


def test_production_leiden_cluster_artifact_records_certification():
    pytest.importorskip("igraph")
    graph = _two_triangles_graph()
    cert = _certify(graph, certification_id="cert-prod-2")
    artifact = LeidenClustering(
        graph, seed=42, execution_mode=PRODUCTION, certification=cert,
        quality_policy=ClusterQualityPolicy(.8, 'fixture-medoid', '1'),
    ).cluster_artifact(
        snapshot_ref="snapshot:2024",
        universe_ref="universe:ashare",
    )
    assert artifact.certification_id == "cert-prod-2"
    assert artifact.execution_mode == "PRODUCTION"
    assert artifact.algorithm == "leiden"
    assert artifact.content_hash
    assert artifact.assignments
    import json
    from factor_assets.clustering.families import ClusterArtifact
    restored = ClusterArtifact(**json.loads(json.dumps(artifact.to_dict())))
    assert restored.certification_id == artifact.certification_id
    assert restored.execution_mode == "PRODUCTION"
    assert restored.content_hash == artifact.content_hash


def test_production_accepts_string_mode_and_exactly_certified_classes():
    pytest.importorskip("igraph")
    graph = _two_triangles_graph()
    for completeness in CERTIFIED_COMPLETENESS_CLASSES:
        cert = _certify(graph, completeness=completeness)
        result = LeidenClustering(
            graph, execution_mode="PRODUCTION", certification=cert,
            quality_policy=ClusterQualityPolicy(.8, 'fixture-medoid', '1'),
        ).cluster()
        assert result.certification_id == "cert-1"
        assert result.execution_mode == "PRODUCTION"


# ---------------------------------------------------------------------------
# (c) research + uncertified -> allowed (relaxed path retained)
# ---------------------------------------------------------------------------


def test_research_mode_keeps_relaxed_path():
    pytest.importorskip("igraph")
    graph = _two_triangles_graph()
    # No certification at all.
    result = LeidenClustering(graph, seed=42).cluster()
    assert result.num_clusters == 2
    assert result.certification_id is None
    assert result.execution_mode == "RESEARCH"
    # Explicit research mode behaves identically.
    again = LeidenClustering(graph, seed=42, execution_mode=RESEARCH).cluster()
    assert again.assignments == result.assignments
    # The other (uncertified) algorithms keep working in research mode.
    cc = ConnectedComponents(graph).find_components()
    assert cc.num_clusters >= 1
    toy = ModularityClustering(graph, allow_toy_algorithm=True).cluster()
    assert toy.num_clusters >= 1
    dendro = HierarchicalClustering(
        graph, missing_distance_policy="treat_missing_as_max"
    ).build_dendrogram()
    assert dendro.factor_ids


def test_research_mode_with_a_stale_graph_is_still_allowed():
    """Research mode never content-hash-checks: the relaxed path is retained."""
    pytest.importorskip("igraph")
    graph = _two_triangles_graph()
    cert = _certify(graph)
    # Deliberately hand back a DIFFERENT graph than the one certified.
    other = SparseCorrelationGraph([CorrelationEdge("A", "B", 0.9)])
    result = LeidenClustering(
        other, seed=42, execution_mode=RESEARCH, certification=cert
    ).cluster()
    assert result.num_clusters >= 1


# ---------------------------------------------------------------------------
# (d) content-hash mismatch (stale / mutated graph) -> raises
# ---------------------------------------------------------------------------


def test_production_mutated_graph_fails_hash_check():
    graph = _two_triangles_graph()
    cert = _certify(graph)
    # Mutate one edge weight AFTER certification: the graph is no longer the
    # certified one.
    mutated = SparseCorrelationGraph(
        [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.9),
            CorrelationEdge("A", "C", 0.9),
            CorrelationEdge("D", "E", 0.9),
            CorrelationEdge("E", "F", 0.9),
            CorrelationEdge("D", "F", 0.9),
            CorrelationEdge("C", "D", 0.9),  # was 0.2
        ]
    )
    with pytest.raises(ProductionClusterViolation, match="content hash mismatch"):
        LeidenClustering(
            graph=mutated,
            execution_mode=PRODUCTION,
            certification=cert,
        )


def test_production_shrunk_graph_fails_node_count_check():
    """A certification whose recorded hash/counts were derived with the SAME
    node/edge totals but a wrong count field is caught by the structural
    check; here the hash rule fires first, so drive the node-count rule
    directly through the gate with a hash-consistent certification whose
    node_count lies."""
    graph = SparseCorrelationGraph([CorrelationEdge("A", "B", 0.9)])
    nodes, edge_records = graph_summary(graph)
    good_hash = compute_graph_content_hash(node_universe=nodes, edge_list=edge_records)
    lying = CertifiedGraphArtifact(
        certification_id="lying-nodes",
        graph_content_hash=good_hash,
        node_count=9,
        edge_count=len(edge_records),
        graph_completeness_class="CERTIFIED_EXACT",
        frozen_until=FROZEN_UNTIL,
    )
    with pytest.raises(ProductionClusterViolation, match="node count mismatch"):
        enforce_certified_graph(PRODUCTION, graph, lying, algorithm="leiden")
    # And a genuinely smaller graph than the certified one is caught by the
    # content-hash rule (checked first, strictly stronger).
    smaller = SparseCorrelationGraph([CorrelationEdge("A", "B", 0.4)])
    with pytest.raises(ProductionClusterViolation, match="content hash mismatch"):
        LeidenClustering(graph=smaller, execution_mode=PRODUCTION, certification=lying)


def test_production_added_edge_fails_edge_count_check():
    graph = SparseCorrelationGraph([CorrelationEdge("A", "B", 0.9)])
    nodes, edge_records = graph_summary(graph)
    good_hash = compute_graph_content_hash(node_universe=nodes, edge_list=edge_records)
    lying = CertifiedGraphArtifact(
        certification_id="lying-edges",
        graph_content_hash=good_hash,
        node_count=len(nodes),
        edge_count=5,
        graph_completeness_class="CERTIFIED_EXACT",
        frozen_until=FROZEN_UNTIL,
    )
    with pytest.raises(ProductionClusterViolation, match="edge count mismatch"):
        enforce_certified_graph(PRODUCTION, graph, lying, algorithm="leiden")


def test_gate_recomputes_hash_from_live_graph_not_from_caller():
    """The gate recomputes the hash from the graph handed to the entry point —
    a caller cannot present a valid certification recorded for other bytes."""
    graph = _two_triangles_graph()
    cert = _certify(graph)
    # Rebuild a graph with the same NODE SET but different weights: node count
    # matches, so only the recomputed hash catches it.
    same_nodes_other_weights = SparseCorrelationGraph(
        [
            CorrelationEdge("A", "B", 0.5),
            CorrelationEdge("B", "C", 0.5),
            CorrelationEdge("A", "C", 0.5),
            CorrelationEdge("D", "E", 0.5),
            CorrelationEdge("E", "F", 0.5),
            CorrelationEdge("D", "F", 0.5),
            CorrelationEdge("C", "D", 0.1),
        ]
    )
    with pytest.raises(ProductionClusterViolation, match="content hash mismatch"):
        enforce_certified_graph(
            PRODUCTION,
            same_nodes_other_weights,
            cert,
            algorithm="leiden",
        )


# ---------------------------------------------------------------------------
# (e) no bypass: every production-mode entry point hits the gate
# ---------------------------------------------------------------------------


def _assert_gate_called(monkeypatch, fn):
    """Helper kept for clarity: run ``fn`` with the gate instrumented."""
    calls = {"n": 0}
    real = cert_module.enforce_certified_graph

    def spy(mode, graph, certification, *, algorithm, now=None):
        calls["n"] += 1
        return real(mode, graph, certification, algorithm=algorithm, now=now)

    monkeypatch.setattr(cert_module, "enforce_certified_graph", spy)
    fn()
    return calls["n"]


def test_every_production_entry_point_hits_the_gate(monkeypatch):
    pytest.importorskip("igraph")
    graph = _two_triangles_graph()
    cert = _certify(graph)

    calls = {"n": 0}
    real = cert_module.enforce_certified_graph

    def spy(mode, g, c, *, algorithm, now=None):
        calls["n"] += 1
        return real(mode, g, c, algorithm=algorithm, now=now)

    monkeypatch.setattr(
        "factor_assets.clustering.families.enforce_certified_graph", spy
    )
    monkeypatch.setattr(
        "factor_assets.clustering.lineage.enforce_certified_graph", spy
    )

    # Leiden: constructor, cluster, cluster_artifact.
    with pytest.raises(ProductionClusterViolation):
        LeidenClustering(graph, execution_mode=PRODUCTION)
    assert calls["n"] >= 1
    before = calls["n"]
    LeidenClustering(graph, execution_mode=RESEARCH, certification=cert).cluster()
    assert calls["n"] > before
    before = calls["n"]
    LeidenClustering(graph, execution_mode=RESEARCH, certification=cert).cluster_artifact()
    assert calls["n"] > before

    # ConnectedComponents: find_components.
    before = calls["n"]
    ConnectedComponents(graph, execution_mode=RESEARCH, certification=cert).find_components()
    assert calls["n"] > before

    # ModularityClustering: constructor and cluster.
    before = calls["n"]
    ModularityClustering(
        graph, allow_toy_algorithm=True, execution_mode=RESEARCH, certification=cert
    )
    assert calls["n"] > before
    before = calls["n"]
    ModularityClustering(
        graph, allow_toy_algorithm=True, execution_mode=RESEARCH, certification=cert
    ).cluster()
    assert calls["n"] > before

    # HierarchicalClustering: build_dendrogram and cluster.
    hc = HierarchicalClustering(
        graph, missing_distance_policy="treat_missing_as_max",
        execution_mode=RESEARCH, certification=cert,
    )
    before = calls["n"]
    hc.build_dendrogram()
    assert calls["n"] > before
    before = calls["n"]
    hc.cluster()
    assert calls["n"] > before

    # LineageDetector: detect_lineage and detect_all_lineages.
    from factor_assets.clustering import LineageDetector
    from factor_assets.clustering.families import ClusterResult

    detector = LineageDetector(graph, execution_mode=RESEARCH, certification=cert)
    before = calls["n"]
    detector.detect_lineage({"A", "B", "C"}, 0)
    assert calls["n"] > before
    before = calls["n"]
    detector.detect_all_lineages(
        ClusterResult(assignments={"A": 0, "B": 0, "C": 0}, cluster_sizes={0: 3})
    )
    assert calls["n"] > before


def test_production_entry_points_refuse_even_when_construction_succeeded_in_research():
    """A caller cannot construct in research mode and then run production: the
    mode is carried on the run, and a research-constructed object has no
    production path.  Constructing directly in production is the only route,
    and that path is gated."""
    pytest.importorskip("igraph")
    graph = _two_triangles_graph()
    # A research-constructed object keeps clustering in research mode only.
    lc = LeidenClustering(graph, seed=42)
    assert lc.execution_mode is ExecutionMode.RESEARCH
    assert lc.cluster().execution_mode == "RESEARCH"

    # Production must be requested explicitly and is then gated.
    with pytest.raises(ProductionClusterViolation):
        LeidenClustering(graph, seed=42, execution_mode=PRODUCTION)


def test_gate_is_imported_and_used_by_both_clustering_modules():
    """The gate must be a single shared function wired into both modules (no
    divergent per-class implementation)."""
    import factor_assets.clustering.families as families_mod
    import factor_assets.clustering.lineage as lineage_mod

    assert families_mod.enforce_certified_graph is cert_module.enforce_certified_graph
    assert lineage_mod.enforce_certified_graph is cert_module.enforce_certified_graph


def test_production_allow_list_is_leiden_only():
    """No certification may authorise a non-Leiden production algorithm."""
    assert cert_module.PRODUCTION_ALLOWED_ALGORITHMS == ("leiden",)


# ---------------------------------------------------------------------------
# gate unit details
# ---------------------------------------------------------------------------


def test_execution_mode_coercion():
    assert ExecutionMode.coerce("PRODUCTION") is PRODUCTION
    assert ExecutionMode.coerce("production") is PRODUCTION
    assert ExecutionMode.coerce("RESEARCH") is RESEARCH
    assert ExecutionMode.coerce(ExecutionMode.PRODUCTION) is PRODUCTION
    with pytest.raises(ValueError, match="execution mode"):
        ExecutionMode.coerce("STAGING")


def test_certification_artifact_hash_is_derived_only():
    graph = _two_triangles_graph()
    nodes, edge_records = graph_summary(graph)
    good_hash = compute_graph_content_hash(node_universe=nodes, edge_list=edge_records)
    with pytest.raises(ValueError, match="content_hash"):
        CertifiedGraphArtifact(
            certification_id="c",
            graph_content_hash=good_hash,
            node_count=len(nodes),
            edge_count=len(edge_records),
            graph_completeness_class="CERTIFIED_EXACT",
            content_hash="bogus",
        )


def test_compute_graph_content_hash_is_order_invariant():
    edges = [
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("B", "C", 0.8),
        CorrelationEdge("A", "C", 0.7),
    ]
    g1 = SparseCorrelationGraph(edges, nodes=["A", "B", "C"])
    g2 = SparseCorrelationGraph(list(reversed(edges)), nodes=["C", "B", "A"])
    assert g1.graph_identity == g2.graph_identity
    n1, e1 = graph_summary(g1)
    n2, e2 = graph_summary(g2)
    assert compute_graph_content_hash(node_universe=n1, edge_list=e1) == (
        compute_graph_content_hash(node_universe=n2, edge_list=e2)
    )
    # A different graph hashes differently.
    g3 = SparseCorrelationGraph([CorrelationEdge("A", "B", 0.9)])
    n3, e3 = graph_summary(g3)
    assert compute_graph_content_hash(node_universe=n3, edge_list=e3) != (
        compute_graph_content_hash(node_universe=n1, edge_list=e1)
    )


def test_gate_clock_accepts_datetime_and_naive_utc():
    graph = _two_triangles_graph()
    cert = _certify(graph, frozen_until="2026-02-01T00:00:00Z")
    with pytest.raises(ProductionClusterViolation, match="stale"):
        enforce_certified_graph(
            PRODUCTION,
            graph,
            cert,
            algorithm="leiden",
            now=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )
    with pytest.raises(ProductionClusterViolation, match="stale"):
        enforce_certified_graph(
            PRODUCTION,
            graph,
            cert,
            algorithm="leiden",
            now=datetime(2026, 5, 1),  # naive treated as UTC
        )
    with pytest.raises(TypeError, match="now must be"):
        enforce_certified_graph(
            PRODUCTION, graph, cert, algorithm="leiden", now=12345
        )


def test_research_mode_passthrough_returns_certification():
    graph = _two_triangles_graph()
    cert = _certify(graph)
    assert (
        enforce_certified_graph(RESEARCH, graph, cert, algorithm="leiden") is cert
    )
    assert enforce_certified_graph(RESEARCH, graph, None, algorithm="leiden") is None


def test_leiden_result_records_research_mode_and_uncertified_provenance():
    pytest.importorskip("igraph")
    graph = _two_triangles_graph()
    result = LeidenClustering(graph, seed=42).cluster()
    assert result.certification_id is None
    assert result.graph_content_hash is None
    assert result.execution_mode == "RESEARCH"
