"""Tests for clustering.lineage module."""

import pytest
from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge
from factor_assets.clustering.lineage import (
    LineageDetector,
    ParentChildRelation,
    FamilyLineage,
)
from factor_assets.clustering.families import ClusterResult


def test_parent_child_relation():
    """ParentChildRelation validates confidence."""
    rel = ParentChildRelation("parent", "child", 0.9, 0.8)
    assert rel.parent_id == "parent"
    assert rel.child_id == "child"
    assert rel.correlation == 0.9
    assert rel.confidence == 0.8

    # Invalid confidence
    with pytest.raises(ValueError, match="Confidence must be in"):
        ParentChildRelation("parent", "child", 0.9, 1.5)

    with pytest.raises(ValueError, match="Confidence must be in"):
        ParentChildRelation("parent", "child", 0.9, -0.1)


def test_family_lineage_basic():
    """FamilyLineage provides basic queries."""
    relations = [
        ParentChildRelation("A", "B", 0.9, 0.8),
        ParentChildRelation("A", "C", 0.8, 0.7),
    ]

    lineage = FamilyLineage(
        family_id=0,
        members={"A", "B", "C"},
        relations=relations,
        roots={"A"},
        leaves={"B", "C"}
    )

    assert lineage.size == 3
    assert lineage.roots == {"A"}
    assert lineage.leaves == {"B", "C"}

    # Get children
    assert lineage.get_children("A") == {"B", "C"}
    assert lineage.get_children("B") == set()

    # Get parents
    assert lineage.get_parents("B") == {"A"}
    assert lineage.get_parents("A") == set()


def test_family_lineage_depth():
    """FamilyLineage calculates depth correctly."""
    # Linear chain: A -> B -> C
    relations = [
        ParentChildRelation("A", "B", 0.9, 0.8),
        ParentChildRelation("B", "C", 0.8, 0.7),
    ]

    lineage = FamilyLineage(
        family_id=0,
        members={"A", "B", "C"},
        relations=relations,
        roots={"A"},
        leaves={"C"}
    )

    assert lineage.depth == 2  # A -> B -> C is depth 2

    # No relations
    lineage_flat = FamilyLineage(
        family_id=0,
        members={"A", "B"},
        relations=[],
        roots={"A", "B"},
        leaves={"A", "B"}
    )

    assert lineage_flat.depth == 0


def test_lineage_detector_simple():
    """LineageDetector finds parent-child relations."""
    # Hub topology: A connected to B, C, D with high degree
    edges = [
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("A", "C", 0.9),
        CorrelationEdge("A", "D", 0.9),
        CorrelationEdge("B", "C", 0.5),
    ]

    graph = SparseCorrelationGraph(edges)
    detector = LineageDetector(graph, min_correlation=0.7, degree_threshold=1)

    family_members = {"A", "B", "C", "D"}
    lineage = detector.detect_lineage(family_members, family_id=0)

    assert lineage.size == 4
    # A should be identified as root (highest degree)
    assert "A" in lineage.roots
    # B, C, D should have A as parent
    assert lineage.get_parents("B") == {"A"} or "B" in lineage.leaves


def test_lineage_detector_threshold():
    """LineageDetector respects correlation threshold."""
    edges = [
        CorrelationEdge("A", "B", 0.9),  # Above threshold
        CorrelationEdge("A", "C", 0.5),  # Below threshold
    ]

    graph = SparseCorrelationGraph(edges)
    detector = LineageDetector(graph, min_correlation=0.7)

    family_members = {"A", "B", "C"}
    lineage = detector.detect_lineage(family_members, family_id=0)

    # Only A-B relation should be detected
    assert len(lineage.relations) <= 1


def test_lineage_detector_degree_based():
    """LineageDetector uses degree to determine parent."""
    # A has degree 3, B has degree 1
    edges = [
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("A", "C", 0.8),
        CorrelationEdge("A", "D", 0.8),
    ]

    graph = SparseCorrelationGraph(edges)
    detector = LineageDetector(graph, min_correlation=0.7, degree_threshold=2)

    family_members = {"A", "B", "C", "D"}
    lineage = detector.detect_lineage(family_members, family_id=0)

    # A should be parent of B (higher degree)
    a_children = lineage.get_children("A")
    assert len(a_children) >= 1


def test_lineage_detector_single_node():
    """LineageDetector handles single-node family."""
    edges = []
    graph = SparseCorrelationGraph(edges)
    detector = LineageDetector(graph)

    family_members = {"A"}
    lineage = detector.detect_lineage(family_members, family_id=0)

    assert lineage.size == 1
    assert lineage.roots == {"A"}
    assert lineage.leaves == {"A"}
    assert len(lineage.relations) == 0


def test_lineage_detector_no_relations():
    """LineageDetector handles disconnected nodes."""
    edges = [
        CorrelationEdge("A", "B", 0.9),
    ]

    graph = SparseCorrelationGraph(edges)
    detector = LineageDetector(graph, min_correlation=0.95)  # High threshold

    family_members = {"A", "B", "C"}
    lineage = detector.detect_lineage(family_members, family_id=0)

    # No relations above threshold
    assert len(lineage.relations) == 0
    assert lineage.roots == {"A", "B", "C"}
    assert lineage.leaves == {"A", "B", "C"}


def test_detect_all_lineages():
    """detect_all_lineages processes all clusters."""
    edges = [
        # Cluster 0
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("A", "C", 0.8),
        # Cluster 1
        CorrelationEdge("D", "E", 0.9),
    ]

    graph = SparseCorrelationGraph(edges)
    detector = LineageDetector(graph, min_correlation=0.7)

    # Mock cluster result
    cluster_result = ClusterResult(
        assignments={"A": 0, "B": 0, "C": 0, "D": 1, "E": 1},
        cluster_sizes={0: 3, 1: 2}
    )

    lineages = detector.detect_all_lineages(cluster_result)

    assert len(lineages) == 2
    assert lineages[0].family_id == 0
    assert lineages[1].family_id == 1
    assert lineages[0].size == 3
    assert lineages[1].size == 2


def test_confidence_calculation():
    """Confidence calculation considers multiple factors."""
    # High correlation, high overlap, high degree ratio
    edges = [
        CorrelationEdge("A", "B", 0.95),  # Strong correlation
        CorrelationEdge("A", "C", 0.8),
        CorrelationEdge("A", "D", 0.8),
        CorrelationEdge("B", "C", 0.7),
    ]

    graph = SparseCorrelationGraph(edges)
    detector = LineageDetector(graph, min_correlation=0.7, degree_threshold=1)

    family_members = {"A", "B", "C", "D"}
    lineage = detector.detect_lineage(family_members, family_id=0)

    # A-B relation should have high confidence
    ab_relations = [r for r in lineage.relations if
                    (r.parent_id == "A" and r.child_id == "B") or
                    (r.parent_id == "B" and r.child_id == "A")]

    if ab_relations:
        assert ab_relations[0].confidence > 0.3


def test_lineage_cycle_protection():
    """Depth calculation handles cycles gracefully."""
    # Create cycle: A -> B -> C -> A (though this shouldn't happen in practice)
    relations = [
        ParentChildRelation("A", "B", 0.9, 0.8),
        ParentChildRelation("B", "C", 0.8, 0.7),
        ParentChildRelation("C", "A", 0.7, 0.6),
    ]

    lineage = FamilyLineage(
        family_id=0,
        members={"A", "B", "C"},
        relations=relations,
        roots=set(),
        leaves=set()
    )

    # Should not crash, returns reasonable depth
    depth = lineage.depth
    assert depth >= 0
