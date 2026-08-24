"""Tests for graph.sparse module."""

import pytest
from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge


def test_correlation_edge_validation():
    """Edge validation catches invalid correlations."""
    # Valid edge
    edge = CorrelationEdge("A", "B", 0.8)
    assert edge.factor_a == "A"
    assert edge.factor_b == "B"
    assert edge.correlation == 0.8
    assert edge.abs_correlation == 0.8

    # Negative correlation
    edge = CorrelationEdge("A", "B", -0.6)
    assert edge.abs_correlation == 0.6

    # Self-loop forbidden
    with pytest.raises(ValueError, match="Self-loops not allowed"):
        CorrelationEdge("A", "A", 0.5)

    # Invalid correlation range
    with pytest.raises(ValueError, match="Correlation must be in"):
        CorrelationEdge("A", "B", 1.5)

    with pytest.raises(ValueError, match="Correlation must be in"):
        CorrelationEdge("A", "B", -1.5)


def test_canonical_form():
    """Canonical form orders factors consistently."""
    edge1 = CorrelationEdge("B", "A", 0.5)
    edge2 = CorrelationEdge("A", "B", 0.5)

    assert edge1.canonical_form() == edge2.canonical_form()
    assert edge1.canonical_form() == ("A", "B")


def test_sparse_graph_construction():
    """Graph builds from edge list."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", 0.6),
        CorrelationEdge("A", "C", 0.7),
    ]

    graph = SparseCorrelationGraph(edges)

    assert graph.node_count == 3
    assert graph.edge_count == 3
    assert graph.nodes == {"A", "B", "C"}


def test_graph_deduplication():
    """Graph deduplicates exact edges and rejects conflicting values."""
    graph = SparseCorrelationGraph(
        [
            CorrelationEdge("A", "B", 0.8),
            CorrelationEdge("B", "A", 0.8),
        ]
    )

    assert graph.edge_count == 1
    assert graph.node_count == 2

    with pytest.raises(ValueError, match="Conflicting duplicate edge"):
        SparseCorrelationGraph(
            [
                CorrelationEdge("A", "B", 0.8),
                CorrelationEdge("A", "B", 0.9),
            ]
        )


def test_neighbors():
    """Neighbors returns correct adjacency list."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("A", "C", 0.6),
        CorrelationEdge("B", "C", 0.5),
    ]

    graph = SparseCorrelationGraph(edges)

    neighbors_a = graph.neighbors("A")
    assert len(neighbors_a) == 2
    assert set(n for n, _ in neighbors_a) == {"B", "C"}

    neighbors_b = graph.neighbors("B")
    assert len(neighbors_b) == 2
    assert set(n for n, _ in neighbors_b) == {"A", "C"}


def test_degree():
    """Degree returns neighbor count."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("A", "C", 0.6),
        CorrelationEdge("A", "D", 0.7),
    ]

    graph = SparseCorrelationGraph(edges)

    assert graph.degree("A") == 3
    assert graph.degree("B") == 1
    assert graph.degree("C") == 1
    assert graph.degree("D") == 1
    assert graph.degree("E") == 0  # Non-existent node


def test_has_edge():
    """has_edge checks edge existence."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", 0.6),
    ]

    graph = SparseCorrelationGraph(edges)

    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "A")  # Undirected
    assert graph.has_edge("B", "C")
    assert not graph.has_edge("A", "C")
    assert not graph.has_edge("A", "D")


def test_get_correlation():
    """get_correlation retrieves edge weight."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", -0.6),
    ]

    graph = SparseCorrelationGraph(edges)

    assert graph.get_correlation("A", "B") == 0.8
    assert graph.get_correlation("B", "A") == 0.8  # Undirected
    assert graph.get_correlation("B", "C") == -0.6
    assert graph.get_correlation("A", "C") is None


def test_subgraph():
    """Subgraph extracts induced subgraph."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", 0.6),
        CorrelationEdge("C", "D", 0.7),
        CorrelationEdge("A", "D", 0.5),
    ]

    graph = SparseCorrelationGraph(edges)

    # Extract subgraph with A, B, C
    subgraph = graph.subgraph({"A", "B", "C"})

    assert subgraph.node_count == 3
    assert subgraph.edge_count == 2  # A-B, B-C
    assert subgraph.has_edge("A", "B")
    assert subgraph.has_edge("B", "C")
    assert not subgraph.has_edge("A", "D")


def test_to_edge_list():
    """to_edge_list exports edges."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", 0.6),
    ]

    graph = SparseCorrelationGraph(edges)
    edge_list = graph.to_edge_list()

    assert len(edge_list) == 2
    edge_pairs = {edge.canonical_form() for edge in edge_list}
    assert ("A", "B") in edge_pairs
    assert ("B", "C") in edge_pairs


def test_density():
    """Density calculation."""
    # Complete graph K3
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", 0.6),
        CorrelationEdge("A", "C", 0.7),
    ]

    graph = SparseCorrelationGraph(edges)

    # K3 has 3 edges, max is 3*2/2 = 3
    assert graph.density() == 1.0

    # Partial graph
    edges = [CorrelationEdge("A", "B", 0.8)]
    graph = SparseCorrelationGraph(edges)

    # 2 nodes, 1 edge out of 1 possible
    assert graph.density() == 1.0

    # 3 nodes, 1 edge out of 3 possible
    edges = [CorrelationEdge("A", "B", 0.8)]
    graph = SparseCorrelationGraph(edges)
    # This will add nodes A and B only
    assert graph.node_count == 2


def test_empty_graph():
    """Empty graph handling."""
    graph = SparseCorrelationGraph([])

    assert graph.node_count == 0
    assert graph.edge_count == 0
    assert graph.nodes == set()
    assert graph.density() == 0.0
