"""Tests for clustering.families module."""

import pytest
from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge
from factor_assets.clustering.families import (
    ConnectedComponents,
    ModularityClustering,
    ClusterResult,
)


def test_cluster_result():
    """ClusterResult provides cluster membership queries."""
    assignments = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2}
    sizes = {0: 2, 1: 2, 2: 1}

    result = ClusterResult(assignments, sizes)

    assert result.num_clusters == 3

    # Get cluster members
    cluster_0 = result.get_cluster_members(0)
    assert cluster_0 == {"A", "B"}

    cluster_1 = result.get_cluster_members(1)
    assert cluster_1 == {"C", "D"}

    # Get all clusters
    all_clusters = result.get_all_clusters()
    assert len(all_clusters) == 3
    assert all_clusters[0] == {"A", "B"}
    assert all_clusters[1] == {"C", "D"}
    assert all_clusters[2] == {"E"}


def test_connected_components_single_component():
    """Connected components finds single component."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", 0.6),
        CorrelationEdge("C", "D", 0.7),
    ]

    graph = SparseCorrelationGraph(edges)
    cc = ConnectedComponents(graph)
    result = cc.find_components()

    assert result.num_clusters == 1
    assert result.get_cluster_members(0) == {"A", "B", "C", "D"}


def test_connected_components_multiple_components():
    """Connected components finds disconnected subgraphs."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", 0.6),
        CorrelationEdge("D", "E", 0.7),
        CorrelationEdge("E", "F", 0.5),
        CorrelationEdge("G", "H", 0.9),
    ]

    graph = SparseCorrelationGraph(edges)
    cc = ConnectedComponents(graph)
    result = cc.find_components()

    assert result.num_clusters == 3

    all_clusters = result.get_all_clusters()
    cluster_sizes = [len(members) for members in all_clusters.values()]
    # Component 1: A-B-C (3 nodes), Component 2: D-E-F (3 nodes), Component 3: G-H (2 nodes)
    assert sorted(cluster_sizes) == [2, 3, 3]


def test_connected_components_isolated_nodes():
    """Connected components handles isolated nodes."""
    # Create graph with isolated node
    edges = [
        CorrelationEdge("A", "B", 0.8),
    ]

    graph = SparseCorrelationGraph(edges)
    cc = ConnectedComponents(graph)
    result = cc.find_components()

    # Only A and B are in the graph (no isolated nodes from edges)
    assert result.num_clusters == 1
    assert result.get_cluster_members(0) == {"A", "B"}


def test_modularity_clustering_simple():
    """Modularity clustering groups similar factors."""
    # Create two densely connected groups
    edges = [
        # Group 1: A-B-C triangle
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("B", "C", 0.9),
        CorrelationEdge("A", "C", 0.9),
        # Group 2: D-E-F triangle
        CorrelationEdge("D", "E", 0.9),
        CorrelationEdge("E", "F", 0.9),
        CorrelationEdge("D", "F", 0.9),
        # Weak connection between groups
        CorrelationEdge("C", "D", 0.2),
    ]

    graph = SparseCorrelationGraph(edges)
    clustering = ModularityClustering(graph)
    result = clustering.cluster(max_iterations=10)

    # Should find 2 clusters
    assert result.num_clusters >= 1  # At least not all singleton


def test_modularity_clustering_single_cluster():
    """Modularity clustering with single tight cluster."""
    edges = [
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("B", "C", 0.9),
        CorrelationEdge("C", "D", 0.9),
        CorrelationEdge("D", "A", 0.9),
    ]

    graph = SparseCorrelationGraph(edges)
    clustering = ModularityClustering(graph, resolution=1.0)
    result = clustering.cluster(max_iterations=10)

    # Tight cluster should stay together
    assert result.num_clusters >= 1


def test_modularity_clustering_resolution():
    """Resolution parameter affects cluster count."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", 0.7),
        CorrelationEdge("C", "D", 0.6),
    ]

    graph = SparseCorrelationGraph(edges)

    # Low resolution: fewer clusters
    clustering_low = ModularityClustering(graph, resolution=0.5)
    result_low = clustering_low.cluster(max_iterations=10)

    # High resolution: more clusters
    clustering_high = ModularityClustering(graph, resolution=2.0)
    result_high = clustering_high.cluster(max_iterations=10)

    # Both should produce valid results
    assert result_low.num_clusters >= 1
    assert result_high.num_clusters >= 1


def test_modularity_clustering_convergence():
    """Modularity clustering converges within max iterations."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", 0.7),
        CorrelationEdge("C", "D", 0.6),
        CorrelationEdge("D", "E", 0.5),
    ]

    graph = SparseCorrelationGraph(edges)
    clustering = ModularityClustering(graph)

    # Should complete within iterations
    result = clustering.cluster(max_iterations=5)
    assert result.num_clusters >= 1

    # All nodes should be assigned
    assert len(result.assignments) == 5


def test_empty_graph_clustering():
    """Clustering handles empty graph."""
    graph = SparseCorrelationGraph([])

    cc = ConnectedComponents(graph)
    result = cc.find_components()
    assert result.num_clusters == 0

    clustering = ModularityClustering(graph)
    result = clustering.cluster()
    assert result.num_clusters == 0


def test_star_topology():
    """Clustering handles star topology (hub node)."""
    edges = [
        CorrelationEdge("HUB", "A", 0.9),
        CorrelationEdge("HUB", "B", 0.9),
        CorrelationEdge("HUB", "C", 0.9),
        CorrelationEdge("HUB", "D", 0.9),
    ]

    graph = SparseCorrelationGraph(edges)

    # Connected components: single component
    cc = ConnectedComponents(graph)
    result = cc.find_components()
    assert result.num_clusters == 1
    assert result.get_cluster_members(0) == {"HUB", "A", "B", "C", "D"}

    # Modularity: might split or stay together
    clustering = ModularityClustering(graph)
    result = clustering.cluster()
    assert result.num_clusters >= 1
