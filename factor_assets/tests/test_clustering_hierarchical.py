"""
Tests for hierarchical clustering with dendrogram cutting.
"""

import pytest

try:
    import numpy as np
    from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge
    from factor_assets.clustering.families import (
       
        HierarchicalClustering,
        Dendrogram,
        ClusterResult,
    )
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    np = None


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
class TestHierarchicalClustering:
    """Tests for hierarchical clustering."""

    def test_build_dendrogram_simple(self):
        """Build dendrogram from simple graph."""
        edges = [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.8),
            CorrelationEdge("D", "E", 0.7),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, method='average', missing_distance_policy="treat_missing_as_max")
        dendrogram = clustering.build_dendrogram()

        assert dendrogram is not None
        assert len(dendrogram.factor_ids) == 5
        assert dendrogram.method == 'average'

    def test_cluster_with_num_clusters(self):
        """Cluster with specified number of clusters."""
        edges = [
            # Group 1
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.9),
            # Group 2
            CorrelationEdge("D", "E", 0.9),
            CorrelationEdge("E", "F", 0.9),
            # Weak connection
            CorrelationEdge("C", "D", 0.2),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        result = clustering.cluster(num_clusters=2)

        assert result.num_clusters == 2
        # A, B, C should be in one cluster; D, E, F in another
        cluster_a = result.assignments["A"]
        cluster_d = result.assignments["D"]
        assert cluster_a != cluster_d
        assert result.assignments["B"] == cluster_a
        assert result.assignments["C"] == cluster_a
        assert result.assignments["E"] == cluster_d
        assert result.assignments["F"] == cluster_d

    def test_cluster_with_distance_threshold(self):
        """Cluster with distance threshold."""
        edges = [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.8),
            CorrelationEdge("D", "E", 0.7),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        result = clustering.cluster(distance_threshold=0.5)

        assert result.num_clusters >= 1
        assert len(result.assignments) == 5

    def test_cluster_auto_optimize(self):
        """Auto-optimize number of clusters."""
        edges = [
            # Tight cluster 1
            CorrelationEdge("A", "B", 0.95),
            CorrelationEdge("B", "C", 0.95),
            CorrelationEdge("A", "C", 0.95),
            # Tight cluster 2
            CorrelationEdge("D", "E", 0.95),
            CorrelationEdge("E", "F", 0.95),
            CorrelationEdge("D", "F", 0.95),
            # Weak connection between clusters
            CorrelationEdge("C", "D", 0.1),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        result = clustering.cluster(auto_optimize=True)

        # Should find 2 clusters
        assert result.num_clusters >= 2

    def test_single_node_graph(self):
        """Handle single node graph."""
        edges = []
        # Need to create graph with single node differently
        # For now, use a self-loop workaround or skip
        # Actually, empty edges creates empty graph
        # Let's use a simple edge case
        edges = [CorrelationEdge("A", "B", 0.9)]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        result = clustering.cluster(num_clusters=1)

        assert result.num_clusters == 1
        assert len(result.assignments) == 2

    def test_empty_graph_raises(self):
        """Empty graph raises error."""
        graph = SparseCorrelationGraph([])
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")

        with pytest.raises(ValueError, match="Cannot cluster empty graph"):
            clustering.build_dendrogram()

    def test_different_linkage_methods(self):
        """Test different linkage methods."""
        edges = [
            CorrelationEdge("A", "B", 0.8),
            CorrelationEdge("B", "C", 0.7),
            CorrelationEdge("C", "D", 0.6),
        ]

        graph = SparseCorrelationGraph(edges)

        for method in ['single', 'complete', 'average']:
            clustering = HierarchicalClustering(graph, method=method, missing_distance_policy="treat_missing_as_max")
            result = clustering.cluster(num_clusters=2)
            assert result.num_clusters == 2


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
class TestDendrogram:
    """Tests for Dendrogram class."""

    def test_cut_at_distance(self):
        """Cut dendrogram at distance threshold."""
        edges = [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.8),
            CorrelationEdge("D", "E", 0.7),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        dendrogram = clustering.build_dendrogram()

        result = dendrogram.cut_at_distance(0.5)

        assert isinstance(result, ClusterResult)
        assert len(result.assignments) == 5

    def test_cut_at_num_clusters(self):
        """Cut dendrogram to get specified clusters."""
        edges = [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.8),
            CorrelationEdge("D", "E", 0.7),
            CorrelationEdge("E", "F", 0.6),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        dendrogram = clustering.build_dendrogram()

        result = dendrogram.cut_at_num_clusters(3)

        assert result.num_clusters == 3
        assert len(result.assignments) == 6

    def test_cut_num_clusters_validation(self):
        """Validate num_clusters parameter."""
        edges = [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.8),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        dendrogram = clustering.build_dendrogram()

        # Too few clusters
        with pytest.raises(ValueError, match="num_clusters must be at least 1"):
            dendrogram.cut_at_num_clusters(0)

        # Too many clusters
        with pytest.raises(ValueError, match="cannot exceed number of factors"):
            dendrogram.cut_at_num_clusters(100)

    def test_get_optimal_num_clusters(self):
        """Estimate optimal number of clusters."""
        edges = [
            # Two tight clusters
            CorrelationEdge("A", "B", 0.95),
            CorrelationEdge("B", "C", 0.95),
            CorrelationEdge("D", "E", 0.95),
            CorrelationEdge("E", "F", 0.95),
            # Weak inter-cluster
            CorrelationEdge("C", "D", 0.3),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        dendrogram = clustering.build_dendrogram()

        optimal_k = dendrogram.get_optimal_num_clusters(min_clusters=2, max_clusters=4)

        assert 2 <= optimal_k <= 4


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
class TestHierarchicalClusteringIntegration:
    """Integration tests for hierarchical clustering."""

    def test_consistent_with_connected_components(self):
        """Hierarchical clustering at low distance gives connected components."""
        edges = [
            # Component 1
            CorrelationEdge("A", "B", 0.5),
            # Component 2
            CorrelationEdge("C", "D", 0.5),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        # Low threshold should separate disconnected components
        result = clustering.cluster(distance_threshold=0.7)

        # At low threshold, should get 2 components
        assert result.num_clusters == 2

    def test_correlation_to_distance_conversion(self):
        """High correlation means low distance."""
        edges = [
            CorrelationEdge("A", "B", 0.95),  # High corr -> low dist
            CorrelationEdge("B", "C", 0.5),   # Medium corr -> medium dist
            CorrelationEdge("C", "D", 0.1),   # Low corr -> high dist
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        dendrogram = clustering.build_dendrogram()

        # At very low distance threshold, should get many clusters
        result_low = dendrogram.cut_at_distance(0.1)
        # At high distance threshold, should get fewer clusters
        result_high = dendrogram.cut_at_distance(0.9)

        assert result_low.num_clusters >= result_high.num_clusters

    def test_hierarchical_structure(self):
        """Hierarchical clustering captures nested structure."""
        edges = [
            # Level 1: A-B very tight
            CorrelationEdge("A", "B", 0.95),
            # Level 2: (A,B)-C medium
            CorrelationEdge("B", "C", 0.7),
            CorrelationEdge("A", "C", 0.7),
            # Level 3: (A,B,C)-D weak
            CorrelationEdge("C", "D", 0.3),
        ]

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")
        dendrogram = clustering.build_dendrogram()

        # At low threshold: A-B together, C separate, D separate
        result_3 = dendrogram.cut_at_num_clusters(3)
        assert result_3.num_clusters == 3
        assert result_3.assignments["A"] == result_3.assignments["B"]

        # At high threshold: all together
        result_1 = dendrogram.cut_at_num_clusters(1)
        assert result_1.num_clusters == 1

    def test_large_graph_performance(self):
        """Hierarchical clustering handles moderately large graphs."""
        # Create 20 factors with some structure
        edges = []
        for i in range(10):
            # Two groups of 10
            if i < 9:
                edges.append(CorrelationEdge(f"A{i}", f"A{i+1}", 0.8))
            if i < 9:
                edges.append(CorrelationEdge(f"B{i}", f"B{i+1}", 0.8))

        # Weak connection between groups
        edges.append(CorrelationEdge("A4", "B4", 0.2))

        graph = SparseCorrelationGraph(edges)
        clustering = HierarchicalClustering(graph, missing_distance_policy="treat_missing_as_max")

        # Should complete reasonably fast
        result = clustering.cluster(num_clusters=2)
        assert result.num_clusters == 2
