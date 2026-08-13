"""
Factor family clustering algorithms.

Connected components for basic grouping, modularity-based clustering
for community detection in correlation graphs, and hierarchical clustering
with dendrogram cutting.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

try:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    hierarchy = None
    squareform = None
    np = None

from factor_assets.graph.sparse import SparseCorrelationGraph


@dataclass(frozen=True)
class ClusterResult:
    """
    Result of a clustering operation.

    Maps each factor to its cluster ID and provides cluster membership.
    """
    assignments: Dict[str, int]
    cluster_sizes: Dict[int, int]

    @property
    def num_clusters(self) -> int:
        """Number of clusters found."""
        return len(self.cluster_sizes)

    def get_cluster_members(self, cluster_id: int) -> Set[str]:
        """Get all factors in a cluster."""
        return {factor for factor, cid in self.assignments.items() if cid == cluster_id}

    def get_all_clusters(self) -> Dict[int, Set[str]]:
        """Get all clusters as dict of cluster_id -> members."""
        clusters = defaultdict(set)
        for factor, cluster_id in self.assignments.items():
            clusters[cluster_id].add(factor)
        return dict(clusters)


class ConnectedComponents:
    """
    Find connected components in correlation graph.

    Uses DFS to identify disconnected subgraphs. Factors in the same
    component have at least one correlation path connecting them.
    """

    def __init__(self, graph: SparseCorrelationGraph):
        """
        Args:
            graph: Correlation graph to cluster
        """
        self.graph = graph

    def find_components(self) -> ClusterResult:
        """
        Find all connected components.

        Returns:
            Cluster assignment where each component gets unique ID
        """
        visited = set()
        assignments = {}
        cluster_id = 0

        for node in self.graph.nodes:
            if node in visited:
                continue

            # DFS to find component
            component = self._dfs(node, visited)

            # Assign cluster ID to all nodes in component
            for member in component:
                assignments[member] = cluster_id

            cluster_id += 1

        # Calculate cluster sizes
        cluster_sizes = defaultdict(int)
        for cid in assignments.values():
            cluster_sizes[cid] += 1

        return ClusterResult(
            assignments=assignments,
            cluster_sizes=dict(cluster_sizes)
        )

    def _dfs(self, start: str, visited: Set[str]) -> Set[str]:
        """DFS to find all reachable nodes from start."""
        component = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node in visited:
                continue

            visited.add(node)
            component.add(node)

            # Add unvisited neighbors
            for neighbor, _ in self.graph.neighbors(node):
                if neighbor not in visited:
                    stack.append(neighbor)

        return component


class ModularityClustering:
    """
    Greedy modularity optimization for community detection.

    Uses Louvain-style algorithm to maximize modularity score.
    Identifies dense subgroups (families) within correlation graph.
    """

    def __init__(self, graph: SparseCorrelationGraph, resolution: float = 1.0):
        """
        Args:
            graph: Correlation graph to cluster
            resolution: Resolution parameter (higher = more/smaller clusters)
        """
        self.graph = graph
        self.resolution = resolution

    def cluster(self, max_iterations: int = 100) -> ClusterResult:
        """
        Run modularity clustering.

        Args:
            max_iterations: Maximum optimization iterations

        Returns:
            Cluster assignments
        """
        # Initialize: each node in its own cluster
        assignments = {node: i for i, node in enumerate(self.graph.nodes)}
        cluster_id_counter = len(self.graph.nodes)

        total_weight = self._total_edge_weight()

        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try moving each node to neighbor's cluster
            for node in self.graph.nodes:
                current_cluster = assignments[node]

                # Find best cluster to move to
                best_cluster = current_cluster
                best_gain = 0.0

                # Consider neighbors' clusters
                neighbor_clusters = set()
                for neighbor, _ in self.graph.neighbors(node):
                    neighbor_clusters.add(assignments[neighbor])

                for target_cluster in neighbor_clusters:
                    if target_cluster == current_cluster:
                        continue

                    gain = self._modularity_gain(
                        node, current_cluster, target_cluster,
                        assignments, total_weight
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_cluster = target_cluster

                # Move if beneficial
                if best_cluster != current_cluster:
                    assignments[node] = best_cluster
                    improved = True

        # Renumber clusters to be contiguous
        unique_clusters = sorted(set(assignments.values()))
        cluster_map = {old: new for new, old in enumerate(unique_clusters)}
        assignments = {node: cluster_map[cid] for node, cid in assignments.items()}

        # Calculate cluster sizes
        cluster_sizes = defaultdict(int)
        for cid in assignments.values():
            cluster_sizes[cid] += 1

        return ClusterResult(
            assignments=assignments,
            cluster_sizes=dict(cluster_sizes)
        )

    def _total_edge_weight(self) -> float:
        """Sum of absolute correlations (edge weights)."""
        total = 0.0
        for edge in self.graph.to_edge_list():
            total += edge.abs_correlation
        return total

    def _modularity_gain(
        self,
        node: str,
        from_cluster: int,
        to_cluster: int,
        assignments: Dict[str, int],
        total_weight: float
    ) -> float:
        """
        Calculate modularity gain from moving node between clusters.

        Simplified calculation using edge weights.
        """
        if total_weight == 0:
            return 0.0

        # Weight of edges from node to target cluster
        weight_to_target = 0.0
        # Weight of edges from node to current cluster
        weight_to_current = 0.0

        for neighbor, corr in self.graph.neighbors(node):
            abs_corr = abs(corr)
            if assignments[neighbor] == to_cluster:
                weight_to_target += abs_corr
            elif assignments[neighbor] == from_cluster:
                weight_to_current += abs_corr

        # Modularity gain approximation
        gain = (weight_to_target - weight_to_current) / total_weight
        return gain * self.resolution


@dataclass(frozen=True)
class Dendrogram:
    """
    Hierarchical clustering dendrogram.

    Stores linkage matrix and factor order for dendrogram visualization.
    """
    linkage_matrix: "np.ndarray"
    factor_ids: List[str]
    method: str

    def cut_at_distance(self, threshold: float) -> ClusterResult:
        """
        Cut dendrogram at specified distance threshold.

        Args:
            threshold: Distance threshold for cutting

        Returns:
            ClusterResult with cluster assignments
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for dendrogram cutting")

        # Cut dendrogram
        labels = hierarchy.fcluster(
            self.linkage_matrix,
            threshold,
            criterion='distance'
        )

        # Convert to ClusterResult
        assignments = {fid: int(label) - 1 for fid, label in zip(self.factor_ids, labels)}

        cluster_sizes = defaultdict(int)
        for cid in assignments.values():
            cluster_sizes[cid] += 1

        return ClusterResult(
            assignments=assignments,
            cluster_sizes=dict(cluster_sizes)
        )

    def cut_at_num_clusters(self, num_clusters: int) -> ClusterResult:
        """
        Cut dendrogram to obtain specified number of clusters.

        Args:
            num_clusters: Desired number of clusters

        Returns:
            ClusterResult with cluster assignments
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for dendrogram cutting")

        if num_clusters < 1:
            raise ValueError("num_clusters must be at least 1")
        if num_clusters > len(self.factor_ids):
            raise ValueError(f"num_clusters cannot exceed number of factors ({len(self.factor_ids)})")

        # Cut dendrogram
        labels = hierarchy.fcluster(
            self.linkage_matrix,
            num_clusters,
            criterion='maxclust'
        )

        # Convert to ClusterResult
        assignments = {fid: int(label) - 1 for fid, label in zip(self.factor_ids, labels)}

        cluster_sizes = defaultdict(int)
        for cid in assignments.values():
            cluster_sizes[cid] += 1

        return ClusterResult(
            assignments=assignments,
            cluster_sizes=dict(cluster_sizes)
        )

    def get_optimal_num_clusters(
        self,
        min_clusters: int = 2,
        max_clusters: Optional[int] = None
    ) -> int:
        """
        Estimate optimal number of clusters using distance gaps.

        Uses second derivative of linkage distances to find elbow.

        Args:
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters (default: n/2)

        Returns:
            Estimated optimal number of clusters
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for optimal cluster estimation")

        n = len(self.factor_ids)
        if max_clusters is None:
            max_clusters = max(min_clusters, n // 2)

        if n < 2:
            return 1

        # Get merge distances from linkage matrix
        merge_distances = self.linkage_matrix[:, 2]

        # Look for largest gaps in distances
        if len(merge_distances) < 2:
            return min_clusters

        # Calculate distance differences (gaps)
        gaps = np.diff(merge_distances)

        # Find largest gap that results in valid cluster count
        valid_gaps = []
        for i, gap in enumerate(gaps):
            num_clusters = n - i
            if min_clusters <= num_clusters <= max_clusters:
                valid_gaps.append((gap, num_clusters))

        if not valid_gaps:
            return min_clusters

        # Return cluster count with largest gap
        _, optimal_k = max(valid_gaps, key=lambda x: x[0])
        return optimal_k


class HierarchicalClustering:
    """
    Hierarchical clustering for factor families.

    Builds dendrogram from correlation graph using agglomerative clustering.
    Supports multiple linkage methods and provides dendrogram cutting.
    """

    def __init__(
        self,
        graph: SparseCorrelationGraph,
        method: str = 'average',
        metric: str = 'correlation'
    ):
        """
        Args:
            graph: Correlation graph to cluster
            method: Linkage method ('single', 'complete', 'average', 'ward')
            metric: Distance metric ('correlation', 'euclidean')

        Raises:
            ImportError: If scipy is not available
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for HierarchicalClustering. "
                "Install with: pip install scipy"
            )

        self.graph = graph
        self.method = method
        self.metric = metric

    def build_dendrogram(self) -> Dendrogram:
        """
        Build hierarchical clustering dendrogram.

        Returns:
            Dendrogram containing linkage matrix and factor IDs
        """
        if self.graph.node_count == 0:
            raise ValueError("Cannot cluster empty graph")

        if self.graph.node_count == 1:
            # Single node: return trivial linkage
            factor_ids = list(self.graph.nodes)
            linkage_matrix = np.zeros((0, 4))
            return Dendrogram(
                linkage_matrix=linkage_matrix,
                factor_ids=factor_ids,
                method=self.method
            )

        # Build distance matrix from correlation graph
        factor_ids = sorted(self.graph.nodes)
        n = len(factor_ids)
        distance_matrix = np.ones((n, n))

        # Convert correlation to distance: dist = 1 - abs(corr)
        for i, fid_a in enumerate(factor_ids):
            distance_matrix[i, i] = 0.0
            for j, fid_b in enumerate(factor_ids):
                if i < j:
                    corr = self.graph.get_correlation(fid_a, fid_b)
                    if corr is not None:
                        dist = 1.0 - abs(corr)
                    else:
                        dist = 1.0  # No edge = maximum distance
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

        # Convert to condensed form for scipy
        condensed_distances = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering
        linkage_matrix = hierarchy.linkage(
            condensed_distances,
            method=self.method,
            metric=self.metric if self.metric != 'correlation' else 'euclidean'
        )

        return Dendrogram(
            linkage_matrix=linkage_matrix,
            factor_ids=factor_ids,
            method=self.method
        )

    def cluster(
        self,
        num_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        auto_optimize: bool = False
    ) -> ClusterResult:
        """
        Perform hierarchical clustering with automatic or manual cutting.

        Args:
            num_clusters: Desired number of clusters (exclusive with distance_threshold)
            distance_threshold: Distance threshold for cutting (exclusive with num_clusters)
            auto_optimize: Automatically determine optimal number of clusters

        Returns:
            ClusterResult with cluster assignments
        """
        dendrogram = self.build_dendrogram()

        if dendrogram.linkage_matrix.shape[0] == 0:
            # Single node case
            return ClusterResult(
                assignments={dendrogram.factor_ids[0]: 0},
                cluster_sizes={0: 1}
            )

        # Determine cutting strategy
        if auto_optimize:
            optimal_k = dendrogram.get_optimal_num_clusters()
            return dendrogram.cut_at_num_clusters(optimal_k)
        elif num_clusters is not None:
            return dendrogram.cut_at_num_clusters(num_clusters)
        elif distance_threshold is not None:
            return dendrogram.cut_at_distance(distance_threshold)
        else:
            # Default: use optimal estimation
            optimal_k = dendrogram.get_optimal_num_clusters()
            return dendrogram.cut_at_num_clusters(optimal_k)
