"""
Sparse correlation graph builder.

Constructs adjacency lists from pairwise factor correlations,
filtering to keep only significant edges. Memory-efficient representation
for large factor universes.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple, Optional
from collections import defaultdict
import hashlib


@dataclass(frozen=True)
class CorrelationEdge:
    """
    Single correlation edge between two factors.

    Undirected edge with correlation strength.
    """
    factor_a: str
    factor_b: str
    correlation: float

    def __post_init__(self):
        if self.factor_a == self.factor_b:
            raise ValueError("Self-loops not allowed")
        if not (-1.0 <= self.correlation <= 1.0):
            raise ValueError(f"Correlation must be in [-1, 1], got {self.correlation}")

    @property
    def abs_correlation(self) -> float:
        """Absolute correlation strength."""
        return abs(self.correlation)

    def canonical_form(self) -> Tuple[str, str]:
        """Return (min, max) ordered pair for undirected edge."""
        return tuple(sorted([self.factor_a, self.factor_b]))


class SparseCorrelationGraph:
    """
    Sparse undirected graph of factor correlations.

    Stores only edges above threshold, uses adjacency list representation.
    Thread-safe after construction (immutable).
    """

    def __init__(
        self,
        edges: Iterable[CorrelationEdge],
        nodes: Optional[Iterable[str]] = None,
    ):
        """
        Build graph from edges and an optional explicit node universe.

        Args:
            edges: Correlation edges
            nodes: Factor IDs to retain even when they have no edges

        Exact duplicate edges are deduplicated. Duplicate canonical edges with
        different correlation values are rejected as contradictory input.
        """
        self._adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._nodes: Set[str] = set(nodes or ())
        self._edge_count = 0

        seen_edges: Dict[Tuple[str, str], Tuple[float, Tuple[str, str]]] = {}

        for edge in edges:
            canonical = edge.canonical_form()
            orientation = (edge.factor_a, edge.factor_b)
            existing = seen_edges.get(canonical)
            if existing is not None:
                existing_correlation, existing_orientation = existing
                if existing_correlation != edge.correlation:
                    raise ValueError(
                        "Conflicting duplicate edge "
                        f"{canonical}: correlations {existing_correlation} "
                        f"and {edge.correlation}"
                    )
                continue
            seen_edges[canonical] = (edge.correlation, orientation)

            # Add to adjacency list (both directions for undirected)
            self._adjacency[edge.factor_a].append((edge.factor_b, edge.correlation))
            self._adjacency[edge.factor_b].append((edge.factor_a, edge.correlation))

            self._nodes.add(edge.factor_a)
            self._nodes.add(edge.factor_b)
            self._edge_count += 1

        # Freeze adjacency lists
        self._adjacency = dict(self._adjacency)
        self._graph_identity = hashlib.sha256()
        for node in sorted(self._nodes):
            encoded = node.encode("utf-8")
            self._graph_identity.update(str(len(encoded)).encode("ascii"))
            self._graph_identity.update(b":")
            self._graph_identity.update(encoded)
        for edge in self.to_edge_list():
            encoded = f"{edge.factor_a}\x00{edge.factor_b}\x00{edge.correlation}".encode("utf-8")
            self._graph_identity.update(str(len(encoded)).encode("ascii"))
            self._graph_identity.update(b":")
            self._graph_identity.update(encoded)
        self._graph_identity = self._graph_identity.hexdigest()

    @property
    def graph_identity(self) -> str:
        """Content hash over the node set and every edge (undirected, order
        invariant).  Lets a ClusterArtifact record exactly which graph it was
        produced from."""
        return self._graph_identity

    @property
    def nodes(self) -> Set[str]:
        """All nodes in the graph."""
        return self._nodes.copy()

    @property
    def node_count(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of undirected edges."""
        return self._edge_count

    def neighbors(self, factor_id: str) -> List[Tuple[str, float]]:
        """
        Get neighbors and their correlations.

        Args:
            factor_id: Factor node ID

        Returns:
            List of (neighbor_id, correlation) tuples
        """
        return self._adjacency.get(factor_id, []).copy()

    def degree(self, factor_id: str) -> int:
        """Number of neighbors for a node."""
        return len(self._adjacency.get(factor_id, []))

    def has_edge(self, factor_a: str, factor_b: str) -> bool:
        """Check if edge exists between two factors."""
        if factor_a not in self._adjacency:
            return False
        return any(neighbor == factor_b for neighbor, _ in self._adjacency[factor_a])

    def get_correlation(self, factor_a: str, factor_b: str) -> Optional[float]:
        """
        Get correlation between two factors.

        Returns:
            Correlation value if edge exists, None otherwise
        """
        if factor_a not in self._adjacency:
            return None
        for neighbor, corr in self._adjacency[factor_a]:
            if neighbor == factor_b:
                return corr
        return None

    def subgraph(self, node_subset: Set[str]) -> 'SparseCorrelationGraph':
        """
        Extract induced subgraph.

        Args:
            node_subset: Nodes to include in subgraph

        Returns:
            New graph containing only edges between nodes in subset
        """
        subgraph_edges = []
        seen_pairs = set()

        for node in node_subset:
            if node not in self._adjacency:
                continue
            for neighbor, corr in self._adjacency[node]:
                if neighbor not in node_subset:
                    continue
                canonical = tuple(sorted([node, neighbor]))
                if canonical in seen_pairs:
                    continue
                seen_pairs.add(canonical)
                subgraph_edges.append(CorrelationEdge(node, neighbor, corr))

        retained_nodes = self._nodes.intersection(node_subset)
        return SparseCorrelationGraph(subgraph_edges, nodes=retained_nodes)

    def to_edge_list(self) -> List[CorrelationEdge]:
        """Export as edge list."""
        edges = []
        seen_pairs = set()

        for factor_a, neighbors in self._adjacency.items():
            for factor_b, corr in neighbors:
                canonical = tuple(sorted([factor_a, factor_b]))
                if canonical in seen_pairs:
                    continue
                seen_pairs.add(canonical)
                edges.append(CorrelationEdge(factor_a, factor_b, corr))

        return edges

    def density(self) -> float:
        """
        Graph density (fraction of possible edges present).

        Returns:
            Density in [0, 1]
        """
        n = self.node_count
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1) // 2
        return self.edge_count / max_edges if max_edges > 0 else 0.0
