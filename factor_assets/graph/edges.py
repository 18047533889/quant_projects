"""
Edge filtering strategies for correlation graphs.

Threshold-based and top-K filtering to reduce graph density
and focus on strongest correlations.
"""

from abc import ABC, abstractmethod
from typing import List
from factor_assets.graph.sparse import CorrelationEdge


class EdgeFilter(ABC):
    """
    Abstract base for edge filtering strategies.

    Filters edges based on correlation strength or other criteria.
    """

    @abstractmethod
    def filter(self, edges: List[CorrelationEdge]) -> List[CorrelationEdge]:
        """
        Filter edge list.

        Args:
            edges: Input edges

        Returns:
            Filtered edges
        """
        pass


class ThresholdFilter(EdgeFilter):
    """
    Filter edges by absolute correlation threshold.

    Keeps edges with |correlation| >= threshold.
    """

    def __init__(self, threshold: float):
        """
        Args:
            threshold: Minimum absolute correlation (in [0, 1])
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def filter(self, edges: List[CorrelationEdge]) -> List[CorrelationEdge]:
        """Keep edges with |correlation| >= threshold."""
        return [edge for edge in edges if edge.abs_correlation >= self.threshold]


class TopKFilter(EdgeFilter):
    """
    Keep top-K strongest edges per node.

    For each node, retains at most K edges with highest absolute correlation.
    Symmetric: if A->B is kept, B->A is also kept.
    """

    def __init__(self, k: int):
        """
        Args:
            k: Maximum edges per node
        """
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self.k = k

    def filter(self, edges: List[CorrelationEdge]) -> List[CorrelationEdge]:
        """Keep top-K edges per node."""
        if self.k == 0:
            return []

        # Build adjacency with all edges
        from collections import defaultdict
        adjacency = defaultdict(list)

        for edge in edges:
            adjacency[edge.factor_a].append((edge.factor_b, edge.abs_correlation, edge))
            adjacency[edge.factor_b].append((edge.factor_a, edge.abs_correlation, edge))

        # For each node, keep top-K by absolute correlation
        kept_edges = set()
        for node, neighbors in adjacency.items():
            # Sort by absolute correlation descending
            neighbors_sorted = sorted(neighbors, key=lambda x: x[1], reverse=True)
            top_k = neighbors_sorted[:self.k]

            # Add canonical form to kept set
            for _, _, edge in top_k:
                canonical = edge.canonical_form()
                kept_edges.add(canonical)

        # Reconstruct edge list
        result = []
        seen = set()
        for edge in edges:
            canonical = edge.canonical_form()
            if canonical in kept_edges and canonical not in seen:
                result.append(edge)
                seen.add(canonical)

        return result


class CompositeFilter(EdgeFilter):
    """
    Apply multiple filters in sequence.

    Edges must pass all filters to be retained.
    """

    def __init__(self, *filters: EdgeFilter):
        """
        Args:
            filters: Sequence of filters to apply
        """
        self.filters = filters

    def filter(self, edges: List[CorrelationEdge]) -> List[CorrelationEdge]:
        """Apply all filters in sequence."""
        result = edges
        for f in self.filters:
            result = f.filter(result)
        return result


class SignFilter(EdgeFilter):
    """
    Filter edges by correlation sign.

    Options: 'positive', 'negative', 'both'.
    """

    def __init__(self, sign: str):
        """
        Args:
            sign: 'positive', 'negative', or 'both'
        """
        if sign not in ('positive', 'negative', 'both'):
            raise ValueError(f"sign must be 'positive', 'negative', or 'both', got {sign}")
        self.sign = sign

    def filter(self, edges: List[CorrelationEdge]) -> List[CorrelationEdge]:
        """Filter by correlation sign."""
        if self.sign == 'both':
            return edges
        elif self.sign == 'positive':
            return [edge for edge in edges if edge.correlation >= 0]
        else:  # negative
            return [edge for edge in edges if edge.correlation < 0]
