"""
Parent/child lineage detection for factor families.

Identifies hierarchical relationships within factor families
based on correlation patterns and structural similarity.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from factor_assets.graph.sparse import SparseCorrelationGraph


@dataclass(frozen=True)
class ParentChildRelation:
    """
    Parent-child relationship between two factors.

    Parent is typically simpler/more fundamental,
    child is derived or more complex variant.
    """
    parent_id: str
    child_id: str
    correlation: float
    confidence: float  # [0, 1] confidence in relationship

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class FamilyLineage:
    """
    Complete lineage structure for a factor family.

    Contains parent-child relations and root/leaf identification.
    """
    family_id: int
    members: Set[str]
    relations: List[ParentChildRelation]
    roots: Set[str]  # Factors with no parents
    leaves: Set[str]  # Factors with no children

    @property
    def size(self) -> int:
        """Number of factors in family."""
        return len(self.members)

    @property
    def depth(self) -> int:
        """Maximum depth of lineage tree."""
        if not self.relations:
            return 0

        # Build parent->children map
        children_map = defaultdict(set)
        for rel in self.relations:
            children_map[rel.parent_id].add(rel.child_id)

        # DFS from each root to find max depth
        max_depth = 0
        for root in self.roots:
            depth = self._calculate_depth(root, children_map, set())
            max_depth = max(max_depth, depth)

        return max_depth

    def _calculate_depth(
        self,
        node: str,
        children_map: Dict[str, Set[str]],
        visited: Set[str]
    ) -> int:
        """Calculate depth from node using DFS."""
        if node in visited:
            return 0  # Cycle detection
        visited.add(node)

        if node not in children_map:
            return 0

        max_child_depth = 0
        for child in children_map[node]:
            child_depth = self._calculate_depth(child, children_map, visited.copy())
            max_child_depth = max(max_child_depth, child_depth)

        return 1 + max_child_depth

    def get_children(self, parent_id: str) -> Set[str]:
        """Get direct children of a parent."""
        return {rel.child_id for rel in self.relations if rel.parent_id == parent_id}

    def get_parents(self, child_id: str) -> Set[str]:
        """Get direct parents of a child."""
        return {rel.parent_id for rel in self.relations if rel.child_id == child_id}


class LineageDetector:
    """
    Detect parent-child relationships within factor families.

    Uses correlation strength, node degree, and neighborhood overlap
    to infer hierarchical structure. Parents typically have:
    - Higher degree (more connections)
    - Stronger average correlations
    - Broader neighborhood overlap with children
    """

    def __init__(
        self,
        graph: SparseCorrelationGraph,
        min_correlation: float = 0.7,
        degree_threshold: int = 2
    ):
        """
        Args:
            graph: Correlation graph
            min_correlation: Minimum correlation for parent-child relation
            degree_threshold: Minimum degree difference to consider parent
        """
        self.graph = graph
        self.min_correlation = min_correlation
        self.degree_threshold = degree_threshold

    def detect_lineage(self, family_members: Set[str], family_id: int) -> FamilyLineage:
        """
        Detect lineage structure within a family.

        Args:
            family_members: Set of factors in the family
            family_id: Family cluster ID

        Returns:
            Family lineage with parent-child relations
        """
        # Extract subgraph for this family
        subgraph = self.graph.subgraph(family_members)

        # Identify parent-child relations
        relations = self._find_relations(subgraph, family_members)

        # Identify roots (no parents) and leaves (no children)
        children = {rel.child_id for rel in relations}
        parents = {rel.parent_id for rel in relations}

        roots = family_members - children
        leaves = family_members - parents

        # Handle single-node families
        if not relations and family_members:
            roots = family_members
            leaves = family_members

        return FamilyLineage(
            family_id=family_id,
            members=family_members,
            relations=relations,
            roots=roots,
            leaves=leaves
        )

    def _find_relations(
        self,
        subgraph: SparseCorrelationGraph,
        members: Set[str]
    ) -> List[ParentChildRelation]:
        """Find all parent-child relations in subgraph."""
        relations = []
        seen_pairs = set()  # Track pairs to avoid duplicates

        # Consider all pairs of connected nodes
        for factor_a in members:
            for factor_b, corr in subgraph.neighbors(factor_a):
                if factor_b not in members:
                    continue

                # Skip if we've already processed this pair
                canonical_pair = tuple(sorted([factor_a, factor_b]))
                if canonical_pair in seen_pairs:
                    continue
                seen_pairs.add(canonical_pair)

                # Check if correlation is strong enough
                if abs(corr) < self.min_correlation:
                    continue

                # Determine parent/child by degree
                degree_a = subgraph.degree(factor_a)
                degree_b = subgraph.degree(factor_b)

                parent, child = None, None
                if degree_a > degree_b + self.degree_threshold:
                    parent, child = factor_a, factor_b
                elif degree_b > degree_a + self.degree_threshold:
                    parent, child = factor_b, factor_a
                else:
                    # Similar degree, use lexicographic order for stability
                    if factor_a < factor_b:
                        parent, child = factor_a, factor_b
                    else:
                        parent, child = factor_b, factor_a

                # Calculate confidence based on neighborhood overlap
                confidence = self._calculate_confidence(
                    parent, child, subgraph
                )

                # Only add if confidence is reasonable
                if confidence >= 0.3:
                    relations.append(ParentChildRelation(
                        parent_id=parent,
                        child_id=child,
                        correlation=corr,
                        confidence=confidence
                    ))

        return relations

    def _calculate_confidence(
        self,
        parent: str,
        child: str,
        subgraph: SparseCorrelationGraph
    ) -> float:
        """
        Calculate confidence in parent-child relationship.

        Based on:
        - Correlation strength
        - Neighborhood overlap (child's neighbors subset of parent's)
        - Degree ratio
        """
        corr = subgraph.get_correlation(parent, child)
        if corr is None:
            return 0.0

        # Correlation component (higher = more confident)
        corr_score = abs(corr)

        # Neighborhood overlap
        parent_neighbors = {n for n, _ in subgraph.neighbors(parent)}
        child_neighbors = {n for n, _ in subgraph.neighbors(child)}

        if not child_neighbors:
            overlap_score = 1.0
        else:
            overlap = len(child_neighbors & parent_neighbors)
            overlap_score = overlap / len(child_neighbors)

        # Degree ratio (parent should have higher degree)
        parent_degree = subgraph.degree(parent)
        child_degree = subgraph.degree(child)

        if parent_degree == 0:
            degree_score = 0.0
        else:
            degree_ratio = child_degree / parent_degree
            degree_score = 1.0 - degree_ratio if degree_ratio <= 1.0 else 0.0

        # Weighted combination
        confidence = (
            0.4 * corr_score +
            0.4 * overlap_score +
            0.2 * degree_score
        )

        return min(confidence, 1.0)

    def detect_all_lineages(
        self,
        cluster_result: 'ClusterResult'
    ) -> List[FamilyLineage]:
        """
        Detect lineages for all families in clustering result.

        Args:
            cluster_result: Result from clustering algorithm

        Returns:
            List of family lineages
        """
        lineages = []
        clusters = cluster_result.get_all_clusters()

        for cluster_id, members in clusters.items():
            lineage = self.detect_lineage(members, cluster_id)
            lineages.append(lineage)

        return lineages
