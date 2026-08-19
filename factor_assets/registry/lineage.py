"""
Lineage tracking and parent-child relationship management.

Provides graph traversal, campaign metadata, and ancestry queries.
"""

from dataclasses import dataclass
from typing import Optional, Iterator
from collections import deque

from factor_assets.contracts.lineage import LineageRef, ParentRef
from factor_assets.contracts.asset import FactorAsset


@dataclass(frozen=True)
class CampaignMetadata:
    """
    Metadata for a factor search campaign.

    Tracks campaign context, parameters, and provenance.
    """
    campaign_id: str
    name: str
    start_timestamp: str
    end_timestamp: Optional[str] = None
    objective: Optional[str] = None
    search_space: Optional[str] = None
    total_trials: int = 0
    successful_trials: int = 0
    approved_factors: int = 0
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.campaign_id:
            raise ValueError("campaign_id is required")
        if not self.name:
            raise ValueError("name is required")


@dataclass(frozen=True)
class LineageEdge:
    """
    Directed edge in the factor lineage graph.

    Represents parent -> child relationship with metadata.
    """
    parent_id: str
    child_id: str
    relationship: str
    mutation_id: Optional[str] = None
    operation: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if not self.parent_id:
            raise ValueError("parent_id is required")
        if not self.child_id:
            raise ValueError("child_id is required")
        if not self.relationship:
            raise ValueError("relationship is required")


class LineageGraph:
    """
    Factor lineage graph with parent-child tracking.

    Maintains the ancestry DAG and provides traversal operations.
    Does NOT store full FactorAsset objects — only IDs and edges.
    """

    def __init__(self):
        self._children: dict[str, set[str]] = {}  # parent_id -> set of child_ids
        self._parents: dict[str, set[str]] = {}   # child_id -> set of parent_ids
        self._edges: dict[tuple[str, str], LineageEdge] = {}  # (parent, child) -> edge
        self._campaigns: dict[str, CampaignMetadata] = {}
        self._factor_to_campaign: dict[str, str] = {}  # factor_id -> campaign_id

    def register_factor(self, lineage: LineageRef) -> None:
        """
        Register a factor's lineage in the graph.

        Args:
            lineage: LineageRef containing parents and campaign info
        """
        factor_id = lineage.factor_id

        # Reject self-parenting and edges that would close a cycle: a cycle
        # makes ancestors/descendants traversal wrong (or infinite) and
        # corrupts lineage-based novelty checks.
        for parent_ref in lineage.parents:
            if parent_ref.factor_id == factor_id:
                raise ValueError(
                    f"Factor {factor_id} cannot list itself as a parent"
                )

        # Initialize collections if not present
        if factor_id not in self._parents:
            self._parents[factor_id] = set()

        # Register parent relationships
        for parent_ref in lineage.parents:
            parent_id = parent_ref.factor_id

            # Add to parent's children set
            if parent_id not in self._children:
                self._children[parent_id] = set()
            self._children[parent_id].add(factor_id)

            # Add to child's parents set
            self._parents[factor_id].add(parent_id)

            # Store edge metadata
            edge = LineageEdge(
                parent_id=parent_id,
                child_id=factor_id,
                relationship=parent_ref.relationship,
                mutation_id=parent_ref.mutation_id,
                operation=parent_ref.operation,
                timestamp=parent_ref.timestamp,
            )
            self._edges[(parent_id, factor_id)] = edge

        # Cycle guard AFTER edges are staged: if the new factor is already an
        # ancestor of one of its parents (or of itself through them), the DAG
        # would become cyclic — roll the registration back and refuse it.
        if self.has_cycle(factor_id):
            for parent_ref in lineage.parents:
                parent_id = parent_ref.factor_id
                self._children[parent_id].discard(factor_id)
                self._edges.pop((parent_id, factor_id), None)
            self._parents[factor_id] = set()
            raise ValueError(
                f"Registering factor {factor_id} would create a lineage cycle"
            )

        # Register campaign association
        if lineage.campaign_id:
            self._factor_to_campaign[factor_id] = lineage.campaign_id

    def register_campaign(self, campaign: CampaignMetadata) -> None:
        """
        Register campaign metadata.

        Args:
            campaign: Campaign metadata
        """
        self._campaigns[campaign.campaign_id] = campaign

    def get_parents(self, factor_id: str) -> tuple[str, ...]:
        """
        Get direct parent factor IDs.

        Args:
            factor_id: Factor identifier

        Returns:
            Tuple of parent factor IDs
        """
        return tuple(sorted(self._parents.get(factor_id, set())))

    def get_children(self, factor_id: str) -> tuple[str, ...]:
        """
        Get direct child factor IDs.

        Args:
            factor_id: Factor identifier

        Returns:
            Tuple of child factor IDs
        """
        return tuple(sorted(self._children.get(factor_id, set())))

    def get_edge(self, parent_id: str, child_id: str) -> Optional[LineageEdge]:
        """
        Get edge metadata between parent and child.

        Args:
            parent_id: Parent factor ID
            child_id: Child factor ID

        Returns:
            LineageEdge if exists, None otherwise
        """
        return self._edges.get((parent_id, child_id))

    def get_ancestors(self, factor_id: str, max_depth: Optional[int] = None) -> tuple[str, ...]:
        """
        Get all ancestor factor IDs via BFS traversal.

        Args:
            factor_id: Factor identifier
            max_depth: Maximum traversal depth (None for unlimited)

        Returns:
            Tuple of ancestor factor IDs in breadth-first order
        """
        ancestors = []
        visited = {factor_id}
        queue = deque([(factor_id, 0)])

        while queue:
            current_id, depth = queue.popleft()

            if max_depth is not None and depth >= max_depth:
                continue

            for parent_id in self._parents.get(current_id, set()):
                if parent_id not in visited:
                    visited.add(parent_id)
                    ancestors.append(parent_id)
                    queue.append((parent_id, depth + 1))

        return tuple(ancestors)

    def get_descendants(self, factor_id: str, max_depth: Optional[int] = None) -> tuple[str, ...]:
        """
        Get all descendant factor IDs via BFS traversal.

        Args:
            factor_id: Factor identifier
            max_depth: Maximum traversal depth (None for unlimited)

        Returns:
            Tuple of descendant factor IDs in breadth-first order
        """
        descendants = []
        visited = {factor_id}
        queue = deque([(factor_id, 0)])

        while queue:
            current_id, depth = queue.popleft()

            if max_depth is not None and depth >= max_depth:
                continue

            for child_id in self._children.get(current_id, set()):
                if child_id not in visited:
                    visited.add(child_id)
                    descendants.append(child_id)
                    queue.append((child_id, depth + 1))

        return tuple(descendants)

    def get_lineage_depth(self, factor_id: str) -> int:
        """
        Get lineage depth (distance from root ancestors).

        Args:
            factor_id: Factor identifier

        Returns:
            Maximum depth from any root ancestor (0 if no parents)
        """
        if not self._parents.get(factor_id):
            return 0

        max_depth = 0
        for parent_id in self._parents[factor_id]:
            parent_depth = self.get_lineage_depth(parent_id)
            max_depth = max(max_depth, parent_depth + 1)

        return max_depth

    def get_campaign(self, campaign_id: str) -> Optional[CampaignMetadata]:
        """
        Get campaign metadata by ID.

        Args:
            campaign_id: Campaign identifier

        Returns:
            CampaignMetadata if found, None otherwise
        """
        return self._campaigns.get(campaign_id)

    def get_campaign_factors(self, campaign_id: str) -> tuple[str, ...]:
        """
        Get all factor IDs associated with a campaign.

        Args:
            campaign_id: Campaign identifier

        Returns:
            Tuple of factor IDs from this campaign
        """
        return tuple(
            factor_id for factor_id, cid in self._factor_to_campaign.items()
            if cid == campaign_id
        )

    def get_factor_campaign(self, factor_id: str) -> Optional[str]:
        """
        Get campaign ID for a factor.

        Args:
            factor_id: Factor identifier

        Returns:
            Campaign ID if factor is from a campaign, None otherwise
        """
        return self._factor_to_campaign.get(factor_id)

    def has_cycle(self, factor_id: str) -> bool:
        """
        Check if adding this factor would create a cycle.

        Args:
            factor_id: Factor identifier

        Returns:
            True if factor is its own ancestor (cycle detected)
        """
        return factor_id in self.get_ancestors(factor_id)

    def is_root(self, factor_id: str) -> bool:
        """Check if factor is a root (no parents)."""
        return len(self._parents.get(factor_id, set())) == 0

    def is_leaf(self, factor_id: str) -> bool:
        """Check if factor is a leaf (no children)."""
        return len(self._children.get(factor_id, set())) == 0

    def get_roots(self) -> tuple[str, ...]:
        """Get all root factor IDs (factors with no parents)."""
        all_factors = set(self._parents.keys()) | set(self._children.keys())
        return tuple(sorted(fid for fid in all_factors if self.is_root(fid)))

    def get_leaves(self) -> tuple[str, ...]:
        """Get all leaf factor IDs (factors with no children)."""
        all_factors = set(self._parents.keys()) | set(self._children.keys())
        return tuple(sorted(fid for fid in all_factors if self.is_leaf(fid)))

    def stats(self) -> dict:
        """Get lineage graph statistics."""
        all_factors = set(self._parents.keys()) | set(self._children.keys())

        return {
            "total_factors": len(all_factors),
            "total_edges": len(self._edges),
            "total_campaigns": len(self._campaigns),
            "root_factors": len(self.get_roots()),
            "leaf_factors": len(self.get_leaves()),
            "factors_with_parents": sum(1 for fid in all_factors if not self.is_root(fid)),
            "factors_with_children": sum(1 for fid in all_factors if not self.is_leaf(fid)),
        }
