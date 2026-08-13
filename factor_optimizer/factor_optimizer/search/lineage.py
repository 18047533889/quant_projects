"""Mutation lineage tracking for provenance and genealogy analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime


@dataclass
class LineageNode:
    """
    A node in the mutation lineage tree.

    Attributes:
        trial_id: Unique identifier for this trial
        parent_ids: Parent trial IDs (empty for seed factors)
        mutation_type: Type of mutation applied
        generation: Generation number (0 for seeds)
        score: Evaluation score (if available)
        created_at: Creation timestamp
        metadata: Additional node metadata
    """
    trial_id: str
    parent_ids: List[str] = field(default_factory=list)
    mutation_type: Optional[str] = None
    generation: int = 0
    score: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_seed(self) -> bool:
        """Check if this is a seed node (no parents)."""
        return len(self.parent_ids) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trial_id": self.trial_id,
            "parent_ids": list(self.parent_ids),
            "mutation_type": self.mutation_type,
            "generation": self.generation,
            "score": self.score,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageNode":
        """Deserialize from dictionary."""
        data = dict(data)
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class LineageTree:
    """
    Tracks mutation lineage as a directed acyclic graph (DAG).

    Each node represents a trial, with edges representing parent-child relationships.
    """

    def __init__(self):
        """Initialize empty lineage tree."""
        self.nodes: Dict[str, LineageNode] = {}
        self.children: Dict[str, List[str]] = {}

    def add_node(self, node: LineageNode) -> None:
        """
        Add a node to the lineage tree.

        Args:
            node: Lineage node to add
        """
        if node.trial_id in self.nodes:
            raise ValueError(f"node {node.trial_id} already exists")

        # Validate parents exist
        for parent_id in node.parent_ids:
            if parent_id not in self.nodes:
                raise ValueError(f"parent {parent_id} not in tree")

        self.nodes[node.trial_id] = node

        # Update children index
        for parent_id in node.parent_ids:
            if parent_id not in self.children:
                self.children[parent_id] = []
            self.children[parent_id].append(node.trial_id)

    def get_node(self, trial_id: str) -> Optional[LineageNode]:
        """Retrieve a node by trial ID."""
        return self.nodes.get(trial_id)

    def get_children(self, trial_id: str) -> List[LineageNode]:
        """Get all direct children of a node."""
        child_ids = self.children.get(trial_id, [])
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_parents(self, trial_id: str) -> List[LineageNode]:
        """Get all direct parents of a node."""
        node = self.get_node(trial_id)
        if node is None:
            return []
        return [self.nodes[pid] for pid in node.parent_ids if pid in self.nodes]

    def get_ancestors(self, trial_id: str) -> Set[str]:
        """
        Get all ancestors of a node (recursive parents).

        Args:
            trial_id: Trial ID to start from

        Returns:
            Set of ancestor trial IDs
        """
        node = self.get_node(trial_id)
        if node is None:
            return set()

        ancestors = set()
        to_visit = list(node.parent_ids)

        while to_visit:
            parent_id = to_visit.pop()
            if parent_id in ancestors:
                continue

            ancestors.add(parent_id)
            parent_node = self.get_node(parent_id)
            if parent_node:
                to_visit.extend(parent_node.parent_ids)

        return ancestors

    def get_descendants(self, trial_id: str) -> Set[str]:
        """
        Get all descendants of a node (recursive children).

        Args:
            trial_id: Trial ID to start from

        Returns:
            Set of descendant trial IDs
        """
        descendants = set()
        to_visit = [trial_id]

        while to_visit:
            current_id = to_visit.pop()
            child_ids = self.children.get(current_id, [])

            for child_id in child_ids:
                if child_id not in descendants:
                    descendants.add(child_id)
                    to_visit.append(child_id)

        return descendants

    def get_seeds(self) -> List[LineageNode]:
        """Get all seed nodes (no parents)."""
        return [node for node in self.nodes.values() if node.is_seed()]

    def get_leaves(self) -> List[LineageNode]:
        """Get all leaf nodes (no children)."""
        return [
            node for node in self.nodes.values()
            if node.trial_id not in self.children or not self.children[node.trial_id]
        ]

    def depth(self, trial_id: str) -> int:
        """
        Get depth of a node (longest path from any seed).

        Args:
            trial_id: Trial ID

        Returns:
            Depth (0 for seeds)
        """
        node = self.get_node(trial_id)
        if node is None:
            return -1

        if node.is_seed():
            return 0

        max_parent_depth = 0
        for parent_id in node.parent_ids:
            parent_depth = self.depth(parent_id)
            max_parent_depth = max(max_parent_depth, parent_depth)

        return max_parent_depth + 1

    def path_to_seed(self, trial_id: str) -> List[str]:
        """
        Get path from node to its earliest seed ancestor.

        Args:
            trial_id: Trial ID

        Returns:
            List of trial IDs from seed to target
        """
        node = self.get_node(trial_id)
        if node is None:
            return []

        if node.is_seed():
            return [trial_id]

        # Find path through first parent (arbitrary choice for multi-parent)
        if not node.parent_ids:
            return [trial_id]

        parent_path = self.path_to_seed(node.parent_ids[0])
        return parent_path + [trial_id]

    def best_in_lineage(self, trial_id: str) -> Optional[LineageNode]:
        """
        Find best scoring node in the lineage of a trial.

        Args:
            trial_id: Trial ID

        Returns:
            Best scoring ancestor/descendant, or None
        """
        # Get full lineage (ancestors + descendants)
        lineage_ids = self.get_ancestors(trial_id) | self.get_descendants(trial_id)
        lineage_ids.add(trial_id)

        # Filter nodes with scores
        scored_nodes = [
            self.nodes[tid] for tid in lineage_ids
            if tid in self.nodes and self.nodes[tid].score is not None
        ]

        if not scored_nodes:
            return None

        return max(scored_nodes, key=lambda n: n.score)

    def subtree_stats(self, trial_id: str) -> Dict[str, Any]:
        """
        Compute statistics for the subtree rooted at a node.

        Args:
            trial_id: Root trial ID

        Returns:
            Dictionary of statistics
        """
        descendants = self.get_descendants(trial_id)
        descendant_nodes = [self.nodes[did] for did in descendants if did in self.nodes]

        scored = [n for n in descendant_nodes if n.score is not None]
        scores = [n.score for n in scored]

        max_depth = 0
        if descendants:
            max_depth = max((self.depth(did) for did in descendants), default=0)

        return {
            "num_descendants": len(descendants),
            "num_scored": len(scored),
            "best_score": max(scores) if scores else None,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "max_depth": max_depth,
        }

    def prune_lineage(self, trial_id: str, keep_ancestors: bool = True) -> int:
        """
        Remove a node and optionally its descendants from the tree.

        Args:
            trial_id: Trial ID to remove
            keep_ancestors: If True, keep ancestors; otherwise remove full lineage

        Returns:
            Number of nodes removed
        """
        if trial_id not in self.nodes:
            return 0

        # Determine nodes to remove
        if keep_ancestors:
            to_remove = {trial_id} | self.get_descendants(trial_id)
        else:
            to_remove = {trial_id} | self.get_ancestors(trial_id) | self.get_descendants(trial_id)

        # Remove nodes
        for node_id in to_remove:
            if node_id in self.nodes:
                del self.nodes[node_id]
            if node_id in self.children:
                del self.children[node_id]

        # Clean up orphaned references
        for child_list in self.children.values():
            child_list[:] = [cid for cid in child_list if cid not in to_remove]

        return len(to_remove)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary."""
        return {
            "nodes": {
                tid: node.to_dict()
                for tid, node in self.nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageTree":
        """Deserialize tree from dictionary."""
        tree = cls()

        # First pass: add all nodes (will need topological sort for parents)
        nodes_data = data.get("nodes", {})

        # Add seeds first
        seeds = [
            (tid, ndata) for tid, ndata in nodes_data.items()
            if not ndata.get("parent_ids")
        ]

        for tid, ndata in seeds:
            node = LineageNode.from_dict(ndata)
            tree.nodes[node.trial_id] = node

        # Add remaining nodes (BFS order)
        remaining = {
            tid: ndata for tid, ndata in nodes_data.items()
            if tid not in tree.nodes
        }

        while remaining:
            added_any = False

            for tid in list(remaining.keys()):
                ndata = remaining[tid]
                parent_ids = ndata.get("parent_ids", [])

                # Check if all parents exist
                if all(pid in tree.nodes for pid in parent_ids):
                    node = LineageNode.from_dict(ndata)
                    tree.add_node(node)
                    del remaining[tid]
                    added_any = True

            if not added_any and remaining:
                # Orphaned nodes, add them anyway
                for tid, ndata in remaining.items():
                    node = LineageNode.from_dict(ndata)
                    tree.nodes[node.trial_id] = node
                break

        return tree


class LineageAnalyzer:
    """
    Analyzes mutation lineage for insights.

    Provides metrics and reports on lineage structure, effectiveness, and diversity.
    """

    def __init__(self, tree: LineageTree):
        """
        Initialize analyzer.

        Args:
            tree: Lineage tree to analyze
        """
        self.tree = tree

    def success_rate_by_mutation_type(self) -> Dict[str, float]:
        """
        Compute success rate for each mutation type.

        Success = scored better than parent.

        Returns:
            Dictionary mapping mutation type to success rate
        """
        type_stats: Dict[str, Dict[str, int]] = {}

        for node in self.tree.nodes.values():
            if node.is_seed() or node.mutation_type is None:
                continue

            if node.mutation_type not in type_stats:
                type_stats[node.mutation_type] = {"total": 0, "success": 0}

            type_stats[node.mutation_type]["total"] += 1

            # Check if better than best parent
            if node.score is not None:
                parents = self.tree.get_parents(node.trial_id)
                parent_scores = [p.score for p in parents if p.score is not None]

                if parent_scores:
                    best_parent_score = max(parent_scores)
                    if node.score > best_parent_score:
                        type_stats[node.mutation_type]["success"] += 1

        # Compute rates
        return {
            mtype: stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
            for mtype, stats in type_stats.items()
        }

    def generation_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Compute statistics for each generation.

        Returns:
            Dictionary mapping generation to stats
        """
        gen_stats: Dict[int, Dict[str, Any]] = {}

        for node in self.tree.nodes.values():
            gen = node.generation

            if gen not in gen_stats:
                gen_stats[gen] = {
                    "count": 0,
                    "scored": 0,
                    "scores": [],
                }

            gen_stats[gen]["count"] += 1

            if node.score is not None:
                gen_stats[gen]["scored"] += 1
                gen_stats[gen]["scores"].append(node.score)

        # Compute aggregates
        for gen, stats in gen_stats.items():
            scores = stats["scores"]
            stats["best_score"] = max(scores) if scores else None
            stats["mean_score"] = sum(scores) / len(scores) if scores else None
            stats["median_score"] = sorted(scores)[len(scores) // 2] if scores else None
            del stats["scores"]

        return gen_stats

    def most_productive_lineage(self) -> Optional[str]:
        """
        Find seed with the most successful descendants.

        Returns:
            Trial ID of most productive seed
        """
        seeds = self.tree.get_seeds()
        if not seeds:
            return None

        best_seed = None
        best_count = 0

        for seed in seeds:
            descendants = self.tree.get_descendants(seed.trial_id)
            scored = sum(
                1 for did in descendants
                if self.tree.nodes[did].score is not None
            )

            if scored > best_count:
                best_count = scored
                best_seed = seed.trial_id

        return best_seed

    def diversity_score(self) -> float:
        """
        Compute lineage diversity (how spread out across seeds).

        Returns:
            Diversity score in [0, 1] (higher = more diverse)
        """
        seeds = self.tree.get_seeds()
        if len(seeds) <= 1:
            return 0.0

        # Count descendants per seed
        seed_sizes = []
        for seed in seeds:
            descendants = self.tree.get_descendants(seed.trial_id)
            seed_sizes.append(len(descendants) + 1)  # +1 for seed itself

        total = sum(seed_sizes)
        if total == 0:
            return 0.0

        # Gini-Simpson index (1 - sum of squared proportions)
        gini = 0.0
        for size in seed_sizes:
            if size > 0:
                p = size / total
                gini += p * p

        # Higher diversity = lower concentration
        return 1.0 - gini
