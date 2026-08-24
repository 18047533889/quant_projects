"""Artifact lineage graphs for the research platform.

Pure standard-library module (``dataclasses``, ``typing``). Provides two
directed graphs:

- :class:`ArtifactGraph`      -- the lineage (dependency) graph of artifacts.
- :class:`DataInvalidationGraph` -- reverse lookup used to find every artifact
  that would be invalidated when a given data snapshot is replaced.

Both are intentionally dependency-light and keep no reference to the wider
quant_projects codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class ArtifactGraph:
    """Directed acyclic lineage graph of artifacts.

    An edge ``source -> dependent`` records that the dependent artifact was
    derived from the source artifact (or sources).
    """

    # node -> set of its immediate upstream (source) node ids
    _parents: Dict[str, Set[str]] = field(default_factory=dict)
    # node -> set of its immediate downstream (dependent) node ids
    _children: Dict[str, Set[str]] = field(default_factory=dict)
    # reverse adjacency for invalidation queries (snapshot_ref -> dependents)
    _by_snapshot: Dict[str, Set[str]] = field(default_factory=dict)

    def add(self, dependent: str, *sources: str) -> None:
        """Register that ``dependent`` was built from ``sources``.

        The dependent node is added even if it has no sources (a root).
        """
        self._parents.setdefault(dependent, set())
        self._children.setdefault(dependent, set())
        for src in sources:
            self._parents.setdefault(src, set())
            self._children.setdefault(src, set())
            self._parents[dependent].add(src)
            self._children[src].add(dependent)

    # ------------------------------------------------------------------
    # Forward lineage
    # ------------------------------------------------------------------
    def ancestors(self, node_id: str) -> List[str]:
        """All transitive ancestors (oldest first), excluding the node itself."""
        visited: Set[str] = {node_id}  # never include the query node itself
        order: List[str] = []
        self._dfs_ancestors(node_id, visited, order)
        return order

    def _dfs_ancestors(self, node: str, visited: Set[str], order: List[str]) -> None:
        for p in self._parents.get(node, ()):
            if p not in visited:
                visited.add(p)
                order.append(p)
                self._dfs_ancestors(p, visited, order)

    def lineage(self, node_id: str) -> List[str]:
        """Alias for :meth:`ancestors`."""
        return self.ancestors(node_id)

    # ------------------------------------------------------------------
    # Reverse lineage
    # ------------------------------------------------------------------
    def dependents(self, node_id: str) -> List[str]:
        """All transitive dependents (downstream) of a node, incl. itself first.

        The returned list starts with ``node_id`` itself, then all artifacts
        that depend (directly or transitively) on it.
        """
        visited: Set[str] = set()
        order: List[str] = []
        self._dfs_dependents(node_id, visited, order)
        return order

    def _dfs_dependents(self, node: str, visited: Set[str], order: List[str]) -> None:
        if node not in visited:
            visited.add(node)
            order.append(node)
            for c in self._children.get(node, set()):
                self._dfs_dependents(c, visited, order)

    def mark_invalid(self, node_id: str) -> List[str]:
        """Mark ``node_id`` and everything downstream as invalid.

        Returns the affected (now-invalid) artifact ids in dependency order.
        """
        return self.dependents(node_id)

    # ------------------------------------------------------------------
    # Snapshot indexing
    # ------------------------------------------------------------------
    def index_snapshot(self, artifact_id: str, snapshot_ref: str) -> None:
        """Register that ``artifact_id`` reads from ``snapshot_ref``."""
        self._by_snapshot.setdefault(snapshot_ref, set()).add(artifact_id)

    def artifacts_using_snapshot(self, snapshot_ref: str) -> List[str]:
        """Artifacts that directly reference the given snapshot."""
        return sorted(self._by_snapshot.get(snapshot_ref, set()))

    @property
    def nodes(self) -> List[str]:
        return list(self._children.keys())


@dataclass
class DataInvalidationGraph:
    """Reverse-lookup invalidation graph.

    This is the graph you query to answer: *"snapshot S was replaced -- which
    artifacts must be recomputed?"*. It wraps an :class:`ArtifactGraph` plus a
    snapshot->artifact index.
    """

    lineage: ArtifactGraph = field(default_factory=ArtifactGraph)

    def add(self, dependent: str, snapshot_ref: str = "", *sources: str) -> None:
        """Add an artifact that depends on ``snapshot_ref`` plus other artifacts."""
        self.lineage.add(dependent, *sources)
        if snapshot_ref:
            self.lineage.index_snapshot(dependent, snapshot_ref)

    def mark_invalid(self, snapshot_ref: str) -> List[str]:
        """Return all artifacts affected by a snapshot replacement.

        Computed via reverse lookup: every artifact that (directly or through
        its transitive lineage) reads the snapshot is invalidated, along with
        everything downstream of those artifacts.
        """
        direct = self.lineage.artifacts_using_snapshot(snapshot_ref)
        affected: Set[str] = set()
        for art in direct:
            affected.update(self.lineage.dependents(art))
        # deterministic ordering
        return _topo_order(affected, self.lineage)


def _topo_order(nodes: Set[str], graph: ArtifactGraph) -> List[str]:
    """Return the affected nodes in dependency (ancestors-first) order."""
    visited: Set[str] = set()
    order: List[str] = []

    def visit(n: str) -> None:
        if n in visited:
            return
        visited.add(n)
        for p in graph._parents.get(n, set()):
            if p in nodes:
                visit(p)
        order.append(n)

    for n in nodes:
        visit(n)
    return order
