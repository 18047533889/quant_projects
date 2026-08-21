"""
Metric dependency graph for ordered evaluation.

Resolves dependencies between metrics to determine evaluation order
and enable intermediate result caching.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum

from quant_evaluator.contracts.errors import InvalidContractError


class MetricKind(Enum):
    """Classification of metric types."""
    COVERAGE = "coverage"
    IC = "ic"
    QUANTILE = "quantile"
    TURNOVER = "turnover"
    RISK = "risk"
    EXPOSURE = "exposure"
    SUMMARY = "summary"
    CUSTOM = "custom"


@dataclass
class MetricNode:
    """
    Node in the metric dependency graph.

    Each node represents a metric computation with its dependencies
    and metadata about computation requirements.
    """
    metric_id: str
    metric_kind: MetricKind
    dependencies: Set[str] = field(default_factory=set)
    estimated_cost: float = 1.0
    requires_full_batch: bool = False
    parallelizable: bool = True
    metadata: Dict = field(default_factory=dict)

    def add_dependency(self, dep_id: str):
        """Add a dependency to this metric."""
        self.dependencies.add(dep_id)

    def has_dependency(self, dep_id: str) -> bool:
        """Check if this metric depends on another."""
        return dep_id in self.dependencies


@dataclass
class MetricDependencyGraph:
    """
    Dependency graph for metric evaluation.

    Tracks dependencies between metrics and provides topological ordering
    for execution.
    """
    nodes: Dict[str, MetricNode] = field(default_factory=dict)
    _sorted_order: Optional[List[str]] = field(default=None, init=False, repr=False)

    def add_node(self, node: MetricNode):
        """Add a metric node to the graph."""
        if node.metric_id in self.nodes:
            raise InvalidContractError(f"Metric {node.metric_id} already exists in graph")
        self.nodes[node.metric_id] = node
        self._sorted_order = None  # Invalidate cached order

    def add_dependency(self, from_metric: str, to_metric: str):
        """
        Add a dependency edge: from_metric depends on to_metric.

        Args:
            from_metric: Metric that has the dependency
            to_metric: Metric that must be computed first
        """
        if from_metric not in self.nodes:
            raise InvalidContractError(f"Metric {from_metric} not found in graph")
        if to_metric not in self.nodes:
            raise InvalidContractError(f"Metric {to_metric} not found in graph")

        self.nodes[from_metric].add_dependency(to_metric)
        self._sorted_order = None  # Invalidate cached order

    def get_node(self, metric_id: str) -> Optional[MetricNode]:
        """Get a metric node by ID."""
        return self.nodes.get(metric_id)

    def has_cycle(self) -> Tuple[bool, Optional[List[str]]]:
        """
        Check for cycles in the dependency graph.

        Returns:
            (has_cycle, cycle_path) where cycle_path is the detected cycle if any
        """
        visited = set()
        rec_stack = set()
        path = []

        def visit(node_id: str) -> Optional[List[str]]:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            node = self.nodes[node_id]
            for dep in node.dependencies:
                if dep not in visited:
                    cycle = visit(dep)
                    if cycle:
                        return cycle
                elif dep in rec_stack:
                    # Found cycle
                    cycle_start = path.index(dep)
                    return path[cycle_start:] + [dep]

            path.pop()
            rec_stack.remove(node_id)
            return None

        for node_id in self.nodes:
            if node_id not in visited:
                cycle = visit(node_id)
                if cycle:
                    return True, cycle

        return False, None

    def topological_sort(self) -> List[str]:
        """
        Return topologically sorted list of metric IDs.

        Metrics with no dependencies come first, followed by metrics
        that depend on them, and so on.

        Returns:
            List of metric IDs in execution order

        Raises:
            InvalidContractError: If graph has cycles
        """
        if self._sorted_order is not None:
            return self._sorted_order

        # Check for cycles first
        has_cycle, cycle_path = self.has_cycle()
        if has_cycle:
            raise InvalidContractError(f"Dependency cycle detected: {' -> '.join(cycle_path)}")

        # Kahn's algorithm for topological sort
        in_degree = {node_id: 0 for node_id in self.nodes}

        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                in_degree[node_id] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        sorted_order = []

        while queue:
            # Sort by estimated cost to process cheaper metrics first
            queue.sort(key=lambda nid: self.nodes[nid].estimated_cost)
            current = queue.pop(0)
            sorted_order.append(current)

            # Find all nodes that depend on current
            for node_id, node in self.nodes.items():
                if current in node.dependencies:
                    in_degree[node_id] -= 1
                    if in_degree[node_id] == 0:
                        queue.append(node_id)

        if len(sorted_order) != len(self.nodes):
            raise InvalidContractError("Failed to compute topological sort (graph may have cycle)")

        self._sorted_order = sorted_order
        return sorted_order

    def get_execution_levels(self) -> List[List[str]]:
        """
        Group metrics into execution levels.

        All metrics in a level can be executed in parallel, as they have
        no dependencies on each other.

        Returns:
            List of levels, where each level is a list of metric IDs
        """
        sorted_ids = self.topological_sort()
        levels = []
        processed = set()

        while processed != set(sorted_ids):
            current_level = []

            for metric_id in sorted_ids:
                if metric_id in processed:
                    continue

                node = self.nodes[metric_id]
                # Check if all dependencies are processed
                if node.dependencies.issubset(processed):
                    current_level.append(metric_id)

            if not current_level:
                raise InvalidContractError("Unable to form execution level (unexpected state)")

            levels.append(current_level)
            processed.update(current_level)

        return levels

    def estimate_total_cost(self) -> float:
        """Estimate total computational cost of all metrics."""
        return sum(node.estimated_cost for node in self.nodes.values())

    def get_critical_path(self) -> Tuple[List[str], float]:
        """
        Find the critical path (longest dependency chain by cost).

        Returns:
            (path, total_cost) where path is list of metric IDs
        """
        sorted_ids = self.topological_sort()

        # Dynamic programming to find longest path
        # dist[node] = longest path cost TO reach that node (not including node itself)
        dist = {node_id: 0.0 for node_id in self.nodes}
        parent = {node_id: None for node_id in self.nodes}

        for node_id in sorted_ids:
            node = self.nodes[node_id]
            current_dist = dist[node_id] + node.estimated_cost

            # Update distances for all dependents
            for dependent_id, dependent in self.nodes.items():
                if node_id in dependent.dependencies:
                    if current_dist > dist[dependent_id]:
                        dist[dependent_id] = current_dist
                        parent[dependent_id] = node_id

        # Find node with maximum distance (including its own cost)
        max_node = None
        max_cost = 0.0
        for node_id in self.nodes:
            node_cost = dist[node_id] + self.nodes[node_id].estimated_cost
            if node_cost > max_cost:
                max_cost = node_cost
                max_node = node_id

        # Reconstruct path
        path = []
        current = max_node
        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()
        return path, max_cost


def resolve_metric_dependencies(
    metric_specs: List[Dict],
) -> MetricDependencyGraph:
    """
    Build dependency graph from metric specifications.

    Args:
        metric_specs: List of metric specification dicts with keys:
            - metric_id: str
            - metric_kind: str or MetricKind
            - dependencies: List[str] (optional)
            - estimated_cost: float (optional)
            - requires_full_batch: bool (optional)
            - parallelizable: bool (optional)

    Returns:
        MetricDependencyGraph ready for execution planning

    Raises:
        InvalidContractError: If specs are invalid or graph has cycles
    """
    graph = MetricDependencyGraph()

    # First pass: create all nodes
    for spec in metric_specs:
        if "metric_id" not in spec:
            raise InvalidContractError("metric_id is required in metric spec")
        if "metric_kind" not in spec:
            raise InvalidContractError(f"metric_kind is required for {spec['metric_id']}")

        metric_kind = spec["metric_kind"]
        if isinstance(metric_kind, str):
            try:
                metric_kind = MetricKind(metric_kind)
            except ValueError:
                metric_kind = MetricKind.CUSTOM

        node = MetricNode(
            metric_id=spec["metric_id"],
            metric_kind=metric_kind,
            dependencies=set(spec.get("dependencies", [])),
            estimated_cost=spec.get("estimated_cost", 1.0),
            requires_full_batch=spec.get("requires_full_batch", False),
            parallelizable=spec.get("parallelizable", True),
            metadata=spec.get("metadata", {}),
        )
        graph.add_node(node)

    # Validate all dependencies exist
    for node in graph.nodes.values():
        for dep in node.dependencies:
            if dep not in graph.nodes:
                raise InvalidContractError(
                    f"Metric {node.metric_id} depends on unknown metric {dep}"
                )

    # Check for cycles
    has_cycle, cycle_path = graph.has_cycle()
    if has_cycle:
        raise InvalidContractError(f"Dependency cycle detected: {' -> '.join(cycle_path)}")

    return graph
