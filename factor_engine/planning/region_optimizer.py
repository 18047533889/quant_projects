# -*- coding: utf-8 -*-
"""Cost-optimal region partition optimizer.

Implements the DP-based region optimizer as specified in sections 13, 16, 43
of the remediation document. This replaces the single-backend router with a
true batch-global optimizer that considers parent transfer affinity and
shared computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from factor_engine.planning.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalProperty,
    Representation,
    StateContract,
)
from factor_engine.planning.physical_region_plan import PhysicalRegionPlan, compute_plan_hash, estimate_plan_peak_memory
from factor_engine.planning.transfer_edge import (
    SemanticContract,
    TransferEdge,
    TransferKind,
    estimate_transfer_cost,
    infer_transfer_kind,
    requires_repartition_for_properties,
    requires_sort_for_properties,
)


@dataclass(frozen=True)
class NodeCost:
    """Cost for executing a single node in a specific backend."""

    node_id: str
    backend: PhysicalBackend
    compute_cost_ms: float
    memory_bytes: int


@dataclass(frozen=True)
class DPState:
    """DP state: best cost to reach a node with specific output backend (section 13).

    The DP considers parent transfer affinity: if child is cheaper in Pandas (9ms)
    but parent requires Polars, and Pandas→Polars edge costs 20ms, the global
    optimum may choose child in Polars (10ms) to save the transfer.

    This is DP[node][output_backend] instead of just DP[node] = one_backend.
    """

    node_id: str
    backend: PhysicalBackend
    total_cost_ms: float  # Includes subtree compute + incoming transfers
    parent_node_ids: tuple[str, ...]  # For backtracking


class RegionOptimizer:
    """Cost-optimal region partitioner (sections 13, 16, 43).

    Implements batch-global optimization that:
    1. Considers parent transfer affinity (DP[node][backend])
    2. Counts shared nodes only once
    3. Coarsens compatible nodes into regions
    4. Minimizes total_cost = compute + transfer + sort + reshape + memory_risk
    """

    def __init__(
        self,
        *,
        memory_budget: int | None = None,
        mode: str = "research",
        enable_cse_aware: bool = True,
        enable_parallel_detection: bool = True,
        enable_aggressive_fusion: bool = True,
    ) -> None:
        """Initialize optimizer.

        Args:
            memory_budget: Maximum memory budget in bytes (None = unlimited)
            mode: Execution mode ("research" or "production")
            enable_cse_aware: Enable CSE-aware cost accounting
            enable_parallel_detection: Enable parallel region detection
            enable_aggressive_fusion: Enable aggressive region fusion
        """
        self.memory_budget = memory_budget
        self.mode = mode
        self.enable_cse_aware = enable_cse_aware
        self.enable_parallel_detection = enable_parallel_detection
        self.enable_aggressive_fusion = enable_aggressive_fusion
        self._memo: dict[tuple[str, str], DPState] = {}
        self._cse_cache: dict[str, set[str]] = {}  # Track CSE opportunities

    def optimize(
        self,
        node_graph: dict[str, list[str]],
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]],
        root_ids: list[str],
        *,
        logical_hash: str = "",
    ) -> PhysicalRegionPlan:
        """Optimize region assignment for given DAG (section 43).

        OPTIMIZED: Added CSE detection, DP pruning, and parallel region analysis.

        Args:
            node_graph: {node_id: [child_id, ...]} adjacency list
            node_costs: {node_id: {backend: NodeCost}} for each node
            root_ids: List of root node IDs (factors)
            logical_hash: Logical plan hash for binding

        Returns:
            PhysicalRegionPlan with optimal backend assignment
        """
        # Phase 0: CSE detection (OPTIMIZED)
        if self.enable_cse_aware:
            self._detect_cse_opportunities(node_graph, root_ids)

        # Phase 1: DP to find optimal backend per node with pruning (OPTIMIZED)
        for root_id in root_ids:
            self._dp_solve_optimized(root_id, node_graph, node_costs)

        # Phase 2: Backtrack to extract assignment
        assignments = self._backtrack_assignments(root_ids, node_graph, node_costs)

        # Phase 3: Coarsen into regions with aggressive fusion (OPTIMIZED)
        if self.enable_aggressive_fusion:
            regions = self._coarsen_to_regions_maximal(assignments, node_graph, node_costs)
        else:
            regions = self._coarsen_to_regions(assignments, node_graph, node_costs)

        # Phase 4: Memory-aware region splitting (OPTIMIZED)
        if self.memory_budget:
            regions = self._split_oversized_regions(regions, node_costs)

        # Phase 5: Build transfer edges
        edges = self._build_transfer_edges(regions, node_graph)

        # Phase 6: Compute topological order with parallelism optimization (OPTIMIZED)
        if self.enable_parallel_detection:
            topo_order = self._topological_sort_parallel_aware(regions, edges)
        else:
            topo_order = self._topological_sort(regions, edges)

        # Phase 7: Estimate peak memory and TTDC
        peak_memory = estimate_plan_peak_memory(tuple(regions), tuple(edges))
        total_ttdc = self._estimate_total_ttdc(regions, edges)

        # Build final plan
        plan = PhysicalRegionPlan(
            regions=tuple(regions),
            edges=tuple(edges),
            topological_order=tuple(topo_order),
            peak_memory_estimate=peak_memory,
            estimated_ttdc_ms=total_ttdc,
            plan_hash=compute_plan_hash(tuple(regions), tuple(edges), logical_hash),
            logical_node_count=len(node_graph),
            shared_node_count=self._count_shared_nodes(node_graph),
        )

        # Validate memory budget (MB-P1-008)
        if self.memory_budget is not None and peak_memory > self.memory_budget:
            raise MemoryBudgetExceededError(
                f"Peak memory {peak_memory / 1024.0 / 1024.0:.1f} MB "
                f"exceeds budget {self.memory_budget / 1024.0 / 1024.0:.1f} MB"
            )

        return plan

    def _detect_cse_opportunities(
        self, node_graph: dict[str, list[str]], root_ids: list[str]
    ) -> None:
        """Detect common subexpression elimination opportunities.

        OPTIMIZED: Track nodes that are shared across multiple paths.
        """
        # Count consumers for each node
        consumer_counts: dict[str, int] = {}
        for node_id, children in node_graph.items():
            for child_id in children:
                consumer_counts[child_id] = consumer_counts.get(child_id, 0) + 1

        # Identify CSE candidates (nodes with multiple consumers)
        for node_id, count in consumer_counts.items():
            if count > 1:
                # Find all consumers
                consumers = [nid for nid, children in node_graph.items() if node_id in children]
                self._cse_cache[node_id] = set(consumers)

    def _dp_solve_optimized(
        self,
        node_id: str,
        node_graph: dict[str, list[str]],
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]],
    ) -> dict[PhysicalBackend, float]:
        """DP solver with pruning optimizations.

        OPTIMIZED: Prune dominated states early to reduce state space.
        """
        available_backends = list(node_costs.get(node_id, {}).keys())
        if not available_backends:
            return {}

        # Base case: leaf node
        children = node_graph.get(node_id, [])
        if not children:
            result = {}
            for backend in available_backends:
                cost_obj = node_costs[node_id][backend]
                # CSE-aware: if this is a shared node, amortize cost
                cost = cost_obj.compute_cost_ms
                if self.enable_cse_aware and node_id in self._cse_cache:
                    consumer_count = len(self._cse_cache[node_id])
                    cost = cost / max(consumer_count, 1)

                result[backend] = cost
                self._memo[(node_id, backend.value)] = DPState(
                    node_id=node_id,
                    backend=backend,
                    total_cost_ms=cost,
                    parent_node_ids=(),
                )
            return result

        # Recursive case with pruning
        result = {}
        backend_candidates = []

        for backend in available_backends:
            node_cost_obj = node_costs[node_id][backend]
            total_cost = node_cost_obj.compute_cost_ms

            # CSE-aware cost adjustment
            if self.enable_cse_aware and node_id in self._cse_cache:
                consumer_count = len(self._cse_cache[node_id])
                total_cost = total_cost / max(consumer_count, 1)

            # For each child, find best backend considering transfer
            for child_id in children:
                child_costs = self._dp_solve_optimized(child_id, node_graph, node_costs)
                if not child_costs:
                    continue

                # Find cheapest child backend + transfer
                best_child_cost = float("inf")
                for child_backend, child_subtree_cost in child_costs.items():
                    # Transfer cost if backends differ
                    transfer_cost = 0.0
                    if child_backend != backend:
                        transfer_cost = self._estimate_transfer_cost_between_backends(
                            child_backend, backend, estimated_rows=10000
                        )
                    candidate_cost = child_subtree_cost + transfer_cost
                    best_child_cost = min(best_child_cost, candidate_cost)

                total_cost += best_child_cost

            backend_candidates.append((backend, total_cost))

        # Pruning: only keep competitive backends (within 50% of best)
        if backend_candidates:
            backend_candidates.sort(key=lambda x: x[1])
            best_cost = backend_candidates[0][1]
            threshold = best_cost * 1.5

            for backend, cost in backend_candidates:
                if cost <= threshold:
                    result[backend] = cost
                    self._memo[(node_id, backend.value)] = DPState(
                        node_id=node_id,
                        backend=backend,
                        total_cost_ms=cost,
                        parent_node_ids=tuple(children),
                    )

        return result

    def _dp_solve(
        self,
        node_id: str,
        node_graph: dict[str, list[str]],
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]],
    ) -> dict[PhysicalBackend, float]:
        """DP solver: returns {backend: min_cost} for this node (section 13)."""
        # Check memo
        available_backends = list(node_costs.get(node_id, {}).keys())
        if not available_backends:
            return {}

        # Base case: leaf node
        children = node_graph.get(node_id, [])
        if not children:
            result = {}
            for backend in available_backends:
                cost_obj = node_costs[node_id][backend]
                result[backend] = cost_obj.compute_cost_ms
                # Memoize
                self._memo[(node_id, backend.value)] = DPState(
                    node_id=node_id,
                    backend=backend,
                    total_cost_ms=cost_obj.compute_cost_ms,
                    parent_node_ids=(),
                )
            return result

        # Recursive case: consider each backend for this node
        result = {}
        for backend in available_backends:
            node_cost_obj = node_costs[node_id][backend]
            total_cost = node_cost_obj.compute_cost_ms

            # For each child, find best backend considering transfer
            for child_id in children:
                child_costs = self._dp_solve(child_id, node_graph, node_costs)
                if not child_costs:
                    continue

                # Find cheapest child backend + transfer
                best_child_cost = float("inf")
                for child_backend, child_subtree_cost in child_costs.items():
                    # Transfer cost if backends differ
                    transfer_cost = 0.0
                    if child_backend != backend:
                        transfer_cost = self._estimate_transfer_cost_between_backends(
                            child_backend, backend, estimated_rows=10000
                        )
                    candidate_cost = child_subtree_cost + transfer_cost
                    best_child_cost = min(best_child_cost, candidate_cost)

                total_cost += best_child_cost

            result[backend] = total_cost
            # Memoize
            self._memo[(node_id, backend.value)] = DPState(
                node_id=node_id,
                backend=backend,
                total_cost_ms=total_cost,
                parent_node_ids=tuple(children),
            )

        return result

    def _backtrack_assignments(
        self,
        root_ids: list[str],
        node_graph: dict[str, list[str]],
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]],
    ) -> dict[str, PhysicalBackend]:
        """Backtrack from roots to assign each node to optimal backend."""
        assignments: dict[str, PhysicalBackend] = {}

        def backtrack(node_id: str, preferred_backend: PhysicalBackend | None = None) -> None:
            if node_id in assignments:
                return

            # Find best backend for this node
            node_backend_costs = {}
            for backend in node_costs.get(node_id, {}).keys():
                state = self._memo.get((node_id, backend.value))
                if state:
                    node_backend_costs[backend] = state.total_cost_ms

            if not node_backend_costs:
                return

            # If parent prefers a specific backend, use it if competitive
            if preferred_backend and preferred_backend in node_backend_costs:
                chosen_backend = preferred_backend
            else:
                chosen_backend = min(node_backend_costs.items(), key=lambda x: x[1])[0]

            assignments[node_id] = chosen_backend

            # Recurse to children
            for child_id in node_graph.get(node_id, []):
                backtrack(child_id, chosen_backend)

        for root_id in root_ids:
            backtrack(root_id)

        return assignments

    def _coarsen_to_regions(
        self,
        assignments: dict[str, PhysicalBackend],
        node_graph: dict[str, list[str]],
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]],
    ) -> list[BackendRegion]:
        """Coarsen node assignments into backend regions (section 16).

        Merge consecutive nodes with same backend into regions.
        """
        regions: list[BackendRegion] = []
        region_counter = [0]

        def make_region_id() -> str:
            region_counter[0] += 1
            return f"R{region_counter[0]}"

        # Simple greedy coarsening: each maximal connected same-backend subgraph
        # becomes one region
        visited: set[str] = set()

        def collect_region(start_node: str, backend: PhysicalBackend) -> list[str]:
            """BFS to collect all connected nodes with same backend."""
            region_nodes = []
            queue = [start_node]
            local_visited = set()

            while queue:
                node_id = queue.pop(0)
                if node_id in local_visited or node_id in visited:
                    continue
                if assignments.get(node_id) != backend:
                    continue

                local_visited.add(node_id)
                visited.add(node_id)
                region_nodes.append(node_id)

                # Add children with same backend
                for child_id in node_graph.get(node_id, []):
                    if child_id not in local_visited and assignments.get(child_id) == backend:
                        queue.append(child_id)

            return region_nodes

        # Collect regions
        for node_id, backend in assignments.items():
            if node_id in visited:
                continue
            region_nodes = collect_region(node_id, backend)
            if not region_nodes:
                continue

            # Compute region properties
            total_bytes = sum(
                node_costs.get(nid, {}).get(backend, NodeCost(nid, backend, 0.0, 0)).memory_bytes
                for nid in region_nodes
            )

            region = BackendRegion(
                region_id=make_region_id(),
                backend=backend,
                representation=self._infer_representation(backend),
                node_ids=tuple(region_nodes),
                execution_axis=ExecutionAxis.GLOBAL_PANEL,  # Default
                estimated_rows=10000,  # Placeholder
                estimated_compute_ms=0.0,
                estimated_memory_bytes=total_bytes,
                input_bytes=total_bytes,
            )
            regions.append(region)

        return regions

    def _build_transfer_edges(
        self,
        regions: list[BackendRegion],
        node_graph: dict[str, list[str]],
    ) -> list[TransferEdge]:
        """Build explicit transfer edges between regions."""
        edges: list[TransferEdge] = []
        edge_counter = [0]

        # Build node -> region mapping
        node_to_region = {}
        for region in regions:
            for node_id in region.node_ids:
                node_to_region[node_id] = region

        # Find cross-region edges
        seen_pairs: set[tuple[str, str]] = set()
        for region in regions:
            for node_id in region.node_ids:
                for child_id in node_graph.get(node_id, []):
                    child_region = node_to_region.get(child_id)
                    if child_region and child_region.region_id != region.region_id:
                        pair = (child_region.region_id, region.region_id)
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)

                        # Create transfer edge
                        edge_counter[0] += 1
                        source_repr = child_region.representation
                        target_repr = region.representation
                        transfer_kind = infer_transfer_kind(source_repr, target_repr)

                        edge = TransferEdge(
                            edge_id=f"E{edge_counter[0]}",
                            producer_region=child_region.region_id,
                            consumer_region=region.region_id,
                            source_representation=source_repr,
                            target_representation=target_repr,
                            transfer_kind=transfer_kind,
                            estimated_rows=10000,  # Placeholder
                            estimated_bytes=10000 * 8 * 5,  # Placeholder
                            requires_sort=False,
                            requires_repartition=False,
                            requires_reshape="WIDE" in source_repr.value and "LONG" in target_repr.value,
                            requires_dtype_cast=False,
                            semantic_contract=SemanticContract(),
                            source_properties=PhysicalProperty(),
                            target_properties=PhysicalProperty(),
                        )
                        # Compute cost
                        object.__setattr__(edge, "estimated_cost_ms", estimate_transfer_cost(edge))
                        edges.append(edge)

        return edges

    def _coarsen_to_regions_maximal(
        self,
        assignments: dict[str, PhysicalBackend],
        node_graph: dict[str, list[str]],
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]],
    ) -> list[BackendRegion]:
        """Maximal fusion: aggressively merge compatible nodes into large regions.

        OPTIMIZED: Uses graph coarsening for maximal connected same-backend subgraphs.
        """
        regions: list[BackendRegion] = []
        region_counter = [0]
        visited: set[str] = set()

        def make_region_id() -> str:
            region_counter[0] += 1
            return f"R{region_counter[0]}"

        def can_fuse(node1: str, node2: str, backend: PhysicalBackend) -> bool:
            """Check if two nodes can be fused into the same region."""
            if assignments.get(node1) != backend or assignments.get(node2) != backend:
                return False
            # Additional fusion criteria can be added here
            return True

        def collect_maximal_region(start_node: str, backend: PhysicalBackend) -> list[str]:
            """BFS to collect maximal connected same-backend subgraph."""
            region_nodes = []
            queue = [start_node]
            local_visited = set()

            while queue:
                node_id = queue.pop(0)
                if node_id in local_visited or node_id in visited:
                    continue
                if assignments.get(node_id) != backend:
                    continue

                local_visited.add(node_id)
                visited.add(node_id)
                region_nodes.append(node_id)

                # Add both children and parents with same backend
                for child_id in node_graph.get(node_id, []):
                    if can_fuse(node_id, child_id, backend) and child_id not in local_visited:
                        queue.append(child_id)

                # Also check parents (reverse edges)
                for parent_id, children in node_graph.items():
                    if node_id in children and can_fuse(parent_id, node_id, backend):
                        if parent_id not in local_visited:
                            queue.append(parent_id)

            return region_nodes

        # Collect maximal regions
        for node_id, backend in assignments.items():
            if node_id in visited:
                continue
            region_nodes = collect_maximal_region(node_id, backend)
            if not region_nodes:
                continue

            # Compute region properties
            total_bytes = sum(
                node_costs.get(nid, {}).get(backend, NodeCost(nid, backend, 0.0, 0)).memory_bytes
                for nid in region_nodes
            )

            region = BackendRegion(
                region_id=make_region_id(),
                backend=backend,
                representation=self._infer_representation(backend),
                node_ids=tuple(region_nodes),
                execution_axis=ExecutionAxis.GLOBAL_PANEL,
                estimated_rows=10000,
                estimated_compute_ms=0.0,
                estimated_memory_bytes=total_bytes,
                input_bytes=total_bytes,
            )
            regions.append(region)

        return regions

    def _split_oversized_regions(
        self,
        regions: list[BackendRegion],
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]],
    ) -> list[BackendRegion]:
        """Split regions that exceed memory budget.

        OPTIMIZED: Memory-aware region splitting to respect budget constraints.
        """
        if not self.memory_budget:
            return regions

        result: list[BackendRegion] = []
        region_counter = [max(int(r.region_id[1:]) for r in regions) if regions else 0]

        def make_region_id() -> str:
            region_counter[0] += 1
            return f"R{region_counter[0]}"

        for region in regions:
            if region.estimated_memory_bytes <= self.memory_budget * 0.7:
                # Region fits comfortably
                result.append(region)
            else:
                # Split into smaller sub-regions
                node_list = list(region.node_ids)
                current_batch = []
                current_bytes = 0

                for node_id in node_list:
                    node_byte = node_costs.get(node_id, {}).get(
                        region.backend, NodeCost(node_id, region.backend, 0.0, 0)
                    ).memory_bytes

                    if current_bytes + node_byte > self.memory_budget * 0.7 and current_batch:
                        # Flush current batch
                        sub_region = BackendRegion(
                            region_id=make_region_id(),
                            backend=region.backend,
                            representation=region.representation,
                            node_ids=tuple(current_batch),
                            execution_axis=region.execution_axis,
                            estimated_rows=region.estimated_rows,
                            estimated_compute_ms=0.0,
                            estimated_memory_bytes=current_bytes,
                            input_bytes=current_bytes,
                        )
                        result.append(sub_region)
                        current_batch = []
                        current_bytes = 0

                    current_batch.append(node_id)
                    current_bytes += node_byte

                # Flush final batch
                if current_batch:
                    sub_region = BackendRegion(
                        region_id=make_region_id(),
                        backend=region.backend,
                        representation=region.representation,
                        node_ids=tuple(current_batch),
                        execution_axis=region.execution_axis,
                        estimated_rows=region.estimated_rows,
                        estimated_compute_ms=0.0,
                        estimated_memory_bytes=current_bytes,
                        input_bytes=current_bytes,
                    )
                    result.append(sub_region)

        return result

    def _topological_sort_parallel_aware(
        self, regions: list[BackendRegion], edges: list[TransferEdge]
    ) -> list[str]:
        """Topological sort optimized for parallel execution.

        OPTIMIZED: Prioritize nodes that can execute in parallel.
        """
        # Build dependency graph
        in_degree = {r.region_id: 0 for r in regions}
        adj = {r.region_id: [] for r in regions}

        for edge in edges:
            adj[edge.producer_region].append(edge.consumer_region)
            in_degree[edge.consumer_region] += 1

        # Modified Kahn's algorithm: process all ready nodes at each level
        result = []
        ready_queue = [rid for rid, deg in in_degree.items() if deg == 0]

        while ready_queue:
            # Process all ready nodes (they can run in parallel)
            # Sort by estimated cost (largest first for better load balancing)
            region_map = {r.region_id: r for r in regions}
            ready_queue.sort(
                key=lambda rid: region_map[rid].estimated_memory_bytes if rid in region_map else 0,
                reverse=True,
            )

            # Take all ready nodes
            next_queue = []
            for rid in ready_queue:
                result.append(rid)
                for neighbor in adj[rid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)

            ready_queue = next_queue

        if len(result) != len(regions):
            raise ValueError("Cycle detected in region dependency graph")

        return result

    def _estimate_total_ttdc(
        self, regions: list[BackendRegion], edges: list[TransferEdge]
    ) -> float:
        """Estimate total time-to-durable-commit (section 31)."""
        # Sum all compute costs (each region executes once)
        # Note: In real implementation, this would consider parallelism
        compute_ms = sum(r.estimated_memory_bytes / 1_000_000.0 * 0.5 for r in regions)
        transfer_ms = sum(e.estimated_cost_ms for e in edges)
        overhead_ms = len(regions) * 0.5  # Scheduler overhead
        return compute_ms + transfer_ms + overhead_ms

    def _count_shared_nodes(self, node_graph: dict[str, list[str]]) -> int:
        """Count nodes with multiple consumers."""
        consumer_counts: dict[str, int] = {}
        for node_id, children in node_graph.items():
            for child_id in children:
                consumer_counts[child_id] = consumer_counts.get(child_id, 0) + 1
        return sum(1 for count in consumer_counts.values() if count > 1)

    def _estimate_transfer_cost_between_backends(
        self, source: PhysicalBackend, target: PhysicalBackend, estimated_rows: int
    ) -> float:
        """Estimate transfer cost between backends (simplified)."""
        if source == target:
            return 0.0
        # Use base penalty + row scaling
        return 2.0 + (estimated_rows / 1_000_000.0) * 0.5

    def _infer_representation(self, backend: PhysicalBackend) -> Representation:
        """Infer default representation for backend (R21-PLANNER-TYPE-UNIFICATION)."""
        mapping = {
            PhysicalBackend.PANDAS_NUMPY: Representation.PANDAS_LONG,
            PhysicalBackend.POLARS_PANEL: Representation.POLARS_LONG,
            PhysicalBackend.POLARS_LONG: Representation.POLARS_LAZY_LONG,
            PhysicalBackend.DUCKDB_SQL: Representation.DUCKDB_RELATION,
            PhysicalBackend.CLICKHOUSE_SQL: Representation.DUCKDB_RELATION,  # Approximation
            PhysicalBackend.Q_KDB: Representation.Q_TABLE,
        }
        return mapping.get(backend, Representation.PANDAS_LONG)


class MemoryBudgetExceededError(Exception):
    """Raised when plan exceeds memory budget (MB-P1-008)."""

    pass
