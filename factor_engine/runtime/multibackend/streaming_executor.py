# -*- coding: utf-8 -*-
"""MB-P1-006: Streaming-first execution priority.

Prioritizes streaming-capable operators over materialization to reduce
peak memory usage:
- Detect streaming-capable operator chains
- Schedule streaming paths first
- Delay materialization until necessary
- Pipeline operators when possible

Key principles:
- Streaming > partial materialization > full materialization
- Prefer push-based pipelines over pull-based
- Materialize only at pipeline breaks (sort, shuffle, aggregation)
- Enable operator fusion in streaming regions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingRegion:
    """A maximal streaming-capable region in the DAG.

    Attributes:
        region_id: Unique region identifier
        task_ids: Tasks in this streaming region
        entry_tasks: Tasks at region entry (receive materialized input)
        exit_tasks: Tasks at region exit (produce materialized output)
        estimated_throughput: Estimated rows/sec throughput
        memory_multiplier: Peak memory as fraction of input
        can_fuse: Whether operators can be fused into single pass
    """
    region_id: str
    task_ids: list[str]
    entry_tasks: list[str]
    exit_tasks: list[str]
    estimated_throughput: float
    memory_multiplier: float
    can_fuse: bool


@dataclass
class StreamingExecutionPlan:
    """Execution plan optimized for streaming.

    Attributes:
        streaming_regions: Detected streaming regions
        materialization_points: Required materialization task IDs
        pipeline_groups: Groups of tasks that can execute in pipeline
        execution_order: Recommended execution order
        metadata: Additional planning metadata
    """
    streaming_regions: list[StreamingRegion] = field(default_factory=list)
    materialization_points: list[str] = field(default_factory=list)
    pipeline_groups: list[list[str]] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingExecutionPlanner:
    """Plans streaming-first execution for DAG.

    Analyzes DAG to identify streaming regions and schedules them
    with priority to minimize memory footprint.
    """

    # Operators that support streaming
    _STREAMING_OPS = {
        "add", "subtract", "multiply", "divide", "negate",
        "clip", "fillna", "replace",
        "ts_delay", "ts_delta",
        "filter", "where",
        "col", "literal",
    }

    # Operators that require materialization (pipeline breaks)
    _MATERIALIZATION_OPS = {
        "rank", "quantile", "sort",
        "group_neutralize", "cs_rank", "cs_neutralize",
        "pivot", "unpivot", "reshape",
        "join", "merge",
        "aggregate", "reduce",
    }

    def __init__(self):
        self._custom_streaming_ops = set()
        self._custom_materialization_ops = set()

    def plan_streaming_execution(
        self,
        dag: Any,
        *,
        enable_fusion: bool = True,
        max_pipeline_depth: int = 10,
    ) -> StreamingExecutionPlan:
        """Plan streaming-optimized execution for DAG.

        Args:
            dag: PhysicalFactorDAG to analyze
            enable_fusion: Enable operator fusion in streaming regions
            max_pipeline_depth: Maximum pipeline depth before materialization

        Returns:
            StreamingExecutionPlan with regions and ordering
        """
        try:
            # Classify operators
            streaming_tasks, materialization_tasks = self._classify_operators(dag)

            # Find streaming regions
            regions = self._find_streaming_regions(
                dag, streaming_tasks, materialization_tasks
            )

            # Build pipeline groups
            pipeline_groups = self._build_pipeline_groups(
                regions, enable_fusion, max_pipeline_depth
            )

            # Determine execution order (streaming regions first)
            execution_order = self._schedule_streaming_first(
                dag, regions, materialization_tasks
            )

            plan = StreamingExecutionPlan(
                streaming_regions=regions,
                materialization_points=materialization_tasks,
                pipeline_groups=pipeline_groups,
                execution_order=execution_order,
                metadata={
                    "streaming_region_count": len(regions),
                    "materialization_point_count": len(materialization_tasks),
                    "pipeline_group_count": len(pipeline_groups),
                    "fusion_enabled": enable_fusion,
                },
            )

            _logger.info(
                f"Planned streaming execution: {len(regions)} regions, "
                f"{len(materialization_tasks)} materialization points"
            )

            return plan

        except Exception as exc:
            _logger.error(f"Streaming execution planning failed: {exc}")
            return StreamingExecutionPlan()

    def _classify_operators(
        self, dag: Any
    ) -> tuple[list[str], list[str]]:
        """Classify tasks into streaming vs materialization.

        Args:
            dag: DAG to analyze

        Returns:
            (streaming_task_ids, materialization_task_ids)
        """
        streaming = []
        materialization = []

        for task_id, task in self._iter_tasks(dag):
            op = self._get_operator(task)
            if op is None:
                continue

            if self._is_streaming_capable(op):
                streaming.append(task_id)
            else:
                materialization.append(task_id)

        return streaming, materialization

    def _find_streaming_regions(
        self,
        dag: Any,
        streaming_tasks: list[str],
        materialization_tasks: list[str],
    ) -> list[StreamingRegion]:
        """Find maximal streaming regions in DAG.

        Args:
            dag: DAG to analyze
            streaming_tasks: Streaming-capable task IDs
            materialization_tasks: Materialization-required task IDs

        Returns:
            List of StreamingRegion
        """
        regions = []
        visited = set()

        for task_id in streaming_tasks:
            if task_id in visited:
                continue

            # BFS to find connected streaming region
            region_tasks = []
            queue = [task_id]
            visited.add(task_id)

            while queue:
                current = queue.pop(0)
                region_tasks.append(current)

                # Explore streaming neighbors
                for neighbor in self._get_neighbors(dag, current):
                    if neighbor in streaming_tasks and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if not region_tasks:
                continue

            # Find entry and exit tasks
            entry_tasks = [
                tid for tid in region_tasks
                if self._has_materialized_input(dag, tid, streaming_tasks)
            ]
            exit_tasks = [
                tid for tid in region_tasks
                if self._has_materialized_output(dag, tid, streaming_tasks)
            ]

            # Estimate region properties
            throughput = 1_000_000.0  # 1M rows/sec default
            memory_mult = 1.2  # 20% overhead for streaming
            can_fuse = len(region_tasks) <= 5  # Small regions can fuse

            region = StreamingRegion(
                region_id=f"stream_{len(regions)}",
                task_ids=region_tasks,
                entry_tasks=entry_tasks or [region_tasks[0]],
                exit_tasks=exit_tasks or [region_tasks[-1]],
                estimated_throughput=throughput,
                memory_multiplier=memory_mult,
                can_fuse=can_fuse,
            )
            regions.append(region)

        return regions

    def _build_pipeline_groups(
        self,
        regions: list[StreamingRegion],
        enable_fusion: bool,
        max_depth: int,
    ) -> list[list[str]]:
        """Build pipeline groups from streaming regions.

        Args:
            regions: Streaming regions
            enable_fusion: Enable fusion
            max_depth: Max pipeline depth

        Returns:
            List of pipeline groups (each group is list of task IDs)
        """
        pipeline_groups = []

        for region in regions:
            if enable_fusion and region.can_fuse and len(region.task_ids) <= max_depth:
                # Entire region is one pipeline
                pipeline_groups.append(region.task_ids)
            else:
                # Break into smaller pipelines
                tasks = region.task_ids
                for i in range(0, len(tasks), max_depth):
                    group = tasks[i:i + max_depth]
                    pipeline_groups.append(group)

        return pipeline_groups

    def _schedule_streaming_first(
        self,
        dag: Any,
        regions: list[StreamingRegion],
        materialization_tasks: list[str],
    ) -> list[str]:
        """Schedule tasks with streaming regions prioritized.

        Args:
            dag: DAG to schedule
            regions: Streaming regions
            materialization_tasks: Materialization tasks

        Returns:
            Ordered list of task IDs
        """
        # Build sets for fast lookup
        streaming_task_set = set()
        for region in regions:
            streaming_task_set.update(region.task_ids)

        # Get topological order
        topo_order = self._topological_order(dag)

        # Select streaming tasks first only among tasks whose dependencies are
        # already satisfied. This preserves topology across pipeline breaks.
        topo_position = {task_id: index for index, task_id in enumerate(topo_order)}
        remaining = set(topo_order)
        completed: set[str] = set()
        order = []
        while remaining:
            ready = [
                task_id
                for task_id in remaining
                if set(self._get_dependencies(dag, task_id)).issubset(completed)
            ]
            if not ready:
                _logger.warning("Cannot schedule cyclic or incomplete task graph")
                return []
            ready.sort(
                key=lambda task_id: (
                    task_id not in streaming_task_set,
                    topo_position[task_id],
                )
            )
            selected = ready[0]
            order.append(selected)
            completed.add(selected)
            remaining.remove(selected)
        return order

    def _is_streaming_capable(self, op: str) -> bool:
        """Check if operator supports streaming execution."""
        if op in self._MATERIALIZATION_OPS or op in self._custom_materialization_ops:
            return False
        return op in self._STREAMING_OPS or op in self._custom_streaming_ops

    def _has_materialized_input(
        self, dag: Any, task_id: str, streaming_tasks: list[str]
    ) -> bool:
        """Check if task receives input from materialized source."""
        deps = self._get_dependencies(dag, task_id)
        return any(d not in streaming_tasks for d in deps)

    def _has_materialized_output(
        self, dag: Any, task_id: str, streaming_tasks: list[str]
    ) -> bool:
        """Check if task produces output consumed by materialized sink."""
        consumers = self._get_consumers(dag, task_id)
        return any(c not in streaming_tasks for c in consumers)

    def _get_operator(self, task: Any) -> str | None:
        """Extract operator name from task."""
        try:
            node = getattr(task, "node_ref", None)
            if node is None:
                return None
            return getattr(node, "op", None)
        except Exception:
            return None

    def _iter_tasks(self, dag: Any) -> list[tuple[str, Any]]:
        """Iterate over tasks in DAG."""
        try:
            tasks = getattr(dag, "tasks", {})
            return list(tasks.items())
        except Exception:
            return []

    def _get_neighbors(self, dag: Any, task_id: str) -> list[str]:
        """Get neighboring tasks (dependencies + consumers)."""
        deps = self._get_dependencies(dag, task_id)
        consumers = self._get_consumers(dag, task_id)
        return deps + consumers

    def _get_dependencies(self, dag: Any, task_id: str) -> list[str]:
        """Get task dependencies."""
        try:
            tasks = getattr(dag, "tasks", {})
            task = tasks.get(task_id)
            if task is None:
                return []
            return list(getattr(task, "dependencies", []))
        except Exception:
            return []

    def _get_consumers(self, dag: Any, task_id: str) -> list[str]:
        """Get tasks consuming this task's output."""
        try:
            consumers = []
            for tid, task in self._iter_tasks(dag):
                deps = self._get_dependencies(dag, tid)
                if task_id in deps:
                    consumers.append(tid)
            return consumers
        except Exception:
            return []

    def _topological_order(self, dag: Any) -> list[str]:
        """Get topological order of tasks."""
        try:
            if hasattr(dag, "topological_order"):
                return dag.topological_order()

            # Manual topological sort
            in_degree = {}
            for task_id, _ in self._iter_tasks(dag):
                deps = self._get_dependencies(dag, task_id)
                in_degree[task_id] = len(deps)

            queue = [tid for tid, deg in in_degree.items() if deg == 0]
            order = []

            while queue:
                task_id = queue.pop(0)
                order.append(task_id)

                for consumer in self._get_consumers(dag, task_id):
                    in_degree[consumer] -= 1
                    if in_degree[consumer] == 0:
                        queue.append(consumer)

            return order

        except Exception as exc:
            _logger.warning(f"Topological sort failed: {exc}")
            return []

    def register_streaming_operator(self, op: str) -> None:
        """Register custom streaming-capable operator."""
        self._custom_streaming_ops.add(op)

    def register_materialization_operator(self, op: str) -> None:
        """Register custom materialization-required operator."""
        self._custom_materialization_ops.add(op)


class StreamingExecutor:
    """Executor for streaming-optimized execution plans.

    Executes streaming regions with minimal materialization.
    """

    def __init__(self, planner: StreamingExecutionPlanner | None = None):
        """Initialize streaming executor.

        Args:
            planner: Optional planner (creates default if None)
        """
        self._planner = planner or StreamingExecutionPlanner()

    def execute_streaming_plan(
        self,
        plan: StreamingExecutionPlan,
        executor: Any,
    ) -> dict[str, Any]:
        """Execute streaming plan using provided executor.

        Args:
            plan: Streaming execution plan
            executor: Underlying task executor

        Returns:
            Execution results and statistics
        """
        results = {}
        stats = {
            "streaming_region_count": len(plan.streaming_regions),
            "materialization_count": len(plan.materialization_points),
            "pipeline_group_count": len(plan.pipeline_groups),
        }

        try:
            # Execute in streaming-first order
            for task_id in plan.execution_order:
                result = self._execute_task(task_id, executor)
                results[task_id] = result

            return {"results": results, "stats": stats}

        except Exception as exc:
            _logger.error(f"Streaming execution failed: {exc}")
            return {"results": results, "stats": stats, "error": str(exc)}

    def _execute_task(self, task_id: str, executor: Any) -> Any:
        """Execute single task using executor."""
        # Delegate to underlying executor
        if hasattr(executor, "execute_task"):
            return executor.execute_task(task_id)
        return None
