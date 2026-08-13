# -*- coding: utf-8 -*-
"""MB-P1-005: Liveness analysis with refcount for timely release.

Tracks data liveness across DAG execution to enable timely memory release:
- Reference counting for intermediate results
- Dead-after-use detection for eager cleanup
- Live range analysis for memory pressure prediction
- Integration with adaptive_batch_scheduler for memory governance

Key principles:
- Track last use of each intermediate
- Release memory immediately after last consumer
- Predict peak memory by analyzing live ranges
- Enable proactive spilling before OOM
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from weakref import WeakValueDictionary

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveRange:
    """Live range for an intermediate result.

    Attributes:
        task_id: Task producing this intermediate
        first_use: Task ID of first consumer
        last_use: Task ID of last consumer
        consumer_count: Total number of consumers
        size_bytes: Memory footprint estimate
        is_materialized: Whether result is materialized
        can_spill: Whether result can be spilled to disk
    """
    task_id: str
    first_use: str
    last_use: str
    consumer_count: int
    size_bytes: int
    is_materialized: bool = False
    can_spill: bool = True


@dataclass
class LivenessState:
    """Current liveness state during execution.

    Attributes:
        live_intermediates: Currently live task_id -> (refcount, size_bytes)
        peak_live_bytes: Peak memory observed
        current_live_bytes: Current live memory
        released_count: Number of intermediates released
        spilled_count: Number of intermediates spilled
    """
    live_intermediates: dict[str, tuple[int, int]] = field(default_factory=dict)
    peak_live_bytes: int = 0
    current_live_bytes: int = 0
    released_count: int = 0
    spilled_count: int = 0


class LivenessAnalyzer:
    """Liveness analyzer for DAG execution.

    Performs static analysis on DAG to determine live ranges, then
    tracks refcounts during execution for timely memory release.
    """

    def __init__(self):
        self._live_ranges: dict[str, LiveRange] = {}
        self._lock = threading.RLock()

    def analyze_dag(
        self,
        dag: Any,
        *,
        size_estimates: dict[str, int] | None = None,
    ) -> dict[str, LiveRange]:
        """Analyze DAG to compute live ranges for all intermediates.

        Args:
            dag: PhysicalFactorDAG to analyze
            size_estimates: Optional size estimates per task

        Returns:
            Mapping from task_id to LiveRange
        """
        with self._lock:
            size_est = size_estimates or {}
            live_ranges = {}

            # Build dependency graph
            dependencies = self._build_dependency_graph(dag)
            reverse_deps = self._build_reverse_dependencies(dependencies)

            # Compute live ranges
            for task_id, task in self._iter_tasks(dag):
                consumers = reverse_deps.get(task_id, [])
                if not consumers:
                    # No consumers (root or unused)
                    continue

                # First use = earliest consumer in topological order
                # Last use = latest consumer
                consumer_ids = [c for c in consumers]
                first_use = min(consumer_ids, key=lambda c: self._task_order(dag, c))
                last_use = max(consumer_ids, key=lambda c: self._task_order(dag, c))

                live_range = LiveRange(
                    task_id=task_id,
                    first_use=first_use,
                    last_use=last_use,
                    consumer_count=len(consumer_ids),
                    size_bytes=size_est.get(task_id, 0),
                    is_materialized=self._is_materialized(task),
                    can_spill=self._can_spill(task),
                )
                live_ranges[task_id] = live_range

            self._live_ranges = live_ranges
            _logger.info(f"Analyzed {len(live_ranges)} live ranges")

            return live_ranges

    def predict_peak_memory(
        self,
        dag: Any,
        *,
        size_estimates: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Predict peak memory usage during execution.

        Args:
            dag: PhysicalFactorDAG
            size_estimates: Optional size estimates per task

        Returns:
            Dict with peak_bytes, critical_path, spill_candidates
        """
        if not self._live_ranges:
            self.analyze_dag(dag, size_estimates=size_estimates)

        # Simulate execution in topological order
        live_at_step: dict[str, set[str]] = defaultdict(set)
        task_order = self._topological_order(dag)

        for i, task_id in enumerate(task_order):
            # Add this task's output to live set
            if task_id in self._live_ranges:
                live_at_step[i].add(task_id)

            # Carry forward previous live intermediates
            if i > 0:
                live_at_step[i].update(live_at_step[i - 1])

            # Remove intermediates whose last use just completed
            for intermediate_id in list(live_at_step[i]):
                lr = self._live_ranges.get(intermediate_id)
                if lr and lr.last_use == task_id:
                    live_at_step[i].discard(intermediate_id)

        # Find step with maximum live memory
        peak_bytes = 0
        peak_step = 0
        for step, live_ids in live_at_step.items():
            step_bytes = sum(
                self._live_ranges[tid].size_bytes
                for tid in live_ids
                if tid in self._live_ranges
            )
            if step_bytes > peak_bytes:
                peak_bytes = step_bytes
                peak_step = step

        # Identify spill candidates (large, long-lived)
        spill_candidates = []
        for task_id, lr in self._live_ranges.items():
            if lr.can_spill and lr.size_bytes > 50 * 1024 * 1024:  # > 50 MB
                live_duration = (
                    self._task_order(dag, lr.last_use)
                    - self._task_order(dag, lr.first_use)
                )
                if live_duration > 5:  # Lives across 5+ tasks
                    spill_candidates.append((task_id, lr.size_bytes, live_duration))

        spill_candidates.sort(key=lambda x: x[1] * x[2], reverse=True)

        return {
            "peak_bytes": peak_bytes,
            "peak_step": peak_step,
            "peak_task": task_order[peak_step] if peak_step < len(task_order) else None,
            "spill_candidates": [
                {"task_id": tid, "size_bytes": sz, "duration": dur}
                for tid, sz, dur in spill_candidates[:10]
            ],
            "total_intermediates": len(self._live_ranges),
        }

    def create_tracker(self) -> LivenessTracker:
        """Create runtime liveness tracker with current live ranges."""
        return LivenessTracker(self._live_ranges.copy())

    def _build_dependency_graph(self, dag: Any) -> dict[str, list[str]]:
        """Build task_id -> [dependency_ids] mapping."""
        deps = {}
        for task_id, task in self._iter_tasks(dag):
            task_deps = []
            for dep in getattr(task, "dependencies", []):
                task_deps.append(dep)
            deps[task_id] = task_deps
        return deps

    def _build_reverse_dependencies(
        self, deps: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Build reverse dependency map (task -> consumers)."""
        reverse = defaultdict(list)
        for task_id, dep_list in deps.items():
            for dep_id in dep_list:
                reverse[dep_id].append(task_id)
        return dict(reverse)

    def _iter_tasks(self, dag: Any) -> list[tuple[str, Any]]:
        """Iterate over all tasks in DAG."""
        try:
            tasks = getattr(dag, "tasks", {})
            return list(tasks.items())
        except Exception:
            return []

    def _topological_order(self, dag: Any) -> list[str]:
        """Get topological order of tasks."""
        try:
            # Try DAG's built-in ordering
            if hasattr(dag, "topological_order"):
                return dag.topological_order()

            # Manual topological sort
            deps = self._build_dependency_graph(dag)
            in_degree = {tid: len(dep_list) for tid, dep_list in deps.items()}
            queue = [tid for tid, deg in in_degree.items() if deg == 0]
            order = []

            while queue:
                task_id = queue.pop(0)
                order.append(task_id)

                # Reduce in-degree of consumers
                reverse_deps = self._build_reverse_dependencies(deps)
                for consumer in reverse_deps.get(task_id, []):
                    in_degree[consumer] -= 1
                    if in_degree[consumer] == 0:
                        queue.append(consumer)

            return order

        except Exception as exc:
            _logger.warning(f"Topological sort failed: {exc}")
            return []

    def _task_order(self, dag: Any, task_id: str) -> int:
        """Get topological position of task."""
        order = self._topological_order(dag)
        try:
            return order.index(task_id)
        except ValueError:
            return len(order)

    def _is_materialized(self, task: Any) -> bool:
        """Check if task result is materialized."""
        task_type = getattr(task, "task_type", "")
        return task_type in ("cse_shared", "materialized")

    def _can_spill(self, task: Any) -> bool:
        """Check if task result can be spilled to disk."""
        # Don't spill streaming results or small intermediates
        task_type = getattr(task, "task_type", "")
        if task_type in ("streaming", "source_scan"):
            return False
        return True


class LivenessTracker:
    """Runtime liveness tracker with refcount-based release.

    Tracks reference counts during execution and triggers cleanup
    when intermediates are no longer needed.
    """

    def __init__(self, live_ranges: dict[str, LiveRange]):
        """Initialize tracker with static live ranges.

        Args:
            live_ranges: Pre-computed live ranges from analysis
        """
        self._live_ranges = live_ranges
        self._state = LivenessState()
        self._lock = threading.RLock()
        self._cleanup_callbacks: dict[str, list[Any]] = defaultdict(list)

    def mark_produced(
        self,
        task_id: str,
        size_bytes: int,
    ) -> None:
        """Mark intermediate as produced and live.

        Args:
            task_id: Task ID that produced result
            size_bytes: Memory footprint
        """
        with self._lock:
            lr = self._live_ranges.get(task_id)
            if lr is None:
                # No consumers, can drop immediately
                return

            # Initialize refcount to consumer count
            refcount = lr.consumer_count
            self._state.live_intermediates[task_id] = (refcount, size_bytes)
            self._state.current_live_bytes += size_bytes

            if self._state.current_live_bytes > self._state.peak_live_bytes:
                self._state.peak_live_bytes = self._state.current_live_bytes

            _logger.debug(
                f"Produced {task_id}: refcount={refcount}, "
                f"{size_bytes // 1024} KB, "
                f"live={self._state.current_live_bytes // 1024 // 1024} MB"
            )

    def mark_consumed(
        self,
        task_id: str,
        consumer_id: str,
    ) -> bool:
        """Mark intermediate as consumed by consumer.

        Args:
            task_id: Intermediate task ID
            consumer_id: Consumer task ID

        Returns:
            True if this was the last use (should cleanup)
        """
        with self._lock:
            if task_id not in self._state.live_intermediates:
                return False

            refcount, size_bytes = self._state.live_intermediates[task_id]
            refcount -= 1

            if refcount <= 0:
                # Last use, cleanup
                del self._state.live_intermediates[task_id]
                self._state.current_live_bytes -= size_bytes
                self._state.released_count += 1

                _logger.debug(
                    f"Released {task_id}: {size_bytes // 1024} KB, "
                    f"live={self._state.current_live_bytes // 1024 // 1024} MB"
                )

                # Trigger cleanup callbacks
                for callback in self._cleanup_callbacks.get(task_id, []):
                    try:
                        callback()
                    except Exception as exc:
                        _logger.warning(f"Cleanup callback failed: {exc}")

                return True
            else:
                # Still has consumers
                self._state.live_intermediates[task_id] = (refcount, size_bytes)
                return False

    def register_cleanup_callback(
        self,
        task_id: str,
        callback: Any,
    ) -> None:
        """Register callback to run when intermediate is released.

        Args:
            task_id: Task ID to watch
            callback: Callable to invoke on cleanup
        """
        with self._lock:
            self._cleanup_callbacks[task_id].append(callback)

    def get_spill_candidates(
        self,
        min_size_bytes: int = 50 * 1024 * 1024,
    ) -> list[tuple[str, int]]:
        """Get currently live intermediates that should be spilled.

        Args:
            min_size_bytes: Minimum size to consider for spilling

        Returns:
            List of (task_id, size_bytes) sorted by size descending
        """
        with self._lock:
            candidates = []
            for task_id, (refcount, size_bytes) in self._state.live_intermediates.items():
                lr = self._live_ranges.get(task_id)
                if lr and lr.can_spill and size_bytes >= min_size_bytes:
                    candidates.append((task_id, size_bytes))

            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates

    def mark_spilled(self, task_id: str) -> None:
        """Mark intermediate as spilled to disk.

        Args:
            task_id: Task ID that was spilled
        """
        with self._lock:
            if task_id in self._state.live_intermediates:
                refcount, size_bytes = self._state.live_intermediates[task_id]
                # Keep refcount, but free memory
                self._state.live_intermediates[task_id] = (refcount, 0)
                self._state.current_live_bytes -= size_bytes
                self._state.spilled_count += 1

                _logger.info(
                    f"Spilled {task_id}: freed {size_bytes // 1024} KB, "
                    f"live={self._state.current_live_bytes // 1024 // 1024} MB"
                )

    def current_live_bytes(self) -> int:
        """Get current live memory."""
        with self._lock:
            return self._state.current_live_bytes

    def stats(self) -> dict[str, Any]:
        """Get liveness tracking statistics."""
        with self._lock:
            return {
                "current_live_bytes": self._state.current_live_bytes,
                "peak_live_bytes": self._state.peak_live_bytes,
                "live_intermediate_count": len(self._state.live_intermediates),
                "released_count": self._state.released_count,
                "spilled_count": self._state.spilled_count,
                "total_refcount": sum(
                    rc for rc, _ in self._state.live_intermediates.values()
                ),
            }
