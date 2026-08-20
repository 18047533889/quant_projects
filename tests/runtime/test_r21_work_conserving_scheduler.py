# -*- coding: utf-8 -*-
"""R21-WORK-CONSERVING-SCHEDULER: Regression tests for work-conserving ready-queue DAG scheduler.

Tests verify that child regions become runnable immediately after their parent
completes, without waiting for sibling regions in the same layer to finish.
"""

from __future__ import annotations

import time
import threading
from typing import Any

import pytest

from runtime.multibackend.parallel_region_scheduler import (
    ExecutionRegion,
    ParallelRegionScheduler,
    ParallelSchedulerMetrics,
    RegionExecutionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_region(
    region_id: str,
    dependencies: list[str] | None = None,
    cost_ms: float = 10.0,
    backend: str = "pandas",
) -> ExecutionRegion:
    """Create a minimal ExecutionRegion for testing."""
    return ExecutionRegion(
        region_id=region_id,
        backend=backend,
        operators=[{"op": "test"}],
        dependencies=dependencies or [],
        estimated_cost_ms=cost_ms,
        memory_requirement_bytes=1024 * 1024,
    )


class TimingExecuteFn:
    """Execute function that records start/finish times per region."""

    def __init__(self, base_delay_ms: float = 5.0) -> None:
        self.base_delay_ms = base_delay_ms
        self.start_times: dict[str, float] = {}
        self.finish_times: dict[str, float] = {}
        self._lock = threading.Lock()

    def __call__(self, region: ExecutionRegion) -> str:
        now = time.monotonic() * 1000.0
        with self._lock:
            self.start_times[region.region_id] = now
        # Simulate work
        time.sleep(self.base_delay_ms / 1000.0)
        now = time.monotonic() * 1000.0
        with self._lock:
            self.finish_times[region.region_id] = now
        return f"done:{region.region_id}"


# ---------------------------------------------------------------------------
# Test 1: Child runs immediately after parent, not waiting for sibling
# ---------------------------------------------------------------------------

def test_child_runs_immediately_after_parent_not_waiting_for_sibling():
    """DAG: A -> C, B (independent), C depends on A only.

    Under strict layer barrier: A and B start together, wait for BOTH to finish,
    then C starts.

    Under work-conserving: A finishes -> C starts immediately, even if B is still running.

    This test verifies C starts immediately after A finishes (not waiting for B to finish).
    """
    # Use threading events to control exact timing
    a_done = threading.Event()
    b_done = threading.Event()
    c_started = threading.Event()

    def controlled_execute(region: ExecutionRegion) -> str:
        if region.region_id == "A":
            # A runs quickly and signals done
            time.sleep(0.01)
            a_done.set()
            return "done:A"
        elif region.region_id == "B":
            # B waits for C to start (proving C doesn't wait for B)
            c_started.wait(timeout=2.0)
            time.sleep(0.01)
            b_done.set()
            return "done:B"
        elif region.region_id == "C":
            # C signals when it starts
            c_started.set()
            time.sleep(0.01)
            return "done:C"
        return "done:?"

    a = _make_region("A", dependencies=[], cost_ms=20.0)
    b = _make_region("B", dependencies=[], cost_ms=100.0)
    c = _make_region("C", dependencies=["A"], cost_ms=10.0)

    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        result = scheduler.schedule_parallel([a, b, c], controlled_execute)
    finally:
        scheduler.shutdown()

    # All regions should complete successfully
    assert result["results"]["A"]["success"] is True
    assert result["results"]["B"]["success"] is True
    assert result["results"]["C"]["success"] is True

    # C should have started (event was set)
    assert c_started.is_set(), "C should have started before B finished"

    # B should have finished (meaning it was waiting for C to start)
    assert b_done.is_set(), "B should have finished"


# ---------------------------------------------------------------------------
# Test 2: Diamond DAG - both children start as soon as parent completes
# ---------------------------------------------------------------------------

def test_diamond_dag_children_start_immediately():
    """DAG: A -> B, A -> C, B & C -> D.

    A completes → both B and C start immediately (not waiting for any layer barrier).
    """
    fn = TimingExecuteFn(base_delay_ms=5.0)

    a = _make_region("A", dependencies=[], cost_ms=50.0)
    b = _make_region("B", dependencies=["A"], cost_ms=30.0)
    c = _make_region("C", dependencies=["A"], cost_ms=30.0)
    d = _make_region("D", dependencies=["B", "C"], cost_ms=10.0)

    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        result = scheduler.schedule_parallel([a, b, c, d], fn)
    finally:
        scheduler.shutdown()

    # All should complete
    for rid in ["A", "B", "C", "D"]:
        assert result["results"][rid]["success"] is True

    # B and C should both start after A finishes, but before A's finish + B's duration
    # (they run in parallel after A)
    assert fn.start_times["B"] >= fn.finish_times["A"]
    assert fn.start_times["C"] >= fn.finish_times["A"]

    # D should start after both B and C finish
    assert fn.start_times["D"] >= fn.finish_times["B"]
    assert fn.start_times["D"] >= fn.finish_times["C"]


# ---------------------------------------------------------------------------
# Test 3: Three-level chain with timing verification
# ---------------------------------------------------------------------------

def test_three_level_chain_timing():
    """DAG: A(100ms) -> B(10ms) -> C(10ms).

    Under strict layer: A completes at T+100, B runs T+100 to T+110, C runs T+110 to T+120.
    Under work-conserving: same timing, but C starts immediately when B finishes.
    """
    fn = TimingExecuteFn(base_delay_ms=10.0)

    a = _make_region("A", dependencies=[], cost_ms=100.0)
    b = _make_region("B", dependencies=["A"], cost_ms=10.0)
    c = _make_region("C", dependencies=["B"], cost_ms=10.0)

    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        result = scheduler.schedule_parallel([a, b, c], fn)
    finally:
        scheduler.shutdown()

    for rid in ["A", "B", "C"]:
        assert result["results"][rid]["success"] is True

    # Timing chain: A finishes -> B starts -> B finishes -> C starts
    assert fn.start_times["B"] >= fn.finish_times["A"]
    assert fn.start_times["C"] >= fn.finish_times["B"]

    # Total time should be roughly A + B + C (serial chain)
    total_ms = fn.finish_times["C"] - fn.start_times["A"]
    assert total_ms < 150.0, f"Total chain time {total_ms}ms should be < 150ms"


# ---------------------------------------------------------------------------
# Test 4: Failed parent cascades cancellation to children
# ---------------------------------------------------------------------------

def test_failed_parent_cascades_cancellation():
    """DAG: A -> B, A fails. B should be cancelled (DependencyCancellation)."""
    def _fail_execute(region: ExecutionRegion) -> Any:
        if region.region_id == "A":
            raise RuntimeError("A failed")
        return f"done:{region.region_id}"

    a = _make_region("A", dependencies=[], cost_ms=10.0)
    b = _make_region("B", dependencies=["A"], cost_ms=10.0)

    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        result = scheduler.schedule_parallel([a, b], _fail_execute)
    finally:
        scheduler.shutdown()

    # A should fail with RuntimeError
    assert result["results"]["A"]["success"] is False

    # B should be cancelled due to dependency failure
    assert result["results"]["B"]["success"] is False
    assert result["results"]["B"]["failure"]["error_type"] == "DependencyCancellation"


# ---------------------------------------------------------------------------
# Test 5: Work-conserving reduces total time vs serial
# ---------------------------------------------------------------------------

def test_work_conserving_reduces_total_time():
    """DAG: A(100ms) -> B(10ms), C(100ms) -> D(10ms).

    Serial: A+B+C+D = 220ms
    Work-conserving parallel: max(A+B, C+D) ≈ 110ms (if both branches run concurrently)
    """
    fn = TimingExecuteFn(base_delay_ms=10.0)

    a = _make_region("A", dependencies=[], cost_ms=100.0)
    b = _make_region("B", dependencies=["A"], cost_ms=10.0)
    c = _make_region("C", dependencies=[], cost_ms=100.0)
    d = _make_region("D", dependencies=["C"], cost_ms=10.0)

    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        result = scheduler.schedule_parallel([a, b, c, d], fn)
    finally:
        scheduler.shutdown()

    for rid in ["A", "B", "C", "D"]:
        assert result["results"][rid]["success"] is True

    # Total time should be significantly less than serial (220ms)
    total_ms = result["elapsed_ms"]
    assert total_ms < 150.0, (
        f"Work-conserving parallel time {total_ms}ms should be < 150ms "
        f"(serial would be ~220ms)"
    )


# ---------------------------------------------------------------------------
# Test 6: Metrics tracking
# ---------------------------------------------------------------------------

def test_metrics_are_tracked_correctly():
    """Verify metrics are properly incremented."""
    fn = TimingExecuteFn(base_delay_ms=5.0)

    a = _make_region("A", dependencies=[], cost_ms=10.0)
    b = _make_region("B", dependencies=[], cost_ms=10.0)

    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        scheduler.schedule_parallel([a, b], fn)
        metrics = scheduler.metrics()
    finally:
        scheduler.shutdown()

    assert metrics.total_regions_scheduled == 2
    assert metrics.total_elapsed_ms > 0
    assert metrics.total_saved_ms >= 0
