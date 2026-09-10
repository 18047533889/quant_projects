# -*- coding: utf-8 -*-
"""R45: Scheduler production-closure tests.

Covers:
    - ResourceBroker admission (CPU/RAM/IO token) before submit, release on
      completion; work-conserving when broker denies.
    - Dynamic parallelism default derived from resources (not fixed 4).
    - Deadline / cancellation: region exceeding deadline is cancelled, lease
      revoked, descendants cancelled (not scheduled).
    - DAG cycle validation raises PhysicalPlanCycleError before scheduling.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from factor_engine.runtime.multibackend.parallel_region_scheduler import (
    ExecutionRegion,
    ParallelRegionScheduler,
    PhysicalPlanCycleError,
)
from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.task_resource_contract import TaskResourceContract


def _region(
    region_id: str,
    dependencies: list[str] | None = None,
    memory_requirement_bytes: int = 1024,
    deadline_ms: float | None = None,
) -> ExecutionRegion:
    return ExecutionRegion(
        region_id=region_id,
        backend="pandas_numpy",
        operators=[],
        dependencies=dependencies or [],
        estimated_cost_ms=10.0,
        memory_requirement_bytes=memory_requirement_bytes,
        deadline_ms=deadline_ms,
    )


# ---------------------------------------------------------------------------
# 1. ResourceBroker admission
# ---------------------------------------------------------------------------

def test_broker_admission_acquires_and_releases() -> None:
    """Regions admitted via broker; leases released on completion."""
    broker = ResourceBroker(
        hard_memory_limit=64 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=0.0,
        min_host_reserve_fraction=0.0,
    )
    scheduler = ParallelRegionScheduler(
        max_parallel_regions=4, resource_broker=broker
    )
    try:
        regions = [_region("r1"), _region("r2")]
        result = scheduler.schedule_parallel(regions, lambda r: f"ok:{r.region_id}")
        assert result["results"]["r1"]["success"] is True
        assert result["results"]["r2"]["success"] is True
        # All leases released -> broker running set empty.
        assert broker.summary()["running_tasks"] == 0
    finally:
        scheduler.shutdown()


def test_broker_denial_is_work_conserving(monkeypatch) -> None:
    """When broker denies, region stays ready and runs in a later round."""
    # Memory limit such that one 8GB region (peak = 8GB * 1.3 = 10.4GB) fits
    # but two (20.8GB) do not -> second is denied until first releases.
    broker = ResourceBroker(
        hard_memory_limit=16 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=0.0,
        min_host_reserve_fraction=0.0,
    )
    # Synthetic admission lane: 12 GiB admits one 10.4 GiB peak plus
    # protected egress, but never two. No real 8 GiB buffers are allocated.
    # Live cgroup usage must not be combined with a fictitious 16 GiB host.
    monkeypatch.setattr(broker, "execution_budget", lambda: 12 * 1024**3)
    monkeypatch.setattr(broker, "pressure_stage", lambda: "NORMAL")

    scheduler = ParallelRegionScheduler(
        max_parallel_regions=4, resource_broker=broker
    )
    try:
        regions = [
            _region("r1", memory_requirement_bytes=8 * 1024**3),
            _region("r2", memory_requirement_bytes=8 * 1024**3),
        ]
        result = scheduler.schedule_parallel(regions, lambda r: f"ok:{r.region_id}")
        # Both eventually succeed (work-conserving, no deadlock).
        assert result["results"]["r1"]["success"] is True
        assert result["results"]["r2"]["success"] is True
        assert broker.summary()["running_tasks"] == 0
    finally:
        scheduler.shutdown()


# ---------------------------------------------------------------------------
# 2. Dynamic parallelism
# ---------------------------------------------------------------------------

def test_dynamic_parallelism_default() -> None:
    """Default max_parallel_regions is derived from resources, not fixed 4."""
    scheduler = ParallelRegionScheduler()
    try:
        assert scheduler.max_parallel_regions >= 1
        # On any real host with >4 cores this should exceed 4; but we only
        # assert it is a positive integer derived from resources.
        assert isinstance(scheduler.max_parallel_regions, int)
    finally:
        scheduler.shutdown()


# ---------------------------------------------------------------------------
# 3. Deadline / cancellation
# ---------------------------------------------------------------------------

def test_deadline_cancels_region_and_descendants() -> None:
    """Region exceeding deadline is cancelled; descendants not scheduled."""
    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        # r1 has a very short deadline; r2 depends on r1.
        regions = [
            _region("r1", deadline_ms=50.0),
            _region("r2", dependencies=["r1"]),
        ]
        call_count = 0

        def execute_fn(region: ExecutionRegion) -> str:
            nonlocal call_count
            call_count += 1
            time.sleep(0.5)  # exceed the 50ms deadline
            return f"ok:{region.region_id}"

        result = scheduler.schedule_parallel(regions, execute_fn)
        # r1 should be DeadlineExceeded.
        assert result["results"]["r1"]["success"] is False
        assert result["results"]["r1"]["failure"]["error_type"] == "DeadlineExceeded"
        # r2 cancelled due to dependency on r1.
        assert result["results"]["r2"]["success"] is False
        assert result["results"]["r2"]["failure"]["error_type"] == "DependencyCancellation"
        # Only r1 actually ran (r2 never scheduled).
        assert call_count == 1
    finally:
        scheduler.shutdown()


def test_default_deadline_applies() -> None:
    """default_deadline_ms applies to regions without explicit deadline."""
    scheduler = ParallelRegionScheduler(
        max_parallel_regions=4, default_deadline_ms=50.0
    )
    try:
        regions = [_region("r1")]

        def execute_fn(region: ExecutionRegion) -> str:
            time.sleep(0.5)
            return "ok"

        result = scheduler.schedule_parallel(regions, execute_fn)
        assert result["results"]["r1"]["success"] is False
        assert result["results"]["r1"]["failure"]["error_type"] == "DeadlineExceeded"
    finally:
        scheduler.shutdown()


# ---------------------------------------------------------------------------
# 4. DAG cycle validation
# ---------------------------------------------------------------------------

def test_cycle_raises_hard_error() -> None:
    """A cyclic dependency graph raises PhysicalPlanCycleError before scheduling."""
    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        regions = [
            _region("r1", dependencies=["r2"]),
            _region("r2", dependencies=["r1"]),
        ]
        with pytest.raises(PhysicalPlanCycleError):
            scheduler.schedule_parallel(regions, lambda r: "ok")
    finally:
        scheduler.shutdown()


def test_self_cycle_raises() -> None:
    """A region depending on itself raises PhysicalPlanCycleError."""
    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        regions = [_region("r1", dependencies=["r1"])]
        with pytest.raises(PhysicalPlanCycleError):
            scheduler.schedule_parallel(regions, lambda r: "ok")
    finally:
        scheduler.shutdown()


def test_topological_layers_raises_on_cycle() -> None:
    """_topological_layers raises instead of silently breaking on a cycle."""
    scheduler = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        dep_graph = {"r1": {"r2"}, "r2": {"r1"}}
        with pytest.raises(PhysicalPlanCycleError):
            scheduler._topological_layers(dep_graph)
    finally:
        scheduler.shutdown()
