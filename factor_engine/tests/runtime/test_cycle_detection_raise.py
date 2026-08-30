# -*- coding: utf-8 -*-
"""R21-CYCLE-DETECTION-RAISE: Regression tests for PhysicalPlanCycleError.

Tests that cycle detection in _topological_layers raises PhysicalPlanCycleError
instead of silently breaking (log + break was the old behavior).
"""

import pytest

from factor_engine.runtime.multibackend.parallel_region_scheduler import (
    ExecutionRegion,
    ParallelRegionScheduler,
    PhysicalPlanCycleError,
)


@pytest.fixture
def scheduler():
    """Create a fresh ParallelRegionScheduler."""
    s = ParallelRegionScheduler(max_parallel_regions=4)
    try:
        yield s
    finally:
        s.shutdown()


class TestCycleDetectionRaise:
    """Verify that cyclic dependencies raise PhysicalPlanCycleError (fail closed)."""

    def test_simple_two_node_cycle(self, scheduler):
        """A -> B -> A is a cycle and must raise."""
        regions = [
            ExecutionRegion(
                region_id="A",
                backend="pandas",
                operators=[],
                dependencies=["B"],
                estimated_cost_ms=100,
                memory_requirement_bytes=1024,
            ),
            ExecutionRegion(
                region_id="B",
                backend="pandas",
                operators=[],
                dependencies=["A"],
                estimated_cost_ms=100,
                memory_requirement_bytes=1024,
            ),
        ]
        with pytest.raises(PhysicalPlanCycleError, match="A"):
            scheduler.schedule_parallel(regions, execute_fn=lambda r: r.to_dict())

    def test_three_node_cycle(self, scheduler):
        """A -> B -> C -> A is a cycle and must raise."""
        regions = [
            ExecutionRegion(
                region_id="A",
                backend="duckdb",
                operators=[],
                dependencies=["C"],
                estimated_cost_ms=50,
                memory_requirement_bytes=512,
            ),
            ExecutionRegion(
                region_id="B",
                backend="duckdb",
                operators=[],
                dependencies=["A"],
                estimated_cost_ms=50,
                memory_requirement_bytes=512,
            ),
            ExecutionRegion(
                region_id="C",
                backend="duckdb",
                operators=[],
                dependencies=["B"],
                estimated_cost_ms=50,
                memory_requirement_bytes=512,
            ),
        ]
        with pytest.raises(PhysicalPlanCycleError, match="A"):
            scheduler.schedule_parallel(regions, execute_fn=lambda r: r.to_dict())

    def test_self_cycle(self, scheduler):
        """A -> A is a self-cycle and must raise."""
        regions = [
            ExecutionRegion(
                region_id="A",
                backend="pandas",
                operators=[],
                dependencies=["A"],
                estimated_cost_ms=10,
                memory_requirement_bytes=256,
            ),
        ]
        with pytest.raises(PhysicalPlanCycleError, match="A"):
            scheduler.schedule_parallel(regions, execute_fn=lambda r: r.to_dict())
