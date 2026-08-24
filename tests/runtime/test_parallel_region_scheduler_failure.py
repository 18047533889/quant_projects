from __future__ import annotations

import pytest

from factor_engine.runtime.multibackend.parallel_region_scheduler import (
    ExecutionRegion,
    ParallelRegionScheduler,
    TypedRegionFailure,
)


def _region(
    region_id: str,
    dependencies: list[str] | None = None,
    estimated_cost_ms: float = 100.0,
) -> ExecutionRegion:
    """Helper to create a test ExecutionRegion."""
    return ExecutionRegion(
        region_id=region_id,
        backend="pandas_numpy",
        operators=[],
        dependencies=dependencies or [],
        estimated_cost_ms=estimated_cost_ms,
        memory_requirement_bytes=1024,
    )


def test_typed_region_failure_structure() -> None:
    """Test that TypedRegionFailure has proper structure and serialization."""
    failure = TypedRegionFailure(
        region_id="r1",
        error_type="ValueError",
        error_message="invalid input",
        exception=ValueError("invalid input"),
    )

    assert failure.region_id == "r1"
    assert failure.error_type == "ValueError"
    assert failure.error_message == "invalid input"
    assert failure.failed_dependencies == []

    d = failure.to_dict()
    assert d["region_id"] == "r1"
    assert d["error_type"] == "ValueError"
    assert "exception" not in d  # Exception is not serializable


def test_typed_region_failure_with_dependencies() -> None:
    """Test TypedRegionFailure with failed dependencies."""
    failure = TypedRegionFailure(
        region_id="r3",
        error_type="DependencyCancellation",
        error_message="Region cancelled due to failed dependencies: ['r1']",
        failed_dependencies=["r1"],
    )

    assert failure.failed_dependencies == ["r1"]
    d = failure.to_dict()
    assert d["failed_dependencies"] == ["r1"]


def test_dependency_chain_cancellation() -> None:
    """Test that failure in region A cancels dependent regions B and C."""
    scheduler = ParallelRegionScheduler(max_parallel_regions=2)

    # r1 fails, r2 depends on r1, r3 depends on r2
    regions = [
        _region("r1"),
        _region("r2", dependencies=["r1"]),
        _region("r3", dependencies=["r2"]),
    ]

    call_count = 0

    def execute_fn(region: ExecutionRegion) -> str:
        nonlocal call_count
        call_count += 1
        if region.region_id == "r1":
            raise RuntimeError("r1 failed")
        return f"result-{region.region_id}"

    result = scheduler.schedule_parallel(regions, execute_fn)

    # Only r1 should have been called (r2 and r3 cancelled)
    assert call_count == 1

    # r1 should have failed
    assert result["results"]["r1"]["success"] is False
    assert result["results"]["r1"]["failure"]["error_type"] == "RuntimeError"
    assert result["results"]["r1"]["failure"]["error_message"] == "r1 failed"

    # r2 should be cancelled due to dependency on r1
    assert result["results"]["r2"]["success"] is False
    assert result["results"]["r2"]["failure"]["error_type"] == "DependencyCancellation"
    assert result["results"]["r2"]["failure"]["failed_dependencies"] == ["r1"]

    # r3 should be cancelled due to dependency on r2
    assert result["results"]["r3"]["success"] is False
    assert result["results"]["r3"]["failure"]["error_type"] == "DependencyCancellation"
    assert result["results"]["r3"]["failure"]["failed_dependencies"] == ["r2"]

    # All regions should be in failed_regions
    assert set(result["failed_regions"]) == {"r1", "r2", "r3"}

    scheduler.shutdown()


def test_independent_region_failure_does_not_cancel_others() -> None:
    """Test that failure in one region does not affect independent regions."""
    scheduler = ParallelRegionScheduler(max_parallel_regions=2)

    # r1 and r2 are independent (no dependencies)
    regions = [_region("r1"), _region("r2")]

    def execute_fn(region: ExecutionRegion) -> str:
        if region.region_id == "r1":
            raise ValueError("r1 failed")
        return "success"

    result = scheduler.schedule_parallel(regions, execute_fn)

    # r1 should have failed
    assert result["results"]["r1"]["success"] is False
    assert result["results"]["r1"]["failure"]["error_type"] == "ValueError"

    # r2 should succeed (independent of r1)
    assert result["results"]["r2"]["success"] is True
    assert result["results"]["r2"]["result"] == "success"

    # Only r1 in failed_regions
    assert result["failed_regions"] == ["r1"]

    scheduler.shutdown()


def test_partial_dependency_chain_cancellation() -> None:
    """Test cancellation when only some dependencies fail."""
    scheduler = ParallelRegionScheduler(max_parallel_regions=2)

    # r1 fails, r2 succeeds, r3 depends on both r1 and r2
    regions = [
        _region("r1"),
        _region("r2"),
        _region("r3", dependencies=["r1", "r2"]),
    ]

    def execute_fn(region: ExecutionRegion) -> str:
        if region.region_id == "r1":
            raise RuntimeError("r1 failed")
        return f"result-{region.region_id}"

    result = scheduler.schedule_parallel(regions, execute_fn)

    # r1 should have failed
    assert result["results"]["r1"]["success"] is False

    # r2 should succeed (independent)
    assert result["results"]["r2"]["success"] is True
    assert result["results"]["r2"]["result"] == "result-r2"

    # r3 should be cancelled due to dependency on r1 (even though r2 succeeded)
    assert result["results"]["r3"]["success"] is False
    assert result["results"]["r3"]["failure"]["error_type"] == "DependencyCancellation"
    assert result["results"]["r3"]["failure"]["failed_dependencies"] == ["r1"]

    # r1 and r3 in failed_regions
    assert set(result["failed_regions"]) == {"r1", "r3"}

    scheduler.shutdown()


def test_success_result_structure() -> None:
    """Test successful result structure."""
    scheduler = ParallelRegionScheduler(max_parallel_regions=2)
    regions = [_region("r1")]

    result = scheduler.schedule_parallel(regions, lambda r: "success")

    assert result["results"]["r1"]["success"] is True
    assert result["results"]["r1"]["result"] == "success"
    assert result["failed_regions"] == []

    scheduler.shutdown()
