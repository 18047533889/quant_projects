# -*- coding: utf-8 -*-
"""R23 P0-5: ParallelRegionScheduler resource reservation + ancestor-fail
blocking + real timeout enforcement tests."""

from __future__ import annotations

import time
from typing import Any

import pytest

from factor_engine.runtime.multibackend.parallel_region_scheduler import (
    ExecutionRegion,
    ParallelRegionScheduler,
)


def _region(
    region_id: str,
    *,
    dependencies: list[str] | None = None,
    memory_bytes: int = 100,
    threads: int = 1,
    cost_ms: float = 1.0,
    deadline_ms: float | None = None,
) -> ExecutionRegion:
    return ExecutionRegion(
        region_id=region_id,
        backend="pandas",
        operators=[{"op": "literal"}],
        dependencies=dependencies or [],
        estimated_cost_ms=cost_ms,
        memory_requirement_bytes=memory_bytes,
        thread_requirement=threads,
        deadline_ms=deadline_ms,
    )


def _ok(region: ExecutionRegion) -> str:
    return f"ok:{region.region_id}"


# ---------------------------------------------------------------------------
# Resource reservation
# ---------------------------------------------------------------------------


def test_resource_reservation_admits_within_budget() -> None:
    sched = ParallelRegionScheduler(
        max_parallel_regions=4,
        memory_limit_bytes=1000,
        thread_limit=4,
    )
    regions = [
        _region("r1", memory_bytes=300, threads=2),
        _region("r2", memory_bytes=300, threads=2),
    ]
    out = sched.schedule_parallel(regions, _ok)
    assert out["results"]["r1"] == "ok:r1"
    assert out["results"]["r2"] == "ok:r2"
    assert out["failed"] == []


def test_resource_reservation_rejects_over_memory_budget() -> None:
    sched = ParallelRegionScheduler(
        max_parallel_regions=4,
        memory_limit_bytes=500,
        thread_limit=4,
    )
    regions = [
        _region("r1", memory_bytes=300),
        _region("r2", memory_bytes=300),
    ]
    out = sched.schedule_parallel(regions, _ok)
    # 500 budget, two 300-byte regions → exactly one admitted, one rejected.
    # Layer order is non-deterministic (set iteration), so assert count not identity.
    admitted = [rid for rid in ("r1", "r2") if out["results"][rid] == f"ok:{rid}"]
    rejected = [
        rid
        for rid in ("r1", "r2")
        if isinstance(out["results"][rid], dict)
        and "resource reservation rejected"
        in out["results"][rid].get("error", "")
    ]
    assert len(admitted) == 1
    assert len(rejected) == 1
    assert out["failed"] == rejected


def test_resource_reservation_rejects_over_thread_budget() -> None:
    sched = ParallelRegionScheduler(
        max_parallel_regions=4,
        memory_limit_bytes=1000,
        thread_limit=3,
    )
    regions = [
        _region("r1", threads=2),
        _region("r2", threads=2),
    ]
    out = sched.schedule_parallel(regions, _ok)
    # 3-thread budget, two 2-thread regions → exactly one admitted, one rejected.
    admitted = [rid for rid in ("r1", "r2") if out["results"][rid] == f"ok:{rid}"]
    rejected = [
        rid
        for rid in ("r1", "r2")
        if isinstance(out["results"][rid], dict)
        and "resource reservation rejected"
        in out["results"][rid].get("error", "")
    ]
    assert len(admitted) == 1
    assert len(rejected) == 1
    assert out["failed"] == rejected


def test_resource_reservation_no_limit_admits_all() -> None:
    sched = ParallelRegionScheduler(max_parallel_regions=4)
    regions = [
        _region("r1", memory_bytes=10**9, threads=100),
        _region("r2", memory_bytes=10**9, threads=100),
    ]
    out = sched.schedule_parallel(regions, _ok)
    assert out["results"]["r1"] == "ok:r1"
    assert out["results"]["r2"] == "ok:r2"
    assert out["failed"] == []


# ---------------------------------------------------------------------------
# Ancestor-fail blocking
# ---------------------------------------------------------------------------


def test_ancestor_failure_blocks_descendants_fail_closed() -> None:
    sched = ParallelRegionScheduler(max_parallel_regions=4)

    def execute(region: ExecutionRegion) -> str:
        if region.region_id == "root":
            raise RuntimeError("root boom")
        return _ok(region)

    regions = [
        _region("root"),
        _region("child", dependencies=["root"]),
        _region("grandchild", dependencies=["child"]),
    ]
    out = sched.schedule_parallel(regions, execute)
    assert "error" in out["results"]["root"]
    assert "ancestor region failed" in out["results"]["child"]["error"]
    assert "ancestor region failed" in out["results"]["grandchild"]["error"]
    assert out["failed"] == ["child", "grandchild", "root"]


def test_ancestor_failure_does_not_block_independent_region() -> None:
    sched = ParallelRegionScheduler(max_parallel_regions=4)

    def execute(region: ExecutionRegion) -> str:
        if region.region_id == "root":
            raise RuntimeError("root boom")
        return _ok(region)

    regions = [
        _region("root"),
        _region("independent"),  # no dependency on root
    ]
    out = sched.schedule_parallel(regions, execute)
    assert "error" in out["results"]["root"]
    assert out["results"]["independent"] == "ok:independent"


def test_ancestor_failure_blocks_descendant_across_layers() -> None:
    sched = ParallelRegionScheduler(max_parallel_regions=4)

    def execute(region: ExecutionRegion) -> str:
        if region.region_id == "a":
            raise RuntimeError("a boom")
        return _ok(region)

    # layer0: a, b ; layer1: c (dep a), d (dep b)
    regions = [
        _region("a"),
        _region("b"),
        _region("c", dependencies=["a"]),
        _region("d", dependencies=["b"]),
    ]
    out = sched.schedule_parallel(regions, execute)
    assert "error" in out["results"]["a"]
    assert out["results"]["b"] == "ok:b"
    assert "ancestor region failed" in out["results"]["c"]["error"]
    assert out["results"]["d"] == "ok:d"


# ---------------------------------------------------------------------------
# Real timeout enforcement
# ---------------------------------------------------------------------------


def test_region_timeout_enforced() -> None:
    sched = ParallelRegionScheduler(max_parallel_regions=4)

    def slow(region: ExecutionRegion) -> str:
        time.sleep(2.0)
        return _ok(region)

    regions = [_region("slow", deadline_ms=100)]
    start = time.monotonic()
    out = sched.schedule_parallel(regions, slow)
    elapsed = time.monotonic() - start

    assert elapsed < 1.5  # must not wait full 2s
    assert "error" in out["results"]["slow"]
    assert "TimeoutError" in out["results"]["slow"]["error"] or "timed out" in out["results"]["slow"]["error"]
    assert out["failed"] == ["slow"]


def test_region_without_deadline_waits_for_completion() -> None:
    sched = ParallelRegionScheduler(max_parallel_regions=4)

    def slow(region: ExecutionRegion) -> str:
        time.sleep(0.2)
        return _ok(region)

    regions = [_region("slow")]  # no deadline
    out = sched.schedule_parallel(regions, slow)
    assert out["results"]["slow"] == "ok:slow"
    assert out["failed"] == []


def test_timeout_failure_blocks_descendants() -> None:
    sched = ParallelRegionScheduler(max_parallel_regions=4)

    def slow(region: ExecutionRegion) -> str:
        if region.region_id == "root":
            time.sleep(2.0)
        return _ok(region)

    regions = [
        _region("root", deadline_ms=100),
        _region("child", dependencies=["root"]),
    ]
    out = sched.schedule_parallel(regions, slow)
    assert "error" in out["results"]["root"]
    assert "ancestor region failed" in out["results"]["child"]["error"]


# ---------------------------------------------------------------------------
# Metrics / summary
# ---------------------------------------------------------------------------


def test_metrics_and_summary_reflect_limits() -> None:
    sched = ParallelRegionScheduler(
        max_parallel_regions=2,
        memory_limit_bytes=500,
        thread_limit=2,
    )
    regions = [_region("r1"), _region("r2")]
    sched.schedule_parallel(regions, _ok)
    summary = sched.summary()
    assert summary["max_parallel_regions"] == 2
    assert summary["memory_limit_bytes"] == 500
    assert summary["thread_limit"] == 2
    assert summary["metrics"]["total_regions_scheduled"] == 2
