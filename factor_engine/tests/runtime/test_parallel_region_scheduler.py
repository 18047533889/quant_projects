# -*- coding: utf-8 -*-
"""R23 P0-5: ParallelRegionScheduler resource reservation + ancestor-fail
blocking + real timeout enforcement tests."""

from __future__ import annotations

import time
import threading
from typing import Any

import pytest

from factor_engine.runtime.multibackend.parallel_region_scheduler import (
    ExecutionRegion,
    ParallelRegionScheduler,
    ResourceAdmissionError,
)
from factor_engine.runtime.resource_broker import ResourceBroker


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
        cpu_tokens=threads,
        deadline_ms=(time.monotonic() * 1000.0 + deadline_ms
                     if deadline_ms is not None else None),
    )


def _ok(region: ExecutionRegion) -> str:
    return f"ok:{region.region_id}"


def _broker(*, memory_bytes: int, cpu_slots: int) -> ResourceBroker:
    return ResourceBroker(
        hard_memory_limit=memory_bytes,
        cpu_slots=cpu_slots,
        min_host_reserve_gb=0.0,
        min_host_reserve_fraction=0.0,
    )


# ---------------------------------------------------------------------------
# Resource reservation
# ---------------------------------------------------------------------------


def test_resource_reservation_admits_within_budget() -> None:
    broker = _broker(memory_bytes=64 * 1024**3, cpu_slots=4)
    sched = ParallelRegionScheduler(max_parallel_regions=4, resource_broker=broker)
    regions = [
        _region("r1", memory_bytes=300 * 1024**2, threads=2),
        _region("r2", memory_bytes=300 * 1024**2, threads=2),
    ]
    out = sched.schedule_parallel(regions, _ok)
    assert out["results"]["r1"]["result"] == "ok:r1"
    assert out["results"]["r2"]["result"] == "ok:r2"
    assert out["failed_regions"] == []
    assert broker.summary()["running_tasks"] == 0


def test_resource_reservation_rejects_over_memory_budget() -> None:
    broker = _broker(memory_bytes=64 * 1024**2, cpu_slots=4)
    sched = ParallelRegionScheduler(max_parallel_regions=4, resource_broker=broker)
    with pytest.raises(ResourceAdmissionError):
        sched.schedule_parallel([_region("r1", memory_bytes=1024**3)], _ok)
    assert broker.summary()["running_tasks"] == 0


def test_resource_reservation_rejects_over_thread_budget() -> None:
    broker = _broker(memory_bytes=64 * 1024**3, cpu_slots=3)
    sched = ParallelRegionScheduler(max_parallel_regions=4, resource_broker=broker)
    regions = [
        _region("r1", threads=2),
        _region("r2", threads=2),
    ]
    active = 0
    peak = 0
    lock = threading.Lock()

    def execute(region):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return _ok(region)

    out = sched.schedule_parallel(regions, execute)
    assert all(out["results"][rid]["success"] for rid in ("r1", "r2"))
    assert peak == 1
    assert broker.summary()["running_tasks"] == 0


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, float("nan"), float("inf")])
def test_cpu_tokens_must_be_a_positive_integer(invalid) -> None:
    broker = _broker(memory_bytes=64 * 1024**3, cpu_slots=4)
    sched = ParallelRegionScheduler(max_parallel_regions=2, resource_broker=broker)
    region = _region("invalid")
    region.cpu_tokens = invalid

    with pytest.raises(ValueError, match="positive integer"):
        sched.schedule_parallel([region], _ok)
    assert broker.summary()["running_tasks"] == 0


def test_submit_failure_releases_just_acquired_lease() -> None:
    broker = _broker(memory_bytes=64 * 1024**3, cpu_slots=2)
    sched = ParallelRegionScheduler(max_parallel_regions=2, resource_broker=broker)
    sched._executor.shutdown(wait=True)

    class RaisingExecutor:
        def submit(self, *_args, **_kwargs):
            raise RuntimeError("submit failed")

    sched._executor = RaisingExecutor()
    with pytest.raises(RuntimeError, match="submit failed"):
        sched.schedule_parallel([_region("r1")], _ok)
    assert broker.summary()["running_tasks"] == 0


def test_submit_failure_keeps_existing_worker_charged_until_completion() -> None:
    broker = _broker(memory_bytes=64 * 1024**3, cpu_slots=2)
    sched = ParallelRegionScheduler(max_parallel_regions=2, resource_broker=broker)
    delegate = sched._executor
    release_worker = threading.Event()
    submit_count = 0

    class FailSecondSubmit:
        def submit(self, fn, region):
            nonlocal submit_count
            submit_count += 1
            if submit_count == 2:
                raise RuntimeError("second submit failed")
            return delegate.submit(fn, region)

    def blocked(_region):
        release_worker.wait(timeout=1.0)
        return "done"

    sched._executor = FailSecondSubmit()
    with pytest.raises(RuntimeError, match="second submit failed"):
        sched.schedule_parallel([_region("r1"), _region("r2")], blocked)
    assert broker.summary()["running_tasks"] == 1
    release_worker.set()
    deadline = time.monotonic() + 1.0
    while broker.summary()["running_tasks"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert broker.summary()["running_tasks"] == 0


def test_resource_reservation_no_limit_admits_all() -> None:
    sched = ParallelRegionScheduler(max_parallel_regions=4)
    regions = [
        _region("r1", memory_bytes=10**9, threads=100),
        _region("r2", memory_bytes=10**9, threads=100),
    ]
    out = sched.schedule_parallel(regions, _ok)
    assert out["results"]["r1"]["result"] == "ok:r1"
    assert out["results"]["r2"]["result"] == "ok:r2"
    assert out["failed_regions"] == []


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
    assert out["results"]["root"]["success"] is False
    assert out["results"]["child"]["failure"]["error_type"] == "DependencyCancellation"
    assert out["results"]["grandchild"]["failure"]["error_type"] == "DependencyCancellation"
    assert sorted(out["failed_regions"]) == ["child", "grandchild", "root"]


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
    assert out["results"]["root"]["success"] is False
    assert out["results"]["independent"]["result"] == "ok:independent"


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
    assert out["results"]["a"]["success"] is False
    assert out["results"]["b"]["result"] == "ok:b"
    assert out["results"]["c"]["failure"]["error_type"] == "DependencyCancellation"
    assert out["results"]["d"]["result"] == "ok:d"


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
    assert out["results"]["slow"]["success"] is False
    assert out["results"]["slow"]["failure"]["error_type"] == "DeadlineExceeded"
    assert out["failed_regions"] == ["slow"]


def test_region_without_deadline_waits_for_completion() -> None:
    sched = ParallelRegionScheduler(max_parallel_regions=4)

    def slow(region: ExecutionRegion) -> str:
        time.sleep(0.2)
        return _ok(region)

    regions = [_region("slow")]  # no deadline
    out = sched.schedule_parallel(regions, slow)
    assert out["results"]["slow"]["result"] == "ok:slow"
    assert out["failed_regions"] == []


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
    assert out["results"]["root"]["failure"]["error_type"] == "DeadlineExceeded"
    assert out["results"]["child"]["failure"]["error_type"] == "DependencyCancellation"


def test_timeout_keeps_broker_lease_until_running_thread_exits() -> None:
    broker = _broker(memory_bytes=64 * 1024**3, cpu_slots=2)
    sched = ParallelRegionScheduler(max_parallel_regions=2, resource_broker=broker)
    release_worker = threading.Event()

    def blocked(_region: ExecutionRegion) -> str:
        release_worker.wait(timeout=1.0)
        return "done"

    out = sched.schedule_parallel([_region("slow", deadline_ms=30)], blocked)
    assert out["results"]["slow"]["failure"]["error_type"] == "DeadlineExceeded"
    assert broker.summary()["running_tasks"] == 1
    release_worker.set()
    deadline = time.monotonic() + 1.0
    while broker.summary()["running_tasks"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert broker.summary()["running_tasks"] == 0


def test_timeout_drains_before_resource_blocked_independent_region() -> None:
    broker = _broker(memory_bytes=64 * 1024**3, cpu_slots=1)
    sched = ParallelRegionScheduler(max_parallel_regions=2, resource_broker=broker)

    def execute(region: ExecutionRegion) -> str:
        if region.region_id == "slow":
            time.sleep(0.15)
        return _ok(region)

    out = sched.schedule_parallel(
        [_region("slow", deadline_ms=30), _region("independent")],
        execute,
    )

    assert out["results"]["slow"]["failure"]["error_type"] == "DeadlineExceeded"
    assert out["results"]["independent"]["result"] == "ok:independent"
    assert broker.summary()["running_tasks"] == 0


# ---------------------------------------------------------------------------
# Metrics / summary
# ---------------------------------------------------------------------------


def test_metrics_and_summary_reflect_limits() -> None:
    broker = _broker(memory_bytes=64 * 1024**3, cpu_slots=2)
    sched = ParallelRegionScheduler(max_parallel_regions=2, resource_broker=broker)
    regions = [_region("r1"), _region("r2")]
    sched.schedule_parallel(regions, _ok)
    summary = sched.summary()
    assert summary["max_parallel_regions"] == 2
    assert summary["metrics"]["total_regions_scheduled"] == 2
    broker_summary = broker.summary()
    assert broker_summary["hard_memory_limit"] == 64 * 1024**3
    assert broker_summary["hard_cpu_slots"] == 2
