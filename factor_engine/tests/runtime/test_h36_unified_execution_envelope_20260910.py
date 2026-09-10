from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest

from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.resource_errors import ResourceBudgetExceeded
from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
from factor_engine.runtime.task_resource_contract import TaskResourceContract


def _broker(execution_budget: int) -> tuple[ResourceBroker, object]:
    hard = 8 * 1024**3
    broker = ResourceBroker(hard_memory_limit=hard, cpu_slots=4)
    snap = replace(
        broker.snapshot(), hard_memory_limit=hard,
        cgroup_memory_current=0, host_mem_available=hard,
        process_rss=0, process_family_rss=0, process_family_pss=0,
        host_mem_available_known=True, spill_free_bytes=3 * 1024**3,
    )
    broker._refresh = lambda force=False: snap
    broker.execution_budget = lambda: execution_budget
    broker.pressure_stage = lambda: "NORMAL"
    return broker, snap


def test_controller_envelope_uses_one_execution_budget_and_keeps_other_fields() -> None:
    budget = 2 * 1024**3
    broker, snap = _broker(budget)
    env = broker._envelope_from_snapshot(snap)
    assert env.safe_memory_bytes == budget
    assert env.safe_memory_bytes <= broker.execution_budget()
    assert env.hard_memory_bytes == broker.hard_memory_limit
    assert env.hard_cpu_tokens == broker.hard_cpu_slots
    assert env.target_cpu_tokens == broker.cpu_budget()
    assert env.spill_free_bytes == snap.spill_free_bytes

    decision = broker._resource_controller().tick(
        broker._signals_from_snapshot(snap), env,
    )
    allocated = (
        decision.read_wave_bytes + decision.factor_block_bytes
        + decision.result_queue_bytes + decision.cache_budget_bytes
    )
    assert 0 < decision.read_wave_bytes
    assert allocated <= budget


def test_known_zero_stays_zero_while_job_cap_and_pressure_remain_effective() -> None:
    broker, snap = _broker(0)
    zero = broker._resource_controller().tick(
        broker._signals_from_snapshot(snap), broker._envelope_from_snapshot(snap),
    )
    assert (
        zero.read_wave_bytes, zero.factor_block_bytes,
        zero.result_queue_bytes, zero.cache_budget_bytes,
    ) == (0, 0, 0, 0)

    broker, snap = _broker(2 * 1024**3)
    env = broker._envelope_from_snapshot(snap)
    cap = 128 * 1024**2
    constrained = broker._resource_controller().tick(
        broker._signals_from_snapshot(snap), env,
        job_memory_lease_bytes=cap, sink_backpressure=0.95,
    )
    allocated = (
        constrained.read_wave_bytes + constrained.factor_block_bytes
        + constrained.result_queue_bytes + constrained.cache_budget_bytes
    )
    assert allocated <= cap
    assert constrained.pressure_state != "NORMAL"


def test_dynamic_wave_unknown_and_zero_do_not_resurrect_four_gib() -> None:
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    failing = SimpleNamespace(
        resource_decision=lambda **_: (_ for _ in ()).throw(RuntimeError("unknown")),
        current_read_budget=lambda: (_ for _ in ()).throw(RuntimeError("unknown")),
    )
    scheduler = object.__new__(AdaptiveBatchScheduler)
    scheduler._job_memory_cap = lambda: None
    scheduler.broker = failing
    assert scheduler._dynamic_wave_budget() == 0

    scheduler.broker = SimpleNamespace(
        resource_decision=lambda **_: SimpleNamespace(read_wave_bytes=0),
        current_read_budget=lambda: 123,
    )
    assert scheduler._dynamic_wave_budget() == 0


def test_zero_wave_rejects_positive_atomic_request_but_allows_empty_plan() -> None:
    from factor_engine.planner.read_wave_planner import ReadWavePlanner

    planner = ReadWavePlanner(wave_memory_budget=0, rows_estimate=1)
    assert planner.plan().waves == []
    planner.register_scan_task(
        "scan", dataset="d", source_scope="s", snapshot_id="snap",
        time_range=None, columns=("close",), estimated_memory_bytes=8,
    )
    with pytest.raises(ResourceBudgetExceeded, match="atomic read request"):
        planner.plan()


def test_task_peaks_and_physical_leases_share_one_commitment_limit() -> None:
    gib = 1024**3
    broker, _snap = _broker(8 * gib)
    read = broker.acquire_memory(MemoryLeaseKind.SOURCE_READ, 4 * gib)
    assert read is not None
    task = TaskResourceContract(
        peak_memory_bytes=4 * gib, uncertainty=1.0,
        cpu_tokens=1, io_tokens=0, backend="pandas_numpy",
    )
    assert not broker.can_admit(task)
    assert broker.try_reserve(task, task_id="overlap") is None
    read.release()
    reservation = broker.try_reserve(task, task_id="compute")
    assert reservation is not None
    assert broker.acquire_memory(MemoryLeaseKind.SOURCE_READ, 4 * gib) is None
    reservation.release()


def test_compute_cannot_consume_protected_egress_and_zero_stale_stays_zero() -> None:
    mib = 1024**2
    broker, _snap = _broker(1024 * mib)
    compute = broker.acquire_memory(MemoryLeaseKind.COMPUTE, 900 * mib)
    assert compute is not None
    egress = broker.acquire_protected_egress(
        50 * mib, 50 * mib, lease_id="egress",
    )
    assert egress is not None
    for lease in egress:
        lease.release()
    compute.release()

    zero, _snap = _broker(0)
    decision = zero._conservative_decision()
    assert (
        decision.read_wave_bytes, decision.factor_block_bytes,
        decision.result_queue_bytes, decision.cache_budget_bytes,
    ) == (0, 0, 0, 0)


def test_memory_lease_and_task_reservation_race_share_one_limit() -> None:
    mib = 1024**2
    budget = 1000 * mib
    broker, _snap = _broker(budget)
    barrier = threading.Barrier(2)
    results: dict[str, object | None] = {}
    task = TaskResourceContract(
        peak_memory_bytes=600 * mib, uncertainty=1.0,
        cpu_tokens=1, io_tokens=0, backend="pandas_numpy",
    )

    def acquire_lease() -> None:
        barrier.wait()
        results["lease"] = broker.acquire_memory(
            MemoryLeaseKind.SOURCE_READ, 600 * mib,
        )

    def reserve_task() -> None:
        barrier.wait()
        results["task"] = broker.try_reserve(task, task_id="racing-task")

    threads = [threading.Thread(target=acquire_lease), threading.Thread(target=reserve_task)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    successes = [handle for handle in results.values() if handle is not None]
    assert len(successes) <= 1
    assert broker._running_peak_sum_bytes() + broker._lease_sum_bytes() <= budget
    for handle in successes:
        handle.release()
    assert broker._running_peak_sum_bytes() == 0
    assert broker._lease_sum_bytes() == 0
