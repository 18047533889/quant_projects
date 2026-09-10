from __future__ import annotations

from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator
from factor_engine.runtime.job_scoped_lease import (
    acquire_job_scoped_memory, reserve_job_scoped_task,
)
from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.resource_broker_ipc import ParentBrokerIPC
from factor_engine.runtime.task_resource_contract import TaskResourceContract


def _job():
    hard = 8 * 1024**3
    broker = ResourceBroker(
        hard_memory_limit=hard, cpu_slots=4,
        min_host_reserve_gb=0, min_host_reserve_fraction=0,
    )
    snap = replace(
        broker.snapshot(), hard_memory_limit=hard, cgroup_memory_current=0,
        host_mem_available=hard, process_rss=0, worker_rss=0,
        process_family_rss=0, process_family_pss=0,
        host_mem_available_known=True,
    )
    broker._refresh = lambda force=False: snap
    coordinator = HostResourceCoordinator(broker=broker)
    job = coordinator.request_job_lease(
        owner="test-job", memory_bytes=4 * 1024**3, cpu_tokens=4,
    )
    assert job is not None
    return broker, coordinator, job


def test_no_active_job_preserves_plain_broker_behavior():
    broker, _coordinator, job = _job()
    before = job.remaining_memory_bytes
    lease = broker.acquire_memory(MemoryLeaseKind.CSE_CACHE, 1024, lease_id="plain")
    assert lease is not None
    assert job.remaining_memory_bytes == before
    lease.release()


def test_raw_memory_and_task_charge_and_release_active_job():
    broker, _coordinator, job = _job()
    with job.bind_context():
        memory = broker.acquire_memory(MemoryLeaseKind.CSE_CACHE, 1024, lease_id="raw")
        task = broker.try_reserve(TaskResourceContract(
            peak_memory_bytes=2048, uncertainty=1.0, cpu_tokens=1,
            io_tokens=0, backend="pandas_numpy",
        ), task_id="raw-task")
    assert memory is not None and task is not None
    assert job.remaining_memory_bytes == job.memory_bytes - 3072
    memory.release()
    memory.release()
    task.release()
    task.release()
    assert job.remaining_memory_bytes == job.memory_bytes


def test_existing_paired_helper_does_not_double_charge():
    broker, _coordinator, job = _job()
    with job.bind_context():
        lease = acquire_job_scoped_memory(
            broker, job, MemoryLeaseKind.READ_WAVE, 4096, lease_id="paired",
        )
    assert lease is not None
    assert job.remaining_memory_bytes == job.memory_bytes - 4096
    lease.release()
    assert job.remaining_memory_bytes == job.memory_bytes


def test_protected_egress_child_lives_until_both_physical_leases_release():
    broker, _coordinator, job = _job()
    with job.bind_context():
        leases = broker.acquire_protected_egress(1024, 2048, lease_id="egress")
    assert leases is not None
    assert job.remaining_memory_bytes == job.memory_bytes - 3072
    leases[0].release()
    assert job.remaining_memory_bytes == job.memory_bytes - 3072
    leases[1].release()
    assert job.remaining_memory_bytes == job.memory_bytes


def test_ipc_controller_binds_explicit_job_scope():
    broker, _coordinator, job = _job()
    ipc = ParentBrokerIPC(broker, job_lease=job)
    proxy = ipc.create_proxy()
    lease = proxy.acquire_memory(MemoryLeaseKind.CSE_CACHE, 1024, lease_id="ipc")
    assert lease is not None
    assert job.remaining_memory_bytes == job.memory_bytes - 1024
    lease.release()
    assert job.remaining_memory_bytes == job.memory_bytes
    ipc.close()


def test_raw_reservation_transfers_to_memory_without_job_child_gap():
    broker, _coordinator, job = _job()
    task = TaskResourceContract(
        peak_memory_bytes=4096, uncertainty=1.0, cpu_tokens=1,
        io_tokens=0, backend="pandas_numpy",
    )
    with job.bind_context():
        reservation = broker.try_reserve(task, task_id="transfer-raw")
    assert reservation is not None
    memory = reservation.transfer_memory_ownership(
        MemoryLeaseKind.CSE_CACHE, 2048, lease_id="transfer-cache",
    )
    assert memory is not None
    assert broker._running_peak_sum_bytes() == 0
    assert broker._lease_sum_bytes() == 2048
    assert job.remaining_memory_bytes == job.memory_bytes - 4096
    reservation.release()
    assert job.remaining_memory_bytes == job.memory_bytes - 4096
    memory.release()
    assert job.remaining_memory_bytes == job.memory_bytes


def test_paired_reservation_transfer_keeps_exactly_one_child():
    broker, _coordinator, job = _job()
    task = TaskResourceContract(
        peak_memory_bytes=4096, uncertainty=1.0, cpu_tokens=1,
        io_tokens=0, backend="pandas_numpy",
    )
    with job.bind_context():
        reservation = reserve_job_scoped_task(
            broker, job, task, task_id="transfer-paired",
        )
    assert reservation is not None
    assert job.remaining_memory_bytes == job.memory_bytes - 4096
    memory = reservation.transfer_memory_ownership(
        MemoryLeaseKind.CSE_CACHE, 1024, lease_id="paired-cache",
    )
    assert memory is not None
    assert job.remaining_memory_bytes == job.memory_bytes - 4096
    memory.release()
    assert job.remaining_memory_bytes == job.memory_bytes


def test_transfer_construction_failure_preserves_running_reservation(monkeypatch):
    import factor_engine.runtime.resource_broker as broker_module

    broker, _coordinator, job = _job()
    task = TaskResourceContract(
        peak_memory_bytes=4096, uncertainty=1.0, cpu_tokens=1,
        io_tokens=0, backend="pandas_numpy",
    )
    with job.bind_context():
        reservation = broker.try_reserve(task, task_id="transfer-failure")
    assert reservation is not None

    class FailingMemoryLease:
        def __init__(self, *args, **kwargs):
            raise MemoryError("injected transfer construction failure")

    monkeypatch.setattr(broker_module, "MemoryLease", FailingMemoryLease)
    try:
        reservation.transfer_memory_ownership(
            MemoryLeaseKind.CSE_CACHE, 1024, lease_id="never-inserted",
        )
    except MemoryError:
        pass
    else:
        raise AssertionError("injected MemoryError was not propagated")
    assert broker._running.get("transfer-failure") is task
    assert broker._cpu.in_use == 1
    assert job.remaining_memory_bytes == job.memory_bytes - 4096
    reservation.release()
    assert job.remaining_memory_bytes == job.memory_bytes


@pytest.mark.parametrize("raises", [False, True])
def test_host_denial_or_exception_rolls_back_job_child(monkeypatch, raises):
    broker, _coordinator, job = _job()

    def fail_host(*args, **kwargs):
        if raises:
            raise RuntimeError("injected host admission failure")
        return None

    monkeypatch.setattr(broker, "_acquire_memory_host_only", fail_host)
    before = job.remaining_memory_bytes
    with job.bind_context():
        if raises:
            with pytest.raises(RuntimeError, match="injected host"):
                broker.acquire_memory(MemoryLeaseKind.CSE_CACHE, 1024, lease_id="fail")
        else:
            assert broker.acquire_memory(
                MemoryLeaseKind.CSE_CACHE, 1024, lease_id="deny",
            ) is None
    assert job.remaining_memory_bytes == before


def test_two_threads_cannot_exceed_job_or_broker_memory_limits():
    broker, _coordinator, job = _job()
    gib = 1024**3
    barrier = threading.Barrier(2)
    leases = []
    lock = threading.Lock()

    def acquire(index):
        with job.bind_context():
            barrier.wait()
            lease = broker.acquire_memory(
                MemoryLeaseKind.CSE_CACHE, 3 * gib, lease_id=f"race-{index}",
            )
        if lease is not None:
            with lock:
                leases.append(lease)

    threads = [threading.Thread(target=acquire, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(leases) == 1
    assert job.memory_bytes - job.remaining_memory_bytes <= job.memory_bytes
    assert broker._committed_memory_bytes_locked() <= broker.execution_budget()
    leases[0].release()
    assert job.remaining_memory_bytes == job.memory_bytes


def test_small_active_job_clamps_budget_helpers_and_fresh_old_autopilot_snapshot():
    broker, _coordinator, job = _job()
    cap = 64 * 1024**2
    job._lease.memory_bytes = cap
    large = replace(
        broker._conservative_decision(),
        read_wave_bytes=256 * 1024**2,
        factor_block_bytes=256 * 1024**2,
        result_queue_bytes=256 * 1024**2,
        cache_budget_bytes=256 * 1024**2,
    )
    broker._autopilot_service = SimpleNamespace(
        started=True,
        last_decision=lambda: SimpleNamespace(is_stale=False, decision=large),
    )
    with job.bind_context():
        assert broker.current_read_budget() <= cap
        assert broker.current_sink_budget() <= cap
        assert broker.automatic_result_queue_budget() <= cap
        decision = broker.resource_decision(job_memory_lease_bytes=cap)
    memory_targets = (
        decision.read_wave_bytes, decision.factor_block_bytes,
        decision.result_queue_bytes, decision.cache_budget_bytes,
    )
    assert sum(memory_targets) <= cap
    assert all(0 <= value <= 256 * 1024**2 for value in memory_targets)


@pytest.mark.parametrize("operation", ["memory", "task"])
def test_explicit_paired_helper_rejects_cross_broker_job_without_charging(operation):
    broker_a, _coordinator_a, job_a = _job()
    broker_b, _coordinator_b, job_b = _job()
    before = job_a.remaining_memory_bytes
    if operation == "memory":
        lease = acquire_job_scoped_memory(
            broker_b, job_a, MemoryLeaseKind.READ_WAVE, 4096,
            lease_id="cross-broker-memory",
        )
    else:
        lease = reserve_job_scoped_task(
            broker_b, job_a,
            TaskResourceContract(
                peak_memory_bytes=4096, uncertainty=1.0, cpu_tokens=1,
                io_tokens=0, backend="pandas_numpy",
            ),
            task_id="cross-broker-task",
        )
    assert lease is None
    assert job_a.remaining_memory_bytes == before
    assert job_b.remaining_memory_bytes == job_b.memory_bytes
    assert broker_a._committed_memory_bytes_locked() == 0
    assert broker_b._committed_memory_bytes_locked() == 0
