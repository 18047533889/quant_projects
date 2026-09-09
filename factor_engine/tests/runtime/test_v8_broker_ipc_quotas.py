import pytest
import os
import time

from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.resource_broker_ipc import (
    BrokerIPCOutstandingLeases, ParentBrokerIPC,
)
from factor_engine.runtime.task_resource_contract import TaskResourceContract
from factor_engine.runtime.supervised_worker import SupervisedReusableWorker
from factor_engine.runtime.default_engine import build_execution_core_from_worker_config


def _write_then_hang(_ignored=None):
    from factor_engine.runtime.supervised_worker import emit_worker_progress
    emit_worker_progress("COMPUTE", ordinal=7)
    emit_worker_progress("WRITE", ordinal=7)
    time.sleep(10)


def _malformed_progress_then_return(_ignored=None):
    import factor_engine.runtime.supervised_worker as supervised
    connection, request_id, generation, send_lock = supervised._WORKER_PROGRESS
    with send_lock:
        connection.send((request_id, generation, "progress", {
            "phase": "WRITE", "ordinal": True, "monotonic": time.monotonic(),
        }))
    return "must-not-be-accepted"


def _thread_environment(_ignored=None):
    return {name: os.environ.get(name) for name in (
        "POLARS_MAX_THREADS", "DUCKDB_THREADS", "OMP_NUM_THREADS",
    )}


def _native_thread_limits(_ignored=None):
    import polars as pl
    from threadpoolctl import threadpool_info
    return pl.thread_pool_size(), [item["num_threads"] for item in threadpool_info()]


def _broker():
    from factor_engine.runtime.auto_memory_budget import AutoMemoryBudget
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3, cpu_slots=4,
        min_host_reserve_gb=0, min_host_reserve_fraction=0,
    )
    broker._refresh_auto_budget = lambda: AutoMemoryBudget(
        hard_memory_limit=8 * 1024**3, emergency_reserve=0,
        safe_live_budget=1024**3, execution_budget=1024**3,
        safety_factor=.8, measurement_state="TEST_INJECTED",
    )
    return broker


def test_unqualified_proxy_preserves_parent_cpu_behavior():
    broker = _broker()
    ipc = ParentBrokerIPC(broker)
    try:
        proxy = ipc.create_proxy()
        assert proxy.cpu_budget() == broker.cpu_budget()
        assert proxy.hard_cpu_slots == broker.hard_cpu_slots
    finally:
        ipc.close()


def test_proxy_reports_only_immutable_client_quota_and_clamps_decision():
    broker = _broker()
    ipc = ParentBrokerIPC(broker)
    try:
        proxy = ipc.create_proxy(cpu_quota=1, io_quota=1)
        assert proxy.cpu_budget() == 1
        assert proxy.hard_cpu_slots == 1
        decision = proxy.resource_decision()
        assert decision.target_cpu_tokens == 1
        assert decision.target_concurrency == 1
        assert decision.io_concurrency == 1
    finally:
        ipc.close()


def test_parent_rejects_client_over_quota_before_global_reservation():
    broker = _broker()
    ipc = ParentBrokerIPC(broker)
    try:
        proxy = ipc.create_proxy(cpu_quota=1, io_quota=1)
        task = TaskResourceContract(cpu_tokens=2, io_tokens=1)
        assert proxy.try_reserve(task, task_id="over-client-quota") is None
        assert ipc.active_tokens() == 0
    finally:
        ipc.close()


def test_partitioned_proxy_quotas_sum_to_current_parent_targets():
    broker = _broker()
    ipc = ParentBrokerIPC(broker)
    try:
        proxies = ipc.create_partitioned_proxies(2)
        assert len(proxies) == 2
        assert sum(proxy.cpu_quota for proxy in proxies) == broker.cpu_budget()
        assert sum(proxy.io_quota for proxy in proxies) == broker.hard_cpu_slots
        assert all(proxy.cpu_quota >= 1 and proxy.io_quota >= 1 for proxy in proxies)
    finally:
        ipc.close()


def test_partition_refuses_more_slots_than_current_target():
    broker = _broker()
    ipc = ParentBrokerIPC(broker)
    try:
        assert ipc.create_partitioned_proxies(broker.cpu_budget() + 1) == []
    finally:
        ipc.close()


@pytest.mark.parametrize("cpu,io", [(0, 4), (-1, 4), (4, 0), (4, -1)])
def test_partition_never_promotes_zero_or_negative_parent_target(cpu, io):
    class ZeroAuthority:
        hard_cpu_slots = io

        def cpu_budget(self):
            return cpu

    ipc = ParentBrokerIPC(ZeroAuthority())
    try:
        assert ipc.create_partitioned_proxies(1) == []
        assert ipc.create_partitioned_proxies(2) == []
    finally:
        ipc.close()


@pytest.mark.parametrize("field", ["cpu_quota", "io_quota"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_client_quota_rejected(field, bad):
    ipc = ParentBrokerIPC(_broker())
    try:
        with pytest.raises(ValueError, match=field):
            ipc.create_proxy(**{field: bad})
    finally:
        ipc.close()


def test_spawned_worker_installs_thread_quota_before_inherited_function():
    worker = SupervisedReusableWorker(
        context="spawn", function=_thread_environment,
        process_environment={
            "POLARS_MAX_THREADS": "2", "DUCKDB_THREADS": "2", "OMP_NUM_THREADS": "2",
        },
    )
    try:
        assert worker.execute(None, timeout_seconds=2).value == {
            "POLARS_MAX_THREADS": "2", "DUCKDB_THREADS": "2", "OMP_NUM_THREADS": "2",
        }
    finally:
        worker.close()


def test_worker_thread_environment_rejects_unapproved_or_invalid_values():
    with pytest.raises(ValueError):
        SupervisedReusableWorker(process_environment={"PATH": "1"})
    with pytest.raises(ValueError):
        SupervisedReusableWorker(process_environment={"OMP_NUM_THREADS": "0"})


def test_spawned_worker_applies_native_polars_and_blas_thread_limits():
    worker = SupervisedReusableWorker(
        context="spawn", function=_native_thread_limits,
        process_environment={
            "POLARS_MAX_THREADS": "2", "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        },
    )
    try:
        polars_threads, blas_threads = worker.execute(None, timeout_seconds=5).value
        assert polars_threads <= 2
        assert all(value <= 2 for value in blas_threads)
    finally:
        worker.close()


def test_default_engine_import_bootstraps_before_polars_under_spawn_quota(tmp_path):
    worker = SupervisedReusableWorker(
        context="spawn", function=build_execution_core_from_worker_config,
        process_environment={
            "POLARS_MAX_THREADS": "1", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        },
    )
    try:
        # default_engine is imported while spawn reconstructs the callable, before
        # _worker_loop installs quotas. Reaching its expected configuration
        # error proves that import path did not trip the early-Polars guard.
        with pytest.raises(TypeError, match="validated ExecutionCoreWorkerConfig"):
            worker.execute(None, timeout_seconds=2)
    finally:
        worker.close()


def test_timeout_reports_last_write_phase_and_proves_worker_retired():
    from factor_engine.runtime.supervised_worker import WorkerTimedOut

    worker = SupervisedReusableWorker(
        context="spawn", function=_write_then_hang,
        cancel_grace_seconds=.01, exit_observation_seconds=.5,
    )
    with pytest.raises(WorkerTimedOut) as caught:
        worker.execute(
            None, timeout_seconds=1, expected_progress_ordinals=frozenset({7})
        )
    assert caught.value.stage == "WRITE"
    assert worker._process is not None
    assert not worker._process.is_alive()


def test_malformed_progress_frame_is_transport_integrity_failure_and_retires_worker():
    from factor_engine.runtime.supervised_worker import WorkerTransportFailed

    worker = SupervisedReusableWorker(
        context="spawn", function=_malformed_progress_then_return,
        exit_observation_seconds=.5,
    )
    with pytest.raises(WorkerTransportFailed, match="invalid worker progress frame"):
        worker.execute(
            None, timeout_seconds=1, expected_progress_ordinals=frozenset({7})
        )
    assert worker._process is not None
    assert not worker._process.is_alive()


def test_ipc_close_refuses_to_silently_drop_unbound_outstanding_lease():
    from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind

    broker = _broker()
    ipc = ParentBrokerIPC(broker)
    proxy = ipc.create_proxy()
    lease = proxy.acquire_memory(MemoryLeaseKind.COMPUTE, 1)
    assert lease is not None and ipc.active_tokens() == 1
    with pytest.raises(BrokerIPCOutstandingLeases) as caught:
        ipc.close()
    assert caught.value.clients == (proxy.client_id,)
    assert ipc.active_tokens() == 1
    lease.release()
    ipc.close()
