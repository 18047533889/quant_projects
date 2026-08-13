from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace

import pytest

from runtime.exceptions import (
    ExecutionLeaseRequiredError,
    ProcessWorkerFailure,
    ProductionExecutionCertificateError,
)
from runtime.hybrid_executor import HybridExecutor
from runtime.production_execution_certificate import ProductionExecutionCertificate
from runtime.resource_broker import ResourceBroker
from runtime.task_resource_contract import TaskResourceContract


def _return_42() -> int:
    return 42


def _certificate() -> ProductionExecutionCertificate:
    return ProductionExecutionCertificate.build(
        structural_hash="r41",
        bound_ops=["ts_mean"],
        backend_eligibility=["duckdb_sql"],
        output_shape_hash="shape",
    )


def test_r41_529_thread_submission_does_not_create_process_pool() -> None:
    executor = HybridExecutor(max_thread_workers=1, max_process_workers=1)
    try:
        assert executor.submit("duckdb_sql", _return_42).result(timeout=10) == 42
        assert executor._thread_pool is not None
        assert executor._process_pool is None
    finally:
        executor.shutdown()


def test_r41_530_production_submit_requires_active_lease() -> None:
    executor = HybridExecutor(max_thread_workers=1)
    try:
        with pytest.raises(ExecutionLeaseRequiredError):
            executor.submit("duckdb_sql", _return_42, run_mode="production")

        broker = ResourceBroker(hard_memory_limit=1024**3, cpu_slots=1)
        lease = broker.try_reserve(TaskResourceContract(cpu_tokens=1), task_id="r41")
        assert lease is not None
        assert executor.submit(
            "duckdb_sql", _return_42, run_mode="production", lease=lease
        ).result(timeout=10) == 42
        lease.release()
        with pytest.raises(ExecutionLeaseRequiredError):
            executor.submit(
                "duckdb_sql", _return_42, run_mode="production", lease=lease
            )
    finally:
        executor.shutdown()


def test_r41_certificate_mismatch_blocks_before_function_runs() -> None:
    executor = HybridExecutor(max_thread_workers=1)
    calls = []
    tampered = replace(_certificate(), certificate_hash="tampered")
    try:
        broker = ResourceBroker(hard_memory_limit=1024**3, cpu_slots=1)
        lease = broker.try_reserve(TaskResourceContract(cpu_tokens=1), task_id="cert")
        assert lease is not None
        with pytest.raises(ProductionExecutionCertificateError):
            executor.submit(
                "duckdb_sql",
                lambda: calls.append("called"),
                certificate=tampered,
                run_mode="production",
                lease=lease,
            )
        assert calls == []
        lease.release()
    finally:
        executor.shutdown()


def test_r41_528_process_submit_failure_never_falls_back_to_thread(monkeypatch) -> None:
    class FailingPool:
        def submit(self, *args, **kwargs):
            raise TypeError("cannot pickle payload")

        def shutdown(self, wait=False):
            return None

    executor = HybridExecutor(max_thread_workers=1, max_process_workers=1)
    executor._process_pool = FailingPool()
    monkeypatch.setattr(executor, "_recover_process_pool", lambda: None)
    try:
        with pytest.raises(ProcessWorkerFailure):
            executor.submit("pandas_numpy", lambda: 42, prefer="process")
        assert executor.summary()["thread_task_count"] == 0
        assert executor.summary()["process_breaker_hits"] == 1
    finally:
        executor.shutdown()


def test_r41_528_async_process_failure_rebuilds_pool(monkeypatch) -> None:
    executor = HybridExecutor(max_process_workers=1)
    recovered = []
    monkeypatch.setattr(executor, "_recover_process_pool", lambda: recovered.append(True))
    future = Future()
    future.set_exception(TypeError("cannot pickle local object"))

    executor._observe_process_future(future)

    assert recovered == [True]
    assert executor.summary()["process_breaker_hits"] == 1
