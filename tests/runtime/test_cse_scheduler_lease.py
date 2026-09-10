from dataclasses import replace

import numpy as np

from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
from factor_engine.runtime.buffer_store import GovernedBufferStore
from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.task_resource_contract import TaskResourceContract
from factor_engine.planner.physical_factor_dag import (
    TASK_CSE_SHARED, TASK_ROOT, PhysicalFactorDAG, PhysicalFactorTask,
)


class _Ctx:
    pass


def _broker(memory=4096, cpu=1):
    broker = ResourceBroker(
        hard_memory_limit=memory,
        cpu_slots=cpu,
        min_host_reserve_gb=0,
        min_host_reserve_fraction=0,
    )
    snap = replace(
        broker.snapshot(),
        hard_memory_limit=memory,
        cgroup_memory_current=0,
        host_mem_available=memory,
        process_rss=0,
        worker_rss=0,
        process_family_rss=0,
        process_family_pss=0,
        host_mem_available_known=True,
    )
    broker._refresh = lambda force=False: snap
    return broker


def _contract(peak, cpu=1):
    return TaskResourceContract(
        peak_memory_bytes=peak,
        uncertainty=1.0,
        cpu_tokens=cpu,
        io_tokens=0,
        backend="pandas_numpy",
    )


def _scheduler():
    scheduler = object.__new__(AdaptiveBatchScheduler)
    scheduler._lease_scope = "cse-test"
    return scheduler


def test_completed_cse_releases_cpu_but_keeps_actual_memory_committed():
    broker = _broker()
    reservation = broker.try_reserve(_contract(2048), task_id="cse-task")
    assert reservation is not None
    ctx = _Ctx()
    ctx.shared_buffers = GovernedBufferStore({}, budget_bytes=4096)
    value = np.arange(128, dtype=np.int64)
    ctx.shared_buffers.put("shared", value, bytes_=value.nbytes)

    assert _scheduler()._transfer_cse_memory_ownership(
        ctx, "cse:shared", reservation
    ) is True
    assert broker._running_peak_sum_bytes() == 0
    assert broker._lease_sum_bytes() == value.nbytes

    # The task's CPU token is immediately reusable by a dependent consumer.
    consumer = broker.try_reserve(_contract(512), task_id="consumer")
    assert consumer is not None
    consumer.release()
    assert broker._lease_sum_bytes() == value.nbytes

    ctx.shared_buffers.release("shared")
    del value


def test_cse_actual_size_above_reserved_peak_fails_without_transfer():
    broker = _broker()
    reservation = broker.try_reserve(_contract(512), task_id="too-small")
    assert reservation is not None
    ctx = _Ctx()
    ctx.shared_buffers = GovernedBufferStore({}, budget_bytes=4096)
    value = np.arange(128, dtype=np.int64)
    ctx.shared_buffers.put("shared", value, bytes_=value.nbytes)

    assert _scheduler()._transfer_cse_memory_ownership(
        ctx, "cse:shared", reservation
    ) is False
    # Failed transfer leaves the original reservation wholly intact so the
    # caller can remove the unaccounted entry before releasing it.
    assert broker._running_peak_sum_bytes() == 512
    assert broker._lease_sum_bytes() == 0
    assert broker.try_reserve(_contract(1), task_id="blocked") is None
    ctx.shared_buffers.release("shared")
    reservation.release()


def test_nonresident_spilled_or_zero_byte_cse_needs_no_transfer():
    broker = _broker()
    reservation = broker.try_reserve(_contract(512), task_id="nonresident")
    assert reservation is not None
    ctx = _Ctx()
    ctx.shared_buffers = GovernedBufferStore({}, budget_bytes=4096)
    assert _scheduler()._transfer_cse_memory_ownership(
        ctx, "cse:spilled", reservation
    ) is None
    ctx.shared_buffers.put("zero", np.empty(0), bytes_=0)
    assert _scheduler()._transfer_cse_memory_ownership(
        ctx, "cse:zero", reservation
    ) is None
    reservation.release()


def test_real_scheduler_cse_hands_cpu_to_dependent_root(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    broker = _broker(memory=4096, cpu=1)
    scheduler = AdaptiveBatchScheduler(broker=broker, execution_policy="thread")
    dag = PhysicalFactorDAG()
    dag.add_task(PhysicalFactorTask(
        task_id="cse:shared", op="ts_mean", task_type=TASK_CSE_SHARED,
        consumers=("root:out",), resource_contract=_contract(2048),
    ))
    dag.add_task(PhysicalFactorTask(
        task_id="root:out", op="identity", task_type=TASK_ROOT,
        inputs=("cse:shared",), resource_contract=_contract(512),
        factor_name="out",
    ))
    dag.roots = ("root:out",)
    ctx = _Ctx()
    ctx.run_mode = "research"
    ctx.runtime_stats = {}
    ctx.shared_result_cache = {}
    ctx.shared_buffers = GovernedBufferStore({}, budget_bytes=4096)
    value = np.arange(128, dtype=np.int64)

    def materialize(sid, _node):
        ctx.shared_buffers.put(sid, value, bytes_=value.nbytes)

    result = scheduler.run(
        dag, backend=None, ctx=ctx,
        materialize_shared=materialize,
        execute_root=lambda task: "completed",
    )
    assert result["results"] == {"out": "completed"}
    assert broker._running_peak_sum_bytes() == 0
    assert broker._lease_sum_bytes() == value.nbytes
    ctx.shared_buffers.release("shared")


def test_serial_scheduler_admits_cse_then_dependent_root():
    broker = _broker(memory=4096, cpu=1)
    scheduler = AdaptiveBatchScheduler(broker=broker, execution_policy="thread")
    dag = PhysicalFactorDAG()
    dag.add_task(PhysicalFactorTask(
        task_id="cse:shared", op="ts_mean", task_type=TASK_CSE_SHARED,
        consumers=("root:out",), resource_contract=_contract(2048),
    ))
    dag.add_task(PhysicalFactorTask(
        task_id="root:out", op="identity", task_type=TASK_ROOT,
        inputs=("cse:shared",), resource_contract=_contract(512),
        factor_name="out",
    ))
    dag.roots = ("root:out",)
    ctx = _Ctx()
    ctx.run_mode, ctx.runtime_stats = "research", {}
    ctx.shared_result_cache = {}
    ctx.shared_buffers = GovernedBufferStore({}, budget_bytes=4096)
    value = np.arange(128, dtype=np.int64)

    result = scheduler.run_serial_fused(
        dag, backend=None, ctx=ctx,
        materialize_shared=lambda sid, node: ctx.shared_buffers.put(
            sid, value, bytes_=value.nbytes
        ),
        execute_root=lambda task: "serial-completed",
    )
    assert result["results"] == {"out": "serial-completed"}
    assert broker._running_peak_sum_bytes() == 0
    assert broker._lease_sum_bytes() == value.nbytes
    ctx.shared_buffers.release("shared")


def test_serial_scheduler_fails_closed_when_task_exceeds_budget():
    import pytest
    from factor_engine.runtime.resource_errors import ResourceBudgetExceeded

    broker = _broker(memory=1024, cpu=1)
    scheduler = AdaptiveBatchScheduler(broker=broker, execution_policy="thread")
    dag = PhysicalFactorDAG()
    dag.add_task(PhysicalFactorTask(
        task_id="root:too-large", op="identity", task_type=TASK_ROOT,
        resource_contract=_contract(2048), factor_name="too-large",
    ))
    dag.roots = ("root:too-large",)
    ctx = _Ctx()
    ctx.run_mode, ctx.runtime_stats = "research", {}
    ctx.shared_result_cache = {}
    called = []
    with pytest.raises(ResourceBudgetExceeded, match="serial root admission denied"):
        scheduler.run_serial_fused(
            dag, backend=None, ctx=ctx,
            execute_root=lambda task: called.append(task.task_id),
        )
    assert called == []
    assert broker._running_peak_sum_bytes() == 0
