# -*- coding: utf-8 -*-
"""P0-016/017/018/019: autopilot live signals + stale fallback + concurrency dual gate
+ JobLease→DA child lease + recursive release 清 child。

验证：
    - scheduler 上报的 sink_backpressure / job lease 进入 live signals，autopilot 消费；
    - decision 过期 → **直接** conservative（不把 stale last decision 当 fresh）；
    - ``_dynamic_concurrency_limit`` 用 target_concurrency；CPU token-sum 门独立生效；
    - release_lease 递归释放后 child 不再留在 ``_leases``；
    - DA governor host-backed admission（JobLease → DAScanLease child）。
"""
from __future__ import annotations

import time

import pytest

from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.resource_autopilot import ResourceDecision
from factor_engine.runtime.resource_autopilot_service import ResourceDecisionSnapshot


def _decision(**kw) -> ResourceDecision:
    base = dict(
        target_concurrency=4,
        target_cpu_tokens=8,
        read_wave_bytes=256 * 1024**2,
        factor_block_bytes=64 * 1024**2,
        result_queue_bytes=128 * 1024**2,
        io_concurrency=2,
        remote_concurrency=2,
        cache_budget_bytes=128 * 1024**2,
        spill_budget_bytes=0,
        pressure_state="NORMAL",
    )
    base.update(kw)
    return ResourceDecision(**base)


def test_broker_records_and_autopilot_consumes_live_signals():
    from factor_engine.runtime.resource_autopilot_service import ResourceAutopilotService

    broker = ResourceBroker()
    seen: dict = {}

    class _Ctl:
        def tick(self, sig, env, *, job_memory_lease_bytes, sink_backpressure):
            seen["job"] = job_memory_lease_bytes
            seen["bp"] = sink_backpressure
            return _decision()

    ap = ResourceAutopilotService(broker)
    ap._controller = _Ctl()
    # scheduler 上报 live signals。
    broker.resource_decision(sink_backpressure=0.83, job_memory_lease_bytes=7 * 1024**3)
    assert broker.live_signals()["sink_backpressure"] == pytest.approx(0.83)
    assert broker.live_signals()["job_memory_lease_bytes"] == 7 * 1024**3
    # autopilot tick 消费（不再硬编码 None/0.0）。
    ap._tick()
    assert seen["bp"] == pytest.approx(0.83)
    assert seen["job"] == 7 * 1024**3


def test_stale_decision_returns_conservative_not_stale():
    broker = ResourceBroker()
    from factor_engine.runtime.resource_autopilot_service import start_resource_autopilot, stop_resource_autopilot

    # 手动构造一个已过期的 snapshot。
    snap = ResourceDecisionSnapshot(
        decision_id="old",
        generated_at_monotonic=time.monotonic() - 100,
        valid_until=time.monotonic() - 90,  # 已过期
        signals_version="sig:0",
        decision=_decision(target_concurrency=64, target_cpu_tokens=64),
    )
    # 注入到 broker（模拟 autopilot 激活且有过期 snapshot）。
    class _AP:
        started = True

        def last_decision(self):
            return snap

    import factor_engine.runtime.resource_autopilot_service as _svc
    old = _svc.get_resource_autopilot
    try:
        _svc.get_resource_autopilot = lambda: _AP()
        decision = broker.resource_decision()
        # 必须回到 conservative（target_concurrency=1），而不是 stale 的 64。
        assert decision.target_concurrency == 1
        assert decision.pressure_state == "NORMAL"
    finally:
        _svc.get_resource_autopilot = old


def test_concurrency_dual_gate_target_concurrency_and_cpu_tokens():
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.runtime.task_resource_contract import TaskResourceContract

    sched = AdaptiveBatchScheduler(broker=ResourceBroker())
    sched._last_decision = _decision(target_concurrency=4, target_cpu_tokens=8)
    # 任务数上限来自 target_concurrency（不再是 target_cpu_tokens）。
    assert sched._dynamic_concurrency_limit(None) == 4
    # CPU token 预算独立。
    assert sched._decision_cpu_token_budget() == 8

    class _T:
        def __init__(self, cpu):
            self.resource_contract = TaskResourceContract(
                predicted_elapsed_ms=1, cpu_tokens=cpu, peak_memory_bytes=1,
                output_bytes=1, backend="pandas_numpy", backend_threads=1,
            )
        task_type = "ROOT"
        op = "x"
        factor_name = "f"
        node_ref = None
        executable = True
        task_id = "t"

    class _DAG(dict):
        pass

    dag = _DAG()
    dag["t1"] = _T(8)
    dag["t2"] = _T(1)
    futures = {"t1": object()}
    # 8-token task 占满 8 预算。
    assert sched._running_cpu_tokens(futures, dag) == 8
    assert 8 + 1 > sched._decision_cpu_token_budget()  # t2 放不下


def test_recursive_release_evicts_children_from_leases():
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator

    c = HostResourceCoordinator()
    job = c.request_job_lease(owner="j1", memory_bytes=10**9, cpu_tokens=1)
    assert job is not None
    child1 = job.request_child(owner="c1", kind="compute", memory_bytes=30)
    child2 = job.request_child(owner="c2", kind="compute", memory_bytes=20)
    assert child1 is not None and child2 is not None
    gchild = c.request_lease(
        owner="g1", kind="compute", memory_bytes=5,
        parent_lease_id=child1.lease_id,
    )
    assert gchild is not None
    assert c.release_lease(job.lease_id) is True
    # 递归释放后 child / grandchild 都不再留在主 dict（P0-019）。
    for lid in (child1.lease_id, child2.lease_id, gchild.lease_id):
        assert c._leases.get(lid) is None
    assert job.lease_id not in c._leases


def test_da_governor_host_backed_admission():
    from data_access.runtime.resource_governor import (
        GlobalResourceGovernor,
        ResourceReservation,
    )
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator

    gov = GlobalResourceGovernor(max_total_reserved_memory=1000)
    c = HostResourceCoordinator()
    job = c.request_job_lease(owner="j", memory_bytes=500, cpu_tokens=1)
    assert job is not None
    c.set_active_job_lease(job)
    gov.set_host_lease_request(c.request_da_child_lease)

    res = ResourceReservation(query_id="q1", principal_id="p",
                              estimated_memory=100, estimated_scan_bytes=50)
    gov.admit(res)
    assert res.host_lease is not None  # host-backed（JobLease child）
    gov.release("q1")
    assert res.host_lease.released
    # 无活跃 job → 回退本地 governor admission（仍受本地上限约束）。
    c.set_active_job_lease(None)
    res2 = ResourceReservation(query_id="q2", principal_id="p",
                               estimated_memory=100)
    gov.admit(res2)
    assert res2.host_lease is None
    gov.release("q2")
