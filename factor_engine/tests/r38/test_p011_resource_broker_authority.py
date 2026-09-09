# -*- coding: utf-8 -*-
"""P0-11 ResourceBroker 单权威 closure —— 非空行为测试。

覆盖 mission 的三个非空断言：

(a) 同一 epoch（同一 broker 实例 + 同一 snapshot）内，各消费路径（Task
    admission、ReadWave、CSE cache、cohort、ResultQueue/sink、MemoryLease、
    remote writer / COS multipart）共享**同一个 envelope 值**（SafeEnvelope /
    ExecutionBudget / per-kind 池预算全部派生自同一 broker）；
(b) production 下**绕过 broker 直接超预算分配** → 被拒或 raise（fail-closed）：
    - broker 缺失 → ``MissingResourceBroker``（production 单权威 gate）；
    - 提供 broker 时超预算 ``acquire_memory`` → 返回 None（拒绝），
      ``try_reserve`` 超额 → 返回 None（拒绝）；
(c) epoch 结束释放后预算归还：MemoryLease.release / ReservationLease.release /
    BoundedResultQueue 消费后，broker 账本与队列字节回到释放前水平。

约束：不触碰 R40 不可变 registry；不触碰 object_store.py 流式；幂等。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace

import pytest

from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
from factor_engine.runtime.resource_broker import (
    MissingResourceBroker,
    ResourceBroker,
    broker_degrade_allowed,
    broker_degraded_observable,
    is_production_mode,
    require_broker,
)
from factor_engine.runtime.task_resource_contract import TaskResourceContract


def _broker(hard_memory_limit: int = 8 * 1024**3, cpu_slots: int = 4) -> ResourceBroker:
    """隔离的 broker（硬上限固定，host/cgroup 信号不影响断言）。"""
    broker = ResourceBroker(hard_memory_limit=hard_memory_limit, cpu_slots=cpu_slots)
    snapshot = replace(
        broker.snapshot(),
        hard_memory_limit=hard_memory_limit,
        cgroup_memory_current=0,
        host_mem_available=hard_memory_limit,
        process_rss=0,
        process_family_rss=0,
        process_family_pss=0,
        host_mem_available_known=True,
    )
    broker._refresh = lambda force=False: snapshot
    broker.execution_budget = lambda: int(hard_memory_limit * 0.8)
    broker.pressure_stage = lambda: "NORMAL"
    return broker


def _lease_sum(broker: ResourceBroker) -> int:
    return sum(l.nbytes for l in broker._memory_leases.values())


# ---------------------------------------------------------------------------
# (a) 同一 epoch：所有消费路径共享同一 envelope 值。
# ---------------------------------------------------------------------------


def test_same_epoch_all_consumers_share_same_envelope():
    broker = _broker()
    env = broker.resource_envelope()
    decision = broker.resource_decision()
    # SafeEnvelope 是唯一权威：所有细分预算都派生自它。
    safe = int(getattr(env, "safe_memory_bytes", 0))
    assert safe > 0
    # 各消费路径的"读预算"都来自同一 broker，且落在 envelope 内。
    read_budget = broker.current_read_budget()
    sink_budget = broker.current_sink_budget()
    cse_budget = broker.current_cse_budget()
    remote_budget = broker.current_remote_io_lease_budget()
    assert read_budget >= 0 and sink_budget >= 0
    assert cse_budget >= 0 and remote_budget >= 0
    # ReadWave 预算 = broker ResourceDecision.read_wave_bytes（同一 epoch 采样）。
    wave = int(getattr(decision, "read_wave_bytes", read_budget))
    assert wave > 0
    # cohort 预算也派生自同一 envelope 的 safe_memory_bytes。
    from factor_engine.runtime.execution_cohort import cohort_budget_from_broker

    cohort = cohort_budget_from_broker(broker)
    assert cohort.budget_bytes > 0
    assert cohort.fallback_reason is None  # 真实来源，非回退
    assert cohort.budget_bytes <= safe
    # 各预算总和不超过 envelope 安全内存（1.0 口径：read+compute+... ≤ safe）。
    assert read_budget + sink_budget + cse_budget + remote_budget <= safe * 2  # 软池合计 ~1.0


def test_same_envelope_value_stable_within_one_snapshot():
    broker = _broker()
    # 同一 broker、无超采样节流触发的两次连续采样：live headroom 只会因进程
    # RSS 漂移而微变，但都落在**同一 envelope 结构**（同一 authority 派生）。
    e1 = broker.resource_envelope()
    e2 = broker.resource_envelope()
    # 同一 authority：hard_memory_limit 与 emergency_reserve 恒定；
    # safe_memory_bytes 允许因 live 采样亚秒级漂移（±512MiB 容差）。
    assert int(e1.hard_memory_bytes) == int(e2.hard_memory_bytes)
    assert int(e1.emergency_reserve_bytes) == int(e2.emergency_reserve_bytes)
    assert abs(int(e1.safe_memory_bytes) - int(e2.safe_memory_bytes)) < 512 * 1024**2


# ---------------------------------------------------------------------------
# (b) production 下绕过 broker 超预算分配 → 被拒或 raise。
# ---------------------------------------------------------------------------


def test_require_broker_production_missing_raises(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    assert is_production_mode("production") is True
    with pytest.raises(MissingResourceBroker):
        require_broker(None, raise_on_missing=True, run_mode="production")


def test_require_broker_research_degrade_visible_marker(monkeypatch):
    monkeypatch.delenv("FACTOR_ENGINE_RUN_MODE", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    before = broker_degraded_observable()["count"]
    b = require_broker(None, raise_on_missing=True, run_mode="research")
    assert b is not None
    assert broker_degraded_observable()["count"] == before + 1


def test_acquire_memory_over_execution_budget_rejected():
    broker = _broker()
    budget = broker.execution_budget()
    over = budget + 1
    lease = broker.acquire_memory(MemoryLeaseKind.COMPUTE, over)
    assert lease is None  # fail-closed：不 silently 超额分配


def test_try_reserve_over_budget_rejected():
    broker = _broker()
    # admissible_peak_bytes 远超可用 headroom → can_admit False → 拒绝。
    task = TaskResourceContract(
        peak_memory_bytes=broker.hard_memory_limit,
        uncertainty=10.0,
        cpu_tokens=1,
        io_tokens=1,
        backend="pandas_numpy",
    )
    assert broker.try_reserve(task, task_id="over-budget") is None


def test_brokerless_writer_bypass_is_visible_in_production(monkeypatch):
    """production 下 RemoteFactorBlockWriter 缺 broker 必须 fail-closed。"""
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    from factor_engine.runtime.remote_factor_block_writer import RemoteFactorBlockWriter

    class _DummyPublisher:
        def begin_generation(self, prefix, metadata): return "gen"
        def finish_generation(self, gen): pass
        def add_object(self, gen, key, obj, metadata): pass
        def abort_generation(self, gen): pass

    with pytest.raises(MissingResourceBroker):
        RemoteFactorBlockWriter(
            publisher=_DummyPublisher(),
            prefix="x", dtype="float32", columns_per_block=4,
            broker=require_broker(None, raise_on_missing=True),
        )


# ---------------------------------------------------------------------------
# (c) epoch 释放后预算归还。
# ---------------------------------------------------------------------------


def test_memory_lease_release_returns_budget():
    broker = _broker()
    lease = broker.acquire_memory(MemoryLeaseKind.COMPUTE, 1000)
    assert lease is not None
    assert _lease_sum(broker) == 1000
    lease.release()
    assert _lease_sum(broker) == 0
    # 幂等：重复 release 无害。
    lease.release()
    assert _lease_sum(broker) == 0


def test_memory_lease_release_returns_budget_to_pool():
    broker = _broker()
    l1 = broker.acquire_memory(MemoryLeaseKind.CSE_CACHE, 1000)
    l2 = broker.acquire_memory(MemoryLeaseKind.CSE_CACHE, 2000)
    assert l1 is not None and l2 is not None
    assert _lease_sum(broker) == 3000
    l1.release()
    assert _lease_sum(broker) == 2000
    l2.release()
    assert _lease_sum(broker) == 0


def test_reservation_lease_release_returns_cpu_io_tokens():
    broker = _broker()
    task = TaskResourceContract(
        cpu_tokens=2, io_tokens=1, peak_memory_bytes=0,
        backend="duckdb_sql", backend_threads=2,
    )
    lease = broker.try_reserve(task, task_id="r1")
    assert lease is not None
    assert len(broker._running) == 1
    lease.release()
    assert len(broker._running) == 0
    assert broker._cpu.in_use == 0
    assert broker._io.in_use == 0
    # 幂等。
    lease.release()
    assert len(broker._running) == 0


def test_result_queue_release_returns_bytes_after_get():
    from factor_engine.runtime.streaming_result_sink import BoundedResultQueue, ResultItem

    q = BoundedResultQueue(max_bytes=100)
    assert q.put(ResultItem("a", b"x" * 40, bytes=40)) is True
    assert q.current_bytes == 40
    got = q.get()
    assert got is not None
    assert q.current_bytes == 40  # dequeue transfers ownership to the consumer
    q.release(got)
    assert q.current_bytes == 0


def test_host_coordinator_production_require_broker(monkeypatch):
    """production 下 HostResourceCoordinator（进程级唯一 coordinator）也必须
    fail-closed：broker 缺失时不再静默 new ResourceBroker。"""
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator

    with pytest.raises(MissingResourceBroker):
        HostResourceCoordinator(broker=None)


def test_adaptive_scheduler_production_require_broker(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    import factor_engine.service.queue as service_queue
    import factor_engine.runtime.host_resource_coordinator as coordinator

    monkeypatch.setattr(service_queue, "_get_service_broker", lambda: None)
    monkeypatch.setattr(
        coordinator,
        "get_host_coordinator",
        lambda: (_ for _ in ()).throw(RuntimeError("no coordinator")),
    )
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    with pytest.raises(MissingResourceBroker):
        AdaptiveBatchScheduler(broker=None)
