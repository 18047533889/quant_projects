# -*- coding: utf-8 -*-
"""P3/P4: AutoMemoryBudget 公式 + MemoryLease 软池 + 硬不变量。

覆盖交付要求的五个断言：
    (a) ExecutionBudget <= HardMemoryLimit
    (b) EmergencyReserve 是预留的（不作为 ExecutionBudget 的一部分）
    (c) 软池弹性借用：空闲池把额度借给忙池（全局硬上限内）
    (d) 硬不变量 SUM(leases)+candidate <= ExecutionBudget 强制（超限拒绝）
    (e) 固定 4GiB 默认已移除（默认预算从 broker 派生，而非字面 4GiB）
"""
from __future__ import annotations

import pytest

from factor_engine.runtime.auto_memory_budget import (
    MemoryLeaseKind,
    compute_auto_memory_budget,
)
from factor_engine.runtime.resource_broker import ResourceBroker


def _broker(hard: int = 8 * 1024**3, *, cpu: int = 8) -> ResourceBroker:
    return ResourceBroker(
        hard_memory_limit=hard,
        cpu_slots=cpu,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )


# -- AutoMemoryBudget 公式 ---------------------------------------------------

def test_execution_budget_never_exceeds_hard_limit():
    # (a) ExecutionBudget <= HardMemoryLimit，且 <= SafeLiveBudget。
    hard = 8 * 1024**3
    budget = compute_auto_memory_budget(
        hard_memory_limit=hard,
        cgroup_current=1 * 1024**3,
        host_mem_available=6 * 1024**3,
        process_family_rss=1 * 1024**3,
        min_reserve_gb=1,
        min_reserve_fraction=0.1,
    )
    assert budget.execution_budget <= budget.hard_memory_limit
    assert budget.execution_budget <= budget.safe_live_budget
    assert budget.safe_live_budget >= 0


def test_emergency_reserve_is_reserved_not_part_of_execution_budget():
    # (b) EmergencyReserve 预留：SafeLiveBudget 已扣除 reserve，
    #     ExecutionBudget = SafeLiveBudget × safety_factor（不含 reserve）。
    hard = 8 * 1024**3
    budget = compute_auto_memory_budget(
        hard_memory_limit=hard,
        cgroup_current=1 * 1024**3,
        host_mem_available=6 * 1024**3,
        process_family_rss=1 * 1024**3,
        min_reserve_gb=2,
        min_reserve_fraction=0.3,
    )
    # reserve = max(2GB, 30%×8GB=2.4GB) = 2.4GB
    assert budget.emergency_reserve >= 2 * 1024**3
    # SafeLiveBudget ≤ hard - reserve（host/cgroup/configured 中最严格）。
    assert budget.safe_live_budget <= hard - budget.emergency_reserve
    # ExecutionBudget 由 SafeLiveBudget 派生，EmergencyReserve 不流入。
    assert budget.execution_budget <= budget.safe_live_budget


def test_safety_factor_explicit_override_remains_capped():
    hard = 8 * 1024**3
    b85 = compute_auto_memory_budget(
        hard_memory_limit=hard, cgroup_current=1 * 1024**3,
        host_mem_available=6 * 1024**3, process_family_rss=1 * 1024**3,
        safety_factor=0.9,
    )
    assert b85.safety_factor == 0.85
    # 用更小的 safety_factor。
    b70 = compute_auto_memory_budget(
        hard_memory_limit=hard, cgroup_current=1 * 1024**3,
        host_mem_available=6 * 1024**3, process_family_rss=1 * 1024**3,
        safety_factor=0.7,
    )
    assert b70.safety_factor == 0.7


def test_unknown_measurement_fails_closed_without_hard_limit_fallback():
    hard = 16 * 1024**3
    budget = compute_auto_memory_budget(
        hard_memory_limit=hard,
        cgroup_current=None,
        host_mem_available=None,
        process_family_rss=None,
        min_reserve_gb=0,
        min_reserve_fraction=0.0,
        safety_factor=0.75,
    )
    assert budget.measurement_state == "UNKNOWN"
    assert budget.execution_budget == 0


def test_failed_host_probe_is_unknown_while_measured_zero_is_known_zero():
    common = dict(
        hard_memory_limit=8 * 1024**3,
        cgroup_current=None,
        cgroup_remaining=4 * 1024**3,
        process_family_rss=1024**3,
        min_reserve_gb=0,
        min_reserve_fraction=0,
    )
    unknown = compute_auto_memory_budget(
        **common, host_mem_available=None, host_measurement_known=False
    )
    zero = compute_auto_memory_budget(
        **common, host_mem_available=0, host_measurement_known=True
    )
    assert unknown.measurement_state == "UNKNOWN"
    assert unknown.execution_budget == 0
    assert zero.measurement_state == "KNOWN_ZERO"
    assert zero.execution_budget == 0


@pytest.mark.parametrize(
    ("cgroup_current", "host_available", "family_rss"),
    [
        (64 * 1024**3, 48 * 1024**3, 1 * 1024**3),
        (1 * 1024**3, 0, 1 * 1024**3),
        (1 * 1024**3, 48 * 1024**3, 64 * 1024**3),
    ],
)
def test_genuine_zero_headroom_never_resurrects_positive_budget(
    cgroup_current, host_available, family_rss
):
    hard = 64 * 1024**3
    budget = compute_auto_memory_budget(
        hard_memory_limit=hard,
        cgroup_current=cgroup_current,
        host_mem_available=host_available,
        process_family_rss=family_rss,
        min_reserve_gb=0,
        min_reserve_fraction=0,
    )
    assert budget.measurement_state == "KNOWN_ZERO"
    assert budget.safe_live_budget == 0
    assert budget.execution_budget == 0


def test_verified_residency_reconstructs_pool_without_using_lease_promises():
    gib = 1024**3
    budget = compute_auto_memory_budget(
        hard_memory_limit=64 * gib,
        cgroup_current=34 * gib,
        host_mem_available=30 * gib,
        process_family_rss=34 * gib,
        min_reserve_gb=0,
        min_reserve_fraction=0,
        verified_pool_residency_by_domain={
            "cgroup_remaining": 30 * gib,
            "host_remaining": 30 * gib,
            "configured_remaining": 30 * gib,
        },
    )
    assert budget.execution_budget == int(0.8 * 60 * gib)


def test_residency_is_domain_matched_and_cannot_bypass_tighter_domain():
    gib = 1024**3
    budget = compute_auto_memory_budget(
        hard_memory_limit=64 * gib,
        cgroup_current=None,
        cgroup_remaining=10 * gib,
        host_mem_available=30 * gib,
        process_family_rss=34 * gib,
        min_reserve_gb=0,
        min_reserve_fraction=0,
        verified_pool_residency_by_domain={"host_remaining": 30 * gib},
    )
    assert budget.execution_budget == 8 * gib


def test_rlimit_address_space_zero_denies_even_when_rss_headroom_positive():
    gib = 1024**3
    budget = compute_auto_memory_budget(
        hard_memory_limit=8 * gib,
        cgroup_current=None,
        cgroup_remaining=8 * gib,
        host_mem_available=8 * gib,
        process_family_rss=1 * gib,
        address_space_remaining=0,
        min_reserve_gb=0,
        min_reserve_fraction=0,
    )
    assert budget.measurement_state == "KNOWN_ZERO"
    assert budget.execution_budget == 0


# -- ResourceBroker MemoryLease 软池 ----------------------------------------

def test_memory_lease_respects_global_hard_invariant():
    # (d) SUM(leases) + candidate <= ExecutionBudget 强制。
    broker = _broker(8 * 1024**3)
    exec_budget = broker.execution_budget()
    assert exec_budget > 0
    # 先拿整块（占满 ExecutionBudget）。
    lease1 = broker.acquire_memory(MemoryLeaseKind.COMPUTE, exec_budget)
    assert lease1 is not None
    assert broker._lease_sum_bytes() <= exec_budget
    # 超过 budget → 拒绝（fail-closed）。
    over = broker.acquire_memory(MemoryLeaseKind.COMPUTE, 1)
    assert over is None
    assert broker._lease_sum_bytes() <= exec_budget
    # 释放后可再拿。
    lease1.release()
    assert broker.acquire_memory(MemoryLeaseKind.COMPUTE, exec_budget) is not None


def test_soft_pool_borrowing_idle_pool_lends_to_busy_pool():
    # (c) 软池弹性借用：忙池（COMPUTE）可借用空闲池（RESULT_QUEUE）的额度，
    #     只要全局硬上限不超。
    broker = _broker(8 * 1024**3)
    exec_budget = broker.execution_budget()
    compute_pool = broker._pool_limit(MemoryLeaseKind.COMPUTE, exec_budget)
    # 把 COMPUTE 池占满。
    l1 = broker.acquire_memory(MemoryLeaseKind.COMPUTE, compute_pool)
    assert l1 is not None
    # 再借 COMPUTE 池额外一块（此时 COMPUTE 已超过其初始份额，向全局借用）。
    l2 = broker.acquire_memory(MemoryLeaseKind.COMPUTE, exec_budget // 4)
    assert l2 is not None
    # 全局硬上限仍然成立（SUM <= ExecutionBudget）。
    assert broker._lease_sum_bytes() <= exec_budget
    # 且累计超过 COMPUTE 初始份额 —— 证明跨池借用了空闲额度。
    assert broker._lease_sum_bytes() > compute_pool
    l1.release()
    l2.release()


def test_lease_release_idempotent():
    broker = _broker(8 * 1024**3)
    lease = broker.acquire_memory(MemoryLeaseKind.CSE_CACHE, 1024**3)
    assert lease is not None
    assert broker._lease_sum_bytes() == 1024**3
    lease.release()
    lease.release()  # 幂等
    assert broker._lease_sum_bytes() == 0


# -- 固定 4GiB 默认已移除 ------------------------------------------------

def test_engine_wave_budget_defaults_to_none_not_fixed_4gib():
    # (e) 引擎/调度器默认不再写死 4GiB。
    import inspect

    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    sig = inspect.signature(AdaptiveBatchScheduler.__init__)
    assert sig.parameters["wave_memory_budget"].default is None

    from factor_engine.runtime.streaming_result_sink import StreamingResultSink
    sig2 = inspect.signature(StreamingResultSink.__init__)
    assert sig2.parameters["queue_bytes"].default is None

    from factor_engine.runtime.multibackend.cse_cache_optimizer import CSECacheOptimizer
    sig3 = inspect.signature(CSECacheOptimizer.__init__)
    assert sig3.parameters["max_cache_bytes"].default is None

    from factor_engine.runtime import engine as engine_mod
    src = inspect.getsource(engine_mod)
    # materialize_many_fast / plan_many_fast 的 wave/writer 默认应为 None。
    assert "wave_memory_budget: int | None = None" in src


def test_broker_current_budgets_resolve_from_live_headroom_not_4gib():
    # (e) broker 派生的 read/sink/cse 预算来自 live headroom（ExecutionBudget ×
    #     软池分数），不是字面 4GiB。
    broker = _broker(64 * 1024**3)
    exec_budget = broker.execution_budget()
    read = broker.current_read_budget()
    sink = broker.current_sink_budget()
    cse = broker.current_cse_budget()
    # 每个预算都来自 ExecutionBudget × fraction（< ExecutionBudget）。
    assert 0 < read <= exec_budget
    assert 0 < sink <= exec_budget
    assert 0 < cse <= exec_budget
    # read+sink+cse 之和不超过 ExecutionBudget（软池分数和 ≤ 1）。
    assert read + sink + cse <= exec_budget
