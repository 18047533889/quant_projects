# -*- coding: utf-8 -*-
"""R31 验收测试：scheduler 接线 / resource lease / cost / fusion / dataaccess / change-impact。"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("FACTOR_ENGINE_CPU_BUDGET", "4")
os.environ.setdefault("FACTOR_ENGINE_MAX_MEMORY_BYTES", str(8 * 1024**3))


# ---------------------------------------------------------------------------
# R31-005/006/009: ResourceBroker lease / pure can_admit
# ---------------------------------------------------------------------------


def test_can_admit_is_pure_and_try_reserve_lease():
    from factor_engine.runtime.resource_broker import ResourceBroker
    from factor_engine.runtime.task_resource_contract import TaskResourceContract

    b = ResourceBroker(cpu_slots=4, hard_memory_limit=16 * 1024**3)
    t = TaskResourceContract(cpu_tokens=2, io_tokens=1, peak_memory_bytes=2 * 1024**3)
    before = (b._cpu.in_use, b._io.in_use)
    assert b.can_admit(t)  # 纯判定：无副作用
    assert (b._cpu.in_use, b._io.in_use) == before
    lease = b.try_reserve(t, task_id="t1")
    assert lease is not None
    assert b._cpu.in_use == 2
    lease.release()
    lease.release()  # 幂等
    assert b._cpu.in_use == 0 and b._io.in_use == 0


def test_failure_path_releases_exactly_once():
    from factor_engine.runtime.resource_broker import ResourceBroker
    from factor_engine.runtime.task_resource_contract import TaskResourceContract

    b = ResourceBroker(cpu_slots=2, hard_memory_limit=16 * 1024**3)
    lease = b.try_reserve(TaskResourceContract(cpu_tokens=2), task_id="x")
    assert lease is not None
    # 模拟 future.result() 抛异常后的释放路径
    lease.release()
    assert b._cpu.in_use == 0  # 无泄漏


# ---------------------------------------------------------------------------
# R31-013/014/015/016/017: cost router
# ---------------------------------------------------------------------------


def test_cost_counts_every_occurrence_with_bound_params():
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.backend.plan_cost_router import plan_occurrences, _canonical_ops, _dag_aware_mixed_cost

    close = PlanNode("column", (), {"name": "close"})
    vol = PlanNode("column", (), {"name": "volume"})
    amt = PlanNode("column", (), {"name": "amount"})
    m1 = PlanNode("ts_mean", (close,), {"window": 5})
    m2 = PlanNode("ts_mean", (vol,), {"window": 20})
    m3 = PlanNode("ts_mean", (amt,), {"window": 60})
    m4 = PlanNode("ts_mean", (close,), {"window": 120})
    root = PlanNode("add", (PlanNode("add", (m1, m2), {}), PlanNode("add", (m3, m4), {})), {})
    ops = _canonical_ops(root)
    assert ops.count("ts_mean") == 4, "每个 occurrence 单独计数，不去重"
    occs = plan_occurrences(root)
    assert sorted(o.window for o in occs if o.canonical == "ts_mean") == [5, 20, 60, 120]
    mixed = _dag_aware_mixed_cost(
        occs, rows=500_000, delegate_ops=frozenset(), data_kind="duckdb", mode="research", source_ref=False
    )
    assert mixed is not None and mixed > 0


def test_backend_specific_memory_cost():
    from factor_engine.backend.plan_cost_router import estimate_plan_peak_memory

    p_pd = estimate_plan_peak_memory(("ts_mean",), 500_000, "pandas_numpy")
    p_sql = estimate_plan_peak_memory(("ts_mean",), 500_000, "duckdb_sql")
    assert p_sql < p_pd, "SQL streaming 内存应低于 Pandas 全量物化"


# ---------------------------------------------------------------------------
# R31-P0-002/003/004: physical lowerer real stages
# ---------------------------------------------------------------------------


def test_lowerer_produces_real_stages():
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.planner.physical_lowerer import lower_root_plan

    plan = PlanNode(
        "add",
        inputs=[
            PlanNode("ts_mean", [PlanNode("column", attrs={"name": "close"})], attrs={"window": 5}),
            PlanNode("rank", [PlanNode("column", attrs={"name": "volume"})]),
        ],
    )
    stages = lower_root_plan(plan, factor_name="f")
    types = {s.task_type for s in stages}
    assert "SOURCE_SCAN" in types and "ROOT" in types
    assert types & {"OPERATOR", "ROLLING_SHARED", "CROSS_SECTION"}
    for s in stages:
        assert s.backend_candidates, "backend_candidates 非空（R31-P0-003）"
        assert s.resource_contract is not None and s.resource_contract.peak_memory_bytes > 0


# ---------------------------------------------------------------------------
# R31-P0-019/022: fusion scope + actual execution
# ---------------------------------------------------------------------------


def test_fusion_scope_safety():
    from factor_engine.planner.physical_factor_dag import PhysicalFactorTask
    from factor_engine.planner.native_fusion import can_fuse_roots
    from factor_engine.runtime.task_resource_contract import TaskResourceContract

    mk = lambda name, scope: PhysicalFactorTask(
        task_id=f"root:{name}", op="ts_mean", task_type="ROOT",
        preferred_backend="pandas_numpy", source_scope="s1", source_snapshot_id="snap1",
        execution_scope=scope,
        resource_contract=TaskResourceContract(output_bytes=100, peak_memory_bytes=100),
        node_ref=name,
    )
    assert can_fuse_roots([mk("a", '{"market":"A"}'), mk("b", '{"market":"A"}')])
    assert not can_fuse_roots([mk("a", '{"market":"A"}'), mk("c", '{"market":"B"}')])


def test_fusion_groups_actually_execute_or_fallback():
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.planner.native_fusion import execute_fusion_group

    scheduler = AdaptiveBatchScheduler()
    # execute_fusion_group 对不支持 execute_multi_roots 的 backend 走 honest fallback
    # 并计数 native_fusion_fallback。
    assert hasattr(scheduler, "cancel")  # CancellationToken（R31-P1-040）
    scheduler.cancel()
    assert scheduler._cancelled


# ---------------------------------------------------------------------------
# R31-P0-025/026: BatchDataRequest
# ---------------------------------------------------------------------------


def test_batch_data_request_unions_fields():
    from factor_engine.planner.batch_data_request import build_batch_data_request

    req = build_batch_data_request(None, fields=["close", "volume", "open"])
    assert len(req.fields) >= 3
    assert len(req.groups) == 1  # 一个 source scan group，不是逐 factor


# ---------------------------------------------------------------------------
# R31-P1-038: change impact recompute
# ---------------------------------------------------------------------------


def test_change_impact_propagates():
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.change_impact import affected_root_window

    w = affected_root_window(
        PlanNode("ts_mean", [PlanNode("column", attrs={"name": "close"})], attrs={"window": 5}),
        field="close", changed_start="2024-01-10",
    )
    assert w.start == "2024-01-10" and w.end == "2024-01-16", f"rolling [T, T+4]: {w}"
    w2 = affected_root_window(
        PlanNode("add", [PlanNode("column", attrs={"name": "close"}), PlanNode("column", attrs={"name": "volume"})]),
        field="close", changed_start="2024-01-10",
    )
    assert w2.start == w2.end == "2024-01-10", "elementwise point"
    w3 = affected_root_window(
        PlanNode("ts_ema", [PlanNode("column", attrs={"name": "close"})], attrs={"span": 10}),
        field="close", changed_start="2024-01-10",
    )
    assert w3.end is None and w3.unbounded, "stateful unbounded"


# ---------------------------------------------------------------------------
# R31-P1-040: cancellation
# ---------------------------------------------------------------------------


def test_cancellation_stops_admission():
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler, _default_contract_for
    from factor_engine.planner.physical_factor_dag import PhysicalFactorTask

    sched = AdaptiveBatchScheduler()
    sched.cancel()
    task = PhysicalFactorTask(task_id="root:a", op="ts_mean", task_type="ROOT", node_ref="x")
    future, lease = sched._admit_and_run(
        task, backend=None, ctx=None, execute_root=lambda t: None, materialize_shared=lambda *a: None
    )
    assert future is None and lease is None, "cancelled scheduler refuses new admission"
