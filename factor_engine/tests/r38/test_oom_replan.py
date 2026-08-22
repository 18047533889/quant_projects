# -*- coding: utf-8 -*-
"""R38 P0-004/005/006（§5）：OOM 必须驱动真实 smaller-shape replan。

行为探针：
    - OOM（MemoryError / DuckDB OutOfMemory / Arrow / Polars）→ ERROR_OOM 分类；
    - semantic / schema / PIT 错误不误判为 OOM（§R38-P0-006）；
    - ``_replace_with_shard_plan`` 真实把 ROOT 换成 shard children + MERGE；
    - ``_handle_oom`` 只生成 shape 签名不同的新计划（§R38_ZERO_SAME_SHAPE_OOM_RETRY）；
    - 同一 shape 的失败被永久记入 ``_failed_shapes``，禁止重试。
"""
from __future__ import annotations

import pandas as pd

from planner.physical_factor_dag import (
    TASK_MERGE,
    TASK_ROOT,
    TASK_SHARD,
    PhysicalFactorDAG,
    PhysicalFactorTask,
    rebase_task,
)
from runtime.adaptive_batch_scheduler import ERROR_OOM, AdaptiveBatchScheduler, classify_error
from runtime.resource_broker import ResourceBroker
from runtime.shard_execution_plan import shape_signature
from runtime.task_resource_contract import TaskResourceContract


def _contract(peak: int = 8 * 1024**3, dim: str = "time"):
    return TaskResourceContract(
        predicted_elapsed_ms=1.0,
        cpu_tokens=1,
        peak_memory_bytes=peak,
        output_bytes=peak,
        backend="pandas_numpy",
        backend_threads=1,
        shardable=True,
        shard_dimension=dim,
        estimate_basis="test",
    )


def _root_task(tid: str = "root:f", op: str = "ts_mean") -> PhysicalFactorTask:
    return PhysicalFactorTask(
        task_id=tid,
        op=op,
        task_type=TASK_ROOT,
        factor_name="f",
        resource_contract=_contract(),
        time_range=("2024-01-01", "2024-06-30"),
        instrument_scope=tuple(f"A{i:03d}" for i in range(50)),
        executable=True,
    )


def _dag_with_root() -> PhysicalFactorDAG:
    dag = PhysicalFactorDAG()
    dag.tasks["root:f"] = _root_task()
    dag.roots = ("root:f",)
    return dag


# ---------------------------------------------------------------------------
# OOM 分类（§R38-P0-006：不误伤 semantic/schema/PIT）
# ---------------------------------------------------------------------------


def test_oom_classified_separately():
    assert classify_error(MemoryError("out of memory")) == ERROR_OOM
    assert classify_error(RuntimeError("duckdb OutOfMemoryException")) == ERROR_OOM
    assert classify_error(RuntimeError("ArrowMemoryError: allocation failed")) == ERROR_OOM
    assert classify_error(RuntimeError("numpy.core._exceptions._ArrayMemoryError")) == ERROR_OOM


def test_semantic_schema_not_oom():
    assert classify_error(RuntimeError("schema mismatch: expected 3 columns")) == "permanent"
    assert classify_error(RuntimeError("PIT violation: future data")) == "permanent"
    assert classify_error(RuntimeError("invalid parameter window=-5")) == "permanent"
    assert classify_error(RuntimeError("unsupported operator foo")) == "permanent"


# ---------------------------------------------------------------------------
# 真实 DAG 替换（§R38_REAL_AUTOSHARD_EXECUTION_PASS）
# ---------------------------------------------------------------------------


def test_replace_with_shard_plan_builds_shard_children_and_merge():
    from runtime.auto_shard_planner import AutoShardPlanner

    dag = _dag_with_root()
    task = dag.tasks["root:f"]
    planner = AutoShardPlanner(min_shards=2)
    plan = planner.build_shard_execution_plan(
        task,
        safe_envelope_bytes=2 * 1024**3,
        time_range=task.time_range,
        instrument_universe=list(task.instrument_scope),
        lookback_bars=20,
    )
    assert plan is not None
    sched = AdaptiveBatchScheduler(broker=ResourceBroker())
    ok = sched._replace_with_shard_plan(dag, "root:f", plan)
    assert ok
    # shard children + merge barrier 都真实存在。
    for sid in plan.shard_task_ids:
        t = dag.tasks[sid]
        assert t.task_type == TASK_SHARD
        assert t.shard_descriptor is not None
    m = dag.tasks[plan.merge_task_id]
    assert m.task_type == TASK_MERGE
    assert m.shard_plan is plan
    assert set(m.inputs) == set(plan.shard_task_ids)
    # roots 指向 merge。
    assert dag.roots == (plan.merge_task_id,)
    # 原 task 不再存在。
    assert "root:f" not in dag.tasks
    # 拓扑序无环。
    dag.topological_order()


# ---------------------------------------------------------------------------
# OOM replan：smaller shape 且签名不同（§R38_ZERO_SAME_SHAPE_OOM_RETRY）
# ---------------------------------------------------------------------------


def test_handle_oom_replans_to_different_signature():
    from runtime.auto_shard_planner import AutoShardPlanner

    dag = _dag_with_root()
    task = dag.tasks["root:f"]
    planner = AutoShardPlanner(min_shards=2)
    plan = planner.build_shard_execution_plan(
        task,
        safe_envelope_bytes=2 * 1024**3,
        time_range=task.time_range,
        instrument_universe=list(task.instrument_scope),
        lookback_bars=20,
    )
    sched = AdaptiveBatchScheduler(broker=ResourceBroker())
    sched._replace_with_shard_plan(dag, "root:f", plan)
    # shape 身份统一按 original_task_id 存（P0-009，不再 merge_id/orig_tid 混用）。
    failed_sig = sched._shard_shape_of["root:f"]

    remaining = set()
    # 模拟 merge 任务 OOM → _handle_oom 必须 replan 到更小 shape。
    handled = sched._handle_oom(plan.merge_task_id, MemoryError("oom"), dag, remaining)
    assert handled is True
    # 新 merge 带 attempt 段（P0-010 attempt 隔离），shape 签名 != 失败签名。
    new_merge_id = f"root:f:attempt:2:merge"
    assert new_merge_id in dag.tasks
    new_sig = sched._shard_shape_of.get("root:f")
    assert new_sig is not None and new_sig != failed_sig
    assert "root:f" not in dag.tasks
    # 旧 attempt 的 task id 已被替换，不会残留。
    assert all(not t.startswith("root:f:attempt:1:") for t in dag.tasks)
    # 新 shard 任务被重新加入 remaining。
    assert len(remaining) >= 2


def test_same_shape_oom_retry_refused():
    from runtime.auto_shard_planner import AutoShardPlanner

    dag = _dag_with_root()
    task = dag.tasks["root:f"]
    planner = AutoShardPlanner(min_shards=2)
    plan = planner.build_shard_execution_plan(
        task,
        safe_envelope_bytes=2 * 1024**3,
        time_range=task.time_range,
        instrument_universe=list(task.instrument_scope),
        lookback_bars=20,
    )
    sched = AdaptiveBatchScheduler(broker=ResourceBroker())
    sched._replace_with_shard_plan(dag, "root:f", plan)
    failed_sig = sched._shard_shape_of["root:f"]
    # 预置该 shape 已经失败 → 再次 OOM 必须拒绝 replan（不陷入无限重试）。
    sched._failed_shapes.add(failed_sig)
    remaining = set()
    assert sched._handle_oom("root:f:shard:0", MemoryError("oom"), dag, remaining) is False


def test_shape_signature_invariant_direct():
    s1 = shape_signature(task_id="t", dimension="time", shard_count=2, per_shard_peak_bytes=100)
    s2 = shape_signature(task_id="t", dimension="time", shard_count=4, per_shard_peak_bytes=50)
    s3 = shape_signature(task_id="t", dimension="time", shard_count=2, per_shard_peak_bytes=100)
    assert s1 == s3
    assert s1 != s2
