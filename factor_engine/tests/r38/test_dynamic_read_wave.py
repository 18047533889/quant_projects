# -*- coding: utf-8 -*-
"""R38 P0-041（§16）：运行中 read-wave JIT repartition。

行为探针（§R38_DYNAMIC_READ_WAVE_SHRINK）：
    - wave 预算显著缩小 → 对未执行 SOURCE_SCAN 重新生成更小 wave；
    - 已执行 wave 覆盖的 task 不再重建（不重复扫）；
    - 新 wave 的 wave_id 与已执行不冲突；
    - 预算未显著缩小 / 无未执行 task → 不重建。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from factor_engine.planner.physical_factor_dag import (
    TASK_SOURCE_SCAN,
    PhysicalFactorDAG,
    PhysicalFactorTask,
)
from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
from factor_engine.runtime.resource_autopilot import ResourceDecision
from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.resource_errors import ResourceBudgetExceeded


def _source_task(tid: str, n_cols: int) -> PhysicalFactorTask:
    """一个 source scan task，携带 n_cols 个列（列越多 wave 内存越大）。"""
    return PhysicalFactorTask(
        task_id=tid,
        op="source_scan",
        task_type=TASK_SOURCE_SCAN,
        required_columns=tuple(f"c{tid}_{j}" for j in range(n_cols)),
        time_range=("2024-01-01", "2024-06-30"),
        source_scope="market:us",
        executable=True,
    )


def _decision(read_wave: int) -> ResourceDecision:
    return ResourceDecision(
        target_concurrency=4, target_cpu_tokens=4,
        read_wave_bytes=read_wave, factor_block_bytes=64 * 1024**2,
        result_queue_bytes=128 * 1024**2, io_concurrency=1,
        remote_concurrency=1, cache_budget_bytes=128 * 1024**2,
        spill_budget_bytes=0, pressure_state="PRESSURE_3",
        memory_constrained=True, reasons=("pressure",),
    )


def _plan_and_sched() -> tuple[SimpleNamespace, AdaptiveBatchScheduler, PhysicalFactorDAG]:
    dag = PhysicalFactorDAG()
    for i in range(6):
        # 每个 task 200 列 → 500k 行 × 200 列 × 8B ≈ 800MB/task。
        dag.tasks[f"src:{i}"] = _source_task(f"src:{i}", 200)
    plan = SimpleNamespace(
        read_waves=SimpleNamespace(
            waves=[SimpleNamespace(wave_id=i, task_ids=(f"src:{i}",),
                                   columns=frozenset(), estimated_memory_bytes=0)
                   for i in range(6)]
        ),
    )
    sched = AdaptiveBatchScheduler(broker=ResourceBroker())
    return plan, sched, dag


def test_wave_budget_shrink_repartitions_unexecuted():
    plan, sched, dag = _plan_and_sched()
    # 大预算 → 1 个 wave 装下全部；模拟 src:0..2 已执行。
    sched._wave_refs = {0: "ref0", 1: "ref1", 2: "ref2"}
    sched._wave_covered_tasks = {f"src:{i}" for i in range(3)}
    sched._last_wave_budget = 4 * 1024**3
    sched._last_decision = _decision(read_wave=512 * 1024**2)  # 显著缩小
    committed = set(sched._wave_covered_tasks)

    # A single 800 MB projected column cannot fit a 512 MiB wave. Splitting by
    # task cannot make that atomic read safe, so repartition must fail closed.
    with pytest.raises(ResourceBudgetExceeded, match="atomic read request"):
        sched._maybe_repartition_waves(plan, dag, committed)


def test_no_repartition_when_budget_not_shrunk():
    plan, sched, dag = _plan_and_sched()
    rw = plan.read_waves
    sched._last_wave_budget = 4 * 1024**3
    sched._last_decision = _decision(read_wave=3 * 1024**3)  # 只缩 25% < 40%
    sched._maybe_repartition_waves(plan, dag, set())
    assert plan.read_waves is rw  # 未重排（同一 plan 对象）


def test_no_repartition_when_all_covered():
    plan, sched, dag = _plan_and_sched()
    sched._wave_covered_tasks = {f"src:{i}" for i in range(6)}
    sched._last_wave_budget = 4 * 1024**3
    sched._last_decision = _decision(read_wave=512 * 1024**2)
    rw = plan.read_waves
    sched._maybe_repartition_waves(plan, dag, set(sched._wave_covered_tasks))
    assert plan.read_waves is rw  # 无未执行 task → 不重建
