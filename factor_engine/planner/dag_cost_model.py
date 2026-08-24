# -*- coding: utf-8 -*-
"""R27-007/012/129: DAG 成本模型 —— 统一 TaskCost + critical path 优先级。

设计
    - ``TaskCost`` = SourceScanCost + OperatorComputeCost + ConversionCost +
      MaterializationCost + WriteCost（R27-012）。
    - 每 task 估算 ``estimated_ms / critical_path_remaining_ms / reuse_count /
      output_bytes / recompute_ms / scan_bytes``（R27-007）。
    - 优先级（R27-007/129, HEFT-like 但第一版简单化）：
        priority = critical_path_remaining_ms
                 + λ1 * reuse_saved_ms
                 + λ2 * locality_benefit
                 - λ3 * memory_pressure_cost
                 - λ4 * io_pressure_cost
    - 高 reuse shared node 优先（R27-008）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from factor_engine.planner.physical_factor_dag import (
    TASK_CSE_SHARED,
    TASK_ROLLING_SHARED,
    TASK_SOURCE_SCAN,
)

#: 优先级权重（R27-007 建议公式）
LAMBDA_REUSE = 0.5
LAMBDA_LOCALITY = 0.2
LAMBDA_MEMORY = 0.3
LAMBDA_IO = 0.2


@dataclass(frozen=True)
class TaskCost:
    """统一 task 成本（R27-012）。"""

    estimated_ms: float = 0.0
    source_scan_ms: float = 0.0
    operator_compute_ms: float = 0.0
    conversion_ms: float = 0.0
    materialization_ms: float = 0.0
    write_ms: float = 0.0
    peak_memory_bytes: int = 0
    output_bytes: int = 0
    scan_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_ms": round(self.estimated_ms, 3),
            "source_scan_ms": round(self.source_scan_ms, 3),
            "operator_compute_ms": round(self.operator_compute_ms, 3),
            "conversion_ms": round(self.conversion_ms, 3),
            "materialization_ms": round(self.materialization_ms, 3),
            "write_ms": round(self.write_ms, 3),
            "peak_memory_bytes": self.peak_memory_bytes,
            "output_bytes": self.output_bytes,
            "scan_bytes": self.scan_bytes,
        }


def _from_plan_cost(plan_cost: dict[str, Any]) -> TaskCost:
    """从 ``backend.operator_cost.estimate_plan_cost`` 的 dict 构造 TaskCost。"""
    total_work = float(plan_cost.get("total_work", 0.0))
    peak = int(plan_cost.get("peak_live_memory_bytes", 0))
    return TaskCost(
        estimated_ms=total_work,
        operator_compute_ms=total_work,
        peak_memory_bytes=peak,
    )


def estimate_task_cost(
    task: Any,
    *,
    plan_cost: dict[str, Any] | None = None,
    scan_cost: Any | None = None,
) -> TaskCost:
    """按 task 类型估算 TaskCost（R27-012/013）。

    - SOURCE_SCAN：ScanCost 的 estimate_ms / selected_bytes（R27-011 直接消费）。
    - CSE_SHARED / ROLLING_SHARED：plan cost（估算物化成本）。
    - 其它：plan cost 或默认。
    """
    if plan_cost is not None:
        base = _from_plan_cost(plan_cost)
    else:
        base = TaskCost()
    if scan_cost is not None:
        est_ms = getattr(scan_cost, "estimate_ms", None) or getattr(scan_cost, "engine_startup_ms", 0.0)
        scan_bytes = getattr(scan_cost, "selected_bytes", None) or 0
        return TaskCost(
            estimated_ms=float(est_ms),
            source_scan_ms=float(est_ms),
            operator_compute_ms=base.operator_compute_ms,
            conversion_ms=0.0,
            materialization_ms=0.0,
            write_ms=0.0,
            peak_memory_bytes=int(getattr(scan_cost, "projection_bytes", None) or 0),
            output_bytes=0,
            scan_bytes=int(scan_bytes or 0),
        )
    return base


def reuse_saved_ms(reuse_count: int, recompute_ms: float) -> float:
    """R27-007: ``reuse_saved_ms ≈ (reuse_count - 1) * recompute_ms``。"""
    return max(0.0, (reuse_count - 1) * recompute_ms)


def priority_score(
    *,
    critical_path_remaining_ms: float,
    reuse_count: int,
    recompute_ms: float,
    output_bytes: int,
    locality_benefit: float,
    memory_pressure_cost: float,
    io_pressure_cost: float,
) -> float:
    """R27-007 优先级公式（HEFT-like 第一版）。"""
    return (
        critical_path_remaining_ms
        + LAMBDA_REUSE * reuse_saved_ms(reuse_count, recompute_ms)
        + LAMBDA_LOCALITY * locality_benefit
        - LAMBDA_MEMORY * memory_pressure_cost
        - LAMBDA_IO * io_pressure_cost
    )


def task_priority(
    task: Any,
    *,
    critical_path_remaining_ms: float,
    reuse_count: int,
    recompute_ms: float,
    memory_pressure_cost: float,
    io_pressure_cost: float,
    locality_benefit: float = 0.0,
) -> float:
    """为单个 task 打分（供 ready queue 排序）。"""
    out_bytes = 0
    contract = getattr(task, "resource_contract", None)
    if contract is not None:
        out_bytes = contract.output_bytes
    # 高 reuse shared node 优先（R27-008）。
    if task.task_type in {TASK_CSE_SHARED, TASK_ROLLING_SHARED} and reuse_count >= 2:
        locality_benefit += 1.0
    return priority_score(
        critical_path_remaining_ms=critical_path_remaining_ms,
        reuse_count=reuse_count,
        recompute_ms=recompute_ms,
        output_bytes=out_bytes,
        locality_benefit=locality_benefit,
        memory_pressure_cost=memory_pressure_cost,
        io_pressure_cost=io_pressure_cost,
    )
