# -*- coding: utf-8 -*-
"""R39-PERF-016：PlanExecutionCertificate —— 编译期执行计划证书。

当前问题（§5 PERF-016）
    scheduler ``plan()`` 已经为每个 task 算出 plan cost（``_plan_cost_bytes`` /
    ``task.estimated_cost``）；运行结束又可能在 batch_service 里
    ``estimate_plan_cost(fp.root)`` / ``summarize_plans`` / ``derive_scheduling_hints``
    重新遍历整棵 DAG，纯粹为了 telemetry。

本模块把规划期已算出的信息固化进 **frozen** 证书：
    structural_hash / bound_ops / cost / backend_eligibility / output_shape /
    history / estimated_memory。

调度器运行期 O(1) 读取证书（``task_execution_certificate``），不再重走 DAG；
batch_service 侧（跳过运行后重走）是另一 cluster 的职责——这里只负责构建 +
    暴露 + 在 scheduler 热路径消费。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanExecutionCertificate:
    """单个 task 在 plan() 阶段的执行证书（R39-PERF-016）。

    ``cost`` 是 ``backend.operator_cost.estimate_plan_cost`` 的摘要 dict
    （含 ``total_work`` / ``peak_live_memory_bytes`` / ``node_count`` / ...）。
    """

    #: TypedIRStructuralHash（``planner.plan_hash.typed_ir_structural_hash``）。
    structural_hash: str = ""
    #: plan 中出现的算子集合（保序去重）。
    bound_ops: tuple[str, ...] = ()
    #: plan cost 摘要 dict（同 ``_plan_cost_bytes`` 输出）。
    cost: dict[str, Any] = field(default_factory=dict)
    #: backend eligibility（task.backend_candidates 或 preferred_backend）。
    backend_eligibility: tuple[str, ...] = ()
    #: 输出 shape ``(rows, cols)``（未知为 None）。
    output_shape: tuple[int, int] | None = None
    #: 历史窗口 ``(start, end)``（task.time_range，未知为 None）。
    history: tuple[str, str] | None = None
    #: 峰值内存字节（cost.peak_live_memory_bytes）。
    estimated_memory: int = 0

    def total_work(self) -> float:
        """O(1) 读取估计 work（替代运行后重走 DAG）。"""
        return float(self.cost.get("total_work", 0.0))

    def peak_memory_bytes(self) -> int:
        """O(1) 读取峰值内存字节。"""
        return int(self.cost.get("peak_live_memory_bytes", self.estimated_memory))

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_hash": self.structural_hash,
            "bound_ops": list(self.bound_ops),
            "cost": self.cost,
            "backend_eligibility": list(self.backend_eligibility),
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "history": list(self.history) if self.history else None,
            "estimated_memory": self.estimated_memory,
        }


def _collect_bound_ops(plan: Any) -> tuple[str, ...]:
    """递归收集 plan 子树中的算子名（带 seen 防 DAG 环）。"""
    ops: set[str] = set()
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        op = str(getattr(node, "op", "") or "")
        if op:
            ops.add(op)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    return tuple(sorted(ops))


def build_plan_execution_certificate(
    task: Any, plan_cost: dict[str, Any]
) -> PlanExecutionCertificate:
    """从 ``PhysicalFactorTask`` + 已算好的 plan cost 构造证书（编译期一次）。

    所有字段构造都是尽力而为——失败只降级为默认值，绝不阻断 scheduling。
    """
    node = getattr(task, "node_ref", None)
    plan = getattr(node, "root", node) if node is not None else None
    structural_hash = ""
    bound_ops: tuple[str, ...] = ()
    output_shape: tuple[int, int] | None = None
    if plan is not None:
        try:
            from planner.plan_hash import typed_ir_structural_hash

            structural_hash = typed_ir_structural_hash(plan)
        except Exception:
            structural_hash = ""
        bound_ops = _collect_bound_ops(plan)
        try:
            shape = getattr(plan, "shape", None)
            if shape is not None and len(shape) >= 2:
                output_shape = (int(shape[0]), int(shape[1]))
        except Exception:
            output_shape = None
    cost = dict(plan_cost) if isinstance(plan_cost, dict) else {}
    backend_eligibility = tuple(getattr(task, "backend_candidates", ()) or ()) or (
        getattr(task, "preferred_backend", "pandas_numpy"),
    )
    history: tuple[str, str] | None = None
    tr = getattr(task, "time_range", None)
    if tr:
        try:
            history = (str(tr[0]), str(tr[1]))
        except Exception:
            history = None
    estimated_memory = int(cost.get("peak_live_memory_bytes", 0)) if cost else 0
    return PlanExecutionCertificate(
        structural_hash=structural_hash,
        bound_ops=bound_ops,
        cost=cost,
        backend_eligibility=backend_eligibility,
        output_shape=output_shape,
        history=history,
        estimated_memory=estimated_memory,
    )
