# -*- coding: utf-8 -*-
"""R39-PERF-018：MicroBatchTask —— scheduler dispatch 合并（不是 native fusion）。

场景（§6 PERF-018）
    大量低成本 root（``rank / add / div / lag / 简单 rolling``）即使整个 batch
    不满足 DIRECT_VECTOR，每个 root 一个 ``concurrent.futures.Future`` 也非常
    浪费（Future 创建 + 提交 + 事件簿记 + 结果解包）。

    本模块把 16–128 个共享 backend + axis + source buffers 的低成本 root 合并进
    **一个** Future，在 Future 内串行执行（错误隔离：单个 root 失败不阻断同批
    其它 root）。

与 backend native fusion 的区别
    - native fusion = 一个物理表达式/SQL（``planner/native_fusion.py``）；
    - microbatch = scheduler dispatch 合并（本模块）。

``dispatch_micro_batch`` 是模块级函数（可 pickle，供 thread/process pool 使用）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from planner.physical_factor_dag import PhysicalFactorTask


@dataclass(frozen=True)
class MicroBatchTask:
    """一个 micro-batch：一个 Future 内串行执行一组低成本 root。

    ``roots``：root task_id 元组（确定性排序）。
    ``backend``：组内统一 backend（调度器 dispatch 时使用）。
    ``same_backend`` / ``same_axis`` / ``same_source_buffers``：合并前确认的
    共享性质（backend=preferred_backend；axis=execution_scope；source
    buffers=source_scope/snapshot）。
    ``estimated_total_work``：组内 root 估计 work 总和（admission 阈值用）。
    """

    roots: tuple[str, ...]
    backend: str = "pandas_numpy"
    same_backend: bool = True
    same_axis: bool = True
    same_source_buffers: bool = True
    estimated_total_work: float = 0.0

    @property
    def root_count(self) -> int:
        return len(self.roots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_count": len(self.roots),
            "roots": list(self.roots),
            "backend": self.backend,
            "same_backend": self.same_backend,
            "same_axis": self.same_axis,
            "same_source_buffers": self.same_source_buffers,
            "estimated_total_work": round(self.estimated_total_work, 3),
        }


def dispatch_micro_batch(
    roots: tuple[str, ...],
    task_by_id: dict[str, PhysicalFactorTask],
    execute_root: Callable[[PhysicalFactorTask], Any],
) -> tuple[dict[str, Any], dict[str, BaseException]]:
    """串行执行 micro-batch 内的每个 root（模块级，可 pickle）。

    **错误隔离**（PERF-018 硬要求）：某个 root 失败不阻断同批其它 root——每个
    root 独立 try/except，失败记入 ``failures``，其余 root 正常执行并返回。

    返回 ``(results_by_root, failures_by_root)``：
        results:  ``{task_id: result}``
        failures: ``{task_id: exception}``
    """
    from runtime.execution_traits import thread_budget

    results: dict[str, Any] = {}
    failures: dict[str, BaseException] = {}
    for tid in roots:
        task = task_by_id.get(tid)
        if task is None:
            failures[tid] = RuntimeError(f"micro-batch: unknown task {tid}")
            continue
        # R35：root 内部 BLAS/OpenMP 线程钉在 broker 给的 CPU budget
        #（FE_workers × inner_threads 不 oversubscribe host）。
        budget = 1
        try:
            contract = task.resource_contract
            if contract is not None:
                bt = getattr(contract, "backend_threads", None)
                if bt and int(bt) >= 1:
                    budget = int(bt)
        except Exception:
            budget = 1
        try:
            with thread_budget(budget):
                results[tid] = execute_root(task)
        except Exception as exc:  # noqa: BLE001 — per-root 隔离
            failures[tid] = exc
    return results, failures
