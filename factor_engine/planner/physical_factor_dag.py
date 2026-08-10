# -*- coding: utf-8 -*-
"""R27-004..006: PhysicalFactorDAG —— 真实可调度 DAG（不是 roots + shared_nodes 两层）。

升级点（R27-001/002/006）
    - 所有 source scan / transform / CSE / rolling / group / root 都成为可调度
      DAG Task，每 task 知道 predecessor / consumers / cost / memory / backend /
      output bytes / spillable / shardable。
    - **禁止**「shared nodes 一定先全部算完」：某 shared node 的 predecessor
      ready 且资源可 admission 就立刻运行（R27-006）。
    - task 类型（R27-005）：SOURCE_SCAN / SOURCE_JOIN / SOURCE_AGG / OPERATOR /
      CSE_SHARED / ROLLING_SHARED / GROUP / CROSS_SECTION / STATEFUL / ROOT / WRITE。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime.task_resource_contract import TaskResourceContract

#: Task 类型（R27-005）
TASK_SOURCE_SCAN = "SOURCE_SCAN"
TASK_SOURCE_JOIN = "SOURCE_JOIN"
TASK_SOURCE_AGG = "SOURCE_AGG"
TASK_OPERATOR = "OPERATOR"
TASK_CSE_SHARED = "CSE_SHARED"
TASK_ROLLING_SHARED = "ROLLING_SHARED"
TASK_GROUP = "GROUP"
TASK_CROSS_SECTION = "CROSS_SECTION"
TASK_STATEFUL = "STATEFUL"
TASK_ROOT = "ROOT"
TASK_WRITE = "WRITE"

ALL_TASK_TYPES = (
    TASK_SOURCE_SCAN,
    TASK_SOURCE_JOIN,
    TASK_SOURCE_AGG,
    TASK_OPERATOR,
    TASK_CSE_SHARED,
    TASK_ROLLING_SHARED,
    TASK_GROUP,
    TASK_CROSS_SECTION,
    TASK_STATEFUL,
    TASK_ROOT,
    TASK_WRITE,
)

#: R27-090/084：semantic shardability。跨截面 / 全市场 / group 的算子不能按
#: asset 切（切了结果变）；纯 instrument-separable 时序因子才可 asset shard。
SHARD_TIME_SAFE = "TIME_SHARD_SAFE"
SHARD_ASSET_SAFE = "ASSET_SHARD_SAFE"
SHARD_GROUP_SAFE = "GROUP_SHARD_SAFE"
SHARD_FACTOR_SAFE = "FACTOR_SHARD_SAFE"
SHARD_SESSION_SAFE = "SESSION_SHARD_SAFE"
SHARD_STATEFUL_CHECKPOINT = "STATEFUL_CHECKPOINT_REQUIRED"


@dataclass(frozen=True)
class ShardSpec:
    """Task 的 shard 规格（R27-085..093）。

    ``dimension`` ∈ {time, asset, group, session, factor, none}。
    ``block_size`` 是每 shard 的目标块大小（自适应：predicted_peak <=
    shard_memory_target 时收缩，R27-092/093）。
    """

    dimension: str = "none"
    legal: bool = False
    block_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "legal": self.legal,
            "block_size": self.block_size,
        }


@dataclass(frozen=True)
class PhysicalFactorTask:
    """可调度 DAG task（R27-004）。

    ``inputs`` = 真正 predecessor（task_id 元组）；``consumers`` = 消费本 task
    的 task_id 元组（供 refcount / 及时释放）。
    """

    task_id: str
    op: str
    task_type: str
    inputs: tuple[str, ...] = ()
    consumers: tuple[str, ...] = ()
    # source scope 身份（R27-160：共享必须同一 market/universe/calendar/snapshot/
    # decision-time policy）
    execution_scope: str = ""
    source_scope: str = ""
    source_snapshot_id: str = ""
    # backend
    backend_candidates: tuple[str, ...] = ()
    preferred_backend: str = "pandas_numpy"
    # cost + resource
    estimated_cost: Any = None
    resource_contract: TaskResourceContract | None = None
    # 语义
    shard_spec: ShardSpec | None = None
    spillable: bool = False
    cacheable: bool = False
    deterministic: bool = True
    # 执行 metadata
    node_ref: Any = None  # 原始 PlanNode / FactorPlan（backend.execute 需要）
    factor_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "op": self.op,
            "task_type": self.task_type,
            "inputs": list(self.inputs),
            "consumers": list(self.consumers),
            "execution_scope": self.execution_scope,
            "source_scope": self.source_scope,
            "source_snapshot_id": self.source_snapshot_id,
            "backend_candidates": list(self.backend_candidates),
            "preferred_backend": self.preferred_backend,
            "resource_contract": self.resource_contract.to_dict() if self.resource_contract else None,
            "shard_spec": self.shard_spec.to_dict() if self.shard_spec else None,
            "spillable": self.spillable,
            "cacheable": self.cacheable,
            "deterministic": self.deterministic,
            "factor_name": self.factor_name,
        }


@dataclass
class PhysicalFactorDAG:
    """真实可调度 DAG。

    R27-006：调度器按 predecessor-ready + resource admission 运行任意 task
    （shared node 不必先全部算完）。``roots`` 是最终输出 task；``writers`` 是
    WriteTask（消费 root 结果，R27-176）。
    """

    tasks: dict[str, PhysicalFactorTask] = field(default_factory=dict)
    roots: tuple[str, ...] = ()
    writers: tuple[str, ...] = ()
    #: 元信息（编译说明 / scope 摘要）
    meta: dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: PhysicalFactorTask) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"duplicate task_id={task.task_id!r}")
        self.tasks[task.task_id] = task

    def predecessors(self, task_id: str) -> list[PhysicalFactorTask]:
        t = self.tasks.get(task_id)
        if t is None:
            return []
        return [self.tasks[i] for i in t.inputs if i in self.tasks]

    def successors(self, task_id: str) -> list[PhysicalFactorTask]:
        t = self.tasks.get(task_id)
        if t is None:
            return []
        return [self.tasks[i] for i in t.consumers if i in self.tasks]

    def ready_tasks(self, committed: set[str]) -> list[str]:
        """predecessor 全部 committed 的 task（R27-175）。"""
        return [
            tid for tid, t in self.tasks.items()
            if tid not in committed and all(p in committed for p in t.inputs)
        ]

    def critical_path_remaining_ms(self, task_id: str, cost_ms: dict[str, float]) -> float:
        """task 后最长剩余路径耗时（R27-007 critical path 优先）。

        R32-P1-048: reverse-topological DP（O(V+E)）—— 不再递归无 memo（DAG
        diamond 结构会指数爆炸）。每次调用线性计算全部 task 的 critical path，
        直接查表返回。
        """
        try:
            order = self.topological_order()
        except RuntimeError:
            order = list(self.tasks.keys())
        memo: dict[str, float] = {}
        for tid in reversed(order):
            t = self.tasks.get(tid)
            if t is None:
                memo[tid] = float(cost_ms.get(tid, 0.0))
                continue
            consumers = [c for c in t.consumers if c in self.tasks]
            if not consumers:
                memo[tid] = float(cost_ms.get(tid, 0.0))
            else:
                memo[tid] = float(cost_ms.get(tid, 0.0)) + max(
                    memo[c] for c in consumers
                )
        return memo.get(task_id, float(cost_ms.get(task_id, 0.0)))

    def topological_order(self) -> list[str]:
        """Kahn 拓扑序（稳定、确定性）。

        R32-P1-049: heapq 取代 ``pop(0) + 反复 sort`` —— 大 DAG 从 O(V²) 降到
        O((V+E)·log V)。
        """
        import heapq

        order: list[str] = []
        indegree = {tid: len(t.inputs) for tid, t in self.tasks.items()}
        ready = [tid for tid, deg in indegree.items() if deg == 0]
        heapq.heapify(ready)
        while ready:
            tid = heapq.heappop(ready)
            order.append(tid)
            for succ in sorted(self.tasks[tid].consumers):
                if succ not in indegree:
                    continue
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    heapq.heappush(ready, succ)
        if len(order) != len(self.tasks):
            raise RuntimeError(
                f"PhysicalFactorDAG has a cycle: {len(order)}/{len(self.tasks)} "
                "tasks ordered"
            )
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_count": len(self.tasks),
            "roots": list(self.roots),
            "writers": list(self.writers),
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "meta": self.meta,
        }
