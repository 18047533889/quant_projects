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
#: R38 P0-001（§4）：真实 shard 子任务 / merge barrier 任务。
TASK_SHARD = "SHARD"
TASK_MERGE = "MERGE"

#: 真正产生 IO/计算工作的 task 类型（其余为规划视图，见 ``executable``）。
_EXECUTABLE_TASK_TYPES = frozenset({
    TASK_SOURCE_SCAN,
    TASK_CSE_SHARED,
    TASK_ROOT,
    TASK_WRITE,
    TASK_SHARD,
    TASK_MERGE,
})

#: 规划视图（barrier 分割产物）：真实计算发生在 ROOT 内部，这些 stage 只承载
#: cost / backend / read-wave / explain 语义，**不**消耗真实 resource lease。
_PLANNING_VIEW_TASK_TYPES = frozenset({
    TASK_OPERATOR,
    TASK_ROLLING_SHARED,
    TASK_GROUP,
    TASK_CROSS_SECTION,
    TASK_STATEFUL,
    TASK_SOURCE_JOIN,
    TASK_SOURCE_AGG,
})


def task_is_executable(task_type: str) -> bool:
    """R33-P0-008：SOURCE_SCAN / CSE_SHARED / ROOT / WRITE 是真实工作单元，
    barrier 分割产生的 planning view（OPERATOR/ROLLING/GROUP/CS/STATEFUL/
    JOIN/AGG）不占真实 resource lease。"""
    return task_type in _EXECUTABLE_TASK_TYPES


@dataclass(frozen=True)
class SourceScopeId:
    """R33-P0-005/006/013：typed source 身份——禁止字符串 ``split("::")``。

    ``dataset`` / ``snapshot_id`` / ``market`` 是独立 typed 字段；``key()`` 是
    唯一 canonical 字符串（内部排序/去重用），业务身份解析永远读字段本身。
    """

    dataset: str = ""
    snapshot_id: str = ""
    market: str = ""
    params_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", str(self.dataset))
        object.__setattr__(self, "snapshot_id", str(self.snapshot_id))
        object.__setattr__(self, "market", str(self.market))
        object.__setattr__(self, "params_digest", str(self.params_digest))

    def key(self) -> str:
        parts = [f"dataset:{self.dataset}"]
        if self.snapshot_id:
            parts.append(f"snapshot:{self.snapshot_id}")
        if self.market:
            parts.append(f"market:{self.market}")
        if self.params_digest:
            parts.append(f"params:{self.params_digest}")
        return "::".join(parts)

    def __str__(self) -> str:
        return self.key()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "snapshot_id": self.snapshot_id,
            "market": self.market,
            "params_digest": self.params_digest,
            "key": self.key(),
        }


def source_scope_from_key(source_scope: str) -> SourceScopeId:
    """从旧字符串 scope 恢复 typed 身份（向后兼容读入；新代码禁止用字符串构造）。

    只解析已知 ``dataset:``/``snapshot:``/``market:`` 前缀，其余部分原样保留在
    ``params_digest``，绝不把业务身份从中间拆出来当 dataset。
    """
    dataset = ""
    snapshot_id = ""
    market = ""
    params: list[str] = []
    for part in str(source_scope).split("::"):
        if not part:
            continue
        if part.startswith("dataset:"):
            dataset = part[len("dataset:"):]
        elif part.startswith("snapshot:"):
            snapshot_id = part[len("snapshot:"):]
        elif part.startswith("market:"):
            market = part[len("market:"):]
        else:
            params.append(part)
    return SourceScopeId(
        dataset=dataset,
        snapshot_id=snapshot_id,
        market=market,
        params_digest="::".join(params),
    )


@dataclass(frozen=True)
class SourceScanSpec:
    """R33-P0-009：SOURCE_SCAN task 的真实 IO 需求（不再丢失）。

    ``required_columns`` 是**物理列**（读 wave / ScanCost / admission 都消费它，
    不是 ``(task.op,)`` 推断）。
    ``time_range`` 是真实加载窗口（warmup 扩展后的 actual load 起点）。
    """

    dataset: str
    required_columns: tuple[str, ...]
    time_range: tuple[str, str] | None = None
    instrument_scope: tuple[str, ...] | None = None
    universe_id: str | None = None
    filters_digest: str = ""
    source_params_digest: str = ""
    snapshot_id: str = ""
    expected_rows: int = 0
    expected_bytes: int = 0
    projected_bytes: int = 0
    remote: bool = False
    ordering: str = ""  # "instrument,time:asc" 或 ""
    output_representation: str = "pandas_column"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "required_columns": list(self.required_columns),
            "time_range": list(self.time_range) if self.time_range else None,
            "instrument_scope": list(self.instrument_scope) if self.instrument_scope else None,
            "universe_id": self.universe_id,
            "filters_digest": self.filters_digest,
            "source_params_digest": self.source_params_digest,
            "snapshot_id": self.snapshot_id,
            "expected_rows": self.expected_rows,
            "expected_bytes": self.expected_bytes,
            "projected_bytes": self.projected_bytes,
            "remote": self.remote,
            "ordering": self.ordering,
            "output_representation": self.output_representation,
        }

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
    # R33-P0-009/010/011：SOURCE_SCAN 的真实 IO 需求与执行性标记。
    source_scan_spec: SourceScanSpec | None = None
    required_columns: tuple[str, ...] = ()
    time_range: tuple[str, str] | None = None
    instrument_scope: tuple[str, ...] | None = None
    executable: bool = True  # False = barrier 规划视图（不占真实 resource lease）
    # R38 P0-001（§4）：真实 shard 执行元数据。shard 子任务携带 descriptor；
    # merge 任务携带 ShardExecutionPlan（含 merge barrier 与 merge contract）。
    shard_descriptor: Any = None
    shard_plan: Any = None

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
            "source_scan_spec": self.source_scan_spec.to_dict() if self.source_scan_spec else None,
            "required_columns": list(self.required_columns),
            "time_range": list(self.time_range) if self.time_range else None,
            "instrument_scope": list(self.instrument_scope) if self.instrument_scope else None,
            "executable": self.executable,
            "shard_descriptor": (
                self.shard_descriptor.to_dict() if self.shard_descriptor is not None else None
            ),
            "shard_plan": self.shard_plan.to_dict() if self.shard_plan is not None else None,
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


def rebase_task(task: PhysicalFactorTask, **overrides: Any) -> PhysicalFactorTask:
    """重建一个 frozen :class:`PhysicalFactorTask`，保留全部既有字段 + 覆盖项。"""
    from dataclasses import replace

    return replace(task, **overrides)
