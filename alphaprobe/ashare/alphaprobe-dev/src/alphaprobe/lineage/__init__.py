"""lineage（任务书 §37）：FactorNode 唯一 + ActionEdge 多父。

跨轮再发现 B：仍连接到同一个 B 节点，不因 round 不同复制 factor。
（持久层在 memory.GlobalMemoryStore；本模块提供图视图。）

plan Task 15（追加）：lineage 图视图旁挂 **trajectory 记录接线**——
``LineageTrajectoryTracker`` 用 search/trajectory 的 :class:`SearchTrajectory`
按 lineage_key 聚合每个 branch 的生成步骤（per-step ΔFitness/ΔPool/novelty/
complexity/cost/failure_reason），并把 ``trajectory_repair_requested`` 暴露给
宏范式调度（paradigm_scheduler 的 TRAJECTORY_REPAIR 资格闸门）。本模块只做
**进程内记录聚合**，持久化复用 memory/ledger 既有 API（不新建表）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionEdge:
    """§37。支持 A + B -> C 多父。"""

    action_id: str
    action_type: str
    parent_factor_ids: list[str]
    child_factor_id: str
    round_id: str
    campaign_id: str
    generation: int
    prompt_version: str = ""
    llm_model: str | None = None
    cost: float = 0.0
    latency_ms: int = 0
    outcome: str = "ok"


@dataclass
class LineageView:
    """只读图视图：active subgraph 才进 RAM（§38）。"""

    store: Any  # GlobalMemoryStore

    def parents_of(self, factor_id: str) -> list[str]:
        return [
            e["parent_id"] for e in self.store.lineage_parents(factor_id)
        ]

    def multi_parent_ok(self, child_factor_id: str) -> bool:
        """§89.5：A + B -> C，lineage 查询必须同时返回两个 parent。"""
        edges = self.store.lineage_parents(child_factor_id)
        parent_ids = {e["parent_id"] for e in edges}
        return len(parent_ids) >= 2

    def depth_of(self, factor_id: str, max_depth: int = 100) -> int:
        seen: set[str] = set()
        frontier = [factor_id]
        depth = 0
        while frontier and depth < max_depth:
            nxt: list[str] = []
            for fid in frontier:
                if fid in seen:
                    continue
                seen.add(fid)
                nxt.extend(self.parents_of(fid))
            if not nxt:
                break
            frontier = nxt
            depth += 1
        return depth


# ---------------------------------------------------------------------------
# plan Task 15：trajectory 记录接线（进程内聚合；持久化复用 memory/ledger）
# ---------------------------------------------------------------------------


class LineageTrajectoryTracker:
    """按 lineage_key 聚合轨迹步骤，并把 repair 信号暴露给范式调度。

    用法（生成链每落一个 candidate / 失败一次就 append 一条 step）：

        tracker = LineageTrajectoryTracker()
        tracker.observe(
            lineage_key="root_factor_id",
            factor_id=child_id, parent_id=parent_id,
            action_type="REFINE",
            delta_fitness=0.02, delta_pool=0.01, novelty=0.05,
            cost=0.1,
        )

    读接口：

    - ``trajectory_of(lineage_key)`` → :class:`SearchTrajectory`（含 per-step
      字段与局部 action 饱和度）。
    - ``critic_verdict_of(lineage_key, stagnation=None, critic_condition=False)``
      → 复用 :class:`TrajectoryCritic`；``critic_condition`` 可注入外部停滞/
      critic 信号。
    - ``repair_requested(lineage_key)`` → 供 paradigm_scheduler 的
      TRAJECTORY_REPAIR 资格闸门读（critic 建议 repair 才 True；无坏边/无
      停滞证据 → False，#13）。

    惰性 import search.trajectory（本模块顶层不 import 兄弟包，保持 lineage
    图视图零额外依赖——与 bayesian_retriever 的惰性 import 同风格）。
    """

    def __init__(self) -> None:
        from alphaprobe.search.trajectory import SearchTrajectory

        self._trajectories: dict[str, SearchTrajectory] = {}
        self._trajectory_cls = SearchTrajectory

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------

    def observe(
        self,
        *,
        lineage_key: str,
        factor_id: str = "",
        parent_id: str = "",
        action_type: str = "",
        action_id: str = "",
        delta_fitness: float | None = None,
        delta_pool: float | None = None,
        novelty: float | None = None,
        complexity: int = 0,
        cost: float = 0.0,
        failure_reason: str = "",
    ) -> Any:
        """往 lineage_key 的轨迹追加一步（返回最新 SearchTrajectory）。"""
        t = self._trajectories.setdefault(
            str(lineage_key), self._trajectory_cls(lineage_key=str(lineage_key))
        )
        return t.append_record(
            factor_id=factor_id,
            parent_id=parent_id,
            action_type=action_type,
            action_id=action_id,
            delta_fitness=delta_fitness,
            delta_pool=delta_pool,
            novelty=novelty,
            complexity=complexity,
            cost=cost,
            failure_reason=failure_reason,
        )

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------

    def trajectory_of(self, lineage_key: str) -> Any:
        """返回该 lineage 的 SearchTrajectory（无 → 空轨迹，不抛）。"""
        return self._trajectories.setdefault(
            str(lineage_key), self._trajectory_cls(lineage_key=str(lineage_key))
        )

    def critic_verdict_of(
        self,
        lineage_key: str,
        *,
        stagnation: float | None = None,
        critic_condition: bool = False,
    ) -> Any:
        """该 lineage 的 critic 判定（坏边隔离 + repair 建议）。"""
        from alphaprobe.search.trajectory import repair_target_of

        return repair_target_of(
            self.trajectory_of(lineage_key),
            stagnation=stagnation,
            critic_condition=critic_condition,
        )

    def repair_requested(
        self,
        lineage_key: str,
        *,
        stagnation: float | None = None,
        critic_condition: bool = False,
    ) -> bool:
        """critic 是否建议 repair（供 TRAJECTORY_REPAIR 范式资格闸门）。"""
        verdict = self.critic_verdict_of(
            lineage_key, stagnation=stagnation, critic_condition=critic_condition
        )
        return bool(getattr(verdict, "suggests_repair", False))

    def lineage_keys(self) -> list[str]:
        return sorted(self._trajectories)

    def reset(self) -> None:
        self._trajectories = {}


#: 与 action_edges 的 ActionEdge 对齐的字段名映射（记录接线方拼 dict 用）
LINEAGE_TRAJECTORY_FIELDS = (
    "lineage_key", "factor_id", "parent_id", "action_type", "action_id",
    "delta_fitness", "delta_pool", "novelty", "complexity", "cost",
    "failure_reason",
)