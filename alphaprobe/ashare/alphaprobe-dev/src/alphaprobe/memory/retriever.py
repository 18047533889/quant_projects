"""MemoryRetriever（任务书 §41/§76 / plan Task 21 MemoryPacket V2）。

Retriever 是 LLM 与 Global Memory 之间的唯一通道——LLM 不直接读库（§40）。
在 store.build_memory_packet 的基础上补 numerical_neighbors：同 parameter family
最近 5 个 factor_node 的 formula + fitness（§41）。

Task 21 / A12（MemoryPacket V2）
--------------------------------
- 组装点从单 parent 升级为 **multi-parent + roles**：``build_packet`` 新增
  可选 ``parents``（父因子池）与 ``roles``（factor_id → role）。调用方不传时
  行为与旧版一致（single parent = store 节点，roles 空）。
- 默认返回 **MemoryPacketV2**（13 段 + token 预算确定性截断）；既有读取面
  （``.parent`` / ``.to_prompt_text()`` / ``.structural_neighbors`` /
  ``.numerical_neighbors`` …）保持兼容。
- **sealed survival 隔离**（Non-negotiable #23）：``research_version`` 非空时
  只把当前研究版本的 survival 放进 packet（旧版本/被封存版本不进当前 prompt）。
- 全量测试不降：``store.build_memory_packet``（V1 旧路径）保持原样返回 V1
  ``MemoryPacket``；只有 retriever 组装点走 V2。
"""

from __future__ import annotations

import json  # noqa: F401  (保留：_fitness_of 历史实现参考 / 未来 schema 解析)

from alphaprobe.memory import GlobalMemoryStore, MemoryPacket
from alphaprobe.memory import _fitness_of
from alphaprobe.memory.memory_packet import (
    MemoryPacketV2,
    sealed_survival_rows,
)


class MemoryRetriever:
    """构建 LLM working memory 快照。线程安全（读路径，store 自带锁）。

    V2 组装默认启用（``build_packet`` 返回 MemoryPacketV2）。需要旧版 V1
    ``MemoryPacket`` 的调用方（round_manager 的 store.build_memory_packet 直调）
    不受影响——store 方法未改。
    """

    def __init__(self, store: GlobalMemoryStore) -> None:
        self.store = store

    # -- §41 numerical neighbors：同 family 最近 5 个节点 ----------------------

    def numerical_neighbors(
        self,
        factor_id: str,
        *,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """同 parameter family 的最近 k 个节点（排除自身）。

        fitness 优先取 evaluations 表（store._fitness_from_evaluations），
        取不到再 fallback schema_json（_fitness_of）。按 fitness 降序取最近
        5 个；family_id 缺失或无同族节点时返回空表。
        """
        return self.store._numerical_neighbors_of(factor_id, k=k)

    @staticmethod
    def _fitness_of(schema_json: Any) -> float | None:
        """从 factor_nodes.schema_json 提取 fitness（single_ic / 归一化值）。

        保留为纯 schema_json fallback；numerical_neighbors 已优先走
        store._fitness_from_evaluations（evaluations 表）再 fallback 本函数。
        """
        return _fitness_of(schema_json)

    # -- §41 build packet（V2 组装点） ------------------------------------------

    def build_packet(
        self,
        factor_id: str,
        *,
        allowed_fields: list[str] | None = None,
        allowed_operators: list[str] | None = None,
        parents: list[dict[str, Any]] | None = None,
        roles: dict[str, str] | None = None,
        research_version: str | None = None,
        budget_max: int | None = None,
        budget_min: int | None = None,
        neighbors: list[dict[str, Any]] | None = None,
        max_items: int = 5,
        objective: str | None = None,
        logic_schema: dict[str, Any] | None = None,
    ) -> MemoryPacketV2 | None:
        """构建 MemoryPacket V2。factor_id 未知时返回 None（不抛异常）。

        Parameters
        ----------
        factor_id : 主检索起点（store 里的真实 factor_id）。
        parents : 可选多父池（dict 列表）。缺省 = [主 parent 节点]（单父兼容）。
            调用方可在检索后把 parent_selector 的 top-k 采样结果传入。
        roles : 可选 factor_id → role 映射（如 main / context / complement）。
            缺省 {}；主 parent 缺省 role 为 ``main``，其余为 ``parent``。
        research_version : 可选当前研究版本标识。非空时 survival exemplars 只
            保留当前版本（sealed 隔离，#23）；None = 不过滤（与旧版一致）。
        budget_max / budget_min : 可选 token 预算覆盖（缺省 4k/2k）。
        neighbors : 可选结构邻居（factor_assets/检索链传入）。缺省 []。
        objective : 可选研究目标文本（research objective 段）。
        logic_schema : 可选 {logic, schema} 当前层引用。

        Returns
        -------
        MemoryPacketV2 | None
            未知 factor_id → None。V2 同时保留旧版读取面（parent /
            structural_neighbors / numerical_neighbors / to_prompt_text …）。
        """
        node = self.store.get_node(factor_id)
        if node is None:
            return None
        parent_pool = list(parents) if parents else []
        primary = node
        # 主 parent 必须是 factor_id 对应的节点（multi-parent 里的第一个）
        if not parent_pool or _pid(parent_pool[0]) != str(factor_id):
            parent_pool = [primary] + [
                p for p in parent_pool if _pid(p) != str(factor_id)
            ]
        parent_pool = parent_pool[: max_items or 5]
        roles = dict(roles or {})
        if not roles:
            for i, p in enumerate(parent_pool):
                pid = _pid(p)
                roles[pid] = "main" if i == 0 else "parent"

        # 邻居 + survival（sealed 隔离）
        nn = self.store._numerical_neighbors_of(str(factor_id))[:max_items]
        structural = [dict(n) for n in (neighbors or [])[:max_items]]
        surv = self.store.survival_exemplars(k=5)
        if research_version is not None:
            surv = sealed_survival_rows(surv, version_key=research_version)
        cluster_ctx = self.store.cluster_summary()
        saturated = [
            e["action_family"]
            for e in self.store.exploration_of(str(factor_id))
            if e.get("saturation", 0) > 0.8
        ]
        explored = {
            e["action_family"] for e in self.store.exploration_of(str(factor_id))
        }
        all_actions = [
            "REFINE", "WINDOW_SCALE", "FIELD_SUBSTITUTION", "OPERATOR_SUBSTITUTION",
            "STATE_CONDITION", "CROSSOVER", "SCHEMA_EXPLORE", "ROBUSTIFY",
            "GENERATION_REPAIR",
        ]
        unexplored = [a for a in all_actions if a not in explored]

        packet = MemoryPacketV2(
            parents=parent_pool,
            parent_roles=roles,
            structural_neighbors=structural,
            numerical_neighbors=nn,
            representative_failures=self.store.recent_failures(limit=5),
            survival_exemplars=surv,
            allowed_fields=list(allowed_fields or []),
            allowed_operators=list(allowed_operators or []),
            cluster_saturation=cluster_ctx or {},
            successful_actions=[],
            failed_actions=[],
            saturated_actions=saturated,
            unexplored_actions=unexplored,
            research_objective=objective or "",
            logic_schema=dict(logic_schema or {}),
        )
        if budget_max is not None:
            packet.budget_max = budget_max
        if budget_min is not None:
            packet.budget_min = budget_min
        return packet


def _pid(p: dict[str, Any]) -> str:
    return str(p.get("factor_id") or p.get("id") or "")


__all__ = ["MemoryRetriever"]
