"""MemoryRetriever（任务书 §41/§76）：GlobalMemoryStore → MemoryPacket → LLM prompt。

Retriever 是 LLM 与 Global Memory 之间的唯一通道——LLM 不直接读库（§40）。
在 store.build_memory_packet 的基础上补 numerical_neighbors：同 parameter family
最近 5 个 factor_node 的 formula + fitness（§41）。
"""

from __future__ import annotations

import json  # noqa: F401  (保留：_fitness_of 历史实现参考 / 未来 schema 解析)

from alphaprobe.memory import GlobalMemoryStore, MemoryPacket
from alphaprobe.memory import _fitness_of


class MemoryRetriever:
    """构建 LLM working memory 快照。线程安全（读路径，store 自带锁）。"""

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

    # -- §41 build packet -------------------------------------------------------

    def build_packet(
        self,
        factor_id: str,
        *,
        allowed_fields: list[str] | None = None,
        allowed_operators: list[str] | None = None,
    ) -> MemoryPacket | None:
        """构建 MemoryPacket。factor_id 未知时返回 None（不抛异常）。

        numerical_neighbors / survival_exemplars / cluster_context /
        allowed_fields / allowed_operators 已由 store.build_memory_packet 填充；
        本方法仅透传 allowed 列表（round_manager/pipeline 接线时使用）。
        """
        node = self.store.get_node(factor_id)
        if node is None:
            return None
        return self.store.build_memory_packet(
            parent_node=node,
            allowed_fields=allowed_fields,
            allowed_operators=allowed_operators,
        )
