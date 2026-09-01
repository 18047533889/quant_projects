"""MemoryRetriever（任务书 §41/§76）：GlobalMemoryStore → MemoryPacket → LLM prompt。

Retriever 是 LLM 与 Global Memory 之间的唯一通道——LLM 不直接读库（§40）。
在 store.build_memory_packet 的基础上补 numerical_neighbors：同 parameter family
最近 5 个 factor_node 的 formula + fitness（§41）。
"""

from __future__ import annotations

import json
from typing import Any

from alphaprobe.memory import GlobalMemoryStore, MemoryPacket


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

        按 fitness（single_ic 归一化）降序取最近 5 个；family_id 缺失或
        无同族节点时返回空表。
        """
        node = self.store.get_node(factor_id)
        if node is None:
            return []
        family_id = node.get("parameter_family_id")
        if not family_id:
            return []
        rows = self.store._conn.execute(
            "SELECT factor_id, canonical_formula, parameter_family_id,"
            " schema_json, last_seen_at FROM factor_nodes"
            " WHERE parameter_family_id=? AND factor_id<>?",
            (family_id, factor_id),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for fid, formula, _fam, schema_json, last_seen in rows:
            fitness = self._fitness_of(schema_json)
            out.append(
                {
                    "factor_id": fid,
                    "formula": formula,
                    "fitness": fitness,
                    "last_seen_at": last_seen,
                }
            )
        out.sort(key=lambda d: (d["fitness"] is not None, d["fitness"] or 0.0), reverse=True)
        return out[:k]

    @staticmethod
    def _fitness_of(schema_json: Any) -> float | None:
        """从 factor_nodes.schema_json 提取 fitness（single_ic / 归一化值）。"""
        if not schema_json:
            return None
        try:
            if isinstance(schema_json, str):
                schema_json = json.loads(schema_json)
        except Exception:
            return None
        if not isinstance(schema_json, dict):
            return None
        for key in ("fitness", "single_ic", "search_fitness", "norm_ic"):
            v = schema_json.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    # -- §41 build packet -------------------------------------------------------

    def build_packet(self, factor_id: str) -> MemoryPacket | None:
        """构建 MemoryPacket。factor_id 未知时返回 None（不抛异常）。

        结构邻居（store 层）与数值邻居（本模块）都缺省为空表，若 store 层
        有邻居来源可后续注入；to_prompt_text 保证 <4000 字符（§32 定长）。
        """
        node = self.store.get_node(factor_id)
        if node is None:
            return None
        packet = self.store.build_memory_packet(parent_node=node)
        packet.numerical_neighbors = self.numerical_neighbors(factor_id)
        return packet
