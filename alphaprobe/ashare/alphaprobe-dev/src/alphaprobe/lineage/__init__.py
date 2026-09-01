"""lineage（任务书 §37）：FactorNode 唯一 + ActionEdge 多父。

跨轮再发现 B：仍连接到同一个 B 节点，不因 round 不同复制 factor。
（持久层在 memory.GlobalMemoryStore；本模块提供图视图。）
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