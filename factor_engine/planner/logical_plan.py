"""逻辑执行计划节点（与 IR 结构相同，列表型 ``inputs`` 便于后端递归求值）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanNode:
    """单棵计划子树：算子名、有序子节点、属性字典。"""
    op: str
    inputs: list["PlanNode"] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    node_id: str | None = None
