"""逻辑执行计划节点（与 IR 结构相同，列表型 ``inputs`` 便于后端递归求值）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanNode:
    """单棵计划子树：算子名、有序子节点、属性字典。

    字段：
        op: 算子 canonical 名（与 ``cleaned_operators`` 一致）
        inputs: 子节点列表（自左向右 / positional 顺序）
        attrs: 算子参数与元数据（窗口、列名、CSE sid 等）
        node_id: 可选调试/溯源 id，不参与结构哈希
    """

    op: str
    inputs: list["PlanNode"] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    node_id: str | None = None
