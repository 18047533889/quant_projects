"""
编译期中间表示（IR）节点。

由 :class:`ir.analyzer.Analyzer` 从 ``Expr`` 下降得到；再经 :class:`planner.lowerer.Lowerer`
转成 :class:`planner.logical_plan.PlanNode`（结构相同，便于后端执行）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IRNode:
    """单棵 IR 子树：算子名 + 子节点元组 + 属性（窗口 d、filter 等）。

    ``attrs`` participates in IR hashing and plan cache keys — keep only operator
    parameters (window, d, …) here.  ``semantic_attrs`` carries field-catalog
    metadata (domain, frequency, cardinality, pit_safe, unit) that is useful for
    type-checking passes but must NOT affect hash identity.
    """

    op: str
    inputs: tuple["IRNode", ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    semantic_attrs: dict[str, Any] = field(
        default_factory=dict, compare=False, hash=False
    )
