# -*- coding: utf-8
"""微观结构简单代理 composite lowering（非 session-aware）。"""
from __future__ import annotations

from planner.composite_lowering import register_lowering
from planner.logical_plan import PlanNode
from planner.lowerings import _helpers as H


@register_lowering("real_turnover_rate")
def lower_real_turnover_rate(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.safe_div(node.inputs[0], node.inputs[1])


@register_lowering("micro_spread")
def lower_micro_spread(node: PlanNode) -> PlanNode:
    """``safe_div(high - low, close)``"""
    if len(node.inputs) < 3:
        return node
    high, low, close = node.inputs[0], node.inputs[1], node.inputs[2]
    spread = H.binop("subtract", high, low)
    return H.safe_div(spread, close)
