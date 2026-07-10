# -*- coding: utf-8
"""基本面比率 composite lowering。"""
from __future__ import annotations

from planner.composite_lowering import register_lowering
from planner.logical_plan import PlanNode
from planner.lowerings import _helpers as H


@register_lowering("operating_margin")
def lower_operating_margin(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.safe_div_null(node.inputs[0], node.inputs[1])


@register_lowering("current_ratio")
def lower_current_ratio(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.safe_div_null(node.inputs[0], node.inputs[1])


@register_lowering("debt_to_equity")
def lower_debt_to_equity(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.safe_div_null(node.inputs[0], node.inputs[1])


@register_lowering("quick_ratio")
def lower_quick_ratio(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    assets, inventory, liabilities = node.inputs[0], node.inputs[1], node.inputs[2]
    num = H.binop("subtract", assets, inventory)
    return H.safe_div_null(num, liabilities)
