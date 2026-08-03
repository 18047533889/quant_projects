# -*- coding: utf-8 -*-
"""A 股与股东结构 composite lowerings。"""
from __future__ import annotations

from planner.composite_lowering import register_lowering
from planner.logical_plan import PlanNode
from planner.lowerings import _helpers as H


def _ratio(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.safe_div(node.inputs[0], node.inputs[1])


for _name in (
    "float_share_ratio",
    "free_float_share_ratio",
    "true_turnover_rate",
    "holder_concentration",
    "benchmark_relative_price",
):
    register_lowering(_name)(_ratio)


@register_lowering("earnings_yield")
def lower_earnings_yield(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.safe_div(node.inputs[0], node.inputs[1])


@register_lowering("book_to_price")
def lower_book_to_price(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.safe_div(node.inputs[0], node.inputs[1])


@register_lowering("benchmark_excess_return")
def lower_benchmark_excess_return(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.binop("subtract", node.inputs[0], node.inputs[1])


@register_lowering("limit_up_state")
def lower_limit_up_state(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.binop("ge", node.inputs[0], node.inputs[1])


@register_lowering("limit_down_state")
def lower_limit_down_state(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.binop("le", node.inputs[0], node.inputs[1])
