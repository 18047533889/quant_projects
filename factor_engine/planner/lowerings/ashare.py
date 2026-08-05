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
    if not node.inputs:
        return node
    pe = node.inputs[-1]
    positive = H.binop("gt", pe, H.literal(0.0))
    ratio = H.safe_div(H.literal(1.0), pe)
    return PlanNode(op="where", inputs=[positive, ratio, H.literal(float("nan"))], attrs={})


@register_lowering("book_to_price")
def lower_book_to_price(node: PlanNode) -> PlanNode:
    if not node.inputs:
        return node
    pb = node.inputs[-1]
    positive = H.binop("gt", pb, H.literal(0.0))
    ratio = H.safe_div(H.literal(1.0), pb)
    return PlanNode(op="where", inputs=[positive, ratio, H.literal(float("nan"))], attrs={})


@register_lowering("benchmark_excess_return")
def lower_benchmark_excess_return(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    return H.binop("subtract", node.inputs[0], node.inputs[1])


@register_lowering("limit_up_close")
@register_lowering("limit_up_state")
def lower_limit_up_state(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    tolerance = node.inputs[2] if len(node.inputs) > 2 else H.literal(0.005)
    threshold = H.binop("subtract", node.inputs[1], tolerance)
    return H.binop("ge", node.inputs[0], threshold)


@register_lowering("limit_down_close")
@register_lowering("limit_down_state")
def lower_limit_down_state(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    tolerance = node.inputs[2] if len(node.inputs) > 2 else H.literal(0.005)
    threshold = H.binop("add", node.inputs[1], tolerance)
    return H.binop("le", node.inputs[0], threshold)
