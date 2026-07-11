# -*- coding: utf-8
"""时序 primitive composite lowering。"""
from __future__ import annotations

from planner.composite_lowering import register_lowering
from planner.logical_plan import PlanNode
from planner.lowerings import _helpers as H


@register_lowering("ts_ratio")
def lower_ts_ratio(node: PlanNode) -> PlanNode:
    """``safe_div_null(x, ts_delay(x, lag))``；默认 lag=1。"""
    if not node.inputs:
        return node
    x = node.inputs[0]
    lag = H.window_int(node, default=1, input_index=1)
    return H.safe_div_null(x, H.ts_delay(x, lag))
