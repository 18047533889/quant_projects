# -*- coding: utf-8
"""时序 primitive composite lowering。"""
from __future__ import annotations

from factor_engine.planner.composite_lowering import register_lowering
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.lowerings import _helpers as H


@register_lowering("ts_ratio", deps=("window", "lag"), min_inputs=2)
def lower_ts_ratio(node: PlanNode) -> PlanNode:
    """``safe_div(x, delay(x, lag))``；默认 lag=1。"""
    if not node.inputs:
        return node
    x = node.inputs[0]
    lag = H.window_int(node, default=1, input_index=1)
    return H.safe_div(x, H.delay(x, lag))
