# -*- coding: utf-8
"""技术指标 composite lowering：高级算子 → 基础 PlanNode DAG。"""
from __future__ import annotations

from planner.composite_lowering import register_lowering
from planner.logical_plan import PlanNode
from planner.lowerings import _helpers as H


@register_lowering("MOM")
def lower_mom(node: PlanNode) -> PlanNode:
    """``ts_delta(price, w)``（与 PIT-safe causal lag 一致）。"""
    if not node.inputs:
        return node
    price = node.inputs[0]
    w = H.window_int(node, default=10, input_index=1)
    return H.ts_delta(price, w)


@register_lowering("ROC")
def lower_roc(node: PlanNode) -> PlanNode:
    """``100 × ts_pct(price, w)``（与 Pandas ROC 百分比口径一致）。"""
    if not node.inputs:
        return node
    price = node.inputs[0]
    w = H.window_int(node, default=10, input_index=1)
    pct = H.ts_pct(price, w)
    return H.binop("multiply", pct, H.literal(100.0))


def _bollinger_band(node: PlanNode, *, upper: bool) -> PlanNode:
    if not node.inputs:
        return node
    x = node.inputs[0]
    w = H.window_int(node, default=20, input_index=1)
    std_dev = H.float_attr(node, "std_dev", "k", default=2.0, input_index=2)
    mean = H.ts_mean(x, w)
    std = H.ts_std(x, w)
    scaled = H.binop("multiply", H.literal(std_dev), std)
    if upper:
        return H.binop("add", mean, scaled)
    return H.binop("subtract", mean, scaled)


@register_lowering("BollingerUpper")
def lower_bollinger_upper(node: PlanNode) -> PlanNode:
    return _bollinger_band(node, upper=True)


@register_lowering("BollingerLower")
def lower_bollinger_lower(node: PlanNode) -> PlanNode:
    return _bollinger_band(node, upper=False)


@register_lowering("DPO")
def lower_dpo(node: PlanNode) -> PlanNode:
    """``close - delay(ts_mean(close, w), w//2 + 1)``"""
    if not node.inputs:
        return node
    close = node.inputs[0]
    w = H.window_int(node, default=20, input_index=1)
    shift = (w // 2) + 1
    mean = H.ts_mean(close, w)
    delayed_mean = H.delay(mean, shift)
    return H.binop("subtract", close, delayed_mean)


@register_lowering("WilliamsR")
def lower_williams_r(node: PlanNode) -> PlanNode:
    """``-100 × (ts_max(high,w) - close) / (ts_max(high,w) - ts_min(low,w))``"""
    if len(node.inputs) < 3:
        return node
    high, low, close = node.inputs[0], node.inputs[1], node.inputs[2]
    w = H.window_int(node, default=14, input_index=3)
    hh = H.ts_max(high, w)
    ll = H.ts_min(low, w)
    num = H.binop("subtract", hh, close)
    den = H.binop("subtract", hh, ll)
    ratio = H.safe_div(num, den)
    return H.binop("multiply", ratio, H.literal(-100.0))


def _stochastic_k(high: PlanNode, low: PlanNode, close: PlanNode, window: int) -> PlanNode:
    ll = H.ts_min(low, window)
    hh = H.ts_max(high, window)
    num = H.binop("subtract", close, ll)
    den = H.binop("subtract", hh, ll)
    return H.binop("multiply", H.safe_div(num, den), H.literal(100.0))


@register_lowering("BollingerBands")
def lower_bollinger_bands(node: PlanNode) -> PlanNode:
    """布林带中轨 = ts_mean(price, window)。"""
    if not node.inputs:
        return node
    x = node.inputs[0]
    w = H.window_int(node, default=20, input_index=1)
    return H.ts_mean(x, w)


@register_lowering("StochasticK")
def lower_stochastic_k(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    high, low, close = node.inputs[0], node.inputs[1], node.inputs[2]
    w = H.window_int(node, default=14, input_index=3)
    return _stochastic_k(high, low, close, w)


@register_lowering("StochasticD")
def lower_stochastic_d(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    high, low, close = node.inputs[0], node.inputs[1], node.inputs[2]
    w = H.window_int(node, default=14, input_index=3)
    k = _stochastic_k(high, low, close, w)
    return H.ts_mean(k, 3)


@register_lowering("OBV")
def lower_obv(node: PlanNode) -> PlanNode:
    """``cum_sum(sign(ts_delta(price,1)) * volume)``"""
    if len(node.inputs) < 2:
        return node
    price, volume = node.inputs[0], node.inputs[1]
    direction = H.fillna_const(H.unary("sign", H.ts_delta(price, 1)), 0.0)
    signed_vol = H.binop("multiply", direction, volume)
    return H.cum_sum(signed_vol)
