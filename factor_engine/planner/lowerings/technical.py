# -*- coding: utf-8
"""技术指标 composite lowering：高级算子 → 基础 PlanNode DAG。"""
from __future__ import annotations

from planner.composite_lowering import register_lowering
from planner.logical_plan import PlanNode
from planner.lowerings import _helpers as H


def _macd_windows(node: PlanNode) -> tuple[int, int, int]:
    """读取显式 MACD 参数；无参数时使用 operator 默认 12/26/9。

    R5-08: 不再 ``int(...)`` 截断——lowering 必须发生在参数校验之后，因此
    ``fast=5.9`` 之类会在此处报错而不是静默变成 5。

    R20-176..179: 显式 ``0`` 是合法标量（尽管对 MACD 非法域），不能用
    ``_literal_input(...) or default`` 的 truthiness 把它当成「未提供」——
    显式 0 必须原样进入 ``strict_int`` 并被域检查拒绝，而不是被默认值吞掉。

    R20-187..189: composite lowering 遇到非法参数域（``fast >= slow``）必须抛
    参数错误，不能 ``return node`` 静默 no-op——legal 域始终要 lower（无参数时
    用默认 12/26/9），illegal 域是参数错误。
    """
    fast_attr = node.attrs.get("fast")
    if fast_attr is not None:
        fast = H.strict_int(fast_attr, "fast")
    else:
        fast_lit = H._literal_input(node, 1)
        fast = H.strict_int(fast_lit if fast_lit is not None else 12, "fast")
    slow_attr = node.attrs.get("slow")
    if slow_attr is not None:
        slow = H.strict_int(slow_attr, "slow")
    else:
        slow_lit = H._literal_input(node, 2)
        slow = H.strict_int(slow_lit if slow_lit is not None else 26, "slow")
    signal_attr = node.attrs.get("signal")
    if signal_attr is not None:
        signal = H.strict_int(signal_attr, "signal")
    else:
        signal_lit = H._literal_input(node, 3)
        signal = H.strict_int(signal_lit if signal_lit is not None else 9, "signal")
    if fast >= slow:
        raise ValueError(
            f"MACD 非法参数域: fast={fast} >= slow={slow}。MACD 要求 fast < slow；"
            "composite lowering 对非法域不再静默 no-op (R20-187..189)。"
        )
    return fast, slow, signal


@register_lowering(
    "MACD_line",
    deps=("fast", "slow"),
    min_inputs=3,
    panel_arity=1,
    scalar_params=("fast", "slow", "signal"),
    branch_params=("fast", "slow"),
    optional_inputs=(1, 2, 3),
)
def lower_macd_line(node: PlanNode) -> PlanNode:
    if not node.inputs:
        return node
    fast, slow, _ = _macd_windows(node)
    x = node.inputs[0]
    return H.binop("subtract", H.ts_ema(x, fast), H.ts_ema(x, slow))


@register_lowering(
    "MACD_signal",
    deps=("fast", "slow", "signal"),
    min_inputs=3,
    panel_arity=1,
    scalar_params=("fast", "slow", "signal"),
    branch_params=("fast", "slow"),
    optional_inputs=(1, 2, 3),
)
def lower_macd_signal(node: PlanNode) -> PlanNode:
    if not node.inputs:
        return node
    fast, slow, signal = _macd_windows(node)
    x = node.inputs[0]
    line = H.binop("subtract", H.ts_ema(x, fast), H.ts_ema(x, slow))
    return H.ts_ema(line, signal)


@register_lowering(
    "MACD_hist",
    deps=("fast", "slow", "signal"),
    min_inputs=3,
    panel_arity=1,
    scalar_params=("fast", "slow", "signal"),
    branch_params=("fast", "slow"),
    optional_inputs=(1, 2, 3),
)
def lower_macd_hist(node: PlanNode) -> PlanNode:
    if not node.inputs:
        return node
    fast, slow, signal = _macd_windows(node)
    x = node.inputs[0]
    line = H.binop("subtract", H.ts_ema(x, fast), H.ts_ema(x, slow))
    return H.binop("subtract", line, H.ts_ema(line, signal))


@register_lowering("MOM", deps=("window",), min_inputs=2)
def lower_mom(node: PlanNode) -> PlanNode:
    """``ts_delta(price, w)``（与 PIT-safe causal lag 一致）。"""
    if not node.inputs:
        return node
    price = node.inputs[0]
    w = H.window_int(node, default=10, input_index=1)
    return H.ts_delta(price, w)


@register_lowering("ROC", deps=("window",), min_inputs=2)
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


@register_lowering("BollingerUpper", deps=("window", "std_dev", "k"), min_inputs=2)
def lower_bollinger_upper(node: PlanNode) -> PlanNode:
    return _bollinger_band(node, upper=True)


@register_lowering("BollingerLower", deps=("window", "std_dev", "k"), min_inputs=2)
def lower_bollinger_lower(node: PlanNode) -> PlanNode:
    return _bollinger_band(node, upper=False)


@register_lowering("DPO", deps=("window",), min_inputs=2)
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


@register_lowering("WilliamsR", deps=("window",), min_inputs=4)
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


@register_lowering("BollingerBands", deps=("window",), min_inputs=2)
def lower_bollinger_bands(node: PlanNode) -> PlanNode:
    """布林带中轨 = ts_mean(price, window)。std_dev/k 不参与中轨 lowering（R20-184）。"""
    if not node.inputs:
        return node
    x = node.inputs[0]
    w = H.window_int(node, default=20, input_index=1)
    return H.ts_mean(x, w)


@register_lowering("StochasticK", deps=("window",), min_inputs=4)
def lower_stochastic_k(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    high, low, close = node.inputs[0], node.inputs[1], node.inputs[2]
    w = H.window_int(node, default=14, input_index=3)
    return _stochastic_k(high, low, close, w)


@register_lowering("StochasticD", deps=("window",), min_inputs=4)
def lower_stochastic_d(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 3:
        return node
    high, low, close = node.inputs[0], node.inputs[1], node.inputs[2]
    w = H.window_int(node, default=14, input_index=3)
    k = _stochastic_k(high, low, close, w)
    return H.ts_mean(k, 3)


@register_lowering("OBV", min_inputs=2)
def lower_obv(node: PlanNode) -> PlanNode:
    """``cum_sum(sign(ts_delta(price,1)) * volume)``"""
    if len(node.inputs) < 2:
        return node
    price, volume = node.inputs[0], node.inputs[1]
    direction = H.fillna_const(H.unary("sign", H.ts_delta(price, 1)), 0.0)
    signed_vol = H.binop("multiply", direction, volume)
    return H.cum_sum(signed_vol)
