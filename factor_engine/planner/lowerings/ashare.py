# -*- coding: utf-8 -*-
"""A 股与股东结构 composite lowerings。"""
from __future__ import annotations

from factor_engine.planner.composite_lowering import register_lowering
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.lowerings import _helpers as H


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


def _resolve_tolerance_unit(node: PlanNode) -> str:
    """R20-195..200: resolve the limit-state tolerance unit explicitly.

    The A-share limit-state kernels treat ``tick_tolerance`` as an ABSOLUTE
    price tolerance in currency (``limit_price - tolerance``, default 0.005 CNY
    = half a tick).  The lowering must NOT silently guess the unit: an explicit
    ``tolerance_unit`` attr (``"price"`` | ``"ratio"``) wins; otherwise the
    operator metadata's declared param unit wins; otherwise it defaults to
    ``"price"`` (matching the kernel).  A ``ratio`` tolerance is applied as
    ``limit_price * (1 - tolerance)``.
    """
    explicit = node.attrs.get("tolerance_unit")
    if explicit in ("price", "ratio"):
        return str(explicit)
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(node.op)
        meta = getattr(op, "metadata", None)
        specs = getattr(meta, "param_specs", None) or {}
        spec = specs.get("tick_tolerance")
        declared_unit = getattr(spec, "unit", None) or getattr(spec, "semantic_unit", None)
        if declared_unit in ("ratio", "percent"):
            return "ratio"
        if declared_unit in ("price", "currency", "cny", "usd", "CNY", "USD"):
            return "price"
    except Exception:  # noqa: BLE001 - metadata unavailable: fall through to default
        pass
    return "price"


def _tolerance_threshold(
    node: PlanNode, limit: PlanNode, *, up: bool
) -> tuple[PlanNode, dict[str, str]]:
    """Build the threshold expression for limit-state lowerings (R20-195..200).

    Returns ``(threshold_node, semantic_attrs)`` where ``semantic_attrs`` records
    the resolved tolerance unit so downstream unit/grain/availability checkers
    see it explicitly instead of guessing.
    """
    unit = _resolve_tolerance_unit(node)
    tolerance = node.inputs[2] if len(node.inputs) > 2 else H.literal(0.005)
    if unit == "ratio":
        # limit_price * (1 - tolerance) — relative tolerance of the limit price.
        one_minus = H.binop("subtract", H.literal(1.0), tolerance)
        threshold = H.binop("multiply", limit, one_minus)
    else:
        # limit_price - tolerance / limit_price + tolerance — absolute currency.
        op = "subtract" if up else "add"
        threshold = H.binop(op, limit, tolerance)
    return threshold, {
        "tolerance_unit": unit,
        "tolerance_unit_explicit": "1" if "tolerance_unit" in node.attrs else "0",
        "tolerance_mode": "absolute_price" if unit == "price" else "relative_ratio",
    }


@register_lowering(
    "limit_up_close",
    scalar_params=("tick_tolerance", "tolerance_unit"),
    context_inputs=(1,),
    optional_inputs=(2,),
)
@register_lowering(
    "limit_up_state",
    scalar_params=("tick_tolerance", "tolerance_unit"),
    context_inputs=(1,),
    optional_inputs=(2,),
)
def lower_limit_up_state(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    threshold, sem = _tolerance_threshold(node, node.inputs[1], up=True)
    out = H.binop("ge", node.inputs[0], threshold)
    merged = dict(out.semantic_attrs)
    merged.update(sem)
    return PlanNode(
        op=out.op,
        inputs=out.inputs,
        attrs=dict(out.attrs),
        semantic_attrs=merged,
        node_id=out.node_id,
    )


@register_lowering(
    "limit_down_close",
    scalar_params=("tick_tolerance", "tolerance_unit"),
    context_inputs=(1,),
    optional_inputs=(2,),
)
@register_lowering(
    "limit_down_state",
    scalar_params=("tick_tolerance", "tolerance_unit"),
    context_inputs=(1,),
    optional_inputs=(2,),
)
def lower_limit_down_state(node: PlanNode) -> PlanNode:
    if len(node.inputs) < 2:
        return node
    threshold, sem = _tolerance_threshold(node, node.inputs[1], up=False)
    out = H.binop("le", node.inputs[0], threshold)
    merged = dict(out.semantic_attrs)
    merged.update(sem)
    return PlanNode(
        op=out.op,
        inputs=out.inputs,
        attrs=dict(out.attrs),
        semantic_attrs=merged,
        node_id=out.node_id,
    )
