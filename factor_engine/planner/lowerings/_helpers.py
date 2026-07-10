# -*- coding: utf-8
"""Composite lowering 计划构造 helper。"""
from __future__ import annotations

from planner.logical_plan import PlanNode


def literal(value: float) -> PlanNode:
    return PlanNode(op="literal", attrs={"value": float(value)}, inputs=[])


def _literal_input(node: PlanNode, index: int) -> float | None:
    if len(node.inputs) <= index:
        return None
    child = node.inputs[index]
    if child.op == "literal" and "value" in child.attrs:
        return float(child.attrs["value"])
    return None


def window_int(node: PlanNode, *, default: int = 1, input_index: int = 1) -> int:
    w = node.attrs.get("window") or node.attrs.get("d")
    if w is not None:
        return int(w)
    lit = _literal_input(node, input_index)
    if lit is not None:
        return int(lit)
    return int(default)


def window_attrs(node: PlanNode, *, default: int = 1, input_index: int = 1) -> dict[str, int]:
    w = window_int(node, default=default, input_index=input_index)
    return {"d": w, "window": w}


def float_attr(
    node: PlanNode,
    *keys: str,
    default: float,
    input_index: int | None = None,
) -> float:
    for key in keys:
        if key in node.attrs and node.attrs[key] is not None:
            return float(node.attrs[key])
    if input_index is not None:
        lit = _literal_input(node, input_index)
        if lit is not None:
            return float(lit)
    return float(default)


def ts_delay(x: PlanNode, window: int) -> PlanNode:
    w = int(window)
    return PlanNode(op="ts_delay", inputs=[x, literal(float(w))], attrs={})


def ts_mean(x: PlanNode, window: int) -> PlanNode:
    w = int(window)
    return PlanNode(op="ts_mean", inputs=[x, literal(float(w))], attrs={})


def ts_std(x: PlanNode, window: int) -> PlanNode:
    w = int(window)
    return PlanNode(op="ts_std", inputs=[x, literal(float(w))], attrs={})


def ts_min(x: PlanNode, window: int) -> PlanNode:
    w = int(window)
    return PlanNode(op="ts_min", inputs=[x, literal(float(w))], attrs={})


def ts_max(x: PlanNode, window: int) -> PlanNode:
    w = int(window)
    return PlanNode(op="ts_max", inputs=[x, literal(float(w))], attrs={})


def ts_delta(x: PlanNode, window: int = 1) -> PlanNode:
    w = int(window)
    return PlanNode(op="ts_delta", inputs=[x, literal(float(w))], attrs={})


def ts_pct(x: PlanNode, window: int = 1) -> PlanNode:
    w = int(window)
    return PlanNode(op="ts_pct", inputs=[x, literal(float(w))], attrs={})


def binop(op: str, left: PlanNode, right: PlanNode) -> PlanNode:
    return PlanNode(op=op, inputs=[left, right], attrs={})


def unary(op: str, inner: PlanNode) -> PlanNode:
    return PlanNode(op=op, inputs=[inner], attrs={})


def cum_sum(x: PlanNode) -> PlanNode:
    return PlanNode(op="cum_sum", inputs=[x], attrs={})


def protected_div(left: PlanNode, right: PlanNode) -> PlanNode:
    return binop("protected_div", left, right)

