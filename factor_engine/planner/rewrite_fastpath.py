# -*- coding: utf-8 -*-
"""自动改写公式/计划为更易进入 production fast path 的形式。"""
from __future__ import annotations

from planner.logical_plan import PlanNode


def _same_column(a: PlanNode, b: PlanNode) -> bool:
    return (
        a.op == "column"
        and b.op == "column"
        and a.attrs.get("name") == b.attrs.get("name")
    )


def _column_node(name: str) -> PlanNode | None:
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


def _window_value(node: PlanNode, *, default: int) -> int:
    value = node.attrs.get("window")
    if value is None:
        value = node.attrs.get("d")
    if value is None and len(node.inputs) > 1 and node.inputs[1].op == "literal":
        value = node.inputs[1].attrs.get("value")
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value <= 0:
        raise ValueError(f"window must be a positive integer, got {value!r}")
    return int(value)


def _window_attrs(node: PlanNode) -> dict:
    """Extract a window using the canonical ``window`` spelling only."""
    return {"window": _window_value(node, default=3)}


def _try_rewrite_ts_zscore(node: PlanNode, inputs: list[PlanNode]) -> PlanNode | None:
    """``(col-ts_mean(col,w))/ts_std(col,w)`` → canonical ``ts_zscore``."""
    if node.op != "divide" or len(inputs) != 2:
        return None
    num, den = inputs
    if num.op != "subtract" or len(num.inputs) != 2:
        return None
    left, mean_node = num.inputs
    if mean_node.op != "ts_mean" or den.op != "ts_std":
        return None
    if not (
        _same_column(left, mean_node.inputs[0])
        and _same_column(left, den.inputs[0])
        and _same_column(mean_node.inputs[0], den.inputs[0])
    ):
        return None
    mean_attrs = _window_attrs(mean_node)
    std_attrs = _window_attrs(den)
    if mean_attrs["window"] != std_attrs["window"]:
        return None
    allowed_mean = {
        "d", "window", "min_periods", "null_policy", "nan_policy",
        "includes_current_bar",
    }
    allowed_std = allowed_mean | {"ddof", "zero_std_policy"}
    if set(mean_node.attrs) - allowed_mean or set(den.attrs) - allowed_std:
        return None
    defaults = {
        "min_periods": 1,
        "null_policy": "ignore",
        "nan_policy": "propagate",
        "includes_current_bar": True,
    }
    for key, default in defaults.items():
        if mean_node.attrs.get(key, default) != den.attrs.get(key, default):
            return None
    w = mean_attrs["window"]
    attrs = {
        "window": w,
        **{key: mean_node.attrs.get(key, default) for key, default in defaults.items()},
        "ddof": den.attrs.get("ddof", 1),
        "zero_std_policy": den.attrs.get("zero_std_policy", "zero"),
    }
    return PlanNode(op="ts_zscore", inputs=[left], attrs=attrs)


def _try_rewrite_log_returns(node: PlanNode, inputs: list[PlanNode]) -> PlanNode | None:
    """``log(divide(col, delay(col,d)))`` → canonical ``ts_log_return``."""
    if node.op != "log" or len(inputs) != 1:
        return None
    inner = inputs[0]
    if inner.op != "divide" or len(inner.inputs) != 2:
        return None
    num, den = inner.inputs
    if num.op != "column":
        return None
    col_name = str(num.attrs.get("name") or "")
    if den.op not in {"ts_delay", "delay"} or len(den.inputs) != 1:
        return None
    if not _same_column(num, den.inputs[0]):
        return None
    w = _window_value(den, default=1)
    col = _column_node(col_name)
    if col is None:
        return None
    # ``d`` is the canonical ts_log_return parameter; do not emit a duplicate
    # ``window`` alias that would later be rejected by execution.
    return PlanNode(op="ts_log_return", inputs=[col], attrs={"d": w})


def rewrite_plan_for_fastpath(
    plan: PlanNode,
    *,
    allow_semantic_rewrites: bool = False,
) -> PlanNode:
    """Rewrite strictly equivalent fast paths; semantic rewrites are opt-in."""
    inputs = [
        rewrite_plan_for_fastpath(c, allow_semantic_rewrites=allow_semantic_rewrites)
        for c in plan.inputs
    ]
    op = plan.op
    attrs = dict(plan.attrs)

    rewritten = _try_rewrite_ts_zscore(PlanNode(op=op, inputs=inputs, attrs=attrs), inputs)
    if rewritten is not None:
        return rewritten

    rewritten = _try_rewrite_log_returns(PlanNode(op=op, inputs=inputs, attrs=attrs), inputs)
    if rewritten is not None:
        return rewritten

    if allow_semantic_rewrites and op in {"divide", "div"} and len(inputs) == 2:
        return PlanNode(op="protected_div", inputs=inputs, attrs=attrs)

    if allow_semantic_rewrites and op in {"subtract", "sub"} and len(inputs) == 2:
        left, right = inputs
        if (
            right.op == "group_mean"
            and len(right.inputs) == 2
            and _same_column(left, right.inputs[0])
        ):
            grp = right.inputs[1]
            return PlanNode(op="group_neutralize", inputs=[left, grp], attrs=attrs)

    return PlanNode(op=op, inputs=inputs, attrs=attrs)


def rewrite_formula_for_fastpath(formula: str) -> str:
    return str(formula or "")
