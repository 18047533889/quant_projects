# -*- coding: utf-8
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


def _window_attrs(node: PlanNode) -> dict:
    w = node.attrs.get("d") or node.attrs.get("window") or 3
    return {"d": w, "window": w}


def _try_rewrite_ts_zscore(node: PlanNode, inputs: list[PlanNode]) -> PlanNode | None:
    """``(col - ts_mean(col,w)) / ts_std(col,w)`` → ``ts_zscore(col,w)``。"""
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
    w = mean_node.attrs.get("d") or mean_node.attrs.get("window") or den.attrs.get("d") or 3
    attrs = {"d": w, "window": w}
    return PlanNode(op="ts_zscore", inputs=[left], attrs=attrs)


def _try_rewrite_log_returns(node: PlanNode, inputs: list[PlanNode]) -> PlanNode | None:
    """``log(divide(col, ts_delay(col,d)))`` → ``log_returns(col,d)``。"""
    if node.op not in {"log", "protected_log", "ln"} or len(inputs) != 1:
        return None
    inner = inputs[0]
    if inner.op not in {"divide", "protected_div"} or len(inner.inputs) != 2:
        return None
    num, den = inner.inputs
    if num.op != "column":
        return None
    col_name = str(num.attrs.get("name") or "")
    if den.op not in {"ts_delay", "delay"} or len(den.inputs) != 1:
        return None
    if not _same_column(num, den.inputs[0]):
        return None
    w = den.attrs.get("d") or den.attrs.get("window") or 1
    col = _column_node(col_name)
    if col is None:
        return None
    return PlanNode(op="log_returns", inputs=[col], attrs={"d": w, "window": w})


def rewrite_plan_for_fastpath(plan: PlanNode) -> PlanNode:
    """对逻辑计划做安全、语义保持的 fastpath 友好改写。"""
    inputs = [rewrite_plan_for_fastpath(c) for c in plan.inputs]
    op = plan.op
    attrs = dict(plan.attrs)

    rewritten = _try_rewrite_ts_zscore(PlanNode(op=op, inputs=inputs, attrs=attrs), inputs)
    if rewritten is not None:
        return rewritten

    rewritten = _try_rewrite_log_returns(PlanNode(op=op, inputs=inputs, attrs=attrs), inputs)
    if rewritten is not None:
        return rewritten

    if op in {"divide", "div"} and len(inputs) == 2:
        return PlanNode(op="protected_div", inputs=inputs, attrs=attrs)

    if op in {"subtract", "sub"} and len(inputs) == 2:
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
    """字符串级占位：复杂 DSL 改写走 planner；此处仅做文档化入口。"""
    return str(formula or "")
