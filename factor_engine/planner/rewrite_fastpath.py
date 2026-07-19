# -*- coding: utf-8 -*-
"""自动改写公式/计划为更易进入 production fast path 的形式。"""
from __future__ import annotations

from planner.logical_plan import PlanNode


def _same_column(a: PlanNode, b: PlanNode) -> bool:
    """判断两节点是否引用同一数据列。"""
    return (
        a.op == "column"
        and b.op == "column"
        and a.attrs.get("name") == b.attrs.get("name")
    )


def _column_node(name: str) -> PlanNode | None:
    """构造列引用节点；空名时返回 ``None``。"""
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


def _window_value(node: PlanNode, *, default: int) -> int:
    value = node.attrs.get("d")
    if value is None:
        value = node.attrs.get("window")
    if value is None and len(node.inputs) > 1 and node.inputs[1].op == "literal":
        value = node.inputs[1].attrs.get("value")
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value <= 0:
        raise ValueError(f"window must be a positive integer, got {value!r}")
    return int(value)


def _window_attrs(node: PlanNode) -> dict:
    """从节点 attrs 或 literal positional input 提取窗口。"""
    w = _window_value(node, default=3)
    return {"d": w, "window": w}


def _try_rewrite_ts_zscore(node: PlanNode, inputs: list[PlanNode]) -> PlanNode | None:
    """``(col - ts_mean(col,w)) / ts_std(col,w)`` → ``ts_zscore(col,w)``。

    参数：
        node: 当前待匹配节点（已递归改写子节点）
        inputs: 改写后的子节点列表

    返回：
        匹配成功时返回 ``ts_zscore`` 节点，否则 ``None``
    """
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
    for key in ("d", "window", "min_periods", "ddof", "null_policy"):
        mean_value = mean_node.attrs.get(key)
        std_value = den.attrs.get(key)
        if key in {"d", "window"}:
            if mean_value is None:
                mean_value = mean_attrs["d"]
            if std_value is None:
                std_value = std_attrs["d"]
        if mean_value != std_value:
            return None
    w = mean_attrs["d"]
    attrs = {"d": w, "window": w}
    return PlanNode(op="ts_zscore", inputs=[left], attrs=attrs)


def _try_rewrite_log_returns(node: PlanNode, inputs: list[PlanNode]) -> PlanNode | None:
    """``log(divide(col, delay(col,d)))`` → ``log_returns(col,d)``。

    参数：
        node: 当前待匹配节点
        inputs: 改写后的子节点列表

    返回：
        匹配成功时返回 ``log_returns`` 节点，否则 ``None``
    """
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
    return PlanNode(op="ts_log_return", inputs=[col], attrs={"d": w, "window": w})


def rewrite_plan_for_fastpath(
    plan: PlanNode,
    *,
    allow_semantic_rewrites: bool = False,
) -> PlanNode:
    """对逻辑计划做 fastpath 改写。

    ``protected_div`` 和 ``group_neutralize`` 不是所有输入下都与原始
    ``divide`` / ``subtract(group_mean)`` 等价。调用方若未明确允许语义
    改写，应保持原始算子，仅使用严格等价的 fusion 规则。
    """

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
    """字符串级占位：复杂 DSL 改写走 planner；此处仅做文档化入口。

    参数：
        formula: 原始公式字符串

    返回：
        未改写的公式字符串（当前实现为透传）
    """
    return str(formula or "")
