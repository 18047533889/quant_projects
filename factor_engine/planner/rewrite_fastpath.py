# -*- coding: utf-8 -*-
"""自动改写公式/计划为更易进入 production fast path 的形式。"""
from __future__ import annotations

from typing import Any, Mapping

from planner.logical_plan import PlanNode


_COLUMN_SOURCE_KEYS = ("source_table", "field_id", "source_field", "field_registry_hash")


def rewrite_node(
    old: PlanNode,
    *,
    op: str | None = None,
    inputs: list[PlanNode] | tuple[PlanNode, ...] | None = None,
    attrs: Mapping[str, Any] | None = None,
    semantic_attrs: Mapping[str, Any] | None = None,
) -> PlanNode:
    """R13 NEW-P1-70: the ONE planner-rewrite constructor.

    Every planner rewrite must go through this instead of scattering bare
    ``PlanNode(...)`` calls, so ``semantic_attrs`` (the typed layer's unit /
    grain / availability / price-basis / source identity) is preserved by default
    and node identity is carried.  ``semantic_attrs`` may be passed explicitly
    when a rewrite legitimately changes the OUTPUT semantics (e.g. a semantic
    transform); otherwise the original node's attrs are inherited.
    """
    return PlanNode(
        op=str(op) if op is not None else old.op,
        inputs=tuple(inputs) if inputs is not None else old.inputs,
        attrs=dict(attrs) if attrs is not None else dict(old.attrs),
        semantic_attrs=dict(semantic_attrs) if semantic_attrs is not None else dict(old.semantic_attrs),
        node_id=old.node_id,
    )


def _column_identity_key(node: PlanNode) -> tuple | None:
    """Full identity of a column node (Review-8 #G).

    A display ``name`` is NOT an identity: ``sourceA.Close`` and
    ``sourceB.Close`` both display "Close" yet carry different data.  When the
    node carries explicit source identity (field_id / source_table /
    source_field from the typed analyzer), that identity is authoritative.
    """
    if node.op != "column":
        return None
    name = node.attrs.get("name")
    if name is None:
        return None
    source_id = tuple(node.attrs.get(k) for k in _COLUMN_SOURCE_KEYS)
    if any(v is not None for v in source_id):
        return (name, source_id)
    return None


def _same_column(a: PlanNode, b: PlanNode) -> bool:
    """判断两节点是否引用同一数据列。

    Review-8 #G: 两列可能共享 display name（sourceA.Close 与 sourceB.Close），
    却完全是不同的数据。当两侧都携带显式 source identity（typed analyzer 写出
    的 field_id / source_table / source_field）时，identity 优先于 name；
    否则退回 name 比较（synthetic/未解析列）。
    """
    if a.op != "column" or b.op != "column":
        return False
    ka = _column_identity_key(a)
    kb = _column_identity_key(b)
    if ka is not None and kb is not None:
        return ka == kb
    return a.attrs.get("name") == b.attrs.get("name")


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
    w = mean_attrs["d"]
    attrs = {
        "d": w,
        "window": w,
        **{key: mean_node.attrs.get(key, default) for key, default in defaults.items()},
        "ddof": den.attrs.get("ddof", 1),
        "zero_std_policy": den.attrs.get("zero_std_policy", "zero"),
    }
    # R13 NEW-P1-70: rewrites carry the original node's semantic attrs (ts_zscore
    # has the same output domain/unit as the divide it replaces).
    return rewrite_node(node, op="ts_zscore", inputs=[left], attrs=attrs)


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
    if den.op not in {"ts_delay", "delay"} or len(den.inputs) != 1:
        return None
    if not _same_column(num, den.inputs[0]):
        return None
    w = _window_value(den, default=1)
    # R13 NEW-P0-16: reuse the ORIGINAL column node ``num`` instead of rebuilding
    # a bare ``Column(name)``.  ``sourceA.Close`` and ``sourceB.Close`` may share
    # a display name but carry different data; rebuilding from the name alone
    # drops the typed source identity (field_id / source_table / source_field /
    # field_registry_hash) and would alias one source to the other.
    return rewrite_node(
        node,
        op="ts_log_return",
        inputs=[num],
        attrs={"d": w, "window": w},
    )


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
    # R13 NEW-P1-70: rewrites are built through ``rewrite_node`` so semantic
    # attrs are never dropped.  The matchers see a node carrying the original
    # node's semantic attrs, and every returned node inherits them.
    current = PlanNode(
        op=op,
        inputs=inputs,
        attrs=attrs,
        semantic_attrs=dict(plan.semantic_attrs),
        node_id=plan.node_id,
    )

    rewritten = _try_rewrite_ts_zscore(current, inputs)
    if rewritten is not None:
        return rewritten

    rewritten = _try_rewrite_log_returns(current, inputs)
    if rewritten is not None:
        return rewritten

    if allow_semantic_rewrites and op in {"divide", "div"} and len(inputs) == 2:
        return rewrite_node(current, op="protected_div", inputs=inputs, attrs=attrs)

    if allow_semantic_rewrites and op in {"subtract", "sub"} and len(inputs) == 2:
        left, right = inputs
        if (
            right.op == "group_mean"
            and len(right.inputs) == 2
            and _same_column(left, right.inputs[0])
        ):
            grp = right.inputs[1]
            return rewrite_node(current, op="group_neutralize", inputs=[left, grp], attrs=attrs)

    return rewrite_node(current)


def rewrite_formula_for_fastpath(formula: str) -> str:
    """字符串级占位：复杂 DSL 改写走 planner；此处仅做文档化入口。

    参数：
        formula: 原始公式字符串

    返回：
        未改写的公式字符串（当前实现为透传）
    """
    return str(formula or "")
