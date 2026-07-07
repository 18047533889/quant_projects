"""表达式 → IR：Analyzer 把 ``Expr`` 树降为可执行的 ``IRNode``。

支持节点
--------
仅三种：``ColumnRef``（列引用）、``Literal``（常量）、``CleanedCall``（算子调用）。
所有算子语义以 ``cleaned_operators`` 为准；此处不做数值计算。

副作用分析
----------
遍历 ``CleanedCall`` 时顺带推导：
- ``lookback``：从 ``d``/``window`` kwargs 或第二 positional 字面量取最大窗口；
- ``has_ts_op`` / ``has_cs_op``：按算子 ``metadata.category`` 与 canonical 名启发式标记；
- ``referenced_columns``：公式依赖的数据字段集合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from expr.base import Expr
from expr.cleaned_call import CleanedCall
from expr.column import ColumnRef
from expr.literal import Literal
from ir.nodes import IRNode


def _lb_max(current: int, candidate: int) -> int:
    return max(current, candidate)


@dataclass
class AnalysisResult:
    ir: IRNode
    lookback: int
    has_ts_op: bool
    has_cs_op: bool
    referenced_columns: set[str]


class Analyzer:
    """将 ``Expr`` 树降为 IR；所有函数调用均来自 ``cleaned_operators``。"""

    def lower(self, expr: Expr) -> AnalysisResult:
        cols: set[str] = set()
        lookback = 0
        has_ts = False
        has_cs = False

        def visit(node: Expr) -> IRNode:
            nonlocal lookback, has_ts, has_cs

            if isinstance(node, ColumnRef):
                cols.add(node.name)
                return IRNode(op="column", attrs={"name": node.name})

            if isinstance(node, Literal):
                return IRNode(op="literal", attrs={"value": node.value})

            if isinstance(node, CleanedCall):
                from backend.cleaned_bridge import ensure_cleaned_loaded
                from cleaned_operators.registry import OperatorRegistry

                ensure_cleaned_loaded()
                canon = OperatorRegistry._aliases.get(node.op, node.op)
                op_impl = OperatorRegistry.get(canon)
                if op_impl is not None:
                    cat = getattr(op_impl.metadata, "category", "") or ""
                    if cat in (
                        "time_series",
                        "shift_diff_cum",
                        "technical_signal",
                        "price_volume",
                        "intraday_microstructure",
                        "signal",
                    ):
                        has_ts = True
                    if cat in ("cross_sectional", "group_neutralization"):
                        has_cs = True
                if canon.startswith("ts_") or canon in {
                    "SMA",
                    "EMA",
                    "WMA",
                    "delay",
                    "decay_linear",
                }:
                    has_ts = True
                if canon in {
                    "rank",
                    "zscore",
                    "scale",
                    "normalize",
                    "winsorize",
                    "quantile",
                    "neutralize",
                } or canon.startswith("group_"):
                    has_cs = True

                inputs = tuple(visit(a) for a in node.args)
                attrs: dict[str, Any] = {}
                for k, v in node.kwargs_dict().items():
                    if isinstance(v, Expr):
                        lit = visit(v)
                        if lit.op != "literal":
                            raise NotImplementedError(
                                f"cleaned op {node.op!r} kwargs must be literals, got {k!r}"
                            )
                        attrs[k] = lit.attrs["value"]
                    else:
                        attrs[k] = v

                for key in ("d", "window"):
                    if key in attrs:
                        try:
                            lookback = _lb_max(lookback, int(attrs[key]))
                        except (TypeError, ValueError):
                            pass
                if len(node.args) >= 2 and isinstance(node.args[1], Literal):
                    try:
                        lookback = _lb_max(lookback, int(node.args[1].value))
                    except (TypeError, ValueError):
                        pass

                if op_impl is not None:
                    from cleaned_operators.operator_policy import infer_operator_policy

                    policy = infer_operator_policy(op_impl, canonical=canon)
                    if policy.lag and policy.lag > 0:
                        lookback = _lb_max(lookback, int(policy.lag))

                return IRNode(op=canon, inputs=inputs, attrs=attrs)

            raise NotImplementedError(f"Unsupported expr: {type(node).__name__}")

        ir = visit(expr)
        return AnalysisResult(
            ir=ir,
            lookback=lookback,
            has_ts_op=has_ts,
            has_cs_op=has_cs,
            referenced_columns=cols,
        )
