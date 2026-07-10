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
    """lookback 取当前已知最大值与候选窗口的较大者。"""
    return max(current, candidate)


@dataclass
class AnalysisResult:
    """``Analyzer.lower`` 的返回值：IR 树 + 副作用分析摘要。"""

    ir: IRNode
    lookback: int              # 公式所需最大历史窗口（用于 prefetch / warmup）
    has_ts_op: bool            # 是否含时序算子（按标的分组滚动）
    has_cs_op: bool            # 是否含截面算子（按时间截面）
    referenced_columns: set[str]  # 公式引用的数据列名集合


class Analyzer:
    """将 ``Expr`` 树降为 IR；所有函数调用均来自 ``cleaned_operators``。"""

    def lower(self, expr: Expr) -> AnalysisResult:
        """遍历 Expr 树，生成 IRNode 并累计 lookback / 引用列 / ts·cs 标记。

        参数：
            expr: 根表达式（通常来自 ``Factor.expr`` 或 ``parse_expr``）

        返回：
            ``AnalysisResult``，其中 ``ir`` 可交给 ``planner.lowerer`` 继续 lower。

        异常：
            ``NotImplementedError``：不支持的 Expr 类型，或 kwargs 含非字面量 Expr。
        """
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
                    if policy.min_periods and policy.min_periods > 1:
                        lookback = _lb_max(lookback, int(policy.min_periods) - 1)
                    if policy.lookback_window is not None and policy.lookback_window > 0:
                        lookback = _lb_max(lookback, int(policy.lookback_window))

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
