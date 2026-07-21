"""表达式 → IR：Analyzer 把 ``Expr`` 树降为可执行的 ``IRNode``。

支持节点
--------
仅三种：``ColumnRef``（列引用）、``Literal``（常量）、``CleanedCall``（算子调用）。
所有算子语义以 ``cleaned_operators`` 为准；此处不做数值计算。

副作用分析
----------
遍历 ``CleanedCall`` 时顺带推导：
- ``lookback``：沿表达式 DAG 的最长依赖路径递归累加，而不是只取全局最大窗口；
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


_LAG_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "ts_delay": ("n", "d", "lag", "periods", "window"),
    "prev": (),
    "ts_delta": ("n", "d", "lag", "periods", "window"),
    "ts_pct": ("d", "n", "lag", "periods", "window"),
    "ts_log_return": ("d", "n", "lag", "periods", "window"),
    "ts_ratio": (),
    "MOM": ("window", "d", "n"),
    "ROC": ("window", "d", "n"),
}
_FIXED_LAGS: dict[str, int] = {
    "prev": 1,
    "ts_ratio": 1,
}
_WINDOW_PARAM_NAMES: tuple[str, ...] = (
    "window",
    "d",
    "span",
    "period",
    "periods",
)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _literal_value(expr: Expr) -> Any | None:
    return expr.value if isinstance(expr, Literal) else None


def _operator_param_values(node: CleanedCall, op_impl: Any) -> dict[str, Any]:
    """Resolve literal positional/keyword parameters using operator metadata."""
    values = dict(node.kwargs_dict())
    param_names = tuple(getattr(getattr(op_impl, "metadata", None), "param_names", ()) or ())
    for index, arg in enumerate(node.args):
        if index >= len(param_names):
            break
        literal = _literal_value(arg)
        if literal is not None or isinstance(arg, Literal):
            values.setdefault(param_names[index], literal)
    return values


def _operator_lookback_increment(
    canon: str,
    node: CleanedCall,
    op_impl: Any,
    policy: Any | None,
) -> int:
    """Return additional historical rows required above the deepest child."""
    params = _operator_param_values(node, op_impl)
    increment = 0

    if canon in _FIXED_LAGS:
        increment = max(increment, _FIXED_LAGS[canon])

    lag_names = _LAG_PARAM_NAMES.get(canon)
    if lag_names is not None:
        for name in lag_names:
            value = _positive_int(params.get(name))
            if value is not None:
                increment = max(increment, value)
        if policy is not None:
            lag = _positive_int(getattr(policy, "lag", None))
            if lag is not None:
                increment = max(increment, lag)
        return increment

    # Rolling/window operators require w-1 rows above their deepest input.
    for name in _WINDOW_PARAM_NAMES:
        value = _positive_int(params.get(name))
        if value is not None:
            increment = max(increment, value - 1)

    # Multi-stage technical indicators have several literal horizons. This is
    # still a finite warm-up approximation; stateful operators should eventually
    # expose a dedicated lookback_fn/state contract.
    if canon in {"MACD", "MACD_line", "MACD_signal", "MACD_hist"}:
        fast = _positive_int(params.get("fast")) or 0
        slow = _positive_int(params.get("slow")) or 0
        signal = _positive_int(params.get("signal")) or 0
        increment = max(increment, max(fast, slow) - 1 + max(signal - 1, 0))

    if policy is not None:
        lag = _positive_int(getattr(policy, "lag", None))
        if lag is not None:
            increment = max(increment, lag)

        lookback_window = _positive_int(getattr(policy, "lookback_window", None))
        if lookback_window is not None:
            increment = max(increment, lookback_window - 1)

        min_periods = _positive_int(getattr(policy, "min_periods", None))
        if min_periods is not None:
            increment = max(increment, min_periods - 1)

    return increment


def _legacy_single_window_lookback(expr: Expr) -> int | None:
    """Preserve the public lookback value for one direct rolling dependency.

    Internally rolling operators use the precise historical-row convention
    (``window - 1``), which composes correctly for nested expressions. Older
    callers, however, exposed the full window for a single rolling input under
    zero-lookback wrappers such as ``rank(ts_mean(close, 20))``.
    """
    def visit(node: Expr) -> int | None:
        if isinstance(node, (ColumnRef, Literal)):
            return 0
        if not isinstance(node, CleanedCall):
            return None
        from backend.cleaned_bridge import ensure_cleaned_loaded
        from cleaned_operators.registry import OperatorRegistry

        ensure_cleaned_loaded()
        canon = OperatorRegistry.resolve_canonical_strict(node.op)
        op_impl = OperatorRegistry.get(canon)
        if op_impl is None:
            return None
        policy = None
        from cleaned_operators.operator_policy import infer_operator_policy

        policy = infer_operator_policy(op_impl, canonical=canon)
        increment = _operator_lookback_increment(canon, node, op_impl, policy)
        expr_children = [arg for arg in node.args if isinstance(arg, Expr)]
        series_children = [
            arg for arg in expr_children if not isinstance(arg, Literal)
        ]
        if increment > 0:
            if len(series_children) != 1 or not isinstance(series_children[0], ColumnRef):
                return None
            params = _operator_param_values(node, op_impl)
            for name in _WINDOW_PARAM_NAMES:
                window = _positive_int(params.get(name))
                if window is not None:
                    return window
            return None
        child_windows = [visit(child) for child in series_children]
        if any(window is None for window in child_windows):
            return None
        return max((window or 0 for window in child_windows), default=0)

    result = visit(expr)
    return result if result and result > 0 else None


@dataclass
class AnalysisResult:
    """``Analyzer.lower`` 的返回值：IR 树 + 副作用分析摘要。"""

    ir: IRNode
    lookback: int
    has_ts_op: bool
    has_cs_op: bool
    referenced_columns: set[str]


class Analyzer:
    """将 ``Expr`` 树降为 IR；所有函数调用均来自 ``cleaned_operators``。"""

    def lower(self, expr: Expr) -> AnalysisResult:
        """遍历 Expr 树，生成 IRNode 并递归推导历史依赖。"""
        cols: set[str] = set()
        has_ts = False
        has_cs = False

        def visit(node: Expr) -> tuple[IRNode, int]:
            nonlocal has_ts, has_cs

            if isinstance(node, ColumnRef):
                cols.add(node.name)
                return IRNode(op="column", attrs={"name": node.name}), 0

            if isinstance(node, Literal):
                return IRNode(op="literal", attrs={"value": node.value}), 0

            if isinstance(node, CleanedCall):
                from backend.cleaned_bridge import ensure_cleaned_loaded
                from cleaned_operators.registry import OperatorRegistry

                ensure_cleaned_loaded()
                if node.op in {"bfill", "causal_bfill"}:
                    from backend.polars_long_policy import UnsupportedCausalOperatorError
                    raise UnsupportedCausalOperatorError(
                        f"{node.op} is disabled: backward-looking fill is not point-in-time safe"
                    )
                canon = OperatorRegistry.resolve_canonical_strict(node.op)
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

                visited_inputs = [visit(arg) for arg in node.args]
                inputs = tuple(item[0] for item in visited_inputs)
                deepest_child = max((item[1] for item in visited_inputs), default=0)

                attrs: dict[str, Any] = {}
                for key, value in node.kwargs_dict().items():
                    if isinstance(value, Expr):
                        lowered, kw_lookback = visit(value)
                        if lowered.op != "literal":
                            raise NotImplementedError(
                                f"cleaned op {node.op!r} kwargs must be literals, got {key!r}"
                            )
                        attrs[key] = lowered.attrs["value"]
                        deepest_child = max(deepest_child, kw_lookback)
                    else:
                        attrs[key] = value

                from backend.parameter_aliases import normalize_parameter_aliases

                attrs = normalize_parameter_aliases(canon, attrs)

                policy = None
                if op_impl is not None:
                    from cleaned_operators.operator_policy import infer_operator_policy

                    policy = infer_operator_policy(op_impl, canonical=canon)

                own_increment = _operator_lookback_increment(
                    canon,
                    node,
                    op_impl,
                    policy,
                )
                return (
                    IRNode(op=canon, inputs=inputs, attrs=attrs),
                    deepest_child + own_increment,
                )

            raise NotImplementedError(f"Unsupported expr: {type(node).__name__}")

        ir, lookback = visit(expr)
        legacy_window = _legacy_single_window_lookback(expr)
        if legacy_window is not None:
            lookback = max(lookback, legacy_window)
        return AnalysisResult(
            ir=ir,
            lookback=lookback,
            has_ts_op=has_ts,
            has_cs_op=has_cs,
            referenced_columns=cols,
        )
