# -*- coding: utf-8
"""PlanNode → Polars Expr 编译器（long-table native，对齐 SQL emitter 的 ts/inst/value 语义）。

``compile_plan_to_polars`` 不经过 ``cleaned_bridge`` / panel 往返；
由 ``PolarsLongBackend`` 或（兼容）``FACTOR_ENGINE_POLARS_EXPR=1`` 调用。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from planner.logical_plan import PlanNode

from .long_frame import (
    LongFrameResult,
    optional_universe_index,
    polars_long_to_multiindex_series,
    series_to_polars_long_lazy,
    value_to_polars_long_lazy,
)
from .pandas_compat import pd
from .polars_long_policy import (
    POLARS_EXPR_CAPABLE,
    POLARS_LONG_CAPABLE,
    POLARS_LONG_COMPATIBLE,
    POLARS_LONG_MAP_GROUPS,
    POLARS_LONG_NATIVE,
    collect_plan_op_stats,
    get_polars_long_capable,
)

from dataclasses import dataclass

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

_TS = "ts"
_INST = "inst"
_VAL = "_v"

_GRP = "_grp"


@dataclass(frozen=True)
class CompiledLongExpr:
    """Expr DAG 节点：单条 ``pl.Expr`` + 依赖列（避免中间 LazyFrame join）。"""

    expr: Any
    required_cols: frozenset[str]
    tmp_exprs: tuple[Any, ...] = ()

# expanding / cum 类算子：rolling 窗口需覆盖单 inst 全长（与 SQL UNBOUNDED PRECEDING 对齐）
_EXPANDING_WINDOW = 100_000

# 能力分层见 ``polars_long_policy``（POLARS_LONG_NATIVE / MAP_GROUPS / COMPATIBLE）


def _resolve(op: str) -> str:
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(op, op)


def _window_int(node: PlanNode, default: int = 3) -> int:
    for key in ("d", "window", "span"):
        if key in node.attrs and node.attrs[key] is not None:
            try:
                return max(int(node.attrs[key]), 1)
            except (TypeError, ValueError):
                pass
    for idx in range(1, len(node.inputs)):
        child = node.inputs[idx]
        if child.op != "literal":
            continue
        val = child.attrs.get("value")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        if isinstance(val, float) and val != int(val):
            continue
        try:
            return max(int(val), 1)
        except (TypeError, ValueError):
            pass
        break
    return default


def _ewm_alpha(node: PlanNode, default_span: int = 20) -> float:
    for key in ("span", "d", "window"):
        if key in node.attrs and node.attrs[key] is not None:
            try:
                span = max(int(node.attrs[key]), 1)
                return 2.0 / (float(span) + 1.0)
            except (TypeError, ValueError):
                pass
    span = _window_int(node, default=default_span)
    return 2.0 / (float(span) + 1.0)


def _wilder_alpha(node: PlanNode, default: int = 14) -> tuple[float, int]:
    w = max(_window_int(node, default=default), 2)
    return 1.0 / float(w), w


def _rsi_wilder_expr(value_col: str, *, window: int, alpha: float) -> pl.Expr:
    delta = pl.col(value_col).diff()
    gain = pl.when(delta.is_null()).then(None).when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta.is_null()).then(None).when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=alpha, adjust=False, min_periods=window)
    avg_loss = loss.ewm_mean(alpha=alpha, adjust=False, min_periods=window)
    rs = avg_gain / avg_loss.replace(0.0, None)
    return (
        pl.when((avg_loss == 0) & (avg_gain > 0))
        .then(100.0)
        .when((avg_gain == 0) & (avg_loss > 0))
        .then(0.0)
        .when((avg_gain == 0) & (avg_loss == 0))
        .then(50.0)
        .otherwise(100.0 - (100.0 / (1.0 + rs)))
    )


def _atr_wilder_expr(
    high_col: str,
    low_col: str,
    close_col: str,
    *,
    window: int,
    alpha: float,
) -> pl.Expr:
    prev_close = pl.col(close_col).shift(1)
    tr = (
        pl.when(prev_close.is_null())
        .then(None)
        .otherwise(
            pl.max_horizontal(
                pl.col(high_col) - pl.col(low_col),
                (pl.col(high_col) - prev_close).abs(),
                (pl.col(low_col) - prev_close).abs(),
            )
        )
    )
    return tr.ewm_mean(alpha=alpha, adjust=False, min_periods=window)


def _float_attr(node: PlanNode, *keys: str, default: float) -> float:
    for key in keys:
        if key in node.attrs and node.attrs[key] is not None:
            return float(node.attrs[key])
    return default


def _const_fill_value(node: PlanNode, *, default: float | None = None) -> float | None:
    """解析常量填充值（``fillna_const`` / ``fillna`` / ``nan_to_num``）。"""
    if "num" in node.attrs and node.attrs["num"] is not None:
        return float(node.attrs["num"])
    for key in ("value", "fill_value", "const", "c", "method"):
        if key in node.attrs and node.attrs[key] is not None:
            raw = node.attrs[key]
            if raw == "zero":
                return 0.0
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return float(raw)
    pos = _literal_value(node, 0)
    if pos is not None:
        return pos
    return default


def _literal_value(node: PlanNode, index: int = 0, *, default: float | None = None) -> float | None:
    pos = index + 1
    if pos >= len(node.inputs):
        return default
    child = node.inputs[pos]
    if child.op != "literal":
        return default
    raw = child.attrs.get("value")
    if raw == "zero":
        return 0.0
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return default


def _int_attr(node: PlanNode, *keys: str, input_index: int | None = None, default: int = 0) -> int:
    for key in keys:
        if key in node.attrs and node.attrs[key] is not None:
            return int(node.attrs[key])
    if input_index is not None:
        pos = _literal_value(node, input_index)
        if pos is not None:
            return int(pos)
    return default


def _cs_ols_exprs(y_col: str, x_col: str, *, ts: str = _TS) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """截面 OLS y ~ x + const（按 ts 分区，pairwise-valid 样本）。"""
    valid = pl.col(y_col).is_not_null() & pl.col(x_col).is_not_null()
    y_v = pl.when(valid).then(pl.col(y_col)).otherwise(None)
    x_v = pl.when(valid).then(pl.col(x_col)).otherwise(None)
    mean_y = y_v.mean().over(ts)
    mean_x = x_v.mean().over(ts)
    xy = (y_v * x_v).mean().over(ts)
    x2 = (x_v**2).mean().over(ts)
    cov_xy = xy - mean_x * mean_y
    var_x = x2 - mean_x**2
    beta = cov_xy / pl.when(var_x.is_null() | (var_x == 0)).then(1.0).otherwise(var_x)
    alpha = mean_y - beta * mean_x
    n_valid = valid.cast(pl.Int64).sum().over(ts)
    return beta, alpha, n_valid


def _rolling_ols_parts(
    y_col: str,
    x_col: str,
    w: int,
    *,
    inst: str = _INST,
    ts: str = _TS,
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """滚动 OLS y ~ x + const（pairwise-valid 样本）。"""
    valid = pl.col(y_col).is_not_null() & pl.col(x_col).is_not_null()
    y_v = pl.when(valid).then(pl.col(y_col)).otherwise(None)
    x_v = pl.when(valid).then(pl.col(x_col)).otherwise(None)
    mean_y = y_v.rolling_mean(window_size=w, min_samples=3).over(inst, order_by=ts)
    mean_x = x_v.rolling_mean(window_size=w, min_samples=3).over(inst, order_by=ts)
    xy = (y_v * x_v).rolling_mean(window_size=w, min_samples=3).over(inst, order_by=ts)
    x2 = (x_v**2).rolling_mean(window_size=w, min_samples=3).over(inst, order_by=ts)
    cov_xy = xy - mean_x * mean_y
    var_x = x2 - mean_x**2
    beta = cov_xy / pl.when(var_x.is_null() | (var_x == 0)).then(None).otherwise(var_x)
    alpha = mean_y - beta * mean_x
    n_valid = (
        valid.cast(pl.Float64)
        .rolling_sum(window_size=w, min_samples=1)
        .over(inst, order_by=ts)
    )
    fit = alpha + beta * pl.col(x_col)
    return beta, alpha, n_valid, fit


def _ts_regression_retval(node: PlanNode) -> str:
    raw = node.attrs.get("retval", node.attrs.get("mode", "slope"))
    if raw is None:
        return "slope"
    retval = str(raw).lower()
    if retval in {"1", "intercept", "alpha"}:
        return "intercept"
    if retval in {"2", "fit", "predict", "prediction"}:
        return "fit"
    if retval in {"0", "resid", "residual", "residuals"}:
        return "resid"
    return "slope"


def _rolling_linear_decay_expr(w: int) -> pl.Expr:
    weights = np.arange(1, w + 1, dtype=np.float64)

    def _fn(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(arr)
        if not valid.any():
            return np.nan
        seg = arr[valid]
        ww = weights[-len(seg) :]
        return float(np.dot(seg, ww) / ww.sum())

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_time_slope_expr(w: int) -> pl.Expr:
    t = np.arange(w, dtype=np.float64)
    t = t - t.mean()
    denom = float(np.dot(t, t))
    if denom == 0.0:
        return pl.lit(None).cast(pl.Float64)
    weights = t / denom

    def _dot(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) == 0:
            return np.nan
        ww = weights[-len(arr) :]
        valid = ~np.isnan(arr)
        if not valid.any():
            return np.nan
        return float(np.dot(arr[valid], ww[valid]))

    return pl.col(_VAL).rolling_map(_dot, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_argext_expr(w: int, *, pick: str) -> pl.Expr:
    def _fn(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).any():
            return 0.0
        if pick == "max":
            return float(np.nanargmax(arr))
        return float(np.nanargmin(arr))

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_skew_expr(w: int) -> pl.Expr:
    def _fn(arr: np.ndarray) -> float:
        s = pd.Series(np.asarray(arr, dtype=np.float64))
        if s.count() < 3:
            return np.nan
        val = s.skew()
        return np.nan if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_quantile_expr(w: int, p: float) -> pl.Expr:
    def _fn(arr: np.ndarray) -> float:
        s = pd.Series(np.asarray(arr, dtype=np.float64))
        if s.count() == 0:
            return np.nan
        return float(s.quantile(p))

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _ts_sharpe_min_periods(w: int) -> int:
    return max(2, w // 3)


def _ts_autocorr_window_lag(node: PlanNode) -> tuple[int, int, int]:
    lag_lit = _literal_value(node, 1)
    if lag_lit is not None:
        lag = max(int(lag_lit), 1)
    else:
        lag = max(_int_attr(node, "lag", default=1), 1)
    win_lit = _literal_value(node, 0)
    if win_lit is not None:
        w = max(int(win_lit), 1)
    else:
        w = _window_int(node)
    w = max(lag + 2, w)
    mp = max(2, w // 3)
    return w, lag, mp


def _expanding_over() -> tuple[str, str]:
    return _INST, _TS


def _expanding_non_null_count() -> pl.Expr:
    inst, ts = _expanding_over()
    return pl.col(_VAL).is_not_null().cast(pl.Float64).cum_sum().over(inst, order_by=ts)


def _expanding_sum_expr() -> pl.Expr:
    inst, ts = _expanding_over()
    return pl.col(_VAL).cum_sum().over(inst, order_by=ts)


def _expanding_mean_expr() -> pl.Expr:
    cnt = _expanding_non_null_count()
    return pl.when(cnt <= 0).then(None).otherwise(_expanding_sum_expr() / cnt)


def _expanding_std_expr() -> pl.Expr:
    inst, ts = _expanding_over()
    cnt = _expanding_non_null_count()
    mean = _expanding_mean_expr()
    sum_sq = (pl.col(_VAL) ** 2).cum_sum().over(inst, order_by=ts)
    var_pop = sum_sq / cnt - mean**2
    var_sample = pl.when(cnt <= 1).then(None).otherwise(var_pop * cnt / (cnt - 1.0))
    return var_sample.sqrt()


def _unary_inst_map_groups(
    inner: pl.LazyFrame,
    apply_fn,
) -> pl.LazyFrame:
    """按 inst 分组，对 ``_VAL`` 列做 pandas/numpy 一元变换（对齐 bridge 语义）。"""
    schema = inner.collect_schema()

    def _apply(g: pl.DataFrame) -> pl.DataFrame:
        out = apply_fn(g[_VAL].to_numpy())
        return g.select(
            pl.col(_TS),
            pl.col(_INST),
            pl.Series(_VAL, out),
        )

    return inner.group_by(_INST, maintain_order=True).map_groups(
        _apply,
        schema={_TS: schema[_TS], _INST: schema[_INST], _VAL: pl.Float64},
    )


def _cs_row_map_groups(inner: pl.LazyFrame, apply_fn) -> pl.LazyFrame:
    """按 ts 截面分组，对 ``_VAL`` 列做行级变换（如 qcut）。"""
    schema = inner.collect_schema()

    def _apply(g: pl.DataFrame) -> pl.DataFrame:
        out = apply_fn(g[_VAL].to_numpy())
        return g.select(
            pl.col(_TS),
            pl.col(_INST),
            pl.Series(_VAL, out),
        )

    return inner.group_by(_TS, maintain_order=True).map_groups(
        _apply,
        schema={_TS: schema[_TS], _INST: schema[_INST], _VAL: pl.Float64},
    )


def _ewm_binary_map_groups(joined: pl.LazyFrame, span: int, *, corr: bool) -> pl.LazyFrame:
    """二元 EWM 矩：按 inst 分组用 pandas ewm 对齐 bridge。"""
    w = max(int(span), 2)
    schema = joined.collect_schema()

    def _apply(g: pl.DataFrame) -> pl.DataFrame:
        xs = pd.Series(g[_VAL].to_numpy(), dtype=float)
        ys = pd.Series(g["_y"].to_numpy(), dtype=float)
        if corr:
            out = xs.ewm(span=w, adjust=False).corr(ys)
        else:
            out = xs.ewm(span=w, adjust=False).cov(ys)
        return g.select(
            pl.col(_TS),
            pl.col(_INST),
            pl.Series(_VAL, out.to_numpy()),
        )

    return joined.group_by(_INST, maintain_order=True).map_groups(
        _apply,
        schema={_TS: schema[_TS], _INST: schema[_INST], _VAL: pl.Float64},
    )


def collect_columns(node: PlanNode, out: set[str] | None = None) -> set[str]:
    acc = out if out is not None else set()
    if node.op == "column":
        name = node.attrs.get("name")
        if name:
            acc.add(str(name))
    for child in node.inputs:
        collect_columns(child, acc)
    return acc


def plan_is_polars_long_capable(plan: PlanNode) -> bool:
    capable = get_polars_long_capable()
    op = _resolve(plan.op)
    if op in {"column", "literal", "materialized_series", "plan_ref"}:
        return all(plan_is_polars_long_capable(c) for c in plan.inputs)
    if op not in capable:
        return False
    return all(plan_is_polars_long_capable(c) for c in plan.inputs)


def plan_is_polars_expr_capable(plan: PlanNode) -> bool:
    """手写 native + map_groups 路径（不含 registry bridge）。"""
    op = _resolve(plan.op)
    if op in {"column", "literal", "materialized_series", "plan_ref"}:
        return all(plan_is_polars_expr_capable(c) for c in plan.inputs)
    if op not in POLARS_LONG_COMPATIBLE:
        return False
    return all(plan_is_polars_expr_capable(c) for c in plan.inputs)


def _join_binary(left: pl.LazyFrame, right: pl.LazyFrame) -> pl.LazyFrame:
    return left.join(
        right.rename({_VAL: "_y"}),
        on=[_TS, _INST],
        how="inner",
    )


def _join_triple(
    left: pl.LazyFrame,
    mid: pl.LazyFrame,
    right: pl.LazyFrame,
) -> pl.LazyFrame:
    return left.join(
        mid.rename({_VAL: "_ym"}),
        on=[_TS, _INST],
        how="inner",
    ).join(
        right.rename({_VAL: "_y"}),
        on=[_TS, _INST],
        how="inner",
    )


def _column_ref_name(node: PlanNode) -> str | None:
    if node.op != "column":
        return None
    name = node.attrs.get("name")
    return str(name) if name else None


def _protected_div_expr(numer: pl.Expr, denom: pl.Expr, node: PlanNode) -> pl.Expr:
    """与 SQL / Pandas 一致：|denom| <= epsilon → default。"""
    from backend.numeric_semantics import protected_div_default, protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    default = _float_attr(node, "default", default=protected_div_default())
    return (
        pl.when(numer.is_null() | denom.is_null())
        .then(default)
        .when(denom.abs() <= eps)
        .then(default)
        .otherwise(numer / denom)
    )


def _protected_log_expr(val: pl.Expr, node: PlanNode) -> pl.Expr:
    """与 SQL / Pandas 一致：x <= epsilon → log(epsilon)。"""
    from backend.numeric_semantics import protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    log_eps = pl.lit(eps).log()
    return pl.when(val.is_null() | (val <= eps)).then(log_eps).otherwise(val.log())


def _truthy_expr(expr: pl.Expr) -> pl.Expr:
    return expr.is_not_null() & (expr != 0)


def _bump_shared_long_lazy_hit(ctx: Any | None) -> None:
    if ctx is None:
        return
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["shared_long_lazy_hits"] = int(runtime.get("shared_long_lazy_hits") or 0) + 1
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


_FUSABLE_TS_ON_COLUMN: frozenset[str] = frozenset(
    {
        "ts_mean",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_std",
        "ts_var",
        "ts_delay",
        "ts_delta",
        "ts_pct",
    }
)


def _parse_ts_op_on_column(node: PlanNode) -> tuple[str, str, int] | None:
    """识别 ``ts_*(column(x), window)`` 形态，返回 (op, col, window)。"""
    op = _resolve(node.op)
    if op not in _FUSABLE_TS_ON_COLUMN:
        return None
    if not node.inputs:
        return None
    col = _column_ref_name(node.inputs[0])
    if not col:
        return None
    w = max(_window_int(node, default=1), 1)
    return op, col, w


def _ts_rolling_expr_on_column(op: str, col_name: str, window: int) -> pl.Expr:
    """在宽表 base 列上直接构造 ts 窗口 expr（用于 DAG fusion）。"""
    from backend.numeric_semantics import std_ddof_value

    c = pl.col(col_name)
    w = window
    if op == "ts_mean":
        return c.rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
    if op == "ts_sum":
        return c.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
    if op == "ts_min":
        return c.rolling_min(window_size=w, min_samples=1).over(_INST, order_by=_TS)
    if op == "ts_max":
        return c.rolling_max(window_size=w, min_samples=1).over(_INST, order_by=_TS)
    if op == "ts_var":
        ddof = std_ddof_value(op)
        return c.rolling_var(window_size=w, min_samples=1, ddof=ddof).over(_INST, order_by=_TS)
    if op == "ts_std":
        ddof = std_ddof_value(op)
        return c.rolling_std(window_size=w, min_samples=1, ddof=ddof).over(_INST, order_by=_TS)
    if op == "ts_delay":
        return c.shift(w).over(_INST, order_by=_TS)
    if op == "ts_delta":
        return c - c.shift(w).over(_INST, order_by=_TS)
    if op == "ts_pct":
        prev = c.shift(w).over(_INST, order_by=_TS)
        return pl.when(prev.is_null() | (prev == 0)).then(None).otherwise((c - prev) / prev)
    raise KeyError(op)


def _binary_fused_expr(op: str, left: pl.Expr, right: pl.Expr, node: PlanNode) -> pl.Expr:
    if op == "add":
        return left + right
    if op == "subtract":
        return left - right
    if op == "multiply":
        return left * right
    if op == "maximum":
        return pl.max_horizontal(left, right)
    if op == "minimum":
        return pl.min_horizontal(left, right)
    if op == "protected_div":
        return _protected_div_expr(left, right, node)
    if op == "divide":
        return left / right
    if op == "power":
        return left.pow(right)
    if op == "gt":
        return pl.when(left > right).then(1.0).otherwise(0.0)
    if op == "lt":
        return pl.when(left < right).then(1.0).otherwise(0.0)
    if op == "eq":
        return pl.when(left == right).then(1.0).otherwise(0.0)
    if op == "ge":
        return pl.when(left >= right).then(1.0).otherwise(0.0)
    if op == "le":
        return pl.when(left <= right).then(1.0).otherwise(0.0)
    if op == "ne":
        return pl.when(left != right).then(1.0).otherwise(0.0)
    if op == "and_":
        return pl.when(_truthy_expr(left) & _truthy_expr(right)).then(1.0).otherwise(0.0)
    if op == "or_":
        return pl.when(_truthy_expr(left) | _truthy_expr(right)).then(1.0).otherwise(0.0)
    raise KeyError(op)


def _try_fuse_binary_ts_on_column(
    node: PlanNode,
    base: pl.LazyFrame,
    op: str,
) -> pl.LazyFrame | None:
    """DAG fusion：同列 ts 子树一次 with_columns，避免 join。"""
    if op not in _BINARY_FUSION_OPS:
        return None
    if op == "add":
        nary = _try_fuse_nary_add_ts(node, base)
        if nary is not None:
            return nary
    if len(node.inputs) != 2:
        return None
    left = _parse_ts_op_on_column(node.inputs[0])
    right = _parse_ts_op_on_column(node.inputs[1])
    if left is None or right is None:
        return None
    lop, lcol, lw = left
    rop, rcol, rw = right
    if lcol != rcol:
        return None
    schema = set(base.collect_schema().names())
    if lcol not in schema:
        return None
    tmp_l, tmp_r = "_fuse_l", "_fuse_r"
    l_expr = _ts_rolling_expr_on_column(lop, lcol, lw)
    r_expr = _ts_rolling_expr_on_column(rop, rcol, rw)
    out_expr = _binary_fused_expr(op, pl.col(tmp_l), pl.col(tmp_r), node)
    return (
        base.with_columns(l_expr.alias(tmp_l), r_expr.alias(tmp_r))
        .select(pl.col(_TS), pl.col(_INST), out_expr.alias(_VAL))
    )


def _flatten_add_ts_leaves(node: PlanNode) -> list[PlanNode] | None:
    """展开 ``add(add(ts_a, ts_b), ts_c)`` 为同列 ts 叶子列表。"""
    op = _resolve(node.op)
    if op == "add" and len(node.inputs) == 2:
        left = _flatten_add_ts_leaves(node.inputs[0])
        right = _flatten_add_ts_leaves(node.inputs[1])
        if left is None or right is None:
            return None
        return left + right
    if _parse_ts_op_on_column(node) is not None:
        return [node]
    return None


def _try_fuse_nary_add_ts(node: PlanNode, base: pl.LazyFrame) -> pl.LazyFrame | None:
    """``add(ts_mean, ts_mean, ts_delta, ...)`` 同列 n-ary fusion。"""
    leaves = _flatten_add_ts_leaves(node)
    if leaves is None or len(leaves) < 2:
        return None
    parsed = [_parse_ts_op_on_column(leaf) for leaf in leaves]
    if any(p is None for p in parsed):
        return None
    cols = {p[1] for p in parsed if p is not None}
    if len(cols) != 1:
        return None
    col_name = next(iter(cols))
    schema = set(base.collect_schema().names())
    if col_name not in schema:
        return None
    tmp_names: list[str] = []
    tmp_exprs: list[pl.Expr] = []
    for i, (ts_op, _, window) in enumerate(parsed):
        assert ts_op is not None
        name = f"_fuse_{i}"
        tmp_names.append(name)
        tmp_exprs.append(_ts_rolling_expr_on_column(ts_op, col_name, window).alias(name))
    sum_expr = pl.col(tmp_names[0])
    for name in tmp_names[1:]:
        sum_expr = sum_expr + pl.col(name)
    return (
        base.with_columns(tmp_exprs)
        .select(pl.col(_TS), pl.col(_INST), sum_expr.alias(_VAL))
    )


def _try_binary_from_base_columns(
    node: PlanNode,
    base: pl.LazyFrame,
    op: str,
) -> pl.LazyFrame | None:
    """宽表 base 上两列直接 expr，避免 left/right 子树 join。"""
    if len(node.inputs) != 2:
        return None
    left_name = _column_ref_name(node.inputs[0])
    right_name = _column_ref_name(node.inputs[1])
    if not left_name or not right_name:
        return None
    schema = set(base.collect_schema().names())
    if left_name not in schema or right_name not in schema:
        return None
    lcol = pl.col(left_name)
    rcol = pl.col(right_name)
    if op == "add":
        expr = lcol + rcol
    elif op == "subtract":
        expr = lcol - rcol
    elif op == "multiply":
        expr = lcol * rcol
    elif op == "maximum":
        expr = pl.max_horizontal(lcol, rcol)
    elif op == "minimum":
        expr = pl.min_horizontal(lcol, rcol)
    elif op == "protected_div":
        expr = _protected_div_expr(lcol, rcol, node)
    elif op == "divide":
        expr = lcol / rcol
    elif op == "power":
        expr = lcol.pow(rcol)
    elif op == "gt":
        expr = pl.when(lcol > rcol).then(1.0).otherwise(0.0)
    elif op == "lt":
        expr = pl.when(lcol < rcol).then(1.0).otherwise(0.0)
    elif op == "eq":
        expr = pl.when(lcol == rcol).then(1.0).otherwise(0.0)
    elif op == "ge":
        expr = pl.when(lcol >= rcol).then(1.0).otherwise(0.0)
    elif op == "le":
        expr = pl.when(lcol <= rcol).then(1.0).otherwise(0.0)
    elif op == "ne":
        expr = pl.when(lcol != rcol).then(1.0).otherwise(0.0)
    elif op == "and_":
        expr = pl.when(_truthy_expr(lcol) & _truthy_expr(rcol)).then(1.0).otherwise(0.0)
    elif op == "or_":
        expr = pl.when(_truthy_expr(lcol) | _truthy_expr(rcol)).then(1.0).otherwise(0.0)
    else:
        return None
    return base.select(pl.col(_TS), pl.col(_INST), expr.alias(_VAL))


def _try_coalesce_from_base_columns(node: PlanNode, base: pl.LazyFrame) -> pl.LazyFrame | None:
    if len(node.inputs) != 2:
        return None
    left_name = _column_ref_name(node.inputs[0])
    right_name = _column_ref_name(node.inputs[1])
    if not left_name or not right_name:
        return None
    schema = set(base.collect_schema().names())
    if left_name not in schema or right_name not in schema:
        return None
    expr = pl.coalesce(pl.col(left_name), pl.col(right_name))
    return base.select(pl.col(_TS), pl.col(_INST), expr.alias(_VAL))


def _try_where_from_base_columns(node: PlanNode, base: pl.LazyFrame) -> pl.LazyFrame | None:
    if len(node.inputs) != 3:
        return None
    cond_name = _column_ref_name(node.inputs[0])
    a_name = _column_ref_name(node.inputs[1])
    b_name = _column_ref_name(node.inputs[2])
    if not cond_name or not a_name or not b_name:
        return None
    schema = set(base.collect_schema().names())
    if cond_name not in schema or a_name not in schema or b_name not in schema:
        return None
    cond = pl.col(cond_name)
    expr = pl.when(_truthy_expr(cond)).then(pl.col(a_name)).otherwise(pl.col(b_name))
    return base.select(pl.col(_TS), pl.col(_INST), expr.alias(_VAL))


def _try_ts_pair_from_base_columns(
    node: PlanNode,
    base: pl.LazyFrame,
    op: str,
) -> pl.LazyFrame | None:
    """两列 base 时序/价量算子融合，避免 binary join。"""
    if len(node.inputs) != 2:
        return None
    left_name = _column_ref_name(node.inputs[0])
    right_name = _column_ref_name(node.inputs[1])
    if not left_name or not right_name:
        return None
    schema = set(base.collect_schema().names())
    if left_name not in schema or right_name not in schema:
        return None
    lcol = pl.col(left_name)
    rcol = pl.col(right_name)
    if op == "ts_corr":
        w = max(_window_int(node), 2)
        expr = pl.rolling_corr(lcol, rcol, window_size=w, min_samples=2).over(_INST, order_by=_TS)
    elif op == "ts_cov":
        w = max(_window_int(node), 2)
        expr = pl.rolling_cov(lcol, rcol, window_size=w, min_samples=2, ddof=1).over(_INST, order_by=_TS)
    elif op == "ts_beta":
        w = max(_window_int(node), 2)
        cov = pl.rolling_cov(lcol, rcol, window_size=w, min_samples=2, ddof=1).over(_INST, order_by=_TS)
        var = rcol.rolling_var(window_size=w, min_samples=2, ddof=1).over(_INST, order_by=_TS)
        expr = pl.when(var.is_null() | (var == 0)).then(None).otherwise(cov / var)
    elif op == "vwap":
        w = _window_int(node, default=20)
        pv = lcol * rcol
        sum_pv = pv.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        sum_v = rcol.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        expr = pl.when(sum_v.is_null() | (sum_v == 0)).then(None).otherwise(sum_pv / sum_v)
    else:
        return None
    return base.select(pl.col(_TS), pl.col(_INST), expr.alias(_VAL))


_BINARY_FUSION_OPS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "protected_div",
        "maximum",
        "minimum",
        "power",
        "gt",
        "lt",
        "eq",
        "ge",
        "le",
        "ne",
        "and_",
        "or_",
    }
)


def _truthy(col: str) -> pl.Expr:
    return pl.col(col).is_not_null() & (pl.col(col) != 0)


def _cs_rank_01_on(value_col: str) -> pl.Expr:
    """截面 0-1 rank，指定列名（DAG fusion 用）。"""
    r = pl.col(value_col).rank(method="average").over(_TS, order_by=_INST)
    n = pl.col(value_col).count().over(_TS, order_by=_INST)
    return (
        pl.when(n <= 1)
        .then(0.5)
        .when(pl.col(value_col).is_null())
        .then(None)
        .otherwise((r - 1.0) / (n - 1.0))
    )


def _zscore_on(value_col: str, *, canon: str = "zscore") -> pl.Expr:
    from backend.numeric_semantics import std_ddof_value, zscore_zero_std_fill

    ddof = std_ddof_value(canon)
    zero_fill = zscore_zero_std_fill(canon)
    mean = pl.col(value_col).mean().over(_TS, order_by=_INST)
    std = pl.col(value_col).std(ddof=ddof).over(_TS, order_by=_INST)
    return (
        pl.when(pl.col(value_col).is_null())
        .then(None)
        .when(std.is_null())
        .then(None)
        .when(std == 0)
        .then(zero_fill)
        .otherwise((pl.col(value_col) - mean) / std)
    )


_UNARY_FUSE_OVER_TS: frozenset[str] = frozenset(
    {"rank", "rank_pct", "cs_pct_rank", "zscore", "neg", "abs", "scale", "normalize"}
)


def _try_fuse_unary_over_ts(node: PlanNode, base: pl.LazyFrame) -> pl.LazyFrame | None:
    """``rank(ts_mean(col,w))`` 等：ts 窗口 + 一元 op 一次 with_columns，避免 join。"""
    op = _resolve(node.op)
    if op not in _UNARY_FUSE_OVER_TS or len(node.inputs) != 1:
        return None
    ts = _parse_ts_op_on_column(node.inputs[0])
    if ts is None:
        return None
    ts_op, col_name, window = ts
    schema = set(base.collect_schema().names())
    if col_name not in schema:
        return None
    tmp = "_fuse_ts"
    ts_expr = _ts_rolling_expr_on_column(ts_op, col_name, window)
    lf = base.with_columns(ts_expr.alias(tmp))
    if op == "rank":
        out_expr = _cs_rank_01_on(tmp)
    elif op in {"rank_pct", "cs_pct_rank"}:
        n = pl.col(tmp).count().over(_TS, order_by=_INST)
        frac = pl.col(tmp).rank(method="average").over(_TS, order_by=_INST) / pl.when(n > 0).then(
            n.cast(pl.Float64)
        ).otherwise(None)
        out_expr = pl.when(pl.col(tmp).is_null()).then(None).otherwise(frac)
    elif op == "zscore":
        out_expr = _zscore_on(tmp, canon="zscore")
    elif op == "neg":
        out_expr = -pl.col(tmp)
    elif op == "abs":
        out_expr = pl.col(tmp).abs()
    elif op == "scale":
        lo = pl.col(tmp).min().over(_TS, order_by=_INST)
        hi = pl.col(tmp).max().over(_TS, order_by=_INST)
        span = hi - lo
        out_expr = pl.when(span.is_null() | (span == 0)).then(0.5).otherwise((pl.col(tmp) - lo) / span)
    elif op == "normalize":
        mean = pl.col(tmp).mean().over(_TS, order_by=_INST)
        std = pl.col(tmp).std(ddof=1).over(_TS, order_by=_INST)
        out_expr = pl.when(std.is_null() | (std == 0)).then(0.0).otherwise((pl.col(tmp) - mean) / std)
    else:
        return None
    return lf.select(pl.col(_TS), pl.col(_INST), out_expr.alias(_VAL))


def _apply_compiled_long_expr(base: pl.LazyFrame, compiled: CompiledLongExpr) -> pl.LazyFrame:
    """CompiledLongExpr → LazyFrame(ts, inst, _v)。"""
    if compiled.tmp_exprs:
        lf = base.with_columns(list(compiled.tmp_exprs))
    else:
        lf = base
    return lf.select(pl.col(_TS), pl.col(_INST), compiled.expr.alias(_VAL))


def _cs_rank_01(*, canon: str = "rank") -> pl.Expr:
    """截面 0-1 rank；语义见 ``numeric_semantics.semantics_for(canon)``。"""
    from backend.numeric_semantics import rank_ignore_nan

    _ = rank_ignore_nan(canon)
    return _cs_rank_01_on(_VAL)


def _cs_rank_pct(*, canon: str = "rank_pct") -> pl.Expr:
    from backend.numeric_semantics import rank_ignore_nan

    _ = rank_ignore_nan(canon)
    n = pl.col(_VAL).count().over(_TS, order_by=_INST)
    frac = pl.col(_VAL).rank(method="average").over(_TS, order_by=_INST) / pl.when(n > 0).then(n.cast(pl.Float64)).otherwise(None)
    return pl.when(pl.col(_VAL).is_null()).then(None).otherwise(frac)


def _plan_structural_key(node: PlanNode) -> tuple[Any, ...]:
    """Plan 子树结构键（CSE 前重复子树 dedupe 编译 / join）。"""
    child_keys = tuple(_plan_structural_key(c) for c in node.inputs)
    attrs_key = tuple(sorted((str(k), repr(v)) for k, v in node.attrs.items()))
    return (_resolve(node.op), attrs_key, child_keys)


def _compile_polars(
    node: PlanNode,
    base: pl.LazyFrame,
    *,
    ctx: Any | None = None,
    memo: dict[tuple[Any, ...], pl.LazyFrame] | None = None,
) -> pl.LazyFrame | None:
    cache: dict[tuple[Any, ...], pl.LazyFrame] = {} if memo is None else memo
    key = _plan_structural_key(node)
    hit = cache.get(key)
    if hit is not None:
        return hit
    result = _compile_polars_impl(node, base, ctx=ctx, memo=cache)
    if result is not None:
        cache[key] = result
    return result


def _compile_polars_impl(
    node: PlanNode,
    base: pl.LazyFrame,
    *,
    ctx: Any | None = None,
    memo: dict[tuple[Any, ...], pl.LazyFrame] | None = None,
) -> pl.LazyFrame | None:
    if pl is None:
        return None
    op = _resolve(node.op)
    if op == "WMA":
        op = "ts_decay_linear"
    elif op == "rolling_beta":
        op = "ts_beta"
    elif op == "cum_std":
        op = "expanding_std"

    if op == "column":
        name = str(node.attrs.get("name") or "")
        if not name:
            return None
        return base.select(pl.col(_TS), pl.col(_INST), pl.col(name).alias(_VAL))

    if op == "materialized_series":
        if ctx is None:
            return None
        sid = str(node.attrs.get("sid") or "")
        lazy_mat = getattr(ctx, "materialized_long_lazy", None) or {}
        if sid in lazy_mat:
            return value_to_polars_long_lazy(lazy_mat[sid], ts_col=_TS, inst_col=_INST, value_col=_VAL)
        mat = getattr(ctx, "materialized_series", None) or {}
        if sid not in mat:
            return None
        return value_to_polars_long_lazy(mat[sid], ts_col=_TS, inst_col=_INST, value_col=_VAL)

    if op == "plan_ref":
        if ctx is None:
            return None
        sid = node.attrs.get("sid")
        lazy_cache = getattr(ctx, "shared_long_lazy_cache", None) or {}
        if sid is not None and sid in lazy_cache:
            _bump_shared_long_lazy_hit(ctx)
            return value_to_polars_long_lazy(lazy_cache[sid], ts_col=_TS, inst_col=_INST, value_col=_VAL)
        sc = getattr(ctx, "shared_result_cache", None)
        if sc is None or sid not in sc:
            return None
        return value_to_polars_long_lazy(sc[sid], ts_col=_TS, inst_col=_INST, value_col=_VAL)

    if op == "literal":
        val = node.attrs.get("value")
        return base.select(pl.col(_TS), pl.col(_INST), pl.lit(val).alias(_VAL))

    if op in _BINARY_FUSION_OPS:
        if len(node.inputs) != 2:
            return None
        fused_ts = _try_fuse_binary_ts_on_column(node, base, op)
        if fused_ts is not None:
            return fused_ts
        fused = _try_binary_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        if op == "add":
            expr = pl.col(_VAL) + pl.col("_y")
        elif op == "subtract":
            expr = pl.col(_VAL) - pl.col("_y")
        elif op == "multiply":
            expr = pl.col(_VAL) * pl.col("_y")
        elif op == "maximum":
            expr = pl.max_horizontal(pl.col(_VAL), pl.col("_y"))
        elif op == "minimum":
            expr = pl.min_horizontal(pl.col(_VAL), pl.col("_y"))
        elif op == "protected_div":
            expr = _protected_div_expr(pl.col(_VAL), pl.col("_y"), node)
        elif op == "divide":
            expr = pl.col(_VAL) / pl.col("_y")
        elif op == "power":
            expr = pl.col(_VAL).pow(pl.col("_y"))
        elif op in {"gt", "lt", "eq", "ge", "le", "ne"}:
            cmp_map = {
                "gt": pl.col(_VAL) > pl.col("_y"),
                "lt": pl.col(_VAL) < pl.col("_y"),
                "eq": pl.col(_VAL) == pl.col("_y"),
                "ge": pl.col(_VAL) >= pl.col("_y"),
                "le": pl.col(_VAL) <= pl.col("_y"),
                "ne": pl.col(_VAL) != pl.col("_y"),
            }
            expr = pl.when(cmp_map[op]).then(1.0).otherwise(0.0)
        elif op == "and_":
            expr = pl.when(_truthy(_VAL) & _truthy("_y")).then(1.0).otherwise(0.0)
        elif op == "or_":
            expr = pl.when(_truthy(_VAL) | _truthy("_y")).then(1.0).otherwise(0.0)
        else:
            return None
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "inverse":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null() | (pl.col(_VAL) == 0))
            .then(None)
            .otherwise(1.0 / pl.col(_VAL))
            .alias(_VAL)
        )

    if op == "is_finite":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            pl.col(_VAL).is_finite().cast(pl.Float64).alias(_VAL)
        )

    if op == "is_nan":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            pl.col(_VAL).is_nan().cast(pl.Float64).alias(_VAL)
        )

    if op == "neg":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        return inner.with_columns((-pl.col(_VAL)).alias(_VAL)) if inner is not None else None

    if op == "abs":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).abs().alias(_VAL)) if inner is not None else None

    if op == "sign":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).sign().alias(_VAL)) if inner is not None else None

    if op == "log":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).log().alias(_VAL)) if inner is not None else None

    if op == "exp":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).exp().alias(_VAL)) if inner is not None else None

    if op == "sqrt":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).sqrt().alias(_VAL)) if inner is not None else None

    if op == "floor":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).floor().alias(_VAL)) if inner is not None else None

    if op == "ceil":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).ceil().alias(_VAL)) if inner is not None else None

    if op == "power":
        if len(node.inputs) != 2:
            return None
        fused = _try_binary_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(pl.col(_VAL).pow(pl.col("_y")).alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "protected_log":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_protected_log_expr(pl.col(_VAL), node).alias(_VAL))

    if op == "protected_sqrt":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        clipped = pl.max_horizontal(pl.col(_VAL), pl.lit(0.0))
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null()).then(None).otherwise(clipped.sqrt()).alias(_VAL)
        )

    if op == "clip":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lo = _float_attr(node, "lo", "min", default=-3.0)
        hi = _float_attr(node, "max", "hi", default=3.0)
        pos_lo = _literal_value(node, 0)
        pos_hi = _literal_value(node, 1)
        if pos_lo is not None:
            lo = pos_lo
        if pos_hi is not None:
            hi = pos_hi
        return inner.with_columns(pl.col(_VAL).clip(lo, hi).alias(_VAL))

    if op in {"ts_mean", "ts_sum", "ts_min", "ts_max", "ts_std", "ts_var", "ts_median"}:
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from backend.numeric_semantics import std_ddof_value

        w = _window_int(node)
        ddof = std_ddof_value(op) if op in {"ts_std", "ts_var"} else 1
        if op == "ts_mean":
            expr = pl.col(_VAL).rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        elif op == "ts_sum":
            expr = pl.col(_VAL).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        elif op == "ts_min":
            expr = pl.col(_VAL).rolling_min(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        elif op == "ts_max":
            expr = pl.col(_VAL).rolling_max(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        elif op == "ts_var":
            expr = pl.col(_VAL).rolling_var(window_size=w, min_samples=1, ddof=ddof).over(_INST, order_by=_TS)
        elif op == "ts_median":
            expr = pl.col(_VAL).rolling_median(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        else:
            expr = pl.col(_VAL).rolling_std(window_size=w, min_samples=1, ddof=ddof).over(_INST, order_by=_TS)
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_zscore":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        from backend.numeric_semantics import std_ddof_value

        ddof = std_ddof_value("ts_zscore")
        mean = pl.col(_VAL).rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        std = pl.col(_VAL).rolling_std(window_size=w, min_samples=1, ddof=ddof).over(_INST, order_by=_TS)
        return inner.with_columns(
            (
                (pl.col(_VAL) - mean)
                / pl.when(std == 0).then(1.0).otherwise(std)
            ).alias(_VAL)
        )

    if op == "ts_corr":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        w = max(_window_int(node), 2)
        joined = _join_binary(left, right)
        corr = pl.rolling_corr(
            pl.col(_VAL),
            pl.col("_y"),
            window_size=w,
            min_samples=2,
        ).over(_INST, order_by=_TS)
        return joined.with_columns(corr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ts_cov":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        w = max(_window_int(node), 2)
        joined = _join_binary(left, right)
        cov = pl.rolling_cov(
            pl.col(_VAL),
            pl.col("_y"),
            window_size=w,
            min_samples=2,
            ddof=1,
        ).over(_INST, order_by=_TS)
        return joined.with_columns(cov.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ts_beta":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        w = max(_window_int(node), 2)
        joined = _join_binary(left, right)
        cov = pl.rolling_cov(
            pl.col(_VAL),
            pl.col("_y"),
            window_size=w,
            min_samples=2,
            ddof=1,
        ).over(_INST, order_by=_TS)
        var = pl.col("_y").rolling_var(window_size=w, min_samples=2, ddof=1).over(_INST, order_by=_TS)
        return joined.with_columns(
            pl.when(var.is_null() | (var == 0)).then(None).otherwise(cov / var).alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "vwap":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        price = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        vol = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if price is None or vol is None:
            return None
        w = _window_int(node, default=20)
        joined = _join_binary(price, vol)
        pv = pl.col(_VAL) * pl.col("_y")
        sum_pv = pv.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        sum_v = pl.col("_y").rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        return joined.with_columns(
            pl.when(sum_v.is_null() | (sum_v == 0))
            .then(None)
            .otherwise(sum_pv / sum_v)
            .alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "cum_sum":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).cum_sum().over(_INST, order_by=_TS).alias(_VAL))

    if op == "cum_max":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).cum_max().over(_INST, order_by=_TS).alias(_VAL))

    if op == "cum_min":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).cum_min().over(_INST, order_by=_TS).alias(_VAL))

    if op == "cum_prod":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).cum_prod().over(_INST, order_by=_TS).alias(_VAL))

    if op == "ts_rank":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        rank = (
            pl.col(_VAL)
            .rolling_rank(window_size=w, min_samples=1, method="average")
            .over(_INST, order_by=_TS)
        )
        cnt = (
            pl.col(_VAL)
            .is_not_null()
            .cast(pl.Float64)
            .rolling_sum(window_size=w, min_samples=1)
            .over(_INST, order_by=_TS)
        )
        return inner.with_columns((rank / cnt).alias(_VAL))

    if op in {"ewm_mean", "ewm_std", "ewm_var"}:
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        alpha = _ewm_alpha(node)
        if op == "ewm_mean":
            expr = pl.col(_VAL).ewm_mean(alpha=alpha, adjust=False)
        elif op == "ewm_std":
            expr = pl.col(_VAL).ewm_std(alpha=alpha, adjust=False)
        else:
            expr = pl.col(_VAL).ewm_var(alpha=alpha, adjust=False)
        return inner.with_columns(expr.over(_INST, order_by=_TS).alias(_VAL))

    if op in {"coalesce"}:
        if len(node.inputs) != 2:
            return None
        fused = _try_coalesce_from_base_columns(node, base)
        if fused is not None:
            return fused
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(pl.coalesce(pl.col(_VAL), pl.col("_y")).alias(_VAL)).select(_TS, _INST, _VAL)

    if op in {"where", "if_else"}:
        if len(node.inputs) != 3:
            return None
        fused = _try_where_from_base_columns(node, base)
        if fused is not None:
            return fused
        cond = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        a = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        b = _compile_polars(node.inputs[2], base, ctx=ctx, memo=memo)
        if cond is None or a is None or b is None:
            return None
        joined = _join_triple(cond, a, b)
        return joined.with_columns(
            pl.when(_truthy(_VAL)).then(pl.col("_ym")).otherwise(pl.col("_y")).alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op in {"gt", "lt", "eq", "ge", "le", "ne"}:
        if len(node.inputs) != 2:
            return None
        fused = _try_binary_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        cmp_map = {
            "gt": pl.col(_VAL) > pl.col("_y"),
            "lt": pl.col(_VAL) < pl.col("_y"),
            "eq": pl.col(_VAL) == pl.col("_y"),
            "ge": pl.col(_VAL) >= pl.col("_y"),
            "le": pl.col(_VAL) <= pl.col("_y"),
            "ne": pl.col(_VAL) != pl.col("_y"),
        }
        return joined.with_columns(
            pl.when(cmp_map[op]).then(1.0).otherwise(0.0).alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "and_":
        if len(node.inputs) != 2:
            return None
        fused = _try_binary_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(
            pl.when(_truthy(_VAL) & _truthy("_y")).then(1.0).otherwise(0.0).alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "or_":
        if len(node.inputs) != 2:
            return None
        fused = _try_binary_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(
            pl.when(_truthy(_VAL) | _truthy("_y")).then(1.0).otherwise(0.0).alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "not_":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            pl.when(_truthy(_VAL)).then(0.0).otherwise(1.0).alias(_VAL)
        )

    if op in {"fillna_const", "fillna"}:
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        if op == "fillna":
            const = _const_fill_value(node)
            if const is None:
                return None
        else:
            const = _const_fill_value(node, default=0.0)
            if const is None:
                const = _float_attr(node, "value", "fill_value", default=0.0)
        return inner.with_columns(
            pl.col(_VAL).fill_nan(const).fill_null(const).alias(_VAL)
        )

    if op == "nan_to_num":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        const = _literal_value(node, 0, default=0.0) or 0.0
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_nan()).then(const).otherwise(pl.col(_VAL)).alias(_VAL)
        )

    if op in {"cs_quantile", "c_percentile"}:
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        p = _float_attr(node, "p", default=0.5)
        pos_p = _literal_value(node, 0)
        if pos_p is not None:
            p = pos_p
        q = pl.col(_VAL).quantile(quantile=p, interpolation="linear").over(_TS, order_by=_INST)
        return inner.with_columns(q.alias(_VAL))

    if op == "winsorize":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lo_p = _literal_value(node, 0)
        hi_p = _literal_value(node, 1)
        if lo_p is None:
            if "a" in node.attrs or "p" in node.attrs:
                a = _float_attr(node, "a", "p", default=0.05)
                lo_p, hi_p = a, 1.0 - a
            else:
                lo_p = _float_attr(node, "min_pct", "lower", default=0.05)
        if hi_p is None:
            hi_p = _float_attr(node, "max_pct", "upper", default=0.95)
        lo = pl.col(_VAL).quantile(quantile=lo_p, interpolation="linear").over(_TS, order_by=_INST)
        hi = pl.col(_VAL).quantile(quantile=hi_p, interpolation="linear").over(_TS, order_by=_INST)
        return inner.with_columns(pl.col(_VAL).clip(lo, hi).alias(_VAL))

    if op in {"cs_resid", "cs_regression"}:
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        x_layer = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if y_layer is None or x_layer is None:
            return None
        joined = y_layer.join(x_layer.rename({_VAL: "_x"}), on=[_TS, _INST], how="inner")
        beta, alpha, n_valid = _cs_ols_exprs(_VAL, "_x")
        fit = alpha + beta * pl.col("_x")
        mode = 0 if op == "cs_resid" else _int_attr(node, "mode", input_index=2, default=0)
        if mode == 1:
            core = beta
        elif mode == 2:
            core = fit
        else:
            core = pl.col(_VAL) - fit
        return joined.with_columns(
            pl.when(pl.col(_VAL).is_null() | pl.col("_x").is_null())
            .then(None)
            .when(n_valid < 3)
            .then(None)
            .otherwise(core)
            .alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op in {
        "group_rank",
        "group_mean",
        "group_zscore",
        "group_neutralize",
        "group_std",
        "group_normalize",
        "group_percentile",
        "group_winsorize",
        "group_decay_linear",
    }:
        if not node.inputs:
            return None
        val = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if val is None:
            return None
        if len(node.inputs) >= 2 and node.inputs[1].op not in {"literal"}:
            grp = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
            if grp is None:
                return None
            joined = val.join(grp.rename({_VAL: _GRP}), on=[_TS, _INST], how="inner")
            over_keys = (_TS, _GRP)
        else:
            joined = val.with_columns(pl.lit(1.0).alias(_GRP))
            over_keys = (_TS, _GRP)
        if op == "group_percentile":
            p = _float_attr(node, "p", default=0.5)
            pos_p = _literal_value(node, 1)
            if pos_p is not None:
                p = pos_p
            n = pl.col(_VAL).count().over(*over_keys, order_by=_INST)
            frac = (
                pl.col(_VAL).rank(method="average").over(*over_keys, order_by=_INST)
                / pl.when(n > 0).then(n.cast(pl.Float64)).otherwise(None)
            )
            expr = (
                pl.when(pl.col(_VAL).is_null())
                .then(0.0)
                .when(frac <= p)
                .then(1.0)
                .otherwise(0.0)
            )
        elif op == "group_winsorize":
            a = _float_attr(node, "a", "p", default=0.05)
            lo_p = a
            hi_p = 1.0 - a
            lo = pl.col(_VAL).quantile(quantile=lo_p, interpolation="linear").over(*over_keys, order_by=_INST)
            hi = pl.col(_VAL).quantile(quantile=hi_p, interpolation="linear").over(*over_keys, order_by=_INST)
            expr = pl.when(pl.col(_VAL).is_null()).then(None).otherwise(pl.col(_VAL).clip(lo, hi))
        elif op == "group_decay_linear":
            n = pl.col(_VAL).count().over(*over_keys, order_by=_INST)
            rank = pl.col(_VAL).rank(method="ordinal").over(*over_keys, order_by=_INST)
            denom = n.cast(pl.Float64) * (n.cast(pl.Float64) + 1.0) / 2.0
            expr = (
                pl.when(pl.col(_VAL).is_null())
                .then(None)
                .when(denom.is_null() | (denom == 0))
                .then(None)
                .otherwise(pl.col(_VAL) * rank / denom)
            )
        elif op == "group_mean":
            expr = pl.col(_VAL).mean().over(*over_keys, order_by=_INST)
        elif op == "group_std":
            from backend.numeric_semantics import std_ddof_value

            expr = pl.col(_VAL).std(ddof=std_ddof_value("group_std")).over(*over_keys, order_by=_INST)
        elif op in {"group_zscore", "group_neutralize"}:
            from backend.numeric_semantics import std_ddof_value, zscore_zero_std_fill

            mean = pl.col(_VAL).mean().over(*over_keys, order_by=_INST)
            if op == "group_neutralize":
                expr = pl.when(pl.col(_VAL).is_null()).then(None).otherwise(pl.col(_VAL) - mean)
            else:
                std = pl.col(_VAL).std(ddof=std_ddof_value("group_zscore")).over(*over_keys, order_by=_INST)
                zero_fill = zscore_zero_std_fill("group_zscore")
                expr = (
                    pl.when(pl.col(_VAL).is_null())
                    .then(None)
                    .when(std.is_null() | (std == 0))
                    .then(zero_fill)
                    .otherwise((pl.col(_VAL) - mean) / std)
                )
        elif op == "group_normalize":
            lo = pl.col(_VAL).min().over(*over_keys, order_by=_INST)
            hi = pl.col(_VAL).max().over(*over_keys, order_by=_INST)
            span = hi - lo
            expr = pl.when(span.is_null() | (span == 0)).then(0.5).otherwise((pl.col(_VAL) - lo) / span)
        else:
            n = pl.col(_VAL).count().over(*over_keys, order_by=_INST)
            frac = (
                pl.col(_VAL).rank(method="average").over(*over_keys, order_by=_INST)
                / pl.when(n > 0).then(n.cast(pl.Float64)).otherwise(None)
            )
            expr = pl.when(pl.col(_VAL).is_null()).then(None).otherwise(frac)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "cs_mad":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        med = pl.col(_VAL).median().over(_TS, order_by=_INST)
        mad = (pl.col(_VAL) - med).abs().median().over(_TS, order_by=_INST)
        return inner.with_columns(mad.alias(_VAL))

    if op == "cs_mad_zscore":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        med = pl.col(_VAL).median().over(_TS, order_by=_INST)
        mad = (pl.col(_VAL) - med).abs().median().over(_TS, order_by=_INST)
        expr = (
            pl.when(mad.is_null() | (mad == 0))
            .then(None)
            .otherwise((pl.col(_VAL) - med) / mad)
        )
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_delay":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return inner.with_columns(pl.col(_VAL).shift(lag).over(_INST, order_by=_TS).alias(_VAL))

    if op == "ts_delta":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        delayed = pl.col(_VAL).shift(lag).over(_INST, order_by=_TS)
        return inner.with_columns((pl.col(_VAL) - delayed).alias(_VAL))

    if op == "ts_pct":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        prev = pl.col(_VAL).shift(lag).over(_INST, order_by=_TS)
        return inner.with_columns((pl.col(_VAL) / prev - 1.0).alias(_VAL))

    if op == "ffill":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).forward_fill().over(_INST, order_by=_TS).alias(_VAL))

    if op == "bfill":
        import logging

        logging.getLogger("backend.polars_expr_emitter").warning(
            "bfill on polars_long is a no-op (causal/PIT); use ffill or research-only fallback"
        )
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner

    if op in {"ts_ema", "ewm_mean"}:
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        alpha = _ewm_alpha(node)
        return inner.with_columns(
            pl.col(_VAL).ewm_mean(alpha=alpha, adjust=False).over(_INST, order_by=_TS).alias(_VAL)
        )

    if op == "rank":
        fused = _try_fuse_unary_over_ts(node, base)
        if fused is not None:
            return fused
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_cs_rank_01().alias(_VAL))

    if op in {"rank_pct", "cs_pct_rank"}:
        fused = _try_fuse_unary_over_ts(node, base)
        if fused is not None:
            return fused
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_cs_rank_pct().alias(_VAL))

    if op == "zscore":
        fused = _try_fuse_unary_over_ts(node, base)
        if fused is not None:
            return fused
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from backend.numeric_semantics import std_ddof_value, zscore_zero_std_fill

        ddof = std_ddof_value("zscore")
        zero_fill = zscore_zero_std_fill("zscore")
        mean = pl.col(_VAL).mean().over(_TS, order_by=_INST)
        std = pl.col(_VAL).std(ddof=ddof).over(_TS, order_by=_INST)
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null())
            .then(None)
            .when(std.is_null())
            .then(None)
            .when(std == 0)
            .then(zero_fill)
            .otherwise((pl.col(_VAL) - mean) / std)
            .alias(_VAL)
        )

    if op == "cs_demean":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        mean = pl.col(_VAL).mean().over(_TS, order_by=_INST)
        return inner.with_columns((pl.col(_VAL) - mean).alias(_VAL))

    if op == "normalize":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lo = pl.col(_VAL).min().over(_TS, order_by=_INST)
        hi = pl.col(_VAL).max().over(_TS, order_by=_INST)
        span = hi - lo
        return inner.with_columns(
            pl.when(span.is_null() | (span == 0))
            .then(0.5)
            .otherwise((pl.col(_VAL) - lo) / span)
            .alias(_VAL)
        )

    if op == "scale":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        s = pl.col(_VAL).abs().sum().over(_TS, order_by=_INST)
        return inner.with_columns(
            pl.when(s.is_null() | (s == 0))
            .then(0.0)
            .otherwise(pl.col(_VAL) / s)
            .alias(_VAL)
        )

    if op == "log_returns":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        prev = pl.col(_VAL).shift(1).over(_INST, order_by=_TS)
        return inner.with_columns(
            pl.when(
                pl.col(_VAL).is_null()
                | prev.is_null()
                | (pl.col(_VAL) <= 0)
                | (prev <= 0)
            )
            .then(None)
            .otherwise((pl.col(_VAL) / prev).log())
            .alias(_VAL)
        )

    if op == "volatility":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        mp = max(2, w // 2)
        scale = float(252**0.5)
        return inner.with_columns(
            (
                pl.col(_VAL)
                .rolling_std(window_size=w, min_samples=mp, ddof=1)
                .over(_INST, order_by=_TS)
                * scale
            ).alias(_VAL)
        )

    if op in {"c_mean", "c_std", "c_sum", "c_count"}:
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        if op == "c_mean":
            expr = pl.col(_VAL).mean().over(_TS, order_by=_INST)
        elif op == "c_std":
            expr = pl.col(_VAL).std(ddof=1).over(_TS, order_by=_INST)
        elif op == "c_sum":
            expr = pl.col(_VAL).sum().over(_TS, order_by=_INST)
        else:
            expr = pl.col(_VAL).count().over(_TS, order_by=_INST)
        return inner.with_columns(expr.alias(_VAL))

    if op == "log_abs":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).abs().log().alias(_VAL))

    if op == "signed_log":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            (pl.col(_VAL).sign() * (pl.col(_VAL).abs() + 1e-10).log()).alias(_VAL)
        )

    if op == "signed_sqrt":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            (pl.col(_VAL).sign() * pl.col(_VAL).abs().sqrt()).alias(_VAL)
        )

    if op == "ts_decay_linear":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_linear_decay_expr(w).alias(_VAL))

    if op == "ts_mad":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        tmp = inner.with_columns(
            pl.col(_VAL)
            .rolling_median(window_size=w, min_samples=1)
            .over(_INST, order_by=_TS)
            .alias("_med")
        )
        return tmp.with_columns(
            (pl.col(_VAL) - pl.col("_med"))
            .abs()
            .rolling_mean(window_size=w, min_samples=1)
            .over(_INST, order_by=_TS)
            .alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "ts_quantile":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        p = _float_attr(node, "q", "p", default=0.5)
        pos_p = _literal_value(node, 1)
        if pos_p is not None:
            p = pos_p
        return inner.with_columns(_rolling_quantile_expr(w, p).alias(_VAL))

    if op == "ts_product":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        eps = 1e-12
        ln_sum = (
            pl.when(pl.col(_VAL).is_null() | (pl.col(_VAL) <= eps))
            .then(None)
            .otherwise(pl.col(_VAL).log())
            .rolling_sum(window_size=w, min_samples=1)
            .over(_INST, order_by=_TS)
        )
        return inner.with_columns(ln_sum.exp().alias(_VAL))

    if op == "ts_skew":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_skew_expr(w).alias(_VAL))

    if op == "ts_argmax":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_argext_expr(w, pick="max").alias(_VAL))

    if op == "ts_argmin":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_argext_expr(w, pick="min").alias(_VAL))

    if op == "Slope":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_time_slope_expr(w).alias(_VAL))

    if op == "ts_regression":
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        x_layer = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if y_layer is None or x_layer is None:
            return None
        w = max(_window_int(node), 3)
        joined = y_layer.join(x_layer.rename({_VAL: "_x"}), on=[_TS, _INST], how="inner")
        beta, alpha, n_valid, fit = _rolling_ols_parts(_VAL, "_x", w)
        mode = _ts_regression_retval(node)
        if mode == "intercept":
            core = alpha
        elif mode == "fit":
            core = fit
        elif mode == "resid":
            core = pl.col(_VAL) - fit
        else:
            core = beta
        return joined.with_columns(
            pl.when(pl.col(_VAL).is_null() | pl.col("_x").is_null())
            .then(None)
            .when(n_valid < 3)
            .then(None)
            .otherwise(core)
            .alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "ts_sharpe":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = max(_window_int(node), 2)
        mp = _ts_sharpe_min_periods(w)
        ann = _float_attr(node, "ann_factor", default=252.0)
        sqrt_af = ann**0.5
        mean = pl.col(_VAL).rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        std = pl.col(_VAL).rolling_std(window_size=w, min_samples=mp, ddof=1).over(_INST, order_by=_TS)
        expr = (
            pl.when((std == 0) & (mean > 0))
            .then(float("inf"))
            .when(std == 0)
            .then(0.0)
            .otherwise(mean / std)
            * sqrt_af
        )
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_autocorr":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w, lag, mp = _ts_autocorr_window_lag(node)
        lagged = pl.col(_VAL).shift(lag).over(_INST, order_by=_TS)
        tmp = inner.with_columns(lagged.alias("_lag"))
        corr = pl.rolling_corr(
            pl.col(_VAL),
            pl.col("_lag"),
            window_size=w,
            min_samples=mp,
        ).over(_INST, order_by=_TS)
        return tmp.with_columns(corr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "cum_delta":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        first = (
            inner.sort(_TS)
            .filter(pl.col(_VAL).is_not_null())
            .group_by(_INST)
            .agg(pl.col(_VAL).first().alias("_first"))
        )
        joined = inner.join(first, on=_INST, how="left")
        return joined.with_columns(
            pl.when(pl.col(_VAL).is_null())
            .then(None)
            .otherwise(pl.col(_VAL) - pl.col("_first"))
            .alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "expanding_sum":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_expanding_sum_expr().alias(_VAL))

    if op == "expanding_mean":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_expanding_mean_expr().alias(_VAL))

    if op == "expanding_std":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_expanding_std_expr().alias(_VAL))

    if op == "count":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_expanding_non_null_count().alias(_VAL))

    if op in {"ewm_corr", "ewm_cov"}:
        if len(node.inputs) < 2:
            return None
        left = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        right = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        span = max(_window_int(node, default=20), 2)
        joined = _join_binary(left, right)
        return _ewm_binary_map_groups(joined, span, corr=(op == "ewm_corr"))

    if op == "ts_ratio":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        prev = pl.col(_VAL).shift(1).over(_INST, order_by=_TS)
        return inner.with_columns(
            pl.when(prev.is_null() | (prev == 0))
            .then(None)
            .otherwise(pl.col(_VAL) / prev)
            .alias(_VAL)
        )

    if op == "ts_kurt":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)

        def _kurt(arr: np.ndarray) -> float:
            s = pd.Series(np.asarray(arr, dtype=np.float64))
            if s.count() == 0:
                return np.nan
            val = s.kurt()
            return np.nan if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)

        return inner.with_columns(
            pl.col(_VAL).rolling_map(_kurt, window_size=w, min_samples=1).over(_INST, order_by=_TS).alias(_VAL)
        )

    if op == "ts_moment":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = max(_int_attr(node, "d", "window", input_index=0, default=3), 1)
        k = max(_int_attr(node, "k", "order", input_index=1, default=2), 1)
        from cleaned_operators._numpy_kernels import ts_moment_

        return _unary_inst_map_groups(inner, lambda arr: ts_moment_(arr, w, k))

    if op == "ts_max_buildup":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        from cleaned_operators._numpy_kernels import ts_max_buildup_

        return _unary_inst_map_groups(inner, lambda arr: ts_max_buildup_(arr, w))

    if op == "expanding_rank":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None

        def _expanding_rank(arr: np.ndarray) -> np.ndarray:
            ranked = pd.Series(arr, dtype=float).expanding(min_periods=1).rank(pct=True)
            return ranked.to_numpy(dtype=float)

        return _unary_inst_map_groups(inner, _expanding_rank)

    if op == "fillna_interpolate":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        method = "linear"
        for key in ("method", "interp"):
            if key in node.attrs and node.attrs[key] is not None:
                method = str(node.attrs[key])
                break
        else:
            if len(node.inputs) > 1 and node.inputs[1].op == "literal":
                raw = node.inputs[1].attrs.get("value")
                if isinstance(raw, str):
                    method = raw
        from cleaned_operators._causal import causal_interpolate_panel

        def _interp(arr: np.ndarray) -> np.ndarray:
            pdf = pd.DataFrame({"v": arr.astype(float, copy=False)})
            return causal_interpolate_panel(pdf, method=method)["v"].to_numpy(dtype=float)

        return _unary_inst_map_groups(inner, _interp)

    if op == "quantile":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        n_bins = max(_int_attr(node, "bins", input_index=0, default=10), 0)

        def _qcut(arr: np.ndarray) -> np.ndarray:
            out = np.full(arr.shape, np.nan, dtype=float)
            valid_mask = np.isfinite(arr)
            valid = arr[valid_mask]
            if len(valid) < 2:
                return out
            try:
                labels = pd.qcut(valid, q=n_bins, labels=False, duplicates="drop")
                out[valid_mask] = labels.astype(float)
            except (ValueError, TypeError):
                pass
            return out

        return _cs_row_map_groups(inner, _qcut)

    if op == "RSI_WILDER":
        inner = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        if inner is None:
            return None
        alpha, w = _wilder_alpha(node)
        delta = pl.col(_VAL).diff().over(_INST, order_by=_TS)
        gain = (
            pl.when(delta.is_null())
            .then(None)
            .when(delta > 0)
            .then(delta)
            .otherwise(0.0)
        )
        loss = (
            pl.when(delta.is_null())
            .then(None)
            .when(delta < 0)
            .then(-delta)
            .otherwise(0.0)
        )
        avg_gain = gain.ewm_mean(alpha=alpha, adjust=False, min_periods=w).over(_INST, order_by=_TS)
        avg_loss = loss.ewm_mean(alpha=alpha, adjust=False, min_periods=w).over(_INST, order_by=_TS)
        rs = avg_gain / pl.when(avg_loss == 0).then(None).otherwise(avg_loss)
        expr = (
            pl.when((avg_loss == 0) & (avg_gain > 0))
            .then(100.0)
            .when((avg_gain == 0) & (avg_loss > 0))
            .then(0.0)
            .when((avg_gain == 0) & (avg_loss == 0))
            .then(50.0)
            .otherwise(100.0 - (100.0 / (1.0 + rs)))
        )
        return inner.with_columns(expr.alias(_VAL))

    if op == "ATR_WILDER":
        if len(node.inputs) < 3:
            return None
        high = _compile_polars(node.inputs[0], base, ctx=ctx, memo=memo)
        low = _compile_polars(node.inputs[1], base, ctx=ctx, memo=memo)
        close = _compile_polars(node.inputs[2], base, ctx=ctx, memo=memo)
        if high is None or low is None or close is None:
            return None
        alpha, w = _wilder_alpha(node)
        joined = _join_triple(high, low, close)
        prev_close = pl.col("_y").shift(1).over(_INST, order_by=_TS)
        tr = (
            pl.when(prev_close.is_null())
            .then(None)
            .otherwise(
                pl.max_horizontal(
                    pl.col(_VAL) - pl.col("_ym"),
                    (pl.col(_VAL) - prev_close).abs(),
                    (pl.col("_ym") - prev_close).abs(),
                )
            )
        )
        expr = tr.ewm_mean(alpha=alpha, adjust=False, min_periods=w).over(_INST, order_by=_TS)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    from .polars_registry_bridge import compile_registry_op

    return compile_registry_op(node, base, lambda n, b: _compile_polars(n, b, ctx=ctx, memo=memo))


def _first_long_lazy_from_plan(plan: PlanNode, ctx: Any) -> Any | None:
    """从 ``plan_ref`` / ``materialized_series`` 缓存取 long LazyFrame。"""
    if plan.op == "plan_ref":
        sid = plan.attrs.get("sid")
        lazy_cache = getattr(ctx, "shared_long_lazy_cache", None) or {}
        if sid is not None and sid in lazy_cache:
            return lazy_cache[sid]
    if plan.op == "materialized_series":
        sid = plan.attrs.get("sid")
        lazy_mat = getattr(ctx, "materialized_long_lazy", None) or {}
        if sid is not None and sid in lazy_mat:
            return lazy_mat[sid]
    for child in plan.inputs:
        found = _first_long_lazy_from_plan(child, ctx)
        if found is not None:
            return found
    return None


def _first_series_spine_from_plan(plan: PlanNode, ctx: Any) -> pd.Series | None:
    """从 ``plan_ref`` / ``materialized_series`` 推断 ts/inst 骨架（无 column 引用时）。"""
    if plan.op == "plan_ref":
        sc = getattr(ctx, "shared_result_cache", None) or {}
        sid = plan.attrs.get("sid")
        val = sc.get(sid) if sid is not None else None
        if isinstance(val, pd.Series):
            return val
    if plan.op == "materialized_series":
        mat = getattr(ctx, "materialized_series", None) or {}
        sid = plan.attrs.get("sid")
        val = mat.get(sid) if sid is not None else None
        if isinstance(val, pd.Series):
            return val
    for child in plan.inputs:
        found = _first_series_spine_from_plan(child, ctx)
        if found is not None:
            return found
    return None


def resolve_base_lazy_for_plan(
    plan: PlanNode,
    ctx: Any,
    scan_fn: Any,
) -> pl.LazyFrame:
    """构建 long-table 基础 LazyFrame：优先 scan 引用列，否则从缓存 Series 推断骨架。"""
    columns = collect_columns(plan)
    if columns:
        return scan_fn(sorted(columns))
    lazy_spine = _first_long_lazy_from_plan(plan, ctx)
    if lazy_spine is not None:
        return value_to_polars_long_lazy(lazy_spine, ts_col=_TS, inst_col=_INST, value_col=_VAL)
    spine = _first_series_spine_from_plan(plan, ctx)
    if spine is not None:
        return series_to_polars_long_lazy(spine)
    raise ValueError("scan_polars_long: no columns")


def _build_base_lazy(ctx: Any, columns: set[str]) -> pl.LazyFrame:
    """从 DataSource 构建 LazyFrame；优先 ``scan_polars_long``。"""
    scan = getattr(ctx.data_source, "scan_polars_long", None)
    if callable(scan):
        return scan(sorted(columns))
    from .polars_long_policy import PolarsLongStrictError, strict_polars_long_fallback

    if strict_polars_long_fallback(ctx):
        raise PolarsLongStrictError("data_source lacks scan_polars_long")
    return _build_base_lazy_from_series(ctx, columns)


def _build_base_lazy_from_series(ctx: Any, columns: set[str]) -> pl.LazyFrame:
    if pl is None:
        raise ImportError("polars is required for polars long-table backend")
    from storage.factor_format import series_to_long_table

    tcol = ctx.timestamp_col
    icol = ctx.instrument_col
    merged: pd.DataFrame | None = None
    for name in sorted(columns):
        series = ctx.data_source.load_column(name)
        part = series_to_long_table(
            series,
            timestamp_col=tcol,
            asset_col=icol,
            value_col=name,
        )
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=[tcol, icol], how="outer")
        merged[name] = merged[name].astype("float64")
    if merged is None:
        raise ValueError("polars long: no columns referenced")
    renamed = merged.rename(columns={tcol: _TS, icol: _INST})
    return pl.from_pandas(renamed).lazy()


def compile_plan_to_polars(
    plan: PlanNode,
    base_lf: pl.LazyFrame,
    *,
    ctx: Any | None = None,
    ts_col: str = _TS,
    inst_col: str = _INST,
) -> LongFrameResult | None:
    """编译 PlanNode 为 long-table Polars LazyFrame + 结果列名。"""
    if pl is None:
        return None
    compiled = _compile_polars(plan, base_lf, ctx=ctx)
    if compiled is None:
        return None
    return LongFrameResult(
        frame=compiled,
        value_col=_VAL,
        ts_col=ts_col,
        inst_col=inst_col,
        is_lazy=True,
    )


def compile_polars_long_lazy(
    plan: PlanNode,
    ctx: Any,
    *,
    base_lf: pl.LazyFrame | None = None,
    lazy_cache_key: str | None = None,
) -> LongFrameResult:
    """编译 long-table 计划为 LazyFrame（不 collect）。"""
    if base_lf is None:
        scan = getattr(ctx.data_source, "scan_polars_long", None)
        if callable(scan):
            base = resolve_base_lazy_for_plan(plan, ctx, scan)
        else:
            cols = collect_columns(plan)
            base = _build_base_lazy(ctx, cols)
    else:
        base = base_lf
    compiled = compile_plan_to_polars(plan, base, ctx=ctx)
    if compiled is None:
        raise RuntimeError(f"polars long compile failed for op={plan.op!r}")
    if lazy_cache_key:
        cache = getattr(ctx, "shared_long_lazy_cache", None)
        if cache is not None:
            cache[lazy_cache_key] = compiled.frame
    return compiled


def execute_polars_long_plan(
    plan: PlanNode,
    ctx: Any,
    *,
    base_lf: pl.LazyFrame | None = None,
    lazy_cache_key: str | None = None,
) -> pd.Series:
    """编译并执行 long-table 计划；仅最终 collect 一次。"""
    compiled = compile_polars_long_lazy(
        plan, ctx, base_lf=base_lf, lazy_cache_key=lazy_cache_key
    )
    frame = (
        compiled.frame.sort([compiled.ts_col, compiled.inst_col])
        .select(
            pl.col(compiled.ts_col).alias(ctx.timestamp_col),
            pl.col(compiled.inst_col).alias(ctx.instrument_col),
            pl.col(compiled.value_col).alias("value"),
        )
        .collect()
    )
    return polars_long_to_multiindex_series(
        frame,
        timestamp_col=ctx.timestamp_col,
        instrument_col=ctx.instrument_col,
        value_col="value",
        template_index=optional_universe_index(ctx),
    ).sort_index()


def execute_polars_expr_plan(plan: PlanNode, ctx: Any) -> pd.Series:
    """兼容旧名：``PolarsBackend`` + ``FACTOR_ENGINE_POLARS_EXPR=1``。"""
    return execute_polars_long_plan(plan, ctx)
