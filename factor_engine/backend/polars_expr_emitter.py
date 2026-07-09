# -*- coding: utf-8
"""PlanNode → Polars Expr 编译器（long-table native，对齐 SQL emitter 的 ts/inst/value 语义）。

``compile_plan_to_polars`` 不经过 ``cleaned_bridge`` / panel 往返；
由 ``PolarsLongBackend`` 或（兼容）``FACTOR_ENGINE_POLARS_EXPR=1`` 调用。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from planner.logical_plan import PlanNode

from .long_frame import LongFrameResult, polars_long_to_multiindex_series
from .pandas_compat import pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

_TS = "ts"
_INST = "inst"
_VAL = "_v"

_GRP = "_grp"

# expanding / cum 类算子：rolling 窗口需覆盖单 inst 全长（与 SQL UNBOUNDED PRECEDING 对齐）
_EXPANDING_WINDOW = 100_000

POLARS_EXPR_CAPABLE: frozenset[str] = frozenset(
    {
        "column",
        "literal",
        "add",
        "subtract",
        "multiply",
        "divide",
        "neg",
        "abs",
        "sign",
        "log",
        "exp",
        "sqrt",
        "clip",
        "power",
        "floor",
        "ceil",
        "protected_div",
        "protected_log",
        "protected_sqrt",
        "ts_mean",
        "ts_sum",
        "ts_std",
        "ts_var",
        "ts_median",
        "ts_min",
        "ts_max",
        "ts_delay",
        "ts_delta",
        "ts_pct",
        "ts_zscore",
        "ts_corr",
        "ffill",
        "cum_sum",
        "cum_max",
        "cum_min",
        "cum_prod",
        "ewm_mean",
        "ewm_std",
        "ewm_var",
        "ts_rank",
        "maximum",
        "minimum",
        "inverse",
        "is_finite",
        "is_nan",
        "rank",
        "zscore",
        "cs_demean",
        "scale",
        "normalize",
        "rank_pct",
        "cs_pct_rank",
        "cs_quantile",
        "c_percentile",
        "winsorize",
        "log_returns",
        "volatility",
        "vwap",
        "ts_beta",
        "ts_cov",
        "coalesce",
        "where",
        "fillna_const",
        "fillna",
        "nan_to_num",
        "gt",
        "lt",
        "eq",
        "ge",
        "le",
        "ne",
        "and_",
        "or_",
        "not_",
        "group_rank",
        "group_mean",
        "group_zscore",
        "group_neutralize",
        "group_std",
        "group_normalize",
        "group_percentile",
        "group_winsorize",
        "group_decay_linear",
        "cs_mad",
        "cs_mad_zscore",
        "cs_resid",
        "cs_regression",
        "ts_ema",
        "ewm_mean",
        "c_mean",
        "c_std",
        "c_sum",
        "c_count",
        "log_abs",
        "signed_log",
        "signed_sqrt",
        "ts_decay_linear",
        "WMA",
        "ts_mad",
        "ts_quantile",
        "ts_product",
        "ts_skew",
        "ts_argmax",
        "ts_argmin",
        "ts_regression",
        "Slope",
        "rolling_beta",
        "ts_sharpe",
        "ts_autocorr",
        "cum_delta",
        "expanding_mean",
        "expanding_std",
        "expanding_sum",
        "count",
        "ewm_corr",
        "ewm_cov",
        "ts_ratio",
        "ts_kurt",
        "ts_moment",
        "ts_max_buildup",
        "expanding_rank",
        "fillna_interpolate",
        "quantile",
        "ATR_WILDER",
        "RSI_WILDER",
    }
)

# 与 SQL emitter 白名单对齐的别名（long-table native 路径）
POLARS_LONG_CAPABLE = POLARS_EXPR_CAPABLE


def plan_is_polars_long_capable(plan: PlanNode) -> bool:
    return plan_is_polars_expr_capable(plan)


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
    for idx in range(len(node.inputs) - 1, 0, -1):
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
    """截面 OLS y ~ x + const（按 ts 分区）。"""
    mean_y = pl.col(y_col).mean().over(ts)
    mean_x = pl.col(x_col).mean().over(ts)
    xy = (pl.col(y_col) * pl.col(x_col)).mean().over(ts)
    x2 = (pl.col(x_col) ** 2).mean().over(ts)
    cov_xy = xy - mean_x * mean_y
    var_x = x2 - mean_x**2
    beta = cov_xy / pl.when(var_x.is_null() | (var_x == 0)).then(1.0).otherwise(var_x)
    alpha = mean_y - beta * mean_x
    n_valid = (
        pl.when(pl.col(y_col).is_not_null() & pl.col(x_col).is_not_null())
        .then(1)
        .otherwise(0)
        .sum()
        .over(ts)
    )
    return beta, alpha, n_valid


def _rolling_ols_parts(
    y_col: str,
    x_col: str,
    w: int,
    *,
    inst: str = _INST,
    ts: str = _TS,
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """滚动 OLS y ~ x + const（按 inst 分区）。"""
    mean_y = pl.col(y_col).rolling_mean(window_size=w, min_samples=3).over(inst, order_by=ts)
    mean_x = pl.col(x_col).rolling_mean(window_size=w, min_samples=3).over(inst, order_by=ts)
    xy = (pl.col(y_col) * pl.col(x_col)).rolling_mean(window_size=w, min_samples=3).over(inst, order_by=ts)
    x2 = (pl.col(x_col) ** 2).rolling_mean(window_size=w, min_samples=3).over(inst, order_by=ts)
    cov_xy = xy - mean_x * mean_y
    var_x = x2 - mean_x**2
    beta = cov_xy / pl.when(var_x.is_null() | (var_x == 0)).then(None).otherwise(var_x)
    alpha = mean_y - beta * mean_x
    n_valid = (
        pl.when(pl.col(y_col).is_not_null() & pl.col(x_col).is_not_null())
        .then(1.0)
        .otherwise(0.0)
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
    lag = max(_int_attr(node, "lag", default=1), 1)
    w = max(lag + 2, _window_int(node))
    mp = max(2, w // 3)
    return w, lag, mp


def _expanding_over() -> tuple[str, str]:
    return _INST, _TS


def _expanding_non_null_count() -> pl.Expr:
    inst, ts = _expanding_over()
    return (
        pl.col(_VAL)
        .is_not_null()
        .cast(pl.Float64)
        .rolling_sum(window_size=_EXPANDING_WINDOW, min_samples=1)
        .over(inst, order_by=ts)
    )


def _expanding_sum_expr() -> pl.Expr:
    inst, ts = _expanding_over()
    return pl.col(_VAL).rolling_sum(window_size=_EXPANDING_WINDOW, min_samples=1).over(inst, order_by=ts)


def _expanding_mean_expr() -> pl.Expr:
    cnt = _expanding_non_null_count()
    return pl.when(cnt <= 0).then(None).otherwise(_expanding_sum_expr() / cnt)


def _expanding_std_expr() -> pl.Expr:
    inst, ts = _expanding_over()
    cnt = _expanding_non_null_count()
    mean = _expanding_mean_expr()
    mean_sq = (
        (pl.col(_VAL) ** 2)
        .rolling_sum(window_size=_EXPANDING_WINDOW, min_samples=1)
        .over(inst, order_by=ts)
        / cnt
    )
    var_pop = mean_sq - mean**2
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


def plan_is_polars_expr_capable(plan: PlanNode) -> bool:
    op = _resolve(plan.op)
    if op not in POLARS_EXPR_CAPABLE:
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


def _truthy(col: str) -> pl.Expr:
    return pl.col(col).is_not_null() & (pl.col(col) != 0)


def _cs_rank_01() -> pl.Expr:
    r = pl.col(_VAL).rank(method="average").over(_TS, order_by=_INST)
    n = pl.col(_VAL).count().over(_TS, order_by=_INST)
    return (
        pl.when(n <= 1)
        .then(0.5)
        .when(pl.col(_VAL).is_null())
        .then(None)
        .otherwise((r - 1.0) / (n - 1.0))
    )


def _cs_rank_pct() -> pl.Expr:
    n = pl.col(_VAL).count().over(_TS, order_by=_INST)
    frac = pl.col(_VAL).rank(method="average").over(_TS, order_by=_INST) / pl.when(n > 0).then(n.cast(pl.Float64)).otherwise(None)
    return pl.when(pl.col(_VAL).is_null()).then(None).otherwise(frac)


def _compile_polars(node: PlanNode, base: pl.LazyFrame) -> pl.LazyFrame | None:
    if pl is None:
        return None
    op = _resolve(node.op)
    if op == "WMA":
        op = "ts_decay_linear"
    elif op == "rolling_beta":
        op = "ts_beta"

    if op == "column":
        name = str(node.attrs.get("name") or "")
        if not name:
            return None
        return base.select(pl.col(_TS), pl.col(_INST), pl.col(name).alias(_VAL))

    if op == "literal":
        val = node.attrs.get("value")
        return base.select(pl.col(_TS), pl.col(_INST), pl.lit(val).alias(_VAL))

    if op in {"add", "subtract", "multiply", "divide", "protected_div", "maximum", "minimum"}:
        if len(node.inputs) != 2:
            return None
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
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
            expr = (
                pl.when(pl.col("_y").is_null() | (pl.col("_y") == 0))
                .then(0.0)
                .otherwise(pl.col(_VAL) / pl.col("_y"))
            )
        else:
            expr = pl.col(_VAL) / pl.col("_y")
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "inverse":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null() | (pl.col(_VAL) == 0))
            .then(None)
            .otherwise(1.0 / pl.col(_VAL))
            .alias(_VAL)
        )

    if op == "is_finite":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(
            pl.col(_VAL).is_finite().cast(pl.Float64).alias(_VAL)
        )

    if op == "is_nan":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(
            pl.col(_VAL).is_nan().cast(pl.Float64).alias(_VAL)
        )

    if op == "neg":
        inner = _compile_polars(node.inputs[0], base)
        return inner.with_columns((-pl.col(_VAL)).alias(_VAL)) if inner is not None else None

    if op == "abs":
        inner = _compile_polars(node.inputs[0], base)
        return inner.with_columns(pl.col(_VAL).abs().alias(_VAL)) if inner is not None else None

    if op == "sign":
        inner = _compile_polars(node.inputs[0], base)
        return inner.with_columns(pl.col(_VAL).sign().alias(_VAL)) if inner is not None else None

    if op == "log":
        inner = _compile_polars(node.inputs[0], base)
        return inner.with_columns(pl.col(_VAL).log().alias(_VAL)) if inner is not None else None

    if op == "exp":
        inner = _compile_polars(node.inputs[0], base)
        return inner.with_columns(pl.col(_VAL).exp().alias(_VAL)) if inner is not None else None

    if op == "sqrt":
        inner = _compile_polars(node.inputs[0], base)
        return inner.with_columns(pl.col(_VAL).sqrt().alias(_VAL)) if inner is not None else None

    if op == "floor":
        inner = _compile_polars(node.inputs[0], base)
        return inner.with_columns(pl.col(_VAL).floor().alias(_VAL)) if inner is not None else None

    if op == "ceil":
        inner = _compile_polars(node.inputs[0], base)
        return inner.with_columns(pl.col(_VAL).ceil().alias(_VAL)) if inner is not None else None

    if op == "power":
        if len(node.inputs) != 2:
            return None
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(pl.col(_VAL).pow(pl.col("_y")).alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "protected_log":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        eps = _float_attr(node, "eps", default=1e-12)
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null() | (pl.col(_VAL) <= eps))
            .then(0.0)
            .otherwise(pl.col(_VAL).log())
            .alias(_VAL)
        )

    if op == "protected_sqrt":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(pl.max_horizontal(pl.col(_VAL), pl.lit(0.0)).sqrt().alias(_VAL))

    if op == "clip":
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)
        if op == "ts_mean":
            expr = pl.col(_VAL).rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        elif op == "ts_sum":
            expr = pl.col(_VAL).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        elif op == "ts_min":
            expr = pl.col(_VAL).rolling_min(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        elif op == "ts_max":
            expr = pl.col(_VAL).rolling_max(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        elif op == "ts_var":
            expr = pl.col(_VAL).rolling_var(window_size=w, min_samples=1, ddof=1).over(_INST, order_by=_TS)
        elif op == "ts_median":
            expr = pl.col(_VAL).rolling_median(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        else:
            expr = pl.col(_VAL).rolling_std(window_size=w, min_samples=1, ddof=1).over(_INST, order_by=_TS)
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_zscore":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)
        mean = pl.col(_VAL).rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        std = pl.col(_VAL).rolling_std(window_size=w, min_samples=1, ddof=1).over(_INST, order_by=_TS)
        return inner.with_columns(
            (
                (pl.col(_VAL) - mean)
                / pl.when(std == 0).then(1.0).otherwise(std)
            ).alias(_VAL)
        )

    if op == "ts_corr":
        if len(node.inputs) < 2:
            return None
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
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
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
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
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
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
        price = _compile_polars(node.inputs[0], base)
        vol = _compile_polars(node.inputs[1], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).cum_sum().over(_INST, order_by=_TS).alias(_VAL))

    if op == "cum_max":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).cum_max().over(_INST, order_by=_TS).alias(_VAL))

    if op == "cum_min":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).cum_min().over(_INST, order_by=_TS).alias(_VAL))

    if op == "cum_prod":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).cum_prod().over(_INST, order_by=_TS).alias(_VAL))

    if op == "ts_rank":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)

        def _rolling_rank_pct(s: pl.Series) -> float | None:
            if len(s) == 0:
                return None
            ranks = s.rank(method="average")
            return float(ranks[-1] / len(s))

        return inner.with_columns(
            pl.col(_VAL)
            .rolling_map(_rolling_rank_pct, window_size=w, min_samples=1)
            .over(_INST, order_by=_TS)
            .alias(_VAL)
        )

    if op in {"ewm_mean", "ewm_std", "ewm_var"}:
        inner = _compile_polars(node.inputs[0], base)
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
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(pl.coalesce(pl.col(_VAL), pl.col("_y")).alias(_VAL)).select(_TS, _INST, _VAL)

    if op in {"where", "if_else"}:
        if len(node.inputs) != 3:
            return None
        cond = _compile_polars(node.inputs[0], base)
        a = _compile_polars(node.inputs[1], base)
        b = _compile_polars(node.inputs[2], base)
        if cond is None or a is None or b is None:
            return None
        joined = cond.join(a.rename({_VAL: "_va"}), on=[_TS, _INST], how="inner").join(
            b.rename({_VAL: "_vb"}),
            on=[_TS, _INST],
            how="inner",
        )
        return joined.with_columns(
            pl.when(_truthy(_VAL)).then(pl.col("_va")).otherwise(pl.col("_vb")).alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op in {"gt", "lt", "eq", "ge", "le", "ne"}:
        if len(node.inputs) != 2:
            return None
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
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
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(
            pl.when(_truthy(_VAL) & _truthy("_y")).then(1.0).otherwise(0.0).alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "or_":
        if len(node.inputs) != 2:
            return None
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(
            pl.when(_truthy(_VAL) | _truthy("_y")).then(1.0).otherwise(0.0).alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "not_":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(
            pl.when(_truthy(_VAL)).then(0.0).otherwise(1.0).alias(_VAL)
        )

    if op in {"fillna_const", "fillna"}:
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        const = _literal_value(node, 0, default=0.0) or 0.0
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_nan()).then(const).otherwise(pl.col(_VAL)).alias(_VAL)
        )

    if op in {"cs_quantile", "c_percentile"}:
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        p = _float_attr(node, "p", default=0.5)
        pos_p = _literal_value(node, 0)
        if pos_p is not None:
            p = pos_p
        q = pl.col(_VAL).quantile(quantile=p, interpolation="linear").over(_TS, order_by=_INST)
        return inner.with_columns(q.alias(_VAL))

    if op == "winsorize":
        inner = _compile_polars(node.inputs[0], base)
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
        y_layer = _compile_polars(node.inputs[0], base)
        x_layer = _compile_polars(node.inputs[1], base)
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
        val = _compile_polars(node.inputs[0], base)
        if val is None:
            return None
        if len(node.inputs) >= 2 and node.inputs[1].op not in {"literal"}:
            grp = _compile_polars(node.inputs[1], base)
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
            expr = pl.col(_VAL).std(ddof=1).over(*over_keys, order_by=_INST)
        elif op in {"group_zscore", "group_neutralize"}:
            mean = pl.col(_VAL).mean().over(*over_keys, order_by=_INST)
            if op == "group_neutralize":
                expr = pl.col(_VAL) - mean
            else:
                std = pl.col(_VAL).std(ddof=1).over(*over_keys, order_by=_INST)
                expr = pl.when(std.is_null() | (std == 0)).then(0.0).otherwise((pl.col(_VAL) - mean) / std)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        med = pl.col(_VAL).median().over(_TS, order_by=_INST)
        mad = (pl.col(_VAL) - med).abs().median().over(_TS, order_by=_INST)
        return inner.with_columns(mad.alias(_VAL))

    if op == "cs_mad_zscore":
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return inner.with_columns(pl.col(_VAL).shift(lag).over(_INST, order_by=_TS).alias(_VAL))

    if op == "ts_delta":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        delayed = pl.col(_VAL).shift(lag).over(_INST, order_by=_TS)
        return inner.with_columns((pl.col(_VAL) - delayed).alias(_VAL))

    if op == "ts_pct":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        prev = pl.col(_VAL).shift(lag).over(_INST, order_by=_TS)
        return inner.with_columns((pl.col(_VAL) / prev - 1.0).alias(_VAL))

    if op == "ffill":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).forward_fill().over(_INST, order_by=_TS).alias(_VAL))

    if op == "bfill":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner

    if op in {"ts_ema", "ewm_mean"}:
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        alpha = _ewm_alpha(node)
        return inner.with_columns(
            pl.col(_VAL).ewm_mean(alpha=alpha, adjust=False).over(_INST, order_by=_TS).alias(_VAL)
        )

    if op == "rank":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(_cs_rank_01().alias(_VAL))

    if op in {"rank_pct", "cs_pct_rank"}:
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(_cs_rank_pct().alias(_VAL))

    if op == "zscore":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        mean = pl.col(_VAL).mean().over(_TS, order_by=_INST)
        std = pl.col(_VAL).std(ddof=1).over(_TS, order_by=_INST)
        return inner.with_columns(
            pl.when(std.is_null() | (std == 0))
            .then(0.0)
            .otherwise((pl.col(_VAL) - mean) / std)
            .alias(_VAL)
        )

    if op == "cs_demean":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        mean = pl.col(_VAL).mean().over(_TS, order_by=_INST)
        return inner.with_columns((pl.col(_VAL) - mean).alias(_VAL))

    if op == "normalize":
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        prev = pl.col(_VAL).shift(1).over(_INST, order_by=_TS)
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null() | prev.is_null() | (prev == 0))
            .then(None)
            .otherwise((pl.col(_VAL) / prev).log())
            .alias(_VAL)
        )

    if op == "volatility":
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).abs().log().alias(_VAL))

    if op == "signed_log":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(
            (pl.col(_VAL).sign() * (pl.col(_VAL).abs() + 1e-10).log()).alias(_VAL)
        )

    if op == "signed_sqrt":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(
            (pl.col(_VAL).sign() * pl.col(_VAL).abs().sqrt()).alias(_VAL)
        )

    if op == "ts_decay_linear":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_linear_decay_expr(w).alias(_VAL))

    if op == "ts_mad":
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)
        p = _float_attr(node, "q", "p", default=0.5)
        pos_p = _literal_value(node, 1)
        if pos_p is not None:
            p = pos_p
        return inner.with_columns(_rolling_quantile_expr(w, p).alias(_VAL))

    if op == "ts_product":
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_skew_expr(w).alias(_VAL))

    if op == "ts_argmax":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_argext_expr(w, pick="max").alias(_VAL))

    if op == "ts_argmin":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_argext_expr(w, pick="min").alias(_VAL))

    if op == "Slope":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_time_slope_expr(w).alias(_VAL))

    if op == "ts_regression":
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_polars(node.inputs[0], base)
        x_layer = _compile_polars(node.inputs[1], base)
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
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(_expanding_sum_expr().alias(_VAL))

    if op == "expanding_mean":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(_expanding_mean_expr().alias(_VAL))

    if op == "expanding_std":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(_expanding_std_expr().alias(_VAL))

    if op == "count":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        return inner.with_columns(_expanding_non_null_count().alias(_VAL))

    if op in {"ewm_corr", "ewm_cov"}:
        if len(node.inputs) < 2:
            return None
        left = _compile_polars(node.inputs[0], base)
        right = _compile_polars(node.inputs[1], base)
        if left is None or right is None:
            return None
        span = max(_window_int(node, default=20), 2)
        joined = _join_binary(left, right)
        return _ewm_binary_map_groups(joined, span, corr=(op == "ewm_corr"))

    if op == "ts_ratio":
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = max(_int_attr(node, "d", "window", input_index=0, default=3), 1)
        k = max(_int_attr(node, "k", "order", input_index=1, default=2), 1)
        from cleaned_operators._numpy_kernels import ts_moment_

        return _unary_inst_map_groups(inner, lambda arr: ts_moment_(arr, w, k))

    if op == "ts_max_buildup":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = _window_int(node)
        from cleaned_operators._numpy_kernels import ts_max_buildup_

        return _unary_inst_map_groups(inner, lambda arr: ts_max_buildup_(arr, w))

    if op == "expanding_rank":
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None

        def _expanding_rank(arr: np.ndarray) -> np.ndarray:
            ranked = pd.Series(arr, dtype=float).expanding(min_periods=1).rank(pct=True)
            return ranked.to_numpy(dtype=float)

        return _unary_inst_map_groups(inner, _expanding_rank)

    if op == "fillna_interpolate":
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
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
        inner = _compile_polars(node.inputs[0], base)
        if inner is None:
            return None
        w = max(_window_int(node, default=14), 2)
        from cleaned_operators.technical.signal import _compute_rsi_wilder

        def _rsi(arr: np.ndarray) -> np.ndarray:
            pdf = pd.DataFrame({"v": arr.astype(float, copy=False)})
            return _compute_rsi_wilder(pdf, w)["v"].to_numpy(dtype=float)

        return _unary_inst_map_groups(inner, _rsi)

    if op == "ATR_WILDER":
        if len(node.inputs) < 3:
            return None
        high = _compile_polars(node.inputs[0], base)
        low = _compile_polars(node.inputs[1], base)
        close = _compile_polars(node.inputs[2], base)
        if high is None or low is None or close is None:
            return None
        w = max(_window_int(node, default=14), 2)
        joined = _join_triple(high, low, close)
        from cleaned_operators.technical.signal import _compute_atr_wilder
        schema = joined.collect_schema()

        def _apply(g: pl.DataFrame) -> pl.DataFrame:
            h = pd.DataFrame({"v": g[_VAL].to_numpy().astype(float, copy=False)})
            lo = pd.DataFrame({"v": g["_ym"].to_numpy().astype(float, copy=False)})
            cl = pd.DataFrame({"v": g["_y"].to_numpy().astype(float, copy=False)})
            out = _compute_atr_wilder(h, lo, cl, w)["v"].to_numpy(dtype=float)
            return g.select(
                pl.col(_TS),
                pl.col(_INST),
                pl.Series(_VAL, out),
            )

        return joined.group_by(_INST, maintain_order=True).map_groups(
            _apply,
            schema={_TS: schema[_TS], _INST: schema[_INST], _VAL: pl.Float64},
        )

    return None


def _build_base_lazy(ctx: Any, columns: set[str]) -> pl.LazyFrame:
    """从 DataSource 构建 LazyFrame；优先 ``scan_polars_long``。"""
    scan = getattr(ctx.data_source, "scan_polars_long", None)
    if callable(scan):
        return scan(sorted(columns))
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
    ts_col: str = _TS,
    inst_col: str = _INST,
) -> LongFrameResult | None:
    """编译 PlanNode 为 long-table Polars LazyFrame + 结果列名。"""
    if pl is None:
        return None
    compiled = _compile_polars(plan, base_lf)
    if compiled is None:
        return None
    return LongFrameResult(
        frame=compiled,
        value_col=_VAL,
        ts_col=ts_col,
        inst_col=inst_col,
        is_lazy=True,
    )


def execute_polars_long_plan(
    plan: PlanNode,
    ctx: Any,
    *,
    base_lf: pl.LazyFrame | None = None,
) -> pd.Series:
    """编译并执行 long-table 计划；仅最终 collect 一次。"""
    cols = collect_columns(plan)
    base = base_lf if base_lf is not None else _build_base_lazy(ctx, cols)
    compiled = compile_plan_to_polars(plan, base)
    if compiled is None:
        raise RuntimeError(f"polars long compile failed for op={plan.op!r}")
    frame = (
        compiled.frame.sort([compiled.ts_col, compiled.inst_col])
        .select(
            pl.col(compiled.ts_col).alias(ctx.timestamp_col),
            pl.col(compiled.inst_col).alias(ctx.instrument_col),
            pl.col(compiled.value_col).alias("value"),
        )
        .collect()
    )
    template = ctx.data_source.load_column(next(iter(cols)))
    return polars_long_to_multiindex_series(
        frame,
        timestamp_col=ctx.timestamp_col,
        instrument_col=ctx.instrument_col,
        value_col="value",
        template_index=template.index,
    )


def execute_polars_expr_plan(plan: PlanNode, ctx: Any) -> pd.Series:
    """兼容旧名：``PolarsBackend`` + ``FACTOR_ENGINE_POLARS_EXPR=1``。"""
    return execute_polars_long_plan(plan, ctx)
