# -*- coding: utf-8
"""PlanNode → Polars Expr 编译器（long-table native，对齐 SQL emitter 的 ts/inst/value 语义）。

本模块是 factor engine **Polars long-table 路径**的核心编译层：将逻辑计划 ``PlanNode`` 递归
lowering 为 ``(ts, inst, _v)`` 形态的 ``pl.LazyFrame`` / ``pl.Expr``，避免经 ``cleaned_bridge``
或 panel 宽表往返，从而与 ``sql_pushdown/emitter`` 保持时序/截面/数值语义一致。

主要职责
--------
* **编译**：``compile_plan_to_polars`` / ``compile_polars_long_lazy`` 在宽表 base LazyFrame 上
  生成 long-table 结果列。
* **执行**：``execute_polars_long_plan`` 一次 collect 并还原为 MultiIndex ``pd.Series``。
* **能力探测**：``plan_is_polars_long_capable`` / ``plan_is_polars_expr_capable`` 判断计划
  是否可走 native / map_groups / registry bridge 路径。
* **DAG 融合**：同列 ``ts_*`` 子树、二元 base 列引用等优化，减少中间 join。
* **桥接**：未手写 native 的算子 fallback 至 ``polars_registry_bridge``。

调用入口
--------
由 ``PolarsLongBackend`` 直接调用；兼容 ``FACTOR_ENGINE_POLARS_EXPR=1`` 时经
``execute_polars_expr_plan`` 走同一执行链。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from factor_engine.planner.logical_plan import PlanNode

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

_PREDICATE_OPS: frozenset[str] = frozenset(
    {"is_nan", "is_null", "is_not_null", "is_finite", "is_infinite"}
)
_NAN_PRESERVE_PARENT_OPS: frozenset[str] = frozenset({"maximum", "minimum"})

# Operators whose PANDAS reference kernel is a pandas rolling / expanding / ewm
# AGGREGATION: pandas' rolling machinery treats ±Inf as MISSING (a window
# containing Inf averages the remaining finite values — it never propagates
# Inf).  The polars path must drop Inf the same way BEFORE the window, or a
# single Inf row corrupts the whole window on polars while pandas stays finite.
# Elementwise / shift operators (ts_delta, ts_pct, returns, log_returns, ...)
# PROPAGATE Inf in the pandas reference and are deliberately NOT listed.
_PANDAS_ROLLING_DROP_INF_OPS: frozenset[str] = frozenset(
    {
        "ts_mean",
        "ts_median",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_std",
        "ts_var",
        "ts_skew",
        "ts_kurt",
        "ts_quantile",
        "ts_zscore",
        "ts_sharpe",
        "ts_autocorr",
        "ts_corr",
        "ts_cov",
        "ts_beta",
        "ts_ema",
        "ema",
        "ts_ewm_mean",
        "MACD_line",
        "MACD_signal",
        "MACD_hist",
        "ts_ewm_std",
        "ts_rank",
        "ts_pct_rank",
        "ts_rank_mean",
        "ts_rank_std",
        "ts_returns",
        "ts_pct_chg",
        "ts_decay_linear",
        "SMA",
        "true_range",
        "atr",
        "avg_true_range_pct",
        "expanding_mean",
        "expanding_std",
        "expanding_sum",
        "expanding_min",
        "expanding_max",
        "cumsum",
        "cum_delta",
        "cum_std",
        "ts_max_drawdown",
        "ts_drawdown",
        "ts_linear_reg_slope",
        "ts_linear_reg_residual",
        "ts_time_slope",
        "ts_range_expansion",
        "volume_ratio",
        "rolling_obv",
        "rolling_pvt",
    }
)


def _sanitize_nan_for_compute(
    inner: pl.LazyFrame | None, *, drop_inf: bool = False
) -> pl.LazyFrame | None:
    """Pandas 数值路径：IEEE NaN 按缺失处理（rolling / coalesce 等）。

    ``drop_inf`` — the pandas rolling/expanding/ewm reference treats ±Inf as
    missing too; drop it to NULL so a windowed aggregation on polars agrees
    with the pandas kernel (which silently excludes Inf rows).
    """
    if inner is None:
        return None
    if drop_inf:
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite())
            .then(None)
            .otherwise(pl.col(_VAL))
            .alias(_VAL)
        )
    return inner.with_columns(
        pl.when(pl.col(_VAL).is_nan()).then(None).otherwise(pl.col(_VAL)).alias(_VAL)
    )


def _compile_child(
    node: PlanNode,
    index: int,
    base: pl.LazyFrame,
    *,
    parent_op: str,
    ctx: Any | None = None,
    memo: dict[tuple[Any, ...], pl.LazyFrame] | None = None,
) -> pl.LazyFrame | None:
    """编译子节点；非 predicate 父算子在子结果上将 NaN 规范为 NULL。

    For rolling/expanding/ewm aggregation parents, ±Inf is dropped to NULL too
    (the pandas reference treats Inf as missing inside rolling machinery), so a
    single Inf input cannot corrupt the polars window while pandas stays finite.
    """
    if index >= len(node.inputs):
        return None
    inner = _compile_polars(node.inputs[index], base, ctx=ctx, memo=memo)
    if inner is None or parent_op in _PREDICATE_OPS or parent_op in _NAN_PRESERVE_PARENT_OPS:
        return inner
    return _sanitize_nan_for_compute(
        inner, drop_inf=parent_op in _PANDAS_ROLLING_DROP_INF_OPS
    )

_GRP = "_grp"


@dataclass(frozen=True)
class CompiledLongExpr:
    """Expr DAG 节点：单条 ``pl.Expr`` + 依赖列（避免中间 LazyFrame join）。

    属性:
        expr: 最终值列表达式。
        required_cols: 编译所需 base 列名集合。
        tmp_exprs: 可选中间 with_columns 表达式元组。
    """

    expr: Any
    required_cols: frozenset[str]
    tmp_exprs: tuple[Any, ...] = ()

# expanding / cum 类算子：rolling 窗口需覆盖单 inst 全长（与 SQL UNBOUNDED PRECEDING 对齐）
_EXPANDING_WINDOW = 100_000

# 能力分层见 ``polars_long_policy``（POLARS_LONG_NATIVE / MAP_GROUPS / COMPATIBLE）


def _resolve(op: str) -> str:
    """解析算子别名至 canonical 名称。"""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(op, op)


def _window_int(node: PlanNode, default: int = 3) -> int:
    """从 PlanNode attrs 或 literal 子节点解析滚动窗口（与 DuckDB 共用）。"""
    return _window_spec(node, default=default).size


def _window_spec(node: PlanNode, default: int = 3):
    """完整 WindowSpec（min_periods / ddof / closed）。"""
    from factor_engine.backend.plan_params import window_spec_from_plan_node
    from factor_engine.backend.window_spec import WindowSpec

    spec = window_spec_from_plan_node(node, default=default)
    if spec.closed != "right":
        from factor_engine.backend.plan_params import PlanParamError

        raise PlanParamError(f"closed={spec.closed!r} 暂未支持，仅 right")
    return spec


def _ewm_alpha(node: PlanNode, default_span: int = 20) -> float:
    from factor_engine.backend.ewm_spec import EwmSpec

    return EwmSpec.from_plan_node(node, default_span=default_span).alpha


def _wilder_alpha(node: PlanNode, default: int = 14) -> tuple[float, int]:
    """解析 Wilder 平滑参数，返回 (alpha=1/w, window)。"""
    w = max(_window_int(node, default=default), 2)
    return 1.0 / float(w), w


def _rsi_wilder_expr(value_col: str, *, window: int, alpha: float) -> pl.Expr:
    """构造 Wilder RSI 的 Polars 表达式（按 inst 时序）。"""
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
    """构造 Wilder ATR 的 Polars 表达式（TR 的 EWM 均值）。"""
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


def _scale_to_value(node: PlanNode, default: float = 1.0) -> float:
    """解析 scale(to=...) 目标缩放值。"""
    if "to" in (node.attrs or {}) and node.attrs["to"] is not None:
        return float(node.attrs["to"])
    for idx in range(1, len(node.inputs)):
        child = node.inputs[idx]
        if child.op == "literal":
            val = child.attrs.get("value")
            if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
    return default


def _float_attr(node: PlanNode, *keys: str, default: float) -> float:
    """按 keys 顺序从 node.attrs 读取首个有限 float 属性。"""
    from factor_engine.backend.plan_params import PlanParamError, parse_finite_float, parse_unit_interval

    for key in keys:
        if key in (node.attrs or {}) and node.attrs[key] is not None:
            raw = node.attrs[key]
            if key in {"p", "q", "lo", "hi", "min_pct", "max_pct", "lower", "upper"}:
                try:
                    return parse_unit_interval(raw, label=key)
                except PlanParamError:
                    return default
            try:
                need_pos = key in {"epsilon", "eps", "to", "ann_factor"}
                return parse_finite_float(raw, label=key, gt=0.0 if need_pos else None)
            except PlanParamError:
                return default
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
    """读取 node.inputs[index+1] 处 literal 子节点的数值，无法解析则返回 default。"""
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
    """从 attrs 或指定 literal 输入解析 int 属性。"""
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
    beta = pl.when(var_x.is_null() | (var_x == 0)).then(None).otherwise(cov_xy / var_x)
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
    """解析 ts_regression 返回值模式：slope / intercept / fit / resid。"""
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
    """构造线性衰减加权滚动均值 expr（权重 1..w，对齐 ts_decay_linear）。"""
    weights = np.arange(1, w + 1, dtype=np.float64)

    def _fn(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(arr)
        if not valid.any():
            return np.nan
        # Age-slot anchored (R16-070, mirrors ``_linear_weighted_1d_numpy``):
        # slice weights by the FULL window slot length (NaN slots included), so
        # each value keeps its original weight position; only then mask by the
        # valid positions.  Slicing by the valid COUNT re-anchors weights onto
        # the wrong slots and diverges from the pandas reference whenever the
        # window contains a NaN/Inf hole.
        ww = weights[-len(arr) :]
        seg = arr[valid]
        w = ww[valid]
        return float(np.dot(seg, w) / w.sum())

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_product_expr(w: int, *, min_periods: int) -> pl.Expr:
    """滚动乘积：正确处理 0、负数符号与 NULL 跳过。"""

    def _fn(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if np.isinf(arr).any():
            return np.nan
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.nan
        if np.any(valid == 0.0):
            if np.isinf(arr).any():
                return np.nan
            return 0.0
        if np.isinf(arr).any():
            return np.nan
        neg_cnt = int(np.sum(valid) < 0)
        sign = -1.0 if neg_cnt % 2 else 1.0
        log_abs = float(np.log(np.abs(valid)).sum())
        if not np.isfinite(log_abs) or log_abs > np.log(np.finfo(np.float64).max):
            return np.nan
        return sign * float(np.exp(log_abs))

    return (
        pl.col(_VAL)
        .rolling_map(_fn, window_size=w, min_samples=min_periods)
        .over(_INST, order_by=_TS)
    )


def _rolling_median_abs_dev_expr(
    w: int,
    *,
    min_periods: int = 1,
    scale: float = 1.0,
) -> pl.Expr:
    """Median Absolute Deviation：median(|x_i - median(window)|)。"""

    def _fn(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.nan
        med = float(np.median(valid))
        return float(scale) * float(np.median(np.abs(valid - med)))

    return (
        pl.col(_VAL)
        .rolling_map(_fn, window_size=w, min_samples=min_periods)
        .over(_INST, order_by=_TS)
    )


def _rolling_mean_abs_dev_expr(w: int) -> pl.Expr:
    """Mean Absolute Deviation：mean(|x_i - mean(window)|)。"""

    def _fn(arr: np.ndarray) -> float:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.nan
        mu = float(valid.mean())
        return float(np.mean(np.abs(valid - mu)))

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_time_slope_expr(w: int, *, min_periods: int = 2) -> pl.Expr:
    """构造窗口内时间序列线性回归斜率 expr（x 为等距时间索引）。

    位置以**实际窗口**为基准（0..len-1，缺失值剔除后重中心化），与 pandas
    参考 ``pd_time_slope`` 一致：部分窗口同样从 2 个有效点起算。旧的实现把
    部分窗口按完整窗口的尾部权重对齐，导致窗口起始处与 pandas 不一致。
    """

    def _dot(arr) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(arr)
        n = int(valid.sum())
        if n < min_periods:
            return np.nan
        t = np.arange(arr.size, dtype=np.float64)[valid]
        y = arr[valid]
        t = t - t.mean()
        denom = float(t @ t)
        if denom <= 0.0:
            return np.nan
        return float(t @ (y - y.mean()) / denom)

    return pl.col(_VAL).rolling_map(_dot, window_size=w, min_samples=min_periods).over(_INST, order_by=_TS)


def _rolling_argext_expr(w: int, *, pick: str) -> pl.Expr:
    """滚动极值距当前 bar 的距离（0=当前；并列取最近）。"""
    from factor_engine.backend.numeric_semantics import ts_argmax_empty_window_is_null

    def _fn(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).any():
            return np.nan if ts_argmax_empty_window_is_null() else 0.0
        extreme = np.nanmax(arr) if pick == "max" else np.nanmin(arr)
        positions = np.flatnonzero(np.isfinite(arr) & (arr == extreme))
        return float(arr.size - 1 - positions[-1])

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_skew_expr(w: int) -> pl.Expr:
    """构造滚动偏度 expr（pandas skew，至少 3 个有效样本）。"""
    def _fn(arr: np.ndarray) -> float:
        s = pd.Series(np.asarray(arr, dtype=np.float64))
        if s.count() < 3:
            return np.nan
        val = s.skew()
        return np.nan if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_quantile_expr(w: int, p: float) -> pl.Expr:
    """构造滚动分位数 expr（pandas quantile，分位 p）。"""
    def _fn(arr: np.ndarray) -> float:
        s = pd.Series(np.asarray(arr, dtype=np.float64))
        if s.count() == 0:
            return np.nan
        return float(s.quantile(p))

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _ts_sharpe_min_periods(w: int) -> int:
    """ts_sharpe 最小有效样本数：max(2, w//3)。"""
    return max(2, w // 3)


def _ts_autocorr_window_lag(node: PlanNode) -> tuple[int, int, int]:
    """解析 ts_autocorr 的 (window, lag, min_periods) 三元组。"""
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
    """expanding/cum 算子的 over 分区键：(inst, ts)。"""
    return _INST, _TS


def _cum_current_row_null_guard(expr: pl.Expr) -> pl.Expr:
    """当前行 NULL → 输出 NULL（对齐 pandas cum/expanding 缺失行语义）。"""
    return pl.when(pl.col(_VAL).is_null()).then(None).otherwise(expr)


def _pandas_aligned_cum_sum_expr() -> pl.Expr:
    inst, ts = _expanding_over()
    c = pl.col(_VAL)
    run = c.fill_null(0).cum_sum().over(inst, order_by=ts)
    return _cum_current_row_null_guard(run)


def _pandas_aligned_cum_max_expr() -> pl.Expr:
    inst, ts = _expanding_over()
    return _cum_current_row_null_guard(pl.col(_VAL).cum_max().over(inst, order_by=ts))


def _pandas_aligned_cum_min_expr() -> pl.Expr:
    inst, ts = _expanding_over()
    return _cum_current_row_null_guard(pl.col(_VAL).cum_min().over(inst, order_by=ts))


def _pandas_aligned_cum_prod_expr() -> pl.Expr:
    inst, ts = _expanding_over()
    return _cum_current_row_null_guard(pl.col(_VAL).cum_prod().over(inst, order_by=ts))


def _valid_obs_expr(col: str = _VAL) -> pl.Expr:
    """Pandas ``notna`` 对齐：非 NULL 且非 IEEE NaN。"""
    from factor_engine.backend.logical_semantics import is_null_polars_expr

    return pl.when(is_null_polars_expr(col) > 0).then(0.0).otherwise(1.0)


def _expanding_non_null_count() -> pl.Expr:
    """expanding 非空计数 expr（cum_sum of is_not_null）。"""
    inst, ts = _expanding_over()
    return _valid_obs_expr().cum_sum().over(inst, order_by=ts)


def _expanding_sum_expr() -> pl.Expr:
    """expanding 累加和 expr（按 inst 时序 cum_sum，NULL 行输出 NULL）。"""
    return _pandas_aligned_cum_sum_expr()


def _expanding_mean_expr() -> pl.Expr:
    """expanding 均值 expr（sum / 非空计数，NULL 行输出 NULL）。"""
    inst, ts = _expanding_over()
    c = pl.col(_VAL)
    cnt = _valid_obs_expr().cum_sum().over(inst, order_by=ts)
    run = c.fill_null(0).cum_sum().over(inst, order_by=ts)
    return (
        pl.when(c.is_null())
        .then(None)
        .when(cnt <= 0)
        .then(None)
        .otherwise(run / cnt)
    )


def _welford_expanding_std_numpy(arr: np.ndarray) -> np.ndarray:
    """按 inst 时序 Welford expanding 样本标准差（skip NULL/NaN）。"""
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    n = 0
    mean = 0.0
    m2 = 0.0
    for i, raw in enumerate(arr):
        if raw is None:
            continue
        xv = float(raw)
        if not np.isfinite(xv):
            continue
        n += 1
        delta = xv - mean
        mean += delta / n
        m2 += delta * (xv - mean)
        if n >= 2:
            out[i] = float(np.sqrt(m2 / (n - 1)))
    return out


def _expanding_std_map_groups(inner: pl.LazyFrame) -> pl.LazyFrame:
    """expanding_std：Welford map_groups（避免 E[x²]-E[x]² 消减误差）。"""
    return _unary_inst_map_groups(inner, _welford_expanding_std_numpy)


def _expanding_std_expr() -> pl.Expr:
    """Deprecated：不稳定 E[x²]-E[x]²；保留供对照。"""
    inst, ts = _expanding_over()
    cnt = _expanding_non_null_count()
    mean = _expanding_mean_expr()
    sum_sq = (pl.col(_VAL) ** 2).cum_sum().over(inst, order_by=ts)
    var_pop = sum_sq / cnt - mean**2
    var_sample = pl.when(cnt <= 1).then(None).otherwise(
        pl.max_horizontal(var_pop * cnt / (cnt - 1.0), pl.lit(0.0))
    )
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


def _rolling_corr_centered_expr(
    x: pl.Expr,
    y: pl.Expr,
    window: int,
    min_periods: int,
) -> pl.Expr:
    """Native centered rolling correlation (pairwise-finite, direct window).

    Route (a) of the R21-P1 static-audit fix: this replaces the former
    ``_rolling_corr_centered_map_groups`` Python-group callback.  Semantics are
    identical (centered two-pass windows to avoid catastrophic cancellation for
    large-offset series; pairwise finite mask applied per window, so a missing
    current row does not suppress a result when the preceding valid pairs
    satisfy min_periods).  Implementation:

    - ``pair_l``/``pair_r`` mask non-finite x/y to NULL *per pair*, matching
      the pandas pairwise-finite contract (±Inf/NaN treated as missing).
    - each trailing window of width ``w`` is materialized horizontally via
      ``shift(k)`` (k = 0..w-1) with an implicit ``over(_INST, order_by=_TS)``
      so instrument partitions never mix;
    - the window anchor (first finite pair in the window) removes the large
      common offset before any squaring, then a second re-centering around the
      window mean keeps the low-order covariance signal (same trick as
      ``cleaned_operators.common.polars_ts_rolling._pairwise_rolling_moments``,
      the authoritative TSCorrNative oracle implementation).
    """
    w = max(int(window), 1)
    mp = max(int(min_periods), 1)
    finite_pair = x.is_finite().fill_null(False) & y.is_finite().fill_null(False)
    pair_l = pl.when(finite_pair).then(x).otherwise(None)
    pair_r = pl.when(finite_pair).then(y).otherwise(None)
    # Trailing window rows: shift(0) = current row, shift(w-1) = window head.
    # (``.over`` is applied to the final expression below; shifts inside an
    # ``over`` window expression are evaluated within each partition.)
    xs = [pair_l.shift(k) for k in range(w)]
    ys = [pair_r.shift(k) for k in range(w)]
    count = pl.sum_horizontal([v.is_not_null().cast(pl.Float64) for v in xs])
    x_anchor = pl.coalesce(xs)
    y_anchor = pl.coalesce(ys)
    xc = [v - x_anchor for v in xs]
    yc = [v - y_anchor for v in ys]
    sum_x = pl.sum_horizontal(xc)
    sum_y = pl.sum_horizontal(yc)
    mean_x = x_anchor + sum_x / count
    mean_y = y_anchor + sum_y / count
    ccx = [v - mean_x for v in xs]
    ccy = [v - mean_y for v in ys]
    ss_x = pl.sum_horizontal([v * v for v in ccx])
    ss_y = pl.sum_horizontal([v * v for v in ccy])
    cross = pl.sum_horizontal([a * b for a, b in zip(ccx, ccy)])
    denom = (ss_x * ss_y).sqrt()
    ready = (count >= mp) & count.is_not_null() & ss_x.is_not_null() & ss_y.is_not_null()
    value = cross / denom
    return pl.when(~ready | (ss_x <= 0) | (ss_y <= 0) | denom.is_null() | (denom <= 0)).then(
        None
    ).otherwise(value)


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
    """递归收集 Plan 子树引用的原始列名。

    参数:
        node: 逻辑计划根或子节点。
        out: 可选累加器；为 None 时新建 set。

    返回:
        所有 ``column`` 节点 ``name`` 属性的集合。
    """
    acc = out if out is not None else set()
    if node.op == "column":
        name = node.attrs.get("name")
        if name:
            acc.add(str(name))
    for child in node.inputs:
        collect_columns(child, acc)
    return acc


def plan_is_polars_long_capable(plan: PlanNode) -> bool:
    """判断计划是否可由 Polars long-table 后端完整执行。

    参数:
        plan: 待检测的逻辑计划根节点。

    返回:
        若 plan 及所有子节点算子均在 ``polars_long_policy`` 能力集内则为 True。
    """
    capable = get_polars_long_capable()
    op = _resolve(plan.op)
    if op in {"column", "literal", "materialized_series", "plan_ref"}:
        return all(plan_is_polars_long_capable(c) for c in plan.inputs)
    if op not in capable:
        return False
    return all(plan_is_polars_long_capable(c) for c in plan.inputs)


def plan_is_polars_expr_capable(plan: PlanNode) -> bool:
    """判断计划是否可走手写 native + map_groups 路径（不含 registry bridge）。

    参数:
        plan: 待检测的逻辑计划根节点。

    返回:
        若 plan 及子节点算子均在 ``POLARS_LONG_COMPATIBLE`` 内则为 True。
    """
    op = _resolve(plan.op)
    if op in {"column", "literal", "materialized_series", "plan_ref"}:
        return all(plan_is_polars_expr_capable(c) for c in plan.inputs)
    if op not in POLARS_LONG_COMPATIBLE:
        return False
    return all(plan_is_polars_expr_capable(c) for c in plan.inputs)


def _join_binary(left: pl.LazyFrame, right: pl.LazyFrame) -> pl.LazyFrame:
    """按 (ts, inst) anchor LEFT JOIN（左操作数保留全部 key）。"""
    from factor_engine.backend.long_alignment import anchor_left_join_binary

    return anchor_left_join_binary(left, right, right_col="_y")


def _join_triple(
    left: pl.LazyFrame,
    mid: pl.LazyFrame,
    right: pl.LazyFrame,
) -> pl.LazyFrame:
    """按 (ts, inst) anchor 串联 LEFT JOIN。"""
    from factor_engine.backend.long_alignment import anchor_left_join_triple

    return anchor_left_join_triple(left, mid, right)


def _join_quad(
    a: pl.LazyFrame,
    b: pl.LazyFrame,
    c: pl.LazyFrame,
    d: pl.LazyFrame,
) -> pl.LazyFrame:
    """按 (ts, inst) anchor 串联 4 路 LEFT JOIN（OHLC 4 输入）。"""
    from factor_engine.backend.long_alignment import anchor_left_join_triple

    a2 = a.join(b.rename({_VAL: "_b"}), on=[_TS, _INST], how="left")
    a3 = a2.join(c.rename({_VAL: "_c"}), on=[_TS, _INST], how="left")
    a4 = a3.join(d.rename({_VAL: "_d"}), on=[_TS, _INST], how="left")
    return a4


def _compile_ohlc3(node, base, parent_op, *, ctx=None, memo=None) -> pl.LazyFrame | None:
    """编译 3 个 OHLC 子输入 (high, low, close) 并 join，产 _VAL/_ym/_y。"""
    h = _compile_child(node, 0, base, parent_op=parent_op, ctx=ctx, memo=memo)
    l = _compile_child(node, 1, base, parent_op=parent_op, ctx=ctx, memo=memo)
    c = _compile_child(node, 2, base, parent_op=parent_op, ctx=ctx, memo=memo)
    if h is None or l is None or c is None:
        return None
    return _join_triple(h, l, c)


def _compile_ohlc4(node, base, parent_op, *, ctx=None, memo=None) -> pl.LazyFrame | None:
    """编译 4 个 OHLC 子输入 (open, high, low, close) 并 join，产 _VAL/_b/_c/_d。"""
    o = _compile_child(node, 0, base, parent_op=parent_op, ctx=ctx, memo=memo)
    h = _compile_child(node, 1, base, parent_op=parent_op, ctx=ctx, memo=memo)
    l = _compile_child(node, 2, base, parent_op=parent_op, ctx=ctx, memo=memo)
    c = _compile_child(node, 3, base, parent_op=parent_op, ctx=ctx, memo=memo)
    if o is None or h is None or l is None or c is None:
        return None
    return _join_quad(o, h, l, c)


def _join_multi(layers: list[pl.LazyFrame]) -> pl.LazyFrame:
    """Join N long layers into one frame with columns ``_v, _y, _z, _w, _u, _t, _s``.

    ``_join_binary`` renames the right operand to ``_y`` every time, which
    collides once a third input is joined.  This helper gives each input a
    distinct column name so the fin_* elementwise family can reference up to
    seven operands.
    """
    from factor_engine.backend.long_alignment import assert_exact_key_set

    names = [_VAL, "_y", "_z", "_w", "_u", "_t", "_s"]
    joined = layers[0]
    for i in range(1, len(layers)):
        assert_exact_key_set(joined, layers[i], context="fin elementwise join")
        joined = joined.join(
            layers[i].rename({_VAL: names[i]}),
            on=[_TS, _INST],
            how="left",
        )
    return joined


# ---------------------------------------------------------------------------
# fin_* elementwise algebraic family (pure Expr, no period walk).
#
# These mirror the pandas reference exactly: a ratio is NULL when the
# denominator is 0 / NULL, and an infinite or NaN quotient is NULL too.  The
# ``period_id`` trailing input (structural PIT-alignment only, never used in the
# computation) is intentionally NOT compiled — only the formula operands are.
# ---------------------------------------------------------------------------

def _fin_ratio_expr(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    q = a / b
    return pl.when(b.is_null() | (b == 0)).then(None).otherwise(
        pl.when(q.is_infinite() | q.is_nan()).then(None).otherwise(q)
    )


def _fin_abs_ratio_expr(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return _fin_ratio_expr(a, b.abs())


def _fin_neg_ocf_ratio_expr(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    neg = pl.when(b < 0).then(b.abs()).otherwise(None)
    return _fin_ratio_expr(a, neg)


def _fin_mask_expr(a: pl.Expr, thr: float) -> pl.Expr:
    return pl.when(a.is_null() | a.is_nan() | a.is_infinite()).then(None).otherwise(
        pl.when(a > thr).then(1.0).otherwise(0.0)
    )


def _fin_ratio2(a): return _fin_ratio_expr(a[0], a[1])
def _fin_abs_ratio2(a): return _fin_abs_ratio_expr(a[0], a[1])
def _fin_ratio3(a): return _fin_ratio_expr(a[0] - a[1], a[2].abs())
def _fin_ratio_sum3(a): return _fin_ratio_expr(a[0] + a[1], a[2])
def _fin_ratio_diff3(a): return _fin_ratio_expr(a[0] - a[1], a[2])
def _fin_ratio4(a): return _fin_ratio_expr(a[0] + a[1] - a[2], a[3])
def _fin_ratio5(a): return _fin_ratio_expr(a[0] + a[1] + a[2] - a[3], a[4])
def _fin_ratio6(a): return _fin_ratio_expr(a[0] - a[1] - a[2] - a[3] - a[4], a[5])
def _fin_ratio7(a): return _fin_ratio_expr(a[0] + a[1] + a[2] + a[3] + a[4] - a[5], a[6].abs())
def _fin_ratio_cap(a): return _fin_ratio_expr(a[0], a[0] + a[1])
def _fin_ratio_disp(a): return _fin_ratio_expr(a[0].abs(), a[1].abs())
def _fin_ratio_dsc(a): return _fin_ratio_expr(a[0], a[1] + a[2].abs())


def _fin_log_ratio(a):
    # log|x| - log|y| == log(|x|/|y|); denominator 0 / NaN / Inf -> NaN
    # (valuation/ops_v2._log_abs: log(x.abs().replace(0, nan))).
    q = a[0].abs() / a[1].abs()
    return (
        pl.when(a[1].is_null() | (a[1] == 0) | a[0].is_null())
        .then(None)
        .otherwise(
            pl.when(q.is_infinite() | q.is_nan() | (q <= 0))
            .then(None)
            .otherwise(q.log())
        )
    )
def _fin_ratio_burn(a): return _fin_neg_ocf_ratio_expr(a[0], a[1])


# op -> (n_operands, formula(cols) -> pl.Expr)
_FIN_ELEMENTWISE_OPS: dict[str, tuple[int, Callable[[list], pl.Expr]]] = {
    "fin_common_size": (2, _fin_ratio2),
    "fin_cash_conversion": (2, _fin_ratio2),
    "fin_acquisition_cash_intensity": (2, _fin_ratio2),
    "fin_borrowing_intensity": (2, _fin_ratio2),
    "fin_capex_intensity": (2, _fin_ratio2),
    "fin_goodwill_intensity": (2, _fin_ratio2),
    "fin_debt_repayment_intensity": (2, _fin_ratio2),
    "fin_contract_asset_intensity": (2, _fin_ratio2),
    "fin_contract_liability_intensity": (2, _fin_ratio2),
    "fin_oci_to_equity": (2, _fin_ratio2),
    "fin_interest_coverage_proxy": (2, _fin_abs_ratio2),
    "fin_discontinued_operation_ratio": (2, _fin_abs_ratio2),
    "fin_minority_profit_share": (2, _fin_abs_ratio2),
    "fin_fair_value_income_dependence": (2, _fin_abs_ratio2),
    "fin_investment_income_dependence": (2, _fin_abs_ratio2),
    "fin_other_earnings_dependence": (2, _fin_abs_ratio2),
    "fin_rd_capitalization_ratio": (2, _fin_ratio_cap),
    "fin_expectation_dispersion": (2, _fin_ratio_disp),
    "fin_cash_burn_runway": (2, _fin_ratio_burn),
    "fin_accrual_ratio": (3, _fin_ratio3),
    "fin_cash_earnings_gap": (3, _fin_ratio3),
    "fin_impairment_intensity": (3, _fin_ratio_sum3),
    "fin_lease_intensity": (3, _fin_ratio_sum3),
    "fin_rd_total_intensity": (3, _fin_ratio_sum3),
    "fin_contract_asset_liability_gap": (3, _fin_ratio_diff3),
    "fin_lease_asset_liability_gap": (3, _fin_ratio_diff3),
    "fin_deferred_tax_gap": (3, _fin_ratio_diff3),
    "fin_comprehensive_income_gap": (3, _fin_ratio_diff3),
    "fin_roe_cash_gap": (3, _fin_ratio_diff3),
    "fin_debt_service_coverage_proxy": (3, _fin_ratio_dsc),
    "fin_actual_expectation_divergence": (3, _fin_ratio3),
    "fin_surprise": (3, _fin_ratio3),
    "fin_net_borrowing_cashflow": (4, _fin_ratio4),
    "fin_financing_gap": (5, _fin_ratio5),
    "fin_core_earnings_ratio": (6, _fin_ratio6),
    "fin_noncore_income_ratio": (7, _fin_ratio7),
    # wave2 fin/valuation (2026-09-07): 纯元素级比值（NumpyKernels 参考，
    # 分母 ≤0/NaN → NaN；分母 abs 形态见 _fin_ratio_disp/_fin_ratio_dsc）。
    "free_float_turnover": (2, _fin_ratio2),
    "real_turnover_rate": (2, _fin_ratio2),
    "true_turnover_rate": (2, _fin_ratio2),
    # log|x| - log|y| == log(|x|/|y|)（_log_abs 参考：log(x.abs().replace(0,nan))，
    # 分母 0 → NaN）。
    "market_cap_free_cap_gap": (2, _fin_log_ratio),
}


def _compile_fin_elementwise(
    node: PlanNode,
    base: pl.LazyFrame,
    *,
    ctx: Any | None = None,
    memo: dict[tuple[Any, ...], pl.LazyFrame] | None = None,
) -> pl.LazyFrame | None:
    """Compile a pure-elementwise fin_* operator (no period walk)."""
    spec = _FIN_ELEMENTWISE_OPS.get(node.op)
    if spec is None:
        return None
    n_operands, formula = spec
    layers = []
    for i in range(n_operands):
        child = _compile_child(node, i, base, parent_op=node.op, ctx=ctx, memo=memo)
        if child is None:
            return None
        layers.append(child)
    joined = _join_multi(layers)
    names = [_VAL, "_y", "_z", "_w", "_u", "_t", "_s"]
    cols = [pl.col(names[i]) for i in range(n_operands)]
    expr = formula(cols)
    return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)


def _column_ref_name(node: PlanNode) -> str | None:
    """若 node 为 ``column`` 算子则返回列名字符串，否则 None。"""
    if node.op != "column":
        return None
    name = node.attrs.get("name")
    return str(name) if name else None


def _safe_div_null_expr(numer: pl.Expr, denom: pl.Expr, node: PlanNode) -> pl.Expr:
    """零/NULL 分母 → NULL（比率语义，不填 default）。"""
    from factor_engine.backend.numeric_semantics import protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    return (
        pl.when(numer.is_null() | denom.is_null())
        .then(None)
        .when(denom.abs() <= eps)
        .then(None)
        .otherwise(numer / denom)
    )


def _protected_div_expr(numer: pl.Expr, denom: pl.Expr, node: PlanNode) -> pl.Expr:
    from factor_engine.backend.elementwise_semantics import protected_div_polars
    from factor_engine.backend.numeric_semantics import protected_div_default, protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    default = _float_attr(node, "default", default=protected_div_default())
    return protected_div_polars(numer, denom, eps=eps, default=default)


def _protected_log_expr(val: pl.Expr, node: PlanNode) -> pl.Expr:
    from factor_engine.backend.elementwise_semantics import protected_log_polars
    from factor_engine.backend.numeric_semantics import protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    return protected_log_polars(val, eps=eps)


def _div_or_default_expr(numer: pl.Expr, denom: pl.Expr, node: PlanNode) -> pl.Expr:
    from factor_engine.backend.elementwise_semantics import div_or_default_polars
    from factor_engine.backend.numeric_semantics import protected_div_default, protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    default = _float_attr(node, "default", default=protected_div_default())
    return div_or_default_polars(numer, denom, eps=eps, default=default)


def _log_fill_invalid_expr(val: pl.Expr, node: PlanNode) -> pl.Expr:
    from factor_engine.backend.elementwise_semantics import log_fill_invalid_polars
    from factor_engine.backend.numeric_semantics import protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    return log_fill_invalid_polars(val, eps=eps)


def _truthy_expr(expr: pl.Expr) -> pl.Expr:
    """将 expr 转为布尔：NULL/NaN/0 → false。"""
    from factor_engine.backend.logical_semantics import truthy_polars_expr

    return truthy_polars_expr(expr)


def _bump_shared_long_lazy_hit(ctx: Any | None) -> None:
    """递增 ctx.runtime_stats 中 shared_long_lazy 缓存命中计数。"""
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


def _parse_ts_op_on_column(node: PlanNode):
    """识别 ``ts_*(column(x), window, ...)``，返回 (op, col, WindowSpec)。"""
    from factor_engine.backend.window_spec import WindowSpec

    op = _resolve(node.op)
    if op not in _FUSABLE_TS_ON_COLUMN:
        return None
    if not node.inputs:
        return None
    col = _column_ref_name(node.inputs[0])
    if not col:
        return None
    spec = _window_spec(node, default=1)
    return op, col, spec


def _ts_rolling_expr_on_column(op: str, col_name: str, spec) -> pl.Expr:
    """在宽表 base 列上直接构造 ts 窗口 expr（用于 DAG fusion，完整 WindowSpec）。"""
    from factor_engine.backend.numeric_semantics import std_ddof_value

    if op in _PANDAS_ROLLING_DROP_INF_OPS:
        # pandas rolling machinery treats ±Inf as missing; drop it so the
        # fused window agrees with the pandas reference and the duckdb SQL path.
        c = pl.when(
            pl.col(col_name).is_nan() | pl.col(col_name).is_infinite()
        ).then(None).otherwise(pl.col(col_name))
    else:
        c = pl.when(pl.col(col_name).is_nan()).then(None).otherwise(pl.col(col_name))
    w = spec.size
    mp = spec.min_periods
    if op == "ts_mean":
        return c.rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
    if op == "ts_sum":
        return c.rolling_sum(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
    if op == "ts_min":
        return c.rolling_min(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
    if op == "ts_max":
        return c.rolling_max(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
    if op == "ts_var":
        ddof = spec.ddof if hasattr(spec, "ddof") else std_ddof_value(op)
        return c.rolling_var(window_size=w, min_samples=mp, ddof=ddof).over(_INST, order_by=_TS)
    if op == "ts_std":
        ddof = spec.ddof if hasattr(spec, "ddof") else std_ddof_value(op)
        return c.rolling_std(window_size=w, min_samples=mp, ddof=ddof).over(_INST, order_by=_TS)
    if op in {"ts_delay", "delay"}:
        return c.shift(w).over(_INST, order_by=_TS)
    if op == "ts_delta":
        return c - c.shift(w).over(_INST, order_by=_TS)
    if op == "ts_pct":
        prev = c.shift(w).over(_INST, order_by=_TS)
        return pl.when(prev.is_null() | (prev == 0)).then(None).otherwise((c - prev) / prev)
    raise KeyError(op)


def _binary_fused_expr(op: str, left: pl.Expr, right: pl.Expr, node: PlanNode) -> pl.Expr:
    """将二元算子 op 应用于已融合的 left/right Expr（含 protected_div 等）。"""
    if op == "add":
        return left + right
    if op == "subtract":
        return left - right
    if op == "multiply":
        return left * right
    if op == "maximum":
        from factor_engine.backend.elementwise_semantics import max_horizontal_polars

        return max_horizontal_polars(left, right)
    if op == "minimum":
        from factor_engine.backend.elementwise_semantics import min_horizontal_polars

        return min_horizontal_polars(left, right)
    if op == "protected_div":
        return _protected_div_expr(left, right, node)
    if op == "div_or_default":
        return _div_or_default_expr(left, right, node)
    if op in {"safe_div_null", "safe_div"}:
        return _safe_div_null_expr(left, right, node)
    if op == "divide":
        from factor_engine.backend.inf_sanitize import apply_inf_policy_polars_fast

        return apply_inf_policy_polars_fast(left / right, "divide")
    if op == "power":
        return _safe_pow_expr(left, right)
    if op == "gt":
        from factor_engine.backend.elementwise_semantics import compare_polars

        return compare_polars("gt", left, right)
    if op == "lt":
        from factor_engine.backend.elementwise_semantics import compare_polars

        return compare_polars("lt", left, right)
    if op == "eq":
        from factor_engine.backend.elementwise_semantics import compare_polars

        return compare_polars("eq", left, right)
    if op == "ge":
        from factor_engine.backend.elementwise_semantics import compare_polars

        return compare_polars("ge", left, right)
    if op == "le":
        from factor_engine.backend.elementwise_semantics import compare_polars

        return compare_polars("le", left, right)
    if op == "ne":
        from factor_engine.backend.elementwise_semantics import compare_polars

        return compare_polars("ne", left, right)
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
    lop, lcol, lspec = left
    rop, rcol, rspec = right
    if lcol != rcol:
        return None
    schema = set(base.collect_schema().names())
    if lcol not in schema:
        return None
    tmp_l, tmp_r = "_fuse_l", "_fuse_r"
    l_expr = _ts_rolling_expr_on_column(lop, lcol, lspec)
    r_expr = _ts_rolling_expr_on_column(rop, rcol, rspec)
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
    parsed_items = [_parse_ts_op_on_column(leaf) for leaf in leaves]
    if any(p is None for p in parsed_items):
        return None
    cols = {p[1] for p in parsed_items if p is not None}
    if len(cols) != 1:
        return None
    col_name = next(iter(cols))
    schema = set(base.collect_schema().names())
    if col_name not in schema:
        return None
    tmp_names: list[str] = []
    tmp_exprs: list[pl.Expr] = []
    for i, item in enumerate(parsed_items):
        assert item is not None
        ts_op, _, spec = item
        name = f"_fuse_{i}"
        tmp_names.append(name)
        tmp_exprs.append(_ts_rolling_expr_on_column(ts_op, col_name, spec).alias(name))
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
        from factor_engine.backend.elementwise_semantics import max_horizontal_polars

        expr = max_horizontal_polars(lcol, rcol)
    elif op == "minimum":
        from factor_engine.backend.elementwise_semantics import min_horizontal_polars

        expr = min_horizontal_polars(lcol, rcol)
    elif op == "protected_div":
        safe_l = pl.when(lcol.is_finite()).then(lcol).otherwise(None)
        safe_r = pl.when(rcol.is_finite()).then(rcol).otherwise(None)
        expr = _protected_div_expr(safe_l, safe_r, node)
    elif op == "div_or_default":
        expr = _div_or_default_expr(lcol, rcol, node)
    elif op in {"safe_div_null", "safe_div"}:
        expr = _safe_div_null_expr(lcol, rcol, node)
    elif op == "divide":
        from factor_engine.backend.inf_sanitize import apply_inf_policy_polars_fast

        expr = apply_inf_policy_polars_fast(lcol / rcol, "divide")
    elif op == "power":
        expr = _safe_pow_expr(lcol, rcol)
    elif op == "gt":
        from factor_engine.backend.elementwise_semantics import compare_polars

        expr = compare_polars("gt", lcol, rcol)
    elif op == "lt":
        from factor_engine.backend.elementwise_semantics import compare_polars

        expr = compare_polars("lt", lcol, rcol)
    elif op == "eq":
        from factor_engine.backend.elementwise_semantics import compare_polars

        expr = compare_polars("eq", lcol, rcol)
    elif op == "ge":
        from factor_engine.backend.elementwise_semantics import compare_polars

        expr = compare_polars("ge", lcol, rcol)
    elif op == "le":
        from factor_engine.backend.elementwise_semantics import compare_polars

        expr = compare_polars("le", lcol, rcol)
    elif op == "ne":
        from factor_engine.backend.elementwise_semantics import compare_polars

        expr = compare_polars("ne", lcol, rcol)
    elif op == "and_":
        expr = pl.when(_truthy_expr(lcol) & _truthy_expr(rcol)).then(1.0).otherwise(0.0)
    elif op == "or_":
        expr = pl.when(_truthy_expr(lcol) | _truthy_expr(rcol)).then(1.0).otherwise(0.0)
    else:
        return None
    return base.select(pl.col(_TS), pl.col(_INST), expr.alias(_VAL))


def _try_coalesce_from_base_columns(node: PlanNode, base: pl.LazyFrame) -> pl.LazyFrame | None:
    """宽表 base 上两列 coalesce 融合，避免子树 join。"""
    if len(node.inputs) != 2:
        return None
    left_name = _column_ref_name(node.inputs[0])
    right_name = _column_ref_name(node.inputs[1])
    if not left_name or not right_name:
        return None
    schema = set(base.collect_schema().names())
    if left_name not in schema or right_name not in schema:
        return None

    def _na_as_null(name: str) -> pl.Expr:
        c = pl.col(name)
        return pl.when(c.is_nan()).then(None).otherwise(c)

    expr = pl.coalesce(_na_as_null(left_name), _na_as_null(right_name))
    return base.select(pl.col(_TS), pl.col(_INST), expr.alias(_VAL))


def _try_where_from_base_columns(node: PlanNode, base: pl.LazyFrame) -> pl.LazyFrame | None:
    """宽表 base 上 where(cond, a, b) 三列融合，避免子树 join。"""
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
    # Audit #19: a missing (NULL/NaN) condition stays missing — `where` is a
    # value selector and must not silently take the else branch (the pandas
    # reference ``ScalarBroadcastWhere`` propagates unknown conditions).
    cond_missing = cond.is_null() | cond.is_nan()
    expr = (
        pl.when(cond_missing)
        .then(None)
        .when(_truthy_expr(cond))
        .then(pl.col(a_name))
        .otherwise(pl.col(b_name))
    )
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
    # Define column references
    lcol = pl.col(left_name)
    rcol = pl.col(right_name)
    # pandas rolling pair stats treat ±Inf as missing (the same contract as
    # single-column rolling aggregations); drop Inf/NaN so a single Inf row
    # cannot poison the whole polars window.
    if op in {"ts_corr", "ts_cov", "ts_beta", "vwap"}:
        lcol = pl.when(lcol.is_nan() | lcol.is_infinite()).then(None).otherwise(lcol)
        rcol = pl.when(rcol.is_nan() | rcol.is_infinite()).then(None).otherwise(rcol)
    from factor_engine.backend.pair_window_spec import PairWindowSpec
    from factor_engine.backend.pairwise_rolling import polars_pairwise_output_guard, polars_ts_beta_expr, polars_vwap_expr

    # ts_beta production default is min_periods=5 (NEW-024 / rolling_beta); corr/cov stay at 2.
    default_mp = 5 if op == "ts_beta" else 2
    pspec = PairWindowSpec.from_plan_node(node, default_min_periods=default_mp)
    w = pspec.size
    mp = pspec.min_periods
    ddof = pspec.ddof
    if op == "ts_corr":
        # Use centered two-pass windows to avoid catastrophic cancellation in
        # Polars' E[xy] - E[x]E[y] implementation for large-offset series.
        expr = _rolling_corr_centered_expr(lcol, rcol, w, mp).over(_INST, order_by=_TS)
    elif op == "ts_cov":
        raw = pl.rolling_cov(lcol, rcol, window_size=w, min_samples=mp, ddof=ddof).over(
            _INST, order_by=_TS
        )
        # R38-blocker fix (R19-030 current-row policy)：与 ts_corr 相同——去掉 guard。
        expr = raw
    elif op == "ts_beta":
        expr = polars_ts_beta_expr(
            lcol, rcol, window=w, min_periods=mp, ddof=ddof
        )
    elif op == "vwap":
        vwap_spec = _window_spec(node, default=20)
        expr = polars_vwap_expr(
            lcol, rcol, window=vwap_spec.size, min_periods=vwap_spec.min_periods
        )
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
        "div_or_default",
        "safe_div_null",
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


def _safe_pow_expr(base: pl.Expr, exp: pl.Expr) -> pl.Expr:
    """非法定义域返回 NULL（对齐 SQL emitter）；溢出 ±Inf → NULL（power semantics）。"""
    from factor_engine.backend.inf_sanitize import apply_inf_policy_polars_fast

    raw = (
        pl.when(base.is_null() | exp.is_null())
        .then(None)
        .when((base < 0) & (exp != exp.floor()))
        .then(None)
        .when((base == 0) & (exp < 0))
        .then(None)
        .otherwise(base.pow(exp))
    )
    return apply_inf_policy_polars_fast(raw, "power")



def _truthy(col: str) -> pl.Expr:
    """列名版 truthy。"""
    from factor_engine.backend.logical_semantics import truthy_polars_expr

    return truthy_polars_expr(pl.col(col))


def _cs_rank_01_on(value_col: str) -> pl.Expr:
    """截面 0-1 rank，指定列名（DAG fusion 用）。"""
    from factor_engine.backend.rank_spec import polars_cs_rank_expr

    return polars_cs_rank_expr(value_col, partition_cols=(_TS,), order_by=_INST, canon="rank")


def _zscore_on(value_col: str, *, canon: str = "zscore") -> pl.Expr:
    """指定列名的截面 zscore expr（按 ts 分区，语义见 numeric_semantics）。"""
    from factor_engine.backend.numeric_semantics import std_ddof_value, zscore_zero_std_fill

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


def _normalize_on(value_col: str) -> pl.Expr:
    """截面 min-max normalize；单有效值 → NULL，常数截面 → 0.5。"""
    lo = pl.col(value_col).min().over(_TS, order_by=_INST)
    hi = pl.col(value_col).max().over(_TS, order_by=_INST)
    span = hi - lo
    cnt = pl.col(value_col).count().over(_TS, order_by=_INST)
    return (
        pl.when(pl.col(value_col).is_null())
        .then(None)
        .when(cnt <= 1)
        .then(None)
        .when(span.is_null() | (span == 0))
        .then(0.5)
        .otherwise((pl.col(value_col) - lo) / span)
    )


def _group_normalize_on(value_col: str, over_keys: tuple[str, ...]) -> pl.Expr:
    """组内 normalize；常数/单值组 → 0.5（对齐 pandas / SQL）。"""
    lo = pl.col(value_col).min().over(*over_keys, order_by=_INST)
    hi = pl.col(value_col).max().over(*over_keys, order_by=_INST)
    span = hi - lo
    return (
        pl.when(pl.col(value_col).is_null())
        .then(None)
        .when(span.is_null() | (span == 0))
        .then(0.5)
        .otherwise((pl.col(value_col) - lo) / span)
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
    ts_op, col_name, spec = ts
    schema = set(base.collect_schema().names())
    if col_name not in schema:
        return None
    tmp = "_fuse_ts"
    ts_expr = _ts_rolling_expr_on_column(ts_op, col_name, spec)
    lf = base.with_columns(ts_expr.alias(tmp))
    if op == "rank":
        out_expr = _cs_rank_01_on(tmp)
    elif op in {"rank_pct", "cs_pct_rank"}:
        from factor_engine.backend.rank_spec import polars_cs_rank_expr

        out_expr = polars_cs_rank_expr(tmp, partition_cols=(_TS,), order_by=_INST, canon=op)
    elif op == "zscore":
        out_expr = _zscore_on(tmp, canon="zscore")
    elif op == "neg":
        out_expr = -pl.col(tmp)
    elif op == "abs":
        out_expr = pl.col(tmp).abs()
    elif op == "scale":
        to_val = _scale_to_value(node)
        from factor_engine.backend.cross_section_spec import polars_scale_expr

        out_expr = polars_scale_expr(tmp, to_val, partition_cols=(_TS,), order_by=_INST)
    elif op == "normalize":
        out_expr = _normalize_on(tmp)
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
    """截面 0-1 rank；语义见 ``rank_spec.rank_spec_for``。"""
    from factor_engine.backend.rank_spec import polars_cs_rank_expr

    return polars_cs_rank_expr(_VAL, partition_cols=(_TS,), order_by=_INST, canon=canon)


def _cs_rank_pct(*, canon: str = "rank_pct") -> pl.Expr:
    """截面百分位 rank expr（rank/n，语义见 rank_spec）。"""
    from factor_engine.backend.rank_spec import polars_cs_rank_expr

    return polars_cs_rank_expr(_VAL, partition_cols=(_TS,), order_by=_INST, canon=canon)


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
    """带结构键 memo 的 PlanNode → long LazyFrame 编译入口（CSE dedupe）。"""
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
    """PlanNode 递归 lowering 核心：按算子分发 native expr / fusion / map_groups / bridge。"""
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

    if op in _FIN_ELEMENTWISE_OPS:
        return _compile_fin_elementwise(node, base, ctx=ctx, memo=memo)

    if op in _BINARY_FUSION_OPS:
        if len(node.inputs) != 2:
            return None
        fused_ts = _try_fuse_binary_ts_on_column(node, base, op)
        if fused_ts is not None:
            return fused_ts
        fused = _try_binary_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
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
            from factor_engine.backend.elementwise_semantics import max_horizontal_polars

            expr = max_horizontal_polars(pl.col(_VAL), pl.col("_y"))
        elif op == "minimum":
            from factor_engine.backend.elementwise_semantics import min_horizontal_polars

            expr = min_horizontal_polars(pl.col(_VAL), pl.col("_y"))
        elif op == "protected_div":
            expr = _protected_div_expr(pl.col(_VAL), pl.col("_y"), node)
        elif op == "div_or_default":
            expr = _div_or_default_expr(pl.col(_VAL), pl.col("_y"), node)
        elif op in {"safe_div_null", "safe_div"}:
            expr = _safe_div_null_expr(pl.col(_VAL), pl.col("_y"), node)
        elif op == "divide":
            from factor_engine.backend.inf_sanitize import apply_inf_policy_polars_fast

            expr = apply_inf_policy_polars_fast(pl.col(_VAL) / pl.col("_y"), "divide")
        elif op == "power":
            expr = _safe_pow_expr(pl.col(_VAL), pl.col("_y"))
        elif op in {"gt", "lt", "eq", "ge", "le", "ne"}:
            from factor_engine.backend.elementwise_semantics import compare_polars

            expr = compare_polars(op, pl.col(_VAL), pl.col("_y"))
        elif op == "and_":
            expr = pl.when(_truthy(_VAL) & _truthy("_y")).then(1.0).otherwise(0.0)
        elif op == "or_":
            expr = pl.when(_truthy(_VAL) | _truthy("_y")).then(1.0).otherwise(0.0)
        else:
            return None
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "inverse":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null() | (pl.col(_VAL) == 0))
            .then(None)
            .otherwise(1.0 / pl.col(_VAL))
            .alias(_VAL)
        )

    if op == "is_finite":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.elementwise_semantics import is_finite_polars_expr

        return inner.with_columns(is_finite_polars_expr(_VAL).alias(_VAL))

    if op == "is_infinite":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.elementwise_semantics import is_infinite_polars_expr

        return inner.with_columns(is_infinite_polars_expr(_VAL).alias(_VAL))

    if op == "is_null":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.logical_semantics import is_null_polars_expr

        return inner.with_columns(is_null_polars_expr(_VAL).alias(_VAL))

    if op == "is_not_null":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.logical_semantics import is_not_null_polars_expr

        return inner.with_columns(is_not_null_polars_expr(_VAL).alias(_VAL))

    if op == "is_nan":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.logical_semantics import is_nan_polars_expr

        return inner.with_columns(is_nan_polars_expr(_VAL).alias(_VAL))

    if op == "neg":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        return inner.with_columns((-pl.col(_VAL)).alias(_VAL)) if inner is not None else None

    if op == "abs":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).abs().alias(_VAL)) if inner is not None else None

    if op == "sign":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).sign().alias(_VAL)) if inner is not None else None

    if op == "log":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null() | (pl.col(_VAL) < 0))
            .then(None)
            .otherwise(pl.col(_VAL).log())
            .alias(_VAL)
        )

    if op == "exp":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.inf_sanitize import apply_inf_policy_polars_fast

        return inner.with_columns(
            apply_inf_policy_polars_fast(pl.col(_VAL).exp(), "exp").alias(_VAL)
        )

    if op == "sqrt":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null() | (pl.col(_VAL) < 0))
            .then(None)
            .otherwise(pl.col(_VAL).sqrt())
            .alias(_VAL)
        )

    if op == "floor":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).floor().alias(_VAL)) if inner is not None else None

    if op == "ceil":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        return inner.with_columns(pl.col(_VAL).ceil().alias(_VAL)) if inner is not None else None

    if op == "power":
        if len(node.inputs) != 2:
            return None
        fused = _try_binary_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(_safe_pow_expr(pl.col(_VAL), pl.col("_y")).alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "protected_log":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_protected_log_expr(pl.col(_VAL), node).alias(_VAL))

    if op == "log_fill_invalid":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_log_fill_invalid_expr(pl.col(_VAL), node).alias(_VAL))

    if op == "protected_sqrt":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        clipped = pl.max_horizontal(pl.col(_VAL), pl.lit(0.0))
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null()).then(None).otherwise(clipped.sqrt()).alias(_VAL)
        )

    if op in {"clip", "cap"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.numeric_semantics import std_ddof_value

        spec = _window_spec(node)
        w = spec.size
        mp = spec.min_periods
        if op in {"ts_std", "ts_var"}:
            ddof = spec.ddof if "ddof" in (node.attrs or {}) else std_ddof_value(op)
        else:
            ddof = 1
        if op == "ts_mean":
            expr = pl.col(_VAL).rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        elif op == "ts_sum":
            expr = pl.col(_VAL).rolling_sum(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        elif op == "ts_min":
            expr = pl.col(_VAL).rolling_min(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        elif op == "ts_max":
            expr = pl.col(_VAL).rolling_max(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        elif op == "ts_var":
            expr = pl.col(_VAL).rolling_var(window_size=w, min_samples=mp, ddof=ddof).over(_INST, order_by=_TS)
        elif op == "ts_median":
            expr = pl.col(_VAL).rolling_median(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        else:
            expr = pl.col(_VAL).rolling_std(window_size=w, min_samples=mp, ddof=ddof).over(_INST, order_by=_TS)
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_zscore":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.cleaned_operators.common.ts_zscore_spec import TSZScoreSpec
        attrs = node.attrs or {}
        spec = TSZScoreSpec.resolve(
            window=_window_spec(node).size,
            min_periods=_window_spec(node).min_periods,
            null_policy=attrs.get("null_policy", "ignore"),
            nan_policy=attrs.get("nan_policy", "propagate"),
            includes_current_bar=attrs.get("includes_current_bar", True),
            ddof=attrs.get("ddof", 1),
            zero_std_policy=attrs.get("zero_std_policy", "zero"),
        )
        current = pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL)).otherwise(None)
        stats = current if spec.includes_current_bar else current.shift(1).over(_INST, order_by=_TS)
        mean = stats.rolling_mean(spec.window.size, min_samples=spec.window.min_periods).over(_INST, order_by=_TS)
        std = stats.rolling_std(spec.window.size, min_samples=spec.window.min_periods, ddof=spec.window.ddof).over(_INST, order_by=_TS)
        out = (pl.when(current.is_null())
            .then(None)
            .when(std.is_null())
            .then(None))
        if spec.window.nan_policy == "propagate" or spec.window.null_policy.value == "propagate":
            bad = stats.is_null().cast(pl.Int64).rolling_sum(spec.window.size, min_samples=1).over(_INST, order_by=_TS) > 0
            out = out.when(bad).then(None)
        out = out.when(std == 0).then(0.0 if spec.zero_std_policy == "zero" else None).otherwise((current - mean) / std)
        return inner.with_columns(out.alias(_VAL))

    if op == "ts_corr":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        from factor_engine.backend.pair_window_spec import PairWindowSpec
        from factor_engine.backend.pairwise_rolling import polars_pairwise_output_guard

        pspec = PairWindowSpec.from_plan_node(node)
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        # Finite-pair mask (same contract as fusion path / pandas).
        lcol = (
            pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite())
            .then(None)
            .otherwise(pl.col(_VAL))
        )
        rcol = (
            pl.when(pl.col("_y").is_nan() | pl.col("_y").is_infinite())
            .then(None)
            .otherwise(pl.col("_y"))
        )
        # PARITY-SWEEP-R56: use polars' native ``rolling_corr`` (matches pandas
        # ``rolling.corr`` bit-for-bit to 1e-10) instead of the former centered
        # two-pass expression whose anchor/subtraction order diverged from
        # pandas by ~1e-10 on large-offset inputs (test_three_backend_parity
        # asserts rtol=atol=1e-10).
        corr = pl.rolling_corr(
            lcol,
            rcol,
            window_size=pspec.size,
            min_samples=pspec.min_periods,
        ).over(_INST, order_by=_TS)
        return joined.with_columns(corr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ts_cov":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        from factor_engine.backend.pair_window_spec import PairWindowSpec
        from factor_engine.backend.pairwise_rolling import polars_pairwise_output_guard

        pspec = PairWindowSpec.from_plan_node(node)
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        lcol = (
            pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite())
            .then(None)
            .otherwise(pl.col(_VAL))
        )
        rcol = (
            pl.when(pl.col("_y").is_nan() | pl.col("_y").is_infinite())
            .then(None)
            .otherwise(pl.col("_y"))
        )
        cov = pl.rolling_cov(
            lcol,
            rcol,
            window_size=pspec.size,
            min_samples=pspec.min_periods,
            ddof=pspec.ddof,
        ).over(_INST, order_by=_TS)
        # R38-blocker fix (R19-030 current-row policy)：去掉 guard（见 ts_corr）。
        out = cov
        return joined.with_columns(out.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ts_beta":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        from factor_engine.backend.pair_window_spec import PairWindowSpec
        from factor_engine.backend.pairwise_rolling import polars_ts_beta_expr

        pspec = PairWindowSpec.from_plan_node(node, default_min_periods=5)
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        beta = polars_ts_beta_expr(
            pl.col(_VAL),
            pl.col("_y"),
            window=pspec.size,
            min_periods=pspec.min_periods,
            ddof=pspec.ddof,
        )
        return joined.with_columns(beta.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vwap":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        from factor_engine.backend.pairwise_rolling import polars_vwap_expr

        spec = _window_spec(node, default=20)
        price = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        vol = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if price is None or vol is None:
            return None
        joined = _join_binary(price, vol)
        vwap = polars_vwap_expr(
            pl.col(_VAL),
            pl.col("_y"),
            window=spec.size,
            min_periods=spec.min_periods,
        )
        return joined.with_columns(vwap.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "cum_sum":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_pandas_aligned_cum_sum_expr().alias(_VAL))

    if op == "cum_max":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_pandas_aligned_cum_max_expr().alias(_VAL))

    if op == "cum_min":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_pandas_aligned_cum_min_expr().alias(_VAL))

    if op == "cum_prod":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_pandas_aligned_cum_prod_expr().alias(_VAL))

    if op == "ts_rank":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        spec = _window_spec(node)
        from factor_engine.backend.rank_spec import polars_ts_rank_expr

        return inner.with_columns(
            polars_ts_rank_expr(
                _VAL,
                window=spec.size,
                inst_col=_INST,
                ts_col=_TS,
                min_periods=spec.min_periods,
            ).alias(_VAL)
        )

    if op in {"ts_ema", "ema", "ewm_std", "ts_ewm_std", "ewm_var"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        alpha = _ewm_alpha(node)
        if op in {"ts_ema", "ema"}:
            # pandas EWM excludes +/-Inf like a missing observation while
            # retaining its position in the recursive weight calculation.
            # The child sanitizer converts it to NULL; ignore_nulls=False is
            # therefore required for exact post-gap values.
            expr = pl.col(_VAL).ewm_mean(
                alpha=alpha, adjust=False, ignore_nulls=False
            )
        elif op in {"ewm_std", "ts_ewm_std"}:
            raw = pl.col(_VAL).ewm_std(
                alpha=alpha,
                adjust=False,
                bias=False,
                min_samples=1,
                ignore_nulls=False,
            )
            expr = pl.when(pl.col(_VAL).is_not_null().cum_sum() < 2).then(None).otherwise(raw)
        else:
            expr = pl.col(_VAL).ewm_var(alpha=alpha, adjust=False)
        # R-EMA-DIVERGENCE: Polars ewm_mean produces null at NaN input positions,
        # while pandas ewm().mean() carries forward the previous EMA state.
        # Forward-fill within each instrument group to match pandas semantics.
        return inner.with_columns(
            expr.forward_fill().over(_INST, order_by=_TS).alias(_VAL)
        )

    if op in {"coalesce"}:
        if len(node.inputs) < 2:
            return None
        if len(node.inputs) == 2:
            fused = _try_coalesce_from_base_columns(node, base)
            if fused is not None:
                return fused
            left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
            right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
            if left is None or right is None:
                return None
            joined = _join_binary(left, right)
            return joined.with_columns(pl.coalesce(pl.col(_VAL), pl.col("_y")).alias(_VAL)).select(
                _TS, _INST, _VAL
            )
        acc = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if acc is None:
            return None
        for child in node.inputs[1:]:
            nxt = _compile_polars(child, base, ctx=ctx, memo=memo)
            if nxt is None:
                return None
            acc = _join_binary(acc, nxt)
            acc = acc.with_columns(pl.coalesce(pl.col(_VAL), pl.col("_y")).alias(_VAL)).select(
                _TS, _INST, _VAL
            )
        return acc

    if op in {"where", "if_else"}:
        if len(node.inputs) != 3:
            return None
        fused = _try_where_from_base_columns(node, base)
        if fused is not None:
            return fused
        cond = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        a = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        b = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        if cond is None or a is None or b is None:
            return None
        joined = _join_triple(cond, a, b)
        # Audit #19: NULL/NaN condition stays missing (pandas reference
        # ``ScalarBroadcastWhere`` propagates unknown conditions; the truthy
        # NULL-as-false rule only applies to boolean-producing ops).
        return joined.with_columns(
            pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
            .then(None)
            .when(_truthy(_VAL))
            .then(pl.col("_ym"))
            .otherwise(pl.col("_y"))
            .alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op in {"gt", "lt", "eq", "ge", "le", "ne"}:
        if len(node.inputs) != 2:
            return None
        fused = _try_binary_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
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
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
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
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        return joined.with_columns(
            pl.when(_truthy(_VAL) | _truthy("_y")).then(1.0).otherwise(0.0).alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "not_":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            pl.when(_truthy(_VAL)).then(0.0).otherwise(1.0).alias(_VAL)
        )

    if op in {"fillna_const", "fillna"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        const = _const_fill_value(node, default=0.0)
        if const is None:
            const = 0.0
        return inner.with_columns(
            pl.when(
                pl.col(_VAL).is_null()
                | pl.col(_VAL).is_nan()
                | pl.col(_VAL).is_infinite()
            )
            .then(const)
            .otherwise(pl.col(_VAL))
            .alias(_VAL)
        )

    if op in {"cs_quantile", "c_percentile"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.stat_valid import polars_rank_input

        p = _float_attr(node, "p", default=0.5)
        pos_p = _literal_value(node, 0)
        if pos_p is not None:
            p = pos_p
        value = polars_rank_input(_VAL, exclude_nan=True)
        q = value.quantile(quantile=p, interpolation="linear").over(_TS, order_by=_INST)
        return inner.with_columns(q.alias(_VAL))

    if op == "winsorize":
        from factor_engine.backend.plan_params import parse_winsorize_quantiles

        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lo_p, hi_p = parse_winsorize_quantiles(node)
        lo = pl.col(_VAL).quantile(quantile=lo_p, interpolation="linear").over(_TS, order_by=_INST)
        hi = pl.col(_VAL).quantile(quantile=hi_p, interpolation="linear").over(_TS, order_by=_INST)
        # pandas winsorize ends with ``.replace([inf, -inf], nan)`` — an Inf
        # input never survives as a clipped extreme.
        from factor_engine.backend.inf_sanitize import apply_inf_policy_polars_fast
        return inner.with_columns(
            apply_inf_policy_polars_fast(pl.col(_VAL).clip(lo, hi), "winsorize").alias(_VAL)
        )

    if op in {"cs_resid", "cs_regression"}:
        from factor_engine.backend.plan_params import int_mode_from_plan_node
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        x_layer = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if y_layer is None or x_layer is None:
            return None
        joined = y_layer.join(x_layer.rename({_VAL: "_x"}), on=[_TS, _INST], how="left")
        beta, alpha, n_valid = _cs_ols_exprs(_VAL, "_x")
        fit = alpha + beta * pl.col("_x")
        mode = 0 if op == "cs_resid" else int_mode_from_plan_node(node, input_index=2, default=0)
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

    if op == "size_neutralize":
        # size_neutralize(x, market_cap) == cs_resid(x, log(market_cap))
        # Legal caps only (R19-024): finite & >0 → log; else null (no silent clip).
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        cap_layer = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if y_layer is None or cap_layer is None:
            return None
        joined = y_layer.join(
            cap_layer.rename({_VAL: "_cap"}), on=[_TS, _INST], how="left"
        ).with_columns(
            pl.when(
                pl.col("_cap").is_not_null()
                & pl.col("_cap").is_finite()
                & (pl.col("_cap") > 0.0)
            )
            .then(pl.col("_cap").log())
            .otherwise(None)
            .alias("_ln")
        )
        beta, alpha, n_valid = _cs_ols_exprs(_VAL, "_ln")
        fit = alpha + beta * pl.col("_ln")
        return joined.with_columns(
            pl.when(pl.col(_VAL).is_null() | pl.col("_ln").is_null())
            .then(None)
            .when(n_valid < 3)
            .then(None)
            .otherwise(pl.col(_VAL) - fit)
            .alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op == "industry_size_neutralize":
        # FWL (R19-023): demean y and log(size) within industry, then residual.
        # log(size) uses legal caps only (R19-024): finite & >0 → log; else null.
        if len(node.inputs) < 3:
            return None
        y_layer = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        ind_layer = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        cap_layer = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        if y_layer is None or ind_layer is None or cap_layer is None:
            return None
        joined = (
            y_layer.join(ind_layer.rename({_VAL: _GRP}), on=[_TS, _INST], how="left")
            .join(cap_layer.rename({_VAL: "_cap"}), on=[_TS, _INST], how="left")
            .with_columns(
                pl.when(
                    pl.col("_cap").is_not_null()
                    & pl.col("_cap").is_finite()
                    & (pl.col("_cap") > 0.0)
                )
                .then(pl.col("_cap").log())
                .otherwise(None)
                .alias("_ln"),
            )
            .with_columns(
                pl.when(pl.col(_VAL).is_null())
                .then(None)
                .otherwise(
                    pl.col(_VAL) - pl.col(_VAL).mean().over(_TS, _GRP, order_by=_INST)
                )
                .alias("_dm"),
                pl.when(pl.col("_ln").is_null())
                .then(None)
                .otherwise(
                    pl.col("_ln") - pl.col("_ln").mean().over(_TS, _GRP, order_by=_INST)
                )
                .alias("_ln_dm"),
            )
        )
        beta, alpha, n_valid = _cs_ols_exprs("_dm", "_ln_dm")
        fit = alpha + beta * pl.col("_ln_dm")
        return joined.with_columns(
            pl.when(pl.col("_dm").is_null() | pl.col("_ln_dm").is_null())
            .then(None)
            .when(n_valid < 3)
            .then(None)
            .otherwise(pl.col("_dm") - fit)
            .alias(_VAL)
        ).select(_TS, _INST, _VAL)

    if op in {
        "group_rank",
        "group_mean",
        "group_sum",
        "group_min",
        "group_max",
        "group_count",
        "group_zscore",
        "group_neutralize",
        "group_std",
        "group_normalize",
        "group_percentile",
        "group_winsorize",
        "group_decay_linear",
        "group_rank_weighted_value",
    }:
        if not node.inputs:
            return None
        val = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if val is None:
            return None
        if len(node.inputs) >= 2 and node.inputs[1].op not in {"literal"}:
            grp = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
            if grp is None:
                return None
            joined = val.join(grp.rename({_VAL: _GRP}), on=[_TS, _INST], how="left")
            over_keys = (_TS, _GRP)
        else:
            if op == "group_percentile":
                # Audited semantics require explicit group labels. Returning
                # unsupported here lets the engine use the audited bridge,
                # which raises instead of silently treating the whole cross
                # section as one group.
                return None
            joined = val.with_columns(pl.lit(1.0).alias(_GRP))
            over_keys = (_TS, _GRP)
        if op == "group_percentile":
            p = _float_attr(node, "p", default=0.5)
            pos_p = _literal_value(node, 1)
            if pos_p is not None:
                p = pos_p
            side = str(node.attrs.get("side", "top")).lower()
            if side not in {"top", "bottom"}:
                return None
            n = pl.col(_VAL).count().over(*over_keys, order_by=_INST)
            rank_input = pl.col(_VAL) if side == "bottom" else -pl.col(_VAL)
            frac = (
                rank_input.rank(method="average").over(*over_keys, order_by=_INST)
                / pl.when(n > 0).then(n.cast(pl.Float64)).otherwise(None)
            )
            expr = (
                pl.when(pl.col(_VAL).is_null())
                .then(None)
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
            # NaN/NULL members must stay NaN/NULL (audit #19).  Inf follows
            # pandas ``notna`` (participates / may poison), not finite-mask.
            expr = (
                pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
                .then(None)
                .otherwise(pl.col(_VAL).mean().over(*over_keys, order_by=_INST))
            )
        elif op == "group_sum":
            expr = (
                pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
                .then(None)
                .otherwise(pl.col(_VAL).sum().over(*over_keys, order_by=_INST))
            )
        elif op == "group_min":
            expr = (
                pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
                .then(None)
                .otherwise(pl.col(_VAL).min().over(*over_keys, order_by=_INST))
            )
        elif op == "group_max":
            expr = (
                pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
                .then(None)
                .otherwise(pl.col(_VAL).max().over(*over_keys, order_by=_INST))
            )
        elif op == "group_count":
            expr = (
                pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
                .then(None)
                .otherwise(
                    pl.col(_VAL).count().over(*over_keys, order_by=_INST).cast(pl.Float64)
                )
            )
        elif op == "group_std":
            from factor_engine.backend.numeric_semantics import std_ddof_value

            # pandas GroupStd uses ``notna`` (Inf included); a group containing
            # ±Inf has std NaN -> the reference fills 0 for the WHOLE group.
            # Polars' std() drops non-finite (giving a finite std of the remaining
            # members), so detect any non-finite member and force the 0 fill.
            group_has_inf = (
                pl.col(_VAL).is_infinite().max().over(*over_keys, order_by=_INST).cast(pl.Boolean)
            )
            cnt = pl.col(_VAL).count().over(*over_keys, order_by=_INST)
            std_expr = pl.col(_VAL).std(ddof=std_ddof_value("group_std")).over(*over_keys, order_by=_INST)
            expr = (
                pl.when(group_has_inf)
                .then(0.0)
                .when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
                .then(None)
                .when(cnt < 2)
                .then(0.0)
                .when(std_expr.is_null() | std_expr.is_infinite())
                .then(0.0)
                .otherwise(std_expr)
            )
        elif op in {"group_zscore", "group_neutralize"}:
            from factor_engine.backend.numeric_semantics import std_ddof_value, zscore_zero_std_fill

            # PARITY-SWEEP-R56: pandas ``GroupZScore`` uses ``x_slice.notna()``
            # (Inf INCLUDED) for the group mask, so a group containing ±Inf has
            # mean=Inf and std=NaN -> the reference's ``std != 0 and not isna``
            # branch is False and the WHOLE group is set to 0.  Replicate: if any
            # group member is non-finite, output the zero_fill for every member
            # (matching the pandas 0 output), instead of dropping Inf and
            # computing a finite z-score.
            group_has_inf = (
                pl.col(_VAL).is_infinite().max().over(*over_keys, order_by=_INST).cast(pl.Boolean)
            )
            mean = pl.col(_VAL).mean().over(*over_keys, order_by=_INST)
            if op == "group_neutralize":
                # PARITY-SWEEP-R56: pandas ``GroupDemean`` (group_neutralize) uses
                # an ``np.isfinite`` mask for group membership — a ±Inf member is
                # EXCLUDED from the group mean (pandas nanmean over finite), the
                # finite members are demeaned by the finite mean, and the Inf cell
                # stays NaN.  Polars' ``mean()`` KEEPS Inf (mean=Inf), so mask Inf
                # to NULL before the mean.
                safe = (
                    pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite())
                    .then(None)
                    .otherwise(pl.col(_VAL))
                )
                mean_f = safe.mean().over(*over_keys, order_by=_INST)
                expr = (
                    pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite())
                    .then(None)
                    .otherwise(pl.col(_VAL) - mean_f)
                )
            else:
                std = pl.col(_VAL).std(ddof=std_ddof_value("group_zscore")).over(*over_keys, order_by=_INST)
                zero_fill = zscore_zero_std_fill("group_zscore")
                expr = (
                    pl.when(group_has_inf)
                    .then(zero_fill)
                    .when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
                    .then(None)
                    .when(std.is_null() | (std == 0) | std.is_infinite())
                    .then(zero_fill)
                    .otherwise((pl.col(_VAL) - mean) / std)
                )
        elif op == "group_normalize":
            expr = _group_normalize_on(_VAL, over_keys)
        elif op == "group_rank_weighted_value":
            # common::GroupRankWeightedValuePolars: x * avg_rank / sum(avg_rank)
            # per group (average-rank; ties averaged) — pure polars over-expr.
            rank_e = pl.col(_VAL).rank(method="average").over(*over_keys, order_by=_INST)
            sum_r = rank_e.sum().over(*over_keys, order_by=_INST)
            expr = pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan()).then(None).otherwise(
                pl.when((sum_r.is_null()) | (sum_r == 0)).then(None).otherwise(pl.col(_VAL) * rank_e / sum_r))
        else:
            from factor_engine.backend.rank_spec import polars_cs_rank_expr

            expr = polars_cs_rank_expr(_VAL, partition_cols=over_keys, order_by=_INST, canon="group_rank")
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "cs_shrink_to_group_mean":
        # common::CsShrinkToGroupMeanPolars: shrunk = x*(1-intensity)+group_mean*intensity;
        # group mean computed over finite x only; unknown/invalid group label -> NaN.
        if not node.inputs:
            return None
        val = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if val is None:
            return None
        joined = val
        if len(node.inputs) >= 2 and node.inputs[1].op not in {"literal"}:
            grp = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
            if grp is None:
                return None
            joined = val.join(grp.rename({_VAL: _GRP}), on=[_TS, _INST], how="left")
            over_keys = (_TS, _GRP)
        else:
            return None
        shrink = _literal_value(node, 2)
        if shrink is not None:
            shrink = float(shrink)
        else:
            shrink = float(_float_attr(node, "shrinkage_intensity", default=0.5))
        gm = pl.col(_VAL).mean().over(*over_keys, order_by=_INST)
        expr = pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan()).then(None).otherwise(
            pl.when(pl.col(_GRP).is_null()).then(None).otherwise(
                (1.0 - shrink) * pl.col(_VAL) + shrink * gm
            )
        )
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "group_weighted_zscore":
        # common::GroupWeightedZscorePolars: (x - Σw·x/Σw) / sqrt(Σw·(x-μ)²/Σw)
        # per group; zero-weight-sum / single-effective-obs -> NaN.  Weight and
        # value pairs come in as separate columns; invalid (null/NaN/inf)
        # pairs are dropped from both numerator and denominator exactly like
        # the pandas reference.
        if len(node.inputs) < 2:
            return None
        val = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if val is None:
            return None
        grp = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if grp is None:
            return None
        joined = val.join(grp.rename({_VAL: _GRP}), on=[_TS, _INST], how="left")
        wname = None
        if len(node.inputs) >= 3 and node.inputs[2].op not in {"literal"}:
            wcol = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
            if wcol is not None:
                joined = joined.join(wcol.rename({_VAL: "_w"}), on=[_TS, _INST], how="left")
                wname = "_w"
        if wname is None:
            wl = _literal_value(node, 2) if len(node.inputs) >= 3 else None
            joined = joined.with_columns(pl.lit(float(wl) if wl is not None else 1.0).alias("_w"))
            wname = "_w"
        over_keys = (_TS, _GRP)
        v = pl.col(_VAL)
        w = pl.col("_w")
        valid_pair = (
            v.is_not_null() & ~v.is_nan() & ~v.is_infinite()
            & w.is_not_null() & ~w.is_nan() & ~w.is_infinite() & (w >= 0)
        )
        vv = pl.when(valid_pair).then(v)
        ww = pl.when(valid_pair).then(w)
        sw = ww.sum().over(*over_keys, order_by=_INST)
        mean = (vv * ww).sum().over(*over_keys, order_by=_INST) / sw
        var = ((vv - mean) ** 2 * ww).sum().over(*over_keys, order_by=_INST) / sw
        expr = (
            pl.when(pl.col(_GRP).is_null())
            .then(None)
            .when(sw.is_null() | (sw <= 0))
            .then(None)
            .when(var.is_null() | (var <= 0))
            .then(None)
            .otherwise((v - mean) / var.sqrt())
        )
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    # ------------------------------------------------------------------
    # overnight/intraday return decomposition family (intraday/overnight.py):
    # overnight = open/pre_close - 1, intraday = close/open - 1 (both derived
    # from three daily-panel inputs), then per-op trailing statistics.  Safe
    # division replicates _safe_ret (denominator 0/NaN/inf -> NaN); pair
    # rolling stats mask non-finite pairs exactly like pandas rolling.cov.
    # ------------------------------------------------------------------

    if op in {"ts_overnight_intraday_cov", "ts_overnight_intraday_spread",
              "ts_overnight_intraday_sign_agreement", "ts_opening_mispricing_score"}:
        if len(node.inputs) < 3:
            return None
        close_in = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        open_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        pre_in = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        if close_in is None or open_in is None or pre_in is None:
            return None
        joined = _join_triple(open_in, pre_in, close_in)
        o_c = pl.col(_VAL)
        pc = pl.col("_ym")
        c_c = pl.col("_y")
        w = _window_int(node, default=60)

        def _safe_ret_expr(num: pl.Expr, den: pl.Expr) -> pl.Expr:
            # pandas _safe_ret: a/b.replace(0, nan) - 1.0 — the "- 1" is part
            # of the return definition, not optional.
            num_ok = pl.when(num.is_nan() | num.is_infinite()).then(None).otherwise(num)
            d = pl.when(den.is_nan() | den.is_infinite()).then(None).otherwise(den)
            r = num_ok / d - 1.0
            return pl.when(d.is_null() | (d == 0) | r.is_null() | r.is_infinite()).then(None).otherwise(r)

        overnight = _safe_ret_expr(o_c, pc)
        intraday = _safe_ret_expr(c_c, o_c)
        if op == "ts_overnight_intraday_cov":
            ov = pl.when(overnight.is_not_null() & intraday.is_not_null()).then(overnight)
            iv = pl.when(overnight.is_not_null() & intraday.is_not_null()).then(intraday)
            cnt = (overnight.is_not_null() & intraday.is_not_null()).cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            cov = pl.rolling_cov(ov, iv, window_size=w, min_samples=5, ddof=1).over(_INST, order_by=_TS)
            expr = pl.when(cnt < 5).then(None).otherwise(cov)
            return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)
        if op == "ts_overnight_intraday_spread":
            # pandas: o.rolling(w).mean() - i.rolling(w).mean() — default
            # min_periods == window (NaN until w valid rows).
            expr = overnight.rolling_mean(window_size=w, min_samples=w).over(_INST, order_by=_TS) - intraday.rolling_mean(window_size=w, min_samples=w).over(_INST, order_by=_TS)
            return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)
        if op == "ts_overnight_intraday_sign_agreement":
            # pandas: (np.sign(o) == np.sign(i)).astype(float) — NaN sign is
            # NaN and NaN==NaN is False, so a NaN pair row contributes 0.0
            # (a *valid* zero) to the window mean.  polars null==null yields
            # null, so map null/NaN operand pairs explicitly to 0.0.
            o_s = pl.when(overnight.is_null() | overnight.is_nan()).then(None).otherwise(overnight.sign())
            i_s = pl.when(intraday.is_null() | intraday.is_nan()).then(None).otherwise(intraday.sign())
            agree = pl.when(
                overnight.is_null() | overnight.is_nan() | intraday.is_null() | intraday.is_nan()
            ).then(0.0).otherwise((o_s == i_s).cast(pl.Float64))
            expr = agree.rolling_mean(window_size=w, min_samples=w).over(_INST, order_by=_TS)
            return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)
        # ts_opening_mispricing_score: beta = cov(ov,iv)/var(ov) over the
        # window (pandas: np.cov / np.var over finite pairs, min 6 valid),
        # output = ov_t - beta * ov_t (expected intraday response to gap).
        ov = overnight
        iv = intraday
        mp = max(6, w // 5)
        pair_ok = ov.is_not_null() & iv.is_not_null()
        pair_cnt = pair_ok.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        cov_w = pl.rolling_cov(ov, iv, window_size=w, min_samples=1, ddof=1).over(_INST, order_by=_TS)
        # pandas reference: beta = np.cov(a,b)[0,1] (ddof=1) / np.var(a)
        # (ddof=0, biased) over finite pairs — mixed ddof is intentional there.
        var_w = (
            pl.when(pair_ok).then(ov)
            .rolling_var(window_size=w, min_samples=1, ddof=0).over(_INST, order_by=_TS)
        )
        beta = pl.when(pair_cnt < mp).then(None).otherwise(cov_w / pl.when(var_w.is_null() | (var_w <= 1e-12)).then(None).otherwise(var_w))
        expr = pl.when(ov.is_null() | beta.is_null()).then(None).otherwise(ov - beta * ov)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "cs_mad":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.stat_valid import polars_rank_input

        value = polars_rank_input(_VAL, exclude_nan=True)
        med = value.median().over(_TS, order_by=_INST)
        mad = (value - med).abs().median().over(_TS, order_by=_INST)
        return inner.with_columns(mad.alias(_VAL))

    if op == "cs_mad_zscore":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.stat_valid import polars_rank_input, polars_row_stat_invalid

        value = polars_rank_input(_VAL, exclude_nan=True)
        med = value.median().over(_TS, order_by=_INST)
        mad = (value - med).abs().median().over(_TS, order_by=_INST)
        expr = (
            pl.when(polars_row_stat_invalid(_VAL, exclude_nan=True))
            .then(None)
            .when(mad.is_null() | (mad == 0))
            .then(None)
            .otherwise((value - med) / mad)
        )
        return inner.with_columns(expr.alias(_VAL))

    if op in {"ts_delay", "delay"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return inner.with_columns(pl.col(_VAL).shift(lag).over(_INST, order_by=_TS).alias(_VAL))

    if op == "ts_delta":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        delayed = pl.col(_VAL).shift(lag).over(_INST, order_by=_TS)
        return inner.with_columns((pl.col(_VAL) - delayed).alias(_VAL))

    if op == "ts_pct":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        prev = pl.col(_VAL).shift(lag).over(_INST, order_by=_TS)
        return inner.with_columns(
            pl.when(prev.is_null() | (prev == 0))
            .then(None)
            .otherwise(pl.col(_VAL) / prev - 1.0)
            .alias(_VAL)
        )

    if op == "ffill":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(pl.col(_VAL).forward_fill().over(_INST, order_by=_TS).alias(_VAL))

    if op == "bfill":
        from factor_engine.backend.polars_long_policy import UnsupportedCausalOperatorError

        raise UnsupportedCausalOperatorError(
            "polars_long 不支持 bfill/causal_bfill（因果占位，非传统 backward fill）；请改用 ffill 或 pandas 研究路径"
        )

    if op == "causal_bfill":
        from factor_engine.backend.polars_long_policy import UnsupportedCausalOperatorError

        raise UnsupportedCausalOperatorError(
            "polars_long 不支持 bfill/causal_bfill（因果占位，非传统 backward fill）；请改用 ffill 或 pandas 研究路径"
        )

    if op == "rank":
        fused = _try_fuse_unary_over_ts(node, base)
        if fused is not None:
            return fused
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_cs_rank_01().alias(_VAL))

    if op in {"rank_pct", "cs_pct_rank"}:
        fused = _try_fuse_unary_over_ts(node, base)
        if fused is not None:
            return fused
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_cs_rank_pct().alias(_VAL))

    if op == "zscore":
        fused = _try_fuse_unary_over_ts(node, base)
        if fused is not None:
            return fused
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.backend.numeric_semantics import std_ddof_value, zscore_zero_std_fill

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
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        mean = pl.col(_VAL).mean().over(_TS, order_by=_INST)
        return inner.with_columns((pl.col(_VAL) - mean).alias(_VAL))

    if op == "normalize":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_normalize_on(_VAL).alias(_VAL))

    if op == "scale":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        to_val = _scale_to_value(node)
        from factor_engine.backend.cross_section_spec import polars_scale_expr

        return inner.with_columns(
            polars_scale_expr(_VAL, to_val, partition_cols=(_TS,), order_by=_INST).alias(_VAL)
        )

    if op in {"log_returns", "ts_log_return"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        d = _window_int(node, default=1)
        prev = pl.col(_VAL).shift(d).over(_INST, order_by=_TS)
        ratio = pl.col(_VAL) / prev
        # pandas reference: ``np.log(ratio.replace([inf,-inf], nan))`` — an Inf
        # ratio (Inf price, or an Inf prev) is censored to missing BEFORE log;
        # a residual -Inf output (log of a ~0 ratio) is NaN at the engine level.
        from factor_engine.backend.inf_sanitize import apply_inf_policy_polars_fast

        return inner.with_columns(
            apply_inf_policy_polars_fast(
                pl.when(
                    pl.col(_VAL).is_null()
                    | prev.is_null()
                    | (pl.col(_VAL) <= 0)
                    | (prev <= 0)
                    | ratio.is_infinite()
                )
                .then(None)
                .otherwise(ratio.log()),
                "log_returns",
            ).alias(_VAL)
        )

    if op == "cs_physical_panel_coverage":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        # Reference: fraction of PHYSICALLY-present panel rows that are finite,
        # per date, broadcast to ALL cells of the date (including the NaN cell).
        cnt_total = pl.col(_VAL).len().over(_TS, order_by=_INST).cast(pl.Float64)
        cnt_finite = pl.col(_VAL).count().over(_TS, order_by=_INST).cast(pl.Float64)
        return inner.with_columns(
            pl.when(cnt_total > 0).then(cnt_finite / cnt_total).otherwise(None).alias(_VAL)
        )

    if op == "cs_universe_coverage":
        # cs_state_ops._cs_universe_coverage: finite-x fraction over the
        # DECLARED universe (universe non-null and nonzero), per date,
        # broadcast to ALL cells; empty universe -> NaN.
        if len(node.inputs) < 2:
            return None
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        uni = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if uni is None:
            return None
        joined = inner.join(uni.rename({_VAL: "_u"}), on=[_TS, _INST], how="left")
        from factor_engine.backend.stat_valid import polars_rank_input

        u = pl.col("_u")
        in_universe = u.is_not_null() & ~u.is_nan() & (u != 0)
        # A physically-present row with a null x still occupies a universe
        # slot (u is present); the NaN-cell broadcast rows (missing from the
        # panel) never appear in the long frame, matching the reference
        # denominator of "in-universe names present in the panel".
        cnt_univ = (
            pl.when(u.is_not_null() & ~u.is_nan() & (u != 0)).then(1)
            .otherwise(0)
            .sum()
            .over(_TS, order_by=_INST)
            .cast(pl.Float64)
        )
        xv = polars_rank_input(_VAL, exclude_nan=True)
        # Finite count restricted to in-universe rows (reference:
        # np.isfinite(xa[r])[u_mask].sum()).
        cnt_finite_in = (
            pl.when(in_universe & xv.is_not_null()).then(1)
            .otherwise(0)
            .sum()
            .over(_TS, order_by=_INST)
        )
        return joined.with_columns(
            pl.when(cnt_univ > 0).then(cnt_finite_in / cnt_univ).otherwise(None).alias(_VAL)
        )

    if op == "price_spread_deviation":
        # NumpyKernels.price_spread_deviation_: x_t / trailing-mean(x, d) - 1
        # where the trailing mean is over FINITE values (nanmean), and a
        # zero/NaN mean -> NaN (np errstate → the raw NaN propagates).
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        d = _window_int(node, default=20)
        xv = pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()).then(None).otherwise(pl.col(_VAL))
        m = (
            pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()).then(None).otherwise(pl.col(_VAL))
            .rolling_mean(window_size=d, min_samples=1)
            .over(_INST, order_by=_TS)
        )
        denom = pl.when(m.is_null() | (m == 0)).then(None).otherwise(m)
        expr = pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan()).then(None).otherwise(
            pl.when(denom.is_null() | (denom == 0)).then(None).otherwise(pl.col(_VAL) / denom - 1.0)
        )
        return inner.with_columns(expr.alias(_VAL))

    if op == "volatility":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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

    cs_aggregates = {
        "c_mean": "mean", "cs_mean": "mean",
        "c_std": "std", "cs_std": "std",
        "c_sum": "sum", "cs_sum": "sum",
        "c_count": "count", "cs_count": "count",
    }
    if op in cs_aggregates:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        # R19-027..029: sample_validity=finite — ±Inf excluded like pandas
        # CrossSectionSampleMask.
        from factor_engine.backend.stat_valid import polars_rank_input, polars_row_stat_invalid

        value = polars_rank_input(_VAL, exclude_nan=True)
        valid = value.count().over(_TS, order_by=_INST)
        kind = cs_aggregates[op]
        if kind == "mean":
            raw = value.mean().over(_TS, order_by=_INST)
            expr = pl.when(valid == 0).then(None).otherwise(raw)
        elif kind == "std":
            raw = value.std(ddof=1).over(_TS, order_by=_INST)
            expr = pl.when(valid == 0).then(None).otherwise(raw)
        elif kind == "sum":
            raw = value.sum().over(_TS, order_by=_INST)
            expr = pl.when(valid == 0).then(None).otherwise(raw)
        else:
            expr = valid.cast(pl.Float64)
        return inner.with_columns(expr.alias(_VAL))

    if op == "log_abs":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        # PARITY-SWEEP-R56: pandas ``LogAbs`` = ``np.log(x.abs()).replace([inf,-inf],
        # nan)`` — log(abs(±Inf)) = Inf is replaced with NaN.  Polars' ``abs().log()``
        # keeps Inf; mask the non-finite log output to None.
        abs_v = pl.col(_VAL).abs()
        log_v = abs_v.log()
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null() | (abs_v == 0))
            .then(None)
            .otherwise(
                pl.when(log_v.is_infinite() | log_v.is_nan()).then(None).otherwise(log_v)
            )
            .alias(_VAL)
        )

    if op == "signed_log":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            (pl.col(_VAL).sign() * (pl.col(_VAL).abs() + 1e-10).log()).alias(_VAL)
        )

    if op == "signed_sqrt":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(
            (pl.col(_VAL).sign() * pl.col(_VAL).abs().sqrt()).alias(_VAL)
        )

    if op in {"ts_decay_linear", "decay_linear"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_linear_decay_expr(w).alias(_VAL))

    if op == "ts_mad":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        spec = _window_spec(node)
        mp = spec.min_periods if node.attrs.get("min_periods") is not None else spec.size
        scale = _float_attr(node, "scale", default=1.0)
        return inner.with_columns(
            _rolling_median_abs_dev_expr(
                spec.size,
                min_periods=mp,
                scale=scale,
            ).alias(_VAL)
        )

    if op == "ts_quantile":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        p = _float_attr(node, "q", "p", default=0.5)
        pos_p = _literal_value(node, 1)
        if pos_p is not None:
            p = pos_p
        return inner.with_columns(_rolling_quantile_expr(w, p).alias(_VAL))

    if op == "ts_product":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        spec = _window_spec(node)
        mp = spec.min_periods if node.attrs.get("min_periods") is not None else spec.size
        return inner.with_columns(
            _rolling_product_expr(spec.size, min_periods=mp).alias(_VAL)
        )

    if op == "ts_median_abs_deviation":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_median_abs_dev_expr(w).alias(_VAL))

    if op == "ts_mean_abs_deviation":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_mean_abs_dev_expr(w).alias(_VAL))

    if op == "ts_skew":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_skew_expr(w).alias(_VAL))

    if op == "ts_argmax":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_argext_expr(w, pick="max").alias(_VAL))

    if op == "ts_argmin":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_argext_expr(w, pick="min").alias(_VAL))

    if op in {"Slope", "ts_time_slope"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_time_slope_expr(w).alias(_VAL))

    if op == "ts_regression_slope":
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        x_layer = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if y_layer is None or x_layer is None:
            return None
        w = max(_window_int(node), 3)
        joined = y_layer.join(x_layer.rename({_VAL: "_x"}), on=[_TS, _INST], how="left")
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
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = max(_window_int(node), 2)
        mp = _ts_sharpe_min_periods(w)
        ann = _float_attr(node, "ann_factor", default=252.0)
        sqrt_af = ann**0.5
        mean = pl.col(_VAL).rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        std = pl.col(_VAL).rolling_std(window_size=w, min_samples=mp, ddof=1).over(_INST, order_by=_TS)
        from factor_engine.backend.numeric_semantics import ts_sharpe_zero_std_is_null

        expr = (
            pl.when(std.is_null() | (std == 0))
            .then(None if ts_sharpe_zero_std_is_null() else 0.0)
            .otherwise(mean / std)
            * sqrt_af
        )
        from factor_engine.backend.inf_sanitize import apply_inf_policy_polars_fast

        return inner.with_columns(apply_inf_policy_polars_fast(expr, "ts_sharpe").alias(_VAL))

    if op == "ts_autocorr":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_expanding_sum_expr().alias(_VAL))

    if op == "expanding_mean":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_expanding_mean_expr().alias(_VAL))

    if op == "expanding_std":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return _expanding_std_map_groups(inner)

    if op == "count":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        return inner.with_columns(_expanding_non_null_count().alias(_VAL))

    if op in {"ewm_corr", "ewm_cov", "ts_ewm_corr", "ts_ewm_cov"}:
        if len(node.inputs) < 2:
            return None
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        span = max(_window_int(node, default=20), 2)
        joined = _join_binary(left, right)
        corr = op in {"ewm_corr", "ts_ewm_corr"}
        return _ewm_binary_map_groups(joined, span, corr=corr)

    if op == "ts_ratio":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        d = _window_int(node, default=1)
        prev = pl.col(_VAL).shift(d).over(_INST, order_by=_TS)
        return inner.with_columns(
            pl.when(prev.is_null() | (prev == 0))
            .then(None)
            .otherwise(pl.col(_VAL) / prev)
            .alias(_VAL)
        )

    if op == "ts_kurt":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        if w < 4:
            raise ValueError("ts_kurt window must be >= 4")

        def _kurt(arr: np.ndarray) -> float:
            # Match the active StableTsKurt authority: full finite windows,
            # unbiased Fisher excess kurtosis, and NaN for zero variance.
            values = np.asarray(arr, dtype=np.float64)
            if len(values) < w or not np.isfinite(values).all():
                return np.nan
            count = len(values)
            centered = values - float(np.mean(values))
            second = float(np.sum(centered * centered))
            if second <= 0.0:
                return np.nan
            fourth = float(np.sum(centered ** 4))
            biased_excess = count * fourth / (second * second) - 3.0
            return float(
                (count - 1)
                / ((count - 2) * (count - 3))
                * ((count + 1) * biased_excess + 6.0)
            )

        return inner.with_columns(
            pl.col(_VAL).rolling_map(_kurt, window_size=w, min_samples=w).over(_INST, order_by=_TS).alias(_VAL)
        )

    if op == "ts_moment":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = max(_int_attr(node, "d", "window", input_index=0, default=3), 1)
        k = max(_int_attr(node, "k", "order", input_index=1, default=2), 1)
        from factor_engine.cleaned_operators._numpy_kernels import ts_moment_

        return _unary_inst_map_groups(inner, lambda arr: ts_moment_(arr, w, k))

    if op == "ts_max_buildup":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)

        def _buildup(arr: np.ndarray) -> np.ndarray:
            # Production per-window record-high count (production_repairs):
            # each trailing window resets its record maximum, so appending future
            # rows never alters past outputs.  The legacy ts_max_buildup_ kernel
            # accumulated over the whole series and is not PIT-safe.
            out = np.full(arr.shape, np.nan, dtype=float)
            for end in range(arr.shape[0]):
                start = max(0, end - w + 1)
                current_max = -np.inf
                count = 0
                for value in arr[start : end + 1]:
                    if not np.isfinite(value):
                        continue
                    if value >= current_max:
                        current_max = value
                        count += 1
                if count > 0:
                    out[end] = float(count)
            return out

        return _unary_inst_map_groups(inner, _buildup)

    if op == "expanding_rank":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None

        def _expanding_rank(arr: np.ndarray) -> np.ndarray:
            ranked = pd.Series(arr, dtype=float).expanding(min_periods=1).rank(pct=True)
            return ranked.to_numpy(dtype=float)

        return _unary_inst_map_groups(inner, _expanding_rank)

    if op == "causal_linear_extrapolate":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from factor_engine.cleaned_operators._causal import causal_linear_extrapolate_panel

        def _extrapolate(arr: np.ndarray) -> np.ndarray:
            pdf = pd.DataFrame({"v": arr.astype(float, copy=False)})
            return causal_linear_extrapolate_panel(pdf)["v"].to_numpy(dtype=float)

        return _unary_inst_map_groups(inner, _extrapolate)

    if op == "quantile":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
        high = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        low = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        close = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
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

    # ------------------------------------------------------------------
    # Tech / candle / misc family: pure-numerical native branches.
    # pandas reference semantics replicated exactly; every window ends at t
    # (inclusive trailing), all windows min_periods=window unless noted.
    # ------------------------------------------------------------------

    if op == "ALMA":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = max(int(_window_int(node, default=10)), 2)
        offset = float(_literal_value(node, 1, default=0.85) or 0.85)
        sigma = float(_literal_value(node, 2, default=6.0) or 6.0)
        sigma = max(sigma, 1e-6)
        m = offset * (w - 1.0)
        s = w / sigma
        idx = np.arange(w, dtype=np.float64)
        weights = np.exp(-((idx - m) ** 2) / (2.0 * s * s))
        total = weights.sum()
        if total <= 1e-12:
            weights = np.ones(w, dtype=np.float64) / w
        else:
            weights = weights / total
        # pandas ALMA: `rolling(window=w, min_periods=w).apply(dot(vals, wg))`;
        # any NaN slot makes the whole window NaN, warmup = w rows.  We build it
        # with explicit shifted weighted sums (no weights kernel) so null
        # propagates exactly like the pandas reference.
        weighted = sum(
            (pl.col(_VAL).shift(i).over(_INST, order_by=_TS)) * float(weights[i])
            for i in range(w)
        )
        expr = pl.when(weighted.is_nan()).then(None).otherwise(weighted)
        return inner.with_columns(expr.alias(_VAL))

    if op == "CoppockCurve":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        roc1 = max(int(_literal_value(node, 1, default=14) or 14), 1)
        roc2 = max(int(_literal_value(node, 2, default=11) or 11), 1)
        wma = max(int(_literal_value(node, 3, default=10) or 10), 1)
        roc_mode = str(node.attrs.get("roc_mode") or node.attrs.get("mode") or "pct")
        c = pl.col(_VAL)
        sh1 = c.shift(roc1).over(_INST, order_by=_TS)
        sh2 = c.shift(roc2).over(_INST, order_by=_TS)
        if roc_mode == "log":
            r1 = pl.when(sh1.is_null() | (sh1 <= 0)).then(None).otherwise((c / sh1).log())
            r2 = pl.when(sh2.is_null() | (sh2 <= 0)).then(None).otherwise((c / sh2).log())
        else:  # pct
            r1 = pl.when(sh1.is_null()).then(None).otherwise(c / sh1 - 1.0)
            r2 = pl.when(sh2.is_null()).then(None).otherwise(c / sh2 - 1.0)
        total = r1 + r2
        if wma <= 1:
            expr = total
        else:
            wg = np.arange(1, wma + 1, dtype=np.float64)
            wg = wg / wg.sum()
            # WMA = sum(shift(i) * wg[i]); NaN total slot propagates the window
            # to NaN exactly like pandas rolling.apply (null → NaN, not 0).
            weighted = sum(
                (total.shift(i).over(_INST, order_by=_TS)) * float(wg[i])
                for i in range(wma)
            )
            expr = weighted
        return inner.with_columns(expr.alias(_VAL))

    if op == "ElderRay":
        if len(node.inputs) < 2:
            return None
        h_in = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        l_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        close_in = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        if h_in is None or l_in is None or close_in is None:
            return None
        joined = _join_triple(h_in, l_in, close_in)
        ema = int(max(_literal_value(node, 3, default=13) or 13, 1))
        out = str(node.attrs.get("output") or "bull")
        ev = (
            pl.when(pl.col("_y").is_nan())
            .then(None)
            .otherwise(pl.col("_y"))
            .ewm_mean(alpha=2.0 / (ema + 1.0), adjust=False, min_periods=ema, ignore_nulls=False)
            .over(_INST, order_by=_TS)
        )
        if out == "bull":
            expr = pl.col(_VAL) - ev
        elif out == "bear":
            expr = pl.col("_ym") - ev
        else:  # spread
            expr = pl.col(_VAL) - pl.col("_ym")
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "FisherTransform":
        if len(node.inputs) < 2:
            return None
        h_in = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        l_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if h_in is None or l_in is None:
            return None
        w = max(int(_window_int(node, default=9)), 2)
        smooth = float(_literal_value(node, 2, default=0.33) or 0.33)
        sig_smooth = float(_literal_value(node, 3, default=0.5) or 0.5)
        out = str(node.attrs.get("output") or "value")
        joined = _join_binary(h_in, l_in)
        source = (pl.col(_VAL) + pl.col("_y")) / 2.0
        roll_min = source.rolling_min(window_size=w, min_samples=w).over(_INST, order_by=_TS)
        roll_max = source.rolling_max(window_size=w, min_samples=w).over(_INST, order_by=_TS)
        rng = roll_max - roll_min
        raw = pl.when(roll_min.is_null() | roll_max.is_null()).then(None).otherwise(
            2.0 * ((source - roll_min) / rng - 0.5)
        )
        raw = pl.when(rng.is_not_null() & (rng <= 1e-12)).then(0.0).otherwise(raw)
        z = (
            pl.when(raw.is_null())
            .then(None)
            .otherwise(
                raw.ewm_mean(alpha=smooth, adjust=False, min_samples=1, ignore_nulls=False)
            )
        )
        z = z.clip(-0.999, 0.999)
        one_m = 1.0 - z
        fisher = 0.5 * (pl.when(one_m <= 0).then(None).otherwise(((1.0 + z) / one_m).log()))
        sig = (
            fisher.ewm_mean(alpha=sig_smooth, adjust=False, min_samples=1, ignore_nulls=False)
            .over(_INST, order_by=_TS)
            .shift(1)
            .over(_INST, order_by=_TS)
        )
        if out == "signal":
            expr = sig
        elif out == "trigger":
            expr = fisher - sig
        else:
            expr = fisher
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op in {"atr_pct", "atr_acceleration", "atr_short_long_ratio"}:
        if len(node.inputs) < 3:
            return None
        joined = _compile_ohlc3(node, base, op, ctx=ctx, memo=memo)
        if joined is None:
            return None
        if op == "atr_short_long_ratio":
            s = max(int(_literal_value(node, 2, default=5) or 5), 2)
            l = max(int(_literal_value(node, 3, default=14) or 14), 2)
            if s >= l:
                return None
            trs = _atr_wilder_expr(_VAL, "_ym", "_y", window=s, alpha=1.0 / s).over(_INST, order_by=_TS)
            trl = _atr_wilder_expr(_VAL, "_ym", "_y", window=l, alpha=1.0 / l).over(_INST, order_by=_TS)
            closep = pl.when(pl.col("_y").is_null() | (pl.col("_y") <= 0)).then(None).otherwise(pl.col("_y"))
            ratio_s = trs / closep
            ratio_l = trl / closep
            denom = pl.when(ratio_l == 0).then(None).otherwise(ratio_l)
            expr = ratio_s / denom
            return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)
        # atr_pct / atr_acceleration
        wl = max(int(_literal_value(node, 3, default=14) or 14), 2)
        tr = _atr_wilder_expr(_VAL, "_ym", "_y", window=wl, alpha=1.0 / wl).over(
            _INST, order_by=_TS
        )
        atr = pl.when(pl.col("_y").is_null() | (pl.col("_y") <= 0)).then(None).otherwise(tr / pl.col("_y"))
        if op == "atr_acceleration":
            atr = atr.diff().over(_INST, order_by=_TS)
        return joined.with_columns(atr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op in {"candle_body_strength", "candle_wick_balance", "candle_range_pct",
              "candle_pattern_count"}:
        if len(node.inputs) < 4:
            return None
        w = max(int(_window_int(node, default=4)), 2)
        joined = _compile_ohlc4(node, base, op, ctx=ctx, memo=memo)
        if joined is None:
            return None
        o, h, l, c = pl.col(_VAL), pl.col("_b"), pl.col("_c"), pl.col("_d")
        maxoc = pl.max_horizontal(o, c)
        minoc = pl.min_horizontal(o, c)
        valid = (
            o.is_not_null() & h.is_not_null() & l.is_not_null() & c.is_not_null()
            & (c > 0.0) & (h > l) & (h >= maxoc) & (l <= minoc)
        )
        rng = pl.when(valid).then(h - l).otherwise(None)
        upper = pl.when(valid).then(h - maxoc).otherwise(None)
        lower = pl.when(valid).then(minoc - l).otherwise(None)
        body = pl.when(valid).then(c - o).otherwise(None)
        absbody = pl.when(valid).then((c - o).abs()).otherwise(None)
        if op == "candle_pattern_count":
            with np.errstate(divide="ignore", invalid="ignore"):
                body_r = absbody / rng
                up_r = upper / rng
                lo_r = lower / rng
            doji = valid & (body_r < 0.1)
            dom_upper = valid & (up_r >= 0.7) & (lo_r <= 0.15)
            dom_lower = valid & (lo_r >= 0.7) & (up_r <= 0.15)
            pattern = (doji | dom_upper | dom_lower).cast(pl.Float64)
            num = pattern.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            den = valid.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            rows = (o.is_not_null() & h.is_not_null() & l.is_not_null() & c.is_not_null())
            rowcnt = rows.cast(pl.Float64).rolling_sum(window_size=w, min_samples=w).over(_INST, order_by=_TS)
            expr = pl.when(den.is_null() | (den <= 0)).then(None).otherwise(num / den)
            expr = pl.when(rowcnt < float(w)).then(None).otherwise(expr)
        else:
            if op == "candle_body_strength":
                geom = pl.when(body.is_null()).then(None).otherwise(body / rng)
            elif op == "candle_wick_balance":
                geom = pl.when(lower.is_null() | upper.is_null()).then(None).otherwise((lower - upper) / rng)
            else:  # candle_range_pct
                geom = pl.when(rng.is_null() | c.is_null() | (c <= 0)).then(None).otherwise(rng / c)
            expr = geom.rolling_mean(window_size=w, min_samples=w).over(_INST, order_by=_TS)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    # =====================================================================
    # ts_* quantile / tail / scale family — pure native polars Expr branches.
    # pandas long-table reference semantics replicated exactly: trailing
    # windows ending at t, linear-interpolation quantiles (np.quantile /
    # pandas Series.quantile == polars rolling_quantile interpolation='linear'
    # resolved via the two nearest ranked observations), and the per-operator
    # min_periods floor. Every window is bounded by .over(_INST, order_by=_TS).
    # =====================================================================

    # window=... bounded helper:  same calling convention as ts_mean etc.
    def _ival(node, idx, default):
        v = _literal_value(node, idx)
        return int(v) if v is not None else default

    if op == "ts_quantile_range":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        lo = float(_literal_value(node, 1) if _literal_value(node, 1) is not None else _float_attr(node, "lower", "q_low", default=0.25))
        hi = float(_literal_value(node, 2) if _literal_value(node, 2) is not None else _float_attr(node, "upper", "q_high", default=0.75))
        qhi = pl.col(_VAL).rolling_quantile(quantile=hi, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        qlo = pl.col(_VAL).rolling_quantile(quantile=lo, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        return inner.with_columns((qhi - qlo).alias(_VAL))

    if op == "ts_quantile_skew":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        ql = pl.col(_VAL).rolling_quantile(quantile=0.25, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        qm = pl.col(_VAL).rolling_quantile(quantile=0.50, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        qh = pl.col(_VAL).rolling_quantile(quantile=0.75, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        iqr = qh - ql
        expr = pl.when(iqr.is_null() | (iqr == 0)).then(None).otherwise((qh + ql - 2.0 * qm) / iqr)
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_quantile_kurtosis":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=60)
        q_a = pl.col(_VAL).rolling_quantile(quantile=0.125, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        q_b = pl.col(_VAL).rolling_quantile(quantile=0.875, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        q25 = pl.col(_VAL).rolling_quantile(quantile=0.25, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        q75 = pl.col(_VAL).rolling_quantile(quantile=0.75, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        iqr = q75 - q25
        expr = pl.when(iqr.is_null() | (iqr == 0)).then(None).otherwise((q_b - q_a) / iqr)
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_tail_ratio":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=60)
        qlo = pl.col(_VAL).rolling_quantile(quantile=0.05, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        qhi = pl.col(_VAL).rolling_quantile(quantile=0.95, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        expr = pl.when(qlo.is_null() | (qlo == 0)).then(None).otherwise(qhi.abs() / qlo.abs())
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_tail_imbalance":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=60)
        mp = max(4, int(_ival(node, 1, 8)))
        if mp > w:
            mp = w
        qlo = pl.col(_VAL).rolling_quantile(quantile=0.05, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        qhi = pl.col(_VAL).rolling_quantile(quantile=0.95, window_size=w, interpolation="linear").over(_INST, order_by=_TS)
        rng = qhi - qlo
        expr = pl.when(qlo.is_null() | rng.is_null() | (rng == 0)).then(None).otherwise((qhi + qlo) / rng)
        # min_periods (>=4) cannot be expressed with the fixed rolling_quantile
        # default; use an explicit count mask to fail-close before mp rows.
        cnt = pl.col(_VAL).is_not_null().cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        expr = pl.when(cnt.is_null() | (cnt < mp)).then(None).otherwise(expr)
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_scale_shift":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        short = _window_int(node, default=20)
        longw = int(_literal_value(node, 1) if _literal_value(node, 1) is not None else 40)
        s = pl.col(_VAL).rolling_std(short, min_samples=short).over(_INST, order_by=_TS)
        l = pl.col(_VAL).rolling_std(longw, min_samples=longw).over(_INST, order_by=_TS)
        expr = pl.when(s.is_null() | l.is_null() | (l == 0)).then(None).otherwise(s / l)
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_median3_causal":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        expr = pl.col(_VAL).rolling_median(window_size=3, min_samples=1).over(_INST, order_by=_TS)
        return inner.with_columns(expr.alias(_VAL))

    if op == "price_spread_deviation":
        # NumpyKernels.price_spread_deviation_: x_t / trailing-mean(x, d) - 1
        # where the trailing mean is over FINITE values (nanmean), and a
        # zero/NaN mean -> NaN (np errstate → the raw NaN propagates).
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        d = _window_int(node, default=20)
        xv = pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()).then(None).otherwise(pl.col(_VAL))
        m = (
            pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()).then(None).otherwise(pl.col(_VAL))
            .rolling_mean(window_size=d, min_samples=1)
            .over(_INST, order_by=_TS)
        )
        denom = pl.when(m.is_null() | (m == 0)).then(None).otherwise(m)
        expr = pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan()).then(None).otherwise(
            pl.when(denom.is_null() | (denom == 0)).then(None).otherwise(pl.col(_VAL) / denom - 1.0)
        )
        return inner.with_columns(expr.alias(_VAL))

    if op == "ts_rolling_median_causal":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=5)
        # pandas rolling median: NaN until the window has enough valid rows
        # (default min_periods == window on the clean series).
        expr = pl.col(_VAL).rolling_median(window_size=w, min_samples=w).over(_INST, order_by=_TS)
        return inner.with_columns(expr.alias(_VAL))

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
    """构建 long-table 基础 LazyFrame：优先 scan 引用列，否则从缓存 Series 推断骨架。

    参数:
        plan: 待编译的逻辑计划。
        ctx: 执行上下文（含 materialized / shared 缓存）。
        scan_fn: ``data_source.scan_polars_long`` 等列扫描回调。

    返回:
        含 ts/inst 及引用列的宽表 LazyFrame。
    """
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
    """无 scan_polars_long 时，逐列 load_column 并 merge 为宽表 LazyFrame。"""
    if pl is None:
        raise ImportError("polars is required for polars long-table backend")
    from factor_engine.storage.factor_format import series_to_long_table

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
    from factor_engine.backend.long_frame import long_table_to_polars_lazy

    return long_table_to_polars_lazy(renamed, float_cols=sorted(columns))


def compile_plan_to_polars(
    plan: PlanNode,
    base_lf: pl.LazyFrame,
    *,
    ctx: Any | None = None,
    ts_col: str = _TS,
    inst_col: str = _INST,
) -> LongFrameResult | None:
    """编译 PlanNode 为 long-table Polars LazyFrame + 结果列名。

    参数:
        plan: 逻辑计划根节点。
        base_lf: 宽表 base LazyFrame（含 ts/inst 及引用列）。
        ctx: 可选执行上下文（materialized_series / plan_ref 等）。
        ts_col: 时间戳列名，默认 ``ts``。
        inst_col: 标的列名，默认 ``inst``。

    返回:
        ``LongFrameResult``；polars 不可用或编译失败时为 None。
    """
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
    """编译 long-table 计划为 LazyFrame（不 collect）。

    参数:
        plan: 逻辑计划根节点。
        ctx: 执行上下文。
        base_lf: 可选预构建 base；为 None 时自动 scan 或 merge 列。
        lazy_cache_key: 非空时将结果 LazyFrame 写入 shared_long_lazy_cache。

    返回:
        含 frame/value_col/ts_col/inst_col 的 ``LongFrameResult``。
    """
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
    """编译并执行 long-table 计划；仅最终 collect 一次。

    参数:
        plan: 逻辑计划根节点。
        ctx: 执行上下文。
        base_lf: 可选预构建 base LazyFrame。
        lazy_cache_key: 可选 shared long lazy 缓存键。

    返回:
        MultiIndex (timestamp, instrument) 的 ``pd.Series`` 因子结果。
    """
    compiled = compile_polars_long_lazy(
        plan, ctx, base_lf=base_lf, lazy_cache_key=lazy_cache_key
    )
    lf = compiled.frame.sort([compiled.ts_col, compiled.inst_col]).select(
        pl.col(compiled.ts_col).alias(ctx.timestamp_col),
        pl.col(compiled.inst_col).alias(ctx.instrument_col),
        pl.col(compiled.value_col).alias("value"),
    )
    # #收官轮 P0：polars-long 快路径的**受控 collect 终端**——不再裸 ``.collect()``
    # 绕过 DataAccess 的 governed terminal。collect 前对数据源做快照 revalidation
    # （production fail-closed：plan→collect 之间源数据被替换 = 执行内容 ≠ 计划
    # 快照），collect 后强制 QueryBudget（deadline/rows）。
    ds = getattr(ctx, "data_source", None)
    revalidate = getattr(ds, "revalidate_for_long_collect", None)
    if callable(revalidate):
        revalidate()
    # QueryBudget is optional when DataAccess is not installed (in-memory /
    # unit-test paths). Production DA sources still enforce the budget.
    try:
        from data_access.read.query_budget import (
            enforce_arrow_budget,
            resolve_query_budget,
        )
    except ImportError:  # pragma: no cover - local FE-only test envs
        return polars_long_to_multiindex_series(
            lf.collect(),
            timestamp_col=ctx.timestamp_col,
            instrument_col=ctx.instrument_col,
            value_col="value",
            template_index=optional_universe_index(ctx),
        ).sort_index()

    import time as _t

    _budget = resolve_query_budget(None)
    _start = _t.perf_counter()
    frame = lf.collect()
    enforce_arrow_budget(
        _budget,
        frame.to_arrow(),
        elapsed_ms=(_t.perf_counter() - _start) * 1000,
    )
    return polars_long_to_multiindex_series(
        frame,
        timestamp_col=ctx.timestamp_col,
        instrument_col=ctx.instrument_col,
        value_col="value",
        template_index=optional_universe_index(ctx),
    ).sort_index()


def execute_polars_expr_plan(plan: PlanNode, ctx: Any) -> pd.Series:
    """兼容旧名：``PolarsBackend`` + ``FACTOR_ENGINE_POLARS_EXPR=1``。

    参数:
        plan: 逻辑计划根节点。
        ctx: 执行上下文。

    返回:
        与 ``execute_polars_long_plan`` 相同的 MultiIndex Series。
    """
    return execute_polars_long_plan(plan, ctx)
