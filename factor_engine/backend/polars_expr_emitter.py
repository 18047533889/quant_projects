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
        arr = np.asarray(arr, dtype=np.float64)
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


def _trailing_contiguous_np(chunk: np.ndarray) -> np.ndarray:
    """窗口内从尾部往前的连续有限段（遇 NaN/Inf 断开，绝不跨缺口重连）。

    与 pandas 参考 ``_trailing_contiguous``（alpha_language_shape / regression_models
    / stateful.drawdown_path）逐点一致：当前行缺失 → 空段 → 调用方输出 NaN。
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    n = chunk.size
    if n == 0 or not np.isfinite(chunk[-1]):
        return chunk[:0]
    end = n
    while end > 0 and np.isfinite(chunk[end - 1]):
        end -= 1
    return chunk[end:]


def _rolling_monotonicity_expr(w: int, *, min_periods: int) -> pl.Expr:
    """Kendall 式单调性 (C-D)/(C+D)：窗口内有限值两两方向一致比例。

    pandas 参考 ``TsMonotonicity``：``_finite`` 压缩（跳过 NaN 后两两配对），
    值差为 0 的对跳过；n < min_periods 或 C+D == 0 → NaN。
    """

    def _fn(arr: np.ndarray) -> float:
        v = np.asarray(arr, dtype=np.float64)
        v = v[np.isfinite(v)]
        n = v.size
        if n < min_periods:
            return np.nan
        c = 0
        d = 0
        for i in range(n):
            for j in range(i + 1, n):
                val_diff = v[j] - v[i]
                if val_diff == 0.0:
                    continue
                if (val_diff > 0) == (j > i):
                    c += 1
                else:
                    d += 1
        total = c + d
        if total == 0:
            return np.nan
        return float((c - d) / total)

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_turning_point_ratio_expr(w: int) -> pl.Expr:
    """方向反转比例：尾部连续有限段 diff 的符号反转次数 / (len(diff)-1)。

    pandas 参考 ``ts_model.complexity._turning_point_ratio``（P1-91 物理时间轴，
    trailing-contiguous，不压缩缺口）；len(finite) < 4 → NaN。
    """

    def _fn(arr: np.ndarray) -> float:
        seg = _trailing_contiguous_np(arr)
        if seg.size < 4:
            return np.nan
        d = np.diff(seg)
        turns = 0
        for i in range(1, d.size):
            if d[i] * d[i - 1] < 0:
                turns += 1
        return float(turns / (d.size - 1))

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _ols_fit_np(y: np.ndarray) -> tuple[float, float, float]:
    """(slope, intercept, 残差 std ddof=0)，y ~ a + b*j（与 pandas 参考一致）。"""
    n = y.size
    x = np.arange(n, dtype=float)
    sx = float(x.sum())
    sy = float(y.sum())
    denom = n * float((x * x).sum()) - sx * sx
    b = (n * float((x * y).sum()) - sx * sy) / denom
    a = (sy - b * sx) / n
    resid = y - (a + b * x)
    return float(b), float(a), float(np.std(resid))


_EPS_TINY = 1e-12


def _rolling_endpoint_deviation_expr(w: int, *, min_periods: int) -> pl.Expr:
    """端点偏离：(x_t - OLS 预测) / 残差 std（尾部连续段，不压缩缺口）。

    pandas 参考 ``TsEndpointDeviation``（P1-I-131）；sigma < eps → 0/NaN 二分。
    """

    def _fn(arr: np.ndarray) -> float:
        v = _trailing_contiguous_np(arr)
        n = v.size
        if n < min_periods:
            return np.nan
        b, a, sigma = _ols_fit_np(v)
        x_hat_last = a + b * float(n - 1)
        num = float(v[-1]) - x_hat_last
        if sigma < _EPS_TINY:
            return 0.0 if abs(num) < _EPS_TINY else np.nan
        return num / sigma

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _trailing_run_halves_np(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按 WINDOW 物理中点切分尾部连续段（pandas 参考逐点一致）。

    ``first`` 为落在窗口前物理半段内的行，``second`` 为其余；空半段表示该
    侧状态不可观测，调用方输出 NaN（fail-closed）。
    """
    vals = _trailing_contiguous_np(segment)
    if vals.size == 0:
        return vals[:0], vals[:0]
    n = segment.shape[0]
    gap = n - vals.size
    mid = n // 2
    first_len = max(0, min(mid, n) - gap)
    first_len = min(first_len, vals.size)
    return vals[:first_len], vals[first_len:]


def _rolling_vol_shift_score_expr(w: int, *, min_periods: int) -> pl.Expr:
    """波动率位移得分：log(后半 std / 前半 std)（物理中点切分，无压缩）。

    pandas 参考 ``TsVolShiftScore``；任一半 < 2 个观测或任一 std <= 0 → NaN。
    """

    def _fn(arr: np.ndarray) -> float:
        segment = np.asarray(arr, dtype=np.float64)
        vals = _trailing_contiguous_np(segment)
        if vals.size < min_periods:
            return np.nan
        first, second = _trailing_run_halves_np(segment)
        if first.size < 2 or second.size < 2:
            return np.nan
        sd1 = float(np.std(first))
        sd2 = float(np.std(second))
        if sd1 <= 0.0 or sd2 <= 0.0:
            return np.nan
        return float(np.log(sd2 / sd1))

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _running_peak_no_carry_np(chunk: np.ndarray) -> np.ndarray:
    """窗口 running peak：在每个缺失行处重置为 -inf（P1-07/#160 缺口策略）。"""
    chunk = np.asarray(chunk, dtype=np.float64)
    running_peak = np.full(chunk.size, np.nan)
    peak = -np.inf
    for k in range(chunk.size):
        if not np.isfinite(chunk[k]):
            peak = -np.inf
            continue
        if chunk[k] > peak:
            peak = chunk[k]
        running_peak[k] = peak
    return running_peak


def _rolling_time_under_water_expr(w: int) -> pl.Expr:
    """低于此前运行最高价的行比例（当前行缺失 → NaN；峰值不跨缺口）。

    pandas 参考 ``TsTimeUnderWater``（R5 P1-36(b) / R11 #160）。
    """

    def _fn(arr: np.ndarray) -> float:
        chunk = np.asarray(arr, dtype=np.float64)
        valid_mask = np.isfinite(chunk)
        if not valid_mask.any() or not valid_mask[-1]:
            return np.nan
        running_peak = _running_peak_no_carry_np(chunk)
        under = int(np.sum((chunk < running_peak) & valid_mask))
        return float(under) / float(valid_mask.sum())

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_drawdown_duration_expr(w: int) -> pl.Expr:
    """当前连续低于窗口运行最高价的行数（当前行缺失 → NaN；NaN 硬断界）。

    pandas 参考 ``TsCurrentDrawdownDuration``（R5 P1-36(a) / P1-07 / P0）。
    """

    def _fn(arr: np.ndarray) -> float:
        chunk = np.asarray(arr, dtype=np.float64)
        valid_mask = np.isfinite(chunk)
        if not valid_mask.any() or not valid_mask[-1]:
            return np.nan
        running_peak = _running_peak_no_carry_np(chunk)
        streak = 0
        for back in range(chunk.size - 1, -1, -1):
            if not valid_mask[back]:
                break
            if chunk[back] < running_peak[back]:
                streak += 1
            else:
                break
        return float(streak)

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_recovery_fraction_expr(w: int) -> pl.Expr:
    """峰谷修复进度 clip((x_t - T)/(P - T + eps), 0, 1)（尾部连续段）。

    pandas 参考 ``TsRecoveryFraction``（P0-006/007、P1-26 非正价格断路、
    P1-79 最近峰值锚定）；段内任一值 <= 0 或段长 < 2 → NaN。
    """

    def _fn(arr: np.ndarray) -> float:
        seg = _trailing_contiguous_np(arr)
        if seg.size < 2 or not np.all(seg > 0.0):
            return np.nan
        vals = seg
        p = float(np.max(vals))
        # P1-79: reverse-argmax picks the MOST RECENT peak occurrence.
        p_pos = int(vals[::-1].argmax())
        p_pos = vals.size - 1 - p_pos
        t = float(np.min(vals[p_pos:]))
        if p - t <= _EPS_TINY:
            return 1.0
        rf = (float(vals[-1]) - t) / (p - t + _EPS_TINY)
        return float(min(max(rf, 0.0), 1.0))

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


def _fin_bounded_ratio(a):
    # shareholder/churn_network._bounded_ratio（R11 #130，权威）：share-count
    # ratio with domain enforcement — num >= 0, den > 0, ratio <= 1.0；任何
    # 违反都是数据错误，fail-closed 到 NaN（不是 0% 或 >100% 的真比率）。
    q = a[0] / a[1]
    return (
        pl.when(
            a[0].is_null()
            | a[1].is_null()
            | (a[0] < 0)
            | (a[1] <= 0)
            | (q > 1.0)
            | q.is_nan()
            | q.is_infinite()
        )
        .then(None)
        .otherwise(q)
    )


def _fin_sub2(a):
    # x - y 元素级差（churn_network._float_concentration_gap /
    # wave1_valuation.val1_relative_valuation_gap 参考）：任一侧 NULL/NaN
    # → NaN（finite 双方才有效）。
    d = a[0] - a[1]
    return (
        pl.when(a[0].is_null() | a[1].is_null() | a[0].is_nan() | a[1].is_nan())
        .then(None)
        .otherwise(d)
    )


def _fin_signed_log(x: pl.Expr) -> pl.Expr:
    # valuation/ops_v2._signed_log: sign(x) * log1p(|x|)（0 → 0*0 = 0）。
    s = pl.when(x.is_null() | x.is_nan()).then(None).otherwise(x)
    return pl.when(s.is_null()).then(None).otherwise(s.sign() * (s.abs() + 1.0).log())


def _fin_signed_log_gap(a):
    # sign(x)*log1p|x| - sign(y)*log1p|y|（valuation_pe_gap_signed_log /
    # valuation_pcf_gap_signed_log 参考）：任侧 NULL/NaN → NaN。
    lhs = _fin_signed_log(a[0])
    rhs = _fin_signed_log(a[1])
    return pl.when(lhs.is_null() | rhs.is_null()).then(None).otherwise(lhs - rhs)


def _fin_positive_log_gap(a):
    # valuation/ops_v2._positive_log_gap: log(x) - log(y)，仅 x>0 且 y>0；
    # 任一 ≤0 / NULL / NaN → NaN（亏损侧 NaN）。
    lx = pl.when(a[0].is_null() | a[0].is_nan() | (a[0] <= 0)).then(None).otherwise(a[0].log())
    ly = pl.when(a[1].is_null() | a[1].is_nan() | (a[1] <= 0)).then(None).otherwise(a[1].log())
    return pl.when(lx.is_null() | ly.is_null()).then(None).otherwise(lx - ly)


# ---------------------------------------------------------------------------
# wave3d ashare limit elementwise family (2026-09-08).  Pandas authority =
# cleaned_operators/ashare/limit_ops.py: an ABSOLUTE tick tolerance compares
# the price against ``limit ± tolerance``, valid requires ALL operands finite
# (non-null, non-NaN, non-Inf), and the output is {1.0, 0.0, NaN}.
# ---------------------------------------------------------------------------

def _ashare_finite_nonnull(x: pl.Expr) -> pl.Expr:
    return x.is_not_null() & ~x.is_nan() & ~x.is_infinite()


def _ashare_limit_bool_expr(valid: pl.Expr, cond: pl.Expr) -> pl.Expr:
    return pl.when(valid).then(pl.when(cond).then(1.0).otherwise(0.0)).otherwise(None)


def _ashare_limit_up_touch(a):
    # limit_ops.AshareLimitUpTouch: high >= upper_limit - tol (absolute tol).
    p, lim, tol = a[0], a[1], a[2]
    valid = (
        _ashare_finite_nonnull(p) & _ashare_finite_nonnull(lim)
        & _ashare_finite_nonnull(tol)
    )
    return _ashare_limit_bool_expr(valid, p >= lim - tol)


def _ashare_limit_down_touch(a):
    # limit_ops.AshareLimitDownTouch: low <= lower_limit + tol.
    p, lim, tol = a[0], a[1], a[2]
    valid = (
        _ashare_finite_nonnull(p) & _ashare_finite_nonnull(lim)
        & _ashare_finite_nonnull(tol)
    )
    return _ashare_limit_bool_expr(valid, p <= lim + tol)


def _ashare_open_at_upper_limit(a):
    # limit_ops.AshareOpenAtUpperLimit: open >= upper_limit - tol.
    p, lim, tol = a[0], a[1], a[2]
    valid = (
        _ashare_finite_nonnull(p) & _ashare_finite_nonnull(lim)
        & _ashare_finite_nonnull(tol)
    )
    return _ashare_limit_bool_expr(valid, p >= lim - tol)


def _ashare_limit_failed(a):
    # limit_ops.AshareLimitFailed: touched (high >= lim - tol) but failed to
    # hold at close (close < lim - tol); valid = all four operands finite.
    h, c, lim, tol = a[0], a[1], a[2], a[3]
    valid = (
        _ashare_finite_nonnull(h) & _ashare_finite_nonnull(c)
        & _ashare_finite_nonnull(lim) & _ashare_finite_nonnull(tol)
    )
    return _ashare_limit_bool_expr(valid, (h >= lim - tol) & (c < lim - tol))


def _ashare_limit_open_failed(a):
    # limit_ops.AshareLimitOpenFailed: opened at limit (open >= lim - tol) and
    # broke intraday (low < lim - tol); valid = all four operands finite.
    o, lo, lim, tol = a[0], a[1], a[2], a[3]
    valid = (
        _ashare_finite_nonnull(o) & _ashare_finite_nonnull(lo)
        & _ashare_finite_nonnull(lim) & _ashare_finite_nonnull(tol)
    )
    return _ashare_limit_bool_expr(valid, (o >= lim - tol) & (lo < lim - tol))


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
    # wave3 valuation/shareholder (2026-09-08)：纯元素级族（NumpyKernels /
    # valuation::ops_v2 / shareholder::churn_network / wave1_valuation 参考）。
    "a_share_cap_ratio": (2, _fin_ratio2),
    "free_float_ratio": (2, _fin_ratio2),
    "holder_pledge_ratio": (2, _fin_bounded_ratio),
    "holder_freeze_ratio": (2, _fin_bounded_ratio),
    "holder_locked_share_ratio": (2, _fin_bounded_ratio),
    "holder_float_concentration_gap": (2, _fin_sub2),
    "val1_relative_valuation_gap": (2, _fin_sub2),
    "valuation_pe_ttm_lyr_gap": (2, _fin_log_ratio),
    "valuation_pcf_definition_gap": (2, _fin_log_ratio),
    "valuation_pe_gap_signed_log": (2, _fin_signed_log_gap),
    "valuation_pcf_gap_signed_log": (2, _fin_signed_log_gap),
    "valuation_pe_gap_positive": (2, _fin_positive_log_gap),
    "valuation_pcf_gap_positive": (2, _fin_positive_log_gap),
    # wave3d ashare limit elementwise (2026-09-08)：涨跌停触碰 / 炸板 / 开板
    # 布尔族（cleaned_operators/ashare/limit_ops.py 参考，绝对 tick 容差，
    # 全操作数 finite 才有效，输出 {1,0,NaN}）。
    "ashare_limit_up_touch": (3, _ashare_limit_up_touch),
    "ashare_limit_down_touch": (3, _ashare_limit_down_touch),
    "ashare_open_at_upper_limit": (3, _ashare_open_at_upper_limit),
    "ashare_limit_failed": (4, _ashare_limit_failed),
    "ashare_limit_open_failed": (4, _ashare_limit_open_failed),
}

# wave3 flow/momentum/quality window family (2026-09-08): trailing-window
# statistics over the wave1_orderflow / wave1_cs_momentum / wave1_earnings
# pandas references.  Native polars branches live in _compile_polars_impl
# (``_FLOWMOM_WINDOW_OPS`` block); DuckDB SQL branches in the SQL emitter.
_FLOWMOM_WINDOW_OPS: frozenset[str] = frozenset({
    "ofi_volume_imbalance",
    "ofi_abs_imbalance_trend",
    "ofi_dominant_direction",
    "ofi_imbalance_agreement",
    "ofi_imbalance_cv",
    "ofi_imbalance_persistence",
    "ofi_reversal_rate",
    "ofi_volume_flow_regime",
    "ofi_zero_flow_balance",
    "m1_momentum_strength",
    "m1_momentum_stability",
    "m1_momentum_speed_change",
    "m1_volume_adjusted_momentum",
    "sv_net_flow_direction",
    "sv_own_flow_fraction",
    "sv_signed_volume_volatility",
    "sv_self_relative_change",
    "aq1_cash_flow_volatility",
    "aq1_accrual_stability",
    "aq1_cash_conversion_strength",
    "aq1_accrual_ratio_dispersion",
    "aq1_working_capital_accrual",
})

_FLOWMOM_DEFAULT_WINDOW: dict[str, int] = {
    "ofi_volume_imbalance": 20,
    "ofi_abs_imbalance_trend": 20,
    "ofi_dominant_direction": 20,
    "ofi_imbalance_agreement": 20,
    "ofi_imbalance_cv": 20,
    "ofi_imbalance_persistence": 30,
    "ofi_reversal_rate": 20,
    "ofi_volume_flow_regime": 20,
    "ofi_zero_flow_balance": 20,
    "m1_momentum_strength": 20,
    "m1_momentum_stability": 20,
    "m1_momentum_speed_change": 20,
    "m1_volume_adjusted_momentum": 20,
    "sv_net_flow_direction": 20,
    "sv_own_flow_fraction": 20,
    "sv_signed_volume_volatility": 20,
    "sv_self_relative_change": 10,
    "aq1_cash_flow_volatility": 8,
    "aq1_accrual_stability": 8,
    "aq1_cash_conversion_strength": 8,
    "aq1_accrual_ratio_dispersion": 8,
    "aq1_working_capital_accrual": 8,
}

_FLOWMOM_DEFAULT_MIN_PERIODS: dict[str, int] = {
    "ofi_volume_imbalance": 5,
    "ofi_abs_imbalance_trend": 3,
    "ofi_dominant_direction": 5,
    "ofi_imbalance_agreement": 5,
    "ofi_imbalance_cv": 3,
    "ofi_imbalance_persistence": 3,
    "ofi_reversal_rate": 3,
    "ofi_volume_flow_regime": 5,
    "ofi_zero_flow_balance": 5,
    "m1_momentum_strength": 4,
    "m1_momentum_stability": 4,
    "m1_momentum_speed_change": 2,
    "m1_volume_adjusted_momentum": 2,
    "sv_net_flow_direction": 5,
    "sv_own_flow_fraction": 5,
    "sv_signed_volume_volatility": 3,
    "sv_self_relative_change": 1,
    "aq1_cash_flow_volatility": 2,
    "aq1_accrual_stability": 2,
    "aq1_cash_conversion_strength": 2,
    "aq1_accrual_ratio_dispersion": 2,
    "aq1_working_capital_accrual": 2,
}

_FLOWMOM_DEFAULT_THRESHOLD: dict[str, float] = {
    "ofi_dominant_direction": 0.25,
    "ofi_volume_flow_regime": 0.3,
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

    if op == "holder_pledge_change":
        # shareholder/churn_network._pledge_change: pledge_ratio - shift(lag)
        # （lag 是第 2 位置参数 / "lag" attr，>=1）。NULL 不参与：任侧 NULL
        # → NULL（P1-135 语义——缺失期不得静默补 0）。
        if not node.inputs:
            return None
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lag = _int_attr(node, "lag", input_index=1, default=1)
        lag = max(int(lag), 1)
        cur = pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()).then(None).otherwise(pl.col(_VAL))
        prev = cur.shift(lag).over(_INST, order_by=_TS)
        expr = pl.when(cur.is_null() | prev.is_null()).then(None).otherwise(cur - prev)
        return inner.with_columns(expr.alias(_VAL))

    if op == "circulating_cap_ratio_change":
        # valuation/ops_v2.circulating_cap_ratio_change:
        #   safe_div(cc, tc) - shift(1)（先比值、后日间差分；shift 作用在比值上）。
        if len(node.inputs) < 2:
            return None
        cc = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if cc is None:
            return None
        tc = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if tc is None:
            return None
        joined = cc.join(tc.rename({_VAL: "_t"}), on=[_TS, _INST], how="left")
        ratio = _fin_ratio_expr(pl.col(_VAL), pl.col("_t"))
        prev = ratio.shift(1).over(_INST, order_by=_TS)
        expr = pl.when(ratio.is_null() | prev.is_null()).then(None).otherwise(ratio - prev)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    # ------------------------------------------------------------------
    # wave3e cs/group family (2026-09-08): elementwise / cross-section /
    # group-axis native branches.  pandas registry operators are the
    # authority (cs_batch1 / group_ext / group_spectrum / overhaul.daily /
    # cross_section.peer_ops); 三方 parity 见
    # tests/backend_parity/test_csgrp_wave3_parity.py.
    # ------------------------------------------------------------------

    if op == "cs_bucket_fixed":
        # overhaul.daily.pd_cs_bucket_fixed: np.searchsorted(breaks, x,
        # side="right") + 1 — x <= b[0] -> 1, (b[i], b[i+1]] -> i+2,
        # x > b[-1] -> len(breaks)+1.  Non-finite x (NaN/±Inf) -> NaN;
        # non-strictly-increasing breaks raise in the reference -> the
        # branch returns None and falls back to the audited bridge.
        if not node.inputs:
            return None
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        breaks = _float_attr(node, "breaks", "thresholds", default=None)
        if breaks is None:
            pos = _literal_value(node, 0)
            if isinstance(pos, (int, float)) and not isinstance(pos, bool):
                return None  # single scalar is not a breaks list
            raw = None
            if len(node.inputs) >= 2 and node.inputs[1].op == "literal":
                raw = node.inputs[1].attrs.get("value")
            if raw is None and "breaks" in (node.attrs or {}):
                raw = node.attrs["breaks"]
            if not isinstance(raw, (list, tuple)) or not raw:
                return None
            breaks = raw
        try:
            bvals = sorted(float(b) for b in breaks)
        except (TypeError, ValueError):
            return None
        if not bvals or any(b2 <= b1 for b1, b2 in zip(bvals, bvals[1:])):
            return None
        bucket = pl.lit(1.0)
        for b in bvals:
            # searchsorted(side="right"): number of breaks <= x.
            bucket = bucket + pl.when(pl.col(_VAL) >= b).then(pl.lit(1.0)).otherwise(pl.lit(0.0))
        expr = pl.when(
            pl.col(_VAL).is_null() | pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()
        ).then(None).otherwise(bucket)
        return inner.with_columns(expr.alias(_VAL))

    if op == "cs_empirical_bayes_shrinkage":
        # cs_batch1.CsEmpiricalBayesShrinkage._empirical_bayes_row:
        # valid = finite(estimate) & finite(std_err) & std_err > 0;
        # cross-section breadth < _MIN_BREADTH (10) -> whole row NaN;
        # shrinkage_factor < 0 -> fail-closed all-NaN;
        # cs_mean = mean(valid), cs_var = var(valid, ddof=1);
        # cs_var <= 0 -> shrink every valid cell to cs_mean;
        # w = clip(1 / (1 + lambda * se^2 / cs_var), 0, 1);
        # shrunk = cs_mean + (estimate - cs_mean) * w (valid cells only).
        if len(node.inputs) < 2:
            return None
        est = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if est is None:
            return None
        se = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if se is None:
            return None
        joined = est.join(se.rename({_VAL: "_se"}), on=[_TS, _INST], how="left")
        lam = _literal_value(node, 1)
        if lam is None:
            lam = _float_attr(node, "shrinkage_factor", "lambda", default=1.0)
        if lam is not None and lam < 0:
            # Invalid shrinkage factor -> fail-closed whole panel NaN.
            return joined.with_columns(pl.lit(None, dtype=pl.Float64).alias(_VAL)).select(_TS, _INST, _VAL)
        if lam is None:
            lam = 1.0
        e = pl.col(_VAL)
        s = pl.col("_se")
        valid = e.is_not_null() & s.is_not_null() & (s > 0)
        vv = pl.when(valid).then(e)
        sv = pl.when(valid).then(s)
        cnt = vv.count().over(_TS, order_by=_INST)
        cs_mean = vv.mean().over(_TS, order_by=_INST)
        cs_var = vv.var(ddof=1).over(_TS, order_by=_INST)
        var_ratio = pl.when(cs_var.is_null() | (cs_var <= 0)).then(None).otherwise(sv ** 2 / cs_var)
        w_raw = pl.when(var_ratio.is_null()).then(None).otherwise(1.0 / (1.0 + float(lam) * var_ratio))
        w = pl.when(w_raw.is_null()).then(None).otherwise(w_raw.clip(0.0, 1.0))
        shrunk = pl.when(cs_var.is_null() | (cs_var <= 0)).then(cs_mean).otherwise(
            cs_mean + (vv - cs_mean) * w
        )
        expr = (
            pl.when(cnt.cast(pl.Float64) < 10.0)
            .then(None)
            .when(~valid)
            .then(None)
            .otherwise(shrunk)
        )
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "group_ex_self_weighted_mean":
        # group_ext.GroupExSelfWeightedMean: (Σw·x - w_j·x_j) / (Σw - w_j)
        # per (ts, group) over members with finite(x) & finite(w) & w >= 0;
        # total_w non-finite or <= 0 -> group stays NaN; denominator <= 0
        # -> NaN; invalid self cell -> NaN.  The group-constant sums make
        # this a pure over expression (no self-join).
        if len(node.inputs) < 3:
            return None
        x = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if x is None:
            return None
        wgt = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if wgt is None:
            return None
        grp = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        if grp is None:
            return None
        joined = (
            x.join(wgt.rename({_VAL: "_w"}), on=[_TS, _INST], how="left")
            .join(grp.rename({_VAL: _GRP}), on=[_TS, _INST], how="left")
        )
        xv = pl.col(_VAL)
        wv = pl.col("_w")
        gv = pl.col(_GRP)
        over_keys = (_TS, _GRP)
        valid = (
            xv.is_not_null() & ~xv.is_nan() & ~xv.is_infinite()
            & wv.is_not_null() & ~wv.is_nan() & ~wv.is_infinite() & (wv >= 0)
            & gv.is_not_null()
        )
        vv = pl.when(valid).then(xv)
        ww = pl.when(valid).then(wv)
        sw = ww.sum().over(*over_keys, order_by=_INST)
        swx = (vv * ww).sum().over(*over_keys, order_by=_INST)
        denom = sw - wv
        numer = swx - wv * xv
        expr = (
            pl.when(gv.is_null())
            .then(None)
            .when(~valid)
            .then(None)
            .when(sw.is_null() | (sw <= 0))
            .then(None)
            .when(denom.is_null() | (denom <= 0))
            .then(None)
            .otherwise(numer / denom)
        )
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "group_feature_valid_member_count":
        # group_spectrum.GroupFeatureValidMemberCount: per (ts, group) the
        # number of members whose ALL THREE features are finite; the count
        # is broadcast to EVERY member of the group (including members with
        # missing features).  Invalid (NaN/±Inf/None/empty) group labels are
        # not memberships -> NaN.
        if len(node.inputs) < 4:
            return None
        f1 = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if f1 is None:
            return None
        f2 = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if f2 is None:
            return None
        f3 = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        if f3 is None:
            return None
        grp = _compile_child(node, 3, base, parent_op=op, ctx=ctx, memo=memo)
        if grp is None:
            return None
        joined = (
            f1.join(f2.rename({_VAL: "_f2"}), on=[_TS, _INST], how="left")
            .join(f3.rename({_VAL: "_f3"}), on=[_TS, _INST], how="left")
            .join(grp.rename({_VAL: _GRP}), on=[_TS, _INST], how="left")
        )
        gv = pl.col(_GRP)
        over_keys = (_TS, _GRP)
        member_ok = (
            pl.col(_VAL).is_not_null() & ~pl.col(_VAL).is_nan() & ~pl.col(_VAL).is_infinite()
            & pl.col("_f2").is_not_null() & ~pl.col("_f2").is_nan() & ~pl.col("_f2").is_infinite()
            & pl.col("_f3").is_not_null() & ~pl.col("_f3").is_nan() & ~pl.col("_f3").is_infinite()
            & gv.is_not_null()
        )
        cnt = (
            pl.when(member_ok).then(1.0).otherwise(0.0)
            .sum().over(*over_keys, order_by=_INST)
        )
        expr = pl.when(gv.is_null()).then(None).otherwise(cnt)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "group_peer_deviation_index":
        # cross_section.peer_ops._peer_deviation_index: sum of per-frame
        # cross-sectional z-scores (over the whole ts partition — the
        # signature has no group input), each frame using ITS OWN finite
        # mask; frames with < 2 finite values or sd <= 1e-12 are skipped;
        # NaN-first accumulation means cells missing from every contributing
        # frame stay NaN (never manufactured 0).  Because every contributing
        # z is finite, NaN-first equals a null-skipping sum.
        if len(node.inputs) < 2:
            return None
        layers = []
        for i in range(len(node.inputs)):
            child = _compile_child(node, i, base, parent_op=op, ctx=ctx, memo=memo)
            if child is None:
                return None
            layers.append(child)
        names = [_VAL, "_y", "_z", "_w", "_u", "_t", "_s"]
        joined = _join_multi(layers)
        z_exprs = []
        for name in names[: len(layers)]:
            v = pl.col(name)
            vv = pl.when(v.is_null() | v.is_nan() | v.is_infinite()).then(None).otherwise(v)
            cnt = vv.count().over(_TS, order_by=_INST)
            m = vv.mean().over(_TS, order_by=_INST)
            sd = vv.std(ddof=0).over(_TS, order_by=_INST)
            z = (
                pl.when(cnt < 2)
                .then(None)
                .when(sd.is_null() | (sd <= 1e-12))
                .then(None)
                .otherwise((vv - m) / sd)
            )
            z_exprs.append(z)
        # NaN-first accumulation: cells missing from every contributing frame
        # stay NULL; otherwise the value is the sum of the finite z-scores
        # (polars null-skipping sum == the reference's NaN-first sum).
        any_finite = pl.sum_horizontal(
            [z.is_not_null().cast(pl.UInt8) for z in z_exprs]
        )
        acc = pl.when(any_finite == 0).then(None).otherwise(pl.sum_horizontal(z_exprs))
        return joined.with_columns(acc.alias(_VAL)).select(_TS, _INST, _VAL)

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

    # =====================================================================
    # wave3 ts path/risk family (2026-09-08) — rolling_map branches over the
    # certified pandas kernels (alpha_language_shape / ts_model.complexity /
    # regression_models / downside_risk / stateful.drawdown_path / robust_stats).
    # These kernels are trailing-run / pairwise loops that a pure over-expression
    # cannot replicate exactly, so they use the audited ``rolling_map`` helper
    # (python_rolling tier, NOT native) and DuckDB windows cannot express them
    # either (nested rank/trailing-run) — those stay OFF sql_tiers.  三方 parity
    # 见 tests/backend_parity/test_ts_wave3_parity.py.
    # =====================================================================

    if op == "ts_monotonicity":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        mp = _int_attr(node, "min_periods", input_index=1, default=3)
        return inner.with_columns(
            _rolling_monotonicity_expr(w, min_periods=max(3, mp)).alias(_VAL)
        )

    if op == "ts_turning_point_ratio":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=60)
        return inner.with_columns(
            _rolling_turning_point_ratio_expr(w).alias(_VAL)
        )

    if op == "ts_endpoint_deviation":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        mp = _int_attr(node, "min_periods", input_index=1, default=3)
        return inner.with_columns(
            _rolling_endpoint_deviation_expr(w, min_periods=max(3, mp)).alias(_VAL)
        )

    if op == "ts_vol_shift_score":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        mp = _int_attr(node, "min_periods", input_index=1, default=5)
        return inner.with_columns(
            _rolling_vol_shift_score_expr(w, min_periods=max(4, mp)).alias(_VAL)
        )

    if op == "ts_time_under_water":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = max(_window_int(node, default=20), 2)
        return inner.with_columns(
            _rolling_time_under_water_expr(w).alias(_VAL)
        )

    if op == "ts_current_drawdown_duration":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = max(_window_int(node, default=20), 2)
        return inner.with_columns(
            _rolling_drawdown_duration_expr(w).alias(_VAL)
        )

    if op == "ts_recovery_fraction":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = max(_window_int(node, default=60), 2)
        return inner.with_columns(
            _rolling_recovery_fraction_expr(w).alias(_VAL)
        )

    # =====================================================================
    # wave3 flow/momentum/quality window family (2026-09-08) — pure native
    # polars Expr branches for the wave1_orderflow / wave1_cs_momentum /
    # wave1_earnings pandas references.  All trailing windows are bounded by
    # ``.over(_INST, order_by=_TS)`` and skip non-finite window rows exactly
    # like the pandas kernels (a NaN input row neither counts toward
    # min_periods nor contributes a sample).  三方 parity 见
    # tests/backend_parity/test_flowmom_wave3_parity.py.
    # =====================================================================

    if op in _FLOWMOM_WINDOW_OPS:
        if len(node.inputs) < 1:
            return None
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=_FLOWMOM_DEFAULT_WINDOW[op])
        mp = _int_attr(node, "min_periods", input_index=1, default=_FLOWMOM_DEFAULT_MIN_PERIODS[op])
        if mp > w:
            mp = w
        # window membership count of FINITE rows (pandas rolling counts only
        # finite samples; NaN rows never contribute).
        v_finite = pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL)).otherwise(None)
        cnt = v_finite.is_not_null().cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        gate = pl.when(cnt.is_null() | (cnt < mp)).then(None)

        if op == "ofi_volume_imbalance":
            # (Σbuy - Σsell) / (Σbuy + Σsell) over the FINITE window slice;
            # buy = vol where sv>0, sell = vol where sv<0 (0 = neutral).
            buy = pl.when(pl.col(_VAL).is_finite() & (pl.col(_VAL) > 0)).then(pl.col(_VAL)).otherwise(0.0)
            sell = pl.when(pl.col(_VAL).is_finite() & (pl.col(_VAL) < 0)).then(-pl.col(_VAL)).otherwise(0.0)
            bsum = buy.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            ssum = sell.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            den = bsum + ssum
            expr = gate.when(den.is_null() | (den <= 0)).then(None).otherwise((bsum - ssum) / den)
            # pandas reference additionally requires the CURRENT row finite.
            expr = pl.when(v_finite.is_null()).then(None).otherwise(expr)
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op in {"ofi_dominant_direction", "ofi_volume_flow_regime"}:
            th = _float_attr(node, "threshold", "th", default=_FLOWMOM_DEFAULT_THRESHOLD[op])
            if not (0.0 <= th <= 1.0):
                from factor_engine.backend.plan_params import PlanParamError

                raise PlanParamError(f"{op}.threshold must be in [0, 1]")
            buy = pl.when(pl.col(_VAL).is_finite() & (pl.col(_VAL) > 0)).then(pl.col(_VAL)).otherwise(0.0)
            sell = pl.when(pl.col(_VAL).is_finite() & (pl.col(_VAL) < 0)).then(-pl.col(_VAL)).otherwise(0.0)
            bsum = buy.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            ssum = sell.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            den = bsum + ssum
            if op == "ofi_dominant_direction":
                # buy_share > th => 1; buy_share < (1 - th) => -1; else 0
                inner_expr = (
                    pl.when(bsum / den > th).then(1.0)
                    .when(bsum / den < (1.0 - th)).then(-1.0)
                    .otherwise(0.0)
                )
            else:
                # net = (buy - sell)/(buy + sell); > th => 1; < -th => -1; else 0
                net = (bsum - ssum) / den
                inner_expr = (
                    pl.when(net > th).then(1.0)
                    .when(net < -th).then(-1.0)
                    .otherwise(0.0)
                )
            expr = gate.when(den.is_null() | (den <= 0)).then(None).otherwise(inner_expr)
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "ofi_imbalance_persistence":
            # lag-1 Pearson correlation of the COMPACTED finite slice
            # (x = vals[:-1], y = vals[1:] after dropping non-finite rows).
            # NaN rows shift the pairing (compacted), which rolling_corr over
            # raw rows cannot express — use rolling_map (python_rolling tier).
            def _persistence_fn(arr: np.ndarray) -> float:
                arr = np.asarray(arr, dtype=np.float64)
                vals = arr[np.isfinite(arr)]
                if vals.size < mp or vals.size < 3:
                    # pandas: x=vals[:-1], y=vals[1:] need >= 2 points each;
                    # with mp >= 3 the mp gate already covers size < 3, but a
                    # zero-size window still fails closed here.
                    return np.nan
                x = vals[:-1]
                y = vals[1:]
                sx = float(np.std(x))
                sy = float(np.std(y))
                if sx <= 0 or sy <= 0:
                    return np.nan
                return float(np.corrcoef(x, y)[0, 1])

            # pandas kernel also gates on the CURRENT row being finite.
            expr = (
                pl.when(v_finite.is_null())
                .then(None)
                .otherwise(
                    pl.col(_VAL)
                    .rolling_map(_persistence_fn, window_size=w, min_samples=mp)
                    .over(_INST, order_by=_TS)
                )
            )
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "ofi_abs_imbalance_trend":
            # sign(sv)*|sv|/(|sv|+1e-12) regressed on the COMPACTED finite-slice
            # time index 0..n-1; slope * sqrt(n) with n = FINITE window rows.
            # Windowed ranks are an affine transform of compacted positions
            # within each window (offset constant per window), so the OLS slope
            # is unchanged; compute it from raw window moments against the
            # window-end means (NaN rows drop out of every aggregate).
            seq = pl.when(pl.col(_VAL).is_finite()).then(
                pl.col(_VAL).sign() * pl.col(_VAL).abs() / (pl.col(_VAL).abs() + 1e-12)
            ).otherwise(None)
            rank = v_finite.is_not_null().cast(pl.Float64).cum_sum().over(_INST, order_by=_TS)
            rank_w = pl.when(v_finite.is_not_null()).then(rank).otherwise(None)
            rank_mean = rank_w.rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            seq_mean = seq.rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            num = (
                (rank_w * seq).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
                - cnt * rank_mean * seq_mean
            )
            den = (
                (rank_w * rank_w).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
                - cnt * rank_mean * rank_mean
            )
            slope = pl.when(den.is_null() | (den <= 0)).then(None).otherwise(num / den)
            expr = gate.when(slope.is_null()).then(None).otherwise(slope * cnt.sqrt())
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "ofi_imbalance_cv":
            # im = |Σ(sv)| / Σ|sv| (zero total -> 0 by the +1e-12 eps contract);
            # output carries the NET SIGN: im * (1 if Σ(sv) >= 0 else -1).
            asum = v_finite.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            abs_v = pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL).abs()).otherwise(None)
            abs_sum = abs_v.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            im = pl.when(abs_sum.is_null() | (abs_sum <= 0)).then(None).otherwise(asum.abs() / (abs_sum + 1e-12))
            expr = gate.when(im.is_null()).then(None).otherwise(
                im * pl.when(asum >= 0).then(1.0).otherwise(-1.0)
            )
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "ofi_imbalance_agreement":
            # |mean(sign(sv))| over the finite slice; sign(0) = 0 counts as a
            # sample (pandas np.sign(0) == 0 participates in the mean).
            sgn = pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL).sign()).otherwise(None)
            smean = sgn.rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            expr = gate.when(smean.is_null()).then(None).otherwise(smean.abs())
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "ofi_zero_flow_balance":
            # mean(sv == 0) over the finite slice (zeros ARE samples here).
            zero = pl.when(pl.col(_VAL).is_finite()).then(
                (pl.col(_VAL) == 0).cast(pl.Float64)
            ).otherwise(None)
            zmean = zero.rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            expr = gate.when(zmean.is_null()).then(None).otherwise(zmean)
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "ofi_reversal_rate":
            # mean(sign[t] != sign[t-1]) over the COMPACTED finite slice
            # pairs.  A pair's window membership depends on the partner row's
            # position relative to the CONSUMING window start (pairs may span
            # NaN gaps), so a pure per-row rolling flag cannot express it
            # exactly — use rolling_map over the window slice (same tier as
            # ts_product / WMA: POLARS_LONG_PYTHON_ROLLING, not native).
            def _reversal_fn(arr: np.ndarray) -> float:
                arr = np.asarray(arr, dtype=np.float64)
                valid = arr[np.isfinite(arr)]
                if valid.size < 2:
                    return np.nan
                signs = np.sign(valid)
                return float(np.mean(signs[1:] != signs[:-1]))

            expr = (
                pl.col(_VAL)
                .rolling_map(_reversal_fn, window_size=w, min_samples=mp)
                .over(_INST, order_by=_TS)
            )
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "sv_net_flow_direction":
            # mean(sign(sv) * |sv|/mean|sv|).  mean|sv| is a per-window
            # constant, so mean(s_i*|v_i|/mabs) == Σ(s_i*|v_i|) / Σ|v_i| —
            # compute the ratio from the two window sums (the naive nested
            # form would mix per-row mabs from shifted windows).  Σ|v| <=
            # 1e-12 (mean|v| <= 1e-12/n) -> 0.0 via the pandas mabs guard.
            signed_v = pl.when(pl.col(_VAL).is_finite()).then(
                pl.col(_VAL).sign() * pl.col(_VAL).abs()
            ).otherwise(None)
            abs_sum = pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL).abs()).otherwise(None).rolling_sum(
                window_size=w, min_samples=1
            ).over(_INST, order_by=_TS)
            ssum = signed_v.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            expr = gate.when(abs_sum.is_null()).then(None).otherwise(
                pl.when(abs_sum <= 1e-12).then(0.0).otherwise(ssum / abs_sum)
            )
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "sv_own_flow_fraction":
            # amount_t / Σ(amount over finite AND > 0 rows); min_periods
            # counts POSITIVE rows (pandas: ok = isfinite & v > 0), and the
            # current row must be finite (a <= 0 numerator is emitted as a
            # negative/small fraction, matching the reference).
            pos = pl.when(pl.col(_VAL).is_finite() & (pl.col(_VAL) > 0)).then(pl.col(_VAL)).otherwise(None)
            pos_cnt = pos.is_not_null().cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            total = pos.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            expr = (
                pl.when(pos_cnt.is_null() | (pos_cnt < mp)).then(None)
                .when(total.is_null() | (total <= 0)).then(None)
                .when(v_finite.is_null()).then(None)
                .otherwise(v_finite / total)
            )
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "sv_signed_volume_volatility":
            # std(vals / std(|vals|)) with the SAME window std(|vals|) for
            # every element — std(x/c) == std(x)/c, so the output is the
            # population-std(vals) / population-std(|vals|) ratio over the
            # window (the naive nested form would mix per-row stds).  std(|v|)
            # <= 1e-12 -> NaN.
            av = pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL).abs()).otherwise(None)
            std_signed = v_finite.rolling_std(window_size=w, min_samples=1, ddof=0).over(_INST, order_by=_TS)
            std_abs = av.rolling_std(window_size=w, min_samples=1, ddof=0).over(_INST, order_by=_TS)
            expr = gate.when(std_abs.is_null() | (std_abs <= 1e-12)).then(None).otherwise(
                std_signed / std_abs
            )
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        def _compound_total(arr: np.ndarray) -> float:
            arr = np.asarray(arr, dtype=np.float64)
            vals = arr[np.isfinite(arr)]
            if vals.size == 0:
                return np.nan
            return float(np.prod(1.0 + vals) - 1.0)

        if op == "m1_momentum_strength":
            # |Π(1+r) - 1| / std(r, ddof=1) over the COMPACTED finite slice.
            # Π(1+r) can go NEGATIVE (r < -1), so log-space summation is not
            # exact — use rolling_map (python_rolling tier).  std <= 1e-12:
            # total == 0 -> 0.0, else NaN (pandas contract).
            total = (
                pl.col(_VAL)
                .rolling_map(_compound_total, window_size=w, min_samples=1)
                .over(_INST, order_by=_TS)
            )
            rstd = v_finite.rolling_std(window_size=w, min_samples=1, ddof=1).over(_INST, order_by=_TS)
            expr = gate.when(rstd.is_null() | (rstd <= 1e-12)).then(None).otherwise(
                total.abs() / rstd
            )
            # pandas: std <= 1e-12 emits 0.0 only when total == 0 exactly.
            expr = pl.when(rstd.is_null() | (rstd <= 1e-12)).then(
                pl.when((total == 0.0) & (cnt >= mp)).then(0.0).otherwise(None)
            ).otherwise(expr)
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "m1_momentum_stability":
            # mean(r > 0) over the finite slice (zeros count as non-positive).
            pos = pl.when(pl.col(_VAL).is_finite()).then((pl.col(_VAL) > 0).cast(pl.Float64)).otherwise(None)
            pmean = pos.rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            expr = gate.when(pmean.is_null()).then(None).otherwise(pmean)
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "m1_momentum_speed_change":
            # mom(fast) - mom(slow), mom = Π(1+r) - 1 over each own window;
            # slow requires >= mp finite rows, fast requires >= 2; current row
            # finite.  fast_window/slow_window arrive as ATTRS (analyzer
            # normalizes the positional args into named params).
            fw = _int_attr(node, "fast_window", input_index=1, default=5)
            sw = _int_attr(node, "slow_window", input_index=2, default=20)
            if fw >= sw:
                from factor_engine.backend.plan_params import PlanParamError

                raise PlanParamError(
                    "m1_momentum_speed_change: fast_window must be < slow_window"
                )
            total_slow = (
                pl.col(_VAL)
                .rolling_map(_compound_total, window_size=sw, min_samples=1)
                .over(_INST, order_by=_TS)
            )
            cnt_slow = v_finite.is_not_null().cast(pl.Float64).rolling_sum(window_size=sw, min_samples=1).over(_INST, order_by=_TS)
            total_fast = (
                pl.col(_VAL)
                .rolling_map(_compound_total, window_size=fw, min_samples=1)
                .over(_INST, order_by=_TS)
            )
            cnt_fast = v_finite.is_not_null().cast(pl.Float64).rolling_sum(window_size=fw, min_samples=1).over(_INST, order_by=_TS)
            expr = (
                pl.when(v_finite.is_null()).then(None)
                .when(cnt_slow.is_null() | (cnt_slow < mp)).then(None)
                .when(cnt_fast.is_null() | (cnt_fast < 2)).then(None)
                .otherwise(total_fast - total_slow)
            )
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "m1_volume_adjusted_momentum":
            if len(node.inputs) < 2:
                return None
            t_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
            if t_in is None:
                return None
            joined = _join_binary(inner, t_in)
            pair = pl.col(_VAL).is_finite() & pl.col("_y").is_finite() & (pl.col("_y") > 0)
            pair_cnt = pair.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            r_v = pl.when(pair).then(pl.col(_VAL)).otherwise(None)
            t_v = pl.when(pair).then(pl.col("_y")).otherwise(None)
            num = (r_v * t_v).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            den = t_v.rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            expr = (
                pl.when(pair_cnt.is_null() | (pair_cnt < mp)).then(None)
                .when(den.is_null() | (den <= 0)).then(None)
                .otherwise(num / den)
            )
            return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "sv_self_relative_change":
            # (amount_t - mean(prior w rows, finite AND > 0)) / mean;
            # base window EXCLUDES the current row, needs >= 1 valid base row.
            base_v = pl.when(pl.col(_VAL).is_finite() & (pl.col(_VAL) > 0)).then(pl.col(_VAL)).otherwise(None)
            prior = base_v.shift(1).over(_INST, order_by=_TS)
            bmean = prior.rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            expr = (
                pl.when(v_finite.is_null()).then(None)
                .when(bmean.is_null() | (bmean <= 0)).then(None)
                .otherwise((v_finite - bmean) / bmean)
            )
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "aq1_cash_flow_volatility":
            # std(ocf, ddof=1) / |mean(ocf)|; |mean| <= 1e-12 -> 0.0.
            cmean = v_finite.rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            cstd = v_finite.rolling_std(window_size=w, min_samples=1, ddof=1).over(_INST, order_by=_TS)
            expr = gate.when(cmean.is_null()).then(None).otherwise(
                pl.when(cmean.abs() <= 1e-12).then(0.0).otherwise(cstd / cmean.abs())
            )
            return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op in {"aq1_accrual_stability", "aq1_accrual_ratio_dispersion"}:
            # Accrual ratio r_i = ΔWC_i / (|E_i| + 1e-12) over the COMPACTED
            # ok slice: ΔWC is np.diff over the ok-filtered window rows, so a
            # pair spans any run of non-ok rows (adjacent-OK pairing, not
            # adjacent-row).  Pair membership depends on the consuming window
            # start — not expressible as per-row rolling flags, so compute
            # per inst with a trailing-window numpy walk (map_groups tier).
            if len(node.inputs) < 2:
                return None
            e_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
            if e_in is None:
                return None
            joined = _join_binary(inner, e_in)

            def _accrual_pair(g: pl.DataFrame) -> pl.DataFrame:
                wc_arr = g[_VAL].to_numpy()
                e_arr = g["_y"].to_numpy()
                rows = wc_arr.size
                out = np.full(rows, np.nan, dtype=np.float64)
                for r in range(rows):
                    # pandas reference skips rows where either input is
                    # non-finite at the CURRENT row (fail-closed).
                    if not (np.isfinite(wc_arr[r]) and np.isfinite(e_arr[r])):
                        continue
                    lo = max(0, r - w + 1)
                    w_win = wc_arr[lo : r + 1]
                    e_win = e_arr[lo : r + 1]
                    ok = (
                        np.isfinite(w_win) & np.isfinite(e_win)
                        & (np.abs(e_win) > 1e-12)
                    )
                    if ok.sum() < mp:
                        continue
                    dw = np.diff(w_win[ok])
                    e_ = np.abs(e_win[ok][1:])
                    if dw.size < 2:
                        continue
                    ratio = dw / (e_ + 1e-12)
                    if op == "aq1_accrual_stability":
                        m = float(np.mean(ratio))
                        s = float(np.std(ratio, ddof=1))
                        out[r] = 0.0 if m <= 1e-12 else -s / abs(m)
                    else:
                        out[r] = float(np.std(np.abs(ratio), ddof=1))
                return g.select(
                    pl.col(_TS),
                    pl.col(_INST),
                    pl.Series(_VAL, out),
                )

            schema = joined.collect_schema()
            return joined.group_by(_INST, maintain_order=True).map_groups(
                _accrual_pair,
                schema={_TS: schema[_TS], _INST: schema[_INST], _VAL: pl.Float64},
            )

        if op == "aq1_cash_conversion_strength":
            if len(node.inputs) < 2:
                return None
            e_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
            if e_in is None:
                return None
            joined = _join_binary(inner, e_in)
            ok = pl.col(_VAL).is_finite() & pl.col("_y").is_finite() & (pl.col("_y").abs() > 1e-12)
            ok_cnt = ok.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            ratio = pl.when(ok).then(pl.col(_VAL) / pl.col("_y")).otherwise(None)
            rmean = ratio.rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
            expr = (
                pl.when(ok_cnt.is_null() | (ok_cnt < mp)).then(None)
                .when(rmean.is_null()).then(None)
                .otherwise(rmean)
            )
            return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        if op == "aq1_working_capital_accrual":
            if len(node.inputs) < 2:
                return None
            e_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
            if e_in is None:
                return None
            joined = _join_binary(inner, e_in)
            both_finite = pl.col(_VAL).is_finite() & pl.col("_y").is_finite()
            # prev_* resets to None on any non-finite row (pandas resets
            # prev_wc/prev_e when either input is NaN at a row).
            prev_ok = both_finite.shift(1).fill_null(False).over(_INST, order_by=_TS) & both_finite
            prev_wc = pl.when(prev_ok).then(pl.col(_VAL).shift(1).over(_INST, order_by=_TS)).otherwise(None)
            dw = pl.col(_VAL) - prev_wc
            expr = (
                pl.when(both_finite).then(
                    pl.when(prev_ok).then(
                        pl.when(pl.col("_y").abs() > 1e-12).then(dw / pl.col("_y").abs()).otherwise(None)
                    ).otherwise(None)
                ).otherwise(None)
            )
            return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

        return None

    # ------------------------------------------------------------------
    # wave3d ashare limit rolling family (2026-09-08).  Pandas authority =
    # cleaned_operators/ashare/state_machine.py: a RELATIVE tick tolerance
    # (limit * (1 ± tol)), per-cell "known" gates (an unknown/NaN row → NaN
    # output, never a 0), trailing-window counts and a run-length streak
    # that is broken (NaN) by missing/unknown/suspended rows.  All windows
    # are trailing inclusive (.over(_INST, order_by=_TS)).
    # ------------------------------------------------------------------

    def _ashare_tol(node_: PlanNode, default: float = 0.005) -> float:
        from factor_engine.backend.plan_params import PlanParamError

        raw = None
        if "tick_tolerance" in (node_.attrs or {}) and node_.attrs["tick_tolerance"] is not None:
            raw = node_.attrs["tick_tolerance"]
        if raw is None:
            # positional literal: tick_tolerance is the LAST numeric literal
            # (the window also arrives as a numeric literal, earlier)
            for child in reversed(node_.inputs[1:]):
                value = child.attrs.get("value") if child.op == "literal" else None
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    raw = value
                    break
        tol = float(raw) if raw is not None else default
        if tol < 0.0:
            raise PlanParamError("tick_tolerance must be non-negative")
        return tol

    def _ashare_side(node_: PlanNode) -> str:
        from factor_engine.backend.plan_params import PlanParamError

        raw = None
        if "side" in (node_.attrs or {}) and node_.attrs["side"] is not None:
            raw = node_.attrs["side"]
        if raw is None:
            for child in node_.inputs[1:]:
                if child.op == "literal" and isinstance(child.attrs.get("value"), str):
                    raw = child.attrs["value"]
                    break
        kind = str(raw or "up").lower()
        if kind not in {"up", "down"}:
            raise PlanParamError("side must be 'up' or 'down'")
        return kind

    def _ashare_finite(e: "pl.Expr") -> "pl.Expr":
        return e.is_not_null() & ~e.is_nan() & ~e.is_infinite()

    if op in {"ashare_limit_touch_count", "ashare_failed_limit_count"}:
        # state_machine: touch_count condition = high >= high_limit*(1-tol)
        # (up) / low <= low_limit*(1+tol) (down); failed_limit_count condition
        # = touched but failed to hold at close.  known = the compared
        # operands all finite; a window with zero known rows → NaN and an
        # unknown CURRENT row → NaN (pandas ``_rolling_count`` gate).
        if op == "ashare_limit_touch_count":
            if len(node.inputs) < 4:
                return None
            side_kind = _ashare_side(node)
            p_in, lim_in = (0, 2) if side_kind != "down" else (1, 3)
            price = _compile_child(node, p_in, base, parent_op=op, ctx=ctx, memo=memo)
            limit = _compile_child(node, lim_in, base, parent_op=op, ctx=ctx, memo=memo)
            if price is None or limit is None:
                return None
            joined = price.join(limit.rename({_VAL: "_lim"}), on=[_TS, _INST], how="left")
        else:
            if len(node.inputs) < 5:
                return None
            side_kind = _ashare_side(node)
            p_in, close_in, lim_in = (0, 2, 3) if side_kind != "down" else (1, 2, 4)
            price = _compile_child(node, p_in, base, parent_op=op, ctx=ctx, memo=memo)
            close_l = _compile_child(node, close_in, base, parent_op=op, ctx=ctx, memo=memo)
            limit = _compile_child(node, lim_in, base, parent_op=op, ctx=ctx, memo=memo)
            if price is None or close_l is None or limit is None:
                return None
            joined = (
                price.join(close_l.rename({_VAL: "_close"}), on=[_TS, _INST], how="left")
                .join(limit.rename({_VAL: "_lim"}), on=[_TS, _INST], how="left")
            )
        side_kind = _ashare_side(node)
        w = _window_int(node, default=20)
        tol = _ashare_tol(node)
        p = pl.col(_VAL)
        lim = pl.col("_lim")
        if op == "ashare_limit_touch_count":
            bound = lim * (1.0 - tol) if side_kind == "up" else lim * (1.0 + tol)
            cond = (p >= bound) if side_kind == "up" else (p <= bound)
            known = _ashare_finite(p) & _ashare_finite(lim)
        else:
            cl = pl.col("_close")
            if side_kind == "up":
                bound = lim * (1.0 - tol)
                cond = (p >= bound) & (cl < bound)
            else:
                bound = lim * (1.0 + tol)
                cond = (p <= bound) & (cl > bound)
            known = _ashare_finite(p) & _ashare_finite(cl) & _ashare_finite(lim)
        cnt_known = known.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        cnt_hits = (known & cond).cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        expr = pl.when(known).then(
            pl.when(cnt_known.is_null() | (cnt_known <= 0)).then(None).otherwise(cnt_hits)
        ).otherwise(None)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ashare_limit_asymmetry":
        # state_machine.AshareLimitAsymmetry: (Σup_event - Σdown_event) /
        # known_count over the trailing window, where known = up|down finite
        # (either side carrying a known status makes the day count); a window
        # with zero known rows → NaN.  A NaN event inside a known day
        # contributes 0 (pandas np.nansum of the where-masked chunk).
        if len(node.inputs) < 2:
            return None
        up_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        dn_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if up_l is None or dn_l is None:
            return None
        joined = up_l.join(dn_l.rename({_VAL: "_dn"}), on=[_TS, _INST], how="left")
        w = _window_int(node, default=20)
        u = pl.col(_VAL)
        d = pl.col("_dn")
        known = _ashare_finite(u) | _ashare_finite(d)
        known_cnt = known.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        up_sum = (
            pl.when(known & (u != 0)).then(pl.when(_ashare_finite(u)).then(u).otherwise(0.0)).otherwise(0.0)
            .rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        )
        dn_sum = (
            pl.when(known & (d != 0)).then(pl.when(_ashare_finite(d)).then(d).otherwise(0.0)).otherwise(0.0)
            .rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        )
        expr = pl.when(known_cnt.is_null() | (known_cnt <= 0)).then(None).otherwise((up_sum - dn_sum) / known_cnt)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op in {"ashare_limit_up_volume_ratio", "ashare_limit_down_volume_ratio"}:
        # state_machine._event_volume_ratio: mean(volume on event days) /
        # mean(volume on volume-known days) over the trailing window, where an
        # event day requires BOTH the volume and the event to be finite and
        # the event != 0; no event day, no volume-known day, or a non-positive
        # base volume → NaN.
        if len(node.inputs) < 2:
            return None
        vol_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        ev_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if vol_l is None or ev_l is None:
            return None
        joined = vol_l.join(ev_l.rename({_VAL: "_ev"}), on=[_TS, _INST], how="left")
        w = _window_int(node, default=20)
        v = pl.col(_VAL)
        e = pl.col("_ev")
        vol_known = _ashare_finite(v)
        event_day = vol_known & _ashare_finite(e) & (e != 0)
        ev_sum = pl.when(event_day).then(v).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        ev_cnt = event_day.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        vol_sum = pl.when(vol_known).then(v).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        vol_cnt = vol_known.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        event_vol = pl.when(ev_cnt.is_null() | (ev_cnt <= 0)).then(None).otherwise(ev_sum / ev_cnt)
        base_vol = pl.when(vol_cnt.is_null() | (vol_cnt <= 0)).then(None).otherwise(vol_sum / vol_cnt)
        expr = pl.when(
            event_vol.is_null() | base_vol.is_null() | (base_vol <= 0.0)
        ).then(None).otherwise(event_vol / base_vol)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ashare_limit_event_density":
        # state_machine.AshareLimitEventDensity: rolling Σevent / known_count,
        # where known = (known_status != 0) & event finite; the CURRENT row's
        # unknown known_status → NaN; zero known rows in the window → NaN;
        # known_status missing/None → all-ones mask.  The window literal may
        # arrive as the 2nd or the 3rd input, so the known_status column is
        # whichever input is NOT a literal.
        non_literal_idx = [
            i for i, child in enumerate(node.inputs)
            if child.op not in {"literal"}
        ]
        if not non_literal_idx:
            return None
        ev_l = _compile_child(node, non_literal_idx[0], base, parent_op=op, ctx=ctx, memo=memo)
        if ev_l is None:
            return None
        has_known_col = len(non_literal_idx) >= 2
        if has_known_col:
            known_mask = _compile_child(node, non_literal_idx[1], base, parent_op=op, ctx=ctx, memo=memo)
            if known_mask is None:
                return None
            ev_l = ev_l.join(known_mask.rename({_VAL: "_ks"}), on=[_TS, _INST], how="left")
        w = _window_int(node, default=20)
        ev = pl.col(_VAL)
        ev_known = _ashare_finite(ev)
        if has_known_col:
            ks = pl.col("_ks")
            current_known = _ashare_finite(ks)
            # pandas: kv[chunk] != 0 — a NaN known_status inside the window
            # still compares != 0 (True in NumPy), so it counts as known; only
            # the CURRENT row's NaN ks gates the output to NaN.
            known = ev_known & ~((ks == 0.0) & _ashare_finite(ks))
        else:
            # no known_status column: the mask is all-ones — every current row
            # outputs (even one whose event is NaN, which just counts as 0),
            # and a window day counts as known only when its event is finite.
            current_known = pl.lit(True)
            known = ev_known
        ev_sum = (
            pl.when(known).then(pl.when(ev_known).then(ev).otherwise(0.0))
            .rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        )
        known_cnt = known.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        expr = pl.when(~current_known).then(None).when(
            known_cnt.is_null() | (known_cnt <= 0)
        ).then(None).otherwise(ev_sum / known_cnt)
        return ev_l.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ashare_limit_up_streak":
        # state_machine.AshareLimitUpStreak: consecutive close-at-limit days
        # (close >= high_limit*(1-tol), RELATIVE tolerance) ending at the
        # current row.  valid_trade is the strict {0,1,NaN} TradableBool: a
        # NaN/unknown/suspended (0) row BREAKS the run and outputs NaN; a
        # valid non-limit day outputs 0.  Run-length = cumulative hits within
        # the current run block (block id = cumulative break count).
        if len(node.inputs) < 2:
            return None
        close_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        limit_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        valid_l = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo) if len(node.inputs) >= 3 and node.inputs[2].op not in {"literal"} else None
        if close_l is None or limit_l is None:
            return None
        if valid_l is not None:
            joined = (
                close_l.join(limit_l.rename({_VAL: "_lim"}), on=[_TS, _INST], how="left")
                .join(valid_l.rename({_VAL: "_vt"}), on=[_TS, _INST], how="left")
            )
        else:
            joined = close_l.join(limit_l.rename({_VAL: "_lim"}), on=[_TS, _INST], how="left")
        tol = _ashare_tol(node)
        c = pl.col(_VAL)
        lim = pl.col("_lim")
        price_known = _ashare_finite(c) & _ashare_finite(lim)
        if valid_l is not None:
            vt = pl.col("_vt")
            # pandas _tradeable: strictly {0,1,NaN} — 1 = tradeable; 0 = a
            # suspended day (breaks the run, output NaN); NaN = unknown break.
            tradeable = _ashare_finite(vt) & (vt == 1.0)
            # A row BREAKS the run only when it is non-tradeable or its price
            # inputs are missing (pandas ``_consecutive_streak``).  A VALID
            # non-limit day stays inside the block contributing 0 — it is not
            # a break, so the streak resets to 0 instead of NaN.
            broken = (~tradeable) | (~price_known)
            ok = tradeable & price_known & (c >= lim * (1.0 - tol))
        else:
            broken = ~price_known
            ok = price_known & (c >= lim * (1.0 - tol))
        block = (~ok).cast(pl.Int64).cum_sum().over(_INST, order_by=_TS)
        streak = ok.cast(pl.Float64).cum_sum().over(_INST, block, order_by=_TS)
        expr = pl.when(broken).then(None).otherwise(streak)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    # =====================================================================
    # wave3c vol/valuation window statistics family (2026-09-08) — pure
    # native polars Expr branches for the wave1_volregime / wave1_valuation /
    # wave1_cs_momentum pandas references (vv1_* / vr1_* / val1_* / vax_*).
    # Every trailing window is bounded by ``.over(_INST, order_by=_TS)`` and
    # skips non-finite window rows exactly like the pandas kernels (a NaN row
    # neither counts toward min_periods nor contributes a sample).  Std is
    # ddof=1 (pandas rolling default); compressed-finite-sequence semantics
    # (fractional_share / long_short_vol_beta / vol_of_vol-of-compressed) use
    # trailing-list + list.eval — still pure Expr, no Python UDF.  三方 parity
    # 见 tests/backend_parity/test_volval_wave3_parity.py.
    # =====================================================================

    def _vv_int_param(node_: PlanNode, key: str, pos: int, default: int) -> int:
        # positional literal wins over attrs (the DSL passes ints positionally),
        # then attrs, then the pandas reference default.
        v = _literal_value(node_, pos)
        if v is not None:
            return int(v)
        return _int_attr(node_, key, default=default)

    def _vv_finite_col() -> pl.Expr:
        return pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL)).otherwise(None)

    # ---- helper: compressed finite trailing list (period = w rows) --------
    def _vv_trailing_list(inner_lf: "pl.LazyFrame", col: str, w: int, *, unbounded: bool = False) -> "pl.LazyFrame":
        """给 inner_lf 追加 ``col`` 的 trailing/unbounded 行列表列 ``_vv_lst``。"""
        period = f"{max(2 ** 40, w)}i" if unbounded else f"{w}i"
        return inner_lf.with_columns(
            pl.col(col).implode()
            .rolling(index_column="_vv_i", period=period, closed="right")
            .over(_INST)
            .alias("_vv_lst")
        )

    def _vv_in_list_std(w: int) -> pl.Expr:
        """Trailing-w std(ddof=1) INSIDE a compressed finite list (per position)."""
        x = pl.element()
        cs = x.cum_sum()
        cs2 = (x * x).cum_sum()
        s = cs - cs.shift(w).fill_null(0.0)
        s2 = cs2 - cs2.shift(w).fill_null(0.0)
        cnt = x.is_not_null().cast(pl.Int64).cum_sum() - x.is_not_null().cast(pl.Int64).cum_sum().shift(w).fill_null(0)
        n = cnt.cast(pl.Float64)
        var = (s2 - s * s / n) / (n - 1.0)
        return pl.when(cnt < w).then(None).otherwise(var.sqrt())

    def _vv_list_std_of(col_expr: pl.Expr) -> pl.Expr:
        """std(ddof=1) of a whole compressed list (single value)."""
        return col_expr.list.eval(pl.element().std()).list.first()

    def _vv_list_len(col_expr: pl.Expr) -> pl.Expr:
        return col_expr.list.len()

    if op == "vv1_vol_of_vol":
        # wave1_volregime: std(|return|) over the trailing window, ddof=1,
        # finite count >= min_periods (mp > w is rejected by the reference —
        # the branch mirrors that with a fail-closed gate).
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        mp = _vv_int_param(node, "min_periods", 1, 5)
        if mp > w:
            return None
        r = _vv_finite_col().abs()
        expr = r.rolling_std(window_size=w, min_samples=mp, ddof=1).over(_INST, order_by=_TS)
        return inner.with_columns(expr.alias(_VAL))

    if op == "vv1_downside_vol_share":
        # std(negative ret) / (std(neg) + std(pos)) over the trailing window;
        # both sides need >= 3 finite samples; den <= 1e-12 -> NaN.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=40)
        mp = _vv_int_param(node, "min_periods", 1, 10)
        if mp > w:
            return None
        v = _vv_finite_col()
        dn = pl.when(v < 0).then(v).otherwise(None)
        up = pl.when(v > 0).then(v).otherwise(None)
        sd = dn.rolling_std(window_size=w, min_samples=3, ddof=1).over(_INST, order_by=_TS)
        su = up.rolling_std(window_size=w, min_samples=3, ddof=1).over(_INST, order_by=_TS)
        cnt_finite = v.is_not_null().cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        den = sd + su
        expr = pl.when(cnt_finite.is_null() | (cnt_finite < mp)).then(None).otherwise(
            pl.when(den.is_null() | (den <= 1e-12)).then(None).otherwise(sd / den)
        )
        return inner.with_columns(expr.alias(_VAL))

    if op == "vv1_fractional_share":
        # 1 - min(1, (std(first-sw finite)^2 / std(all finite)^2)) over the
        # trailing long window — COMPRESSED finite positions (list semantics).
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        sw = _vv_int_param(node, "short_window", 0, 5)
        lw = _vv_int_param(node, "long_window", 1, 60)
        mp = _vv_int_param(node, "min_periods", 2, 15)
        if sw >= lw:
            return None
        inner = inner.with_columns(
            pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL)).otherwise(None).alias("_vv_san"),
            pl.int_range(pl.len()).over(_INST, order_by=_TS).alias("_vv_i"),
        )
        comp = (
            _vv_trailing_list(inner, "_vv_san", lw)
            .select(_TS, _INST, "_vv_lst")
            .with_columns(pl.col("_vv_lst").list.drop_nulls())
        )
        lst_col = pl.col("_vv_lst")
        n = lst_col.list.len()
        full = lst_col.list.eval(pl.element().std()).list.first()
        s_vol = lst_col.list.eval(pl.element().head(sw).std()).list.first()
        first_len = lst_col.list.eval(pl.element().head(sw).len()).list.first()
        rest_len = lst_col.list.eval(pl.element().slice(sw).len()).list.first()
        raw = 1.0 - pl.when(full.is_null() | (full <= 1e-12)).then(None).otherwise(
            pl.when(s_vol.is_null()).then(None).otherwise(
                (s_vol * s_vol / (full * full)).clip(0.0, 1.0)
            )
        )
        expr = (
            pl.when(n.is_null() | (n < mp)).then(None)
            .when(first_len.is_null() | (first_len < 2) | rest_len.is_null() | (rest_len < 2))
            .then(None)
            .otherwise(raw)
        )
        return comp.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vv1_long_short_vol_beta":
        # b = cov(sv, lv)/var(sv) with lv CONSTANT (all-rows long std) -> cov=0
        # -> b = 1e-12; out = 1e-12 * mean(trailing-sw std of compressed vals)
        # / std(all compressed vals).  sv positions i in [sw, n) (0-based).
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        sw = _vv_int_param(node, "short_window", 0, 5)
        lw = _vv_int_param(node, "long_window", 1, 60)
        mp = _vv_int_param(node, "min_periods", 2, 15)
        if sw >= lw:
            return None
        inner = inner.with_columns(
            pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL)).otherwise(None).alias("_vv_san"),
            pl.int_range(pl.len()).over(_INST, order_by=_TS).alias("_vv_i"),
        )
        comp_lf = (
            _vv_trailing_list(inner, "_vv_san", lw)
            .select(_TS, _INST, "_vv_lst")
            .with_columns(pl.col("_vv_lst").list.drop_nulls())
        )
        lst_col = pl.col("_vv_lst")
        n = lst_col.list.len()
        lv_full = lst_col.list.eval(pl.element().std()).list.first()
        svs = lst_col.list.eval(_vv_in_list_std(sw)).list.drop_nulls()
        # pandas: s_vols built for i in [sw, n) -> the FIRST compressed
        # position (i = 0) is excluded by the n<sw gate; drop_nulls also
        # removes partial-window positions.
        mean_sv = svs.list.eval(pl.element().mean()).list.first()
        expr = (
            pl.when(n.is_null() | (n < mp) | (n < sw + 2)).then(None)
            .when(lv_full.is_null() | (lv_full <= 1e-12)).then(None)
            .when(mean_sv.is_null()).then(None)
            .otherwise(1e-12 * (mean_sv / lv_full))
        )
        return comp_lf.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vv1_vol_acceleration":
        # pandas reference: trend starts 0.0 and the update line keeps it at
        # 0.0 forever (``trend if isfinite(trend) else ...`` — 0.0 IS finite),
        # so diffv = (vol - 0)/(|0| + 1e-12) == vol * 1e12.  vol = trailing
        # vol_window std(ddof=1) with >= min_periods finite samples.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        vw = _vv_int_param(node, "vol_window", 0, 10)
        mp = _vv_int_param(node, "min_periods", 2, 4)
        v = _vv_finite_col()
        vol = v.rolling_std(window_size=vw, min_samples=mp, ddof=1).over(_INST, order_by=_TS)
        expr = pl.when(vol.is_null()).then(None).otherwise(vol * 1e12)
        return inner.with_columns(expr.alias(_VAL))

    if op == "vv1_dispersion_vol":
        # cross-sectional std(ddof=0) of per-instrument trailing vol_window
        # std(ddof=1) (>= 3 finite per instrument); needs >= min_breadth
        # instrument vols; broadcast to every cell of the row.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        vw = _vv_int_param(node, "vol_window", 0, 10)
        mb = _vv_int_param(node, "min_breadth", 1, 10)
        v = _vv_finite_col()
        vol = v.rolling_std(window_size=vw, min_samples=3, ddof=1).over(_INST, order_by=_TS)
        mid = inner.with_columns(vol.alias("_vv_vol"))
        disp = pl.col("_vv_vol").std(ddof=0).over(_TS, order_by=_INST)
        cnt = pl.col("_vv_vol").is_not_null().sum().over(_TS, order_by=_INST)
        expr = pl.when(cnt.is_null() | (cnt < mb)).then(None).otherwise(disp)
        return mid.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vv1_regime_change_ratio":
        # std(short)/std(long); outside [1-band, 1+band] -> |ratio-1|/band,
        # else 0.0; long std needs >= mp finite, short std >= 2 finite.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        sw = _vv_int_param(node, "short_window", 0, 5)
        lw = _vv_int_param(node, "long_window", 1, 60)
        mp = _vv_int_param(node, "min_periods", 3, 4)
        if sw >= lw:
            return None
        if mp > lw:
            return None
        bnd_raw = _literal_value(node, 2)
        bnd = float(bnd_raw) if bnd_raw is not None else _float_attr(node, "band", default=0.5)
        if bnd <= 0:
            return None
        v = _vv_finite_col()
        s_long = v.rolling_std(window_size=lw, min_samples=mp, ddof=1).over(_INST, order_by=_TS)
        s_short = v.rolling_std(window_size=sw, min_samples=2, ddof=1).over(_INST, order_by=_TS)
        ratio = pl.when(s_long.is_null() | (s_long <= 1e-12)).then(None).otherwise(s_short / s_long)
        expr = pl.when(ratio.is_null()).then(None).otherwise(
            pl.when((ratio > 1.0 + bnd) | (ratio < 1.0 - bnd))
            .then((ratio - 1.0).abs() / bnd)
            .otherwise(0.0)
        )
        return inner.with_columns(expr.alias(_VAL))

    if op == "vv1_vol_level_score":
        # sigmoid(3 * (cur/hmean - 1)): cur = trailing short-window std
        # (>= 2 finite) of raw positions; hmean = mean(|finite|) over the
        # short_window-1 rows ENDING AT row-1 (exclusive of current), needs
        # >= min_periods finite and hmean > 0.  The pandas reference requires
        # mp finite INSIDE a (sw-1)-row slice — when mp > sw-1 the output is
        # always NaN, mirrored by the explicit cnt gate below.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        sw = _vv_int_param(node, "short_window", 0, 5)
        hw = _vv_int_param(node, "history_window", 1, 120)
        mp = _vv_int_param(node, "min_periods", 2, 20)
        if sw >= hw:
            return None
        if mp > hw:
            return None
        v = _vv_finite_col()
        cur = v.rolling_std(window_size=sw, min_samples=2, ddof=1).over(_INST, order_by=_TS)
        # pandas hist_slice = rv[row - sw : row] → exactly sw rows ending at
        # row-1 (exclusive of current), finite count >= mp, mean of |x|.
        hist_w = sw
        hist = (
            v.abs().rolling_mean(
                window_size=hist_w, min_samples=min(mp, hist_w)
            ).over(_INST, order_by=_TS).shift(1)
            .over(_INST, order_by=_TS)
        )
        cnt_h = (
            v.is_not_null().cast(pl.Float64).rolling_sum(
                window_size=hist_w, min_samples=1
            ).over(_INST, order_by=_TS).shift(1)
            .over(_INST, order_by=_TS)
        )
        expr = pl.when(cur.is_null() | hist.is_null() | (hist <= 0)).then(None).otherwise(
            pl.when(cnt_h.is_null() | (cnt_h < mp)).then(None).otherwise(
                1.0 / (1.0 + (-((cur / hist) - 1.0) * 3.0).exp())
            )
        )
        return inner.with_columns(expr.alias(_VAL))

    if op == "vr1_range_everage":
        # mean(|high-low|/close) / std(close_return) over the joint-finite
        # window (|close| > 0); close std needs >= 2 samples (ddof=1) and
        # s <= 1e-12 -> NaN; finite rows >= mp.
        if len(node.inputs) < 4:
            return None
        h_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        l_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        c_l = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        r_l = _compile_child(node, 3, base, parent_op=op, ctx=ctx, memo=memo)
        if h_l is None or l_l is None or c_l is None or r_l is None:
            return None
        w = _window_int(node, default=20)
        mp = _vv_int_param(node, "min_periods", 1, 5)
        if mp > w:
            return None
        joined = (
            h_l.join(l_l.rename({_VAL: "_lo"}), on=[_TS, _INST], how="left")
            .join(c_l.rename({_VAL: "_cl"}), on=[_TS, _INST], how="left")
            .join(r_l.rename({_VAL: "_rr"}), on=[_TS, _INST], how="left")
        )
        h, lo, cl, rr = pl.col(_VAL), pl.col("_lo"), pl.col("_cl"), pl.col("_rr")
        ok = h.is_finite() & lo.is_finite() & cl.is_finite() & rr.is_finite() & (cl.abs() > 0)
        range_v = pl.when(ok).then((h - lo).abs() / cl).otherwise(None)
        ret_v = pl.when(ok).then(rr).otherwise(None)
        m = range_v.rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        s = ret_v.rolling_std(window_size=w, min_samples=2, ddof=1).over(_INST, order_by=_TS)
        cnt = ok.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        expr = (
            pl.when(cnt.is_null() | (cnt < mp)).then(None)
            .when(s.is_null() | (s <= 1e-12)).then(None)
            .otherwise(m / s)
        )
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vr1_parkinson_close_scale":
        # sqrt(mean((high-low)/close)^2) / std(1-period close return) over the
        # joint-finite (close > 0) window; the close return set is the COMPRESSED
        # diff of the compressed finite closes (pandas cc_ret = diff(cc[ok])).
        if len(node.inputs) < 3:
            return None
        h_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        l_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        c_l = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        if h_l is None or l_l is None or c_l is None:
            return None
        w = _window_int(node, default=20)
        mp = _vv_int_param(node, "min_periods", 1, 5)
        if mp > w:
            return None
        joined = (
            h_l.join(l_l.rename({_VAL: "_lo"}), on=[_TS, _INST], how="left")
            .join(c_l.rename({_VAL: "_cl"}), on=[_TS, _INST], how="left")
        )
        h, lo, cl = pl.col(_VAL), pl.col("_lo"), pl.col("_cl")
        ok = h.is_finite() & lo.is_finite() & cl.is_finite() & (cl > 0)
        park_row = pl.when(ok).then((h - lo) / cl).otherwise(None)
        park = (
            (park_row * park_row).rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        ).sqrt()
        cc = pl.when(ok).then(cl).otherwise(None)
        # compressed closes -> 1-period returns -> std over the compressed list
        base2 = joined.with_columns(
            cc.alias("_vv_san"),
            pl.int_range(pl.len()).over(_INST, order_by=_TS).alias("_vv_i"),
        )
        cc_ret_std = (
            _vv_trailing_list(base2, "_vv_san", w)
            .select(_TS, _INST, "_vv_lst")
            .with_columns(pl.col("_vv_lst").list.drop_nulls())
            .with_columns(
                # pandas cc_ret = diff(cc[ok]) / cc[ok][:-1] (relative returns
                # between consecutive COMPRESSED closes).
                pl.col("_vv_lst").list.eval(
                    (
                        pl.element().diff().drop_nulls()
                        / pl.element().shift(1).drop_nulls()
                    ).std()
                ).list.first().alias("_rr_std")
            )
        )
        cnt = ok.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        out_lf = base2.with_columns(cnt.alias("_vv_cnt")).join(
            cc_ret_std.select(_TS, _INST, "_rr_std"), on=[_TS, _INST], how="left"
        )
        park_val = (
            (park_row * park_row).rolling_mean(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        ).sqrt()
        expr = (
            pl.when(pl.col("_vv_cnt").is_null() | (pl.col("_vv_cnt") < mp)).then(None)
            .when(park_val.is_null()).then(None)
            .when(pl.col("_rr_std").is_null() | (pl.col("_rr_std") <= 1e-12)).then(None)
            .otherwise(park_val / pl.col("_rr_std"))
        )
        return out_lf.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vr1_rogers_satchell":
        # sqrt(mean(max(ln(H/C)ln(H/O) + ln(L/C)ln(L/O), 0))) — pandas clamps
        # the MEAN at 0 before the sqrt; joint finite (o>0, c>0) window.
        if len(node.inputs) < 4:
            return None
        o_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        h_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        l_l = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        c_l = _compile_child(node, 3, base, parent_op=op, ctx=ctx, memo=memo)
        if o_l is None or h_l is None or l_l is None or c_l is None:
            return None
        w = _window_int(node, default=20)
        mp = _vv_int_param(node, "min_periods", 1, 5)
        if mp > w:
            return None
        joined = (
            o_l.join(h_l.rename({_VAL: "_hi"}), on=[_TS, _INST], how="left")
            .join(l_l.rename({_VAL: "_lo"}), on=[_TS, _INST], how="left")
            .join(c_l.rename({_VAL: "_cl"}), on=[_TS, _INST], how="left")
        )
        o, h, lo, cl = pl.col(_VAL), pl.col("_hi"), pl.col("_lo"), pl.col("_cl")
        ok = o.is_finite() & h.is_finite() & lo.is_finite() & cl.is_finite() & (o > 0) & (cl > 0)
        pos_ok = ok & (h > 0) & (lo > 0)
        terms = (
            pl.when(pos_ok).then(
                (h / cl).log() * (h / o).log() + (lo / cl).log() * (lo / o).log()
            ).otherwise(None)
        )
        mean_t = terms.rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        expr = pl.when(mean_t.is_null()).then(None).otherwise(
            pl.when(mean_t < 0).then(0.0).otherwise(mean_t.sqrt())
        )
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vr1_garman_klass_ext":
        # sqrt(mean(0.5 ln(H/L)^2 - (2ln2-1) ln(C/O)^2)) with a mean>=0 clamp;
        # joint finite (o>0, c>0) window.
        if len(node.inputs) < 4:
            return None
        o_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        h_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        l_l = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        c_l = _compile_child(node, 3, base, parent_op=op, ctx=ctx, memo=memo)
        if o_l is None or h_l is None or l_l is None or c_l is None:
            return None
        w = _window_int(node, default=20)
        mp = _vv_int_param(node, "min_periods", 1, 5)
        if mp > w:
            return None
        joined = (
            o_l.join(h_l.rename({_VAL: "_hi"}), on=[_TS, _INST], how="left")
            .join(l_l.rename({_VAL: "_lo"}), on=[_TS, _INST], how="left")
            .join(c_l.rename({_VAL: "_cl"}), on=[_TS, _INST], how="left")
        )
        o, h, lo, cl = pl.col(_VAL), pl.col("_hi"), pl.col("_lo"), pl.col("_cl")
        ok = o.is_finite() & h.is_finite() & lo.is_finite() & cl.is_finite() & (o > 0) & (cl > 0)
        pos_ok = ok & (h > 0) & (lo > 0)
        v = pl.when(pos_ok).then(
            0.5 * ((h / lo).log() ** 2) - (2.0 * np.log(2.0) - 1.0) * ((cl / o).log() ** 2)
        ).otherwise(None)
        m = v.rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        expr = pl.when(m.is_null()).then(None).otherwise(
            pl.when(m < 0).then(0.0).otherwise(m.sqrt())
        )
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vr1_range_to_close_eff":
        # mean(|close-open|/(high-low)) over rows where the range is > 0 —
        # pandas skips the WHOLE window if ANY range <= 0 (np.any gate).
        if len(node.inputs) < 4:
            return None
        o_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        h_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        l_l = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        c_l = _compile_child(node, 3, base, parent_op=op, ctx=ctx, memo=memo)
        if o_l is None or h_l is None or l_l is None or c_l is None:
            return None
        w = _window_int(node, default=20)
        mp = _vv_int_param(node, "min_periods", 1, 5)
        if mp > w:
            return None
        joined = (
            o_l.join(h_l.rename({_VAL: "_hi"}), on=[_TS, _INST], how="left")
            .join(l_l.rename({_VAL: "_lo"}), on=[_TS, _INST], how="left")
            .join(c_l.rename({_VAL: "_cl"}), on=[_TS, _INST], how="left")
        )
        o, h, lo, cl = pl.col(_VAL), pl.col("_hi"), pl.col("_lo"), pl.col("_cl")
        ok = o.is_finite() & h.is_finite() & lo.is_finite() & cl.is_finite()
        hl = pl.when(ok).then(h - lo).otherwise(None)
        bad_range = (hl <= 0).fill_null(False)
        any_bad = bad_range.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        eff = pl.when(ok & (hl > 0)).then((cl - o).abs() / hl).otherwise(None)
        m = eff.rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        cnt_ok = ok.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        expr = (
            pl.when(cnt_ok.is_null() | (cnt_ok < mp)).then(None)
            .when(any_bad > 0).then(None)
            .otherwise(m)
        )
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vr1_ewma_range_vol":
        # acc_{t} = coef * acc_{t-1} + rng_t for valid rows, coef * acc_{t-1}
        # for invalid rows (after the first valid row) — a full-history decay
        # with NO window.  Closed form: acc_t = coef^t * Σ_{k<=t} rng_k *
        # coef^-k (rng=0 at invalid rows), NaN before the first valid row.
        if len(node.inputs) < 4:
            return None
        o_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        h_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        l_l = _compile_child(node, 2, base, parent_op=op, ctx=ctx, memo=memo)
        c_l = _compile_child(node, 3, base, parent_op=op, ctx=ctx, memo=memo)
        if o_l is None or h_l is None or l_l is None or c_l is None:
            return None
        ds_raw = _literal_value(node, 3)
        ds = float(ds_raw) if ds_raw is not None else _float_attr(node, "decay_scale", default=20.0)
        if ds < 1.0:
            return None
        joined = (
            o_l.join(h_l.rename({_VAL: "_hi"}), on=[_TS, _INST], how="left")
            .join(l_l.rename({_VAL: "_lo"}), on=[_TS, _INST], how="left")
            .join(c_l.rename({_VAL: "_cl"}), on=[_TS, _INST], how="left")
        )
        o, h, lo, cl = pl.col(_VAL), pl.col("_hi"), pl.col("_lo"), pl.col("_cl")
        ok = o.is_finite() & h.is_finite() & lo.is_finite() & cl.is_finite() & (cl > 0)
        rng = pl.when(ok).then((h - lo) / cl).otherwise(0.0)
        t = pl.int_range(pl.len()).over(_INST, order_by=_TS).cast(pl.Float64)
        ln_c = -1.0 / float(ds)
        acc = (ln_c * t).exp() * (rng * (-ln_c * t).exp()).cum_sum().over(_INST, order_by=_TS)
        first_valid = ok.fill_null(False).cast(pl.Float64).cum_max().over(_INST, order_by=_TS)
        expr = pl.when(first_valid > 0).then(acc).otherwise(None)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "val1_valuation_z_own":
        # (cur - mean(hist+[cur])) / std(hist+[cur]) over the trailing
        # history_window frame (hist = hw-1 rows EXCLUDING current + cur);
        # ddof=1, std <= 1e-12 -> 0.0, hist finite >= mp, cur finite.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        hw = _vv_int_param(node, "history_window", 0, 60)
        mp = _vv_int_param(node, "min_periods", 1, 10)
        if mp > hw:
            return None
        v = _vv_finite_col()
        hist_w = max(hw - 1, 1)
        hist_ok = v.is_not_null().cast(pl.Float64)
        # pandas hist = the hw-1 rows ENDING AT row-1 (exclusive of current);
        # the current value is added back exactly once below.
        sh = ((v * v).rolling_sum(window_size=hist_w, min_samples=1).over(_INST, order_by=_TS).shift(1)
              .over(_INST, order_by=_TS))
        sm = (v.rolling_sum(window_size=hist_w, min_samples=1).over(_INST, order_by=_TS).shift(1)
              .over(_INST, order_by=_TS))
        cnt = (hist_ok.rolling_sum(window_size=hist_w, min_samples=1).over(_INST, order_by=_TS).shift(1)
               .over(_INST, order_by=_TS))
        cur = pl.col(_VAL)
        cur_ok = pl.col(_VAL).is_finite()
        n = cnt + 1.0
        m = (sm + cur) / n
        var = (sh + cur * cur - m * (sm + cur)) / (n - 1.0)
        std = pl.when(var >= 0).then(var.sqrt()).otherwise(None)
        expr = (
            pl.when(cur_ok).then(
                pl.when(cnt.is_null() | (cnt < mp)).then(None)
                .when(std.is_null() | (std <= 1e-12)).then(0.0)
                .otherwise((cur - m) / std)
            ).otherwise(None)
        )
        return inner.with_columns(expr.alias(_VAL))

    if op == "val1_valuation_percentile_own":
        # (#[hist < cur] + 0.5 * #[hist == cur]) / #hist over the trailing
        # history_window-1 rows EXCLUDING current; hist finite >= mp, cur finite.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        hw = _vv_int_param(node, "history_window", 0, 60)
        mp = _vv_int_param(node, "min_periods", 1, 10)
        if mp > hw:
            return None
        inner = inner.with_columns(
            pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL)).otherwise(None).alias("_vv_san"),
            pl.col(_VAL).alias("_vv_cur"),
            pl.int_range(pl.len()).over(_INST, order_by=_TS).alias("_vv_i"),
        )
        prev_lf = (
            _vv_trailing_list(inner, "_vv_san", max(hw - 1, 1))
            .select(_TS, _INST, "_vv_lst", "_vv_cur")
            .with_columns(pl.col("_vv_lst").shift(1).over(_INST, order_by=_TS).alias("_vv_lst"))
            .with_columns(pl.col("_vv_lst").list.drop_nulls())
        )
        cur = pl.col("_vv_cur")
        n = pl.col("_vv_lst").list.len()
        # list.eval cannot reference outer columns — count comparisons on the
        # exploded long form instead (still pure native polars).
        hist_long = prev_lf.explode("_vv_lst")
        agg = hist_long.group_by([_TS, _INST]).agg(
            (pl.col("_vv_lst") < pl.col("_vv_cur")).sum().alias("_vv_less"),
            (pl.col("_vv_lst") == pl.col("_vv_cur")).sum().alias("_vv_eq"),
            pl.col("_vv_lst").is_not_null().sum().alias("_vv_n"),
        )
        out_lf = prev_lf.join(agg, on=[_TS, _INST], how="left")
        less = pl.col("_vv_less")
        eq = pl.col("_vv_eq")
        n = pl.col("_vv_n")
        expr = (
            pl.when(cur.is_finite()).then(
                pl.when(n.is_null() | (n < mp)).then(None).otherwise((less + 0.5 * eq) / n.cast(pl.Float64))
            ).otherwise(None)
        )
        return out_lf.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "val1_earnings_yield_ma_diff":
        # cur - mean(past baseline_window rows EXCLUDING current, finite >= mp);
        # cur finite.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _vv_int_param(node, "baseline_window", 0, 12)
        mp = _vv_int_param(node, "min_periods", 1, 4)
        if mp > w:
            return None
        v = _vv_finite_col()
        base = (
            v.rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS).shift(1)
            .over(_INST, order_by=_TS)
        )
        cur = pl.col(_VAL)
        expr = pl.when(cur.is_finite()).then(
            pl.when(base.is_null()).then(None).otherwise(cur - base)
        ).otherwise(None)
        return inner.with_columns(expr.alias(_VAL))

    if op == "val1_valuations_lag_component":
        # (mean(last fw finite) - mean(last sw finite)) / std(last sw finite)
        # over the EXPANDING finite-value cache (compressed: NaN rows are never
        # appended but still output); std <= 1e-12 -> 0.0; len(cache) >= sw.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        fw = _vv_int_param(node, "fast_window", 0, 5)
        sw = _vv_int_param(node, "slow_window", 1, 20)
        mp = _vv_int_param(node, "min_periods", 2, 4)
        if fw >= sw:
            return None
        inner = inner.with_columns(
            pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL)).otherwise(None).alias("_vv_san"),
            pl.int_range(pl.len()).over(_INST, order_by=_TS).alias("_vv_i"),
        )
        comp_lf = (
            _vv_trailing_list(inner, "_vv_san", sw, unbounded=True)
            .select(_TS, _INST, "_vv_lst")
            .with_columns(pl.col("_vv_lst").list.drop_nulls())
        )
        lst_col = pl.col("_vv_lst")
        n = lst_col.list.len()
        slow_m = lst_col.list.eval(pl.element().tail(sw).mean()).list.first()
        fast_m = lst_col.list.eval(pl.element().tail(fw).mean()).list.first()
        s = lst_col.list.eval(pl.element().tail(sw).std()).list.first()
        expr = (
            pl.when(n.is_null() | (n < sw) | (n < mp)).then(None)
            .when(s.is_null() | (s <= 1e-12)).then(0.0)
            .otherwise((fast_m - slow_m) / s)
        )
        return comp_lf.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "val1_earnings_yield_slope":
        # OLS slope of the COMPRESSED finite values on t=0..n-1, * sqrt(n);
        # n >= mp; t-spread den <= 1e-12 -> NaN; the CURRENT row must be
        # finite (the pandas reference gates on ``np.isfinite(yv[row])``).
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=12)
        mp = _vv_int_param(node, "min_periods", 1, 5)
        if mp > w:
            return None
        inner = inner.with_columns(
            pl.when(pl.col(_VAL).is_finite()).then(pl.col(_VAL)).otherwise(None).alias("_vv_san"),
            pl.col(_VAL).is_finite().alias("_vv_cur_ok"),
            pl.int_range(pl.len()).over(_INST, order_by=_TS).alias("_vv_i"),
        )
        comp_lf = (
            _vv_trailing_list(inner, "_vv_san", w)
            .select(_TS, _INST, "_vv_lst", "_vv_cur_ok")
            .with_columns(pl.col("_vv_lst").list.drop_nulls())
        )
        lst_col = pl.col("_vv_lst")
        n = lst_col.list.len().cast(pl.Float64)
        # Σ t*y via in-list index transform; t = int_range over the list.
        sum_ty = lst_col.list.eval(
            (pl.element() * pl.int_range(pl.len()).cast(pl.Float64)).sum()
        ).list.first()
        mean_y = lst_col.list.eval(pl.element().mean()).list.first()
        t_mean = (n - 1.0) / 2.0
        den = n * (n * n - 1.0) / 12.0
        slope = pl.when(den.is_null() | (den <= 1e-12)).then(None).otherwise(
            (sum_ty - n * t_mean * mean_y) / den
        )
        expr = (
            pl.when(pl.col("_vv_cur_ok").fill_null(False)).then(
                pl.when(n.is_null() | (n < mp)).then(None).otherwise(slope * n.sqrt())
            ).otherwise(None)
        )
        return comp_lf.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "vax_liquidity_penalty_exposure":
        # mean(amihud) over the trailing window (finite AND >= 0 rows only);
        # count >= mp.
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        mp = _vv_int_param(node, "min_periods", 1, 5)
        if mp > w:
            return None
        a = pl.when(pl.col(_VAL).is_finite() & (pl.col(_VAL) >= 0)).then(pl.col(_VAL)).otherwise(None)
        m = a.rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        return inner.with_columns(m.alias(_VAL))

    if op == "vax_ret_per_liquidity_unit":
        # (Π(1+r) - 1) / mean(amount) over rows where BOTH ret and amount are
        # finite and amount > 0; pairs >= mp; mean(amount) > 0.
        if len(node.inputs) < 2:
            return None
        r_l = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        a_l = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if r_l is None or a_l is None:
            return None
        w = _window_int(node, default=20)
        mp = _vv_int_param(node, "min_periods", 1, 10)
        if mp > w:
            return None
        joined = r_l.join(a_l.rename({_VAL: "_amt"}), on=[_TS, _INST], how="left")
        r, amt = pl.col(_VAL), pl.col("_amt")
        ok = r.is_finite() & amt.is_finite() & (amt > 0)
        # Π(1+r) over COMPRESSED valid pairs (pandas: r[ok]); guarded ln.
        base2 = joined.with_columns(
            pl.when(ok).then(r).otherwise(None).alias("_vv_san"),
            pl.int_range(pl.len()).over(_INST, order_by=_TS).alias("_vv_i"),
        )
        prod_lf = (
            _vv_trailing_list(base2, "_vv_san", w)
            .select(_TS, _INST, "_vv_lst")
            .with_columns(pl.col("_vv_lst").list.drop_nulls())
            .with_columns(
                pl.col("_vv_lst").list.eval(
                    (1.0 + pl.element()).log().sum()
                ).list.first().alias("_vv_logprod")
            )
        )
        log_prod = pl.col("_vv_logprod")
        prod = pl.when(log_prod.is_null()).then(None).otherwise(log_prod.exp() - 1.0)
        am = pl.when(ok).then(amt).otherwise(None).rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        cnt = ok.cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over(_INST, order_by=_TS)
        out_lf = base2.with_columns(
            am.alias("_vv_am"),
            cnt.alias("_vv_cnt"),
        ).join(prod_lf.select(_TS, _INST, "_vv_logprod"), on=[_TS, _INST], how="left")
        expr = (
            pl.when(pl.col("_vv_cnt").is_null() | (pl.col("_vv_cnt") < mp)).then(None)
            .when(pl.col("_vv_am").is_null() | (pl.col("_vv_am") <= 0)).then(None)
            .otherwise(prod / pl.col("_vv_am"))
        )
        return out_lf.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    # ------------------------------------------------------------------
    # wave3f ts2: prior-extreme / range / consolidation / liquidity-beta
    # family (2026-09-08).  Pandas authorities:
    #   * price_volume/technical_extensions.py (prev_high family,
    #     ts_days_since_extreme, ts_range_expansion)
    #   * price_volume/structure_patterns_v2.py L301 (ts_consolidation_width)
    #   * cross_section/peer_ops.py _rolling_regression +
    #     ts_model/polars_regression.py _pairwise_rolling (liquidity betas)
    # All trailing windows REPLACE ±Inf with NULL before the window: the
    # pandas rolling machinery (aggregations AND .apply) silently treats Inf
    # as missing and excludes it from the min_periods count.
    # ------------------------------------------------------------------

    if op in {
        "ts_prev_high",
        "ts_prev_low",
        "ts_distance_to_high",
        "ts_distance_to_low",
        "ts_breakout_high",
        "ts_breakdown_low",
        "ts_new_high",
        "ts_new_low",
        "ts_channel_position",
    }:
        if len(node.inputs) < 1:
            return None
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        # The WINDOW side (prev_hi/prev_lo baselines) drops ±Inf like the
        # pandas rolling machinery.  The CURRENT bar keeps ±Inf as a real
        # value (pandas: x.notna() is False only for NaN, and x / prev with
        # x = ±Inf yields ±Inf / a clipped 0) — only NaN maps to NULL.
        cur = pl.when(pl.col(_VAL).is_nan()).then(None).otherwise(pl.col(_VAL))
        x_win = pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()).then(None).otherwise(pl.col(_VAL))
        inner = inner.with_columns(cur.alias("_x_cur"), x_win.alias("_x_win"))
        shifted = pl.col("_x_win").shift(1).over(_INST, order_by=_TS)
        prev_hi = shifted.rolling_max(window_size=w, min_samples=w).over(_INST, order_by=_TS)
        prev_lo = shifted.rolling_min(window_size=w, min_samples=w).over(_INST, order_by=_TS)

        def _safe_div_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
            # pandas _safe_div: num / den.replace(0, NaN) → 0/NaN denom → NaN.
            # num.is_null() only for NaN current; num = ±Inf flows through.
            return pl.when(
                num.is_null() | den.is_null() | (den == 0)
            ).then(None).otherwise(num / den)

        if op == "ts_prev_high":
            expr = prev_hi
        elif op == "ts_prev_low":
            expr = prev_lo
        elif op == "ts_distance_to_high":
            expr = _safe_div_ratio(pl.col("_x_cur"), prev_hi) - 1.0
        elif op == "ts_distance_to_low":
            expr = _safe_div_ratio(pl.col("_x_cur"), prev_lo) - 1.0
        elif op == "ts_breakout_high":
            expr = _safe_div_ratio(pl.col("_x_cur"), prev_hi) - 1.0
            expr = pl.when(expr.is_null()).then(None).otherwise(expr.clip(lower_bound=0.0))
        elif op == "ts_breakdown_low":
            expr = _safe_div_ratio(prev_lo, pl.col("_x_cur")) - 1.0
            expr = pl.when(expr.is_null()).then(None).otherwise(expr.clip(lower_bound=0.0))
        elif op in {"ts_new_high", "ts_new_low"}:
            # pandas: x.gt(prev).astype(float).where(x.notna() & prev.notna())
            # — a missing (NaN) current value or baseline emits NaN, never 0
            # (review P0-07); a ±Inf current value IS known and compares.
            cur_known = pl.col("_x_cur").is_not_null()
            base_known = prev_hi.is_not_null() if op == "ts_new_high" else prev_lo.is_not_null()
            base = prev_hi if op == "ts_new_high" else prev_lo
            if op == "ts_new_high":
                hit = pl.col("_x_cur") > base
            else:
                hit = pl.col("_x_cur") < base
            expr = pl.when(cur_known & base_known).then(pl.when(hit).then(1.0).otherwise(0.0)).otherwise(None)
        else:  # ts_channel_position
            # pandas: _safe_div(x - lo, hi - lo) — NO 0..1 clamp; a zero
            # channel width (hi == lo) → NaN.
            rng = prev_hi - prev_lo
            num = pl.col("_x_cur") - prev_lo
            expr = pl.when(
                num.is_null() | rng.is_null() | (rng == 0)
            ).then(None).otherwise(num / rng)
        return inner.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op in {"ts_days_since_high", "ts_days_since_low"}:
        # DEFER (wave3f ts2): the pandas reference is an argmax-POSITION
        # semantic over the prior window (``nanargmax`` of the REVERSED
        # window, ties → most recent hit).  A faithful lowering needs the
        # latest in-window row attaining the CURRENT row's window extreme —
        # the hit predicate depends on the OUTPUT row T, so it cannot be
        # precomputed per historical row and rolled up with ``rolling_max``
        # (a windowed aggregate of a T-dependent predicate is not expressible
        # as a pure ``over`` chain).  Keep this operator on the registry
        # bridge (per-inst ``rolling_map`` path, polars_misc_v2._days_since)
        # which reproduces the pandas kernel exactly; the DuckDB SQL emitter
        # has its own exact subquery-chain implementation (R16-061).
        # NOTE: returning None here is the intended hand-off — the caller of
        # ``_compile_polars_impl`` catches the compile failure for
        # registry-tier plans and retries via ``compile_registry_op``
        # (strict-fallback policy); see ``compile_polars_long_lazy``.
        from .polars_registry_bridge import compile_registry_op

        bridge = compile_registry_op(node, base, lambda n, b: _compile_polars(n, b, ctx=ctx, memo=memo))
        if bridge is not None:
            return bridge
        return None

    if op == "ts_range_expansion":
        # technical_extensions._ts_range_expansion:
        # current = high - low; baseline = current.shift(1).rolling(w, mp=w).mean()
        # result = _safe_div(current, baseline) - 1.0.
        if len(node.inputs) < 2:
            return None
        h_in = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        l_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if h_in is None or l_in is None:
            return None
        joined = _join_binary(h_in, l_in)
        w = _window_int(node, default=20)
        h_fin = pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()).then(None).otherwise(pl.col(_VAL))
        l_fin = pl.when(pl.col("_y").is_nan() | pl.col("_y").is_infinite()).then(None).otherwise(pl.col("_y"))
        current = h_fin - l_fin
        baseline = current.shift(1).over(_INST, order_by=_TS).rolling_mean(window_size=w, min_samples=w).over(_INST, order_by=_TS)
        expr = pl.when(
            current.is_null() | baseline.is_null() | (baseline == 0)
        ).then(None).otherwise(current / baseline - 1.0)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ts_consolidation_width":
        # structure_patterns_v2.ts_consolidation_width (NO shift — the window
        # INCLUDES the current bar): high.rolling(w, mp=w).max()
        # - low.rolling(w, mp=w).min().
        if len(node.inputs) < 2:
            return None
        h_in = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        l_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if h_in is None or l_in is None:
            return None
        joined = _join_binary(h_in, l_in)
        w = _window_int(node, default=20)
        h_fin = pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()).then(None).otherwise(pl.col(_VAL))
        l_fin = pl.when(pl.col("_y").is_nan() | pl.col("_y").is_infinite()).then(None).otherwise(pl.col("_y"))
        hmax = h_fin.rolling_max(window_size=w, min_samples=w).over(_INST, order_by=_TS)
        lmin = l_fin.rolling_min(window_size=w, min_samples=w).over(_INST, order_by=_TS)
        expr = pl.when(hmax.is_null() | lmin.is_null()).then(None).otherwise(hmax - lmin)
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op in {"ts_market_liquidity_beta", "ts_industry_liquidity_beta"}:
        # peer_ops._liquidity_beta: regress own_return on liquidity.diff(1)
        # with w = window, mp = max(3, window // 5); pairwise-finite mask;
        # population cov / population var (ddof=0 identity, R35-P0-M12);
        # n > 1 and var > 0 guards (peer_ops var <= _EPS → NaN).
        if len(node.inputs) < 2:
            return None
        y_in = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        x_in = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if y_in is None or x_in is None:
            return None
        joined = _join_binary(y_in, x_in)
        w = _window_int(node, default=60)
        mp = max(3, w // 5)
        y_fin = pl.when(pl.col(_VAL).is_nan() | pl.col(_VAL).is_infinite()).then(None).otherwise(pl.col(_VAL))
        x_raw = pl.when(pl.col("_y").is_nan() | pl.col("_y").is_infinite()).then(None).otherwise(pl.col("_y"))
        x_fin = x_raw.diff().over(_INST, order_by=_TS)
        # PAIRWISE mask FIRST (both sides share the same valid rows) —
        # per-series masks would let the moments drift onto different rows.
        pair = y_fin.is_not_null() & x_fin.is_not_null()
        ym = pl.when(pair).then(y_fin).otherwise(None)
        xm = pl.when(pair).then(x_fin).otherwise(None)
        mean_ab = (ym * xm).rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        mean_a = ym.rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        mean_b = xm.rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        mean_b2 = (xm * xm).rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        n = ym.is_not_null().cast(pl.Float64).rolling_sum(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        pop_cov = mean_ab - mean_a * mean_b
        pop_var = mean_b2 - mean_b * mean_b
        expr = pl.when(
            n.is_null() | (n <= 1) | pop_var.is_null() | (pop_var <= 1e-12)
        ).then(None).otherwise(pop_cov / pop_var)
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
