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

_PREDICATE_OPS: frozenset[str] = frozenset(
    {"is_nan", "is_null", "is_not_null", "is_finite", "is_infinite"}
)
_NAN_PRESERVE_PARENT_OPS: frozenset[str] = frozenset({"maximum", "minimum"})


def _sanitize_nan_for_compute(inner: pl.LazyFrame | None) -> pl.LazyFrame | None:
    """Pandas 数值路径：IEEE NaN 按缺失处理（rolling / coalesce 等）。"""
    if inner is None:
        return None
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
    """编译子节点；非 predicate 父算子在子结果上将 NaN 规范为 NULL。"""
    if index >= len(node.inputs):
        return None
    inner = _compile_polars(node.inputs[index], base, ctx=ctx, memo=memo)
    if inner is None or parent_op in _PREDICATE_OPS or parent_op in _NAN_PRESERVE_PARENT_OPS:
        return inner
    return _sanitize_nan_for_compute(inner)

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
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(op, op)


def _window_int(node: PlanNode, default: int = 3) -> int:
    """从 PlanNode attrs 或 literal 子节点解析滚动窗口（与 DuckDB 共用）。"""
    return _window_spec(node, default=default).size


def _window_spec(node: PlanNode, default: int = 3):
    """完整 WindowSpec（min_periods / ddof / closed）。"""
    from backend.plan_params import window_spec_from_plan_node
    from backend.window_spec import WindowSpec

    spec = window_spec_from_plan_node(node, default=default)
    if spec.closed != "right":
        from backend.plan_params import PlanParamError

        raise PlanParamError(f"closed={spec.closed!r} 暂未支持，仅 right")
    return spec


def _ewm_alpha(node: PlanNode, default_span: int = 20) -> float:
    from backend.ewm_spec import EwmSpec

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
    from backend.plan_params import PlanParamError, parse_finite_float, parse_unit_interval

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
        seg = arr[valid]
        ww = weights[-len(seg) :]
        return float(np.dot(seg, ww) / ww.sum())

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_product_expr(w: int) -> pl.Expr:
    """滚动乘积：正确处理 0、负数符号与 NULL 跳过。"""

    def _fn(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.nan
        if np.any(valid == 0.0):
            return 0.0
        neg_cnt = int(np.sum(valid < 0))
        sign = -1.0 if neg_cnt % 2 else 1.0
        return sign * float(np.exp(np.log(np.abs(valid)).sum()))

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_median_abs_dev_expr(w: int) -> pl.Expr:
    """Median Absolute Deviation：median(|x_i - median(window)|)。"""

    def _fn(arr: np.ndarray) -> float:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.nan
        med = float(np.median(valid))
        return float(np.median(np.abs(valid - med)))

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_mean_abs_dev_expr(w: int) -> pl.Expr:
    """Mean Absolute Deviation：mean(|x_i - mean(window)|)。"""

    def _fn(arr: np.ndarray) -> float:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.nan
        mu = float(valid.mean())
        return float(np.mean(np.abs(valid - mu)))

    return pl.col(_VAL).rolling_map(_fn, window_size=w, min_samples=1).over(_INST, order_by=_TS)


def _rolling_time_slope_expr(w: int) -> pl.Expr:
    """构造窗口内时间序列线性回归斜率 expr（x 为等距时间索引）。"""
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

    return pl.col(_VAL).rolling_map(_dot, window_size=w, min_samples=w).over(_INST, order_by=_TS)


def _rolling_argext_expr(w: int, *, pick: str) -> pl.Expr:
    """构造滚动 argmax/argmin expr（窗口全无效 → NULL；tie → first）。"""
    from backend.numeric_semantics import ts_argmax_empty_window_is_null

    def _fn(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).any():
            return np.nan if ts_argmax_empty_window_is_null() else 0.0
        if pick == "max":
            return float(np.nanargmax(arr))
        return float(np.nanargmin(arr))

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
    from backend.logical_semantics import is_null_polars_expr

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
    from backend.long_alignment import anchor_left_join_binary

    return anchor_left_join_binary(left, right, right_col="_y")


def _join_triple(
    left: pl.LazyFrame,
    mid: pl.LazyFrame,
    right: pl.LazyFrame,
) -> pl.LazyFrame:
    """按 (ts, inst) anchor 串联 LEFT JOIN。"""
    from backend.long_alignment import anchor_left_join_triple

    return anchor_left_join_triple(left, mid, right)


def _column_ref_name(node: PlanNode) -> str | None:
    """若 node 为 ``column`` 算子则返回列名字符串，否则 None。"""
    if node.op != "column":
        return None
    name = node.attrs.get("name")
    return str(name) if name else None


def _safe_div_null_expr(numer: pl.Expr, denom: pl.Expr, node: PlanNode) -> pl.Expr:
    """零/NULL 分母 → NULL（比率语义，不填 default）。"""
    from backend.numeric_semantics import protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    return (
        pl.when(numer.is_null() | denom.is_null())
        .then(None)
        .when(denom.abs() <= eps)
        .then(None)
        .otherwise(numer / denom)
    )


def _protected_div_expr(numer: pl.Expr, denom: pl.Expr, node: PlanNode) -> pl.Expr:
    from backend.elementwise_semantics import protected_div_polars
    from backend.numeric_semantics import protected_div_default, protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    default = _float_attr(node, "default", default=protected_div_default())
    return protected_div_polars(numer, denom, eps=eps, default=default)


def _protected_log_expr(val: pl.Expr, node: PlanNode) -> pl.Expr:
    from backend.elementwise_semantics import protected_log_polars
    from backend.numeric_semantics import protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    return protected_log_polars(val, eps=eps)


def _div_or_default_expr(numer: pl.Expr, denom: pl.Expr, node: PlanNode) -> pl.Expr:
    from backend.elementwise_semantics import div_or_default_polars
    from backend.numeric_semantics import protected_div_default, protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    default = _float_attr(node, "default", default=protected_div_default())
    return div_or_default_polars(numer, denom, eps=eps, default=default)


def _log_fill_invalid_expr(val: pl.Expr, node: PlanNode) -> pl.Expr:
    from backend.elementwise_semantics import log_fill_invalid_polars
    from backend.numeric_semantics import protected_epsilon_default

    eps = _float_attr(node, "epsilon", "eps", default=protected_epsilon_default())
    return log_fill_invalid_polars(val, eps=eps)


def _truthy_expr(expr: pl.Expr) -> pl.Expr:
    """将 expr 转为布尔：NULL/NaN/0 → false。"""
    from backend.logical_semantics import truthy_polars_expr

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
        "delay",
        "ts_delta",
        "ts_pct",
    }
)


def _parse_ts_op_on_column(node: PlanNode):
    """识别 ``ts_*(column(x), window, ...)``，返回 (op, col, WindowSpec)。"""
    from backend.window_spec import WindowSpec

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
    from backend.numeric_semantics import std_ddof_value

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
    if op == "delay":
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
        from backend.elementwise_semantics import max_horizontal_polars

        return max_horizontal_polars(left, right)
    if op == "minimum":
        from backend.elementwise_semantics import min_horizontal_polars

        return min_horizontal_polars(left, right)
    if op == "protected_div":
        return _protected_div_expr(left, right, node)
    if op == "div_or_default":
        return _div_or_default_expr(left, right, node)
    if op == "safe_div":
        return _safe_div_null_expr(left, right, node)
    if op == "divide":
        from backend.inf_sanitize import apply_inf_policy_polars_fast

        return apply_inf_policy_polars_fast(left / right, "divide")
    if op == "power":
        return _safe_pow_expr(left, right)
    if op == "gt":
        from backend.elementwise_semantics import compare_polars

        return compare_polars("gt", left, right)
    if op == "lt":
        from backend.elementwise_semantics import compare_polars

        return compare_polars("lt", left, right)
    if op == "eq":
        from backend.elementwise_semantics import compare_polars

        return compare_polars("eq", left, right)
    if op == "ge":
        from backend.elementwise_semantics import compare_polars

        return compare_polars("ge", left, right)
    if op == "le":
        from backend.elementwise_semantics import compare_polars

        return compare_polars("le", left, right)
    if op == "ne":
        from backend.elementwise_semantics import compare_polars

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
        from backend.elementwise_semantics import max_horizontal_polars

        expr = max_horizontal_polars(lcol, rcol)
    elif op == "minimum":
        from backend.elementwise_semantics import min_horizontal_polars

        expr = min_horizontal_polars(lcol, rcol)
    elif op == "protected_div":
        expr = _protected_div_expr(lcol, rcol, node)
    elif op == "div_or_default":
        expr = _div_or_default_expr(lcol, rcol, node)
    elif op == "safe_div":
        expr = _safe_div_null_expr(lcol, rcol, node)
    elif op == "divide":
        from backend.inf_sanitize import apply_inf_policy_polars_fast

        expr = apply_inf_policy_polars_fast(lcol / rcol, "divide")
    elif op == "power":
        expr = _safe_pow_expr(lcol, rcol)
    elif op == "gt":
        from backend.elementwise_semantics import compare_polars

        expr = compare_polars("gt", lcol, rcol)
    elif op == "lt":
        from backend.elementwise_semantics import compare_polars

        expr = compare_polars("lt", lcol, rcol)
    elif op == "eq":
        from backend.elementwise_semantics import compare_polars

        expr = compare_polars("eq", lcol, rcol)
    elif op == "ge":
        from backend.elementwise_semantics import compare_polars

        expr = compare_polars("ge", lcol, rcol)
    elif op == "le":
        from backend.elementwise_semantics import compare_polars

        expr = compare_polars("le", lcol, rcol)
    elif op == "ne":
        from backend.elementwise_semantics import compare_polars

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
    from backend.pair_window_spec import PairWindowSpec
    from backend.pairwise_rolling import polars_pairwise_output_guard, polars_ts_beta_expr, polars_vwap_expr

    pspec = PairWindowSpec.from_plan_node(node)
    w = pspec.size
    mp = pspec.min_periods
    ddof = pspec.ddof
    if op == "ts_corr":
        raw = pl.rolling_corr(lcol, rcol, window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        expr = polars_pairwise_output_guard(lcol, rcol, raw)
    elif op == "ts_cov":
        raw = pl.rolling_cov(lcol, rcol, window_size=w, min_samples=mp, ddof=ddof).over(
            _INST, order_by=_TS
        )
        expr = polars_pairwise_output_guard(lcol, rcol, raw)
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
        "safe_div",
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
    """非法定义域返回 NULL（对齐 SQL emitter）。"""
    return (
        pl.when(base.is_null() | exp.is_null())
        .then(None)
        .when((base < 0) & (exp != exp.floor()))
        .then(None)
        .when((base == 0) & (exp < 0))
        .then(None)
        .otherwise(base.pow(exp))
    )


def _truthy(col: str) -> pl.Expr:
    """列名版 truthy。"""
    from backend.logical_semantics import truthy_polars_expr

    return truthy_polars_expr(pl.col(col))


def _cs_rank_01_on(value_col: str) -> pl.Expr:
    """截面 0-1 rank，指定列名（DAG fusion 用）。"""
    from backend.rank_spec import polars_cs_rank_expr

    return polars_cs_rank_expr(value_col, partition_cols=(_TS,), order_by=_INST, canon="rank")


def _zscore_on(value_col: str, *, canon: str = "zscore") -> pl.Expr:
    """指定列名的截面 zscore expr（按 ts 分区，语义见 numeric_semantics）。"""
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
        from backend.rank_spec import polars_cs_rank_expr

        out_expr = polars_cs_rank_expr(tmp, partition_cols=(_TS,), order_by=_INST, canon=op)
    elif op == "zscore":
        out_expr = _zscore_on(tmp, canon="zscore")
    elif op == "neg":
        out_expr = -pl.col(tmp)
    elif op == "abs":
        out_expr = pl.col(tmp).abs()
    elif op == "scale":
        to_val = _scale_to_value(node)
        from backend.cross_section_spec import polars_scale_expr

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
    from backend.rank_spec import polars_cs_rank_expr

    return polars_cs_rank_expr(_VAL, partition_cols=(_TS,), order_by=_INST, canon=canon)


def _cs_rank_pct(*, canon: str = "rank_pct") -> pl.Expr:
    """截面百分位 rank expr（rank/n，语义见 rank_spec）。"""
    from backend.rank_spec import polars_cs_rank_expr

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
        op = "decay_linear"
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
            from backend.elementwise_semantics import max_horizontal_polars

            expr = max_horizontal_polars(pl.col(_VAL), pl.col("_y"))
        elif op == "minimum":
            from backend.elementwise_semantics import min_horizontal_polars

            expr = min_horizontal_polars(pl.col(_VAL), pl.col("_y"))
        elif op == "protected_div":
            expr = _protected_div_expr(pl.col(_VAL), pl.col("_y"), node)
        elif op == "div_or_default":
            expr = _div_or_default_expr(pl.col(_VAL), pl.col("_y"), node)
        elif op == "safe_div":
            expr = _safe_div_null_expr(pl.col(_VAL), pl.col("_y"), node)
        elif op == "divide":
            from backend.inf_sanitize import apply_inf_policy_polars_fast

            expr = apply_inf_policy_polars_fast(pl.col(_VAL) / pl.col("_y"), "divide")
        elif op == "power":
            expr = _safe_pow_expr(pl.col(_VAL), pl.col("_y"))
        elif op in {"gt", "lt", "eq", "ge", "le", "ne"}:
            from backend.elementwise_semantics import compare_polars

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
        from backend.elementwise_semantics import is_finite_polars_expr

        return inner.with_columns(is_finite_polars_expr(_VAL).alias(_VAL))

    if op == "is_infinite":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from backend.elementwise_semantics import is_infinite_polars_expr

        return inner.with_columns(is_infinite_polars_expr(_VAL).alias(_VAL))

    if op == "is_null":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from backend.logical_semantics import is_null_polars_expr

        return inner.with_columns(is_null_polars_expr(_VAL).alias(_VAL))

    if op == "is_not_null":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from backend.logical_semantics import is_not_null_polars_expr

        return inner.with_columns(is_not_null_polars_expr(_VAL).alias(_VAL))

    if op == "is_nan":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        from backend.logical_semantics import is_nan_polars_expr

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
        from backend.inf_sanitize import apply_inf_policy_polars_fast

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

    if op == "cap":
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
        from backend.numeric_semantics import std_ddof_value

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
        spec = _window_spec(node)
        w = spec.size
        mp = spec.min_periods
        from backend.numeric_semantics import std_ddof_value, zscore_zero_std_fill

        ddof = spec.ddof if "ddof" in (node.attrs or {}) else std_ddof_value("ts_zscore")
        zero_fill = zscore_zero_std_fill("ts_zscore")
        mean = pl.col(_VAL).rolling_mean(window_size=w, min_samples=mp).over(_INST, order_by=_TS)
        std = pl.col(_VAL).rolling_std(window_size=w, min_samples=mp, ddof=ddof).over(_INST, order_by=_TS)
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

    if op == "ts_corr":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        from backend.pair_window_spec import PairWindowSpec
        from backend.pairwise_rolling import polars_pairwise_output_guard

        pspec = PairWindowSpec.from_plan_node(node)
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        corr = pl.rolling_corr(
            pl.col(_VAL),
            pl.col("_y"),
            window_size=pspec.size,
            min_samples=pspec.min_periods,
        ).over(_INST, order_by=_TS)
        out = polars_pairwise_output_guard(pl.col(_VAL), pl.col("_y"), corr)
        return joined.with_columns(out.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ts_cov":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        from backend.pair_window_spec import PairWindowSpec
        from backend.pairwise_rolling import polars_pairwise_output_guard

        pspec = PairWindowSpec.from_plan_node(node)
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        joined = _join_binary(left, right)
        cov = pl.rolling_cov(
            pl.col(_VAL),
            pl.col("_y"),
            window_size=pspec.size,
            min_samples=pspec.min_periods,
            ddof=pspec.ddof,
        ).over(_INST, order_by=_TS)
        out = polars_pairwise_output_guard(pl.col(_VAL), pl.col("_y"), cov)
        return joined.with_columns(out.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "ts_beta":
        if len(node.inputs) < 2:
            return None
        fused = _try_ts_pair_from_base_columns(node, base, op)
        if fused is not None:
            return fused
        from backend.pair_window_spec import PairWindowSpec
        from backend.pairwise_rolling import polars_ts_beta_expr

        pspec = PairWindowSpec.from_plan_node(node)
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
        from backend.pairwise_rolling import polars_vwap_expr

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
        from backend.rank_spec import polars_ts_rank_expr

        return inner.with_columns(
            polars_ts_rank_expr(
                _VAL,
                window=spec.size,
                inst_col=_INST,
                ts_col=_TS,
                min_periods=spec.min_periods,
            ).alias(_VAL)
        )

    if op in {"ema", "ewm_std", "ewm_var"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        alpha = _ewm_alpha(node)
        if op == "ema":
            expr = pl.col(_VAL).ewm_mean(alpha=alpha, adjust=False)
        elif op == "ewm_std":
            expr = pl.col(_VAL).ewm_std(alpha=alpha, adjust=False)
        else:
            expr = pl.col(_VAL).ewm_var(alpha=alpha, adjust=False)
        return inner.with_columns(expr.over(_INST, order_by=_TS).alias(_VAL))

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
        return joined.with_columns(
            pl.when(_truthy(_VAL)).then(pl.col("_ym")).otherwise(pl.col("_y")).alias(_VAL)
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
        p = _float_attr(node, "p", default=0.5)
        pos_p = _literal_value(node, 0)
        if pos_p is not None:
            p = pos_p
        q = pl.col(_VAL).quantile(quantile=p, interpolation="linear").over(_TS, order_by=_INST)
        return inner.with_columns(q.alias(_VAL))

    if op == "winsorize":
        from backend.plan_params import parse_winsorize_quantiles

        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        lo_p, hi_p = parse_winsorize_quantiles(node)
        lo = pl.col(_VAL).quantile(quantile=lo_p, interpolation="linear").over(_TS, order_by=_INST)
        hi = pl.col(_VAL).quantile(quantile=hi_p, interpolation="linear").over(_TS, order_by=_INST)
        return inner.with_columns(pl.col(_VAL).clip(lo, hi).alias(_VAL))

    if op in {"cs_resid", "cs_regression"}:
        from backend.plan_params import int_mode_from_plan_node
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
            expr = pl.col(_VAL).mean().over(*over_keys, order_by=_INST)
        elif op == "group_sum":
            expr = (
                pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
                .then(None)
                .otherwise(pl.col(_VAL).sum().over(*over_keys, order_by=_INST))
            )
        elif op == "group_min":
            expr = pl.col(_VAL).min().over(*over_keys, order_by=_INST)
        elif op == "group_max":
            expr = pl.col(_VAL).max().over(*over_keys, order_by=_INST)
        elif op == "group_count":
            expr = (
                pl.when(pl.col(_VAL).is_null() | pl.col(_VAL).is_nan())
                .then(None)
                .otherwise(
                    pl.col(_VAL).count().over(*over_keys, order_by=_INST).cast(pl.Float64)
                )
            )
        elif op == "group_std":
            from backend.numeric_semantics import std_ddof_value

            cnt = pl.col(_VAL).count().over(*over_keys, order_by=_INST)
            std_expr = pl.col(_VAL).std(ddof=std_ddof_value("group_std")).over(*over_keys, order_by=_INST)
            expr = (
                pl.when(pl.col(_VAL).is_null())
                .then(None)
                .when(cnt < 2)
                .then(0.0)
                .when(std_expr.is_null())
                .then(0.0)
                .otherwise(std_expr)
            )
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
            expr = _group_normalize_on(_VAL, over_keys)
        else:
            from backend.rank_spec import polars_cs_rank_expr

            expr = polars_cs_rank_expr(_VAL, partition_cols=over_keys, order_by=_INST, canon="group_rank")
        return joined.with_columns(expr.alias(_VAL)).select(_TS, _INST, _VAL)

    if op == "cs_mad":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        med = pl.col(_VAL).median().over(_TS, order_by=_INST)
        mad = (pl.col(_VAL) - med).abs().median().over(_TS, order_by=_INST)
        return inner.with_columns(mad.alias(_VAL))

    if op == "cs_mad_zscore":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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

    if op == "delay":
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
        from backend.polars_long_policy import UnsupportedCausalOperatorError

        raise UnsupportedCausalOperatorError(
            "polars_long 不支持 bfill/causal_bfill（因果占位，非传统 backward fill）；请改用 ffill 或 pandas 研究路径"
        )

    if op == "causal_bfill":
        from backend.polars_long_policy import UnsupportedCausalOperatorError

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
        from backend.cross_section_spec import polars_scale_expr

        return inner.with_columns(
            polars_scale_expr(_VAL, to_val, partition_cols=(_TS,), order_by=_INST).alias(_VAL)
        )

    if op == "log_returns":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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

    if op in {"c_mean", "c_std", "c_sum", "c_count"}:
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        valid = pl.col(_VAL).count().over(_TS, order_by=_INST)
        if op == "c_mean":
            raw = pl.col(_VAL).mean().over(_TS, order_by=_INST)
            expr = pl.when(valid == 0).then(None).otherwise(raw)
        elif op == "c_std":
            raw = pl.col(_VAL).std(ddof=1).over(_TS, order_by=_INST)
            expr = pl.when(valid == 0).then(None).otherwise(raw)
        elif op == "c_sum":
            raw = pl.col(_VAL).sum().over(_TS, order_by=_INST)
            expr = pl.when(valid == 0).then(None).otherwise(raw)
        else:
            expr = valid.cast(pl.Float64)
        return inner.with_columns(expr.alias(_VAL))

    if op == "log_abs":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        abs_v = pl.col(_VAL).abs()
        return inner.with_columns(
            pl.when(pl.col(_VAL).is_null() | (abs_v == 0))
            .then(None)
            .otherwise(abs_v.log())
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

    if op == "decay_linear":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        return inner.with_columns(_rolling_linear_decay_expr(w).alias(_VAL))

    if op == "ts_mad":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
        w = _window_int(node)
        return inner.with_columns(_rolling_product_expr(w).alias(_VAL))

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
        from backend.numeric_semantics import ts_sharpe_zero_std_is_null

        expr = (
            pl.when(std.is_null() | (std == 0))
            .then(None if ts_sharpe_zero_std_is_null() else 0.0)
            .otherwise(mean / std)
            * sqrt_af
        )
        from backend.inf_sanitize import apply_inf_policy_polars_fast

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

    if op in {"ewm_corr", "ewm_cov"}:
        if len(node.inputs) < 2:
            return None
        left = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        right = _compile_child(node, 1, base, parent_op=op, ctx=ctx, memo=memo)
        if left is None or right is None:
            return None
        span = max(_window_int(node, default=20), 2)
        joined = _join_binary(left, right)
        return _ewm_binary_map_groups(joined, span, corr=(op == "ewm_corr"))

    if op == "ts_ratio":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = max(_int_attr(node, "d", "window", input_index=0, default=3), 1)
        k = max(_int_attr(node, "k", "order", input_index=1, default=2), 1)
        from cleaned_operators._numpy_kernels import ts_moment_

        return _unary_inst_map_groups(inner, lambda arr: ts_moment_(arr, w, k))

    if op == "ts_max_buildup":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None
        w = _window_int(node)
        from cleaned_operators._numpy_kernels import ts_max_buildup_

        return _unary_inst_map_groups(inner, lambda arr: ts_max_buildup_(arr, w))

    if op == "expanding_rank":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
        if inner is None:
            return None

        def _expanding_rank(arr: np.ndarray) -> np.ndarray:
            ranked = pd.Series(arr, dtype=float).expanding(min_periods=1).rank(pct=True)
            return ranked.to_numpy(dtype=float)

        return _unary_inst_map_groups(inner, _expanding_rank)

    if op == "fillna_interpolate":
        inner = _compile_child(node, 0, base, parent_op=op, ctx=ctx, memo=memo)
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
    from backend.long_frame import long_table_to_polars_lazy

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
    """兼容旧名：``PolarsBackend`` + ``FACTOR_ENGINE_POLARS_EXPR=1``。

    参数:
        plan: 逻辑计划根节点。
        ctx: 执行上下文。

    返回:
        与 ``execute_polars_long_plan`` 相同的 MultiIndex Series。
    """
    return execute_polars_long_plan(plan, ctx)
