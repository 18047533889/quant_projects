# -*- coding: utf-8 -*-
"""Polars 扩展：时序 / 截面 row / 累计簇算子。

累计簇说明（``cum_*``）
--------------------
- ``cum_delta``：相对 **首个有效值** 的差分，非简单 diff。
- ``cum_first``：截至当前行的首个有效值（前缀填充）。
- ``cum_positive_streak``：连续正值计数；遇 NaN 重置，非正非 NaN 记 0。
- 上述三者在 Polars 无单行表达式，故逐列 NumPy 循环，与 pandas 语义对齐。
"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _rolling_window(x: pl.DataFrame, window: int, expr_fn) -> pl.DataFrame:
    w = max(int(window), 1)
    cols = _numeric_cols(x)
    return x.with_columns([expr_fn(pl.col(c), w).alias(c) for c in cols])


@register_operator(name="m_var", category="time_series", business_category="time_series", canonical="ts_var", source="factor_dsl_polars")
class TSVarPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="m_var", category="time_series", description="滚动方差",
        param_names=["x", "window"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_window(x, w, lambda c, n: c.rolling_var(window_size=n, min_samples=1))


@register_operator(name="m_median", category="time_series", business_category="time_series", canonical="ts_median", source="factor_dsl_polars")
class TSMedianPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="m_median", category="time_series", description="滚动中位数",
        param_names=["x", "window"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_window(x, w, lambda c, n: c.rolling_median(window_size=n, min_samples=1))


@register_operator(name="m_mad", category="time_series", business_category="time_series", canonical="ts_mad", source="factor_dsl_polars")
class TSMadPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="m_mad", category="time_series", description="滚动 MAD",
        param_names=["x", "window"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(x)

        def _mad(arr: np.ndarray) -> float:
            if len(arr) == 0:
                return np.nan
            med = np.nanmedian(arr)
            return float(np.nanmean(np.abs(arr - med)))

        return x.with_columns([
            pl.col(c).rolling_map(_mad, window_size=w, min_samples=1).alias(c) for c in cols
        ])


@register_operator(name="power", category="math", business_category="elementwise_math", canonical="power", source="factor_dsl_polars")
class PowerPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="power", category="math", description="幂",
        param_names=["x", "p"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, p: float = 2.0, **kwargs) -> pl.DataFrame:
        exp = float(kwargs.get("exp", p))
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).pow(exp).alias(c) for c in cols])


@register_operator(name="log10", category="math", business_category="elementwise_math", canonical="log10", source="factor_dsl_polars")
class Log10Polars(SeriesOperator):
    metadata = OperatorMetadata(
        name="log10", category="math", description="log10",
        param_names=["x"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).log10().alias(c) for c in cols])


@register_operator(name="tanh", category="math", business_category="elementwise_math", canonical="tanh", source="factor_dsl_polars")
class TanhPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="tanh", category="math", description="tanh",
        param_names=["x"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).tanh().alias(c) for c in cols])


@register_operator(name="minimum", category="math", business_category="elementwise_math", canonical="minimum", source="factor_dsl_polars")
class MinimumPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="minimum", category="math", description="逐元素 min(x,y)",
        param_names=["x", "y"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in _numeric_cols(x) if c in y.columns]
        return x.with_columns([
            pl.min_horizontal(pl.col(c), y[c]).alias(c) for c in cols
        ])


@register_operator(name="maximum", category="math", business_category="elementwise_math", canonical="maximum", source="factor_dsl_polars")
class MaximumPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="maximum", category="math", description="逐元素 max(x,y)",
        param_names=["x", "y"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in _numeric_cols(x) if c in y.columns]
        out = x.with_columns([
            pl.max_horizontal(pl.col(c), y[c]).alias(c) for c in cols
        ])
        return out


@register_operator(name="if_else", category="signal", business_category="technical_signal", canonical="if_else", source="factor_dsl_polars")
class IfElsePolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="if_else", category="signal", description="条件选择",
        param_names=["condition", "v1", "v2"], return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(self, condition: pl.DataFrame, v1: pl.DataFrame, v2: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in _numeric_cols(condition) if c in v1.columns and c in v2.columns]
        return condition.select([
            pl.when(pl.col(c).cast(pl.Boolean, strict=False)).then(v1[c]).otherwise(v2[c]).alias(c)
            for c in cols
        ] + ([condition["date"]] if "date" in condition.columns else []))


@register_operator(name="ifnan", category="signal", business_category="technical_signal", canonical="ifnan", source="factor_dsl_polars")
class IfNaNPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ifnan", category="signal", description="NaN 替换",
        param_names=["x", "v"], return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, v: float = 0.0, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        fill = float(kwargs.get("value", v))
        return x.with_columns([pl.col(c).fill_nan(fill).alias(c) for c in cols])


@register_operator(name="row_sum", category="cross_sectional", business_category="cross_sectional", canonical="row_sum", source="factor_dsl_polars")
class RowSumPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_sum", category="cross_sectional", description="行求和",
        param_names=["x"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        s = pl.sum_horizontal(*[pl.col(c) for c in cols])
        return x.with_columns([s.alias(c) for c in cols])


@register_operator(name="row_avg", category="cross_sectional", business_category="cross_sectional", canonical="row_avg", source="factor_dsl_polars")
class RowAvgPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_avg", category="cross_sectional", description="行均值",
        param_names=["x"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        m = pl.mean_horizontal(*[pl.col(c) for c in cols])
        return x.with_columns([m.alias(c) for c in cols])


@register_operator(name="row_std", category="cross_sectional", business_category="cross_sectional", canonical="row_std", source="factor_dsl_polars")
class RowStdPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_std", category="cross_sectional", description="行标准差",
        param_names=["x"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        arr = x.select(cols).to_numpy()
        out = np.nanstd(arr, axis=1, ddof=1, keepdims=True)
        out = np.repeat(out, len(cols), axis=1)
        result = pl.DataFrame(out, schema=cols)
        if "date" in x.columns:
            result = result.with_columns(x["date"])
        return result


@register_operator(name="row_min", category="cross_sectional", business_category="cross_sectional", canonical="row_min", source="factor_dsl_polars")
class RowMinPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_min", category="cross_sectional", description="行最小",
        param_names=["x"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        m = pl.min_horizontal(*[pl.col(c) for c in cols])
        return x.with_columns([m.alias(c) for c in cols])


@register_operator(name="row_max", category="cross_sectional", business_category="cross_sectional", canonical="row_max", source="factor_dsl_polars")
class RowMaxPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_max", category="cross_sectional", description="行最大",
        param_names=["x"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        m = pl.max_horizontal(*[pl.col(c) for c in cols])
        return x.with_columns([m.alias(c) for c in cols])


@register_operator(name="cum_sum", category="time_series", business_category="shift_diff_cum", canonical="cum_sum", source="factor_dsl_polars")
class CumSumPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_sum", category="time_series", description="累计和",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_sum().alias(c) for c in cols])


@register_operator(name="cum_max", category="time_series", business_category="shift_diff_cum", canonical="cum_max", source="factor_dsl_polars")
class CumMaxPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_max", category="time_series", description="累计最大",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_max().alias(c) for c in cols])


@register_operator(name="cum_min", category="time_series", business_category="shift_diff_cum", canonical="cum_min", source="factor_dsl_polars")
class CumMinPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_min", category="time_series", description="累计最小",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_min().alias(c) for c in cols])


@register_operator(name="cum_prod", category="time_series", business_category="shift_diff_cum", canonical="cum_prod", source="factor_dsl_polars")
class CumProdPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_prod", category="time_series", description="累计积",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_prod().alias(c) for c in cols])


@register_operator(name="cum_count", category="time_series", business_category="shift_diff_cum", canonical="cum_count", source="factor_dsl_polars")
class CumCountPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_count", category="time_series", description="累积非空计数",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).is_not_null().cast(pl.Int64).cum_sum().alias(c) for c in cols
        ])


@register_operator(name="cum_last", category="time_series", business_category="shift_diff_cum", canonical="cum_last", source="factor_dsl_polars")
class CumLastPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_last", category="time_series", description="截至目前最末有效值",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).forward_fill().alias(c) for c in cols])


def _cum_delta_col(arr: np.ndarray) -> np.ndarray:
    """x_t - x_first_valid；首值之前为 NaN。"""
    out = np.full_like(arr, np.nan, dtype=float)
    first = np.nan
    for i, v in enumerate(arr):
        if np.isfinite(v) and np.isnan(first):
            first = v
        if np.isfinite(v) and np.isfinite(first):
            out[i] = v - first
    return out


def _cum_first_col(arr: np.ndarray) -> np.ndarray:
    """前缀内首个有效值向前填充。"""
    out = np.full_like(arr, np.nan, dtype=float)
    first = np.nan
    for i, v in enumerate(arr):
        if np.isfinite(v) and np.isnan(first):
            first = v
        if np.isfinite(first):
            out[i] = first
    return out


def _cum_positive_streak_col(arr: np.ndarray) -> np.ndarray:
    """连续正值长度；NaN 中断 streak，零/负值记 0。"""
    out = np.full_like(arr, np.nan, dtype=float)
    streak = 0
    for i, v in enumerate(arr):
        if not np.isfinite(v):
            streak = 0
            out[i] = np.nan
        elif v > 0:
            streak += 1
            out[i] = float(streak)
        else:
            streak = 0
            out[i] = 0.0
    return out


def _apply_col_kernel(x: pl.DataFrame, kernel) -> pl.DataFrame:
    """宽表逐列应用 1D NumPy 内核，保留 date 列。"""
    cols = _numeric_cols(x)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = kernel(x[c].to_numpy())
    result = pl.DataFrame(out)
    if "date" in x.columns:
        result = result.with_columns(x["date"])
    return result


@register_operator(name="cum_delta", category="time_series", business_category="shift_diff_cum", canonical="cum_delta", source="factor_dsl_polars")
class CumDeltaPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_delta", category="time_series", description="累积变化量 x - first(x)",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _apply_col_kernel(x, _cum_delta_col)


@register_operator(name="cum_first", category="time_series", business_category="shift_diff_cum", canonical="cum_first", source="factor_dsl_polars")
class CumFirstPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_first", category="time_series", description="截至目前首个有效值",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _apply_col_kernel(x, _cum_first_col)


@register_operator(name="cum_positive_streak", category="time_series", business_category="shift_diff_cum", canonical="cum_positive_streak", source="factor_dsl_polars")
class CumPositiveStreakPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_positive_streak", category="time_series", description="连续正值计数",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _apply_col_kernel(x, _cum_positive_streak_col)


@register_operator(name="row_median", category="cross_sectional", business_category="cross_sectional", canonical="row_median", source="factor_dsl_polars")
class RowMedianPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_median", category="cross_sectional", description="行中位数",
        param_names=["x"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        arr = x.select(cols).to_numpy()
        out = np.nanmedian(arr, axis=1, keepdims=True)
        out = np.repeat(out, len(cols), axis=1)
        result = pl.DataFrame(out, schema=cols)
        if "date" in x.columns:
            result = result.with_columns(x["date"])
        return result


@register_operator(name="row_count", category="cross_sectional", business_category="cross_sectional", canonical="row_count", source="factor_dsl_polars")
class RowCountPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_count", category="cross_sectional", description="行非空计数",
        param_names=["x"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        cnt = pl.sum_horizontal(*[pl.col(c).is_not_null().cast(pl.Int64) for c in cols])
        return x.with_columns([cnt.alias(c) for c in cols])


@register_operator(name="row_var", category="cross_sectional", business_category="cross_sectional", canonical="row_var", source="factor_dsl_polars")
class RowVarPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_var", category="cross_sectional", description="行方差",
        param_names=["x"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        arr = x.select(cols).to_numpy()
        out = np.nanvar(arr, axis=1, ddof=1, keepdims=True)
        out = np.repeat(out, len(cols), axis=1)
        result = pl.DataFrame(out, schema=cols)
        if "date" in x.columns:
            result = result.with_columns(x["date"])
        return result


@register_operator(name="row_corr", category="cross_sectional", business_category="cross_sectional", canonical="row_corr", source="factor_dsl_polars")
class RowCorrPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_corr", category="cross_sectional", description="行相关（两输入同名列）",
        param_names=["x", "y"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in _numeric_cols(x) if c in y.columns]
        arr_x = x.select(cols).to_numpy()
        arr_y = y.select(cols).to_numpy()
        out = np.full_like(arr_x, np.nan, dtype=float)
        for i in range(len(arr_x)):
            a, b = arr_x[i], arr_y[i]
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() >= 2:
                out[i, :] = np.corrcoef(a[mask], b[mask])[0, 1]
        result = pl.DataFrame(out, schema=cols)
        if "date" in x.columns:
            result = result.with_columns(x["date"])
        return result


@register_operator(name="row_beta", category="cross_sectional", business_category="cross_sectional", canonical="row_beta", source="factor_dsl_polars")
class RowBetaPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="row_beta", category="cross_sectional", description="行 Beta",
        param_names=["y", "x"], return_type="series", tags=["cross_sectional", "polars"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in _numeric_cols(y) if c in x.columns]
        arr_y = y.select(cols).to_numpy()
        arr_x = x.select(cols).to_numpy()
        out = np.full_like(arr_y, np.nan, dtype=float)
        for i in range(len(arr_y)):
            a, b = arr_y[i], arr_x[i]
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() >= 2:
                var_b = np.var(b[mask])
                if var_b > 0:
                    out[i, :] = np.cov(a[mask], b[mask])[0, 1] / var_b
        result = pl.DataFrame(out, schema=cols)
        if "date" in y.columns:
            result = result.with_columns(y["date"])
        return result
