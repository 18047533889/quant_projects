# -*- coding: utf-8 -*-
"""统计 / 回归类算子 Polars 实现。

假设检验簇
----------
``corr_test`` / ``durbin_watson_test`` 等需 scipy/statsmodels，
且为 **扩展窗口**（仅用前缀样本，因果安全）。Polars 无对应 API，
经 ``expanding_bivariate`` / ``expanding_univariate`` + pandas 桥接。
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


def _align_cols(*dfs: pl.DataFrame) -> list[str]:
    cols = _numeric_cols(dfs[0])
    for df in dfs[1:]:
        cols = [c for c in cols if c in df.columns]
    return cols


def _time_slope_col(col: pl.Expr, window: int) -> pl.Expr:
    w = max(int(window), 1)
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
        ww = weights[-len(arr):]
        valid = ~np.isnan(arr)
        if not valid.any():
            return np.nan
        return float(np.dot(arr[valid], ww[valid]))

    return col.rolling_map(_dot, window_size=w, min_samples=1)


@register_operator(name="Slope", category="statistics", business_category="statistics_regression", canonical="Slope", source="factor_dsl_polars")
class SlopePolars(SeriesOperator):
    """Polars 滚动时间斜率"""
    metadata = OperatorMetadata(
        name="Slope", category="statistics", description="滚动时间斜率",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(x)
        return x.with_columns([_time_slope_col(pl.col(c), w).alias(c) for c in cols])


@register_operator(name="ts_regression", category="time_series", business_category="time_series", canonical="ts_regression", source="factor_dsl_polars")
class TSRegressionPolars(SeriesOperator):
    """Polars 滚动回归"""
    metadata = OperatorMetadata(
        name="ts_regression", category="time_series", description="滚动回归",
        param_names=["y", "x", "window", "lag", "retval"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(
        self,
        y: pl.DataFrame,
        x: pl.DataFrame,
        window: int = 252,
        lag: int = 0,
        retval: str = "slope",
        **kwargs,
    ) -> pl.DataFrame:
        from cleaned_operators._rolling_fast import rolling_regression

        w = max(int(kwargs.get("d", window)), 3)
        lag_i = int(kwargs.get("lag", lag))
        ret = str(kwargs.get("retval", retval))
        cols = _align_cols(y, x)
        py = y.select(cols).to_pandas()
        px = x.select(cols).to_pandas()
        out = rolling_regression(py, px, window=w, min_periods=3, lag=lag_i, retval=ret)
        return y.with_columns([
            pl.Series(name=c, values=np.asarray(out[c], dtype=np.float64)) for c in cols
        ])


def _rolling_unary(x: pl.DataFrame, window: int, expr_fn) -> pl.DataFrame:
    w = max(int(window), 1)
    cols = _numeric_cols(x)
    return x.with_columns([expr_fn(pl.col(c), w).alias(c) for c in cols])


def _rolling_binary(
    left: pl.DataFrame,
    right: pl.DataFrame,
    window: int,
    expr_fn,
) -> pl.DataFrame:
    w = max(int(window), 1)
    cols = _align_cols(left, right)
    return left.with_columns([expr_fn(left[c], right[c], w).alias(c) for c in cols])


@register_operator(name="avg", category="statistics", business_category="statistics_regression", canonical="avg", source="factor_dsl_polars")
class AvgPolars(SeriesOperator):
    """Polars 滚动均值"""
    metadata = OperatorMetadata(
        name="avg", category="statistics", description="滚动均值",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_unary(x, w, lambda c, n: c.rolling_mean(window_size=n, min_samples=1))


@register_operator(name="Var", category="statistics", business_category="statistics_regression", canonical="Var", source="factor_dsl_polars")
class VarPolars(SeriesOperator):
    """Polars 滚动方差"""
    metadata = OperatorMetadata(
        name="Var", category="statistics", description="滚动方差",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_unary(x, w, lambda c, n: c.rolling_var(window_size=n, min_samples=1))


@register_operator(name="Skew", category="statistics", business_category="statistics_regression", canonical="Skew", source="factor_dsl_polars")
class SkewPolars(SeriesOperator):
    """Polars 滚动偏度"""
    metadata = OperatorMetadata(
        name="Skew", category="statistics", description="滚动偏度",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_unary(x, w, lambda c, n: c.rolling_skew(window_size=n, min_samples=1))


@register_operator(name="Kurt", category="statistics", business_category="statistics_regression", canonical="Kurt", source="factor_dsl_polars")
class KurtPolars(SeriesOperator):
    """Polars 滚动峰度"""
    metadata = OperatorMetadata(
        name="Kurt", category="statistics", description="滚动峰度",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))

        def _kurt(arr: np.ndarray) -> float:
            if len(arr) < 3:
                return np.nan
            m = np.nanmean(arr)
            s = np.nanstd(arr, ddof=1)
            if s == 0 or not np.isfinite(s):
                return np.nan
            return float(np.nanmean(((arr - m) / s) ** 4) - 3.0)

        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).rolling_map(_kurt, window_size=w, min_samples=3).alias(c) for c in cols
        ])


@register_operator(name="Median", category="statistics", business_category="statistics_regression", canonical="Median", source="factor_dsl_polars")
class MedianPolars(SeriesOperator):
    """Polars 滚动中位数"""
    metadata = OperatorMetadata(
        name="Median", category="statistics", description="滚动中位数",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_unary(x, w, lambda c, n: c.rolling_median(window_size=n, min_samples=1))


@register_operator(name="count", category="statistics", business_category="statistics_regression", canonical="count", source="factor_dsl_polars")
class CountPolars(SeriesOperator):
    """Polars 滚动非空计数"""
    metadata = OperatorMetadata(
        name="count", category="statistics", description="滚动非空计数",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_unary(
            x, w, lambda c, n: c.is_not_null().cast(pl.Float64).rolling_sum(window_size=n, min_samples=1)
        )


@register_operator(name="Beta", category="statistics", business_category="statistics_regression", canonical="Beta", source="factor_dsl_polars")
class BetaPolars(SeriesOperator):
    """Polars 滚动 Beta"""
    metadata = OperatorMetadata(
        name="Beta", category="statistics", description="滚动 Beta",
        param_names=["y", "x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 2)

        def _beta(a, b, n):
            return pl.rolling_cov(a, b, window_size=n, min_samples=2) / b.rolling_var(
                window_size=n, min_samples=2
            )

        return _rolling_binary(y, x, w, _beta)


@register_operator(name="Corr", category="statistics", business_category="statistics_regression", canonical="Corr", source="factor_dsl_polars")
class CorrPolars(SeriesOperator):
    """Polars 滚动相关"""
    metadata = OperatorMetadata(
        name="Corr", category="statistics", description="滚动相关",
        param_names=["x", "y", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_binary(
            x, y, w, lambda a, b, n: pl.rolling_corr(a, b, window_size=n, min_samples=1)
        )


@register_operator(name="Cov", category="statistics", business_category="statistics_regression", canonical="Cov", source="factor_dsl_polars")
class CovPolars(SeriesOperator):
    """Polars 滚动协方差"""
    metadata = OperatorMetadata(
        name="Cov", category="statistics", description="滚动协方差",
        param_names=["x", "y", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_binary(
            x, y, w, lambda a, b, n: pl.rolling_cov(a, b, window_size=n, min_samples=1)
        )


@register_operator(name="Covariance", category="statistics", business_category="statistics_regression", canonical="Covariance", source="factor_dsl_polars")
class CovariancePolars(CovPolars):
    """Polars 滚动协方差"""
    metadata = OperatorMetadata(
        name="Covariance", category="statistics", description="滚动协方差",
        param_names=["x", "y", "window"], return_type="series", tags=["statistics", "polars"],
    )


@register_operator(name="intercept", category="statistics", business_category="statistics_regression", canonical="intercept", source="factor_dsl_polars")
class InterceptPolars(SeriesOperator):
    """Polars 滚动 OLS 截距"""
    metadata = OperatorMetadata(
        name="intercept", category="statistics", description="滚动 OLS 截距",
        param_names=["y", "x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 2)
        cols = _align_cols(y, x)
        exprs = []
        for c in cols:
            cov = pl.rolling_cov(y[c], x[c], window_size=w, min_samples=2)
            var_x = x[c].rolling_var(window_size=w, min_samples=2)
            slope = cov / var_x
            mean_y = y[c].rolling_mean(window_size=w, min_samples=2)
            mean_x = x[c].rolling_mean(window_size=w, min_samples=2)
            exprs.append((mean_y - slope * mean_x).alias(c))
        return y.with_columns(exprs)


@register_operator(name="r_squared", category="statistics", business_category="statistics_regression", canonical="r_squared", source="factor_dsl_polars")
class RSquaredPolars(SeriesOperator):
    """Polars 滚动 R²"""
    metadata = OperatorMetadata(
        name="r_squared", category="statistics", description="滚动 R²",
        param_names=["y", "x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 2)
        cols = _align_cols(y, x)
        return y.with_columns([
            (pl.rolling_corr(y[c], x[c], window_size=w, min_samples=2).pow(2)).alias(c) for c in cols
        ])


@register_operator(name="residual", category="statistics", business_category="statistics_regression", canonical="residual", source="factor_dsl_polars")
class ResidualPolars(SeriesOperator):
    """Polars 滚动回归残差均值"""
    metadata = OperatorMetadata(
        name="residual", category="statistics", description="滚动回归残差均值",
        param_names=["y", "x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 3)
        cols = _align_cols(y, x)
        exprs = []
        for c in cols:
            cov = pl.rolling_cov(y[c], x[c], window_size=w, min_samples=3)
            var_x = x[c].rolling_var(window_size=w, min_samples=3)
            slope = cov / var_x
            mean_y = y[c].rolling_mean(window_size=w, min_samples=3)
            mean_x = x[c].rolling_mean(window_size=w, min_samples=3)
            intercept = mean_y - slope * mean_x
            resid = y[c] - (slope * x[c] + intercept)
            exprs.append(resid.rolling_mean(window_size=w, min_samples=3).alias(c))
        return y.with_columns(exprs)


@register_operator(name="Mode", category="statistics", business_category="statistics_regression", canonical="Mode", source="factor_dsl_polars")
class ModePolars(SeriesOperator):
    """Polars 滚动众数"""
    metadata = OperatorMetadata(
        name="Mode", category="statistics", description="滚动众数",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 1)

        def _mode(arr: np.ndarray) -> float:
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                return np.nan
            vals, counts = np.unique(valid, return_counts=True)
            return float(vals[int(np.argmax(counts))])

        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).rolling_map(_mode, window_size=w, min_samples=1).alias(c) for c in cols
        ])


@register_operator(name="autocorr", category="statistics", business_category="statistics_regression", canonical="autocorr", source="factor_dsl_polars")
class AutocorrPolars(SeriesOperator):
    """Polars 自相关系数"""
    metadata = OperatorMetadata(
        name="autocorr", category="statistics", description="自相关系数",
        param_names=["x", "lag"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        l = max(int(kwargs.get("d", lag)), 1)
        w = max(l + 2, 5)
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.rolling_corr(pl.col(c), pl.col(c).shift(l), window_size=w, min_samples=l + 2).alias(c)
            for c in cols
        ])


@register_operator(name="beta", category="statistics", business_category="statistics_regression", canonical="beta", source="factor_dsl_polars")
class betaPolars(BetaPolars):
    """Polars 滚动 Beta（小写）"""
    metadata = OperatorMetadata(
        name="beta", category="statistics", description="滚动 Beta（小写）",
        param_names=["y", "x", "window"], return_type="series", tags=["statistics", "polars"],
    )


@register_operator(name="corr_test", category="statistics", business_category="statistics_regression", canonical="corr_test", source="factor_dsl_polars")
class CorrTestPolars(SeriesOperator):
    """扩展窗口 Pearson 相关检验 p 值；前缀样本数 < 3 时为 NaN。"""

    metadata = OperatorMetadata(
        name="corr_test", category="statistics", description="Pearson 相关检验 p 值",
        param_names=["x", "y"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from scipy import stats

        from cleaned_operators._causal import expanding_bivariate

        cols = _align_cols(x, y)
        pdf_x = x.select(cols).to_pandas()
        pdf_y = y.select(cols).to_pandas()
        out = expanding_bivariate(
            pdf_x, pdf_y, lambda a, b: stats.pearsonr(a, b)[1], min_periods=3
        )
        return x.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


@register_operator(name="durbin_watson_test", category="statistics", business_category="statistics_regression", canonical="durbin_watson_test", source="factor_dsl_polars")
class DurbinWatsonTestPolars(SeriesOperator):
    """扩展窗口 Durbin-Watson 统计量（非 p 值）；用于残差自相关诊断。"""

    metadata = OperatorMetadata(
        name="durbin_watson_test", category="statistics", description="Durbin-Watson 统计量",
        param_names=["residuals"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, residuals: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from statsmodels.stats.stattools import durbin_watson

        from cleaned_operators._causal import expanding_univariate

        cols = _numeric_cols(residuals)
        pdf = residuals.select(cols).to_pandas()
        out = expanding_univariate(pdf, durbin_watson, min_periods=2)
        return residuals.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])
