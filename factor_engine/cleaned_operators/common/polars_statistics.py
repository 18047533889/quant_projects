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

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _align_cols(*dfs: pl.DataFrame) -> list[str]:
    cols = _numeric_cols(dfs[0])
    for df in dfs[1:]:
        cols = [c for c in cols if c in df.columns]
    return cols


def _mean_abs_dev_1d(arr) -> float:
    """单窗口平均绝对离差：``mean(|x_i - mean(window)|)``（单一中心=窗口均值）。"""
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    mu = float(a.mean())
    return float(np.mean(np.abs(a - mu)))


def _median_abs_dev_1d(arr) -> float:
    """单窗口中位数绝对离差：``median(|x_i - median(window)|)``（单一中心=窗口中位数）。"""
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


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

    return col.rolling_map(_dot, window_size=w, min_samples=w)


class SlopePolars(SeriesOperator):
    """Polars 滚动时间斜率"""
    metadata = OperatorMetadata(
        name="ts_time_slope", category="statistics", description="滚动时间斜率",
        param_names=["x", "d"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(x)
        return x.with_columns([_time_slope_col(pl.col(c), w).alias(c) for c in cols])


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
        from factor_engine.cleaned_operators._rolling_fast import rolling_regression

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


class AvgPolars(SeriesOperator):
    """Polars 滚动均值"""
    metadata = OperatorMetadata(
        name="avg", category="statistics", description="滚动均值",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_unary(x, w, lambda c, n: c.rolling_mean(window_size=n, min_samples=1))


class VarPolars(SeriesOperator):
    """Polars 滚动方差"""
    metadata = OperatorMetadata(
        name="Var", category="statistics", description="滚动方差",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_unary(x, w, lambda c, n: c.rolling_var(window_size=n, min_samples=1))


class SkewPolars(SeriesOperator):
    """Polars 滚动偏度"""
    metadata = OperatorMetadata(
        name="Skew", category="statistics", description="滚动偏度",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_unary(x, w, lambda c, n: c.rolling_skew(window_size=n, min_samples=1))


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


class MedianPolars(SeriesOperator):
    """Polars 滚动中位数"""
    metadata = OperatorMetadata(
        name="Median", category="statistics", description="滚动中位数",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_unary(x, w, lambda c, n: c.rolling_median(window_size=n, min_samples=1))


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


class CovariancePolars(CovPolars):
    """Polars 滚动协方差"""
    metadata = OperatorMetadata(
        name="Covariance", category="statistics", description="滚动协方差",
        param_names=["x", "y", "window"], return_type="series", tags=["statistics", "polars"],
    )


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


class ResidualPolars(SeriesOperator):
    """Polars 滚动回归残差**均值**。

    语义（round-3 audit）：输出 = 滚动窗口内**样本内当前残差**的窗口均值
    （residual_mean），与 pandas ``residual``/``Residual`` 一致——先计算
    ``y_t - ŷ_t``（用当前窗口 OLS 拟合），再对残差序列取滚动均值。**不是**
    原始残差序列，**不是** out-of-sample 预测误差。
    """
    metadata = OperatorMetadata(
        name="residual", category="statistics",
        description="滚动回归残差均值（样本内当前残差的滚动均值）",
        param_names=["y", "x", "window"], return_type="series", tags=["statistics", "polars"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import rolling_regression

        w = max(int(kwargs.get("d", window)), 2)
        cols = _align_cols(y, x)
        py = y.select(cols).to_pandas()
        px = x.select(cols).to_pandas()
        out = rolling_regression(py, px, window=w, min_periods=2, retval="residual")
        return y.with_columns([
            pl.Series(name=c, values=np.asarray(out[c], dtype=np.float64)) for c in cols
        ])


class ModePolars(SeriesOperator):
    """Polars 滚动众数。

    语义（round-3 audit）：仅当最高频次 >= 2 时输出众数；全不同值窗口返回
    ``NaN``（不退化为一期 min）；众数并列时返回众数集合的**中位数**。
    """
    metadata = OperatorMetadata(
        name="Mode", category="statistics",
        description="滚动众数（频次>=2 才输出，并列取中位数，全不同值→NaN）",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 1)

        def _mode(arr: np.ndarray) -> float:
            a = np.asarray(arr, dtype=np.float64)
            valid = a[np.isfinite(a)]
            if len(valid) == 0:
                return np.nan
            vals, counts = np.unique(valid, return_counts=True)
            max_freq = int(counts.max())
            if max_freq < 2:
                return np.nan
            modes = vals[counts == max_freq]
            return float(np.median(modes))

        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).rolling_map(_mode, window_size=w, min_samples=1).alias(c) for c in cols
        ])


class AutocorrPolars(SeriesOperator):
    """Polars 自相关系数。

    语义（round-3 audit）：与 pandas ``autocorr`` 一致——``lag==0`` 时
    ``corr(x, x) = 1.0``（不再被静默改写成 lag=1）；短样本 / 常数序列 /
    不可行 lag 时 ``rolling_corr`` 返回 null（NaN），不伪造 0。
    """
    metadata = OperatorMetadata(
        name="autocorr", category="statistics", description="自相关系数",
        param_names=["x", "lag"], return_type="series", tags=["statistics", "polars"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=0, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        l = int(kwargs.get("d", lag))
        if l < 0:
            from factor_engine.backend.operator_errors import FutureReferenceError

            raise FutureReferenceError(f"negative lag {l} references future data")

        def _expanding_autocorr(values: np.ndarray) -> list[float]:
            result = np.full(len(values), np.nan, dtype=np.float64)
            for i in range(len(values)):
                prefix = np.asarray(values[: i + 1], dtype=np.float64)
                if l == 0:
                    if np.isfinite(prefix).sum() >= 2:
                        result[i] = 1.0
                    continue

                # Preserve physical row positions: a missing value at either
                # endpoint invalidates that lagged pair instead of reconnecting
                # the finite observations on either side of the gap.
                if prefix.size <= l:
                    continue
                left = prefix[:-l]
                right = prefix[l:]
                pairs = np.isfinite(left) & np.isfinite(right)
                if int(pairs.sum()) < 2:
                    continue
                left = left[pairs]
                right = right[pairs]
                left_centered = left - left.mean()
                right_centered = right - right.mean()
                denom = float(
                    np.sqrt(np.dot(left_centered, left_centered)
                           * np.dot(right_centered, right_centered))
                )
                if denom == 0.0 or not np.isfinite(denom):
                    continue
                result[i] = float(np.dot(left_centered, right_centered) / denom)
            return result.tolist()

        cols = _numeric_cols(x)
        return x.with_columns([
            pl.Series(name=c, values=_expanding_autocorr(x[c].to_numpy()))
            for c in cols
        ])


class betaPolars(BetaPolars):
    """Polars 滚动 Beta（小写）"""
    metadata = OperatorMetadata(
        name="beta", category="statistics", description="滚动 Beta（小写）",
        param_names=["y", "x", "window"], return_type="series", tags=["statistics", "polars"],
    )


class CorrTestPolars(SeriesOperator):
    """扩展窗口 Pearson 相关检验 p 值；前缀样本数 < 3 时为 NaN。"""

    metadata = OperatorMetadata(
        name="corr_test", category="statistics", description="Pearson 相关检验 p 值",
        param_names=["x", "y"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from scipy import stats

        from factor_engine.cleaned_operators._causal import expanding_bivariate

        cols = _align_cols(x, y)
        pdf_x = x.select(cols).to_pandas()
        pdf_y = y.select(cols).to_pandas()
        out = expanding_bivariate(
            pdf_x, pdf_y, lambda a, b: stats.pearsonr(a, b)[1], min_periods=3
        )
        return x.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


class DurbinWatsonTestPolars(SeriesOperator):
    """扩展窗口 Durbin-Watson 统计量（非 p 值）；用于残差自相关诊断。"""

    metadata = OperatorMetadata(
        name="durbin_watson_test", category="statistics", description="Durbin-Watson 统计量",
        param_names=["residuals"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, residuals: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from statsmodels.stats.stattools import durbin_watson

        from factor_engine.cleaned_operators._causal import expanding_univariate

        cols = _numeric_cols(residuals)
        pdf = residuals.select(cols).to_pandas()
        out = expanding_univariate(pdf, durbin_watson, min_periods=2)
        return residuals.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


class MeanAbsDeviationPolars(SeriesOperator):
    """Polars 滚动平均绝对离差：``mean(|x_i - mean(window)|)``。

    语义（round-3 audit）：单一中心=窗口均值，一个窗口输出一个值；与 pandas
    ``ts_mean_abs_deviation`` 同义（非 ``ts_mad`` 的双重滚动近似）。
    """
    metadata = OperatorMetadata(
        name="ts_mean_abs_deviation", category="statistics",
        description="平均绝对离差（围绕窗口均值的单一中心）",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 1)
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).rolling_map(_mean_abs_dev_1d, window_size=w, min_samples=1).alias(c) for c in cols
        ])


class MedianAbsDeviationPolars(SeriesOperator):
    """Polars 滚动中位数绝对离差（标准 MAD）：``median(|x_i - median(window)|)``。

    语义（round-3 audit）：单一中心=窗口中位数，一个窗口输出一个值；与 pandas
    ``ts_median_abs_deviation`` 同义（非 ``ts_mad`` 的双重滚动近似，不带
    1.4826 比例因子）。
    """
    metadata = OperatorMetadata(
        name="ts_median_abs_deviation", category="statistics",
        description="中位数绝对离差（围绕窗口中位数的单一中心）",
        param_names=["x", "window"], return_type="series", tags=["statistics", "polars"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 1)
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).rolling_map(_median_abs_dev_1d, window_size=w, min_samples=1).alias(c) for c in cols
        ])


# ---------------------------------------------------------------------------
# in-module aliases（round-3 audit）：长拼写 -> 短 canonical（与 statistics.py 一致）。
# ---------------------------------------------------------------------------
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402

OperatorRegistry.register_alias("ts_mean_absolute_deviation", "ts_mean_abs_deviation")
OperatorRegistry.register_alias("ts_median_absolute_deviation", "ts_median_abs_deviation")
