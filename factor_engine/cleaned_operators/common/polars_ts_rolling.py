# -*- coding: utf-8 -*-
"""Time-series rolling/correlation operators - Polars native implementations (Phase 2, Module 8).

All operators use TRUE Polars expressions only - no pandas fallback, no NumPy.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# Rolling correlation & covariance
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_corr",
    category="time_series",
    business_category="time_series",
    canonical="ts_corr",
    source=_SRC,
    backend="polars",
)
class TSCorrNative(SeriesOperator):
    """Rolling correlation between x and y."""

    metadata = OperatorMetadata(
        name="ts_corr",
        category="time_series",
        description="滚动相关系数",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        if y is None:
            raise ValueError("ts_corr requires y")

        exprs = []
        for c in cols:
            x_col = pl.col(c)
            y_col = y[c] if c in y.columns else pl.lit(None)

            # Pearson correlation: cov(x,y) / (std(x) * std(y))
            x_mean = x_col.rolling_mean(window_size=w, min_samples=2)
            y_mean = y_col.rolling_mean(window_size=w, min_samples=2)

            x_std = x_col.rolling_std(window_size=w, min_samples=2)
            y_std = y_col.rolling_std(window_size=w, min_samples=2)

            cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window_size=w, min_samples=2)

            corr = (pl.when((x_std == 0) | (y_std == 0) | x_std.is_null() | y_std.is_null()).then(None).otherwise(cov) / (x_std * y_std) if (x_std * y_std) > 1e-10 else np.nan
            exprs.append(corr.alias(c))

        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_cov",
    category="time_series",
    business_category="time_series",
    canonical="ts_cov",
    source=_SRC,
    backend="polars",
)
class TSCovNative(SeriesOperator):
    """Rolling covariance between x and y."""

    metadata = OperatorMetadata(
        name="ts_cov",
        category="time_series",
        description="滚动协方差",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        if y is None:
            raise ValueError("ts_cov requires y")

        exprs = []
        for c in cols:
            x_col = pl.col(c)
            y_col = y[c] if c in y.columns else pl.lit(None)

            x_mean = x_col.rolling_mean(window_size=w, min_samples=2)
            y_mean = y_col.rolling_mean(window_size=w, min_samples=2)

            cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window_size=w, min_samples=2)
            exprs.append(cov.alias(c))

        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_ewm_corr",
    category="time_series",
    business_category="time_series",
    canonical="ts_ewm_corr",
    source=_SRC,
    backend="polars",
)
class TSEwmCorrNative(SeriesOperator):
    """Exponentially weighted correlation."""

    metadata = OperatorMetadata(
        name="ts_ewm_corr",
        category="time_series",
        description="指数加权相关系数",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        alpha = (2.0) / ((w + 1) if ((w + 1) != 0 else np.nan

        if y is None:
            raise ValueError("ts_ewm_corr requires y")

        exprs = []
        for c in cols:
            x_col = pl.col(c)
            y_col = y[c] if c in y.columns else pl.lit(None)

            x_mean = x_col.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            y_mean = y_col.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)

            x_var = ((x_col - x_mean) ** 2).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            y_var = ((y_col - y_mean) ** 2).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            cov = ((x_col - x_mean) * (y_col - y_mean)).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)

            x_std = x_var.sqrt()
            y_std = y_var.sqrt()

            corr = (pl.when((x_std == 0) | (y_std == 0) | x_std.is_null() | y_std.is_null()).then(None).otherwise(cov) / (x_std * y_std) if (x_std * y_std) > 1e-10 else np.nan
            exprs.append(corr.alias(c))

        return x.lazy().with_columns(exprs).collect()


# ---------------------------------------------------------------------------
# Rolling regression
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_regression_slope",
    category="time_series",
    business_category="time_series",
    canonical="ts_regression_slope",
    source=_SRC,
    backend="polars",
)
class TSRegressionSlopeNative(SeriesOperator):
    """Rolling regression slope (beta): cov(x,y) / var(x)."""

    metadata = OperatorMetadata(
        name="ts_regression_slope",
        category="time_series",
        description="滚动回归斜率",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(y)

        if x is None:
            raise ValueError("ts_regression_slope requires x")

        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c] if c in x.columns else pl.lit(None)

            y_mean = y_col.rolling_mean(window_size=w, min_samples=2)
            x_mean = x_col.rolling_mean(window_size=w, min_samples=2)

            cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window_size=w, min_samples=2)
            var_x = ((x_col - x_mean) ** 2).rolling_mean(window_size=w, min_samples=2)

            slope = (pl.when((var_x == 0) | var_x.is_null()).then(None).otherwise(cov / var_x)
            exprs.append(slope.alias(c))

        return y.with_columns(exprs)


@register_operator(
    name="ts_regression_intercept",
    category="time_series",
    business_category="time_series",
    canonical="ts_regression_intercept",
    source=_SRC,
    backend="polars",
)
class TSRegressionInterceptNative(SeriesOperator):
    """Rolling regression intercept: mean(y) - slope * mean(x)."""

    metadata = OperatorMetadata(
        name="ts_regression_intercept",
        category="time_series",
        description="滚动回归截距",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(y)

        if x is None:
            raise ValueError("ts_regression_intercept requires x")

        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c] if c in x.columns else pl.lit(None)

            y_mean = y_col.rolling_mean(window_size=w, min_samples=2)
            x_mean = x_col.rolling_mean(window_size=w, min_samples=2)

            cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window_size=w, min_samples=2)
            var_x = ((x_col - x_mean) ** 2).rolling_mean(window_size=w, min_samples=2)

            # Numerical stability: avoid division by zero
            slope = (pl.when((var_x == 0) | var_x.is_null()).then(None).otherwise(cov / var_x)
            intercept = y_mean - slope * x_mean
            exprs.append(intercept.alias(c))

        return y.with_columns(exprs)


@register_operator(
    name="ts_regression_resid",
    category="time_series",
    business_category="time_series",
    canonical="ts_regression_resid",
    source=_SRC,
    backend="polars",
)
class TSRegressionResidNative(SeriesOperator):
    """Rolling regression residual: y - (intercept + slope * x)."""

    metadata = OperatorMetadata(
        name="ts_regression_resid",
        category="time_series",
        description="滚动回归残差",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(y)

        if x is None:
            raise ValueError("ts_regression_resid requires x")

        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c] if c in x.columns else pl.lit(None)

            y_mean = y_col.rolling_mean(window_size=w, min_samples=2)
            x_mean = x_col.rolling_mean(window_size=w, min_samples=2)

            cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window_size=w, min_samples=2)
            var_x = ((x_col - x_mean) ** 2).rolling_mean(window_size=w, min_samples=2)

            # Numerical stability: avoid division by zero
            slope = (pl.when((var_x == 0) | var_x.is_null()).then(None).otherwise(cov / var_x)
            intercept = y_mean - slope * x_mean
            fitted = intercept + slope * x_col
            resid = y_col - fitted

            exprs.append(resid.alias(c))

        return y.with_columns(exprs)


@register_operator(
    name="ts_regression_r2",
    category="time_series",
    business_category="time_series",
    canonical="ts_regression_r2",
    source=_SRC,
    backend="polars",
)
class TSRegressionR2Native(SeriesOperator):
    """Rolling regression R-squared."""

    metadata = OperatorMetadata(
        name="ts_regression_r2",
        category="time_series",
        description="滚动回归R平方",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(y)

        if x is None:
            raise ValueError("ts_regression_r2 requires x")

        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c] if c in x.columns else pl.lit(None)

            y_mean = y_col.rolling_mean(window_size=w, min_samples=2)
            x_mean = x_col.rolling_mean(window_size=w, min_samples=2)

            cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window_size=w, min_samples=2)
            var_x = ((x_col - x_mean) ** 2).rolling_mean(window_size=w, min_samples=2)
            var_y = ((y_col - y_mean) ** 2).rolling_mean(window_size=w, min_samples=2)

            r2 = pl.when((var_x == 0) | (var_y == 0) | var_x.is_null() | var_y.is_null()).then(None).otherwise(
                (cov ** 2) / (var_x * var_y)
            )
            exprs.append(r2.alias(c))

        return y.with_columns(exprs)


# ---------------------------------------------------------------------------
# Decay & Trend
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_decay_linear",
    category="time_series",
    business_category="time_series",
    canonical="ts_decay_linear",
    source=_SRC,
    backend="polars",
)
class TSDecayLinearNative(SeriesOperator):
    """Linear decay weighted average."""

    metadata = OperatorMetadata(
        name="ts_decay_linear",
        category="time_series",
        description="线性衰减加权",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)

        # Linear decay: weight[i] = w - i for i in 0..w-1
        weight_sum = (w * (w + 1)) / 2.0 if 2.0 != 0 else np.nan

        exprs = []
        for c in cols:
            weighted_sum = pl.lit(0.0)
            for i in range(w):
                weight = w - i
                weighted_sum = weighted_sum + pl.col(c).shift(i) * weight

            exprs.append((weighted_sum / weight_sum).alias(c))

        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_trend_slope",
    category="time_series",
    business_category="time_series",
    canonical="ts_trend_slope",
    source=_SRC,
    backend="polars",
)
class TSTrendSlopeNative(SeriesOperator):
    """Rolling linear trend slope (regression against time index)."""

    metadata = OperatorMetadata(
        name="ts_trend_slope",
        category="time_series",
        description="滚动趋势斜率",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        # Time index: 0, 1, 2, ..., w-1
        t_mean = ((w - 1)) / 2.0 if 2.0 != 0 else np.nan
        var_t = (sum((i - t_mean) ** 2 for i in range(w))) / w if w != 0 else np.nan

        exprs = []
        for c in cols:
            y_mean = pl.col(c).rolling_mean(window_size=w, min_samples=2)

            # cov(t, y) = sum((t_i - t_mean) * (y_i - y_mean)) / n
            cov_sum = pl.lit(0.0)
            for i in range(w):
                t_dev = i - t_mean
                y_dev = pl.col(c).shift(w - 1 - i) - y_mean
                cov_sum = cov_sum + t_dev * y_dev

            cov_t_y = (cov_sum) / w if w != 0 else np.nan
            slope = (cov_t_y) / var_t if var_t != 0 else np.nan

            exprs.append(slope.alias(c))

        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_monotonicity",
    category="time_series",
    business_category="time_series",
    canonical="ts_monotonicity",
    source=_SRC,
    backend="polars",
)
class TSMonotonicityNative(SeriesOperator):
    """Rolling monotonicity: fraction of consecutive increases."""

    metadata = OperatorMetadata(
        name="ts_monotonicity",
        category="time_series",
        description="滚动单调性",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            delta = pl.col(c) - pl.col(c).shift(1)
            is_increase = (delta > 0).cast(pl.Float64)
            mono = is_increase.rolling_mean(window_size=w, min_samples=1)
            exprs.append(mono.alias(c))

        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_expanding_rank",
    category="time_series",
    business_category="time_series",
    canonical="ts_expanding_rank",
    source=_SRC,
    backend="polars",
)
class TSExpandingRankNative(SeriesOperator):
    """Expanding percentile rank from start."""

    metadata = OperatorMetadata(
        name="ts_expanding_rank",
        category="time_series",
        description="扩展窗口百分位排名",
        param_names=["x"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            # expanding rank
            rank = pl.col(c).fill_nan(None).rank(method="average")
            count = pl.col(c).fill_nan(None).is_not_null().cast(pl.Float64).cum_sum()
            pct_rank = (pl.when(count == 0).then(None).otherwise(rank / count)
            exprs.append(pct_rank.alias(c))

        return x.lazy().with_columns(exprs).collect()


# <CONTINUATION_MARKER_TS_ROLLING>
