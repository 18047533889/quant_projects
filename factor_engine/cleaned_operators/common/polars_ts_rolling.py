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

_PAIRWISE_CANONICALS = frozenset({
    "ts_corr",
    "ts_cov",
    "ts_regression_slope",
    "ts_regression_intercept",
    "ts_regression_resid",
    "ts_regression_r2",
})


def _register_rolling_operator(**kwargs):
    """Register only the repaired pairwise surface during normal bootstrap."""
    if kwargs.get("canonical") in _PAIRWISE_CANONICALS:
        return register_operator(**kwargs)
    return lambda cls: cls

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


def _pairwise_rolling_moments(
    x_col: pl.Expr,
    y_col: pl.Expr,
    *,
    window: int,
    min_periods: int,
) -> dict[str, pl.Expr]:
    """Return direct-window moments over finite ``(x, y)`` pairs only."""
    valid = x_col.is_finite().fill_null(False) & y_col.is_finite().fill_null(False)
    x_valid = pl.when(valid).then(x_col).otherwise(None)
    y_valid = pl.when(valid).then(y_col).otherwise(None)
    count = valid.cast(pl.Float64).rolling_sum(window_size=window, min_samples=1)
    # Translation does not change centered moments.  Subtract one finite global
    # anchor before rolling arithmetic so large common offsets cannot erase the
    # within-window variation in sum-of-squares identities.
    x_anchor = x_valid.forward_fill().backward_fill().first()
    y_anchor = y_valid.forward_fill().backward_fill().first()
    x_centered = x_valid - x_anchor
    y_centered = y_valid - y_anchor
    sum_x = x_centered.rolling_sum(window_size=window, min_samples=1)
    sum_y = y_centered.rolling_sum(window_size=window, min_samples=1)
    sum_xx = (x_centered * x_centered).rolling_sum(window_size=window, min_samples=1)
    sum_yy = (y_centered * y_centered).rolling_sum(window_size=window, min_samples=1)
    sum_xy = (x_centered * y_centered).rolling_sum(window_size=window, min_samples=1)
    ready = count >= min_periods

    return {
        "count": count,
        "mean_x": pl.when(ready).then(x_anchor + sum_x / count).otherwise(None),
        "mean_y": pl.when(ready).then(y_anchor + sum_y / count).otherwise(None),
        "cross": pl.when(ready).then(sum_xy - sum_x * sum_y / count).otherwise(None),
        "ss_x": pl.when(ready).then(sum_xx - sum_x * sum_x / count).otherwise(None),
        "ss_y": pl.when(ready).then(sum_yy - sum_y * sum_y / count).otherwise(None),
        "x_endpoint": x_valid,
        "y_endpoint": y_valid,
    }


# ---------------------------------------------------------------------------
# Rolling correlation & covariance
# ---------------------------------------------------------------------------


@_register_rolling_operator(
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
        param_names=["x", "y", "window", "min_periods"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int = 2, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        if y is None:
            raise ValueError("ts_corr requires y")

        exprs = []
        for c in cols:
            x_col = pl.col(c)
            y_col = y[c] if c in y.columns else pl.lit(None, dtype=pl.Float64)

            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=min_periods)
            corr = pl.when((moments["ss_x"] <= 0) | (moments["ss_y"] <= 0)).then(
                None
            ).otherwise(moments["cross"] / (moments["ss_x"] * moments["ss_y"]).sqrt())
            exprs.append(corr.alias(c))

        return x.lazy().with_columns(exprs).collect()


@_register_rolling_operator(
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
        param_names=["x", "y", "window", "ddof", "min_periods"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None,
                         window: int = 20, ddof: int = 1,
                         min_periods: int | None = None, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        ddof_val = strict_integer(ddof, "ddof", minimum=0)

        # Default min_periods to 2 if not specified (need at least 2 pairs)
        if min_periods is None:
            min_p = 2
        else:
            min_p = strict_integer(min_periods, "min_periods", minimum=1)

        cols = _numeric_cols(x)

        if y is None:
            raise ValueError("ts_cov requires y")

        exprs = []
        for c in cols:
            x_col = pl.col(c)
            y_col = y[c] if c in y.columns else pl.lit(None, dtype=pl.Float64)

            moments = _pairwise_rolling_moments(
                x_col, y_col, window=w, min_periods=min_p
            )
            cov = pl.when(moments["count"] > ddof_val).then(
                moments["cross"] / (moments["count"] - ddof_val)
            ).otherwise(None)

            exprs.append(cov.alias(c))

        return x.lazy().with_columns(exprs).collect()


@_register_rolling_operator(
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
        alpha = 2.0 / (w + 1)

        if y is None:
            raise ValueError("ts_ewm_corr requires y")

        exprs = []
        for c in cols:
            x_col = pl.col(c)
            y_col = y[c] if c in y.columns else pl.lit(None, dtype=pl.Float64)

            x_mean = x_col.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            y_mean = y_col.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)

            x_var = ((x_col - x_mean) ** 2).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            y_var = ((y_col - y_mean) ** 2).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            cov = ((x_col - x_mean) * (y_col - y_mean)).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)

            x_std = x_var.sqrt()
            y_std = y_var.sqrt()

            corr = pl.when((x_std == 0) | (y_std == 0) | x_std.is_null() | y_std.is_null()).then(None).otherwise(cov / (x_std * y_std))
            exprs.append(corr.alias(c))

        return x.lazy().with_columns(exprs).collect()


# ---------------------------------------------------------------------------
# Rolling regression
# ---------------------------------------------------------------------------


@_register_rolling_operator(
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
        param_names=["y", "x", "window", "lag", "retval", "min_periods", "add_intercept"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, *legacy_args, lag=None, retval=None, min_periods: int = 2, add_intercept=True, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(y)

        if x is None:
            raise ValueError("ts_regression_slope requires x")

        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c] if c in x.columns else pl.lit(None, dtype=pl.Float64)

            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=min_periods)
            slope = pl.when(moments["ss_x"] <= 0).then(None).otherwise(
                moments["cross"] / moments["ss_x"]
            )
            exprs.append(slope.alias(c))

        return y.with_columns(exprs)


@_register_rolling_operator(
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
        param_names=["y", "x", "window", "min_periods", "add_intercept"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int = 2, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(y)

        if x is None:
            raise ValueError("ts_regression_intercept requires x")

        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c] if c in x.columns else pl.lit(None, dtype=pl.Float64)

            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=min_periods)
            slope = pl.when(moments["ss_x"] <= 0).then(None).otherwise(
                moments["cross"] / moments["ss_x"]
            )
            intercept = moments["mean_y"] - slope * moments["mean_x"]
            exprs.append(intercept.alias(c))

        return y.with_columns(exprs)


@_register_rolling_operator(
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
        param_names=["y", "x", "window", "min_periods", "add_intercept"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int = 2, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(y)

        if x is None:
            raise ValueError("ts_regression_resid requires x")

        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c] if c in x.columns else pl.lit(None, dtype=pl.Float64)

            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=min_periods)
            slope = pl.when(moments["ss_x"] <= 0).then(None).otherwise(
                moments["cross"] / moments["ss_x"]
            )
            resid = (moments["y_endpoint"] - moments["mean_y"]) - slope * (
                moments["x_endpoint"] - moments["mean_x"]
            )

            exprs.append(resid.alias(c))

        return y.with_columns(exprs)


@_register_rolling_operator(
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
        param_names=["y", "x", "window", "min_periods", "add_intercept"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None,
                         window: int = 20, min_periods: int = 2, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(y)

        if x is None:
            raise ValueError("ts_regression_r2 requires x")

        exprs = []
        for c in cols:
            y_col = pl.col(c)
            x_col = x[c] if c in x.columns else pl.lit(None, dtype=pl.Float64)

            moments = _pairwise_rolling_moments(x_col, y_col, window=w, min_periods=min_periods)
            r2 = pl.when((moments["ss_x"] <= 0) | (moments["ss_y"] <= 0)).then(None).otherwise(
                (moments["cross"] ** 2) / (moments["ss_x"] * moments["ss_y"])
            )
            exprs.append(r2.alias(c))

        return y.with_columns(exprs)


# ---------------------------------------------------------------------------
# Decay & Trend
# ---------------------------------------------------------------------------


@_register_rolling_operator(
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
        weight_sum = (w * (w + 1)) / 2.0

        exprs = []
        for c in cols:
            weighted_sum = pl.lit(0.0)
            for i in range(w):
                weight = w - i
                weighted_sum = weighted_sum + pl.col(c).shift(i) * weight

            exprs.append((weighted_sum / weight_sum).alias(c))

        return x.lazy().with_columns(exprs).collect()


@_register_rolling_operator(
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
        t_mean = (w - 1) / 2.0
        var_t = sum((i - t_mean) ** 2 for i in range(w)) / w

        exprs = []
        for c in cols:
            y_mean = pl.col(c).rolling_mean(window_size=w, min_samples=2)

            # cov(t, y) = sum((t_i - t_mean) * (y_i - y_mean)) / n
            cov_sum = pl.lit(0.0)
            for i in range(w):
                t_dev = i - t_mean
                y_dev = pl.col(c).shift(w - 1 - i) - y_mean
                cov_sum = cov_sum + t_dev * y_dev

            cov_t_y = cov_sum / w
            slope = cov_t_y / var_t if var_t != 0 else np.nan

            exprs.append(slope.alias(c))

        return x.lazy().with_columns(exprs).collect()


@_register_rolling_operator(
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


@_register_rolling_operator(
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
            pct_rank = pl.when(count == 0).then(None).otherwise(rank / count)
            exprs.append(pct_rank.alias(c))

        return x.lazy().with_columns(exprs).collect()


# <CONTINUATION_MARKER_TS_ROLLING>
