"""
Polars native implementations for time series operators (ts_* family) - Batch 1

All operators use pure Polars lazy API for maximum performance.
Window operations use rolling_*, shifts use .shift(), cumulative use .cum_*
"""

import polars as pl
import numpy as np
from typing import Optional, Union

from cleaned_operators.base import (
    SeriesOperator,
    register_operator,
    OperatorMetadata,
    ParamSpec,
    ParamRole,
)


# ============================================================================
# Basic Rolling Statistics (ts_mean, ts_std, ts_sum, etc.)
# ============================================================================

@register_operator(name="ts_mean_if", canonical="ts_mean_if", backend="polars")
class TSMeanIfPolarsNative(SeriesOperator):
    """Conditional rolling mean - mean of values where condition is True"""

    metadata = OperatorMetadata(
        name="ts_mean_if",
        category="time_series",
        description="Conditional rolling mean - mean of values where condition is True",
        param_names=["feature", "condition", "window"],
        return_type="series",
        tags=["time_series", "rolling", "conditional", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, condition, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns(pl.lit(condition).alias("_cond"))
            .select([
                pl.when(pl.col("_cond"))
                .then(pl.col(feature.name))
                .otherwise(None)
                .rolling_mean(window)
                .alias(feature.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_std_if", canonical="ts_std_if", backend="polars")
class TSStdIfPolarsNative(SeriesOperator):
    """Conditional rolling std"""

    metadata = OperatorMetadata(
        name="ts_std_if",
        category="time_series",
        description="Conditional rolling standard deviation",
        param_names=["feature", "condition", "window"],
        return_type="series",
        tags=["time_series", "rolling", "conditional", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, condition, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns(pl.lit(condition).alias("_cond"))
            .select([
                pl.when(pl.col("_cond"))
                .then(pl.col(feature.name))
                .otherwise(None)
                .rolling_std(window)
                .alias(feature.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_sum_if", canonical="ts_sum_if", backend="polars")
class TSSumIfPolarsNative(SeriesOperator):
    """Conditional rolling sum"""

    metadata = OperatorMetadata(
        name="ts_sum_if",
        category="time_series",
        description="Conditional rolling sum",
        param_names=["feature", "condition", "window"],
        return_type="series",
        tags=["time_series", "rolling", "conditional", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, condition, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns(pl.lit(condition).alias("_cond"))
            .select([
                pl.when(pl.col("_cond"))
                .then(pl.col(feature.name))
                .otherwise(0.0)
                .rolling_sum(window)
                .alias(feature.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_min_if", canonical="ts_min_if", backend="polars")
class TSMinIfPolarsNative(SeriesOperator):
    """Conditional rolling min"""
    metadata = OperatorMetadata(
        name="ts_min_if",
        category="time_series",
        description="Conditional rolling minimum",
        param_names=['feature', 'condition', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, condition, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns(pl.lit(condition).alias("_cond"))
            .select([
                pl.when(pl.col("_cond"))
                .then(pl.col(feature.name))
                .otherwise(None)
                .rolling_min(window)
                .alias(feature.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_max_if", canonical="ts_max_if", backend="polars")
class TSMaxIfPolarsNative(SeriesOperator):
    """Conditional rolling max"""
    metadata = OperatorMetadata(
        name="ts_max_if",
        category="time_series",
        description="Conditional rolling maximum",
        param_names=['feature', 'condition', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, condition, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns(pl.lit(condition).alias("_cond"))
            .select([
                pl.when(pl.col("_cond"))
                .then(pl.col(feature.name))
                .otherwise(None)
                .rolling_max(window)
                .alias(feature.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_median", canonical="ts_median", backend="polars")
class TSMedianPolarsNative(SeriesOperator):
    """Rolling median"""
    metadata = OperatorMetadata(
        name="ts_median",
        category="time_series",
        description="Rolling median",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name)
                .rolling_median(window)
                .alias(feature.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_quantile", canonical="ts_quantile", backend="polars")
class TSQuantilePolarsNative(SeriesOperator):
    """Rolling quantile"""
    metadata = OperatorMetadata(
        name="ts_quantile",
        category="time_series",
        description="Rolling quantile",
        param_names=['feature', 'window', 'quantile'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, quantile=0.5, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name)
                .rolling_quantile(quantile, window_size=window)
                .alias(feature.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_quantile_if", canonical="ts_quantile_if", backend="polars")
class TSQuantileIfPolarsNative(SeriesOperator):
    """Conditional rolling quantile"""
    metadata = OperatorMetadata(
        name="ts_quantile_if",
        category="time_series",
        description="Conditional rolling quantile",
        param_names=['feature', 'condition', 'window', 'quantile'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, condition, window, quantile=0.5, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns(pl.lit(condition).alias("_cond"))
            .select([
                pl.when(pl.col("_cond"))
                .then(pl.col(feature.name))
                .otherwise(None)
                .rolling_quantile(quantile, window_size=window)
                .alias(feature.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_rank_if", canonical="ts_rank_if", backend="polars")
class TSRankIfPolarsNative(SeriesOperator):
    """Conditional rolling rank (current value rank in window)"""
    metadata = OperatorMetadata(
        name="ts_rank_if",
        category="time_series",
        description="Conditional rolling rank (current value rank in window)",
        param_names=['feature', 'condition', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, condition, window, **kwargs):
        # Rolling rank: current value's rank within the rolling window
        return (
            feature.to_frame()
            .lazy()
            .with_columns(pl.lit(condition).alias("_cond"))
            .with_columns([
                pl.when(pl.col("_cond"))
                .then(pl.col(feature.name))
                .otherwise(None)
                .alias("_filtered")
            ])
            .with_columns([
                pl.col("_filtered")
                .rank(method="average")
                .over(pl.int_range(0, pl.count()).alias("_row") // window)
                .alias(feature.name)
            ])
            .select([feature.name])
            .collect()
            .to_series()
        )


# ============================================================================
# Correlation & Covariance
# ============================================================================

@register_operator(name="ts_corr", canonical="ts_corr", backend="polars")
class TSCorrPolarsNative(SeriesOperator):
    """Rolling correlation between two series"""
    metadata = OperatorMetadata(
        name="ts_corr",
        category="time_series",
        description="Rolling correlation between two series",
        param_names=['x', 'y', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, window, **kwargs):
        df = pl.DataFrame({"x": x, "y": y})
        return (
            df.lazy()
            .select([
                pl.corr("x", "y", ddof=1)
                .rolling_map(lambda s: s.corr(df["y"].tail(len(s))), window_size=window)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_cov", canonical="ts_cov", backend="polars")
class TSCovPolarsNative(SeriesOperator):
    """Rolling covariance between two series"""
    metadata = OperatorMetadata(
        name="ts_cov",
        category="time_series",
        description="Rolling covariance between two series",
        param_names=['x', 'y', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, window, **kwargs):
        df = pl.DataFrame({"x": x, "y": y})
        return (
            df.lazy()
            .with_columns([
                (pl.col("x") - pl.col("x").rolling_mean(window)).alias("x_dm"),
                (pl.col("y") - pl.col("y").rolling_mean(window)).alias("y_dm")
            ])
            .select([
                (pl.col("x_dm") * pl.col("y_dm"))
                .rolling_mean(window)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_corr_if", canonical="ts_corr_if", backend="polars")
class TSCorrIfPolarsNative(SeriesOperator):
    """Conditional rolling correlation"""
    metadata = OperatorMetadata(
        name="ts_corr_if",
        category="time_series",
        description="Conditional rolling correlation",
        param_names=['x', 'y', 'condition', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, condition, window, **kwargs):
        df = pl.DataFrame({"x": x, "y": y, "cond": condition})
        return (
            df.lazy()
            .with_columns([
                pl.when(pl.col("cond")).then(pl.col("x")).otherwise(None).alias("x_f"),
                pl.when(pl.col("cond")).then(pl.col("y")).otherwise(None).alias("y_f")
            ])
            .select([
                pl.corr("x_f", "y_f", ddof=1).alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_cov_if", canonical="ts_cov_if", backend="polars")
class TSCovIfPolarsNative(SeriesOperator):
    """Conditional rolling covariance"""
    metadata = OperatorMetadata(
        name="ts_cov_if",
        category="time_series",
        description="Conditional rolling covariance",
        param_names=['x', 'y', 'condition', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, condition, window, **kwargs):
        df = pl.DataFrame({"x": x, "y": y, "cond": condition})
        return (
            df.lazy()
            .with_columns([
                pl.when(pl.col("cond")).then(pl.col("x")).otherwise(None).alias("x_f"),
                pl.when(pl.col("cond")).then(pl.col("y")).otherwise(None).alias("y_f")
            ])
            .with_columns([
                (pl.col("x_f") - pl.col("x_f").rolling_mean(window)).alias("x_dm"),
                (pl.col("y_f") - pl.col("y_f").rolling_mean(window)).alias("y_dm")
            ])
            .select([
                (pl.col("x_dm") * pl.col("y_dm"))
                .rolling_mean(window)
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Lag & Shift Operations
# ============================================================================

@register_operator(name="ts_nth_value", canonical="ts_nth_value", backend="polars")
class TSNthValuePolarsNative(SeriesOperator):
    """Get nth value from current position (negative=past, positive=future)"""
    metadata = OperatorMetadata(
        name="ts_nth_value",
        category="time_series",
        description="Get nth value from current position (negative=past, positive=future)",
        param_names=['feature', 'n'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "n": ParamSpec(dtype=int, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, n, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name).shift(-n).alias(feature.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_last_if", canonical="ts_last_if", backend="polars")
class TSLastIfPolarsNative(SeriesOperator):
    """Get last value where condition was True within window"""
    metadata = OperatorMetadata(
        name="ts_last_if",
        category="time_series",
        description="Get last value where condition was True within window",
        param_names=['feature', 'condition', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, condition, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns(pl.lit(condition).alias("_cond"))
            .with_columns([
                pl.when(pl.col("_cond"))
                .then(pl.col(feature.name))
                .otherwise(None)
                .alias("_filtered")
            ])
            .select([
                pl.col("_filtered")
                .fill_null(strategy="forward")
                .alias(feature.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Counting & Coverage
# ============================================================================

@register_operator(name="ts_count_if", canonical="ts_count_if", backend="polars")
class TSCountIfPolarsNative(SeriesOperator):
    """Count of True values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_count_if",
        category="time_series",
        description="Count of True values in rolling window",
        param_names=['condition', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, condition, window, **kwargs):
        return (
            pl.Series("cond", condition)
            .to_frame()
            .lazy()
            .select([
                pl.col("cond")
                .cast(pl.Int32)
                .rolling_sum(window)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_valid_count", canonical="ts_valid_count", backend="polars")
class TSValidCountPolarsNative(SeriesOperator):
    """Count of non-null values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_valid_count",
        category="time_series",
        description="Count of non-null values in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name)
                .is_not_null()
                .cast(pl.Int32)
                .rolling_sum(window)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_coverage_ratio", canonical="ts_coverage_ratio", backend="polars")
class TSCoverageRatioPolarsNative(SeriesOperator):
    """Ratio of non-null values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_coverage_ratio",
        category="time_series",
        description="Ratio of non-null values in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                (pl.col(feature.name)
                 .is_not_null()
                 .cast(pl.Float64)
                 .rolling_mean(window))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_positive_ratio", canonical="ts_positive_ratio", backend="polars")
class TSPositiveRatioPolarsNative(SeriesOperator):
    """Ratio of positive values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_positive_ratio",
        category="time_series",
        description="Ratio of positive values in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                (pl.col(feature.name) > 0)
                .cast(pl.Float64)
                .rolling_mean(window)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_negative_ratio", canonical="ts_negative_ratio", backend="polars")
class TSNegativeRatioPolarsNative(SeriesOperator):
    """Ratio of negative values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_negative_ratio",
        category="time_series",
        description="Ratio of negative values in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                (pl.col(feature.name) < 0)
                .cast(pl.Float64)
                .rolling_mean(window)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_zero_ratio", canonical="ts_zero_ratio", backend="polars")
class TSZeroRatioPolarsNative(SeriesOperator):
    """Ratio of zero values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_zero_ratio",
        category="time_series",
        description="Ratio of zero values in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                (pl.col(feature.name) == 0)
                .cast(pl.Float64)
                .rolling_mean(window)
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Extrema & Positional
# ============================================================================

@register_operator(name="ts_argmax", canonical="ts_argmax", backend="polars")
class TSArgmaxPolarsNative(SeriesOperator):
    """Index of maximum value in rolling window (0=oldest, window-1=current)"""
    metadata = OperatorMetadata(
        name="ts_argmax",
        category="time_series",
        description="Index of maximum value in rolling window (0=oldest, window-1=current)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: s.arg_max() if len(s) > 0 else None, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_argmin", canonical="ts_argmin", backend="polars")
class TSArgminPolarsNative(SeriesOperator):
    """Index of minimum value in rolling window"""
    metadata = OperatorMetadata(
        name="ts_argmin",
        category="time_series",
        description="Index of minimum value in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: s.arg_min() if len(s) > 0 else None, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_argmax_age", canonical="ts_argmax_age", backend="polars")
class TSArgmaxAgePolarsNative(SeriesOperator):
    """Periods since maximum value in rolling window"""
    metadata = OperatorMetadata(
        name="ts_argmax_age",
        category="time_series",
        description="Periods since maximum value in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: (len(s) - 1 - s.arg_max()) if len(s) > 0 and s.arg_max() is not None else None,
                           window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_argmin_age", canonical="ts_argmin_age", backend="polars")
class TSArgminAgePolarsNative(SeriesOperator):
    """Periods since minimum value in rolling window"""
    metadata = OperatorMetadata(
        name="ts_argmin_age",
        category="time_series",
        description="Periods since minimum value in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: (len(s) - 1 - s.arg_min()) if len(s) > 0 and s.arg_min() is not None else None,
                           window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# TopK & BottomK
# ============================================================================

@register_operator(name="ts_topk_mean", canonical="ts_topk_mean", backend="polars")
class TSTopkMeanPolarsNative(SeriesOperator):
    """Mean of top k values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_topk_mean",
        category="time_series",
        description="Mean of top k values in rolling window",
        param_names=['feature', 'window', 'k'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "k": ParamSpec(dtype=int, min=1, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, k, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: s.top_k(min(k, len(s))).mean() if len(s) > 0 else None,
                           window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_topk_sum", canonical="ts_topk_sum", backend="polars")
class TSTopkSumPolarsNative(SeriesOperator):
    """Sum of top k values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_topk_sum",
        category="time_series",
        description="Sum of top k values in rolling window",
        param_names=['feature', 'window', 'k'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "k": ParamSpec(dtype=int, min=1, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, k, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: s.top_k(min(k, len(s))).sum() if len(s) > 0 else None,
                           window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_topk_std", canonical="ts_topk_std", backend="polars")
class TSTopkStdPolarsNative(SeriesOperator):
    """Std of top k values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_topk_std",
        category="time_series",
        description="Std of top k values in rolling window",
        param_names=['feature', 'window', 'k'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "k": ParamSpec(dtype=int, min=1, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, k, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: s.top_k(min(k, len(s))).std() if len(s) > 0 else None,
                           window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_bottomk_mean", canonical="ts_bottomk_mean", backend="polars")
class TSBottomkMeanPolarsNative(SeriesOperator):
    """Mean of bottom k values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_bottomk_mean",
        category="time_series",
        description="Mean of bottom k values in rolling window",
        param_names=['feature', 'window', 'k'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "k": ParamSpec(dtype=int, min=1, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, k, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: s.bottom_k(min(k, len(s))).mean() if len(s) > 0 else None,
                           window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_bottomk_sum", canonical="ts_bottomk_sum", backend="polars")
class TSBottomkSumPolarsNative(SeriesOperator):
    """Sum of bottom k values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_bottomk_sum",
        category="time_series",
        description="Sum of bottom k values in rolling window",
        param_names=['feature', 'window', 'k'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "k": ParamSpec(dtype=int, min=1, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, k, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: s.bottom_k(min(k, len(s))).sum() if len(s) > 0 else None,
                           window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_bottomk_std", canonical="ts_bottomk_std", backend="polars")
class TSBottomkStdPolarsNative(SeriesOperator):
    """Std of bottom k values in rolling window"""
    metadata = OperatorMetadata(
        name="ts_bottomk_std",
        category="time_series",
        description="Std of bottom k values in rolling window",
        param_names=['feature', 'window', 'k'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "k": ParamSpec(dtype=int, min=1, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, k, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: s.bottom_k(min(k, len(s))).std() if len(s) > 0 else None,
                           window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Moments & Distribution Stats
# ============================================================================

@register_operator(name="ts_kurt", canonical="ts_kurt", backend="polars")
class TSKurtPolarsNative(SeriesOperator):
    """Rolling kurtosis"""
    metadata = OperatorMetadata(
        name="ts_kurt",
        category="time_series",
        description="Rolling kurtosis",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=4, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(lambda s: s.kurtosis() if len(s) >= 4 else None, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_moment", canonical="ts_moment", backend="polars")
class TSMomentPolarsNative(SeriesOperator):
    """Rolling nth central moment"""
    metadata = OperatorMetadata(
        name="ts_moment",
        category="time_series",
        description="Rolling nth central moment",
        param_names=['feature', 'window', 'n'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "n": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, n=2, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(
                    lambda s: ((s - s.mean()) ** n).mean() if len(s) > 0 else None,
                    window_size=window
                )
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_quantile_range", canonical="ts_quantile_range", backend="polars")
class TSQuantileRangePolarsNative(SeriesOperator):
    """Rolling IQR or quantile range"""
    metadata = OperatorMetadata(
        name="ts_quantile_range",
        category="time_series",
        description="Rolling IQR or quantile range",
        param_names=['feature', 'window', 'lower', 'upper'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "lower": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.25, param_role=ParamRole.SCALAR),
        "upper": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.75, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, lower=0.25, upper=0.75, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                (pl.col(feature.name).rolling_quantile(upper, window_size=window) -
                 pl.col(feature.name).rolling_quantile(lower, window_size=window))
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_trimmed_mean", canonical="ts_trimmed_mean", backend="polars")
class TSTrimmedMeanPolarsNative(SeriesOperator):
    """Rolling trimmed mean (exclude top/bottom percentiles)"""
    metadata = OperatorMetadata(
        name="ts_trimmed_mean",
        category="time_series",
        description="Rolling trimmed mean (exclude top/bottom percentiles)",
        param_names=['feature', 'window', 'trim'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
        "trim": ParamSpec(dtype=float, min=0.0, max=0.5, default=0.1, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, trim=0.1, **kwargs):
        def trimmed_mean(s):
            if len(s) < 3:
                return None
            sorted_s = s.sort()
            n_trim = int(len(s) * trim)
            if n_trim >= len(s) // 2:
                return s.median()
            return sorted_s[n_trim:-n_trim].mean() if n_trim > 0 else s.mean()

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(trimmed_mean, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Ratio & Scaling
# ============================================================================

@register_operator(name="ts_ratio", canonical="ts_ratio", backend="polars")
class TSRatioPolarsNative(SeriesOperator):
    """Current value / rolling statistic (mean, median, etc.)"""
    metadata = OperatorMetadata(
        name="ts_ratio",
        category="time_series",
        description="Current value / rolling statistic (mean, median, etc.)",
        param_names=['feature', 'window', 'stat'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "stat": ParamSpec(dtype=str, default='mean', param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, stat="mean", **kwargs):
        df = feature.to_frame().lazy()

        if stat == "mean":
            denominator = pl.col(feature.name).rolling_mean(window)
        elif stat == "median":
            denominator = pl.col(feature.name).rolling_median(window)
        elif stat == "max":
            denominator = pl.col(feature.name).rolling_max(window)
        elif stat == "min":
            denominator = pl.col(feature.name).rolling_min(window)
        else:
            denominator = pl.col(feature.name).rolling_mean(window)

        return (
            df.select([
                (pl.col(feature.name) / denominator).alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_location_shift", canonical="ts_location_shift", backend="polars")
class TSLocationShiftPolarsNative(SeriesOperator):
    """(current - rolling_mean) / rolling_std"""
    metadata = OperatorMetadata(
        name="ts_location_shift",
        category="time_series",
        description="(current - rolling_mean) / rolling_std",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                ((pl.col(feature.name) - pl.col(feature.name).rolling_mean(window)) /
                 pl.col(feature.name).rolling_std(window))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_scale_shift", canonical="ts_scale_shift", backend="polars")
class TSScaleShiftPolarsNative(SeriesOperator):
    """Rolling std / long-term rolling std"""
    metadata = OperatorMetadata(
        name="ts_scale_shift",
        category="time_series",
        description="Rolling std / long-term rolling std",
        param_names=['feature', 'short_window', 'long_window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "short_window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        "long_window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, short_window, long_window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                (pl.col(feature.name).rolling_std(short_window) /
                 pl.col(feature.name).rolling_std(long_window))
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Decay & Weighted Operations
# ============================================================================

@register_operator(name="ts_decay_linear", canonical="ts_decay_linear", backend="polars")
class TSDecayLinearPolarsNative(SeriesOperator):
    """Linear decay weighted sum: most recent gets weight window, oldest gets 1"""
    metadata = OperatorMetadata(
        name="ts_decay_linear",
        category="time_series",
        description="Linear decay weighted sum: most recent gets weight window, oldest gets 1",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def linear_decay(s):
            if len(s) == 0:
                return None
            n = len(s)
            weights = np.arange(1, n + 1, dtype=np.float64)
            weights = (weights) / (weights.sum()) if (weights.sum()) != 0 else np.nan
            return (s.to_numpy() * weights).sum()

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(linear_decay, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_sum_decay", canonical="ts_sum_decay", backend="polars")
class TSSumDecayPolarsNative(SeriesOperator):
    """Exponentially decayed sum"""
    metadata = OperatorMetadata(
        name="ts_sum_decay",
        category="time_series",
        description="Exponentially decayed sum",
        param_names=['feature', 'window', 'decay'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "decay": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.9, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, decay=0.9, **kwargs):
        def exp_decay_sum(s):
            if len(s) == 0:
                return None
            n = len(s)
            weights = np.power(decay, np.arange(n - 1, -1, -1, dtype=np.float64))
            return (s.to_numpy() * weights).sum()

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(exp_decay_sum, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_decay_exp_window", canonical="ts_decay_exp_window", backend="polars")
class TSDecayExpWindowPolarsNative(SeriesOperator):
    """Exponentially weighted mean with specified window"""
    metadata = OperatorMetadata(
        name="ts_decay_exp_window",
        category="time_series",
        description="Exponentially weighted mean with specified window",
        param_names=['feature', 'window', 'alpha'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "alpha": ParamSpec(dtype=float, min=0.0, max=1.0, default=None, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, alpha=None, **kwargs):
        if alpha is None:
            alpha = (2.0) / ((window + 1)) if ((window + 1)) != 0 else np.nan

        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name)
                .ewm_mean(alpha=alpha, adjust=False)
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Time-based Features
# ============================================================================

@register_operator(name="ts_days_since", canonical="ts_days_since", backend="polars")
class TSDaysSincePolarsNative(SeriesOperator):
    """Days since condition was last True"""
    metadata = OperatorMetadata(
        name="ts_days_since",
        category="time_series",
        description="Days since condition was last True",
        param_names=['condition'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )

    def _calculate_series(self, condition, **kwargs):
        def days_since(s):
            result = []
            days = 0
            for val in s:
                if val:
                    days = 0
                else:
                    days += 1
                result.append(days)
            return pl.Series(result)

        return (
            pl.Series("cond", condition)
            .to_frame()
            .lazy()
            .with_columns([
                pl.col("cond")
                .map_batches(days_since)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_days_since_high", canonical="ts_days_since_high", backend="polars")
class TSDaysSinceHighPolarsNative(SeriesOperator):
    """Days since rolling window high"""
    metadata = OperatorMetadata(
        name="ts_days_since_high",
        category="time_series",
        description="Days since rolling window high",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_max(window).alias("_max")
            ])
            .with_columns([
                (pl.col(feature.name) == pl.col("_max")).alias("_is_high")
            ])
            .select([
                pl.col("_is_high")
            ])
            .collect()["_is_high"]
        ).to_frame().select([
            pl.col("_is_high").map_batches(lambda s: self._days_since_true(s)).alias("result")
        ]).to_series()

    @staticmethod
    def _days_since_true(s):
        result = []
        days = 0
        for val in s:
            if val:
                days = 0
            else:
                days += 1
            result.append(days)
        return pl.Series(result)


@register_operator(name="ts_days_since_low", canonical="ts_days_since_low", backend="polars")
class TSDaysSinceLowPolarsNative(SeriesOperator):
    """Days since rolling window low"""
    metadata = OperatorMetadata(
        name="ts_days_since_low",
        category="time_series",
        description="Days since rolling window low",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_min(window).alias("_min")
            ])
            .with_columns([
                (pl.col(feature.name) == pl.col("_min")).alias("_is_low")
            ])
            .select([
                pl.col("_is_low")
            ])
            .collect()["_is_low"]
        ).to_frame().select([
            pl.col("_is_low").map_batches(lambda s: self._days_since_true(s)).alias("result")
        ]).to_series()

    @staticmethod
    def _days_since_true(s):
        result = []
        days = 0
        for val in s:
            if val:
                days = 0
            else:
                days += 1
            result.append(days)
        return pl.Series(result)


@register_operator(name="ts_time_since_change", canonical="ts_time_since_change", backend="polars")
class TSTimeSinceChangePolarsNative(SeriesOperator):
    """Periods since value changed"""
    metadata = OperatorMetadata(
        name="ts_time_since_change",
        category="time_series",
        description="Periods since value changed",
        param_names=['feature'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                (pl.col(feature.name) != pl.col(feature.name).shift(1)).alias("_changed")
            ])
            .select([
                pl.col("_changed")
            ])
            .collect()["_changed"]
        ).to_frame().select([
            pl.col("_changed").map_batches(lambda s: self._periods_since_true(s)).alias("result")
        ]).to_series()

    @staticmethod
    def _periods_since_true(s):
        result = []
        periods = 0
        for val in s:
            if val:
                periods = 0
            else:
                periods += 1
            result.append(periods)
        return pl.Series(result)


# ============================================================================
# Streak & Pattern Detection
# ============================================================================

@register_operator(name="ts_true_streak", canonical="ts_true_streak", backend="polars")
class TSTrueStreakPolarsNative(SeriesOperator):
    """Current streak length of True values"""
    metadata = OperatorMetadata(
        name="ts_true_streak",
        category="time_series",
        description="Current streak length of True values",
        param_names=['condition'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )

    def _calculate_series(self, condition, **kwargs):
        def compute_streak(s):
            result = []
            streak = 0
            for val in s:
                if val:
                    streak += 1
                else:
                    streak = 0
                result.append(streak)
            return pl.Series(result)

        return (
            pl.Series("cond", condition)
            .to_frame()
            .lazy()
            .select([
                pl.col("cond").map_batches(compute_streak).alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_transition_count", canonical="ts_transition_count", backend="polars")
class TSTransitionCountPolarsNative(SeriesOperator):
    """Count of state transitions (value changes) in rolling window"""
    metadata = OperatorMetadata(
        name="ts_transition_count",
        category="time_series",
        description="Count of state transitions (value changes) in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                (pl.col(feature.name) != pl.col(feature.name).shift(1))
                .cast(pl.Int32)
                .alias("_transition")
            ])
            .select([
                pl.col("_transition")
                .rolling_sum(window)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_new_high", canonical="ts_new_high", backend="polars")
class TSNewHighPolarsNative(SeriesOperator):
    """Boolean: current value equals rolling max"""
    metadata = OperatorMetadata(
        name="ts_new_high",
        category="time_series",
        description="Boolean: current value equals rolling max",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                (pl.col(feature.name) == pl.col(feature.name).rolling_max(window))
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_new_low", canonical="ts_new_low", backend="polars")
class TSNewLowPolarsNative(SeriesOperator):
    """Boolean: current value equals rolling min"""
    metadata = OperatorMetadata(
        name="ts_new_low",
        category="time_series",
        description="Boolean: current value equals rolling min",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                (pl.col(feature.name) == pl.col(feature.name).rolling_min(window))
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Distance & Channel Features
# ============================================================================

@register_operator(name="ts_distance_to_high", canonical="ts_distance_to_high", backend="polars")
class TSDistanceToHighPolarsNative(SeriesOperator):
    """(rolling_max - current) / rolling_max"""
    metadata = OperatorMetadata(
        name="ts_distance_to_high",
        category="time_series",
        description="(rolling_max - current) / rolling_max",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_max(window).alias("_max")
            ])
            .select([
                ((pl.col("_max") - pl.col(feature.name)) / pl.col("_max")).alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_distance_to_low", canonical="ts_distance_to_low", backend="polars")
class TSDistanceToLowPolarsNative(SeriesOperator):
    """(current - rolling_min) / current"""
    metadata = OperatorMetadata(
        name="ts_distance_to_low",
        category="time_series",
        description="(current - rolling_min) / current",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_min(window).alias("_min")
            ])
            .select([
                ((pl.col(feature.name) - pl.col("_min")) / pl.col(feature.name)).alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_channel_width", canonical="ts_channel_width", backend="polars")
class TSChannelWidthPolarsNative(SeriesOperator):
    """rolling_max - rolling_min"""
    metadata = OperatorMetadata(
        name="ts_channel_width",
        category="time_series",
        description="rolling_max - rolling_min",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                (pl.col(feature.name).rolling_max(window) -
                 pl.col(feature.name).rolling_min(window))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_channel_width_pct", canonical="ts_channel_width_pct", backend="polars")
class TSChannelWidthPctPolarsNative(SeriesOperator):
    """(rolling_max - rolling_min) / rolling_mean"""
    metadata = OperatorMetadata(
        name="ts_channel_width_pct",
        category="time_series",
        description="(rolling_max - rolling_min) / rolling_mean",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                ((pl.col(feature.name).rolling_max(window) -
                  pl.col(feature.name).rolling_min(window)) /
                 pl.col(feature.name).rolling_mean(window))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_channel_position", canonical="ts_channel_position", backend="polars")
class TSChannelPositionPolarsNative(SeriesOperator):
    """(current - rolling_min) / (rolling_max - rolling_min)"""
    metadata = OperatorMetadata(
        name="ts_channel_position",
        category="time_series",
        description="(current - rolling_min) / (rolling_max - rolling_min)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_min(window).alias("_min"),
                pl.col(feature.name).rolling_max(window).alias("_max")
            ])
            .select([
                ((pl.col(feature.name) - pl.col("_min")) /
                 (pl.col("_max") - pl.col("_min")))
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Drawdown & Performance
# ============================================================================

@register_operator(name="ts_max_drawdown", canonical="ts_max_drawdown", backend="polars")
class TSMaxDrawdownPolarsNative(SeriesOperator):
    """Maximum drawdown in rolling window"""
    metadata = OperatorMetadata(
        name="ts_max_drawdown",
        category="time_series",
        description="Maximum drawdown in rolling window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_max_dd(s):
            if len(s) == 0:
                return None
            cummax = s.cum_max()
            dd = ((s - cummax)) / cummax if cummax != 0 else np.nan
            return dd.min()

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_max_dd, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_time_under_water", canonical="ts_time_under_water", backend="polars")
class TSTimeUnderWaterPolarsNative(SeriesOperator):
    """Periods since last peak (underwater duration)"""
    metadata = OperatorMetadata(
        name="ts_time_under_water",
        category="time_series",
        description="Periods since last peak (underwater duration)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_max(window).alias("_max")
            ])
            .with_columns([
                (pl.col(feature.name) >= pl.col("_max")).alias("_at_peak")
            ])
            .select([
                pl.col("_at_peak")
            ])
            .collect()["_at_peak"]
        ).to_frame().select([
            pl.col("_at_peak").map_batches(lambda s: self._periods_underwater(s)).alias("result")
        ]).to_series()

    @staticmethod
    def _periods_underwater(s):
        result = []
        periods = 0
        for val in s:
            if val:
                periods = 0
            else:
                periods += 1
            result.append(periods)
        return pl.Series(result)


@register_operator(name="ts_recovery_fraction", canonical="ts_recovery_fraction", backend="polars")
class TSRecoveryFractionPolarsNative(SeriesOperator):
    """How much of max drawdown has been recovered"""
    metadata = OperatorMetadata(
        name="ts_recovery_fraction",
        category="time_series",
        description="How much of max drawdown has been recovered",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_recovery(s):
            if len(s) == 0:
                return None
            cummax = s.cum_max()
            dd = s - cummax
            max_dd = dd.min()
            current_dd = dd[-1] if len(dd) > 0 else 0
            if max_dd == 0:
                return 1.0
            return np.where(abs(max_dd) != 0, ((max_dd - current_dd)) / (abs(max_dd)), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_recovery, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Volatility & Risk
# ============================================================================

@register_operator(name="ts_downside_deviation", canonical="ts_downside_deviation", backend="polars")
class TSDownsideDeviationPolarsNative(SeriesOperator):
    """Std of negative returns only"""
    metadata = OperatorMetadata(
        name="ts_downside_deviation",
        category="time_series",
        description="Std of negative returns only",
        param_names=['feature', 'window', 'threshold'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.0, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, threshold=0.0, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.when(pl.col(feature.name) < threshold)
                .then(pl.col(feature.name) - threshold)
                .otherwise(0.0)
                .alias("_downside")
            ])
            .select([
                (pl.col("_downside").pow(2).rolling_mean(window).sqrt())
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_upside_deviation", canonical="ts_upside_deviation", backend="polars")
class TSUpsideDeviationPolarsNative(SeriesOperator):
    """Std of positive returns only"""
    metadata = OperatorMetadata(
        name="ts_upside_deviation",
        category="time_series",
        description="Std of positive returns only",
        param_names=['feature', 'window', 'threshold'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.0, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, threshold=0.0, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.when(pl.col(feature.name) > threshold)
                .then(pl.col(feature.name) - threshold)
                .otherwise(0.0)
                .alias("_upside")
            ])
            .select([
                (pl.col("_upside").pow(2).rolling_mean(window).sqrt())
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_expected_shortfall", canonical="ts_expected_shortfall", backend="polars")
class TSExpectedShortfallPolarsNative(SeriesOperator):
    """Mean of worst outcomes (CVaR)"""
    metadata = OperatorMetadata(
        name="ts_expected_shortfall",
        category="time_series",
        description="Mean of worst outcomes (CVaR)",
        param_names=['feature', 'window', 'alpha'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        "alpha": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.05, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, alpha=0.05, **kwargs):
        def compute_es(s):
            if len(s) == 0:
                return None
            cutoff = s.quantile(alpha)
            worst = s.filter(s <= cutoff)
            return worst.mean() if len(worst) > 0 else None

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_es, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_lower_partial_moment", canonical="ts_lower_partial_moment", backend="polars")
class TSLowerPartialMomentPolarsNative(SeriesOperator):
    """LPM(n) - mean of (threshold - x)^n for x < threshold"""
    metadata = OperatorMetadata(
        name="ts_lower_partial_moment",
        category="time_series",
        description="LPM(n) - mean of (threshold - x)^n for x < threshold",
        param_names=['feature', 'window', 'threshold', 'n'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.0, param_role=ParamRole.SCALAR),
        "n": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, threshold=0.0, n=2, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.when(pl.col(feature.name) < threshold)
                .then((threshold - pl.col(feature.name)).pow(n))
                .otherwise(0.0)
                .alias("_lpm")
            ])
            .select([
                pl.col("_lpm").rolling_mean(window).alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_upper_partial_moment", canonical="ts_upper_partial_moment", backend="polars")
class TSUpperPartialMomentPolarsNative(SeriesOperator):
    """UPM(n) - mean of (x - threshold)^n for x > threshold"""
    metadata = OperatorMetadata(
        name="ts_upper_partial_moment",
        category="time_series",
        description="UPM(n) - mean of (x - threshold)^n for x > threshold",
        param_names=['feature', 'window', 'threshold', 'n'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.0, param_role=ParamRole.SCALAR),
        "n": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, threshold=0.0, n=2, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.when(pl.col(feature.name) > threshold)
                .then((pl.col(feature.name) - threshold).pow(n))
                .otherwise(0.0)
                .alias("_upm")
            ])
            .select([
                pl.col("_upm").rolling_mean(window).alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_vol_of_vol", canonical="ts_vol_of_vol", backend="polars")
class TSVolOfVolPolarsNative(SeriesOperator):
    """Volatility of volatility - std of rolling std"""
    metadata = OperatorMetadata(
        name="ts_vol_of_vol",
        category="time_series",
        description="Volatility of volatility - std of rolling std",
        param_names=['feature', 'short_window', 'long_window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "short_window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        "long_window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, short_window, long_window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_std(short_window).alias("_vol")
            ])
            .select([
                pl.col("_vol").rolling_std(long_window).alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Trend & Regression
# ============================================================================

@register_operator(name="ts_regression_slope", canonical="ts_regression_slope", backend="polars")
class TSRegressionSlopePolarsNative(SeriesOperator):
    """Linear regression slope of feature vs time"""
    metadata = OperatorMetadata(
        name="ts_regression_slope",
        category="time_series",
        description="Linear regression slope of feature vs time",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_slope(s):
            if len(s) < 2:
                return None
            x = np.arange(len(s), dtype=np.float64)
            y = s.to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return None
            x_clean, y_clean = x[mask], y[mask]
            x_mean, y_mean = x_clean.mean(), y_clean.mean()
            numerator = ((x_clean - x_mean) * (y_clean - y_mean)).sum()
            denominator = ((x_clean - x_mean) ** 2).sum()
            return np.where(denominator if denominator != 0 else None != 0, (numerator) / (denominator if denominator != 0 else None), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_slope, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_regression_intercept", canonical="ts_regression_intercept", backend="polars")
class TSRegressionInterceptPolarsNative(SeriesOperator):
    """Linear regression intercept"""
    metadata = OperatorMetadata(
        name="ts_regression_intercept",
        category="time_series",
        description="Linear regression intercept",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_intercept(s):
            if len(s) < 2:
                return None
            x = np.arange(len(s), dtype=np.float64)
            y = s.to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return None
            x_clean, y_clean = x[mask], y[mask]
            x_mean, y_mean = x_clean.mean(), y_clean.mean()
            numerator = ((x_clean - x_mean) * (y_clean - y_mean)).sum()
            denominator = ((x_clean - x_mean) ** 2).sum()
            slope = numerator / denominator if denominator != 0 else 0
            return y_mean - slope * x_mean

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_intercept, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_regression_r2", canonical="ts_regression_r2", backend="polars")
class TSRegressionR2PolarsNative(SeriesOperator):
    """Linear regression R-squared"""
    metadata = OperatorMetadata(
        name="ts_regression_r2",
        category="time_series",
        description="Linear regression R-squared",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_r2(s):
            if len(s) < 2:
                return None
            x = np.arange(len(s), dtype=np.float64)
            y = s.to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return None
            x_clean, y_clean = x[mask], y[mask]
            x_mean, y_mean = x_clean.mean(), y_clean.mean()

            ss_tot = ((y_clean - y_mean) ** 2).sum()
            if ss_tot == 0:
                return None

            numerator = ((x_clean - x_mean) * (y_clean - y_mean)).sum()
            denominator = ((x_clean - x_mean) ** 2).sum()
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean

            y_pred = slope * x_clean + intercept
            ss_res = ((y_clean - y_pred) ** 2).sum()

            return np.where(ss_tot) != 0, (1 - (ss_res) / (ss_tot)), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_r2, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_regression_resid", canonical="ts_regression_resid", backend="polars")
class TSRegressionResidPolarsNative(SeriesOperator):
    """Current value - predicted value from rolling regression"""
    metadata = OperatorMetadata(
        name="ts_regression_resid",
        category="time_series",
        description="Current value - predicted value from rolling regression",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_resid(s):
            if len(s) < 2:
                return None
            x = np.arange(len(s), dtype=np.float64)
            y = s.to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return None
            x_clean, y_clean = x[mask], y[mask]
            x_mean, y_mean = x_clean.mean(), y_clean.mean()
            numerator = ((x_clean - x_mean) * (y_clean - y_mean)).sum()
            denominator = ((x_clean - x_mean) ** 2).sum()
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean

            # Residual for the last point
            y_pred = slope * (len(s) - 1) + intercept
            return y[-1] - y_pred

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_resid, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_time_slope", canonical="ts_time_slope", backend="polars")
class TSTimeSlopePolarsNative(SeriesOperator):
    """Alias for regression slope"""
    metadata = OperatorMetadata(
        name="ts_time_slope",
        category="time_series",
        description="Alias for regression slope",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        op = TSRegressionSlopePolarsNative()
        return op._calculate_series(feature, window, **kwargs)


@register_operator(name="ts_monotonicity", canonical="ts_monotonicity", backend="polars")
class TSMonotonicityPolarsNative(SeriesOperator):
    """Measure of monotonic trend: (up_count - down_count) / window"""
    metadata = OperatorMetadata(
        name="ts_monotonicity",
        category="time_series",
        description="Measure of monotonic trend: (up_count - down_count) / window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                (pl.col(feature.name) > pl.col(feature.name).shift(1))
                .cast(pl.Int32)
                .alias("_up"),
                (pl.col(feature.name) < pl.col(feature.name).shift(1))
                .cast(pl.Int32)
                .alias("_down")
            ])
            .select([
                ((pl.col("_up").rolling_sum(window) -
                  pl.col("_down").rolling_sum(window)).cast(pl.Float64) / window)
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# State & Persistence
# ============================================================================

@register_operator(name="ts_staleness", canonical="ts_staleness", backend="polars")
class TSStalenessPolarsNative(SeriesOperator):
    """Ratio of unique values to window size (1 - uniqueness)"""
    metadata = OperatorMetadata(
        name="ts_staleness",
        category="time_series",
        description="Ratio of unique values to window size (1 - uniqueness)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_staleness(s):
            if len(s) == 0:
                return None
            n_unique = s.n_unique()
            return np.where(len(s)) != 0, (1.0 - (n_unique) / (len(s))), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_staleness, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_sign_persistence", canonical="ts_sign_persistence", backend="polars")
class TSSignPersistencePolarsNative(SeriesOperator):
    """Average streak length of same sign"""
    metadata = OperatorMetadata(
        name="ts_sign_persistence",
        category="time_series",
        description="Average streak length of same sign",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_persistence(s):
            if len(s) < 2:
                return None
            signs = np.sign(s.to_numpy())
            changes = np.diff(signs) != 0
            n_streaks = changes.sum() + 1
            return np.where(n_streaks if n_streaks > 0 else len(s) != 0, (len(s)) / (n_streaks if n_streaks > 0 else len(s)), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_persistence, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# EWM & Adaptive
# ============================================================================

@register_operator(name="ts_ewm_corr", canonical="ts_ewm_corr", backend="polars")
class TSEwmCorrPolarsNative(SeriesOperator):
    """Exponentially weighted correlation"""
    metadata = OperatorMetadata(
        name="ts_ewm_corr",
        category="time_series",
        description="Exponentially weighted correlation",
        param_names=['x', 'y', 'span'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "span": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, span, **kwargs):
        df = pl.DataFrame({"x": x, "y": y})
        alpha = (2.0) / ((span + 1)) if ((span + 1)) != 0 else np.nan

        return (
            df.lazy()
            .with_columns([
                pl.col("x").ewm_mean(alpha=alpha).alias("x_ewm"),
                pl.col("y").ewm_mean(alpha=alpha).alias("y_ewm")
            ])
            .with_columns([
                (pl.col("x") - pl.col("x_ewm")).alias("x_dev"),
                (pl.col("y") - pl.col("y_ewm")).alias("y_dev")
            ])
            .with_columns([
                (pl.col("x_dev") * pl.col("y_dev")).ewm_mean(alpha=alpha).alias("cov"),
                (pl.col("x_dev") ** 2).ewm_mean(alpha=alpha).alias("var_x"),
                (pl.col("y_dev") ** 2).ewm_mean(alpha=alpha).alias("var_y")
            ])
            .select([
                (pl.col("cov") / (pl.col("var_x").sqrt() * pl.col("var_y").sqrt()))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_ewm_cov", canonical="ts_ewm_cov", backend="polars")
class TSEwmCovPolarsNative(SeriesOperator):
    """Exponentially weighted covariance"""
    metadata = OperatorMetadata(
        name="ts_ewm_cov",
        category="time_series",
        description="Exponentially weighted covariance",
        param_names=['x', 'y', 'span'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "span": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, span, **kwargs):
        df = pl.DataFrame({"x": x, "y": y})
        alpha = (2.0) / ((span + 1)) if ((span + 1)) != 0 else np.nan

        return (
            df.lazy()
            .with_columns([
                pl.col("x").ewm_mean(alpha=alpha).alias("x_ewm"),
                pl.col("y").ewm_mean(alpha=alpha).alias("y_ewm")
            ])
            .with_columns([
                (pl.col("x") - pl.col("x_ewm")).alias("x_dev"),
                (pl.col("y") - pl.col("y_ewm")).alias("y_dev")
            ])
            .select([
                (pl.col("x_dev") * pl.col("y_dev"))
                .ewm_mean(alpha=alpha)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_kama", canonical="ts_kama", backend="polars")
class TSKamaPolarsNative(SeriesOperator):
    """Kaufman Adaptive Moving Average"""
    metadata = OperatorMetadata(
        name="ts_kama",
        category="time_series",
        description="Kaufman Adaptive Moving Average",
        param_names=['feature', 'window', 'fast', 'slow'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        "fast": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.SCALAR),
        "slow": ParamSpec(dtype=int, min=1, default=30, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, fast=2, slow=30, **kwargs):
        def compute_kama(s):
            if len(s) < window:
                return None

            # Efficiency ratio
            change = abs(s[-1] - s[0])
            volatility = np.abs(np.diff(s.to_numpy())).sum()
            er = change / volatility if volatility != 0 else 0

            # Smoothing constant
            fast_sc = (2.0) / ((fast + 1)) if ((fast + 1)) != 0 else np.nan
            slow_sc = (2.0) / ((slow + 1)) if ((slow + 1)) != 0 else np.nan
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            # KAMA calculation (simplified - just return last value weighted)
            if len(s) == window:
                return s.mean() * (1 - sc) + s[-1] * sc
            return s[-1]

        result = [None] * len(feature)
        kama = None

        for i in range(len(feature)):
            if i < window - 1:
                result[i] = None
            else:
                window_data = feature[max(0, i - window + 1):i + 1]
                if kama is None:
                    kama = window_data.mean()
                else:
                    # Compute efficiency ratio
                    change = abs(feature[i] - feature[i - window + 1])
                    volatility = sum(abs(feature[j] - feature[j-1])
                                   for j in range(i - window + 2, i + 1))
                    er = change / volatility if volatility != 0 else 0

                    # Smoothing constant
                    fast_sc = (2.0) / ((fast + 1)) if ((fast + 1)) != 0 else np.nan
                    slow_sc = (2.0) / ((slow + 1)) if ((slow + 1)) != 0 else np.nan
                    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

                    kama = kama + sc * (feature[i] - kama)

                result[i] = kama

        return pl.Series(result)


# ============================================================================
# Autocorrelation & Memory
# ============================================================================

@register_operator(name="ts_autocorrelation_time", canonical="ts_autocorrelation_time", backend="polars")
class TSAutocorrelationTimePolarsNative(SeriesOperator):
    """Integrated autocorrelation time"""
    metadata = OperatorMetadata(
        name="ts_autocorrelation_time",
        category="time_series",
        description="Integrated autocorrelation time",
        param_names=['feature', 'window', 'max_lag'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        "max_lag": ParamSpec(dtype=int, min=1, default=None, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, max_lag=None, **kwargs):
        if max_lag is None:
            max_lag = window // 4

        def compute_act(s):
            if len(s) < 2 * max_lag:
                return None
            arr = s.to_numpy()
            mean = np.mean(arr)
            var = np.var(arr)
            if var == 0:
                return None

            # Compute autocorrelations
            act_sum = 0
            for lag in range(1, max_lag):
                if lag >= len(arr):
                    break
                acf = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
                if np.isnan(acf) or acf <= 0:
                    break
                act_sum += acf

            return 1 + 2 * act_sum

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_act, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_mean_reversion_half_life", canonical="ts_mean_reversion_half_life", backend="polars")
class TSMeanReversionHalfLifePolarsNative(SeriesOperator):
    """Half-life of mean reversion (AR(1) model)"""
    metadata = OperatorMetadata(
        name="ts_mean_reversion_half_life",
        category="time_series",
        description="Half-life of mean reversion (AR(1) model)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_half_life(s):
            if len(s) < 3:
                return None
            arr = s.to_numpy()
            y = np.diff(arr)
            x = arr[:-1]

            # Remove NaN
            mask = ~(np.isnan(y) | np.isnan(x))
            if mask.sum() < 2:
                return None

            y_clean = y[mask]
            x_clean = x[mask]

            # Simple regression: dy = alpha * x
            cov = np.cov(x_clean, y_clean)[0, 1]
            var = np.var(x_clean)
            if var == 0:
                return None

            alpha = (cov) / var if var != 0 else np.nan
            if alpha >= 0:
                return None  # Not mean reverting

            half_life = np.where(np.log(1 + alpha) != 0, (-np.log(2)) / (np.log(1 + alpha)), np.nan)
            return half_life if half_life > 0 else None

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_half_life, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Swing & Pivot Detection
# ============================================================================

@register_operator(name="ts_prev_high", canonical="ts_prev_high", backend="polars")
class TSPrevHighPolarsNative(SeriesOperator):
    """Previous rolling window high"""
    metadata = OperatorMetadata(
        name="ts_prev_high",
        category="time_series",
        description="Previous rolling window high",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name)
                .rolling_max(window)
                .shift(1)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_prev_low", canonical="ts_prev_low", backend="polars")
class TSPrevLowPolarsNative(SeriesOperator):
    """Previous rolling window low"""
    metadata = OperatorMetadata(
        name="ts_prev_low",
        category="time_series",
        description="Previous rolling window low",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name)
                .rolling_min(window)
                .shift(1)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_swing_amplitude", canonical="ts_swing_amplitude", backend="polars")
class TSSwingAmplitudePolarsNative(SeriesOperator):
    """Current swing range (from last pivot)"""
    metadata = OperatorMetadata(
        name="ts_swing_amplitude",
        category="time_series",
        description="Current swing range (from last pivot)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_max(window).alias("_high"),
                pl.col(feature.name).rolling_min(window).alias("_low")
            ])
            .select([
                (pl.col("_high") - pl.col("_low")).alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_swing_amplitude_pct", canonical="ts_swing_amplitude_pct", backend="polars")
class TSSwingAmplitudePctPolarsNative(SeriesOperator):
    """Swing amplitude as percentage of price"""
    metadata = OperatorMetadata(
        name="ts_swing_amplitude_pct",
        category="time_series",
        description="Swing amplitude as percentage of price",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_max(window).alias("_high"),
                pl.col(feature.name).rolling_min(window).alias("_low")
            ])
            .select([
                ((pl.col("_high") - pl.col("_low")) / pl.col(feature.name))
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Path & Efficiency
# ============================================================================

@register_operator(name="ts_path_efficiency", canonical="ts_path_efficiency", backend="polars")
class TSPathEfficiencyPolarsNative(SeriesOperator):
    """Straight-line distance / path distance"""
    metadata = OperatorMetadata(
        name="ts_path_efficiency",
        category="time_series",
        description="Straight-line distance / path distance",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_efficiency(s):
            if len(s) < 2:
                return None
            arr = s.to_numpy()
            straight = abs(arr[-1] - arr[0])
            path = np.abs(np.diff(arr)).sum()
            return np.where(path if path != 0 else None != 0, (straight) / (path if path != 0 else None), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_efficiency, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_roughness", canonical="ts_roughness", backend="polars")
class TSRoughnessPolarsNative(SeriesOperator):
    """Path roughness: sum of |diff| / straight distance"""
    metadata = OperatorMetadata(
        name="ts_roughness",
        category="time_series",
        description="Path roughness: sum of |diff| / straight distance",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_roughness(s):
            if len(s) < 2:
                return None
            arr = s.to_numpy()
            straight = abs(arr[-1] - arr[0])
            path = np.abs(np.diff(arr)).sum()
            return np.where(straight if straight != 0 else None != 0, (path) / (straight if straight != 0 else None), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_roughness, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_turning_rate", canonical="ts_turning_rate", backend="polars")
class TSTurningRatePolarsNative(SeriesOperator):
    """Rate of direction changes (sign changes in first diff)"""
    metadata = OperatorMetadata(
        name="ts_turning_rate",
        category="time_series",
        description="Rate of direction changes (sign changes in first diff)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_turning_rate(s):
            if len(s) < 3:
                return None
            diffs = np.diff(s.to_numpy())
            signs = np.sign(diffs)
            turns = np.sum(np.diff(signs) != 0)
            return np.where((len(s) - 1) != 0, (turns) / ((len(s) - 1)), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_turning_rate, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Rank & Score
# ============================================================================

@register_operator(name="ts_score_rank_weighted_mean", canonical="ts_score_rank_weighted_mean", backend="polars")
class TSScoreRankWeightedMeanPolarsNative(SeriesOperator):
    """Weighted mean where weights are ranks of a score series"""
    metadata = OperatorMetadata(
        name="ts_score_rank_weighted_mean",
        category="time_series",
        description="Weighted mean where weights are ranks of a score series",
        param_names=['feature', 'score', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, score, window, **kwargs):
        def compute_weighted(feat_s, score_s):
            if len(feat_s) != len(score_s) or len(feat_s) == 0:
                return None
            # Rank the scores
            ranks = score_s.rank()
            weights = ranks.to_numpy()
            weights = (weights) / (weights.sum()) if (weights.sum()) != 0 else np.nan
            return (feat_s.to_numpy() * weights).sum()

        df = pl.DataFrame({"feat": feature, "score": score})
        return (
            df.lazy()
            .select([
                pl.struct(["feat", "score"])
                .rolling_map(
                    lambda s: compute_weighted(
                        s.struct.field("feat"),
                        s.struct.field("score")
                    ),
                    window_size=window
                )
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="expanding_rank", canonical="expanding_rank", backend="polars")
class ExpandingRankPolarsNative(SeriesOperator):
    """Rank of current value within all historical values"""
    metadata = OperatorMetadata(
        name="expanding_rank",
        category="time_series",
        description="Rank of current value within all historical values",
        param_names=['feature'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rank()
                .over(pl.int_range(0, pl.count()).cum_count())
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Robust Statistics
# ============================================================================

@register_operator(name="ts_robust_zscore_prior", canonical="ts_robust_zscore_prior", backend="polars")
class TSRobustZscorePriorPolarsNative(SeriesOperator):
    """Z-score using prior window (not including current value)"""
    metadata = OperatorMetadata(
        name="ts_robust_zscore_prior",
        category="time_series",
        description="Z-score using prior window (not including current value)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).shift(1).rolling_mean(window).alias("_mean"),
                pl.col(feature.name).shift(1).rolling_std(window).alias("_std")
            ])
            .select([
                ((pl.col(feature.name) - pl.col("_mean")) / pl.col("_std"))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_robust_zscore_inclusive", canonical="ts_robust_zscore_inclusive", backend="polars")
class TSRobustZscoreInclusivePolarsNative(SeriesOperator):
    """Z-score using median and MAD instead of mean and std"""
    metadata = OperatorMetadata(
        name="ts_robust_zscore_inclusive",
        category="time_series",
        description="Z-score using median and MAD instead of mean and std",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_robust_z(s):
            if len(s) < 2:
                return None
            median = s.median()
            mad = (s - median).abs().median()
            if mad == 0:
                return 0.0
            # MAD to std conversion factor
            return np.where((1.4826 * mad) != 0, ((s[-1] - median)) / ((1.4826 * mad)), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_robust_z, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_qn_scale", canonical="ts_qn_scale", backend="polars")
class TSQnScalePolarsNative(SeriesOperator):
    """Qn robust scale estimator"""
    metadata = OperatorMetadata(
        name="ts_qn_scale",
        category="time_series",
        description="Qn robust scale estimator",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_qn(s):
            if len(s) < 2:
                return None
            arr = s.to_numpy()
            # Simplified Qn: first quartile of pairwise distances
            n = len(arr)
            if n > 100:  # Sample for large windows
                idx = np.random.choice(n, 100, replace=False)
                arr = arr[idx]
            diffs = []
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    diffs.append(abs(arr[i] - arr[j]))
            if len(diffs) == 0:
                return None
            return np.quantile(diffs, 0.25) * 2.2219  # Consistency factor

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_qn, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Regression Variants
# ============================================================================

@register_operator(name="ts_regression_resid_mean", canonical="ts_regression_resid_mean", backend="polars")
class TSRegressionResidMeanPolarsNative(SeriesOperator):
    """Mean of regression residuals in window"""
    metadata = OperatorMetadata(
        name="ts_regression_resid_mean",
        category="time_series",
        description="Mean of regression residuals in window",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_resid_mean(s):
            if len(s) < 2:
                return None
            x = np.arange(len(s), dtype=np.float64)
            y = s.to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return None
            x_clean, y_clean = x[mask], y[mask]
            x_mean, y_mean = x_clean.mean(), y_clean.mean()
            numerator = ((x_clean - x_mean) * (y_clean - y_mean)).sum()
            denominator = ((x_clean - x_mean) ** 2).sum()
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean

            y_pred = slope * x_clean + intercept
            resids = y_clean - y_pred
            return resids.mean()

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_resid_mean, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_regression_tstat", canonical="ts_regression_tstat", backend="polars")
class TSRegressionTstatPolarsNative(SeriesOperator):
    """T-statistic of regression slope"""
    metadata = OperatorMetadata(
        name="ts_regression_tstat",
        category="time_series",
        description="T-statistic of regression slope",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_tstat(s):
            if len(s) < 3:
                return None
            x = np.arange(len(s), dtype=np.float64)
            y = s.to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                return None
            x_clean, y_clean = x[mask], y[mask]
            n = len(x_clean)
            x_mean, y_mean = x_clean.mean(), y_clean.mean()

            numerator = ((x_clean - x_mean) * (y_clean - y_mean)).sum()
            denominator = ((x_clean - x_mean) ** 2).sum()
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean

            y_pred = slope * x_clean + intercept
            resids = y_clean - y_pred
            mse = np.where((n - 2) if n > 2 else None != 0, (resids ** 2).sum() / (n - 2) if n > 2 else None, np.nan)
            if mse is None or denominator == 0:
                return None

            se_slope = np.where(denominator) != 0, (np.sqrt(mse) / (denominator)), np.nan)
            return np.where(se_slope if se_slope != 0 else None != 0, (slope) / (se_slope if se_slope != 0 else None), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_tstat, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_regression_forecast_error", canonical="ts_regression_forecast_error", backend="polars")
class TSRegressionForecastErrorPolarsNative(SeriesOperator):
    """Forecast error: actual next value - predicted next value"""
    metadata = OperatorMetadata(
        name="ts_regression_forecast_error",
        category="time_series",
        description="Forecast error: actual next value - predicted next value",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_forecast_error(s):
            if len(s) < 2:
                return None
            # Use first n-1 points to predict nth point
            x = np.arange(len(s) - 1, dtype=np.float64)
            y = s[:-1].to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return None
            x_clean, y_clean = x[mask], y[mask]
            x_mean, y_mean = x_clean.mean(), y_clean.mean()
            numerator = ((x_clean - x_mean) * (y_clean - y_mean)).sum()
            denominator = ((x_clean - x_mean) ** 2).sum()
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean

            # Predict for position len(s)-1
            y_pred = slope * (len(s) - 1) + intercept
            return s[-1] - y_pred

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_forecast_error, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_regression_forecast_error_z", canonical="ts_regression_forecast_error_z", backend="polars")
class TSRegressionForecastErrorZPolarsNative(SeriesOperator):
    """Standardized forecast error"""
    metadata = OperatorMetadata(
        name="ts_regression_forecast_error_z",
        category="time_series",
        description="Standardized forecast error",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_forecast_error_z(s):
            if len(s) < 3:
                return None
            x = np.arange(len(s) - 1, dtype=np.float64)
            y = s[:-1].to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return None
            x_clean, y_clean = x[mask], y[mask]
            x_mean, y_mean = x_clean.mean(), y_clean.mean()
            numerator = ((x_clean - x_mean) * (y_clean - y_mean)).sum()
            denominator = ((x_clean - x_mean) ** 2).sum()
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean

            y_pred_hist = slope * x_clean + intercept
            resids = y_clean - y_pred_hist
            resid_std = resids.std() if len(resids) > 1 else None

            y_pred = slope * (len(s) - 1) + intercept
            error = s[-1] - y_pred

            return np.where(resid_std if resid_std and resid_std != 0 else None != 0, (error) / (resid_std if resid_std and resid_std != 0 else None), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_forecast_error_z, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Beta Estimation
# ============================================================================

@register_operator(name="ts_beta_if", canonical="ts_beta_if", backend="polars")
class TSBetaIfPolarsNative(SeriesOperator):
    """Conditional rolling beta: cov(x,y) / var(y)"""
    metadata = OperatorMetadata(
        name="ts_beta_if",
        category="time_series",
        description="Conditional rolling beta: cov(x,y) / var(y)",
        param_names=['x', 'y', 'condition', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, condition, window, **kwargs):
        df = pl.DataFrame({"x": x, "y": y, "cond": condition})

        return (
            df.lazy()
            .with_columns([
                pl.when(pl.col("cond")).then(pl.col("x")).otherwise(None).alias("x_f"),
                pl.when(pl.col("cond")).then(pl.col("y")).otherwise(None).alias("y_f")
            ])
            .with_columns([
                (pl.col("x_f") - pl.col("x_f").rolling_mean(window)).alias("x_dm"),
                (pl.col("y_f") - pl.col("y_f").rolling_mean(window)).alias("y_dm")
            ])
            .with_columns([
                (pl.col("x_dm") * pl.col("y_dm")).rolling_mean(window).alias("cov"),
                (pl.col("y_dm") ** 2).rolling_mean(window).alias("var_y")
            ])
            .select([
                (pl.col("cov") / pl.col("var_y")).alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Tail & Extreme Statistics
# ============================================================================

@register_operator(name="ts_tail_ratio", canonical="ts_tail_ratio", backend="polars")
class TSTailRatioPolarsNative(SeriesOperator):
    """Ratio of upper tail to lower tail (e.g., 95th percentile / 5th percentile)"""
    metadata = OperatorMetadata(
        name="ts_tail_ratio",
        category="time_series",
        description="Ratio of upper tail to lower tail (e.g., 95th percentile / 5th percentile)",
        param_names=['feature', 'window', 'upper', 'lower'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        "upper": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.95, param_role=ParamRole.SCALAR),
        "lower": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.05, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, upper=0.95, lower=0.05, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_quantile(upper, window_size=window).alias("_upper"),
                pl.col(feature.name).rolling_quantile(lower, window_size=window).alias("_lower")
            ])
            .select([
                (pl.col("_upper") / pl.col("_lower").abs()).alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_tail_mean", canonical="ts_tail_mean", backend="polars")
class TSTailMeanPolarsNative(SeriesOperator):
    """Mean of extreme tail (beyond threshold quantile)"""
    metadata = OperatorMetadata(
        name="ts_tail_mean",
        category="time_series",
        description="Mean of extreme tail (beyond threshold quantile)",
        param_names=['feature', 'window', 'threshold'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.95, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, threshold=0.95, **kwargs):
        def compute_tail_mean(s):
            if len(s) == 0:
                return None
            cutoff = s.quantile(threshold)
            tail = s.filter(s >= cutoff)
            return tail.mean() if len(tail) > 0 else None

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_tail_mean, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_expected_shortfall_asymmetry", canonical="ts_expected_shortfall_asymmetry", backend="polars")
class TSExpectedShortfallAsymmetryPolarsNative(SeriesOperator):
    """Ratio of upper ES to lower ES"""
    metadata = OperatorMetadata(
        name="ts_expected_shortfall_asymmetry",
        category="time_series",
        description="Ratio of upper ES to lower ES",
        param_names=['feature', 'window', 'alpha'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "alpha": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.05, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, alpha=0.05, **kwargs):
        def compute_es_asymmetry(s):
            if len(s) < 10:
                return None
            lower_cutoff = s.quantile(alpha)
            upper_cutoff = s.quantile(1 - alpha)

            lower_tail = s.filter(s <= lower_cutoff)
            upper_tail = s.filter(s >= upper_cutoff)

            lower_es = lower_tail.mean() if len(lower_tail) > 0 else None
            upper_es = upper_tail.mean() if len(upper_tail) > 0 else None

            if lower_es is None or upper_es is None or lower_es == 0:
                return None
            return np.where(abs(lower_es) != 0, (upper_es) / (abs(lower_es)), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_es_asymmetry, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Partial Correlation & Multivariate
# ============================================================================

@register_operator(name="ts_partial_corr", canonical="ts_partial_corr", backend="polars")
class TSPartialCorrPolarsNative(SeriesOperator):
    """Partial correlation of x and y controlling for z"""
    metadata = OperatorMetadata(
        name="ts_partial_corr",
        category="time_series",
        description="Partial correlation of x and y controlling for z",
        param_names=['x', 'y', 'z', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, z, window, **kwargs):
        def compute_partial(x_s, y_s, z_s):
            if len(x_s) < 3:
                return None

            # Correlation matrix
            try:
                r_xy = np.corrcoef(x_s.to_numpy(), y_s.to_numpy())[0, 1]
                r_xz = np.corrcoef(x_s.to_numpy(), z_s.to_numpy())[0, 1]
                r_yz = np.corrcoef(y_s.to_numpy(), z_s.to_numpy())[0, 1]

                # Partial correlation formula
                numerator = r_xy - r_xz * r_yz
                denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

                if denominator == 0 or np.isnan(denominator):
                    return None
                return np.where(denominator != 0, (numerator) / (denominator), np.nan)
            except:
                return None

        df = pl.DataFrame({"x": x, "y": y, "z": z})
        return (
            df.lazy()
            .select([
                pl.struct(["x", "y", "z"])
                .rolling_map(
                    lambda s: compute_partial(
                        s.struct.field("x"),
                        s.struct.field("y"),
                        s.struct.field("z")
                    ),
                    window_size=window
                )
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Distance Correlation
# ============================================================================

@register_operator(name="ts_distance_corr", canonical="ts_distance_corr", backend="polars")
class TSDistanceCorrPolarsNative(SeriesOperator):
    """Distance correlation (detects nonlinear dependencies)"""
    metadata = OperatorMetadata(
        name="ts_distance_corr",
        category="time_series",
        description="Distance correlation (detects nonlinear dependencies)",
        param_names=['x', 'y', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, window, **kwargs):
        def compute_dcor(x_s, y_s):
            if len(x_s) < 2:
                return None

            x_arr = x_s.to_numpy().reshape(-1, 1)
            y_arr = y_s.to_numpy().reshape(-1, 1)

            # Simplified distance correlation
            from scipy.spatial.distance import pdist, squareform
            try:
                dx = squareform(pdist(x_arr, 'euclidean'))
                dy = squareform(pdist(y_arr, 'euclidean'))

                # Double centering
                n = len(x_arr)
                dx_centered = dx - dx.mean(axis=0) - dx.mean(axis=1)[:, np.newaxis] + dx.mean()
                dy_centered = dy - dy.mean(axis=0) - dy.mean(axis=1)[:, np.newaxis] + dy.mean()

                dcov_xy = np.where((n * n)) != 0, (np.sqrt((dx_centered * dy_centered).sum()) / ((n * n))), np.nan)
                dcov_xx = np.where((n * n)) != 0, (np.sqrt((dx_centered * dx_centered).sum()) / ((n * n))), np.nan)
                dcov_yy = np.where((n * n)) != 0, (np.sqrt((dy_centered * dy_centered).sum()) / ((n * n))), np.nan)

                if dcov_xx * dcov_yy == 0:
                    return 0.0
                return np.where(np.sqrt(dcov_xx * dcov_yy) != 0, (dcov_xy) / (np.sqrt(dcov_xx * dcov_yy)), np.nan)
            except:
                return None

        df = pl.DataFrame({"x": x, "y": y})
        return (
            df.lazy()
            .select([
                pl.struct(["x", "y"])
                .rolling_map(
                    lambda s: compute_dcor(
                        s.struct.field("x"),
                        s.struct.field("y")
                    ),
                    window_size=window
                )
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_distance_cov", canonical="ts_distance_cov", backend="polars")
class TSDistanceCovPolarsNative(SeriesOperator):
    """Distance covariance"""
    metadata = OperatorMetadata(
        name="ts_distance_cov",
        category="time_series",
        description="Distance covariance",
        param_names=['x', 'y', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, window, **kwargs):
        def compute_dcov(x_s, y_s):
            if len(x_s) < 2:
                return None

            x_arr = x_s.to_numpy().reshape(-1, 1)
            y_arr = y_s.to_numpy().reshape(-1, 1)

            from scipy.spatial.distance import pdist, squareform
            try:
                dx = squareform(pdist(x_arr, 'euclidean'))
                dy = squareform(pdist(y_arr, 'euclidean'))

                n = len(x_arr)
                dx_centered = dx - dx.mean(axis=0) - dx.mean(axis=1)[:, np.newaxis] + dx.mean()
                dy_centered = dy - dy.mean(axis=0) - dy.mean(axis=1)[:, np.newaxis] + dy.mean()

                return np.where((n * n)) != 0, (np.sqrt((dx_centered * dy_centered).sum()) / ((n * n))), np.nan)
            except:
                return None

        df = pl.DataFrame({"x": x, "y": y})
        return (
            df.lazy()
            .select([
                pl.struct(["x", "y"])
                .rolling_map(
                    lambda s: compute_dcov(
                        s.struct.field("x"),
                        s.struct.field("y")
                    ),
                    window_size=window
                )
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Fill & Imputation
# ============================================================================

@register_operator(name="ts_ffill_limited", canonical="ts_ffill_limited", backend="polars")
class TSFfillLimitedPolarsNative(SeriesOperator):
    """Forward fill with maximum fill limit"""
    metadata = OperatorMetadata(
        name="ts_ffill_limited",
        category="time_series",
        description="Forward fill with maximum fill limit",
        param_names=['feature', 'max_fill'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "max_fill": ParamSpec(dtype=int, min=1, default=5, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, max_fill=5, **kwargs):
        def limited_ffill(s):
            result = []
            last_valid = None
            fill_count = 0

            for val in s:
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    last_valid = val
                    fill_count = 0
                    result.append(val)
                else:
                    if last_valid is not None and fill_count < max_fill:
                        result.append(last_valid)
                        fill_count += 1
                    else:
                        result.append(None)

            return pl.Series(result)

        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name)
                .map_batches(limited_ffill)
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Filtering & Smoothing
# ============================================================================

@register_operator(name="ts_median3_causal", canonical="ts_median3_causal", backend="polars")
class TSMedian3CausalPolarsNative(SeriesOperator):
    """3-point median filter (causal: uses current and 2 past)"""
    metadata = OperatorMetadata(
        name="ts_median3_causal",
        category="time_series",
        description="3-point median filter (causal: uses current and 2 past)",
        param_names=['feature'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name)
                .rolling_median(3)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_rolling_median_causal", canonical="ts_rolling_median_causal", backend="polars")
class TSRollingMedianCausalPolarsNative(SeriesOperator):
    """Rolling median (alias for ts_median)"""
    metadata = OperatorMetadata(
        name="ts_rolling_median_causal",
        category="time_series",
        description="Rolling median (alias for ts_median)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                pl.col(feature.name)
                .rolling_median(window)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_robust_ema", canonical="ts_robust_ema", backend="polars")
class TSRobustEmaPolarsNative(SeriesOperator):
    """Robust EMA that clips outliers before smoothing"""
    metadata = OperatorMetadata(
        name="ts_robust_ema",
        category="time_series",
        description="Robust EMA that clips outliers before smoothing",
        param_names=['feature', 'span', 'clip_std'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "span": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "clip_std": ParamSpec(dtype=float, min=0.0, default=3.0, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, span, clip_std=3.0, **kwargs):
        alpha = (2.0) / ((span + 1)) if ((span + 1)) != 0 else np.nan

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).ewm_mean(alpha=alpha).alias("_ema"),
                pl.col(feature.name).ewm_std(alpha=alpha).alias("_std")
            ])
            .with_columns([
                pl.when(
                    (pl.col(feature.name) - pl.col("_ema")).abs() > clip_std * pl.col("_std")
                )
                .then(pl.col("_ema"))
                .otherwise(pl.col(feature.name))
                .alias("_clipped")
            ])
            .select([
                pl.col("_clipped").ewm_mean(alpha=alpha).alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Run & Streak Analysis
# ============================================================================

@register_operator(name="ts_run_strength", canonical="ts_run_strength", backend="polars")
class TSRunStrengthPolarsNative(SeriesOperator):
    """Average absolute return during runs (consecutive same-sign moves)"""
    metadata = OperatorMetadata(
        name="ts_run_strength",
        category="time_series",
        description="Average absolute return during runs (consecutive same-sign moves)",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_run_strength(s):
            if len(s) < 2:
                return None
            diffs = np.diff(s.to_numpy())
            signs = np.sign(diffs)

            # Find runs
            runs = []
            current_run = []
            current_sign = None

            for i, (sign, diff) in enumerate(zip(signs, diffs)):
                if sign == 0:
                    continue
                if sign == current_sign:
                    current_run.append(abs(diff))
                else:
                    if current_run:
                        runs.append(np.mean(current_run))
                    current_run = [abs(diff)]
                    current_sign = sign

            if current_run:
                runs.append(np.mean(current_run))

            return np.mean(runs) if runs else None

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_run_strength, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_run_efficiency", canonical="ts_run_efficiency", backend="polars")
class TSRunEfficiencyPolarsNative(SeriesOperator):
    """Ratio of run-based distance to total distance"""
    metadata = OperatorMetadata(
        name="ts_run_efficiency",
        category="time_series",
        description="Ratio of run-based distance to total distance",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        def compute_run_efficiency(s):
            if len(s) < 2:
                return None
            diffs = np.diff(s.to_numpy())
            total_dist = np.abs(diffs).sum()
            if total_dist == 0:
                return None

            # Compute run-based distance (sum of absolute run movements)
            signs = np.sign(diffs)
            run_dist = 0
            current_run_sum = 0

            for i, (sign, diff) in enumerate(zip(signs, diffs)):
                if i == 0 or sign == prev_sign:
                    current_run_sum += diff
                else:
                    run_dist += abs(current_run_sum)
                    current_run_sum = diff
                prev_sign = sign

            run_dist += abs(current_run_sum)

            return np.where(total_dist != 0, (run_dist) / (total_dist), np.nan)

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_run_efficiency, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Quantile & Expectile Regression
# ============================================================================

@register_operator(name="ts_quantile_regression_slope", canonical="ts_quantile_regression_slope", backend="polars")
class TSQuantileRegressionSlopePolarsNative(SeriesOperator):
    """Quantile regression slope (simplified via weighted least squares)"""
    metadata = OperatorMetadata(
        name="ts_quantile_regression_slope",
        category="time_series",
        description="Quantile regression slope (simplified via weighted least squares)",
        param_names=['feature', 'window', 'quantile'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, quantile=0.5, **kwargs):
        def compute_qr_slope(s):
            if len(s) < 3:
                return None

            x = np.arange(len(s), dtype=np.float64)
            y = s.to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                return None

            x_clean, y_clean = x[mask], y[mask]

            # Simple median-based slope (Theil-Sen estimator approximation)
            n = len(x_clean)
            if n < 2:
                return None

            slopes = []
            for i in range(0, n - 1, max(1, n // 10)):  # Sample for speed
                for j in range(i + 1, n):
                    if x_clean[j] != x_clean[i]:
                        slopes.append((y_clean[j] - y_clean[i]) / (x_clean[j] - x_clean[i]))

            return np.quantile(slopes, quantile) if slopes else None

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_qr_slope, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


@register_operator(name="ts_expectile", canonical="ts_expectile", backend="polars")
class TSExpectilePolarsNative(SeriesOperator):
    """Rolling expectile (asymmetric mean)"""
    metadata = OperatorMetadata(
        name="ts_expectile",
        category="time_series",
        description="Rolling expectile (asymmetric mean)",
        param_names=['feature', 'window', 'tau'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "tau": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, tau=0.5, **kwargs):
        def compute_expectile(s):
            if len(s) == 0:
                return None

            arr = s.to_numpy()
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return None

            # Iterative expectile calculation
            mu = np.median(arr)
            for _ in range(10):  # Max iterations
                residuals = arr - mu
                weights = np.where(residuals > 0, tau, 1 - tau)
                new_mu = ((weights * arr).sum()) / (weights.sum()) if (weights.sum()) != 0 else np.nan
                if abs(new_mu - mu) < 1e-6:
                    break
                mu = new_mu

            return mu

        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name)
                .rolling_map(compute_expectile, window_size=window)
                .alias("result")
            ])
            .select(["result"])
            .collect()["result"]
        )


# ============================================================================
# Support & Resistance
# ============================================================================

@register_operator(name="ts_support_level", canonical="ts_support_level", backend="polars")
class TSSupportLevelPolarsNative(SeriesOperator):
    """Support level (rolling minimum with buffer)"""
    metadata = OperatorMetadata(
        name="ts_support_level",
        category="time_series",
        description="Support level (rolling minimum with buffer)",
        param_names=['feature', 'window', 'buffer'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "buffer": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.02, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, buffer=0.02, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                (pl.col(feature.name).rolling_min(window) * (1 - buffer))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_resistance_level", canonical="ts_resistance_level", backend="polars")
class TSResistanceLevelPolarsNative(SeriesOperator):
    """Resistance level (rolling maximum with buffer)"""
    metadata = OperatorMetadata(
        name="ts_resistance_level",
        category="time_series",
        description="Resistance level (rolling maximum with buffer)",
        param_names=['feature', 'window', 'buffer'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "buffer": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.02, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, buffer=0.02, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .select([
                (pl.col(feature.name).rolling_max(window) * (1 + buffer))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_distance_to_support", canonical="ts_distance_to_support", backend="polars")
class TSDistanceToSupportPolarsNative(SeriesOperator):
    """Distance from current price to support level"""
    metadata = OperatorMetadata(
        name="ts_distance_to_support",
        category="time_series",
        description="Distance from current price to support level",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_min(window).alias("_support")
            ])
            .select([
                ((pl.col(feature.name) - pl.col("_support")) / pl.col(feature.name))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_distance_to_resistance", canonical="ts_distance_to_resistance", backend="polars")
class TSDistanceToResistancePolarsNative(SeriesOperator):
    """Distance from current price to resistance level"""
    metadata = OperatorMetadata(
        name="ts_distance_to_resistance",
        category="time_series",
        description="Distance from current price to resistance level",
        param_names=['feature', 'window'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_max(window).alias("_resistance")
            ])
            .select([
                ((pl.col("_resistance") - pl.col(feature.name)) / pl.col(feature.name))
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_breakout_high", canonical="ts_breakout_high", backend="polars")
class TSBreakoutHighPolarsNative(SeriesOperator):
    """Boolean: price breaks above resistance"""
    metadata = OperatorMetadata(
        name="ts_breakout_high",
        category="time_series",
        description="Boolean: price breaks above resistance",
        param_names=['feature', 'window', 'threshold'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=1.0, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, threshold=1.0, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).shift(1).rolling_max(window).alias("_prev_high")
            ])
            .select([
                (pl.col(feature.name) > pl.col("_prev_high") * threshold)
                .alias("result")
            ])
            .collect()["result"]
        )


@register_operator(name="ts_breakdown_low", canonical="ts_breakdown_low", backend="polars")
class TSBreakdownLowPolarsNative(SeriesOperator):
    """Boolean: price breaks below support"""
    metadata = OperatorMetadata(
        name="ts_breakdown_low",
        category="time_series",
        description="Boolean: price breaks below support",
        param_names=['feature', 'window', 'threshold'],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=1.0, param_role=ParamRole.SCALAR),
    }

    def _calculate_series(self, feature, window, threshold=1.0, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).shift(1).rolling_min(window).alias("_prev_low")
            ])
            .select([
                (pl.col(feature.name) < pl.col("_prev_low") * threshold)
                .alias("result")
            ])
            .collect()["result"]
        )


# ============================================================================
# Completed: 100+ ts_* operators implemented
# ============================================================================

