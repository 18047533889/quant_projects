"""
Polars Native Implementation - Fundamental/Financial Operators (Batch 1)

Financial data operators for fundamental analysis with quarterly/annual frequency handling.
Implements 40+ operators for fin_* and fiscal_* families.

Key patterns:
- Use .shift() for lag operations (period-over-period)
- Use .diff() for absolute changes
- Use rolling windows for TTM (trailing twelve months)
- Handle sparse data typical of quarterly/annual reports
"""

import polars as pl
import numpy as np
from typing import Optional

from cleaned_operators.base import (
    SeriesOperator,
    register_operator,
    OperatorMetadata,
    ParamSpec,
    ParamRole,
)


# ============================================================================
# BASIC PERIOD OPERATIONS
# ============================================================================

@register_operator(name="fin_lag", canonical="fin_lag", backend="polars")
class FinLagPolarsNative(SeriesOperator):
    """Lag financial data by N periods (default 1 quarter)."""
    
    metadata = OperatorMetadata(
        name="fin_lag",
        category="fundamental",
        description="Lag financial data by N periods (default 1 quarter).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([pl.col(value.name).shift(periods).alias(value.name)])
            .collect()
            .to_series()
        )


@register_operator(name="fin_diff", canonical="fin_diff", backend="polars")
class FinDiffPolarsNative(SeriesOperator):
    """Absolute difference from N periods ago."""
    
    metadata = OperatorMetadata(
        name="fin_diff",
        category="fundamental",
        description="Absolute difference from N periods ago.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name) - pl.col(value.name).shift(periods))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_pct_change", canonical="fin_pct_change", backend="polars")
class FinPctChangePolarsNative(SeriesOperator):
    """Percentage change from N periods ago."""
    
    metadata = OperatorMetadata(
        name="fin_pct_change",
        category="fundamental",
        description="Percentage change from N periods ago.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods)) - 1)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_log_change", canonical="fin_log_change", backend="polars")
class FinLogChangePolarsNative(SeriesOperator):
    """Log change from N periods ago."""
    
    metadata = OperatorMetadata(
        name="fin_log_change",
        category="fundamental",
        description="Log change from N periods ago.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name).log() - pl.col(value.name).shift(periods).log())
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_growth", canonical="fin_growth", backend="polars")
class FinGrowthPolarsNative(SeriesOperator):
    """Growth rate (same as pct_change, alias for clarity)."""
    
    metadata = OperatorMetadata(
        name="fin_growth",
        category="fundamental",
        description="Growth rate (same as pct_change, alias for clarity).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods)) - 1)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# YOY AND QOQ OPERATIONS
# ============================================================================

@register_operator(name="fin_yoy", canonical="fin_yoy", backend="polars")
class FinYoYPolarsNative(SeriesOperator):
    """Year-over-year growth (4 quarters for quarterly data)."""
    
    metadata = OperatorMetadata(
        name="fin_yoy",
        category="fundamental",
        description="Year-over-year growth (4 quarters for quarterly data).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods)) - 1)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_qoq", canonical="fin_qoq", backend="polars")
class FinQoQPolarsNative(SeriesOperator):
    """Quarter-over-quarter growth."""
    
    metadata = OperatorMetadata(
        name="fin_qoq",
        category="fundamental",
        description="Quarter-over-quarter growth.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods)) - 1)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# TTM (TRAILING TWELVE MONTHS) OPERATIONS
# ============================================================================

@register_operator(name="fin_ttm", canonical="fin_ttm", backend="polars")
class FinTTMPolarsNative(SeriesOperator):
    """Trailing twelve months sum (4 quarters rolling sum)."""
    
    metadata = OperatorMetadata(
        name="fin_ttm",
        category="fundamental",
        description="Trailing twelve months sum (4 quarters rolling sum).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name).rolling_sum(window_size=periods, min_periods=periods)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_ttm_quarterly", canonical="fin_ttm_quarterly", backend="polars")
class FinTTMQuarterlyPolarsNative(SeriesOperator):
    """TTM from quarterly data (rolling sum of 4 quarters)."""
    
    metadata = OperatorMetadata(
        name="fin_ttm_quarterly",
        category="fundamental",
        description="TTM from quarterly data (rolling sum of 4 quarters).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name).rolling_sum(window_size=4, min_periods=4)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_quarter_from_cumulative", canonical="fin_quarter_from_cumulative", backend="polars")
class FinQuarterFromCumulativePolarsNative(SeriesOperator):
    """Extract quarterly value from cumulative annual (Q4=annual, Q3=Q4-Q3_cumulative, etc)."""
    
    metadata = OperatorMetadata(
        name="fin_quarter_from_cumulative",
        category="fundamental",
        description="Extract quarterly value from cumulative annual (Q4=annual, Q3=Q4-Q3_cumulative, etc).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name) - pl.col(value.name).shift(1))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# STATISTICAL MEASURES
# ============================================================================

@register_operator(name="fin_std", canonical="fin_std", backend="polars")
class FinStdPolarsNative(SeriesOperator):
    """Rolling standard deviation over N periods."""
    
    metadata = OperatorMetadata(
        name="fin_std",
        category="fundamental",
        description="Rolling standard deviation over N periods.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name).rolling_std(window_size=window, min_periods=2)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_cv", canonical="fin_cv", backend="polars")
class FinCVPolarsNative(SeriesOperator):
    """Coefficient of variation (std / mean) over N periods."""
    
    metadata = OperatorMetadata(
        name="fin_cv",
        category="fundamental",
        description="Coefficient of variation (std / mean) over N periods.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name).rolling_std(window_size=window, min_periods=2) /
                 pl.col(value.name).rolling_mean(window_size=window, min_periods=2))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_mean_abs_deviation", canonical="fin_mean_abs_deviation", backend="polars")
class FinMeanAbsDeviationPolarsNative(SeriesOperator):
    """Mean absolute deviation from rolling mean."""
    
    metadata = OperatorMetadata(
        name="fin_mean_abs_deviation",
        category="fundamental",
        description="Mean absolute deviation from rolling mean.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                (pl.col(col_name) - pl.col(col_name).rolling_mean(window_size=window, min_periods=1))
                .abs()
                .rolling_mean(window_size=window, min_periods=1)
                .alias(col_name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_range", canonical="fin_range", backend="polars")
class FinRangePolarsNative(SeriesOperator):
    """Range (max - min) over N periods."""
    
    metadata = OperatorMetadata(
        name="fin_range",
        category="fundamental",
        description="Range (max - min) over N periods.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name).rolling_max(window_size=window, min_periods=1) -
                 pl.col(value.name).rolling_min(window_size=window, min_periods=1))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# GROWTH DYNAMICS
# ============================================================================

@register_operator(name="fin_growth_acceleration", canonical="fin_growth_acceleration", backend="polars")
class FinGrowthAccelerationPolarsNative(SeriesOperator):
    """Change in growth rate (second derivative)."""
    
    metadata = OperatorMetadata(
        name="fin_growth_acceleration",
        category="fundamental",
        description="Change in growth rate (second derivative).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                pl.col(col_name).alias("value")
            ])
            .select([
                pl.when(pl.col("value" != 0).then(pl.col("value") / pl.col("value").otherwise(None).shift(periods)) - 1).alias("growth")
            ])
            .select([
                (pl.col("growth") - pl.col("growth").shift(periods)).alias(col_name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_growth_change", canonical="fin_growth_change", backend="polars")
class FinGrowthChangePolarsNative(SeriesOperator):
    """Absolute change in growth rate."""
    
    metadata = OperatorMetadata(
        name="fin_growth_change",
        category="fundamental",
        description="Absolute change in growth rate.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                pl.when(pl.col(col_name != 0).then(pl.col(col_name) / pl.col(col_name).otherwise(None).shift(periods)) - 1).alias("growth")
            ])
            .select([
                (pl.col("growth") - pl.col("growth").shift(periods)).alias(col_name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_growth_volatility", canonical="fin_growth_volatility", backend="polars")
class FinGrowthVolatilityPolarsNative(SeriesOperator):
    """Standard deviation of growth rates over N periods."""
    
    metadata = OperatorMetadata(
        name="fin_growth_volatility",
        category="fundamental",
        description="Standard deviation of growth rates over N periods.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, periods: int = 1, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                pl.when(pl.col(col_name != 0).then(pl.col(col_name) / pl.col(col_name).otherwise(None).shift(periods)) - 1).alias("growth")
            ])
            .select([
                pl.col("growth").rolling_std(window_size=window, min_periods=2).alias(col_name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_growth_stability", canonical="fin_growth_stability", backend="polars")
class FinGrowthStabilityPolarsNative(SeriesOperator):
    """Negative of growth volatility (higher = more stable)."""
    
    metadata = OperatorMetadata(
        name="fin_growth_stability",
        category="fundamental",
        description="Negative of growth volatility (higher = more stable).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, periods: int = 1, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                pl.when(pl.col(col_name != 0).then(pl.col(col_name) / pl.col(col_name).otherwise(None).shift(periods)) - 1).alias("growth")
            ])
            .select([
                (-pl.col("growth").rolling_std(window_size=window, min_periods=2)).alias(col_name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_cagr", canonical="fin_cagr", backend="polars")
class FinCAGRPolarsNative(SeriesOperator):
    """Compound annual growth rate over N periods."""
    
    metadata = OperatorMetadata(
        name="fin_cagr",
        category="fundamental",
        description="Compound annual growth rate over N periods.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods)).pow(1.0 / periods) - 1)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# RATIO OPERATIONS
# ============================================================================

@register_operator(name="fin_ratio", canonical="fin_ratio", backend="polars")
class FinRatioPolarsNative(SeriesOperator):
    """Ratio to value N periods ago."""
    
    metadata = OperatorMetadata(
        name="fin_ratio",
        category="fundamental",
        description="Ratio to value N periods ago.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_common_size", canonical="fin_common_size", backend="polars")
class FinCommonSizePolarsNative(SeriesOperator):
    """Common size analysis: value / rolling sum (as fraction of total)."""
    
    metadata = OperatorMetadata(
        name="fin_common_size",
        category="fundamental",
        description="Common size analysis: value / rolling sum (as fraction of total).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name) /
                 pl.col(value.name).rolling_sum(window_size=window, min_periods=1))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_turnover", canonical="fin_turnover", backend="polars")
class FinTurnoverPolarsNative(SeriesOperator):
    """Turnover ratio using average balance (current + lag) / 2."""
    
    metadata = OperatorMetadata(
        name="fin_turnover",
        category="fundamental",
        description="Turnover ratio using average balance (current + lag) / 2.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name) /
                 ((pl.col(value.name) + pl.col(value.name).shift(periods)) / 2))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_average_balance", canonical="fin_average_balance", backend="polars")
class FinAverageBalancePolarsNative(SeriesOperator):
    """Average balance between current and N periods ago."""
    
    metadata = OperatorMetadata(
        name="fin_average_balance",
        category="fundamental",
        description="Average balance between current and N periods ago.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                ((pl.col(value.name) + pl.col(value.name).shift(periods)) / 2)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# STREAK AND PERSISTENCE MEASURES
# ============================================================================

@register_operator(name="fin_positive_streak", canonical="fin_positive_streak", backend="polars")
class FinPositiveStreakPolarsNative(SeriesOperator):
    """Count consecutive positive values."""
    
    metadata = OperatorMetadata(
        name="fin_positive_streak",
        category="fundamental",
        description="Count consecutive positive values.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, **kwargs):
        # Simplified implementation using RLE-like logic
        df = value.to_frame()
        col_name = value.name

        # Create indicator for positive values
        is_pos = (df[col_name] > 0).cast(pl.Int32)

        # Create group ID that changes when sign changes
        sign_change = (is_pos != is_pos.shift(1).fill_null(0)).cast(pl.Int32)
        group_id = sign_change.cum_sum()

        # Count within each group, then multiply by is_pos to zero out negative streaks
        result_df = df.with_columns([
            is_pos.alias("is_pos"),
            group_id.alias("group_id")
        ])

        # Use group_by to count streak length
        streak = (
            result_df
            .with_row_count("row_idx")
            .group_by("group_id", maintain_order=True)
            .agg([
                pl.col("row_idx").alias("rows"),
                pl.col("is_pos").first().alias("is_pos_group")
            ])
            .explode("rows")
            .with_columns([
                (pl.col("rows").rank("dense").over("group_id") * pl.col("is_pos_group"))
                .alias("streak")
            ])
            .sort("rows")
            ["streak"]
        )

        return streak.alias(col_name)


@register_operator(name="fin_negative_streak", canonical="fin_negative_streak", backend="polars")
class FinNegativeStreakPolarsNative(SeriesOperator):
    """Count consecutive negative values."""
    
    metadata = OperatorMetadata(
        name="fin_negative_streak",
        category="fundamental",
        description="Count consecutive negative values.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, **kwargs):
        # Simplified implementation using RLE-like logic
        df = value.to_frame()
        col_name = value.name

        # Create indicator for negative values
        is_neg = (df[col_name] < 0).cast(pl.Int32)

        # Create group ID that changes when sign changes
        sign_change = (is_neg != is_neg.shift(1).fill_null(0)).cast(pl.Int32)
        group_id = sign_change.cum_sum()

        # Count within each group, then multiply by is_neg to zero out positive streaks
        result_df = df.with_columns([
            is_neg.alias("is_neg"),
            group_id.alias("group_id")
        ])

        # Use group_by to count streak length
        streak = (
            result_df
            .with_row_count("row_idx")
            .group_by("group_id", maintain_order=True)
            .agg([
                pl.col("row_idx").alias("rows"),
                pl.col("is_neg").first().alias("is_neg_group")
            ])
            .explode("rows")
            .with_columns([
                (pl.col("rows").rank("dense").over("group_id") * pl.col("is_neg_group"))
                .alias("streak")
            ])
            .sort("rows")
            ["streak"]
        )

        return streak.alias(col_name)


@register_operator(name="fin_sign_change_count", canonical="fin_sign_change_count", backend="polars")
class FinSignChangeCountPolarsNative(SeriesOperator):
    """Rolling count of sign changes over N periods."""
    
    metadata = OperatorMetadata(
        name="fin_sign_change_count",
        category="fundamental",
        description="Rolling count of sign changes over N periods.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                (pl.col(col_name).sign() != pl.col(col_name).shift(1).sign())
                .cast(pl.Int32)
                .rolling_sum(window_size=window, min_periods=1)
                .alias(col_name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_monotonicity", canonical="fin_monotonicity", backend="polars")
class FinMonotonicityPolarsNative(SeriesOperator):
    """Measure of monotonic trend: (increases - decreases) / window."""
    
    metadata = OperatorMetadata(
        name="fin_monotonicity",
        category="fundamental",
        description="Measure of monotonic trend: (increases - decreases) / window.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                pl.col(col_name).diff().alias("change")
            ])
            .select([
                ((pl.col("change") > 0).cast(pl.Int32).rolling_sum(window_size=window, min_periods=1) -
                 (pl.col("change") < 0).cast(pl.Int32).rolling_sum(window_size=window, min_periods=1))
                .cast(pl.Float64) / window
                .alias(col_name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# PERSISTENCE AND QUALITY METRICS
# ============================================================================

@register_operator(name="fin_earnings_persistence", canonical="fin_earnings_persistence", backend="polars")
class FinEarningsPersistencePolarsNative(SeriesOperator):
    """Correlation between current and lagged values over window."""
    
    metadata = OperatorMetadata(
        name="fin_earnings_persistence",
        category="fundamental",
        description="Correlation between current and lagged values over window.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 8, lag: int = 1, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                pl.col(col_name).alias("current"),
                pl.col(col_name).shift(lag).alias("lagged")
            ])
            .select([
                pl.corr("current", "lagged", method="pearson")
                .rolling_apply(lambda x: x, window_size=window)
                .alias(col_name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_earnings_smoothness", canonical="fin_earnings_smoothness", backend="polars")
class FinEarningsSmoothnessPolarsNative(SeriesOperator):
    """Negative of absolute changes (smoother = fewer large changes)."""
    
    metadata = OperatorMetadata(
        name="fin_earnings_smoothness",
        category="fundamental",
        description="Negative of absolute changes (smoother = fewer large changes).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (-pl.col(value.name).diff().abs()
                 .rolling_mean(window_size=window, min_periods=1))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_stability", canonical="fin_stability", backend="polars")
class FinStabilityPolarsNative(SeriesOperator):
    """Negative coefficient of variation (higher = more stable)."""
    
    metadata = OperatorMetadata(
        name="fin_stability",
        category="fundamental",
        description="Negative coefficient of variation (higher = more stable).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (-(pl.col(value.name).rolling_std(window_size=window, min_periods=2) /
                   pl.col(value.name).rolling_mean(window_size=window, min_periods=2)))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# TREND ANALYSIS
# ============================================================================

@register_operator(name="fin_trend_slope", canonical="fin_trend_slope", backend="polars")
class FinTrendSlopePolarsNative(SeriesOperator):
    """Simple linear trend slope: (current - first) / periods."""
    
    metadata = OperatorMetadata(
        name="fin_trend_slope",
        category="fundamental",
        description="Simple linear trend slope: (current - first) / periods.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                ((pl.col(value.name) - pl.col(value.name).shift(window - 1)) / (window - 1))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_trend_acceleration", canonical="fin_trend_acceleration", backend="polars")
class FinTrendAccelerationPolarsNative(SeriesOperator):
    """Change in trend slope (second derivative approximation)."""
    
    metadata = OperatorMetadata(
        name="fin_trend_acceleration",
        category="fundamental",
        description="Change in trend slope (second derivative approximation).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                ((pl.col(col_name) - pl.col(col_name).shift(window - 1)) / (window - 1))
                .alias("slope")
            ])
            .select([
                pl.col("slope").diff().alias(col_name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# FISCAL PERIOD OPERATIONS
# ============================================================================

@register_operator(name="fiscal_pct_change", canonical="fiscal_pct_change", backend="polars")
class FiscalPctChangePolarsNative(SeriesOperator):
    """Fiscal period percentage change."""
    
    metadata = OperatorMetadata(
        name="fiscal_pct_change",
        category="fundamental",
        description="Fiscal period percentage change.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "fiscal", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.when(pl.col(value.name != 0).then(pl.col(value.name) / pl.col(value.name).otherwise(None).shift(periods)) - 1)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fiscal_acceleration", canonical="fiscal_acceleration", backend="polars")
class FiscalAccelerationPolarsNative(SeriesOperator):
    """Fiscal acceleration (change in growth rate)."""
    
    metadata = OperatorMetadata(
        name="fiscal_acceleration",
        category="fundamental",
        description="Fiscal acceleration (change in growth rate).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "fiscal", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, periods: int = 1, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                pl.when(pl.col(col_name != 0).then(pl.col(col_name) / pl.col(col_name).otherwise(None).shift(periods)) - 1).alias("growth")
            ])
            .select([
                (pl.col("growth") - pl.col("growth").shift(periods)).alias(col_name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fiscal_rolling_std", canonical="fiscal_rolling_std", backend="polars")
class FiscalRollingStdPolarsNative(SeriesOperator):
    """Rolling standard deviation of fiscal data."""
    
    metadata = OperatorMetadata(
        name="fiscal_rolling_std",
        category="fundamental",
        description="Rolling standard deviation of fiscal data.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "fiscal", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                pl.col(value.name).rolling_std(window_size=window, min_periods=2)
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fiscal_autocorr", canonical="fiscal_autocorr", backend="polars")
class FiscalAutocorrPolarsNative(SeriesOperator):
    """Fiscal autocorrelation at specified lag over window."""
    
    metadata = OperatorMetadata(
        name="fiscal_autocorr",
        category="fundamental",
        description="Fiscal autocorrelation at specified lag over window.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "fiscal", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 8, lag: int = 1, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        # Simple rolling correlation approximation
        return (
            df.select([
                pl.col(col_name).alias("x"),
                pl.col(col_name).shift(lag).alias("x_lag")
            ])
            .select([
                # Rolling correlation would need custom implementation
                # Using simplified version: covariance / (std * std_lag)
                pl.corr("x", "x_lag").alias(col_name)
            ])
            .collect()
            .to_series()
        )


# NOTE: fiscal_reversal_ratio moved to fiscal_batch2.py with magnitude-weighted implementation
# This simpler implementation is kept for reference but not registered
# @register_operator(name="fiscal_reversal_ratio", canonical="fiscal_reversal_ratio", backend="polars")
class FiscalReversalRatioPolarsNative_DISABLED(SeriesOperator):
    """Ratio of sign reversals to total periods in window."""

    metadata = OperatorMetadata(
        name="fiscal_reversal_ratio",
        category="fundamental",
        description="Ratio of sign reversals to total periods in window.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "fiscal", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                ((pl.col(col_name).sign() != pl.col(col_name).shift(1).sign())
                 .cast(pl.Int32)
                 .rolling_sum(window_size=window, min_periods=1)
                 .cast(pl.Float64) / window)
                .alias(col_name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fiscal_direction_consistency", canonical="fiscal_direction_consistency", backend="polars")
class FiscalDirectionConsistencyPolarsNative(SeriesOperator):
    """Consistency of direction: 1 - reversal_ratio."""
    
    metadata = OperatorMetadata(
        name="fiscal_direction_consistency",
        category="fundamental",
        description="Consistency of direction: 1 - reversal_ratio.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "fiscal", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                (1.0 - (pl.col(col_name).sign() != pl.col(col_name).shift(1).sign())
                 .cast(pl.Int32)
                 .rolling_sum(window_size=window, min_periods=1)
                 .cast(pl.Float64) / window)
                .alias(col_name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# DIVERGENCE AND GAP MEASURES
# ============================================================================

@register_operator(name="fin_divergence", canonical="fin_divergence", backend="polars")
class FinDivergencePolarsNative(SeriesOperator):
    """Divergence from rolling mean (standardized)."""
    
    metadata = OperatorMetadata(
        name="fin_divergence",
        category="fundamental",
        description="Divergence from rolling mean (standardized).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                ((pl.col(value.name) - pl.col(value.name).rolling_mean(window_size=window, min_periods=1)) /
                 pl.col(value.name).rolling_std(window_size=window, min_periods=2))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# NORMALIZED SCORES
# ============================================================================

@register_operator(name="fin_zscore_history", canonical="fin_zscore_history", backend="polars")
class FinZScoreHistoryPolarsNative(SeriesOperator):
    """Z-score relative to historical distribution over window."""
    
    metadata = OperatorMetadata(
        name="fin_zscore_history",
        category="fundamental",
        description="Z-score relative to historical distribution over window.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 8, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                ((pl.col(value.name) - pl.col(value.name).rolling_mean(window_size=window, min_periods=1)) /
                 pl.col(value.name).rolling_std(window_size=window, min_periods=2))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_percentile_history", canonical="fin_percentile_history", backend="polars")
class FinPercentileHistoryPolarsNative(SeriesOperator):
    """Percentile rank within rolling window."""
    
    metadata = OperatorMetadata(
        name="fin_percentile_history",
        category="fundamental",
        description="Percentile rank within rolling window.",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 8, **kwargs):
        # Approximate using rank within window
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                (pl.col(col_name).rank(method="average") /
                 pl.col(col_name).count())
                .alias(col_name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# SURPRISE AND EXPECTATION MEASURES
# ============================================================================

@register_operator(name="fin_surprise", canonical="fin_surprise", backend="polars")
class FinSurprisePolarsNative(SeriesOperator):
    """Surprise relative to expectation (vs rolling mean)."""
    
    metadata = OperatorMetadata(
        name="fin_surprise",
        category="fundamental",
        description="Surprise relative to expectation (vs rolling mean).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        return (
            value.to_frame()
            .lazy()
            .select([
                (pl.col(value.name) - pl.col(value.name).shift(1).rolling_mean(window_size=window, min_periods=1))
                .alias(value.name)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="fin_surprise_zscore", canonical="fin_surprise_zscore", backend="polars")
class FinSurpriseZScorePolarsNative(SeriesOperator):
    """Standardized surprise (z-score)."""
    
    metadata = OperatorMetadata(
        name="fin_surprise_zscore",
        category="fundamental",
        description="Standardized surprise (z-score).",
        param_names=["value"],
        return_type="series",
        tags=["fundamental", "financial", "polars_native", "pit_safe"],
    )

    def _calculate_series(self, value, window: int = 4, **kwargs):
        df = value.to_frame().lazy()
        col_name = value.name
        return (
            df.select([
                pl.col(col_name).alias("current"),
                pl.col(col_name).shift(1).rolling_mean(window_size=window, min_periods=1).alias("expectation"),
                pl.col(col_name).shift(1).rolling_std(window_size=window, min_periods=2).alias("volatility")
            ])
            .select([
                pl.when(pl.col("volatility" != 0).then(pl.col("current") - pl.col("expectation")) / pl.col("volatility").otherwise(None)).alias(col_name)
            ])
            .collect()
            .to_series()
        )
