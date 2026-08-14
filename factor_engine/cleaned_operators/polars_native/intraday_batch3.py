"""
Polars native implementations for intraday (intra_*) operators - Batch 3

Interval measures, VWAP dynamics, extremes, volume profiles
"""

import polars as pl
import numpy as np
from typing import Optional
from operator_registry import register_operator
from operators.base import SeriesOperator, PanelOperator


# ============================================================================
# Interval Amount/Volume Measures
# ============================================================================

@register_operator(name="intra_interval_amount_share", backend="polars", research_only=True)
class IntraIntervalAmountSharePolarsNative(SeriesOperator):
    """Share of daily dollar volume in a specific time interval."""

    def _calculate_series(self, amount, start_pct: float = 0.0, end_pct: float = 0.5, **kwargs):
        return (
            amount.to_frame("amount")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("seq").count().alias("total"),
                pl.col("amount").sum().alias("total_amt"),
                pl.col("amount").alias("amt"),
                pl.col("seq").alias("seq_list")
            ])
            .collect()
            .explode(["amt", "seq_list"])
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("seq_list") / pl.col("total")).otherwise(None).alias("pct")
            )
            .filter(
                (pl.col("pct") >= start_pct) & (pl.col("pct") < end_pct)
            )
            .group_by("date")
            .agg([
                pl.col("amt").sum().alias("interval_amt"),
                pl.col("total_amt").first().alias("total_amt")
            ])
            .with_columns(
                pl.when(pl.col("total_amt") != 0).then(pl.col("interval_amt") / pl.col("total_amt")).otherwise(None).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_interval_illiquidity", backend="polars", research_only=True)
class IntraIntervalIlliquidityPolarsNative(SeriesOperator):
    """Amihud measure for a specific interval."""

    def _calculate_series(self, returns, volume, start_pct: float = 0.0, end_pct: float = 0.5, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                volume.alias("volume"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("seq").count().alias("total"),
                pl.col("returns").alias("ret"),
                pl.col("volume").alias("vol"),
                pl.col("seq").alias("seq_list")
            ])
            .collect()
            .explode(["ret", "vol", "seq_list"])
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("seq_list") / pl.col("total")).otherwise(None).alias("pct")
            )
            .filter(
                (pl.col("pct") >= start_pct) & (pl.col("pct") < end_pct)
            )
            .group_by("date")
            .agg([
                (pl.col("ret").abs() / pl.col("vol")).mean().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_interval_vwap_deviation", backend="polars", research_only=True)
class IntraIntervalVwapDeviationPolarsNative(SeriesOperator):
    """Average price deviation from interval VWAP."""

    def _calculate_series(self, price, volume, start_pct: float = 0.0, end_pct: float = 0.5, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
        )

        daily_count = (
            df.lazy()
            .group_by("date")
            .agg([
                pl.col("seq").count().alias("total")
            ])
            .collect()
        )

        interval_df = (
            df.join(daily_count, on="date")
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("seq") / pl.col("total")).otherwise(None).alias("pct")
            )
            .filter(
                (pl.col("pct") >= start_pct) & (pl.col("pct") < end_pct)
            )
        )

        # Calculate interval VWAP
        vwap_df = (
            interval_df.group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        # Calculate deviation
        return (
            interval_df.join(vwap_df, on="date")
            .with_columns(
                ((pl.col("price") - pl.col("vwap")).abs() * pl.col("volume")).alias("weighted_dev")
            )
            .group_by("date")
            .agg([
                pl.col("weighted_dev").sum().alias("total_dev"),
                pl.col("volume").sum().alias("total_vol")
            ])
            .with_columns(
                pl.when(pl.col("total_vol") != 0).then(pl.col("total_dev") / pl.col("total_vol")).otherwise(None).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# Segment Measures (Multiple intervals)
# ============================================================================

@register_operator(name="intra_segment_return", backend="polars", research_only=True)
class IntraSegmentReturnPolarsNative(SeriesOperator):
    """Return in segment N of K equal segments."""

    def _calculate_series(self, returns, segment: int = 1, n_segments: int = 4, **kwargs):
        """segment: 1-indexed segment number"""
        start_pct = ((segment - 1)) / n_segments if n_segments != 0 else np.nan
        end_pct = (segment) / n_segments if n_segments != 0 else np.nan

        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("seq").count().alias("total"),
                pl.col("returns").alias("ret"),
                pl.col("seq").alias("seq_list")
            ])
            .collect()
            .explode(["ret", "seq_list"])
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("seq_list") / pl.col("total")).otherwise(None).alias("pct")
            )
            .filter(
                (pl.col("pct") >= start_pct) & (pl.col("pct") < end_pct)
            )
            .group_by("date")
            .agg([
                pl.col("ret").sum().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_segment_realized_vol", backend="polars", research_only=True)
class IntraSegmentRealizedVolPolarsNative(SeriesOperator):
    """Realized volatility in segment N of K equal segments."""

    def _calculate_series(self, returns, segment: int = 1, n_segments: int = 4, **kwargs):
        start_pct = ((segment - 1)) / n_segments if n_segments != 0 else np.nan
        end_pct = (segment) / n_segments if n_segments != 0 else np.nan

        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("seq").count().alias("total"),
                pl.col("returns").alias("ret"),
                pl.col("seq").alias("seq_list")
            ])
            .collect()
            .explode(["ret", "seq_list"])
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("seq_list") / pl.col("total")).otherwise(None).alias("pct")
            )
            .filter(
                (pl.col("pct") >= start_pct) & (pl.col("pct") < end_pct)
            )
            .group_by("date")
            .agg([
                (pl.col("ret") ** 2).sum().sqrt().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_segment_volume_share", backend="polars", research_only=True)
class IntraSegmentVolumeSharePolarsNative(SeriesOperator):
    """Volume share in segment N of K equal segments."""

    def _calculate_series(self, volume, segment: int = 1, n_segments: int = 4, **kwargs):
        start_pct = ((segment - 1)) / n_segments if n_segments != 0 else np.nan
        end_pct = (segment) / n_segments if n_segments != 0 else np.nan

        return (
            volume.to_frame("volume")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("seq").count().alias("total"),
                pl.col("volume").sum().alias("total_vol"),
                pl.col("volume").alias("vol"),
                pl.col("seq").alias("seq_list")
            ])
            .collect()
            .explode(["vol", "seq_list"])
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("seq_list") / pl.col("total")).otherwise(None).alias("pct")
            )
            .filter(
                (pl.col("pct") >= start_pct) & (pl.col("pct") < end_pct)
            )
            .group_by("date")
            .agg([
                pl.col("vol").sum().alias("seg_vol"),
                pl.col("total_vol").first().alias("total_vol")
            ])
            .with_columns(
                pl.when(pl.col("total_vol") != 0).then(pl.col("seg_vol") / pl.col("total_vol")).otherwise(None).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_segment_amount_share", backend="polars", research_only=True)
class IntraSegmentAmountSharePolarsNative(SeriesOperator):
    """Dollar volume share in segment N of K equal segments."""

    def _calculate_series(self, amount, segment: int = 1, n_segments: int = 4, **kwargs):
        start_pct = ((segment - 1)) / n_segments if n_segments != 0 else np.nan
        end_pct = (segment) / n_segments if n_segments != 0 else np.nan

        return (
            amount.to_frame("amount")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("seq").count().alias("total"),
                pl.col("amount").sum().alias("total_amt"),
                pl.col("amount").alias("amt"),
                pl.col("seq").alias("seq_list")
            ])
            .collect()
            .explode(["amt", "seq_list"])
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("seq_list") / pl.col("total")).otherwise(None).alias("pct")
            )
            .filter(
                (pl.col("pct") >= start_pct) & (pl.col("pct") < end_pct)
            )
            .group_by("date")
            .agg([
                pl.col("amt").sum().alias("seg_amt"),
                pl.col("total_amt").first().alias("total_amt")
            ])
            .with_columns(
                pl.when(pl.col("total_amt") != 0).then(pl.col("seg_amt") / pl.col("total_amt")).otherwise(None).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_segment_vwap_deviation", backend="polars", research_only=True)
class IntraSegmentVwapDeviationPolarsNative(SeriesOperator):
    """Price deviation from segment VWAP in segment N."""

    def _calculate_series(self, price, volume, segment: int = 1, n_segments: int = 4, **kwargs):
        start_pct = ((segment - 1)) / n_segments if n_segments != 0 else np.nan
        end_pct = (segment) / n_segments if n_segments != 0 else np.nan

        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
        )

        daily_count = (
            df.lazy()
            .group_by("date")
            .agg([
                pl.col("seq").count().alias("total")
            ])
            .collect()
        )

        segment_df = (
            df.join(daily_count, on="date")
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("seq") / pl.col("total")).otherwise(None).alias("pct")
            )
            .filter(
                (pl.col("pct") >= start_pct) & (pl.col("pct") < end_pct)
            )
        )

        vwap_df = (
            segment_df.group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        return (
            segment_df.join(vwap_df, on="date")
            .with_columns(
                ((pl.col("price") - pl.col("vwap")).abs() * pl.col("volume")).alias("weighted_dev")
            )
            .group_by("date")
            .agg([
                pl.col("weighted_dev").sum().alias("total_dev"),
                pl.col("volume").sum().alias("total_vol")
            ])
            .with_columns(
                pl.when(pl.col("total_vol") != 0).then(pl.col("total_dev") / pl.col("total_vol")).otherwise(None).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# VWAP Path Dynamics
# ============================================================================

@register_operator(name="intra_vwap_path_slope", backend="polars", research_only=True)
class IntraVwapPathSlopePolarsNative(SeriesOperator):
    """Slope of price relative to VWAP over the day."""

    def _calculate_series(self, price, volume, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
        )

        # Calculate daily VWAP
        vwap_df = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        # Linear regression of (price - vwap) on time
        return (
            df.join(vwap_df, on="date")
            .with_columns(
                (pl.col("price") - pl.col("vwap")).alias("dev")
            )
            .lazy()
            .group_by("date")
            .agg([
                pl.cov("seq", "dev").alias("cov"),
                pl.var("seq").alias("var")
            ])
            .collect()
            .with_columns(
                (pl.col("cov") / pl.col("var")).fill_null(0).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_vwap_path_curvature", backend="polars", research_only=True)
class IntraVwapPathCurvaturePolarsNative(SeriesOperator):
    """Curvature (quadratic term) of price deviation from VWAP."""

    def _calculate_series(self, price, volume, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
        )

        vwap_df = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        # Simplified: use second difference as proxy for curvature
        return (
            df.join(vwap_df, on="date")
            .with_columns(
                (pl.col("price") - pl.col("vwap")).alias("dev")
            )
            .with_columns(
                pl.col("dev").diff().over("date").alias("first_diff")
            )
            .with_columns(
                pl.col("first_diff").diff().over("date").alias("second_diff")
            )
            .group_by("date")
            .agg([
                pl.col("second_diff").abs().mean().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_vwap_reversion_speed", backend="polars", research_only=True)
class IntraVwapReversionSpeedPolarsNative(SeriesOperator):
    """Speed of mean reversion to VWAP (AR(1) coefficient of deviations)."""

    def _calculate_series(self, price, volume, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
        )

        vwap_df = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        return (
            df.join(vwap_df, on="date")
            .with_columns(
                (pl.col("price") - pl.col("vwap")).alias("dev")
            )
            .with_columns(
                pl.col("dev").shift(1).over("date").alias("dev_lag")
            )
            .filter(pl.col("dev_lag").is_not_null())
            .lazy()
            .group_by("date")
            .agg([
                pl.corr("dev", "dev_lag").alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_time_above_vwap", backend="polars", research_only=True)
class IntraTimeAboveVwapPolarsNative(SeriesOperator):
    """Total time (in bars) price is above VWAP."""

    def _calculate_series(self, price, volume, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
        )

        vwap_df = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        return (
            df.join(vwap_df, on="date")
            .with_columns(
                (pl.col("price") > pl.col("vwap")).cast(pl.Int32).alias("above")
            )
            .group_by("date")
            .agg([
                pl.col("above").sum().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_longest_above_vwap_streak", backend="polars", research_only=True)
class IntraLongestAboveVwapStreakPolarsNative(SeriesOperator):
    """Longest consecutive bars price stays above VWAP."""

    def _calculate_series(self, price, volume, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
        )

        vwap_df = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        return (
            df.join(vwap_df, on="date")
            .with_columns(
                (pl.col("price") > pl.col("vwap")).cast(pl.Int32).alias("above")
            )
            .with_columns(
                pl.when(pl.col("above") == 0)
                .then(pl.lit(1))
                .otherwise(0)
                .cum_sum()
                .over("date")
                .alias("group_id")
            )
            .filter(pl.col("above") == 1)
            .group_by(["date", "group_id"])
            .agg([
                pl.count().alias("streak")
            ])
            .group_by("date")
            .agg([
                pl.col("streak").max().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_longest_below_vwap_streak", backend="polars", research_only=True)
class IntraLongestBelowVwapStreakPolarsNative(SeriesOperator):
    """Longest consecutive bars price stays below VWAP."""

    def _calculate_series(self, price, volume, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
        )

        vwap_df = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        return (
            df.join(vwap_df, on="date")
            .with_columns(
                (pl.col("price") < pl.col("vwap")).cast(pl.Int32).alias("below")
            )
            .with_columns(
                pl.when(pl.col("below") == 0)
                .then(pl.lit(1))
                .otherwise(0)
                .cum_sum()
                .over("date")
                .alias("group_id")
            )
            .filter(pl.col("below") == 1)
            .group_by(["date", "group_id"])
            .agg([
                pl.count().alias("streak")
            ])
            .group_by("date")
            .agg([
                pl.col("streak").max().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_price_vwap_max_positive_excursion", backend="polars", research_only=True)
class IntraPriceVwapMaxPositiveExcursionPolarsNative(SeriesOperator):
    """Maximum positive excursion of price from VWAP."""

    def _calculate_series(self, price, volume, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
        )

        vwap_df = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        return (
            df.join(vwap_df, on="date")
            .with_columns(
                pl.when(pl.col("vwap" != 0).then(pl.col("price") - pl.col("vwap")) / pl.col("vwap").otherwise(None)).alias("dev_pct")
            )
            .group_by("date")
            .agg([
                pl.col("dev_pct").max().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_price_vwap_max_negative_excursion", backend="polars", research_only=True)
class IntraPriceVwapMaxNegativeExcursionPolarsNative(SeriesOperator):
    """Maximum negative excursion of price from VWAP."""

    def _calculate_series(self, price, volume, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
        )

        vwap_df = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .select(["date", "vwap"])
        )

        return (
            df.join(vwap_df, on="date")
            .with_columns(
                pl.when(pl.col("vwap" != 0).then(pl.col("price") - pl.col("vwap")) / pl.col("vwap").otherwise(None)).alias("dev_pct")
            )
            .group_by("date")
            .agg([
                pl.col("dev_pct").min().alias("returns")
            ])
            .to_series()
        )


# ============================================================================
# Bar Range and Volatility Measures
# ============================================================================

@register_operator(name="intra_bar_range_deviation", backend="polars", research_only=True)
class IntraBarRangeDeviationPolarsNative(SeriesOperator):
    """Standard deviation of bar ranges (high - low)."""

    def _calculate_series(self, high, low, **kwargs):
        return (
            high.to_frame("high")
            .with_columns([
                pl.col("date"),
                low.alias("low")
            ])
            .with_columns(
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .lazy()
            .group_by("date")
            .agg([
                pl.col("range").std().alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_bar_range_persistence", backend="polars", research_only=True)
class IntraBarRangePersistencePolarsNative(SeriesOperator):
    """Autocorrelation of bar ranges."""

    def _calculate_series(self, high, low, **kwargs):
        return (
            high.to_frame("high")
            .with_columns([
                pl.col("date"),
                low.alias("low")
            ])
            .with_columns(
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .with_columns(
                pl.col("range").shift(1).over("date").alias("range_lag")
            )
            .filter(pl.col("range_lag").is_not_null())
            .lazy()
            .group_by("date")
            .agg([
                pl.corr("range", "range_lag").alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_extreme_bar_return", backend="polars", research_only=True)
class IntraExtremeBarReturnPolarsNative(SeriesOperator):
    """Return of the bar with the largest absolute return."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .with_columns(
                pl.col("returns").abs().alias("abs_ret")
            )
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").filter(pl.col("abs_ret") == pl.col("abs_ret").max()).first().alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_high_low_affinity", backend="polars", research_only=True)
class IntraHighLowAffinityPolarsNative(SeriesOperator):
    """Correlation between high and low prices within day."""

    def _calculate_series(self, high, low, **kwargs):
        return (
            high.to_frame("high")
            .with_columns([
                pl.col("date"),
                low.alias("low")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.corr("high", "low").alias("returns")
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Tail Measures
# ============================================================================

@register_operator(name="intra_tail_event_count", backend="polars", research_only=True)
class IntraTailEventCountPolarsNative(SeriesOperator):
    """Count of extreme tail events (beyond threshold)."""

    def _calculate_series(self, returns, threshold: float = 2.5, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std")
            ])
            .collect()
            .join(
                returns.to_frame("returns").with_columns(pl.col("date")),
                on="date"
            )
            .with_columns(
                (pl.col("returns").abs() > (threshold * pl.col("std"))).cast(pl.Int32).alias("is_tail")
            )
            .group_by("date")
            .agg([
                pl.col("is_tail").sum().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_tail_volume_share", backend="polars", research_only=True)
class IntraTailVolumeSharePolarsNative(SeriesOperator):
    """Share of volume during tail events."""

    def _calculate_series(self, returns, volume, threshold: float = 2.5, **kwargs):
        stats = (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std")
            ])
            .collect()
        )

        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .join(stats, on="date")
            .with_columns(
                (pl.col("returns").abs() > (threshold * pl.col("std"))).alias("is_tail")
            )
            .group_by("date")
            .agg([
                pl.when(pl.col("is_tail")).then(pl.col("volume")).sum().alias("tail_vol"),
                pl.col("volume").sum().alias("total_vol")
            ])
            .with_columns(
                (pl.col("tail_vol") / pl.col("total_vol")).fill_null(0).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_signed_tail_variation_ratio", backend="polars", research_only=True)
class IntraSignedTailVariationRatioPolarsNative(SeriesOperator):
    """Ratio of positive to negative tail variation."""

    def _calculate_series(self, returns, threshold: float = 2.5, **kwargs):
        stats = (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std")
            ])
            .collect()
        )

        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .join(stats, on="date")
            .with_columns(
                (pl.col("returns").abs() > (threshold * pl.col("std"))).alias("is_tail")
            )
            .filter(pl.col("is_tail"))
            .group_by("date")
            .agg([
                pl.when(pl.col("returns") > 0).then(pl.col("returns") ** 2).sum().alias("pos_var"),
                pl.when(pl.col("returns") < 0).then(pl.col("returns") ** 2).sum().alias("neg_var")
            ])
            .with_columns(
                (pl.col("pos_var") / (pl.col("neg_var") + 1e-10)).fill_null(1).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# Same-Slot Measures (comparison with same time on different days)
# ============================================================================

@register_operator(name="intra_same_slot_momentum", backend="polars", research_only=True)
class IntraSameSlotMomentumPolarsNative(SeriesOperator):
    """Average return at same intraday slot across recent days."""

    def _calculate_series(self, returns, lookback: int = 5, **kwargs):
        """
        For each intraday slot, compute average return at that slot over past lookback days
        Then average across all slots for the current day
        """
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("slot")
            ])
        )

        # For each date-slot, compute average of same slot in past
        return (
            df.with_columns(
                pl.col("returns")
                .rolling_mean(window_size=lookback, min_periods=1)
                .over("slot")
                .alias("slot_avg")
            )
            .group_by("date")
            .agg([
                pl.col("slot_avg").mean().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_same_slot_reversal", backend="polars", research_only=True)
class IntraSameSlotReversalPolarsNative(SeriesOperator):
    """Negative of same-slot momentum (reversal signal)."""

    def _calculate_series(self, returns, lookback: int = 5, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("slot")
            ])
        )

        return (
            df.with_columns(
                -pl.col("returns")
                .rolling_mean(window_size=lookback, min_periods=1)
                .over("slot")
                .alias("slot_avg")
            )
            .group_by("date")
            .agg([
                pl.col("slot_avg").mean().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_same_slot_zscore", backend="polars", research_only=True)
class IntraSameSlotZscorePolarsNative(SeriesOperator):
    """Average z-score of returns relative to same-slot distribution."""

    def _calculate_series(self, returns, lookback: int = 20, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("slot")
            ])
        )

        return (
            df.with_columns([
                pl.col("returns")
                .rolling_mean(window_size=lookback, min_periods=2)
                .over("slot")
                .alias("slot_mean"),
                pl.col("returns")
                .rolling_std(window_size=lookback, min_periods=2)
                .over("slot")
                .alias("slot_std")
            ])
            .with_columns(
                ((pl.col("returns") - pl.col("slot_mean")) / (pl.col("slot_std") + 1e-10)).alias("zscore")
            )
            .group_by("date")
            .agg([
                pl.col("zscore").mean().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_consolidation_quality", backend="polars", research_only=True)
class IntraConsolidationQualityPolarsNative(SeriesOperator):
    """Measure of price stability: inverse of range/mean ratio."""

    def _calculate_series(self, high, low, close, **kwargs):
        return (
            high.to_frame("high")
            .with_columns([
                pl.col("date"),
                low.alias("low"),
                close.alias("close")
            ])
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("high").max() - pl.col("low").min()).alias("range"),
                pl.col("close").mean().alias("mean_price")
            ])
            .collect()
            .with_columns(
                (1.0 / ((pl.col("range") / pl.col("mean_price")) + 1e-10)).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_eod_reversal_decomposition", backend="polars", research_only=True)
class IntraEodReversalDecompositionPolarsNative(SeriesOperator):
    """Return from intraday high/low to close (measuring end-of-day reversal)."""

    def _calculate_series(self, high, low, close, **kwargs):
        """Negative if close near low, positive if close near high"""
        return (
            high.to_frame("high")
            .with_columns([
                pl.col("date"),
                low.alias("low"),
                close.alias("close")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close")
            ])
            .collect()
            .with_columns(
                ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))).alias("returns")
            )
            .select("returns")
            .to_series()
        )
