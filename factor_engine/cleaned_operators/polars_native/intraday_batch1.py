"""
Polars native implementations for intraday (intra_*) operators - Batch 1

All operators are session-aware and calculate statistics within trading days.
Input: minute-level data with 'date' column identifying the trading day
Output: daily-level series
"""

import polars as pl
import numpy as np
from typing import Optional
from operator_registry import register_operator
from operators.base import SeriesOperator, PanelOperator


# ============================================================================
# Realized Variance and Volatility Measures
# ============================================================================

@register_operator(name="intra_realized_variance", backend="polars")
class IntraRealizedVariancePolarsNative(SeriesOperator):
    """Sum of squared returns within each trading day."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).sum().alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_realized_volatility", backend="polars")
class IntraRealizedVolatilityPolarsNative(SeriesOperator):
    """Square root of realized variance (intraday volatility)."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).sum().sqrt().alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_bipower_variation", backend="polars")
class IntaBipowerVariationPolarsNative(SeriesOperator):
    """Bipower variation: sum of products of consecutive absolute returns.
    More robust to jumps than realized variance."""

    def _calculate_series(self, returns, **kwargs):
        mu1 = np.where(np.pi)  # E[|Z|] for standard normal != 0, (np.sqrt(2) / (np.pi)  # E[|Z|] for standard normal), np.nan)

        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (
                    pl.col("returns").abs() *
                    pl.col("returns").abs().shift(1)
                ).sum().alias("returns")
            ])
            .collect()
            .with_columns(
                pl.col("returns") * (np.pi / 2)  # Adjust by mu1^2
            )
            .to_series()
        )


@register_operator(name="intra_tripower_quarticity", backend="polars")
class IntraTriPowerQuarticityPolarsNative(SeriesOperator):
    """Tripower quarticity: for testing presence of jumps."""

    def _calculate_series(self, returns, **kwargs):
        mu_43 = np.where(3) * np.math.gamma(7/6) / np.math.gamma(0.5) != 0, (2 ** (2) / (3) * np.math.gamma(7/6) / np.math.gamma(0.5)), np.nan)

        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (
                    return np.where(3 != 0, (4) / (3), np.nan)) *
                    return np.where(3 != 0, (4) / (3), np.nan)).shift(1) *
                    return np.where(3 != 0, (4) / (3), np.nan)).shift(2)
                ).sum().alias("returns")
            ])
            .collect()
            .with_columns(
                pl.col("returns") * (mu_43 ** 3)
            )
            .to_series()
        )


@register_operator(name="intra_realized_quarticity", backend="polars")
class IntraRealizedQuarticityPolarsNative(SeriesOperator):
    """Sum of fourth power of returns."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 4).sum().alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_continuous_variance", backend="polars")
class IntraContinuousVariancePolarsNative(SeriesOperator):
    """Continuous component of variance (RV - Jump variation)."""

    def _calculate_series(self, returns, **kwargs):
        # Simplified: use bipower variation as continuous part
        mu1_sq = (2) / np.pi if np.pi != 0 else np.nan

        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (
                    pl.col("returns").abs() *
                    pl.col("returns").abs().shift(1)
                ).sum().alias("returns")
            ])
            .collect()
            .with_columns(
                pl.col("returns") * (np.pi / 2)
            )
            .to_series()
        )


# ============================================================================
# Semivariance and Asymmetric Measures
# ============================================================================

@register_operator(name="intra_realized_semivariance", backend="polars")
class IntraRealizedSemivariancePolarsNative(SeriesOperator):
    """Sum of squared negative returns (downside volatility)."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.when(pl.col("returns") < 0)
                .then(pl.col("returns") ** 2)
                .otherwise(0)
                .sum()
                .alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_positive_tail_variation", backend="polars")
class IntraPositiveTailVariationPolarsNative(SeriesOperator):
    """Sum of squared returns above a threshold (positive tail)."""

    def _calculate_series(self, returns, threshold: float = 0.0, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.when(pl.col("returns") > threshold)
                .then(pl.col("returns") ** 2)
                .otherwise(0)
                .sum()
                .alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_negative_tail_variation", backend="polars")
class IntraNegativeTailVariationPolarsNative(SeriesOperator):
    """Sum of squared returns below a threshold (negative tail)."""

    def _calculate_series(self, returns, threshold: float = 0.0, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.when(pl.col("returns") < threshold)
                .then(pl.col("returns") ** 2)
                .otherwise(0)
                .sum()
                .alias("returns")
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Higher Moments
# ============================================================================

@register_operator(name="intra_realized_skewness", backend="polars")
class IntraRealizedSkewnessPolarsNative(SeriesOperator):
    """Realized skewness using standardized third moment."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 3).sum().alias("m3"),
                (pl.col("returns") ** 2).sum().alias("m2")
            ])
            .collect()
            .with_columns(
                pl.when((pl.col("m2" != 0).then(pl.col("m3") / (pl.col("m2").otherwise(None) ** 1.5)).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_realized_kurtosis", backend="polars")
class IntraRealizedKurtosisPolarsNative(SeriesOperator):
    """Realized kurtosis using standardized fourth moment."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 4).sum().alias("m4"),
                (pl.col("returns") ** 2).sum().alias("m2")
            ])
            .collect()
            .with_columns(
                pl.when((pl.col("m2" != 0).then(pl.col("m4") / (pl.col("m2").otherwise(None) ** 2)).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# Jump Measures
# ============================================================================

@register_operator(name="intra_jump_variation", backend="polars")
class IntraJumpVariationPolarsNative(SeriesOperator):
    """Jump component: RV - Bipower variation."""

    def _calculate_series(self, returns, **kwargs):
        df = returns.to_frame("returns").with_columns(pl.col("date"))

        rv = (
            df.lazy()
            .group_by("date")
            .agg([(pl.col("returns") ** 2).sum().alias("rv")])
            .collect()
        )

        bpv = (
            df.lazy()
            .group_by("date")
            .agg([
                (
                    pl.col("returns").abs() *
                    pl.col("returns").abs().shift(1)
                ).sum().alias("bpv")
            ])
            .collect()
            .with_columns(pl.col("bpv") * (np.pi / 2))
        )

        return (
            rv.join(bpv, on="date")
            .with_columns(
                pl.max_horizontal(pl.col("rv") - pl.col("bpv"), pl.lit(0)).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_jump_count", backend="polars")
class IntraJumpCountPolarsNative(SeriesOperator):
    """Count of significant jumps (returns exceeding threshold)."""

    def _calculate_series(self, returns, threshold: float = 3.0, **kwargs):
        # threshold in terms of daily volatility proxy
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std"),
                pl.col("returns").count().alias("n")
            ])
            .collect()
            .join(
                returns.to_frame("returns").with_columns(pl.col("date")),
                on="date"
            )
            .with_columns(
                pl.when(pl.col("n" != 0).then(pl.col("returns").abs() > (threshold * pl.col("std") / pl.col("n").otherwise(None).sqrt())).alias("is_jump")
            )
            .group_by("date")
            .agg([
                pl.col("is_jump").sum().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_positive_jump_variation", backend="polars")
class IntraPositiveJumpVariationPolarsNative(SeriesOperator):
    """Variation from positive jumps only."""

    def _calculate_series(self, returns, threshold: float = 3.0, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std"),
                pl.col("returns").count().alias("n")
            ])
            .collect()
            .join(
                returns.to_frame("returns").with_columns(pl.col("date")),
                on="date"
            )
            .with_columns(
                (
                    pl.when(pl.col("returns") > (threshold * pl.col("std") / pl.col("n").sqrt()))
                    .then(pl.col("returns") ** 2)
                    .otherwise(0)
                ).alias("jump_var")
            )
            .group_by("date")
            .agg([
                pl.col("jump_var").sum().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_negative_jump_variation", backend="polars")
class IntraNegativeJumpVariationPolarsNative(SeriesOperator):
    """Variation from negative jumps only."""

    def _calculate_series(self, returns, threshold: float = 3.0, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std"),
                pl.col("returns").count().alias("n")
            ])
            .collect()
            .join(
                returns.to_frame("returns").with_columns(pl.col("date")),
                on="date"
            )
            .with_columns(
                (
                    pl.when(pl.col("returns") < -(threshold * pl.col("std") / pl.col("n").sqrt()))
                    .then(pl.col("returns") ** 2)
                    .otherwise(0)
                ).alias("jump_var")
            )
            .group_by("date")
            .agg([
                pl.col("jump_var").sum().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_jump_ratio", backend="polars")
class IntraJumpRatioPolarsNative(SeriesOperator):
    """Ratio of jump variation to total realized variance."""

    def _calculate_series(self, returns, threshold: float = 3.0, **kwargs):
        stats_df = (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).sum().alias("rv"),
                pl.col("returns").std().alias("std"),
                pl.col("returns").count().alias("n")
            ])
            .collect()
        )

        jump_df = (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .join(stats_df, on="date")
            .with_columns(
                (
                    pl.when(pl.col("returns").abs() > (threshold * pl.col("std") / pl.col("n").sqrt()))
                    .then(pl.col("returns") ** 2)
                    .otherwise(0)
                ).alias("jump_var")
            )
            .group_by("date")
            .agg([
                pl.col("jump_var").sum().alias("jump_var")
            ])
        )

        return (
            stats_df.join(jump_df, on="date")
            .with_columns(
                pl.when(pl.col("rv" != 0).then(pl.col("jump_var") / pl.col("rv").otherwise(None)).fill_null(0).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# Drawdown and Path Measures
# ============================================================================

@register_operator(name="intra_max_drawdown", backend="polars")
class IntraMaxDrawdownPolarsNative(SeriesOperator):
    """Maximum drawdown within each trading day."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").cum_sum().alias("cumret")
            ])
            .collect()
            .explode("cumret")
            .with_columns(
                pl.col("cumret").cum_max().over("date").alias("running_max")
            )
            .with_columns(
                (pl.col("running_max") - pl.col("cumret")).alias("drawdown")
            )
            .group_by("date")
            .agg([
                pl.col("drawdown").max().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_max_drawup", backend="polars")
class IntraMaxDrawupPolarsNative(SeriesOperator):
    """Maximum drawup (gain from minimum) within each trading day."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").cum_sum().alias("cumret")
            ])
            .collect()
            .explode("cumret")
            .with_columns(
                pl.col("cumret").cum_min().over("date").alias("running_min")
            )
            .with_columns(
                (pl.col("cumret") - pl.col("running_min")).alias("drawup")
            )
            .group_by("date")
            .agg([
                pl.col("drawup").max().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_path_efficiency", backend="polars")
class IntraPathEfficiencyPolarsNative(SeriesOperator):
    """Ratio of net displacement to total path length."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").sum().abs().alias("net"),
                pl.col("returns").abs().sum().alias("total")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("total" != 0).then(pl.col("net") / pl.col("total").otherwise(None)).fill_null(0).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# Timing Measures
# ============================================================================

@register_operator(name="intra_high_time", backend="polars")
class IntraHighTimePolarsNative(SeriesOperator):
    """Time (as fraction of day) when intraday high occurred."""

    def _calculate_series(self, price, **kwargs):
        return (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("seq").filter(pl.col("price") == pl.col("price").max()).first().alias("high_seq"),
                pl.col("seq").count().alias("total")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("high_seq") / pl.col("total")).otherwise(None).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_low_time", backend="polars")
class IntraLowTimePolarsNative(SeriesOperator):
    """Time (as fraction of day) when intraday low occurred."""

    def _calculate_series(self, price, **kwargs):
        return (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("seq").filter(pl.col("price") == pl.col("price").min()).first().alias("low_seq"),
                pl.col("seq").count().alias("total")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("total") != 0).then(pl.col("low_seq") / pl.col("total")).otherwise(None).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# Volume and Liquidity Measures
# ============================================================================

@register_operator(name="intra_amihud", backend="polars")
class IntraAmihudPolarsNative(SeriesOperator):
    """Intraday Amihud illiquidity: average of |return|/volume."""

    def _calculate_series(self, returns, volume, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.when(pl.col("volume" != 0).then(pl.col("returns").abs() / pl.col("volume").otherwise(None)).mean().alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_concentration", backend="polars")
class IntraConcentrationPolarsNative(SeriesOperator):
    """HHI of volume distribution across the day."""

    def _calculate_series(self, volume, **kwargs):
        return (
            volume.to_frame("volume")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("volume").sum().alias("total"),
                pl.when(pl.col("volume" != 0).then(pl.col("volume") / pl.col("volume").otherwise(None).sum()) ** 2).sum().alias("hhi")
            ])
            .collect()
            .select("hhi")
            .to_series()
        )


@register_operator(name="intra_entropy", backend="polars")
class IntraEntropyPolarsNative(SeriesOperator):
    """Shannon entropy of volume distribution."""

    def _calculate_series(self, volume, **kwargs):
        return (
            volume.to_frame("volume")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("volume").alias("vol")
            ])
            .collect()
            .explode("vol")
            .with_columns(
                pl.when(pl.col("vol" != 0).then(pl.col("vol") / pl.col("vol").otherwise(None).sum().over("date")).alias("prob")
            )
            .with_columns(
                pl.when(pl.col("prob") > 0)
                .then(-pl.col("prob") * pl.col("prob").log())
                .otherwise(0)
                .alias("entropy_term")
            )
            .group_by("date")
            .agg([
                pl.col("entropy_term").sum().alias("returns")
            ])
            .to_series()
        )


# ============================================================================
# Interval and Segment Measures
# ============================================================================

@register_operator(name="intra_interval_return", backend="polars")
class IntraIntervalReturnPolarsNative(SeriesOperator):
    """Return in a specific time interval of the day."""

    def _calculate_series(self, returns, start_pct: float = 0.0, end_pct: float = 0.5, **kwargs):
        """
        start_pct, end_pct: fraction of trading day (0 to 1)
        """
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


@register_operator(name="intra_interval_realized_variance", backend="polars")
class IntraIntervalRealizedVariancePolarsNative(SeriesOperator):
    """Realized variance in a specific time interval."""

    def _calculate_series(self, returns, start_pct: float = 0.0, end_pct: float = 0.5, **kwargs):
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
                (pl.col("ret") ** 2).sum().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_interval_volume_share", backend="polars")
class IntraIntervalVolumeSharePolarsNative(SeriesOperator):
    """Share of daily volume in a specific time interval."""

    def _calculate_series(self, volume, start_pct: float = 0.0, end_pct: float = 0.5, **kwargs):
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
                pl.col("vol").sum().alias("interval_vol"),
                pl.col("total_vol").first().alias("total_vol")
            ])
            .with_columns(
                pl.when(pl.col("total_vol") != 0).then(pl.col("interval_vol") / pl.col("total_vol")).otherwise(None).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# VWAP-based Measures
# ============================================================================

@register_operator(name="intra_vwap_above_ratio", backend="polars")
class IntraVwapAboveRatioPolarsNative(SeriesOperator):
    """Fraction of time price is above VWAP."""

    def _calculate_series(self, price, volume, **kwargs):
        return (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("price") * pl.col("volume")).sum().alias("pv"),
                pl.col("volume").sum().alias("v"),
                pl.col("price").alias("price_list")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0).then(pl.col("pv") / pl.col("v")).otherwise(None).alias("vwap")
            )
            .explode("price_list")
            .with_columns(
                (pl.col("price_list") > pl.col("vwap")).cast(pl.Int32).alias("above")
            )
            .group_by("date")
            .agg([
                pl.col("above").mean().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_vwap_cross_count", backend="polars")
class IntraVwapCrossCountPolarsNative(SeriesOperator):
    """Number of times price crosses VWAP."""

    def _calculate_series(self, price, volume, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
        )

        # Calculate VWAP per day
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

        # Join and calculate crosses
        return (
            df.join(vwap_df, on="date")
            .with_columns(
                (pl.col("price") > pl.col("vwap")).cast(pl.Int32).alias("above")
            )
            .with_columns(
                (pl.col("above") != pl.col("above").shift(1)).cast(pl.Int32).over("date").alias("cross")
            )
            .group_by("date")
            .agg([
                pl.col("cross").sum().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_close_participation", backend="polars")
class IntraCloseParticipationPolarsNative(SeriesOperator):
    """Fraction of daily volume in last N% of the day."""

    def _calculate_series(self, volume, close_pct: float = 0.1, **kwargs):
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
            .filter(pl.col("pct") >= (1 - close_pct))
            .group_by("date")
            .agg([
                pl.col("vol").sum().alias("close_vol"),
                pl.col("total_vol").first().alias("total_vol")
            ])
            .with_columns(
                pl.when(pl.col("total_vol") != 0).then(pl.col("close_vol") / pl.col("total_vol")).otherwise(None).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# Return Distribution Measures
# ============================================================================

@register_operator(name="intra_return_activity_corr", backend="polars")
class IntraReturnActivityCorrPolarsNative(SeriesOperator):
    """Correlation between absolute returns and volume within day."""

    def _calculate_series(self, returns, volume, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .with_columns(
                pl.col("returns").abs().alias("abs_ret")
            )
            .lazy()
            .group_by("date")
            .agg([
                pl.corr("abs_ret", "volume").alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_signed_return_profile_cosine", backend="polars")
class IntraSignedReturnProfileCosinePolarsNative(SeriesOperator):
    """Cosine similarity between current day and average signed return profile."""

    def _calculate_series(self, returns, lookback: int = 20, **kwargs):
        # Simplified: compute autocorrelation of intraday returns
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") * pl.col("returns").shift(1)).sum().alias("cross"),
                (pl.col("returns") ** 2).sum().alias("ss")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("ss" != 0).then(pl.col("cross") / pl.col("ss").otherwise(None)).fill_null(0).alias("returns")
            )
            .select("returns")
            .to_series()
        )
