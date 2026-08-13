"""
Polars native implementations for intraday (intra_*) operators - Batch 2

Advanced measures: beta, correlation, session effects, drawdown dynamics
"""

import polars as pl
import numpy as np
from typing import Optional
from operator_registry import register_operator
from operators.base import SeriesOperator, PanelOperator


# ============================================================================
# Beta and Correlation Measures (Panel operators)
# ============================================================================

@register_operator(name="intra_realized_beta", backend="polars")
class IntraRealizedBetaPolarsNative(PanelOperator):
    """Intraday realized beta against benchmark."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        """
        returns: stock returns (minute level)
        benchmark_returns: market returns (minute level)
        """
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        return (
            df.lazy()
            .group_by(["date", "asset_id"])
            .agg([
                pl.cov("returns", "mkt_ret").alias("cov"),
                pl.var("mkt_ret").alias("mkt_var")
            ])
            .collect()
            .with_columns(
                (pl.col("cov") / pl.col("mkt_var")).fill_null(0).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


@register_operator(name="intra_realized_correlation", backend="polars")
class IntraRealizedCorrelationPolarsNative(PanelOperator):
    """Intraday realized correlation with benchmark."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        return (
            df.lazy()
            .group_by(["date", "asset_id"])
            .agg([
                pl.corr("returns", "mkt_ret").alias("value")
            ])
            .collect()
            .select(["date", "asset_id", "value"])
        )


@register_operator(name="intra_idiosyncratic_variance", backend="polars")
class IntraIdiosyncraticVariancePolarsNative(PanelOperator):
    """Idiosyncratic variance after removing market component."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        # Calculate beta and residual variance
        return (
            df.lazy()
            .group_by(["date", "asset_id"])
            .agg([
                pl.cov("returns", "mkt_ret").alias("cov"),
                pl.var("mkt_ret").alias("mkt_var"),
                pl.var("returns").alias("total_var")
            ])
            .collect()
            .with_columns(
                (pl.col("cov") / pl.col("mkt_var")).fill_null(0).alias("beta")
            )
            .with_columns(
                pl.max_horizontal(
                    pl.col("total_var") - (pl.col("beta") ** 2) * pl.col("mkt_var"),
                    pl.lit(0)
                ).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


@register_operator(name="intra_idiosyncratic_skewness", backend="polars")
class IntraIdiosyncraticSkewnessPolarsNative(PanelOperator):
    """Skewness of idiosyncratic returns."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        # Calculate beta
        beta_df = (
            df.lazy()
            .group_by(["date", "asset_id"])
            .agg([
                (pl.cov("returns", "mkt_ret") / pl.var("mkt_ret")).fill_null(0).alias("beta")
            ])
            .collect()
        )

        # Calculate residuals and their skewness
        return (
            df.join(beta_df, on=["date", "asset_id"])
            .with_columns(
                (pl.col("returns") - pl.col("beta") * pl.col("mkt_ret")).alias("resid")
            )
            .group_by(["date", "asset_id"])
            .agg([
                (pl.col("resid") ** 3).sum().alias("m3"),
                (pl.col("resid") ** 2).sum().alias("m2")
            ])
            .with_columns(
                (pl.col("m3") / (pl.col("m2") ** 1.5)).fill_null(0).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


@register_operator(name="intra_idiosyncratic_kurtosis", backend="polars")
class IntraIdiosyncraticKurtosisPolarsNative(PanelOperator):
    """Kurtosis of idiosyncratic returns."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        # Calculate beta
        beta_df = (
            df.lazy()
            .group_by(["date", "asset_id"])
            .agg([
                (pl.cov("returns", "mkt_ret") / pl.var("mkt_ret")).fill_null(0).alias("beta")
            ])
            .collect()
        )

        # Calculate residuals and their kurtosis
        return (
            df.join(beta_df, on=["date", "asset_id"])
            .with_columns(
                (pl.col("returns") - pl.col("beta") * pl.col("mkt_ret")).alias("resid")
            )
            .group_by(["date", "asset_id"])
            .agg([
                (pl.col("resid") ** 4).sum().alias("m4"),
                (pl.col("resid") ** 2).sum().alias("m2")
            ])
            .with_columns(
                (pl.col("m4") / (pl.col("m2") ** 2)).fill_null(0).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


@register_operator(name="intra_market_model_r2", backend="polars")
class IntraMarketModelR2PolarsNative(PanelOperator):
    """R-squared of market model within day."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        return (
            df.lazy()
            .group_by(["date", "asset_id"])
            .agg([
                pl.corr("returns", "mkt_ret").alias("corr")
            ])
            .collect()
            .with_columns(
                (pl.col("corr") ** 2).fill_null(0).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


# ============================================================================
# Semi-Beta Measures
# ============================================================================

@register_operator(name="intra_up_up_semibeta", backend="polars")
class IntraUpUpSemibetaPolarsNative(PanelOperator):
    """Beta computed only on periods when both stock and market are up."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        return (
            df.filter((pl.col("returns") > 0) & (pl.col("mkt_ret") > 0))
            .lazy()
            .group_by(["date", "asset_id"])
            .agg([
                pl.cov("returns", "mkt_ret").alias("cov"),
                pl.var("mkt_ret").alias("mkt_var")
            ])
            .collect()
            .with_columns(
                (pl.col("cov") / pl.col("mkt_var")).fill_null(0).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


@register_operator(name="intra_down_down_semibeta", backend="polars")
class IntraDownDownSemibetaPolarsNative(PanelOperator):
    """Beta computed only on periods when both stock and market are down."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        return (
            df.filter((pl.col("returns") < 0) & (pl.col("mkt_ret") < 0))
            .lazy()
            .group_by(["date", "asset_id"])
            .agg([
                pl.cov("returns", "mkt_ret").alias("cov"),
                pl.var("mkt_ret").alias("mkt_var")
            ])
            .collect()
            .with_columns(
                (pl.col("cov") / pl.col("mkt_var")).fill_null(0).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


@register_operator(name="intra_up_down_semibeta", backend="polars")
class IntraUpDownSemibetaPolarsNative(PanelOperator):
    """Beta when stock up but market down."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        return (
            df.filter((pl.col("returns") > 0) & (pl.col("mkt_ret") < 0))
            .lazy()
            .group_by(["date", "asset_id"])
            .agg([
                pl.cov("returns", "mkt_ret").alias("cov"),
                pl.var("mkt_ret").alias("mkt_var")
            ])
            .collect()
            .with_columns(
                (pl.col("cov") / pl.col("mkt_var")).fill_null(0).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


@register_operator(name="intra_down_up_semibeta", backend="polars")
class IntraDownUpSemibetaPolarsNative(PanelOperator):
    """Beta when stock down but market up."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        return (
            df.filter((pl.col("returns") < 0) & (pl.col("mkt_ret") > 0))
            .lazy()
            .group_by(["date", "asset_id"])
            .agg([
                pl.cov("returns", "mkt_ret").alias("cov"),
                pl.var("mkt_ret").alias("mkt_var")
            ])
            .collect()
            .with_columns(
                (pl.col("cov") / pl.col("mkt_var")).fill_null(0).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


@register_operator(name="intra_beta_asymmetry", backend="polars")
class IntraBetaAsymmetryPolarsNative(PanelOperator):
    """Difference between down-market beta and up-market beta."""

    def _calculate_panel(self, returns, benchmark_returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("asset_id"),
                benchmark_returns.alias("mkt_ret")
            ])
        )

        # Down market beta
        down_beta = (
            df.filter(pl.col("mkt_ret") < 0)
            .lazy()
            .group_by(["date", "asset_id"])
            .agg([
                (pl.cov("returns", "mkt_ret") / pl.var("mkt_ret")).fill_null(0).alias("beta_down")
            ])
            .collect()
        )

        # Up market beta
        up_beta = (
            df.filter(pl.col("mkt_ret") > 0)
            .lazy()
            .group_by(["date", "asset_id"])
            .agg([
                (pl.cov("returns", "mkt_ret") / pl.var("mkt_ret")).fill_null(0).alias("beta_up")
            ])
            .collect()
        )

        return (
            down_beta.join(up_beta, on=["date", "asset_id"], how="outer")
            .with_columns(
                (pl.col("beta_down").fill_null(0) - pl.col("beta_up").fill_null(0)).alias("value")
            )
            .select(["date", "asset_id", "value"])
        )


# ============================================================================
# Session Boundary Effects
# ============================================================================

@register_operator(name="intra_session_boundary_jump", backend="polars")
class IntraSessionBoundaryJumpPolarsNative(SeriesOperator):
    """Jump at market open (first return of the day)."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .with_columns(
                pl.col("returns").first().over("date").alias("open_ret")
            )
            .group_by("date")
            .agg([
                pl.col("open_ret").first().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_lunch_gap_return", backend="polars")
class IntraLunchGapReturnPolarsNative(SeriesOperator):
    """Return over lunch break (for A-share market with lunch break)."""

    def _calculate_series(self, returns, lunch_start_pct: float = 0.4, lunch_end_pct: float = 0.6, **kwargs):
        """
        Assumes lunch break is around 40%-60% of the day
        Return from last bar before lunch to first bar after lunch
        """
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
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

        return (
            df.join(daily_count, on="date")
            .with_columns(
                (pl.col("seq") / pl.col("total")).alias("pct")
            )
            .with_columns([
                pl.when(pl.col("pct") < lunch_start_pct)
                .then(pl.col("returns"))
                .otherwise(None)
                .last()
                .over("date")
                .alias("before_lunch_ret"),
                pl.when(pl.col("pct") >= lunch_end_pct)
                .then(pl.col("returns"))
                .otherwise(None)
                .first()
                .over("date")
                .alias("after_lunch_ret")
            ])
            .group_by("date")
            .agg([
                (pl.col("after_lunch_ret").first() - pl.col("before_lunch_ret").first()).alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_session_return_asymmetry", backend="polars")
class IntraSessionReturnAsymmetryPolarsNative(SeriesOperator):
    """Difference between morning and afternoon returns."""

    def _calculate_series(self, returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
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

        return (
            df.join(daily_count, on="date")
            .with_columns(
                (pl.col("seq") < pl.col("total") / 2).alias("is_morning")
            )
            .group_by("date")
            .agg([
                pl.when(pl.col("is_morning")).then(pl.col("returns")).sum().alias("morning_ret"),
                pl.when(~pl.col("is_morning")).then(pl.col("returns")).sum().alias("afternoon_ret")
            ])
            .with_columns(
                (pl.col("morning_ret") - pl.col("afternoon_ret")).alias("returns")
            )
            .select("returns")
            .to_series()
        )


# ============================================================================
# Drawdown Dynamics
# ============================================================================

@register_operator(name="intra_drawdown_duration", backend="polars")
class IntraDrawdownDurationPolarsNative(SeriesOperator):
    """Duration (in bars) of maximum drawdown period."""

    def _calculate_series(self, returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
        )

        return (
            df.lazy()
            .with_columns(
                pl.col("returns").cum_sum().over("date").alias("cumret")
            )
            .with_columns(
                pl.col("cumret").cum_max().over("date").alias("running_max")
            )
            .with_columns(
                (pl.col("running_max") - pl.col("cumret")).alias("drawdown")
            )
            .collect()
            .with_columns(
                (pl.col("drawdown") == pl.col("drawdown").max().over("date")).alias("is_max_dd")
            )
            .filter(pl.col("is_max_dd"))
            .group_by("date")
            .agg([
                (pl.col("seq").max() - pl.col("seq").min() + 1).alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_drawdown_depth", backend="polars")
class IntraDrawdownDepthPolarsNative(SeriesOperator):
    """Same as max_drawdown, but more explicit name."""

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .with_columns(
                pl.col("returns").cum_sum().over("date").alias("cumret")
            )
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


@register_operator(name="intra_drawdown_recovery_half_life", backend="polars")
class IntraDrawdownRecoveryHalfLifePolarsNative(SeriesOperator):
    """Time to recover half of maximum drawdown."""

    def _calculate_series(self, returns, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
            .with_columns(
                pl.col("returns").cum_sum().over("date").alias("cumret")
            )
            .with_columns(
                pl.col("cumret").cum_max().over("date").alias("running_max")
            )
            .with_columns(
                (pl.col("running_max") - pl.col("cumret")).alias("drawdown")
            )
        )

        # Find max drawdown point and subsequent recovery
        max_dd = (
            df.group_by("date")
            .agg([
                pl.col("drawdown").max().alias("max_dd"),
                pl.col("seq").filter(pl.col("drawdown") == pl.col("drawdown").max()).first().alias("dd_seq")
            ])
        )

        return (
            df.join(max_dd, on="date")
            .filter(pl.col("seq") >= pl.col("dd_seq"))
            .with_columns(
                (pl.col("drawdown") <= pl.col("max_dd") / 2).alias("recovered_half")
            )
            .filter(pl.col("recovered_half"))
            .group_by("date")
            .agg([
                (pl.col("seq").first() - pl.col("dd_seq").first()).alias("returns")
            ])
            .to_series()
        )


# ============================================================================
# Illiquidity and Market Impact
# ============================================================================

@register_operator(name="intra_kyle_lambda_proxy", backend="polars")
class IntraKyleLambdaProxyPolarsNative(SeriesOperator):
    """Kyle's lambda: price impact per unit volume."""

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
                pl.cov("returns", "volume").abs().alias("cov"),
                pl.var("volume").alias("vol_var")
            ])
            .collect()
            .with_columns(
                (pl.col("cov") / pl.col("vol_var")).fill_null(0).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_signed_imbalance_proxy", backend="polars")
class IntraSignedImbalanceProxyPolarsNative(SeriesOperator):
    """Correlation between signed returns and volume."""

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
                pl.corr("returns", "volume").alias("returns")
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# State and Regime Measures
# ============================================================================

@register_operator(name="intra_state_count", backend="polars")
class IntraStateCountPolarsNative(SeriesOperator):
    """Count of distinct states (discretized price levels) visited."""

    def _calculate_series(self, price, n_bins: int = 10, **kwargs):
        return (
            price.to_frame("price")
            .with_columns(pl.col("date"))
            .with_columns(
                pl.col("price").qcut(n_bins, labels=[str(i) for i in range(n_bins)], maintain_order=True).over("date").alias("bin")
            )
            .group_by("date")
            .agg([
                pl.col("bin").n_unique().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_state_transition_entropy", backend="polars")
class IntraStateTransitionEntropyPolarsNative(SeriesOperator):
    """Entropy of state transitions."""

    def _calculate_series(self, price, n_bins: int = 5, **kwargs):
        df = (
            price.to_frame("price")
            .with_columns(pl.col("date"))
            .with_columns(
                pl.col("price").qcut(n_bins, labels=[str(i) for i in range(n_bins)], maintain_order=True).over("date").alias("state")
            )
            .with_columns(
                pl.col("state").shift(1).over("date").alias("prev_state")
            )
            .filter(pl.col("prev_state").is_not_null())
        )

        # Count transitions
        transition_counts = (
            df.group_by(["date", "prev_state", "state"])
            .agg([
                pl.count().alias("count")
            ])
        )

        # Calculate entropy
        return (
            transition_counts
            .with_columns(
                pl.col("count").sum().over("date").alias("total")
            )
            .with_columns(
                (pl.col("count") / pl.col("total")).alias("prob")
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
# Jump Timing Measures
# ============================================================================

@register_operator(name="intra_jump_first_time", backend="polars")
class IntraJumpFirstTimePolarsNative(SeriesOperator):
    """Time (as fraction of day) of first significant jump."""

    def _calculate_series(self, returns, threshold: float = 3.0, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
        )

        stats = (
            df.lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std"),
                pl.col("seq").count().alias("total")
            ])
            .collect()
        )

        return (
            df.join(stats, on="date")
            .with_columns(
                (pl.col("returns").abs() > (threshold * pl.col("std") / pl.col("total").sqrt())).alias("is_jump")
            )
            .filter(pl.col("is_jump"))
            .group_by("date")
            .agg([
                (pl.col("seq").min() / pl.col("total").first()).alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_jump_last_time", backend="polars")
class IntraJumpLastTimePolarsNative(SeriesOperator):
    """Time (as fraction of day) of last significant jump."""

    def _calculate_series(self, returns, threshold: float = 3.0, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
        )

        stats = (
            df.lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std"),
                pl.col("seq").count().alias("total")
            ])
            .collect()
        )

        return (
            df.join(stats, on="date")
            .with_columns(
                (pl.col("returns").abs() > (threshold * pl.col("std") / pl.col("total").sqrt())).alias("is_jump")
            )
            .filter(pl.col("is_jump"))
            .group_by("date")
            .agg([
                (pl.col("seq").max() / pl.col("total").first()).alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_jump_clustering", backend="polars")
class IntraJumpClusteringPolarsNative(SeriesOperator):
    """Measure of temporal clustering of jumps (inverse of average spacing)."""

    def _calculate_series(self, returns, threshold: float = 3.0, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.arange(0, pl.count()).over("date").alias("seq")
            ])
        )

        stats = (
            df.lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std"),
                pl.col("seq").count().alias("total")
            ])
            .collect()
        )

        jump_df = (
            df.join(stats, on="date")
            .with_columns(
                (pl.col("returns").abs() > (threshold * pl.col("std") / pl.col("total").sqrt())).alias("is_jump")
            )
            .filter(pl.col("is_jump"))
            .with_columns(
                pl.col("seq").diff().over("date").alias("spacing")
            )
        )

        return (
            jump_df.group_by("date")
            .agg([
                pl.col("spacing").filter(pl.col("spacing").is_not_null()).mean().alias("avg_spacing")
            ])
            .with_columns(
                (1.0 / pl.col("avg_spacing")).fill_null(0).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_jump_concentration", backend="polars")
class IntraJumpConcentrationPolarsNative(SeriesOperator):
    """HHI of jump squared returns (concentration in few large jumps)."""

    def _calculate_series(self, returns, threshold: float = 3.0, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
        )

        stats = (
            df.lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std"),
                pl.col("returns").count().alias("n")
            ])
            .collect()
        )

        return (
            df.join(stats, on="date")
            .with_columns(
                pl.when(pl.col("returns").abs() > (threshold * pl.col("std") / pl.col("n").sqrt()))
                .then(pl.col("returns") ** 2)
                .otherwise(0)
                .alias("jump_var")
            )
            .with_columns(
                pl.col("jump_var").sum().over("date").alias("total_jump_var")
            )
            .with_columns(
                pl.when(pl.col("total_jump_var") > 0)
                .then((pl.col("jump_var") / pl.col("total_jump_var")) ** 2)
                .otherwise(0)
                .alias("share_sq")
            )
            .group_by("date")
            .agg([
                pl.col("share_sq").sum().alias("returns")
            ])
            .to_series()
        )


@register_operator(name="intra_signed_jump_ratio", backend="polars")
class IntraSignedJumpRatioPolarsNative(SeriesOperator):
    """Ratio of positive to negative jump variation."""

    def _calculate_series(self, returns, threshold: float = 3.0, **kwargs):
        df = (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
        )

        stats = (
            df.lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("std"),
                pl.col("returns").count().alias("n")
            ])
            .collect()
        )

        jump_stats = (
            df.join(stats, on="date")
            .with_columns(
                (pl.col("returns").abs() > (threshold * pl.col("std") / pl.col("n").sqrt())).alias("is_jump")
            )
            .filter(pl.col("is_jump"))
            .group_by("date")
            .agg([
                pl.when(pl.col("returns") > 0).then(pl.col("returns") ** 2).sum().alias("pos_var"),
                pl.when(pl.col("returns") < 0).then(pl.col("returns") ** 2).sum().alias("neg_var")
            ])
        )

        return (
            jump_stats
            .with_columns(
                (pl.col("pos_var") / (pl.col("neg_var") + 1e-10)).fill_null(1).alias("returns")
            )
            .select("returns")
            .to_series()
        )
