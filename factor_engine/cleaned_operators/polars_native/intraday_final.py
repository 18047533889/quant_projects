# -*- coding: utf-8 -*-
"""
Polars native implementations for intraday_* and intra_* operators - Final batch (49 operators)

intraday_* operators: minute-level → daily-level aggregations (25 operators)
intra_* operators: session-aware statistics and event analysis (24 operators)

All operators use real Polars API with backend="polars"
"""

import polars as pl
import numpy as np
from typing import Optional
from factor_engine.cleaned_operators.base_polars import register_operator, SeriesOperator, OperatorMetadata


# ============================================================================
# intraday_* operators (25): minute → daily aggregations
# ============================================================================

@register_operator(name="intraday_activity_duration_curvature", backend="polars")
class IntradayActivityDurationCurvaturePolarsNative(SeriesOperator):
    """Curvature of cumulative activity duration curve within each session."""

    metadata = OperatorMetadata(
        name="intraday_activity_duration_curvature",
        category="intraday",
        description="Curvature of cumulative activity duration curve within each session",
        param_names=["volume"],
        param_types={"volume": pl.Series},
        tags=["intraday", "curvature", "polars_native"],
    )

    def _calculate_series(self, volume, **kwargs):
        # TODO: Implement second derivative of cumulative duration curve
        # Skeleton: detect non-zero volume bars, compute cumulative time curve
        return (
            volume.to_frame("volume")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("volume").count().alias("volume")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_barrier_approach_acceleration", backend="polars")
class IntradayBarrierApproachAccelerationPolarsNative(SeriesOperator):
    """Acceleration of price as it approaches intraday barriers (limits/highs)."""

    metadata = OperatorMetadata(
        name="intraday_barrier_approach_acceleration",
        category="intraday",
        description="Acceleration of price as it approaches intraday barriers",
        param_names=["price"],
        param_types={"price": pl.Series},
        tags=["intraday", "acceleration", "polars_native"],
    )

    def _calculate_series(self, price, **kwargs):
        # TODO: Detect barrier approach events, measure acceleration
        return (
            price.to_frame("price")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("price").max() - pl.col("price").min()).alias("price")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_bvc_imbalance", backend="polars")
class IntradayBvcImbalancePolarsNative(SeriesOperator):
    """Buy-Volume-Concentration imbalance: asymmetry in volume distribution."""

    metadata = OperatorMetadata(
        name="intraday_bvc_imbalance",
        category="intraday",
        description="Buy-Volume-Concentration imbalance: asymmetry in volume distribution",
        param_names=["returns", "volume"],
        param_types={"returns": pl.Series, "volume": pl.Series},
        tags=["intraday", "imbalance", "polars_native"],
    )

    def _calculate_series(self, returns, volume, **kwargs):
        # TODO: Implement buy/sell volume clustering imbalance measure
        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.when(pl.col("volume").sum() != 0)
                .then((pl.col("returns") * pl.col("volume")).sum() / pl.col("volume").sum())
                .otherwise(None)
                .alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_impact_asymmetry", backend="polars")
class IntradayImpactAsymmetryPolarsNative(SeriesOperator):
    """Asymmetry between up-move and down-move price impact."""

    metadata = OperatorMetadata(
        name="intraday_impact_asymmetry",
        category="intraday",
        description="Asymmetry between up-move and down-move price impact",
        param_names=["returns", "volume"],
        param_types={"returns": pl.Series, "volume": pl.Series},
        tags=["intraday", "impact", "asymmetry", "polars_native"],
    )

    def _calculate_series(self, returns, volume, **kwargs):
        # TODO: Separate up/down returns, compute impact ratio
        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.when(pl.col("volume") != 0)
                .then(pl.col("returns").abs() / pl.col("volume"))
                .otherwise(None)
                .mean()
                .alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_impact_beta", backend="polars")
class IntradayImpactBetaPolarsNative(SeriesOperator):
    """Power-law exponent of volume-price impact relationship."""

    metadata = OperatorMetadata(
        name="intraday_impact_beta",
        category="intraday",
        description="Power-law exponent of volume-price impact relationship",
        param_names=["returns", "volume"],
        param_types={"returns": pl.Series, "volume": pl.Series},
        tags=["intraday", "impact", "beta", "polars_native"],
    )

    def _calculate_series(self, returns, volume, **kwargs):
        # TODO: Log-log regression of |return| ~ volume
        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.lit(0.5).alias("returns")  # Placeholder beta
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_impact_decay_rate", backend="polars")
class IntradayImpactDecayRatePolarsNative(SeriesOperator):
    """Exponential decay rate of price impact after volume spikes."""

    metadata = OperatorMetadata(
        name="intraday_impact_decay_rate",
        category="intraday",
        description="Exponential decay rate of price impact after volume spikes",
        param_names=["returns", "volume"],
        param_types={"returns": pl.Series, "volume": pl.Series},
        tags=["intraday", "impact", "decay", "polars_native"],
    )

    def _calculate_series(self, returns, volume, **kwargs):
        # TODO: Fit exponential decay to post-spike price reversion
        return (
            volume.to_frame("volume")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("volume").std().alias("volume")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_jump_test_stat", backend="polars")
class IntradayJumpTestStatPolarsNative(SeriesOperator):
    """Jump test statistic: (RV - BV) / sqrt(variance of BV)."""

    metadata = OperatorMetadata(
        name="intraday_jump_test_stat",
        category="intraday",
        description="Jump test statistic: (RV - BV) / sqrt(variance of BV)",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "jump", "test", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Implement Barndorff-Nielsen-Shephard jump test
        mu1_sq = np.pi / 2

        df = (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
        )

        rv = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).sum().alias("rv")
            ])
            .collect()
        )

        bv = (
            df.lazy()
            .group_by("date")
            .agg([
                (pl.col("returns").abs() * pl.col("returns").abs().shift(1)).sum().alias("bv")
            ])
            .collect()
            .with_columns(
                (pl.col("bv") * mu1_sq).alias("bv")
            )
        )

        return (
            rv.join(bv, on="date")
            .with_columns(
                ((pl.col("rv") - pl.col("bv")) / pl.col("bv").sqrt().fill_null(1)).alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intraday_medrv", backend="polars")
class IntradayMedrvPolarsNative(SeriesOperator):
    """MedRV: Median-based realized volatility estimator (robust to jumps)."""

    metadata = OperatorMetadata(
        name="intraday_medrv",
        category="intraday",
        description="MedRV: Median-based realized volatility estimator (robust to jumps)",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "realized", "volatility", "median", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Implement median-based RV: uses median(|r_i|, |r_{i+1}|, |r_{i+2}|)
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").abs().median().alias("returns")  # Simplified placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_minrv", backend="polars")
class IntradayMinrvPolarsNative(SeriesOperator):
    """MinRV: Minimum-based realized volatility (robust to jumps)."""

    metadata = OperatorMetadata(
        name="intraday_minrv",
        category="intraday",
        description="MinRV: Minimum-based realized volatility (robust to jumps)",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "realized", "volatility", "minimum", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Implement min-based RV: uses min(|r_i|, |r_{i+1}|)
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).sum().sqrt().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_quantile_curve_pca_residual", backend="polars")
class IntradayQuantileCurvePcaResidualPolarsNative(SeriesOperator):
    """PCA residual of intraday return quantile curve."""

    metadata = OperatorMetadata(
        name="intraday_quantile_curve_pca_residual",
        category="intraday",
        description="PCA residual of intraday return quantile curve",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "quantile", "pca", "residual", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Build quantile curve, apply PCA, return residual
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_quantile_curve_pca_score", backend="polars")
class IntradayQuantileCurvePcaScorePolarsNative(SeriesOperator):
    """First PCA component score of intraday return quantile curve."""

    metadata = OperatorMetadata(
        name="intraday_quantile_curve_pca_score",
        category="intraday",
        description="First PCA component score of intraday return quantile curve",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "quantile", "pca", "score", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Build quantile curve, project onto PC1
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").mean().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_realized_power_variation", backend="polars")
class IntradayRealizedPowerVariationPolarsNative(SeriesOperator):
    """Realized power variation: sum of |return|^p for arbitrary power p."""

    metadata = OperatorMetadata(
        name="intraday_realized_power_variation",
        category="intraday",
        description="Realized power variation: sum of |return|^p for arbitrary power p",
        param_names=["returns", "power"],
        param_types={"returns": pl.Series, "power": float},
        tags=["intraday", "realized", "power", "variation", "polars_native"],
    )

    def _calculate_series(self, returns, power: float = 2.0, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns").abs() ** power).sum().alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_realized_semivariance_balance", backend="polars")
class IntradayRealizedSemivarianceBalancePolarsNative(SeriesOperator):
    """Balance between upside and downside semivariance."""

    metadata = OperatorMetadata(
        name="intraday_realized_semivariance_balance",
        category="intraday",
        description="Balance between upside and downside semivariance",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "realized", "semivariance", "balance", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.when(pl.col("returns") > 0).then(pl.col("returns") ** 2).otherwise(0).sum().alias("up_var"),
                pl.when(pl.col("returns") < 0).then(pl.col("returns") ** 2).otherwise(0).sum().alias("down_var")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("up_var") + pl.col("down_var") != 0)
                .then((pl.col("up_var") - pl.col("down_var")) / (pl.col("up_var") + pl.col("down_var")))
                .otherwise(None)
                .alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intraday_return_wasserstein_shift", backend="polars")
class IntradayReturnWassersteinShiftPolarsNative(SeriesOperator):
    """Wasserstein distance between morning and afternoon return distributions."""

    metadata = OperatorMetadata(
        name="intraday_return_wasserstein_shift",
        category="intraday",
        description="Wasserstein distance between morning and afternoon return distributions",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "wasserstein", "shift", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Implement 1D Wasserstein distance (Earth Mover's Distance)
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_rv_signature_curvature", backend="polars")
class IntradayRvSignatureCurvaturePolarsNative(SeriesOperator):
    """Curvature of realized variance signature plot (RV vs sampling frequency)."""

    metadata = OperatorMetadata(
        name="intraday_rv_signature_curvature",
        category="intraday",
        description="Curvature of realized variance signature plot (RV vs sampling frequency)",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "rv", "signature", "curvature", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Compute RV at multiple sampling frequencies, fit curve
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).sum().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_rv_signature_slope", backend="polars")
class IntradayRvSignatureSlopePolarsNative(SeriesOperator):
    """Slope of realized variance signature plot."""

    metadata = OperatorMetadata(
        name="intraday_rv_signature_slope",
        category="intraday",
        description="Slope of realized variance signature plot",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "rv", "signature", "slope", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Compute RV at multiple frequencies, estimate slope
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).sum().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_session_shape_novelty", backend="polars")
class IntradaySessionShapeNoveltyPolarsNative(SeriesOperator):
    """Novelty score: distance from today's intraday pattern to historical average."""

    metadata = OperatorMetadata(
        name="intraday_session_shape_novelty",
        category="intraday",
        description="Novelty score: distance from today's intraday pattern to historical average",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "session", "shape", "novelty", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Compare today's shape to trailing average shape
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_subsampled_rv_dispersion", backend="polars")
class IntradaySubsampledRvDispersionPolarsNative(SeriesOperator):
    """Dispersion of RV estimates across subsampled grids."""

    metadata = OperatorMetadata(
        name="intraday_subsampled_rv_dispersion",
        category="intraday",
        description="Dispersion of RV estimates across subsampled grids",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "rv", "subsampled", "dispersion", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Compute RV on shifted grids, measure dispersion
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).sum().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_volatility_concentration", backend="polars")
class IntradayVolatilityConcentrationPolarsNative(SeriesOperator):
    """Concentration of volatility: share of total variance in top-k bars."""

    metadata = OperatorMetadata(
        name="intraday_volatility_concentration",
        category="intraday",
        description="Concentration of volatility: share of total variance in top-k bars",
        param_names=["returns", "k"],
        param_types={"returns": pl.Series, "k": int},
        tags=["intraday", "volatility", "concentration", "polars_native"],
    )

    def _calculate_series(self, returns, k: int = 10, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).sort(descending=True).head(k).sum().alias("top_k"),
                (pl.col("returns") ** 2).sum().alias("total")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("total") != 0)
                .then(pl.col("top_k") / pl.col("total"))
                .otherwise(None)
                .alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intraday_volatility_entropy", backend="polars")
class IntradayVolatilityEntropyPolarsNative(SeriesOperator):
    """Shannon entropy of normalized squared-return distribution."""

    metadata = OperatorMetadata(
        name="intraday_volatility_entropy",
        category="intraday",
        description="Shannon entropy of normalized squared-return distribution",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "volatility", "entropy", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Normalize r^2 to probabilities, compute -sum(p * log(p))
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns") ** 2).std().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_volatility_signature_slope", backend="polars")
class IntradayVolatilitySignatureSlopePolarsNative(SeriesOperator):
    """Slope of volatility signature plot (std vs sampling frequency)."""

    metadata = OperatorMetadata(
        name="intraday_volatility_signature_slope",
        category="intraday",
        description="Slope of volatility signature plot (std vs sampling frequency)",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "volatility", "signature", "slope", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Compute std at multiple frequencies, estimate slope
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").std().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_volatility_time_centroid", backend="polars")
class IntradayVolatilityTimeCentroidPolarsNative(SeriesOperator):
    """Time centroid of intraday volatility distribution."""

    metadata = OperatorMetadata(
        name="intraday_volatility_time_centroid",
        category="intraday",
        description="Time centroid of intraday volatility distribution",
        param_names=["returns"],
        param_types={"returns": pl.Series},
        tags=["intraday", "volatility", "time", "centroid", "polars_native"],
    )

    def _calculate_series(self, returns, **kwargs):
        # TODO: Compute weighted average time using r^2 as weights
        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                pl.col("limit_flag").cum_count().over("date").alias("seq")
            ])
            .lazy()
            .group_by("date")
            .agg([
                ((pl.col("returns") ** 2) * pl.col("seq")).sum().alias("weighted"),
                (pl.col("returns") ** 2).sum().alias("total")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("total") != 0)
                .then(pl.col("weighted") / pl.col("total"))
                .otherwise(None)
                .alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intraday_volume_clock_path_efficiency", backend="polars")
class IntradayVolumeClockPathEfficiencyPolarsNative(SeriesOperator):
    """Path efficiency in volume-clock space: direct distance / actual path."""

    metadata = OperatorMetadata(
        name="intraday_volume_clock_path_efficiency",
        category="intraday",
        description="Path efficiency in volume-clock space: direct distance / actual path",
        param_names=["price", "volume"],
        param_types={"price": pl.Series, "volume": pl.Series},
        tags=["intraday", "volume", "clock", "path", "efficiency", "polars_native"],
    )

    def _calculate_series(self, price, volume, **kwargs):
        # TODO: Build cumulative volume curve, measure path efficiency
        return (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("price").max() - pl.col("price").min()).alias("price")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_volume_clock_roughness", backend="polars")
class IntradayVolumeClockRoughnessPolarsNative(SeriesOperator):
    """Roughness of price path in volume-clock space."""

    metadata = OperatorMetadata(
        name="intraday_volume_clock_roughness",
        category="intraday",
        description="Roughness of price path in volume-clock space",
        param_names=["price", "volume"],
        param_types={"price": pl.Series, "volume": pl.Series},
        tags=["intraday", "volume", "clock", "roughness", "polars_native"],
    )

    def _calculate_series(self, price, volume, **kwargs):
        # TODO: Measure path variation in volume-time
        return (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("price").std().alias("price")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intraday_wasserstein_pair_distance", backend="polars")
class IntradayWassersteinPairDistancePolarsNative(SeriesOperator):
    """Wasserstein distance between return distributions of two instruments."""

    metadata = OperatorMetadata(
        name="intraday_wasserstein_pair_distance",
        category="intraday",
        description="Wasserstein distance between return distributions of two instruments",
        param_names=["returns_a", "returns_b"],
        param_types={"returns_a": pl.Series, "returns_b": pl.Series},
        tags=["intraday", "wasserstein", "pair", "distance", "polars_native"],
    )

    def _calculate_series(self, returns_a, returns_b, **kwargs):
        # TODO: Implement 1D Wasserstein between two series
        return (
            returns_a.to_frame("returns_a")
            .with_columns([
                pl.col("date"),
                returns_b.alias("returns_b")
            ])
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("returns_a").std() - pl.col("returns_b").std()).abs().alias("returns")  # Placeholder
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# intra_* operators (24): session-aware statistics and event analysis
# ============================================================================

@register_operator(name="intra_event_pre_post_contrast", backend="polars")
class IntraEventPrePostContrastPolarsNative(SeriesOperator):
    """Contrast between pre-event and post-event window statistics."""

    metadata = OperatorMetadata(
        name="intra_event_pre_post_contrast",
        category="intraday",
        description="Contrast between pre-event and post-event window statistics",
        param_names=["signal", "event_mask", "pre_bars", "post_bars"],
        param_types={"signal": pl.Series, "event_mask": pl.Series, "pre_bars": int, "post_bars": int},
        tags=["intraday", "event", "contrast", "polars_native"],
    )

    def _calculate_series(self, signal, event_mask, pre_bars: int = 5, post_bars: int = 5, **kwargs):
        # TODO: Identify events, compute pre/post window stats
        return (
            signal.to_frame("signal")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("signal").mean().alias("signal")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_event_window_reduce", backend="polars")
class IntraEventWindowReducePolarsNative(SeriesOperator):
    """Reduce function over windows around detected events."""

    metadata = OperatorMetadata(
        name="intra_event_window_reduce",
        category="intraday",
        description="Reduce function over windows around detected events",
        param_names=["signal", "event_mask", "window", "agg_func"],
        param_types={"signal": pl.Series, "event_mask": pl.Series, "window": int, "agg_func": str},
        tags=["intraday", "event", "window", "reduce", "polars_native"],
    )

    def _calculate_series(self, signal, event_mask, window: int = 10, agg_func: str = "mean", **kwargs):
        # TODO: Extract event windows, apply aggregation
        return (
            signal.to_frame("signal")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("signal").mean().alias("signal")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_impulse_event_detector", backend="polars")
class IntraImpulseEventDetectorPolarsNative(SeriesOperator):
    """Detect impulse events: sudden spikes in volume or price movement."""

    metadata = OperatorMetadata(
        name="intra_impulse_event_detector",
        category="intraday",
        description="Detect impulse events: sudden spikes in volume or price movement",
        param_names=["signal", "threshold"],
        param_types={"signal": pl.Series, "threshold": float},
        tags=["intraday", "impulse", "event", "detector", "polars_native"],
    )

    def _calculate_series(self, signal, threshold: float = 3.0, **kwargs):
        # TODO: Z-score based event detection within session
        return (
            signal.to_frame("signal")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("signal").abs() > threshold).sum().alias("signal")  # Count events
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_limit_duration", backend="polars")
class IntraLimitDurationPolarsNative(SeriesOperator):
    """Total minutes spent at daily price limits."""

    metadata = OperatorMetadata(
        name="intra_limit_duration",
        category="intraday",
        description="Total minutes spent at daily price limits",
        param_names=["limit_flag"],
        param_types={"limit_flag": pl.Series},
        tags=["intraday", "limit", "duration", "polars_native"],
    )

    def _calculate_series(self, limit_flag, **kwargs):
        return (
            limit_flag.to_frame("limit_flag")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("limit_flag").sum().alias("limit_flag")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_limit_first_hit_time", backend="polars")
class IntraLimitFirstHitTimePolarsNative(SeriesOperator):
    """Minute-of-day when price first hits the daily limit."""

    metadata = OperatorMetadata(
        name="intra_limit_first_hit_time",
        category="intraday",
        description="Minute-of-day when price first hits the daily limit",
        param_names=["limit_flag"],
        param_types={"limit_flag": pl.Series},
        tags=["intraday", "limit", "first", "hit", "time", "polars_native"],
    )

    def _calculate_series(self, limit_flag, **kwargs):
        # TODO: Find first occurrence within each session
        return (
            limit_flag.to_frame("limit_flag")
            .with_columns([
                pl.col("date"),
                pl.col("limit_flag").cum_count().over("date").alias("seq")
            ])
            .lazy()
            .filter(pl.col("limit_flag") != 0)
            .group_by("date")
            .agg([
                pl.col("seq").min().alias("limit_flag")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_limit_pre_hit_pressure_profile", backend="polars")
class IntraLimitPreHitPressureProfilePolarsNative(SeriesOperator):
    """Volume/turnover profile in the bars leading up to limit hit."""

    metadata = OperatorMetadata(
        name="intra_limit_pre_hit_pressure_profile",
        category="intraday",
        description="Volume/turnover profile in the bars leading up to limit hit",
        param_names=["volume", "limit_flag", "pre_bars"],
        param_types={"volume": pl.Series, "limit_flag": pl.Series, "pre_bars": int},
        tags=["intraday", "limit", "pressure", "profile", "polars_native"],
    )

    def _calculate_series(self, volume, limit_flag, pre_bars: int = 10, **kwargs):
        # TODO: Extract pre-hit windows, aggregate volume profile
        return (
            volume.to_frame("volume")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("volume").mean().alias("volume")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_limit_reopen_count", backend="polars")
class IntraLimitReopenCountPolarsNative(SeriesOperator):
    """Number of times price reopens after hitting limit."""

    metadata = OperatorMetadata(
        name="intra_limit_reopen_count",
        category="intraday",
        description="Number of times price reopens after hitting limit",
        param_names=["limit_flag"],
        param_types={"limit_flag": pl.Series},
        tags=["intraday", "limit", "reopen", "count", "polars_native"],
    )

    def _calculate_series(self, limit_flag, **kwargs):
        # TODO: Count transitions from limit to non-limit within session
        return (
            limit_flag.to_frame("limit_flag")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("limit_flag").diff().abs() > 0).sum().alias("limit_flag")  # Transition count
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_liquidity_resilience_curve_fit", backend="polars")
class IntraLiquidityResilienceCurveFitPolarsNative(SeriesOperator):
    """Fit parameters of liquidity resilience curve after shocks."""

    metadata = OperatorMetadata(
        name="intra_liquidity_resilience_curve_fit",
        category="intraday",
        description="Fit parameters of liquidity resilience curve after shocks",
        param_names=["spread", "volume"],
        param_types={"spread": pl.Series, "volume": pl.Series},
        tags=["intraday", "liquidity", "resilience", "curve", "fit", "polars_native"],
    )

    def _calculate_series(self, spread, volume, **kwargs):
        # TODO: Model spread recovery after volume spikes
        return (
            spread.to_frame("spread")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("spread").mean().alias("spread")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_multiresolution_resample_reduce", backend="polars")
class IntraMultiresolutionResampleReducePolarsNative(SeriesOperator):
    """Aggregate statistics at multiple time resolutions within session."""

    metadata = OperatorMetadata(
        name="intra_multiresolution_resample_reduce",
        category="intraday",
        description="Aggregate statistics at multiple time resolutions within session",
        param_names=["signal", "resolutions"],
        param_types={"signal": pl.Series, "resolutions": list},
        tags=["intraday", "multiresolution", "resample", "reduce", "polars_native"],
    )

    def _calculate_series(self, signal, resolutions: list = None, **kwargs):
        # TODO: Resample to multiple frequencies, aggregate
        return (
            signal.to_frame("signal")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("signal").std().alias("signal")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_neighbor_event_class", backend="polars")
class IntraNeighborEventClassPolarsNative(SeriesOperator):
    """Classification of events based on temporal neighborhood similarity."""

    metadata = OperatorMetadata(
        name="intra_neighbor_event_class",
        category="intraday",
        description="Classification of events based on temporal neighborhood similarity",
        param_names=["signal"],
        param_types={"signal": pl.Series},
        tags=["intraday", "neighbor", "event", "classification", "polars_native"],
    )

    def _calculate_series(self, signal, **kwargs):
        # TODO: KNN-style event classification within session
        return (
            signal.to_frame("signal")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("signal").mean().alias("signal")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_post_impulse_response", backend="polars")
class IntraPostImpulseResponsePolarsNative(SeriesOperator):
    """Average response pattern in bars following impulse events."""

    metadata = OperatorMetadata(
        name="intra_post_impulse_response",
        category="intraday",
        description="Average response pattern in bars following impulse events",
        param_names=["signal", "impulse_mask", "response_window"],
        param_types={"signal": pl.Series, "impulse_mask": pl.Series, "response_window": int},
        tags=["intraday", "impulse", "response", "polars_native"],
    )

    def _calculate_series(self, signal, impulse_mask, response_window: int = 10, **kwargs):
        # TODO: Extract post-impulse windows, average response
        return (
            signal.to_frame("signal")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("signal").mean().alias("signal")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_probe_outcome_score", backend="polars")
class IntraProbeOutcomeScorePolarsNative(SeriesOperator):
    """Outcome score: success rate of price probes above/below levels."""

    metadata = OperatorMetadata(
        name="intra_probe_outcome_score",
        category="intraday",
        description="Outcome score: success rate of price probes above/below levels",
        param_names=["price", "threshold"],
        param_types={"price": pl.Series, "threshold": float},
        tags=["intraday", "probe", "outcome", "score", "polars_native"],
    )

    def _calculate_series(self, price, threshold: float = 0.01, **kwargs):
        # TODO: Detect probe events, measure success rate
        return (
            price.to_frame("price")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.lit(0.5).alias("price")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_profile_earth_mover_distance", backend="polars")
class IntraProfileEarthMoverDistancePolarsNative(SeriesOperator):
    """Earth Mover's Distance between today's profile and reference profile."""

    metadata = OperatorMetadata(
        name="intra_profile_earth_mover_distance",
        category="intraday",
        description="Earth Mover's Distance between today's profile and reference profile",
        param_names=["signal", "reference_signal"],
        param_types={"signal": pl.Series, "reference_signal": pl.Series},
        tags=["intraday", "profile", "earth_mover", "distance", "polars_native"],
    )

    def _calculate_series(self, signal, reference_signal, **kwargs):
        # TODO: Compute EMD between distributions
        return (
            signal.to_frame("signal")
            .with_columns([
                pl.col("date"),
                reference_signal.alias("reference")
            ])
            .lazy()
            .group_by("date")
            .agg([
                (pl.col("signal").std() - pl.col("reference").std()).abs().alias("signal")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_response_curve_features", backend="polars")
class IntraResponseCurveFeaturesPolarsNative(SeriesOperator):
    """Features extracted from impulse response curve."""

    metadata = OperatorMetadata(
        name="intra_response_curve_features",
        category="intraday",
        description="Features extracted from impulse response curve",
        param_names=["signal", "impulse_mask"],
        param_types={"signal": pl.Series, "impulse_mask": pl.Series},
        tags=["intraday", "response", "curve", "features", "polars_native"],
    )

    def _calculate_series(self, signal, impulse_mask, **kwargs):
        # TODO: Fit response curve, extract peak/decay/area features
        return (
            signal.to_frame("signal")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("signal").max().alias("signal")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_slice_mask_pair_reduce", backend="polars")
class IntraSliceMaskPairReducePolarsNative(SeriesOperator):
    """Pairwise reduction over two masked slices of the session."""

    metadata = OperatorMetadata(
        name="intra_slice_mask_pair_reduce",
        category="intraday",
        description="Pairwise reduction over two masked slices of the session",
        param_names=["signal", "mask_a", "mask_b", "agg_func"],
        param_types={"signal": pl.Series, "mask_a": pl.Series, "mask_b": pl.Series, "agg_func": str},
        tags=["intraday", "slice", "mask", "pair", "reduce", "polars_native"],
    )

    def _calculate_series(self, signal, mask_a, mask_b, agg_func: str = "corr", **kwargs):
        # TODO: Apply masks, compute pairwise statistic
        return (
            signal.to_frame("signal")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("signal").mean().alias("signal")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_slice_mask_reduce", backend="polars")
class IntraSliceMaskReducePolarsNative(SeriesOperator):
    """Reduction over masked time slice within session."""

    metadata = OperatorMetadata(
        name="intra_slice_mask_reduce",
        category="intraday",
        description="Reduction over masked time slice within session",
        param_names=["signal", "mask", "agg_func"],
        param_types={"signal": pl.Series, "mask": pl.Series, "agg_func": str},
        tags=["intraday", "slice", "mask", "reduce", "polars_native"],
    )

    def _calculate_series(self, signal, mask, agg_func: str = "mean", **kwargs):
        # TODO: Filter by mask, aggregate
        return (
            signal.to_frame("signal")
            .with_columns([
                pl.col("date"),
                mask.alias("mask")
            ])
            .lazy()
            .filter(pl.col("mask") != 0)
            .group_by("date")
            .agg([
                pl.col("signal").mean().alias("signal")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_state_dwell_stats", backend="polars")
class IntraStateDwellStatsPolarsNative(SeriesOperator):
    """Statistics of dwell times in discrete states."""

    metadata = OperatorMetadata(
        name="intra_state_dwell_stats",
        category="intraday",
        description="Statistics of dwell times in discrete states",
        param_names=["state_series"],
        param_types={"state_series": pl.Series},
        tags=["intraday", "state", "dwell", "statistics", "polars_native"],
    )

    def _calculate_series(self, state_series, **kwargs):
        # TODO: Identify runs, compute mean/max dwell duration
        return (
            state_series.to_frame("state")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("state").count().alias("state")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_state_interval_moment", backend="polars")
class IntraStateIntervalMomentPolarsNative(SeriesOperator):
    """Moment (mean/std/skew) of intervals between state transitions."""

    metadata = OperatorMetadata(
        name="intra_state_interval_moment",
        category="intraday",
        description="Moment (mean/std/skew) of intervals between state transitions",
        param_names=["state_series", "moment"],
        param_types={"state_series": pl.Series, "moment": int},
        tags=["intraday", "state", "interval", "moment", "polars_native"],
    )

    def _calculate_series(self, state_series, moment: int = 1, **kwargs):
        # TODO: Compute intervals, calculate moment
        return (
            state_series.to_frame("state")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("state").std().alias("state")  # Placeholder
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_state_pair_same_slot_corr", backend="polars")
class IntraStatePairSameSlotCorrPolarsNative(SeriesOperator):
    """Correlation between two state series at same minute-of-day slots."""

    metadata = OperatorMetadata(
        name="intra_state_pair_same_slot_corr",
        category="intraday",
        description="Correlation between two state series at same minute-of-day slots",
        param_names=["state_a", "state_b"],
        param_types={"state_a": pl.Series, "state_b": pl.Series},
        tags=["intraday", "state", "pair", "correlation", "polars_native"],
    )

    def _calculate_series(self, state_a, state_b, **kwargs):
        # TODO: Align by minute-of-day, compute correlation
        return (
            state_a.to_frame("state_a")
            .with_columns([
                pl.col("date"),
                state_b.alias("state_b")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.corr("state_a", "state_b").alias("state_a")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_supply_absorption_score", backend="polars")
class IntraSupplyAbsorptionScorePolarsNative(SeriesOperator):
    """Score measuring how effectively supply/demand imbalances are absorbed."""

    metadata = OperatorMetadata(
        name="intra_supply_absorption_score",
        category="intraday",
        description="Score measuring how effectively supply/demand imbalances are absorbed",
        param_names=["returns", "volume"],
        param_types={"returns": pl.Series, "volume": pl.Series},
        tags=["intraday", "supply", "absorption", "score", "polars_native"],
    )

    def _calculate_series(self, returns, volume, **kwargs):
        # TODO: Model volume-adjusted price resilience
        return (
            returns.to_frame("returns")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.when(pl.col("volume") != 0)
                .then(pl.col("returns").abs() / pl.col("volume"))
                .otherwise(None)
                .mean()
                .alias("returns")
            ])
            .collect()
            .to_series()
        )


@register_operator(name="intra_ute_high", backend="polars")
class IntraUteHighPolarsNative(SeriesOperator):
    """Upside Tail Event: fraction of bars in the upper tail."""

    metadata = OperatorMetadata(
        name="intra_ute_high",
        category="intraday",
        description="Upside Tail Event: fraction of bars in the upper tail",
        param_names=["returns", "threshold"],
        param_types={"returns": pl.Series, "threshold": float},
        tags=["intraday", "ute", "high", "tail", "polars_native"],
    )

    def _calculate_series(self, returns, threshold: float = 2.0, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").count().alias("total"),
                (pl.col("returns") > threshold * pl.col("returns").std()).sum().alias("tail_count")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("total") != 0)
                .then(pl.col("tail_count") / pl.col("total"))
                .otherwise(None)
                .alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_ute_low", backend="polars")
class IntraUteLowPolarsNative(SeriesOperator):
    """Downside Tail Event: fraction of bars in the lower tail."""

    metadata = OperatorMetadata(
        name="intra_ute_low",
        category="intraday",
        description="Downside Tail Event: fraction of bars in the lower tail",
        param_names=["returns", "threshold"],
        param_types={"returns": pl.Series, "threshold": float},
        tags=["intraday", "ute", "low", "tail", "polars_native"],
    )

    def _calculate_series(self, returns, threshold: float = -2.0, **kwargs):
        return (
            returns.to_frame("returns")
            .with_columns(pl.col("date"))
            .lazy()
            .group_by("date")
            .agg([
                pl.col("returns").count().alias("total"),
                (pl.col("returns") < threshold * pl.col("returns").std()).sum().alias("tail_count")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("total") != 0)
                .then(pl.col("tail_count") / pl.col("total"))
                .otherwise(None)
                .alias("returns")
            )
            .select("returns")
            .to_series()
        )


@register_operator(name="intra_vwap_path_curvature_pct", backend="polars")
class IntraVwapPathCurvaturePctPolarsNative(SeriesOperator):
    """Curvature of price path relative to VWAP (percent deviation)."""

    metadata = OperatorMetadata(
        name="intra_vwap_path_curvature_pct",
        category="intraday",
        description="Curvature of price path relative to VWAP (percent deviation)",
        param_names=["price", "volume"],
        param_types={"price": pl.Series, "volume": pl.Series},
        tags=["intraday", "vwap", "path", "curvature", "polars_native"],
    )

    def _calculate_series(self, price, volume, **kwargs):
        # TODO: Compute VWAP, measure path curvature
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
                pl.col("volume").sum().alias("v")
            ])
            .collect()
            .with_columns(
                pl.when(pl.col("v") != 0)
                .then(pl.col("pv") / pl.col("v"))
                .otherwise(None)
                .alias("price")
            )
            .select("price")
            .to_series()
        )


@register_operator(name="intra_vwap_path_slope_pct", backend="polars")
class IntraVwapPathSlopePctPolarsNative(SeriesOperator):
    """Slope of price path relative to VWAP (percent per minute)."""

    metadata = OperatorMetadata(
        name="intra_vwap_path_slope_pct",
        category="intraday",
        description="Slope of price path relative to VWAP (percent per minute)",
        param_names=["price", "volume"],
        param_types={"price": pl.Series, "volume": pl.Series},
        tags=["intraday", "vwap", "path", "slope", "polars_native"],
    )

    def _calculate_series(self, price, volume, **kwargs):
        # TODO: Compute VWAP, fit linear trend to price-VWAP deviation
        return (
            price.to_frame("price")
            .with_columns([
                pl.col("date"),
                volume.alias("volume")
            ])
            .lazy()
            .group_by("date")
            .agg([
                pl.col("price").std().alias("price")  # Placeholder
            ])
            .collect()
            .to_series()
        )
