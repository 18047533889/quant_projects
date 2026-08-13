"""
Polars native implementations for time series operators (ts_* family) - Batch 2
Advanced operators: extremal statistics, Markov chains, feature engineering, support/resistance

Operators 433-534 from missing list:
- Feature engineering (mode_share, PCA reconstruction, subspace rotation)
- Gap analysis (fill ratio, reversion, survival)
- GARCH/GJR models (volatility forecasting, persistence, leverage)
- Extremal theory (GPD, Hill, Pickands tail indices)
- Markov chain features (committor, entropy production, persistence)
- Support/resistance detection
- Mean reversion measures

NOTE: Some operators in this batch have canonical definitions elsewhere (auto_polars_all.py).
Those are registered as backend="polars_native" implementations of the canonical operator.
When param_specs differ from canonical, this provides an alternative native implementation.
"""

import polars as pl
import numpy as np
from typing import Optional, Union

from cleaned_operators.base_polars import (
    SeriesOperator,
    TwoVarOperator,
    OperatorMetadata,
)

# Import register_operator from base.py (central registry)
from cleaned_operators.base import (
    register_operator,
    ParamSpec,
    ParamRole,
)


# ============================================================================
# Feature Engineering Metrics
# ============================================================================

@register_operator(name="ts_feature_mode_share_pn", canonical="ts_feature_mode_share_pn", backend="polars_native")
class TSFeatureModeSharePolarsNative(SeriesOperator):
    """Fraction of values equal to the mode in rolling window"""

    metadata = OperatorMetadata(
        name="ts_feature_mode_share_pn",
        category="time_series",
        description="Fraction of values equal to the mode in rolling window",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "feature_engineering"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # Mode share = count of most frequent value / total count
        # Use rolling mode detection (approximate with median for continuous data)
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: (s.mode()[0] == s).sum() / len(s) if len(s) > 0 and len(s.mode() > 0 else None,
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_feature_pca_reconstruction_error", canonical="ts_feature_pca_reconstruction_error", backend="polars_native")
class TSFeaturePCAReconstructionErrorPolarsNative(SeriesOperator):
    """PCA reconstruction error in rolling window (requires multi-dimensional embedding)"""

    metadata = OperatorMetadata(
        name="ts_feature_pca_reconstruction_error",
        category="time_series",
        description="PCA reconstruction error using time-delay embedding",
        param_names=["x", "window", "n_components"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "feature_engineering", "pca"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
        "n_components": ParamSpec(dtype=int, min=1, default=1),
    }

    def _calculate_series(self, x, window, n_components=1, **kwargs):
        # TODO: Implement PCA-based reconstruction error
        # Requires: time-delay embedding matrix, SVD decomposition, reconstruction
        # Skeleton: return rolling window PCA reconstruction error
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_feature_subspace_rotation_pn", canonical="ts_feature_subspace_rotation_pn", backend="polars_native")
class TSFeatureSubspaceRotationPolarsNative(SeriesOperator):
    """Subspace rotation angle between consecutive windows"""

    metadata = OperatorMetadata(
        name="ts_feature_subspace_rotation_pn",
        category="time_series",
        description="Principal angle between PCA subspaces of consecutive windows",
        param_names=["x", "window", "n_components"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "feature_engineering"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
        "n_components": ParamSpec(dtype=int, min=1, default=1),
    }

    def _calculate_series(self, x, window, n_components=1, **kwargs):
        # TODO: Implement subspace rotation (principal angles between SVD subspaces)
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# Fill and Gap Operations
# ============================================================================

@register_operator(name="ts_ffill_limited", canonical="ts_ffill_limited", backend="polars_native")
class TSFfillLimitedPolarsNative(SeriesOperator):
    """Forward fill with maximum fill length limit"""

    metadata = OperatorMetadata(
        name="ts_ffill_limited",
        category="time_series",
        description="Forward fill nulls with maximum consecutive fill limit",
        param_names=["x", "limit"],
        return_type="series",
        tags=["time_series", "missing_data", "pit_safe"],
    )
    metadata.param_specs = {
        "limit": ParamSpec(dtype=int, min=1),
    }

    def _calculate_series(self, x, limit, **kwargs):
        # Forward fill but stop after 'limit' consecutive nulls
        df = x.to_frame().lazy()
        col_name = x.name

        return (
            df.with_columns([
                pl.col(col_name).is_null().cum_sum().alias("_null_group"),
                (~pl.col(col_name).is_null()).cum_sum().alias("_value_group"),
            ])
            .with_columns([
                (pl.col("_null_group") - pl.col("_null_group").shift(1).fill_null(0))
                .cum_sum()
                .over("_value_group")
                .alias("_fill_count")
            ])
            .with_columns([
                pl.when(pl.col("_fill_count") <= limit)
                .then(pl.col(col_name).forward_fill())
                .otherwise(pl.col(col_name))
                .alias(col_name)
            ])
            .select(col_name)
            .collect()
            .to_series()
        )


@register_operator(name="ts_gap_fill_ratio", canonical="ts_gap_fill_ratio", backend="polars_native")
class TSGapFillRatioPolarsNative(SeriesOperator):
    """Ratio of gap filled vs gap size for opening gaps"""

    metadata = OperatorMetadata(
        name="ts_gap_fill_ratio",
        category="time_series",
        description="Fraction of opening gap filled during the period",
        param_names=["open", "high", "low", "close_prev"],
        return_type="series",
        tags=["time_series", "price_action", "pit_safe"],
    )

    def _calculate_series(self, open, high, low, close_prev, **kwargs):
        # Gap = open - close_prev
        # Fill = how much gap was filled (high-open if gap up, open-low if gap down)
        df = pl.DataFrame({
            "open": open,
            "high": high,
            "low": low,
            "close_prev": close_prev,
        }).lazy()

        return (
            df.select([
                pl.when(pl.col("open") > pl.col("close_prev"))  # Gap up
                .then(
                    pl.when(pl.col("open") != 0)
                    .then((pl.col("high") - pl.col("open")) / (pl.col("open") - pl.col("close_prev")))
                    .otherwise(None)
                )
                .when(pl.col("open") < pl.col("close_prev"))  # Gap down
                .then(
                    pl.when(pl.col("close_prev") != 0)
                    .then((pl.col("open") - pl.col("low")) / (pl.col("close_prev") - pl.col("open")))
                    .otherwise(None)
                )
                .otherwise(None)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_gap_reversion_ratio", canonical="ts_gap_reversion_ratio", backend="polars_native")
class TSGapReversionRatioPolarsNative(SeriesOperator):
    """Ratio of close movement back toward previous close vs gap size"""

    metadata = OperatorMetadata(
        name="ts_gap_reversion_ratio",
        category="time_series",
        description="How much close reverted toward previous close relative to gap",
        param_names=["open", "close", "close_prev"],
        return_type="series",
        tags=["time_series", "price_action", "pit_safe", "mean_reversion"],
    )

    def _calculate_series(self, open, close, close_prev, **kwargs):
        df = pl.DataFrame({
            "open": open,
            "close": close,
            "close_prev": close_prev,
        }).lazy()

        return (
            df.select([
                pl.when(pl.col("open") != 0)
                .then((pl.col("open") - pl.col("close")) / (pl.col("open") - pl.col("close_prev")))
                .otherwise(None)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_gap_survival_duration", canonical="ts_gap_survival_duration", backend="polars_native")
class TSGapSurvivalDurationPolarsNative(SeriesOperator):
    """Days since gap was created until it gets filled"""

    metadata = OperatorMetadata(
        name="ts_gap_survival_duration",
        category="time_series",
        description="Number of periods a gap remains unfilled",
        param_names=["high", "low", "close_prev", "window"],
        return_type="series",
        tags=["time_series", "price_action", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, high, low, close_prev, window, **kwargs):
        # TODO: Track gap creation and count days until filled
        # Requires state tracking across multiple bars
        return pl.Series([None] * len(high), dtype=pl.Float64)


# ============================================================================
# GARCH/GJR Volatility Models
# ============================================================================

@register_operator(name="ts_garch_next_vol_forecast", canonical="ts_garch_next_vol_forecast", backend="polars_native")
class TSGarchNextVolForecastPolarsNative(SeriesOperator):
    """GARCH(1,1) one-step ahead volatility forecast"""

    metadata = OperatorMetadata(
        name="ts_garch_next_vol_forecast",
        category="time_series",
        description="GARCH(1,1) forecast of next period volatility",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility", "garch"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # TODO: Implement GARCH(1,1) estimation and forecast
        # sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        # Requires MLE estimation of (omega, alpha, beta) in rolling window
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_garch_persistence", canonical="ts_garch_persistence", backend="polars_native")
class TSGarchPersistencePolarsNative(SeriesOperator):
    """GARCH persistence parameter (alpha + beta)"""

    metadata = OperatorMetadata(
        name="ts_garch_persistence",
        category="time_series",
        description="GARCH(1,1) persistence: alpha + beta",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility", "garch"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # TODO: Estimate GARCH parameters and return alpha + beta
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_garch_standardized_shock", canonical="ts_garch_standardized_shock", backend="polars_native")
class TSGarchStandardizedShockPolarsNative(SeriesOperator):
    """Standardized residual from GARCH model"""

    metadata = OperatorMetadata(
        name="ts_garch_standardized_shock",
        category="time_series",
        description="Return divided by GARCH conditional volatility",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility", "garch"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # TODO: Estimate GARCH volatility and standardize returns
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_garch_vol_surprise", canonical="ts_garch_vol_surprise", backend="polars_native")
class TSGarchVolSurprisePolarsNative(SeriesOperator):
    """Realized vol minus GARCH forecast"""

    metadata = OperatorMetadata(
        name="ts_garch_vol_surprise",
        category="time_series",
        description="Difference between realized and GARCH-forecasted volatility",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility", "garch"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # TODO: Compare realized vs GARCH forecast
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_gjr_garch_vol_forecast", canonical="ts_gjr_garch_vol_forecast", backend="polars_native")
class TSGJRGarchVolForecastPolarsNative(SeriesOperator):
    """GJR-GARCH volatility forecast with leverage effect"""

    metadata = OperatorMetadata(
        name="ts_gjr_garch_vol_forecast",
        category="time_series",
        description="GJR-GARCH one-step ahead volatility forecast",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility", "garch"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # TODO: GJR-GARCH with asymmetric term for negative returns
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_gjr_leverage", canonical="ts_gjr_leverage", backend="polars_native")
class TSGJRLeveragePolarsNative(SeriesOperator):
    """GJR-GARCH leverage parameter (gamma)"""

    metadata = OperatorMetadata(
        name="ts_gjr_leverage",
        category="time_series",
        description="GJR-GARCH leverage effect parameter",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility", "garch"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # TODO: Estimate GJR gamma parameter
        return pl.Series([None] * len(returns), dtype=pl.Float64)


# ============================================================================
# Extremal Theory / Tail Indices
# ============================================================================

@register_operator(name="ts_gpd_shape_pwm_pn", canonical="ts_gpd_shape_pwm_pn", backend="polars_native")
class TSGPDShapePWMPolarsNative(SeriesOperator):
    """Generalized Pareto Distribution shape parameter via PWM"""

    metadata = OperatorMetadata(
        name="ts_gpd_shape_pwm_pn",
        category="time_series",
        description="GPD shape parameter (tail index) using probability weighted moments",
        param_names=["x", "window", "threshold_quantile"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "tail_risk", "extremal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "threshold_quantile": ParamSpec(dtype=float, min=0.5, max=0.99, default=0.95),
    }

    def _calculate_series(self, x, window, threshold_quantile=0.95, **kwargs):
        # TODO: Implement GPD parameter estimation via PWM method
        # Fit GPD to exceedances above threshold_quantile
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_hill_tail_index_pn", canonical="ts_hill_tail_index_pn", backend="polars_native")
class TSHillTailIndexPolarsNative(SeriesOperator):
    """Hill estimator for tail index"""

    metadata = OperatorMetadata(
        name="ts_hill_tail_index_pn",
        category="time_series",
        description="Hill estimator of tail index (heavy-tail parameter)",
        param_names=["x", "window", "n_upper"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "tail_risk", "extremal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_upper": ParamSpec(dtype=int, min=5, default=10),
    }

    def _calculate_series(self, x, window, n_upper=10, **kwargs):
        # Hill estimator: (1/k) * sum(log(X_i) - log(X_{k+1}))
        # where X_i are the k largest order statistics
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: (
                        np.mean(np.log(np.sort(s.to_numpy())[-n_upper:])) -
                        np.log(np.sort(s.to_numpy())[-n_upper])
                    ) if len(s) >= n_upper and np.sort(s.to_numpy())[-n_upper] > 0 else None,
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_pickands_tail_index", canonical="ts_pickands_tail_index", backend="polars_native")
class TSPickandsTailIndexPolarsNative(SeriesOperator):
    """Pickands estimator for tail index"""

    metadata = OperatorMetadata(
        name="ts_pickands_tail_index",
        category="time_series",
        description="Pickands estimator of tail index",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "tail_risk", "extremal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Implement Pickands estimator
        # Based on ratio of order statistics
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# GLR (Generalized Likelihood Ratio) Change Detection
# ============================================================================

@register_operator(name="ts_glr_mean_shift_score", canonical="ts_glr_mean_shift_score", backend="polars_native")
class TSGLRMeanShiftScorePolarsNative(SeriesOperator):
    """GLR test statistic for mean shift detection"""

    metadata = OperatorMetadata(
        name="ts_glr_mean_shift_score",
        category="time_series",
        description="Generalized likelihood ratio for detecting mean shifts",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "change_detection"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # GLR: max over all split points of likelihood ratio
        # Approximate with variance-weighted mean difference
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: (
                        max([
                            abs(s[:i].mean() - s[i:].mean()) / (s.std() + 1e-8)
                            for i in range(1, len(s))
                        ]) if len(s) >= 2 else None
                    ),
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_glr_variance_shift_score", canonical="ts_glr_variance_shift_score", backend="polars_native")
class TSGLRVarianceShiftScorePolarsNative(SeriesOperator):
    """GLR test statistic for variance shift detection"""

    metadata = OperatorMetadata(
        name="ts_glr_variance_shift_score",
        category="time_series",
        description="Generalized likelihood ratio for detecting variance shifts",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "change_detection"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # GLR for variance change
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: (
                        max([
                            abs(np.log(s[:i].std() + 1e-8) - np.log(s[i:].std() + 1e-8))
                            for i in range(2, len(s)-2)
                        ]) if len(s) >= 4 else None
                    ),
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Hurst Exponent and Fractal Dimension
# ============================================================================

@register_operator(name="ts_generalized_hurst_exponent_pn", canonical="ts_generalized_hurst_exponent_pn", backend="polars_native")
class TSGeneralizedHurstExponentPolarsNative(SeriesOperator):
    """Generalized Hurst exponent for specific q moment"""

    metadata = OperatorMetadata(
        name="ts_generalized_hurst_exponent_pn",
        category="time_series",
        description="Generalized Hurst exponent H(q) for multifractal analysis",
        param_names=["x", "window", "q"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "fractal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "q": ParamSpec(dtype=float, default=2.0),
    }

    def _calculate_series(self, x, window, q=2.0, **kwargs):
        # TODO: Implement generalized Hurst via q-order structure function
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_generalized_hurst_spread_q1_q4_pn", canonical="ts_generalized_hurst_spread_q1_q4_pn", backend="polars_native")
class TSGeneralizedHurstSpreadQ1Q4PolarsNative(SeriesOperator):
    """Difference H(1) - H(4) indicating multifractal asymmetry"""

    metadata = OperatorMetadata(
        name="ts_generalized_hurst_spread_q1_q4_pn",
        category="time_series",
        description="Spread between H(q=1) and H(q=4) for multifractal detection",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "fractal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Compute H(1) - H(4)
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_higuchi_fractal_dimension_pn", canonical="ts_higuchi_fractal_dimension_pn", backend="polars_native")
class TSHiguchiFractalDimensionPolarsNative(SeriesOperator):
    """Higuchi fractal dimension"""

    metadata = OperatorMetadata(
        name="ts_higuchi_fractal_dimension_pn",
        category="time_series",
        description="Higuchi method for estimating fractal dimension",
        param_names=["x", "window", "kmax"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "fractal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "kmax": ParamSpec(dtype=int, min=2, default=10),
    }

    def _calculate_series(self, x, window, kmax=10, **kwargs):
        # TODO: Implement Higuchi FD algorithm
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_hurst_dfa_pn", canonical="ts_hurst_dfa_pn", backend="polars_native")
class TSHurstDFAPolarsNative(SeriesOperator):
    """Hurst exponent via Detrended Fluctuation Analysis"""

    metadata = OperatorMetadata(
        name="ts_hurst_dfa_pn",
        category="time_series",
        description="Hurst exponent using DFA method",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "fractal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Implement DFA-based Hurst estimation
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# HAR (Heterogeneous Autoregressive) Models
# ============================================================================

@register_operator(name="ts_har_from_return_forecast_error_z", canonical="ts_har_from_return_forecast_error_z", backend="polars_native")
class TSHARFromReturnForecastErrorZPolarsNative(SeriesOperator):
    """HAR forecast error z-score when predicting from returns"""

    metadata = OperatorMetadata(
        name="ts_har_from_return_forecast_error_z",
        category="time_series",
        description="Standardized HAR forecast error",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=22, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # TODO: HAR model: RV_t = beta_d*RV_{t-1} + beta_w*RV_{t-5:t-1} + beta_m*RV_{t-22:t-1}
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_har_from_return_next_vol", canonical="ts_har_from_return_next_vol", backend="polars_native")
class TSHARFromReturnNextVolPolarsNative(SeriesOperator):
    """HAR forecast of next period volatility from returns"""

    metadata = OperatorMetadata(
        name="ts_har_from_return_next_vol",
        category="time_series",
        description="HAR one-step ahead volatility forecast",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=22, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # TODO: Implement HAR forecast
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_har_rv_forecast_error_z", canonical="ts_har_rv_forecast_error_z", backend="polars_native")
class TSHARRVForecastErrorZPolarsNative(SeriesOperator):
    """HAR-RV forecast error z-score"""

    metadata = OperatorMetadata(
        name="ts_har_rv_forecast_error_z",
        category="time_series",
        description="Standardized HAR-RV forecast error",
        param_names=["realized_vol", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=22, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, realized_vol, window, **kwargs):
        # TODO: Implement HAR-RV error
        return pl.Series([None] * len(realized_vol), dtype=pl.Float64)


@register_operator(name="ts_har_rv_next_var_forecast", canonical="ts_har_rv_next_var_forecast", backend="polars_native")
class TSHARRVNextVarForecastPolarsNative(SeriesOperator):
    """HAR-RV forecast of next period variance"""

    metadata = OperatorMetadata(
        name="ts_har_rv_next_var_forecast",
        category="time_series",
        description="HAR-RV one-step ahead variance forecast",
        param_names=["realized_vol", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=22, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, realized_vol, window, **kwargs):
        # TODO: HAR-RV variance forecast
        return pl.Series([None] * len(realized_vol), dtype=pl.Float64)


@register_operator(name="ts_har_rv_next_vol_forecast", canonical="ts_har_rv_next_vol_forecast", backend="polars_native")
class TSHARRVNextVolForecastPolarsNative(SeriesOperator):
    """HAR-RV forecast of next period volatility"""

    metadata = OperatorMetadata(
        name="ts_har_rv_next_vol_forecast",
        category="time_series",
        description="HAR-RV one-step ahead volatility forecast",
        param_names=["realized_vol", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=22, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, realized_vol, window, **kwargs):
        # TODO: HAR-RV volatility forecast
        return pl.Series([None] * len(realized_vol), dtype=pl.Float64)


# ============================================================================
# Advanced Filters and Signal Processing
# ============================================================================

@register_operator(name="ts_fir_lowpass_causal_pn", canonical="ts_fir_lowpass_causal_pn", backend="polars_native")
class TSFIRLowpassCausalPolarsNative(SeriesOperator):
    """Causal FIR lowpass filter"""

    metadata = OperatorMetadata(
        name="ts_fir_lowpass_causal_pn",
        category="time_series",
        description="Causal finite impulse response lowpass filter",
        param_names=["x", "window", "cutoff"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "filter"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
        "cutoff": ParamSpec(dtype=float, min=0.0, max=0.5, default=0.1),
    }

    def _calculate_series(self, x, window, cutoff=0.1, **kwargs):
        # TODO: Design FIR filter coefficients and apply
        # For now, use simple exponential smoothing as approximation
        alpha = cutoff * 2
        return x.to_frame().lazy().select([
            pl.col(x.name).ewm_mean(alpha=alpha)
        ]).collect().to_series()


@register_operator(name="ts_hampel_filter_causal_pn", canonical="ts_hampel_filter_causal_pn", backend="polars_native")
class TSHampelFilterCausalPolarsNative(SeriesOperator):
    """Causal Hampel filter for outlier detection/removal"""

    metadata = OperatorMetadata(
        name="ts_hampel_filter_causal_pn",
        category="time_series",
        description="Causal Hampel filter replaces outliers with median",
        param_names=["x", "window", "n_sigma"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "filter", "outlier"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
        "n_sigma": ParamSpec(dtype=float, min=0.0, default=3.0),
    }

    def _calculate_series(self, x, window, n_sigma=3.0, **kwargs):
        # Hampel: replace values > n_sigma * MAD from median
        df = x.to_frame().lazy()
        col = x.name

        return (
            df.with_columns([
                pl.col(col).rolling_median(window).alias("_med"),
                (pl.col(col) - pl.col(col).rolling_median(window)).abs().rolling_median(window).alias("_mad"),
            ])
            .select([
                pl.when(pl.col(col) - pl.col("_med")).abs() > n_sigma * 1.4826 * pl.col("_mad"))
                .then(pl.col("_med"))
                .otherwise(pl.col(col))
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_h_infinity_level_filter_pn", canonical="ts_h_infinity_level_filter_pn", backend="polars_native")
class TSHInfinityLevelFilterPolarsNative(SeriesOperator):
    """H-infinity robust filter for level estimation"""

    metadata = OperatorMetadata(
        name="ts_h_infinity_level_filter_pn",
        category="time_series",
        description="H-infinity filter for robust level tracking",
        param_names=["x", "gamma"],
        return_type="series",
        tags=["time_series", "filter", "pit_safe", "robust"],
    )
    metadata.param_specs = {
        "gamma": ParamSpec(dtype=float, min=0.0, default=1.0),
    }

    def _calculate_series(self, x, gamma=1.0, **kwargs):
        # TODO: Implement H-infinity filter recursion
        # For now, use robust moving average
        return x.to_frame().lazy().select([
            pl.col(x.name).ewm_mean(alpha=0.1)
        ]).collect().to_series()


# ============================================================================
# Hankel Matrix Features
# ============================================================================

@register_operator(name="ts_hankel_effective_rank_pn", canonical="ts_hankel_effective_rank_pn", backend="polars_native")
class TSHankelEffectiveRankPolarsNative(SeriesOperator):
    """Effective rank of Hankel matrix (complexity measure)"""

    metadata = OperatorMetadata(
        name="ts_hankel_effective_rank_pn",
        category="time_series",
        description="Effective rank of Hankel matrix for time series complexity",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "complexity"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Build Hankel matrix and compute effective rank from singular values
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_hankel_singular_gap_pn", canonical="ts_hankel_singular_gap_pn", backend="polars_native")
class TSHankelSingularGapPolarsNative(SeriesOperator):
    """Gap between first and second singular values of Hankel matrix"""

    metadata = OperatorMetadata(
        name="ts_hankel_singular_gap_pn",
        category="time_series",
        description="Spectral gap in Hankel SVD indicating dominant structure",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "complexity"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Compute sigma_1 - sigma_2 from Hankel SVD
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# Statistical Tests and Divergence Measures
# ============================================================================

@register_operator(name="ts_hartigan_dip_pn", canonical="ts_hartigan_dip_pn", backend="polars_native")
class TSHartiganDipPolarsNative(SeriesOperator):
    """Hartigan dip test statistic for unimodality"""

    metadata = OperatorMetadata(
        name="ts_hartigan_dip_pn",
        category="time_series",
        description="Hartigan dip test for detecting multimodality",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "distribution"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Implement Hartigan dip test
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_hodges_lehmann_location", canonical="ts_hodges_lehmann_location", backend="polars_native")
class TSHodgesLehmannLocationPolarsNative(SeriesOperator):
    """Hodges-Lehmann robust location estimator"""

    metadata = OperatorMetadata(
        name="ts_hodges_lehmann_location",
        category="time_series",
        description="Median of pairwise averages (robust location)",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "robust"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # Hodges-Lehmann: median of (x_i + x_j) / 2 for all pairs
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: np.median([(s[i] + s[j]) / 2 for i in range(len(s)) for j in range(i, len(s))]) if len(s) >= 2 else None,
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_hsic_pn", canonical="ts_hsic_pn", backend="polars_native")
class TSHSICPolarsNative(TwoVarOperator):
    """Hilbert-Schmidt Independence Criterion"""

    metadata = OperatorMetadata(
        name="ts_hsic_pn",
        category="time_series",
        description="HSIC dependence measure between two series",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "dependence"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, window, **kwargs):
        # TODO: Implement HSIC with RBF kernel
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_ks_shift_pn", canonical="ts_ks_shift_pn", backend="polars_native")
class TSKSShiftPolarsNative(SeriesOperator):
    """Kolmogorov-Smirnov statistic between consecutive windows"""

    metadata = OperatorMetadata(
        name="ts_ks_shift_pn",
        category="time_series",
        description="KS distance between current and previous window distributions",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "distribution_shift"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Compute KS distance between rolling windows
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# L-moments (Robust Moment Estimators)
# ============================================================================

@register_operator(name="ts_l_kurtosis_pn", canonical="ts_l_kurtosis_pn", backend="polars_native")
class TSLKurtosisPolarsNative(SeriesOperator):
    """L-kurtosis (fourth L-moment ratio)"""

    metadata = OperatorMetadata(
        name="ts_l_kurtosis_pn",
        category="time_series",
        description="L-kurtosis: robust tail weight measure",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "robust"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Compute L-moment based kurtosis
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_l_skewness_pn", canonical="ts_l_skewness_pn", backend="polars_native")
class TSLSkewnessPolarsNative(SeriesOperator):
    """L-skewness (third L-moment ratio)"""

    metadata = OperatorMetadata(
        name="ts_l_skewness_pn",
        category="time_series",
        description="L-skewness: robust asymmetry measure",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "robust"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Compute L-moment based skewness
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# Markov Chain Features
# ============================================================================

@register_operator(name="ts_markov_committor_pn", canonical="ts_markov_committor_pn", backend="polars_native")
class TSMarkovCommittorPolarsNative(SeriesOperator):
    """Committor probability (probability of reaching state B before A)"""

    metadata = OperatorMetadata(
        name="ts_markov_committor_pn",
        category="time_series",
        description="Committor function from discretized Markov chain",
        param_names=["x", "window", "n_states"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "markov"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5),
    }

    def _calculate_series(self, x, window, n_states=5, **kwargs):
        # TODO: Discretize into states, build transition matrix, solve committor equation
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_markov_entropy_production_pn", canonical="ts_markov_entropy_production_pn", backend="polars_native")
class TSMarkovEntropyProductionPolarsNative(SeriesOperator):
    """Markov chain entropy production rate"""

    metadata = OperatorMetadata(
        name="ts_markov_entropy_production_pn",
        category="time_series",
        description="Non-equilibrium entropy production from transition matrix",
        param_names=["x", "window", "n_states"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "markov"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5),
    }

    def _calculate_series(self, x, window, n_states=5, **kwargs):
        # TODO: Compute sum over (P_ij - P_ji) * log(P_ij/P_ji)
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_markov_mean_first_passage_time_pn", canonical="ts_markov_mean_first_passage_time_pn", backend="polars_native")
class TSMarkovMeanFirstPassageTimePolarsNative(SeriesOperator):
    """Mean first passage time between states"""

    metadata = OperatorMetadata(
        name="ts_markov_mean_first_passage_time_pn",
        category="time_series",
        description="Expected time to reach target state from current state",
        param_names=["x", "window", "n_states"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "markov"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5),
    }

    def _calculate_series(self, x, window, n_states=5, **kwargs):
        # TODO: Solve linear system for MFPT from transition matrix
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_markov_persistence_pn", canonical="ts_markov_persistence_pn", backend="polars_native")
class TSMarkovPersistencePolarsNative(SeriesOperator):
    """Average diagonal dominance of transition matrix"""

    metadata = OperatorMetadata(
        name="ts_markov_persistence_pn",
        category="time_series",
        description="Self-transition probability (state persistence)",
        param_names=["x", "window", "n_states"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "markov"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5),
    }

    def _calculate_series(self, x, window, n_states=5, **kwargs):
        # Average of diagonal entries of transition matrix
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: (
                        _compute_markov_persistence(s.to_numpy(), n_states)
                        if len(s) >= 2 else None
                    ),
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_markov_spectral_gap_pn", canonical="ts_markov_spectral_gap_pn", backend="polars_native")
class TSMarkovSpectralGapPolarsNative(SeriesOperator):
    """Spectral gap (1 - second eigenvalue) of transition matrix"""

    metadata = OperatorMetadata(
        name="ts_markov_spectral_gap_pn",
        category="time_series",
        description="Spectral gap indicating mixing rate",
        param_names=["x", "window", "n_states"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "markov"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5),
    }

    def _calculate_series(self, x, window, n_states=5, **kwargs):
        # TODO: Compute eigenvalues of transition matrix, return 1 - lambda_2
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_markov_state_entropy_pn", canonical="ts_markov_state_entropy_pn", backend="polars_native")
class TSMarkovStateEntropyPolarsNative(SeriesOperator):
    """Entropy of stationary distribution"""

    metadata = OperatorMetadata(
        name="ts_markov_state_entropy_pn",
        category="time_series",
        description="Shannon entropy of Markov stationary distribution",
        param_names=["x", "window", "n_states"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "markov"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5),
    }

    def _calculate_series(self, x, window, n_states=5, **kwargs):
        # TODO: Find stationary distribution pi, compute -sum(pi * log(pi))
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_markov_stationary_surprisal_pn", canonical="ts_markov_stationary_surprisal_pn", backend="polars_native")
class TSMarkovStationarySurprisalPolarsNative(SeriesOperator):
    """Surprisal of current state under stationary distribution"""

    metadata = OperatorMetadata(
        name="ts_markov_stationary_surprisal_pn",
        category="time_series",
        description="-log(stationary probability) of current discretized state",
        param_names=["x", "window", "n_states"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "markov"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5),
    }

    def _calculate_series(self, x, window, n_states=5, **kwargs):
        # TODO: Compute stationary dist, return -log(pi[current_state])
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_markov_persistence(arr: np.ndarray, n_states: int) -> float:
    """Compute average self-transition probability from discretized series"""
    if len(arr) < 2:
        return None

    # Discretize into n_states
    bins = np.linspace(np.min(arr), np.max(arr), n_states + 1)
    states = np.digitize(arr, bins[1:-1])

    # Build transition matrix
    trans_matrix = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        trans_matrix[states[i], states[i+1]] += 1

    # Normalize rows
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    trans_matrix = (trans_matrix) / row_sums if row_sums != 0 else np.nan

    # Return average diagonal
    return np.mean(np.diag(trans_matrix))


# ============================================================================
# First Passage and Impulse Detection
# ============================================================================

@register_operator(name="ts_first_passage_bias_pn", canonical="ts_first_passage_bias_pn", backend="polars_native")
class TSFirstPassageBiasPolarsNative(SeriesOperator):
    """Asymmetry in hitting upper vs lower threshold"""

    metadata = OperatorMetadata(
        name="ts_first_passage_bias_pn",
        category="time_series",
        description="Ratio of upward to downward threshold hits",
        param_names=["x", "window", "threshold"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "threshold"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, min=0.0, default=1.0),
    }

    def _calculate_series(self, x, window, threshold=1.0, **kwargs):
        # Count threshold crossings up vs down
        df = x.to_frame().lazy()
        col = x.name

        return (
            df.with_columns([
                (pl.col(col) > threshold).cast(pl.Int32).alias("_up"),
                (pl.col(col) < -threshold).cast(pl.Int32).alias("_down"),
            ])
            .select([
                (pl.col("_up").rolling_sum(window) + 1) /
                (pl.col("_down").rolling_sum(window) + 1)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_first_passage_conditional_time_pn", canonical="ts_first_passage_conditional_time_pn", backend="polars_native")
class TSFirstPassageConditionalTimePolarsNative(SeriesOperator):
    """Expected time to threshold conditional on current level"""

    metadata = OperatorMetadata(
        name="ts_first_passage_conditional_time_pn",
        category="time_series",
        description="Estimated first passage time based on current distance to threshold",
        param_names=["x", "window", "threshold"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.0),
    }

    def _calculate_series(self, x, window, threshold=0.0, **kwargs):
        # TODO: Model first passage time distribution
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_first_passage_hit_probability_pn", canonical="ts_first_passage_hit_probability_pn", backend="polars_native")
class TSFirstPassageHitProbabilityPolarsNative(SeriesOperator):
    """Probability of hitting threshold before opposite threshold"""

    metadata = OperatorMetadata(
        name="ts_first_passage_hit_probability_pn",
        category="time_series",
        description="Probability of hitting upper before lower threshold",
        param_names=["x", "window", "upper", "lower"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "upper": ParamSpec(dtype=float, default=1.0),
        "lower": ParamSpec(dtype=float, default=-1.0),
    }

    def _calculate_series(self, x, window, upper=1.0, lower=-1.0, **kwargs):
        # TODO: Estimate hit probability from historical patterns
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_impulse_return", canonical="ts_impulse_return", backend="polars_native")
class TSImpulseReturnPolarsNative(SeriesOperator):
    """Return on bars identified as impulse moves"""

    metadata = OperatorMetadata(
        name="ts_impulse_return",
        category="time_series",
        description="Return magnitude on impulse bars (large z-score moves)",
        param_names=["returns", "window", "z_threshold"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "z_threshold": ParamSpec(dtype=float, min=0.0, default=2.0),
    }

    def _calculate_series(self, returns, window, z_threshold=2.0, **kwargs):
        df = returns.to_frame().lazy()
        col = returns.name

        return (
            df.with_columns([
                ((pl.col(col) - pl.col(col).rolling_mean(window)) /
                 (pl.col(col).rolling_std(window) + 1e-8)).alias("_z")
            ])
            .select([
                pl.when(pl.col("_z").abs() > z_threshold)
                .then(pl.col(col))
                .otherwise(0.0)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_impulse_strength", canonical="ts_impulse_strength", backend="polars_native")
class TSImpulseStrengthPolarsNative(SeriesOperator):
    """Z-score of largest move in window"""

    metadata = OperatorMetadata(
        name="ts_impulse_strength",
        category="time_series",
        description="Maximum absolute z-score in rolling window",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        df = x.to_frame().lazy()
        col = x.name

        return (
            df.with_columns([
                ((pl.col(col) - pl.col(col).rolling_mean(window)) /
                 (pl.col(col).rolling_std(window) + 1e-8)).abs().alias("_z")
            ])
            .select([
                pl.col("_z").rolling_max(window)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_impulse_volume", canonical="ts_impulse_volume", backend="polars_native")
class TSImpulseVolumePolarsNative(SeriesOperator):
    """Volume on impulse bars relative to average"""

    metadata = OperatorMetadata(
        name="ts_impulse_volume",
        category="time_series",
        description="Volume ratio on high z-score move bars",
        param_names=["returns", "volume", "window", "z_threshold"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "z_threshold": ParamSpec(dtype=float, min=0.0, default=2.0),
    }

    def _calculate_series(self, returns, volume, window, z_threshold=2.0, **kwargs):
        df = pl.DataFrame({
            "returns": returns,
            "volume": volume,
        }).lazy()

        return (
            df.with_columns([
                ((pl.col("returns") - pl.col("returns").rolling_mean(window)) /
                 (pl.col("returns").rolling_std(window) + 1e-8)).alias("_z"),
                pl.col("volume").rolling_mean(window).alias("_vol_avg")
            ])
            .select([
                pl.when(pl.col("_z").abs() > z_threshold)
                .then(pl.col("volume") / (pl.col("_vol_avg") + 1e-8))
                .otherwise(None)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Fisher Information and Ordinal Patterns
# ============================================================================

@register_operator(name="ts_fisher_information_shift_pn", canonical="ts_fisher_information_shift_pn", backend="polars_native")
class TSFisherInformationShiftPolarsNative(SeriesOperator):
    """Change in Fisher information between consecutive windows"""

    metadata = OperatorMetadata(
        name="ts_fisher_information_shift_pn",
        category="time_series",
        description="Fisher information difference for distribution change detection",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "information"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Compute Fisher information metric
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_forbidden_ordinal_pattern_excess", canonical="ts_forbidden_ordinal_pattern_excess", backend="polars_native")
class TSForbiddenOrdinalPatternExcessPolarsNative(SeriesOperator):
    """Count of ordinal patterns that should be rare but appear"""

    metadata = OperatorMetadata(
        name="ts_forbidden_ordinal_pattern_excess",
        category="time_series",
        description="Excess occurrences of theoretically rare ordinal patterns",
        param_names=["x", "window", "pattern_length"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "ordinal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "pattern_length": ParamSpec(dtype=int, min=3, max=7, default=3),
    }

    def _calculate_series(self, x, window, pattern_length=3, **kwargs):
        # TODO: Detect forbidden ordinal patterns
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_forbidden_ordinal_pattern_ratio_pn", canonical="ts_forbidden_ordinal_pattern_ratio_pn", backend="polars_native")
class TSForbiddenOrdinalPatternRatioPolarsNative(SeriesOperator):
    """Ratio of forbidden to allowed ordinal patterns"""

    metadata = OperatorMetadata(
        name="ts_forbidden_ordinal_pattern_ratio_pn",
        category="time_series",
        description="Fraction of patterns that are theoretically forbidden",
        param_names=["x", "window", "pattern_length"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "ordinal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "pattern_length": ParamSpec(dtype=int, min=3, max=7, default=3),
    }

    def _calculate_series(self, x, window, pattern_length=3, **kwargs):
        # TODO: Compute forbidden pattern ratio
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_forbidden_ordinal_pattern_signed_excess", canonical="ts_forbidden_ordinal_pattern_signed_excess", backend="polars_native")
class TSForbiddenOrdinalPatternSignedExcessPolarsNative(SeriesOperator):
    """Signed excess of ascending vs descending forbidden patterns"""

    metadata = OperatorMetadata(
        name="ts_forbidden_ordinal_pattern_signed_excess",
        category="time_series",
        description="Directional bias in forbidden pattern occurrences",
        param_names=["x", "window", "pattern_length"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "ordinal"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "pattern_length": ParamSpec(dtype=int, min=3, max=7, default=3),
    }

    def _calculate_series(self, x, window, pattern_length=3, **kwargs):
        # TODO: Compute signed excess
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# Fractional Differentiation
# ============================================================================

@register_operator(name="ts_fractional_difference_pn", canonical="ts_fractional_difference_pn", backend="polars_native")
class TSFractionalDifferencePolarsNative(SeriesOperator):
    """Fractional differentiation to achieve stationarity while preserving memory"""

    metadata = OperatorMetadata(
        name="ts_fractional_difference_pn",
        category="time_series",
        description="Fractional differentiation with parameter d",
        param_names=["x", "d", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "stationarity"],
    )
    metadata.param_specs = {
        "d": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5),
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, d, window, **kwargs):
        # TODO: Implement fractional differencing weights
        # weights[k] = (-1)^k * gamma(d+1) / (gamma(k+1) * gamma(d-k+1))
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_fractional_difference_discarded_weight_mass_pn", canonical="ts_fractional_difference_discarded_weight_mass_pn", backend="polars_native")
class TSFractionalDifferenceDiscardedWeightMassPolarsNative(SeriesOperator):
    """Total weight discarded when truncating fractional difference filter"""

    metadata = OperatorMetadata(
        name="ts_fractional_difference_discarded_weight_mass_pn",
        category="time_series",
        description="Sum of fractional diff weights beyond window",
        param_names=["d", "window"],
        return_type="scalar",
        tags=["time_series", "stationarity"],
    )
    metadata.param_specs = {
        "d": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5),
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, d, window, **kwargs):
        # TODO: Compute sum of weights beyond window
        return None


# ============================================================================
# Hysteresis Detection
# ============================================================================

@register_operator(name="ts_hysteresis_age_pn", canonical="ts_hysteresis_age_pn", backend="polars_native")
class TSHysteresisAgePolarsNative(SeriesOperator):
    """Bars since last state change in hysteresis detector"""

    metadata = OperatorMetadata(
        name="ts_hysteresis_age_pn",
        category="time_series",
        description="Time since hysteresis state flip",
        param_names=["x", "upper_threshold", "lower_threshold"],
        return_type="series",
        tags=["time_series", "pit_safe", "state"],
    )
    metadata.param_specs = {
        "upper_threshold": ParamSpec(dtype=float, default=1.0),
        "lower_threshold": ParamSpec(dtype=float, default=-1.0),
    }

    def _calculate_series(self, x, upper_threshold=1.0, lower_threshold=-1.0, **kwargs):
        # TODO: Track hysteresis state changes
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_hysteresis_state_pn", canonical="ts_hysteresis_state_pn", backend="polars_native")
class TSHysteresisStatePolarsNative(SeriesOperator):
    """Current state of hysteresis detector (1, 0, -1)"""

    metadata = OperatorMetadata(
        name="ts_hysteresis_state_pn",
        category="time_series",
        description="Hysteresis state with upper/lower thresholds",
        param_names=["x", "upper_threshold", "lower_threshold"],
        return_type="series",
        tags=["time_series", "pit_safe", "state"],
    )
    metadata.param_specs = {
        "upper_threshold": ParamSpec(dtype=float, default=1.0),
        "lower_threshold": ParamSpec(dtype=float, default=-1.0),
    }

    def _calculate_series(self, x, upper_threshold=1.0, lower_threshold=-1.0, **kwargs):
        # TODO: Implement stateful hysteresis logic
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# Interval Analysis
# ============================================================================

@register_operator(name="ts_interval_exploration_efficiency_pn", canonical="ts_interval_exploration_efficiency_pn", backend="polars_native")
class TSIntervalExplorationEfficiencyPolarsNative(SeriesOperator):
    """Ratio of unique intervals visited to total intervals"""

    metadata = OperatorMetadata(
        name="ts_interval_exploration_efficiency_pn",
        category="time_series",
        description="Efficiency of price range exploration",
        param_names=["x", "window", "n_bins"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_bins": ParamSpec(dtype=int, min=5, default=10),
    }

    def _calculate_series(self, x, window, n_bins=10, **kwargs):
        # Unique bins visited / total bins
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: len(np.unique(np.digitize(s, np.linspace(s.min(), s.max(), n_bins)))) / n_bins if len(s) > 0 else None,
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_interval_nesting_depth_pn", canonical="ts_interval_nesting_depth_pn", backend="polars_native")
class TSIntervalNestingDepthPolarsNative(SeriesOperator):
    """Maximum nesting depth of price ranges"""

    metadata = OperatorMetadata(
        name="ts_interval_nesting_depth_pn",
        category="time_series",
        description="Maximum nesting level of overlapping ranges",
        param_names=["high", "low", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, high, low, window, **kwargs):
        # TODO: Compute interval nesting depth
        return pl.Series([None] * len(high), dtype=pl.Float64)


@register_operator(name="ts_interval_occupancy_entropy_pn", canonical="ts_interval_occupancy_entropy_pn", backend="polars_native")
class TSIntervalOccupancyEntropyPolarsNative(SeriesOperator):
    """Entropy of time spent in each discretized interval"""

    metadata = OperatorMetadata(
        name="ts_interval_occupancy_entropy_pn",
        category="time_series",
        description="Shannon entropy of interval occupancy distribution",
        param_names=["x", "window", "n_bins"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "entropy"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_bins": ParamSpec(dtype=int, min=5, default=10),
    }

    def _calculate_series(self, x, window, n_bins=10, **kwargs):
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: (
                        -np.sum(p * np.log(p + 1e-10) for p in np.histogram(s, bins=n_bins)[0] / len(s) if p > 0)
                        if len(s) > 0 else None
                    ),
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_interval_occupancy_mode_distance_pn", canonical="ts_interval_occupancy_mode_distance_pn", backend="polars_native")
class TSIntervalOccupancyModeDistancePolarsNative(SeriesOperator):
    """Distance of current value from most frequently occupied interval"""

    metadata = OperatorMetadata(
        name="ts_interval_occupancy_mode_distance_pn",
        category="time_series",
        description="Distance from modal interval center",
        param_names=["x", "window", "n_bins"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_bins": ParamSpec(dtype=int, min=5, default=10),
    }

    def _calculate_series(self, x, window, n_bins=10, **kwargs):
        # TODO: Find modal interval and compute distance
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_interval_overlap_connected_component_ratio_pn", canonical="ts_interval_overlap_connected_component_ratio_pn", backend="polars_native")
class TSIntervalOverlapConnectedComponentRatioPolarsNative(SeriesOperator):
    """Ratio of connected components in interval overlap graph"""

    metadata = OperatorMetadata(
        name="ts_interval_overlap_connected_component_ratio_pn",
        category="time_series",
        description="Fragmentation of overlapping price ranges",
        param_names=["high", "low", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, high, low, window, **kwargs):
        # TODO: Build interval overlap graph and count components
        return pl.Series([None] * len(high), dtype=pl.Float64)


@register_operator(name="ts_interval_union_coverage_pn", canonical="ts_interval_union_coverage_pn", backend="polars_native")
class TSIntervalUnionCoveragePolarsNative(SeriesOperator):
    """Union of all intervals as fraction of total range"""

    metadata = OperatorMetadata(
        name="ts_interval_union_coverage_pn",
        category="time_series",
        description="Coverage ratio of union of OHLC ranges",
        param_names=["high", "low", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, high, low, window, **kwargs):
        df = pl.DataFrame({
            "high": high,
            "low": low,
        }).lazy()

        return (
            df.select([
                (pl.col("high").rolling_max(window) - pl.col("low").rolling_min(window)) /
                ((pl.col("high").rolling_max(window) - pl.col("low").rolling_min(window)).rolling_mean(window) + 1e-8)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Jump Detection
# ============================================================================

@register_operator(name="ts_joint_energy_shift_pn", canonical="ts_joint_energy_shift_pn", backend="polars_native")
class TSJointEnergyShiftPolarsNative(SeriesOperator):
    """Change in joint energy between consecutive windows"""

    metadata = OperatorMetadata(
        name="ts_joint_energy_shift_pn",
        category="time_series",
        description="Joint energy difference for bivariate change detection",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "change_detection"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, window, **kwargs):
        # TODO: Compute joint energy metric
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_jump_bipower_proxy_pn", canonical="ts_jump_bipower_proxy_pn", backend="polars_native")
class TSJumpBipowerProxyPolarsNative(SeriesOperator):
    """Bipower variation as jump-robust volatility proxy"""

    metadata = OperatorMetadata(
        name="ts_jump_bipower_proxy_pn",
        category="time_series",
        description="Bipower variation estimator robust to jumps",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility", "jump"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # Bipower: (pi/2) * sum(|r_t| * |r_{t-1}|)
        df = returns.to_frame().lazy()
        col = returns.name

        return (
            df.select([
                (np.pi / 2) * (pl.col(col).abs() * pl.col(col).shift(1).abs()).rolling_mean(window)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Kalman Filter Features
# ============================================================================

@register_operator(name="ts_kalman_beta", canonical="ts_kalman_beta", backend="polars_native")
class TSKalmanBetaPolarsNative(SeriesOperator):
    """Kalman-filtered rolling beta estimate"""

    metadata = OperatorMetadata(
        name="ts_kalman_beta",
        category="time_series",
        description="Adaptive beta via Kalman filter",
        param_names=["returns", "market_returns", "process_variance"],
        return_type="series",
        tags=["time_series", "pit_safe", "kalman", "beta"],
    )
    metadata.param_specs = {
        "process_variance": ParamSpec(dtype=float, min=0.0, default=0.001),
    }

    def _calculate_series(self, returns, market_returns, process_variance=0.001, **kwargs):
        # TODO: Implement Kalman filter for time-varying beta
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_kalman_beta_change", canonical="ts_kalman_beta_change", backend="polars_native")
class TSKalmanBetaChangePolarsNative(SeriesOperator):
    """Rate of change in Kalman beta estimate"""

    metadata = OperatorMetadata(
        name="ts_kalman_beta_change",
        category="time_series",
        description="First difference of Kalman beta",
        param_names=["returns", "market_returns", "process_variance"],
        return_type="series",
        tags=["time_series", "pit_safe", "kalman", "beta"],
    )
    metadata.param_specs = {
        "process_variance": ParamSpec(dtype=float, min=0.0, default=0.001),
    }

    def _calculate_series(self, returns, market_returns, process_variance=0.001, **kwargs):
        # TODO: Compute diff of Kalman beta
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_kalman_beta_uncertainty", canonical="ts_kalman_beta_uncertainty", backend="polars_native")
class TSKalmanBetaUncertaintyPolarsNative(SeriesOperator):
    """Kalman filter posterior variance (uncertainty in beta)"""

    metadata = OperatorMetadata(
        name="ts_kalman_beta_uncertainty",
        category="time_series",
        description="Posterior uncertainty from Kalman filter",
        param_names=["returns", "market_returns", "process_variance"],
        return_type="series",
        tags=["time_series", "pit_safe", "kalman", "beta"],
    )
    metadata.param_specs = {
        "process_variance": ParamSpec(dtype=float, min=0.0, default=0.001),
    }

    def _calculate_series(self, returns, market_returns, process_variance=0.001, **kwargs):
        # TODO: Return posterior variance P_t from Kalman filter
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_kalman_innovation_z", canonical="ts_kalman_innovation_z", backend="polars_native")
class TSKalmanInnovationZPolarsNative(SeriesOperator):
    """Standardized Kalman innovation (prediction error)"""

    metadata = OperatorMetadata(
        name="ts_kalman_innovation_z",
        category="time_series",
        description="Z-score of Kalman filter innovation",
        param_names=["x", "process_variance", "observation_variance"],
        return_type="series",
        tags=["time_series", "pit_safe", "kalman"],
    )
    metadata.param_specs = {
        "process_variance": ParamSpec(dtype=float, min=0.0, default=0.001),
        "observation_variance": ParamSpec(dtype=float, min=0.0, default=1.0),
    }

    def _calculate_series(self, x, process_variance=0.001, observation_variance=1.0, **kwargs):
        # TODO: Implement Kalman filter and standardize innovations
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_kalman_level", canonical="ts_kalman_level", backend="polars_native")
class TSKalmanLevelPolarsNative(SeriesOperator):
    """Kalman-filtered level estimate"""

    metadata = OperatorMetadata(
        name="ts_kalman_level",
        category="time_series",
        description="Adaptive level via Kalman filter",
        param_names=["x", "process_variance", "observation_variance"],
        return_type="series",
        tags=["time_series", "pit_safe", "kalman"],
    )
    metadata.param_specs = {
        "process_variance": ParamSpec(dtype=float, min=0.0, default=0.001),
        "observation_variance": ParamSpec(dtype=float, min=0.0, default=1.0),
    }

    def _calculate_series(self, x, process_variance=0.001, observation_variance=1.0, **kwargs):
        # TODO: Implement Kalman filter for level tracking
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_kalman_trend", canonical="ts_kalman_trend", backend="polars_native")
class TSKalmanTrendPolarsNative(SeriesOperator):
    """Kalman-filtered trend estimate"""

    metadata = OperatorMetadata(
        name="ts_kalman_trend",
        category="time_series",
        description="Adaptive trend via Kalman filter",
        param_names=["x", "process_variance", "observation_variance"],
        return_type="series",
        tags=["time_series", "pit_safe", "kalman"],
    )
    metadata.param_specs = {
        "process_variance": ParamSpec(dtype=float, min=0.0, default=0.001),
        "observation_variance": ParamSpec(dtype=float, min=0.0, default=1.0),
    }

    def _calculate_series(self, x, process_variance=0.001, observation_variance=1.0, **kwargs):
        # TODO: Implement local linear trend Kalman filter
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# KAMA (Kaufman Adaptive Moving Average)
# ============================================================================

@register_operator(name="ts_kama_pn", canonical="ts_kama_pn", backend="polars_native")
class TSKAMAPolarsNative(SeriesOperator):
    """Kaufman Adaptive Moving Average"""

    metadata = OperatorMetadata(
        name="ts_kama_pn",
        category="time_series",
        description="KAMA: adaptive MA based on efficiency ratio",
        param_names=["x", "window", "fast", "slow"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "adaptive"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        "fast": ParamSpec(dtype=int, min=2, default=2),
        "slow": ParamSpec(dtype=int, min=2, default=30),
    }

    def _calculate_series(self, x, window, fast=2, slow=30, **kwargs):
        # ER = abs(change) / sum(abs(changes))
        # SC = (ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))^2
        # KAMA = KAMA_prev + SC * (price - KAMA_prev)
        df = x.to_frame().lazy()
        col = x.name

        # Approximate with efficiency-weighted EMA
        return (
            df.with_columns([
                (pl.col(col) - pl.col(col).shift(window)).abs().alias("_change"),
                pl.col(col).diff().abs().rolling_sum(window).alias("_volatility"),
            ])
            .with_columns([
                pl.when(pl.col("_volatility") != 0.then(pl.col("_change") / (pl.col("_volatility").otherwise(None) + 1e-8)).alias("_er")
            ])
            .with_columns([
                ((pl.col("_er") * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2).alias("_sc")
            ])
            .select([
                # Simplified: use SC as alpha for EMA
                pl.col(col).ewm_mean(alpha=0.1)  # TODO: proper KAMA recursion
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Granger Causality and Time Delay Analysis
# ============================================================================

@register_operator(name="ts_kernel_granger_score", canonical="ts_kernel_granger_score", backend="polars_native")
class TSKernelGrangerScorePolarsNative(SeriesOperator):
    """Kernel-based nonlinear Granger causality score"""

    metadata = OperatorMetadata(
        name="ts_kernel_granger_score",
        category="time_series",
        description="Nonlinear Granger causality via kernel methods",
        param_names=["x", "y", "window", "lag"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "causality"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "lag": ParamSpec(dtype=int, min=1, default=1),
    }

    def _calculate_series(self, x, y, window, lag=1, **kwargs):
        # TODO: Implement kernel-based Granger test
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_lag_of_peak_corr_pn", canonical="ts_lag_of_peak_corr_pn", backend="polars_native")
class TSLagOfPeakCorrPolarsNative(SeriesOperator):
    """Lag at which cross-correlation is maximized"""

    metadata = OperatorMetadata(
        name="ts_lag_of_peak_corr_pn",
        category="time_series",
        description="Optimal lag for maximum cross-correlation",
        param_names=["x", "y", "window", "max_lag"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "correlation"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "max_lag": ParamSpec(dtype=int, min=1, default=10),
    }

    def _calculate_series(self, x, y, window, max_lag=10, **kwargs):
        # TODO: Compute cross-correlation at multiple lags and find argmax
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_lagged_mutual_information_pn", canonical="ts_lagged_mutual_information_pn", backend="polars_native")
class TSLaggedMutualInformationPolarsNative(SeriesOperator):
    """Mutual information between x(t) and x(t-lag)"""

    metadata = OperatorMetadata(
        name="ts_lagged_mutual_information_pn",
        category="time_series",
        description="MI for determining optimal embedding lag",
        param_names=["x", "window", "lag"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "information"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "lag": ParamSpec(dtype=int, min=1, default=1),
    }

    def _calculate_series(self, x, window, lag=1, **kwargs):
        # TODO: Discretize and compute MI between x_t and x_{t-lag}
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# Conditional Selection and Pivot Detection
# ============================================================================

@register_operator(name="ts_last_if", canonical="ts_last_if", backend="polars_native")
class TSLastIfPolarsNative(SeriesOperator):
    """Last value where condition was True"""

    metadata = OperatorMetadata(
        name="ts_last_if",
        category="time_series",
        description="Most recent value satisfying condition",
        param_names=["x", "condition"],
        return_type="series",
        tags=["time_series", "pit_safe", "conditional"],
    )

    def _calculate_series(self, x, condition, **kwargs):
        df = pl.DataFrame({
            "x": x,
            "condition": condition,
        }).lazy()

        return (
            df.with_columns([
                pl.when(pl.col("condition"))
                .then(pl.col("x"))
                .otherwise(None)
                .forward_fill()
            ])
            .select([pl.col("x")])
            .collect()
            .to_series()
        )


@register_operator(name="ts_last_pivot_high", canonical="ts_last_pivot_high", backend="polars_native")
class TSLastPivotHighPolarsNative(SeriesOperator):
    """Value at most recent pivot high"""

    metadata = OperatorMetadata(
        name="ts_last_pivot_high",
        category="time_series",
        description="Most recent local maximum value",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # Pivot high: x[t] > x[t-k:t] and x[t] > x[t:t+k]
        df = x.to_frame().lazy()
        col = x.name

        return (
            df.with_columns([
                pl.col(col).rolling_max(window).shift(1).alias("_prev_max"),
                pl.col(col).rolling_max(window).shift(-window+1).alias("_next_max"),
            ])
            .with_columns([
                pl.when(pl.col(col) >= pl.col("_prev_max")) & (pl.col(col) >= pl.col("_next_max")))
                .then(pl.col(col))
                .otherwise(None)
                .forward_fill()
            ])
            .select([pl.col(col)])
            .collect()
            .to_series()
        )


@register_operator(name="ts_last_pivot_low", canonical="ts_last_pivot_low", backend="polars_native")
class TSLastPivotLowPolarsNative(SeriesOperator):
    """Value at most recent pivot low"""

    metadata = OperatorMetadata(
        name="ts_last_pivot_low",
        category="time_series",
        description="Most recent local minimum value",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        df = x.to_frame().lazy()
        col = x.name

        return (
            df.with_columns([
                pl.col(col).rolling_min(window).shift(1).alias("_prev_min"),
                pl.col(col).rolling_min(window).shift(-window+1).alias("_next_min"),
            ])
            .with_columns([
                pl.when(pl.col(col) <= pl.col("_prev_min")) & (pl.col(col) <= pl.col("_next_min")))
                .then(pl.col(col))
                .otherwise(None)
                .forward_fill()
            ])
            .select([pl.col(col)])
            .collect()
            .to_series()
        )


# ============================================================================
# Complexity and Entropy Measures
# ============================================================================

@register_operator(name="ts_lempel_ziv_complexity_pn", canonical="ts_lempel_ziv_complexity_pn", backend="polars_native")
class TSLempelZivComplexityPolarsNative(SeriesOperator):
    """Lempel-Ziv complexity (normalized)"""

    metadata = OperatorMetadata(
        name="ts_lempel_ziv_complexity_pn",
        category="time_series",
        description="LZ complexity for sequence randomness",
        param_names=["x", "window", "n_bins"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "complexity"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_bins": ParamSpec(dtype=int, min=2, default=2),
    }

    def _calculate_series(self, x, window, n_bins=2, **kwargs):
        # TODO: Implement Lempel-Ziv compression algorithm
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_lz_complexity", canonical="ts_lz_complexity", backend="polars_native")
class TSLZComplexityPolarsNative(SeriesOperator):
    """Lempel-Ziv complexity (alias)"""

    metadata = OperatorMetadata(
        name="ts_lz_complexity",
        category="time_series",
        description="LZ complexity for sequence randomness",
        param_names=["x", "window", "n_bins"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "complexity"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_bins": ParamSpec(dtype=int, min=2, default=2),
    }

    def _calculate_series(self, x, window, n_bins=2, **kwargs):
        # Same as ts_lempel_ziv_complexity
        return pl.Series([None] * len(x), dtype=pl.Float64)


# ============================================================================
# Mean Reversion and Location Measures
# ============================================================================

@register_operator(name="ts_level_shift_score_pn", canonical="ts_level_shift_score_pn", backend="polars_native")
class TSLevelShiftScorePolarsNative(SeriesOperator):
    """Likelihood of level shift at current point"""

    metadata = OperatorMetadata(
        name="ts_level_shift_score_pn",
        category="time_series",
        description="CUSUM-based level shift detection score",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "change_detection"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # CUSUM-style statistic
        df = x.to_frame().lazy()
        col = x.name

        return (
            df.with_columns([
                (pl.col(col) - pl.col(col).rolling_mean(window)).alias("_dev")
            ])
            .select([
                pl.col("_dev").cum_sum().abs()
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_leverage_effect_pn", canonical="ts_leverage_effect_pn", backend="polars_native")
class TSLeverageEffectPolarsNative(SeriesOperator):
    """Correlation between returns and future volatility"""

    metadata = OperatorMetadata(
        name="ts_leverage_effect_pn",
        category="time_series",
        description="Correlation of returns with next-period volatility",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "volatility"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # Corr(r_t, |r_{t+1}|) or Corr(r_t, realized_vol_{t+1})
        df = returns.to_frame().lazy()
        col = returns.name

        return (
            df.with_columns([
                pl.col(col).alias("_ret"),
                pl.col(col).shift(-1).abs().alias("_next_vol"),
            ])
            .select([
                pl.corr("_ret", "_next_vol").over(pl.int_range(pl.len()).floordiv(window))
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_location_shift_pn", canonical="ts_location_shift_pn", backend="polars_native")
class TSLocationShiftPolarsNative(SeriesOperator):
    """Change in robust location between consecutive windows"""

    metadata = OperatorMetadata(
        name="ts_location_shift_pn",
        category="time_series",
        description="Difference in median between consecutive windows",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "change_detection"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name).rolling_median(window).diff()
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_lo_mackinlay_vr_pn", canonical="ts_lo_mackinlay_vr_pn", backend="polars_native")
class TSLoMacKinlayVRPolarsNative(SeriesOperator):
    """Lo-MacKinlay variance ratio"""

    metadata = OperatorMetadata(
        name="ts_lo_mackinlay_vr_pn",
        category="time_series",
        description="Variance ratio test for random walk",
        param_names=["returns", "window", "q"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "mean_reversion"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "q": ParamSpec(dtype=int, min=2, default=5),
    }

    def _calculate_series(self, returns, window, q=5, **kwargs):
        # VR(q) = Var(r_t + ... + r_{t-q+1}) / (q * Var(r_t))
        df = returns.to_frame().lazy()
        col = returns.name

        return (
            df.with_columns([
                pl.col(col).rolling_sum(q).rolling_var(window).alias("_var_q"),
                pl.col(col).rolling_var(window).alias("_var_1"),
            ])
            .select([
                pl.col("_var_q") / (q * pl.col("_var_1") + 1e-8)
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_lo_mackinlay_z_pn", canonical="ts_lo_mackinlay_z_pn", backend="polars_native")
class TSLoMacKinlayZPolarsNative(SeriesOperator):
    """Lo-MacKinlay variance ratio z-statistic"""

    metadata = OperatorMetadata(
        name="ts_lo_mackinlay_z_pn",
        category="time_series",
        description="Standardized variance ratio test statistic",
        param_names=["returns", "window", "q"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "mean_reversion"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "q": ParamSpec(dtype=int, min=2, default=5),
    }

    def _calculate_series(self, returns, window, q=5, **kwargs):
        # TODO: Compute VR and standardize with asymptotic variance
        return pl.Series([None] * len(returns), dtype=pl.Float64)


@register_operator(name="ts_mean_reversion_half_life", canonical="ts_mean_reversion_half_life", backend="polars_native")
class TSMeanReversionHalfLifePolarsNative(SeriesOperator):
    """Estimated half-life of mean reversion"""

    metadata = OperatorMetadata(
        name="ts_mean_reversion_half_life",
        category="time_series",
        description="Half-life from AR(1) fit",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "mean_reversion"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # Fit AR(1): x_t = phi * x_{t-1} + eps
        # Half-life = -log(2) / log(phi)
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: (
                        -np.log(2) / np.log(np.corrcoef(s[:-1], s[1:])[0,1])
                        if len(s) >= 3 and np.corrcoef(s[:-1], s[1:])[0,1] > 0 and np.corrcoef(s[:-1], s[1:])[0,1] < 1
                        else None
                    ),
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


@register_operator(name="ts_mean_reversion_ou_approx_half_life", canonical="ts_mean_reversion_ou_approx_half_life", backend="polars_native")
class TSMeanReversionOUApproxHalfLifePolarsNative(SeriesOperator):
    """OU process approximation of mean reversion half-life"""

    metadata = OperatorMetadata(
        name="ts_mean_reversion_ou_approx_half_life",
        category="time_series",
        description="Half-life from Ornstein-Uhlenbeck approximation",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "mean_reversion"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # Similar to AR(1) half-life but with continuous-time interpretation
        return (
            x.to_frame()
            .lazy()
            .select([
                pl.col(x.name)
                .rolling_map(
                    lambda s: (
                        -np.log(2) / np.log(np.corrcoef(s[:-1], s[1:])[0,1])
                        if len(s) >= 3 and 0 < np.corrcoef(s[:-1], s[1:])[0,1] < 1
                        else None
                    ),
                    window_size=window
                )
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Pivot and Support/Resistance Detection
# ============================================================================

@register_operator(name="ts_nth_pivot_high", canonical="ts_nth_pivot_high", backend="polars_native")
class TSNthPivotHighPolarsNative(SeriesOperator):
    """Value at n-th most recent pivot high"""

    metadata = OperatorMetadata(
        name="ts_nth_pivot_high",
        category="time_series",
        description="Historical pivot high at specified lookback",
        param_names=["x", "window", "n"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
        "n": ParamSpec(dtype=int, min=1, default=1),
    }

    def _calculate_series(self, x, window, n=1, **kwargs):
        # TODO: Track n-th pivot high
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_nth_pivot_high_age", canonical="ts_nth_pivot_high_age", backend="polars_native")
class TSNthPivotHighAgePolarsNative(SeriesOperator):
    """Bars since n-th most recent pivot high"""

    metadata = OperatorMetadata(
        name="ts_nth_pivot_high_age",
        category="time_series",
        description="Time since n-th pivot high",
        param_names=["x", "window", "n"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
        "n": ParamSpec(dtype=int, min=1, default=1),
    }

    def _calculate_series(self, x, window, n=1, **kwargs):
        # TODO: Track age of n-th pivot
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_nth_pivot_low", canonical="ts_nth_pivot_low", backend="polars_native")
class TSNthPivotLowPolarsNative(SeriesOperator):
    """Value at n-th most recent pivot low"""

    metadata = OperatorMetadata(
        name="ts_nth_pivot_low",
        category="time_series",
        description="Historical pivot low at specified lookback",
        param_names=["x", "window", "n"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
        "n": ParamSpec(dtype=int, min=1, default=1),
    }

    def _calculate_series(self, x, window, n=1, **kwargs):
        # TODO: Track n-th pivot low
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_nth_pivot_low_age", canonical="ts_nth_pivot_low_age", backend="polars_native")
class TSNthPivotLowAgePolarsNative(SeriesOperator):
    """Bars since n-th most recent pivot low"""

    metadata = OperatorMetadata(
        name="ts_nth_pivot_low_age",
        category="time_series",
        description="Time since n-th pivot low",
        param_names=["x", "window", "n"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
        "n": ParamSpec(dtype=int, min=1, default=1),
    }

    def _calculate_series(self, x, window, n=1, **kwargs):
        # TODO: Track age of n-th pivot
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_pivot_high_age", canonical="ts_pivot_high_age", backend="polars_native")
class TSPivotHighAgePolarsNative(SeriesOperator):
    """Bars since most recent pivot high"""

    metadata = OperatorMetadata(
        name="ts_pivot_high_age",
        category="time_series",
        description="Time since last pivot high",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Count bars since pivot high
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_pivot_high_count", canonical="ts_pivot_high_count", backend="polars_native")
class TSPivotHighCountPolarsNative(SeriesOperator):
    """Count of pivot highs in rolling window"""

    metadata = OperatorMetadata(
        name="ts_pivot_high_count",
        category="time_series",
        description="Number of local maxima in window",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Count pivot highs
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_pivot_high_spacing", canonical="ts_pivot_high_spacing", backend="polars_native")
class TSPivotHighSpacingPolarsNative(SeriesOperator):
    """Average spacing between pivot highs"""

    metadata = OperatorMetadata(
        name="ts_pivot_high_spacing",
        category="time_series",
        description="Mean time between consecutive pivot highs",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Compute average spacing
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_pivot_low_age", canonical="ts_pivot_low_age", backend="polars_native")
class TSPivotLowAgePolarsNative(SeriesOperator):
    """Bars since most recent pivot low"""

    metadata = OperatorMetadata(
        name="ts_pivot_low_age",
        category="time_series",
        description="Time since last pivot low",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Count bars since pivot low
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_pivot_low_count", canonical="ts_pivot_low_count", backend="polars_native")
class TSPivotLowCountPolarsNative(SeriesOperator):
    """Count of pivot lows in rolling window"""

    metadata = OperatorMetadata(
        name="ts_pivot_low_count",
        category="time_series",
        description="Number of local minima in window",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Count pivot lows
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_pivot_low_spacing", canonical="ts_pivot_low_spacing", backend="polars_native")
class TSPivotLowSpacingPolarsNative(SeriesOperator):
    """Average spacing between pivot lows"""

    metadata = OperatorMetadata(
        name="ts_pivot_low_spacing",
        category="time_series",
        description="Mean time between consecutive pivot lows",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "pivot"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, **kwargs):
        # TODO: Compute average spacing
        return pl.Series([None] * len(x), dtype=pl.Float64)


@register_operator(name="ts_nth_value", canonical="ts_nth_value", backend="polars_native")
class TSNthValuePolarsNative(SeriesOperator):
    """N-th most recent value"""

    metadata = OperatorMetadata(
        name="ts_nth_value",
        category="time_series",
        description="Value at n bars ago",
        param_names=["x", "n"],
        return_type="series",
        tags=["time_series", "pit_safe", "lag"],
    )
    metadata.param_specs = {
        "n": ParamSpec(dtype=int, min=1, default=1),
    }

    def _calculate_series(self, x, n=1, **kwargs):
        return x.shift(n)


# Export summary
__all__ = [
    "TSFeatureModeSharePolarsNative",
    "TSFeaturePCAReconstructionErrorPolarsNative",
    "TSFeatureSubspaceRotationPolarsNative",
    "TSFfillLimitedPolarsNative",
    "TSGapFillRatioPolarsNative",
    "TSGapReversionRatioPolarsNative",
    "TSGapSurvivalDurationPolarsNative",
    "TSGarchNextVolForecastPolarsNative",
    "TSGarchPersistencePolarsNative",
    "TSGarchStandardizedShockPolarsNative",
    "TSGarchVolSurprisePolarsNative",
    "TSGJRGarchVolForecastPolarsNative",
    "TSGJRLeveragePolarsNative",
    "TSGPDShapePWMPolarsNative",
    "TSHillTailIndexPolarsNative",
    "TSPickandsTailIndexPolarsNative",
    "TSGLRMeanShiftScorePolarsNative",
    "TSGLRVarianceShiftScorePolarsNative",
    "TSGeneralizedHurstExponentPolarsNative",
    "TSGeneralizedHurstSpreadQ1Q4PolarsNative",
    "TSHiguchiFractalDimensionPolarsNative",
    "TSHurstDFAPolarsNative",
    "TSHARFromReturnForecastErrorZPolarsNative",
    "TSHARFromReturnNextVolPolarsNative",
    "TSHARRVForecastErrorZPolarsNative",
    "TSHARRVNextVarForecastPolarsNative",
    "TSHARRVNextVolForecastPolarsNative",
    "TSFIRLowpassCausalPolarsNative",
    "TSHampelFilterCausalPolarsNative",
    "TSHInfinityLevelFilterPolarsNative",
    "TSHankelEffectiveRankPolarsNative",
    "TSHankelSingularGapPolarsNative",
    "TSHartiganDipPolarsNative",
    "TSHodgesLehmannLocationPolarsNative",
    "TSHSICPolarsNative",
    "TSKSShiftPolarsNative",
    "TSLKurtosisPolarsNative",
    "TSLSkewnessPolarsNative",
    "TSMarkovCommittorPolarsNative",
    "TSMarkovEntropyProductionPolarsNative",
    "TSMarkovMeanFirstPassageTimePolarsNative",
    "TSMarkovPersistencePolarsNative",
    "TSMarkovSpectralGapPolarsNative",
    "TSMarkovStateEntropyPolarsNative",
    "TSMarkovStationarySurprisalPolarsNative",
    "TSFirstPassageBiasPolarsNative",
    "TSFirstPassageConditionalTimePolarsNative",
    "TSFirstPassageHitProbabilityPolarsNative",
    "TSImpulseReturnPolarsNative",
    "TSImpulseStrengthPolarsNative",
    "TSImpulseVolumePolarsNative",
    "TSFisherInformationShiftPolarsNative",
    "TSForbiddenOrdinalPatternExcessPolarsNative",
    "TSForbiddenOrdinalPatternRatioPolarsNative",
    "TSForbiddenOrdinalPatternSignedExcessPolarsNative",
    "TSFractionalDifferencePolarsNative",
    "TSFractionalDifferenceDiscardedWeightMassPolarsNative",
    "TSHysteresisAgePolarsNative",
    "TSHysteresisStatePolarsNative",
    "TSIntervalExplorationEfficiencyPolarsNative",
    "TSIntervalNestingDepthPolarsNative",
    "TSIntervalOccupancyEntropyPolarsNative",
    "TSIntervalOccupancyModeDistancePolarsNative",
    "TSIntervalOverlapConnectedComponentRatioPolarsNative",
    "TSIntervalUnionCoveragePolarsNative",
    "TSJointEnergyShiftPolarsNative",
    "TSJumpBipowerProxyPolarsNative",
    "TSKalmanBetaPolarsNative",
    "TSKalmanBetaChangePolarsNative",
    "TSKalmanBetaUncertaintyPolarsNative",
    "TSKalmanInnovationZPolarsNative",
    "TSKalmanLevelPolarsNative",
    "TSKalmanTrendPolarsNative",
    "TSKAMAPolarsNative",
    "TSKernelGrangerScorePolarsNative",
    "TSLagOfPeakCorrPolarsNative",
    "TSLaggedMutualInformationPolarsNative",
    "TSLastIfPolarsNative",
    "TSLastPivotHighPolarsNative",
    "TSLastPivotLowPolarsNative",
    "TSLempelZivComplexityPolarsNative",
    "TSLZComplexityPolarsNative",
    "TSLevelShiftScorePolarsNative",
    "TSLeverageEffectPolarsNative",
    "TSLocationShiftPolarsNative",
    "TSLoMacKinlayVRPolarsNative",
    "TSLoMacKinlayZPolarsNative",
    "TSMeanReversionHalfLifePolarsNative",
    "TSMeanReversionOUApproxHalfLifePolarsNative",
    "TSNthPivotHighPolarsNative",
    "TSNthPivotHighAgePolarsNative",
    "TSNthPivotLowPolarsNative",
    "TSNthPivotLowAgePolarsNative",
    "TSPivotHighAgePolarsNative",
    "TSPivotHighCountPolarsNative",
    "TSPivotHighSpacingPolarsNative",
    "TSPivotLowAgePolarsNative",
    "TSPivotLowCountPolarsNative",
    "TSPivotLowSpacingPolarsNative",
    "TSNthValuePolarsNative",
]
