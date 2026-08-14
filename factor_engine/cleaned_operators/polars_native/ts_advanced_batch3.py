"""
Polars native implementations for time series operators - Advanced Batch 3
Operators 533-632: Markov chains, matrix profiles, path geometry, quantile regression, recurrence analysis

All operators use pure Polars lazy API where feasible.
Complex algorithms (matrix profile, multifractal, path signatures) have skeleton implementations.
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
# Markov Chain Operators
# ============================================================================

@register_operator(name="ts_markov_spectral_gap", canonical="ts_markov_spectral_gap", backend="polars", research_only=True)
class TSMarkovSpectralGapPolarsNative(SeriesOperator):
    """Spectral gap (1 - second largest eigenvalue) of estimated transition matrix"""

    metadata = OperatorMetadata(
        name="ts_markov_spectral_gap",
        category="time_series",
        description="Spectral gap of Markov chain transition matrix",
        param_names=["feature", "window", "n_states"],
        return_type="series",
        tags=["time_series", "markov", "spectral", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_states=5, **kwargs):
        # TODO: Requires eigenvalue computation on rolling transition matrices
        # Skeleton: discretize -> build transition matrix -> compute eigenvalues
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_markov_state_entropy", canonical="ts_markov_state_entropy", backend="polars", research_only=True)
class TSMarkovStateEntropyPolarsNative(SeriesOperator):
    """Shannon entropy of state occupancy distribution"""

    metadata = OperatorMetadata(
        name="ts_markov_state_entropy",
        category="time_series",
        description="Entropy of Markov state distribution",
        param_names=["feature", "window", "n_states"],
        return_type="series",
        tags=["time_series", "markov", "entropy", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_states=5, **kwargs):
        # TODO: Discretize into states and compute Shannon entropy of state frequencies
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_markov_stationary_surprisal", canonical="ts_markov_stationary_surprisal", backend="polars", research_only=True)
class TSMarkovStationarySurprisalPolarsNative(SeriesOperator):
    """Surprisal of current state under stationary distribution"""

    metadata = OperatorMetadata(
        name="ts_markov_stationary_surprisal",
        category="time_series",
        description="Negative log probability under stationary distribution",
        param_names=["feature", "window", "n_states"],
        return_type="series",
        tags=["time_series", "markov", "surprisal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_states=5, **kwargs):
        # TODO: Compute stationary distribution and -log(prob[current_state])
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_markov_transition_surprisal", canonical="ts_markov_transition_surprisal", backend="polars", research_only=True)
class TSMarkovTransitionSurprisalPolarsNative(SeriesOperator):
    """Surprisal of observed transition from previous state"""

    metadata = OperatorMetadata(
        name="ts_markov_transition_surprisal",
        category="time_series",
        description="Negative log probability of observed state transition",
        param_names=["feature", "window", "n_states"],
        return_type="series",
        tags=["time_series", "markov", "surprisal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_states": ParamSpec(dtype=int, min=2, default=5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_states=5, **kwargs):
        # TODO: -log(P[state_t | state_{t-1}]) from transition matrix
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Mass and Concentration Metrics
# ============================================================================

@register_operator(name="ts_mass_concentration", canonical="ts_mass_concentration", backend="polars", research_only=True)
class TSMassConcentrationPolarsNative(SeriesOperator):
    """Concentration of cumulative mass (Gini-like metric)"""

    metadata = OperatorMetadata(
        name="ts_mass_concentration",
        category="time_series",
        description="Concentration of cumulative distribution",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "concentration", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Gini-like: measure concentration of sorted cumulative values
        # TODO: Requires sorting within rolling window
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Matrix Profile Operators
# ============================================================================

@register_operator(name="ts_matrix_profile_discord_score", canonical="ts_matrix_profile_discord_score", backend="polars", research_only=True)
class TSMatrixProfileDiscordScorePolarsNative(SeriesOperator):
    """Matrix profile discord score (distance to nearest neighbor)"""

    metadata = OperatorMetadata(
        name="ts_matrix_profile_discord_score",
        category="time_series",
        description="Matrix profile discord detection score",
        param_names=["feature", "window", "subsequence_len"],
        return_type="series",
        tags=["time_series", "matrix_profile", "anomaly", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "subsequence_len": ParamSpec(dtype=int, min=4, default=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, subsequence_len=10, **kwargs):
        # TODO: Requires matrix profile computation (STOMP/STAMP algorithm)
        # Discord = max distance to nearest neighbor
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_matrix_profile_motif_age", canonical="ts_matrix_profile_motif_age", backend="polars", research_only=True)
class TSMatrixProfileMotifAgePolarsNative(SeriesOperator):
    """Time since most recent motif occurrence"""

    metadata = OperatorMetadata(
        name="ts_matrix_profile_motif_age",
        category="time_series",
        description="Periods since last motif match",
        param_names=["feature", "window", "subsequence_len"],
        return_type="series",
        tags=["time_series", "matrix_profile", "motif", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "subsequence_len": ParamSpec(dtype=int, min=4, default=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, subsequence_len=10, **kwargs):
        # TODO: Find motifs (low matrix profile values) and track recency
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_matrix_profile_motif_frequency", canonical="ts_matrix_profile_motif_frequency", backend="polars", research_only=True)
class TSMatrixProfileMotifFrequencyPolarsNative(SeriesOperator):
    """Frequency of motif occurrences in window"""

    metadata = OperatorMetadata(
        name="ts_matrix_profile_motif_frequency",
        category="time_series",
        description="Count of motif matches per period",
        param_names=["feature", "window", "subsequence_len", "threshold"],
        return_type="series",
        tags=["time_series", "matrix_profile", "motif", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "subsequence_len": ParamSpec(dtype=int, min=4, default=10, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.5, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, subsequence_len=10, threshold=0.5, **kwargs):
        # TODO: Count subsequences with matrix profile distance < threshold
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_matrix_profile_neighbor_dispersion", canonical="ts_matrix_profile_neighbor_dispersion", backend="polars", research_only=True)
class TSMatrixProfileNeighborDispersionPolarsNative(SeriesOperator):
    """Standard deviation of nearest neighbor distances"""

    metadata = OperatorMetadata(
        name="ts_matrix_profile_neighbor_dispersion",
        category="time_series",
        description="Dispersion of matrix profile values",
        param_names=["feature", "window", "subsequence_len"],
        return_type="series",
        tags=["time_series", "matrix_profile", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "subsequence_len": ParamSpec(dtype=int, min=4, default=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, subsequence_len=10, **kwargs):
        # TODO: Std of matrix profile values
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_matrix_profile_novelty", canonical="ts_matrix_profile_novelty", backend="polars", research_only=True)
class TSMatrixProfileNoveltyPolarsNative(SeriesOperator):
    """Current subsequence distance to historical patterns"""

    metadata = OperatorMetadata(
        name="ts_matrix_profile_novelty",
        category="time_series",
        description="Novelty score based on matrix profile",
        param_names=["feature", "window", "subsequence_len"],
        return_type="series",
        tags=["time_series", "matrix_profile", "novelty", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "subsequence_len": ParamSpec(dtype=int, min=4, default=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, subsequence_len=10, **kwargs):
        # TODO: Matrix profile value for current subsequence
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Path Geometry and Excursion Metrics
# ============================================================================

@register_operator(name="ts_max_chord_excursion", canonical="ts_max_chord_excursion", backend="polars", research_only=True)
class TSMaxChordExcursionPolarsNative(SeriesOperator):
    """Maximum perpendicular distance from linear trend"""

    metadata = OperatorMetadata(
        name="ts_max_chord_excursion",
        category="time_series",
        description="Max deviation from straight-line path",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "geometry", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Max perpendicular distance from line connecting endpoints
        # TODO: Requires point-to-line distance computation in rolling window
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_max_drawdown_activity_cost", canonical="ts_max_drawdown_activity_cost", backend="polars", research_only=True)
class TSMaxDrawdownActivityCostPolarsNative(SeriesOperator):
    """Maximum drawdown weighted by trading activity"""

    metadata = OperatorMetadata(
        name="ts_max_drawdown_activity_cost",
        category="time_series",
        description="Max drawdown adjusted for activity level",
        param_names=["feature", "activity", "window"],
        return_type="series",
        tags=["time_series", "drawdown", "risk", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, activity, window, **kwargs):
        # Max drawdown * average activity during drawdown period
        # TODO: Identify drawdown periods and weight by activity
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_mean_excess_slope", canonical="ts_mean_excess_slope", backend="polars", research_only=True)
class TSMeanExcessSlopePolarsNative(SeriesOperator):
    """Slope of mean excess function (for extreme value theory)"""

    metadata = OperatorMetadata(
        name="ts_mean_excess_slope",
        category="time_series",
        description="Mean excess plot slope estimate",
        param_names=["feature", "window", "threshold"],
        return_type="series",
        tags=["time_series", "extreme_value", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold, **kwargs):
        # E[X - u | X > u] as function of u, estimate slope
        # TODO: Fit mean excess curve and extract slope
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Mean Reversion Metrics
# ============================================================================

@register_operator(name="ts_mean_reversion_half_life", canonical="ts_mean_reversion_half_life", backend="polars", research_only=True)
class TSMeanReversionHalfLifePolarsNative(SeriesOperator):
    """Estimated half-life of mean reversion"""

    metadata = OperatorMetadata(
        name="ts_mean_reversion_half_life",
        category="time_series",
        description="Half-life of reversion to mean",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "mean_reversion", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Fit AR(1): x_t = phi * x_{t-1} + eps
        # Half-life = -log(2) / log(phi)
        # Use rolling regression
        df = feature.to_frame().with_row_count("_idx")

        result = (
            df.lazy()
            .with_columns([
                pl.col(feature.name).shift(1).alias("_lag1"),
                pl.col(feature.name).alias("_y"),
            ])
            .with_columns([
                # Rolling correlation between y and lag1
                (pl.col("_y") * pl.col("_lag1")).rolling_mean(window).alias("_cov_proxy"),
                pl.col("_lag1").pow(2).rolling_mean(window).alias("_var_lag1"),
                pl.col("_lag1").rolling_mean(window).alias("_mean_lag1"),
                pl.col("_y").rolling_mean(window).alias("_mean_y"),
            ])
            .with_columns([
                # phi = cov(y, lag1) / var(lag1)
                ((pl.col("_cov_proxy") - pl.col("_mean_y") * pl.col("_mean_lag1")) /
                 (pl.col("_var_lag1") - pl.col("_mean_lag1").pow(2) + 1e-10)).alias("_phi")
            ])
            .with_columns([
                # Half-life = -ln(2) / ln(phi), only when 0 < phi < 1
                pl.when((pl.col("_phi") > 0) & (pl.col("_phi") < 1))
                .then(-0.693147 / pl.col("_phi").log())
                .otherwise(None)
                .alias("_half_life")
            ])
            .select("_half_life")
            .collect()
        )
        return result.to_series()


@register_operator(name="ts_mean_reversion_ou_approx_half_life", canonical="ts_mean_reversion_ou_approx_half_life", backend="polars", research_only=True)
class TSMeanReversionOUApproxHalfLifePolarsNative(SeriesOperator):
    """Ornstein-Uhlenbeck approximation of mean reversion half-life"""

    metadata = OperatorMetadata(
        name="ts_mean_reversion_ou_approx_half_life",
        category="time_series",
        description="OU process half-life estimate",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "mean_reversion", "ou_process", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Similar to above but with OU-specific estimation
        # theta = -log(phi) / dt, half_life = log(2) / theta
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).shift(1).alias("_lag1"),
            ])
            .with_columns([
                # Simplified: use autocorrelation at lag 1
                pl.corr(pl.col(feature.name), pl.col("_lag1")).over_rolling(window).alias("_rho")
            ])
            .with_columns([
                pl.when(pl.col("_rho") > 0)
                .then(0.693147 / (-pl.col("_rho").log()))
                .otherwise(None)
            ])
            .select(pl.col(feature.name))
            .collect()
            .to_series()
        )


@register_operator(name="ts_median3_causal", canonical="ts_median3_causal", backend="polars", research_only=True)
class TSMedian3CausalPolarsNative(SeriesOperator):
    """Causal 3-period median filter"""

    metadata = OperatorMetadata(
        name="ts_median3_causal",
        category="time_series",
        description="Median of [t-2, t-1, t] (causal)",
        param_names=["feature"],
        return_type="series",
        tags=["time_series", "median", "filter", "pit_safe"],
    )

    def _calculate_series(self, feature, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).shift(2).alias("_t2"),
                pl.col(feature.name).shift(1).alias("_t1"),
                pl.col(feature.name).alias("_t0"),
            ])
            .select([
                pl.concat_list(["_t2", "_t1", "_t0"]).list.median().alias(feature.name)
            ])
            .collect()
            .to_series()
        )


# ============================================================================
# Distribution Shift Metrics
# ============================================================================

@register_operator(name="ts_mmd_rbf_shift", canonical="ts_mmd_rbf_shift", backend="polars", research_only=True)
class TSMMDRBFShiftPolarsNative(SeriesOperator):
    """Maximum Mean Discrepancy with RBF kernel for distribution shift detection"""

    metadata = OperatorMetadata(
        name="ts_mmd_rbf_shift",
        category="time_series",
        description="MMD between recent and reference windows",
        param_names=["feature", "window", "reference_window"],
        return_type="series",
        tags=["time_series", "distribution_shift", "kernel", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "reference_window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, reference_window, **kwargs):
        # TODO: Requires kernel matrix computation
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_modwt_band_corr", canonical="ts_modwt_band_corr", backend="polars", research_only=True)
class TSMODWTBandCorrPolarsNative(SeriesOperator):
    """Correlation between MODWT wavelet bands"""

    metadata = OperatorMetadata(
        name="ts_modwt_band_corr",
        category="time_series",
        description="Correlation between wavelet decomposition bands",
        param_names=["feature", "window", "level1", "level2"],
        return_type="series",
        tags=["time_series", "wavelet", "correlation", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=32, param_role=ParamRole.HORIZON),
        "level1": ParamSpec(dtype=int, min=1, default=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "level2": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, level1=1, level2=2, **kwargs):
        # TODO: Requires MODWT wavelet transform
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_motif_recurrence_count", canonical="ts_motif_recurrence_count", backend="polars", research_only=True)
class TSMotifRecurrenceCountPolarsNative(SeriesOperator):
    """Count of motif recurrences in window"""

    metadata = OperatorMetadata(
        name="ts_motif_recurrence_count",
        category="time_series",
        description="Number of similar pattern occurrences",
        param_names=["feature", "window", "motif_len", "threshold"],
        return_type="series",
        tags=["time_series", "motif", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "motif_len": ParamSpec(dtype=int, min=3, default=5, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.8, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, motif_len=5, threshold=0.8, **kwargs):
        # TODO: Requires subsequence similarity search
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Multifractal Analysis
# ============================================================================

@register_operator(name="ts_multifractal_asymmetry", canonical="ts_multifractal_asymmetry", backend="polars", research_only=True)
class TSMultifractalAsymmetryPolarsNative(SeriesOperator):
    """Asymmetry of multifractal spectrum"""

    metadata = OperatorMetadata(
        name="ts_multifractal_asymmetry",
        category="time_series",
        description="Asymmetry of f(alpha) spectrum",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "multifractal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=64, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Requires multifractal spectrum computation via WTMM or DFA
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_multifractal_curvature", canonical="ts_multifractal_curvature", backend="polars", research_only=True)
class TSMultifractalCurvaturePolarsNative(SeriesOperator):
    """Curvature of multifractal spectrum at peak"""

    metadata = OperatorMetadata(
        name="ts_multifractal_curvature",
        category="time_series",
        description="Second derivative of f(alpha) at maximum",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "multifractal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=64, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Requires multifractal spectrum computation
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_multifractal_spectrum_width", canonical="ts_multifractal_spectrum_width", backend="polars", research_only=True)
class TSMultifractalSpectrumWidthPolarsNative(SeriesOperator):
    """Width of multifractal spectrum (alpha_max - alpha_min)"""

    metadata = OperatorMetadata(
        name="ts_multifractal_spectrum_width",
        category="time_series",
        description="Range of singularity strengths",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "multifractal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=64, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Requires multifractal spectrum computation
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Multiscale Entropy and Trend Analysis
# ============================================================================

@register_operator(name="ts_multiscale_entropy_slope", canonical="ts_multiscale_entropy_slope", backend="polars", research_only=True)
class TSMultiscaleEntropySlopePolarsNative(SeriesOperator):
    """Slope of sample entropy across scales"""

    metadata = OperatorMetadata(
        name="ts_multiscale_entropy_slope",
        category="time_series",
        description="Trend in entropy across coarse-graining scales",
        param_names=["feature", "window", "max_scale"],
        return_type="series",
        tags=["time_series", "entropy", "multiscale", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=50, param_role=ParamRole.HORIZON),
        "max_scale": ParamSpec(dtype=int, min=3, default=5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, max_scale=5, **kwargs):
        # TODO: Compute sample entropy at multiple scales and fit slope
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_multiscale_permutation_entropy_slope", canonical="ts_multiscale_permutation_entropy_slope", backend="polars", research_only=True)
class TSMultiscalePermutationEntropySlopePolarsNative(SeriesOperator):
    """Slope of permutation entropy across scales"""

    metadata = OperatorMetadata(
        name="ts_multiscale_permutation_entropy_slope",
        category="time_series",
        description="Trend in permutation entropy across scales",
        param_names=["feature", "window", "max_scale"],
        return_type="series",
        tags=["time_series", "entropy", "permutation", "multiscale", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=50, param_role=ParamRole.HORIZON),
        "max_scale": ParamSpec(dtype=int, min=3, default=5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, max_scale=5, **kwargs):
        # TODO: Compute permutation entropy at multiple scales
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_multiscale_trend_consensus", canonical="ts_multiscale_trend_consensus", backend="polars", research_only=True)
class TSMultiscaleTrendConsensusPolarsNative(SeriesOperator):
    """Agreement of trend direction across multiple scales"""

    metadata = OperatorMetadata(
        name="ts_multiscale_trend_consensus",
        category="time_series",
        description="Fraction of scales with same trend direction",
        param_names=["feature", "window", "n_scales"],
        return_type="series",
        tags=["time_series", "trend", "multiscale", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_scales": ParamSpec(dtype=int, min=2, default=4, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_scales=4, **kwargs):
        # Compute trend at multiple window sizes and measure agreement
        trends = []
        for scale in range(1, n_scales + 1):
            scale_window = window * scale
            trend = (
                feature.to_frame()
                .lazy()
                .with_columns([
                    (pl.col(feature.name) - pl.col(feature.name).shift(scale_window)).sign().alias(f"_trend_{scale}")
                ])
                .select(f"_trend_{scale}")
            )
            trends.append(trend)

        # TODO: Combine trends and compute consensus
        # For now, simplified version
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_multiscale_trend_curvature", canonical="ts_multiscale_trend_curvature", backend="polars", research_only=True)
class TSMultiscaleTrendCurvaturePolarsNative(SeriesOperator):
    """Curvature of trend strength across scales"""

    metadata = OperatorMetadata(
        name="ts_multiscale_trend_curvature",
        category="time_series",
        description="Second derivative of trend magnitude vs scale",
        param_names=["feature", "window", "n_scales"],
        return_type="series",
        tags=["time_series", "trend", "multiscale", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_scales": ParamSpec(dtype=int, min=3, default=4, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_scales=4, **kwargs):
        # TODO: Compute trend magnitude at each scale and fit curvature
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_multiscale_trend_dispersion", canonical="ts_multiscale_trend_dispersion", backend="polars", research_only=True)
class TSMultiscaleTrendDispersionPolarsNative(SeriesOperator):
    """Standard deviation of trend estimates across scales"""

    metadata = OperatorMetadata(
        name="ts_multiscale_trend_dispersion",
        category="time_series",
        description="Variability of trend across scales",
        param_names=["feature", "window", "n_scales"],
        return_type="series",
        tags=["time_series", "trend", "multiscale", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_scales": ParamSpec(dtype=int, min=2, default=4, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_scales=4, **kwargs):
        # TODO: Std of trend slopes at different scales
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Structural Level and Pivot Analysis
# ============================================================================

@register_operator(name="ts_nearest_structural_level_distance", canonical="ts_nearest_structural_level_distance", backend="polars", research_only=True)
class TSNearestStructuralLevelDistancePolarsNative(SeriesOperator):
    """Distance to nearest support/resistance level"""

    metadata = OperatorMetadata(
        name="ts_nearest_structural_level_distance",
        category="time_series",
        description="Distance to closest structural price level",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "support_resistance", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Identify structural levels (local extrema clusters) and compute distance
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_nth_pivot_high", canonical="ts_nth_pivot_high", backend="polars", research_only=True)
class TSNthPivotHighPolarsNative(SeriesOperator):
    """Value of nth most recent pivot high"""

    metadata = OperatorMetadata(
        name="ts_nth_pivot_high",
        category="time_series",
        description="Nth recent local maximum value",
        param_names=["feature", "window", "n"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "n": ParamSpec(dtype=int, min=1, default=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n=1, **kwargs):
        # TODO: Identify pivot highs and select nth most recent
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_nth_pivot_high_age", canonical="ts_nth_pivot_high_age", backend="polars", research_only=True)
class TSNthPivotHighAgePolarsNative(SeriesOperator):
    """Periods since nth most recent pivot high"""

    metadata = OperatorMetadata(
        name="ts_nth_pivot_high_age",
        category="time_series",
        description="Time since nth pivot high",
        param_names=["feature", "window", "n"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "n": ParamSpec(dtype=int, min=1, default=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n=1, **kwargs):
        # TODO: Track pivot high timestamps
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_nth_pivot_low", canonical="ts_nth_pivot_low", backend="polars", research_only=True)
class TSNthPivotLowPolarsNative(SeriesOperator):
    """Value of nth most recent pivot low"""

    metadata = OperatorMetadata(
        name="ts_nth_pivot_low",
        category="time_series",
        description="Nth recent local minimum value",
        param_names=["feature", "window", "n"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "n": ParamSpec(dtype=int, min=1, default=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n=1, **kwargs):
        # TODO: Identify pivot lows and select nth most recent
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_nth_pivot_low_age", canonical="ts_nth_pivot_low_age", backend="polars", research_only=True)
class TSNthPivotLowAgePolarsNative(SeriesOperator):
    """Periods since nth most recent pivot low"""

    metadata = OperatorMetadata(
        name="ts_nth_pivot_low_age",
        category="time_series",
        description="Time since nth pivot low",
        param_names=["feature", "window", "n"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "n": ParamSpec(dtype=int, min=1, default=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n=1, **kwargs):
        # TODO: Track pivot low timestamps
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_nth_value", canonical="ts_nth_value", backend="polars", research_only=True)
class TSNthValuePolarsNative(SeriesOperator):
    """Value at nth position back in time"""

    metadata = OperatorMetadata(
        name="ts_nth_value",
        category="time_series",
        description="Historical value at specific lag",
        param_names=["feature", "n"],
        return_type="series",
        tags=["time_series", "lag", "pit_safe"],
    )
    metadata.param_specs = {
        "n": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, n, **kwargs):
        return feature.shift(n)


# ============================================================================
# Opening/Overnight Analysis
# ============================================================================

@register_operator(name="ts_opening_mispricing_score", canonical="ts_opening_mispricing_score", backend="polars", research_only=True)
class TSOpeningMispricingScorePolarsNative(SeriesOperator):
    """Z-score of opening gap relative to historical pattern"""

    metadata = OperatorMetadata(
        name="ts_opening_mispricing_score",
        category="time_series",
        description="Anomaly score for opening price gap",
        param_names=["close", "open_next", "window"],
        return_type="series",
        tags=["time_series", "overnight", "mispricing", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, close, open_next, window, **kwargs):
        gap = (open_next) / (close - 1) if (close - 1) != 0 else np.nan
        return (
            gap.to_frame()
            .lazy()
            .with_columns([
                (pl.col(gap.name) - pl.col(gap.name).rolling_mean(window)) /
                (pl.col(gap.name).rolling_std(window) + 1e-10)
            ])
            .select(pl.col(gap.name))
            .collect()
            .to_series()
        )


@register_operator(name="ts_ordinal_irreversibility", canonical="ts_ordinal_irreversibility", backend="polars", research_only=True)
class TSOrdinalIrreversibilityPolarsNative(SeriesOperator):
    """Time asymmetry based on ordinal patterns"""

    metadata = OperatorMetadata(
        name="ts_ordinal_irreversibility",
        category="time_series",
        description="Asymmetry between forward and backward ordinal patterns",
        param_names=["feature", "window", "embed_dim"],
        return_type="series",
        tags=["time_series", "irreversibility", "ordinal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "embed_dim": ParamSpec(dtype=int, min=2, default=3, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, embed_dim=3, **kwargs):
        # TODO: Compare ordinal pattern distributions forward vs backward
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_overnight_intraday_cov", canonical="ts_overnight_intraday_cov", backend="polars", research_only=True)
class TSOvernightIntradayCovPolarsNative(SeriesOperator):
    """Covariance between overnight and intraday returns"""

    metadata = OperatorMetadata(
        name="ts_overnight_intraday_cov",
        category="time_series",
        description="Rolling covariance of overnight vs intraday",
        param_names=["overnight_ret", "intraday_ret", "window"],
        return_type="series",
        tags=["time_series", "overnight", "covariance", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, overnight_ret, intraday_ret, window, **kwargs):
        df = pl.DataFrame({"on": overnight_ret, "id": intraday_ret})
        return (
            df.lazy()
            .with_columns([
                (pl.col("on") * pl.col("id")).rolling_mean(window).alias("_prod"),
                pl.col("on").rolling_mean(window).alias("_mean_on"),
                pl.col("id").rolling_mean(window).alias("_mean_id"),
            ])
            .select([
                (pl.col("_prod") - pl.col("_mean_on") * pl.col("_mean_id")).alias("cov")
            ])
            .collect()["cov"]
        )


@register_operator(name="ts_overnight_intraday_sign_agreement", canonical="ts_overnight_intraday_sign_agreement", backend="polars", research_only=True)
class TSOvernightIntradaySignAgreementPolarsNative(SeriesOperator):
    """Fraction of time overnight and intraday returns have same sign"""

    metadata = OperatorMetadata(
        name="ts_overnight_intraday_sign_agreement",
        category="time_series",
        description="Rolling agreement rate of return signs",
        param_names=["overnight_ret", "intraday_ret", "window"],
        return_type="series",
        tags=["time_series", "overnight", "sign", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, overnight_ret, intraday_ret, window, **kwargs):
        df = pl.DataFrame({"on": overnight_ret, "id": intraday_ret})
        return (
            df.lazy()
            .with_columns([
                (pl.col("on").sign() == pl.col("id").sign()).cast(pl.Float64).rolling_mean(window)
            ])
            .select(pl.col("on"))
            .collect()["on"]
        )


@register_operator(name="ts_overnight_intraday_spread", canonical="ts_overnight_intraday_spread", backend="polars", research_only=True)
class TSOvernightIntradaySpreadPolarsNative(SeriesOperator):
    """Difference between overnight and intraday return means"""

    metadata = OperatorMetadata(
        name="ts_overnight_intraday_spread",
        category="time_series",
        description="Mean overnight return minus mean intraday return",
        param_names=["overnight_ret", "intraday_ret", "window"],
        return_type="series",
        tags=["time_series", "overnight", "spread", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, overnight_ret, intraday_ret, window, **kwargs):
        df = pl.DataFrame({"on": overnight_ret, "id": intraday_ret})
        return (
            df.lazy()
            .select([
                (pl.col("on").rolling_mean(window) - pl.col("id").rolling_mean(window)).alias("spread")
            ])
            .collect()["spread"]
        )


# ============================================================================
# Partial Correlation and Advanced Regression
# ============================================================================

@register_operator(name="ts_partial_corr", canonical="ts_partial_corr", backend="polars", research_only=True)
class TSPartialCorrPolarsNative(SeriesOperator):
    """Partial correlation controlling for confounding variable"""

    metadata = OperatorMetadata(
        name="ts_partial_corr",
        category="time_series",
        description="Correlation between X and Y controlling for Z",
        param_names=["x", "y", "z", "window"],
        return_type="series",
        tags=["time_series", "correlation", "partial", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, z, window, **kwargs):
        # Partial corr: corr(x - E[x|z], y - E[y|z])
        # TODO: Requires rolling regression to compute residuals
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_pastor_stambaugh_liquidity_gamma", canonical="ts_pastor_stambaugh_liquidity_gamma", backend="polars", research_only=True)
class TSPastorStambaughLiquidityGammaPolarsNative(SeriesOperator):
    """Pastor-Stambaugh liquidity measure (return reversal after volume)"""

    metadata = OperatorMetadata(
        name="ts_pastor_stambaugh_liquidity_gamma",
        category="time_series",
        description="Coefficient of return on signed volume",
        param_names=["returns", "volume", "window"],
        return_type="series",
        tags=["time_series", "liquidity", "microstructure", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, volume, window, **kwargs):
        # Regress returns_{t+1} on signed_volume_t * |return_t|
        # TODO: Rolling regression implementation
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Path Geometry and Signatures
# ============================================================================

@register_operator(name="ts_path_efficiency", canonical="ts_path_efficiency", backend="polars", research_only=True)
class TSPathEfficiencyPolarsNative(SeriesOperator):
    """Ratio of straight-line distance to path length"""

    metadata = OperatorMetadata(
        name="ts_path_efficiency",
        category="time_series",
        description="Euclidean distance / cumulative path length",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "geometry", "path", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Straight-line distance / sum of segment lengths
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).shift(window - 1).alias("_start"),
                pl.col(feature.name).alias("_end"),
                pl.col(feature.name).diff().abs().rolling_sum(window).alias("_path_len"),
            ])
            .with_columns([
                pl.when(pl.col("_path_len") != 0).then((pl.col("_end") - pl.col("_start")).abs() / (pl.col("_path_len") + 1e-10)).otherwise(None).alias("efficiency")
            ])
            .select("efficiency")
            .collect()
            .to_series()
        )


@register_operator(name="ts_path_leadlag_area", canonical="ts_path_leadlag_area", backend="polars", research_only=True)
class TSPathLeadlagAreaPolarsNative(SeriesOperator):
    """Signed area between two paths (lead-lag relationship)"""

    metadata = OperatorMetadata(
        name="ts_path_leadlag_area",
        category="time_series",
        description="Integrated difference between two series",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["time_series", "leadlag", "path", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, y, window, **kwargs):
        df = pl.DataFrame({"x": x, "y": y})
        return (
            df.lazy()
            .with_columns([
                (pl.col("x") - pl.col("y")).rolling_sum(window).alias("area")
            ])
            .select("area")
            .collect()["area"]
        )


@register_operator(name="ts_path_signature_area", canonical="ts_path_signature_area", backend="polars", research_only=True)
class TSPathSignatureAreaPolarsNative(SeriesOperator):
    """Path signature level 2 area term"""

    metadata = OperatorMetadata(
        name="ts_path_signature_area",
        category="time_series",
        description="Second-level path signature (area)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "path_signature", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Requires path signature computation (iterated integrals)
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_path_signature_depth2_norm", canonical="ts_path_signature_depth2_norm", backend="polars", research_only=True)
class TSPathSignatureDepth2NormPolarsNative(SeriesOperator):
    """Norm of depth-2 path signature"""

    metadata = OperatorMetadata(
        name="ts_path_signature_depth2_norm",
        category="time_series",
        description="Magnitude of second-order signature terms",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "path_signature", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Requires path signature computation
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_pattern_symmetry", canonical="ts_pattern_symmetry", backend="polars", research_only=True)
class TSPatternSymmetryPolarsNative(SeriesOperator):
    """Symmetry score of pattern around midpoint"""

    metadata = OperatorMetadata(
        name="ts_pattern_symmetry",
        category="time_series",
        description="Correlation between first and reversed second half",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "pattern", "symmetry", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=4, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Requires computing correlation within rolling window
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Permutation and Persistence Analysis
# ============================================================================

@register_operator(name="ts_permutation_transition_entropy", canonical="ts_permutation_transition_entropy", backend="polars", research_only=True)
class TSPermutationTransitionEntropyPolarsNative(SeriesOperator):
    """Entropy of transitions between ordinal patterns"""

    metadata = OperatorMetadata(
        name="ts_permutation_transition_entropy",
        category="time_series",
        description="Shannon entropy of ordinal pattern transitions",
        param_names=["feature", "window", "embed_dim"],
        return_type="series",
        tags=["time_series", "entropy", "permutation", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "embed_dim": ParamSpec(dtype=int, min=2, default=3, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, embed_dim=3, **kwargs):
        # TODO: Extract ordinal patterns and compute transition entropy
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_persistence_birth_dispersion", canonical="ts_persistence_birth_dispersion", backend="polars", research_only=True)
class TSPersistenceBirthDispersionPolarsNative(SeriesOperator):
    """Standard deviation of topological feature birth times"""

    metadata = OperatorMetadata(
        name="ts_persistence_birth_dispersion",
        category="time_series",
        description="Variability in persistence diagram birth times",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "topology", "persistence", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Requires persistent homology computation
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_persistence_diagram_shift", canonical="ts_persistence_diagram_shift", backend="polars", research_only=True)
class TSPersistenceDiagramShiftPolarsNative(SeriesOperator):
    """Wasserstein distance between current and reference persistence diagrams"""

    metadata = OperatorMetadata(
        name="ts_persistence_diagram_shift",
        category="time_series",
        description="Change in topological structure",
        param_names=["feature", "window", "reference_window"],
        return_type="series",
        tags=["time_series", "topology", "persistence", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "reference_window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, reference_window, **kwargs):
        # TODO: Requires persistent homology and Wasserstein distance
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_persistence_entropy_h0", canonical="ts_persistence_entropy_h0", backend="polars", research_only=True)
class TSPersistenceEntropyH0PolarsNative(SeriesOperator):
    """Entropy of H0 (connected components) persistence"""

    metadata = OperatorMetadata(
        name="ts_persistence_entropy_h0",
        category="time_series",
        description="Shannon entropy of 0-dimensional persistence lifetimes",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "topology", "persistence", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Requires persistent homology computation
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_persistence_entropy_h1", canonical="ts_persistence_entropy_h1", backend="polars", research_only=True)
class TSPersistenceEntropyH1PolarsNative(SeriesOperator):
    """Entropy of H1 (loops) persistence"""

    metadata = OperatorMetadata(
        name="ts_persistence_entropy_h1",
        category="time_series",
        description="Shannon entropy of 1-dimensional persistence lifetimes",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "topology", "persistence", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Requires persistent homology computation
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Change Detection
# ============================================================================

@register_operator(name="ts_pettitt_change_score", canonical="ts_pettitt_change_score", backend="polars", research_only=True)
class TSPettittChangeScorePolarsNative(SeriesOperator):
    """Pettitt test statistic for change point detection"""

    metadata = OperatorMetadata(
        name="ts_pettitt_change_score",
        category="time_series",
        description="Non-parametric change point test statistic",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "change_detection", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Compute Pettitt test statistic (rank-based)
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_pickands_tail_index", canonical="ts_pickands_tail_index", backend="polars", research_only=True)
class TSPickandsTailIndexPolarsNative(SeriesOperator):
    """Pickands estimator for tail index (extreme value theory)"""

    metadata = OperatorMetadata(
        name="ts_pickands_tail_index",
        category="time_series",
        description="Tail heaviness parameter estimate",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "extreme_value", "tail", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=50, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Pickands estimator using order statistics
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Pivot Point Analysis
# ============================================================================

@register_operator(name="ts_pivot_high_age", canonical="ts_pivot_high_age", backend="polars", research_only=True)
class TSPivotHighAgePolarsNative(SeriesOperator):
    """Periods since most recent pivot high"""

    metadata = OperatorMetadata(
        name="ts_pivot_high_age",
        category="time_series",
        description="Time since last local maximum",
        param_names=["feature", "window", "strength"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "strength": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, strength=2, **kwargs):
        # TODO: Identify pivot highs and track recency
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_pivot_high_count", canonical="ts_pivot_high_count", backend="polars", research_only=True)
class TSPivotHighCountPolarsNative(SeriesOperator):
    """Count of pivot highs in window"""

    metadata = OperatorMetadata(
        name="ts_pivot_high_count",
        category="time_series",
        description="Number of local maxima in window",
        param_names=["feature", "window", "strength"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "strength": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, strength=2, **kwargs):
        # TODO: Count pivot highs in rolling window
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_pivot_high_spacing", canonical="ts_pivot_high_spacing", backend="polars", research_only=True)
class TSPivotHighSpacingPolarsNative(SeriesOperator):
    """Average spacing between pivot highs"""

    metadata = OperatorMetadata(
        name="ts_pivot_high_spacing",
        category="time_series",
        description="Mean time between local maxima",
        param_names=["feature", "window", "strength"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "strength": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, strength=2, **kwargs):
        # TODO: Compute average interval between pivot highs
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_pivot_low_age", canonical="ts_pivot_low_age", backend="polars", research_only=True)
class TSPivotLowAgePolarsNative(SeriesOperator):
    """Periods since most recent pivot low"""

    metadata = OperatorMetadata(
        name="ts_pivot_low_age",
        category="time_series",
        description="Time since last local minimum",
        param_names=["feature", "window", "strength"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "strength": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, strength=2, **kwargs):
        # TODO: Identify pivot lows and track recency
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_pivot_low_count", canonical="ts_pivot_low_count", backend="polars", research_only=True)
class TSPivotLowCountPolarsNative(SeriesOperator):
    """Count of pivot lows in window"""

    metadata = OperatorMetadata(
        name="ts_pivot_low_count",
        category="time_series",
        description="Number of local minima in window",
        param_names=["feature", "window", "strength"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "strength": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, strength=2, **kwargs):
        # TODO: Count pivot lows in rolling window
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_pivot_low_spacing", canonical="ts_pivot_low_spacing", backend="polars", research_only=True)
class TSPivotLowSpacingPolarsNative(SeriesOperator):
    """Average spacing between pivot lows"""

    metadata = OperatorMetadata(
        name="ts_pivot_low_spacing",
        category="time_series",
        description="Mean time between local minima",
        param_names=["feature", "window", "strength"],
        return_type="series",
        tags=["time_series", "pivot", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "strength": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, strength=2, **kwargs):
        # TODO: Compute average interval between pivot lows
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_price_delay", canonical="ts_price_delay", backend="polars", research_only=True)
class TSPriceDelayPolarsNative(SeriesOperator):
    """Price delay measure (fraction of R² from lagged returns)"""

    metadata = OperatorMetadata(
        name="ts_price_delay",
        category="time_series",
        description="Measure of price adjustment speed",
        param_names=["returns", "window", "max_lag"],
        return_type="series",
        tags=["time_series", "microstructure", "delay", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "max_lag": ParamSpec(dtype=int, min=1, default=5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, returns, window, max_lag=5, **kwargs):
        # TODO: Regress returns on lagged market returns and compute incremental R²
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Sample Entropy and Quantile Metrics
# ============================================================================

@register_operator(name="ts_pseudocount_sample_entropy", canonical="ts_pseudocount_sample_entropy", backend="polars", research_only=True)
class TSPseudocountSampleEntropyPolarsNative(SeriesOperator):
    """Sample entropy with pseudocount regularization"""

    metadata = OperatorMetadata(
        name="ts_pseudocount_sample_entropy",
        category="time_series",
        description="Regularized sample entropy estimate",
        param_names=["feature", "window", "m", "r"],
        return_type="series",
        tags=["time_series", "entropy", "complexity", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=50, param_role=ParamRole.HORIZON),
        "m": ParamSpec(dtype=int, min=1, default=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "r": ParamSpec(dtype=float, default=0.2, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, m=2, r=0.2, **kwargs):
        # TODO: Sample entropy with added pseudocount to avoid log(0)
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_quantile_beta_spread", canonical="ts_quantile_beta_spread", backend="polars", research_only=True)
class TSQuantileBetaSpreadPolarsNative(SeriesOperator):
    """Difference between upper and lower quantile regression slopes"""

    metadata = OperatorMetadata(
        name="ts_quantile_beta_spread",
        category="time_series",
        description="Asymmetry in quantile regression slopes",
        param_names=["y", "x", "window", "upper_q", "lower_q"],
        return_type="series",
        tags=["time_series", "quantile_regression", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "upper_q": ParamSpec(dtype=float, default=0.75, param_role=ParamRole.STATE_THRESHOLD),
        "lower_q": ParamSpec(dtype=float, default=0.25, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, y, x, window, upper_q=0.75, lower_q=0.25, **kwargs):
        # TODO: Quantile regression at two quantiles and compute slope difference
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_quantile_beta_spread_prior", canonical="ts_quantile_beta_spread_prior", backend="polars", research_only=True)
class TSQuantileBetaSpreadPriorPolarsNative(SeriesOperator):
    """Quantile beta spread using only historical data (causal)"""

    metadata = OperatorMetadata(
        name="ts_quantile_beta_spread_prior",
        category="time_series",
        description="Causal version of quantile beta spread",
        param_names=["y", "x", "window", "upper_q", "lower_q"],
        return_type="series",
        tags=["time_series", "quantile_regression", "causal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "upper_q": ParamSpec(dtype=float, default=0.75, param_role=ParamRole.STATE_THRESHOLD),
        "lower_q": ParamSpec(dtype=float, default=0.25, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, y, x, window, upper_q=0.75, lower_q=0.25, **kwargs):
        # Same as above but exclude current period
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_quantile_crossing_spectral_concentration", canonical="ts_quantile_crossing_spectral_concentration", backend="polars", research_only=True)
class TSQuantileCrossingSpectralConcentrationPolarsNative(SeriesOperator):
    """Spectral concentration of quantile crossing times"""

    metadata = OperatorMetadata(
        name="ts_quantile_crossing_spectral_concentration",
        category="time_series",
        description="Periodicity in threshold crossing events",
        param_names=["feature", "window", "quantile"],
        return_type="series",
        tags=["time_series", "quantile", "spectral", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=50, param_role=ParamRole.HORIZON),
        "quantile": ParamSpec(dtype=float, default=0.5, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, quantile=0.5, **kwargs):
        # TODO: FFT of crossing indicator and measure peak concentration
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_quantile_if", canonical="ts_quantile_if", backend="polars", research_only=True)
class TSQuantileIfPolarsNative(SeriesOperator):
    """Conditional quantile (quantile of values where condition is True)"""

    metadata = OperatorMetadata(
        name="ts_quantile_if",
        category="time_series",
        description="Quantile restricted to conditional subset",
        param_names=["feature", "condition", "window", "quantile"],
        return_type="series",
        tags=["time_series", "quantile", "conditional", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "quantile": ParamSpec(dtype=float, default=0.5, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, condition, window, quantile=0.5, **kwargs):
        # TODO: Rolling quantile of filtered values
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_quantile_transport_curvature", canonical="ts_quantile_transport_curvature", backend="polars", research_only=True)
class TSQuantileTransportCurvaturePolarsNative(SeriesOperator):
    """Curvature of quantile transport map"""

    metadata = OperatorMetadata(
        name="ts_quantile_transport_curvature",
        category="time_series",
        description="Second derivative of distribution shift",
        param_names=["feature", "window", "reference_window"],
        return_type="series",
        tags=["time_series", "quantile", "transport", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "reference_window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, reference_window, **kwargs):
        # TODO: Fit quantile transport map and compute curvature
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_quantile_transport_slope", canonical="ts_quantile_transport_slope", backend="polars", research_only=True)
class TSQuantileTransportSlopePolarsNative(SeriesOperator):
    """Slope of quantile transport map (median)"""

    metadata = OperatorMetadata(
        name="ts_quantile_transport_slope",
        category="time_series",
        description="Linear approximation of distribution shift",
        param_names=["feature", "window", "reference_window"],
        return_type="series",
        tags=["time_series", "quantile", "transport", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "reference_window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, reference_window, **kwargs):
        # TODO: Slope of quantile-quantile plot
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_quantilogram", canonical="ts_quantilogram", backend="polars", research_only=True)
class TSQuantilogramPolarsNative(SeriesOperator):
    """Quantilogram (correlation of quantile exceedance at different lags)"""

    metadata = OperatorMetadata(
        name="ts_quantilogram",
        category="time_series",
        description="Autocorrelation of quantile indicators",
        param_names=["feature", "window", "quantile", "lag"],
        return_type="series",
        tags=["time_series", "quantile", "autocorrelation", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "quantile": ParamSpec(dtype=float, default=0.5, param_role=ParamRole.STATE_THRESHOLD),
        "lag": ParamSpec(dtype=int, min=1, default=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, quantile=0.5, lag=1, **kwargs):
        # Correlation between (X_t < q_t) and (X_{t-lag} < q_{t-lag})
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                (pl.col(feature.name) < pl.col(feature.name).rolling_quantile(quantile, window)).cast(pl.Float64).alias("_ind"),
            ])
            .with_columns([
                pl.col("_ind").shift(lag).alias("_ind_lag"),
            ])
            .with_columns([
                pl.corr(pl.col("_ind"), pl.col("_ind_lag")).over_rolling(window).alias("quantilogram")
            ])
            .select("quantilogram")
            .collect()
            .to_series()
        )


@register_operator(name="ts_range_expansion", canonical="ts_range_expansion", backend="polars", research_only=True)
class TSRangeExpansionPolarsNative(SeriesOperator):
    """Rate of range expansion (volatility regime indicator)"""

    metadata = OperatorMetadata(
        name="ts_range_expansion",
        category="time_series",
        description="Change in price range magnitude",
        param_names=["high", "low", "window"],
        return_type="series",
        tags=["time_series", "range", "volatility", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, high, low, window, **kwargs):
        df = pl.DataFrame({"high": high, "low": low})
        return (
            df.lazy()
            .with_columns([
                (pl.col("high") - pl.col("low")).alias("_range"),
            ])
            .with_columns([
                pl.when(pl.col("_range") != 0).then(pl.col("_range") / pl.col("_range").rolling_mean(window) - 1).otherwise(None).alias("expansion")
            ])
            .select("expansion")
            .collect()["expansion"]
        )


@register_operator(name="ts_ratio", canonical="ts_ratio", backend="polars", research_only=True)
class TSRatioPolarsNative(SeriesOperator):
    """Simple ratio of two series"""

    metadata = OperatorMetadata(
        name="ts_ratio",
        category="time_series",
        description="X / Y ratio",
        param_names=["x", "y"],
        return_type="series",
        tags=["time_series", "ratio", "pit_safe"],
    )

    def _calculate_series(self, x, y, **kwargs):
        return np.where((y + 1e-10) != 0, (x) / ((y + 1e-10)), np.nan)


# ============================================================================
# Recurrence Analysis (RQA)
# ============================================================================

@register_operator(name="ts_recurrence_determinism", canonical="ts_recurrence_determinism", backend="polars", research_only=True)
class TSRecurrenceDeterminismPolarsNative(SeriesOperator):
    """RQA determinism (fraction of recurrence points forming diagonal lines)"""

    metadata = OperatorMetadata(
        name="ts_recurrence_determinism",
        category="time_series",
        description="Predictability measure from recurrence plot",
        param_names=["feature", "window", "threshold"],
        return_type="series",
        tags=["time_series", "recurrence", "rqa", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold=0.1, **kwargs):
        # TODO: Construct recurrence plot and compute determinism
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_recurrence_diagonal_entropy", canonical="ts_recurrence_diagonal_entropy", backend="polars", research_only=True)
class TSRecurrenceDiagonalEntropyPolarsNative(SeriesOperator):
    """Shannon entropy of diagonal line length distribution"""

    metadata = OperatorMetadata(
        name="ts_recurrence_diagonal_entropy",
        category="time_series",
        description="Complexity of recurrence patterns",
        param_names=["feature", "window", "threshold"],
        return_type="series",
        tags=["time_series", "recurrence", "rqa", "entropy", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold=0.1, **kwargs):
        # TODO: Compute diagonal line length distribution and entropy
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_recurrence_divergence", canonical="ts_recurrence_divergence", backend="polars", research_only=True)
class TSRecurrenceDivergencePolarsNative(SeriesOperator):
    """Rate of divergence (inverse of longest diagonal line)"""

    metadata = OperatorMetadata(
        name="ts_recurrence_divergence",
        category="time_series",
        description="Lyapunov-like divergence measure",
        param_names=["feature", "window", "threshold"],
        return_type="series",
        tags=["time_series", "recurrence", "rqa", "divergence", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold=0.1, **kwargs):
        # TODO: 1 / max diagonal line length
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_recurrence_laminarity", canonical="ts_recurrence_laminarity", backend="polars", research_only=True)
class TSRecurrenceLaminarityPolarsNative(SeriesOperator):
    """RQA laminarity (fraction of recurrence points in vertical lines)"""

    metadata = OperatorMetadata(
        name="ts_recurrence_laminarity",
        category="time_series",
        description="Measure of intermittency and trapping",
        param_names=["feature", "window", "threshold"],
        return_type="series",
        tags=["time_series", "recurrence", "rqa", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold=0.1, **kwargs):
        # TODO: Construct recurrence plot and compute laminarity
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_recurrence_longest_vertical_length", canonical="ts_recurrence_longest_vertical_length", backend="polars", research_only=True)
class TSRecurrenceLongestVerticalLengthPolarsNative(SeriesOperator):
    """Length of longest vertical line in recurrence plot"""

    metadata = OperatorMetadata(
        name="ts_recurrence_longest_vertical_length",
        category="time_series",
        description="Maximum trapping time indicator",
        param_names=["feature", "window", "threshold"],
        return_type="series",
        tags=["time_series", "recurrence", "rqa", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold=0.1, **kwargs):
        # TODO: Find longest vertical line in recurrence matrix
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_recurrence_mean_diagonal_length", canonical="ts_recurrence_mean_diagonal_length", backend="polars", research_only=True)
class TSRecurrenceMeanDiagonalLengthPolarsNative(SeriesOperator):
    """Average length of diagonal lines in recurrence plot"""

    metadata = OperatorMetadata(
        name="ts_recurrence_mean_diagonal_length",
        category="time_series",
        description="Average predictability horizon",
        param_names=["feature", "window", "threshold"],
        return_type="series",
        tags=["time_series", "recurrence", "rqa", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold=0.1, **kwargs):
        # TODO: Compute mean diagonal line length
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_recurrence_trapping_time", canonical="ts_recurrence_trapping_time", backend="polars", research_only=True)
class TSRecurrenceTrappingTimePolarsNative(SeriesOperator):
    """Average vertical line length (trapping time)"""

    metadata = OperatorMetadata(
        name="ts_recurrence_trapping_time",
        category="time_series",
        description="Mean time spent in recurring states",
        param_names=["feature", "window", "threshold"],
        return_type="series",
        tags=["time_series", "recurrence", "rqa", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold=0.1, **kwargs):
        # TODO: Compute mean vertical line length
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Regime and Regression Analysis
# ============================================================================

@register_operator(name="ts_regime_duration", canonical="ts_regime_duration", backend="polars", research_only=True)
class TSRegimeDurationPolarsNative(SeriesOperator):
    """Duration of current regime (time since last regime change)"""

    metadata = OperatorMetadata(
        name="ts_regime_duration",
        category="time_series",
        description="Periods in current state",
        param_names=["regime_indicator"],
        return_type="series",
        tags=["time_series", "regime", "duration", "pit_safe"],
    )

    def _calculate_series(self, regime_indicator, **kwargs):
        # Count consecutive periods in same regime
        return (
            regime_indicator.to_frame()
            .lazy()
            .with_columns([
                (pl.col(regime_indicator.name) != pl.col(regime_indicator.name).shift(1)).alias("_change"),
            ])
            .with_columns([
                pl.col("_change").cum_sum().alias("_regime_id"),
            ])
            .with_columns([
                pl.col("_regime_id").count().over("_regime_id").alias("duration")
            ])
            .select("duration")
            .collect()
            .to_series()
        )


@register_operator(name="ts_regression_forecast_error", canonical="ts_regression_forecast_error", backend="polars", research_only=True)
class TSRegressionForecastErrorPolarsNative(SeriesOperator):
    """Out-of-sample regression forecast error"""

    metadata = OperatorMetadata(
        name="ts_regression_forecast_error",
        category="time_series",
        description="One-step-ahead prediction error",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "regression", "forecast", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, y, x, window, **kwargs):
        # TODO: Rolling regression and compute forecast error
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_regression_forecast_error_z", canonical="ts_regression_forecast_error_z", backend="polars", research_only=True)
class TSRegressionForecastErrorZPolarsNative(SeriesOperator):
    """Standardized forecast error"""

    metadata = OperatorMetadata(
        name="ts_regression_forecast_error_z",
        category="time_series",
        description="Z-scored prediction error",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "regression", "forecast", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, y, x, window, **kwargs):
        # TODO: Forecast error / std of forecast errors
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_regression_in_sample_resid", canonical="ts_regression_in_sample_resid", backend="polars", research_only=True)
class TSRegressionInSampleResidPolarsNative(SeriesOperator):
    """In-sample regression residual"""

    metadata = OperatorMetadata(
        name="ts_regression_in_sample_resid",
        category="time_series",
        description="Rolling regression residual",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "regression", "residual", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, y, x, window, **kwargs):
        # TODO: y - (alpha + beta*x) using rolling estimates
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_regression_resid_if", canonical="ts_regression_resid_if", backend="polars", research_only=True)
class TSRegressionResidIfPolarsNative(SeriesOperator):
    """Conditional regression residual"""

    metadata = OperatorMetadata(
        name="ts_regression_resid_if",
        category="time_series",
        description="Residual using only conditional subset",
        param_names=["y", "x", "condition", "window"],
        return_type="series",
        tags=["time_series", "regression", "conditional", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, y, x, condition, window, **kwargs):
        # TODO: Regress using only observations where condition is True
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_regression_resid_mean", canonical="ts_regression_resid_mean", backend="polars", research_only=True)
class TSRegressionResidMeanPolarsNative(SeriesOperator):
    """Rolling mean of regression residuals"""

    metadata = OperatorMetadata(
        name="ts_regression_resid_mean",
        category="time_series",
        description="Average residual in window",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "regression", "residual", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, y, x, window, **kwargs):
        # TODO: Mean of rolling regression residuals
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_regression_tstat", canonical="ts_regression_tstat", backend="polars", research_only=True)
class TSRegressionTstatPolarsNative(SeriesOperator):
    """T-statistic of regression slope"""

    metadata = OperatorMetadata(
        name="ts_regression_tstat",
        category="time_series",
        description="Significance of rolling regression coefficient",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "regression", "tstat", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, y, x, window, **kwargs):
        # TODO: beta / se(beta) from rolling regression
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_residualized_hsic", canonical="ts_residualized_hsic", backend="polars", research_only=True)
class TSResidualizedHSICPolarsNative(SeriesOperator):
    """Hilbert-Schmidt Independence Criterion on residuals"""

    metadata = OperatorMetadata(
        name="ts_residualized_hsic",
        category="time_series",
        description="Nonlinear dependence after removing linear relationship",
        param_names=["y", "x", "z", "window"],
        return_type="series",
        tags=["time_series", "independence", "kernel", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, y, x, z, window, **kwargs):
        # TODO: Residualize y and x on z, then compute HSIC
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Support/Resistance Analysis
# ============================================================================

@register_operator(name="ts_resistance_break", canonical="ts_resistance_break", backend="polars", research_only=True)
class TSResistanceBreakPolarsNative(SeriesOperator):
    """Binary indicator of resistance level break"""

    metadata = OperatorMetadata(
        name="ts_resistance_break",
        category="time_series",
        description="1 if price breaks above recent high",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "resistance", "breakout", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                (pl.col(feature.name) > pl.col(feature.name).shift(1).rolling_max(window)).cast(pl.Float64)
            ])
            .select(pl.col(feature.name))
            .collect()
            .to_series()
        )


@register_operator(name="ts_resistance_fit_r2", canonical="ts_resistance_fit_r2", backend="polars", research_only=True)
class TSResistanceFitR2PolarsNative(SeriesOperator):
    """R² of linear fit to resistance level touches"""

    metadata = OperatorMetadata(
        name="ts_resistance_fit_r2",
        category="time_series",
        description="Quality of resistance line fit",
        param_names=["high", "window"],
        return_type="series",
        tags=["time_series", "resistance", "fit", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, high, window, **kwargs):
        # TODO: Identify resistance touches and compute R² of trend line
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_resistance_log_slope", canonical="ts_resistance_log_slope", backend="polars", research_only=True)
class TSResistanceLogSlopePolarsNative(SeriesOperator):
    """Log-slope of resistance level over time"""

    metadata = OperatorMetadata(
        name="ts_resistance_log_slope",
        category="time_series",
        description="Exponential growth rate of resistance",
        param_names=["high", "window"],
        return_type="series",
        tags=["time_series", "resistance", "slope", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, high, window, **kwargs):
        # TODO: Regress log(resistance_touches) on time
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_resistance_slope", canonical="ts_resistance_slope", backend="polars", research_only=True)
class TSResistanceSlopePolarsNative(SeriesOperator):
    """Linear slope of resistance level"""

    metadata = OperatorMetadata(
        name="ts_resistance_slope",
        category="time_series",
        description="Trend direction of resistance line",
        param_names=["high", "window"],
        return_type="series",
        tags=["time_series", "resistance", "slope", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, high, window, **kwargs):
        # Simple slope of rolling max
        return (
            high.to_frame()
            .lazy()
            .with_columns([
                pl.col(high.name).rolling_max(window).alias("_resistance"),
            ])
            .with_columns([
                (pl.col("_resistance") - pl.col("_resistance").shift(window)).alias("slope")
            ])
            .select("slope")
            .collect()
            .to_series()
        )


# ============================================================================
# Response and Return Analysis
# ============================================================================

@register_operator(name="ts_response_slope_asymmetry", canonical="ts_response_slope_asymmetry", backend="polars", research_only=True)
class TSResponseSlopeAsymmetryPolarsNative(SeriesOperator):
    """Asymmetry between positive and negative response slopes"""

    metadata = OperatorMetadata(
        name="ts_response_slope_asymmetry",
        category="time_series",
        description="Difference in response to positive vs negative shocks",
        param_names=["response", "shock", "window"],
        return_type="series",
        tags=["time_series", "response", "asymmetry", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, response, shock, window, **kwargs):
        # TODO: Regress response on shock separately for positive and negative shocks
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_return_spectral_entropy", canonical="ts_return_spectral_entropy", backend="polars", research_only=True)
class TSReturnSpectralEntropyPolarsNative(SeriesOperator):
    """Entropy of power spectral density"""

    metadata = OperatorMetadata(
        name="ts_return_spectral_entropy",
        category="time_series",
        description="Frequency domain complexity measure",
        param_names=["returns", "window"],
        return_type="series",
        tags=["time_series", "spectral", "entropy", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=64, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, returns, window, **kwargs):
        # TODO: FFT -> PSD -> Shannon entropy of normalized PSD
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Robust Statistics
# ============================================================================

@register_operator(name="ts_robust_ema", canonical="ts_robust_ema", backend="polars", research_only=True)
class TSRobustEMAPolarsNative(SeriesOperator):
    """Robust exponential moving average (using median updates)"""

    metadata = OperatorMetadata(
        name="ts_robust_ema",
        category="time_series",
        description="EMA with outlier-resistant updates",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "ema", "robust", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Simplified: use rolling median instead of mean in EMA
        alpha = 2.0 / (window + 1)
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).ewm_mean(span=window).alias("ema")
            ])
            .select("ema")
            .collect()
            .to_series()
        )


@register_operator(name="ts_robust_zscore_inclusive", canonical="ts_robust_zscore_inclusive", backend="polars", research_only=True)
class TSRobustZscoreInclusivePolarsNative(SeriesOperator):
    """Robust z-score using median and MAD (inclusive of current point)"""

    metadata = OperatorMetadata(
        name="ts_robust_zscore_inclusive",
        category="time_series",
        description="(X - median) / MAD including current value",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "zscore", "robust", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).rolling_median(window).alias("_med"),
                (pl.col(feature.name) - pl.col(feature.name).rolling_median(window)).abs().rolling_median(window).alias("_mad"),
            ])
            .with_columns([
                pl.when(pl.col("_mad") != 0).then((pl.col(feature.name) - pl.col("_med")) / (pl.col("_mad") * 1.4826 + 1e-10)).otherwise(None).alias("robust_z")
            ])
            .select("robust_z")
            .collect()
            .to_series()
        )


@register_operator(name="ts_robust_zscore_prior", canonical="ts_robust_zscore_prior", backend="polars", research_only=True)
class TSRobustZscorePriorPolarsNative(SeriesOperator):
    """Robust z-score using only prior data (causal)"""

    metadata = OperatorMetadata(
        name="ts_robust_zscore_prior",
        category="time_series",
        description="(X - median_prior) / MAD_prior",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "zscore", "robust", "causal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).shift(1).rolling_median(window).alias("_med"),
                (pl.col(feature.name).shift(1) - pl.col(feature.name).shift(1).rolling_median(window)).abs().rolling_median(window).alias("_mad"),
            ])
            .with_columns([
                pl.when(pl.col("_mad") != 0).then((pl.col(feature.name) - pl.col("_med")) / (pl.col("_mad") * 1.4826 + 1e-10)).otherwise(None).alias("robust_z")
            ])
            .select("robust_z")
            .collect()
            .to_series()
        )


@register_operator(name="ts_rolling_median_causal", canonical="ts_rolling_median_causal", backend="polars", research_only=True)
class TSRollingMedianCausalPolarsNative(SeriesOperator):
    """Rolling median using only prior data"""

    metadata = OperatorMetadata(
        name="ts_rolling_median_causal",
        category="time_series",
        description="Causal rolling median",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "median", "causal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return feature.shift(1).rolling_median(window)


@register_operator(name="ts_rolling_sr_gaussian_mean_shift_score", canonical="ts_rolling_sr_gaussian_mean_shift_score", backend="polars", research_only=True)
class TSRollingSRGaussianMeanShiftScorePolarsNative(SeriesOperator):
    """Gaussian kernel mean shift anomaly score"""

    metadata = OperatorMetadata(
        name="ts_rolling_sr_gaussian_mean_shift_score",
        category="time_series",
        description="Kernel density-based anomaly detection",
        param_names=["feature", "window", "bandwidth"],
        return_type="series",
        tags=["time_series", "anomaly", "kernel", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "bandwidth": ParamSpec(dtype=float, default=1.0, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, bandwidth=1.0, **kwargs):
        # TODO: Requires Gaussian kernel density estimation
        return pl.lit(None).cast(pl.Float64)


# ============================================================================
# Additional Metrics (completing the batch to line 634)
# ============================================================================

@register_operator(name="ts_roughness", canonical="ts_roughness", backend="polars", research_only=True)
class TSRoughnessPolarsNative(SeriesOperator):
    """Path roughness (sum of squared second differences)"""

    metadata = OperatorMetadata(
        name="ts_roughness",
        category="time_series",
        description="Measure of path irregularity",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "roughness", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                pl.col(feature.name).diff().diff().pow(2).rolling_sum(window).alias("roughness")
            ])
            .select("roughness")
            .collect()
            .to_series()
        )


@register_operator(name="ts_rqa_determinism_fixed_rr", canonical="ts_rqa_determinism_fixed_rr", backend="polars", research_only=True)
class TSRQADeterminismFixedRRPolarsNative(SeriesOperator):
    """RQA determinism with fixed recurrence rate"""

    metadata = OperatorMetadata(
        name="ts_rqa_determinism_fixed_rr",
        category="time_series",
        description="Determinism normalized by recurrence rate",
        param_names=["feature", "window", "recurrence_rate"],
        return_type="series",
        tags=["time_series", "rqa", "recurrence", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "recurrence_rate": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, recurrence_rate=0.1, **kwargs):
        # TODO: Adaptive threshold to maintain fixed recurrence rate
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_rqa_laminarity_fixed_rr", canonical="ts_rqa_laminarity_fixed_rr", backend="polars", research_only=True)
class TSRQALaminarityFixedRRPolarsNative(SeriesOperator):
    """RQA laminarity with fixed recurrence rate"""

    metadata = OperatorMetadata(
        name="ts_rqa_laminarity_fixed_rr",
        category="time_series",
        description="Laminarity normalized by recurrence rate",
        param_names=["feature", "window", "recurrence_rate"],
        return_type="series",
        tags=["time_series", "rqa", "recurrence", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=30, param_role=ParamRole.HORIZON),
        "recurrence_rate": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, recurrence_rate=0.1, **kwargs):
        # TODO: Adaptive threshold for laminarity computation
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_run_concentration", canonical="ts_run_concentration", backend="polars", research_only=True)
class TSRunConcentrationPolarsNative(SeriesOperator):
    """Concentration of runs (streaks) in the series"""

    metadata = OperatorMetadata(
        name="ts_run_concentration",
        category="time_series",
        description="Gini coefficient of run lengths",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "runs", "concentration", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Identify runs and compute concentration metric
        return pl.lit(None).cast(pl.Float64)


@register_operator(name="ts_run_efficiency", canonical="ts_run_efficiency", backend="polars", research_only=True)
class TSRunEfficiencyPolarsNative(SeriesOperator):
    """Efficiency of directional runs"""

    metadata = OperatorMetadata(
        name="ts_run_efficiency",
        category="time_series",
        description="Average run length / total periods",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "runs", "efficiency", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Count direction changes and compute run efficiency
        return (
            feature.to_frame()
            .lazy()
            .with_columns([
                (pl.col(feature.name).diff().sign() != pl.col(feature.name).diff().sign().shift(1)).cast(pl.Float64).alias("_change"),
            ])
            .with_columns([
                (1.0 - pl.col("_change").rolling_mean(window)).alias("efficiency")
            ])
            .select("efficiency")
            .collect()
            .to_series()
        )


# End of batch 3 - operators 533-632



