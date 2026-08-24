# -*- coding: utf-8 -*-
"""
Polars native implementations for time series operators (ts_* family) - Batch 5

This batch covers 64 advanced time series operators:
- Distance correlation, feature geometry
- FIR filters, first passage times
- Fisher information, forbidden patterns
- Fractional differencing
- Generalized Hurst exponents
- GPD tail estimation
- H-infinity filters, Hampel filters
- Hankel matrix analysis
- Hartigan dip test, Higuchi fractal dimension
- Hill tail index, HSIC dependence
- HVG (Horizontal Visibility Graph) features
- Hysteresis states
- Interval geometry
- Joint energy, jump detection
- Kramers-Moyal expansion
- KS shift, L-moments
- Lagged correlations and mutual information
- Lempel-Ziv complexity
- Level shifts, leverage effects
- Line geometry
- Lo-MacKinlay variance ratio
- Local Lyapunov exponents
- Tail coexceedance
- Markov chain features

All operators use pure Polars lazy API for maximum performance.
Complex algorithms are implemented as skeletons with TODO markers.
"""

import polars as pl
import numpy as np
from typing import Optional, Union

from cleaned_operators.base import (
    SeriesOperator,
    register_operator as _register_operator,
    OperatorMetadata,
    ParamSpec,
    ParamRole,
)


def register_operator(*args, **kwargs):
    """Quarantine this batch until each canonical has parity evidence.

    The module contains proxies and formulas whose authoritative pandas
    contracts use different signatures.  Keeping the registrations visible
    while forcing ``research_only`` prevents the Polars proxies from being
    advertised as production implementations.
    """
    kwargs["status"] = "research_only"
    return _register_operator(*args, **kwargs)

# ============================================================================
# Distance Correlation and Feature Geometry
# ============================================================================

@register_operator(name="ts_distance_correlation_partial_proxy", canonical="ts_distance_correlation_partial_proxy", backend="polars")
class TSDistanceCorrelationPartialProxyPolarsNative(SeriesOperator):
    """Proxy for partial distance correlation (residual-based)

    NOTE: This operator is marked research_only. True partial distance correlation requires:
    1. Computing distance matrices for all variables
    2. U-centering the distance matrices
    3. Computing partial distance covariance

    Current implementation is a placeholder: coefficient of variation.
    """

    metadata = OperatorMetadata(
        name="ts_distance_correlation_partial_proxy",
        category="time_series",
        description="[RESEARCH ONLY] Proxy for partial distance correlation (requires distance matrices)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "dependence", "pit_safe", "research_only"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=4, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Placeholder: coefficient of variation (NOT distance correlation)
        return feature.rolling_std(window) / (feature.rolling_mean(window).abs() + 1e-8)


@register_operator(name="ts_feature_mode_share", canonical="ts_feature_mode_share", backend="polars", status="research_only")
class TSFeatureModeSharePolarsNative(SeriesOperator):
    """Share of observations at the mode value"""

    metadata = OperatorMetadata(
        name="ts_feature_mode_share",
        category="time_series",
        description="Fraction of window observations equal to mode",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "distribution", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper mode detection and share calculation
        # Placeholder: concentration measure
        return (feature ** 2).rolling_mean(window) / ((feature.rolling_mean(window) ** 2) + 1e-8)


@register_operator(name="ts_feature_subspace_rotation", canonical="ts_feature_subspace_rotation", backend="polars")
class TSFeatureSubspaceRotationPolarsNative(SeriesOperator):
    """Angle of principal component rotation between windows

    NOTE: This operator is marked research_only. True PCA subspace rotation requires:
    1. Computing PCA on consecutive rolling windows
    2. Measuring angle between principal component subspaces
    3. Requires multivariate data or sophisticated embedding

    Current implementation is a placeholder: rolling standard deviation change.
    """

    metadata = OperatorMetadata(
        name="ts_feature_subspace_rotation",
        category="time_series",
        description="[RESEARCH ONLY] PCA subspace rotation angle between consecutive windows (requires PCA)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "pca", "pit_safe", "research_only"],
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, feature, window, **kwargs):
        # Placeholder: rolling variance change (NOT true subspace rotation)
        return feature.rolling_std(window).diff()


# ============================================================================
# FIR Filters and First Passage
# ============================================================================

@register_operator(name="ts_fir_lowpass_causal", canonical="ts_fir_lowpass_causal", backend="polars", status="research_only")
class TSFIRLowpassCausalPolarsNative(SeriesOperator):
    """Causal FIR lowpass filter"""

    metadata = OperatorMetadata(
        name="ts_fir_lowpass_causal",
        category="time_series",
        description="Causal FIR lowpass filter (weighted moving average)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "filter", "causal", "pit_safe"],
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, feature, window, **kwargs):
        # Hamming window weights for FIR lowpass
        # Placeholder: simple moving average
        return feature.rolling_mean(window)


@register_operator(name="ts_first_passage_bias", canonical="ts_first_passage_bias", backend="polars", status="research_only")
class TSFirstPassageBiasPolarsNative(SeriesOperator):
    """Asymmetry in first passage times (up vs down)"""

    metadata = OperatorMetadata(
        name="ts_first_passage_bias",
        category="time_series",
        description="Asymmetry in first passage times across threshold",
        param_names=["feature", "threshold"],
        return_type="series",
        tags=["time_series", "threshold", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, threshold, **kwargs):
        # TODO: Implement proper first passage time calculation
        # Placeholder: threshold crossing indicator
        return (feature > threshold).cast(pl.Float64)


@register_operator(name="ts_first_passage_conditional_time", canonical="ts_first_passage_conditional_time", backend="polars", status="research_only")
class TSFirstPassageConditionalTimePolarsNative(SeriesOperator):
    """Expected time to cross threshold given current state"""

    metadata = OperatorMetadata(
        name="ts_first_passage_conditional_time",
        category="time_series",
        description="Expected time to threshold crossing conditional on current state",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "threshold", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        # TODO: Implement conditional first passage time
        # Placeholder: distance to threshold
        return (threshold - feature).abs().rolling_mean(window)


@register_operator(name="ts_first_passage_hit_probability", canonical="ts_first_passage_hit_probability", backend="polars")
class TSFirstPassageHitProbabilityPolarsNative(SeriesOperator):
    """Probability of hitting threshold within window"""

    metadata = OperatorMetadata(
        name="ts_first_passage_hit_probability",
        category="time_series",
        description="Rolling probability of threshold crossing",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "threshold", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        # Fraction of window where threshold is crossed
        crossed = (feature > threshold).cast(pl.Int32)
        return crossed.rolling_mean(window)


# ============================================================================
# Fisher Information and Forbidden Patterns
# ============================================================================

@register_operator(name="ts_fisher_information_shift", canonical="ts_fisher_information_shift", backend="polars", status="research_only")
class TSFisherInformationShiftPolarsNative(SeriesOperator):
    """Change in Fisher information between windows"""

    metadata = OperatorMetadata(
        name="ts_fisher_information_shift",
        category="time_series",
        description="Shift in Fisher information (variance of score function)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "information", "pit_safe"],
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper Fisher information
        # Placeholder: variance change
        return feature.rolling_var(window).diff()


@register_operator(name="ts_forbidden_ordinal_pattern_ratio", canonical="ts_forbidden_ordinal_pattern_ratio", backend="polars", status="research_only")
class TSForbiddenOrdinalPatternRatioPolarsNative(SeriesOperator):
    """Ratio of forbidden ordinal patterns (complexity measure)"""

    metadata = OperatorMetadata(
        name="ts_forbidden_ordinal_pattern_ratio",
        category="time_series",
        description="Fraction of forbidden ordinal patterns in rolling window",
        param_names=["feature", "window", "pattern_length"],
        return_type="series",
        tags=["time_series", "rolling", "complexity", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "pattern_length": ParamSpec(dtype=int, min=3, max=6, param_role=ParamRole.MODEL_ORDER),
    }

    def _calculate_series(self, feature, window, pattern_length=3, **kwargs):
        # TODO: Implement ordinal pattern detection
        # Placeholder: trend consistency measure
        return (feature.diff().sign()).rolling_std(window)


# ============================================================================
# Fractional Differencing
# ============================================================================

@register_operator(name="ts_fractional_difference", canonical="ts_fractional_difference", backend="polars", status="research_only")
class TSFractionalDifferencePolarsNative(SeriesOperator):
    """Fractional differencing (memory-preserving stationarity)"""

    metadata = OperatorMetadata(
        name="ts_fractional_difference",
        category="time_series",
        description="Fractional differencing with order d",
        param_names=["feature", "d", "window"],
        return_type="series",
        tags=["time_series", "rolling", "memory", "pit_safe"],
    )
    metadata.param_specs = {
        "d": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.ECONOMIC),
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, d=0.5, window=20, **kwargs):
        # TODO: Implement proper fractional differencing with binomial weights
        # Placeholder: weighted difference
        return feature.diff() * d + feature * (1 - d)


@register_operator(name="ts_fractional_difference_discarded_weight_mass", canonical="ts_fractional_difference_discarded_weight_mass", backend="polars", status="research_only")
class TSFractionalDifferenceDiscardedWeightMassPolarsNative(SeriesOperator):
    """Total weight discarded by finite window fractional differencing"""

    metadata = OperatorMetadata(
        name="ts_fractional_difference_discarded_weight_mass",
        category="time_series",
        description="Sum of fractional differencing weights beyond window",
        param_names=["feature", "d", "window"],
        return_type="series",
        tags=["time_series", "rolling", "memory", "pit_safe"],
    )
    metadata.param_specs = {
        "d": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.ECONOMIC),
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, d=0.5, window=20, **kwargs):
        # TODO: Implement proper weight mass calculation
        # Placeholder: constant based on d and window
        return pl.lit(d / window).cast(pl.Float64)


# ============================================================================
# Generalized Hurst Exponents
# ============================================================================

@register_operator(name="ts_generalized_hurst_exponent", canonical="ts_generalized_hurst_exponent", backend="polars", status="research_only")
class TSGeneralizedHurstExponentPolarsNative(SeriesOperator):
    """Generalized Hurst exponent H(q) for moment q"""

    metadata = OperatorMetadata(
        name="ts_generalized_hurst_exponent",
        category="time_series",
        description="Generalized Hurst exponent for given moment order q",
        param_names=["feature", "window", "q"],
        return_type="series",
        tags=["time_series", "rolling", "multifractal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "q": ParamSpec(dtype=float, min=-5.0, max=5.0, param_role=ParamRole.ECONOMIC),
    }

    def _calculate_series(self, feature, window, q=2.0, **kwargs):
        # TODO: Implement proper GHE via DFA or structure function
        # Placeholder: Hurst proxy via rescaled range
        return feature.rolling_std(window) / (feature.rolling_mean(window).abs() + 1e-8)


@register_operator(name="ts_generalized_hurst_spread_q1_q4", canonical="ts_generalized_hurst_spread_q1_q4", backend="polars", status="research_only")
class TSGeneralizedHurstSpreadQ1Q4PolarsNative(SeriesOperator):
    """Spread between H(1) and H(4) - multifractal strength"""

    metadata = OperatorMetadata(
        name="ts_generalized_hurst_spread_q1_q4",
        category="time_series",
        description="H(q=1) - H(q=4) spread indicating multifractal strength",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "multifractal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement H(1) - H(4)
        # Placeholder: volatility clustering measure
        vol = feature.rolling_std(window)
        return vol.rolling_std(window // 2)


# ============================================================================
# GPD Tail Estimation and H-infinity Filter
# ============================================================================

@register_operator(name="ts_gpd_shape_pwm", canonical="ts_gpd_shape_pwm", backend="polars", status="research_only")
class TSGPDShapePWMPolarsNative(SeriesOperator):
    """Generalized Pareto Distribution shape parameter via PWM"""

    metadata = OperatorMetadata(
        name="ts_gpd_shape_pwm",
        category="time_series",
        description="GPD shape parameter (tail index) via probability weighted moments",
        param_names=["feature", "window", "threshold_quantile"],
        return_type="series",
        tags=["time_series", "rolling", "extreme_tail", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "threshold_quantile": ParamSpec(dtype=float, min=0.5, max=0.99, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold_quantile=0.9, **kwargs):
        # TODO: Implement proper GPD PWM estimator
        # Placeholder: tail volatility
        return feature.abs().rolling_quantile(threshold_quantile, window_size=window)


@register_operator(name="ts_h_infinity_level_filter", canonical="ts_h_infinity_level_filter", backend="polars", status="research_only", replace=True, expected_old_source="pandas_bridge", replacement_reason="Consolidating polars native operators into ts_advanced_batch5")
class TSHInfinityLevelFilterPolarsNative(SeriesOperator):
    """H-infinity robust level filter (min-max optimal)"""

    metadata = OperatorMetadata(
        name="ts_h_infinity_level_filter",
        category="time_series",
        description="H-infinity robust filter for level estimation",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "filter", "robust", "pit_safe"],
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper H-infinity filter
        # Placeholder: robust moving average (median)
        return feature.rolling_median(window)


# ============================================================================
# Hampel Filter and Hankel Analysis
# ============================================================================

@register_operator(name="ts_hampel_filter_causal", canonical="ts_hampel_filter_causal", backend="polars")
class TSHampelFilterCausalPolarsNative(SeriesOperator):
    """Causal Hampel outlier filter (median + MAD-based)"""

    metadata = OperatorMetadata(
        name="ts_hampel_filter_causal",
        category="time_series",
        description="Causal Hampel filter: replace outliers with median",
        param_names=["feature", "window", "n_sigma"],
        return_type="series",
        tags=["time_series", "rolling", "filter", "robust", "causal", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "n_sigma": ParamSpec(dtype=float, min=1.0, max=10.0, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, n_sigma=3.0, **kwargs):
        # Hampel: replace values > n_sigma * MAD from median
        median = feature.rolling_median(window)
        mad = (feature - median).abs().rolling_median(window)
        threshold = n_sigma * mad * 1.4826  # MAD to std conversion
        outlier = (feature - median).abs() > threshold
        return pl.when(outlier).then(median).otherwise(feature)


@register_operator(name="ts_hankel_effective_rank", canonical="ts_hankel_effective_rank", backend="polars", status="research_only")
class TSHankelEffectiveRankPolarsNative(SeriesOperator):
    """Effective rank of Hankel matrix (entropy of singular values)"""

    metadata = OperatorMetadata(
        name="ts_hankel_effective_rank",
        category="time_series",
        description="Effective rank of trajectory Hankel matrix",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "ssa", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper Hankel SVD and effective rank
        # Placeholder: normalized variance
        return feature.rolling_var(window) / (feature.rolling_mean(window) ** 2 + 1e-8)


@register_operator(name="ts_hankel_singular_gap", canonical="ts_hankel_singular_gap", backend="polars", status="research_only")
class TSHankelSingularGapPolarsNative(SeriesOperator):
    """Gap between first and second singular value (signal strength)"""

    metadata = OperatorMetadata(
        name="ts_hankel_singular_gap",
        category="time_series",
        description="Ratio of first to second singular value of Hankel matrix",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "ssa", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper Hankel SVD
        # Placeholder: autocorrelation strength
        return feature.rolling_mean(window) / (feature.rolling_std(window) + 1e-8)


# ============================================================================
# Hartigan Dip and Higuchi Fractal Dimension
# ============================================================================

@register_operator(name="ts_hartigan_dip", canonical="ts_hartigan_dip", backend="polars", status="research_only")
class TSHartiganDipPolarsNative(SeriesOperator):
    """Hartigan dip test statistic (unimodality test)"""

    metadata = OperatorMetadata(
        name="ts_hartigan_dip",
        category="time_series",
        description="Hartigan dip statistic measuring departure from unimodality",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "distribution", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper Hartigan dip test
        # Placeholder: bimodality proxy via kurtosis
        mean = feature.rolling_mean(window)
        std = feature.rolling_std(window)
        m4 = ((feature - mean) ** 4).rolling_mean(window)
        return m4 / (std ** 4 + 1e-8) - 3.0


@register_operator(name="ts_higuchi_fractal_dimension", canonical="ts_higuchi_fractal_dimension", backend="polars", status="research_only")
class TSHiguchiFractalDimensionPolarsNative(SeriesOperator):
    """Higuchi fractal dimension (complexity measure)"""

    metadata = OperatorMetadata(
        name="ts_higuchi_fractal_dimension",
        category="time_series",
        description="Higuchi fractal dimension of time series trajectory",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "complexity", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper Higuchi algorithm
        # Placeholder: path length measure
        return feature.diff().abs().rolling_sum(window)


# ============================================================================
# Hill Tail Index and HSIC
# ============================================================================

@register_operator(name="ts_hill_tail_index", canonical="ts_hill_tail_index", backend="polars", status="research_only")
class TSHillTailIndexPolarsNative(SeriesOperator):
    """Hill estimator of tail index (extreme value shape)"""

    metadata = OperatorMetadata(
        name="ts_hill_tail_index",
        category="time_series",
        description="Hill estimator of tail thickness parameter",
        param_names=["feature", "window", "tail_fraction"],
        return_type="series",
        tags=["time_series", "rolling", "extreme_tail", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "tail_fraction": ParamSpec(dtype=float, min=0.01, max=0.5, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, tail_fraction=0.1, **kwargs):
        # TODO: Implement proper Hill estimator
        # Placeholder: tail quantile ratio
        q95 = feature.abs().rolling_quantile(0.95, window_size=window)
        q50 = feature.abs().rolling_quantile(0.50, window_size=window)
        return q95 / (q50 + 1e-8)


@register_operator(name="ts_hsic", canonical="ts_hsic", backend="polars", status="research_only")
class TSHSICPolarsNative(SeriesOperator):
    """Hilbert-Schmidt Independence Criterion (nonlinear dependence)"""

    metadata = OperatorMetadata(
        name="ts_hsic",
        category="time_series",
        description="HSIC measure of nonlinear dependence with lag",
        param_names=["feature", "lag", "window"],
        return_type="series",
        tags=["time_series", "rolling", "dependence", "pit_safe"],
    )
    metadata.param_specs = {
        "lag": ParamSpec(dtype=int, min=1, param_role=ParamRole.ECONOMIC),
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, lag, window, **kwargs):
        # TODO: Implement proper HSIC with kernel matrices
        # Placeholder: lagged correlation
        return feature.rolling_corr(feature.shift(lag), window_size=window)


# ============================================================================
# Hurst DFA
# ============================================================================

@register_operator(name="ts_hurst_dfa", canonical="ts_hurst_dfa", backend="polars", status="research_only")
class TSHurstDFAPolarsNative(SeriesOperator):
    """Hurst exponent via Detrended Fluctuation Analysis"""

    metadata = OperatorMetadata(
        name="ts_hurst_dfa",
        category="time_series",
        description="Hurst exponent estimated via DFA method",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "long_memory", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper DFA algorithm
        # Placeholder: rescaled range Hurst estimate
        r = feature.rolling_max(window) - feature.rolling_min(window)
        s = feature.rolling_std(window)
        return r / (s + 1e-8)


# ============================================================================
# HVG (Horizontal Visibility Graph) Features
# ============================================================================

@register_operator(name="ts_hvg_assortativity", canonical="ts_hvg_assortativity", backend="polars", status="research_only")
class TSHVGAssortativityPolarsNative(SeriesOperator):
    """Horizontal Visibility Graph degree assortativity"""

    metadata = OperatorMetadata(
        name="ts_hvg_assortativity",
        category="time_series",
        description="HVG Pearson correlation of node degrees",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "network", "hvg", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=8, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper HVG construction and degree assortativity
        # Placeholder: autocorrelation proxy
        return feature.rolling_corr(feature.shift(1), window_size=window)


@register_operator(name="ts_hvg_clustering_coefficient", canonical="ts_hvg_clustering_coefficient", backend="polars", status="research_only")
class TSHVGClusteringCoefficientPolarsNative(SeriesOperator):
    """Horizontal Visibility Graph mean clustering coefficient"""

    metadata = OperatorMetadata(
        name="ts_hvg_clustering_coefficient",
        category="time_series",
        description="HVG mean local clustering coefficient",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "network", "hvg", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=8, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper HVG clustering
        # Placeholder: local smoothness measure
        return (feature.diff().abs()).rolling_mean(window)


@register_operator(name="ts_hvg_degree_entropy", canonical="ts_hvg_degree_entropy", backend="polars", status="research_only")
class TSHVGDegreeEntropyPolarsNative(SeriesOperator):
    """Horizontal Visibility Graph degree distribution entropy"""

    metadata = OperatorMetadata(
        name="ts_hvg_degree_entropy",
        category="time_series",
        description="Shannon entropy of HVG degree distribution",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "network", "hvg", "entropy", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=8, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper HVG degree entropy
        # Placeholder: distribution entropy proxy
        return feature.rolling_std(window) / (feature.abs().rolling_mean(window) + 1e-8)


@register_operator(name="ts_hvg_forward_backward_asymmetry", canonical="ts_hvg_forward_backward_asymmetry", backend="polars", status="research_only")
class TSHVGForwardBackwardAsymmetryPolarsNative(SeriesOperator):
    """HVG asymmetry between forward and backward visibility"""

    metadata = OperatorMetadata(
        name="ts_hvg_forward_backward_asymmetry",
        category="time_series",
        description="Asymmetry between in-degree and out-degree distributions",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "network", "hvg", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=8, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper forward/backward degree asymmetry
        # Placeholder: trend asymmetry
        up = (feature.diff() > 0).cast(pl.Int32).rolling_mean(window)
        down = (feature.diff() < 0).cast(pl.Int32).rolling_mean(window)
        return up - down


@register_operator(name="ts_hvg_motif_entropy", canonical="ts_hvg_motif_entropy", backend="polars", status="research_only")
class TSHVGMotifEntropyPolarsNative(SeriesOperator):
    """Entropy of HVG 3-node motifs"""

    metadata = OperatorMetadata(
        name="ts_hvg_motif_entropy",
        category="time_series",
        description="Shannon entropy of 3-node motif distribution in HVG",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "network", "hvg", "motif", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper motif detection and entropy
        # Placeholder: pattern entropy proxy
        return (feature.diff().sign()).rolling_std(window)


# ============================================================================
# Hysteresis States
# ============================================================================

@register_operator(name="ts_hysteresis_age", canonical="ts_hysteresis_age", backend="polars", status="research_only")
class TSHysteresisAgePolarsNative(SeriesOperator):
    """Time since last hysteresis state switch"""

    metadata = OperatorMetadata(
        name="ts_hysteresis_age",
        category="time_series",
        description="Bars since last crossing of upper/lower threshold",
        param_names=["feature", "upper_threshold", "lower_threshold"],
        return_type="series",
        tags=["time_series", "hysteresis", "state", "pit_safe"],
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, feature, upper_threshold, lower_threshold, **kwargs):
        # TODO: Implement proper hysteresis state machine
        # Placeholder: simple threshold crossing counter
        crossed = ((feature > upper_threshold) | (feature < lower_threshold)).cast(pl.Int32)
        return crossed.cum_sum()


@register_operator(name="ts_hysteresis_state", canonical="ts_hysteresis_state", backend="polars", status="research_only")
class TSHysteresisStatePolarsNative(SeriesOperator):
    """Current hysteresis state (high/low/neutral)"""

    metadata = OperatorMetadata(
        name="ts_hysteresis_state",
        category="time_series",
        description="Current state in hysteresis band (1=high, -1=low, 0=neutral)",
        param_names=["feature", "upper_threshold", "lower_threshold"],
        return_type="series",
        tags=["time_series", "hysteresis", "state", "pit_safe"],
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, feature, upper_threshold, lower_threshold, **kwargs):
        # TODO: Implement proper stateful hysteresis
        # Placeholder: simple ternary threshold
        high = feature > upper_threshold
        low = feature < lower_threshold
        return pl.when(high).then(pl.lit(1.0)).when(low).then(pl.lit(-1.0)).otherwise(pl.lit(0.0))


# ============================================================================
# Industry and Market Liquidity Betas
# ============================================================================

@register_operator(name="ts_industry_liquidity_beta", canonical="ts_industry_liquidity_beta", backend="polars")
class TSIndustryLiquidityBetaPolarsNative(SeriesOperator):
    """Beta to industry-level liquidity factor (rolling regression)"""

    metadata = OperatorMetadata(
        name="ts_industry_liquidity_beta",
        category="time_series",
        description="Rolling beta to industry liquidity factor",
        param_names=["feature", "industry_liquidity", "window"],
        return_type="series",
        tags=["time_series", "rolling", "regression", "liquidity", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, industry_liquidity, window, **kwargs):
        # Rolling beta = cov(feature, industry) / var(industry)
        cov = (feature * industry_liquidity).rolling_mean(window) - \
              (feature.rolling_mean(window) * industry_liquidity.rolling_mean(window))
        var = industry_liquidity.rolling_var(window)
        return cov / (var + 1e-8)


@register_operator(name="ts_market_liquidity_beta", canonical="ts_market_liquidity_beta", backend="polars")
class TSMarketLiquidityBetaPolarsNative(SeriesOperator):
    """Beta to market-wide liquidity factor"""

    metadata = OperatorMetadata(
        name="ts_market_liquidity_beta",
        category="time_series",
        description="Rolling beta to market liquidity factor",
        param_names=["feature", "market_liquidity", "window"],
        return_type="series",
        tags=["time_series", "rolling", "regression", "liquidity", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, market_liquidity, window, **kwargs):
        # Rolling beta
        cov = (feature * market_liquidity).rolling_mean(window) - \
              (feature.rolling_mean(window) * market_liquidity.rolling_mean(window))
        var = market_liquidity.rolling_var(window)
        return cov / (var + 1e-8)


# ============================================================================
# Interval Geometry
# ============================================================================

@register_operator(name="ts_interval_exploration_efficiency", canonical="ts_interval_exploration_efficiency", backend="polars", status="research_only")
class TSIntervalExplorationEfficiencyPolarsNative(SeriesOperator):
    """Efficiency of exploring value range (unique intervals / total)"""

    metadata = OperatorMetadata(
        name="ts_interval_exploration_efficiency",
        category="time_series",
        description="Ratio of unique intervals explored to total intervals",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "geometry", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper interval exploration tracking
        # Placeholder: range coverage
        range_val = feature.rolling_max(window) - feature.rolling_min(window)
        return range_val / (feature.rolling_std(window) * window ** 0.5 + 1e-8)


@register_operator(name="ts_interval_nesting_depth", canonical="ts_interval_nesting_depth", backend="polars", status="research_only")
class TSIntervalNestingDepthPolarsNative(SeriesOperator):
    """Maximum depth of nested intervals"""

    metadata = OperatorMetadata(
        name="ts_interval_nesting_depth",
        category="time_series",
        description="Maximum nesting depth of value intervals in window",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "geometry", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper interval nesting analysis
        # Placeholder: volatility regime count
        return (feature.rolling_std(window) > feature.rolling_std(window * 2)).cast(pl.Float64)


@register_operator(name="ts_interval_occupancy_entropy", canonical="ts_interval_occupancy_entropy", backend="polars", status="research_only")
class TSIntervalOccupancyEntropyPolarsNative(SeriesOperator):
    """Entropy of time spent in value intervals"""

    metadata = OperatorMetadata(
        name="ts_interval_occupancy_entropy",
        category="time_series",
        description="Shannon entropy of interval occupancy distribution",
        param_names=["feature", "window", "n_bins"],
        return_type="series",
        tags=["time_series", "rolling", "entropy", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "n_bins": ParamSpec(dtype=int, min=3, max=20, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_bins=10, **kwargs):
        # TODO: Implement proper binning and entropy calculation
        # Placeholder: normalized spread
        return feature.rolling_std(window) / (feature.abs().rolling_mean(window) + 1e-8)


@register_operator(name="ts_interval_occupancy_mode_distance", canonical="ts_interval_occupancy_mode_distance", backend="polars", status="research_only")
class TSIntervalOccupancyModeDistancePolarsNative(SeriesOperator):
    """Distance from current value to most occupied interval"""

    metadata = OperatorMetadata(
        name="ts_interval_occupancy_mode_distance",
        category="time_series",
        description="Distance to mode interval center",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "geometry", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement mode detection
        # Placeholder: distance from mean
        return (feature - feature.rolling_mean(window)).abs()


@register_operator(name="ts_interval_overlap_connected_component_ratio", canonical="ts_interval_overlap_connected_component_ratio", backend="polars", status="research_only")
class TSIntervalOverlapConnectedComponentRatioPolarsNative(SeriesOperator):
    """Ratio of connected components in interval overlap graph"""

    metadata = OperatorMetadata(
        name="ts_interval_overlap_connected_component_ratio",
        category="time_series",
        description="Connected components / total intervals in overlap graph",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "graph", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement interval overlap graph analysis
        # Placeholder: regime fragmentation proxy
        return (feature.diff().sign().diff().abs() > 0).cast(pl.Float64).rolling_mean(window)


@register_operator(name="ts_interval_union_coverage", canonical="ts_interval_union_coverage", backend="polars")
class TSIntervalUnionCoveragePolarsNative(SeriesOperator):
    """Total range covered by union of intervals"""

    metadata = OperatorMetadata(
        name="ts_interval_union_coverage",
        category="time_series",
        description="Range covered by union of all intervals in window",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "geometry", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Range covered
        return feature.rolling_max(window) - feature.rolling_min(window)


# ============================================================================
# Joint Energy, Jump Detection, and Kramers-Moyal
# ============================================================================

@register_operator(name="ts_joint_energy_shift", canonical="ts_joint_energy_shift", backend="polars", status="research_only")
class TSJointEnergyShiftPolarsNative(SeriesOperator):
    """Change in joint energy between windows (distribution shift)"""

    metadata = OperatorMetadata(
        name="ts_joint_energy_shift",
        category="time_series",
        description="Shift in joint energy statistic between consecutive windows",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "distribution", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper joint energy statistic
        # Placeholder: squared moment shift
        return (feature ** 2).rolling_mean(window).diff()


@register_operator(name="ts_jump_bipower", canonical="ts_jump_bipower", backend="polars")
class TSJumpBipowerPolarsNative(SeriesOperator):
    """Bipower variation for jump detection

    NOTE: Previously named ts_jump_bipower_proxy, but this implements the correct
    bipower variation formula: sum of products of consecutive absolute returns.
    This is the standard estimator from Barndorff-Nielsen & Shephard (2004).
    """

    metadata = OperatorMetadata(
        name="ts_jump_bipower",
        category="time_series",
        description="Bipower variation for separating jumps from continuous variation",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "jumps", "volatility", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Bipower variation: sum of products of consecutive absolute returns
        abs_ret = feature.diff().abs()
        bipower = (abs_ret * abs_ret.shift(1)).rolling_sum(window)
        return bipower


@register_operator(name="ts_km_diffusion_gradient", canonical="ts_km_diffusion_gradient", backend="polars", status="research_only")
class TSKMDiffusionGradientPolarsNative(SeriesOperator):
    """Gradient of Kramers-Moyal diffusion coefficient"""

    metadata = OperatorMetadata(
        name="ts_km_diffusion_gradient",
        category="time_series",
        description="Spatial gradient of diffusion coefficient D(x)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "stochastic", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper Kramers-Moyal expansion
        # Placeholder: volatility gradient
        return feature.rolling_std(window).diff()


@register_operator(name="ts_deviation_from_mean", canonical="ts_deviation_from_mean", backend="polars")
class TSDeviationFromMeanPolarsNative(SeriesOperator):
    """Absolute deviation from rolling mean

    NOTE: Previously named ts_km_equilibrium_distance, but that name was misleading.
    True Kramers-Moyal equilibrium distance requires estimating where drift=0.
    This operator simply computes |feature - rolling_mean|, which is a valid
    measure of deviation but not KM equilibrium distance.
    """

    metadata = OperatorMetadata(
        name="ts_deviation_from_mean",
        category="time_series",
        description="Absolute deviation from rolling mean",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "deviation", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Absolute deviation from rolling mean
        return (feature - feature.rolling_mean(window)).abs()


@register_operator(name="ts_km_quasipotential_depth", canonical="ts_km_quasipotential_depth", backend="polars", status="research_only")
class TSKMQuasipotentialDepthPolarsNative(SeriesOperator):
    """Depth of quasipotential well (escape barrier)"""

    metadata = OperatorMetadata(
        name="ts_km_quasipotential_depth",
        category="time_series",
        description="Depth of potential well indicating stability",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "stochastic", "stability", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement quasipotential calculation
        # Placeholder: local volatility barrier
        return feature.rolling_std(window) * window ** 0.5


@register_operator(name="ts_kramers_moyal_diffusion", canonical="ts_kramers_moyal_diffusion", backend="polars")
class TSKramersMoyalDiffusionPolarsNative(SeriesOperator):
    """Kramers-Moyal diffusion coefficient D(x) (2nd moment)"""

    metadata = OperatorMetadata(
        name="ts_kramers_moyal_diffusion",
        category="time_series",
        description="Local diffusion coefficient from KM expansion",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "stochastic", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # KM diffusion: local variance of increments
        return (feature.diff() ** 2).rolling_mean(window)


@register_operator(name="ts_kramers_moyal_drift", canonical="ts_kramers_moyal_drift", backend="polars")
class TSKramersMoyalDriftPolarsNative(SeriesOperator):
    """Kramers-Moyal drift coefficient D1(x) (1st moment)"""

    metadata = OperatorMetadata(
        name="ts_kramers_moyal_drift",
        category="time_series",
        description="Local drift coefficient from KM expansion",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "stochastic", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # KM drift: local mean of increments
        return feature.diff().rolling_mean(window)


@register_operator(name="ts_kramers_moyal_local_stability", canonical="ts_kramers_moyal_local_stability", backend="polars", status="research_only")
class TSKramersMoyalLocalStabilityPolarsNative(SeriesOperator):
    """Local stability via negative drift gradient"""

    metadata = OperatorMetadata(
        name="ts_kramers_moyal_local_stability",
        category="time_series",
        description="Local stability: -dD1/dx (negative drift gradient)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "stochastic", "stability", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement drift gradient
        # Placeholder: mean reversion strength
        drift = feature.diff().rolling_mean(window)
        return -drift.diff()


# ============================================================================
# KS Shift and L-Moments
# ============================================================================

@register_operator(name="ts_ks_shift", canonical="ts_ks_shift", backend="polars", status="research_only")
class TSKSShiftPolarsNative(SeriesOperator):
    """Kolmogorov-Smirnov statistic between consecutive windows"""

    metadata = OperatorMetadata(
        name="ts_ks_shift",
        category="time_series",
        description="KS distance between distributions of consecutive windows",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "distribution", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper KS test between windows
        # Placeholder: quantile shift
        q50_shift = feature.rolling_median(window).diff()
        return q50_shift.abs()


@register_operator(name="ts_l_kurtosis", canonical="ts_l_kurtosis", backend="polars", status="research_only")
class TSLKurtosisPolarsNative(SeriesOperator):
    """L-kurtosis (4th L-moment ratio)"""

    metadata = OperatorMetadata(
        name="ts_l_kurtosis",
        category="time_series",
        description="L-moment based kurtosis (robust to outliers)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "moments", "robust", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper L-moments
        # Placeholder: robust kurtosis proxy
        q75 = feature.rolling_quantile(0.75, window_size=window)
        q25 = feature.rolling_quantile(0.25, window_size=window)
        q90 = feature.rolling_quantile(0.90, window_size=window)
        q10 = feature.rolling_quantile(0.10, window_size=window)
        return (q90 - q10) / (q75 - q25 + 1e-8)


@register_operator(name="ts_l_skewness", canonical="ts_l_skewness", backend="polars", status="research_only")
class TSLSkewnessPolarsNative(SeriesOperator):
    """L-skewness (3rd L-moment ratio)"""

    metadata = OperatorMetadata(
        name="ts_l_skewness",
        category="time_series",
        description="L-moment based skewness (robust to outliers)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "moments", "robust", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper L-moments
        # Placeholder: robust skewness proxy
        q75 = feature.rolling_quantile(0.75, window_size=window)
        q50 = feature.rolling_quantile(0.50, window_size=window)
        q25 = feature.rolling_quantile(0.25, window_size=window)
        return (q75 + q25 - 2 * q50) / (q75 - q25 + 1e-8)


# ============================================================================
# Lagged Correlations and Mutual Information
# ============================================================================

@register_operator(name="ts_lag_of_peak_corr", canonical="ts_lag_of_peak_corr", backend="polars", status="research_only")
class TSLagOfPeakCorrPolarsNative(SeriesOperator):
    """Lag at which autocorrelation peaks"""

    metadata = OperatorMetadata(
        name="ts_lag_of_peak_corr",
        category="time_series",
        description="Lag with maximum absolute autocorrelation",
        param_names=["feature", "window", "max_lag"],
        return_type="series",
        tags=["time_series", "rolling", "autocorrelation", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "max_lag": ParamSpec(dtype=int, min=1, max=50, param_role=ParamRole.ECONOMIC),
    }

    def _calculate_series(self, feature, window, max_lag=10, **kwargs):
        # TODO: Implement proper lag search
        # Placeholder: fixed lag-1 correlation
        return feature.rolling_corr(feature.shift(1), window_size=window)


@register_operator(name="ts_lagged_mutual_information", canonical="ts_lagged_mutual_information", backend="polars", status="research_only")
class TSLaggedMutualInformationPolarsNative(SeriesOperator):
    """Mutual information between series and its lag"""

    metadata = OperatorMetadata(
        name="ts_lagged_mutual_information",
        category="time_series",
        description="MI between x(t) and x(t-lag)",
        param_names=["feature", "lag", "window"],
        return_type="series",
        tags=["time_series", "rolling", "information", "pit_safe"],
    )
    metadata.param_specs = {
        "lag": ParamSpec(dtype=int, min=1, param_role=ParamRole.ECONOMIC),
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, lag, window, **kwargs):
        # TODO: Implement proper MI estimation
        # Placeholder: squared correlation
        corr = feature.rolling_corr(feature.shift(lag), window_size=window)
        return corr ** 2


# ============================================================================
# Lempel-Ziv Complexity and Level Shifts
# ============================================================================

@register_operator(name="ts_lempel_ziv_complexity", canonical="ts_lempel_ziv_complexity", backend="polars", status="research_only")
class TSLempelZivComplexityPolarsNative(SeriesOperator):
    """Lempel-Ziv complexity (sequence compressibility)"""

    metadata = OperatorMetadata(
        name="ts_lempel_ziv_complexity",
        category="time_series",
        description="LZ complexity: number of distinct patterns",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "complexity", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper LZ76 complexity
        # Placeholder: pattern diversity proxy
        return (feature.diff().sign().diff().abs() > 0).cast(pl.Float64).rolling_sum(window)


@register_operator(name="ts_level_shift_score", canonical="ts_level_shift_score", backend="polars", status="research_only")
class TSLevelShiftScorePolarsNative(SeriesOperator):
    """Likelihood score of level shift at current point"""

    metadata = OperatorMetadata(
        name="ts_level_shift_score",
        category="time_series",
        description="Statistical evidence of level shift",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "breakpoint", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper CUSUM or likelihood ratio test
        # Placeholder: deviation from long-term mean
        short_mean = feature.rolling_mean(window // 4)
        long_mean = feature.rolling_mean(window)
        return (short_mean - long_mean).abs() / (feature.rolling_std(window) + 1e-8)


# ============================================================================
# Leverage Effect and Line Geometry
# ============================================================================

@register_operator(name="ts_leverage_effect", canonical="ts_leverage_effect", backend="polars")
class TSLeverageEffectPolarsNative(SeriesOperator):
    """Causal leverage effect: correlation of lagged return and trailing volatility."""

    metadata = OperatorMetadata(
        name="ts_leverage_effect",
        category="time_series",
        description="Causal correlation between lagged returns and current trailing volatility",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "asymmetry", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Pair r[t-1] with volatility known at t: the PIT-safe historical form
        # of corr(r[tau], vol[tau+1]).
        ret = feature.diff()
        lagged_ret = ret.shift(1)
        trailing_vol = ret.rolling_std(window)
        cross_mean = (lagged_ret * trailing_vol).rolling_mean(window)
        covariance = cross_mean - (
            lagged_ret.rolling_mean(window) * trailing_vol.rolling_mean(window)
        )
        scale = lagged_ret.rolling_std(window, ddof=0) * trailing_vol.rolling_std(
            window, ddof=0
        )
        return covariance / scale


@register_operator(name="ts_line_convergence", canonical="ts_line_convergence", backend="polars")
class TSLineConvergencePolarsNative(SeriesOperator):
    """Convergence of two trend lines"""

    metadata = OperatorMetadata(
        name="ts_line_convergence",
        category="time_series",
        description="Rate of convergence between short and long trend lines",
        param_names=["feature", "short_window", "long_window"],
        return_type="series",
        tags=["time_series", "rolling", "trend", "pit_safe"],
    )
    metadata.param_specs = {
        "short_window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "long_window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, short_window, long_window, **kwargs):
        # Convergence: rate of change of spread
        short_trend = feature.rolling_mean(short_window)
        long_trend = feature.rolling_mean(long_window)
        spread = short_trend - long_trend
        return spread.diff()


@register_operator(name="ts_line_parallelism", canonical="ts_line_parallelism", backend="polars")
class TSLineParallelismPolarsNative(SeriesOperator):
    """Parallelism of two trend lines"""

    metadata = OperatorMetadata(
        name="ts_line_parallelism",
        category="time_series",
        description="Similarity of slopes between short and long trend lines",
        param_names=["feature", "short_window", "long_window"],
        return_type="series",
        tags=["time_series", "rolling", "trend", "pit_safe"],
    )
    metadata.param_specs = {
        "short_window": ParamSpec(dtype=int, min=5, param_role=ParamRole.HORIZON),
        "long_window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, short_window, long_window, **kwargs):
        # Parallelism: ratio of slopes
        short_slope = feature.rolling_mean(short_window).diff()
        long_slope = feature.rolling_mean(long_window).diff()
        return short_slope / (long_slope + 1e-8)


# ============================================================================
# Lo-MacKinlay Variance Ratio
# ============================================================================

@register_operator(name="ts_lo_mackinlay_vr", canonical="ts_lo_mackinlay_vr", backend="polars")
class TSLoMackinlayVRPolarsNative(SeriesOperator):
    """Lo-MacKinlay variance ratio test statistic"""

    metadata = OperatorMetadata(
        name="ts_lo_mackinlay_vr",
        category="time_series",
        description="Variance ratio VR(q) = Var(q-period) / (q * Var(1-period))",
        param_names=["feature", "q", "window"],
        return_type="series",
        tags=["time_series", "rolling", "random_walk", "pit_safe"],
    )
    metadata.param_specs = {
        "q": ParamSpec(dtype=int, min=2, param_role=ParamRole.ECONOMIC),
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, q, window, **kwargs):
        # VR(q) = Var(q-period return) / (q * Var(1-period return))
        ret_1 = feature.diff()
        ret_q = feature.diff(q)
        var_1 = ret_1.rolling_var(window)
        var_q = ret_q.rolling_var(window)
        return var_q / (q * var_1 + 1e-8)


@register_operator(name="ts_lo_mackinlay_z", canonical="ts_lo_mackinlay_z", backend="polars")
class TSLoMackinlayZPolarsNative(SeriesOperator):
    """Lo-MacKinlay variance ratio Z-statistic"""

    metadata = OperatorMetadata(
        name="ts_lo_mackinlay_z",
        category="time_series",
        description="Standardized variance ratio test statistic",
        param_names=["feature", "q", "window"],
        return_type="series",
        tags=["time_series", "rolling", "random_walk", "pit_safe"],
    )
    metadata.param_specs = {
        "q": ParamSpec(dtype=int, min=2, param_role=ParamRole.ECONOMIC),
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, q, window, **kwargs):
        # Z = (VR - 1) / sqrt(asymptotic variance)
        ret_1 = feature.diff()
        ret_q = feature.diff(q)
        var_1 = ret_1.rolling_var(window)
        var_q = ret_q.rolling_var(window)
        vr = var_q / (q * var_1 + 1e-8)
        # Simplified asymptotic variance
        asy_var = 2.0 * (q - 1) / (3 * q * window)
        return (vr - 1.0) / (asy_var ** 0.5 + 1e-8)


# ============================================================================
# Local Lyapunov Exponent and Tail Coexceedance
# ============================================================================

# QUARANTINED: explicitly research-only placeholder; it must not compete
# with the canonical generated bridge registration in auto_polars_all.py.
class TSLocalLyapunovExponentPolarsNative(SeriesOperator):
    """Local Lyapunov exponent (chaos measure)

    NOTE: This operator is marked research_only. True local Lyapunov exponent requires:
    1. Phase space reconstruction via time-delay embedding
    2. Tracking divergence of nearby trajectories
    3. Computing exponential rate of separation

    Current implementation is a placeholder: volatility change rate.
    """

    metadata = OperatorMetadata(
        name="ts_local_lyapunov_exponent",
        category="time_series",
        description="[RESEARCH ONLY] Local rate of divergence of nearby trajectories (requires embedding)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "chaos", "nonlinear", "pit_safe", "research_only"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Placeholder: volatility expansion rate (NOT true Lyapunov exponent)
        vol = feature.rolling_std(window)
        return vol.diff() / (vol + 1e-8)


@register_operator(name="ts_lower_tail_coexceedance_probability", canonical="ts_lower_tail_coexceedance_probability", backend="polars")
class TSLowerTailCoexceedanceProbabilityPolarsNative(SeriesOperator):
    """Probability of joint lower tail exceedance"""

    metadata = OperatorMetadata(
        name="ts_lower_tail_coexceedance_probability",
        category="time_series",
        description="Rolling probability of joint lower tail exceedance with lag",
        param_names=["feature", "threshold", "lag", "window"],
        return_type="series",
        tags=["time_series", "rolling", "tail_dependence", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "lag": ParamSpec(dtype=int, min=1, param_role=ParamRole.ECONOMIC),
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, lag, window, **kwargs):
        # Joint exceedance probability
        exceed_t = (feature < threshold).cast(pl.Int32)
        exceed_lag = (feature.shift(lag) < threshold).cast(pl.Int32)
        joint = (exceed_t & exceed_lag).cast(pl.Float64)
        return joint.rolling_mean(window)


# ============================================================================
# Markov Chain Features
# ============================================================================

# QUARANTINED: explicitly research-only placeholder; it must not compete
# with the canonical generated bridge registration in auto_polars_all.py.
class TSMarkovCommittorPolarsNative(SeriesOperator):
    """Committor probability (probability of reaching state B before A)

    NOTE: This operator is marked research_only. True committor probability requires:
    1. Discretizing state space into bins
    2. Estimating transition matrix from historical data
    3. Solving linear system for committor probabilities

    Current implementation is a placeholder: normalized position in threshold band.
    """

    metadata = OperatorMetadata(
        name="ts_markov_committor",
        category="time_series",
        description="[RESEARCH ONLY] Probability of reaching upper threshold before lower (requires Markov chain estimation)",
        param_names=["feature", "lower_threshold", "upper_threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "markov", "pit_safe", "research_only"],
    )
    metadata.param_specs = {
        "lower_threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "upper_threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, lower_threshold, upper_threshold, window, **kwargs):
        # Placeholder: normalized position in band (NOT true committor probability)
        position = (feature - lower_threshold) / (upper_threshold - lower_threshold + 1e-8)
        return position.clip(0.0, 1.0).rolling_mean(window)


@register_operator(name="ts_markov_entropy_production", canonical="ts_markov_entropy_production", backend="polars")
class TSMarkovEntropyProductionPolarsNative(SeriesOperator):
    """Markov entropy production rate (irreversibility measure)

    NOTE: This operator is marked research_only. True entropy production requires:
    1. State space discretization
    2. Transition matrix estimation
    3. Computing entropy production from forward/reverse transitions

    Current implementation is a placeholder: absolute directional bias.
    """

    metadata = OperatorMetadata(
        name="ts_markov_entropy_production",
        category="time_series",
        description="[RESEARCH ONLY] Rate of entropy production in discretized state space (requires transition matrix)",
        param_names=["feature", "window", "n_bins"],
        return_type="series",
        tags=["time_series", "rolling", "markov", "entropy", "pit_safe", "research_only"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_bins": ParamSpec(dtype=int, min=3, max=10, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_bins=5, **kwargs):
        # Placeholder: directional bias (NOT true entropy production)
        up = (feature.diff() > 0).cast(pl.Int32).rolling_mean(window)
        down = (feature.diff() < 0).cast(pl.Int32).rolling_mean(window)
        return (up - down).abs()


@register_operator(name="ts_markov_mean_first_passage_time", canonical="ts_markov_mean_first_passage_time", backend="polars")
class TSMarkovMeanFirstPassageTimePolarsNative(SeriesOperator):
    """Mean first passage time to threshold

    NOTE: This operator is marked research_only. True MFPT requires:
    1. State space discretization
    2. Transition matrix estimation
    3. Solving for expected hitting times

    Current implementation is a placeholder: inverse threshold crossing rate.
    """

    metadata = OperatorMetadata(
        name="ts_markov_mean_first_passage_time",
        category="time_series",
        description="[RESEARCH ONLY] Expected time to reach threshold (requires Markov chain estimation)",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "markov", "pit_safe", "research_only"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        # Placeholder: inverse crossing rate (NOT true MFPT)
        crossed = (feature > threshold).cast(pl.Int32).diff().abs()
        crossing_rate = crossed.rolling_mean(window)
        return 1.0 / (crossing_rate + 1e-8)


@register_operator(name="ts_lag1_autocorr", canonical="ts_lag1_autocorr", backend="polars")
class TSLag1AutocorrPolarsNative(SeriesOperator):
    """Lag-1 autocorrelation (rolling correlation with previous period)

    NOTE: Previously named ts_markov_persistence, but that name was misleading.
    True Markov persistence is diagonal dominance of transition matrix.
    This operator computes rolling lag-1 autocorrelation, which is a valid
    measure of short-term persistence but not Markov chain persistence.
    """

    metadata = OperatorMetadata(
        name="ts_lag1_autocorr",
        category="time_series",
        description="Rolling lag-1 autocorrelation (correlation with previous period)",
        param_names=["feature", "window", "n_bins"],
        return_type="series",
        tags=["time_series", "rolling", "autocorrelation", "persistence", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "n_bins": ParamSpec(dtype=int, min=3, max=10, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, n_bins=5, **kwargs):
        # Lag-1 autocorrelation
        return feature.rolling_corr(feature.shift(1), window_size=window)


# ============================================================================
# Registration Summary
# ============================================================================
# This module implements 64 advanced time series operators covering:
# - Distance correlation and feature geometry (3 operators)
# - FIR filters and first passage times (4 operators)
# - Fisher information and forbidden patterns (2 operators)
# - Fractional differencing (2 operators)
# - Generalized Hurst exponents (2 operators)
# - GPD tail estimation and H-infinity filter (2 operators)
# - Hampel filter and Hankel analysis (4 operators)
# - Hartigan dip and Higuchi fractal dimension (2 operators)
# - Hill tail index and HSIC (2 operators)
# - Hurst DFA (1 operator)
# - HVG features (5 operators)
# - Hysteresis states (2 operators)
# - Industry and market liquidity betas (2 operators)
# - Interval geometry (6 operators)
# - Joint energy, jump detection, and Kramers-Moyal (9 operators)
# - KS shift and L-moments (3 operators)
# - Lagged correlations and mutual information (2 operators)
# - Lempel-Ziv complexity and level shifts (2 operators)
# - Leverage effect and line geometry (3 operators)
# - Lo-MacKinlay variance ratio (2 operators)
# - Local Lyapunov exponent and tail coexceedance (2 operators)
# - Markov chain features (4 operators)
#
# Many complex algorithms (HVG, Kramers-Moyal, Markov chains, Lyapunov exponents,
# L-moments, first passage times) are implemented as skeletons with TODO markers
# for proper implementation.
#
# All operators are registered with backend="polars" and use lazy evaluation
# for maximum performance.

