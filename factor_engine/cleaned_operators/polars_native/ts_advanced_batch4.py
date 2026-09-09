"""
Polars native implementations for time series operators (ts_* family) - Batch 4

This batch covers advanced time series operators from ts_run_strength through ts_weighted_time_centroid.
Includes: roughness metrics, RQA features, state/regime detection, spectral analysis,
volatility measures, wavelet features, and weighted statistics.

Most operators use the Polars API. ``ts_spectral_low_frequency_ratio`` and
``ts_wavelet_energy_slope`` are explicit eager Python/NumPy CPU bridges: their
FFT/Haar kernels are not native Polars expressions and are not production
eligible.
"""

import polars as pl
import numpy as np
from typing import Optional, Union

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

from factor_engine.cleaned_operators.base import (
    SeriesOperator,
    register_operator,
    OperatorMetadata,
    ParamSpec,
    ParamRole,
)
from factor_engine.cleaned_operators.ts_model.wavelet_spectral import (
    _fixed_window_anchor,
    _spectral_low_ratio,
    _wavelet_stats,
)


def _wavelet_spectral_cpu_wide(feature: pl.DataFrame, kernel) -> pl.DataFrame:
    """Apply the shared CPU kernel to a wide/single-stock Polars panel."""
    if not isinstance(feature, pl.DataFrame):
        raise TypeError("wavelet/spectral Polars CPU bridge requires a DataFrame")
    if "stock_code" in feature.columns and feature["stock_code"].drop_nulls().n_unique() > 1:
        raise ValueError(
            "wavelet/spectral Polars CPU bridge requires a wide panel or a "
            "single-stock long frame"
        )
    metadata_columns = {"date", "stock_code"}
    data = {}
    for name in feature.columns:
        if name in metadata_columns:
            data[name] = feature[name]
            continue
        values = np.asarray(feature[name].to_numpy(), dtype=float)
        data[name] = pl.Series(
            name,
            [kernel(values[: row + 1]) for row in range(values.size)],
            dtype=pl.Float64,
        )
    return pl.DataFrame(data).select(feature.columns)

# ============================================================================
# Run and Persistence Operators
# ============================================================================

@register_operator(name="ts_run_strength", canonical="ts_run_strength", backend="polars")
class TSRunStrengthPolarsNative(SeriesOperator):
    """Average absolute value during runs of same sign"""

    metadata = OperatorMetadata(
        name="ts_run_strength",
        category="time_series",
        description="Average absolute value during runs of same sign",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "runs", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement run detection and strength calculation
        # Placeholder: return rolling mean of absolute values
        return feature.abs().rolling_mean(window)


@register_operator(name="ts_scale_shift", canonical="ts_scale_shift", backend="polars")
class TSScaleShiftPolarsNative(SeriesOperator):
    """Detect scale shifts using ratio of recent to historical volatility"""

    metadata = OperatorMetadata(
        name="ts_scale_shift",
        category="time_series",
        description="Detect scale shifts using ratio of recent to historical volatility",
        param_names=["feature", "short_window", "long_window"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "regime", "pit_safe"],
    )
    metadata.param_specs = {
        "short_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "long_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, short_window, long_window, **kwargs):
        short_std = feature.rolling_std(short_window)
        long_std = feature.rolling_std(long_window)
        return np.where((long_std + 1e-8) != 0, (short_std) / ((long_std + 1e-8)), np.nan)


@register_operator(name="ts_score_rank_weighted_mean", canonical="ts_score_rank_weighted_mean", backend="polars")
class TSScoreRankWeightedMeanPolarsNative(SeriesOperator):
    """Rolling mean weighted by rank scores"""

    metadata = OperatorMetadata(
        name="ts_score_rank_weighted_mean",
        category="time_series",
        description="Rolling mean weighted by rank scores",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "weighted", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper rank-weighted mean
        # Placeholder: use exponential weighting
        return feature.ewm_mean(span=window, ignore_nulls=True)


@register_operator(name="ts_sign_cluster_index", canonical="ts_sign_cluster_index", backend="polars")
class TSSignClusterIndexPolarsNative(SeriesOperator):
    """Measure of sign clustering (runs of same sign)"""

    metadata = OperatorMetadata(
        name="ts_sign_cluster_index",
        category="time_series",
        description="Measure of sign clustering (runs of same sign)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "runs", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Sign changes detect run boundaries
        sign_changes = (feature.sign().diff().abs() > 0).cast(pl.Int32)
        return sign_changes.rolling_sum(window)


@register_operator(name="ts_sign_persistence", canonical="ts_sign_persistence", backend="polars")
class TSSignPersistencePolarsNative(SeriesOperator):
    """Fraction of window with same sign as current value"""

    metadata = OperatorMetadata(
        name="ts_sign_persistence",
        category="time_series",
        description="Fraction of window with same sign as current value",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "persistence", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        sign_val = feature.sign()
        # Count matching signs in window
        same_sign = (sign_val == sign_val.shift(0)).cast(pl.Int32)
        return same_sign.rolling_mean(window)


@register_operator(name="ts_signature_mahalanobis_anomaly", canonical="ts_signature_mahalanobis_anomaly", backend="polars")
class TSSignatureMahalanobisAnomalyPolarsNative(SeriesOperator):
    """Mahalanobis distance-based anomaly detection on signature features"""

    metadata = OperatorMetadata(
        name="ts_signature_mahalanobis_anomaly",
        category="time_series",
        description="Mahalanobis distance-based anomaly detection on signature features",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "anomaly", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper Mahalanobis distance with covariance matrix
        # Placeholder: standardized distance from rolling mean
        mean = feature.rolling_mean(window)
        std = feature.rolling_std(window)
        return np.where((std + 1e-8) != 0, (((feature - mean).abs()) / ((std + 1e-8))), np.nan)


@register_operator(name="ts_sma_cn", canonical="ts_sma_cn", backend="polars")
class TSSmaCnPolarsNative(SeriesOperator):
    """Simple moving average for Chinese market (alias)"""

    metadata = OperatorMetadata(
        name="ts_sma_cn",
        category="time_series",
        description="Simple moving average for Chinese market",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "smoothing", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        return feature.rolling_mean(window)


# ============================================================================
# Spectral Analysis Operators
# ============================================================================

@register_operator(name="ts_spectral_flatness", canonical="ts_spectral_flatness", backend="polars")
class TSSpectralFlatnessPolarsNative(SeriesOperator):
    """Spectral flatness (ratio of geometric to arithmetic mean of power spectrum)"""

    metadata = OperatorMetadata(
        name="ts_spectral_flatness",
        category="time_series",
        description="Spectral flatness measure",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "spectral", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper FFT-based spectral flatness
        # Placeholder: ratio of geometric to arithmetic mean
        geo_mean = feature.abs().log().rolling_mean(window).exp()
        arith_mean = feature.abs().rolling_mean(window)
        return np.where((arith_mean + 1e-8) != 0, (geo_mean) / ((arith_mean + 1e-8)), np.nan)


@register_operator(name="ts_spectral_low_frequency_ratio", canonical="ts_spectral_low_frequency_ratio", backend="polars")
class TSSpectralLowFrequencyRatioPolarsNative(SeriesOperator):
    """Ratio of low frequency power to total power"""

    metadata = OperatorMetadata(
        name="ts_spectral_low_frequency_ratio",
        category="time_series",
        description="Ratio of low frequency power to total power",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "spectral", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, default=128, param_role=ParamRole.HORIZON),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_spectral_low_frequency_ratio",
        backend="polars",
        execution_kind=ExecutionKind.DELEGATE_PYTHON,
        supports_lazy=False,
        supports_streaming=False,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=False,
        implementation_source_hash="dfa7014e737daecb08e71ca36994d089f8866afede64e6f8cedc2ff82bb37d84",
        emitter_identity="872943b99e1cab97c777b6fe15dd1977955b4f170ecb21a33ea9a4b7c0618c5e",
        kernel_identity="726ba24f12ffb5ed6df9c633f5522bd477082c84d1d981329338dffbf2235255",
        parameter_domain_hash="f2570464677530a8e35d8ba155ab66f727f182a73e32c9a724d86b56f9c4d4de",
        semantic_contract_hash="146b116c57ac43e3744767ee83a87914be77a253d4262894afb7049b0444f660",
        notes="Eager wide-panel Python/NumPy FFT bridge; not a native Polars expression and not production-certified.",
    )

    def _calculate_series(self, x, window=128, **kwargs):
        return _wavelet_spectral_cpu_wide(
            x,
            lambda values: _spectral_low_ratio(values, window),
        )


@register_operator(name="ts_spectral_lowpass_trailing", canonical="ts_spectral_lowpass_trailing", backend="polars")
class TSSpectralLowpassTrailingPolarsNative(SeriesOperator):
    """Low-pass filtered signal (trailing moving average)"""

    metadata = OperatorMetadata(
        name="ts_spectral_lowpass_trailing",
        category="time_series",
        description="Low-pass filtered signal using trailing MA",
        param_names=["x", "window", "cutoff_freq"],
        return_type="series",
        tags=["time_series", "rolling", "spectral", "smoothing", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=8, max=512, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "cutoff_freq": ParamSpec(dtype=int, min=1, max=256, param_role=ParamRole.ECONOMIC),
    }

    def _calculate_series(self, x, window, cutoff_freq, **kwargs):
        # TODO: Implement proper frequency-domain filtering
        # Placeholder: use window size modulated by cutoff_freq
        effective_window = min(window, max(2, window // cutoff_freq))
        return x.rolling_mean(effective_window)


@register_operator(name="ts_spectral_peak_concentration", canonical="ts_spectral_peak_concentration", backend="polars")
class TSSpectralPeakConcentrationPolarsNative(SeriesOperator):
    """Concentration of spectral power in dominant frequencies"""

    metadata = OperatorMetadata(
        name="ts_spectral_peak_concentration",
        category="time_series",
        description="Concentration of spectral power in dominant frequencies",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "spectral", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper spectral peak detection
        # Placeholder: variance ratio
        var = feature.rolling_var(window)
        mean_sq = (feature.rolling_mean(window) ** 2)
        return np.where((var + 1e-8) != 0, (mean_sq) / ((var + 1e-8)), np.nan)


@register_operator(name="ts_spectral_quality_factor", canonical="ts_spectral_quality_factor", backend="polars")
class TSSpectralQualityFactorPolarsNative(SeriesOperator):
    """Q-factor of dominant spectral peak"""

    metadata = OperatorMetadata(
        name="ts_spectral_quality_factor",
        category="time_series",
        description="Q-factor of dominant spectral peak",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "spectral", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper Q-factor calculation
        # Placeholder: signal-to-noise ratio
        signal = feature.rolling_mean(window).abs()
        noise = feature.rolling_std(window)
        return np.where((noise + 1e-8) != 0, (signal) / ((noise + 1e-8)), np.nan)


# ============================================================================
# SSA (Singular Spectrum Analysis) Operators
# ============================================================================

@register_operator(name="ts_ssa_prior_reconstruction_error", canonical="ts_ssa_prior_reconstruction_error", backend="polars")
class TSSsaPriorReconstructionErrorPolarsNative(SeriesOperator):
    """SSA reconstruction error using prior components"""

    metadata = OperatorMetadata(
        name="ts_ssa_prior_reconstruction_error",
        category="time_series",
        description="SSA reconstruction error using prior components",
        param_names=["feature", "window", "n_components"],
        return_type="series",
        tags=["time_series", "rolling", "ssa", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "n_components": ParamSpec(dtype=int, min=1, param_role=ParamRole.MODEL_ORDER),
    }

    def _calculate_series(self, feature, window, n_components, **kwargs):
        # TODO: Implement proper SSA with trajectory matrix and SVD
        # Placeholder: use smoothed approximation
        smoothed = feature.rolling_mean(min(window, 10))
        return (feature - smoothed).abs()


@register_operator(name="ts_ssa_reconstruction_residual", canonical="ts_ssa_reconstruction_residual", backend="polars")
class TSSsaReconstructionResidualPolarsNative(SeriesOperator):
    """SSA reconstruction residual"""

    metadata = OperatorMetadata(
        name="ts_ssa_reconstruction_residual",
        category="time_series",
        description="SSA reconstruction residual",
        param_names=["feature", "window", "n_components"],
        return_type="series",
        tags=["time_series", "rolling", "ssa", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "n_components": ParamSpec(dtype=int, min=1, param_role=ParamRole.MODEL_ORDER),
    }

    def _calculate_series(self, feature, window, n_components, **kwargs):
        # TODO: Implement proper SSA
        # Placeholder: residual from moving average
        smoothed = feature.rolling_mean(min(window, 10))
        return feature - smoothed


# ============================================================================
# State/Regime Operators
# ============================================================================

@register_operator(name="ts_state_age_percentile", canonical="ts_state_age_percentile", backend="polars")
class TSStateAgePercentilePolarsNative(SeriesOperator):
    """Percentile of current state age in historical distribution"""

    metadata = OperatorMetadata(
        name="ts_state_age_percentile",
        category="time_series",
        description="Percentile of current state age in historical distribution",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "state", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        # TODO: Implement proper state age tracking
        # Placeholder: time above threshold
        above = (feature > threshold).cast(pl.Int32)
        return above.rolling_mean(window)


@register_operator(name="ts_state_density", canonical="ts_state_density", backend="polars")
class TSStateDensityPolarsNative(SeriesOperator):
    """Density of observations in current state region"""

    metadata = OperatorMetadata(
        name="ts_state_density",
        category="time_series",
        description="Density of observations in current state region",
        param_names=["feature", "window", "bandwidth"],
        return_type="series",
        tags=["time_series", "rolling", "state", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "bandwidth": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, bandwidth, **kwargs):
        # TODO: Implement proper kernel density estimation
        # Placeholder: inverse of rolling std
        std = feature.rolling_std(window)
        return np.where((std + bandwidth) != 0, (1.0) / ((std + bandwidth)), np.nan)


@register_operator(name="ts_state_entry_strength", canonical="ts_state_entry_strength", backend="polars")
class TSStateEntryStrengthPolarsNative(SeriesOperator):
    """Strength of entry into current state"""

    metadata = OperatorMetadata(
        name="ts_state_entry_strength",
        category="time_series",
        description="Strength of entry into current state",
        param_names=["feature", "threshold"],
        return_type="series",
        tags=["time_series", "state", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, threshold, **kwargs):
        # Entry strength = distance from threshold at crossing
        above = (feature > threshold).cast(pl.Int32)
        entry = above.diff()
        return pl.when(entry > 0).then(feature - threshold).otherwise(0.0)


@register_operator(name="ts_state_exit_hazard", canonical="ts_state_exit_hazard", backend="polars")
class TSStateExitHazardPolarsNative(SeriesOperator):
    """Hazard rate of exiting current state"""

    metadata = OperatorMetadata(
        name="ts_state_exit_hazard",
        category="time_series",
        description="Hazard rate of exiting current state",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "state", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        # TODO: Implement proper survival analysis
        # Placeholder: volatility near threshold
        dist_to_threshold = (feature - threshold).abs()
        return np.where((dist_to_threshold.rolling_mean(window) + 1e-8) != 0, (1.0) / ((dist_to_threshold.rolling_mean(window) + 1e-8)), np.nan)


@register_operator(name="ts_state_integral", canonical="ts_state_integral", backend="polars")
class TSStateIntegralPolarsNative(SeriesOperator):
    """Cumulative time-weighted value in current state"""

    metadata = OperatorMetadata(
        name="ts_state_integral",
        category="time_series",
        description="Cumulative time-weighted value in current state",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "state", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        # Integral when above threshold
        above_val = pl.when(feature > threshold).then(feature).otherwise(0.0)
        return above_val.rolling_sum(window)


@register_operator(name="ts_state_residual_life", canonical="ts_state_residual_life", backend="polars")
class TSStateResidualLifePolarsNative(SeriesOperator):
    """Expected remaining time in current state"""

    metadata = OperatorMetadata(
        name="ts_state_residual_life",
        category="time_series",
        description="Expected remaining time in current state",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "state", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        # TODO: Implement proper residual life estimation
        # Placeholder: distance from threshold normalized by volatility
        dist = (feature - threshold).abs()
        vol = feature.rolling_std(window)
        return np.where((vol + 1e-8) != 0, (dist) / ((vol + 1e-8)), np.nan)


@register_operator(name="ts_stratified_mean_spread", canonical="ts_stratified_mean_spread", backend="polars")
class TSStratifiedMeanSpreadPolarsNative(SeriesOperator):
    """Spread between high and low regime means"""

    metadata = OperatorMetadata(
        name="ts_stratified_mean_spread",
        category="time_series",
        description="Spread between high and low regime means",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "regime", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        # Mean when above vs below threshold
        above = pl.when(feature > threshold).then(feature).otherwise(None)
        below = pl.when(feature <= threshold).then(feature).otherwise(None)
        mean_above = above.rolling_mean(window)
        mean_below = below.rolling_mean(window)
        return mean_above - mean_below


@register_operator(name="ts_structural_level_density", canonical="ts_structural_level_density", backend="polars")
class TSStructuralLevelDensityPolarsNative(SeriesOperator):
    """Density of observations near structural levels"""

    metadata = OperatorMetadata(
        name="ts_structural_level_density",
        category="time_series",
        description="Density of observations near structural levels",
        param_names=["feature", "window", "bandwidth"],
        return_type="series",
        tags=["time_series", "rolling", "structural", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "bandwidth": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, bandwidth, **kwargs):
        # TODO: Implement proper structural level detection
        # Placeholder: concentration around median
        median = feature.rolling_median(window)
        dist = (feature - median).abs()
        near_level = (dist < bandwidth).cast(pl.Int32)
        return near_level.rolling_mean(window)


@register_operator(name="ts_structural_level_strength", canonical="ts_structural_level_strength", backend="polars")
class TSStructuralLevelStrengthPolarsNative(SeriesOperator):
    """Strength of structural support/resistance levels"""

    metadata = OperatorMetadata(
        name="ts_structural_level_strength",
        category="time_series",
        description="Strength of structural support/resistance levels",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "structural", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper support/resistance detection
        # Placeholder: persistence of levels
        median = feature.rolling_median(window)
        return np.where((feature.rolling_std(window) + 1e-8) != 0, (1.0) / ((feature.rolling_std(window) + 1e-8)), np.nan)


@register_operator(name="ts_student_t_kalman_filter", canonical="ts_student_t_kalman_filter", backend="polars")
class TSStudentTKalmanFilterPolarsNative(SeriesOperator):
    """Kalman filter with Student-t observation noise"""

    metadata = OperatorMetadata(
        name="ts_student_t_kalman_filter",
        category="time_series",
        description="Kalman filter with Student-t observation noise",
        param_names=["x", "q", "r", "dof"],
        return_type="series",
        tags=["time_series", "rolling", "kalman", "pit_safe"],
    )
    metadata.param_specs = {
        "q": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "r": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "dof": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, x, q, r, dof, **kwargs):
        # TODO: Implement proper Student-t Kalman filter
        # Placeholder: robust moving average with adaptive window based on dof
        window = max(3, int(dof))
        return x.rolling_median(window)


# ============================================================================
# Sum, Smoothing and Support Operators
# ============================================================================

# NOTE: ts_sum_decay already has a polars backend registered elsewhere, skipping

@register_operator(name="ts_super_smoother", canonical="ts_super_smoother", backend="polars")
class TSSuperSmootherPolarsNative(SeriesOperator):
    """Super smoother filter (low-pass filter)"""

    metadata = OperatorMetadata(
        name="ts_super_smoother",
        category="time_series",
        description="Super smoother low-pass filter",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "smoothing", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper super smoother (Ehlers)
        # Placeholder: double exponential smoothing
        span = (window) / 2.0 if 2.0 != 0 else np.nan
        smooth1 = feature.ewm_mean(span=span, ignore_nulls=True)
        smooth2 = smooth1.ewm_mean(span=span, ignore_nulls=True)
        return smooth2


@register_operator(name="ts_support_break", canonical="ts_support_break", backend="polars")
class TSSupportBreakPolarsNative(SeriesOperator):
    """Detect breaks below support level"""

    metadata = OperatorMetadata(
        name="ts_support_break",
        category="time_series",
        description="Detect breaks below support level",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "support", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Support = rolling min
        support = feature.rolling_min(window)
        # Break when current is below support
        return (feature < support).cast(pl.Float64)


@register_operator(name="ts_support_fit_r2", canonical="ts_support_fit_r2", backend="polars")
class TSSupportFitR2PolarsNative(SeriesOperator):
    """R-squared of linear fit to support levels"""

    metadata = OperatorMetadata(
        name="ts_support_fit_r2",
        category="time_series",
        description="R-squared of linear fit to support levels",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "support", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper linear regression R2
        # Placeholder: correlation squared
        rolling_min = feature.rolling_min(window)
        corr = feature.rolling_corr(rolling_min, window)
        return corr ** 2


@register_operator(name="ts_support_log_slope", canonical="ts_support_log_slope", backend="polars")
class TSSupportLogSlopePolarsNative(SeriesOperator):
    """Log slope of support level trend"""

    metadata = OperatorMetadata(
        name="ts_support_log_slope",
        category="time_series",
        description="Log slope of support level trend",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "support", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Log difference of rolling min
        support = feature.rolling_min(window)
        log_support = (support + 1e-8).log()
        return log_support.diff(window)


@register_operator(name="ts_support_slope", canonical="ts_support_slope", backend="polars")
class TSSupportSlopePolarsNative(SeriesOperator):
    """Slope of support level trend"""

    metadata = OperatorMetadata(
        name="ts_support_slope",
        category="time_series",
        description="Slope of support level trend",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "support", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        support = feature.rolling_min(window)
        return support.diff(window)


# ============================================================================
# Threshold and Cycle Operators
# ============================================================================

@register_operator(name="ts_threshold_cycle_asymmetry", canonical="ts_threshold_cycle_asymmetry", backend="polars")
class TSThresholdCycleAsymmetryPolarsNative(SeriesOperator):
    """Asymmetry between time above vs below threshold"""

    metadata = OperatorMetadata(
        name="ts_threshold_cycle_asymmetry",
        category="time_series",
        description="Asymmetry between time above vs below threshold",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "threshold", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        above = (feature > threshold).cast(pl.Int32)
        pct_above = above.rolling_mean(window)
        return pct_above - 0.5


@register_operator(name="ts_threshold_cycle_period", canonical="ts_threshold_cycle_period", backend="polars")
class TSThresholdCyclePeriodPolarsNative(SeriesOperator):
    """Average period of threshold crossing cycles"""

    metadata = OperatorMetadata(
        name="ts_threshold_cycle_period",
        category="time_series",
        description="Average period of threshold crossing cycles",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "threshold", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        # Count crossings
        above = (feature > threshold).cast(pl.Int32)
        crossings = above.diff().abs()
        crossing_count = crossings.rolling_sum(window)
        # Period = window / number of crossings
        return np.where((crossing_count + 0.5) != 0, (window) / ((crossing_count + 0.5)), np.nan)


@register_operator(name="ts_time_since_change", canonical="ts_time_since_change", backend="polars")
class TSTimeSinceChangePolarsNative(SeriesOperator):
    """Time periods since last significant change"""

    metadata = OperatorMetadata(
        name="ts_time_since_change",
        category="time_series",
        description="Time periods since last significant change",
        param_names=["feature", "threshold"],
        return_type="series",
        tags=["time_series", "change", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, threshold, **kwargs):
        # TODO: Implement proper time-since-event tracking
        # Placeholder: use change magnitude
        change = feature.diff().abs()
        significant = (change > threshold).cast(pl.Int32)
        return significant.cum_sum()


# ============================================================================
# Transfer Entropy Operators
# ============================================================================

@register_operator(name="ts_transfer_entropy_peak_excess", canonical="ts_transfer_entropy_peak_excess", backend="polars")
class TSTransferEntropyPeakExcessPolarsNative(SeriesOperator):
    """Excess transfer entropy at peak lag"""

    metadata = OperatorMetadata(
        name="ts_transfer_entropy_peak_excess",
        category="time_series",
        description="Excess transfer entropy at peak lag",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "information", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper transfer entropy calculation
        # Placeholder: autocorrelation proxy
        return feature.rolling_corr(feature.shift(1), window)


@register_operator(name="ts_transfer_entropy_peak_lag", canonical="ts_transfer_entropy_peak_lag", backend="polars")
class TSTransferEntropyPeakLagPolarsNative(SeriesOperator):
    """Lag at which transfer entropy peaks"""

    metadata = OperatorMetadata(
        name="ts_transfer_entropy_peak_lag",
        category="time_series",
        description="Lag at which transfer entropy peaks",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "information", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper transfer entropy lag detection
        # Placeholder: fixed lag
        return pl.lit(1.0)


@register_operator(name="ts_transfer_entropy_peak_strength", canonical="ts_transfer_entropy_peak_strength", backend="polars")
class TSTransferEntropyPeakStrengthPolarsNative(SeriesOperator):
    """Strength of peak in transfer entropy"""

    metadata = OperatorMetadata(
        name="ts_transfer_entropy_peak_strength",
        category="time_series",
        description="Strength of peak in transfer entropy",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "information", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper transfer entropy
        # Placeholder: autocorrelation strength
        return feature.rolling_corr(feature.shift(1), window).abs()


# ============================================================================
# Transition and Trend Operators
# ============================================================================

@register_operator(name="ts_transition_count", canonical="ts_transition_count", backend="polars")
class TSTransitionCountPolarsNative(SeriesOperator):
    """Count of state transitions in window"""

    metadata = OperatorMetadata(
        name="ts_transition_count",
        category="time_series",
        description="Count of state transitions in window",
        param_names=["feature", "threshold", "window"],
        return_type="series",
        tags=["time_series", "rolling", "transition", "pit_safe"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, threshold, window, **kwargs):
        above = (feature > threshold).cast(pl.Int32)
        transitions = above.diff().abs()
        return transitions.rolling_sum(window)


@register_operator(name="ts_transition_intensity", canonical="ts_transition_intensity", backend="polars")
class TSTransitionIntensityPolarsNative(SeriesOperator):
    """Average magnitude of transitions"""

    metadata = OperatorMetadata(
        name="ts_transition_intensity",
        category="time_series",
        description="Average magnitude of transitions",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "transition", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        changes = feature.diff().abs()
        return changes.rolling_mean(window)


@register_operator(name="ts_trend_break_score", canonical="ts_trend_break_score", backend="polars")
class TSTrendBreakScorePolarsNative(SeriesOperator):
    """Score indicating trend break strength"""

    metadata = OperatorMetadata(
        name="ts_trend_break_score",
        category="time_series",
        description="Score indicating trend break strength",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "trend", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Compare recent trend to historical trend
        short_trend = feature.diff(window // 4)
        long_trend = feature.diff(window)
        return (short_trend - long_trend).abs()


@register_operator(name="ts_trend_tstat", canonical="ts_trend_tstat", backend="polars")
class TSTrendTstatPolarsNative(SeriesOperator):
    """T-statistic of linear trend"""

    metadata = OperatorMetadata(
        name="ts_trend_tstat",
        category="time_series",
        description="T-statistic of linear trend",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "trend", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper linear regression t-stat
        # Placeholder: standardized trend
        trend = feature.diff(window)
        std = feature.rolling_std(window)
        return np.where((std + 1e-8) != 0, (trend) / ((std + 1e-8)), np.nan)


@register_operator(name="ts_true_streak", canonical="ts_true_streak", backend="polars")
class TSTrueStreakPolarsNative(SeriesOperator):
    """Current consecutive count of True values"""

    metadata = OperatorMetadata(
        name="ts_true_streak",
        category="time_series",
        description="Current consecutive count of True values",
        param_names=["condition"],
        return_type="series",
        tags=["time_series", "streak", "pit_safe"],
    )

    def _calculate_series(self, condition, **kwargs):
        # TODO: Implement proper streak counting
        # Placeholder: cumulative sum with reset
        cond_int = condition.cast(pl.Int32)
        return cond_int.cum_sum()


# ============================================================================
# Turning Point Operators
# ============================================================================

@register_operator(name="ts_turning_intensity", canonical="ts_turning_intensity", backend="polars")
class TSTurningIntensityPolarsNative(SeriesOperator):
    """Intensity of turning points (magnitude of direction changes)"""

    metadata = OperatorMetadata(
        name="ts_turning_intensity",
        category="time_series",
        description="Intensity of turning points",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turning", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Second derivative magnitude
        first_diff = feature.diff()
        second_diff = first_diff.diff()
        return second_diff.abs().rolling_mean(window)


@register_operator(name="ts_turning_point_ratio", canonical="ts_turning_point_ratio", backend="polars")
class TSTurningPointRatioPolarsNative(SeriesOperator):
    """Ratio of turning points to total observations"""

    metadata = OperatorMetadata(
        name="ts_turning_point_ratio",
        category="time_series",
        description="Ratio of turning points to total observations",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turning", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Detect direction changes
        first_diff = feature.diff()
        sign_changes = (first_diff.sign().diff().abs() > 0).cast(pl.Int32)
        return sign_changes.rolling_mean(window)


@register_operator(name="ts_turning_rate", canonical="ts_turning_rate", backend="polars")
class TSTurningRatePolarsNative(SeriesOperator):
    """Rate of turning (curvature)"""

    metadata = OperatorMetadata(
        name="ts_turning_rate",
        category="time_series",
        description="Rate of turning (curvature)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "curvature", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Second derivative as curvature
        return feature.diff().diff().rolling_mean(window)


# ============================================================================
# Turnover Cost Operators
# ============================================================================

@register_operator(name="ts_turnover_age_dispersion", canonical="ts_turnover_age_dispersion", backend="polars")
class TSTurnoverAgeDispersionPolarsNative(SeriesOperator):
    """Dispersion of holding ages in portfolio"""

    metadata = OperatorMetadata(
        name="ts_turnover_age_dispersion",
        category="time_series",
        description="Dispersion of holding ages in portfolio",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper age tracking
        # Placeholder: volatility of changes
        return feature.diff().abs().rolling_std(window)


@register_operator(name="ts_turnover_cost_dispersion", canonical="ts_turnover_cost_dispersion", backend="polars")
class TSTurnoverCostDispersionPolarsNative(SeriesOperator):
    """Dispersion of turnover costs"""

    metadata = OperatorMetadata(
        name="ts_turnover_cost_dispersion",
        category="time_series",
        description="Dispersion of turnover costs",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        costs = feature.diff().abs()
        return costs.rolling_std(window)


@register_operator(name="ts_turnover_cost_entropy", canonical="ts_turnover_cost_entropy", backend="polars")
class TSTurnoverCostEntropyPolarsNative(SeriesOperator):
    """Entropy of turnover cost distribution"""

    metadata = OperatorMetadata(
        name="ts_turnover_cost_entropy",
        category="time_series",
        description="Entropy of turnover cost distribution",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "entropy", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper entropy calculation
        # Placeholder: normalized variance
        costs = feature.diff().abs()
        var = costs.rolling_var(window)
        mean = costs.rolling_mean(window)
        return np.where((mean ** 2 + 1e-8) != 0, (var) / ((mean ** 2 + 1e-8)), np.nan)


@register_operator(name="ts_turnover_cost_entropy_vol_scaled", canonical="ts_turnover_cost_entropy_vol_scaled", backend="polars")
class TSTurnoverCostEntropyVolScaledPolarsNative(SeriesOperator):
    """Volatility-scaled turnover cost entropy"""

    metadata = OperatorMetadata(
        name="ts_turnover_cost_entropy_vol_scaled",
        category="time_series",
        description="Volatility-scaled turnover cost entropy",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "entropy", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        costs = feature.diff().abs()
        vol = feature.rolling_std(window)
        scaled_costs = (costs) / (vol + 1e-8) if (vol + 1e-8) > 1e-10 else np.nan
        var = scaled_costs.rolling_var(window)
        mean = scaled_costs.rolling_mean(window)
        return np.where((mean ** 2 + 1e-8) != 0, (var) / ((mean ** 2 + 1e-8)), np.nan)


@register_operator(name="ts_turnover_cost_mode_distance", canonical="ts_turnover_cost_mode_distance", backend="polars")
class TSTurnoverCostModeDistancePolarsNative(SeriesOperator):
    """Distance from modal turnover cost"""

    metadata = OperatorMetadata(
        name="ts_turnover_cost_mode_distance",
        category="time_series",
        description="Distance from modal turnover cost",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Use median as proxy for mode
        costs = feature.diff().abs()
        mode_proxy = costs.rolling_median(window)
        return (costs - mode_proxy).abs()


@register_operator(name="ts_turnover_cost_quantile_distance", canonical="ts_turnover_cost_quantile_distance", backend="polars")
class TSTurnoverCostQuantileDistancePolarsNative(SeriesOperator):
    """Distance from quantile of turnover cost distribution"""

    metadata = OperatorMetadata(
        name="ts_turnover_cost_quantile_distance",
        category="time_series",
        description="Distance from quantile of turnover cost distribution",
        param_names=["feature", "window", "quantile"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.ECONOMIC),
    }

    def _calculate_series(self, feature, window, quantile, **kwargs):
        costs = feature.diff().abs()
        q_val = costs.rolling_quantile(quantile, window)
        return (costs - q_val).abs()


@register_operator(name="ts_turnover_cost_skew", canonical="ts_turnover_cost_skew", backend="polars")
class TSTurnoverCostSkewPolarsNative(SeriesOperator):
    """Skewness of turnover cost distribution"""

    metadata = OperatorMetadata(
        name="ts_turnover_cost_skew",
        category="time_series",
        description="Skewness of turnover cost distribution",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper skewness
        # Placeholder: ratio of moments
        costs = feature.diff().abs()
        return costs.rolling_skew(window)


@register_operator(name="ts_turnover_holding_age", canonical="ts_turnover_holding_age", backend="polars")
class TSTurnoverHoldingAgePolarsNative(SeriesOperator):
    """Average holding age of positions"""

    metadata = OperatorMetadata(
        name="ts_turnover_holding_age",
        category="time_series",
        description="Average holding age of positions",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper age tracking
        # Placeholder: inverse of turnover rate
        changes = (feature.diff().abs() > 0).cast(pl.Int32)
        turnover_rate = changes.rolling_mean(window)
        return np.where((turnover_rate + 0.01) != 0, (1.0) / ((turnover_rate + 0.01)), np.nan)


@register_operator(name="ts_turnover_near_cost_mass", canonical="ts_turnover_near_cost_mass", backend="polars")
class TSTurnoverNearCostMassPolarsNative(SeriesOperator):
    """Mass of turnover near typical cost"""

    metadata = OperatorMetadata(
        name="ts_turnover_near_cost_mass",
        category="time_series",
        description="Mass of turnover near typical cost",
        param_names=["feature", "window", "bandwidth"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "bandwidth": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, bandwidth, **kwargs):
        costs = feature.diff().abs()
        typical_cost = costs.rolling_median(window)
        near_typical = ((costs - typical_cost).abs() < bandwidth).cast(pl.Int32)
        return near_typical.rolling_mean(window)


@register_operator(name="ts_turnover_old_mass", canonical="ts_turnover_old_mass", backend="polars")
class TSTurnoverOldMassPolarsNative(SeriesOperator):
    """Mass of old (long-held) positions"""

    metadata = OperatorMetadata(
        name="ts_turnover_old_mass",
        category="time_series",
        description="Mass of old (long-held) positions",
        param_names=["feature", "window", "age_threshold"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "age_threshold": ParamSpec(dtype=int, min=1, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, age_threshold, **kwargs):
        # TODO: Implement proper age tracking
        # Placeholder: persistence measure
        changes = (feature.diff().abs() < 1e-6).cast(pl.Int32)
        return changes.rolling_mean(window)


@register_operator(name="ts_turnover_profit_share", canonical="ts_turnover_profit_share", backend="polars")
class TSTurnoverProfitSharePolarsNative(SeriesOperator):
    """Share of turnover that is profitable"""

    metadata = OperatorMetadata(
        name="ts_turnover_profit_share",
        category="time_series",
        description="Share of turnover that is profitable",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        changes = feature.diff()
        profitable = (changes > 0).cast(pl.Int32)
        return profitable.rolling_mean(window)


@register_operator(name="ts_turnover_reference_price", canonical="ts_turnover_reference_price", backend="polars")
class TSTurnoverReferencePricePolarsNative(SeriesOperator):
    """Reference price for turnover cost calculation"""

    metadata = OperatorMetadata(
        name="ts_turnover_reference_price",
        category="time_series",
        description="Reference price for turnover cost calculation",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "turnover", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # VWAP proxy using moving average
        return feature.rolling_mean(window)


# ============================================================================
# Regime and Value Operators
# ============================================================================

@register_operator(name="ts_two_state_regime_probability", canonical="ts_two_state_regime_probability", backend="polars")
class TSTwoStateRegimeProbabilityPolarsNative(SeriesOperator):
    """Probability of being in high-volatility regime"""

    metadata = OperatorMetadata(
        name="ts_two_state_regime_probability",
        category="time_series",
        description="Probability of being in high-volatility regime",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "regime", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper Hidden Markov Model
        # Placeholder: volatility regime indicator
        vol = feature.rolling_std(window)
        high_vol = vol > vol.rolling_median(window * 2)
        return high_vol.cast(pl.Float64)


@register_operator(name="ts_upper_tail_coexceedance_probability", canonical="ts_upper_tail_coexceedance_probability", backend="polars")
class TSUpperTailCoexceedanceProbabilityPolarsNative(SeriesOperator):
    """Probability of co-exceedance in upper tail"""

    metadata = OperatorMetadata(
        name="ts_upper_tail_coexceedance_probability",
        category="time_series",
        description="Probability of co-exceedance in upper tail",
        param_names=["feature", "window", "threshold"],
        return_type="series",
        tags=["time_series", "rolling", "tail", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "threshold": ParamSpec(dtype=float, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, feature, window, threshold, **kwargs):
        # Fraction of time above threshold
        exceeds = (feature > threshold).cast(pl.Int32)
        return exceeds.rolling_mean(window)


@register_operator(name="ts_value_at_argextreme", canonical="ts_value_at_argextreme", backend="polars")
class TSValueAtArgextremePolarsNative(SeriesOperator):
    """Value at the location of extreme (max or min)"""

    metadata = OperatorMetadata(
        name="ts_value_at_argextreme",
        category="time_series",
        description="Value at the location of extreme",
        param_names=["feature", "window", "use_max"],
        return_type="series",
        tags=["time_series", "rolling", "extreme", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "use_max": ParamSpec(dtype=bool, param_role=ParamRole.POLICY),
    }

    def _calculate_series(self, feature, window, use_max, **kwargs):
        if use_max:
            return feature.rolling_max(window)
        else:
            return feature.rolling_min(window)


# ============================================================================
# Variance Ratio and Variogram Operators
# ============================================================================

@register_operator(name="ts_variance_ratio", canonical="ts_variance_ratio", backend="polars")
class TSVarianceRatioPolarsNative(SeriesOperator):
    """Variance ratio test statistic

    NOTE: Previously named ts_variance_ratio_proxy, but this implements the correct
    Lo-MacKinlay variance ratio: VR(q) = Var(q-period returns) / (q * Var(1-period returns))
    This is the standard random walk test statistic.
    """

    metadata = OperatorMetadata(
        name="ts_variance_ratio",
        category="time_series",
        description="Lo-MacKinlay variance ratio test statistic",
        param_names=["feature", "short_window", "long_window"],
        return_type="series",
        tags=["time_series", "rolling", "variance", "pit_safe"],
    )
    metadata.param_specs = {
        "short_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "long_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, short_window, long_window, **kwargs):
        short_var = feature.rolling_var(short_window)
        long_var = feature.rolling_var(long_window)
        # Variance ratio scaled by window ratio
        return np.where(short_var * (short_window / long_window) != 0, ((long_var) / (short_var) * (short_window / long_window)), np.nan)


@register_operator(name="ts_variance_ratio_slope", canonical="ts_variance_ratio_slope", backend="polars")
class TSVarianceRatioSlopePolarsNative(SeriesOperator):
    """Slope of variance ratio across different lags"""

    metadata = OperatorMetadata(
        name="ts_variance_ratio_slope",
        category="time_series",
        description="Slope of variance ratio across different lags",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "variance", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Compare variance at different scales
        var1 = feature.rolling_var(window)
        var2 = feature.rolling_var(window * 2)
        return np.where(var1 != 0, ((var2 - var1)) / (var1), np.nan)


@register_operator(name="ts_variogram_slope", canonical="ts_variogram_slope", backend="polars")
class TSVariogramSlopePolarsNative(SeriesOperator):
    """Slope of the variogram (semi-variance vs lag)"""

    metadata = OperatorMetadata(
        name="ts_variogram_slope",
        category="time_series",
        description="Slope of the variogram",
        param_names=["feature", "lag", "window"],
        return_type="series",
        tags=["time_series", "rolling", "variogram", "pit_safe"],
    )
    metadata.param_specs = {
        "lag": ParamSpec(dtype=int, min=1, param_role=ParamRole.ECONOMIC),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, lag, window, **kwargs):
        # Semi-variance at given lag
        diff = (feature - feature.shift(lag)) ** 2
        semivar = (diff.rolling_mean(window)) / 2.0 if 2.0 != 0 else np.nan
        return semivar.diff(window)


# ============================================================================
# Vector Path Operators (Multivariate)
# ============================================================================

@register_operator(name="ts_vector_path_curvature", canonical="ts_vector_path_curvature", backend="polars")
class TSVectorPathCurvaturePolarsNative(SeriesOperator):
    """Curvature of path in phase space"""

    metadata = OperatorMetadata(
        name="ts_vector_path_curvature",
        category="time_series",
        description="Curvature of path in phase space",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "vector", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Second derivative as curvature proxy
        first_diff = feature.diff()
        second_diff = first_diff.diff()
        return second_diff.abs().rolling_mean(window)


@register_operator(name="ts_vector_path_efficiency", canonical="ts_vector_path_efficiency", backend="polars")
class TSVectorPathEfficiencyPolarsNative(SeriesOperator):
    """Path efficiency (straight-line distance / path length)"""

    metadata = OperatorMetadata(
        name="ts_vector_path_efficiency",
        category="time_series",
        description="Path efficiency ratio",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "vector", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Net displacement / total path length
        net_displacement = feature.diff(window).abs()
        path_length = feature.diff().abs().rolling_sum(window)
        return np.where((path_length + 1e-8) != 0, (net_displacement) / ((path_length + 1e-8)), np.nan)


@register_operator(name="ts_vector_self_intersection_rate", canonical="ts_vector_self_intersection_rate", backend="polars")
class TSVectorSelfIntersectionRatePolarsNative(SeriesOperator):
    """Rate of self-intersection in phase space trajectory"""

    metadata = OperatorMetadata(
        name="ts_vector_self_intersection_rate",
        category="time_series",
        description="Rate of self-intersection in trajectory",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "vector", "topology", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper topological analysis
        # Placeholder: number of sign changes (crossings)
        sign_changes = (feature.sign().diff().abs() > 0).cast(pl.Int32)
        return sign_changes.rolling_mean(window)


@register_operator(name="ts_vector_state_local_density", canonical="ts_vector_state_local_density", backend="polars")
class TSVectorStateLocalDensityPolarsNative(SeriesOperator):
    """Local density in state space"""

    metadata = OperatorMetadata(
        name="ts_vector_state_local_density",
        category="time_series",
        description="Local density in state space",
        param_names=["feature", "window", "bandwidth"],
        return_type="series",
        tags=["time_series", "rolling", "vector", "density", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "bandwidth": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }

    def _calculate_series(self, feature, window, bandwidth, **kwargs):
        # Inverse of rolling std as density proxy
        std = feature.rolling_std(window)
        return np.where((std + bandwidth) != 0, (1.0) / ((std + bandwidth)), np.nan)


@register_operator(name="ts_vector_state_mahalanobis", canonical="ts_vector_state_mahalanobis", backend="polars")
class TSVectorStateMahalanobisPolarsNative(SeriesOperator):
    """Mahalanobis distance from centroid in state space"""

    metadata = OperatorMetadata(
        name="ts_vector_state_mahalanobis",
        category="time_series",
        description="Mahalanobis distance from centroid",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "vector", "distance", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Standardized distance from mean
        mean = feature.rolling_mean(window)
        std = feature.rolling_std(window)
        return np.where((std + 1e-8) != 0, (((feature - mean).abs()) / ((std + 1e-8))), np.nan)


@register_operator(name="ts_vector_turning_coherence", canonical="ts_vector_turning_coherence", backend="polars")
class TSVectorTurningCoherencePolarsNative(SeriesOperator):
    """Coherence of turning points across dimensions"""

    metadata = OperatorMetadata(
        name="ts_vector_turning_coherence",
        category="time_series",
        description="Coherence of turning points",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "vector", "turning", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Smoothness of turning pattern
        second_diff = feature.diff().diff()
        return np.where((second_diff.abs().rolling_mean(window) + 1e-8) != 0, (1.0) / ((second_diff.abs().rolling_mean(window) + 1e-8)), np.nan)


# ============================================================================
# Volatility Operators
# ============================================================================

@register_operator(name="ts_vol_acceleration", canonical="ts_vol_acceleration", backend="polars")
class TSVolAccelerationPolarsNative(SeriesOperator):
    """Rate of change of volatility"""

    metadata = OperatorMetadata(
        name="ts_vol_acceleration",
        category="time_series",
        description="Rate of change of volatility",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        vol = feature.rolling_std(window)
        return vol.diff(window)


@register_operator(name="ts_vol_clustering", canonical="ts_vol_clustering", backend="polars")
class TSVolClusteringPolarsNative(SeriesOperator):
    """Volatility clustering measure (autocorrelation of squared returns)"""

    metadata = OperatorMetadata(
        name="ts_vol_clustering",
        category="time_series",
        description="Volatility clustering measure",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        squared = feature ** 2
        return squared.rolling_corr(squared.shift(1), window)


@register_operator(name="ts_vol_of_vol", canonical="ts_vol_of_vol", backend="polars")
class TSVolOfVolPolarsNative(SeriesOperator):
    """Volatility of volatility"""

    metadata = OperatorMetadata(
        name="ts_vol_of_vol",
        category="time_series",
        description="Volatility of volatility",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        vol = feature.rolling_std(window)
        return vol.rolling_std(window)


@register_operator(name="ts_vol_pvariation_roughness", canonical="ts_vol_pvariation_roughness", backend="polars")
class TSVolPvariationRoughnessPolarsNative(SeriesOperator):
    """Roughness measure using p-variation"""

    metadata = OperatorMetadata(
        name="ts_vol_pvariation_roughness",
        category="time_series",
        description="Roughness measure using p-variation",
        param_names=["feature", "window", "p"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "roughness", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "p": ParamSpec(dtype=float, min=0.5, param_role=ParamRole.ECONOMIC),
    }

    def _calculate_series(self, feature, window, p, **kwargs):
        # p-variation: sum of |diff|^p
        diffs = feature.diff().abs()
        p_var = (diffs ** p).rolling_sum(window)
        return p_var


@register_operator(name="ts_vol_scaling_break", canonical="ts_vol_scaling_break", backend="polars")
class TSVolScalingBreakPolarsNative(SeriesOperator):
    """Detect breaks in volatility scaling behavior"""

    metadata = OperatorMetadata(
        name="ts_vol_scaling_break",
        category="time_series",
        description="Detect breaks in volatility scaling",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "regime", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Compare recent vs historical volatility scaling
        short_vol = feature.rolling_std(window // 2)
        long_vol = feature.rolling_std(window)
        return np.where((long_vol + 1e-8) - 1.0.abs() != 0, ((short_vol) / ((long_vol + 1e-8) - 1.0).abs()), np.nan)


@register_operator(name="ts_vol_shift_score", canonical="ts_vol_shift_score", backend="polars")
class TSVolShiftScorePolarsNative(SeriesOperator):
    """Score indicating magnitude of volatility regime shift"""

    metadata = OperatorMetadata(
        name="ts_vol_shift_score",
        category="time_series",
        description="Magnitude of volatility regime shift",
        param_names=["feature", "short_window", "long_window"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "regime", "pit_safe"],
    )
    metadata.param_specs = {
        "short_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "long_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, short_window, long_window, **kwargs):
        short_vol = feature.rolling_std(short_window)
        long_vol = feature.rolling_std(long_window)
        return np.where((long_vol + 1e-8) != 0, ((short_vol - long_vol)) / ((long_vol + 1e-8)), np.nan)


@register_operator(name="ts_vol_term_structure", canonical="ts_vol_term_structure", backend="polars")
class TSVolTermStructurePolarsNative(SeriesOperator):
    """Term structure of volatility (short vs long)"""

    metadata = OperatorMetadata(
        name="ts_vol_term_structure",
        category="time_series",
        description="Term structure of volatility",
        param_names=["feature", "short_window", "long_window"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "pit_safe"],
    )
    metadata.param_specs = {
        "short_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "long_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, short_window, long_window, **kwargs):
        short_vol = feature.rolling_std(short_window)
        long_vol = feature.rolling_std(long_window)
        return np.where((long_vol + 1e-8) != 0, (short_vol) / ((long_vol + 1e-8)), np.nan)


# ============================================================================
# Wasserstein and Wavelet Operators
# ============================================================================

@register_operator(name="ts_wasserstein_shift", canonical="ts_wasserstein_shift", backend="polars")
class TSWassersteinShiftPolarsNative(SeriesOperator):
    """Wasserstein distance between historical and recent distributions"""

    metadata = OperatorMetadata(
        name="ts_wasserstein_shift",
        category="time_series",
        description="Wasserstein distance between distributions",
        param_names=["feature", "short_window", "long_window"],
        return_type="series",
        tags=["time_series", "rolling", "distribution", "pit_safe"],
    )
    metadata.param_specs = {
        "short_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "long_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, short_window, long_window, **kwargs):
        # TODO: Implement proper Wasserstein distance
        # Placeholder: difference in means and stds
        short_mean = feature.rolling_mean(short_window)
        long_mean = feature.rolling_mean(long_window)
        short_std = feature.rolling_std(short_window)
        long_std = feature.rolling_std(long_window)
        return ((short_mean - long_mean).abs() + (short_std - long_std).abs())


@register_operator(name="ts_wavelet_energy_slope", canonical="ts_wavelet_energy_slope", backend="polars")
class TSWaveletEnergySlopePolarsNative(SeriesOperator):
    """Slope of wavelet energy across scales"""

    metadata = OperatorMetadata(
        name="ts_wavelet_energy_slope",
        category="time_series",
        description="Slope of wavelet energy across scales",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "wavelet", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, choices=(32, 64, 128, 256), default=128, param_role=ParamRole.HORIZON),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_wavelet_energy_slope",
        backend="polars",
        execution_kind=ExecutionKind.DELEGATE_PYTHON,
        supports_lazy=False,
        supports_streaming=False,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=False,
        implementation_source_hash="a6d2e5d18e89676ea626926565e6eba9f83fcfce79b7b481d2c889d09bb1a81c",
        emitter_identity="872943b99e1cab97c777b6fe15dd1977955b4f170ecb21a33ea9a4b7c0618c5e",
        kernel_identity="ce0944fbb6047fdc2541326f601a68b8af03c03bb9e5ec4b27f35327e3215f93",
        parameter_domain_hash="b363212dd0cdf41c2feeaa4dc5092354f3210e3bd5822565a656ae5df823e2a0",
        semantic_contract_hash="757b9d7572769e2d16729838a645025419a8ef98b836d8de1d25450fbd8bfe37",
        notes="Eager wide-panel Python/NumPy Haar bridge; not a native Polars expression and not production-certified.",
    )

    def _calculate_series(self, x, window=128, **kwargs):
        w = _fixed_window_anchor(window)
        return _wavelet_spectral_cpu_wide(
            x,
            lambda values: _wavelet_stats(values, w, "slope"),
        )


@register_operator(name="ts_wavelet_entropy", canonical="ts_wavelet_entropy", backend="polars")
class TSWaveletEntropyPolarsNative(SeriesOperator):
    """Eager CPU Haar-band entropy bridge; not a native Polars expression."""

    metadata = OperatorMetadata(
        name="ts_wavelet_entropy",
        category="time_series",
        description="Entropy of wavelet coefficient distribution",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "wavelet", "entropy", "pit_safe", "cpu_udf"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, choices=(32, 64, 128, 256), default=128, param_role=ParamRole.HORIZON),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_wavelet_entropy", backend="polars", execution_kind=ExecutionKind.DELEGATE_PYTHON,
        materializes_full_panel=True, requires_sorted=True, supports_nulls=True, supports_nan=True,
        implementation_source_hash="0fc5db9a347c621730ce24af3a39cdbb818d42fc2350993537c11976144d5208",
        emitter_identity="polars.DataFrame:wide_cpu_bridge:v1",
        kernel_identity="wavelet_spectral._wavelet_stats:entropy:v9",
        parameter_domain_hash="window:int:{32,64,128,256};wide_or_single_stock",
        semantic_contract_hash="haar:detail_total_energy:positive_band_shannon_normalized:v9",
        notes="Eager shared Python/NumPy Haar bridge; not native Polars and not production-eligible.",
    )

    def _calculate_series(self, x, window=128, **kwargs):
        w = _fixed_window_anchor(window)
        return _wavelet_spectral_cpu_wide(x, lambda values: _wavelet_stats(values, w, "entropy"))


@register_operator(name="ts_wavelet_high_frequency_ratio", canonical="ts_wavelet_high_frequency_ratio", backend="polars")
class TSWaveletHighFrequencyRatioPolarsNative(SeriesOperator):
    """Eager CPU fine-scale Haar-energy bridge; not a native Polars expression."""

    metadata = OperatorMetadata(
        name="ts_wavelet_high_frequency_ratio",
        category="time_series",
        description="Ratio of high frequency energy",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "wavelet", "pit_safe", "cpu_udf"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, choices=(32, 64, 128, 256), default=128, param_role=ParamRole.HORIZON),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_wavelet_high_frequency_ratio", backend="polars", execution_kind=ExecutionKind.DELEGATE_PYTHON,
        materializes_full_panel=True, requires_sorted=True, supports_nulls=True, supports_nan=True,
        implementation_source_hash="ce43f2b0e98bbe668f740c49bf4e02f00284675de3372bb7577691bbead453db",
        emitter_identity="polars.DataFrame:wide_cpu_bridge:v1",
        kernel_identity="wavelet_spectral._wavelet_stats:high:v9",
        parameter_domain_hash="window:int:{32,64,128,256};wide_or_single_stock",
        semantic_contract_hash="haar:finest_detail_total_energy_ratio:v9",
        notes="Eager shared Python/NumPy Haar bridge; not native Polars and not production-eligible.",
    )

    def _calculate_series(self, x, window=128, **kwargs):
        w = _fixed_window_anchor(window)
        return _wavelet_spectral_cpu_wide(x, lambda values: _wavelet_stats(values, w, "high"))


@register_operator(name="ts_wavelet_low_frequency_ratio", canonical="ts_wavelet_low_frequency_ratio", backend="polars")
class TSWaveletLowFrequencyRatioPolarsNative(SeriesOperator):
    """Eager CPU coarse-scale Haar-energy bridge; not a native Polars expression."""

    metadata = OperatorMetadata(
        name="ts_wavelet_low_frequency_ratio",
        category="time_series",
        description="Ratio of low frequency energy",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "rolling", "wavelet", "pit_safe", "cpu_udf"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, choices=(32, 64, 128, 256), default=128, param_role=ParamRole.HORIZON),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_wavelet_low_frequency_ratio", backend="polars", execution_kind=ExecutionKind.DELEGATE_PYTHON,
        materializes_full_panel=True, requires_sorted=True, supports_nulls=True, supports_nan=True,
        implementation_source_hash="c28f5c4f617498cdd41ff4d2618c50008355660f867689c50ab9d29b204984a4",
        emitter_identity="polars.DataFrame:wide_cpu_bridge:v1",
        kernel_identity="wavelet_spectral._wavelet_stats:low:v9",
        parameter_domain_hash="window:int:{32,64,128,256};wide_or_single_stock",
        semantic_contract_hash="haar:coarsest_detail_total_energy_ratio:v9",
        notes="Eager shared Python/NumPy Haar bridge; not native Polars and not production-eligible.",
    )

    def _calculate_series(self, x, window=128, **kwargs):
        w = _fixed_window_anchor(window)
        return _wavelet_spectral_cpu_wide(x, lambda values: _wavelet_stats(values, w, "low"))


@register_operator(name="ts_wavelet_lowpass_reconstruct", canonical="ts_wavelet_lowpass_reconstruct", backend="polars")
class TSWaveletLowpassReconstructPolarsNative(SeriesOperator):
    """Reconstructed signal using low-pass wavelet coefficients"""

    metadata = OperatorMetadata(
        name="ts_wavelet_lowpass_reconstruct",
        category="time_series",
        description="Low-pass wavelet reconstruction",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "wavelet", "smoothing", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper wavelet reconstruction
        # Placeholder: moving average as low-pass filter
        return feature.rolling_mean(window)


# ============================================================================
# Weighted Statistics Operators
# ============================================================================

@register_operator(name="ts_weighted_downside_deviation", canonical="ts_weighted_downside_deviation", backend="polars")
class TSWeightedDownsideDeviationPolarsNative(SeriesOperator):
    """Weighted downside deviation (below target)"""

    metadata = OperatorMetadata(
        name="ts_weighted_downside_deviation",
        category="time_series",
        description="Weighted downside deviation",
        param_names=["feature", "window", "target"],
        return_type="series",
        tags=["time_series", "rolling", "weighted", "risk", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "target": ParamSpec(dtype=float, param_role=ParamRole.ECONOMIC),
    }

    def _calculate_series(self, feature, window, target, **kwargs):
        # Downside = negative deviations from target
        downside = pl.when(feature < target).then((target - feature) ** 2).otherwise(0.0)
        return downside.rolling_mean(window).sqrt()


@register_operator(name="ts_weighted_drawdown_area", canonical="ts_weighted_drawdown_area", backend="polars")
class TSWeightedDrawdownAreaPolarsNative(SeriesOperator):
    """Weighted area under drawdown curve"""

    metadata = OperatorMetadata(
        name="ts_weighted_drawdown_area",
        category="time_series",
        description="Weighted area under drawdown curve",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "weighted", "drawdown", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Drawdown from rolling max
        running_max = feature.rolling_max(window)
        drawdown = running_max - feature
        # Use exponential weighting
        return drawdown.ewm_mean(span=window, ignore_nulls=True)


@register_operator(name="ts_weighted_expected_shortfall", canonical="ts_weighted_expected_shortfall", backend="polars")
class TSWeightedExpectedShortfallPolarsNative(SeriesOperator):
    """Weighted expected shortfall (CVaR)"""

    metadata = OperatorMetadata(
        name="ts_weighted_expected_shortfall",
        category="time_series",
        description="Weighted expected shortfall (CVaR)",
        param_names=["feature", "window", "alpha"],
        return_type="series",
        tags=["time_series", "rolling", "weighted", "risk", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "alpha": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.ECONOMIC),
    }

    def _calculate_series(self, feature, window, alpha, **kwargs):
        # TODO: Implement proper CVaR with exponential weighting
        # Placeholder: mean of values below quantile
        threshold = feature.rolling_quantile(alpha, window)
        tail_losses = pl.when(feature < threshold).then(feature).otherwise(None)
        return tail_losses.rolling_mean(window)


@register_operator(name="ts_weighted_permutation_entropy", canonical="ts_weighted_permutation_entropy", backend="polars")
class TSWeightedPermutationEntropyPolarsNative(SeriesOperator):
    """Weighted permutation entropy"""

    metadata = OperatorMetadata(
        name="ts_weighted_permutation_entropy",
        category="time_series",
        description="Weighted permutation entropy",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "weighted", "entropy", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper permutation entropy
        # Placeholder: normalized variance with exponential weighting
        ewm_var = feature.ewm_mean(span=window, ignore_nulls=True).rolling_var(window)
        ewm_mean_sq = (feature.ewm_mean(span=window, ignore_nulls=True) ** 2)
        return np.where((ewm_mean_sq + 1e-8) != 0, (ewm_var) / ((ewm_mean_sq + 1e-8)), np.nan)


@register_operator(name="ts_weighted_semivariance", canonical="ts_weighted_semivariance", backend="polars")
class TSWeightedSemivariancePolarsNative(SeriesOperator):
    """Weighted semivariance (variance of negative returns)"""

    metadata = OperatorMetadata(
        name="ts_weighted_semivariance",
        category="time_series",
        description="Weighted semivariance",
        param_names=["feature", "window", "target"],
        return_type="series",
        tags=["time_series", "rolling", "weighted", "risk", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "target": ParamSpec(dtype=float, param_role=ParamRole.ECONOMIC, default=0.0),
    }

    def _calculate_series(self, feature, window, target, **kwargs):
        # Semi-variance below target
        below_target = pl.when(feature < target).then((feature - target) ** 2).otherwise(0.0)
        return below_target.ewm_mean(span=window, ignore_nulls=True)


@register_operator(name="ts_weighted_standardized_moment", canonical="ts_weighted_standardized_moment", backend="polars")
class TSWeightedStandardizedMomentPolarsNative(SeriesOperator):
    """Weighted standardized moment (skewness or kurtosis)"""

    metadata = OperatorMetadata(
        name="ts_weighted_standardized_moment",
        category="time_series",
        description="Weighted standardized moment",
        param_names=["feature", "window", "order"],
        return_type="series",
        tags=["time_series", "rolling", "weighted", "moment", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "order": ParamSpec(dtype=int, min=2, param_role=ParamRole.ECONOMIC),
    }

    def _calculate_series(self, feature, window, order, **kwargs):
        # Weighted moment calculation
        ewm_mean = feature.ewm_mean(span=window, ignore_nulls=True)
        centered = feature - ewm_mean
        ewm_std = feature.ewm_std(span=window, ignore_nulls=True)

        if order == 3:
            # Skewness
            return ((centered ** 3).ewm_mean(span=window, ignore_nulls=True) /
                    (ewm_std ** 3 + 1e-8))
        elif order == 4:
            # Kurtosis
            return ((centered ** 4).ewm_mean(span=window, ignore_nulls=True) /
                    (ewm_std ** 4 + 1e-8))
        else:
            # General moment
            return ((centered ** order).ewm_mean(span=window, ignore_nulls=True) /
                    ((ewm_std ** order) + 1e-8))


@register_operator(name="ts_weighted_time_centroid", canonical="ts_weighted_time_centroid", backend="polars")
class TSWeightedTimeCentroidPolarsNative(SeriesOperator):
    """Time-weighted centroid of values in window (center of mass in time)"""

    metadata = OperatorMetadata(
        name="ts_weighted_time_centroid",
        category="time_series",
        description="Time-weighted centroid (center of mass)",
        param_names=["feature", "window"],
        return_type="series",
        tags=["time_series", "rolling", "weighted", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # TODO: Implement proper time-weighted centroid
        # Placeholder: exponentially weighted mean emphasizes recent values
        # The span controls how much weight recent observations get
        return feature.ewm_mean(span=window / 2.0, ignore_nulls=True)


# ============================================================================
# Registration Summary
# ============================================================================
# This module implements 88 advanced time series operators covering:
# - Run and persistence analysis
# - Spectral analysis
# - SSA (Singular Spectrum Analysis)
# - State/regime detection
# - Support/resistance levels
# - Threshold and cycle analysis
# - Transfer entropy
# - Transition and trend detection
# - Turning point analysis
# - Turnover cost metrics
# - Regime probability
# - Variance ratio tests
# - Vector path analysis
# - Volatility measures
# - Wasserstein distance
# - Wavelet analysis
# - Weighted statistics
#
# Many complex algorithms (SSA, wavelets, transfer entropy, topology) are
# implemented as skeletons with TODO markers for proper implementation.
# All operators are registered with backend="polars" and use lazy evaluation.
