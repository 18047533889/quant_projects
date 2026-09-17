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

import copy
import polars as pl
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
import numpy as np
from typing import Optional, Union

from factor_engine.cleaned_operators.base import (
    SeriesOperator,
    register_operator as _register_operator,
    OperatorMetadata,
    ParamSpec,
    ParamRole,
)
from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate


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
    """Exact reference dependence; explicit Pandas delegation, not a proxy."""
    from factor_engine.cleaned_operators.common.dependence_delegate import metadata as _metadata, physical_spec as _spec
    metadata = _metadata("ts_distance_correlation_partial_proxy")
    _physical_spec = _spec("ts_distance_correlation_partial_proxy")
    @property
    def _contract_callable(self):
        from factor_engine.cleaned_operators.common.dependence_delegate import reference
        return reference("ts_distance_correlation_partial_proxy")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        from factor_engine.cleaned_operators.common.dependence_delegate import calculate
        return calculate("ts_distance_correlation_partial_proxy",*args,**kwargs)


@register_operator(name="ts_feature_mode_share", canonical="ts_feature_mode_share", backend="polars", status="research_only")
class TSFeatureModeSharePolarsNative(SeriesOperator):
    """Exact Polars-panel delegate to shared robust feature geometry."""
    from factor_engine.cleaned_operators.feature_geometry import TsFeatureModeShare as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    metadata.tags = list(metadata.tags or []) + ["polars", "delegate:pandas_numpy"]

    def _calculate_series(self, f1, f2, f3, window=60, **kwargs):
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate(
            "ts_feature_mode_share", (f1, f2, f3), {"window": window, **kwargs},
        )


@register_operator(name="ts_feature_subspace_rotation", canonical="ts_feature_subspace_rotation", backend="polars")
class TSFeatureSubspaceRotationPolarsNative(SeriesOperator):
    """Exact Polars-panel delegate to shared robust feature geometry."""
    from factor_engine.cleaned_operators.feature_geometry import TsFeatureSubspaceRotation as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    metadata.tags = list(metadata.tags or []) + ["polars", "delegate:pandas_numpy"]

    def _calculate_series(self, f1, f2, f3, recent_window=30,
                          prior_window=90, eigen_gap=0.02, **kwargs):
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate(
            "ts_feature_subspace_rotation", (f1, f2, f3), {
                "recent_window": recent_window, "prior_window": prior_window,
                "eigen_gap": eigen_gap, **kwargs,
            },
        )


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
        param_names=["x", "ntaps", "cutoff", "window"],
        return_type="series",
        tags=["time_series", "rolling", "filter", "causal", "pit_safe"],
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, x, ntaps: int = 10, cutoff: float = 0.5,
                          window: int = 20, **kwargs):
        feature = x
        # Hamming window weights for FIR lowpass
        # Placeholder: simple moving average
        return feature.rolling_mean(window)


@register_operator(name="ts_first_passage_bias", canonical="ts_first_passage_bias", backend="polars")
class TSFirstPassageBiasPolarsNative(SeriesOperator):
    """Exact first-passage reference; never a threshold-crossing surrogate."""
    from factor_engine.cleaned_operators.first_passage import TsFirstPassageBias as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_first_passage_bias", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False, supports_streaming=False, materializes_full_panel=True,
        supports_nulls=True, supports_nan=True, supports_inf=True,
    )

    @property
    def _contract_callable(self):
        return self._Reference()._calculate_series

    def _calculate_series(self, x, scale, window=120, barrier=1.0, horizon=10,
                          min_anchors=3, scale_horizon=1, **kwargs):
        return _call_pandas_delegate("ts_first_passage_bias", (x, scale), {
            "window": window, "barrier": barrier, "horizon": horizon,
            "min_anchors": min_anchors, "scale_horizon": scale_horizon, **kwargs,
        })


@register_operator(name="ts_first_passage_conditional_time", canonical="ts_first_passage_conditional_time", category="first_passage", business_category="first_passage", backend="polars", status="research_only")
class TSFirstPassageConditionalTimePolarsNative(SeriesOperator):
    """Exact delegate to the authoritative conditional first-passage time."""
    from factor_engine.cleaned_operators.first_passage import TsFirstPassageConditionalTime as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_first_passage_conditional_time", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False, supports_streaming=False, materializes_full_panel=True,
        supports_nulls=True, supports_nan=True, supports_inf=True,
    )

    def _calculate_series(self, x, scale, window=120, barrier=1.0, horizon=10,
                          min_anchors=3, scale_horizon=1, side="upper", **kwargs):
        return _call_pandas_delegate("ts_first_passage_conditional_time", (x, scale), {
            "window": window, "barrier": barrier, "horizon": horizon,
            "min_anchors": min_anchors, "scale_horizon": scale_horizon,
            "side": side, **kwargs,
        })


@register_operator(name="ts_first_passage_hit_probability", canonical="ts_first_passage_hit_probability", category="first_passage", business_category="first_passage", backend="polars")
class TSFirstPassageHitProbabilityPolarsNative(SeriesOperator):
    """Exact delegate to the authoritative first-passage hit probability."""
    from factor_engine.cleaned_operators.first_passage import TsFirstPassageHitProbability as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_first_passage_hit_probability", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False, supports_streaming=False, materializes_full_panel=True,
        supports_nulls=True, supports_nan=True, supports_inf=True,
    )

    def _calculate_series(self, x, scale, window=120, barrier=1.0, horizon=10,
                          min_anchors=3, scale_horizon=1, side="upper", **kwargs):
        return _call_pandas_delegate("ts_first_passage_hit_probability", (x, scale), {
            "window": window, "barrier": barrier, "horizon": horizon,
            "min_anchors": min_anchors, "scale_horizon": scale_horizon,
            "side": side, **kwargs,
        })


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
        param_names=["x", "recent_window", "prior_window"],
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
    """Explicit canonical memory reference delegate; not native acceleration."""
    from factor_engine.cleaned_operators.common import memory_delegate as _delegate
    metadata = _delegate.metadata("ts_fractional_difference")
    @property
    def _contract_callable(self):
        return self._delegate.reference(self.metadata.name)._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._delegate.calculate(self.metadata.name,*args,**kwargs)
    def physical_spec(self):
        return self._delegate.physical_spec(self.metadata.name)


@register_operator(name="ts_fractional_difference_discarded_weight_mass", canonical="ts_fractional_difference_discarded_weight_mass", backend="polars", status="research_only")
class TSFractionalDifferenceDiscardedWeightMassPolarsNative(SeriesOperator):
    """Explicit canonical memory reference delegate; not native acceleration."""
    from factor_engine.cleaned_operators.common import memory_delegate as _delegate
    metadata = _delegate.metadata("ts_fractional_difference_discarded_weight_mass")
    @property
    def _contract_callable(self):
        return self._delegate.reference(self.metadata.name)._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._delegate.calculate(self.metadata.name,*args,**kwargs)
    def physical_spec(self):
        return self._delegate.physical_spec(self.metadata.name)


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
        param_names=["x","window","side","tail_fraction","min_tail_count"],
        return_type="series",
        tags=["time_series", "rolling", "extreme_tail", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "side": ParamSpec(dtype=str, default="both", param_role=ParamRole.POLICY),
        "tail_fraction": ParamSpec(dtype=float, default=0.1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }
    def _calculate_series(self, x, window=120, side="both", tail_fraction=0.1,
                          min_tail_count=10, **kwargs):
        # R4-100 parity: canonical (x, window, side, tail_fraction,
        # min_tail_count); ``feature/threshold_quantile`` are legacy aliases.
        feature = x
        threshold_quantile = 1.0 - float(tail_fraction)
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
    """Exact causal despike CPU kernel, preserving canonical defaults."""
    from factor_engine.cleaned_operators.common import despike_native as _despike
    metadata=_despike.metadata("ts_hampel_filter_causal")
    @property
    def _contract_callable(self):
        return self._despike.reference("ts_hampel_filter_causal")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._despike.calculate("ts_hampel_filter_causal",*args,**kwargs)
    def physical_spec(self):
        return self._despike.physical_spec("ts_hampel_filter_causal")


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

@register_operator(name="ts_hartigan_dip", canonical="ts_hartigan_dip", category="moments", business_category="moments", backend="polars", status="research_only", source="factor_dsl_polars_native", replace=True, expected_old_source="pandas_bridge", replacement_reason="Replace placeholder with authoritative Hartigan dip delegate")
class TSHartiganDipPolarsNative(SeriesOperator):
    from factor_engine.cleaned_operators.moments_ext import TsHartiganDip as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    _physical_spec = PhysicalImplementationSpec(canonical="ts_hartigan_dip", backend="polars", execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE, supports_lazy=False, supports_streaming=False, materializes_full_panel=True, supports_nulls=True, supports_nan=True, supports_inf=True)
    def _calculate_series(self, x, window=120, min_periods=20, min_coverage_fraction=0.8, **kwargs):
        return _call_pandas_delegate("ts_hartigan_dip", (x,), {"window":window,"min_periods":min_periods,"min_coverage_fraction":min_coverage_fraction,**kwargs})


@register_operator(name="ts_higuchi_fractal_dimension", canonical="ts_higuchi_fractal_dimension", backend="polars", source="factor_dsl_polars_native", replace=True, expected_old_source="pandas_bridge", replacement_reason="Replace placeholder with authoritative Higuchi delegate")
class TSHiguchiFractalDimensionPolarsNative(SeriesOperator):
    """Exact eager delegate to the authoritative Higuchi implementation."""

    metadata = OperatorMetadata(
        name="ts_higuchi_fractal_dimension",
        category="complexity",
        description="Higuchi fractal dimension via the authoritative CPU reference",
        param_names=["x", "window", "k_max"],
        return_type="series",
        tags=["complexity", "rolling", "pit_safe", "causal", "pandas_delegate"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, max=512, default=120, history_semantics="max_rows", param_role=ParamRole.HORIZON),
        "k_max": ParamSpec(dtype=int, min=1, max=32, default=8, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_higuchi_fractal_dimension", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        materializes_full_panel=True, requires_sorted=True,
        supports_nulls=True, supports_nan=True, supports_inf=False,
        implementation_source_hash="5911e76ea2da9be8f247aff850e4fcd4146f056556e1cc96392e8d6bb773f28d",
        emitter_identity="rolling_pack._call_pandas_delegate:polars_to_pandas_to_polars:v1",
        kernel_identity="sequence_complexity:TsHiguchiFractalDimension._calculate_series",
        parameter_domain_hash="ca1676da0b07e78538e5c6e2538f4cfa2659ddcb3e6c5b7c4c569d5923dbf736",
        semantic_contract_hash="082fab16cf846a558e0af004c034e0a3cdcd61bf681af8fd691e6593d8f234e2",
        notes="Exact eager pandas-reference delegation; never a native Polars expression.",
    )

    def _calculate_series(self, x, window=120, k_max=8, **kwargs):
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate(
            "ts_higuchi_fractal_dimension", (x,),
            {"window": window, "k_max": k_max, **kwargs},
        )


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
        param_names=["x","window","side","tail_fraction","min_tail_count"],
        return_type="series",
        tags=["time_series", "rolling", "extreme_tail", "pit_safe", "polars", "cpu_udf"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "tail_fraction": ParamSpec(dtype=float, min=0.01, max=0.5, param_role=ParamRole.STATE_THRESHOLD),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_hill_tail_index",
        backend="polars",
        execution_kind=ExecutionKind.DELEGATE_PYTHON,
        supports_lazy=False,
        supports_streaming=False,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash="polars_native.ts_advanced_batch5:hill_shared_numpy:v1",
        emitter_identity="polars.DataFrame.with_columns:python_numpy_udf",
        kernel_identity="extreme_tail._hill_series:classic_hill:v1",
        parameter_domain_hash="window>=20;side=upper|lower;0.01<=tail_fraction<=0.5;min_tail_count>=3",
        semantic_contract_hash="classic_hill:positive_threshold:mirrored_lower:shared_cpu_kernel:v1",
        notes="Eager per-column Python/NumPy UDF; explicitly not a native Polars expression.",
    )

    def _calculate_series(
        self,
        feature,
        window=120,
        side="upper",
        tail_fraction=0.2,
        min_tail_count=10,
        **kwargs,
    ):
        """Run the authoritative Hill kernel without a pandas round trip."""
        from factor_engine.cleaned_operators.extreme_tail import _hill_series

        side_k = str(side).lower()
        if side_k not in {"upper", "lower"}:
            raise ValueError("ts_hill_tail_index requires side in {'upper','lower'}")
        if "stock_code" in feature.columns and feature["stock_code"].drop_nulls().n_unique() > 1:
            raise ValueError(
                "ts_hill_tail_index Polars CPU UDF requires a wide panel or a single-stock "
                "long frame; multi-stock long input must be isolated by stock_code first"
            )

        value_columns = [
            name for name in feature.columns if name not in {"date", "stock_code"}
        ]
        results = []
        for name in value_columns:
            values = feature.select(name).to_series().to_numpy().astype(np.float64)
            results.append(
                pl.Series(
                    name=name,
                    values=_hill_series(
                        values,
                        window,
                        side_k,
                        tail_fraction,
                        min_tail_count,
                    ),
                    dtype=pl.Float64,
                )
            )
        return feature.with_columns(results)


@register_operator(name="ts_hsic", canonical="ts_hsic", backend="polars", status="research_only")
class TSHSICPolarsNative(SeriesOperator):
    """Exact reference dependence; explicit Pandas delegation, not a proxy."""
    from factor_engine.cleaned_operators.common.dependence_delegate import metadata as _metadata, physical_spec as _spec
    metadata = _metadata("ts_hsic")
    _physical_spec = _spec("ts_hsic")
    @property
    def _contract_callable(self):
        from factor_engine.cleaned_operators.common.dependence_delegate import reference
        return reference("ts_hsic")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        from factor_engine.cleaned_operators.common.dependence_delegate import calculate
        return calculate("ts_hsic",*args,**kwargs)


# ============================================================================
# Hurst DFA
# ============================================================================

@register_operator(name="ts_hurst_dfa", canonical="ts_hurst_dfa", backend="polars", source="factor_dsl_polars_native", replace=True, expected_old_source="pandas_bridge", replacement_reason="Replace placeholder with authoritative DFA delegate")
class TSHurstDFAPolarsNative(SeriesOperator):
    """Exact eager delegate to the authoritative DFA implementation."""

    metadata = OperatorMetadata(
        name="ts_hurst_dfa",
        category="complexity",
        description="DFA Hurst exponent via the authoritative CPU reference",
        param_names=["x", "window", "min_scale", "max_scale", "n_scales"],
        return_type="series",
        tags=["complexity", "rolling", "long_memory", "pit_safe", "causal", "pandas_delegate"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, max=512, default=120, history_semantics="max_rows", param_role=ParamRole.HORIZON),
        "min_scale": ParamSpec(dtype=int, min=2, default=4, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "max_scale": ParamSpec(dtype=int, min=2, default=None, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "n_scales": ParamSpec(dtype=int, min=3, default=6, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_hurst_dfa", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        materializes_full_panel=True, requires_sorted=True,
        supports_nulls=True, supports_nan=True, supports_inf=False,
        implementation_source_hash="5d43466147ebdb3d02e887741d184110a377c7cc0f3f238e879c34a74c5bbcb7",
        emitter_identity="rolling_pack._call_pandas_delegate:polars_to_pandas_to_polars:v1",
        kernel_identity="sequence_complexity:TsHurstDfa._calculate_series",
        parameter_domain_hash="9094a72dd2e618f310622461bd6f2e8068d14330d585d70fd7eaa4b02384543b",
        semantic_contract_hash="ea89402f6b2c0634f38b1f2dcb419d8b21bf839a4a203c09e0619ef144c674b4",
        notes="Exact eager pandas-reference delegation; never a native Polars expression.",
    )

    def _calculate_series(self, x, window=120, min_scale=4, max_scale=None, n_scales=6, **kwargs):
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate(
            "ts_hurst_dfa", (x,), {
                "window": window, "min_scale": min_scale,
                "max_scale": max_scale, "n_scales": n_scales, **kwargs,
            },
        )


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
        param_names=["x","window","min_periods","min_nodes","min_coverage_fraction"],
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
        param_names=["x","window","min_periods","min_nodes","min_coverage_fraction"],
        return_type="series",
        tags=["time_series", "rolling", "network", "hvg", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=8, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, window, min_periods=5, min_nodes=6, min_coverage_fraction=1.0, **kwargs):
        # R4-100 parity: canonical (x, window, min_periods, min_nodes,
        # min_coverage_fraction); ``feature`` is legacy.
        feature = x
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
        param_names=["x","window","min_periods","min_nodes","min_coverage_fraction"],
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
        param_names=["x","window","min_periods","min_nodes","min_coverage_fraction"],
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
        param_names=["x","window","min_periods","min_nodes","min_coverage_fraction"],
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
        param_names=["z","upper","lower","cap","missing_policy"],
        return_type="series",
        tags=["time_series", "hysteresis", "state", "pit_safe"],
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, z, upper=None, lower=None, cap=10, missing_policy="keep", **kwargs):
        feature = z
        upper_threshold = upper
        lower_threshold = lower
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
        param_names=["z", "upper", "lower", "cap", "missing_policy"],
        return_type="series",
        tags=["time_series", "hysteresis", "state", "pit_safe"],
    )
    # metadata.param_specs intentionally omitted - use canonical contract from pandas backend

    def _calculate_series(self, z, upper=None, lower=None, cap=10, missing_policy="keep", **kwargs):
        feature = z
        upper_threshold = upper
        lower_threshold = lower
        # TODO: Implement proper stateful hysteresis
        # Placeholder: simple ternary threshold
        high = feature > upper_threshold
        low = feature < lower_threshold
        return pl.when(high).then(pl.lit(1.0)).when(low).then(pl.lit(-1.0)).otherwise(pl.lit(0.0))


# ============================================================================
# Industry and Market Liquidity Betas
# ============================================================================

@register_operator(name="ts_industry_liquidity_beta", canonical="ts_industry_liquidity_beta", backend="polars", semantic_version="2.0")
class TSIndustryLiquidityBetaPolarsNative(SeriesOperator):
    """Beta to industry-level liquidity factor (rolling regression)"""

    metadata = OperatorMetadata(
        name="ts_industry_liquidity_beta",
        category="time_series",
        description="Rolling beta to industry liquidity factor",
        param_names=["own_return", "industry_liquidity", "window"],
        return_type="series",
        tags=["time_series", "rolling", "regression", "liquidity", "pit_safe", "native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, default=60, param_role=ParamRole.HORIZON),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_industry_liquidity_beta", backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR, supports_lazy=False,
        materializes_full_panel=True, requires_sorted=True, supports_nulls=True,
        supports_nan=True, supports_inf=True,
        implementation_source_hash="polars_native.ts_advanced_batch5:industry_liquidity_beta:v2",
        kernel_identity="polars.diff+rolling_cov_ddof0/rolling_var_ddof0:centered-origin:v2",
        parameter_domain_hash="window:int:min=2:default=60",
        semantic_contract_hash="liquidity_beta:on_liquidity_change:pairwise_finite:v2",
    )

    def _calculate_series(self, own_return, industry_liquidity, window=60, **kwargs):
        from factor_engine.cleaned_operators.ts_model.polars_regression import (
            _liquidity_delta, _pairwise_rolling, align_cols,
        )
        w = int(window)
        if w < 3:
            return own_return.with_columns([
                pl.lit(None, dtype=pl.Float64).alias(c)
                for c in align_cols(own_return, industry_liquidity)
            ])
        return _pairwise_rolling(
            own_return, _liquidity_delta(industry_liquidity), w, max(3, w // 5)
        )


@register_operator(name="ts_market_liquidity_beta", canonical="ts_market_liquidity_beta", backend="polars", semantic_version="2.0")
class TSMarketLiquidityBetaPolarsNative(SeriesOperator):
    """Beta to market-wide liquidity factor"""

    metadata = OperatorMetadata(
        name="ts_market_liquidity_beta",
        category="time_series",
        description="Rolling beta to market liquidity factor",
        param_names=["own_return", "market_liquidity", "window"],
        return_type="series",
        tags=["time_series", "rolling", "regression", "liquidity", "pit_safe", "native"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, default=60, param_role=ParamRole.HORIZON),
    }
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_market_liquidity_beta", backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR, supports_lazy=False,
        materializes_full_panel=True, requires_sorted=True, supports_nulls=True,
        supports_nan=True, supports_inf=True,
        implementation_source_hash="polars_native.ts_advanced_batch5:market_liquidity_beta:v2",
        kernel_identity="polars.diff+rolling_cov_ddof0/rolling_var_ddof0:centered-origin:v2",
        parameter_domain_hash="window:int:min=2:default=60",
        semantic_contract_hash="liquidity_beta:on_liquidity_change:pairwise_finite:v2",
    )

    def _calculate_series(self, own_return, market_liquidity, window=60, **kwargs):
        from factor_engine.cleaned_operators.ts_model.polars_regression import (
            _liquidity_delta, _pairwise_rolling, align_cols,
        )
        w = int(window)
        if w < 3:
            return own_return.with_columns([
                pl.lit(None, dtype=pl.Float64).alias(c)
                for c in align_cols(own_return, market_liquidity)
            ])
        return _pairwise_rolling(
            own_return, _liquidity_delta(market_liquidity), w, max(3, w // 5)
        )


# ============================================================================
# Interval Geometry
# ============================================================================

@register_operator(name="ts_interval_exploration_efficiency", canonical="ts_interval_exploration_efficiency", backend="polars", status="research_only")
class TSIntervalExplorationEfficiencyPolarsNative(SeriesOperator):
    """Exact interval-geometry delegate, with authored canonical input roles."""
    from factor_engine.cleaned_operators.common import interval_delegate as _interval
    metadata=_interval.metadata("ts_interval_exploration_efficiency")
    _physical_spec=_interval.physical_spec("ts_interval_exploration_efficiency")
    @property
    def _contract_callable(self):
        return self._interval.reference("ts_interval_exploration_efficiency")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._interval.calculate("ts_interval_exploration_efficiency",*args,**kwargs)


@register_operator(name="ts_interval_nesting_depth", canonical="ts_interval_nesting_depth", backend="polars", status="research_only")
class TSIntervalNestingDepthPolarsNative(SeriesOperator):
    """Exact interval-geometry delegate, with authored canonical input roles."""
    from factor_engine.cleaned_operators.common import interval_delegate as _interval
    metadata=_interval.metadata("ts_interval_nesting_depth")
    _physical_spec=_interval.physical_spec("ts_interval_nesting_depth")
    @property
    def _contract_callable(self):
        return self._interval.reference("ts_interval_nesting_depth")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._interval.calculate("ts_interval_nesting_depth",*args,**kwargs)


@register_operator(name="ts_interval_occupancy_entropy", canonical="ts_interval_occupancy_entropy", backend="polars", status="research_only")
class TSIntervalOccupancyEntropyPolarsNative(SeriesOperator):
    """Exact interval-geometry delegate, with authored canonical input roles."""
    from factor_engine.cleaned_operators.common import interval_delegate as _interval
    metadata=_interval.metadata("ts_interval_occupancy_entropy")
    _physical_spec=_interval.physical_spec("ts_interval_occupancy_entropy")
    @property
    def _contract_callable(self):
        return self._interval.reference("ts_interval_occupancy_entropy")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._interval.calculate("ts_interval_occupancy_entropy",*args,**kwargs)


@register_operator(name="ts_interval_occupancy_mode_distance", canonical="ts_interval_occupancy_mode_distance", backend="polars", status="research_only")
class TSIntervalOccupancyModeDistancePolarsNative(SeriesOperator):
    """Exact interval-geometry delegate, with authored canonical input roles."""
    from factor_engine.cleaned_operators.common import interval_delegate as _interval
    metadata=_interval.metadata("ts_interval_occupancy_mode_distance")
    _physical_spec=_interval.physical_spec("ts_interval_occupancy_mode_distance")
    @property
    def _contract_callable(self):
        return self._interval.reference("ts_interval_occupancy_mode_distance")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._interval.calculate("ts_interval_occupancy_mode_distance",*args,**kwargs)


@register_operator(name="ts_interval_overlap_connected_component_ratio", canonical="ts_interval_overlap_connected_component_ratio", backend="polars", status="research_only")
class TSIntervalOverlapConnectedComponentRatioPolarsNative(SeriesOperator):
    """Exact interval-geometry delegate, with authored canonical input roles."""
    from factor_engine.cleaned_operators.common import interval_delegate as _interval
    metadata=_interval.metadata("ts_interval_overlap_connected_component_ratio")
    _physical_spec=_interval.physical_spec("ts_interval_overlap_connected_component_ratio")
    @property
    def _contract_callable(self):
        return self._interval.reference("ts_interval_overlap_connected_component_ratio")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._interval.calculate("ts_interval_overlap_connected_component_ratio",*args,**kwargs)


@register_operator(name="ts_interval_union_coverage", canonical="ts_interval_union_coverage", backend="polars")
class TSIntervalUnionCoveragePolarsNative(SeriesOperator):
    """Exact interval-geometry delegate, with authored canonical input roles."""
    from factor_engine.cleaned_operators.common import interval_delegate as _interval
    metadata=_interval.metadata("ts_interval_union_coverage")
    _physical_spec=_interval.physical_spec("ts_interval_union_coverage")
    @property
    def _contract_callable(self):
        return self._interval.reference("ts_interval_union_coverage")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._interval.calculate("ts_interval_union_coverage",*args,**kwargs)

@register_operator(name="ts_joint_energy_shift", canonical="ts_joint_energy_shift", backend="polars", status="research_only")
class TSJointEnergyShiftPolarsNative(SeriesOperator):
    """Exact distribution reference, not a scalar proxy or zero placeholder."""
    from factor_engine.cleaned_operators.common import distribution_delegate as _delegate
    metadata=_delegate.metadata("ts_joint_energy_shift")
    @property
    def _contract_callable(self):
        return self._delegate.reference("ts_joint_energy_shift")._calculate_series
    def _calculate_series(self,*args,**kwargs):
        return self._delegate.calculate("ts_joint_energy_shift",*args,**kwargs)
    def physical_spec(self):
        return self._delegate.physical_spec("ts_joint_energy_shift")


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
        param_names=["x","window","bins","lag","min_count","min_state_support","min_history"],
        return_type="series",
        tags=["time_series", "rolling", "stochastic", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        # P0-B1-FINALIZE: canonical R4-100 contract 7-param; trailing scalar
        # params (bins/lag/min_count/...) are accepted via **kwargs in the
        # kernel below.  ``bins`` etc are NOT declared in param_names so leave
        # param_specs to the canonical backfill (P0-23 divergence warn-only).
    }

    def _calculate_series(self, x, window, bins=10, lag=1, min_count=5, min_state_support=5, min_history=20, **kwargs):
        # R4-100 parity: canonical (x, window, bins, lag, min_count,
        # min_state_support, min_history); ``feature`` is legacy.
        feature = x
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
        param_names=["x", "window", "bins", "lag", "min_count", "min_state_support", "min_history"],
        return_type="series",
        tags=["time_series", "rolling", "stochastic", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, feature, window, **kwargs):
        # Placeholder: local volatility barrier
        return feature.rolling_std(window) * window ** 0.5


@register_operator(name="ts_kramers_moyal_diffusion", canonical="ts_kramers_moyal_diffusion", backend="polars")
class TSKramersMoyalDiffusionPolarsNative(SeriesOperator):
    """Causal canonical reference with explicit Pandas conversion."""
    from factor_engine.cleaned_operators.common import structure_delegate as _delegate
    metadata = _delegate.metadata("ts_kramers_moyal_diffusion")
    _physical_spec = _delegate.physical_spec("ts_kramers_moyal_diffusion")

    def _calculate_series(self, *args, **kwargs):
        return self._delegate.calculate("ts_kramers_moyal_diffusion", *args, **kwargs)


@register_operator(name="ts_kramers_moyal_drift", canonical="ts_kramers_moyal_drift", backend="polars")
class TSKramersMoyalDriftPolarsNative(SeriesOperator):
    """Causal canonical reference with explicit Pandas conversion."""
    from factor_engine.cleaned_operators.common import structure_delegate as _delegate
    metadata = _delegate.metadata("ts_kramers_moyal_drift")
    _physical_spec = _delegate.physical_spec("ts_kramers_moyal_drift")

    def _calculate_series(self, *args, **kwargs):
        return self._delegate.calculate("ts_kramers_moyal_drift", *args, **kwargs)


@register_operator(name="ts_kramers_moyal_local_stability", canonical="ts_kramers_moyal_local_stability", backend="polars")
class TSKramersMoyalLocalStabilityPolarsNative(SeriesOperator):
    """Exact canonical Markov calculation with explicit reference conversion."""
    from factor_engine.cleaned_operators.common import markov_reference_delegate as _delegate
    metadata = _delegate.metadata("ts_kramers_moyal_local_stability")

    @property
    def _contract_callable(self):
        return self._delegate.reference("ts_kramers_moyal_local_stability")._calculate_series

    def physical_spec(self):
        return self._delegate.physical_spec("ts_kramers_moyal_local_stability")

    def _calculate_series(self, *args, **kwargs):
        return self._delegate.calculate("ts_kramers_moyal_local_stability", *args, **kwargs)


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
        param_names=["x","recent_window","old_window","min_periods"],
        return_type="series",
        tags=["time_series", "rolling", "distribution", "pit_safe"],
    )
    metadata.param_specs = {
        "recent_window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
        "old_window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, x, recent_window=20, old_window=20, min_periods=10, **kwargs):
        # R4-100 parity: canonical (x, recent_window, old_window, min_periods).
        feature = x
        window = recent_window
        # TODO: Implement proper KS test between windows
        # Placeholder: quantile shift
        q50_shift = feature.rolling_median(window).diff()
        return q50_shift.abs()


@register_operator(name="ts_l_kurtosis", canonical="ts_l_kurtosis", category="moments", business_category="moments", backend="polars", status="research_only", source="factor_dsl_polars_native", replace=True, expected_old_source="pandas_bridge", replacement_reason="Replace placeholder with authoritative L-kurtosis delegate")
class TSLKurtosisPolarsNative(SeriesOperator):
    from factor_engine.cleaned_operators.moments_ext import TsLKurtosis as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    _physical_spec = PhysicalImplementationSpec(canonical="ts_l_kurtosis", backend="polars", execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE, supports_lazy=False, supports_streaming=False, materializes_full_panel=True, supports_nulls=True, supports_nan=True, supports_inf=True)
    def _calculate_series(self, x, window=60, min_periods=20, min_coverage_fraction=0.5, **kwargs):
        return _call_pandas_delegate("ts_l_kurtosis", (x,), {"window":window,"min_periods":min_periods,"min_coverage_fraction":min_coverage_fraction,**kwargs})


@register_operator(name="ts_l_skewness", canonical="ts_l_skewness", category="moments", business_category="moments", backend="polars", status="research_only", source="factor_dsl_polars_native", replace=True, expected_old_source="pandas_bridge", replacement_reason="Replace placeholder with authoritative L-skewness delegate")
class TSLSkewnessPolarsNative(SeriesOperator):
    from factor_engine.cleaned_operators.moments_ext import TsLSkewness as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    _physical_spec = PhysicalImplementationSpec(canonical="ts_l_skewness", backend="polars", execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE, supports_lazy=False, supports_streaming=False, materializes_full_panel=True, supports_nulls=True, supports_nan=True, supports_inf=True)
    def _calculate_series(self, x, window=60, min_periods=20, min_coverage_fraction=0.5, **kwargs):
        return _call_pandas_delegate("ts_l_skewness", (x,), {"window":window,"min_periods":min_periods,"min_coverage_fraction":min_coverage_fraction,**kwargs})


# ============================================================================
# Lagged Correlations and Mutual Information
# ============================================================================

@register_operator(name="ts_lag_of_peak_corr", canonical="ts_lag_of_peak_corr", backend="polars", status="research_only")
class TSLagOfPeakCorrPolarsNative(SeriesOperator):
    """Exact labelled-panel delegate to the two-input lag-search authority."""
    from factor_engine.cleaned_operators.stateful.sequential import TsLagOfPeakCorr as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    metadata.tags = list(metadata.tags) + ["polars", "delegate:pandas_numpy"]
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_lag_of_peak_corr", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False, supports_streaming=False, materializes_full_panel=True,
        supports_nulls=True, supports_nan=True, supports_inf=True,
    )

    def _calculate_series(self, x, y, window=20, max_lag=5, min_periods=None, **kwargs):
        from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
        return _call_pandas_delegate("ts_lag_of_peak_corr", (x, y), {
            "window": window, "max_lag": max_lag, "min_periods": min_periods,
            **kwargs,
        })


@register_operator(name="ts_lagged_mutual_information", canonical="ts_lagged_mutual_information", backend="polars", status="research_only")
class TSLaggedMutualInformationPolarsNative(SeriesOperator):
    """Exact delegate to the deterministic lagged-MI reference."""
    from factor_engine.cleaned_operators.nonlinear_dependence import TsLaggedMutualInformation as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    metadata.tags = list(metadata.tags or []) + ["polars", "delegate:pandas_numpy"]
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_lagged_mutual_information", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False, supports_streaming=False, materializes_full_panel=True,
        supports_nulls=True, supports_nan=True, supports_inf=True,
    )

    def _calculate_series(self, x, y, window=40, lag=1, bins=5, min_periods=10, bias_correction=True, **kwargs):
        return _call_pandas_delegate(
            "ts_lagged_mutual_information", (x, y),
            {"window": window, "lag": lag, "bins": bins, "min_periods": min_periods,
             "bias_correction": bias_correction, **kwargs},
        )


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
    """ts_level_shift_score: exact canonical estimator with finite-support policy."""
    from factor_engine.cleaned_operators.common import regression_model_polars as _model
    metadata=OperatorMetadata(name="ts_level_shift_score",category="time_series",**_model.contract("ts_level_shift_score"))
    _physical_spec=_model.physical_spec("ts_level_shift_score")
    _contract_callable=staticmethod(_model.ts_level_shift_score)
    def _calculate_series(self,*args,**kwargs):
        return self._model.ts_level_shift_score(*args,**kwargs)

@register_operator(name="ts_leverage_effect", canonical="ts_leverage_effect", backend="polars")
class TSLeverageEffectPolarsNative(SeriesOperator):
    """Causal leverage effect: correlation of lagged return and trailing volatility."""

    metadata = OperatorMetadata(
        name="ts_leverage_effect",
        category="time_series",
        description="Causal correlation between lagged returns and current trailing volatility",
        param_names=["ret","window","min_periods"],
        return_type="series",
        tags=["time_series", "rolling", "volatility", "asymmetry", "pit_safe"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=10, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, ret, window, min_periods=1, **kwargs):
        feature = ret
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
@register_operator(name="ts_line_convergence", canonical="ts_line_convergence", backend="polars", source="pandas_bridge.structure_parity")
class TSLineConvergencePolarsNative(SeriesOperator):
    """Exact conversion of the authoritative canonical structure operator."""
    metadata = OperatorMetadata(
        name="ts_line_convergence", category="time_series",
        description="Canonical ts_line_convergence conversion bridge.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points'], panel_params=('high', 'low'),
        scalar_params=('left_window', 'right_window', 'history_window', 'points'), input_arity=2,
        panel_arity=2, total_positional_arity=6,
        param_specs={'left_window': ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON), 'right_window': ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON), 'history_window': ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON), 'points': ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON)},
        tags=["time_series", "pandas_bridge", "pit_safe"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import ts_line_convergence
        return ts_line_convergence(high, low, left_window, right_window, history_window, points)


@register_operator(name="ts_line_parallelism", canonical="ts_line_parallelism", backend="polars")
@register_operator(name="ts_line_parallelism", canonical="ts_line_parallelism", backend="polars", source="pandas_bridge.structure_parity")
class TSLineParallelismPolarsNative(SeriesOperator):
    """Exact conversion of the authoritative canonical structure operator."""
    metadata = OperatorMetadata(
        name="ts_line_parallelism", category="time_series",
        description="Canonical ts_line_parallelism conversion bridge.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points'], panel_params=('high', 'low'),
        scalar_params=('left_window', 'right_window', 'history_window', 'points'), input_arity=2,
        panel_arity=2, total_positional_arity=6,
        param_specs={'left_window': ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON), 'right_window': ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON), 'history_window': ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON), 'points': ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON)},
        tags=["time_series", "pandas_bridge", "pit_safe"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import ts_line_parallelism
        return ts_line_parallelism(high, low, left_window, right_window, history_window, points)


# ============================================================================
# Lo-MacKinlay Variance Ratio
# ============================================================================

def _lo_mackinlay_frame(x, window, q, min_periods, *, z_score):
    """Pure-Polars rolling Lo–MacKinlay estimator on contiguous level runs."""
    w, periods = int(window), int(q)
    if periods < 2 or periods >= w:
        raise ValueError("q must be >= 2 and < window")
    mp = max(6, int(min_periods))
    value_cols = [name for name, dtype in x.schema.items() if dtype.is_numeric()]
    axes = [name for name in x.columns if name not in value_cols]
    outputs = []
    for number, column in enumerate(value_cols):
        prefix = f"__lm_{number}_"
        group, ret, qret = prefix + "group", prefix + "ret", prefix + "qret"
        n, mu, s1, s2 = prefix + "n", prefix + "mu", prefix + "s1", prefix + "s2"
        qn, qs1, qs2 = prefix + "qn", prefix + "qs1", prefix + "qs2"
        sigma_a, vr = prefix + "sigma_a", prefix + "vr"
        lf = x.lazy().with_columns(
            pl.col(column).is_null().cast(pl.UInt32).cum_sum().alias(group)
        ).with_columns(
            pl.col(column).diff().over(group).alias(ret),
            pl.col(column).diff(periods).over(group).alias(qret),
        ).with_columns(
            pl.col(ret).is_not_null().cast(pl.UInt32).rolling_sum(
                w - 1, min_samples=1).over(group).alias(n),
            pl.col(ret).rolling_sum(w - 1, min_samples=1).over(group).alias(s1),
            (pl.col(ret) ** 2).rolling_sum(w - 1, min_samples=1).over(group).alias(s2),
            pl.col(qret).is_not_null().cast(pl.UInt32).rolling_sum(
                w - periods, min_samples=1).over(group).alias(qn),
            pl.col(qret).rolling_sum(w - periods, min_samples=1).over(group).alias(qs1),
            (pl.col(qret) ** 2).rolling_sum(w - periods, min_samples=1).over(group).alias(qs2),
        ).with_columns(
            (pl.col(s1) / pl.col(n)).alias(mu),
            ((pl.col(s2) - pl.col(s1) ** 2 / pl.col(n)) / (pl.col(n) - 1)).alias(sigma_a),
        ).with_columns(
            (
                ((pl.col(qs2) - 2 * periods * pl.col(mu) * pl.col(qs1)
                  + pl.col(qn) * (periods * pl.col(mu)) ** 2)
                 / (periods * pl.col(qn) * (1 - periods / pl.col(n))))
                / pl.col(sigma_a)
            ).alias(vr)
        )
        valid = ((pl.col(n) >= mp - 1) & (pl.col(n) >= periods + 1) &
                 (pl.col(qn) == pl.col(n) - periods + 1) & (pl.col(sigma_a) > 0))
        if z_score:
            theta_terms = []
            for lag in range(1, periods):
                a2b2, a2b, ab2, a2, b2, ab, a1, b1, count = (
                    prefix + f"k{lag}_{suffix}" for suffix in
                    ("a2b2", "a2b", "ab2", "a2", "b2", "ab", "a1", "b1", "count")
                )
                pair_window = w - 1 - lag
                shifted = pl.col(ret).shift(lag).over(group)
                lf = lf.with_columns(
                    ((pl.col(ret) ** 2 * shifted ** 2).rolling_sum(pair_window, min_samples=1).over(group)).alias(a2b2),
                    ((pl.col(ret) ** 2 * shifted).rolling_sum(pair_window, min_samples=1).over(group)).alias(a2b),
                    ((pl.col(ret) * shifted ** 2).rolling_sum(pair_window, min_samples=1).over(group)).alias(ab2),
                    (pl.when(shifted.is_not_null()).then(pl.col(ret) ** 2)
                     .rolling_sum(pair_window, min_samples=1).over(group)).alias(a2),
                    (pl.when(pl.col(ret).is_not_null()).then(shifted ** 2)
                     .rolling_sum(pair_window, min_samples=1).over(group)).alias(b2),
                    ((pl.col(ret) * shifted).rolling_sum(pair_window, min_samples=1).over(group)).alias(ab),
                    (pl.when(shifted.is_not_null()).then(pl.col(ret))
                     .rolling_sum(pair_window, min_samples=1).over(group)).alias(a1),
                    (pl.when(pl.col(ret).is_not_null()).then(shifted)
                     .rolling_sum(pair_window, min_samples=1).over(group)).alias(b1),
                    ((pl.col(ret).is_not_null() & shifted.is_not_null()).cast(pl.UInt32)
                     .rolling_sum(pair_window, min_samples=1).over(group)).alias(count),
                )
                numerator = (
                    pl.col(a2b2) - 2 * pl.col(mu) * (pl.col(a2b) + pl.col(ab2))
                    + pl.col(mu) ** 2 * (pl.col(a2) + pl.col(b2) + 4 * pl.col(ab))
                    - 2 * pl.col(mu) ** 3 * (pl.col(a1) + pl.col(b1))
                    + pl.col(count) * pl.col(mu) ** 4
                )
                theta_terms.append((2.0 * (periods - lag) / periods) ** 2 *
                                   numerator / (pl.col(s2) - pl.col(s1) ** 2 / pl.col(n)) ** 2)
            theta = sum(theta_terms)
            expression = pl.when(valid & (theta > 0)).then(
                (pl.col(vr) - 1.0) / theta.sqrt()
            ).otherwise(None).alias(column)
        else:
            expression = pl.when(valid).then(pl.col(vr)).otherwise(None).alias(column)
        outputs.append(lf.select(expression).collect().to_series())
    result = x.select(axes) if axes else pl.DataFrame()
    return result.with_columns(outputs)

@register_operator(name="ts_lo_mackinlay_vr", canonical="ts_lo_mackinlay_vr", backend="polars")
class TSLoMackinlayVRPolarsNative(SeriesOperator):
    """ts_lo_mackinlay_vr: exact canonical estimator with finite-support policy."""
    from factor_engine.cleaned_operators.common import regression_model_polars as _model
    metadata=OperatorMetadata(name="ts_lo_mackinlay_vr",category="time_series",**_model.contract("ts_lo_mackinlay_vr"))
    _physical_spec=_model.physical_spec("ts_lo_mackinlay_vr")
    _contract_callable=staticmethod(_model.ts_lo_mackinlay_vr)
    def _calculate_series(self,*args,**kwargs):
        return self._model.ts_lo_mackinlay_vr(*args,**kwargs)


@register_operator(name="ts_lo_mackinlay_z", canonical="ts_lo_mackinlay_z", backend="polars")
class TSLoMackinlayZPolarsNative(SeriesOperator):
    """ts_lo_mackinlay_z: exact canonical estimator with finite-support policy."""
    from factor_engine.cleaned_operators.common import regression_model_polars as _model
    metadata=OperatorMetadata(name="ts_lo_mackinlay_z",category="time_series",**_model.contract("ts_lo_mackinlay_z"))
    _physical_spec=_model.physical_spec("ts_lo_mackinlay_z")
    _contract_callable=staticmethod(_model.ts_lo_mackinlay_z)
    def _calculate_series(self,*args,**kwargs):
        return self._model.ts_lo_mackinlay_z(*args,**kwargs)


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
    """Exact delegate to the directional fixed-mass lower-tail reference."""
    from factor_engine.cleaned_operators.nonlinear_dependence import TsLowerTailDependence as _Reference
    metadata = copy.deepcopy(_Reference.metadata)
    metadata.tags = list(metadata.tags or []) + ["polars", "delegate:pandas_numpy"]
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_lower_tail_coexceedance_probability", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False, supports_streaming=False, materializes_full_panel=True,
        supports_nulls=True, supports_nan=True, supports_inf=True,
    )

    def _calculate_series(self, x, y, window=60, q=0.1, min_tail_count=5, **kwargs):
        return _call_pandas_delegate(
            "ts_lower_tail_coexceedance_probability", (x, y),
            {"window": window, "q": q, "min_tail_count": min_tail_count, **kwargs},
        )


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
    """Exact canonical Markov calculation with explicit reference conversion."""
    from factor_engine.cleaned_operators.common import markov_reference_delegate as _delegate
    metadata = _delegate.metadata("ts_markov_entropy_production")

    @property
    def _contract_callable(self):
        return self._delegate.reference("ts_markov_entropy_production")._calculate_series

    def physical_spec(self):
        return self._delegate.physical_spec("ts_markov_entropy_production")

    def _calculate_series(self, *args, **kwargs):
        return self._delegate.calculate("ts_markov_entropy_production", *args, **kwargs)


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
        param_names=["x","window","bins","lag","min_count","min_state_support","min_history","target"],
        return_type="series",
        tags=["time_series", "rolling", "markov", "pit_safe", "research_only"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20, param_role=ParamRole.HORIZON),
        "target": ParamSpec(dtype=str, default="upper", param_role=ParamRole.POLICY),
    }

    def _calculate_series(self, x, window, bins=10, lag=1, min_count=5, min_state_support=5, min_history=20, target="upper", **kwargs):
        # R4-100 parity: canonical 8-param contract.
        feature = x
        threshold = 0.0
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
