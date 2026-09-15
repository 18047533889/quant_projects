# -*- coding: utf-8 -*-
"""Polars backends for the 2026-08 geometry/math expansion.

The three envelope operators use native Polars expressions. Other slots
delegate to the matching pandas reference and explicitly declare that conversion
path; a Polars-facing API is not evidence of native acceleration or production
certification. Parameter and time-axis contracts remain enforced.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from factor_engine.cleaned_operators.registry import OperatorRegistry

try:
    import polars as pl  # noqa: F401
except Exception:  # pragma: no cover - optional backend
    pl = None

_SKIP_PANEL = frozenset({"date", "stock_code"})

# All canonicals of the 2026-08 geometry/math expansion that have a surviving
# polars delegation backend here (delegates to the pandas_numpy reference).
_GEOMETRY_MATH_CANONICALS: tuple[str, ...] = (
    "ts_joint_energy_shift","ts_energy_break_score","ts_copula_central_asymmetry",
    # interval geometry
    "ts_interval_union_coverage", "ts_interval_occupancy_entropy",
    "ts_interval_occupancy_mode_distance", "ts_interval_nesting_depth",
    "ts_interval_exploration_efficiency", "ts_interval_overlap_connected_component_ratio",
    # structural levels
    "ts_structural_level_density", "ts_nearest_structural_level_distance",
    "ts_structural_level_strength",
    # candle state space
    "ts_vector_state_mahalanobis", "ts_vector_state_local_density",
    "ts_matrix_profile_novelty", "ts_matrix_profile_motif_age",
    # multiscale trend
    "ts_multiscale_trend_consensus", "ts_multiscale_trend_dispersion",
    "ts_multiscale_trend_curvature",
    # envelope / crossing
    "ts_envelope_compression", "ts_envelope_pressure", "ts_envelope_boundary_dwell",
    "ts_crossing_speed", "ts_crossing_acceleration",
    # extrema divergence / threshold cycles
    "ts_extrema_divergence_strength", "ts_extrema_confirmation_rate",
    "ts_threshold_cycle_period", "ts_threshold_cycle_asymmetry",
    # state-episode excursions
    "state_episode_mfe", "state_episode_mae", "state_episode_efficiency",
    "state_episode_retrace_ratio", "state_episode_excursion_balance",
    # jump-robust intraday variation
    "intraday_medrv", "intraday_minrv", "intraday_jump_test_stat",
    # intraday volatility shape
    "intraday_volatility_time_centroid", "intraday_volatility_concentration",
    "intraday_volatility_entropy", "intraday_realized_semivariance_balance",
    "intraday_rv_signature_curvature",
    # volatility roughness
    "ts_vol_pvariation_roughness", "ts_vol_scaling_break",
    # binned response curves
    "ts_binned_response_monotonicity", "ts_binned_response_curvature",
    "ts_response_slope_asymmetry",
    # nonlinear dependence
    "ts_chatterjee_xi", "ts_hsic", "ts_conditional_mutual_information",
    "ts_distance_correlation_partial_proxy",
    # 2D joint trajectory geometry
    "ts_vector_path_efficiency", "ts_vector_turning_coherence",
    "ts_vector_path_curvature", "ts_vector_self_intersection_rate",
    # point-process interval stats
    "event_interval_memory", "event_local_variation", "event_fano_factor",
    # string / ordinal complexity
    "ts_lempel_ziv_complexity", "ts_forbidden_ordinal_pattern_ratio",
    "ts_forbidden_ordinal_pattern_excess", "ts_forbidden_ordinal_pattern_signed_excess",
    # spectral shape
    "ts_spectral_centroid", "ts_spectral_flatness",
    "ts_spectral_peak_concentration", "ts_spectral_quality_factor",
    # Hankel / SSA
    "ts_hankel_effective_rank", "ts_hankel_singular_gap",
    "ts_ssa_reconstruction_residual",
    # multifractal
    "ts_generalized_hurst_exponent", "ts_generalized_hurst_spread_q1_q4",
    "ts_multifractal_spectrum_width", "ts_multifractal_curvature",
    # serial-dependence memory
    "ts_autocorrelation_time", "ts_fractional_difference",
    # L-moments / Hartigan dip / intrinsic dimension / persistence entropy
    "ts_l_skewness", "ts_l_kurtosis", "ts_hartigan_dip",
    "ts_delay_intrinsic_dimension",
    "ts_persistence_entropy_h0", "ts_persistence_entropy_h1",
    # cs / group locality
    "cs_knn_local_moran", "cs_isotonic_residual", "cs_isotonic_residual_lagged_direction",
    "group_current_members_tail_coexceedance", "group_corr_mst_length",
    # intraday session shape
    "intraday_session_shape_novelty", "intraday_profile_pca_residual",
)


def _pl_to_pd(frame: Any) -> Any:
    cols = [c for c in frame.columns if c not in _SKIP_PANEL]
    return frame.select(cols).to_pandas()


def _pl_rebuild(base: Any, pdf: Any) -> Any:
    cols = [c for c in base.columns if c not in _SKIP_PANEL]
    return base.with_columns(
        [pl.Series(name=c, values=np.asarray(pdf[c], dtype=np.float64)) for c in cols]
    )


def _register_polars_backends() -> None:
    if pl is None:
        return
    from factor_engine.cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
    from factor_engine.cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator

    for _canon in _GEOMETRY_MATH_CANONICALS:
        # Skip ops whose module did not load (e.g. optional import failure).
        pandas_op = OperatorRegistry.get(_canon, "pandas_numpy")
        if pandas_op is None:
            continue
        if _canon in {"ts_crossing_speed","ts_crossing_acceleration"}:
            from factor_engine.cleaned_operators.common.crossing_native import make
            OperatorRegistry.register(make(_canon)(),canonical=_canon,backend="polars",
                source="crossing_numpy",status="implemented",backend_explicit=True)
            continue

        if _canon in {"ts_joint_energy_shift","ts_energy_break_score","ts_copula_central_asymmetry"}:
            from factor_engine.cleaned_operators.common.distribution_delegate import register
            register(_canon)
            continue

        if _canon in {"ts_vector_path_efficiency", "ts_vector_turning_coherence",
                      "ts_vector_path_curvature", "ts_vector_self_intersection_rate"}:
            from factor_engine.cleaned_operators.common.vector_path_native import make
            OperatorRegistry.register(
                make(_canon)(), canonical=_canon, backend="polars",
                source="polars_geometry_math", status="implemented",
                backend_explicit=True,
            )
            continue

        if _canon in {"ts_chatterjee_xi","ts_hsic","ts_conditional_mutual_information",
                      "ts_distance_correlation_partial_proxy"}:
            from factor_engine.cleaned_operators.common.dependence_delegate import register
            register(_canon)
            continue

        if _canon in {"ts_structural_level_density","ts_structural_level_strength",
                      "ts_nearest_structural_level_distance"}:
            from factor_engine.cleaned_operators.common.structural_delegate import register
            register(_canon)
            continue

        if _canon in {"intraday_medrv", "intraday_minrv", "intraday_jump_test_stat"}:
            from factor_engine.cleaned_operators.common.jump_robust_delegate import register
            register(_canon)
            continue

        if _canon in {"ts_l_skewness", "ts_l_kurtosis", "ts_hartigan_dip"}:
            from factor_engine.cleaned_operators.common.moments_delegate import register
            register(_canon)
            continue

        # R6-196: carry the availability contract from the pandas reference so
        # the polars delegate slot reports the same session-close availability.
        _ref_meta = getattr(pandas_op, "metadata", None)
        _ref_available_at = getattr(_ref_meta, "available_at", None)
        _ref_same_session = getattr(_ref_meta, "same_session_usable", None)

        class _PolarsGeometryMathBackend(PolarsSeriesOperator):
            # ``_canon`` is bound per-iteration via the default arg (loop-var
            # closure over ``canonical`` would resolve every op to the last one).
            metadata = PolarsMetadata(
                name=_canon, category="geometry_math_polars",
                param_names=list(getattr(_ref_meta, "param_names", None) or ()),
                available_at=_ref_available_at,
                same_session_usable=_ref_same_session,
                **({
                    "panel_params": tuple(_ref_meta.panel_params),
                    "scalar_params": tuple(_ref_meta.scalar_params),
                    "param_specs": dict(_ref_meta.param_specs),
                    "param_aliases": dict(_ref_meta.param_aliases or {}),
                } if _canon in {"ts_envelope_compression", "ts_envelope_pressure", "ts_envelope_boundary_dwell"} else {}),
                **({
                    "panel_params": tuple(getattr(_ref_meta, "panel_params", ()) or ()),
                    "panel_arity": getattr(_ref_meta, "panel_arity", None),
                    "scalar_params": tuple(getattr(_ref_meta, "scalar_params", ()) or ()),
                    "param_specs": dict(getattr(_ref_meta, "param_specs", None) or {}),
                    "param_aliases": dict(getattr(_ref_meta, "param_aliases", None) or {}),
                    "input_units": dict(getattr(_ref_meta, "input_units", None) or {}),
                    "output_unit": getattr(_ref_meta, "output_unit", None),
                    "window_semantics": getattr(_ref_meta, "window_semantics", None),
                    "relational_specs": list(getattr(_ref_meta, "relational_specs", None) or ()),
                } if _canon in {
                    "ts_vector_state_mahalanobis", "ts_vector_state_local_density",
                    "ts_multiscale_trend_consensus", "ts_multiscale_trend_dispersion",
                    "ts_multiscale_trend_curvature",
                    "ts_vol_pvariation_roughness", "ts_vol_scaling_break",
                    "ts_generalized_hurst_spread_q1_q4", "ts_multifractal_spectrum_width",
                    "ts_multifractal_curvature",
                    "ts_multifractal_asymmetry",
                    "ts_threshold_cycle_period", "ts_threshold_cycle_asymmetry",
                    "intraday_session_shape_novelty", "intraday_profile_pca_residual",
                    "ts_extrema_divergence_strength", "ts_extrema_confirmation_rate",
                } else {}),
            )

            @property
            def _contract_callable(self):
                reference = OperatorRegistry.get(self.metadata.name, "pandas_numpy", mode="research")
                return (getattr(reference, "_contract_callable", None)
                        or getattr(reference, "_fn", None)
                        or reference._calculate_series)

            def _calculate_series(self, *frames, _canon=_canon, **params):
                if _canon in {"ts_envelope_compression", "ts_envelope_pressure", "ts_envelope_boundary_dwell"}:
                    from factor_engine.cleaned_operators.envelope import _polars_envelope
                    bound = dict(zip(self.metadata.param_names, frames))
                    bound.update(params)
                    panel_names = self.metadata.panel_params
                    kind = {"ts_envelope_compression": "compression",
                            "ts_envelope_pressure": "pressure",
                            "ts_envelope_boundary_dwell": "dwell"}[_canon]
                    return _polars_envelope(
                        kind, tuple(bound[p] for p in panel_names),
                        bound.get("window", 20), bound.get("quantile", 0.8),
                    )
                from factor_engine.cleaned_operators.common._polars_bridge import (
                    to_pandas_panel, from_pandas_panel,
                )
                op = OperatorRegistry.get(_canon, "pandas_numpy", mode="research")
                panels = [f for f in (*frames, *params.values()) if isinstance(f, pl.DataFrame)]
                if not panels:
                    raise TypeError(f"{_canon}: at least one Polars panel is required")
                convert = lambda value: to_pandas_panel(value) if isinstance(value, pl.DataFrame) else value
                # Scalars stay scalars, and keyword panels are converted too.
                out = op.calculate(*(convert(f) for f in frames),
                                   **{k: convert(v) for k, v in params.items()})
                return from_pandas_panel(panels[0], out)

        if _canon in {"ts_vector_state_mahalanobis", "ts_vector_state_local_density"}:
            import copy
            # These four-panel operators require exact logical-contract parity;
            # a partial reconstruction silently loses descriptions, tags and
            # future metadata fields.
            _PolarsGeometryMathBackend.metadata = copy.deepcopy(_ref_meta)

        if _canon in {"ts_envelope_compression", "ts_envelope_pressure", "ts_envelope_boundary_dwell"}:
            import hashlib
            import inspect
            from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
            from factor_engine.cleaned_operators.envelope import _polars_envelope
            _PolarsGeometryMathBackend._physical_spec = PhysicalImplementationSpec(
                canonical=_canon, backend="polars",
                execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
                materializes_full_panel=True, requires_sorted=True,
                supports_nulls=True, supports_nan=True, supports_inf=True,
                implementation_source_hash=hashlib.sha256(inspect.getsource(_polars_envelope).encode()).hexdigest(),
                emitter_identity="envelope._polars_envelope:polars-expressions:v2",
                kernel_identity="envelope._polars_envelope",
                parameter_domain_hash="envelope:window:int>=2;quantile:float(0,1):v2",
                semantic_contract_hash=_canon + ":supplied-bands:finite-support:prefix-causal:v2",
                notes="Native per-instrument Polars expressions; no pandas/numpy delegation.",
            )

        else:
            import hashlib
            import inspect
            from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
            if _canon in {"ts_vector_state_mahalanobis", "ts_vector_state_local_density"}:
                from pathlib import Path
                from factor_engine.cleaned_operators import candle_state_space
                _source_hash = hashlib.sha256(
                    Path(candle_state_space.__file__).read_bytes() + Path(__file__).read_bytes()
                ).hexdigest()
                _supports_nonfinite = True
            else:
                _source_hash = hashlib.sha256(
                    inspect.getsource(_PolarsGeometryMathBackend._calculate_series).encode()
                ).hexdigest()
                _supports_nonfinite = False
            _PolarsGeometryMathBackend._physical_spec = PhysicalImplementationSpec(
                canonical=_canon, backend="polars",
                execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
                materializes_full_panel=True, requires_sorted=True,
                supports_nulls=(_supports_nonfinite or bool(getattr(getattr(pandas_op, "_physical_spec", None), "supports_nulls", False))),
                supports_nan=(_supports_nonfinite or bool(getattr(getattr(pandas_op, "_physical_spec", None), "supports_nan", False))),
                supports_inf=(_supports_nonfinite or bool(getattr(getattr(pandas_op, "_physical_spec", None), "supports_inf", False))),
                implementation_source_hash=_source_hash,
                emitter_identity="polars_geometry_math:polars-pandas-polars:v2",
                kernel_identity="pandas_reference:" + _canon,
                parameter_domain_hash=_canon + ":reference-contract",
                semantic_contract_hash=_canon + ":reference-delegation:preserve-axes",
                notes="Pandas delegation with scalar-aware binding; not native Polars acceleration.",
            )

        OperatorRegistry.register(
            _PolarsGeometryMathBackend(),
            canonical=_canon,
            backend="polars",
            source="polars_geometry_math",
            status="implemented",
            backend_explicit=True,
        )


_register_polars_backends()
