# -*- coding: utf-8 -*-
"""Polars backends for the 2026-08 geometry/math expansion.

The pandas_numpy reference is the certified production backend for every op in
this expansion.  These slots give the polars runtimes a surviving ``polars``
backend (polars I/O around the same numpy kernels), matching the convention used
by ``polars_dynamics`` for the state-dynamics pack: registered under
``source="polars_geometry_math"`` (no "bridge" marker), so
``_remove_declared_bridges`` keeps them after production hardening.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from cleaned_operators.registry import OperatorRegistry

try:
    import polars as pl  # noqa: F401
except Exception:  # pragma: no cover - optional backend
    pl = None

_SKIP_PANEL = frozenset({"date", "stock_code"})

# All 79 canonicals of the 2026-08 geometry/math expansion.
_GEOMETRY_MATH_CANONICALS: tuple[str, ...] = (
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
    from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
    from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator

    for _canon in _GEOMETRY_MATH_CANONICALS:
        # Skip ops whose module did not load (e.g. optional import failure).
        pandas_op = OperatorRegistry.get(_canon, "pandas_numpy")
        if pandas_op is None:
            continue

        class _PolarsGeometryMathBackend(PolarsSeriesOperator):
            # ``_canon`` is bound per-iteration via the default arg (loop-var
            # closure over ``canonical`` would resolve every op to the last one).
            metadata = PolarsMetadata(
                name=_canon, category="geometry_math_polars", param_names=[]
            )

            def _calculate_series(self, *frames, _canon=_canon, **params):
                op = OperatorRegistry.get(_canon, "pandas_numpy")
                pdfs = [_pl_to_pd(f) for f in frames]
                out = op.calculate(*pdfs, **params)
                return _pl_rebuild(frames[0], out)

        OperatorRegistry.register(
            _PolarsGeometryMathBackend(),
            canonical=_canon,
            backend="polars",
            source="polars_geometry_math",
            status="implemented",
            backend_explicit=True,
        )


_register_polars_backends()
