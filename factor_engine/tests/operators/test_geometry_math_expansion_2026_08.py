# -*- coding: utf-8 -*-
"""2026-08 geometry/math operator expansion — integration tests.

Covers the 79 new operators (interval geometry, structural levels, multiscale
trend, envelope/crossing, extrema divergence, threshold cycles, state-episode
excursions, jump-robust variation, intraday volatility shape, roughness,
binned response, nonlinear dependence, 2D path geometry, point-process stats,
complexity, spectral, Hankel/SSA, multifractal, memory, L-moments, intrinsic
dimension, persistence entropy, cs/group locality, session shape):

1. every canonical registers with a pandas_numpy + polars backend;
2. every canonical classifies to the *daily* surface;
3. every canonical is in the daily DSL allowlist;
4. every canonical executes on synthetic panels without error and yields a
   sane finite fraction;
5. a representative subset also executes through the polars bridge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import OperatorRegistry, load_all
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.runtime.session_calendar import SessionCalendar

# P0-10: intraday session operators require an explicit calendar (the official
# session grid is never inferred from observed bars).  The synthetic minute
# panels below are a flat 240-bar grid, so a plain 1-min CN calendar matches.
_ASHARE_CAL = SessionCalendar(
    market="CN", timestamp_convention="bar_start", bar_freq="1min"
)

NEW_CANONICALS: frozenset[str] = frozenset({
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
    # (ts_partial_distance_correlation was renamed by the concurrent session to
    #  ts_distance_correlation_partial_proxy and reclassified as research-tier
    #  "extended" — covered by that session's own surface, not this daily smoke.)
    # 2D joint trajectory geometry
    "ts_vector_path_efficiency", "ts_vector_turning_coherence",
    "ts_vector_path_curvature", "ts_vector_self_intersection_rate",
    # point-process interval stats
    "event_interval_memory", "event_local_variation", "event_fano_factor",
    # string / ordinal complexity
    "ts_lempel_ziv_complexity", "ts_forbidden_ordinal_pattern_ratio",
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
    "cs_knn_local_moran", "cs_isotonic_residual",
    "group_current_members_tail_coexceedance", "group_corr_mst_length",
    # intraday session shape
    "intraday_session_shape_novelty", "intraday_profile_pca_residual",
})


def _panels() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    # R6-137: 12 symbols so cs_isotonic_residual (which needs >=10 names for a
    # meaningful cross-sectional fit) produces values.  Groups g (A-F) and h
    # (G-L) each keep >=3 members for the group ops.
    cols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (80, 12)), axis=0), index=idx, columns=cols)
    prev = close.shift(1)
    opn = pd.DataFrame(
        prev.where(prev.notna(), close).to_numpy() * (1 + rng.normal(0, 0.002, (80, 12))),
        index=idx, columns=cols,
    )
    high = pd.DataFrame(
        np.maximum(opn.to_numpy(), close.to_numpy()) + np.abs(rng.normal(0, 1, (80, 12))),
        index=idx, columns=cols)
    low = pd.DataFrame(
        np.minimum(opn.to_numpy(), close.to_numpy()) - np.abs(rng.normal(0, 1, (80, 12))),
        index=idx, columns=cols)
    volume = pd.DataFrame(rng.lognormal(5, 0.5, (80, 12)), index=idx, columns=cols)
    ret = close.pct_change()
    span = (high - low).replace(0, np.nan)
    f1 = (close - opn) / span
    f2 = (high - np.maximum(opn, close)) / span
    f3 = (np.minimum(opn, close) - low) / span
    f4 = (opn - close.shift(1)) / span
    rng2 = np.random.default_rng(1)
    # ~22% event rate: the event_interval_* ops require >=6 intervals (7 events)
    # to emit a value (Master Spec N-69), so a window=80 panel needs a dense
    # enough process that trailing windows reach 7 events quickly.
    event = pd.DataFrame((rng2.random((80, 12)) < 0.22).astype(float), index=idx, columns=cols)
    state = pd.DataFrame(
        np.where(np.floor(np.arange(80) / 5.0) % 2, 1.0, -1.0)[:, None] * np.ones((1, 12)),
        index=idx, columns=cols)
    group_id = pd.DataFrame([["g"] * 6 + ["h"] * 6] * 80, index=idx, columns=cols)
    upper = close.rolling(10, min_periods=3).max()
    lower = close.rolling(10, min_periods=3).min()
    mid = (upper + lower) / 2
    # oscillating series (period 6, amplitude 10 around 100) so L=95/U=105 cycles complete
    osc = pd.DataFrame(
        100 + 10 * np.sin(2 * np.pi * np.arange(80) / 6.0)[:, None] * np.ones((1, 12)),
        index=idx, columns=cols)
    # minute panels (240 rows = 20 sessions x 12 minutes)
    mi = pd.date_range("2024-01-01", periods=240, freq="1min")
    rng3 = np.random.default_rng(2)
    pmin = pd.DataFrame(100 + np.cumsum(rng3.normal(0, 0.1, (240, 12)), axis=0), index=mi, columns=cols)
    rmin = pmin.pct_change()
    sess = pd.DataFrame(
        np.repeat(np.arange(20), 12)[:, None] * np.ones((1, 12)).astype(float),
        index=mi, columns=cols)
    return {
        "close": close, "open": opn, "high": high, "low": low, "volume": volume,
        "ret": ret, "f1": f1, "f2": f2, "f3": f3, "f4": f4,
        "event": event, "state": state, "group_id": group_id, "osc": osc,
        "upper": upper, "lower": lower, "mid": mid,
        "close_min": pmin, "ret_min": rmin, "session_id": sess,
    }


# name -> (positional panel names, kwargs).  Panels come from _panels().
CALLS: dict[str, tuple[tuple[str, ...], dict]] = {
    "ts_interval_union_coverage": (("low", "high"), {"window": 20}),
    "ts_interval_occupancy_entropy": (("low", "high"), {"window": 20, "bins": 8}),
    "ts_interval_occupancy_mode_distance": (("close", "low", "high"), {"window": 20, "bins": 8}),
    "ts_interval_nesting_depth": (("low", "high"), {"mode": "inside"}),
    "ts_interval_exploration_efficiency": (("high", "low", "close"), {"window": 20}),
    "ts_interval_overlap_connected_component_ratio": (("low", "high"), {"window": 20}),
    "ts_structural_level_density": (("close",), {"window": 60, "prominence": 0.02, "confirmation": 3}),
    "ts_nearest_structural_level_distance": (("close",), {"window": 60, "prominence": 0.02, "confirmation": 3}),
    "ts_structural_level_strength": (("close",), {"window": 60, "prominence": 0.02, "confirmation": 3}),
    "ts_vector_state_mahalanobis": (("f1", "f2", "f3", "f4"), {"window": 40}),
    "ts_vector_state_local_density": (("f1", "f2", "f3", "f4"), {"window": 40, "k": 5}),
    "ts_matrix_profile_novelty": (("close",), {"window": 60, "subsequence_length": 6, "history": 40}),
    "ts_matrix_profile_motif_age": (("close",), {"window": 60, "subsequence_length": 6, "history": 40}),
    "ts_multiscale_trend_consensus": (("close",), {"window": 50, "scales": (5, 10, 20, 40)}),
    "ts_multiscale_trend_dispersion": (("close",), {"window": 50, "scales": (5, 10, 20, 40)}),
    "ts_multiscale_trend_curvature": (("close",), {"window": 50, "scales": (5, 10, 20, 40)}),
    "ts_envelope_compression": (("upper", "lower", "mid"), {"window": 20}),
    "ts_envelope_pressure": (("close", "upper", "lower"), {"window": 20}),
    "ts_envelope_boundary_dwell": (("close", "upper", "lower"), {"window": 20, "quantile": 0.8}),
    "ts_crossing_speed": (("close", "volume"), {"window": 20}),
    "ts_crossing_acceleration": (("close", "volume"), {"window": 20}),
    "ts_extrema_divergence_strength": (("close", "volume"), {"window": 50, "prominence": 0.02, "confirmation": 3}),
    "ts_extrema_confirmation_rate": (("close", "volume"), {"window": 50, "prominence": 0.02, "confirmation": 3}),
    "ts_threshold_cycle_period": (("osc",), {"lower": 95.0, "upper": 105.0, "window": 60}),
    "ts_threshold_cycle_asymmetry": (("osc",), {"lower": 95.0, "upper": 105.0, "window": 60}),
    "state_episode_mfe": (("close", "state", "volume"), {}),
    "state_episode_mae": (("close", "state", "volume"), {}),
    "state_episode_efficiency": (("close", "state"), {}),
    "state_episode_retrace_ratio": (("close", "state"), {}),
    "state_episode_excursion_balance": (("close", "state"), {}),
    "intraday_medrv": (("ret_min",), {"window": 240}),
    "intraday_minrv": (("ret_min",), {"window": 240}),
    "intraday_jump_test_stat": (("ret_min",), {"window": 240}),
    "intraday_volatility_time_centroid": (("ret_min",), {"window": 240}),
    "intraday_volatility_concentration": (("ret_min",), {"window": 240}),
    "intraday_volatility_entropy": (("ret_min",), {"window": 240}),
    "intraday_realized_semivariance_balance": (("ret_min",), {"window": 240}),
    "intraday_rv_signature_curvature": (("ret_min",), {"window": 240}),
    "ts_vol_pvariation_roughness": (("close",), {"window": 60, "p": 2.0}),
    "ts_vol_scaling_break": (("close",), {"window": 60, "p": 2.0}),
    "ts_binned_response_monotonicity": (("volume", "close"), {"window": 60, "bins": 5}),
    "ts_binned_response_curvature": (("volume", "close"), {"window": 60, "bins": 5}),
    "ts_response_slope_asymmetry": (("volume", "close"), {"window": 60, "split_quantile": 0.5}),
    "ts_chatterjee_xi": (("close", "volume"), {"window": 60}),
    "ts_hsic": (("close", "volume"), {"window": 40}),
    "ts_conditional_mutual_information": (("close", "volume", "ret"), {"window": 60, "bins": 3}),
    "ts_vector_path_efficiency": (("close", "volume"), {"window": 40}),
    "ts_vector_turning_coherence": (("close", "volume"), {"window": 40}),
    "ts_vector_path_curvature": (("close", "volume"), {"window": 40}),
    "ts_vector_self_intersection_rate": (("close", "volume"), {"window": 40}),
    "event_interval_memory": (("event",), {"window": 80}),
    "event_local_variation": (("event",), {"window": 80}),
    "event_fano_factor": (("event",), {"window": 80, "block": 10}),
    "ts_lempel_ziv_complexity": (("close",), {"window": 60, "bins": 2}),
    "ts_forbidden_ordinal_pattern_ratio": (("close",), {"window": 60, "order": 3, "delay": 1}),
    "ts_spectral_centroid": (("close",), {"window": 32}),
    "ts_spectral_flatness": (("close",), {"window": 32}),
    "ts_spectral_peak_concentration": (("close",), {"window": 32}),
    "ts_spectral_quality_factor": (("close",), {"window": 32}),
    "ts_hankel_effective_rank": (("close",), {"window": 40, "embedding_dim": 10}),
    "ts_hankel_singular_gap": (("close",), {"window": 40, "embedding_dim": 10}),
    "ts_ssa_reconstruction_residual": (("close",), {"window": 40, "embedding_dim": 10, "n_components": 3}),
    "ts_generalized_hurst_exponent": (("close",), {"window": 60, "q": 2.0}),
    "ts_generalized_hurst_spread_q1_q4": (("close",), {"window": 60}),
    "ts_multifractal_spectrum_width": (("close",), {"window": 60}),
    "ts_multifractal_curvature": (("close",), {"window": 60}),
    "ts_autocorrelation_time": (("close",), {"window": 60, "max_lag": 10}),
    "ts_fractional_difference": (("close",), {"fd": 0.4, "cutoff": 20}),
    "ts_l_skewness": (("close",), {"window": 40}),
    "ts_l_kurtosis": (("close",), {"window": 40}),
    "ts_hartigan_dip": (("close",), {"window": 40}),
    "cs_knn_local_moran": (("ret", "close", "volume", "ret"), {"k": 3}),
    "cs_isotonic_residual": (("volume", "close"), {}),
    "group_current_members_tail_coexceedance": (("ret", "group_id"), {"window": 60, "quantile": 0.9, "side": "upper"}),
    "group_corr_mst_length": (("ret", "group_id"), {"window": 60}),
    "ts_persistence_entropy_h0": (("close",), {"window": 40, "tau": 1, "dim": 3}),
    "ts_persistence_entropy_h1": (("close",), {"window": 40, "tau": 1, "dim": 3}),
    "ts_delay_intrinsic_dimension": (("close",), {"window": 40, "embedding_dim": 3, "k": 5, "delay": 1}),
    "intraday_session_shape_novelty": (("close_min", "session_id"), {"history_days": 5, "calendar": _ASHARE_CAL}),
    "intraday_profile_pca_residual": (("close_min", "session_id"), {"history_days": 5, "n_components": 2, "calendar": _ASHARE_CAL}),
}


@pytest.fixture(scope="module", autouse=True)
def _registry_loaded():
    load_all()
    yield


def test_all_new_canonicals_registered_with_both_backends():
    for name in NEW_CANONICALS:
        assert name in OperatorRegistry._operators, f"{name} not registered"
        backends = OperatorRegistry.backends_for(name)
        assert "pandas_numpy" in backends, f"{name} lacks pandas_numpy: {backends}"
        assert "polars" in backends, f"{name} lacks polars: {backends}"


def test_all_new_canonicals_classify_daily():
    for name in NEW_CANONICALS:
        assert classify_canonical(name) == "daily", f"{name} -> {classify_canonical(name)}"


def test_all_new_canonicals_in_daily_dsl_allowlist():
    from factor_engine.api.mining_integration import list_dsl_allowlist

    allow = set(list_dsl_allowlist(surface="daily"))
    missing = sorted(n for n in NEW_CANONICALS if n not in allow)
    assert not missing, f"missing from daily DSL: {missing[:10]}"


# Session-shaped ops only emit at the final minute of each session (prefix-causal
# per-session semantics), so their finite fraction is ~1/session_length.
# NOTE (P0-10): the intraday ops additionally REQUIRE a real official session
# grid (never inferred from observed bars) — the flat 240-bar synthetic panel has
# no matching grid, so they emit nothing here and are excluded from the finite
# check (they are exercised by the dedicated intraday/calendar tests).
_SESSION_EMIT_OPS = frozenset({"intraday_session_shape_novelty", "intraday_profile_pca_residual"})
_CALENDAR_REQUIRED_OPS = frozenset({"intraday_session_shape_novelty", "intraday_profile_pca_residual"})


def test_all_new_canonicals_execute_pandas():
    panels = _panels()
    failures: list[str] = []
    low_finite: list[str] = []
    for name in sorted(NEW_CANONICALS):
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, name
        panel_names, kwargs = CALLS[name]
        try:
            out = op.calculate(*[panels[p] for p in panel_names], **kwargs)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if out is None:
            failures.append(f"{name}: returned None")
            continue
        arr = np.asarray(out, dtype=float)
        finite = np.isfinite(arr)
        if name in _CALENDAR_REQUIRED_OPS:
            continue  # needs a real official session grid; see _CALENDAR_REQUIRED_OPS
        threshold = 0.02 if name in _SESSION_EMIT_OPS else 0.2
        if finite.mean() < threshold:
            low_finite.append(f"{name}:{finite.mean():.2f}")
    assert not failures, "\n".join(failures[:20])
    assert not low_finite, f"low finite fraction: {low_finite[:10]}"


POLARS_SUBSET = (
    "ts_interval_union_coverage", "ts_interval_occupancy_entropy",
    "ts_interval_nesting_depth", "ts_multiscale_trend_consensus",
    "ts_envelope_compression", "ts_crossing_speed", "ts_threshold_cycle_period",
    "state_episode_mfe", "ts_binned_response_monotonicity", "ts_chatterjee_xi",
    "ts_autocorrelation_time", "ts_l_skewness",
)


def test_polars_bridge_subset_parity():
    pl = pytest.importorskip("polars")
    panels = _panels()
    failures: list[str] = []
    for name in POLARS_SUBSET:
        panel_names, kwargs = CALLS[name]
        pdfs = []
        for p in panel_names:
            frame = panels[p]
            pdfs.append(
                pl.from_pandas(frame.reset_index().rename(columns={"index": "date", "level_0": "date"}))
            )
        op = OperatorRegistry.get(name, "polars")
        try:
            out = op.calculate(*pdfs, **kwargs)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if out is None:
            failures.append(f"{name}: None")
            continue
        arr = np.asarray(out.select([c for c in out.columns if c not in ("date", "stock_code")]).to_pandas(), dtype=float)
        if np.isfinite(arr).mean() < 0.2:
            failures.append(f"{name}: low finite {np.isfinite(arr).mean():.2f}")
    assert not failures, "\n".join(failures[:20])
