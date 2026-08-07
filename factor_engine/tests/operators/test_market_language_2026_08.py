# -*- coding: utf-8 -*-
"""Market-state description language operators (2026-08-08, §18) tests.

Covers the quantile-hit / extreme-dependence / expectile / directional-change /
feature-geometry / conditional-dependence / spread-estimator / local-cross-
section / systemic-tail / marked-event / update-clock families.  Golden values
are hand-computed; invariance and PIT properties are checked alongside; the
polars bridge must reproduce the pandas_numpy reference exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all

load_all()

from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.registry import OperatorRegistry


def _frame(values: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=values.shape[0], freq="B")
    return pd.DataFrame(values, index=idx, columns=[f"S{i}" for i in range(values.shape[1])])


def _col(data: list[float]) -> pd.DataFrame:
    return _frame(np.asarray(data, dtype=float).reshape(-1, 1))


def _ones(n: int) -> pd.DataFrame:
    return _frame(np.ones((n, 1)))


# ---------------------------------------------------------------------------
# Registration / surface classification
# ---------------------------------------------------------------------------
def test_market_language_ops_registered_and_classified():
    from cleaned_operators.operator_surface import (
        DAILY_CANONICALS,
        EXTENDED_ONLY_CANONICALS,
        RESEARCH_ONLY_CANONICALS,
    )

    ALL_OPS = DAILY_CANONICALS | EXTENDED_ONLY_CANONICALS | RESEARCH_ONLY_CANONICALS

    p1 = {
        "ts_quantilogram", "ts_cross_quantilogram", "ts_extremogram",
        "ts_cross_extremogram", "ts_expectile", "ts_expectile_beta",
        "ts_dc_overshoot_ratio", "ts_dc_event_rate", "ts_dc_duration_asymmetry",
        "ts_dc_overshoot_asymmetry", "ts_feature_mode_share",
        "ts_feature_effective_rank", "ts_feature_subspace_rotation",
        "ts_beta_break_score", "intraday_subsampled_rv_dispersion",
        "intraday_volatility_signature_slope", "intraday_profile_surprise_energy",
        "ohlc_corwin_schultz_spread", "cs_knn_local_linear_residual",
        "group_tail_centrality", "event_mark_autocorr",
        "event_interval_mark_coupling", "update_path_efficiency",
        "update_acceleration", "update_surprise", "update_direction_persistence",
    }
    p2 = {
        "ts_quantile_crossing_spectral_concentration",
        "ts_extremal_dependence_decay", "ts_conditional_transfer_entropy",
        "ts_modwt_band_corr", "intraday_realized_power_variation",
        "intraday_profile_phase_shift", "cs_knn_tangent_residual",
        "cs_knn_local_gradient_norm", "cs_rank_copula_mi", "cs_rank_copula_entropy",
        "group_tail_lead_score", "relation_diffusion_score", "ts_roll_effective_spread",
    }
    all_new = p1 | p2
    assert all_new <= ALL_OPS
    for c in p1:
        assert classify_canonical(c) in {"daily", "extended"}, c
    for c in p2:
        assert classify_canonical(c) == "research", c
    # polars bridge present for every new op (survives the overhaul cleanup).
    for c in all_new:
        assert "polars" in OperatorRegistry.backends_for(c), c


# ---------------------------------------------------------------------------
# Quantilogram / extremogram golden
# ---------------------------------------------------------------------------
def test_quantilogram_step_reference():
    x = _col([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = OperatorRegistry.get("ts_quantilogram", "pandas_numpy").calculate(
        x, window=6, quantile=0.5, lag=1, side="lower"
    )
    # H = [0.5,0.5,0.5,-0.5,-0.5,-0.5]; corr(H[1:], H[:-1]) = 0.16/0.24 = 2/3.
    assert out.iloc[-1, 0] == pytest.approx(2.0 / 3.0)


def test_cross_quantilogram_self_equals_quantilogram():
    x = _col([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    q = OperatorRegistry.get("ts_quantilogram", "pandas_numpy").calculate(
        x, window=6, quantile=0.5, lag=1, side="lower"
    )
    cq = OperatorRegistry.get("ts_cross_quantilogram", "pandas_numpy").calculate(
        x, x, window=6, target_q=0.5, source_q=0.5, lag=1,
        target_side="lower", source_side="lower",
    )
    assert np.allclose(q.to_numpy(dtype=float), cq.to_numpy(dtype=float), equal_nan=True)


def test_extremogram_alternating_excess():
    x = _col([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    out = OperatorRegistry.get("ts_extremogram", "pandas_numpy").calculate(
        x, window=10, quantile=0.5, lag=1, side="lower"
    )
    # E = [1,0,1,0,...]; P(E_{t+1}|E_t)=0, P(E)=0.5 -> excess = -0.5.
    assert out.iloc[-1, 0] == pytest.approx(-0.5)


def test_cross_extremogram_self_equals_extremogram():
    x = _col([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    e = OperatorRegistry.get("ts_extremogram", "pandas_numpy").calculate(
        x, window=10, quantile=0.5, lag=1, side="lower"
    )
    ce = OperatorRegistry.get("ts_cross_extremogram", "pandas_numpy").calculate(
        x, x, window=10, target_q=0.5, source_q=0.5, lag=1,
        target_side="lower", source_side="lower",
    )
    assert np.allclose(e.to_numpy(dtype=float), ce.to_numpy(dtype=float), equal_nan=True)


def test_quantile_crossing_spectral_concentration_periodic_high():
    # Perfectly alternating hits -> power concentrated at Nyquist.
    x = _col([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    out = OperatorRegistry.get(
        "ts_quantile_crossing_spectral_concentration", "pandas_numpy"
    ).calculate(x, window=12, quantile=0.5, side="lower")
    assert out.iloc[-1, 0] > 0.5


def test_extremal_dependence_decay_clustered_positive():
    # Blocks of six low then six high -> extremes cluster -> decaying positive
    # excess at h=1 (0.333) and h=2 (0.167) -> finite positive decay time.
    x = _col([1.0] * 6 + [2.0] * 6 + [1.0] * 6 + [2.0] * 6)
    out = OperatorRegistry.get("ts_extremal_dependence_decay", "pandas_numpy").calculate(
        x, window=24, quantile=0.5, side="lower", max_lag=3
    )
    val = out.iloc[-1, 0]
    assert np.isfinite(val) and val > 0.0


# ---------------------------------------------------------------------------
# Expectile golden
# ---------------------------------------------------------------------------
def test_expectile_tau_half_is_mean():
    x = _col([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = OperatorRegistry.get("ts_expectile", "pandas_numpy").calculate(x, window=6, tau=0.5)
    assert out.iloc[-1, 0] == pytest.approx(3.5)


def test_expectile_beta_recovers_line():
    x = _col([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = _col([3.0, 5.0, 7.0, 9.0, 11.0, 13.0])  # y = 2x + 1
    out = OperatorRegistry.get("ts_expectile_beta", "pandas_numpy").calculate(
        y, x, window=6, tau=0.5
    )
    assert out.iloc[-1, 0] == pytest.approx(2.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Directional-change golden
# ---------------------------------------------------------------------------
_DC_X = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0]


def test_dc_overshoot_ratio_golden():
    x = _col(_DC_X)
    scale = _ones(9)
    out = OperatorRegistry.get("ts_dc_overshoot_ratio", "pandas_numpy").calculate(
        x, scale, threshold=1.0, window=10
    )
    # events at 4 (down) and 7 (up); last completed leg down@4 overshoots 2.0.
    assert out.iloc[-1, 0] == pytest.approx(2.0)


def test_dc_event_rate_golden():
    x = _col(_DC_X)
    scale = _ones(9)
    out = OperatorRegistry.get("ts_dc_event_rate", "pandas_numpy").calculate(
        x, scale, threshold=1.0, window=10
    )
    # 2 events over 9 finite bars.
    assert out.iloc[-1, 0] == pytest.approx(2.0 / 9.0)


def test_dc_asymmetry_golden():
    x = _col([0.0, 2.0, 4.0, 3.0, 2.0, 1.0, 3.0, 5.0, 4.0, 3.0])
    scale = _ones(10)
    d = OperatorRegistry.get("ts_dc_duration_asymmetry", "pandas_numpy").calculate(
        x, scale, threshold=1.0, window=12
    )
    # durations: down@3 -> 3 bars, up@6 -> 2 bars -> asym = (2-3)/(2+3) = -0.2.
    assert d.iloc[-1, 0] == pytest.approx(-0.2)
    o = OperatorRegistry.get("ts_dc_overshoot_asymmetry", "pandas_numpy").calculate(
        x, scale, threshold=1.0, window=12
    )
    # overshoots: down@3 -> 2, up@6 -> 2 -> asym = 0.
    assert o.iloc[-1, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Feature geometry
# ---------------------------------------------------------------------------
def test_feature_mode_share_identical_is_one():
    rng = np.random.default_rng(7)
    f = _frame(np.abs(rng.standard_normal((60, 3))) + 0.1)
    out = OperatorRegistry.get("ts_feature_mode_share", "pandas_numpy").calculate(
        f, f, f, window=60
    )
    assert out.iloc[-1, 0] == pytest.approx(1.0, abs=1e-6)


def test_feature_effective_rank_identical_is_one():
    rng = np.random.default_rng(8)
    f = _frame(np.abs(rng.standard_normal((60, 3))) + 0.1)
    out = OperatorRegistry.get("ts_feature_effective_rank", "pandas_numpy").calculate(
        f, f, f, window=60
    )
    assert out.iloc[-1, 0] == pytest.approx(1.0, abs=1e-6)


def test_feature_subspace_rotation_identical_is_zero():
    rng = np.random.default_rng(9)
    f = _frame(np.abs(rng.standard_normal((120, 3))) + 0.1)
    out = OperatorRegistry.get("ts_feature_subspace_rotation", "pandas_numpy").calculate(
        f, f, f, window=120, recent_window=30, prior_window=90
    )
    assert out.iloc[-1, 0] == pytest.approx(0.0, abs=1e-6)


def test_beta_break_stable_line_is_zero():
    x = _col([float(i) for i in range(1, 121)])
    y = _col([2.0 * i + 1.0 for i in range(1, 121)])
    out = OperatorRegistry.get("ts_beta_break_score", "pandas_numpy").calculate(
        y, x, window=120, recent_window=30, prior_window=90
    )
    assert out.iloc[-1, 0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Conditional dependence / MODWT
# ---------------------------------------------------------------------------
def test_modwt_band_corr_identical_is_one():
    rng = np.random.default_rng(10)
    x = _col(rng.standard_normal(120))
    out = OperatorRegistry.get("ts_modwt_band_corr", "pandas_numpy").calculate(
        x, x, window=120, level=3, band=1
    )
    assert out.iloc[-1, 0] == pytest.approx(1.0)


def test_conditional_transfer_entropy_constant_fail_closed():
    x = _col([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = OperatorRegistry.get("ts_conditional_transfer_entropy", "pandas_numpy").calculate(
        x, x, x, window=8, bins=3, lag=1
    )
    assert np.isnan(out.iloc[-1, 0])


# ---------------------------------------------------------------------------
# Spread estimators
# ---------------------------------------------------------------------------
def test_corwin_schultz_two_day_golden():
    high = _col([10.0, 11.0])
    low = _col([9.0, 9.5])
    out = OperatorRegistry.get("ohlc_corwin_schultz_spread", "pandas_numpy").calculate(
        high, low, smooth_window=1
    )
    # beta=ln(11/9)^2, gamma=ln(10/9)^2+ln(11/9.5)^2 -> S ~ 0.0487.
    assert out.iloc[-1, 0] == pytest.approx(0.0487, rel=1e-2)


def test_roll_spread_geometric_zero():
    x = _col([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    out = OperatorRegistry.get("ts_roll_effective_spread", "pandas_numpy").calculate(
        x, window=5
    )
    # constant log-diffs -> zero covariance -> S = 0.
    assert out.iloc[-1, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Intraday sampling-scale / profile
# ---------------------------------------------------------------------------
def _minute_panel(n_days: int = 5, bars: int = 12, base: float = 0.01, jitter: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for b in range(bars):
            idx.append(day + pd.Timedelta(minutes=30 + b))
    vals = np.full(len(idx), base, dtype=float)
    if jitter > 0:
        vals = vals + rng.normal(0.0, jitter, len(idx))
    return pd.DataFrame(vals, index=pd.DatetimeIndex(idx), columns=["S0"])


def _shaped_minute_panel(n_days: int = 5, bars: int = 12, amp: float = 0.02) -> pd.DataFrame:
    """Deterministic intraday shape (sine wave) identical every day.

    A sine profile has a unique best alignment at shift 0 (a linear ramp does
    not: correlation is affine-shift invariant, so the phase is ambiguous).
    """
    idx = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for b in range(bars):
            idx.append(day + pd.Timedelta(minutes=30 + b))
    shape = 0.01 + amp * np.sin(2.0 * np.pi * np.arange(bars) / bars)
    vals = np.tile(shape, n_days)
    return pd.DataFrame(vals, index=pd.DatetimeIndex(idx), columns=["S0"])


def test_intraday_subsampled_rv_dispersion_constant_zero():
    p = _minute_panel()
    out = OperatorRegistry.get("intraday_subsampled_rv_dispersion", "pandas_numpy").calculate(
        p, sampling=3
    )
    assert out.iloc[-1, 0] == pytest.approx(0.0, abs=1e-9)


def test_intraday_realized_power_variation_constant():
    p = _minute_panel(bars=10)
    out = OperatorRegistry.get("intraday_realized_power_variation", "pandas_numpy").calculate(
        p, order=2.0, sampling=1
    )
    # 10 bars of 0.01 -> 10 * 1e-4.
    assert out.iloc[-1, 0] == pytest.approx(0.001)


def test_intraday_profile_surprise_energy_identical_zero():
    p = _minute_panel(n_days=5, bars=12)
    out = OperatorRegistry.get("intraday_profile_surprise_energy", "pandas_numpy").calculate(
        p, history_days=3, n_slots=6
    )
    # All days identical -> z=0 -> energy 0.
    assert out.iloc[-1, 0] == pytest.approx(0.0, abs=1e-9)


def test_intraday_profile_phase_shift_identical_zero():
    # Same shaped profile every day -> best phase shift 0.
    p = _shaped_minute_panel(n_days=5, bars=12)
    out = OperatorRegistry.get("intraday_profile_phase_shift", "pandas_numpy").calculate(
        p, history_days=3, max_shift=2, n_slots=6
    )
    assert out.iloc[-1, 0] == pytest.approx(0.0, abs=1e-6)


def test_intraday_volatility_signature_slope_random_walk_flat():
    # Zero-mean random walk minute returns -> RV(interval) flat -> slope ~ 0.
    rng = np.random.default_rng(31)
    idx = []
    for d in range(2):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for b in range(64):
            idx.append(day + pd.Timedelta(minutes=30 + b))
    vals = rng.normal(0.0, 0.01, len(idx))
    p = pd.DataFrame(vals, index=pd.DatetimeIndex(idx), columns=["S0"])
    out = OperatorRegistry.get("intraday_volatility_signature_slope", "pandas_numpy").calculate(
        p, max_interval=16
    )
    assert np.isfinite(out.iloc[-1, 0])
    assert abs(out.iloc[-1, 0]) < 0.5


# ---------------------------------------------------------------------------
# Local cross-section
# ---------------------------------------------------------------------------
def test_cs_knn_local_residual_deterministic():
    # Rank-quantised features can create exact distance ties, broken by stable
    # argsort (the documented convention shared with cs_knn_peer_mean_ex_self),
    # so the contract is determinism, not column-permutation invariance.
    rng = np.random.default_rng(12)
    rows, n = 60, 8
    f1 = _frame(rng.normal(0.0, 1.0, (rows, n)))
    f2 = _frame(rng.normal(0.0, 1.0, (rows, n)))
    f3 = _frame(rng.normal(0.0, 1.0, (rows, n)))
    tgt = _frame(rng.normal(0.0, 1.0, (rows, n)))
    op = OperatorRegistry.get("cs_knn_local_linear_residual", "pandas_numpy")
    a = op.calculate(tgt, f1, f2, f3, k=5, ridge=1e-3)
    b = op.calculate(tgt, f1, f2, f3, k=5, ridge=1e-3)
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_cs_rank_copula_mi_self_stronger_than_independent():
    rng = np.random.default_rng(14)
    rows, n = 10, 80  # >= 2*grid^2 = 72 names per day
    a = _frame(rng.normal(0.0, 1.0, (rows, n)))
    b = _frame(rng.normal(0.0, 1.0, (rows, n)))
    op = OperatorRegistry.get("cs_rank_copula_mi", "pandas_numpy")
    mi_self = op.calculate(a, a, grid=6)
    mi_ind = op.calculate(a, b, grid=6)
    assert mi_self.iloc[-1, 0] > mi_ind.iloc[-1, 0]


def test_cs_rank_copula_entropy_self_lower_than_independent():
    rng = np.random.default_rng(15)
    rows, n = 10, 80
    a = _frame(rng.normal(0.0, 1.0, (rows, n)))
    b = _frame(rng.normal(0.0, 1.0, (rows, n)))
    op = OperatorRegistry.get("cs_rank_copula_entropy", "pandas_numpy")
    ent_self = op.calculate(a, a, grid=6)
    ent_ind = op.calculate(a, b, grid=6)
    assert ent_self.iloc[-1, 0] < ent_ind.iloc[-1, 0]


# ---------------------------------------------------------------------------
# Systemic tail / relation diffusion
# ---------------------------------------------------------------------------
def test_group_tail_centrality_identical_group_is_one():
    rng = np.random.default_rng(16)
    x = _frame(rng.normal(0.0, 1.0, (60, 2)))  # two identical-ish names
    x = x.copy()
    x.iloc[:, 1] = x.iloc[:, 0].values
    g = _frame(np.full((60, 2), "G0", dtype=object))
    out = OperatorRegistry.get("group_tail_centrality", "pandas_numpy").calculate(
        x, g, window=60, quantile=0.5, side="lower"
    )
    assert out.iloc[-1, 0] == pytest.approx(1.0, abs=1e-9)


def test_group_tail_no_peers_fail_closed():
    rng = np.random.default_rng(17)
    x = _frame(rng.normal(0.0, 1.0, (60, 2)))
    g = _frame(np.array([["A", "B"]] * 60, dtype=object))  # each its own group
    out = OperatorRegistry.get("group_tail_centrality", "pandas_numpy").calculate(
        x, g, window=60, quantile=0.5, side="lower"
    )
    assert np.isnan(out.iloc[-1, 0])


def test_relation_diffusion_two_member_golden():
    # Two names in the same group on the same day -> per-day cross-section.
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    x = pd.DataFrame(np.array([[1.0, 3.0], [1.0, 3.0]]), index=idx, columns=["S0", "S1"])
    g = pd.DataFrame(np.array([["G0", "G0"], ["G0", "G0"]]), index=idx, columns=["S0", "S1"])
    out = OperatorRegistry.get("relation_diffusion_score", "pandas_numpy").calculate(
        x, g, alpha=0.5, steps=1
    )
    # P = [[0,1],[1,0]]; D = 0.5 * P @ x = [1.5, 0.5].
    assert out.iloc[-1, 0] == pytest.approx(1.5)
    assert out.iloc[-1, 1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Marked event
# ---------------------------------------------------------------------------
def test_event_mark_autocorr_perfect_linear():
    ev = _col([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    mk = _col([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0])
    out = OperatorRegistry.get("event_mark_autocorr", "pandas_numpy").calculate(
        ev, mk, history_window=10, event_lag=1
    )
    # marks [1,2,3,4,5] -> corr of consecutive = 1.
    assert out.iloc[-1, 0] == pytest.approx(1.0, abs=1e-6)


def test_event_interval_mark_coupling_monotone():
    ev = _col([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    mk = _col([1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 16.0])
    out = OperatorRegistry.get("event_interval_mark_coupling", "pandas_numpy").calculate(
        ev, mk, window=15
    )
    # events at 0,2,5,9,14 -> intervals [2,3,4,5] vs marks [2,4,8,16] -> strong +corr.
    assert out.iloc[-1, 0] > 0.9


# ---------------------------------------------------------------------------
# Update clock
# ---------------------------------------------------------------------------
def test_update_path_efficiency_golden():
    x = _col([10.0, 10.0, 10.0, 12.0, 12.0, 15.0, 15.0])
    ev = _col([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    out = OperatorRegistry.get("update_path_efficiency", "pandas_numpy").calculate(
        x, ev, n_updates=3
    )
    # updates [10,12,15]: 5/(2+3) = 1.0.
    assert out.iloc[-1, 0] == pytest.approx(1.0)


def test_update_acceleration_golden():
    x = _col([10.0, 12.0, 16.0, 21.0, 27.0, 27.0])
    ev = _col([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    out = OperatorRegistry.get("update_acceleration", "pandas_numpy").calculate(
        x, ev, n_updates=5
    )
    # deltas [2,4,5,6]: median-MAD = median(2.5,0.5,0.5,1.5) = 1.0 -> (6-5)/1.0 = 1.0.
    assert out.iloc[-1, 0] == pytest.approx(1.0)


def test_update_surprise_golden():
    x = _col([10.0, 12.0, 16.0, 21.0, 27.0, 27.0])
    ev = _col([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    out = OperatorRegistry.get("update_surprise", "pandas_numpy").calculate(
        x, ev, n_updates=5
    )
    # deltas [2,4,5,6]: (6 - median([2,4,5]))/(1.4826*MAD([2,4,5]))
    # = 2/(1.4826*1) ~ 1.349.
    assert out.iloc[-1, 0] == pytest.approx(2.0 / 1.4826, rel=1e-3)


def test_update_direction_persistence_all_positive_one():
    x = _col([10.0, 12.0, 16.0, 21.0, 27.0, 27.0])
    ev = _col([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    out = OperatorRegistry.get("update_direction_persistence", "pandas_numpy").calculate(
        x, ev, n_updates=5
    )
    assert out.iloc[-1, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PIT / prefix causality
# ---------------------------------------------------------------------------
def test_prefix_causality_quantilogram():
    rng = np.random.default_rng(20)
    x = _frame(rng.normal(0.0, 1.0, (200, 2)))
    op = OperatorRegistry.get("ts_quantilogram", "pandas_numpy")
    full = op.calculate(x, window=60, quantile=0.1, lag=1, side="lower")
    prefix = op.calculate(x.iloc[:160], window=60, quantile=0.1, lag=1, side="lower")
    assert np.allclose(
        full.to_numpy(dtype=float)[:160],
        prefix.to_numpy(dtype=float),
        equal_nan=True,
        rtol=1e-6,
        atol=1e-8,
    )


def test_prefix_causality_dc_overshoot():
    rng = np.random.default_rng(21)
    x = _frame(np.cumsum(rng.normal(0.0, 0.1, (200, 1)), axis=0))
    scale = _frame(np.full((200, 1), np.std(np.diff(x.to_numpy(dtype=float).ravel())) + 0.1))
    op = OperatorRegistry.get("ts_dc_overshoot_ratio", "pandas_numpy")
    full = op.calculate(x, scale, threshold=1.0, window=60)
    prefix = op.calculate(x.iloc[:160], scale.iloc[:160], threshold=1.0, window=60)
    assert np.allclose(
        full.to_numpy(dtype=float)[:160],
        prefix.to_numpy(dtype=float),
        equal_nan=True,
        rtol=1e-6,
        atol=1e-8,
    )


def test_all_nan_input_robustness():
    x = _frame(np.full((30, 2), np.nan))
    for canon, kwargs in [
        ("ts_quantilogram", {"window": 20, "quantile": 0.1, "lag": 1, "side": "lower"}),
        ("ts_expectile", {"window": 20, "tau": 0.1}),
        ("ts_extremogram", {"window": 20, "quantile": 0.1, "lag": 1, "side": "lower"}),
        ("ts_feature_mode_share", {"window": 20}),
        ("ohlc_corwin_schultz_spread", {"smooth_window": 3}),
    ]:
        op = OperatorRegistry.get(canon, "pandas_numpy")
        args = (x,) if canon in {"ts_quantilogram", "ts_expectile", "ts_extremogram"} else (x, x, x)
        if canon == "ohlc_corwin_schultz_spread":
            args = (x, x)
        if canon == "ts_feature_mode_share":
            args = (x, x, x)
        out = op.calculate(*args, **kwargs)
        assert np.isnan(out.to_numpy(dtype=float)).all(), canon


# ---------------------------------------------------------------------------
# Polars bridge parity
# ---------------------------------------------------------------------------
def test_polars_bridge_parity_quantilogram():
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(22)
    n = 120
    np_x = rng.normal(0.0, 1.0, (n, 2))
    x = _frame(np_x)
    np_out = OperatorRegistry.get("ts_quantilogram", "pandas_numpy").calculate(
        x, window=60, quantile=0.1, lag=1, side="lower"
    )
    pl_x = pl.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=n, freq="B"), "S0": np_x[:, 0], "S1": np_x[:, 1]}
    )
    pl_out = OperatorRegistry.get("ts_quantilogram", "polars").calculate(
        pl_x, window=60, quantile=0.1, lag=1, side="lower"
    )
    for c in ("S0", "S1"):
        assert np.allclose(
            np.asarray(pl_out[c].to_list(), dtype=float),
            np_out[c].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_polars_bridge_parity_expectile():
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(23)
    n = 80
    np_x = rng.normal(0.0, 1.0, (n, 1))
    x = _frame(np_x)
    np_out = OperatorRegistry.get("ts_expectile", "pandas_numpy").calculate(x, window=40, tau=0.25)
    pl_x = pl.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=n, freq="B"), "S0": np_x[:, 0]}
    )
    pl_out = OperatorRegistry.get("ts_expectile", "polars").calculate(pl_x, window=40, tau=0.25)
    assert np.allclose(
        np.asarray(pl_out["S0"].to_list(), dtype=float),
        np_out["S0"].to_numpy(dtype=float),
        equal_nan=True,
    )
