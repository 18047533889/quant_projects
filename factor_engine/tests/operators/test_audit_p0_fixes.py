# -*- coding: utf-8 -*-
"""Regression tests for the 2026-08 repo-level audit P0 fixes.

Each test pins one audited defect:
- panel_or_scalar cell-wise broadcast (P0-001)
- run_efficiency / run_concentration NaN breaks the episode (P0-003/004)
- survival NaN is UNKNOWN, not a state exit (P0-005)
- recovery_fraction / drawdown_area never re-use a stale current value and
  never compress NaN time gaps (P0-006/007)
- variance_ratio_slope does not bridge suspensions into adjacent returns (P0-008)
- cs_local_density / cs_neighbor_gap degenerate cross-sections -> NaN, never
  a huge ``gap/eps`` alpha (P0-009/010)
- cs_rank_churn / cs_tail_retention cohort-consistent group samples (P0-011/012)
- turnover survival: missing turnover is UNKNOWN, weighted quantiles ignore
  zero-weight / missing prices (P0-033/035)
- ts_quantile_* now minimise the pinball loss, not the expectile (P0-036)
- ts_ar_fitted_value / ts_ar_in_sample_resid honest names (P0-037)
- HAR forecast-error timing capped at t-1 (P0-039)
- Kalman / state-space inputs are aligned before recursion (P0-002/041)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _dates(n: int):
    return pd.bdate_range("2024-01-02", periods=n)


def _get(canonical):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, canonical
    return op


# ---------------------------------------------------------------------------
# P0-001: panel_or_scalar cell-wise broadcast
# ---------------------------------------------------------------------------

def test_state_slew_limit_panel_limit_is_per_stock():
    dates = _dates(4)
    cols = ["A", "B", "C"]
    x = pd.DataFrame([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                     index=dates, columns=cols)
    limit = pd.DataFrame([[0.01, 0.10, 1.00]] * 4, index=dates, columns=cols)
    out = _get("state_slew_limit").calculate(x, limit=limit)
    # row 0 initialises at target 0; row 1 each stock moves by its own limit.
    assert out["A"].iloc[1] == pytest.approx(0.01)
    assert out["B"].iloc[1] == pytest.approx(0.10)
    assert out["C"].iloc[1] == pytest.approx(1.00)


def test_state_deadband_panel_band_is_per_stock():
    dates = _dates(3)
    cols = ["A", "B"]
    x = pd.DataFrame([[0.0, 0.0], [0.5, 0.5], [0.5, 0.5]], index=dates, columns=cols)
    band = pd.DataFrame([[0.3, 0.4]] * 3, index=dates, columns=cols)
    out = _get("state_deadband").calculate(x, band=band)
    # A: d=0.5 > 0.3 -> move 0.2 ; B: d=0.5 > 0.4 -> move 0.1
    assert out["A"].iloc[1] == pytest.approx(0.2)
    assert out["B"].iloc[1] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# P0-003 / P0-004: run efficiency / concentration NaN breaks the episode
# ---------------------------------------------------------------------------

def test_run_efficiency_nan_breaks_episode():
    dates = _dates(6)
    x = pd.DataFrame([1.0, 1.0, np.nan, 1.0, 1.0, 1.0], index=dates, columns=["A"])
    state = pd.DataFrame([1.0] * 6, index=dates, columns=["A"])
    out = _get("ts_run_efficiency").calculate(x, state, max_run=20, min_periods=2)
    v = out["A"].tolist()
    assert v[1] == pytest.approx(1.0)      # run [1,1]
    assert np.isnan(v[2])                  # NaN driver -> break + NaN
    assert np.isnan(v[3])                  # restarted run, length 1 < mp
    assert v[5] == pytest.approx(1.0)      # run [1,1,1]


def test_run_concentration_nan_breaks_episode():
    dates = _dates(6)
    x = pd.DataFrame([1.0, 2.0, np.nan, 1.0, 1.0, 1.0], index=dates, columns=["A"])
    state = pd.DataFrame([1.0] * 6, index=dates, columns=["A"])
    out = _get("ts_run_concentration").calculate(x, state, max_run=20, min_periods=2)
    v = out["A"].tolist()
    assert v[1] == pytest.approx(2.0 / 3.0)  # run [1,2]
    assert np.isnan(v[2])
    assert np.isnan(v[3])


# ---------------------------------------------------------------------------
# P0-005: survival NaN is UNKNOWN, never a state exit
# ---------------------------------------------------------------------------

def test_survival_nan_does_not_end_episode():
    dates = _dates(12)
    state = pd.DataFrame(0.0, index=dates, columns=["A"])
    state.iloc[0:2] = 1.0      # completed run length 2
    state.iloc[4:6] = 1.0      # completed run length 2
    state.iloc[[8, 10, 11]] = 1.0   # a run split by a NaN gap at row 9
    state.iloc[9] = np.nan
    out = _get("ts_state_age_percentile").calculate(
        state, history_window=60, min_completed_runs=2)
    assert np.isnan(out["A"].iloc[9])       # UNKNOWN -> NaN, not 0
    assert out["A"].iloc[10] == pytest.approx(0.0)  # re-based age 1 in {2,2}
    assert out["A"].iloc[11] == pytest.approx(1.0)  # age 2 in {2,2}


# ---------------------------------------------------------------------------
# P0-006 / P0-007: recovery / drawdown current-NaN and no time compression
# ---------------------------------------------------------------------------

def test_recovery_fraction_current_nan_is_nan():
    dates = _dates(8)
    close = pd.DataFrame([100, 90, 80, 90, 100, 110, np.nan, 120.0],
                         index=dates, columns=["A"])
    out = _get("ts_recovery_fraction").calculate(close, window=10)
    assert np.isnan(out["A"].iloc[6])   # never a stale last-valid fraction


def test_drawdown_area_nan_gap_not_bridged():
    dates = _dates(5)
    a = pd.DataFrame([100, 90, 80, 70, 60], index=dates, columns=["A"])
    b = pd.DataFrame([100, 90, np.nan, np.nan, 60], index=dates, columns=["A"])
    out_a = _get("ts_current_drawdown_area").calculate(a, window=10)["A"].iloc[-1]
    out_b = _get("ts_current_drawdown_area").calculate(b, window=10)["A"].iloc[-1]
    assert np.isfinite(out_a)          # contiguous block is real
    assert np.isnan(out_b)             # gap breaks the block, no bridging


# ---------------------------------------------------------------------------
# P0-008: variance-ratio slope does not bridge gaps
# ---------------------------------------------------------------------------

def test_variance_ratio_slope_gap_not_bridged():
    dates = _dates(6)
    a = pd.DataFrame([100, 101, 103, 106, 110, 115], index=dates, columns=["A"])
    b = pd.DataFrame([100, 101, np.nan, np.nan, 104, 105], index=dates, columns=["A"])
    out_a = _get("ts_variance_ratio_slope").calculate(a, window=10, max_q=4, min_periods=3)["A"].iloc[-1]
    out_b = _get("ts_variance_ratio_slope").calculate(b, window=10, max_q=4, min_periods=3)["A"].iloc[-1]
    assert np.isfinite(out_a)
    assert np.isnan(out_b)   # contiguous block [104,105] too short


# ---------------------------------------------------------------------------
# P0-009 / P0-010: CS degeneracy -> NaN, not a huge value
# ---------------------------------------------------------------------------

def _finite_maxabs(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return float(finite.max()) if finite.size else 0.0


def test_cs_degenerate_cross_section_is_nan():
    dates = _dates(3)
    cols = ["A", "B", "C", "D"]
    equal = pd.DataFrame(np.tile([1.0, 1.0, 1.0, 1.0], (3, 1)), index=dates, columns=cols)
    out = _get("cs_local_density").calculate(equal, k=1)
    assert np.isnan(out.iloc[-1].to_numpy()).all()   # not ~2k/eps
    assert _finite_maxabs(out.to_numpy() < 1e6

    outlier = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]] * 2 + [[1.0, 1.0, 1.0, 2.0]],
                           index=dates, columns=cols)
    gap = _get("cs_neighbor_gap").calculate(outlier, k=1)
    assert np.isnan(gap.iloc[-1].to_numpy()).all()   # MAD~0 -> NaN, not gap/eps
    assert _finite_maxabs(gap.to_numpy() < 1e6


# ---------------------------------------------------------------------------
# P0-011: group-vintage cohort-consistent rank churn
# ---------------------------------------------------------------------------

def test_cs_rank_churn_group_vintage_excludes_rotated_stock():
    dates = _dates(3)
    cols = ["A", "B", "C"]
    x = pd.DataFrame([[0.1, 0.5, 0.4], [0.2, 0.6, 0.5], [0.3, 0.7, 0.6]],
                     index=dates, columns=cols)
    group = pd.DataFrame([["g0", "g0", "g1"], ["g0", "g0", "g1"], ["g0", "g1", "g1"]],
                         index=dates, columns=cols)
    out = _get("cs_rank_churn").calculate(x, lag=1, group=group)
    # B rotates g0 -> g1 at row 2: it must not enter either group's matched sample.
    assert np.isnan(out["B"].iloc[2])
    # A (stable in g0) and C (stable in g1) still get a churn value.
    assert np.isfinite(out["A"].iloc[2])
    assert np.isfinite(out["C"].iloc[2])


# ---------------------------------------------------------------------------
# P0-033 / P0-035: turnover survival missing / weighted-quantile
# ---------------------------------------------------------------------------

def test_turnover_missing_turnover_is_unknown_not_zero():
    dates = _dates(60)
    close = pd.DataFrame(np.linspace(10.0, 20.0, 60), index=dates, columns=["A"])
    # R3-144: old-mass fail threshold tightened 0.30 -> 0.10, so use a daily
    # turnover high enough that (1-t)^window stays under 0.10 ((0.85)^30≈0.008)
    # and the reference price is well-defined.
    turnover = pd.DataFrame(0.15, index=dates, columns=["A"])
    turnover.iloc[20, 0] = np.nan
    out = _get("ts_turnover_reference_price").calculate(close, turnover, window=30)
    # Rows whose trailing window contains the missing turnover are UNKNOWN -> NaN.
    assert np.isnan(out["A"].iloc[40])
    # Rows before the gap are still computed.
    assert np.isfinite(out["A"].iloc[18])


def test_turnover_qdist_ignores_missing_prices():
    dates = _dates(60)
    close = pd.DataFrame(np.linspace(10.0, 20.0, 60), index=dates, columns=["A"])
    # R3-144: old-mass threshold 0.30 -> 0.10; turnover 0.15/day over a 40-day
    # window keeps old_mass ≈ (0.85)^40 ≈ 0.002 below the gate.
    turnover = pd.DataFrame(0.15, index=dates, columns=["A"])
    close.iloc[30, 0] = np.nan   # missing price at a lag
    turnover.iloc[30, 0] = 0.0   # R3-143: a genuine 0-turnover day is a
                                 # zero-weight row; a KNOWN positive turnover
                                 # with a missing price is data-inconsistent
                                 # (chips traded at unknown cost) and would
                                 # fail the whole day closed.
    out = _get("ts_turnover_cost_quantile_distance").calculate(close, turnover, window=40)
    vals = out["A"].iloc[50:].to_numpy()
    # The weighted quantile must not be polluted into NaN by the zero-weight row.
    assert np.all(np.isfinite(vals)), f"qdist not finite: {vals}"


# ---------------------------------------------------------------------------
# P0-036: pinball quantile regression (golden LP reference)
# ---------------------------------------------------------------------------

def test_pinball_quantile_matches_lp_reference():
    rng = np.random.default_rng(5)
    n = 120
    xv = rng.normal(size=n)
    yv = 1.5 * xv + rng.standard_t(3, size=n) * 0.5
    x = pd.DataFrame(xv, index=_dates(n), columns=["A"])
    y = pd.DataFrame(yv, index=_dates(n), columns=["A"])
    out = _get("ts_quantile_regression_coeff").calculate(
        y, x, window=100, q=0.9, min_periods=10)["A"].iloc[-1]
    from scipy.optimize import linprog
    X = np.column_stack([np.ones(100), xv[-100:]])
    yy = yv[-100:]
    c = np.concatenate([np.zeros(2), 0.9 * np.ones(100), 0.1 * np.ones(100)])
    a_eq = np.column_stack([X, np.eye(100), -np.eye(100)])
    res = linprog(c, A_eq=a_eq, b_eq=yy, bounds=[(None, None)] * 2 + [(0, None)] * 200,
                  method="highs")
    assert res.success
    assert out == pytest.approx(float(res.x[1]), abs=1e-6)


def test_pinball_quantile_differs_from_expectile_on_asymmetric_noise():
    rng = np.random.default_rng(11)
    n = 300
    xv = rng.normal(size=n)
    # X-dependent heavy right tail: 20% of points get a 4|x| positive error,
    # the rest a -0.5|x| error.  The expectile (asymmetric *squared* loss) is
    # pulled hard by the 4|x| outliers; the pinball quantile (linear loss) is
    # far more robust, so the slopes must diverge.
    noise = np.where(rng.random(size=n) < 0.2, 4.0 * np.abs(xv), -0.5 * np.abs(xv))
    yv = 1.0 * xv + noise
    x = pd.DataFrame(xv, index=_dates(n), columns=["A"])
    y = pd.DataFrame(yv, index=_dates(n), columns=["A"])
    q = _get("ts_quantile_regression_coeff").calculate(
        y, x, window=250, q=0.9, min_periods=20)["A"].iloc[-1]
    e = _get("ts_expectile_regression_coeff").calculate(
        y, x, window=250, q=0.9, min_periods=20)["A"].iloc[-1]
    assert abs(q - e) > 0.05, f"expectile={e} quantile={q} not distinct"


# ---------------------------------------------------------------------------
# P0-037: honest AR fitted / in-sample names
# ---------------------------------------------------------------------------

def test_ar_honest_names_match_legacy_kernels():
    rng = np.random.default_rng(2)
    n = 120
    x = pd.DataFrame(np.cumsum(rng.normal(size=n)), index=_dates(n), columns=["A"])
    fitted = _get("ts_ar_fitted_value").calculate(x, window=40, order=1)
    legacy_f = _get("ts_ar_forecast").calculate(x, window=40, order=1)
    assert np.allclose(fitted.fillna(0.0).to_numpy(), legacy_f.fillna(0.0).to_numpy(),
                       equal_nan=True)
    resid = _get("ts_ar_in_sample_resid").calculate(x, window=40, order=1)
    legacy_i = _get("ts_ar_innovation").calculate(x, window=40, order=1)
    assert np.allclose(resid.fillna(0.0).to_numpy(), legacy_i.fillna(0.0).to_numpy(),
                       equal_nan=True)


# ---------------------------------------------------------------------------
# P0-039: HAR forecast-error timing capped at t-1
# ---------------------------------------------------------------------------

def test_har_forecast_error_z_is_current_surprise():
    rng = np.random.default_rng(4)
    n = 120
    rv = 0.0004 * (1.0 + 0.01 * np.sin(np.arange(n)))
    rv = rv + rng.normal(0.0, 5e-5, n)  # realistic noise keeps sd > 0
    rv[-1] = 0.05  # extreme current RV
    rv_panel = pd.DataFrame(rv, index=_dates(n), columns=["A"])
    out = _get("ts_har_rv_forecast_error_z").calculate(rv_panel, window=100)["A"]
    last = out.iloc[-1]
    assert np.isfinite(last)
    # A ~100x current-RV spike is a large positive current surprise (forecast
    # built strictly from t-1 information, so the spike itself is not learned).
    assert last > 3.0


# ---------------------------------------------------------------------------
# P0-040: GARCH shock determinism / finiteness with prior fit
# ---------------------------------------------------------------------------

def test_garch_shock_deterministic_and_finite():
    rng = np.random.default_rng(6)
    n = 300
    ret = pd.DataFrame(rng.standard_normal((n, 1)) * 0.02, index=_dates(n), columns=["A"])
    op = _get("ts_garch_standardized_shock")
    a = op.calculate(ret, window=150)["A"]
    b = op.calculate(ret, window=150)["A"]
    assert np.allclose(a.fillna(0.0).to_numpy(), b.fillna(0.0).to_numpy(), equal_nan=True)
    assert a.dropna().size > 0
    assert np.isfinite(a.dropna().to_numpy()).all()


# ---------------------------------------------------------------------------
# P0-002 / P0-041: state-space inputs aligned before recursion
# ---------------------------------------------------------------------------

def test_kalman_beta_misaligned_columns_fail_fast():
    """Model inputs must not silently mispair: reordered columns are rejected
    by the base panel-axes validation before any recursion runs."""
    rng = np.random.default_rng(7)
    n = 80
    dates = _dates(n)
    y = pd.DataFrame(rng.normal(size=(n, 2)), index=dates, columns=["A", "B"])
    x = pd.DataFrame(rng.normal(size=(n, 2)), index=dates, columns=["A", "B"])
    with pytest.raises(ValueError):
        _get("ts_kalman_beta").calculate(y, x[["B", "A"]], q=0.001, r=1.0)
    # A correctly-aligned panel is deterministic.
    base = _get("ts_kalman_beta").calculate(y, x, q=0.001, r=1.0)
    again = _get("ts_kalman_beta").calculate(y, x, q=0.001, r=1.0)
    assert np.allclose(base.fillna(0.0).to_numpy(), again.fillna(0.0).to_numpy(),
                       equal_nan=True)
