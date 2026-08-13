# -*- coding: utf-8 -*-
"""Round-14 expectile / binned-response operator fixes (2026-08).

Fixes under test:
* P0-15  ``ts_expectile`` output unit metadata is ``same_as:x`` (was the
         unresolvable ``same_as:target``).
* P1-16  expectile fixed-point / IRLS emit ONLY a converged iterate; a
         budget-exhausted (non-converged) window emits NaN, never a stale
         last-iterate.
* P1-17  effective-tail sample gate ``N_eff * min(tau, 1-tau) >= n_min``
         (default 3) on ``ts_expectile`` and ``ts_expectile_beta``.
* P1-18  ``ts_binned_response_curvature`` fits against the groups' EMPIRICAL
         percentile centres instead of nominal quantile centres (which are
         wrong when ``x`` has ties / empty / huge groups).
* P1-19  ``ts_response_slope_asymmetry`` uses a tie-balanced rank split so the
         low/high cohorts stay balanced under threshold ties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Register ONLY the operators under test.  A full ``load_all()`` is heavy and
# can be broken mid-edit by concurrent sessions in this workspace; importing
# each module runs its ``_register_surface()`` and registers exactly the
# operators exercised here.
import cleaned_operators.advanced_expectile as _ae  # noqa: F401
import cleaned_operators.binned_response as _br  # noqa: F401

from cleaned_operators.binned_response import (
    _balanced_quantile_split,
    _binned_empirical_medians,
    _binned_medians,
)
from cleaned_operators.registry import OperatorRegistry

_EPS = 1e-12


def _col(data) -> pd.DataFrame:
    data = np.asarray(data, dtype=float).reshape(-1, 1)
    idx = pd.date_range("2024-01-01", periods=data.shape[0], freq="B")
    return pd.DataFrame(data, index=idx, columns=["S0"])


def _frame(data) -> pd.DataFrame:
    data = np.asarray(data, dtype=float)
    idx = pd.date_range("2024-01-01", periods=data.shape[0], freq="B")
    return pd.DataFrame(data, index=idx, columns=[f"S{i}" for i in range(data.shape[1])])


# ---------------------------------------------------------------------------
# P0-15 unit metadata
# ---------------------------------------------------------------------------
def test_expectile_unit_metadata_is_same_as_x():
    op = OperatorRegistry.get("ts_expectile", "pandas_numpy")
    unit_tags = [t for t in op.metadata.tags if t.startswith("unit:")]
    assert "unit:same_as:x" in unit_tags, unit_tags
    assert not any("same_as:target" in t for t in unit_tags), unit_tags
    # the new effective-tail parameter is part of the declared contract
    assert "n_min" in op.metadata.param_names
    spec = op.metadata.param_specs.get("n_min")
    assert spec is not None and spec.dtype is int and spec.min == 1


# ---------------------------------------------------------------------------
# P1-17 effective-tail sample gate
# ---------------------------------------------------------------------------
def test_expectile_extreme_tau_short_window_is_nan():
    rng = np.random.default_rng(0)
    x = _frame(rng.standard_t(2.5, (20, 2)))
    op = OperatorRegistry.get("ts_expectile", "pandas_numpy")
    for tau in (0.01, 0.99):
        arr = op.calculate(x, window=20, tau=tau).to_numpy(dtype=float)
        # 20 * min(tau, 1-tau) = 0.2 < n_min: the extreme-tail estimate is
        # unstable, so EVERY window (partial or full) must emit NaN — never a
        # bogus number.
        assert np.isnan(arr).all(), (tau, arr[np.isfinite(arr)][:5])


def test_expectile_tail_gate_holds_on_large_window():
    rng = np.random.default_rng(1)
    n = 200
    x = _frame(rng.standard_t(3.0, (n, 2)))
    op = OperatorRegistry.get("ts_expectile", "pandas_numpy")
    arr = op.calculate(x, window=n, tau=0.05).to_numpy(dtype=float)
    # 200 * 0.05 = 10 >= n_min=3 -> finite at the full-window tail row.
    assert np.isfinite(arr[-1]).all()
    # expectile at tau=0.05 lies in the lower tail -> below the sample mean.
    mean = x.to_numpy(dtype=float).mean(axis=0)
    assert np.all(arr[-1] < mean)


def test_expectile_high_leverage_returns_mean_at_tau_half():
    # One huge outlier: tau=0.5 expectile == mean exactly (finite and correct,
    # not a divergent/NaN iterate).
    n = 200
    v = np.r_[np.zeros(n - 1), 1e9]
    out = OperatorRegistry.get("ts_expectile", "pandas_numpy").calculate(
        _col(v), window=n, tau=0.5
    )
    assert out.iloc[-1, 0] == pytest.approx(float(v.mean()))


# ---------------------------------------------------------------------------
# P1-16 convergence guard: budget-exhausted window emits NaN, never a stale
# last-iterate.  (Both fixed-point and IRLS converge for finite data in
# practice, so the guard is exercised by exhausting the iteration budget.)
# ---------------------------------------------------------------------------
def test_expectile_non_converged_budget_emits_nan(monkeypatch):
    # 20 * min(0.2, 0.8) = 4 >= n_min: passes the tail gate and converges with
    # the real 200-iteration budget...
    v = np.r_[np.zeros(10), np.ones(10)]
    assert np.isfinite(_ae._expectile(v, 0.2))
    # ...but a 1-iteration budget is NOT converged -> NaN, not the last iterate.
    monkeypatch.setattr(_ae, "_MAX_ITER_EXPECTILE", 1)
    assert np.isnan(_ae._expectile(v, 0.2))


def test_expectile_beta_near_singular_design_is_nan():
    # x is constant to machine precision -> std(x) < EPS -> fail-closed NaN
    # instead of a bogus ~1e12 slope.
    rng = np.random.default_rng(2)
    y = _col(rng.normal(size=(40, 1)))
    x = _col(np.r_[np.ones(39), 1.0 + 1e-12])
    out = OperatorRegistry.get("ts_expectile_beta", "pandas_numpy").calculate(
        y, x, window=40, tau=0.5
    )
    assert np.isnan(out.to_numpy(dtype=float)).all()


def test_expectile_beta_converges_on_well_posed_design():
    n = 40
    xv = np.linspace(1.0, n, n)
    yv = 2.0 * xv + 1.0
    out = OperatorRegistry.get("ts_expectile_beta", "pandas_numpy").calculate(
        _col(yv), _col(xv), window=n, tau=0.5
    )
    assert out.iloc[-1, 0] == pytest.approx(2.0, rel=1e-4)


def test_expectile_beta_non_converged_budget_emits_nan(monkeypatch):
    # Deterministic data with non-zero residuals so the asymmetric (tau != 0.5)
    # IRLS reweight genuinely changes beta and needs more than one iteration.
    xv = np.linspace(0.0, 1.0, 40)
    yv = 2.0 * xv + np.sin(np.linspace(0.0, 6.0, 40))
    assert np.isfinite(_ae._expectile_slope(yv, xv, 0.2))
    monkeypatch.setattr(_ae, "_MAX_ITER_IRLS", 1)
    assert np.isnan(_ae._expectile_slope(yv, xv, 0.2))


# ---------------------------------------------------------------------------
# P1-18 empirical percentile centres for binned-response curvature
# ---------------------------------------------------------------------------
def _manual_curvature(yv, xv, bins, min_per_bin, empirical):
    """Re-implement the kernel fit with either empirical or nominal centres."""
    if empirical:
        medians, centers, counts = _binned_empirical_medians(yv, xv, bins)
    else:
        medians, counts = _binned_medians(yv, xv, bins)
        centers = (np.arange(bins) + 0.5) / bins
    usable = counts >= min_per_bin
    if int(usable.sum() < 3:
        return np.nan
    qq, mm = centers[usable], medians[usable]
    if not np.all(np.isfinite(qq)) or np.unique(qq).size < 3:
        return np.nan
    X = np.column_stack([np.ones_like(qq), qq, qq ** 2])
    coeff, *_ = np.linalg.lstsq(X, mm, rcond=None)
    sd = float(np.std(mm))
    return float(coeff[2]) / (sd + _EPS)


def test_curvature_uses_empirical_centers_not_nominal():
    rng = np.random.default_rng(6)
    n = 60
    # Heavily tied x: 20 exact zeros + 40 distinct values.  The zero mass
    # distorts the quantile cuts, so nominal ``(i+0.5)/bins`` centres are wrong.
    xv = np.r_[np.zeros(20), rng.uniform(1.0, 10.0, 40)]
    xs = np.sort(xv)
    yv = 0.5 * xs + 0.02 * xs ** 2 + rng.normal(scale=0.2, size=n)
    out = OperatorRegistry.get("ts_binned_response_curvature", "pandas_numpy").calculate(
        _col(yv), _col(xv), window=n, bins=5, min_per_bin=3
    )
    val = float(out.iloc[-1, 0])
    assert np.isfinite(val)
    # The operator MUST reproduce the empirical-centre manual fit exactly...
    assert val == pytest.approx(
        _manual_curvature(yv, xv, 5, 3, empirical=True), rel=1e-9
    )
    # ...and that fit differs materially from the old nominal-centre fit (the
    # tie mismatch that produced the wild curvature).
    nominal = _manual_curvature(yv, xv, 5, 3, empirical=False)
    assert np.isfinite(nominal)
    assert abs(val - nominal) > 1.0


def test_curvature_fully_tied_input_fails_closed_nan():
    # All x identical -> at most one usable group -> <3 usable bins -> NaN, not
    # a wild curvature from a degenerate fit.
    rng = np.random.default_rng(7)
    n = 60
    xv = np.zeros(n)
    yv = rng.normal(size=n)
    out = OperatorRegistry.get("ts_binned_response_curvature", "pandas_numpy").calculate(
        _col(yv), _col(xv), window=n, bins=5, min_per_bin=3
    )
    assert np.isnan(out.to_numpy(dtype=float)).all()


# ---------------------------------------------------------------------------
# P1-19 tie-balanced split for response-slope asymmetry
# ---------------------------------------------------------------------------
def test_balanced_quantile_split_distributes_ties():
    # 16 zeros + 4 distinct: a raw ``xv <= median`` split would push all 16
    # zeros into the low cohort (16 vs 4).  The rank split must balance them.
    xv = np.r_[np.zeros(16), np.arange(1.0, 5.0)]
    lo, hi = _balanced_quantile_split(xv, 0.5)
    assert lo.sum() == 10 and hi.sum() == 10
    # the split is monotone in value: every hi-member is >= every lo-member.
    assert np.all(xv[hi] >= np.max(xv[lo]))


def test_slope_asymmetry_tie_case_not_extreme_artifact():
    rng = np.random.default_rng(8)
    n = 60
    # 25 exact zeros + 35 distinct positives.  A raw threshold split at the
    # median (a positive) would leave the low cohort all zeros -> sx=0 -> no
    # low-side slope.  The tie-balanced rank split keeps 30/30 with variance on
    # both sides, so the asymmetry reflects the response, not cohort size.
    xv = np.r_[np.zeros(25), rng.uniform(1.0, 2.0, 35)]
    yv = 2.0 * xv + rng.normal(scale=0.1, size=n)
    out = OperatorRegistry.get("ts_response_slope_asymmetry", "pandas_numpy").calculate(
        _col(yv), _col(xv), window=n, split_quantile=0.5
    )
    val = float(out.iloc[-1, 0])
    assert np.isfinite(val)
    # Linear response: both sides share the same standardised slope -> asymmetry
    # near 0, far from the ±1 artifact the cohort-size imbalance would create.
    assert abs(val) < 0.1


def test_slope_asymmetry_near_tie_free_control():
    rng = np.random.default_rng(9)
    n = 60
    xv = rng.uniform(0.0, 2.0, n)
    yv = 2.0 * xv + rng.normal(scale=0.1, size=n)
    out = OperatorRegistry.get("ts_response_slope_asymmetry", "pandas_numpy").calculate(
        _col(yv), _col(xv), window=n, split_quantile=0.5
    )
    val = float(out.iloc[-1, 0])
    assert np.isfinite(val)
    assert abs(val) < 0.1


# ---------------------------------------------------------------------------
# smoke import / eval
# ---------------------------------------------------------------------------
def test_smoke_import_eval_both_modules():
    x = _col([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert OperatorRegistry.get("ts_expectile", "pandas_numpy").calculate(
        x, window=6, tau=0.5
    ).iloc[-1, 0] == pytest.approx(3.5)
    y = _col([3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
    beta = OperatorRegistry.get("ts_expectile_beta", "pandas_numpy").calculate(
        y, x, window=6, tau=0.5
    )
    assert beta.iloc[-1, 0] == pytest.approx(2.0, rel=1e-4)
