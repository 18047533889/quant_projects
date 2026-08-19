# -*- coding: utf-8 -*-
"""R20 causal dimensionless regression DirectUse rehabilitation oracles.

Slice: R20-REGRESSION-CAUSAL-DIRECTUSE.

Duplicate-audit outcome (survey BEFORE implementing — documented skips):
* ``ts_mean_reversion_half_life`` / ``ts_mean_reversion_ou_approx_half_life``
  (cleaned_operators/ts_model/ar_meanrev.py, experimental) already implement
  BOTH the exact discrete AR(1) half-life ln(0.5)/ln(phi) and the OU
  approximation -ln(2)/b, with the same 0<phi<1 NaN guard requested here.
  A new ``mean_reversion_half_life`` would be a near-exact duplicate of the
  same estimator -> SKIPPED.
* ``ts_regression_slope/r2/tstat/resid/forecast_error(_z)`` /
  ``ts_trend_tstat`` (cleaned_operators/overhaul/regression.py,
  EXTENDED_ONLY) are general two-frame (y, x) rolling OLS kernels with
  pairwise-finite NaN handling.  The landed family is the technical-signal
  canonicalization: close-only, single trailing window, min_periods=window
  (fail-closed NaN-in-window), strict-positive close masking (R5-38),
  dimensionless output, plus a target-bar-EXCLUDING forecast variant.  No
  duplicate canonical name is registered.

Landed canonicals (all rolling-only, trailing windows END at t,
min_periods=window -> any NaN inside the window -> NaN, fail-closed;
strict-positive close masking, never abs()-laundered; NOT in
``_RECURSIVE_EWM``; degenerate denominators -> NaN, never 0):
* ``reg_forecast_error_pct``  — (close_t - OLS forecast of close_t)/close_t;
  the fit window t-w..t-1 EXCLUDES the forecast target bar t.
* ``reg_slope_tstat``         — trailing OLS time-trend slope / SE(slope).
* ``reg_r2_trailing``         — trailing OLS time-trend R^2 in [0,1];
  constant window -> NaN (degenerate denominator).
* ``reg_residual_zscore``     — contemporaneous residual / trailing residual
  std; the fit window ends at t and INCLUDES t (contemporaneous, NOT
  look-ahead — no data after t enters the fit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    from cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("reg_forecast_error_pct", "pandas_numpy") is not None:
        return
    from cleaned_operators.technical import signal  # noqa: F401
    from cleaned_operators.technical import polars_signal  # noqa: F401
    from cleaned_operators import composite_fastpath  # noqa: F401
    from cleaned_operators.technical import indicators_v2  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_technical_chain()


def _op(name: str):
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


REG_NAMES = (
    "reg_forecast_error_pct",
    "reg_slope_tstat",
    "reg_r2_trailing",
    "reg_residual_zscore",
)


def _random_panel(n=60, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": 100.0 + np.cumsum(rng.normal(0.05, 0.8, n))})


# ---------------------------------------------------------------------------
# manual oracles — explicit closed-form sums / np.polyfit, fully independent
# of the operator helpers in indicators_v2
# ---------------------------------------------------------------------------
def _ols(y):
    n = y.size
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    s_tt = float(np.dot(tc, tc))
    slope = float(np.dot(tc, y - y.mean()) / s_tt)
    intercept = float(y.mean() - slope * t.mean())
    resid = y - (intercept + slope * t)
    ss_res = float(np.dot(resid, resid))
    centered = y - y.mean()
    ss_tot = float(np.dot(centered, centered))
    return slope, intercept, resid, ss_res, ss_tot


def _manual_oracle(close, window, kind):
    arr = close.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        if kind == "fcst":
            fit = arr[t - window + 1 : t]  # EXCLUDES the target bar t
            y_t = arr[t]
            if not (np.isfinite(fit).all() and np.isfinite(y_t) and y_t > 0.0):
                out.append(np.nan)
                continue
            slope, intercept, _, _, _ = _ols(fit)
            out.append((y_t - (intercept + slope * float(fit.size))) / y_t)
        else:
            a = arr[t - window + 1 : t + 1]  # window ends at t, includes t
            if not np.isfinite(a).all() or np.any(a <= 0.0):
                out.append(np.nan)
                continue
            slope, intercept, resid, ss_res, ss_tot = _ols(a)
            if kind == "tstat":
                t = np.arange(a.size, dtype=float)
                se = np.sqrt(ss_res / (a.size - 2) / np.dot(t - t.mean(), t - t.mean()))
                out.append(np.nan if not np.isfinite(se) or se <= 0 else slope / se)
            elif kind == "r2":
                out.append(np.nan if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot)
            elif kind == "zscore":
                sd = np.sqrt(ss_res / (a.size - 2))
                fitted_t = intercept + slope * float(a.size - 1)
                out.append(np.nan if not np.isfinite(sd) or sd <= 0 else (a[-1] - fitted_t) / sd)
            else:
                raise AssertionError(kind)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


@pytest.mark.parametrize("name,kind", [
    ("reg_forecast_error_pct", "fcst"),
    ("reg_slope_tstat", "tstat"),
    ("reg_r2_trailing", "r2"),
    ("reg_residual_zscore", "zscore"),
])
def test_matches_manual_oracle(name, kind):
    close = _random_panel(n=60, seed=7)
    out = _op(name).calculate(close, window=10)
    expected = _manual_oracle(close, 10, kind)
    np.testing.assert_allclose(
        out.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # warmup: first w-1 bars are NaN
    assert out.iloc[:9].isna().all().all()
    assert out.iloc[9:].notna().all().all()


def test_oracle_cross_checked_with_polyfit():
    # independence check of the oracle itself: slope/intercept agree with
    # numpy.polyfit on a fixed window.
    y = _random_panel(n=20, seed=3).to_numpy(float).ravel()[-10:]
    slope, intercept, _, _, _ = _ols(y)
    p_slope, p_intercept = np.polyfit(np.arange(10, dtype=float), y, 1)
    np.testing.assert_allclose([slope, intercept], [p_slope, p_intercept], rtol=1e-10)


# ---------------------------------------------------------------------------
# perfect linear trend panel: forecast error ~ 0, R^2 == 1, |t-stat| large
# ---------------------------------------------------------------------------
def test_perfect_linear_trend_panel():
    w = 10
    lin = pd.DataFrame({"A": 100.0 + 0.5 * np.arange(48)})
    fcst = _op("reg_forecast_error_pct").calculate(lin, window=w)
    tail = fcst.iloc[w - 1 :]
    assert np.nanmax(np.abs(tail.to_numpy())) < 1e-10
    r2 = _op("reg_r2_trailing").calculate(lin, window=w)
    np.testing.assert_allclose(r2.iloc[w - 1 :].to_numpy(), 1.0, rtol=1e-12)
    # perfect line: residual std is exactly 0 -> degenerate SE -> NaN
    # (fail-closed, never a fabricated infinity)
    tstat = _op("reg_slope_tstat").calculate(lin, window=w)
    assert tstat.iloc[w - 1 :].isna().all().all()
    z = _op("reg_residual_zscore").calculate(lin, window=w)
    assert z.iloc[w - 1 :].isna().all().all()
    # near-perfect line (tiny noise): |t-stat| becomes very large
    rng = np.random.default_rng(2)
    near = pd.DataFrame({"A": 100.0 + 0.5 * np.arange(64) + rng.normal(0.0, 1e-6, 64)})
    t2 = _op("reg_slope_tstat").calculate(near, window=w).iloc[w - 1 :]
    assert (t2.abs() > 1e5).all().all()


# ---------------------------------------------------------------------------
# white-noise panel: R^2 low, residual z-score finite and O(1)
# ---------------------------------------------------------------------------
def test_white_noise_panel():
    w = 20
    rng = np.random.default_rng(11)
    noise = pd.DataFrame({"A": 100.0 + rng.normal(0.0, 1.0, 200)})
    r2 = _op("reg_r2_trailing").calculate(noise, window=w).iloc[w - 1 :]
    assert (r2 < 0.5).all().all()
    z = _op("reg_residual_zscore").calculate(noise, window=w).iloc[w - 1 :]
    assert np.isfinite(z.to_numpy()).all()
    assert np.nanmax(np.abs(z.to_numpy())) < 10.0
    # t-stat on noise: |t| < 5 almost surely over 180 draws
    t = _op("reg_slope_tstat").calculate(noise, window=w).iloc[w - 1 :]
    assert (t.abs() < 5.0).all().all()


# ---------------------------------------------------------------------------
# AR(1) panel with known phi: half-life estimator sanity — exercised on the
# EXISTING canonical (duplicate-skip assertion), not a new operator
# ---------------------------------------------------------------------------
def test_ar1_half_life_duplicate_skip_and_estimator():
    from cleaned_operators.registry import OperatorRegistry

    # the skipped candidate name must NOT be registered by this slice
    assert OperatorRegistry.get("mean_reversion_half_life") is None
    # the existing experimental canonical implements the same estimator
    op = OperatorRegistry.get("ts_mean_reversion_half_life") or OperatorRegistry.get(
        "ts_mean_reversion_half_life", "pandas_numpy"
    )
    assert op is not None  # duplicate rationale: same estimator, kept once
    # estimator sanity on a synthetic AR(1) with phi=0.75 -> half-life ln2/ln(4/3)
    import warnings

    rng = np.random.default_rng(5)
    phi = 0.75
    x = np.zeros(6000)
    eps = rng.normal(0.0, 1.0, 6000)
    for i in range(1, 6000):
        x[i] = phi * x[i - 1] + eps[i]
    panel = pd.DataFrame({"A": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = op.calculate(panel, window=3000, min_periods=100)
    hl = float(out.iloc[-1, 0])
    expected = -np.log(2.0) / np.log(phi)
    assert np.isfinite(hl)
    assert abs(hl - expected) < 0.5


# ---------------------------------------------------------------------------
# causality: mutating bar t must not change any fit that excludes t
# ---------------------------------------------------------------------------
def test_forecast_family_excludes_target_bar():
    close = _random_panel(n=40, seed=9)
    w = 8
    base = _op("reg_forecast_error_pct").calculate(close, window=w)
    mutated = close.copy()
    mutated.iloc[25, 0] = mutated.iloc[25, 0] * 1.5  # mutate bar t=25
    after = _op("reg_forecast_error_pct").calculate(mutated, window=w)
    # bar 25's own output changes (it is the forecast TARGET) ...
    assert not np.isclose(base.iloc[25, 0], after.iloc[25, 0])
    # ... but bars whose fit window lies strictly before 25 (t <= 25, fit
    # window ends at t-1 <= 24) are untouched.
    pd.testing.assert_frame_equal(base.iloc[:25], after.iloc[:25])
    # the contemporaneous family (fit includes t) must NOT see future bars:
    # mutating bar 30 leaves every output at t <= 29 unchanged.
    mut2 = close.copy()
    mut2.iloc[30, 0] = mut2.iloc[30, 0] * 0.5
    for name in ("reg_slope_tstat", "reg_r2_trailing", "reg_residual_zscore"):
        a = _op(name).calculate(close, window=w)
        b = _op(name).calculate(mut2, window=w)
        pd.testing.assert_frame_equal(a.iloc[:30], b.iloc[:30])


def test_prefix_invariance():
    close = _random_panel(n=50, seed=13)
    for name in REG_NAMES:
        full = _op(name).calculate(close, window=8)
        head = _op(name).calculate(close.iloc[:30], window=8)
        pd.testing.assert_frame_equal(full.iloc[:30], head)


# ---------------------------------------------------------------------------
# bad-close masking (R5-38) + NaN-in-window fail-closed
# ---------------------------------------------------------------------------
def test_masks_non_positive_close():
    close = _random_panel(n=40, seed=5)
    close.iloc[20, 0] = -1.0
    for name in REG_NAMES:
        out = _op(name).calculate(close, window=8)
        # the bad bar and every window containing it -> NaN
        assert np.isnan(out.iloc[20, 0]), name
        assert out.iloc[20:28].isna().all().all(), name
        assert out.iloc[28:].notna().all().all(), name
        assert out.iloc[7:20].notna().all().all(), name


def test_zero_close_masked():
    close = _random_panel(n=30, seed=6)
    close.iloc[10, 0] = 0.0
    for name in REG_NAMES:
        out = _op(name).calculate(close, window=6)
        assert out.iloc[10:16].isna().all().all(), name


def test_nan_in_window_fail_closed():
    close = _random_panel(n=40, seed=5)
    close.iloc[20, 0] = np.nan
    for name in REG_NAMES:
        out = _op(name).calculate(close, window=8)
        # every window containing bar 20 (t in 20..27) -> NaN — never zero-filled
        assert out.iloc[20:28].isna().all().all(), name
        assert out.iloc[28:].notna().all().all(), name
        assert out.iloc[7:20].notna().all().all(), name


def test_constant_window_degenerate_nan():
    # constant window: R^2 denominator (total variation) is 0 -> NaN (never
    # a fabricated 0/1); residual std is 0 -> zscore NaN.
    close = pd.DataFrame({"A": [10.0] * 20})
    r2 = _op("reg_r2_trailing").calculate(close, window=8)
    z = _op("reg_residual_zscore").calculate(close, window=8)
    t = _op("reg_slope_tstat").calculate(close, window=8)
    assert r2.iloc[7:].isna().all().all()
    assert z.iloc[7:].isna().all().all()
    assert t.iloc[7:].isna().all().all()


# ---------------------------------------------------------------------------
# governance / promotion
# ---------------------------------------------------------------------------
def test_param_specs_and_governance():
    for name in REG_NAMES:
        specs = _op(name).metadata.param_specs
        assert set(specs) == {"window"}, name
        # forecast error's effective minimum is 4 (fit excludes the target
        # bar: window=3 leaves a 2-bar fit -> all-NaN); the contemporaneous
        # three are genuinely valid at 3 (review P1).
        expected_min = 4 if name == "reg_forecast_error_pct" else 3
        assert specs["window"].min == expected_min, name
        assert specs["window"].dtype is int, name
        assert specs["window"].param_role is not None
        assert not _op(name).metadata.relational_specs, name

    from cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    for name in REG_NAMES:
        assert name not in _RECURSIVE_EWM, name
        tags = _op(name).metadata.tags
        assert "stateful" not in tags, name
        assert "causal" in tags and "pit_safe" in tags, name


def test_promotion_membership_and_duplicate_skip():
    from mining.direct_use import _RELATIVE_ALPHA_OPS

    assert set(REG_NAMES) <= _RELATIVE_ALPHA_OPS

    from cleaned_operators.operator_surface import (
        _DAILY_REGRESSION_PACK_2026_08,
        classify_canonical,
        daily_factor_migrated,
    )

    assert _DAILY_REGRESSION_PACK_2026_08 == frozenset(REG_NAMES)
    assert _DAILY_REGRESSION_PACK_2026_08 <= daily_factor_migrated()
    for name in REG_NAMES:
        assert classify_canonical(name) == "daily", name

    # duplicate-skip: the skipped family names stay un-promoted here
    assert "mean_reversion_half_life" not in _RELATIVE_ALPHA_OPS
    assert "mean_reversion_half_life" not in daily_factor_migrated()

    from mining.direct_use import _PRICE_LEVEL_INTERMEDIATE_OPS

    for name in REG_NAMES:
        assert name not in _PRICE_LEVEL_INTERMEDIATE_OPS, name


def test_window_below_three_rejected():
    close = _random_panel(n=20, seed=4)
    for name in REG_NAMES:
        with pytest.raises(ValueError):
            _op(name).calculate(close, window=2)


def test_forecast_window_three_rejected():
    # window=3 is a silently all-NaN search node for the FORECAST canonical
    # (fit on t-w..t-1 = 2 bars -> _reg_fit_window returns None for every
    # bar); the runtime guard now rejects it up front (review P1).
    close = _random_panel(n=20, seed=4)
    with pytest.raises(ValueError):
        _op("reg_forecast_error_pct").calculate(close, window=3)
    # the contemporaneous three remain valid at window=3
    for name in ("reg_slope_tstat", "reg_r2_trailing", "reg_residual_zscore"):
        out = _op(name).calculate(close, window=3)
        assert out.iloc[2:].notna().all().all(), name
