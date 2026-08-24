# -*- coding: utf-8 -*-
"""R11 round-2 regression tests for the intraday realized-beta operator family.

Three P0 math bugs fixed in ``cleaned_operators/intraday/realized_beta.py``:

* TASK 1 — ddof mismatch in ``_market_model``: ``np.cov`` defaults to ddof=1
  while ``np.var`` defaults to ddof=0, which scaled beta by ``(n-1)/n`` and
  polluted every derived market-model quantity (idiosyncratic variance /
  skewness / kurtosis and market R²).  ``_market_model`` now unifies on
  population moments (the OLS slope).
* TASK 2 — overnight returns were mixed into minute returns: the first minute of
  each session paired with the last minute of the previous session.  The first
  minute of every calendar-day session is now NaN.
* TASK 3 — the ex-self path re-broadcast market weights WITHOUT the
  finite-and-positive filter, letting NaN/0/negative caps excluded from the
  full-market return re-enter the leave-one-out market return.  The single
  validated weight panel is now built once and reused across every realized-beta
  series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

ensure_cleaned_loaded()

from factor_engine.cleaned_operators.intraday.realized_beta import (  # noqa: E402
    _aligned_market,
    _idio_kurtosis,
    _idio_skewness,
    _idio_variance,
    _market_model,
    _market_r2,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _minute_panel(days: int = 2, cols: int = 3, seed: int = 0, bars: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = []
    for d in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=d)
        # A-share session minute-of-day: 09:31..11:30 (571..690), 13:01..15:00 (781..900)
        for m in list(range(571, 691))[: bars // 2] + list(range(781, 901))[: bars // 2]:
            timestamps.append(day + pd.Timedelta(minutes=m))
    n = len(timestamps)
    return pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((n, cols)) * 0.01, axis=0)) * 100,
        index=pd.DatetimeIndex(timestamps),
        columns=[f"C{i}" for i in range(cols)],
    )


def _daily_weights(days: int = 2, cols: int = 3, value: float = 1e8) -> pd.DataFrame:
    dates = [pd.Timestamp("2024-01-03") + pd.Timedelta(days=d) for d in range(days)]
    return pd.DataFrame(
        np.full((days, cols), value),
        index=pd.DatetimeIndex(dates),
        columns=[f"C{i}" for i in range(cols)],
    )


def _price_from_returns(r: np.ndarray, start: float = 100.0) -> np.ndarray:
    """Prices whose successive log returns are exactly ``r`` (first row NaN)."""
    return start * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def _day_boundaries(idx: pd.DatetimeIndex) -> tuple[np.ndarray, list[int]]:
    """Return (boundary bool mask, [row 0, first row of each later session])."""
    days = idx.normalize().to_numpy()
    boundary = np.zeros(len(idx), dtype=bool)
    boundary[0] = True
    if len(idx) > 1:
        boundary[1:] = days[1:] != days[:-1]
    return boundary, [int(i) for i in np.flatnonzero(boundary)]


# ---------------------------------------------------------------------------
# TASK 1 — _market_model population moments
# ---------------------------------------------------------------------------
def test_market_model_population_slope() -> None:
    rng = np.random.default_rng(3)
    m = rng.standard_normal(60)
    slope = 2.5
    r = 1.0 + slope * m + 0.001 * rng.standard_normal(60)

    rbar, mbar = float(np.mean(r)), float(np.mean(m))
    b_ols = float(np.mean((r - rbar) * (m - mbar))) / float(np.mean((m - mbar) ** 2))
    a_ols = rbar - b_ols * mbar

    fit = _market_model(r, m)
    assert fit is not None
    a, b, e = fit
    # the old code mixed np.cov (ddof=1) with np.var (ddof=0), scaling beta by
    # n/(n-1) — 60/59 ≈ 1.7%, a fail above the tolerance below.
    assert b == pytest.approx(b_ols, rel=1e-12)
    assert a == pytest.approx(a_ols, rel=1e-9)
    # OLS residual properties: mean zero and orthogonal to the market factor.
    assert float(np.mean(e)) == pytest.approx(0.0, abs=1e-10)
    assert float(np.mean(e * m)) == pytest.approx(0.0, abs=1e-9)
    assert b == pytest.approx(slope, abs=0.005)


def test_market_model_derived_quantities_inherit_ols_slope() -> None:
    """Idio variance/skew/kurt and R² must all consume the OLS (population) slope."""
    rng = np.random.default_rng(21)
    m = rng.standard_normal(80)
    r = 0.5 + 1.7 * m + rng.standard_normal(80) * 0.1

    rbar, mbar = float(np.mean(r)), float(np.mean(m))
    b_ols = float(np.mean((r - rbar) * (m - mbar))) / float(np.mean((m - mbar) ** 2))
    a_ols = rbar - b_ols * mbar
    e_ols = r - (a_ols + b_ols * m)
    ss_tot = float(np.sum((r - rbar) ** 2))
    expected_r2 = 1.0 - float(np.sum(e_ols * e_ols)) / ss_tot
    sd_ols = float(np.std(e_ols))

    fit = _market_model(r, m)
    assert fit is not None
    a, b, e = fit
    assert b == pytest.approx(b_ols, rel=1e-12)
    np.testing.assert_allclose(e, e_ols, rtol=1e-9)

    assert _market_r2(r, m) == pytest.approx(max(0.0, expected_r2), rel=1e-9)
    assert _idio_variance(r, m) == pytest.approx(float(np.mean(e_ols * e_ols)), rel=1e-9)
    if sd_ols > 1e-12:
        assert _idio_skewness(r, m) == pytest.approx(
            float(np.mean(((e_ols - np.mean(e_ols)) / sd_ols) ** 3)), rel=1e-9
        )
        assert _idio_kurtosis(r, m) == pytest.approx(
            float(np.mean(((e_ols - np.mean(e_ols)) / sd_ols) ** 4)), rel=1e-9
        )
    assert a == pytest.approx(a_ols, rel=1e-9)


def test_realized_beta_recovers_constant_slope() -> None:
    """Black-box: beta recovers known slopes on a perfect linear relationship."""
    idx = _minute_panel(days=1, cols=1, seed=1).index
    n = len(idx) - 1
    rng = np.random.default_rng(11)
    common = rng.standard_normal(n)
    # A tracks the factor at slope 0.8; B IS the factor.  B dominates the cap so
    # the value-weighted market return ≈ factor -> beta_A ≈ 0.8, beta_B ≈ 1.0.
    close = pd.DataFrame(
        {
            "A": _price_from_returns(0.8 * common),
            "B": _price_from_returns(common),
        },
        index=idx,
    )
    w = pd.DataFrame({"A": [1e6], "B": [1e12]}, index=pd.DatetimeIndex(["2024-01-03"]))

    beta = OperatorRegistry.get("intra_realized_beta").calculate(close, w)
    assert beta.shape == (1, 2)
    assert np.isfinite(beta["A"].iloc[0]) and np.isfinite(beta["B"].iloc[0])
    assert beta["A"].iloc[0] == pytest.approx(0.8, abs=0.01)
    assert beta["B"].iloc[0] == pytest.approx(1.0, abs=0.01)

    corr = OperatorRegistry.get("intra_realized_correlation").calculate(close, w)
    assert corr["A"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert corr["B"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    idio = OperatorRegistry.get("intra_idiosyncratic_variance").calculate(close, w)
    assert np.isfinite(idio["A"].iloc[0])
    assert idio["A"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    r2 = OperatorRegistry.get("intra_market_model_r2").calculate(close, w)
    assert r2["A"].iloc[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TASK 2 — no overnight return inside a trading session
# ---------------------------------------------------------------------------
def test_first_minute_of_second_session_is_nan() -> None:
    idx = _minute_panel(days=2, cols=2, seed=5).index
    boundary, starts = _day_boundaries(idx)
    assert len(starts) == 2  # row 0 + the second session's first minute
    day2_start = starts[1]

    rng = np.random.default_rng(5)
    base = 100.0 * np.exp(np.cumsum(rng.standard_normal((len(idx), 2)) * 0.005, axis=0))
    close = pd.DataFrame(base, index=idx, columns=["C0", "C1"])
    # inject a 10x overnight gap between day 1 close and day 2 open
    close.iloc[day2_start:] = base[day2_start:] * 10.0

    rets, mkt, w_bc, w_ret = _aligned_market(close, _daily_weights(days=2, cols=2))

    # the first minute of the second session must be NaN (no overnight return)
    assert rets.iloc[day2_start].isna().all()
    # the very first row of the panel is NaN too
    assert rets.iloc[0].isna().all()
    # every intra-session minute is finite
    assert rets.iloc[1:day2_start].notna().all().all()
    assert rets.iloc[day2_start + 1:].notna().all().all()
    # the market return at the session boundary is NaN as well
    assert np.isnan(mkt.iloc[day2_start])


def test_overnight_gap_does_not_contaminate_beta() -> None:
    idx = _minute_panel(days=2, cols=2, seed=7).index
    _, starts = _day_boundaries(idx)
    day2_start = starts[1]

    rng = np.random.default_rng(7)
    base = 100.0 * np.exp(np.cumsum(rng.standard_normal((len(idx), 2)) * 0.005, axis=0))
    clean = pd.DataFrame(base, index=idx, columns=["C0", "C1"])
    gap = clean.copy()
    gap.iloc[day2_start:, 0] = clean.iloc[day2_start:, 0] * 100.0  # huge overnight gap in C0

    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")])
    w = pd.DataFrame({"C0": [1e8, 1e8], "C1": [2e8, 2e8]}, index=dates)
    op = OperatorRegistry.get("intra_realized_beta")

    clean_out = op.calculate(clean, w)
    gap_out = op.calculate(gap, w)
    # with the fix the day-2 first minute (an overnight gap) is NaN and dropped
    # in BOTH panels, so day-1 and day-2 betas are identical.
    pd.testing.assert_frame_equal(clean_out, gap_out, check_dtype=False)


# ---------------------------------------------------------------------------
# TASK 3 — validated weights reused by the ex-self path
# ---------------------------------------------------------------------------
def test_aligned_market_filters_invalid_weights() -> None:
    close = _minute_panel(days=1, cols=3, seed=2)
    weights = pd.DataFrame(
        {"C0": [1e8], "C1": [np.nan], "C2": [-3e8]}, index=pd.DatetimeIndex(["2024-01-03"])
    )
    rets, mkt, w_bc, w_ret = _aligned_market(close, weights)
    # NaN and negative caps are dropped from the validated panel; C0 survives.
    assert w_bc["C1"].isna().all()
    assert w_bc["C2"].isna().all()
    assert w_bc["C0"].notna().all()
    # with a single valid member the market return equals that member's return
    np.testing.assert_allclose(
        mkt.dropna().to_numpy(), rets["C0"].dropna().to_numpy(), rtol=1e-12
    )


def test_invalid_weight_excluded_from_market_and_ex_self() -> None:
    close = _minute_panel(days=2, cols=3, seed=5)
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")])
    weights_neg = pd.DataFrame(
        {"C0": [1e8, 1e8], "C1": [2e8, 2e8], "C2": [-5e8, -5e8]}, index=dates
    )
    weights_zero = weights_neg.copy()
    weights_zero["C2"] = 0.0

    op_full = OperatorRegistry.get("intra_realized_beta")
    op_ex = OperatorRegistry.get("intra_realized_beta_ex_self")

    full_neg = op_full.calculate(close, weights_neg)
    full_zero = op_full.calculate(close, weights_zero)
    ex_neg = op_ex.calculate(close, weights_neg)
    ex_zero = op_ex.calculate(close, weights_zero)

    # A negative or zero cap must be excluded IDENTICALLY from the full-market
    # return and from every healthy name's ex-self market return.  (Before the
    # fix the ex-self path re-broadcast raw weights, so the negative cap leaked
    # back in and broke the ex-self outputs.)
    pd.testing.assert_frame_equal(full_neg, full_zero, check_dtype=False)
    pd.testing.assert_frame_equal(ex_neg, ex_zero, check_dtype=False)

    # the invalid-cap name itself drops out of its own leave-one-out market
    # (its exclusion leaves no market) -> its ex-self beta is NaN.
    assert np.isnan(ex_neg["C2"]).all()
    assert np.isnan(ex_zero["C2"]).all()
    # healthy names still produce finite ex-self betas
    assert np.isfinite(ex_neg["C0"]).all()
    assert np.isfinite(ex_neg["C1"]).all()
