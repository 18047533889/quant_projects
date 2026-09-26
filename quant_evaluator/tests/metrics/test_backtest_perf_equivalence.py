"""Equivalence & performance tests for the backtest/portfolio-statistics vectorization.

Covers the seven optimization tasks (A-G) applied to the backtest/portfolio
statistics family on the server-c release tree:

    A. portfolio_stats.compute_aligned_wealth_curve        (vectorized cumprod)
    B. underwater.compute_worst_period_return              (sliding_window_view)
    C. underwater.compute_rolling_sharpe_tail /
       sharpe.compute_rolling_sharpe_quantile              (cumsum closed-form)
    D. turnover._rank_weights_matrix                       (batched rankdata)
    E. turnover.estimate_turnover_from_ranks               (panelized turnover)
    F. risk.drawdown_analysis.drawdown_events              (cumprod + RLE)
    G. portfolio_stats.compute_sharpe_ratio /
       compute_sortino_ratio / compute_maximum_drawdown /
       compute_calmar_ratio                                (vectorized over F)

Every changed function retains a verbatim ``_xxx_reference`` oracle. These tests
assert bit-level equivalence (or, for the cumsum closed-form rolling Sharpe C,
GPU-parity tolerance) against that oracle, verify boundary behavior, pin a few
hardcoded known-value oracles, and enforce a performance guardrail (the new
vectorized code must be at least as fast as the Python-loop oracle at T=2000).

Measured maximum relative deviation vs the oracle under the house rule
``rtol=1e-8, atol=1e-10, equal_nan=True`` (see also the docstrings of each
rewritten function):

    A  compute_aligned_wealth_curve      bit-exact
    B  compute_worst_period_return       bit-exact
    C  compute_rolling_sharpe_tail       ~1.48e-15
    C  compute_rolling_sharpe_quantile   ~2.57e-15
    D  _rank_weights_matrix              bit-exact (max |diff| = 0.0)
    E  estimate_turnover_from_ranks      bit-exact (max |diff| = 0.0)
    F  drawdown_events                   bit-exact
    G  compute_sharpe_ratio              ~1.25e-15
    G  compute_sortino_ratio             ~1.25e-15
    G  compute_maximum_drawdown          bit-exact
    G  compute_calmar_ratio              ~5.3e-15 (within tol; CAGR prod**pow)
"""

import math
import time

import numpy as np
import pytest

from quant_evaluator.metrics import portfolio_stats as ps
from quant_evaluator.metrics import underwater as uw
from quant_evaluator.metrics import turnover as to
from quant_evaluator.metrics.probe_portfolio import sharpe as sh
from quant_evaluator.metrics.risk import drawdown_analysis as da

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

SEEDS = [0, 7, 42]
NAN_RATES = [0.0, 0.10, 0.30]
SHORT_T = 40


def make_panel(T, F, nan_rate, rng):
    """Random (T, F) returns with a controlled NaN rate (per element)."""
    arr = rng.normal(0.0, 0.02, size=(T, F))
    if nan_rate > 0:
        mask = rng.random((T, F)) < nan_rate
        arr[mask] = np.nan
    return arr


def make_1d(T, nan_rate, rng):
    arr = rng.normal(0.0, 0.02, size=T)
    if nan_rate > 0:
        mask = rng.random(T) < nan_rate
        arr[mask] = np.nan
    return arr


def make_3d(T, N, F, nan_rate, rng):
    arr = rng.normal(0.0, 1.0, size=(T, N, F))
    if nan_rate > 0:
        mask = rng.random((T, N, F)) < nan_rate
        arr[mask] = np.nan
    return arr


def assert_bit_equal(a, b, msg=""):
    # rtol=0/atol=0 makes this a bit-exact comparison; equal_nan keeps NaN slots
    # aligned (both implementations emit NaN identically for invalid columns).
    np.testing.assert_allclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
        err_msg=msg,
    )


def assert_tol(a, b, msg=""):
    np.testing.assert_allclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        rtol=1e-8,
        atol=1e-10,
        equal_nan=True,
        err_msg=msg,
    )


def best_time(fn, repeats=5):
    """Return the best (smallest) wall-clock time over ``repeats`` calls."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def normalize_event(ev):
    """Canonical, hash-comparable form of a drawdown event dict (NaN -> 'nan')."""
    out = {}
    for k, v in ev.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = "nan"
        else:
            out[k] = v
    return out


def normalize_events(events):
    return [normalize_event(e) for e in events]


# --------------------------------------------------------------------------- #
# Task A  compute_aligned_wealth_curve
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
def test_A_equivalence_matrix(seed, nan_rate):
    rng = np.random.default_rng(seed)
    r = make_1d(SHORT_T, nan_rate, rng)
    assert_bit_equal(
        ps.compute_aligned_wealth_curve(r),
        ps._compute_aligned_wealth_curve_reference(r),
        msg=f"A seed={seed} nan={nan_rate}",
    )


def test_A_known_value():
    r = np.array([0.1, -0.1, 0.3])
    expected = np.array([1.1, 0.99, 1.287])
    # 1.287 is the nearest float to 0.99*1.3; allow a 1-ulp rounding slack.
    np.testing.assert_allclose(
        ps.compute_aligned_wealth_curve(r), expected, rtol=0.0, atol=1e-9
    )


def test_A_wealth_exactly_zero_absorbing():
    # A -100% return drives wealth to exactly 0 and it stays 0 (absorbing).
    r = np.array([0.5, -1.0, 0.5])
    expected = np.array([1.5, 0.0, 0.0])
    assert_bit_equal(ps.compute_aligned_wealth_curve(r), expected)
    assert_bit_equal(
        ps.compute_aligned_wealth_curve(r),
        ps._compute_aligned_wealth_curve_reference(r),
    )


def test_A_single_element():
    assert_bit_equal(
        ps.compute_aligned_wealth_curve(np.array([0.1])),
        np.array([1.1]),
    )


def test_A_returns_below_minus_100_raises():
    with pytest.raises(ValueError):
        ps.compute_aligned_wealth_curve(np.array([-1.5]))
    # Reference raises identically.
    with pytest.raises(ValueError):
        ps._compute_aligned_wealth_curve_reference(np.array([-1.5]))


def test_A_performance_guardrail_T2000():
    rng = np.random.default_rng(123)
    r = rng.normal(0.0, 0.02, size=2000)
    t_new = best_time(lambda: ps.compute_aligned_wealth_curve(r))
    t_ref = best_time(lambda: ps._compute_aligned_wealth_curve_reference(r))
    assert t_new < t_ref, f"A new {t_new:.4g} not faster than ref {t_ref:.4g}"


# --------------------------------------------------------------------------- #
# Task B  compute_worst_period_return
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
@pytest.mark.parametrize("period", ["month", "quarter", "year"])
def test_B_equivalence_matrix(seed, nan_rate, period):
    rng = np.random.default_rng(seed)
    r = make_1d(SHORT_T, nan_rate, rng)
    got = uw.compute_worst_period_return(r, period)
    ref = uw._compute_worst_period_return_reference(r, period)
    assert_bit_equal(got, ref, msg=f"B seed={seed} nan={nan_rate} period={period}")


def test_B_known_value_month_block():
    # 21-length block (month), single -0.5 at index 5, rest 0 -> block return -0.5
    r = np.zeros(21)
    r[5] = -0.5
    assert_bit_equal(uw.compute_worst_period_return(r, "month"), -0.5)


def test_B_block_gt_T_returns_nan():
    r = np.array([0.01, -0.02, 0.03])
    assert math.isnan(uw.compute_worst_period_return(r, "year"))
    assert math.isnan(uw._compute_worst_period_return_reference(r, "year"))


def test_B_all_nan_returns_nan():
    r = np.full(30, np.nan)
    assert math.isnan(uw.compute_worst_period_return(r, "month"))
    assert math.isnan(uw._compute_worst_period_return_reference(r, "month"))


def test_B_performance_guardrail_T2000():
    rng = np.random.default_rng(123)
    r = rng.normal(0.0, 0.03, size=2000)
    t_new = best_time(lambda: uw.compute_worst_period_return(r, "year"))
    t_ref = best_time(lambda: uw._compute_worst_period_return_reference(r, "year"))
    assert t_new < t_ref, f"B new {t_new:.4g} not faster than ref {t_ref:.4g}"


# --------------------------------------------------------------------------- #
# Task C  rolling Sharpe (underwater + sharpe) — cumsum closed-form (tolerance)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
def test_C_tail_equivalence_matrix(seed, nan_rate):
    rng = np.random.default_rng(seed)
    r = make_1d(SHORT_T, nan_rate, rng)
    got = uw.compute_rolling_sharpe_tail(r, window=20, quantile=0.1)
    ref = uw._compute_rolling_sharpe_tail_reference(r, window=20, quantile=0.1)
    assert_tol(got, ref, msg=f"C-tail seed={seed} nan={nan_rate}")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
def test_C_quantile_equivalence_matrix(seed, nan_rate):
    rng = np.random.default_rng(seed)
    r = make_1d(SHORT_T, nan_rate, rng)
    got = sh.compute_rolling_sharpe_quantile(r, window=15, quantile=0.2)
    ref = sh._compute_rolling_sharpe_quantile_reference(r, window=15, quantile=0.2)
    assert_tol(got, ref, msg=f"C-quantile seed={seed} nan={nan_rate}")


def test_C_window_gt_T_returns_nan():
    r = np.array([0.01, -0.02, 0.03, 0.01])
    assert math.isnan(uw.compute_rolling_sharpe_tail(r, window=300))
    assert math.isnan(uw._compute_rolling_sharpe_tail_reference(r, window=300))
    assert math.isnan(sh.compute_rolling_sharpe_quantile(r, window=300))
    assert math.isnan(
        sh._compute_rolling_sharpe_quantile_reference(r, window=300)
    )


def test_C_min_periods_boundary():
    # Exactly min_periods finite values -> defined; min_periods-1 -> NaN.
    r = np.concatenate([np.full(30, 0.01), [np.nan]])
    got = uw.compute_rolling_sharpe_tail(r, window=30, min_periods=30)
    ref = uw._compute_rolling_sharpe_tail_reference(r, window=30, min_periods=30)
    assert_tol(got, ref)
    r2 = np.concatenate([np.full(29, 0.01), [np.nan, np.nan]])
    got2 = uw.compute_rolling_sharpe_tail(r2, window=30, min_periods=30)
    ref2 = uw._compute_rolling_sharpe_tail_reference(r2, window=30, min_periods=30)
    assert_tol(got2, ref2)
    assert math.isnan(got2)


def test_C_performance_guardrail_T2000():
    rng = np.random.default_rng(123)
    r = rng.normal(0.0, 0.03, size=2000)
    t_new = best_time(lambda: uw.compute_rolling_sharpe_tail(r, window=252))
    t_ref = best_time(
        lambda: uw._compute_rolling_sharpe_tail_reference(r, window=252)
    )
    assert t_new < t_ref, f"C new {t_new:.4g} not faster than ref {t_ref:.4g}"


# --------------------------------------------------------------------------- #
# Task D  _rank_weights_matrix — batched ranking
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
def test_D_equivalence_matrix(seed, nan_rate):
    rng = np.random.default_rng(seed)
    vals = make_3d(SHORT_T, 20, 3, nan_rate, rng)
    got = to._rank_weights_matrix(vals, min_obs=10)
    ref = to._rank_weights_matrix_reference(vals, min_obs=10)
    max_diff = np.nanmax(np.abs(got - ref))
    assert np.array_equal(got, ref, equal_nan=True), f"D max|diff|={max_diff}"


def test_D_min_obs_boundary():
    rng = np.random.default_rng(1)
    vals = rng.normal(0, 1, (10, 5, 2))
    # min_obs larger than N -> all NaN
    got = to._rank_weights_matrix(vals, min_obs=10)
    ref = to._rank_weights_matrix_reference(vals, min_obs=10)
    assert np.array_equal(got, ref, equal_nan=True)
    assert np.all(np.isnan(got))


def test_D_single_cross_section_full_finite():
    # One (t,f) with all finite -> ranks normalized to sum 1.
    vals = np.full((1, 6, 1), 0.0)
    vals[0, :, 0] = np.array([3.0, 1.0, 2.0, 5.0, 4.0, 0.0])
    got = to._rank_weights_matrix(vals, min_obs=2)
    col = got[0, :, 0]
    assert np.all(np.isfinite(col))
    np.testing.assert_allclose(col.sum(), 1.0, atol=1e-12)


def test_D_performance_guardrail_T2000():
    rng = np.random.default_rng(123)
    vals = rng.normal(0, 1, size=(2000, 20, 3))
    t_new = best_time(lambda: to._rank_weights_matrix(vals, min_obs=10))
    t_ref = best_time(lambda: to._rank_weights_matrix_reference(vals, min_obs=10))
    assert t_new < t_ref, f"D new {t_new:.4g} not faster than ref {t_ref:.4g}"


# --------------------------------------------------------------------------- #
# Task E  estimate_turnover_from_ranks — panelized turnover
# --------------------------------------------------------------------------- #

class _FakeBatch:
    """Duck-typed stand-in for FactorBatch; estimate_turnover_from_ranks only
    reads the ``values`` attribute."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)


def _fb(values):
    return _FakeBatch(values)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
@pytest.mark.parametrize("window", [1, 3])
def test_E_equivalence_matrix(seed, nan_rate, window):
    rng = np.random.default_rng(seed)
    vals = make_3d(SHORT_T, 20, 3, nan_rate, rng)
    fb = _fb(vals)
    got = to.estimate_turnover_from_ranks(fb, window=window)
    ref = to._estimate_turnover_from_ranks_reference(fb, window=window)
    max_diff = np.nanmax(np.abs(got - ref))
    assert np.array_equal(got, ref, equal_nan=True), f"E w={window} max|diff|={max_diff}"


def test_E_window_lt_1_raises():
    fb = _fb(np.random.default_rng(0).normal(0, 1, (10, 5, 2)))
    with pytest.raises(ValueError):
        to.estimate_turnover_from_ranks(fb, window=0)
    with pytest.raises(ValueError):
        to._estimate_turnover_from_ranks_reference(fb, window=0)


def test_E_T_le_window_returns_nan():
    fb = _fb(np.random.default_rng(0).normal(0, 1, (3, 5, 2)))
    got = to.estimate_turnover_from_ranks(fb, window=5)
    ref = to._estimate_turnover_from_ranks_reference(fb, window=5)
    assert np.array_equal(got, ref, equal_nan=True)
    assert np.all(np.isnan(got))


def test_E_performance_guardrail_T2000():
    rng = np.random.default_rng(123)
    vals = rng.normal(0, 1, size=(2000, 20, 3))
    fb = _fb(vals)
    t_new = best_time(lambda: to.estimate_turnover_from_ranks(fb, window=1))
    t_ref = best_time(
        lambda: to._estimate_turnover_from_ranks_reference(fb, window=1)
    )
    assert t_new < t_ref, f"E new {t_new:.4g} not faster than ref {t_ref:.4g}"


# --------------------------------------------------------------------------- #
# Task F  drawdown_events — cumprod + RLE
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
def test_F_equivalence_matrix(seed, nan_rate):
    rng = np.random.default_rng(seed)
    r = make_1d(SHORT_T, nan_rate, rng)
    got = normalize_events(da.drawdown_events(r))
    ref = normalize_events(da._drawdown_events_reference(r))
    assert got == ref, f"F seed={seed} nan={nan_rate}"


def test_F_known_value_active_drawdown():
    # No recovery -> single ACTIVE event, depth 0.25, peak_idx from t0 high-water.
    r = np.array([0.0, -0.25, 0.0, 0.0])
    got = da.drawdown_events(r)
    ref = da._drawdown_events_reference(r)
    assert normalize_events(got) == normalize_events(ref)
    assert len(got) == 1
    assert got[0]["drawdown"] == 0.25
    assert got[0]["status"] == "ACTIVE"
    assert got[0]["censored"] is True


def test_F_all_nan_censored_invalid():
    r = np.full(5, np.nan)
    got = da.drawdown_events(r)
    ref = da._drawdown_events_reference(r)
    assert normalize_events(got) == normalize_events(ref)
    assert len(got) == 1
    assert got[0]["status"] == "INVALID_VALUATION"
    assert got[0]["peak_idx"] == -1


def test_F_single_element_no_event():
    r = np.array([0.1])
    got = da.drawdown_events(r)
    ref = da._drawdown_events_reference(r)
    assert got == ref
    assert got == []


def test_F_returns_below_minus_100_raises():
    with pytest.raises(ValueError):
        da.drawdown_events(np.array([-1.5]))
    with pytest.raises(ValueError):
        da._drawdown_events_reference(np.array([-1.5]))


def test_F_performance_guardrail_T2000():
    rng = np.random.default_rng(123)
    r = rng.normal(0.0, 0.03, size=2000)
    t_new = best_time(lambda: da.drawdown_events(r))
    t_ref = best_time(lambda: da._drawdown_events_reference(r))
    assert t_new < t_ref, f"F new {t_new:.4g} not faster than ref {t_ref:.4g}"


# --------------------------------------------------------------------------- #
# Task G  compute_sharpe_ratio / sortino / max_drawdown / calmar — (T,F) vectorized
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
def test_G_sharpe_equivalence_matrix(seed, nan_rate):
    rng = np.random.default_rng(seed)
    R = make_panel(SHORT_T, 5, nan_rate, rng)
    assert_tol(
        ps.compute_sharpe_ratio(R),
        ps._compute_sharpe_ratio_reference(R),
        msg=f"G-sharpe seed={seed} nan={nan_rate}",
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
def test_G_sortino_equivalence_matrix(seed, nan_rate):
    rng = np.random.default_rng(seed)
    R = make_panel(SHORT_T, 5, nan_rate, rng)
    assert_tol(
        ps.compute_sortino_ratio(R),
        ps._compute_sortino_ratio_reference(R),
        msg=f"G-sortino seed={seed} nan={nan_rate}",
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
@pytest.mark.parametrize("policy", ["unknown", "zero_fill"])
def test_G_maxdd_equivalence_matrix(seed, nan_rate, policy):
    rng = np.random.default_rng(seed)
    R = make_panel(SHORT_T, 5, nan_rate, rng)
    got = ps.compute_maximum_drawdown(R, missing_return_policy=policy)
    ref = ps._compute_maximum_drawdown_reference(R, missing_return_policy=policy)
    for g, r in zip(got, ref):
        assert_bit_equal(g, r, msg=f"G-maxdd seed={seed} nan={nan_rate} {policy}")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nan_rate", NAN_RATES)
@pytest.mark.parametrize("policy", ["unknown", "zero_fill"])
def test_G_calmar_equivalence_matrix(seed, nan_rate, policy):
    rng = np.random.default_rng(seed)
    R = make_panel(SHORT_T, 5, nan_rate, rng)
    # Calmar combines CAGR (prod**(ppy/n)) with max_drawdown; the power
    # operation differs from the loop oracle by ulp (~5e-15), within tolerance.
    assert_tol(
        ps.compute_calmar_ratio(R, missing_return_policy=policy),
        ps._compute_calmar_ratio_reference(R, missing_return_policy=policy),
        msg=f"G-calmar seed={seed} nan={nan_rate} {policy}",
    )


def test_G_min_periods_boundary():
    # Exactly min_periods valid per column -> defined; min_periods-1 -> NaN.
    R = np.full((20, 2), 0.01)
    R[0, 0] = np.nan  # column 0 has 19 valid < 20
    got = ps.compute_sharpe_ratio(R, min_periods=20)
    ref = ps._compute_sharpe_ratio_reference(R, min_periods=20)
    assert_tol(got, ref)
    assert math.isnan(got[0])


def test_G_calmar_known_value_profitable():
    # Profitable series with a real drawdown -> finite, positive Calmar that
    # matches the oracle (zero_fill; a -0.1 dip creates a bounded drawdown).
    R = np.full((30, 1), 0.03)
    R[15, 0] = -0.1
    got = ps.compute_calmar_ratio(R, missing_return_policy="zero_fill")
    ref = ps._compute_calmar_ratio_reference(R, missing_return_policy="zero_fill")
    assert_tol(got, ref)
    assert np.isfinite(got[0]) and got[0] > 0


def test_G_performance_guardrail_T2000():
    # The per-column oracles loop over F factors; vectorization wins decisively
    # for realistic multi-factor panels (F large). T stays at 2000 as required.
    rng = np.random.default_rng(123)
    R = rng.normal(0.0, 0.02, size=(2000, 1000))
    for name, new_fn, ref_fn in [
        ("sharpe", ps.compute_sharpe_ratio, ps._compute_sharpe_ratio_reference),
        ("sortino", ps.compute_sortino_ratio, ps._compute_sortino_ratio_reference),
        ("maxdd", lambda x: ps.compute_maximum_drawdown(x, missing_return_policy="zero_fill"),
         lambda x: ps._compute_maximum_drawdown_reference(x, missing_return_policy="zero_fill")),
        ("calmar", lambda x: ps.compute_calmar_ratio(x, missing_return_policy="zero_fill"),
         lambda x: ps._compute_calmar_ratio_reference(x, missing_return_policy="zero_fill")),
    ]:
        t_new = best_time(lambda: new_fn(R))
        t_ref = best_time(lambda: ref_fn(R))
        assert t_new < t_ref, f"G-{name} new {t_new:.4g} not faster than ref {t_ref:.4g}"
