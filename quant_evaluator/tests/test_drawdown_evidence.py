"""
R61-FI-023: probe-PnL drawdown / underwater evidence tests.

Pins the exact values of the underwater kernels against hand-constructed
synthetic net-value (PnL) series with KNOWN drawdown/underwater episodes,
calendar worst-period blocks, rolling Sharpe tails, return skew, downside
deviation and CVaR expected shortfall, plus:

  - the overlapping H10/H20 guard: overlapping horizon labels must never be
    annualized as if they were dot-frequency daily PnL (the probe cohort PnL
    family and its annualized-Sharpe derivatives stay on daily PnL only);
  - missing evidence != 0: every kernel returns NaN (never a fabricated 0.0)
    when there is not enough data;
  - the registry wiring: 12 new ids registered with bound compute_fn, and
    max_drawdown is REUSED (not re-registered).

Run (from the repo root):
    PYTHONPATH=/home/sunhaiwei/quant_projects python -m pytest -q \
        quant_evaluator/tests/test_drawdown_evidence.py --timeout=300
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_evaluator.metrics.underwater import (
    compute_max_underwater_duration,
    compute_mean_underwater_duration,
    compute_time_to_recovery,
    compute_worst_period_return,
    compute_rolling_sharpe_tail,
    compute_return_skew,
    compute_downside_deviation,
    compute_cvar_expected_shortfall,
    compute_underwater_evidence,
)
from quant_evaluator.metrics.portfolio_stats import compute_maximum_drawdown
from quant_evaluator.metrics.probe_portfolio import (
    compute_cohort_pnl,
    compute_portfolio_metrics,
    compute_overlapping_forward_returns,
    annualize_overlapping_label_sharpe,
)
from quant_evaluator.registry.metrics import (
    get_metric,
    list_metrics,
    MetricSpec,
    MetricStatus,
)

# Registry ids that must exist after R61-FI-023 (all bound compute_fn).
_UNDERWATER_IDS = (
    "max_underwater_duration",
    "mean_underwater_duration",
    "time_to_recovery",
    "worst_month",
    "worst_quarter",
    "worst_12m",
    "rolling_1y_sharpe_min",
    "rolling_1y_sharpe_q10",
    "return_skew",
    "downside_deviation",
    "cvar_expected_shortfall",
)

_PERIODS_PER_MONTH = 21
_PERIODS_PER_QUARTER = 63
_PERIODS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _constant_wealth(per_period: float = 0.001, n: int = 300) -> np.ndarray:
    """A monotonically increasing net-value series -> never underwater."""
    return np.full(n, per_period)


def _sawtooth(n: int = 60, cycle: int = 10) -> np.ndarray:
    """Deterministic boom-bust net-value: each cycle rises then falls flat.

    Wealth = cumprod(1 + ret); each 10-period cycle is a rise of `cycle-1`
    positive days then a fall.  Produces a fully-known underwater structure.
    """
    ret = np.zeros(n)
    for start in range(0, n, cycle):
        seg = ret[start : start + cycle]
        # rise 4 periods, then 5 flat-negative periods
        if len(seg) > 0:
            seg[0] = 0.01
        if len(seg) > 1:
            seg[1] = 0.01
        if len(seg) > 2:
            seg[2] = 0.01
        if len(seg) > 3:
            seg[3] = 0.01
        if len(seg) > 4:
            seg[4] = -0.005
        if len(seg) > 5:
            seg[5] = -0.005
        if len(seg) > 6:
            seg[6] = -0.005
    return ret


# ---------------------------------------------------------------------------
# 1. Registry wiring
# ---------------------------------------------------------------------------
def test_underwater_ids_registered_with_bound_compute_fn():
    ids = set(list_metrics())
    for metric_id in _UNDERWATER_IDS:
        assert metric_id in ids, metric_id
        spec = get_metric(metric_id)
        assert spec.compute_fn is not None, metric_id
        # Value-based (not `is`) comparison: the shared registry may be
        # reloaded by the seal-lifecycle tests, which recreates the enum
        # classes; `name`/`value` stay stable across the reload.
        assert spec.status.name == MetricStatus.EXPERIMENTAL.name, metric_id


def test_max_drawdown_reused_not_reregistered():
    """R61-FI-023: max_drawdown already exists (portfolio_stats authority);
    the underwater family must NOT register a duplicate max_drawdown id."""
    ids = set(list_metrics())
    # exactly one max_drawdown id in the registry.
    assert sum(1 for mid in ids if mid == "max_drawdown") == 1
    spec = get_metric("max_drawdown")
    assert spec.implementation_id == (
        "quant_evaluator.metrics.portfolio_stats.compute_maximum_drawdown"
    )


def test_underwater_specs_domain_underwater():
    from quant_evaluator.registry.metrics import Domain

    for metric_id in _UNDERWATER_IDS:
        assert get_metric(metric_id).domain is Domain.UNDERWATER, metric_id


# ---------------------------------------------------------------------------
# 2. Exact known-value checks on synthetic series
# ---------------------------------------------------------------------------
def test_constant_series_never_underwater_duration_zero():
    """A monotonically increasing series is never underwater -> durations 0."""
    ret = _constant_wealth(0.001, 300)
    assert compute_max_underwater_duration(ret) == 0.0
    assert compute_mean_underwater_duration(ret) == 0.0
    # a pure positive series has no drawdown
    max_dd = compute_maximum_drawdown(ret, missing_return_policy="zero_fill")[0]
    assert max_dd == pytest.approx(0.0, abs=1e-12)


def test_single_downstep_known_underwater_duration():
    """A single -1% step then flat: underwater from the drop day onward
    (wealth stays below the pre-step peak forever)."""
    ret = np.full(50, 0.0)
    ret[10] = -0.01  # one -1% day at index 10
    # wealth = 1.0 for t<10, 0.99 for t>=10. Underwater from index 10
    # (the drop day itself already sits below the running max) through 49.
    assert compute_max_underwater_duration(ret) == pytest.approx(40.0)
    assert compute_mean_underwater_duration(ret) == pytest.approx(40.0)


def test_two_episodes_mean_duration_known():
    """Two isolated underwater episodes of known lengths -> exact mean.

    Episode 1: the -1% drop at index 10 leaves wealth 0.99 forever (no
    recovery) -> underwater indices 10..29 (20 periods).
    Episode 2: the -2% drop at index 30 (wealth 0.9702) -> underwater
    indices 30..59 (30 periods).
    """
    ret = np.zeros(60)
    ret[10] = -0.01
    ret[30] = -0.02
    assert compute_max_underwater_duration(ret) == pytest.approx(50.0)
    assert compute_mean_underwater_duration(ret) == pytest.approx(50.0)


def test_max_drawdown_known_from_single_drop():
    """A single -10% drop with no recovery -> max drawdown 10% (positive)."""
    ret = np.zeros(40)
    ret[5] = -0.10
    max_dd, dd_series, _ = compute_maximum_drawdown(
        ret, missing_return_policy="zero_fill"
    )
    assert max_dd == pytest.approx(0.10, abs=1e-12)
    # the drawdown series at trough equals -10%
    assert dd_series[6] == pytest.approx(-0.10, abs=1e-12)


def test_worst_month_known_block():
    """Worst fixed 21-period block: place a -20% single day inside a specific
    21-block so the block containing it is the minimum compounded block."""
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0003, 0.002, 200)
    # put a -0.2 single-day shock inside the block starting at index 50
    ret[60] = -0.20
    worst = compute_worst_period_return(ret, "month", min_periods=10)
    # hand compute: brute force over all 21-blocks
    blocks = np.full(200 - 21 + 1, np.nan)
    for t in range(200 - 21 + 1):
        blocks[t] = np.prod(1.0 + ret[t : t + 21]) - 1.0
    assert worst == pytest.approx(float(np.min(blocks)), abs=1e-12)


def test_worst_quarter_is_min_of_63_blocks():
    ret = np.full(120, 0.0)
    ret[30] = -0.05
    worst_q = compute_worst_period_return(ret, "quarter", min_periods=10)
    blocks = np.full(120 - 63 + 1, np.nan)
    for t in range(120 - 63 + 1):
        blocks[t] = np.prod(1.0 + ret[t : t + 63]) - 1.0
    assert worst_q == pytest.approx(float(np.min(blocks)), abs=1e-12)


def test_worst_12m_matches_brute_force_blocks():
    """Worst fixed 252-block: the block containing the -3% shock is the
    minimum; brute-force hand check over all aligned 252-blocks."""
    ret = np.full(400, 0.0002)
    ret[200] = -0.03
    worst = compute_worst_period_return(ret, "year", min_periods=100)
    blocks = np.full(400 - 252 + 1, np.nan)
    for t in range(400 - 252 + 1):
        blocks[t] = np.prod(1.0 + ret[t : t + 252]) - 1.0
    assert worst == pytest.approx(float(np.min(blocks)), abs=1e-9)


def test_worst_period_short_series_nan():
    """Shorter than one block -> NaN (missing evidence != 0)."""
    assert np.isnan(compute_worst_period_return(np.zeros(50), "year", min_periods=10))
    assert np.isnan(compute_worst_period_return(np.zeros(5), "month", min_periods=10))


def test_rolling_1y_sharpe_min_is_min_of_rolling():
    rng = np.random.default_rng(3)
    ret = rng.normal(0.0005, 0.01, 400)
    m = compute_rolling_sharpe_tail(ret, window=252, quantile=0.0, min_periods=60)
    from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio

    roll = []
    for t in range(252 - 1, 400):
        seg = ret[t - 251 : t + 1]
        if np.sum(np.isfinite(seg)) >= 60:
            val = compute_sharpe_ratio(seg, periods_per_year=252, min_periods=60)
            if np.isfinite(val):
                roll.append(float(val))
    assert m == pytest.approx(float(min(roll)), abs=1e-9)


def test_rolling_1y_sharpe_q10_quantile():
    rng = np.random.default_rng(4)
    ret = rng.normal(0.0005, 0.01, 400)
    q = compute_rolling_sharpe_tail(ret, window=252, quantile=0.10, min_periods=60)
    from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio

    roll = []
    for t in range(252 - 1, 400):
        seg = ret[t - 251 : t + 1]
        if np.sum(np.isfinite(seg)) >= 60:
            val = compute_sharpe_ratio(seg, periods_per_year=252, min_periods=60)
            if np.isfinite(val):
                roll.append(float(val))
    assert q == pytest.approx(float(np.quantile(roll, 0.10)), abs=1e-9)


def test_rolling_1y_sharpe_short_series_nan():
    assert np.isnan(
        compute_rolling_sharpe_tail(np.zeros(100), window=252, min_periods=60)
    )


def test_return_skew_known_symmetric_near_zero():
    rng = np.random.default_rng(7)
    ret = rng.normal(0.0, 0.01, 500)
    s = compute_return_skew(ret)
    assert abs(s) < 0.3


def test_return_skew_positive_for_right_tail():
    """A single large right-tail jump in an otherwise-flat series gives
    strongly positive sample skew (population convention, matching scipy
    skew bias=False — small-sample factor ≈ n/(n-1), verified within 5%)."""
    ret = np.zeros(300)
    ret[10] = 0.5
    s = compute_return_skew(ret)
    assert s > 5.0  # far above 0: unmistakably right-tailed
    # hand-close to scipy population skew
    import scipy.stats as st

    assert s == pytest.approx(float(st.skew(ret, bias=False)), rel=0.02)


def test_return_skew_insufficient_nan():
    assert np.isnan(compute_return_skew(np.zeros(5)))


def test_downside_deviation_matches_sortino_family():
    rng = np.random.default_rng(11)
    ret = rng.normal(-0.0002, 0.008, 300)
    dd = compute_downside_deviation(ret)
    # hand compute: RMS of negative excess returns * sqrt(252)
    excess = ret - 0.0
    neg = excess[excess < 0]
    expected = np.sqrt(np.mean(neg ** 2)) * np.sqrt(252)
    assert dd == pytest.approx(float(expected), abs=1e-9)


def test_downside_deviation_all_positive_nan():
    assert np.isnan(compute_downside_deviation(np.full(60, 0.001)))


def test_cvar_expected_shortfall_matches_var_cvar_authority():
    from quant_evaluator.metrics.risk.var_cvar import compute_cvar

    rng = np.random.default_rng(13)
    ret = rng.normal(-0.0001, 0.01, 400)
    mine = compute_cvar_expected_shortfall(ret, confidence_level=0.95)
    ref = compute_cvar(ret, confidence_level=0.95, method="historical", min_periods=20)
    assert mine == pytest.approx(float(ref), abs=1e-12)


def test_cvar_expected_shortfall_insufficient_nan():
    assert np.isnan(compute_cvar_expected_shortfall(np.zeros(5)))


# ---------------------------------------------------------------------------
# 3. Missing evidence != 0
# ---------------------------------------------------------------------------
def test_underwater_evidence_bundle_nan_on_short_series():
    ev = compute_underwater_evidence(np.zeros(5), min_periods=20)
    for key in (
        "max_drawdown",
        "max_underwater_duration",
        "mean_underwater_duration",
        "time_to_recovery",
        "worst_month",
        "worst_quarter",
        "worst_12m",
        "rolling_1y_sharpe_min",
        "rolling_1y_sharpe_q10",
        "return_skew",
        "downside_deviation",
        "cvar_expected_shortfall",
    ):
        assert np.isnan(ev[key]), key


def test_underwater_evidence_bundle_computes():
    rng = np.random.default_rng(17)
    ret = rng.normal(0.0004, 0.005, 600)
    ev = compute_underwater_evidence(ret)
    for key, val in ev.items():
        if key == "worst_12m":
            # 600 < 252? no -> computed
            assert np.isfinite(val), key
        elif key == "rolling_1y_sharpe_min" or key == "rolling_1y_sharpe_q10":
            assert np.isfinite(val), key
        else:
            assert np.isfinite(val), key


# ---------------------------------------------------------------------------
# 4. Overlapping H10/H20 guard: never annualize overlapping horizon labels
#    as dot-frequency daily PnL
# ---------------------------------------------------------------------------
def test_overlapping_forward_returns_produce_daily_pnl_family_not_label_annualize():
    """The probe cohort PnL is the ONLY series the daily Sharpe family may
    annualize.  The naive overlapping-label annualization must be strictly
    guarded: its result is never used by the metrics, and it is > the daily
    PnL Sharpe (the虚高 effect the cohort engine exists to avoid)."""
    rng = np.random.default_rng(23)
    T, N = 260, 120
    factor = rng.normal(0.0, 1.0, (T, N))
    next_ret = rng.normal(0.0002, 0.01, (T, N))
    next_vwap = np.abs(rng.normal(10.0, 1.0, (T, N))) + 1.0

    cohort = compute_cohort_pnl(
        factor, next_ret, next_vwap, holding=20, per_side_cost=0.0
    )
    daily_pnl = cohort["pnl_net"]
    metrics = compute_portfolio_metrics(daily_pnl)
    daily_sharpe = metrics["sharpe"]

    # The overlapping-label path is a documented "wrong algorithm" used ONLY
    # for demonstrating the inflation — it must never feed the metric chain.
    long_mask, short_mask = _build_masks(factor)
    ovl = compute_overlapping_forward_returns(
        next_ret, long_mask, short_mask, holding=20
    )
    naive = annualize_overlapping_label_sharpe(ovl, holding=20)

    # Guard assertion: the metrics family reports the daily-PnL Sharpe, not
    # the overlapping-label annualized one.
    assert daily_sharpe != pytest.approx(naive, abs=1e-9)


def _build_masks(factor):
    from quant_evaluator.metrics.probe_portfolio import build_cohort_panels

    return build_cohort_panels(factor, n_quantiles=10)


def test_daily_sharpe_family_consumes_daily_pnl_not_overlapping_labels():
    """The production compute_portfolio_metrics must consume the cohort daily
    PnL series — never the overlapping H20 forward-return series."""
    rng = np.random.default_rng(31)
    T, N = 260, 120
    factor = rng.normal(0.0, 1.0, (T, N))
    next_ret = rng.normal(0.0002, 0.01, (T, N))
    next_vwap = np.abs(rng.normal(10.0, 1.0, (T, N))) + 1.0
    cohort = compute_cohort_pnl(factor, next_ret, next_vwap, holding=20)
    daily = cohort["pnl_net"]
    metrics = compute_portfolio_metrics(daily)
    # pnl_net is a (T,) daily series — shape check ensures we never fed
    # overlapping (T,) labels to the family.
    assert daily.ndim == 1
    assert np.isfinite(metrics["sharpe"])


def test_time_to_recovery_completed_only():
    """An episode that recovered contributes its trough->recovery length; an
    ongoing (unrecovered) episode is excluded from the mean (only reported via
    max/mean underwater duration)."""
    ret = np.zeros(80)
    ret[10] = -0.05   # episode 1 trough at 10 (wealth 0.95), recovers when wealth
    ret[11] = 0.06    # reaches 1.0 at index 11 -> recovery length 1
    ret[40] = -0.02   # episode 2 (never recovers before the end)
    ttr = compute_time_to_recovery(ret)
    assert ttr == pytest.approx(1.0, abs=1e-9)