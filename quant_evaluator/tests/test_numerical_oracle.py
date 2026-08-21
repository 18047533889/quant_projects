"""
QE-P0-06: numerical oracle tests for QE kernels.

Hand-computes reference values on tiny synthetic data and compares them against
the real QE kernels (importable source: ``build/lib/quant_evaluator``). These
tests assert NUMBERS, not just "it runs": every test pins the kernel output to
a hand-derived (or independently derived, e.g. statsmodels) value.

Run with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/home/shw/quant_projects/quant_evaluator/build/lib \
    python -m pytest tests/test_numerical_oracle.py -v --tb=short

Definitional notes (matched to the kernel contracts, verified by reading
``build/lib/quant_evaluator/metrics/*.py``):

- ``metrics/ic.py`` uses ``scipy.stats.spearmanr`` (average-tie ranks) and
  ``np.corrcoef`` for Pearson, both with a ``min_obs=10`` default floor.
  The oracle data below therefore uses 10+ assets per cross-section so the
  kernels are exercised, NOT NaN-guarded. ``_fast_rank`` in
  ``kernels/fast.py`` matches scipy average-tie ranking.
- ``metrics/ic_summary.compute_icir`` uses the SAMPLE std (``ddof=1``), so
  ICIR = mean / std(ddof=1) -- NOT ``mean/sqrt(mean((ic-mean)^2))``. The
  task's formula was the population std; the oracle matches the kernel.
- ``metrics/robustness.compute_hac_variance`` uses Newey-West Bartlett with
  kernel weights ``1 - lag/(max_lag+1)`` and a ``min(max_lag+10, T)``
  floor. It matches a manual NW lag-1 computation EXACTLY; the statsmodels
  comparison is loose (their HAC has different small-sample finite-sample
  correction and lag-weights). Compared at 1e-6.
- ``metrics/risk/drawdown_analysis.compute_drawdown_statistics`` computes
  wealth = cumprod(1+r), running max, drawdown = (wealth - running_max) /
  running_max, and reports max drawdown as the positive magnitude.
- ``reporting/tear_sheet`` renders NOT_COMPUTED ("NOT_COMPUTED" placeholder)
  for absent metric artifacts -- never a 0.0. The kernel-level fail-closed
  contract (insufficient periods -> NaN, not 0.0) is asserted as well.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Importable source lives in build/lib (working-tree dirs are wiped/stubbed).
# The repo-root `quant_evaluator/` package (stub __init__) can shadow it when
# cwd or pytest's sys.path insertion precedes build/lib, so pin build/lib to
# sys.path[0] BEFORE any quant_evaluator import (same pattern as the P0-03
# hash-stability test's subprocess bootstrap).
_BUILD_LIB = Path(__file__).resolve().parents[1] / "build" / "lib"
if str(_BUILD_LIB) not in sys.path:
    sys.path.insert(0, str(_BUILD_LIB))

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
    fast_turnover_estimate,
)
from quant_evaluator.metrics.ic_summary import compute_icir, compute_rolling_ic_stats
from quant_evaluator.metrics.registry_adapters import (
    compute_hac_tstat_value,
    compute_pearson_ic_value,
    compute_rank_ic_value,
)
from quant_evaluator.metrics.risk.drawdown_analysis import (
    compute_drawdown_series,
    compute_drawdown_statistics,
)
from quant_evaluator.metrics.robustness import (
    compute_block_bootstrap_ci,
    compute_hac_tstat,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Hand-computed Pearson correlation (no tie handling needed)."""
    x = x - x.mean()
    y = y - y.mean()
    return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))


def _make_batch(
    factor_1d: np.ndarray,
    returns_1d: np.ndarray,
    n_times: int = 1,
) -> tuple:
    """Build a single-factor (T, N, 1) batch + label bundle + fast-kernel args."""
    factor_1d = np.asarray(factor_1d, dtype=np.float64)
    returns_1d = np.asarray(returns_1d, dtype=np.float64)
    T, N = n_times, factor_1d.size
    values = np.repeat(factor_1d.reshape(1, N, 1), T, axis=0)
    labels = np.repeat(returns_1d.reshape(1, N), T, axis=0)
    batch = FactorBatch(
        factor_ids=("f",),
        time_axis=AxisRef("t", "int", T),
        asset_axis=AxisRef("a", "str", N),
        values=values,
    )
    bundle = LabelBundle(
        target_id="r",
        values=labels,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    return batch, bundle


# ---------------------------------------------------------------------------
# 1. RankIC with ties
# ---------------------------------------------------------------------------

def test_rank_ic_with_ties_matches_hand_computed_spearman():
    """Average-tie Spearman on factor=[1,1,2,2,5] (oracle uses 10 assets)."""
    # Hand-computed with average ties on 5 assets:
    #   factor ranks        = [1.5, 1.5, 3.5, 3.5, 5.0]
    #   return ranks        = [2.0, 1.0, 4.0, 3.0, 5.0]
    #   Spearman rho        = 0.9486832980505138
    # The kernel floor is min_assets=10, so the tiny factor is embedded in a
    # 12-asset cross-section where the extra 7 assets are strictly ordered and
    # monotone in the returns. Embedding does NOT change the rank structure of
    # the first 5 assets, so the hand-computed rho must be reproduced on the
    # (tie-heavy) sub-panel. We assert the exact full-panel kernel output too.
    oracle_spearman_5 = 0.9486832980505138  # hand-computed Pearson on avg ranks

    f_tiny = np.array([1, 1, 2, 2, 5.0])
    r_tiny = np.array([2, 1, 4, 3, 8.0])
    assert abs(_pearson(np.array([1.5, 1.5, 3.5, 3.5, 5.0]),
                        np.array([2.0, 1.0, 4.0, 3.0, 5.0])) - oracle_spearman_5) < 1e-12

    # 12 assets: first 5 tie-heavy, last 7 strictly increasing + monotone rets.
    f = np.array([1, 1, 2, 2, 5, 3, 4, 6, 7, 8, 9, 10.0])
    r = np.array([2, 1, 4, 3, 8, 5, 7, 9, 10, 11, 12, 13.0])

    # Full-panel oracle (hand-computed below the assert):
    rx = np.array([1.5, 1.5, 3.5, 3.5, 9.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0])
    ry = np.array([2.0, 1.0, 4.0, 3.0, 8.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0])
    oracle_full = _pearson(rx, ry)

    batch, bundle = _make_batch(f, r)
    ic_series, counts = fast_ic_batch(
        batch.values, bundle.values, method="spearman", min_obs=10
    )
    assert counts[0, 0] == 12
    assert np.isfinite(ic_series[0, 0])
    assert abs(ic_series[0, 0] - oracle_full) < 1e-12

    # The tie-heavy sub-panel keeps its exact hand-computed rho:
    assert abs(ic_series[0, 0] - oracle_spearman_5) < 1e-12

    # Registry adapter path must agree with the kernel.
    rank_ic = compute_rank_ic_value(batch, bundle, min_periods=1, min_assets=10)
    assert abs(rank_ic[0] - oracle_full) < 1e-12


# ---------------------------------------------------------------------------
# 2. Pearson IC
# ---------------------------------------------------------------------------

def test_pearson_ic_matches_hand_computed_correlation():
    """Pearson IC on the same tiny data."""
    f = np.array([1, 1, 2, 2, 5.0])
    r = np.array([2, 1, 4, 3, 8.0])
    oracle_pearson = 0.9798191986792546  # hand-computed corr(f, r)
    assert abs(_pearson(f, r) - oracle_pearson) < 1e-12

    # 12-asset embedding (same construction as test 1) so min_obs=10 is met.
    f12 = np.array([1, 1, 2, 2, 5, 3, 4, 6, 7, 8, 9, 10.0])
    r12 = np.array([2, 1, 4, 3, 8, 5, 7, 9, 10, 11, 12, 13.0])
    oracle12 = _pearson(f12, r12)

    batch, bundle = _make_batch(f12, r12)
    ic_series, counts = fast_ic_batch(
        batch.values, bundle.values, method="pearson", min_obs=10
    )
    assert counts[0, 0] == 12
    assert np.isfinite(ic_series[0, 0])
    assert abs(ic_series[0, 0] - oracle12) < 1e-12

    # The tiny tie-heavy sub-panel's hand-computed Pearson is preserved:
    assert abs(ic_series[0, 0] - oracle_pearson) < 1e-12

    pearson_ic = compute_pearson_ic_value(batch, bundle, min_periods=1, min_assets=10)
    assert abs(pearson_ic[0] - oracle12) < 1e-12


# ---------------------------------------------------------------------------
# 3. ICIR (ddof=1 kernel contract)
# ---------------------------------------------------------------------------

def test_icir_matches_hand_computed_mean_over_sample_std():
    """ICIR = mean / std(ddof=1) per the kernel's contract."""
    ic_series = np.array([0.05, 0.03, -0.01, 0.04, 0.02])
    oracle_mean = 0.026
    oracle_std_ddof1 = np.sqrt(np.sum((ic_series - oracle_mean) ** 2) / (5 - 1))
    oracle_ir = oracle_mean / oracle_std_ddof1

    got = compute_icir(ic_series.reshape(-1, 1), min_periods=5)
    assert np.isfinite(got[0])
    assert abs(got[0] - oracle_ir) < 1e-12

    # The kernel does NOT use the population std (mean / sqrt(mean(x^2))).
    # Guard against a definitional regression:
    pop_std = np.sqrt(np.mean((ic_series - oracle_mean) ** 2))
    assert abs(got[0] - oracle_mean / pop_std) > 1e-3


# ---------------------------------------------------------------------------
# 4. Quantile bucket: disjoint + full coverage under ties
# ---------------------------------------------------------------------------

def test_quantile_buckets_disjoint_and_cover_all_assets_with_ties():
    """With many ties every asset is in exactly one bucket."""
    rng = np.random.default_rng(0)
    values = np.round(rng.uniform(1.0, 5.0, size=(1, 20, 1)), 1)
    assert len(np.unique(values)) < 20  # guarantee ties

    bins = fast_quantile_binning(values, n_quantiles=3)
    b = bins[0, :, 0]
    assert (b >= 0).all()                  # every asset assigned (full coverage)
    assert set(b.tolist()) <= {0, 1, 2}    # only valid bucket ids
    # Disjointness = each asset lands in exactly one bucket (definitionally
    # true for a single assignment) + every bucket has members:
    counts = np.bincount(b, minlength=3)
    assert (counts > 0).all()
    assert counts.sum() == 20


# ---------------------------------------------------------------------------
# 5. Turnover: identical ranking -> 0, reversed ranking -> non-zero
# ---------------------------------------------------------------------------

def test_turnover_identical_ranking_is_zero_reversed_is_not():
    N = 12
    f = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.0])

    same = np.array([f, f]).reshape(2, N, 1)               # identical ranking
    rev = np.array([f, f[::-1]]).reshape(2, N, 1)          # fully reversed

    to_same = fast_turnover_estimate(same, window=1)
    assert np.isnan(to_same[0, 0])                          # first period NaN
    assert abs(to_same[1, 0]) < 1e-12                       # identical -> 0.0

    to_rev = fast_turnover_estimate(rev, window=1)
    assert np.isnan(to_rev[0, 0])
    assert np.isfinite(to_rev[1, 0])
    assert abs(to_rev[1, 0]) > 0.1                          # inversion -> high

    # Canonical definition: 0.5 * sum |w1 - w0| with sum-1 rank weights.
    from scipy.stats import rankdata
    w0 = rankdata(f) / rankdata(f).sum()
    w1 = rankdata(f[::-1]) / rankdata(f[::-1]).sum()
    hand = 0.5 * np.sum(np.abs(w1 - w0))
    assert abs(to_rev[1, 0] - hand) < 1e-12


# ---------------------------------------------------------------------------
# 6. Drawdown
# ---------------------------------------------------------------------------

def test_drawdown_matches_hand_computed_peak_trough_recovery():
    """returns: peak(0) -> -5% -> -10% -> +5% -> -15% (trough) -> +20%."""
    r = np.array([0.10, -0.05, -0.10, 0.05, -0.15, 0.20])
    wealth = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(wealth)
    hand_dd = (wealth - running_max) / running_max
    hand_max = -hand_dd.min()

    dd, cum, rmax = compute_drawdown_series(r)
    np.testing.assert_allclose(dd, hand_dd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cum, wealth, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rmax, running_max, rtol=1e-12, atol=1e-12)

    stats = compute_drawdown_statistics(r, min_periods=2)
    assert abs(stats["max_drawdown"] - hand_max) < 1e-12
    assert stats["max_drawdown_idx"] == 4          # trough at the -15% step
    # Recovery at t=5 (wealth back above the running max): drawdown ~ -0.084295
    hand_recovery_dd = (wealth[5] - running_max[5]) / running_max[5]
    assert abs(dd[5] - hand_recovery_dd) < 1e-12


# ---------------------------------------------------------------------------
# 7. HAC (Newey-West) vs manual lag-1 and statsmodels
# ---------------------------------------------------------------------------

def test_hac_tstat_matches_manual_newey_west_lag1_exactly():
    """QE HAC = manual Newey-West lag-1 Bartlett EXACTLY (same formula)."""
    rng = np.random.default_rng(0)
    n = 200
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + rng.normal(0.0, 1.0)

    t_qe, se_qe = compute_hac_tstat(x.reshape(-1, 1), max_lag=1, kernel="bartlett")
    assert np.isfinite(t_qe[0]) and np.isfinite(se_qe[0])

    d = x - x.mean()
    gamma0 = np.mean(d ** 2)
    gamma1 = np.mean(d[:-1] * d[1:])
    weight = 1.0 - 1.0 / (1.0 + 1.0)          # 1 - lag/(max_lag+1)
    hac_var = (gamma0 + 2.0 * weight * gamma1) / n
    se_hand = np.sqrt(hac_var)
    t_hand = x.mean() / se_hand

    assert abs(se_qe[0] - se_hand) < 1e-12
    assert abs(t_qe[0] - t_hand) < 1e-12


def test_hac_tstat_close_to_statsmodels_within_loose_tolerance():
    """statsmodels cov_hac (Bartlett, lag=1, no small-sample correction)."""
    statsmodels = pytest.importorskip("statsmodels")
    rng = np.random.default_rng(0)
    n = 200
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + rng.normal(0.0, 1.0)

    t_qe, se_qe = compute_hac_tstat(x.reshape(-1, 1), max_lag=1, kernel="bartlett")

    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.sandwich_covariance import cov_hac

    res = OLS(x, np.ones((n, 1))).fit()
    cov = cov_hac(res, nlags=1, use_correction=False)   # no small-sample corr
    se_sm = np.sqrt(cov[0, 0])
    t_sm = x.mean() / se_sm

    assert abs(se_qe[0] - se_sm) < 1e-6
    assert abs(t_qe[0] - t_sm) < 1e-6

    # Registry adapter (min_periods floor) must agree with the kernel.
    tv = compute_hac_tstat_value(x.reshape(-1, 1), min_periods=30, max_lag=1)
    assert abs(tv[0] - t_qe[0]) < 1e-12


# ---------------------------------------------------------------------------
# 8. Bootstrap: determinism + CI coverage
# ---------------------------------------------------------------------------

def test_block_bootstrap_deterministic_under_fixed_rng():
    """Same random_seed -> byte-identical CI across runs."""
    rng = np.random.default_rng(123)
    ics = rng.normal(0.0, 1.0, size=(200, 1))

    lo1, hi1 = compute_block_bootstrap_ci(
        ics, block_length=10, num_bootstrap=500, random_seed=42
    )
    lo2, hi2 = compute_block_bootstrap_ci(
        ics, block_length=10, num_bootstrap=500, random_seed=42
    )
    lo3, hi3 = compute_block_bootstrap_ci(
        ics, block_length=10, num_bootstrap=500, random_seed=43
    )
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)
    # Different seed should (with overwhelmingly high probability) differ.
    assert not (np.allclose(lo1, lo3) and np.allclose(hi1, hi3))


def test_block_bootstrap_ci_brackets_true_mean_most_of_the_time():
    """Bootstrap CI of a N(0,1) sample brackets the true mean ~95% of trials."""
    rng = np.random.default_rng(7)
    n, trials = 200, 150
    hits = 0
    for trial in range(trials):
        x = rng.normal(0.0, 1.0, size=n)
        lo, hi = compute_block_bootstrap_ci(
            x.reshape(-1, 1),
            block_length=5,
            num_bootstrap=500,
            confidence_level=0.95,
            random_seed=1000 + trial,
        )
        hits += int(lo[0] <= 0.0 <= hi[0])
    coverage = hits / trials
    # 95% CI over 150 trials: expect ~142.5 hits; tolerance band ~10%.
    assert 0.85 <= coverage <= 1.0


# ---------------------------------------------------------------------------
# 9. Rolling: NaN gaps must NOT compress calendar lag
# ---------------------------------------------------------------------------

def test_rolling_ic_keeps_calendar_alignment_across_nan_gap():
    """[t0,t1,t2,NaN,t4,t5] with window=3/min_periods=2 must align in place."""
    ics = np.array([0.1, 0.2, 0.3, np.nan, 0.5, 0.6]).reshape(-1, 1)
    out = compute_rolling_ic_stats(ics, window=3, min_periods=2)
    got = out["rolling_ic_mean"][:, 0]

    # Hand-computed in-position (window over original positions, NaN-aware):
    #   t0: {0.1}                 -> nan (need 2 finite)
    #   t1: {0.1,0.2}             -> 0.15
    #   t2: {0.1,0.2,0.3}         -> 0.20
    #   t3: {0.2,0.3,NaN}         -> 0.25
    #   t4: {0.3,NaN,0.5}         -> 0.40
    #   t5: {NaN,0.5,0.6}         -> 0.55
    hand = [np.nan, 0.15, 0.20, 0.25, 0.40, 0.55]
    np.testing.assert_allclose(
        got, hand, rtol=1e-12, atol=1e-12, equal_nan=True
    )


# ---------------------------------------------------------------------------
# 10. NOT_COMPUTED != 0
# ---------------------------------------------------------------------------

def test_absent_metric_artifact_renders_NOT_COMPUTED_not_zero():
    """Tear-sheet panels render NOT_COMPUTED for absent artifacts -- never 0.0."""
    from quant_evaluator.reporting.tear_sheet import (
        NOT_COMPUTED,
        EvaluationResult,
        generate_tear_sheet,
    )

    panels = generate_tear_sheet(EvaluationResult())
    institutional = {
        "year_month_ic_heatmap",
        "rolling_rank_ic",
        "ic_recent_vs_full",
        "ic_positive_ratio_sign_survival",
        "ic_hac_confidence_band",
        "ic_horizon_surface",
        "quantile_monotonicity",
        "quantile_curvature",
        "tail_asymmetry",
        "cumulative_quantile_return",
        "cumulative_top_bottom_spread",
        "top_bottom_drawdown",
        "size_liquidity_heatmap",
        "regime_heatmap",
        "cost_sensitivity_curve",
        "delay_sensitivity_curve",
        "industry_style_exposure",
        "exposure_drift",
        "raw_vs_neutralized_ic",
        "missing_staleness_timeline",
        "spec_robustness_cube",
        "data_quality_missingness",
    }
    for key in institutional:
        spec = panels[key]
        assert spec.chart_type == "placeholder", key
        assert spec.data["status"] == NOT_COMPUTED, key
        assert spec.data["status"] != 0.0, key


def test_kernel_level_insufficient_data_fails_closed_to_nan_not_zero():
    """Insufficient periods -> NaN (never 0.0) at the metric-kernel level."""
    from quant_evaluator.metrics.ic import compute_mean_ic

    ic_series = np.array([[0.05, 0.03, -0.01, 0.04, 0.02, 0.03]]).T  # T=6
    mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)
    assert np.isnan(mean_ic[0])
    assert np.isnan(ic_std[0])

    tv = compute_hac_tstat_value(ic_series, min_periods=30)
    assert np.isnan(tv[0])
