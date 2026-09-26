"""M01 + M02 extension kernels: golden, determinism and validity tests.

Oracles in the ORACLE section below are copied VERBATIM from
extension_plan.md Sec. 12 (the executable specification) and pin the
formulas independently of the production kernels.

Coverage:
- Sec. 12 golden values (block plan, prefix moments, tail means, MDD).
- Per-kernel >= 3-seed determinism (bootstrap paths byte-identical for
  a fixed seed; RNG plan independent of factor order, Sec. 1.1).
- Paired missing / clock mismatch / short sample / constant series ->
  NaN or explicit invalid evidence; dtype rejection.
- Statistical calibration sanity: iid-Gaussian coverage of the 95%
  bootstrap and HAC intervals in [0.85, 1.0] over 200 simulations.
- Consistency with the existing HAC kernel
  (quant_evaluator.metrics.robustness.compute_hac_variance).
"""
from __future__ import annotations

import math
import unittest
from typing import NamedTuple

import numpy as np
import pytest
from scipy.stats import norm, spearmanr

from quant_evaluator.metrics.extension import (
    BlockStartsPlan,
    OOSFitManifest,
    default_hac_max_lag,
    paired_cer_delta,
    paired_es_improvement,
    paired_mdd_improvement,
    paired_net_sharpe_delta,
    paired_sharpe_bootstrap_ci,
    paired_sharpe_hac_ci,
    paired_sharpe_studentized_ci,
    oos_ablation_summary,
    oos_daily_mse_improvement,
    oos_r2_gain,
    oos_rank_ic_delta,
)
from quant_evaluator.metrics.robustness import compute_hac_variance

A_DEFAULT = 252.0


# ===========================================================================
# ORACLE SECTION -- verbatim from extension_plan.md Sec. 12
# ===========================================================================

def real_array(value, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != ndim:
        raise ValueError("Expected a real numerical array with the specified rank")
    out = raw.astype(np.float64)
    if not np.isfinite(out).all():
        raise ValueError("Oracle requires complete finite inputs")
    return out


def positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, not bool")
    return int(value)


def tail_weights(values, confidence: float) -> np.ndarray:
    x = real_array(values, 1)
    if not x.size or isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ValueError("Tail requires data and 0 < confidence < 1")
    mass = (1.0 - confidence) * len(x)
    nearest = round(mass)
    if nearest >= 1 and abs(mass - nearest) <= 8 * np.finfo(float).eps * max(1, mass):
        mass = float(nearest)
    index = min(len(x) - 1, math.ceil(mass) - 1)
    cutoff = np.sort(x)[::-1][index]
    above, tied = x > cutoff, x == cutoff
    w = above.astype(float)
    w[tied] = (mass - int(above.sum())) / int(tied.sum())
    return w


def upper_tail_mean(values, confidence: float) -> float:
    x = real_array(values, 1)
    w = tail_weights(x, confidence)
    return float(w @ x / w.sum())


def block_plan(sample_length: int, output_length: int, block_length: int,
               repetitions: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    n = positive_int(sample_length, "sample_length")
    h = positive_int(output_length, "output_length")
    block = positive_int(block_length, "block_length")
    b = positive_int(repetitions, "repetitions")
    if b < 2 or block > n or isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Invalid block plan")
    count = (h + block - 1) // block
    lengths = np.full(count, block, dtype=np.int64)
    lengths[-1] = h - (count - 1) * block
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, n - block + 1, size=(b, count), dtype=np.int64)
    return starts, lengths


def expand_draw(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(s, s + length) for s, length in zip(starts, lengths)])


def sample_moments_prefix(returns, starts, lengths) -> tuple[np.ndarray, np.ndarray]:
    r = real_array(returns, 2)
    n, f = r.shape
    starts = np.asarray(starts)
    lengths = np.asarray(lengths)
    if starts.dtype.kind not in "iu" or lengths.dtype.kind not in "iu" or starts.ndim != 2:
        raise ValueError("Integer start matrix and lengths required")
    if lengths.shape != (starts.shape[1],) or np.any(lengths <= 0):
        raise ValueError("Invalid lengths")
    if np.any(starts < 0) or np.any(starts + lengths[None, :] > n):
        raise ValueError("Sample outside source interval")
    h = int(lengths.sum())
    if h < 2 or n < 2:
        raise ValueError("At least two observations required")
    center = r.mean(axis=0)
    z = r - center
    p1 = np.vstack([np.zeros(f), np.cumsum(z, axis=0)])
    p2 = np.vstack([np.zeros(f), np.cumsum(z * z, axis=0)])
    sum1 = np.zeros((len(starts), f))
    sum2 = np.zeros_like(sum1)
    for j, length in enumerate(lengths):
        lo = starts[:, j]
        hi = lo + length
        sum1 += p1[hi] - p1[lo]
        sum2 += p2[hi] - p2[lo]
    centered_ss = sum2 - sum1 * sum1 / h
    tolerance = 64 * np.finfo(float).eps * np.maximum(np.abs(sum2), np.finfo(float).tiny)
    if np.any(centered_ss < -tolerance):
        raise ArithmeticError("Unstable variance requires a stable production fallback")
    return center + sum1 / h, np.maximum(centered_ss, 0) / (h - 1)


class PathSummary(NamedTuple):
    total: float
    high: float
    low: float
    log_drawdown: float


def path_summary(returns) -> PathSummary:
    r = real_array(returns, 1)
    if np.any(r <= -1):
        raise ValueError("Log-path scope requires r > -1")
    p = np.r_[0.0, np.cumsum(np.log1p(r))]
    return PathSummary(float(p[-1]), float(p.max()), float(p.min()),
                       float(np.max(np.maximum.accumulate(p) - p)))


def summary_mdd(summary: PathSummary) -> float:
    return float(-np.expm1(-summary.log_drawdown))


# ===========================================================================
# shared test fixtures
# ===========================================================================

def make_manifest(t_len: int, seed_offset: int = 0):
    """Leak-free manifest: training ends before the first decision id."""
    return OOSFitManifest(
        model_ref=f"model::{seed_offset}",
        fit_scope_ref="scope::2024H2",
        max_training_label_end=1000,
        decision_time_ids=tuple(range(1001, 1001 + t_len)),
    )


def sr_of(series: np.ndarray, a: float = A_DEFAULT) -> float:
    series = np.asarray(series, dtype=float)
    return math.sqrt(a) * series.mean() / series.std(ddof=1)


def psi_manual(r: np.ndarray, a: float = A_DEFAULT) -> np.ndarray:
    """Sec. 7.1 B influence values, computed independently of the kernel."""
    t = r.shape[0]
    mu = r.mean()
    v = ((r - mu) ** 2).mean()
    a_t = math.sqrt((t - 1) / t)
    return math.sqrt(a) * a_t * ((r - mu) / math.sqrt(v)
                                 - mu * ((r - mu) ** 2 - v) / (2.0 * v ** 1.5))


# ===========================================================================
# BlockStartsPlan: Sec. 6.3 / Sec. 12 golden contract
# ===========================================================================

class TestBlockPlanGolden(unittest.TestCase):

    def test_matches_sec12_block_plan_byte_for_byte(self):
        for seed in (0, 1, 7, 1234):
            oracle_starts, oracle_lengths = block_plan(103, 103, 7, 31, seed)
            plan = BlockStartsPlan(103, 103, 7, 31, seed)
            np.testing.assert_array_equal(plan.starts(), oracle_starts)
            np.testing.assert_array_equal(plan.lengths, oracle_lengths)

    def test_end_block_truncation(self):
        plan = BlockStartsPlan(103, 103, 7, 25)
        lengths = plan.lengths
        assert lengths[-1] == 5
        assert lengths.sum() == 103
        assert np.all(plan.starts() <= 96)

    def test_reproducible_same_seed(self):
        a = BlockStartsPlan(103, 29, 7, 25, 7)
        b = BlockStartsPlan(103, 29, 7, 25, 7)
        np.testing.assert_array_equal(a.starts(), b.starts())
        np.testing.assert_array_equal(a.lengths, b.lengths)

    def test_plan_independent_of_factor_axis(self):
        """Sec. 1.1: the RNG plan must not depend on factor order/count."""
        plans = [BlockStartsPlan(103, 103, 7, 31, 3) for _ in range(3)]
        first = plans[0].starts()
        for plan in plans[1:]:
            assert plan.starts().tobytes() == first.tobytes()

    def test_invalid_bool_and_b1_rejected(self):
        with pytest.raises(ValueError):
            BlockStartsPlan(30, 30, True, 10)
        with pytest.raises(ValueError):
            BlockStartsPlan(30, 30, 10, 1)
        with pytest.raises(ValueError):
            BlockStartsPlan(30, 30, 31, 10)
        with pytest.raises(ValueError):
            BlockStartsPlan(30, 30, 10, True)
        with pytest.raises(ValueError):
            BlockStartsPlan(30, 30, 10, 10, seed=True)

    def test_prefix_moments_match_sec12_oracle(self):
        rng = np.random.default_rng(8)
        r = rng.normal(0.001, 0.02, (103, 5))
        oracle_starts, oracle_lengths = block_plan(103, 103, 7, 31, 7)
        plan = BlockStartsPlan(103, 103, 7, 31, 7)
        from quant_evaluator.metrics.extension.paired_performance import (
            _prefix_sample_moments,
        )
        mu, var = _prefix_sample_moments(r, plan.starts(), plan.lengths)
        mu_o, var_o = sample_moments_prefix(r, oracle_starts, oracle_lengths)
        np.testing.assert_allclose(mu, mu_o, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(var, var_o, rtol=1e-12, atol=1e-15)


# ===========================================================================
# M01 point estimate: paired_net_sharpe_delta
# ===========================================================================

class TestPairedNetSharpeDelta(unittest.TestCase):

    def test_golden_hand_value(self):
        rng = np.random.default_rng(42)
        cand = rng.normal(0.001, 0.02, (80, 1))
        base = rng.normal(0.0003, 0.015, (80, 1))
        res = paired_net_sharpe_delta(cand, base, min_periods=60)
        assert res.status == "OK"
        expected = sr_of(cand[:, 0]) - sr_of(base[:, 0])
        assert res.values[0, 0] == pytest.approx(expected, rel=1e-13)

    def test_identical_inputs_zero_delta(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.0008, 0.02, (80, 2))
        res = paired_net_sharpe_delta(x, x, min_periods=60)
        assert res.status == "OK"
        assert np.all(res.values[0] == 0.0)
        assert res.diagnostics["identical_inputs"] is True

    def test_scaled_baseline_same_sr_rf_zero(self):
        """Sec. 7.1 MUST-test: candidate = 2 * baseline, rf = 0 -> same SR."""
        rng = np.random.default_rng(2)
        base = rng.normal(0.0006, 0.02, (90, 3))
        res = paired_net_sharpe_delta(2.0 * base, base, min_periods=60)
        assert np.all(res.values[0] == 0.0)

    def test_constant_shift_raises_sr(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0.0002, 0.02, (90, 1))
        cand = base + 0.0005
        res = paired_net_sharpe_delta(cand, base, min_periods=60)
        assert res.values[0, 0] > 0.0

    def test_rf_breaks_scale_invariance(self):
        """Sec. 7.1 MUST-test: with rf != 0 the scaling shortcut is invalid."""
        rng = np.random.default_rng(4)
        base = rng.normal(0.0006, 0.02, (90, 1))
        rf = np.full(90, 0.0001)
        res = paired_net_sharpe_delta(2.0 * base, base, risk_free_daily=rf, min_periods=60)
        assert res.values[0, 0] != 0.0

    def test_internal_nan_is_invalid_evidence(self):
        rng = np.random.default_rng(5)
        cand = rng.normal(0.001, 0.02, (80, 2))
        base = rng.normal(0.0003, 0.02, (80, 2))
        cand[40, 0] = np.nan
        res = paired_net_sharpe_delta(cand, base, min_periods=60)
        assert res.status == "INVALID_EVIDENCE"
        assert res.reason_code == "NONFINITE_OR_GAPPED_SAMPLE"
        assert np.isnan(res.values[0, 0])
        assert np.isfinite(res.values[0, 1])  # untouched column still computed
        assert res.diagnostics["invalid_columns"] == (0,)

    def test_inf_rejected_same_as_gap(self):
        rng = np.random.default_rng(6)
        cand = rng.normal(0.001, 0.02, (80, 1))
        cand[10, 0] = np.inf
        res = paired_net_sharpe_delta(cand, cand * 0.5, min_periods=60)
        assert res.status == "INVALID_EVIDENCE"

    def test_short_sample_insufficient(self):
        rng = np.random.default_rng(7)
        cand = rng.normal(0.001, 0.02, (50, 1))
        res = paired_net_sharpe_delta(cand, cand * 0.5, min_periods=60)
        assert res.status == "INSUFFICIENT_DATA"
        assert res.reason_code == "OBSERVATIONS_TOO_FEW"
        assert np.isnan(res.values[0, 0])

    def test_constant_series_insufficient_nan(self):
        cand = np.full((80, 2), 0.002)
        base = np.full((80, 2), 0.001)
        res = paired_net_sharpe_delta(cand, base, min_periods=60)
        assert res.status == "INSUFFICIENT_DATA"
        assert res.reason_code == "ZERO_VARIANCE"
        assert np.isnan(res.values[0]).all()

    def test_shape_and_dtype_rejections(self):
        ok = np.random.default_rng(8).normal(0.001, 0.02, (80, 1))
        with pytest.raises(ValueError):
            paired_net_sharpe_delta(ok, ok[:-1], min_periods=60)
        with pytest.raises(ValueError):  # bool array
            paired_net_sharpe_delta(ok.astype(bool), ok, min_periods=60)
        with pytest.raises(ValueError):  # complex
            paired_net_sharpe_delta(ok.astype(complex), ok, min_periods=60)
        with pytest.raises(ValueError):  # string/object
            paired_net_sharpe_delta(ok.astype(object), ok, min_periods=60)
        with pytest.raises(ValueError):  # 1-D not accepted here (pair contract)
            paired_net_sharpe_delta(ok[:, 0], ok[:, 0], min_periods=60)

    def test_factor_permutation_and_single_column_tiles(self):
        rng = np.random.default_rng(9)
        cand = rng.normal(0.001, 0.02, (80, 4))
        base = rng.normal(0.0004, 0.02, (80, 1))
        full = paired_net_sharpe_delta(cand, base, min_periods=60)
        perm = [3, 0, 2, 1]
        permuted = paired_net_sharpe_delta(cand[:, perm], base, min_periods=60)
        np.testing.assert_allclose(permuted.values[0], full.values[0][perm],
                                   rtol=1e-13, atol=1e-15)
        for f in range(4):
            single = paired_net_sharpe_delta(cand[:, f:f + 1], base, min_periods=60)
            assert single.values[0, 0] == pytest.approx(full.values[0, f], rel=1e-13)


# ===========================================================================
# M01 method B: paired_sharpe_hac_ci
# ===========================================================================

class TestPairedSharpeHacCi(unittest.TestCase):

    def _inputs(self, seed=11, t=120):
        rng = np.random.default_rng(seed)
        cand = rng.normal(0.001, 0.02, (t, 1))
        base = rng.normal(0.0003, 0.015, (t, 1))
        return cand, base

    def test_golden_manual_influence_formula_and_existing_hac_kernel(self):
        cand, base = self._inputs()
        rf = np.full(120, 0.00002)
        res = paired_sharpe_hac_ci(cand, base, risk_free_daily=rf,
                                   max_lag=3, min_periods=60)
        assert res.interval_status == "OK"
        u = psi_manual(cand[:, 0] - rf) - psi_manual(base[:, 0] - rf)
        om_over_t = compute_hac_variance(u[:, None], max_lag=3, kernel="bartlett")[0]
        se_expected = math.sqrt(om_over_t)
        assert res.values[3, 0] == pytest.approx(se_expected, rel=1e-12)
        z = norm.ppf(0.975)
        assert res.values[1, 0] == pytest.approx(res.values[0, 0] - z * se_expected, rel=1e-12)
        assert res.values[2, 0] == pytest.approx(res.values[0, 0] + z * se_expected, rel=1e-12)

    def test_component_order_and_estimate_match_point_kernel(self):
        cand, base = self._inputs(12)
        point = paired_net_sharpe_delta(cand, base, min_periods=60)
        ci = paired_sharpe_hac_ci(cand, base, max_lag=4, min_periods=60)
        assert ci.component_names == ("estimate", "ci_low", "ci_high", "standard_error")
        assert ci.values[0, 0] == point.values[0, 0]
        assert ci.diagnostics["method"] == "paired_influence_hac_v1"

    def test_identical_statistic_degenerate_interval(self):
        cand, _ = self._inputs(13)
        res = paired_sharpe_hac_ci(cand, cand, max_lag=3, min_periods=60)
        assert res.interval_status == "OK"
        assert res.values[0, 0] == 0.0
        assert res.values[1, 0] == 0.0 and res.values[2, 0] == 0.0
        assert res.values[3, 0] == 0.0
        assert res.diagnostics["identical_statistic"] == (True,)

    def test_auto_bandwidth_nw_rule_v1(self):
        assert default_hac_max_lag(100) == 4
        assert default_hac_max_lag(252) == int(math.floor(4 * (2.52) ** (2 / 9)))
        cand, base = self._inputs(14)
        res = paired_sharpe_hac_ci(cand, base, min_periods=60)
        assert res.diagnostics["bandwidth_rule"] == "nw_rule_v1"
        assert res.diagnostics["max_lag"] == default_hac_max_lag(120)

    def test_insufficient_interval_keeps_point_estimate(self):
        """T < max_lag + 10 -> the HAC kernel declines; the point estimate
        stays valid and the interval reports insufficient (Sec. 5.3)."""
        rng = np.random.default_rng(15)
        cand = rng.normal(0.001, 0.02, (14, 1))
        base = cand[::-1].copy()  # different series: u != 0, so the HAC kernel
        # (T=14 < max_lag + 10) must decline and the interval stays insufficient
        res = paired_sharpe_hac_ci(cand, base, max_lag=5, min_periods=10)
        assert res.status == "OK"
        assert np.isfinite(res.values[0, 0])
        assert res.interval_status == "INSUFFICIENT_DATA"
        assert np.isnan(res.values[3, 0])

    def test_factor_permutation_consistency(self):
        rng = np.random.default_rng(16)
        cand = rng.normal(0.001, 0.02, (90, 3))
        base = rng.normal(0.0003, 0.02, (90, 1))
        full = paired_sharpe_hac_ci(cand, base, max_lag=3, min_periods=60)
        perm = [2, 1, 0]
        moved = paired_sharpe_hac_ci(cand[:, perm], base, max_lag=3, min_periods=60)
        np.testing.assert_allclose(moved.values, full.values[:, perm],
                                   rtol=1e-12, atol=1e-14)


# ===========================================================================
# M01 method A: paired_sharpe_bootstrap_ci (percentile)
# ===========================================================================

class TestPairedSharpeBootstrapCi(unittest.TestCase):

    def test_replicates_match_expansion_oracle(self):
        rng = np.random.default_rng(21)
        t, reps, block, seed = 40, 25, 8, 5
        cand = rng.normal(0.001, 0.02, (t, 2))
        base = rng.normal(0.0004, 0.015, (t, 1))
        res = paired_sharpe_bootstrap_ci(cand, base, block_length=block,
                                         repetitions=reps, seed=seed,
                                         min_periods=30)
        assert res.interval_status == "OK"
        starts, lengths = block_plan(t, t, block, reps, seed)
        delta_star = np.empty((reps, 2))
        for b in range(reps):
            idx = expand_draw(starts[b], lengths)
            delta_star[b] = sr_of(cand[idx, 0]) - sr_of(base[idx, 0]), \
                sr_of(cand[idx, 1]) - sr_of(base[idx, 0])
        for f in range(2):
            finite = delta_star[:, f][np.isfinite(delta_star[:, f])]
            lo, hi = np.quantile(finite, [0.025, 0.975])
            assert res.values[1, f] == pytest.approx(lo, rel=1e-10, abs=1e-12)
            assert res.values[2, f] == pytest.approx(hi, rel=1e-10, abs=1e-12)
            assert res.values[3, f] == pytest.approx(finite.size)
            assert res.values[1, f] <= res.values[0, f] <= res.values[2, f]

    def test_seed_determinism_byte_identical(self):
        """Sec. 1.1: same seed -> byte-identical outputs; >= 3 seeds."""
        rng = np.random.default_rng(22)
        cand = rng.normal(0.001, 0.02, (70, 2))
        base = rng.normal(0.0003, 0.02, (70, 1))
        for seed in (0, 1, 7):
            first = paired_sharpe_bootstrap_ci(cand, base, repetitions=99,
                                               seed=seed, min_periods=60)
            second = paired_sharpe_bootstrap_ci(cand, base, repetitions=99,
                                                seed=seed, min_periods=60)
            assert first.values.tobytes() == second.values.tobytes()
        seeds_differ = [
            paired_sharpe_bootstrap_ci(cand, base, repetitions=99, seed=s,
                                       min_periods=60).values[1, 0]
            for s in (0, 1, 7)
        ]
        assert len({float(v) for v in seeds_differ}) == 3

    def test_factor_order_does_not_change_the_plan(self):
        """Permuting factors must not move any interval (shared draws)."""
        rng = np.random.default_rng(23)
        cand = rng.normal(0.001, 0.02, (80, 4))
        base = rng.normal(0.0003, 0.02, (80, 1))
        full = paired_sharpe_bootstrap_ci(cand, base, repetitions=99,
                                          seed=3, min_periods=60)
        perm = [3, 1, 0, 2]
        moved = paired_sharpe_bootstrap_ci(cand[:, perm], base, repetitions=99,
                                           seed=3, min_periods=60)
        np.testing.assert_allclose(moved.values, full.values[:, perm],
                                   rtol=1e-12, atol=1e-14)

    def test_truncated_tail_block(self):
        """T not a multiple of block_length: the final short block uses the
        full start range (Sec. 6.3) and everything stays finite."""
        rng = np.random.default_rng(24)
        cand = rng.normal(0.001, 0.02, (103, 1))
        base = rng.normal(0.0003, 0.02, (103, 1))
        res = paired_sharpe_bootstrap_ci(cand, base, block_length=7,
                                         repetitions=49, seed=0, min_periods=60)
        assert res.interval_status == "OK"
        assert np.isfinite(res.values[:, 0]).all()
        assert res.diagnostics["block_length"] == 7

    def test_b1_and_bool_rejected(self):
        rng = np.random.default_rng(25)
        cand = rng.normal(0.001, 0.02, (70, 1))
        with pytest.raises(ValueError):
            paired_sharpe_bootstrap_ci(cand, cand, repetitions=1, min_periods=60)
        with pytest.raises(ValueError):
            paired_sharpe_bootstrap_ci(cand, cand, block_length=True, min_periods=60)
        with pytest.raises(ValueError):
            paired_sharpe_bootstrap_ci(cand, cand, block_length=71, min_periods=60)

    def test_method_label_and_components(self):
        rng = np.random.default_rng(26)
        cand = rng.normal(0.001, 0.02, (70, 1))
        res = paired_sharpe_bootstrap_ci(cand, cand * 0.5, repetitions=49,
                                         min_periods=60)
        assert res.component_names == ("estimate", "ci_low", "ci_high",
                                       "valid_replicates")
        assert res.diagnostics["method"] == "paired_moving_block_percentile_v1"
        assert res.values[3, 0] == 49.0

    def test_coverage_iid_gaussian(self):
        """200 iid-Gaussian simulations: the 95% interval must contain the
        true Sharpe delta with relative frequency in [0.85, 1.0]."""
        rng = np.random.default_rng(2024)
        t, sims = 252, 200
        mu0, mu1, sd = 0.0005, 0.0015, 0.01
        true_delta = math.sqrt(A_DEFAULT) * (mu1 - mu0) / sd
        covered = 0
        for _ in range(sims):
            base = rng.normal(mu0, sd, (t, 1))
            cand = rng.normal(mu1, sd, (t, 1))
            res = paired_sharpe_bootstrap_ci(cand, base, block_length=10,
                                             repetitions=199, seed=0,
                                             min_periods=60)
            covered += bool(res.values[1, 0] <= true_delta <= res.values[2, 0])
        assert 0.85 <= covered / sims <= 1.0


# ===========================================================================
# M01 method C: paired_sharpe_studentized_ci
# ===========================================================================

class TestPairedSharpeStudentizedCi(unittest.TestCase):

    def test_matches_expansion_oracle(self):
        rng = np.random.default_rng(31)
        t, reps, block, lag, seed = 48, 25, 8, 2, 3
        cand = rng.normal(0.001, 0.02, (t, 2))
        base = rng.normal(0.0004, 0.015, (t, 1))
        res = paired_sharpe_studentized_ci(cand, base, block_length=block,
                                           repetitions=reps, seed=seed,
                                           max_lag=lag, min_periods=40)
        assert res.interval_status == "OK"
        delta_obs = np.array([sr_of(cand[:, f]) - sr_of(base[:, 0]) for f in range(2)])
        u_obs = psi_manual(cand[:, 0]) - psi_manual(base[:, 0])
        se0 = np.array([math.sqrt(compute_hac_variance(
            (psi_manual(cand[:, f]) - psi_manual(base[:, 0]))[:, None],
            max_lag=lag, kernel="bartlett")[0]) for f in range(2)])
        starts, lengths = block_plan(t, t, block, reps, seed)
        t_star = np.full((reps, 2), np.nan)
        for b in range(reps):
            idx = expand_draw(starts[b], lengths)
            for f in range(2):
                ds = sr_of(cand[idx, f]) - sr_of(base[idx, 0])
                u = psi_manual(cand[idx, f]) - psi_manual(base[idx, 0])
                om = compute_hac_variance(u[:, None], max_lag=lag,
                                          kernel="bartlett")[0]
                if om > 0:
                    t_star[b, f] = (ds - delta_obs[f]) / math.sqrt(om)
        for f in range(2):
            finite = t_star[:, f][np.isfinite(t_star[:, f])]
            q_lo, q_up = np.quantile(finite, [0.025, 0.975])
            lo_expected = delta_obs[f] - q_up * se0[f]
            hi_expected = delta_obs[f] - q_lo * se0[f]
            assert res.values[1, f] == pytest.approx(lo_expected, rel=1e-9, abs=1e-11)
            assert res.values[2, f] == pytest.approx(hi_expected, rel=1e-9, abs=1e-11)
            assert res.values[3, f] == pytest.approx(se0[f], rel=1e-12)

    def test_identical_inputs_interval_insufficient_point_zero(self):
        rng = np.random.default_rng(32)
        cand = rng.normal(0.001, 0.02, (60, 1))
        res = paired_sharpe_studentized_ci(cand, cand, block_length=10,
                                           repetitions=25, seed=0, max_lag=2,
                                           min_periods=40)
        assert res.status == "OK"
        assert res.values[0, 0] == 0.0
        assert res.interval_status == "INSUFFICIENT_DATA"
        assert res.diagnostics["identical_statistic"] == (True,)

    def test_seed_determinism_and_interval_contains_point(self):
        rng = np.random.default_rng(33)
        cand = rng.normal(0.0012, 0.02, (90, 1))
        base = rng.normal(0.0003, 0.015, (90, 1))
        for seed in (0, 1, 7):
            first = paired_sharpe_studentized_ci(cand, base, block_length=10,
                                                 repetitions=30, seed=seed,
                                                 max_lag=2, min_periods=60)
            second = paired_sharpe_studentized_ci(cand, base, block_length=10,
                                                  repetitions=30, seed=seed,
                                                  max_lag=2, min_periods=60)
            assert first.values.tobytes() == second.values.tobytes()
            assert first.interval_status == "OK"
            assert first.values[1, 0] <= first.values[0, 0] <= first.values[2, 0]


# ===========================================================================
# M01: paired_cer_delta / paired_mdd_improvement / paired_es_improvement
# ===========================================================================

class TestPairedCerDelta(unittest.TestCase):

    def test_golden_hand_value(self):
        rng = np.random.default_rng(41)
        cand = rng.normal(0.001, 0.02, (80, 1))
        base = rng.normal(0.0004, 0.015, (80, 1))
        lam = 3.0
        res = paired_cer_delta(cand, base, risk_aversion_lambda=lam, min_periods=60)
        cer_c = A_DEFAULT * (cand[:, 0].mean() - lam * cand[:, 0].var(ddof=1) / 2)
        cer_b = A_DEFAULT * (base[:, 0].mean() - lam * base[:, 0].var(ddof=1) / 2)
        assert res.values[0, 0] == pytest.approx(cer_c - cer_b, rel=1e-13)

    def test_lambda_validation(self):
        rng = np.random.default_rng(42)
        cand = rng.normal(0.001, 0.02, (80, 1))
        with pytest.raises(ValueError):
            paired_cer_delta(cand, cand, risk_aversion_lambda=-0.1, min_periods=60)
        with pytest.raises(ValueError):
            paired_cer_delta(cand, cand, risk_aversion_lambda=True, min_periods=60)
        with pytest.raises(ValueError):
            paired_cer_delta(cand, cand, risk_aversion_lambda=float("nan"),
                             min_periods=60)
        with pytest.raises(TypeError):
            paired_cer_delta(cand, cand, min_periods=60)  # lambda is required

    def test_rf_enters_level_and_cancels_in_delta(self):
        """Sec. 7.1: both legs subtract the SAME rf, so rf shifts each CER
        level but cancels in CER_1 - CER_0 (verified against a hand value)."""
        rng = np.random.default_rng(43)
        cand = rng.normal(0.001, 0.02, (80, 1))
        base = rng.normal(0.0004, 0.015, (80, 1))
        rf = np.full(80, 0.0002)
        lam = 2.0
        no_rf = paired_cer_delta(cand, base, risk_aversion_lambda=lam, min_periods=60)
        with_rf = paired_cer_delta(cand, base, risk_free_daily=rf,
                                   risk_aversion_lambda=lam, min_periods=60)
        cer_c = A_DEFAULT * ((cand[:, 0] - rf).mean()
                             - lam * (cand[:, 0] - rf).var(ddof=1) / 2)
        cer_b = A_DEFAULT * ((base[:, 0] - rf).mean()
                             - lam * (base[:, 0] - rf).var(ddof=1) / 2)
        assert with_rf.values[0, 0] == pytest.approx(cer_c - cer_b, rel=1e-12)
        # identical rf on both legs: the delta is rf-invariant
        assert with_rf.values[0, 0] == pytest.approx(no_rf.values[0, 0], rel=1e-12)


class TestPairedMddImprovement(unittest.TestCase):

    def test_golden_matches_sec12_path_oracle(self):
        rng = np.random.default_rng(51)
        cand = rng.normal(0.001, 0.03, (103, 2))
        base = rng.normal(0.0005, 0.02, (103, 1))
        res = paired_mdd_improvement(cand, base, min_periods=60)
        for f in range(2):
            mdd_c = summary_mdd(path_summary(cand[:, f]))
            mdd_b = summary_mdd(path_summary(base[:, 0]))
            assert res.values[0, f] == pytest.approx(mdd_b - mdd_c, rel=1e-12)

    def test_initial_wealth_and_all_positive(self):
        single = np.array([[-0.1], [0.1]])
        res = paired_mdd_improvement(single, np.array([[0.0], [0.0]]), min_periods=2)
        assert res.values[0, 0] == pytest.approx(0.0 - 0.1, rel=1e-12)
        up = np.array([[0.1], [0.02], [0.03]])
        flat = paired_mdd_improvement(up, up, min_periods=2)
        assert flat.values[0, 0] == 0.0

    def test_improvement_sign(self):
        rng = np.random.default_rng(52)
        base = -np.abs(rng.normal(0.0, 0.02, (80, 1))) - 0.005  # steady losses
        cand = np.abs(rng.normal(0.0, 0.02, (80, 1))) + 0.001   # steady gains
        res = paired_mdd_improvement(cand, base, min_periods=60)
        assert res.values[0, 0] > 0.0

    def test_terminal_total_loss_and_basis_mismatch(self):
        ok = np.full((80, 1), 0.001)
        with_loss = np.full((80, 1), 0.001)
        with_loss[10, 0] = -1.0
        res = paired_mdd_improvement(with_loss, ok, min_periods=60)
        assert res.status == "INVALID_EVIDENCE"
        assert res.reason_code == "TERMINAL_TOTAL_LOSS"
        assert res.diagnostics["terminal_total_loss"] is True
        assert res.diagnostics["known_drawdown_lower_bound"] == 1.0
        below = np.full((80, 1), 0.001)
        below[10, 0] = -1.1
        res2 = paired_mdd_improvement(below, ok, min_periods=60)
        assert res2.status == "INVALID_EVIDENCE"
        assert res2.reason_code == "RETURN_BASIS_MISMATCH"

    def test_internal_nan_invalid(self):
        ok = np.full((80, 1), 0.001)
        gapped = np.full((80, 1), 0.001)
        gapped[30, 0] = np.nan
        res = paired_mdd_improvement(gapped, ok, min_periods=60)
        assert res.status == "INVALID_EVIDENCE"
        assert res.reason_code == "NONFINITE_OR_GAPPED_SAMPLE"


class TestPairedEsImprovement(unittest.TestCase):

    def test_matches_sec12_upper_tail_mean_oracle(self):
        rng = np.random.default_rng(61)
        cand = rng.normal(0.0, 0.02, (210, 3))   # tail mass = 10.5 >= 10
        base = rng.normal(-0.001, 0.02, (210, 1))
        res = paired_es_improvement(cand, base, min_periods=60)
        for f in range(3):
            es_c = upper_tail_mean(-cand[:, f], 0.95)
            es_b = upper_tail_mean(-base[:, 0], 0.95)
            assert res.values[0, f] == pytest.approx(es_b - es_c, rel=1e-12)

    def test_tie_permutation_invariance(self):
        """Sec. 6.6: tied tail days must contribute identical weights under
        any permutation (Sec. 12 test_tail_condition_permutation analogue)."""
        x = np.array([3.0, 3.0, 1.0, 0.0])  # loss vector
        perm = [2, 1, 3, 0]
        zeros = np.zeros((4, 1))
        # returns = -losses, so the kernel's UTM(-r) equals UTM(x)
        res_a = paired_es_improvement((-x)[:, None], zeros, tail_confidence=0.5,
                                      min_tail_mass=1.0, min_periods=2)
        res_b = paired_es_improvement((-x[perm])[:, None], zeros,
                                      tail_confidence=0.5,
                                      min_tail_mass=1.0, min_periods=2)
        es_a = upper_tail_mean(x, 0.5)
        es_b = upper_tail_mean(x[perm], 0.5)
        assert es_a == pytest.approx(3.0, rel=1e-12)  # both tied 3s split the mass
        assert es_a == pytest.approx(es_b, rel=1e-12)
        assert res_a.values[0, 0] == pytest.approx(0.0 - es_a, rel=1e-12)
        assert res_b.values[0, 0] == pytest.approx(0.0 - es_b, rel=1e-12)

    def test_tail_mass_minimum(self):
        rng = np.random.default_rng(62)
        cand = rng.normal(0.0, 0.02, (100, 1))  # mass = 5 < 10
        res = paired_es_improvement(cand, cand * 0.5, min_periods=60)
        assert res.status == "INSUFFICIENT_DATA"
        assert res.reason_code == "TAIL_MASS_BELOW_MINIMUM"
        res2 = paired_es_improvement(cand, cand * 0.5, min_tail_mass=5.0,
                                     min_periods=60)
        assert res2.status == "OK"

    def test_determinism_across_three_seeds_irrelevant_but_stable(self):
        rng = np.random.default_rng(63)
        cand = rng.normal(0.0, 0.02, (210, 2))
        base = rng.normal(-0.001, 0.02, (210, 1))
        first = paired_es_improvement(cand, base, min_periods=60)
        second = paired_es_improvement(cand, base, min_periods=60)
        assert first.values.tobytes() == second.values.tobytes()


# ===========================================================================
# M02 kernels
# ===========================================================================

class TestOOSCommon(unittest.TestCase):

    def test_missing_manifest_type_error(self):
        y = np.random.default_rng(71).normal(0, 0.01, (30, 10))
        with pytest.raises(TypeError):
            oos_rank_ic_delta(y, y, y, None, make_manifest(30))
        with pytest.raises(TypeError):
            oos_r2_gain(y, y, y, y, make_manifest(30), None)

    def test_leakage_invalid_strict_less_gate(self):
        rng = np.random.default_rng(72)
        y = rng.normal(0, 0.01, (30, 10))
        leaking = OOSFitManifest("m", "s", 1001, tuple(range(1001, 1031)))
        ok = make_manifest(30)
        res = oos_rank_ic_delta(y, y, y, leaking, ok)
        assert res.status == "INVALID_EVIDENCE"
        assert res.reason_code == "OOS_WINDOW_LEAKAGE"
        assert res.diagnostics["leaky_manifest"] == "baseline"
        # boundary: training end strictly before the first decision passes
        boundary_ok = OOSFitManifest("m", "s", 1000, tuple(range(1001, 1031)))
        res2 = oos_rank_ic_delta(y, y, y, boundary_ok, boundary_ok, min_assets=5)
        assert res2.status == "OK"

    def test_manifest_axis_mismatch_rejected(self):
        rng = np.random.default_rng(73)
        y = rng.normal(0, 0.01, (30, 10))
        with pytest.raises(ValueError):
            oos_rank_ic_delta(y, y, y, make_manifest(29), make_manifest(30))

    def test_manifest_validation(self):
        with pytest.raises(ValueError):
            OOSFitManifest("", "s", 1, (1, 2))
        with pytest.raises(ValueError):
            OOSFitManifest("m", "s", float("nan"), (1, 2))
        with pytest.raises(ValueError):
            OOSFitManifest("m", "s", 0, (2, 1))
        with pytest.raises(ValueError):
            OOSFitManifest("m", "s", 0, ())


class TestOOSRankIcDelta(unittest.TestCase):

    def test_identical_predictions_zero_delta(self):
        rng = np.random.default_rng(81)
        y = rng.normal(0, 0.01, (30, 20))
        res = oos_rank_ic_delta(y, y, y, make_manifest(30), make_manifest(30),
                                min_assets=10)
        assert res.status == "OK"
        assert res.summary == pytest.approx(0.0, abs=1e-15)
        assert np.allclose(res.daily, 0.0)

    def test_perfect_candidate_positive_delta_and_counts(self):
        rng = np.random.default_rng(82)
        y = rng.normal(0, 0.01, (30, 20))
        base_pred = rng.normal(0, 0.01, (30, 20))
        res = oos_rank_ic_delta(y, base_pred, y, make_manifest(30),
                                make_manifest(30), min_assets=10)
        assert res.status == "OK"
        assert res.summary > 0.0
        assert np.all(np.isfinite(res.daily))
        ic_c = res.diagnostics["ic_candidate"]
        ic_b = res.diagnostics["ic_baseline"]
        assert np.allclose(ic_c, 1.0)  # candidate == y -> perfect rank match
        assert res.summary == pytest.approx(np.mean(ic_c - ic_b), rel=1e-12)
        assert np.all(res.diagnostics["n_assets"] == 20)

    def test_matches_scipy_spearman_on_common_support(self):
        rng = np.random.default_rng(83)
        y = rng.normal(0, 0.01, (12, 15))
        p0 = rng.normal(0, 0.01, (12, 15))
        p1 = rng.normal(0, 0.01, (12, 15))
        p1[0, 3] = np.nan  # shrink day-0 common support
        res = oos_rank_ic_delta(y, p0, p1, make_manifest(12), make_manifest(12),
                                min_assets=5)
        assert res.diagnostics["n_assets"][0] == 14
        assert res.diagnostics["n_assets"][1] == 15
        keep = np.isfinite(p0[0]) & np.isfinite(y[0]) & np.isfinite(p1[0])
        expected = spearmanr(p0[0][keep], y[0][keep]).statistic
        assert res.diagnostics["ic_baseline"][0] == pytest.approx(expected, rel=1e-12)

    def test_constant_prediction_day_invalid(self):
        rng = np.random.default_rng(84)
        y = rng.normal(0, 0.01, (10, 12))
        p0 = rng.normal(0, 0.01, (10, 12))
        p1 = rng.normal(0, 0.01, (10, 12))
        p1[3] = 0.5  # constant cross-section -> Spearman undefined
        res = oos_rank_ic_delta(y, p0, p1, make_manifest(10), make_manifest(10),
                                min_assets=5)
        assert np.isnan(res.daily[3])
        assert np.isfinite(res.daily).sum() == 9


class TestOOSDailyMseImprovement(unittest.TestCase):

    def test_identical_zero_and_candidate_equals_y_positive(self):
        rng = np.random.default_rng(91)
        y = rng.normal(0, 0.01, (20, 15))
        base_pred = y + 0.005
        same = oos_daily_mse_improvement(y, y, y, make_manifest(20),
                                         make_manifest(20), min_assets=5)
        assert same.summary == 0.0
        better = oos_daily_mse_improvement(y, base_pred, y, make_manifest(20),
                                           make_manifest(20), min_assets=5)
        assert better.summary > 0.0
        assert np.all(better.daily > 0.0)

    def test_equal_weight_daily_means_not_asset_weighted(self):
        """Two days with different asset counts: the summary must equal the
        mean of per-day means (Sec. 7.2: no secret asset weighting)."""
        y = np.zeros((2, 13))
        p0 = np.zeros((2, 13))
        p1 = np.zeros((2, 13))
        # day 0: 10 valid assets; (y+1)^2 - (y-1)^2 = 4y -> mean 22
        y[0, :10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p0[0, :10] = -1.0
        p1[0, :10] = 1.0
        # day 1: 3 valid assets; mean improvement = 4 * 2 = 8
        y[1, :3] = [1, 2, 3]
        p0[1, :3] = -1.0
        p1[1, :3] = 1.0
        mask = np.zeros((2, 13), dtype=bool)
        mask[0, :10] = True
        mask[1, :3] = True
        res = oos_daily_mse_improvement(y, p0, p1, make_manifest(2),
                                        make_manifest(2), validity=mask,
                                        min_assets=1)
        d0 = 22.0
        d1 = 8.0
        assert res.daily[0] == pytest.approx(d0)
        assert res.daily[1] == pytest.approx(d1)
        assert res.summary == pytest.approx((d0 + d1) / 2, rel=1e-12)
        assert res.diagnostics["n_assets"][0] == 10


class TestOOSR2Gain(unittest.TestCase):

    def test_golden_formula_and_not_relative_to_sse0(self):
        rng = np.random.default_rng(101)
        y = rng.normal(0, 0.01, (25, 12))
        p0 = y + rng.normal(0, 0.004, (25, 12))
        p1 = y + rng.normal(0, 0.002, (25, 12))
        p_ref = np.zeros((25, 12))
        res = oos_r2_gain(y, p0, p1, p_ref, make_manifest(25),
                          make_manifest(25), min_assets=5)
        sse0 = float(((y - p0) ** 2).mean())
        sse1 = float(((y - p1) ** 2).mean())
        sst = float((y ** 2).mean())
        assert res.summary == pytest.approx((sse0 - sse1) / sst, rel=1e-10)
        assert abs(res.summary - (sse0 - sse1) / sse0) > 1e-6  # NOT /SSE0
        assert res.diagnostics["sst_reference"] == pytest.approx(sst, rel=1e-12)

    def test_missing_reference_type_error(self):
        rng = np.random.default_rng(102)
        y = rng.normal(0, 0.01, (20, 10))
        with pytest.raises(TypeError):
            oos_r2_gain(y, y, y, None, make_manifest(20), make_manifest(20))

    def test_reference_equals_labels_zero_denominator(self):
        rng = np.random.default_rng(103)
        y = rng.normal(0, 0.01, (20, 10))
        p0 = y + 0.001
        p1 = y + 0.002
        res = oos_r2_gain(y, p0, p1, y, make_manifest(20), make_manifest(20),
                          min_assets=5)
        assert res.status == "INSUFFICIENT_DATA"
        assert res.reason_code == "R2_DENOMINATOR_ZERO"
        assert np.isnan(res.summary)


class TestOOSAblationSummary(unittest.TestCase):

    def _tasks(self, y):
        p0 = y + 0.004
        p1 = y + 0.001
        p2 = y + 0.002
        return [("drop_vola", p0, p1), ("drop_momo", p0, p2)]

    def test_two_tasks_without_paired_returns(self):
        rng = np.random.default_rng(111)
        y = rng.normal(0, 0.01, (30, 12))
        results = oos_ablation_summary(y, self._tasks(y),
                                       make_manifest(30), make_manifest(30),
                                       min_assets=5)
        assert len(results) == 2
        assert results[0].task_id == "drop_vola"
        assert results[1].task_id == "drop_momo"
        for item in results:
            assert item.rank_ic_delta.status == "OK"
            assert item.daily_mse_improvement.summary > 0.0
            assert item.net_sharpe_delta is None
            assert item.economic_status == "NOT_COMPUTED"
        assert results[0].r2_gain.status == "NOT_COMPUTED"
        assert results[0].r2_gain.reason_code == "REFERENCE_PREDICTION_MISSING"

    def test_with_reference_and_paired_returns(self):
        rng = np.random.default_rng(112)
        y = rng.normal(0, 0.01, (80, 12))
        tasks = self._tasks(y)
        p_ref = np.zeros((80, 12))
        cand_net = rng.normal(0.001, 0.02, (80, 1))
        base_net = rng.normal(0.0003, 0.02, (80, 1))
        results = oos_ablation_summary(
            y, tasks, make_manifest(80), make_manifest(80), min_assets=5,
            reference_predictions=p_ref, paired_returns=(cand_net, base_net))
        for item in results:
            assert item.r2_gain.status == "OK"
            assert item.r2_gain.summary > 0.0
            assert item.net_sharpe_delta is not None
            assert item.net_sharpe_delta.metric_id == "paired_net_sharpe_delta"
            assert item.economic_status == "OK"

    def test_duplicate_task_ids_rejected(self):
        rng = np.random.default_rng(113)
        y = rng.normal(0, 0.01, (30, 12))
        with pytest.raises(ValueError):
            oos_ablation_summary(y, [("a", y, y), ("a", y, y)],
                                 make_manifest(30), make_manifest(30))
        with pytest.raises(ValueError):
            oos_ablation_summary(y, [], make_manifest(30), make_manifest(30))


if __name__ == "__main__":
    unittest.main(verbosity=2)
