"""Golden-fixture tests for the extension CPU reference primitives and contracts.

The oracle section below reproduces the plan §12 reference algorithms verbatim
(independent small-data oracles; not production kernels).  Every primitive is
pinned to those oracles plus the plan's known values, per-seed byte
determinism, dtype rejection (bool/complex/object; FP32 promoted exactly for
FP64 reduction) and NaN/short-sample/boundary fail-closed behavior.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from quant_evaluator.contracts.errors import (
    InvalidContractError,
    MissingInputError,
    SchemaVersionError,
)
from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_instance import EvaluationScenario
from quant_evaluator.contracts.extension_inputs import (
    SCHEMA_VERSION,
    ExtensionInputs,
    OOSPredictionInput,
    PairedReturnInput,
    PredictionDistributionInput,
    ScenarioGridInput,
    TrialFamilyInput,
    UnboundInputRef,
)
from quant_evaluator.contracts.extension_policy import (
    ResamplingPlan,
    ResamplingPolicy,
    default_hac_max_lag,
)
from quant_evaluator.metrics.extension.primitives import (
    hac_lrv,
    make_resampling_plan,
    merge_path_summaries,
    path_block_summaries,
    path_summary_1d,
    prefix_moments,
    sample_moments,
    sample_path_mdd,
    shared_sort,
    upper_tail_mean,
    upper_tail_weights,
)
from quant_evaluator.metrics.robustness import compute_hac_tstat


# ===========================================================================
# Plan §12 oracle (verbatim algorithms; independent of production kernels)
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


def oracle_tail_weights(values, confidence: float) -> np.ndarray:
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


def oracle_upper_tail_mean(values, confidence: float) -> float:
    x = real_array(values, 1)
    w = oracle_tail_weights(x, confidence)
    return float(w @ x / w.sum())


def oracle_block_plan(sample_length: int, output_length: int, block_length: int,
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


def oracle_sample_moments_prefix(returns, starts, lengths) -> tuple[np.ndarray, np.ndarray]:
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


def oracle_path_summary(returns) -> tuple:
    r = real_array(returns, 1)
    if np.any(r <= -1):
        raise ValueError("Log-path scope requires r > -1; total loss is separately flagged")
    p = np.r_[0.0, np.cumsum(np.log1p(r))]
    return (float(p[-1]), float(p.max()), float(p.min()),
            float(np.max(np.maximum.accumulate(p) - p)))


def oracle_merge_path(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], max(a[1], a[0] + b[1]),
            min(a[2], a[0] + b[2]),
            max(a[3], b[3], a[1] - a[0] - b[2]))


def oracle_summary_mdd(summary: tuple) -> float:
    return float(-np.expm1(-summary[3]))


def oracle_sampled_mdd_from_summaries(returns, starts, lengths) -> np.ndarray:
    r = real_array(returns, 1)
    if np.any(r <= -1):
        raise ValueError("Out-of-domain log-path input")
    lookup = {(int(s), int(length)): oracle_path_summary(r[s:s + length])
              for length in np.unique(lengths)
              for s in range(len(r) - int(length) + 1)}
    out = np.empty(len(starts))
    for b in range(len(starts)):
        state = (0.0, 0.0, 0.0, 0.0)
        for s, length in zip(starts[b], lengths):
            state = oracle_merge_path(state, lookup[(int(s), int(length))])
        out[b] = oracle_summary_mdd(state)
    return out


def hac_oracle_omega(x, max_lag: int) -> float:
    """Direct plan §6.7 formula: γl with denominator T, fixed Bartlett weights."""
    x = np.asarray(x, dtype=np.float64)
    u = x - x.mean()
    n = len(x)
    omega = float(np.mean(u * u))
    for lag in range(1, max_lag + 1):
        gamma = float(np.sum(u[lag:] * u[:-lag]) / n)
        omega += 2.0 * (1.0 - lag / (max_lag + 1)) * gamma
    return omega


# ===========================================================================
# shared fixtures / helpers
# ===========================================================================

def axis(values, name: str) -> AxisRef:
    values = np.asarray(values)
    return AxisRef(name=name, dtype=str(values.dtype), size=len(values), values=values)


def make_policy(**overrides) -> ResamplingPolicy:
    fields = dict(block_length=7, repetitions=25, confidence_level=0.95, seed=0,
                  sample_length=103, output_length=103)
    fields.update(overrides)
    return ResamplingPolicy(**fields)


def make_paired_input(t: int = 5, f: int = 2, seed: int = 0) -> PairedReturnInput:
    rng = np.random.default_rng(seed)
    candidate = rng.normal(0.001, 0.02, (t, f))
    baseline = rng.normal(0.0, 0.01, (t, 1))
    return PairedReturnInput(
        candidate=candidate,
        baseline=baseline,
        baseline_map=tuple(f"b{i}" for i in range(f)),
        rf_daily=np.zeros(t, dtype=np.float64),
        time_axis=axis(np.arange(t), "time"),
        comparison_manifest={
            "baseline_portfolio_ref": "portfolio://base",
            "candidate_portfolio_ref": "portfolio://cand",
            "cost_policy_ref": "cost://net",
            "risk_policy_ref": "risk://base",
            "comparison_kind": "paired_difference",
        },
    )


def full_comparison_manifest() -> dict:
    return {
        "baseline_portfolio_ref": "portfolio://base",
        "candidate_portfolio_ref": "portfolio://cand",
        "cost_policy_ref": "cost://net",
        "risk_policy_ref": "risk://base",
        "comparison_kind": "paired_difference",
    }


def average_ranks(artifact) -> np.ndarray:
    """Scatter (run_start+run_end+1)/2 back to original positions (plan §6.2)."""
    ranks_sorted = (artifact.run_start + artifact.run_end_exclusive + 1) / 2.0
    ranks = np.empty_like(ranks_sorted)
    np.put_along_axis(ranks, artifact.order, ranks_sorted, axis=1)
    return ranks


# ===========================================================================
# ResamplingPlan (plan §6.3)
# ===========================================================================

class TestResamplingPlan:
    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_matches_oracle_bitwise(self, seed):
        plan = make_resampling_plan(make_policy(seed=seed), "time://session-1")
        oracle_starts, oracle_lengths = oracle_block_plan(103, 103, 7, 25, seed)
        np.testing.assert_array_equal(plan.starts, oracle_starts)
        np.testing.assert_array_equal(plan.lengths, oracle_lengths)

    def test_end_block_length(self):
        plan = make_resampling_plan(make_policy(), "time://s")
        assert plan.lengths[-1] == 5
        assert int(plan.lengths.sum()) == 103
        assert np.all(plan.starts <= 96)

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_deterministic_byte_identical(self, seed):
        a = make_resampling_plan(make_policy(seed=seed), "time://s")
        b = make_resampling_plan(make_policy(seed=seed), "time://s")
        assert a.starts.tobytes() == b.starts.tobytes()
        assert a.lengths.tobytes() == b.lengths.tobytes()
        assert a.plan_hash == b.plan_hash

    def test_policy_hash_changes_with_seed_and_lengths(self):
        base = make_policy()
        assert make_policy(seed=1).policy_hash != base.policy_hash
        assert make_policy(output_length=29).policy_hash != base.policy_hash
        assert make_policy(sample_length=120).policy_hash != base.policy_hash
        assert make_policy(block_length=10).policy_hash != base.policy_hash

    def test_plan_hash_changes_with_source_time_ref(self):
        p = make_policy()
        assert make_resampling_plan(p, "time://a").plan_hash \
            != make_resampling_plan(p, "time://b").plan_hash

    def test_rejects_bool_and_bad_values(self):
        with pytest.raises(InvalidContractError):
            ResamplingPolicy(block_length=True, repetitions=25, confidence_level=0.95,
                             seed=0, sample_length=103, output_length=103)
        with pytest.raises(InvalidContractError):
            make_policy(repetitions=1)  # B >= 2 required
        with pytest.raises(InvalidContractError):
            make_policy(block_length=200)  # sample_length >= block_length
        with pytest.raises(InvalidContractError):
            make_policy(confidence_level=1.0)
        with pytest.raises(InvalidContractError):
            make_policy(method="stationary")

    def test_plan_validates_starts_bounds_and_lengths_vector(self):
        policy = make_policy()
        plan = make_resampling_plan(policy, "time://s")
        ResamplingPlan(starts=plan.starts, lengths=plan.lengths,
                       source_time_ref="time://s", policy=policy)  # ok
        with pytest.raises(InvalidContractError):
            ResamplingPlan(starts=plan.starts + 500, lengths=plan.lengths,
                           source_time_ref="time://s", policy=policy)
        with pytest.raises(InvalidContractError):
            ResamplingPlan(starts=plan.starts, lengths=plan.lengths[:-1],
                           source_time_ref="time://s", policy=policy)
        with pytest.raises(InvalidContractError):
            ResamplingPlan(starts=plan.starts[:, :3], lengths=plan.lengths,
                           source_time_ref="time://s", policy=policy)


# ===========================================================================
# exact block moments (plan §6.4)
# ===========================================================================

class TestBlockMoments:
    def test_prefix_matches_day_by_day_expansion(self):
        r = np.random.default_rng(8).normal(0.001, 0.02, (103, 5))
        starts, lengths = oracle_block_plan(103, 103, 7, 31, 7)
        pm = prefix_moments(r)
        result = sample_moments(pm.center, pm.p1, pm.p2, starts, lengths)
        mu, var = oracle_sample_moments_prefix(r, starts, lengths)
        np.testing.assert_allclose(result.mean, mu, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(result.var, var, rtol=1e-11, atol=1e-14)
        assert result.valid.all()

    def test_factor_reordering_and_single_column_tiles(self):
        r = np.random.default_rng(3).normal(0, 0.03, (103, 5))
        starts, lengths = oracle_block_plan(103, 103, 7, 31)
        pm = prefix_moments(r)
        result = sample_moments(pm.center, pm.p1, pm.p2, starts, lengths)
        m, v = result.mean, result.var
        p = [4, 1, 2, 0, 3]
        pm_p = prefix_moments(r[:, p])
        result_p = sample_moments(pm_p.center, pm_p.p1, pm_p.p2, starts, lengths)
        mp, vp = result_p.mean, result_p.var
        np.testing.assert_allclose(mp, m[:, p], atol=1e-14)
        np.testing.assert_allclose(vp, v[:, p], atol=1e-14)
        for f in range(5):
            pm_f = prefix_moments(r[:, f:f + 1])
            result_f = sample_moments(pm_f.center, pm_f.p1, pm_f.p2, starts, lengths)
            np.testing.assert_allclose(result_f.mean[:, 0], m[:, f], atol=1e-14)
            np.testing.assert_allclose(result_f.var[:, 0], v[:, f], atol=1e-14)

    def test_constant_series_zero_variance(self):
        starts, lengths = oracle_block_plan(60, 60, 10, 11)
        pm = prefix_moments(np.full((60, 2), 0.002))
        result = sample_moments(pm.center, pm.p1, pm.p2, starts, lengths)
        np.testing.assert_allclose(result.mean, 0.002, atol=1e-15)
        np.testing.assert_allclose(result.var, 0, atol=1e-30)

    def test_large_offset_variance(self):
        r = 1e8 + np.random.default_rng(1).normal(0, 0.1, (103, 3))
        starts, lengths = oracle_block_plan(103, 103, 7, 5)
        pm = prefix_moments(r)
        result = sample_moments(pm.center, pm.p1, pm.p2, starts, lengths)
        for b in range(len(starts)):
            sample = r[expand_draw(starts[b], lengths)]
            np.testing.assert_allclose(result.mean[b], sample.mean(0), rtol=1e-14)
            np.testing.assert_allclose(result.var[b], sample.var(0, ddof=1),
                                       rtol=1e-10, atol=1e-13)

    def test_fp32_input_equals_fp64_reduction_exactly(self):
        r32 = np.random.default_rng(11).normal(0, 0.01, (40, 3)).astype(np.float32)
        r64 = r32.astype(np.float64)  # exact promotion; same values, no precision loss
        starts, lengths = oracle_block_plan(40, 40, 5, 9, 3)
        pm32 = prefix_moments(r32)
        pm64 = prefix_moments(r64)
        assert pm32.p1.dtype == np.float64  # reduced in FP64
        out32 = sample_moments(pm32.center, pm32.p1, pm32.p2, starts, lengths)
        out64 = sample_moments(pm64.center, pm64.p1, pm64.p2, starts, lengths)
        assert out32.mean.tobytes() == out64.mean.tobytes()
        assert out32.var.tobytes() == out64.var.tobytes()

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_deterministic_byte_identical(self, seed):
        r = np.random.default_rng(seed + 50).normal(0.001, 0.02, (60, 4))
        starts, lengths = oracle_block_plan(60, 60, 8, 17, seed)
        pm = prefix_moments(r)
        a = sample_moments(pm.center, pm.p1, pm.p2, starts, lengths)
        b = sample_moments(pm.center, pm.p1, pm.p2, starts, lengths)
        assert a.mean.tobytes() == b.mean.tobytes()
        assert a.var.tobytes() == b.var.tobytes()

    def test_dtype_rejection(self):
        for bad in (np.array([[True, False], [False, True]]),
                    np.array([[1 + 2j, 0], [0, 1 + 2j]]),
                    np.array([["a", "b"], ["c", "d"]])):
            with pytest.raises(InvalidContractError):
                prefix_moments(bad)

    def test_nan_inf_rejected_no_compression(self):
        r = np.random.default_rng(2).normal(0, 0.01, (30, 2))
        r[10, 0] = np.nan
        with pytest.raises(InvalidContractError):
            prefix_moments(r)
        r[10, 0] = np.inf
        with pytest.raises(InvalidContractError):
            prefix_moments(r)

    def test_short_sample_rejected(self):
        r = np.random.default_rng(4).normal(0, 0.01, (1, 2))
        pm = prefix_moments(r)
        with pytest.raises(InvalidContractError):
            sample_moments(pm.center, pm.p1, pm.p2, np.zeros((3, 1), dtype=np.int64),
                           np.array([1], dtype=np.int64))

    def test_out_of_range_starts_rejected(self):
        r = np.random.default_rng(5).normal(0, 0.01, (20, 2))
        pm = prefix_moments(r)
        with pytest.raises(InvalidContractError):
            sample_moments(pm.center, pm.p1, pm.p2,
                           np.array([[18]], dtype=np.int64), np.array([5], dtype=np.int64))


# ===========================================================================
# HAC long-run variance (plan §6.7)
# ===========================================================================

class TestHacLrv:
    def test_matches_direct_formula(self):
        rng = np.random.default_rng(21)
        x = np.cumsum(rng.normal(0, 1, 120)) * 0.1 + rng.normal(0, 0.5, 120)
        result = hac_lrv(x[:, None], max_lag=8)
        assert result.valid[0]
        np.testing.assert_allclose(result.omega2[0], hac_oracle_omega(x, 8), rtol=1e-12)

    def test_matches_existing_tstat_kernel(self):
        rng = np.random.default_rng(22)
        x = np.cumsum(rng.normal(0, 1, 200)) * 0.05 + 0.01
        result = hac_lrv(x[:, None], max_lag=10)
        tstat, se = compute_hac_tstat(x[:, None], max_lag=10, kernel="bartlett")
        assert result.valid[0]
        np.testing.assert_allclose(se[0], np.sqrt(result.omega2[0] / len(x)), rtol=1e-12)
        np.testing.assert_allclose(tstat[0], x.mean() / se[0], rtol=1e-12)

    def test_per_column_independence(self):
        rng = np.random.default_rng(23)
        cols = [np.cumsum(rng.normal(0, 1, 90)) * 0.1 for _ in range(3)]
        matrix = np.column_stack(cols)
        result = hac_lrv(matrix, max_lag=5)
        for i, col in enumerate(cols):
            np.testing.assert_allclose(result.omega2[i], hac_oracle_omega(col, 5), rtol=1e-12)
        assert result.valid.all()

    def test_internal_nan_marked_invalid_not_compressed(self):
        x = np.arange(60.0)
        y = x.copy()
        y[30] = np.nan
        result = hac_lrv(np.column_stack([x, y]), max_lag=5)
        assert result.valid[0]
        assert not result.valid[1]
        assert np.isnan(result.omega2[1])

    def test_short_sample_below_kernel_floor_invalid(self):
        x = np.arange(12.0)  # needs max_lag + 10
        result = hac_lrv(x[:, None], max_lag=5)
        assert not result.valid[0]

    def test_constant_series_rounding_dust_only(self):
        # A constant column demeans to rounding dust: Ω ends up ~1e-36 (the
        # existing kernel's documented demean-then-reduce behavior), positive
        # hence valid; no epsilon is ADDED anywhere (plan §6.7).
        x = np.full((80, 1), 0.003)
        result = hac_lrv(x, max_lag=5)
        assert 0.0 <= result.omega2[0] < 1e-30
        assert result.valid[0]

    def test_all_zero_column_exact_zero_invalid(self):
        # u ≡ 0 exactly: Ω == 0.0 exactly -> invalid, never epsilon-padded.
        result = hac_lrv(np.zeros((80, 1)), max_lag=5)
        assert result.omega2[0] == 0.0
        assert not result.valid[0]

    def test_alternating_series_golden_lag_convention(self):
        # Pins the plan §6.7 convention: γl = T^-1 Σ_{t=l+1}^T u_t u_{t-l}
        # (T-l terms over denominator T).  u = ±1 alternating, T=80:
        # γ0 = 1, γ1 = -79/80, Ω = 1 + 2·(1/2)·(-79/80) = 1/80 exactly.
        x = np.array([1.0, -1.0] * 40)
        result = hac_lrv(x[:, None], max_lag=1)
        assert result.valid[0]
        np.testing.assert_allclose(result.omega2[0], 0.0125, rtol=1e-12)

    def test_dtype_rejection(self):
        with pytest.raises(InvalidContractError):
            hac_lrv(np.array([[True], [False]] * 30), max_lag=2)
        with pytest.raises(InvalidContractError):
            hac_lrv(np.array([[1 + 1j]] * 30), max_lag=2)
        with pytest.raises(InvalidContractError):
            hac_lrv(np.array([["a"]] * 30), max_lag=2)

    def test_max_lag_validation(self):
        x = np.arange(60.0)
        with pytest.raises(InvalidContractError):
            hac_lrv(x[:, None], max_lag=True)
        with pytest.raises(InvalidContractError):
            hac_lrv(x[:, None], max_lag=-1)


# ===========================================================================
# ordered path summaries / exact MDD (plan §6.5)
# ===========================================================================

class TestPathSummaries:
    def test_initial_wealth_included(self):
        np.testing.assert_allclose(
            oracle_summary_mdd(tuple(path_summary_1d(np.array([-.1, .1])))), 0.1)

    def test_all_positive_zero_drawdown(self):
        assert float(path_summary_1d(np.array([.1, .02, .03]))[3]) == 0.0

    def test_cross_block_drawdown(self):
        a = path_summary_1d(np.array([.2, -.1]))
        b = path_summary_1d(np.array([-.2, .1]))
        merged = merge_path_summaries(a, b)
        np.testing.assert_allclose(-np.expm1(-merged[3]), 0.28)
        direct = path_summary_1d(np.array([.2, -.1, -.2, .1]))
        np.testing.assert_allclose(merged, direct, atol=1e-14)

    def test_path_associativity(self):
        a = path_summary_1d(np.array([.1, -.15]))
        b = path_summary_1d(np.array([.02, -.03]))
        c = path_summary_1d(np.array([.05, -.1]))
        np.testing.assert_allclose(merge_path_summaries(merge_path_summaries(a, b), c),
                                   merge_path_summaries(a, merge_path_summaries(b, c)),
                                   atol=1e-14)

    def test_path_non_commutative(self):
        a = path_summary_1d(np.array([.2, -.1]))
        b = path_summary_1d(np.array([-.2, .1]))
        assert not np.isclose(merge_path_summaries(a, b)[3], merge_path_summaries(b, a)[3])

    def test_identity_element(self):
        a = path_summary_1d(np.array([.1, -.2, .02]))
        identity = np.zeros(4)
        np.testing.assert_allclose(merge_path_summaries(a, identity), a, atol=1e-14)
        np.testing.assert_allclose(merge_path_summaries(identity, a), a, atol=1e-14)

    def test_sampled_mdd_matches_oracle_and_naive(self):
        r = np.random.default_rng(3).normal(0.001, 0.03, 103)
        starts, lengths = oracle_block_plan(103, 29, 7, 37, 1)
        block = path_block_summaries(r[:, None], lengths)
        result = sample_path_mdd(block.summary_lookup, starts, lengths)
        oracle = oracle_sampled_mdd_from_summaries(r, starts, lengths)
        naive = np.array([oracle_summary_mdd(oracle_path_summary(r[expand_draw(s, lengths)]))
                          for s in starts])
        np.testing.assert_allclose(result.mdd[:, 0], oracle, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result.mdd[:, 0], naive, rtol=1e-12, atol=1e-14)
        assert result.valid[0]

    def test_multi_factor_columns_independent(self):
        rng = np.random.default_rng(6)
        r = np.abs(rng.normal(0.001, 0.02, (80, 4)))
        starts, lengths = oracle_block_plan(80, 40, 6, 23, 2)
        block = path_block_summaries(r, lengths)
        result = sample_path_mdd(block.summary_lookup, starts, lengths)
        for f in range(4):
            oracle = oracle_sampled_mdd_from_summaries(r[:, f], starts, lengths)
            np.testing.assert_allclose(result.mdd[:, f], oracle, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_deterministic_byte_identical(self, seed):
        r = np.abs(np.random.default_rng(seed).normal(0.001, 0.02, (50, 2)))
        starts, lengths = oracle_block_plan(50, 24, 5, 11, seed)
        a = sample_path_mdd(path_block_summaries(r, lengths).summary_lookup, starts, lengths)
        b = sample_path_mdd(path_block_summaries(r, lengths).summary_lookup, starts, lengths)
        assert a.mdd.tobytes() == b.mdd.tobytes()

    def test_bankruptcy_and_basis_mismatch_rejected(self):
        for value in (-1.0, -1.1):
            with pytest.raises(InvalidContractError):
                path_summary_1d(np.array([.01, value]))
            with pytest.raises(InvalidContractError):
                path_block_summaries(np.array([[.01, value]]).T, np.array([2], dtype=np.int64))

    def test_dtype_rejection(self):
        with pytest.raises(InvalidContractError):
            path_block_summaries(np.array([[True], [False]] * 5), np.array([2], dtype=np.int64))
        with pytest.raises(InvalidContractError):
            path_block_summaries(np.array([[1 + 1j]] * 10), np.array([2], dtype=np.int64))

    def test_lookup_covers_only_legal_starts(self):
        r = np.abs(np.random.default_rng(9).normal(0.001, 0.01, (20, 1)))
        lengths = np.array([4, 6], dtype=np.int64)
        block = path_block_summaries(r, lengths)
        # start range for length 6 is 0..14; row 15 col for slot of 6 must be NaN
        slot = int(np.searchsorted(block.length_slots, 6))
        assert np.isnan(block.summary_lookup[15, slot, 0, 0])
        assert np.isfinite(block.summary_lookup[15, int(np.searchsorted(block.length_slots, 4)), 0, 0])


# ===========================================================================
# upper tail mean (plan §6.6)
# ===========================================================================

class TestUpperTailMean:
    def test_golden_fractional_mass(self):
        result = upper_tail_mean(np.array([[1.], [2.], [3.], [4.]]), confidence=0.625)
        np.testing.assert_allclose(result.value[0], 11 / 3)
        assert result.valid[0]

    def test_golden_integer_mass(self):
        result = upper_tail_mean(np.array([[1.], [2.], [3.], [4.]]), confidence=0.5)
        np.testing.assert_allclose(result.value[0], 3.5)

    def test_golden_ties_distribute(self):
        w = upper_tail_weights(np.array([[3.], [3.], [1.], [0.]]), confidence=0.625)
        np.testing.assert_allclose(w[:, 0], [0.75, 0.75, 0, 0])

    def test_tied_dates_permutation_invariant(self):
        book = np.array([[3.], [3.], [1.], [0.]])
        target = np.array([[1.], [7.], [10.], [20.]])
        p = [2, 1, 3, 0]
        w1 = upper_tail_weights(book, confidence=0.625)
        w2 = upper_tail_weights(book[p], confidence=0.625)
        v1 = float(w1[:, 0] @ target[:, 0] / w1[:, 0].sum())
        v2 = float(w2[:, 0] @ target[p][:, 0] / w2[:, 0].sum())
        np.testing.assert_allclose(v1, v2)

    def test_matches_oracle(self):
        rng = np.random.default_rng(31)
        x = rng.normal(0, 1, (200, 3))
        for confidence in (0.9, 0.95, 0.625):
            result = upper_tail_mean(x, confidence=confidence)
            for f in range(3):
                np.testing.assert_allclose(result.value[f],
                                           oracle_upper_tail_mean(x[:, f], confidence),
                                           rtol=1e-13)

    def test_min_mass_gating(self):
        x = np.arange(1.0, 11.0)[:, None]  # n=10, q=0.95 -> m=0.5
        result = upper_tail_mean(x, confidence=0.95, min_mass=10.0)
        assert not result.valid[0]  # 0.5 < 10 required tail mass
        result_ok = upper_tail_mean(x, confidence=0.95, min_mass=0.5)
        assert result_ok.valid[0]

    def test_invalid_columns_fail_closed(self):
        x = np.array([[1.], [np.nan], [3.], [4.]])
        result = upper_tail_mean(x, confidence=0.5)
        assert not result.valid[0]
        assert np.isnan(result.value[0])
        empty = upper_tail_mean(np.empty((0, 1)), confidence=0.9)
        assert not empty.valid[0]
        with pytest.raises(InvalidContractError):
            upper_tail_mean(np.ones((4, 1)), confidence=1.0)
        with pytest.raises(InvalidContractError):
            upper_tail_mean(np.ones((4, 1)), confidence=0)

    def test_dtype_rejection(self):
        with pytest.raises(InvalidContractError):
            upper_tail_mean(np.array([[True], [False], [True], [False]]), confidence=0.5)
        with pytest.raises(InvalidContractError):
            upper_tail_mean(np.array([[1j]] * 4), confidence=0.5)

    def test_fp32_column_matches_fp64(self):
        x32 = np.random.default_rng(33).normal(0, 1, (50, 1)).astype(np.float32)
        x64 = x32.astype(np.float64)  # exact promotion of the same values
        a = upper_tail_mean(x32, confidence=0.9)
        b = upper_tail_mean(x64, confidence=0.9)
        assert a.value.tobytes() == b.value.tobytes()


# ===========================================================================
# shared sort (plan §6.2)
# ===========================================================================

class TestSharedSort:
    def test_average_rank_with_ties(self):
        values = np.array([[[3., 1., 3., 2.]]])  # (T=1,F=1,N=4) -> one R row
        mask = np.ones_like(values, dtype=bool)
        artifact = shared_sort(values, mask, products=("rank", "distinct_count"))
        ranks = average_ranks(artifact)
        np.testing.assert_allclose(ranks[0], [3.5, 1.0, 3.5, 2.0])
        # distinct levels via run boundaries: 3 unique values
        starts = artifact.run_start[0]
        assert (starts == [0, 1, 2, 2]).all()
        ends = artifact.run_end_exclusive[0]
        assert (ends == [1, 2, 4, 4]).all()

    def test_masked_entries_excluded_and_sentineled(self):
        values = np.array([[[5., 1., 5., 2.]]])
        mask = np.array([[[True, True, False, True]]])
        artifact = shared_sort(values, mask, products=("rank",))
        assert artifact.finite_count[0] == 3
        sorted_values = artifact.sorted_values[0]
        np.testing.assert_allclose(sorted_values, [1., 2., 5., 5.])  # masked 5 at tail
        assert artifact.order[0, -1] == 2  # masked position sorted last (stable)

    def test_masked_values_never_share_runs(self):
        values = np.array([[[1., 1., 1., 1.]]])
        mask = np.array([[[True, True, True, False]]])
        artifact = shared_sort(values, mask, products=("rank",))
        starts = artifact.run_start[0]
        ends = artifact.run_end_exclusive[0]
        # valid run [0,3); invalid entry is its own empty run
        np.testing.assert_array_equal(starts, [0, 0, 0, 3])
        np.testing.assert_array_equal(ends, [3, 3, 3, 4])

    def test_fp32_ranks_match_fp64_exactly(self):
        rng = np.random.default_rng(41)
        v64 = rng.normal(0, 1, (2, 2, 6))
        v32 = v64.astype(np.float32)
        mask = np.ones((2, 2, 6), dtype=bool)
        mask[0, 1, 3] = False
        a = shared_sort(v32, mask, products=("rank",))
        b = shared_sort(v64, mask, products=("rank",))
        assert a.order.tobytes() == b.order.tobytes()
        assert a.sorted_values.dtype == np.float32
        assert str(a.dtype) == "float32"

    def test_products_validation(self):
        values = np.zeros((1, 1, 2))
        mask = np.ones((1, 1, 2), dtype=bool)
        with pytest.raises(InvalidContractError):
            shared_sort(values, mask, products=())
        with pytest.raises(InvalidContractError):
            shared_sort(values, mask, products=("median",))

    def test_dtype_and_mask_validation(self):
        values = np.zeros((1, 1, 2))
        with pytest.raises(InvalidContractError):
            shared_sort(np.array([[[True, False]]]), np.ones((1, 1, 2), dtype=bool))
        with pytest.raises(InvalidContractError):
            shared_sort(values, np.ones((1, 1, 2)))  # non-bool mask

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, (3, 2, 8))
        mask = rng.random((3, 2, 8)) > 0.2
        a = shared_sort(values, mask, products=("rank", "distinct_count"))
        b = shared_sort(values, mask, products=("rank", "distinct_count"))
        assert a.order.tobytes() == b.order.tobytes()
        assert a.sorted_values.tobytes() == b.sorted_values.tobytes()

    def test_nan_treated_as_invalid(self):
        values = np.array([[[1., np.nan, 0.5, 2.]]])
        mask = np.ones_like(values, dtype=bool)
        artifact = shared_sort(values, mask, products=("rank",))
        assert artifact.finite_count[0] == 3
        # NaN is behind the +inf sentinel: its slot is the tail
        assert artifact.order[0, -1] == 1


# ===========================================================================
# nw_rule_v1 bandwidth (plan §6.1)
# ===========================================================================

class TestDefaultHacMaxLag:
    @pytest.mark.parametrize("t,expected", [(59, 3), (100, 4), (252, 4), (1000, 6)])
    def test_golden_values(self, t, expected):
        assert default_hac_max_lag(t) == expected

    def test_declared_min_lag_is_floor(self):
        assert default_hac_max_lag(100, declared_min_lag=8) == 8
        assert default_hac_max_lag(1000, declared_min_lag=2) == 6

    def test_rejects_bad_input(self):
        with pytest.raises(InvalidContractError):
            default_hac_max_lag(True)
        with pytest.raises(InvalidContractError):
            default_hac_max_lag(0)


# ===========================================================================
# contracts: extension inputs (plan §5.1 / §8.2)
# ===========================================================================

class TestPairedReturnInput:
    def test_construction_and_content_ref(self):
        inp = make_paired_input()
        assert len(inp.content_ref) == 64
        again = make_paired_input()
        assert inp.content_ref == again.content_ref  # deterministic across processes

    def test_content_hash_binds_payload_bytes(self):
        inp = make_paired_input(seed=1)
        changed = make_paired_input(seed=1)
        tampered = np.array(changed.candidate, copy=True)
        tampered[0, 0] += 1e-12
        changed2 = PairedReturnInput(
            candidate=tampered, baseline=changed.baseline, baseline_map=changed.baseline_map,
            rf_daily=changed.rf_daily, time_axis=changed.time_axis,
            comparison_manifest=dict(changed.comparison_manifest))
        assert inp.content_ref == changed.content_ref
        assert inp.content_ref != changed2.content_ref

    def test_fail_closed_missing_evidence(self):
        good = make_paired_input()
        with pytest.raises(InvalidContractError):
            PairedReturnInput(
                candidate=None, baseline=good.baseline, baseline_map=good.baseline_map,
                rf_daily=good.rf_daily, time_axis=good.time_axis,
                comparison_manifest=full_comparison_manifest())
        with pytest.raises(InvalidContractError):  # baseline_map wrong length
            PairedReturnInput(
                candidate=good.candidate, baseline=good.baseline, baseline_map=("only",),
                rf_daily=good.rf_daily, time_axis=good.time_axis,
                comparison_manifest=full_comparison_manifest())
        with pytest.raises(InvalidContractError):  # rf_daily must be float64
            PairedReturnInput(
                candidate=good.candidate, baseline=good.baseline, baseline_map=good.baseline_map,
                rf_daily=np.zeros(5, dtype=np.float32), time_axis=good.time_axis,
                comparison_manifest=full_comparison_manifest())
        with pytest.raises(InvalidContractError):  # incomplete comparison protocol
            PairedReturnInput(
                candidate=good.candidate, baseline=good.baseline, baseline_map=good.baseline_map,
                rf_daily=good.rf_daily, time_axis=good.time_axis,
                comparison_manifest={"comparison_kind": "paired"})

    def test_dtype_rejection(self):
        good = make_paired_input()
        with pytest.raises(InvalidContractError):
            PairedReturnInput(
                candidate=(good.candidate > 0), baseline=good.baseline,
                baseline_map=good.baseline_map, rf_daily=good.rf_daily,
                time_axis=good.time_axis, comparison_manifest=full_comparison_manifest())
        with pytest.raises(InvalidContractError):
            PairedReturnInput(
                candidate=good.candidate + 0j, baseline=good.baseline,
                baseline_map=good.baseline_map, rf_daily=good.rf_daily,
                time_axis=good.time_axis, comparison_manifest=full_comparison_manifest())

    def test_refs_only_round_trip(self):
        inp = make_paired_input()
        payload = inp.to_dict()
        assert set(payload) == {"schema_version", "input_type", "content_ref", "bound"}
        assert "candidate" not in payload  # arrays are never embedded (plan §5.1)
        unbound = PairedReturnInput.from_dict(payload)
        assert isinstance(unbound, UnboundInputRef)
        assert not unbound.bound
        assert unbound.content_ref == inp.content_ref
        with pytest.raises(MissingInputError):
            unbound.require()

    def test_from_dict_rejects_unknown_keys_and_bad_schema(self):
        inp = make_paired_input()
        payload = inp.to_dict()
        with pytest.raises(TypeError):
            PairedReturnInput.from_dict({**payload, "candidate": [[1]]})
        with pytest.raises(SchemaVersionError):
            PairedReturnInput.from_dict({**payload, "schema_version": "9.9.9"})

    def test_frozen(self):
        inp = make_paired_input()
        with pytest.raises(Exception):
            inp.candidate = np.zeros((5, 2))  # type: ignore[misc]
        # caller-side aliasing cannot mutate the frozen payload
        outer = np.array(inp.candidate, copy=True)
        outer[0, 0] = 999.0
        np.testing.assert_array_equal(inp.candidate, make_paired_input().candidate)


class TestOOSPredictionInput:
    def _build(self, t=6, n=3, f=2):
        rng = np.random.default_rng(7)
        return OOSPredictionInput(
            baseline=rng.normal(0, 1, (t, n)),
            candidates=rng.normal(0, 1, (t, n, f)),
            reference_prediction=rng.normal(0, 1, (t, n)),
            time_axis=axis(np.arange(t), "time"),
            asset_axis=axis(np.arange(n), "asset"),
            candidate_ids=tuple(f"c{i}" for i in range(f)),
            validity=np.ones((t, n, f), dtype=bool),
            fit_manifests=tuple({"model_ref": "m", "train_window": "w"} for _ in range(f)),
            prediction_available_at=np.zeros(t, dtype=np.int64),
            labels_ref="label://daily",
        )

    def test_round_trip_and_hash(self):
        inp = self._build()
        assert inp.content_ref == self._build().content_ref
        unbound = OOSPredictionInput.from_dict(inp.to_dict())
        assert unbound.content_ref == inp.content_ref

    def test_reference_prediction_required(self):
        inp = self._build()
        with pytest.raises(InvalidContractError):
            OOSPredictionInput(
                baseline=inp.baseline, candidates=inp.candidates, reference_prediction=None,
                time_axis=inp.time_axis, asset_axis=inp.asset_axis,
                candidate_ids=inp.candidate_ids, validity=inp.validity,
                fit_manifests=inp.fit_manifests,
                prediction_available_at=inp.prediction_available_at,
                labels_ref=inp.labels_ref)

    def test_shape_and_id_validation(self):
        inp = self._build()
        with pytest.raises(InvalidContractError):
            OOSPredictionInput(
                baseline=inp.baseline, candidates=inp.candidates[:, :, :1],
                reference_prediction=inp.reference_prediction,
                time_axis=inp.time_axis, asset_axis=inp.asset_axis,
                candidate_ids=inp.candidate_ids, validity=inp.validity,
                fit_manifests=inp.fit_manifests,
                prediction_available_at=inp.prediction_available_at,
                labels_ref=inp.labels_ref)
        with pytest.raises(InvalidContractError):  # duplicate candidate ids
            OOSPredictionInput(
                baseline=inp.baseline, candidates=inp.candidates,
                reference_prediction=inp.reference_prediction,
                time_axis=inp.time_axis, asset_axis=inp.asset_axis,
                candidate_ids=("c", "c"), validity=inp.validity,
                fit_manifests=inp.fit_manifests,
                prediction_available_at=inp.prediction_available_at,
                labels_ref=inp.labels_ref)


class TestTrialFamilyInput:
    def _time_axis(self, t=8):
        return axis(np.arange(t), "time")

    def test_requires_complete_payload_not_winner_only(self):
        ledger = {"loss_formula_ref": "l", "window": "w", "direction": "d",
                  "universe_ref": "u", "model_ref": "m", "parameters": {}}
        with pytest.raises(InvalidContractError):  # no numeric payload at all
            TrialFamilyInput(trial_ids=("t1", "t2"), time_axis=self._time_axis(),
                             ledger_manifest=ledger, loss_kind="mse",
                             cost_ref="cost://x", portfolio_ref="p://x",
                             sampling_scope="family")
        inp = TrialFamilyInput(trial_ids=("t1", "t2"), time_axis=self._time_axis(),
                               ledger_manifest=ledger, loss_kind="mse",
                               cost_ref="cost://x", portfolio_ref="p://x",
                               differentials=np.zeros((8, 2)), sampling_scope="family")
        assert inp.content_ref == TrialFamilyInput(
            trial_ids=("t1", "t2"), time_axis=self._time_axis(), ledger_manifest=ledger,
            loss_kind="mse", cost_ref="cost://x", portfolio_ref="p://x",
            differentials=np.zeros((8, 2)), sampling_scope="family").content_ref

    def test_family_width_frozen_by_trial_ids(self):
        ledger = {"loss_formula_ref": "l", "window": "w", "direction": "d",
                  "universe_ref": "u", "model_ref": "m", "parameters": {}}
        with pytest.raises(InvalidContractError):  # C mismatch: winner-only array
            TrialFamilyInput(trial_ids=("t1", "t2", "t3"), time_axis=self._time_axis(),
                             ledger_manifest=ledger, loss_kind="mse",
                             cost_ref="cost://x", portfolio_ref="p://x",
                             differentials=np.zeros((8, 1)), sampling_scope="family")

    def test_ledger_manifest_keys_required(self):
        with pytest.raises(InvalidContractError):
            TrialFamilyInput(trial_ids=("t1", "t2"), time_axis=self._time_axis(),
                             ledger_manifest={"note": "trust me"}, loss_kind="mse",
                             cost_ref="cost://x", portfolio_ref="p://x",
                             differentials=np.zeros((8, 2)), sampling_scope="family")


class TestScenarioGridInput:
    @staticmethod
    def _scenario():
        t = np.arange(5).astype("datetime64[s]").tolist()
        bundle = LabelBundle(
            target_id="t", values=np.zeros((2, 1)), horizon=1,
            signal_available_time=t[:2], decision_time=t[1:3],
            execution_time=t[1:3], label_start_time=t[2:4], label_end_time=t[3:5],
        )
        return EvaluationScenario(label_bundle=bundle)

    def test_grid_and_reference_validation(self):
        coords = (("d0", "h1"), ("d0", "h5"), ("d1", "h1"))
        inp = ScenarioGridInput(
            ordered_coordinates=coords,
            scenarios={coord: self._scenario() for coord in coords},
            reference_coordinate=("d0", "h1"),
            sample_policy="common_continuous_grid",
            scenario_config_ref="scenario-config://v1",
        )
        assert inp.content_ref == ScenarioGridInput(
            ordered_coordinates=coords,
            scenarios={coord: self._scenario() for coord in coords},
            reference_coordinate=("d0", "h1"),
            sample_policy="common_continuous_grid",
            scenario_config_ref="scenario-config://v1",
        ).content_ref

    def test_missing_cell_rejected(self):
        coords = (("d0", "h1"), ("d0", "h5"))
        with pytest.raises(InvalidContractError):
            ScenarioGridInput(
                ordered_coordinates=coords,
                scenarios={("d0", "h1"): self._scenario()},  # incomplete grid
                reference_coordinate=("d0", "h1"),
                sample_policy="p", scenario_config_ref="ref://x")

    def test_string_scenario_rejected(self):
        coords = (("d0", "h1"),)
        with pytest.raises(InvalidContractError):
            ScenarioGridInput(
                ordered_coordinates=coords,
                scenarios={("d0", "h1"): "complete=True"},  # a string is not evidence
                reference_coordinate=("d0", "h1"),
                sample_policy="p", scenario_config_ref="ref://x")


class TestPredictionDistributionInput:
    def test_quantile_kind_contract(self):
        t, n, m = 5, 2, 3
        inp = PredictionDistributionInput(
            output_kind="quantiles",
            prediction=np.sort(np.random.default_rng(1).normal(0, 1, (t, n, m)), axis=2),
            target=np.zeros((t, n)),
            validity=np.ones((t, n, m), dtype=bool),
            time_axis=axis(np.arange(t), "time"),
            asset_axis=axis(np.arange(n), "asset"),
            candidate_ids=(),
            available_at=np.zeros(t, dtype=np.int64),
            model_ref="model://x",
            quantile_levels=np.array([0.1, 0.5, 0.9]),
        )
        assert inp.content_ref == inp.content_ref
        with pytest.raises(InvalidContractError):  # non-monotone quantile levels
            PredictionDistributionInput(
                output_kind="quantiles",
                prediction=np.zeros((t, n, m)),
                target=np.zeros((t, n)),
                validity=np.ones((t, n, m), dtype=bool),
                time_axis=axis(np.arange(t), "time"),
                asset_axis=axis(np.arange(n), "asset"),
                candidate_ids=(),
                available_at=np.zeros(t, dtype=np.int64),
                model_ref="model://x",
                quantile_levels=np.array([0.5, 0.1, 0.9]),
            )

    def test_binary_probability_range(self):
        t, n, f = 4, 2, 2
        with pytest.raises(InvalidContractError):
            PredictionDistributionInput(
                output_kind="binary",
                prediction=np.full((t, n, f), 1.5),
                target=np.zeros((t, n)),
                validity=np.ones((t, n, f), dtype=bool),
                time_axis=axis(np.arange(t), "time"),
                asset_axis=axis(np.arange(n), "asset"),
                candidate_ids=("a", "b"),
                available_at=np.zeros(t, dtype=np.int64),
                model_ref="model://x",
            )


class TestExtensionInputs:
    def test_fixed_fourteen_fields(self):
        assert len(ExtensionInputs.__dataclass_fields__) == 14

    def test_unknown_field_type_error(self):
        with pytest.raises(TypeError):
            ExtensionInputs(bogus_payload=1)
        with pytest.raises(TypeError):
            ExtensionInputs.from_dict({
                "schema_version": SCHEMA_VERSION,
                "content_ref": "x" * 64,
                "inputs": {},
                "bogus": 1,
            })

    def test_require_fails_closed_on_missing(self):
        inputs = ExtensionInputs()
        with pytest.raises(MissingInputError) as exc:
            inputs.require("paired_returns")
        assert "SOURCE_ARTIFACT_MISSING" in str(exc.value)
        with pytest.raises(TypeError):
            inputs.require("not_a_field")

    def test_container_round_trip_preserves_content_refs(self):
        inputs = ExtensionInputs(paired_returns=make_paired_input())
        payload = inputs.to_dict()
        restored = ExtensionInputs.from_dict(payload)
        assert restored.content_ref == inputs.content_ref
        assert restored.refs["paired_returns"].content_ref == inputs.paired_returns.content_ref
        assert restored.refs["trial_family"] is None
        with pytest.raises(MissingInputError):
            restored.require_bound("paired_returns", "trial_family")

    def test_container_hash_tracks_sidecar_content(self):
        a = ExtensionInputs(paired_returns=make_paired_input(seed=1))
        b = ExtensionInputs(paired_returns=make_paired_input(seed=1))
        c = ExtensionInputs(paired_returns=make_paired_input(seed=2))
        assert a.content_ref == b.content_ref
        assert a.content_ref != c.content_ref

    def test_wrong_type_rejected(self):
        with pytest.raises(InvalidContractError):
            ExtensionInputs(paired_returns="net")
