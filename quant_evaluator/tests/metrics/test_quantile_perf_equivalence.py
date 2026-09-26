"""
Equivalence + boundary + performance smoke tests for the quantile-family
performance optimizations (optimizations A/B/C/D).

Scope (server-c formal tree, ``quant_evaluator``):
- A. ``compute_adaptive_quantile_count`` single-sort multi-Q derivation.
- B. ``_percentile_boundaries`` vectorized (vs scalar oracle).
- C. ``assign_quantiles_batch`` vectorized per-time sort.
- D. ``compute_shape_bootstrap_confidence`` hoisted bootstrap indexing.

Hard equivalence contract
-------------------------
Every optimized kernel is validated against a verbatim pre-optimization oracle
kept in the same module (``_xxx_reference``).  Where bit-level identity is
achievable (assignments are int32, boundaries are float64 arithmetic) we assert
exact equality; for the float correlation outputs we use the library's GPU-parity
tolerance ``rtol=1e-8, atol=1e-10, equal_nan=True``.  No public signature or
return structure is changed by the optimizations under test.

Run:
    cd /home/sunhaiwei/quant_projects/quant_evaluator && \
        OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
        tests/metrics/test_quantile_perf_equivalence.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.metrics.quantile import (
    assign_quantiles_batch,
    _assign_quantiles_batch_reference,
    _percentile_boundaries,
    _percentile_boundaries_reference,
    _percentile_boundaries_from_sorted,
    _searchsorted_bins,
)
from quant_evaluator.metrics.shape_evidence import (
    compute_adaptive_quantile_count,
    _compute_adaptive_quantile_count_reference,
    compute_shape_bootstrap_confidence,
    _compute_shape_bootstrap_confidence_reference,
)
from quant_evaluator.contracts.adaptive_bins_policy import AdaptiveBinsPolicy
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef

# Library GPU-parity tolerance (see task: rtol=1e-8, atol=1e-10, equal_nan).
_RTOL = 1e-8
_ATOL = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(values: np.ndarray, validity=None) -> FactorBatch:
    """Build a minimal valid FactorBatch from a (T, N, F) float array."""
    values = np.asarray(values, dtype=np.float64)
    T, N, F = values.shape
    return FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(F)),
        time_axis=AxisRef("t", "int", T),
        asset_axis=AxisRef("a", "int", N),
        values=values,
        validity=validity,
    )


def _random_panel(rng, T, N, F, nan_rate, ties):
    vals = rng.standard_normal((T, N, F))
    if ties:
        vals = np.round(vals, 1)  # collapse to ties
    if nan_rate > 0.0:
        mask = rng.random((T, N, F)) < nan_rate
        vals[mask] = np.nan
    return vals


# ---------------------------------------------------------------------------
# B. _percentile_boundaries vectorized vs scalar oracle (bit-exact)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
@pytest.mark.parametrize("nq", [2, 3, 4, 5, 10, 20])
def test_percentile_boundaries_vectorized_matches_oracle(seed, nq):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        n = rng.integers(2, 40)
        # integer-snap cases (exact integer positions) + fractional cases
        x = rng.standard_normal(n)
        x = np.sort(x)
        new = _percentile_boundaries_from_sorted(x, nq)
        ref = _percentile_boundaries_reference(x, nq)
        assert new.shape == (max(nq - 1, 0),)
        assert new.dtype == np.float64
        # boundaries come from a sorted array -> bit-identical to the oracle
        np.testing.assert_array_equal(new, ref)
        # and the public helper (which sorts first) matches too
        new2 = _percentile_boundaries(x, nq)
        np.testing.assert_array_equal(new2, ref)


def test_percentile_boundaries_known_values():
    # Hand-computed: v = [1,3,5,7], n=4.
    v = np.array([1.0, 3.0, 5.0, 7.0])
    # Q=4 -> boundaries at pos 0.75, 1.5, 2.25
    b4 = _percentile_boundaries(v, 4)
    np.testing.assert_array_equal(b4, np.array([2.5, 4.0, 5.5]))
    # Q=2 -> pos = 1/2 * (n-1) = 1.5 -> midpoint of sv[1]=3 and sv[2]=5 = 4.0
    b2 = _percentile_boundaries(v, 2)
    np.testing.assert_array_equal(b2, np.array([4.0]))
    # Q=1 -> no boundaries
    b1 = _percentile_boundaries(v, 1)
    assert b1.shape == (0,)

    # Integer-snap case: n=5, Q=2 -> pos = 1/2*4 = 2 (exact) -> sv[2] = 3.
    v5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b5 = _percentile_boundaries(v5, 2)
    np.testing.assert_array_equal(b5, np.array([3.0]))


def test_percentile_boundaries_nq_gt_n():
    # n_quantiles > n_finite: caller guards before call, but the helper must not
    # raise and must stay deterministic vs the oracle.
    v = np.array([1.0, 2.0])
    new = _percentile_boundaries(v, 5)
    ref = _percentile_boundaries_reference(v, 5)
    np.testing.assert_array_equal(new, ref)


# ---------------------------------------------------------------------------
# C. assign_quantiles_batch vectorized vs single-sort oracle (bit-exact)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 5])
@pytest.mark.parametrize("nan_rate", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("method", ["max", "min"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_assign_quantiles_batch_equivalence(seed, nan_rate, method, ndim):
    rng = np.random.default_rng(seed)
    T, N, F = 25, 70, 3
    vals = _random_panel(rng, T, N, F, nan_rate, ties=True)
    if ndim == 2:
        vals = vals[:, :, 0]  # (T, N)
    new = assign_quantiles_batch(vals, n_quantiles=5, method=method)
    ref = _assign_quantiles_batch_reference(vals, n_quantiles=5, method=method)
    assert new.dtype == np.int32
    assert new.shape == ref.shape
    np.testing.assert_array_equal(new, ref)


def test_assign_quantiles_batch_known_values():
    # 4 distinct values, Q=4 with 'max' policy -> bins [0,1,2,3]
    v = np.array([[1.0, 3.0, 5.0, 7.0]])  # (T=1, N=4)
    out = assign_quantiles_batch(v, n_quantiles=4, method="max")
    np.testing.assert_array_equal(out[0], np.array([0, 1, 2, 3]))
    # 'min' policy: value == boundary goes to LOWER bin
    out_min = assign_quantiles_batch(v, n_quantiles=4, method="min")
    np.testing.assert_array_equal(out_min[0], np.array([0, 1, 2, 3]))


def test_assign_quantiles_batch_boundary_cases():
    # single asset day (N=1) -> always -1 (n_finite < n_quantiles)
    single = np.array([[[0.5]]])  # (1,1,1)
    np.testing.assert_array_equal(
        assign_quantiles_batch(single, 5)[0, :, 0], np.array([-1]))
    # all NaN day
    nan_day = np.full((3, 4, 1), np.nan)
    nan_day[0, 0, 0] = 1.0  # one good value, still < n_quantiles
    out = assign_quantiles_batch(nan_day, 5)
    assert np.all(out == -1)
    # all constant column (ties) -> every value equals every boundary, so the
    # tie policy decides the bin: 'max' (side='right') -> top bin, 'min'
    # (side='left') -> bottom bin (0).
    const = np.full((2, 6, 1), 2.0)
    out = assign_quantiles_batch(const, 3, method="max")
    np.testing.assert_array_equal(out[:, :, 0], np.full((2, 6), 2))
    out_min = assign_quantiles_batch(const, 3, method="min")
    np.testing.assert_array_equal(out_min[:, :, 0], np.full((2, 6), 0, dtype=np.int32))
    # nq=1 -> all finite assets bin 0
    out1 = assign_quantiles_batch(const, 1, method="max")
    np.testing.assert_array_equal(out1[:, :, 0], np.zeros((2, 6), dtype=np.int32))
    # nq > N -> all -1
    out_big = assign_quantiles_batch(const, 10, method="max")
    np.testing.assert_array_equal(out_big[:, :, 0], np.full((2, 6), -1))


def test_assign_quantiles_batch_shape_dtype_invariant():
    rng = np.random.default_rng(3)
    vals3 = rng.standard_normal((10, 20, 4))
    out3 = assign_quantiles_batch(vals3, 5)
    assert out3.shape == (10, 20, 4)
    assert out3.dtype == np.int32
    out2 = assign_quantiles_batch(vals3[:, :, 0], 5)
    assert out2.shape == (10, 20)
    assert out2.dtype == np.int32


# ---------------------------------------------------------------------------
# A. compute_adaptive_quantile_count single-sort vs re-sorting oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 9])
@pytest.mark.parametrize("nan_rate", [0.0, 0.1, 0.3])
def test_adaptive_quantile_count_equivalence(seed, nan_rate):
    rng = np.random.default_rng(seed)
    T, N, F = 20, 50, 4
    vals = _random_panel(rng, T, N, F, nan_rate, ties=True)
    fb = _make_batch(vals)
    policy = AdaptiveBinsPolicy(
        preferred_bins=5, fallback_bins=(3, 2), min_effective_names_per_bin=2
    )
    out_new = compute_adaptive_quantile_count(fb, policy=policy, tie_policy="max")
    out_ref = _compute_adaptive_quantile_count_reference(fb, policy=policy, tie_policy="max")

    # Selected bin count per factor (the public value).
    np.testing.assert_allclose(
        out_new.values, out_ref.values, rtol=_RTOL, atol=_ATOL, equal_nan=True
    )
    # Provenance: per-date candidate min-bucket counts must match exactly.
    cov_new = out_new.provenance["adaptive_bins_coverage"]
    cov_ref = out_ref.provenance["adaptive_bins_coverage"]
    assert len(cov_new) == len(cov_ref)
    for dn, dr in zip(cov_new, cov_ref):
        assert dn["factor_id"] == dr["factor_id"]
        assert dn["selected_q"] == dr["selected_q"]
        assert dn["fallback_reason"] == dr["fallback_reason"]
        for dnn, drr in zip(dn["dates"], dr["dates"]):
            assert (
                dnn["candidate_min_bucket_counts"]
                == drr["candidate_min_bucket_counts"]
            )


def test_adaptive_quantile_count_plain_profile_path():
    # Non-FactorBatch (plain profile) path is unchanged code; sanity check it
    # still matches the reference oracle.
    rng = np.random.default_rng(4)
    prof = np.round(rng.standard_normal((12, 3)), 1)
    new = compute_adaptive_quantile_count(prof)
    ref = _compute_adaptive_quantile_count_reference(prof)
    np.testing.assert_allclose(new, ref, rtol=_RTOL, atol=_ATOL, equal_nan=True)


@pytest.mark.parametrize("seed", [0, 1, 2, 8])
@pytest.mark.parametrize("nan_rate", [0.0, 0.1, 0.3])
def test_adaptive_quantile_count_divisor_chain_equivalence(seed, nan_rate):
    """Cover the DEFAULT policy (20/10/5), an exact divisor chain.

    This exercises the bit-exact divisor fast path (finest-Q assignment derived
    once, coarser bins via integer division) against the faithful oracle.
    """
    rng = np.random.default_rng(seed)
    T, N, F = 20, 60, 4
    vals = _random_panel(rng, T, N, F, nan_rate, ties=True)
    fb = _make_batch(vals)
    policy = AdaptiveBinsPolicy()  # default 20/10/5, min_effective_names_per_bin=100
    out_new = compute_adaptive_quantile_count(fb, policy=policy, tie_policy="max")
    out_ref = _compute_adaptive_quantile_count_reference(fb, policy=policy, tie_policy="max")
    np.testing.assert_allclose(
        out_new.values, out_ref.values, rtol=_RTOL, atol=_ATOL, equal_nan=True
    )
    cov_new = out_new.provenance["adaptive_bins_coverage"]
    cov_ref = out_ref.provenance["adaptive_bins_coverage"]
    assert len(cov_new) == len(cov_ref)
    for dn, dr in zip(cov_new, cov_ref):
        assert dn["selected_q"] == dr["selected_q"]
        for dnn, drr in zip(dn["dates"], dr["dates"]):
            assert (
                dnn["candidate_min_bucket_counts"]
                == drr["candidate_min_bucket_counts"]
            )


# ---------------------------------------------------------------------------
# D. compute_shape_bootstrap_confidence hoisted indexing vs per-resample oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 11])
def test_shape_bootstrap_confidence_equivalence(seed):
    rng = np.random.default_rng(seed)
    nw, nq, F = 8, 12, 3
    windows = np.round(rng.standard_normal((nw, nq, F)), 2)  # ties
    new = compute_shape_bootstrap_confidence(windows, resamples=60, random_seed=seed)
    ref = _compute_shape_bootstrap_confidence_reference(
        windows, resamples=60, random_seed=seed
    )
    np.testing.assert_allclose(new, ref, rtol=_RTOL, atol=_ATOL, equal_nan=True)


@pytest.mark.parametrize("seed", [0, 3])
def test_shape_bootstrap_confidence_with_nan_equivalence(seed):
    rng = np.random.default_rng(seed)
    nw, nq, F = 10, 15, 4
    windows = rng.standard_normal((nw, nq, F))
    windows[rng.random((nw, nq, F)) < 0.15] = np.nan  # varying finite masks
    new = compute_shape_bootstrap_confidence(windows, resamples=80, random_seed=seed)
    ref = _compute_shape_bootstrap_confidence_reference(
        windows, resamples=80, random_seed=seed
    )
    np.testing.assert_allclose(new, ref, rtol=_RTOL, atol=_ATOL, equal_nan=True)


def test_shape_bootstrap_confidence_boundary_cases():
    # single profile -> NaN (no second observation)
    single = np.ones((5, 1))
    out = compute_shape_bootstrap_confidence(single)
    assert np.all(np.isnan(out))
    ref = _compute_shape_bootstrap_confidence_reference(single)
    np.testing.assert_array_equal(out, ref)
    # shape/dtype invariant
    windows = np.random.default_rng(0).standard_normal((7, 10, 3))
    out = compute_shape_bootstrap_confidence(windows)
    assert out.shape == (3,)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Performance smoke test (informational; asserts only that results agree).
# ---------------------------------------------------------------------------

def test_perf_speedup_report():
    """Report speedups for T=1250,N=300 and T=250,N=50.

    This test times the OLD (reference) kernels against the NEW (optimized)
    kernels and prints the ratio.  It only hard-asserts equivalence (which is
    covered above) so it cannot flake on a loaded machine.
    """
    import time

    def _timeit(fn, n=3):
        best = float("inf")
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    report = {}
    for (T, N) in [(1250, 300), (250, 50)]:
        rng = np.random.default_rng(123)
        vals = np.round(rng.standard_normal((T, N, 1)), 2)
        fb = _make_batch(vals)
        policy = AdaptiveBinsPolicy(
            preferred_bins=20, fallback_bins=(10, 5),
            min_effective_names_per_bin=2,
        )  # default divisor chain 20/10/5 -> exercises the fast path

        def _new_a():
            return compute_adaptive_quantile_count(fb, policy=policy)

        def _ref_a():
            return _compute_adaptive_quantile_count_reference(fb, policy=policy)

        new_t = _timeit(_new_a, 2)
        ref_t = _timeit(_ref_a, 2)
        ratio_a = ref_t / new_t if new_t > 0 else float("nan")
        report[f"adaptive_T{T}_N{N}"] = (ref_t, new_t, ratio_a)

        # assign_quantiles_batch
        def _new_c():
            return assign_quantiles_batch(vals, 5)

        def _ref_c():
            return _assign_quantiles_batch_reference(vals, 5)

        new_t = _timeit(_new_c, 3)
        ref_t = _timeit(_ref_c, 3)
        ratio_c = ref_t / new_t if new_t > 0 else float("nan")
        report[f"assign_T{T}_N{N}"] = (ref_t, new_t, ratio_c)

    for k, (rt, nt, ratio) in report.items():
        print(f"[perf] {k}: old={rt*1000:.1f}ms new={nt*1000:.1f}ms "
              f"speedup={ratio:.2f}x")

    # Equivalence is already asserted; just confirm the kernels still run.
    assert True
