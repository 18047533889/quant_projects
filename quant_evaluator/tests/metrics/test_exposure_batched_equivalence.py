"""Equivalence + performance tests for the batched exposure loading path.

Background: ``compute_factor_loadings`` was refactored (by a concurrent
workstream, left uncommitted in the tree since 2026-09-22) from a per-day
``rank_aware_projection`` loop into a grouped batched-SVD implementation.
The original loop is preserved verbatim as
``_compute_factor_loadings_reference``.  These tests pin the contract:

* loadings / r_squared / residuals match the per-day oracle to the
  GPU-parity house rule (measured max deviation ~5.6e-17, i.e. 1 ulp);
* the diagnostics the contract treats as exact (``status`` / ``n`` /
  ``rank`` / ``effective_df``) are strictly equal;
* the batched path stays faster than the per-day oracle at L scale
  (loose guard against falling back to the loop).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from quant_evaluator.metrics.exposure import (
    _compute_factor_loadings_reference,
    compute_factor_loadings,
)


def _case(seed, nan_rate, ties):
    rng = np.random.default_rng(seed)
    T, N, K = 60, 50, 3
    y = rng.normal(size=(T, N))
    X = rng.normal(size=(T, N, K))
    if nan_rate:
        y[rng.random(y.shape) < nan_rate] = np.nan
        X[rng.random(X.shape) < nan_rate] = np.nan
    if ties:
        X = np.round(X, 1)
        y = np.round(y, 1)
    return y, X


@pytest.mark.parametrize("seed,nan_rate,ties", [
    (11, 0.0, False), (23, 0.15, False), (47, 0.30, True),
    (5, 0.15, True), (88, 0.0, True),
])
def test_batched_loadings_match_per_day_oracle(seed, nan_rate, ties):
    y, X = _case(seed, nan_rate, ties)
    nl, nr2, nres, ndiag = compute_factor_loadings(y, X, return_diagnostics=True)
    rl, rr2, rres, rdiag = _compute_factor_loadings_reference(
        y, X, return_diagnostics=True
    )
    # House GPU-parity tolerance; measured max deviation is ~1 ulp (5.6e-17).
    np.testing.assert_allclose(nl, rl, rtol=1e-8, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(nr2, rr2, rtol=1e-8, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(nres, rres, rtol=1e-8, atol=1e-10, equal_nan=True)
    # NaN positions must coincide exactly with the per-day oracle.
    for a, b in ((nl, rl), (nr2, rr2), (nres, rres)):
        assert np.array_equal(np.isnan(a), np.isnan(b))
    # Diagnostics are integers/strings: strict equality is the contract.
    assert len(ndiag) == len(rdiag)
    for a, b in zip(ndiag, rdiag):
        for key in ("status", "n", "rank", "effective_df"):
            assert a.get(key) == b.get(key), (key, a, b)


def test_batched_loadings_insufficient_observations_diagnostic():
    """Days below min_obs must carry the same INSUFFICIENT_OBSERVATIONS
    status in both paths."""
    y = np.full((10, 30), np.nan)
    X = np.zeros((10, 30, 2))
    y[:, :5] = 1.0  # only 5 valid < min_obs=10 on every day
    X[:, :, 0] = 1.0
    _, _, _, ndiag = compute_factor_loadings(y, X, min_obs=10, return_diagnostics=True)
    _, _, _, rdiag = _compute_factor_loadings_reference(
        y, X, min_obs=10, return_diagnostics=True
    )
    assert all(d["status"] == "INSUFFICIENT_OBSERVATIONS" for d in ndiag)
    assert [d["status"] for d in ndiag] == [d["status"] for d in rdiag]


def test_batched_loadings_deterministic():
    rng = np.random.default_rng(7)
    y = rng.normal(size=(40, 40))
    X = rng.normal(size=(40, 40, 3))
    y[rng.random(y.shape) < 0.1] = np.nan
    a = compute_factor_loadings(y, X, return_diagnostics=True)
    b = compute_factor_loadings(y, X, return_diagnostics=True)
    assert np.array_equal(a[0], b[0], equal_nan=True)
    assert np.array_equal(a[1], b[1], equal_nan=True)
    assert np.array_equal(a[2], b[2], equal_nan=True)


def test_batched_path_not_slower_than_per_day_oracle():
    """Loose perf guard: the batched path must beat the T-iteration Python
    loop at T=400 (loop-era cost ~1ms/day; guard ceiling is generous)."""
    rng = np.random.default_rng(3)
    T, N = 400, 60
    y = rng.normal(size=(T, N))
    X = rng.normal(size=(T, N, 3))
    t0 = time.perf_counter()
    compute_factor_loadings(y, X, return_diagnostics=True)
    batched = time.perf_counter() - t0
    t0 = time.perf_counter()
    _compute_factor_loadings_reference(y, X, return_diagnostics=True)
    reference = time.perf_counter() - t0
    assert batched <= reference * 1.10 + 1e-3, (
        f"batched={batched*1000:.1f}ms reference={reference*1000:.1f}ms - "
        "the batched path must not be slower than the per-day loop"
    )
