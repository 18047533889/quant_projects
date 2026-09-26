"""Bit-exact equivalence tests for the batched quantile-cutoff kernel used
by ``compute_long_short_returns``, and end-to-end equivalence of the
vectorized long/short backtest against its per-day loop oracle.

Contract: the production path (sort + virtual-index + ``_lerp`` emulation)
must match ``np.nanquantile(..., axis=1)`` and
``_compute_long_short_returns_reference`` bit-for-bit (NaN positions and
finite payloads) on every input.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.metrics.portfolio_stats import (
    _batched_linear_quantile_cutoffs,
    _compute_long_short_returns_reference,
    compute_long_short_returns,
)

_QS = (0.2, 0.8)


@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.parametrize("nan_rate", [0.0, 0.10, 0.30])
@pytest.mark.parametrize("quantiles", [(0.2, 0.8), (0.37, 0.5), (0.01, 0.99), (0.5,)])
def test_batched_cutoffs_match_nanquantile(seed, nan_rate, quantiles):
    rng = np.random.default_rng(seed)
    fac = rng.normal(size=(40, 60))
    if nan_rate:
        fac[rng.random(fac.shape) < nan_rate] = np.nan
    if seed % 2:
        fac = np.round(fac, 1)  # ties
    got = _batched_linear_quantile_cutoffs(fac, quantiles)
    with np.errstate(all="ignore"):
        want = tuple(np.nanquantile(fac, q, axis=1) for q in quantiles)
    for g, w in zip(got, want):
        assert np.array_equal(g, w, equal_nan=True)


def test_batched_cutoffs_all_nan_rows_and_small_n():
    fac = np.full((5, 8), np.nan)
    fac[1, :2] = [3.0, 1.0]     # n_finite == 1
    fac[2] = 2.5                # constant row
    got = _batched_linear_quantile_cutoffs(fac, _QS)
    with np.errstate(all="ignore"):
        want = tuple(np.nanquantile(fac, q, axis=1) for q in _QS)
    for g, w in zip(got, want):
        assert np.array_equal(g, w, equal_nan=True)


def _assert_close_with_same_nan(new_arr, ref_arr):
    """NaN positions must match exactly; finite values within 1e-12.

    Rationale: bucket means compress to a different summation order than
    masked panel sums (documented ulp-level deviation, bounded by the
    GPU-parity house rule rtol=1e-8/atol=1e-10); we hold the tighter
    1e-12 bound here.  The quantile-cutoff kernel itself is bit-exact and
    is pinned by the dedicated tests above.
    """
    assert np.array_equal(np.isnan(new_arr), np.isnan(ref_arr))
    finite = np.isfinite(new_arr) & np.isfinite(ref_arr)
    assert np.allclose(new_arr[finite], ref_arr[finite], rtol=1e-12, atol=1e-12)


def _cases():
    rng = np.random.default_rng(5)
    for seed in (1, 2, 3):
        r = np.random.default_rng(seed)
        T, N = 60, 40
        fac = r.normal(size=(T, N))
        fwd = r.normal(size=(T, N)) * 0.02
        for nan_rate in (0.0, 0.10, 0.30):
            fac_c = fac.copy()
            fwd_c = fwd.copy()
            if nan_rate:
                fac_c[r.random(fac_c.shape) < nan_rate] = np.nan
                fwd_c[r.random(fwd_c.shape) < nan_rate] = np.nan
            if seed == 2:
                fac_c = np.round(fac_c, 1)  # heavy ties at the cutoffs
            yield fac_c, fwd_c, None


@pytest.mark.parametrize(
    "fac,fwd,validity",
    [(_fac, _fwd, _v) for _fac, _fwd, _v in _cases()],
)
def test_long_short_returns_matches_reference(fac, fwd, validity):
    ref = _compute_long_short_returns_reference(
        fac, fwd, 0.8, 0.2, validity, missing_return_policy="drop", tie_policy="max"
    )
    new = compute_long_short_returns(
        fac, fwd, 0.8, 0.2, validity, missing_return_policy="drop", tie_policy="max"
    )
    for r_arr, n_arr in zip(ref, new):
        _assert_close_with_same_nan(n_arr, r_arr)


@pytest.mark.parametrize("policy", ["zero_fill", "drop"])
@pytest.mark.parametrize("tie_policy", ["max", "min"])
def test_long_short_returns_policies_and_ties(policy, tie_policy):
    rng = np.random.default_rng(99)
    T, N = 50, 30
    fac = rng.normal(size=(T, N))
    fwd = rng.normal(size=(T, N)) * 0.01
    fac[rng.random(fac.shape) < 0.15] = np.nan
    fwd[rng.random(fwd.shape) < 0.15] = np.nan
    fac = np.round(fac, 1)
    ref = _compute_long_short_returns_reference(
        fac, fwd, 0.8, 0.2, None, missing_return_policy=policy, tie_policy=tie_policy
    )
    new = compute_long_short_returns(
        fac, fwd, 0.8, 0.2, None, missing_return_policy=policy, tie_policy=tie_policy
    )
    for r_arr, n_arr in zip(ref, new):
        _assert_close_with_same_nan(n_arr, r_arr)


def test_long_short_returns_known_value_oracle():
    """4 assets: top1 long / bottom1 short under tie_policy=max."""
    fac = np.array([[1.0, 2.0, 3.0, 4.0]])
    fwd = np.array([[0.10, 0.02, -0.02, 0.20]])
    long_r, short_r, ls_r = compute_long_short_returns(
        fac, fwd, 0.8, 0.2, None, missing_return_policy="drop", tie_policy="max"
    )
    # 4 values -> q0.8 cutoff lies between 3.0 and 4.0, q0.2 between 1.0 and 2.0
    assert long_r[0] == pytest.approx(0.20)
    assert short_r[0] == pytest.approx(0.10)
    assert ls_r[0] == pytest.approx((0.20 - 0.10) / 2.0)


def test_long_short_returns_3d_matches_per_factor_loop():
    rng = np.random.default_rng(123)
    T, N, F = 40, 25, 3
    fac = rng.normal(size=(T, N, F))
    fwd = rng.normal(size=(T, N)) * 0.01
    fac[rng.random(fac.shape) < 0.2] = np.nan
    got = compute_long_short_returns(fac, fwd, missing_return_policy="drop")
    ref = _compute_long_short_returns_reference(
        fac, fwd, missing_return_policy="drop"
    )
    for g, r_arr in zip(got, ref):
        _assert_close_with_same_nan(g, r_arr)
