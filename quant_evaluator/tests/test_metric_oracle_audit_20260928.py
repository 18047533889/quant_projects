"""Independent seeded oracle audit for GPU rank stability."""

import numpy as np
import pytest
from scipy.stats import rankdata

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.stability import batched_rank_stability
from quant_evaluator.metrics.temporal import compute_rank_stability


def _spearman_oracle(values, lag, min_obs):
    """Pairwise-finite Spearman, independently ranked with SciPy."""
    values = np.asarray(values, dtype=np.float64)
    t_count, _, factor_count = values.shape
    result = np.full((t_count - lag, factor_count), np.nan)
    for t in range(t_count - lag):
        for f in range(factor_count):
            left = values[t, :, f]
            right = values[t + lag, :, f]
            valid = np.isfinite(left) & np.isfinite(right)
            if valid.sum() < min_obs:
                continue
            left_rank = rankdata(left[valid], method="average")
            right_rank = rankdata(right[valid], method="average")
            if np.ptp(left_rank) == 0 or np.ptp(right_rank) == 0:
                continue
            result[t, f] = np.corrcoef(left_rank, right_rank)[0, 1]
    return result


def _assert_matches_oracles(values, lag, min_obs=10):
    gpu_values = cp.asnumpy(
        batched_rank_stability(np.transpose(values, (0, 2, 1)), lag=lag, min_obs=min_obs)
    )
    scipy_oracle = _spearman_oracle(values, lag, min_obs)
    cpu_values = compute_rank_stability(values, lag=lag, method="spearman")
    np.testing.assert_array_equal(np.isnan(gpu_values), np.isnan(scipy_oracle))
    np.testing.assert_allclose(gpu_values, scipy_oracle, rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(cpu_values), np.isnan(scipy_oracle))
    np.testing.assert_allclose(cpu_values, scipy_oracle, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_gpu_rank_stability_matches_scipy_on_sparse_tie_heavy_seeded_panels():
    rng = np.random.default_rng(20260928)
    for case in range(16):
        values = rng.normal(size=(7, 73, 4))
        values = np.round(values, decimals=case % 4)
        values[rng.random(values.shape) < (0.08 + 0.01 * (case % 5))] = np.nan
        values[rng.random(values.shape) < 0.025] = np.inf
        if case % 3 == 0:
            # One day/factor is constant only within the pairwise-finite set.
            values[2, :31, 1] = 5.0
            values[3, 31:, 1] = np.nan
        if case % 4 == 0:
            # Include very large offsets without changing the rank ordering.
            values[:, :, 2] += 1e12
        for lag in (1, 2, 3):
            _assert_matches_oracles(values, lag=lag)


def test_gpu_rank_stability_min_obs_boundary_and_pairwise_finite_mask():
    values = np.full((2, 13, 3), np.nan)
    for factor, count in enumerate((9, 10, 11)):
        left = np.linspace(-2.0, 2.0, count)
        right = np.roll(left, 2)
        # Ties exercise average-rank handling at the threshold boundary.
        left[::3] = 0.0
        right[1::4] = 1.0
        values[0, :count, factor] = left
        values[1, :count, factor] = right
    # Non-finite values must be excluded pairwise, including infinities.
    values[0, 0, 2] = np.inf
    values[1, 1, 2] = -np.inf

    _assert_matches_oracles(values, lag=1, min_obs=10)
    actual = cp.asnumpy(
        batched_rank_stability(np.transpose(values, (0, 2, 1)), lag=1, min_obs=10)
    )
    assert np.isnan(actual[0, 0])  # nine valid pairs: below threshold
    assert np.isfinite(actual[0, 1])  # ten valid pairs: threshold included
    assert np.isnan(actual[0, 2])  # eleven initial, reduced to nine by infinities


@pytest.mark.parametrize("lag", [0, -1, True, np.bool_(True), 1.5])
def test_gpu_rank_stability_rejects_invalid_lag_like_cpu(lag):
    values = np.ones((3, 12, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="lag"):
        compute_rank_stability(values, lag=lag, method="spearman")
    with pytest.raises(ValueError, match="lag"):
        batched_rank_stability(np.transpose(values, (0, 2, 1)), lag=lag)
