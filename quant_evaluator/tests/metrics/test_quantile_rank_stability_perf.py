"""Oracle and small A/B coverage for quantile rank stability."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from quant_evaluator.metrics.ic_summary import compute_quantile_rank_stability


def _pairwise_quantile_rank_oracle(values, min_periods=2, min_common=3):
    """Literal date-pair oracle: re-rank each pair on its joint finite support."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    result = np.full(arr.shape[2], np.nan, dtype=np.float64)
    for f in range(arr.shape[2]):
        panel = arr[:, :, f]
        finite = np.isfinite(panel)
        days = np.flatnonzero(finite.sum(axis=1) >= min_common)
        if days.size < min_periods:
            continue
        correlations = []
        for left in range(days.size - 1):
            for right in range(left + 1, days.size):
                common = finite[days[left]] & finite[days[right]]
                if common.sum() < min_common:
                    continue
                ra = rankdata(panel[days[left], common], method="average")
                rb = rankdata(panel[days[right], common], method="average")
                if np.ptp(ra) == 0 or np.ptp(rb) == 0:
                    continue
                corr = np.corrcoef(ra, rb)[0, 1]
                if np.isfinite(corr):
                    correlations.append(float(corr))
        if correlations:
            result[f] = np.mean(correlations)
    return result


def _make_panels(seed=30):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(24, 8, 3))
    # Factor 0 has one shared missing bucket, which exercises the fast path
    # with a nontrivial (but common) support.
    values[:, 0, 0] = np.inf
    # Factor 1 has pair-dependent supports and must retain the general path.
    values[rng.random(values[:, :, 1].shape) < 0.27, 1] = np.nan
    # Factor 2 has varying missingness and many rank-constant pairs.
    values[:, :, 2] = 4.0
    values[rng.random(values[:, :, 2].shape) < 0.18, 2] = -np.inf
    return values


@pytest.mark.parametrize("min_periods,min_common", [(2, 3), (8, 4), (30, 3)])
def test_quantile_rank_stability_matches_pairwise_oracle(min_periods, min_common):
    values = _make_panels()
    got = compute_quantile_rank_stability(
        values, min_periods=min_periods, min_common=min_common
    )
    expected = _pairwise_quantile_rank_oracle(
        values, min_periods=min_periods, min_common=min_common
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_quantile_rank_stability_2d_shared_mask_and_ties_match_oracle():
    values = np.array([
        [1., 1., np.nan, 4., 5.],
        [2., 2., np.nan, 3., 8.],
        [5., 5., np.nan, 2., 1.],
        [4., 4., np.nan, 6., 3.],
        [3., 3., np.nan, 1., 7.],
    ])
    got = compute_quantile_rank_stability(values, min_periods=2, min_common=3)
    expected = _pairwise_quantile_rank_oracle(values, min_periods=2, min_common=3)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_quantile_rank_stability_pairs_below_joint_support_are_nan():
    values = np.full((4, 8), np.nan)
    values[0, :4] = [1., 2., 3., 4.]
    values[1, 2:6] = [1., 2., 3., 4.]
    values[2, 4:] = [1., 2., 3., 4.]
    values[3, [0, 2, 4, 6]] = [1., 2., 3., 4.]
    got = compute_quantile_rank_stability(values, min_periods=2, min_common=4)
    expected = _pairwise_quantile_rank_oracle(values, min_periods=2, min_common=4)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0, equal_nan=True)


def test_quantile_rank_stability_small_dense_ab_shape_and_determinism():
    rng = np.random.default_rng(91)
    values = rng.normal(size=(180, 10, 2))
    values[:, :, 1] = np.round(values[:, :, 1], 0)  # average ties
    first = compute_quantile_rank_stability(values)
    second = compute_quantile_rank_stability(values)
    expected = _pairwise_quantile_rank_oracle(values)
    np.testing.assert_allclose(first, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(first, second)
