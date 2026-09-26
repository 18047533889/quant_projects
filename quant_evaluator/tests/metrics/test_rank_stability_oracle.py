"""Independent numerical and parameter checks for registered rank stability."""

import numpy as np
import pytest
from scipy import stats

from quant_evaluator.metrics.temporal import (
    compute_mean_rank_stability,
    compute_rank_stability,
)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_pairwise_finite_rank_stability_matches_scipy(method):
    x = np.array([
        [1., 2., 2., 4., 5., 6., 7., 8., 9., 10., 11., np.nan],
        [5., 4., 4., 3., 2., 1., 7., 8., 9., 10., np.nan, 12.],
        [2., 4., 4., 6., 8., 10., 14., 16., 18., 20., 22., 24.],
    ])[:, :, None]
    got = compute_rank_stability(x, lag=1, method=method)[:, 0]
    expected = []
    for before, after in zip(x[:-1, :, 0], x[1:, :, 0]):
        joint = np.isfinite(before) & np.isfinite(after)
        expected.append(getattr(stats, method + "r")(before[joint], after[joint]).statistic)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)
    assert compute_mean_rank_stability(x, lag=1, method=method, min_periods=2)[0] == pytest.approx(np.mean(expected))


@pytest.mark.parametrize("bad_lag", [0, -1, True, 1.5])
def test_rank_stability_requires_positive_integer_lag(bad_lag):
    x = np.arange(36., dtype=float).reshape(3, 12, 1)
    with pytest.raises(ValueError, match="lag must be a positive integer"):
        compute_rank_stability(x, lag=bad_lag)


def test_rank_stability_rejects_unknown_method():
    x = np.arange(36., dtype=float).reshape(3, 12, 1)
    with pytest.raises(ValueError, match="method must be"):
        compute_rank_stability(x, method="spearmann")
