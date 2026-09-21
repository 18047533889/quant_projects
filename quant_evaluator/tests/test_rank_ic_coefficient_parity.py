import numpy as np
import pytest
from scipy import stats

from quant_evaluator.metrics.ic import _spearman_rank_correlation


@pytest.mark.parametrize("size", [2, 3, 7, 64, 256])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_rank_ic_is_exact_scipy_coefficient_for_ties_and_distinct_values(size, dtype):
    rng = np.random.default_rng(20260922)
    for ties in (False, True):
        for _ in range(10):
            x = (rng.integers(-3, 4, size) if ties else rng.normal(size=size)).astype(dtype)
            y = (rng.integers(-3, 4, size) if ties else rng.normal(size=size)).astype(dtype)
            if np.unique(x).size < 2 or np.unique(y).size < 2:
                assert np.isnan(_spearman_rank_correlation(x, y, min_obs=2))
            else:
                assert _spearman_rank_correlation(x, y, min_obs=2) == stats.spearmanr(x, y).statistic


def test_rank_ic_pairwise_extremes_and_minimum_count_are_preserved():
    x = np.array([1e308, -1e308, 0., 0., np.nan, 3., np.inf])
    y = np.array([3., 0., 1., 1., 99., -np.inf, 5.])
    assert _spearman_rank_correlation(x, y, min_obs=4) == stats.spearmanr(x[:4], y[:4]).statistic
    assert np.isnan(_spearman_rank_correlation(x, y, min_obs=5))
    assert np.isnan(_spearman_rank_correlation(np.array([]), np.array([]), min_obs=2))
