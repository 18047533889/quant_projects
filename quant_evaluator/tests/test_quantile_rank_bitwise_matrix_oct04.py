"""Supported-runtime bit parity across odd/even and tile-limit profiles."""
import numpy as np
import pytest
from scipy.stats import spearmanr

from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity


@pytest.mark.parametrize("quantiles", [3, 4, 5, 9, 20, 63, 257, 512])
@pytest.mark.parametrize("ties", [False, True])
def test_random_profiles_match_scipy_bits_on_supported_runtime(quantiles, ties):
    rng = np.random.default_rng(20261004 + quantiles)
    matrix = rng.normal(size=(quantiles, 48))
    if ties:
        matrix = np.round(matrix, decimals=0)
    matrix[:, 0] = 0.0
    matrix[0, 1] = np.nan
    matrix[-1, 2] = np.inf
    original = matrix.copy()
    actual = compute_quantile_rank_monotonicity(matrix)
    expected = np.full(48, np.nan)
    for feature in range(48):
        profile = matrix[:, feature]
        if np.isfinite(profile).all() and np.any(profile != profile[0]):
            expected[feature] = spearmanr(np.arange(quantiles), profile).statistic
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))
    np.testing.assert_array_equal(matrix, original)


def test_exact_rank_bound_uses_element_limited_tiles_without_mutation(monkeypatch):
    import scipy.stats

    quantiles = 65536
    index = np.arange(quantiles, dtype=np.float64)
    matrix = np.column_stack((index, -index, index // 3, (index * 37) % quantiles, index // 7))
    original = matrix.copy()
    calls = []
    original_rankdata = scipy.stats.rankdata

    def recording_rankdata(values, *args, **kwargs):
        calls.append(values.shape)
        return original_rankdata(values, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(scipy.stats, "rankdata", recording_rankdata)
        actual = compute_quantile_rank_monotonicity(matrix)
    assert calls == [(4, quantiles), (1, quantiles)]
    assert all(rows * columns <= 262144 for rows, columns in calls)
    expected = np.array([
        spearmanr(np.arange(quantiles), matrix[:, feature]).statistic
        for feature in range(matrix.shape[1])
    ])
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(matrix, original)
