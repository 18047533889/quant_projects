"""Independent oracles for vectorized quantile-rank monotonicity."""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import spearmanr


def _valid_profile(column):
    column = np.asarray(column, dtype=np.float64)
    return (
        column.size >= 3
        and np.isfinite(column).all()
        and np.any(column != column[0])
    )


def _scipy_legacy_reference(values):
    """Freeze the prior per-column SciPy definition without calling the SUT."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    if matrix.shape[0] < 3:
        return result
    bucket_indices = np.arange(matrix.shape[0])
    for feature in range(matrix.shape[1]):
        column = matrix[:, feature]
        if _valid_profile(column):
            result[feature] = spearmanr(bucket_indices, column).statistic
    return result


def _average_ranks_fraction(column):
    """Return exact average ranks; equal signed zeros share a rank."""
    values = [float(value) for value in column]
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [Fraction(0) for _ in values]
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = Fraction((start + 1) + stop, 2)
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def _decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _fraction_decimal_oracle(column):
    """Independent average-rank Pearson oracle with exact centered moments."""
    values = np.asarray(column, dtype=np.float64)
    if not _valid_profile(values):
        return np.nan

    n = len(values)
    bucket_ranks = [Fraction(index + 1) for index in range(n)]
    value_ranks = _average_ranks_fraction(values)
    mean_rank = Fraction(n + 1, 2)
    centered_bucket = [rank - mean_rank for rank in bucket_ranks]
    centered_values = [rank - mean_rank for rank in value_ranks]
    covariance = sum(
        (left * right for left, right in zip(centered_bucket, centered_values)),
        Fraction(),
    )
    bucket_variance = sum((value * value for value in centered_bucket), Fraction())
    value_variance = sum((value * value for value in centered_values), Fraction())
    if covariance == 0:
        return 0.0

    with localcontext() as context:
        context.prec = 80
        numerator = _decimal(covariance)
        denominator = (_decimal(bucket_variance) * _decimal(value_variance)).sqrt()
        return float(numerator / denominator)


def _fraction_oracle(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return np.array(
        [_fraction_decimal_oracle(matrix[:, feature])
         for feature in range(matrix.shape[1])],
        dtype=np.float64,
    )


def _assert_both_references(actual, matrix, *, exact_scipy=False):
    scipy_expected = _scipy_legacy_reference(matrix)
    exact_expected = _fraction_oracle(matrix)
    if exact_scipy:
        finite = np.isfinite(actual) & np.isfinite(scipy_expected)
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(scipy_expected))
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(scipy_expected))
        np.testing.assert_array_equal(
            actual[finite].view(np.uint64), scipy_expected[finite].view(np.uint64),
        )
    else:
        np.testing.assert_allclose(
            actual, scipy_expected, rtol=0.0, atol=1e-14, equal_nan=True,
        )
    np.testing.assert_allclose(
        actual, exact_expected, rtol=0.0, atol=1e-14, equal_nan=True,
    )


@pytest.mark.parametrize("n_quantiles", [0, 1, 2, 3, 4, 5, 6])
def test_quantile_count_boundaries_and_odd_even_profiles(n_quantiles):
    from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity

    indices = np.arange(n_quantiles, dtype=np.float64)
    matrix = np.column_stack((
        indices,
        -indices,
        np.zeros(n_quantiles, dtype=np.float64),
        indices // 2,
    )) if n_quantiles else np.empty((0, 4), dtype=np.float64)
    original = matrix.copy()
    actual = compute_quantile_rank_monotonicity(matrix)

    assert actual.shape == (4,)
    assert actual.dtype == np.float64
    if n_quantiles < 3:
        assert np.isnan(actual).all()
    else:
        _assert_both_references(actual, matrix)
        assert actual[0] == pytest.approx(1.0, rel=0.0, abs=1e-14)
        assert actual[1] == pytest.approx(-1.0, rel=0.0, abs=1e-14)
        assert np.isnan(actual[2])
    np.testing.assert_array_equal(matrix, original)


def test_empty_feature_axis_and_empty_quantile_axis():
    from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity

    empty_features = np.empty((5, 0), dtype=np.float64)
    actual_empty = compute_quantile_rank_monotonicity(empty_features)
    assert actual_empty.shape == (0,)
    assert actual_empty.dtype == np.float64

    empty_quantiles = np.empty((0, 3), dtype=np.float64)
    actual_empty_q = compute_quantile_rank_monotonicity(empty_quantiles)
    np.testing.assert_array_equal(actual_empty_q, np.full(3, np.nan))


def test_ties_signed_zero_subnormal_and_extreme_finite_values_match_oracles():
    from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity

    unit = float.fromhex("0x0.0000000000001p-1022")
    maximum = np.finfo(np.float64).max
    matrix = np.array([
        [-4.0,  0.0,  0.0, -maximum, -2 * unit],
        [-4.0, -0.0, -0.0, -1e300,  -unit],
        [-1.0,  0.0,  0.0,  0.0,    -0.0],
        [-1.0, -0.0, -0.0,  1e300,   unit],
        [ 2.0,  1.0,  0.0,  maximum, 2 * unit],
        [ 2.0,  1.0,  0.0,  maximum, 3 * unit],
        [ 8.0,  2.0,  0.0,  maximum, 4 * unit],
    ], dtype=np.float64)
    original = matrix.copy()
    actual = compute_quantile_rank_monotonicity(matrix)

    _assert_both_references(actual, matrix)
    assert np.isnan(actual[2])
    np.testing.assert_array_equal(matrix, original)


def test_any_nan_or_infinity_rejects_the_entire_column():
    from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity

    matrix = np.column_stack((
        np.arange(7, dtype=np.float64),
        np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0]),
        np.array([0.0, 1.0, np.inf, 3.0, 4.0, 5.0, 6.0]),
        np.array([0.0, 1.0, -np.inf, 3.0, 4.0, 5.0, 6.0]),
        np.full(7, np.nan),
        np.full(7, np.inf),
    ))
    original = matrix.copy()
    actual = compute_quantile_rank_monotonicity(matrix)

    _assert_both_references(actual, matrix)
    assert actual[0] == 1.0
    assert np.isnan(actual[1:]).all()
    np.testing.assert_array_equal(matrix, original)


def test_normal_profiles_preserve_legacy_scipy_bits_and_exact_rank_oracle():
    from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity

    rng = np.random.default_rng(20261004)
    matrix = rng.normal(size=(9, 64))
    original = matrix.copy()
    actual = compute_quantile_rank_monotonicity(matrix)

    _assert_both_references(actual, matrix, exact_scipy=True)
    np.testing.assert_array_equal(matrix, original)


def test_transposed_and_strided_large_feature_inputs_match_distinct_oracles():
    from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity

    profiles = np.array([
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        [ 3.0,  2.0,  1.0, 0.0,-1.0,-2.0,-3.0],
        [ 0.0,  0.0,  1.0, 1.0, 2.0, 2.0, 3.0],
        [ 2.0,  1.0,  2.0, 1.0, 0.0, 0.0,-1.0],
        [-2.0, -2.0, -2.0, 0.0, 0.0, 3.0, 3.0],
        [-0.0,  0.0,  0.0, 1.0, 1.0, 2.0, 2.0],
        [ 1.0, -1.0,  1.0,-1.0, 1.0,-1.0, 1.0],
        [ 4.0,  3.0,  2.0, 1.0, 0.0,-1.0,-2.0],
    ], dtype=np.float64)
    feature_count = 2049
    repeats, remainder = divmod(feature_count, len(profiles))
    transposed = np.tile(profiles, (repeats + 1, 1)).T[:, :feature_count]
    assert transposed.shape == (7, feature_count)
    assert not transposed.flags.c_contiguous
    expected_patterns = np.array(
        [_fraction_decimal_oracle(profile) for profile in profiles],
        dtype=np.float64,
    )
    expected = np.concatenate((
        np.tile(expected_patterns, repeats), expected_patterns[:remainder],
    ))
    original = transposed.copy()

    actual = compute_quantile_rank_monotonicity(transposed)
    np.testing.assert_allclose(
        actual, expected, rtol=0.0, atol=1e-14, equal_nan=True,
    )
    np.testing.assert_array_equal(transposed, original)

    backing = np.empty((7, feature_count * 2), dtype=np.float64)
    backing[:, ::2] = transposed
    backing[:, 1::2] = -999.0
    strided = backing[:, ::2]
    assert strided.shape == (7, feature_count)
    assert not strided.flags.c_contiguous
    strided_original = strided.copy()
    strided_actual = compute_quantile_rank_monotonicity(strided)
    np.testing.assert_array_equal(strided_actual, actual)
    np.testing.assert_array_equal(strided, strided_original)


def test_larger_quantile_axis_matches_oracles_for_distinct_tie_patterns(monkeypatch):
    from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity

    n_quantiles = 257
    index = np.arange(n_quantiles, dtype=np.float64)
    profiles = np.column_stack((
        index,
        -index,
        index // 4,
        (index * 37) % n_quantiles,
        np.ones(n_quantiles),
    ))
    expected_patterns = np.array(
        [_fraction_decimal_oracle(profiles[:, feature])
         for feature in range(profiles.shape[1])],
        dtype=np.float64,
    )
    scipy_patterns = _scipy_legacy_reference(profiles)
    np.testing.assert_allclose(
        expected_patterns, scipy_patterns, rtol=0.0, atol=1e-14, equal_nan=True,
    )

    feature_count = 2049
    repeats, remainder = divmod(feature_count, profiles.shape[1])
    matrix = np.tile(profiles.T, (repeats + 1, 1))[:feature_count, :].T
    assert matrix.shape == (n_quantiles, feature_count)
    assert not matrix.flags.c_contiguous
    original = matrix.copy()

    import scipy.stats
    original_rankdata = scipy.stats.rankdata
    rankdata_shapes = []

    def recording_rankdata(values, *args, **kwargs):
        rankdata_shapes.append(tuple(values.shape))
        return original_rankdata(values, *args, **kwargs)

    monkeypatch.setattr(scipy.stats, "rankdata", recording_rankdata)
    actual = compute_quantile_rank_monotonicity(matrix)
    expected = np.concatenate((
        np.tile(expected_patterns, repeats), expected_patterns[:remainder],
    ))
    np.testing.assert_allclose(
        actual, expected, rtol=0.0, atol=1e-14, equal_nan=True,
    )
    assert rankdata_shapes
    assert all(rows <= 512 for rows, _columns in rankdata_shapes)
    assert all(rows * columns <= 262144 for rows, columns in rankdata_shapes)
    np.testing.assert_array_equal(matrix, original)


def test_quantile_axis_above_exact_rank_bound_falls_back_to_scipy():
    from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity

    matrix = np.arange(65537, dtype=np.float64)[:, None]
    original = matrix.copy()
    actual = compute_quantile_rank_monotonicity(matrix)
    expected = _scipy_legacy_reference(matrix)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(matrix, original)
