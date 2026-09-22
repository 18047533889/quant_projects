"""Bit-exact oracle coverage for the bounded block-bootstrap kernel."""

import numpy as np
import pytest

from quant_evaluator.metrics.robustness import compute_block_bootstrap_ci


def _legacy_ci(values, block_length, num_bootstrap, confidence_level, random_seed):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    time_count, factor_count = values.shape
    rng = np.random.default_rng(random_seed)
    block_count = (time_count + block_length - 1) // block_length
    starts = rng.integers(
        0, time_count - block_length + 1, size=(num_bootstrap, block_count)
    )
    alpha = 1.0 - confidence_level
    lower = np.full(factor_count, np.nan)
    upper = np.full(factor_count, np.nan)
    for factor in range(factor_count):
        series = values[:, factor]
        if not np.isfinite(series).all() or len(series) < block_length * 2:
            continue
        means = np.full(num_bootstrap, np.nan)
        for replicate in range(num_bootstrap):
            sample = []
            for start in starts[replicate]:
                sample.extend(series[start : start + block_length])
            means[replicate] = np.mean(np.array(sample[: len(series)]))
        lower[factor] = np.percentile(means, 100 * alpha / 2)
        upper[factor] = np.percentile(means, 100 * (1 - alpha / 2))
    return lower, upper


def _assert_exact(actual, expected):
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    ("time_count", "block_length", "num_bootstrap", "seed"),
    [
        (20, 10, 31, 0),
        (37, 8, 257, 7),
        (500, 13, 1000, 19),
        (503, 19, 1103, 31),
        (1003, 17, 101, 91),
        (2003, 19, 1001, 113),
    ],
)
def test_bounded_kernel_is_bit_exact_to_legacy(
    time_count, block_length, num_bootstrap, seed
):
    rng = np.random.default_rng(seed + 1)
    values = rng.normal(size=(time_count, 4))
    values[:, 1] = values[:, 0]
    values[time_count // 3, 3] = np.nan
    kwargs = dict(
        block_length=block_length,
        num_bootstrap=num_bootstrap,
        confidence_level=0.91,
        random_seed=seed,
    )
    _assert_exact(
        compute_block_bootstrap_ci(values, **kwargs),
        _legacy_ci(values, **kwargs),
    )


def test_cancellation_order_and_factor_permutation_are_exact():
    base = np.resize(np.array([1e16, 1.0, -1e16, 3.0, -7.0, 7.0]), 503)
    values = np.column_stack((base, -base, base[::-1], np.ones_like(base)))
    kwargs = dict(
        block_length=19,
        num_bootstrap=1103,
        confidence_level=0.95,
        random_seed=314159,
    )
    expected = _legacy_ci(values, **kwargs)
    actual = compute_block_bootstrap_ci(values, **kwargs)
    _assert_exact(actual, expected)

    order = np.array([2, 0, 3, 1])
    permuted = compute_block_bootstrap_ci(values[:, order], **kwargs)
    np.testing.assert_array_equal(permuted[0], actual[0][order])
    np.testing.assert_array_equal(permuted[1], actual[1][order])
