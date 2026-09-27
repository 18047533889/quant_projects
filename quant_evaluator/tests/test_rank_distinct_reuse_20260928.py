"""Parity coverage for distinct counts reused from the GPU rank sort."""

import numpy as np
import pytest
from scipy.stats import rankdata

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu import rank as rank_kernel
from quant_evaluator.kernels.gpu.correlation import batched_spearman_ic
from quant_evaluator.kernels.gpu.rank import (
    batched_distinct_level_count,
    batched_rank,
)


def test_rank_distinct_counts_match_independent_count_with_ties_and_nonfinite():
    values = np.array([
        [1.0, 1.0, 2.0, np.nan, np.inf, -np.inf],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [np.nan, np.inf, -np.inf, np.nan, np.inf, -np.inf],
        [-2.0, 0.0, -2.0, 3.0, 0.0, 3.0],
    ], dtype=np.float64)
    dev = cp.asarray(values)

    ranks, distinct = batched_rank(dev, return_distinct=True)
    legacy_ranks = batched_rank(dev)
    independent = batched_distinct_level_count(dev).reshape(values.shape[:-1])

    cp.testing.assert_array_equal(ranks, legacy_ranks)
    cp.testing.assert_array_equal(distinct, independent)
    cp.testing.assert_array_equal(distinct, cp.asarray([2, 1, 0, 3], dtype=cp.int32))


def test_rank_distinct_counts_preserve_non_last_axis_shape():
    rng = np.random.default_rng(929)
    values = np.round(rng.normal(size=(3, 7, 4)), 0)
    values[0, 1, 2] = np.nan
    values[2, 5, 1] = np.inf
    dev = cp.asarray(values)

    ranks, distinct = batched_rank(dev, axis_n=1, return_distinct=True)
    expected_ranks = batched_rank(dev, axis_n=1)
    moved = cp.moveaxis(dev, 1, -1)
    expected_distinct = batched_distinct_level_count(moved).reshape(moved.shape[:-1])

    assert ranks.shape == values.shape
    assert distinct.shape == (values.shape[0], values.shape[2])
    cp.testing.assert_array_equal(ranks, expected_ranks)
    cp.testing.assert_array_equal(distinct, expected_distinct)


def test_rank_distinct_counts_are_collected_across_bounded_chunks(monkeypatch):
    values = np.array([
        [1.0, 1.0, 2.0, np.nan],
        [3.0, 4.0, 3.0, 4.0],
        [np.inf, -np.inf, np.nan, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [0.0, 1.0, 2.0, 3.0],
    ], dtype=np.float64).reshape(5, 1, 4)
    # One cross-section per rank chunk, exercising both bounded outputs.
    monkeypatch.setattr(rank_kernel, "_MAX_CHUNK_BYTES", 4 * (8 + 8 + 8 + 8 + 16) * 4)
    dev = cp.asarray(values)

    ranks, distinct = batched_rank(dev, return_distinct=True)
    expected_ranks = batched_rank(dev)
    expected_distinct = batched_distinct_level_count(dev).reshape(values.shape[:-1])

    cp.testing.assert_array_equal(ranks, expected_ranks)
    cp.testing.assert_array_equal(distinct, expected_distinct)


def test_spearman_reuse_matches_pairwise_average_rank_oracle():
    rng = np.random.default_rng(20260928)
    T, F, N = 7, 3, 24
    x = np.round(rng.normal(size=(T, F, N)), 0)
    y = np.round(rng.normal(size=(T, N)), 0)
    x[rng.random(x.shape) < 0.12] = np.nan
    y[rng.random(y.shape) < 0.09] = np.nan
    x[0, 0, 0] = np.inf
    y[1, 1] = -np.inf

    got, counts = batched_spearman_ic(cp.asarray(x), cp.asarray(y), min_obs=5)
    got = cp.asnumpy(got)
    counts = cp.asnumpy(counts)
    expected = np.full((T, F), np.nan, dtype=np.float64)
    expected_counts = np.zeros((T, F), dtype=np.int32)
    for t in range(T):
        for f in range(F):
            finite = np.isfinite(x[t, f]) & np.isfinite(y[t])
            expected_counts[t, f] = finite.sum()
            if finite.sum() < 5:
                continue
            xr = rankdata(x[t, f, finite], method="average")
            yr = rankdata(y[t, finite], method="average")
            if np.unique(xr).size < 2 or np.unique(yr).size < 2:
                continue
            expected[t, f] = np.corrcoef(xr, yr)[0, 1]

    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
