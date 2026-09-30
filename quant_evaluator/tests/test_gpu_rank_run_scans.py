"""Focused parity checks for GPU average-rank run scans."""

import numpy as np
import pytest
from scipy.stats import rankdata

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu import rank as rank_module


def _old_bincount_rank(values):
    """Small local reference for the former per-group bincount implementation."""
    flat = values.reshape(-1, values.shape[-1])
    mask = ~cp.isfinite(flat)
    safe = cp.where(mask, cp.inf, flat)
    order = cp.argsort(safe, axis=1, kind="stable")
    sorted_values = cp.take_along_axis(safe, order, axis=1)
    finite = sorted_values != cp.inf
    starts = cp.zeros_like(sorted_values, dtype=cp.bool_)
    starts[:, 0] = True
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group_id = cp.cumsum(starts, axis=1, dtype=cp.int32) - 1
    positions = cp.arange(1, flat.shape[1] + 1, dtype=cp.float64)[None, :]
    weights = cp.where(finite, positions, 0.0)
    max_groups = int(group_id.max().item()) + 1
    keys = (cp.arange(flat.shape[0], dtype=cp.int64)[:, None] * max_groups + group_id).ravel()
    bins = flat.shape[0] * max_groups
    sums = cp.bincount(keys, weights=weights.ravel(), minlength=bins)
    counts = cp.bincount(keys, minlength=bins).astype(cp.float64)
    means = sums / cp.maximum(counts, 1.0)
    sorted_ranks = means[keys].reshape(flat.shape)
    ranks = cp.empty_like(sorted_ranks)
    ranks[cp.arange(flat.shape[0])[:, None], order] = sorted_ranks
    return cp.where(mask, cp.nan, ranks).reshape(values.shape)


def _oracle(values, axis=-1, pct=False):
    moved = np.moveaxis(values, axis, -1)
    expected = np.full(moved.shape, np.nan, dtype=np.float64)
    for index in np.ndindex(moved.shape[:-1]):
        row = moved[index]
        finite = np.isfinite(row)
        if finite.any():
            expected[index][finite] = rankdata(row[finite], method="average")
            if pct:
                expected[index][finite] /= finite.sum()
    return np.moveaxis(expected, -1, axis)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rank_run_scans_match_scipy_and_old_bincount(dtype):
    values = np.array([
        [3, 3, 1, 2, np.inf, -np.inf, np.nan],
        [1, 1, 1, 1, 1, 1, 1],
        [np.nan, np.inf, -np.inf, np.nan, np.inf, -np.inf, np.nan],
        [-2, 0, 0, 4, 5, 5, np.nan],
        [-0.0, +0.0, 1, 1, 3, 2, 2],
    ], dtype=dtype)
    device = cp.asarray(values)
    actual = rank_module.batched_rank(device)
    np.testing.assert_allclose(cp.asnumpy(actual), _oracle(values), rtol=0, atol=0, equal_nan=True)
    cp.testing.assert_array_equal(actual, _old_bincount_rank(device))


def test_rank_run_scans_distinct_counts_constant_and_empty_rows():
    values = cp.asarray([
        [4.0, 4.0, 4.0, 4.0],
        [np.nan, np.inf, -np.inf, np.nan],
        [1.0, 1.0, 2.0, np.nan],
    ], dtype=cp.float64)
    ranks, distinct = rank_module.batched_rank(values, return_distinct=True)
    cp.testing.assert_array_equal(ranks[0], cp.asarray([2.5, 2.5, 2.5, 2.5]))
    cp.testing.assert_array_equal(distinct, cp.asarray([1, 0, 2], dtype=cp.int32))
    assert bool(cp.all(cp.isnan(ranks[1])).item())
    empty = cp.empty((2, 0), dtype=cp.float32)
    assert rank_module.batched_rank(empty).shape == (2, 0)


def test_rank_run_scans_pct_nonlast_axis_and_forced_chunks(monkeypatch):
    values = np.array([
        [[3, 1, 1, np.nan], [2, 2, 4, 3], [7, 7, 7, 7]],
        [[5, 4, 3, 2], [np.inf, 1, 1, 2], [8, 9, 8, 9]],
    ], dtype=np.float32)
    n = values.shape[1]
    monkeypatch.setattr(rank_module, "_MAX_CHUNK_BYTES", 4 * (4 + 8 + 8 + 8 + 16) * n)
    device = cp.asarray(values)
    actual = rank_module.batched_rank(device, axis_n=1, pct=True)
    np.testing.assert_allclose(
        cp.asnumpy(actual), _oracle(values, axis=1, pct=True),
        rtol=0, atol=0, equal_nan=True,
    )
    moved = cp.moveaxis(device, 1, -1)
    old = _old_bincount_rank(moved)
    valid_counts = cp.sum(cp.isfinite(moved), axis=-1, keepdims=True)
    old_pct = cp.moveaxis(old / cp.maximum(valid_counts, 1), -1, 1)
    cp.testing.assert_array_equal(actual, old_pct)
