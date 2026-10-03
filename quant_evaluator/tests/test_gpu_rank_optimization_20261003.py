"""Independent edge and shape parity for GPU average-rank allocation changes."""

import numpy as np
import pytest
from scipy.stats import rankdata

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu import rank as rank_module


def _oracle(values, axis=-1, pct=False):
    moved = np.moveaxis(values, axis, -1)
    expected = np.full(moved.shape, np.nan, dtype=np.float64)
    distinct = np.zeros(moved.shape[:-1], dtype=np.int32)
    for row_idx in np.ndindex(moved.shape[:-1]):
        row = moved[row_idx]
        valid = np.isfinite(row)
        distinct[row_idx] = np.unique(row[valid]).size
        if valid.any():
            expected[row_idx][valid] = rankdata(row[valid], method="average")
            if pct:
                expected[row_idx][valid] /= valid.sum()
    return np.moveaxis(expected, -1, axis), distinct


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pct", [False, True])
def test_rank_invalid_ties_distinct_pct_and_scatter(dtype, pct):
    values = np.array(
        [
            [3, 3, 1, 2, np.nan, np.inf, -np.inf, 3],
            [np.nan, np.inf, -np.inf, np.nan, np.inf, -np.inf, np.nan, np.inf],
            [5, 5, 5, 5, 5, 5, 5, 5],
            [-0.0, +0.0, 1, 1, 3, 2, 2, np.nan],
        ],
        dtype=dtype,
    )
    expected, expected_distinct = _oracle(values, pct=pct)
    device = cp.asarray(values)
    actual, distinct = rank_module.batched_rank(
        device, pct=pct, return_distinct=True
    )
    ranks_only = rank_module.batched_rank(device, pct=pct)

    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=0, atol=0, equal_nan=True)
    cp.testing.assert_array_equal(actual, ranks_only)
    cp.testing.assert_array_equal(distinct, cp.asarray(expected_distinct))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rank_nonlast_axis_and_forced_chunks_keep_nan_positions(dtype, monkeypatch):
    values = np.array(
        [
            [[3, 1, 1, np.nan], [2, 2, 4, 3], [7, 7, 7, 7],
             [np.nan, np.inf, -np.inf, np.nan], [1, 3, 2, 2]],
            [[5, 4, 3, 2], [np.inf, 1, 1, 2], [8, 9, 8, 9],
             [6, 6, 6, 6], [-np.inf, 0, np.nan, 0]],
        ],
        dtype=dtype,
    )
    axis = 1
    width = values.shape[axis]
    per_row = (np.dtype(dtype).itemsize + 8 + 8 + 8 + 16) * width * 4
    monkeypatch.setattr(rank_module, "_MAX_CHUNK_BYTES", per_row * 2)
    expected, expected_distinct = _oracle(values, axis=axis, pct=True)
    actual, distinct = rank_module.batched_rank(
        cp.asarray(values), axis_n=axis, pct=True, return_distinct=True
    )
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=0, atol=0, equal_nan=True)
    cp.testing.assert_array_equal(distinct, cp.asarray(expected_distinct))


def test_rank_chunk_all_invalid_row_scattered_nan():
    values = cp.asarray([[np.nan, np.inf, -np.inf], [2.0, np.nan, 1.0]])
    actual = rank_module.batched_rank(values)
    assert bool(cp.all(cp.isnan(actual[0])).item())
    cp.testing.assert_array_equal(actual[1], cp.asarray([2.0, np.nan, 1.0]))
