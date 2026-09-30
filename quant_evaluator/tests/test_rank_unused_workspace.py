import numpy as np
import pytest
from scipy.stats import rankdata


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_rank_distinct_handles_ties_nonfinite_and_multiple_chunks(monkeypatch, dtype):
    cp = pytest.importorskip("cupy")
    from quant_evaluator.kernels.gpu import rank as rank_module

    values = np.array(
        [
            [3, 3, 1, 2, np.inf, -np.inf, np.nan],
            [1, 1, 1, 1, 1, 1, 1],
            [np.nan, np.inf, -np.inf, np.nan, np.inf, -np.inf, np.nan],
            [-2, 0, 0, 4, 5, 5, np.nan],
            [1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=dtype,
    )
    # Force three rank chunks (two rows, two rows, one row) with this small N.
    monkeypatch.setattr(rank_module, "_MAX_CHUNK_BYTES", 2500)

    actual, distinct = rank_module.batched_rank(
        cp.asarray(values), return_distinct=True
    )
    actual = cp.asnumpy(actual)
    distinct = cp.asnumpy(distinct)

    expected = np.full(values.shape, np.nan, dtype=np.float64)
    expected_distinct = np.zeros(values.shape[0], dtype=np.int32)
    for row_index, row in enumerate(values):
        finite = np.isfinite(row)
        expected[row_index, finite] = rankdata(row[finite], method="average")
        expected_distinct[row_index] = np.unique(row[finite]).size

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(distinct, expected_distinct)
