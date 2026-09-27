"""Seeded cross-backend quantile assignment checks for awkward small panels."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.rank import batched_quantile_assignment
from quant_evaluator.metrics.quantile import assign_quantiles


def test_gpu_quantile_assignment_matches_cpu_over_ties_and_sparse_rows():
    rng = np.random.default_rng(20260928)
    for n_quantiles in (2, 3, 5, 10):
        for n_assets in (1, 2, 3, 7, 19, 53, 100):
            for method in ("min", "max"):
                for case in range(8):
                    values = rng.normal(size=(3, n_assets))
                    values = np.round(values, case % 4)
                    values[rng.random(values.shape) < 0.17] = np.nan
                    actual = cp.asnumpy(batched_quantile_assignment(
                        values, n_quantiles=n_quantiles, method=method))
                    expected = assign_quantiles(
                        values, n_quantiles=n_quantiles, method=method)
                    np.testing.assert_array_equal(actual, expected)
