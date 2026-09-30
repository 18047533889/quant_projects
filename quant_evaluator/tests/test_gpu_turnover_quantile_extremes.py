"""CPU parity at quantile interpolation boundaries and extreme magnitudes."""

import warnings

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.tradability import batched_factor_turnover_rate
from quant_evaluator.metrics.temporal import compute_factor_turnover_rate


@pytest.mark.parametrize(
    ("row0", "row1", "quantile", "expected_turnover"),
    [
        (
            [-1e308] * 9 + [1e308],
            [-1e308] * 10,
            0.95,
            0.0,
        ),
        (
            [0.0] * 9 + [np.nextafter(0.0, 1.0)],
            [0.0] * 10,
            8.5 / 9.0,
            0.9,
        ),
        (
            [-0.0] * 9 + [0.0],
            [0.0] * 10,
            0.95,
            0.0,
        ),
        (
            list(range(10)),
            list(reversed(range(10))),
            0.75,
            None,
        ),
    ],
    ids=("finite-overflow", "subnormal-halfway", "signed-zero", "ordinary"),
)
def test_gpu_turnover_quantile_matches_numpy_reference(
    row0, row1, quantile, expected_turnover,
):
    values = np.asarray([row0, row1], dtype=np.float64)[:, :, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = compute_factor_turnover_rate(values, quantile=quantile)
    actual = cp.asnumpy(
        batched_factor_turnover_rate(
            cp.asarray(np.transpose(values, (0, 2, 1))),
            quantile=quantile,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)
    if expected_turnover is not None:
        assert expected[0, 0] == pytest.approx(expected_turnover, abs=0.0)
