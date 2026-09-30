import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.correlation import batched_pearson_ic


def test_pearson_repair_crosswarp_overflow_flags_and_pairwise_masks():
    n = 1025
    signs = np.where(np.arange(n) % 2, 1.0, -1.0)
    x = np.empty((1, 3, n * 2), dtype=np.float64)
    y = np.empty_like(x)
    x[:, :, ::2] = 0.0
    y[:, :, ::2] = 0.0
    xx, yy = x[:, :, ::2], y[:, :, ::2]
    xx[0, 0], yy[0, 0] = signs * 1.7e308, signs * 2.0
    xx[0, 1], yy[0, 1] = signs * 2.0, signs * 1.7e308
    xx[0, 2], yy[0, 2] = signs * 1.7e308, -signs * 1.7e308
    xx[0, 0, ::37] = np.nan
    yy[0, 0, ::43] = np.inf
    xx[0, 1, ::41] = np.inf
    yy[0, 1, ::47] = np.nan
    xg, yg = cp.asarray(x)[:, :, ::2], cp.asarray(y)[:, :, ::2]
    assert not xg.flags.c_contiguous and not yg.flags.c_contiguous
    actuals, count_results = [], []
    for _ in range(32):
        actual, counts = batched_pearson_ic(xg, yg, min_obs=2)
        actuals.append(actual)
        count_results.append(counts)
    actual = cp.stack(actuals)
    host_counts = cp.asnumpy(cp.stack(count_results))
    assert np.all(host_counts[:, 0, :2] < n)
    for row in range(3):
        valid = np.isfinite(xx[0, row]) & np.isfinite(yy[0, row])
        a, b = xx[0, row, valid], yy[0, row, valid]
        a /= np.max(np.abs(a)); b /= np.max(np.abs(b))
        a -= a.mean(); b -= b.mean()
        expected = np.dot(a, b) / np.sqrt(np.dot(a, a)) / np.sqrt(np.dot(b, b))
        np.testing.assert_allclose(cp.asnumpy(actual)[:, 0, row], expected, atol=1e-12)
