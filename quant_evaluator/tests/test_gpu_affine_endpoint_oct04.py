"""GPU exact affine Pearson endpoints under changing pairwise masks."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.correlation import batched_pearson_ic
from quant_evaluator.kernels.gpu.pearson_endpoint import certify_gpu_pearson_endpoints


def test_gpu_affine_pearson_rows_are_exact_endpoints_with_pairwise_missingness(monkeypatch):
    rng = np.random.default_rng(19)
    labels = np.broadcast_to(np.arange(100, dtype=np.float64), (65, 100)).copy()
    validity = rng.random(labels.shape) > 0.2
    factors = np.stack((3.0 * labels + 7.0, 7.0 - 3.0 * labels), axis=1)
    host_shapes = []
    asnumpy = cp.asnumpy

    def capture_transfer(array, *args, **kwargs):
        host_shapes.append(tuple(array.shape))
        return asnumpy(array, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", capture_transfer)
    ic, counts = batched_pearson_ic(factors, labels, min_obs=20,
                                    factor_validity=np.broadcast_to(validity[:, None, :], factors.shape))
    result = asnumpy(ic)
    count_result = asnumpy(counts)
    np.testing.assert_array_equal(result, np.broadcast_to([1.0, -1.0], (65, 2)))
    np.testing.assert_array_equal(count_result, validity.sum(axis=1)[:, None].repeat(2, axis=1))
    vector_transfers = [shape for shape in host_shapes if len(shape) == 2 and shape[1] == 100]
    assert vector_transfers
    assert max(shape[0] for shape in vector_transfers) <= 64
    assert all(len(shape) < 3 for shape in host_shapes)


def test_gpu_endpoint_certifier_preserves_one_ulp_non_affine_row():
    x = cp.asarray(np.arange(100, dtype=np.float64)[None, None, :])
    y_host = 3.0 * np.arange(100, dtype=np.float64) + 7.0
    y_host[-1] = np.nextafter(y_host[-1], np.inf)
    y = cp.asarray(y_host[None, None, :])
    finite = cp.ones_like(x, dtype=cp.bool_)
    rounded = cp.asarray([[np.nextafter(1.0, 0.0)]])
    out = certify_gpu_pearson_endpoints(rounded, x, y, finite)
    assert cp.asnumpy(out)[0, 0] == np.nextafter(1.0, 0.0)
