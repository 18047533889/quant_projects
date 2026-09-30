import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.correlation import batched_pearson_ic


@pytest.mark.parametrize("factor_specific_y", [False, True])
def test_pearson_repair_noncontiguous_mixed_dtype_and_y_layout(factor_specific_y):
    t, f, n = 3, 2, 257
    source = cp.asarray(np.random.default_rng(41).normal(size=(t, f, n * 2)))
    x = source[:, :, ::2]
    y = cp.asarray(np.random.default_rng(42).normal(size=(t, n * 2)).astype(np.float32))[:, ::2]
    assert not x.flags.c_contiguous
    assert not y.flags.c_contiguous
    x[1, 1] = 1e16 + 2 * cp.arange(n, dtype=cp.float64)
    y[1] = cp.arange(n, dtype=cp.float32)
    if factor_specific_y:
        y = cp.broadcast_to(y[:, None, :], (t, f, n)).copy()
    ic, counts = batched_pearson_ic(x, y, min_obs=2)
    assert float(ic[1, 1]) == pytest.approx(1.0, abs=1e-12)
    assert int(counts[1, 1]) == n


@pytest.mark.parametrize("shape", [(0, 2, 64), (3, 0, 64)])
def test_pearson_repair_empty_rows_do_not_launch_zero_grid(shape):
    x = cp.empty(shape, dtype=cp.float64)
    y = cp.empty((shape[0], 64), dtype=cp.float32)
    ic, counts = batched_pearson_ic(x, y, min_obs=2)
    assert ic.shape == shape[:2]
    assert counts.shape == shape[:2]


def test_pearson_float32_keeps_bounded_fast_path(monkeypatch):
    import quant_evaluator.kernels.gpu.correlation as correlation
    def unexpected_repair(*args, **kwargs):
        raise AssertionError("bounded float32 input should bypass repair kernel")
    monkeypatch.setattr(correlation, "repair_unsafe_pearson_rows", unexpected_repair)
    x = cp.arange(32, dtype=cp.float32)[None, None, :]
    y = cp.arange(32, dtype=cp.float32)[None, :]
    ic, counts = correlation.batched_pearson_ic(x, y, min_obs=2)
    assert float(ic[0, 0]) == pytest.approx(1.0)
    assert int(counts[0, 0]) == 32


@pytest.mark.parametrize("bad_axis", ["time", "asset", "factor"])
def test_pearson_rejects_unsafe_singleton_broadcast_shapes(bad_axis):
    x = cp.zeros((2, 3, 64), dtype=cp.float64)
    if bad_axis == "time":
        y = cp.zeros((1, 64), dtype=cp.float64)
    elif bad_axis == "asset":
        y = cp.zeros((2, 63), dtype=cp.float64)
    else:
        y = cp.zeros((2, 2, 64), dtype=cp.float64)
    with pytest.raises(ValueError, match="Pearson"):
        batched_pearson_ic(x, y, min_obs=2)
