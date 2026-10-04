"""CUDA parity tests for fused rolling rank-IC mean and IR statistics."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device unavailable")
except Exception as exc:
    pytest.skip(f"CUDA runtime unavailable: {exc}")

from quant_evaluator.kernels.gpu.predictive import rolling_rank_ic_ir, rolling_rank_ic_mean


def _oracle(x, window=60, minp=20):
    mean = np.full(x.shape, np.nan)
    ir = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        for f in range(x.shape[1]):
            v = x[max(0, t + 1 - window):t + 1, f]
            v = v[np.isfinite(v)]
            if v.size >= minp:
                mean[t, f] = np.mean(v)
                sd = np.std(v, ddof=1)
                if sd > 1e-12:
                    ir[t, f] = mean[t, f] / sd
    return mean, ir


@pytest.mark.parametrize("shape", [(100, 6), (80, 1093), (61, 1), (60, 0)])
def test_fused_cuda_rolling_statistics_match_direct_window_oracle(shape):
    T, F = shape
    rng = np.random.default_rng(20261004 + F)
    x = rng.uniform(-0.8, 0.8, size=shape)
    if F:
        x[rng.random(shape) < 0.13] = np.nan
        x[:, 0] = 0.25
        if F > 1:
            x[:20, 1] = 10.0
            x[20:, 1] = 0.25 + np.where(np.arange(T - 20) % 2, 2e-8, -2e-8)
        if F > 2:
            x[:, 2] = 0.25 + np.where(np.arange(T) % 2, 5.04e-13, -5.04e-13)
        if F > 3:
            x[:, 3] = 0.25 + np.where(np.arange(T) % 2, 2.017e-12, -2.017e-12)
    expected_mean, expected_ir = _oracle(x)
    storage = cp.asarray(np.repeat(x, 2, axis=1))
    device = storage[:, ::2]
    got_mean = rolling_rank_ic_mean(device)
    got_ir = rolling_rank_ic_ir(device)
    # Public metrics are averages across valid rolling windows.
    def aggregate(a):
        with np.errstate(invalid="ignore"):
            return np.nanmean(a, axis=0)
    np.testing.assert_allclose(got_mean, aggregate(expected_mean),
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(got_ir, aggregate(expected_ir),
                               rtol=1e-10, atol=1e-12, equal_nan=True)


def test_noncontiguous_transposed_float32_input_is_promoted_and_matches_oracle():
    rng = np.random.default_rng(711)
    storage = rng.uniform(-1, 1, size=(90, 180)).astype(np.float32)
    view = storage.T[:, ::2]
    x = np.asarray(view, dtype=np.float64)
    expected_mean, expected_ir = _oracle(x)
    device_view = cp.asarray(storage).T[:, ::2]
    got_mean = rolling_rank_ic_mean(device_view)
    got_ir = rolling_rank_ic_ir(device_view)
    np.testing.assert_allclose(got_mean, np.nanmean(expected_mean, axis=0),
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(got_ir, np.nanmean(expected_ir, axis=0),
                               rtol=1e-10, atol=1e-12, equal_nan=True)
    assert got_mean.dtype == np.float64
    assert got_ir.dtype == np.float64


@pytest.mark.parametrize("name,value", [
    ("window", 0), ("window", -1), ("window", 1.5), ("window", True),
    ("min_periods", 0), ("min_periods", -1), ("min_periods", 2.5),
    ("min_periods", False),
])
def test_invalid_rolling_parameters_raise_value_error(name, value):
    from quant_evaluator.kernels.gpu.rolling_ic_statistics import rolling_ic_mean_ir

    kwargs = {"window": 60, "min_periods": 20}
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        rolling_ic_mean_ir(cp.zeros((80, 2)), **kwargs)



def test_numpy_integer_parameters_are_accepted_and_min_periods_above_window_is_nan():
    from quant_evaluator.kernels.gpu.rolling_ic_statistics import rolling_ic_mean_ir


    mean, ir = rolling_ic_mean_ir(cp.ones((8, 2)),
                                  window=np.int64(3),
                                  min_periods=np.int64(4))
    assert cp.all(cp.isnan(mean)).item()
    assert cp.all(cp.isnan(ir)).item()
