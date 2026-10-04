"""Centered pairwise autocorrelation regression tests on CUDA."""
import numpy as np
import pytest
from scipy.stats import pearsonr

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device unavailable")
except Exception as exc:
    pytest.skip(f"CUDA runtime unavailable: {exc}")

from quant_evaluator.kernels.gpu.predictive import _autocorr_lag, rank_ic_decay


def _fixture():
    t = np.arange(100_000)
    x = np.empty((len(t), 3), dtype=np.float64)
    x[:, 0] = 0.25 + np.where(t % 2, 2e-8, -2e-8)
    rng = np.random.default_rng(408)
    x[:, 1] = 0.25 + rng.integers(-4, 5, len(t)) * 2e-8
    x[:, 2] = 0.25
    x[rng.random(x.shape) < 0.07] = np.nan
    x[:, 2] = 0.25
    return x


def _pairwise_oracle(x, lag):
    out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        left, right = x[:-lag, f], x[lag:, f]
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.sum() >= 2:
            a, b = left[valid], right[valid]
            if np.ptp(a) > 0 and np.ptp(b) > 0:
                out[f] = pearsonr(a, b).statistic
    return out



def test_gpu_autocorrelation_uses_stable_pairwise_centered_covariance():
    x = _fixture()
    got = _autocorr_lag(cp.asarray(x), 5).get()
    expected = _pairwise_oracle(x, 5)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10,
                               equal_nan=True)
    assert got[0] == pytest.approx(-1.0, abs=1e-10)
    assert np.isnan(got[2])



def test_gpu_rank_ic_decay_matches_pairwise_centered_oracle():
    x = _fixture()
    expected = np.nanmean(np.stack([_pairwise_oracle(x, lag)
                                    for lag in (1, 5, 10, 20)]), axis=0)
    got = rank_ic_decay(cp.asarray(x))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10,
                               equal_nan=True)
    assert got[0] == pytest.approx(0.0, abs=1e-10)
    assert np.isnan(got[2])


def test_nonbinary_constant_and_nan_columns_have_nan_autocorrelation():
    rng = np.random.default_rng(9904)
    t = np.arange(100_000)
    x = np.empty((len(t), 6), dtype=np.float64)
    x[:, 0] = 0.1
    x[:, 1] = 0.3
    x[:, 2] = 0.996908
    x[:, 3] = 0.1
    x[:, 4] = 0.3
    x[:, 5] = 0.25 + rng.integers(-3, 4, len(t)) * 2e-8
    x[t % 17 == 0, 3] = np.nan
    x[t % 5 == 4, 4] = np.nan
    x[rng.random(len(t)) < 0.08, 5] = np.nan
    for lag in (1, 5, 10, 20):
        got = _autocorr_lag(cp.asarray(x), lag).get()
        expected = _pairwise_oracle(x, lag)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10,
                                   equal_nan=True)
        assert np.isnan(got[:5]).all()
    decay = rank_ic_decay(cp.asarray(x))
    assert np.isnan(decay[:5]).all()
