"""Small independent edge fixtures for the bounded-Q finance guard."""
from fractions import Fraction
import numpy as np
import pytest
from quant_evaluator.kernels.gpu import quantile_finance_guard as guard

def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1: pytest.skip("CUDA device unavailable")
    except Exception as exc: pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp

def _exact_mean(values):
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite: return np.nan
    return float(sum((Fraction.from_float(v) for v in finite), Fraction()) / len(finite))

@pytest.mark.parametrize("q", [1, 2, 5, 20, 32])
def test_single_observation_rows_cover_all_q_and_min_assets(q):
    cp = _cuda()
    bucket = np.array([[-1], [0]], dtype=np.int32)
    values = np.array([[np.nan], [0.03125]], dtype=np.float64)
    counts = np.zeros((2, q), dtype=np.int64); counts[1, 0] = 1
    means = np.full((2, q), np.nan, dtype=np.float64); means[1, 0] = 0.03125
    kernel = guard.compile_finance_risk_guard(cp)
    risk = cp.empty((2*q,), dtype=cp.uint8); error = cp.zeros((), dtype=cp.int32)
    kernel((2,), (1,), (cp.asarray(bucket), cp.asarray(values), cp.asarray(counts), cp.asarray(means), risk, error, 2, 1, q, 1))
    cp.cuda.Stream.null.synchronize()
    result = cp.asnumpy(risk).reshape(2, q)
    assert result[0].tolist() == [0]*q and result[1].tolist() == [0]*q
    assert int(error.item()) == 0
    assert means[1, 0] == _exact_mean(values[1])
    risk2 = cp.empty((2*q,), dtype=cp.uint8); error2 = cp.zeros((), dtype=cp.int32)
    kernel((2,), (1,), (cp.asarray(bucket), cp.asarray(values), cp.asarray(counts), cp.asarray(means), risk2, error2, 2, 1, q, 2))
    cp.cuda.Stream.null.synchronize()
    assert cp.asnumpy(risk2).reshape(2, q)[1, 0] == 0
    assert int(error2.item()) == 0

def test_bucket_id_and_finite_count_mismatch_are_reported():
    cp = _cuda()
    bucket = np.array([[0, 0, 1], [0, -1, 2]], dtype=np.int32)
    values = np.array([[0.25, -0.25, 0.5], [0.5, np.nan, 0.75]], dtype=np.float64)
    counts = np.array([[2, 1], [0, 1]], dtype=np.int64)
    means = np.array([[0.0, 0.5], [np.nan, 0.75]], dtype=np.float64)
    risk = cp.empty((4,), dtype=cp.uint8); error = cp.zeros((), dtype=cp.int32)
    kernel = guard.compile_finance_risk_guard(cp)
    kernel((2,), (1,), (cp.asarray(bucket), cp.asarray(values), cp.asarray(counts), cp.asarray(means), risk, error, 2, 3, 2, 1))
    cp.cuda.Stream.null.synchronize()
    assert int(error.item()) == 1
    assert cp.asnumpy(risk).reshape(2, 2)[0].tolist() == [0, 0]

def test_local_workspace_uses_compiled_per_thread_size_and_rounded_launch():
    class Kernel:
        attributes = {"localSizeBytes": 17}
        @staticmethod
        def compile(): return None
    assert guard.local_workspace_bytes(Kernel(), rows=129, block_size=128) == {
        "local_size_bytes_per_thread": 17, "launched_threads": 256,
        "local_workspace_bytes": 4352,
    }
