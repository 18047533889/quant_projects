"""Bounded finance-risk guard contract and small CUDA oracle."""
from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.kernels.gpu import quantile_finance_guard as finance_guard


def test_quantile_count_is_bounded_for_static_bucket_aggregates():
    assert finance_guard.validate_quantile_count(1) == 1
    assert finance_guard.validate_quantile_count(32) == 32
    for q in (0, 33, True, 1.5):
        with pytest.raises(ValueError):
            finance_guard.validate_quantile_count(q)


def test_local_workspace_reports_compiled_bytes_times_rounded_threads():
    class Kernel:
        attributes = {"local_size_bytes": 96}

        @staticmethod
        def compile():
            return None

    report = finance_guard.local_workspace_bytes(Kernel(), rows=129, block_size=128)
    assert report == {
        "local_size_bytes_per_thread": 96,
        "launched_threads": 256,
        "local_workspace_bytes": 96 * 256,
    }


def test_small_cuda_guard_oracle_covers_safe_risk_subnormal_mixedscale_and_count_error():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    q, rows, n = 2, 6, 32
    bucket = np.zeros((rows, n), dtype=np.int32)
    values = np.full((rows, n), 0.02, dtype=np.float64)
    counts = np.zeros((rows, q), dtype=np.int64)
    means = np.zeros((rows, q), dtype=np.float64)

    # Ordinary finite buckets: the N-based bound remains below 1e-12.
    bucket[0, n // 2:] = 1
    counts[0] = (n // 2, n // 2)
    means[0] = (0.02, 0.02)

    # Global finite-sum overflow.
    values[1] = np.finfo(np.float64).max
    counts[1] = (n, 0)
    means[1] = (np.inf, np.nan)

    # Global subnormal row.
    unit = float.fromhex("0x0.0000000000001p-1022")
    values[2] = unit
    counts[2] = (n, 0)
    means[2] = (unit, np.nan)

    # Mixed-scale finite row that must remain on exact repair path.
    values[3] = 0.0
    values[3, :6] = (1.0, 2.0**-53, 0.0, 1e4, -1e4, 0.0)
    counts[3] = (n, 0)
    means[3] = (float(np.sum(values[3]) / n), np.nan)

    # Non-finite selected mean.
    counts[4] = (n, 0)
    means[4] = (np.nan, np.nan)

    # Deliberate finite-selected-count mismatch must set error metadata.
    counts[5] = (n - 1, 0)
    means[5] = (0.02, np.nan)

    bucket_device = cp.asarray(bucket)
    values_device = cp.asarray(values)
    counts_device = cp.asarray(counts)
    means_device = cp.asarray(means)
    risk_device = cp.empty((rows * q,), dtype=cp.uint8)
    error_device = cp.zeros((), dtype=cp.int32)
    kernel = finance_guard.compile_finance_risk_guard(cp)
    kernel((rows,), (1,), (
        bucket_device, values_device, counts_device, means_device,
        risk_device, error_device, rows, n, q, 10,
    ))
    cp.cuda.Stream.null.synchronize()

    risk = cp.asnumpy(risk_device).reshape(rows, q)
    assert risk[0].tolist() == [0, 0]
    assert risk[1].tolist() == [1, 0]
    assert risk[2].tolist() == [1, 0]
    assert risk[3].tolist() == [1, 0]
    assert risk[4].tolist() == [1, 0]
    assert int(error_device.item()) == 1

def test_small_cuda_finance_rows_refine_global_risk_without_losing_n_gamma():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    q, n = 20, 6000
    bucket = np.arange(n, dtype=np.int32).reshape(1, n) % q
    values = np.full((1, n), 0.02, dtype=np.float64)
    counts = np.full((1, q), n // q, dtype=np.int64)
    means = np.full((1, q), 0.02, dtype=np.float64)
    risk = cp.empty((q,), dtype=cp.uint8)
    error = cp.zeros((), dtype=cp.int32)
    kernel = finance_guard.compile_finance_risk_guard(cp)
    kernel((1,), (1,), (
        cp.asarray(bucket), cp.asarray(values), cp.asarray(counts),
        cp.asarray(means), risk, error, 1, n, q, 10,
    ))
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(risk), np.zeros(q, dtype=np.uint8))
    assert int(error.item()) == 0
