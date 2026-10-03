from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from quant_evaluator.kernels.gpu import quantile_finance_guard as finance_guard


class _MockKernel:
    def __init__(self, source, name, options):
        self.source = source
        self.name = name
        self.options = options
        self.attributes = {"local_size_bytes": 96}
        self.compile_calls = 0

    def compile(self):
        self.compile_calls += 1


class _MockCuPy:
    def __init__(self):
        self.calls = []

    def RawKernel(self, source, name, *, options):
        kernel = _MockKernel(source, name, options)
        self.calls.append(kernel)
        return kernel


@pytest.mark.parametrize("q,capacity", [(1, 1), (2, 2), (5, 8), (20, 32), (32, 32)])
def test_factory_uses_only_whitelisted_capacity_flags(q, capacity):
    cp = _MockCuPy()

    kernel = finance_guard.compile_finance_risk_guard(cp, n_quantiles=q)

    assert kernel.name == "finance_risk_guard"
    assert kernel.options == ("--std=c++11", f"-DFINANCE_Q_CAPACITY={capacity}")
    assert kernel.compile_calls == 1
    assert f"nq>FINANCE_Q_CAPACITY" in kernel.source
    assert kernel.source.count("[FINANCE_Q_CAPACITY]") == 4
    assert "q<FINANCE_Q_CAPACITY" in kernel.source


def test_none_keeps_legacy_capacity_32_without_define():
    cp = _MockCuPy()

    kernel = finance_guard.compile_finance_risk_guard(cp)

    assert kernel.options == ("--std=c++11",)
    assert "#define FINANCE_Q_CAPACITY 32" in kernel.source


@pytest.mark.parametrize("q", [0, 33, True, 1.5, "5"])
def test_invalid_specialization_rejected_before_rawkernel(q):
    cp = _MockCuPy()

    with pytest.raises(ValueError):
        finance_guard.compile_finance_risk_guard(cp, n_quantiles=q)

    assert cp.calls == []


def test_specialized_compiled_local_workspace_remains_reported():
    cp = _MockCuPy()
    kernel = finance_guard.compile_finance_risk_guard(cp, n_quantiles=5)

    report = finance_guard.local_workspace_bytes(kernel, rows=129, block_size=128)

    assert report == {
        "local_size_bytes_per_thread": 96,
        "launched_threads": 256,
        "local_workspace_bytes": 96 * 256,
    }


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # pragma: no cover - host dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def _fraction_mean(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return np.nan
    total = sum((Fraction.from_float(value) for value in finite), Fraction())
    return float(total / len(finite))


def _run_kernel(cp, kernel, bucket, data, counts, means, q):
    rows, ncols = bucket.shape
    risk = cp.empty((rows * q,), dtype=cp.uint8)
    error = cp.zeros((), dtype=cp.int32)
    kernel((rows,), (1,), (
        cp.asarray(bucket, dtype=cp.int32),
        cp.asarray(data, dtype=cp.float64),
        cp.asarray(counts, dtype=cp.int64),
        cp.asarray(means, dtype=cp.float64),
        risk, error, rows, ncols, q, 1,
    ))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(risk).reshape(rows, q), int(error.item())


def test_small_specializations_match_legacy_guard_and_fraction_oracle():
    cp = _cuda()
    for q in (1, 2, 5, 20, 32):
        rows, ncols = 6, 64
        bucket = np.tile(np.arange(ncols, dtype=np.int32) % q, (rows, 1))
        data = np.full((rows, ncols), 0.02, dtype=np.float64)
        data[0, 0] = np.nan
        data[0, 1] = np.inf

        data[1] = np.finfo(np.float64).max
        data[2] = float.fromhex("0x0.0000000000001p-1022")
        data[3] = 0.0
        data[3, :6] = (1.0, 2.0**-53, 0.0, 1e4, -1e4, 0.0)

        counts = np.stack([
            np.sum((bucket == index) & np.isfinite(data), axis=1, dtype=np.int64)
            for index in range(q)
        ], axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            means = np.stack([
                np.sum(np.where((bucket == index) & np.isfinite(data),
                                data, 0.0), axis=1)
                / np.maximum(counts[:, index], 1)
                for index in range(q)
            ], axis=1)
        # Make one selected mean non-finite and one count inconsistent.
        means[4, 0] = np.nan
        counts[5, 0] -= 1

        legacy = finance_guard.compile_finance_risk_guard(cp)
        specialized = finance_guard.compile_finance_risk_guard(cp, n_quantiles=q)
        legacy_risk, legacy_error = _run_kernel(
            cp, legacy, bucket, data, counts, means, q,
        )
        actual_risk, actual_error = _run_kernel(
            cp, specialized, bucket, data, counts, means, q,
        )

        np.testing.assert_array_equal(actual_risk, legacy_risk)
        assert actual_error == legacy_error == 1
        assert actual_risk[1, 0] == 1
        assert actual_risk[2, 0] == 1
        assert actual_risk[4, 0] == 1
        assert actual_risk[5, 0] == 0
        for index in range(q):
            if counts[0, index] > 0 and actual_risk[0, index] == 0:
                exact = _fraction_mean(data[0, bucket[0] == index])
                assert abs(means[0, index] - exact) <= 1e-12

        workspace = finance_guard.local_workspace_bytes(
            specialized, rows=rows, block_size=128,
        )
        assert workspace["launched_threads"] == 128
        assert workspace["local_workspace_bytes"] == (
            workspace["local_size_bytes_per_thread"] * 128
        )
