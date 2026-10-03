"""CUDA guard refinement for finance-scale quantile means."""
from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # pragma: no cover - hardware-dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def _track_fixed_kernel(cp, monkeypatch, *, forbid=False):
    original = cp.RawKernel
    calls = []

    def raw_kernel(source, name, *args, **kwargs):
        if name == "fixed_mean":
            calls.append(name)
            if forbid:
                raise AssertionError("finance-scale safe buckets must skip exact repair")
        return original(source, name, *args, **kwargs)

    monkeypatch.setattr(cp, "RawKernel", raw_kernel)
    return calls


def test_q20_finance_buckets_refine_global_risk_without_exact_kernel(monkeypatch):
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    n, q = 6000, 20
    labels = np.full(n, 0.02, dtype=np.float64)
    bucket = np.arange(n, dtype=np.int32) % q
    bucket[-2:] = -1
    labels[-2] = np.nan
    labels[-1] = np.inf
    counts_host = np.bincount(bucket[bucket >= 0], minlength=q).astype(np.int64)
    means = cp.full((1, q), 0.02, dtype=cp.float64)
    calls = _track_fixed_kernel(cp, monkeypatch, forbid=True)

    numeric.repair_quantile_means_gpu(
        cp.asarray(bucket.reshape(1, -1)),
        cp.asarray(labels.reshape(1, -1)),
        means,
        cp.asarray(counts_host.reshape(1, -1)),
        min_assets=10,
        workspace_bytes=1 << 20,
    )
    np.testing.assert_array_equal(cp.asnumpy(means), np.full((1, q), 0.02))
    assert calls == []


def test_mixed_scale_double_rounding_fixture_still_uses_exact_kernel(monkeypatch):
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    values = [1.0, 2.0 ** -53, 0.0, 1e4, -1e4, 0.0]
    with localcontext() as context:
        context.prec = 2000
        expected = float(sum((Decimal.from_float(v) for v in values), Decimal(0))
                         / Decimal(len(values)))
    labels = cp.asarray([values], dtype=cp.float64)
    bucket = cp.zeros((1, len(values)), dtype=cp.int32)
    counts = cp.asarray([[len(values)]], dtype=cp.int64)
    means = cp.asarray([[float(np.sum(values) / len(values))]], dtype=cp.float64)
    calls = _track_fixed_kernel(cp, monkeypatch)

    numeric.repair_quantile_means_gpu(
        bucket, labels, means, counts, min_assets=1, workspace_bytes=1 << 20,
    )
    assert calls
    assert cp.asnumpy(means)[0, 0].hex() == expected.hex()


def test_all_subnormal_bucket_keeps_exact_repair(monkeypatch):
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    unit = float.fromhex("0x0.0000000000001p-1022")
    values = [unit] * 10 + [1.0] * 10
    labels = cp.asarray([values], dtype=cp.float64)
    bucket = cp.asarray([[0] * 10 + [1] * 10], dtype=cp.int32)
    counts = cp.asarray([[10, 10]], dtype=cp.int64)
    means = cp.asarray([[unit, 1.0]], dtype=cp.float64)
    calls = _track_fixed_kernel(cp, monkeypatch)

    numeric.repair_quantile_means_gpu(
        bucket, labels, means, counts, min_assets=10, workspace_bytes=1 << 20,
    )
    assert calls
    got = cp.asnumpy(means)
    assert got[0, 0].hex() == unit.hex()
    assert got[0, 1] == 1.0
