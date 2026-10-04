"""Budget-derived CUDA launch width for exact quantile-mean repairs."""
from __future__ import annotations

from decimal import Decimal, localcontext
import math

import pytest


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # pragma: no cover - hardware-dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def _decimal_mean(values):
    with localcontext() as ctx:
        ctx.prec = 2000
        return float(sum((Decimal.from_float(v) for v in values), Decimal(0)) /
                     Decimal(len(values)))


def _workspace_for_threads(cp, numeric, bucket, means, threads):
    rows, ncols = bucket.shape
    q = means.shape[1]
    guard = cp.RawKernel(numeric._RISK_GUARD, "risk_guard", options=("--std=c++11",))
    fixed = cp.RawKernel(numeric._FIXED_MEAN, "fixed_mean", options=("--std=c++11",))
    guard_local = numeric._local_size_bytes(guard)
    fixed_local = numeric._local_size_bytes(fixed)
    base = rows * (34 + 40 * q) + 4
    guard_threads = ((rows + 127) // 128) * 128
    normal = base + guard_local * guard_threads
    return normal + threads * max(8192, fixed_local)


def _capture_real_launches(monkeypatch, cp):
    real_raw_kernel = cp.RawKernel
    launches = []

    class RecordingKernel:
        def __init__(self, code, name, **kwargs):
            self.name = name
            self.real = real_raw_kernel(code, name, **kwargs)

        def compile(self):
            return self.real.compile()

        @property
        def attributes(self):
            return self.real.attributes

        def __call__(self, grid, block, args):
            launches.append((self.name, grid, block))
            return self.real(grid, block, args)

    monkeypatch.setattr(cp, "RawKernel", RecordingKernel)
    return launches


def test_budget_for_one_fixed_mean_thread_allows_real_exact_repairs(monkeypatch):
    """Catches a fixed 32-thread minimum that rejects a valid one-thread budget."""
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    bucket, labels = cp.asarray([[q for q in range(4) for _ in range(3)]], dtype=cp.int32), cp.asarray(
        [[v for q in range(4) for v in (1e308, -1e308, float(q + 1))]], dtype=cp.float64)
    means = cp.full((1, 4), cp.nan, dtype=cp.float64)
    counts = cp.full((1, 4), 3, dtype=cp.int64)
    launches = _capture_real_launches(monkeypatch, cp)
    workspace = _workspace_for_threads(cp, numeric, bucket, means, 1)

    numeric.repair_quantile_means_gpu(bucket, labels, means, counts, 1,
                                      workspace_bytes=workspace)

    fixed_launches = [entry for entry in launches if entry[0] == "fixed_mean"]
    assert len(fixed_launches) == 4
    assert all(grid == (1,) and block == (1,) for _, grid, block in fixed_launches)
    actual = cp.asnumpy(means)[0]
    expected = [_decimal_mean([1e308, -1e308, float(q + 1)]) for q in range(4)]
    assert [v.hex() for v in actual] == [v.hex() for v in expected]


def test_less_than_one_actual_local_memory_thread_fails_closed(monkeypatch):
    """Catches launching an exact kernel when its measured per-thread budget is unavailable."""
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    bucket, labels = cp.asarray([[q for q in range(4) for _ in range(3)]], dtype=cp.int32), cp.asarray(
        [[v for q in range(4) for v in (1e308, -1e308, float(q + 1))]], dtype=cp.float64)
    means = cp.full((1, 4), 17.0, dtype=cp.float64)
    counts = cp.full((1, 4), 3, dtype=cp.int64)
    launches = _capture_real_launches(monkeypatch, cp)
    workspace = _workspace_for_threads(cp, numeric, bucket, means, 1) - 1

    with pytest.raises(MemoryError, match="exact repair requires"):
        numeric.repair_quantile_means_gpu(bucket, labels, means, counts, 1,
                                          workspace_bytes=workspace)

    assert not any(name == "fixed_mean" for name, _, _ in launches)
    assert cp.asnumpy(means).tolist() == [[17.0] * 4]


def test_two_thread_budget_tiles_risky_buckets_with_real_kernel_and_decimal_oracle(monkeypatch):
    """Catches ignoring the admitted thread cap or mis-sized multi-ID launches."""
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    bucket, labels = cp.asarray([[q for q in range(5) for _ in range(3)]], dtype=cp.int32), cp.asarray(
        [[v for q in range(5) for v in (1e308, -1e308, float(q + 1))]], dtype=cp.float64)
    means = cp.full((1, 5), cp.nan, dtype=cp.float64)
    counts = cp.full((1, 5), 3, dtype=cp.int64)
    launches = _capture_real_launches(monkeypatch, cp)
    workspace = _workspace_for_threads(cp, numeric, bucket, means, 2)

    numeric.repair_quantile_means_gpu(bucket, labels, means, counts, 1,
                                      workspace_bytes=workspace)

    fixed_launches = [entry for entry in launches if entry[0] == "fixed_mean"]
    # The final tile contains one ID, but launch width remains the admitted cap.
    assert [block for _, _, block in fixed_launches] == [(2,), (2,), (2,)]
    assert all(grid == (1,) for _, grid, _ in fixed_launches)
    actual = cp.asnumpy(means)[0]
    expected = [_decimal_mean([1e308, -1e308, float(q + 1)]) for q in range(5)]
    assert [v.hex() for v in actual] == [v.hex() for v in expected]


def test_ample_thread_budget_caps_launch_at_risk_count(monkeypatch):
    """Catches launching the full budgeted width when fewer risky IDs exist."""
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    bucket, labels = cp.asarray([[q for q in range(3) for _ in range(3)]], dtype=cp.int32), cp.asarray(
        [[v for q in range(3) for v in (1e308, -1e308, float(q + 1))]], dtype=cp.float64)
    means = cp.full((1, 3), cp.nan, dtype=cp.float64)
    counts = cp.full((1, 3), 3, dtype=cp.int64)
    launches = _capture_real_launches(monkeypatch, cp)
    workspace = _workspace_for_threads(cp, numeric, bucket, means, 32)

    numeric.repair_quantile_means_gpu(bucket, labels, means, counts, 1,
                                      workspace_bytes=workspace)

    fixed_launches = [entry for entry in launches if entry[0] == "fixed_mean"]
    assert [(grid, block) for _, grid, block in fixed_launches] == [((1,), (3,))]
    actual = cp.asnumpy(means)[0]
    expected = [_decimal_mean([1e308, -1e308, float(q + 1)]) for q in range(3)]
    assert [v.hex() for v in actual] == [v.hex() for v in expected]


def test_8192_workspace_rejects_risky_exact_repair_before_fixed_launch(monkeypatch):
    """Catches treating the normal-guard minimum as enough for exact repair."""
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    bucket, labels = cp.asarray([[0, 0, 0]], dtype=cp.int32), cp.asarray(
        [[1e308, -1e308, 1.0]], dtype=cp.float64)
    means = cp.full((1, 1), cp.nan, dtype=cp.float64)
    counts = cp.asarray([[3]], dtype=cp.int64)
    launches = _capture_real_launches(monkeypatch, cp)

    with pytest.raises(MemoryError, match="requires"):
        numeric.repair_quantile_means_gpu(bucket, labels, means, counts, 1,
                                          workspace_bytes=8192)

    assert not any(name == "fixed_mean" for name, _, _ in launches)
