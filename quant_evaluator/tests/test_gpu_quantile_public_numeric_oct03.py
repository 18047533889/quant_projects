"""Public quantile-return integration checks with a Decimal CPU oracle."""
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from quant_evaluator.kernels.gpu.quantile import batched_quantile_returns


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # pragma: no cover - hardware-dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def _decimal_mean(values):
    finite = [float(v) for v in values if math.isfinite(float(v))]
    with localcontext() as ctx:
        ctx.prec = 2000
        return float(sum((Decimal.from_float(v) for v in finite), Decimal(0)) / Decimal(len(finite)))


def test_public_extreme_cancellation_matches_decimal():
    cp = _cuda()
    min_subnormal = float.fromhex("0x0.0000000000001p-1022")
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    returns = [maximum, 3.0 * min_subnormal, -maximum]
    factors = cp.asarray([[[0.0, 1.0, 2.0]]], dtype=cp.float64)
    values, counts = batched_quantile_returns(
        factors, cp.asarray([returns], dtype=cp.float64), n_quantiles=1,
        min_assets=np.int64(1), return_counts=True, workspace_bytes=1 << 20,
    )
    expected = _decimal_mean(returns)
    actual = cp.asnumpy(values)[0, 0, 0]
    assert actual.hex() == expected.hex()
    assert cp.asnumpy(counts)[0, 0, 0] == 3


def test_public_float32_returns_cast_and_numpy_min_assets_with_tiny_budget():
    cp = _cuda()
    factors = cp.asarray([[[0.0, 1.0]]], dtype=cp.float64)
    returns = cp.asarray([[1.0, 3.0]], dtype=cp.float32)
    values, counts = batched_quantile_returns(
        factors, returns, n_quantiles=np.int64(1), min_assets=np.int64(2),
        return_counts=True, workspace_bytes=8192,
    )
    assert values.dtype == cp.float64
    assert cp.asnumpy(values)[0, 0, 0] == 2.0
    assert cp.asnumpy(counts)[0, 0, 0] == 2


def test_public_finite_pair_count_excludes_nan_inf_and_invalid_factor():
    cp = _cuda()
    factors = cp.asarray([[[0.0, 1.0, 2.0, 3.0, 4.0, cp.nan]]], dtype=cp.float64)
    returns = cp.asarray([[1.0, cp.nan, 4.0, cp.inf, 5.0, 100.0]], dtype=cp.float64)
    values, counts = batched_quantile_returns(
        factors, returns, n_quantiles=2, min_assets=2, return_counts=True,
        workspace_bytes=1 << 20,
    )
    got_values = cp.asnumpy(values)[0, :, 0]
    got_counts = cp.asnumpy(counts)[0, :, 0]
    assert math.isnan(got_values[0])
    assert got_counts.tolist() == [1, 2]
    assert got_values[1] == 4.5


def test_public_exact_path_fails_closed_when_tiny_budget_cannot_admit_it():
    cp = _cuda()
    factors = cp.asarray([[[0.0, 1.0]]], dtype=cp.float64)
    returns = cp.asarray([[1e308, -1e308]], dtype=cp.float64)
    with pytest.raises(MemoryError, match="exact repair"):
        batched_quantile_returns(
            factors, returns, n_quantiles=1, min_assets=np.int64(1),
            workspace_bytes=8192,
        )
