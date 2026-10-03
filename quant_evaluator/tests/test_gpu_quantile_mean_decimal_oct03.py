"""Independent Decimal oracle for the experimental CUDA bucket-mean repair."""
from __future__ import annotations

from decimal import Decimal, localcontext
from itertools import permutations
import math

import pytest


def _decimal_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return math.nan
    with localcontext() as ctx:
        ctx.prec = 2000
        return float(sum((Decimal.from_float(v) for v in finite), Decimal(0)) / Decimal(len(finite)))


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # pragma: no cover - hardware-dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def _repair_rows(rows: list[list[float]], counts_override=None, min_assets=1, workspace_bytes=1 << 20):
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    r, n = len(rows), len(rows[0])
    x = cp.asarray(rows, dtype=cp.float64)
    bucket = cp.zeros((r, n), dtype=cp.int32)
    finite_counts = [sum(math.isfinite(v) for v in row) for row in rows]
    if counts_override is not None:
        finite_counts = counts_override
    counts = cp.asarray(finite_counts, dtype=cp.int64).reshape(r, 1)
    means = cp.full((r, 1), cp.nan, dtype=cp.float64)
    numeric.repair_quantile_means_gpu(bucket, x, means, counts, min_assets,
                                      workspace_bytes=workspace_bytes)
    return cp.asnumpy(means[:, 0])


@pytest.mark.parametrize("values", [
    [1e308, 1e308, -1e308],
    [1e308, -1e308, 1.0, 1.0],
    [1e308, -1e308, 1e308, -1e308],
    [0.0, 0.0],
    [float.fromhex("0x0.0000000000001p-1022"), 1e308, -1e308],
    [float.fromhex("0x1.fffffffffffffp+1023"),
     float.fromhex("0x0.0000000000001p-1022") * 3,
     -float.fromhex("0x1.fffffffffffffp+1023")],
    [float.fromhex("0x0.0000000000001p-1022")] * 3,
    [float.fromhex("0x1.fffffffffffffp+1023"),
     float.fromhex("0x1.fffffffffffffp+1023"),
     -float.fromhex("0x1.fffffffffffffp+1023")],
    [float("nan"), 3.0, -1.0, 4.0],
    [1.0, float("inf"), -float("inf"), float("nan"), 3.0],
])
def test_exact_decimal_mean_cases(values):
    actual = _repair_rows([values])[0]
    expected = _decimal_mean(values)
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual.hex() == expected.hex()


def test_all_permutations_of_cancellation_fixture_match_decimal():
    fixture = [1e308, 1.0, -1e308, 1.0]
    cases = [list(p) for p in permutations(fixture)]
    actual = _repair_rows(cases)
    expected = _decimal_mean(fixture)
    assert all(v.hex() == expected.hex() for v in actual)


def test_normal_h52_uses_division_remainder_for_ties_to_even():
    unit = float.fromhex("0x0.0000000000001p-1022")
    smallest_normal = float.fromhex("0x1.0000000000000p-1022")
    values = [smallest_normal + unit, smallest_normal + 2 * unit]
    actual = _repair_rows([values])[0]
    expected = _decimal_mean(values)
    assert actual.hex() == expected.hex()


def test_count_min_assets_all_nan_and_workspace_contract():
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    x = cp.asarray([[1e308, -1e308, 7.0, cp.nan], [cp.nan] * 4], dtype=cp.float64)
    bucket = cp.zeros((2, 4), dtype=cp.int32)
    counts = cp.asarray([[3], [0]], dtype=cp.int64)
    means = cp.asarray([[123.0], [cp.nan]], dtype=cp.float64)
    numeric.repair_quantile_means_gpu(bucket, x, means, counts, 4, workspace_bytes=1 << 20)
    got = cp.asnumpy(means[:, 0])
    assert got[0] == 123.0  # below min_assets: untouched
    assert math.isnan(got[1])  # empty/all-NaN bucket: untouched

    with pytest.raises(MemoryError, match="requires"):
        numeric.repair_quantile_means_gpu(bucket, x, means, counts, 1, workspace_bytes=1)


def test_mismatched_count_is_not_silently_repaired():
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    x = cp.asarray([[1e308, -1e308]], dtype=cp.float64)
    bucket = cp.zeros((1, 2), dtype=cp.int32)
    counts = cp.asarray([[1]], dtype=cp.int64)  # actual selected count is 2
    means = cp.asarray([[42.0]], dtype=cp.float64)
    with pytest.raises(ValueError, match="does not match"):
        numeric.repair_quantile_means_gpu(bucket, x, means, counts, 1, workspace_bytes=1 << 20)


def test_normal_only_tiny_workspace_and_risky_tiny_rejection():
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    bucket = cp.zeros((1, 2), dtype=cp.int32)
    ordinary = cp.asarray([[1.0, 2.0]], dtype=cp.float64)
    count = cp.asarray([[2]], dtype=cp.int64)
    means = cp.asarray([[1.5]], dtype=cp.float64)
    numeric.repair_quantile_means_gpu(bucket, ordinary, means, count, 1,
                                      workspace_bytes=8192)
    assert cp.asnumpy(means)[0, 0] == 1.5

    extreme = cp.asarray([[1e308, -1e308]], dtype=cp.float64)
    means = cp.full((1, 1), cp.nan, dtype=cp.float64)
    with pytest.raises(MemoryError, match="exact repair"):
        numeric.repair_quantile_means_gpu(bucket, extreme, means, count, 1,
                                          workspace_bytes=8192)


def test_rejects_noncontiguous_inputs_and_output_aliases():
    cp = _cuda()
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    bucket = cp.zeros((1, 4), dtype=cp.int32)[:, ::2]
    labels = cp.asarray([[1.0, 2.0]], dtype=cp.float64)
    counts = cp.asarray([[2]], dtype=cp.int64)
    means = cp.asarray([[1.5]], dtype=cp.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        numeric.repair_quantile_means_gpu(bucket, labels, means, counts, 1,
                                          workspace_bytes=1 << 20)

    bucket = cp.zeros((1, 2), dtype=cp.int32)
    labels = cp.asarray([[1.0, 2.0]], dtype=cp.float64)
    aliased_means = labels[:, :1]
    with pytest.raises(ValueError, match="overlap labels"):
        numeric.repair_quantile_means_gpu(bucket, labels, aliased_means, counts, 1,
                                          workspace_bytes=1 << 20)


def test_rejects_inputs_outside_current_device_when_multiple_devices_exist():
    cp = _cuda()
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count < 2:
        pytest.skip("current-device mismatch requires at least two CUDA devices")
    import quant_evaluator.kernels.gpu.quantile_numeric as numeric

    current = cp.cuda.Device().id
    other = (current + 1) % device_count
    with cp.cuda.Device(other):
        bucket = cp.zeros((1, 2), dtype=cp.int32)
        labels = cp.asarray([[1.0, 2.0]], dtype=cp.float64)
        means = cp.asarray([[1.5]], dtype=cp.float64)
        counts = cp.asarray([[2]], dtype=cp.int64)
    with pytest.raises(ValueError, match="current CUDA device"):
        numeric.repair_quantile_means_gpu(bucket, labels, means, counts, 1,
                                          workspace_bytes=1 << 20)
