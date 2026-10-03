"""Fraction-oracle tests for GPU quantile-shape linear arithmetic."""

from fractions import Fraction
import math

import numpy as np
import pytest


def _exact(value):
    return Fraction.from_float(float(value))


def _round_once(value):
    try:
        return float(value)
    except OverflowError:
        return -math.inf if value < 0 else math.inf


def _oracle(values, metric):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    nq, factor_count = matrix.shape
    output = np.full(factor_count, np.nan, dtype=np.float64)
    for factor in range(factor_count):
        column = matrix[:, factor]
        finite = np.isfinite(column)
        terms = []
        if metric == "curvature":
            terms = [
                _exact(column[q + 1]) - 2 * _exact(column[q]) + _exact(column[q - 1])
                for q in range(1, nq - 1)
                if finite[q - 1] and finite[q] and finite[q + 1]
            ]
            if terms:
                output[factor] = _round_once(sum(terms, Fraction()) / len(terms))
        elif metric == "spread":
            terms = [
                abs(_exact(column[q + 1]) - _exact(column[q]))
                for q in range(nq - 1)
                if finite[q] and finite[q + 1]
            ]
            if terms:
                output[factor] = _round_once(sum(terms, Fraction()) / len(terms))
        elif metric == "extreme_cliff":
            if nq >= 2 and finite[0] and finite[1] and finite[-2] and finite[-1]:
                bottom = _exact(column[1]) - _exact(column[0])
                top = _exact(column[-1]) - _exact(column[-2])
                output[factor] = _round_once((top + bottom) / 2)
        elif metric == "tail_asymmetry":
            mid = nq // 2
            if nq >= 3 and finite[0] and finite[mid] and finite[-1]:
                output[factor] = _round_once(
                    (_exact(column[-1]) - _exact(column[mid]))
                    - (_exact(column[mid]) - _exact(column[0]))
                )
        elif metric == "top_cliff":
            if nq >= 2 and finite[-2] and finite[-1]:
                output[factor] = _round_once(_exact(column[-1]) - _exact(column[-2]))
        elif metric == "bottom_cliff":
            if nq >= 2 and finite[0] and finite[1]:
                output[factor] = _round_once(_exact(column[1]) - _exact(column[0]))
        else:
            raise AssertionError(metric)
    return output


def _gpu_kernels():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from quant_evaluator.kernels.gpu import quantile_shape as kernels

    return cp, {
        "curvature": kernels.quantile_curvature,
        "spread": kernels.quantile_adjacent_spread,
        "extreme_cliff": kernels.quantile_extreme_cliff,
        "tail_asymmetry": kernels.quantile_tail_asymmetry,
        "top_cliff": kernels.top_quantile_cliff,
        "bottom_cliff": kernels.bottom_quantile_cliff,
    }


def _assert_matches_oracle(cp, kernels, matrix, metrics, *, bitwise=False):
    device_values = cp.asarray(matrix, dtype=cp.float64)
    for name in metrics:
        actual = kernels[name](device_values)
        expected = _oracle(matrix, name)
        if bitwise:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_array_equal(actual, expected)


def test_large_finite_linear_metrics_match_once_rounded_fraction_oracles():
    cp, kernels = _gpu_kernels()
    maximum = np.finfo(np.float64).max
    previous = np.nextafter(maximum, 0.0)

    # The stencil's exact sum is zero despite individually overflowing terms.
    constant = np.full((4, 1), 1e308, dtype=np.float64)
    np.testing.assert_array_equal(_oracle(constant, "curvature"), [0.0])
    _assert_matches_oracle(cp, kernels, constant, ("curvature",))

    # Exact cancellation of large dyadic inputs leaves only two ulps.
    near_max_cancellation = np.array([[maximum], [previous], [maximum]])
    expected_curvature = _oracle(near_max_cancellation, "curvature")
    assert expected_curvature[0] == 2.0 * (maximum - previous)
    _assert_matches_oracle(cp, kernels, near_max_cancellation, ("curvature",))

    # Cancellation can also lose low bits at scale 1e200.
    large_offset = np.array([[
        float.fromhex("-0x1.c410b118ae524p+660"),
        float.fromhex("-0x1.c3132eb82f949p+663"),
        float.fromhex("0x1.5ae902d53fdbbp+664"),
        float.fromhex("0x1.27e04abef903ap+665"),
        float.fromhex("0x1.b53035839fa8fp+663"),
    ]]).T
    _assert_matches_oracle(cp, kernels, large_offset, ("curvature",))

    # Each edge is finite, but summing the two edge magnitudes overflows before
    # division in the naive adjacent-spread implementation.
    opposite_extremes = np.array([[-1e308], [0.0], [1e308]])
    assert _oracle(opposite_extremes, "spread")[0] == 1e308
    assert _oracle(opposite_extremes, "extreme_cliff")[0] == 1e308
    assert _oracle(opposite_extremes, "tail_asymmetry")[0] == 0.0
    assert _oracle(opposite_extremes, "top_cliff")[0] == 1e308
    assert _oracle(opposite_extremes, "bottom_cliff")[0] == 1e308
    _assert_matches_oracle(
        cp,
        kernels,
        opposite_extremes,
        ("spread", "extreme_cliff", "tail_asymmetry", "top_cliff", "bottom_cliff"),
    )




def test_small_dyadic_and_nonfinite_masks_match_fraction_oracles_bitwise():
    cp, kernels = _gpu_kernels()
    normal = np.array(
        [[0.0, -4.0], [1.0, -2.0], [4.0, 3.0], [3.0, 5.0], [8.0, 1.0]],
        dtype=np.float64,
    )
    metrics = (
        "curvature", "spread", "extreme_cliff", "tail_asymmetry", "top_cliff", "bottom_cliff",
    )
    _assert_matches_oracle(cp, kernels, normal, metrics, bitwise=True)

    masked = np.array(
        [
            [1.0, np.nan, np.inf],
            [2.0, 2.0, np.nan],
            [np.nan, 3.0, -np.inf],
            [4.0, 4.0, np.nan],
            [8.0, 8.0, np.inf],
        ],
        dtype=np.float64,
    )
    _assert_matches_oracle(cp, kernels, masked, metrics, bitwise=True)


def test_subnormal_ties_true_overflow_and_signed_infinity_match_fraction():
    cp, kernels = _gpu_kernels()
    unit = float.fromhex("0x0.0000000000001p-1022")
    subnormal = np.array([[0.0], [unit], [unit]])
    half_subnormal_cliff = np.array([[0.0], [unit], [unit]])

    for matrix, name in (
        (subnormal, "curvature"),
        (subnormal, "spread"),
        (half_subnormal_cliff, "extreme_cliff"),
    ):
        expected = _oracle(matrix, name)
        actual = kernels[name](cp.asarray(matrix))
        np.testing.assert_array_equal(actual, expected)
    assert _oracle(half_subnormal_cliff, "extreme_cliff")[0] == 0.0

    maximum = np.finfo(np.float64).max
    positive_overflow = np.array([[-maximum], [maximum]])
    negative_overflow = np.array([[maximum], [-maximum]])
    for matrix, name in (
        (positive_overflow, "top_cliff"),
        (positive_overflow, "spread"),
        (negative_overflow, "bottom_cliff"),
    ):
        expected = _oracle(matrix, name)
        actual = kernels[name](cp.asarray(matrix))
        np.testing.assert_array_equal(actual, expected)
    assert np.isposinf(_oracle(positive_overflow, "top_cliff")[0])
    assert np.isneginf(_oracle(negative_overflow, "bottom_cliff")[0])


def test_shape_linear_repair_rejects_bad_workspace_dtype_and_overlap():
    cp, _kernels = _gpu_kernels()
    from quant_evaluator.kernels.gpu.shape_linear_numeric import repair_shape_linear_gpu

    values = cp.asarray(np.array([[1.0], [2.0], [3.0]], dtype=np.float64))
    output = cp.zeros((1,), dtype=cp.float64)
    with pytest.raises(ValueError, match="workspace_bytes"):
        repair_shape_linear_gpu(values, output, "curvature", workspace_bytes=0)
    with pytest.raises(MemoryError, match="workspace"):
        repair_shape_linear_gpu(values, output, "curvature", workspace_bytes=1)
    with pytest.raises(TypeError, match="float64"):
        repair_shape_linear_gpu(values.astype(cp.float32), output, "curvature")
    with pytest.raises(ValueError, match="overlap"):
        repair_shape_linear_gpu(values, values[0], "curvature")
    noncontiguous = cp.asarray(np.arange(6.0).reshape(3, 2))[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        repair_shape_linear_gpu(noncontiguous, output, "curvature")
    for invalid_budget in (True, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="workspace_bytes"):
            repair_shape_linear_gpu(values, output, "curvature", workspace_bytes=invalid_budget)


def test_shape_error_bound_guard_includes_exact_boundary_and_adjacent_floats():
    cp, _kernels = _gpu_kernels()
    from quant_evaluator.kernels.gpu.shape_linear_numeric import repair_shape_linear_gpu
    from quant_evaluator.metrics.shape_linear_numeric import linear_risk_columns

    # top_cliff has coefficient L1=2, hence the guard's (L1+2) term is 4.
    eps = np.finfo(np.float64).eps
    error_bound = 1.0e-12 / (2.0 * eps * (2.0 + 2.0))
    magnitudes = np.array((
        np.nextafter(error_bound, 0.0),
        error_bound,
        np.nextafter(error_bound, np.inf),
    ))
    np.testing.assert_array_equal(
        linear_risk_columns(magnitudes[None, :], coefficient_l1=2),
        [False, True, True],
    )

    sentinel = -123.0
    for index, magnitude in enumerate(magnitudes):
        values = cp.asarray(np.array([[0.0], [magnitude]], dtype=np.float64))
        output = cp.full((1,), sentinel, dtype=cp.float64)
        repair_shape_linear_gpu(values, output, "top_cliff")
        expected = magnitude if index else sentinel
        np.testing.assert_array_equal(output.get(), [expected])


def test_near_underflow_guard_boundary_cancellation_matches_fraction_oracle():
    cp, kernels = _gpu_kernels()
    # For Q=3 curvature uses L1=4 and tests the tiny*(L1+2) boundary.
    tiny = np.finfo(np.float64).tiny
    boundary = tiny * (4.0 + 2.0)
    magnitudes = (
        np.nextafter(boundary, 0.0),
        boundary,
        np.nextafter(boundary, np.inf),
    )
    columns = []
    for magnitude in magnitudes:
        columns.append((magnitude, np.nextafter(magnitude, np.inf), magnitude))
    matrix = np.asarray(columns, dtype=np.float64).T
    actual = kernels["curvature"](cp.asarray(matrix))
    expected = _oracle(matrix, "curvature")
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.isfinite(expected))


def test_shape_linear_repair_rejects_noncurrent_device_when_available():
    cp, _kernels = _gpu_kernels()
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("only one CUDA device is available")
    from quant_evaluator.kernels.gpu.shape_linear_numeric import repair_shape_linear_gpu

    with cp.cuda.Device(0):
        foreign = cp.asarray(np.array([[1.0], [2.0], [3.0]], dtype=np.float64))
    with cp.cuda.Device(1):
        output = cp.zeros((1,), dtype=cp.float64)
        with pytest.raises(ValueError, match="current CUDA device"):
            repair_shape_linear_gpu(foreign, output, "curvature")
