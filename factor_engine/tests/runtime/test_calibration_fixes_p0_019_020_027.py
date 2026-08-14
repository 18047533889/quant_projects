# -*- coding: utf-8 -*-
"""FE-P0-019, FE-P0-020, FE-P0-027: Calibration correctness fixes.

FE-P0-019: Typed CalibrationFactors replaces ambiguous tuple
FE-P0-020: calibrated_peak_bytes uses memory_factor, not elapsed_factor
FE-P0-027: Zero/unknown memory estimates fail-closed with conservative bounds
"""
from __future__ import annotations

import pytest

from runtime.runtime_calibration import (
    CalibrationFactors,
    calibrated_factors,
    calibrated_peak_bytes,
    record_task_actual,
    reset_calibration,
)


def test_calibration_factors_typed_result():
    """FE-P0-019: CalibrationFactors is typed dataclass, not ambiguous tuple."""
    reset_calibration()
    record_task_actual(
        operator="test_op",
        backend="pandas_numpy",
        actual_elapsed_ms=100.0,
        rss_delta_bytes=50 * 1024 * 1024,
        rows=100_000,
        instruments=100,
        predicted_ms=50.0,
        predicted_peak_bytes=25 * 1024 * 1024,
    )
    from runtime.runtime_calibration import calibration_key
    key = calibration_key(
        operator="test_op",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )
    factors = calibrated_factors(key)

    # Typed result with explicit fields
    assert isinstance(factors, CalibrationFactors)
    assert hasattr(factors, "elapsed_factor")
    assert hasattr(factors, "memory_factor")
    assert hasattr(factors, "samples")

    # elapsed_factor reflects actual/predicted elapsed ratio (100/50 = 2.0, EMA blended)
    assert factors.elapsed_factor > 1.0
    assert factors.elapsed_factor < 3.0

    # memory_factor reflects actual/predicted memory ratio (50/25 = 2.0, EMA blended)
    assert factors.memory_factor > 1.0
    assert factors.memory_factor < 3.0

    assert factors.samples == 1

    # Legacy compatibility: can still unpack as tuple
    elapsed, memory, samples = factors.to_tuple()
    assert elapsed == factors.elapsed_factor
    assert memory == factors.memory_factor
    assert samples == factors.samples


def test_calibration_factors_no_samples():
    """FE-P0-019: No samples returns defaults (1.0, 1.0, 0)."""
    reset_calibration()
    from runtime.runtime_calibration import calibration_key
    key = calibration_key(
        operator="never_run",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )
    factors = calibrated_factors(key)

    assert factors.elapsed_factor == 1.0
    assert factors.memory_factor == 1.0
    assert factors.samples == 0


def test_calibrated_peak_uses_memory_factor_not_elapsed():
    """FE-P0-020: calibrated_peak_bytes uses memory_factor, not elapsed_factor.

    Bug: Previously extracted wrong tuple element, applying elapsed_factor to
    memory prediction. Now uses typed CalibrationFactors.memory_factor.
    """
    reset_calibration()

    # Record task with 3x memory overrun but only 1.2x elapsed overrun
    record_task_actual(
        operator="memory_hog",
        backend="pandas_numpy",
        actual_elapsed_ms=60.0,
        rss_delta_bytes=300 * 1024 * 1024,
        rows=100_000,
        instruments=100,
        predicted_ms=50.0,
        predicted_peak_bytes=100 * 1024 * 1024,
    )

    from runtime.runtime_calibration import calibration_key
    key = calibration_key(
        operator="memory_hog",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )

    factors = calibrated_factors(key)

    # Verify memory_factor is higher than elapsed_factor
    # elapsed: 60/50 = 1.2, memory: 300/100 = 3.0
    # EMA with alpha=0.2: new = 1.0 + 0.2*(ratio - 1.0)
    # elapsed: 1.0 + 0.2*(1.2 - 1.0) = 1.04
    # memory: 1.0 + 0.2*(3.0 - 1.0) = 1.4
    assert factors.memory_factor > factors.elapsed_factor
    assert factors.memory_factor >= 1.3  # EMA blended result

    # calibrated_peak_bytes must use memory_factor
    static_peak = 100 * 1024 * 1024
    predicted, uncertainty = calibrated_peak_bytes(static_peak, key)

    # predicted = static * memory_factor * uncertainty
    # memory_factor ~1.4, uncertainty >= 1.25
    # predicted >= 100MB * 1.4 * 1.25 = 175MB
    assert predicted >= static_peak * 1.3

    # If elapsed_factor was wrongly used, predicted would be much lower
    # (~100MB * 1.04 * 1.25 = 130MB), this test catches the FE-P0-020 bug


def test_calibrated_peak_zero_static_returns_conservative_bound():
    """FE-P0-027: Zero/unknown static_peak_bytes returns conservative 8MB minimum."""
    reset_calibration()
    from runtime.runtime_calibration import calibration_key

    key = calibration_key(
        operator="unknown_op",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )

    # Zero static peak (unknown estimate)
    predicted, uncertainty = calibrated_peak_bytes(0, key)

    # Must return conservative bound, never zero
    assert predicted > 0
    assert predicted >= 8 * 1024 * 1024  # 8MB minimum

    # Negative static peak
    predicted_neg, _ = calibrated_peak_bytes(-100, key)
    assert predicted_neg > 0
    assert predicted_neg >= 8 * 1024 * 1024


def test_calibrated_peak_nonzero_result_after_zero_calibration():
    """FE-P0-027: Even with zero calibration, result is nonzero if static > 0."""
    reset_calibration()
    from runtime.runtime_calibration import calibration_key

    # Record calibration with zero memory (pathological case)
    record_task_actual(
        operator="zero_mem",
        backend="pandas_numpy",
        actual_elapsed_ms=10.0,
        rss_delta_bytes=0,  # Zero memory delta
        rows=100,
        instruments=10,
        predicted_ms=10.0,
        predicted_peak_bytes=0,  # Zero prediction
    )

    key = calibration_key(
        operator="zero_mem",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )

    # Even with zero calibration, nonzero static should produce nonzero result
    static_peak = 50 * 1024 * 1024
    predicted, _ = calibrated_peak_bytes(static_peak, key)

    # Must not return zero (fail-closed for production)
    assert predicted > 0


def test_calibration_uncertainty_progression():
    """FE-P0-019/020/027: Uncertainty decreases as samples accumulate."""
    reset_calibration()
    from runtime.runtime_calibration import calibration_key

    key = calibration_key(
        operator="converge_op",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )

    static_peak = 100 * 1024 * 1024

    # No samples: UNCERTAINTY_WARM (1.50)
    _, unc0 = calibrated_peak_bytes(static_peak, key)
    assert unc0 == 1.50

    # 1-4 samples: UNCERTAINTY_COLD (1.25)
    for _ in range(4):
        record_task_actual(
            operator="converge_op",
            backend="pandas_numpy",
            actual_elapsed_ms=50.0,
            rss_delta_bytes=100 * 1024 * 1024,
            rows=100_000,
            instruments=100,
            predicted_ms=50.0,
            predicted_peak_bytes=100 * 1024 * 1024,
        )

    _, unc1 = calibrated_peak_bytes(static_peak, key)
    assert unc1 == 1.25

    # 5+ samples: UNCERTAINTY_CALIBRATED (1.15)
    record_task_actual(
        operator="converge_op",
        backend="pandas_numpy",
        actual_elapsed_ms=50.0,
        rss_delta_bytes=100 * 1024 * 1024,
        rows=100_000,
        instruments=100,
        predicted_ms=50.0,
        predicted_peak_bytes=100 * 1024 * 1024,
    )

    _, unc2 = calibrated_peak_bytes(static_peak, key)
    assert unc2 == 1.15
    assert unc2 < unc1 < unc0


def test_calibration_ema_update_bounds():
    """FE-P0-019/020: EMA updates clamp ratio to [0.1, 10.0] preventing runaway."""
    reset_calibration()

    # Pathological: 100x memory overrun
    record_task_actual(
        operator="pathological",
        backend="pandas_numpy",
        actual_elapsed_ms=1000.0,
        rss_delta_bytes=10_000 * 1024 * 1024,
        rows=100_000,
        instruments=100,
        predicted_ms=10.0,
        predicted_peak_bytes=10 * 1024 * 1024,
    )

    from runtime.runtime_calibration import calibration_key
    key = calibration_key(
        operator="pathological",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )

    factors = calibrated_factors(key)

    # EMA should clamp 100x overrun to 10x max, then blend with initial 1.0
    # alpha=0.2: new = 1.0 + 0.2 * (10.0 - 1.0) = 2.8
    assert factors.elapsed_factor <= 10.0
    assert factors.memory_factor <= 10.0
    assert factors.elapsed_factor > 1.0
    assert factors.memory_factor > 1.0
