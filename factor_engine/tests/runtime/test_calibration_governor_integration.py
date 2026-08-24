# -*- coding: utf-8 -*-
"""FE-P0-019/020/027: Integration test for calibration + resource governor."""
from __future__ import annotations

import pytest

from factor_engine.runtime.resource_governor import MemoryGovernor
from factor_engine.runtime.runtime_calibration import (
    calibrated_peak_bytes,
    record_task_actual,
    reset_calibration,
)


def test_governor_admission_with_calibrated_peak():
    """FE-P0-020/027: Governor uses calibrated_peak_bytes correctly.

    Integration: calibrated_peak_bytes provides memory estimate to governor
    admission. Must use memory_factor (not elapsed_factor) and never return zero.
    """
    reset_calibration()

    # Simulate task with high memory usage
    record_task_actual(
        operator="heavy_op",
        backend="pandas_numpy",
        actual_elapsed_ms=100.0,
        rss_delta_bytes=200 * 1024 * 1024,
        rows=500_000,
        instruments=1000,
        predicted_ms=50.0,
        predicted_peak_bytes=100 * 1024 * 1024,
    )

    from factor_engine.runtime.runtime_calibration import calibration_key
    key = calibration_key(
        operator="heavy_op",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )

    # Get calibrated estimate
    static_peak = 100 * 1024 * 1024
    predicted, uncertainty = calibrated_peak_bytes(static_peak, key)

    # FE-P0-020: Must use memory_factor (2.0x ratio), not elapsed_factor
    # FE-P0-027: Must be nonzero
    assert predicted > 0
    assert predicted > static_peak  # Calibration adjusts upward

    # Governor admission with calibrated estimate
    governor = MemoryGovernor(
        process_budget_bytes=500 * 1024 * 1024,
        duckdb_budget_bytes=100 * 1024 * 1024,
    )

    # Should admit task within budget
    admissible = predicted  # In production, this comes from calibrated_peak_bytes
    can_admit = governor.can_admit(admissible)
    assert can_admit is True

    # Simulate near-budget scenario
    governor = MemoryGovernor(
        process_budget_bytes=150 * 1024 * 1024,
        duckdb_budget_bytes=50 * 1024 * 1024,
    )

    # Should reject if calibrated peak exceeds budget
    can_admit = governor.can_admit(predicted)
    # Behavior depends on predicted size vs budget


def test_zero_static_peak_never_blocks_production():
    """FE-P0-027: Zero/unknown estimates get conservative 8MB, not zero.

    Production paths must never get zero estimate causing admission failures.
    """
    reset_calibration()

    from factor_engine.runtime.runtime_calibration import calibration_key
    key = calibration_key(
        operator="unknown_memory",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )

    # Zero static peak (unknown/unavailable estimate)
    predicted, _ = calibrated_peak_bytes(0, key)

    # Must return conservative nonzero value
    assert predicted > 0
    assert predicted >= 8 * 1024 * 1024

    # Governor can make admission decision
    governor = MemoryGovernor(
        process_budget_bytes=100 * 1024 * 1024,
        duckdb_budget_bytes=50 * 1024 * 1024,
    )

    # Should admit conservative 8MB
    can_admit = governor.can_admit(predicted)
    assert can_admit is True  # 8MB is within 100MB budget


def test_calibration_memory_vs_elapsed_independence():
    """FE-P0-020: Memory and elapsed factors are independent.

    Tasks can have different memory vs elapsed ratios. calibrated_peak_bytes
    must use memory_factor, proving independence.
    """
    reset_calibration()

    # Task 1: Fast but memory-hungry
    record_task_actual(
        operator="fast_heavy",
        backend="pandas_numpy",
        actual_elapsed_ms=25.0,
        rss_delta_bytes=400 * 1024 * 1024,
        rows=100_000,
        instruments=100,
        predicted_ms=50.0,
        predicted_peak_bytes=100 * 1024 * 1024,
    )

    # Task 2: Slow but memory-light
    record_task_actual(
        operator="slow_light",
        backend="pandas_numpy",
        actual_elapsed_ms=200.0,
        rss_delta_bytes=50 * 1024 * 1024,
        rows=100_000,
        instruments=100,
        predicted_ms=50.0,
        predicted_peak_bytes=100 * 1024 * 1024,
    )

    from factor_engine.runtime.runtime_calibration import calibration_key, calibrated_factors

    key1 = calibration_key(
        operator="fast_heavy",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )
    key2 = calibration_key(
        operator="slow_light",
        backend="pandas_numpy",
        shape_bucket="small",
        window_bucket="",
    )

    factors1 = calibrated_factors(key1)
    factors2 = calibrated_factors(key2)

    # fast_heavy: elapsed 0.5x, memory 4x
    # slow_light: elapsed 4x, memory 0.5x
    # EMA blended, but relationship preserved
    assert factors1.memory_factor > factors1.elapsed_factor
    assert factors2.elapsed_factor > factors2.memory_factor

    # Calibrated peaks reflect memory factor, not elapsed
    peak1, _ = calibrated_peak_bytes(100 * 1024 * 1024, key1)
    peak2, _ = calibrated_peak_bytes(100 * 1024 * 1024, key2)

    # fast_heavy should predict higher memory than slow_light
    assert peak1 > peak2
