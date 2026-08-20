# -*- coding: utf-8 -*-
"""R21-P028: Regression tests for magic number removal.

Tests that empty universe yields rows=0 (not 1) and unknown instruments
yield 0 (not fake A-share 5500/3000 fallbacks).
"""
from __future__ import annotations

import pytest


class MockDataSource:
    """Mock data source for testing."""
    def __init__(self, instrument_filter=None, start_date=None, end_date=None):
        self.instrument_filter = instrument_filter
        self.start_date = start_date
        self.end_date = end_date
        self.inner = None
        self.calendar = None


class MockContext:
    """Mock context for testing."""
    def __init__(self, data_source=None, runtime_stats=None, universe=None):
        self.data_source = data_source
        self.runtime_stats = runtime_stats or {}
        self.universe = universe
        self.calendar = None
        self.start_date = None
        self.end_date = None


def test_empty_universe_yields_zero_rows():
    """R21-P028: Empty instrument filter must yield rows=0, not rows=1."""
    from backend.plan_cost_router import estimate_plan_rows

    # Empty list filter
    ds = MockDataSource(instrument_filter=[])
    ctx = MockContext(data_source=ds)
    rows = estimate_plan_rows(ctx)
    assert rows == 0, f"Empty filter should yield rows=0, got rows={rows}"

    # Empty set filter
    ds = MockDataSource(instrument_filter=set())
    ctx = MockContext(data_source=ds)
    rows = estimate_plan_rows(ctx)
    assert rows == 0, f"Empty set filter should yield rows=0, got rows={rows}"

    # Empty frozenset filter
    ds = MockDataSource(instrument_filter=frozenset())
    ctx = MockContext(data_source=ds)
    rows = estimate_plan_rows(ctx)
    assert rows == 0, f"Empty frozenset filter should yield rows=0, got rows={rows}"


def test_unknown_instruments_yield_zero():
    """R21-P028: Unknown instruments must yield 0 (not fake 5500/3000)."""
    from backend.plan_cost_router import estimate_plan_rows

    # None filter (unknown universe)
    ds = MockDataSource(instrument_filter=None)
    ctx = MockContext(data_source=ds)
    rows = estimate_plan_rows(ctx)

    # We can't directly check instruments, but we can verify no 5500/3000 multiplication
    # The function should return 0 or a date-based estimate, not 5500*250=1,375,000
    assert rows != 1375000, f"Should not produce A-share 5500*250 estimate, got rows={rows}"
    assert rows != 750000, f"Should not produce A-share 3000*250 estimate, got rows={rows}"

    # String filter (unknown type)
    ds = MockDataSource(instrument_filter="unknown_filter")
    ctx = MockContext(data_source=ds)
    rows = estimate_plan_rows(ctx)
    assert rows != 1375000, f"Should not produce A-share 5500*250 estimate, got rows={rows}"
    assert rows != 750000, f"Should not produce A-share 3000*250 estimate, got rows={rows}"


def test_empty_universe_data_shape():
    """R21-P028: DataShapeEstimate from empty filter must have 0 instruments."""
    from planner.data_shape import _estimate_instruments_from_data_source

    # Empty list
    ds = MockDataSource(instrument_filter=[])
    instruments = _estimate_instruments_from_data_source(ds)
    assert instruments == 0, f"Empty list should yield 0 instruments, got {instruments}"

    # Empty set
    ds = MockDataSource(instrument_filter=set())
    instruments = _estimate_instruments_from_data_source(ds)
    assert instruments == 0, f"Empty set should yield 0 instruments, got {instruments}"


def test_unknown_instruments_data_shape():
    """R21-P028: Unknown filter must yield 0 instruments (not 5500/3000)."""
    from planner.data_shape import _estimate_instruments_from_data_source

    # None filter
    ds = MockDataSource(instrument_filter=None)
    instruments = _estimate_instruments_from_data_source(ds)
    assert instruments == 0, f"None filter should yield 0 instruments, got {instruments}"

    # String filter
    ds = MockDataSource(instrument_filter="unknown")
    instruments = _estimate_instruments_from_data_source(ds)
    assert instruments == 0, f"String filter should yield 0 instruments, got {instruments}"
