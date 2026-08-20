# -*- coding: utf-8 -*-
"""R21-DATASHAPE-KNOWN-ZERO: Regression tests for rows_known field.

Tests that DataShapeEstimate correctly distinguishes between known-zero
and unknown rows. The optimizer can now use rows_known to decide whether
to skip computation (known empty) or apply conservative estimates (unknown).
"""
from __future__ import annotations

import pytest

from planner.data_shape import (
    DataShapeEstimate,
    estimate_shape_from_context,
    _conservative_default_shape,
)


class TestRowsKnownField:
    """Test rows_known field in DataShapeEstimate."""

    def test_rows_known_true_default(self):
        """DataShapeEstimate defaults rows_known=True."""
        shape = DataShapeEstimate(
            estimated_rows=1000,
            estimated_dates=252,
            estimated_instruments=100,
            estimated_columns=10,
            estimated_bytes=80000,
            average_row_width_bytes=80.0,
            density=0.95,
            frequency="daily",
        )
        assert shape.rows_known is True

    def test_rows_known_false_when_set(self):
        """DataShapeEstimate can be created with rows_known=False."""
        shape = DataShapeEstimate(
            estimated_rows=0,
            estimated_dates=252,
            estimated_instruments=0,
            estimated_columns=10,
            estimated_bytes=0,
            average_row_width_bytes=64.0,
            density=0.95,
            frequency="daily",
            rows_known=False,
        )
        assert shape.rows_known is False

    def test_rows_known_in_to_dict(self):
        """to_dict includes rows_known field."""
        shape = DataShapeEstimate(
            estimated_rows=100,
            estimated_dates=25,
            estimated_instruments=4,
            estimated_columns=5,
            estimated_bytes=4000,
            average_row_width_bytes=40.0,
            density=0.95,
            frequency="daily",
            rows_known=False,
        )
        d = shape.to_dict()
        assert "rows_known" in d
        assert d["rows_known"] is False


class TestConservativeDefaultShape:
    """Test _conservative_default_shape returns rows_known=False."""

    def test_conservative_default_rows_known_false(self):
        """Conservative fallback has rows_known=False."""
        shape = _conservative_default_shape()
        assert shape.rows_known is False
        assert shape.estimated_rows == 0
        assert shape.estimated_instruments == 0


class TestEstimateFromContextWithUnknownUniverse:
    """Test rows_known=False when universe is unknown."""

    def test_unknown_universe_rows_known_false(self):
        """Unknown universe (instruments=0) sets rows_known=False."""
        # Create a mock context with no data_source
        class MockCtx:
            pass

        ctx = MockCtx()
        shape = estimate_shape_from_context(ctx)

        # Conservative default: instruments=0, rows=0, rows_known=False
        assert shape.estimated_instruments == 0
        assert shape.estimated_rows == 0
        assert shape.rows_known is False


class TestKnownZeroVsUnknown:
    """Regression test: known-zero vs unknown distinction."""

    def test_known_zero_not_treated_as_unknown(self):
        """Known zero rows (rows_known=True) should not be treated as unknown."""
        # Simulate known empty dataset
        shape_known_zero = DataShapeEstimate(
            estimated_rows=0,
            estimated_dates=252,
            estimated_instruments=100,
            estimated_columns=10,
            estimated_bytes=0,
            average_row_width_bytes=64.0,
            density=0.95,
            frequency="daily",
            rows_known=True,
        )

        # Simulate unknown dataset
        shape_unknown = DataShapeEstimate(
            estimated_rows=0,
            estimated_dates=0,
            estimated_instruments=0,
            estimated_columns=8,
            estimated_bytes=0,
            average_row_width_bytes=64.0,
            density=0.95,
            frequency="daily",
            rows_known=False,
        )

        # The optimizer can now distinguish these cases
        assert shape_known_zero.rows_known is True
        assert shape_unknown.rows_known is False

        # Both have estimated_rows=0, but different rows_known semantics
        assert shape_known_zero.estimated_rows == 0
        assert shape_unknown.estimated_rows == 0

        # Test to_dict includes rows_known
        d_known = shape_known_zero.to_dict()
        d_unknown = shape_unknown.to_dict()
        assert d_known["rows_known"] is True
        assert d_unknown["rows_known"] is False
