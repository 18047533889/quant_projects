# -*- coding: utf-8 -*-
"""Tests for fiscal TRUE_GAP batch 1 operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.fundamental.fiscal_batch1 import (
    pd_fiscal_acceleration,
    pd_fiscal_pct_change,
    pd_fiscal_rolling_std,
    pd_fiscal_accrual_quality,
    pd_fiscal_direction_consistency,
)


@pytest.fixture
def simple_fiscal_panel():
    """Simple fiscal panel for testing."""
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    # Two instruments with quarterly fiscal events
    values = pd.DataFrame(
        {
            "A": [np.nan, 100, np.nan, np.nan, 110, np.nan, np.nan, 125, np.nan, 145],
            "B": [np.nan, 50, np.nan, np.nan, 48, np.nan, np.nan, 52, np.nan, 55],
        },
        index=index,
    )
    period_ids = pd.DataFrame(
        {
            "A": [None, "2023Q1", None, None, "2023Q2", None, None, "2023Q3", None, "2023Q4"],
            "B": [None, "2023Q1", None, None, "2023Q2", None, None, "2023Q3", None, "2023Q4"],
        },
        index=index,
    )
    return values, period_ids


@pytest.fixture
def consecutive_fiscal_panel():
    """Consecutive fiscal events for acceleration testing."""
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    # Values: 10, 12, 15, 19, 24, 30 (increasing acceleration)
    values = pd.DataFrame(
        {"A": [10, np.nan, 12, np.nan, 15, np.nan, 19, np.nan, 24, np.nan, 30, np.nan]},
        index=index,
    )
    period_ids = pd.DataFrame(
        {"A": ["2023Q1", None, "2023Q2", None, "2023Q3", None, "2023Q4", None, "2024Q1", None, "2024Q2", None]},
        index=index,
    )
    return values, period_ids


def test_fiscal_acceleration_basic(consecutive_fiscal_panel):
    """Test fiscal acceleration computation."""
    values, period_ids = consecutive_fiscal_panel
    result = pd_fiscal_acceleration(values, period_ids, lag=1, periods=8, min_periods=3)

    # At row 4 (2023Q3=15): acceleration = 15 - 2*12 + 10 = 1
    # At row 6 (2023Q4=19): acceleration = 19 - 2*15 + 12 = 1
    # At row 8 (2024Q1=24): acceleration = 24 - 2*19 + 15 = 1
    # At row 10 (2024Q2=30): acceleration = 30 - 2*24 + 19 = 1

    assert result.shape == values.shape
    assert np.isnan(result.iloc[0, 0])  # First event, no history
    assert np.isnan(result.iloc[2, 0])  # Second event, need 3
    assert np.isclose(result.iloc[4, 0], 1.0)  # Third event
    assert np.isclose(result.iloc[6, 0], 1.0)
    assert np.isclose(result.iloc[8, 0], 1.0)
    assert np.isclose(result.iloc[10, 0], 1.0)


def test_fiscal_acceleration_lag2(consecutive_fiscal_panel):
    """Test fiscal acceleration with lag=2."""
    values, period_ids = consecutive_fiscal_panel
    # Extend to have more events
    index_ext = pd.date_range("2024-01-01", periods=16, freq="D")
    values_ext = pd.DataFrame(
        {"A": [10, np.nan, 12, np.nan, 15, np.nan, 19, np.nan, 24, np.nan, 30, np.nan, 37, np.nan, 45, np.nan]},
        index=index_ext,
    )
    period_ids_ext = pd.DataFrame(
        {"A": ["2023Q1", None, "2023Q2", None, "2023Q3", None, "2023Q4", None,
               "2024Q1", None, "2024Q2", None, "2024Q3", None, "2024Q4", None]},
        index=index_ext,
    )

    result = pd_fiscal_acceleration(values_ext, period_ids_ext, lag=2, periods=10, min_periods=5)

    # At row 8 (2024Q1=24): acceleration = 24 - 2*15 + 10 = 4
    assert np.isclose(result.iloc[8, 0], 4.0)


def test_fiscal_pct_change_basic(simple_fiscal_panel):
    """Test fiscal percentage change."""
    values, period_ids = simple_fiscal_panel
    result = pd_fiscal_pct_change(values, period_ids, lag=1, periods=8, min_periods=2)

    # A at row 4: (110 - 100) / 100 = 0.1
    # A at row 7: (125 - 110) / 110 = 0.136...
    # B at row 4: (48 - 50) / 50 = -0.04

    assert result.shape == values.shape
    assert np.isnan(result.iloc[1, 0])  # First event, no history
    assert np.isclose(result.iloc[4, 0], 0.1, atol=1e-6)
    assert np.isclose(result.iloc[7, 0], 15.0 / 110.0, atol=1e-6)
    assert np.isclose(result.iloc[4, 1], -0.04, atol=1e-6)


def test_fiscal_pct_change_near_zero():
    """Test fiscal pct_change handles near-zero denominator."""
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    values = pd.DataFrame({"A": [0.0, np.nan, 10.0, np.nan, 20.0, np.nan]}, index=index)
    period_ids = pd.DataFrame({"A": ["2023Q1", None, "2023Q2", None, "2023Q3", None]}, index=index)

    result = pd_fiscal_pct_change(values, period_ids, lag=1)

    # Row 2: (10 - 0) / |0| -> should be NaN (denominator near zero)
    assert np.isnan(result.iloc[2, 0])
    # Row 4: (20 - 10) / |10| = 1.0
    assert np.isclose(result.iloc[4, 0], 1.0)


def test_fiscal_rolling_std_basic(simple_fiscal_panel):
    """Test fiscal rolling standard deviation."""
    values, period_ids = simple_fiscal_panel
    result = pd_fiscal_rolling_std(values, period_ids, periods=4, min_periods=3)

    # A values: 100, 110, 125, 145
    # At row 7 (3rd event): std([100, 110, 125])
    expected_std_3 = np.std([100, 110, 125], ddof=1)
    assert np.isclose(result.iloc[7, 0], expected_std_3, atol=1e-6)

    # At row 9 (4th event): std([100, 110, 125, 145])
    expected_std_4 = np.std([100, 110, 125, 145], ddof=1)
    assert np.isclose(result.iloc[9, 0], expected_std_4, atol=1e-6)


def test_fiscal_rolling_std_ddof():
    """Test fiscal rolling std with different ddof."""
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    values = pd.DataFrame({"A": [10, np.nan, 20, np.nan, 30, np.nan, 40, np.nan]}, index=index)
    period_ids = pd.DataFrame({"A": ["Q1", None, "Q2", None, "Q3", None, "Q4", None]}, index=index)

    result_ddof1 = pd_fiscal_rolling_std(values, period_ids, periods=4, min_periods=3, ddof=1)
    result_ddof0 = pd_fiscal_rolling_std(values, period_ids, periods=4, min_periods=3, ddof=0)

    # At row 4 (3rd event): values [10, 20, 30]
    expected_ddof1 = np.std([10, 20, 30], ddof=1)
    expected_ddof0 = np.std([10, 20, 30], ddof=0)

    assert np.isclose(result_ddof1.iloc[4, 0], expected_ddof1)
    assert np.isclose(result_ddof0.iloc[4, 0], expected_ddof0)
    assert result_ddof1.iloc[4, 0] > result_ddof0.iloc[4, 0]  # ddof=1 gives larger std


def test_fiscal_accrual_quality_basic():
    """Test fiscal accrual quality (negative std)."""
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    # Low volatility accruals
    values_stable = pd.DataFrame(
        {"A": [0.05, np.nan, 0.06, np.nan, 0.05, np.nan, 0.06, np.nan, 0.05, np.nan]},
        index=index,
    )
    # High volatility accruals
    values_volatile = pd.DataFrame(
        {"B": [0.10, np.nan, -0.05, np.nan, 0.15, np.nan, -0.10, np.nan, 0.20, np.nan]},
        index=index,
    )

    period_ids = pd.DataFrame(
        {
            "A": ["2023Q1", None, "2023Q2", None, "2023Q3", None, "2023Q4", None, "2024Q1", None],
            "B": ["2023Q1", None, "2023Q2", None, "2023Q3", None, "2023Q4", None, "2024Q1", None],
        },
        index=index,
    )

    result_stable = pd_fiscal_accrual_quality(values_stable, period_ids[["A"]], periods=8, min_periods=4)
    result_volatile = pd_fiscal_accrual_quality(values_volatile, period_ids[["B"]], periods=8, min_periods=4)

    # Both should be negative (it's -std)
    assert result_stable.iloc[-2, 0] < 0
    assert result_volatile.iloc[-2, 0] < 0

    # Stable should have better quality (closer to 0)
    assert result_stable.iloc[-2, 0] > result_volatile.iloc[-2, 0]


def test_fiscal_accrual_quality_minimum_periods():
    """Test fiscal accrual quality respects min_periods."""
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    values = pd.DataFrame({"A": [0.05, np.nan, 0.06, np.nan, 0.07, np.nan, 0.08, np.nan]}, index=index)
    period_ids = pd.DataFrame({"A": ["Q1", None, "Q2", None, "Q3", None, "Q4", None]}, index=index)

    result = pd_fiscal_accrual_quality(values, period_ids, periods=8, min_periods=4)

    # Rows 0, 2, 4 have < 4 events
    assert np.isnan(result.iloc[0, 0])
    assert np.isnan(result.iloc[2, 0])
    assert np.isnan(result.iloc[4, 0])

    # Row 6 has 4 events: should compute
    assert np.isfinite(result.iloc[6, 0])


def test_fiscal_direction_consistency_basic():
    """Test fiscal direction consistency."""
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    # Consistently increasing: 10, 12, 15, 19, 24, 30
    values_up = pd.DataFrame(
        {"A": [10, np.nan, 12, np.nan, 15, np.nan, 19, np.nan, 24, np.nan, 30, np.nan]},
        index=index,
    )
    # Alternating: 10, 15, 12, 18, 14, 20
    values_alt = pd.DataFrame(
        {"B": [10, np.nan, 15, np.nan, 12, np.nan, 18, np.nan, 14, np.nan, 20, np.nan]},
        index=index,
    )

    period_ids = pd.DataFrame(
        {
            "A": ["Q1", None, "Q2", None, "Q3", None, "Q4", None, "Q5", None, "Q6", None],
            "B": ["Q1", None, "Q2", None, "Q3", None, "Q4", None, "Q5", None, "Q6", None],
        },
        index=index,
    )

    result_up = pd_fiscal_direction_consistency(values_up, period_ids[["A"]], periods=8, min_periods=3)
    result_alt = pd_fiscal_direction_consistency(values_alt, period_ids[["B"]], periods=8, min_periods=3)

    # Consistently increasing should have consistency close to 1.0
    # At row 10 (6th event), all 5 changes are positive
    assert np.isclose(result_up.iloc[10, 0], 1.0)

    # Alternating should have lower consistency
    # At row 10 (6th event): +5, -3, +6, -4, +6 -> last is +, so count +5,+6,+6 = 3 out of 5 = 0.6
    assert 0.4 <= result_alt.iloc[10, 0] <= 0.8


def test_fiscal_direction_consistency_minimum_changes():
    """Test fiscal direction consistency requires sufficient changes."""
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    values = pd.DataFrame({"A": [10, np.nan, 12, np.nan, 15, np.nan]}, index=index)
    period_ids = pd.DataFrame({"A": ["Q1", None, "Q2", None, "Q3", None]}, index=index)

    result = pd_fiscal_direction_consistency(values, period_ids, periods=8, min_periods=3)

    # Row 0: 1 event -> NaN
    assert np.isnan(result.iloc[0, 0])
    # Row 2: 2 events -> 1 change, but min_periods=3 -> NaN
    assert np.isnan(result.iloc[2, 0])
    # Row 4: 3 events -> 2 changes, should compute
    assert np.isfinite(result.iloc[4, 0])


def test_fiscal_direction_consistency_zero_changes():
    """Test fiscal direction consistency filters out zero changes."""
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    # Values with zero change: 10, 10, 15, 15
    values = pd.DataFrame({"A": [10, np.nan, 10, np.nan, 15, np.nan, 15, np.nan]}, index=index)
    period_ids = pd.DataFrame({"A": ["Q1", None, "Q2", None, "Q3", None, "Q4", None]}, index=index)

    result = pd_fiscal_direction_consistency(values, period_ids, periods=8, min_periods=3)

    # At row 6 (4th event): changes are 0, +5, 0 -> only +5 is non-zero
    # Need at least 2 non-zero changes for min_periods=3, but we only have 1
    assert np.isnan(result.iloc[6, 0])


def test_revision_policy_latest_vs_first():
    """Test revision policy handling."""
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    # Revision at row 2: Q1 is revised from 100 to 105
    values = pd.DataFrame(
        {"A": [100, np.nan, 105, np.nan, 110, np.nan, 115, np.nan]},
        index=index,
    )
    period_ids = pd.DataFrame(
        {"A": ["2023Q1", None, "2023Q1", None, "2023Q2", None, "2023Q3", None]},
        index=index,
    )

    result_latest = pd_fiscal_pct_change(values, period_ids, lag=1, revision_policy="latest_available")
    result_first = pd_fiscal_pct_change(values, period_ids, lag=1, revision_policy="first_available")

    # At row 4 (2023Q2):
    # latest_available: (110 - 105) / 105
    # first_available: (110 - 100) / 100
    pct_latest = (110 - 105) / 105
    pct_first = (110 - 100) / 100

    assert np.isclose(result_latest.iloc[4, 0], pct_latest, atol=1e-6)
    assert np.isclose(result_first.iloc[4, 0], pct_first, atol=1e-6)
    assert not np.isclose(pct_latest, pct_first)


def test_require_consecutive_flag():
    """Test require_consecutive flag."""
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    # Non-consecutive quarters: Q1, Q3, Q4 (missing Q2)
    values = pd.DataFrame({"A": [100, np.nan, np.nan, np.nan, 110, np.nan, np.nan, 120, np.nan, 130]}, index=index)
    period_ids = pd.DataFrame(
        {"A": ["2023Q1", None, None, None, "2023Q3", None, None, "2023Q4", None, "2024Q1"]},
        index=index,
    )

    result_consecutive = pd_fiscal_pct_change(values, period_ids, lag=1, require_consecutive=True)
    result_non_consecutive = pd_fiscal_pct_change(values, period_ids, lag=1, require_consecutive=False)

    # With require_consecutive=True, at row 4 (Q3), history only includes Q3 (breaks at gap)
    # So no prior quarter for comparison -> NaN
    assert np.isnan(result_consecutive.iloc[4, 0])

    # With require_consecutive=False, at row 4 (Q3), can use Q1 -> (110-100)/100 but lag=1 means Q3-Q2
    # Q2 is missing, so still NaN
    assert np.isnan(result_non_consecutive.iloc[4, 0])

    # At row 7 (Q4): consecutive can use Q3, non-consecutive can also use Q3
    assert np.isclose(result_consecutive.iloc[7, 0], (120 - 110) / 110, atol=1e-6)
    assert np.isclose(result_non_consecutive.iloc[7, 0], (120 - 110) / 110, atol=1e-6)


def test_parameter_validation():
    """Test parameter validation."""
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    values = pd.DataFrame({"A": [10, 20, 30, 40]}, index=index)
    period_ids = pd.DataFrame({"A": ["Q1", "Q2", "Q3", "Q4"]}, index=index)

    # Invalid lag (non-positive)
    with pytest.raises(ValueError):
        pd_fiscal_acceleration(values, period_ids, lag=0)

    # Invalid min_periods for acceleration
    with pytest.raises(ValueError):
        pd_fiscal_acceleration(values, period_ids, lag=2, min_periods=4)  # Need at least 5 for lag=2

    # Invalid ddof
    with pytest.raises(ValueError):
        pd_fiscal_rolling_std(values, period_ids, ddof=-1)

    # min_periods too small for ddof
    with pytest.raises(ValueError):
        pd_fiscal_rolling_std(values, period_ids, ddof=1, min_periods=1)

    # Invalid revision_policy
    with pytest.raises(ValueError):
        pd_fiscal_pct_change(values, period_ids, revision_policy="invalid_policy")


def test_empty_and_all_nan():
    """Test handling of empty and all-NaN inputs."""
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    values_empty = pd.DataFrame({"A": [np.nan, np.nan, np.nan, np.nan]}, index=index)
    period_ids = pd.DataFrame({"A": ["Q1", "Q2", "Q3", "Q4"]}, index=index)

    result = pd_fiscal_pct_change(values_empty, period_ids)

    assert result.shape == values_empty.shape
    assert result.isna().all().all()


def test_multiple_instruments():
    """Test computation across multiple instruments."""
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    values = pd.DataFrame(
        {
            "A": [100, np.nan, 110, np.nan, 120, np.nan, 130, np.nan],
            "B": [50, np.nan, 55, np.nan, 60, np.nan, 65, np.nan],
            "C": [200, np.nan, 190, np.nan, 180, np.nan, 170, np.nan],
        },
        index=index,
    )
    period_ids = pd.DataFrame(
        {
            "A": ["Q1", None, "Q2", None, "Q3", None, "Q4", None],
            "B": ["Q1", None, "Q2", None, "Q3", None, "Q4", None],
            "C": ["Q1", None, "Q2", None, "Q3", None, "Q4", None],
        },
        index=index,
    )

    result = pd_fiscal_pct_change(values, period_ids, lag=1)

    # A at row 2: (110-100)/100 = 0.1
    # B at row 2: (55-50)/50 = 0.1
    # C at row 2: (190-200)/200 = -0.05

    assert np.isclose(result.iloc[2, 0], 0.1)
    assert np.isclose(result.iloc[2, 1], 0.1)
    assert np.isclose(result.iloc[2, 2], -0.05)


def test_polars_backend_availability():
    """Test that Polars backend is registered when available."""
    try:
        import polars as pl
        from cleaned_operators.registry import OperatorRegistry

        # Check that polars backend exists
        backends = OperatorRegistry.list_backends("fiscal_acceleration")
        assert "pandas_numpy" in backends
        assert "polars" in backends

    except ImportError:
        pytest.skip("Polars not available")


def test_operator_registration():
    """Test that operators are registered correctly."""
    from cleaned_operators.registry import OperatorRegistry

    operators = [
        "fiscal_acceleration",
        "fiscal_pct_change",
        "fiscal_rolling_std",
        "fiscal_accrual_quality",
        "fiscal_direction_consistency",
    ]

    for op_name in operators:
        assert op_name in OperatorRegistry._operators
        op = OperatorRegistry._operators[op_name]
        assert op["category"] == "fundamental_period"
        assert "fundamental" in op.get("tags", []) or op.get("business_category") == "fundamental"


def test_extended_surface_registration():
    """Test that operators are on the extended surface."""
    from cleaned_operators import operator_surface

    extended = operator_surface.extended_only_canonicals()

    operators = [
        "fiscal_acceleration",
        "fiscal_pct_change",
        "fiscal_rolling_std",
        "fiscal_accrual_quality",
        "fiscal_direction_consistency",
    ]

    for op_name in operators:
        assert op_name in extended


def test_explicit_policies():
    """Test that explicit policies are declared."""
    from cleaned_operators.fundamental.fiscal_batch1 import _EXPLICIT_POLICIES

    operators = [
        "fiscal_acceleration",
        "fiscal_pct_change",
        "fiscal_rolling_std",
        "fiscal_accrual_quality",
        "fiscal_direction_consistency",
    ]

    for op_name in operators:
        assert op_name in _EXPLICIT_POLICIES
        policy = _EXPLICIT_POLICIES[op_name]
        assert policy["scope"] == "fundamental_period"
        assert policy["pit_safe"] is True
        assert "min_periods" in policy
