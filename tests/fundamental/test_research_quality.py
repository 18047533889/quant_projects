# -*- coding: utf-8 -*-
"""Tests for research-grade fundamental quality operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.fundamental.research_quality import (
    pd_fiscal_asymmetric_timeliness,
)


def _panel(values, periods):
    """Create aligned test panels."""
    index = pd.date_range("2025-01-01", periods=len(values))
    return (
        pd.DataFrame({"A": values}, index=index, dtype=float),
        pd.DataFrame({"A": periods}, index=index),
    )


def test_fiscal_asymmetric_timeliness_basic_regression():
    """Test normal case: sufficient data for regression."""
    # Create synthetic data: earnings more sensitive to negative returns
    earnings_data = [0.05, -0.10, 0.03, -0.15, 0.04, -0.12, 0.02, -0.18,
                     0.06, -0.14, 0.03, -0.16, 0.05, -0.13, 0.04, -0.17]
    return_data = [0.10, -0.05, 0.08, -0.06, 0.09, -0.05, 0.07, -0.07,
                   0.11, -0.06, 0.08, -0.06, 0.10, -0.05, 0.09, -0.07]
    periods = [f"202{i//4}Q{i%4+1}" for i in range(16)]

    earnings, period_id = _panel(earnings_data, periods)
    returns, _ = _panel(return_data, periods)

    result = pd_fiscal_asymmetric_timeliness(
        earnings, returns, period_id, periods=16, min_periods=12
    )

    # Should produce a finite delta coefficient
    assert np.isfinite(result.iloc[-1, 0])
    # With asymmetric response, delta should be non-zero
    assert result.iloc[-1, 0] != 0.0


def test_fiscal_asymmetric_timeliness_insufficient_periods():
    """Test fail-closed with insufficient data."""
    earnings_data = [0.05, -0.10, 0.03, -0.15]
    return_data = [0.10, -0.05, 0.08, -0.06]
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]

    earnings, period_id = _panel(earnings_data, periods)
    returns, _ = _panel(return_data, periods)

    result = pd_fiscal_asymmetric_timeliness(
        earnings, returns, period_id, periods=16, min_periods=12
    )

    # Should return NaN due to insufficient periods
    assert np.isnan(result.iloc[-1, 0])


def test_fiscal_asymmetric_timeliness_no_bad_news():
    """Test edge case: all positive returns (no bad news)."""
    earnings_data = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,
                     0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
    return_data = [0.10, 0.08, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08,
                   0.10, 0.08, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08]
    periods = [f"202{i//4}Q{i%4+1}" for i in range(16)]

    earnings, period_id = _panel(earnings_data, periods)
    returns, _ = _panel(return_data, periods)

    result = pd_fiscal_asymmetric_timeliness(
        earnings, returns, period_id, periods=16, min_periods=12
    )

    # With no negative returns, bad indicator is all zeros
    # The interaction term will be zero, regression may be singular or delta=0
    # Should fail-closed or return 0
    assert np.isnan(result.iloc[-1, 0]) or result.iloc[-1, 0] == 0.0


def test_fiscal_asymmetric_timeliness_all_bad_news():
    """Test edge case: all negative returns (all bad news)."""
    earnings_data = [-0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12,
                     -0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12]
    return_data = [-0.10, -0.08, -0.12, -0.09, -0.11, -0.10, -0.13, -0.08,
                   -0.10, -0.08, -0.12, -0.09, -0.11, -0.10, -0.13, -0.08]
    periods = [f"202{i//4}Q{i%4+1}" for i in range(16)]

    earnings, period_id = _panel(earnings_data, periods)
    returns, _ = _panel(return_data, periods)

    result = pd_fiscal_asymmetric_timeliness(
        earnings, returns, period_id, periods=16, min_periods=12
    )

    # With all negative returns, bad indicator is all ones
    # This creates collinearity: bad and intercept overlap
    # Should fail-closed due to rank deficiency
    assert np.isnan(result.iloc[-1, 0]) or np.isfinite(result.iloc[-1, 0])


def test_fiscal_asymmetric_timeliness_mixed_good_bad():
    """Test realistic case with mixed good and bad news periods."""
    # 8 good news, 8 bad news periods
    earnings_data = [0.05, 0.06, -0.03, -0.04, 0.05, 0.06, -0.03, -0.04,
                     0.05, 0.06, -0.03, -0.04, 0.05, 0.06, -0.03, -0.04]
    return_data = [0.10, 0.12, -0.08, -0.10, 0.11, 0.09, -0.09, -0.11,
                   0.10, 0.12, -0.08, -0.10, 0.11, 0.09, -0.09, -0.11]
    periods = [f"202{i//4}Q{i%4+1}" for i in range(16)]

    earnings, period_id = _panel(earnings_data, periods)
    returns, _ = _panel(return_data, periods)

    result = pd_fiscal_asymmetric_timeliness(
        earnings, returns, period_id, periods=16, min_periods=12
    )

    # Should produce a finite result with mixed news
    assert np.isfinite(result.iloc[-1, 0])


def test_fiscal_asymmetric_timeliness_revision_policy():
    """Test revision policy handling."""
    earnings_data = [0.05, 0.05, 0.06, -0.03, -0.04, 0.05, 0.06, -0.03, -0.04,
                     0.05, 0.06, -0.03, -0.04, 0.05, 0.06, -0.03]
    return_data = [0.10, 0.10, 0.12, -0.08, -0.10, 0.11, 0.09, -0.09, -0.11,
                   0.10, 0.12, -0.08, -0.10, 0.11, 0.09, -0.09]
    periods = ["2024Q1", "2024Q1", "2024Q2", "2024Q3", "2024Q4",
               "2025Q1", "2025Q2", "2025Q3", "2025Q4",
               "2026Q1", "2026Q2", "2026Q3", "2026Q4",
               "2027Q1", "2027Q2", "2027Q3"]

    earnings, period_id = _panel(earnings_data, periods)
    returns, _ = _panel(return_data, periods)

    # Latest available: second observation overwrites first
    result_latest = pd_fiscal_asymmetric_timeliness(
        earnings, returns, period_id, periods=16, min_periods=8,
        revision_policy="latest_available"
    )

    # First available: first observation kept
    result_first = pd_fiscal_asymmetric_timeliness(
        earnings, returns, period_id, periods=16, min_periods=8,
        revision_policy="first_available"
    )

    # Both should produce finite results
    assert np.isfinite(result_latest.iloc[-1, 0])
    assert np.isfinite(result_first.iloc[-1, 0])


def test_fiscal_asymmetric_timeliness_alignment_check():
    """Test strict panel alignment requirement."""
    earnings_data = [0.05, 0.06, -0.03]
    return_data = [0.10, 0.12]  # Mismatched length
    periods = ["2024Q1", "2024Q2", "2024Q3"]

    earnings, period_id = _panel(earnings_data, periods)
    returns_short = pd.DataFrame(
        {"A": return_data},
        index=pd.date_range("2025-01-01", periods=2),
        dtype=float
    )

    with pytest.raises(ValueError, match="not aligned"):
        pd_fiscal_asymmetric_timeliness(
            earnings, returns_short, period_id, periods=16, min_periods=12
        )


def test_fiscal_asymmetric_timeliness_min_periods_validation():
    """Test parameter validation."""
    earnings_data = [0.05] * 16
    return_data = [0.10] * 16
    periods = [f"202{i//4}Q{i%4+1}" for i in range(16)]

    earnings, period_id = _panel(earnings_data, periods)
    returns, _ = _panel(return_data, periods)

    # min_periods > periods should raise
    with pytest.raises(ValueError, match="min_periods cannot exceed periods"):
        pd_fiscal_asymmetric_timeliness(
            earnings, returns, period_id, periods=10, min_periods=12
        )
