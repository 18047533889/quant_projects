# -*- coding: utf-8 -*-
"""Tests for industry_fiscal_resid operator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.fundamental.transforms_v2 import industry_fiscal_resid


def _panel(values, periods, industries=None):
    """Helper to create test panels."""
    index = pd.date_range("2025-01-01", periods=len(values))
    n_instruments = len(values[0]) if isinstance(values[0], (list, tuple)) else 1

    if n_instruments == 1:
        y = pd.DataFrame({"A": values}, index=index, dtype=float)
        period_id = pd.DataFrame({"A": periods}, index=index)
        if industries is not None:
            industry = pd.DataFrame({"A": industries}, index=index)
        else:
            industry = pd.DataFrame({"A": [1] * len(values)}, index=index)
    else:
        cols = [f"I{i}" for i in range(n_instruments)]
        y = pd.DataFrame({col: [v[i] for v in values] for i, col in enumerate(cols)}, index=index, dtype=float)
        period_id = pd.DataFrame({col: [p[i] if isinstance(p, (list, tuple)) else p for p in periods] for i, col in enumerate(cols)}, index=index)
        if industries is not None:
            industry = pd.DataFrame({col: [ind[i] if isinstance(ind, (list, tuple)) else ind for ind in industries] for i, col in enumerate(cols)}, index=index)
        else:
            industry = pd.DataFrame({col: [1] * len(values) for col in cols}, index=index)

    return y, period_id, industry


def test_basic_industry_grouped_regression():
    """Each industry fits independent regression."""
    # Industry 1: y = 2*x + 1, Industry 2: y = 3*x - 1
    y_vals = [3, 5, 7, 9,   # Industry 1: 2*1+1, 2*2+1, 2*3+1, 2*4+1
               2, 5, 8, 11]  # Industry 2: 3*1-1, 3*2-1, 3*3-1, 3*4-1
    x_vals = [1, 2, 3, 4, 1, 2, 3, 4]
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"] * 2
    industries = [1, 1, 1, 1, 2, 2, 2, 2]

    y, period_id, industry = _panel(y_vals, periods, industries)
    x1, _, _ = _panel(x_vals, periods, industries)

    result = industry_fiscal_resid(
        y, period_id, industry, x1,
        periods=4, min_obs=3, add_intercept=True, require_consecutive=True
    )

    # With perfect fit, residuals should be near zero
    assert result.shape == y.shape
    # First 2 rows have insufficient history
    assert np.isnan(result.iloc[0, 0])
    assert np.isnan(result.iloc[1, 0])
    # Rows 2-3 should have near-zero residuals (industry 1)
    assert abs(result.iloc[2, 0]) < 1e-10
    assert abs(result.iloc[3, 0]) < 1e-10
    # Rows 4-5 insufficient for industry 2
    assert np.isnan(result.iloc[4, 0])
    assert np.isnan(result.iloc[5, 0])
    # Rows 6-7 should have near-zero residuals (industry 2)
    assert abs(result.iloc[6, 0]) < 1e-10
    assert abs(result.iloc[7, 0]) < 1e-10


def test_missing_fiscal_quarter_fails_closed():
    """When require_consecutive=True, skip fails closed."""
    y_vals = [1, 2, 3, 4]
    x_vals = [1, 2, 3, 4]
    # Missing Q2 between Q1 and Q3
    periods = ["2024Q1", "2024Q3", "2024Q4", "2025Q1"]

    y, period_id, industry = _panel(y_vals, periods)
    x1, _, _ = _panel(x_vals, periods)

    result = industry_fiscal_resid(
        y, period_id, industry, x1,
        periods=4, require_consecutive=True
    )

    # All should be NaN due to non-consecutive periods
    assert np.all(np.isnan(result.values))


def test_revision_updates_from_revision_row_only():
    """Revision of fiscal period updates only from revision timestamp forward."""
    # Initial: Q1=10, then revision at row 3 changes Q1 to 15
    y_vals = [10, 20, 30, 15, 40]
    x_vals = [1, 2, 3, 1, 4]
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q1", "2024Q4"]

    y, period_id, industry = _panel(y_vals, periods)
    x1, _, _ = _panel(x_vals, periods)

    result = industry_fiscal_resid(
        y, period_id, industry, x1,
        periods=4, min_obs=3, revision_policy="latest_available"
    )

    # Row 2 (Q3) uses Q1=10
    # Row 4 (Q4) uses Q1=15 (revised)
    assert not np.isnan(result.iloc[2, 0])  # Has 3 points
    assert not np.isnan(result.iloc[4, 0])  # Has 4 points including revision


def test_industry_missing_returns_nan():
    """Instruments with missing industry return NaN."""
    y_vals = [1, 2, 3, 4]
    x_vals = [1, 2, 3, 4]
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    industries = [1, 1, np.nan, np.nan]  # Industry missing for Q3/Q4

    y, period_id, industry = _panel(y_vals, periods, industries)
    x1, _, _ = _panel(x_vals, periods, industries)

    result = industry_fiscal_resid(
        y, period_id, industry, x1,
        periods=4, min_obs=3
    )

    # Rows with missing industry should be NaN
    assert np.isnan(result.iloc[2, 0])
    assert np.isnan(result.iloc[3, 0])


def test_variadic_regressors():
    """Test with multiple regressors x1, x2, x3."""
    # y = 2*x1 + 3*x2 - x3 + 1
    # Make x2 and x3 different from x1 to avoid collinearity
    y_vals = [2*1 + 3*2 - 0.5 + 1,   # 8.5
              2*2 + 3*4 - 1.0 + 1,    # 17
              2*3 + 3*6 - 1.5 + 1,    # 25.5
              2*4 + 3*8 - 2.0 + 1]    # 33
    x1_vals = [1, 2, 3, 4]
    x2_vals = [2, 4, 6, 8]   # x2 = 2*x1
    x3_vals = [0.5, 1.0, 1.5, 2.0]  # x3 = 0.5*x1
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]

    y, period_id, industry = _panel(y_vals, periods)
    x1, _, _ = _panel(x1_vals, periods)
    x2, _, _ = _panel(x2_vals, periods)
    x3, _, _ = _panel(x3_vals, periods)

    result = industry_fiscal_resid(
        y, period_id, industry, x1, x2, x3,
        periods=4, min_obs=4, add_intercept=True
    )

    # With perfect fit, residual should be near zero
    # However, x2 and x3 are perfectly correlated with x1, causing rank deficiency
    # This should return NaN due to rank deficiency
    assert np.isnan(result.iloc[3, 0])


def test_variadic_regressors_independent():
    """Test with multiple independent regressors x1, x2."""
    # y = 2*x1 + 3*x2 + 1, with x1 and x2 truly independent
    y_vals = [2*1 + 3*5 + 1,    # 18
              2*2 + 3*3 + 1,    # 14
              2*3 + 3*8 + 1,    # 31
              2*4 + 3*2 + 1]    # 15
    x1_vals = [1, 2, 3, 4]
    x2_vals = [5, 3, 8, 2]  # Independent pattern
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]

    y, period_id, industry = _panel(y_vals, periods)
    x1, _, _ = _panel(x1_vals, periods)
    x2, _, _ = _panel(x2_vals, periods)

    result = industry_fiscal_resid(
        y, period_id, industry, x1, x2,
        periods=4, min_obs=4, add_intercept=True
    )

    # With perfect fit and independent regressors, residual should be near zero
    assert abs(result.iloc[3, 0]) < 1e-10


def test_no_intercept_mode():
    """Test regression without intercept."""
    # y = 2*x (no intercept)
    y_vals = [2, 4, 6, 8]
    x_vals = [1, 2, 3, 4]
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]

    y, period_id, industry = _panel(y_vals, periods)
    x1, _, _ = _panel(x_vals, periods)

    result = industry_fiscal_resid(
        y, period_id, industry, x1,
        periods=4, min_obs=3, add_intercept=False
    )

    # Perfect fit: residuals near zero
    assert abs(result.iloc[2, 0]) < 1e-10
    assert abs(result.iloc[3, 0]) < 1e-10


def test_insufficient_observations():
    """Returns NaN when fewer than min_obs observations available."""
    y_vals = [1, 2]
    x_vals = [1, 2]
    periods = ["2024Q1", "2024Q2"]

    y, period_id, industry = _panel(y_vals, periods)
    x1, _, _ = _panel(x_vals, periods)

    result = industry_fiscal_resid(
        y, period_id, industry, x1,
        periods=10, min_obs=5
    )

    # Insufficient observations
    assert np.all(np.isnan(result.values))


def test_rank_deficiency_returns_nan():
    """Rank-deficient design matrix returns NaN."""
    # x1 and x2 are perfectly correlated
    y_vals = [1, 2, 3, 4]
    x1_vals = [1, 2, 3, 4]
    x2_vals = [2, 4, 6, 8]  # x2 = 2*x1
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]

    y, period_id, industry = _panel(y_vals, periods)
    x1, _, _ = _panel(x1_vals, periods)
    x2, _, _ = _panel(x2_vals, periods)

    result = industry_fiscal_resid(
        y, period_id, industry, x1, x2,
        periods=4, min_obs=3, add_intercept=True
    )

    # Rank deficiency should produce NaN
    assert np.all(np.isnan(result.values[2:]))


def test_multi_instrument_independent_industries():
    """Multiple instruments with different industries are independent."""
    # Two instruments, different industries
    y_vals = [(3, 5), (5, 8), (7, 11), (9, 14)]  # I1: 2*x+1, I2: 3*x+2
    x_vals = [(1, 1), (2, 2), (3, 3), (4, 4)]
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    industries = [1, 2]  # I1 in industry 1, I2 in industry 2

    index = pd.date_range("2025-01-01", periods=4)
    y = pd.DataFrame({"I1": [3, 5, 7, 9], "I2": [5, 8, 11, 14]}, index=index, dtype=float)
    period_id = pd.DataFrame({"I1": periods, "I2": periods}, index=index)
    industry = pd.DataFrame({"I1": [1]*4, "I2": [2]*4}, index=index)
    x1 = pd.DataFrame({"I1": [1, 2, 3, 4], "I2": [1, 2, 3, 4]}, index=index, dtype=float)

    result = industry_fiscal_resid(
        y, period_id, industry, x1,
        periods=4, min_obs=3, add_intercept=True
    )

    # Each instrument fits independently, residuals near zero
    assert abs(result.iloc[2, 0]) < 1e-10  # I1 at Q3
    assert abs(result.iloc[2, 1]) < 1e-10  # I2 at Q3


def test_first_available_revision_policy():
    """first_available ignores later revisions."""
    y_vals = [10, 20, 30, 15, 40]  # Q1 revised from 10 to 15
    x_vals = [1, 2, 3, 1, 4]
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q1", "2024Q4"]

    y, period_id, industry = _panel(y_vals, periods)
    x1, _, _ = _panel(x_vals, periods)

    result = industry_fiscal_resid(
        y, period_id, industry, x1,
        periods=4, min_obs=3, revision_policy="first_available"
    )

    # Should use Q1=10 throughout (first seen)
    assert not np.isnan(result.iloc[2, 0])
    assert not np.isnan(result.iloc[4, 0])


def test_window_respects_periods_parameter():
    """Regression window only includes last `periods` events."""
    # 8 quarters, but only use last 4
    y_vals = [1, 2, 3, 4, 5, 6, 7, 8]
    x_vals = [1, 2, 3, 4, 5, 6, 7, 8]
    periods = ["2023Q1", "2023Q2", "2023Q3", "2023Q4",
               "2024Q1", "2024Q2", "2024Q3", "2024Q4"]

    y, period_id, industry = _panel(y_vals, periods)
    x1, _, _ = _panel(x_vals, periods)

    result = industry_fiscal_resid(
        y, period_id, industry, x1,
        periods=4, min_obs=3
    )

    # Should have valid results starting from row with 3 observations
    assert not np.isnan(result.iloc[2, 0])
    # Last row uses only Q1-Q4 of 2024 (periods=4)
    assert not np.isnan(result.iloc[7, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
