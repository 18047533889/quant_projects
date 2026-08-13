"""
Test suite for test helper assertions.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from integration_tests.fixtures import (
    assert_panel_shape,
    assert_factor_properties,
    assert_timing_consistency,
    assert_numeric_close,
    assert_no_lookahead,
    compare_evaluation_results,
    assert_correlation_structure,
    assert_cross_sectional_neutrality,
    assert_panel_aligned,
    assert_monotonic_increasing,
    assert_stationary,
    summarize_panel,
)


def test_assert_panel_shape_pass():
    """Test panel shape assertion passes with correct shape."""
    data = np.random.randn(100, 50, 3)
    assert_panel_shape(data, (100, 50, 3))


def test_assert_panel_shape_fail():
    """Test panel shape assertion fails with wrong shape."""
    data = np.random.randn(100, 50, 3)

    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_panel_shape(data, (100, 50, 5))


def test_assert_factor_properties_finite():
    """Test finite value checking."""
    data = np.random.randn(50, 30)

    # Should pass
    assert_factor_properties(data, check_finite=True)

    # Add infinities
    data[0, 0] = np.inf

    with pytest.raises(AssertionError, match="infinite values"):
        assert_factor_properties(data, check_finite=True)


def test_assert_factor_properties_range():
    """Test range checking."""
    data = np.random.randn(50, 30) * 0.5  # Keep in [-2, 2] roughly

    # Should pass
    assert_factor_properties(data, check_range=(-5.0, 5.0))

    # Add out of range value
    data[0, 0] = 10.0

    with pytest.raises(AssertionError, match="outside expected range"):
        assert_factor_properties(data, check_range=(-5.0, 5.0))


def test_assert_factor_properties_missing_rate():
    """Test missing rate checking."""
    data = np.random.randn(100, 50)

    # Add 5% missing
    mask = np.random.rand(100, 50) < 0.05
    data[mask] = np.nan

    # Should pass with 10% threshold
    assert_factor_properties(data, max_missing_rate=0.1)

    # Should fail with 2% threshold
    with pytest.raises(AssertionError, match="missing rate"):
        assert_factor_properties(data, max_missing_rate=0.02)


def test_assert_timing_consistency_pass():
    """Test timing consistency with valid timing."""
    decision_time = ("2024-01-01", "2024-01-02", "2024-01-03")
    label_start_time = ("2024-01-02", "2024-01-03", "2024-01-04")
    label_end_time = ("2024-01-03", "2024-01-04", "2024-01-05")

    assert_timing_consistency(
        decision_time,
        label_start_time,
        label_end_time,
        min_horizon=1,
    )


def test_assert_timing_consistency_wrong_length():
    """Test timing consistency fails with mismatched lengths."""
    decision_time = ("2024-01-01", "2024-01-02")
    label_start_time = ("2024-01-02", "2024-01-03", "2024-01-04")
    label_end_time = ("2024-01-03", "2024-01-04")

    with pytest.raises(AssertionError, match="length"):
        assert_timing_consistency(decision_time, label_start_time, label_end_time)


def test_assert_timing_consistency_wrong_order():
    """Test timing consistency fails with wrong temporal order."""
    decision_time = ("2024-01-03",)  # After label start
    label_start_time = ("2024-01-02",)
    label_end_time = ("2024-01-04",)

    with pytest.raises(AssertionError, match="decision_time .* is after"):
        assert_timing_consistency(decision_time, label_start_time, label_end_time)


def test_assert_timing_consistency_label_order():
    """Test label start/end ordering."""
    decision_time = ("2024-01-01",)
    label_start_time = ("2024-01-04",)  # After label end
    label_end_time = ("2024-01-03",)

    with pytest.raises(AssertionError, match="label_start_time .* is after"):
        assert_timing_consistency(decision_time, label_start_time, label_end_time)


def test_assert_numeric_close_pass():
    """Test numeric comparison passes for close values."""
    assert_numeric_close(1.0000001, 1.0, rtol=1e-5, atol=1e-8)


def test_assert_numeric_close_fail():
    """Test numeric comparison fails for different values."""
    with pytest.raises(AssertionError, match="mismatch"):
        assert_numeric_close(1.1, 1.0, rtol=1e-5, atol=1e-8)


def test_assert_numeric_close_nan():
    """Test numeric comparison handles NaN."""
    assert_numeric_close(np.nan, np.nan)


def test_assert_no_lookahead_pass():
    """Test no lookahead check passes with proper timing."""
    T, N = 10, 5
    factor_values = np.random.randn(T, N, 2)
    label_values = np.random.randn(T, N)

    base_date = datetime(2024, 1, 1)
    decision_time = tuple((base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(T))
    label_start_time = tuple((base_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(T))

    # Should not raise (may warn)
    assert_no_lookahead(
        factor_values,
        label_values,
        decision_time,
        label_start_time,
        min_lag=1,
    )


def test_assert_no_lookahead_shape_mismatch():
    """Test no lookahead fails with shape mismatch."""
    factor_values = np.random.randn(10, 5, 2)
    label_values = np.random.randn(8, 5)  # Wrong T
    decision_time = tuple(f"2024-01-{i:02d}" for i in range(1, 11))
    label_start_time = tuple(f"2024-01-{i:02d}" for i in range(2, 12))

    with pytest.raises(AssertionError, match="label_values time dimension"):
        assert_no_lookahead(factor_values, label_values, decision_time, label_start_time)


def test_compare_evaluation_results_match():
    """Test evaluation result comparison with matching results."""
    result1 = {
        "pearson_ic": {"value": 0.05, "std_error": 0.01},
        "rank_ic": {"value": 0.04, "std_error": 0.01},
    }
    result2 = {
        "pearson_ic": {"value": 0.050001, "std_error": 0.01},
        "rank_ic": {"value": 0.040001, "std_error": 0.01},
    }

    comparison = compare_evaluation_results(result1, result2, rtol=1e-3)

    assert comparison["pearson_ic"] is True
    assert comparison["rank_ic"] is True


def test_compare_evaluation_results_mismatch():
    """Test evaluation result comparison with different results."""
    result1 = {"ic": {"value": 0.05}}
    result2 = {"ic": {"value": 0.10}}

    comparison = compare_evaluation_results(result1, result2, rtol=1e-5)

    assert comparison["ic"] is False


def test_compare_evaluation_results_missing_metric():
    """Test comparison with missing metrics."""
    result1 = {"ic": 0.05, "sharpe": 1.5}
    result2 = {"ic": 0.05}

    comparison = compare_evaluation_results(
        result1, result2,
        metric_names=["ic", "sharpe"],
    )

    assert comparison["ic"] is True
    assert comparison["sharpe"] is False


def test_assert_correlation_structure_pass():
    """Test correlation structure assertion passes."""
    # Generate data with known correlation
    T, N = 500, 100
    F = 3

    # Create correlated factors
    base = np.random.randn(T, N)
    factor_panel = np.zeros((T, N, F))
    factor_panel[:, :, 0] = base
    factor_panel[:, :, 1] = 0.6 * base + 0.8 * np.random.randn(T, N)
    factor_panel[:, :, 2] = 0.3 * base + 0.95 * np.random.randn(T, N)

    expected_corr = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.0],
        [0.3, 0.0, 1.0],
    ])

    # Should pass with reasonable tolerance
    assert_correlation_structure(factor_panel, expected_corr, rtol=0.2)


def test_assert_correlation_structure_fail():
    """Test correlation structure assertion fails with wrong structure."""
    factor_panel = np.random.randn(200, 50, 2)

    expected_corr = np.array([
        [1.0, 0.9],
        [0.9, 1.0],
    ])

    with pytest.raises(AssertionError, match="Correlation structure mismatch"):
        assert_correlation_structure(factor_panel, expected_corr, rtol=0.1)


def test_assert_cross_sectional_neutrality_pass():
    """Test neutrality assertion with neutral factor."""
    T, N, K = 100, 50, 2

    exposure_matrix = np.random.randn(T, N, K)

    # Create factor orthogonal to exposures
    factor_values = np.random.randn(T, N)

    # Should pass (random factor has low correlation, but use higher tolerance)
    assert_cross_sectional_neutrality(factor_values, exposure_matrix, rtol=0.5)


def test_assert_cross_sectional_neutrality_fail():
    """Test neutrality assertion fails with correlated factor."""
    T, N, K = 100, 50, 1

    exposure_matrix = np.random.randn(T, N, K)

    # Create factor highly correlated with exposure
    factor_values = exposure_matrix[:, :, 0] + 0.1 * np.random.randn(T, N)

    with pytest.raises(AssertionError, match="not neutral"):
        assert_cross_sectional_neutrality(factor_values, exposure_matrix, rtol=0.1)


def test_assert_panel_aligned_pass():
    """Test panel alignment assertion passes."""
    panel1 = np.random.randn(100, 50, 3)
    panel2 = np.random.randn(100, 50, 5)

    # Aligned on time (axis 0)
    assert_panel_aligned(panel1, panel2, axis=0)

    # Aligned on asset (axis 1)
    assert_panel_aligned(panel1, panel2, axis=1)


def test_assert_panel_aligned_fail():
    """Test panel alignment assertion fails."""
    panel1 = np.random.randn(100, 50, 3)
    panel2 = np.random.randn(80, 50, 3)

    with pytest.raises(AssertionError, match="not aligned"):
        assert_panel_aligned(panel1, panel2, axis=0)


def test_assert_monotonic_increasing_pass():
    """Test monotonic increasing assertion passes."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert_monotonic_increasing(values)


def test_assert_monotonic_increasing_fail():
    """Test monotonic increasing assertion fails."""
    values = np.array([1.0, 3.0, 2.0, 4.0])

    with pytest.raises(AssertionError, match="not monotonic increasing"):
        assert_monotonic_increasing(values)


def test_assert_monotonic_increasing_strict():
    """Test strict monotonic increasing."""
    values = np.array([1.0, 2.0, 2.0, 3.0])  # Has equal consecutive values

    # Non-strict should pass
    assert_monotonic_increasing(values, strict=False)

    # Strict should fail
    with pytest.raises(AssertionError, match="not strictly monotonic"):
        assert_monotonic_increasing(values, strict=True)


def test_assert_stationary_pass():
    """Test stationarity check passes for stationary series."""
    # White noise is stationary
    values = np.random.randn(500)
    assert_stationary(values, max_autocorr=0.95)


def test_assert_stationary_fail():
    """Test stationarity check fails for non-stationary series."""
    # Random walk is non-stationary
    values = np.cumsum(np.random.randn(500))

    with pytest.raises(AssertionError, match="not stationary"):
        assert_stationary(values, max_autocorr=0.5)


def test_assert_stationary_2d():
    """Test stationarity check with 2D input."""
    # Multiple stationary series
    values = np.random.randn(500, 3)
    assert_stationary(values, max_autocorr=0.95)


def test_summarize_panel():
    """Test panel summary generation."""
    data = np.random.randn(100, 50, 3)

    summary = summarize_panel(data, name="test_panel")

    assert summary["name"] == "test_panel"
    assert summary["shape"] == (100, 50, 3)
    assert "mean" in summary
    assert "std" in summary
    assert "min" in summary
    assert "max" in summary
    assert "missing_rate" in summary


def test_summarize_panel_all_missing():
    """Test summary with all missing data."""
    data = np.full((10, 5), np.nan)

    summary = summarize_panel(data)

    assert summary["all_missing"] is True
    assert "mean" not in summary


def test_summarize_panel_with_missing():
    """Test summary with some missing data."""
    data = np.random.randn(100, 50)
    mask = np.random.rand(100, 50) < 0.2
    data[mask] = np.nan

    summary = summarize_panel(data)

    assert summary["all_missing"] is False
    assert 0.15 < summary["missing_rate"] < 0.25
    assert "mean" in summary


def test_summarize_panel_with_inf():
    """Test summary with infinite values."""
    data = np.random.randn(100, 50)
    data[0, 0] = np.inf
    data[1, 0] = -np.inf

    summary = summarize_panel(data)

    assert summary["n_inf"] == 2
