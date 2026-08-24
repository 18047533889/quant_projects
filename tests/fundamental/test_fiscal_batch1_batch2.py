# -*- coding: utf-8 -*-
"""Tests for fiscal_batch1_batch2 operators (10 operators total).

Tests verify:
- FiscalEventView kernel PIT-safe semantics
- Dual backend consistency (pandas_numpy + polars)
- Parameter validation
- Fiscal gap fail-closed behavior
- Edge cases (min_periods, revision_policy, consecutive periods)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def _panel(vals, periods):
    """Create test panel with fiscal period_id."""
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="B")
    return pd.DataFrame({"A": vals}, index=idx), pd.DataFrame({"A": periods}, index=idx)


def _calc(canonical, *args, backend="pandas_numpy", **kwargs):
    """Calculate operator with specified backend."""
    op = OperatorRegistry.get(canonical, backend)
    assert op is not None, f"{canonical} not found for backend {backend}"
    return op.calculate(*args, **kwargs)


# ============================================================================
# BATCH 1 Tests: fiscal_acceleration
# ============================================================================


def test_fiscal_acceleration_basic():
    """Test fiscal acceleration basic computation."""
    # Create consecutive quarters with linear growth
    x, pid = _panel([1.0, 2.0, 3.0, 4.0, 5.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"])
    out = _calc("fiscal_acceleration", x, pid, lag=1, min_periods=3)

    # acceleration = x_t - 2*x_{t-1} + x_{t-2}
    # For linear series with constant first difference, acceleration should be 0
    assert np.allclose(out["A"].iloc[2:].dropna(), 0.0, atol=1e-10)


def test_fiscal_acceleration_requires_min_periods():
    """Test fiscal acceleration respects min_periods."""
    x, pid = _panel([1.0, 2.0], ["2023Q1", "2023Q2"])
    out = _calc("fiscal_acceleration", x, pid, lag=1, min_periods=3)
    assert out["A"].isna().all()


def test_fiscal_acceleration_fails_on_gap():
    """Test fiscal acceleration fails closed on fiscal gap."""
    x, pid = _panel([1.0, 2.0, 4.0, 5.0], ["2023Q1", "2023Q3", "2023Q4", "2024Q1"])
    out = _calc("fiscal_acceleration", x, pid, lag=1, min_periods=3, require_consecutive=True)
    # Gap between Q1 and Q3 should cause NaN
    assert np.isnan(out["A"].iloc[2])


# ============================================================================
# BATCH 1 Tests: fiscal_pct_change
# ============================================================================


def test_fiscal_pct_change_basic():
    """Test fiscal percentage change basic computation."""
    x, pid = _panel([100.0, 110.0, 121.0], ["2023Q1", "2023Q2", "2023Q3"])
    out = _calc("fiscal_pct_change", x, pid, lag=1, min_periods=2)

    # (110 - 100) / 100 = 0.1
    assert np.allclose(out["A"].iloc[1], 0.1, atol=1e-10)
    # (121 - 110) / 110 = 0.1
    assert np.allclose(out["A"].iloc[2], 0.1, atol=1e-10)


def test_fiscal_pct_change_handles_zero():
    """Test fiscal percentage change handles zero denominator."""
    x, pid = _panel([0.0, 10.0, 20.0], ["2023Q1", "2023Q2", "2023Q3"])
    out = _calc("fiscal_pct_change", x, pid, lag=1, min_periods=2)
    # Denominator is 0, should be NaN
    assert np.isnan(out["A"].iloc[1])
    # (20 - 10) / 10 = 1.0
    assert np.allclose(out["A"].iloc[2], 1.0, atol=1e-10)


def test_fiscal_pct_change_uses_abs_denominator():
    """Test fiscal percentage change uses absolute value in denominator."""
    x, pid = _panel([-100.0, 50.0], ["2023Q1", "2023Q2"])
    out = _calc("fiscal_pct_change", x, pid, lag=1, min_periods=2)
    # (50 - (-100)) / |-100| = 150 / 100 = 1.5
    assert np.allclose(out["A"].iloc[1], 1.5, atol=1e-10)


# ============================================================================
# BATCH 1 Tests: fiscal_rolling_std
# ============================================================================


def test_fiscal_rolling_std_basic():
    """Test fiscal rolling standard deviation."""
    x, pid = _panel([1.0, 2.0, 3.0, 4.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    out = _calc("fiscal_rolling_std", x, pid, periods=4, min_periods=3, ddof=1)
    # Last value should have std of [1, 2, 3, 4]
    expected_std = np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    assert np.allclose(out["A"].iloc[3], expected_std, atol=1e-10)


def test_fiscal_rolling_std_respects_ddof():
    """Test fiscal rolling std respects ddof parameter."""
    x, pid = _panel([1.0, 2.0, 3.0, 4.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    out_ddof1 = _calc("fiscal_rolling_std", x, pid, periods=4, min_periods=3, ddof=1)
    out_ddof0 = _calc("fiscal_rolling_std", x, pid, periods=4, min_periods=3, ddof=0)

    # ddof=1 (sample std) should be larger than ddof=0 (population std)
    assert out_ddof1["A"].iloc[3] > out_ddof0["A"].iloc[3]


# ============================================================================
# BATCH 1 Tests: fiscal_accrual_quality
# ============================================================================


def test_fiscal_accrual_quality_basic():
    """Test fiscal accrual quality returns negative std."""
    accruals, pid = _panel([0.1, 0.15, 0.12, 0.14, 0.13], ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"])
    out = _calc("fiscal_accrual_quality", accruals, pid, periods=5, min_periods=4)

    # Should be negative (lower volatility is better)
    assert out["A"].iloc[4] < 0
    # Absolute value should match std
    expected_std = np.std([0.1, 0.15, 0.12, 0.14, 0.13], ddof=1)
    assert np.allclose(abs(out["A"].iloc[4]), expected_std, atol=1e-10)


def test_fiscal_accrual_quality_requires_min_periods():
    """Test fiscal accrual quality requires min_periods."""
    accruals, pid = _panel([0.1, 0.15, 0.12], ["2023Q1", "2023Q2", "2023Q3"])
    out = _calc("fiscal_accrual_quality", accruals, pid, periods=5, min_periods=4)
    # Not enough periods
    assert out["A"].isna().all()


# ============================================================================
# BATCH 1 Tests: fiscal_direction_consistency
# ============================================================================


def test_fiscal_direction_consistency_all_positive():
    """Test direction consistency with all positive changes."""
    x, pid = _panel([1.0, 2.0, 3.0, 4.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    out = _calc("fiscal_direction_consistency", x, pid, periods=4, min_periods=3)
    # All changes are positive, consistency should be 1.0
    assert np.allclose(out["A"].iloc[3], 1.0, atol=1e-10)


def test_fiscal_direction_consistency_alternating():
    """Test direction consistency with alternating changes."""
    x, pid = _panel([1.0, 2.0, 1.5, 2.5, 2.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"])
    out = _calc("fiscal_direction_consistency", x, pid, periods=5, min_periods=3)
    # Changes: +1, -0.5, +1, -0.5
    # Last change is -0.5 (negative), only 2 of 4 are negative
    assert np.allclose(out["A"].iloc[4], 0.5, atol=1e-10)


# ============================================================================
# BATCH 2 Tests: fiscal_reversal_ratio
# ============================================================================


def test_fiscal_reversal_ratio_no_reversals():
    """Test reversal ratio with no sign changes."""
    x, pid = _panel([1.0, 2.0, 3.0, 4.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    out = _calc("fiscal_reversal_ratio", x, pid, periods=4, min_pairs=3)
    # No sign reversals, ratio should be 0
    assert np.allclose(out["A"].iloc[3], 0.0, atol=1e-10)


def test_fiscal_reversal_ratio_all_reversals():
    """Test reversal ratio with all sign changes."""
    x, pid = _panel([1.0, -1.0, 1.0, -1.0, 1.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"])
    out = _calc("fiscal_reversal_ratio", x, pid, periods=5, min_pairs=3)
    # All consecutive pairs have sign reversals
    # numerator = sum(min(1,1)) * 4 = 4, denominator = 4, ratio = 1.0
    assert np.allclose(out["A"].iloc[4], 1.0, atol=1e-10)


# ============================================================================
# BATCH 2 Tests: fiscal_standardized_surprise
# ============================================================================


def test_fiscal_standardized_surprise_basic():
    """Test standardized surprise basic computation."""
    # Create seasonal pattern with YoY growth
    x, pid = _panel(
        [100.0, 110.0, 120.0, 130.0, 105.0, 115.0, 125.0, 135.0, 110.0],
        ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"],
    )
    out = _calc("fiscal_standardized_surprise", x, pid, seasonal_lag=4, lookback_periods=4, min_history=3)
    # Should produce z-score for seasonal surprise
    assert np.isfinite(out["A"].iloc[8])


def test_fiscal_standardized_surprise_requires_history():
    """Test standardized surprise requires sufficient history."""
    x, pid = _panel([100.0, 110.0, 120.0], ["2023Q1", "2023Q2", "2023Q3"])
    out = _calc("fiscal_standardized_surprise", x, pid, seasonal_lag=4, lookback_periods=4, min_history=3)
    # Not enough seasonal lags
    assert out["A"].isna().all()


# ============================================================================
# BATCH 2 Tests: fiscal_pair_direction_agreement
# ============================================================================


def test_fiscal_pair_direction_agreement_perfect():
    """Test pair direction agreement with perfect alignment."""
    x, pid = _panel([1.0, 2.0, 3.0, 4.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    y, _ = _panel([10.0, 20.0, 30.0, 40.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    out = _calc("fiscal_pair_direction_agreement", x, y, pid, periods=4, min_periods=3)
    # Both positive at all points, agreement = 1.0
    assert np.allclose(out["A"].iloc[3], 1.0, atol=1e-10)


def test_fiscal_pair_direction_agreement_opposite():
    """Test pair direction agreement with opposite signs."""
    x, pid = _panel([1.0, 2.0, 3.0, 4.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    y, _ = _panel([-10.0, -20.0, -30.0, -40.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    out = _calc("fiscal_pair_direction_agreement", x, y, pid, periods=4, min_periods=3)
    # Opposite signs, agreement = 0.0
    assert np.allclose(out["A"].iloc[3], 0.0, atol=1e-10)


# ============================================================================
# BATCH 2 Tests: fiscal_asymmetric_elasticity
# ============================================================================


def test_fiscal_asymmetric_elasticity_basic():
    """Test asymmetric elasticity basic computation."""
    # Create sticky cost pattern: cost rises faster than it falls
    cost, pid = _panel(
        [100.0, 110.0, 120.0, 115.0, 125.0, 135.0, 130.0, 140.0, 150.0, 145.0, 155.0, 165.0, 160.0],
        [
            "2021Q1",
            "2021Q2",
            "2021Q3",
            "2021Q4",
            "2022Q1",
            "2022Q2",
            "2022Q3",
            "2022Q4",
            "2023Q1",
            "2023Q2",
            "2023Q3",
            "2023Q4",
            "2024Q1",
        ],
    )
    activity, _ = _panel(
        [1000.0, 1100.0, 1200.0, 1150.0, 1250.0, 1350.0, 1300.0, 1400.0, 1500.0, 1450.0, 1550.0, 1650.0, 1600.0],
        [
            "2021Q1",
            "2021Q2",
            "2021Q3",
            "2021Q4",
            "2022Q1",
            "2022Q2",
            "2022Q3",
            "2022Q4",
            "2023Q1",
            "2023Q2",
            "2023Q3",
            "2023Q4",
            "2024Q1",
        ],
    )
    out = _calc(
        "fiscal_asymmetric_elasticity",
        cost,
        activity,
        pid,
        periods=12,
        mode="down_minus_up",
        add_intercept=True,
        min_obs_per_regime=3,
    )
    # Should produce finite value
    assert np.isfinite(out["A"].iloc[12])


def test_fiscal_asymmetric_elasticity_requires_both_regimes():
    """Test asymmetric elasticity requires sufficient observations in both regimes."""
    cost, pid = _panel([100.0, 110.0, 120.0, 130.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    activity, _ = _panel([1000.0, 1100.0, 1200.0, 1300.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    out = _calc(
        "fiscal_asymmetric_elasticity",
        cost,
        activity,
        pid,
        periods=4,
        mode="down_minus_up",
        min_obs_per_regime=3,
    )
    # Only up regime, no down regime observations
    assert out["A"].isna().all()


# ============================================================================
# BATCH 2 Tests: fiscal_logit_score
# ============================================================================


def test_fiscal_logit_score_basic():
    """Test logit score basic computation."""
    # Create margin series in [0, 1]
    x, pid = _panel([0.1, 0.15, 0.2, 0.18, 0.22, 0.25], ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2"])
    out = _calc("fiscal_logit_score", x, pid, periods=6, min_periods=4)
    # Should produce z-score
    assert np.isfinite(out["A"].iloc[5])


def test_fiscal_logit_score_clamps_values():
    """Test logit score clamps extreme values."""
    # Values outside [0, 1] should be clamped
    x, pid = _panel([-0.1, 0.5, 1.1, 0.6], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    out = _calc("fiscal_logit_score", x, pid, periods=4, min_periods=3)
    # Should not raise error, values clamped to (0.001, 0.999)
    assert np.isfinite(out["A"].iloc[3])


# ============================================================================
# Backend parity tests (pandas_numpy vs polars)
# ============================================================================


@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
def test_fiscal_acceleration_backend_parity():
    """Test fiscal_acceleration pandas vs polars parity."""
    x, pid = _panel([1.0, 2.0, 4.0, 7.0, 11.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"])

    # Pandas backend
    out_pd = _calc("fiscal_acceleration", x, pid, lag=1, min_periods=3, backend="pandas_numpy")

    # Polars backend
    x_pl = pl.from_pandas(x)
    pid_pl = pl.from_pandas(pid)
    out_pl = _calc("fiscal_acceleration", x_pl, pid_pl, lag=1, min_periods=3, backend="polars")
    out_pl_pd = out_pl.to_pandas()

    # Compare
    pd.testing.assert_frame_equal(out_pd, out_pl_pd, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
def test_fiscal_reversal_ratio_backend_parity():
    """Test fiscal_reversal_ratio pandas vs polars parity."""
    x, pid = _panel([1.0, -1.0, 2.0, -2.0, 3.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"])

    # Pandas backend
    out_pd = _calc("fiscal_reversal_ratio", x, pid, periods=5, min_pairs=3, backend="pandas_numpy")

    # Polars backend
    x_pl = pl.from_pandas(x)
    pid_pl = pl.from_pandas(pid)
    out_pl = _calc("fiscal_reversal_ratio", x_pl, pid_pl, periods=5, min_pairs=3, backend="polars")
    out_pl_pd = out_pl.to_pandas()

    # Compare
    pd.testing.assert_frame_equal(out_pd, out_pl_pd, rtol=1e-10, atol=1e-10)


# ============================================================================
# Parameter validation tests
# ============================================================================


def test_fiscal_acceleration_validates_min_periods():
    """Test fiscal_acceleration validates min_periods constraint."""
    x, pid = _panel([1.0, 2.0, 3.0], ["2023Q1", "2023Q2", "2023Q3"])
    with pytest.raises(ValueError, match="min_periods must be at least"):
        _calc("fiscal_acceleration", x, pid, lag=2, min_periods=2)


def test_fiscal_pct_change_validates_min_periods():
    """Test fiscal_pct_change validates min_periods constraint."""
    x, pid = _panel([1.0, 2.0], ["2023Q1", "2023Q2"])
    with pytest.raises(ValueError, match="min_periods must be at least"):
        _calc("fiscal_pct_change", x, pid, lag=2, min_periods=1)


def test_fiscal_rolling_std_validates_ddof():
    """Test fiscal_rolling_std validates ddof constraint."""
    x, pid = _panel([1.0, 2.0, 3.0], ["2023Q1", "2023Q2", "2023Q3"])
    with pytest.raises(ValueError, match="min_periods must be greater than ddof"):
        _calc("fiscal_rolling_std", x, pid, periods=5, min_periods=1, ddof=1)


def test_fiscal_asymmetric_elasticity_validates_mode():
    """Test fiscal_asymmetric_elasticity validates mode parameter."""
    cost, pid = _panel([100.0, 110.0], ["2023Q1", "2023Q2"])
    activity, _ = _panel([1000.0, 1100.0], ["2023Q1", "2023Q2"])
    with pytest.raises(ValueError, match="mode must be 'down_minus_up'"):
        _calc("fiscal_asymmetric_elasticity", cost, activity, pid, mode="invalid")


# ============================================================================
# Revision policy tests
# ============================================================================


def test_fiscal_operators_support_revision_policies():
    """Test fiscal operators support both revision policies."""
    x, pid = _panel([1.0, 2.0, 3.0, 4.0], ["2023Q1", "2023Q2", "2023Q3", "2023Q4"])

    # Both should work without error
    out_latest = _calc("fiscal_acceleration", x, pid, revision_policy="latest_available")
    out_first = _calc("fiscal_acceleration", x, pid, revision_policy="first_available")

    # Both should produce results (may differ with actual revisions)
    assert out_latest.notna().any().any()
    assert out_first.notna().any().any()


def test_fiscal_operators_reject_invalid_revision_policy():
    """Test fiscal operators reject invalid revision policy."""
    x, pid = _panel([1.0, 2.0], ["2023Q1", "2023Q2"])
    with pytest.raises(ValueError, match="revision_policy must be"):
        _calc("fiscal_acceleration", x, pid, revision_policy="invalid")
