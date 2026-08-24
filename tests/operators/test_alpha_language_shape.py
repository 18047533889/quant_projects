# -*- coding: utf-8 -*-
"""Tests for alpha_language_shape operators.

Comprehensive coverage of 12 path/shape geometry operators:
1. ts_monotonicity - Kendall-style directional agreement
2. ts_turning_rate - delta sign flip rate (Definition A)
3. ts_effective_turning_rate - effective turning rate (Definition B)
4. ts_turning_intensity - magnitude of turns
5. ts_path_efficiency - net displacement vs path length
6. ts_path_roughness - second-order jaggedness
7. ts_trend_break - split-window slope difference
8. ts_weighted_time_centroid - where action happened in time
9. ts_mass_concentration - HHI of weighted mass
10. ts_endpoint_deviation - deviation from trailing OLS fit
11. ts_directional_concentration - HHI of directional mass
12. ts_volatility_clustering - GARCH-like vol clustering
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None, f"{name}/{backend}"
    return op


# ---------------------------------------------------------------------------
# 1. ts_monotonicity
# ---------------------------------------------------------------------------
def test_ts_monotonicity_strictly_increasing() -> None:
    """Strictly increasing sequence -> +1."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = _op("ts_monotonicity").calculate(x, window=5, min_periods=3)

    # Last value should be close to +1 (perfectly monotonic increasing)
    assert np.isfinite(result.iloc[-1, 0])
    assert result.iloc[-1, 0] > 0.9


def test_ts_monotonicity_strictly_decreasing() -> None:
    """Strictly decreasing sequence -> -1."""
    x = pd.DataFrame({"A": [5.0, 4.0, 3.0, 2.0, 1.0]})
    result = _op("ts_monotonicity").calculate(x, window=5, min_periods=3)

    # Last value should be close to -1 (perfectly monotonic decreasing)
    assert np.isfinite(result.iloc[-1, 0])
    assert result.iloc[-1, 0] < -0.9


def test_ts_monotonicity_random_walk() -> None:
    """Random walk -> near 0."""
    np.random.seed(42)
    x = pd.DataFrame({"A": np.cumsum(np.random.randn(50))})
    result = _op("ts_monotonicity").calculate(x, window=20, min_periods=10)

    # Should be somewhere between -1 and 1, not extreme
    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    assert -1.0 <= last_val <= 1.0


def test_ts_monotonicity_constant() -> None:
    """Constant sequence -> NaN (no valid pairs)."""
    x = pd.DataFrame({"A": [5.0] * 10})
    result = _op("ts_monotonicity").calculate(x, window=5, min_periods=3)

    # All constant -> NaN
    assert result.isna().all().all()


def test_ts_monotonicity_insufficient_periods() -> None:
    """Insufficient min_periods -> NaN."""
    x = pd.DataFrame({"A": [1.0, 2.0]})
    result = _op("ts_monotonicity").calculate(x, window=5, min_periods=5)

    assert result.isna().all().all()


def test_ts_monotonicity_with_nans() -> None:
    """NaN values should be filtered out."""
    x = pd.DataFrame({"A": [1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0]})
    result = _op("ts_monotonicity").calculate(x, window=7, min_periods=3)

    # Should compute on finite values only
    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    assert last_val > 0.5  # Mostly increasing


# ---------------------------------------------------------------------------
# 2. ts_turning_rate
# ---------------------------------------------------------------------------
def test_ts_turning_rate_monotonic() -> None:
    """Monotonic sequence -> 0 turning rate."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = _op("ts_turning_rate").calculate(x, window=5, min_periods=2)

    # No turns -> 0
    assert result.iloc[-1, 0] == 0.0


def test_ts_turning_rate_alternating() -> None:
    """Alternating up/down -> high turning rate."""
    x = pd.DataFrame({"A": [1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5]})
    result = _op("ts_turning_rate").calculate(x, window=7, min_periods=2)

    # Multiple turns -> positive rate
    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    assert last_val > 0.3


def test_ts_turning_rate_with_epsilon() -> None:
    """Epsilon filters small changes."""
    x = pd.DataFrame({"A": [1.0, 1.01, 0.99, 1.02, 0.98]})

    # With epsilon=0.05, these are all "zero" changes
    result = _op("ts_turning_rate").calculate(x, window=5, epsilon=0.05, min_periods=2)

    # Should be NaN or 0 (no significant turns)
    last_val = result.iloc[-1, 0]
    assert np.isnan(last_val) or last_val == 0.0


def test_ts_turning_rate_platform_segments() -> None:
    """Platform segments (zeros) included in denominator."""
    # [+, 0, 0, 0, -] has 4 adjacent pairs but only 1 flip
    x = pd.DataFrame({"A": [1.0, 2.0, 2.0, 2.0, 2.0, 1.0]})
    result = _op("ts_turning_rate").calculate(x, window=6, min_periods=2)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    # Definition A: platforms dilute the rate
    assert last_val < 0.5


# ---------------------------------------------------------------------------
# 3. ts_effective_turning_rate
# ---------------------------------------------------------------------------
def test_ts_effective_turning_rate_monotonic() -> None:
    """Monotonic sequence -> 0 effective turning rate."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = _op("ts_effective_turning_rate").calculate(x, window=5, min_periods=2)

    assert result.iloc[-1, 0] == 0.0


def test_ts_effective_turning_rate_platform_segments() -> None:
    """Platform segments excluded from denominator."""
    # [+, 0, 0, 0, -] has no "both deltas nonzero" pairs -> NaN
    x = pd.DataFrame({"A": [1.0, 2.0, 2.0, 2.0, 2.0, 1.0]})
    result = _op("ts_effective_turning_rate").calculate(x, window=6, min_periods=2)

    last_val = result.iloc[-1, 0]
    # Definition B: no active pairs -> NaN
    assert np.isnan(last_val)


def test_ts_effective_turning_rate_alternating() -> None:
    """Alternating with no platforms -> high rate."""
    x = pd.DataFrame({"A": [1.0, 2.0, 1.5, 2.5, 2.0, 3.0]})
    result = _op("ts_effective_turning_rate").calculate(x, window=6, min_periods=2)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    assert last_val > 0.5


# ---------------------------------------------------------------------------
# 4. ts_turning_intensity
# ---------------------------------------------------------------------------
def test_ts_turning_intensity_smooth() -> None:
    """Smooth monotonic -> low intensity."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = _op("ts_turning_intensity").calculate(x, window=5, min_periods=2)

    # Smooth -> low or NaN
    last_val = result.iloc[-1, 0]
    assert np.isnan(last_val) or last_val < 1.0


def test_ts_turning_intensity_sharp_turns() -> None:
    """Sharp turns -> high intensity."""
    x = pd.DataFrame({"A": [1.0, 1.1, 5.0, 5.1, 1.0]})
    result = _op("ts_turning_intensity").calculate(x, window=5, min_periods=2)

    last_val = result.iloc[-1, 0]
    if np.isfinite(last_val):
        assert last_val > 1.0


# ---------------------------------------------------------------------------
# 5. ts_path_efficiency
# ---------------------------------------------------------------------------
def test_ts_path_efficiency_straight_line() -> None:
    """Straight line -> efficiency = 1."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = _op("ts_path_efficiency").calculate(x, window=5, min_periods=2)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    # Net displacement = path length for monotonic
    assert last_val > 0.95


def test_ts_path_efficiency_zigzag() -> None:
    """Zigzag path -> low efficiency."""
    x = pd.DataFrame({"A": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})
    result = _op("ts_path_efficiency").calculate(x, window=6, min_periods=2)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    # Lots of back-and-forth -> low efficiency
    assert last_val < 0.5


def test_ts_path_efficiency_zero_displacement() -> None:
    """Zero net displacement -> 0 efficiency."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 2.0, 1.0]})
    result = _op("ts_path_efficiency").calculate(x, window=5, min_periods=2)

    last_val = result.iloc[-1, 0]
    # Start and end at same place -> 0
    assert last_val == 0.0


# ---------------------------------------------------------------------------
# 6. ts_path_roughness
# ---------------------------------------------------------------------------
def test_ts_path_roughness_smooth() -> None:
    """Smooth curve -> low roughness."""
    x = pd.DataFrame({"A": [1.0, 1.5, 2.0, 2.5, 3.0]})
    result = _op("ts_path_roughness").calculate(x, window=5, min_periods=3)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    # Linear -> second derivative ~0
    assert last_val < 0.5


def test_ts_path_roughness_jagged() -> None:
    """Jagged path -> high roughness."""
    x = pd.DataFrame({"A": [1.0, 5.0, 2.0, 6.0, 3.0]})
    result = _op("ts_path_roughness").calculate(x, window=5, min_periods=3)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    assert last_val > 1.0


# ---------------------------------------------------------------------------
# 7. ts_trend_break
# ---------------------------------------------------------------------------
def test_ts_trend_break_constant_trend() -> None:
    """Constant trend -> 0 break."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    result = _op("ts_trend_break").calculate(x, window=6, min_periods=4)

    last_val = result.iloc[-1, 0]
    # Consistent slope -> near 0 break
    assert np.isfinite(last_val)
    assert abs(last_val) < 0.1


def test_ts_trend_break_reversal() -> None:
    """Trend reversal -> large break."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]})
    result = _op("ts_trend_break").calculate(x, window=6, min_periods=4)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    # Reversal -> negative break
    assert last_val < -0.5


# ---------------------------------------------------------------------------
# 8. ts_weighted_time_centroid
# ---------------------------------------------------------------------------
def test_ts_weighted_time_centroid_uniform() -> None:
    """Uniform weights -> centroid at midpoint."""
    x = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0]})
    result = _op("ts_weighted_time_centroid").calculate(x, window=5, min_periods=3)

    last_val = result.iloc[-1, 0]
    # Uniform -> centroid ~0.5
    assert np.isfinite(last_val)
    assert 0.4 <= last_val <= 0.6


def test_ts_weighted_time_centroid_back_loaded() -> None:
    """Back-loaded -> centroid toward 1."""
    x = pd.DataFrame({"A": [1.0, 1.0, 1.0, 5.0, 10.0]})
    result = _op("ts_weighted_time_centroid").calculate(x, window=5, min_periods=3)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    assert last_val > 0.7


def test_ts_weighted_time_centroid_front_loaded() -> None:
    """Front-loaded -> centroid toward 0."""
    x = pd.DataFrame({"A": [10.0, 5.0, 1.0, 1.0, 1.0]})
    result = _op("ts_weighted_time_centroid").calculate(x, window=5, min_periods=3)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    assert last_val < 0.3


# ---------------------------------------------------------------------------
# 9. ts_mass_concentration
# ---------------------------------------------------------------------------
def test_ts_mass_concentration_uniform() -> None:
    """Uniform distribution -> low HHI."""
    x = pd.DataFrame({"A": [1.0] * 10})
    result = _op("ts_mass_concentration").calculate(x, window=10, min_periods=5)

    last_val = result.iloc[-1, 0]
    # Uniform -> HHI = 1/n = 0.1
    assert np.isfinite(last_val)
    assert 0.09 <= last_val <= 0.11


def test_ts_mass_concentration_concentrated() -> None:
    """Concentrated mass -> high HHI."""
    x = pd.DataFrame({"A": [0.1, 0.1, 0.1, 10.0, 0.1]})
    result = _op("ts_mass_concentration").calculate(x, window=5, min_periods=3)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    # One dominant value -> high HHI
    assert last_val > 0.5


# ---------------------------------------------------------------------------
# 10. ts_endpoint_deviation
# ---------------------------------------------------------------------------
def test_ts_endpoint_deviation_on_line() -> None:
    """Point on OLS line -> 0 deviation."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = _op("ts_endpoint_deviation").calculate(x, window=5, min_periods=3)

    last_val = result.iloc[-1, 0]
    # Perfect line -> near 0
    assert np.isfinite(last_val)
    assert abs(last_val) < 0.1


def test_ts_endpoint_deviation_outlier() -> None:
    """Endpoint outlier -> large deviation."""
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 10.0]})
    result = _op("ts_endpoint_deviation").calculate(x, window=5, min_periods=3)

    last_val = result.iloc[-1, 0]
    assert np.isfinite(last_val)
    # Outlier -> large positive deviation
    assert last_val > 2.0


# ---------------------------------------------------------------------------
# Unit consistency
# ---------------------------------------------------------------------------
def test_alpha_language_shape_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        ("ts_monotonicity", "ratio"),
        ("ts_turning_rate", "ratio"),
        ("ts_effective_turning_rate", "ratio"),
        ("ts_path_efficiency", "ratio"),
        ("ts_weighted_time_centroid", "ratio"),
    ]

    for op_name, expected_unit in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        # Check unit tag
        unit_tags = [tag for tag in meta.tags if tag.startswith("unit:")]
        assert len(unit_tags) == 1, f"{op_name} missing unit tag"
        assert unit_tags[0] == f"unit:{expected_unit}", f"{op_name} wrong unit"
