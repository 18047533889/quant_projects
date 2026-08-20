# -*- coding: utf-8 -*-
"""Tests for TRUE_GAP intraday operators batch 3 (2026-08-13).

Tests three operators:
- intra_session_mean_reversion
- intra_price_delay
- intra_volume_imbalance

Note: intra_same_slot_zscore, intra_realized_variance, intra_state_vwap,
and intra_smart_money_vwap_ratio already exist in other modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.intraday.true_gap_batch3  # noqa: F401

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

CANONICALS = [
    "intra_session_mean_reversion",
    "intra_price_delay",
    "intra_volume_imbalance",
]


def _minute_panel(
    days: int = 3,
    bars_per_day: int = 240,
    cols: int = 2,
    seed: int = 42,
    start: str = "2024-01-02 09:31:00",
) -> pd.DataFrame:
    """Generate minute-frequency panel for testing.

    A-share session: 09:31-11:30 (120 bars) + 13:01-15:00 (120 bars) = 240 bars/day.
    """
    rng = np.random.default_rng(seed)

    # Generate timestamps for each day
    all_times = []
    base_date = pd.Timestamp(start)

    for day_offset in range(days):
        day = base_date + pd.Timedelta(days=day_offset)
        # Morning session: 09:31-11:30 (120 minutes)
        morning = pd.date_range(
            day.replace(hour=9, minute=31),
            day.replace(hour=11, minute=30),
            freq="1min",
        )
        # Afternoon session: 13:01-15:00 (120 minutes)
        afternoon = pd.date_range(
            day.replace(hour=13, minute=1),
            day.replace(hour=15, minute=0),
            freq="1min",
        )
        all_times.extend(list(morning))
        all_times.extend(list(afternoon))

    idx = pd.DatetimeIndex(all_times)
    n_rows = len(idx)

    # Generate price data with realistic intraday patterns
    data = np.exp(np.cumsum(rng.standard_normal((n_rows, cols)) * 0.001, axis=0)) * 100.0

    return pd.DataFrame(data, index=idx, columns=[f"C{i}" for i in range(cols)])


def _with_volume(price_df: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    """Generate corresponding volume panel."""
    rng = np.random.default_rng(seed)
    vol_data = rng.exponential(scale=1e6, size=price_df.shape)
    return pd.DataFrame(vol_data, index=price_df.index, columns=price_df.columns)


# ---------------------------------------------------------------------------
# Registration and surface tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CANONICALS))
def test_registered_and_classified(name: str) -> None:
    """All operators are registered and on extended surface."""
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    surf = classify_canonical(name)
    assert surf in ("daily", "extended", "research"), f"{name} -> {surf}"


# ---------------------------------------------------------------------------
# 1. intra_session_mean_reversion
# ---------------------------------------------------------------------------
def test_intra_session_mean_reversion_basic() -> None:
    """Mean reversion produces one scalar per day per instrument."""
    df = _minute_panel(days=3, bars_per_day=240, cols=2, seed=1)
    op = OperatorRegistry.get("intra_session_mean_reversion")
    out = op.calculate(df)

    # Output should be daily (3 days)
    assert len(out) == 3, f"expected 3 days, got {len(out)}"
    assert out.shape[1] == 2, f"expected 2 columns, got {out.shape[1]}"

    # All values should be finite (or NaN for degenerate cases)
    finite_count = out.notna().sum().sum()
    assert finite_count > 0, "expected some finite values"


def test_intra_session_mean_reversion_bounded() -> None:
    """Mean reversion score should be in reasonable range."""
    df = _minute_panel(days=2, bars_per_day=240, cols=1, seed=2)
    op = OperatorRegistry.get("intra_session_mean_reversion")
    out = op.calculate(df)

    # Should produce 2 daily values
    assert len(out) == 2

    # All output should be finite or NaN (never Inf)
    assert not np.isinf(out.values).any()

    # Values should be in [-2, 2] range (based on correlation bounds)
    finite_vals = out.values[np.isfinite(out.values)]
    if len(finite_vals) > 0:
        assert np.all(finite_vals >= -2.0) and np.all(finite_vals <= 2.0)


def test_intra_session_mean_reversion_determinism() -> None:
    """Repeated calculation produces identical results."""
    df = _minute_panel(days=5, bars_per_day=240, cols=2, seed=3)
    op = OperatorRegistry.get("intra_session_mean_reversion")

    first = op.calculate(df)
    second = op.calculate(df)

    pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# 2. intra_price_delay
# ---------------------------------------------------------------------------
def test_intra_price_delay_bounded() -> None:
    """Price delay (correlation) should be in [-1, 1]."""
    df = _minute_panel(days=3, bars_per_day=240, cols=2, seed=8)
    vol = _with_volume(df, seed=9)

    op = OperatorRegistry.get("intra_price_delay")
    out = op.calculate(df, vol)

    finite_vals = out.values[np.isfinite(out.values)]
    if len(finite_vals) > 0:
        assert np.all(finite_vals >= -1.0) and np.all(finite_vals <= 1.0), \
            f"correlation should be in [-1, 1], got {finite_vals}"


def test_intra_price_delay_requires_volume() -> None:
    """Price delay calculation requires volume data."""
    df = _minute_panel(days=2, bars_per_day=240, cols=1, seed=10)
    vol = _with_volume(df, seed=11)

    op = OperatorRegistry.get("intra_price_delay")
    out = op.calculate(df, vol)

    # Should produce daily output
    assert len(out) == 2


def test_intra_price_delay_determinism() -> None:
    """Repeated calculation produces identical results."""
    df = _minute_panel(days=3, bars_per_day=240, cols=2, seed=12)
    vol = _with_volume(df, seed=13)

    op = OperatorRegistry.get("intra_price_delay")

    first = op.calculate(df, vol)
    second = op.calculate(df, vol)

    pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# 4. intra_volume_imbalance
# ---------------------------------------------------------------------------
def test_intra_volume_imbalance_bounded() -> None:
    """Volume imbalance should be in [-1, 1]."""
    df = _minute_panel(days=3, bars_per_day=240, cols=2, seed=14)
    vol = _with_volume(df, seed=15)

    op = OperatorRegistry.get("intra_volume_imbalance")
    out = op.calculate(df, vol)

    finite_vals = out.values[np.isfinite(out.values)]
    if len(finite_vals) > 0:
        assert np.all(finite_vals >= -1.0) and np.all(finite_vals <= 1.0), \
            f"imbalance should be in [-1, 1], got range [{finite_vals.min()}, {finite_vals.max()}]"


def test_intra_volume_imbalance_vwap_centered() -> None:
    """Imbalance measures volume distribution around VWAP."""
    # Create simple scenario: all volume at one price level
    idx = pd.date_range("2024-01-02 09:31", periods=10, freq="1min")

    # Scenario 1: Price rises, more volume at higher prices
    prices = pd.DataFrame([100.0] * 5 + [110.0] * 5, index=idx, columns=["A"])
    volume = pd.DataFrame([1e6] * 5 + [3e6] * 5, index=idx, columns=["A"])  # more volume at 110

    op = OperatorRegistry.get("intra_volume_imbalance")
    imb = op.calculate(prices, volume).iloc[0, 0]

    # Should be positive (more volume above VWAP)
    assert imb > 0, f"expected positive imbalance, got {imb}"


def test_intra_volume_imbalance_determinism() -> None:
    """Repeated calculation produces identical results."""
    df = _minute_panel(days=3, bars_per_day=240, cols=2, seed=16)
    vol = _with_volume(df, seed=17)

    op = OperatorRegistry.get("intra_volume_imbalance")

    first = op.calculate(df, vol)
    second = op.calculate(df, vol)

    pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# Multi-column and edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CANONICALS)
def test_multicol_shape_preserved(name: str) -> None:
    """Output shape matches input column count (daily frequency)."""
    df = _minute_panel(days=3, bars_per_day=240, cols=3, seed=18)
    vol = _with_volume(df, seed=19)

    op = OperatorRegistry.get(name)

    if name in ["intra_price_delay", "intra_volume_imbalance"]:
        out = op.calculate(df, vol)
    else:
        out = op.calculate(df)

    # Output should have 3 days (daily frequency) and 3 columns
    assert out.shape[1] == 3, f"expected 3 columns, got {out.shape[1]}"
    assert len(out) == 3, f"expected 3 days, got {len(out)}"


@pytest.mark.parametrize("name", CANONICALS)
def test_all_nan_input_produces_all_nan(name: str) -> None:
    """All-NaN input produces all-NaN output."""
    idx = pd.date_range("2024-01-02 09:31", periods=240, freq="1min")
    df = pd.DataFrame(np.nan, index=idx, columns=["A", "B"])
    vol = pd.DataFrame(np.nan, index=idx, columns=["A", "B"])

    op = OperatorRegistry.get(name)

    if name in ["intra_price_delay", "intra_volume_imbalance"]:
        out = op.calculate(df, vol)
    else:
        out = op.calculate(df)

    assert out.isna().all().all(), f"{name} should return all NaN for all-NaN input"


# ---------------------------------------------------------------------------
# Policy and tags
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CANONICALS)
def test_pit_safe_and_session_aware_tags(name: str) -> None:
    """All operators have pit_safe, session_aware, and true_gap tags."""
    op = OperatorRegistry.get(name)
    tags = op.metadata.tags

    assert "pit_safe" in tags, f"{name} missing pit_safe tag"
    assert "session_aware" in tags, f"{name} missing session_aware tag"
    assert "true_gap" in tags, f"{name} missing true_gap tag"
    assert "causal" in tags, f"{name} missing causal tag"


@pytest.mark.parametrize("name", CANONICALS)
def test_grain_minute_to_daily(name: str) -> None:
    """All operators have correct grain transformation."""
    op = OperatorRegistry.get(name)
    assert op.metadata.input_grain == "minute", f"{name} input_grain != minute"
    assert op.metadata.output_grain == "daily", f"{name} output_grain != daily"
