# -*- coding: utf-8 -*-
"""Tests for panel_batch1 TRUE_GAP operators (2026-08-13).

Tests registration, shape preservation, determinism, and golden-value semantics
for the 5 panel_batch1 operators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.cross_section.panel_batch1  # noqa: F401
import cleaned_operators.cross_section.panel_batch1_polars  # noqa: F401

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

BATCH1_CANONICALS = [
    "panel_day_night_beta_gap",
    "pastor_stambaugh_beta",
    "price_delay_score",
    "report_asof",
    "event_window_return_asof",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _daily_panel(days: int = 100, cols: int = 3, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="B")
    return pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((days, cols)) * 0.01, axis=0)) * 100.0,
        index=idx,
        columns=[f"C{i}" for i in range(cols)],
    )


def _event_panel(days: int = 100, cols: int = 3, seed: int = 0, event_prob: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    events = (rng.random((days, cols)) < event_prob).astype(float)
    return pd.DataFrame(events, index=idx, columns=[f"C{i}" for i in range(cols)])


# ---------------------------------------------------------------------------
# registration + surface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(BATCH1_CANONICALS)))
def test_registered_and_classified(name: str) -> None:
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    assert classify_canonical(name) in ("daily", "extended", "research"), name


@pytest.mark.parametrize("name", sorted(set(BATCH1_CANONICALS)))
def test_has_pandas_backend(name: str) -> None:
    backends = OperatorRegistry.backends_for(name)
    assert "pandas_numpy" in backends, f"{name} missing pandas_numpy backend"


# ---------------------------------------------------------------------------
# 1. panel_day_night_beta_gap
# ---------------------------------------------------------------------------
def test_panel_day_night_beta_gap_shape_and_determinism() -> None:
    day = _daily_panel(days=80, cols=2, seed=1)
    night = _daily_panel(days=80, cols=2, seed=2) * 0.8
    mkt = _daily_panel(days=80, cols=2, seed=3) * 1.2
    op = OperatorRegistry.get("panel_day_night_beta_gap")
    first = op.calculate(day, night, mkt, window=20, min_periods=10)
    second = op.calculate(day, night, mkt, window=20, min_periods=10)
    assert first.shape == day.shape
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_panel_day_night_beta_gap_zero_gap_when_identical() -> None:
    """When day and night returns are identical, gap should be ~0."""
    ret = _daily_panel(days=60, cols=2, seed=5)
    mkt = _daily_panel(days=60, cols=2, seed=6)
    op = OperatorRegistry.get("panel_day_night_beta_gap")
    out = op.calculate(ret, ret, mkt, window=30, min_periods=15)
    valid = out.notna()
    # Gap should be very close to zero (numerical precision tolerance)
    assert (out[valid].abs() < 0.01).all().all()


def test_panel_day_night_beta_gap_warmup() -> None:
    day = _daily_panel(days=50, cols=2, seed=7)
    night = _daily_panel(days=50, cols=2, seed=8)
    mkt = _daily_panel(days=50, cols=2, seed=9)
    op = OperatorRegistry.get("panel_day_night_beta_gap")
    out = op.calculate(day, night, mkt, window=20, min_periods=15)
    # Should have NaN for first min_periods-1 rows
    assert out.iloc[:14].isna().all().all()


# ---------------------------------------------------------------------------
# 2. pastor_stambaugh_beta
# ---------------------------------------------------------------------------
def test_pastor_stambaugh_beta_shape_and_determinism() -> None:
    ret = _daily_panel(days=150, cols=2, seed=10)
    mkt = _daily_panel(days=150, cols=2, seed=11)
    liq = _daily_panel(days=150, cols=2, seed=12) * 0.5
    op = OperatorRegistry.get("pastor_stambaugh_beta")
    first = op.calculate(ret, mkt, liq, window=60, min_periods=40)
    second = op.calculate(ret, mkt, liq, window=60, min_periods=40)
    assert first.shape == ret.shape
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_pastor_stambaugh_beta_warmup() -> None:
    ret = _daily_panel(days=80, cols=2, seed=13)
    mkt = _daily_panel(days=80, cols=2, seed=14)
    liq = _daily_panel(days=80, cols=2, seed=15)
    op = OperatorRegistry.get("pastor_stambaugh_beta")
    out = op.calculate(ret, mkt, liq, window=50, min_periods=30)
    # Should have NaN for first min_periods-1 rows
    assert out.iloc[:29].isna().all().all()


def test_pastor_stambaugh_beta_constant_liquidity_zero_beta() -> None:
    """When liquidity is constant, orthogonalized liquidity has no variance -> NaN."""
    ret = _daily_panel(days=100, cols=2, seed=16)
    mkt = _daily_panel(days=100, cols=2, seed=17)
    liq_const = pd.DataFrame(100.0, index=ret.index, columns=ret.columns)
    op = OperatorRegistry.get("pastor_stambaugh_beta")
    out = op.calculate(ret, mkt, liq_const, window=40, min_periods=20)
    # Constant liquidity -> zero variance after orthogonalization -> NaN
    assert out.isna().all().all()


# ---------------------------------------------------------------------------
# 3. price_delay_score
# ---------------------------------------------------------------------------
def test_price_delay_score_shape_and_determinism() -> None:
    ret = _daily_panel(days=150, cols=2, seed=20)
    mkt = _daily_panel(days=150, cols=2, seed=21)
    op = OperatorRegistry.get("price_delay_score")
    first = op.calculate(ret, mkt, window=80, lag=4, min_periods=50)
    second = op.calculate(ret, mkt, window=80, lag=4, min_periods=50)
    assert first.shape == ret.shape
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_price_delay_score_nonnegative() -> None:
    """Delay score = unrestricted R² - restricted R² should be >= 0."""
    ret = _daily_panel(days=200, cols=2, seed=22)
    mkt = _daily_panel(days=200, cols=2, seed=23)
    op = OperatorRegistry.get("price_delay_score")
    out = op.calculate(ret, mkt, window=100, lag=5, min_periods=60)
    valid = out.notna()
    # Delay score should be non-negative by construction
    assert (out[valid] >= -1e-9).all().all()


def test_price_delay_score_warmup() -> None:
    ret = _daily_panel(days=100, cols=2, seed=24)
    mkt = _daily_panel(days=100, cols=2, seed=25)
    op = OperatorRegistry.get("price_delay_score")
    out = op.calculate(ret, mkt, window=60, lag=3, min_periods=40)
    # Should have NaN for early rows
    assert out.iloc[:42].isna().all().all()


# ---------------------------------------------------------------------------
# 4. report_asof
# ---------------------------------------------------------------------------
def test_report_asof_shape_and_determinism() -> None:
    event = _event_panel(days=100, cols=2, seed=30, event_prob=0.1)
    op = OperatorRegistry.get("report_asof")
    first = op.calculate(event, max_lookback=60)
    second = op.calculate(event, max_lookback=60)
    assert first.shape == event.shape
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_report_asof_event_day_is_zero() -> None:
    """On the event day, days_since should be 0."""
    days = 20
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    event = pd.DataFrame(0.0, index=idx, columns=["A"])
    event.iloc[10, 0] = 1.0  # Event on day 10
    op = OperatorRegistry.get("report_asof")
    out = op.calculate(event, max_lookback=30)
    assert out.iloc[10, 0] == 0.0  # Event day = 0
    assert out.iloc[11, 0] == 1.0  # Next day = 1
    assert out.iloc[12, 0] == 2.0  # Two days after = 2


def test_report_asof_max_lookback_cutoff() -> None:
    """Beyond max_lookback days from last event, output should be NaN."""
    days = 50
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    event = pd.DataFrame(0.0, index=idx, columns=["A"])
    event.iloc[5, 0] = 1.0  # Event on day 5
    op = OperatorRegistry.get("report_asof")
    out = op.calculate(event, max_lookback=10)
    # Days 5-15 should have values (0-10), beyond should be NaN
    assert out.iloc[5:16, 0].notna().all()
    assert out.iloc[16:, 0].isna().all()


def test_report_asof_no_event_all_nan() -> None:
    """When there's no event, output should be all NaN."""
    event = pd.DataFrame(0.0, index=pd.date_range("2024-01-01", periods=30, freq="B"), columns=["A"])
    op = OperatorRegistry.get("report_asof")
    out = op.calculate(event, max_lookback=100)
    assert out.isna().all().all()


# ---------------------------------------------------------------------------
# 5. event_window_return_asof
# ---------------------------------------------------------------------------
def test_event_window_return_asof_shape_and_determinism() -> None:
    ret = _daily_panel(days=100, cols=2, seed=40)
    event = _event_panel(days=100, cols=2, seed=41, event_prob=0.08)
    op = OperatorRegistry.get("event_window_return_asof")
    first = op.calculate(ret, event, pre_window=3, post_window=5, max_lookback=60)
    second = op.calculate(ret, event, pre_window=3, post_window=5, max_lookback=60)
    assert first.shape == ret.shape
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_event_window_return_asof_golden() -> None:
    """Golden test: known event, known returns, verify cumulative window return."""
    days = 30
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    ret = pd.DataFrame(0.01, index=idx, columns=["A"])  # Constant 1% daily return
    event = pd.DataFrame(0.0, index=idx, columns=["A"])
    event.iloc[10, 0] = 1.0  # Event on day 10

    # Window: [event-2, event+3] = [8, 13], 6 days total
    # Cumulative return = (1.01)^6 - 1 ≈ 0.061520
    op = OperatorRegistry.get("event_window_return_asof")
    out = op.calculate(ret, event, pre_window=2, post_window=3, max_lookback=50)

    # Can only compute after window completes (after day 13)
    expected_cum = (1.01 ** 6) - 1.0
    for t in range(14, days):  # From day 14 onwards
        assert out.iloc[t, 0] == pytest.approx(expected_cum, rel=1e-6)

    # Before window completes, should be NaN
    assert out.iloc[:14, 0].isna().all()


def test_event_window_return_asof_no_forward_looking() -> None:
    """Output should only appear after the event window fully completes."""
    days = 40
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    ret = _daily_panel(days=days, cols=1, seed=50)
    event = pd.DataFrame(0.0, index=idx, columns=["C0"])
    event.iloc[15, 0] = 1.0  # Event on day 15

    op = OperatorRegistry.get("event_window_return_asof")
    out = op.calculate(ret, event, pre_window=5, post_window=10, max_lookback=60)

    # Window: [10, 25], completes on day 25
    # Should have NaN before day 26
    assert out.iloc[:26, 0].isna().all()
    # After day 25, should have values (if returns are valid)
    # Note: may still be NaN if there are NaN returns in the window


def test_event_window_return_asof_max_lookback() -> None:
    """Events beyond max_lookback should be ignored."""
    days = 80
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    ret = pd.DataFrame(0.01, index=idx, columns=["A"])
    event = pd.DataFrame(0.0, index=idx, columns=["A"])
    event.iloc[10, 0] = 1.0  # Event on day 10

    op = OperatorRegistry.get("event_window_return_asof")
    out = op.calculate(ret, event, pre_window=2, post_window=3, max_lookback=20)

    # Window completes on day 13; within lookback until day 30 (10 + 20)
    # After day 30, should be NaN
    assert out.iloc[31:, 0].isna().all()


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_all_operators_handle_all_nan_input() -> None:
    """All operators should handle all-NaN input gracefully."""
    nan_panel = pd.DataFrame(np.nan, index=pd.date_range("2024-01-01", periods=50, freq="B"), columns=["A", "B"])

    for name in BATCH1_CANONICALS:
        op = OperatorRegistry.get(name)
        if name == "panel_day_night_beta_gap":
            out = op.calculate(nan_panel, nan_panel, nan_panel)
        elif name == "pastor_stambaugh_beta":
            out = op.calculate(nan_panel, nan_panel, nan_panel)
        elif name == "price_delay_score":
            out = op.calculate(nan_panel, nan_panel)
        elif name == "report_asof":
            out = op.calculate(nan_panel)
        elif name == "event_window_return_asof":
            out = op.calculate(nan_panel, nan_panel)
        else:
            continue

        assert out.shape == nan_panel.shape, name
        assert out.isna().all().all(), name


def test_all_operators_handle_single_column() -> None:
    """All operators should handle single-column panels."""
    single = _daily_panel(days=60, cols=1, seed=100)
    event_single = _event_panel(days=60, cols=1, seed=101, event_prob=0.1)

    for name in BATCH1_CANONICALS:
        op = OperatorRegistry.get(name)
        if name == "panel_day_night_beta_gap":
            out = op.calculate(single, single, single, window=20, min_periods=10)
        elif name == "pastor_stambaugh_beta":
            out = op.calculate(single, single, single, window=30, min_periods=15)
        elif name == "price_delay_score":
            out = op.calculate(single, single, window=30, lag=2, min_periods=15)
        elif name == "report_asof":
            out = op.calculate(event_single, max_lookback=30)
        elif name == "event_window_return_asof":
            out = op.calculate(single, event_single, pre_window=2, post_window=3, max_lookback=30)
        else:
            continue

        assert out.shape == single.shape, name
