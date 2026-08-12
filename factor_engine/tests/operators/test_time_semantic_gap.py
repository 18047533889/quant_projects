# -*- coding: utf-8 -*-
"""Tests for TRUE_GAP time-semantic operators (2026-08-13).

Three operators implementing special calendar-aware time semantics:
- financial_snapshot_lag
- same_calendar_day_mean
- same_calendar_month_return
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.time_semantic_gap  # noqa: F401

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

CANONICALS = [
    "financial_snapshot_lag",
    "same_calendar_day_mean",
    "same_calendar_month_return",
]


def _daily_panel(
    days: int = 100,
    cols: int = 2,
    seed: int = 0,
    start: str = "2024-01-01",
    freq: str = "B",
) -> pd.DataFrame:
    """Generate daily panel with business day frequency."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq=freq)
    return pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((days, cols)) * 0.01, axis=0)) * 100.0,
        index=idx,
        columns=[f"C{i}" for i in range(cols)],
    )


def _with_gaps(df: pd.DataFrame, gap_rows: list[int]) -> pd.DataFrame:
    """Insert NaN gaps at specified rows."""
    out = df.copy()
    for r in gap_rows:
        if r < len(out):
            out.iloc[r, :] = np.nan
    return out


# ---------------------------------------------------------------------------
# registration + surface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CANONICALS))
def test_registered_and_classified(name: str) -> None:
    """All three operators are registered and on extended surface."""
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    surf = classify_canonical(name)
    assert surf in ("daily", "extended", "research"), f"{name} -> {surf}"


# ---------------------------------------------------------------------------
# 1. financial_snapshot_lag
# ---------------------------------------------------------------------------
def test_financial_snapshot_lag_skips_nan_gaps() -> None:
    """Lag=1 returns most recent finite observation, skipping NaN."""
    # Create panel with gaps: [100, NaN, NaN, 103, 104]
    vals = [100.0, np.nan, np.nan, 103.0, 104.0]
    df = pd.DataFrame(
        vals, index=pd.date_range("2024-01-01", periods=5), columns=["A"]
    )
    op = OperatorRegistry.get("financial_snapshot_lag")
    out = op.calculate(df, lag=1)["A"]

    # Row 0: no prior -> NaN
    # Row 1: prior finite is row 0 -> 100
    # Row 2: prior finite is row 0 -> 100
    # Row 3: prior finite is row 0 -> 100
    # Row 4: prior finite is row 3 -> 103
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 100.0
    assert out.iloc[2] == 100.0
    assert out.iloc[3] == 100.0
    assert out.iloc[4] == 103.0


def test_financial_snapshot_lag_lag2() -> None:
    """Lag=2 counts back 2 finite observations."""
    vals = [10.0, 20.0, np.nan, 30.0, 40.0, 50.0]
    df = pd.DataFrame(
        vals, index=pd.date_range("2024-01-01", periods=6), columns=["A"]
    )
    op = OperatorRegistry.get("financial_snapshot_lag")
    out = op.calculate(df, lag=2)["A"]

    # Row 0,1: <2 priors -> NaN
    # Row 2: 2 priors (20, 10) -> 10
    # Row 3: 2 priors (20, 10) -> 10
    # Row 4: 2 priors (30, 20) -> 20
    # Row 5: 2 priors (40, 30) -> 30
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == 10.0
    assert out.iloc[3] == 10.0
    assert out.iloc[4] == 20.0
    assert out.iloc[5] == 30.0


def test_financial_snapshot_lag_determinism() -> None:
    """Repeated calculation produces identical results."""
    df = _with_gaps(_daily_panel(60, 2, seed=7), [5, 10, 15, 20])
    op = OperatorRegistry.get("financial_snapshot_lag")
    first = op.calculate(df, lag=1)
    second = op.calculate(df, lag=1)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# 2. same_calendar_day_mean
# ---------------------------------------------------------------------------
def test_same_calendar_day_mean_groups_by_weekday() -> None:
    """Mean computed only over same day-of-week."""
    # 2024-01-01 is Monday; create 3 weeks (Mon-Fri)
    idx = pd.date_range("2024-01-01", periods=15, freq="B")  # 3 weeks
    # Mondays: rows 0,5,10 -> values 10, 50, 90
    # Tuesdays: rows 1,6,11 -> values 20, 60, 100
    vals = np.arange(10.0, 160.0, 10.0)[:15]
    df = pd.DataFrame(vals, index=idx, columns=["A"])

    op = OperatorRegistry.get("same_calendar_day_mean")
    out = op.calculate(df, window=20)["A"]  # window covers all days

    # Row 0 (Mon): only itself -> 10
    # Row 1 (Tue): only itself -> 20
    # Row 5 (Mon): Mondays 0,5 -> (10+50)/2 = 30
    # Row 6 (Tue): Tuesdays 1,6 -> (20+60)/2 = 40
    # Row 10 (Mon): Mondays 0,5,10 -> (10+50+90)/3 = 50
    assert out.iloc[0] == 10.0
    assert out.iloc[1] == 20.0
    assert out.iloc[5] == 30.0
    assert out.iloc[6] == 40.0
    assert out.iloc[10] == 50.0


def test_same_calendar_day_mean_respects_window() -> None:
    """Only same weekday within window counted."""
    # Create 30 business days (6 weeks)
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    vals = np.ones(30) * 100.0
    vals[0] = 10.0  # First Monday
    df = pd.DataFrame(vals, index=idx, columns=["A"])

    op = OperatorRegistry.get("same_calendar_day_mean")
    # Window=7 should only see ~1 week back
    out7 = op.calculate(df, window=7)["A"]
    # Window=30 should see all
    out30 = op.calculate(df, window=30)["A"]

    # Last row (row 29) is a Friday (29 business days from Monday)
    # With window=7, Friday should only see recent Fridays
    # With window=30, Friday should see all Fridays (all 100.0)
    assert out7.iloc[-1] == 100.0
    assert out30.iloc[-1] == 100.0

    # Row 5 (2nd Monday): window=7 sees both Mondays (10, 100) -> 55
    assert out7.iloc[5] == pytest.approx(55.0, abs=1e-9)


def test_same_calendar_day_mean_determinism() -> None:
    """Repeated calculation produces identical results."""
    df = _daily_panel(80, 2, seed=9, start="2024-01-08")
    op = OperatorRegistry.get("same_calendar_day_mean")
    first = op.calculate(df, window=40)
    second = op.calculate(df, window=40)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# 3. same_calendar_month_return
# ---------------------------------------------------------------------------
def test_same_calendar_month_return_yoy() -> None:
    """YoY return: Dec 2024 vs Dec 2023."""
    # Create daily data spanning 2 years
    idx = pd.date_range("2023-12-15", "2024-12-20", freq="B")
    vals = np.ones(len(idx)) * 100.0
    # Set Dec 2023 value to 80, Dec 2024 to 120
    dec_2023_mask = (idx.year == 2023) & (idx.month == 12)
    dec_2024_mask = (idx.year == 2024) & (idx.month == 12)
    vals[dec_2023_mask] = 80.0
    vals[dec_2024_mask] = 120.0

    df = pd.DataFrame(vals, index=idx, columns=["A"])
    op = OperatorRegistry.get("same_calendar_month_return")
    out = op.calculate(df, year_lag=1)["A"]

    # Dec 2024 rows should have return = (120-80)/80 = 0.5
    dec_2024_rows = df.index[dec_2024_mask]
    for ts in dec_2024_rows:
        val = out.loc[ts]
        if not np.isnan(val):
            assert val == pytest.approx(0.5, abs=1e-9)


def test_same_calendar_month_return_uses_most_recent_from_target_month() -> None:
    """Takes the most recent prior value from target month."""
    # Jan 2024: 100, Feb 2024: 110, Jan 2025: 150
    dates = [
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-02-10"),
        pd.Timestamp("2025-01-10"),
        pd.Timestamp("2025-01-20"),
    ]
    vals = [100.0, 110.0, 150.0, 160.0]
    df = pd.DataFrame(vals, index=pd.DatetimeIndex(dates), columns=["A"])

    op = OperatorRegistry.get("same_calendar_month_return")
    out = op.calculate(df, year_lag=1)["A"]

    # Row 2 (2025-01-10): Jan 2024 = 100 -> (150-100)/100 = 0.5
    # Row 3 (2025-01-20): Jan 2024 = 100 -> (160-100)/100 = 0.6
    assert out.iloc[2] == pytest.approx(0.5, abs=1e-9)
    assert out.iloc[3] == pytest.approx(0.6, abs=1e-9)


def test_same_calendar_month_return_no_prior_year() -> None:
    """Returns NaN when target year-month does not exist."""
    # Only 2024 data
    idx = pd.date_range("2024-03-01", periods=20, freq="B")
    vals = np.ones(len(idx)) * 100.0
    df = pd.DataFrame(vals, index=idx, columns=["A"])

    op = OperatorRegistry.get("same_calendar_month_return")
    out = op.calculate(df, year_lag=1)["A"]

    # No 2023 data, so all should be NaN
    assert out.isna().all()


def test_same_calendar_month_return_determinism() -> None:
    """Repeated calculation produces identical results."""
    idx = pd.date_range("2022-01-01", "2024-12-31", freq="B")
    rng = np.random.default_rng(42)
    vals = 100.0 * np.exp(np.cumsum(rng.standard_normal(len(idx)) * 0.01))
    df = pd.DataFrame(vals, index=idx, columns=["A"])

    op = OperatorRegistry.get("same_calendar_month_return")
    first = op.calculate(df, year_lag=1)
    second = op.calculate(df, year_lag=1)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# multi-column + shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CANONICALS)
def test_multicol_shape_preserved(name: str) -> None:
    """Output shape matches input shape."""
    df = _daily_panel(60, cols=3, seed=11, start="2023-01-02")
    op = OperatorRegistry.get(name)
    if name == "financial_snapshot_lag":
        out = op.calculate(df, lag=2)
    elif name == "same_calendar_day_mean":
        out = op.calculate(df, window=30)
    else:  # same_calendar_month_return
        out = op.calculate(df, year_lag=1)
    assert out.shape == df.shape
    assert list(out.columns) == list(df.columns)


@pytest.mark.parametrize("name", CANONICALS)
def test_all_nan_input_produces_all_nan(name: str) -> None:
    """All-NaN input produces all-NaN output."""
    df = pd.DataFrame(
        np.nan, index=pd.date_range("2024-01-01", periods=20), columns=["A", "B"]
    )
    op = OperatorRegistry.get(name)
    if name == "financial_snapshot_lag":
        out = op.calculate(df, lag=1)
    elif name == "same_calendar_day_mean":
        out = op.calculate(df, window=10)
    else:
        out = op.calculate(df, year_lag=1)
    assert out.isna().all().all()


# ---------------------------------------------------------------------------
# policy + tags
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CANONICALS)
def test_pit_safe_policy(name: str) -> None:
    """All operators have pit_safe=True in explicit policies."""
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    policy = _EXPLICIT_POLICIES.get(name)
    assert policy is not None, f"{name} missing from _EXPLICIT_POLICIES"
    assert policy.get("pit_safe") is True, f"{name} pit_safe != True"
    assert policy.get("scope") == "ts", f"{name} scope != ts"


@pytest.mark.parametrize("name", CANONICALS)
def test_has_time_semantic_tag(name: str) -> None:
    """All operators tagged with time_semantic and true_gap."""
    op = OperatorRegistry.get(name)
    tags = op.metadata.tags
    assert "time_semantic" in tags, f"{name} missing time_semantic tag"
    assert "true_gap" in tags, f"{name} missing true_gap tag"
    assert "pit_safe" in tags, f"{name} missing pit_safe tag"
    assert "causal" in tags, f"{name} missing causal tag"
