# -*- coding: utf-8 -*-
"""Round-13: unify A-share 1-minute session in intraday_agg with DataAccess.

DataAccess has standardised the A-share 1-minute session to
09:31--11:30 + 13:01--15:00 (240 bars, no lunch bar).  The legacy
``cleaned_operators/microstructure/intraday_agg.py`` hard-coded
09:30--11:30 + 13:00--15:00 (242 bars), which silently diverged from the
minute data supplied by DataAccess.  These tests lock the new session
constants and verify the aggregation honours the 09:31 / 13:01 boundaries
(09:30 auction bar and 13:00 bar are NOT trading bars).
"""
import numpy as np
import pandas as pd
import pytest

from cleaned_operators.microstructure import intraday_agg as ia


def _full_session_minutes() -> list[int]:
    """Minute-of-day for the unified 240-bar A-share session: 09:31..11:30 (120)
    + 13:01..15:00 (120)."""
    return list(range(571, 691)) + list(range(781, 901))


def test_session_constants_match_dataaccess():
    assert ia._MORNING == (571, 690)    # 09:31 .. 11:30
    assert ia._AFTERNOON == (781, 900)  # 13:01 .. 15:00


def test_session_bar_counts_sum_240():
    n_morning = ia._MORNING[1] - ia._MORNING[0] + 1
    n_afternoon = ia._AFTERNOON[1] - ia._AFTERNOON[0] + 1
    assert n_morning == 120
    assert n_afternoon == 120
    assert n_morning + n_afternoon == 240
    assert len(_full_session_minutes()) == 240


def test_seg_mask_counts_on_full_240_bar_grid():
    times = np.asarray(
        [pd.Timestamp("2024-01-02") + pd.Timedelta(minutes=m) for m in _full_session_minutes()],
        dtype="datetime64[ns]",
    )
    m_mask = ia._seg_mask(times, "morning")
    a_mask = ia._seg_mask(times, "afternoon")
    assert int(np.sum(m_mask)) == 120
    assert int(np.sum(a_mask)) == 120
    assert int(np.sum(m_mask | a_mask)) == 240
    assert int(np.sum(m_mask & a_mask)) == 0  # segments are disjoint


def test_auction_bar_and_1300_excluded():
    """09:30 (minute 570) and 13:00 (minute 780) are NOT trading bars in the
    240-bar session: a day with only those bars yields NaN, not a value."""
    morning = pd.DataFrame(
        {"A": [10.0, 10.5]},
        index=pd.DatetimeIndex([
            pd.Timestamp("2024-01-02 09:30"),  # 570, legacy only
            pd.Timestamp("2024-01-02 09:31"),  # 571, first real bar
        ]),
    )
    out = ia.IntraSegmentReturn()._calculate_series(morning, segment="morning")
    assert np.isnan(out.loc[pd.Timestamp("2024-01-02"), "A"])

    afternoon = pd.DataFrame(
        {"A": [10.0, 10.5]},
        index=pd.DatetimeIndex([
            pd.Timestamp("2024-01-02 13:00"),  # 780, legacy only
            pd.Timestamp("2024-01-02 13:01"),  # 781, first real bar
        ]),
    )
    out = ia.IntraSegmentReturn()._calculate_series(afternoon, segment="afternoon")
    assert np.isnan(out.loc[pd.Timestamp("2024-01-02"), "A"])


def test_segment_return_uses_0931_and_1301_boundaries():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 09:31"),  # 571, first morning bar
        pd.Timestamp("2024-01-02 10:00"),
        pd.Timestamp("2024-01-02 11:30"),  # 690, last morning bar
        pd.Timestamp("2024-01-02 13:01"),  # 781, first afternoon bar
        pd.Timestamp("2024-01-02 15:00"),  # 900, last afternoon bar
    ])
    close = pd.DataFrame({"A": [10.0, 10.5, 11.0, 11.05, 11.1]}, index=idx)
    out_morning = ia.IntraSegmentReturn()._calculate_series(close, segment="morning")
    assert out_morning.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(11.0 / 10.0 - 1.0, rel=1e-9)
    out_afternoon = ia.IntraSegmentReturn()._calculate_series(close, segment="afternoon")
    assert out_afternoon.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(11.1 / 11.05 - 1.0, rel=1e-9)


def test_lunch_gap_default_uses_1301_first_afternoon_bar():
    """Default lunch-gap: first afternoon OPEN at 13:01 vs last morning CLOSE at
    11:30.  A legacy 13:00 row must be ignored by the default '13:01' start."""
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 11:30"),  # 690, morning last close
        pd.Timestamp("2024-01-02 13:00"),  # 780, NOT a trading bar
        pd.Timestamp("2024-01-02 13:01"),  # 781, first afternoon bar
        pd.Timestamp("2024-01-02 14:00"),
    ])
    close = pd.DataFrame({"A": [10.0, np.nan, np.nan, np.nan]}, index=idx)
    open_px = pd.DataFrame({"A": [10.0, 10.99, 11.0, 11.1]}, index=idx)
    out = ia.IntraLunchGapReturn()._calculate_series(close, open_px)
    assert out.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(11.0 / 10.0 - 1.0, rel=1e-9)


def test_full_240_bar_day_feeds_daily_aggregator():
    """A full 240-bar session day feeds the daily aggregator and yields one
    finite scalar per (date, symbol), matching a manual realized-variance."""
    minutes = _full_session_minutes()
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02") + pd.Timedelta(minutes=m) for m in minutes])
    closes = np.linspace(10.0, 11.0, len(idx))
    panel = pd.DataFrame({"A": closes}, index=idx)
    out = ia.IntraRealizedVariance()._calculate_series(panel)
    assert len(out) == 1  # one daily scalar, not 240 rows
    v = out.loc[pd.Timestamp("2024-01-02"), "A"]
    assert np.isfinite(v) and v > 0.0
    r = np.diff(np.log(closes))
    assert v == pytest.approx(float(np.sum(r * r)), rel=1e-9)
