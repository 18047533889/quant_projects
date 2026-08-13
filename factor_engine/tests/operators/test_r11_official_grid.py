# -*- coding: utf-8
"""P0-10: the official session grid is NEVER inferred from the observed data.

The old ``_session_expected_grid`` computed the official session width and the
official close as the MODAL values over the observed sessions.  If the dataset
systematically drops the final minute of every session (239 bars instead of
240), the modal width becomes 239 and the modal close 14:58 — the incomplete
data then certifies itself as "completed".  The operators now take an explicit
``calendar`` (``runtime.session_calendar.SessionCalendar``) and the grid comes
from the calendar, never from a modal of the data; without a calendar they fail
closed (all NaN + one RuntimeWarning).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.intraday_session import IntradaySessionShapeNovelty
from cleaned_operators.intraday_activity_duration import IntradayActivityDurationCurvature
from runtime.session_calendar import SessionCalendar

# A-share regular session, bar_start minute labels: 09:30..11:29 + 13:00..14:59.
_ASHARE_CAL = SessionCalendar(
    market="CN", timestamp_convention="bar_start", bar_freq="1min"
)
_FULL_MODS = list(range(570, 690)) + list(range(780, 900))       # 240 bars, ends 14:59
_MISSING_LAST = list(range(570, 690)) + list(range(780, 899))    # 239 bars, ends 14:58


def _session_panel(day_mods_by_day):
    """Minute panel + integral session_id, one column, deterministic values."""
    rng = np.random.default_rng(0)
    idx, sids, vals = [], [], []
    for d, mods in enumerate(day_mods_by_day):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for m in mods:
            idx.append(day + pd.Timedelta(minutes=m))
            sids.append(float(d))
            vals.append(100.0 + 0.01 * m + d + rng.normal(0.0, 0.001))
    index = pd.DatetimeIndex(idx)
    x = pd.DataFrame({"S0": vals}, index=index)
    sid = pd.DataFrame({"S0": sids}, index=index)
    return x, sid


def _novelty(day_mods, **kw):
    x, sid = _session_panel(day_mods)
    out = IntradaySessionShapeNovelty()._calculate_series(
        x, sid, history_days=8, min_history_sessions=2, calendar=_ASHARE_CAL, **kw
    )
    return x, out["S0"].to_numpy(dtype=float)


def test_every_day_missing_last_minute_never_completes():
    """239-bar days must NOT self-certify: with a calendar the official close is
    14:59 (mod 899) and the expected grid is 240 slots, so every day is partial."""
    _, arr = _novelty([_MISSING_LAST] * 4)
    assert np.isnan(arr).all(), "self-certification: missing last minute became official"


def test_full_240bar_grid_completes_with_calendar():
    """Full 240-bar days + the same calendar DO complete and emit (control)."""
    x, arr = _novelty([_FULL_MODS] * 5)
    # days 2,3,4 have >= 2 completed historical sessions -> 3 finite emissions
    assert float(np.nansum(np.isfinite(arr))) == 3
    # ... and every emission sits on the OFFICIAL close minute (14:59 = mod 899)
    for row in np.flatnonzero(np.isfinite(arr)):
        minute_of_day = x.index[row].hour * 60 + x.index[row].minute
        assert minute_of_day == 899, "emission not at the official close minute"


def test_missing_only_the_official_close_minute_is_partial():
    """A day that stops exactly at 14:58 (close minus 1 minute) is partial — the
    calendar's official close (14:59) is not reached."""
    _, arr = _novelty([_FULL_MODS, _FULL_MODS, _FULL_MODS, _MISSING_LAST, _FULL_MODS])
    # day 3 (index 3) ends at 14:58 -> partial -> the last observed minute is NaN
    day3_end = len(_FULL_MODS) * 3 + len(_MISSING_LAST) - 1
    assert np.isnan(arr[day3_end])
    # a completed day with history still emits
    assert float(np.nansum(np.isfinite(arr))) >= 1


def test_without_calendar_fails_closed_with_warning():
    x, sid = _session_panel([_FULL_MODS] * 3)
    with pytest.warns(RuntimeWarning, match="calendar"):
        out = IntradaySessionShapeNovelty()._calculate_series(
            x, sid, history_days=8, min_history_sessions=2
        )
    assert np.isnan(out.to_numpy(dtype=float)).all()


def _constant_activity(mods_by_day, value: float = 1.0):
    idx, vals = [], []
    for d, mods in enumerate(mods_by_day):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for m in mods:
            idx.append(day + pd.Timedelta(minutes=m))
            vals.append(value)
    return pd.DataFrame({"S0": vals}, index=pd.DatetimeIndex(idx))


def test_activity_duration_missing_last_minute_self_certification_gone():
    """The modal-grid self-certification in intraday_activity_duration_curvature:
    239-bar days reindexed onto the calendar's 240-slot grid expose the missing
    last minute as NaN -> every day fails closed.  Full days are finite (0.0 for
    constant activity)."""
    missing = _constant_activity([_MISSING_LAST] * 3)
    out_missing = IntradayActivityDurationCurvature()._calculate_series(
        missing, buckets=5, calendar=_ASHARE_CAL
    )
    assert np.isnan(out_missing.to_numpy(dtype=float)).all(), (
        "239-bar days self-certified as complete by a modal grid"
    )

    full = _constant_activity([_FULL_MODS] * 3)
    out_full = IntradayActivityDurationCurvature()._calculate_series(
        full, buckets=5, calendar=_ASHARE_CAL
    )
    fin = out_full.to_numpy(dtype=float)
    assert float(np.nansum(np.isfinite(fin))) == 3
    assert np.allclose(fin, 0.0)  # constant activity -> flat duration curve


def test_activity_duration_without_calendar_fails_closed():
    x = _constant_activity([_FULL_MODS] * 2)
    with pytest.warns(RuntimeWarning, match="calendar"):
        out = IntradayActivityDurationCurvature()._calculate_series(x, buckets=5)
    assert np.isnan(out.to_numpy(dtype=float)).all()
