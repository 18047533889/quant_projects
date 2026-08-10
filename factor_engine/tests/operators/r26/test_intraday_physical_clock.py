# -*- coding: utf-8 -*-
"""R26-013..039: SessionPanel physical-clock canonicalization.

* bar width from DECLARED calendar (never observed deltas);
* physically absent minutes -> explicit missing slots;
* gated log returns (no cross-gap fusion);
* coverage by unique valid official slots; duplicate slot = DQ fail;
* exact endpoints for segment / lunch gap;
* slot-ordinal high/low time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.session_calendar import SessionCalendar
from runtime.session_panel import build_session_panel


@pytest.fixture(scope="module")
def ashare_cal():
    return SessionCalendar.ashare(bar_freq="1min", timestamp_convention="bar_end")


def test_physically_absent_minute_is_explicit_missing_slot(ashare_cal):
    ts = pd.to_datetime(["2024-01-02 09:31", "2024-01-02 09:33", "2024-01-02 09:35"])
    panel = build_session_panel(
        ts.to_numpy(), np.array([100.0, 101.0, 102.0]), ashare_cal,
        market="ashare", session_timezone="Asia/Shanghai",
    )
    assert panel.n_slots == 240
    assert np.isfinite(panel.values[0]) and np.isfinite(panel.values[2]) and np.isfinite(panel.values[4])
    assert np.isnan(panel.values[1])  # 09:32 absent -> explicit NaN
    # R26-020: no cross-gap log return
    assert np.all(np.isnan(panel.log_returns()[:5]))


def test_duplicate_official_slot_is_dq(ashare_cal):
    ts = pd.to_datetime(["2024-01-02 09:31", "2024-01-02 09:31", "2024-01-02 09:32"])
    panel = build_session_panel(
        ts.to_numpy(), np.array([100.0, 100.5, 101.0]), ashare_cal,
        market="ashare", session_timezone="Asia/Shanghai",
    )
    assert bool(panel.is_duplicate[0]) is True
    assert bool(panel.is_valid_bar[0]) is False


def test_exact_endpoint_policy(ashare_cal):
    ts = pd.to_datetime(["2024-01-02 09:31", "2024-01-02 11:30"])
    panel = build_session_panel(
        ts.to_numpy(), np.array([100.0, 110.0]), ashare_cal,
        market="ashare", session_timezone="Asia/Shanghai",
    )
    assert panel.value_at_minute(571) == 100.0   # 09:31
    assert panel.value_at_minute(690) == 110.0   # 11:30
    assert np.isnan(panel.value_at_minute(572))  # absent 09:32


def test_tz_aware_utc_converts_to_session_day(ashare_cal):
    # 01:31 UTC = 09:31 Beijing
    ts = pd.to_datetime(["2024-01-02 01:31"]).tz_localize("UTC")
    panel = build_session_panel(
        ts.to_numpy(), np.array([100.0]), ashare_cal,
        market="ashare", session_timezone="Asia/Shanghai", source_timezone="UTC",
    )
    assert panel.n_slots == 240
    assert np.isfinite(panel.values[0])


def test_bar_width_from_declared_not_deltas():
    # A 1-min source dropping every other bar (deltas of 2) must NOT re-certify
    # itself as a 2-min source: declared 1-min grid -> coverage is low.
    cal = SessionCalendar.ashare(bar_freq="1min", timestamp_convention="bar_end")
    ts = pd.to_datetime(["2024-01-02 09:31", "2024-01-02 09:33", "2024-01-02 09:35"])
    panel = build_session_panel(
        ts.to_numpy(), np.array([100.0, 101.0, 102.0]), cal,
        market="ashare", session_timezone="Asia/Shanghai",
    )
    assert panel.n_slots == 240  # declared 1-min grid
    assert panel.coverage() == pytest.approx(3.0 / 240.0)
