# -*- coding: utf-8 -*-
"""R23-094..097 / R23-298: first sample observation is left-censored, NOT age 0.

A backtest that starts mid-history sees a report that may have been published
weeks earlier — age=0 on the first row under-reports staleness.  Without a
knowledge_time the true publication time is unknown, so the first observation is
left-censored (NaN) until a genuinely OBSERVED update resets the clock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.fundamental.transforms_repairs_v2 import fin_days_since_update
from cleaned_operators.fundamental.expectation_v2 import fin_days_since_expectation_revision


def _panel(vals, periods):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="B")
    return pd.DataFrame({"A": vals}, index=idx), pd.DataFrame({"A": periods}, index=idx)


def test_days_since_update_first_observation_is_left_censored():
    x, pid = _panel([10.0, 10.0, 12.0, 12.0], ["2023Q1", "2023Q1", "2023Q2", "2023Q2"])
    out = fin_days_since_update(x, pid, max_days=504)
    vals = out["A"].to_numpy()
    assert np.isnan(vals[0]), "first observation must be left-censored NaN, not age 0"
    assert np.isnan(vals[1]), "rows before the first observed update stay NaN"
    assert vals[2] == 0.0, "first genuinely observed update resets the clock to 0"
    assert vals[3] == 1.0


def test_days_since_expectation_revision_first_observation_is_unknown_not_stale():
    x, pid = _panel([100.0, 100.0, 105.0, 105.0], ["2024Q1", "2024Q1", "2024Q1", "2024Q1"])
    out = fin_days_since_expectation_revision(x, pid, max_days=252)
    vals = out["A"].to_numpy()
    assert np.isnan(vals[0]), "no history = unknown (NaN), not max_days (stale)"
    assert vals[2] == 0.0, "first observed estimate revision resets to 0"


def test_provider_gap_recovery_does_not_reset_age():
    # A resumed value identical to the pre-gap value is NOT an update; the
    # elapsed unobservable days join the confirmed age (R23-105/106, R4-27).
    x, pid = _panel([10.0, np.nan, 10.0, 11.0], ["2023Q1", "2023Q1", "2023Q1", "2023Q2"])
    out = fin_days_since_update(x, pid, max_days=504)
    vals = out["A"].to_numpy()
    assert np.isnan(vals[0])  # left-censored first observation
    assert np.isnan(vals[1])  # missing row -> NaN
    assert np.isnan(vals[2]), "gap recovery identical to anchor is NOT an update"
    assert vals[3] == 0.0, "the value change is the first observed update"
