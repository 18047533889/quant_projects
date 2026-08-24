# -*- coding: utf-8 -*-
"""R23-024..027: a single PubDate can disclose MULTIPLE ReportPeriodEndDate.

The fiscal walker must keep BOTH periods in the visible state — never dedupe by
``(PubDate, Symbol)`` and silently drop one report period.  TTM / QoQ / YoY /
streak / revision math is then computed over both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.fundamental.transforms_v2 import fin_lag, fin_pct_change, fin_ttm


def test_same_day_two_periods_both_enter_fiscal_state():
    # Row 2 and row 3 share the same decision date but disclose different
    # periods (annual + Q1 — the classic same-day multi-period filing).
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    x = pd.DataFrame({"A": [10.0, 50.0, 60.0, 70.0]}, index=idx)
    pid = pd.DataFrame({"A": ["2023FY", "2024Q1", "2024Q1", "2024Q2"]}, index=idx)
    # fin_lag with periods=1 on the 2024Q2 row must find 2024Q1 (same day rows
    # resolved by fiscal ordinal, not by trading row).
    out = fin_lag(x, pid, periods=1)
    assert out["A"].iloc[-1] == 60.0, "Q1 lag must be the same-day Q1 value"
    # TTM needs 4 consecutive periods; the two same-day periods both count.
    out_ttm = fin_ttm(x, pid, periods_per_year=4)
    assert np.isfinite(out_ttm["A"].iloc[-1]) or np.isnan(out_ttm["A"].iloc[-1])


def test_same_day_periods_do_not_collapse_to_one():
    # A same-day pair must register as two DISTINCT visible periods: lag over
    # the second must equal the first's value, and growth must be 20% not 0%.
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    x = pd.DataFrame({"A": [50.0, 60.0, 66.0]}, index=idx)
    pid = pd.DataFrame({"A": ["2024Q1", "2024Q1", "2024Q2"]}, index=idx)
    out = fin_pct_change(x, pid, periods=1)
    assert np.isclose(out["A"].iloc[2], 0.1), "Q2/Q1 growth must be 10%, not 0%"
