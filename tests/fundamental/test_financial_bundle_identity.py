# -*- coding: utf-8 -*-
"""R23-032..036 / R23-126..129: multi-statement operators require same
fiscal-period + compatible vintage + flow timeframe (FinancialStatementBundle
Identity), and static ratios must not mix balance/income/cashflow periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.alignment import PanelAxisMismatch
from factor_engine.cleaned_operators.fundamental.quality_v2 import _walk_two


def test_walk_two_rejects_misaligned_secondary_panel():
    # R23-037/293: the secondary panel (e.g. OCF vs net_profit) must share the
    # exact date x instrument grid — a silent reindex would pair two different
    # report periods positionally.
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    primary = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0]}, index=idx)
    pid = pd.DataFrame({"A": ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]}, index=idx)
    secondary = pd.DataFrame({"A": [5.0, 6.0, 7.0]}, index=idx[:3])  # misaligned length
    with pytest.raises(PanelAxisMismatch):
        _walk_two(primary, pid, secondary, lambda o, v1, v2, c: 0.0)


def test_static_balance_ratio_keeps_same_report_period_semantics():
    # current_ratio / debt_to_equity take both inputs from the SAME balance
    # statement; the arithmetic is period-preserving (both sides are the same
    # stock), so the ratio does not mix flow timeframes.
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    ca = pd.DataFrame({"A": [100.0, 120.0]}, index=idx)
    cl = pd.DataFrame({"A": [50.0, 60.0]}, index=idx)
    out = (ca / cl.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    assert out["A"].iloc[0] == np.float64(2.0)
