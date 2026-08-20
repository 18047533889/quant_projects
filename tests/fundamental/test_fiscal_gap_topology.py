# -*- coding: utf-8 -*-
"""R23-073..075 / R23-070..072: AR(1) persistence / smoothness / volatility and
period-growth math require CONSECUTIVE fiscal periods.

Q1 -> Q3 is NOT a one-step lag; a skipped report must fail the window closed
(CONSECUTIVE_REQUIRED default) instead of substituting a non-adjacent quarter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.registry import OperatorRegistry


def _panel(vals, periods):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="B")
    return pd.DataFrame({"A": vals}, index=idx), pd.DataFrame({"A": periods}, index=idx)


def _calc(canonical, *args, **kw):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, canonical
    return op.calculate(*args, **kw)


def test_persistence_fails_closed_on_skipped_quarter():
    # periods Q1, Q3 (Q2 missing) — an AR(1) over a skipped quarter must NOT
    # bridge the gap as if it were a one-step lag.
    x, pid = _panel([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    ["2023Q1", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"])
    out = _calc("fin_earnings_persistence", x, pid, periods=8)
    assert np.isnan(out["A"].iloc[-1]), "AR(1) must fail closed on a fiscal gap"


def test_persistence_consecutive_sequence_is_finite():
    x, pid = _panel([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4"])
    out = _calc("fin_earnings_persistence", x, pid, periods=8)
    assert np.isfinite(out["A"].iloc[-1]), "a consecutive window must produce a value"


def test_gap_volatility_fails_closed_on_skipped_quarter():
    x1, pid = _panel([10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                     ["2023Q1", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3"])
    x2, _ = _panel([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], ["2023Q1"] * 6)
    x3, _ = _panel([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], ["2023Q1"] * 6)
    # params: net_profit, ocf, avg_assets, period_id, periods, flow_type
    out = _calc("fin_earnings_cash_gap_volatility", x1, x2, x3, pid, 8)
    assert np.isnan(out["A"].iloc[-1]), "gap volatility must fail closed on a fiscal gap"


def test_smoothness_fails_closed_on_skipped_quarter():
    x1, pid = _panel([10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                     ["2023Q1", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3"])
    x2, _ = _panel([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], ["2023Q1"] * 6)
    # params: net_profit, ocf, period_id, periods, flow_type
    out = _calc("fin_earnings_smoothness", x1, x2, pid, 8)
    assert np.isnan(out["A"].iloc[-1]), "smoothness must fail closed on a fiscal gap"


def test_growth_rejects_skipped_quarter_lag():
    # fin_growth with periods=1 on Q1->Q3 must be NaN (no adjacent prior).
    x, pid = _panel([10.0, 20.0, 30.0], ["2023Q1", "2023Q3", "2023Q4"])
    # params: x, period_id, periods, flow_type
    out = _calc("fin_growth", x, pid, 1)
    assert np.isnan(out["A"].iloc[1]), "Q1->Q3 is not a one-step growth"
    assert np.isfinite(out["A"].iloc[2]), "Q3->Q4 adjacent growth is valid"
