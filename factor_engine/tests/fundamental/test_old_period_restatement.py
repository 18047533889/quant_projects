# -*- coding: utf-8 -*-
"""R23-028..031 / R23-244: an old-period restatement must NOT regress the
current visible period.

``latest_visible_period`` stays anchored at the most recent period even when a
restated older period arrives; only a revision-detection factor looks at the
updated (older) period.  The strict fiscal kernels anchor lag/growth/TTM on the
current visible ordinal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.fundamental.transforms_v2 import fin_lag, fin_pct_change


def test_restated_old_period_does_not_move_current_anchor():
    # Visible order: Q3 known, then a restated Q1 arrives (same decision day),
    # then a revised Q3.  fin_lag(periods=1) must anchor on the latest visible
    # period (Q3 -> Q2 = 20), NOT treat the restated Q1 as the current period.
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    x = pd.DataFrame({"A": [10.0, 20.0, 30.0, 8.0, 40.0]}, index=idx)
    pid = pd.DataFrame({"A": ["2024Q1", "2024Q2", "2024Q3", "2024Q1", "2024Q3"]}, index=idx)
    out = fin_lag(x, pid, periods=1)
    # Row 3 is the restated Q1 — its own one-step lag is the prior fiscal year
    # Q4 (not in the visible window) -> NaN.
    assert np.isnan(out["A"].iloc[3]), "restated Q1 has no Q1-prior -> NaN"
    # Row 4 (latest visible = Q3) still lags to Q2 (20), NOT the restated Q1:
    # latest_visible_period stays anchored at Q3 (R23-028..031).
    assert out["A"].iloc[-1] == np.float64(20.0), "Q3 anchor must lag to Q2, not the restated Q1"


def test_growth_anchors_on_visible_ordinal_sequence():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    x = pd.DataFrame({"A": [10.0, 20.0, 8.0, 40.0]}, index=idx)
    pid = pd.DataFrame({"A": ["2024Q1", "2024Q2", "2024Q1", "2024Q3"]}, index=idx)
    out = fin_pct_change(x, pid, periods=1)
    # Q1->Q2 growth on row 1: (20-10)/10 = 1.0
    assert np.allclose(out["A"].iloc[1], 1.0, equal_nan=True)
    # Q2->Q3 growth on last row: (40-20)/20 = 1.0 (the restated Q1 row is
    # BETWEEN Q2 and Q3 ordinally, so the one-step lag from Q3 is Q2).
    assert np.allclose(out["A"].iloc[-1], 1.0, equal_nan=True)
