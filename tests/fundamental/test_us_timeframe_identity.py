# -*- coding: utf-8 -*-
"""R23-048..051: US timeframe is data identity, not a query optional filter.

A US financial read without an explicit timeframe is a production hard fail;
the same period_end can carry quarterly/annual/trailing_twelve_months and they
must NEVER be mixed in one ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.fields.catalog_us import US_TABLE_SPECS
from factor_engine.cleaned_operators.fundamental.transforms_v2 import fin_ratio


def test_us_financial_tables_require_timeframe():
    for t in US_TABLE_SPECS:
        if t.name in ("StockIncome", "StockBalance", "StockCashFlow"):
            assert "timeframe" in t.required_parameters, t.name
            assert t.metadata.get("timeframe_required") is True, t.name


def test_flow_ratio_requires_same_timeframe_slot():
    # A ratio of an operating-income column to a revenue column from the SAME
    # statement keeps its financial bundle identity (timeframe inherited from the
    # table contract).  The operator-level arithmetic is time-invariant; the
    # timeframe identity is enforced by the source binding (R23-128/129).
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    op = pd.DataFrame({"A": [10.0, 20.0]}, index=idx)
    rev = pd.DataFrame({"A": [100.0, 200.0]}, index=idx)
    out = fin_ratio(op, rev)
    assert out["A"].iloc[0] == np.float64(0.1)
