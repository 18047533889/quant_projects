# -*- coding: utf-8 -*-
"""算子边界测试：inf、极短窗、零方差。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


def _op(name: str):
    op = OperatorRegistry.get(name)
    assert op is not None
    return op


def _panel(values, col: str = "A") -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({col: list(values)}, index=idx)


def test_ts_mean_short_window():
    x = _panel([1.0, 2.0, 3.0])
    out = _op("ts_mean").calculate(x, d=3)
    # min_periods=1：短窗仍产出值；第 0 点为单样本均值
    assert out.iloc[0]["A"] == pytest.approx(1.0)
    assert out.iloc[2]["A"] == pytest.approx(2.0)


def test_normalize_zero_range_returns_nan():
    x = _panel([5.0, 5.0, 5.0])
    out = _op("normalize").calculate(x)
    assert out.isna().all().all()


def test_zscore_inf_propagates():
    x = _panel([1.0, np.inf, 3.0])
    out = _op("zscore").calculate(x)
    assert np.isnan(out.iloc[1]["A"]) or np.isinf(out.iloc[1]["A"])


def test_winsorize_single_observation():
    x = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=pd.date_range("2020-01-01", periods=1))
    out = _op("winsorize").calculate(x)
    assert len(out) == 1
