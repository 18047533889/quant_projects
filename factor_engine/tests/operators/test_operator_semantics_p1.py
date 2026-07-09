# -*- coding: utf-8
"""P1 算子语义 golden tests：rolling_beta、Wilder RSI/ATR、protected_div。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_rolling_beta_production_allowed(_loaded):
    from cleaned_operators.operator_spec import build_operator_spec

    spec = build_operator_spec("rolling_beta")
    assert spec is not None
    assert spec.allow_in_production is True
    assert spec.pit_safe is True


def test_rolling_beta_constant_ratio(_loaded):
    op = OperatorRegistry.get("rolling_beta")
    idx = pd.date_range("2024-01-01", periods=8)
    bench = pd.DataFrame({"M": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}, index=idx)
    ret = bench * 2.0
    ret.columns = ["A"]
    out = op.calculate(ret, bench, window=3, min_periods=2)
    assert out.iloc[-1, 0] == pytest.approx(2.0, rel=1e-6)


def test_rolling_beta_broadcasts_single_benchmark(_loaded):
    op = OperatorRegistry.get("rolling_beta")
    idx = pd.date_range("2024-01-01", periods=6)
    ret = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [2, 4, 6, 8, 10, 12]}, index=idx, dtype=float)
    bench = pd.DataFrame({"M": [1, 2, 3, 4, 5, 6]}, index=idx, dtype=float)
    out = op.calculate(ret, bench, window=3, min_periods=2)
    assert out.shape == ret.shape
    assert out.iloc[-1, 0] == pytest.approx(1.0, rel=1e-6)
    assert out.iloc[-1, 1] == pytest.approx(2.0, rel=1e-6)


def test_rolling_beta_to_market_requires_benchmark(_loaded):
    op = OperatorRegistry.get("rolling_beta_to_market")
    ret = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="rolling_beta"):
        op.calculate(ret, window=3)


def test_rsi_wilder_differs_from_sma_after_warmup(_loaded):
    rsi_sma = OperatorRegistry.get("RSI")
    rsi_w = OperatorRegistry.get("RSI_WILDER")
    close = pd.DataFrame({"A": [44, 44.5, 43, 44.5, 45, 46, 47, 46, 45, 46, 47, 48, 49, 50]})
    w = 14
    sma_out = rsi_sma.calculate(close, w)
    wilder_out = rsi_w.calculate(close, w)
    assert not np.isclose(sma_out.iloc[-1, 0], wilder_out.iloc[-1, 0], rtol=1e-3, equal_nan=True)


def test_atr_wilder_differs_from_sma(_loaded):
    atr_sma = OperatorRegistry.get("ATR")
    atr_w = OperatorRegistry.get("ATR_WILDER")
    idx = pd.date_range("2024-01-01", periods=30)
    rng = np.array([2, 5, 3, 8, 4, 6, 7, 3, 9, 4] * 3, dtype=float)
    close = pd.DataFrame({"A": 100 + np.cumsum(rng[:30])}, index=idx)
    high = close + pd.DataFrame({"A": [1, 3, 2, 4, 1, 2, 5, 2, 3, 4] * 3}, index=idx)
    low = close - pd.DataFrame({"A": [1, 2, 1, 3, 2, 1, 2, 3, 2, 1] * 3}, index=idx)
    w = 14
    sma_out = atr_sma.calculate(high, low, close, w)
    wilder_out = atr_w.calculate(high, low, close, w)
    assert not np.isclose(sma_out.iloc[-1, 0], wilder_out.iloc[-1, 0], rtol=1e-4, equal_nan=True)


def test_protected_div_zero_denominator(_loaded):
    op = OperatorRegistry.get("protected_div")
    x = pd.DataFrame({"A": [1.0, 2.0]})
    y = pd.DataFrame({"A": [0.0, 5.0]})
    out = op.calculate(x, y, epsilon=1e-12, default=0.0)
    assert out.iloc[0, 0] == pytest.approx(0.0)
    assert out.iloc[1, 0] == pytest.approx(0.4)


def test_zscore_cs_by_row(_loaded):
    op = OperatorRegistry.get("zscore")
    x = pd.DataFrame({"A": [1.0, 10.0], "B": [3.0, 30.0]})
    out = op.calculate(x)
    expected = (x.loc[x.index[0]] - x.loc[x.index[0]].mean()) / x.loc[x.index[0]].std(ddof=1)
    assert out.loc[x.index[0], "A"] == pytest.approx(expected["A"])
    assert out.loc[x.index[0], "B"] == pytest.approx(expected["B"])


def test_quarter_fiscal_q1_is_level(_loaded):
    op = OperatorRegistry.get("quarter")
    idx = pd.date_range("2024-01-01", periods=4, freq="QS")
    cum = pd.DataFrame({"A": [100.0, 250.0, 420.0, 600.0]}, index=idx)
    fq = pd.DataFrame({"A": [1, 2, 3, 4]}, index=idx)
    out = op.calculate(cum, fq)
    assert out.iloc[0, 0] == pytest.approx(100.0)
    assert out.iloc[1, 0] == pytest.approx(150.0)
    assert out.iloc[2, 0] == pytest.approx(170.0)
    assert out.iloc[3, 0] == pytest.approx(180.0)


def test_quarter_fiscal_rollover_q1(_loaded):
    op = OperatorRegistry.get("quarter")
    idx = pd.date_range("2024-01-01", periods=5, freq="QS")
    cum = pd.DataFrame({"A": [600.0, 120.0, 280.0, 450.0, 630.0]}, index=idx)
    fq = pd.DataFrame({"A": [4, 1, 2, 3, 4]}, index=idx)
    out = op.calculate(cum, fq)
    assert out.iloc[1, 0] == pytest.approx(120.0)


def test_production_core_includes_wilder_and_protected(_loaded):
    from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS

    for canon in ("RSI_WILDER", "ATR_WILDER", "protected_div", "rolling_beta"):
        assert canon in PRODUCTION_CORE_CANONICALS
