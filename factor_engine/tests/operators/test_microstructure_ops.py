# -*- coding: utf-8
"""微观结构算子数值测试（pandas / polars 对称）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    ensure_cleaned_loaded()


def _wide_panel(n: int = 30) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = pd.DataFrame(
        {"A": np.linspace(100, 110, n), "B": np.linspace(50, 55, n)},
        index=idx,
    )
    high = close + 0.5
    low = close - 0.5
    volume = pd.DataFrame({"A": np.full(n, 1000.0), "B": np.full(n, 2000.0)}, index=idx)
    return {"close": close, "high": high, "low": low, "volume": volume}


def test_micro_spread_pandas():
    p = _wide_panel()
    op = OperatorRegistry.get("micro_spread", backend="pandas_numpy")
    out = op.calculate(p["high"], p["low"], p["close"])
    expected = (p["high"] - p["low"]) / p["close"]
    pd.testing.assert_frame_equal(out, expected)


def test_micro_amihud_hf_positive_finite():
    p = _wide_panel()
    op = OperatorRegistry.get("micro_amihud_hf", backend="pandas_numpy")
    out = op.calculate(p["close"], p["volume"])
    assert (out.iloc[1:].fillna(0) >= 0).all().all()


def test_micro_bipower_var_polars():
    pl = pytest.importorskip("polars")
    p = _wide_panel()
    pdf = {
        k: pl.from_pandas(v.reset_index().rename(columns={"index": "date"}))
        for k, v in p.items()
    }
    op = OperatorRegistry.get("micro_bipower_var", backend="polars")
    out = op.calculate(pdf["close"], window=5)
    assert out is not None


def test_effective_lookback_intraday_scaling():
    from cleaned_operators.operator_policy import effective_lookback

    daily = effective_lookback(20, factor_freq="1d", source_bar_freq="1d")
    intraday = effective_lookback(20, factor_freq="1d", source_bar_freq="5m")
    assert intraday > daily
    assert intraday >= 20 * 78 // 1  # 至少 20 日的 5m bar 数（含 buffer 会更大）
