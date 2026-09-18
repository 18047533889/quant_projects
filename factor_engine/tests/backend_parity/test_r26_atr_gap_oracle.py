"""Wilder ATR long emitter must preserve missing-observation state semantics."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource
from tests.backend_parity.test_r24_rsi_gap_oracle import _ewm


@pytest.mark.parametrize("backend", ["pandas", "polars_long"])
@pytest.mark.parametrize("canonical", ["ATR_WILDER", "atr_pct", "atr_acceleration", "atr_short_long_ratio"])
@pytest.mark.parametrize("style", ["positional", "keyword"])
def test_atr_partial_ohlc_gaps_against_recursive_oracle(backend, canonical, style):
    dates = pd.date_range("2026-01-01", periods=24)
    assets = ["A", "B"]
    t = np.arange(24, dtype=float)
    close = 100 + t[:, None] + 3 * np.sin(t[:, None] + np.arange(2))
    high, low = close + 2, close - 1
    high[[9, 23], 0] = np.nan
    low[14, 1] = np.nan
    close[[0, 17], 0] = np.nan
    index = pd.MultiIndex.from_product([dates, assets], names=["timestamp", "instrument"])

    def execute(h, l, c):
        source = InMemorySeriesSource(data={
            name: pd.Series(values.ravel(), index=index)
            for name, values in (("high", h), ("low", l), ("close", c))
        })
        panels = [col("high"), col("low"), col("close")]
        scalars = {"short_window": 3, "long_window": 7} if canonical == "atr_short_long_ratio" else {"window": 3}
        expr = (F(canonical)(*panels, **scalars) if style == "keyword"
                else F(canonical)(*panels, *scalars.values()))
        result = FactorEngine(
            backend=build_backend(backend), data_source=source, run_mode="research",
        ).run(Factor(name="atr_gap", expr=expr))
        if backend == "polars_long":
            assert result.get("used_polars_long_path") is True
        return result["result"].reindex(index).to_numpy().reshape(24, 2)

    previous = np.vstack([np.full((1, 2), np.nan), close[:-1]])
    tr = np.maximum(np.maximum(high - low, np.abs(high - previous)), np.abs(low - previous))
    expected = np.column_stack([_ewm(tr[:, j], 3) for j in range(2)])
    if canonical != "ATR_WILDER":
        expected = expected / np.where(close > 0, close, np.nan)
    if canonical == "atr_acceleration":
        expected = np.vstack([np.full((1, 2), np.nan), np.diff(expected, axis=0)])
    elif canonical == "atr_short_long_ratio":
        longer = np.column_stack([_ewm(tr[:, j], 7) for j in range(2)])
        longer = longer / np.where(close > 0, close, np.nan)
        expected = expected / np.where(longer != 0, longer, np.nan)
    actual = execute(high, low, close)
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)
    future_h, future_l, future_c = high.copy(), low.copy(), close.copy()
    for values in (future_h, future_l, future_c):
        values[20:] *= 2.0
    mutated = execute(future_h, future_l, future_c)
    np.testing.assert_allclose(actual[:20], mutated[:20], equal_nan=True, rtol=1e-12, atol=1e-12)
