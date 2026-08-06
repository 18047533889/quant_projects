# -*- coding: utf-8 -*-
"""Parity tests for native Polars Keltner/Donchian/Bollinger/supertrend/PSAR."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(1234)
    index = pd.date_range("2022-01-01", periods=150, freq="D")
    close = pd.DataFrame(
        {
            "A": 100.0 + np.cumsum(rng.normal(0.1, 1.0, len(index))),
            "B": 50.0 + np.cumsum(rng.normal(-0.03, 0.8, len(index))),
        },
        index=index,
    )
    spread = pd.DataFrame(rng.uniform(0.5, 2.5, close.shape), index=index, columns=close.columns)
    high = close + spread
    low = close - spread
    return high, low, close


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def _assert_parity(name, args, kwargs, rtol=1e-8, atol=1e-8):
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    assert pandas_op is not None, f"{name} missing pandas"
    assert polars_op is not None, f"{name} missing polars"
    pandas_out = pandas_op.calculate(*args, **kwargs)
    polars_out = polars_op.calculate(*[_polars(arg) for arg in args], **kwargs)
    assert list(pandas_out.columns) == list(polars_out.columns)
    for column in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[column].to_numpy(),
            polars_out[column].to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_native_polars_tech_misc_matches_pandas(panels):
    high, low, close = panels
    cases = [
        ("KeltnerMid", (close,), {"ema_window": 20}),
        ("KeltnerUpper", (high, low, close), {"ema_window": 20, "atr_window": 14, "multiplier": 2.0}),
        ("KeltnerLower", (high, low, close), {"ema_window": 20, "atr_window": 14, "multiplier": 2.0}),
        ("KeltnerPosition", (high, low, close), {"ema_window": 20, "atr_window": 14, "multiplier": 2.0}),
        ("donchian_upper", (high,), {"window": 20}),
        ("donchian_lower", (low,), {"window": 20}),
        ("donchian_mid", (high, low), {"window": 20}),
        ("donchian_position", (close, high, low), {"window": 20}),
        ("bollinger_pct_b", (close,), {"window": 20, "std_dev": 2.0}),
        ("bollinger_width", (close,), {"window": 20, "std_dev": 2.0}),
        ("efficiency_ratio", (close,), {"window": 10}),
        ("choppiness_index", (high, low, close), {"window": 14}),
        ("Supertrend", (high, low, close), {"atr_window": 14, "multiplier": 3.0}),
        ("SupertrendDirection", (high, low, close), {"atr_window": 14, "multiplier": 3.0}),
        ("PSAR", (high, low), {"acceleration": 0.02, "maximum": 0.2}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)


def test_native_polars_tech_misc_nan_warmup_matches_pandas():
    rng = np.random.default_rng(77)
    index = pd.date_range("2023-01-01", periods=100, freq="D")
    mask = rng.random((100, 2)) < 0.1
    close = pd.DataFrame(np.cumsum(rng.normal(0.0, 1.0, (100, 2)), axis=0) + 100, index=index, columns=["A", "B"])
    close[mask] = np.nan
    spread = pd.DataFrame(rng.uniform(0.5, 2.0, close.shape), index=index, columns=close.columns)
    high = close + spread
    low = close - spread
    high[mask] = np.nan
    low[mask] = np.nan
    cases = [
        ("KeltnerMid", (close,), {"ema_window": 10}),
        ("KeltnerUpper", (high, low, close), {"ema_window": 10, "atr_window": 8, "multiplier": 2.0}),
        ("KeltnerPosition", (high, low, close), {"ema_window": 10, "atr_window": 8, "multiplier": 2.0}),
        ("donchian_upper", (high,), {"window": 10}),
        ("donchian_position", (close, high, low), {"window": 10}),
        ("bollinger_pct_b", (close,), {"window": 10, "std_dev": 2.0}),
        ("efficiency_ratio", (close,), {"window": 8}),
        ("choppiness_index", (high, low, close), {"window": 8}),
        ("Supertrend", (high, low, close), {"atr_window": 8, "multiplier": 3.0}),
        ("SupertrendDirection", (high, low, close), {"atr_window": 8, "multiplier": 3.0}),
        ("PSAR", (high, low), {"acceleration": 0.02, "maximum": 0.2}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)
