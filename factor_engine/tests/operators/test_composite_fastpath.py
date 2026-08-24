# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def _frames(n: int = 120):
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.linspace(80.0, 120.0, n)
    close = pd.DataFrame({"A": base + np.sin(np.arange(n) / 4), "B": base * 0.8 + np.cos(np.arange(n) / 7)}, index=index)
    high = close + 1.5
    low = close - 1.2
    return high, low, close


def test_only_fused_composite_fastpaths_remain_active() -> None:
    for name in ("ATR_WILDER", "RSI_WILDER"):
        catalog = OperatorRegistry.catalog()[name]
        assert catalog["backend_meta"]["pandas_numpy"]["source"] == "composite_fastpath_primitives"
        if "polars" in OperatorRegistry.backends_for(name):
            assert catalog["backend_meta"]["polars"]["source"] == "composite_fastpath_native_polars"

    # KAMA is intentionally excluded: it has no factor_recipes replacement and
    # is registered as a real stateful operator via technical_indicators_v2
    # (extended canonical), unlike the other names below which are simple
    # recipe-replaced or deleted canonicals.
    for recipe_name in ("BollingerUpper", "StochasticD", "TRIX", "vpmacd", "volatility", "vwap"):
        assert OperatorRegistry.get(recipe_name) is None


def test_polars_fastpath_never_uses_pandas_bridge() -> None:
    import factor_engine.cleaned_operators.composite_fastpath as module

    source = inspect.getsource(module)
    assert ".to_pandas(" not in source
    assert "from_pandas" not in source


@pytest.mark.parametrize(
    "name,args",
    [
        ("ATR_WILDER", (0, 1, 2, 14)),
        ("RSI_WILDER", (2, 14)),
        ("ADX", (0, 1, 2, 14)),
        ("MACD_line", (2, 12, 26)),
    ],
)
def test_native_polars_parity_for_retained_composites(name, args) -> None:
    pl = pytest.importorskip("polars")
    high, low, close = _frames()
    frames = (high, low, close)
    pandas_args = [frames[value] if isinstance(value, int) and value < 3 else value for value in args]
    polars_args = [
        pl.DataFrame({c: value[c].to_numpy() for c in value.columns}) if isinstance(value, pd.DataFrame) else value
        for value in pandas_args
    ]
    expected = OperatorRegistry.get(name, backend="pandas_numpy").calculate(*pandas_args)
    actual = OperatorRegistry.get(name, backend="polars").calculate(*polars_args)
    np.testing.assert_allclose(
        actual.select(expected.columns.tolist()).to_numpy(),
        expected.to_numpy(),
        equal_nan=True,
        rtol=1e-7,
        atol=1e-7,
    )
