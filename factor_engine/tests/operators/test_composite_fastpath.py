# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def _frames(n: int = 120):
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.linspace(80.0, 120.0, n)
    close = pd.DataFrame({"A": base + np.sin(np.arange(n) / 4), "B": base * 0.8 + np.cos(np.arange(n) / 7)}, index=index)
    high = close + 1.5
    low = close - 1.2
    open_ = close.shift(1).fillna(close) + 0.2
    volume = pd.DataFrame({"A": np.linspace(1000, 3000, n), "B": np.linspace(1500, 2600, n)}, index=index)
    return open_, high, low, close, volume


def test_fastpath_sources_are_active() -> None:
    for name in (
        "ATR_WILDER",
        "RSI_WILDER",
        "BollingerUpper",
        "StochasticD",
        "TRIX",
        "KAMA",
        "vpmacd",
        "vpmacd_signal",
        "volatility",
        "vwap",
    ):
        catalog = OperatorRegistry.catalog()[name]
        assert catalog["backend_meta"]["pandas_numpy"]["source"] == "composite_fastpath_primitives"
        if "polars" in OperatorRegistry.backends_for(name):
            assert catalog["backend_meta"]["polars"]["source"] == "composite_fastpath_native_polars"


def test_polars_fastpath_never_uses_pandas_bridge() -> None:
    import cleaned_operators.composite_fastpath as module

    source = inspect.getsource(module)
    assert ".to_pandas(" not in source
    assert "from_pandas" not in source


def test_williams_r_reuses_stochastic_identity() -> None:
    _, high, low, close, _ = _frames()
    k = OperatorRegistry.get("StochasticK").calculate(high, low, close, 14)
    wr = OperatorRegistry.get("WilliamsR").calculate(high, low, close, 14)
    pd.testing.assert_frame_equal(wr, k - 100.0)


def test_stochastic_d_is_three_period_mean_of_k() -> None:
    _, high, low, close, _ = _frames()
    k = OperatorRegistry.get("StochasticK").calculate(high, low, close, 14)
    expected = k.rolling(3, min_periods=1).mean()
    actual = OperatorRegistry.get("StochasticD").calculate(high, low, close, 14)
    pd.testing.assert_frame_equal(actual, expected)


def test_adxr_reuses_adx_and_delay() -> None:
    _, high, low, close, _ = _frames()
    adx = OperatorRegistry.get("ADX").calculate(high, low, close, 14)
    expected = (adx + adx.shift(14)) * 0.5
    actual = OperatorRegistry.get("ADXR").calculate(high, low, close, 14)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True, rtol=1e-10, atol=1e-10)


def test_vpmacd_weighted_price_is_computed_once(monkeypatch) -> None:
    import cleaned_operators.composite_fastpath as module

    open_, high, low, close, volume = _frames(80)
    calls = {"count": 0}
    original = module._pd_vp_weighted_price

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_pd_vp_weighted_price", wrapped)
    OperatorRegistry.get("vpmacd").calculate(close, volume, open_, high, low, 0.9)
    assert calls["count"] == 1


def test_vpmacd_signal_is_discrete() -> None:
    open_, high, low, close, volume = _frames(100)
    signal = OperatorRegistry.get("vpmacd_signal").calculate(close, volume, open_, high, low, 0.9)
    assert set(np.unique(signal.to_numpy())) <= {-1.0, 0.0, 1.0}


@pytest.mark.parametrize(
    "name,args",
    [
        ("ATR_WILDER", (1, 2, 3, 14)),
        ("RSI_WILDER", (3, 14)),
        ("BollingerUpper", (3, 20, 2.0)),
        ("StochasticD", (1, 2, 3, 14)),
        ("TRIX", (3, 12)),
        ("KAMA", (3, 10)),
        ("WMA", (3, 10)),
        ("volatility", (3, 20)),
        ("vwap", (3, 4, 20)),
    ],
)
def test_native_polars_parity(name, args) -> None:
    pl = pytest.importorskip("polars")
    open_, high, low, close, volume = _frames()
    frames = (open_, high, low, close, volume)
    pandas_args = [frames[value] if isinstance(value, int) and value < 5 else value for value in args]
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
