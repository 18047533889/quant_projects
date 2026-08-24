# -*- coding: utf-8 -*-
"""Parity tests for native Polars volatility / Ichimoku / ts_* structure ops."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(31415)
    index = pd.date_range("2022-01-01", periods=160, freq="D")
    open_ = pd.DataFrame(rng.uniform(5, 50, (160, 2)), index=index, columns=["A", "B"])
    close = open_ + pd.DataFrame(rng.normal(0, 2.5, (160, 2)), index=index, columns=["A", "B"])
    high = pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()), index=index, columns=["A", "B"]) + pd.DataFrame(rng.uniform(0.1, 2.0, (160, 2)), index=index, columns=["A", "B"])
    low = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()), index=index, columns=["A", "B"]) - pd.DataFrame(rng.uniform(0.1, 2.0, (160, 2)), index=index, columns=["A", "B"])
    return open_, high, low, close


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def _assert_parity(name, args, kwargs, rtol=1e-6, atol=1e-6):
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


def test_native_polars_misc_v2_matches_pandas(panels):
    o, h, l, c = panels
    cases = [
        ("parkinson_vol", (h, l), {"window": 20}),
        ("garman_klass_vol", (o, h, l, c), {"window": 20}),
        ("rogers_satchell_vol", (o, h, l, c), {"window": 20}),
        ("yang_zhang_vol", (o, h, l, c), {"window": 20}),
        ("overnight_volatility", (o, c), {"window": 20}),
        ("intraday_volatility", (o, c), {"window": 20}),
        ("range_volatility", (h, l, c), {"window": 20}),
        ("ulcer_index", (c,), {"window": 20}),
        ("ichimoku_tenkan", (h, l), {"tenkan_window": 9}),
        ("ichimoku_kijun", (h, l), {"kijun_window": 26}),
        ("ichimoku_senkou_a", (h, l), {"tenkan_window": 9, "kijun_window": 26}),
        ("ichimoku_senkou_b", (h, l), {"senkou_b_window": 52}),
        ("ichimoku_cloud_width", (h, l), {"tenkan_window": 9, "kijun_window": 26, "senkou_b_window": 52}),
        ("ichimoku_cloud_position", (h, l, c), {"tenkan_window": 9, "kijun_window": 26, "senkou_b_window": 52}),
        ("ts_prev_high", (c,), {"window": 10}),
        ("ts_prev_low", (c,), {"window": 10}),
        ("ts_distance_to_high", (c,), {"window": 10}),
        ("ts_distance_to_low", (c,), {"window": 10}),
        ("ts_breakout_high", (c,), {"window": 10}),
        ("ts_breakdown_low", (c,), {"window": 10}),
        ("ts_new_high", (c,), {"window": 10}),
        ("ts_new_low", (c,), {"window": 10}),
        ("ts_channel_position", (c,), {"window": 10}),
        ("ts_days_since_high", (c,), {"window": 10}),
        ("ts_days_since_low", (c,), {"window": 10}),
        ("ts_range_expansion", (h, l), {"window": 10}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)


def test_native_polars_misc_v2_nan_warmup_matches_pandas():
    rng = np.random.default_rng(88)
    index = pd.date_range("2023-01-01", periods=100, freq="D")
    mask = rng.random((100, 2)) < 0.1
    open_ = pd.DataFrame(rng.uniform(5, 50, (100, 2)), index=index, columns=["A", "B"])
    open_[mask] = np.nan
    close = open_ + pd.DataFrame(rng.normal(0, 2.5, (100, 2)), index=index, columns=["A", "B"])
    high = pd.DataFrame(np.maximum(np.nan_to_num(open_.to_numpy(), nan=-1e9), np.nan_to_num(close.to_numpy(), nan=-1e9)), index=index, columns=["A", "B"])
    low = pd.DataFrame(np.minimum(np.nan_to_num(open_.to_numpy(), nan=1e9), np.nan_to_num(close.to_numpy(), nan=1e9)), index=index, columns=["A", "B"])
    high[mask] = np.nan
    low[mask] = np.nan
    cases = [
        ("parkinson_vol", (high, low), {"window": 10}),
        ("garman_klass_vol", (open_, high, low, close), {"window": 10}),
        ("rogers_satchell_vol", (open_, high, low, close), {"window": 10}),
        ("yang_zhang_vol", (open_, high, low, close), {"window": 10}),
        ("overnight_volatility", (open_, close), {"window": 10}),
        ("intraday_volatility", (open_, close), {"window": 10}),
        ("range_volatility", (high, low, close), {"window": 10}),
        ("ulcer_index", (close,), {"window": 10}),
        ("ichimoku_tenkan", (high, low), {"tenkan_window": 8}),
        ("ichimoku_senkou_a", (high, low), {"tenkan_window": 8, "kijun_window": 12}),
        ("ichimoku_cloud_width", (high, low), {"tenkan_window": 8, "kijun_window": 12, "senkou_b_window": 20}),
        ("ichimoku_cloud_position", (high, low, close), {"tenkan_window": 8, "kijun_window": 12, "senkou_b_window": 20}),
        ("ts_prev_high", (close,), {"window": 8}),
        ("ts_distance_to_high", (close,), {"window": 8}),
        ("ts_breakout_high", (close,), {"window": 8}),
        ("ts_new_high", (close,), {"window": 8}),
        ("ts_channel_position", (close,), {"window": 8}),
        ("ts_days_since_high", (close,), {"window": 8}),
        ("ts_days_since_low", (close,), {"window": 8}),
        ("ts_range_expansion", (high, low), {"window": 8}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)
