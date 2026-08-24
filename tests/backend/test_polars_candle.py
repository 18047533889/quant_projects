# -*- coding: utf-8 -*-
"""Parity tests for native Polars candle geometry and pattern operators."""
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
    rng = np.random.default_rng(777)
    index = pd.date_range("2022-01-01", periods=100, freq="D")
    open_ = pd.DataFrame(rng.uniform(5, 50, (100, 2)), index=index, columns=["A", "B"])
    close = open_ + pd.DataFrame(rng.normal(0, 2.5, (100, 2)), index=index, columns=["A", "B"])
    high = pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()), index=index, columns=["A", "B"]) + pd.DataFrame(rng.uniform(0.1, 2.0, (100, 2)), index=index, columns=["A", "B"])
    low = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()), index=index, columns=["A", "B"]) - pd.DataFrame(rng.uniform(0.1, 2.0, (100, 2)), index=index, columns=["A", "B"])
    return open_, high, low, close


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


_OHLC = (0, 1, 2, 3)  # open, high, low, close


@pytest.mark.parametrize(
    ("name", "inputs", "kwargs"),
    [
        ("candle_body", (0, 3), {}),
        ("candle_abs_body", (0, 3), {}),
        ("candle_range", (1, 2), {}),
        ("candle_body_ratio", _OHLC, {}),
        ("candle_upper_shadow", (0, 1, 3), {}),
        ("candle_lower_shadow", (0, 2, 3), {}),
        ("candle_upper_shadow_ratio", _OHLC, {}),
        ("candle_lower_shadow_ratio", _OHLC, {}),
        ("candle_close_location", (1, 2, 3), {}),
        ("candle_gap", (0, 3), {}),
        ("candle_gap_pct", (0, 3), {}),
        ("candle_direction", (0, 3), {}),
        ("candle_range_atr", _OHLC, {"window": 10}),
        ("cdl_doji", _OHLC, {}),
        ("cdl_hammer", _OHLC, {}),
        ("cdl_inverted_hammer", _OHLC, {}),
        ("cdl_shooting_star", _OHLC, {}),
        ("cdl_marubozu", _OHLC, {}),
        ("cdl_spinning_top", _OHLC, {}),
        ("cdl_engulfing", _OHLC, {}),
        ("cdl_inside_bar", _OHLC, {}),
        ("cdl_outside_bar", _OHLC, {}),
        ("candle_body_zscore", (0, 3), {"window": 15}),
        ("candle_range_zscore", (1, 2), {"window": 15}),
        ("candle_upper_shadow_zscore", (0, 1, 3), {"window": 15}),
        ("candle_lower_shadow_zscore", (0, 2, 3), {"window": 15}),
        ("candle_body_percentile", (0, 3), {"window": 15}),
        ("candle_range_percentile", (1, 2), {"window": 15}),
        ("candle_gap_atr", _OHLC, {"atr_window": 10}),
        ("candle_body_position", _OHLC, {}),
        ("candle_overlap_ratio", (1, 2), {}),
        ("candle_inside_ratio", (1, 2), {}),
        ("candle_close_strength", (1, 2, 3), {}),
        ("candle_rejection_upper", _OHLC, {}),
        ("candle_rejection_lower", _OHLC, {}),
        ("cdl_dragonfly_doji", _OHLC, {}),
        ("cdl_gravestone_doji", _OHLC, {}),
        ("cdl_hanging_man", _OHLC, {}),
        ("cdl_harami", _OHLC, {}),
        ("cdl_harami_cross", _OHLC, {}),
        ("cdl_piercing", _OHLC, {}),
        ("cdl_dark_cloud_cover", _OHLC, {}),
        ("cdl_morning_star", _OHLC, {}),
        ("cdl_evening_star", _OHLC, {}),
        ("cdl_three_white_soldiers", _OHLC, {}),
        ("cdl_three_black_crows", _OHLC, {}),
        ("cdl_tweezer_top", _OHLC, {}),
        ("cdl_tweezer_bottom", _OHLC, {}),
    ],
)
def test_native_polars_candle_matches_pandas(panels, name, inputs, kwargs):
    _assert_parity(name, tuple(panels[index] for index in inputs), kwargs)


def test_native_polars_candle_nan_warmup_matches_pandas():
    rng = np.random.default_rng(31)
    index = pd.date_range("2023-01-01", periods=80, freq="D")
    mask = rng.random((80, 2)) < 0.10
    open_ = pd.DataFrame(rng.uniform(5, 50, (80, 2)), index=index, columns=["A", "B"])
    open_[mask] = np.nan
    close = open_ + pd.DataFrame(rng.normal(0, 2.5, (80, 2)), index=index, columns=["A", "B"])
    high = pd.DataFrame(np.maximum(np.nan_to_num(open_.to_numpy(), nan=-1e9), np.nan_to_num(close.to_numpy(), nan=-1e9)), index=index, columns=["A", "B"])
    low = pd.DataFrame(np.minimum(np.nan_to_num(open_.to_numpy(), nan=1e9), np.nan_to_num(close.to_numpy(), nan=1e9)), index=index, columns=["A", "B"])
    # 保证 high >= open/close、low <= open/close（NaN 参与比较则让整体 NaN 或单调化）
    high[mask] = np.nan
    low[mask] = np.nan
    for name, args, kwargs in (
        ("candle_body", (open_, close), {}),
        ("candle_abs_body", (open_, close), {}),
        ("candle_range", (high, low), {}),
        ("candle_body_ratio", (open_, high, low, close), {}),
        ("candle_upper_shadow", (open_, high, close), {}),
        ("candle_lower_shadow", (open_, low, close), {}),
        ("candle_upper_shadow_ratio", (open_, high, low, close), {}),
        ("candle_lower_shadow_ratio", (open_, high, low, close), {}),
        ("candle_close_location", (high, low, close), {}),
        ("candle_gap", (open_, close), {}),
        ("candle_gap_pct", (open_, close), {}),
        ("candle_direction", (open_, close), {}),
        ("candle_range_atr", (open_, high, low, close), {"window": 5}),
        ("cdl_doji", (open_, high, low, close), {}),
        ("cdl_hammer", (open_, high, low, close), {}),
        ("cdl_inverted_hammer", (open_, high, low, close), {}),
        ("cdl_shooting_star", (open_, high, low, close), {}),
        ("cdl_marubozu", (open_, high, low, close), {}),
        ("cdl_spinning_top", (open_, high, low, close), {}),
        ("cdl_engulfing", (open_, high, low, close), {}),
        ("cdl_inside_bar", (open_, high, low, close), {}),
        ("cdl_outside_bar", (open_, high, low, close), {}),
        ("candle_body_zscore", (open_, close), {"window": 8}),
        ("candle_range_zscore", (high, low), {"window": 8}),
        ("candle_upper_shadow_zscore", (open_, high, close), {"window": 8}),
        ("candle_lower_shadow_zscore", (open_, low, close), {"window": 8}),
        ("candle_body_percentile", (open_, close), {"window": 8}),
        ("candle_range_percentile", (high, low), {"window": 8}),
        ("candle_gap_atr", (open_, high, low, close), {"atr_window": 5}),
        ("candle_body_position", (open_, high, low, close), {}),
        ("candle_overlap_ratio", (high, low), {}),
        ("candle_inside_ratio", (high, low), {}),
        ("candle_close_strength", (high, low, close), {}),
        ("candle_rejection_upper", (open_, high, low, close), {}),
        ("candle_rejection_lower", (open_, high, low, close), {}),
        ("cdl_dragonfly_doji", (open_, high, low, close), {}),
        ("cdl_gravestone_doji", (open_, high, low, close), {}),
        ("cdl_hanging_man", (open_, high, low, close), {}),
        ("cdl_harami", (open_, high, low, close), {}),
        ("cdl_harami_cross", (open_, high, low, close), {}),
        ("cdl_piercing", (open_, high, low, close), {}),
        ("cdl_dark_cloud_cover", (open_, high, low, close), {}),
        ("cdl_morning_star", (open_, high, low, close), {}),
        ("cdl_evening_star", (open_, high, low, close), {}),
        ("cdl_three_white_soldiers", (open_, high, low, close), {}),
        ("cdl_three_black_crows", (open_, high, low, close), {}),
        ("cdl_tweezer_top", (open_, high, low, close), {}),
        ("cdl_tweezer_bottom", (open_, high, low, close), {}),
    ):
        _assert_parity(name, args, kwargs)
