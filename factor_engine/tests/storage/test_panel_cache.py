# -*- coding: utf-8 -*-
"""panel_cache 键：不同 Series 不可共用同一 unstack 结果。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import series_to_panel
from factor_engine.backend.context import ExecutionContext
from factor_engine.cache.panel_cache import series_panel_cache_key
from tests.helpers import InMemorySeriesSource


def test_panel_cache_does_not_collide_across_columns():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 20.0, 11.0, 21.0, 12.0, 22.0], index=idx)
    low = close - 1.0
    src = InMemorySeriesSource(data={"close": close, "low": low})
    ctx = ExecutionContext(data_source=src, panel_cache={})
    p_close = series_to_panel(src.load_column("close"), ctx)
    p_low = series_to_panel(src.load_column("low"), ctx)
    assert (p_close - p_low).abs().max().max() == pytest.approx(1.0)


def test_panel_cache_key_dtype_distinguishes_int64_float64():
    """int64 和 float64 零数组的 tobytes() 相同，key 必须包含 dtype 以避免碰撞。"""
    idx = pd.date_range("2024-01-01", periods=3)
    s_int = pd.Series(np.zeros(3, dtype=np.int64), index=idx, name="x")
    s_flt = pd.Series(np.zeros(3, dtype=np.float64), index=idx, name="x")
    key_int = series_panel_cache_key(s_int)
    key_flt = series_panel_cache_key(s_flt)
    assert key_int != key_flt, "int64 和 float64 零数组必须有不同的 cache key"


def test_panel_cache_key_dtype_distinguishes_bool_int8():
    """bool([True, False]) 和 int8([1, 0]) 的 tobytes() 相同，dtype 必须纳入。"""
    idx = pd.date_range("2024-01-01", periods=2)
    s_bool = pd.Series([True, False], index=idx, name="y")
    s_i8 = pd.Series(np.array([1, 0], dtype=np.int8), index=idx, name="y")
    key_bool = series_panel_cache_key(s_bool)
    key_i8 = series_panel_cache_key(s_i8)
    assert key_bool != key_i8, "bool 和 int8 必须有不同的 cache key"


def test_panel_cache_key_dtype_distinguishes_int32_int64():
    """同值不同字长（int32 vs int64）也应有不同的 key，防止宽度混淆。"""
    idx = pd.RangeIndex(3)
    s_i32 = pd.Series(np.array([100, 200, 300], dtype=np.int32), index=idx)
    s_i64 = pd.Series(np.array([100, 200, 300], dtype=np.int64), index=idx)
    key_i32 = series_panel_cache_key(s_i32)
    key_i64 = series_panel_cache_key(s_i64)
    assert key_i32 != key_i64, "int32 和 int64 必须有不同的 cache key"


def test_panel_cache_semantic_isolation_int_vs_float():
    """真实场景回归：int64 ticker_code 和 float64 close 不可误用同一 panel。

    如果 panel_cache_key 不含 dtype，当两者值的字节表示碰巧相同时（如全零），
    会导致 unstack 后的 DataFrame 列类型错乱（ticker_code 本应是整数标识）。
    """
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    ticker = pd.Series([600000, 600001, 600000, 600001], index=idx, dtype=np.int64, name="ticker")
    price = pd.Series([10.5, 20.3, 11.0, 21.5], index=idx, dtype=np.float64, name="close")
    src = InMemorySeriesSource(data={"ticker": ticker, "close": price})
    ctx = ExecutionContext(data_source=src, panel_cache={})
    p_ticker = series_to_panel(src.load_column("ticker"), ctx)
    p_close = series_to_panel(src.load_column("close"), ctx)
    assert p_ticker.dtypes["A"] == np.int64, "ticker panel 必须是 int64"
    assert p_close.dtypes["A"] == np.float64, "price panel 必须是 float64"
    assert len(ctx.panel_cache) == 2, "两个不同 dtype 的 Series 应产生两个独立缓存项"
