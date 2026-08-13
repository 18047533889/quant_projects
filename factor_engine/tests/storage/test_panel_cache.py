# -*- coding: utf-8 -*-
"""panel_cache 键：不同 Series 不可共用同一 unstack 结果。"""
from __future__ import annotations

import pandas as pd
import pytest

from backend.cleaned_bridge import series_to_panel
from backend.context import ExecutionContext
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


def test_panel_cache_key_includes_series_dtype():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["A"]],
        names=["timestamp", "instrument"],
    )
    integer = pd.Series([1, 2], index=idx, name="value", dtype="int64")
    floating = integer.astype("float64")

    from cache.panel_cache import series_panel_cache_key

    assert series_panel_cache_key(integer) != series_panel_cache_key(floating)

