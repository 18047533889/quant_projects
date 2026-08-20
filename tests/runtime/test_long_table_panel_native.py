# -*- coding: utf-8 -*-
"""长表模式 + panel-native：避免逐算子 stack/unstack。"""
from __future__ import annotations

import pandas as pd
import pytest

from backend.context import ExecutionContext
from backend.panel_native import panel_native_enabled
from storage.long_table_source import LongTableDataSource
from tests.helpers import InMemorySeriesSource


@pytest.fixture
def long_table_source():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4, freq="D"), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=idx)
    inner = InMemorySeriesSource(data={"close": close})
    return LongTableDataSource(inner)


def test_panel_native_enabled_with_long_table(long_table_source):
    ctx = ExecutionContext(data_source=long_table_source, prefer_long_table=True)
    assert panel_native_enabled(ctx) is True


def test_long_table_load_column_panel_cached(long_table_source):
    p1 = long_table_source.load_column_panel("close")
    p2 = long_table_source.load_column_panel("close")
    assert p1 is p2
    assert list(p1.columns) == ["A", "B"]
    assert len(p1) == 4
