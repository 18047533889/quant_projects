# -*- coding: utf-8 -*-
"""P0-012: ShardableView 统一切片协议 —— 不再靠 __getattr__ 猜接口透传。

验证：
    - SliceDataSource 显式投影 load_column / load_column_panel / load_columns；
    - 未知数据方法 / 数据读取属性 fail-loud（AttributeError），不静默透传；
    - SliceCache 完整 Mapping 语义（iter/items/values/len），迭代看到切片后数据；
    - np.ndarray 请求切片时 fail-closed（拒绝返回完整面板）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.shard_executor import SliceCache, SliceDataSource


class _InnerSource:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.dataset = "test_ds"
        self.market = "ashare"

    @property
    def columns(self):
        return list(self._frame.columns)

    def load_column(self, name: str) -> pd.Series:
        return self._frame[name]

    def load_column_panel(self, name: str) -> pd.DataFrame:
        return self._frame

    def load_columns(self, names: list[str]) -> dict[str, pd.Series]:
        return {n: self._frame[n] for n in names}


def _frame() -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-05", "2026-01-16")
    return pd.DataFrame(
        np.arange(len(idx) * 3, dtype=float).reshape(len(idx), 3),
        index=idx,
        columns=["A", "B", "C"],
    )


def test_slice_data_source_explicit_projection():
    src = SliceDataSource(_InnerSource(_frame()), instruments=("A", "B"))
    panel = src.load_column_panel("close")
    assert set(panel.columns) == {"A", "B"}
    assert src.load_columns(["A", "B"])["A"].shape == (panel.shape[0],)
    assert src.columns == ["A", "B"]
    assert src.instrument_filter == ("A", "B")
    # 元数据透传。
    assert src.dataset == "test_ds"
    assert src.market == "ashare"


def test_slice_data_source_unknown_attribute_fails_loud():
    src = SliceDataSource(_InnerSource(_frame()))
    # 数据读取方法不能可靠投影 → fail-loud（显式 NotImplementedError）。
    with pytest.raises(NotImplementedError):
        _ = src.scan_polars_long(["close"])
    # 未知属性 / 数据方法不静默透传 inner。
    with pytest.raises(AttributeError):
        _ = src.some_unknown_data_method()


def test_slice_cache_full_mapping_and_iteration():
    base = {"s1": _frame(), "s2": _frame().iloc[:, :2]}
    cache = SliceCache(base, instruments=("A", "C"))
    # 完整 Mapping：len / iter / items / values 都可见切片后数据。
    assert len(cache) == 2
    assert set(cache) == {"s1", "s2"}
    items = dict(cache.items())
    assert set(items["s1"].columns) == {"A", "C"}
    # s2 只有 A/B 两列 → 切片到 {A, C} 后仅剩 A（列不新增，只收窄）。
    assert set(items["s2"].columns) == {"A"}
    vals = list(cache.values())
    assert all(set(v.columns) <= {"A", "C"} for v in vals)
    assert "s1" in cache


def test_slice_cache_ndarray_fails_closed_when_sliced():
    base = {"n": np.arange(24.0).reshape(6, 4)}
    cache = SliceCache(base, instruments=("A",))
    # 请求了切片但表示无标签 → 拒绝返回完整面板。
    with pytest.raises(ValueError):
        _ = cache["n"]
    # 未请求切片 → 原样（无 slice 语义负担）。
    noop = SliceCache(base)
    assert noop["n"] is base["n"]


def test_slice_time_projection():
    src = SliceDataSource(
        _InnerSource(_frame()),
        time_range=("2026-01-08", "2026-01-12"),
        warmup_start="2026-01-06",
    )
    panel = src.load_column_panel("close")
    assert panel.index[0] == pd.Timestamp("2026-01-06")  # warmup 起点
    assert panel.index[-1] <= pd.Timestamp("2026-01-12")
    assert pd.Timestamp(src.start_date) == pd.Timestamp("2026-01-06")
    assert pd.Timestamp(src.end_date) == pd.Timestamp("2026-01-12")
