# -*- coding: utf-8 -*-
"""R39-PERF-028 / 031 / 032 —— 表示转换批量 fast path + KPI + 临时 buffer arena。

- PERF-028: ``backend.panel_polars`` 的 Arrow 单块传输 fast path，默认开启但仅在
  Arrow 安全块生效；字节一致 vs 逐列参考路径。
- PERF-031: ``backend.rep_transition`` KPI（transition_count / transition_bytes）。
- PERF-032: ``runtime.temp_array_arena`` 临时 NumPy 数组池（投毒回收 + 复用计数）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import factor_engine.backend.panel_polars as pp
from factor_engine.backend.panel_polars import (
    convert_panel_to_polars_bulk,
    convert_polars_to_panel_bulk,
    _panel_to_polars_column_loop,
    _polars_to_panel_column_loop,
    panel_to_polars,
    polars_to_panel,
)
from factor_engine.backend.rep_transition import (
    get_rep_transition_tracker,
    rep_transition_snapshot,
)
from factor_engine.runtime.temp_array_arena import (
    TemporaryArrayArena,
    with_scratch,
)

FE_TIME = pp.FE_TIME_COL


@pytest.fixture(autouse=True)
def _reset_kpi():
    """每个测试前重置 KPI tracker（计数器是面板级模块变量，用 delta 断言）。"""
    get_rep_transition_tracker().reset()
    yield


def _float_panel(rows: int = 4) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows)
    return pd.DataFrame(
        {
            "a": [1.0, float("nan"), float("inf"), -float("inf")],
            "b": [4.0, 5.0, float("nan"), 8.0],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# (a) panel_to_polars bulk == 逐列参考路径（float panel）
# ---------------------------------------------------------------------------
def test_bulk_equals_column_loop_float_panel():
    panel = _float_panel()
    bulk = panel_to_polars(panel)  # use_arrow=True default
    ref = _panel_to_polars_column_loop(panel)
    assert bulk.schema == ref.schema, f"schema mismatch: {bulk.schema} vs {ref.schema}"
    assert bulk.equals(ref), "bulk fast path diverged from per-column reference"
    # NaN/Inf 位置必须保留（非 NULL）——参考路径语义
    assert bulk["a"].to_list()[1] != bulk["a"].to_list()[1]  # is NaN
    assert bulk["a"].to_list()[2] == float("inf")
    assert bulk["a"].to_list()[3] == float("-inf")
    assert FE_TIME in bulk.columns
    assert str(bulk.schema[FE_TIME]) == "Datetime(time_unit='ns', time_zone=None)"


def test_convert_panel_to_polars_bulk_matches_reference():
    panel = _float_panel()
    assert convert_panel_to_polars_bulk(panel).equals(_panel_to_polars_column_loop(panel))


# ---------------------------------------------------------------------------
# (b) polars_to_panel bulk == 参考路径（对齐 template index）
# ---------------------------------------------------------------------------
def test_polars_to_panel_bulk_equals_reference():
    idx = pd.date_range("2020-01-01", periods=4)
    template = pd.DataFrame({"a": [0.0] * 4, "b": [0.0] * 4}, index=idx)
    result = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})

    bulk = polars_to_panel(result, template=template)
    ref = _polars_to_panel_column_loop(result, template)
    pd.testing.assert_frame_equal(bulk, ref)
    assert list(bulk.index) == list(template.index)


def test_polars_to_panel_bulk_with_nan():
    idx = pd.date_range("2020-01-01", periods=3)
    template = pd.DataFrame({"a": [0.0] * 3}, index=idx)
    result = pl.DataFrame({"a": [1.0, None, 3.0]})
    bulk = polars_to_panel(result, template=template)
    ref = _polars_to_panel_column_loop(result, template)
    pd.testing.assert_frame_equal(bulk, ref)
    assert np.isnan(bulk["a"].iloc[1])


# ---------------------------------------------------------------------------
# (c) 计数器只在 fast path 自增
# ---------------------------------------------------------------------------
def test_bulk_counters_increment_only_on_fast_path():
    b0, c0 = pp.rep_conversion_bulk_count, pp.rep_conversion_column_loop_count
    panel_to_polars(_float_panel())  # arrow-safe -> bulk
    assert pp.rep_conversion_bulk_count == b0 + 1
    assert pp.rep_conversion_column_loop_count == c0

    obj = pd.DataFrame({"s": ["x", "y", "z", "w"], "v": [1.0, 2.0, 3.0, 4.0]})
    panel_to_polars(obj)  # object column -> fallback
    assert pp.rep_conversion_column_loop_count == c0 + 1
    assert pp.rep_conversion_bulk_count == b0 + 1

    # polars_to_panel 侧
    res = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    template = pd.DataFrame({"a": [0.0] * 4})
    b1, c1 = pp.rep_conversion_bulk_count, pp.rep_conversion_column_loop_count
    polars_to_panel(res, template=template)
    assert pp.rep_conversion_bulk_count == b1 + 1
    assert pp.rep_conversion_column_loop_count == c1

    # mixed int+float result -> to_numpy 上转 float64，与 to_pandas 不一致 -> fallback
    res2 = pl.DataFrame({"i": [1, 2, 3, 4], "f": [1.5, 2.5, 3.5, 4.5]})
    template2 = pd.DataFrame({"i": [0] * 4, "f": [0.0] * 4})
    b2, c2 = pp.rep_conversion_bulk_count, pp.rep_conversion_column_loop_count
    polars_to_panel(res2, template=template2)
    assert pp.rep_conversion_column_loop_count == c2 + 1
    assert pp.rep_conversion_bulk_count == b2


def test_use_arrow_false_forces_column_loop():
    b0, c0 = pp.rep_conversion_bulk_count, pp.rep_conversion_column_loop_count
    panel_to_polars(_float_panel(), use_arrow=False)
    assert pp.rep_conversion_column_loop_count == c0 + 1
    assert pp.rep_conversion_bulk_count == b0


# ---------------------------------------------------------------------------
# (d) fallback 对 mixed-dtype / object panel 保持正确
# ---------------------------------------------------------------------------
def test_fallback_mixed_object_panel():
    idx = pd.date_range("2020-01-01", periods=3)
    mixed = pd.DataFrame(
        {"i": [1, 2, 3], "f": [1.5, 2.5, 3.5], "s": ["a", "b", "c"]}, index=idx
    )
    bulk = panel_to_polars(mixed)  # object col -> fallback
    ref = _panel_to_polars_column_loop(mixed)
    assert bulk.schema == ref.schema
    assert bulk.equals(ref)
    assert str(bulk.schema["i"]) == "Int64"
    assert str(bulk.schema["s"]) == "String"


def test_fallback_nullable_extension_dtype():
    # pandas nullable Int64（extension dtype）—— Arrow 单块会改变表示，必须回退
    idx = pd.date_range("2020-01-01", periods=3)
    df = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64")}, index=idx)
    bulk = panel_to_polars(df)
    ref = _panel_to_polars_column_loop(df)
    assert bulk.schema == ref.schema
    assert bulk.equals(ref)


def test_float16_falls_back():
    idx = pd.date_range("2020-01-01", periods=3)
    df = pd.DataFrame({"a": np.array([1.0, 2.0, 3.0], dtype="float16")}, index=idx)
    bulk = panel_to_polars(df)
    ref = _panel_to_polars_column_loop(df)
    assert bulk.schema == ref.schema
    assert bulk.equals(ref)


# ---------------------------------------------------------------------------
# (e) rep_transition KPI 记录 (from, to) + bytes
# ---------------------------------------------------------------------------
def test_rep_transition_snapshot_records_bytes():
    panel = _float_panel()
    panel_to_polars(panel)
    result = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    template = pd.DataFrame({"a": [0.0] * 4})
    polars_to_panel(result, template=template)

    snap = rep_transition_snapshot()
    p2p = snap["transition_count"].get(("pandas", "polars"), 0)
    p2pd = snap["transition_count"].get(("polars", "pandas"), 0)
    assert p2p >= 1
    assert p2pd >= 1
    assert snap["transition_bytes"][("pandas", "polars")] > 0
    assert snap["transition_bytes"][("polars", "pandas")] > 0


def test_rep_transition_tracker_direct():
    tracker = get_rep_transition_tracker()
    tracker.record_transition("pandas", "polars", 100)
    tracker.record_transition("pandas", "polars", 50)
    tracker.record_transition("polars", "pandas", 30)
    snap = tracker.snapshot()
    assert snap["transition_count"][("pandas", "polars")] == 2
    assert snap["transition_bytes"][("pandas", "polars")] == 150
    assert snap["transition_bytes"][("polars", "pandas")] == 30


# ---------------------------------------------------------------------------
# (f) TemporaryArrayArena：复用 + 投毒检测
# ---------------------------------------------------------------------------
def test_arena_reuses_same_shape():
    arena = TemporaryArrayArena()
    arr1 = arena.acquire((100,), np.float64)
    arr1[:] = 7.0
    arena.release(arr1)
    arr2 = arena.acquire((100,), np.float64)
    assert arr2.shape == (100,)
    # 复用了（同一 shape bucket）：reuse_count 增加，alloc_count 不变
    assert arena.scratch_array_reuse_count > 0
    assert arena.scratch_alloc_count == 1
    arena.release(arr2)


def test_arena_poison_detects_aliasing():
    arena = TemporaryArrayArena()
    arr = arena.acquire((50,), np.float64)
    arr[:] = 1.2345  # 写入“结果”
    arena.release(arr)  # 投毒
    again = arena.acquire((50,), np.float64)
    # 释放后的数组已被哨兵填满（NaN）——若把 release 后仍持有的引用写进输出，
    # 会被 NaN 污染，立刻可检测而非静默错误。
    assert np.isnan(again).all()
    arena.release(again)


def test_arena_int_poison():
    arena = TemporaryArrayArena()
    arr = arena.acquire((8,), np.int64)
    arr[:] = 0
    arena.release(arr)
    again = arena.acquire((8,), np.int64)
    assert (again == np.iinfo(np.int64).max).all()
    arena.release(again)


def test_arena_new_shape_allocates():
    arena = TemporaryArrayArena()
    a = arena.acquire((16,), np.float64)
    arena.release(a)
    alloc_after_16 = arena.scratch_alloc_count
    b = arena.acquire((32,), np.float64)  # 不同 shape bucket -> 新分配
    assert arena.scratch_alloc_count == alloc_after_16 + 1
    arena.release(b)
    # 回到 (16,) 复用
    c = arena.acquire((16,), np.float64)
    assert arena.scratch_alloc_count == alloc_after_16 + 1
    arena.release(c)


def test_with_scratch_context():
    arena = TemporaryArrayArena()
    with with_scratch(arena, (10, 10), np.float32) as buf:
        buf[:] = 3.0
        assert buf.shape == (10, 10)
        assert buf.dtype == np.float32
    # 退出后数组被投毒回池
    again = arena.acquire((10, 10), np.float32)
    assert np.isnan(again).all()
    arena.release(again)


def test_arena_stats():
    arena = TemporaryArrayArena()
    a = arena.acquire((5,), np.float64)
    arena.release(a)
    s = arena.stats()
    assert s["scratch_alloc_count"] == 1
    assert s["pooled_arrays"] == 1
    b = arena.acquire((5,), np.float64)
    arena.release(b)
    assert arena.scratch_array_reuse_count == 1


# ---------------------------------------------------------------------------
# 边界：空 panel
# ---------------------------------------------------------------------------
def test_empty_panel_bulk_ok():
    idx = pd.date_range("2020-01-01", periods=2)
    empty = pd.DataFrame(index=idx)
    bulk = panel_to_polars(empty)
    ref = _panel_to_polars_column_loop(empty)
    assert bulk.schema == ref.schema
    assert bulk.equals(ref)
    assert FE_TIME in bulk.columns
