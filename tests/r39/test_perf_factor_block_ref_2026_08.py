# -*- coding: utf-8 -*-
"""R39-P0-PERF-030：FactorBlockRef —— 多个因子共享一份 axis。

覆盖：
  (a) FactorBlockRef 字段完整、frozen；AxisBufferRef 在 buffer_ref.py 中不存在，
      由此模块定义；values_ref/validity_ref 复用 runtime.buffer_ref.BufferRef。
  (b) build_factor_block：index 共享（对象身份 is）、values shape rows×factor_count
      对齐、dtype、validity NaN/Inf 掩码（全有效 None）。
  (c) share_axis_across：同 axis True、异 axis False（证明 axis 不重复）。
  (d) group_by_shared_axis：同 axis 多因子只记一次（同 index 不重复）。
  (e) to_arrow_table / to_parquet round-trip 值一致（整 block 一次写）。
  (f) matrix block 写路径真实构造 FactorBlockRef（factor_block_ref_used >= 1，
      随 manifest/summary 记录）；legacy 路径不记录。
  (g) materialize_batch 同 axis batch 产生共享 axis 计数（同 index 不重复），
      异 axis 为 0。
  (h) 默认路径回归：materialize_many_fast / matrix materialize 输出不变。
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runtime import factor_block_ref as fbr
from runtime.buffer_ref import BufferRef


def _idx(dates=("2024-01-01", "2024-01-02"), assets=("A", "B")):
    dts = [pd.Timestamp(d) for d in dates]
    return pd.MultiIndex.from_product(
        [dts, assets], names=["timestamp", "instrument"]
    )


def _ser(data, name="instrument"):
    idx = pd.MultiIndex.from_tuples(list(data.keys()), names=["datetime", name])
    return pd.Series(list(data.values()), index=idx, dtype="float64")


def _engine(dates: int = 8):
    from api import rank, ts_mean, ts_std
    from api.columns import col
    from api.factor import Factor
    from backend.pandas_backend import PandasBackend
    from runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    dts = pd.bdate_range("2024-01-02", periods=dates)
    idx = pd.MultiIndex.from_product([dts, ["A", "B"]], names=["timestamp", "instrument"])
    close = pd.Series([float(i) for i in range(len(idx))], index=idx)
    vol = pd.Series([float(i * 2) for i in range(len(idx))], index=idx)
    eng = FactorEngine(
        data_source=InMemorySeriesSource({"close": close, "volume": vol}),
        backend=PandasBackend(),
    )
    factors = [
        Factor(name="f_mean5", expr=ts_mean(col("close"), 5)),
        Factor(name="f_std20", expr=ts_std(col("close"), 3)),
        Factor(name="f_rank", expr=rank(col("close"))),
    ]
    return eng, factors


# ---------------------------------------------------------------------------
# (a) dataclass 字段 / frozen / AxisBufferRef 定义 / BufferRef 复用
# ---------------------------------------------------------------------------


def test_r39_030_factor_block_ref_fields_frozen():
    fbr.reset_counters()
    idx = _idx()
    vals = np.arange(8, dtype="float64").reshape(4, 2)
    ref = fbr.build_factor_block(["a", "b"], vals, idx, dtype="float64")
    assert isinstance(ref, fbr.FactorBlockRef)
    # 规格字段完整
    assert ref.factor_ids == ("a", "b")
    assert ref.dtype == "float64"
    assert ref.row_count == 4
    assert ref.factor_count == 2
    assert isinstance(ref.axis_ref, fbr.AxisBufferRef)
    assert isinstance(ref.values_ref, BufferRef)
    # frozen：任何赋值都抛 FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        ref.row_count = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        ref.factor_ids = ("x",)  # type: ignore[misc]
    # AxisBufferRef 在 runtime.buffer_ref 中不存在（本模块定义极简形态）
    assert not hasattr(__import__("runtime.buffer_ref", fromlist=["x"]), "AxisBufferRef")


def test_r39_030_values_and_validity_reuse_bufferref():
    """values_ref / validity_ref 复用 runtime.buffer_ref.BufferRef。"""
    fbr.reset_counters()
    idx = _idx()
    vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype="float32")
    ref = fbr.build_factor_block(["a", "b"], vals, idx, dtype="float32")
    assert isinstance(ref.values_ref, BufferRef)
    assert ref.values_ref.representation == "numpy_block"
    assert ref.validity_ref is None  # 全有效
    assert ref.values_ref.bytes == int(vals.nbytes)
    assert ref.values_ref.schema == ("a", "b")


# ---------------------------------------------------------------------------
# (b) build_factor_block：index 共享 / shape / dtype / validity
# ---------------------------------------------------------------------------


def test_r39_030_build_index_shared_shape_dtype():
    fbr.reset_counters()
    idx = _idx()
    vals = np.arange(8, dtype="float64").reshape(4, 2)
    ref = fbr.build_factor_block(["a", "b"], vals, idx, dtype="float64")
    # index 单次引用（不复制）：同一对象
    assert ref.axis_ref.index is idx
    # values 单 block：rows×factor_count 对齐，连续共享（dtype 相同则无复制）
    loc = ref.values_ref.location
    assert loc.shape == (4, 2)
    assert loc.dtype == np.dtype("float64")
    assert np.shares_memory(loc, vals)
    # factor_count / row_count 一致
    assert ref.row_count == len(idx) == 4
    assert ref.factor_count == len(ref.factor_ids) == 2


def test_r39_030_build_axis_key_equal_labels():
    """label 相等但对象不同的 axis → axis_key 相同（同 axis 判定与身份无关）。"""
    fbr.reset_counters()
    idx1 = _idx()
    idx2 = _idx()  # 不同对象，相同 label
    ref1 = fbr.build_factor_block(
        ["a"], np.arange(4, dtype="float32").reshape(4, 1), idx1, dtype="float32"
    )
    ref2 = fbr.build_factor_block(
        ["b"], np.arange(4, dtype="float32").reshape(4, 1) * 10, idx2, dtype="float32"
    )
    assert ref1.axis_ref.index is not ref2.axis_ref.index
    assert ref1.axis_ref.axis_key == ref2.axis_ref.axis_key
    assert fbr.share_axis_across([ref1, ref2]) is True


def test_r39_030_build_validity_mask_nan_inf():
    """NaN/Inf → validity_ref 非 None 且掩码形状 rows×factors；全有效 → None。"""
    fbr.reset_counters()
    idx = _idx()
    vals = np.ones((4, 2), dtype="float64")
    vals[0, 0] = np.nan
    vals[1, 1] = np.inf
    ref = fbr.build_factor_block(["a", "b"], vals, idx, dtype="float64")
    assert ref.validity_ref is not None
    mask = ref.validity_ref.location
    assert mask.shape == (4, 2)
    assert mask.dtype == np.dtype(bool)
    assert bool(mask[0, 0]) is True and bool(mask[1, 1]) is True
    assert bool(mask[0, 1]) is False
    # 全有效 → None
    ref2 = fbr.build_factor_block(
        ["a"], np.ones((4, 1), dtype="float32"), idx, dtype="float32"
    )
    assert ref2.validity_ref is None


def test_r39_030_build_dict_input_column_order():
    """dict 输入按 factor_ids 列顺序 column-stack。"""
    fbr.reset_counters()
    idx = _idx()
    a = pd.Series([1.0, 3.0, 5.0, 7.0], index=idx)
    b = pd.Series([2.0, 4.0, 6.0, 8.0], index=idx)
    ref = fbr.build_factor_block(
        ["b", "a"], {"a": a, "b": b}, idx, dtype="float64"
    )
    loc = ref.values_ref.location
    assert list(ref.factor_ids) == ["b", "a"]
    assert list(loc[:, 0]) == [2.0, 4.0, 6.0, 8.0]  # 第一列是 b
    assert list(loc[:, 1]) == [1.0, 3.0, 5.0, 7.0]  # 第二列是 a


# ---------------------------------------------------------------------------
# (c) share_axis_across / (d) group_by_shared_axis
# ---------------------------------------------------------------------------


def test_r39_030_share_axis_across_same_and_diff():
    fbr.reset_counters()
    idx1 = _idx()
    idx2 = _idx(dates=("2024-02-01", "2024-02-02"))
    r1 = fbr.build_factor_block(["a"], np.ones((4, 1), dtype="float32"), idx1, dtype="float32")
    r2 = fbr.build_factor_block(["b"], np.ones((4, 1), dtype="float32"), idx1, dtype="float32")
    r3 = fbr.build_factor_block(["c"], np.ones((4, 1), dtype="float32"), idx2, dtype="float32")
    assert fbr.share_axis_across([r1, r2]) is True
    assert fbr.share_axis_across([r1, r2, r3]) is False
    assert fbr.share_axis_across([]) is True
    assert fbr.share_axis_across([r1]) is True


def test_r39_030_group_by_shared_axis_dedup():
    """同 axis 多因子只记一次（同 index 不重复）；异 axis 单独成组。"""
    fbr.reset_counters()
    idx1 = _idx()
    idx2 = _idx(dates=("2024-02-01", "2024-02-02"))
    series = {
        "a": pd.Series(np.arange(4, dtype=float), index=idx1),
        "b": pd.Series(np.arange(4, dtype=float) * 10, index=idx1),
        "c": pd.Series(np.arange(4, dtype=float) * 100, index=idx1),  # 同一 idx1
        "d": pd.Series(np.arange(4, dtype=float), index=idx2),
    }
    groups = fbr.group_by_shared_axis(series)
    by_rows = {g[0].row_count: sorted(g[1]) for g in groups}
    # idx1 共享组：a/b/c（只记一次）；idx2 只有 d 一个因子，不构成共享组
    assert by_rows.get(4) == ["a", "b", "c"]
    assert len(groups) == 1


# ---------------------------------------------------------------------------
# (e) to_arrow_table / to_parquet 整 block 一次写，round-trip 值一致
# ---------------------------------------------------------------------------


def test_r39_030_to_arrow_table_columns_values():
    fbr.reset_counters()
    idx = _idx()
    vals = np.arange(8, dtype="float64").reshape(4, 2)
    ref = fbr.build_factor_block(["a", "b"], vals, idx, dtype="float64")
    table = ref.to_arrow_table()
    assert set(table.schema.names) == {"a", "b", "timestamp", "instrument"}
    df = table.to_pandas()
    # 值一致（列顺序与 factor_ids 对齐）
    assert list(df["a"]) == [0.0, 2.0, 4.0, 6.0]
    assert list(df["b"]) == [1.0, 3.0, 5.0, 7.0]


def test_r39_030_to_parquet_roundtrip_values(tmp_path):
    fbr.reset_counters()
    idx = _idx()
    vals = np.array(
        [[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [7.0, 8.0]], dtype="float64"
    )
    ref = fbr.build_factor_block(["a", "b"], vals, idx, dtype="float64")
    path = ref.to_parquet(tmp_path / "block.parquet")
    assert Path(path).exists()
    back = pd.read_parquet(path)
    # NaN 位置 round-trip 一致，数值一致
    assert (back["a"].to_numpy() == vals[:, 0]).all() | (
        back["a"].isna().to_numpy() == np.isnan(vals[:, 0])
    ).all()
    np.testing.assert_allclose(
        back["b"].to_numpy(dtype="float64", na_value=np.nan), vals[:, 1]
    )
    # 单文件一次写
    assert len(list(Path(path).parent.glob("block.parquet"))) == 1


# ---------------------------------------------------------------------------
# (f) matrix block 写路径真实构造 FactorBlockRef（计数器 + manifest/summary）
# ---------------------------------------------------------------------------


def test_r39_030_matrix_block_path_records_factor_block_ref(tmp_path, monkeypatch):
    from storage.materialize.factor_matrix_materializer import (
        FactorMatrixMaterializer,
        _read_manifest,
    )

    monkeypatch.setenv("FACTOR_ENGINE_MATRIX_BLOCK_LAYOUT", "1")
    fbr.reset_counters()
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    summary = m.materialize(
        {
            "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0}),
            "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
            "c": _ser({(d1, "A"): 100.0, (d2, "A"): 200.0}),
        },
        universe="u",
        production=True,
    )
    counters = fbr.get_counters()
    assert counters["factor_block_ref_used"] >= 1
    assert counters["factor_block_ref_built"] >= 1
    # summary 随记录
    assert summary.get("factor_block_refs"), "summary 应记录 factor_block_refs"
    meta = summary["factor_block_refs"][0]
    assert meta["factor_count"] == 3
    assert set(meta["factor_ids"]) == {"a", "b", "c"}
    assert meta["row_count"] == 2
    assert meta["dtype"] == "float32"
    assert meta["block_id"] == "0000"
    # manifest 随记录
    manifest = _read_manifest(tmp_path / "matrix" / "universe=u" / "freq=1d")
    assert manifest.get("factor_block_refs") == summary["factor_block_refs"]
    # 共享 axis 证明：manifest 里所有 block ref 同 axis_key
    keys = {e["axis_key"] for e in manifest["factor_block_refs"]}
    assert len(keys) == 1


def test_r39_030_matrix_legacy_path_not_recording(tmp_path):
    """legacy（非 block）路径不构造 FactorBlockRef（默认路径行为不变）。"""
    from storage.materialize.factor_matrix_materializer import FactorMatrixMaterializer

    fbr.reset_counters()
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    summary = m.materialize(
        {
            "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0}),
            "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
        },
        universe="u",
    )
    assert "factor_block_refs" not in summary
    assert fbr.get_counters()["factor_block_ref_used"] == 0
    assert summary["layout"] == "legacy"


# ---------------------------------------------------------------------------
# (g) materialize_batch 同 axis 共享计数（同 index 不重复；异 axis 为 0）
# ---------------------------------------------------------------------------


def test_r39_030_materialize_batch_shared_axis_count_dedup(tmp_path):
    from runtime.materialize_batch import (
        MaterializeItem,
        execute_materialize_batch,
    )

    eng, factors = _engine()
    items = []

    def _run_output(engine, factor):
        run = engine.run(factor)
        return {"factor": factor, "analysis": run["analysis"], "result": run["result"]}

    for f in factors:
        items.append(
            MaterializeItem(
                factor=f,
                output=_run_output(eng, f),
                factor_id=f.name,
                options={"write_target": "local"},
            )
        )
    out = execute_materialize_batch(
        eng,
        items,
        shared_options={
            "lake_root": tmp_path / "lake",
            "write_target": "local",
            "value_dtype": "float32",
        },
    )
    counters = out["counters"]
    assert counters["item_count"] == 3
    # 三个因子共享同一 MultiIndex → 不同共享 axis 数 = 1（不是 3）
    assert counters["factor_block_shared_axis_count"] == 1
    # 逐因子写不受影响（仍逐因子 catalog/write）
    assert counters["batch_write_transaction_count"] == 1
    assert counters["physical_partition_write_rounds"] == 3
    assert set(out["materializations"].keys()) == {"f_mean5", "f_std20", "f_rank"}


def test_r39_030_record_shared_axes_different_axis_zero():
    """异 axis batch → 共享 axis 计数 0。"""
    from runtime.materialize_batch import _record_shared_axes

    idx1 = _idx()
    idx2 = _idx(dates=("2024-02-01", "2024-02-02"))
    preps = [
        {
            "factor_id": "a",
            "output": {"result": pd.Series(np.arange(4, dtype=float), index=idx1)},
            "opts": {"value_dtype": "float32"},
        },
        {
            "factor_id": "b",
            "output": {"result": pd.Series(np.arange(4, dtype=float), index=idx2)},
            "opts": {"value_dtype": "float32"},
        },
    ]
    fbr.reset_counters()
    assert _record_shared_axes(preps) == 0


# ---------------------------------------------------------------------------
# (h) 默认路径回归：materialize_many_fast 输出不变
# ---------------------------------------------------------------------------


def test_r39_030_default_path_materialize_many_fast_unchanged(tmp_path):
    eng, factors = _engine(dates=6)
    os.environ["FACTOR_ENGINE_HYBRID_FORCE"] = "thread"
    try:
        out = eng.materialize_many_fast(
            factors,
            materialize_kwargs={
                "lake_root": str(tmp_path / "lake"),
                "write_target": "local",
                "value_dtype": "float32",
            },
        )
    finally:
        os.environ.pop("FACTOR_ENGINE_HYBRID_FORCE", None)
    # 输出结构与既有契约不变
    assert set(out["materializations"].keys()) == {"f_mean5", "f_std20", "f_rank"}
    assert out["generation"].startswith("batch-")
    assert out["factor_ids"] == ["f_mean5", "f_std20", "f_rank"]
    assert out["batch_write_transaction_count"] >= 1
    assert not out.get("writer_errors")
    for name, s in out["materializations"].items():
        assert "error" not in s
        assert s.get("rows_written", 0) > 0


def test_r39_030_counters_getter_reset():
    fbr.reset_counters()
    assert fbr.get_counters() == {
        "factor_block_ref_built": 0,
        "factor_block_ref_used": 0,
        "factor_block_shared_axis_count": 0,
    }
    idx = _idx()
    fbr.build_factor_block(["a"], np.ones((4, 1), dtype="float32"), idx, dtype="float32")
    assert fbr.get_counters()["factor_block_ref_built"] == 1
    fbr.reset_counters()
    assert fbr.get_counters()["factor_block_ref_built"] == 0
