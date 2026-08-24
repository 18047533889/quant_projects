# -*- coding: utf-8 -*-
"""R39 PERF-060..067 factor_matrix 性能整改测试。

覆盖:
  (a) PERF-060 equal-axis  -> direct column-stack, ``matrix_join_count == 0``，
      输出与旧 pairwise-merge reference 逐值/逐 dtype 一致。
  (b) PERF-060 different-axis -> 单次 canonical-reindex（``matrix_join_count == 1``），
      输出仍与旧 pairwise-merge reference 一致。
  (c) PERF-061/062 column-factor block layout：新增因子进新 block，既有 block
      字节不变（COW hardlink），``load_matrix(factor_ids=[...])`` 只扫命中的 block。
  (d) PERF-064 checksum staging proof：read-back 校验 == write-time checksums，
      byte flip 被检出。
  (e) PERF-065 ``load_matrix`` pushdown：``factor_ids`` / ``time_range`` /
      ``instrument_filter`` 只读需要的数据。
  (f) PERF-067 MaterializationIdentityCertificate 跨调用稳定，且 writer 消费它把
      身份字段写入 manifest。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import factor_engine.storage.matrix_block_layout as mbl
from factor_engine.storage.factor_format import series_to_long_table
from factor_engine.storage.materialize.factor_matrix_materializer import (
    FactorMatrixMaterializer,
    _read_manifest,
)


def _ser(data: dict[tuple, float], name: str = "instrument") -> pd.Series:
    idx = pd.MultiIndex.from_tuples(list(data.keys()), names=["datetime", name])
    return pd.Series(list(data.values()), index=idx, dtype="float64")


def _reference_merge(results: dict[str, pd.Series], value_dtype: str = "float32") -> pd.DataFrame:
    """Pre-R39 per-factor pairwise pandas outer merge (the OLD writer reference)."""
    fids = sorted(results.keys())
    merged: pd.DataFrame | None = None
    for fid in fids:
        long_df = series_to_long_table(results[fid])
        long_df = long_df.rename(columns={"value": fid})
        if merged is None:
            merged = long_df
        else:
            merged = merged.merge(long_df, on=["datetime", "asset"], how="outer")
    assert merged is not None
    for fid in fids:
        merged[fid] = merged[fid].astype(value_dtype)
    return merged.sort_values(["datetime", "asset"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# (a) PERF-060 equal-axis -> direct column-stack, zero join
# ---------------------------------------------------------------------------


def test_r39_060_equal_axis_zero_join(tmp_path):
    mbl.reset_perf_counters()
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    results = {
        "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0}),
        "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
        "c": _ser({(d1, "A"): 100.0, (d2, "A"): 200.0}),
    }
    axis, wide, joins = mbl.build_matrix_block(results)
    assert joins == 0
    assert mbl.matrix_join_count == 0

    merged = mbl.wide_to_merged(wide, sorted(results), value_dtype="float32")
    ref = _reference_merge(results)
    pd.testing.assert_frame_equal(merged, ref)

    # materialize 主链同样命中零 join 路径
    mbl.reset_perf_counters()
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    summary = m.materialize(results, universe="u")
    assert summary["matrix_join_count"] == 0
    loaded = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), ref)


# ---------------------------------------------------------------------------
# (b) PERF-060 different-axis -> single canonical-reindex
# ---------------------------------------------------------------------------


def test_r39_060_different_axis_canonical_reindex(tmp_path):
    mbl.reset_perf_counters()
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    results = {
        "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0, (d1, "B"): 3.0}),
        "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
        "c": _ser({(d1, "A"): 100.0, (d2, "A"): 200.0, (d2, "C"): 7.0}),
    }
    axis, wide, joins = mbl.build_matrix_block(results)
    assert joins == 1  # exactly one canonical reindex, not N merges
    assert mbl.matrix_join_count == 1

    merged = mbl.wide_to_merged(wide, sorted(results), value_dtype="float32")
    ref = _reference_merge(results)
    pd.testing.assert_frame_equal(merged, ref)

    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    m.materialize(results, universe="u")
    loaded = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), ref)


# ---------------------------------------------------------------------------
# (c) PERF-061/062 block layout: new factor -> new block, no rewrite of existing
# ---------------------------------------------------------------------------


def test_r39_061_block_layout_new_factor_no_rewrite(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_MATRIX_BLOCK_LAYOUT", "1")
    mbl.reset_perf_counters()
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    m.materialize(
        {
            "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0}),
            "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
            "c": _ser({(d1, "A"): 100.0, (d2, "A"): 200.0}),
        },
        universe="u",
        production=True,
    )
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    manifest = _read_manifest(base)
    assert manifest["layout"] == "block"
    block_ids = {
        fid: manifest["factors"][fid]["block_id"] for fid in ("a", "b", "c")
    }
    assert len(set(block_ids.values())) == 1  # 3 factors -> 1 block

    gen_dir = base / "generation"
    old_blocks = {
        str(p.relative_to(gen_dir)): p.read_bytes()
        for p in gen_dir.rglob("block=*.parquet")
    }
    assert old_blocks

    # 新增 1 因子 -> 新 block；既有 block 字节不变（COW hardlink）。
    summary = m.materialize(
        {"d": _ser({(d1, "A"): 5.0, (d2, "A"): 6.0})},
        universe="u",
        production=True,
    )
    manifest = _read_manifest(base)
    assert manifest["factors"]["d"]["block_id"] != block_ids["a"]
    new_blocks = {
        str(p.relative_to(gen_dir)): p.read_bytes()
        for p in gen_dir.rglob("block=*.parquet")
    }
    for rel, payload in old_blocks.items():
        assert rel in new_blocks, f"既有 block {rel} 丢失"
        assert new_blocks[rel] == payload, f"既有 block {rel} 被重写（写放大）"
    assert summary["matrix_rewrite_amplification"] == 0.0

    # load pushdown 只扫命中 block=0001
    read_paths: list[str] = []
    orig_read = pd.read_parquet

    def _spy(path, *a, **k):
        read_paths.append(str(path))
        return orig_read(path, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", _spy)
    loaded = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u", factor_ids=["d"]
    )
    assert list(loaded.columns) == ["datetime", "asset", "d"]
    assert list(loaded["d"]) == [5.0, 6.0]
    assert read_paths, "pushdown 应实际读取 block 文件"
    assert all("block=0001.parquet" in p for p in read_paths), read_paths


# ---------------------------------------------------------------------------
# (d) PERF-064 checksum staging proof + byte-flip detection
# ---------------------------------------------------------------------------


def test_r39_064_checksum_proof_and_byte_flip(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_MATRIX_CHECKSUM_PROOF", "1")
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    m.materialize(
        {
            "a": _ser({(d1, "A"): 1.0, (d2, "A"): np.nan}),
            "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
        },
        universe="u",
        production=True,
    )
    gen_dir = tmp_path / "matrix" / "universe=u" / "freq=1d" / "generation"
    pq = list(gen_dir.rglob("data.parquet"))[0]
    df = pd.read_parquet(pq)

    exp = mbl.compute_matrix_checksums(df)
    act = mbl.read_parquet_checksums(pq, df.columns)
    assert mbl.compare_checksums(exp, act) == []

    # byte flip（magic 后首个数据字节）必须被检出：读错误或 checksum 不一致。
    data = pq.read_bytes()
    corrupt = data[:4] + bytes([data[4] ^ 0xFF]) + data[5:]
    cp = pq.parent / "corrupt.parquet"
    cp.write_bytes(corrupt)
    detected = False
    try:
        act2 = mbl.read_parquet_checksums(cp, df.columns)
        detected = bool(mbl.compare_checksums(exp, act2))
    except Exception:  # noqa: BLE001 - 解析失败同样算检出（fail-closed）
        detected = True
    assert detected


# ---------------------------------------------------------------------------
# (e) PERF-065 load_matrix pushdown: columns / time_range / instrument_filter
# ---------------------------------------------------------------------------


def test_r39_065_load_pushdown_columns_and_time(tmp_path):
    d1, d2, d3 = (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-03-10"),
    )
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    m.materialize(
        {
            "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0, (d3, "A"): 3.0}),
            "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0, (d3, "A"): 30.0}),
        },
        universe="u",
    )
    # columns pushdown：只返回请求因子列
    only = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u", factor_ids=["b"]
    )
    assert list(only.columns) == ["datetime", "asset", "b"]

    # time_range pushdown：hive 分区裁剪，只返回 2 月
    feb = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix",
        universe="u",
        time_range=(pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-29")),
    )
    assert set(feb["datetime"].dt.month) == {2}
    assert list(feb["a"]) == [2.0]

    # instrument_filter：row 过滤
    none_rows = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u", instrument_filter=["NOPE"]
    )
    assert len(none_rows) == 0


# ---------------------------------------------------------------------------
# (f) PERF-067 MaterializationIdentityCertificate stable + writer reuse
# ---------------------------------------------------------------------------


def _mini_engine():
    from factor_engine.api import rank
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    data = {"close": pd.Series(np.arange(4, dtype=float) + 1.0, index=idx)}
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=data)
    )
    f = Factor(name="x", expr=rank(col("close")))
    return eng, f, data["close"]


def test_r39_067_identity_certificate_stable_and_writer_reuse(tmp_path, monkeypatch):
    from factor_engine.runtime import matrix_service
    from factor_engine.runtime.engine import _scope_from_factor

    eng, f, series = _mini_engine()
    scope = _scope_from_factor(f, data_source=eng.data_source)
    monkeypatch.setattr(
        matrix_service, "_matrix_factor_digest", lambda *a, **k: "d" * 16
    )
    monkeypatch.setattr(matrix_service, "_operator_manifest_hash", lambda: "op-hash")

    kwargs = dict(
        analyses={"x": SimpleNamespace(ir=None)},
        results={"x": series},
        scopes=[scope],
        effective_config={},
        pit_enforce=False,
        universe="u",
        frequency="1d",
        value_dtype="float32",
    )
    certs1 = matrix_service.build_certificates(eng, [f], ["x"], **kwargs)
    certs2 = matrix_service.build_certificates(eng, [f], ["x"], **kwargs)
    assert certs1["x"].semantic_digest == "d" * 16
    assert certs1["x"] == certs2["x"]  # 跨调用稳定（dataclass frozen __eq__）

    # writer 消费 certificates：身份字段落入 manifest，不重建。
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    m.materialize({"x": series}, universe="u", certificates=certs1)
    manifest = _read_manifest(tmp_path / "matrix" / "universe=u" / "freq=1d")
    entry = manifest["factors"]["x"]
    assert entry["semantic_digest"] == "d" * 16
    assert entry["operator_manifest_hash"] == "op-hash"
    assert entry["storage_precision"] == "float32"
    assert entry["universe"] is not None


# ---------------------------------------------------------------------------
# PERF-066 streaming adapter feeds writer; block-mode result == dict result
# ---------------------------------------------------------------------------


def test_r39_066_streaming_adapter_block_result(tmp_path):
    from factor_engine.runtime.matrix_service import _iter_matrix_results

    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    results = {
        "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0}),
        "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
    }
    run_out = {"results": dict(results), "analyses": {}}
    factors = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    items = list(_iter_matrix_results(run_out, factors, ["a", "b"]))
    assert [fid for fid, _s in items] == ["a", "b"]

    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    summary = m.materialize_from_iterable(
        iter(items), universe="u", layout="block"
    )
    assert summary["layout"] == "block"
    loaded = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        _reference_merge(results),
    )
