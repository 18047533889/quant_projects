# -*- coding: utf-8 -*-
"""R39 PERF-063: COW generation 不再每次 rglob 全旧 generation 再 hardlink/copy。

覆盖:
  (a) PartitionObjectRef 字段 + content_id 稳定（路径+rows+schema_hash 指纹）；
  (b) build_partition_inventory 在含 data.parquet 与 block=*.parquet 混合目录上
      正确（跳过 .staging/.quarantine）；
  (c) materialize_generation_from_inventory 精确产出目标 rel_path 集（与 rglob
      结果一致）、hardlink inode 复用（st_ino 相同证明未复制）；
  (d) 第二次 generation 切换 ``generation_rglob_discovery_count`` 不增加；
  (e) 缺 manifest inventory 的首建路径计数 == 1 且结果正确；
  (f) 首次 publish（无旧代）不 rglob；
  (g) block 布局 COW 用 inventory，既有 block inode 复用；
  (h) 读路径优先用 manifest inventory（无 rglob fallback）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from storage.materialize.factor_matrix_materializer import (
    FactorMatrixMaterializer,
    _read_manifest,
    _write_manifest_atomic,
)
from storage.partition_object_ref import (
    PartitionObjectRef,
    build_partition_inventory,
    content_id_for,
    get_generation_counters,
    inventory_from_dicts,
    inventory_to_dicts,
    materialize_generation_from_inventory,
    ref_from_frame,
    reset_generation_counters,
)


def _ser(data: dict[tuple, float]) -> pd.Series:
    idx = pd.MultiIndex.from_tuples(
        list(data.keys()), names=["datetime", "instrument"]
    )
    return pd.Series(list(data.values()), index=idx, dtype="float64")


def _df(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [f"2024-01-{i+1:02d}" for i in range(rows)]
            ),
            "asset": ["A"] * rows,
            "f": [float(i + 1) for i in range(rows)],
        }
    )


# ---------------------------------------------------------------------------
# (a) PartitionObjectRef 字段 + content_id 稳定
# ---------------------------------------------------------------------------


def test_ref_fields_and_content_id_stable():
    ref = PartitionObjectRef(
        rel_path="year=2024/month=01/data.parquet",
        content_id="abc",
        rows=10,
        schema_hash="s1",
    )
    assert ref.rel_path == "year=2024/month=01/data.parquet"
    assert ref.content_id == "abc"
    assert ref.rows == 10
    assert ref.schema_hash == "s1"
    # frozen dataclass：字段不可变
    with pytest.raises(Exception):
        ref.rows = 11  # type: ignore[misc]
    # content_id 稳定指纹：相同输入 → 相同输出
    assert content_id_for("x", 1, "s") == content_id_for("x", 1, "s")
    # 任一输入变化 → 指纹变化
    assert content_id_for("x", 1, "s") != content_id_for("y", 1, "s")
    assert content_id_for("x", 2, "s") != content_id_for("x", 1, "s")
    assert content_id_for("x", 1, "t") != content_id_for("x", 1, "s")


def test_ref_from_frame_content_id_stable():
    df = _df(2)
    r1 = ref_from_frame("year=2024/month=01/data.parquet", df)
    r2 = ref_from_frame("year=2024/month=01/data.parquet", df)
    assert r1 == r2
    assert r1.rows == 2
    assert r1.schema_hash
    assert r1.content_id == content_id_for(r1.rel_path, r1.rows, r1.schema_hash)
    # 列顺序是 schema 的一部分：调序 → 不同 schema_hash → 不同 content_id
    r3 = ref_from_frame("year=2024/month=01/data.parquet", df[["f", "asset", "datetime"]])
    assert r3.content_id != r1.content_id


# ---------------------------------------------------------------------------
# (b) build_partition_inventory 混合目录
# ---------------------------------------------------------------------------


def test_build_partition_inventory_mixed(tmp_path):
    reset_generation_counters()
    g = tmp_path / "gen"
    (g / "year=2024" / "month=01").mkdir(parents=True)
    (g / "year=2024" / "month=02").mkdir(parents=True)
    _df(1).to_parquet(g / "year=2024" / "month=01" / "data.parquet", index=False)
    _df(2).to_parquet(g / "year=2024" / "month=02" / "block=0001.parquet", index=False)
    # 干扰项：.staging / .quarantine 下的 parquet 必须被跳过
    (g / "year=2024" / "month=01" / ".staging").mkdir()
    _df(3).to_parquet(g / "year=2024" / "month=01" / ".staging" / "data.parquet", index=False)
    (g / "year=2024" / "month=01" / ".quarantine").mkdir()
    _df(4).to_parquet(g / "year=2024" / "month=01" / ".quarantine" / "data.parquet", index=False)

    inv = build_partition_inventory(g)
    by_rel = {r.rel_path: r for r in inv}
    assert sorted(by_rel) == [
        "year=2024/month=01/data.parquet",
        "year=2024/month=02/block=0001.parquet",
    ]
    assert by_rel["year=2024/month=01/data.parquet"].rows == 1
    assert by_rel["year=2024/month=02/block=0001.parquet"].rows == 2
    assert all(r.content_id and r.schema_hash for r in inv)
    # 构建确实走了 rglob 发现（计数 +1）
    assert get_generation_counters()["generation_rglob_discovery_count"] == 1


# ---------------------------------------------------------------------------
# (c) materialize_generation_from_inventory：精确 rel_path + inode 复用
# ---------------------------------------------------------------------------


def test_materialize_exact_rel_paths_and_inode_reuse(tmp_path):
    reset_generation_counters()
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "year=2024" / "month=01").mkdir(parents=True)
    (src / "year=2024" / "month=02").mkdir(parents=True)
    _df(1).to_parquet(src / "year=2024" / "month=01" / "data.parquet", index=False)
    _df(2).to_parquet(src / "year=2024" / "month=02" / "block=0001.parquet", index=False)

    inv = build_partition_inventory(src)
    n = materialize_generation_from_inventory(src, inv, dst)
    assert n == 2

    # 目标 rel_path 集与源 rglob 结果一致
    src_set = {
        str(p.relative_to(src))
        for p in src.rglob("*.parquet")
        if ".staging" not in p.parts and ".quarantine" not in p.parts
    }
    dst_set = {str(p.relative_to(dst)) for p in dst.rglob("*.parquet")}
    assert dst_set == src_set

    # hardlink inode 复用（st_ino 相同 → 未复制字节）
    for rel in dst_set:
        assert os.stat(src / rel).st_ino == os.stat(dst / rel).st_ino

    # skip_rel 排除已写入的分区
    dst2 = tmp_path / "dst2"
    n2 = materialize_generation_from_inventory(
        src, inv, dst2, skip_rel={"year=2024/month=01/data.parquet"}
    )
    assert n2 == 1
    assert (dst2 / "year=2024" / "month=02" / "block=0001.parquet").exists()
    assert not (dst2 / "year=2024" / "month=01" / "data.parquet").exists()

    # 计数：一次 inventory 驱动 materialize == +1
    assert get_generation_counters()["generation_inventory_materialize_count"] == 2


# ---------------------------------------------------------------------------
# (d) 第二次 generation 切换不 rglob
# ---------------------------------------------------------------------------


def test_second_generation_no_rglob_discovery(tmp_path):
    reset_generation_counters()
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-15")
    m.materialize({"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0})}, universe="u")
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    assert _read_manifest(base).get("partition_inventory")

    # 第二次 generation 切换：manifest 已有 inventory → 不 rglob，精确 COW。
    m.materialize({"f": _ser({(d1, "A"): 10.0})}, universe="u")
    c = get_generation_counters()
    assert c["generation_rglob_discovery_count"] == 0
    assert c["generation_inventory_materialize_count"] == 1

    # 第三次仍不 rglob
    m.materialize({"f": _ser({(d1, "A"): 100.0})}, universe="u")
    c = get_generation_counters()
    assert c["generation_rglob_discovery_count"] == 0
    assert c["generation_inventory_materialize_count"] == 2

    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    by_month = dict(zip(frame["datetime"].dt.month, frame["f"]))
    assert by_month == {1: 100.0, 2: 2.0}


# ---------------------------------------------------------------------------
# (e) 缺 manifest inventory 的首建路径计数 == 1 且结果正确
# ---------------------------------------------------------------------------


def test_first_build_legacy_manifest_counts_one(tmp_path):
    reset_generation_counters()
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-15")
    m.materialize({"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0})}, universe="u")
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    # 模拟 pre-R39 legacy manifest：剥离 partition_inventory
    md = json.loads((base / "manifest.json").read_text())
    md.pop("partition_inventory", None)
    _write_manifest_atomic(base, md)
    reset_generation_counters()

    # 第二次 publish：旧代无 inventory → 一次性 rglob 构建（计数 == 1，诚实）
    m.materialize({"f": _ser({(d1, "A"): 10.0})}, universe="u")
    assert get_generation_counters()["generation_rglob_discovery_count"] == 1

    # 第三次：manifest 已带 inventory → 不再 rglob
    m.materialize({"f": _ser({(d1, "A"): 100.0})}, universe="u")
    c = get_generation_counters()
    assert c["generation_rglob_discovery_count"] == 1
    assert c["generation_inventory_materialize_count"] == 2

    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    by_month = dict(zip(frame["datetime"].dt.month, frame["f"]))
    assert by_month == {1: 100.0, 2: 2.0}


# ---------------------------------------------------------------------------
# (f) 首次 publish（无旧代）不 rglob
# ---------------------------------------------------------------------------


def test_first_gen_no_rglob(tmp_path):
    reset_generation_counters()
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-15")
    m.materialize({"f": _ser({(d1, "A"): 1.0})}, universe="u")
    c = get_generation_counters()
    assert c["generation_rglob_discovery_count"] == 0
    assert c["generation_inventory_materialize_count"] == 0
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    assert len(_read_manifest(base).get("partition_inventory", [])) == 1


# ---------------------------------------------------------------------------
# (g) block 布局 COW 用 inventory，既有 block inode 复用、不 rglob
# ---------------------------------------------------------------------------


def test_block_layout_inventory_cow_no_rglob(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_MATRIX_BLOCK_LAYOUT", "1")
    reset_generation_counters()
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-15")
    s1 = m.materialize(
        {
            "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0}),
            "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
        },
        universe="u",
        production=True,
    )
    g1 = s1["generation"]
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    gen_dir = base / "generation"

    # 新增因子 c -> 新 block；既有 block 必须 COW hardlink（inode 复用）。
    s2 = m.materialize({"c": _ser({(d1, "A"): 5.0})}, universe="u", production=True)
    g2 = s2["generation"]
    assert g2 != g1
    g1dir, g2dir = gen_dir / g1, gen_dir / g2
    old_inodes = {
        str(p.relative_to(g1dir)): os.stat(p).st_ino
        for p in g1dir.rglob("block=*.parquet")
    }
    new_inodes = {
        str(p.relative_to(g2dir)): os.stat(p).st_ino
        for p in g2dir.rglob("block=*.parquet")
    }
    assert old_inodes, "上一代应有 block 文件"
    shared = set(old_inodes) & set(new_inodes)
    assert shared, "既有 block 应出现在新 generation（COW）"
    for rel, ino in old_inodes.items():
        assert new_inodes.get(rel) == ino, f"{rel} 应 inode 复用（未复制）"
    # 既有 block 字节不变
    for rel in old_inodes:
        assert (g1dir / rel).read_bytes() == (g2dir / rel).read_bytes()

    assert get_generation_counters()["generation_rglob_discovery_count"] == 0
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    assert {"a", "b", "c"} <= set(frame.columns)
    assert list(frame["a"]) == [1.0, 2.0]
    # c 只有 1 月数据 → 1 月有值、2 月 NaN（新增 block 未覆盖 2 月分区）
    jan_c = frame.loc[frame["datetime"].dt.month == 1, "c"].tolist()
    assert jan_c == [5.0]
    assert frame["c"].isna().sum() == 1


# ---------------------------------------------------------------------------
# (h) 读路径优先用 manifest inventory（无 rglob fallback）
# ---------------------------------------------------------------------------


def test_frames_from_legacy_prefers_inventory(tmp_path):
    gen = tmp_path / "gen"
    (gen / "year=2024" / "month=01").mkdir(parents=True)
    (gen / "year=2024" / "month=02").mkdir(parents=True)
    _df(1).to_parquet(gen / "year=2024" / "month=01" / "data.parquet", index=False)
    _df(2).to_parquet(gen / "year=2024" / "month=02" / "data.parquet", index=False)
    # inventory 只列出 month=01 → 读路径应只读该文件，忽略 month=02
    inv = inventory_to_dicts(
        [
            PartitionObjectRef(
                rel_path="year=2024/month=01/data.parquet",
                content_id="x",
                rows=1,
                schema_hash="s",
            )
        ]
    )
    frames = FactorMatrixMaterializer._frames_from_legacy(
        gen, {"partition_inventory": inv}, factor_ids=None,
        time_range=None, instrument_filter=None,
    )
    assert len(frames) == 1
    assert list(frames[0]["f"]) == [1.0]


def test_read_path_counter_untouched(tmp_path):
    reset_generation_counters()
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-15")
    m.materialize({"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0})}, universe="u")
    m.materialize({"f": _ser({(d1, "A"): 10.0})}, universe="u")
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    assert list(frame["f"]) == [10.0, 2.0]
    # 读路径走 inventory（写路径计数器不因读而增加）
    assert get_generation_counters()["generation_rglob_discovery_count"] == 0


# ---------------------------------------------------------------------------
# inventory JSON 序列化 round-trip
# ---------------------------------------------------------------------------


def test_inventory_serialization_roundtrip():
    inv = [
        PartitionObjectRef("year=2024/month=01/data.parquet", "c1", 5, "s1"),
        PartitionObjectRef("year=2024/month=02/block=0001.parquet", "c2", 7, "s2"),
    ]
    dicts = inventory_to_dicts(inv)
    assert dicts[0] == {
        "rel_path": "year=2024/month=01/data.parquet",
        "content_id": "c1",
        "rows": 5,
        "schema_hash": "s1",
    }
    back = inventory_from_dicts(dicts)
    assert back == inv
    assert [r.rel_path for r in back] == [
        "year=2024/month=01/data.parquet",
        "year=2024/month=02/block=0001.parquet",
    ]
