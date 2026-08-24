# -*- coding: utf-8 -*-
"""R14 #1 factor_matrix generation-atomic publish 测试。

旧实现「先写 manifest、再逐分区 os.replace(staging, live)」在 publish 中途崩溃会
出现 manifest 已新、物理数据半新半旧的 mixed generation，而 load_matrix 直接 glob
live data.parquet 读得到它。R14 #1 改为：

  * 整个 (universe, freq) 是一个 generation：完整写入 ``generation/<gid>/``
    （未触达分区 copy-on-write 硬链接），全部 validate 后只原子切一次
    ``manifest.json`` 的 ``generation`` 指针；
  * reader 只读 ``manifest.generation`` 指向的那一代。

本文件模拟「崩溃在不同点停止」：reader 只能看到完整 old generation 或完整
new generation，绝无半新半旧。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from factor_engine.storage.materialize.factor_matrix_materializer import (
    FactorMatrixMaterializer,
    _read_manifest,
)


def _ser(data: dict[tuple, float]) -> pd.Series:
    idx = pd.MultiIndex.from_tuples(
        list(data.keys()), names=["datetime", "instrument"]
    )
    return pd.Series(list(data.values()), index=idx, dtype="float64")


def _manifest(base: Path) -> dict:
    return json.loads((base / "manifest.json").read_text())


def _gen_dirs(base: Path) -> list[Path]:
    gdir = base / "generation"
    if not gdir.is_dir():
        return []
    return sorted(p for p in gdir.iterdir() if p.is_dir())


def test_r14_two_month_matrix_two_partitions(tmp_path):
    """跨两个月 → 两个 partition，load 能合并出完整矩阵。"""
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    m.materialize(
        {
            "f": _ser(
                {
                    (pd.Timestamp("2024-01-15"), "A"): 1.0,
                    (pd.Timestamp("2024-02-15"), "A"): 2.0,
                }
            )
        },
        universe="u",
    )
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    # 两个分区都在同一个 generation 目录下
    assert len(_gen_dirs(base)) == 1
    gens = list((base / "generation").glob("*/year=*/month=*/data.parquet"))
    assert len(gens) == 2
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    assert len(frame) == 2


def test_r14_reader_isolation_while_new_generation_half_written(tmp_path):
    """模拟「新 generation 只写了一半就崩溃」：reader 必须仍读到完整 old gen。

    构造两个月的数据 → 第一次 publish（gen A，manifest 指向 A）。然后手工模拟
    第二次 publish 只写完第一个 partition 就崩溃：写 ``generation/<B>/year=01``
    的 data.parquet（新值 999），但**不**切 manifest。load_matrix 必须返回完整
    old generation（两月旧值），绝不能混入 999。
    """
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-15")
    summary = m.materialize({"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0})}, universe="u")
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    gen_a = summary["generation"]

    # 模拟崩溃：新 generation B 只写了 1/2 个分区（新值 999），manifest 未切。
    gen_b = uuid.uuid4().hex
    partial_path = (
        base / "generation" / gen_b / "year=2024" / "month=01" / "data.parquet"
    )
    partial_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-15"]),
            "asset": ["A"],
            "f": [999.0],
        }
    ).to_parquet(partial_path, index=False)

    # reader 只看 manifest 指向的 gen A —— 完整旧值，无 mixed。
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    by_month = dict(zip(frame["datetime"].dt.month, frame["f"]))
    assert by_month == {1: 1.0, 2: 2.0}
    assert _manifest(base)["generation"] == gen_a
    # 孤儿 gen B 存在但不可见
    assert (base / "generation" / gen_b).is_dir()


def test_r14_manifest_flip_is_single_atomic_switch(tmp_path):
    """publish 成功后 manifest.generation == 新 gen，且旧 gen 完整保留给在途 reader。

    三次 publish 后磁盘上只保留当前 + 上一代（GC 上限 2），load 始终正确。
    """
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-15")
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"

    s1 = m.materialize({"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0})}, universe="u")
    g1 = s1["generation"]
    assert _manifest(base)["generation"] == g1

    s2 = m.materialize({"f": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0})}, universe="u")
    g2 = s2["generation"]
    assert _manifest(base)["generation"] == g2
    # GC 保留上一代 g1（在途 reader 仍可能按旧 manifest 读它）
    assert {p.name for p in _gen_dirs(base)} == {g1, g2}

    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    assert list(frame["f"]) == [10.0, 20.0]

    s3 = m.materialize({"f": _ser({(d1, "A"): 100.0, (d2, "A"): 200.0})}, universe="u")
    g3 = s3["generation"]
    # 第三次 publish → GC 掉最早一代 g1，最多保留 2 代
    assert {p.name for p in _gen_dirs(base)} == {g2, g3}
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    assert list(frame["f"]) == [100.0, 200.0]


def test_r14_inflight_reader_sees_previous_generation_until_flip(tmp_path):
    """reader 读到旧 manifest（gen A）后、publish 切到 gen B 前，gen A 文件必须
    仍在（上一代不即时删除）——否则在途 reader 会读到被 GC 的半套文件。"""
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-15")
    s1 = m.materialize({"f": _ser({(d1, "A"): 1.0})}, universe="u")
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    g1 = s1["generation"]
    # 第二次 publish 前，先「模拟在途 reader 拿到 old manifest」
    old_manifest = _manifest(base)
    assert old_manifest["generation"] == g1
    # publish 第二次
    m.materialize({"f": _ser({(d1, "A"): 2.0})}, universe="u")
    # 上一代 gen 目录仍然完整保留（供在途 reader 读完）
    assert (base / "generation" / g1).is_dir()
    assert list((base / "generation" / g1).rglob("data.parquet"))


def test_r14_research_and_production_both_generation_based(tmp_path):
    """research 与 production 走同一 generation 模型：partial 更新跨代保留。"""
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-15")
    m.materialize(
        {
            "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0}),
            "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
        },
        universe="u",
        production=True,
    )
    # 只增量因子 a：b 列经 hardlink 保留，a 的 1/2 月被更新
    m.materialize({"a": _ser({(d1, "A"): 100.0})}, universe="u", production=True)
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")
    assert {"a", "b"} <= set(frame.columns)
    by_month = dict(zip(frame["datetime"].dt.month, frame["a"]))
    assert by_month == {1: 100.0, 2: 2.0}
    assert list(frame["b"]) == [10.0, 20.0]


def test_r14_orphan_generation_cleaned_on_failure(tmp_path):
    """publish 中途抛错（如 digest 变化）→ 未引用的孤儿 generation 被清理。"""
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-15")
    m.materialize(
        {"f": _ser({(d1, "A"): 1.0})},
        universe="u",
        factor_versions={"f": "digest-v1"},
    )
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    from factor_engine.storage.materialize.factor_matrix_materializer import (
        FactorMatrixVersionMismatchError,
    )

    try:
        m.materialize(
            {"f": _ser({(d1, "A"): 2.0})},
            universe="u",
            factor_versions={"f": "digest-v2"},
        )
        raise AssertionError("应抛 FactorMatrixVersionMismatchError")
    except FactorMatrixVersionMismatchError:
        pass
    # 失败时孤儿 generation 目录被清理，只留被 manifest 引用的那代
    assert len(_gen_dirs(base)) == 1
    assert _manifest(base)["generation"]
