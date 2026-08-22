# -*- coding: utf-8 -*-
"""R14 #2 factor_matrix generation fail-closed 测试。

外部 AI 复查第二轮 P0：`load_matrix` 对「manifest 有 generation 但 generation
目录消失」仍然 fail-open（legacy fallback 会把 previous/orphan/legacy 数据捞
出来）；`_read_existing_or_quarantine` 会把已发布 generation 的原文件 os.replace
到 quarantine——新 generation 构造失败后 manifest 仍指向被移走文件的旧代。

R14 #2 修复：
  * manifest 声明 generation 但目录缺失 → ``FactorMatrixCorruptionError``（fail
    closed），绝不 legacy fallback；
  * 已发布 generation 文件损坏 → 只复制到 quarantine 留证、**不移原文件**并
    hard fail（该代 immutable，在途 reader 仍按 manifest 引用它）；
  * materialize 对旧代目录缺失同样 fail-closed（否则 read-merge-write 静默丢历史）。
"""

from __future__ import annotations

import pandas as pd
import pytest

from storage.materialize.factor_matrix_materializer import (
    FactorMatrixCorruptionError,
    FactorMatrixMaterializer,
    FactorMatrixReadError,
    _read_manifest,
)


def _ser(data: dict[tuple, float]) -> pd.Series:
    idx = pd.MultiIndex.from_tuples(
        list(data.keys()), names=["datetime", "instrument"]
    )
    return pd.Series(list(data.values()), index=idx, dtype="float64")


def _write_manifest(base, gen_id: str) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    manifest = {
        "universe": "u",
        "frequency": "1d",
        "factors": {},
        "updated_at": "2026-08-09T00:00:00Z",
        "manifest_version": 1,
        "generation": gen_id,
    }
    from storage.materialize.factor_matrix_materializer import (
        _write_manifest_atomic,
    )

    _write_manifest_atomic(base, manifest)
    return manifest


def test_r14_load_missing_generation_dir_fails_closed(tmp_path):
    """manifest.generation 指向缺失目录 → 抛 FactorMatrixCorruptionError。

    即使 base 下仍残留一个 legacy `year=*/month=*/data.parquet`，也不能 fallback
    读到它——那会把「当前发布代损坏」伪装成可读。
    """
    matrix_root = tmp_path / "matrix"
    base = matrix_root / "universe=u" / "freq=1d"
    # 构造 legacy 数据（诱饵）：base 直接放分区
    legacy = base / "year=2024" / "month=01" / "data.parquet"
    legacy.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-15"]),
            "asset": ["A"],
            "f": [999.0],
        }
    ).to_parquet(legacy, index=False)
    # manifest 声明 G_MISSING，但目录不存在
    _write_manifest(base, "G_MISSING")
    assert not (base / "generation" / "G_MISSING").is_dir()

    with pytest.raises(FactorMatrixCorruptionError):
        FactorMatrixMaterializer.load_matrix(matrix_root, universe="u")


def test_r14_load_missing_generation_dir_ignores_orphan(tmp_path):
    """fail-closed 同时不能 glob 到 orphan generation 数据。"""
    matrix_root = tmp_path / "matrix"
    base = matrix_root / "universe=u" / "freq=1d"
    # 构造一个 orphan generation 目录（有数据，但 manifest 不指向它）
    orphan = base / "generation" / "ORPHAN" / "year=2024" / "month=01"
    orphan.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-15"]),
            "asset": ["A"],
            "f": [42.0],
        }
    ).to_parquet(orphan / "data.parquet", index=False)
    _write_manifest(base, "G_MISSING")

    with pytest.raises(FactorMatrixCorruptionError):
        FactorMatrixMaterializer.load_matrix(matrix_root, universe="u")


def test_r14_materialize_missing_old_generation_fails_closed(tmp_path):
    """materialize 时 manifest 有 generation 但旧代目录缺失 → fail-closed。

    旧实现把 ``old_gen_dir=None`` 当「无旧代」——read-merge-write 从零重建会静默
    丢掉全部未触达分区历史。现在必须 hard fail。
    """
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    _write_manifest(base, "G_MISSING")

    with pytest.raises(FactorMatrixCorruptionError):
        m.materialize(
            {"f": _ser({(pd.Timestamp("2024-01-15"), "A"): 1.0})},
            universe="u",
        )


def test_r14_immutable_quarantine_copies_not_moves(tmp_path):
    """已发布 generation 文件损坏 → 只复制留证、原文件保留，并 hard fail。"""
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-15")
    summary = m.materialize({"f": _ser({(d1, "A"): 1.0})}, universe="u")
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    gid = summary["generation"]
    parts = list((base / "generation" / gid).rglob("data.parquet"))
    assert parts, "generation 下应有分区文件"
    good = parts[0]
    assert good.exists()

    # 用不可读的坏文件替换该代的一个分区（损坏该 generation 文件内容）
    good.write_bytes(b"not a parquet file at all")
    with pytest.raises(FactorMatrixReadError):
        m.materialize({"f": _ser({(d1, "A"): 2.0})}, universe="u")
    # immutable：原文件仍保留（在途 reader / manifest 引用），未被 os.replace 移走
    assert good.exists()
    # 留证副本在 .quarantine/
    qfiles = list((good.parent / ".quarantine").rglob("*"))
    assert qfiles, "应留下 quarantine 留证副本"
    # manifest 仍指向原 generation（新 generation 构造失败被清理）
    assert _read_manifest(base)["generation"] == gid
