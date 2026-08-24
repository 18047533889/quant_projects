# -*- coding: utf-8 -*-
"""R13 factor_matrix 治理收口测试。

覆盖：
  * P0-21 read-merge-write 部分日期更新保留旧值 + 显式 tombstone
  * P0-22 旧分区读失败 → quarantine + FactorMatrixReadError（绝不按空分区覆盖）
  * P0-23 factor_version 绑定：semantic_digest 变化拒绝混列
  * P1-14 production staging→validate→manifest→publish
  * P1-15 manifest CAS：并发写版本冲突抛 FactorMatrixConcurrentWriteError
  * P1-16 重复 (datetime, asset) key → DuplicateMatrixKeyError

全部使用 tmp_path 隔离，不污染真实 factor_matrix。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_engine.storage.materialize.factor_matrix_materializer import (
    DuplicateMatrixKeyError,
    FactorMatrixConcurrentWriteError,
    FactorMatrixMaterializer,
    FactorMatrixReadError,
    FactorMatrixVersionMismatchError,
    _merge_matrix_frames,
)


def _ser(data: dict[tuple, float]) -> pd.Series:
    idx = pd.MultiIndex.from_tuples(
        list(data.keys()), names=["datetime", "instrument"]
    )
    return pd.Series(list(data.values()), index=idx, dtype="float64")


def _data_parquets(matrix_root: Path, universe: str = "u") -> list[Path]:
    base = Path(matrix_root) / f"universe={universe}" / "freq=1d"
    return [
        p
        for p in base.rglob("data.parquet")
        if ".staging" not in p.parts and ".quarantine" not in p.parts
    ]


# ---------------------------------------------------------------------------
# P0-21 同一 factor 部分日期更新会丢旧值（read-merge-write）
# ---------------------------------------------------------------------------


def test_p0_21_merge_partial_date_update_keeps_old():
    old = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset": ["A", "A", "A"],
            "factor": [1.0, 2.0, 3.0],
        }
    )
    new = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02"]),
            "asset": ["A"],
            "factor": [20.0],
        }
    )
    out = _merge_matrix_frames(old, new, value_dtype="float32")
    assert list(out["factor"]) == [1.0, 20.0, 3.0]
    assert list(out["datetime"].dt.day) == [1, 2, 3]


def test_p0_21_merge_explicit_nan_is_tombstone():
    old = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset": ["A", "A", "A"],
            "factor": [1.0, 2.0, 3.0],
        }
    )
    new = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02"]),
            "asset": ["A"],
            "factor": [np.nan],
        }
    )
    out = _merge_matrix_frames(old, new, value_dtype="float32")
    vals = out["factor"].tolist()
    assert vals[0] == 1.0 and vals[2] == 3.0
    assert np.isnan(vals[1])  # tombstone：显式 NaN 覆盖旧值


def test_p0_21_materialize_partial_date_update(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2, d3 = (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    )
    m.materialize(
        {"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0, (d3, "A"): 3.0})},
        universe="u",
    )
    # 增量只更新 D2 → 不能丢 D1/D3 旧值
    m.materialize({"f": _ser({(d2, "A"): 20.0})}, universe="u")
    frame = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u"
    )
    by_day = dict(zip(frame["datetime"].dt.day, frame["f"]))
    assert by_day[1] == 1.0
    assert by_day[2] == 20.0
    assert by_day[3] == 3.0


def test_p0_21_materialize_explicit_tombstone(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2, d3 = (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    )
    m.materialize(
        {"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0, (d3, "A"): 3.0})},
        universe="u",
    )
    m.materialize({"f": _ser({(d2, "A"): np.nan})}, universe="u")
    frame = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u"
    )
    by_day = dict(zip(frame["datetime"].dt.day, frame["f"]))
    assert by_day[1] == 1.0
    assert np.isnan(by_day[2])
    assert by_day[3] == 3.0


def test_p0_21_subset_factor_update_preserves_old_columns(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    m.materialize(
        {
            "a": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0}),
            "b": _ser({(d1, "A"): 10.0, (d2, "A"): 20.0}),
        },
        universe="u",
    )
    # 只更新因子 a：b 列必须完整保留（不能变 NaN / 改名）
    m.materialize({"a": _ser({(d2, "A"): 200.0})}, universe="u")
    frame = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u"
    )
    assert {"a", "b"} <= set(frame.columns)
    assert list(frame["b"]) == [10.0, 20.0]
    assert list(frame["a"]) == [1.0, 200.0]


# ---------------------------------------------------------------------------
# P0-22 旧分区读失败 → quarantine + FactorMatrixReadError（绝不空分区覆盖）
# ---------------------------------------------------------------------------


def test_p0_22_corrupt_parquet_quarantine_hard_fail(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    m.materialize(
        {"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0})}, universe="u"
    )
    parquets = _data_parquets(tmp_path / "matrix")
    assert len(parquets) == 1
    target = parquets[0]
    target.write_bytes(b"this is not a parquet file")

    # 增量写另一个因子 → 读失败必须 hard fail，绝不按空分区覆盖
    with pytest.raises(FactorMatrixReadError, match="quarantine"):
        m.materialize({"g": _ser({(d2, "A"): 9.0})}, universe="u")

    # R14 #2：已发布 generation immutable——坏文件只**复制**到 quarantine 留证，
    # 不移动原文件（manifest 仍引用该代，在途 reader 可能正读它）。
    qdir = target.parent / ".quarantine"
    assert list(qdir.glob("*-data.parquet"))
    assert target.exists()  # 原文件仍在（immutable），只留证副本
    # load_matrix 读当前 generation → 坏文件 fail loud（不再静默 FileNotFound）
    with pytest.raises(FactorMatrixReadError):
        FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")


def test_p0_22_read_failure_never_treated_as_empty(tmp_path):
    # 即便 recovery=True 也不再允许破坏性重建（P0-22 删除空分区 fallback）
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-01")
    m.materialize({"f": _ser({(d1, "A"): 1.0})}, universe="u")
    target = _data_parquets(tmp_path / "matrix")[0]
    target.write_bytes(b"corrupt")
    with pytest.raises(FactorMatrixReadError):
        m.materialize({"f": _ser({(d1, "A"): 2.0})}, universe="u", recovery=True)


# ---------------------------------------------------------------------------
# P0-23 factor_version 绑定（manifest.json + semantic_digest 校验）
# ---------------------------------------------------------------------------


def test_p0_23_manifest_written_with_digest(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-01")
    m.materialize(
        {"f": _ser({(d1, "A"): 1.0})},
        universe="u",
        factor_versions={"f": "digest-v1"},
    )
    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    manifest_path = base / "manifest.json"
    assert manifest_path.exists()
    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["universe"] == "u"
    assert manifest["frequency"] == "1d"
    assert manifest["factors"]["f"]["semantic_digest"] == "digest-v1"
    assert manifest["manifest_version"] == 1
    assert manifest["updated_at"]


def test_p0_23_version_mismatch_rejects_mixing(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-01")
    m.materialize(
        {"f": _ser({(d1, "A"): 1.0})},
        universe="u",
        factor_versions={"f": "digest-v1"},
    )
    # 同一 factor 换 semantic_digest → 禁止混列
    with pytest.raises(FactorMatrixVersionMismatchError, match="digest-v1"):
        m.materialize(
            {"f": _ser({(d1, "A"): 2.0})},
            universe="u",
            factor_versions={"f": "digest-v2"},
        )


def test_p0_23_manifest_without_versions_only_tracks_time(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-01")
    m.materialize({"f": _ser({(d1, "A"): 1.0})}, universe="u")
    import json

    base = tmp_path / "matrix" / "universe=u" / "freq=1d"
    manifest = json.loads((base / "manifest.json").read_text())
    # factor_versions 缺省 → 不做 digest 绑定
    assert manifest["factors"] == {}


# ---------------------------------------------------------------------------
# P1-14 production staging→validate→manifest→publish
# ---------------------------------------------------------------------------


def test_p1_14_production_staging_publish(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    summary = m.materialize(
        {"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0})},
        universe="u",
        production=True,
    )
    assert summary["manifest_version"] == 1
    frame = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u"
    )
    assert len(frame) == 2
    assert list(frame["f"]) == [1.0, 2.0]
    # staging 已清理，未残留
    for pq in (tmp_path / "matrix").rglob(".staging"):
        assert not list(pq.rglob("data.parquet"))
    # manifest 已写
    assert (tmp_path / "matrix" / "universe=u" / "freq=1d" / "manifest.json").exists()


def test_p1_14_production_incremental_read_merge(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2, d3 = (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    )
    m.materialize(
        {"f": _ser({(d1, "A"): 1.0, (d2, "A"): 2.0, (d3, "A"): 3.0})},
        universe="u",
        production=True,
    )
    m.materialize({"f": _ser({(d2, "A"): 20.0})}, universe="u", production=True)
    frame = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u"
    )
    by_day = dict(zip(frame["datetime"].dt.day, frame["f"]))
    assert by_day == {1: 1.0, 2: 20.0, 3: 3.0}


# ---------------------------------------------------------------------------
# P1-15 manifest CAS（flock 之外的乐观并发控制）
# ---------------------------------------------------------------------------


def test_p1_15_cas_conflict_raises(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-01")
    m.materialize({"f": _ser({(d1, "A"): 1.0})}, universe="u")
    # 期望版本 0 但当前已是 1 → CAS 失败
    with pytest.raises(FactorMatrixConcurrentWriteError, match="CAS"):
        m.materialize(
            {"f": _ser({(d1, "A"): 2.0})},
            universe="u",
            expected_manifest_version=0,
        )


def test_p1_15_cas_match_passes(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-01")
    m.materialize({"f": _ser({(d1, "A"): 1.0})}, universe="u")
    summary = m.materialize(
        {"f": _ser({(d1, "A"): 2.0})},
        universe="u",
        expected_manifest_version=1,
    )
    assert summary["manifest_version"] == 2


# ---------------------------------------------------------------------------
# P1-16 重复 key 加载契约损坏 → DuplicateMatrixKeyError
# ---------------------------------------------------------------------------


def test_p1_16_duplicate_key_raises(tmp_path):
    base = tmp_path / "matrix" / "universe=u" / "freq=1d" / "year=2024" / "month=01"
    base.mkdir(parents=True)
    bad = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "asset": ["A", "A"],
            "f": [1.0, 2.0],
        }
    )
    bad.to_parquet(base / "data.parquet", index=False)
    with pytest.raises(DuplicateMatrixKeyError, match="重复"):
        FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u")


def test_p1_16_load_skips_staging_and_quarantine(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-01")
    m.materialize({"f": _ser({(d1, "A"): 1.0})}, universe="u", production=True)
    # 人为造一个 .staging/data.parquet，load 必须跳过它
    target = _data_parquets(tmp_path / "matrix")[0]
    staging = target.parent / ".staging" / "data.parquet"
    staging.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-02-01"]),
            "asset": ["A"],
            "f": [999.0],
        }
    ).to_parquet(staging, index=False)
    frame = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u"
    )
    assert len(frame) == 1
    assert list(frame["f"]) == [1.0]
