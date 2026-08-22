"""manifest.py partial epoch freshness fail-closed 回归测试。

#P0-FRESHNESS-AUDIT：验证 partial dual-epoch state (source_epoch 存在但
manifest_built_epoch 缺失，或反之) 必须 fail-closed → 不 fresh → planner 回退
glob 全文件列表，避免用损坏/不完整的 manifest min/max 做错误裁剪。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.read.manifest import (
    DatasetManifest,
    ManifestFile,
    is_manifest_fresh,
    manifest_root_for_paths,
    MANIFEST_FILENAME,
    _MANIFEST_META_FILENAME,
)


def test_is_fresh_epoch_partial_source_only_fails_closed(tmp_path):
    """source_epoch 存在但 manifest_built_epoch=None → fail-closed (False)。"""
    manifest = DatasetManifest(
        dataset="test",
        source_epoch="42",
        manifest_built_epoch=None,
        manifest_epoch=None,
    )
    # Partial dual-epoch state 必须 fail-closed
    assert manifest.is_fresh_epoch is False


def test_is_fresh_epoch_partial_built_only_fails_closed(tmp_path):
    """manifest_built_epoch 存在但 source_epoch=None → fail-closed (False)。"""
    manifest = DatasetManifest(
        dataset="test",
        source_epoch=None,
        manifest_built_epoch="42",
        manifest_epoch=None,
    )
    # Partial dual-epoch state 必须 fail-closed
    assert manifest.is_fresh_epoch is False


def test_is_fresh_epoch_complete_dual_epoch_equal_is_fresh(tmp_path):
    """双 epoch 都存在且相等 → fresh (True)。"""
    manifest = DatasetManifest(
        dataset="test",
        source_epoch="42",
        manifest_built_epoch="42",
        manifest_epoch=None,
    )
    assert manifest.is_fresh_epoch is True


def test_is_fresh_epoch_complete_dual_epoch_mismatch_is_stale(tmp_path):
    """双 epoch 都存在但不等 → stale (False)。"""
    manifest = DatasetManifest(
        dataset="test",
        source_epoch="43",
        manifest_built_epoch="42",
        manifest_epoch=None,
    )
    assert manifest.is_fresh_epoch is False


def test_is_fresh_epoch_legacy_single_epoch_is_fresh(tmp_path):
    """老单 epoch 格式（source/built 都缺失，manifest_epoch 存在）→ fresh (True)。"""
    manifest = DatasetManifest(
        dataset="test",
        source_epoch=None,
        manifest_built_epoch=None,
        manifest_epoch="42",
    )
    assert manifest.is_fresh_epoch is True


def test_is_fresh_epoch_no_epoch_at_all_is_stale(tmp_path):
    """完全无 epoch → stale (False)。"""
    manifest = DatasetManifest(
        dataset="test",
        source_epoch=None,
        manifest_built_epoch=None,
        manifest_epoch=None,
    )
    assert manifest.is_fresh_epoch is False


def test_is_manifest_fresh_partial_source_only_fails_closed(tmp_path):
    """is_manifest_fresh() 从 sidecar 读到 partial source_epoch → fail-closed。"""
    # 写入 3 个数据文件
    for i in range(3):
        pq.write_table(
            pa.table({"col": [i]}),
            str(tmp_path / f"data_{i}.parquet"),
        )
    # 写入 partial dual-epoch sidecar（source 存在，built 缺失）
    meta_path = tmp_path / _MANIFEST_META_FILENAME
    meta_path.write_text(
        json.dumps({
            "dataset": "test",
            "source_epoch": "42",
            # manifest_built_epoch 缺失（partial write / corruption）
            "file_count": 3,
        }),
        encoding="utf-8",
    )
    # 写入 manifest parquet（配合 sidecar）
    manifest = DatasetManifest(
        dataset="test",
        files=(
            ManifestFile(path=str(tmp_path / "data_0.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_1.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_2.parquet"), rows=1, bytes=100),
        ),
    )
    table = manifest.to_table()
    pq.write_table(table, str(tmp_path / MANIFEST_FILENAME))

    # 加载后检查：partial state 必须被判定为 not fresh
    loaded = DatasetManifest.load(tmp_path)
    assert loaded is not None
    glob_paths = [str(tmp_path / "*.parquet")]
    assert is_manifest_fresh(loaded, glob_paths) is False


def test_is_manifest_fresh_partial_built_only_fails_closed(tmp_path):
    """is_manifest_fresh() 从 sidecar 读到 partial manifest_built_epoch → fail-closed。"""
    for i in range(3):
        pq.write_table(
            pa.table({"col": [i]}),
            str(tmp_path / f"data_{i}.parquet"),
        )
    meta_path = tmp_path / _MANIFEST_META_FILENAME
    meta_path.write_text(
        json.dumps({
            "dataset": "test",
            # source_epoch 缺失（partial write / corruption）
            "manifest_built_epoch": "42",
            "file_count": 3,
        }),
        encoding="utf-8",
    )
    manifest = DatasetManifest(
        dataset="test",
        files=(
            ManifestFile(path=str(tmp_path / "data_0.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_1.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_2.parquet"), rows=1, bytes=100),
        ),
    )
    table = manifest.to_table()
    pq.write_table(table, str(tmp_path / MANIFEST_FILENAME))

    loaded = DatasetManifest.load(tmp_path)
    assert loaded is not None
    glob_paths = [str(tmp_path / "*.parquet")]
    assert is_manifest_fresh(loaded, glob_paths) is False


def test_is_manifest_fresh_complete_dual_epoch_equal_is_fresh(tmp_path):
    """is_manifest_fresh() 读到完整且相等双 epoch → fresh。"""
    for i in range(3):
        pq.write_table(
            pa.table({"col": [i]}),
            str(tmp_path / f"data_{i}.parquet"),
        )
    meta_path = tmp_path / _MANIFEST_META_FILENAME
    meta_path.write_text(
        json.dumps({
            "dataset": "test",
            "source_epoch": "42",
            "manifest_built_epoch": "42",
            "file_count": 3,
        }),
        encoding="utf-8",
    )
    manifest = DatasetManifest(
        dataset="test",
        files=(
            ManifestFile(path=str(tmp_path / "data_0.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_1.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_2.parquet"), rows=1, bytes=100),
        ),
    )
    table = manifest.to_table()
    pq.write_table(table, str(tmp_path / MANIFEST_FILENAME))

    loaded = DatasetManifest.load(tmp_path)
    assert loaded is not None
    glob_paths = [str(tmp_path / "*.parquet")]
    assert is_manifest_fresh(loaded, glob_paths) is True


def test_is_manifest_fresh_complete_dual_epoch_mismatch_is_stale(tmp_path):
    """is_manifest_fresh() 读到完整但不等双 epoch → stale。"""
    for i in range(3):
        pq.write_table(
            pa.table({"col": [i]}),
            str(tmp_path / f"data_{i}.parquet"),
        )
    meta_path = tmp_path / _MANIFEST_META_FILENAME
    meta_path.write_text(
        json.dumps({
            "dataset": "test",
            "source_epoch": "43",  # mutation 后递增
            "manifest_built_epoch": "42",  # manifest 未重建
            "file_count": 3,
        }),
        encoding="utf-8",
    )
    manifest = DatasetManifest(
        dataset="test",
        files=(
            ManifestFile(path=str(tmp_path / "data_0.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_1.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_2.parquet"), rows=1, bytes=100),
        ),
    )
    table = manifest.to_table()
    pq.write_table(table, str(tmp_path / MANIFEST_FILENAME))

    loaded = DatasetManifest.load(tmp_path)
    assert loaded is not None
    glob_paths = [str(tmp_path / "*.parquet")]
    assert is_manifest_fresh(loaded, glob_paths) is False


def test_is_manifest_fresh_legacy_single_epoch_is_fresh(tmp_path):
    """is_manifest_fresh() 读到老单 epoch 格式 → fresh（兼容）。"""
    for i in range(3):
        pq.write_table(
            pa.table({"col": [i]}),
            str(tmp_path / f"data_{i}.parquet"),
        )
    meta_path = tmp_path / _MANIFEST_META_FILENAME
    meta_path.write_text(
        json.dumps({
            "dataset": "test",
            "manifest_epoch": "42",  # 老格式：只有单 epoch
            "file_count": 3,
        }),
        encoding="utf-8",
    )
    manifest = DatasetManifest(
        dataset="test",
        files=(
            ManifestFile(path=str(tmp_path / "data_0.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_1.parquet"), rows=1, bytes=100),
            ManifestFile(path=str(tmp_path / "data_2.parquet"), rows=1, bytes=100),
        ),
    )
    table = manifest.to_table()
    pq.write_table(table, str(tmp_path / MANIFEST_FILENAME))

    loaded = DatasetManifest.load(tmp_path)
    assert loaded is not None
    glob_paths = [str(tmp_path / "*.parquet")]
    assert is_manifest_fresh(loaded, glob_paths) is True
