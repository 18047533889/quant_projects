"""read/manifest.py：数据集 manifest 构建/加载/裁剪单测。"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.read.manifest import (
    DatasetManifest,
    is_manifest_fresh,
    manifest_root_for_paths,
)
from data_access.store import DataAccessStore


def _make_store(tmp_path):
    for day in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        pq.write_table(
            pa.table({"date": [day], "symbol": ["AAPL"], "close": [150.0]}),
            str(tmp_path / f"{day}.parquet"),
        )
    (tmp_path / "datasets.yaml").write_text(
        f"""
daily_ds:
  kind: static
  access_mode: published
  layout: plain
  format: parquet
  root: {tmp_path}
  glob: "*.parquet"
  time_column: date
  instrument_column: symbol
  schema:
    date: string
    symbol: string
    close: double
""",
        encoding="utf-8",
    )
    reg = load_registry(tmp_path / "datasets.yaml")
    return DataAccessStore(registry=reg, engine=DuckDBEngine(threads=2, enable_object_cache=False))


def test_build_load_prune(tmp_path):
    store = _make_store(tmp_path)
    res = store.build_dataset_manifest("daily_ds")
    assert res["files"] == 3 and res["rows"] == 3

    manifest = DatasetManifest.load(tmp_path)
    assert manifest is not None
    assert manifest.time_column == "date"
    pruned = manifest.prune(time_range=("2024-01-03", "2024-01-03"))
    assert len(pruned) == 1 and "2024-01-03" in pruned[0]
    assert len(manifest.manifest_generation_id or "") == 32
    assert len(manifest.dataset_version) == 64
    assert len(manifest.partition_version) == 64
    assert all(len(value) == 64 for value in manifest.schema_hashes)


def test_load_rejects_malformed_generation_identity(tmp_path):
    store = _make_store(tmp_path)
    store.build_dataset_manifest("daily_ds")
    meta_path = tmp_path / "_manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["manifest_generation_id"] = "not-a-uuid"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert DatasetManifest.load(tmp_path) is None


def _make_int_time_store(tmp_path):
    """int64 时间列数据集：文件 max 为 9 和 10，字符串比较会出错。"""
    for day in [9, 10]:
        pq.write_table(
            pa.table({"ts": [day], "symbol": ["AAPL"], "close": [150.0]}),
            str(tmp_path / f"{day}.parquet"),
        )
    (tmp_path / "datasets.yaml").write_text(
        f"""
int_ds:
  kind: static
  access_mode: published
  layout: plain
  format: parquet
  root: {tmp_path}
  glob: "*.parquet"
  time_column: ts
  instrument_column: symbol
  schema:
    ts: int64
    symbol: string
    close: double
""",
        encoding="utf-8",
    )
    reg = load_registry(tmp_path / "datasets.yaml")
    return DataAccessStore(registry=reg, engine=DuckDBEngine(threads=2, enable_object_cache=False))


def test_prune_int64_time_typed_comparison(tmp_path):
    """#19 int64 时间列必须数值比较：start=10 时 max=9 的文件要裁掉。"""
    store = _make_int_time_store(tmp_path)
    store.build_dataset_manifest("int_ds")
    manifest = DatasetManifest.load(tmp_path)
    assert manifest is not None and manifest.time_dtype == "int64"
    pruned = manifest.prune(time_range=(10, 15))
    assert len(pruned) == 1 and "10.parquet" in pruned[0]
    assert "9.parquet" not in pruned[0]


def test_read_empty_after_manifest_prune(tmp_path):
    """#21 manifest 确认无匹配文件 → 返回空结果，绝不回退全量扫描。"""
    store = _make_store(tmp_path)
    store.build_dataset_manifest("daily_ds")
    h = store.read("daily_ds", time_range=("2025-01-01", "2025-12-31"))
    tbl = h.to_arrow()
    assert tbl.num_rows == 0
    assert set(tbl.column_names) == {"date", "symbol", "close"}


def test_freshness_ignores_manifest_file(tmp_path):
    store = _make_store(tmp_path)
    store.build_dataset_manifest("daily_ds")
    manifest = DatasetManifest.load(tmp_path)
    paths = store._resolve_raw_paths(
        store.registry.get("daily_ds"), time_range=None, params={}
    )
    assert is_manifest_fresh(manifest, paths) is True


def test_manifest_root(tmp_path):
    paths = [f"{tmp_path}/year=2024/*.parquet"]
    root = manifest_root_for_paths(paths)
    assert str(root) == f"{tmp_path}/year=2024"


def test_read_prunes_to_single_file(tmp_path):
    store = _make_store(tmp_path)
    store.build_dataset_manifest("daily_ds")
    paths = store._prepare_dataset_read(
        store.registry.get("daily_ds"),
        time_range=("2024-01-03", "2024-01-03"),
        params={},
    )
    assert len(paths) == 1 and "2024-01-03" in paths[0]
