"""read/manifest.py：数据集 manifest 构建/加载/裁剪单测。"""
from __future__ import annotations

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
