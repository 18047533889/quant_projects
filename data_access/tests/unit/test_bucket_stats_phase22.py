"""bucket 路径剪枝、predicate hive_filters、stats sidecar 测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access import reset_store
from data_access.engine import DuckDBEngine
from data_access.layout_policy import prune_glob_paths_for_buckets
from data_access.predicate import Predicate, compile_predicate
from data_access.registry import load_registry
from data_access.stats import (
    build_dataset_stats_snapshot,
    load_stats_sidecar,
    write_stats_sidecar,
)
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


def test_prune_glob_paths_for_buckets():
    paths = ["/data/year=*/month=*/bucket=*/*.parquet"]
    out = prune_glob_paths_for_buckets(paths, "bucket", [0, 3])
    assert len(out) == 2
    assert any("bucket=0" in p for p in out)
    assert any("bucket=3" in p for p in out)


def test_compile_predicate_hive_filters():
    pred = Predicate(
        time_range=("2024-01-01", "2024-12-31"),
        instrument_filter=["AAPL"],
        hive_filters={"bucket": [0, 1]},
    )
    compiled = compile_predicate(
        pred,
        time_column="timestamp",
        instrument_column="ticker",
    )
    assert '"bucket"' in compiled.where_sql
    assert compiled.params[-1] == [0, 1]


def _write_bucket_fixture(root: Path) -> None:
    for bucket in (0, 1, 2):
        out_dir = root / "year=2024" / "month=01" / f"bucket={bucket}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01"]),
                "ticker": ["AAPL"],
                "close": [1.0],
            }
        )
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_dir / "data.parquet")


def test_bucket_path_pruning_reduces_glob_count(tmp_path: Path):
    root = tmp_path / "minute"
    _write_bucket_fixture(root)
    yaml = f"""
minute_ds:
  kind: static
  access_mode: published
  layout: hive
  root: {root}
  glob: "year=*/month=*/bucket=*/*.parquet"
  partition_columns: [year, month, bucket]
  layout_policy:
    bucket:
      column: bucket
      count: 64
  time_column: timestamp
  instrument_column: ticker
  hive_partitioning: true
  union_by_name: true
  schema:
    timestamp: timestamp
    ticker: string
    close: double
"""
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml, encoding="utf-8")
    store = DataAccessStore(load_registry(cfg), DuckDBEngine())

    ds = store._registry.get("minute_ds")
    all_paths = store._resolve_paths(ds, {}, instrument_filter=None)
    pruned = store._resolve_paths(ds, {}, instrument_filter=["AAPL"])
    assert len(pruned) <= len(all_paths)
    assert any("bucket=" in p for p in pruned)


def test_stats_sidecar_roundtrip(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()
    out_dir = root / "year=2024"
    out_dir.mkdir()
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["A", "B"],
            "close": [1.0, 2.0],
        }
    )
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_dir / "data.parquet")

    yaml = f"""
ds:
  kind: static
  access_mode: published
  layout: hive
  root: {root}
  glob: "year=*/*.parquet"
  time_column: timestamp
  instrument_column: ticker
  hive_partitioning: true
  union_by_name: true
  schema:
    timestamp: timestamp
    ticker: string
    close: double
"""
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml, encoding="utf-8")
    store = DataAccessStore(load_registry(cfg), DuckDBEngine())
    snap = build_dataset_stats_snapshot(store, "ds")
    write_stats_sidecar(root, snap)
    loaded = load_stats_sidecar(root)
    assert loaded is not None
    assert loaded.num_rows == 2
    assert loaded.num_files == 1
