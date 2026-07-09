"""dataset footer 统计与 read_auto 路由。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access import reset_store
from data_access.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.stats import (
    DatasetReadStats,
    estimate_parquet_rows,
    expand_parquet_paths,
    suggest_read_mode,
)
from data_access.store import DataAccessStore


def _write_fixture(root: Path, rows: int) -> None:
    out_dir = root / "year=2024"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "align_time": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "ticker": ["AAPL"] * rows,
            "close": range(rows),
        }
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_dir / "data.parquet")


def _store(tmp_path: Path, yaml: str) -> DataAccessStore:
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml, encoding="utf-8")
    registry = load_registry(cfg)
    return DataAccessStore(registry, DuckDBEngine())


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


def test_expand_and_estimate_parquet_rows(tmp_path: Path):
    root = tmp_path / "data"
    _write_fixture(root, 42)
    files = expand_parquet_paths([str(root / "year=*" / "data.parquet")])
    assert len(files) == 1
    assert estimate_parquet_rows(files) == 42


def test_suggest_read_mode_thresholds():
    assert suggest_read_mode(100, arrow_max_rows=1_000, stream_min_rows=500) == "arrow"
    assert suggest_read_mode(2_000_000, arrow_max_rows=1_000, stream_min_rows=500) == "stream"
    assert suggest_read_mode(
        2_000_000,
        arrow_max_rows=1_000,
        stream_min_rows=500,
        prefer_polars=True,
    ) == "polars"


def test_read_auto_small_dataset_uses_arrow(tmp_path: Path):
    root = tmp_path / "aggs"
    _write_fixture(root, 10)
    yaml = f"""
test_small:
  kind: static
  access_mode: published
  root: "{root.as_posix()}"
  glob: "year=*/*.parquet"
  layout: hive
  time_column: align_time
  instrument_column: ticker
  hive_partitioning: true
"""
    store = _store(tmp_path, yaml)
    stats = store.dataset_read_stats("test_small")
    assert isinstance(stats, DatasetReadStats)
    assert stats.estimated_rows == 10
    assert stats.suggested_mode == "arrow"

    tbl = store.read_auto(
        "test_small",
        columns=["align_time", "ticker", "close"],
    )
    assert tbl.num_rows == 10


def test_read_auto_large_dataset_uses_stream(tmp_path: Path, monkeypatch):
    root = tmp_path / "big"
    _write_fixture(root, 50)
    yaml = f"""
test_big:
  kind: static
  access_mode: published
  root: "{root.as_posix()}"
  glob: "year=*/*.parquet"
  layout: hive
  time_column: align_time
  instrument_column: ticker
  hive_partitioning: true
"""
    store = _store(tmp_path, yaml)
    monkeypatch.setenv("DATA_ACCESS_READ_AUTO_ARROW_MAX_ROWS", "10")
    monkeypatch.setenv("DATA_ACCESS_READ_AUTO_STREAM_MIN_ROWS", "20")

    stats = store.dataset_read_stats("test_big")
    assert stats.suggested_mode == "stream"

    tbl = store.read_auto(
        "test_big",
        columns=["align_time", "ticker", "close"],
        mode="auto",
    )
    assert tbl.num_rows == 50
