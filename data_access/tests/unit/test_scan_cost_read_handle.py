"""read/scan_cost.py + read/read_handle.py 单测。"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.read.read_handle import ReadHandle
from data_access.read.scan_cost import estimate_scan_cost, suggest_read_strategy
from data_access.store import DataAccessStore


def test_read_handle_conversions(tmp_path):
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    handle = ReadHandle(table=table)
    assert handle.rows == 3
    assert handle.columns == ["a", "b"]
    assert len(handle.to_pandas()) == 3
    df = handle.to_polars()
    assert df.height == 3
    assert handle.to_lazy().collect().height == 3
    batches = list(handle.stream(batch_size=2))
    assert sum(b.num_rows for b in batches) == 3


def test_suggest_strategy():
    from data_access.read.scan_cost import ScanCost

    small = ScanCost(dataset="d", file_count=1, total_bytes=1000, estimated_rows=100,
                     projected_columns=3, total_columns=3, remote=False)
    engine, result = suggest_read_strategy(small)
    assert engine == "duckdb" and result == "arrow"

    big = ScanCost(dataset="d", file_count=100, total_bytes=10**10, estimated_rows=50_000_000,
                   projected_columns=3, total_columns=3, remote=True)
    engine, result = suggest_read_strategy(big)
    assert result == "stream"

    big_wide = ScanCost(dataset="d", file_count=50, total_bytes=10**9, estimated_rows=50_000_000,
                        projected_columns=3, total_columns=500, remote=False)
    engine, result = suggest_read_strategy(big_wide, prefer_polars=True)
    assert engine == "polars" and result == "lazy"


def test_estimate_scan_cost(tmp_path):
    pq.write_table(pa.table({"t": ["2024-01-02"], "s": ["AAPL"], "v": [1.0]}),
                   str(tmp_path / "a.parquet"))
    (tmp_path / "datasets.yaml").write_text(
        f"""
ds:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "*.parquet"
  time_column: t
  instrument_column: s
  schema:
    t: string
    s: string
    v: double
""",
        encoding="utf-8",
    )
    store = DataAccessStore(registry=load_registry(tmp_path / "datasets.yaml"),
                            engine=DuckDBEngine(threads=2, enable_object_cache=False))
    cost = estimate_scan_cost(store, "ds", columns=["v"])
    assert cost.dataset == "ds"
    assert cost.file_count == 1
    assert cost.estimated_rows == 1
    assert cost.projected_columns == 1
    assert cost.total_columns == 3
    assert cost.remote is False
