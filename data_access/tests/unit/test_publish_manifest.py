# -*- coding: utf-8
"""publish manifest 与 revision dedup 单元测试。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.read.adapters import arrow_table_to_multiindex_columns
from data_access.read.key_policy import KeyPolicy
from data_access.write.publish_manifest import read_publish_manifest, write_publish_manifest
from data_access.read.read_contract import merge_sql_data_snapshots, build_data_snapshot


def test_write_publish_manifest_atomic(tmp_path: Path):
    target = tmp_path / "published" / "f1"
    target.mkdir(parents=True)
    pq.write_table(pa.table({"x": [1]}), target / "data.parquet")

    path = write_publish_manifest(
        target,
        staging_name="stg",
        target_name="pub",
        params={"factor_id": "f1"},
        rows=1,
        archive_path=None,
        elapsed_ms=12.5,
    )
    assert path.exists()
    manifest = read_publish_manifest(target)
    assert manifest is not None
    assert manifest["rows"] == 1
    assert manifest["params"]["factor_id"] == "f1"
    assert manifest["file_count"] == 1


def test_revision_column_dedup_keeps_latest():
    tbl = pa.table(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "instrument": ["AAPL", "AAPL"],
            "close": [100.0, 200.0],
            "revision": [1, 3],
        }
    )
    out = arrow_table_to_multiindex_columns(
        tbl,
        timestamp_column="timestamp",
        instrument_column="instrument",
        value_columns=["close"],
        key_policy=KeyPolicy(
            invalid_key="error",
            duplicate_key="keep_last",
            duplicate_resolution="revision",
        ),
    )
    assert len(out["close"]) == 1
    assert out["close"].iloc[0] == pytest.approx(200.0)


def test_merge_sql_snapshots_differs_from_single():
    a = build_data_snapshot(
        dataset="ds_a",
        registry_hash="reg",
        schema={"x": "int"},
        paths=["/tmp/a/*.parquet"],
        params={"k": "a"},
    )
    b = build_data_snapshot(
        dataset="ds_b",
        registry_hash="reg",
        schema={"y": "double"},
        paths=["/tmp/b/*.parquet"],
        params={"k": "b"},
    )
    merged = merge_sql_data_snapshots([a, b], registry_hash="reg")
    assert merged.snapshot_id != a.snapshot_id
    assert merged.snapshot_id != b.snapshot_id
    assert "ds_a" in merged.dataset and "ds_b" in merged.dataset
