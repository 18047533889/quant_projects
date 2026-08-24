# -*- coding: utf-8
"""schema_migration 与 write_targets 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from factor_engine.storage.schema_migration import migrate_factor_lake_tree, migrate_factor_parquet_file
from factor_engine.storage.write_targets import StagingWriteTarget, resolve_write_target


def test_migrate_factor_parquet_adds_metadata_columns(tmp_path):
    path = tmp_path / "data.parquet"
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01"]),
            "asset": ["A"],
            "value": [1.0],
        }
    )
    pq.write_table(
        __import__("pyarrow").Table.from_pandas(df, preserve_index=False),
        path,
    )
    dry = migrate_factor_parquet_file(path, dry_run=True)
    assert dry["changed"] is True
    assert "calc_time" in dry["columns_added"]

    live = migrate_factor_parquet_file(path, dry_run=False)
    assert live["changed"] is True
    reread = pq.read_table(path).to_pandas()
    assert "is_valid" in reread.columns
    assert int(reread["is_valid"].iloc[0]) == 1


def test_migrate_factor_lake_tree_scans_partitions(tmp_path):
    part = tmp_path / "factors" / "f1" / "year=2024"
    part.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01"]),
            "asset": ["A"],
            "value": [1.0],
        }
    )
    pq.write_table(
        __import__("pyarrow").Table.from_pandas(df, preserve_index=False),
        part / "data.parquet",
    )
    report = migrate_factor_lake_tree(tmp_path, factor_id="f1", dry_run=False)
    assert report["files_changed"] == 1


def test_resolve_write_target_custom_dataset():
    target = resolve_write_target("staging:my_staging_lake")
    assert isinstance(target, StagingWriteTarget)
    assert target.dataset == "my_staging_lake"
