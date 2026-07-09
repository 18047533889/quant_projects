"""delete_rows dry_run / max_rows 单元测试。"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data_access.paths import PathAuthorizer
from data_access.registry import ParametricDataset
from data_access.upsert import delete_rows_from_dataset


@pytest.fixture
def staging_dataset(tmp_path):
    return ParametricDataset(
        name="factor_lake_staging",
        access_mode="staging",
        layout="hive",
        time_column="datetime",
        instrument_column="asset",
        hive_partitioning=True,
        union_by_name=True,
        root_template=str(tmp_path / "factors/{factor_id}"),
        glob_template="year=*/*.parquet",
        params_schema={"factor_id": "str"},
        schema={},
    )


def test_delete_rows_dry_run_counts_without_mutating(staging_dataset, tmp_path):
    factor_dir = tmp_path / "factors" / "test_f" / "year=2024"
    factor_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "asset": ["A", "B"],
            "value": [1.0, 2.0],
        }
    )
    pq.write_table(
        __import__("pyarrow").Table.from_pandas(df, preserve_index=False),
        factor_dir / "data.parquet",
    )

    authorizer = PathAuthorizer(allowed_roots=[tmp_path])
    target_dir = tmp_path / "factors" / "test_f"
    result = delete_rows_from_dataset(
        ds=staging_dataset,
        authorizer=authorizer,
        target_dir=target_dir,
        time_column="datetime",
        start="2024-05-01",
        dry_run=True,
        params={"factor_id": "test_f"},
    )
    assert result["dry_run"] is True
    assert result["rows_deleted"] == 1
    reread = pq.read_table(factor_dir / "data.parquet").to_pydict()
    assert len(reread["asset"]) == 2


def test_delete_rows_max_rows_guard(staging_dataset, tmp_path):
    factor_dir = tmp_path / "factors" / "test_f" / "year=2024"
    factor_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "asset": ["A", "B"],
            "value": [1.0, 2.0],
        }
    )
    pq.write_table(
        __import__("pyarrow").Table.from_pandas(df, preserve_index=False),
        factor_dir / "data.parquet",
    )
    authorizer = PathAuthorizer(allowed_roots=[tmp_path])
    target_dir = tmp_path / "factors" / "test_f"
    with pytest.raises(Exception, match="max_rows"):
        delete_rows_from_dataset(
            ds=staging_dataset,
            authorizer=authorizer,
            target_dir=target_dir,
            time_column="datetime",
            max_rows=0,
            params={"factor_id": "test_f"},
        )
