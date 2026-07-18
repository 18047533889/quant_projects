# -*- coding: utf-8 -*-
"""compute_and_write：读 → 运算 → 写别处（不改源）。"""
from __future__ import annotations

from textwrap import dedent

import pandas as pd
import pyarrow as pa
import pytest

from data_access import reset_store
from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import ValidationError
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture
def compute_store(tmp_path, monkeypatch):
    src = tmp_path / "src"
    stg = tmp_path / "stg"
    src.mkdir()
    stg.mkdir()
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Symbol": ["000001.SZ", "000001.SZ"],
            "Close": [10.0, 11.0],
        }
    ).to_parquet(src / "2024-01-02.parquet")
    # second file also needed for range; reuse same rows ok for unit test
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-03"]),
            "Symbol": ["000002.SZ"],
            "Close": [20.0],
        }
    ).to_parquet(src / "2024-01-03.parquet")

    yaml_text = dedent(
        f"""
        prices:
          kind: static
          access_mode: published
          layout: plain
          root: {src}
          glob: "**/*.parquet"
          time_column: TradeDate
          instrument_column: Symbol
          hive_partitioning: false
          union_by_name: true
          schema:
            TradeDate: timestamp
            Symbol: string
            Close: double

        results_stg:
          kind: parametric
          access_mode: staging
          layout: hive
          root_template: {stg}/factors/{{factor_id}}
          glob_template: "year=*/*.parquet"
          partition_columns: [year]
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true
        """
    )
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("QUANT_DATASETS_YAML", str(cfg))
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "test_ns")
    reset_store()
    store = DataAccessStore(load_registry(cfg), DuckDBEngine())
    yield store
    reset_store()


def test_compute_and_write_to_staging(compute_store):
    result = compute_store.compute_and_write(
        """
        SELECT TradeDate AS datetime, Symbol AS asset, Close AS value,
               CAST(strftime(TradeDate, '%Y') AS INTEGER) AS year
        FROM {{prices}}
        """,
        read_datasets=["prices"],
        read_time_ranges={"prices": ("2024-01-02", "2024-01-03")},
        write_dataset="results_stg",
        factor_id="close_copy_v1",
        mode="overwrite",
        partition_by=["year"],
    )
    assert result["rows_computed"] >= 1
    assert "prices" in result["source_datasets"]

    back = compute_store.read_frame(
        "results_stg",
        factor_id="close_copy_v1",
        columns=["datetime", "asset", "value"],
    )
    assert len(back) >= 1
    assert set(back["asset"]) <= {"000001.SZ", "000002.SZ"}


def test_compute_and_write_rejects_published(compute_store):
    with pytest.raises(ValidationError, match="published"):
        compute_store.compute_and_write(
            "SELECT * FROM {{prices}}",
            read_datasets=["prices"],
            write_dataset="prices",
        )


def test_sql_uses_prepare_with_time_range(compute_store, monkeypatch):
    """sql() 应把 time_range 传给 prepare（此处走本地，结果非空）。"""
    tbl = compute_store.sql(
        "SELECT COUNT(*) AS n FROM {{prices}}",
        read_datasets=["prices"],
        read_time_ranges={"prices": ("2024-01-02", "2024-01-03")},
    )
    assert tbl.num_rows == 1
    assert tbl["n"][0].as_py() >= 1
