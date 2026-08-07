"""read/factors.py + store.read_factors / FactorCatalog 单测。"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.read.factors import FactorMeta, build_factor_pivot_sql, build_factor_union_sql
from data_access.store import DataAccessStore


def _make_store(tmp_path):
    lake = tmp_path / "lake"
    for fid in ["mom_3d", "vol_20"]:
        d = lake / "factors" / fid / "year=2024"
        d.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "datetime": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
                "asset": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "value": [1.1, 1.2, 2.1, 2.2],
            }),
            str(d / "data.parquet"),
        )
    (lake / "factors" / "mom_3d" / "_factor_meta.json").write_text(
        json.dumps({"factor_version": "v1", "frequency": "daily", "status": "active"})
    )
    (tmp_path / "datasets.yaml").write_text(
        f"""
factor_lake:
  kind: parametric
  access_mode: published
  layout: hive
  root_template: {lake}/factors/{{factor_id}}
  glob_template: "year=*/data.parquet"
  partition_columns: [year]
  params_schema:
    factor_id: str
  time_column: datetime
  instrument_column: asset
  hive_partitioning: true
  union_by_name: true
  schema:
    datetime: timestamp
    asset: string
    value: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(registry=load_registry(tmp_path / "datasets.yaml"),
                           engine=DuckDBEngine(threads=2, enable_object_cache=False))


def test_build_union_sql():
    sql, params = build_factor_union_sql([("f1", ["/p1.parquet"]), ("f2", ["/p2.parquet"])])
    assert "UNION ALL" in sql
    assert params[0] == "f1"
    # 单路径绑定为字符串（多路径才是 list）
    assert params[1] == "/p1.parquet"
    assert params[3] == "/p2.parquet"


def test_build_pivot_sql():
    union, up = build_factor_union_sql([("f1", ["/p1.parquet"]), ("f2", ["/p2.parquet"])])
    sql, params = build_factor_pivot_sql(union, up, factor_ids=["f1", "f2"])
    assert "PIVOT" in sql
    assert '"f1"' in sql and '"f2"' in sql


def test_read_factors_long(tmp_path):
    store = _make_store(tmp_path)
    handle = store.read_factors(["mom_3d", "vol_20"], columns=["datetime", "asset", "value"])
    table = handle.to_arrow()
    assert table.num_rows == 8
    assert set(table.column("factor_id").to_pylist()) == {"mom_3d", "vol_20"}


def test_read_factors_wide(tmp_path):
    store = _make_store(tmp_path)
    handle = store.read_factors(["mom_3d", "vol_20"], layout="wide",
                                columns=["datetime", "asset", "value"])
    table = handle.to_arrow()
    assert "mom_3d" in table.column_names
    assert "vol_20" in table.column_names
    assert table.num_rows == 4


def test_read_factors_time_range(tmp_path):
    store = _make_store(tmp_path)
    handle = store.read_factors(["mom_3d"], time_range=("2024-01-03", "2024-01-03"),
                                columns=["datetime", "asset", "value"])
    assert handle.to_arrow().num_rows == 2


def test_factor_catalog(tmp_path):
    store = _make_store(tmp_path)
    catalog = store.get_factor_catalog()
    assert "mom_3d" in catalog
    assert catalog.get("mom_3d").status == "active"
    refreshed = store.refresh_factor_catalog()
    assert refreshed["factors"] == 1


def test_factor_meta_roundtrip():
    meta = FactorMeta(factor_id="f1", factor_version="v2", frequency="daily", status="active")
    restored = FactorMeta.from_dict(meta.to_dict())
    assert restored.factor_version == "v2"
