"""RelationHandle 治理 + ScanCost 校准（#28/#30/#31/#27）。"""
from __future__ import annotations

import pytest

from data_access import ValidationError
from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.read.relation_handle import RelationHandle
from data_access.read.scan_cost import (
    _calibrated_factor,
    record_scan_actual,
    reset_calibration,
)
from data_access.store import DataAccessStore

import pyarrow as pa
import pyarrow.parquet as pq


@pytest.fixture()
def store(tmp_path):
    (tmp_path / "d.parquet").write_bytes(b"")
    pq.write_table(
        pa.table(
            {
                "ts": ["2024-01-01", "2024-01-02"],
                "asset": ["A", "A"],
                "value": [1.0, 2.0],
            }
        ),
        str(tmp_path / "d.parquet"),
    )
    (tmp_path / "datasets.yaml").write_text(
        f"""
ds:
  kind: static
  access_mode: published
  layout: plain
  format: parquet
  root: {tmp_path}
  glob: "*.parquet"
  time_column: ts
  instrument_column: asset
  schema:
    ts: string
    asset: string
    value: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(
        registry=load_registry(tmp_path / "datasets.yaml"),
        engine=DuckDBEngine(threads=2, enable_object_cache=False),
    )


def test_relation_handle_safe_inspection_no_fetch(store):
    """#31：schema/columns/types 只读检查，不触发 collect。"""
    rh = store.sql_relation(
        "SELECT ts, asset, value FROM read_parquet(?)",
        params=[str(store._registry.get("ds").root / "*.parquet")],
    )
    assert set(rh.columns()) >= {"ts", "asset", "value"}
    assert len(rh.types()) == 3
    assert "VARCHAR" in " ".join(rh.types())


def test_relation_handle_production_ban_fetch(store, monkeypatch):
    """#31：production 禁公开活 relation（逃生口），只留只读检查 API。"""
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    rh = store.sql_relation(
        "SELECT ts, asset, value FROM read_parquet(?)",
        params=[str(store._registry.get("ds").root / "*.parquet")],
    )
    with pytest.raises(ValidationError):
        _ = rh.relation
    # 只读检查仍可用
    assert len(rh.columns()) == 3


def test_relation_handle_sql_fragment_param_order(store):
    """#30：sql_fragment 显式子查询参数顺序——子查询参数在前，外层参数在后。"""
    root = str(store._registry.get("ds").root / "*.parquet")
    rh = store.sql_relation("SELECT * FROM read_parquet(?)", params=[root])
    # 外层 SELECT 有自己参数（WHERE value > ?）；{sub} 是子查询占位
    rh2 = rh.sql_fragment("SELECT * FROM {sub} WHERE value > ?", params=[1.0])
    tbl = rh2.collect()
    assert tbl.num_rows == 1
    assert tbl.column("value").to_pylist() == [2.0]


def test_read_handle_lazy_rows_no_collect(store):
    """#28：lazy 形态 rows 不触发 collect（避免 repr 意外物化）。"""
    lf = store.sql_relation(
        "SELECT ts, asset, value FROM read_parquet(?)",
        params=[str(store._registry.get("ds").root / "*.parquet")],
    )
    from data_access.read.read_handle import ReadHandle

    h = ReadHandle(lazy=lf.to_polars_lazy())
    assert h.rows is None
    # repr 不物化
    assert "kind=lazy" in repr(h)


def test_scan_cost_calibration():
    """#27：estimate-vs-actual 在线校准 EMA 更新。"""
    reset_calibration()
    assert _calibrated_factor("ds_x") == 1.0
    record_scan_actual("ds_x", estimated_score=1e6, actual_elapsed_ms=2000.0)
    factor = _calibrated_factor("ds_x")
    assert factor > 1.0  # 实际 > 估算 → 乘子上调
    reset_calibration()
    assert _calibrated_factor("ds_x") == 1.0
