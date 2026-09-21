"""Real registry/runtime/engine reads must prune daily exposure snapshots."""
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.cos_runtime import install_cos_runtime
from data_access.read.query_budget import QueryBudget
from data_access.registry.loader import load_registry
from data_access.store import DataAccessStore


@pytest.fixture
def exposure_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ASHARE_PARQUET_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("QUANT_SCHEMA_CHECK", "strict")
    registry = load_registry(Path(__file__).resolve().parents[3] / "data_access/config/datasets.yaml")
    for name in ("ashare_stock_industry", "ashare_stock_valuation_daily"):
        ds = registry.get(name)
        ds.root.mkdir(parents=True)
        for day in (1, 2, 3):
            row = {}
            for col, kind in ds.schema.items():
                if col == "TradeDate":
                    value = date(2024, 8, day)
                elif col == "UpdateTime":
                    value = datetime(2024, 8, day, 8, tzinfo=timezone.utc)
                elif kind == "string":
                    value = "sw_l1" if col == "IndustrySource" else "000001.SZ"
                else:
                    value = float(day)
                row[col] = [value]
            pq.write_table(pa.table(row), ds.root / f"2024-08-{day:02}.parquet")
    engine = DuckDBEngine(threads=1)
    store = install_cos_runtime(DataAccessStore(registry, engine))
    yield store
    engine.close()


@pytest.mark.parametrize("name", ["ashare_stock_industry", "ashare_stock_valuation_daily"])
def test_one_day_exposure_read_fits_one_file_budget(exposure_store, name):
    result = exposure_store.read(name, columns=["TradeDate", "Symbol", "UpdateTime"],
        time_range=("2024-08-02", "2024-08-02"),
        filters={"IndustrySource": "sw_l1"} if name.endswith("industry") else None,
        query_budget=QueryBudget(max_scan_files=1, max_rows=10),
        result="arrow").to_arrow()
    assert result.num_rows == 1
    assert result["TradeDate"].to_pylist() == [date(2024, 8, 2)]
    assert result["UpdateTime"].to_pylist() == [datetime(2024, 8, 2, 8, tzinfo=timezone.utc)]


def test_cos_industry_override_preserves_strict_aware_time(exposure_store):
    ds = exposure_store.get_dataset("ashare_stock_industry")
    result = exposure_store.read(ds.name, columns=["UpdateTime"],
        physical_scope=str(ds.root / "2024-08-02.parquet"),
        filters={"IndustrySource": "sw_l1"}, result="arrow",
        query_budget=QueryBudget(max_scan_files=1, max_rows=10)).to_arrow()
    assert result["UpdateTime"].to_pylist() == [datetime(2024, 8, 2, 8, tzinfo=timezone.utc)]
