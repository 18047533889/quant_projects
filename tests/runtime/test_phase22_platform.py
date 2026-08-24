"""Phase 22：增量调度 / bucket 剪枝 / stats sidecar。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.api import ts_mean
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.incremental_scheduler import DataEvent, plan_updates_from_data_event
from factor_engine.runtime.materialize_service import execute_materialize
from factor_engine.storage.catalog import FactorCatalog
from factor_engine.storage.materializer import ParquetMaterializer
from tests.helpers import InMemorySeriesSource


def test_dependency_catalog_and_event_planner(tmp_path):
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    catalog.record_factor_dependency(
        "alpha_001",
        referenced_columns=["close", "volume"],
        lookback=20,
        frequency="1d",
        source_dataset="us_stocks",
    )
    catalog.record_factor_dependency(
        "alpha_002",
        referenced_columns=["open"],
        lookback=5,
        frequency="1d",
        source_dataset="us_stocks",
    )

    event = DataEvent(
        dataset="us_stocks",
        column="close",
        updated_date="2026-07-09",
    )
    plans = plan_updates_from_data_event(catalog, event, lookback_extra=2)
    assert len(plans) == 1
    assert plans[0].factor_id == "alpha_001"
    assert plans[0].lookback_bars >= 20


def test_execute_materialize_records_dependency(tmp_path, monkeypatch):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    series = pd.Series([1.0, 2.0], index=idx)
    eng = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": series}),
    )
    factor = Factor(name="m", expr=ts_mean(col("close"), 2))
    output = eng.run(factor)

    class _DS:
        dataset = "mock_ds"

    eng.data_source = _DS()

    monkeypatch.setattr(
        "factor_engine.runtime.materialize_service.dual_write_service.dual_write_clickhouse",
        lambda *args, **kwargs: args[1],
    )
    execute_materialize(
        eng,
        factor,
        output,
        target="local",
        lake_root=tmp_path / "lake",
        staging_dataset="factor_lake_staging",
        factor_id="alpha_dep",
        author="test",
        frequency="1d",
        description=None,
        expression=None,
        dq_check=False,
        dq_strict=True,
        dq_thresholds=None,
        write_metadata=False,
        data_source_config={"type": "mock", "dataset": "mock_ds"},
        resume_materialize=False,
        isolate_partition_failures=True,
        preserve_invalid_rows=False,
        value_dtype="float32",
        clickhouse_table=None,
        ch_ensure_table=True,
        ch_host=None,
        ch_port=None,
        ch_database=None,
        ch_username=None,
        ch_password=None,
        ch_secure=None,
    )
    catalog = ParquetMaterializer(lake_root=tmp_path / "lake").catalog
    dep = catalog.get_factor_dependency("alpha_dep")
    assert dep is not None
    assert "close" in dep["referenced_columns"]


def test_engine_plan_incremental_from_event(tmp_path):
    lake = tmp_path / "lake"
    lake.mkdir()
    mat = ParquetMaterializer(lake_root=lake)
    mat.catalog.record_factor_dependency(
        "f1",
        referenced_columns=["close"],
        lookback=10,
        source_dataset="ds1",
    )

    eng = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(
            data={
                "close": pd.Series(
                    [1.0],
                    index=pd.MultiIndex.from_tuples(
                        [(pd.Timestamp("2024-01-01"), "A")],
                        names=["timestamp", "instrument"],
                    ),
                )
            }
        ),
    )
    out = eng.plan_incremental_from_event(
        {"dataset": "ds1", "column": "close", "updated_date": "2026-07-01"},
        lake_root=lake,
    )
    assert out["factor_count"] == 1
    assert out["plans"][0]["factor_id"] == "f1"
