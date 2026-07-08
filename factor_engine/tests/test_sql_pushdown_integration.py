# -*- coding: utf-8
"""SQL 下推端到端：DuckDB 内算 vs Pandas 对齐。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from runtime.engine import FactorEngine
from storage.factory import build_data_source


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.delenv("DATA_ACCESS_CONFIG", raising=False)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    yield
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass


def _write_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
test_daily:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {root}
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
    Open: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_data(root: Path) -> None:
    root.mkdir(parents=True)
    rows = []
    for d in pd.date_range("2024-01-02", periods=5, freq="D"):
        for sym, base in [("A", 10.0), ("B", 20.0)]:
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Close": base + d.day,
                    "Open": base + d.day - 0.5,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def test_duckdb_sql_pushdown_matches_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    expr = rank(ts_mean(col("Close"), 2))
    factor = Factor(name="t", expr=expr)

    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)

    a = eng_pd.run(factor)["result"]
    b = eng_sql.run(factor)["result"]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)


def test_hybrid_backend_matches_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")

    source = build_data_source({"type": "data_access", "dataset": "test_daily"})
    expr = rank(ts_mean(col("Close"), 2))
    factor = Factor(name="t", expr=expr)

    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_h = FactorEngine(backend=build_backend("auto"), data_source=source)

    pd.testing.assert_series_equal(
        eng_pd.run(factor)["result"],
        eng_h.run(factor)["result"],
        check_names=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_ts_corr_pushdown_matches_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    root.mkdir(parents=True)
    rows = []
    for d in pd.date_range("2024-01-02", periods=6, freq="D"):
        for sym in ["A", "B"]:
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Close": float(d.day + (1 if sym == "A" else 2)),
                    "Open": float(d.day),
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")

    from api import ts_corr
    from api.columns import col

    source = build_data_source({"type": "data_access", "dataset": "test_daily"})
    expr = ts_corr(col("Close"), col("Open"), 3)
    factor = Factor(name="t", expr=expr)

    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)

    a = eng_pd.run(factor)["result"]
    b = eng_sql.run(factor)["result"]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


def test_clickhouse_dialect_emitter():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql
    from planner.logical_plan import PlanNode

    plan = PlanNode(op="ts_mean", inputs=[PlanNode(op="column", attrs={"name": "close"})], attrs={"d": 3})
    compiled = compile_plan_to_sql(
        plan,
        table="panel_daily",
        time_column="trade_date",
        instrument_column="ticker",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert compiled is not None
    assert "panel_daily" in compiled.query

    from backend.sql_pushdown.emitter import (
        SqlDialect,
        SqlPushdownFilter,
        compile_plan_to_sql,
    )
    from planner.logical_plan import PlanNode

    plan = PlanNode(op="column", attrs={"name": "Close"})
    compiled = compile_plan_to_sql(
        plan,
        dataset="test_daily",
        time_column="TradeDate",
        instrument_column="Symbol",
        filt=SqlPushdownFilter(
            time_column="TradeDate",
            start="2024-01-01",
            end="2024-12-31",
            instrument_column="Symbol",
            instruments=("A", "B"),
        ),
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert '"TradeDate" >=' in compiled.query
    assert '"Symbol" IN' in compiled.query
