# -*- coding: utf-8
"""部分 SQL 子树下推 + Python fallback 集成测试。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_engine.api import rank, ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.planner.sql_lowerer import lower_to_physical_plan
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source


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


def _registry(tmp_path: Path, root: Path) -> Path:
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


def _seed(root: Path) -> None:
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


def test_sql_lowerer_splits_macd_mixed_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_registry(tmp_path, tmp_path / "data")))
    _seed(tmp_path / "data")

    from factor_engine.api.cleaned_ops import make_cleaned_call_factory

    where = make_cleaned_call_factory("where")
    macd = make_cleaned_call_factory("MACD")
    expr = where(macd(col("Close")), ts_mean(col("Close"), 2), col("Open"))
    factor = Factor(name="mixed", expr=expr)
    engine = FactorEngine(backend=build_backend("pandas"), data_source=build_data_source({"type": "data_access", "dataset": "test_daily"}))
    plan, _ = engine.compile(factor)

    physical = lower_to_physical_plan(plan)
    assert not physical.fully_sql
    assert len(physical.sql_subtrees) >= 1
    assert physical.root.op != "materialized_series" or physical.root.op == plan.op


def test_partial_sql_hybrid_matches_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_registry(tmp_path, tmp_path / "data")))
    _seed(tmp_path / "data")

    from factor_engine.api.cleaned_ops import make_cleaned_call_factory

    where = make_cleaned_call_factory("where")
    macd = make_cleaned_call_factory("MACD")
    expr = where(macd(col("Close")), ts_mean(col("Close"), 2), col("Open"))
    factor = Factor(name="mixed", expr=expr)

    source = build_data_source({"type": "data_access", "dataset": "test_daily", "long_table": True})
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_h = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)

    a = eng_pd.run(factor)["result"]
    b = eng_h.run(factor)["result"]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


def test_rank_ts_mean_still_fully_sql(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_registry(tmp_path, tmp_path / "data")))
    _seed(tmp_path / "data")

    source = build_data_source({"type": "data_access", "dataset": "test_daily"})
    factor = Factor(name="t", expr=rank(ts_mean(col("Close"), 2)))
    engine = FactorEngine(backend=build_backend("auto"), data_source=source)
    plan, _ = engine.compile(factor)
    physical = lower_to_physical_plan(plan)
    assert physical.fully_sql
