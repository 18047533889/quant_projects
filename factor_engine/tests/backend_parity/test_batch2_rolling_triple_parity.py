# -*- coding: utf-8
"""Batch-2 rolling primitive 三后端 parity：ts_argmax / ts_argmin。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource

BATCH2_ROLLING_CASES = [
    ("ts_argmax", lambda: make_cleaned_call_factory("ts_argmax")(col("close"), 3)),
    ("ts_argmin", lambda: make_cleaned_call_factory("ts_argmin")(col("close"), 3)),
]

# NaN / tie 窗口 edge（与 BATCH2_ROLLING_CASES 同表达式，数据源含 NaN）
BATCH2_EDGE_CASES = BATCH2_ROLLING_CASES


@pytest.fixture(scope="module")
def mem_source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
            (pd.Timestamp("2024-01-06"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
            (pd.Timestamp("2024-01-05"), "B"),
            (pd.Timestamp("2024-01-06"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series(
        [10.0, 13.0, 11.0, 15.0, 12.0, 20.0, 18.0, 22.0, 19.0, 21.0],
        index=idx,
    )
    return InMemorySeriesSource(data={"close": close})


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
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
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for (ts, sym), close in mem.data["close"].items():
        rows.append(
            {
                "TradeDate": ts.date(),
                "Symbol": sym,
                "Close": float(close) if pd.notna(close) else None,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch, mem_source):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data")))
    _seed_duckdb(tmp_path / "data", mem_source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def _col(name: str):
    return col({"close": "Close"}.get(name, name))


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


@pytest.mark.parametrize("name,expr_builder", BATCH2_ROLLING_CASES)
def test_batch2_polars_long_matches_pandas(mem_source, name, expr_builder):
    from backend.polars_long_policy import infer_polars_long_tier

    assert infer_polars_long_tier(name) == "python_rolling"
    expr = expr_builder()
    pd_out = _series(_run(mem_source, expr, "pandas"))
    pl_run = _run(mem_source, expr, "polars_long")
    assert pl_run.get("used_polars_long_path") is True
    assert not pl_run.get("polars_long_fallback_reason")
    long_out = _series(pl_run)
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name,expr_builder", BATCH2_ROLLING_CASES)
def test_batch2_duckdb_matches_pandas(mem_source, duckdb_source, name, expr_builder):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    mem_expr = expr_builder()
    pd_out = _series(_run(mem_source, mem_expr, "pandas"))
    duck_expr = make_cleaned_call_factory(name)(_col("close"), 3)
    sql_run = _run(duckdb_source, duck_expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _series(sql_run)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name,expr_builder", BATCH2_ROLLING_CASES)
def test_batch2_duckdb_matches_polars_long(mem_source, duckdb_source, name, expr_builder):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    long_out = _series(_run(mem_source, expr_builder(), "polars_long"))
    sql_run = _run(duckdb_source, make_cleaned_call_factory(name)(_col("close"), 3), "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _series(sql_run)
    pd.testing.assert_series_equal(long_out, sql_out, check_names=False, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name,expr_builder", BATCH2_EDGE_CASES)
def test_batch2_edge_polars_long_matches_pandas(mem_source, name, expr_builder):
    expr = expr_builder()
    pd_out = _series(_run(mem_source, expr, "pandas"))
    long_out = _series(_run(mem_source, expr, "polars_long"))
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name,expr_builder", BATCH2_EDGE_CASES)
def test_batch2_edge_duckdb_matches_pandas(mem_source, duckdb_source, name, expr_builder):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    pd_out = _series(_run(mem_source, expr_builder(), "pandas"))
    sql_run = _run(duckdb_source, make_cleaned_call_factory(name)(_col("close"), 3), "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _series(sql_run)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-5, atol=1e-5)
