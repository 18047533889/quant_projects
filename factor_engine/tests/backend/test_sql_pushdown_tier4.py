# -*- coding: utf-8
"""DuckDB SQL 下推 Tier4：WMA / ewm / Slope / argext / floor / count。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from api.dsl_parser import parse_expr
from api.factor import Factor
from backend.factory import build_backend
from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS
from planner.logical_plan import PlanNode
from runtime.engine import FactorEngine
from storage.factory import build_data_source


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def test_tier4_ops_in_registry():
    for op in (
        "WMA",
        "ewm_std",
        "ewm_var",
        "cum_std",
        "floor",
        "ceil",
        "inverse",
        "count",
        "Slope",
        "ts_argmax",
        "ts_argmin",
    ):
        assert op in SQL_CAPABLE_CANONICALS


@pytest.mark.parametrize(
    "op,frag",
    [
        ("WMA", "LAG(_v"),
        ("ewm_std", "POW("),
        ("ewm_var", "POW("),
        ("cum_std", "STDDEV"),
        ("expanding_std", "STDDEV"),
        ("floor", "floor("),
        ("inverse", "1.0 /"),
        ("count", "SUM(CASE WHEN _v IS NOT NULL"),
        ("Slope", "LAG(_v"),
        ("ts_argmax", "w_ext"),
        ("ts_argmin", "w_ext"),
    ],
)
def test_tier4_emit_sql(op, frag):
    if op in {"WMA", "ewm_std", "ewm_var", "Slope", "ts_argmax", "ts_argmin"}:
        plan = PlanNode(op=op, inputs=[_col("close")], attrs={"window": 5, "d": 5, "span": 5})
    else:
        plan = PlanNode(op=op, inputs=[_col("close")])
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(plan, dataset="d", time_column="t", instrument_column="i")
    assert sql is not None
    assert frag in sql.query


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
    for d in pd.date_range("2024-01-02", periods=10, freq="D"):
        for sym, base in [("A", 10.0), ("B", 20.0)]:
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Close": base + d.day * 0.05,
                    "Open": base + d.day * 0.05 - 0.1,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def test_duckdb_tier4_wma_floor_count_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "fields": {"close": "Close", "open": "Open"},
        }
    )
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)

    cases = [
        ("wma", parse_expr("WMA(col('close'), 5)")),
        ("fl", parse_expr("floor(col('close'))")),
        ("cnt", parse_expr("count(col('close'))")),
        ("slope", parse_expr("Slope(col('close'), 5)")),
    ]
    for name, expr in cases:
        f = Factor(name, expr=expr)
        pd_out = eng_pd.run(f)["result"]
        sql_out = eng_sql.run(f)["result"]
        aligned = pd_out.align(sql_out, join="inner")
        pd_cmp = aligned[0].dropna()
        sql_cmp = aligned[1].dropna()
        if len(pd_cmp) == 0:
            continue
        rtol = 1e-4 if name in {"wma", "slope"} else 1e-5
        assert (pd_cmp - sql_cmp).abs().max() < rtol, name


def test_duckdb_tier4_ts_argmax_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "fields": {"close": "Close"},
        }
    )
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
    f = Factor("arg", expr=parse_expr("ts_argmax(col('close'), 5)"))
    pd_out = eng_pd.run(f)["result"]
    sql_out = eng_sql.run(f)["result"]
    aligned = pd_out.align(sql_out, join="inner")
    pd.testing.assert_series_equal(
        aligned[0].dropna(),
        aligned[1].dropna(),
        check_names=False,
        rtol=1e-10,
        atol=1e-10,
    )
