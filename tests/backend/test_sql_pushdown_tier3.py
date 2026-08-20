# -*- coding: utf-8
"""DuckDB SQL 下推 Tier3：比较 / 逻辑 / ts_cov / cum / group_std 等。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy SQL rollout tier superseded by canonical evidence certification")

from api.dsl_parser import parse_expr
from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql, plan_is_sql_capable
from planner.logical_plan import PlanNode
from runtime.engine import FactorEngine
from storage.factory import build_data_source


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _lit(val: float) -> PlanNode:
    return PlanNode(op="literal", attrs={"value": val})


@pytest.mark.parametrize(
    "op,sym",
    [
        ("gt", ">"),
        ("lt", "<"),
        ("eq", "="),
        ("ge", ">="),
        ("le", "<="),
        ("ne", "<>"),
    ],
)
def test_compare_ops_emit_sql(op, sym):
    plan = PlanNode(op=op, inputs=[_col("close"), _col("open")])
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(plan, dataset="d", time_column="t", instrument_column="i")
    assert sql is not None
    assert sym in sql.query


def test_power_and_logic_emit_sql():
    plan = PlanNode(op="power", inputs=[_col("close"), _lit(2.0)])
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(plan, dataset="d", time_column="t", instrument_column="i")
    assert sql is not None
    assert "POW" in sql.query

    and_plan = PlanNode(op="and_", inputs=[_col("a"), _col("b")])
    assert plan_is_sql_capable(and_plan)
    not_plan = PlanNode(op="not_", inputs=[_col("flag")])
    assert plan_is_sql_capable(not_plan)


def test_tier3_ts_ops_emit_sql():
    for op, frag in [
        ("ts_cov", "covar_samp"),
        ("ts_quantile", "quantile_cont"),
        ("ts_product", "EXP(SUM"),
        ("ts_skew", "skewness"),
        ("cum_sum", "UNBOUNDED PRECEDING"),
        ("group_std", "STDDEV_SAMP"),
    ]:
        if op == "ts_cov":
            plan = PlanNode(op=op, inputs=[_col("a"), _col("b")], attrs={"window": 5})
        elif op == "ts_quantile":
            plan = PlanNode(op=op, inputs=[_col("close")], attrs={"window": 5, "q": 0.5})
        elif op == "group_std":
            plan = PlanNode(op=op, inputs=[_col("close"), _col("ind")])
        else:
            plan = PlanNode(op=op, inputs=[_col("close")], attrs={"window": 5})
        assert plan_is_sql_capable(plan), op
        sql = compile_plan_to_sql(plan, dataset="d", time_column="t", instrument_column="i")
        assert sql is not None, op
        assert frag in sql.query, op


def test_ts_regression_slope_emit_sql():
    plan = PlanNode(
        op="ts_regression",
        inputs=[_col("y"), _col("x")],
        attrs={"window": 10, "retval": "slope"},
    )
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(plan, dataset="d", time_column="t", instrument_column="i")
    assert sql is not None
    assert "covar_samp" in sql.query


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
    Ret: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_data(root: Path) -> None:
    root.mkdir(parents=True)
    rows = []
    for d in pd.date_range("2024-01-02", periods=8, freq="D"):
        for sym, base in [("A", 10.0), ("B", 20.0)]:
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Close": base + d.day * 0.1,
                    "Open": base + d.day * 0.1 - 0.2,
                    "Ret": 0.01 * d.day,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def test_duckdb_tier3_ts_cov_and_cum_sum_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "fields": {"close": "Close", "open": "Open", "ret": "Ret"},
        }
    )
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)

    f_cov = Factor("cov", expr=parse_expr("ts_cov(col('close'), col('open'), 5)"))
    f_cum = Factor("cum", expr=parse_expr("cum_sum(col('close'))"))

    for factor in (f_cov, f_cum):
        pd_out = eng_pd.run(factor)["result"]
        sql_out = eng_sql.run(factor)["result"]
        aligned = pd_out.align(sql_out, join="inner")
        pd_cmp = aligned[0].dropna()
        sql_cmp = aligned[1].dropna()
        if len(pd_cmp) == 0:
            continue
        diff = (pd_cmp - sql_cmp).abs()
        assert diff.max() < 1e-5, f"{factor.name} max diff {diff.max()}"


def test_duckdb_tier3_power_gt_match_pandas(tmp_path, monkeypatch):
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

    f = Factor("pw", expr=parse_expr("gt(power(col('close'), 2), col('open'))"))
    pd_out = eng_pd.run(f)["result"]
    sql_out = eng_sql.run(f)["result"]
    aligned = pd_out.align(sql_out, join="inner")
    pd_cmp = aligned[0].dropna()
    sql_cmp = aligned[1].dropna()
    assert (pd_cmp - sql_cmp).abs().max() < 1e-5


def test_clickhouse_dialect_tier3_skew():
    plan = PlanNode(op="ts_skew", inputs=[_col("close")], attrs={"window": 5})
    sql = compile_plan_to_sql(
        plan,
        table="panel",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert sql is not None
    assert "skewSamp" in sql.query
