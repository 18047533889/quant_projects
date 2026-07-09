# -*- coding: utf-8
"""DuckDB SQL 下推 Tier7：二元最值 / cum·expanding / 截面广播 / 符号数学 parity。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.dsl_parser import parse_expr
from api.factor import Factor
from backend.factory import build_backend
from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql, plan_is_sql_capable
from backend.sql_pushdown.plan_fixtures import minimal_plan
from backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS, register_sql_backends
from runtime.engine import FactorEngine
from storage.factory import build_data_source


_TIER7_OPS = frozenset(
    {
        "maximum",
        "minimum",
        "cum_prod",
        "cum_delta",
        "expanding_mean",
        "expanding_sum",
        "log_abs",
        "signed_log",
        "signed_sqrt",
        "c_mean",
        "c_std",
        "c_sum",
        "c_count",
    }
)


@pytest.fixture(scope="module", autouse=True)
def _load():
    from cleaned_operators import load_all

    load_all()
    register_sql_backends()


def test_tier7_in_sql_registry():
    assert _TIER7_OPS <= SQL_CAPABLE_CANONICALS


@pytest.mark.parametrize("op,frag", sorted([
    ("maximum", "GREATEST"),
    ("minimum", "LEAST"),
    ("cum_prod", "product"),
    ("cum_delta", "FIRST_VALUE"),
    ("expanding_mean", "AVG"),
    ("expanding_sum", "UNBOUNDED PRECEDING"),
    ("log_abs", "abs"),
    ("signed_log", "sign"),
    ("signed_sqrt", "sqrt"),
    ("c_mean", "PARTITION BY ts"),
    ("c_std", "STDDEV_SAMP"),
    ("c_sum", "SUM"),
    ("c_count", "COUNT"),
]))
def test_tier7_ops_emit_sql(op: str, frag: str):
    plan = minimal_plan(op)
    assert plan_is_sql_capable(plan), op
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None, op
    assert frag in compiled.query, op


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
    Volume: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_data(root: Path) -> None:
    root.mkdir(parents=True)
    rows = []
    for d in pd.date_range("2024-01-02", periods=8, freq="D"):
        for sym, base in [("A", 10.0), ("B", 20.0), ("C", 30.0)]:
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Close": base + d.day * 0.1,
                    "Open": base + d.day * 0.1 - 0.5,
                    "Volume": 100.0 + d.day * 10 + (1 if sym == "A" else 2),
                }
            )
    df = pd.DataFrame(rows)
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[2]), "Close"] = float("nan")
    df.to_parquet(root / "panel.parquet", index=False)


def test_duckdb_tier7_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-09",
        }
    )
    maximum = make_cleaned_call_factory("maximum")
    minimum = make_cleaned_call_factory("minimum")
    cum_prod = make_cleaned_call_factory("cum_prod")
    cum_delta = make_cleaned_call_factory("cum_delta")
    expanding_mean = make_cleaned_call_factory("expanding_mean")
    log_abs = make_cleaned_call_factory("log_abs")
    signed_log = make_cleaned_call_factory("signed_log")
    c_mean = make_cleaned_call_factory("c_mean")

    for expr, name in (
        (maximum(col("Close"), col("Open")), "maximum"),
        (minimum(col("Close"), col("Open")), "minimum"),
        (cum_prod(parse_expr("1 + ts_pct(col('Close'), 1)")), "cum_prod"),
        (cum_delta(col("Close")), "cum_delta"),
        (expanding_mean(col("Close")), "expanding_mean"),
        (log_abs(col("Close")), "log_abs"),
        (signed_log(col("Close")), "signed_log"),
        (c_mean(col("Close")), "c_mean"),
    ):
        factor = Factor(name=name, expr=expr)
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
        a = eng_pd.run(factor)["result"]
        b = eng_sql.run(factor)["result"]
        pd.testing.assert_series_equal(
            a, b, check_names=False, rtol=1e-5, atol=1e-5
        )


def test_tier7_clickhouse_dialect():
    for op in sorted(_TIER7_OPS):
        plan = minimal_plan(op)
        compiled = compile_plan_to_sql(
            plan,
            table="panel",
            time_column="ts",
            instrument_column="inst",
            dialect=SqlDialect.CLICKHOUSE,
        )
        assert compiled is not None, op
        q = compiled.query.strip()
        assert q, op
        if op in {"maximum", "minimum"}:
            assert "greatest" in q or "least" in q
            assert "isNull" in q or "IS NULL" in q
        elif op in {"log_abs", "signed_log", "signed_sqrt"}:
            assert "log(" in q or "sqrt(" in q
        else:
            assert "OVER (" in q or "PARTITION BY" in q, op
