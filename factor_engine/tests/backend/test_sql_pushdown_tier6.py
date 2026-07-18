# -*- coding: utf-8
"""DuckDB SQL 下推 Tier6：价量 / 截面百分位 parity。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy SQL rollout tier superseded by canonical evidence certification")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql, plan_is_sql_capable
from backend.sql_pushdown.plan_fixtures import minimal_plan
from backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS, register_sql_backends
from runtime.engine import FactorEngine
from storage.factory import build_data_source


_TIER6_OPS = frozenset(
    {
        "rank_pct",
        "cs_pct_rank",
        "cs_quantile",
        "c_percentile",
        "log_returns",
        "volatility",
        "vwap",
    }
)


@pytest.fixture(scope="module", autouse=True)
def _load():
    from cleaned_operators import load_all

    load_all()
    register_sql_backends()


def test_tier6_in_sql_registry():
    assert _TIER6_OPS <= SQL_CAPABLE_CANONICALS


@pytest.mark.parametrize("op", sorted(_TIER6_OPS))
def test_tier6_ops_emit_sql(op: str):
    plan = minimal_plan(op)
    assert plan_is_sql_capable(plan), op
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None, op
    assert compiled.query.strip(), op


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


def test_duckdb_tier6_cs_and_price_volume_match_pandas(tmp_path, monkeypatch):
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
    rank_pct = make_cleaned_call_factory("rank_pct")
    cs_pct_rank = make_cleaned_call_factory("cs_pct_rank")
    cs_quantile = make_cleaned_call_factory("cs_quantile")
    log_returns = make_cleaned_call_factory("log_returns")
    volatility = make_cleaned_call_factory("volatility")
    vwap = make_cleaned_call_factory("vwap")
    ts_pct = make_cleaned_call_factory("ts_pct")

    ret_expr = ts_pct(col("Close"), 1)

    for expr, name in (
        (rank_pct(col("Close")), "rank_pct"),
        (cs_pct_rank(col("Close")), "cs_pct_rank"),
        (cs_quantile(col("Close"), 0.5), "cs_quantile"),
        (log_returns(col("Close")), "log_returns"),
        (volatility(ret_expr, 3), "volatility"),
        (vwap(col("Close"), col("Volume"), 3), "vwap"),
    ):
        factor = Factor(name=name, expr=expr)
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
        a = eng_pd.run(factor)["result"]
        b = eng_sql.run(factor)["result"]
        pd.testing.assert_series_equal(
            a, b, check_names=False, rtol=1e-6, atol=1e-6
        )


def test_rank_pct_emits_pct_rank_not_cs_rank_01():
    plan = minimal_plan("rank_pct")
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    sql = compiled.query
    assert "cnt_le" in sql or "countIf" in sql or "COUNT(*) FILTER" in sql
    assert "/ s.cnt" in sql or "nullIf(s.cnt" in sql
    assert "- 1.0) / NULLIF(s.cnt - 1" not in sql
