# -*- coding: utf-8
"""Production core 算子 SQL emitter 覆盖。"""

from __future__ import annotations

import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql
from backend.sql_pushdown.sql_registry import is_sql_capable
from ir.analyzer import Analyzer


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()
    from backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()


def _plan(expr):
    from api.factor import Factor
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource
    import pandas as pd

    idx = pd.MultiIndex.from_product([[pd.Timestamp("2024-01-01")], ["A"]], names=["timestamp", "instrument"])
    ret = pd.Series([0.01], index=idx)
    source = InMemorySeriesSource(data={"ret": ret})
    eng = FactorEngine(backend=build_backend("pandas"), data_source=source)
    plan, _ = eng.compile(Factor(name="t", expr=expr))
    return plan


def test_production_core_sql_capable():
    from backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS

    for name in ("ts_sharpe", "ts_autocorr", "rolling_beta"):
        assert name in SQL_CAPABLE_CANONICALS, name


def test_ts_sharpe_emits_window_sql():
    from api.cleaned_ops import make_cleaned_call_factory
    from api.columns import col

    ts_sharpe = make_cleaned_call_factory("ts_sharpe")
    plan = _plan(ts_sharpe(col("ret"), 20))
    compiled = compile_plan_to_sql(
        plan,
        dataset="test_daily",
        time_column="trade_date",
        instrument_column="ticker",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    sql = compiled.query.lower()
    assert "stddev" in sql or "std" in sql
    assert "15.874" in compiled.query or "sqrt" in sql
