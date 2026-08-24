# -*- coding: utf-8
"""P0 path summary / SQL telemetry / parallel ctx 隔离。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.backend.path_summary import infer_primary_route, snapshot_from_run_output
from factor_engine.backend.sql_pushdown.strict import SqlLongPushdownError
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def loaded():
    load_all()
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()


@pytest.fixture
def source():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 20.0, 21.0, 10.5, 12.0], index=idx)
    return InMemorySeriesSource(data={"close": close})


def test_run_includes_full_backend_path(loaded, source):
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run(Factor(name="t", expr=make_cleaned_call_factory("ts_mean")(col("close"), 2)))
    assert "backend_path" in out
    bp = out["backend_path"]
    assert bp.get("primary_route") == "polars_long_native"
    assert bp.get("backend_path_summary")
    assert "polars_long_native_ops" in bp or out.get("polars_long_native_ops") is not None


def test_snapshot_from_run_output_prefers_backend_path():
    payload = {
        "backend_path": {
            "primary_route": "duckdb_sql_full",
            "sql_fully_pushed": True,
            "backend_path_summary": {"fully_sql": True},
        },
        "used_polars_long_native": False,
    }
    snap = snapshot_from_run_output(payload)
    assert snap["primary_route"] == "duckdb_sql_full"
    assert snap["sql_fully_pushed"] is True


def test_infer_duckdb_sql_full_route():
    route = infer_primary_route(
        {
            "backend": "hybrid_long",
            "fully_sql": True,
            "used_sql_pushdown": True,
            "sql_fully_pushed": True,
            "sql_query_count": 1,
            "sql_dialect": "duckdb",
        }
    )
    assert route == "duckdb_sql_full"


def test_infer_sql_full_execution_failed_route():
    route = infer_primary_route(
        {
            "backend": "duckdb_sql",
            "fully_sql": True,
            "sql_fully_pushed": False,
            "sql_full_execution_failed": True,
            "sql_query_count": 0,
        }
    )
    assert route == "sql_full_execution_failed"


def test_execute_root_uses_isolated_runtime_stats(loaded, source):
    """parallel 路径：每个 root 使用独立 runtime_stats，避免互相覆盖。"""
    from dataclasses import replace

    from factor_engine.runtime.batch_service import _execute_root_with_path

    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    ctx = eng._make_context()
    ctx.runtime_stats = {"marker": "shared"}  # type: ignore[attr-defined]
    plan = eng.compile(Factor(name="t", expr=make_cleaned_call_factory("ts_mean")(col("close"), 2)))[0]
    _execute_root_with_path(eng.backend, plan, ctx)
    assert ctx.runtime_stats.get("marker") == "shared"
    assert ctx.runtime_stats.get("used_polars_long_native") is not True


def test_strict_sql_long_raises(monkeypatch):
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.backend.sql_pushdown.strict import handle_sql_long_pushdown_failure

    monkeypatch.setenv("FACTOR_ENGINE_STRICT_SQL_LONG", "1")
    ctx = ExecutionContext(data_source=object())
    with pytest.raises(SqlLongPushdownError):
        handle_sql_long_pushdown_failure(ctx, sid="s1", exc=RuntimeError("boom"), phase="long_single")


def test_final_collect_not_sql_when_execution_failed():
    from factor_engine.backend.path_summary import _final_collect

    assert _final_collect({"fully_sql": True, "sql_full_execution_failed": True}) == (
        "fallback_to_pandas_or_polars"
    )
    assert _final_collect(
        {"fully_sql": True, "sql_fully_pushed": True, "sql_query_count": 2}
    ) == "sql_to_pandas"
    assert _final_collect({"fully_sql": True, "sql_query_count": 0}) == (
        "fallback_to_pandas_or_polars"
    )
