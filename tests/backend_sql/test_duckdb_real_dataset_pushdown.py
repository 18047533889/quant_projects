# -*- coding: utf-8
"""真实 parquet + data_access DuckDB 下推测试。"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="legacy SQL alias matrix superseded by real-SQL primitive certification")

from tests.backend_parity.test_production_core_triple_parity import (
    DUCKDB_CASES,
    _col,
    _memory_source,
    _result_series,
    _run,
    _seed_duckdb_panel,
    _write_duckdb_registry,
)


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch):
    mem = _memory_source()
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG",
        str(_write_duckdb_registry(tmp_path, tmp_path / "data")),
    )
    _seed_duckdb_panel(tmp_path / "data", mem)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    from factor_engine.storage.factory import build_data_source

    return build_data_source({"type": "data_access", "dataset": "test_daily"})


from factor_engine.api.cleaned_ops import make_cleaned_call_factory

EXTRA_DUCKDB_CASES = [
    ("vwap", lambda: make_cleaned_call_factory("vwap")(_col("close"), _col("volume"), 3)),
    (
        "where",
        lambda: make_cleaned_call_factory("where")(
            make_cleaned_call_factory("gt")(_col("close"), 0.0),
            _col("close"),
            _col("volume"),
        ),
    ),
    ("scale", lambda: make_cleaned_call_factory("scale")(_col("close"))),
    ("winsorize", lambda: make_cleaned_call_factory("winsorize")(_col("close"))),
]


@pytest.mark.parametrize("name,expr_builder", DUCKDB_CASES + EXTRA_DUCKDB_CASES)
def test_duckdb_real_dataset_pushdown_matches_pandas(duckdb_source, name, expr_builder):
    from factor_engine.api.factor import Factor
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine

    expr = expr_builder()
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))
    from factor_engine.backend.factory import build_backend

    out = FactorEngine(
        backend=build_backend("duckdb_sql"), data_source=duckdb_source
    ).run(Factor(name="t", expr=expr))
    bps = out.get("backend_path_summary") or {}
    assert (
        out.get("used_sql_pushdown")
        or out.get("fully_sql")
        or bps.get("fully_sql")
        or bps.get("used_sql_pushdown")
    ), f"{name}: 未触发 SQL pushdown"
    sql_out = out["result"].sort_index()
    import pandas as pd

    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_auto_long_hybrid_matches_pandas(duckdb_source):
    """auto_long + DuckDB：parity 且 primary_route 为 SQL 或 hybrid polars。"""
    from factor_engine.api.factor import Factor
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine

    f = make_cleaned_call_factory
    expr = f("rank")(f("ts_mean")(_col("close"), 3))
    out = FactorEngine(
        backend=build_backend("auto_long"), data_source=duckdb_source
    ).run(Factor(name="hybrid", expr=expr))
    bps = out.get("backend_path_summary") or {}
    route = str(bps.get("primary_route") or "")
    assert route in {
        "duckdb_sql_full",
        "clickhouse_sql_full",
        "sql_partial_polars_long",
        "sql_partial",
        "polars_long_native",
        "polars_long",
    }, route
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))
    import pandas as pd

    pd.testing.assert_series_equal(
        pd_out, out["result"].sort_index(), check_names=False, rtol=1e-5, atol=1e-5
    )
