# -*- coding: utf-8
"""Production core 三后端 parity：Pandas / PolarsLong / DuckDB SQL。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource


def _memory_source():
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
        [10.0, 11.0, np.nan, 12.0, 11.5, 20.0, 21.0, 20.5, 22.0, 21.5],
        index=idx,
    )
    open_ = close - 0.5
    volume = pd.Series(
        [100.0, 110.0, 105.0, 120.0, 115.0, 200.0, 210.0, 205.0, 220.0, 215.0],
        index=idx,
    )
    ret = pd.Series(
        [0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05, 0.03, 0.02],
        index=idx,
    )
    grp = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], index=idx, dtype=float)
    return InMemorySeriesSource(
        data={"close": close, "open": open_, "volume": volume, "ret": ret, "group_id": grp}
    )


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
    Open: double
    Volume: double
    Ret: double
    group_id: int64
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb_panel(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for (ts, sym), close in mem.data["close"].items():
        rows.append(
            {
                "TradeDate": ts.date(),
                "Symbol": sym,
                "Close": float(close) if pd.notna(close) else None,
                "Open": float(mem.data["open"].loc[(ts, sym)]),
                "Volume": float(mem.data["volume"].loc[(ts, sym)]),
                "Ret": float(mem.data["ret"].loc[(ts, sym)]),
                "group_id": int(mem.data["group_id"].loc[(ts, sym)]),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture(scope="module")
def mem_source():
    return _memory_source()


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch, mem_source):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data")))
    _seed_duckdb_panel(tmp_path / "data", mem_source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def _col(name: str):
    mapping = {
        "close": "Close",
        "open": "Open",
        "volume": "Volume",
        "ret": "Ret",
    }
    return col(mapping.get(name, name))


def _run(source, expr, backend_name: str):
    # Certification executes untrusted candidates in research mode.  Production
    # capability is granted only after this suite writes a valid artifact.
    return FactorEngine(
        backend=build_backend(backend_name), data_source=source, run_mode="research"
    ).run(
        Factor(name="t", expr=expr)
    )


def _result_series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


MEMORY_CASES = [
    ("ts_mean", lambda: make_cleaned_call_factory("ts_mean")(col("close"), 3)),
    ("ts_std", lambda: make_cleaned_call_factory("ts_std")(col("close"), 3)),
    ("ts_zscore", lambda: make_cleaned_call_factory("ts_zscore")(col("close"), 5)),
    ("rank", lambda: make_cleaned_call_factory("rank")(col("close"))),
    ("zscore", lambda: make_cleaned_call_factory("zscore")(col("close"))),
    ("protected_div", lambda: make_cleaned_call_factory("protected_div")(col("close"), col("volume"))),
    ("protected_log", lambda: make_cleaned_call_factory("protected_log")(col("close"))),
    ("log_returns", lambda: make_cleaned_call_factory("log_returns")(col("close"))),
    ("group_mean", lambda: make_cleaned_call_factory("group_mean")(col("close"), col("group_id"))),
    ("group_zscore", lambda: make_cleaned_call_factory("group_zscore")(col("close"), col("group_id"))),
    ("group_rank", lambda: make_cleaned_call_factory("group_rank")(col("close"), col("group_id"))),
    ("group_percentile", lambda: make_cleaned_call_factory("group_percentile")(col("close"), col("group_id"), 0.5)),
    ("ts_corr", lambda: make_cleaned_call_factory("ts_corr")(col("close"), col("open"), 3)),
    ("ts_cov", lambda: make_cleaned_call_factory("ts_cov")(col("ret"), col("close"), 3)),
    ("ts_beta", lambda: make_cleaned_call_factory("ts_beta")(col("ret"), col("close"), 3, min_periods=3)),
    ("group_winsorize", lambda: make_cleaned_call_factory("group_winsorize")(col("close"), col("group_id"))),
    ("cs_mad", lambda: make_cleaned_call_factory("cs_mad")(col("close"))),
    ("cs_mad_zscore", lambda: make_cleaned_call_factory("cs_mad_zscore")(col("close"))),
    ("vwap", lambda: make_cleaned_call_factory("vwap")(col("close"), col("volume"), 3)),
    ("power", lambda: make_cleaned_call_factory("power")(col("close"), col("volume"))),
    ("gt", lambda: make_cleaned_call_factory("gt")(col("close"), col("open"))),
    ("cs_mean", lambda: make_cleaned_call_factory("cs_mean")(col("close"))),
    ("cs_pct_rank", lambda: make_cleaned_call_factory("cs_pct_rank")(col("close"))),
    ("ts_median", lambda: make_cleaned_call_factory("ts_median")(col("close"), 3)),
    ("rank_pct", lambda: make_cleaned_call_factory("rank_pct")(col("close"))),
    ("add", lambda: make_cleaned_call_factory("add")(col("close"), col("open"))),
    ("ts_rank", lambda: make_cleaned_call_factory("ts_rank")(col("close"), 3)),
    ("ts_sharpe", lambda: make_cleaned_call_factory("ts_sharpe")(col("close"), 3)),
    ("ts_autocorr", lambda: make_cleaned_call_factory("ts_autocorr")(col("close"), 4, 1)),
    ("cum_delta", lambda: make_cleaned_call_factory("cum_delta")(col("close"))),
    ("expanding_mean", lambda: make_cleaned_call_factory("expanding_mean")(col("close"))),
    ("where", lambda: make_cleaned_call_factory("where")(col("group_id"), col("close"), col("open"))),
]


@pytest.mark.parametrize("name,expr_builder", MEMORY_CASES)
def test_production_core_polars_long_matches_pandas(mem_source, name, expr_builder):
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry._aliases.get(name, name) not in DAILY_CANONICALS:
        pytest.skip("not on the production daily surface")
    expr = expr_builder()
    pd_out = _result_series(_run(mem_source, expr, "pandas"))
    long_out = _result_series(_run(mem_source, expr, "polars_long"))
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


DUCKDB_CASES = [
    ("ts_mean", lambda: make_cleaned_call_factory("ts_mean")(_col("close"), 3)),
    ("ts_std", lambda: make_cleaned_call_factory("ts_std")(_col("close"), 3)),
    ("rank", lambda: make_cleaned_call_factory("rank")(_col("close"))),
    ("zscore", lambda: make_cleaned_call_factory("zscore")(_col("close"))),
    ("protected_div", lambda: make_cleaned_call_factory("protected_div")(_col("close"), _col("volume"))),
    ("protected_log", lambda: make_cleaned_call_factory("protected_log")(_col("close"))),
    ("group_mean", lambda: make_cleaned_call_factory("group_mean")(_col("close"), _col("group_id"))),
    ("group_zscore", lambda: make_cleaned_call_factory("group_zscore")(_col("close"), _col("group_id"))),
    ("ts_corr", lambda: make_cleaned_call_factory("ts_corr")(_col("close"), _col("open"), 3)),
    ("ts_cov", lambda: make_cleaned_call_factory("ts_cov")(_col("ret"), _col("close"), 3)),
    ("ts_beta", lambda: make_cleaned_call_factory("ts_beta")(_col("ret"), _col("close"), 3, min_periods=3)),
    ("group_winsorize", lambda: make_cleaned_call_factory("group_winsorize")(_col("close"), _col("group_id"))),
    ("cs_mad", lambda: make_cleaned_call_factory("cs_mad")(_col("close"))),
    ("cs_mad_zscore", lambda: make_cleaned_call_factory("cs_mad_zscore")(_col("close"))),
]


@pytest.mark.parametrize("name,expr_builder", DUCKDB_CASES)
def test_production_core_duckdb_matches_pandas(duckdb_source, name, expr_builder):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry._aliases.get(name, name) not in DAILY_CANONICALS:
        pytest.skip("not on the production daily surface")

    expr = expr_builder()
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))
    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,expr_builder", DUCKDB_CASES)
def test_production_core_duckdb_matches_polars_long(mem_source, duckdb_source, name, expr_builder):
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry._aliases.get(name, name) not in DAILY_CANONICALS:
        pytest.skip("not on the production daily surface")
    mem_cases = {n: b for n, b in MEMORY_CASES}
    if name not in mem_cases:
        pytest.skip("no memory expr")
    expr = mem_cases[name]()
    long_out = _result_series(_run(mem_source, expr, "polars_long"))
    sql_run = _run(duckdb_source, expr_builder(), "duckdb_sql")
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)
    pd.testing.assert_series_equal(long_out, sql_out, check_names=False, rtol=1e-6, atol=1e-6)
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry._aliases.get(name, name) not in DAILY_CANONICALS:
        pytest.skip("not on the production daily surface")
