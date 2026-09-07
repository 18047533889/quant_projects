# -*- coding: utf-8
"""cs_ 截面 + group_ 分组家族原生后端 gap parity.

三后端 parity：Pandas（authority）/ PolarsLong（原生 expr）/ DuckDB SQL。

本套件只覆盖**纯可原生表达**且已落原生分支的算子：
  - polars_long native:  cs_physical_panel_coverage, group_rank_weighted_value
  - duckdb SQL:          cs_physical_panel_coverage（截面广播聚合）
其余 rowwise leave-one-out / OLS 模型类算子（group_ex_self_* / group_multi_resid /
group_impute_median / group_tail_ratio）语义无法用纯 over 窗口表达式忠实复现，
保持走 audited bridge（不在本套件中做近似断言）。
"""
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
    insts = ["A", "B", "C", "D", "E", "F"]
    ts = pd.date_range("2024-01-02", periods=4)
    idx = pd.MultiIndex.from_tuples(
        [(t, i) for t in ts for i in insts], names=["timestamp", "instrument"]
    )
    rng = np.random.default_rng(7)
    close = pd.Series(rng.normal(100.0, 20.0, len(idx)), index=idx)
    close.iloc[5] = np.nan  # one NaN (inst B, date 2) -> panel coverage 5/6
    open_ = close - 0.5
    volume = pd.Series(rng.normal(1000.0, 100.0, len(idx)), index=idx)
    grp = pd.Series(
        [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
        index=idx,
        dtype=float,
    )
    return InMemorySeriesSource(
        data={"close": close, "open": open_, "volume": volume, "group_id": grp}
    )


def _col(name: str):
    mapping = {"close": "Close", "open": "Open", "volume": "Volume"}
    return col(mapping.get(name, name))


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
test_csg:
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
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data"))
    )
    _seed_duckdb_panel(tmp_path / "data", mem_source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_csg"})


def _run(source, expr, backend_name: str):
    return FactorEngine(
        backend=build_backend(backend_name), data_source=source, run_mode="research"
    ).run(Factor(name="t", expr=expr))


def _result(pd_out) -> pd.Series:
    return pd_out["result"].sort_index()


# --- candidates that run on native polars_long / duckdb_sql -----------------
# polars_long-native + duckdb_sql both native (full parity).
NATIVE_CASES = [
    ("cs_physical_panel_coverage", lambda: make_cleaned_call_factory("cs_physical_panel_coverage")(col("close"))),
]

# polars_long-native only (SQL defers to the audited bridge).
POLARS_ONLY_CASES = [
    ("group_rank_weighted_value", lambda: make_cleaned_call_factory("group_rank_weighted_value")(col("close"), col("group_id"))),
]


@pytest.mark.parametrize("name,expr_builder", NATIVE_CASES + POLARS_ONLY_CASES)
def test_polars_long_matches_pandas(mem_source, name, expr_builder):
    expr = expr_builder()
    pd_out = _result(_run(mem_source, expr, "pandas"))
    long_out = _result(_run(mem_source, expr, "polars_long"))
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,expr_builder", NATIVE_CASES)
def test_duckdb_matches_pandas(mem_source, duckdb_source, name, expr_builder):
    pd_out = _result(_run(mem_source, expr_builder(), "pandas"))
    sql_run = _run(duckdb_source, expr_builder(), "duckdb_sql")
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result(sql_run)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-6, atol=1e-6)


# --- non-native rowwise kernels: honestly NOT native-polars / NOT SQL ------=
# These are leave-one-out / min_group_size / OLS rowwise kernels with no
# pure-over polars expression and no DuckDB window SQL.  They are NOT promoted
# to POLARS_LONG_NATIVE and are NOT in SQL_IMPLEMENTED_CANONICALS — they keep
# running on the pandas backend.  Assert that classification is honest (we do
# NOT silently claim a native backend we didn't implement).
@pytest.mark.parametrize(
    "name",
    [
        "group_ex_self_std",
        "group_ex_self_mad",
        "group_ex_self_quantile",
        "group_impute_median",
        "group_tail_ratio",
    ],
)
def test_rowwise_rowover_ops_stay_pandas_only(name):
    from factor_engine.backend.polars_long_policy import infer_polars_long_tier
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    # Not claimed native on the polars_long fastpath.
    assert infer_polars_long_tier(name) != "native"
    # Not claimed SQL-implemented.
    assert name not in SQL_IMPLEMENTED_CANONICALS
