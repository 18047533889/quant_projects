# -*- coding: utf-8
"""P1 pending 算子 golden：边界 case 三后端 parity（升级 production-safe 前）。"""
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

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def edge_source():
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
    # ties + NaN + flat segment (zero sharpe std)
    close = pd.Series([10.0, 10.0, 11.0, np.nan, 12.0, 5.0, 5.0, 5.0, 6.0, 7.0], index=idx)
    y = pd.Series([1.0, np.nan, 2.0, 3.0, 4.0, 1.0, 2.0, np.nan, 4.0, 5.0], index=idx)
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 1.0, np.nan, 3.0, 4.0, 5.0], index=idx)
    return InMemorySeriesSource(data={"close": close, "y": y, "x": x})


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
    Y: double
    X: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for (ts, sym) in mem.data["close"].index:
        rows.append(
            {
                "TradeDate": ts.date(),
                "Symbol": sym,
                "Close": _f(mem.data["close"].loc[(ts, sym)]),
                "Y": _f(mem.data["y"].loc[(ts, sym)]),
                "X": _f(mem.data["x"].loc[(ts, sym)]),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def _f(v):
    return float(v) if pd.notna(v) else None


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch, edge_source):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data")))
    _seed_duckdb(tmp_path / "data", edge_source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


def _assert_close(a, b, *, rtol=1e-5, atol=1e-5):
    left = pd.Series(a, dtype=float)
    right = pd.Series(b, dtype=float)
    pd.testing.assert_series_equal(left, right, check_names=False, rtol=rtol, atol=atol)


PENDING_GOLDEN_CASES = [
    ("ts_rank", lambda: F("ts_rank")(col("close"), 3)),
    ("ts_sharpe", lambda: F("ts_sharpe")(col("close"), 3)),
    ("ts_autocorr", lambda: F("ts_autocorr")(col("close"), 4, 1)),
    ("cs_resid", lambda: F("cs_resid")(col("y"), col("x"))),
    ("cs_regression", lambda: F("cs_regression")(col("y"), col("x"))),
]

PENDING_DUCKDB_CASES = [
    ("ts_rank", lambda: F("ts_rank")(col("Close"), 3)),
    ("ts_sharpe", lambda: F("ts_sharpe")(col("Close"), 3)),
    ("ts_autocorr", lambda: F("ts_autocorr")(col("Close"), 4, 1)),
    ("cs_resid", lambda: F("cs_resid")(col("Y"), col("X"))),
    ("cs_regression", lambda: F("cs_regression")(col("Y"), col("X"))),
]


@pytest.mark.parametrize("name,expr_builder", PENDING_GOLDEN_CASES)
def test_pending_polars_long_matches_pandas(edge_source, name, expr_builder):
    expr = expr_builder()
    _assert_close(_run(edge_source, expr, "pandas"), _run(edge_source, expr, "polars_long"))


@pytest.mark.parametrize("name,expr_builder", PENDING_DUCKDB_CASES)
def test_pending_duckdb_matches_pandas(duckdb_source, name, expr_builder):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    expr = expr_builder()
    pd_out = _run(duckdb_source, expr, "pandas")
    sql_run = FactorEngine(backend=build_backend("duckdb_sql"), data_source=duckdb_source).run(
        Factor(name="t", expr=expr)
    )
    assert_duckdb_real_sql_execution(sql_run)
    _assert_close(pd_out, sql_run["result"].sort_index())


def test_cs_resid_asymmetric_null_pairwise_duckdb(duckdb_source, edge_source):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    expr = F("cs_resid")(col("y"), col("x"))
    pd_out = _run(edge_source, expr, "pandas")
    duck_expr = F("cs_resid")(col("Y"), col("X"))
    sql_run = FactorEngine(backend=build_backend("duckdb_sql"), data_source=duckdb_source).run(
        Factor(name="t", expr=duck_expr)
    )
    assert_duckdb_real_sql_execution(sql_run)
    _assert_close(pd_out, sql_run["result"].sort_index())


def test_cs_resid_asymmetric_null_pairwise(edge_source):
    """x 有值 y 缺失 / 反之：pairwise mask 须与 pandas 一致。"""
    expr = F("cs_resid")(col("y"), col("x"))
    _assert_close(_run(edge_source, expr, "pandas"), _run(edge_source, expr, "polars_long"))


def test_ts_rank_tie_values(edge_source):
    expr = F("ts_rank")(col("close"), 3)
    pd_out = _run(edge_source, expr, "pandas")
    pl_out = _run(edge_source, expr, "polars_long")
    _assert_close(pd_out, pl_out)
    # instrument B 前三天全为 5.0 — rank 应一致
    b_idx = pd_out.index.get_level_values("instrument") == "B"
    assert pd_out[b_idx].iloc[2:5].notna().any()


def test_ts_sharpe_flat_segment(edge_source):
    expr = F("ts_sharpe")(col("close"), 3)
    pd_out = _run(edge_source, expr, "pandas")
    pl_out = _run(edge_source, expr, "polars_long")
    _assert_close(pd_out, pl_out)
