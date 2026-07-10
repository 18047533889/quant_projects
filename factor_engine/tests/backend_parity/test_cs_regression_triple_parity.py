# -*- coding: utf-8
"""cs_regression 三 mode 三后端 parity + DuckDB 真实 SQL。"""
from __future__ import annotations

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
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def ols_source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-02"), "C"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-03"), "C"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-04"), "B"),
            (pd.Timestamp("2024-01-04"), "C"),
        ],
        names=["timestamp", "instrument"],
    )
    y = pd.Series([1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0], index=idx)
    x = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0], index=idx)
    return InMemorySeriesSource(data={"y": y, "x": x})


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_cs_regression_modes_polars_match_pandas(ols_source, mode):
    expr = F("cs_regression")(col("y"), col("x"), mode)
    pd_out = FactorEngine(backend=build_backend("pandas"), data_source=ols_source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()
    pl_out = FactorEngine(backend=build_backend("polars_long"), data_source=ols_source).run(
        Factor(name="t", expr=expr)
    )
    assert pl_out.get("used_polars_long_native") is True
    long_out = pl_out["result"].sort_index()
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-5, atol=1e-5)
    assert pd_out.notna().any()


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_cs_regression_modes_duckdb_matches_pandas(ols_source, mode, tmp_path, monkeypatch):
    from pathlib import Path
    from storage.factory import build_data_source

    def _write_registry(root: Path) -> Path:
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
    Y: double
    X: double
"""
        path = tmp_path / "datasets.yaml"
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return path

    rows = []
    for (ts, sym), yv in ols_source.data["y"].items():
        rows.append(
            {
                "TradeDate": ts.date(),
                "Symbol": sym,
                "Y": float(yv),
                "X": float(ols_source.data["x"].loc[(ts, sym)]),
            }
        )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(data_root / "panel.parquet")

    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(data_root)))
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    duck_source = build_data_source({"type": "data_access", "dataset": "test_daily"})
    expr = F("cs_regression")(col("y"), col("x"), mode)
    pd_out = FactorEngine(backend=build_backend("pandas"), data_source=ols_source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()
    duck_expr = F("cs_regression")(col("Y"), col("X"), mode)
    sql_run = FactorEngine(backend=build_backend("duckdb_sql"), data_source=duck_source).run(
        Factor(name="t", expr=duck_expr)
    )
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = sql_run["result"].sort_index()
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-5, atol=1e-5)


def test_cs_regression_modes_distinct(ols_source):
    outs = []
    for mode in (0, 1, 2):
        expr = F("cs_regression")(col("y"), col("x"), mode)
        outs.append(
            FactorEngine(backend=build_backend("pandas"), data_source=ols_source).run(
                Factor(name="t", expr=expr)
            )["result"].sort_index()
        )
    assert not outs[0].equals(outs[1])
    assert not outs[0].equals(outs[2])
    assert not outs[1].equals(outs[2])
