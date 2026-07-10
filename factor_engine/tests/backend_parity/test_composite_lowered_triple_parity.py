# -*- coding: utf-8
"""Composite lowering 后三后端 parity（Pandas / PolarsLong native / DuckDB SQL）。"""
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
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def composite_source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
            (pd.Timestamp("2024-01-05"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 20.0, 19.0, 21.0, 22.0], index=idx)
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series([100.0, 110.0, 0.0, 120.0, 200.0, 210.0, 190.0, 220.0], index=idx)
    return InMemorySeriesSource(
        data={"close": close, "high": high, "low": low, "volume": volume}
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
    High: double
    Low: double
    Volume: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "close": "Close",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    }
    rows = []
    for (ts, sym) in mem.data["close"].index:
        row = {"TradeDate": ts.date(), "Symbol": sym}
        for src, dst in mapping.items():
            val = mem.data[src].loc[(ts, sym)]
            row[dst] = float(val) if pd.notna(val) else None
        rows.append(row)
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture(scope="module")
def duckdb_composite_source(tmp_path_factory, composite_source):
    tmp = tmp_path_factory.mktemp("duckdb_composite")
    import os

    os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
    reg = _write_duckdb_registry(tmp, tmp / "data")
    os.environ["DATA_ACCESS_CONFIG"] = str(reg)
    _seed_duckdb(tmp / "data", composite_source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def _col_mem(name: str):
    return col(name)


def _col_duck(name: str):
    return col(
        {
            "close": "Close",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
        }.get(name, name)
    )


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _assert_lowered_triple(
    mem_source,
    duckdb_source,
    formula_mem,
    formula_duck=None,
    *,
    rtol=1e-5,
    atol=1e-5,
):
    duck_formula = formula_duck or formula_mem
    pd_out = _run(mem_source, formula_mem, "pandas")["result"].sort_index()
    long_run = _run(mem_source, formula_mem, "polars_long")
    long_out = long_run["result"].sort_index()
    duck_run = _run(duckdb_source, duck_formula, "duckdb_sql")
    duck_out = duck_run["result"].sort_index()

    assert long_run.get("used_polars_long_native") is True, long_run
    assert long_run.get("used_polars_long_registry") is not True, long_run
    assert_duckdb_real_sql_execution(duck_run)

    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=rtol, atol=atol)
    pd.testing.assert_series_equal(pd_out, duck_out, check_names=False, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "mem_builder,duck_builder",
    [
        (
            lambda: F("MOM")(_col_mem("close"), 3),
            lambda: F("MOM")(_col_duck("close"), 3),
        ),
        (
            lambda: F("ROC")(_col_mem("close"), 3),
            lambda: F("ROC")(_col_duck("close"), 3),
        ),
        (
            lambda: F("WilliamsR")(_col_mem("high"), _col_mem("low"), _col_mem("close"), 3),
            lambda: F("WilliamsR")(_col_duck("high"), _col_duck("low"), _col_duck("close"), 3),
        ),
        (
            lambda: F("StochasticK")(_col_mem("high"), _col_mem("low"), _col_mem("close"), 3),
            lambda: F("StochasticK")(_col_duck("high"), _col_duck("low"), _col_duck("close"), 3),
        ),
        (
            lambda: F("OBV")(_col_mem("close"), _col_mem("volume")),
            lambda: F("OBV")(_col_duck("close"), _col_duck("volume")),
        ),
    ],
)
def test_composite_lowered_triple_parity(composite_source, duckdb_composite_source, mem_builder, duck_builder):
    _assert_lowered_triple(
        composite_source,
        duckdb_composite_source,
        mem_builder(),
        duck_builder(),
    )
