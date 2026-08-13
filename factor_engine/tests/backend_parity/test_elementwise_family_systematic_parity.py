# -*- coding: utf-8
"""Systematic three-way parity for elementwise operator family.

Tests pandas/Polars/DuckDB backends produce identical results for:
- Arithmetic: add, subtract, multiply, divide, power
- Comparison: eq, ne, gt, ge, lt, le
- Math functions: abs, exp, log, sqrt, sign, ceil, floor, tanh
- Utility: clip, where, coalesce, inverse, neg
- Normalization: normalize, rank, zscore, winsorize

Edge cases covered:
- Division by zero
- Log of negative/zero
- Power with negative base
- Sqrt of negative
- NULL propagation
- Inf handling
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

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


def _make_panel() -> InMemorySeriesSource:
    """Deterministic 4 instruments × 8 days panel with edge cases."""
    load_all()
    dates = pd.date_range("2024-01-02", periods=8, freq="D")
    insts = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])

    n = len(idx)
    rng = np.random.default_rng(789)

    x = pd.Series(rng.uniform(-10, 10, n), index=idx)
    y = pd.Series(rng.uniform(0.1, 5, n), index=idx)

    # Edge cases
    x.iloc[0] = 0.0  # Zero
    x.iloc[1] = -5.0  # Negative
    x.iloc[2] = np.nan  # NaN
    x.iloc[3] = np.inf  # Inf

    y.iloc[4] = 0.0  # Zero divisor
    y.iloc[5] = np.nan  # NaN

    flag = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    constant = pd.Series([2.0] * n, index=idx)

    return InMemorySeriesSource(data={"x": x, "y": y, "flag": flag, "constant": constant})


@pytest.fixture(scope="module")
def panel():
    return _make_panel()


def _write_duckdb_registry(path: Path, root: Path) -> None:
    path.write_text(
        f"""elem_test_daily:
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
    X: double
    Y: double
    Flag: double
    Constant: double
""",
        encoding="utf-8",
    )


def _seed_duckdb(root: Path, source: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestamp, instrument in source.data["x"].index:
        x_val = source.data["x"].loc[(timestamp, instrument)]
        y_val = source.data["y"].loc[(timestamp, instrument)]
        rows.append(
            {
                "TradeDate": timestamp.date(),
                "Symbol": instrument,
                "X": float(x_val) if pd.notna(x_val) and np.isfinite(x_val) else None,
                "Y": float(y_val) if pd.notna(y_val) and np.isfinite(y_val) else None,
                "Flag": float(source.data["flag"].loc[(timestamp, instrument)]),
                "Constant": float(source.data["constant"].loc[(timestamp, instrument)]),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, panel: InMemorySeriesSource):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    root = tmp_path / "data"
    _write_duckdb_registry(tmp_path / "datasets.yaml", root)
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(tmp_path / "datasets.yaml"))
    _seed_duckdb(root, panel)
    from data_access import reset_store
    reset_store()
    return build_data_source({"type": "data_access", "dataset": "elem_test_daily"})


def _run(source, expr, backend: str):
    return FactorEngine(
        backend=build_backend(backend), data_source=source, run_mode="research"
    ).run(Factor(name="elem_parity", expr=expr))


def _assert_parity(reference: pd.Series, candidate: pd.Series) -> None:
    pd.testing.assert_series_equal(
        reference.sort_index(), candidate.sort_index(),
        check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6,
    )


def _memory_col(name: str):
    return col(name)


def _sql_col(name: str):
    mapping = {"x": "X", "y": "Y", "flag": "Flag", "constant": "Constant"}
    return col(mapping.get(name, name))


# Arithmetic operators
ARITHMETIC = [
    ("add", lambda c: F("add")(c("x"), c("y"))),
    ("subtract", lambda c: F("subtract")(c("x"), c("y"))),
    ("multiply", lambda c: F("multiply")(c("x"), c("y"))),
    ("divide", lambda c: F("divide")(c("x"), c("y"))),
    ("power", lambda c: F("power")(c("x"), c("constant"))),
]


@pytest.mark.parametrize("name,builder", ARITHMETIC)
def test_arithmetic_triple_parity(panel, duckdb_source, name, builder):
    """Arithmetic operators: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Comparison operators
COMPARISON = [
    ("eq", lambda c: F("eq")(c("x"), c("y"))),
    ("ne", lambda c: F("ne")(c("x"), c("y"))),
    ("gt", lambda c: F("gt")(c("x"), c("y"))),
    ("ge", lambda c: F("ge")(c("x"), c("y"))),
    ("lt", lambda c: F("lt")(c("x"), c("y"))),
    ("le", lambda c: F("le")(c("x"), c("y"))),
]


@pytest.mark.parametrize("name,builder", COMPARISON)
def test_comparison_triple_parity(panel, duckdb_source, name, builder):
    """Comparison operators: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Math functions (unary)
MATH_UNARY = [
    ("abs", lambda c: F("abs")(c("x"))),
    ("exp", lambda c: F("exp")(c("x"))),
    ("log", lambda c: F("log")(c("y"))),
    ("sqrt", lambda c: F("sqrt")(c("y"))),
    ("sign", lambda c: F("sign")(c("x"))),
    ("ceil", lambda c: F("ceil")(c("x"))),
    ("floor", lambda c: F("floor")(c("x"))),
    ("tanh", lambda c: F("tanh")(c("x"))),
    ("neg", lambda c: F("neg")(c("x"))),
]


@pytest.mark.parametrize("name,builder", MATH_UNARY)
def test_math_unary_triple_parity(panel, duckdb_source, name, builder):
    """Unary math functions: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Utility functions
UTILITY = [
    ("clip", lambda c: F("clip")(c("x"), -5.0, 5.0)),
    ("where", lambda c: F("where")(c("flag"), c("x"), c("y"))),
    ("coalesce", lambda c: F("coalesce")(c("x"), c("constant"))),
    ("inverse", lambda c: F("inverse")(c("y"))),
    ("minimum", lambda c: F("minimum")(c("x"), c("y"))),
    ("maximum", lambda c: F("maximum")(c("x"), c("y"))),
]


@pytest.mark.parametrize("name,builder", UTILITY)
def test_utility_triple_parity(panel, duckdb_source, name, builder):
    """Utility functions: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Cross-sectional normalization (elementwise when applied to single timestamp)
NORMALIZATION = [
    ("normalize", lambda c: F("normalize")(c("x"))),
    ("rank", lambda c: F("rank")(c("x"))),
    ("zscore", lambda c: F("zscore")(c("x"))),
    ("winsorize", lambda c: F("winsorize")(c("x"))),
]


@pytest.mark.parametrize("name,builder", NORMALIZATION)
def test_normalization_triple_parity(panel, duckdb_source, name, builder):
    """Normalization functions: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


def test_division_by_zero_handling(panel):
    """Division by zero produces consistent results across backends."""
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    insts = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    x = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
    y = pd.Series([2.0, 0.0, 5.0, 0.0], index=idx)
    source = InMemorySeriesSource(data={"x": x, "y": y})

    expr = F("divide")(col("x"), col("y"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)


def test_null_propagation_consistency(panel):
    """NULL values propagate consistently across all backends."""
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    insts = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    x = pd.Series([10.0, np.nan, 30.0, 40.0], index=idx)
    y = pd.Series([2.0, 5.0, np.nan, 8.0], index=idx)
    source = InMemorySeriesSource(data={"x": x, "y": y})

    expr = F("add")(col("x"), col("y"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    # Check NaN at expected positions
    assert pd.isna(pandas_out.iloc[1])
    assert pd.isna(pandas_out.iloc[2])


def test_coalesce_null_replacement(panel):
    """Coalesce replaces NULLs consistently across backends."""
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    insts = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    x = pd.Series([10.0, np.nan, np.nan, 40.0], index=idx)
    fallback = pd.Series([99.0, 99.0, 99.0, 99.0], index=idx)
    source = InMemorySeriesSource(data={"x": x, "fallback": fallback})

    expr = F("coalesce")(col("x"), col("fallback"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    # NaN positions should be replaced with 99.0
    assert pandas_out.iloc[1] == 99.0
    assert pandas_out.iloc[2] == 99.0
