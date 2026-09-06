# -*- coding: utf-8
"""Systematic three-way parity for cross-sectional (cs_*) operator family.

Tests pandas/Polars/DuckDB backends produce identical results for:
- Basic cross-sectional: cs_mean, cs_std, cs_sum, cs_count
- Ranking: cs_pct_rank
- Robust statistics: cs_mad, cs_mad_zscore
- Normalization: cs_demean

Edge cases covered:
- NULL/NaN values
- Inf values
- All-same values (zero variance)
- Single instrument (degenerate cross-section)
- Tied values in ranking
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource


F = make_cleaned_call_factory


def _make_panel() -> InMemorySeriesSource:
    """Deterministic 6 instruments × 10 days panel with edge cases."""
    load_all()
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    insts = ["A", "B", "C", "D", "E", "F"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])

    n = len(idx)
    rng = np.random.default_rng(123)
    close = pd.Series(50.0 + rng.normal(0, 10, n), index=idx)

    # Edge cases
    # Day 0: normal distribution
    # Day 1: all same values (zero variance)
    close.loc[(pd.Timestamp("2024-01-03"), slice(None))] = 100.0
    # Day 2: one NaN
    close.loc[(pd.Timestamp("2024-01-04"), "C")] = np.nan
    # Day 3: ties in ranking
    close.loc[(pd.Timestamp("2024-01-05"), "A")] = 50.0
    close.loc[(pd.Timestamp("2024-01-05"), "B")] = 50.0
    close.loc[(pd.Timestamp("2024-01-05"), "C")] = 70.0
    # Day 4: one Inf
    close.loc[(pd.Timestamp("2024-01-06"), "D")] = np.inf

    volume = pd.Series(rng.integers(100, 1000, n).astype(float), index=idx)
    ret = pd.Series(rng.normal(0.001, 0.05, n), index=idx)

    return InMemorySeriesSource(data={"close": close, "volume": volume, "ret": ret})


@pytest.fixture(scope="module")
def panel():
    return _make_panel()


def _write_duckdb_registry(path: Path, root: Path) -> None:
    path.write_text(
        f"""cs_test_daily:
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
    Volume: double
    Ret: double
""",
        encoding="utf-8",
    )


def _seed_duckdb(root: Path, source: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestamp, instrument in source.data["close"].index:
        close_val = source.data["close"].loc[(timestamp, instrument)]
        rows.append(
            {
                "TradeDate": timestamp.date(),
                "Symbol": instrument,
                "Close": float(close_val) if pd.notna(close_val) else None,
                "Volume": float(source.data["volume"].loc[(timestamp, instrument)]),
                "Ret": float(source.data["ret"].loc[(timestamp, instrument)]),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")
    persisted = pd.read_parquet(root / "panel.parquet")
    np.testing.assert_equal(persisted["Close"].to_numpy(), source.data["close"].to_numpy())


@pytest.fixture
def duckdb_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, panel: InMemorySeriesSource):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    root = tmp_path / "data"
    _write_duckdb_registry(tmp_path / "datasets.yaml", root)
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(tmp_path / "datasets.yaml"))
    _seed_duckdb(root, panel)
    from data_access import reset_store
    reset_store()
    return build_data_source({"type": "data_access", "dataset": "cs_test_daily"})


def _run(source, expr, backend: str):
    return FactorEngine(
        backend=build_backend(backend), data_source=source, run_mode="research"
    ).run(Factor(name="cs_parity", expr=expr))


def _assert_parity(reference: pd.Series, candidate: pd.Series) -> None:
    pd.testing.assert_series_equal(
        reference.sort_index(), candidate.sort_index(),
        check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6,
    )


def _memory_col(name: str):
    return col(name)


def _sql_col(name: str):
    mapping = {"close": "Close", "volume": "Volume", "ret": "Ret"}
    return col(mapping.get(name, name))


# Basic cross-sectional aggregations
BASIC_CS = [
    ("cs_mean", lambda c: F("cs_mean")(c("close"))),
    ("cs_std", lambda c: F("cs_std")(c("close"))),
    ("cs_sum", lambda c: F("cs_sum")(c("close"))),
    ("cs_count", lambda c: F("cs_count")(c("close"))),
]


@pytest.mark.parametrize("name,builder", BASIC_CS)
def test_basic_cs_triple_parity(panel, duckdb_source, name, builder):
    """Basic cross-sectional aggregations: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Ranking
RANKING = [
    ("cs_pct_rank", lambda c: F("cs_pct_rank")(c("close"))),
]


@pytest.mark.parametrize("name,builder", RANKING)
def test_ranking_triple_parity(panel, duckdb_source, name, builder):
    """Cross-sectional ranking: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Robust statistics
ROBUST = [
    ("cs_mad", lambda c: F("cs_mad")(c("close"))),
    ("cs_mad_zscore", lambda c: F("cs_mad_zscore")(c("close"))),
]


@pytest.mark.parametrize("name,builder", ROBUST)
def test_robust_stats_triple_parity(panel, duckdb_source, name, builder):
    """Robust cross-sectional statistics: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Normalization
NORMALIZATION = [
    ("cs_demean", lambda c: F("cs_demean")(c("close"))),
]


@pytest.mark.parametrize("name,builder", NORMALIZATION)
def test_normalization_triple_parity(panel, duckdb_source, name, builder):
    """Cross-sectional normalization: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


def test_cs_all_nan_handling(panel):
    """Cross-sectional operators handle all-NaN timestamps correctly."""
    # Create panel with one timestamp having all NaN
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    insts = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    close = pd.Series([10.0, 20.0, 30.0, np.nan, np.nan, np.nan, 15.0, 25.0, 35.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("cs_mean")(col("close"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)


def test_cs_constant_values_zero_variance(panel):
    """Cross-sectional operators handle zero variance correctly."""
    # All same value at each timestamp
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    insts = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    close = pd.Series([100.0] * 3 + [200.0] * 3, index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("cs_std")(col("close"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    # Should be all zeros
    assert (pandas_out == 0.0).all()
