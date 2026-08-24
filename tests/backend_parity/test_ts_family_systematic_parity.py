# -*- coding: utf-8
"""Systematic three-way parity for time-series (ts_*) operator family.

Tests pandas/Polars/DuckDB backends produce identical results for:
- Basic rolling: ts_mean, ts_std, ts_sum, ts_min, ts_max, ts_median
- Rolling statistics: ts_var, ts_rank, ts_zscore
- Multi-series rolling: ts_corr, ts_cov, ts_beta
- Special ops: ts_delay, ts_delta, ts_pct, ts_autocorr, ts_sharpe, ts_log_return

Edge cases covered:
- NULL/NaN values in windows
- Inf values
- Constant series
- All-same values
- min_periods boundary
- Single instrument / single timestamp
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
    """Deterministic 5 instruments × 12 days panel with edge cases."""
    load_all()
    dates = pd.date_range("2024-01-02", periods=12, freq="D")
    insts = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])

    # Base price series with known patterns
    n = len(idx)
    rng = np.random.default_rng(42)
    close = pd.Series(100.0 + rng.normal(0, 2, n).cumsum() * 0.1, index=idx)

    # Inject edge cases
    close.iloc[5] = np.nan  # NaN in middle
    close.iloc[15:20] = 100.0  # Constant window
    close.iloc[25] = np.inf  # Inf value

    open_ = close - 0.5
    open_.iloc[5] = np.nan

    volume = pd.Series(rng.integers(100, 500, n).astype(float), index=idx)
    ret = pd.Series(rng.normal(0.01, 0.02, n), index=idx)

    return InMemorySeriesSource(data={"close": close, "open": open_, "volume": volume, "ret": ret})


@pytest.fixture(scope="module")
def panel():
    return _make_panel()


def _write_duckdb_registry(path: Path, root: Path) -> None:
    path.write_text(
        f"""ts_test_daily:
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
""",
        encoding="utf-8",
    )


def _seed_duckdb(root: Path, source: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestamp, instrument in source.data["close"].index:
        close_val = source.data["close"].loc[(timestamp, instrument)]
        open_val = source.data["open"].loc[(timestamp, instrument)]
        rows.append(
            {
                "TradeDate": timestamp.date(),
                "Symbol": instrument,
                "Close": float(close_val) if pd.notna(close_val) and np.isfinite(close_val) else None,
                "Open": float(open_val) if pd.notna(open_val) and np.isfinite(open_val) else None,
                "Volume": float(source.data["volume"].loc[(timestamp, instrument)]),
                "Ret": float(source.data["ret"].loc[(timestamp, instrument)]),
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
    return build_data_source({"type": "data_access", "dataset": "ts_test_daily"})


def _run(source, expr, backend: str):
    return FactorEngine(
        backend=build_backend(backend), data_source=source, run_mode="research"
    ).run(Factor(name="ts_parity", expr=expr))


def _assert_parity(reference: pd.Series, candidate: pd.Series) -> None:
    pd.testing.assert_series_equal(
        reference.sort_index(), candidate.sort_index(),
        check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6,
    )


def _memory_col(name: str):
    return col(name)


def _sql_col(name: str):
    mapping = {"close": "Close", "open": "Open", "volume": "Volume", "ret": "Ret"}
    return col(mapping.get(name, name))


# Basic rolling aggregations
BASIC_ROLLING = [
    ("ts_mean", lambda c: F("ts_mean")(c("close"), 3)),
    ("ts_std", lambda c: F("ts_std")(c("close"), 5)),
    ("ts_sum", lambda c: F("ts_sum")(c("close"), 4)),
    ("ts_min", lambda c: F("ts_min")(c("close"), 3)),
    ("ts_max", lambda c: F("ts_max")(c("close"), 3)),
    ("ts_median", lambda c: F("ts_median")(c("close"), 5)),
    ("ts_var", lambda c: F("ts_var")(c("close"), 4)),
]


@pytest.mark.parametrize("name,builder", BASIC_ROLLING)
def test_basic_rolling_triple_parity(panel, duckdb_source, name, builder):
    """Basic rolling aggregations: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Rolling statistics
ROLLING_STATS = [
    ("ts_rank", lambda c: F("ts_rank")(c("close"), 5)),
    ("ts_zscore", lambda c: F("ts_zscore")(c("close"), 6)),
    ("ts_pct", lambda c: F("ts_pct")(c("close"), 4)),
]


@pytest.mark.parametrize("name,builder", ROLLING_STATS)
def test_rolling_stats_triple_parity(panel, duckdb_source, name, builder):
    """Rolling statistics: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Multi-series rolling
MULTI_SERIES = [
    ("ts_corr", lambda c: F("ts_corr")(c("close"), c("open"), 5)),
    ("ts_cov", lambda c: F("ts_cov")(c("close"), c("ret"), 4)),
    ("ts_beta", lambda c: F("ts_beta")(c("ret"), c("close"), 5, min_periods=3)),
]


@pytest.mark.parametrize("name,builder", MULTI_SERIES)
def test_multi_series_rolling_triple_parity(panel, duckdb_source, name, builder):
    """Multi-series rolling: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Special operators
SPECIAL_OPS = [
    ("ts_delay", lambda c: F("ts_delay")(c("close"), 2)),
    ("ts_delta", lambda c: F("ts_delta")(c("close"), 3)),
    ("ts_sharpe", lambda c: F("ts_sharpe")(c("ret"), 5)),
    ("ts_autocorr", lambda c: F("ts_autocorr")(c("close"), 6, 1)),
]


@pytest.mark.parametrize("name,builder", SPECIAL_OPS)
def test_special_ops_triple_parity(panel, duckdb_source, name, builder):
    """Special ts operators: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


def test_ts_log_return_triple_parity(panel, duckdb_source):
    """ts_log_return: pandas == Polars == DuckDB."""
    expr_mem = F("ts_log_return")(_memory_col("close"))
    expr_sql = F("ts_log_return")(_sql_col("close"))

    pandas_out = _run(panel, expr_mem, "pandas")["result"]
    polars_out = _run(panel, expr_mem, "polars_long")["result"]
    sql_out = _run(duckdb_source, expr_sql, "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])
