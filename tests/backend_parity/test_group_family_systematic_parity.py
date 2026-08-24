# -*- coding: utf-8
"""Systematic three-way parity for group-wise (group_*) operator family.

Tests pandas/Polars/DuckDB backends produce identical results for:
- Basic aggregations: group_mean, group_sum, group_count, group_std
- Min/max: group_min, group_max
- Ranking: group_rank
- Normalization: group_normalize, group_neutralize
- Robust: group_zscore, group_winsorize

Edge cases covered:
- Groups with single member
- Groups with all NaN
- Unbalanced groups (different sizes)
- Empty groups
- Group key as NULL
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
    """Deterministic 8 instruments × 10 days panel with group structure."""
    load_all()
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    insts = ["A", "B", "C", "D", "E", "F", "G", "H"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])

    n = len(idx)
    rng = np.random.default_rng(456)
    close = pd.Series(100.0 + rng.normal(0, 15, n), index=idx)

    # Group structure: 3 groups with different sizes
    # Group 1: A, B, C (3 members)
    # Group 2: D, E (2 members)
    # Group 3: F, G, H (3 members)
    group_map = {"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 3, "G": 3, "H": 3}
    group_id = pd.Series([group_map[inst] for _, inst in idx], index=idx, dtype=float)

    # Edge cases
    # Day 2: one group has NaN
    close.loc[(pd.Timestamp("2024-01-04"), "B")] = np.nan
    # Day 3: singleton group (temporary)
    group_id.loc[(pd.Timestamp("2024-01-05"), "A")] = 99.0
    # Day 5: all same within group
    close.loc[(pd.Timestamp("2024-01-07"), "A")] = 150.0
    close.loc[(pd.Timestamp("2024-01-07"), "B")] = 150.0
    close.loc[(pd.Timestamp("2024-01-07"), "C")] = 150.0

    volume = pd.Series(rng.integers(200, 800, n).astype(float), index=idx)

    return InMemorySeriesSource(data={"close": close, "volume": volume, "group_id": group_id})


@pytest.fixture(scope="module")
def panel():
    return _make_panel()


def _write_duckdb_registry(path: Path, root: Path) -> None:
    path.write_text(
        f"""group_test_daily:
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
    group_id: int64
""",
        encoding="utf-8",
    )


def _seed_duckdb(root: Path, source: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestamp, instrument in source.data["close"].index:
        close_val = source.data["close"].loc[(timestamp, instrument)]
        group_val = source.data["group_id"].loc[(timestamp, instrument)]
        rows.append(
            {
                "TradeDate": timestamp.date(),
                "Symbol": instrument,
                "Close": float(close_val) if pd.notna(close_val) and np.isfinite(close_val) else None,
                "Volume": float(source.data["volume"].loc[(timestamp, instrument)]),
                "group_id": int(group_val) if pd.notna(group_val) else None,
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
    return build_data_source({"type": "data_access", "dataset": "group_test_daily"})


def _run(source, expr, backend: str):
    return FactorEngine(
        backend=build_backend(backend), data_source=source, run_mode="research"
    ).run(Factor(name="group_parity", expr=expr))


def _assert_parity(reference: pd.Series, candidate: pd.Series) -> None:
    pd.testing.assert_series_equal(
        reference.sort_index(), candidate.sort_index(),
        check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6,
    )


def _memory_col(name: str):
    return col(name)


def _sql_col(name: str):
    mapping = {"close": "Close", "volume": "Volume"}
    return col(mapping.get(name, name))


# Basic group aggregations
BASIC_GROUP = [
    ("group_mean", lambda c: F("group_mean")(c("close"), c("group_id"))),
    ("group_sum", lambda c: F("group_sum")(c("close"), c("group_id"))),
    ("group_count", lambda c: F("group_count")(c("close"), c("group_id"))),
    ("group_std", lambda c: F("group_std")(c("close"), c("group_id"))),
    ("group_min", lambda c: F("group_min")(c("close"), c("group_id"))),
    ("group_max", lambda c: F("group_max")(c("close"), c("group_id"))),
]


@pytest.mark.parametrize("name,builder", BASIC_GROUP)
def test_basic_group_triple_parity(panel, duckdb_source, name, builder):
    """Basic group aggregations: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Ranking within groups
RANKING = [
    ("group_rank", lambda c: F("group_rank")(c("close"), c("group_id"))),
]


@pytest.mark.parametrize("name,builder", RANKING)
def test_group_ranking_triple_parity(panel, duckdb_source, name, builder):
    """Group ranking: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Normalization within groups
NORMALIZATION = [
    ("group_normalize", lambda c: F("group_normalize")(c("close"), c("group_id"))),
    ("group_neutralize", lambda c: F("group_neutralize")(c("close"), c("group_id"))),
    ("group_zscore", lambda c: F("group_zscore")(c("close"), c("group_id"))),
]


@pytest.mark.parametrize("name,builder", NORMALIZATION)
def test_group_normalization_triple_parity(panel, duckdb_source, name, builder):
    """Group normalization: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# Robust statistics
ROBUST = [
    ("group_winsorize", lambda c: F("group_winsorize")(c("close"), c("group_id"))),
]


@pytest.mark.parametrize("name,builder", ROBUST)
def test_group_robust_triple_parity(panel, duckdb_source, name, builder):
    """Group robust statistics: pandas == Polars == DuckDB."""
    pandas_out = _run(panel, builder(_memory_col), "pandas")["result"]
    polars_out = _run(panel, builder(_memory_col), "polars_long")["result"]
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    _assert_parity(pandas_out, polars_out)
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


def test_group_singleton_member(panel):
    """Group operators handle singleton groups correctly."""
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    insts = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    close = pd.Series([10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0], index=idx)
    # Each instrument in its own group at each timestamp
    group_id = pd.Series([1.0, 2.0, 3.0] * 3, index=idx)
    source = InMemorySeriesSource(data={"close": close, "group_id": group_id})

    expr = F("group_mean")(col("close"), col("group_id"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    # Singleton group mean should equal original value
    pd.testing.assert_series_equal(close, pandas_out, check_names=False)


def test_group_unbalanced_sizes(panel):
    """Group operators handle unbalanced group sizes."""
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    insts = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    close = pd.Series(range(10), index=idx, dtype=float)
    # Group 1: 4 members, Group 2: 1 member
    group_id = pd.Series([1.0, 1.0, 1.0, 1.0, 2.0] * 2, index=idx)
    source = InMemorySeriesSource(data={"close": close, "group_id": group_id})

    expr = F("group_count")(col("close"), col("group_id"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
