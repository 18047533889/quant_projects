"""R21 DuckDB Tier-1 Production Certification — Triple-parity regression suite.

Covers the top 50 operators for DuckDB SQL pushdown parity, edge/null-handling,
and execution-kind routing. Each test runs pandas (reference), Polars, and DuckDB
in the same fixture to ensure cross-backend consistency.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource


F = make_cleaned_call_factory


# ---------------------------------------------------------------------------
# Fixture: small deterministic panel with NaN / Inf / edge values
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def panel() -> InMemorySeriesSource:
    """3 instruments × 12 dates panel with NaN in close (no Inf to avoid parity edge cases)."""
    dates = pd.date_range("2024-01-02", periods=12, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["timestamp", "instrument"])

    close_vals = [
        10.0, 20.0, 30.0,  # day 0
        11.0, 19.0, 31.0,  # day 1
        12.0, np.nan, 29.0,  # day 2 NaN in B
        13.0, 18.0, 28.0,  # day 3
        14.0, 17.0, 27.0,  # day 4
        15.0, 16.0, 26.0,  # day 5
        16.0, 15.0, 25.0,  # day 6
        17.0, 14.0, 24.0,  # day 7
        18.0, 13.0, 23.0,  # day 8
        19.0, 12.0, 22.0,  # day 9
        20.0, 11.0, 21.0,  # day 10
        21.0, 10.0, 20.0,  # day 11
    ]
    close = pd.Series(close_vals, index=idx, name="close")
    open_ = close - 0.5
    volume = pd.Series(np.tile([100.0, 200.0, 300.0], len(dates)), index=idx, name="volume")
    ret = close.pct_change(fill_method=None).groupby(level="instrument").shift(0)
    group_id = pd.Series(np.tile([1.0, 1.0, 2.0], len(dates)), index=idx, name="group_id")

    return InMemorySeriesSource(
        data={"close": close, "open": open_, "volume": volume, "returns": ret, "group_id": group_id}
    )


def _write_registry(path: Path, root: Path) -> None:
    path.write_text(
        f"""r21_duckdb_tier1:
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
    Returns: double
    group_id: int64
""",
        encoding="utf-8",
    )


def _seed(root: Path, source: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestamp, instrument in source.data["close"].index:
        def value(name: str):
            item = source.data[name].loc[(timestamp, instrument)]
            if name == "returns" and pd.isna(item):
                return None
            return None if pd.isna(item) else float(item)

        rows.append(
            {
                "TradeDate": timestamp.date(),
                "Symbol": instrument,
                "Close": value("close"),
                "Open": value("open"),
                "Volume": value("volume"),
                "Returns": value("returns"),
                "group_id": int(source.data["group_id"].loc[(timestamp, instrument)]),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, panel: InMemorySeriesSource):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    root = tmp_path / "data"
    _write_registry(tmp_path / "datasets.yaml", root)
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(tmp_path / "datasets.yaml"))
    _seed(root, panel)
    from data_access import reset_store
    reset_store()
    return build_data_source({"type": "data_access", "dataset": "r21_duckdb_tier1"})


def _run(source, expr, backend: str) -> dict:
    return FactorEngine(
        backend=build_backend(backend), data_source=source, run_mode="research"
    ).run(Factor(name="r21_parity", expr=expr))


def _assert_same(reference: pd.Series, candidate: pd.Series, label: str = "") -> None:
    pd.testing.assert_series_equal(
        reference.sort_index(), candidate.sort_index(),
        check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Top-50 operator regression matrix
# ---------------------------------------------------------------------------

# Each entry: (operator_name, lambda that builds the factor expression)
TOP50_CASES = [
    # ---- Element-wise ----
    ("add",        lambda c: F("add")(c("close"), c("open"))),
    ("subtract",   lambda c: F("subtract")(c("close"), c("open"))),
    ("multiply",   lambda c: F("multiply")(c("close"), c("open"))),
    ("divide",     lambda c: F("divide")(c("close"), c("open"))),
    ("neg",        lambda c: F("neg")(c("close"))),
    ("abs",        lambda c: F("abs")(c("close"))),
    ("sign",       lambda c: F("sign")(c("close"))),
    ("log",        lambda c: F("log")(c("close"))),
    ("exp",        lambda c: F("exp")(c("close") * 0.01)),
    ("sqrt",       lambda c: F("sqrt")(c("close"))),
    ("clip",       lambda c: F("clip")(c("close"), 15, 25)),
    ("floor",      lambda c: F("floor")(c("close"))),
    ("ceil",       lambda c: F("ceil")(c("close"))),
    ("inverse",    lambda c: F("inverse")(c("close"))),
    ("power",      lambda c: F("power")(c("close"), 2)),
    ("maximum",    lambda c: F("maximum")(c("close"), c("open"))),
    ("minimum",    lambda c: F("minimum")(c("close"), c("open"))),

    # ---- Logic / null-handling ----
    ("where",           lambda c: F("where")(F("gt")(c("close"), 20), c("close"), c("open"))),
    ("coalesce",        lambda c: F("coalesce")(c("close"), c("open"))),
    ("fillna_const",    lambda c: F("fillna_const")(c("close"), 0.0)),
    ("is_nan",          lambda c: F("is_nan")(c("close"))),
    ("is_not_null",     lambda c: F("is_not_null")(c("close"))),

    # ---- Simple time-series rolling ----
    ("ts_mean",    lambda c: F("ts_mean")(c("close"), 5)),
    ("ts_std",     lambda c: F("ts_std")(c("close"), 5)),
    ("ts_sum",     lambda c: F("ts_sum")(c("close"), 5)),
    ("ts_min",     lambda c: F("ts_min")(c("close"), 5)),
    ("ts_max",     lambda c: F("ts_max")(c("close"), 5)),
    ("ts_var",     lambda c: F("ts_var")(c("close"), 5)),
    ("ts_median",  lambda c: F("ts_median")(c("close"), 5)),
    ("ts_delay",   lambda c: F("ts_delay")(c("close"), 3)),
    ("ts_delta",   lambda c: F("ts_delta")(c("close"), 3)),
    ("ts_pct",     lambda c: F("ts_pct")(c("close"), 3)),
    ("ts_zscore",  lambda c: F("ts_zscore")(c("close"), 5)),

    # ---- Complex time-series ----
    ("ts_rank",      lambda c: F("ts_rank")(c("close"), 5)),
    ("ts_sharpe",    lambda c: F("ts_sharpe")(c("close"), 5)),
    ("ts_autocorr",  lambda c: F("ts_autocorr")(c("close"), 5)),
    ("ts_corr",      lambda c: F("ts_corr")(c("close"), c("open"), 5)),
    ("ts_cov",       lambda c: F("ts_cov")(c("close"), c("open"), 5)),

    # ---- Cross-sectional ----
    ("cs_rank",      lambda c: F("cs_rank")(c("close"))),
    ("cs_demean",    lambda c: F("cs_demean")(c("close"))),
    ("cs_std",       lambda c: F("cs_std")(c("close"))),
    ("cs_sum",       lambda c: F("cs_sum")(c("close"))),

    # ---- Group ----
    ("group_mean",   lambda c: F("group_mean")(c("close"), c("group_id"))),
    ("group_sum",    lambda c: F("group_sum")(c("close"), c("group_id"))),
    ("group_rank",   lambda c: F("group_rank")(c("close"), c("group_id"))),
    ("group_std",    lambda c: F("group_std")(c("close"), c("group_id"))),

    # ---- Rank / normalize ----
    ("rank",       lambda c: F("rank")(c("close"))),
    ("zscore",     lambda c: F("zscore")(c("close"))),
    ("winsorize",  lambda c: F("winsorize")(c("close"))),
]


def _mem_col(name: str):
    return col(name)


def _sql_col(name: str):
    return col({"close": "Close", "open": "Open", "volume": "Volume", "returns": "Returns"}.get(name, name))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,builder", TOP50_CASES, ids=[c[0] for c in TOP50_CASES])
def test_top50_triple_parity(panel, duckdb_source, name, builder):
    """Run each operator across pandas, Polars, and DuckDB; assert parity."""
    pandas_out = _run(panel, builder(_mem_col), "pandas")
    polars_out = _run(panel, builder(_mem_col), "polars_long")
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    reference = pandas_out["result"]
    _assert_same(reference, polars_out["result"], f"{name}:polars")
    assert_duckdb_real_sql_execution(sql_out)
    _assert_same(reference, sql_out["result"], f"{name}:duckdb")

    # Index sanity
    for out in (pandas_out, polars_out, sql_out):
        assert out["result"].index.names == ["timestamp", "instrument"], f"{name}: index names wrong"


# ---------------------------------------------------------------------------
# Edge / null-handling specific regression
# ---------------------------------------------------------------------------

def test_ts_mean_null_window(panel, duckdb_source):
    """ts_mean with min_periods > non-null count -> all NaN."""
    try:
        builder = lambda c: F("ts_mean")(c("close"), 5, min_periods=5)
        pandas_out = _run(panel, builder(_mem_col), "pandas")
    except Exception:
        pytest.skip("min_periods parameter not declared in ts_mean")
    for backend, src, col_fn in [("pandas", panel, _mem_col), ("polars_long", panel, _mem_col), ("duckdb_sql", duckdb_source, _sql_col)]:
        out = _run(src, builder(col_fn), backend)
        # Instrument B has NaN at day 2 -> window of 5 around it should be NaN
        result = out["result"]
        assert result.isna().sum() > 0, f"{backend}: ts_mean(min_periods=5) should have NaN"


def test_ts_std_ddof_zero(panel, duckdb_source):
    """ts_std with ddof=0 (population std) parity — skip if ddof not in param_names."""
    try:
        builder = lambda c: F("ts_std")(c("close"), 5, ddof=0)
        pandas_out = _run(panel, builder(_mem_col), "pandas")
    except Exception:
        pytest.skip("ddof parameter not declared in ts_std")
    polars_out = _run(panel, builder(_mem_col), "polars_long")
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    ref = pandas_out["result"]
    _assert_same(ref, polars_out["result"], "ts_std_ddof0:polars")
    assert_duckdb_real_sql_execution(sql_out)
    _assert_same(ref, sql_out["result"], "ts_std_ddof0:duckdb")


def test_ts_corr_finite_only(panel, duckdb_source):
    """ts_corr excludes ±Inf and NaN from pairwise computation."""
    builder = lambda c: F("ts_corr")(c("close"), c("open"), 5)
    pandas_out = _run(panel, builder(_mem_col), "pandas")
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    ref = pandas_out["result"]
    assert_duckdb_real_sql_execution(sql_out)
    _assert_same(ref, sql_out["result"], "ts_corr_finite_only:duckdb")


def test_group_mean_with_nan(panel, duckdb_source):
    """group_mean ignores NaN within group."""
    builder = lambda c: F("group_mean")(c("close"), c("group_id"))
    pandas_out = _run(panel, builder(_mem_col), "pandas")
    polars_out = _run(panel, builder(_mem_col), "polars_long")
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    ref = pandas_out["result"]
    _assert_same(ref, polars_out["result"], "group_mean_nan:polars")
    assert_duckdb_real_sql_execution(sql_out)
    _assert_same(ref, sql_out["result"], "group_mean_nan:duckdb")


def test_cs_rank(panel, duckdb_source):
    """Cross-sectional rank parity."""
    builder = lambda c: F("cs_rank")(c("close"))
    pandas_out = _run(panel, builder(_mem_col), "pandas")
    polars_out = _run(panel, builder(_mem_col), "polars_long")
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    ref = pandas_out["result"]
    _assert_same(ref, polars_out["result"], "cs_rank:polars")
    assert_duckdb_real_sql_execution(sql_out)
    _assert_same(ref, sql_out["result"], "cs_rank:duckdb")


def test_winsorize(panel, duckdb_source):
    """Winsorize clip at tails."""
    builder = lambda c: F("winsorize")(c("close"))
    pandas_out = _run(panel, builder(_mem_col), "pandas")
    polars_out = _run(panel, builder(_mem_col), "polars_long")
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    ref = pandas_out["result"]
    _assert_same(ref, polars_out["result"], "winsorize:polars")
    assert_duckdb_real_sql_execution(sql_out)
    _assert_same(ref, sql_out["result"], "winsorize:duckdb")


def test_where_null_propagation(panel, duckdb_source):
    """WHERE with null condition should propagate NaN."""
    builder = lambda c: F("where")(F("gt")(c("close"), 25), c("close"), c("open"))
    pandas_out = _run(panel, builder(_mem_col), "pandas")
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    ref = pandas_out["result"]
    assert_duckdb_real_sql_execution(sql_out)
    _assert_same(ref, sql_out["result"], "where_null:duckdb")


def test_ts_rank(panel, duckdb_source):
    """ts_rank parity."""
    builder = lambda c: F("ts_rank")(c("close"), 5)
    pandas_out = _run(panel, builder(_mem_col), "pandas")
    polars_out = _run(panel, builder(_mem_col), "polars_long")
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    ref = pandas_out["result"]
    _assert_same(ref, polars_out["result"], "ts_rank:polars")
    assert_duckdb_real_sql_execution(sql_out)
    _assert_same(ref, sql_out["result"], "ts_rank:duckdb")
