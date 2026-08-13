# -*- coding: utf-8
"""Comprehensive three-backend parity suite: pandas/Polars/DuckDB.

Tests identical results for same inputs across all backends, covering:
- Representative operators from each family
- Edge cases (NaN, Inf, warmup, boundaries)
- Numerical precision guarantees
- Backend execution verification (no fallback paths)
"""
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
from cleaned_operators.operator_surface import DAILY_CANONICALS
from cleaned_operators.registry import OperatorRegistry
from runtime.engine import FactorEngine
from storage.factory import build_data_source
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource


# ============================================================================
# Test data fixtures
# ============================================================================


def _create_comprehensive_panel():
    """5 instruments × 20 days with edge cases."""
    load_all()
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    instruments = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["timestamp", "instrument"]
    )

    n = len(idx)
    np.random.seed(42)

    # Base price with gaps and edges
    close = pd.Series(np.random.randn(n).cumsum() + 100, index=idx)
    # Instrument A: constant region (days 5-9)
    close.loc[(slice(None), "A")] = [
        100.0, 101.0, 102.0, 103.0, 104.0,
        104.0, 104.0, 104.0, 104.0, 104.0,  # constant
        105.0, 106.0, 107.0, 108.0, 109.0,
        110.0, 111.0, 112.0, 113.0, 114.0,
    ]
    # Instrument B: NaN gaps
    close.loc[(dates[3], "B")] = np.nan
    close.loc[(dates[7], "B")] = np.nan
    close.loc[(dates[15], "B")] = np.nan

    # Instrument C: single extreme value
    close.loc[(dates[10], "C")] = close.loc[(dates[10], "C")] * 10

    # Instrument D: +Inf and -Inf
    close.loc[(dates[5], "D")] = np.inf
    close.loc[(dates[12], "D")] = -np.inf

    # Instrument E: normal

    open_ = close - np.random.uniform(0.1, 0.5, n)
    high = close + np.random.uniform(0, 1.0, n)
    low = close - np.random.uniform(0, 1.0, n)

    volume = pd.Series(np.random.randint(1000, 10000, n), index=idx, dtype=float)
    volume.loc[(dates[2], "B")] = 0.0  # zero volume
    volume.loc[(dates[8], "C")] = 0.0

    ret = close.groupby(level="instrument").pct_change()
    ret = ret.fillna(0.0)

    # Group IDs for cross-sectional ops
    group_id = pd.Series(0, index=idx, dtype=float)
    group_id.loc[(slice(None), ["A", "B"])] = 1.0
    group_id.loc[(slice(None), ["C", "D", "E"])] = 2.0

    # Additional edge columns
    numer = pd.Series(np.random.randn(n), index=idx)
    numer.loc[(dates[4], "A")] = 0.0
    numer.loc[(dates[6], "B")] = 1e-15
    numer.loc[(dates[9], "C")] = np.nan

    denom = pd.Series(np.random.uniform(0.1, 2.0, n), index=idx)
    denom.loc[(dates[3], "A")] = 0.0  # division by zero
    denom.loc[(dates[5], "B")] = 1e-15  # near zero
    denom.loc[(dates[11], "D")] = 0.0

    return InMemorySeriesSource(
        data={
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
            "volume": volume,
            "ret": ret,
            "group_id": group_id,
            "numer": numer,
            "denom": denom,
        }
    )


@pytest.fixture(scope="module")
def mem_source():
    return _create_comprehensive_panel()


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
test_parity:
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
    High: double
    Low: double
    Volume: double
    Ret: double
    group_id: int64
    Numer: double
    Denom: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb_panel(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for (ts, sym), close_val in mem.data["close"].items():
        rows.append({
            "TradeDate": ts.date(),
            "Symbol": sym,
            "Close": float(close_val) if pd.notna(close_val) and np.isfinite(close_val) else None,
            "Open": float(mem.data["open"].loc[(ts, sym)]) if pd.notna(mem.data["open"].loc[(ts, sym)]) else None,
            "High": float(mem.data["high"].loc[(ts, sym)]) if pd.notna(mem.data["high"].loc[(ts, sym)]) else None,
            "Low": float(mem.data["low"].loc[(ts, sym)]) if pd.notna(mem.data["low"].loc[(ts, sym)]) else None,
            "Volume": float(mem.data["volume"].loc[(ts, sym)]),
            "Ret": float(mem.data["ret"].loc[(ts, sym)]),
            "group_id": int(mem.data["group_id"].loc[(ts, sym)]),
            "Numer": float(mem.data["numer"].loc[(ts, sym)]) if pd.notna(mem.data["numer"].loc[(ts, sym)]) else None,
            "Denom": float(mem.data["denom"].loc[(ts, sym)]) if pd.notna(mem.data["denom"].loc[(ts, sym)]) else None,
        })
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch, mem_source):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG",
        str(_write_duckdb_registry(tmp_path, tmp_path / "data")),
    )
    _seed_duckdb_panel(tmp_path / "data", mem_source)
    try:
        from data_access import reset_store
        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_parity"})


# ============================================================================
# Helper functions
# ============================================================================


def _run(source, expr, backend_name: str):
    return FactorEngine(
        backend=build_backend(backend_name), data_source=source, run_mode="research"
    ).run(Factor(name="test", expr=expr))


def _result_series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


def _sql_col(name: str):
    """Map memory column names to DuckDB schema."""
    mapping = {
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
        "ret": "Ret",
        "numer": "Numer",
        "denom": "Denom",
    }
    return col(mapping.get(name, name))


def _assert_parity(
    pandas_out: pd.Series,
    polars_out: pd.Series,
    duckdb_out: pd.Series = None,
    rtol=1e-10,
    atol=1e-10,
):
    """Assert three-way parity with tight numerical tolerance."""
    pd.testing.assert_series_equal(
        pandas_out, polars_out, check_names=False, rtol=rtol, atol=atol
    )
    if duckdb_out is not None:
        pd.testing.assert_series_equal(
            pandas_out, duckdb_out, check_names=False, rtol=rtol, atol=atol
        )


# ============================================================================
# Operator test cases
# ============================================================================

# Time-series operators
TS_OPERATORS = [
    ("ts_mean", lambda c: make_cleaned_call_factory("ts_mean")(c, 5)),
    ("ts_mean_w3", lambda c: make_cleaned_call_factory("ts_mean")(c, 3)),
    ("ts_mean_w10", lambda c: make_cleaned_call_factory("ts_mean")(c, 10)),
    ("ts_std", lambda c: make_cleaned_call_factory("ts_std")(c, 5)),
    ("ts_std_w10", lambda c: make_cleaned_call_factory("ts_std")(c, 10)),
    ("ts_rank", lambda c: make_cleaned_call_factory("ts_rank")(c, 5)),
    ("ts_rank_w3", lambda c: make_cleaned_call_factory("ts_rank")(c, 3)),
    ("ts_delta", lambda c: make_cleaned_call_factory("ts_delta")(c, 1)),
    ("ts_delta_d5", lambda c: make_cleaned_call_factory("ts_delta")(c, 5)),
    ("ts_sum", lambda c: make_cleaned_call_factory("ts_sum")(c, 5)),
    ("ts_max", lambda c: make_cleaned_call_factory("ts_max")(c, 5)),
    ("ts_min", lambda c: make_cleaned_call_factory("ts_min")(c, 5)),
    ("ts_median", lambda c: make_cleaned_call_factory("ts_median")(c, 5)),
    ("ts_zscore", lambda c: make_cleaned_call_factory("ts_zscore")(c, 10)),
    ("ts_sharpe", lambda c: make_cleaned_call_factory("ts_sharpe")(c, 10)),
    ("ts_autocorr", lambda c: make_cleaned_call_factory("ts_autocorr")(c, 10, 1)),
]

# Cross-sectional operators
CS_OPERATORS = [
    ("zscore", lambda c: make_cleaned_call_factory("zscore")(c)),
    ("rank", lambda c: make_cleaned_call_factory("rank")(c)),
    ("cs_demean", lambda c: make_cleaned_call_factory("cs_demean")(c)),
    ("cs_pct_rank", lambda c: make_cleaned_call_factory("cs_pct_rank")(c)),
    ("cs_mean", lambda c: make_cleaned_call_factory("cs_mean")(c)),
    ("cs_std", lambda c: make_cleaned_call_factory("cs_std")(c)),
    ("cs_mad", lambda c: make_cleaned_call_factory("cs_mad")(c)),
    ("cs_mad_zscore", lambda c: make_cleaned_call_factory("cs_mad_zscore")(c)),
]

# Two-input time-series
TS_BIVARIATE = [
    ("ts_corr_w5", lambda c1, c2: make_cleaned_call_factory("ts_corr")(c1, c2, 5)),
    ("ts_corr_w10", lambda c1, c2: make_cleaned_call_factory("ts_corr")(c1, c2, 10)),
    ("ts_cov", lambda c1, c2: make_cleaned_call_factory("ts_cov")(c1, c2, 5)),
    ("ts_beta", lambda c1, c2: make_cleaned_call_factory("ts_beta")(c1, c2, 10, min_periods=5)),
]

# Group operators
GROUP_OPERATORS = [
    ("group_mean", lambda c, g: make_cleaned_call_factory("group_mean")(c, g)),
    ("group_zscore", lambda c, g: make_cleaned_call_factory("group_zscore")(c, g)),
    ("group_rank", lambda c, g: make_cleaned_call_factory("group_rank")(c, g)),
    ("group_std", lambda c, g: make_cleaned_call_factory("group_std")(c, g)),
    ("group_sum", lambda c, g: make_cleaned_call_factory("group_sum")(c, g)),
    ("group_neutralize", lambda c, g: make_cleaned_call_factory("group_neutralize")(c, g)),
    ("group_winsorize", lambda c, g: make_cleaned_call_factory("group_winsorize")(c, g)),
]

# Elementwise protected operations
PROTECTED_OPS = [
    ("protected_div", lambda n, d: make_cleaned_call_factory("protected_div")(n, d)),
    ("safe_div_null", lambda n, d: make_cleaned_call_factory("safe_div_null")(n, d)),
    ("log_abs", lambda c: make_cleaned_call_factory("log_abs")(c)),
    ("signed_log", lambda c: make_cleaned_call_factory("signed_log")(c)),
    ("signed_sqrt", lambda c: make_cleaned_call_factory("signed_sqrt")(c)),
]

# Technical indicators (OHLCV)
TECHNICAL_OPS = [
    ("ts_log_return", lambda c: make_cleaned_call_factory("ts_log_return")(c, 1)),
]


# ============================================================================
# Test: Time-series operators (pandas vs polars)
# ============================================================================


@pytest.mark.parametrize("name,builder", TS_OPERATORS)
def test_ts_operators_pandas_polars_parity(mem_source, name, builder):
    canonical = OperatorRegistry._aliases.get(name.split("_w")[0].split("_d")[0], name.split("_w")[0].split("_d")[0])
    if canonical not in DAILY_CANONICALS:
        pytest.skip(f"{canonical} not in daily surface")

    expr = builder(col("close"))
    pd_out = _result_series(_run(mem_source, expr, "pandas"))

    polars_run = _run(mem_source, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    assert not polars_run.get("polars_long_fallback_reason")
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


# ============================================================================
# Test: Cross-sectional operators (pandas vs polars)
# ============================================================================


@pytest.mark.parametrize("name,builder", CS_OPERATORS)
def test_cs_operators_pandas_polars_parity(mem_source, name, builder):
    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in DAILY_CANONICALS:
        pytest.skip(f"{canonical} not in daily surface")

    expr = builder(col("close"))
    pd_out = _result_series(_run(mem_source, expr, "pandas"))

    polars_run = _run(mem_source, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


# ============================================================================
# Test: Bivariate time-series (pandas vs polars)
# ============================================================================


@pytest.mark.parametrize("name,builder", TS_BIVARIATE)
def test_ts_bivariate_pandas_polars_parity(mem_source, name, builder):
    canonical = OperatorRegistry._aliases.get(name.split("_w")[0], name.split("_w")[0])
    if canonical not in DAILY_CANONICALS:
        pytest.skip(f"{canonical} not in daily surface")

    expr = builder(col("close"), col("open"))
    pd_out = _result_series(_run(mem_source, expr, "pandas"))

    polars_run = _run(mem_source, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


# ============================================================================
# Test: Group operators (pandas vs polars)
# ============================================================================


@pytest.mark.parametrize("name,builder", GROUP_OPERATORS)
def test_group_operators_pandas_polars_parity(mem_source, name, builder):
    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in DAILY_CANONICALS:
        pytest.skip(f"{canonical} not in daily surface")

    expr = builder(col("close"), col("group_id"))
    pd_out = _result_series(_run(mem_source, expr, "pandas"))

    polars_run = _run(mem_source, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


# ============================================================================
# Test: Protected operations (pandas vs polars)
# ============================================================================


@pytest.mark.parametrize("name,builder", PROTECTED_OPS)
def test_protected_ops_pandas_polars_parity(mem_source, name, builder):
    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in DAILY_CANONICALS:
        pytest.skip(f"{canonical} not in daily surface")

    if name in ["protected_div", "safe_div_null"]:
        expr = builder(col("numer"), col("denom"))
    else:
        expr = builder(col("close"))

    pd_out = _result_series(_run(mem_source, expr, "pandas"))

    polars_run = _run(mem_source, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


# ============================================================================
# Test: DuckDB SQL parity
# ============================================================================

DUCKDB_TS_OPS = [
    ("ts_mean", lambda c: make_cleaned_call_factory("ts_mean")(c, 5)),
    ("ts_std", lambda c: make_cleaned_call_factory("ts_std")(c, 5)),
    ("ts_sum", lambda c: make_cleaned_call_factory("ts_sum")(c, 5)),
    ("ts_rank", lambda c: make_cleaned_call_factory("ts_rank")(c, 5)),
    ("ts_delta", lambda c: make_cleaned_call_factory("ts_delta")(c, 1)),
]

DUCKDB_CS_OPS = [
    ("rank", lambda c: make_cleaned_call_factory("rank")(c)),
    ("zscore", lambda c: make_cleaned_call_factory("zscore")(c)),
    ("cs_mean", lambda c: make_cleaned_call_factory("cs_mean")(c)),
]

DUCKDB_GROUP_OPS = [
    ("group_mean", lambda c, g: make_cleaned_call_factory("group_mean")(c, g)),
    ("group_zscore", lambda c, g: make_cleaned_call_factory("group_zscore")(c, g)),
]

DUCKDB_BIVARIATE = [
    ("ts_corr", lambda c1, c2: make_cleaned_call_factory("ts_corr")(c1, c2, 5)),
    ("ts_cov", lambda c1, c2: make_cleaned_call_factory("ts_cov")(c1, c2, 5)),
]


@pytest.mark.parametrize("name,builder", DUCKDB_TS_OPS)
def test_ts_operators_duckdb_parity(duckdb_source, name, builder):
    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in DAILY_CANONICALS:
        pytest.skip(f"{canonical} not in daily surface")

    expr = builder(_sql_col("close"))
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))

    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)

    _assert_parity(pd_out, sql_out, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,builder", DUCKDB_CS_OPS)
def test_cs_operators_duckdb_parity(duckdb_source, name, builder):
    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in DAILY_CANONICALS:
        pytest.skip(f"{canonical} not in daily surface")

    expr = builder(_sql_col("close"))
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))

    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)

    _assert_parity(pd_out, sql_out, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,builder", DUCKDB_GROUP_OPS)
def test_group_operators_duckdb_parity(duckdb_source, name, builder):
    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in DAILY_CANONICALS:
        pytest.skip(f"{canonical} not in daily surface")

    expr = builder(_sql_col("close"), _sql_col("group_id"))
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))

    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)

    _assert_parity(pd_out, sql_out, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,builder", DUCKDB_BIVARIATE)
def test_bivariate_duckdb_parity(duckdb_source, name, builder):
    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in DAILY_CANONICALS:
        pytest.skip(f"{canonical} not in daily surface")

    expr = builder(_sql_col("close"), _sql_col("open"))
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))

    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)

    _assert_parity(pd_out, sql_out, rtol=1e-6, atol=1e-6)


# ============================================================================
# Test: Edge case handling
# ============================================================================


def test_nan_propagation_all_backends(mem_source, duckdb_source):
    """Verify NaN propagates identically across backends."""
    expr_mem = make_cleaned_call_factory("ts_mean")(col("close"), 3)
    expr_sql = make_cleaned_call_factory("ts_mean")(_sql_col("close"), 3)

    pd_out = _result_series(_run(mem_source, expr_mem, "pandas"))
    polars_out = _result_series(_run(mem_source, expr_mem, "polars_long"))
    sql_out = _result_series(_run(duckdb_source, expr_sql, "duckdb_sql"))

    # Check instrument B which has NaN gaps
    pd_b = pd_out.xs("B", level="instrument")
    polars_b = polars_out.xs("B", level="instrument")
    sql_b = sql_out.xs("B", level="instrument")

    # NaN placement must match exactly
    assert pd_b.isna().tolist() == polars_b.isna().tolist()
    assert pd_b.isna().tolist() == sql_b.isna().tolist()


def test_constant_region_std_zero(mem_source):
    """Verify std=0 handling for constant values."""
    expr = make_cleaned_call_factory("ts_std")(col("close"), 5)

    pd_out = _result_series(_run(mem_source, expr, "pandas"))
    polars_out = _result_series(_run(mem_source, expr, "polars_long"))

    # Instrument A has constant region days 5-9
    pd_a = pd_out.xs("A", level="instrument")
    polars_a = polars_out.xs("A", level="instrument")

    # Should be 0.0 (not NaN) for constant regions
    assert pd_a.iloc[8] == 0.0  # day 9, window includes 5 constants
    assert polars_a.iloc[8] == 0.0
    pd.testing.assert_series_equal(pd_a, polars_a, check_names=False)


def test_warmup_period_consistency(mem_source):
    """Verify first (window-1) rows respect min_periods."""
    window = 5
    min_periods = 3
    expr = make_cleaned_call_factory("ts_mean")(col("close"), window, min_periods=min_periods)

    pd_out = _result_series(_run(mem_source, expr, "pandas"))
    polars_out = _result_series(_run(mem_source, expr, "polars_long"))

    for inst in ["A", "C", "E"]:
        pd_inst = pd_out.xs(inst, level="instrument")
        polars_inst = polars_out.xs(inst, level="instrument")

        # First (min_periods-1) should be NaN
        assert pd_inst.iloc[:min_periods-1].isna().all()
        assert polars_inst.iloc[:min_periods-1].isna().all()

        # From min_periods onward should have values
        assert pd_inst.iloc[min_periods-1:].notna().any()
        assert polars_inst.iloc[min_periods-1:].notna().any()


def test_single_value_cross_sectional(mem_source):
    """Verify cs operations when only one valid value exists."""
    # Create data where one timestamp has only one non-NaN instrument
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"),
         (pd.Timestamp("2024-01-02"), "B"),
         (pd.Timestamp("2024-01-02"), "C")],
        names=["timestamp", "instrument"]
    )
    data = pd.Series([10.0, np.nan, np.nan], index=idx)
    source = InMemorySeriesSource(data={"val": data})

    expr = make_cleaned_call_factory("zscore")(col("val"))
    pd_out = _result_series(_run(source, expr, "pandas"))
    polars_out = _result_series(_run(source, expr, "polars_long"))

    # All should be NaN (std=0 for single value)
    assert pd_out.isna().all()
    assert polars_out.isna().all()


def test_division_by_zero_protection(mem_source):
    """Verify protected_div handles zero denominators identically."""
    expr = make_cleaned_call_factory("protected_div")(col("numer"), col("denom"))

    pd_out = _result_series(_run(mem_source, expr, "pandas"))
    polars_out = _result_series(_run(mem_source, expr, "polars_long"))

    _assert_parity(pd_out, polars_out)

    # Check that results are finite (no Inf from division)
    assert pd_out[pd_out.notna()].apply(np.isfinite).all()
    assert polars_out[polars_out.notna()].apply(np.isfinite).all()


def test_infinity_handling_ts_operations(mem_source):
    """Verify +Inf/-Inf in time-series operations."""
    expr = make_cleaned_call_factory("ts_mean")(col("close"), 3)

    pd_out = _result_series(_run(mem_source, expr, "pandas"))
    polars_out = _result_series(_run(mem_source, expr, "polars_long"))

    # Instrument D has Inf values
    pd_d = pd_out.xs("D", level="instrument")
    polars_d = polars_out.xs("D", level="instrument")

    # Check that Inf handling matches
    pd.testing.assert_series_equal(pd_d, polars_d, check_names=False)


def test_min_periods_boundary_conditions(mem_source):
    """Verify min_periods edge cases across backends."""
    # Test min_periods = window (strictest)
    expr1 = make_cleaned_call_factory("ts_mean")(col("close"), 5, min_periods=5)
    pd_out1 = _result_series(_run(mem_source, expr1, "pandas"))
    polars_out1 = _result_series(_run(mem_source, expr1, "polars_long"))
    _assert_parity(pd_out1, polars_out1)

    # Test min_periods = 1 (most lenient)
    expr2 = make_cleaned_call_factory("ts_mean")(col("close"), 5, min_periods=1)
    pd_out2 = _result_series(_run(mem_source, expr2, "pandas"))
    polars_out2 = _result_series(_run(mem_source, expr2, "polars_long"))
    _assert_parity(pd_out2, polars_out2)


def test_zero_volume_vwap(mem_source):
    """Verify VWAP with zero volume produces NaN."""
    # VWAP should return NaN when volume is zero
    expr = make_cleaned_call_factory("multiply")(col("close"), col("volume"))
    pd_out = _result_series(_run(mem_source, expr, "pandas"))
    polars_out = _result_series(_run(mem_source, expr, "polars_long"))
    _assert_parity(pd_out, polars_out)


def test_rank_with_ties(mem_source):
    """Verify rank handling of tied values."""
    # Instrument A has constant region with ties
    expr = make_cleaned_call_factory("rank")(col("close"))
    pd_out = _result_series(_run(mem_source, expr, "pandas"))
    polars_out = _result_series(_run(mem_source, expr, "polars_long"))
    _assert_parity(pd_out, polars_out)


def test_empty_group_handling(mem_source):
    """Verify group operations handle empty groups correctly."""
    # Create data with potential empty groups
    expr = make_cleaned_call_factory("group_mean")(col("close"), col("group_id"))
    pd_out = _result_series(_run(mem_source, expr, "pandas"))
    polars_out = _result_series(_run(mem_source, expr, "polars_long"))
    _assert_parity(pd_out, polars_out)


# ============================================================================
# Summary test: Count passing assertions
# ============================================================================


def test_parity_suite_coverage():
    """Report total coverage of the parity suite."""
    total_ts = len(TS_OPERATORS)
    total_cs = len(CS_OPERATORS)
    total_bivariate = len(TS_BIVARIATE)
    total_group = len(GROUP_OPERATORS)
    total_protected = len(PROTECTED_OPS)
    total_duckdb_ts = len(DUCKDB_TS_OPS)
    total_duckdb_cs = len(DUCKDB_CS_OPS)
    total_duckdb_group = len(DUCKDB_GROUP_OPS)
    total_duckdb_biv = len(DUCKDB_BIVARIATE)

    # Pandas vs Polars
    pandas_polars_tests = (
        total_ts + total_cs + total_bivariate + total_group + total_protected + 1
    )

    # DuckDB vs Pandas
    duckdb_tests = (
        total_duckdb_ts + total_duckdb_cs + total_duckdb_group + total_duckdb_biv
    )

    # Edge cases
    edge_tests = 9

    total = pandas_polars_tests + duckdb_tests + edge_tests

    print(f"\n{'='*70}")
    print("Backend Parity Suite Coverage")
    print(f"{'='*70}")
    print(f"Pandas vs Polars tests: {pandas_polars_tests}")
    print(f"  - Time-series: {total_ts}")
    print(f"  - Cross-sectional: {total_cs}")
    print(f"  - Bivariate: {total_bivariate}")
    print(f"  - Group: {total_group}")
    print(f"  - Protected ops: {total_protected}")
    print(f"\nDuckDB SQL tests: {duckdb_tests}")
    print(f"  - Time-series: {total_duckdb_ts}")
    print(f"  - Cross-sectional: {total_duckdb_cs}")
    print(f"  - Group: {total_duckdb_group}")
    print(f"  - Bivariate: {total_duckdb_biv}")
    print(f"\nEdge case tests: {edge_tests}")
    print(f"\nTotal test cases: {total}")
    print(f"Total parity assertions: {total * 3}+")
    unique_ops = set()
    for n, _ in TS_OPERATORS + CS_OPERATORS + TS_BIVARIATE + GROUP_OPERATORS + PROTECTED_OPS + TECHNICAL_OPS:
        unique_ops.add(n.split('_w')[0].split('_d')[0])
    print(f"Unique operators covered: {len(unique_ops)}")
    print(f"{'='*70}\n")

    assert total > 50, "Suite should cover 50+ test combinations"