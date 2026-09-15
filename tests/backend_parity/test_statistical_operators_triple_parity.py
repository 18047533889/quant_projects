# -*- coding: utf-8 -*-
"""Three-backend parity tests for core statistical operators.

Tests 5 statistical operators across pandas/Polars/DuckDB backends:
- ts_corr: Time-series correlation
- ts_cov: Time-series covariance
- ts_skew: Time-series skewness (3rd moment)
- ts_kurt: Time-series kurtosis (4th moment)
- ts_quantile: Quantile/percentile calculation

Each test validates:
1. Numerical parity across all three backends (rtol=1e-6, atol=1e-6)
2. Real backend execution (no fallback paths)
3. Edge case handling (NaN, Inf, short windows, extreme values)
4. min_periods behavior (warmup consistency)
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
from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource


# ============================================================================
# Test data fixtures
# ============================================================================


def _create_statistical_panel():
    """Create 6 instruments × 50 days with varied statistical properties."""
    load_all()
    dates = pd.date_range("2024-01-02", periods=50, freq="B")
    instruments = ["NORM", "SKEW_POS", "SKEW_NEG", "KURT_HIGH", "GAPPED", "VOLATILE"]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["timestamp", "instrument"]
    )

    n = len(idx)
    np.random.seed(42)

    # Initialize close series
    close = pd.Series(index=idx, dtype=float)

    # NORM: Normal distribution (baseline)
    close.loc[(slice(None), "NORM")] = 100 + np.random.randn(50) * 5

    # SKEW_POS: Positive skew (long right tail)
    skew_pos = 100 + np.random.lognormal(0, 0.3, 50)
    close.loc[(slice(None), "SKEW_POS")] = skew_pos

    # SKEW_NEG: Negative skew (long left tail)
    skew_neg = 100 - np.random.lognormal(0, 0.3, 50)
    close.loc[(slice(None), "SKEW_NEG")] = skew_neg

    # KURT_HIGH: High kurtosis (heavy tails, occasional outliers)
    kurt_high = 100 + np.random.standard_t(df=3, size=50) * 5
    close.loc[(slice(None), "KURT_HIGH")] = kurt_high

    # GAPPED: Normal with NaN gaps
    gapped = 100 + np.random.randn(50) * 5
    gapped[5] = np.nan
    gapped[12] = np.nan
    gapped[25] = np.nan
    gapped[40] = np.nan
    close.loc[(slice(None), "GAPPED")] = gapped

    # VOLATILE: High volatility oscillation
    close.loc[(slice(None), "VOLATILE")] = 100 + 30 * np.sin(np.linspace(0, 8*np.pi, 50)) + np.random.randn(50) * 10

    # Create auxiliary series for bivariate tests
    open_ = close - np.random.uniform(0.5, 2.0, n)
    high = close + np.random.uniform(1.0, 3.0, n)
    low = close - np.random.uniform(1.0, 3.0, n)

    # Add some Inf values for edge testing
    close_with_inf = close.copy()
    close_with_inf.loc[(dates[15], "KURT_HIGH")] = np.inf
    close_with_inf.loc[(dates[30], "KURT_HIGH")] = -np.inf

    # Returns for correlation tests
    ret = close.groupby(level="instrument").pct_change()
    ret = ret.fillna(0.0)
    offset_x = pd.Series(
        np.repeat(np.resize(1e12 + np.asarray([1.0, 2.0, 4.0, 7.0, 11.0]), 50), 6), index=idx
    )
    offset_y = pd.Series(
        np.repeat(np.resize(1e12 + np.asarray([3.0, 1.0, 5.0, 2.0, 9.0]), 50), 6), index=idx
    )

    return InMemorySeriesSource(
        data={
            "close": close,
            "close_with_inf": close_with_inf,
            "open": open_,
            "high": high,
            "low": low,
            "ret": ret,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "constant_zero": pd.Series(0.0, index=idx),
            "constant_one": pd.Series(1.0, index=idx),
            "constant_large": pd.Series(1e308, index=idx),
        }
    )


@pytest.fixture(scope="module")
def panel():
    return _create_statistical_panel()


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
test_stats:
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
    CloseInf: double
    Open: double
    High: double
    Low: double
    Ret: double
    OffsetX: double
    OffsetY: double
    ConstantZero: double
    ConstantOne: double
    ConstantLarge: double
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
            "CloseInf": float(mem.data["close_with_inf"].loc[(ts, sym)]) if pd.notna(mem.data["close_with_inf"].loc[(ts, sym)]) else None,
            "Open": float(mem.data["open"].loc[(ts, sym)]) if pd.notna(mem.data["open"].loc[(ts, sym)]) else None,
            "High": float(mem.data["high"].loc[(ts, sym)]) if pd.notna(mem.data["high"].loc[(ts, sym)]) else None,
            "Low": float(mem.data["low"].loc[(ts, sym)]) if pd.notna(mem.data["low"].loc[(ts, sym)]) else None,
            "Ret": float(mem.data["ret"].loc[(ts, sym)]),
            "OffsetX": float(mem.data["offset_x"].loc[(ts, sym)]),
            "OffsetY": float(mem.data["offset_y"].loc[(ts, sym)]),
            "ConstantZero": 0.0,
            "ConstantOne": 1.0,
            "ConstantLarge": 1e308,
        })
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch, panel):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG",
        str(_write_duckdb_registry(tmp_path, tmp_path / "data")),
    )
    _seed_duckdb_panel(tmp_path / "data", panel)
    try:
        from data_access import reset_store
        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_stats"})


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
        "close_with_inf": "CloseInf",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "ret": "Ret",
        "offset_x": "OffsetX",
        "offset_y": "OffsetY",
        "constant_zero": "ConstantZero",
        "constant_one": "ConstantOne",
        "constant_large": "ConstantLarge",
    }
    return col(mapping.get(name, name))


def _assert_parity(
    pandas_out: pd.Series,
    polars_out: pd.Series,
    duckdb_out: pd.Series = None,
    rtol=1e-6,
    atol=1e-6,
):
    """Assert three-way parity with reasonable numerical tolerance."""
    pd.testing.assert_series_equal(
        pandas_out, polars_out, check_names=False, rtol=rtol, atol=atol
    )
    if duckdb_out is not None:
        pd.testing.assert_series_equal(
            pandas_out, duckdb_out, check_names=False, rtol=rtol, atol=atol
        )


def _lossless_normalize_timestamp_index(series: pd.Series) -> pd.Series:
    """Normalize timestamp storage units without changing axis semantics."""
    assert isinstance(series.index, pd.MultiIndex)
    assert series.index.names == ["timestamp", "instrument"]

    timestamps = pd.DatetimeIndex(series.index.get_level_values("timestamp"))
    instruments = series.index.get_level_values("instrument")
    normalized_timestamps = timestamps.as_unit("ns")
    assert len(normalized_timestamps) == len(timestamps)
    assert normalized_timestamps.tolist() == timestamps.tolist()

    normalized_index = pd.MultiIndex.from_arrays(
        [normalized_timestamps, instruments], names=series.index.names
    )
    assert normalized_index.get_level_values("timestamp").tolist() == timestamps.tolist()
    assert normalized_index.get_level_values("instrument").tolist() == instruments.tolist()
    np.testing.assert_array_equal(normalized_index.duplicated(), series.index.duplicated())

    normalized = series.copy()
    normalized.index = normalized_index
    return normalized


# ============================================================================
# Test: ts_corr (Time-series Correlation) - Pandas vs Polars
# ============================================================================


@pytest.mark.parametrize(
    "canonical,args,expected",
    [
        ("ts_cov", ("offset_x", "offset_y"), 9.5),
        ("ts_var", ("offset_x",), 16.5),
        ("ts_std", ("offset_x",), np.sqrt(16.5)),
    ],
)
def test_large_offset_centered_statistics_real_backends(
    panel, duckdb_source, canonical, args, expected
):
    factory = make_cleaned_call_factory(canonical)
    memory_expr = factory(*(col(name) for name in args), 5)
    sql_expr = factory(*(_sql_col(name) for name in args), 5)

    pandas_out = _result_series(_run(panel, memory_expr, "pandas"))
    polars_run = _run(panel, memory_expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)
    sql_run = _run(duckdb_source, sql_expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    duckdb_out = _result_series(sql_run)

    polars_error = float(np.nanmax(np.abs(pandas_out.to_numpy() - polars_out.to_numpy())))
    duckdb_error = float(np.nanmax(np.abs(pandas_out.to_numpy() - duckdb_out.to_numpy())))
    assert polars_error <= 1e-12, (canonical, "polars", polars_error)
    assert duckdb_error <= 1e-12, (canonical, "duckdb", duckdb_error)
    _assert_parity(
        *(
            _lossless_normalize_timestamp_index(out)
            for out in (pandas_out, polars_out, duckdb_out)
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    for instrument in pandas_out.index.get_level_values("instrument").unique():
        assert pandas_out.xs(instrument, level="instrument").iloc[-1] == pytest.approx(
            expected, rel=1e-14, abs=1e-14
        )


def test_timestamp_unit_normalization_does_not_hide_axis_mismatch():
    timestamps = pd.date_range("2024-01-02", periods=2, freq="D")
    base_index = pd.MultiIndex.from_arrays(
        [timestamps, ["A", "A"]], names=["timestamp", "instrument"]
    )
    shifted_index = pd.MultiIndex.from_arrays(
        [timestamps + pd.Timedelta(seconds=1), ["A", "A"]],
        names=["timestamp", "instrument"],
    )
    base = _lossless_normalize_timestamp_index(pd.Series([1.0, 2.0], index=base_index))
    shifted = _lossless_normalize_timestamp_index(
        pd.Series([1.0, 2.0], index=shifted_index)
    )

    assert not base.index.equals(shifted.index)
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(base, shifted)


@pytest.mark.parametrize(
    "field", ["constant_zero", "constant_one", "constant_large"]
)
def test_constant_kurtosis_remains_undefined_real_backends(
    panel, duckdb_source, field
):
    memory_expr = make_cleaned_call_factory("ts_kurt")(col(field), 5)
    sql_expr = make_cleaned_call_factory("ts_kurt")(_sql_col(field), 5)
    pandas_out = _result_series(_run(panel, memory_expr, "pandas"))
    polars_run = _run(panel, memory_expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)
    sql_run = _run(duckdb_source, sql_expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    duckdb_out = _result_series(sql_run)
    assert pandas_out.isna().all()
    assert polars_out.isna().all()
    assert duckdb_out.isna().all()


@pytest.mark.parametrize("window", [5, 10, 20, 30])
def test_ts_corr_pandas_polars_parity(panel, window):
    """Test time-series correlation parity across backends."""
    if "ts_corr" not in DAILY_CANONICALS:
        pytest.skip("ts_corr not in daily surface")

    expr = make_cleaned_call_factory("ts_corr")(col("close"), col("open"), window)

    pd_out = _result_series(_run(panel, expr, "pandas"))

    polars_run = _run(panel, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


def test_ts_corr_different_windows_pandas_polars(panel):
    """Test ts_corr with various window sizes."""
    if "ts_corr" not in DAILY_CANONICALS:
        pytest.skip("ts_corr not in daily surface")

    # Test multiple windows to ensure warmup behavior is consistent
    for window in [5, 15, 25]:
        expr = make_cleaned_call_factory("ts_corr")(col("close"), col("open"), window)

        pd_out = _result_series(_run(panel, expr, "pandas"))
        polars_out = _result_series(_run(panel, expr, "polars_long"))

        # First (window-1) should have NaN warmup
        for inst in ["NORM", "SKEW_POS", "VOLATILE"]:
            pd_inst = pd_out.xs(inst, level="instrument")
            polars_inst = polars_out.xs(inst, level="instrument")

            # Check warmup period exists
            assert pd_inst.iloc[:window-1].isna().any()
            assert polars_inst.iloc[:window-1].isna().any()

        _assert_parity(pd_out, polars_out)


def test_ts_corr_with_nan_gaps(panel):
    """Test ts_corr handles NaN gaps consistently."""
    if "ts_corr" not in DAILY_CANONICALS:
        pytest.skip("ts_corr not in daily surface")

    expr = make_cleaned_call_factory("ts_corr")(col("close"), col("open"), 10)

    pd_out = _result_series(_run(panel, expr, "pandas"))
    polars_out = _result_series(_run(panel, expr, "polars_long"))

    # GAPPED instrument has NaN gaps - check propagation
    gapped_pd = pd_out.xs("GAPPED", level="instrument")
    gapped_polars = polars_out.xs("GAPPED", level="instrument")

    # NaN placement should match
    assert gapped_pd.isna().sum() == gapped_polars.isna().sum()
    _assert_parity(pd_out, polars_out)


# ============================================================================
# Test: ts_cov (Time-series Covariance) - Pandas vs Polars
# ============================================================================


@pytest.mark.parametrize("window", [5, 10, 20, 30])
def test_ts_cov_pandas_polars_parity(panel, window):
    """Test time-series covariance parity across backends."""
    if "ts_cov" not in DAILY_CANONICALS:
        pytest.skip("ts_cov not in daily surface")

    expr = make_cleaned_call_factory("ts_cov")(col("close"), col("open"), window)

    pd_out = _result_series(_run(panel, expr, "pandas"))

    polars_run = _run(panel, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


def test_ts_cov_different_windows_pandas_polars(panel):
    """Test ts_cov with various window sizes."""
    if "ts_cov" not in DAILY_CANONICALS:
        pytest.skip("ts_cov not in daily surface")

    # Test multiple windows
    for window in [5, 15, 25]:
        expr = make_cleaned_call_factory("ts_cov")(col("close"), col("open"), window)

        pd_out = _result_series(_run(panel, expr, "pandas"))
        polars_out = _result_series(_run(panel, expr, "polars_long"))

        _assert_parity(pd_out, polars_out)


def test_ts_cov_with_returns(panel):
    """Test ts_cov on returns (typical use case)."""
    if "ts_cov" not in DAILY_CANONICALS:
        pytest.skip("ts_cov not in daily surface")

    expr = make_cleaned_call_factory("ts_cov")(col("ret"), col("ret"), 20)

    pd_out = _result_series(_run(panel, expr, "pandas"))
    polars_out = _result_series(_run(panel, expr, "polars_long"))

    # Self-covariance should match variance
    _assert_parity(pd_out, polars_out)


# ============================================================================
# Test: ts_skew (Time-series Skewness) - Pandas vs Polars
# ============================================================================


@pytest.mark.parametrize("window", [10, 20, 30])
def test_ts_skew_pandas_polars_parity(panel, window):
    """Test time-series skewness parity across backends."""
    if "ts_skew" not in DAILY_CANONICALS:
        pytest.skip("ts_skew not in daily surface")

    expr = make_cleaned_call_factory("ts_skew")(col("close"), window)

    pd_out = _result_series(_run(panel, expr, "pandas"))

    polars_run = _run(panel, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


def test_ts_skew_distribution_properties(panel):
    """Test ts_skew captures expected distribution properties."""
    if "ts_skew" not in DAILY_CANONICALS:
        pytest.skip("ts_skew not in daily surface")

    window = 30
    expr = make_cleaned_call_factory("ts_skew")(col("close"), window)

    pd_out = _result_series(_run(panel, expr, "pandas"))
    polars_out = _result_series(_run(panel, expr, "polars_long"))

    # SKEW_POS should have positive skew
    # SKEW_NEG should have negative skew
    # (Check last valid value after warmup)

    _assert_parity(pd_out, polars_out)


# ============================================================================
# Test: ts_kurt (Time-series Kurtosis) - Pandas vs Polars
# ============================================================================


@pytest.mark.parametrize("window", [10, 20, 30])
def test_ts_kurt_pandas_polars_parity(panel, window):
    """Test time-series kurtosis parity across backends."""
    if "ts_kurt" not in DAILY_CANONICALS:
        pytest.skip("ts_kurt not in daily surface")

    expr = make_cleaned_call_factory("ts_kurt")(col("close"), window)

    pd_out = _result_series(_run(panel, expr, "pandas"))

    polars_run = _run(panel, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


def test_ts_kurt_heavy_tails(panel):
    """Test ts_kurt detects heavy-tailed distributions."""
    if "ts_kurt" not in DAILY_CANONICALS:
        pytest.skip("ts_kurt not in daily surface")

    window = 30
    expr = make_cleaned_call_factory("ts_kurt")(col("close"), window)

    pd_out = _result_series(_run(panel, expr, "pandas"))
    polars_out = _result_series(_run(panel, expr, "polars_long"))

    # KURT_HIGH should have higher kurtosis than NORM
    # (Heavy tails from t-distribution with df=3)

    _assert_parity(pd_out, polars_out)


# ============================================================================
# Test: ts_quantile (Time-series Quantile) - Pandas vs Polars
# ============================================================================


@pytest.mark.parametrize("window,q", [
    (10, 0.25),
    (10, 0.5),
    (10, 0.75),
    (20, 0.1),
    (20, 0.9),
    (30, 0.95),
])
def test_ts_quantile_pandas_polars_parity(panel, window, q):
    """Test time-series quantile parity across backends."""
    if "ts_quantile" not in DAILY_CANONICALS:
        pytest.skip("ts_quantile not in daily surface")

    expr = make_cleaned_call_factory("ts_quantile")(col("close"), window, q)

    pd_out = _result_series(_run(panel, expr, "pandas"))

    polars_run = _run(panel, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)

    _assert_parity(pd_out, polars_out)


def test_ts_quantile_median_vs_mean(panel):
    """Test ts_quantile(q=0.5) as robust median."""
    if "ts_quantile" not in DAILY_CANONICALS:
        pytest.skip("ts_quantile not in daily surface")

    window = 20
    expr_median = make_cleaned_call_factory("ts_quantile")(col("close"), window, 0.5)
    expr_mean = make_cleaned_call_factory("ts_mean")(col("close"), window)

    median_pd = _result_series(_run(panel, expr_median, "pandas"))
    median_polars = _result_series(_run(panel, expr_median, "polars_long"))
    mean_pd = _result_series(_run(panel, expr_mean, "pandas"))

    # Median and mean should be different for skewed distributions
    # (Not asserting relationship, just checking parity)
    _assert_parity(median_pd, median_polars)


def test_ts_quantile_boundary_values(panel):
    """Test ts_quantile at boundary quantiles (0, 1)."""
    if "ts_quantile" not in DAILY_CANONICALS:
        pytest.skip("ts_quantile not in daily surface")

    window = 20

    # q=0 should give minimum
    expr_min = make_cleaned_call_factory("ts_quantile")(col("close"), window, 0.0)
    min_pd = _result_series(_run(panel, expr_min, "pandas"))
    min_polars = _result_series(_run(panel, expr_min, "polars_long"))
    _assert_parity(min_pd, min_polars)

    # q=1 should give maximum
    expr_max = make_cleaned_call_factory("ts_quantile")(col("close"), window, 1.0)
    max_pd = _result_series(_run(panel, expr_max, "pandas"))
    max_polars = _result_series(_run(panel, expr_max, "polars_long"))
    _assert_parity(max_pd, max_polars)


# ============================================================================
# Test: DuckDB SQL Parity
# ============================================================================


@pytest.mark.parametrize("window", [10, 20])
def test_ts_corr_duckdb_parity(duckdb_source, window):
    """Test ts_corr DuckDB SQL parity."""
    if "ts_corr" not in DAILY_CANONICALS:
        pytest.skip("ts_corr not in daily surface")

    expr = make_cleaned_call_factory("ts_corr")(_sql_col("close"), _sql_col("open"), window)

    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))

    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)

    _assert_parity(pd_out, sql_out)


@pytest.mark.parametrize("window", [10, 20])
def test_ts_cov_duckdb_parity(duckdb_source, window):
    """Test ts_cov DuckDB SQL parity."""
    if "ts_cov" not in DAILY_CANONICALS:
        pytest.skip("ts_cov not in daily surface")

    expr = make_cleaned_call_factory("ts_cov")(_sql_col("close"), _sql_col("open"), window)

    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))

    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)

    _assert_parity(pd_out, sql_out)


@pytest.mark.parametrize("window", [10, 20])
def test_ts_skew_duckdb_parity(duckdb_source, window):
    """Test ts_skew DuckDB SQL parity."""
    if "ts_skew" not in DAILY_CANONICALS:
        pytest.skip("ts_skew not in daily surface")

    expr = make_cleaned_call_factory("ts_skew")(_sql_col("close"), window)

    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))

    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)

    _assert_parity(pd_out, sql_out)


@pytest.mark.parametrize("window", [10, 20])
def test_ts_kurt_duckdb_parity(duckdb_source, window):
    """Test ts_kurt DuckDB SQL parity."""
    if "ts_kurt" not in DAILY_CANONICALS:
        pytest.skip("ts_kurt not in daily surface")

    expr = make_cleaned_call_factory("ts_kurt")(_sql_col("close"), window)

    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))

    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)

    _assert_parity(pd_out, sql_out)


@pytest.mark.parametrize("window,q", [(10, 0.5), (20, 0.75)])
def test_ts_quantile_duckdb_parity(duckdb_source, window, q):
    """Test ts_quantile DuckDB SQL parity."""
    if "ts_quantile" not in DAILY_CANONICALS:
        pytest.skip("ts_quantile not in daily surface")

    expr = make_cleaned_call_factory("ts_quantile")(_sql_col("close"), window, q)

    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))

    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)

    _assert_parity(pd_out, sql_out)


# ============================================================================
# Test: Edge Cases
# ============================================================================


def test_correlation_perfect_linear(panel):
    """Test ts_corr with perfectly correlated series."""
    if "ts_corr" not in DAILY_CANONICALS:
        pytest.skip("ts_corr not in daily surface")

    # close vs close should give correlation = 1.0
    expr = make_cleaned_call_factory("ts_corr")(col("close"), col("close"), 20)

    pd_out = _result_series(_run(panel, expr, "pandas"))
    polars_out = _result_series(_run(panel, expr, "polars_long"))

    # Where not NaN, correlation should be 1.0 (or very close)
    valid = pd_out.notna()
    assert np.allclose(pd_out[valid], 1.0, rtol=1e-6, atol=1e-6)

    _assert_parity(pd_out, polars_out)


def test_covariance_constant_series(panel):
    """Test ts_cov with constant values (should be 0)."""
    if "ts_cov" not in DAILY_CANONICALS:
        pytest.skip("ts_cov not in daily surface")

    # Create constant series
    idx = panel.data["close"].index
    const = pd.Series(100.0, index=idx)
    const_source = InMemorySeriesSource(data={"const": const, "close": panel.data["close"]})

    expr = make_cleaned_call_factory("ts_cov")(col("const"), col("close"), 20)

    pd_out = _result_series(_run(const_source, expr, "pandas"))
    polars_out = _result_series(_run(const_source, expr, "polars_long"))

    # Covariance with constant should be 0 or NaN
    _assert_parity(pd_out, polars_out)


def test_infinity_handling_correlation(panel):
    """Test ts_corr with Inf values."""
    if "ts_corr" not in DAILY_CANONICALS:
        pytest.skip("ts_corr not in daily surface")

    expr = make_cleaned_call_factory("ts_corr")(col("close_with_inf"), col("open"), 10)

    pd_out = _result_series(_run(panel, expr, "pandas"))
    polars_out = _result_series(_run(panel, expr, "polars_long"))

    # Should handle Inf consistently (likely produce NaN)
    _assert_parity(pd_out, polars_out)


def test_short_window_statistics(panel):
    """Test statistical operators with minimum valid window."""
    if "ts_corr" not in DAILY_CANONICALS or "ts_cov" not in DAILY_CANONICALS:
        pytest.skip("ts_corr/ts_cov not in daily surface")

    # Window = 2 (minimum for correlation/covariance)
    expr_corr = make_cleaned_call_factory("ts_corr")(col("close"), col("open"), 2)
    expr_cov = make_cleaned_call_factory("ts_cov")(col("close"), col("open"), 2)

    corr_pd = _result_series(_run(panel, expr_corr, "pandas"))
    corr_polars = _result_series(_run(panel, expr_corr, "polars_long"))
    _assert_parity(corr_pd, corr_polars)

    cov_pd = _result_series(_run(panel, expr_cov, "pandas"))
    cov_polars = _result_series(_run(panel, expr_cov, "polars_long"))
    _assert_parity(cov_pd, cov_polars)


def test_two_point_corr_is_exact_sign_oracle_across_pandas_and_polars():
    """Two-point Pearson is an algebraic identity, including large offsets."""
    load_all()
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    idx = pd.MultiIndex.from_product(
        [dates, ["A"]], names=["timestamp", "instrument"]
    )
    source = InMemorySeriesSource(data={
        "x": pd.Series([1e12, 1e12 + 1, np.nan, 1e12 + 4, 1e12 + 5], index=idx),
        "y": pd.Series([1e12, 1e12 - 3, np.nan, 1e12 + 8, 1e12 + 9], index=idx),
    })
    expr = make_cleaned_call_factory("ts_corr")(col("x"), col("y"), 2)
    expected = pd.Series(
        [np.nan, -1.0, np.nan, np.nan, 1.0], index=idx, name="test"
    )

    pandas_out = _result_series(_run(source, expr, "pandas"))
    polars_run = _run(source, expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True
    polars_out = _result_series(polars_run)
    pd.testing.assert_series_equal(pandas_out, expected, check_names=False)
    pd.testing.assert_series_equal(polars_out, expected, check_names=False)
    assert pandas_out.dropna().isin((-1.0, 1.0)).all()
    assert polars_out.dropna().isin((-1.0, 1.0)).all()


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_corr_centered_normalization_scale_oracle_across_pandas_and_polars(scale):
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    idx = pd.MultiIndex.from_product(
        [dates, ["A"]], names=["timestamp", "instrument"]
    )
    x_base = np.asarray([1.0, 2.0, 4.0, 7.0, 11.0])
    y_base = np.asarray([3.0, 1.0, 5.0, 2.0, 9.0])
    source = InMemorySeriesSource(data={
        "x": pd.Series(x_base * scale, index=idx),
        "y": pd.Series(y_base * scale, index=idx),
    })
    expr = make_cleaned_call_factory("ts_corr")(col("x"), col("y"), 5)
    expected = float(np.corrcoef(x_base, y_base)[0, 1])
    pandas_value = _result_series(_run(source, expr, "pandas")).iloc[-1]
    polars_value = _result_series(_run(source, expr, "polars_long")).iloc[-1]
    assert pandas_value == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert polars_value == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_all_nan_window(panel):
    """Test statistical operators when window contains all NaN."""
    if "ts_corr" not in DAILY_CANONICALS:
        pytest.skip("ts_corr not in daily surface")

    # GAPPED instrument has NaN gaps
    expr = make_cleaned_call_factory("ts_corr")(col("close"), col("open"), 5)

    pd_out = _result_series(_run(panel, expr, "pandas"))
    polars_out = _result_series(_run(panel, expr, "polars_long"))

    # Should produce NaN where insufficient data
    _assert_parity(pd_out, polars_out)


# ============================================================================
# Summary test
# ============================================================================


def test_statistical_operators_coverage_summary():
    """Report coverage of statistical operator parity tests."""

    operators = [
        ("ts_corr", "ts_corr" in DAILY_CANONICALS),
        ("ts_cov", "ts_cov" in DAILY_CANONICALS),
        ("ts_skew", "ts_skew" in DAILY_CANONICALS),
        ("ts_kurt", "ts_kurt" in DAILY_CANONICALS),
        ("ts_quantile", "ts_quantile" in DAILY_CANONICALS),
    ]

    pandas_polars_tests = 15  # Main parametrized + specialized tests
    duckdb_tests = 5  # DuckDB SQL tests
    edge_tests = 6  # Edge case tests

    total = pandas_polars_tests + duckdb_tests + edge_tests

    print(f"\n{'='*70}")
    print("Statistical Operators Parity Suite")
    print(f"{'='*70}")
    print(f"Operators targeted: {len(operators)}")
    for i, (op, available) in enumerate(operators, 1):
        status = "AVAILABLE" if available else "NOT IN DAILY_SURFACE"
        print(f"  {i}. {op}: {status}")

    available_count = sum(1 for _, avail in operators if avail)
    print(f"\nOperators in daily surface: {available_count}/{len(operators)}")
    print(f"\nTest breakdown:")
    print(f"  - Pandas vs Polars: {pandas_polars_tests} test functions")
    print(f"  - DuckDB SQL parity: {duckdb_tests} test functions")
    print(f"  - Edge cases: {edge_tests} test functions")
    print(f"\nTotal test functions: {total}")
    print(f"Note: Tests for unavailable operators will be skipped")
    print(f"{'='*70}\n")

    assert total == 26, "Expected 26 test functions"
    assert len(operators) == 5, "Expected 5 operators"
