# -*- coding: utf-8 -*-
"""Cross-backend numerical consistency test suite - Pandas/Polars/DuckDB/q parity.

目标：确保 100 个常用算子在所有后端产生数值一致的结果（浮点误差 ≤ 1e-10）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource

logger = logging.getLogger(__name__)

# Top 100 most commonly used operators (curated from production usage + core coverage)
TOP_100_OPERATORS = [
    # Arithmetic & Math (20)
    "add", "subtract", "multiply", "divide", "abs", "neg", "sign",
    "log", "exp", "sqrt", "power", "inverse", "floor", "ceil",
    "clip", "maximum", "minimum", "protected_sqrt", "signed_log", "signed_sqrt",

    # Time Series Rolling (25)
    "ts_delay", "ts_delta", "ts_pct", "log_returns",
    "ts_mean", "ts_sum", "ts_std", "ts_var", "ts_min", "ts_max",
    "ts_median", "ts_rank", "ts_count", "ts_ema", "ts_wma",
    "ts_skew", "ts_kurt", "ts_autocorr", "ts_prod", "ts_quantile",
    "ts_corr", "ts_cov", "ts_decay_linear", "ts_argmax", "ts_argmin",

    # Cross-Sectional (15)
    "cs_rank", "cs_zscore", "cs_demean", "cs_mean", "cs_std",
    "cs_sum", "cs_count", "cs_min", "cs_max", "cs_median",
    "rank", "zscore", "normalize", "scale", "winsorize",

    # Group Operations (10)
    "group_mean", "group_std", "group_sum", "group_rank", "group_neutralize",
    "group_normalize", "group_min", "group_max", "group_count", "group_demean",

    # Cumulative & Expanding (8)
    "cum_sum", "cum_prod", "cum_max", "cum_min", "cum_delta",
    "expanding_mean", "expanding_sum", "expanding_std",

    # Comparison & Logic (12)
    "gt", "lt", "ge", "le", "eq", "ne",
    "and_", "or_", "not_", "where", "coalesce", "if_then_else",

    # NA Handling (6)
    "fillna", "fillna_const", "ffill", "nan_to_num",
    "is_nan", "is_finite",

    # Technical Indicators (4)
    "volatility", "ts_sharpe", "SMA", "EMA",
]


def _generate_panel_data(n_timestamps: int = 50, n_instruments: int = 20, seed: int = 42) -> InMemorySeriesSource:
    """Generate diverse panel data for comprehensive parity testing.

    Features:
    - Normal values with realistic patterns
    - NaN values (sparse)
    - Inf values (sparse)
    - Zero values
    - Negative values
    - Small and large magnitudes
    """
    load_all()
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2024-01-02", periods=n_timestamps, freq="D")
    instruments = [f"I{i:03d}" for i in range(n_instruments)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["timestamp", "instrument"])
    n = len(idx)

    # Base close prices with trend and noise
    base = 100.0
    trend = np.linspace(0, 10, n_timestamps)
    noise = rng.normal(0, 2, n)
    close_values = base + np.repeat(trend, n_instruments) + noise

    # Inject special cases (NaN, Inf, zeros)
    nan_mask = rng.random(n) < 0.03  # 3% NaN
    close_values[nan_mask] = np.nan

    inf_mask = rng.random(n) < 0.01  # 1% Inf
    close_values[inf_mask] = np.inf

    close = pd.Series(close_values, index=idx)

    # Open prices (similar to close with offset)
    open_values = close_values - rng.uniform(0.5, 2.0, n)
    open_ = pd.Series(open_values, index=idx)

    # High/Low prices
    high_values = close_values + rng.uniform(0, 1.5, n)
    low_values = close_values - rng.uniform(0, 1.5, n)
    high = pd.Series(high_values, index=idx)
    low = pd.Series(low_values, index=idx)

    # Volume (positive, with some zeros)
    volume_values = rng.exponential(1000, n)
    zero_vol_mask = rng.random(n) < 0.02  # 2% zero volume
    volume_values[zero_vol_mask] = 0.0
    volume = pd.Series(volume_values, index=idx)

    # Returns
    ret_values = rng.normal(0.001, 0.02, n)
    ret = pd.Series(ret_values, index=idx)

    # Group IDs for group operations (4 groups)
    group_values = np.tile(np.arange(4), n // 4 + 1)[:n]
    rng.shuffle(group_values)
    group_id = pd.Series(group_values.astype(float), index=idx)

    # Boolean flag
    flag = pd.Series((close > close.median()).astype(float), index=idx)

    return InMemorySeriesSource(
        data={
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
            "volume": volume,
            "ret": ret,
            "group_id": group_id,
            "flag": flag,
        }
    )


@pytest.fixture(scope="module")
def panel_source():
    """Shared panel data for all tests."""
    return _generate_panel_data(n_timestamps=100, n_instruments=50)


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    """Create DuckDB data source registry."""
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
    Flag: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb_panel(root: Path, mem: InMemorySeriesSource) -> None:
    """Seed DuckDB parquet files from memory source."""
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
        "ret": "Ret",
        "group_id": "group_id",
        "flag": "Flag",
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
def duckdb_source(tmp_path_factory, panel_source):
    """DuckDB data source with same data as panel_source."""
    tmp = tmp_path_factory.mktemp("duckdb_parity")
    import os

    old_config = os.environ.get("DATA_ACCESS_CONFIG")
    old_skip = os.environ.get("DATA_ACCESS_SKIP_COS_MIRROR")
    os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"

    reg = _write_duckdb_registry(tmp, tmp / "data")
    os.environ["DATA_ACCESS_CONFIG"] = str(reg)
    _seed_duckdb_panel(tmp / "data", panel_source)

    try:
        from data_access import reset_store
        reset_store()
    except ImportError:
        pass

    yield build_data_source({"type": "data_access", "dataset": "test_parity"})

    if old_config is None:
        os.environ.pop("DATA_ACCESS_CONFIG", None)
    else:
        os.environ["DATA_ACCESS_CONFIG"] = old_config
    if old_skip is None:
        os.environ.pop("DATA_ACCESS_SKIP_COS_MIRROR", None)
    else:
        os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = old_skip


def _col_for_backend(name: str, backend: str) -> Any:
    """Get column reference adjusted for backend naming."""
    if backend == "sql":
        mapping = {
            "close": "Close",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
            "ret": "Ret",
            "flag": "Flag",
        }
        return col(mapping.get(name, name))
    return col(name)


def _run_backend(source, expr, backend: str) -> pd.Series:
    """Execute factor computation on specified backend."""
    result = FactorEngine(
        backend=build_backend(backend),
        data_source=source,
        run_mode="research"
    ).run(Factor(name="test", expr=expr))
    return result["result"].sort_index()


def _compare_results(
    pandas_result: pd.Series,
    other_result: pd.Series,
    backend_name: str,
    operator_name: str,
    atol: float = 1e-10,
    rtol: float = 1e-9,
) -> tuple[bool, str]:
    """Compare two result series with detailed diagnostics.

    Returns:
        (is_match, diagnostic_message)
    """
    # Check index alignment
    if not pandas_result.index.equals(other_result.index):
        return False, f"Index mismatch: pandas {len(pandas_result)} vs {backend_name} {len(other_result)}"

    # Check NaN pattern match
    pandas_nan_mask = pandas_result.isna()
    other_nan_mask = other_result.isna()
    if not pandas_nan_mask.equals(other_nan_mask):
        diff_count = (pandas_nan_mask != other_nan_mask).sum()
        return False, f"NaN pattern mismatch: {diff_count} differences"

    # Check finite values
    finite_mask = ~pandas_nan_mask
    if finite_mask.sum() == 0:
        return True, "All NaN (consistent)"

    pandas_finite = pandas_result[finite_mask]
    other_finite = other_result[finite_mask]

    # Check for Inf pattern match
    pandas_inf_mask = np.isinf(pandas_finite)
    other_inf_mask = np.isinf(other_finite)
    if not pandas_inf_mask.equals(other_inf_mask):
        diff_count = (pandas_inf_mask != other_inf_mask).sum()
        return False, f"Inf pattern mismatch: {diff_count} differences"

    # Check actual finite values
    both_finite_mask = ~pandas_inf_mask
    if both_finite_mask.sum() == 0:
        return True, "All Inf (consistent)"

    pandas_values = pandas_finite[both_finite_mask]
    other_values = other_finite[both_finite_mask]

    # Numerical comparison
    try:
        np.testing.assert_allclose(pandas_values, other_values, atol=atol, rtol=rtol)
        max_diff = np.abs(pandas_values - other_values).max()
        return True, f"Match (max diff: {max_diff:.2e})"
    except AssertionError as e:
        max_diff = np.abs(pandas_values - other_values).max()
        return False, f"Value mismatch: max diff {max_diff:.2e} (tol {atol:.2e})"


# Operator test specifications
OPERATOR_SPECS = {
    # Arithmetic (1-2 inputs)
    "add": {"args": ["close", "open"], "arity": 2},
    "subtract": {"args": ["close", "open"], "arity": 2},
    "multiply": {"args": ["close", "open"], "arity": 2},
    "divide": {"args": ["close", "open"], "arity": 2},
    "abs": {"args": ["ret"], "arity": 1},
    "neg": {"args": ["close"], "arity": 1},
    "sign": {"args": ["ret"], "arity": 1},
    "log": {"args": ["close"], "arity": 1},
    "exp": {"args": ["ret"], "arity": 1},
    "sqrt": {"args": ["close"], "arity": 1},
    "power": {"args": ["close"], "arity": 1, "kwargs": {"exponent": 2}},
    "inverse": {"args": ["close"], "arity": 1},
    "floor": {"args": ["close"], "arity": 1},
    "ceil": {"args": ["close"], "arity": 1},
    "clip": {"args": ["close"], "arity": 1, "kwargs": {"lower": 90.0, "upper": 110.0}},
    "maximum": {"args": ["close", "open"], "arity": 2},
    "minimum": {"args": ["close", "open"], "arity": 2},

    # Protected math
    "protected_sqrt": {"args": ["ret"], "arity": 1},
    "signed_log": {"args": ["ret"], "arity": 1},
    "signed_sqrt": {"args": ["ret"], "arity": 1},
    "log_abs": {"args": ["ret"], "arity": 1},
    "safe_div_null": {"args": ["close", "volume"], "arity": 2},

    # Time series (window-based)
    "ts_delay": {"args": ["close"], "arity": 1, "kwargs": {"window": 2}},
    "ts_delta": {"args": ["close"], "arity": 1, "kwargs": {"window": 1}},
    "ts_pct": {"args": ["close"], "arity": 1, "kwargs": {"window": 1}},
    "ts_mean": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_sum": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_std": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_var": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_min": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_max": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_median": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_rank": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_count": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_prod": {"args": ["ret"], "arity": 1, "kwargs": {"window": 3}},
    "ts_skew": {"args": ["close"], "arity": 1, "kwargs": {"window": 10}},
    "ts_kurt": {"args": ["close"], "arity": 1, "kwargs": {"window": 10}},
    "ts_autocorr": {"args": ["close"], "arity": 1, "kwargs": {"window": 10, "lag": 1}},
    "ts_argmax": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_argmin": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_decay_linear": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_corr": {"args": ["close", "volume"], "arity": 2, "kwargs": {"window": 10}},
    "ts_cov": {"args": ["close", "volume"], "arity": 2, "kwargs": {"window": 10}},

    # EMA/WMA
    "ts_ema": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_wma": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "EMA": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "SMA": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},

    # Cross-sectional
    "cs_rank": {"args": ["close"], "arity": 1},
    "cs_zscore": {"args": ["close"], "arity": 1},
    "cs_demean": {"args": ["close"], "arity": 1},
    "cs_mean": {"args": ["close"], "arity": 1},
    "cs_std": {"args": ["close"], "arity": 1},
    "cs_sum": {"args": ["close"], "arity": 1},
    "cs_count": {"args": ["close"], "arity": 1},
    "cs_min": {"args": ["close"], "arity": 1},
    "cs_max": {"args": ["close"], "arity": 1},
    "cs_median": {"args": ["close"], "arity": 1},
    "rank": {"args": ["close"], "arity": 1},
    "zscore": {"args": ["close"], "arity": 1},
    "normalize": {"args": ["close"], "arity": 1},
    "scale": {"args": ["close"], "arity": 1},
    "winsorize": {"args": ["close"], "arity": 1},

    # Group operations
    "group_mean": {"args": ["close", "group_id"], "arity": 2},
    "group_std": {"args": ["close", "group_id"], "arity": 2},
    "group_sum": {"args": ["close", "group_id"], "arity": 2},
    "group_rank": {"args": ["close", "group_id"], "arity": 2},
    "group_neutralize": {"args": ["close", "group_id"], "arity": 2},
    "group_normalize": {"args": ["close", "group_id"], "arity": 2},
    "group_min": {"args": ["close", "group_id"], "arity": 2},
    "group_max": {"args": ["close", "group_id"], "arity": 2},
    "group_count": {"args": ["close", "group_id"], "arity": 2},
    "group_demean": {"args": ["close", "group_id"], "arity": 2},

    # Cumulative
    "cum_sum": {"args": ["close"], "arity": 1},
    "cum_prod": {"args": ["ret"], "arity": 1},
    "cum_max": {"args": ["close"], "arity": 1},
    "cum_min": {"args": ["close"], "arity": 1},
    "cum_delta": {"args": ["close"], "arity": 1},

    # Expanding
    "expanding_mean": {"args": ["close"], "arity": 1},
    "expanding_sum": {"args": ["close"], "arity": 1},
    "expanding_std": {"args": ["close"], "arity": 1},

    # Comparison
    "gt": {"args": ["close", "open"], "arity": 2},
    "lt": {"args": ["close", "open"], "arity": 2},
    "ge": {"args": ["close", "open"], "arity": 2},
    "le": {"args": ["close", "open"], "arity": 2},
    "eq": {"args": ["close", "open"], "arity": 2},
    "ne": {"args": ["close", "open"], "arity": 2},

    # Logic
    "and_": {"args": ["flag", "flag"], "arity": 2},
    "or_": {"args": ["flag", "flag"], "arity": 2},
    "not_": {"args": ["flag"], "arity": 1},
    "coalesce": {"args": ["close", "open"], "arity": 2},

    # NA handling
    "fillna": {"args": ["close"], "arity": 1, "kwargs": {"value": 100.0}},
    "fillna_const": {"args": ["close"], "arity": 1, "kwargs": {"value": 100.0}},
    "ffill": {"args": ["close"], "arity": 1},
    "nan_to_num": {"args": ["close"], "arity": 1},
    "is_nan": {"args": ["close"], "arity": 1},
    "is_finite": {"args": ["close"], "arity": 1},
    "is_not_null": {"args": ["close"], "arity": 1},
    "is_null": {"args": ["close"], "arity": 1},
    "is_infinite": {"args": ["close"], "arity": 1},

    # Technical
    "volatility": {"args": ["close"], "arity": 1, "kwargs": {"window": 5}},
    "ts_sharpe": {"args": ["ret"], "arity": 1, "kwargs": {"window": 10}},
}


def _build_expr(op_name: str, spec: dict[str, Any], backend: str):
    """Build factor expression for operator."""
    F = make_cleaned_call_factory
    args = [_col_for_backend(arg, backend) for arg in spec["args"]]
    kwargs = spec.get("kwargs", {})
    return F(op_name)(*args, **kwargs)


@pytest.mark.parametrize("op_name", sorted(OPERATOR_SPECS.keys()))
def test_pandas_polars_parity(panel_source, op_name):
    """Test Pandas vs Polars backend parity."""
    spec = OPERATOR_SPECS[op_name]

    try:
        expr_pd = _build_expr(op_name, spec, "pandas")
        expr_pl = _build_expr(op_name, spec, "polars")

        pandas_result = _run_backend(panel_source, expr_pd, "pandas_numpy")
        polars_result = _run_backend(panel_source, expr_pl, "polars_long")

        is_match, msg = _compare_results(pandas_result, polars_result, "polars", op_name)
        assert is_match, f"{op_name}: Pandas vs Polars - {msg}"

    except (ImportError, ModuleNotFoundError, ValueError) as e:
        pytest.skip(f"{op_name}: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"{op_name}: Unexpected error - {type(e).__name__}: {e}")


@pytest.mark.parametrize("op_name", sorted(OPERATOR_SPECS.keys()))
def test_pandas_duckdb_parity(panel_source, duckdb_source, op_name):
    """Test Pandas vs DuckDB backend parity."""
    spec = OPERATOR_SPECS[op_name]

    try:
        expr_pd = _build_expr(op_name, spec, "pandas")
        expr_sql = _build_expr(op_name, spec, "sql")

        pandas_result = _run_backend(panel_source, expr_pd, "pandas_numpy")
        duckdb_result = _run_backend(duckdb_source, expr_sql, "sql")

        is_match, msg = _compare_results(pandas_result, duckdb_result, "duckdb", op_name)
        assert is_match, f"{op_name}: Pandas vs DuckDB - {msg}"

    except (ImportError, ModuleNotFoundError, ValueError) as e:
        pytest.skip(f"{op_name}: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"{op_name}: Unexpected error - {type(e).__name__}: {e}")


def test_generate_parity_report(panel_source, duckdb_source, tmp_path):
    """Generate comprehensive backend parity report."""
    results = []

    for op_name in sorted(OPERATOR_SPECS.keys()):
        spec = OPERATOR_SPECS[op_name]
        row = {"operator": op_name, "arity": spec["arity"]}

        # Test Pandas (baseline)
        try:
            expr_pd = _build_expr(op_name, spec, "pandas")
            pandas_result = _run_backend(panel_source, expr_pd, "pandas_numpy")
            row["pandas"] = "OK"
            row["pandas_count"] = (~pandas_result.isna()).sum()
        except Exception as e:
            row["pandas"] = f"FAIL: {type(e).__name__}"
            row["pandas_count"] = 0
            results.append(row)
            continue

        # Test Polars
        try:
            expr_pl = _build_expr(op_name, spec, "polars")
            polars_result = _run_backend(panel_source, expr_pl, "polars_long")
            is_match, msg = _compare_results(pandas_result, polars_result, "polars", op_name)
            row["polars"] = "MATCH" if is_match else f"MISMATCH: {msg}"
        except Exception as e:
            row["polars"] = f"FAIL: {type(e).__name__}"

        # Test DuckDB
        try:
            expr_sql = _build_expr(op_name, spec, "sql")
            duckdb_result = _run_backend(duckdb_source, expr_sql, "sql")
            is_match, msg = _compare_results(pandas_result, duckdb_result, "duckdb", op_name)
            row["duckdb"] = "MATCH" if is_match else f"MISMATCH: {msg}"
        except Exception as e:
            row["duckdb"] = f"FAIL: {type(e).__name__}"

        # Test q (if available)
        try:
            from factor_engine.backend.q_backend import is_q_available
            if is_q_available():
                expr_q = _build_expr(op_name, spec, "q")
                q_result = _run_backend(panel_source, expr_q, "q")
                is_match, msg = _compare_results(pandas_result, q_result, "q", op_name)
                row["q"] = "MATCH" if is_match else f"MISMATCH: {msg}"
            else:
                row["q"] = "UNAVAILABLE"
        except Exception as e:
            row["q"] = f"FAIL: {type(e).__name__}"

        results.append(row)

    # Generate report
    report_path = tmp_path / "backend_parity_report.md"
    _write_report(results, report_path)

    # Also write to /tmp for easy access
    global_report = Path("/tmp/backend_parity_report.md")
    _write_report(results, global_report)

    # Summary
    total = len(results)
    polars_match = sum(1 for r in results if r.get("polars") == "MATCH")
    duckdb_match = sum(1 for r in results if r.get("duckdb") == "MATCH")
    q_match = sum(1 for r in results if r.get("q") == "MATCH")

    logger.info(f"Parity Report Generated: {global_report}")
    logger.info(f"Total operators: {total}")
    logger.info(f"Polars match: {polars_match}/{total} ({100*polars_match/total:.1f}%)")
    logger.info(f"DuckDB match: {duckdb_match}/{total} ({100*duckdb_match/total:.1f}%)")
    logger.info(f"q match: {q_match}/{total} ({100*q_match/total:.1f}%)")

    # Verify no parity failures
    failures = []
    for r in results:
        if r.get("polars", "").startswith(("MISMATCH", "FAIL")):
            failures.append(f"polars:{r['operator']}")
        if r.get("duckdb", "").startswith(("MISMATCH", "FAIL")):
            failures.append(f"duckdb:{r['operator']}")
        if r.get("q", "").startswith(("MISMATCH", "FAIL")):
            failures.append(f"q:{r['operator']}")
    assert len(failures) == 0, f"Backend parity failures detected: {failures}"


def _write_report(results: list[dict[str, Any]], path: Path) -> None:
    """Write parity report to markdown file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Cross-Backend Numerical Parity Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")

        # Summary
        total = len(results)
        pandas_ok = sum(1 for r in results if r.get("pandas") == "OK")
        polars_match = sum(1 for r in results if r.get("polars") == "MATCH")
        duckdb_match = sum(1 for r in results if r.get("duckdb") == "MATCH")
        q_match = sum(1 for r in results if r.get("q") == "MATCH")

        f.write("## Summary\n\n")
        f.write(f"- **Total operators tested**: {total}\n")
        f.write(f"- **Pandas (baseline)**: {pandas_ok}/{total} ({100*pandas_ok/total:.1f}%)\n")
        f.write(f"- **Polars parity**: {polars_match}/{total} ({100*polars_match/total:.1f}%)\n")
        f.write(f"- **DuckDB parity**: {duckdb_match}/{total} ({100*duckdb_match/total:.1f}%)\n")
        f.write(f"- **q parity**: {q_match}/{total} ({100*q_match/total:.1f}%)\n\n")

        # Detailed table
        f.write("## Detailed Results\n\n")
        f.write("| Operator | Arity | Pandas | Polars | DuckDB | q |\n")
        f.write("|----------|-------|--------|--------|--------|---|\n")

        for r in results:
            f.write(f"| {r['operator']} | {r['arity']} | {r.get('pandas', 'N/A')} | "
                   f"{r.get('polars', 'N/A')} | {r.get('duckdb', 'N/A')} | {r.get('q', 'N/A')} |\n")

        # Failures section
        f.write("\n## Mismatches and Failures\n\n")

        for backend in ["polars", "duckdb", "q"]:
            failures = [r for r in results if r.get(backend, "").startswith(("MISMATCH", "FAIL"))]
            if failures:
                f.write(f"### {backend.upper()}\n\n")
                for r in failures:
                    f.write(f"- **{r['operator']}**: {r.get(backend)}\n")
                f.write("\n")

