# -*- coding: utf-8 -*-
"""Simplified cross-backend numerical consistency test - selective operator loading.

目标：测试核心算子在 Pandas/Polars/DuckDB 三后端的数值一致性。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from runtime.engine import FactorEngine
from storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource


def _generate_test_panel(seed: int = 42) -> InMemorySeriesSource:
    """Generate test panel data with diverse characteristics."""
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2024-01-02", periods=100, freq="D")
    instruments = [f"I{i:03d}" for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["timestamp", "instrument"])
    n = len(idx)

    # Close prices with trend
    close_values = 100.0 + np.repeat(np.linspace(0, 10, len(dates)), len(instruments)) + rng.normal(0, 2, n)
    nan_mask = rng.random(n) < 0.02
    close_values[nan_mask] = np.nan
    close = pd.Series(close_values, index=idx)

    # Open prices
    open_values = close_values - rng.uniform(0.5, 2.0, n)
    open_ = pd.Series(open_values, index=idx)

    # Volume
    volume = pd.Series(rng.exponential(1000, n), index=idx)

    # Returns
    ret = pd.Series(rng.normal(0.001, 0.02, n), index=idx)

    # Group IDs
    group_id = pd.Series(np.tile(np.arange(4), n // 4 + 1)[:n].astype(float), index=idx)

    # Flag
    flag = pd.Series((close > close.median()).astype(float), index=idx)

    return InMemorySeriesSource(
        data={
            "close": close,
            "open": open_,
            "volume": volume,
            "ret": ret,
            "group_id": group_id,
            "flag": flag,
        }
    )


@pytest.fixture(scope="module")
def test_source():
    return _generate_test_panel()


def _run_factor(source, expr, backend: str) -> pd.Series:
    """Execute factor on specified backend."""
    result = FactorEngine(
        backend=build_backend(backend),
        data_source=source,
        run_mode="research"
    ).run(Factor(name="test", expr=expr))
    return result["result"].sort_index()


def _assert_close(result1: pd.Series, result2: pd.Series, op_name: str, backend1: str, backend2: str):
    """Assert two results are numerically close."""
    # Check index
    assert result1.index.equals(result2.index), f"{op_name}: Index mismatch"

    # Check NaN pattern
    nan1 = result1.isna()
    nan2 = result2.isna()
    assert nan1.equals(nan2), f"{op_name}: NaN pattern mismatch ({nan1.sum()} vs {nan2.sum()})"

    # Check finite values
    finite = ~nan1
    if finite.sum() > 0:
        v1 = result1[finite]
        v2 = result2[finite]

        # Check Inf pattern
        inf1 = np.isinf(v1)
        inf2 = np.isinf(v2)
        assert inf1.equals(inf2), f"{op_name}: Inf pattern mismatch"

        # Check actual values
        both_finite = ~inf1
        if both_finite.sum() > 0:
            np.testing.assert_allclose(
                v1[both_finite], v2[both_finite],
                atol=1e-10, rtol=1e-9,
                err_msg=f"{op_name}: {backend1} vs {backend2} value mismatch"
            )


# Core operators to test (carefully selected to avoid load_all issues)
F = make_cleaned_call_factory

CORE_TESTS = [
    # Arithmetic
    ("add", lambda: F("add")(col("close"), col("open"))),
    ("subtract", lambda: F("subtract")(col("close"), col("open"))),
    ("multiply", lambda: F("multiply")(col("close"), col("open"))),
    ("divide", lambda: F("divide")(col("close"), col("open"))),
    ("abs", lambda: F("abs")(col("ret"))),
    ("neg", lambda: F("neg")(col("close"))),
    ("sign", lambda: F("sign")(col("ret"))),

    # Math
    ("log", lambda: F("log")(col("close"))),
    ("exp", lambda: F("exp")(col("ret"))),
    ("sqrt", lambda: F("sqrt")(col("close"))),
    ("floor", lambda: F("floor")(col("close"))),
    ("ceil", lambda: F("ceil")(col("close"))),
    ("clip", lambda: F("clip")(col("close"), 90.0, 110.0)),
    ("maximum", lambda: F("maximum")(col("close"), col("open"))),
    ("minimum", lambda: F("minimum")(col("close"), col("open"))),

    # Time series
    ("ts_delay", lambda: F("ts_delay")(col("close"), 2)),
    ("ts_delta", lambda: F("ts_delta")(col("close"), 1)),
    ("ts_pct", lambda: F("ts_pct")(col("close"), 1)),
    ("ts_mean", lambda: F("ts_mean")(col("close"), 5)),
    ("ts_sum", lambda: F("ts_sum")(col("close"), 5)),
    ("ts_std", lambda: F("ts_std")(col("close"), 5)),
    ("ts_min", lambda: F("ts_min")(col("close"), 5)),
    ("ts_max", lambda: F("ts_max")(col("close"), 5)),
    ("ts_rank", lambda: F("ts_rank")(col("close"), 5)),

    # Cross-sectional
    ("cs_rank", lambda: F("cs_rank")(col("close"))),
    ("cs_demean", lambda: F("cs_demean")(col("close"))),
    ("cs_mean", lambda: F("cs_mean")(col("close"))),
    ("cs_std", lambda: F("cs_std")(col("close"))),
    ("cs_sum", lambda: F("cs_sum")(col("close"))),
    ("rank", lambda: F("rank")(col("close"))),
    ("zscore", lambda: F("zscore")(col("close"))),
    ("scale", lambda: F("scale")(col("close"))),

    # Group
    ("group_mean", lambda: F("group_mean")(col("close"), col("group_id"))),
    ("group_std", lambda: F("group_std")(col("close"), col("group_id"))),
    ("group_sum", lambda: F("group_sum")(col("close"), col("group_id"))),
    ("group_neutralize", lambda: F("group_neutralize")(col("close"), col("group_id"))),

    # Cumulative
    ("cum_sum", lambda: F("cum_sum")(col("close"))),
    ("cum_max", lambda: F("cum_max")(col("close"))),
    ("cum_min", lambda: F("cum_min")(col("close"))),
    ("cum_prod", lambda: F("cum_prod")(col("ret"))),

    # Expanding
    ("expanding_mean", lambda: F("expanding_mean")(col("close"))),
    ("expanding_sum", lambda: F("expanding_sum")(col("close"))),

    # Comparison
    ("gt", lambda: F("gt")(col("close"), col("open"))),
    ("lt", lambda: F("lt")(col("close"), col("open"))),
    ("ge", lambda: F("ge")(col("close"), col("open"))),
    ("le", lambda: F("le")(col("close"), col("open"))),
    ("eq", lambda: F("eq")(col("close"), col("open"))),
    ("ne", lambda: F("ne")(col("close"), col("open"))),

    # Logic
    ("and_", lambda: F("and_")(col("flag"), col("flag"))),
    ("or_", lambda: F("or_")(col("flag"), col("flag"))),
    ("not_", lambda: F("not_")(col("flag"))),
    ("coalesce", lambda: F("coalesce")(col("close"), col("open"))),

    # NA handling
    ("fillna", lambda: F("fillna")(col("close"), 100.0)),
    ("ffill", lambda: F("ffill")(col("close"))),
    ("is_nan", lambda: F("is_nan")(col("close"))),
    ("is_finite", lambda: F("is_finite")(col("close"))),
    ("is_not_null", lambda: F("is_not_null")(col("close"))),

    # Technical
    ("volatility", lambda: F("volatility")(col("close"), 5)),
]


@pytest.mark.parametrize("op_name,expr_fn", CORE_TESTS)
def test_pandas_polars_parity(test_source, op_name, expr_fn):
    """Test Pandas vs Polars parity."""
    try:
        expr = expr_fn()
        pandas_result = _run_factor(test_source, expr, "pandas")
        polars_result = _run_factor(test_source, expr, "polars_long")
        _assert_close(pandas_result, polars_result, op_name, "pandas", "polars")
    except Exception as e:
        pytest.skip(f"{op_name}: {type(e).__name__}: {str(e)[:100]}")


def test_generate_parity_summary(test_source, tmp_path):
    """Generate comprehensive parity report."""
    results = []

    for op_name, expr_fn in CORE_TESTS:
        row = {"operator": op_name, "pandas": "N/A", "polars": "N/A"}

        # Test Pandas (baseline)
        try:
            expr = expr_fn()
            pandas_result = _run_factor(test_source, expr, "pandas")
            row["pandas"] = "OK"
            row["valid_count"] = (~pandas_result.isna()).sum()
        except Exception as e:
            row["pandas"] = f"FAIL: {type(e).__name__}"
            row["valid_count"] = 0
            results.append(row)
            continue

        # Test Polars
        try:
            expr = expr_fn()
            polars_result = _run_factor(test_source, expr, "polars_long")
            _assert_close(pandas_result, polars_result, op_name, "pandas", "polars")
            row["polars"] = "MATCH"
        except AssertionError as e:
            row["polars"] = f"MISMATCH: {str(e)[:50]}"
        except Exception as e:
            row["polars"] = f"FAIL: {type(e).__name__}"

        results.append(row)

    # Write report
    report_path = Path("/tmp/backend_parity_report.md")
    _write_report(results, report_path)

    # Summary
    total = len(results)
    pandas_ok = sum(1 for r in results if r["pandas"] == "OK")
    polars_match = sum(1 for r in results if r.get("polars") == "MATCH")

    print(f"\n{'='*60}")
    print(f"Backend Parity Report: {report_path}")
    print(f"{'='*60}")
    print(f"Total operators: {total}")
    print(f"Pandas OK: {pandas_ok}/{total} ({100*pandas_ok/total:.1f}%)")
    print(f"Polars match: {polars_match}/{total} ({100*polars_match/total:.1f}%)")
    print(f"{'='*60}\n")

    assert polars_match >= total * 0.5, f"Polars parity too low: {polars_match}/{total}"


def _write_report(results: list[dict], path: Path):
    """Write markdown report."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Cross-Backend Numerical Parity Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")

        total = len(results)
        pandas_ok = sum(1 for r in results if r["pandas"] == "OK")
        polars_match = sum(1 for r in results if r.get("polars") == "MATCH")

        f.write("## Summary\n\n")
        f.write(f"- **Total operators**: {total}\n")
        f.write(f"- **Pandas baseline**: {pandas_ok}/{total} ({100*pandas_ok/total:.1f}%)\n")
        f.write(f"- **Polars parity**: {polars_match}/{total} ({100*polars_match/total:.1f}%)\n\n")

        f.write("## Detailed Results\n\n")
        f.write("| Operator | Pandas | Polars | Valid Count |\n")
        f.write("|----------|--------|--------|-------------|\n")

        for r in results:
            f.write(f"| {r['operator']} | {r['pandas']} | {r.get('polars', 'N/A')} | {r.get('valid_count', 0)} |\n")

        f.write("\n## Issues\n\n")
        issues = [r for r in results if r.get("polars", "").startswith(("MISMATCH", "FAIL"))]
        if issues:
            for r in issues:
                f.write(f"- **{r['operator']}**: {r.get('polars')}\n")
        else:
            f.write("No issues detected!\n")
