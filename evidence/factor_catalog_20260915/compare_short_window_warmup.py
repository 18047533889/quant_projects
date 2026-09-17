#!/usr/bin/env python3
"""Compare short auto-warmup outputs with a long-window real-data baseline."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


FORMULAS = (
    ("mean_252", "ts_mean(close, 252)"),
    ("spectral_120", "ts_return_spectral_entropy(ret, window=120)"),
    ("nested_tema", "ts_mean(TEMA(close, 30), 20)"),
)
SYMBOLS = (
    "000001.SZ", "000002.SZ", "000063.SZ", "000333.SZ",
    "000651.SZ", "000858.SZ", "600000.SH", "600519.SH",
)
LONG_START = "2025-01-01"
SHORT_START = "2026-04-27"
END = "2026-04-30"


def frame(value):
    if isinstance(value, dict) and "result" in value:
        value = value["result"]
    if isinstance(value, pd.Series):
        value = value.unstack() if value.index.nlevels >= 2 else value.to_frame("value")
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"expected pandas DataFrame, got {type(value).__name__}")
    return value.sort_index().sort_index(axis=1)


def source(fields, start):
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    return DataAccessSource(
        dataset="ashare_stock_daily_adj",
        fields=fields,
        start_date=start,
        end_date=END,
        instrument_filter=list(SYMBOLS),
        run_mode="interactive_research",
        production=False,
        read_auto=False,
    )


def compare(left, right):
    index_equal = left.index.equals(right.index)
    columns_equal = left.columns.equals(right.columns)
    if not index_equal or not columns_equal:
        return {"index_equal": index_equal, "columns_equal": columns_equal}
    a = left.to_numpy(dtype=float)
    b = right.to_numpy(dtype=float)
    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    mask_equal = np.array_equal(finite_a, finite_b)
    both = finite_a & finite_b
    abs_error = np.abs(a[both] - b[both]) if both.any() else np.array([])
    denom = np.maximum(np.abs(a[both]), np.abs(b[both])) if both.any() else np.array([])
    rel_error = abs_error / np.maximum(denom, np.finfo(float).tiny) if both.any() else np.array([])
    return {
        "index_equal": True,
        "columns_equal": True,
        "finite_mask_equal": bool(mask_equal),
        "finite_count_long_slice": int(finite_a.sum()),
        "finite_count_short": int(finite_b.sum()),
        "max_abs_error": float(abs_error.max()) if abs_error.size else 0.0,
        "max_rel_error": float(rel_error.max()) if rel_error.size else 0.0,
        "allclose_rtol_1e-10_atol_1e-12": bool(
            mask_equal and np.allclose(a[both], b[both], rtol=1e-10, atol=1e-12)
        ),
    }


def main():
    os.environ["ASHARE_PARQUET_ROOT"] = "/home/sunhaiwei/cos_data"
    os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    os.environ["DATA_ACCESS_RUN_MODE"] = "interactive_research"

    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators import load_all
    from factor_engine.runtime.engine import FactorEngine
    from smoke_catalog import bind_fields

    load_all()
    parser = DSLParser(surface="compat_research", dialect="native")
    expressions = []
    physical_fields = {}
    for name, formula in FORMULAS:
        expr = parser.parse(formula)
        bindings, failures = bind_fields(expr)
        if failures:
            raise RuntimeError(f"{name} binding failures: {failures}")
        for binding in bindings:
            if binding["dataset"] != "ashare_stock_daily_adj":
                raise RuntimeError(f"{name} unexpected dataset: {binding}")
            physical_fields[binding["input"]] = binding["column"]
        expressions.append((name, formula, expr))

    report = {
        "long_window": [LONG_START, END],
        "short_output_window": [SHORT_START, END],
        "symbols": list(SYMBOLS),
        "results": [],
    }
    for name, formula, expr in expressions:
        factor = Factor(
            name=name, expr=expr, source_expr=formula, surface="compat_research"
        )
        long_engine = FactorEngine(
            PandasBackend(), source(physical_fields, LONG_START), run_mode="research"
        )
        started = time.monotonic()
        long_result = frame(long_engine.run(factor))
        long_seconds = time.monotonic() - started

        short_engine = FactorEngine(
            PandasBackend(), source(physical_fields, SHORT_START), run_mode="research"
        )
        plan, analysis = short_engine.compile(factor)
        started = time.monotonic()
        short_context = short_engine.run(
            factor,
            plan=plan,
            analysis=analysis,
            auto_warmup=True,
            trim_warmup=True,
            market="ashare",
        )
        short_seconds = time.monotonic() - started
        short_result = frame(short_context)
        long_slice = long_result.loc[
            (long_result.index >= pd.Timestamp(SHORT_START))
            & (long_result.index <= pd.Timestamp(END))
        ]
        run_window = short_context.get("run_window") if isinstance(short_context, dict) else None
        item = {
            "name": name,
            "formula": formula,
            "analysis_lookback": analysis.lookback,
            "long_seconds": round(long_seconds, 6),
            "short_seconds": round(short_seconds, 6),
            "short_run_window": str(run_window) if run_window is not None else None,
            **compare(long_slice, short_result),
        }
        report["results"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    report["safe_for_these_representatives"] = all(
        row.get("index_equal")
        and row.get("columns_equal")
        and row.get("finite_mask_equal")
        and row.get("allclose_rtol_1e-10_atol_1e-12")
        for row in report["results"]
    )
    output = Path(__file__).with_name("short-window-warmup-comparison.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "safe": report["safe_for_these_representatives"]}))


if __name__ == "__main__":
    main()
