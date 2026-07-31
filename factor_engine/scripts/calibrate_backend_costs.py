#!/usr/bin/env python3
"""Calibrate FactorEngine backend costs on the actual deployment host.

This command is intentionally offline. Production requests must never execute all
backends just to choose one. Run this after deployment/runtime upgrades, then
commit or distribute ``benchmarks/backend_cost_baseline.json`` with the exact
runtime family. ``backend.type=auto`` consumes the file only when
``provenance.measured=true`` and the software family matches.

Examples
--------
python factor_engine/scripts/calibrate_backend_costs.py
python factor_engine/scripts/calibrate_backend_costs.py --rows 20000 100000 --repeats 5
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

FE_ROOT = Path(__file__).resolve().parents[1]
for p in (str(FE_ROOT.parent), str(FE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = FE_ROOT / "benchmarks" / "backend_cost_baseline.json"


def _runtime_family() -> dict[str, str]:
    out = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for module in ("numpy", "pandas", "polars", "duckdb", "pyarrow"):
        try:
            mod = __import__(module)
            out[module] = ".".join(str(getattr(mod, "__version__", "")).split(".")[:2])
        except Exception:
            out[module] = "missing"
    out["os"] = platform.system()
    out["architecture"] = platform.machine()
    out["processor"] = platform.processor()
    return out


def _median_ms(fn: Callable[[], Any], repeats: int) -> float:
    fn()  # warmup/import/JIT allocation is not steady-state kernel cost
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return float(statistics.median(samples))


def _fit_cost(points: list[tuple[int, float]], backend: str) -> dict[str, Any]:
    if not points:
        return {"error": "no successful samples"}
    if len(points) == 1:
        rows, ms = points[0]
        per_million = ms / max(rows / 1_000_000.0, 0.001)
        fixed = 0.0
    else:
        xs = np.array([rows / 1_000_000.0 for rows, _ in points], dtype=float)
        ys = np.array([ms for _, ms in points], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        per_million = max(0.0, float(slope))
        fixed = max(0.0, float(intercept))
    return {
        "fixed_overhead_ms": round(fixed, 6),
        "per_million_rows_ms": round(per_million, 6),
        "memory_factor": 1.0,
        "requires_conversion": backend in {"polars_panel", "polars_long"},
        "samples": [{"rows": int(r), "p50_ms": round(ms, 6)} for r, ms in points],
    }


def _shape_for_rows(rows: int) -> tuple[int, int]:
    # Cross-sectional operators need a non-trivial width while TS operators need
    # enough dates for 20/60-day windows.  Keep width bounded to avoid huge Python
    # object overhead in the calibration harness itself.
    instruments = min(256, max(8, int(math.sqrt(max(rows, 1)))))
    dates = max(96, int(math.ceil(rows / instruments)))
    return dates, instruments


def _pandas_case(canonical: str, rows: int):
    from scripts.audit_all_factor_production import _build_call, _panels
    from cleaned_operators.registry import OperatorRegistry

    dates, instruments = _shape_for_rows(rows)
    panels = _panels(rows=dates, cols=instruments)
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    if op is None:
        raise RuntimeError("no pandas backend")
    args, kwargs = _build_call(canonical, op, panels)
    return lambda: op.calculate(*args, **kwargs), dates * instruments


def _polars_panel_case(canonical: str, rows: int):
    from scripts.audit_all_factor_production import _build_call, _panels
    from cleaned_operators.registry import OperatorRegistry
    from backend.panel_polars import panel_to_polars

    dates, instruments = _shape_for_rows(rows)
    panels = _panels(rows=dates, cols=instruments)
    op = OperatorRegistry.get(canonical, "polars")
    if op is None:
        raise RuntimeError("no polars backend")
    p_op = OperatorRegistry.get(canonical, "pandas_numpy") or op
    args, kwargs = _build_call(canonical, p_op, panels)

    def convert(value: Any) -> Any:
        return panel_to_polars(value) if isinstance(value, pd.DataFrame) else value

    p_args = [convert(x) for x in args]
    p_kwargs = {k: convert(v) for k, v in kwargs.items()}
    return lambda: op.calculate(*p_args, **p_kwargs), dates * instruments


def _minimal_plan(canonical: str):
    from backend.sql_pushdown.plan_fixtures import minimal_plan
    return minimal_plan(canonical)


def _polars_long_case(canonical: str, rows: int):
    from backend.fastpath_plan_probe import ProbeSchemaBuilder
    from backend.polars_expr_emitter import compile_plan_to_polars

    plan = _minimal_plan(canonical)
    instruments = min(128, max(4, int(math.sqrt(max(rows, 1)))))
    probe = ProbeSchemaBuilder.build_polars_probe_frame(
        plan,
        n_rows=max(rows, instruments),
        n_instruments=instruments,
    )

    def run():
        compiled = compile_plan_to_polars(plan, probe, ctx=None)
        if compiled is None:
            raise RuntimeError("compile_plan_to_polars returned None")
        return compiled.frame.collect()

    return run, max(rows, instruments)


def _duckdb_case(canonical: str, rows: int):
    import duckdb
    from backend.fastpath_plan_probe import ProbeSchemaBuilder
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = _minimal_plan(canonical)
    instruments = min(128, max(4, int(math.sqrt(max(rows, 1)))))
    dataset = "__backend_calibration__"
    compiled = compile_plan_to_sql(
        plan,
        dataset=dataset,
        time_column="ts",
        instrument_column="inst",
        dialect=SqlDialect.DUCKDB,
    )
    if compiled is None:
        raise RuntimeError("compile_plan_to_sql returned None")
    table = ProbeSchemaBuilder.build_duckdb_probe_table(
        plan,
        n_rows=max(rows, instruments),
        n_instruments=instruments,
    )
    sql = str(compiled.query or "").replace(f"{{{{{dataset}}}}}", dataset)

    def run():
        con = duckdb.connect()
        try:
            con.register(dataset, table)
            return con.execute(sql).fetch_arrow_table()
        finally:
            con.close()

    return run, len(table)


def _benchmark_backend(
    canonical: str,
    backend: str,
    sizes: list[int],
    repeats: int,
) -> dict[str, Any]:
    builders = {
        "pandas_numpy": _pandas_case,
        "polars_panel": _polars_panel_case,
        "polars_long": _polars_long_case,
        "duckdb_sql": _duckdb_case,
    }
    points: list[tuple[int, float]] = []
    try:
        for requested in sizes:
            run, actual_rows = builders[backend](canonical, requested)
            points.append((actual_rows, _median_ms(run, repeats)))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return _fit_cost(points, backend)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[20_000, 100_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    sizes = sorted({int(x) for x in args.rows if int(x) > 0})
    if not sizes:
        raise SystemExit("--rows must contain positive sizes")

    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends
    from cleaned_operators.production_hardening import factor_production_targets
    from backend.operator_capability import supports_pandas, supports_polars, supports_sql
    from backend.polars_long_production import is_polars_long_native_production_safe

    load_all()
    register_sql_backends()
    operators: dict[str, dict[str, Any]] = {}
    for canonical in sorted(factor_production_targets()):
        row: dict[str, Any] = {}
        if supports_pandas(canonical, mode="production"):
            row["pandas_numpy"] = _benchmark_backend(canonical, "pandas_numpy", sizes, args.repeats)
        if supports_polars(canonical, mode="production"):
            row["polars_panel"] = _benchmark_backend(canonical, "polars_panel", sizes, args.repeats)
        if is_polars_long_native_production_safe(canonical):
            row["polars_long"] = _benchmark_backend(canonical, "polars_long", sizes, args.repeats)
        if supports_sql(canonical, data_source_kind="duckdb", mode="production"):
            row["duckdb_sql"] = _benchmark_backend(canonical, "duckdb_sql", sizes, args.repeats)
        operators[canonical] = row
        print(canonical, {k: "ok" if "error" not in v else v["error"] for k, v in row.items()})

    payload = {
        "schema_version": 2,
        "generated_by": "calibrate_backend_costs.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "measured": True,
            "runtime_family": _runtime_family(),
            "requested_rows": sizes,
            "repeats": int(args.repeats),
            "note": "deployment-host steady-state p50 calibration; rerun after hardware/runtime changes",
        },
        "operators": operators,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote measured backend cost baseline: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
