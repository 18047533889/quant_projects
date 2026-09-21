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


def _deployment_identity() -> dict[str, Any]:
    """Deployment identity recorded alongside the measured costs.

    Taken from the router's own resolver because ``_measured_baseline`` compares
    these keys verbatim against the live host — re-deriving them here would drift
    the moment either side moves.  The multibackend cost calibrator supplies the
    per-backend runtime versions and the engine build identity, which is what
    binds the numbers to a specific build.
    """
    from factor_engine.backend.plan_cost_router import _runtime_family

    family = dict(_runtime_family())
    versions: dict[str, str] = {}
    engine_identity = "unknown"
    try:
        from factor_engine.runtime.multibackend.cost_model_calibrator import (
            _backend_version,
            _build_identity,
        )

        for backend in ("pandas_numpy", "polars_panel", "polars_long", "duckdb_sql"):
            versions[backend] = _backend_version(backend)
        engine_identity = _build_identity()
    except Exception as exc:  # noqa: BLE001 - identity is still recorded without it
        versions["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "runtime_family": family,
        "backend_versions": versions,
        "engine_identity": engine_identity,
    }


def _agent_usable_canonicals() -> frozenset[str]:
    """The agent-usable authoring surface (the reviewed daily surface).

    Calibration follows the surface agents actually author against (the daily
    allowlist alphaprobe consumes), not the narrower production-target set: a
    measured cost for an operator an agent can use is useful before that operator
    is production-hardened.
    """
    from factor_engine.cleaned_operators.operator_surface import (
        REVIEWED_MIGRATION_MANIFEST,
    )

    return frozenset(REVIEWED_MIGRATION_MANIFEST)


def _implementation_hash(canonicals: list[str]) -> str:
    """Bind the baseline to the implementations it measured.

    The cost authority refuses a baseline without this, and its purpose is that
    changing an operator's implementation invalidates the measured cost instead of
    silently reusing it.  Digests are taken in ``mode="any"`` so agent-usable
    operators that are not yet production-hardened are still bound.
    """
    import hashlib

    from factor_engine.backend.evidence_provenance import _source_hash
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    parts: list[str] = []
    for canonical in sorted(canonicals):
        digests: list[str] = []
        for backend in ("pandas_numpy", "polars"):
            implementation = OperatorRegistry.get(canonical, backend, mode="any")
            if implementation is None:
                continue
            source_file = getattr(implementation.__class__, "__module__", "") or ""
            try:
                module = __import__(source_file, fromlist=["*"])
            except ImportError:
                continue
            raw = getattr(module, "__file__", "") or ""
            if not raw or not Path(raw).is_file():
                continue
            digest = _source_hash(Path(raw))
            if digest:
                digests.append(f"{backend}={digest}")
        if digests:
            parts.append(canonical + ":" + ",".join(digests))
    if not parts:
        return ""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


#: window-like integer parameters the fixtures actually passed, so the required
#: ``window_distribution`` provenance field is a measurement, not a placeholder.
_WINDOW_CAPTURE: dict[str, int] = {}
_WINDOW_TOKENS = ("window", "period", "lookback", "days", "span", "lag", "d")


def _bump(value: int) -> None:
    _WINDOW_CAPTURE[str(value)] = _WINDOW_CAPTURE.get(str(value), 0) + 1


def _capture_windows(values: Any, positional: Any = None) -> None:
    """Record the integer parameters the fixture actually passed.

    Windows reach a kernel two ways: as a named kwarg, and positionally — the
    audit fixture passes an operator's declared params in order, so the positional
    integers ARE the declared integer parameters.  Scanning both keeps the
    required ``window_distribution`` field a measurement, not a placeholder.
    """
    if isinstance(values, dict):
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if any(token in str(key).lower() for token in _WINDOW_TOKENS):
                _bump(value)
    if isinstance(positional, (list, tuple)):
        for value in positional:
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            _bump(value)


def _window_distribution() -> dict[str, Any]:
    return {
        "source": "fixture positional args + window-like kwargs",
        "histogram": dict(sorted(_WINDOW_CAPTURE.items())),
    }


def _shape_summary(sizes: list[int]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for size in sizes:
        dates, instruments = _shape_for_rows(int(size))
        out.append(
            {
                "requested_rows": int(size),
                "dates": int(dates),
                "instruments": int(instruments),
            }
        )
    return out


def _build_provenance(
    sizes: list[int], canonicals: list[str], repeats: int
) -> dict[str, Any]:
    """Provenance the cost authority requires before it will trust the baseline.

    ``operator_cost._BENCHMARK_PROVENANCE_REQUIRED`` discards a baseline missing
    any of its fields, and ``plan_cost_router._measured_baseline`` only accepts one
    whose recorded runtime family matches the live host.  Both requirements are
    encoded here so the written file is usable immediately.
    """
    identity = _deployment_identity()
    family = identity["runtime_family"]
    shapes = _shape_summary(sizes)
    return {
        "measured": True,
        "implementation_hash": _implementation_hash(canonicals),
        "runtime_family": family,
        "backend_versions": identity["backend_versions"],
        "engine_identity": identity["engine_identity"],
        "cpu_model": str(family.get("cpu_model") or platform.processor() or "unknown"),
        "ram_gb": float(family.get("ram_gb") or 0.0),
        "rows": shapes,
        "columns": [shape["instruments"] for shape in shapes],
        "window_distribution": _window_distribution(),
        "null_rate": 0.0,
        "group_cardinality": 1,
        "requested_rows": [int(size) for size in sizes],
        "repeats": int(repeats),
        "surface": "daily (agent-usable)",
        "canonical_count": len(canonicals),
        "note": (
            "deployment-host steady-state p50 calibration over the agent-usable "
            "daily surface; rerun after hardware/runtime changes"
        ),
    }


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
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    dates, instruments = _shape_for_rows(rows)
    panels = _panels(rows=dates, columns=instruments)
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    if op is None:
        raise RuntimeError("no pandas backend")
    args, kwargs = _build_call(canonical, op, panels)
    _capture_windows(kwargs, args)
    return lambda: op.calculate(*args, **kwargs), dates * instruments


def _polars_panel_case(canonical: str, rows: int):
    from scripts.audit_all_factor_production import _build_call, _panels
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.backend.panel_polars import panel_to_polars

    dates, instruments = _shape_for_rows(rows)
    panels = _panels(rows=dates, columns=instruments)
    op = OperatorRegistry.get(canonical, "polars")
    if op is None:
        raise RuntimeError("no polars backend")
    p_op = OperatorRegistry.get(canonical, "pandas_numpy") or op
    args, kwargs = _build_call(canonical, p_op, panels)
    _capture_windows(kwargs, args)

    def convert(value: Any) -> Any:
        return panel_to_polars(value) if isinstance(value, pd.DataFrame) else value

    p_args = [convert(x) for x in args]
    p_kwargs = {k: convert(v) for k, v in kwargs.items()}
    return lambda: op.calculate(*p_args, **p_kwargs), dates * instruments


def _minimal_plan(canonical: str):
    from factor_engine.backend.sql_pushdown.plan_fixtures import minimal_plan
    return minimal_plan(canonical)


def _polars_long_case(canonical: str, rows: int):
    from factor_engine.backend.fastpath_plan_probe import ProbeSchemaBuilder
    from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars

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


# R58: one pooled DuckDB connection + its prepared-statement cache is reused for
# the entire calibration run (calibration is a single-threaded one-shot script).
# This removes the per-call duckdb.connect() fixed cost (~10-18 ms) that used to
# inflate the duckdb_sql backend-cost measurement to ~9000 ms/M.
_CALIB_DUCKDB_HANDLE = None


def _get_calib_duckdb_handle():
    global _CALIB_DUCKDB_HANDLE
    if _CALIB_DUCKDB_HANDLE is None:
        from factor_engine.backend.sql_pushdown.duckdb_connection_pool import (
            get_shared_duckdb_pool,
        )
        _CALIB_DUCKDB_HANDLE = get_shared_duckdb_pool().acquire()
    return _CALIB_DUCKDB_HANDLE


def _duckdb_case(canonical: str, rows: int):
    from factor_engine.backend.fastpath_plan_probe import ProbeSchemaBuilder
    from factor_engine.backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

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
        # R58: reuse a single pooled connection; re-register the dataset view on
        # every case so different canonicals never alias on the shared connection.
        handle = _get_calib_duckdb_handle()
        handle.register(dataset, table)
        return handle.execute(sql).to_arrow_table()

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
    parser.add_argument(
        "--limit", type=int, default=0,
        help="calibrate only the first N canonicals (smoke runs)",
    )
    parser.add_argument(
        "--canonicals", default="",
        help="comma-separated canonical allowlist (smoke runs)",
    )
    args = parser.parse_args()
    sizes = sorted({int(x) for x in args.rows if int(x) > 0})
    if not sizes:
        raise SystemExit("--rows must contain positive sizes")

    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends
    from factor_engine.backend.operator_capability import supports_pandas, supports_polars, supports_sql
    from factor_engine.backend.polars_long_production import polars_long_production_tier

    load_all()
    register_sql_backends()
    operators: dict[str, dict[str, Any]] = {}
    canonicals = sorted(_agent_usable_canonicals())
    if args.canonicals:
        wanted = {c.strip() for c in args.canonicals.split(",") if c.strip()}
        canonicals = [c for c in canonicals if c in wanted]
    if args.limit and int(args.limit) > 0:
        canonicals = canonicals[: int(args.limit)]
    print(f"calibrating {len(canonicals)} agent-usable canonicals")
    for canonical in canonicals:
        row: dict[str, Any] = {}
        if supports_pandas(canonical, mode="research"):
            row["pandas_numpy"] = _benchmark_backend(canonical, "pandas_numpy", sizes, args.repeats)
        if supports_polars(canonical, mode="research"):
            row["polars_panel"] = _benchmark_backend(canonical, "polars_panel", sizes, args.repeats)
        if polars_long_production_tier(canonical) != "unsupported":
            row["polars_long"] = _benchmark_backend(canonical, "polars_long", sizes, args.repeats)
        if supports_sql(canonical, data_source_kind="duckdb", mode="research"):
            row["duckdb_sql"] = _benchmark_backend(canonical, "duckdb_sql", sizes, args.repeats)
        operators[canonical] = row
        print(canonical, {k: "ok" if "error" not in v else v["error"] for k, v in row.items()})

    payload = {
        "schema_version": 2,
        "generated_by": "calibrate_backend_costs.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": _build_provenance(sizes, canonicals, int(args.repeats)),
        "operators": operators,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote measured backend cost baseline: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
