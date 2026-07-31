# -*- coding: utf-8 -*-
"""Whole-plan backend routing over production-certified candidate plans.

The optimizer never executes multiple backends just to discover the fastest one.
Instead it selects among *eligible* physical plans using an offline measured cost
baseline when that baseline matches the runtime environment, otherwise a clearly
labelled conservative estimate. Per-operator routing remains available inside the
mixed SQL/Python plan, but a fully portable DAG is routed as one unit to avoid
Pandas↔Polars↔SQL conversion thrashing.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import platform
import sys
from pathlib import Path
from typing import Any

from planner.logical_plan import PlanNode


@dataclass(frozen=True)
class PlanRoute:
    backend: str  # pandas_numpy | polars_long | duckdb_sql | hybrid
    estimated_cost: float
    routing_basis: str  # measured | estimated
    candidate_costs: tuple[tuple[str, float], ...]
    row_count_estimate: int
    ops: tuple[str, ...]
    reason: str = ""


_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "backend_cost_baseline.json"
_META_OPS = frozenset({"column", "literal", "plan_ref", "materialized_series"})


def _runtime_family() -> dict[str, str]:
    out = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for module in ("numpy", "pandas", "polars", "duckdb", "pyarrow"):
        try:
            mod = __import__(module)
            version = str(getattr(mod, "__version__", ""))
            out[module] = ".".join(version.split(".")[:2])
        except Exception:
            out[module] = "missing"
    out["os"] = platform.system()
    out["architecture"] = platform.machine()
    return out


def _measured_baseline() -> tuple[dict[str, Any], bool]:
    if not _BENCHMARK_PATH.is_file():
        return {}, False
    try:
        payload = json.loads(_BENCHMARK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}, False
    if str(payload.get("generated_by") or "") == "seed_defaults":
        return payload, False
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    if not provenance.get("measured"):
        return payload, False
    recorded = provenance.get("runtime_family") or provenance.get("runtime_versions") or {}
    if not isinstance(recorded, dict) or not recorded:
        return payload, False
    current = _runtime_family()
    for key in ("python", "numpy", "pandas", "polars", "duckdb"):
        expected = str(recorded.get(key, ""))
        if key != "python":
            expected = ".".join(expected.split(".")[:2])
        else:
            expected = ".".join(expected.split(".")[:2])
        if expected and expected != current.get(key):
            return payload, False
    # Hardware identity is intentionally coarse. A deployment-specific benchmark
    # may include it; absence does not invalidate software-family measurements.
    return payload, True


def _canonical_ops(plan: PlanNode) -> tuple[str, ...]:
    from cleaned_operators.registry import OperatorRegistry

    seen: set[str] = set()
    ordered: list[str] = []

    def walk(node: PlanNode) -> None:
        op = str(getattr(node, "op", "") or "")
        if op and op not in _META_OPS:
            canonical = OperatorRegistry._aliases.get(op, op)
            if canonical not in seen:
                seen.add(canonical)
                ordered.append(canonical)
        for child in getattr(node, "inputs", ()) or ():
            walk(child)

    walk(plan)
    return tuple(ordered)


def _contains_source_ref(plan: PlanNode) -> bool:
    from api.source_ref import decode_source_ref

    found = False

    def walk(node: PlanNode) -> None:
        nonlocal found
        if found:
            return
        if node.op == "column":
            name = str((node.attrs or {}).get("name") or "")
            if name and decode_source_ref(name) is not None:
                found = True
                return
        for child in getattr(node, "inputs", ()) or ():
            walk(child)

    walk(plan)
    return found


def _data_source_kind(ctx: Any) -> str:
    ds = getattr(ctx, "data_source", None)
    seen: set[int] = set()
    while ds is not None and id(ds) not in seen:
        seen.add(id(ds))
        name = type(ds).__name__.lower()
        if "clickhouse" in name:
            return "clickhouse"
        if "duckdb" in name or "dataaccess" in name or "parquet" in name:
            return "duckdb"
        ds = getattr(ds, "inner", None) or getattr(ds, "_inner", None)
    return "memory"


def estimate_plan_rows(ctx: Any) -> int:
    """Estimate long-table rows without forcing a full read."""
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    for key in ("row_count_estimate", "input_row_count", "estimated_rows"):
        try:
            value = int(runtime.get(key) or 0)
            if value > 0:
                return value
        except Exception:
            pass
    ds = getattr(ctx, "data_source", None)
    # unwrap the logical SourceRef facade
    inner = getattr(ds, "inner", None)
    if inner is not None:
        ds = inner
    try:
        filt = getattr(ds, "instrument_filter", None)
        instruments = len(filt) if filt else 3000
        start = getattr(ds, "start_date", None)
        end = getattr(ds, "end_date", None)
        if start and end:
            import pandas as pd
            dates = max(1, len(pd.bdate_range(start, end)))
            return max(1, dates * max(1, instruments))
    except Exception:
        pass
    return 500_000


def _cost(canonical: str, backend: str, rows: int) -> float:
    from backend.operator_cost import estimate_backend_cost

    key = "polars" if backend == "polars_long" else backend
    return estimate_backend_cost(
        canonical,
        key,
        row_count_estimate=rows,
        requires_conversion=(backend == "polars_long"),
    )


def _candidate_is_measured(ops: tuple[str, ...], backend: str) -> bool:
    payload, compatible = _measured_baseline()
    if not compatible:
        return False
    entries = payload.get("operators") or {}
    keys = (backend,)
    if backend == "polars_long":
        keys = ("polars_long", "polars", "polars_panel")
    for op in ops:
        row = entries.get(op)
        if not isinstance(row, dict):
            return False
        if not any(isinstance(row.get(key), dict) and "error" not in row.get(key, {}) for key in keys):
            return False
    return True


def choose_plan_route(plan: PlanNode, ctx: Any) -> PlanRoute:
    """Choose the cheapest certified physical plan for the current DAG."""
    from backend.operator_capability import supports_pandas, supports_polars, supports_sql
    from backend.polars_long_production import is_polars_long_native_production_safe

    mode = str(getattr(ctx, "run_mode", "research") or "research").lower()
    production = mode == "production"
    ops = _canonical_ops(plan)
    rows = estimate_plan_rows(ctx)
    source_ref = _contains_source_ref(plan)
    data_kind = _data_source_kind(ctx)
    candidates: dict[str, float] = {}

    pandas_ok = all(supports_pandas(op, mode=mode) for op in ops)
    if pandas_ok:
        candidates["pandas_numpy"] = sum(_cost(op, "pandas_numpy", rows) for op in ops)

    # Long-native and full SQL are intentionally disallowed for logical SourceRef
    # columns; those have source-specific PIT/join contracts at the Arrow/Pandas boundary.
    polars_ok = bool(ops) and not source_ref and all(
        is_polars_long_native_production_safe(op) if production else supports_polars(op, mode=mode)
        for op in ops
    )
    if polars_ok:
        candidates["polars_long"] = sum(_cost(op, "polars_long", rows) for op in ops)

    sql_ok = bool(ops) and not source_ref and data_kind in {"duckdb", "clickhouse"} and all(
        supports_sql(op, data_source_kind=data_kind, mode=mode) for op in ops
    )
    sql_backend = "clickhouse_sql" if data_kind == "clickhouse" else "duckdb_sql"
    if sql_ok:
        candidates["duckdb_sql"] = sum(_cost(op, sql_backend, rows) for op in ops)

    # Mixed plan: plan-level SQL pushdown plus evidence-constrained per-op routing.
    # It is eligible whenever every operator has at least one physical path and a
    # pushdown-capable source exists. Penalise representation boundaries so a DAG
    # does not oscillate backends merely because individual kernels are faster.
    if data_kind in {"duckdb", "clickhouse"}:
        mixed = 0.0
        mixed_ok = True
        for op in ops:
            per_op: list[float] = []
            if supports_pandas(op, mode=mode):
                per_op.append(_cost(op, "pandas_numpy", rows))
            if supports_polars(op, mode=mode):
                per_op.append(_cost(op, "polars_long", rows))
            if not source_ref and supports_sql(op, data_source_kind=data_kind, mode=mode):
                per_op.append(_cost(op, sql_backend, rows))
            if not per_op:
                mixed_ok = False
                break
            mixed += min(per_op)
        if mixed_ok:
            # one materialization/representation boundary budget per mixed DAG
            mixed += 8.0 + 0.15 * max(rows / 1_000_000.0, 0.001)
            candidates["hybrid"] = mixed

    if not candidates:
        # Fail closed in production; research caller receives a useful capability error.
        from backend.operator_capability import UnsupportedOperatorBackendError
        raise UnsupportedOperatorBackendError(
            "no eligible physical plan for canonicals: " + ", ".join(ops)
        )

    chosen = min(candidates.items(), key=lambda item: (item[1], item[0]))
    # The route is called measured only if the chosen full-plan backend has a
    # measured baseline for every node. Hybrid remains estimated until a dedicated
    # mixed-plan benchmark corpus is available.
    basis = "measured" if chosen[0] != "hybrid" and _candidate_is_measured(ops, chosen[0]) else "estimated"
    return PlanRoute(
        backend=chosen[0],
        estimated_cost=float(chosen[1]),
        routing_basis=basis,
        candidate_costs=tuple(sorted((name, float(cost)) for name, cost in candidates.items())),
        row_count_estimate=rows,
        ops=ops,
        reason=(
            "offline measured baseline" if basis == "measured"
            else "certified capability + workload estimate; no compatible measured baseline"
        ),
    )


def record_plan_route(ctx: Any, route: PlanRoute) -> None:
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["plan_backend_route"] = {
        "backend": route.backend,
        "routing_basis": route.routing_basis,
        "estimated_cost": route.estimated_cost,
        "candidate_costs": dict(route.candidate_costs),
        "row_count_estimate": route.row_count_estimate,
        "ops": list(route.ops),
        "reason": route.reason,
    }
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]
