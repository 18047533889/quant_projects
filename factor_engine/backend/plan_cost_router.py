# -*- coding: utf-8 -*-
"""Whole-plan backend routing over production-certified candidate plans.

The optimizer never executes multiple backends just to discover the fastest one.
Instead it selects among eligible physical plans using an offline measured cost
baseline when that baseline matches the runtime environment, otherwise a clearly
labelled conservative estimate. Wide Polars and Polars-long are separate plans:
not every production-safe Polars operator is long-native.
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
    backend: str  # pandas_numpy | polars_panel | polars_long | duckdb_sql | hybrid
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
    # Phase 5 P1-2：服务器硬件 fingerprint——不同机器（16核32GB vs 64核512GB、
    # 本地NVMe vs COS远程）的最优 backend 不同，不能共享同一 measured baseline。
    try:
        from runtime.resource_governor import (
            effective_cpu_slots,
            effective_memory_limit_bytes,
            spill_disk_speed_class,
        )

        out["cpu_model"] = platform.processor() or "unknown"
        out["effective_cores"] = str(effective_cpu_slots())
        ram = effective_memory_limit_bytes()
        out["ram_gb"] = str(round(ram / 1024**3, 1))
        out["storage_class"] = spill_disk_speed_class()
    except Exception:
        pass
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
    # 审计 #353：provenance 必须携带实现 hash，否则无法对应到当前算子实现，
    # 即使运行时版本匹配也不视为 measured baseline。
    if not provenance.get("implementation_hash"):
        return payload, False
    recorded = provenance.get("runtime_family") or provenance.get("runtime_versions") or {}
    if not isinstance(recorded, dict) or not recorded:
        return payload, False
    current = _runtime_family()
    for key in ("python", "numpy", "pandas", "polars", "duckdb"):
        expected = str(recorded.get(key, ""))
        expected = ".".join(expected.split(".")[:2])
        if expected and expected != current.get(key):
            return payload, False
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
    found = False

    try:
        # 审计 #392：三态判定——looks_like_source_ref 只判前缀形态；
        # payload 损坏时 decode_source_ref_strict 抛 ValueError，不吞异常。
        from api.source_ref import decode_source_ref_strict, looks_like_source_ref
    except ImportError:
        # 另一个代理尚未落 api/source_ref.py 的新原语：回退到旧 decode_source_ref，
        # 损坏时仍 re-raise（与旧行为一致）。
        from api.source_ref import decode_source_ref

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

    def walk(node: PlanNode) -> None:
        nonlocal found
        if found:
            return
        if node.op == "column":
            name = str((node.attrs or {}).get("name") or "")
            if name and looks_like_source_ref(name):
                decode_source_ref_strict(name)  # payload 损坏则抛 ValueError，不吞
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
        # 审计 #355：优先用数据源显式声明的 capabilities，避免用 class name 猜
        # backend 而误判 wrapper / custom source。
        caps = getattr(ds, "capabilities", None)
        if caps is not None:
            engine = getattr(caps, "engine_kind", None)
            if isinstance(engine, str) and engine:
                kind = engine.lower()
                if "clickhouse" in kind:
                    return "clickhouse"
                if "duckdb" in kind:
                    return "duckdb"
                if "pandas" in kind:
                    return "memory"
            dialect = getattr(caps, "dialect", None)
            if isinstance(dialect, str) and dialect:
                dl = dialect.lower()
                if "clickhouse" in dl:
                    return "clickhouse"
                if "duckdb" in dl:
                    return "duckdb"
            if getattr(caps, "supports_sql_pushdown", False):
                dl = str(getattr(caps, "dialect", "") or "").lower()
                if "clickhouse" in dl:
                    return "clickhouse"
                if "duckdb" in dl:
                    return "duckdb"
        # 回退：class-name 启发（保留现有逻辑）。
        name = type(ds).__name__.lower()
        if "clickhouse" in name:
            return "clickhouse"
        if "duckdb" in name or "dataaccess" in name or "parquet" in name:
            return "duckdb"
        ds = getattr(ds, "inner", None) or getattr(ds, "_inner", None)
    return "memory"


def estimate_plan_rows(ctx: Any) -> int:
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    for key in ("row_count_estimate", "input_row_count", "estimated_rows"):
        try:
            value = int(runtime.get(key) or 0)
            if value > 0:
                return value
        except Exception:
            pass
    ds = getattr(ctx, "data_source", None)
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

    if backend in {"polars_long", "polars_panel"}:
        key = "polars"
    else:
        key = backend
    return estimate_backend_cost(
        canonical,
        key,
        row_count_estimate=rows,
        requires_conversion=backend in {"polars_long", "polars_panel"},
    )


def _candidate_is_measured(ops: tuple[str, ...], backend: str) -> bool:
    payload, compatible = _measured_baseline()
    if not compatible:
        return False
    entries = payload.get("operators") or {}
    keys = (backend,)
    if backend == "polars_long":
        keys = ("polars_long", "polars")
    elif backend == "polars_panel":
        keys = ("polars_panel", "polars")
    for op in ops:
        row = entries.get(op)
        if not isinstance(row, dict):
            return False
        if not any(
            isinstance(row.get(key), dict) and "error" not in row.get(key, {})
            for key in keys
        ):
            return False
    return True


def _execution_memory_budget(ctx: Any) -> int | None:
    """执行期进程内存预算（Phase 5 P1-1）；无显式资源配置返回 ``None``。

    仅当用户显式配置了内存/结果/spill 任一资源（env 或 PerfConfig 字段）时参与
    路由，避免改变默认行为。
    """
    perf = getattr(ctx, "perf", None)
    if perf is None:
        return None
    explicit = (
        getattr(perf, "memory_limit_bytes", None)
        or getattr(perf, "result_budget_bytes", None)
        or getattr(perf, "spill_budget_bytes", None)
    )
    if not explicit:
        return None
    try:
        plan = perf.build_resource_plan()
    except Exception:
        return None
    if plan is None:
        return None
    return int(plan.process_budget_bytes)


def estimate_plan_peak_memory(ops: tuple[str, ...], rows: int) -> int:
    """估算整计划峰值内存（字节）。

    以 float64 panel（8B/格）为基准，乘以算子内存档位放大系数：high×3、
    medium×2、low×1，加一次表示转换缓冲。用于把「峰值内存 > budget」的候选
    backend 移出路由（Phase 5 P1-1）。
    """
    from backend.operator_cost import get_operator_cost

    cells = max(1, rows) * 8
    factor = 1.0
    for op in ops:
        cost = get_operator_cost(op)
        if cost.memory == "high":
            factor += 0.5
        elif cost.memory == "medium":
            factor += 0.25
    # 表示转换 / 中间物化缓冲 ≈ 基准 × 1.5
    return int(cells * factor * 1.5)


def choose_plan_route(plan: PlanNode, ctx: Any) -> PlanRoute:
    from backend.operator_capability import supports_pandas, supports_polars, supports_sql
    from backend.polars_long_production import is_polars_long_native_production_safe

    mode = str(getattr(ctx, "run_mode", "research") or "research").lower()
    production = mode == "production"
    ops = _canonical_ops(plan)
    rows = estimate_plan_rows(ctx)
    source_ref = _contains_source_ref(plan)
    data_kind = _data_source_kind(ctx)
    candidates: dict[str, float] = {}
    mem_budget = _execution_memory_budget(ctx)
    peak = estimate_plan_peak_memory(ops, rows)
    if mem_budget is not None and peak > mem_budget:
        # 显式资源预算下整计划峰值已超预算：无任何候选可安全执行。
        from backend.operator_capability import UnsupportedOperatorBackendError
        raise UnsupportedOperatorBackendError(
            "no eligible physical plan: estimated peak memory "
            f"{peak / 1024**2:.0f} MiB exceeds execution budget {mem_budget / 1024**2:.0f} MiB"
        )

    def _within_budget(est_peak: int) -> bool:
        return mem_budget is None or est_peak <= mem_budget

    pandas_ok = all(supports_pandas(op, mode=mode) for op in ops)
    if pandas_ok and _within_budget(peak):
        candidates["pandas_numpy"] = sum(_cost(op, "pandas_numpy", rows) for op in ops)

    # Wide Polars is valid even when an operator is not Polars-long-native. The
    # plan remains on one wide representation, so conversion is paid once below.
    polars_panel_ok = bool(ops) and not source_ref and all(
        supports_polars(op, mode=mode) for op in ops
    )
    if polars_panel_ok:
        cost = sum(_cost(op, "polars_panel", rows) for op in ops)
        cost += 2.0 + 0.05 * max(rows / 1_000_000.0, 0.001)
        candidates["polars_panel"] = cost

    polars_long_ok = bool(ops) and not source_ref and all(
        is_polars_long_native_production_safe(op) if production else supports_polars(op, mode=mode)
        for op in ops
    )
    if polars_long_ok:
        candidates["polars_long"] = sum(_cost(op, "polars_long", rows) for op in ops)

    sql_ok = bool(ops) and not source_ref and data_kind in {"duckdb", "clickhouse"} and all(
        supports_sql(op, data_source_kind=data_kind, mode=mode) for op in ops
    )
    sql_backend = "clickhouse_sql" if data_kind == "clickhouse" else "duckdb_sql"
    if sql_ok:
        candidates["duckdb_sql"] = sum(_cost(op, sql_backend, rows) for op in ops)

    if data_kind in {"duckdb", "clickhouse"}:
        mixed = 0.0
        mixed_ok = True
        previous_backend: str | None = None
        transitions = 0
        for op in ops:
            per_op: list[tuple[str, float]] = []
            if supports_pandas(op, mode=mode):
                per_op.append(("pandas_numpy", _cost(op, "pandas_numpy", rows)))
            if supports_polars(op, mode=mode):
                per_op.append(("polars_panel", _cost(op, "polars_panel", rows)))
            if not source_ref and supports_sql(op, data_source_kind=data_kind, mode=mode):
                per_op.append(("sql", _cost(op, sql_backend, rows)))
            if not per_op:
                mixed_ok = False
                break
            chosen_backend, chosen_cost = min(per_op, key=lambda item: (item[1], item[0]))
            if previous_backend is not None and chosen_backend != previous_backend:
                transitions += 1
            previous_backend = chosen_backend
            mixed += chosen_cost
        if mixed_ok:
            millions = max(rows / 1_000_000.0, 0.001)
            mixed += transitions * (3.0 + 0.10 * millions)
            candidates["hybrid"] = mixed

    if not candidates:
        from backend.operator_capability import UnsupportedOperatorBackendError
        raise UnsupportedOperatorBackendError(
            "no eligible physical plan for canonicals: " + ", ".join(ops)
        )

    chosen = min(candidates.items(), key=lambda item: (item[1], item[0]))
    basis = (
        "measured"
        if chosen[0] != "hybrid" and _candidate_is_measured(ops, chosen[0])
        else "estimated"
    )
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
