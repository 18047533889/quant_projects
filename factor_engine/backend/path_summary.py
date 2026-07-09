# -*- coding: utf-8
"""Backend 执行路径快照：单因子 / run_many 可观测性。"""
from __future__ import annotations

from typing import Any


def infer_primary_route(runtime: dict[str, Any]) -> str:
    """从 ``runtime_stats`` 推断主执行路径。"""
    if runtime.get("polars_long_fallback_reason") and not runtime.get("used_polars_long_path"):
        backend = str(runtime.get("backend") or "")
        if backend == "polars":
            return "polars_panel_fallback"
        return "pandas_fallback"
    if runtime.get("used_polars_long_native"):
        return "polars_long_native"
    if runtime.get("used_polars_long_map_groups"):
        return "polars_long_map_groups"
    if runtime.get("used_polars_long_registry"):
        return "polars_long_registry"
    if runtime.get("used_polars_long_passthrough"):
        return "polars_long_passthrough"
    if runtime.get("used_polars_long_path"):
        return "polars_long"
    backend = str(runtime.get("backend") or "")
    if backend in {"polars_long", "auto_long", "hybrid_long"}:
        return "polars_long"
    if backend == "polars":
        return "polars_panel"
    if backend in {"duckdb_sql", "sql", "clickhouse_sql"}:
        return "sql"
    if runtime.get("polars_expr"):
        return "polars_expr"
    if runtime.get("polars_expr_fallback"):
        return "polars_expr_fallback"
    return backend or "pandas"


def snapshot_backend_path(runtime: dict[str, Any] | None) -> dict[str, Any]:
    """捕获单次 execute / run 的 backend path 摘要。"""
    r = dict(runtime or {})
    out: dict[str, Any] = {
        "backend": r.get("backend"),
        "primary_route": infer_primary_route(r),
        "used_polars_long_path": bool(r.get("used_polars_long_path")),
        "used_polars_long_native": bool(r.get("used_polars_long_native")),
        "used_polars_long_map_groups": bool(r.get("used_polars_long_map_groups")),
        "used_polars_long_registry": bool(r.get("used_polars_long_registry")),
        "used_polars_long_passthrough": bool(r.get("used_polars_long_passthrough")),
    }
    for key in (
        "polars_long_native_ops",
        "polars_long_map_group_ops",
        "polars_long_registry_ops",
        "polars_long_passthrough_ops",
        "polars_long_other_ops",
        "polars_long_columns",
    ):
        val = r.get(key)
        if val:
            out[key] = val
    reason = r.get("polars_long_fallback_reason")
    if reason:
        out["polars_long_fallback_reason"] = reason
    return out


def snapshot_from_run_output(run_out: dict[str, Any]) -> dict[str, Any]:
    """从 ``FactorEngine.run()`` 返回 dict 提取 path 摘要。"""
    runtime = {
        "backend": run_out.get("backend"),
        "used_polars_long_path": run_out.get("used_polars_long_path"),
        "used_polars_long_native": run_out.get("used_polars_long_native"),
        "used_polars_long_map_groups": run_out.get("used_polars_long_map_groups"),
        "used_polars_long_registry": run_out.get("used_polars_long_registry"),
        "used_polars_long_passthrough": run_out.get("used_polars_long_passthrough"),
        "polars_long_fallback_reason": run_out.get("polars_long_fallback_reason"),
        "polars_expr": run_out.get("polars_expr"),
        "polars_expr_fallback": run_out.get("polars_expr_fallback"),
    }
    return snapshot_backend_path(runtime)


def summarize_batch_backend_paths(paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """run_many 批量 path  rollup。"""
    if not paths:
        return {}
    routes: dict[str, int] = {}
    for p in paths.values():
        route = str(p.get("primary_route") or "unknown")
        routes[route] = routes.get(route, 0) + 1
    return {
        "factor_count": len(paths),
        "primary_route_counts": routes,
        "all_native": all(p.get("used_polars_long_native") for p in paths.values()),
        "any_fallback": any(
            p.get("polars_long_fallback_reason") or p.get("primary_route") == "pandas_fallback"
            for p in paths.values()
        ),
    }
