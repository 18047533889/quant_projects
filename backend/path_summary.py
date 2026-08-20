# -*- coding: utf-8
"""Backend 执行路径快照：单因子 / run_many 可观测性。"""
from __future__ import annotations

from typing import Any


def _fallback_route(runtime: dict[str, Any]) -> str:
    """更细的 fallback 路由分类。"""
    reason = runtime.get("polars_long_fallback_reason")
    if reason:
        if runtime.get("used_polars_long_path"):
            return "polars_long_degraded"
        backend = str(runtime.get("backend") or "")
        if backend in {"polars", "polars_panel"}:
            return "polars_long_to_polars_panel"
        return "polars_long_to_pandas"
    if runtime.get("sql_long_pushdown_failed_sid") or runtime.get("sql_long_pushdown_error_type"):
        if runtime.get("sql_series_subtrees") or runtime.get("used_sql_pushdown"):
            return "sql_long_to_series_sql"
        return "sql_to_python_fallback"
    if runtime.get("production_pandas_fallbacks"):
        backend = str(runtime.get("backend") or "")
        if backend in {"hybrid_long", "duckdb_sql", "sql"}:
            return "sql_to_python_fallback"
        if backend == "polars":
            return "polars_panel_to_pandas"
        return "pandas_fallback"
    return "unknown"


def infer_primary_route(runtime: dict[str, Any]) -> str:
    """从 ``runtime_stats`` 推断主执行路径。"""
    query_count = int(runtime.get("sql_query_count") or 0)
    if runtime.get("sql_fully_pushed") and query_count > 0:
        dialect = str(runtime.get("sql_dialect") or "duckdb").lower()
        if "clickhouse" in dialect:
            return "clickhouse_sql_full"
        return "duckdb_sql_full"
    if runtime.get("sql_full_execution_failed") or (
        runtime.get("fully_sql") and not runtime.get("sql_fully_pushed") and query_count <= 0
    ):
        return "sql_full_execution_failed"
    fb = _fallback_route(runtime)
    if fb not in {"unknown", "pandas_fallback"} and (
        runtime.get("polars_long_fallback_reason")
        or runtime.get("sql_long_pushdown_failed_sid")
        or runtime.get("production_pandas_fallbacks")
        or int(runtime.get("sql_fallback_subtree_count") or 0) > 0
    ):
        return fb
    if runtime.get("polars_long_fallback_reason") and not runtime.get("used_polars_long_path"):
        backend = str(runtime.get("backend") or "")
        if backend == "polars":
            return "polars_long_to_polars_panel"
        return "polars_long_to_pandas"
    if runtime.get("used_sql_pushdown") and query_count > 0:
        if runtime.get("used_polars_long_path"):
            return "sql_partial_polars_long"
        return "sql_partial"
    if runtime.get("fully_sql") or runtime.get("sql_plan_fully_compilable"):
        return "sql_plan_only"
    if runtime.get("used_polars_long_native"):
        return "polars_long_native"
    if runtime.get("used_polars_long_python_rolling"):
        return "polars_long_python_rolling"
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


def _final_collect(runtime: dict[str, Any]) -> str:
    """推断最终结果物化到 Pandas 的 collect 路径。"""
    if runtime.get("sql_full_execution_failed"):
        return "fallback_to_pandas_or_polars"
    if runtime.get("sql_fully_pushed") and int(runtime.get("sql_query_count") or 0) > 0:
        return "sql_to_pandas"
    if runtime.get("fully_sql") and int(runtime.get("sql_query_count") or 0) <= 0:
        return "fallback_to_pandas_or_polars"
    if runtime.get("used_polars_long_path"):
        return "polars_long_to_pandas"
    backend = str(runtime.get("backend") or "")
    if backend == "polars":
        return "polars_panel_to_pandas"
    if backend in {"duckdb_sql", "sql", "clickhouse_sql"}:
        if int(runtime.get("sql_query_count") or 0) > 0:
            return "sql_to_pandas"
        return "fallback_to_pandas_or_polars"
    return "pandas"


def _infer_fastpath_route(runtime: dict[str, Any]) -> str:
    """真快 / 半快 / fallback 粗分类。"""
    if runtime.get("polars_long_fallback_reason") or runtime.get("production_pandas_fallbacks"):
        return "fallback"
    if int(runtime.get("sql_fallback_subtree_count") or 0) > 0:
        return "fallback"
    if runtime.get("sql_full_execution_failed"):
        return "fallback"
    if runtime.get("used_sql_pushdown") and int(runtime.get("sql_query_count") or 0) <= 0:
        return "fallback"
    route = infer_primary_route(runtime)
    if route in {
        "duckdb_sql_full",
        "clickhouse_sql_full",
        "polars_long_native",
        "sql_partial_polars_long",
    }:
        return "fast"
    if route == "polars_long_python_rolling":
        return "slow"
    if route in {"sql_partial", "polars_long", "sql_plan_only"}:
        return "mixed"
    if route in {"polars_long_map_groups", "polars_long_registry", "polars_long_passthrough"}:
        return "slow"
    if route in {"sql_full_execution_failed", "sql_to_python_fallback"}:
        return "fallback"
    return route or "unknown"


def build_backend_path_summary(runtime: dict[str, Any] | None) -> dict[str, Any]:
    """Production 级 path summary JSON。"""
    from backend.runtime_events import rebuild_runtime_from_events

    r = rebuild_runtime_from_events(dict(runtime or {}))
    summary: dict[str, Any] = {
        "backend": r.get("backend"),
        "primary_route": infer_primary_route(r),
        "sql_plan_fully_compilable": bool(r.get("sql_plan_fully_compilable") or r.get("fully_sql")),
        "fully_sql": bool(r.get("fully_sql")),
        "sql_full_execution_failed": bool(r.get("sql_full_execution_failed")),
        "used_sql_pushdown": bool(r.get("used_sql_pushdown")) and int(r.get("sql_query_count") or 0) > 0,
        "sql_subtree_count": int(r.get("sql_subtree_count") or 0),
        "sql_subtrees": list(r.get("sql_subtrees") or []),
        "sql_fully_pushed": bool(r.get("sql_fully_pushed")),
        "sql_partial_pushed": bool(r.get("sql_partial_pushed")),
        "sql_long_lazy_subtrees": list(r.get("sql_long_lazy_subtrees") or []),
        "sql_series_subtrees": list(r.get("sql_series_subtrees") or []),
        "sql_fallback_subtree_count": int(r.get("sql_fallback_subtree_count") or 0),
        "python_fallback_subtree_sids": list(r.get("python_fallback_subtree_sids") or []),
        "sql_dialect": r.get("sql_dialect"),
        "sql_query_count": int(r.get("sql_query_count") or 0),
        "used_polars_long_path": bool(r.get("used_polars_long_path")),
        "used_polars_long_native": bool(r.get("used_polars_long_native")),
        "used_polars_long_python_rolling": bool(r.get("used_polars_long_python_rolling")),
        "used_polars_long_map_groups": bool(r.get("used_polars_long_map_groups")),
        "used_polars_long_registry": bool(r.get("used_polars_long_registry")),
        "polars_long_native_ops": list(r.get("polars_long_native_ops") or []),
        "polars_long_python_rolling_ops": list(r.get("polars_long_python_rolling_ops") or []),
        "polars_long_map_group_ops": list(r.get("polars_long_map_group_ops") or []),
        "polars_long_registry_ops": list(r.get("polars_long_registry_ops") or []),
        "polars_long_passthrough_ops": list(r.get("polars_long_passthrough_ops") or []),
        "polars_long_other_ops": list(r.get("polars_long_other_ops") or []),
        "pandas_fallback_ops": [
            str(x.get("op", "")) for x in (r.get("production_pandas_fallbacks") or []) if isinstance(x, dict)
        ],
        "shared_long_lazy_hits": int(r.get("shared_long_lazy_hits") or 0),
        "production_fastpath_ok": r.get("production_fastpath_ok"),
        "fastpath_route": _infer_fastpath_route(r),
        "final_collect": _final_collect(r),
        "runtime_event_count": int(r.get("runtime_event_count") or 0),
    }
    if r.get("runtime_event_backends"):
        summary["runtime_event_backends"] = list(r["runtime_event_backends"])
    reason = r.get("polars_long_fallback_reason")
    if reason:
        summary["polars_long_fallback_reason"] = reason
    fp_violations = r.get("production_fastpath_violations")
    if fp_violations:
        summary["production_fastpath_violations"] = list(fp_violations)
    for key in (
        "sql_long_pushdown_failed_sid",
        "sql_long_pushdown_error_type",
        "sql_long_pushdown_error_message",
        "shared_lazy_compile_failures",
    ):
        val = r.get(key)
        if val:
            summary[key] = val
    return summary


def summarize_lazy_caches(ctx: Any | None) -> dict[str, Any]:
    """LazyFrame 级缓存快照（CSE / SQL partial pushdown）。"""
    if ctx is None:
        return {}
    shared = getattr(ctx, "shared_long_lazy_cache", None) or {}
    materialized = getattr(ctx, "materialized_long_lazy", None) or {}
    out: dict[str, Any] = {}
    if shared:
        out["shared_long_lazy_sids"] = sorted(str(k) for k in shared.keys())
        out["shared_long_lazy_count"] = len(shared)
    if materialized:
        out["materialized_long_lazy_sids"] = sorted(str(k) for k in materialized.keys())
        out["materialized_long_lazy_count"] = len(materialized)
    hits = int((getattr(ctx, "runtime_stats", None) or {}).get("shared_long_lazy_hits") or 0)
    if hits:
        out["shared_long_lazy_hits"] = hits
    return out


def snapshot_backend_path(runtime: dict[str, Any] | None) -> dict[str, Any]:
    """捕获单次 execute / run 的 backend path 摘要。"""
    from backend.runtime_events import rebuild_runtime_from_events

    r = rebuild_runtime_from_events(dict(runtime or {}))
    out: dict[str, Any] = {
        "backend": r.get("backend"),
        "primary_route": infer_primary_route(r),
        "used_polars_long_path": bool(r.get("used_polars_long_path")),
        "used_polars_long_native": bool(r.get("used_polars_long_native")),
        "used_polars_long_python_rolling": bool(r.get("used_polars_long_python_rolling")),
        "used_polars_long_map_groups": bool(r.get("used_polars_long_map_groups")),
        "used_polars_long_registry": bool(r.get("used_polars_long_registry")),
        "used_polars_long_passthrough": bool(r.get("used_polars_long_passthrough")),
        "used_sql_pushdown": bool(r.get("used_sql_pushdown")),
        "fully_sql": bool(r.get("fully_sql")),
        "sql_subtree_count": int(r.get("sql_subtree_count") or 0),
    }
    for key in (
        "polars_long_native_ops",
        "polars_long_python_rolling_ops",
        "polars_long_map_group_ops",
        "polars_long_registry_ops",
        "polars_long_passthrough_ops",
        "polars_long_other_ops",
        "polars_long_columns",
        "sql_subtrees",
    ):
        val = r.get(key)
        if val:
            out[key] = val
    reason = r.get("polars_long_fallback_reason")
    if reason:
        out["polars_long_fallback_reason"] = reason
    out["backend_path_summary"] = build_backend_path_summary(r)
    return out


def snapshot_from_run_output(run_out: dict[str, Any]) -> dict[str, Any]:
    """从 ``FactorEngine.run()`` 返回 dict 提取 path 摘要。"""
    if run_out.get("backend_path"):
        return dict(run_out["backend_path"])
    runtime = {
        "backend": run_out.get("backend"),
        "used_polars_long_path": run_out.get("used_polars_long_path"),
        "used_polars_long_native": run_out.get("used_polars_long_native"),
        "used_polars_long_python_rolling": run_out.get("used_polars_long_python_rolling"),
        "used_polars_long_map_groups": run_out.get("used_polars_long_map_groups"),
        "used_polars_long_registry": run_out.get("used_polars_long_registry"),
        "used_polars_long_passthrough": run_out.get("used_polars_long_passthrough"),
        "polars_long_fallback_reason": run_out.get("polars_long_fallback_reason"),
        "polars_expr": run_out.get("polars_expr"),
        "polars_expr_fallback": run_out.get("polars_expr_fallback"),
        "used_sql_pushdown": run_out.get("used_sql_pushdown"),
        "fully_sql": run_out.get("fully_sql"),
        "sql_subtree_count": run_out.get("sql_subtree_count"),
        "sql_subtrees": run_out.get("sql_subtrees"),
        "production_pandas_fallbacks": run_out.get("production_pandas_fallbacks"),
        "production_fastpath_ok": run_out.get("production_fastpath_ok"),
        "production_fastpath_violations": run_out.get("production_fastpath_violations"),
        "polars_long_native_ops": run_out.get("polars_long_native_ops"),
        "polars_long_python_rolling_ops": run_out.get("polars_long_python_rolling_ops"),
        "polars_long_map_group_ops": run_out.get("polars_long_map_group_ops"),
        "polars_long_registry_ops": run_out.get("polars_long_registry_ops"),
        "polars_long_passthrough_ops": run_out.get("polars_long_passthrough_ops"),
        "polars_long_columns": run_out.get("polars_long_columns"),
        "sql_fully_pushed": run_out.get("sql_fully_pushed"),
        "sql_partial_pushed": run_out.get("sql_partial_pushed"),
        "sql_long_lazy_subtrees": run_out.get("sql_long_lazy_subtrees"),
        "sql_series_subtrees": run_out.get("sql_series_subtrees"),
        "sql_dialect": run_out.get("sql_dialect"),
    }
    bps = run_out.get("backend_path_summary")
    if bps:
        runtime.update(bps)
    return snapshot_backend_path(runtime)


def summarize_batch_backend_paths(paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """run_many 批量 path rollup。"""
    if not paths:
        return {}
    routes: dict[str, int] = {}
    sql_count = 0
    fastpath_ok = 0
    for p in paths.values():
        route = str(p.get("primary_route") or "unknown")
        routes[route] = routes.get(route, 0) + 1
        if p.get("used_sql_pushdown") or p.get("sql_subtree_count", 0) > 0:
            sql_count += 1
        bps = p.get("backend_path_summary") or {}
        if bps.get("production_fastpath_ok") is True:
            fastpath_ok += 1
    return {
        "factor_count": len(paths),
        "primary_route_counts": routes,
        "all_native": all(p.get("used_polars_long_native") for p in paths.values()),
        "any_fallback": any(
            p.get("polars_long_fallback_reason") or p.get("primary_route") == "pandas_fallback"
            for p in paths.values()
        ),
        "any_sql_pushdown": sql_count > 0,
        "production_fastpath_ok_count": fastpath_ok,
        "shared_long_lazy_hits_total": max(
            (
                int((p.get("backend_path_summary") or {}).get("shared_long_lazy_hits") or 0)
                for p in paths.values()
            ),
            default=0,
        ),
    }
