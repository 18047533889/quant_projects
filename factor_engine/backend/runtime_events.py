# -*- coding: utf-8
"""runtime_stats event log：batch / parallel / nested backend 可汇总 telemetry。"""
from __future__ import annotations

from typing import Any


def _runtime(ctx: Any) -> dict[str, Any]:
    """从执行上下文读取 runtime_stats 副本。"""
    return dict(getattr(ctx, "runtime_stats", None) or {})


def _write_runtime(ctx: Any, runtime: dict[str, Any]) -> None:
    """将 runtime_stats 写回执行上下文。"""
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


def append_runtime_event(
    ctx: Any,
    event_type: str,
    *,
    backend: str | None = None,
    **payload: Any,
) -> None:
    """追加事件并更新 ``latest`` 快照。

    参数:
        ctx: 执行上下文，需支持 ``runtime_stats`` 属性。
        event_type: 事件类型标签。
        backend: 可选 backend 名称。
        **payload: 附加事件字段。
    """
    runtime = _runtime(ctx)
    event: dict[str, Any] = {"type": str(event_type)}
    if backend is not None:
        event["backend"] = backend
    event.update(payload)
    events = list(runtime.get("events") or [])
    events.append(event)
    runtime["events"] = events
    latest = dict(runtime.get("latest") or {})
    latest.update({k: v for k, v in event.items() if k != "type"})
    runtime["latest"] = latest
    _write_runtime(ctx, runtime)


def merge_runtime(ctx: Any, **fields: Any) -> None:
    """合并字段到 runtime 顶层 + ``latest``（兼容旧 dict 写法）。

    参数:
        ctx: 执行上下文。
        **fields: 要合并的 telemetry 字段。
    """
    runtime = _runtime(ctx)
    runtime.update(fields)
    latest = dict(runtime.get("latest") or {})
    latest.update(fields)
    runtime["latest"] = latest
    _write_runtime(ctx, runtime)


def rollup_runtime_fields(ctx: Any | None) -> dict[str, Any]:
    """从 ``latest`` + 顶层 + events 汇总 path summary 用字段。

    参数:
        ctx: 执行上下文；为 ``None`` 时返回空字典。

    返回:
        重建后的完整 runtime 视图。
    """
    if ctx is None:
        return {}
    return rebuild_runtime_from_events(_runtime(ctx))


def rebuild_runtime_from_events(runtime: dict[str, Any] | None) -> dict[str, Any]:
    """从 event log 重建完整 runtime 视图（path summary 主入口）。

    参数:
        runtime: 含 ``events``、``latest`` 及顶层字段的 runtime 字典。

    返回:
        合并事件后的扁平 runtime 字典。
    """
    base = dict(runtime or {})
    out = {k: v for k, v in base.items() if k not in {"events", "latest"}}
    out.update(dict(base.get("latest") or {}))

    events = list(base.get("events") or [])
    native_ops: list[str] = list(out.get("polars_long_native_ops") or [])
    map_ops: list[str] = list(out.get("polars_long_map_group_ops") or [])
    registry_ops: list[str] = list(out.get("polars_long_registry_ops") or [])

    for event in events:
        et = str(event.get("type") or "")
        backend = event.get("backend")
        if backend and not out.get("backend"):
            out["backend"] = backend

        if et == "sql_fully_pushed":
            out["sql_fully_pushed"] = True
            out["used_sql_pushdown"] = True
            out["fully_sql"] = True
        elif et == "sql_partial_pushed":
            out["sql_partial_pushed"] = True
            out["used_sql_pushdown"] = True
        elif et == "polars_long_native":
            out["used_polars_long_path"] = True
            out["used_polars_long_native"] = True
            for key, bucket in (
                ("polars_long_native_ops", native_ops),
                ("polars_long_map_group_ops", map_ops),
                ("polars_long_registry_ops", registry_ops),
            ):
                for op in event.get(key) or []:
                    if op not in bucket:
                        bucket.append(str(op))
        elif et == "polars_long_fallback":
            out["polars_long_fallback_reason"] = event.get("reason") or out.get("polars_long_fallback_reason")
        elif et == "shared_lazy_compile_failed":
            failures = list(out.get("shared_lazy_compile_failures") or [])
            failures.append(
                {
                    "sid": event.get("sid"),
                    "op": event.get("op"),
                    "error_type": event.get("error_type"),
                    "error": event.get("error"),
                }
            )
            out["shared_lazy_compile_failures"] = failures

    if native_ops:
        out["polars_long_native_ops"] = native_ops
    if map_ops:
        out["polars_long_map_group_ops"] = map_ops
    if registry_ops:
        out["polars_long_registry_ops"] = registry_ops

    if events:
        out["runtime_event_count"] = len(events)
        backends = sorted({str(e.get("backend")) for e in events if e.get("backend")})
        if backends:
            out["runtime_event_backends"] = backends
            if len(backends) == 1:
                out.setdefault("backend", backends[0])
    return out
