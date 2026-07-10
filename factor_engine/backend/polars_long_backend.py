# -*- coding: utf-8
"""Polars long-table native 后端：不经过 wide panel ↔ polars 往返。

用法：``build_backend('polars_long')`` 或 YAML ``backend.type: polars_long``。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from planner.logical_plan import PlanNode

from .base import Backend
from .context import ExecutionContext
from .panel_native import finalize_panel_result
from .polars_backend import PolarsBackend
from .polars_long_policy import (
    PolarsLongStrictError,
    assert_native_only_plan,
    collect_plan_op_stats,
    long_path_telemetry_flags,
    strict_polars_long_fallback,
)


def _record_long_stats(
    ctx: ExecutionContext,
    *,
    used_long_path: bool,
    telemetry: dict[str, bool] | None = None,
    fallback_reason: str | None = None,
    fallback_op: str | None = None,
    fallback_exc_type: str | None = None,
    columns: list[str] | None = None,
    op_stats: dict[str, list[str]] | None = None,
) -> None:
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["used_polars_long_path"] = used_long_path
    if telemetry:
        runtime.update(telemetry)
    elif not used_long_path:
        runtime["used_polars_long_native"] = False
    if columns is not None:
        runtime["polars_long_columns"] = columns
    if op_stats:
        runtime.update(op_stats)
    if fallback_reason:
        runtime["polars_long_fallback_reason"] = fallback_reason
    if fallback_op:
        runtime["polars_long_fallback_plan_op"] = fallback_op
    if fallback_exc_type:
        runtime["polars_long_fallback_exception_type"] = fallback_exc_type
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]
    from backend.runtime_events import append_runtime_event

    if used_long_path:
        append_runtime_event(
            ctx,
            "polars_long_native",
            backend=str(runtime.get("backend") or "polars_long"),
            columns=columns,
            **(telemetry or {}),
            **{k: v for k, v in (op_stats or {}).items() if k.endswith("_ops")},
        )
    elif fallback_reason:
        append_runtime_event(
            ctx,
            "polars_long_fallback",
            backend=str(runtime.get("backend") or "polars_long"),
            reason=fallback_reason,
            op=fallback_op,
            error_type=fallback_exc_type,
        )


def _fallback_or_raise(
    backend: PolarsBackend,
    plan: PlanNode,
    ctx: ExecutionContext,
    root_ctx: ExecutionContext,
    *,
    reason: str,
    exc: Exception | None = None,
) -> Any:
    if strict_polars_long_fallback(ctx):
        msg = f"polars_long strict: {reason}"
        if exc is not None:
            raise PolarsLongStrictError(msg) from exc
        raise PolarsLongStrictError(msg)
    _record_long_stats(
        ctx,
        used_long_path=False,
        fallback_reason=reason,
        fallback_op=plan.op,
        fallback_exc_type=type(exc).__name__ if exc is not None else None,
    )
    root_ctx.runtime_stats = ctx.runtime_stats  # type: ignore[attr-defined]
    return backend.execute(plan, ctx)


def _fresh_long_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    """每次 ``execute`` 开始时清除上一轮 long-path 标记，避免 batch 污染。"""
    out = dict(runtime)
    for key in (
        "used_polars_long_path",
        "used_polars_long_native",
        "used_polars_long_map_groups",
        "used_polars_long_registry",
        "used_polars_long_passthrough",
        "polars_long_columns",
        "polars_long_native_ops",
        "polars_long_map_group_ops",
        "polars_long_registry_ops",
        "polars_long_passthrough_ops",
        "polars_long_other_ops",
        "polars_long_fallback_reason",
        "polars_long_fallback_plan_op",
        "polars_long_fallback_exception_type",
    ):
        out.pop(key, None)
    return out


class PolarsLongBackend(Backend):
    """Long-table native Polars：``scan_polars_long`` → expr emitter → 一次 collect。"""

    runtime_backend_label = "polars_long"

    prefers_native_scan = True
    supports_lazy_shared = True

    def __init__(self) -> None:
        self._fallback = PolarsBackend()

    def compile_lazy_shared(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        sid: str,
    ) -> bool:
        """CSE 共享子树：编译 LazyFrame 写入 ``shared_long_lazy_cache``，不 collect。"""
        return self._compile_lazy_shared_impl(plan, ctx, sid=sid)

    def compile_lazy(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        sid: str | None = None,
    ) -> bool:
        """别名：``compile_lazy_shared``（run_many CSE lazy-only）。"""
        if sid is None:
            return False
        return self.compile_lazy_shared(plan, ctx, sid=sid)

    def _compile_lazy_shared_impl(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        sid: str,
    ) -> bool:
        scan_fn = getattr(ctx.data_source, "scan_polars_long", None) if ctx.data_source else None
        if not callable(scan_fn):
            return False

        from .polars_expr_emitter import (
            compile_polars_long_lazy,
            plan_is_polars_long_capable,
            resolve_base_lazy_for_plan,
        )

        if not plan_is_polars_long_capable(plan):
            return False

        if plan.op == "literal":
            return False

        if plan.op == "column":
            import polars as pl

            from .polars_expr_emitter import _INST, _TS, _VAL

            name = str(plan.attrs.get("name") or "")
            if not name:
                return False
            lf = scan_fn([name]).select(
                pl.col(_TS),
                pl.col(_INST),
                pl.col(name).alias(_VAL),
            )
            cache = getattr(ctx, "shared_long_lazy_cache", None)
            if cache is not None:
                cache[str(sid)] = lf
            return True

        try:
            base_lf = resolve_base_lazy_for_plan(plan, ctx, scan_fn)
            compile_polars_long_lazy(plan, ctx, base_lf=base_lf, lazy_cache_key=str(sid))
            return True
        except Exception as exc:
            runtime = dict(getattr(ctx, "runtime_stats", None) or {})
            failures = list(runtime.get("shared_lazy_compile_failures") or [])
            failures.append(
                {
                    "sid": str(sid),
                    "op": str(plan.op),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                }
            )
            runtime["shared_lazy_compile_failures"] = failures
            runtime["shared_lazy_compile_failed_sid"] = str(sid)
            runtime["shared_lazy_compile_failed_op"] = str(plan.op)
            runtime["shared_lazy_compile_error_type"] = type(exc).__name__
            ctx.runtime_stats = runtime  # type: ignore[attr-defined]
            from backend.runtime_events import append_runtime_event

            append_runtime_event(
                ctx,
                "shared_lazy_compile_failed",
                backend="polars_long",
                sid=str(sid),
                op=str(plan.op),
                error_type=type(exc).__name__,
                error=str(exc)[:300],
            )
            return False

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        root_ctx = ctx
        runtime = _fresh_long_runtime(dict(getattr(ctx, "runtime_stats", None) or {}))
        if str(runtime.get("backend") or "") not in {"hybrid_long", "auto_long"}:
            runtime["backend"] = "polars_long"
        ctx = replace(ctx, runtime_stats=runtime)

        scan_fn = getattr(ctx.data_source, "scan_polars_long", None) if ctx.data_source else None
        if not callable(scan_fn):
            return _fallback_or_raise(
                self._fallback,
                plan,
                ctx,
                root_ctx,
                reason="data_source lacks scan_polars_long",
            )

        from .polars_expr_emitter import (
            collect_columns,
            execute_polars_long_plan,
            plan_is_polars_long_capable,
            resolve_base_lazy_for_plan,
        )

        if not plan_is_polars_long_capable(plan):
            return _fallback_or_raise(
                self._fallback,
                plan,
                ctx,
                root_ctx,
                reason="plan not in POLARS_LONG_CAPABLE",
            )

        try:
            columns = sorted(collect_columns(plan))
            base_lf = resolve_base_lazy_for_plan(plan, ctx, scan_fn)
            runtime = dict(getattr(ctx, "runtime_stats", None) or {})
            sid = runtime.get("polars_long_shared_sid")
            lazy_key = str(sid) if sid is not None else None
            result = execute_polars_long_plan(
                plan,
                ctx,
                base_lf=base_lf,
                lazy_cache_key=lazy_key,
            )
            op_stats = collect_plan_op_stats(plan)
            assert_native_only_plan(op_stats, ctx)
            telemetry = long_path_telemetry_flags(op_stats)
            _record_long_stats(
                ctx,
                used_long_path=True,
                telemetry=telemetry,
                columns=columns,
                op_stats=op_stats,
            )
            root_ctx.runtime_stats = ctx.runtime_stats  # type: ignore[attr-defined]
            result = finalize_panel_result(result, ctx)
            from runtime.production_policy import assert_no_production_pandas_fallbacks

            assert_no_production_pandas_fallbacks(ctx, context="polars_long_execute")
            return result
        except PolarsLongStrictError:
            raise
        except Exception as exc:
            return _fallback_or_raise(
                self._fallback,
                plan,
                ctx,
                root_ctx,
                reason=str(exc),
                exc=exc,
            )
