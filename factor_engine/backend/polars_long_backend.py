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


def _record_long_stats(
    ctx: ExecutionContext,
    *,
    used_native: bool,
    fallback_reason: str | None = None,
    fallback_op: str | None = None,
) -> None:
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["used_polars_long_native"] = used_native
    if used_native:
        runtime["polars_long_native_ops"] = runtime.get("polars_long_native_ops") or []
    if fallback_reason:
        runtime["polars_long_fallback_reason"] = fallback_reason
    if fallback_op:
        runtime["polars_long_fallback_plan_op"] = fallback_op
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


class PolarsLongBackend(Backend):
    """Long-table native Polars：``scan_polars_long`` → expr emitter → 一次 collect。"""

    def __init__(self) -> None:
        self._fallback = PolarsBackend()

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        root_ctx = ctx
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        runtime["backend"] = "polars_long"
        ctx = replace(ctx, runtime_stats=runtime)

        scan_fn = getattr(ctx.data_source, "scan_polars_long", None) if ctx.data_source else None
        if not callable(scan_fn):
            _record_long_stats(
                ctx,
                used_native=False,
                fallback_reason="data_source lacks scan_polars_long",
                fallback_op=plan.op,
            )
            root_ctx.runtime_stats = ctx.runtime_stats  # type: ignore[attr-defined]
            return self._fallback.execute(plan, ctx)

        from .polars_expr_emitter import (
            collect_columns,
            execute_polars_long_plan,
            plan_is_polars_long_capable,
        )

        if not plan_is_polars_long_capable(plan):
            _record_long_stats(
                ctx,
                used_native=False,
                fallback_reason="plan not in POLARS_LONG_CAPABLE",
                fallback_op=plan.op,
            )
            root_ctx.runtime_stats = ctx.runtime_stats  # type: ignore[attr-defined]
            return self._fallback.execute(plan, ctx)

        try:
            columns = collect_columns(plan)
            base_lf = scan_fn(sorted(columns))
            result = execute_polars_long_plan(plan, ctx, base_lf=base_lf)
            _record_long_stats(ctx, used_native=True)
            runtime = dict(getattr(ctx, "runtime_stats", None) or {})
            runtime["polars_long_native_ops"] = sorted(columns)
            ctx.runtime_stats = runtime  # type: ignore[attr-defined]
            root_ctx.runtime_stats = runtime  # type: ignore[attr-defined]
            result = finalize_panel_result(result, ctx)
            from runtime.production_policy import assert_no_production_pandas_fallbacks

            assert_no_production_pandas_fallbacks(ctx, context="polars_long_execute")
            return result
        except Exception as exc:
            _record_long_stats(
                ctx,
                used_native=False,
                fallback_reason=str(exc),
                fallback_op=plan.op,
            )
            root_ctx.runtime_stats = ctx.runtime_stats  # type: ignore[attr-defined]
            return self._fallback.execute(plan, ctx)
