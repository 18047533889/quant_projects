# -*- coding: utf-8
"""Polars long-table 原生执行后端。

本后端绕过 wide panel ↔ polars 往返，直接以 ``scan_polars_long`` 扫描长表，
通过 ``polars_expr_emitter`` 将逻辑计划编译为 Polars 表达式链并一次 collect。

用法：``build_backend('polars_long')`` 或 YAML ``backend.type: polars_long``。

回退策略
--------
当数据源缺少 ``scan_polars_long``、计划不在 ``POLARS_LONG_CAPABLE`` 集合、
或执行抛出异常时，根据 ``strict_polars_long_fallback`` 配置决定：
严格模式抛 ``PolarsLongStrictError``，否则回退到 ``PolarsBackend``。
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
    """将 long-table 执行路径的遥测信息写入 ``ctx.runtime_stats``。

    参数
    ----
    ctx : ExecutionContext
        目标执行上下文。
    used_long_path : bool
        是否成功走 long native 路径。
    telemetry : dict[str, bool] | None
        long 路径细分标记（native/map_groups/registry 等）。
    fallback_reason : str | None
        回退原因描述。
    fallback_op : str | None
        触发回退的计划算子名。
    fallback_exc_type : str | None
        回退时捕获的异常类型名。
    columns : list[str] | None
        long 路径扫描的列名列表。
    op_stats : dict[str, list[str]] | None
        计划中各类别算子的统计信息。
    """
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
    """long 路径不可用时，按策略回退到 PolarsBackend 或抛出严格模式异常。

    参数
    ----
    backend : PolarsBackend
        回退目标后端实例。
    plan : PlanNode
        待执行的计划根节点。
    ctx : ExecutionContext
        当前执行上下文。
    root_ctx : ExecutionContext
        根执行上下文，用于同步 ``runtime_stats``。
    reason : str
        回退原因描述。
    exc : Exception | None
        触发回退的原始异常（严格模式下作为 ``__cause__`` 链）。

    返回
    ----
    Any
        非严格模式下 ``PolarsBackend.execute`` 的求值结果。

    异常
    ----
    PolarsLongStrictError
        严格模式（``strict_polars_long_fallback``）下禁止回退时抛出。
    """
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
    """每次 ``execute`` 开始时清除上一轮 long-path 标记，避免 batch 污染。

    参数
    ----
    runtime : dict[str, Any]
        当前 ``runtime_stats`` 字典的副本。

    返回
    ----
    dict[str, Any]
        已移除所有 ``polars_long_*`` 相关键的新字典。
    """
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
    """Long-table 原生 Polars 执行后端。

    数据流：``scan_polars_long`` → ``polars_expr_emitter`` 表达式链 → 一次 collect。
    支持 CSE 共享子树的 lazy 编译（``compile_lazy_shared``），
    不 collect 即写入 ``shared_long_lazy_cache``。
    """

    runtime_backend_label = "polars_long"

    prefers_native_scan = True
    supports_lazy_shared = True

    def __init__(self) -> None:
        """初始化 Polars long 后端，并创建 PolarsBackend 作为回退执行器。"""
        self._fallback = PolarsBackend()

    def compile_lazy_shared(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        sid: str,
    ) -> bool:
        """CSE 共享子树：编译 LazyFrame 写入 ``shared_long_lazy_cache``，不 collect。

        参数
        ----
        plan : PlanNode
            共享子树的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文，需提供 ``data_source.scan_polars_long``。
        sid : str
            共享子树标识符，用作 lazy 缓存键。

        返回
        ----
        bool
            编译成功写入缓存为 ``True``，不支持或失败为 ``False``。
        """
        return self._compile_lazy_shared_impl(plan, ctx, sid=sid)

    def compile_lazy(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        sid: str | None = None,
    ) -> bool:
        """``compile_lazy_shared`` 的别名（供 run_many CSE lazy-only 路径调用）。

        参数
        ----
        plan : PlanNode
            共享子树的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文。
        sid : str | None
            共享子树标识符；为 ``None`` 时直接返回 ``False``。

        返回
        ----
        bool
            与 ``compile_lazy_shared`` 相同。
        """
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
        """共享子树 lazy 编译的内部实现。

        参数
        ----
        plan : PlanNode
            共享子树的逻辑计划节点。
        ctx : ExecutionContext
            执行期上下文。
        sid : str
            共享子树标识符。

        返回
        ----
        bool
            编译并缓存 LazyFrame 成功为 ``True``；不支持、字面量节点或异常时为 ``False``。
        """
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
        """执行逻辑计划：long-table 原生 Polars 路径，必要时回退 PolarsBackend。

        参数
        ----
        plan : PlanNode
            待执行的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文，数据源需提供 ``scan_polars_long``。

        返回
        ----
        Any
            经 ``finalize_panel_result`` 对齐后的因子结果。

        异常
        ----
        PolarsLongStrictError
            严格模式下 long 路径失败且禁止回退时抛出。
        """
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

        from .polars_long_policy import assert_no_blocked_causal_plan

        assert_no_blocked_causal_plan(plan, backend="polars_long")

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
