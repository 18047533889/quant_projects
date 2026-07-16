# -*- coding: utf-8 -*-
"""SqlBackend：整树 SQL + 部分子树 SQL 预计算 + Python/Polars fallback。

本后端将逻辑计划物理化为 SQL 可下推子树与 Python 剩余节点，
通过 DuckDB / ClickHouse 执行 SQL 部分，将物化结果注入
``ExecutionContext.materialized_series`` 或 ``materialized_long_lazy``，
再由 Polars/Pandas 算子层完成根节点求值。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from planner.logical_plan import PlanNode
from planner.sql_lowerer import lower_to_physical_plan

from .base import Backend
from .context import ExecutionContext
from .panel_native import finalize_panel_result
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend
from .runtime_events import append_runtime_event, merge_runtime
from .runtime_labels import resolve_runtime_backend_label
from .sql_pushdown.executor import (
    extract_pushdown_context,
    try_execute_sql_pushdown,
    try_execute_sql_pushdown_batch,
    try_execute_sql_pushdown_batch_long,
    try_execute_sql_pushdown_long,
)


def _merge_runtime(ctx: ExecutionContext, **fields: Any) -> None:
    """将运行时统计字段合并写入 ``ctx.runtime_stats``。"""
    merge_runtime(ctx, **fields)


class SqlBackend(Backend):
    """混合 SQL 执行后端（DuckDB / ClickHouse + pandas/polars）。"""

    runtime_backend_label = "duckdb_sql"

    def __init__(self, *, operator_backend: str = "auto") -> None:
        self._operator_backend = operator_backend
        self._python = PandasBackend()
        self._polars = PolarsBackend()

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        from .polars_long_policy import assert_no_blocked_causal_plan

        assert_no_blocked_causal_plan(plan, backend="duckdb_sql")
        physical = lower_to_physical_plan(plan)
        pctx = extract_pushdown_context(ctx)
        dialect = pctx.dialect.value if pctx is not None else None

        mat_cache: dict[str, Any] = dict(getattr(ctx, "materialized_series", None) or {})
        mat_lazy: dict[str, Any] = dict(getattr(ctx, "materialized_long_lazy", None) or {})
        python_fallback_cache: dict[str, Any] = {}
        sql_series_sids: set[str] = set()
        sql_long_sids: set[str] = set()
        prefer_long = bool(getattr(ctx, "materialize_sql_as_long_lazy", False))
        query_count = 0
        sql_fallback_count = 0

        _merge_runtime(
            ctx,
            backend=resolve_runtime_backend_label(self),
            sql_plan_fully_compilable=bool(physical.fully_sql),
            fully_sql=bool(physical.fully_sql),
            sql_subtrees=sorted(physical.sql_subtrees.keys()),
            sql_subtree_count=len(physical.sql_subtrees),
            sql_dialect=dialect,
            sql_fully_pushed=False,
            sql_partial_pushed=False,
            sql_full_execution_failed=False,
            used_sql_pushdown=False,
            sql_query_count=0,
            sql_fallback_subtree_count=0,
            python_fallback_subtree_sids=[],
        )

        if physical.fully_sql:
            pushed = try_execute_sql_pushdown(physical.root, ctx)
            if pushed is not None:
                query_count += 1
                _merge_runtime(
                    ctx,
                    used_sql_pushdown=True,
                    sql_fully_pushed=True,
                    sql_query_count=query_count,
                    sql_long_lazy_subtrees=sorted(mat_lazy.keys()),
                    sql_series_subtrees=sorted(mat_cache.keys()),
                )
                append_runtime_event(
                    ctx,
                    "sql_fully_pushed",
                    backend=resolve_runtime_backend_label(self),
                    dialect=dialect,
                    query_count=query_count,
                )
                return finalize_panel_result(pushed, ctx)
            _merge_runtime(ctx, sql_full_execution_failed=True)

        pending = {
            sid: sub
            for sid, sub in physical.sql_subtrees.items()
            if sid not in mat_cache and sid not in mat_lazy
        }
        if len(pending) > 1:
            if prefer_long:
                batch_long = try_execute_sql_pushdown_batch_long(pending, ctx)
                if batch_long:
                    query_count += 1
                    for sid in batch_long:
                        mat_lazy[sid] = batch_long[sid]
                        sql_long_sids.add(str(sid))
                    pending = {sid: sub for sid, sub in pending.items() if sid not in mat_lazy}
            if pending:
                batch = try_execute_sql_pushdown_batch(pending, ctx)
                if batch:
                    query_count += 1
                    for sid in batch:
                        mat_cache[sid] = batch[sid]
                        sql_series_sids.add(str(sid))
                    pending = {sid: sub for sid, sub in pending.items() if sid not in mat_cache}

        for sid, sub in pending.items():
            if sid in mat_cache or sid in mat_lazy:
                continue
            if prefer_long:
                pushed_long = try_execute_sql_pushdown_long(sub, ctx, sid=sid)
                if pushed_long is not None:
                    mat_lazy[sid] = pushed_long
                    sql_long_sids.add(str(sid))
                    query_count += 1
                    continue
            pushed = try_execute_sql_pushdown(sub, ctx)
            if pushed is not None:
                mat_cache[sid] = pushed
                sql_series_sids.add(str(sid))
                query_count += 1
            else:
                python_fallback_cache[str(sid)] = self._eval_python(sub, ctx)
                mat_cache[sid] = python_fallback_cache[str(sid)]
                sql_fallback_count += 1

        used_sql = query_count > 0
        partial = used_sql and not physical.fully_sql
        if used_sql or python_fallback_cache or mat_lazy or mat_cache:
            _merge_runtime(
                ctx,
                used_sql_pushdown=used_sql,
                sql_partial_pushed=partial and used_sql,
                sql_long_lazy_subtrees=sorted(sql_long_sids),
                sql_series_subtrees=sorted(sql_series_sids),
                python_fallback_subtree_sids=sorted(python_fallback_cache.keys()),
                sql_fallback_subtree_count=sql_fallback_count,
                sql_query_count=query_count,
            )
            if used_sql:
                append_runtime_event(
                    ctx,
                    "sql_partial_pushed",
                    backend=resolve_runtime_backend_label(self),
                    dialect=dialect,
                    long_lazy_count=len(sql_long_sids),
                    series_count=len(sql_series_sids),
                    fallback_count=sql_fallback_count,
                    query_count=query_count,
                )
            if python_fallback_cache:
                append_runtime_event(
                    ctx,
                    "sql_python_fallback",
                    backend=resolve_runtime_backend_label(self),
                    dialect=dialect,
                    fallback_sids=sorted(python_fallback_cache.keys()),
                    fallback_count=sql_fallback_count,
                    query_count=query_count,
                )

        ctx = replace(ctx, materialized_series=mat_cache, materialized_long_lazy=mat_lazy)
        return self._eval_hybrid(physical.root, ctx)

    def _eval_hybrid(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        if self._operator_backend == "pandas_numpy":
            return self._python.execute(plan, ctx)
        return self._polars.execute(plan, ctx)

    def _eval_python(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        backend = self._python if self._operator_backend == "pandas_numpy" else self._polars
        if hasattr(backend, "_eval"):
            return backend._eval(plan, ctx)
        return backend.execute(plan, ctx)
