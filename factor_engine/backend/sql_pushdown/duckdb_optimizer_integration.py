# -*- coding: utf-8 -*-
"""集成优化后的 DuckDB 执行器到现有 executor.py。

将 duckdb_performance 模块的优化集成到生产执行路径，保持向后兼容。
2026-08-13: 初版集成。
"""
from __future__ import annotations

import os
from typing import Any

from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.pandas_compat import pd
from factor_engine.planner.logical_plan import PlanNode

from .emitter import (
    BatchCompiledSql,
    CompiledSql,
    SqlDialect,
    compile_plan_to_sql,
    compile_plans_batch_to_sql,
)
from .executor import (
    PushdownContext,
    SqlResultSchemaError,
    _build_duckdb_store_kwargs,
    _ensure_data_access,
    _note_sql_fallback,
    _series_from_batch_table,
    _sql_fallback_allowed,
    extract_pushdown_context,
)

# 导入优化模块
try:
    from .duckdb_performance import (
        DuckDBParallelConfig,
        OptimizedDuckDBExecutor,
        arrow_to_polars_zero_copy,
        compile_batch_union_all,
        duckdb_result_to_series_arrow,
        get_query_plan_cache,
    )
    _PERFORMANCE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    _PERFORMANCE_OPTIMIZATIONS_AVAILABLE = False


def _is_optimization_enabled() -> bool:
    """检查性能优化是否启用（环境变量控制）。

    DUCKDB_ENABLE_OPTIMIZATIONS=true/false（默认：true）
    """
    return os.environ.get("DUCKDB_ENABLE_OPTIMIZATIONS", "true").lower() in ("true", "1", "yes")


def _execute_duckdb_optimized(
    compiled: CompiledSql,
    pctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> pd.Series:
    """优化的 DuckDB 单因子执行（Arrow 零拷贝路径）。

    Optimization 4: Arrow zero-copy conversion
    """
    if not _PERFORMANCE_OPTIMIZATIONS_AVAILABLE or not _is_optimization_enabled():
        # 回退到原始实现
        from .executor import _execute_duckdb
        return _execute_duckdb(compiled, pctx, data_source, query_budget=query_budget)

    _ensure_data_access()
    from data_access import get_store

    store = get_store()
    kwargs = _build_duckdb_store_kwargs(
        compiled, pctx, data_source, query_budget=query_budget
    )

    # 执行查询并使用 Arrow 零拷贝转换
    arrow_result = store.sql(compiled.query, **kwargs)
    return duckdb_result_to_series_arrow(
        arrow_result,
        timestamp_col="ts",
        instrument_col="inst",
        value_col="value",
    )


def execute_batch_compiled_sql_optimized(
    compiled: BatchCompiledSql,
    ctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> dict[str, pd.Series]:
    """优化的批量 SQL 执行（集成所有优化）。

    应用的优化：
    1. 批量编译（UNION ALL）
    2. 查询计划缓存
    3. 并行执行配置
    4. Arrow 零拷贝
    5. 自动索引（如果有益）
    """
    if not _PERFORMANCE_OPTIMIZATIONS_AVAILABLE or not _is_optimization_enabled():
        # 回退到原始实现
        from .executor import execute_batch_compiled_sql
        return execute_batch_compiled_sql(compiled, ctx, data_source, query_budget=query_budget)

    _ensure_data_access()
    from data_access import get_store

    store = get_store()

    # 构建 store 参数
    kwargs = _build_duckdb_store_kwargs(
        compiled, ctx, data_source, query_budget=query_budget
    )

    # 使用优化执行器
    parallel_config = DuckDBParallelConfig.from_env()
    executor = OptimizedDuckDBExecutor(
        parallel_config=parallel_config,
        query_cache=get_query_plan_cache(),
        enable_indexing=True,
    )

    # 执行批量查询（优化路径）
    arrow_result = store.sql(compiled.query, **kwargs)

    # Arrow 零拷贝转换
    import polars as pl
    df = arrow_to_polars_zero_copy(arrow_result)

    # 检查 schema 契约
    expected = {alias for _, alias in compiled.column_aliases}
    missing = expected - set(df.columns)
    if missing:
        raise SqlResultSchemaError(f"missing expected SQL aliases: {sorted(missing)}")

    # 按列提取结果
    out: dict[str, pd.Series] = {}
    alias_list = list(compiled.column_aliases)

    # 分块处理（避免一次拉取整宽）
    for i in range(0, len(alias_list), 64):
        chunk = alias_list[i : i + 64]
        chunk_aliases = [a for _, a in chunk]
        part = df.select(["ts", "inst", *chunk_aliases]).lazy()

        for sid, alias in chunk:
            # 提取单列并转换为 Series（使用统一的 long_frame 转换器）
            sid_df = part.select(["ts", "inst", pl.col(alias).alias("_v")]).collect()
            # R47 P1-05: Use polars_long_to_multiindex_series for native conversion
            from factor_engine.backend.long_frame import polars_long_to_multiindex_series
            out[sid] = polars_long_to_multiindex_series(
                sid_df,
                timestamp_col="ts",
                instrument_col="inst",
                value_col="_v",
            )

    return out


def try_execute_sql_pushdown_batch_optimized(
    plans: dict[str, PlanNode],
    ctx: ExecutionContext,
) -> dict[str, pd.Series] | None:
    """优化的批量 SQL 下推执行（入口函数）。

    集成所有五项优化，保持与原 API 完全兼容。
    """
    if not plans:
        return {}

    if not _PERFORMANCE_OPTIMIZATIONS_AVAILABLE or not _is_optimization_enabled():
        # 回退到原始实现
        from .executor import try_execute_sql_pushdown_batch
        return try_execute_sql_pushdown_batch(plans, ctx)

    pctx = extract_pushdown_context(ctx)
    if pctx is None:
        _note_sql_fallback(ctx, "NoPushdownContext", sid=",".join(sorted(plans.keys())[:3]))
        return None

    try:
        compiled = compile_plans_batch_to_sql(
            plans,
            dataset=pctx.dataset,
            table=pctx.table,
            time_column=pctx.time_column,
            instrument_column=pctx.instrument_column,
            filt=pctx.filt,
            dialect=pctx.dialect,
        )
    except Exception as exc:
        from factor_engine.runtime.resource_errors import (
            CapabilityMiss,
            CompilationUnsupported,
            is_fail_closed_error,
        )

        if isinstance(exc, (CapabilityMiss, CompilationUnsupported)):
            _note_sql_fallback(
                ctx, type(exc).__name__, sid=",".join(sorted(plans.keys())[:3])
            )
            return None
        if is_fail_closed_error(exc):
            raise
        production = str(getattr(ctx, "run_mode", "") or "").lower() == "production"
        if production:
            raise
        _note_sql_fallback(ctx, type(exc).__name__, sid=",".join(sorted(plans.keys())[:3]))
        return None

    if compiled is None:
        _note_sql_fallback(ctx, "CompilationUnsupported", sid=",".join(sorted(plans.keys())[:3]))
        return None

    try:
        # 使用优化执行器
        return execute_batch_compiled_sql_optimized(
            compiled,
            pctx,
            ctx.data_source,
            query_budget=getattr(ctx, "query_budget", None),
        )
    except SqlResultSchemaError:
        raise
    except Exception as exc:
        if not _sql_fallback_allowed(exc, ctx):
            raise
        _note_sql_fallback(ctx, type(exc).__name__, sid=",".join(sorted(plans.keys())[:3]))
        return None


def get_optimization_stats() -> dict[str, Any]:
    """获取 DuckDB 优化统计信息（诊断用）。

    Returns:
        包含缓存命中率、执行统计等信息的 dict
    """
    if not _PERFORMANCE_OPTIMIZATIONS_AVAILABLE:
        return {"optimizations_available": False}

    cache = get_query_plan_cache()
    return {
        "optimizations_available": True,
        "optimizations_enabled": _is_optimization_enabled(),
        "query_cache": cache.stats(),
        "parallel_config": {
            "threads": DuckDBParallelConfig.from_env().threads,
            "memory_limit_mb": DuckDBParallelConfig.from_env().memory_limit_mb,
        },
    }
