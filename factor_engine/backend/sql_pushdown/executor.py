# -*- coding: utf-8 -*-
"""执行编译后的因子 SQL（DuckDB / ClickHouse）。"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from backend.context import ExecutionContext
from backend.pandas_compat import pd
from planner.logical_plan import PlanNode
from workspace_paths import quant_projects_root

from .emitter import (
    BatchCompiledSql,
    CompiledSql,
    SqlDialect,
    SqlPushdownFilter,
    compile_plan_to_sql,
    compile_plans_batch_to_sql,
)
from .source_resolver import resolve_pushdown_source


def _ensure_data_access() -> None:
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass(frozen=True)
class PushdownContext:
    dialect: SqlDialect
    time_column: str
    instrument_column: str
    dataset: str | None = None
    table: str | None = None
    filt: SqlPushdownFilter | None = None
    ch_config: dict[str, Any] | None = None


def extract_pushdown_context(ctx: ExecutionContext) -> PushdownContext | None:
    """从 DataAccess / ClickHouse / Composite / LongTable 提取 SQL 下推上下文。"""
    ds = resolve_pushdown_source(ctx.data_source)
    if ds is None:
        return None

    dataset = getattr(ds, "dataset", None)
    table = getattr(ds, "table", None)

    if dataset:
        axis_fn = getattr(ds, "dataset_axis_columns", None)
        if not callable(axis_fn):
            return None
        time_col, inst_col = axis_fn()
        start = getattr(ds, "start_date", None)
        end = getattr(ds, "end_date", None)
        inst_filter = getattr(ds, "instrument_filter", None) or []
        filt = SqlPushdownFilter(
            time_column=time_col,
            start=start,
            end=end,
            instrument_column=inst_col,
            instruments=tuple(inst_filter),
        )
        return PushdownContext(
            dialect=SqlDialect.DUCKDB,
            dataset=str(dataset),
            time_column=time_col,
            instrument_column=inst_col,
            filt=filt,
        )

    time_col = getattr(ds, "timestamp_column", "trade_date")
    inst_col = getattr(ds, "instrument_column", "instrument")
    start = getattr(ds, "start_date", None)
    end = getattr(ds, "end_date", None)
    inst_filter = getattr(ds, "instrument_filter", None) or []
    ch_overrides = getattr(ds, "_ch_overrides", None)
    return PushdownContext(
        dialect=SqlDialect.CLICKHOUSE,
        table=str(table),
        time_column=str(time_col),
        instrument_column=str(inst_col),
        filt=SqlPushdownFilter(
            time_column=str(time_col),
            start=start,
            end=end,
            instrument_column=str(inst_col),
            instruments=tuple(inst_filter),
        ),
        ch_config=dict(ch_overrides) if ch_overrides else None,
    )


def _series_from_sql_table(table, *, timestamp_col: str, instrument_col: str) -> pd.Series:
    from data_access.adapters import arrow_to_multiindex_series

    return arrow_to_multiindex_series(
        table,
        timestamp_column=timestamp_col,
        instrument_column=instrument_col,
        value_column="value",
        output_name="value",
    )


def execute_compiled_sql(
    compiled: CompiledSql,
    ctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> pd.Series:
    if compiled.dialect == SqlDialect.CLICKHOUSE:
        return _execute_clickhouse(compiled, ctx)
    return _execute_duckdb(compiled, ctx, data_source, query_budget=query_budget)


def _series_from_batch_table(
    table,
    *,
    timestamp_col: str,
    instrument_col: str,
    value_column: str,
) -> pd.Series:
    from data_access.adapters import arrow_to_multiindex_series

    return arrow_to_multiindex_series(
        table,
        timestamp_column=timestamp_col,
        instrument_column=instrument_col,
        value_column=value_column,
        output_name=value_column,
    )


def execute_batch_compiled_sql(
    compiled: BatchCompiledSql,
    ctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> dict[str, pd.Series]:
    if compiled.dialect == SqlDialect.CLICKHOUSE:
        table = _execute_clickhouse_table(compiled, ctx)
    else:
        table = _execute_duckdb_table(
            compiled, ctx, data_source, query_budget=query_budget
        )

    out: dict[str, pd.Series] = {}
    col_names = list(getattr(table, "column_names", []) or getattr(table, "schema", {}).names or [])
    for sid, alias in compiled.column_aliases:
        if alias in col_names:
            out[sid] = _series_from_batch_table(
                table,
                timestamp_col="ts",
                instrument_col="inst",
                value_column=alias,
            )
    return out


def _build_duckdb_store_kwargs(
    compiled: CompiledSql | BatchCompiledSql,
    pctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> dict[str, Any]:
    """组装 store.sql() 参数：view_columns / read_params / query_budget。"""
    cols: set[str] = set(compiled.referenced_columns)
    cols.add(pctx.time_column)
    cols.add(pctx.instrument_column)
    fields = getattr(data_source, "fields", None) or {}
    physical = sorted({fields.get(c, c) for c in cols})
    view_columns = {name: physical for name in compiled.read_datasets}

    read_params: dict[str, dict[str, Any]] = {}
    ds_params = dict(getattr(data_source, "params", None) or {})
    for name in compiled.read_datasets:
        if ds_params:
            read_params[name] = ds_params

    _ensure_data_access()
    from data_access.query_budget import QueryBudget, resolve_query_budget

    explicit = query_budget
    if explicit is None:
        explicit = QueryBudget(max_rows=50_000_000)
    return {
        "read_datasets": list(compiled.read_datasets),
        "view_columns": view_columns,
        "read_params": read_params or None,
        "query_budget": resolve_query_budget(explicit),
    }


def _execute_duckdb_table(
    compiled: CompiledSql | BatchCompiledSql,
    pctx: PushdownContext | None = None,
    data_source: Any | None = None,
    query_budget: Any | None = None,
):
    _ensure_data_access()
    from data_access import get_store

    store = get_store()
    kwargs: dict[str, Any] = {}
    if pctx is not None and data_source is not None:
        kwargs = _build_duckdb_store_kwargs(
            compiled, pctx, data_source, query_budget=query_budget
        )
    return store.sql(compiled.query, **kwargs)


def _execute_clickhouse_table(compiled: CompiledSql | BatchCompiledSql, ctx: PushdownContext):
    _ensure_data_access()
    from data_access.clickhouse_panel import ClickHouseConfig, execute_query

    config = ClickHouseConfig.from_env(**(ctx.ch_config or {}))
    return execute_query(config=config, sql=compiled.query)


def _execute_duckdb(
    compiled: CompiledSql,
    pctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> pd.Series:
    table = _execute_duckdb_table(
        compiled, pctx, data_source, query_budget=query_budget
    )
    return _series_from_sql_table(table, timestamp_col="ts", instrument_col="inst")


def _execute_clickhouse(compiled: CompiledSql, ctx: PushdownContext) -> pd.Series:
    table = _execute_clickhouse_table(compiled, ctx)
    return _series_from_sql_table(table, timestamp_col="ts", instrument_col="inst")


def try_execute_sql_pushdown(
    plan: PlanNode,
    ctx: ExecutionContext,
) -> pd.Series | None:
    """若计划可 SQL 化则在 DuckDB/ClickHouse 内执行。"""
    pctx = extract_pushdown_context(ctx)
    if pctx is None:
        return None

    compiled = compile_plan_to_sql(
        plan,
        dataset=pctx.dataset,
        table=pctx.table,
        time_column=pctx.time_column,
        instrument_column=pctx.instrument_column,
        filt=pctx.filt,
        dialect=pctx.dialect,
    )
    if compiled is None:
        return None

    return execute_compiled_sql(
        compiled, pctx, ctx.data_source, query_budget=getattr(ctx, "query_budget", None)
    )


def try_execute_sql_pushdown_batch(
    plans: dict[str, PlanNode],
    ctx: ExecutionContext,
) -> dict[str, pd.Series] | None:
    """批量 SQL 下推：共享 base CTE，一次 round-trip。"""
    if not plans:
        return {}
    pctx = extract_pushdown_context(ctx)
    if pctx is None:
        return None

    compiled = compile_plans_batch_to_sql(
        plans,
        dataset=pctx.dataset,
        table=pctx.table,
        time_column=pctx.time_column,
        instrument_column=pctx.instrument_column,
        filt=pctx.filt,
        dialect=pctx.dialect,
    )
    if compiled is None:
        return None

    return execute_batch_compiled_sql(
        compiled,
        pctx,
        ctx.data_source,
        query_budget=getattr(ctx, "query_budget", None),
    )
