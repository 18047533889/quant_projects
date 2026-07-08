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


def execute_compiled_sql(compiled: CompiledSql, ctx: PushdownContext) -> pd.Series:
    if compiled.dialect == SqlDialect.CLICKHOUSE:
        return _execute_clickhouse(compiled, ctx)
    return _execute_duckdb(compiled)


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
) -> dict[str, pd.Series]:
    if compiled.dialect == SqlDialect.CLICKHOUSE:
        table = _execute_clickhouse_table(compiled, ctx)
    else:
        table = _execute_duckdb_table(compiled)

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


def _execute_duckdb_table(compiled: CompiledSql | BatchCompiledSql):
    _ensure_data_access()
    from data_access import get_store

    store = get_store()
    return store.sql(
        compiled.query,
        read_datasets=list(compiled.read_datasets),
    )


def _execute_clickhouse_table(compiled: CompiledSql | BatchCompiledSql, ctx: PushdownContext):
    _ensure_data_access()
    from data_access.clickhouse_panel import ClickHouseConfig, execute_query

    config = ClickHouseConfig.from_env(**(ctx.ch_config or {}))
    return execute_query(config=config, sql=compiled.query)


def _execute_duckdb(compiled: CompiledSql) -> pd.Series:
    table = _execute_duckdb_table(compiled)
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

    return execute_compiled_sql(compiled, pctx)


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

    return execute_batch_compiled_sql(compiled, pctx)
