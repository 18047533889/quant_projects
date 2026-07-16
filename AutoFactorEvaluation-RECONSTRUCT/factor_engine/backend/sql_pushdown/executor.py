# -*- coding: utf-8 -*-
"""执行编译后的因子 SQL（DuckDB / ClickHouse）。

从 ``ExecutionContext`` 提取下推上下文，编译 PlanNode 为 SQL 并在目标引擎执行，
支持单因子与批量（共享 base CTE）两种模式，以及 pandas Series 与 Polars LazyFrame 输出。
"""
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
    """确保 ``quant_projects`` 根目录在 ``sys.path`` 中以便导入 data_access。"""
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass(frozen=True)
class PushdownContext:
    """SQL 下推执行上下文：方言、数据源定位、轴列名与可选过滤条件。"""
    dialect: SqlDialect
    time_column: str
    instrument_column: str
    dataset: str | None = None
    table: str | None = None
    filt: SqlPushdownFilter | None = None
    ch_config: dict[str, Any] | None = None


def extract_pushdown_context(ctx: ExecutionContext) -> PushdownContext | None:
    """从 ExecutionContext 的数据源提取 SQL 下推上下文。

    支持 DataAccess（DuckDB 数据集）、ClickHouse 表、Composite 与 LongTable 包装；
    无法解析时返回 ``None``。
    """
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
    """将 Arrow/SQL 查询结果转为 MultiIndex (ts, inst) Series。"""
    from data_access.read.adapters import arrow_to_multiindex_series

    return arrow_to_multiindex_series(
        table,
        timestamp_column=timestamp_col,
        instrument_column=instrument_col,
        value_column="value",
        output_name="value",
    )


def _lazy_from_sql_table(table) -> Any:
    """Arrow/SQL 结果 → long-table LazyFrame（``ts, inst, _v``），无 pandas 往返。"""
    import polars as pl

    if hasattr(table, "to_pandas"):
        df = pl.from_arrow(table)
    else:
        df = pl.DataFrame(table)
    rename: dict[str, str] = {}
    if "value" in df.columns and "_v" not in df.columns:
        rename["value"] = "_v"
    if rename:
        df = df.rename(rename)
    if "_v" not in df.columns:
        raise ValueError("SQL result missing value column")
    return df.select(["ts", "inst", "_v"]).lazy()


def execute_compiled_sql(
    compiled: CompiledSql,
    ctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> pd.Series:
    """在 DuckDB 或 ClickHouse 上执行已编译 SQL，返回 MultiIndex Series。"""
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
    """批量查询结果中按列别名提取 MultiIndex Series。"""
    from data_access.read.adapters import arrow_to_multiindex_series

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
    """执行批量编译 SQL，返回 ``sid -> MultiIndex Series`` 映射。"""
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
    from data_access.read.query_budget import QueryBudget, resolve_query_budget

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
    """通过 data_access store 在 DuckDB 上执行 SQL，返回 Arrow 表。"""
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
    """在 ClickHouse 上执行 SQL，返回查询结果表。"""
    _ensure_data_access()
    from data_access.clickhouse.panel import ClickHouseConfig, execute_query

    config = ClickHouseConfig.from_env(**(ctx.ch_config or {}))
    return execute_query(config=config, sql=compiled.query)


def _execute_duckdb(
    compiled: CompiledSql,
    pctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> pd.Series:
    """DuckDB 单因子执行：SQL → Arrow 表 → MultiIndex Series。"""
    table = _execute_duckdb_table(
        compiled, pctx, data_source, query_budget=query_budget
    )
    return _series_from_sql_table(table, timestamp_col="ts", instrument_col="inst")


def _execute_clickhouse(compiled: CompiledSql, ctx: PushdownContext) -> pd.Series:
    """ClickHouse 单因子执行：SQL → 结果表 → MultiIndex Series。"""
    table = _execute_clickhouse_table(compiled, ctx)
    return _series_from_sql_table(table, timestamp_col="ts", instrument_col="inst")


def try_execute_sql_pushdown_long(
    plan: PlanNode,
    ctx: ExecutionContext,
    *,
    sid: str | None = None,
) -> Any | None:
    """尝试将 SQL 可编译子树下推为 long-table LazyFrame（``ts, inst, _v``）。

    编译或执行失败时返回 ``None``；strict 模式下失败会抛出异常。
    """
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

    try:
        if compiled.dialect == SqlDialect.CLICKHOUSE:
            table = _execute_clickhouse_table(compiled, pctx)
        else:
            table = _execute_duckdb_table(
                compiled, pctx, ctx.data_source, query_budget=getattr(ctx, "query_budget", None)
            )
        return _lazy_from_sql_table(table)
    except Exception as exc:
        from backend.sql_pushdown.strict import handle_sql_long_pushdown_failure

        try:
            handle_sql_long_pushdown_failure(ctx, sid=sid, exc=exc, phase="long_single")
        except Exception:
            raise
        return None


def try_execute_sql_pushdown_batch_long(
    plans: dict[str, PlanNode],
    ctx: ExecutionContext,
) -> dict[str, Any] | None:
    """批量 SQL 下推为 ``sid -> LazyFrame``；共享 base CTE，一次 round-trip。"""
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

    try:
        if compiled.dialect == SqlDialect.CLICKHOUSE:
            table = _execute_clickhouse_table(compiled, pctx)
        else:
            table = _execute_duckdb_table(
                compiled, pctx, ctx.data_source, query_budget=getattr(ctx, "query_budget", None)
            )
        import polars as pl

        df = pl.from_arrow(table) if hasattr(table, "to_pandas") else pl.DataFrame(table)
        out: dict[str, Any] = {}
        for sid, alias in compiled.column_aliases:
            if alias not in df.columns:
                continue
            part = df.select(["ts", "inst", pl.col(alias).alias("_v")]).lazy()
            out[sid] = part
        return out
    except Exception as exc:
        from backend.sql_pushdown.strict import handle_sql_long_pushdown_failure

        try:
            handle_sql_long_pushdown_failure(
                ctx,
                sid=",".join(sorted(plans.keys())[:3]),
                exc=exc,
                phase="long_batch",
            )
        except Exception:
            raise
        return None


def try_execute_sql_pushdown(
    plan: PlanNode,
    ctx: ExecutionContext,
) -> pd.Series | None:
    """若计划可 SQL 化则在 DuckDB/ClickHouse 内执行并返回 Series。

    上下文不可下推、编译失败或执行异常时返回 ``None``，由调用方回退 Python/Polars。
    """
    pctx = extract_pushdown_context(ctx)
    if pctx is None:
        return None

    try:
        compiled = compile_plan_to_sql(
            plan,
            dataset=pctx.dataset,
            table=pctx.table,
            time_column=pctx.time_column,
            instrument_column=pctx.instrument_column,
            filt=pctx.filt,
            dialect=pctx.dialect,
        )
    except Exception:
        return None
    if compiled is None:
        return None

    try:
        return execute_compiled_sql(
            compiled, pctx, ctx.data_source, query_budget=getattr(ctx, "query_budget", None)
        )
    except Exception:
        return None


def try_execute_sql_pushdown_batch(
    plans: dict[str, PlanNode],
    ctx: ExecutionContext,
) -> dict[str, pd.Series] | None:
    """批量 SQL 下推：多子树合并为一条 WITH 查询，返回 ``sid -> Series``。"""
    if not plans:
        return {}
    pctx = extract_pushdown_context(ctx)
    if pctx is None:
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
    except Exception:
        return None
    if compiled is None:
        return None

    try:
        return execute_batch_compiled_sql(
            compiled,
            pctx,
            ctx.data_source,
            query_budget=getattr(ctx, "query_budget", None),
        )
    except Exception:
        return None
