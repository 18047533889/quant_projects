# -*- coding: utf-8 -*-
"""通过 data_access.store.sql 执行编译后的因子 SQL。"""
from __future__ import annotations

import sys
from typing import Any

from backend.context import ExecutionContext
from backend.pandas_compat import pd
from planner.logical_plan import PlanNode
from workspace_paths import quant_projects_root

from .emitter import compile_plan_to_sql


def _ensure_data_access() -> None:
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _series_from_sql_table(table, *, timestamp_col: str, instrument_col: str) -> pd.Series:
    from data_access.adapters import arrow_to_multiindex_series

    return arrow_to_multiindex_series(
        table,
        timestamp_column=timestamp_col,
        instrument_column=instrument_col,
        value_column="value",
        output_name="value",
    )


def try_execute_sql_pushdown(
    plan: PlanNode,
    ctx: ExecutionContext,
) -> pd.Series | None:
    """若计划可 SQL 化且数据源为 data_access，则在 DuckDB 内执行并返回 Series。"""
    ds = ctx.data_source
    dataset = getattr(ds, "dataset", None)
    if not dataset:
        return None

    axis_fn = getattr(ds, "dataset_axis_columns", None)
    if not callable(axis_fn):
        return None
    time_col, inst_col = axis_fn()

    compiled = compile_plan_to_sql(
        plan,
        dataset=dataset,
        time_column=time_col,
        instrument_column=inst_col,
    )
    if compiled is None:
        return None

    _ensure_data_access()
    from data_access import get_store

    store = get_store()
    table = store.sql(
        compiled.query,
        read_datasets=list(compiled.read_datasets),
    )
    return _series_from_sql_table(
        table,
        timestamp_col="ts",
        instrument_col="inst",
    )
