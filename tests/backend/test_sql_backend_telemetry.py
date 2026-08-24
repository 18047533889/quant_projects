# -*- coding: utf-8
"""SqlBackend telemetry：fallback 与 used_sql_pushdown 分离。"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.sql_backend import SqlBackend
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.physical_plan import PhysicalPlan


def _physical(*, fully_sql: bool = False, subtrees: dict | None = None) -> PhysicalPlan:
    root = PlanNode(op="column", attrs={"name": "close"})
    return PhysicalPlan(
        root=root,
        fully_sql=fully_sql,
        sql_subtrees=subtrees or {},
    )


def test_all_sql_fallback_does_not_mark_used_sql_pushdown():
    sub = PlanNode(op="ts_mean", inputs=[PlanNode(op="column", attrs={"name": "close"})], attrs={"d": 3})
    backend = SqlBackend(operator_backend="pandas_numpy")
    ctx = ExecutionContext(data_source=object())

    with patch("factor_engine.backend.sql_backend.lower_to_physical_plan", return_value=_physical(subtrees={"s1": sub})):
        with patch("factor_engine.backend.sql_backend.try_execute_sql_pushdown", return_value=None):
            with patch("factor_engine.backend.sql_backend.try_execute_sql_pushdown_batch", return_value=None):
                with patch.object(backend, "_eval_python", return_value="py_result"):
                    with patch.object(backend, "_eval_hybrid", return_value="final"):
                        backend.execute(PlanNode(op="column", attrs={"name": "close"}), ctx)

    runtime = dict(getattr(ctx, "runtime_stats", {}) or {})
    assert runtime.get("sql_query_count", 0) == 0
    assert runtime.get("used_sql_pushdown") is False
    assert int(runtime.get("sql_fallback_subtree_count") or 0) == 1
    assert runtime.get("python_fallback_subtree_sids") == ["s1"]


def test_fully_sql_failure_records_execution_failed():
    backend = SqlBackend(operator_backend="pandas_numpy")
    ctx = ExecutionContext(data_source=object())
    phys = _physical(fully_sql=True)

    with patch("factor_engine.backend.sql_backend.lower_to_physical_plan", return_value=phys):
        with patch("factor_engine.backend.sql_backend.try_execute_sql_pushdown", return_value=None):
            with patch.object(backend, "_eval_hybrid", return_value="final"):
                backend.execute(phys.root, ctx)

    runtime = dict(getattr(ctx, "runtime_stats", {}) or {})
    assert runtime.get("sql_full_execution_failed") is True
    assert runtime.get("used_sql_pushdown") is False
