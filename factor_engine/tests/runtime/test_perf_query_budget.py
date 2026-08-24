"""PerfConfig query budget 与 ExecutionContext 传递。"""

from __future__ import annotations

from factor_engine.backend.context import ExecutionContext
from factor_engine.runtime.perf_config import PerfConfig


def test_perf_config_build_query_budget(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_QUERY_MAX_ROWS", "1000000")
    monkeypatch.setenv("FACTOR_ENGINE_QUERY_MAX_RESULT_BYTES", "500000000")
    perf = PerfConfig.from_env()
    budget = perf.build_query_budget()
    assert budget is not None
    assert budget.max_rows == 1_000_000
    assert budget.max_result_bytes == 500_000_000


def test_execution_context_accepts_query_budget():
    from data_access.read.query_budget import QueryBudget

    budget = QueryBudget(max_rows=100)
    ctx = ExecutionContext(data_source=object(), query_budget=budget)
    assert ctx.query_budget.max_rows == 100
