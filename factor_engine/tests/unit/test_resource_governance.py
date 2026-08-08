# -*- coding: utf-8 -*-
"""Phase 5 资源治理单测：资源发现 / ExecutionResourcePlan / MemoryGovernor /
异常分级 / 结果预算 / CSE 引用计数 / one-to-many 降维守卫。

运行：``python3 -m pytest tests/unit/test_resource_governance.py -q``
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, "/home/shw/quant_projects/factor_engine")


# ---------------------------------------------------------------------------
# 资源发现
# ---------------------------------------------------------------------------


def test_effective_cpu_slots_env_override(monkeypatch):
    from runtime.resource_governor import effective_cpu_slots

    monkeypatch.setenv("FACTOR_ENGINE_CPU_BUDGET", "3")
    assert effective_cpu_slots() == 3
    assert effective_cpu_slots(explicit=7) == 7


def test_effective_memory_limit_env_override(monkeypatch):
    from runtime.resource_governor import effective_memory_limit_bytes

    monkeypatch.setenv("FACTOR_ENGINE_MAX_MEMORY_BYTES", str(1024**3))
    assert effective_memory_limit_bytes() == 1024**3
    assert effective_memory_limit_bytes(explicit_bytes=2 * 1024**3) == 2 * 1024**3


def test_execution_resource_plan_budgets_positive(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_MAX_MEMORY_BYTES", str(8 * 1024**3))
    monkeypatch.setenv("FACTOR_ENGINE_CPU_BUDGET", "4")
    from runtime.resource_governor import ExecutionResourcePlan

    plan = ExecutionResourcePlan.auto()
    assert plan.process_budget_bytes > 0
    assert plan.duckdb_budget_bytes > 0
    assert plan.result_budget_bytes > 0
    assert plan.max_workers >= 1
    # n_jobs × duckdb_threads <= cpu
    assert plan.max_workers * plan.duckdb_threads <= 4 or plan.max_workers == 1
    # process budget 不超有效内存
    assert plan.process_budget_bytes <= 8 * 1024**3


def test_resource_plan_from_dict_invalid_fraction_raises(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_MAX_MEMORY_BYTES", str(8 * 1024**3))
    from runtime.resource_governor import ExecutionResourcePlan

    with pytest.raises(ValueError):
        ExecutionResourcePlan.from_dict({"memory": {"process_fraction": 1.5}})


# ---------------------------------------------------------------------------
# MemoryGovernor
# ---------------------------------------------------------------------------


def test_memory_governor_reserve_evict_release(monkeypatch):
    from runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=10_000, duckdb_budget_bytes=2_000)
    gov.register_layer("test", lambda target: 5_000)  # evict hook 释放 5000
    assert gov.reserve("test", 6_000) is True
    # 12000 > 10000 → 触发 evict（5000）→ 7000，仍可容纳本次 6000
    assert gov.reserve("test", 6_000) is True
    assert gov.evictions, "超预算应触发 evict"
    gov.release("test", 6_000)
    assert gov.total_usage <= 10_000


def test_memory_governor_reject_when_evict_insufficient(monkeypatch):
    from runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=1_000, duckdb_budget_bytes=200)
    gov.register_layer("test", lambda target: 0)  # evict 不释放任何东西
    assert gov.reserve("big", 999) is True
    assert gov.reserve("huge", 500) is False  # 单对象放不下 → 拒绝缓存（宁可重算）


# ---------------------------------------------------------------------------
# 异常分级：只有 capability 类允许 fallback
# ---------------------------------------------------------------------------


def test_exception_classification():
    from runtime.resource_errors import (
        CapabilityMiss,
        CompilationUnsupported,
        ResourceBudgetExceeded,
        classify_exception,
        is_fail_closed_error,
        may_fallback,
    )

    assert may_fallback(CapabilityMiss("no"))
    assert may_fallback(CompilationUnsupported("no"))
    assert not may_fallback(ResourceBudgetExceeded("oom"))
    assert is_fail_closed_error(ResourceBudgetExceeded("oom"))
    assert not is_fail_closed_error(CapabilityMiss("no"))
    # 未知异常 → 默认 fail-closed（不静默 fallback）
    may, closed = classify_exception(ValueError("?"))

    assert not may and closed


def test_dataaccess_errors_fail_closed(monkeypatch):
    from runtime.resource_errors import is_data_access_governance_failure

    try:
        import data_access.core.exceptions as da

        assert is_data_access_governance_failure(da.ValidationError("x"))
        assert is_data_access_governance_failure(da.DeadlineExceeded("x"))
        assert not is_data_access_governance_failure(ValueError("x"))
    except ImportError:
        pytest.skip("dataaccess 未安装")


# ---------------------------------------------------------------------------
# 结果字节预算
# ---------------------------------------------------------------------------


def test_result_budget_no_explicit_passes():
    from runtime.result_budget import enforce_result_budget

    import pandas as pd

    series = pd.Series([1.0, 2.0])
    assert enforce_result_budget(series, None) is True  # 无显式预算 → 放行


def test_result_budget_production_raises(monkeypatch):
    from runtime.result_budget import enforce_result_budget
    from runtime.resource_errors import ResourceBudgetExceeded

    import pandas as pd

    series = pd.Series([1.0, 2.0])

    class _Perf:
        result_budget_bytes = 8  # 1 byte per value → 超限

    with pytest.raises(ResourceBudgetExceeded):
        enforce_result_budget(series, _Perf(), run_mode="production")


def test_result_budget_research_warn_only(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_RESULT_BUDGET_WARN_ONLY", "1")
    from runtime.result_budget import enforce_result_budget

    import pandas as pd

    class _Perf:
        result_budget_bytes = 8

    assert enforce_result_budget(pd.Series([1.0, 2.0]), _Perf(), run_mode="research") is True
    monkeypatch.delenv("FACTOR_ENGINE_RESULT_BUDGET_WARN_ONLY")


# ---------------------------------------------------------------------------
# CSE 引用计数
# ---------------------------------------------------------------------------


def _plan_node(op, attrs=None, inputs=()):
    from planner.logical_plan import PlanNode

    return PlanNode(op=op, attrs=dict(attrs or {}), inputs=list(inputs), node_id=0)


def test_cse_consumer_counts_and_release():
    from planner.cse import collect_consumed_sids, cse_consumer_counts

    ref = lambda sid: _plan_node("plan_ref", {"sid": sid})
    root_a = _plan_node("add", inputs=[ref("s1"), ref("s2")])
    root_b = _plan_node("add", inputs=[ref("s1"), _plan_node("col", {"name": "x"})])
    counts = cse_consumer_counts([root_a, root_b])
    assert counts == {"s1": 2, "s2": 1}
    assert sorted(collect_consumed_sids(root_a)) == ["s1", "s2"]


def test_release_consumed_sids_evicts_zero():
    from runtime.batch_service import _release_consumed_sids
    from types import SimpleNamespace

    store = {"s1": 1, "s2": 2}
    ctx = SimpleNamespace(
        shared_result_cache=store,
        _cse_refcounts={"s1": 1, "s2": 1},
        expression_cache=None,
    )
    _release_consumed_sids(ctx, _plan_node("plan_ref", {"sid": "s1"}))
    assert "s1" not in store  # 引用计数归零 → 立即释放
    assert "s2" in store


# ---------------------------------------------------------------------------
# one-to-many 降维守卫
# ---------------------------------------------------------------------------


def test_one_to_many_guard_requires_reduction(monkeypatch):
    from storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource
    from storage.sources.logical_tables import LogicalTableContract
    from api.source_ref import SourceRefSpec

    contract = LogicalTableContract("ashare_stock_topten_shareholder", "relation_pit", cardinality="one_to_many")
    spec = SourceRefSpec(table="StockTopTenShareholder", field="ShareRatio", params={})
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    from runtime.resource_errors import SemanticContractError

    try:
        with pytest.raises(SemanticContractError):
            LQTPLogicalDataSource._assert_relation_scalar("StockTopTenShareholder", spec, contract, "ShareRatio")
    finally:
        monkeypatch.delenv("QUANT_PRODUCTION_MODE", None)
    # 显式 rank → 通过
    spec2 = SourceRefSpec(table="StockTopTenShareholder", field="ShareRatio", params={"rank": 1})
    LQTPLogicalDataSource._assert_relation_scalar("StockTopTenShareholder", spec2, contract, "ShareRatio")
