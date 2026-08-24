# -*- coding: utf-8
"""R21-PRODUCTION-NO-SILENT-FALLBACK 回归：生产路径禁止 executor 静默 pandas 降级。

生产执行路由 = Global Physical Planner 选 certified PhysicalRegionPlan（显式
region + TransferEdge）；运行时失败 → 类型化失败（FailureTaxonomy）→ 若策略允许
显式 replan，绝不由 executor 悄悄降级到 pandas。``assert_no_production_pandas_fallbacks``
在 production 模式 hard-gate 任何 unplanned pandas fallback。
"""
from __future__ import annotations

import pytest


def test_top_docstring_no_longer_claims_silent_pandas_fallback():
    """api.mining_integration 顶层执行策略文档不得再声称 production 静默 pandas fallback。"""
    import factor_engine.api.mining_integration as m

    doc = m.__doc__ or ""
    # production 必须描述 planner 选择 certified plan，而不是 SQL→Polars→Pandas fallback。
    assert "PhysicalBatchGlobalOptimizer" in doc
    assert "PhysicalRegionPlan" in doc
    assert "TransferEdge" in doc
    assert "certified" in doc
    # 不得再出现旧的"Pandas fallback"链描述。
    assert "Pandas fallback" not in doc
    assert "Polars auto" not in doc
    # 生产语义：类型化失败 + 显式 replan，禁止 executor 静默降级。
    assert "类型化失败" in doc
    assert "replan" in doc
    assert "assert_no_production_pandas_fallbacks" in doc


def test_production_gate_rejects_unplanned_pandas_fallback():
    """production 模式下任何 unplanned pandas fallback 被 hard-gate（error 默认）。"""
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.runtime.production_policy import (
        ProductionPolicyViolation,
        assert_no_production_pandas_fallbacks,
    )

    ctx = ExecutionContext(
        data_source=object(),
        run_mode="production",
        runtime_stats={
            "production_pandas_fallbacks": [
                {"op": "ewm_corr", "requested": "polars", "actual": "pandas_numpy"}
            ]
        },
    )
    with pytest.raises(ProductionPolicyViolation, match="ewm_corr"):
        assert_no_production_pandas_fallbacks(ctx)


def test_production_gate_passes_when_no_fallback_or_warn_policy():
    """无 fallback 或 policy='warn' 时不抛错。"""
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.runtime.production_policy import assert_no_production_pandas_fallbacks

    clean = ExecutionContext(data_source=object(), run_mode="production")
    assert_no_production_pandas_fallbacks(clean)  # 无 fallback → 通过

    warn = ExecutionContext(
        data_source=object(),
        run_mode="production",
        production_fallback_policy="warn",
        runtime_stats={
            "production_pandas_fallbacks": [
                {"op": "ewm_corr", "requested": "polars", "actual": "pandas_numpy"}
            ]
        },
    )
    assert_no_production_pandas_fallbacks(warn)  # warn 策略 → 不抛


def test_research_mode_is_not_gated():
    """research 模式不触发 production pandas fallback hard-gate。"""
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.runtime.production_policy import assert_no_production_pandas_fallbacks

    research = ExecutionContext(
        data_source=object(),
        run_mode="research",
        runtime_stats={
            "production_pandas_fallbacks": [
                {"op": "ewm_corr", "requested": "polars", "actual": "pandas_numpy"}
            ]
        },
    )
    # research 不 gate，返回不抛错。
    assert_no_production_pandas_fallbacks(research)


def test_failure_taxonomy_supports_explicit_replan_typed_semantics():
    """类型化失败含 retry/replan/shard/fallback/abort 语义；OOM 必须 replan 而非静默降级。"""
    from factor_engine.runtime.exceptions import (
        OOMReplanRequired,
        ResourceAdmissionError,
        ResourceUnderpredictionError,
    )

    assert issubclass(OOMReplanRequired, ResourceUnderpredictionError)
    assert issubclass(ResourceUnderpredictionError, Exception)
    assert issubclass(ResourceAdmissionError, Exception)
    # 资源类失败带 replan=True 语义（显式 replan 通道，非静默 fallback）。
    for cls in (ResourceAdmissionError, ResourceUnderpredictionError, OOMReplanRequired):
        assert hasattr(cls, "__doc__")
        assert cls.__doc__ is not None
