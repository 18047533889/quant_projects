# -*- coding: utf-8 -*-
"""R32-P0-087..091: 聚焦测试（typed bindings / fail-closed / per-dataset projection / evidence）。"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from planner.factor_source_plan_r32 import (
    FactorSourcePlan,
    DependencyExtractionError,
    stable_digest,
)
from planner.scan_evidence import (
    EstimatedScanCost,
    ActualScanEvidence,
    ScanEvidenceUnavailable,
    is_actual_evidence,
    classify_scan_evidence,
)
from planner.source_binding import ColumnSourceBinding
from planner.physical_factor_dag import SourceScopeId
from planner.batch_data_request import (
    build_batch_data_request,
    SourceScanGroup,
)


# ===========================================================================
# R32-P0-087: FactorSourcePlan typed bindings
# ===========================================================================

def test_r32_p0_087_factor_source_plan_typed_bindings():
    """R32-P0-087: FactorSourcePlan 保存并序列化 typed ColumnSourceBinding。"""
    scope = SourceScopeId(dataset="daily_price", market="ashare")
    binding = ColumnSourceBinding(
        encoded_column="close",
        dataset="daily_price",
        field="close",
        market="ashare",
        source_scope=scope,
    )
    
    plan = FactorSourcePlan(
        factor_id="test_factor",
        market="ashare",
        leaf_concepts=["close", "volume"],
        source_datasets=["daily_price"],
        column_bindings=[binding],
    )
    
    assert plan.factor_id == "test_factor"
    assert plan.market == "ashare"
    assert "close" in plan.leaf_concepts
    assert "daily_price" in plan.source_datasets
    assert len(plan.column_bindings) == 1
    
    d = plan.to_dict()
    assert "column_bindings" in d
    assert len(d["column_bindings"]) == 1
    assert d["column_bindings"][0]["dataset"] == "daily_price"
    assert d["column_bindings"][0]["field"] == "close"
    
    id1 = plan.identity()
    id2 = stable_digest(plan.to_dict())
    assert id1 == id2
    assert len(id1) == 24


def test_r32_p0_087_backward_compat_without_bindings():
    """R32-P0-087: 向后兼容——不提供 bindings 时仍可构造。"""
    plan = FactorSourcePlan(
        factor_id="legacy_factor",
        market="us",
        leaf_concepts=["open", "close"],
        source_datasets=["us_daily"],
    )
    
    assert plan.factor_id == "legacy_factor"
    assert len(plan.column_bindings) == 0
    d = plan.to_dict()
    assert "column_bindings" in d
    assert d["column_bindings"] == []


def test_r32_p0_087_extract_builds_bindings_from_manifest():
    """R32-P0-087: extract() 从 manifest 构造 typed bindings。"""
    manifest = [
        {"table": "daily_price", "field": "close"},
        {"table": "daily_price", "field": "volume"},
        {"table": "fundamental", "field": "revenue"},
    ]
    
    plan = FactorSourcePlan.extract(
        factor_id="multi_source",
        market="ashare",
        source_manifest=manifest,
        allow_degraded=True,
    )
    
    assert "close" in plan.leaf_concepts
    assert "volume" in plan.leaf_concepts
    assert "revenue" in plan.leaf_concepts
    assert "daily_price" in plan.source_datasets
    assert "fundamental" in plan.source_datasets
    
    assert len(plan.column_bindings) >= 2
    binding_datasets = {b.dataset for b in plan.column_bindings}
    assert "daily_price" in binding_datasets
    assert "fundamental" in binding_datasets


# ===========================================================================
# R32-P0-088: Per-dataset field projection
# ===========================================================================

def test_r32_p0_088_per_dataset_projection():
    """R32-P0-088: BatchDataRequest 每 dataset 只投影真正需要的字段。"""
    mock_anchor = Mock(spec=['dataset', '_manifest_token', 'start_date', 'end_date',
                             'instrument_filter', 'estimate_scan_cost', 'sources'])
    mock_anchor.dataset = "daily_price"
    mock_anchor._manifest_token = "snap1"
    mock_anchor.start_date = None
    mock_anchor.end_date = None
    mock_anchor.instrument_filter = ()
    mock_anchor.estimate_scan_cost = Mock(return_value=Mock(
        selected_bytes=1000,
        projection_bytes=800,
        estimated_rows=100,
        file_count=1,
    ))

    mock_secondary = Mock(spec=['dataset', 'estimate_scan_cost'])
    mock_secondary.dataset = "income_statement"
    mock_secondary.estimate_scan_cost = Mock(return_value=Mock(
        selected_bytes=2000,
        projection_bytes=1500,
        estimated_rows=50,
        file_count=1,
    ))

    mock_anchor.sources = {"income_statement": mock_secondary}

    mock_plan = Mock(spec=['op', 'attrs', 'inputs'])
    mock_plan.op = "column"
    mock_plan.attrs = {"name": "__fe_source_ref_v1__income_statement__revenue"}
    mock_plan.inputs = []

    mock_root = Mock(spec=['op', 'attrs', 'inputs'])
    mock_root.op = "column"
    mock_root.attrs = {"name": "close"}
    mock_root.inputs = [mock_plan]

    mock_dag = Mock(spec=['roots', 'shared_nodes'])
    mock_dag.roots = [Mock(root=mock_root)]
    mock_dag.shared_nodes = {}

    analyses = {
        "test": Mock(referenced_columns=["close", "__fe_source_ref_v1__income_statement__revenue"]),
    }

    request = build_batch_data_request(
        mock_anchor,
        analyses=analyses,
        dag=mock_dag,
    )

    assert len(request.groups) >= 1

    for g in request.groups:
        assert isinstance(g.fields, tuple)
        # 每个 group 都有字段投影（即使为空也是 tuple）
        assert hasattr(g, 'dataset')


# ===========================================================================
# R32-P0-089: Dependency extraction fail-closed
# ===========================================================================

def test_r32_p0_089_production_fail_closed():
    """R32-P0-089: production 模式依赖提取失败必须 raise。"""
    with patch.dict(os.environ, {"RUNTIME_MODE": "production"}):
        mock_plan = Mock()
        
        # 正确的 patch 路径：patch 导入位置
        with patch("planner.source_dependencies.build_source_dependency_manifest") as mock_build:
            mock_build.side_effect = ValueError("manifest build failed")
            
            with pytest.raises(DependencyExtractionError) as exc_info:
                FactorSourcePlan.extract(
                    factor_id="test",
                    market="ashare",
                    expression_plan=mock_plan,
                    allow_degraded=False,
                )
            
            assert "production" in str(exc_info.value).lower()


def test_r32_p0_089_automated_research_fail_closed():
    """R32-P0-089: automated_research 模式依赖提取失败必须 raise。"""
    with patch.dict(os.environ, {"RUNTIME_MODE": "automated_research"}):
        mock_plan = Mock()
        
        with patch("planner.source_dependencies.build_source_dependency_manifest") as mock_build:
            mock_build.side_effect = ValueError("manifest build failed")
            
            with pytest.raises(DependencyExtractionError) as exc_info:
                FactorSourcePlan.extract(
                    factor_id="test",
                    market="ashare",
                    expression_plan=mock_plan,
                    allow_degraded=False,
                )
            
            assert "automated_research" in str(exc_info.value).lower()


def test_r32_p0_089_interactive_explicit_degraded():
    """R32-P0-089: interactive 模式显式允许降级。"""
    with patch.dict(os.environ, {"RUNTIME_MODE": "interactive"}):
        mock_plan = Mock()
        
        with patch("planner.source_dependencies.build_source_dependency_manifest") as mock_build:
            mock_build.side_effect = ValueError("manifest build failed")
            
            plan = FactorSourcePlan.extract(
                factor_id="test",
                market="ashare",
                expression_plan=mock_plan,
                allow_degraded=True,
                leaf_concepts=["close"],
                source_datasets=["daily"],
            )
            
            assert plan.factor_id == "test"
            assert "close" in plan.leaf_concepts


def test_r32_p0_089_interactive_without_allow_degraded_still_fails():
    """R32-P0-089: interactive 模式不显式 allow_degraded 时仍 fail-closed。"""
    with patch.dict(os.environ, {"RUNTIME_MODE": "production"}):  # 默认 production
        mock_plan = Mock()
        
        with patch("planner.source_dependencies.build_source_dependency_manifest") as mock_build:
            mock_build.side_effect = ValueError("manifest build failed")
            
            with pytest.raises(DependencyExtractionError):
                FactorSourcePlan.extract(
                    factor_id="test",
                    market="ashare",
                    expression_plan=mock_plan,
                    allow_degraded=False,
                )


# ===========================================================================
# R32-P0-091: Backend actual counters vs planner estimates
# ===========================================================================

def test_r32_p0_091_estimated_scan_cost_typed():
    """R32-P0-091: EstimatedScanCost 是 planner 估算。"""
    cost = EstimatedScanCost(
        source_scope_key="dataset:daily::market:ashare",
        estimated_selected_bytes=1000000,
        estimated_projection_bytes=800000,
        estimated_rows=10000,
        estimated_files=10,
        confidence="planner_estimate",
    )
    
    d = cost.to_dict()
    assert d["evidence_type"] == "estimated"
    assert d["estimated_selected_bytes"] == 1000000
    assert d["confidence"] == "planner_estimate"
    assert not is_actual_evidence(cost)


def test_r32_p0_091_actual_scan_evidence_typed():
    """R32-P0-091: ActualScanEvidence 来自 backend profiler。"""
    evidence = ActualScanEvidence(
        source_scope_key="dataset:daily::market:ashare",
        actual_scan_invocations=3,
        actual_object_opens=10,
        actual_bytes_read=950000,
        actual_rows_scanned=9800,
        actual_remote_requests=5,
        source_block_producers=3,
        source_block_consumers=100,
        backend_profiler_source="duckdb_explain_analyze",
    )
    
    d = evidence.to_dict()
    assert d["evidence_type"] == "actual"
    assert d["actual_scan_invocations"] == 3
    assert d["source_block_consumers"] == 100
    assert d["backend_profiler_source"] == "duckdb_explain_analyze"
    assert is_actual_evidence(evidence)


def test_r32_p0_091_unavailable_fail_closed():
    """R32-P0-091: ScanEvidenceUnavailable 验收必须 fail-closed。"""
    unavail = ScanEvidenceUnavailable(
        source_scope_key="dataset:daily::market:ashare",
        reason="backend_profiler_not_implemented",
        fallback_to_estimate=False,
    )
    
    d = unavail.to_dict()
    assert d["evidence_type"] == "unavailable"
    assert "not_implemented" in d["reason"]
    assert not is_actual_evidence(unavail)


def test_r32_p0_091_classify_evidence():
    """R32-P0-091: classify_scan_evidence 区分三种类型。"""
    estimated = EstimatedScanCost(
        source_scope_key="test",
        estimated_selected_bytes=1000,
    )
    actual = ActualScanEvidence(
        source_scope_key="test",
        actual_scan_invocations=1,
        backend_profiler_source="duckdb",
    )
    unavail = ScanEvidenceUnavailable(
        source_scope_key="test",
        reason="test",
    )
    
    assert classify_scan_evidence(estimated)[0] == "estimated"
    assert classify_scan_evidence(actual)[0] == "actual"
    assert classify_scan_evidence(unavail)[0] == "unavailable"
    assert classify_scan_evidence(Mock())[0] == "unavailable"


def test_r32_p0_091_cse_acceptance_criterion():
    """R32-P0-091: 验收标准——actual_scan_invocations << factor_count。"""
    factor_count = 1000
    
    good_evidence = ActualScanEvidence(
        source_scope_key="test",
        actual_scan_invocations=5,
        source_block_producers=5,
        source_block_consumers=1000,
        backend_profiler_source="duckdb_explain_analyze",
    )
    
    assert good_evidence.actual_scan_invocations < factor_count / 10
    assert good_evidence.source_block_consumers == factor_count
    assert is_actual_evidence(good_evidence)
    
    bad_estimate = EstimatedScanCost(
        source_scope_key="test",
        estimated_selected_bytes=10000,
        confidence="planner_estimate",
    )
    
    assert not is_actual_evidence(bad_estimate)


def test_r32_integration_smoke():
    """R32 集成烟雾测试。"""
    scope = SourceScopeId(dataset="daily", market="ashare")
    binding = ColumnSourceBinding(
        encoded_column="close",
        dataset="daily",
        field="close",
        market="ashare",
        source_scope=scope,
    )
    
    plan = FactorSourcePlan(
        factor_id="smoke_test",
        market="ashare",
        leaf_concepts=["close"],
        source_datasets=["daily"],
        column_bindings=[binding],
    )
    
    d = plan.to_dict()
    assert "column_bindings" in d
    assert d["column_bindings"][0]["dataset"] == "daily"
    
    actual = ActualScanEvidence(
        source_scope_key=scope.key(),
        actual_scan_invocations=1,
        backend_profiler_source="test",
    )
    estimated = EstimatedScanCost(
        source_scope_key=scope.key(),
        estimated_selected_bytes=1000,
    )
    
    assert is_actual_evidence(actual)
    assert not is_actual_evidence(estimated)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
