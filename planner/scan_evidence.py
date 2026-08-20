# -*- coding: utf-8 -*-
"""R32-P0-091: Actual backend scan evidence vs planner estimates（typed separation）。

Production acceptance 必须区分：
    - **EstimatedScanCost**: planner 估算（ScanCost / cost model）
    - **ActualScanEvidence**: backend 真实计数器（DuckDB profiling / Polars / PyArrow）
    
禁止用 planner ``source_group_count`` 冒充 ``actual_backend_scan_count``。
如果集成边界缺真实 counter，必须 typed unavailable + fail-closed 于验收。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "EstimatedScanCost",
    "ActualScanEvidence",
    "ScanEvidenceUnavailable",
    "is_actual_evidence",
]


@dataclass(frozen=True)
class EstimatedScanCost:
    """R32-P0-091: Planner 估算的扫描成本（不是真实 backend counter）。
    
    来源：
        - ScanCost.selected_bytes / projection_bytes
        - SourceScanGroup.scan_cost
        - CostModel 推导
    
    **不得**用于验收 CSE 实际效果（那需要 ActualScanEvidence）。
    """
    
    source_scope_key: str
    estimated_selected_bytes: int = 0
    estimated_projection_bytes: int = 0
    estimated_rows: int = 0
    estimated_files: int = 0
    estimated_objects: int = 0
    confidence: str = "planner_estimate"  # planner_estimate / fallback / degraded
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_scope_key": self.source_scope_key,
            "estimated_selected_bytes": self.estimated_selected_bytes,
            "estimated_projection_bytes": self.estimated_projection_bytes,
            "estimated_rows": self.estimated_rows,
            "estimated_files": self.estimated_files,
            "estimated_objects": self.estimated_objects,
            "confidence": self.confidence,
            "evidence_type": "estimated",
        }


@dataclass(frozen=True)
class ActualScanEvidence:
    """R32-P0-091: Backend 真实扫描证据（DuckDB profiling / Polars / PyArrow counters）。
    
    来源：
        - DuckDB EXPLAIN ANALYZE / profiling output
        - Polars LazyFrame.collect() metrics
        - PyArrow dataset scan statistics
        - Source block producer/consumer instrumentation
    
    **这是**验收 1000/10000 factors CSE 的唯一合法证据。
    """
    
    source_scope_key: str
    actual_scan_invocations: int = 0  # backend scan() 调用次数
    actual_object_opens: int = 0      # 实际打开的 file/object 数
    actual_bytes_read: int = 0        # backend 报告的真实字节数
    actual_rows_scanned: int = 0      # backend 扫描的行数
    actual_remote_requests: int = 0   # 远程 LIST/HEAD/GET 请求数
    source_block_producers: int = 0   # SourceBlock 生产者数（CSE 前）
    source_block_consumers: int = 0   # SourceBlock 消费者数（复用因子数）
    backend_profiler_source: str = "" # duckdb_explain_analyze / polars_metrics / ...
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_scope_key": self.source_scope_key,
            "actual_scan_invocations": self.actual_scan_invocations,
            "actual_object_opens": self.actual_object_opens,
            "actual_bytes_read": self.actual_bytes_read,
            "actual_rows_scanned": self.actual_rows_scanned,
            "actual_remote_requests": self.actual_remote_requests,
            "source_block_producers": self.source_block_producers,
            "source_block_consumers": self.source_block_consumers,
            "backend_profiler_source": self.backend_profiler_source,
            "evidence_type": "actual",
        }


@dataclass(frozen=True)
class ScanEvidenceUnavailable:
    """R32-P0-091: 真实 backend evidence 不可得（集成边界未完成 / profiler 缺失）。
    
    Production acceptance: ScanEvidenceUnavailable != PASS。
    必须显式标记 unavailable，fail-closed 于验收。
    """
    
    source_scope_key: str
    reason: str  # backend_profiler_not_implemented / integration_boundary_incomplete / ...
    fallback_to_estimate: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_scope_key": self.source_scope_key,
            "reason": self.reason,
            "fallback_to_estimate": self.fallback_to_estimate,
            "evidence_type": "unavailable",
        }


def is_actual_evidence(evidence: Any) -> bool:
    """判断是否为真实 backend evidence（而非 planner estimate）。
    
    R32-P0-091 验收必须：
        actual_backend_scan_count << factor_count
        AND evidence_type == "actual"
        AND backend_profiler_source != ""
    """
    if isinstance(evidence, ActualScanEvidence):
        return bool(evidence.backend_profiler_source)
    return False


def classify_scan_evidence(
    evidence: Any,
) -> tuple[str, dict[str, Any]]:
    """分类扫描证据类型（estimated / actual / unavailable）。
    
    Returns:
        (evidence_type, evidence_dict)
    """
    if isinstance(evidence, ActualScanEvidence):
        return ("actual", evidence.to_dict())
    if isinstance(evidence, EstimatedScanCost):
        return ("estimated", evidence.to_dict())
    if isinstance(evidence, ScanEvidenceUnavailable):
        return ("unavailable", evidence.to_dict())
    # 未知类型：fail-closed
    return ("unavailable", {
        "source_scope_key": "unknown",
        "reason": f"unknown evidence type: {type(evidence).__name__}",
        "fallback_to_estimate": False,
    })
