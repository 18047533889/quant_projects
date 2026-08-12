# -*- coding: utf-8 -*-
"""Region 全链路 Telemetry：Region/transfer/sort/spill/source scan 监控 (MB-P2-009)。

记录每次执行的实际指标，用于：
1. 成本模型校准 (actual vs predicted)
2. 性能回归检测
3. 生产监控与调优
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceScanTelemetry:
    """Source scan 执行遥测 (P2-009)。"""

    scan_id: str
    dataset: str
    columns: tuple[str, ...]
    date_range: tuple[str, str] | None
    universe_size: int
    remote: bool
    storage_kind: str  # parquet / cos / local
    # 预测 vs 实际
    predicted_bytes: int
    actual_bytes: int
    predicted_ms: float
    actual_ms: float
    # 执行细节
    row_groups_scanned: int
    row_groups_pruned: int
    projection_pushdown: bool
    predicate_pushdown: bool
    # 性能指标
    throughput_mbps: float
    scan_started_at: float
    scan_finished_at: float


@dataclass
class RegionExecutionTelemetry:
    """单个 Region 执行遥测 (P2-009)。"""

    region_id: str
    backend: str
    representation: str
    execution_axis: str
    node_count: int
    operator_list: tuple[str, ...]
    # 预测 vs 实际
    predicted_compute_ms: float
    actual_compute_ms: float
    predicted_memory_mb: float
    actual_peak_memory_mb: float
    predicted_rows: int
    actual_rows: int
    # 执行细节
    planned_backend: str
    fallback_occurred: bool
    fallback_reason: str
    native_execution: bool
    delegate_count: int
    # 时间戳
    region_started_at: float
    region_finished_at: float
    # 线程/资源
    parallelism: int
    cpu_tokens_used: int
    ram_tokens_used: int


@dataclass
class TransferEdgeTelemetry:
    """Region 间 TransferEdge 执行遥测 (P2-009)。"""

    edge_id: str
    producer_region: str
    consumer_region: str
    source_representation: str
    target_representation: str
    # 预测 vs 实际
    predicted_bytes: int
    actual_bytes: int
    predicted_ms: float
    actual_ms: float
    # 转换细节
    requires_sort: bool
    actual_sort_ms: float
    requires_repartition: bool
    actual_repartition_ms: float
    requires_reshape: bool
    actual_reshape_ms: float
    # 数据质量
    row_count: int
    column_count: int
    schema_version: str
    # 时间戳
    transfer_started_at: float
    transfer_finished_at: float


@dataclass
class SpillEventTelemetry:
    """Spill 事件遥测 (P2-009)。"""

    spill_id: str
    node_id: str
    region_id: str
    reason: str  # memory_pressure / policy / explicit
    format: str  # arrow / parquet
    # 数据规模
    memory_bytes: int
    spill_bytes: int
    compression_ratio: float
    # 性能
    spill_ms: float
    reload_ms: float
    spill_started_at: float
    spill_finished_at: float
    reload_started_at: float | None
    reload_finished_at: float | None
    # 存储
    spill_path: str
    cleanup_policy: str


@dataclass
class SortEventTelemetry:
    """Sort 操作遥测 (P2-009)。"""

    sort_id: str
    region_id: str
    sort_keys: tuple[str, ...]
    row_count: int
    predicted_ms: float
    actual_ms: float
    algorithm: str  # quicksort / timsort / external_merge
    in_memory: bool
    spill_occurred: bool
    sort_started_at: float
    sort_finished_at: float


@dataclass
class BatchExecutionTelemetry:
    """整个 batch 执行的聚合遥测 (P2-009)。"""

    batch_id: str
    plan_hash: str
    total_regions: int
    total_transfers: int
    total_sorts: int
    total_spills: int
    total_source_scans: int
    # 全局预测 vs 实际
    predicted_ttdc_ms: float
    actual_ttdc_ms: float
    predicted_peak_memory_mb: float
    actual_peak_memory_mb: float
    # Backend 分布
    backend_distribution: dict[str, int]
    backend_switches: int
    fallback_count: int
    # 明细事件
    regions: tuple[RegionExecutionTelemetry, ...]
    transfers: tuple[TransferEdgeTelemetry, ...]
    sorts: tuple[SortEventTelemetry, ...]
    spills: tuple[SpillEventTelemetry, ...]
    source_scans: tuple[SourceScanTelemetry, ...]
    # 资源与时间
    batch_started_at: float
    batch_finished_at: float
    within_memory_budget: bool
    memory_budget_mb: float | None
    # 元数据
    execution_mode: str  # production / research
    hardware_fingerprint: dict[str, str]


def _compute_accuracy(predicted: float, actual: float) -> float:
    """计算预测准确率：1 - |predicted - actual| / actual。"""
    if actual == 0:
        return 1.0 if predicted == 0 else 0.0
    return max(0.0, 1.0 - abs(predicted - actual) / actual)


def summarize_telemetry(telemetry: BatchExecutionTelemetry) -> dict[str, Any]:
    """汇总 telemetry 为统计摘要 (P2-009)。

    Returns:
        包含以下维度的字典：
        - 全局准确率：TTDC / memory prediction accuracy
        - 每 Region 准确率分布
        - Backend 性能对比
        - Transfer overhead 占比
        - Sort/spill 频率与成本
    """
    ttdc_accuracy = _compute_accuracy(telemetry.predicted_ttdc_ms, telemetry.actual_ttdc_ms)
    memory_accuracy = _compute_accuracy(
        telemetry.predicted_peak_memory_mb, telemetry.actual_peak_memory_mb
    )

    # Region 级准确率
    region_compute_accuracies = [
        _compute_accuracy(r.predicted_compute_ms, r.actual_compute_ms)
        for r in telemetry.regions
        if r.actual_compute_ms > 0
    ]
    region_memory_accuracies = [
        _compute_accuracy(r.predicted_memory_mb, r.actual_peak_memory_mb)
        for r in telemetry.regions
        if r.actual_peak_memory_mb > 0
    ]

    # Transfer 准确率
    transfer_accuracies = [
        _compute_accuracy(t.predicted_ms, t.actual_ms)
        for t in telemetry.transfers
        if t.actual_ms > 0
    ]

    # Backend 性能汇总
    backend_times: dict[str, list[float]] = {}
    for region in telemetry.regions:
        backend_times.setdefault(region.backend, []).append(region.actual_compute_ms)

    backend_avg_ms = {
        backend: sum(times) / len(times) if times else 0.0
        for backend, times in backend_times.items()
    }

    # Transfer overhead
    total_compute_ms = sum(r.actual_compute_ms for r in telemetry.regions)
    total_transfer_ms = sum(t.actual_ms for t in telemetry.transfers)
    total_sort_ms = sum(s.actual_ms for s in telemetry.sorts)
    total_spill_ms = sum(s.spill_ms + (s.reload_ms or 0) for s in telemetry.spills)

    return {
        "global": {
            "ttdc_accuracy": ttdc_accuracy,
            "memory_accuracy": memory_accuracy,
            "predicted_ttdc_ms": telemetry.predicted_ttdc_ms,
            "actual_ttdc_ms": telemetry.actual_ttdc_ms,
            "predicted_memory_mb": telemetry.predicted_peak_memory_mb,
            "actual_memory_mb": telemetry.actual_peak_memory_mb,
        },
        "regions": {
            "count": len(telemetry.regions),
            "avg_compute_accuracy": (
                sum(region_compute_accuracies) / len(region_compute_accuracies)
                if region_compute_accuracies
                else 0.0
            ),
            "avg_memory_accuracy": (
                sum(region_memory_accuracies) / len(region_memory_accuracies)
                if region_memory_accuracies
                else 0.0
            ),
            "fallback_count": telemetry.fallback_count,
            "fallback_rate": (
                telemetry.fallback_count / len(telemetry.regions)
                if telemetry.regions
                else 0.0
            ),
        },
        "transfers": {
            "count": len(telemetry.transfers),
            "avg_accuracy": (
                sum(transfer_accuracies) / len(transfer_accuracies) if transfer_accuracies else 0.0
            ),
            "total_ms": total_transfer_ms,
            "overhead_pct": (
                100.0 * total_transfer_ms / total_compute_ms if total_compute_ms > 0 else 0.0
            ),
        },
        "backends": {
            "distribution": telemetry.backend_distribution,
            "switches": telemetry.backend_switches,
            "avg_time_ms": backend_avg_ms,
        },
        "sorts": {
            "count": len(telemetry.sorts),
            "total_ms": total_sort_ms,
            "overhead_pct": (
                100.0 * total_sort_ms / total_compute_ms if total_compute_ms > 0 else 0.0
            ),
        },
        "spills": {
            "count": len(telemetry.spills),
            "total_ms": total_spill_ms,
            "overhead_pct": (
                100.0 * total_spill_ms / total_compute_ms if total_compute_ms > 0 else 0.0
            ),
        },
        "source_scans": {
            "count": len(telemetry.source_scans),
            "total_bytes": sum(s.actual_bytes for s in telemetry.source_scans),
            "total_ms": sum(s.actual_ms for s in telemetry.source_scans),
            "avg_throughput_mbps": (
                sum(s.throughput_mbps for s in telemetry.source_scans)
                / len(telemetry.source_scans)
                if telemetry.source_scans
                else 0.0
            ),
        },
    }


def format_telemetry_report(telemetry: BatchExecutionTelemetry) -> str:
    """格式化 telemetry 为人类可读报告 (P2-009)。"""
    summary = summarize_telemetry(telemetry)
    lines = ["=" * 80, "Batch Execution Telemetry Report", "=" * 80, ""]

    # 全局统计
    lines.append(f"Batch ID: {telemetry.batch_id}")
    lines.append(f"Plan Hash: {telemetry.plan_hash}")
    lines.append(f"Execution Mode: {telemetry.execution_mode}")
    lines.append("")

    # 预测准确率
    glob = summary["global"]
    lines.append("Global Prediction Accuracy:")
    lines.append(f"  TTDC: {glob['ttdc_accuracy']:.1%}")
    lines.append(
        f"    Predicted: {glob['predicted_ttdc_ms']:.1f}ms, "
        f"Actual: {glob['actual_ttdc_ms']:.1f}ms"
    )
    lines.append(f"  Memory: {glob['memory_accuracy']:.1%}")
    lines.append(
        f"    Predicted: {glob['predicted_memory_mb']:.1f}MB, "
        f"Actual: {glob['actual_memory_mb']:.1f}MB"
    )
    lines.append("")

    # Region 统计
    reg = summary["regions"]
    lines.append(f"Regions: {reg['count']}")
    lines.append(f"  Avg compute accuracy: {reg['avg_compute_accuracy']:.1%}")
    lines.append(f"  Avg memory accuracy: {reg['avg_memory_accuracy']:.1%}")
    lines.append(f"  Fallbacks: {reg['fallback_count']} ({reg['fallback_rate']:.1%})")
    lines.append("")

    # Backend 分布
    be = summary["backends"]
    lines.append("Backend Distribution:")
    for backend, count in sorted(be["distribution"].items()):
        avg_ms = be["avg_time_ms"].get(backend, 0.0)
        lines.append(f"  {backend}: {count} regions, avg {avg_ms:.1f}ms")
    lines.append(f"  Backend switches: {be['switches']}")
    lines.append("")

    # Transfer/Sort/Spill
    trans = summary["transfers"]
    lines.append(f"Transfers: {trans['count']}")
    lines.append(
        f"  Total: {trans['total_ms']:.1f}ms ({trans['overhead_pct']:.1f}% overhead)"
    )
    lines.append(f"  Avg accuracy: {trans['avg_accuracy']:.1%}")
    lines.append("")

    sorts = summary["sorts"]
    lines.append(f"Sorts: {sorts['count']}")
    lines.append(
        f"  Total: {sorts['total_ms']:.1f}ms ({sorts['overhead_pct']:.1f}% overhead)"
    )
    lines.append("")

    spills = summary["spills"]
    lines.append(f"Spills: {spills['count']}")
    lines.append(
        f"  Total: {spills['total_ms']:.1f}ms ({spills['overhead_pct']:.1f}% overhead)"
    )
    lines.append("")

    # Source scans
    scans = summary["source_scans"]
    lines.append(f"Source Scans: {scans['count']}")
    lines.append(f"  Total bytes: {scans['total_bytes'] // 1024 // 1024}MB")
    lines.append(f"  Total time: {scans['total_ms']:.1f}ms")
    lines.append(f"  Avg throughput: {scans['avg_throughput_mbps']:.1f}MB/s")
    lines.append("")

    # 内存预算
    if telemetry.memory_budget_mb:
        status = "✓ WITHIN" if telemetry.within_memory_budget else "✗ EXCEEDED"
        lines.append(f"Memory Budget: {telemetry.memory_budget_mb:.1f}MB {status}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


class RegionTelemetryCollector:
    """Region 执行遥测收集器 (P2-009)。

    用于运行时收集 Region/transfer/sort/spill/scan 事件。
    """

    def __init__(self) -> None:
        self.regions: list[RegionExecutionTelemetry] = []
        self.transfers: list[TransferEdgeTelemetry] = []
        self.sorts: list[SortEventTelemetry] = []
        self.spills: list[SpillEventTelemetry] = []
        self.source_scans: list[SourceScanTelemetry] = []
        self.batch_started_at: float = 0.0
        self.batch_finished_at: float = 0.0

    def start_batch(self) -> None:
        """标记 batch 开始。"""
        self.batch_started_at = time.time()

    def finish_batch(self) -> None:
        """标记 batch 结束。"""
        self.batch_finished_at = time.time()

    def record_region(self, telemetry: RegionExecutionTelemetry) -> None:
        """记录单个 Region 执行遥测。"""
        self.regions.append(telemetry)

    def record_transfer(self, telemetry: TransferEdgeTelemetry) -> None:
        """记录 TransferEdge 遥测。"""
        self.transfers.append(telemetry)

    def record_sort(self, telemetry: SortEventTelemetry) -> None:
        """记录 Sort 事件遥测。"""
        self.sorts.append(telemetry)

    def record_spill(self, telemetry: SpillEventTelemetry) -> None:
        """记录 Spill 事件遥测。"""
        self.spills.append(telemetry)

    def record_source_scan(self, telemetry: SourceScanTelemetry) -> None:
        """记录 Source scan 遥测。"""
        self.source_scans.append(telemetry)

    def build_batch_telemetry(
        self,
        batch_id: str,
        plan_hash: str,
        predicted_ttdc_ms: float,
        predicted_peak_memory_mb: float,
        actual_peak_memory_mb: float,
        backend_distribution: dict[str, int],
        backend_switches: int,
        execution_mode: str = "production",
        memory_budget_mb: float | None = None,
        hardware_fingerprint: dict[str, str] | None = None,
    ) -> BatchExecutionTelemetry:
        """构建完整的 BatchExecutionTelemetry (P2-009)。"""
        actual_ttdc_ms = (self.batch_finished_at - self.batch_started_at) * 1000.0

        fallback_count = sum(1 for r in self.regions if r.fallback_occurred)

        within_budget = True
        if memory_budget_mb is not None:
            within_budget = actual_peak_memory_mb <= memory_budget_mb

        return BatchExecutionTelemetry(
            batch_id=batch_id,
            plan_hash=plan_hash,
            total_regions=len(self.regions),
            total_transfers=len(self.transfers),
            total_sorts=len(self.sorts),
            total_spills=len(self.spills),
            total_source_scans=len(self.source_scans),
            predicted_ttdc_ms=predicted_ttdc_ms,
            actual_ttdc_ms=actual_ttdc_ms,
            predicted_peak_memory_mb=predicted_peak_memory_mb,
            actual_peak_memory_mb=actual_peak_memory_mb,
            backend_distribution=backend_distribution,
            backend_switches=backend_switches,
            fallback_count=fallback_count,
            regions=tuple(self.regions),
            transfers=tuple(self.transfers),
            sorts=tuple(self.sorts),
            spills=tuple(self.spills),
            source_scans=tuple(self.source_scans),
            batch_started_at=self.batch_started_at,
            batch_finished_at=self.batch_finished_at,
            within_memory_budget=within_budget,
            memory_budget_mb=memory_budget_mb,
            execution_mode=execution_mode,
            hardware_fingerprint=hardware_fingerprint or {},
        )
