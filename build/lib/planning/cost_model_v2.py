# -*- coding: utf-8 -*-
"""MB-P1-005/006/010: Cost model v2 - single authority, no double-count.

Fixes:
    - MB-P1-005: Conversion penalty vs TTDC conversion (no double-count)
    - MB-P1-006: Operator compute vs TTDC execute (no double-count)
    - MB-P1-010: Shared benefit from real DAG (not fixed 10%)
    - MB-P1-011: Scheduler/DQ/write cost as calibratable components

Design:
    Cost_total = Cost_source + Cost_compute + Cost_transfer + Cost_reshape
                 + Cost_sort + Cost_materialize + Cost_spill + Cost_schedule
                 + Cost_memory_risk + Cost_sink

    Each component added exactly once. No overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from planning.memory_model import DataShapeEstimate, EdgeMemoryCost


@dataclass(frozen=True)
class CostBreakdown:
    """MB-P1-011, §30: Single-authority cost breakdown.

    Each component counted exactly once. No double-counting.
    All components are calibratable.
    """

    source_scan_ms: float = 0.0
    compute_ms: float = 0.0
    transfer_ms: float = 0.0
    reshape_ms: float = 0.0
    sort_ms: float = 0.0
    materialize_ms: float = 0.0
    spill_ms: float = 0.0
    schedule_overhead_ms: float = 0.0
    memory_risk_penalty_ms: float = 0.0
    sink_write_ms: float = 0.0

    def total_ms(self) -> float:
        """Total estimated cost."""
        return (
            self.source_scan_ms
            + self.compute_ms
            + self.transfer_ms
            + self.reshape_ms
            + self.sort_ms
            + self.materialize_ms
            + self.spill_ms
            + self.schedule_overhead_ms
            + self.memory_risk_penalty_ms
            + self.sink_write_ms
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_scan_ms": round(self.source_scan_ms, 3),
            "compute_ms": round(self.compute_ms, 3),
            "transfer_ms": round(self.transfer_ms, 3),
            "reshape_ms": round(self.reshape_ms, 3),
            "sort_ms": round(self.sort_ms, 3),
            "materialize_ms": round(self.materialize_ms, 3),
            "spill_ms": round(self.spill_ms, 3),
            "schedule_overhead_ms": round(self.schedule_overhead_ms, 3),
            "memory_risk_penalty_ms": round(self.memory_risk_penalty_ms, 3),
            "sink_write_ms": round(self.sink_write_ms, 3),
            "total_ms": round(self.total_ms(), 3),
        }


@dataclass(frozen=True)
class SharedNodeBenefit:
    """MB-P1-010, §44: Shared benefit from real DAG topology.

    Not fixed 10% heuristic. Derived from:
        - Avoided source scans
        - Avoided shared compute
        - Avoided transfer
    """

    avoided_source_scans_ms: float
    avoided_compute_ms: float
    avoided_transfer_ms: float
    shared_node_count: int
    total_consumer_count: int

    def total_saved_ms(self) -> float:
        """Total time saved by sharing."""
        return (
            self.avoided_source_scans_ms
            + self.avoided_compute_ms
            + self.avoided_transfer_ms
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "avoided_source_scans_ms": round(self.avoided_source_scans_ms, 3),
            "avoided_compute_ms": round(self.avoided_compute_ms, 3),
            "avoided_transfer_ms": round(self.avoided_transfer_ms, 3),
            "shared_node_count": self.shared_node_count,
            "total_consumer_count": self.total_consumer_count,
            "total_saved_ms": round(self.total_saved_ms(), 3),
        }


def estimate_source_scan_cost(
    *,
    shape: DataShapeEstimate,
    projected_columns: int,
    remote: bool = False,
    predicate_selectivity: float = 1.0,
) -> float:
    """MB-P1-019, §41: Source scan cost (not O(1) column).

    Args:
        shape: Data shape estimate
        projected_columns: Number of columns to scan
        remote: Remote storage (COS, S3)
        predicate_selectivity: Filter selectivity (0.0-1.0)

    Returns:
        Estimated scan time in milliseconds
    """
    selected_bytes = int(
        shape.estimated_rows
        * projected_columns
        * shape.average_row_width_bytes
        / max(1, shape.estimated_columns)
        * predicate_selectivity
    )

    if remote:
        # Remote: network latency + bandwidth
        latency_ms = 50.0  # ~50ms baseline latency
        bandwidth_gbps = 1.0  # ~1Gbps typical
        transfer_ms = (selected_bytes / (1024**3)) / bandwidth_gbps * 8000.0
        return latency_ms + transfer_ms
    else:
        # Local: disk I/O
        # NVMe: ~3GB/s, SATA SSD: ~500MB/s
        throughput_gbps = 2.0  # ~2GB/s conservative
        return (selected_bytes / (1024**3)) / throughput_gbps * 1000.0


def estimate_compute_cost(
    *,
    operator: str,
    backend: str,
    shape: DataShapeEstimate,
    window: int | None = None,
    feature_dim: int | None = None,
    sparsity: float | None = None,
    cpu_cores: int = 8,
) -> float:
    """MB-P1-006, §30: Pure compute cost (no conversion/materialize).

    OPTIMIZED: Added operator-specific costs, data-dependent scaling, hardware-aware cost.

    This is the ONLY place compute cost is added. No double-counting.

    Args:
        operator: Operator name
        backend: Backend name
        shape: Data shape
        window: Window parameter (for O(NW) operators)
        feature_dim: Feature dimension (for O(NK^2) operators)
        sparsity: Data sparsity (0.0-1.0, None = infer from shape)
        cpu_cores: Available CPU cores

    Returns:
        Estimated compute time in milliseconds
    """
    try:
        from backend.operator_cost import CostContext, estimate_backend_cost

        ctx = CostContext(
            rows=shape.estimated_rows,
            instruments=shape.estimated_instruments,
            window=window,
            k=feature_dim,
            feature_dim=feature_dim,
        )
        base_cost = estimate_backend_cost(
            operator,
            backend,
            row_count_estimate=shape.estimated_rows,
            requires_conversion=False,  # Conversion counted separately
            cost_ctx=ctx,
        )
    except Exception:
        # Fallback: operator-specific cost functions
        base_cost = _estimate_operator_cost_fallback(operator, shape, window, feature_dim)

    # Data-dependent scaling: sparsity adjustment
    if sparsity is None:
        sparsity = 1.0 - shape.density if hasattr(shape, 'density') else 1.0

    sparsity_factor = 1.0
    if sparsity > 0.5:
        # High sparsity benefits certain operations
        if operator in ("multiply", "add", "mask"):
            sparsity_factor = 0.7 + 0.3 * (1.0 - sparsity)

    # Hardware-aware scaling: CPU cores for parallel backends
    parallel_factor = 1.0
    if backend in ("polars_lazy", "polars_eager", "duckdb_sql"):
        # These backends can utilize multiple cores
        effective_cores = min(cpu_cores, 16)  # Cap at 16 for diminishing returns
        parallel_efficiency = 0.7 + 0.02 * effective_cores  # 70-100% efficiency
        parallel_factor = 1.0 / (effective_cores * parallel_efficiency)

    # Non-linear scaling for large data
    scaling_factor = 1.0
    if shape.estimated_rows > 10_000_000:
        # Large data gets better cache locality
        scaling_factor = 0.9
    elif shape.estimated_rows > 100_000_000:
        scaling_factor = 0.85

    return base_cost * sparsity_factor * parallel_factor * scaling_factor


def _estimate_operator_cost_fallback(
    operator: str,
    shape: DataShapeEstimate,
    window: int | None,
    feature_dim: int | None,
) -> float:
    """Operator-specific cost estimation fallback.

    OPTIMIZED: More accurate cost functions per operator type.
    """
    rows = shape.estimated_rows
    op_lower = operator.lower()

    # Elementwise operations: O(N)
    if any(op in op_lower for op in ["add", "multiply", "divide", "abs", "log"]):
        return rows / 10_000_000.0 * 10.0  # ~10ms per 10M rows

    # Window operations: O(N*W)
    if any(op in op_lower for op in ["rolling", "ts_", "mavg", "ewm"]):
        w = window if window else 20
        return rows * w / 50_000_000.0  # ~1ms per 50M row-window ops

    # Cross-sectional operations: O(N*log(N))
    if any(op in op_lower for op in ["rank", "zscore", "cs_"]):
        import math
        if rows > 0:
            return rows * math.log2(max(2, rows)) / 5_000_000.0

    # Regression/neutralization: O(N*K^2)
    if any(op in op_lower for op in ["regress", "neutralize", "resid"]):
        k = feature_dim if feature_dim else 5
        return rows * (k ** 2) / 20_000_000.0

    # Correlation/covariance: O(N*W*K)
    if any(op in op_lower for op in ["corr", "cov", "beta"]):
        w = window if window else 20
        k = feature_dim if feature_dim else 2
        return rows * w * k / 30_000_000.0

    # Default: simple O(N)
    return rows / 5_000_000.0 * 20.0  # ~20ms per 5M rows


def estimate_transfer_cost(
    *,
    edge: EdgeMemoryCost,
) -> float:
    """MB-P1-005/007, §32: Transfer cost (no conversion double-count).

    This is the ONLY place conversion cost is added.

    Args:
        edge: Edge memory cost with transfer details

    Returns:
        Total transfer time in milliseconds
    """
    return edge.total_transfer_ms()


def estimate_materialize_cost(
    *,
    shape: DataShapeEstimate,
    format: str = "arrow",
) -> float:
    """MB-P1-005, §30: Materialization cost (separate from conversion).

    Args:
        shape: Data shape
        format: arrow, parquet, etc.

    Returns:
        Materialization time in milliseconds
    """
    bytes_size = shape.estimated_bytes

    if format == "arrow":
        # Arrow: fast serialization (~2GB/s)
        return (bytes_size / (1024**3)) / 2.0 * 1000.0
    elif format == "parquet":
        # Parquet: compression (~500MB/s)
        return (bytes_size / (1024**3)) / 0.5 * 1000.0
    else:
        # Generic pickle
        return (bytes_size / (1024**3)) / 1.0 * 1000.0


def estimate_spill_cost(
    *,
    bytes_to_spill: int,
    disk_speed_class: str = "ssd",
) -> float:
    """MB-P1-015, §48: Spill cost.

    Args:
        bytes_to_spill: Bytes to write to disk
        disk_speed_class: nvme, ssd, hdd

    Returns:
        Spill time in milliseconds
    """
    speeds = {
        "nvme": 3.0,  # 3GB/s
        "ssd": 0.5,  # 500MB/s
        "hdd": 0.1,  # 100MB/s
    }
    throughput_gbps = speeds.get(disk_speed_class, 0.5)
    return (bytes_to_spill / (1024**3)) / throughput_gbps * 1000.0


def estimate_schedule_overhead(
    *,
    task_count: int,
    region_count: int,
) -> float:
    """MB-P1-011, §30: Scheduler overhead.

    Args:
        task_count: Number of tasks
        region_count: Number of regions

    Returns:
        Scheduling overhead in milliseconds
    """
    # Python GIL + queue management
    per_task_ms = 0.1  # ~0.1ms per task
    per_region_ms = 1.0  # ~1ms per region boundary
    return task_count * per_task_ms + region_count * per_region_ms


def estimate_sink_write_cost(
    *,
    shape: DataShapeEstimate,
    sink_format: str = "parquet",
    remote: bool = False,
) -> float:
    """MB-P1-025, §50: Sink write cost.

    Args:
        shape: Data shape
        sink_format: parquet, arrow, csv
        remote: Remote storage

    Returns:
        Write time in milliseconds
    """
    bytes_size = shape.estimated_bytes

    # Serialization
    if sink_format == "parquet":
        serialize_ms = (bytes_size / (1024**3)) / 0.5 * 1000.0  # ~500MB/s
    elif sink_format == "arrow":
        serialize_ms = (bytes_size / (1024**3)) / 2.0 * 1000.0  # ~2GB/s
    else:
        serialize_ms = (bytes_size / (1024**3)) / 1.0 * 1000.0  # ~1GB/s

    # I/O
    if remote:
        io_ms = (bytes_size / (1024**3)) / 1.0 * 8000.0  # ~1Gbps
    else:
        io_ms = (bytes_size / (1024**3)) / 2.0 * 1000.0  # ~2GB/s

    return serialize_ms + io_ms


def estimate_memory_risk_penalty(
    *,
    peak_bytes: int,
    available_bytes: int,
    safety_margin: float = 0.8,
) -> float:
    """MB-P1-008, §30: Memory risk penalty.

    If peak exceeds safety margin, add penalty for potential OOM/spill.

    Args:
        peak_bytes: Peak memory estimate
        available_bytes: Available memory budget
        safety_margin: Safety factor (0.0-1.0)

    Returns:
        Penalty in milliseconds
    """
    safe_limit = int(available_bytes * safety_margin)
    if peak_bytes <= safe_limit:
        return 0.0

    # Excess memory: assume spill or OOM risk
    excess_ratio = (peak_bytes - safe_limit) / safe_limit
    # Penalty: exponential in excess ratio
    return 100.0 * (2.0**excess_ratio - 1.0)


def compute_shared_benefit(
    *,
    shared_nodes: list[dict[str, Any]],
    reuse_counts: dict[str, int],
) -> SharedNodeBenefit:
    """MB-P1-010, §44: Compute shared benefit from DAG topology.

    Args:
        shared_nodes: List of shared node metadata
            [{"node_id": "n1", "compute_ms": 10.0, "scan_ms": 5.0, ...}, ...]
        reuse_counts: {"node_id": consumer_count, ...}

    Returns:
        SharedNodeBenefit with avoided costs
    """
    avoided_scans = 0.0
    avoided_compute = 0.0
    avoided_transfer = 0.0
    shared_count = 0
    total_consumers = 0

    for node in shared_nodes:
        node_id = node.get("node_id", "")
        reuse = reuse_counts.get(node_id, 1)
        if reuse <= 1:
            continue

        shared_count += 1
        total_consumers += reuse

        # Avoided: (reuse - 1) * cost
        # We only compute once, avoiding (reuse - 1) redundant computations
        avoided_scans += (reuse - 1) * node.get("scan_ms", 0.0)
        avoided_compute += (reuse - 1) * node.get("compute_ms", 0.0)
        avoided_transfer += (reuse - 1) * node.get("transfer_ms", 0.0)

    return SharedNodeBenefit(
        avoided_source_scans_ms=avoided_scans,
        avoided_compute_ms=avoided_compute,
        avoided_transfer_ms=avoided_transfer,
        shared_node_count=shared_count,
        total_consumer_count=total_consumers,
    )


def build_cost_breakdown(
    *,
    source_scan_ms: float = 0.0,
    compute_ms: float = 0.0,
    transfer_edges: list[EdgeMemoryCost] | None = None,
    materialize_ms: float = 0.0,
    spill_ms: float = 0.0,
    schedule_overhead_ms: float = 0.0,
    memory_risk_penalty_ms: float = 0.0,
    sink_write_ms: float = 0.0,
) -> CostBreakdown:
    """MB-P1-011, §30-31: Build unified cost breakdown.

    Each component added exactly once. Audit for double-counting.

    Args:
        source_scan_ms: Source scan cost
        compute_ms: Pure compute cost (no conversion)
        transfer_edges: List of transfer edges (conversion counted here)
        materialize_ms: Materialization cost
        spill_ms: Spill cost
        schedule_overhead_ms: Scheduler overhead
        memory_risk_penalty_ms: Memory risk penalty
        sink_write_ms: Sink write cost

    Returns:
        CostBreakdown with no double-counting
    """
    edges = transfer_edges or []

    # Transfer: sum of all edges (sort + repartition + cast + reshape)
    total_transfer = sum(e.total_transfer_ms() for e in edges)

    # Sort: already in edge costs
    total_sort = sum(e.sort_cost_ms for e in edges)

    # Reshape: already in edge costs
    total_reshape = sum(e.reshape_cost_ms for e in edges)

    # Note: transfer_ms includes sort + reshape; we separate for visibility
    # But total should not double-count
    transfer_only = total_transfer - total_sort - total_reshape

    return CostBreakdown(
        source_scan_ms=source_scan_ms,
        compute_ms=compute_ms,
        transfer_ms=transfer_only,
        reshape_ms=total_reshape,
        sort_ms=total_sort,
        materialize_ms=materialize_ms,
        spill_ms=spill_ms,
        schedule_overhead_ms=schedule_overhead_ms,
        memory_risk_penalty_ms=memory_risk_penalty_ms,
        sink_write_ms=sink_write_ms,
    )


def audit_double_count(breakdown: CostBreakdown) -> dict[str, Any]:
    """MB-P1-005/006, §31: Audit cost breakdown for double-counting.

    OPTIMIZED: Enhanced auditing with confidence intervals and visualization data.

    Returns:
        Audit report with warnings if potential overlap detected
    """
    warnings = []

    # Check for suspiciously high ratios
    total = breakdown.total_ms()
    if total > 0:
        compute_ratio = breakdown.compute_ms / total
        transfer_ratio = breakdown.transfer_ms / total

        if compute_ratio > 0.9:
            warnings.append("compute_ms dominates (>90%): verify no conversion inside")
        if transfer_ratio > 0.9:
            warnings.append("transfer_ms dominates (>90%): verify no compute inside")

        # Check for unrealistic total cost
        if total < 1.0 and breakdown.compute_ms > 0:
            warnings.append("Total cost < 1ms: may be underestimated")
        if total > 3600_000:  # > 1 hour
            warnings.append("Total cost > 1 hour: may be overestimated or problematic query")

    # Compute confidence intervals (simple heuristic-based)
    confidence_intervals = {
        "compute_ms": (breakdown.compute_ms * 0.7, breakdown.compute_ms * 1.5),
        "transfer_ms": (breakdown.transfer_ms * 0.8, breakdown.transfer_ms * 1.3),
        "total_ms": (total * 0.75, total * 1.4),
    }

    # Visualization data
    visualization = {
        "breakdown_pct": {
            "source_scan": round(breakdown.source_scan_ms / total * 100, 1) if total > 0 else 0,
            "compute": round(breakdown.compute_ms / total * 100, 1) if total > 0 else 0,
            "transfer": round(breakdown.transfer_ms / total * 100, 1) if total > 0 else 0,
            "materialize": round(breakdown.materialize_ms / total * 100, 1) if total > 0 else 0,
            "memory_risk": round(breakdown.memory_risk_penalty_ms / total * 100, 1) if total > 0 else 0,
            "other": round((breakdown.sort_ms + breakdown.reshape_ms + breakdown.spill_ms +
                          breakdown.schedule_overhead_ms + breakdown.sink_write_ms) / total * 100, 1) if total > 0 else 0,
        },
        "absolute_ms": breakdown.to_dict(),
    }

    return {
        "total_ms": round(total, 3),
        "component_ratios": {
            "source_scan": round(breakdown.source_scan_ms / total, 3) if total > 0 else 0,
            "compute": round(breakdown.compute_ms / total, 3) if total > 0 else 0,
            "transfer": round(breakdown.transfer_ms / total, 3) if total > 0 else 0,
            "materialize": round(breakdown.materialize_ms / total, 3) if total > 0 else 0,
        },
        "confidence_intervals": {
            k: (round(v[0], 2), round(v[1], 2)) for k, v in confidence_intervals.items()
        },
        "visualization": visualization,
        "warnings": warnings,
    }
