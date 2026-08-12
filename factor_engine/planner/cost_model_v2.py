# -*- coding: utf-8 -*-
"""§27-30: Cost Model V2 —— 使用 DataShapeEstimate metadata-only shape + 统一成本组件。

R27-30 要求：
    1. 使用 DataShapeEstimate（metadata-only，去掉固定 3000 instruments）。
    2. 成本函数拆成单一 authority 的 components（§30）：
       C_total = C_source + C_compute + C_transfer + C_reshape + C_sort +
                 C_materialize + C_spill + C_schedule + C_memory_risk + C_sink
    3. 审计当前 double-count（§31）：operator backend cost / _one_conversion_penalty /
       predict_ttdc conversion / predict_ttdc execute / materialize。

设计：
    - :class:`CostComponentsV2`: frozen dataclass，每个成本组件只加一次。
    - :func:`estimate_operator_cost_v2`: 使用 DataShapeEstimate + backend.operator_cost。
    - :func:`estimate_source_cost_v2`: 从 DataAccess ScanCost 估算 source 读取成本。
    - :func:`estimate_transfer_cost_v2`: backend 间数据传输成本（避免 double-count）。
    - :func:`estimate_total_cost_v2`: 汇总所有组件，审计 double-count。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.operator_cost import CostContext, _cost_spec_for_registered, _resolve_canonical_name
from planner.data_shape_estimate import DataShapeEstimate, shape_to_cost_context

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostComponentsV2:
    """§30: 成本函数拆成单一 authority 的 components（任何 component 只能加一次）。

    字段：
        source_cost_ms: 数据源扫描成本（ScanCost）
        compute_cost_ms: 算子计算成本（operator backend cost）
        transfer_cost_ms: backend 间数据传输成本（conversion penalty，去掉 double-count）
        reshape_cost_ms: 数据 reshape 成本（pivot / melt / stack / unstack）
        sort_cost_ms: 排序成本（order by / sort_values）
        materialize_cost_ms: 物化成本（cache / persist / materialize）
        spill_cost_ms: spill 成本（内存不足时 spill to disk）
        schedule_cost_ms: 调度成本（task spawn / wait / synchronization）
        memory_risk_cost_ms: 内存风险成本（OOM risk penalty）
        sink_cost_ms: 输出成本（write to file / sink to database）
        total_cost_ms: 总成本（所有组件和，去掉 double-count）
        peak_memory_bytes: 峰值内存（字节）
        output_bytes: 输出字节数
    """

    source_cost_ms: float = 0.0
    compute_cost_ms: float = 0.0
    transfer_cost_ms: float = 0.0
    reshape_cost_ms: float = 0.0
    sort_cost_ms: float = 0.0
    materialize_cost_ms: float = 0.0
    spill_cost_ms: float = 0.0
    schedule_cost_ms: float = 0.0
    memory_risk_cost_ms: float = 0.0
    sink_cost_ms: float = 0.0
    total_cost_ms: float = 0.0
    peak_memory_bytes: int = 0
    output_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_cost_ms": round(self.source_cost_ms, 3),
            "compute_cost_ms": round(self.compute_cost_ms, 3),
            "transfer_cost_ms": round(self.transfer_cost_ms, 3),
            "reshape_cost_ms": round(self.reshape_cost_ms, 3),
            "sort_cost_ms": round(self.sort_cost_ms, 3),
            "materialize_cost_ms": round(self.materialize_cost_ms, 3),
            "spill_cost_ms": round(self.spill_cost_ms, 3),
            "schedule_cost_ms": round(self.schedule_cost_ms, 3),
            "memory_risk_cost_ms": round(self.memory_risk_cost_ms, 3),
            "sink_cost_ms": round(self.sink_cost_ms, 3),
            "total_cost_ms": round(self.total_cost_ms, 3),
            "peak_memory_bytes": self.peak_memory_bytes,
            "output_bytes": self.output_bytes,
        }


def estimate_operator_cost_v2(
    operator_name: str,
    *,
    shape: DataShapeEstimate,
    backend: str = "pandas",
    window: int | None = None,
    feature_dim: int | None = None,
    k: int | None = None,
) -> CostComponentsV2:
    """§27-30: 使用 DataShapeEstimate 估算算子成本（去掉固定 3000 instruments）。

    参数:
        operator_name: 算子名称（canonical name）
        shape: DataShapeEstimate 实例（metadata-only shape）
        backend: 执行后端（pandas/polars/duckdb/numba，可选）
        window: 窗口参数（可选，O(NW) 算子）
        feature_dim: 特征维度（可选，O(NK^2) 算子）
        k: K 参数（可选，group/neutralize）

    返回:
        CostComponentsV2 实例
    """
    canonical_name = _resolve_canonical_name(operator_name)
    cost_spec = _cost_spec_for_registered(canonical_name)

    # 构造 CostContext（使用 DataShapeEstimate，§28：去掉固定 3000）
    ctx_dict = shape_to_cost_context(shape)
    if window is not None:
        ctx_dict["window"] = window
    if feature_dim is not None:
        ctx_dict["feature_dim"] = feature_dim
    if k is not None:
        ctx_dict["k"] = k
    ctx = CostContext(**ctx_dict)

    # 基础成本（O(N) / O(NW) / O(NK^2)）
    rows = shape.estimated_rows
    instruments = shape.estimated_instruments
    compute_ms = 0.0

    if cost_spec is None:
        # 未登记算子，默认 O(N) 中等成本
        compute_ms = rows / 1_000_000 * 10.0  # 10ms per million rows
    else:
        complexity = cost_spec.time_complexity
        if "N log N" in complexity:
            compute_ms = rows * (1 + 0.5 * (rows / 1_000_000)) / 1_000_000 * 20.0
        elif "NW" in complexity:
            w = window or 20
            compute_ms = rows * w / 1_000_000 * 5.0
        elif "NK^2" in complexity or "NK2" in complexity:
            k_val = k or feature_dim or 10
            compute_ms = rows * k_val * k_val / 1_000_000 * 50.0
        elif "O(N)" in complexity:
            compute_ms = rows / 1_000_000 * 8.0
        else:
            compute_ms = rows / 1_000_000 * 10.0

    # backend overhead
    backend_overhead = {"pandas": 1.0, "polars": 0.5, "duckdb": 0.3, "numba": 0.2}.get(
        backend, 1.0
    )
    compute_ms *= backend_overhead

    # 峰值内存（rows × instruments × 8 bytes per float64）
    peak_memory = int(rows * 8 * 1.5)  # 1.5x for intermediate results
    output_bytes = int(rows * 8)

    return CostComponentsV2(
        compute_cost_ms=compute_ms,
        total_cost_ms=compute_ms,
        peak_memory_bytes=peak_memory,
        output_bytes=output_bytes,
    )


def estimate_source_cost_v2(
    *,
    shape: DataShapeEstimate,
    scan_cost: Any | None = None,
    projected_columns: int | None = None,
) -> CostComponentsV2:
    """§30: 数据源扫描成本（从 DataAccess ScanCost 估算，避免 double-count）。

    参数:
        shape: DataShapeEstimate 实例
        scan_cost: DataAccess ScanCost 实例（可选）
        projected_columns: 投影列数（可选）

    返回:
        CostComponentsV2 实例
    """
    if scan_cost is not None:
        # 真实 ScanCost
        source_ms = float(
            getattr(scan_cost, "estimate_ms", None)
            or getattr(scan_cost, "engine_startup_ms", 0.0)
        )
        scan_bytes = int(
            getattr(scan_cost, "selected_bytes", None) or shape.estimated_bytes
        )
    else:
        # metadata-only 估算
        if shape.remote:
            # 远程存储：网络延迟 + 传输
            source_ms = 50.0 + shape.estimated_bytes / (100 * 1024 * 1024)  # 100 MB/s
        else:
            # 本地存储：SSD/NVMe
            if shape.storage_kind == "memory":
                source_ms = shape.estimated_bytes / (5 * 1024 * 1024 * 1024)  # 5 GB/s
            elif shape.storage_kind in {"nvme", "ssd"}:
                source_ms = shape.estimated_bytes / (2 * 1024 * 1024 * 1024)  # 2 GB/s
            else:
                source_ms = shape.estimated_bytes / (500 * 1024 * 1024)  # 500 MB/s
        scan_bytes = shape.estimated_bytes

    return CostComponentsV2(
        source_cost_ms=source_ms,
        total_cost_ms=source_ms,
        peak_memory_bytes=int(scan_bytes * 1.2),
        output_bytes=scan_bytes,
    )


def estimate_transfer_cost_v2(
    *,
    shape: DataShapeEstimate,
    from_backend: str,
    to_backend: str,
) -> CostComponentsV2:
    """§30-31: backend 间数据传输成本（去掉 double-count：只算一次 conversion penalty）。

    参数:
        shape: DataShapeEstimate 实例
        from_backend: 源 backend（pandas/polars/duckdb）
        to_backend: 目标 backend

    返回:
        CostComponentsV2 实例
    """
    if from_backend == to_backend:
        # 同 backend 无传输成本
        return CostComponentsV2()

    # conversion penalty（一次性，去掉 double-count）
    bytes_to_transfer = shape.estimated_bytes
    if from_backend == "polars" and to_backend == "pandas":
        # Polars → Pandas: to_pandas() zero-copy or minimal copy
        transfer_ms = bytes_to_transfer / (2 * 1024 * 1024 * 1024)  # 2 GB/s
    elif from_backend == "pandas" and to_backend == "polars":
        # Pandas → Polars: pl.from_pandas() minimal copy
        transfer_ms = bytes_to_transfer / (1.5 * 1024 * 1024 * 1024)  # 1.5 GB/s
    elif from_backend == "duckdb" and to_backend in {"pandas", "polars"}:
        # DuckDB → Pandas/Polars: fetch_df() / pl()
        transfer_ms = bytes_to_transfer / (1 * 1024 * 1024 * 1024)  # 1 GB/s
    elif to_backend == "duckdb":
        # Pandas/Polars → DuckDB: register / from_df
        transfer_ms = bytes_to_transfer / (800 * 1024 * 1024)  # 800 MB/s
    else:
        # 默认 conversion penalty
        transfer_ms = bytes_to_transfer / (500 * 1024 * 1024)  # 500 MB/s

    return CostComponentsV2(
        transfer_cost_ms=transfer_ms,
        total_cost_ms=transfer_ms,
        peak_memory_bytes=int(bytes_to_transfer * 2.0),  # 2x for double buffering
        output_bytes=bytes_to_transfer,
    )


def estimate_materialize_cost_v2(
    *,
    shape: DataShapeEstimate,
    backend: str = "pandas",
) -> CostComponentsV2:
    """§30: 物化成本（cache / persist / materialize）。

    参数:
        shape: DataShapeEstimate 实例
        backend: 执行后端（可选）

    返回:
        CostComponentsV2 实例
    """
    bytes_to_materialize = shape.estimated_bytes
    if backend == "polars":
        # Polars LazyFrame.collect()
        materialize_ms = bytes_to_materialize / (3 * 1024 * 1024 * 1024)  # 3 GB/s
    elif backend == "duckdb":
        # DuckDB result materialization
        materialize_ms = bytes_to_materialize / (2 * 1024 * 1024 * 1024)  # 2 GB/s
    else:
        # Pandas copy
        materialize_ms = bytes_to_materialize / (1 * 1024 * 1024 * 1024)  # 1 GB/s

    return CostComponentsV2(
        materialize_cost_ms=materialize_ms,
        total_cost_ms=materialize_ms,
        peak_memory_bytes=int(bytes_to_materialize * 2.0),  # 2x for intermediate
        output_bytes=bytes_to_materialize,
    )


def estimate_total_cost_v2(
    *,
    source_cost: CostComponentsV2 | None = None,
    compute_cost: CostComponentsV2 | None = None,
    transfer_cost: CostComponentsV2 | None = None,
    materialize_cost: CostComponentsV2 | None = None,
    reshape_cost: CostComponentsV2 | None = None,
    sort_cost: CostComponentsV2 | None = None,
    sink_cost: CostComponentsV2 | None = None,
) -> CostComponentsV2:
    """§30-31: 汇总所有成本组件（审计 double-count，每个组件只加一次）。

    参数:
        source_cost: 数据源扫描成本（可选）
        compute_cost: 算子计算成本（可选）
        transfer_cost: backend 间传输成本（可选）
        materialize_cost: 物化成本（可选）
        reshape_cost: reshape 成本（可选）
        sort_cost: 排序成本（可选）
        sink_cost: 输出成本（可选）

    返回:
        CostComponentsV2 实例（汇总后）
    """
    total_source = (source_cost.source_cost_ms if source_cost else 0.0)
    total_compute = (compute_cost.compute_cost_ms if compute_cost else 0.0)
    total_transfer = (transfer_cost.transfer_cost_ms if transfer_cost else 0.0)
    total_materialize = (materialize_cost.materialize_cost_ms if materialize_cost else 0.0)
    total_reshape = (reshape_cost.reshape_cost_ms if reshape_cost else 0.0)
    total_sort = (sort_cost.sort_cost_ms if sort_cost else 0.0)
    total_sink = (sink_cost.sink_cost_ms if sink_cost else 0.0)

    total_ms = (
        total_source
        + total_compute
        + total_transfer
        + total_materialize
        + total_reshape
        + total_sort
        + total_sink
    )

    # 峰值内存（取最大）
    peak_memory = max(
        (source_cost.peak_memory_bytes if source_cost else 0),
        (compute_cost.peak_memory_bytes if compute_cost else 0),
        (transfer_cost.peak_memory_bytes if transfer_cost else 0),
        (materialize_cost.peak_memory_bytes if materialize_cost else 0),
    )

    # 输出字节数（取最后一个有效值）
    output_bytes = 0
    for cost in [sink_cost, materialize_cost, compute_cost, source_cost]:
        if cost and cost.output_bytes > 0:
            output_bytes = cost.output_bytes
            break

    return CostComponentsV2(
        source_cost_ms=total_source,
        compute_cost_ms=total_compute,
        transfer_cost_ms=total_transfer,
        reshape_cost_ms=total_reshape,
        sort_cost_ms=total_sort,
        materialize_cost_ms=total_materialize,
        spill_cost_ms=0.0,
        schedule_cost_ms=0.0,
        memory_risk_cost_ms=0.0,
        sink_cost_ms=total_sink,
        total_cost_ms=total_ms,
        peak_memory_bytes=peak_memory,
        output_bytes=output_bytes,
    )
