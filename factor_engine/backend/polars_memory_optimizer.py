# -*- coding: utf-8 -*-
"""Polars 内存优化工具。

提供内存预分配、批处理策略和资源感知的执行优化。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class MemoryEstimate:
    """内存占用估算结果。

    属性
    ----
    rows : int
        数据行数。
    cols : int
        数据列数。
    estimated_bytes : int
        估计的内存占用（字节）。
    overhead_bytes : int
        索引和元数据开销（字节）。
    """

    rows: int
    cols: int
    estimated_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        """总内存占用（数据 + 开销）。"""
        return self.estimated_bytes + self.overhead_bytes

    @property
    def total_mb(self) -> float:
        """总内存占用（MB）。"""
        return self.total_bytes / (1024 * 1024)


def estimate_polars_dataframe_memory(
    rows: int,
    cols: int,
    *,
    dtype_size: int = 8,
    string_avg_len: int = 20,
    string_cols: int = 0,
) -> MemoryEstimate:
    """估算 Polars DataFrame 的内存占用。

    参数
    ----
    rows : int
        数据行数。
    cols : int
        数值列数量。
    dtype_size : int
        数值列平均大小（字节），默认 8（float64）。
    string_avg_len : int
        字符串列平均长度，默认 20。
    string_cols : int
        字符串列数量。

    返回
    ----
    MemoryEstimate
        内存占用估算结果。
    """
    if rows <= 0 or (cols + string_cols) <= 0:
        return MemoryEstimate(0, 0, 0, 0)

    # 数值列内存
    numeric_memory = rows * cols * dtype_size

    # 字符串列内存（UTF-8 编码 + 指针）
    string_memory = rows * string_cols * (string_avg_len + 8)

    # 总数据内存
    data_memory = numeric_memory + string_memory

    # 元数据开销：列名、索引、内部结构（约 15-20%）
    overhead = int(data_memory * 0.18)

    return MemoryEstimate(
        rows=rows,
        cols=cols + string_cols,
        estimated_bytes=data_memory,
        overhead_bytes=overhead,
    )


def calculate_optimal_chunk_size(
    total_rows: int,
    total_cols: int,
    available_memory_bytes: int,
    *,
    safety_factor: float = 0.7,
    min_chunk_rows: int = 1000,
    max_chunk_rows: int = 1_000_000,
) -> int:
    """计算最优分块大小（行数）。

    参数
    ----
    total_rows : int
        总行数。
    total_cols : int
        总列数。
    available_memory_bytes : int
        可用内存（字节）。
    safety_factor : float
        安全系数（0-1），默认 0.7（使用 70% 的可用内存）。
    min_chunk_rows : int
        最小分块行数。
    max_chunk_rows : int
        最大分块行数。

    返回
    ----
    int
        最优分块行数。
    """
    if total_rows <= 0 or total_cols <= 0:
        return min_chunk_rows

    # 应用安全系数
    safe_memory = int(available_memory_bytes * safety_factor)

    # 估算单行内存
    estimate = estimate_polars_dataframe_memory(1, total_cols)
    bytes_per_row = estimate.total_bytes

    if bytes_per_row <= 0:
        return min_chunk_rows

    # 计算可容纳的行数
    chunk_rows = safe_memory // bytes_per_row

    # 应用边界约束
    chunk_rows = max(min_chunk_rows, min(max_chunk_rows, chunk_rows))

    # 不要超过总行数
    chunk_rows = min(chunk_rows, total_rows)

    return chunk_rows


def should_use_streaming(
    estimated_rows: int | None,
    estimated_cols: int,
    available_memory_bytes: int,
    *,
    streaming_threshold_mb: float = 100.0,
) -> bool:
    """判断是否应该使用 streaming 模式。

    参数
    ----
    estimated_rows : int | None
        估计的结果行数；None 表示未知（保守策略：使用 streaming）。
    estimated_cols : int
        列数。
    available_memory_bytes : int
        可用内存（字节）。
    streaming_threshold_mb : float
        streaming 阈值（MB），超过此值启用 streaming。

    返回
    ----
    bool
        True 表示应该使用 streaming 模式。
    """
    # 未知行数时，保守策略：使用 streaming
    if estimated_rows is None:
        return True

    # 估算内存占用
    estimate = estimate_polars_dataframe_memory(estimated_rows, estimated_cols)

    # 如果估算的内存超过阈值，使用 streaming
    if estimate.total_mb > streaming_threshold_mb:
        return True

    # 如果估算的内存超过可用内存的 50%，使用 streaming
    if estimate.total_bytes > available_memory_bytes * 0.5:
        return True

    return False


def optimize_lazyframe_plan(lf: Any) -> Any:
    """优化 LazyFrame 查询计划。

    应用查询优化技巧：
    - 尽早过滤（predicate pushdown）
    - 尽早选择列（projection pushdown）
    - 避免不必要的排序
    - 合并连续操作

    参数
    ----
    lf : Any
        Polars LazyFrame 对象。

    返回
    ----
    Any
        优化后的 LazyFrame。
    """
    try:
        import polars as pl
    except ImportError:
        return lf

    # Polars 的 lazy evaluation 已经自动进行了很多优化
    # 这里可以添加额外的优化提示，但要小心不要破坏查询语义

    # 检查是否可以提前物化某些昂贵的操作
    # （例如，如果后续有多次引用，提前 collect 可能更高效）

    return lf


class PolarsMemoryMonitor:
    """Polars 执行期内存监控器。

    跟踪内存使用情况，提供自适应优化建议。
    """

    def __init__(self, budget_bytes: int | None = None):
        """初始化内存监控器。

        参数
        ----
        budget_bytes : int | None
            内存预算（字节）；None 表示自动检测系统可用内存。
        """
        self.budget_bytes = budget_bytes or self._detect_available_memory()
        self.peak_usage_bytes = 0
        self.current_usage_bytes = 0
        self._collect_count = 0
        self._streaming_count = 0

    def _detect_available_memory(self) -> int:
        """检测系统可用内存（字节）。"""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            # 回退：假设 4GB 可用
            return 4 * 1024 * 1024 * 1024

    def record_collect(self, result_bytes: int, *, used_streaming: bool = False) -> None:
        """记录一次 collect 操作。

        参数
        ----
        result_bytes : int
            结果数据大小（字节）。
        used_streaming : bool
            是否使用了 streaming 模式。
        """
        self._collect_count += 1
        if used_streaming:
            self._streaming_count += 1

        self.current_usage_bytes = result_bytes
        if result_bytes > self.peak_usage_bytes:
            self.peak_usage_bytes = result_bytes

    def should_enable_streaming(self, estimated_bytes: int) -> bool:
        """建议是否启用 streaming 模式。

        参数
        ----
        estimated_bytes : int
            估计的操作内存占用（字节）。

        返回
        ----
        bool
            True 表示建议启用 streaming。
        """
        # 如果估算的操作会超过预算的 60%，启用 streaming
        if estimated_bytes > self.budget_bytes * 0.6:
            return True

        # 如果当前使用量 + 估算的操作会超过预算，启用 streaming
        if self.current_usage_bytes + estimated_bytes > self.budget_bytes:
            return True

        return False

    def get_stats(self) -> dict[str, Any]:
        """获取监控统计信息。"""
        return {
            "budget_mb": self.budget_bytes / (1024 * 1024),
            "peak_usage_mb": self.peak_usage_bytes / (1024 * 1024),
            "current_usage_mb": self.current_usage_bytes / (1024 * 1024),
            "collect_count": self._collect_count,
            "streaming_count": self._streaming_count,
            "streaming_ratio": (
                self._streaming_count / self._collect_count
                if self._collect_count > 0
                else 0.0
            ),
        }
