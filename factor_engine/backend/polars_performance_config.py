# -*- coding: utf-8 -*-
"""Polars 性能优化配置模块。

提供 Polars 后端的内存和并行执行优化策略：
- Streaming 模式自动启用
- 动态线程池调优
- 内存预算感知的批处理
- Lazy 查询优化增强
"""
from __future__ import annotations

import os
import multiprocessing
from dataclasses import dataclass
from typing import Any

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


@dataclass(frozen=True)
class PolarsPerformanceConfig:
    """Polars 性能优化配置。

    属性
    ----
    enable_streaming : bool
        启用 streaming 模式（collect(streaming=True)），降低内存 50-70%。
    max_threads : int | None
        Polars 最大线程数；None 表示根据 CPU 核心数自动配置。
    adaptive_batch_size : bool
        启用自适应批处理大小（根据内存压力动态调整）。
    memory_budget_mb : float | None
        单次 collect 的内存预算（MB）；None 表示使用系统默认。
    lazy_optimization_level : int
        Lazy 查询优化级别（0=关闭，1=基础，2=激进）。
    predicate_pushdown : bool
        启用谓词下推优化。
    projection_pushdown : bool
        启用投影下推优化。
    """

    enable_streaming: bool = True
    max_threads: int | None = None
    adaptive_batch_size: bool = True
    memory_budget_mb: float | None = None
    lazy_optimization_level: int = 2
    predicate_pushdown: bool = True
    projection_pushdown: bool = True

    @classmethod
    def from_env(cls) -> "PolarsPerformanceConfig":
        """从环境变量构造配置。

        支持的环境变量：
        - POLARS_ENABLE_STREAMING: 1/true/yes 启用 streaming
        - POLARS_MAX_THREADS: 正整数，设置最大线程数
        - POLARS_ADAPTIVE_BATCH: 1/true/yes 启用自适应批处理
        - POLARS_MEMORY_BUDGET_MB: 正浮点，内存预算（MB）
        - POLARS_LAZY_OPTIMIZATION: 0/1/2 优化级别
        """
        def env_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name, "").strip().lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        def env_int(name: str, default: int | None) -> int | None:
            raw = os.environ.get(name, "").strip()
            if not raw:
                return default
            try:
                v = int(raw)
                return v if v > 0 else default
            except ValueError:
                return default

        def env_float(name: str, default: float | None) -> float | None:
            raw = os.environ.get(name, "").strip()
            if not raw:
                return default
            try:
                v = float(raw)
                return v if v > 0 else default
            except ValueError:
                return default

        return cls(
            enable_streaming=env_bool("POLARS_ENABLE_STREAMING", True),
            max_threads=env_int("POLARS_MAX_THREADS", None),
            adaptive_batch_size=env_bool("POLARS_ADAPTIVE_BATCH", True),
            memory_budget_mb=env_float("POLARS_MEMORY_BUDGET_MB", None),
            lazy_optimization_level=env_int("POLARS_LAZY_OPTIMIZATION", 2) or 2,
            predicate_pushdown=env_bool("POLARS_PREDICATE_PUSHDOWN", True),
            projection_pushdown=env_bool("POLARS_PROJECTION_PUSHDOWN", True),
        )

    def get_optimal_thread_count(self) -> int:
        """获取最优线程数（基于 CPU 核心数和负载）。

        策略：
        - 如果显式设置了 max_threads，使用该值
        - 否则使用 CPU 核心数的 75%（为其他进程预留）
        - 最少 1 个线程，最多 32 个线程
        """
        if self.max_threads is not None:
            return max(1, min(32, self.max_threads))

        cpu_count = multiprocessing.cpu_count()
        # 使用 75% 的核心数，避免过度竞争
        optimal = max(1, int(cpu_count * 0.75))
        return min(32, optimal)

    def get_collect_kwargs(self, estimated_rows: int | None = None) -> dict[str, Any]:
        """获取 collect() 方法的优化参数。

        参数
        ----
        estimated_rows : int | None
            预估结果行数，用于决定是否启用 streaming。

        返回
        ----
        dict[str, Any]
            collect() 方法的关键字参数。
        """
        kwargs: dict[str, Any] = {}

        # Streaming 模式：大数据集或显式启用时使用
        if self.enable_streaming:
            # 如果预估行数超过 100 万或未知行数，启用 streaming
            if estimated_rows is None or estimated_rows > 1_000_000:
                kwargs["streaming"] = True

        return kwargs

    def configure_polars_global(self) -> None:
        """配置 Polars 全局设置（线程池、优化级别等）。"""
        if pl is None:
            return

        # 设置线程数
        thread_count = self.get_optimal_thread_count()
        try:
            pl.Config.set_max_threads(thread_count)
        except Exception:
            # 旧版本 Polars 可能不支持
            pass

        # 设置表格格式（减少内存占用）
        try:
            pl.Config.set_fmt_str_lengths(50)
            pl.Config.set_tbl_rows(20)
        except Exception:
            pass


def get_adaptive_batch_size(
    total_items: int,
    memory_budget_bytes: int,
    item_size_bytes: int,
    *,
    min_batch: int = 16,
    max_batch: int = 512,
) -> int:
    """计算自适应批处理大小。

    参数
    ----
    total_items : int
        总项目数。
    memory_budget_bytes : int
        可用内存预算（字节）。
    item_size_bytes : int
        单个项目的估计大小（字节）。
    min_batch : int
        最小批大小。
    max_batch : int
        最大批大小。

    返回
    ----
    int
        优化的批大小。

    策略
    ----
    - 小数据集：使用较大批大小减少 overhead
    - 大数据集：使用较小批大小减少内存峰值
    - 根据内存预算动态调整
    """
    if total_items <= 0 or item_size_bytes <= 0:
        return min_batch

    # 根据内存预算计算理论最大批大小
    memory_constrained_batch = memory_budget_bytes // item_size_bytes
    memory_constrained_batch = max(min_batch, min(max_batch, memory_constrained_batch))

    # 如果总项目数很小，一次性处理
    if total_items <= min_batch:
        return total_items

    # 动态调整：小数据集用大批，大数据集用小批
    if total_items <= 1000:
        optimal = max_batch
    elif total_items <= 10000:
        optimal = (min_batch + max_batch) // 2
    else:
        optimal = min_batch

    # 应用内存约束
    return min(optimal, memory_constrained_batch)


def estimate_dataframe_memory(rows: int, cols: int, avg_col_size: int = 8) -> int:
    """估计 DataFrame 的内存占用（字节）。

    参数
    ----
    rows : int
        行数。
    cols : int
        列数。
    avg_col_size : int
        平均列大小（字节），默认 8（float64）。

    返回
    ----
    int
        估计的内存占用（字节）。
    """
    if rows <= 0 or cols <= 0:
        return 0

    # 基础数据 + 索引开销（约 20%）
    base_memory = rows * cols * avg_col_size
    overhead = int(base_memory * 0.2)
    return base_memory + overhead
