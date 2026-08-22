#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polars 性能优化使用示例。

展示如何使用新的性能优化功能。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def example_1_basic_configuration():
    """示例 1: 基本配置使用。"""
    print("=" * 60)
    print("示例 1: 基本配置")
    print("=" * 60)

    from backend.polars_performance_config import PolarsPerformanceConfig

    # 从环境变量加载配置
    config = PolarsPerformanceConfig.from_env()
    print(f"Streaming 启用: {config.enable_streaming}")
    print(f"最优线程数: {config.get_optimal_thread_count()}")
    print(f"自适应批处理: {config.adaptive_batch_size}")

    # 获取 collect 参数（大数据集）
    collect_kwargs = config.get_collect_kwargs(estimated_rows=2_000_000)
    print(f"\n大数据集 collect 参数: {collect_kwargs}")

    # 获取 collect 参数（小数据集）
    collect_kwargs = config.get_collect_kwargs(estimated_rows=10_000)
    print(f"小数据集 collect 参数: {collect_kwargs}")


def example_2_custom_configuration():
    """示例 2: 自定义配置。"""
    print("\n" + "=" * 60)
    print("示例 2: 自定义配置")
    print("=" * 60)

    from backend.polars_performance_config import PolarsPerformanceConfig

    # 显式创建配置
    config = PolarsPerformanceConfig(
        enable_streaming=True,
        max_threads=8,
        adaptive_batch_size=True,
        memory_budget_mb=1024.0,
        lazy_optimization_level=2,
    )

    print(f"配置: {config}")
    print(f"线程数: {config.get_optimal_thread_count()}")

    # 配置 Polars 全局设置
    config.configure_polars_global()
    print("✓ Polars 全局设置已配置")


def example_3_memory_estimation():
    """示例 3: 内存估算。"""
    print("\n" + "=" * 60)
    print("示例 3: 内存估算")
    print("=" * 60)

    from backend.polars_memory_optimizer import (
        estimate_polars_dataframe_memory,
        should_use_streaming,
    )

    # 估算 DataFrame 内存
    estimate = estimate_polars_dataframe_memory(
        rows=1_000_000, cols=50, dtype_size=8, string_cols=5, string_avg_len=20
    )

    print(f"数据行数: {estimate.rows:,}")
    print(f"数据列数: {estimate.cols}")
    print(f"估算内存: {estimate.total_mb:.2f} MB")
    print(f"  - 数据: {estimate.estimated_bytes / (1024**2):.2f} MB")
    print(f"  - 开销: {estimate.overhead_bytes / (1024**2):.2f} MB")

    # 判断是否使用 streaming
    available_memory = 4 * 1024**3  # 4GB
    use_streaming = should_use_streaming(1_000_000, 50, available_memory)
    print(f"\n是否使用 streaming: {use_streaming}")


def example_4_adaptive_batch_size():
    """示例 4: 自适应批处理。"""
    print("\n" + "=" * 60)
    print("示例 4: 自适应批处理")
    print("=" * 60)

    from backend.polars_memory_optimizer import calculate_optimal_chunk_size

    available_memory = 2 * 1024**3  # 2GB

    # 不同规模数据集的最优分块
    for total_rows in [10_000, 100_000, 1_000_000, 10_000_000]:
        chunk_size = calculate_optimal_chunk_size(
            total_rows=total_rows,
            total_cols=50,
            available_memory_bytes=available_memory,
            safety_factor=0.7,
        )
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        print(f"{total_rows:>10,} 行 → 分块 {chunk_size:>7,} 行/块 ({num_chunks} 块)")


def example_5_memory_monitoring():
    """示例 5: 内存监控。"""
    print("\n" + "=" * 60)
    print("示例 5: 内存监控")
    print("=" * 60)

    from backend.polars_memory_optimizer import PolarsMemoryMonitor

    monitor = PolarsMemoryMonitor(budget_bytes=4 * 1024**3)  # 4GB

    # 模拟多次 collect 操作
    operations = [
        (100, False),  # 100MB, eager
        (250, True),  # 250MB, streaming
        (150, False),  # 150MB, eager
        (500, True),  # 500MB, streaming
    ]

    for size_mb, streaming in operations:
        size_bytes = size_mb * 1024**2
        monitor.record_collect(result_bytes=size_bytes, used_streaming=streaming)

    # 获取统计信息
    stats = monitor.get_stats()
    print(f"总操作数: {stats['collect_count']}")
    print(f"Streaming 操作数: {stats['streaming_count']}")
    print(f"Streaming 使用率: {stats['streaming_ratio']:.1%}")
    print(f"峰值内存: {stats['peak_usage_mb']:.2f} MB")
    print(f"当前内存: {stats['current_usage_mb']:.2f} MB")
    print(f"内存预算: {stats['budget_mb']:.2f} MB")

    # 判断是否需要启用 streaming
    next_op_size = 800 * 1024**2  # 800MB
    should_stream = monitor.should_enable_streaming(next_op_size)
    print(f"\n下一个操作 (800MB) 是否需要 streaming: {should_stream}")


def example_6_environment_variables():
    """示例 6: 环境变量配置。"""
    print("\n" + "=" * 60)
    print("示例 6: 环境变量配置")
    print("=" * 60)

    print("设置环境变量以配置 Polars 性能:")
    print()
    print("# Streaming 模式")
    print("export POLARS_ENABLE_STREAMING=1")
    print("export POLARS_MEMORY_BUDGET_MB=512")
    print()
    print("# 线程池调优")
    print("export POLARS_MAX_THREADS=8")
    print()
    print("# Lazy 优化")
    print("export POLARS_LAZY_OPTIMIZATION=2")
    print("export POLARS_PREDICATE_PUSHDOWN=1")
    print("export POLARS_PROJECTION_PUSHDOWN=1")
    print()
    print("# 自适应批处理")
    print("export POLARS_ADAPTIVE_BATCH=1")

    # 显示当前环境变量
    print("\n当前环境变量:")
    for key in [
        "POLARS_ENABLE_STREAMING",
        "POLARS_MAX_THREADS",
        "POLARS_MEMORY_BUDGET_MB",
        "POLARS_LAZY_OPTIMIZATION",
    ]:
        value = os.environ.get(key, "(未设置)")
        print(f"  {key}: {value}")


def main():
    """运行所有示例。"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Polars 性能优化使用示例" + " " * 24 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    example_1_basic_configuration()
    example_2_custom_configuration()
    example_3_memory_estimation()
    example_4_adaptive_batch_size()
    example_5_memory_monitoring()
    example_6_environment_variables()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
