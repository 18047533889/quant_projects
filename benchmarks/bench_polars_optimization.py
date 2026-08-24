#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polars 优化效果验证基准测试。

对比优化前后的内存峰值、吞吐、延迟表现。

运行方式:
    python benchmarks/bench_polars_optimization.py --mode baseline
    python benchmarks/bench_polars_optimization.py --mode optimized
    python benchmarks/bench_polars_optimization.py --mode compare
"""
import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class BenchmarkResult:
    """单次基准测试结果。"""
    mode: str  # baseline / optimized
    factor_count: int
    row_count: int
    column_count: int
    elapsed_ms: float
    peak_memory_mb: float
    throughput_qps: float  # factors per second
    streaming_enabled: bool
    thread_count: int
    error: str | None = None


def get_peak_memory_mb() -> float:
    """获取当前进程峰值内存（MB）。"""
    if HAS_PSUTIL:
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024
    return 0.0


def generate_synthetic_data(
    n_dates: int = 252,
    n_instruments: int = 100,
    n_columns: int = 10,
) -> tuple[Any, dict[str, Any]]:
    """生成合成测试数据。

    Returns:
        (data_source, ctx): 可用于 backend.execute() 的上下文。
    """
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.storage.in_memory_data_source import InMemoryDataSource

    # 生成时间序列
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    instruments = [f"INST_{i:04d}" for i in range(n_instruments)]

    # 生成随机面板数据
    data = {}
    for col_idx in range(n_columns):
        col_name = f"col_{col_idx}"
        values = np.random.randn(n_dates * n_instruments)
        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=["date", "instrument"]
        )
        data[col_name] = pd.Series(values, index=index)

    ds = InMemoryDataSource(data)
    ctx = ExecutionContext(
        data_source=ds,
        universe=instruments,
        start_date=dates[0],
        end_date=dates[-1],
    )

    return ds, ctx


def build_test_factors(n_factors: int = 50) -> list[str]:
    """构建测试因子表达式。

    Returns:
        因子表达式列表（DSL 字符串）。
    """
    # 混合不同算子类型
    templates = [
        "ts_mean(col_0, 20)",
        "ts_std(col_1, 20)",
        "rank(ts_mean(col_2, 10))",
        "zscore(col_3)",
        "ts_corr(col_4, col_5, 20)",
        "add(col_6, col_7)",
        "multiply(ts_mean(col_0, 5), 2.0)",
        "ts_delta(col_1, 1)",
        "ffill(col_2)",
        "clip(col_3, -3.0, 3.0)",
    ]

    factors = []
    for i in range(n_factors):
        template = templates[i % len(templates)]
        # 轮换列引用
        col_idx = i % 10
        factors.append(template.replace("col_0", f"col_{col_idx}"))

    return factors


def run_baseline_benchmark(
    factors: list[str],
    ctx: Any,
) -> BenchmarkResult:
    """运行基线基准测试（无优化）。"""
    from factor_engine.backend.polars_backend import PolarsBackend
    from factor_engine.planner.dsl_parser import parse_factor

    backend = PolarsBackend()
    gc.collect()

    start_mem = get_peak_memory_mb()
    start_time = time.perf_counter()

    results = []
    for expr in factors:
        plan = parse_factor(expr)
        result = backend.execute(plan, ctx)
        results.append(result)

    elapsed = (time.perf_counter() - start_time) * 1000  # ms
    peak_mem = get_peak_memory_mb() - start_mem

    return BenchmarkResult(
        mode="baseline",
        factor_count=len(factors),
        row_count=len(results[0]) if results else 0,
        column_count=10,
        elapsed_ms=elapsed,
        peak_memory_mb=peak_mem,
        throughput_qps=len(factors) / (elapsed / 1000),
        streaming_enabled=False,
        thread_count=1,
    )


def run_optimized_benchmark(
    factors: list[str],
    ctx: Any,
) -> BenchmarkResult:
    """运行优化后基准测试（streaming + 线程调优）。"""
    from factor_engine.backend.polars_backend import PolarsBackend
    from factor_engine.backend.polars_streaming_policy import should_use_streaming
    from factor_engine.backend.polars_thread_config import (
        configure_polars_for_execution,
        get_physical_cores,
    )
    from factor_engine.planner.dsl_parser import parse_factor

    # 应用优化配置
    config = configure_polars_for_execution(
        max_workers=get_physical_cores(),
        memory_mb=4096,
    )

    backend = PolarsBackend()
    gc.collect()

    start_mem = get_peak_memory_mb()
    start_time = time.perf_counter()

    results = []
    streaming_count = 0

    for expr in factors:
        plan = parse_factor(expr)

        # 检查是否可流式
        can_stream, _ = should_use_streaming(plan)
        if can_stream:
            streaming_count += 1

        # 执行（优化路径会自动应用 streaming）
        result = backend.execute(plan, ctx)
        results.append(result)

    elapsed = (time.perf_counter() - start_time) * 1000  # ms
    peak_mem = get_peak_memory_mb() - start_mem

    return BenchmarkResult(
        mode="optimized",
        factor_count=len(factors),
        row_count=len(results[0]) if results else 0,
        column_count=10,
        elapsed_ms=elapsed,
        peak_memory_mb=peak_mem,
        throughput_qps=len(factors) / (elapsed / 1000),
        streaming_enabled=streaming_count > 0,
        thread_count=config.get("thread_count", 1),
    )


def compare_results(baseline: BenchmarkResult, optimized: BenchmarkResult) -> dict[str, Any]:
    """对比基线与优化结果。"""
    memory_reduction = (
        (baseline.peak_memory_mb - optimized.peak_memory_mb)
        / baseline.peak_memory_mb * 100
    )
    throughput_improvement = (
        (optimized.throughput_qps - baseline.throughput_qps)
        / baseline.throughput_qps * 100
    )
    speedup = baseline.elapsed_ms / optimized.elapsed_ms

    return {
        "baseline": asdict(baseline),
        "optimized": asdict(optimized),
        "improvements": {
            "memory_reduction_pct": round(memory_reduction, 2),
            "throughput_improvement_pct": round(throughput_improvement, 2),
            "speedup": round(speedup, 2),
        },
        "goals": {
            "memory_reduction_target": 50.0,  # 50-70%
            "throughput_improvement_target": 30.0,  # 30-50%
            "memory_achieved": memory_reduction >= 50.0,
            "throughput_achieved": throughput_improvement >= 30.0,
        }
    }


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """保存结果到 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def print_comparison(comparison: dict[str, Any]) -> None:
    """打印对比报告。"""
    print("\n" + "=" * 80)
    print("POLARS OPTIMIZATION BENCHMARK RESULTS")
    print("=" * 80)

    baseline = comparison["baseline"]
    optimized = comparison["optimized"]
    improvements = comparison["improvements"]
    goals = comparison["goals"]

    print("\n📊 Baseline Performance:")
    print(f"  Elapsed:     {baseline['elapsed_ms']:.1f} ms")
    print(f"  Memory Peak: {baseline['peak_memory_mb']:.1f} MB")
    print(f"  Throughput:  {baseline['throughput_qps']:.2f} factors/sec")

    print("\n🚀 Optimized Performance:")
    print(f"  Elapsed:     {optimized['elapsed_ms']:.1f} ms")
    print(f"  Memory Peak: {optimized['peak_memory_mb']:.1f} MB")
    print(f"  Throughput:  {optimized['throughput_qps']:.2f} factors/sec")
    print(f"  Streaming:   {optimized['streaming_enabled']}")
    print(f"  Threads:     {optimized['thread_count']}")

    print("\n📈 Improvements:")
    print(f"  Memory Reduction:      {improvements['memory_reduction_pct']:+.1f}%")
    print(f"  Throughput Improvement: {improvements['throughput_improvement_pct']:+.1f}%")
    print(f"  Speedup:               {improvements['speedup']:.2f}x")

    print("\n🎯 Goal Achievement:")
    mem_status = "✅" if goals["memory_achieved"] else "❌"
    thr_status = "✅" if goals["throughput_achieved"] else "❌"
    print(f"  {mem_status} Memory Reduction:      {improvements['memory_reduction_pct']:.1f}% (target: {goals['memory_reduction_target']:.1f}%)")
    print(f"  {thr_status} Throughput Improvement: {improvements['throughput_improvement_pct']:.1f}% (target: {goals['throughput_improvement_target']:.1f}%)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Polars optimization benchmark")
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized", "compare"],
        default="compare",
        help="Benchmark mode"
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=50,
        help="Number of factors to test"
    )
    parser.add_argument(
        "--dates",
        type=int,
        default=252,
        help="Number of dates"
    )
    parser.add_argument(
        "--instruments",
        type=int,
        default=100,
        help="Number of instruments"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/polars_optimization.json"),
        help="Output JSON path"
    )

    args = parser.parse_args()

    # 生成测试数据
    print(f"Generating synthetic data ({args.dates} dates × {args.instruments} instruments)...")
    ds, ctx = generate_synthetic_data(
        n_dates=args.dates,
        n_instruments=args.instruments,
        n_columns=10,
    )

    # 构建测试因子
    print(f"Building {args.factors} test factors...")
    factors = build_test_factors(args.factors)

    if args.mode == "baseline":
        print("\n🏃 Running baseline benchmark...")
        result = run_baseline_benchmark(factors, ctx)
        print(f"\nBaseline: {result.elapsed_ms:.1f} ms, {result.peak_memory_mb:.1f} MB")
        save_results({"baseline": asdict(result)}, args.output)

    elif args.mode == "optimized":
        print("\n🚀 Running optimized benchmark...")
        result = run_optimized_benchmark(factors, ctx)
        print(f"\nOptimized: {result.elapsed_ms:.1f} ms, {result.peak_memory_mb:.1f} MB")
        save_results({"optimized": asdict(result)}, args.output)

    else:  # compare
        print("\n🏃 Running baseline benchmark...")
        baseline = run_baseline_benchmark(factors, ctx)

        print("\n🚀 Running optimized benchmark...")
        optimized = run_optimized_benchmark(factors, ctx)

        comparison = compare_results(baseline, optimized)
        print_comparison(comparison)
        save_results(comparison, args.output)


if __name__ == "__main__":
    main()
