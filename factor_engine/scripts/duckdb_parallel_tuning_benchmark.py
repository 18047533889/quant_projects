#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DuckDB 并行执行配置调优 Benchmark。

测试不同的 threads/memory/batch_size 配置，生成最优配置建议。

测试场景：
- B01: 小规模查询（100 factors × 300 stocks × 252 days）
- B02: 中规模查询（300 factors × 1000 stocks × 504 days）
- B03: 大规模查询（1000 factors × 1500 stocks × 504 days）
- B04: 聚合密集型（100 aggregations × 1000 stocks × 504 days）
- B05: JOIN 密集型（多表 JOIN × 500 stocks × 252 days）
- B06: 窗口函数密集型（rolling windows × 1000 stocks × 504 days）
- B07: 分组密集型（GROUP BY × 1000 stocks × 504 days）
- B08: UNION ALL 批量（500 queries × 300 stocks × 252 days）
- B09: Arrow 零拷贝路径（large result set × 1000 stocks × 504 days）

每个场景测试不同的 threads 配置：1/4/8/16/32
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# 添加项目根目录到 sys.path
_FE_ROOT = Path(__file__).resolve().parents[1]
if str(_FE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FE_ROOT))

import duckdb
import pandas as pd
import psutil


@dataclass
class BenchmarkConfig:
    """Benchmark 配置。"""
    threads: int
    memory_limit_mb: int
    enable_object_cache: bool = True
    batch_size: int = 1000


@dataclass
class BenchmarkResult:
    """Benchmark 结果。"""
    scenario: str
    config: BenchmarkConfig
    execution_time_ms: float
    memory_peak_mb: float
    rows_processed: int
    throughput_rows_per_sec: float
    cpu_utilization_pct: float


def create_synthetic_data(n_stocks: int, n_days: int, n_factors: int = 1) -> pd.DataFrame:
    """创建合成测试数据（模拟因子数据）。"""
    dates = pd.date_range("2021-01-04", periods=n_days, freq="D")
    stocks = [f"s{i:04d}" for i in range(n_stocks)]

    # 创建 MultiIndex
    idx = pd.MultiIndex.from_product(
        [dates, stocks],
        names=["timestamp", "instrument"]
    )

    import numpy as np

    data = {
        "close": 100.0 + (np.random.randn(len(idx)) * 10),
        "volume": 1_000_000 + (np.random.randn(len(idx)) * 100_000),
        "open": 100.0 + (np.random.randn(len(idx)) * 10),
        "high": 110.0 + (np.random.randn(len(idx)) * 10),
        "low": 90.0 + (np.random.randn(len(idx)) * 10),
    }

    # 添加因子列
    for i in range(n_factors):
        data[f"factor_{i}"] = np.random.randn(len(idx))

    df = pd.DataFrame(data, index=idx).reset_index()
    return df


def apply_duckdb_config(conn: duckdb.DuckDBPyConnection, config: BenchmarkConfig) -> None:
    """应用 DuckDB 配置。"""
    conn.execute(f"SET threads TO {config.threads}")
    conn.execute(f"SET memory_limit = '{config.memory_limit_mb}MB'")
    if config.enable_object_cache:
        conn.execute("SET enable_object_cache TO true")
    else:
        conn.execute("SET enable_object_cache TO false")


def benchmark_b01_small_query(config: BenchmarkConfig) -> BenchmarkResult:
    """B01: 小规模查询（100 factors × 300 stocks × 252 days）。"""
    df = create_synthetic_data(300, 252, n_factors=100)
    conn = duckdb.connect(":memory:")
    apply_duckdb_config(conn, config)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()

    # 注册 DataFrame
    conn.register("data", df)

    # 简单聚合查询
    result = conn.execute("""
        SELECT instrument,
               AVG(close) as avg_close,
               AVG(volume) as avg_volume
        FROM data
        GROUP BY instrument
    """).fetchdf()

    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    conn.close()

    return BenchmarkResult(
        scenario="B01-small-query",
        config=config,
        execution_time_ms=elapsed_ms,
        memory_peak_mb=mem_after - mem_before,
        rows_processed=len(df),
        throughput_rows_per_sec=len(df) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        cpu_utilization_pct=psutil.cpu_percent(interval=0.1),
    )


def benchmark_b02_medium_query(config: BenchmarkConfig) -> BenchmarkResult:
    """B02: 中规模查询（300 factors × 1000 stocks × 504 days）。"""
    df = create_synthetic_data(1000, 504, n_factors=10)
    conn = duckdb.connect(":memory:")
    apply_duckdb_config(conn, config)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()

    conn.register("data", df)

    # 时间序列聚合
    result = conn.execute("""
        SELECT timestamp, instrument,
               AVG(close) OVER (PARTITION BY instrument ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20,
               STDDEV(close) OVER (PARTITION BY instrument ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as std20
        FROM data
    """).fetchdf()

    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    conn.close()

    return BenchmarkResult(
        scenario="B02-medium-query",
        config=config,
        execution_time_ms=elapsed_ms,
        memory_peak_mb=mem_after - mem_before,
        rows_processed=len(df),
        throughput_rows_per_sec=len(df) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        cpu_utilization_pct=psutil.cpu_percent(interval=0.1),
    )


def benchmark_b03_large_query(config: BenchmarkConfig) -> BenchmarkResult:
    """B03: 大规模查询（1000 factors × 1500 stocks × 504 days）。"""
    df = create_synthetic_data(1500, 504, n_factors=20)
    conn = duckdb.connect(":memory:")
    apply_duckdb_config(conn, config)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()

    conn.register("data", df)

    # 复杂聚合 + 窗口函数
    result = conn.execute("""
        SELECT timestamp, instrument,
               AVG(close) as avg_close,
               SUM(volume) as total_volume,
               AVG(close) OVER (PARTITION BY instrument ORDER BY timestamp ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as ma50,
               RANK() OVER (PARTITION BY timestamp ORDER BY close DESC) as price_rank
        FROM data
        GROUP BY timestamp, instrument, close, volume
    """).fetchdf()

    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    conn.close()

    return BenchmarkResult(
        scenario="B03-large-query",
        config=config,
        execution_time_ms=elapsed_ms,
        memory_peak_mb=mem_after - mem_before,
        rows_processed=len(df),
        throughput_rows_per_sec=len(df) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        cpu_utilization_pct=psutil.cpu_percent(interval=0.1),
    )


def benchmark_b04_aggregation_intensive(config: BenchmarkConfig) -> BenchmarkResult:
    """B04: 聚合密集型（100 aggregations × 1000 stocks × 504 days）。"""
    df = create_synthetic_data(1000, 504, n_factors=10)
    conn = duckdb.connect(":memory:")
    apply_duckdb_config(conn, config)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()

    conn.register("data", df)

    # 多重聚合
    result = conn.execute("""
        SELECT instrument,
               COUNT(*) as cnt,
               AVG(close) as avg_close,
               STDDEV(close) as std_close,
               MIN(close) as min_close,
               MAX(close) as max_close,
               SUM(volume) as total_volume,
               AVG(volume) as avg_volume,
               STDDEV(volume) as std_volume,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY close) as median_close
        FROM data
        GROUP BY instrument
    """).fetchdf()

    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    conn.close()

    return BenchmarkResult(
        scenario="B04-aggregation-intensive",
        config=config,
        execution_time_ms=elapsed_ms,
        memory_peak_mb=mem_after - mem_before,
        rows_processed=len(df),
        throughput_rows_per_sec=len(df) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        cpu_utilization_pct=psutil.cpu_percent(interval=0.1),
    )


def benchmark_b05_join_intensive(config: BenchmarkConfig) -> BenchmarkResult:
    """B05: JOIN 密集型（多表 JOIN × 500 stocks × 252 days）。"""
    df1 = create_synthetic_data(500, 252, n_factors=5)
    df2 = create_synthetic_data(500, 252, n_factors=5)

    conn = duckdb.connect(":memory:")
    apply_duckdb_config(conn, config)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()

    conn.register("data1", df1)
    conn.register("data2", df2)

    # 多表 JOIN
    result = conn.execute("""
        SELECT d1.timestamp, d1.instrument,
               d1.close as close1,
               d2.close as close2,
               d1.volume + d2.volume as total_volume
        FROM data1 d1
        INNER JOIN data2 d2
            ON d1.timestamp = d2.timestamp
            AND d1.instrument = d2.instrument
    """).fetchdf()

    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    conn.close()

    return BenchmarkResult(
        scenario="B05-join-intensive",
        config=config,
        execution_time_ms=elapsed_ms,
        memory_peak_mb=mem_after - mem_before,
        rows_processed=len(df1) + len(df2),
        throughput_rows_per_sec=(len(df1) + len(df2)) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        cpu_utilization_pct=psutil.cpu_percent(interval=0.1),
    )


def benchmark_b06_window_intensive(config: BenchmarkConfig) -> BenchmarkResult:
    """B06: 窗口函数密集型（rolling windows × 1000 stocks × 504 days）。"""
    df = create_synthetic_data(1000, 504, n_factors=5)
    conn = duckdb.connect(":memory:")
    apply_duckdb_config(conn, config)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()

    conn.register("data", df)

    # 多重窗口函数
    result = conn.execute("""
        SELECT timestamp, instrument, close,
               AVG(close) OVER w5 as ma5,
               AVG(close) OVER w10 as ma10,
               AVG(close) OVER w20 as ma20,
               AVG(close) OVER w60 as ma60,
               STDDEV(close) OVER w20 as std20
        FROM data
        WINDOW
            w5 AS (PARTITION BY instrument ORDER BY timestamp ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
            w10 AS (PARTITION BY instrument ORDER BY timestamp ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
            w20 AS (PARTITION BY instrument ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
            w60 AS (PARTITION BY instrument ORDER BY timestamp ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
    """).fetchdf()

    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    conn.close()

    return BenchmarkResult(
        scenario="B06-window-intensive",
        config=config,
        execution_time_ms=elapsed_ms,
        memory_peak_mb=mem_after - mem_before,
        rows_processed=len(df),
        throughput_rows_per_sec=len(df) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        cpu_utilization_pct=psutil.cpu_percent(interval=0.1),
    )


def benchmark_b07_groupby_intensive(config: BenchmarkConfig) -> BenchmarkResult:
    """B07: 分组密集型（GROUP BY × 1000 stocks × 504 days）。"""
    df = create_synthetic_data(1000, 504, n_factors=10)
    conn = duckdb.connect(":memory:")
    apply_duckdb_config(conn, config)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()

    conn.register("data", df)

    # 多级分组
    result = conn.execute("""
        SELECT
            EXTRACT(YEAR FROM timestamp) as year,
            EXTRACT(MONTH FROM timestamp) as month,
            instrument,
            COUNT(*) as cnt,
            AVG(close) as avg_close,
            SUM(volume) as total_volume
        FROM data
        GROUP BY year, month, instrument
    """).fetchdf()

    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    conn.close()

    return BenchmarkResult(
        scenario="B07-groupby-intensive",
        config=config,
        execution_time_ms=elapsed_ms,
        memory_peak_mb=mem_after - mem_before,
        rows_processed=len(df),
        throughput_rows_per_sec=len(df) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        cpu_utilization_pct=psutil.cpu_percent(interval=0.1),
    )


def benchmark_b08_union_all_batch(config: BenchmarkConfig) -> BenchmarkResult:
    """B08: UNION ALL 批量（500 queries × 300 stocks × 252 days）。"""
    df = create_synthetic_data(300, 252, n_factors=20)
    conn = duckdb.connect(":memory:")
    apply_duckdb_config(conn, config)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()

    conn.register("data", df)

    # 构造 UNION ALL 查询
    unions = []
    for i in range(50):  # 50 个 UNION ALL（模拟批量查询）
        unions.append(f"""
            SELECT '{i}' as query_id, instrument, AVG(factor_{i % 10}) as avg_factor
            FROM data
            GROUP BY instrument
        """)

    union_query = " UNION ALL ".join(unions)
    result = conn.execute(union_query).fetchdf()

    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    conn.close()

    return BenchmarkResult(
        scenario="B08-union-all-batch",
        config=config,
        execution_time_ms=elapsed_ms,
        memory_peak_mb=mem_after - mem_before,
        rows_processed=len(df) * 50,
        throughput_rows_per_sec=(len(df) * 50) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        cpu_utilization_pct=psutil.cpu_percent(interval=0.1),
    )


def benchmark_b09_arrow_zero_copy(config: BenchmarkConfig) -> BenchmarkResult:
    """B09: Arrow 零拷贝路径（large result set × 1000 stocks × 504 days）。"""
    df = create_synthetic_data(1000, 504, n_factors=30)
    conn = duckdb.connect(":memory:")
    apply_duckdb_config(conn, config)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()

    conn.register("data", df)

    # Arrow 零拷贝路径
    result_arrow = conn.execute("""
        SELECT * FROM data
        WHERE close > 100
    """).arrow()

    elapsed_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    conn.close()

    return BenchmarkResult(
        scenario="B09-arrow-zero-copy",
        config=config,
        execution_time_ms=elapsed_ms,
        memory_peak_mb=mem_after - mem_before,
        rows_processed=len(df),
        throughput_rows_per_sec=len(df) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        cpu_utilization_pct=psutil.cpu_percent(interval=0.1),
    )


def run_benchmark_suite(thread_configs: list[int]) -> dict[str, list[BenchmarkResult]]:
    """运行完整的 benchmark suite。"""
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    total_memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))

    results: dict[str, list[BenchmarkResult]] = {}

    benchmarks = [
        ("B01", benchmark_b01_small_query),
        ("B02", benchmark_b02_medium_query),
        ("B03", benchmark_b03_large_query),
        ("B04", benchmark_b04_aggregation_intensive),
        ("B05", benchmark_b05_join_intensive),
        ("B06", benchmark_b06_window_intensive),
        ("B07", benchmark_b07_groupby_intensive),
        ("B08", benchmark_b08_union_all_batch),
        ("B09", benchmark_b09_arrow_zero_copy),
    ]

    for scenario_id, benchmark_fn in benchmarks:
        print(f"\n{'='*60}")
        print(f"Running {scenario_id}: {benchmark_fn.__doc__.split('.')[0].strip()}")
        print(f"{'='*60}")

        scenario_results = []

        for threads in thread_configs:
            # 动态调整内存限制（避免超出系统内存）
            memory_limit_mb = min(
                int(total_memory_mb * 0.5),
                max(2048, int(total_memory_mb * threads / cpu_count * 0.4))
            )

            config = BenchmarkConfig(
                threads=threads,
                memory_limit_mb=memory_limit_mb,
                enable_object_cache=True,
            )

            print(f"\n  Testing threads={threads}, memory={memory_limit_mb}MB")

            try:
                result = benchmark_fn(config)
                scenario_results.append(result)

                print(f"    Execution time: {result.execution_time_ms:.2f} ms")
                print(f"    Throughput: {result.throughput_rows_per_sec:,.0f} rows/sec")
                print(f"    Memory peak: {result.memory_peak_mb:.2f} MB")
                print(f"    CPU utilization: {result.cpu_utilization_pct:.1f}%")

            except Exception as e:
                print(f"    ERROR: {e}")

        results[scenario_id] = scenario_results

    return results


def analyze_results(results: dict[str, list[BenchmarkResult]]) -> dict[str, Any]:
    """分析 benchmark 结果，生成最优配置建议。"""
    import multiprocessing

    analysis = {
        "overall_best_config": None,
        "scenario_best_configs": {},
        "recommendations": {},
        "detailed_results": {},
    }

    # 计算每个场景的最佳配置
    for scenario, scenario_results in results.items():
        if not scenario_results:
            continue

        # 按执行时间排序
        best = min(scenario_results, key=lambda r: r.execution_time_ms)

        analysis["scenario_best_configs"][scenario] = {
            "threads": best.config.threads,
            "memory_limit_mb": best.config.memory_limit_mb,
            "execution_time_ms": best.execution_time_ms,
            "throughput_rows_per_sec": best.throughput_rows_per_sec,
        }

        # 详细结果
        analysis["detailed_results"][scenario] = [
            {
                "threads": r.config.threads,
                "memory_limit_mb": r.config.memory_limit_mb,
                "execution_time_ms": r.execution_time_ms,
                "throughput_rows_per_sec": r.throughput_rows_per_sec,
                "memory_peak_mb": r.memory_peak_mb,
                "cpu_utilization_pct": r.cpu_utilization_pct,
            }
            for r in scenario_results
        ]

    # 计算总体最佳配置（加权平均）
    thread_scores = {}
    for scenario, scenario_results in results.items():
        for result in scenario_results:
            threads = result.config.threads
            if threads not in thread_scores:
                thread_scores[threads] = []

            # 归一化分数（越低越好）
            min_time = min(r.execution_time_ms for r in scenario_results)
            normalized_score = result.execution_time_ms / min_time if min_time > 0 else 1.0
            thread_scores[threads].append(normalized_score)

    # 计算每个线程配置的平均分数
    avg_scores = {
        threads: sum(scores) / len(scores)
        for threads, scores in thread_scores.items()
    }

    if not avg_scores:
        # 没有成功的测试结果，使用默认配置
        best_threads = min(8, multiprocessing.cpu_count())
        avg_normalized_score = 1.0
    else:
        best_threads = min(avg_scores, key=avg_scores.get)
        avg_normalized_score = avg_scores[best_threads]

    # 根据系统资源推荐配置
    cpu_count = multiprocessing.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)

    analysis["overall_best_config"] = {
        "threads": best_threads,
        "memory_limit_mb": int(total_memory_gb * 0.5 * 1024),
        "enable_object_cache": True,
        "avg_normalized_score": avg_normalized_score,
    }

    # 生成配置建议
    analysis["recommendations"] = {
        "default": {
            "threads": best_threads,
            "memory_limit_mb": int(total_memory_gb * 0.5 * 1024),
            "description": f"通用默认配置（最佳平均性能）",
        },
        "small_workload": {
            "threads": min(4, cpu_count),
            "memory_limit_mb": 2048,
            "description": "小规模工作负载（< 100K rows）",
        },
        "medium_workload": {
            "threads": min(8, cpu_count),
            "memory_limit_mb": int(total_memory_gb * 0.3 * 1024),
            "description": "中等规模工作负载（100K - 1M rows）",
        },
        "large_workload": {
            "threads": cpu_count,
            "memory_limit_mb": int(total_memory_gb * 0.6 * 1024),
            "description": "大规模工作负载（> 1M rows）",
        },
        "memory_constrained": {
            "threads": min(4, cpu_count),
            "memory_limit_mb": 1024,
            "description": "内存受限环境",
        },
    }

    return analysis


def generate_report(results: dict[str, list[BenchmarkResult]], analysis: dict[str, Any], output_path: Path) -> None:
    """生成调优报告。"""
    report_lines = [
        "# DuckDB 并行执行配置调优报告",
        "",
        f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 系统信息",
        "",
        f"- **CPU 核心数**: {psutil.cpu_count()}",
        f"- **总内存**: {psutil.virtual_memory().total / (1024**3):.2f} GB",
        f"- **DuckDB 版本**: {duckdb.__version__}",
        "",
        "## 测试配置",
        "",
        "测试了以下线程配置: 1, 4, 8, 16, 32",
        "",
        "## 总体最佳配置",
        "",
        f"- **线程数**: {analysis['overall_best_config']['threads']}",
        f"- **内存限制**: {analysis['overall_best_config']['memory_limit_mb']} MB",
        f"- **启用对象缓存**: {analysis['overall_best_config']['enable_object_cache']}",
        f"- **平均归一化分数**: {analysis['overall_best_config']['avg_normalized_score']:.3f}",
        "",
        "## 不同负载下的配置建议",
        "",
    ]

    for workload_type, config in analysis["recommendations"].items():
        report_lines.extend([
            f"### {workload_type.replace('_', ' ').title()}",
            "",
            f"- **描述**: {config['description']}",
            f"- **线程数**: {config['threads']}",
            f"- **内存限制**: {config['memory_limit_mb']} MB",
            "",
        ])

    report_lines.extend([
        "## 各场景最佳配置",
        "",
    ])

    for scenario, config in analysis["scenario_best_configs"].items():
        report_lines.extend([
            f"### {scenario}",
            "",
            f"- **线程数**: {config['threads']}",
            f"- **内存限制**: {config['memory_limit_mb']} MB",
            f"- **执行时间**: {config['execution_time_ms']:.2f} ms",
            f"- **吞吐量**: {config['throughput_rows_per_sec']:,.0f} rows/sec",
            "",
        ])

    report_lines.extend([
        "## 详细测试结果",
        "",
    ])

    for scenario, scenario_results in analysis["detailed_results"].items():
        report_lines.extend([
            f"### {scenario}",
            "",
            "| Threads | Memory (MB) | Execution Time (ms) | Throughput (rows/sec) | Memory Peak (MB) | CPU % |",
            "|---------|-------------|---------------------|----------------------|------------------|-------|",
        ])

        for result in scenario_results:
            report_lines.append(
                f"| {result['threads']:7d} | {result['memory_limit_mb']:11d} | "
                f"{result['execution_time_ms']:19.2f} | {result['throughput_rows_per_sec']:20,.0f} | "
                f"{result['memory_peak_mb']:16.2f} | {result['cpu_utilization_pct']:5.1f} |"
            )

        report_lines.append("")

    report_lines.extend([
        "## Python 配置代码",
        "",
        "```python",
        "# 推荐的默认配置",
        "from backend.sql_pushdown.duckdb_performance import DuckDBParallelConfig",
        "",
        f"config = DuckDBParallelConfig(",
        f"    threads={analysis['overall_best_config']['threads']},",
        f"    memory_limit_mb={analysis['overall_best_config']['memory_limit_mb']},",
        f"    enable_object_cache={analysis['overall_best_config']['enable_object_cache']},",
        f")",
        "",
        "# 应用到连接",
        "import duckdb",
        "conn = duckdb.connect(':memory:')",
        "config.apply_to_connection(conn)",
        "```",
        "",
        "## 环境变量配置",
        "",
        "```bash",
        f"export DUCKDB_THREADS={analysis['overall_best_config']['threads']}",
        f"export DUCKDB_MEMORY_LIMIT_MB={analysis['overall_best_config']['memory_limit_mb']}",
        "export DUCKDB_OBJECT_CACHE=true",
        "```",
        "",
    ])

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告已生成: {output_path}")


def main() -> int:
    """主函数。"""
    print("="*80)
    print("DuckDB 并行执行配置调优 Benchmark")
    print("="*80)

    # 测试线程配置
    thread_configs = [1, 4, 8, 16, 32]

    # 运行 benchmark
    results = run_benchmark_suite(thread_configs)

    # 分析结果
    analysis = analyze_results(results)

    # 生成报告
    output_path = Path("/tmp/duckdb_parallel_tuning_report.md")
    generate_report(results, analysis, output_path)

    # 保存 JSON 格式的详细结果
    json_path = Path("/tmp/duckdb_parallel_tuning_results.json")
    json_data = {
        "analysis": analysis,
        "raw_results": {
            scenario: [asdict(r) for r in scenario_results]
            for scenario, scenario_results in results.items()
        },
    }

    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    print(f"详细结果已保存: {json_path}")

    print("\n" + "="*80)
    print("Benchmark 完成！")
    print("="*80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
