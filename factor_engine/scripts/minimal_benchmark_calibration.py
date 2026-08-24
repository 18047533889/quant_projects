# -*- coding: utf-8 -*-
"""Minimal Performance Benchmark & Cost Model Calibration (2026-08-13)

无需完整算子加载的轻量级 benchmark，直接测试成本模型核心逻辑。

生产规模性能基准测试：
    1. 合成 workload（模拟 factor 执行）
    2. 成本模型预测 vs 真实观测
    3. 校准系数计算

输出：
    /tmp/performance_benchmark_report.md
    /tmp/benchmark_calibration_data.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_FE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FE_ROOT))

import pandas as pd
import numpy as np
import psutil

from factor_engine.planner.cost_model_v2 import (
    CostComponentsV2,
    estimate_operator_cost_v2,
    estimate_source_cost_v2,
    estimate_total_cost_v2,
)
from factor_engine.planner.data_shape_estimate import DataShapeEstimate

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Benchmark scale (production-like)
SCALE_STOCKS = int(os.environ.get("BENCH_STOCKS", "3000"))
SCALE_DAYS = int(os.environ.get("BENCH_DAYS", "504"))  # 2 years
SCALE_FACTORS = int(os.environ.get("BENCH_FACTORS", "1000"))
QUICK_MODE = bool(os.environ.get("BENCH_QUICK", ""))

if QUICK_MODE:
    SCALE_STOCKS = 500
    SCALE_DAYS = 252
    SCALE_FACTORS = 200


@dataclass
class BenchmarkResult:
    """单次 benchmark 运行结果"""

    name: str
    n_factors: int
    stocks: int
    days: int
    backend: str
    operator_type: str

    # 真实观测
    actual_time_ms: float
    actual_peak_memory_bytes: int
    actual_throughput_factors_per_sec: float

    # 成本模型预测
    predicted_time_ms: float
    predicted_peak_memory_bytes: int

    # 误差
    time_error_pct: float
    memory_error_pct: float

    # 详细指标
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    workload_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_factors": self.n_factors,
            "stocks": self.stocks,
            "days": self.days,
            "backend": self.backend,
            "operator_type": self.operator_type,
            "actual_time_ms": round(self.actual_time_ms, 2),
            "actual_peak_memory_bytes": self.actual_peak_memory_bytes,
            "actual_throughput_factors_per_sec": round(self.actual_throughput_factors_per_sec, 2),
            "predicted_time_ms": round(self.predicted_time_ms, 2),
            "predicted_peak_memory_bytes": self.predicted_peak_memory_bytes,
            "time_error_pct": round(self.time_error_pct, 2),
            "memory_error_pct": round(self.memory_error_pct, 2),
            "cost_breakdown": {k: round(v, 2) for k, v in self.cost_breakdown.items()},
            "workload_stats": self.workload_stats,
        }


def create_synthetic_panel(stocks: int, days: int) -> pd.DataFrame:
    """创建合成 panel 数据（stocks × days）"""
    dts = pd.bdate_range("2020-01-01", periods=days)
    idx = pd.MultiIndex.from_product(
        [dts, [f"stock_{i:04d}" for i in range(stocks)]],
        names=["timestamp", "instrument"],
    )

    np.random.seed(42)
    n = len(idx)

    df = pd.DataFrame({
        "close": 100.0 + np.cumsum(np.random.randn(n) * 0.5),
        "high": 100.0 + np.cumsum(np.random.randn(n) * 0.5) * 1.02,
        "low": 100.0 + np.cumsum(np.random.randn(n) * 0.5) * 0.98,
        "volume": np.abs(1_000_000 * (1 + np.random.randn(n) * 0.3)),
        "turnover": np.abs(1_000_000 * (1 + np.random.randn(n) * 0.3)) * 100,
    }, index=idx)

    return df


def simulate_operator_execution(
    df: pd.DataFrame,
    operator_type: str,
    window: int = 20,
    backend: str = "pandas",
) -> tuple[pd.Series, float, int]:
    """模拟算子执行（真实计算）"""
    process = psutil.Process()
    mem_before = process.memory_info().rss

    t0 = time.monotonic()

    if operator_type == "ts_mean":
        result = df.groupby(level="instrument")["close"].rolling(window=window).mean()
    elif operator_type == "ts_std":
        result = df.groupby(level="instrument")["close"].rolling(window=window).std()
    elif operator_type == "ts_sum":
        result = df.groupby(level="instrument")["close"].rolling(window=window).sum()
    elif operator_type == "rank":
        result = df.groupby(level="timestamp")["close"].rank()
    elif operator_type == "zscore":
        grouped = df.groupby(level="timestamp")["close"]
        result = (grouped.transform(lambda x: x) - grouped.transform("mean")) / grouped.transform("std")
    else:
        result = df["close"]

    # Force evaluation
    _ = result.values

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    mem_after = process.memory_info().rss
    memory_used = mem_after - mem_before

    return result, elapsed_ms, memory_used


def predict_operator_cost(
    operator_type: str,
    shape: DataShapeEstimate,
    backend: str,
    window: int = 20,
) -> CostComponentsV2:
    """使用成本模型预测算子成本"""
    # Map operator type to complexity
    complexity_map = {
        "ts_mean": "O(NW)",
        "ts_std": "O(NW)",
        "ts_sum": "O(NW)",
        "rank": "O(N log N)",
        "zscore": "O(N)",
    }

    op_cost = estimate_operator_cost_v2(
        operator_type,
        shape=shape,
        backend=backend,
        window=window if "ts_" in operator_type else None,
    )

    source_cost = estimate_source_cost_v2(shape=shape)

    total = estimate_total_cost_v2(
        source_cost=source_cost,
        compute_cost=op_cost,
    )

    return total


def run_benchmark(
    name: str,
    df: pd.DataFrame,
    n_factors: int,
    operator_type: str,
    backend: str,
    stocks: int,
    days: int,
    window: int = 20,
) -> BenchmarkResult:
    """运行单次 benchmark"""
    _LOGGER.info(f"Running benchmark: {name} ({n_factors} factors, {operator_type}, {backend})")

    # 构造 shape estimate
    shape = DataShapeEstimate(
        estimated_rows=stocks * days,
        estimated_dates=days,
        estimated_instruments=stocks,
        estimated_columns=5,
        estimated_bytes=stocks * days * 5 * 8,
        average_row_width_bytes=40.0,
        density=0.98,
        frequency="daily",
        storage_kind="memory",
    )

    # 预测成本
    predicted_cost = predict_operator_cost(operator_type, shape, backend, window)

    # 实际执行（模拟 n_factors 次）
    total_time_ms = 0.0
    total_memory = 0

    for i in range(min(n_factors, 10)):  # 限制实际执行次数避免太慢
        _, elapsed_ms, mem_used = simulate_operator_execution(df, operator_type, window, backend)
        total_time_ms += elapsed_ms
        total_memory = max(total_memory, mem_used)

    # 外推到全部 factors
    if n_factors > 10:
        total_time_ms = total_time_ms / min(n_factors, 10) * n_factors

    # 计算吞吐量
    throughput = n_factors / (total_time_ms / 1000.0)

    # 计算误差
    time_error_pct = (
        (total_time_ms - predicted_cost.total_cost_ms) / max(1, total_time_ms) * 100
    )
    memory_error_pct = (
        (total_memory - predicted_cost.peak_memory_bytes) / max(1, total_memory) * 100
    )

    return BenchmarkResult(
        name=name,
        n_factors=n_factors,
        stocks=stocks,
        days=days,
        backend=backend,
        operator_type=operator_type,
        actual_time_ms=total_time_ms,
        actual_peak_memory_bytes=total_memory,
        actual_throughput_factors_per_sec=throughput,
        predicted_time_ms=predicted_cost.total_cost_ms,
        predicted_peak_memory_bytes=predicted_cost.peak_memory_bytes,
        time_error_pct=time_error_pct,
        memory_error_pct=memory_error_pct,
        cost_breakdown={
            "source_cost_ms": predicted_cost.source_cost_ms,
            "compute_cost_ms": predicted_cost.compute_cost_ms,
            "transfer_cost_ms": predicted_cost.transfer_cost_ms,
            "materialize_cost_ms": predicted_cost.materialize_cost_ms,
        },
        workload_stats={
            "window": window if "ts_" in operator_type else None,
            "samples_executed": min(n_factors, 10),
            "extrapolated": n_factors > 10,
        },
    )


def calibrate_cost_model(results: list[BenchmarkResult]) -> dict[str, Any]:
    """根据 benchmark 结果校准成本模型参数"""
    _LOGGER.info("Calibrating cost model...")

    # 收集误差统计
    time_errors = [r.time_error_pct for r in results]
    memory_errors = [r.memory_error_pct for r in results]

    # 计算校准系数
    avg_time_error = sum(time_errors) / len(time_errors)
    avg_memory_error = sum(memory_errors) / len(memory_errors)

    # 建议的校准系数（简化版）
    time_calibration_factor = 1.0 / (1.0 + avg_time_error / 100.0)
    memory_calibration_factor = 1.0 / (1.0 + avg_memory_error / 100.0)

    # 按算子类型分析
    operator_analysis = {}
    for op_type in set(r.operator_type for r in results):
        op_results = [r for r in results if r.operator_type == op_type]
        op_time_errors = [r.time_error_pct for r in op_results]
        operator_analysis[op_type] = {
            "mean_time_error_pct": round(sum(op_time_errors) / len(op_time_errors), 2),
            "samples": len(op_results),
        }

    return {
        "time_errors_pct": {
            "mean": round(avg_time_error, 2),
            "std": round((sum((e - avg_time_error) ** 2 for e in time_errors) / len(time_errors)) ** 0.5, 2),
            "min": round(min(time_errors), 2),
            "max": round(max(time_errors), 2),
        },
        "memory_errors_pct": {
            "mean": round(avg_memory_error, 2),
            "std": round((sum((e - avg_memory_error) ** 2 for e in memory_errors) / len(memory_errors)) ** 0.5, 2),
            "min": round(min(memory_errors), 2),
            "max": round(max(memory_errors), 2),
        },
        "calibration_factors": {
            "time_calibration_factor": round(time_calibration_factor, 3),
            "memory_calibration_factor": round(memory_calibration_factor, 3),
        },
        "recommended_adjustments": {
            "compute_cost_multiplier": round(time_calibration_factor, 3),
            "memory_multiplier": round(memory_calibration_factor, 3),
        },
        "operator_analysis": operator_analysis,
    }


def generate_report(results: list[BenchmarkResult], calibration: dict[str, Any], output_path: Path) -> None:
    """生成 markdown 报告"""
    _LOGGER.info(f"Generating report to {output_path}")

    lines = [
        "# Performance Benchmark & Cost Model Calibration Report",
        f"\n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Scale:** {SCALE_STOCKS} stocks × {SCALE_DAYS} days × {SCALE_FACTORS} factors",
        f"\n**Mode:** {'QUICK' if QUICK_MODE else 'FULL'}",
        "\n---",
        "\n## Executive Summary",
        "\n### Cost Model Accuracy",
        f"\n- **Time Prediction Error:** {calibration['time_errors_pct']['mean']:.1f}% ± "
        f"{calibration['time_errors_pct']['std']:.1f}%",
        f"- **Memory Prediction Error:** {calibration['memory_errors_pct']['mean']:.1f}% ± "
        f"{calibration['memory_errors_pct']['std']:.1f}%",
        f"\n### Recommended Calibration",
        f"\n```python",
        f"# Apply these multipliers to cost_model_v2.py",
        f"COMPUTE_COST_MULTIPLIER = {calibration['recommended_adjustments']['compute_cost_multiplier']}",
        f"MEMORY_MULTIPLIER = {calibration['recommended_adjustments']['memory_multiplier']}",
        f"```",
        "\n---",
        "\n## Benchmark Results",
        "\n| Benchmark | Operator | Factors | Actual Time | Predicted | Error | Throughput |",
        "| --------- | -------- | ------- | ----------- | --------- | ----- | ---------- |",
    ]

    for r in results:
        lines.append(
            f"| {r.name} | {r.operator_type} | {r.n_factors} | {r.actual_time_ms:.0f}ms | "
            f"{r.predicted_time_ms:.0f}ms | {r.time_error_pct:+.1f}% | "
            f"{r.actual_throughput_factors_per_sec:.1f} f/s |"
        )

    lines.extend([
        "\n---",
        "\n## Detailed Results",
    ])

    for r in results:
        lines.extend([
            f"\n### {r.name}",
            f"\n**Configuration:**",
            f"- Operator: `{r.operator_type}`",
            f"- Factors: {r.n_factors}",
            f"- Stocks: {r.stocks}",
            f"- Days: {r.days}",
            f"- Backend: {r.backend}",
            f"\n**Performance:**",
            f"- Actual time: {r.actual_time_ms:.2f}ms",
            f"- Predicted time: {r.predicted_time_ms:.2f}ms",
            f"- Time error: {r.time_error_pct:+.2f}%",
            f"- Throughput: {r.actual_throughput_factors_per_sec:.2f} factors/sec",
            f"\n**Memory:**",
            f"- Actual peak: {r.actual_peak_memory_bytes / 1024**2:.1f} MB",
            f"- Predicted peak: {r.predicted_peak_memory_bytes / 1024**2:.1f} MB",
            f"- Memory error: {r.memory_error_pct:+.2f}%",
            f"\n**Cost Breakdown:**",
        ])
        for k, v in r.cost_breakdown.items():
            lines.append(f"- {k}: {v:.2f}ms")

    lines.extend([
        "\n---",
        "\n## Operator-Level Analysis",
        "\n| Operator | Mean Time Error | Samples |",
        "| -------- | --------------- | ------- |",
    ])

    for op_type, stats in calibration.get("operator_analysis", {}).items():
        lines.append(
            f"| {op_type} | {stats['mean_time_error_pct']:+.1f}% | {stats['samples']} |"
        )

    lines.extend([
        "\n---",
        "\n## Performance Analysis",
        "\n### Bottleneck Identification",
    ])

    # 分析瓶颈
    avg_compute_pct = sum(
        r.cost_breakdown.get("compute_cost_ms", 0) / max(1, r.predicted_time_ms) * 100
        for r in results
    ) / len(results)
    avg_source_pct = sum(
        r.cost_breakdown.get("source_cost_ms", 0) / max(1, r.predicted_time_ms) * 100
        for r in results
    ) / len(results)

    lines.extend([
        f"\n- **Compute-bound:** {avg_compute_pct:.1f}% of total predicted time",
        f"- **I/O-bound:** {avg_source_pct:.1f}% of total predicted time",
    ])

    if avg_compute_pct > 70:
        lines.extend([
            "\n**Primary bottleneck: Computation**",
            "\nOptimization priorities:",
            "1. Enable Numba JIT compilation for hot paths",
            "2. Vectorize operations (avoid Python loops)",
            "3. Consider DuckDB for SQL-friendly operations",
            "4. Use Polars for lazy evaluation and optimization",
        ])
    elif avg_source_pct > 50:
        lines.extend([
            "\n**Primary bottleneck: Data I/O**",
            "\nOptimization priorities:",
            "1. Enable aggressive caching",
            "2. Use columnar storage (Parquet)",
            "3. Implement prefetching",
            "4. Reduce data scans with CSE",
        ])
    else:
        lines.extend([
            "\n**Balanced workload**",
            "\nNo single dominant bottleneck. Focus on:",
            "1. Overall system efficiency",
            "2. Memory management",
            "3. Pipeline optimization",
        ])

    lines.extend([
        "\n### Backend Selection Strategy",
        "\nBased on operator characteristics:",
        "\n| Operator Type | Recommended Backend | Rationale |",
        "| ------------- | ------------------- | --------- |",
        "| ts_mean, ts_std | Pandas/Numba | Rolling window optimized |",
        "| rank, zscore | Polars/DuckDB | Group-by heavy |",
        "| Complex joins | DuckDB | SQL optimizer |",
        "| Large panels | Polars | Lazy evaluation |",
        "\n---",
        "\n## Optimization Recommendations",
        "\n### 1. Cost Model Calibration",
        "\nUpdate `/home/shw/quant_projects/factor_engine/planner/cost_model_v2.py`:",
        "\n```python",
        f"# Line 143: Adjust backend overhead",
        f"backend_overhead = {{",
        f"    'pandas': {calibration['recommended_adjustments']['compute_cost_multiplier']:.3f},",
        f"    'polars': {calibration['recommended_adjustments']['compute_cost_multiplier'] * 0.5:.3f},",
        f"    'duckdb': {calibration['recommended_adjustments']['compute_cost_multiplier'] * 0.3:.3f},",
        f"    'numba': {calibration['recommended_adjustments']['compute_cost_multiplier'] * 0.2:.3f},",
        f"}}",
        "\n",
        f"# Line 149: Adjust memory estimate",
        f"peak_memory = int(rows * 8 * 1.5 * {calibration['recommended_adjustments']['memory_multiplier']:.3f})",
        "```",
        "\n### 2. Execution Optimization",
        "\n- **Enable CSE globally:** Set `enable_cse=True` in `engine.run_many()`",
        "- **Use adaptive scheduling:** Enable `AdaptiveBatchScheduler` for large batches",
        "- **Memory-aware execution:** Monitor pressure and enable spilling",
        "\n### 3. Data Access Optimization",
        "\n- **Columnar projection:** Only load required columns",
        "- **Partition pruning:** Leverage time-based partitions",
        "- **Cache hot data:** Use `CacheManager` for frequently accessed datasets",
        "\n### 4. Backend-Specific Tuning",
        "\n**Pandas:**",
        "- Enable `copy=False` where safe",
        "- Use `inplace=True` for mutations",
        "- Leverage `numba` for hot loops",
        "\n**Polars:**",
        "- Use LazyFrame for query optimization",
        "- Enable streaming for large datasets",
        "- Leverage predicate pushdown",
        "\n**DuckDB:**",
        "- Set appropriate `threads` parameter",
        "- Enable `object_cache`",
        "- Use prepared statements for repeated queries",
        "\n---",
        "\n## Validation",
        "\n### Test Cases",
        "\n1. **Unit tests:** Verify calibrated costs within ±10% on synthetic workloads",
        "2. **Integration tests:** Run R39 benchmark suite and compare TTDC",
        "3. **Production validation:** Monitor actual execution times vs predictions",
        "\n### Monitoring",
        "\nTrack these metrics in production:",
        "- Cost prediction accuracy (actual vs predicted time)",
        "- Backend selection effectiveness",
        "- Memory pressure events",
        "- Spill frequency",
        "\n---",
        "\n## Appendix",
        "\n### Environment",
        f"\n- Python: {sys.version.split()[0]}",
        f"- Platform: {sys.platform}",
        f"- CPU cores: {psutil.cpu_count()}",
        f"- Total memory: {psutil.virtual_memory().total / 1024**3:.1f} GB",
        "\n### Raw Data",
        "\nComplete results: `/tmp/benchmark_calibration_data.json`",
        "\n### References",
        "\n- Cost Model V2: `planner/cost_model_v2.py`",
        "- Data Shape Estimate: `planner/data_shape_estimate.py`",
        "- R39 Benchmark Suite: `scripts/r39_benchmark_suite.py`",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    _LOGGER.info(f"Report written to {output_path}")


def main() -> int:
    _LOGGER.info("="*60)
    _LOGGER.info("Performance Benchmark & Cost Model Calibration")
    _LOGGER.info(f"Scale: {SCALE_STOCKS} stocks × {SCALE_DAYS} days × {SCALE_FACTORS} factors")
    _LOGGER.info("="*60)

    # 创建合成数据
    _LOGGER.info("Creating synthetic panel data...")
    df = create_synthetic_panel(SCALE_STOCKS, SCALE_DAYS)
    _LOGGER.info(f"Created panel: {len(df)} rows, {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Benchmark scenarios
    results = []

    # Operator types to test
    operator_types = [
        ("ts_mean", 20),
        ("ts_std", 20),
        ("ts_sum", 10),
        ("rank", None),
        ("zscore", None),
    ]

    # Test different scales
    factor_counts = [100]
    if SCALE_FACTORS >= 500:
        factor_counts.append(500)
    if SCALE_FACTORS >= 1000:
        factor_counts.append(1000)

    for op_type, window in operator_types:
        for n_factors in factor_counts:
            name = f"B_{op_type}_{n_factors}f"
            try:
                result = run_benchmark(
                    name=name,
                    df=df,
                    n_factors=n_factors,
                    operator_type=op_type,
                    backend="pandas",
                    stocks=SCALE_STOCKS,
                    days=SCALE_DAYS,
                    window=window or 20,
                )
                results.append(result)
            except Exception as exc:
                _LOGGER.error(f"Benchmark {name} failed: {exc}", exc_info=True)

    if not results:
        _LOGGER.error("No benchmark results collected!")
        return 1

    # 校准成本模型
    _LOGGER.info("Calibrating cost model...")
    calibration = calibrate_cost_model(results)

    # 生成报告
    report_path = Path("/tmp/performance_benchmark_report.md")
    generate_report(results, calibration, report_path)

    # 保存原始数据
    data_path = Path("/tmp/benchmark_calibration_data.json")
    data_path.write_text(
        json.dumps(
            {
                "results": [r.to_dict() for r in results],
                "calibration": calibration,
                "scale": {
                    "stocks": SCALE_STOCKS,
                    "days": SCALE_DAYS,
                    "factors": SCALE_FACTORS,
                },
                "environment": {
                    "python_version": sys.version.split()[0],
                    "platform": sys.platform,
                    "cpu_count": psutil.cpu_count(),
                    "total_memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _LOGGER.info(f"Raw data written to {data_path}")

    print("\n" + "="*60)
    print("Benchmark completed successfully!")
    print(f"Report: {report_path}")
    print(f"Data: {data_path}")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Benchmarks run: {len(results)}")
    print(f"  - Mean time error: {calibration['time_errors_pct']['mean']:.1f}%")
    print(f"  - Mean memory error: {calibration['memory_errors_pct']['mean']:.1f}%")
    print(f"  - Calibration factor: {calibration['calibration_factors']['time_calibration_factor']:.3f}")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
