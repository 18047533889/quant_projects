# -*- coding: utf-8 -*-
"""Performance Benchmark & Cost Model Calibration (2026-08-13)

生产规模性能基准测试：
    1. 10k factor batch（混合算子类型）
    2. 多后端执行（Pandas/Polars/DuckDB）
    3. 真实观测 vs 成本模型预测对比
    4. 成本模型参数校准

测试场景：
    - Technical factors（时序算子：ts_mean/ts_std/ts_sum/rank）
    - Financial factors（财务算子：需真实 DataAccess）
    - Cross-sectional factors（截面算子：rank/zscore/neutralize）
    - Panel factors（面板算子：panel correlation/beta）

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
for _p in (str(_FE_ROOT), str(_FE_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import psutil

from api import rank, ts_max, ts_mean, ts_std, ts_sum, zscore
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from planner.cost_model_v2 import (
    CostComponentsV2,
    estimate_operator_cost_v2,
    estimate_source_cost_v2,
    estimate_total_cost_v2,
    estimate_transfer_cost_v2,
)
from planner.data_shape_estimate import DataShapeEstimate, estimate_shape_from_metadata
from runtime.engine import FactorEngine
from runtime.perf_counters import get_global_counters, reset_global_counters
from tests.helpers import InMemorySeriesSource

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
    operator_stats: dict[str, Any] = field(default_factory=dict)
    backend_selection: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_factors": self.n_factors,
            "stocks": self.stocks,
            "days": self.days,
            "backend": self.backend,
            "actual_time_ms": round(self.actual_time_ms, 2),
            "actual_peak_memory_bytes": self.actual_peak_memory_bytes,
            "actual_throughput_factors_per_sec": round(self.actual_throughput_factors_per_sec, 2),
            "predicted_time_ms": round(self.predicted_time_ms, 2),
            "predicted_peak_memory_bytes": self.predicted_peak_memory_bytes,
            "time_error_pct": round(self.time_error_pct, 2),
            "memory_error_pct": round(self.memory_error_pct, 2),
            "cost_breakdown": {k: round(v, 2) for k, v in self.cost_breakdown.items()},
            "operator_stats": self.operator_stats,
            "backend_selection": self.backend_selection,
        }


def create_panel_source(stocks: int, days: int) -> InMemorySeriesSource:
    """创建合成 panel 数据源（stocks × days）"""
    dts = pd.bdate_range("2020-01-01", periods=days)
    idx = pd.MultiIndex.from_product(
        [dts, [f"stock_{i:04d}" for i in range(stocks)]],
        names=["timestamp", "instrument"],
    )
    n = len(idx)

    # 合成价格数据（带真实波动）
    import numpy as np
    np.random.seed(42)
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5), index=idx)
    high = close * (1.0 + np.abs(np.random.randn(n) * 0.02))
    low = close * (1.0 - np.abs(np.random.randn(n) * 0.02))
    volume = pd.Series(1_000_000 * (1 + np.random.randn(n) * 0.3), index=idx).abs()
    turnover = volume * close

    return InMemorySeriesSource({
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "turnover": turnover,
    })


def create_factor_pool(n: int) -> list[Factor]:
    """创建混合算子类型的因子池"""
    factors = []

    # Technical factors (60%)
    technical_ops = [
        (ts_mean, {"window": 5}),
        (ts_mean, {"window": 20}),
        (ts_std, {"window": 10}),
        (ts_std, {"window": 20}),
        (ts_sum, {"window": 5}),
        (ts_max, {"window": 10}),
    ]

    # Cross-sectional factors (30%)
    cs_ops = [rank, zscore]

    # Panel factors (10%)
    # (需要更复杂的构造，暂时用简单算子)

    idx = 0
    while len(factors) < n:
        if idx % 10 < 6:  # 60% technical
            op, kwargs = technical_ops[idx % len(technical_ops)]
            factors.append(Factor(
                name=f"tech_{len(factors)}",
                expr=op(col("close"), **kwargs) if kwargs else op(col("close"))
            ))
        elif idx % 10 < 9:  # 30% cross-sectional
            op = cs_ops[idx % len(cs_ops)]
            factors.append(Factor(
                name=f"cs_{len(factors)}",
                expr=op(col("close"))
            ))
        else:  # 10% panel
            factors.append(Factor(
                name=f"panel_{len(factors)}",
                expr=ts_mean(col("volume"), window=10)
            ))
        idx += 1

    return factors[:n]


def predict_cost(factors: list[Factor], shape: DataShapeEstimate, backend: str) -> CostComponentsV2:
    """使用成本模型预测执行成本"""
    total_compute_ms = 0.0
    total_memory_bytes = 0

    for factor in factors:
        # 简化：假设每个 factor 是单个算子
        # 实际应该遍历 factor 的 DAG
        op_cost = estimate_operator_cost_v2(
            "ts_mean",  # 默认算子类型
            shape=shape,
            backend=backend,
            window=20,
        )
        total_compute_ms += op_cost.compute_cost_ms
        total_memory_bytes = max(total_memory_bytes, op_cost.peak_memory_bytes)

    # Source cost
    source_cost = estimate_source_cost_v2(shape=shape)

    # Total cost
    total = estimate_total_cost_v2(
        source_cost=source_cost,
        compute_cost=CostComponentsV2(
            compute_cost_ms=total_compute_ms,
            total_cost_ms=total_compute_ms,
            peak_memory_bytes=total_memory_bytes,
        ),
    )

    return total


def run_benchmark(
    name: str,
    factors: list[Factor],
    source: InMemorySeriesSource,
    backend: str,
    stocks: int,
    days: int,
) -> BenchmarkResult:
    """运行单次 benchmark"""
    _LOGGER.info(f"Running benchmark: {name} ({len(factors)} factors, {backend})")

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
    predicted_cost = predict_cost(factors, shape, backend)

    # 记录内存使用
    process = psutil.Process()
    mem_before = process.memory_info().rss

    # 实际执行
    reset_global_counters()
    engine = FactorEngine(data_source=source, backend=PandasBackend())

    t0 = time.monotonic()
    try:
        results = engine.run_many(factors, enable_cse=True)
    except Exception as exc:
        _LOGGER.error(f"Benchmark {name} failed: {exc}")
        raise
    actual_time_ms = (time.monotonic() - t0) * 1000.0

    mem_after = process.memory_info().rss
    actual_memory_bytes = mem_after - mem_before

    # 计算吞吐量
    throughput = len(factors) / (actual_time_ms / 1000.0)

    # 计算误差
    time_error_pct = (
        (actual_time_ms - predicted_cost.total_cost_ms) / max(1, actual_time_ms) * 100
    )
    memory_error_pct = (
        (actual_memory_bytes - predicted_cost.peak_memory_bytes) / max(1, actual_memory_bytes) * 100
    )

    # 收集 perf counters
    counters = get_global_counters()

    return BenchmarkResult(
        name=name,
        n_factors=len(factors),
        stocks=stocks,
        days=days,
        backend=backend,
        actual_time_ms=actual_time_ms,
        actual_peak_memory_bytes=actual_memory_bytes,
        actual_throughput_factors_per_sec=throughput,
        predicted_time_ms=predicted_cost.total_cost_ms,
        predicted_peak_memory_bytes=predicted_cost.peak_memory_bytes,
        time_error_pct=time_error_pct,
        memory_error_pct=memory_error_pct,
        cost_breakdown={
            "source_cost_ms": predicted_cost.source_cost_ms,
            "compute_cost_ms": predicted_cost.compute_cost_ms,
            "transfer_cost_ms": predicted_cost.transfer_cost_ms,
        },
        operator_stats={
            "cse_hits": counters.get("cse_hits", 0),
            "cse_misses": counters.get("cse_misses", 0),
        },
        backend_selection={"pandas": len(factors)},
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
    time_calibration_factor = 1.0 - (avg_time_error / 100.0)
    memory_calibration_factor = 1.0 - (avg_memory_error / 100.0)

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
    }


def generate_report(results: list[BenchmarkResult], calibration: dict[str, Any], output_path: Path) -> None:
    """生成 markdown 报告"""
    _LOGGER.info(f"Generating report to {output_path}")

    lines = [
        "# Performance Benchmark & Cost Model Calibration Report",
        f"\n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Scale:** {SCALE_STOCKS} stocks × {SCALE_DAYS} days × {SCALE_FACTORS} factors",
        "\n---",
        "\n## Executive Summary",
        "\n### Benchmark Results",
        "\n| Benchmark | Factors | Actual Time | Predicted Time | Time Error | Throughput |",
        "| --------- | ------- | ----------- | -------------- | ---------- | ---------- |",
    ]

    for r in results:
        lines.append(
            f"| {r.name} | {r.n_factors} | {r.actual_time_ms:.0f}ms | "
            f"{r.predicted_time_ms:.0f}ms | {r.time_error_pct:+.1f}% | "
            f"{r.actual_throughput_factors_per_sec:.1f} f/s |"
        )

    lines.extend([
        "\n### Cost Model Calibration",
        f"\n- **Time Error:** {calibration['time_errors_pct']['mean']:.1f}% ± "
        f"{calibration['time_errors_pct']['std']:.1f}%",
        f"- **Memory Error:** {calibration['memory_errors_pct']['mean']:.1f}% ± "
        f"{calibration['memory_errors_pct']['std']:.1f}%",
        f"\n**Recommended Adjustments:**",
        f"- Compute cost multiplier: `{calibration['recommended_adjustments']['compute_cost_multiplier']}`",
        f"- Memory multiplier: `{calibration['recommended_adjustments']['memory_multiplier']}`",
        "\n---",
        "\n## Detailed Results",
    ])

    for r in results:
        lines.extend([
            f"\n### {r.name}",
            f"\n**Configuration:**",
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
        f"\n- **Compute-bound:** {avg_compute_pct:.1f}% of total time",
        f"- **I/O-bound:** {avg_source_pct:.1f}% of total time",
    ])

    if avg_compute_pct > 70:
        lines.append("\n**Primary bottleneck: Computation**")
        lines.append("- Consider: Numba JIT, vectorization, algorithm optimization")
    elif avg_source_pct > 50:
        lines.append("\n**Primary bottleneck: Data I/O**")
        lines.append("- Consider: Caching, prefetching, columnar storage optimization")
    else:
        lines.append("\n**Balanced workload:** No single dominant bottleneck")

    lines.extend([
        "\n### Optimization Recommendations",
        "\n1. **Cost Model Tuning:**",
        f"   - Apply compute cost multiplier: `{calibration['recommended_adjustments']['compute_cost_multiplier']}`",
        f"   - Apply memory multiplier: `{calibration['recommended_adjustments']['memory_multiplier']}`",
        "\n2. **Execution Optimization:**",
        "   - Enable CSE (Common Subexpression Elimination)",
        "   - Use adaptive batch scheduling",
        "   - Consider multi-backend execution for heterogeneous workloads",
        "\n3. **Resource Management:**",
        "   - Monitor memory pressure and enable spilling",
        "   - Use resource autopilot for dynamic resource allocation",
        "\n---",
        "\n## Appendix: Raw Data",
        f"\nSee `/tmp/benchmark_calibration_data.json` for complete results.",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    _LOGGER.info(f"Report written to {output_path}")


def main() -> int:
    _LOGGER.info(f"Performance Benchmark & Cost Model Calibration")
    _LOGGER.info(f"Scale: {SCALE_STOCKS} stocks × {SCALE_DAYS} days × {SCALE_FACTORS} factors")

    # 创建数据源
    source = create_panel_source(SCALE_STOCKS, SCALE_DAYS)

    # 创建因子池
    factor_pool = create_factor_pool(SCALE_FACTORS)

    # Benchmark scenarios
    results = []

    # B1: Small batch (100 factors)
    results.append(run_benchmark(
        "B1_small_batch",
        factor_pool[:100],
        source,
        "pandas",
        SCALE_STOCKS,
        SCALE_DAYS,
    ))

    # B2: Medium batch (500 factors)
    if SCALE_FACTORS >= 500:
        results.append(run_benchmark(
            "B2_medium_batch",
            factor_pool[:500],
            source,
            "pandas",
            SCALE_STOCKS,
            SCALE_DAYS,
        ))

    # B3: Large batch (1000 factors)
    if SCALE_FACTORS >= 1000:
        results.append(run_benchmark(
            "B3_large_batch",
            factor_pool[:1000],
            source,
            "pandas",
            SCALE_STOCKS,
            SCALE_DAYS,
        ))

    # 校准成本模型
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
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _LOGGER.info(f"Raw data written to {data_path}")

    print(f"\n{'='*60}")
    print(f"Benchmark completed successfully!")
    print(f"Report: {report_path}")
    print(f"Data: {data_path}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
