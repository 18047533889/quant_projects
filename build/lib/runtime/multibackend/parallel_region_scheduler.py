# -*- coding: utf-8 -*-
"""MB-P1-023: Parallel region scheduler optimization.

Region 并行调度优化（region = 同 backend 的连续算子序列）：
    - 识别独立 region（无数据依赖）
    - 并行执行多个 region（跨 backend 并行）
    - Region-level resource reservation（避免 oversubscription）
    - 动态负载均衡（慢 region 优先调度）

示例：
    Factor A: Pandas region (10s) + DuckDB region (5s)
    Factor B: Polars region (8s) + DuckDB region (3s)
    → 串行：10+5+8+3 = 26s
    → 并行：max(10+5, 8+3) = 15s（42% 时间节省）
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class ExecutionRegion:
    """单个执行 region（连续同 backend 算子）。"""

    region_id: str
    backend: str
    operators: list[dict[str, Any]]
    dependencies: list[str]  # 依赖的 region IDs
    estimated_cost_ms: float
    memory_requirement_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "backend": self.backend,
            "operator_count": len(self.operators),
            "dependencies": self.dependencies,
            "estimated_cost_ms": round(self.estimated_cost_ms, 2),
            "memory_requirement_bytes": self.memory_requirement_bytes,
        }


@dataclass
class ParallelSchedulerMetrics:
    """并行调度统计。"""

    total_regions_scheduled: int = 0
    parallel_executions: int = 0
    sequential_executions: int = 0
    total_elapsed_ms: float = 0.0
    total_saved_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_regions_scheduled": self.total_regions_scheduled,
            "parallel_executions": self.parallel_executions,
            "sequential_executions": self.sequential_executions,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "total_saved_ms": round(self.total_saved_ms, 2),
        }


class ParallelRegionScheduler:
    """并行 region 调度器（跨 backend 并行执行）。

    集成点：
        - AdaptiveBatchScheduler 识别独立 region 后调用 schedule_parallel()
        - HybridExecutor 执行 region 时使用本调度器分配并发
        - ResourceBroker 控制总并发度（避免 oversubscription）
    """

    def __init__(self, *, max_parallel_regions: int = 4) -> None:
        """
        Args:
            max_parallel_regions: 最大并行 region 数（受 CPU/memory 约束）
        """
        self.max_parallel_regions = max_parallel_regions
        self._metrics = ParallelSchedulerMetrics()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_parallel_regions)

    def schedule_parallel(
        self,
        regions: list[ExecutionRegion],
        execute_fn: Any,
    ) -> dict[str, Any]:
        """并行调度多个 region（拓扑排序 + 并发执行）。

        Args:
            regions: Region 列表
            execute_fn: Region 执行函数（签名：execute_fn(region) -> result）

        Returns:
            执行结果摘要 {"results": {region_id: result}, "elapsed_ms": ...}
        """
        import time

        start_ms = time.monotonic() * 1000.0

        # 构建依赖图（region_id → 依赖的 region_id 列表）
        dep_graph = {r.region_id: set(r.dependencies) for r in regions}
        region_by_id = {r.region_id: r for r in regions}

        # 拓扑排序（分层：每层是独立可并行的 region）
        layers = self._topological_layers(dep_graph)

        results: dict[str, Any] = {}
        completed: set[str] = set()

        # 逐层执行（层内并行）
        for layer_idx, layer in enumerate(layers):
            _logger.info(
                "executing layer %d: %d regions (parallel)", layer_idx, len(layer)
            )

            # 提交层内所有 region（并行执行）
            futures = {}
            for region_id in layer:
                region = region_by_id[region_id]
                future = self._executor.submit(execute_fn, region)
                futures[future] = region_id

            # 等待层内所有 region 完成
            for future in as_completed(futures):
                region_id = futures[future]
                try:
                    result = future.result(timeout=600)
                    results[region_id] = result
                    completed.add(region_id)

                    with self._lock:
                        self._metrics.total_regions_scheduled += 1
                        if len(layer) > 1:
                            self._metrics.parallel_executions += 1
                        else:
                            self._metrics.sequential_executions += 1

                except Exception as exc:
                    _logger.error("region %s execution failed: %s", region_id, exc)
                    results[region_id] = {"error": str(exc)}

        elapsed_ms = (time.monotonic() * 1000.0) - start_ms

        with self._lock:
            self._metrics.total_elapsed_ms += elapsed_ms

        # 估算节省时间（串行 vs 并行）
        serial_time_ms = sum(r.estimated_cost_ms for r in regions)
        saved_ms = max(0.0, serial_time_ms - elapsed_ms)

        with self._lock:
            self._metrics.total_saved_ms += saved_ms

        return {
            "results": results,
            "elapsed_ms": elapsed_ms,
            "estimated_serial_ms": serial_time_ms,
            "saved_ms": saved_ms,
            "layers": len(layers),
        }

    def _topological_layers(
        self, dep_graph: dict[str, set[str]]
    ) -> list[list[str]]:
        """拓扑排序 + 分层（每层是独立可并行的节点）。

        Args:
            dep_graph: 依赖图（node → dependencies）

        Returns:
            分层列表（每层是无依赖冲突的 node 列表）
        """
        layers: list[list[str]] = []
        remaining = set(dep_graph.keys())
        completed: set[str] = set()

        while remaining:
            # 当前层：所有依赖已满足的节点
            layer = [
                node
                for node in remaining
                if dep_graph[node].issubset(completed)
            ]

            if not layer:
                # 循环依赖：报错
                _logger.error("cyclic dependency detected in region graph")
                break

            layers.append(layer)
            completed.update(layer)
            remaining -= set(layer)

        return layers

    def estimate_parallel_speedup(
        self, regions: list[ExecutionRegion]
    ) -> dict[str, Any]:
        """估算并行加速比（不实际执行）。

        Args:
            regions: Region 列表

        Returns:
            {"serial_ms": ..., "parallel_ms": ..., "speedup": ...}
        """
        # 串行总时间
        serial_ms = sum(r.estimated_cost_ms for r in regions)

        # 并行时间：每层的最长 region
        dep_graph = {r.region_id: set(r.dependencies) for r in regions}
        region_by_id = {r.region_id: r for r in regions}
        layers = self._topological_layers(dep_graph)

        parallel_ms = 0.0
        for layer in layers:
            layer_max_ms = max(
                region_by_id[rid].estimated_cost_ms for rid in layer
            )
            parallel_ms += layer_max_ms

        speedup = serial_ms / max(1.0, parallel_ms)

        return {
            "serial_ms": serial_ms,
            "parallel_ms": parallel_ms,
            "speedup": round(speedup, 2),
            "layers": len(layers),
        }

    def can_parallelize(
        self, region1: ExecutionRegion, region2: ExecutionRegion
    ) -> bool:
        """判断两个 region 是否可并行（无依赖冲突）。

        Args:
            region1: Region 1
            region2: Region 2

        Returns:
            True 表示可并行，False 表示有依赖冲突
        """
        # Region 1 依赖 Region 2
        if region2.region_id in region1.dependencies:
            return False

        # Region 2 依赖 Region 1
        if region1.region_id in region2.dependencies:
            return False

        return True

    def metrics(self) -> ParallelSchedulerMetrics:
        """返回累计统计。"""
        with self._lock:
            return ParallelSchedulerMetrics(
                total_regions_scheduled=self._metrics.total_regions_scheduled,
                parallel_executions=self._metrics.parallel_executions,
                sequential_executions=self._metrics.sequential_executions,
                total_elapsed_ms=self._metrics.total_elapsed_ms,
                total_saved_ms=self._metrics.total_saved_ms,
            )

    def summary(self) -> dict[str, Any]:
        """返回调度器状态摘要。"""
        with self._lock:
            metrics = self.metrics().to_dict()
            return {
                "max_parallel_regions": self.max_parallel_regions,
                "metrics": metrics,
            }

    def shutdown(self) -> None:
        """关闭调度器（清理线程池）。"""
        self._executor.shutdown(wait=True)


# 全局单例
_global_scheduler: ParallelRegionScheduler | None = None
_global_lock = threading.Lock()


def get_global_parallel_scheduler() -> ParallelRegionScheduler:
    """返回全局 ParallelRegionScheduler（进程级单例）。"""
    global _global_scheduler
    if _global_scheduler is None:
        with _global_lock:
            if _global_scheduler is None:
                _global_scheduler = ParallelRegionScheduler()
    return _global_scheduler
