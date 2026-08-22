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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable

_logger = logging.getLogger(__name__)


@dataclass
class RegionOutputBundle:
    """Multiple named live-out values produced by a single execution region.

    A region may drive several downstream consumers (transfer edges, factor
    roots, or materialized sinks), each requiring its own output value. This
    bundle is the per-region result value: it holds every named live-out so
    consumers retrieve exactly the output they depend on instead of being
    forced to share one scalar result.

    Behavior:
        - Values are stored by stable output name (``dict[str, Any]``).
        - ``get``/``__getitem__`` retrieve a named output; a missing name is
          a contract violation and fails closed (KeyError).
        - ``only()`` returns the single output when the region produces
          exactly one value (backward-compatible scalar access).
        - ``merge`` combines bundles (or named values) from executed
          sub-regions into one aggregate result.
        - ``to_dict``/``names`` preserve telemetry; bundles that wrap a plain
          scalar value are flattened in ``to_dict`` so single-output regions
          keep their previous summary shape.
    """

    outputs: dict[str, Any] = field(default_factory=dict)
    region_id: str | None = None

    @property
    def is_single(self) -> bool:
        """True when the bundle holds exactly one named output."""
        return len(self.outputs) == 1

    @property
    def names(self) -> list[str]:
        """Live-out names in insertion order."""
        return list(self.outputs)

    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve a named live-out value (with explicit default).

        Args:
            name: Live-out/edge output name.
            default: Returned when ``name`` is absent instead of raising.

        Returns:
            The value bound to ``name``, or ``default`` when missing.
        """
        return self.outputs.get(name, default)

    def __getitem__(self, name: str) -> Any:
        """Retrieve a named live-out value.

        Raises:
            KeyError: When ``name`` is not a live-out written by the region.
        """
        if name not in self.outputs:
            raise KeyError(
                f"RegionOutputBundle {self.region_id or '?'} has no live-out "
                f"{name!r}; known outputs: {sorted(self.outputs)}"
            )
        return self.outputs[name]

    def only(self) -> Any:
        """Return the region's single output value.

        Raises:
            ValueError: When the bundle holds zero or multiple named outputs
                (multi-live-out regions must be consumed by output name).
        """
        names = list(self.outputs)
        if len(names) != 1:
            raise ValueError(
                f"RegionOutputBundle {self.region_id or '?'} has "
                f"{len(names)} live-outs ({sorted(names)}); "
                "multi-live-out regions require by-name retrieval (get/[]) "
                "for each output edge"
            )
        return self.outputs[names[0]]

    def add(self, name: str, value: Any) -> RegionOutputBundle:
        """Bind one named live-out value (in place) and return self."""
        self.outputs[name] = value
        return self

    @classmethod
    def single(cls, value: Any, *, name: str = "result", region_id: str | None = None) -> RegionOutputBundle:
        """Wrap one value as the region's single named live-out."""
        return cls(outputs={name: value}, region_id=region_id)

    @classmethod
    def merge(
        cls,
        values: Iterable[RegionOutputBundle] | dict[str, Any],
        *,
        region_id: str | None = None,
    ) -> RegionOutputBundle:
        """Merge several sub-region bundles (or a bare name->value map) into
        one combined bundle.

        Args:
            values: Bundles to merge, or a raw ``{name: value}`` mapping.
            region_id: Optional id attached to the merged bundle.

        Returns:
            A bundle holding the union of all named outputs. Later bundles win
            when two sources define the same output name (last writer wins),
            which matches deterministic execution order.
        """
        combined: dict[str, Any] = {}
        if isinstance(values, dict):
            combined.update(values)
        else:
            for bundle in values:
                combined.update(bundle.outputs)
        return cls(outputs=combined, region_id=region_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bundle.

        Single-output bundles are flattened to just their scalar value so the
        scheduler's summary dict keeps ``{region_id: <scalar>}`` for legacy
        single-live-out regions. Multi-output bundles serialize to a dict of
        ``{name: value}`` entries.
        """
        if len(self.outputs) == 1:
            return next(iter(self.outputs.values()))
        return dict(self.outputs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RegionOutputBundle(region_id={self.region_id!r}, "
            f"outputs={sorted(self.outputs)})"
        )


@dataclass
class ExecutionRegion:
    """单个执行 region（连续同 backend 算子）。"""

    region_id: str
    backend: str
    operators: list[dict[str, Any]]
    dependencies: list[str]  # 依赖的 region IDs
    estimated_cost_ms: float
    memory_requirement_bytes: int
    thread_requirement: int = 1  # 需要的线程数（资源预留用）
    deadline_ms: float | None = None  # 单 region 执行 deadline（可选）
    output_names: tuple[str, ...] = ()  # 多 live-out 输出名（P0-6）

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "backend": self.backend,
            "operator_count": len(self.operators),
            "dependencies": self.dependencies,
            "estimated_cost_ms": round(self.estimated_cost_ms, 2),
            "memory_requirement_bytes": self.memory_requirement_bytes,
            "thread_requirement": self.thread_requirement,
            "deadline_ms": self.deadline_ms,
            "output_names": list(self.output_names),
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

    def __init__(
        self,
        *,
        max_parallel_regions: int = 4,
        memory_limit_bytes: int | None = None,
        thread_limit: int | None = None,
    ) -> None:
        """
        Args:
            max_parallel_regions: 最大并行 region 数（受 CPU/memory 约束）
            memory_limit_bytes: 总内存预算（资源预留用）。None 表示不限制。
            thread_limit: 总线程预算（资源预留用）。None 表示不限制。
        """
        self.max_parallel_regions = max_parallel_regions
        self.memory_limit_bytes = memory_limit_bytes
        self.thread_limit = thread_limit
        self._metrics = ParallelSchedulerMetrics()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_parallel_regions)

    def _admit_region(
        self,
        region: ExecutionRegion,
        *,
        reserved_memory: int,
        reserved_threads: int,
    ) -> bool:
        """资源预留准入：仅当总预留 + 本 region 需求 <= 预算时放行。

        Args:
            region: 待准入 region
            reserved_memory: 已预留内存字节
            reserved_threads: 已预留线程数

        Returns:
            True 表示可准入（资源充足），False 表示资源不足（拒绝）。
        """
        if self.memory_limit_bytes is not None:
            if reserved_memory + region.memory_requirement_bytes > self.memory_limit_bytes:
                return False
        if self.thread_limit is not None:
            if reserved_threads + region.thread_requirement > self.thread_limit:
                return False
        return True

    def schedule_parallel(
        self,
        regions: list[ExecutionRegion],
        execute_fn: Any,
    ) -> dict[str, Any]:
        """并行调度多个 region（拓扑排序 + 并发执行 + 资源预留 + 祖先失败阻断 + 超时）。

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
        failed: set[str] = set()  # 已失败的 region（含被阻断的）

        # 逐层执行（层内并行）
        for layer_idx, layer in enumerate(layers):
            _logger.info(
                "executing layer %d: %d regions (parallel)", layer_idx, len(layer)
            )

            # 资源预留：层内按依赖顺序累计已预留资源
            reserved_memory = 0
            reserved_threads = 0
            admitted: list[str] = []
            rejected: list[str] = []

            for region_id in layer:
                region = region_by_id[region_id]
                # 祖先失败阻断：任一依赖已失败 → 本 region 直接 fail-closed，不执行
                if dep_graph[region_id] & failed:
                    _logger.error(
                        "region %s blocked: ancestor region(s) %s failed",
                        region_id,
                        sorted(dep_graph[region_id] & failed),
                    )
                    results[region_id] = {
                        "error": "ancestor region failed; descendant blocked (fail-closed)"
                    }
                    failed.add(region_id)
                    continue

                # 资源预留准入
                if not self._admit_region(
                    region,
                    reserved_memory=reserved_memory,
                    reserved_threads=reserved_threads,
                ):
                    _logger.warning(
                        "region %s rejected: resource budget exceeded "
                        "(mem %d/%s, threads %d/%s)",
                        region_id,
                        reserved_memory + region.memory_requirement_bytes,
                        self.memory_limit_bytes,
                        reserved_threads + region.thread_requirement,
                        self.thread_limit,
                    )
                    results[region_id] = {
                        "error": "resource reservation rejected (budget exceeded)"
                    }
                    failed.add(region_id)
                    rejected.append(region_id)
                    continue

                admitted.append(region_id)
                reserved_memory += region.memory_requirement_bytes
                reserved_threads += region.thread_requirement

            # 提交已准入 region（并行执行）
            futures = {}
            for region_id in admitted:
                region = region_by_id[region_id]
                future = self._executor.submit(execute_fn, region)
                futures[future] = region_id

            # 等待层内所有已准入 region 完成（带真实超时）。
            # 注意：不能用 as_completed —— 它要等 future 真正完成才 yield，
            # 导致 future.result(timeout=...) 永远没有机会触发超时。
            # 改为直接遍历 futures，对每个 future 用其 region 的 deadline 调用 result()。
            for future, region_id in futures.items():
                region = region_by_id[region_id]
                deadline_ms = region.deadline_ms
                try:
                    if deadline_ms is not None:
                        result = future.result(timeout=deadline_ms / 1000.0)
                    else:
                        result = future.result()
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
                    results[region_id] = {
                        "error": f"{type(exc).__name__}: {exc}"
                    }
                    failed.add(region_id)

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
            "failed": sorted(failed),
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
                # 循环依赖：fail closed
                from ..exceptions import PhysicalPlanCycleError

                raise PhysicalPlanCycleError(
                    f"Cyclic dependency detected in region graph: "
                    f"remaining nodes {sorted(remaining)}"
                )

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
                "memory_limit_bytes": self.memory_limit_bytes,
                "thread_limit": self.thread_limit,
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
