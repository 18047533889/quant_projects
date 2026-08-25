# -*- coding: utf-8 -*-
"""MB-P1-023: Parallel region scheduler optimization.

Region 并行调度优化（region = 同 backend 的连续算子序列）：
    - 识别独立 region（无数据依赖）
    - 并行执行多个 region（跨 backend 并行）
    - Region-level resource reservation（避免 oversubscription）
    - 动态负载均衡（慢 region 优先调度）

R45 production-closure additions:
    - ResourceBroker admission：每个 region 在提交前经 broker ``try_reserve``
      获取 CPU/RAM/IO token，完成/取消时释放（避免 4×8GB 在 16GB 主机上超订）。
    - 动态并行度：``max_parallel_regions`` 默认从可用 CPU/RAM 推导，而非固定 4。
    - Deadline / cancellation：region 超时 → ``future.cancel()`` + 释放租约 +
      级联取消后代（后代不再调度）。
    - DAG cycle 校验：调度前硬报错（``PhysicalPlanCycleError``），不再静默 no-op。

示例：
    Factor A: Pandas region (10s) + DuckDB region (5s)
    Factor B: Polars region (8s) + DuckDB region (3s)
    → 串行：10+5+8+3 = 26s
    → 并行：max(10+5, 8+3) = 15s（42% 时间节省）
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


class PhysicalPlanCycleError(Exception):
    """Region 依赖图存在环时抛出（R45：调度前硬校验，不再静默 no-op）。"""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__(
            f"cyclic dependency detected in region graph: {' -> '.join(cycle)}"
        )


class ResourceAdmissionError(Exception):
    """All ready regions were rejected by the ResourceBroker and none is running.

    R50: the scheduler must fail closed instead of silently breaking out of the
    scheduling loop when the broker denies every ready region with no task in
    flight. The caller must treat this as an admission failure (raise throughput,
    free resources, or retry), never as a successful no-op.
    """


@dataclass
class ExecutionRegion:
    """单个执行 region（连续同 backend 算子）。"""

    region_id: str
    backend: str
    operators: list[dict[str, Any]]
    dependencies: list[str]  # 依赖的 region IDs
    estimated_cost_ms: float
    memory_requirement_bytes: int
    deadline_ms: float | None = None  # R45：region 绝对 deadline（monotonic ms）

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "backend": self.backend,
            "operator_count": len(self.operators),
            "dependencies": self.dependencies,
            "estimated_cost_ms": round(self.estimated_cost_ms, 2),
            "memory_requirement_bytes": self.memory_requirement_bytes,
            "deadline_ms": self.deadline_ms,
        }


@dataclass
class TypedRegionFailure:
    """Typed region failure with structured error information."""

    region_id: str
    error_type: str
    error_message: str
    exception: Exception | None = None
    failed_dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "failed_dependencies": self.failed_dependencies,
        }


@dataclass
class RegionExecutionResult:
    """Result of a single region execution."""

    region_id: str
    success: bool
    result: Any = None
    failure: TypedRegionFailure | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.success:
            return {"region_id": self.region_id, "success": True, "result": self.result}
        return {
            "region_id": self.region_id,
            "success": False,
            "failure": self.failure.to_dict() if self.failure else None,
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


def _derive_default_parallelism() -> int:
    """R45：从可用 CPU / RAM 推导默认并行度（替代固定 4）。

    优先 cgroup/psutil 探测；不可用时回退保守启发式。结果 clamp 到
    ``[1, cpu_slots]`` 且受 RAM 预算约束。
    """
    cpu_slots = 0
    mem_limit = 0
    try:
        from factor_engine.runtime.resource_governor import (
            effective_cpu_slots,
            effective_memory_limit_bytes,
        )

        cpu_slots = int(effective_cpu_slots())
        mem_limit = int(effective_memory_limit_bytes())
    except Exception:
        pass
    if cpu_slots <= 0:
        try:
            cpu_slots = int(os.sysconf("SC_NPROCESSORS_ONLN"))
        except (ValueError, OSError, AttributeError):
            cpu_slots = 4
    if mem_limit <= 0:
        try:
            import psutil  # type: ignore[import-untyped]

            mem_limit = int(psutil.virtual_memory().total)
        except Exception:
            mem_limit = 0

    by_cpu = max(1, cpu_slots)
    if mem_limit > 0:
        # 保守启发式：每个 region 默认预留 2GB 峰值（无更精确契约时）。
        per_region_guess = 2 * 1024**3
        by_mem = max(1, mem_limit // per_region_guess)
    else:
        by_mem = by_cpu
    return max(1, min(by_cpu, by_mem))


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
        max_parallel_regions: int | None = None,
        resource_broker: Any | None = None,
        default_deadline_ms: float | None = None,
        production: bool = False,
    ) -> None:
        """
        Args:
            max_parallel_regions: 最大并行 region 数（受 CPU/memory 约束）。
                ``None`` 时从可用资源动态推导（R45）。
            resource_broker: ``ResourceBroker`` 实例。提供时每个 region
                提交前经 ``try_reserve`` 获取 CPU/RAM/IO token，完成/取消时释放。
            default_deadline_ms: 未在 region 上显式指定 deadline 时的默认
                deadline（monotonic ms）。``None`` 表示不设默认 deadline。
            production: ``True`` 强制要求 ``resource_broker``（R50）。生产环境
                必须经 broker 准入避免 oversubscription；传 ``False``（研究/向后
                兼容）允许 ``None``。
        """
        if production and resource_broker is None:
            raise ValueError(
                "ParallelRegionScheduler(production=True) requires a resource_broker; "
                "production must never schedule regions without resource admission."
            )
        if max_parallel_regions is None:
            max_parallel_regions = _derive_default_parallelism()
        self.max_parallel_regions = max(1, int(max_parallel_regions))
        self.resource_broker = resource_broker
        self.production = bool(production)
        self.default_deadline_ms = default_deadline_ms
        self._metrics = ParallelSchedulerMetrics()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self.max_parallel_regions)

    # -- R45: DAG cycle validation --

    def _validate_dag(self, dep_graph: dict[str, set[str]]) -> None:
        """R45：调度前校验依赖图无环；有环抛 ``PhysicalPlanCycleError``。

        用 Kahn 拓扑排序检测环，并提取一个环路径用于报错。
        """
        indegree = {rid: len(deps) for rid, deps in dep_graph.items()}
        reverse: dict[str, set[str]] = {rid: set() for rid in dep_graph}
        for rid, deps in dep_graph.items():
            for d in deps:
                if d in reverse:
                    reverse[d].add(rid)
        queue = [rid for rid, deg in indegree.items() if deg == 0]
        processed = 0
        while queue:
            node = queue.pop()
            processed += 1
            for child in reverse.get(node, set()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if processed != len(dep_graph):
            # 剩余 indegree>0 的节点构成环。沿依赖走出一条环路径。
            remaining = [rid for rid, deg in indegree.items() if deg > 0]
            cycle: list[str] = []
            seen: set[str] = set()
            start = remaining[0]
            cur = start
            while cur not in seen:
                seen.add(cur)
                cycle.append(cur)
                nxt = next(
                    (d for d in dep_graph[cur] if d in seen or indegree.get(d, 0) > 0),
                    None,
                )
                if nxt is None:
                    break
                cur = nxt
            # 截取环起点
            if start in cycle:
                idx = cycle.index(start)
                cycle = cycle[idx:] + [start]
            raise PhysicalPlanCycleError(cycle)

    def schedule_parallel(
        self,
        regions: list[ExecutionRegion],
        execute_fn: Any,
    ) -> dict[str, Any]:
        """并行调度多个 region（topological ready-queue, work-conserving）。

        与旧版 strict-layer-barrier 的区别：
            - 旧版：Layer 0 全部完成 → Layer 1 全部完成 → ...（层内等待）
            - 新版：region 完成后立即唤醒其依赖者（ready queue push），不再等
              同层未完成的兄弟。
            - 优势：减少关键路径延迟（关键 path 上的 region 不被非关键 path 上
              的兄弟阻塞）。

        Args:
            regions: Region 列表
            execute_fn: Region 执行函数（签名：execute_fn(region) -> result）

        Returns:
            执行结果摘要 {"results": {region_id: result}, "elapsed_ms": ...}

        Raises:
            PhysicalPlanCycleError: 依赖图存在环时（调度前硬校验）。
        """
        import time

        start_ms = time.monotonic() * 1000.0

        # 构建依赖图
        dep_graph = {r.region_id: set(r.dependencies) for r in regions}
        region_by_id = {r.region_id: r for r in regions}

        # R45: 调度前硬校验 DAG 无环（不再静默 no-op）。
        self._validate_dag(dep_graph)

        # 反向依赖图（用于推入 ready queue）
        reverse_deps: dict[str, set[str]] = {rid: set() for rid in dep_graph}
        for rid, deps in dep_graph.items():
            for d in deps:
                if d in reverse_deps:
                    reverse_deps[d].add(rid)

        results: dict[str, RegionExecutionResult] = {}
        failed_regions: set[str] = set()
        completed: set[str] = set()

        # Ready queue：所有依赖已满足的 region
        ready_queue: list[str] = [
            rid for rid, deps in dep_graph.items() if not deps
        ]

        # 在跑 futures: {region_id: Future}
        running: dict[str, Any] = {}
        # R45: 每个 admitted region 的 broker 租约（完成/取消时释放）。
        leases: dict[str, Any] = {}
        # R45: 每个 admitted region 的 deadline（monotonic ms）。
        deadlines: dict[str, float] = {}
        admitted: set[str] = set()

        def _build_contract(region: ExecutionRegion) -> Any:
            """把 region 映射为 TaskResourceContract（供 broker admission）。"""
            from factor_engine.runtime.task_resource_contract import TaskResourceContract

            return TaskResourceContract(
                predicted_elapsed_ms=region.estimated_cost_ms,
                cpu_tokens=1,
                io_tokens=1,
                peak_memory_bytes=region.memory_requirement_bytes,
                backend=region.backend,
                estimate_basis="region_scheduler",
            )

        def _release_lease(rid: str) -> None:
            lease = leases.pop(rid, None)
            if lease is not None:
                try:
                    lease.release()
                except Exception as exc:  # pragma: no cover - defensive
                    _logger.warning("lease release failed for %s: %s", rid, exc)

        def _admit_from_queue() -> None:
            """从 ready_queue 提交可运行的 region 到线程池（broker 准入）。"""
            nonlocal ready_queue
            admitted_this_round: list[str] = []
            rejected_any = False
            for rid in ready_queue:
                if len(running) >= self.max_parallel_regions:
                    break
                if rid in admitted or rid in completed or rid in failed_regions:
                    continue
                # 二次检查依赖（可能在 queue 排序期间被取消）
                failed_deps = [d for d in dep_graph[rid] if d in failed_regions]
                if failed_deps:
                    results[rid] = RegionExecutionResult(
                        region_id=rid,
                        success=False,
                        failure=TypedRegionFailure(
                            region_id=rid,
                            error_type="DependencyCancellation",
                            error_message=f"Region cancelled due to failed dependencies: {failed_deps}",
                            failed_dependencies=failed_deps,
                        ),
                    )
                    failed_regions.add(rid)
                    continue
                region = region_by_id[rid]
                # R45: broker admission —— 拒绝则留在 ready_queue 等下一轮
                #（work-conserving，不 deadlock）。
                if self.resource_broker is not None:
                    contract = _build_contract(region)
                    lease = self.resource_broker.try_reserve(
                        contract, task_id=rid
                    )
                    if lease is None:
                        rejected_any = True
                        continue  # 资源不足：本轮不提交，留待后续
                    leases[rid] = lease
                future = self._executor.submit(execute_fn, region)
                running[rid] = future
                admitted.add(rid)
                admitted_this_round.append(rid)
                # R45: 记录 deadline（region 显式 > 默认）。
                if region.deadline_ms is not None:
                    deadlines[rid] = region.deadline_ms
                elif self.default_deadline_ms is not None:
                    deadlines[rid] = start_ms + self.default_deadline_ms
            # Remove admitted regions from queue (keep unadmitted ones for next round)
            ready_queue = [r for r in ready_queue if r not in admitted_this_round]
            # R50: broker 拒绝全部 ready region 且无任何在跑任务 → fail closed，
            # 绝不静默 break 出主循环（会假成功返回空结果）。
            if (
                self.resource_broker is not None
                and rejected_any
                and not admitted_this_round
                and not running
            ):
                raise ResourceAdmissionError(
                    "ResourceBroker rejected every ready region and no region is "
                    "running; unable to make scheduling progress (fail closed)."
                )

        def _cancel_descendants(failed_rid: str) -> None:
            """Recursively cancel all descendants of a failed region."""
            for child in reverse_deps.get(failed_rid, set()):
                if child in completed or child in failed_regions or child in admitted:
                    continue
                child_deps = dep_graph[child]
                failed_deps = [d for d in child_deps if d in failed_regions]
                if failed_deps:
                    results[child] = RegionExecutionResult(
                        region_id=child,
                        success=False,
                        failure=TypedRegionFailure(
                            region_id=child,
                            error_type="DependencyCancellation",
                            error_message=f"Region cancelled due to failed dependencies: {failed_deps}",
                            failed_dependencies=failed_deps,
                        ),
                    )
                    failed_regions.add(child)
                    # Recursively cancel this child's descendants
                    _cancel_descendants(child)

        def _on_complete(rid: str) -> None:
            """region 完成后的回调：记录结果 + 推入 ready queue。"""
            completed.add(rid)
            running.pop(rid, None)
            deadlines.pop(rid, None)
            _release_lease(rid)
            # 推入 ready queue：依赖此 region 且所有依赖已满足
            for child in reverse_deps.get(rid, set()):
                if child in completed or child in failed_regions or child in admitted:
                    continue
                child_deps = dep_graph[child]
                if child_deps.issubset(completed):
                    ready_queue.append(child)

        def _check_deadlines(now_ms: float) -> None:
            """R45: 检查 running region 是否超时；超时则取消 + 级联取消后代。"""
            expired = [
                rid
                for rid, dl in deadlines.items()
                if now_ms >= dl and rid in running
            ]
            for rid in expired:
                future = running.get(rid)
                if future is not None:
                    future.cancel()
                _release_lease(rid)
                running.pop(rid, None)
                deadlines.pop(rid, None)
                results[rid] = RegionExecutionResult(
                    region_id=rid,
                    success=False,
                    failure=TypedRegionFailure(
                        region_id=rid,
                        error_type="DeadlineExceeded",
                        error_message=f"Region exceeded its deadline ({deadlines.get(rid, '?')}ms)",
                    ),
                )
                failed_regions.add(rid)
                _cancel_descendants(rid)

        # 主循环：work-conserving ready-queue 调度
        while ready_queue or running:
            _admit_from_queue()
            if not running:
                break
            # 等待任意一个完成（事件驱动，非轮询）
            done_set, _ = wait(
                list(running.values()),
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )
            now_ms = time.monotonic() * 1000.0
            # R45: 先处理 deadline 超时（含 wait 超时返回的轮次）。
            _check_deadlines(now_ms)
            for future in done_set:
                # 找到对应的 region_id
                rid = None
                for r, f in list(running.items()):
                    if f is future:
                        rid = r
                        break
                if rid is None:
                    continue
                try:
                    result = future.result(timeout=0)
                    results[rid] = RegionExecutionResult(
                        region_id=rid, success=True, result=result
                    )
                    with self._lock:
                        self._metrics.total_regions_scheduled += 1
                except Exception as exc:
                    _logger.error("region %s execution failed: %s", rid, exc)
                    failure = TypedRegionFailure(
                        region_id=rid,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        exception=exc,
                    )
                    results[rid] = RegionExecutionResult(
                        region_id=rid, success=False, failure=failure
                    )
                    failed_regions.add(rid)
                    # Cancel all descendants of this failed region
                    _cancel_descendants(rid)
                _on_complete(rid)

        elapsed_ms = (time.monotonic() * 1000.0) - start_ms

        with self._lock:
            self._metrics.total_elapsed_ms += elapsed_ms

        # 估算节省时间（串行 vs 并行）
        serial_time_ms = sum(r.estimated_cost_ms for r in regions)
        saved_ms = max(0.0, serial_time_ms - elapsed_ms)

        with self._lock:
            self._metrics.total_saved_ms += saved_ms

        return {
            "results": {rid: res.to_dict() for rid, res in results.items()},
            "elapsed_ms": elapsed_ms,
            "estimated_serial_ms": serial_time_ms,
            "saved_ms": saved_ms,
            "layers": self._topological_depth(dep_graph),
            "failed_regions": list(failed_regions),
        }

    def _topological_depth(self, dep_graph: dict[str, set[str]]) -> int:
        """计算 DAG 的拓扑深度（关键路径层数），用于估算加速比。"""
        depths: dict[str, int] = {}

        def _depth(rid: str) -> int:
            if rid in depths:
                return depths[rid]
            deps = dep_graph.get(rid, set())
            if not deps:
                depths[rid] = 0
                return 0
            d = max(_depth(d) for d in deps if d in dep_graph) + 1
            depths[rid] = d
            return d

        for rid in dep_graph:
            _depth(rid)
        return max(depths.values()) + 1 if depths else 0

    def _topological_layers(
        self, dep_graph: dict[str, set[str]]
    ) -> list[list[str]]:
        """拓扑排序 + 分层（每层是独立可并行的节点）。

        Args:
            dep_graph: 依赖图（node → dependencies）

        Returns:
            分层列表（每层是无依赖冲突的 node 列表）

        Raises:
            PhysicalPlanCycleError: 依赖图存在环时（R45：不再 ``break`` 静默）。
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
                # 循环依赖：硬报错（R45）。
                self._validate_dag(dep_graph)  # 会抛 PhysicalPlanCycleError
                # 防御：validate 未抛（理论上不会）则显式抛。
                raise PhysicalPlanCycleError(list(remaining))

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
