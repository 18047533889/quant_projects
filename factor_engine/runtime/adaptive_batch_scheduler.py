# -*- coding: utf-8 -*-
"""R27-173/175/193/204: AdaptiveBatchScheduler —— 资源感知 DAG 调度器。

编排（R27-193）：topological ready queue + resource admission + priority
scoring + work stealing。**不是**「shared nodes 先全部算完再按 root layer 执行」。

核心行为
    - 共享 node 与 root node 统一进入 topological ready queue（R27-002/006）。
    - 每个 task 执行前经 :class:`ResourceBroker` memory/CPU/IO/spill token
      admission（R27-037..044）。
    - 并行路径 as_completed：future 完成 → 结果校验 → DQ → sink/write → release
      （R27-103/104/166），而不是先形成整层 list。
    - 并发度在 run 过程中随 live headroom / 外部负载变化（R27-204）；运行中不
      强行减少已运行 task，只改变新 admission（R27-205）。
    - 高 reuse shared node / critical path task 优先（R27-007/008/129）。
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from factor_engine.planner.dag_cost_model import task_priority
from factor_engine.planner.physical_factor_dag import (
    TASK_CSE_SHARED,
    TASK_MERGE,
    TASK_ROOT,
    TASK_SHARD,
    TASK_SOURCE_SCAN,
    PhysicalFactorDAG,
    PhysicalFactorTask,
    rebase_task,
    task_is_executable,
)
from factor_engine.planner.read_wave_planner import ReadWavePlan, build_waves_from_dag
from factor_engine.runtime.buffer_ref import SourceWaveExecutor
from factor_engine.runtime.hybrid_executor import HybridExecutor, classify_backend_execution
from factor_engine.runtime.micro_batch_task import MicroBatchTask, dispatch_micro_batch
from factor_engine.runtime.plan_execution_certificate import (
    PlanExecutionCertificate,
    build_plan_execution_certificate,
)
from factor_engine.runtime.resource_broker import (
    MissingResourceBroker,
    ReservationLease,
    ResourceBroker,
    require_broker,
)
from factor_engine.runtime.streaming_result_sink import ResultItem, StreamingResultSink

_logger = logging.getLogger(__name__)

_V2_DISABLE_INNER_RETRIES: ContextVar[bool] = ContextVar(
    "factor_engine_v2_disable_inner_retries", default=False
)


@contextmanager
def disable_inner_retries_for_v2():
    token = _V2_DISABLE_INNER_RETRIES.set(True)
    try:
        yield
    finally:
        _V2_DISABLE_INNER_RETRIES.reset(token)


def _inner_retry_default() -> int:
    return 0 if _V2_DISABLE_INNER_RETRIES.get() else 1


def _retain_wave_lease(ref: Any, lease: Any) -> None:
    """Release a read-wave lease only after every physical buffer owner dies."""
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    mapping = (getattr(ref, "meta", None) or {}).get("loaded_columns")
    owners: list[Any] = [ref]
    seen = {id(ref)}
    if isinstance(mapping, dict):
        for value in mapping.values():
            physical = DataAccessSource._physical_cache_owners(value)
            if not physical:
                # Ownership is unknown: retain conservatively for process lifetime.
                return
            for owner in physical:
                if id(owner) not in seen:
                    seen.add(id(owner))
                    owners.append(owner)
    elif mapping:
        return
    native_buffer = (getattr(ref, "meta", None) or {}).get("native_buffer")
    if native_buffer is not None:
        # Polars derivations retain Rust Arc owners rather than the originating
        # Python wrapper. Until descendants propagate a lease token, retain
        # native admission conservatively instead of releasing it early.
        return
    state = {"remaining": len(owners), "lease": lease, "lock": threading.Lock()}

    def release_owner(owner_state=state):
        with owner_state["lock"]:
            owner_state["remaining"] -= 1
            if owner_state["remaining"] == 0:
                owner_state["lease"].release()
                owner_state["lease"] = None

    for owner in owners:
        weakref.finalize(owner, release_owner)


def _resolve_execution_policy() -> str | None:
    """P0#8（100k GO §11）：scheduler 执行策略单一权威。

    返回 ``None`` → 交给 HybridExecutor 的 backend classifier（默认，推荐）；
    返回 ``"thread"`` / ``"process"`` → 运维显式强制（``FACTOR_ENGINE_SCHEDULER``
    env 权威覆盖 classifier）。scheduler 不再无条件覆盖 executor classifier。
    """
    raw = os.environ.get("FACTOR_ENGINE_SCHEDULER", "").strip().lower()
    if raw in {"thread", "process"}:
        return raw
    return None

#: R33-P0-040：``wait(FIRST_COMPLETED)`` 事件驱动，超时只作兜底上限（不再固定
#: 50ms 轮询）。
_EVENT_WAIT_TIMEOUT_S = 0.05

#: R39-PERF-018（§6）：micro-batch 参数。
#: 每个 micro-batch 合并 16–128 个低成本 root，组内估计总 work < 阈值。
_MICRO_BATCH_MIN_ROOTS = 16
_MICRO_BATCH_MAX_ROOTS = 128
_MICRO_BATCH_MAX_TOTAL_WORK = 1_000_000.0
_MICRO_BATCH_ROOT_WORK_CAP = 100_000.0


@dataclass
class SchedulerPlan:
    """一次批量的完整调度计划（R27-173/142 dry-run 产物）。"""

    physical_dag: PhysicalFactorDAG
    read_waves: ReadWavePlan
    fusion_groups: list[Any] = field(default_factory=list)
    cost_by_task: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dag": self.physical_dag.to_dict(),
            "read_waves": self.read_waves.to_dict(),
            "fusion_groups": [g.to_dict() for g in self.fusion_groups],
            "cost_by_task": {k: v.to_dict() if hasattr(v, "to_dict") else v
                             for k, v in self.cost_by_task.items()},
            "meta": self.meta,
        }


def _plan_cost_bytes(plan: Any) -> dict[str, Any]:
    try:
        from factor_engine.backend.operator_cost import estimate_plan_cost

        return estimate_plan_cost(plan)
    except Exception:
        return {"total_work": 0.0, "peak_live_memory_bytes": 0}


def _dispatch(
    task: PhysicalFactorTask,
    backend: Any,
    ctx: Any,
    execute_root: Callable[[PhysicalFactorTask], Any],
    materialize_shared: Callable[[str, Any], Any],
) -> tuple[str, Any]:
    """模块级执行函数（可 pickle，供 process pool 使用）：返回 (task_id, result)。

    R27-042：不直接 kill 运行中 task；失败重试耗尽由 scheduler 上层抛出。
    """
    if task.task_type == TASK_CSE_SHARED:
        sid = task.task_id.split(":", 1)[1]
        materialize_shared(sid, task.node_ref)
        return task.task_id, None
    if task.task_type == TASK_ROOT:
        # R35 §23/§24: pin the task's inner BLAS/OpenMP threads to its
        # broker-granted CPU budget so FE_workers × inner_threads never
        # oversubscribes the host.  thread_budget is a no-op when threadpoolctl
        # is unavailable (performance extra not installed).
        budget = 1
        try:
            contract = task.resource_contract
            if contract is not None:
                bt = getattr(contract, "backend_threads", None)
                if bt and int(bt) >= 1:
                    budget = int(bt)
        except Exception:
            budget = 1
        from factor_engine.runtime.execution_traits import thread_budget

        with thread_budget(budget):
            result = execute_root(task)
        return task.task_id, result
    return task.task_id, None


#: 错误分类（R31-006）：只有 transient 类自动 retry；确定性 / 语义类禁止 retry。
#: R38 P0-004（§5）：OOM 是独立类别——same shape 不可重试，smaller shape 可重试。
ERROR_TRANSIENT = "transient"
ERROR_PERMANENT = "permanent"
ERROR_UNKNOWN = "unknown"
ERROR_OOM = "oom"

_TRANSIENT_EXC_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,
)
#: 明确「重跑也白跑」的错误标记（PIT/语义/参数/不支持算子/schema）。
_PERMANENT_MARKERS = (
    "pit ",
    "pit_",
    "semantic",
    "invalid parameter",
    "unsupported operator",
    "schema mismatch",
    "deterministic numeric",
    "dq error",
)


def _is_transient_timeout(exc: TimeoutError) -> bool:
    """P0-FIX: Distinguish transient (network/IO) from permanent (query) timeouts.

    Query complexity timeouts are permanent and should not be retried.
    Network/IO timeouts are transient and can be retried.
    """
    msg = str(exc).lower()
    # Permanent timeout indicators
    if any(marker in msg for marker in ["query", "execution", "compute", "complexity"]):
        return False
    # Transient timeout (network/IO)
    return True


def classify_error(exc: BaseException) -> str:
    """R31-006 + R38-P0-004：错误分类——transient 自动 retry；permanent 禁止 retry。

    - ``OSError/ConnectionError`` 及显式 transient 标记 → transient
    - ``TimeoutError`` 根据消息区分：query timeout 是 permanent，network timeout 是 transient
    - PIT violation / semantic violation / invalid param / unsupported op /
      schema mismatch / deterministic numeric / DQ error → permanent
    - OOM（MemoryError / DuckDB OutOfMemory / Arrow / …）→ ``ERROR_OOM``：
      进入 smaller-shape replan 路径（same shape 禁止重试，R38-P0-005）。
    - 其余 → unknown（失败关闭，不重放未知失败）
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    from factor_engine.runtime.resource_errors import typed_retry_kind

    typed = typed_retry_kind(exc)
    if typed is not None:
        return typed
    if _is_oom(exc):
        return ERROR_OOM
    if any(m in msg for m in _PERMANENT_MARKERS):
        return ERROR_PERMANENT
    # P0-FIX: Distinguish permanent vs transient timeouts
    if isinstance(exc, TimeoutError):
        return ERROR_TRANSIENT if _is_transient_timeout(exc) else ERROR_PERMANENT
    if isinstance(exc, _TRANSIENT_EXC_TYPES) or "transient" in msg:
        return ERROR_TRANSIENT
    return ERROR_UNKNOWN


def _is_oom(exc: BaseException) -> bool:
    try:
        from factor_engine.runtime.resource_errors import is_oom_error

        return is_oom_error(exc)
    except Exception as import_err:
        # P0-FIX: Log fallback OOM detection for observability
        import logging
        logging.getLogger(__name__).warning(
            f"OOM detection fallback active (import failed: {import_err}). "
            f"Using simplified pattern matching."
        )
        # Comprehensive fallback patterns for DuckDB/Arrow/Polars OOM errors
        exc_str = str(exc).lower()
        exc_type = type(exc).__name__.lower()
        return (
            exc_type == "memoryerror"
            or "oom" in exc_str
            or "out of memory" in exc_str
            or "cannot allocate" in exc_str
            or "memory budget exceeded" in exc_str
            or "memory limit exceeded" in exc_str
        )


def _dispatch_fusion(
    group: Any,
    task_by_id: dict[str, PhysicalFactorTask],
    backend: Any,
    ctx: Any,
    execute_root: Callable[[PhysicalFactorTask], Any],
) -> tuple[str, dict[str, Any]]:
    """模块级 fusion 执行函数：返回 ``(group_key, {tid: result})``。

    R31-P0-022：fusion group 真正执行（backend 支持 ``execute_multi_roots`` 则
    一次 native query；否则诚实 per-root fallback 并计数）。
    """
    from factor_engine.planner.native_fusion import execute_fusion_group

    group_key = f"fusion:{group.group_id}"
    results = execute_fusion_group(
        group,
        backend=backend,
        task_by_id=task_by_id,
        ctx=ctx,
        execute_root=execute_root,
    )
    return group_key, results


def _release_lease(lease: Any) -> None:
    """释放单租约或 fusion group 的租约列表（each exactly once）。"""
    if lease is None:
        return
    if isinstance(lease, (list, tuple)):
        for item in lease:
            if item is not None:
                item.release()
    else:
        lease.release()


def _default_contract_for(task_type: str, plan_cost: dict[str, Any]) -> Any:
    from factor_engine.runtime.task_resource_contract import TaskResourceContract

    peak = int(plan_cost.get("peak_live_memory_bytes", 0))
    out_bytes = peak  # 保守：输出 ≈ 峰值面板
    return TaskResourceContract(
        predicted_elapsed_ms=float(plan_cost.get("total_work", 0.0)),
        cpu_tokens=1,
        io_tokens=1 if task_type == TASK_CSE_SHARED else 0,
        peak_memory_bytes=peak,
        output_bytes=out_bytes,
        gil_bound=False,
        releases_gil=True,
        backend="pandas_numpy",
        backend_threads=1,
        shardable=False,
        shard_dimension=None,
        estimate_basis="plan-cost",
    )


class AdaptiveBatchScheduler:
    """统一 cost/resource 调度器（R27-173 API: plan/run/materialize）。"""

    def __init__(
        self,
        *,
        broker: ResourceBroker | None = None,
        executor: HybridExecutor | None = None,
        sink: StreamingResultSink | None = None,
        # P3/P4: ``None`` → 从 broker ResourceDecision 取动态 read-wave 预算，
        # 不再固定 4GiB。broker 不可用时由 ``_dynamic_wave_budget`` 回退绝对上限。
        wave_memory_budget: int | None = None,
        max_concurrency: int | None = None,
        execution_policy: str | None = None,
    ) -> None:
        if execution_policy not in {None, "thread", "process"}:
            raise ValueError("execution_policy must be thread, process, or None")
        if broker is None:
            # R31-P1-039 + R36 P0-016：优先用 service ContextVar 里的共享 broker
            # （job admission + task admission 统一）；没有则用进程级唯一
            # HostResourceCoordinator 的 broker——不再各自 new 独立 ResourceBroker
            #（§52 one host resource authority）。
            try:
                from factor_engine.service.queue import _get_service_broker

                broker = _get_service_broker()
            except Exception:
                broker = None
        if broker is None:
            try:
                from factor_engine.runtime.host_resource_coordinator import get_host_coordinator

                broker = get_host_coordinator().broker
            except Exception:
                broker = None
# P0-11 单权威 gate：production 下 broker 仍为 None → raise
        # MissingResourceBroker（fail-closed）；research/dev 经 require_broker
        # 降级并显式标记（degraded observable）。
        self.broker = require_broker(
            broker,
            raise_on_missing=True,
            run_mode=os.environ.get("FACTOR_ENGINE_RUN_MODE", "").strip().lower() or None,
        )
        self.executor = executor or HybridExecutor(
            broker=self.broker,
            max_thread_workers=max_concurrency,
            max_process_workers=max_concurrency,
        )
        # P0#8（100k GO §11）：scheduler 强制 thread 与 HybridExecutor classifier
        # 冲突的单一权威。默认 ``None`` → 交给 HybridExecutor 的 backend classifier
        # （pandas/GIL-bound → process，native → thread）。仅当运维显式设置
        # ``FACTOR_ENGINE_SCHEDULER=thread|process`` 时，scheduler 才强制覆盖
        # classifier（env 权威）。scheduler 不再无条件覆盖 executor classifier。
        self._execution_policy = (execution_policy if execution_policy is not None
                                  else _resolve_execution_policy())
        if self._execution_policy is not None:
            _logger.info(
                "scheduler execution policy override active: %s "
                "(explicit constructor policy takes precedence over environment)",
                self._execution_policy,
            )
        self.sink = sink
        self.wave_memory_budget = wave_memory_budget
        self.max_concurrency = max_concurrency
        self._lock = threading.RLock()
        self._committed: set[str] = set()
        self._running: dict[str, Future] = {}
        self._results: dict[str, Any] = {}
        self._task_timing: dict[str, dict[str, Any]] = {}
        self._reuse_counts: dict[str, int] = {}
        self._explanations: list[str] = []  # R27-143 可解释性
        self._done = 0
        self._failed_once: set[str] = set()
        self._retries_remaining: dict[str, int] = {}
        # R31-069/P1-040：CancellationToken——cancel() 后不再 admit 新 task，
        # 在跑 task 自然完成；无 future 时 run 提前结束（不阻塞、不泄漏租约）。
        self._cancelled = False
        # R33-P0-016：read wave 执行器（跨循环持久，幂等）+ BufferRef 结果表。
        self._wave_executor: Any | None = None
        self._wave_refs: dict[int, Any] = {}
        self._wave_pending_consumers: dict[int, set[str]] = {}
        self._wave_source_tasks: dict[int, tuple[str, ...]] = {}
        self._wave_summary: dict[str, Any] = {"waves_planned": 0, "waves_executed": 0, "events": []}
        self._input_dq_reports: list[Any] = []
        # R33-P0-009：SOURCE_SCAN task 的 BufferRef 输出（独立命名空间，不污染
        # 因子结果 ``self._results``）。
        self._buffer_results: dict[str, Any] = {}
        # R33-P0-008/§44：real vs virtual task 比（scheduler runtime evidence）。
        self._scheduler_stats: dict[str, Any] = {
            "real_task_done": 0, "virtual_task_done": 0, "virtual_task_ratio": 0.0,
        }
        # R36 P0-002：最近一次 ResourceDecision（broker 建议值，admission 消费）。
        self._last_decision: Any | None = None
        # R38 P0-001/004（§4/§5）：真实 shard 执行状态。
        self._shard_executor: Any | None = None
        #: merge_task_id -> {shard_id: partial_result|SpooledShard}
        self._shard_partials: dict[str, dict[str, Any]] = {}
        #: 已失败（OOM）的 shape 签名——same shape 禁止重试（R38_P0_ZERO_SAME_SHAPE_OOM_RETRY）。
        self._failed_shapes: set[str] = set()
        #: original_task_id -> 该 task 当前 attempt 的 shape 签名（OOM 时定位失败
        #: shape）。统一按 **original_task_id** 存（不再 merge_id/orig_tid 混用）。
        self._shard_shape_of: dict[str, str] = {}
        #: original_task_id -> 分片 attempt 序号（OOM replan 起 >=2，task id 带
        #: ``:attempt:N:`` 段，隔离旧 in-flight future，P0-010）。
        self._shard_attempts: dict[str, int] = {}
        #: original_root_id -> 原始 ROOT task（shard 替换后保留，OOM replan 用）。
        self._shard_original_task: dict[str, Any] = {}
        self._oom_replans = 0
        self._preshards = 0
        # R38 P0-041（§16）：read-wave JIT repartition 状态。
        self._last_wave_budget: int | None = None
        self._wave_covered_tasks: set[str] = set()
        # R38 P0-007：task 提交时刻（dispatch 时记录，完成时算真实 duration）。
        self._task_started_at: dict[str, float] = {}
        # R39-PERF-016：plan() 阶段构建的 task 执行证书（运行期 O(1) 读取）。
        self._certificates: dict[str, PlanExecutionCertificate] = {}
        # R39-PERF-016：critical-path 成本图缓存（避免每 ready task 重走整棵 DAG
        # 构建 cost_ms；DAG 被 shard/OOM replan 时置 None 强制重建）。
        self._cost_ms_cache: dict[str, float] | None = None
        # Cache the reverse-DAG critical-path DP as well.  Recomputing it for
        # every ready task makes a wide ready queue O(V * (V + E)).
        self._critical_path_cache: tuple[int, dict[str, float]] | None = None
        # R39-PERF-018/019：结构化性能计数（run() 开始时重置）。
        self._perf_metrics: dict[str, Any] = {
            "future_count": 0,
            "factor_count": 0,
            "micro_batch_task_count": 0,
            "micro_batch_root_count": 0,
            "scheduler_wait_polling_count": 0,
        }
        self._micro_batch_counter = 0

    def cancel(self) -> None:
        """请求取消：后续 admission 一律拒绝；run 在无在跑任务时提前结束。"""
        self._cancelled = True
        self._explain("CANCELLED: no new task admitted; finishing running tasks")

    def _dynamic_wave_budget(self) -> int:
        """R36 P0-010：从 broker ResourceDecision 取动态 read-wave 预算。

        P3/P4：不再回退固定 4GiB——broker 无法给出真实预算时从 live headroom
        派生（``current_read_budget``）；两者都不可用时才用绝对上限 4GiB 作 cap。
        """
        try:
            decision = self.broker.resource_decision()
            self._last_decision = decision
            return max(1, int(decision.read_wave_bytes))
        except Exception:
            pass
        try:
            live = self.broker.current_read_budget()
            if live > 0:
                return max(1, live)
        except Exception:
            pass
        return 4 * 1024**3

    # -- plan --

    def plan(
        self,
        dag: Any,
        analyses: dict[str, Any] | None = None,
        *,
        enable_cse: bool = True,
        scan_cost_map: dict[str, Any] | None = None,
        scope_scan_cost_map: dict[str, Any] | None = None,
        fusion_backend_capability: dict[str, bool] | None = None,
        ctx: Any | None = None,
    ) -> SchedulerPlan:
        """从 ``DAGPlan``（roots + shared_nodes）构建 **真实 physical DAG** 计划。

        R31-P0-002/003/004：经 :mod:`planner.physical_lowerer` lower 成真实 stage
        （SOURCE_SCAN / OPERATOR / barrier / ROOT），stage 携带真实 backend
        context（来自 ``choose_plan_route``）与 calibrated 资源契约——不再固定
        Pandas / 1 token / 0 bytes。
        """
        from factor_engine.planner.cse import collect_consumed_sids
        from factor_engine.planner.physical_lowerer import contract_for_plan, lower_batch_dag

        # 尝试经 ctx 拿真实行数/仪器数（SourceScanStage 用真实 ScanCost 估计）。
        rows: int | None = None
        instruments = 0
        if ctx is not None:
            try:
                from factor_engine.backend.plan_cost_router import estimate_plan_rows

                rows = estimate_plan_rows(ctx)
            except Exception:
                rows = None
        physical = lower_batch_dag(
            dag,
            analyses=analyses,
            ctx=ctx,
            rows=rows,
            instruments=instruments,
            scan_cost_map=scan_cost_map,
        )
        cost_by_task: dict[str, Any] = {}
        # R39-PERF-016：plan() 阶段为每个 task 构建执行证书——复用这里已算好的
        # plan cost，固化成 frozen 证书，运行期 O(1) 读取，不再重走 DAG。
        self._certificates = {}
        self._cost_ms_cache = None
        self._critical_path_cache = None
        for tid, task in physical.tasks.items():
            plan_cost = (
                task.estimated_cost
                if task.estimated_cost is not None
                else _plan_cost_bytes(task.node_ref)
            )
            cost_by_task[tid] = plan_cost
            self._certificates[tid] = build_plan_execution_certificate(task, plan_cost)
            self._reuse_counts[tid] = len(task.consumers)
        # shared ROOT inputs：root 消费的 CSE sid 记入 ROOT.inputs（可执行依赖），
        # 同时给 CSE shared task 补 consumer 边（topological_order 靠它释放）。
        for fp in dag.roots:
            consumed = collect_consumed_sids(fp.root)
            rid = f"root:{fp.factor_name}"
            if rid in physical.tasks and consumed:
                cur = physical.tasks[rid]
                physical.tasks[rid] = rebase_task(
                    cur,
                    inputs=tuple(sorted(set((*cur.inputs, *(f"cse:{s}" for s in consumed))))),
                )
                for sid in consumed:
                    cid = f"cse:{sid}"
                    if cid not in physical.tasks:
                        continue
                    prev_c = physical.tasks[cid].consumers
                    physical.tasks[cid] = rebase_task(
                        physical.tasks[cid],
                        consumers=tuple(sorted((*prev_c, rid))),
                    )

        # R36 P0-010：read wave 预算不再固定 4GB——plan 时从 broker decision 取
        # 动态值（§35：min(calibrated_optimum, SafeEnvelope×wave_fraction, job lease)）。
        wave_budget = self._dynamic_wave_budget()
        read_waves = build_waves_from_dag(
            physical,
            wave_memory_budget=wave_budget,
            scan_cost_map=scan_cost_map,
            scope_scan_cost_map=scope_scan_cost_map,
        )
        # fusion groups（仅对 ROOT task；R33-P0-042：不再 batch-global
        # ``can_fuse_roots(all_roots)`` 一票否决——按 (backend, source_scope,
        # execution_scope) 分组后每组独立 can_fuse/block）。
        root_tasks = [physical.tasks[t] for t in physical.roots if t in physical.tasks]
        from factor_engine.planner.native_fusion import native_fusion_capability_map, plan_native_fusion_groups

        fusion_groups = []
        if enable_cse and root_tasks:
            capability = (fusion_backend_capability if fusion_backend_capability is not None
                          else native_fusion_capability_map(ctx))
            fusion_groups = plan_native_fusion_groups(
                root_tasks,
                backend_capability=capability,
            )
        return SchedulerPlan(
            physical_dag=physical,
            read_waves=read_waves,
            fusion_groups=fusion_groups,
            cost_by_task=cost_by_task,
            meta={"n_shared": len(dag.shared_nodes or {}), "n_roots": len(dag.roots)},
        )

    # -- certificate (R39-PERF-016) --

    def task_execution_certificate(
        self, task_id: str
    ) -> PlanExecutionCertificate | None:
        """PERF-016：O(1) 读取 task 的执行证书（plan() 阶段构建；无则 None）。"""
        return self._certificates.get(task_id)

    def _cert_cost_for(self, task: PhysicalFactorTask) -> float:
        """O(1) 读取 task 的估计 work：优先证书，回退 contract。"""
        cert = self._certificates.get(task.task_id)
        if cert is not None:
            return cert.total_work()
        contract = task.resource_contract
        if contract is not None:
            return float(contract.predicted_elapsed_ms or 0.0)
        return 0.0

    # -- micro-batch (R39-PERF-018) --

    def _next_micro_batch_id(self) -> int:
        self._micro_batch_counter += 1
        return self._micro_batch_counter

    def _micro_batch_contract(
        self, mb: MicroBatchTask, dag: PhysicalFactorDAG
    ) -> Any | None:
        """构造 micro-batch 的合成资源契约（串行执行的真实 footprint）。

        micro-batch 在**一个**线程内串行执行 root —— 同一时刻只占一个 root 的
        CPU 与峰值内存（工作集），但全部输出会累积到结果 dict。因此：
            cpu_tokens        = max(组内 root)（串行：不是求和）
            peak_memory_bytes = max(组内 root peak) + sum(输出 bytes)
            output_bytes      = sum(组内 root 输出)
        admission 仍走 ``broker.try_reserve``（R36/R37/R38 资源治理原样保留），
        只是把「一个串行 micro-batch」作为一个执行单元计租——不再 per-root
        over-reserve CPU token（那会让 16-root batch 在 8 核机器上永远无法准入）。
        """
        roots = [dag.tasks[t] for t in mb.roots if t in dag.tasks]
        if not roots:
            return None
        try:
            from factor_engine.runtime.task_resource_contract import TaskResourceContract
        except Exception:
            return None
        peak = 0
        out = 0
        cpu = 0
        io = 0
        elapsed = 0.0
        backend_threads = 1
        for t in roots:
            c = getattr(t, "resource_contract", None)
            if c is None:
                continue
            peak = max(peak, int(getattr(c, "peak_memory_bytes", 0) or 0))
            out += int(getattr(c, "output_bytes", 0) or 0)
            cpu = max(cpu, int(getattr(c, "cpu_tokens", 0) or 0))
            io = max(io, int(getattr(c, "io_tokens", 0) or 0))
            elapsed += float(getattr(c, "predicted_elapsed_ms", 0.0) or 0.0)
            backend_threads = max(
                backend_threads, int(getattr(c, "backend_threads", 1) or 1)
            )
        return TaskResourceContract(
            predicted_elapsed_ms=max(1.0, elapsed),
            cpu_tokens=max(1, cpu),
            io_tokens=io,
            peak_memory_bytes=max(1, peak + out),
            output_bytes=max(1, out),
            spill_bytes=0,
            gil_bound=True,
            releases_gil=False,
            backend=mb.backend,
            backend_threads=backend_threads,
            shardable=False,
            uncertainty=1.0,
            estimate_basis="micro-batch",
        )

    def _make_micro_batch(
        self, backend: str, roots: list[str], work: float
    ) -> MicroBatchTask:
        return MicroBatchTask(
            roots=tuple(roots),
            backend=backend,
            same_backend=True,
            same_axis=True,
            same_source_buffers=True,
            estimated_total_work=float(work),
        )

    def _plan_micro_batches(
        self,
        ready: list[tuple[float, str]],
        dag: PhysicalFactorDAG,
        group_by_root: dict[str, Any],
        micro_batched: set[str],
    ) -> list[MicroBatchTask]:
        """R39-PERF-018：把低成本 ready ROOT 聚成 micro-batch（一个 Future 串行）。

        - 只考虑：ROOT、executable、未 fusion、未在跑、成本 < per-root cap。
        - 按 (preferred_backend, source_scope, execution_scope) 分组 → 共享
          backend + axis + source buffers。
        - 每组内按 task_id 确定性排序，切成 16–128 个 root、总 work 有界的 batch。
        不足 ``_MICRO_BATCH_MIN_ROOTS`` 的尾组不形成 micro-batch（走逐 root 路径）。
        """
        candidates: list[PhysicalFactorTask] = []
        for _priority, tid in ready:
            if tid in group_by_root or tid in micro_batched:
                continue
            task = dag.tasks.get(tid)
            if task is None or task.task_type != TASK_ROOT or not task.executable:
                continue
            if self._cert_cost_for(task) >= _MICRO_BATCH_ROOT_WORK_CAP:
                continue
            candidates.append(task)
        if len(candidates) < _MICRO_BATCH_MIN_ROOTS:
            return []
        by_scope: dict[tuple[str, str, str], list[PhysicalFactorTask]] = {}
        for task in candidates:
            key = (task.preferred_backend, task.source_scope, task.execution_scope)
            by_scope.setdefault(key, []).append(task)
        batches: list[MicroBatchTask] = []
        for key in sorted(by_scope.keys()):
            backend, _src, _exec = key
            tasks = sorted(by_scope[key], key=lambda t: t.task_id)
            chunk: list[str] = []
            chunk_work = 0.0
            for task in tasks:
                work = self._cert_cost_for(task)
                if len(chunk) >= _MICRO_BATCH_MAX_ROOTS:
                    if len(chunk) >= _MICRO_BATCH_MIN_ROOTS:
                        batches.append(self._make_micro_batch(backend, chunk, chunk_work))
                    chunk = []
                    chunk_work = 0.0
                if chunk and chunk_work + work > _MICRO_BATCH_MAX_TOTAL_WORK:
                    if len(chunk) >= _MICRO_BATCH_MIN_ROOTS:
                        batches.append(self._make_micro_batch(backend, chunk, chunk_work))
                    chunk = []
                    chunk_work = 0.0
                chunk.append(task.task_id)
                chunk_work += work
            if len(chunk) >= _MICRO_BATCH_MIN_ROOTS:
                batches.append(self._make_micro_batch(backend, chunk, chunk_work))
        return batches

    # -- run --

    def _explain(self, msg: str) -> None:
        self._explanations.append(msg)

    def _admit_and_run(
        self,
        task: PhysicalFactorTask,
        *,
        backend: Any,
        ctx: Any,
        execute_root: Callable[[PhysicalFactorTask], Any],
        materialize_shared: Callable[[str, Any], Any],
    ) -> tuple[Future | None, ReservationLease | None]:
        """admission 通过则 dispatch；返回 ``(future, lease)``，不通过返回 ``(None, None)``。

        R31-005/006：admission 返回 **lease**，task 终态统一 ``lease.release()``
        释放，保证 exactly-once（SUCCESS / FAILED / CANCELLED / TIMEOUT / BROKEN
        WORKER 都不泄漏 CPU/IO/memory reservation）。
        """
        if self._cancelled:
            self._explain(f"task={task.task_id}: not admitted (cancelled)")
            return None, None
        # R40 #96: request-scoped CancellationToken（HTTP service → FE）——cancel
        # 事件 / deadline 一到就停止新 admission（在跑 task 自然完成）。
        try:
            from factor_engine.runtime.exceptions import get_active_cancellation_token

            _token = get_active_cancellation_token()
            if _token is not None and (_token.is_cancelled or _token.expired):
                self._cancelled = True
                self._explain(
                    f"task={task.task_id}: not admitted (request cancellation token set)"
                )
                return None, None
        except Exception:  # noqa: BLE001 - token 缺失/异常时退化为既有行为
            pass
        contract = task.resource_contract
        fn = _dispatch
        if task.task_type in (TASK_SHARD, TASK_MERGE):
            # R38 P0-001：shard 子任务 / merge barrier 任务走真实分片执行器。
            fn = self._dispatch_shard_or_merge
        if contract is None:
            self._explain(f"task={task.task_id}: no contract, admit (vacuous)")
            self._pin_consumed_sids(ctx, task)
            # R38 P0-007：先记录 dispatch 时刻再 submit——极快 task 也有真实起点
            #（避免 started_at 还没写入任务就完成 → elapsed 失真 / 负值）。
            import time

            self._task_started_at[task.task_id] = time.monotonic() * 1000.0
            future = self.executor.submit(
                task.preferred_backend,
                fn, task, backend, ctx, execute_root, materialize_shared,
                prefer=self._execution_policy,
            )
            return future, None
        stage = self.broker.pressure_stage()
        if stage in {"PRESSURE_3", "PRESSURE_4", "CRITICAL"}:
            self._explain(f"task={task.task_id}: blocked by pressure_stage={stage}")
            return None, None
        lease = self.broker.try_reserve(contract, task_id=task.task_id)
        if lease is None:
            self._explain(
                f"task={task.task_id}: admission rejected "
                f"(stage={stage}, headroom={self.broker.snapshot().live_headroom})"
            )
            return None, None
        self._explain(f"task={task.task_id}: admitted (stage={stage})")
        # P0-015：消费的 CSE 共享 sid 在消费期内 pin（最后一个消费者完成时才
        # unpin+release），防止 LRU 把仍有消费者的共享结果逐出 → plan_ref miss。
        self._pin_consumed_sids(ctx, task)
        # R31-P0-007 诚实声明：FE root 执行模型（backend.execute + 共享 ctx/cache）
        # 的 payload 不可 pickle；进程执行需要 worker-local runtime（Phase D）。
        # scheduler 统一走 thread pool，并发由 ResourceBroker CPU token 约束。
        # R38 P0-007：**先**记录 dispatch 时刻（真实 duration 的起点）再 submit——
        # 极快 task 可能在 started_at 写入前完成（elapsed 失真），submit 失败再清理。
        import time

        self._task_started_at[task.task_id] = time.monotonic() * 1000.0
        try:
            future = self.executor.submit(
                task.preferred_backend,
                fn, task, backend, ctx, execute_root, materialize_shared,
                prefer=self._execution_policy,
            )
        except Exception:
            self._task_started_at.pop(task.task_id, None)
            _release_lease(lease)
            raise
        return future, lease

    def _shard_executor(self) -> Any:
        """惰性创建真实 shard 执行器（spool 目录按 run 隔离）。"""
        if self._shard_executor is None:
            import tempfile

            from factor_engine.runtime.shard_executor import ShardExecutor

            self._shard_executor = ShardExecutor(
                spool_dir=tempfile.mkdtemp(prefix="fe_r38_spool_")
            )
        return self._shard_executor

    def _dispatch_shard_or_merge(
        self,
        task: PhysicalFactorTask,
        backend: Any,
        ctx: Any,
        execute_root: Callable[[PhysicalFactorTask], Any],
        materialize_shared: Callable[[str, Any], Any],
    ) -> tuple[str, Any]:
        """R38 P0-001：shard 子任务真实执行 + merge barrier。

        模块级 ``_dispatch`` 只处理 CSE_SHARED / ROOT；这里补 SHARD / MERGE。
        """
        if task.task_type == TASK_SHARD:
            result = self._shard_executor().execute_shard(
                backend, task.node_ref, ctx, task.shard_descriptor
            )
            return task.task_id, result
        if task.task_type == TASK_MERGE:
            partials = self._shard_partials.pop(task.task_id, {})
            merged = self._shard_executor().execute_merge(
                backend, ctx, task.shard_plan, partials
            )
            return task.task_id, merged
        return _dispatch(task, backend, ctx, execute_root, materialize_shared)

    def _dynamic_concurrency_limit(self, sink: StreamingResultSink | None) -> int:
        """R33-P0-038 + R36 P0-001/002/012：显式并发上限。

        ``hard_target = min(user_max_concurrency, broker decision target_concurrency)``
        （§4）；sink backpressure 再降（§40：>0.70 reduce，>0.90 stop non-critical）。
        ResourceBroker 的 ``try_reserve`` 仍做精确 memory/CPU admission。
        """
        limit = self.max_concurrency
        if limit is None or limit <= 0:
            limit = 2 ** 31
        # R36 P0-002：broker 算出的动态目标必须被消费（推荐值真正生效）。
        # P0-017：``target_concurrency``（AIMD 控制的**任务数**上限）才是并发闸；
        # ``target_cpu_tokens`` 是 CPU 线程预算，由 admission 内的 token-sum 门消费
        #（一个 8-thread DuckDB task ≠ 八个单线程 Numba task）。
        decision = getattr(self, "_last_decision", None)
        broker_limit = 1
        if decision is not None:
            broker_limit = max(1, int(decision.target_concurrency))
        else:
            try:
                broker_limit = max(1, int(self.broker.cpu_budget()))
            except Exception:
                broker_limit = max(1, int(getattr(self.broker, "hard_cpu_slots", 1) or 1))
        limit = min(limit, broker_limit)
        if sink is not None:
            try:
                # R33-P0-046 + R36 P0-012：backpressure 显式 0.70 / 0.90 双档。
                ratio = getattr(sink, "backpressure_ratio", None)
                if ratio is None:
                    ratio = sink.queue.backpressure_ratio
                ratio = float(ratio)
                if ratio >= 0.90:
                    limit = max(1, limit // 4)
                elif ratio >= 0.70:
                    limit = max(1, int(limit * 0.5))
            except Exception:
                pass
        return max(1, limit)

    def _decision_cpu_token_budget(self) -> int | None:
        """P0-017：当前 decision 的 CPU token 总预算（admission 内 token-sum 门）。"""
        decision = getattr(self, "_last_decision", None)
        if decision is None:
            return None
        try:
            return max(1, int(decision.target_cpu_tokens))
        except Exception:
            return None

    def _running_cpu_tokens(
        self,
        futures: dict[str, Future],
        dag: Any,
        micro_batch_contracts: dict[str, Any] | None = None,
    ) -> int:
        """P0-017：在跑 task 的 CPU token 总和（fusion group 走 lease 记账，跳过）。

        R39-PERF-018：micro-batch future 的 key 是 ``microbatch:N``（不在
        dag.tasks）——用其合成契约（串行执行的真实 footprint，cpu_tokens =
        max(组内 root)）计入 token-sum 门。
        """
        tasks = getattr(dag, "tasks", dag)  # PhysicalFactorDAG（.tasks）或裸 dict
        total = 0
        for key in futures:
            task = tasks.get(key)
            if task is None:
                contract = (micro_batch_contracts or {}).get(key)
            else:
                contract = task.resource_contract
            if contract is None:
                continue
            try:
                total += max(0, int(getattr(contract, "cpu_tokens", 0) or 0))
            except (TypeError, ValueError):
                continue
        return total

    def _execute_read_waves(
        self,
        plan: SchedulerPlan,
        dag: PhysicalFactorDAG,
        committed: set[str],
        remaining: set[str],
        ctx: Any,
        *,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds: Any = None,
        defer_on_pressure: bool = False,
    ) -> int:
        """R33-P0-016：把 read wave 真正接进 scheduler 主路径。

        对每个 wave：当它覆盖的 SOURCE_SCAN task 全部 ready（inputs committed）
        时执行一次 scan（``SourceWaveExecutor.execute_wave`` → prefetch union 列
        → 共享列缓存），并把 SOURCE_SCAN task 标记 committed（其 IO 已在 wave
        中完成），BufferRef 写入结果表。executor / refs 为实例状态（幂等：同一
        wave_id 只执行一次）。wave 失败记录 event 并向上传播；失败的
        SOURCE_SCAN 不会标记 committed。返回本次提交的 SOURCE_SCAN task 数。
        """
        waves = list(getattr(plan, "read_waves", ReadWavePlan()).waves or [])
        if not waves:
            return 0
        source = getattr(ctx, "data_source", None)
        if source is None:
            return 0
        if self._wave_executor is None:
            self._wave_executor = SourceWaveExecutor(source, ctx)
        executor = self._wave_executor
        refs = self._wave_refs
        committed_count = 0
        for wave in waves:
            wid = wave.wave_id
            if wid in refs:
                continue
            tids = [
                t
                for t in (getattr(wave, "source_tasks", ()) or wave.task_ids)
                if t in dag.tasks
            ]
            if not tids:
                continue
            if all(t in committed for t in tids):
                continue
            if any(
                not all(p in committed for p in dag.tasks[t].inputs)
                for t in tids
            ):
                continue
            from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
            from factor_engine.runtime.resource_errors import ResourceBudgetExceeded
            wave_bytes = int(getattr(wave, "estimated_memory_bytes", 0) or 0)
            wave_lease = self.broker.acquire_memory(
                MemoryLeaseKind.READ_WAVE, wave_bytes,
                lease_id=f"read-wave:{id(self)}:{wid}",
            )
            if wave_lease is None:
                capacity_fn = getattr(self.broker, "execution_budget", None)
                current_capacity = (
                    int(capacity_fn()) if callable(capacity_fn) else wave_bytes
                )
                # Current execution budget can recover as host pressure falls;
                # only the configured hard limit proves an atomic wave can
                # never fit. Planner admission already rejects waves above its
                # own calibrated budget.
                hard_capacity = int(
                    getattr(self.broker, "hard_memory_limit", current_capacity)
                    or current_capacity
                )
                if wave_bytes > hard_capacity:
                    raise ResourceBudgetExceeded(
                        f"read wave {wid} requires {wave_bytes} bytes; "
                        f"atomic broker capacity is {hard_capacity}"
                    )
                if defer_on_pressure:
                    # Existing waves/tasks can complete and return residency.
                    # The outer scheduler has a finite stuck/deadline policy.
                    self._wave_summary["events"].append(
                        f"deferred:{wid}:{wave_bytes}@{current_capacity}"
                    )
                    return committed_count
                raise ResourceBudgetExceeded(
                    f"read wave {wid} requires {wave_bytes} bytes; "
                    "broker admission denied: temporarily busy and no concurrent consumer can release it"
                )
            try:
                ref = executor.execute_wave(wave, consumer_ids=tids)
            except BaseException:
                wave_lease.release()
                raise
            # Charge until every reachable pandas/NumPy physical owner dies;
            # mappings and zero-copy views can outlive the BufferRef wrapper.
            _retain_wave_lease(ref, wave_lease)
            refs[wid] = ref
            pending_consumers = set(getattr(wave, "consumer_tasks", ()) or ())
            # Keep the physical buffer through every reachable terminal. Planner
            # consumer annotations may omit a root separated by planning views.
            queue = list(tids)
            reachable: set[str] = set()
            while queue:
                predecessor = queue.pop()
                for consumer in getattr(dag.tasks[predecessor], "consumers", ()) or ():
                    if consumer in reachable or consumer not in dag.tasks:
                        continue
                    reachable.add(consumer)
                    queue.append(consumer)
            reachable_terminals = {
                task_id for task_id in reachable
                if dag.tasks[task_id].task_type in (TASK_ROOT, TASK_MERGE)
            }
            pending_consumers.update(reachable_terminals)
            overlapping_wave = any(
                other.wave_id != wid
                and bool(set(getattr(other, "columns", ()) or ()) & set(wave.columns))
                for other in waves
            )
            if overlapping_wave:
                pending_consumers.update(
                    task_id for task_id, candidate in dag.tasks.items()
                    if candidate.task_type in (TASK_ROOT, TASK_MERGE)
                )
            if not reachable_terminals:
                # Missing consumer annotations must extend, never shorten,
                # physical residency. Conservatively retain through all
                # terminal tasks in this physical DAG.
                pending_consumers.update(
                    task_id for task_id, candidate in dag.tasks.items()
                    if candidate.task_type in (TASK_ROOT, TASK_MERGE)
                )
            self._wave_pending_consumers[wid] = pending_consumers
            self._wave_source_tasks[wid] = tuple(tids)
            # R33-P0-017：input DQ 按 wave 在 scan 后做（一次 scan 同时产生 data
            # buffer + DQ stats，不另扫一遍）。
            if input_dq_check and ref is not None and wave.columns:
                try:
                    from factor_engine.runtime.input_dq import assert_input_dq
                    from factor_engine.runtime.input_dq import (
                        adjust_input_dq_thresholds_from_stats,
                        load_dataset_stats_for_source,
                    )

                    stats = load_dataset_stats_for_source(source)
                    thresholds = adjust_input_dq_thresholds_from_stats(
                        input_dq_thresholds, stats, list(wave.columns)
                    )
                    report = assert_input_dq(
                        source,
                        list(wave.columns),
                        prefetched=(ref.meta or {}).get("loaded_columns"),
                        raise_on_fail=input_dq_strict,
                        thresholds=thresholds,
                    )
                    self._input_dq_reports.append(report)
                except Exception:
                    if input_dq_strict:
                        raise
            for t in tids:
                task = dag.tasks[t]
                if task.task_type == TASK_SOURCE_SCAN:
                    self._buffer_results[t] = ref
                    committed.add(t)
                    remaining.discard(t)
                    self._done += 1
                    self._record_timing(t, task)
                    # R38 P0-041：wave 已覆盖 → 后续重排不再重建该 task 的 wave。
                    self._wave_covered_tasks.add(t)
                    committed_count += 1
        self._wave_summary = executor.summary()
        return committed_count

    def _maybe_repartition_waves(
        self,
        plan: SchedulerPlan,
        dag: PhysicalFactorDAG,
        committed: set[str],
    ) -> None:
        """R38 P0-041（§16）：运行中 wave 预算显著缩小时，对**未执行** SOURCE_SCAN
        重排 wave（JIT repartition）。

        - 预算缩小 >40% 且有未执行 source scan 才重排（避免频繁重建）；
        - 只对未执行 source tasks 重排（已执行 wave 的 ref 保留，不再重复扫）；
        - 新 wave 的 wave_id 重新编号（避免与已执行 wave id 冲突被跳过）。
        """
        decision = getattr(self, "_last_decision", None)
        if decision is None:
            return
        budget = max(1, int(decision.read_wave_bytes))
        last = getattr(self, "_last_wave_budget", budget)
        if budget >= last * 0.6:
            self._last_wave_budget = budget
            return
        unexecuted = [
            t
            for t in dag.tasks.values()
            if t.task_type == TASK_SOURCE_SCAN
            and t.task_id not in committed
            and t.task_id not in getattr(self, "_wave_covered_tasks", set())
        ]
        if not unexecuted:
            self._last_wave_budget = budget
            return
        from types import SimpleNamespace

        from factor_engine.planner.read_wave_planner import build_waves_from_dag

        sub = SimpleNamespace(tasks={t.task_id: t for t in unexecuted})
        new_waves = build_waves_from_dag(sub, wave_memory_budget=budget)
        if not new_waves.waves:
            self._last_wave_budget = budget
            return
        # 重新编号（避免与已执行 wave id 冲突）。
        base = max(getattr(self, "_wave_refs", {}).keys(), default=-1) + 1
        from dataclasses import replace

        renumbered = [
            replace(w, wave_id=base + i) for i, w in enumerate(new_waves.waves)
        ]
        new_waves.waves = renumbered
        plan.read_waves = new_waves
        self._last_wave_budget = budget
        self._wave_summary["events"].append(
            f"repartition:{len(new_waves.waves)}waves@{budget}"
        )
        self._explain(
            f"R38_DYNAMIC_READ_WAVE_SHRINK: wave budget {last}->{budget}, "
            f"repartitioned {len(unexecuted)} unexecuted source tasks into "
            f"{len(new_waves.waves)} waves"
        )

    def _admit_virtual(
        self,
        dag: PhysicalFactorDAG,
        remaining: set[str],
        committed: set[str],
        futures: dict[str, Future],
    ) -> int:
        """R33-P0-008：barrier 规划视图（executable=False）自动提交——不占 lease、
        不 submit future（控制面零开销，资源零预占）。"""
        admitted = 0
        for tid in list(remaining):
            if tid in futures or tid in committed:
                continue
            task = dag.tasks[tid]
            if not all(p in committed for p in task.inputs):
                continue
            if task.executable:
                continue
            committed.add(tid)
            remaining.discard(tid)
            self._done += 1
            self._record_timing(tid, task)
            admitted += 1
        return admitted

    def run(
        self,
        plan: SchedulerPlan,
        *,
        backend: Any,
        ctx: Any,
        execute_root: Callable[[PhysicalFactorTask], Any] | None = None,
        materialize_shared: Callable[[str, Any], Any] | None = None,
        sink: StreamingResultSink | None = None,
        result_handler: Callable[[str, Any], None] | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds: Any = None,
    ) -> dict[str, Any]:
        """Run with exception-safe leases; running jobs keep reservations until done."""
        self._cleanup_leases = {}
        self._processing_lease = None
        failure: BaseException | None = None
        try:
            return self._run_impl(
                plan, backend=backend, ctx=ctx, execute_root=execute_root,
                materialize_shared=materialize_shared, sink=sink,
                result_handler=result_handler, input_dq_check=input_dq_check,
                input_dq_strict=input_dq_strict, input_dq_thresholds=input_dq_thresholds,
            )
        except BaseException as exc:
            failure = exc
            raise
        finally:
            # The current result's future has completed, even if its sink failed.
            _release_lease(self._processing_lease)
            self._processing_lease = None
            for future, lease in tuple(self._cleanup_leases.items()):
                # cancel() affects queued jobs only. Running jobs retain their
                # reservation until the completion callback, including failures.
                future.add_done_callback(lambda done, lease=lease: _release_lease(lease))
                future.cancel()
            self._cleanup_leases.clear()
            if failure is not None:
                target_sink = sink if sink is not None else self.sink
                if target_sink is not None:
                    try:
                        # Finish only already accepted results; no new compute
                        # admission or replay of failed receipts is permitted.
                        target_sink.finish()
                    except BaseException as cleanup_error:
                        if hasattr(failure, "add_note"):
                            failure.add_note(f"sink cleanup failed: {cleanup_error}")

    def _run_impl(
        self,
        plan: SchedulerPlan,
        *,
        backend: Any,
        ctx: Any,
        execute_root: Callable[[PhysicalFactorTask], Any] | None = None,
        materialize_shared: Callable[[str, Any], Any] | None = None,
        sink: StreamingResultSink | None = None,
        result_handler: Callable[[str, Any], None] | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds: Any = None,
    ) -> dict[str, Any]:
        """执行 DAG：read waves + ready queue + admission + FIRST_COMPLETED。

        R33 升级：
            - P0-016 read wave 在 SOURCE_SCAN ready 时真实执行（prefetch union 列
              一次 → BufferRef 进结果表 → consumers 复用共享缓存）。
            - P0-008 barrier 规划视图自动提交（不占 resource lease）。
            - P0-038 ``running < dynamic_concurrency_limit`` 显式并发闸。
            - P0-039 ready set 按 priority 排序主导 admission。
            - P0-040 ``wait(FIRST_COMPLETED)`` 事件驱动（不再固定 50ms 轮询）。
        """
        # 本次 run 的 ctx（shard 时间窗 / 真实交易日历推导用，P0-009）。
        self._run_ctx = ctx
        from factor_engine.runtime.batch_service import _execute_root_with_path, _materialize_shared_subplan

        def _default_execute_root(task: PhysicalFactorTask) -> Any:
            # R31-P0-002：lowerer 生成的 ROOT task 的 node_ref 是裸 plan；
            # 旧 DAGPlan 路径 node_ref 是 FactorPlan（带 .root）。
            node = getattr(task, "node_ref", None)
            plan = getattr(node, "root", node)
            result, _path = _execute_root_with_path(
                backend, plan, ctx, run_mode=getattr(ctx, "run_mode", None),
                factor_name=task.factor_name,
                task_id=task.task_id,
                factor_id=task.factor_name,
            )
            return result

        execute_root = execute_root or _default_execute_root
        materialize_shared = materialize_shared or (
            lambda sid, node: _materialize_shared_subplan(backend, node, ctx, sid)
        )
        self._input_dq_reports = []
        sink = sink or self.sink
        # 接受 SchedulerPlan（含 physical_dag）或裸 PhysicalFactorDAG。
        dag = getattr(plan, "physical_dag", plan)

        pending: list[str] = dag.topological_order()
        committed: set[str] = set()
        futures: dict[str, Future] = {}
        # R31-005: future → task_id / lease 双向绑定（失败/取消也能知道归属并释放）。
        future_to_task_id: dict[Future, str] = {}
        future_leases: dict[Future, ReservationLease] = {}
        self._cleanup_leases = future_leases
        # R31-P0-022：fusion group 登记（root task_id → group）。
        fusion_groups = list(getattr(plan, "fusion_groups", []) or [])
        group_by_root: dict[str, Any] = {}
        fusion_done: set[str] = set()
        future_group_roots: dict[Future, tuple[str, ...]] = {}
        for _g in fusion_groups:
            for _tid in _g.roots:
                group_by_root[_tid] = _g
        remaining = set(pending)
        # R39-PERF-018/019：本 run 的 micro-batch 状态 + 结构化性能计数。
        micro_batched: set[str] = set()
        future_micro_batch_roots: dict[Future, tuple[str, ...]] = {}
        #: microbatch key -> 合成资源契约（token-sum 门用其真实 footprint）。
        micro_batch_contracts: dict[str, Any] = {}
        self._perf_metrics = {
            "future_count": 0,
            "factor_count": sum(
                1
                for _t in dag.tasks.values()
                if getattr(_t, "task_type", "") == TASK_ROOT
            ),
            "micro_batch_task_count": 0,
            "micro_batch_root_count": 0,
            "scheduler_wait_polling_count": 0,
        }
        # R33-P0-016：read wave 真实执行 trace。
        self._wave_summary = {"waves_planned": 0, "waves_executed": 0, "events": []}
        self._scheduler_stats: dict[str, Any] = {}
        virtual_done = 0
        real_done = 0
        wave_scan_done = 0
        admitted_this_round = 0
        no_progress_rounds = 0
        while remaining or futures:
            # R31-069/P1-040：取消 → 不再 admit 新 task；无在跑任务时提前结束。
            if self._cancelled and not futures:
                self._explain(
                    f"CANCELLED: stopping with {len(remaining)} pending tasks not admitted"
                )
                break
            # R40 #96: request-scoped CancellationToken —— cancel/deadline 一到，
            # 若无在跑 future 则提前结束（有在跑 future 时 _admit_and_run 拒新 admission）。
            try:
                from factor_engine.runtime.exceptions import get_active_cancellation_token

                _req_token = get_active_cancellation_token()
                if (
                    _req_token is not None
                    and (_req_token.is_cancelled or _req_token.expired)
                    and not futures
                ):
                    self._cancelled = True
                    self._explain(
                        f"REQUEST_CANCELLED: stopping with {len(remaining)} "
                        "pending tasks not admitted"
                    )
                    break
            except Exception:  # noqa: BLE001
                pass
                self._explain(
                    f"CANCELLED: stopping with {len(remaining)} pending tasks not admitted"
                )
                break
            # P0-023：writer fatal → 立即停止 admission（不等 finish() 才暴露）——
            # compute 不再往已死的 writer 队列塞结果。
            if sink is not None and getattr(sink, "fatal_error", None) is not None:
                self._explain(
                    f"SINK_FATAL: writer failed ({type(sink.fatal_error).__name__}); "
                    "stopping admission"
                )
                break
            # R36 P0-002/003：每个 control tick 消费 broker 的 ResourceDecision——
            # 压力升高 AIMD fast down、压力解除稳定后 slow up、动态 budgets
            # （wave/block/sink）全部在此刷新（§301 伪代码 scheduler.set_targets）。
            decision = self.broker.resource_decision(
                job_memory_lease_bytes=None,
                sink_backpressure=float(sink.backpressure_ratio) if sink is not None else 0.0,
            )
            self._last_decision = decision
            # R36 P0-010：dynamic read wave（§35/36）——运行中不拆已执行 wave，
            # 只影响后续 plan / 新 wave。
            self.wave_memory_budget = int(decision.read_wave_bytes)
            # R38 P0-043（§17）：sink 队列预算随压力弹性调整（缩容不丢 item，
            # producer 阻塞等消费降到新 target 以下）。
            if sink is not None:
                try:
                    sink.set_target_bytes(int(decision.result_queue_bytes))
                except Exception:
                    pass
            # R38 P0-041（§16）：wave 预算显著缩小时，对未执行 SOURCE_SCAN 重排
            # wave（运行中压力变大 → 已生成的 read waves 重新拆小）。
            try:
                self._maybe_repartition_waves(plan, dag, committed)
            except Exception:
                pass
            # 0) read waves：SOURCE_SCAN ready → 真实 scan（R33-P0-016）。
            wave_scan_done += self._execute_read_waves(
                plan, dag, committed, remaining, ctx,
                input_dq_check=input_dq_check,
                input_dq_strict=input_dq_strict,
                input_dq_thresholds=input_dq_thresholds,
                defer_on_pressure=True,
            )
            # 0.5) barrier 规划视图自动提交（R33-P0-008，不占 lease）。
            virtual_done += self._admit_virtual(dag, remaining, committed, futures)
            # 1) fusion group admission：组内全部 root 就绪 → 一次 native query。
            for group in fusion_groups:
                gkey = f"fusion:{group.group_id}"
                if gkey in futures or gkey in fusion_done:
                    continue
                roots = [t for t in group.roots if t in dag.tasks]
                if not roots or any(t in futures or t in committed for t in roots):
                    continue
                if not all(all(p in committed for p in dag.tasks[t].inputs) for t in roots):
                    continue
                if len(futures) >= self._dynamic_concurrency_limit(sink):
                    self._explain(
                        f"fusion group {group.group_id}: deferred (concurrency limit)"
                    )
                    continue
                leases: list[ReservationLease] = []
                ok = True
                for t in roots:
                    contract = dag.tasks[t].resource_contract
                    if contract is not None:
                        lease = self.broker.try_reserve(contract, task_id=t)
                        if lease is None:
                            ok = False
                            break
                        leases.append(lease)
                if not ok:
                    for lease in leases:
                        lease.release()
                    continue
                try:
                    future = self.executor.submit(
                        group.backend,
                        _dispatch_fusion, group, dag.tasks, backend, ctx, execute_root,
                        prefer=self._execution_policy,
                    )
                except BaseException:
                    _release_lease(leases)
                    raise
                futures[gkey] = future
                future_to_task_id[future] = gkey
                future_leases[future] = leases
                future_group_roots[future] = tuple(roots)
                for t in roots:
                    remaining.discard(t)
                admitted_this_round += 1
                self._perf_metrics["future_count"] += 1
                self._explain(
                    f"fusion group {group.group_id}: admitted {len(roots)} roots "
                    f"(backend={group.backend})"
                )
            # 2) admission：predecessor 全 committed 的 executable task，按 priority
            #    排序主导（R33-P0-039）。显式并发上限（R33-P0-038）。
            # 2.0) R38 P1-065：P99 已超 SafeEnvelope 的 ready ROOT 在首次 admission
            #     前就真实 pre-shard（不等 200 轮 no-progress）。
            self._maybe_preshard_oversized(dag, remaining)
            concurrency_limit = self._dynamic_concurrency_limit(sink)
            cpu_token_budget = self._decision_cpu_token_budget()
            ready: list[tuple[float, str]] = []
            for tid in remaining:
                if tid in futures or tid in committed or tid in group_by_root:
                    continue
                task = dag.tasks[tid]
                if not task.executable:
                    continue
                if not all(p in committed for p in task.inputs):
                    continue
                priority = self._priority_score(dag, task)
                ready.append((priority, tid))
            ready.sort(reverse=True)
            # 2.1) R39-PERF-018：低成本 ready ROOT 合并成 micro-batch（scheduler
            #      dispatch coalescing，不是 native fusion）。先 micro-batch（共享
            #      backend + axis + source buffers 的 cheap root 一组一个 Future），
            #      剩余 root 再逐 root 走原有 admission。admission 仍走 broker
            #      try_reserve（R36/R37/R38 资源治理原样保留），但 micro-batch 是
            #      串行执行单元 → 用合成契约（真实 footprint）计一份租约。
            if not self._cancelled:
                try:
                    stage = self.broker.pressure_stage()
                except Exception:
                    stage = ""
                if stage not in {"PRESSURE_3", "PRESSURE_4", "CRITICAL"} and \
                        os.environ.get("FACTOR_ENGINE_MICRO_BATCH") != "0":
                    micro_batches = self._plan_micro_batches(
                        ready, dag, group_by_root, micro_batched
                    )
                    for mb in micro_batches:
                        if len(futures) >= concurrency_limit:
                            self._explain(
                                f"microbatch: deferred (concurrency_limit={concurrency_limit})"
                            )
                            break
                        mb_key = f"microbatch:{self._next_micro_batch_id()}"
                        mb_contract = self._micro_batch_contract(mb, dag)
                        lease = None
                        if mb_contract is not None:
                            lease = self.broker.try_reserve(
                                mb_contract, task_id=mb_key
                            )
                            if lease is None:
                                self._explain(
                                    f"microbatch {mb_key}: deferred (admission)"
                                )
                                continue
                        # P0-017：micro-batch 合成契约的 CPU token 也受
                        # ``target_cpu_tokens`` 约束（token-sum 门）。
                        if cpu_token_budget is not None:
                            mb_tokens = (
                                int(getattr(mb_contract, "cpu_tokens", 0) or 0)
                                if mb_contract is not None
                                else 0
                            )
                            running_tokens = self._running_cpu_tokens(
                                futures, dag, micro_batch_contracts
                            )
                            if running_tokens + mb_tokens > cpu_token_budget:
                                _release_lease(lease)
                                self._explain(
                                    f"microbatch {mb_key}: deferred (cpu_tokens "
                                    f"{running_tokens}+{mb_tokens} > "
                                    f"target_cpu_tokens={cpu_token_budget})"
                                )
                                continue
                        # P0-015：消费的 CSE 共享 sid 在消费期内 pin（同逐 root 路径）。
                        for tid in mb.roots:
                            _t = dag.tasks.get(tid)
                            if _t is not None:
                                self._pin_consumed_sids(ctx, _t)
                        import time

                        for tid in mb.roots:
                            self._task_started_at[tid] = time.monotonic() * 1000.0
                        try:
                            future = self.executor.submit(
                                mb.backend, dispatch_micro_batch, mb.roots, dag.tasks,
                                execute_root, prefer=self._execution_policy,
                            )
                        except BaseException:
                            _release_lease(lease)
                            raise
                        futures[mb_key] = future
                        future_to_task_id[future] = mb_key
                        future_micro_batch_roots[future] = mb.roots
                        if mb_contract is not None:
                            future_leases[future] = lease
                            micro_batch_contracts[mb_key] = mb_contract
                        for tid in mb.roots:
                            micro_batched.add(tid)
                            remaining.discard(tid)
                        admitted_this_round += 1
                        self._perf_metrics["future_count"] += 1
                        self._perf_metrics["micro_batch_task_count"] += 1
                        self._perf_metrics["micro_batch_root_count"] += len(mb.roots)
                        self._explain(
                            f"microbatch {mb_key}: {len(mb.roots)} roots "
                            f"(backend={mb.backend}, work={mb.estimated_total_work:.0f})"
                        )
            # 2.2) 逐 root admission（与 R33 相同；跳过已在 micro-batch 中的 root）。
            for _priority, tid in ready:
                if tid in futures or tid in committed or tid in micro_batched:
                    continue
                if len(futures) >= concurrency_limit:
                    self._explain(f"task={tid}: deferred (concurrency_limit={concurrency_limit})")
                    break
                # P0-017：CPU token 总和也受 ``target_cpu_tokens`` 约束——任务数达标
                # 但 token 总和已满（8-thread DuckDB 占 8 token）时不再 admit。
                task = dag.tasks[tid]
                if cpu_token_budget is not None:
                    token_cost = int(getattr(task.resource_contract, "cpu_tokens", 0) or 0)
                    running_tokens = self._running_cpu_tokens(
                        futures, dag, micro_batch_contracts
                    )
                    if running_tokens + token_cost > cpu_token_budget:
                        if not futures and running_tokens == 0:
                            decision = self.broker.request_minimum_cpu_width(
                                task.resource_contract, decision=self._last_decision)
                            self._last_decision = decision
                            cpu_token_budget = int(decision.target_cpu_tokens)
                            concurrency_limit = 1
                            self._explain(f"task={tid}: broker minimum CPU width admitted={cpu_token_budget}")
                        else:
                            self._explain(
                                f"task={tid}: deferred (cpu_tokens {running_tokens}+"
                                f"{token_cost} > target_cpu_tokens={cpu_token_budget})"
                            )
                            continue
                future, lease = self._admit_and_run(
                    task,
                    backend=backend,
                    ctx=ctx,
                    execute_root=execute_root,
                    materialize_shared=materialize_shared,
                )
                if future is not None:
                    futures[tid] = future
                    future_to_task_id[future] = tid
                    if lease is not None:
                        future_leases[future] = lease
                    remaining.discard(tid)
                    admitted_this_round += 1
                    self._perf_metrics["future_count"] += 1
            # 3) FIRST_COMPLETED：有 future 时等第一个完成（R33-P0-040 事件驱动）。
            # R39-PERF-019：正常路径不依赖周期 polling——wait 事件驱动返回非空
            # done；只有 wait 因 timeout 兜底返回空（watchdog/deadlock fallback）
            # 才计一次 ``scheduler_wait_polling_count``（happy path 应为 0）。
            done: set[Future] = set()
            if futures:
                done = wait(list(futures.values()), timeout=_EVENT_WAIT_TIMEOUT_S,
                            return_when=FIRST_COMPLETED)[0]
                if not done:
                    self._perf_metrics["scheduler_wait_polling_count"] += 1
            for future in done:
                key = future_to_task_id.pop(future, None)
                if key is None:
                    continue
                lease = future_leases.pop(future, None)
                self._processing_lease = lease
                futures.pop(key, None)
                # R39-PERF-018：micro-batch future 完成 → 逐 root 处理（结果 /
                # 失败按 root 独立；成功 root 照常提交，失败 root 走原有 retry/raise）。
                # 整个 micro-batch 只持有一份合成租约（``lease``），在组内全部
                # root 处理完一次性释放（exactly once）。
                mb_roots = future_micro_batch_roots.pop(future, None)
                if mb_roots is not None:
                    micro_batch_contracts.pop(key, None)
                    try:
                        results_by_root, failures_by_root = future.result(timeout=1.0)
                    except Exception as exc:  # noqa: BLE001
                        _release_lease(lease)
                        kind = classify_error(exc)
                        for _tid in mb_roots:
                            micro_batched.discard(_tid)
                            if _tid not in dag.tasks:
                                continue
                            if kind == ERROR_OOM:
                                if self._handle_oom(_tid, exc, dag, remaining):
                                    continue
                                raise
                            retries = self._retries_remaining.get(_tid, _inner_retry_default())
                            if kind != ERROR_TRANSIENT or retries <= 0:
                                raise
                            self._retries_remaining[_tid] = retries - 1
                            remaining.add(_tid)
                        self._explain(
                            f"microbatch {key}: FUTURE_FAILED "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                    for _tid in mb_roots:
                        micro_batched.discard(_tid)
                        # P0-010 attempt 隔离：OOM replan 后旧 root 已不在 dag.tasks
                        # → 结果/异常都不触碰新 DAG 状态（租约在组末统一释放）。
                        if _tid not in dag.tasks:
                            continue
                        if _tid in failures_by_root:
                            _exc = failures_by_root[_tid]
                            _kind = classify_error(_exc)
                            if _kind == ERROR_OOM:
                                if self._handle_oom(_tid, _exc, dag, remaining):
                                    continue
                                self._explain(
                                    f"task={_tid}: OOM_NOT_REPLANNABLE "
                                    f"{type(_exc).__name__}: {_exc}"
                                )
                                raise _exc
                            _retries = self._retries_remaining.get(_tid, _inner_retry_default())
                            if _retries > 0 and _kind == ERROR_TRANSIENT:
                                self._retries_remaining[_tid] = _retries - 1
                                self._explain(
                                    f"task={_tid}: FAILED {type(_exc).__name__}: "
                                    f"{_exc} (class={_kind}, retrying, "
                                    f"{_retries - 1} left)"
                                )
                                remaining.add(_tid)
                                continue
                            self._explain(
                                f"task={_tid}: FAILED_FATAL {type(_exc).__name__}: "
                                f"{_exc} (class={_kind})"
                            )
                            raise _exc
                        committed.add(_tid)
                        real_done += 1
                        self._record_timing(_tid, dag.tasks[_tid])
                        self._record_task_calibration(
                            _tid, dag.tasks[_tid], results_by_root[_tid]
                        )
                        _tt = dag.tasks[_tid].task_type
                        if _tt in (TASK_ROOT, TASK_MERGE):
                            if sink is not None:
                                if not sink.submit(
                                    dag.tasks[_tid].factor_name, results_by_root[_tid]
                                ):
                                    self._explain(
                                        f"sink.submit FALSE (microbatch) for "
                                        f"{dag.tasks[_tid].factor_name} — writer "
                                        "fatal; abort generation"
                                    )
                                    raise RuntimeError(
                                        f"sink.submit returned False for "
                                        f"{dag.tasks[_tid].factor_name} "
                                        "(R39-PERF-018: writer failure must abort)"
                                    )
                            if result_handler is not None:
                                result_handler(
                                    dag.tasks[_tid].factor_name, results_by_root[_tid]
                                )
                            elif sink is None:
                                self._results[dag.tasks[_tid].factor_name] = (
                                    results_by_root[_tid]
                                )
                        if _tt in (TASK_ROOT, TASK_MERGE):
                            self._release_consumed(dag, _tid, ctx)
                    _release_lease(lease)
                    continue
                # P0-010 attempt 隔离：OOM replan 后旧 attempt 的 in-flight future
                # 完成时其 task id 已不在 dag.tasks（被 ``:attempt:N:`` 新 id 取代）→
                # 直接丢弃（结果/异常都不触碰新 DAG 状态，绝不写入新 partial map）。
                if key not in dag.tasks and not key.startswith("fusion:"):
                    _release_lease(lease)
                    continue
                try:
                    _ret_key, result = future.result(timeout=1.0)
                except Exception as exc:  # noqa: BLE001
                    _release_lease(lease)
                    kind = classify_error(exc)
                    # R38 P0-004/005（§5）：OOM → smaller-shape replan，然后重试。
                    if kind == ERROR_OOM:
                        if self._handle_oom(key, exc, dag, remaining):
                            continue
                        self._explain(
                            f"task={key}: OOM_NOT_REPLANNABLE {type(exc).__name__}: {exc}"
                        )
                        raise
                    retries = self._retries_remaining.get(key, _inner_retry_default())
                    # R31-006：只有 transient（或未知=保守单次）自动 retry；
                    # permanent（PIT/semantic/参数/不支持算子/确定性错误）不重跑。
                    if retries > 0 and kind == ERROR_TRANSIENT:
                        self._retries_remaining[key] = retries - 1
                        self._explain(
                            f"task={key}: FAILED {type(exc).__name__}: {exc} "
                            f"(class={kind}, retrying, {retries - 1} left)"
                        )
                        if key.startswith("fusion:"):
                            # fusion 失败 → 回退逐 root 重新调度（honest fallback）。
                            gr = future_group_roots.pop(future, None)
                            if gr is not None:
                                for _t in gr:
                                    remaining.add(_t)
                        else:
                            remaining.add(key)
                        continue
                    # R27-042：不直接 kill 运行中 task；永久失败如实抛出。
                    self._explain(
                        f"task={key}: FAILED_FATAL {type(exc).__name__}: {exc} "
                        f"(class={kind})"
                    )
                    raise
                _release_lease(lease)
                # fusion group 完成：逐 root 处理结果（honest per-root fallback
                # 结果也已进入 dict）。
                group_roots = future_group_roots.pop(future, None)
                if group_roots is not None:
                    fusion_done.add(key)
                    results_by_root = result if isinstance(result, dict) else {}
                    for _tid in group_roots:
                        committed.add(_tid)
                        real_done += 1
                        # P0-011：先取 ``_res`` 再用——旧顺序在 ``_record_task_calibration``
                        # 时 ``_res`` 还没赋值（首轮 UnboundLocalError / 上一轮残留值）。
                        _res = results_by_root.get(_tid)
                        self._record_timing(_tid, dag.tasks[_tid])
                        self._record_task_calibration(_tid, dag.tasks[_tid], _res)
                        if dag.tasks[_tid].task_type == TASK_ROOT:
                            if sink is not None:
                                if not sink.submit(dag.tasks[_tid].factor_name, _res):
                                    self._explain(
                                        f"sink.submit FALSE (fusion) for "
                                        f"{dag.tasks[_tid].factor_name} — writer fatal; "
                                        "abort generation"
                                    )
                                    raise RuntimeError(
                                        f"sink.submit returned False for "
                                        f"{dag.tasks[_tid].factor_name} (P0-023: writer "
                                        "failure must abort the generation)"
                                    )
                            if result_handler is not None:
                                result_handler(dag.tasks[_tid].factor_name, _res)
                            elif sink is None:
                                self._results[dag.tasks[_tid].factor_name] = _res
                        self._release_consumed(dag, _tid, ctx)
                    continue
                committed.add(key)
                real_done += 1
                self._record_timing(key, dag.tasks[key])
                self._record_task_calibration(key, dag.tasks[key], result)
                task_type = dag.tasks[key].task_type
                if task_type == TASK_SHARD:
                    # R38 P0-001：shard 子任务完成 → 结果写入 merge 的 partials
                    #（大结果 spool 到磁盘，merge 前不全部常驻内存，§P0-003）。
                    merge_id = key.rsplit(":shard:", 1)[0] + ":merge"
                    # R39-PERF-027：spool 阈值自适应（live_headroom / writer 队列
                    # 余量 / 盘吞吐 / merge 邻近）。全部信号缺失时 policy 安全回退
                    # 固定 512MiB（与原行为一致），绝不比旧行为更激进地 spool。
                    try:
                        from factor_engine.runtime.spool_policy import default_spool_policy_factory

                        policy = default_spool_policy_factory(
                            broker=self.broker, sink=self.sink
                        )
                        stored = self._shard_executor().spool_or_keep(
                            result, policy=policy
                        )
                    except Exception:  # noqa: BLE001
                        stored = self._shard_executor().spool_or_keep(result)
                    self._shard_partials.setdefault(merge_id, {})[key] = stored
                    continue
                # R38 P0-001：MERGE 任务结果 = 因子最终结果（同 ROOT 处理）。
                if task_type in (TASK_ROOT, TASK_MERGE):
                    if sink is not None:
                        if not sink.submit(dag.tasks[key].factor_name, result):
                            self._explain(
                                f"sink.submit FALSE for {dag.tasks[key].factor_name} "
                                f"— writer fatal / queue closed; abort generation"
                            )
                            raise RuntimeError(
                                f"sink.submit returned False for "
                                f"{dag.tasks[key].factor_name} (R38-P0-046: writer "
                                "failure must abort the generation)"
                            )
                    if result_handler is not None:
                        result_handler(dag.tasks[key].factor_name, result)
                    elif sink is None:
                        self._results[dag.tasks[key].factor_name] = result
                # 释放已消费的 CSE sid（引用计数归零立即释放）。
                if task_type in (TASK_ROOT, TASK_MERGE):
                    self._release_consumed(dag, key, ctx)
            # 4) R36 P0-003：动态 CPU 由 ResourceController 双向控制（每 tick 在
            #    顶部决策里 fast down / slow up）——不再一次性写死 0.6。
            # 5) 无进展保护：admission 全部被拒且没有在跑 future → 等资源释放后
            #    再试；连续多轮无进展则**先 auto-shard replan smaller**（§100），
            #    再失败——不能只报 stuck。
            if admitted_this_round == 0 and not futures and remaining:
                no_progress_rounds += 1
                if no_progress_rounds >= 200:
                    replanned = self._auto_shard_replan(dag, remaining)
                    if replanned:
                        self._explain(
                            f"R36_AUTO_SHARD: replanned {replanned} task(s) smaller "
                            f"(remaining={len(remaining)})"
                        )
                        no_progress_rounds = 0
                        continue
                    self._explain(
                        "STUCK: no task admitted for 200 rounds; broker headroom="
                        f"{self.broker.snapshot().live_headroom} remaining={sorted(remaining)[:5]}"
                    )
                    raise RuntimeError(
                        "AdaptiveBatchScheduler made no progress: all ready tasks "
                        "rejected by ResourceBroker admission (auto-shard not legal); "
                        f"broker={self.broker.summary()} remaining={sorted(remaining)[:10]}"
                    )
            else:
                no_progress_rounds = 0
        if sink is not None:
            sink.finish()
        self._done = real_done + virtual_done + wave_scan_done
        self._scheduler_stats = {
            "real_task_done": real_done,
            "virtual_task_done": virtual_done,
            "wave_scan_done": wave_scan_done,
            "virtual_task_ratio": round(
                virtual_done / max(1, real_done + virtual_done), 4
            ),
            "concurrency_limit_final": self._dynamic_concurrency_limit(sink),
            "read_waves": self._wave_summary,
            "resource_decision": (
                self._last_decision.to_dict()
                if self._last_decision is not None
                else None
            ),
            "resource_controller": self.broker.resource_controller_summary(),
            # R39-PERF-018/019：结构化性能计数。
            "future_count": self._perf_metrics["future_count"],
            "factor_count": self._perf_metrics["factor_count"],
            "micro_batch_task_count": self._perf_metrics["micro_batch_task_count"],
            "micro_batch_root_count": self._perf_metrics["micro_batch_root_count"],
            "future_per_factor": round(
                self._perf_metrics["future_count"]
                / max(1, self._perf_metrics["factor_count"]),
                4,
            ),
            "scheduler_wait_polling_count": self._perf_metrics[
                "scheduler_wait_polling_count"
            ],
            # P0#8（100k GO §11）：执行策略单一权威——None=classifier 权威，
            # thread/process=env 强制覆盖。telemetry 可见。
            "execution_policy": self._execution_policy,
        }
        return {
            "results": self._results,
            "task_timing": self._task_timing,
            "explanations": self._explanations,
            "done": self._done,
            "broker": self.broker.summary(),
            "executor": self.executor.summary(),
            "scheduler_stats": self._scheduler_stats,
        }

    def run_serial_fused(
        self,
        plan: SchedulerPlan,
        *,
        backend: Any,
        ctx: Any,
        execute_root: Callable[[PhysicalFactorTask], Any] | None = None,
        materialize_shared: Callable[[str, Any], Any] | None = None,
        sink: StreamingResultSink | None = None,
        result_handler: Callable[[str, Any], None] | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds: Any = None,
    ) -> dict[str, Any]:
        """R33 §22/§39：small-batch AUTO bypass——serial fused。

        极少量简单因子时，scheduler 开销 > compute savings：不建 future、不占
        resource lease、不走 priority queue。仍先执行 read waves（真实 scan 一次）
        再串行物化 shared + 执行 roots（拓扑序）。结果与 ``run`` 完全一致。
        """
        from factor_engine.runtime.batch_service import _execute_root_with_path, _materialize_shared_subplan

        def _default_execute_root(task: PhysicalFactorTask) -> Any:
            node = getattr(task, "node_ref", None)
            root = getattr(node, "root", node)
            result, _path = _execute_root_with_path(
                backend, root, ctx, run_mode=getattr(ctx, "run_mode", None),
                factor_name=task.factor_name,
                task_id=task.task_id,
                factor_id=task.factor_name,
            )
            return result

        execute_root = execute_root or _default_execute_root
        materialize_shared = materialize_shared or (
            lambda sid, node: _materialize_shared_subplan(backend, node, ctx, sid)
        )
        self._input_dq_reports = []
        sink = sink or self.sink
        dag = getattr(plan, "physical_dag", plan)
        committed: set[str] = set()
        remaining = set(dag.topological_order())
        # Interleave one admitted wave with every consumer it makes ready.
        # A small pool can then reclaim wave residency before admitting the next
        # wave; source tasks are never virtually committed without a real read.
        while remaining:
            self._execute_read_waves(
                plan,
                dag,
                committed,
                remaining,
                ctx,
                input_dq_check=input_dq_check,
                input_dq_strict=input_dq_strict,
                input_dq_thresholds=input_dq_thresholds,
                defer_on_pressure=True,
            )
            progressed = False
            for tid in dag.topological_order():
                if tid in committed or tid not in remaining:
                    continue
                task = dag.tasks.get(tid)
                if task is None or not all(parent in committed for parent in task.inputs):
                    continue
                if task.task_type == TASK_SOURCE_SCAN:
                    # Only _execute_read_waves may commit a physical scan.
                    continue
                if task.task_type == TASK_CSE_SHARED:
                    sid = task.task_id.split(":", 1)[1]
                    materialize_shared(sid, task.node_ref)
                    committed.add(tid)
                    remaining.discard(tid)
                    self._record_timing(tid, task)
                    progressed = True
                    continue
                if task.task_type == TASK_ROOT:
                    result = execute_root(task)
                    committed.add(tid)
                    remaining.discard(tid)
                    self._record_timing(tid, task)
                    self._record_task_calibration(tid, task, result)
                    if sink is not None:
                        if not sink.submit(task.factor_name, result):
                            raise RuntimeError(
                                f"sink.submit returned False for {task.factor_name}"
                            )
                    if result_handler is not None:
                        result_handler(task.factor_name, result)
                    elif sink is None:
                        self._results[task.factor_name] = result
                    self._release_consumed(dag, tid, ctx)
                    progressed = True
                    continue
                # Planning views are committed only after their predecessors.
                committed.add(tid)
                remaining.discard(tid)
                self._record_timing(tid, task)
                progressed = True
            if not progressed and remaining:
                # No consumer can return admission. Re-run without deferral to
                # produce the bounded typed denial instead of spinning.
                self._execute_read_waves(
                    plan,
                    dag,
                    committed,
                    remaining,
                    ctx,
                    input_dq_check=input_dq_check,
                    input_dq_strict=input_dq_strict,
                    input_dq_thresholds=input_dq_thresholds,
                    defer_on_pressure=False,
                )
        if sink is not None:
            sink.finish()
        self._done = len(committed)
        self._scheduler_stats = {
            "mode": "DIRECT_VECTOR",
            "serial_fused": True,
            "real_task_done": len(committed),
            "virtual_task_done": 0,
            "read_waves": self._wave_summary,
            # R39-PERF-018/019：serial-fused 路径不建 future / 不轮询。
            "future_count": 0,
            "factor_count": sum(
                1
                for _t in dag.tasks.values()
                if getattr(_t, "task_type", "") == TASK_ROOT
            ),
            "micro_batch_task_count": 0,
            "micro_batch_root_count": 0,
            "future_per_factor": 0.0,
            "scheduler_wait_polling_count": 0,
        }
        return {
            "results": self._results,
            "task_timing": self._task_timing,
            "explanations": self._explanations,
            "done": self._done,
            "broker": self.broker.summary(),
            "executor": self.executor.summary(),
            "scheduler_stats": self._scheduler_stats,
        }

    def _critical_path_scores(
        self, dag: PhysicalFactorDAG, cost_ms: dict[str, float]
    ) -> dict[str, float]:
        """Return all critical-path scores with one reverse-DAG pass per run.

        ``_priority_score`` is called once per ready task.  Calling the DAG's
        single-task helper there repeated a full topological traversal for every
        candidate, which is quadratic for a wide batch.  The DAG is immutable
        during normal scheduling; OOM/shard replan explicitly invalidates this
        cache below.
        """
        cache = self._critical_path_cache
        dag_key = id(dag)
        if cache is not None and cache[0] == dag_key:
            return cache[1]
        try:
            order = dag.topological_order()
        except RuntimeError:
            order = list(dag.tasks)
        scores: dict[str, float] = {}
        for tid in reversed(order):
            task = dag.tasks.get(tid)
            consumers = (
                [c for c in task.consumers if c in dag.tasks]
                if task is not None else []
            )
            own = float(cost_ms.get(tid, 0.0))
            scores[tid] = own + (max(scores[c] for c in consumers) if consumers else 0.0)
        self._critical_path_cache = (dag_key, scores)
        return scores

    def _priority_score(self, dag: PhysicalFactorDAG, task: PhysicalFactorTask) -> float:
        """R33-P0-039：ready queue 优先级 = critical path + reuse + fanout - memory.

        R39-PERF-016：cost_ms 从证书 O(1) 读取并缓存；critical-path DP 也只
        计算一次 per DAG（而不是每个 ready task 重走整棵 DAG）。
        """
        cost_ms = self._cost_ms_cache
        if cost_ms is None:
            cost_ms = {}
            for tid, t in dag.tasks.items():
                cert = self._certificates.get(tid)
                if cert is not None:
                    cost_ms[tid] = cert.total_work()
                else:
                    cost_ms[tid] = float(
                        (t.resource_contract.predicted_elapsed_ms or 0.0)
                        if t.resource_contract else 0.0
                    )
            self._cost_ms_cache = cost_ms
        critical = self._critical_path_scores(dag, cost_ms).get(
            task.task_id, float(cost_ms.get(task.task_id, 0.0))
        )
        reuse = self._reuse_counts.get(task.task_id, len(task.consumers))
        fanout = len(task.consumers)
        cert = self._certificates.get(task.task_id)
        if cert is not None:
            memory = float(cert.peak_memory_bytes())
        else:
            memory = float(
                task.resource_contract.peak_memory_bytes
                if task.resource_contract else 0
            )
        return critical * 1.0 + reuse * 1000.0 + fanout * 100.0 - memory / (1024**3)

    def _pin_consumed_sids(self, ctx: Any, task: PhysicalFactorTask) -> None:
        """P0-015：任务即将执行，其消费的 CSE 共享 sid 在消费期内必须 pin。

        防止 LRU 把仍有消费者的共享结果逐出（``plan_ref`` 命中后被 evict → miss）。
        最后一个消费者完成时 ``_release_consumed_sids`` → ``store.release(sid)``
        才会 unpin + 释放（refcount 归零）。
        """
        node = getattr(task, "node_ref", None)
        if node is None or ctx is None:
            return
        try:
            from factor_engine.planner.cse import collect_consumed_sids

            store = getattr(ctx, "shared_buffers", None)
            if store is None or getattr(store, "pin", None) is None:
                return
            for sid in collect_consumed_sids(getattr(node, "root", node)):
                store.pin(sid)
        except Exception:
            pass

    def _release_consumed(self, dag: PhysicalFactorDAG, tid: str, ctx: Any) -> None:
        """root 完成后对其消费的 CSE sid 引用计数归零则释放（复用 batch_service）。

        R33-P0-041：释放失败**不静默**——记录 ``CSE_RELEASE_FAILURE``（telemetry +
        fail-safe cleanup），不再 ``except: pass`` 吞掉内存回收失败。
        """
        try:
            from factor_engine.runtime.batch_service import _release_consumed_sids

            task = dag.tasks[tid]
            if task.node_ref is not None:
                node = getattr(task.node_ref, "root", task.node_ref)
                _release_consumed_sids(ctx, node)
        except Exception as exc:  # noqa: BLE001
            self._explain(f"CSE_RELEASE_FAILURE task={tid}: {type(exc).__name__}: {exc}")
            try:
                runtime = dict(getattr(ctx, "runtime_stats", None) or {})
                fails = runtime.get("cse_release_failures", 0)
                runtime["cse_release_failures"] = int(fails) + 1
                ctx.runtime_stats = runtime  # type: ignore[attr-defined]
            except Exception:
                pass

    def _mark_wave_consumer_done(self, tid: str) -> None:
        """Drop scheduler-owned wave refs after their final physical consumer."""
        for wid, pending in list(self._wave_pending_consumers.items()):
            pending.discard(tid)
            if pending:
                continue
            self._wave_pending_consumers.pop(wid, None)
            source_tasks = self._wave_source_tasks.pop(wid, ())
            for source_tid in source_tasks:
                self._buffer_results.pop(source_tid, None)
            self._wave_refs.pop(wid, None)
            release_wave = getattr(self._wave_executor, 'release_wave', None)
            if callable(release_wave):
                release_wave(wid)

    def _record_timing(self, tid: str, task: PhysicalFactorTask) -> None:
        import time

        now_ms = time.monotonic() * 1000.0
        started = self._task_started_at.pop(tid, now_ms)
        self._task_timing[tid] = {
            "op": task.op,
            "task_type": task.task_type,
            "factor_name": task.factor_name,
            # R38 P0-007：started_at（dispatch 时）+ finished_at（完成时）→
            # elapsed_ms 是真实 duration。
            "started_at_ms": round(started, 3),
            "finished_at_ms": round(now_ms, 3),
            "elapsed_ms": round(max(0.0, now_ms - started), 3),
            "preferred_backend": task.preferred_backend,
        }
        self._mark_wave_consumer_done(tid)

    def _auto_shard_replan(self, dag: PhysicalFactorDAG, remaining: set[str]) -> int:
        """R38 P0-001（§4）：ready task 无法 admission 且 shardable → **真实**替换成
        shard children + merge barrier（不再是只改 peak_memory 数字的伪 sharding）。

        返回真实 replan 的 task 数。切片信息（time_range / instrument_universe）
        不足时无法构造真实切片 → 返回 0（后续 STUCK 诚实失败）。
        """
        if not getattr(self, "_shard_replanned", None):
            self._shard_replanned: set[str] = set()
        try:
            from factor_engine.runtime.auto_shard_planner import AutoShardPlanner

            env = self.broker.resource_envelope()
            safe = max(1, int(env.safe_memory_bytes))
            candidates = [
                dag.tasks[t]
                for t in remaining
                if t in dag.tasks and dag.tasks[t].task_type == TASK_ROOT
                and dag.tasks[t].executable and t not in self._shard_replanned
            ]
            planner = AutoShardPlanner()
            n = 0
            for task in candidates:
                tid = task.task_id
                plan = planner.build_shard_execution_plan(
                    task,
                    safe_envelope_bytes=safe,
                    time_range=task.time_range,
                    instrument_universe=list(task.instrument_scope)
                    if task.instrument_scope else None,
                    lookback_bars=self._lookback_bars(task),
                    calendar=self._trading_calendar_for(),
                )
                if plan is None:
                    continue
                if not self._replace_with_shard_plan(dag, tid, plan):
                    continue
                self._shard_replanned.add(tid)
                for sid in plan.shard_task_ids:
                    remaining.add(sid)
                remaining.add(plan.merge_task_id)
                n += 1
            return n
        except Exception:
            return 0

    def _lookback_bars(self, task: PhysicalFactorTask) -> int:
        """估算 time-shard warmup 需要的 lookback 交易日（rolling 覆盖窗口）。"""
        try:
            from factor_engine.runtime.resource_shape import ResourceShapeKey

            shape = ResourceShapeKey.from_task(task)
            # window_bucket 中点近似（保守 ×2 覆盖 lookback）。
            midpoints = {0: 5, 1: 20, 2: 60, 3: 120, 4: 252, 5: 500}
            return max(20, int(midpoints.get(shape.window_bucket, 60)) * 2)
        except Exception:
            return 60

    def _next_shard_attempt(self, original_task_id: str) -> int:
        """分配分片 attempt 序号（P0-010 attempt 隔离）。

        pre-shard 首次 = 1（task id 用旧命名）；OOM replan 起每次递增（>=2），
        新 shard/merge id 带 ``:attempt:N:`` 段，与旧 attempt 的 in-flight
        future 彻底隔离。
        """
        with self._lock:
            n = self._shard_attempts.get(original_task_id, 1) + 1
            self._shard_attempts[original_task_id] = n
            return n

    def _trading_calendar_for(self) -> Any | None:
        """从 run ctx 的数据源推导真实交易日历（无 ctx / 无法推断 → None）。

        有真实日历时 time-shard 按真实 session 切分 + warmup 按 N 个真实交易日
        回退；None 时 planner 回退 bdate 近似（research / 未知市场不阻塞 shard）。
        """
        try:
            ctx = getattr(self, "_run_ctx", None)
            if ctx is None:
                return None
            ds = getattr(ctx, "data_source", None)
            dataset = str(getattr(ds, "dataset", "") or "")
            universe = str(getattr(ctx, "universe", "") or "")
            from factor_engine.storage.trading_calendar import get_trading_calendar, infer_market

            market = infer_market(universe=universe, dataset=dataset)
            if not market:
                return None
            return get_trading_calendar(market, allow_approximate_calendar=True)
        except Exception:
            return None

    def _replace_with_shard_plan(
        self, dag: PhysicalFactorDAG, original_task_id: str, plan: Any
    ) -> bool:
        """把 ROOT task 替换成 shard children + MERGE barrier（真实 DAG 改造，§4）。

        - shard 子任务：task_type=SHARD，carry shard_descriptor + 每片契约；
        - MERGE 任务：task_type=MERGE，carry ShardExecutionPlan（merge barrier +
          merge contract），inputs = 全部 shard id，consumers = 原 task consumers。
        - 原 task 的 predecessor/consumer 引用全部重定向到 MERGE。
        """
        from dataclasses import replace as _replace

        from factor_engine.runtime.shard_execution_plan import shape_signature

        # 已被替换过（OOM 后第二次 replan）时，原 ROOT 从 preserved dict 取。
        orig = self._shard_original_task.get(original_task_id) or dag.tasks.get(original_task_id)
        if orig is None or orig.task_type != TASK_ROOT:
            return False
        contract = orig.resource_contract
        shard_contract = (
            _replace(
                contract,
                peak_memory_bytes=plan.per_shard_peak_bytes,
                output_bytes=plan.per_shard_peak_bytes,
                estimate_basis=f"shard:{plan.dimension}",
            )
            if contract is not None
            else None
        )
        # Merge 任务用**独立内存模型**（P0-003）：不再退回 original_peak（那正是
        # 超 SafeEnvelope 被拆的原因——merge 会把自己卡死；auto-shard 不拆 MERGE）。
        # merge 峰值 ≈ 一片输入 + 累计输出（大 shard 已 spool，逐片 reload）。
        from factor_engine.runtime.shard_execution_plan import merge_resource_model

        merge_peak = merge_resource_model(
            original_peak_bytes=plan.original_peak_bytes,
            per_shard_peak_bytes=plan.per_shard_peak_bytes,
            output_bytes=int(getattr(contract, "output_bytes", 0) or 0)
            if contract is not None else 0,
            shard_count=len(plan.shards),
        )
        merge_contract = (
            _replace(
                contract,
                peak_memory_bytes=merge_peak,
                output_bytes=max(1, merge_peak),
                estimate_basis="shard-merge",
            )
            if contract is not None
            else contract
        )
        # 移除旧 task 及其已存在的 shard/merge（幂等重建）。attempt>=2 的旧命名
        # ``root:x:attempt:N:*`` 一并移除（OOM replan 旧 attempt 的 in-flight
        # future 完成后按「不在 dag.tasks」直接丢弃，不复用同名 id）。
        old_ids = [
            tid
            for tid in list(dag.tasks)
            if tid == original_task_id
            or tid.startswith(original_task_id + ":shard:")
            or tid == f"{original_task_id}:merge"
            or tid.startswith(original_task_id + ":attempt:")
        ]
        for tid in old_ids:
            dag.tasks.pop(tid, None)
        consumers = orig.consumers
        # shard/merge 需要可执行 PlanNode（ROOT 的 node_ref 可能是 FactorPlan）。
        node = getattr(orig.node_ref, "root", None) or orig.node_ref
        shard_ids: list[str] = []
        for descriptor in plan.shards:
            sid = descriptor.shard_id
            shard_ids.append(sid)
            dag.tasks[sid] = rebase_task(
                orig,
                task_id=sid,
                task_type=TASK_SHARD,
                op="shard",
                node_ref=node,
                factor_name=orig.factor_name,
                resource_contract=shard_contract,
                inputs=orig.inputs,
                consumers=(plan.merge_task_id,),
                shard_descriptor=descriptor,
                shard_plan=None,
            )
        merge_id = plan.merge_task_id
        dag.tasks[merge_id] = rebase_task(
            orig,
            task_id=merge_id,
            task_type=TASK_MERGE,
            op="shard_merge",
            node_ref=node,
            factor_name=orig.factor_name,
            resource_contract=merge_contract,
            inputs=tuple(shard_ids),
            consumers=consumers,
            shard_descriptor=None,
            shard_plan=plan,
        )
        # roots 与全图引用重定向（predecessor.consumers / 其它 task 的 inputs）。
        dag.roots = tuple(merge_id if r == original_task_id else r for r in dag.roots)
        for tid in list(dag.tasks):
            t = dag.tasks[tid]
            new_inputs = tuple(
                merge_id if i == original_task_id else i for i in t.inputs
            )
            new_consumers = tuple(
                merge_id if c == original_task_id else c for c in t.consumers
            )
            if new_inputs != t.inputs or new_consumers != t.consumers:
                dag.tasks[tid] = rebase_task(
                    t, inputs=new_inputs, consumers=new_consumers
                )
        self._shard_original_task[original_task_id] = orig
        sig = shape_signature(
            task_id=original_task_id,
            dimension=plan.dimension,
            shard_count=len(plan.shards),
            per_shard_peak_bytes=plan.per_shard_peak_bytes,
        )
        # shape 身份统一按 **original_task_id** 存（OOM 查找也是 orig_tid，不再
        # merge_id/orig_id 混用导致查找永远 miss、same-shape 保护失效）。
        self._shard_shape_of[original_task_id] = sig
        # R39-PERF-016：DAG 被真实改造（shard/merge 新 task）→ 强制重建成本图缓存。
        self._cost_ms_cache = None
        self._critical_path_cache = None
        self._explain(
            f"R38_REAL_AUTOSHARD: {original_task_id} -> {len(plan.shards)} "
            f"{plan.dimension} shards + merge (sig {sig[:8]})"
        )
        return True

    def _maybe_preshard_oversized(
        self, dag: PhysicalFactorDAG, remaining: set[str]
    ) -> int:
        """R38 P1-065（§§47 优先级第一组）：ready ROOT 的 P99 已超 SafeEnvelope
        时，在第一次 admission 前就 pre-shard —— 不等 200 轮 no-progress。"""
        if not getattr(self, "_shard_replanned", None):
            self._shard_replanned: set[str] = set()
        try:
            from factor_engine.runtime.auto_shard_planner import AutoShardPlanner

            env = self.broker.resource_envelope()
            safe = max(1, int(env.safe_memory_bytes))
            n = 0
            for tid in list(remaining):
                if tid in self._shard_replanned:
                    continue
                task = dag.tasks.get(tid)
                if task is None or task.task_type != TASK_ROOT or not task.executable:
                    continue
                if not all(p not in remaining for p in task.inputs):
                    continue
                contract = task.resource_contract
                peak = int(getattr(contract, "peak_memory_bytes", 0) or 0)
                if peak <= safe:
                    continue
                if not getattr(contract, "shardable", False):
                    continue
                planner = AutoShardPlanner()
                plan = planner.build_shard_execution_plan(
                    task,
                    safe_envelope_bytes=safe,
                    time_range=task.time_range,
                    instrument_universe=list(task.instrument_scope)
                    if task.instrument_scope else None,
                    lookback_bars=self._lookback_bars(task),
                    calendar=self._trading_calendar_for(),
                )
                if plan is None:
                    continue
                if not self._replace_with_shard_plan(dag, tid, plan):
                    continue
                self._shard_replanned.add(tid)
                remaining.discard(tid)
                for sid in plan.shard_task_ids:
                    remaining.add(sid)
                remaining.add(plan.merge_task_id)
                n += 1
            if n:
                self._preshards += n
            return n
        except Exception:
            return 0

    def _handle_oom(
        self,
        key: str,
        exc: BaseException,
        dag: PhysicalFactorDAG,
        remaining: set[str],
    ) -> bool:
        """R38 P0-005（§5）：OOM → smaller-shape replan → 重试（same shape 禁止）。

        流程：标记 shape underpredicted → calibration record(oom=True) → 降并发/
        wave/block → replan_after_oom → 确保 new_shape_signature != failed → 重建
        DAG → 重新调度 shard。返回是否已 replan（False → 上层如实 raise）。
        """
        try:
            from factor_engine.runtime.resource_calibration_store import global_calibration_store
            from factor_engine.runtime.resource_shape import ResourceShapeKey
            from factor_engine.runtime.shard_execution_plan import shape_signature

            # 1) 定位 original root task 与失败 shape。
            orig_tid = key
            plan = None
            merge_task = None
            if key.endswith(":merge"):
                merge_task = dag.tasks.get(key)
                if merge_task is not None and merge_task.shard_plan is not None:
                    plan = merge_task.shard_plan
                    orig_tid = plan.original_task_id
            elif ":shard:" in key:
                merge_id = key.rsplit(":shard:", 1)[0] + ":merge"
                m = dag.tasks.get(merge_id)
                if m is not None and m.shard_plan is not None:
                    plan = m.shard_plan
                    orig_tid = plan.original_task_id
            failed_sig = self._shard_shape_of.get(orig_tid) or f"full:{orig_tid}"
            # 2) same-shape 禁止重试。
            if failed_sig in self._failed_shapes:
                self._explain(
                    f"OOM_REPLAN_REFUSED: {orig_tid} shape {failed_sig[:8]} already "
                    "failed — same shape never retried (R38_P0_ZERO_SAME_SHAPE_OOM_RETRY)"
                )
                return False
            self._failed_shapes.add(failed_sig)
            # 3) calibration 标记 OOM（尾部立即上调，不被均值稀释）。
            orig_task = self._shard_original_task.get(orig_tid) or dag.tasks.get(orig_tid)
            if orig_task is not None:
                try:
                    global_calibration_store().record(
                        ResourceShapeKey.from_task(orig_task),
                        elapsed_ms=0.0,
                        peak_mem=0,
                        oom=True,
                    )
                except Exception:
                    pass
            # 4) 降低并发 / wave / block（让后续小 shard 更容易 admission）。
            try:
                self.broker._cpu.set_soft_budget(
                    max(1, self.broker._cpu.soft_budget // 2)
                )
            except Exception:
                pass
            # 5) replan smaller（更多 shard → 更小 per-shard）。SafeEnvelope 用
            #    失败计划自己的 envelope（OOM 时点的资源约束，不随 live 波动）。
            from factor_engine.runtime.auto_shard_planner import AutoShardPlanner

            min_shards = 4 if plan is None else len(plan.shards) + 2
            planner = AutoShardPlanner(min_shards=min_shards)
            time_range = (
                plan.time_range
                if plan is not None
                else (orig_task.time_range if orig_task is not None else None)
            )
            universe = (
                plan.instrument_universe
                if plan is not None
                else (orig_task.instrument_scope if orig_task is not None else None)
            )
            safe = int(
                plan.safe_envelope_bytes
                if plan is not None and plan.safe_envelope_bytes
                else (self.broker.resource_envelope().safe_memory_bytes or 0)
            )
            safe = max(1, safe)
            # attempt 隔离（P0-010）：每次 OOM replan 递增 attempt，新 shard/merge
            # task id 带 ``:attempt:N:`` 段——旧 attempt 的 in-flight future 完成后
            # 按「id 不在 dag.tasks」丢弃，绝不写入新 attempt 的 partial map。
            attempt = self._next_shard_attempt(orig_tid)
            new_plan = planner.build_shard_execution_plan(
                orig_task,
                safe_envelope_bytes=safe,
                time_range=time_range,
                instrument_universe=list(universe) if universe else None,
                lookback_bars=self._lookback_bars(orig_task),
                failed_signature=failed_sig,
                calendar=self._trading_calendar_for(),
                attempt_id=attempt,
            )
            if new_plan is None:
                return False
            new_sig = shape_signature(
                task_id=orig_tid,
                dimension=new_plan.dimension,
                shard_count=len(new_plan.shards),
                per_shard_peak_bytes=new_plan.per_shard_peak_bytes,
            )
            if new_sig == failed_sig:
                return False
            # 6) 重建 DAG 并重新调度。
            if not self._replace_with_shard_plan(dag, orig_tid, new_plan):
                return False
            # 旧 attempt 的 shard/merge id 从 remaining 移除（已在 dag 中被替换，
            # 不能留在调度队列里造成 KeyError 或旧 merge 空跑）。
            if plan is not None:
                for _old in (*plan.shard_task_ids, plan.merge_task_id):
                    remaining.discard(_old)
            for sid in new_plan.shard_task_ids:
                remaining.add(sid)
            remaining.add(new_plan.merge_task_id)
            self._oom_replans += 1
            self._explain(
                f"OOM_REPLAN: {key} -> {len(new_plan.shards)} smaller shards "
                f"(sig {new_sig[:8]} != {failed_sig[:8]})"
            )
            return True
        except Exception:
            return False

    def _record_task_calibration(
        self, tid: str, task: PhysicalFactorTask, result: Any = None
    ) -> None:
        """R36 P0-005/006 + R38 P0-007/008/009：calibration 只接收真实观测。

        - ``elapsed_ms``：真实 duration（finished - started，dispatch 时刻起）；
        - ``output_bytes``：``estimate_object_bytes(result)``（真实输出，不再用
          contract 预测值）；
        - ``peak_mem``：本版本无 isolated per-task PSS（§171），如实标记
          ``attribution_quality="unattributed"`` → 不进入 P99 主模型（避免「拿
          预测当真实」的污染闭环）；predicted 值仅作诊断。
        """
        try:
            from factor_engine.runtime.resource_calibration_store import global_calibration_store
            from factor_engine.runtime.resource_shape import ResourceShapeKey
            from factor_engine.runtime.task_run_observation import estimate_output_bytes

            contract = getattr(task, "resource_contract", None)
            predicted_peak = int(contract.peak_memory_bytes) if contract is not None else 0
            timing = self._task_timing.get(tid, {})
            elapsed = float(timing.get("elapsed_ms", 0.0))
            if elapsed <= 0:
                # 无 dispatch 起点（virtual/wave）→ 真实 duration 不可得，跳过。
                elapsed = 0.0
            out = estimate_output_bytes(result) if result is not None else 0
            spill = int(contract.spill_bytes) if contract is not None else 0
            global_calibration_store().record(
                ResourceShapeKey.from_task(task),
                elapsed_ms=elapsed,
                # R38 P0-008：unattributed → 峰值不进入 P99 主模型。
                peak_mem=predicted_peak,
                attribution_quality="unattributed",
                peak_is_trusted=False,
                output_bytes=out,
                spill_bytes=spill,
            )
        except Exception:
            pass  # calibration 是性能数据，不因失败影响正确执行

    # -- materialize --

    def materialize(
        self,
        plan: SchedulerPlan,
        *,
        backend: Any,
        ctx: Any,
        writer: Callable[[list[ResultItem]], None],
        queue_bytes: int | None = None,
        batch_size: int = 1,
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        """run + 流式写（R27-102..109 compute 与 write pipeline 重叠）。

        R36 P0-011：``queue_bytes=None`` 时从 broker ResourceDecision 取动态
        sink 预算（§38/39：min(SafeEnvelope×0.05, job_lease×0.10, cap)）。
        """
        if queue_bytes is None:
            try:
                decision = self.broker.resource_decision()
                self._last_decision = decision
                queue_bytes = max(1, int(decision.result_queue_bytes))
            except Exception:
                # P3/P4: 从 broker live headroom 派生 sink 预算（不再固定 4GiB）；
                # 都不行才回退绝对上限。
                try:
                    queue_bytes = self.broker.current_sink_budget() or (4 * 1024**3)
                except Exception:
                    queue_bytes = 4 * 1024**3
        sink = StreamingResultSink(
            writer=writer,
            queue_bytes=queue_bytes,
            batch_size=batch_size,
            writer_threads=1,
        )
        sink.start()
        out = self.run(plan, backend=backend, ctx=ctx, sink=sink, **run_kwargs)
        out["streaming_sink"] = sink.summary()
        return out
