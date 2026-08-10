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
import threading
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from planner.dag_cost_model import task_priority
from planner.physical_factor_dag import (
    TASK_CSE_SHARED,
    TASK_ROOT,
    TASK_SOURCE_SCAN,
    PhysicalFactorDAG,
    PhysicalFactorTask,
    rebase_task,
    task_is_executable,
)
from planner.read_wave_planner import ReadWavePlan, build_waves_from_dag
from runtime.buffer_ref import SourceWaveExecutor
from runtime.hybrid_executor import HybridExecutor, classify_backend_execution
from runtime.resource_broker import ReservationLease, ResourceBroker
from runtime.streaming_result_sink import ResultItem, StreamingResultSink

_logger = logging.getLogger(__name__)

#: R33-P0-040：``wait(FIRST_COMPLETED)`` 事件驱动，超时只作兜底上限（不再固定
#: 50ms 轮询）。
_EVENT_WAIT_TIMEOUT_S = 0.05


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
        from backend.operator_cost import estimate_plan_cost

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
        from runtime.execution_traits import thread_budget

        with thread_budget(budget):
            result = execute_root(task)
        return task.task_id, result
    return task.task_id, None


#: 错误分类（R31-006）：只有 transient 类自动 retry；确定性 / 语义类禁止 retry。
ERROR_TRANSIENT = "transient"
ERROR_PERMANENT = "permanent"
ERROR_UNKNOWN = "unknown"

_TRANSIENT_EXC_TYPES: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
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


def classify_error(exc: BaseException) -> str:
    """R31-006：错误分类——transient 自动 retry；permanent 禁止 retry。

    - ``OSError/TimeoutError/ConnectionError`` 及显式 transient 标记 → transient
    - PIT violation / semantic violation / invalid param / unsupported op /
      schema mismatch / deterministic numeric / DQ error → permanent
    - 其余 → unknown（保守单次 retry）
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if any(m in msg for m in _PERMANENT_MARKERS):
        return ERROR_PERMANENT
    if isinstance(exc, _TRANSIENT_EXC_TYPES) or "transient" in msg:
        return ERROR_TRANSIENT
    if "oom" in msg or "out of memory" in msg or name in {"memoryerror"}:
        # OOM 不算 transient：原尺寸重跑只会再 OOM（R31-006 要求降 shard/并发）。
        return ERROR_PERMANENT
    return ERROR_UNKNOWN


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
    from planner.native_fusion import execute_fusion_group

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
    from runtime.task_resource_contract import TaskResourceContract

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
        wave_memory_budget: int = 4 * 1024**3,
        max_concurrency: int | None = None,
    ) -> None:
        if broker is None:
            # R31-P1-039 + R36 P0-016：优先用 service ContextVar 里的共享 broker
            # （job admission + task admission 统一）；没有则用进程级唯一
            # HostResourceCoordinator 的 broker——不再各自 new 独立 ResourceBroker
            #（§52 one host resource authority）。
            try:
                from service.queue import _get_service_broker

                broker = _get_service_broker()
            except Exception:
                broker = None
        if broker is None:
            try:
                from runtime.host_resource_coordinator import get_host_coordinator

                broker = get_host_coordinator().broker
            except Exception:
                broker = None
        self.broker = broker or ResourceBroker()
        self.executor = executor or HybridExecutor(broker=self.broker)
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
        self._wave_summary: dict[str, Any] = {"waves_planned": 0, "waves_executed": 0, "events": []}
        # R33-P0-009：SOURCE_SCAN task 的 BufferRef 输出（独立命名空间，不污染
        # 因子结果 ``self._results``）。
        self._buffer_results: dict[str, Any] = {}
        # R33-P0-008/§44：real vs virtual task 比（scheduler runtime evidence）。
        self._scheduler_stats: dict[str, Any] = {
            "real_task_done": 0, "virtual_task_done": 0, "virtual_task_ratio": 0.0,
        }
        # R36 P0-002：最近一次 ResourceDecision（broker 建议值，admission 消费）。
        self._last_decision: Any | None = None

    def cancel(self) -> None:
        """请求取消：后续 admission 一律拒绝；run 在无在跑任务时提前结束。"""
        self._cancelled = True
        self._explain("CANCELLED: no new task admitted; finishing running tasks")

    def _dynamic_wave_budget(self) -> int:
        """R36 P0-010：从 broker ResourceDecision 取动态 read-wave 预算。"""
        try:
            decision = self.broker.resource_decision()
            self._last_decision = decision
            return max(1, int(decision.read_wave_bytes))
        except Exception:
            return self.wave_memory_budget

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
        from planner.cse import collect_consumed_sids
        from planner.physical_lowerer import contract_for_plan, lower_batch_dag

        # 尝试经 ctx 拿真实行数/仪器数（SourceScanStage 用真实 ScanCost 估计）。
        rows: int | None = None
        instruments = 0
        if ctx is not None:
            try:
                from backend.plan_cost_router import estimate_plan_rows

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
        for tid, task in physical.tasks.items():
            plan_cost = (
                task.estimated_cost
                if task.estimated_cost is not None
                else _plan_cost_bytes(task.node_ref)
            )
            cost_by_task[tid] = plan_cost
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
        from planner.native_fusion import native_fusion_capability_map, plan_native_fusion_groups

        fusion_groups = []
        if enable_cse and root_tasks:
            capability = fusion_backend_capability or native_fusion_capability_map(ctx)
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
        contract = task.resource_contract
        fn = _dispatch
        if contract is None:
            self._explain(f"task={task.task_id}: no contract, admit (vacuous)")
            return (
                self.executor.submit(
                    task.preferred_backend,
                    fn, task, backend, ctx, execute_root, materialize_shared,
                    prefer="thread",
                ),
                None,
            )
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
        # R31-P0-007 诚实声明：FE root 执行模型（backend.execute + 共享 ctx/cache）
        # 的 payload 不可 pickle；进程执行需要 worker-local runtime（Phase D）。
        # scheduler 统一走 thread pool，并发由 ResourceBroker CPU token 约束。
        future = self.executor.submit(
            task.preferred_backend,
            fn, task, backend, ctx, execute_root, materialize_shared,
            prefer="thread",
        )
        return future, lease

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
        decision = getattr(self, "_last_decision", None)
        broker_limit = 1
        if decision is not None:
            broker_limit = max(1, int(decision.target_cpu_tokens))
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
    ) -> int:
        """R33-P0-016：把 read wave 真正接进 scheduler 主路径。

        对每个 wave：当它覆盖的 SOURCE_SCAN task 全部 ready（inputs committed）
        时执行一次 scan（``SourceWaveExecutor.execute_wave`` → prefetch union 列
        → 共享列缓存），并把 SOURCE_SCAN task 标记 committed（其 IO 已在 wave
        中完成），BufferRef 写入结果表。executor / refs 为实例状态（幂等：同一
        wave_id 只执行一次）。wave 失败仍记录 event + committed（root 执行会按需
        load 并如实抛错）——不静默。返回本次提交的 SOURCE_SCAN task 数。
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
            tids = [t for t in wave.task_ids if t in dag.tasks]
            if not tids:
                continue
            if any(
                not all(p in committed for p in dag.tasks[t].inputs)
                for t in tids
            ):
                continue
            ref = executor.execute_wave(wave, consumer_ids=tids)
            refs[wid] = ref
            # R33-P0-017：input DQ 按 wave 在 scan 后做（一次 scan 同时产生 data
            # buffer + DQ stats，不另扫一遍）。
            if input_dq_check and ref is not None and wave.columns:
                try:
                    from runtime.input_dq import assert_input_dq
                    from runtime.input_dq import (
                        adjust_input_dq_thresholds_from_stats,
                        load_dataset_stats_for_source,
                    )

                    stats = load_dataset_stats_for_source(source)
                    thresholds = adjust_input_dq_thresholds_from_stats(
                        input_dq_thresholds, stats, list(wave.columns)
                    )
                    assert_input_dq(
                        source,
                        list(wave.columns),
                        raise_on_fail=input_dq_strict,
                        thresholds=thresholds,
                    )
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
                    committed_count += 1
        self._wave_summary = executor.summary()
        return committed_count

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
        """执行 DAG：read waves + ready queue + admission + FIRST_COMPLETED。

        R33 升级：
            - P0-016 read wave 在 SOURCE_SCAN ready 时真实执行（prefetch union 列
              一次 → BufferRef 进结果表 → consumers 复用共享缓存）。
            - P0-008 barrier 规划视图自动提交（不占 resource lease）。
            - P0-038 ``running < dynamic_concurrency_limit`` 显式并发闸。
            - P0-039 ready set 按 priority 排序主导 admission。
            - P0-040 ``wait(FIRST_COMPLETED)`` 事件驱动（不再固定 50ms 轮询）。
        """
        from runtime.batch_service import _execute_root_with_path, _materialize_shared_subplan

        def _default_execute_root(task: PhysicalFactorTask) -> Any:
            # R31-P0-002：lowerer 生成的 ROOT task 的 node_ref 是裸 plan；
            # 旧 DAGPlan 路径 node_ref 是 FactorPlan（带 .root）。
            node = getattr(task, "node_ref", None)
            plan = getattr(node, "root", node)
            result, _path = _execute_root_with_path(
                backend, plan, ctx, run_mode=getattr(ctx, "run_mode", None),
                factor_name=task.factor_name,
            )
            return result

        execute_root = execute_root or _default_execute_root
        materialize_shared = materialize_shared or (
            lambda sid, node: _materialize_shared_subplan(backend, node, ctx, sid)
        )
        sink = sink or self.sink
        # 接受 SchedulerPlan（含 physical_dag）或裸 PhysicalFactorDAG。
        dag = getattr(plan, "physical_dag", plan)

        pending: list[str] = dag.topological_order()
        committed: set[str] = set()
        futures: dict[str, Future] = {}
        # R31-005: future → task_id / lease 双向绑定（失败/取消也能知道归属并释放）。
        future_to_task_id: dict[Future, str] = {}
        future_leases: dict[Future, ReservationLease] = {}
        # R31-P0-022：fusion group 登记（root task_id → group）。
        fusion_groups = list(getattr(plan, "fusion_groups", []) or [])
        group_by_root: dict[str, Any] = {}
        fusion_done: set[str] = set()
        future_group_roots: dict[Future, tuple[str, ...]] = {}
        for _g in fusion_groups:
            for _tid in _g.roots:
                group_by_root[_tid] = _g
        remaining = set(pending)
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
            # 0) read waves：SOURCE_SCAN ready → 真实 scan（R33-P0-016）。
            wave_scan_done += self._execute_read_waves(
                plan, dag, committed, remaining, ctx,
                input_dq_check=input_dq_check,
                input_dq_strict=input_dq_strict,
                input_dq_thresholds=input_dq_thresholds,
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
                future = self.executor.submit(
                    group.backend,
                    _dispatch_fusion, group, dag.tasks, backend, ctx, execute_root,
                    prefer="thread",
                )
                futures[gkey] = future
                future_to_task_id[future] = gkey
                future_leases[future] = leases
                future_group_roots[future] = tuple(roots)
                for t in roots:
                    remaining.discard(t)
                admitted_this_round += 1
                self._explain(
                    f"fusion group {group.group_id}: admitted {len(roots)} roots "
                    f"(backend={group.backend})"
                )
            # 2) admission：predecessor 全 committed 的 executable task，按 priority
            #    排序主导（R33-P0-039）。显式并发上限（R33-P0-038）。
            concurrency_limit = self._dynamic_concurrency_limit(sink)
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
            for _priority, tid in ready:
                if tid in futures or tid in committed:
                    continue
                if len(futures) >= concurrency_limit:
                    self._explain(f"task={tid}: deferred (concurrency_limit={concurrency_limit})")
                    break
                task = dag.tasks[tid]
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
            # 3) FIRST_COMPLETED：有 future 时等第一个完成（R33-P0-040 事件驱动）。
            done: set[Future] = set()
            if futures:
                done = wait(list(futures.values()), timeout=_EVENT_WAIT_TIMEOUT_S,
                            return_when=FIRST_COMPLETED)[0]
            for future in done:
                key = future_to_task_id.pop(future, None)
                if key is None:
                    continue
                lease = future_leases.pop(future, None)
                futures.pop(key, None)
                try:
                    _ret_key, result = future.result(timeout=1.0)
                except Exception as exc:  # noqa: BLE001
                    _release_lease(lease)
                    kind = classify_error(exc)
                    retries = self._retries_remaining.get(key, 1)
                    # R31-006：只有 transient（或未知=保守单次）自动 retry；
                    # permanent（PIT/semantic/参数/不支持算子/确定性错误/OOM）不重跑。
                    if retries > 0 and kind in {ERROR_TRANSIENT, ERROR_UNKNOWN}:
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
                        self._record_timing(_tid, dag.tasks[_tid])
                        self._record_task_calibration(_tid, dag.tasks[_tid])
                        _res = results_by_root.get(_tid)
                        if dag.tasks[_tid].task_type == TASK_ROOT:
                            if sink is not None:
                                sink.submit(dag.tasks[_tid].factor_name, _res)
                            if result_handler is not None:
                                result_handler(dag.tasks[_tid].factor_name, _res)
                            else:
                                self._results[dag.tasks[_tid].factor_name] = _res
                        self._release_consumed(dag, _tid, ctx)
                    continue
                committed.add(key)
                real_done += 1
                self._record_timing(key, dag.tasks[key])
                self._record_task_calibration(key, dag.tasks[key])
                if dag.tasks[key].task_type == TASK_ROOT:
                    if sink is not None:
                        sink.submit(dag.tasks[key].factor_name, result)
                    if result_handler is not None:
                        result_handler(dag.tasks[key].factor_name, result)
                    else:
                        self._results[dag.tasks[key].factor_name] = result
                # 释放已消费的 CSE sid（引用计数归零立即释放）。
                if dag.tasks[key].task_type == TASK_ROOT:
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
    ) -> dict[str, Any]:
        """R33 §22/§39：small-batch AUTO bypass——serial fused。

        极少量简单因子时，scheduler 开销 > compute savings：不建 future、不占
        resource lease、不走 priority queue。仍先执行 read waves（真实 scan 一次）
        再串行物化 shared + 执行 roots（拓扑序）。结果与 ``run`` 完全一致。
        """
        from runtime.batch_service import _execute_root_with_path, _materialize_shared_subplan

        def _default_execute_root(task: PhysicalFactorTask) -> Any:
            node = getattr(task, "node_ref", None)
            root = getattr(node, "root", node)
            result, _path = _execute_root_with_path(
                backend, root, ctx, run_mode=getattr(ctx, "run_mode", None),
                factor_name=task.factor_name,
            )
            return result

        execute_root = execute_root or _default_execute_root
        materialize_shared = materialize_shared or (
            lambda sid, node: _materialize_shared_subplan(backend, node, ctx, sid)
        )
        sink = sink or self.sink
        dag = getattr(plan, "physical_dag", plan)
        committed: set[str] = set()
        remaining = set(dag.topological_order())
        # 1) read waves（真实 scan 一次）。
        self._execute_read_waves(plan, dag, committed, remaining, ctx)
        # 2) 串行按拓扑序执行（shared 先物化，root 后执行）。
        for tid in dag.topological_order():
            if tid in committed:
                continue
            task = dag.tasks.get(tid)
            if task is None:
                continue
            if task.task_type == TASK_SOURCE_SCAN:
                committed.add(tid)
                continue
            if task.task_type == TASK_CSE_SHARED:
                sid = task.task_id.split(":", 1)[1]
                materialize_shared(sid, task.node_ref)
                committed.add(tid)
                self._record_timing(tid, task)
                continue
            if task.task_type == TASK_ROOT:
                result = execute_root(task)
                committed.add(tid)
                self._record_timing(tid, task)
                self._record_task_calibration(tid, task)
                if sink is not None:
                    sink.submit(task.factor_name, result)
                if result_handler is not None:
                    result_handler(task.factor_name, result)
                else:
                    self._results[task.factor_name] = result
                self._release_consumed(dag, tid, ctx)
                continue
            # planning view：直接跳过（不占资源）。
            committed.add(tid)
            self._record_timing(tid, task)
        if sink is not None:
            sink.finish()
        self._done = len(committed)
        self._scheduler_stats = {
            "mode": "DIRECT_VECTOR",
            "serial_fused": True,
            "real_task_done": len(committed),
            "virtual_task_done": 0,
            "read_waves": self._wave_summary,
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

    def _priority_score(self, dag: PhysicalFactorDAG, task: PhysicalFactorTask) -> float:
        """R33-P0-039：ready queue 优先级 = critical path + reuse + fanout - memory。"""
        try:
            cost_ms = {
                tid: float((t.resource_contract.predicted_elapsed_ms or 0.0) if t.resource_contract else 0.0)
                for tid, t in dag.tasks.items()
            }
            critical = dag.critical_path_remaining_ms(task.task_id, cost_ms)
        except Exception:
            critical = 0.0
        reuse = self._reuse_counts.get(task.task_id, len(task.consumers))
        fanout = len(task.consumers)
        memory = float(task.resource_contract.peak_memory_bytes if task.resource_contract else 0)
        return critical * 1.0 + reuse * 1000.0 + fanout * 100.0 - memory / (1024**3)

    def _release_consumed(self, dag: PhysicalFactorDAG, tid: str, ctx: Any) -> None:
        """root 完成后对其消费的 CSE sid 引用计数归零则释放（复用 batch_service）。

        R33-P0-041：释放失败**不静默**——记录 ``CSE_RELEASE_FAILURE``（telemetry +
        fail-safe cleanup），不再 ``except: pass`` 吞掉内存回收失败。
        """
        try:
            from runtime.batch_service import _release_consumed_sids

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

    def _record_timing(self, tid: str, task: PhysicalFactorTask) -> None:
        import time

        self._task_timing[tid] = {
            "op": task.op,
            "task_type": task.task_type,
            "factor_name": task.factor_name,
            "finished_at_ms": round(time.monotonic() * 1000, 3),
            "preferred_backend": task.preferred_backend,
        }

    def _auto_shard_replan(self, dag: PhysicalFactorDAG, remaining: set[str]) -> int:
        """R36 P0-020（§100）：ready task 始终无法 admission 且 shardable 时，
        先 replan smaller shard，再失败。返回 replan 的 task 数。
        """
        if not getattr(self, "_shard_replanned", None):
            self._shard_replanned: set[str] = set()
        try:
            from dataclasses import replace as _replace

            from runtime.auto_shard_planner import AutoShardPlanner

            planner = AutoShardPlanner()
            env = self.broker.resource_envelope()
            safe = max(1, int(env.safe_memory_bytes))
            candidates = [
                dag.tasks[t]
                for t in remaining
                if t in dag.tasks and dag.tasks[t].executable
                and t not in self._shard_replanned
            ]
            if not candidates:
                return 0
            plans = planner.plan_for_tasks(candidates, safe_envelope_bytes=safe)
            n = 0
            for tid, plan in plans.items():
                task = dag.tasks[tid]
                contract = getattr(task, "resource_contract", None)
                if contract is None:
                    continue
                new_contract = _replace(
                    contract,
                    peak_memory_bytes=plan.per_shard_peak_bytes,
                    output_bytes=plan.per_shard_peak_bytes,
                    estimate_basis=f"auto-shard:{plan.dimension}",
                )
                dag.tasks[tid] = rebase_task(
                    task, resource_contract=new_contract
                )
                self._shard_replanned.add(tid)
                n += 1
            return n
        except Exception:
            return 0

    def _record_task_calibration(self, tid: str, task: PhysicalFactorTask) -> None:
        """R36 P0-005/006：task 完成后把真实 elapsed/output 记入 calibration store。

        peak 内存为当前契约估计（per-task 精确 PSS attribution 需要 isolated
        calibration / concurrent marginal model，§171——第一版如实标记 basis）。
        """
        try:
            from runtime.resource_calibration_store import global_calibration_store
            from runtime.resource_shape import ResourceShapeKey

            contract = getattr(task, "resource_contract", None)
            peak = int(contract.peak_memory_bytes) if contract is not None else 0
            out = int(contract.output_bytes) if contract is not None else 0
            elapsed = self._task_timing.get(tid, {}).get("finished_at_ms", 0.0)
            global_calibration_store().record(
                ResourceShapeKey.from_task(task),
                elapsed_ms=float(elapsed),
                peak_mem=peak,
                output_bytes=out,
                spill_bytes=int(contract.spill_bytes) if contract is not None else 0,
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
