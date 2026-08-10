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
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from planner.dag_cost_model import task_priority
from planner.physical_factor_dag import (
    TASK_CSE_SHARED,
    TASK_ROOT,
    PhysicalFactorDAG,
    PhysicalFactorTask,
)
from planner.read_wave_planner import ReadWavePlan, build_waves_from_dag
from runtime.hybrid_executor import HybridExecutor, classify_backend_execution
from runtime.resource_broker import ReservationLease, ResourceBroker
from runtime.streaming_result_sink import ResultItem, StreamingResultSink

_logger = logging.getLogger(__name__)


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
            # R31-P1-039：service job 执行期间共享进程级 broker（job admission +
            # task admission 统一），避免 HTTP job 之间/与内部 workers 叠加过载。
            try:
                from service.queue import _get_service_broker

                broker = _get_service_broker()
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

    def cancel(self) -> None:
        """请求取消：后续 admission 一律拒绝；run 在无在跑任务时提前结束。"""
        self._cancelled = True
        self._explain("CANCELLED: no new task admitted; finishing running tasks")

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
                physical.tasks[rid] = PhysicalFactorTask(
                    task_id=rid,
                    op=cur.op,
                    task_type=cur.task_type,
                    inputs=tuple(sorted(set((*cur.inputs, *(f"cse:{s}" for s in consumed))))),
                    consumers=cur.consumers,
                    execution_scope=cur.execution_scope,
                    source_scope=cur.source_scope,
                    source_snapshot_id=cur.source_snapshot_id,
                    backend_candidates=cur.backend_candidates,
                    preferred_backend=cur.preferred_backend,
                    estimated_cost=cur.estimated_cost,
                    resource_contract=cur.resource_contract,
                    shard_spec=cur.shard_spec,
                    spillable=cur.spillable,
                    cacheable=cur.cacheable,
                    deterministic=cur.deterministic,
                    node_ref=cur.node_ref,
                    factor_name=cur.factor_name,
                )
                for sid in consumed:
                    cid = f"cse:{sid}"
                    if cid not in physical.tasks:
                        continue
                    prev_c = physical.tasks[cid].consumers
                    physical.tasks[cid] = PhysicalFactorTask(
                        task_id=cid,
                        op=physical.tasks[cid].op,
                        task_type=physical.tasks[cid].task_type,
                        inputs=physical.tasks[cid].inputs,
                        consumers=tuple(sorted((*prev_c, rid))),
                        execution_scope=physical.tasks[cid].execution_scope,
                        source_scope=physical.tasks[cid].source_scope,
                        source_snapshot_id=physical.tasks[cid].source_snapshot_id,
                        backend_candidates=physical.tasks[cid].backend_candidates,
                        preferred_backend=physical.tasks[cid].preferred_backend,
                        estimated_cost=physical.tasks[cid].estimated_cost,
                        resource_contract=physical.tasks[cid].resource_contract,
                        shard_spec=physical.tasks[cid].shard_spec,
                        spillable=physical.tasks[cid].spillable,
                        cacheable=physical.tasks[cid].cacheable,
                        deterministic=physical.tasks[cid].deterministic,
                        node_ref=physical.tasks[cid].node_ref,
                        factor_name=physical.tasks[cid].factor_name,
                    )

        read_waves = build_waves_from_dag(
            physical,
            wave_memory_budget=self.wave_memory_budget,
            scan_cost_map=scan_cost_map,
            scope_scan_cost_map=scope_scan_cost_map,
        )
        # fusion groups（仅对 ROOT task，且 backend 支持才生成）
        root_tasks = [physical.tasks[t] for t in physical.roots if t in physical.tasks]
        from planner.native_fusion import can_fuse_roots, plan_native_fusion_groups

        fusion_groups = []
        if enable_cse and (can_fuse_roots(root_tasks) if root_tasks else False):
            fusion_groups = plan_native_fusion_groups(
                root_tasks,
                backend_capability=fusion_backend_capability,
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
    ) -> dict[str, Any]:
        """执行 DAG：ready queue + admission + as_completed + 流式 sink。

        ``execute_root`` 缺省走 ``runtime.batch_service`` 的 root 执行（含
        backend path / production fast-path 校验）。
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
        no_progress_rounds = 0
        while remaining or futures:
            admitted_this_round = 0
            # R31-069/P1-040：取消 → 不再 admit 新 task；无在跑任务时提前结束。
            if self._cancelled and not futures:
                self._explain(
                    f"CANCELLED: stopping with {len(remaining)} pending tasks not admitted"
                )
                break
            # 0) fusion group admission：组内全部 root 就绪 → 一次 native query。
            for group in fusion_groups:
                gkey = f"fusion:{group.group_id}"
                if gkey in futures or gkey in fusion_done:
                    continue
                roots = [t for t in group.roots if t in dag.tasks]
                if not roots or any(t in futures or t in committed for t in roots):
                    continue
                if not all(all(p in committed for p in dag.tasks[t].inputs) for t in roots):
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
            # 1) admission：所有 predecessor 已 committed 且未在跑的 task。
            for tid in list(remaining):
                if tid in futures or tid in committed:
                    continue
                if tid in group_by_root:
                    continue  # 已由 fusion group 接管
                task = dag.tasks[tid]
                if not all(p in committed for p in task.inputs):
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
            # 2) as_completed：先处理完成项（R27-103/104 立即 sink/release）。
            done = wait(list(futures.values()), timeout=0.05)[0]
            for future in done:
                # R31-005：tid 由映射绑定，不再依赖 future.result() 返回值——
                # result() 抛异常时也能确定是哪个 task。
                key = future_to_task_id.pop(future, None)
                if key is None:
                    continue
                lease = future_leases.pop(future, None)
                futures.pop(key, None)
                try:
                    _ret_key, result = future.result(timeout=1.0)
                except Exception as exc:  # noqa: BLE001
                    # 释放租约 exactly once（成功/失败/取消统一 finally 语义；
                    # fusion group 的 lease 是 list）。
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
                        self._done += 1
                        self._record_timing(_tid, dag.tasks[_tid])
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
                self._done += 1
                self._record_timing(key, dag.tasks[key])
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
            # 3) 动态并发（R27-204）：外部负载高 → 降 soft CPU budget。
            # 用缓存的压力档而非强制 snapshot（force 会触发采样，热循环里慢）。
            if self.broker.pressure_stage() in {"PRESSURE_1", "PRESSURE_2"} \
                    and self.executor._thread_task_count > 0:
                self.broker.lower_soft_cpu_budget(factor=0.6)
            # 4) 无进展保护：admission 全部被拒且没有在跑 future → 等资源释放后
            #    再试；连续多轮无进展则诊断（不空转）。
            if admitted_this_round == 0 and not futures and remaining:
                no_progress_rounds += 1
                if no_progress_rounds >= 200:
                    self._explain(
                        "STUCK: no task admitted for 200 rounds; broker headroom="
                        f"{self.broker.snapshot().live_headroom} remaining={sorted(remaining)[:5]}"
                    )
                    raise RuntimeError(
                        "AdaptiveBatchScheduler made no progress: all ready tasks "
                        "rejected by ResourceBroker admission; "
                        f"broker={self.broker.summary()} remaining={sorted(remaining)[:10]}"
                    )
            else:
                no_progress_rounds = 0
        if sink is not None:
            sink.finish()
        return {
            "results": self._results,
            "task_timing": self._task_timing,
            "explanations": self._explanations,
            "done": self._done,
            "broker": self.broker.summary(),
            "executor": self.executor.summary(),
        }

    def _release_consumed(self, dag: PhysicalFactorDAG, tid: str, ctx: Any) -> None:
        """root 完成后对其消费的 CSE sid 引用计数归零则释放（复用 batch_service）。"""
        try:
            from runtime.batch_service import _release_consumed_sids

            task = dag.tasks[tid]
            if task.node_ref is not None:
                node = getattr(task.node_ref, "root", task.node_ref)
                _release_consumed_sids(ctx, node)
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

    # -- materialize --

    def materialize(
        self,
        plan: SchedulerPlan,
        *,
        backend: Any,
        ctx: Any,
        writer: Callable[[list[ResultItem]], None],
        queue_bytes: int = 4 * 1024**3,
        batch_size: int = 1,
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        """run + 流式写（R27-102..109 compute 与 write pipeline 重叠）。"""
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
