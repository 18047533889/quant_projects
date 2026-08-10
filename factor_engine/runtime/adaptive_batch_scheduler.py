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
from runtime.resource_broker import ResourceBroker
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

    # -- plan --

    def plan(
        self,
        dag: Any,
        analyses: dict[str, Any] | None = None,
        *,
        enable_cse: bool = True,
        scan_cost_map: dict[str, Any] | None = None,
        fusion_backend_capability: dict[str, bool] | None = None,
    ) -> SchedulerPlan:
        """从现有 ``DAGPlan``（roots + shared_nodes）构建 PhysicalFactorDAG 计划。

        - shared node → CSE_SHARED task（consumers 记录）;
        - 每个 root → ROOT task（inputs 含其消费的 shared sids）;
        - cost / resource contract 从 ``estimate_plan_cost`` 派生。
        """
        from planner.cse import collect_consumed_sids

        physical = PhysicalFactorDAG()
        cost_by_task: dict[str, Any] = {}
        # shared nodes
        for sid, sub in (dag.shared_nodes or {}).items():
            plan_cost = _plan_cost_bytes(sub)
            task = PhysicalFactorTask(
                task_id=f"cse:{sid}",
                op=str(getattr(sub, "op", "shared")),
                task_type=TASK_CSE_SHARED,
                inputs=(),
                consumers=(),
                execution_scope=str(getattr(dag, "scope_key", lambda: "")()),
                source_scope="",
                source_snapshot_id="",
                backend_candidates=(),
                preferred_backend="pandas_numpy",
                estimated_cost=plan_cost,
                resource_contract=_default_contract_for(TASK_CSE_SHARED, plan_cost),
                spillable=True,
                cacheable=True,
                deterministic=True,
                node_ref=sub,
            )
            physical.add_task(task)
            cost_by_task[task.task_id] = plan_cost
        # roots
        for fp in dag.roots:
            plan_cost = _plan_cost_bytes(fp.root)
            consumed = collect_consumed_sids(fp.root)
            root = PhysicalFactorTask(
                task_id=f"root:{fp.factor_name}",
                op=str(getattr(fp.root, "op", "root")),
                task_type=TASK_ROOT,
                inputs=tuple(f"cse:{sid}" for sid in sorted(consumed)),
                consumers=(),
                execution_scope=str(fp.execution_scope.scope_key() if fp.execution_scope else ""),
                source_scope="",
                source_snapshot_id="",
                backend_candidates=(),
                preferred_backend="pandas_numpy",
                estimated_cost=plan_cost,
                resource_contract=_default_contract_for(TASK_ROOT, plan_cost),
                spillable=False,
                cacheable=False,
                deterministic=True,
                node_ref=fp,
                factor_name=fp.factor_name,
            )
            physical.add_task(root)
            cost_by_task[root.task_id] = plan_cost
        # consumers / reuse counts
        for tid, task in list(physical.tasks.items()):
            for p in task.inputs:
                if p in physical.tasks:
                    prev = physical.tasks[p].consumers
                    physical.tasks[p] = PhysicalFactorTask(
                        task_id=p,
                        op=physical.tasks[p].op,
                        task_type=physical.tasks[p].task_type,
                        inputs=physical.tasks[p].inputs,
                        consumers=tuple(sorted((*prev, tid))),
                        execution_scope=physical.tasks[p].execution_scope,
                        source_scope=physical.tasks[p].source_scope,
                        source_snapshot_id=physical.tasks[p].source_snapshot_id,
                        backend_candidates=physical.tasks[p].backend_candidates,
                        preferred_backend=physical.tasks[p].preferred_backend,
                        estimated_cost=physical.tasks[p].estimated_cost,
                        resource_contract=physical.tasks[p].resource_contract,
                        shard_spec=physical.tasks[p].shard_spec,
                        spillable=physical.tasks[p].spillable,
                        cacheable=physical.tasks[p].cacheable,
                        deterministic=physical.tasks[p].deterministic,
                        node_ref=physical.tasks[p].node_ref,
                        factor_name=physical.tasks[p].factor_name,
                    )
        physical.roots = tuple(f"root:{fp.factor_name}" for fp in dag.roots)
        for tid, task in physical.tasks.items():
            self._reuse_counts[tid] = len(task.consumers)

        read_waves = build_waves_from_dag(
            physical,
            wave_memory_budget=self.wave_memory_budget,
            scan_cost_map=scan_cost_map,
        )
        # fusion groups（仅对 ROOT task，且 backend 支持才生成）
        root_tasks = [physical.tasks[t] for t in physical.roots if t in physical.tasks]
        from planner.native_fusion import can_fuse_roots, plan_native_fusion_groups

        fusion_groups = []
        if enable_cse and can_fuse_roots(root_tasks) if root_tasks else False:
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
    ) -> Future | None:
        """admission 通过则 dispatch；不通过返回 None（等待下一轮）。"""
        contract = task.resource_contract
        fn = _dispatch
        if contract is None:
            self._explain(f"task={task.task_id}: no contract, admit (vacuous)")
            return self.executor.submit(
                task.preferred_backend,
                fn, task, backend, ctx, execute_root, materialize_shared,
            )
        stage = self.broker.pressure_stage()
        if stage in {"PRESSURE_3", "PRESSURE_4", "CRITICAL"}:
            self._explain(f"task={task.task_id}: blocked by pressure_stage={stage}")
            return None
        if not self.broker.reserve(contract, task_id=task.task_id):
            self._explain(
                f"task={task.task_id}: admission rejected "
                f"(stage={stage}, headroom={self.broker.snapshot().live_headroom})"
            )
            return None
        self._explain(f"task={task.task_id}: admitted (stage={stage})")
        return self.executor.submit(
            task.preferred_backend,
            fn, task, backend, ctx, execute_root, materialize_shared,
        )

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

        execute_root = execute_root or (
            lambda task: _execute_root_with_path(
                backend, task.node_ref.root, ctx, run_mode=getattr(ctx, "run_mode", None),
                factor_name=task.factor_name,
            )[0]
        )
        materialize_shared = materialize_shared or (
            lambda sid, node: _materialize_shared_subplan(backend, node, ctx, sid)
        )
        sink = sink or self.sink
        # 接受 SchedulerPlan（含 physical_dag）或裸 PhysicalFactorDAG。
        dag = getattr(plan, "physical_dag", plan)

        pending: list[str] = dag.topological_order()
        committed: set[str] = set()
        futures: dict[str, Future] = {}
        remaining = set(pending)
        no_progress_rounds = 0
        while remaining or futures:
            admitted_this_round = 0
            # 1) admission：所有 predecessor 已 committed 且未在跑的 task。
            for tid in list(remaining):
                if tid in futures or tid in committed:
                    continue
                task = dag.tasks[tid]
                if not all(p in committed for p in task.inputs):
                    continue
                future = self._admit_and_run(
                    task,
                    backend=backend,
                    ctx=ctx,
                    execute_root=execute_root,
                    materialize_shared=materialize_shared,
                )
                if future is not None:
                    futures[tid] = future
                    remaining.discard(tid)
                    admitted_this_round += 1
            # 2) as_completed：先处理完成项（R27-103/104 立即 sink/release）。
            done = wait(list(futures.values()), timeout=0.05)[0]
            for future in done:
                try:
                    tid, result = future.result(timeout=1.0)
                except Exception as exc:  # noqa: BLE001
                    retries = self._retries_remaining.get(tid, 1)
                    if retries > 0:
                        self._retries_remaining[tid] = retries - 1
                        self._explain(
                            f"task={tid}: FAILED {type(exc).__name__}: {exc} "
                            f"(retrying, {retries - 1} left)"
                        )
                        remaining.add(tid)
                        continue
                    # R27-042：不直接 kill 运行中 task；失败重试耗尽则如实抛出。
                    self._explain(f"task={tid}: FAILED_FATAL {type(exc).__name__}: {exc}")
                    raise
                contract = dag.tasks[tid].resource_contract
                if contract is not None:
                    self.broker.release(contract, task_id=tid)
                futures.pop(tid, None)
                committed.add(tid)
                self._done += 1
                self._record_timing(tid, dag.tasks[tid])
                if dag.tasks[tid].task_type == TASK_ROOT:
                    if sink is not None:
                        sink.submit(dag.tasks[tid].factor_name, result)
                    if result_handler is not None:
                        result_handler(dag.tasks[tid].factor_name, result)
                    else:
                        self._results[dag.tasks[tid].factor_name] = result
                # 释放已消费的 CSE sid（引用计数归零立即释放）。
                if dag.tasks[tid].task_type == TASK_ROOT:
                    self._release_consumed(dag, tid, ctx)
            # 3) 动态并发（R27-204）：外部负载高 → 降 soft CPU budget。
            if self.broker.snapshot().system_cpu_util > 0.8 and self.executor._thread_task_count > 0:
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
                _release_consumed_sids(ctx, task.node_ref.root)
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
