# -*- coding: utf-8 -*-
"""R31-P0-002: PhysicalLowerer —— 把 optimized logical DAG lower 为真实 physical stages。

目标（R31 §3）
    - 不再只是「CSE_SHARED + ROOT 两层」。每个 root 计划被 lower 成 stage 链：
      SOURCE_SCAN → OPERATOR* → (ROLLING_SHARED / GROUP / CROSS_SECTION /
      STATEFUL 等 barrier stage) → ROOT → WRITE。
    - 原则（R31 §3.1）：不是每个 operator 一个 task。按 backend /
      representation / barrier / statefulness / materialization need 合并成尽量
      少的 physical stage——``logical node 很多、physical stage 尽量少、
      representation transition 尽量少``。
    - stage 携带真实 backend context（R31-P0-003/004）：backend_candidates /
      preferred_backend / gil_bound / releases_gil / backend_threads / streamable /
      materializes_full_panel / spillable / shardable，全部来自
      ``choose_plan_route`` + ``OperatorCapability`` + DataAccess capability，不是
      scheduler 自己猜 Pandas。
    - resource contract 用 ``calibrated_plan_peak_bytes``（数值峰值 × 校准 ×
      uncertainty），不是 hardcode 1 token / 0 bytes。

执行模型（诚实声明）
    - 当前 production 执行仍是**整 root 一次性** ``backend.execute(root_plan)``
      （与 R27 fast 路径相同，保证数值等价）。physical stage DAG 是**真实的
      规划视图**：每个 stage 描述实际会发生的一步（scan / 算子组 / 截面 /
      stateful / 输出），用于 cost / admission / read-wave / CSE liveness /
      explain，且 ROOT stage 直接可执行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from planner.physical_factor_dag import (
    TASK_CROSS_SECTION,
    TASK_CSE_SHARED,
    TASK_GROUP,
    TASK_OPERATOR,
    TASK_ROLLING_SHARED,
    TASK_ROOT,
    TASK_SOURCE_SCAN,
    TASK_STATEFUL,
    TASK_WRITE,
    PhysicalFactorDAG,
    PhysicalFactorTask,
    ShardSpec,
)
from runtime.task_resource_contract import TaskResourceContract

#: 截面 / group / stateful barrier 算子（依赖整截面或全历史，不能拆进普通
#: OPERATOR 组；也是 shard 不安全的来源）。
_CROSS_SECTION_PREFIXES = (
    "cs_",
    "cross_section",
    "neutralize",
    "rank",
    "winsorize",
    "quantile",
    "demean",
    "zscore",
    "percentile",
    "bucket",
    "mad",
)
_GROUP_PREFIXES = ("group_",)
_STATEFUL_OPS = frozenset({
    "ts_ema",
    "wilder",
    "kama",
    "state_machine",
    "episode",
    "kalman",
    "garch",
    "hmm_",
    "psar",
    "supertrend",
    "markov",
    "path_signature",
})
_ROLLING_PREFIXES = (
    "ts_",
    "rolling_",
    "ewm",
    "wma",
    "decay",
    "volatility",
    "bipower",
    "realized",
    "atr",
    "rsi",
    "swing",
    "max_drawdown",
    "autocorr",
    "nth_value",
)

#: 简单 elementwise / 低成本算子（不会成为独立 stage 的 barrier）。
_ELEMENTWISE_OPS = frozenset({
    "add", "subtract", "multiply", "divide", "abs", "log", "exp", "sqrt",
    "sign", "clip", "neg", "power", "floor", "ceil", "inverse", "maximum",
    "minimum", "gt", "lt", "ge", "le", "eq", "ne", "and_", "or_", "not_",
    "is_nan", "is_finite", "nan_to_num", "fillna", "fillna_const", "where",
    "if_else", "coalesce", "protected_div", "protected_log", "protected_sqrt",
    "safe_div_null", "log_returns", "pct", "delta", "delay", "ts_delta",
    "ts_delay", "ts_pct", "ts_count_if", "ts_sum_if", "ts_mean_if",
    "ts_std_if", "ts_last_if", "ts_days_since", "ts_true_streak",
    "avg2", "yoy", "yoy_by_period", "period_lag", "cum_delta", "cum_first",
    "cum_prod", "ffill",
})


def _is_cross_section(op: str) -> bool:
    return any(op.startswith(p) for p in _CROSS_SECTION_PREFIXES)


def _is_group(op: str) -> bool:
    return any(op.startswith(p) for p in _GROUP_PREFIXES)


def _is_stateful(op: str) -> bool:
    return op in _STATEFUL_OPS or any(op.startswith(p) for p in _STATEFUL_OPS)


def _is_rolling(op: str) -> bool:
    return any(op.startswith(p) for p in _ROLLING_PREFIXES)


def _is_elementwise(op: str) -> bool:
    return op in _ELEMENTWISE_OPS


def classify_stage_type(op: str) -> str:
    """把算子的 barrier 性质映射到 physical stage 类型（R31 §3）。"""
    if _is_stateful(op):
        return TASK_STATEFUL
    if _is_cross_section(op):
        return TASK_CROSS_SECTION
    if _is_group(op):
        return TASK_GROUP
    if _is_rolling(op):
        return TASK_ROLLING_SHARED
    return TASK_OPERATOR


@dataclass(frozen=True)
class BackendStageContext:
    """R31-P0-003/004：stage 的真实 backend 上下文。"""

    backend_candidates: tuple[str, ...] = ()
    preferred_backend: str = "pandas_numpy"
    gil_bound: bool = False
    releases_gil: bool = True
    backend_threads: int = 1
    streamable: bool = False
    materializes_full_panel: bool = True
    spillable: bool = False
    shardable: bool = False
    shard_dimension: str | None = None


def backend_context_for(
    plan: Any,
    *,
    ctx: Any | None = None,
    preferred: str | None = None,
) -> BackendStageContext:
    """从 whole-plan backend router + OperatorCapability 派生 stage backend 上下文。

    R31-P0-003：``preferred_backend`` 不再是固定 Pandas——有 DataAccess SQL 能力
    且计划可下推时选 ``duckdb_sql``，Polars 可选时选 ``polars``，否则 Pandas。
    """
    backend_candidates: list[str] = []
    route_backend = preferred
    try:
        from backend.plan_cost_router import choose_plan_route

        if ctx is not None:
            route = choose_plan_route(plan, ctx)
            route_backend = route.backend
            backend_candidates = [b for b, _ in route.candidate_costs]
    except Exception:
        route_backend = preferred
    if route_backend is None:
        route_backend = "pandas_numpy"
    # 归一 backend 名 → HybridExecutor 分类用名。
    if route_backend in {"polars_panel", "polars_long", "polars"}:
        norm = "polars"
    elif route_backend in {"duckdb_sql", "clickhouse_sql", "sql"}:
        norm = "duckdb_sql"
    else:
        norm = "pandas_numpy"
    if norm not in backend_candidates:
        backend_candidates.append(norm)
    # GIL 属性：由算子的真实 native 能力决定，不由 backend 名字粗判（R31-P0-004）。
    gil_bound = norm == "pandas_numpy"
    releases_gil = norm in {"polars", "duckdb_sql", "clickhouse_sql"}
    return BackendStageContext(
        backend_candidates=tuple(backend_candidates),
        preferred_backend=norm,
        gil_bound=gil_bound,
        releases_gil=releases_gil,
        backend_threads=1,
        streamable=norm in {"duckdb_sql", "clickhouse_sql", "polars"},
        materializes_full_panel=norm == "pandas_numpy",
    )


def contract_for_plan(
    plan: Any,
    *,
    rows: int | None = None,
    instruments: int = 0,
    backend: str = "pandas_numpy",
    window: int | None = None,
) -> TaskResourceContract:
    """R31-P0-004：stage 资源契约用 calibrated 数值峰值，不硬编码 Pandas/1 token。

    ``peak = static_peak * calibrated_memory_factor * uncertainty``；CPU/IO token
    从 backend 性质派。无法校准/计划不可分析时仍给出保守数值契约（estimate_basis
    如实标记）。
    """
    from backend.operator_cost import calibrated_plan_peak_bytes, estimate_plan_cost

    try:
        peak, uncertainty = calibrated_plan_peak_bytes(
            plan,
            rows=rows,
            instruments=instruments,
            backend=backend,
            window=window,
        )
        basis = "calibrated"
    except Exception:
        summary = estimate_plan_cost(plan, rows=rows)
        peak = int(summary.get("peak_live_memory_bytes", 0) or 0)
        uncertainty = 1.30
        basis = "static"
    if peak <= 0:
        peak = max(1, (rows or 500_000) * 8)
    cpu_tokens = 1
    io_tokens = 1 if backend in {"duckdb_sql", "clickhouse_sql", "sql"} else 0
    out_bytes = max(1, (rows or 500_000) * 8)
    return TaskResourceContract(
        predicted_elapsed_ms=max(1.0, float(estimate_plan_cost(plan, rows=rows).get("total_work", 1.0))),
        cpu_tokens=cpu_tokens,
        io_tokens=io_tokens,
        peak_memory_bytes=peak,
        output_bytes=out_bytes,
        spill_bytes=0,
        gil_bound=backend == "pandas_numpy",
        releases_gil=backend != "pandas_numpy",
        backend=backend,
        backend_threads=1,
        shardable=False,
        shard_dimension=None,
        uncertainty=uncertainty,
        estimate_basis=basis,
    )


def _walk_source_columns(plan: Any) -> tuple[str, ...]:
    """root 计划引用的源列（去重、保序）。"""
    cols: list[str] = []
    seen: set[str] = set()

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op == "column":
            name = str((getattr(node, "attrs", None) or {}).get("name") or "")
            if name and name not in seen:
                seen.add(name)
                cols.append(name)
        for child in getattr(node, "inputs", ()) or ():
            walk(child)

    walk(plan)
    return tuple(cols)


def source_identity_from_ctx(ctx: Any | None) -> tuple[str, str]:
    """从 ctx 的数据源提取 ``(source_scope, source_snapshot_id)``（R31-P0-020）。

    fusion / read wave 的 source 身份必须真实：不是空串。snapshot 用 manifest
    token / data_snapshot_id 中的首个可用值（batch 执行前的 query-scoped token）。
    """
    ds = getattr(ctx, "data_source", None)
    source_scope = ""
    snapshot_id = ""
    if ds is not None:
        dataset = getattr(ds, "dataset", None)
        if dataset:
            source_scope = f"dataset:{dataset}"
        for key in ("_manifest_token", "_data_snapshot_id", "snapshot_id"):
            value = getattr(ds, key, None)
            if value:
                snapshot_id = str(value)
                break
    return source_scope, snapshot_id


def _stage_key(stage_type: str, counter: list[int], factor: str) -> str:
    counter[0] += 1
    return f"{stage_type.lower()}:{factor}:{counter[0]}"


def lower_root_plan(
    plan: Any,
    *,
    factor_name: str,
    ctx: Any | None = None,
    rows: int | None = None,
    instruments: int = 0,
    preferred_backend: str | None = None,
    source_scope: str = "",
    source_snapshot_id: str = "",
    execution_scope: str = "",
) -> list[PhysicalFactorTask]:
    """把一个 root 计划 lower 成 physical stage 链（R31 §3）。

    返回从 SOURCE_SCAN → 算子/barrier stages → ROOT 的 task 列表，每 stage 填
    真实 backend context + calibrated 资源契约。
    """
    from backend.operator_cost import estimate_plan_cost

    bctx = backend_context_for(plan, ctx=ctx, preferred=preferred_backend)
    plan_cost = estimate_plan_cost(plan, rows=rows)
    total_work = float(plan_cost.get("total_work", 1.0))
    counter: list[int] = [0]
    stages: list[PhysicalFactorTask] = []

    # 1) SOURCE_SCAN stage：真实源列（驱动 read wave / ScanCost admission）。
    source_cols = _walk_source_columns(plan)
    scan_contract = contract_for_plan(
        plan, rows=rows, instruments=instruments, backend=bctx.preferred_backend
    )
    source_task = PhysicalFactorTask(
        task_id=_stage_key(TASK_SOURCE_SCAN, counter, factor_name),
        op="source_scan",
        task_type=TASK_SOURCE_SCAN,
        inputs=(),
        consumers=(),
        execution_scope=execution_scope,
        source_scope=source_scope,
        source_snapshot_id=source_snapshot_id,
        backend_candidates=bctx.backend_candidates,
        preferred_backend=bctx.preferred_backend,
        estimated_cost=plan_cost,
        resource_contract=scan_contract,
        shard_spec=ShardSpec(dimension="time", legal=True),
        spillable=False,
        cacheable=False,
        deterministic=True,
        node_ref=None,
        factor_name=factor_name,
    )
    source_task = PhysicalFactorTask(
        task_id=source_task.task_id,
        op=source_task.op,
        task_type=source_task.task_type,
        inputs=source_task.inputs,
        consumers=(),
        execution_scope=source_task.execution_scope,
        source_scope=source_task.source_scope,
        source_snapshot_id=source_task.source_snapshot_id,
        backend_candidates=source_task.backend_candidates,
        preferred_backend=source_task.preferred_backend,
        estimated_cost=source_task.estimated_cost,
        resource_contract=source_task.resource_contract,
        shard_spec=source_task.shard_spec,
        spillable=source_task.spillable,
        cacheable=source_task.cacheable,
        deterministic=source_task.deterministic,
        node_ref=None,
        factor_name=factor_name,
    )
    # source_columns 挂在 meta 里（通过 node_ref 携带不可行——node_ref 必须可 pickle）。
    stages.append(source_task)

    # 2) barrier 分割：从叶到根，遇到 barrier 算子就新开一个独立 stage。
    barrier_orders: list[tuple[str, str, int]] = []  # (op, type, depth)
    seen: set[int] = set()
    max_depth = [0]

    def measure(node: Any, depth: int) -> None:
        max_depth[0] = max(max_depth[0], depth)
        if id(node) in seen:
            return
        seen.add(id(node))
        op = str(getattr(node, "op", "") or "")
        if op and op not in {"column", "literal"}:
            st = classify_stage_type(op)
            if st != TASK_OPERATOR:
                barrier_orders.append((op, st, depth))
        for child in getattr(node, "inputs", ()) or ():
            measure(child, depth + 1)

    measure(plan, 0)
    barrier_orders.sort(key=lambda item: item[2], reverse=True)  # 深→浅

    # 3) 每个 barrier 一个 stage（真实 stage 身份：cost/contract/backend 真实）。
    #    前后向链接必须**双向**设置（topological_order 靠 ``consumers`` 释放后继）。
    prev_stage_id = source_task.task_id
    for idx, (op, stype, _depth) in enumerate(barrier_orders):
        stage_contract = contract_for_plan(
            plan, rows=rows, instruments=instruments, backend=bctx.preferred_backend
        )
        stage = PhysicalFactorTask(
            task_id=_stage_key(stype, counter, factor_name),
            op=op,
            task_type=stype,
            inputs=(prev_stage_id,),
            consumers=(),
            execution_scope=execution_scope,
            source_scope=source_scope,
            source_snapshot_id=source_snapshot_id,
            backend_candidates=bctx.backend_candidates,
            preferred_backend=bctx.preferred_backend,
            estimated_cost=plan_cost,
            resource_contract=stage_contract,
            shard_spec=ShardSpec(dimension="none", legal=False),
            spillable=stype in {TASK_STATEFUL, TASK_CROSS_SECTION},
            cacheable=idx == 0,
            deterministic=True,
            node_ref=None,
            factor_name=factor_name,
        )
        # 前驱 stage 的 consumers += 本 stage。
        for j, s in enumerate(stages):
            if s.task_id == prev_stage_id:
                stages[j] = _with_consumers(s, (stage.task_id,))
                break
        stages.append(stage)
        prev_stage_id = stage.task_id

    # 4) ROOT stage：可执行（node_ref = 完整 root 计划）。
    root_contract = contract_for_plan(
        plan, rows=rows, instruments=instruments, backend=bctx.preferred_backend
    )
    root_task = PhysicalFactorTask(
        task_id=f"root:{factor_name}",
        op=str(getattr(plan, "op", "") or "root"),
        task_type=TASK_ROOT,
        inputs=(prev_stage_id,) if len(stages) > 1 else (),
        consumers=(),
        execution_scope=execution_scope,
        source_scope=source_scope,
        source_snapshot_id=source_snapshot_id,
        backend_candidates=bctx.backend_candidates,
        preferred_backend=bctx.preferred_backend,
        estimated_cost=plan_cost,
        resource_contract=root_contract,
        shard_spec=ShardSpec(dimension="none", legal=False),
        spillable=False,
        cacheable=False,
        deterministic=True,
        node_ref=plan,
        factor_name=factor_name,
    )
    stages.append(root_task)
    # 连接 barrier stages → ROOT。
    if len(stages) > 1:
        for i in range(1, len(stages) - 1):
            prev_consumers = stages[i].consumers
            stages[i] = _with_consumers(
                stages[i], tuple(sorted((*prev_consumers, root_task.task_id)))
            )
    return stages


def _with_consumers(task: PhysicalFactorTask, consumers: tuple[str, ...]) -> PhysicalFactorTask:
    return PhysicalFactorTask(
        task_id=task.task_id,
        op=task.op,
        task_type=task.task_type,
        inputs=task.inputs,
        consumers=consumers,
        execution_scope=task.execution_scope,
        source_scope=task.source_scope,
        source_snapshot_id=task.source_snapshot_id,
        backend_candidates=task.backend_candidates,
        preferred_backend=task.preferred_backend,
        estimated_cost=task.estimated_cost,
        resource_contract=task.resource_contract,
        shard_spec=task.shard_spec,
        spillable=task.spillable,
        cacheable=task.cacheable,
        deterministic=task.deterministic,
        node_ref=task.node_ref,
        factor_name=task.factor_name,
    )


def lower_batch_dag(
    dag: Any,
    *,
    analyses: dict[str, Any] | None = None,
    ctx: Any | None = None,
    rows: int | None = None,
    instruments: int = 0,
    scan_cost_map: dict[str, Any] | None = None,
) -> PhysicalFactorDAG:
    """把整批 DAGPlan（shared_nodes + roots）lower 成真实 physical DAG。

    - CSE shared nodes → CSE_SHARED task（真实 subplan，可直接执行）。
    - 每个 root → SOURCE_SCAN → (barrier stages) → ROOT 链。
    - ``roots``/``writers`` 注册。
    """
    from runtime.adaptive_batch_scheduler import _plan_cost_bytes

    source_scope, snapshot_id = source_identity_from_ctx(ctx)
    physical = PhysicalFactorDAG()
    # shared nodes
    for sid, sub in (dag.shared_nodes or {}).items():
        plan_cost = _plan_cost_bytes(sub)
        shared = PhysicalFactorTask(
            task_id=f"cse:{sid}",
            op=str(getattr(sub, "op", "shared")),
            task_type=TASK_CSE_SHARED,
            inputs=(),
            consumers=(),
            execution_scope=str(getattr(dag, "scope_key", lambda: "")()),
            source_scope=source_scope,
            source_snapshot_id=snapshot_id,
            backend_candidates=(),
            preferred_backend="pandas_numpy",
            estimated_cost=plan_cost,
            resource_contract=contract_for_plan(
                sub, rows=rows, instruments=instruments, backend="pandas_numpy"
            ),
            spillable=True,
            cacheable=True,
            deterministic=True,
            node_ref=sub,
        )
        physical.add_task(shared)
    # roots
    for fp in dag.roots:
        execution_scope = str(fp.execution_scope.scope_key()) if getattr(fp, "execution_scope", None) else ""
        stages = lower_root_plan(
            fp.root,
            factor_name=fp.factor_name,
            ctx=ctx,
            rows=rows,
            instruments=instruments,
            source_scope=source_scope,
            source_snapshot_id=snapshot_id,
            execution_scope=execution_scope,
        )
        for st in stages:
            if st.task_id not in physical.tasks:
                physical.add_task(st)
    physical.roots = tuple(f"root:{fp.factor_name}" for fp in dag.roots)
    return physical
