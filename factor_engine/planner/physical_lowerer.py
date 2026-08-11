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

import os
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
    SourceScopeId,
    SourceScanSpec,
    rebase_task,
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


#: stage 类型 → 合法 shard 维度（交集 barrier wins；asset > time > session）。
_STAGE_SHARD_POLICY: dict[str, tuple[str, ...]] = {
    TASK_STATEFUL: ("asset",),              # time 切分需 checkpoint 状态连续性（§93）
    TASK_CROSS_SECTION: ("time",),          # asset 分片每片 rank 会改变因子（§91）
    TASK_GROUP: ("time",),                  # group 完整性（§92）
    TASK_ROLLING_SHARED: ("asset", "time"),  # time 需 warmup overlap
    TASK_OPERATOR: ("asset", "time"),       # elementwise 平凡可切
}
#: 累计 / 全历史类算子：禁 time shard（需从最起点重放，片间不独立）。
_FULL_HISTORY_OPS = frozenset({
    "cum_delta", "cum_first", "cum_prod", "cumsum", "cum_max", "cum_min",
    "cummean", "cumcount", "ffill",
})


def single_op_shard_policy(op: str) -> tuple[str, ...] | None:
    """单个 op 的合法 shard 维度（结构化分类，不做脆弱子串匹配）。"""
    if op in _FULL_HISTORY_OPS:
        return ("asset",)
    st = classify_stage_type(op)
    return _STAGE_SHARD_POLICY.get(st, ("asset", "time"))


def _replace_node(root: Any, target: Any, replacement: Any) -> Any:
    """重建 plan 树，把 ``target`` 节点替换成 ``replacement``（PlanNode 不可变）。"""
    from dataclasses import replace as _replace

    if root is target:
        return replacement
    inputs = tuple(_replace_node(c, target, replacement) for c in getattr(root, "inputs", ()) or ())
    if inputs == getattr(root, "inputs", ()):
        return root
    return _replace(root, inputs=inputs)


def extract_stage_subplans(
    plan: Any,
    *,
    factor_name: str = "f",
) -> tuple[Any, list[tuple[str, Any]]]:
    """把整棵 plan 在 **barrier 节点** 处拆成真实可独立执行的 stage 子计划。

    P0-020（R33-P0 关键链的落地基建）：physical DAG 的 barrier stage 目前只是
    规划视图（``executable=False``），真实计算仍是 ROOT 一次 ``backend.execute(
    full_plan)``。本函数提供**真实 stage-by-stage 的管线**：

        - 从叶到根逐个提取 barrier 子树（ts_rolling / group / cross-section /
          stateful），每个子树是一个可独立执行的 stage 子计划；
        - 提取后把原树中该 barrier 节点替换为 ``plan_ref(sid)``（复用既有
          CSE plan_ref 解析机制）；
        - 返回 ``(reference_plan, [(sid, subplan), ...])``，其中
          ``backend.execute(reference_plan, ctx)`` 在 stage 子计划已物化到
          shared buffer 时 == ``backend.execute(plan, ctx)``（数值等价）。

    示例 ``cs_rank(ts_mean(x))``：
        stage:0 = ts_mean(x)
        stage:1 = cs_rank(plan_ref(stage:0))
        reference = plan_ref(stage:1)
    依次 ``backend.execute`` 各 stage（先叶后根）再执行 reference，就是真实的
    SOURCE_SCAN → barrier → barrier → ROOT 分阶段计算。

    注意：本函数是 stage 执行管线的**落地基建**；production scheduler 默认仍走
    整 root 一次性执行（数值等价，已大量验证）。把 stage 执行接成 production 默认
    （stage 级 spill / backend region routing / 逐 stage 写）是后续接线步骤。
    """
    from dataclasses import replace as _replace

    from planner.logical_plan import PlanNode

    stages: list[tuple[str, Any]] = []
    cur = plan
    while True:
        best: tuple[Any, int] | None = None

        def _deepest_barrier(node: Any, depth: int) -> None:
            nonlocal best
            op = str(getattr(node, "op", "") or "")
            if op in {"column", "literal", "plan_ref"}:
                return
            if classify_stage_type(op) != TASK_OPERATOR:
                if best is None or depth > best[1]:
                    best = (node, depth)
            for child in getattr(node, "inputs", ()) or ():
                _deepest_barrier(child, depth + 1)

        _deepest_barrier(cur, 0)
        if best is None:
            break
        barrier, _depth = best
        sid = f"stage:{factor_name}:{len(stages)}"
        stages.append((sid, barrier))
        cur = _replace_node(
            cur,
            barrier,
            PlanNode(op="plan_ref", inputs=(), attrs={"sid": sid}),
        )
    if not stages:
        return plan, []
    return cur, stages


def plan_shard_semantics(plan: Any) -> tuple[bool, str | None]:
    """递归整棵 plan 的最严格 shard 语义（交集 barrier wins，不看顶层 op 名）。

    返回 ``(shardable, shard_dimension)``：
        - 纯 elementwise → (True, "asset")
        - 含 cs/group 节点 → 交集收窄到 (time,)
        - 含 stateful/full-history → (asset,)（time 需 checkpoint / 全历史重放）
        - 交集为空（如 cs + stateful 并存）→ (False, None)（禁止任意切分）
    例如 ``add(cs_rank(x), y)``：cs 节点给出 {time}，add 给出 {asset, time}，
    交集 = {time} —— 顶层 op 是 add，但整树只能 time shard。
    """
    if plan is None:
        return False, None
    dim_sets: list[set[str]] = []
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        op = str(getattr(node, "op", "") or "")
        if op and op not in {"column", "literal"}:
            policy = single_op_shard_policy(op)
            valid = {d for d in policy if d}
            dim_sets.append(valid if valid else set())
        for child in getattr(node, "inputs", ()) or ():
            walk(child)

    walk(plan)
    if not dim_sets:
        # 纯 column/literal 计划：平凡 elementwise。
        return True, "asset"
    acc = set(dim_sets[0])
    for ds in dim_sets[1:]:
        acc &= ds
        if not acc:
            return False, None
    for preferred in ("asset", "time", "session"):
        if preferred in acc:
            return True, preferred
    return False, None


@dataclass(frozen=True)
class BackendStageContext:
    """R31-P0-003/004：stage 的真实 backend 上下文。"""

    backend_candidates: tuple[str, ...] = ()
    preferred_backend: str = "pandas_numpy"
    gil_bound: bool = False
    releases_gil: bool = True
    backend_threads: int = 1
    # R36 P0-008（§21/22/25）：admission token 必须 ≥ 实际 engine threads。
    cpu_tokens: int = 1
    streamable: bool = False
    materializes_full_panel: bool = True
    spillable: bool = False
    shardable: bool = False
    shard_dimension: str | None = None


def _engine_threads_for(backend: str) -> int:
    """R36 P0-008：backend 的真实 engine threads（DUCKDB/POLARS 从 env 读）。

    DuckDB ``SET threads=N`` 的 task 不能只 ``cpu_tokens=1``（§21：否则资源账本
    失真）。env 由 ``ExecutionResourceScope`` 按资源计划设置。
    """
    if backend not in {"duckdb_sql", "clickhouse_sql", "sql", "polars"}:
        return 1
    env_key = "DUCKDB_MAX_THREADS" if backend != "polars" else "POLARS_MAX_THREADS"
    raw = os.environ.get(env_key, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    # 未设置时保守默认：duckdb 通常多线程。
    return 4 if backend != "polars" else 2


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
    # R36 P0-008（§21/22）：actual engine threads 与 admission tokens 一致——
    # DuckDB ``threads=N`` 的 stage 声明 N 个 CPU token。
    engine_threads = _engine_threads_for(norm)
    return BackendStageContext(
        backend_candidates=tuple(backend_candidates),
        preferred_backend=norm,
        gil_bound=gil_bound,
        releases_gil=releases_gil,
        backend_threads=engine_threads,
        cpu_tokens=engine_threads,
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
    # R36 P0-008（§22）：cpu_tokens / backend_threads 与实际 engine threads 一致。
    engine_threads = _engine_threads_for(backend)
    cpu_tokens = engine_threads
    io_tokens = 1 if backend in {"duckdb_sql", "clickhouse_sql", "sql"} else 0
    out_bytes = max(1, (rows or 500_000) * 8)
    # 整棵 plan 的最严格 shard 语义（交集 barrier wins）——不再统一 False。
    shardable, shard_dim = plan_shard_semantics(plan)
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
        backend_threads=engine_threads,
        shardable=shardable,
        shard_dimension=shard_dim,
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


def source_identity_from_ctx(ctx: Any | None) -> tuple[SourceScopeId, str]:
    """从 ctx 的数据源提取 ``(SourceScopeId, source_snapshot_id)``（R31-P0-020）。

    R33-P0-005：返回 **typed** :class:`SourceScopeId`（dataset / snapshot /
    market 字段），不再拼字符串。snapshot 用 manifest token / data_snapshot_id
    中的首个可用值（batch 执行前的 query-scoped token）。
    """
    ds = getattr(ctx, "data_source", None)
    market = str(getattr(ctx, "market", "") or "")
    if ds is None:
        return SourceScopeId(dataset="", market=market), ""
    dataset = str(getattr(ds, "dataset", "") or "")
    snapshot_id = ""
    for key in ("_manifest_token", "_data_snapshot_id", "snapshot_id"):
        value = getattr(ds, key, None)
        if value:
            snapshot_id = str(value)
            break
    return SourceScopeId(dataset=dataset, snapshot_id=snapshot_id, market=market), snapshot_id


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
    scan_cost: Any | None = None,
) -> list[PhysicalFactorTask]:
    """把一个 root 计划 lower 成 physical stage 链（R31 §3 / R33-P0-007..009）。

    返回从 SOURCE_SCAN → 算子/barrier stages → ROOT 的 task 列表：
        - SOURCE_SCAN 携带真实 :class:`SourceScanSpec`（required_columns /
          time_range / instrument_scope），``executable=True``；
        - barrier stage（ROLLING/GROUP/CS/STATEFUL）是**规划视图**，
          ``executable=False``（不占真实 resource lease，R33-P0-008）；
        - ROOT ``executable=True``（node_ref = 完整 root 计划）。
    ``scan_cost``（来自 BatchDataRequest scope 的 DataAccess ScanCost）直接写进
    SOURCE_SCAN 的 resource_contract（R33-P0-034）。
    """
    from backend.operator_cost import estimate_plan_cost

    bctx = backend_context_for(plan, ctx=ctx, preferred=preferred_backend)
    plan_cost = estimate_plan_cost(plan, rows=rows)
    total_work = float(plan_cost.get("total_work", 1.0))
    counter: list[int] = [0]
    stages: list[PhysicalFactorTask] = []

    # 1) SOURCE_SCAN stage：真实源列（驱动 read wave / ScanCost admission）。
    source_cols = _walk_source_columns(plan)
    if scan_cost is not None:
        # R33-P0-034：ScanCost 直接进 source task 资源契约（真实 IO 需求）。
        scan_contract = _contract_from_scan_cost(scan_cost, plan_cost)
    else:
        scan_contract = contract_for_plan(
            plan, rows=rows, instruments=instruments, backend=bctx.preferred_backend
        )
    _ds = getattr(ctx, "data_source", None)
    spec = SourceScanSpec(
        dataset=str(getattr(_ds, "dataset", "") or "") if _ds is not None else "",
        required_columns=source_cols,
        time_range=_source_time_range_from_ctx(ctx),
        instrument_scope=_instrument_scope_from_ctx(ctx),
        snapshot_id=source_snapshot_id,
        expected_rows=int(getattr(scan_cost, "estimated_rows", 0) or 0),
        expected_bytes=int(getattr(scan_cost, "selected_bytes", 0) or 0),
        projected_bytes=int(getattr(scan_cost, "projection_bytes", 0) or 0),
        remote=bool(getattr(scan_cost, "remote", False)),
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
        source_scan_spec=spec,
        required_columns=source_cols,
        time_range=spec.time_range,
        instrument_scope=spec.instrument_scope,
        executable=True,
    )
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
    #    R33-P0-008：barrier stage 是规划视图，``executable=False``（不占真实 lease）。
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
            executable=False,
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
        executable=True,
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


def _source_time_range_from_ctx(ctx: Any | None) -> tuple[str, str] | None:
    ds = getattr(ctx, "data_source", None)
    if ds is None:
        return None
    start = getattr(ds, "start_date", None)
    end = getattr(ds, "end_date", None)
    if start is None and end is None:
        return None
    return (str(start), str(end))


def _instrument_scope_from_ctx(ctx: Any | None) -> tuple[str, ...] | None:
    ds = getattr(ctx, "data_source", None)
    if ds is None:
        return None
    filt = getattr(ds, "instrument_filter", None)
    if not filt:
        return None
    try:
        return tuple(str(x) for x in filt)
    except TypeError:
        return None


def _contract_from_scan_cost(scan_cost: Any, plan_cost: dict[str, Any]) -> Any:
    """R33-P0-034：DataAccess ScanCost 直接进 SOURCE_SCAN 资源契约。"""
    rows = max(1, int(getattr(scan_cost, "estimated_rows", 0) or 0))
    selected = int(getattr(scan_cost, "selected_bytes", 0) or 0)
    projected = int(getattr(scan_cost, "projection_bytes", 0) or 0)
    remote = bool(getattr(scan_cost, "remote", False))
    return TaskResourceContract(
        predicted_elapsed_ms=max(
            1.0,
            float(plan_cost.get("total_work", 0.0)),
        ),
        cpu_tokens=1,
        io_tokens=1 if selected or remote else 0,
        peak_memory_bytes=max(1, projected or (rows * 8)),
        output_bytes=max(1, selected or (rows * 8)),
        spill_bytes=0,
        gil_bound=False,
        releases_gil=True,
        backend="duckdb_sql" if remote else "pandas_numpy",
        backend_threads=1,
        shardable=True,
        shard_dimension="time",
        uncertainty=1.10,
        estimate_basis="scan-cost",
    )


def _with_consumers(task: PhysicalFactorTask, consumers: tuple[str, ...]) -> PhysicalFactorTask:
    return rebase_task(task, consumers=consumers)


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
    - R33-P0-033：CSE shared backend 不再硬编码 pandas——经 ``backend_context_for``
      按算子能力路由（polars/duckdb 可下推的 shared subplan 走 native）。
    - R33-P0-034：``scan_cost_map``（task/source-scope -> ScanCost）写进各 root
      SOURCE_SCAN 的资源契约。
    """
    from runtime.adaptive_batch_scheduler import _plan_cost_bytes

    source_scope, snapshot_id = source_identity_from_ctx(ctx)
    source_scope_str = source_scope.key() if isinstance(source_scope, SourceScopeId) else str(source_scope)
    physical = PhysicalFactorDAG()
    # shared nodes
    for sid, sub in (dag.shared_nodes or {}).items():
        plan_cost = _plan_cost_bytes(sub)
        bctx = backend_context_for(sub, ctx=ctx)
        shared = PhysicalFactorTask(
            task_id=f"cse:{sid}",
            op=str(getattr(sub, "op", "shared")),
            task_type=TASK_CSE_SHARED,
            inputs=(),
            consumers=(),
            execution_scope=str(getattr(dag, "scope_key", lambda: "")()),
            source_scope=source_scope_str,
            source_snapshot_id=snapshot_id,
            backend_candidates=bctx.backend_candidates,
            preferred_backend=bctx.preferred_backend,
            estimated_cost=plan_cost,
            resource_contract=contract_for_plan(
                sub, rows=rows, instruments=instruments, backend=bctx.preferred_backend
            ),
            spillable=True,
            cacheable=True,
            deterministic=True,
            node_ref=sub,
            executable=True,
        )
        physical.add_task(shared)
    # R33（嵌套 CSE）：CSE shared subplan 内部的 plan_ref（inner shared sid）必须
    # 成为 cse task 的 input 边——否则 outer shared 可能在 inner 物化前执行 →
    # plan_ref 查缓存 KeyError（调度顺序 race）。
    from planner.cse import collect_consumed_sids as _collect_sids

    for sid, sub in (dag.shared_nodes or {}).items():
        cid = f"cse:{sid}"
        inner = _collect_sids(sub)
        if inner:
            cur = physical.tasks[cid]
            physical.tasks[cid] = rebase_task(
                cur,
                inputs=tuple(sorted(set((*cur.inputs, *(f"cse:{s}" for s in inner))))),
            )
            for s in inner:
                icid = f"cse:{s}"
                if icid not in physical.tasks:
                    continue
                prev_c = physical.tasks[icid].consumers
                physical.tasks[icid] = rebase_task(
                    physical.tasks[icid],
                    consumers=tuple(sorted((*prev_c, cid))),
                )
    # roots
    for fp in dag.roots:
        execution_scope = str(fp.execution_scope.scope_key()) if getattr(fp, "execution_scope", None) else ""
        cost = _scan_cost_for_root(scan_cost_map, source_scope_str)
        stages = lower_root_plan(
            fp.root,
            factor_name=fp.factor_name,
            ctx=ctx,
            rows=rows,
            instruments=instruments,
            source_scope=source_scope_str,
            source_snapshot_id=snapshot_id,
            execution_scope=execution_scope,
            scan_cost=cost,
        )
        for st in stages:
            if st.task_id not in physical.tasks:
                physical.add_task(st)
    physical.roots = tuple(f"root:{fp.factor_name}" for fp in dag.roots)
    return physical


def _scan_cost_for_root(
    scan_cost_map: dict[str, Any] | None,
    source_scope: str,
) -> Any | None:
    """从 scan_cost_map 取 source-scope 的 ScanCost（task-id 级未 lower 时按 scope 回退）。

    R33-P0-034：ScanCost 直接写进 SOURCE_SCAN 的 resource_contract。
    """
    scan_cost_map = scan_cost_map or {}
    return scan_cost_map.get(source_scope)
