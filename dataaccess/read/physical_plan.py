"""
data_access.read.physical_plan —— DataRequest 的物理执行计划 DAG（#4）

背景
    ``DataRequest`` 声明的 ``aggregations / transforms / field_params / pit /
    frequency`` 以前只是「接口预留」，execute() 没有真正消费。本模块把一次
    逻辑数据需求编译成物理节点 DAG：

        ScanNode → FilterNode → TemporalJoinNode → AggregationNode
                → NormalizeNode → ProjectNode

    - ``build_physical_plan(...)``：纯编译，不做 IO；
    - ``ReadPlan.explain()`` 渲染节点树（P0-4 要求 explain 能直接显示这些节点）；
    - ``ReadPlan.execute()`` 按 DAG 顶层的 AggregationNode 走一次 scan 多聚合
      （aggregate_minute_bundle），其余节点沿用 read_joined / read 的执行路径。

节点语义
    ScanNode           一个数据集的物理扫描（路径/列/成本）
    FilterNode         time_range + instrument_filter + filters 谓词
    TemporalJoinNode   一个语义 join（policy/period_selection/availability）
    AggregationNode    分钟→日聚合（aggregations 列表）
    NormalizeNode      输出层单位归一化（normalize_units）
    ProjectNode        输出字段投影（fields）

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class PlanNode:
    """物理计划的一个节点。"""

    node_type: str            # scan/filter/temporal_join/aggregation/normalize/project
    label: str
    meta: dict[str, Any] = field(default_factory=dict)
    children: list["PlanNode"] = field(default_factory=list)

    def render(self, indent: int = 0, out: list[str] | None = None) -> list[str]:
        out = out if out is not None else []
        pad = "  " * indent
        extra = ""
        if self.meta:
            items = [
                f"{k}={v}"
                for k, v in self.meta.items()
                if v not in (None, [], (), "")
            ]
            if items:
                extra = "  [" + ", ".join(items) + "]"
        out.append(f"{pad}{self.node_type}: {self.label}{extra}")
        for child in self.children:
            child.render(indent + 1, out)
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PlanNode {self.node_type}: {self.label} children={len(self.children)}>"


def _trunc(values: Any, n: int = 6) -> Any:
    if isinstance(values, (list, tuple, set, frozenset)):
        vals = list(values)
        if len(vals) > n:
            return f"{vals[:n]}…(+{len(vals) - n})"
        return vals
    return values


def build_physical_plan(
    *,
    request: Any,
    fields: Sequence[Any],
    datasets: Sequence[str],
    join_policies: Mapping[str, str],
    scan_costs: Mapping[str, Any] | None = None,
) -> PlanNode:
    """把 DataRequest + 已解析字段编译成物理节点 DAG（不执行、不做 IO）。"""
    req = request

    # 1) 每数据集一个 ScanNode
    scans: dict[str, PlanNode] = {}
    for ds in datasets:
        cost = (scan_costs or {}).get(ds)
        meta: dict[str, Any] = {
            "columns": _trunc(
                [f.physical_name for f in fields if f.dataset == ds]
            )
        }
        if cost is not None:
            meta["rows"] = getattr(cost, "estimated_rows", None)
            meta["files"] = getattr(cost, "file_count", None)
        scans[ds] = PlanNode("scan", ds, meta=meta)

    # 2) 锚点：FilterNode 包住 ScanNode
    anchor = getattr(req, "anchor", None) or (datasets[0] if datasets else None)
    anchor_node = scans.get(anchor)
    if anchor_node is None and datasets:
        anchor_node = scans[datasets[0]]
    filter_meta: dict[str, Any] = {
        "time": _trunc(req.time_range) if req.time_range is not None else None,
        "instruments": _trunc(req.instruments),
        "filters": bool(req.filters),
        "universe": req.universe,
    }
    filter_node = PlanNode(
        "filter", f"anchor={anchor}", meta=filter_meta, children=[anchor_node]
    )

    # 3) TemporalJoinNode：每张右表一个（非 exact 语义 join）
    joins: list[PlanNode] = []
    for ds in datasets:
        if ds == anchor:
            continue
        policy = (join_policies or {}).get(ds, "exact")
        spec = None
        raw = dict(getattr(req, "joins", {}) or {})
        raw.update(dict(getattr(req, "join_specs", {}) or {}))
        if ds in raw:
            spec = raw[ds]
        jmeta: dict[str, Any] = {"policy": policy}
        if spec is not None and not isinstance(spec, str):
            if isinstance(spec, Mapping):
                for k in (
                    "knowledge_time",
                    "period_time",
                    "availability",
                    "period_selection",
                    "revision_order",
                    "future_cutoff",
                ):
                    if spec.get(k) is not None:
                        jmeta[k] = _trunc(spec[k])
        joins.append(
            PlanNode("temporal_join", ds, meta=jmeta, children=[scans[ds]])
        )

    # 4) 聚合：request.aggregations → AggregationNode
    agg_meta: dict[str, Any] = {}
    if getattr(req, "aggregations", None):
        agg_meta["count"] = len(req.aggregations)
        agg_meta["frequency"] = req.frequency
    if getattr(req, "transforms", None):
        agg_meta["transforms"] = _trunc(list(req.transforms.keys()))
    if getattr(req, "field_params", None):
        agg_meta["field_params"] = _trunc(list(req.field_params.keys()))
    aggregation_node = PlanNode(
        "aggregation",
        "minute_to_daily" if agg_meta else "none",
        meta=agg_meta,
        children=joins or [filter_node],
    )

    # 5) NormalizeNode
    norm_node = PlanNode(
        "normalize",
        "units" if getattr(req, "normalize_units", False) else "none",
        meta={"normalize_units": bool(getattr(req, "normalize_units", False))},
        children=[aggregation_node],
    )

    # 6) ProjectNode（根）
    root = PlanNode(
        "project",
        f"{len(fields)} fields",
        meta={
            "pit": bool(getattr(req, "pit", False)),
            "frequency": getattr(req, "frequency", None),
            "fields": _trunc([getattr(f, "logical_name", None) for f in fields]),
        },
        children=[norm_node],
    )
    return root


def render_plan(root: PlanNode | None) -> str:
    """渲染物理计划树（ReadPlan.explain() 用）。"""
    if root is None:
        return "  (无物理计划)"
    lines = ["  " + line for line in root.render()]
    return "\n".join(lines)


def plan_uses_aggregation(request: Any) -> bool:
    """DataRequest 是否声明了分钟→日聚合（execute 走 bundle 路径）。"""
    return bool(getattr(request, "aggregations", None))


# ---------------------------------------------------------------------------
# #11 PhysicalPlanExecutor —— 节点式执行（聚合 + join 组合）
# ---------------------------------------------------------------------------

_QI = '"{}"'


def _qi(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def execute_physical_plan(store: Any, plan: Any) -> Any:
    """#11 节点式执行：聚合（Scan→Aggregate）与 join（TemporalJoin）组合。

    目前真正需要组合的是「分钟锚点聚合 + 财务/行业 join」：旧 ``ReadPlan.execute()``
    遇到 aggregations 会提前返回，join 被丢弃。本函数把锚点聚合结果注册成
    DuckDB 虚拟表，再对右表做 ASOF/exact join——一次 SQL 输出。

    非组合场景（无聚合 / 单数据集）返回 None，由 ``ReadPlan.execute()`` 走
    现有 read/read_joined/aggregate_minute_bundle 路径（节点语义等价）。
    """
    # #4 只消费 plan 编译时冻结的 compiled（execute 已不读活的 request）。
    req = plan.compiled if plan.compiled is not None else plan.request
    if not getattr(req, "aggregations", None) or len(plan.datasets) <= 1:
        return None
    return _execute_composed(store, plan, req)


def _execute_composed(store: Any, plan: Any, req: Any) -> Any:
    """#P0-1 聚合锚点 → 统一 JoinCompiler（``_read_joined_sql``）组合执行。

    删除旧 ``_join_aggregated_anchor`` 的「第二套 join 语义」：聚合结果物化到
    临时 parquet 作为 ``anchor_override``，交给与 ``read_joined`` **同一个**
    ``store._read_joined_sql`` 编译——period_selection / revision 去重 /
    session availability / seed+window / future_cutoff / fanout 守卫 / universe /
    filters_by_dataset / budget 全部与普通 read_joined 逐字一致。
    """
    import os
    import tempfile
    import uuid

    import pyarrow.parquet as pq

    from data_access.core.exceptions import (
        CapabilityUnavailableError,
        SourceResolutionError,
        ValidationError,
    )
    from data_access.read.aggregation import AggregationItem, aggregate_minute_bundle
    from data_access.read.read_contract import (
        ReadStats,
        SqlReadLineage,
        merge_sql_data_snapshots,
    )
    from data_access.read.read_handle import ReadHandle
    from data_access.runtime.prepared_read import (
        DeadlineContext,
        EmptyPhysicalScope,
        reset_deadline_context,
    )

    anchor = plan.anchor
    # R39 P0 #33：组合读 deadline 从**最外层入口**开始——minute aggregation / path
    # resolution / snapshot / join SQL compile 全部计入同一请求预算（旧代码
    # ``_composed_deadline_at`` 在这些 prep 之后才建，早期阶段完全不在预算内）。
    budget = store._resolve_sql_budget(list(plan.datasets), None)
    deadline_ctx = DeadlineContext.start(budget, source="composed_read")
    _deadline_token = deadline_ctx.enter()
    items: list[AggregationItem] = []
    for raw in req.aggregations:
        if isinstance(raw, AggregationItem):
            items.append(raw)
            continue
        if isinstance(raw, Mapping):
            fld = str(raw.get("field") or raw.get("column") or "")
            if not fld:
                raise ValidationError(f"aggregations 项缺少 field: {raw!r}")
            # 兼容限定名：minute_ds.Volume → Volume（aggregate_minute_bundle 只认
            # 锚点数据集的物理列名）。
            fld = fld.split(".", 1)[1] if "." in fld else fld
            items.append(
                AggregationItem(
                    field=fld,
                    spec=raw.get("spec"),
                    output_name=(
                        str(raw["output_name"]) if raw.get("output_name") else None
                    ),
                )
            )
        else:
            raise ValidationError("aggregations 项必须是 AggregationItem 或 dict")
    if not items:
        raise ValidationError("aggregations 为空")

    # #4 compiled（execute_physical_plan 已传 req；防御兜底）
    if req is None:
        req = plan.compiled if plan.compiled is not None else plan.request
    ds_params = req.dataset_params(anchor)
    market = str(ds_params.get("market") or "") or None
    timezone = str(ds_params.get("timezone") or "") or None
    # #2 静态 universe：先展开成 instruments（与 read_joined 非组合路径一致），
    # 时变 universe 才下沉成 join 内的 INNER JOIN（_read_joined_sql 处理）。
    insts = plan.instruments
    time_varying = bool(getattr(req, "time_varying_universe", True))
    if req.universe and not time_varying:
        insts = store._resolve_universe_instruments(req.universe, plan.time_range, insts)
    if plan.snapshot_policy in {"pin", "verified_fail_if_changed"}:
        from data_access.core.exceptions import SnapshotBuildError

        raise SnapshotBuildError(
            f"snapshot_policy={plan.snapshot_policy} 的 composed aggregation 尚不能把"
            "已核验 publisher/physical scope 贯穿到 aggregation terminal scan；"
            "为避免 verify→scan TOCTOU 已 fail closed。"
        )
    agg_handle = aggregate_minute_bundle(
        store,
        anchor,
        items,
        time_range=plan.time_range,
        instrument_filter=insts,
        params=ds_params,
        market=market,
        timezone=timezone,
    )
    agg_table = agg_handle.to_arrow()

    # R29-P0 #204：聚合锚点优先注册成**共享 pool 数据库里的持久视图**（跨连接
    # 可见，免写临时 parquet、免压缩/解压 round-trip）；pool 数据库不可用时回退
    # 临时 parquet（旧机制）。行空间列固定为 ts / inst（aggregate_minute_bundle
    # 输出）。
    anchor_view: str | None = None
    anchor_path: str | None = None
    try:
        anchor_view = f"da_composed_anchor_{uuid.uuid4().hex[:8]}"
        store._engine.register_anchor_relation(anchor_view, agg_table)
    except CapabilityUnavailableError:
        # R39 P0 #43：只有**能力缺失**（共享 pool 数据库不可用）才回退临时 parquet；
        # 数据库损坏 / schema 错误 / pool 状态错误必须原样传播，禁止 ``except Exception``
        # 一把抓当能力缺失回退。
        anchor_view = None
        _anchor_fd, anchor_path = tempfile.mkstemp(
            suffix=".parquet", prefix="da_plan_agg_"
        )
        os.close(_anchor_fd)
        pq.write_table(agg_table, anchor_path)

    # 锚点投影 = ts/inst + 聚合输出列（非 ts/inst 的物化列）
    agg_cols = [c for c in agg_table.column_names if c not in ("ts", "inst")]
    per_ds: dict[str, list[str]] = dict(plan.per_dataset_columns)
    anchor_out = [
        c
        for c in per_ds.get(anchor, [])
        if c not in ("ts", "inst")
    ]
    for c in agg_cols:
        if c not in anchor_out:
            anchor_out.append(c)
    per_ds[anchor] = [c for c in anchor_out if c in agg_cols]

    # #P0-2/#P1-final closure 4 统一 effective_join_specs：直接用 plan 阶段冻结的
    # ``plan.join_specs_effective``（字段语义 → COS 契约 → 显式覆盖已由
    # store.plan 编译），**不再重新推导**——组合执行与普通 read_joined / explain
    # 消费同一份 join 语义（旧代码在这里重算，可能与 plan 时刻的冻结 spec 漂移）。
    effective_specs = dict(plan.join_specs_effective or {})
    if not effective_specs:
        raw_joins: dict[str, Any] = {}
        raw_joins.update(dict(req.joins or {}))
        raw_joins.update(dict(req.join_specs or {}))
        effective_specs = store._effective_join_specs(per_ds, plan.fields, raw_joins)

    pbd: dict[str, dict[str, Any]] = {
        ds: dict(req.dataset_params(ds)) for ds in plan.datasets
    }

    anchor_dsobj = store._registry.get(anchor)
    source_paths: list[str] = []
    try:
        _src = store._prepare_dataset_read(
            anchor_dsobj,
            time_range=plan.time_range,
            params=ds_params,
            instrument_filter=insts,
        )
    except Exception as exc:
        # R39 P0 #44：解析失败 ≠ 空数据集——抛 typed SourceResolutionError，绝不
        # 用 ``source_paths=[]`` 静默冒充「合法空」。
        raise SourceResolutionError(
            f"组合读锚点 '{anchor}' 物理读取范围解析失败（R39 P0 #44）："
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not _src:
        # R39 P0 #44：legitimately empty dataset → typed EmptyPhysicalScope（list
        # 子类，下游 list()/迭代兼容，但语义明确是「空」而非「失败」）。
        source_paths = EmptyPhysicalScope(
            dataset=anchor, reason="no matching partitions for composed read"
        )
    else:
        source_paths = list(_src)

    sql, sql_params, datasets, per_ds_paths = store._read_joined_sql(
        anchor,
        per_ds,
        effective_specs,
        time_range=plan.time_range,
        instrument_filter=insts,
        filters=req.filters,
        filters_by_dataset=getattr(req, "filters_by_dataset", None),
        params_by_dataset=pbd,
        limit=getattr(req, "limit", None),
        # #2 时变 universe 才下沉 INNER JOIN；静态 universe 已展开进 insts
        universe=(req.universe if time_varying else None),
        anchor_override={
            # R29-P0 #204：有持久视图用 relation（免临时 parquet）；否则 path 回退。
            "relation": anchor_view,
            "path": anchor_path,
            "time_column": "ts",
            "instrument_column": "inst",
            "source_paths": source_paths,
        },
        order_by=getattr(req, "order_by", None),
    )

    # 多数据集 snapshot（组合读的 lineage 基础；#14 fail-closed）——先构建，
    # 与执行顺序无关，流式/物化共用同一份。
    snapshots = []
    for ds in datasets:
        try:
            dsobj = store._registry.get(ds)
            paths = per_ds_paths.get(ds) or store._prepare_dataset_read(
                dsobj,
                time_range=plan.time_range,
                params=req.dataset_params(ds),
                instrument_filter=insts,
            )
            files = store._files_for_snapshot(dsobj, ds, paths)
            publisher = plan.plan_publisher_snapshots.get(ds)
            resolved = store._pipeline.resolve_snapshot(
                ds,
                files=files,
                paths=paths,
                policy=plan.snapshot_policy,
                pin_snapshot_id=(
                    str(publisher["content_digest"])
                    if publisher is not None
                    else None
                ),
            )
            if publisher is not None and (
                resolved.source_generation != publisher.get("source_generation")
                or resolved.content_digest != publisher.get("content_digest")
            ):
                from data_access.core.exceptions import ValidationError

                raise ValidationError(
                    f"dataset={ds!r} publisher source_generation/content_digest "
                    "在 plan 后变化；拒绝执行 composed read。"
                )
            snapshots.append(
                store._build_snapshot(
                    dataset=ds,
                    ds=dsobj,
                    paths=paths,
                    params=req.dataset_params(ds),
                    files=files,
                )
            )
        except Exception as exc:
            if plan.snapshot_policy in {"verified_fail_if_changed", "pin"}:
                raise
            if _strict_mode():
                from data_access.core.exceptions import SnapshotBuildError

                raise SnapshotBuildError(
                    f"组合读参与数据集 '{ds}' snapshot 构建失败"
                ) from exc
    snapshot = (
        merge_sql_data_snapshots(snapshots, registry_hash=store.registry_fingerprint())
        if snapshots
        else None
    )
    lineage = SqlReadLineage(datasets=tuple(datasets), query_preview="composed:" + sql[:120])

    # #P1-final closure 4 预算 parity：与 read_joined 一致，合并**全部参与数据集**
    # 的 query_policy 取最严——旧代码只取 anchor 的 policy（``_resolve_read_budget
    # (anchor_dsobj, None)``），非 anchor 表 max_rows/max_scan_files 全被绕过。
    # R39 P0 #33：budget 已在最外层入口解析；deadline 由 DeadlineContext 持有，
    # resolve/snapshot/join SQL compile 全部计入同一请求预算。
    if deadline_ctx is not None:
        deadline_ctx.check(context="composed_read(snapshot/join-sql)")
    _composed_deadline_at = deadline_ctx.deadline_at if deadline_ctx is not None else None

    # #P1-final closure 4 max_scan_files：组合读同样对每张参与表的实际匹配文件数
    # 做硬限制（read_joined 在 snapshot 阶段 enforce，这里逐 dataset 补上）。
    # **注意**：只有「路径解析失败」才在非严格模式下放行；``_enforce_scan_files``
    # 抛出的 max_scan_files ValidationError 是预算强制，必须始终传播。
    for ds in datasets:
        try:
            paths_ds = per_ds_paths.get(ds)
            if not paths_ds:
                paths_ds = store._prepare_dataset_read(
                    store._registry.get(ds),
                    time_range=plan.time_range,
                    params=req.dataset_params(ds),
                    instrument_filter=insts,
                )
        except Exception:
            if _strict_mode():
                raise
            # 非严格模式：路径解析失败不拦组合读（snapshot 已尽力构建）
            continue
        store._enforce_scan_files(budget, paths_ds)

    # R29-P0 #197：组合执行走**统一 ReadPipeline**——admit → verify_before →
    # execute(duckdb_slot + counters.execute) → verify_after → release。旧代码
    # 直接 ``engine.execute_*`` 绕过 governor reservation / snapshot verify，
    # 组合读是 PreparedRead 之外的第二条旁路（与 read_joined / read_factors 对齐）。
    from data_access.security.execution_context import current_principal

    join_snapshot = None
    if snapshots:
        all_files: list[Any] = []
        for s in snapshots:
            all_files.extend(list(getattr(s, "files", ()) or ()))
        join_snapshot = store._pipeline.resolve_snapshot(
            anchor, files=all_files, paths=source_paths or None
        )
        store._pipeline.enforce_budget(budget, snapshot=join_snapshot)
    ctx_principal = current_principal() or store._principal
    pid = getattr(ctx_principal, "principal_id", "unknown")
    # R39 P0 #35：组合读同样形成真实内存 P99 admission——先
    # QueryBudget.max_estimated_memory 硬门（执行前拒绝），再请求 governor lease。
    _estimated_memory = 0
    try:
        _est_bytes = (
            join_snapshot.total_bytes if join_snapshot is not None else 0
        )
        _estimated_memory = max(int(_est_bytes * 1.5), int(agg_table.nbytes or 0))
    except Exception:
        _estimated_memory = 0
    from data_access.read.query_budget import enforce_memory_budget

    enforce_memory_budget(budget, estimated_memory=_estimated_memory)
    res = store._pipeline.admit(
        request_identity=f"composed:{anchor}:{uuid.uuid4().hex[:12]}",
        principal_id=pid,
        estimated_scan_bytes=(
            join_snapshot.total_bytes if join_snapshot is not None else 0
        ),
        estimated_memory=_estimated_memory,
        remote_requests=(
            sum(
                1
                for o in join_snapshot.objects
                if str(o.uri).startswith(("s3://", "cos://"))
            )
            if join_snapshot is not None
            else 0
        ),
    )

    need_normalize = bool(getattr(req, "normalize_units", False) and plan.fields)
    result_mode = str(getattr(req, "result", "auto") or "auto")

    if result_mode == "stream" and not need_normalize:
        # #2 组合路径也 honor result="stream"：直接跑 engine reader，不物化整表。
        # #P1-final closure 4：流式分支累计 max_rows/max_result_bytes 硬限制
        # （逐 batch 累计，超限立即抛，不物化到超限才发现）。
        from data_access.read.query_budget import enforce_stream_budget

        # R29-P0 #197/#201：组合流式同样 verify_before → execute（engine 内部持
        # governor 同一信号量）→ verify_after。
        store._pipeline.verify_before(join_snapshot)
        store._pipeline.counters.execute += 1
        # R29-P0 #204（沿用物化分支）：anchor 视图活在**共享 pool 数据库**里——
        # 无显式 budget deadline 时也必须给默认 deadline 强制走 pool 连接
        # （execute_reader 无 deadline 落到 self._conn `:memory:`，看不到视图）。
        _reader_deadline_ms = _remaining_ms(budget, _composed_deadline_at)
        if _reader_deadline_ms is None and anchor_view is not None:
            _reader_deadline_ms = 30_000.0  # 默认 30s 请求预算（强制 pool 路由）
        reader = store._engine.execute_reader(
            sql,
            sql_params,
            batch_size=100_000,
            deadline_ms=_reader_deadline_ms,
        )
        acc_rows = 0
        acc_bytes = 0
        stream_start = _perf_counter()

        def _gen() -> Any:
            nonlocal acc_rows, acc_bytes
            try:
                for batch in reader:
                    # R39 P0 #37：流式逐 batch 检查绝对 deadline（执行中强制终止）。
                    if deadline_ctx is not None:
                        deadline_ctx.check(context="composed_stream")
                    acc_rows += batch.num_rows
                    acc_bytes += int(getattr(batch, "nbytes", 0) or 0)
                    enforce_stream_budget(
                        budget,
                        total_rows=acc_rows,
                        total_bytes=acc_bytes,
                        elapsed_ms=(_perf_counter() - stream_start) * 1000,
                    )
                    yield batch
                store._pipeline.verify_after(join_snapshot)
            finally:
                try:
                    close = getattr(reader, "close", None)
                    if close is not None:
                        close()
                except Exception:
                    pass
                # R39 P0 #38：reservation 释放 + anchor 清理挂到 ReadHandle cleanup
                # 回调（create-then-close 从不迭代也释放）；生成器只负责 reader 关闭
                # 与 verify_after。reservation release 幂等，两条路径双保险。

        def _composed_cleanup() -> None:
            _cleanup_anchor(store, anchor_view, anchor_path)
            store._pipeline.release_reservation(res)

        stats = ReadStats(rows=0, bytes=0, elapsed_ms=0.0)
        handle = ReadHandle(
            stream=_gen(),
            snapshot=snapshot,
            stats=stats,
            lineage=lineage,
            _cleanup_callbacks=[_composed_cleanup],
            _deadline=deadline_ctx,
        )
        reset_deadline_context(_deadline_token)
        return handle

    start_clock = _perf_counter()
    try:
        # R29-P0 #197/#201：组合物化同样 verify_before → execute（engine 内部持
        # governor 同一信号量）→ verify_after。
        if deadline_ctx is not None:
            deadline_ctx.check(context="composed_read(execute)")
        store._pipeline.verify_before(join_snapshot)
        store._pipeline.counters.execute += 1
        # R29-P0 #204：anchor 视图活在**共享 pool 数据库**里（跨连接可见）——
        # 无显式 budget deadline 时也必须给个默认 deadline 强制走 pool 连接
        # （execute_arrow 无 deadline 会落到 self._conn `:memory:`，看不到视图）。
        _deadline_ms = _remaining_ms(budget, _composed_deadline_at)
        if _deadline_ms is None and anchor_view is not None:
            _deadline_ms = 30_000.0  # 默认 30s 请求预算（强制 pool 路由）
        table = store._engine.execute_arrow(
            sql, sql_params, deadline_ms=_deadline_ms
        )
        store._pipeline.verify_after(join_snapshot)
    finally:
        _cleanup_anchor(store, anchor_view, anchor_path)
        store._pipeline.release_reservation(res)
    elapsed_ms = (_perf_counter() - start_clock) * 1000

    if need_normalize:
        from data_access.read.semantic_catalog import normalize_table_units

        table = normalize_table_units(table, plan.fields)

    # #P1-final closure 4：物化分支补 enforce_arrow_budget（与 aggregate_minute_bundle
    # / read 一致）——旧组合路径只有 deadline，没有 max_rows/max_result_bytes。
    from data_access.read.query_budget import enforce_arrow_budget

    enforce_arrow_budget(budget, table, elapsed_ms=elapsed_ms)
    stats = ReadStats(rows=table.num_rows, bytes=table.nbytes, elapsed_ms=elapsed_ms)
    reset_deadline_context(_deadline_token)
    return ReadHandle(table=table, snapshot=snapshot, stats=stats, lineage=lineage)


def _perf_counter() -> float:
    import time

    return time.perf_counter()


def _cleanup_anchor(store: Any, anchor_view: str | None, anchor_path: str | None) -> None:
    """R29-P0 #204：组合锚点清理——持久视图 DROP / 临时 parquet unlink（都幂等）。"""
    import os

    if anchor_view:
        try:
            store._engine.drop_anchor_relation(anchor_view)
        except Exception:
            pass
    if anchor_path:
        try:
            os.unlink(anchor_path)
        except OSError:
            pass


def _remaining_ms(budget: Any, deadline_at: float | None) -> float | None:
    """R29-P0 #201：全请求 absolute deadline 的剩余时间（ms）。

    ``deadline_at`` 缺省 → 返回 budget.max_elapsed_ms（保持旧语义）；
    已过 → 抛 DeadlineExceeded（fail-fast，不跑完才报超时）。
    """
    if deadline_at is None:
        return getattr(budget, "max_elapsed_ms", None)
    import time as _tm

    remaining = deadline_at - _tm.monotonic()
    if remaining <= 0:
        from data_access.core.exceptions import DeadlineExceeded

        raise DeadlineExceeded(
            "组合读已超过请求 deadline（R29-P0 #201：resolve/snapshot 计入请求预算）。"
        )
    return remaining * 1000.0


def _strict_mode() -> bool:
    from data_access.read.query_budget import is_strict_semantics

    return is_strict_semantics()
