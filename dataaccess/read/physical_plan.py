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
    req = plan.request
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

    import pyarrow.parquet as pq

    from data_access.core.exceptions import ValidationError
    from data_access.read.aggregation import AggregationItem, aggregate_minute_bundle
    from data_access.read.read_contract import (
        ReadStats,
        SqlReadLineage,
        merge_sql_data_snapshots,
    )
    from data_access.read.read_handle import ReadHandle

    anchor = plan.anchor
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

    ds_params = req.dataset_params(anchor)
    market = str(ds_params.get("market") or "") or None
    timezone = str(ds_params.get("timezone") or "") or None
    agg_handle = aggregate_minute_bundle(
        store,
        anchor,
        items,
        time_range=plan.time_range,
        instrument_filter=plan.instruments,
        params=ds_params,
        market=market,
        timezone=timezone,
    )
    agg_table = agg_handle.to_arrow()

    # 聚合锚点 → 临时 parquet（DuckDB 游标对 register() 的对象不可见，临时文件
    # 最稳、可移植）。行空间列固定为 ts / inst（aggregate_minute_bundle 输出）。
    anchor_fd, anchor_path = tempfile.mkstemp(suffix=".parquet", prefix="da_plan_agg_")
    os.close(anchor_fd)
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

    # #P0-2 统一 effective_join_specs：字段语义 → COS 契约 → 显式覆盖。
    raw_joins: dict[str, Any] = {}
    raw_joins.update(dict(req.joins or {}))
    raw_joins.update(dict(req.join_specs or {}))
    effective_specs = store._effective_join_specs(per_ds, plan.fields, raw_joins)

    pbd: dict[str, dict[str, Any]] = {
        ds: dict(req.dataset_params(ds)) for ds in plan.datasets
    }

    anchor_dsobj = store._registry.get(anchor)
    source_paths = []
    try:
        source_paths = store._prepare_dataset_read(
            anchor_dsobj,
            time_range=plan.time_range,
            params=ds_params,
            instrument_filter=plan.instruments,
        )
    except Exception:
        source_paths = []

    sql, sql_params, datasets, per_ds_paths = store._read_joined_sql(
        anchor,
        per_ds,
        effective_specs,
        time_range=plan.time_range,
        instrument_filter=plan.instruments,
        filters=req.filters,
        filters_by_dataset=getattr(req, "filters_by_dataset", None),
        params_by_dataset=pbd,
        limit=getattr(req, "limit", None),
        universe=plan.universe,
        anchor_override={
            "path": anchor_path,
            "time_column": "ts",
            "instrument_column": "inst",
            "source_paths": source_paths,
        },
        order_by=getattr(req, "order_by", None),
    )

    start_clock = _perf_counter()
    try:
        budget = store._resolve_read_budget(anchor_dsobj, None)
        table = store._engine.execute_arrow(
            sql, sql_params, deadline_ms=budget.max_elapsed_ms
        )
    finally:
        try:
            os.unlink(anchor_path)
        except OSError:
            pass
    elapsed_ms = (_perf_counter() - start_clock) * 1000

    if getattr(req, "normalize_units", False) and plan.fields:
        from data_access.read.semantic_catalog import normalize_table_units

        table = normalize_table_units(table, plan.fields)

    # 多数据集 snapshot（组合读的 lineage 基础；#14 fail-closed）
    snapshots = []
    for ds in datasets:
        try:
            dsobj = store._registry.get(ds)
            paths = per_ds_paths.get(ds) or store._prepare_dataset_read(
                dsobj,
                time_range=plan.time_range,
                params=req.dataset_params(ds),
                instrument_filter=plan.instruments,
            )
            snapshots.append(
                store._build_snapshot(
                    dataset=ds, ds=dsobj, paths=paths, params=req.dataset_params(ds)
                )
            )
        except Exception:
            if _strict_mode():
                from data_access.core.exceptions import SnapshotBuildError

                raise SnapshotBuildError(
                    f"组合读参与数据集 '{ds}' snapshot 构建失败"
                ) from None
    snapshot = (
        merge_sql_data_snapshots(snapshots, registry_hash=store.registry_fingerprint())
        if snapshots
        else None
    )
    lineage = SqlReadLineage(datasets=tuple(datasets), query_preview="composed:" + sql[:120])
    stats = ReadStats(rows=table.num_rows, bytes=table.nbytes, elapsed_ms=elapsed_ms)
    return ReadHandle(table=table, snapshot=snapshot, stats=stats, lineage=lineage)


def _perf_counter() -> float:
    import time

    return time.perf_counter()


def _strict_mode() -> bool:
    from data_access.read.query_budget import is_strict_semantics

    return is_strict_semantics()
