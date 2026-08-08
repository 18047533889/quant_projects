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
