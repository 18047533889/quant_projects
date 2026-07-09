"""多因子批量依赖图：列重叠、lookback 与可并行层。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from api.factor import Factor
    from ir.analyzer import AnalysisResult


@dataclass(frozen=True)
class FactorNode:
    """单因子分析摘要。"""

    name: str
    referenced_columns: frozenset[str]
    lookback: int
    has_ts_op: bool
    has_cs_op: bool


@dataclass
class FactorBatchGraph:
    """多因子批量执行的依赖与分组视图。"""

    nodes: dict[str, FactorNode] = field(default_factory=dict)
    column_union: frozenset[str] = frozenset()
    max_lookback: int = 0
    #: 共享至少一列的因子组（用于 CSE / prefetch 提示）
    column_overlap_groups: list[list[str]] = field(default_factory=list)
    #: 列集合不相交的因子可并行（同 data_scope 内）
    parallel_layers: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factors": sorted(self.nodes.keys()),
            "column_union": sorted(self.column_union),
            "max_lookback": self.max_lookback,
            "column_overlap_groups": self.column_overlap_groups,
            "parallel_layers": self.parallel_layers,
            "nodes": {
                name: {
                    "referenced_columns": sorted(node.referenced_columns),
                    "lookback": node.lookback,
                    "has_ts_op": node.has_ts_op,
                    "has_cs_op": node.has_cs_op,
                }
                for name, node in self.nodes.items()
            },
        }


def build_factor_batch_graph(
    factors: Sequence["Factor"],
    analyses: dict[str, "AnalysisResult"],
) -> FactorBatchGraph:
    """从因子与分析结果构建批量依赖图。"""
    nodes: dict[str, FactorNode] = {}
    column_union: set[str] = set()
    max_lookback = 0

    for factor in factors:
        analysis = analyses[factor.name]
        cols = frozenset(analysis.referenced_columns)
        column_union |= cols
        max_lookback = max(max_lookback, int(analysis.lookback))
        nodes[factor.name] = FactorNode(
            name=factor.name,
            referenced_columns=cols,
            lookback=int(analysis.lookback),
            has_ts_op=bool(analysis.has_ts_op),
            has_cs_op=bool(analysis.has_cs_op),
        )

    overlap_groups = _column_overlap_groups(nodes)
    parallel_layers = _parallel_layers(nodes)

    return FactorBatchGraph(
        nodes=nodes,
        column_union=frozenset(column_union),
        max_lookback=max_lookback,
        column_overlap_groups=overlap_groups,
        parallel_layers=parallel_layers,
    )


def _column_overlap_groups(nodes: dict[str, FactorNode]) -> list[list[str]]:
    """并查集：共享列的因子归为一组。"""
    names = sorted(nodes.keys())
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    col_to_factors: dict[str, list[str]] = {}
    for name, node in nodes.items():
        for col in node.referenced_columns:
            col_to_factors.setdefault(col, []).append(name)

    for factor_names in col_to_factors.values():
        if len(factor_names) < 2:
            continue
        base = factor_names[0]
        for other in factor_names[1:]:
            union(base, other)

    groups: dict[str, list[str]] = {}
    for name in names:
        root = find(name)
        groups.setdefault(root, []).append(name)

    return [sorted(g) for g in groups.values() if len(g) > 1]


def _parallel_layers(nodes: dict[str, FactorNode]) -> list[list[str]]:
    """列集合互不相交的因子可置于同一并行层。"""
    remaining = set(nodes.keys())
    layers: list[list[str]] = []

    while remaining:
        layer: list[str] = []
        used_cols: set[str] = set()
        for name in sorted(remaining):
            cols = nodes[name].referenced_columns
            if cols & used_cols:
                continue
            layer.append(name)
            used_cols |= cols
        if not layer:
            # 全部冲突：逐个串行
            name = min(remaining)
            layer = [name]
        for name in layer:
            remaining.discard(name)
        layers.append(layer)
    return layers


def merge_analyses_column_union(analyses: Iterable["AnalysisResult"]) -> frozenset[str]:
    out: set[str] = set()
    for analysis in analyses:
        out |= analysis.referenced_columns
    return frozenset(out)
