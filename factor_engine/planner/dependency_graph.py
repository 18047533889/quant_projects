"""多因子批量依赖图：真实依赖、lookback 与 locality 分组。

R27-164/232：**列重叠 ≠ 执行依赖**。``referenced_columns`` 相交绝不阻止并行
（``ts_mean(close,5)`` 与 ``ts_std(close,20)`` 共享只读 ``close``，恰恰更适合放
同一 locality wave，避免重复 scan/解码）。真正 dependency 只能来自：

    - PlanNode dependency / SourceTransform dependency
    - CSE dependency（shared sid 消费关系）
    - state/checkpoint dependency
    - materialization ordering

因此 ``parallel_layers`` 改为按**真实依赖图**分层；列重叠仅作为
``locality_groups`` hint（R27-164 明确要求保留 column overlap 作为
locality/reuse hint，不当作冲突）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from factor_engine.api.factor import Factor
    from factor_engine.ir.analyzer import AnalysisResult


@dataclass(frozen=True)
class FactorNode:
    """单因子分析摘要。

    字段：
        name: 因子名称
        referenced_columns: 公式引用的数据列集合（**locality hint**，非依赖）
        lookback: 最大历史窗口长度
        has_ts_op: 是否含时序算子
        has_cs_op: 是否含截面算子
        dependencies: 真实因子级依赖（批内另一因子 root 嵌入本因子计划时）——
            R27-164：列重叠不是依赖；此集合只含真实 subplan 嵌入。
    """

    name: str
    referenced_columns: frozenset[str]
    lookback: int
    has_ts_op: bool
    has_cs_op: bool
    dependencies: frozenset[str] = frozenset()


@dataclass
class FactorBatchGraph:
    """多因子批量执行的依赖与分组视图。

    字段：
        nodes: 因子名 → 分析摘要
        column_union: 所有因子引用列的并集
        max_lookback: 批量内最大 lookback
        locality_groups: 共享至少一列的因子组（**locality/reuse hint**，
            不是并行冲突；R27-164）
        column_overlap_groups: ``locality_groups`` 的兼容别名
        parallel_layers: 按**真实依赖图**分层的可并行层（同层因子互不依赖；
            R27-232 不再按列重叠拆层）
    """

    nodes: dict[str, FactorNode] = field(default_factory=dict)
    column_union: frozenset[str] = frozenset()
    max_lookback: int = 0
    #: 共享至少一列的因子组（locality/reuse hint —— R27-164 明确定为 hint）。
    locality_groups: list[list[str]] = field(default_factory=list)
    #: 兼容别名（历史名称保留）。
    column_overlap_groups: list[list[str]] = field(default_factory=list)
    #: 按真实依赖图分层的可并行层（R27-232：列重叠不再造成 false dependency）。
    parallel_layers: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """序列化为可 JSON 化的摘要字典。

        返回：
            含因子列表、列并集、lookback、locality 组、真实依赖并行层与各节点
            详情的字典。
        """
        return {
            "factors": sorted(self.nodes.keys()),
            "column_union": sorted(self.column_union),
            "max_lookback": self.max_lookback,
            "locality_groups": self.locality_groups,
            "column_overlap_groups": self.column_overlap_groups,
            "parallel_layers": self.parallel_layers,
            "nodes": {
                name: {
                    "referenced_columns": sorted(node.referenced_columns),
                    "lookback": node.lookback,
                    "has_ts_op": node.has_ts_op,
                    "has_cs_op": node.has_cs_op,
                    "dependencies": sorted(node.dependencies),
                }
                for name, node in self.nodes.items()
            },
        }


def build_factor_batch_graph(
    factors: Sequence["Factor"],
    analyses: dict[str, "AnalysisResult"],
) -> FactorBatchGraph:
    """从因子与分析结果构建批量依赖图。

    参数：
        factors: 待批量执行的因子序列
        analyses: 因子名 → ``Analyzer.lower`` 分析结果的映射

    返回：
        含 locality 组与真实依赖并行层的 ``FactorBatchGraph``
    """
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

    # R27-164：真实依赖只在因子 root 嵌入另一因子 root 的 subplan 时存在
    # （列重叠不是依赖）。
    deps = _derive_true_factor_dependencies(nodes, analyses)
    nodes = {
        name: FactorNode(
            name=name,
            referenced_columns=node.referenced_columns,
            lookback=node.lookback,
            has_ts_op=node.has_ts_op,
            has_cs_op=node.has_cs_op,
            dependencies=frozenset(deps.get(name, set())),
        )
        for name, node in nodes.items()
    }

    locality = _column_overlap_groups(nodes)
    layers = _dependency_layers(nodes)

    return FactorBatchGraph(
        nodes=nodes,
        column_union=frozenset(column_union),
        max_lookback=max_lookback,
        locality_groups=locality,
        column_overlap_groups=locality,
        parallel_layers=layers,
    )


def _collect_subplan_ids(root: object) -> set[int]:
    """深度优先收集计划树所有节点 id（含子节点）。

    用于检测因子 root 是否**真实嵌入**另一因子的 subplan——只有对象级 identity
    命中才算依赖（R27-164：这不是列重叠启发式）。
    """
    out: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        out.add(id(node))
        for child in getattr(node, "inputs", []) or []:
            stack.append(child)
    return out


def _derive_true_factor_dependencies(
    nodes: dict[str, FactorNode],
    analyses: dict[str, "AnalysisResult"],
) -> dict[str, set[str]]:
    """因子级真实依赖：A 依赖 B ⇔ A 的 IR 内嵌 B 的 IR root（对象 identity）。

    批内每个因子独立编译，通常**无**跨因子 root 嵌入（共享子树由 CSE 提到
    shared_nodes），因此这里自然返回空依赖集 → 单层可并行。若未来某因子 root
    真嵌入另一因子 root，则必须串行。
    """
    deps: dict[str, set[str]] = {name: set() for name in nodes}
    roots: dict[str, object] = {}
    for name in nodes:
        analysis = analyses.get(name)
        roots[name] = getattr(analysis, "ir", None)
    for name, ir in roots.items():
        if ir is None:
            continue
        subplan = _collect_subplan_ids(ir)
        for other, other_ir in roots.items():
            if other == name or other_ir is None:
                continue
            # 真实依赖：name 的 IR 内嵌 other 的 IR root（identity 命中）。
            if id(other_ir) in subplan:
                deps[name].add(other)
    return deps


def _column_overlap_groups(nodes: dict[str, FactorNode]) -> list[list[str]]:
    """并查集：共享列的因子归为一组（**locality hint**，R27-164 不是依赖）。"""
    names = sorted(nodes.keys())
    parent = {n: n for n in names}

    def find(x: str) -> str:
        """并查集 find（带路径压缩）。"""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        """合并两因子所在连通分量。"""
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


def _dependency_layers(nodes: dict[str, FactorNode]) -> list[list[str]]:
    """按**真实依赖图**分层：同层因子互不依赖（R27-164/232）。

    不再用「列集合不相交」作为并行条件（false dependency）。真实依赖来自
    ``FactorNode.dependencies``（因子 root 内嵌另一因子 root）。无依赖时全部
    因子归入单层——它们都只读 shared_nodes + source columns，可同时被调度器
    admission（内存由 ResourceBroker 约束）。
    """
    names = sorted(nodes.keys())
    deps: dict[str, set[str]] = {
        name: {d for d in node.dependencies if d in nodes} for name, node in nodes.items()
    }
    # R32-P1-051: 标准 Kahn indegree —— 反向邻接（consumers）逐边扣入度，
    # 不再每层反复扫描 entire remaining（大 DAG O(V²) → O(V+E)）。
    consumers: dict[str, list[str]] = {n: [] for n in names}
    indegree: dict[str, int] = {n: 0 for n in names}
    for name in names:
        for dep in deps[name]:
            indegree[name] += 1
            consumers[dep].append(name)
    layers: list[list[str]] = []
    ready = sorted(n for n in names if indegree[n] == 0)
    while ready:
        layers.append(list(ready))
        next_ready: list[str] = []
        for n in ready:
            for other in consumers[n]:
                indegree[other] -= 1
                if indegree[other] == 0:
                    next_ready.append(other)
        ready = sorted(next_ready)
    scheduled = {n for layer in layers for n in layer}
    if len(scheduled) != len(names):
        # 环（异常）：未排入层的节点按 name 序逐个串行，保证终态。
        remaining = [n for n in names if n not in scheduled]
        for n in remaining:
            layers.append([n])
    return layers


def merge_analyses_column_union(analyses: Iterable["AnalysisResult"]) -> frozenset[str]:
    """合并多份分析结果的引用列并集。

    参数：
        analyses: ``AnalysisResult`` 可迭代序列

    返回：
        所有 ``referenced_columns`` 的并集（不可变 frozenset）
    """
    out: set[str] = set()
    for analysis in analyses:
        out |= analysis.referenced_columns
    return frozenset(out)
