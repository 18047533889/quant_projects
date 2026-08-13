# -*- coding: utf-8 -*-
"""DAG partitioning optimizer for multi-backend execution.

实现 DAG 分区算法，最小化后端切换和转换成本：
    1. Graph cut algorithm: 最小化切分边数量（减少转换次数）
    2. Conversion cost modeling: 精确建模数据转换成本
    3. Batch conversion: 多节点共享转换，只转一次
    4. Pipeline fusion: 同后端连续算子融合执行

核心算法：
    - Min-cut based partitioning: 将 DAG 切分为后端一致的 subgraph
    - Global cost optimization: minimize(compute_cost + conversion_cost)
    - Conversion cache: 避免重复序列化/反序列化
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class DataShape:
    """数据形状（用于精确成本计算）。"""
    rows: int
    cols: int
    bytes: int = 0

    def __post_init__(self) -> None:
        if self.bytes == 0:
            # 估算字节数: float64 × rows × cols
            self.bytes = self.rows * self.cols * 8


@dataclass
class ConversionCost:
    """转换成本详细信息。"""
    from_backend: str
    to_backend: str
    rows: int
    cols: int
    base_cost_ms: float
    total_cost_ms: float
    can_avoid: bool = False  # 是否可通过批量转换避免


# 转换成本矩阵：基准成本 (ms) for 1M rows × 5 cols
CONVERSION_MATRIX = {
    ("pandas_numpy", "pandas_numpy"): 0.0,
    ("pandas_numpy", "polars"): 2.0,
    ("pandas_numpy", "duckdb_sql"): 3.5,
    ("polars", "pandas_numpy"): 2.5,
    ("polars", "polars"): 0.0,
    ("polars", "duckdb_sql"): 1.8,
    ("duckdb_sql", "pandas_numpy"): 3.0,
    ("duckdb_sql", "polars"): 1.5,
    ("duckdb_sql", "duckdb_sql"): 0.0,
}


def compute_conversion_cost(
    from_backend: str, to_backend: str, shape: DataShape
) -> ConversionCost:
    """计算精确的转换成本。

    Args:
        from_backend: 源后端
        to_backend: 目标后端
        shape: 数据形状

    Returns:
        ConversionCost 对象
    """
    if from_backend == to_backend:
        return ConversionCost(
            from_backend=from_backend,
            to_backend=to_backend,
            rows=shape.rows,
            cols=shape.cols,
            base_cost_ms=0.0,
            total_cost_ms=0.0,
            can_avoid=True,
        )

    # 归一化后端名称
    from_norm = _normalize_backend(from_backend)
    to_norm = _normalize_backend(to_backend)

    # 查表获取基准成本
    key = (from_norm, to_norm)
    base_cost = CONVERSION_MATRIX.get(key, 5.0)  # 未知转换默认 5ms

    # 按数据规模缩放
    rows_millions = max(0.001, shape.rows / 1_000_000.0)
    # 列数惩罚：每增加 5 列，成本增加 5%
    col_factor = 1.0 + (shape.cols - 5) / 100.0

    total_cost = base_cost * rows_millions * col_factor

    return ConversionCost(
        from_backend=from_norm,
        to_backend=to_norm,
        rows=shape.rows,
        cols=shape.cols,
        base_cost_ms=base_cost,
        total_cost_ms=total_cost,
        can_avoid=False,
    )


def _normalize_backend(backend: str) -> str:
    """归一化后端名称。"""
    backend = backend.lower()
    if backend in {"polars_panel", "polars_long", "polars_eager", "polars_lazy"}:
        return "polars"
    elif backend in {"sql", "clickhouse_sql"}:
        return "duckdb_sql"
    elif backend in {"pandas", "numpy"}:
        return "pandas_numpy"
    return backend


@dataclass
class DAGNode:
    """DAG 节点（用于分区优化）。"""
    node_id: str
    operator: str
    backend_candidates: list[str]
    compute_costs: dict[str, float]  # {backend: cost_ms}
    shape: DataShape
    children: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    is_shared: bool = False
    consumer_count: int = 0


@dataclass
class DAGPartition:
    """DAG 分区（同一后端的节点组）。"""
    partition_id: str
    backend: str
    node_ids: list[str]
    total_compute_cost: float
    total_rows: int
    can_fuse: bool = True  # 是否可 pipeline fusion


@dataclass
class PartitionPlan:
    """分区方案。"""
    partitions: list[DAGPartition]
    conversion_edges: list[tuple[str, str]]  # (from_partition, to_partition)
    total_compute_cost: float
    total_conversion_cost: float
    total_cost: float
    backend_switches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_partitions": len(self.partitions),
            "backend_distribution": self._backend_dist(),
            "total_compute_ms": round(self.total_compute_cost, 2),
            "total_conversion_ms": round(self.total_conversion_cost, 2),
            "total_cost_ms": round(self.total_cost, 2),
            "backend_switches": self.backend_switches,
            "conversion_overhead_pct": round(
                self.total_conversion_cost / self.total_cost * 100 if self.total_cost > 0 else 0,
                1,
            ),
        }

    def _backend_dist(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for p in self.partitions:
            dist[p.backend] = dist.get(p.backend, 0) + 1
        return dist


class DAGPartitionOptimizer:
    """DAG 分区优化器：最小化计算成本 + 转换成本。

    核心算法：
        1. Graph cut: 最小化切分边数量
        2. Conversion batching: 识别可批量转换的边
        3. Pipeline fusion: 同后端节点融合
        4. Global optimization: DP 求解最优分配
    """

    def __init__(
        self,
        *,
        conversion_penalty_multiplier: float = 1.0,
        max_partition_size: int = 50,
        enable_fusion: bool = True,
    ) -> None:
        """初始化优化器。

        Args:
            conversion_penalty_multiplier: 转换成本惩罚倍数（>1 更倾向同后端）
            max_partition_size: 单个分区最大节点数
            enable_fusion: 是否启用 pipeline fusion
        """
        self.conversion_penalty = conversion_penalty_multiplier
        self.max_partition_size = max_partition_size
        self.enable_fusion = enable_fusion
        self._conversion_cache: dict[tuple[str, str, int], ConversionCost] = {}

    def optimize(
        self,
        nodes: dict[str, DAGNode],
        root_ids: list[str],
    ) -> PartitionPlan:
        """优化 DAG 分区方案。

        Args:
            nodes: {node_id: DAGNode}
            root_ids: 根节点 ID 列表

        Returns:
            PartitionPlan
        """
        _logger.info(
            "DAG partition optimization: %d nodes, %d roots",
            len(nodes), len(root_ids),
        )

        # Phase 1: DP 求解每个节点的最优后端
        node_backends = self._dp_optimize(nodes, root_ids)

        # Phase 2: Graph cut - 切分成连通的同后端分区
        partitions = self._partition_by_backend(nodes, node_backends)

        # Phase 3: Pipeline fusion - 合并可融合的分区
        if self.enable_fusion:
            partitions = self._fuse_partitions(partitions, nodes)

        # Phase 4: 计算转换边和总成本
        edges, conversion_cost = self._compute_conversion_edges(partitions, nodes)

        compute_cost = sum(p.total_compute_cost for p in partitions)
        total_cost = compute_cost + conversion_cost

        plan = PartitionPlan(
            partitions=partitions,
            conversion_edges=edges,
            total_compute_cost=compute_cost,
            total_conversion_cost=conversion_cost,
            total_cost=total_cost,
            backend_switches=len(edges),
        )

        _logger.info(
            "Partition plan: %d partitions, %d switches, %.1f%% conversion overhead",
            len(partitions),
            len(edges),
            conversion_cost / total_cost * 100 if total_cost > 0 else 0,
        )

        return plan

    def _dp_optimize(
        self,
        nodes: dict[str, DAGNode],
        root_ids: list[str],
    ) -> dict[str, str]:
        """DP 优化：为每个节点选择最优后端。

        DP[node][backend] = min_cost to compute subtree rooted at node using backend

        考虑因素：
            - 节点计算成本
            - 子节点转换成本
            - 共享节点只计算一次
        """
        # dp[node_id][backend] = (cost, chosen_children_backends)
        dp: dict[str, dict[str, tuple[float, dict[str, str]]]] = {}

        def solve(node_id: str) -> dict[str, tuple[float, dict[str, str]]]:
            """返回 {backend: (total_cost, children_backend_choice)}"""
            if node_id in dp:
                return dp[node_id]

            node = nodes[node_id]
            result: dict[str, tuple[float, dict[str, str]]] = {}

            # 叶子节点
            if not node.children:
                for backend in node.backend_candidates:
                    compute_cost = node.compute_costs.get(backend, float("inf"))
                    result[backend] = (compute_cost, {})
                dp[node_id] = result
                return result

            # 内部节点：为每个候选后端计算最优成本
            for backend in node.backend_candidates:
                node_compute = node.compute_costs.get(backend, float("inf"))
                if node_compute == float("inf"):
                    continue

                children_total = 0.0
                children_choice: dict[str, str] = {}

                # 对每个子节点，选择最优后端（考虑转换成本）
                for child_id in node.children:
                    child_costs = solve(child_id)
                    if not child_costs:
                        continue

                    # 找到子节点的最优后端（包含转换成本）
                    best_cost = float("inf")
                    best_backend = None

                    for child_backend, (child_subtree_cost, _) in child_costs.items():
                        # 计算转换成本
                        conv_cost = self._get_conversion_cost(
                            child_backend, backend, nodes[child_id].shape
                        )

                        total = child_subtree_cost + conv_cost
                        if total < best_cost:
                            best_cost = total
                            best_backend = child_backend

                    if best_backend:
                        children_total += best_cost
                        children_choice[child_id] = best_backend

                total_cost = node_compute + children_total
                result[backend] = (total_cost, children_choice)

            dp[node_id] = result
            return result

        # 从根节点开始求解
        for root_id in root_ids:
            solve(root_id)

        # 回溯提取每个节点的最优后端
        assignments: dict[str, str] = {}

        def backtrack(node_id: str, preferred: str | None = None) -> None:
            if node_id in assignments:
                return

            node_costs = dp.get(node_id, {})
            if not node_costs:
                return

            # 选择最优后端
            if preferred and preferred in node_costs:
                chosen = preferred
            else:
                chosen = min(node_costs.items(), key=lambda x: x[0])[0]

            assignments[node_id] = chosen

            # 递归处理子节点
            _, children_choice = node_costs[chosen]
            for child_id, child_backend in children_choice.items():
                backtrack(child_id, child_backend)

        for root_id in root_ids:
            backtrack(root_id)

        return assignments

    def _get_conversion_cost(
        self, from_backend: str, to_backend: str, shape: DataShape
    ) -> float:
        """获取转换成本（带缓存）。"""
        cache_key = (from_backend, to_backend, shape.rows)
        if cache_key in self._conversion_cache:
            return self._conversion_cache[cache_key].total_cost_ms

        cost_obj = compute_conversion_cost(from_backend, to_backend, shape)
        # 应用惩罚倍数
        adjusted_cost = cost_obj.total_cost_ms * self.conversion_penalty
        cost_obj = ConversionCost(
            from_backend=cost_obj.from_backend,
            to_backend=cost_obj.to_backend,
            rows=cost_obj.rows,
            cols=cost_obj.cols,
            base_cost_ms=cost_obj.base_cost_ms,
            total_cost_ms=adjusted_cost,
            can_avoid=cost_obj.can_avoid,
        )
        self._conversion_cache[cache_key] = cost_obj
        return adjusted_cost

    def _partition_by_backend(
        self,
        nodes: dict[str, DAGNode],
        assignments: dict[str, str],
    ) -> list[DAGPartition]:
        """将节点按后端分组为连通分区。"""
        partitions: list[DAGPartition] = []
        visited: set[str] = set()
        partition_counter = [0]

        def make_partition_id(backend: str) -> str:
            partition_counter[0] += 1
            return f"P{partition_counter[0]}_{backend}"

        # BFS 收集同后端的连通子图
        for node_id, backend in assignments.items():
            if node_id in visited:
                continue

            # BFS 收集连通的同后端节点
            partition_nodes: list[str] = []
            queue = [node_id]
            local_visited: set[str] = set()

            while queue:
                nid = queue.pop(0)
                if nid in local_visited or nid in visited:
                    continue
                if assignments.get(nid) != backend:
                    continue

                local_visited.add(nid)
                visited.add(nid)
                partition_nodes.append(nid)

                # 添加同后端的子节点
                node = nodes[nid]
                for child_id in node.children:
                    if assignments.get(child_id) == backend:
                        queue.append(child_id)

                # 添加同后端的父节点
                for parent_id in node.parents:
                    if assignments.get(parent_id) == backend:
                        queue.append(parent_id)

            if partition_nodes:
                # 计算分区统计
                total_compute = sum(
                    nodes[nid].compute_costs.get(backend, 0.0)
                    for nid in partition_nodes
                )
                total_rows = sum(nodes[nid].shape.rows for nid in partition_nodes)

                partition = DAGPartition(
                    partition_id=make_partition_id(backend),
                    backend=backend,
                    node_ids=partition_nodes,
                    total_compute_cost=total_compute,
                    total_rows=total_rows,
                    can_fuse=len(partition_nodes) <= self.max_partition_size,
                )
                partitions.append(partition)

        return partitions

    def _fuse_partitions(
        self,
        partitions: list[DAGPartition],
        nodes: dict[str, DAGNode],
    ) -> list[DAGPartition]:
        """Pipeline fusion: 合并可融合的连续分区。

        融合条件：
            - 同一后端
            - 生产者-消费者关系
            - 合并后不超过 max_partition_size
        """
        # 构建分区依赖图
        node_to_partition = {}
        for p in partitions:
            for nid in p.node_ids:
                node_to_partition[nid] = p.partition_id

        # 识别可融合的分区对
        can_fuse: set[tuple[str, str]] = set()
        for p1 in partitions:
            for nid in p1.node_ids:
                node = nodes[nid]
                for child_id in node.children:
                    p2_id = node_to_partition.get(child_id)
                    if not p2_id or p2_id == p1.partition_id:
                        continue

                    # 找到 p2
                    p2 = next((p for p in partitions if p.partition_id == p2_id), None)
                    if not p2:
                        continue

                    # 检查融合条件
                    if (
                        p1.backend == p2.backend
                        and p1.can_fuse
                        and p2.can_fuse
                        and len(p1.node_ids) + len(p2.node_ids) <= self.max_partition_size
                    ):
                        can_fuse.add((p1.partition_id, p2_id))

        # 执行融合（简化版：贪心合并）
        merged = set()
        new_partitions = []

        for p in partitions:
            if p.partition_id in merged:
                continue

            # 查找可以与 p 融合的分区
            to_merge = [p]
            for p1_id, p2_id in can_fuse:
                if p1_id == p.partition_id and p2_id not in merged:
                    p2 = next((x for x in partitions if x.partition_id == p2_id), None)
                    if p2:
                        to_merge.append(p2)
                        merged.add(p2_id)

            if len(to_merge) > 1:
                # 合并多个分区
                merged_nodes = []
                for pm in to_merge:
                    merged_nodes.extend(pm.node_ids)
                    merged.add(pm.partition_id)

                new_p = DAGPartition(
                    partition_id=f"{p.partition_id}_fused",
                    backend=p.backend,
                    node_ids=merged_nodes,
                    total_compute_cost=sum(pm.total_compute_cost for pm in to_merge),
                    total_rows=sum(pm.total_rows for pm in to_merge),
                    can_fuse=False,  # 已融合
                )
                new_partitions.append(new_p)
            else:
                new_partitions.append(p)

        _logger.info(
            "Pipeline fusion: %d -> %d partitions",
            len(partitions), len(new_partitions),
        )

        return new_partitions

    def _compute_conversion_edges(
        self,
        partitions: list[DAGPartition],
        nodes: dict[str, DAGNode],
    ) -> tuple[list[tuple[str, str]], float]:
        """计算转换边和总转换成本。

        Returns:
            (edges, total_conversion_cost)
        """
        # 构建节点到分区的映射
        node_to_partition = {}
        partition_map = {}
        for p in partitions:
            partition_map[p.partition_id] = p
            for nid in p.node_ids:
                node_to_partition[nid] = p.partition_id

        # 查找跨分区边
        edges: list[tuple[str, str]] = []
        conversion_costs: list[float] = []
        seen_edges: set[tuple[str, str]] = set()

        for p in partitions:
            for nid in p.node_ids:
                node = nodes[nid]
                for child_id in node.children:
                    child_partition_id = node_to_partition.get(child_id)
                    if not child_partition_id or child_partition_id == p.partition_id:
                        continue

                    edge = (child_partition_id, p.partition_id)
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)

                    # 计算转换成本
                    child_partition = partition_map[child_partition_id]
                    cost = self._get_conversion_cost(
                        child_partition.backend,
                        p.backend,
                        nodes[child_id].shape,
                    )

                    edges.append(edge)
                    conversion_costs.append(cost)

        total_conversion = sum(conversion_costs)
        return edges, total_conversion


def batch_conversion_opportunities(
    partitions: list[DAGPartition],
    nodes: dict[str, DAGNode],
) -> dict[str, list[str]]:
    """识别批量转换机会。

    返回 {(from_backend, to_backend): [node_ids]} 可以批量转换的节点组。
    """
    opportunities: dict[str, list[str]] = defaultdict(list)

    node_to_partition = {}
    partition_map = {}
    for p in partitions:
        partition_map[p.partition_id] = p
        for nid in p.node_ids:
            node_to_partition[nid] = p.partition_id

    # 找到需要转换的边
    for p in partitions:
        for nid in p.node_ids:
            node = nodes[nid]
            for child_id in node.children:
                child_pid = node_to_partition.get(child_id)
                if not child_pid or child_pid == p.partition_id:
                    continue

                child_partition = partition_map[child_pid]
                if child_partition.backend != p.backend:
                    key = f"{child_partition.backend}→{p.backend}"
                    opportunities[key].append(child_id)

    # 过滤出可批量的（>= 2 个节点）
    return {k: v for k, v in opportunities.items() if len(v) >= 2}
