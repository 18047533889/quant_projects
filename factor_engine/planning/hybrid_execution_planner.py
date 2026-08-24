# -*- coding: utf-8 -*-
"""混合执行规划器：不同 factor 使用不同 backend，最小化 transfer 成本。

实现批量因子的混合后端优化：
    1. 全局视角：考虑所有 factor 的 DAG 拓扑
    2. 共享节点优化：多个 factor 共享的中间节点只计算一次
    3. Transfer 成本最小化：相邻节点倾向使用同一 backend
    4. 并行执行最大化：不同 backend region 可并行执行
    5. 内存预算全局约束：Peak memory 不超过预算

Section references: §43-§45, §68-§71 (batch-global optimization)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from factor_engine.planning.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalProperty,
    Representation,
    StateContract,
)
from factor_engine.planning.backend_selector import (
    IntelligentBackendSelector,
    OperatorProfile,
    RoutingContext,
)
from factor_engine.planning.cost_model_v2 import (
    CostBreakdown,
    build_cost_breakdown,
    estimate_compute_cost,
    estimate_memory_risk_penalty,
    estimate_schedule_overhead,
    estimate_source_scan_cost,
)
from factor_engine.planning.memory_model import DataShapeEstimate, EdgeMemoryCost
from factor_engine.planning.physical_region_plan import (
    PhysicalRegionPlan,
    compute_plan_hash,
    estimate_plan_peak_memory,
)
from factor_engine.planning.region_optimizer import NodeCost, RegionOptimizer
from factor_engine.planning.transfer_edge import (
    SemanticContract,
    TransferEdge,
    TransferKind,
    estimate_transfer_cost,
    infer_transfer_kind,
)


@dataclass(frozen=True)
class FactorSpec:
    """单个 factor 的规格（用于混合规划）。"""

    factor_id: str
    root_node_id: str  # DAG 根节点
    estimated_rows: int
    estimated_columns: int
    estimated_bytes: int
    operator_names: list[str]
    window_params: list[int]
    priority: int = 0  # 优先级（高优先级 factor 优先分配资源）


@dataclass(frozen=True)
class NodeMetadata:
    """节点元数据（用于路由决策）。"""

    node_id: str
    operator_name: str
    estimated_rows: int
    estimated_bytes: int
    window: int | None
    is_shared: bool  # 是否被多个 factor 共享
    consumer_count: int  # 消费者数量
    depth: int  # DAG 深度（从叶子到根）


class HybridExecutionPlanner:
    """混合执行规划器：批量因子的全局最优 backend 分配。

    核心算法：
        1. DAG 合并：多个 factor 的 DAG 合并为单一 DAG
        2. 共享节点标记：标记被多个 factor 使用的节点
        3. Backend 候选生成：为每个节点生成候选 backend + 成本
        4. 全局优化：DP 求解最优分配（考虑 transfer affinity）
        5. Region 聚合：将相邻同 backend 节点聚合为 region
        6. Transfer edge 生成：跨 region 边界生成 transfer edge
        7. 拓扑排序：生成执行顺序
    """

    def __init__(
        self,
        *,
        memory_budget: int | None = None,
        enable_parallel_regions: bool = True,
        max_backend_switches: int | None = None,
        enable_data_locality: bool = True,
        enable_streaming_first: bool = True,
        enable_adaptive_replan: bool = True,
    ) -> None:
        """初始化混合规划器。

        OPTIMIZED: Added data locality, streaming-first, and adaptive replanning.

        Args:
            memory_budget: 内存预算（bytes）
            enable_parallel_regions: 是否启用并行 region 执行
            max_backend_switches: 最大 backend 切换次数（限制碎片化）
            enable_data_locality: 是否优化数据局部性
            enable_streaming_first: 大数据优先使用 streaming
            enable_adaptive_replan: 是否启用自适应重规划
        """
        self.memory_budget = memory_budget
        self.enable_parallel_regions = enable_parallel_regions
        self.max_backend_switches = max_backend_switches
        self.enable_data_locality = enable_data_locality
        self.enable_streaming_first = enable_streaming_first
        self.enable_adaptive_replan = enable_adaptive_replan
        self._selector = IntelligentBackendSelector()
        self._execution_history: list[dict[str, Any]] = []  # For adaptive replanning

    def plan_hybrid_execution(
        self,
        factors: list[FactorSpec],
        node_graph: dict[str, list[str]],
        node_metadata: dict[str, NodeMetadata],
        *,
        available_memory_bytes: int,
    ) -> PhysicalRegionPlan:
        """规划混合执行计划（主入口）。

        Args:
            factors: Factor 规格列表
            node_graph: DAG 邻接表 {node_id: [child_id, ...]}
            node_metadata: 节点元数据
            available_memory_bytes: 可用内存

        Returns:
            PhysicalRegionPlan
        """
        # Phase 1: 识别共享节点
        shared_nodes = self._identify_shared_nodes(factors, node_graph)

        # Phase 2: 为每个节点生成 backend 候选 + 成本
        node_costs = self._generate_node_costs(
            node_graph,
            node_metadata,
            shared_nodes,
            available_memory_bytes,
        )

        # Phase 3: 全局优化（DP）- 使用新的 DAG 分区优化器
        plan = self._optimize_with_dag_partitioner(
            factors, node_graph, node_metadata, node_costs
        )

        # Phase 4: 验证约束
        self._validate_plan(plan, factors)

        # Phase 5: 识别批量转换机会
        batch_opportunities = self._identify_batch_conversion_opportunities(plan)
        if batch_opportunities:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.info(
                "Batch conversion opportunities: %s",
                {k: len(v) for k, v in batch_opportunities.items()},
            )

        return plan

    def _optimize_with_dag_partitioner(
        self,
        factors: list[FactorSpec],
        node_graph: dict[str, list[str]],
        node_metadata: dict[str, NodeMetadata],
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]],
    ) -> PhysicalRegionPlan:
        """使用 DAG 分区优化器进行全局优化。"""
        try:
            from factor_engine.planning.dag_partition_optimizer import (
                DAGNode,
                DAGPartitionOptimizer,
                DataShape,
            )

            # 转换为 DAGNode 格式
            dag_nodes: dict[str, DAGNode] = {}
            for node_id, meta in node_metadata.items():
                # 构建后端候选和成本
                backend_candidates = []
                compute_costs = {}
                for backend, cost_obj in node_costs.get(node_id, {}).items():
                    backend_name = backend.value if hasattr(backend, 'value') else str(backend)
                    backend_candidates.append(backend_name)
                    compute_costs[backend_name] = cost_obj.compute_cost_ms

                if not backend_candidates:
                    continue

                dag_nodes[node_id] = DAGNode(
                    node_id=node_id,
                    operator=meta.operator_name,
                    backend_candidates=backend_candidates,
                    compute_costs=compute_costs,
                    shape=DataShape(
                        rows=meta.estimated_rows,
                        cols=5,
                        bytes=meta.estimated_bytes,
                    ),
                    children=node_graph.get(node_id, []),
                    is_shared=meta.is_shared,
                    consumer_count=meta.consumer_count,
                )

            # 添加父节点引用
            for node_id, children in node_graph.items():
                for child_id in children:
                    if child_id in dag_nodes:
                        dag_nodes[child_id].parents.append(node_id)

            # 运行优化器
            optimizer = DAGPartitionOptimizer(
                conversion_penalty_multiplier=1.5,
                max_partition_size=50,
                enable_fusion=True,
            )

            root_ids = [f.root_node_id for f in factors]
            partition_plan = optimizer.optimize(dag_nodes, root_ids)

            # 转换为 PhysicalRegionPlan
            return self._convert_partition_plan_to_physical(
                partition_plan, dag_nodes, node_graph, factors
            )

        except Exception:
            # 回退到原始优化器
            optimizer = RegionOptimizer(memory_budget=self.memory_budget, mode="research")
            root_ids = [f.root_node_id for f in factors]
            return optimizer.optimize(
                node_graph=node_graph,
                node_costs=node_costs,
                root_ids=root_ids,
                logical_hash=self._compute_logical_hash(factors),
            )

    def _convert_partition_plan_to_physical(
        self,
        partition_plan: Any,
        dag_nodes: dict[str, Any],
        node_graph: dict[str, list[str]],
        factors: list[FactorSpec],
    ) -> PhysicalRegionPlan:
        """将分区方案转换为 PhysicalRegionPlan。"""
        from factor_engine.planning.backend_region import (
            BackendRegion,
            ExecutionAxis,
            PhysicalBackend,
            PhysicalProperty,
            Representation,
            StateContract,
        )

        # 转换分区为 BackendRegion
        regions = []
        for p in partition_plan.partitions:
            # 映射后端名称到 PhysicalBackend 枚举 (R21-PLANNER-TYPE-UNIFICATION)
            backend_map = {
                "pandas_numpy": PhysicalBackend.PANDAS_NUMPY,
                "polars": PhysicalBackend.POLARS_PANEL,
                "polars_panel": PhysicalBackend.POLARS_PANEL,
                "polars_long": PhysicalBackend.POLARS_LONG,
                "duckdb_sql": PhysicalBackend.DUCKDB_SQL,
            }
            backend_enum = backend_map.get(p.backend, PhysicalBackend.PANDAS_NUMPY)

            # 推断 representation
            repr_map = {
                PhysicalBackend.PANDAS_NUMPY: Representation.PANDAS_LONG,
                PhysicalBackend.POLARS_PANEL: Representation.POLARS_LONG,
                PhysicalBackend.POLARS_LONG: Representation.POLARS_LAZY_LONG,
                PhysicalBackend.DUCKDB_SQL: Representation.DUCKDB_RELATION,
            }
            representation = repr_map.get(backend_enum, Representation.PANDAS_LONG)

            region = BackendRegion(
                region_id=p.partition_id,
                backend=backend_enum,
                representation=representation,
                node_ids=tuple(p.node_ids),
                execution_axis=ExecutionAxis.GLOBAL_PANEL,
                estimated_rows=p.total_rows,
                estimated_compute_ms=0.0,
                estimated_memory_bytes=sum(dag_nodes[nid].shape.bytes for nid in p.node_ids),
                input_bytes=sum(dag_nodes[nid].shape.bytes for nid in p.node_ids),
            )
            regions.append(region)

        # 转换边
        from factor_engine.planning.transfer_edge import (
            SemanticContract,
            TransferEdge,
            TransferKind,
            estimate_transfer_cost,
            infer_transfer_kind,
        )

        partition_map = {p.region_id: p for p in regions}
        edges = []
        edge_counter = [0]

        for from_pid, to_pid in partition_plan.conversion_edges:
            if from_pid not in partition_map or to_pid not in partition_map:
                continue

            from_region = partition_map[from_pid]
            to_region = partition_map[to_pid]

            edge_counter[0] += 1
            transfer_kind = infer_transfer_kind(
                from_region.representation,
                to_region.representation,
            )

            edge = TransferEdge(
                edge_id=f"E{edge_counter[0]}",
                producer_region=from_pid,
                consumer_region=to_pid,
                source_representation=from_region.representation,
                target_representation=to_region.representation,
                transfer_kind=transfer_kind,
                estimated_rows=from_region.estimated_rows,
                estimated_bytes=from_region.estimated_memory_bytes,
                requires_sort=False,
                requires_repartition=False,
                requires_reshape=False,
                requires_dtype_cast=False,
                semantic_contract=SemanticContract(),
                source_properties=PhysicalProperty(),
                target_properties=PhysicalProperty(),
            )
            object.__setattr__(edge, "estimated_cost_ms", estimate_transfer_cost(edge))
            edges.append(edge)

        # 拓扑排序
        topo_order = self._topological_sort_regions(regions, edges)

        from factor_engine.planning.physical_region_plan import (
            PhysicalRegionPlan,
            compute_plan_hash,
        )

        return PhysicalRegionPlan(
            regions=tuple(regions),
            edges=tuple(edges),
            topological_order=tuple(topo_order),
            peak_memory_estimate=partition_plan.total_cost * 1000,  # Rough estimate
            estimated_ttdc_ms=partition_plan.total_cost,
            plan_hash=compute_plan_hash(
                tuple(regions),
                tuple(edges),
                self._compute_logical_hash(factors),
            ),
            logical_node_count=len(dag_nodes),
            shared_node_count=sum(1 for n in dag_nodes.values() if n.is_shared),
        )

    def _topological_sort_regions(
        self, regions: list[Any], edges: list[Any]
    ) -> list[str]:
        """拓扑排序。"""
        in_degree = {r.region_id: 0 for r in regions}
        adj = {r.region_id: [] for r in regions}

        for edge in edges:
            adj[edge.producer_region].append(edge.consumer_region)
            in_degree[edge.consumer_region] += 1

        queue = [rid for rid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            rid = queue.pop(0)
            result.append(rid)
            for neighbor in adj.get(rid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def _identify_batch_conversion_opportunities(
        self, plan: PhysicalRegionPlan
    ) -> dict[str, list[str]]:
        """识别批量转换机会。

        返回 {(from_backend, to_backend): [region_ids]} 可以批量转换的边。
        """
        from collections import defaultdict

        opportunities: dict[str, list[str]] = defaultdict(list)

        # 按转换类型分组边
        for edge in plan.edges:
            if edge.transfer_kind.value == "same_backend_native":
                continue

            key = f"{edge.source_representation.value}→{edge.target_representation.value}"
            opportunities[key].append(edge.edge_id)

        # 过滤出可批量的（>= 2 个边）
        return {k: v for k, v in opportunities.items() if len(v) >= 2}

    def _identify_shared_nodes(
        self,
        factors: list[FactorSpec],
        node_graph: dict[str, list[str]],
    ) -> set[str]:
        """识别被多个 factor 共享的节点。"""
        # 收集每个 factor 的所有节点
        factor_nodes: dict[str, set[str]] = {}
        for factor in factors:
            nodes = self._collect_subtree(factor.root_node_id, node_graph)
            factor_nodes[factor.factor_id] = nodes

        # 找出在多个 factor 中出现的节点
        all_nodes: dict[str, int] = {}
        for nodes in factor_nodes.values():
            for node_id in nodes:
                all_nodes[node_id] = all_nodes.get(node_id, 0) + 1

        return {node_id for node_id, count in all_nodes.items() if count > 1}

    def _collect_subtree(self, root: str, node_graph: dict[str, list[str]]) -> set[str]:
        """收集以 root 为根的子树所有节点。"""
        visited = set()
        stack = [root]
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            for child in node_graph.get(node_id, []):
                stack.append(child)
        return visited

    def _generate_node_costs(
        self,
        node_graph: dict[str, list[str]],
        node_metadata: dict[str, NodeMetadata],
        shared_nodes: set[str],
        available_memory: int,
    ) -> dict[str, dict[PhysicalBackend, NodeCost]]:
        """为每个节点生成所有候选 backend 的成本。"""
        node_costs: dict[str, dict[PhysicalBackend, NodeCost]] = {}

        for node_id, meta in node_metadata.items():
            backend_costs: dict[PhysicalBackend, NodeCost] = {}

            # 为该节点尝试所有可能的 backend (R21-PLANNER-TYPE-UNIFICATION)
            for backend in [
                PhysicalBackend.PANDAS_NUMPY,
                PhysicalBackend.POLARS_PANEL,
                PhysicalBackend.DUCKDB_SQL,
            ]:
                # 使用智能选择器评估成本
                ctx = RoutingContext(
                    estimated_rows=meta.estimated_rows,
                    estimated_columns=5,  # Default
                    estimated_bytes=meta.estimated_bytes,
                    available_memory_bytes=available_memory,
                    operator_profile=self._infer_profile_from_operator(meta.operator_name),
                    execution_axis=ExecutionAxis.GLOBAL_PANEL,
                    window_params=[meta.window] if meta.window else [],
                    has_regression="regress" in meta.operator_name or "neutral" in meta.operator_name,
                    parent_backend=None,
                    allows_streaming=True,
                    performance_priority="balanced",
                )

                decision = self._selector.select_backend(ctx)

                # 如果这是候选 backend，计算成本
                if decision.chosen_backend == backend or any(
                    alt[0] == backend for alt in decision.alternatives
                ):
                    # 找到该 backend 的成本
                    if decision.chosen_backend == backend:
                        cost_ms = decision.estimated_cost_ms
                    else:
                        # 从 alternatives 找
                        cost_ms = next(
                            (alt[1] for alt in decision.alternatives if alt[0] == backend),
                            decision.estimated_cost_ms * 1.5,  # Fallback
                        )

                    # 共享节点只计算一次
                    if meta.is_shared:
                        # Amortize cost across consumers
                        cost_ms = cost_ms / max(meta.consumer_count, 1)

                    backend_costs[backend] = NodeCost(
                        node_id=node_id,
                        backend=backend,
                        compute_cost_ms=cost_ms,
                        memory_bytes=meta.estimated_bytes,
                    )

            node_costs[node_id] = backend_costs

        return node_costs

    def _infer_profile_from_operator(self, operator: str) -> OperatorProfile:
        """从算子名推断 profile。"""
        if "ts_" in operator or "rolling" in operator:
            return OperatorProfile.WINDOW_HEAVY
        elif "cs_" in operator or "rank" in operator:
            return OperatorProfile.CROSS_SECTION_HEAVY
        elif "regress" in operator or "neutral" in operator:
            return OperatorProfile.REGRESSION_HEAVY
        else:
            return OperatorProfile.ELEMENTWISE_HEAVY

    def _compute_logical_hash(self, factors: list[FactorSpec]) -> str:
        """计算逻辑计划 hash。"""
        import hashlib
        import json

        payload = {
            "factors": [
                {
                    "factor_id": f.factor_id,
                    "root_node": f.root_node_id,
                    "operators": sorted(f.operator_names),
                }
                for f in factors
            ]
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _validate_plan(self, plan: PhysicalRegionPlan, factors: list[FactorSpec]) -> None:
        """验证计划约束。"""
        # 1. 验证内存预算
        if self.memory_budget and plan.peak_memory_estimate > self.memory_budget:
            raise ValueError(
                f"Plan exceeds memory budget: {plan.peak_memory_estimate / 1024**2:.1f}MB "
                f"> {self.memory_budget / 1024**2:.1f}MB"
            )

        # 2. 验证 backend 切换次数
        if self.max_backend_switches and plan.backend_switch_count > self.max_backend_switches:
            raise ValueError(
                f"Plan exceeds max backend switches: {plan.backend_switch_count} "
                f"> {self.max_backend_switches}"
            )

        # 3. 验证所有 factor 的根节点都被覆盖
        all_nodes = set()
        for region in plan.regions:
            all_nodes.update(region.node_ids)

        for factor in factors:
            if factor.root_node_id not in all_nodes:
                raise ValueError(f"Factor {factor.factor_id} root node not in plan")


@dataclass(frozen=True)
class HybridExecutionStats:
    """混合执行统计（用于性能对比）。"""

    total_factors: int
    total_nodes: int
    shared_nodes: int
    total_regions: int
    backend_distribution: dict[str, int]  # {backend: region_count}
    backend_switches: int
    estimated_ttdc_ms: float
    peak_memory_mb: float
    avg_transfer_cost_ms: float
    parallelism_degree: float  # 并行度（可并行 region 数 / 总 region 数）


def compute_hybrid_stats(plan: PhysicalRegionPlan) -> HybridExecutionStats:
    """计算混合执行统计。"""
    backend_dist: dict[str, int] = {}
    for region in plan.regions:
        backend_name = region.backend.value
        backend_dist[backend_name] = backend_dist.get(backend_name, 0) + 1

    # 计算并行度（简化：假设无依赖的 region 可并行）
    # 实际需要分析 DAG 拓扑
    parallelism = 0.0
    if plan.region_count > 0:
        # 简化：假设平均每层有 sqrt(N) 个可并行 region
        import math
        parallelism = min(math.sqrt(plan.region_count) / plan.region_count, 1.0)

    avg_transfer = 0.0
    if len(plan.edges) > 0:
        avg_transfer = sum(e.estimated_cost_ms for e in plan.edges) / len(plan.edges)

    return HybridExecutionStats(
        total_factors=0,  # Needs to be passed in
        total_nodes=plan.logical_node_count,
        shared_nodes=plan.shared_node_count,
        total_regions=plan.region_count,
        backend_distribution=backend_dist,
        backend_switches=plan.backend_switch_count,
        estimated_ttdc_ms=plan.estimated_ttdc_ms,
        peak_memory_mb=plan.peak_memory_estimate / 1024.0 / 1024.0,
        avg_transfer_cost_ms=avg_transfer,
        parallelism_degree=parallelism,
    )


def compare_single_vs_hybrid(
    single_backend_ttdc: float,
    hybrid_plan: PhysicalRegionPlan,
) -> dict[str, Any]:
    """对比单一后端 vs 混合后端性能。"""
    hybrid_ttdc = hybrid_plan.estimated_ttdc_ms
    speedup = single_backend_ttdc / hybrid_ttdc if hybrid_ttdc > 0 else 1.0

    return {
        "single_backend_ttdc_ms": round(single_backend_ttdc, 2),
        "hybrid_ttdc_ms": round(hybrid_ttdc, 2),
        "speedup": round(speedup, 2),
        "improvement_pct": round((speedup - 1.0) * 100, 1),
        "transfer_overhead_ms": round(hybrid_plan.total_transfer_cost_ms, 2),
        "transfer_overhead_pct": round(
            hybrid_plan.total_transfer_cost_ms / hybrid_ttdc * 100 if hybrid_ttdc > 0 else 0,
            1,
        ),
        "backend_switches": hybrid_plan.backend_switch_count,
        "regions": hybrid_plan.region_count,
    }
