# -*- coding: utf-8 -*-
"""ExplainPlan: 解释每个 Region 为何选择该 backend (MB-P2-010)。

为每个 Region 生成人类可读的决策解释，包括：
- 节点数与算子能力
- 输入表示与 boundary 成本
- 内存估计与候选后端对比
- 最终选择理由
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RegionCapabilitySummary:
    """Region 内算子能力汇总。"""

    total_nodes: int
    native_nodes: int
    delegate_nodes: int
    unsupported_nodes: int
    native_fraction: float
    operator_list: tuple[str, ...]


@dataclass(frozen=True)
class BackendCandidateScore:
    """单个 backend 候选的评分细节。"""

    backend: str
    compute_cost_ms: float
    transfer_cost_ms: float
    memory_estimate_mb: float
    capability_score: float
    reason: str
    rejected: bool = False
    rejection_reason: str = ""


@dataclass(frozen=True)
class RegionExplanation:
    """单个 Region 的选择解释 (P2-010)。"""

    region_id: str
    chosen_backend: str
    node_count: int
    capability_summary: RegionCapabilitySummary
    input_representation: str
    output_representation: str
    estimated_rows: int
    estimated_bytes: int
    candidates: tuple[BackendCandidateScore, ...]
    decision_reason: str
    boundary_cost_ms: float
    requires_sort: bool
    requires_repartition: bool
    execution_axis: str


@dataclass(frozen=True)
class ExplainPlan:
    """全局 PhysicalRegionPlan 的人类可读解释 (MB-P2-010)。

    对应文档 §71 示例：
        Region R2 → Polars
        - 43 native expressions
        - input already Arrow/Polars
        - no additional global sort
        - estimated 310MB
        - DuckDB saves only 4ms
        - boundary cost estimated 12ms
        => stay Polars
    """

    total_regions: int
    total_logical_nodes: int
    total_shared_nodes: int
    backend_switches: int
    regions: tuple[RegionExplanation, ...]
    global_decision_basis: str
    peak_memory_estimate_mb: float
    estimated_ttdc_ms: float
    plan_hash: str
    within_memory_budget: bool
    memory_budget_mb: float | None = None


def _format_region_explanation(region: RegionExplanation) -> str:
    """格式化单个 Region 的解释为多行文本。"""
    lines = [f"Region {region.region_id} → {region.chosen_backend}"]
    cap = region.capability_summary
    lines.append(f"- {cap.native_nodes}/{cap.total_nodes} native expressions ({cap.native_fraction:.1%})")
    if cap.delegate_nodes > 0:
        lines.append(f"- {cap.delegate_nodes} delegate nodes (not native)")
    if cap.unsupported_nodes > 0:
        lines.append(f"- {cap.unsupported_nodes} unsupported nodes")
    lines.append(f"- input representation: {region.input_representation}")
    if region.output_representation != region.input_representation:
        lines.append(f"- output representation: {region.output_representation}")
    lines.append(f"- estimated {region.estimated_rows:,} rows, {region.estimated_bytes // 1024 // 1024}MB")
    if region.requires_sort:
        lines.append("- requires sort")
    if region.requires_repartition:
        lines.append("- requires repartition")
    lines.append(f"- execution axis: {region.execution_axis}")

    # 候选后端对比
    if len(region.candidates) > 1:
        lines.append("- candidates:")
        for cand in region.candidates:
            if cand.rejected:
                lines.append(f"  × {cand.backend}: rejected ({cand.rejection_reason})")
            else:
                total = cand.compute_cost_ms + cand.transfer_cost_ms
                lines.append(
                    f"  · {cand.backend}: {total:.1f}ms "
                    f"(compute {cand.compute_cost_ms:.1f}ms + transfer {cand.transfer_cost_ms:.1f}ms), "
                    f"{cand.memory_estimate_mb:.1f}MB"
                )

    if region.boundary_cost_ms > 0:
        lines.append(f"- boundary cost: {region.boundary_cost_ms:.1f}ms")

    lines.append(f"=> {region.decision_reason}")
    return "\n".join(lines)


def format_explain_plan(plan: ExplainPlan) -> str:
    """格式化 ExplainPlan 为人类可读文本 (P2-010)。

    返回:
        多行格式化文本，包含全局统计和每个 Region 的详细解释。
    """
    lines = ["=" * 80, "Physical Region Plan Explanation", "=" * 80, ""]

    # 全局统计
    lines.append(f"Total logical nodes: {plan.total_logical_nodes:,}")
    lines.append(f"Shared nodes: {plan.total_shared_nodes:,}")
    lines.append(f"Total regions: {plan.total_regions}")
    lines.append(f"Backend switches: {plan.backend_switches}")
    lines.append(f"Peak memory estimate: {plan.peak_memory_estimate_mb:.1f}MB")
    if plan.memory_budget_mb:
        lines.append(f"Memory budget: {plan.memory_budget_mb:.1f}MB")
        status = "✓" if plan.within_memory_budget else "✗ EXCEEDED"
        lines.append(f"Within budget: {status}")
    lines.append(f"Estimated TTDC: {plan.estimated_ttdc_ms:.1f}ms")
    lines.append(f"Plan hash: {plan.plan_hash}")
    lines.append(f"Decision basis: {plan.global_decision_basis}")
    lines.append("")

    # 每个 Region 详情
    for region in plan.regions:
        lines.append(_format_region_explanation(region))
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def build_explain_plan(
    regions: list[Any],
    *,
    total_logical_nodes: int,
    total_shared_nodes: int,
    backend_switches: int,
    peak_memory_mb: float,
    estimated_ttdc_ms: float,
    plan_hash: str,
    global_decision_basis: str = "batch-global cost minimization",
    memory_budget_mb: float | None = None,
) -> ExplainPlan:
    """从 PhysicalRegionPlan 构建 ExplainPlan (P2-010)。

    Args:
        regions: BackendRegion 列表
        total_logical_nodes: 全局逻辑节点总数
        total_shared_nodes: 共享节点数
        backend_switches: backend 切换次数
        peak_memory_mb: 峰值内存估计 (MB)
        estimated_ttdc_ms: 预估端到端耗时 (ms)
        plan_hash: 物理计划 hash
        global_decision_basis: 全局决策依据
        memory_budget_mb: 内存预算 (MB)，可选

    Returns:
        ExplainPlan 实例
    """
    region_explanations: list[RegionExplanation] = []

    for reg in regions:
        # 从 BackendRegion 提取信息（假设有这些字段）
        region_id = getattr(reg, "region_id", "unknown")
        backend = getattr(reg, "backend", "unknown")
        node_ids = getattr(reg, "node_ids", ())
        representation = getattr(reg, "representation", "unknown")
        execution_axis = getattr(reg, "execution_axis", "unknown")

        # 能力汇总（需要从外部传入或从 region metadata 提取）
        capability = RegionCapabilitySummary(
            total_nodes=len(node_ids),
            native_nodes=getattr(reg, "native_nodes", len(node_ids)),
            delegate_nodes=getattr(reg, "delegate_nodes", 0),
            unsupported_nodes=getattr(reg, "unsupported_nodes", 0),
            native_fraction=getattr(reg, "native_fraction", 1.0),
            operator_list=getattr(reg, "operator_list", ()),
        )

        # 候选后端（需要从决策过程传入）
        candidates = getattr(reg, "candidates", ())
        if not isinstance(candidates, tuple):
            candidates = tuple(candidates) if candidates else ()

        # 边界属性
        estimated_rows = getattr(reg, "estimated_rows", 0)
        estimated_bytes = getattr(reg, "estimated_bytes", 0)
        requires_sort = getattr(reg, "requires_sort", False)
        requires_repartition = getattr(reg, "requires_repartition", False)
        boundary_cost_ms = getattr(reg, "boundary_cost_ms", 0.0)
        decision_reason = getattr(reg, "decision_reason", "selected by planner")

        region_explanations.append(
            RegionExplanation(
                region_id=region_id,
                chosen_backend=backend,
                node_count=len(node_ids),
                capability_summary=capability,
                input_representation=representation,
                output_representation=representation,
                estimated_rows=estimated_rows,
                estimated_bytes=estimated_bytes,
                candidates=candidates,
                decision_reason=decision_reason,
                boundary_cost_ms=boundary_cost_ms,
                requires_sort=requires_sort,
                requires_repartition=requires_repartition,
                execution_axis=execution_axis,
            )
        )

    within_budget = True
    if memory_budget_mb is not None:
        within_budget = peak_memory_mb <= memory_budget_mb

    return ExplainPlan(
        total_regions=len(regions),
        total_logical_nodes=total_logical_nodes,
        total_shared_nodes=total_shared_nodes,
        backend_switches=backend_switches,
        regions=tuple(region_explanations),
        global_decision_basis=global_decision_basis,
        peak_memory_estimate_mb=peak_memory_mb,
        estimated_ttdc_ms=estimated_ttdc_ms,
        plan_hash=plan_hash,
        within_memory_budget=within_budget,
        memory_budget_mb=memory_budget_mb,
    )
