# -*- coding: utf-8
"""Production 算子级 checklist 主索引（收工清单，不再无限找问题）。

四件事收敛路线：
1. 修已有实现中的具体语义问题
2. 84 候选按批次完成六证 evidence
3. 少量高杠杆 primitive（非堆 emitter）
4. Stateful/PIT/session/external kernel → 第二阶段
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChecklistStatus = Literal[
    "defined",
    "implemented",
    "tested",
    "wired_polars",
    "wired_duckdb",
    "gate_enforced",
    "parity_tested",
    "production_verified",
    "deferred_phase2",
]

PHASE1_COMPLETE_STATUSES = frozenset(
    {"gate_enforced", "parity_tested", "production_verified", "tested"}
)


@dataclass(frozen=True)
class ChecklistItem:
    section: str
    topic: str
    status: ChecklistStatus
    module: str


PRODUCTION_CHECKLIST: tuple[ChecklistItem, ...] = (
    ChecklistItem("§1", "CalendarSpec 时区/交易日", "defined", "backend.calendar_spec"),
    ChecklistItem("§2", "row_delay 第一阶段认证", "defined", "backend.lag_spec"),
    ChecklistItem("§3", "ffill limit/max_age", "deferred_phase2", "backend.fill_semantics"),
    ChecklistItem("§4", "重复 (ts,inst) 报错", "implemented", "backend.long_alignment"),
    ChecklistItem("§5", "输出 sort(ts,inst)", "implemented", "backend.ordering_spec"),
    ChecklistItem("§6", "numeric promotion", "defined", "backend.promotion_spec"),
    ChecklistItem("§7", "group key Utf8/Categorical", "defined", "backend.group_key_spec"),
    ChecklistItem("§8", "GroupSpec min_valid", "defined", "backend.group_spec"),
    ChecklistItem("§9", "winsorize 参数 planner 校验", "implemented", "backend.plan_params"),
    ChecklistItem("§10", "count 语义拆分", "defined", "backend.count_semantics"),
    ChecklistItem("§11", "sum(all null)→NULL", "defined", "backend.aggregation_spec"),
    ChecklistItem("§12", "min/max NaN→NULL", "defined", "backend.aggregation_spec"),
    ChecklistItem("§13", "power/exp/log overflow", "defined", "backend.math_domain_semantics"),
    ChecklistItem("§14", "pairwise corr/cov/beta", "implemented", "backend.pairwise_alignment"),
    ChecklistItem("§15", "zscore std epsilon", "defined", "backend.cross_section_spec"),
    ChecklistItem("§16", "vwap 精度/溢出", "defined", "backend.financial_semantics"),
    ChecklistItem("§17", "cumulative incremental state", "deferred_phase2", "backend.cumulative_state_spec"),
    ChecklistItem("§18", "空 universe shape", "defined", "backend.universe_spec"),
    ChecklistItem("§19", "代数性质测试", "parity_tested", "tests.backend_parity.test_production_invariants"),
    ChecklistItem("§20", "fuzz parity", "parity_tested", "tests.backend_parity.fuzz_parity"),
    ChecklistItem("§21", "plan 性能预算", "defined", "backend.plan_performance_spec"),
    ChecklistItem("§22", "plan cost tags", "defined", "backend.plan_cost_tags"),
    ChecklistItem("§23", "optimizer on/off parity", "parity_tested", "tests.backend_parity.test_plan_variant_parity"),
    ChecklistItem("§24", "operator_semantic_version", "defined", "backend.operator_semantic_version"),
    ChecklistItem("§25", "production 按签名准入", "gate_enforced", "backend.production_signature"),
)


def checklist_complete_for_phase1() -> bool:
    """Phase-1 仅接受 gate_enforced / parity_tested / production_verified / tested。"""
    allowed_deferred = {"§3", "§17"}
    for item in PRODUCTION_CHECKLIST:
        if item.section in allowed_deferred:
            continue
        if item.status not in PHASE1_COMPLETE_STATUSES:
            return False
    return True


def checklist_summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in PRODUCTION_CHECKLIST:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts
