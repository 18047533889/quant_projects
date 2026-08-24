# -*- coding: utf-8
"""用户级因子 explain / validate API（隐藏内部 tier 集合）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TargetProfile = Literal[
    "production",
    "dual_backend_production",
    "research",
]


@dataclass
class FactorExplanation:
    formula: str
    production_ok: bool
    dual_backend_ok: bool
    polars_long_native: bool
    duckdb_fully_pushed: bool
    contains_deferred: bool
    contains_composite: bool
    fallback_risk: str
    operators: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "production_ok": self.production_ok,
            "dual_backend_ok": self.dual_backend_ok,
            "polars_long_native": self.polars_long_native,
            "duckdb_fully_pushed": self.duckdb_fully_pushed,
            "contains_deferred": self.contains_deferred,
            "contains_composite": self.contains_composite,
            "fallback_risk": self.fallback_risk,
            "operators": self.operators,
            "violations": self.violations,
            "suggestions": self.suggestions,
        }


def _compile_plan(formula: str):
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.planner.lowerer import Lowerer
    from factor_engine.planner.optimizer import Optimizer

    expr = parse_expr(str(formula or ""))
    analysis = Analyzer().lower(expr)
    original = Lowerer().to_logical_plan(analysis.ir)
    optimized, pre_lowering, _trace = Optimizer().optimize_with_trace(original)
    return optimized, pre_lowering


def explain_factor(formula: str, *, mode: str = "research") -> FactorExplanation:
    """解释因子表达式的 production / 双后端 / fallback 风险。"""
    from factor_engine.backend.production_fastpath_gate import check_production_fastpath_plan_ops
    from factor_engine.backend.production_fastpath_tiers import FASTPATH_DEFERRED_CANONICALS
    from factor_engine.planner.composite_lowering import has_composite_lowering

    optimized, pre_lowering = _compile_plan(formula)
    prod = check_production_fastpath_plan_ops(
        optimized,
        original_plan=pre_lowering,
        mode="production",
        check_full_plan=False,
    )
    dual = check_production_fastpath_plan_ops(
        optimized,
        original_plan=pre_lowering,
        mode="production",
        require_mode="all",
        check_full_plan=False,
    )

    ops = list(dict.fromkeys(prod.ops_checked + prod.original_ops))
    deferred = [o for o in ops if o in FASTPATH_DEFERRED_CANONICALS]
    composites = [o for o in ops if has_composite_lowering(o)]

    suggestions: list[str] = []
    for v in prod.violations:
        if "fillna" in v and "fillna_const" in v:
            suggestions.append("将 fillna(x, 0) 改为 fillna_const(x, 0)")
        if "bfill" in v:
            suggestions.append("禁止使用 bfill；因果填充请用 ffill")
        if "pending" in v:
            suggestions.append("该 composite 第一阶段仍为 research；请改用 lowered primitive 组合")

    fallback = "low"
    if not dual.ok:
        fallback = "medium"
    if deferred:
        fallback = "high"
    if any("fallback" in v for v in prod.violations):
        fallback = "high"

    return FactorExplanation(
        formula=formula,
        production_ok=prod.ok,
        dual_backend_ok=dual.ok,
        polars_long_native=dual.ok,
        duckdb_fully_pushed=dual.ok,
        contains_deferred=bool(deferred),
        contains_composite=bool(composites),
        fallback_risk=fallback,
        operators=ops,
        violations=list(prod.violations),
        suggestions=suggestions,
    )


def validate_factor(
    formula: str,
    *,
    target: TargetProfile = "dual_backend_production",
    mode: str | None = None,
) -> tuple[bool, FactorExplanation]:
    """校验因子是否满足目标 profile；返回 (ok, explanation)。"""
    exp = explain_factor(formula, mode=mode or ("production" if target != "research" else "research"))
    if target == "research":
        return True, exp
    if target == "production":
        return exp.production_ok, exp
    return exp.dual_backend_ok, exp
