"""Fail-closed production admission for cold-start formulas.

The historical ``daily``/``extended`` authoring surfaces are parser concerns, not
production-readiness claims. This module compiles each formula through the same
Expr -> IR -> PlanNode pipeline used by :class:`runtime.engine.FactorEngine`, then
applies production policy, evidence, source-contract and whole-plan routing
gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Iterable

from .model import ColdStartFactor

_NON_OPERATOR_PLAN_NODES = frozenset(
    {"", "column", "literal", "plan_ref", "materialized_series"}
)


@dataclass(frozen=True)
class ProductionAdmission:
    eligible: bool
    canonical_operators: tuple[str, ...] = ()
    certified_backends: tuple[tuple[str, tuple[str, ...]], ...] = ()
    physical_plan_backend: str = ""
    physical_plan_candidates: tuple[str, ...] = ()
    routing_basis: str = ""
    lookback: int | None = None
    violations: tuple[str, ...] = ()

    @property
    def backend_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.certified_backends)


def _walk_plan_operators(plan) -> tuple[str, ...]:
    names: set[str] = set()

    def visit(node) -> None:
        name = str(getattr(node, "op", "") or "")
        if name not in _NON_OPERATOR_PLAN_NODES:
            names.add(name)
        for child in getattr(node, "inputs", ()) or ():
            visit(child)

    visit(plan)
    return tuple(sorted(names))


def _walk_source_contracts(ir) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return independently executable logical-source contracts in an IR DAG.

    Source-backed minute-to-daily features intentionally lower to ``column`` IR
    nodes, because aggregation belongs at the data-source boundary rather than
    inside the factor operator registry. They therefore require their own
    auditable production contract instead of a fake operator registration.
    """
    from api.intraday_daily import INTRADAY_DAILY_DSL_FUNCTIONS
    from api.source_ref import decode_source_ref

    supported_features: set[str] = set()
    for function in INTRADAY_DAILY_DSL_FUNCTIONS.values():
        try:
            specification = decode_source_ref(function())
        except Exception:
            continue
        if specification is None or specification.transform != "intraday_feature":
            continue
        feature = str(specification.transform_params_dict().get("feature") or "")
        if feature:
            supported_features.add(feature)

    rows: set[tuple[str, tuple[str, ...]]] = set()

    def visit(node) -> None:
        if str(getattr(node, "op", "") or "") == "column":
            attrs = dict(getattr(node, "attrs", {}) or {})
            specification = decode_source_ref(str(attrs.get("name") or ""))
            if specification is not None and specification.transform:
                params = specification.transform_params_dict()
                if specification.transform != "intraday_feature":
                    raise ValueError(
                        f"uncertified SourceRef transform: {specification.transform!r}"
                    )
                feature = str(params.get("feature") or "")
                if feature not in supported_features:
                    raise ValueError(
                        f"intraday feature has no installed runtime contract: {feature!r}"
                    )
                logical = (
                    f"source::{specification.table}::"
                    f"{specification.transform}::{feature}"
                )
                rows.add((logical, ("source_runtime",)))
        for child in getattr(node, "inputs", ()) or ():
            visit(child)

    visit(ir)
    return tuple(sorted(rows))


def _compile_production_plan(formula: str):
    from api.dsl_parser import parse_expr
    from ir.analyzer import Analyzer
    from planner.lowerer import Lowerer
    from planner.optimizer import Optimizer
    from runtime.pit_audit import assert_pit_safe
    from runtime.production_policy import (
        assert_no_unapproved_map_groups_in_production,
        assert_production_fastpath_plan,
        assert_production_plan_ops,
    )

    expr = parse_expr(str(formula), surface="compat")
    analysis = Analyzer().lower(expr)
    assert_pit_safe(analysis.ir, enforce=True, forbid_forward_fill=False)
    plan = Lowerer().to_logical_plan(analysis.ir)
    plan = Optimizer().optimize(plan, production=True)
    assert_production_plan_ops(plan, mode="production", context="cold_start")
    assert_no_unapproved_map_groups_in_production(
        plan, mode="production", context="cold_start"
    )
    assert_production_fastpath_plan(plan, mode="production", context="cold_start")
    return plan, analysis


def _production_routing_context() -> SimpleNamespace:
    return SimpleNamespace(
        run_mode="production",
        data_source=None,
        runtime_stats={"row_count_estimate": 500_000},
    )


@lru_cache(maxsize=16384)
def admit_formula(formula: str) -> ProductionAdmission:
    """Return current production admission for one DSL formula.

    Admission is derived from checked-in implementation-bound evidence. Stale
    or missing evidence removes a formula from the default catalog instead of
    silently downgrading to an uncertified backend.
    """
    try:
        from backend.operator_capability import production_eligible_backends
        from backend.plan_cost_router import choose_plan_route
        from cleaned_operators import load_all
        from cleaned_operators.operator_spec import infer_production_policy
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        plan, analysis = _compile_production_plan(formula)

        source_rows = _walk_source_contracts(analysis.ir)
        canonical_names: set[str] = {name for name, _ in source_rows}
        backend_rows: list[tuple[str, tuple[str, ...]]] = list(source_rows)
        violations: list[str] = []
        for raw_name in _walk_plan_operators(plan):
            try:
                canonical = OperatorRegistry.resolve_canonical_strict(raw_name)
            except Exception as exc:
                violations.append(f"{raw_name}: canonical resolution failed: {exc}")
                continue
            canonical_names.add(canonical)
            policy = infer_production_policy(canonical)
            if policy != "allowed":
                violations.append(f"{canonical}: production_policy={policy}")
                continue
            backends = tuple(sorted(production_eligible_backends(canonical)))
            if not backends:
                violations.append(f"{canonical}: no evidence-backed production backend")
                continue
            backend_rows.append((canonical, backends))

        lookback_raw = getattr(analysis, "lookback", None)
        try:
            lookback = int(lookback_raw) if lookback_raw is not None else None
        except (TypeError, ValueError):
            lookback = None

        if violations:
            return ProductionAdmission(
                eligible=False,
                canonical_operators=tuple(sorted(canonical_names)),
                certified_backends=tuple(sorted(backend_rows)),
                lookback=lookback,
                violations=tuple(violations),
            )

        operator_names = _walk_plan_operators(plan)
        if source_rows and not operator_names:
            return ProductionAdmission(
                eligible=True,
                canonical_operators=tuple(sorted(canonical_names)),
                certified_backends=tuple(sorted(backend_rows)),
                physical_plan_backend="source_runtime",
                physical_plan_candidates=("source_runtime",),
                routing_basis="source_contract",
                lookback=lookback,
                violations=(),
            )

        route = choose_plan_route(plan, _production_routing_context())
        candidates = tuple(name for name, _ in route.candidate_costs)
        return ProductionAdmission(
            eligible=True,
            canonical_operators=tuple(sorted(canonical_names)),
            certified_backends=tuple(sorted(backend_rows)),
            physical_plan_backend=route.backend,
            physical_plan_candidates=candidates,
            routing_basis=route.routing_basis,
            lookback=lookback,
            violations=(),
        )
    except Exception as exc:
        return ProductionAdmission(eligible=False, violations=(str(exc),))


def admit_factor(factor: ColdStartFactor) -> ProductionAdmission:
    return admit_formula(factor.formula)


def filter_production_factors(
    rows: Iterable[ColdStartFactor],
) -> tuple[ColdStartFactor, ...]:
    return tuple(row for row in rows if admit_factor(row).eligible)


def rejected_production_factors(
    rows: Iterable[ColdStartFactor],
) -> tuple[tuple[ColdStartFactor, ProductionAdmission], ...]:
    rejected: list[tuple[ColdStartFactor, ProductionAdmission]] = []
    for row in rows:
        result = admit_factor(row)
        if not result.eligible:
            rejected.append((row, result))
    return tuple(rejected)
