"""Fail-closed production admission for cold-start formulas.

The historical ``daily``/``extended`` authoring surfaces are parser concerns, not
production-readiness claims.  This module asks the current FactorEngine policy,
parameter-signature and backend-evidence layers whether a fully built expression
is eligible for production routing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .model import ColdStartFactor

_NON_OPERATOR_PLAN_NODES = frozenset({"", "column", "literal"})


@dataclass(frozen=True)
class ProductionAdmission:
    eligible: bool
    canonical_operators: tuple[str, ...] = ()
    certified_backends: tuple[tuple[str, tuple[str, ...]], ...] = ()
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


@lru_cache(maxsize=16384)
def admit_formula(formula: str) -> ProductionAdmission:
    """Return current production admission for one DSL formula.

    Admission is deliberately derived at runtime from the checked-in
    FactorEngine evidence.  A stale or missing evidence artifact therefore
    removes a formula from the default cold-start pack instead of silently
    downgrading to an uncertified backend.
    """
    try:
        from api.dsl_parser import parse_expr
        from backend.operator_capability import production_eligible_backends
        from cleaned_operators import load_all
        from cleaned_operators.operator_spec import infer_production_policy
        from cleaned_operators.registry import OperatorRegistry
        from runtime.production_policy import assert_production_plan_ops

        load_all()
        # Production factor operators may originate from the former Extended
        # authoring surface.  ``compat`` only permits parsing; the production
        # policy below is the actual admission gate.
        plan = parse_expr(str(formula), surface="compat")
        assert_production_plan_ops(plan, mode="production", context="cold_start")

        canonical_names: set[str] = set()
        backend_rows: list[tuple[str, tuple[str, ...]]] = []
        violations: list[str] = []
        for raw_name in _walk_plan_operators(plan):
            try:
                canonical = OperatorRegistry.resolve_canonical_strict(raw_name)
            except Exception as exc:  # fail closed on aliases or unregistered ops
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

        return ProductionAdmission(
            eligible=not violations,
            canonical_operators=tuple(sorted(canonical_names)),
            certified_backends=tuple(sorted(backend_rows)),
            violations=tuple(violations),
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
