"""Fail-closed production admission for cold-start formulas.

The historical ``daily``/``extended`` authoring surfaces are parser concerns, not
production-readiness claims.  This module asks the current FactorEngine policy,
parameter-signature, backend-evidence and whole-plan routing layers whether a
fully built expression is eligible for production execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Iterable

from .model import ColdStartFactor

_NON_OPERATOR_PLAN_NODES = frozenset({"", "column", "literal"})


@dataclass(frozen=True)
class ProductionAdmission:
    eligible: bool
    canonical_operators: tuple[str, ...] = ()
    certified_backends: tuple[tuple[str, tuple[str, ...]], ...] = ()
    physical_plan_backend: str = ""
    physical_plan_candidates: tuple[str, ...] = ()
    routing_basis: str = ""
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


def _production_routing_context() -> SimpleNamespace:
    # Cold-start admission is source-agnostic and must have a safe in-memory
    # execution route.  SourceRef/PIT/session requirements are validated by the
    # production policy and dedicated source-contract suites.  Runtime may later
    # choose another certified route for a concrete DuckDB/ClickHouse source.
    return SimpleNamespace(
        run_mode="production",
        data_source=None,
        runtime_stats={"row_count_estimate": 500_000},
    )


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
        from backend.plan_cost_router import choose_plan_route
        from cleaned_operators import load_all
        from cleaned_operators.operator_spec import infer_production_policy
        from cleaned_operators.registry import OperatorRegistry
        from runtime.production_policy import assert_production_plan_ops

        load_all()
        # Production factor operators may originate from the former Extended
        # authoring surface.  ``compat`` only permits parsing; production policy
        # and physical-plan selection below are the actual admission gates.
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

        if violations:
            return ProductionAdmission(
                eligible=False,
                canonical_operators=tuple(sorted(canonical_names)),
                certified_backends=tuple(sorted(backend_rows)),
                violations=tuple(violations),
            )

        # Node-by-node capability is insufficient: two certified operators can
        # still form a DAG for which no complete physical plan exists.  Reuse
        # the same whole-plan router used by production execution and reject the
        # formula unless it can produce a complete certified route.
        route = choose_plan_route(plan, _production_routing_context())
        candidates = tuple(name for name, _ in route.candidate_costs)
        return ProductionAdmission(
            eligible=True,
            canonical_operators=tuple(sorted(canonical_names)),
            certified_backends=tuple(sorted(backend_rows)),
            physical_plan_backend=route.backend,
            physical_plan_candidates=candidates,
            routing_basis=route.routing_basis,
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