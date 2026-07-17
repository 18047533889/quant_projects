# -*- coding: utf-8 -*-
"""Post-finalization compatibility that does not widen the authoring surface."""
from cleaned_operators.registry import OperatorRegistry
from research_tools.registry import ResearchToolRegistry

_APPLIED = False


def _restore_compiler_internal(canonical: str) -> None:
    implementations = ResearchToolRegistry._tools.pop(canonical, {})
    metadata = ResearchToolRegistry._catalog.pop(canonical, {})
    for backend, operator in implementations.items():
        OperatorRegistry.register(
            operator,
            canonical=canonical,
            backend=backend,
            source="compiler_internal",
            status="internal",
            backend_explicit=True,
            semantic_version="2.0",
        )
    if canonical in OperatorRegistry._catalog:
        OperatorRegistry._catalog[canonical].update({
            "surface": "internal",
            "scope": "elementwise",
            "pit_safe": True,
            "required_cutoff": "input_dependent",
            "earliest_trade_time": "input_dependent",
            "migration_reason": metadata.get("migration_reason", "compiler lowering node"),
        })


def apply_post_governance() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from cleaned_operators import operator_surface

    # The parser lowers `/` to protected_div.  Keep it as an internal kernel,
    # never as an author-facing daily operator.
    _restore_compiler_internal("protected_div")
    if "protected_div" in OperatorRegistry._operators:
        operator_surface.INTERNAL_ONLY_CANONICALS = frozenset(
            set(operator_surface.INTERNAL_ONLY_CANONICALS) | {"protected_div"}
        )

    # Historical GTJA formulas may spell linear decay as WMA.  It is a one-hop
    # compatibility alias to the primitive, not another canonical implementation.
    if "ts_decay_linear" in OperatorRegistry._operators:
        OperatorRegistry.register_alias("WMA", "ts_decay_linear")

    renamed = {"cs_count", "cs_mean", "cs_std", "cs_sum"} & set(OperatorRegistry._operators)
    operator_surface.DAILY_CANONICALS = frozenset(set(operator_surface.DAILY_CANONICALS) | renamed)
    for canonical in renamed:
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is not None:
            catalog["surface"] = "daily"
            catalog["scope"] = "cross_sectional"
            catalog["pit_safe"] = True
    _APPLIED = True
