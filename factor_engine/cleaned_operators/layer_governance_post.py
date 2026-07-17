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


def _normalize_research_aliases() -> None:
    """Make research manifests deterministic across Python hash seeds."""
    forced = {
        "cum_max": "expanding_max",
        "cumulative_max": "expanding_max",
        "cum_min": "expanding_min",
        "cumulative_min": "expanding_min",
        "cum_sum": "expanding_sum",
        "cumulative_sum": "expanding_sum",
        "cumulative_mean": "expanding_mean",
    }
    forced_names = set(forced)
    for catalog in ResearchToolRegistry._catalog.values():
        aliases = set(catalog.get("aliases") or []) - forced_names
        catalog["aliases"] = sorted(aliases)
    for alias, canonical in forced.items():
        catalog = ResearchToolRegistry._catalog.get(canonical)
        if catalog is not None:
            catalog["aliases"] = sorted(set(catalog.get("aliases") or []) | {alias})


def _attach_checkpoint_contracts() -> None:
    from stateful_contract import StatefulCheckpointRegistry

    for canonical, contract in StatefulCheckpointRegistry.catalog().items():
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        catalog["stateful"] = True
        catalog["checkpoint_contract"] = contract
        catalog["segmented_execution_requires_checkpoint"] = bool(
            contract["checkpoint_required_for_segmented"]
        )


def apply_post_governance() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from cleaned_operators import operator_surface

    _restore_compiler_internal("protected_div")
    if "protected_div" in OperatorRegistry._operators:
        operator_surface.INTERNAL_ONLY_CANONICALS = frozenset(
            set(operator_surface.INTERNAL_ONLY_CANONICALS) | {"protected_div"}
        )

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

    _normalize_research_aliases()
    _attach_checkpoint_contracts()
    _APPLIED = True
