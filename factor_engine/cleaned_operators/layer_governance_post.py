# -*- coding: utf-8 -*-
"""Post-governance compatibility and backend-independent production metadata."""
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
        catalog["segmented_execution_supported"] = bool(
            contract.get("segmented_execution_supported", True)
        )


def _mark_pandas_first_production() -> None:
    """Admit reviewed semantics without requiring SQL/Polars portability.

    This updates lifecycle metadata only for active operators that actually have
    a Pandas/NumPy runtime and remain PIT-safe according to the policy layer.
    ``operator_spec`` will still fail closed on PIT/shape/denied rules.
    """
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.production_tiers import PANDAS_FIRST_PRODUCTION_CANONICALS

    for canonical in sorted(PANDAS_FIRST_PRODUCTION_CANONICALS):
        if "pandas_numpy" not in OperatorRegistry.backends_for(canonical):
            continue
        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        if operator is None:
            continue
        policy = infer_operator_policy(operator, canonical=canonical)
        if not bool(getattr(policy, "pit_safe", False)):
            continue
        if not bool(getattr(policy, "shape_preserving", True)):
            continue
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        from cleaned_operators.semantic_certification import should_fail_closed

        if should_fail_closed(canonical):
            # Registered experimental/research or isolated: never force-promote
            # here.  ``apply_production_hardening`` keeps these experimental
            # and non-PIT-safe; ``apply_evidence_certification_overlay`` is the
            # only authority that can set ``production_certified``.
            catalog["status"] = "experimental"
            catalog["lifecycle_status"] = "experimental"
            catalog["pit_safe"] = False
            catalog["production_certified"] = False
            continue
        catalog["status"] = "production"
        catalog["lifecycle_status"] = "production"
        catalog["production_backend_policy"] = "at_least_one_certified_backend"
        catalog["production_portability_required"] = False
        backend_meta = dict(catalog.get("backend_meta") or {})
        pandas_meta = dict(backend_meta.get("pandas_numpy") or {})
        pandas_meta.update({
            "production_certified": True,
            "certification_tier": "pandas_first",
            "reference_backend": True,
        })
        backend_meta["pandas_numpy"] = pandas_meta
        catalog["backend_meta"] = backend_meta


def apply_post_governance() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _restore_compiler_internal("protected_div")
    if "neg" in OperatorRegistry._operators:
        OperatorRegistry.unregister("reverse")
        OperatorRegistry.register_alias("reverse", "neg")
    if "ts_decay_linear" in OperatorRegistry._operators:
        OperatorRegistry.register_alias("WMA", "ts_decay_linear")

    renamed = {"cs_count", "cs_mean", "cs_std", "cs_sum"} & set(OperatorRegistry._operators)
    for canonical in renamed:
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is not None:
            catalog["surface"] = "daily"
            catalog["scope"] = "cross_sectional"
            catalog["pit_safe"] = True

    _normalize_research_aliases()
    _attach_checkpoint_contracts()
    _mark_pandas_first_production()
    _APPLIED = True
