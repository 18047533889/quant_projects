# -*- coding: utf-8 -*-
"""Final-registry backend capability synchronization.

Static emitter/evidence sets are historical inputs, not authority.  This module
intersects them with the fully governed runtime registry and publishes one
active view used by routing, reports and CI.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from cleaned_operators.registry import OperatorRegistry

STRUCTURAL_SQL_NODES = frozenset({"column", "literal"})

# Source-level classification is deliberately conservative.  Only explicitly
# audited expression/long sources are called expression-native.
_EXPRESSION_NATIVE_SOURCES = frozenset({
    "final_expression_native_polars",
    "layer_governance_native_polars",
    "operator_overhaul_native_polars",
    "semantic_hardening_native_polars",
    "technical_native_polars",
    "price_volume_native_polars",
    "microstructure_native_polars",
})
_CALLBACK_OR_NUMPY_POLARS = frozenset({
    "period_lag", "cs_rank_gaussian", "ts_tail_mean", "ts_time_slope",
})


@dataclass(frozen=True)
class BackendCapability:
    canonical: str
    surface: str
    pandas_runtime: bool
    polars_registered: bool
    polars_no_pandas_bridge: bool
    polars_expression_native: bool
    polars_parity_verified: bool
    polars_production_safe: bool
    duckdb_emitter_implemented: bool
    duckdb_reference_parity: bool
    duckdb_production_safe: bool
    clickhouse_production_safe: bool
    polars_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _active_runtime_names() -> frozenset[str]:
    return frozenset(OperatorRegistry._operators)


def _polars_source(name: str) -> str:
    catalog = OperatorRegistry._catalog.get(name, {})
    return str(((catalog.get("backend_meta") or {}).get("polars") or {}).get("source", ""))


def _non_bridge(source: str) -> bool:
    lowered = source.lower()
    return bool(source) and "bridge" not in lowered and source != "daily_panel_polars"


def _expression_native(name: str, source: str) -> bool:
    if name in _CALLBACK_OR_NUMPY_POLARS:
        return False
    return source in _EXPRESSION_NATIVE_SOURCES or source.startswith("final_expression_")


def synchronize_active_backend_sets() -> None:
    """Mutate legacy module-level sets to the final active canonical view."""
    active = _active_runtime_names()

    from cleaned_operators import operator_policy
    operator_policy.POLARS_PARITY_VERIFIED = frozenset(
        name for name in operator_policy.POLARS_PARITY_VERIFIED
        if name in active and "polars" in OperatorRegistry.backends_for(name)
    )
    operator_policy.POLARS_PRODUCTION_SAFE = frozenset(
        name for name in operator_policy.POLARS_PRODUCTION_SAFE
        if name in active
        and "polars" in OperatorRegistry.backends_for(name)
        and _non_bridge(_polars_source(name))
    )

    from backend import primitive_evidence, sql_tiers
    from backend.sql_pushdown import emitter, sql_registry

    # Evidence is never inherited through an alias: only an exact, currently
    # active canonical can retain certification.
    evidence_attributes = (
        "POLARS_REFERENCE_PARITY_VERIFIED",
        "POLARS_EDGE_VERIFIED",
        "DUCKDB_REFERENCE_PARITY_VERIFIED",
        "DUCKDB_REAL_SQL_VERIFIED",
        "DUCKDB_EDGE_VERIFIED",
        "DUCKDB_NULL_EDGE_VERIFIED",
        "DUCKDB_NAN_EDGE_VERIFIED",
        "DUCKDB_INF_EDGE_VERIFIED",
        "POLARS_NO_FALLBACK_VERIFIED",
        "NO_FALLBACK_VERIFIED",
        "PRIMITIVE_BACKEND_EXECUTION_CERTIFIED",
        "PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE",
    )
    for attribute in evidence_attributes:
        current = frozenset(getattr(primitive_evidence, attribute, frozenset()))
        setattr(primitive_evidence, attribute, frozenset(name for name in current if name in active))

    # SQL markers are attached to final runtime canonicals.  This prevents old
    # emitter aliases from widening planner capability after registry cleanup.
    implemented = frozenset(
        name for name in active if "sql" in OperatorRegistry.backends_for(name)
    ) | STRUCTURAL_SQL_NODES
    parity = frozenset(
        name for name in sql_tiers.SQL_PARITY_VERIFIED_CANONICALS if name in implemented
    )
    production = frozenset(
        name for name in sql_tiers.SQL_PRODUCTION_SAFE_CANONICALS if name in parity
    )

    sql_tiers.SQL_IMPLEMENTED_CANONICALS = implemented
    sql_tiers.SQL_CAPABLE_CANONICALS = implemented
    sql_tiers.SQL_PARITY_VERIFIED_CANONICALS = parity
    sql_tiers.SQL_PRODUCTION_SAFE_CANONICALS = production
    sql_tiers.DUCKDB_SQL_PARITY_VERIFIED = parity
    sql_tiers.DUCKDB_SQL_PRODUCTION_SAFE = production

    sql_registry.SQL_IMPLEMENTED_CANONICALS = implemented
    sql_registry.SQL_CAPABLE_CANONICALS = implemented
    sql_registry.SQL_PARITY_VERIFIED_CANONICALS = parity
    sql_registry.SQL_PRODUCTION_SAFE_CANONICALS = production
    emitter.SQL_CAPABLE_OPS = implemented


def capability_matrix() -> dict[str, BackendCapability]:
    synchronize_active_backend_sets()
    from cleaned_operators import operator_policy
    from backend import sql_tiers

    result: dict[str, BackendCapability] = {}
    for name in sorted(_active_runtime_names()):
        catalog = OperatorRegistry._catalog.get(name, {})
        backends = set(OperatorRegistry.backends_for(name))
        source = _polars_source(name)
        result[name] = BackendCapability(
            canonical=name,
            surface=str(catalog.get("surface", "unclassified")),
            pandas_runtime="pandas_numpy" in backends,
            polars_registered="polars" in backends,
            polars_no_pandas_bridge="polars" in backends and _non_bridge(source),
            polars_expression_native=(
                "polars" in backends and _non_bridge(source) and _expression_native(name, source)
            ),
            polars_parity_verified=name in operator_policy.POLARS_PARITY_VERIFIED,
            polars_production_safe=name in operator_policy.POLARS_PRODUCTION_SAFE,
            duckdb_emitter_implemented=name in sql_tiers.SQL_IMPLEMENTED_CANONICALS,
            duckdb_reference_parity=name in sql_tiers.SQL_PARITY_VERIFIED_CANONICALS,
            duckdb_production_safe=name in sql_tiers.SQL_PRODUCTION_SAFE_CANONICALS,
            clickhouse_production_safe=name in sql_tiers.CLICKHOUSE_SQL_PRODUCTION_SAFE,
            polars_source=source,
        )
    return result


def backend_contract_errors() -> list[str]:
    synchronize_active_backend_sets()
    active = _active_runtime_names()
    errors: list[str] = []

    from cleaned_operators import operator_policy, operator_surface
    from backend import primitive_evidence, sql_tiers

    if not operator_policy.POLARS_PARITY_VERIFIED <= active:
        errors.append("Polars parity set contains inactive canonicals")
    if not operator_policy.POLARS_PRODUCTION_SAFE <= operator_policy.POLARS_PARITY_VERIFIED | frozenset(
        name for name in active if name in operator_policy.POLARS_PRODUCTION_SAFE_CORE
    ):
        errors.append("Polars production-safe set lacks parity/core evidence")
    if not (sql_tiers.SQL_IMPLEMENTED_CANONICALS - STRUCTURAL_SQL_NODES) <= active:
        errors.append("SQL implemented set contains inactive canonicals")
    if not sql_tiers.SQL_PRODUCTION_SAFE_CANONICALS <= sql_tiers.SQL_PARITY_VERIFIED_CANONICALS:
        errors.append("DuckDB production-safe set is not a subset of parity")

    for attribute in (
        "POLARS_REFERENCE_PARITY_VERIFIED",
        "DUCKDB_REAL_SQL_VERIFIED",
        "PRIMITIVE_BACKEND_EXECUTION_CERTIFIED",
    ):
        stale = set(getattr(primitive_evidence, attribute, ())) - active
        if stale:
            errors.append(f"{attribute} contains inactive canonicals: {sorted(stale)}")

    daily = set(operator_surface.DAILY_CANONICALS)
    extended = set(getattr(operator_surface, "EXTENDED_ONLY_CANONICALS", ()))
    overlap = daily & extended
    if overlap:
        errors.append(f"daily/extended surface overlap: {sorted(overlap)}")

    for name, item in capability_matrix().items():
        if item.polars_production_safe and not item.polars_no_pandas_bridge:
            errors.append(f"{name}: Polars safe backend is a bridge")
        if item.polars_production_safe and not item.polars_registered:
            errors.append(f"{name}: Polars safe without registered backend")
        if item.duckdb_production_safe and not item.duckdb_reference_parity:
            errors.append(f"{name}: DuckDB safe without reference parity")
    return errors


__all__ = [
    "BackendCapability",
    "STRUCTURAL_SQL_NODES",
    "synchronize_active_backend_sets",
    "capability_matrix",
    "backend_contract_errors",
]
