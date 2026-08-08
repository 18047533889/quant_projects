"""Unified semantic field plan shared by all storage sources (P0-11).

A ``NormalizedFieldPlan`` is the single plan object every source's column
resolution produces for one logical field, mapping the logical concept to its
physical read shape and its unit/scale normalization.  Scale normalization
(``_normalize_contract_columns`` / ``_normalize_from_plans`` / the polars-long
scan step) reads from the *same* plan object, so a ``Return(bp)`` field
normalizes identically across the DataAccess, ClickHouse and long-table sources
instead of each source re-deriving scale from a different registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedFieldPlan:
    """One logical field resolved to its physical read + normalization contract.

    Attributes:
        logical_concept: the requested logical field name.
        physical_dataset: registry/dataset name that owns the field (if known).
        physical_fields: physical column(s) to read for this concept.
        transform: optional derived-field expression (when not a plain column).
        derived_from: physical (dataset.column) dependencies of a derived field.
        canonical_unit: canonical unit the field is normalized into.
        scale: multiplier applied at the scan boundary to reach the canonical unit.
        frequency: daily / minute / tick / quarterly ... declared by the field.
        grain: economic grain, e.g. (instrument, time) or (flow, ytd).
        knowledge_time: knowledge-time (data visibility) column, if any.
        effective_time: effective-time column, if any.
        revision_order: version/revision columns to dedupe by before joining.
        universe: market / universe scope (ashare / us / any ...).
        coverage: semantic field coverage — historical_panel / current_snapshot /
            sparse_event / partial_history (round-7 P0, mirrored from the
            SemanticFieldCatalog temporal model when not declared).
        mining_allowed: whether the field may be used as a mined factor input
            (mirrored from the catalog / FE FieldSpec; round-7 P0).
        price_basis: RAW / CONTINUOUS / RAW_OFFICIAL_LIMIT / RETURN.
        flow_semantics: stock / single_period_flow / cumulative_ytd_flow / ttm_flow.
        source: provenance discriminator used by normalization:
            "catalog" (SemanticFieldCatalog), "registry" (FE FIELD_REGISTRY),
            or "raw" (no registered contract — research pass-through).
    """

    logical_concept: str
    physical_dataset: str | None = None
    physical_fields: tuple[str, ...] = field(default_factory=tuple)
    transform: str | None = None
    derived_from: tuple[str, ...] = field(default_factory=tuple)
    canonical_unit: str | None = None
    scale: float | None = None
    frequency: str | None = None
    grain: tuple[str, ...] | str | None = None
    knowledge_time: str | None = None
    effective_time: str | None = None
    revision_order: tuple[str, ...] = field(default_factory=tuple)
    universe: str | None = None
    coverage: str | None = None
    mining_allowed: bool = True
    price_basis: str | None = None
    flow_semantics: str | None = None
    source: str = "raw"

    @property
    def is_scale_applicable(self) -> bool:
        """True when this plan carries a real unit/scale normalization."""
        return self.scale is not None and float(self.scale) != 1.0

    @property
    def primary_physical(self) -> str | None:
        """First physical column, or None when the plan has no physical read."""
        return self.physical_fields[0] if self.physical_fields else None

    @property
    def is_derived(self) -> bool:
        """True when this plan is a derived field, not a plain physical column.

        A derived field carries a ``transform`` (derived expression); its
        ``primary_physical`` (if any) is the base read, never the derived value
        itself (round-7 P0).
        """
        return bool(self.transform)


#: Coverage semantics derived from a field's temporal model when the field does
#: not declare ``coverage`` explicitly.  The vocabulary matches the DataAccess
#: SemanticFieldCatalog temporal_model values (round-7 P0).
_COVERAGE_FROM_TEMPORAL_MODEL = {
    "panel": "historical_panel",
    "exact": "historical_panel",
    "exact_daily": "historical_panel",
    "financial_event": "partial_history",
    "financial_pit": "partial_history",
    "sparse_event": "sparse_event",
    "sparse_snapshot": "current_snapshot",
    "effective_only": "sparse_event",
    "current_snapshot": "current_snapshot",
}


def _coverage_from_temporal_model(value) -> str | None:
    if not value:
        return None
    return _COVERAGE_FROM_TEMPORAL_MODEL.get(str(value).strip().lower())


def _catalog_coverage(field) -> str | None:
    """Coverage for a DataAccess ``SemanticField``: declared or derived."""
    declared = getattr(field, "coverage", None)
    if declared:
        return str(declared)
    return _coverage_from_temporal_model(getattr(field, "temporal_model", None))


def plan_from_field_spec(name: str, spec: Any) -> NormalizedFieldPlan:
    """Build a plan from a FactorEngine ``FieldSpec`` (registry)."""
    return NormalizedFieldPlan(
        logical_concept=name,
        physical_dataset=getattr(spec, "dataset", None),
        physical_fields=(
            (spec.source_name,) if getattr(spec, "source_name", None) else ()
        ),
        canonical_unit=getattr(spec, "canonical_unit", None),
        scale=getattr(spec, "scale_to_canonical", None),
        frequency=getattr(spec, "frequency", None),
        grain=getattr(spec, "grain", None),
        knowledge_time=getattr(spec, "knowledge_time_column", None),
        effective_time=getattr(spec, "effective_time_column", None),
        revision_order=tuple(getattr(spec, "revision_columns", ()) or ()),
        universe=getattr(spec, "domain", None),
        coverage=_coverage_from_temporal_model(getattr(spec, "temporal_model", None)),
        mining_allowed=bool(getattr(spec, "mining_allowed", True)),
        price_basis=getattr(spec, "price_basis", None),
        flow_semantics=getattr(spec, "flow_semantics", None),
        source="registry",
    )


def plan_from_catalog_field(name: str, field: Any) -> NormalizedFieldPlan:
    """Build a plan from a DataAccess ``SemanticField`` (catalog).

    ``mining_allowed`` and ``coverage`` are copied from the catalog semantic
    field instead of being dropped (round-7 P0); a field that carries a
    ``derived_expression`` is marked derived (``transform`` + ``derived_from``)
    so the resolve path never reads its raw physical column as the value.
    """
    return NormalizedFieldPlan(
        logical_concept=name,
        physical_dataset=getattr(field, "dataset", None),
        physical_fields=(
            (field.physical_name,) if getattr(field, "physical_name", None) else ()
        ),
        transform=getattr(field, "derived_expression", None),
        derived_from=tuple(getattr(field, "derived_from", ()) or ()),
        canonical_unit=getattr(field, "canonical_unit", None),
        scale=getattr(field, "scale", None),
        frequency=getattr(field, "frequency", None),
        grain=getattr(field, "grain", None),
        knowledge_time=getattr(field, "knowledge_time", None),
        effective_time=getattr(field, "effective_time", None),
        revision_order=tuple(getattr(field, "revision_order", ()) or ()),
        universe=getattr(field, "market", None),
        coverage=_catalog_coverage(field),
        mining_allowed=bool(getattr(field, "mining_allowed", True)),
        price_basis=getattr(field, "price_basis", None),
        flow_semantics=getattr(field, "flow_semantics", None),
        source="catalog",
    )


__all__ = [
    "NormalizedFieldPlan",
    "plan_from_catalog_field",
    "plan_from_field_spec",
]
