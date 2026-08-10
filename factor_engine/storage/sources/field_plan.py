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

import enum
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
        required_filters: mandatory dataset filters that must be applied to read
            this field safely (mirrored from the catalog / FE FieldSpec, round-7
            WS-E #281).
        applicability: market/universe applicability labels (round-7 WS-E #281).
        allowed_operator_families: operator families permitted over this field
            (round-7 WS-E #281).
        null_policy: preserve / drop / zero_fill ... declared by the field
            (round-7 WS-E #281/#316).
        semantic_kind: typed semantic kind (e.g. ReturnDecimal / PriceRaw /
            NonNegativeActivity) when the field declares one (round-7 WS-E #281).
        strict_pit_allowed: field-level PIT eligibility (round-7 WS-E #282).
        current_snapshot_only: the field must never backfill historical panels
            (round-7 WS-E #280).
        role: catalog field role (feature / time / instrument / ...); drives
            mining and MissingSemantic inference (round-7 WS-E #284/#316).
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
    # R17-003: ``universe`` was overloaded with three different meanings across
    # the two plan builders (registry path -> spec.domain, catalog path ->
    # field.market).  It is now a DEPRECATED compatibility view; the three
    # explicit fields below are authoritative.
    universe: str | None = None
    market: str | None = None
    universe_id: str | None = None
    semantic_domain: str | None = None
    coverage: str | None = None
    mining_allowed: bool = True
    price_basis: str | None = None
    flow_semantics: str | None = None
    source: str = "raw"
    required_filters: tuple[str, ...] = field(default_factory=tuple)
    applicability: tuple[str, ...] = field(default_factory=tuple)
    allowed_operator_families: tuple[str, ...] = field(default_factory=tuple)
    null_policy: str = "preserve"
    semantic_kind: str | None = None
    # R17-005: tri-state PIT eligibility.  ``None`` = UNKNOWN (never declared).
    # Production treats ``None`` as fail-closed (cannot prove PIT-safe is not
    # PIT-safe); the raw-plan default was ``True``, which silently certified
    # every undeclared field as PIT-eligible.
    strict_pit_allowed: bool | None = None
    current_snapshot_only: bool = False
    role: str = "feature"

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


def _resolve_table_spec(table: str | None, market: str | None) -> TableSpec | None:
    """Resolve a ``TableSpec`` within an EXPLICIT market (R17-002).

    Same-named tables carry DIFFERENT semantics per market (US
    ``StockValuationDaily`` is an X0 current snapshot while A-share's is a full
    D1; ``StockCapitalDaily`` is split/dual-schema in US vs S1 in A-share).
    Resolving a bare table name against the legacy A-share-only
    ``FIELD_REGISTRY`` would interpret a US table with A-share semantics.
    """
    if not table:
        return None
    try:
        if market:
            from fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

            return MULTI_MARKET_FIELD_REGISTRY.registry_for(market).resolve_table(
                str(table)
            )
        from fields import FIELD_REGISTRY

        return FIELD_REGISTRY.resolve_table(str(table))
    except Exception:  # pragma: no cover - defensive
        return None


def _table_current_snapshot_only(table: str | None, market: str | None) -> bool:
    """Look up a table's ``current_snapshot_only`` within one market."""
    table_spec = _resolve_table_spec(table, market)
    return bool(table_spec is not None and getattr(table_spec, "current_snapshot_only", False))


def _market_from_dataset(dataset: str | None) -> str | None:
    """Infer the market from a physical dataset name when no explicit market.

    DataAccess dataset names are market-prefixed (``ashare_stock_*`` /
    ``us_stock_*``).  ``None`` when the dataset does not declare a market.
    """
    if not dataset:
        return None
    low = str(dataset).strip().lower()
    if low.startswith("us_"):
        return "us"
    if low.startswith("ashare_") or low.startswith("cn_"):
        return "ashare"
    return None


def plan_from_field_spec(name: str, spec: Any, *, market: str | None = None) -> NormalizedFieldPlan:
    """Build a plan from a FactorEngine ``FieldSpec`` (registry).

    R17-003: ``universe`` is split into ``market`` / ``universe_id`` /
    ``semantic_domain``.  The registry path now keeps the three explicit
    fields; ``universe`` remains a deprecated compatibility view.
    """
    dataset = getattr(spec, "dataset", None)
    eff_market = market or _market_from_dataset(dataset)
    domain = getattr(spec, "domain", None)
    return NormalizedFieldPlan(
        logical_concept=name,
        physical_dataset=dataset,
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
        universe=domain if eff_market is None else eff_market,
        market=eff_market,
        universe_id=(
            str(getattr(spec, "universe_id", None))
            if getattr(spec, "universe_id", None) is not None
            else None
        ),
        semantic_domain=domain,
        coverage=_coverage_from_temporal_model(getattr(spec, "temporal_model", None)),
        mining_allowed=bool(getattr(spec, "mining_allowed", True)),
        price_basis=getattr(spec, "price_basis", None),
        flow_semantics=getattr(spec, "flow_semantics", None),
        source="registry",
        required_filters=tuple(getattr(spec, "required_filters", ()) or ()),
        applicability=tuple(getattr(spec, "applicability", ()) or ()),
        allowed_operator_families=tuple(
            getattr(spec, "allowed_operator_families", ()) or ()
        ),
        null_policy=str(getattr(spec, "null_policy", "preserve") or "preserve"),
        semantic_kind=getattr(spec, "semantic_kind", None),
        strict_pit_allowed=getattr(spec, "strict_pit_allowed", None),
        current_snapshot_only=_table_current_snapshot_only(getattr(spec, "table", None), eff_market),
        role=str(getattr(spec, "role", "feature") or "feature"),
    )


def plan_from_catalog_field(name: str, field: Any, *, market: str | None = None) -> NormalizedFieldPlan:
    """Build a plan from a DataAccess ``SemanticField`` (catalog).

    ``mining_allowed`` and ``coverage`` are copied from the catalog semantic
    field instead of being dropped (round-7 P0); a field that carries a
    ``derived_expression`` is marked derived (``transform`` + ``derived_from``)
    so the resolve path never reads its raw physical column as the value.

    R17-003: ``market`` comes from the catalog field's market, ``semantic_domain``
    from its domain, ``universe`` is deprecated.
    """
    dataset = getattr(field, "dataset", None)
    eff_market = market or getattr(field, "market", None) or _market_from_dataset(dataset)
    domain = getattr(field, "domain", None)
    return NormalizedFieldPlan(
        logical_concept=name,
        physical_dataset=dataset,
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
        universe=eff_market,
        market=eff_market,
        universe_id=(
            str(getattr(field, "universe_id", None))
            if getattr(field, "universe_id", None) is not None
            else None
        ),
        semantic_domain=domain,
        coverage=_catalog_coverage(field),
        mining_allowed=bool(getattr(field, "mining_allowed", True)),
        price_basis=getattr(field, "price_basis", None),
        flow_semantics=getattr(field, "flow_semantics", None),
        source="catalog",
        required_filters=tuple(getattr(field, "required_filters", ()) or ()),
        applicability=tuple(getattr(field, "applicability", ()) or ()),
        allowed_operator_families=tuple(
            getattr(field, "allowed_operator_families", ()) or ()
        ),
        null_policy=str(getattr(field, "null_policy", "preserve") or "preserve"),
        semantic_kind=getattr(field, "semantic_kind", None),
        strict_pit_allowed=getattr(field, "strict_pit_allowed", None),
        current_snapshot_only=_table_current_snapshot_only(getattr(field, "table", None), eff_market),
        role=str(getattr(field, "role", "feature") or "feature"),
    )


class MissingSemantic(enum.Enum):
    """Semantic meaning of a missing value for a field (round-7 WS-E #316).

    Sparse-event fields (announcement / event) treat a missing value as
    ``NO_EVENT`` (there was no announcement that day); dense fundamental / price
    fields treat missing as ``UNKNOWN`` (the value is expected but not observed
    yet).  ``NOT_APPLICABLE`` marks fields that only exist for a subset of
    instruments/dates; ``NOT_TRADING`` marks suspension / no-session gaps;
    ``STRUCTURAL_ZERO`` marks fields whose absence is economically a zero.
    """

    UNKNOWN = "unknown"
    NO_EVENT = "no_event"
    NOT_APPLICABLE = "not_applicable"
    NOT_TRADING = "not_trading"
    STRUCTURAL_ZERO = "structural_zero"


#: Field temporal models whose missing cells are semantically "no event row
#: occurred" rather than "value unknown" (R17-004).
#:
#: R17-004: ``financial_event`` / ``financial_pit`` are REMOVED from this set.
#: A financial statement table that has no row on a date means "no filing that
#: day" (the event absent), but a row that EXISTS with a NaN numeric field is
#: an UNKNOWN value — mapping the whole financial field to NO_EVENT conflates
#: "today had no announcement" with "the announced number is missing" and lets
#: event-count / zero-fill / mask operators wrongly consume a missing financial.
#: Only pure event/announcement tables (no row == no event) stay NO_EVENT.
_NO_EVENT_TEMPORAL_MODELS = frozenset({
    "sparse_event", "effective_only", "event", "event_series",
})
#: R17-004: ``partial_history`` / ``current_snapshot`` are removed — a
#: partial-history financial field missing a date is UNKNOWN (expected but not
#: yet observed), and a current-snapshot table not covering a historical date is
#: NOT_APPLICABLE / OUT_OF_COVERAGE (a different semantic than "no event").
_NO_EVENT_COVERAGE = frozenset({"sparse_event"})


def missing_semantic_for_plan(plan: Any) -> MissingSemantic:
    """Map a field (``NormalizedFieldPlan`` / ``FieldSpec``) to its MissingSemantic.

    R17-004 splits missing semantics (never collapse them through one temporal
    model):

    * event/announcement table has no row          -> ``NO_EVENT``
    * an existing row has a NaN numeric field       -> ``UNKNOWN``
    * current-snapshot table not covering a date    -> ``NOT_APPLICABLE``
    * field structurally absent for the security    -> ``NOT_APPLICABLE``
    * suspension / no session                       -> ``NOT_TRADING``
    * explicit structural zero only                 -> ``STRUCTURAL_ZERO``

    A financial field (``financial_pit`` / ``partial_history``) whose numeric
    value is missing resolves to ``UNKNOWN`` — it is never reinterpreted as
    "no event".
    """
    null_policy = str(getattr(plan, "null_policy", "preserve") or "preserve").lower()
    if null_policy in {"zero", "zero_fill", "as_zero", "structural_zero"}:
        return MissingSemantic.STRUCTURAL_ZERO
    temporal_model = str(getattr(plan, "temporal_model", "") or "").strip().lower()
    # Financial event rows: an existing row's numeric NaN is UNKNOWN, but a
    # *financial event table* with no row on a date still means NO_EVENT for the
    # event (e.g. no dividend that day).  Only the row-level numeric missing is
    # UNKNOWN — that is a different object than the table-level absence.
    if temporal_model in _NO_EVENT_TEMPORAL_MODELS:
        return MissingSemantic.NO_EVENT
    if temporal_model in {"financial_event", "financial_pit"}:
        role = str(getattr(plan, "role", "") or "").lower()
        if role in {"knowledge_time", "period_id", "effective_time", "ingestion_time"}:
            return MissingSemantic.NOT_APPLICABLE
        return MissingSemantic.UNKNOWN
    coverage = str(getattr(plan, "coverage", "") or "").strip().lower()
    if coverage in _NO_EVENT_COVERAGE:
        return MissingSemantic.NO_EVENT
    if coverage == "current_snapshot":
        return MissingSemantic.NOT_APPLICABLE
    role = str(getattr(plan, "role", "") or "").lower()
    if role in {"knowledge_time", "period_id", "effective_time", "ingestion_time"}:
        return MissingSemantic.NOT_APPLICABLE
    return MissingSemantic.UNKNOWN


__all__ = [
    "MissingSemantic",
    "NormalizedFieldPlan",
    "missing_semantic_for_plan",
    "plan_from_catalog_field",
    "plan_from_field_spec",
]
