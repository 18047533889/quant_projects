"""Immutable field and source-table semantic contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .units import canonical_unit, unit_normalization


FIELD_CATALOG_SCHEMA_VERSION = "factor_engine.fields.v2"


@dataclass(frozen=True)
class FieldSpec:
    """Semantic description of one logical factor input field.

    ``name`` remains the compatibility spelling for the canonical DSL field.
    The v2 attributes make temporal, unit, cardinality, and mining semantics
    explicit while retaining the v1 constructor surface.
    """

    name: str
    table: str
    source_name: str
    dtype: str = "float64"
    unit: str = "dimensionless"
    frequency: str = "daily"
    aliases: tuple[str, ...] = ()
    description: str = ""
    nullable: bool = True
    role: str = "feature"
    dataset: str | None = None
    adjustment: str | None = None
    field_id: str | None = None
    domain: str = "auxiliary"
    value_kind: str = "numeric"
    temporal_model: str = "exact"
    price_basis: str | None = None
    flow_semantics: str | None = None
    grain: tuple[str, ...] | str = ("instrument", "time")
    cardinality: str = "many_to_one"
    time_column: str | None = None
    instrument_column: str | None = None
    knowledge_time_column: str | None = None
    effective_time_column: str | None = None
    period_id_column: str | None = None
    revision_columns: tuple[str, ...] = ()
    source_unit: str | None = None
    canonical_unit: str | None = None
    scale_to_canonical: float | None = None
    strict_pit_allowed: bool = True
    null_policy: str = "preserve"
    required_filters: tuple[str, ...] = ()
    applicability: tuple[str, ...] = ()
    allowed_operator_families: tuple[str, ...] = ()
    mining_allowed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        canonical = canonical_unit(self.canonical_unit or self.unit)
        source = canonical_unit(self.source_unit or canonical)
        scale = (
            float(self.scale_to_canonical)
            if self.scale_to_canonical is not None
            else float(unit_normalization(source, canonical).multiplier)
        )
        object.__setattr__(self, "unit", canonical)
        object.__setattr__(self, "canonical_unit", canonical)
        object.__setattr__(self, "source_unit", source)
        object.__setattr__(self, "scale_to_canonical", scale)
        object.__setattr__(self, "field_id", self.field_id or f"{self.table}.{self.name}")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        raw_grain = self.grain
        if isinstance(raw_grain, str):
            raw_grain = tuple(part for part in raw_grain.split("_") if part)
        object.__setattr__(self, "grain", tuple(raw_grain))
        if self.flow_semantics is None:
            # Derive the reporting-flow semantics from the economic grain when a
            # catalog field does not declare it explicitly.  A-share
            # Income/CashFlow statements carry ``flow_ytd`` (fiscal-year-to-date
            # cumulative); StockBalance carries ``balance`` (point-in-time stock).
            raw_grain_set = set(self.grain)
            if "ytd" in raw_grain_set:
                object.__setattr__(self, "flow_semantics", "cumulative_ytd_flow")
            elif "balance" in raw_grain_set:
                object.__setattr__(self, "flow_semantics", "stock")
        object.__setattr__(self, "revision_columns", tuple(self.revision_columns))
        object.__setattr__(self, "required_filters", tuple(self.required_filters))
        object.__setattr__(self, "applicability", tuple(self.applicability))
        object.__setattr__(
            self, "allowed_operator_families", tuple(self.allowed_operator_families)
        )
        if self.role in {
            "time", "instrument", "label", "identifier", "group_key",
            "knowledge_time", "effective_time", "period_id", "ingestion_time",
        }:
            object.__setattr__(self, "mining_allowed", False)

    @property
    def canonical_name(self) -> str:
        return self.name

    @property
    def physical_name(self) -> str:
        return self.source_name

    @property
    def qualified_name(self) -> str:
        return f"{self.table}.{self.source_name}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "aliases", "grain", "revision_columns", "required_filters",
            "applicability", "allowed_operator_families",
        ):
            result[key] = list(result[key])
        result["metadata"] = dict(sorted(self.metadata.items()))
        return result


@dataclass(frozen=True)
class TableSpec:
    """Logical table contract used by resolution, joins, and catalog export."""

    name: str
    dataset: str
    time_column: str | None = "TradeDate"
    instrument_column: str | None = "Symbol"
    frequency: str = "daily"
    domain: str = "auxiliary"
    aliases: tuple[str, ...] = ()
    description: str = ""
    fields: tuple[str, ...] = ()
    table_kind: str = "panel"
    join_policy: str = "exact"
    required_parameters: tuple[str, ...] = ()
    knowledge_time_column: str | None = None
    effective_time_column: str | None = None
    period_id_column: str | None = None
    revision_column: str | None = None
    timezone: str | None = None
    session_calendar: str | None = None
    dedupe_keys: tuple[str, ...] = ()
    revision_order: tuple[str, ...] = ()
    current_snapshot_only: bool = False
    cardinality: str = "many_to_one"
    strict_pit_allowed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        for key in (
            "aliases", "fields", "required_parameters", "dedupe_keys", "revision_order"
        ):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if self.current_snapshot_only:
            object.__setattr__(self, "strict_pit_allowed", False)

    @property
    def logical_table(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "aliases", "fields", "required_parameters", "dedupe_keys", "revision_order"
        ):
            result[key] = list(result[key])
        result["metadata"] = dict(sorted(self.metadata.items()))
        return result


__all__ = ["FIELD_CATALOG_SCHEMA_VERSION", "FieldSpec", "TableSpec"]
