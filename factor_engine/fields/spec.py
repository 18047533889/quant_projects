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
    # R10-P0-014: the safe default is UNKNOWN (None), not "allowed".  A field
    # whose PIT / mining eligibility was never declared must NOT silently be
    # PIT-safe / mineable as hundreds of fields are added; production gates
    # treat ``None`` as fail-closed.  The catalog helpers (``_f`` / ``_table``)
    # still pass their explicit True/False, so catalog semantics are unchanged.
    strict_pit_allowed: bool | None = None
    null_policy: str = "preserve"
    required_filters: tuple[str, ...] = ()
    applicability: tuple[str, ...] = ()
    allowed_operator_families: tuple[str, ...] = ()
    mining_allowed: bool | None = None
    # Round-7 WS-C (#269-#273): the *declared* typed-IR semantic kind.  When set,
    # it is authoritative — callers MUST NOT guess from the field name (review
    # #269).  ``None`` means "not declared"; the mapping helper
    # :func:`semantic_kind_of_field` may then fall back to name-derived kinds
    # only for raw/research columns.
    semantic_kind: str | None = None
    # Round-7 WS-C (#277): explicit availability descriptor (``session_close`` /
    # ``PubDate`` / ``filing`` / ...).  When absent, availability is derived from
    # ``knowledge_time_column`` / ``role`` / exact-name defaults — never from
    # substring matching on the field name.
    available_at: str | None = None
    # Round-7 WS-C (#277): canonical concept id (e.g. ``price.close``,
    # ``volume``).  Availability / semantics may key off the concept instead of
    # the physical column name.
    concept: str | None = None
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
            # R40 #131/#132: 完整推导矩阵 —— 拆开 period_duration（annual /
            # quarter / single_period）与 flow 语义；``flow_annual`` 推导
            # ``annual_flow``，``flow_quarter`` / ``single_period`` 推导
            # ``single_period_flow``。带 period_duration 但不带 ``flow`` 的 grain
            # （如 annual balance sheet）保持 None（不是 flow）。
            raw_grain_set = set(self.grain)
            if "balance" in raw_grain_set or "stock" in raw_grain_set:
                object.__setattr__(self, "flow_semantics", "stock")
            elif "ytd" in raw_grain_set:
                object.__setattr__(self, "flow_semantics", "cumulative_ytd_flow")
            elif "ttm" in raw_grain_set:
                object.__setattr__(self, "flow_semantics", "ttm_flow")
            elif "annual" in raw_grain_set and "flow" in raw_grain_set:
                object.__setattr__(self, "flow_semantics", "annual_flow")
            elif "quarter" in raw_grain_set and "flow" in raw_grain_set:
                object.__setattr__(self, "flow_semantics", "single_period_flow")
            elif "single_period" in raw_grain_set:
                object.__setattr__(self, "flow_semantics", "single_period_flow")
            elif "flow" in raw_grain_set:
                object.__setattr__(self, "flow_semantics", "single_period_flow")
        object.__setattr__(self, "revision_columns", tuple(self.revision_columns))
        object.__setattr__(self, "required_filters", tuple(self.required_filters))
        object.__setattr__(self, "applicability", tuple(self.applicability))
        object.__setattr__(
            self, "allowed_operator_families", tuple(self.allowed_operator_families)
        )
        if self.role in {
            "time", "instrument", "label", "identifier", "group_key",
            "knowledge_time", "effective_time", "period_id", "ingestion_time",
            # R10-P0-015: a STATUS role is a categorical market-state descriptor
            # (listing_status / ST_status / trade_status).  It is a condition /
            # mask / group / state-transition input, never a mined numeric
            # feature — ``ts_mean(status)`` / ``rank(status)`` are meaningless.
            "status",
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
    # R10-P0-014: UNKNOWN (None) by default — production PIT gates fail closed
    # unless the table explicitly declares PIT eligibility.
    strict_pit_allowed: bool | None = None
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


# ---------------------------------------------------------------------------
# Round-7 WS-C (#269-#273): semantic-kind mapping helper.
#
# A FieldSpec with an explicit ``semantic_kind`` is authoritative — the helper
# never overrides it with a name guess.  For raw/research columns (or catalog
# fields that carry no explicit kind) the name-based fallback below covers the
# canonical activity / mask / group roles without touching price-basis /
# flow-semantics derivation, which lives in ``ir.types.semantic_type_of``.
# ---------------------------------------------------------------------------
#: name -> declared semantic kind for the well-known non-price activity roles.
_SEMANTIC_KIND_NAME_MAP = {
    "volume": "NonNegativeActivity",
    "volume_ratio": "NonNegativeActivity",
    "amount": "NonNegativeActivity",
    "amount_ratio": "NonNegativeActivity",
    "turnover": "NonNegativeActivity",
    "turnover_ratio": "NonNegativeActivity",
    "universe_mask": "MaskBool",
    "mask": "MaskBool",
    "group_id": "GroupKey",
    "group": "GroupKey",
    "event": "EventBool",
    "is_event": "EventBool",
    "event_flag": "EventBool",
}


def semantic_kind_of_field(
    spec_or_name,
    *,
    production: bool | None = None,
) -> str | None:
    """Resolve a field's typed-IR semantic kind.

    Priority (review #269 — never guess where a kind is declared):
    1. ``FieldSpec.semantic_kind`` (explicit, authoritative);
    2. ``price_basis`` / ``flow_semantics`` derivation
       (:func:`ir.types.semantic_type_of`);
    3. exact-name fallback for raw/research columns (Volume -> NonNegativeActivity,
       UniverseMask -> MaskBool, GroupId -> GroupKey, ...).

    R40 #220: production 模式下 name fallback **禁用** —— 未显式声明
    ``semantic_kind`` 且 price_basis/flow_semantics 不可证的字段语义是 UNKNOWN
    （返回 ``None``），由调用方 fail legality；绝不在 production 用名字猜
    （同形/改名列会静默得到错误语义）。name fallback 只在 research raw-column
    兼容用。``production=None`` 时按 run mode 自动解析。
    """
    if spec_or_name is None:
        return None
    if hasattr(spec_or_name, "semantic_kind"):
        declared = getattr(spec_or_name, "semantic_kind", None)
        if declared:
            return str(declared)
    price_basis = getattr(spec_or_name, "price_basis", None)
    flow_semantics = getattr(spec_or_name, "flow_semantics", None)
    if price_basis or flow_semantics:
        from factor_engine.ir.types import semantic_type_of

        kind = semantic_type_of(
            price_basis=price_basis,
            flow_semantics=flow_semantics,
            frequency=getattr(spec_or_name, "frequency", None),
            domain=getattr(spec_or_name, "domain", None),
        )
        if kind is not None:
            return kind.value
    if production is None:
        from factor_engine.runtime.production_policy import is_production_mode

        production = is_production_mode()
    if production:
        # R40 #220: production 拒绝对未声明字段做 name-based 语义猜测。
        return None
    name = getattr(spec_or_name, "name", spec_or_name)
    low = str(name or "").lower()
    return _SEMANTIC_KIND_NAME_MAP.get(low)


__all__ = [
    "FIELD_CATALOG_SCHEMA_VERSION",
    "FieldSpec",
    "TableSpec",
    "semantic_kind_of_field",
]
