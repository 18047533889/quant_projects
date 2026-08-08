"""IR result schema with optional field semantics."""

from dataclasses import dataclass

from .types import ValueType, normalize_dtype


@dataclass(frozen=True)
class Schema:
    """描述某个 IR 子树输出结果的静态 schema。

    Round-6 P0-50: the schema now carries the PIT/availability coordinate a
    factor needs to prove it is decision-time-safe — ``available_at`` (the
    latest input knowledge-time descriptor, propagated bottom-up),
    ``knowledge_model``, ``fiscal_grain``, ``source_vintage`` and
    ``universe_id``.  These are *semantic descriptors*, not concrete dates: at
    compile time the actual clock is data-dependent, so the schema records what
    governs the availability (session_close / next_session_open / filing / ...)
    and the analyzer propagates ``available_at = max(all_inputs)`` up the IR.
    """

    value_type: ValueType  # scalar / series / panel
    dtype: str             # 如 float64
    index: tuple[str, ...]  # MultiIndex 层名，如 ("timestamp", "instrument")
    unit: str | None = None
    field_name: str | None = None
    source_table: str | None = None
    source_field: str | None = None
    frequency: str = "daily"
    temporal_model: str = "exact"
    cardinality: str = "many_to_one"
    domain: str = "auxiliary"
    price_basis: str | None = None  # RAW / CONTINUOUS / RAW_OFFICIAL_LIMIT / RETURN
    flow_semantics: str | None = None  # stock / single_period_flow / cumulative_ytd_flow / ttm_flow
    pit_safe: bool = True
    nullable: bool = True
    # Round-6 P0-50 PIT/availability coordinate.  ``available_at`` is a
    # knowledge-time descriptor propagated as ``max(all_input.available_at)``
    # (audit §51: factor.available_at = max(all_input.available_at,
    # operator_processing_latency)).
    available_at: str | None = None
    knowledge_model: str | None = None  # exact / asof / financial_pit / next_trading_day ...
    fiscal_grain: str | None = None  # flow / ytd / balance / ttm (derived from flow_semantics)
    source_vintage: str | None = None
    universe_id: str | None = None

    def __post_init__(self) -> None:
        # Round-6 P0-50: derive the fiscal grain from flow semantics when the
        # caller did not set it explicitly (stock -> balance, single-period flow
        # -> flow, YTD cumulative -> ytd, TTM -> ttm).
        if self.fiscal_grain is None and self.flow_semantics is not None:
            derived = _fiscal_grain_of(self.flow_semantics)
            if derived is not None:
                object.__setattr__(self, "fiscal_grain", derived)
        if self.knowledge_model is None and self.temporal_model not in {"exact", "unknown"}:
            object.__setattr__(self, "knowledge_model", self.temporal_model)

    @classmethod
    def from_field(cls, spec, *, value_type: ValueType = ValueType.PANEL) -> "Schema":
        """Create a panel schema from a ``FieldSpec`` without coupling imports.

        Round-6 P0-50: maps the field's knowledge-time column / temporal model /
        flow semantics / vintage / universe onto the schema's PIT coordinate.
        """
        flow_semantics = getattr(spec, "flow_semantics", None)
        return cls(
            value_type=value_type,
            dtype=normalize_dtype(spec.dtype),
            index=("timestamp", "instrument"),
            unit=spec.unit,
            field_name=spec.name,
            source_table=spec.table,
            source_field=spec.source_name,
            frequency=spec.frequency,
            temporal_model=spec.temporal_model,
            cardinality=spec.cardinality,
            domain=spec.domain,
            price_basis=getattr(spec, "price_basis", None),
            flow_semantics=flow_semantics,
            pit_safe=spec.strict_pit_allowed,
            nullable=spec.nullable,
            available_at=_available_at_of(spec),
            knowledge_model=_knowledge_model_of(getattr(spec, "temporal_model", None)),
            fiscal_grain=_fiscal_grain_of(flow_semantics),
            source_vintage=getattr(spec, "revision_columns", None),
            universe_id=getattr(spec, "universe_id", None),
        )

    def semantic_kind(self):
        """Map this schema's propagated attrs to a typed-IR semantic kind.

        Uses :func:`ir.types.semantic_type_of`; ``None`` means no semantic kind
        is pinned down (generic derived numeric series).
        """
        from .types import semantic_type_of

        return semantic_type_of(
            price_basis=self.price_basis,
            flow_semantics=self.flow_semantics,
            frequency=self.frequency,
            domain=self.domain,
        )


# Round-6 P0-50: availability descriptors, ranked by lateness.  A factor's
# ``available_at`` is the descriptor of the LATEST input — ``max(all_inputs)``
# by this ordering — because the factor cannot be used for a decision until its
# last-arriving input is knowable (audit §51).  Unknown descriptors sort last
# (conservative: unknown availability is treated as the latest).
_AVAILABILITY_RANK = [
    "unknown",
    "midnight",
    "session_open",
    "pre_close",
    "local_close",
    "session_close",
    "after_close",
    "next_session_open",
    "next_trading_day",
    "filing_date",
    "pub_date",
    "declaration_date",
    "knowledge_time",
]


def _rank_availability(descriptor: str | None) -> int:
    if descriptor is None:
        return len(_AVAILABILITY_RANK)  # unknown sorts latest (conservative)
    low = str(descriptor).lower()
    for index, candidate in enumerate(_AVAILABILITY_RANK):
        if candidate in low:
            return index
    return len(_AVAILABILITY_RANK)


def propagate_available_at(descriptors: tuple[str | None, ...]) -> str | None:
    """Return the latest availability descriptor across inputs (audit §51).

    Only KNOWN descriptors compete — an input whose availability is unknown
    (``None``) must not silently win the max (that would discard the latest
    known input's constraint).  If every input is unknown, returns ``None``.
    """
    known = [descriptor for descriptor in descriptors if descriptor is not None]
    if not known:
        return None
    ranked = [(descriptor, _rank_availability(descriptor)) for descriptor in known]
    latest = max(ranked, key=lambda pair: pair[1])
    return latest[0]


def _knowledge_model_of(temporal_model: str | None) -> str | None:
    if not temporal_model or temporal_model in {"exact", "unknown"}:
        return None
    return str(temporal_model)


# Round-6 P0-13/14/15: EOD fields cannot be known until the session closes; the
# day's open only after the opening auction.  When a field carries no explicit
# knowledge-time column these name-based defaults stop a factor that reads
# Close_t / VWAP_t / Volume_t from being treated as available intra-day.
_EOD_AVAILABILITY_MARKERS = (
    "close", "high", "low", "vwap", "volume", "amount", "turnover", "limit_up",
    "limit_down", "adjfactor", "factor", "market_cap", "pe", "pb", "pct_chg",
)


def _available_at_of(spec) -> str | None:
    """Explicit knowledge time first; name-based EOD default second."""
    explicit = getattr(spec, "available_at", None) or getattr(spec, "knowledge_time_column", None)
    if explicit:
        return str(explicit)
    name = str(getattr(spec, "name", "") or "")
    low = name.lower()
    if low in {"open", "pre_open"}:
        return "session_open"
    if any(marker in low for marker in _EOD_AVAILABILITY_MARKERS):
        return "session_close"
    return None


def _fiscal_grain_of(flow_semantics: str | None) -> str | None:
    mapping = {
        "stock": "balance",
        "single_period_flow": "flow",
        "cumulative_ytd_flow": "ytd",
        "ttm_flow": "ttm",
    }
    return mapping.get(str(flow_semantics)) if flow_semantics else None


DEFAULT_COLUMN_SCHEMA = Schema(ValueType.PANEL, "float64", ("timestamp", "instrument"))


__all__ = [
    "DEFAULT_COLUMN_SCHEMA",
    "Schema",
    "propagate_available_at",
]
