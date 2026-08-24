"""IR result schema with optional field semantics.

Round-7 WS-C (review #265-#278): the schema carries a **semantic lattice**
(:class:`ir.types.SemanticLattice`) and an :class:`ir.types.AvailabilityExpr`
knowledge-time descriptor instead of the string ``_AVAILABILITY_RANK`` total
order.  ``available_at`` remains a *string label* on the public surface for
backward compatibility; the typed expression and structured vintage live beside
it and resolve to concrete timestamps at the data layer.
"""

from dataclasses import dataclass, field

from .types import (
    AvailabilityExpr,
    SemanticLattice,
    SourceVintageSpec,
    UNKNOWN,
    ValueType,
    availability_expr_of,
    latest_availability,
    normalize_dtype,
    semantic_type_of,
)


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

    Round-7 WS-C: ``availability_expr`` is the typed :class:`AvailabilityExpr`
    whose ``label`` is mirrored onto ``available_at``; ``source_vintage`` is a
    structured :class:`SourceVintageSpec`; ``semantic_lattice`` is the
    five-dimension semantic lattice of the subtree.
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
    # R24-129/130: pit_safe defaults to UNKNOWN (None), NOT True — a field whose
    # PIT eligibility is undeclared must never be certified safe by omission.
    # Research raw-compat paths may opt in explicitly.
    pit_safe: bool | None = None
    nullable: bool = True
    # Round-6 P0-50 PIT/availability coordinate.  ``available_at`` is the
    # *label* of the knowledge-time descriptor propagated as
    # ``max(all_input.available_at)`` (audit §51: factor.available_at =
    # max(all_input.available_at, operator_processing_latency)).
    available_at: str | None = None
    # Round-7 WS-C: the typed expression (label mirrors ``available_at``).
    availability_expr: AvailabilityExpr | None = None
    knowledge_model: str | None = None  # exact / asof / financial_pit / next_trading_day ...
    fiscal_grain: str | None = None  # flow / ytd / balance / ttm (derived from flow_semantics)
    source_vintage: SourceVintageSpec | str | None = None
    universe_id: str | None = None
    semantic_lattice: SemanticLattice | None = None

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
        # Round-7 WS-C: keep the typed expression and the label in sync.
        if self.availability_expr is None and self.available_at is not None:
            expr = availability_expr_of(self.available_at)
            if not isinstance(expr, type(UNKNOWN)):
                object.__setattr__(self, "availability_expr", expr)
        if self.availability_expr is not None and self.available_at is None:
            object.__setattr__(self, "available_at", self.availability_expr.label)

    @classmethod
    def from_field(cls, spec, *, value_type: ValueType = ValueType.PANEL) -> "Schema":
        """Create a panel schema from a ``FieldSpec`` without coupling imports.

        Round-6 P0-50: maps the field's knowledge-time column / temporal model /
        flow semantics / vintage / universe onto the schema's PIT coordinate.
        Round-7 WS-C: the availability descriptor is a typed
        :class:`AvailabilityExpr`, the vintage is a :class:`SourceVintageSpec`,
        and the field's semantic attrs are captured in a :class:`SemanticLattice`.
        """
        flow_semantics = getattr(spec, "flow_semantics", None)
        available_at = _available_at_of(spec)
        vintage = SourceVintageSpec.from_field_spec(spec)
        kind = _semantic_kind_of_field_spec(spec)
        lattice = SemanticLattice.from_field_attrs(
            domain=getattr(spec, "domain", None),
            frequency=getattr(spec, "frequency", None),
            source_vintage=vintage,
            universe_id=getattr(spec, "universe_id", None),
            semantic_kind=kind,
        )
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
            available_at=available_at,
            availability_expr=availability_expr_of(available_at),
            knowledge_model=_knowledge_model_of(getattr(spec, "temporal_model", None)),
            fiscal_grain=_fiscal_grain_of(flow_semantics),
            source_vintage=vintage,
            universe_id=getattr(spec, "universe_id", None),
            semantic_lattice=lattice,
        )

    def semantic_kind(self) -> str | None:
        """Map this schema's propagated attrs to a typed-IR semantic kind.

        Uses :func:`ir.types.semantic_type_of`; ``None`` means no semantic kind
        is pinned down (generic derived numeric series).
        """
        if self.semantic_lattice is not None and self.semantic_lattice.scalar_semantic_kind():
            return self.semantic_lattice.scalar_semantic_kind()
        kind = semantic_type_of(
            price_basis=self.price_basis,
            flow_semantics=self.flow_semantics,
            frequency=self.frequency,
            domain=self.domain,
        )
        return kind.value if kind is not None else None


def _semantic_kind_of_field_spec(spec) -> str | None:
    """FieldSpec semantic-kind: declared ``semantic_kind`` first, then helpers."""
    declared = getattr(spec, "semantic_kind", None)
    if declared:
        return str(declared)
    try:
        from factor_engine.fields.spec import semantic_kind_of_field

        return semantic_kind_of_field(spec)
    except ImportError:  # pragma: no cover - spec always importable
        return None


# ---------------------------------------------------------------------------
# Round-7 WS-C availability propagation.
#
# The legacy string ``_AVAILABILITY_RANK`` total order is DELETED.  Ordering now
# lives on the :class:`AvailabilityExpr` classes (``lateness``); the compat shim
# below converts legacy string labels to expressions.  UNKNOWN sorts fail-closed
# (+infinity): a single unknown input makes the root unknown (review #274/#275).
# ---------------------------------------------------------------------------
def _lateness_of(descriptor) -> float:
    return availability_expr_of(descriptor).lateness


def _rank_availability(descriptor) -> float:
    """Backward-compat shim: lateness of a descriptor (higher = later).

    Previously a list-index total order with ``"unknown"`` at index 0 (the bug,
    review #274).  Now UNKNOWN is +infinity — it always sorts latest.
    """
    return _lateness_of(descriptor)


def propagate_available_at(descriptors) -> str | None:
    """Return the latest availability descriptor across inputs (audit §51).

    WS-C fail-closed semantics (review #275): if ANY input's availability is
    unknown (``None`` or the literal ``"unknown"``), the result is ``"unknown"``
    — the unknown input is NOT dropped in favour of the latest known one.
    Otherwise returns the label of the latest known descriptor.

    Accepts either legacy string labels or :class:`AvailabilityExpr` objects.
    """
    exprs = [availability_expr_of(item) for item in descriptors] if descriptors else []
    if not exprs:
        return None
    # R11 P0-06: delegate to latest_availability so the STRING propagation and
    # the typed expression share one authority — a :class:`MaxAvailability`
    # whose label is the latest KNOWN descriptor (unknown stays fail-closed) and
    # whose resolve() picks the max of the CONCRETE timestamps at decision time.
    latest = latest_availability(exprs)
    return latest.label


def latest_availability_expr(descriptors) -> AvailabilityExpr:
    """Typed variant of :func:`propagate_available_at` returning the expression."""
    exprs = [availability_expr_of(item) for item in descriptors] if descriptors else []
    return latest_availability(exprs)


def _knowledge_model_of(temporal_model: str | None) -> str | None:
    if not temporal_model or temporal_model in {"exact", "unknown"}:
        return None
    return str(temporal_model)


# Round-7 WS-C (#277): the substring ``_EOD_AVAILABILITY_MARKERS`` list
# (which matched ``"pe"``/``"pb"``/``"close"`` inside ANY field name) is gone.
# Availability is decided from the FieldSpec's explicit ``available_at`` /
# ``knowledge_time_column`` / ``role`` first, then an EXACT-name EOD default
# (no substring matching).  An unrecognised name stays unknown (fail-closed).
_EOD_AVAILABILITY_EXACT_NAMES = frozenset({
    "close", "high", "low", "vwap", "volume", "amount", "turnover",
    "turnover_ratio", "limit_up", "limit_down", "adjfactor", "factor",
    "market_cap", "pe", "pb", "pct_chg", "pct_change", "ret", "return",
})
_OPEN_AVAILABILITY_EXACT_NAMES = frozenset({"open", "pre_open", "local_open"})


def _available_at_of(spec) -> str | None:
    """Resolve a field's availability descriptor (label string).

    Resolution order (review #277):
    1. explicit ``FieldSpec.available_at`` or ``knowledge_time_column``;
    2. ``FieldSpec.role`` — group keys / masks are day-level, not EOD;
    3. exact-name EOD / open defaults (``"close"`` -> ``session_close``);
    4. ``None`` — availability unknown (fail-closed).
    """
    explicit = getattr(spec, "available_at", None) or getattr(spec, "knowledge_time_column", None)
    if explicit:
        return str(explicit)
    role = str(getattr(spec, "role", "") or "").lower()
    if role in {"group_key", "identifier", "time", "instrument"}:
        return None
    name = str(getattr(spec, "name", "") or "").lower()
    if name in _OPEN_AVAILABILITY_EXACT_NAMES:
        return "session_open"
    if name in _EOD_AVAILABILITY_EXACT_NAMES:
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
    "_available_at_of",
    "latest_availability_expr",
    "propagate_available_at",
]
