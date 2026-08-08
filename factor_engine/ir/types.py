"""IR static types, semantic lattice, AvailabilityExpr, and SourceVintageSpec.

Round-7 WS-C (review #265-#278):

* ``SemanticType`` — the typed-IR semantic-kind vocabulary (a **lattice**
  dimension).  Each field / operator input carries one; the compiler validates
  operator arguments against per-parameter declared type sets instead of
  inheriting the first input's semantics.
* ``SemanticLattice`` — the five-dimension semantic lattice (``domains``,
  ``frequencies``, ``source_vintages``, ``universe_ids``, ``semantic_kinds``).
  Operator nodes join their children's lattices bottom-up rather than copying
  ``input[0]``.
* ``AvailabilityExpr`` — replaces the string ``_AVAILABILITY_RANK`` total order.
  Each expression resolves to a concrete knowledge-time coordinate at the data
  layer (``SessionClose(TradeDate)``, ``NextTradingOpen(PubDate)``,
  ``TimestampColumn("declaration_date")``, ...).  UNKNOWN availability sorts
  fail-closed (+infinity): any unknown input makes the root unknown.
* ``SourceVintageSpec`` — structured ``(knowledge_time, revision_time,
  version_id, snapshot_id)`` replacement for the loose ``str | None`` vintage.
* ``OPERATOR_INPUT_TYPE_CONTRACTS`` — the typed-input gate: per-parameter
  ``input_types`` declarations validated by ``ir.analyzer.validate_input_type_contracts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ValueType(str, Enum):
    """编译期/运行期对因子结果形态的粗分类（目前 schema 层使用较少）。"""

    SCALAR = "scalar"   # 单值
    SERIES = "series"   # 一维序列（通常指单标的时序）
    PANEL = "panel"     # 二维面板 date × instrument


_DTYPE_ALIASES = {
    "bool": "bool",
    "boolean": "bool",
    "date": "date",
    "date32": "date",
    "datetime": "datetime64[ns]",
    "timestamp": "datetime64[ns]",
    "float": "float64",
    "double": "float64",
    "float64": "float64",
    "int": "int64",
    "integer": "int64",
    "int64": "int64",
    "uint64": "uint64",
    "str": "string",
    "string": "string",
    "object": "string",
}


def normalize_dtype(dtype: str | None) -> str:
    """Normalize source/catalog dtype spellings without requiring NumPy."""

    raw = str(dtype or "float64").strip().lower()
    if raw.startswith("timestamp") or raw.startswith("datetime"):
        return "datetime64[ns]"
    return _DTYPE_ALIASES.get(raw, raw)


def infer_field_type(value: Any, *, table: str | None = None):
    """Infer a registered ``FieldSpec`` for a name/ColumnRef/SourceRef.

    ``None`` means the legacy column has no semantic registration; callers
    should retain their historical float-panel fallback in that case.
    """

    from fields import resolve_field

    return resolve_field(value, table=table, strict=False)


class SemanticType(str, Enum):
    """Typed-IR semantic kind vocabulary (review P0-30 + WS-C).

    The analyzer propagates ``price_basis`` / ``flow_semantics`` /
    ``frequency`` / ``domain`` onto ``IRNode.semantic_attrs`` and maps them to
    one of these kinds so compile-time contract checks stop guessing from field
    names.  This is additive: the coarse :class:`ValueType` (scalar/series/
    panel) and the dtype aliases above are untouched.
    """

    PRICE_RAW = "PriceRaw"
    PRICE_CONTINUOUS = "PriceContinuous"
    OFFICIAL_LIMIT_PRICE = "OfficialLimitPrice"
    RETURN_DECIMAL = "ReturnDecimal"
    POSITIVE_LEVEL = "PositiveLevel"
    NON_NEGATIVE_ACTIVITY = "NonNegativeActivity"
    NON_NEGATIVE_WEIGHT = "NonNegativeWeight"
    EVENT_BOOL = "EventBool"
    MASK_BOOL = "MaskBool"
    STATE_SIGNED = "StateSigned"
    GROUP_KEY = "GroupKey"
    MINUTE_SERIES = "MinuteSeries"
    DAILY_SERIES = "DailySeries"
    FINANCIAL_STOCK = "FinancialStock"
    FINANCIAL_SINGLE_PERIOD_FLOW = "FinancialSinglePeriodFlow"
    FINANCIAL_CUMULATIVE_YTD_FLOW = "FinancialCumulativeYTDFlow"
    FINANCIAL_TTM_FLOW = "FinancialTTMFlow"
    RELATION_GRAPH = "RelationGraph"


# SEMANTIC_TYPE registry: string keys from the existing ``Schema``/field
# vocabulary -> semantic kind.  ``semantic_type_of`` below is the primary entry
# point; the registry is kept for introspection / reverse lookups.
SEMANTIC_TYPE: dict[str, SemanticType] = {
    "price_basis.raw": SemanticType.PRICE_RAW,
    "price_basis.continuous": SemanticType.PRICE_CONTINUOUS,
    "price_basis.raw_official_limit": SemanticType.OFFICIAL_LIMIT_PRICE,
    "price_basis.return": SemanticType.RETURN_DECIMAL,
    "flow_semantics.stock": SemanticType.FINANCIAL_STOCK,
    "flow_semantics.single_period_flow": SemanticType.FINANCIAL_SINGLE_PERIOD_FLOW,
    "flow_semantics.cumulative_ytd_flow": SemanticType.FINANCIAL_CUMULATIVE_YTD_FLOW,
    "flow_semantics.ttm_flow": SemanticType.FINANCIAL_TTM_FLOW,
    "frequency.minute": SemanticType.MINUTE_SERIES,
    "frequency.daily": SemanticType.DAILY_SERIES,
    "role.group_key": SemanticType.GROUP_KEY,
    "domain.relation": SemanticType.RELATION_GRAPH,
    "role.mask": SemanticType.MASK_BOOL,
}

_PRICE_BASIS_TO_SEMANTIC: dict[str, SemanticType] = {
    "RAW": SemanticType.PRICE_RAW,
    "CONTINUOUS": SemanticType.PRICE_CONTINUOUS,
    "RAW_OFFICIAL_LIMIT": SemanticType.OFFICIAL_LIMIT_PRICE,
    "RETURN": SemanticType.RETURN_DECIMAL,
}

_FLOW_SEMANTICS_TO_SEMANTIC: dict[str, SemanticType] = {
    "stock": SemanticType.FINANCIAL_STOCK,
    "single_period_flow": SemanticType.FINANCIAL_SINGLE_PERIOD_FLOW,
    "cumulative_ytd_flow": SemanticType.FINANCIAL_CUMULATIVE_YTD_FLOW,
    "ttm_flow": SemanticType.FINANCIAL_TTM_FLOW,
}


def semantic_type_of(
    *,
    price_basis: str | None = None,
    flow_semantics: str | None = None,
    frequency: str | None = None,
    domain: str | None = None,
    role: str | None = None,
) -> SemanticType | None:
    """Map propagated semantic attributes to a :class:`SemanticType`.

    Precedence: explicit flow semantics (financial), then price basis, then the
    coarse shape descriptors.  ``None`` means the attributes do not pin down a
    semantic kind (generic derived numeric series).
    """
    if flow_semantics:
        kind = _FLOW_SEMANTICS_TO_SEMANTIC.get(str(flow_semantics).lower())
        if kind is not None:
            return kind
    if price_basis:
        kind = _PRICE_BASIS_TO_SEMANTIC.get(str(price_basis).upper())
        if kind is not None:
            return kind
    if role == "group_key":
        return SemanticType.GROUP_KEY
    if role in {"mask", "universe_mask"}:
        return SemanticType.MASK_BOOL
    if domain == "relation":
        return SemanticType.RELATION_GRAPH
    if str(frequency or "").lower() == "minute":
        return SemanticType.MINUTE_SERIES
    if str(frequency or "").lower() == "daily":
        return SemanticType.DAILY_SERIES
    return None


# ---------------------------------------------------------------------------
# Semantic lattice (#265-#268).  Replaces first-input semantic inheritance.
#
# The lattice is a five-dimensional product where each dimension is a set
# (join = set union, meet = set intersection).  An operator node's lattice is
# the join of its children's lattices; a dimension with exactly one distinct
# value stays a single scalar, a dimension with several distinct values is
# "mixed" and recorded as such (no blind copy of input[0]).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SemanticLattice:
    """Five-dimension semantic lattice carried by :class:`Schema` / IR nodes.

    Each dimension holds the *set* of distinct values observed on the subtree.
    A single-element tuple means the dimension is pinned; a multi-element tuple
    means the subtree is mixed on that dimension.
    """

    domains: tuple[str, ...] = ()
    frequencies: tuple[str, ...] = ()
    source_vintages: tuple[Any, ...] = ()
    universe_ids: tuple[str, ...] = ()
    semantic_kinds: tuple[str, ...] = ()

    def join(self, other: "SemanticLattice | None") -> "SemanticLattice":
        """Lattice join: union of each dimension (order-preserving, de-duped)."""
        if other is None:
            return self
        return SemanticLattice(
            domains=_dedup_tuple(self.domains, other.domains),
            frequencies=_dedup_tuple(self.frequencies, other.frequencies),
            source_vintages=_dedup_tuple(self.source_vintages, other.source_vintages),
            universe_ids=_dedup_tuple(self.universe_ids, other.universe_ids),
            semantic_kinds=_dedup_tuple(self.semantic_kinds, other.semantic_kinds),
        )

    @classmethod
    def from_field_attrs(
        cls,
        *,
        domain: str | None = None,
        frequency: str | None = None,
        source_vintage: Any = None,
        universe_id: str | None = None,
        semantic_kind: str | None = None,
    ) -> "SemanticLattice":
        return cls(
            domains=(domain,) if domain else (),
            frequencies=(frequency,) if frequency else (),
            source_vintages=(source_vintage,) if source_vintage is not None else (),
            universe_ids=(universe_id,) if universe_id else (),
            semantic_kinds=(semantic_kind,) if semantic_kind else (),
        )

    def scalar_domain(self) -> str | None:
        """The pinned domain if unambiguous, else None (mixed)."""
        return self.domains[0] if len(self.domains) == 1 else None

    def scalar_frequency(self) -> str | None:
        return self.frequencies[0] if len(self.frequencies) == 1 else None

    def scalar_semantic_kind(self) -> str | None:
        return self.semantic_kinds[0] if len(self.semantic_kinds) == 1 else None


def _dedup_tuple(*tuples: tuple[Any, ...]) -> tuple[Any, ...]:
    out: list[Any] = []
    for tup in tuples:
        for item in tup:
            if item not in out:
                out.append(item)
    return tuple(out)


def lattice_join_semantic_attrs(
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge child ``semantic_attrs`` via the semantic lattice.

    Replaces the round-6 first-input inheritance loop.  For each scalar
    dimension that is unambiguous across the children the scalar is propagated;
    when the children disagree the first child's value is kept for
    backward-compatible scalar consumers *and* a ``mixed_<dim>`` marker is
    recorded so the mix is visible instead of silently inherited.
    """
    result: dict[str, Any] = {}
    scalar_dims = (
        "domain",
        "frequency",
        "cardinality",
        "temporal_model",
        "unit",
        "price_basis",
        "flow_semantics",
    )
    for dim in scalar_dims:
        values: list[Any] = []
        for child in children:
            value = child.get(dim)
            if value is not None:
                values.append(value)
        distinct = list(dict.fromkeys(values))  # order-preserving dedup
        if len(distinct) == 1:
            result[dim] = distinct[0]
        elif len(distinct) > 1:
            result[dim] = distinct[0]
            result[f"mixed_{dim}"] = tuple(distinct)
    # semantic-kind lattice join: unambiguous -> propagate; ambiguous -> record.
    kinds = [
        child["semantic_kind"]
        for child in children
        if child.get("semantic_kind") is not None
    ]
    distinct_kinds = list(dict.fromkeys(kinds))
    if len(distinct_kinds) == 1:
        result["semantic_kind"] = distinct_kinds[0]
    elif len(distinct_kinds) > 1:
        result["semantic_kind"] = distinct_kinds[0]
        result["mixed_semantic_kind"] = tuple(distinct_kinds)
    # pit_safe: AND across children (any non-pit-safe input makes the root unsafe).
    if children:
        result["pit_safe"] = all(
            child.get("pit_safe", True) for child in children
        )
    return result


# ---------------------------------------------------------------------------
# AvailabilityExpr (#274-#277).  Replaces the string ``_AVAILABILITY_RANK``.
#
# An expression is a *knowledge-time descriptor*: the coordinate at which the
# value becomes decision-usable.  Concrete timestamps are resolved at the data
# layer via ``resolve()`` (a calendar/clock provider is injected there).  At
# compile time the schema records the expression and propagates the LATEST one
# bottom-up.  UNKNOWN sorts fail-closed (treat as +infinity / propagate): any
# unknown input makes the root unknown rather than being silently dropped.
# ---------------------------------------------------------------------------
class AvailabilityExpr:
    """Base class for a knowledge-time availability expression."""

    __slots__ = ()

    #: ordering key — larger means *later* (less usable at an earlier decision).
    #: UNKNOWN uses +infinity so it always wins the "latest" propagation.
    lateness: float

    @property
    def label(self) -> str:
        """Legacy string descriptor (backward-compatible schema surface)."""
        raise NotImplementedError

    def resolve(self, provider: Any = None) -> Any:
        """Resolve to a concrete timestamp at the data layer.

        ``provider`` is a calendar / knowledge-time clock; subclasses that only
        carry a column reference (``TimestampColumn``) resolve against it.
        """
        raise NotImplementedError

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label

    def __repr__(self) -> str:
        return f"<{self.label}>"

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.__dict__ == getattr(other, "__dict__", {})

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(vars(self).items()))))


class UnknownAvailability(AvailabilityExpr):
    """The availability is unknown.  Sorts fail-closed (lateness +inf)."""

    __slots__ = ()
    lateness = float("inf")

    @property
    def label(self) -> str:
        return "unknown"

    def resolve(self, provider: Any = None) -> Any:
        raise ValueError("UnknownAvailability has no concrete timestamp; fail-closed")


class Midnight(AvailabilityExpr):
    """Value usable from local midnight of its trading date."""

    __slots__ = ()
    lateness = 0.0
    label = "midnight"


@dataclass(frozen=True)
class SessionOpen(AvailabilityExpr):
    """Value usable only after the opening auction of ``calendar``."""

    calendar: str = "TradeDate"
    lateness: float = 10.0

    @property
    def label(self) -> str:
        return "session_open"


@dataclass(frozen=True)
class PreClose(AvailabilityExpr):
    calendar: str = "TradeDate"
    lateness: float = 20.0

    @property
    def label(self) -> str:
        return "pre_close"


@dataclass(frozen=True)
class LocalClose(AvailabilityExpr):
    calendar: str = "TradeDate"
    lateness: float = 30.0

    @property
    def label(self) -> str:
        return "local_close"


@dataclass(frozen=True)
class SessionClose(AvailabilityExpr):
    """EOD value usable only after the session close of ``calendar``."""

    calendar: str = "TradeDate"
    lateness: float = 40.0

    @property
    def label(self) -> str:
        return "session_close"


@dataclass(frozen=True)
class AfterClose(AvailabilityExpr):
    calendar: str = "TradeDate"
    lateness: float = 50.0

    @property
    def label(self) -> str:
        return "after_close"


@dataclass(frozen=True)
class NextTradingOpen(AvailabilityExpr):
    """Value usable only at the next trading session's open (after ``reference``)."""

    reference: str = "PubDate"
    lateness: float = 60.0

    @property
    def label(self) -> str:
        return "next_session_open"


@dataclass(frozen=True)
class NextTradingDay(AvailabilityExpr):
    reference: str = "PubDate"
    lateness: float = 70.0

    @property
    def label(self) -> str:
        return "next_trading_day"


@dataclass(frozen=True)
class FilingDate(AvailabilityExpr):
    """Value usable only from the regulatory filing date of ``reference``."""

    reference: str = "ReportPeriodEndDate"
    lateness: float = 80.0

    @property
    def label(self) -> str:
        return "filing_date"


@dataclass(frozen=True)
class PubDate(AvailabilityExpr):
    """Value usable only from the publication date of ``reference``."""

    reference: str = "PubDate"
    lateness: float = 90.0

    @property
    def label(self) -> str:
        return "PubDate"


@dataclass(frozen=True)
class DeclarationDate(AvailabilityExpr):
    reference: str = "DeclarationDate"
    lateness: float = 100.0

    @property
    def label(self) -> str:
        return "declaration_date"


@dataclass(frozen=True)
class TimestampColumn(AvailabilityExpr):
    """Value usable only from the timestamp stored in ``column``."""

    column: str
    lateness: float = 100.0

    @property
    def label(self) -> str:
        return str(self.column)


@dataclass(frozen=True)
class KnowledgeTime(AvailabilityExpr):
    """Generic knowledge-time coordinate (latest descriptor)."""

    lateness: float = 1000.0

    @property
    def label(self) -> str:
        return "knowledge_time"


# Sentinel for "availability unknown" — fail-closed +infinity.
UNKNOWN = UnknownAvailability()


def _normalize_label(text: str) -> str:
    """Normalize a descriptor label for lookup (case + separators insensitive).

    ``"PubDate"``, ``"pub_date"``, ``"pub-date"`` all normalize to ``"pubdate"``
    so the legacy schema labels resolve to their typed expression.
    """
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


#: Canonical label -> expression constructor (for the compat string shim).
#: Keys are normalized (see :func:`_normalize_label`).
_LABEL_TO_EXPR: dict[str, AvailabilityExpr] = {
    _normalize_label(label): expr
    for label, expr in (
        ("unknown", UNKNOWN),
        ("midnight", Midnight()),
        ("session_open", SessionOpen()),
        ("local_open", SessionOpen()),
        ("pre_close", PreClose()),
        ("local_close", LocalClose()),
        ("session_close", SessionClose()),
        ("after_close", AfterClose()),
        ("next_session_open", NextTradingOpen()),
        ("next_open", NextTradingOpen()),
        ("next_trading_day", NextTradingDay()),
        ("filing_date", FilingDate()),
        ("filing", FilingDate()),
        ("report_date", FilingDate()),
        ("pub_date", PubDate()),
        ("PubDate", PubDate()),
        ("declaration_date", DeclarationDate()),
        ("declaration", DeclarationDate()),
        ("ex_date", DeclarationDate()),
        ("knowledge_time", KnowledgeTime()),
    )
}


def availability_expr_of(descriptor: Any) -> AvailabilityExpr:
    """Normalize a legacy string label / ``AvailabilityExpr`` to an expr.

    ``None`` and the literal ``"unknown"`` both map to :data:`UNKNOWN` (the
    fail-closed +infinity sentinel).  Unknown string labels map to UNKNOWN so
    an unregistered descriptor cannot accidentally sort *early* (review #274).
    """
    if descriptor is None:
        return UNKNOWN
    if isinstance(descriptor, AvailabilityExpr):
        return descriptor
    low = _normalize_label(descriptor)
    if low in {"", "none", "unknown", "nan"}:
        return UNKNOWN
    return _LABEL_TO_EXPR.get(low, UNKNOWN)


def latest_availability(expressions: Iterable[Any]) -> AvailabilityExpr:
    """Return the LATEST availability expression (fail-closed).

    Unknown (+inf) wins: if any input is unknown the result is unknown
    (review #275 — do not drop the unknown input and keep a known one).
    """
    exprs = [availability_expr_of(item) for item in expressions]
    if not exprs:
        return UNKNOWN
    return max(exprs, key=lambda item: item.lateness)


# ---------------------------------------------------------------------------
# SourceVintageSpec (#278).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceVintageSpec:
    """Structured source-vintage coordinate replacing the loose ``str | None``.

    ``knowledge_time`` is the knowledge-time column (e.g. ``PubDate``);
    ``revision_time`` / ``version_id`` describe the revision chain; ``snapshot_id``
    marks a specific point-in-time snapshot.  All fields optional — an
    unversioned field carries a spec with ``None`` members, never a mismatched
    tuple or bare string.
    """

    knowledge_time: str | None = None
    revision_time: str | None = None
    version_id: str | None = None
    snapshot_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_time": self.knowledge_time,
            "revision_time": self.revision_time,
            "version_id": self.version_id,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_field_spec(cls, spec: Any) -> "SourceVintageSpec":
        """Build a spec from a ``FieldSpec``'s knowledge/revision columns.

        A ``revision_columns`` tuple is interpreted as a revision chain: the
        first column is the revision-time marker, the last is the latest version.
        """
        revision_columns = tuple(getattr(spec, "revision_columns", None) or ())
        knowledge_time = getattr(spec, "knowledge_time_column", None)
        return cls(
            knowledge_time=knowledge_time,
            revision_time=revision_columns[0] if revision_columns else None,
            version_id=revision_columns[-1] if len(revision_columns) > 1 else None,
            snapshot_id=None,
        )


# ---------------------------------------------------------------------------
# Per-argument typed input contracts (#269-#273).
#
# ``OPERATOR_INPUT_TYPE_CONTRACTS[canonical]`` is a tuple of
# ``ArgumentTypeContract`` aligned with the operator's *panel* inputs (scalar
# window / lag parameters are excluded — they are validated by the parameter
# contract layer).  ``ir.analyzer.validate_input_type_contracts`` rejects a
# call whose argument's ``semantic_kind`` is present but not in the declared
# set (e.g. swapping Volume into a close slot, or Return into a volume slot).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArgumentTypeContract:
    """Declared allowed semantic-kind set for one input parameter."""

    parameter: str
    allowed_semantic_kinds: frozenset[str]

    def accepts(self, semantic_kind: str | None) -> bool:
        if semantic_kind is None:
            # No declared semantic kind — cannot prove a mismatch; the gate
            # stays permissive for untyped/research columns.
            return True
        return semantic_kind in self.allowed_semantic_kinds


OPERATOR_INPUT_TYPE_CONTRACTS: dict[str, tuple[ArgumentTypeContract, ...]] = {
    "dollar_volume_zscore": (
        ArgumentTypeContract("close", frozenset({
            "PriceRaw", "PriceContinuous", "PositiveLevel", "OfficialLimitPrice",
        })),
        ArgumentTypeContract("volume", frozenset({"NonNegativeActivity"})),
    ),
    "volume_zscore": (
        ArgumentTypeContract("volume", frozenset({"NonNegativeActivity"})),
    ),
    "abnormal_volume": (
        ArgumentTypeContract("volume", frozenset({"NonNegativeActivity"})),
    ),
    "relative_volume": (
        ArgumentTypeContract("volume", frozenset({"NonNegativeActivity"})),
    ),
    "volume_autocorr": (
        ArgumentTypeContract("volume", frozenset({"NonNegativeActivity"})),
    ),
    "turnover_autocorr": (
        ArgumentTypeContract("turnover", frozenset({"NonNegativeActivity"})),
    ),
    "event_historical_response_mean": (
        ArgumentTypeContract("response", frozenset({
            "ReturnDecimal", "PriceRaw", "PriceContinuous", "PositiveLevel",
            "NonNegativeActivity", "NonNegativeWeight",
        })),
        ArgumentTypeContract("event", frozenset({"EventBool", "MaskBool"})),
    ),
    "group_mean": (
        ArgumentTypeContract("group", frozenset({"GroupKey"})),
    ),
    "group_rank": (
        ArgumentTypeContract("group", frozenset({"GroupKey"})),
    ),
}


__all__ = [
    "AfterClose",
    "ArgumentTypeContract",
    "AvailabilityExpr",
    "DeclarationDate",
    "FilingDate",
    "KnowledgeTime",
    "LocalClose",
    "Midnight",
    "NextTradingDay",
    "NextTradingOpen",
    "OPERATOR_INPUT_TYPE_CONTRACTS",
    "PreClose",
    "PubDate",
    "SEMANTIC_TYPE",
    "SessionClose",
    "SessionOpen",
    "SemanticLattice",
    "SemanticType",
    "SourceVintageSpec",
    "TimestampColumn",
    "UNKNOWN",
    "UnknownAvailability",
    "ValueType",
    "availability_expr_of",
    "infer_field_type",
    "latest_availability",
    "lattice_join_semantic_attrs",
    "normalize_dtype",
    "semantic_type_of",
]
