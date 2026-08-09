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

import datetime
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


#: Scalar-lattice MIXED marker (review R9-P0-003).  When an operator's inputs
#: disagree on a semantic dimension (>=2 distinct values), the scalar slot holds
#: this marker — NEVER the first input's value — so the merged semantic is
#: operand-order independent: ``semantic(A,B) == semantic(B,A)`` on every
#: dimension.  The ``mixed_<dim>`` diagnostic tuple keeps the actual distinct
#: values for auditing.
MIXED = "MIXED"


def lattice_join_semantic_attrs(
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge child ``semantic_attrs`` via the semantic lattice.

    Replaces the round-6 first-input inheritance loop.  For each scalar
    dimension that is unambiguous across the children the scalar is propagated;
    when the children disagree (>=2 distinct values) the scalar slot holds the
    :data:`MIXED` marker — never ``distinct[0]`` (review R9-P0-003) — and a
    ``mixed_<dim>`` tuple records the actual distinct values so the mix is
    visible instead of silently inherited.
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
            result[dim] = MIXED
            result[f"mixed_{dim}"] = tuple(distinct)
    # semantic-kind lattice join: unambiguous -> propagate; ambiguous -> MIXED.
    kinds = [
        child["semantic_kind"]
        for child in children
        if child.get("semantic_kind") is not None
    ]
    distinct_kinds = list(dict.fromkeys(kinds))
    if len(distinct_kinds) == 1:
        result["semantic_kind"] = distinct_kinds[0]
    elif len(distinct_kinds) > 1:
        result["semantic_kind"] = MIXED
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
# layer via ``resolve(row, calendar, timezone, decision_context)`` (review
# R9-P0-008).  At compile time the schema records the expression and propagates
# the LATEST one bottom-up.  UNKNOWN sorts fail-closed (treat as +infinity /
# propagate): any unknown input makes the root unknown rather than being
# silently dropped.
# ---------------------------------------------------------------------------
def _coerce_timestamp(value: Any) -> datetime.datetime | None:
    """Best-effort coercion of a row/calendar/context value to a datetime.

    Returns ``None`` when the value cannot be interpreted as a timestamp —
    callers then raise a clear resolution error instead of guessing.
    """
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day)
    if isinstance(value, datetime.time):
        return None
    try:
        import pandas as pd  # type: ignore[import-not-found]

        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        ts = pd.Timestamp(value)
        return ts.to_pydatetime()
    except Exception:
        pass
    try:
        text = str(value).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.datetime.strptime(text, fmt)
            except ValueError:
                continue
    except Exception:
        pass
    return None


def _extract_row_value(row: Any, column: str) -> Any:
    """Extract ``column`` from a row (mapping or attribute object).

    Lookup is case-insensitive for mapping keys (``"PubDate"`` matches a
    ``"pub_date"`` key).  Raises a clear error when the row cannot supply it.
    """
    if row is None:
        raise ValueError(f"resolve() needs a row carrying column {column!r}")
    if isinstance(row, dict):
        if column in row:
            return row[column]
        for key in row:
            if str(key).lower() == column.lower():
                return row[key]
        raise KeyError(
            f"row has no column {column!r} (available: {sorted(map(str, row))})"
        )
    value = getattr(row, column, None)
    if value is None:
        try:
            value = row[column]
        except Exception:
            value = None
    if value is None:
        raise ValueError(f"row has no column {column!r}")
    return value


def _resolve_column(row: Any, primary: str, aliases: tuple[str, ...] = ()) -> Any:
    """Extract a column value trying ``primary`` then ``aliases`` (or ``None``)."""
    if row is None:
        return None
    for name in (primary, *aliases):
        try:
            return _extract_row_value(row, name)
        except Exception:
            continue
    return None


def _localize(dt: datetime.datetime, timezone: Any) -> datetime.datetime:
    """Attach a timezone to a naive best-effort timestamp (defensive)."""
    if timezone is None:
        return dt
    try:
        if isinstance(timezone, str):
            from zoneinfo import ZoneInfo

            return dt.replace(tzinfo=ZoneInfo(timezone))
        return dt.replace(tzinfo=timezone)
    except Exception:
        return dt


def _session_date(
    row: Any,
    calendar: Any,
    decision_context: Any,
) -> datetime.datetime | None:
    """Best-effort trading date for wall-clock expressions.

    Resolution order: ``decision_context`` (asof / trading date), then a date
    column on ``row``, then a ``calendar`` object's date.  ``None`` means no
    date is available anywhere.
    """
    if decision_context is not None:
        for attr in ("date", "trading_date", "asof", "asof_date", "trade_date"):
            d = _coerce_timestamp(getattr(decision_context, attr, None))
            if d is not None:
                return d
    if row is not None:
        for name in (
            "date",
            "trade_date",
            "trading_date",
            "timestamp",
            "asof",
            "day",
            "datetime",
        ):
            try:
                d = _coerce_timestamp(_extract_row_value(row, name))
            except Exception:
                continue
            if d is not None:
                return d
    if calendar is not None and not isinstance(calendar, str):
        for attr in ("date", "trading_date", "asof", "asof_date"):
            d = _coerce_timestamp(getattr(calendar, attr, None))
            if d is not None:
                return d
    return None


def _calendar_time(calendar: Any, kind: str) -> tuple[int, int] | None:
    """Best-effort wall-clock ``(hour, minute)`` from a calendar object."""
    if calendar is None or isinstance(calendar, str):
        return None
    for attr in (f"session_{kind}", f"{kind}_time", f"market_{kind}", kind):
        value = getattr(calendar, attr, None)
        if value is None:
            continue
        if isinstance(value, (datetime.datetime, datetime.time)):
            return (value.hour, value.minute)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            try:
                return (int(value[0]), int(value[1]))
            except (TypeError, ValueError):
                continue
    return None


def _wallclock_date(
    row: Any,
    calendar: Any,
    timezone: Any,
    decision_context: Any,
    default_time: tuple[int, int],
    what: str,
) -> datetime.datetime:
    """Combine a trading date with a best-effort wall-clock time."""
    date = _session_date(row, calendar, decision_context)
    if date is None:
        raise NotImplementedError(
            f"{what} needs a trading date (row.date / decision_context.date / "
            "calendar.date) to build the wall-clock coordinate"
        )
    return _localize(
        datetime.datetime(date.year, date.month, date.day, *default_time),
        timezone,
    )


def _resolve_timestamp_column(
    expr: "AvailabilityExpr",
    value: Any,
    row: Any,
    what: str,
    timezone: Any,
) -> datetime.datetime:
    """Resolve a column-based expression to a timestamp.

    ``value`` is the already-extracted column value (``None`` when the row did
    not carry it).  Raises a clear error rather than guessing when the column is
    missing or unparseable.
    """
    if value is None:
        if row is None:
            raise NotImplementedError(
                f"{what} needs a row carrying the column; got no row"
            )
        raise NotImplementedError(
            f"{what} needs the column on the row; row has no such column"
        )
    ts = _coerce_timestamp(value)
    if ts is None:
        raise ValueError(
            f"{what} could not interpret column value {value!r} as a timestamp"
        )
    return _localize(ts, timezone)


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

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        """Resolve to a concrete timestamp at the data layer.

        ``row`` carries the observable columns (a mapping / pandas Series /
        record object); ``calendar`` is a trading-calendar provider (may be a
        bare string label when only a *hint* is known); ``timezone`` is a
        ``zoneinfo`` key / ``tzinfo``; ``decision_context`` carries the decision
        / asof instant.  Subclasses return a timestamp-like value
        (``datetime`` / ``pd.Timestamp``) or raise a clear error when the
        provider they need is unavailable — they never fall back to a
        compile-time total-order number (review R9-P0-008).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.resolve() has no concrete timestamp without "
            "the provider it needs (row / calendar / timezone / decision_context)"
        )

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

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        raise ValueError("UnknownAvailability has no concrete timestamp; fail-closed")


class Midnight(AvailabilityExpr):
    """Value usable from local midnight of its trading date."""

    __slots__ = ()
    lateness = 0.0
    label = "midnight"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        return _wallclock_date(
            row, calendar, timezone, decision_context, (0, 0), "Midnight.resolve()"
        )


@dataclass(frozen=True)
class SessionOpen(AvailabilityExpr):
    """Value usable only after the opening auction of ``calendar``."""

    calendar: str = "TradeDate"
    lateness: float = 10.0

    @property
    def label(self) -> str:
        return "session_open"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        open_time = _calendar_time(calendar, "open") or (9, 30)
        return _wallclock_date(
            row, calendar, timezone, decision_context, open_time,
            "SessionOpen.resolve()",
        )


@dataclass(frozen=True)
class PreClose(AvailabilityExpr):
    calendar: str = "TradeDate"
    lateness: float = 20.0

    @property
    def label(self) -> str:
        return "pre_close"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        preclose_time = _calendar_time(calendar, "pre_close") or (14, 57)
        return _wallclock_date(
            row, calendar, timezone, decision_context, preclose_time,
            "PreClose.resolve()",
        )


@dataclass(frozen=True)
class LocalClose(AvailabilityExpr):
    calendar: str = "TradeDate"
    lateness: float = 30.0

    @property
    def label(self) -> str:
        return "local_close"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        close_time = _calendar_time(calendar, "close") or (15, 0)
        return _wallclock_date(
            row, calendar, timezone, decision_context, close_time,
            "LocalClose.resolve()",
        )


@dataclass(frozen=True)
class SessionClose(AvailabilityExpr):
    """EOD value usable only after the session close of ``calendar``."""

    calendar: str = "TradeDate"
    lateness: float = 40.0

    @property
    def label(self) -> str:
        return "session_close"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        close_time = _calendar_time(calendar, "close") or (15, 0)
        return _wallclock_date(
            row, calendar, timezone, decision_context, close_time,
            "SessionClose.resolve()",
        )


@dataclass(frozen=True)
class AfterClose(AvailabilityExpr):
    calendar: str = "TradeDate"
    lateness: float = 50.0

    @property
    def label(self) -> str:
        return "after_close"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        close_time = _calendar_time(calendar, "close") or (15, 0)
        base = _wallclock_date(
            row, calendar, timezone, decision_context, close_time,
            "AfterClose.resolve()",
        )
        return base + datetime.timedelta(hours=1)


@dataclass(frozen=True)
class NextTradingOpen(AvailabilityExpr):
    """Value usable only at the next trading session's open (after ``reference``)."""

    reference: str = "PubDate"
    lateness: float = 60.0

    @property
    def label(self) -> str:
        return "next_session_open"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        ref = _resolve_column(row, self.reference, ("pub_date", "publication_date"))
        if ref is None and decision_context is not None:
            ref = getattr(decision_context, "asof", None) or getattr(
                decision_context, "date", None
            )
        ref_ts = _coerce_timestamp(ref)
        if ref_ts is None:
            raise NotImplementedError(
                "NextTradingOpen.resolve() needs the reference date "
                f"(row[{self.reference!r}]) and a trading calendar to compute "
                "the next session open"
            )
        if calendar is None or isinstance(calendar, str):
            raise NotImplementedError(
                "NextTradingOpen.resolve() needs a real trading calendar object "
                f"with next_open(); got {calendar!r}"
            )
        for attr in ("next_open", "next_trading_open", "next_session_open", "session_open_after"):
            fn = getattr(calendar, attr, None)
            if not callable(fn):
                continue
            try:
                result = fn(ref_ts)
            except TypeError:
                result = fn(ref_ts.date())
            ts = _coerce_timestamp(result)
            if ts is not None:
                return _localize(ts, timezone)
        raise NotImplementedError(
            f"calendar {type(calendar).__name__} provides no next_open() for "
            "NextTradingOpen.resolve()"
        )


@dataclass(frozen=True)
class NextTradingDay(AvailabilityExpr):
    reference: str = "PubDate"
    lateness: float = 70.0

    @property
    def label(self) -> str:
        return "next_trading_day"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        ref = _resolve_column(row, self.reference, ("pub_date", "publication_date"))
        if ref is None and decision_context is not None:
            ref = getattr(decision_context, "asof", None) or getattr(
                decision_context, "date", None
            )
        ref_ts = _coerce_timestamp(ref)
        if ref_ts is None:
            raise NotImplementedError(
                "NextTradingDay.resolve() needs the reference date "
                f"(row[{self.reference!r}]) and a trading calendar to compute "
                "the next trading day"
            )
        if calendar is None or isinstance(calendar, str):
            raise NotImplementedError(
                "NextTradingDay.resolve() needs a real trading calendar object "
                f"with next_day(); got {calendar!r}"
            )
        for attr in ("next_day", "next_trading_day", "next_session", "next_session_day"):
            fn = getattr(calendar, attr, None)
            if not callable(fn):
                continue
            try:
                result = fn(ref_ts)
            except TypeError:
                result = fn(ref_ts.date())
            ts = _coerce_timestamp(result)
            if ts is not None:
                return _localize(ts, timezone)
        raise NotImplementedError(
            f"calendar {type(calendar).__name__} provides no next_day() for "
            "NextTradingDay.resolve()"
        )


@dataclass(frozen=True)
class FilingDate(AvailabilityExpr):
    """Value usable only from the regulatory filing date of ``reference``."""

    reference: str = "ReportPeriodEndDate"
    lateness: float = 80.0

    @property
    def label(self) -> str:
        return "filing_date"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        value = _resolve_column(
            row,
            "filing_date",
            ("announcement_date", "filingdate", "report_date", self.reference),
        )
        return _resolve_timestamp_column(
            self, value, row, "FilingDate.resolve()", timezone
        )


@dataclass(frozen=True)
class PubDate(AvailabilityExpr):
    """Value usable only from the publication date of ``reference``."""

    reference: str = "PubDate"
    lateness: float = 90.0

    @property
    def label(self) -> str:
        return "PubDate"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        value = _resolve_column(
            row,
            self.reference,
            ("pub_date", "publication_date", "publish_date"),
        )
        return _resolve_timestamp_column(
            self, value, row, "PubDate.resolve()", timezone
        )


@dataclass(frozen=True)
class DeclarationDate(AvailabilityExpr):
    reference: str = "DeclarationDate"
    lateness: float = 100.0

    @property
    def label(self) -> str:
        return "declaration_date"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        value = _resolve_column(
            row,
            self.reference,
            ("declaration_date", "declarationdate", "decl_date"),
        )
        return _resolve_timestamp_column(
            self, value, row, "DeclarationDate.resolve()", timezone
        )


@dataclass(frozen=True)
class ExDate(AvailabilityExpr):
    """Ex-dividend date — DISTINCT from :class:`DeclarationDate` (R9-P0-009)."""

    reference: str = "ExDate"
    lateness: float = 100.0

    @property
    def label(self) -> str:
        return "ex_date"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        value = _resolve_column(
            row,
            self.reference,
            ("ex_date", "exdate", "ex_dividend_date"),
        )
        return _resolve_timestamp_column(self, value, row, "ExDate.resolve()", timezone)


@dataclass(frozen=True)
class RecordDate(AvailabilityExpr):
    """Record date for a corporate action (distinct from declaration/ex dates)."""

    reference: str = "RecordDate"
    lateness: float = 100.0

    @property
    def label(self) -> str:
        return "record_date"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        value = _resolve_column(row, self.reference, ("record_date", "recorddate"))
        return _resolve_timestamp_column(
            self, value, row, "RecordDate.resolve()", timezone
        )


@dataclass(frozen=True)
class PaymentDate(AvailabilityExpr):
    """Payment date for a corporate action (distinct from declaration/ex dates)."""

    reference: str = "PaymentDate"
    lateness: float = 100.0

    @property
    def label(self) -> str:
        return "payment_date"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        value = _resolve_column(
            row,
            self.reference,
            ("payment_date", "paymentdate", "pay_date"),
        )
        return _resolve_timestamp_column(
            self, value, row, "PaymentDate.resolve()", timezone
        )


@dataclass(frozen=True)
class EffectiveDate(AvailabilityExpr):
    """Effective date (value usable from the effective/生效 instant)."""

    reference: str = "EffectiveDate"
    lateness: float = 100.0

    @property
    def label(self) -> str:
        return "effective_date"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        value = _resolve_column(
            row,
            self.reference,
            ("effective_date", "effectivedate", "effective_time"),
        )
        return _resolve_timestamp_column(
            self, value, row, "EffectiveDate.resolve()", timezone
        )


@dataclass(frozen=True)
class TimestampColumn(AvailabilityExpr):
    """Value usable only from the timestamp stored in ``column``."""

    column: str
    lateness: float = 100.0

    @property
    def label(self) -> str:
        return str(self.column)

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        value = _resolve_column(row, self.column)
        return _resolve_timestamp_column(
            self, value, row, f"TimestampColumn({self.column!r}).resolve()", timezone
        )


@dataclass(frozen=True)
class KnowledgeTime(AvailabilityExpr):
    """Generic knowledge-time coordinate (latest descriptor)."""

    lateness: float = 1000.0

    @property
    def label(self) -> str:
        return "knowledge_time"

    def resolve(
        self,
        row: Any = None,
        calendar: Any = None,
        timezone: Any = None,
        decision_context: Any = None,
    ) -> Any:
        raise NotImplementedError(
            "KnowledgeTime.resolve() has no single concrete coordinate (generic "
            "knowledge-time descriptor); resolve the underlying column instead"
        )


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
        # R9-P0-009: ex-date is a DISTINCT corporate-action date from the
        # declaration date — never collapse them in the semantic mapping.
        ("ex_date", ExDate()),
        ("ex_dividend_date", ExDate()),
        ("record_date", RecordDate()),
        ("payment_date", PaymentDate()),
        ("effective_date", EffectiveDate()),
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
    """Declared allowed semantic-kind set for one input parameter.

    ``allowed_semantic_kinds=None`` means the position is *unconstrained* (any
    numeric/untyped series passes) — used to keep a positional contract aligned
    with every panel input while only constraining the typed slots.
    """

    parameter: str | None
    allowed_semantic_kinds: frozenset[str] | None = None

    def accepts(self, semantic_kind: str | None) -> bool:
        if self.allowed_semantic_kinds is None:
            return True
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
        ArgumentTypeContract("x"),
        ArgumentTypeContract("group", frozenset({"GroupKey"})),
    ),
    "group_rank": (
        ArgumentTypeContract("x"),
        ArgumentTypeContract("group", frozenset({"GroupKey"})),
    ),
}


__all__ = [
    "AfterClose",
    "ArgumentTypeContract",
    "AvailabilityExpr",
    "DeclarationDate",
    "EffectiveDate",
    "ExDate",
    "FilingDate",
    "KnowledgeTime",
    "LocalClose",
    "MIXED",
    "Midnight",
    "NextTradingDay",
    "NextTradingOpen",
    "OPERATOR_INPUT_TYPE_CONTRACTS",
    "PaymentDate",
    "PreClose",
    "PubDate",
    "RecordDate",
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
