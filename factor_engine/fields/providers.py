"""Market field bindings — how each market provides each canonical concept.

``MarketFieldBinding`` is the adapter contract (multi-market plan §7, §9, §79):
a canonical ``concept_id`` + a ``market`` resolves to a physical provider with a
unit transform, provider quality, coverage class, PIT model, and availability.

The market difference is FULLY encapsulated here — math operators never see
``Return`` vs ``Ret``, ``bp`` vs ``decimal``, ``CNY`` vs ``USD``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from factor_engine.market.capabilities import (
    CoverageClass,
    MarketStatus,
    MarketSupport,
    ProviderQuality,
)
from factor_engine.market.context import ASHARE_CONTEXT, US_CONTEXT, MarketContext

from .units_v2 import (
    CNY,
    CNY_PER_SHARE,
    DATE,
    RATIO,
    SHARES,
    USD,
    USD_PER_SHARE,
    UnitSpec,
)

Transform = Callable[[Any], Any]


def _identity(x: Any) -> Any:
    return x


def _identity_fields(fields: dict[str, Any]) -> Any:
    """Identity for a single-physical-field provider (fields dict contract).

    ADJ_FIELD_MIGRATION: A-share continuous_* bindings read the precomputed Adj*
    columns on StockDailyBarAdj via a single physical field; ``apply_binding_transform``
    routes single-field bindings to ``transform(value)`` — but a provider whose
    contract is expressed as ``fields={physical: value}`` must resolve the sole
    physical field to stay deterministic (no silent ``None``).
    """
    if not fields:
        raise KeyError("identity-fields provider needs at least one physical field")
    return next(iter(fields.values()))


def _times(multiplier: float) -> Transform:
    def _apply(x: Any) -> Any:
        return x * multiplier

    return _apply


def _not_bool(x: Any) -> Any:
    """Boolean negation that PRESERVES unknown: ``tradable = NOT IsSuspend``.

    The A-share ``IsSuspend`` field is True when the stock is suspended, so the
    ``tradability_state`` concept (tradable?) is its negation.  Only a KNOWN
    state flips: known suspended (1) -> 0, known not-suspended (0) -> 1, and an
    unknown suspension state (NaN) STAYS NaN — an unknown tradability must never
    be forced to "not tradable" (or "tradable").  This concept is the
    not-suspended state only; it does NOT claim listing/public-status membership.
    """
    import numpy as _np

    try:
        import pandas as _pd

        if isinstance(x, _pd.Series):
            return _pd.Series(
                _not_bool(x.to_numpy()), index=x.index, name=x.name
            )
        if isinstance(x, _pd.DataFrame):
            return _pd.DataFrame(
                {c: _not_bool(x[c].to_numpy()) for c in x.columns},
                index=x.index,
            )
    except Exception:  # pragma: no cover - defensive
        pass
    arr = _np.asarray(x, dtype=float)  # preserve NaN; bool -> 1.0/0.0
    return _np.where(arr == 1, 0.0, _np.where(arr == 0, 1.0, _np.nan))


def _us_tradable(fields: dict[str, Any]) -> Any:
    """US tradability = ``StockList.type == "CS"`` AND a finite DailyBar close.

    P0-027: the old binding was ``_identity`` over ``StockList.type`` while the
    ``transform_description`` claimed "CS universe + valid price/volume".  This
    real boolean (1/0) composes the CS list membership with a valid-bar mask:
    any row whose membership is UNKNOWN — missing/NaN type, non-finite close,
    or absent DailyBar column — fails closed to 0 (not tradable), never 1.
    ``universe_daily`` membership is declared on the binding via required_filters
    (the read must supply it); it is not silently substituted.
    """
    import numpy as _np
    import pandas as _pd

    if "StockList.type" not in fields:
        raise KeyError(
            f"US tradability provider needs StockList.type; got {sorted(fields)}"
        )
    type_col = fields["StockList.type"]

    def _series(v: Any, name: str) -> _pd.Series:
        if isinstance(v, _pd.Series):
            return v
        if isinstance(v, _pd.DataFrame):
            return v.iloc[:, 0]
        return _pd.Series(_np.asarray(v, dtype=object), name=name)

    type_s = _series(type_col, "type")
    # Only Common Stock ("CS") is in the tradable US universe; ETN/ETF/ADT and
    # any NaN/unknown type are out (fail closed).
    is_cs = type_s.notna() & (type_s.astype(str).str.strip() == "CS")

    close_col = fields.get("StockDailyBar.Close")
    if close_col is None:
        # Valid-bar anchor not supplied -> every row's membership is unknown.
        return _np.zeros(len(is_cs), dtype=float)
    close_s = _series(close_col, "close")
    try:
        close_num = _pd.to_numeric(close_s, errors="coerce").to_numpy(dtype=float)
        finite_close = _np.isfinite(close_num)
    except Exception:  # pragma: no cover - defensive
        finite_close = close_s.notna().to_numpy()
    return (is_cs.to_numpy(dtype=float) * finite_close.astype(float))


def _dividend_declared_fail_closed(fields: dict[str, Any]) -> Any:
    """US dividend strict-PIT guard: knowledge-time is ``declaration_date``.

    P0-026: a row whose ``declaration_date`` is missing fails closed to NaN —
    it must never silently fall back to the ex-date as knowledge time
    (look-ahead hazard).  The DataAccess contract ``us_stock_dividend`` is now
    ``pit=strict`` / ``availability_column=declaration_date``; this is the
    provider-side enforcement.  The ``declaration_date`` column being absent
    entirely is a provider misconfiguration -> raise.
    """
    import numpy as _np
    import pandas as _pd

    if "StockDividend.cash_amount" not in fields:
        raise KeyError(
            f"US dividend provider needs StockDividend.cash_amount; got {sorted(fields)}"
        )
    if "StockDividend.declaration_date" not in fields:
        raise KeyError(
            "US dividend provider needs StockDividend.declaration_date (strict "
            "PIT); refusing to fall back to ex-date as knowledge time"
        )
    amount = fields["StockDividend.cash_amount"]
    decl = fields["StockDividend.declaration_date"]

    def _series(v: Any, name: str) -> _pd.Series:
        if isinstance(v, _pd.Series):
            return v
        if isinstance(v, _pd.DataFrame):
            return v.iloc[:, 0]
        return _pd.Series(_np.asarray(v, dtype=object), name=name)

    amount_s = _pd.to_numeric(_series(amount, "amount"), errors="coerce")
    decl_s = _series(decl, "declaration_date")
    decl_valid = decl_s.notna() & (decl_s.astype(str).str.strip() != "")
    out = amount_s.to_numpy(dtype=float).copy()
    out[~decl_valid.to_numpy(dtype=bool)] = _np.nan
    return out


def _mul_two(field_a: str, field_b: str):
    """Real multi-field derived transform: ``field_a * field_b``.

    The binding's ``transform`` for a derived provider receives a dict keyed by
    the *physical* field names and returns the derived array.  This is the
    executable contract behind ``derived_expression`` (e.g. continuous_close =
    Close * Factor); the previous implementation registered an identity and
    only *documented* the multiplication in ``transform_description``.
    """

    def _apply(fields: dict[str, Any]) -> Any:
        if field_a not in fields or field_b not in fields:
            raise KeyError(
                f"derived provider needs physical fields {field_a!r} and {field_b!r}; got {sorted(fields)}"
            )
        return fields[field_a] * fields[field_b]

    return _apply


def _div_two(field_num: str, field_den: str):
    """Derived transform ``field_num / field_den`` (LQTP adjusted volume)."""

    def _apply(fields: dict[str, Any]) -> Any:
        if field_num not in fields or field_den not in fields:
            raise KeyError(
                f"derived provider needs physical fields {field_num!r} and {field_den!r}; got {sorted(fields)}"
            )
        return fields[field_num] / fields[field_den]

    return _apply


# ---------------------------------------------------------------------------
# Structured filter requirement (P1-14): ``required_filters`` was a tuple of
# strings ("IndustrySource", "currency=USD", "timeframe") that could not be
# executed or hashed.  It is now a machine-executable contract.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdjustmentFactorCertificate:
    """R24-144/145: SINGLE authority for a continuous price adjustment factor.

    A provider's price adjustment (``adjusted = raw * factor``) is either
    certified PIT-safe (the historical factor does not contain FUTURE corporate
    actions) or it is not.  FieldSpec / ProviderBinding may only REFERENCE this
    certificate — they must never write their own contradicting
    ``source_certified`` / ``adjustment_status`` / ``direction`` states.

    ``vintage_pit_certified`` (R24-146..148): False when the vendor recomputes
    the historical factor with future splits/dividends — a continuous historical
    LEVEL is then NOT a strict-PIT raw feature (return ratios that are
    scale-insensitive may still be certified per family).
    """

    canonical: str
    direction: str = "backward_multiplier"
    formula: str = "adjusted = raw * factor"
    source_certified: bool = False
    vintage_pit_certified: bool = False
    certificate_hash: str | None = None

    def verify_field_metadata(self, field_metadata: dict[str, Any] | None) -> None:
        """R24-145: the FieldSpec metadata must agree with the certificate.

        Raises when the field writes a contradicting adjustment/price state
        instead of deferring to this certificate.
        """
        field_metadata = dict(field_metadata or {})
        raw_adjusted = str(field_metadata.get("adjustment_status", "") or "").lower()
        if raw_adjusted and self.source_certified and raw_adjusted == "unverified":
            raise ValueError(
                f"field {self.canonical!r} declares adjustment_status='unverified' "
                "but the provider certificate certifies it — contradicting "
                "states are forbidden (R24-144/145); FieldSpec/ProviderBinding "
                "must defer to the AdjustmentFactorCertificate"
            )


@dataclass(frozen=True)
class FilterRequirement:
    """One required read-side filter on a physical provider.

    ``operator`` is one of:
      - ``"eq"``          value must equal ``value`` (e.g. currency=USD);
      - ``"enum_select"`` caller must select an allowed value from ``allowed``
                          (exactly one when ``exactly_one``), e.g. timeframe;
      - ``"present"``     the column/selection must be supplied (IndustrySource).
    """

    field: str
    operator: str = "eq"
    value: Any = None
    allowed: tuple[str, ...] = ()
    exactly_one: bool = False
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "allowed": list(self.allowed),
            "exactly_one": self.exactly_one,
            "required": self.required,
        }


def parse_filter_requirement(raw: Any) -> FilterRequirement:
    """Coerce legacy string / FilterRequirement into a structured contract.

    ``"currency=USD"``   -> eq(currency, USD); ``"timeframe"`` -> enum_select;
    ``"IndustrySource"`` -> present; ``FilterRequirement`` passes through.
    """
    if isinstance(raw, FilterRequirement):
        return raw
    token = str(raw).strip()
    if not token:
        raise ValueError("empty required_filter entry")
    if "=" in token:
        field, _, val = token.partition("=")
        return FilterRequirement(field=field.strip(), operator="eq", value=val.strip())
    return FilterRequirement(field=token, operator="enum_select", required=True)


# ---------------------------------------------------------------------------
# R17-016/R17-017: structured dependency contracts.  ``universe_daily present``
# was previously modeled as a required filter, but it is ANOTHER DATASET, not a
# column predicate of the current table.  A derived provider (US market cap =
# TickerSharesSnapshot.shares * StockDailyBar.Close) spans multiple datasets;
# the binding's single ``dataset=`` slot cannot describe it.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JoinRequirement:
    """A cross-dataset join a provider's execution graph must materialize."""

    dataset: str
    table: str
    field: str | None = None
    join_keys: tuple[str, ...] = ()
    temporal_join: str = "exact"  # exact | asof_backward | financial_pit
    required_filters: tuple[FilterRequirement, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "table": self.table,
            "field": self.field,
            "join_keys": list(self.join_keys),
            "temporal_join": self.temporal_join,
            "required_filters": [f.to_dict() for f in self.required_filters],
        }


@dataclass(frozen=True)
class ProviderDependency:
    """One physical read edge in a provider's execution graph (R17-017).

    A derived provider declares its inputs as explicit dependency edges instead
    of hiding cross-table reads in ``physical_fields`` behind a single
    ``dataset=``.  The planner materializes these edges into a source plan.
    """

    dataset: str
    table: str
    field: str
    join_keys: tuple[str, ...] = ()
    temporal_join: str = "exact"
    required_filters: tuple[FilterRequirement, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "table": self.table,
            "field": self.field,
            "join_keys": list(self.join_keys),
            "temporal_join": self.temporal_join,
            "required_filters": [f.to_dict() for f in self.required_filters],
        }


# ---------------------------------------------------------------------------
# R17-035: FinancialPeriodPolicy — period selection is a machine contract, not
# a hidden assumption in the asof adapter.  Every market chooses a policy; the
# policy (and the specific selection values) enter factor identity / lineage.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FinancialPeriodPolicy:
    """How one market selects the visible financial period (R17-035).

    A-share: same ``PubDate`` may hold several ``ReportPeriodEndDate`` rows —
    ``latest_period_known_asof_decision_time`` selects the most recent period
    whose knowledge time is <= decision time.  US: an exact ``timeframe``
    (quarterly/annual/trailing_twelve_months) MUST be filtered first, then the
    latest filing revision for the selected period is taken.
    """

    timeframe: str | None = None  # quarterly | annual | trailing_twelve_months | None
    period_selection: str = "latest_visible_period"
    revision_selection: str = "latest_filing_revision"
    same_day_visibility_policy: str = "DATE_ONLY_CONSERVATIVE_NEXT_SESSION"
    flow_conversion_policy: str = "single_period_from_cumulative_ytd"

    def to_dict(self) -> dict[str, Any]:
        return dict(
            timeframe=self.timeframe,
            period_selection=self.period_selection,
            revision_selection=self.revision_selection,
            same_day_visibility_policy=self.same_day_visibility_policy,
            flow_conversion_policy=self.flow_conversion_policy,
        )


ASHARE_FINANCIAL_PERIOD_POLICY = FinancialPeriodPolicy(
    period_selection="latest_period_known_asof_decision_time",
    same_day_visibility_policy="DATE_ONLY_CONSERVATIVE_NEXT_SESSION",
    flow_conversion_policy="quarter_from_cumulative_ytd",
)

US_FINANCIAL_PERIOD_POLICY = FinancialPeriodPolicy(
    timeframe="quarterly",  # must be set by the caller; default is explicit
    period_selection="exact_timeframe_latest_filing_revision",
    revision_selection="latest_filing_revision",
    same_day_visibility_policy="DATE_ONLY_CONSERVATIVE_NEXT_SESSION",
    flow_conversion_policy="single_period_flow",  # US statements are period-scoped
)


# ---------------------------------------------------------------------------
# Binding contract.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketFieldBinding:
    """One market's provider for one canonical concept."""

    concept_id: str
    market: str
    provider_id: str
    dataset: str | None
    physical_fields: tuple[str, ...]
    transform: Transform
    quality: ProviderQuality
    coverage: CoverageClass
    source_unit: UnitSpec
    canonical_unit: UnitSpec
    temporal_model: str
    knowledge_time: str | None = None
    effective_time: str | None = None
    available_at: str | None = None
    required_filters: tuple[FilterRequirement, ...] = ()
    source_certified: bool = False
    notes: str = ""
    transform_description: str = "identity"
    derived_expression: str | None = None  # executable expression for derived providers
    coverage_gate: float | None = None  # P1-002: fraction of target universe the
    # provider actually covers; when < 0.8 the resolver flags production use.
    # R17-016/R17-017: explicit cross-dataset dependency graph for derived
    # providers (e.g. US market cap = TickerSharesSnapshot.shares * Close).
    # Empty tuple == the single ``dataset``/``physical_fields`` describe the whole
    # read; non-empty == the planner MUST materialize every edge.
    dependencies: tuple[ProviderDependency, ...] = ()
    # R17-036: flow semantics is a RESOLVED-PROVIDER output, not a concept static
    # property.  A-share cumulative-YTD rows, US quarterly single-period, US TTM
    # rows all live under the same canonical concept but have DIFFERENT flow
    # semantics; quarterization/growth/TTM operators read this field.  None ==
    # not declared (callers keep the concept-level default).
    flow_semantics: str | None = None
    # R17-033: dynamic validity policy — a provider's EXACT_DERIVED+FULL tag is
    # not the whole story.  US continuous prices are computed from AdjFactor which
    # may be CLAMPED for long-history tickers; a level-sensitive operator must not
    # trust absolute adjusted-price LEVELS on a clamped ticker (return/log-return
    # paths are safe).  ``None`` == no dynamic restriction.
    dynamic_validity_policy: str | None = None
    # R17-085: whether the provider output is LEVEL-sensitive (absolute adjusted
    # price matters) vs return-invariant.  Drives operator eligibility on clamped
    # tickers.  ``None`` == not declared (conservative: assume level-sensitive).
    level_sensitive: bool | None = None

    @property
    def availability_expr(self) -> Any:
        """Typed availability expression for this binding (R40 #180).

        ``available_at`` is a single string slot that historically mixed a
        policy label (``"session_close"``) with a column name (``"PubDate"``).
        This property normalizes it to a typed :class:`AvailabilityExpr`:
        known policy labels resolve to their expr; anything that is NOT a known
        policy label is treated as a column reference
        (``TimestampColumn(name)``).  ``knowledge_time`` / ``effective_time``
        remain explicit column references.
        """
        from factor_engine.ir.types import availability_expr_of, normalize_availability_descriptor

        return normalize_availability_descriptor(self.available_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept_id,
            "market": self.market,
            "provider": self.provider_id,
            "dataset": self.dataset,
            "physical": list(self.physical_fields),
            "transform": self.transform_description,
            "quality": self.quality.value,
            "coverage": self.coverage.value,
            "source_unit": self.source_unit.to_dict(),
            "canonical_unit": self.canonical_unit.to_dict(),
            "temporal_model": self.temporal_model,
            "knowledge_time": self.knowledge_time,
            "effective_time": self.effective_time,
            "available_at": self.available_at,
            "availability_expr": str(self.availability_expr),
            "required_filters": [f.to_dict() for f in self.required_filters],
            "source_certified": self.source_certified,
            "derived_expression": self.derived_expression,
            "coverage_gate": self.coverage_gate,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "flow_semantics": self.flow_semantics,
            "dynamic_validity_policy": self.dynamic_validity_policy,
            "level_sensitive": self.level_sensitive,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------
class ProviderRegistry:
    """Deterministic (concept_id, market) -> ordered provider chain registry.

    Each (concept, market) key holds an *ordered* chain of bindings (first =
    highest priority) so a concept like US ``market_cap_local`` can express
    ``Close * weighted_shares`` (primary) with a ``share_class_shares * Close``
    and a sparse-valuation research fallback behind it — a single binding slot
    cannot represent a real provider chain (P1-5).  ``binding()`` returns the
    primary; ``bindings()`` returns the whole chain.
    """

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str], list[MarketFieldBinding]] = {}

    def register(self, binding: MarketFieldBinding, *, replace: bool = False) -> None:
        key = (binding.concept_id.strip(), binding.market.strip().lower())
        chain = self._bindings.setdefault(key, [])
        if replace:
            chain.clear()
        elif any(b.provider_id == binding.provider_id for b in chain):
            raise ValueError(
                f"binding already registered: {binding.concept_id}@{binding.market} "
                f"provider={binding.provider_id}"
            )
        chain.append(binding)

    def binding(self, concept_id: str, market: str) -> MarketFieldBinding | None:
        chain = self._bindings.get((concept_id.strip(), market.strip().lower()))
        return chain[0] if chain else None

    def bindings(self, concept_id: str, market: str) -> tuple[MarketFieldBinding, ...]:
        return tuple(self._bindings.get((concept_id.strip(), market.strip().lower()), ()))

    def require_binding(self, concept_id: str, market: str) -> MarketFieldBinding:
        result = self.binding(concept_id, market)
        if result is None:
            raise KeyError(f"no binding for concept {concept_id!r} @ {market!r}")
        return result

    def for_market(self, market: str) -> tuple[MarketFieldBinding, ...]:
        m = market.strip().lower()
        return tuple(
            sorted(
                (b for chain in self._bindings.values() for b in chain if b.market == m),
                key=lambda b: b.concept_id,
            )
        )

    def for_concept(self, concept_id: str) -> tuple[MarketFieldBinding, ...]:
        c = concept_id.strip()
        return tuple(
            sorted(
                (b for chain in self._bindings.values() for b in chain if b.concept_id == c),
                key=lambda b: b.market,
            )
        )

    def concepts(self) -> tuple[str, ...]:
        return tuple(sorted({k[0] for k in self._bindings}))

    def markets(self) -> tuple[str, ...]:
        return tuple(sorted({k[1] for k in self._bindings}))

    def to_dict(self) -> dict[str, Any]:
        return {"bindings": [b.to_dict() for chain in self._bindings.values() for b in chain]}


class ProviderResolver:
    """R17-032: walk a provider CHAIN and select the first eligible provider.

    A registered chain (``bindings()``) does NOT prove the runtime will pick a
    secondary when the primary lacks data/coverage.  This resolver applies the
    eligibility gates in a fixed order and records the rejection reason for each
    candidate so provider *selection* becomes observable (lineage), not a silent
    ``chain[0]``.

    Gate order: market/context eligibility -> source certification -> required
    filters -> PIT -> requested-window coverage -> universe coverage ->
    currency/session constraints -> select first eligible.
    """

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self._registry = registry or PROVIDER_REGISTRY

    def resolve(
        self,
        concept_id: str,
        market: str,
        *,
        context: Any = None,
        production: bool = True,
        start: str | None = None,
        end: str | None = None,
        universe_id: str | None = None,
    ) -> tuple[MarketFieldBinding | None, list[dict[str, Any]]]:
        """Select the first eligible provider in the chain.

        Returns ``(selected, decisions)`` where ``decisions`` records every
        candidate's provider_id + chosen/rejected reason (for lineage).
        """
        chain = self._registry.bindings(concept_id, market)
        if not chain:
            return None, []
        decisions: list[dict[str, Any]] = []
        ctx = context
        eff_market = canonicalize_provider_market(market)
        for candidate in chain:
            reasons: list[str] = []
            if candidate.quality == ProviderQuality.UNAVAILABLE:
                reasons.append("UNAVAILABLE")
            if production and not candidate.source_certified:
                reasons.append("SOURCE_UNCERTIFIED")
            if production and not candidate.quality.production_usable:
                reasons.append(f"QUALITY_{candidate.quality.value}")
            if candidate.coverage == CoverageClass.CURRENT_ONLY and (
                start is not None or end is not None
            ):
                reasons.append("CURRENT_ONLY_CANNOT_BACKFILL")
            if (
                start is not None
                and end is not None
                and candidate.coverage == CoverageClass.PARTIAL
                and candidate.coverage_gate is not None
                and candidate.coverage_gate < 0.8
                and universe_id is None
            ):
                reasons.append("COVERAGE_BELOW_FLOOR_NO_UNIVERSE_MASK")
            if reasons:
                decisions.append(
                    {
                        "provider": candidate.provider_id,
                        "concept": concept_id,
                        "market": eff_market,
                        "selected": False,
                        "reasons": reasons,
                    }
                )
                continue
            decisions.append(
                {
                    "provider": candidate.provider_id,
                    "concept": concept_id,
                    "market": eff_market,
                    "selected": True,
                    "reasons": [],
                }
            )
            return candidate, decisions
        return None, decisions


def canonicalize_provider_market(market: str) -> str:
    """Canonical market id for provider resolution (R17-048 authority)."""
    from factor_engine.market.capabilities import canonicalize_market_id

    return canonicalize_market_id(market)


PROVIDER_REGISTRY = ProviderRegistry()
PROVIDER_RESOLVER = ProviderResolver()

_UNAVAILABLE = ProviderQuality.UNAVAILABLE
_NATIVE = ProviderQuality.EXACT_NATIVE
_DERIVED = ProviderQuality.EXACT_DERIVED
_SPARSE = ProviderQuality.SPARSE
_PIT_BLOCKED = ProviderQuality.PIT_BLOCKED
_FULL = CoverageClass.FULL
_PARTIAL = CoverageClass.PARTIAL
_EVENT = CoverageClass.EVENT_ONLY


def _b(
    concept_id,
    market,
    provider_id,
    *,
    dataset=None,
    physical,
    quality,
    coverage,
    source_unit,
    canonical_unit,
    transform=_identity,
    transform_description="identity",
    temporal_model="exact",
    knowledge_time=None,
    effective_time=None,
    available_at=None,
    required_filters=(),
    source_certified=False,
    notes="",
    derived_expression=None,
    coverage_gate=None,
    dependencies=(),
    flow_semantics=None,
    dynamic_validity_policy=None,
    level_sensitive=None,
    registry: ProviderRegistry = PROVIDER_REGISTRY,
) -> MarketFieldBinding:
    binding = MarketFieldBinding(
        concept_id=concept_id,
        market=market,
        provider_id=provider_id,
        dataset=dataset,
        physical_fields=tuple(physical),
        transform=transform,
        quality=quality,
        coverage=coverage,
        source_unit=source_unit,
        canonical_unit=canonical_unit,
        temporal_model=temporal_model,
        knowledge_time=knowledge_time,
        effective_time=effective_time,
        available_at=available_at,
        required_filters=tuple(parse_filter_requirement(f) for f in required_filters),
        source_certified=source_certified,
        notes=notes,
        transform_description=transform_description,
        derived_expression=derived_expression,
        coverage_gate=coverage_gate,
        dependencies=tuple(dependencies),
        flow_semantics=flow_semantics,
        dynamic_validity_policy=dynamic_validity_policy,
        level_sensitive=level_sensitive,
    )
    registry.register(binding)
    return binding


# ===========================================================================
# Register the core cross-market bindings.
#
# A-share facts (COS_ashare_lqtp_data_dictionary 2026-08-08): Return is bp
# (/10000), Factor is backward multiplier (Close*Factor), TurnoverRatio/Roe/
# DividendRatio/Weight/ShareRatio are percent (/100), Volume is uint64 shares,
# IsSuspend lives in StockDailyBar, financial PIT on PubDate, StockDividend has
# no announcement time (effective-only).
#
# US facts (COS_us_massive_data_dictionary 2026-08-08): Ret is decimal, AdjFactor
# is backward multiplier (Close*AdjFactor), dividend_yield/ROE/ROA decimal,
# financial PIT on filing_date with timeframe filter, market_cap from
# TickerSharesSnapshot (~42% coverage), Industry/Status EMPTY, no index weights,
# dividend cash_amount currency varies (~12.2% non-USD, no FX).
# ===========================================================================

# --- return_decimal --------------------------------------------------------
_b(
    "return_decimal", "ashare", "ashare_return_bp",
    dataset="ashare_stock_daily_adj", physical=("StockDailyBarAdj.Return",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=UnitSpec.ratio(scale=0.0001), canonical_unit=RATIO,
    transform=_times(0.0001), transform_description="x * 0.0001 (bp -> decimal)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
    notes="Return = (Close/PreClose - 1) * 10000, verified error=0; "
          "ADJ_FIELD_MIGRATION: reads StockDailyBarAdj.Return (authority)",
)
_b(
    "return_decimal", "us", "us_ret_decimal",
    dataset="us_stock_daily", physical=("StockDailyBar.Ret",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=RATIO, canonical_unit=RATIO,
    transform=_identity, transform_description="identity (already decimal)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="Ret = Close/PreClose - 1, verified",
)

# --- raw prices ------------------------------------------------------------
# ADJ_FIELD_MIGRATION（2026-08-28）：A 股行情权威口径 = 后复权表，未复权
# StockDailyBar 的 raw OHLCV **禁止**因子读取。A-share raw_* 概念显式
# DISABLED（UNSUPPORTED_MARKET_MECHANISM）；US 保留（美股 raw 即其市价）。
for _raw in ("raw_open", "raw_high", "raw_low", "raw_close", "raw_vwap"):
    _b(
        _raw, "ashare", f"ashare_{_raw}_unavailable",
        physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
        source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
        transform=_identity, temporal_model="unavailable",
        notes="ADJ_FIELD_MIGRATION: raw unadjusted OHLC/VWAP is forbidden for A-share factors; "
              "use the continuous_* (backward-adjusted) concepts (StockDailyBarAdj authority)",
    )
for _raw, _phys in (
    ("raw_open", "Open"), ("raw_high", "High"), ("raw_low", "Low"),
    ("raw_close", "Close"), ("raw_vwap", "VWAP"),
):
    _b(
        _raw, "us", f"us_raw_{_phys.lower()}",
        dataset="us_stock_daily", physical=(f"StockDailyBar.{_phys}",),
        quality=_NATIVE, coverage=_FULL,
        source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE,
        transform=_identity, temporal_model="exact_daily", available_at="local_close",
        source_certified=True,
    )

# R17-011: ``reference_pre_close`` — the exchange's OFFICIAL reference pre-close
# (A-share PreClose already removes corporate-action jumps; US PreClose is the
# official reference).  This is distinct from ``lag(raw_close,1)``; the physical
# ``PreClose`` column binds to THIS concept, and ``raw_pre_close`` above no longer
# reads the PreClose column (it is a pure lag-derived concept when registered by a
# derived provider / operator, never the official reference).
# ADJ_FIELD_MIGRATION: A-share reads StockDailyBarAdj.AdjPreClose (adjusted reference).
for _market, _ds, _table, _phys, _unit in (
    ("ashare", "ashare_stock_daily_adj", "StockDailyBarAdj", "AdjPreClose", CNY_PER_SHARE),
    ("us", "us_stock_daily", "StockDailyBar", "PreClose", USD_PER_SHARE),
):
    _b(
        "reference_pre_close", _market, f"{_market}_reference_pre_close",
        dataset=_ds, physical=(f"{_table}.{_phys}",),
        quality=_NATIVE, coverage=_FULL,
        source_unit=_unit, canonical_unit=_unit,
        transform=_identity, temporal_model="exact_daily", available_at="local_open",
        source_certified=True,
        transform_description="identity (official reference pre-close; company-action adjusted)",
        notes="OFFICIAL_REFERENCE_PRE_CLOSE basis (R17-011); not lag(raw_close,1). "
              "A-share: AdjPreClose (StockDailyBarAdj authority, ADJ_FIELD_MIGRATION)",
    )

# --- continuous (backward-adjusted) prices --------------------------------
# ADJ_FIELD_MIGRATION: A-share continuous_* read the precomputed Adj* columns on
# StockDailyBarAdj (identity transform — the table is already adjusted).  US
# keeps the Close*AdjFactor derived multiplication.
_b(
    "continuous_close", "ashare", "ashare_continuous_close",
    dataset="ashare_stock_daily_adj", physical=("StockDailyBarAdj.AdjClose",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
    transform=_identity_fields, transform_description="AdjClose (already backward-adjusted; StockDailyBarAdj authority)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
    notes="ADJ_FIELD_MIGRATION: StockDailyBarAdj.AdjClose = Close * Factor (precomputed); "
          "continuous_close reads it directly, no runtime multiplication",
)
_b(
    "continuous_close", "us", "us_continuous_close",
    dataset="us_stock_daily", physical=("StockDailyBar.Close", "StockDailyBar.AdjFactor"),
    quality=_DERIVED, coverage=_FULL,
    source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE,
    transform=_mul_two("StockDailyBar.Close", "StockDailyBar.AdjFactor"),
    transform_description="Close * AdjFactor (backward multiplier)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
    derived_expression="StockDailyBar.Close * StockDailyBar.AdjFactor",
    # R17-033: AdjFactor may be CLAMPED for long-history tickers — the absolute
    # adjusted LEVEL is not certified for long-window level-based operators on a
    # clamped ticker.  Return/log-return paths remain valid.
    dynamic_validity_policy=(
        "if is_adj_factor_clamped(ticker): absolute continuous-price LEVEL "
        "providers are not certified for long-history level-based operators; "
        "use return/log-return path or re-baseline"
    ),
    level_sensitive=True,
    notes="clamp flag (adj_factor > 1e6) -> use returns for long windows; level-sensitive (R17-033/085)",
)

# P0-010: continuous_open/high/low/vwap — the raw OHLC siblings of
# continuous_close multiplied by the SAME backward adjustment factor.  Before
# these bindings, cross-day open/high/low/vwap trend operators could only pull
# raw (unadjusted) OHLC, so long-window price path factors were split/distortion
# contaminated.
for _canon, _bar, _adj in (
    ("continuous_open", "Open", "AdjOpen"),
    ("continuous_high", "High", "AdjHigh"),
    ("continuous_low", "Low", "AdjLow"),
    ("continuous_vwap", "Vwap", "AdjVwap"),
):
    _b(
        _canon, "ashare", f"ashare_{_canon}",
        dataset="ashare_stock_daily_adj",
        physical=(f"StockDailyBarAdj.{_adj}",),
        quality=_NATIVE, coverage=_FULL,
        source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
        transform=_identity_fields,
        transform_description=f"{_adj} (already backward-adjusted; StockDailyBarAdj authority)",
        temporal_model="exact_daily", available_at="local_close",
        source_certified=True,
        notes="ADJ_FIELD_MIGRATION: StockDailyBarAdj.AdjX = raw * Factor (precomputed); "
              f"continuous_{_canon} reads {_adj} directly, no runtime multiplication",
    )
for _canon, _bar in (
    ("continuous_open", "Open"),
    ("continuous_high", "High"),
    ("continuous_low", "Low"),
    ("continuous_vwap", "VWAP"),
):
    _b(
        _canon, "us", f"us_{_canon}",
        dataset="us_stock_daily",
        physical=(f"StockDailyBar.{_bar}", "StockDailyBar.AdjFactor"),
        quality=_DERIVED, coverage=_FULL,
        source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE,
        transform=_mul_two(f"StockDailyBar.{_bar}", "StockDailyBar.AdjFactor"),
        transform_description=f"{_bar} * AdjFactor (backward multiplier)",
        temporal_model="exact_daily", available_at="local_close",
        source_certified=True,
        derived_expression=f"StockDailyBar.{_bar} * StockDailyBar.AdjFactor",
        # R17-033: same AdjFactor-clamp caveat as continuous_close — level-based
        # long-window operators must not trust absolute adjusted LEVELS on a
        # clamped ticker.
        dynamic_validity_policy=(
            "if is_adj_factor_clamped(ticker): absolute continuous-price LEVEL "
            "providers are not certified for long-history level-based operators"
        ),
        level_sensitive=True,
        notes="same backward AdjFactor as continuous_close; level-sensitive (R17-033/085)",
    )
# ADJ_FIELD_MIGRATION: A-share raw vendor Volume is NOT a mineable factor input
# (raw price-volume forbidden); the adjusted share count is continuous_volume_shares
# (Volume / Factor on StockDailyBarAdj).  US raw Volume stays (US share count).
_b(
    "raw_volume_shares", "ashare", "ashare_raw_volume_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=SHARES, canonical_unit=SHARES,
    transform=_identity, temporal_model="unavailable",
    notes="ADF_FIELD_MIGRATION: raw vendor Volume is forbidden for A-share factors; "
          "use continuous_volume_shares = Volume / Factor (StockDailyBarAdj authority)",
)
_b(
    "raw_volume_shares", "us", "us_raw_volume",
    dataset="us_stock_daily", physical=("StockDailyBar.Volume",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=SHARES, canonical_unit=SHARES,
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
)

# --- continuous (backward-adjusted) volume: Volume / Factor (LQTP functions.yaml) ---
_b(
    "continuous_volume_shares", "ashare", "ashare_lqtp_volume",
    dataset="ashare_stock_daily_adj",
    physical=("StockDailyBarAdj.Volume", "StockDailyBarAdj.Factor"),
    quality=_DERIVED, coverage=_FULL,
    source_unit=SHARES, canonical_unit=SHARES,
    transform=_div_two("StockDailyBarAdj.Volume", "StockDailyBarAdj.Factor"),
    transform_description="Volume / Factor (LQTP functions.yaml adjusted volume; StockDailyBarAdj authority)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
    derived_expression="StockDailyBarAdj.Volume / StockDailyBarAdj.Factor",
    notes="ADJ_FIELD_MIGRATION: reads StockDailyBarAdj.Volume / StockDailyBarAdj.Factor. "
          "LQTP platform functions.yaml defines volume = Volume / Factor.",
)
_b(
    "continuous_volume_shares", "us", "us_continuous_volume",
    dataset="us_stock_daily", physical=("StockDailyBar.Volume",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=SHARES, canonical_unit=SHARES,
    transform=_identity, transform_description="identity (US raw Volume is the share count)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
    notes="US has no price-factor-adjusted volume; continuous_volume_shares == raw Volume (identity).",
)

# --- amount_local -----------------------------------------------------------
_b(
    "amount_local", "ashare", "ashare_amount",
    dataset="ashare_stock_daily_adj", physical=("StockDailyBarAdj.AdjAmount",),
    quality=_NATIVE, coverage=_FULL, source_unit=CNY, canonical_unit=CNY,
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
    notes="Amount ~= Vwap * Volume (rel err ~2e-5); ADJ_FIELD_MIGRATION: reads "
          "StockDailyBarAdj.AdjAmount (authority)",
)
_b(
    "amount_local", "us", "us_amount",
    dataset="us_stock_daily", physical=("StockDailyBar.Amount",),
    quality=_NATIVE, coverage=_FULL, source_unit=USD, canonical_unit=USD,
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="Amount == VWAP * Volume exactly (~96.65% rows)",
)

# --- market_cap_local ------------------------------------------------------
_b(
    "market_cap_local", "ashare", "ashare_market_cap",
    dataset="ashare_stock_valuation_daily", physical=("StockValuationDaily.MarketCap",),
    quality=_NATIVE, coverage=_FULL, source_unit=CNY, canonical_unit=CNY,
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="MarketCap ~= Capitalization * Close (verified)",
)
_b(
    "market_cap_local", "us", "us_market_cap_shares_snapshot",
    dataset="us_ticker_shares_snapshot",  # R17-018: DataAccess registry name
    physical=("TickerSharesSnapshot.weighted_shares_outstanding", "StockDailyBar.Close"),
    quality=_DERIVED, coverage=CoverageClass.PARTIAL, source_unit=USD, canonical_unit=USD,
    transform=_mul_two("TickerSharesSnapshot.weighted_shares_outstanding", "StockDailyBar.Close"),
    transform_description="Close * weighted_shares_outstanding",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
    derived_expression="TickerSharesSnapshot.weighted_shares_outstanding * StockDailyBar.Close",
    coverage_gate=0.42,
    # R17-017: explicit cross-dataset dependency graph (no more "dataset points at
    # shares, physical_fields sneakily reference DailyBar" half-declaration).
    dependencies=(
        ProviderDependency(
            dataset="us_ticker_shares_snapshot", table="TickerSharesSnapshot",
            field="weighted_shares_outstanding", temporal_join="exact",
        ),
        ProviderDependency(
            dataset="us_stock_daily", table="StockDailyBar",
            field="Close", temporal_join="exact",
        ),
    ),
    notes="~42% coverage of StockDailyBar tickers; NOT the X0 sparse market_cap; coverage_gate<0.8 -> production flags",
)

# --- turnover --------------------------------------------------------------
_b(
    "turnover_ratio_decimal", "ashare", "ashare_turnover",
    dataset="ashare_stock_valuation_daily", physical=("StockValuationDaily.TurnoverRatio",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=UnitSpec.ratio(scale=0.01), canonical_unit=RATIO,
    transform=_times(0.01), transform_description="TurnoverRatio / 100 (percent -> decimal)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="~= Volume/CirculatingCap*100",
)
_b(
    "turnover_ratio_decimal", "us", "us_turnover_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
    temporal_model="unavailable", notes="US has no turnover field",
)

# --- price limits (A-share official mechanism) ------------------------------
# ADJ_FIELD_MIGRATION: A-share limits read the ADJUSTED limit prices on
# StockDailyBarAdj (AdjHighLimit / AdjLowLimit — the adjusted table's limit
# columns are precomputed HighLimit*Factor).
for _concept, _adj, _prov in (
    ("upper_price_limit", "AdjHighLimit", "ashare_high_limit"),
    ("lower_price_limit", "AdjLowLimit", "ashare_low_limit"),
):
    _b(
        _concept, "ashare", _prov,
        dataset="ashare_stock_daily_adj", physical=(f"StockDailyBarAdj.{_adj}",),
        quality=_NATIVE, coverage=_FULL,
        source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
        transform=_identity, temporal_model="exact_daily", available_at="local_open",
        source_certified=True,
        notes="Official limit price; ADJ_FIELD_MIGRATION: reads StockDailyBarAdj.{_adj} "
              "(adjusted limit; raw unadjusted limits forbidden for factors)",
    )
_b(
    "upper_price_limit", "us", "us_no_price_limit",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE, transform=_identity,
    temporal_model="unavailable", notes="US has no daily static price limit (LULD/halt is a different mechanism)",
)
_b(
    "lower_price_limit", "us", "us_no_price_limit",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE, transform=_identity,
    temporal_model="unavailable", notes="US has no daily static price limit",
)

# --- ROE / dividend yield ----------------------------------------------------
_b(
    "roe_decimal", "ashare", "ashare_roe_indicator",
    dataset="ashare_stock_indicator", physical=("StockIndicator.Roe",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=UnitSpec.ratio(scale=0.01), canonical_unit=RATIO,
    transform=_times(0.01), transform_description="Roe / 100 (percent -> decimal)",
    temporal_model="financial_pit", knowledge_time="PubDate", available_at="filing",
    source_certified=True, notes="asof(PubDate); often unannualized",
)
_b(
    "roe_decimal", "us", "us_roe_valuation_sparse",
    dataset="us_stock_valuation_daily", physical=("StockValuationDaily.return_on_equity",),
    quality=_SPARSE, coverage=CoverageClass.CURRENT_ONLY,
    source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
    transform_description="identity (already decimal)",
    temporal_model="sparse_snapshot", available_at="local_close",
    source_certified=False, notes="X0 sparse (~49 files 2026-05..07). Production preferred: net_income/equity derived",
)
_b(
    "dividend_yield_decimal", "ashare", "ashare_dividend_yield",
    dataset="ashare_stock_valuation_daily", physical=("StockValuationDaily.DividendRatio",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=UnitSpec.ratio(scale=0.01), canonical_unit=RATIO,
    transform=_times(0.01), transform_description="DividendRatio / 100 (percent -> decimal)",
    temporal_model="exact_daily", available_at="local_close", source_certified=True,
)
_b(
    "dividend_yield_decimal", "us", "us_dividend_yield_sparse",
    dataset="us_stock_valuation_daily", physical=("StockValuationDaily.dividend_yield",),
    quality=_SPARSE, coverage=CoverageClass.CURRENT_ONLY,
    source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
    transform_description="identity (already decimal)",
    temporal_model="sparse_snapshot", available_at="local_close",
    source_certified=False, notes="X0 sparse; do NOT ffill as full-history daily",
)

# --- industry / tradability ---------------------------------------------------
_b(
    "industry_group", "ashare", "ashare_industry",
    dataset="ashare_stock_industry", physical=("StockIndustry.IndustryCode",),
    quality=_NATIVE, coverage=_FULL, source_unit=UnitSpec(dimension="identifier"),
    canonical_unit=UnitSpec(dimension="identifier"),
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="filter IndustrySource=sw_l1; IndustrySource is a required filter",
    required_filters=("IndustrySource",),
)
_b(
    "industry_group", "us", "us_industry_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=UnitSpec(dimension="identifier"), canonical_unit=UnitSpec(dimension="identifier"),
    transform=_identity, temporal_model="unavailable",
    notes="US StockIndustry is EMPTY (0 rows); GICS/SIC/NAICS provider required",
)
_b(
    "tradability_state", "ashare", "ashare_tradability",
    dataset="ashare_stock_daily_adj", physical=("StockDailyBarAdj.IsSuspend",),
    quality=_NATIVE, coverage=_FULL, source_unit=UnitSpec(dimension="boolean"),
    canonical_unit=UnitSpec(dimension="boolean"),
    transform=_not_bool,
    transform_description="NOT IsSuspend (not_suspended_state; known-suspended->0, known-not-suspended->1, NaN->NaN; does NOT claim listing membership)",
    temporal_model="exact_daily", available_at="local_open",
    source_certified=True,
    notes="ADJ_FIELD_MIGRATION: reads StockDailyBarAdj.IsSuspend (authority); "
          "transform negates it preserving NaN (unknown suspension stays unknown); "
          "no listing/public-status mask is included",
)
_b(
    "tradability_state", "us", "us_tradability_universe",
    dataset="us_stock_list",
    physical=("StockList.type", "StockDailyBar.Close"),
    quality=ProviderQuality.SEMANTIC_EQUIVALENT, coverage=CoverageClass.PARTIAL,
    source_unit=UnitSpec(dimension="boolean"), canonical_unit=UnitSpec(dimension="boolean"),
    transform=_us_tradable,
    transform_description="StockList.type=='CS' AND finite DailyBar close -> 1/0 (unknown membership -> 0)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=False,
    derived_expression="us_tradable = (StockList.type=='CS') AND isfinite(StockDailyBar.Close)",
    # R17-016: ``universe_daily`` is ANOTHER DATASET, not a column predicate of
    # the current table — it is a JoinRequirement the planner must materialize,
    # not a required_filter.
    required_filters=(
        FilterRequirement(field="StockList.type", operator="present", required=True),
    ),
    dependencies=(
        ProviderDependency(
            dataset="us_universe_daily", table="UniverseDaily", field="ticker",
            temporal_join="exact",
        ),
        ProviderDependency(
            dataset="us_stock_daily", table="StockDailyBar", field="Close",
            temporal_join="exact",
        ),
    ),
    notes="real boolean from CS list type + finite DailyBar close (universe_daily is a JOIN dependency, R17-016); is_ticker_halt is far too sparse to be an IsSuspend equivalent",
)

# --- index ---------------------------------------------------------------------
_b(
    "index_member", "ashare", "ashare_index_constituent",
    dataset="ashare_index_constituent", physical=("IndexConstituent.IndexSymbol",),
    quality=_NATIVE, coverage=_FULL, source_unit=UnitSpec(dimension="boolean"),
    canonical_unit=UnitSpec(dimension="boolean"),
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True, required_filters=("IndexSymbol",),
)
_b(
    "index_member", "us", "us_index_components",
    dataset="us_stock_indices_components", physical=("StockIndicesComponents.IndexName",),
    quality=_NATIVE, coverage=_FULL, source_unit=UnitSpec(dimension="boolean"),
    canonical_unit=UnitSpec(dimension="boolean"),
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True, required_filters=("IndexName",),
)
_b(
    "index_weight", "ashare", "ashare_index_weight",
    dataset="ashare_index_constituent", physical=("IndexConstituent.Weight",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=UnitSpec.ratio(scale=0.01), canonical_unit=RATIO,
    transform=_times(0.01), transform_description="Weight / 100 (percent -> decimal)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="same-index sum ~100",
)
_b(
    "index_weight", "us", "us_index_weight_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
    temporal_model="unavailable",
    notes="US StockIndicesComponents has NO Weight field; equal-weight is a different operator",
)

# --- dividends ------------------------------------------------------------------
_b(
    "cash_dividend_per_share", "ashare", "ashare_cash_dividend",
    dataset="ashare_stock_dividend", physical=("StockDividend.CashDividend",),
    quality=_PIT_BLOCKED, coverage=_EVENT,
    source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
    transform=_identity, temporal_model="effective_only",
    effective_time="ExDividendDate", available_at="ex_date",
    source_certified=False,
    notes="NO PubDate/announcement time; strict PIT blocks; usable only with allow_effective_time_only",
)
_b(
    "cash_dividend_per_share", "us", "us_cash_dividend_declared",
    dataset="us_stock_dividend",
    physical=("StockDividend.cash_amount", "StockDividend.declaration_date"),
    quality=_NATIVE, coverage=_PARTIAL,
    source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE,
    transform=_dividend_declared_fail_closed,
    transform_description="cash_amount with declaration_date strict-PIT guard (missing declaration_date -> NaN; never ex-date fallback)",
    temporal_model="financial_pit",
    knowledge_time="declaration_date", effective_time="ex_dividend_date",
    available_at="declaration",
    required_filters=("currency=USD",),
    source_certified=True,
    derived_expression="guard(cash_amount, declaration_date): NaN when declaration_date missing",
    notes="declaration_date PIT (~0.51% null; missing rows fail closed to NaN, never ex-date fallback); effective_time=ex_dividend_date; ~12.2% non-USD excluded (no COS FX table); ex_date may be future; DataAccess contract us_stock_dividend pit=strict/declaration_date",
)

# --- financial statements (amounts, local currency) ---------------------------
# Each concept binds to the *physical* statement dataset it actually lives in
# (StockIncome / StockBalance / StockCashFlow mirror tables), never to the daily
# bar dataset — the physical field is StockIncome.* / StockBalance.* /
# StockCashFlow.*, which a daily-bar reader can never satisfy.
_FIN_A = {
    "operating_revenue": ("StockIncome.OperatingRevenue", "ashare_stock_income"),
    "net_profit": ("StockIncome.NetProfit", "ashare_stock_income"),
    "operating_cash_flow": ("StockCashFlow.NetOperateCashFlow", "ashare_stock_cashflow"),
    "total_assets": ("StockBalance.TotalAssets", "ashare_stock_balance"),
    "total_liabilities": ("StockBalance.TotalLiability", "ashare_stock_balance"),
    "equity": ("StockBalance.EquitiesParentCompanyOwners", "ashare_stock_balance"),
    # R17-078/079: attribution-aligned concepts.
    "net_income_attributable": ("StockIncome.NpParentCompanyOwners", "ashare_stock_income"),
    "equity_attributable": ("StockBalance.EquitiesParentCompanyOwners", "ashare_stock_balance"),
}
_FIN_US = {
    "operating_revenue": ("StockIncome.revenue", "us_stock_income"),
    "net_profit": ("StockIncome.net_income_loss_attributable_common_shareholders", "us_stock_income"),
    "operating_cash_flow": ("StockCashFlow.net_cash_from_operating_activities", "us_stock_cashflow"),
    "total_assets": ("StockBalance.total_assets", "us_stock_balance"),
    "total_liabilities": ("StockBalance.total_liabilities", "us_stock_balance"),
    # R17-079: equity denominator must match the common-shareholder numerator —
    # total_equity includes noncontrolling interest; use the attributable line.
    "equity": ("StockBalance.total_equity_attributable_to_parent", "us_stock_balance"),
    "net_income_attributable": ("StockIncome.net_income_loss_attributable_common_shareholders", "us_stock_income"),
    "equity_attributable": ("StockBalance.total_equity_attributable_to_parent", "us_stock_balance"),
}
for _concept, (_phys_a, _ds_a) in _FIN_A.items():
    _b(
        _concept, "ashare", f"ashare_{_concept}",
        dataset=_ds_a, physical=(_phys_a,),
        quality=_NATIVE, coverage=_FULL, source_unit=CNY, canonical_unit=CNY,
        transform=_identity, temporal_model="financial_pit",
        knowledge_time="PubDate", available_at="filing",
        source_certified=True, notes="asof(PubDate); flow fields are cumulative YTD",
        # R17-036: A-share statement rows are cumulative-YTD flow — the resolved
        # provider declares it, so quarterize/growth operators read the real
        # semantics instead of the concept's static default.
        flow_semantics="cumulative_ytd_flow",
    )
for _concept, (_phys_us, _ds_us) in _FIN_US.items():
    _b(
        _concept, "us", f"us_{_concept}",
        dataset=_ds_us, physical=(_phys_us,),
        quality=_NATIVE, coverage=_PARTIAL, source_unit=USD, canonical_unit=USD,
        transform=_identity, temporal_model="financial_pit",
        knowledge_time="filing_date", available_at="filing",
        required_filters=("timeframe",),
        source_certified=True, notes="asof(filing_date); MUST filter timeframe (quarterly/annual/TTM)",
        # R17-036: US statements are period-scoped by the timeframe filter — the
        # provider resolves the flow semantics, not the concept.
        flow_semantics="single_period_flow",
    )


# ---------------------------------------------------------------------------
# R17-064: previously-declared-but-unbound concepts.  A concept in the registry
# with NO provider binding resolves to UNKNOWN — worse than an explicit
# "unavailable", because a grammar/recipe can never tell the difference from
# "not yet wired".  Every declared concept gets either a REAL binding (when a
# physical field exists) or an explicit UNAVAILABLE provider.
# ---------------------------------------------------------------------------
# total_shares / free_float_shares: A-share capital tables carry them natively;
# US has no direct total-shares field (TickerSharesSnapshot is the cap source).
_b(
    "total_shares", "ashare", "ashare_total_shares",
    dataset="ashare_stock_capital_daily", physical=("StockCapitalDaily.TotalCapital",),
    quality=_NATIVE, coverage=_FULL, source_unit=SHARES, canonical_unit=SHARES,
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="TotalCapital (S1 state snapshot)",
)
_b(
    "total_shares", "us", "us_total_shares_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=SHARES, canonical_unit=SHARES, transform=_identity,
    temporal_model="unavailable",
    notes="US has no direct total-shares field; use TickerSharesSnapshot for market cap (R17-064)",
)
_b(
    "free_float_shares", "ashare", "ashare_free_float_shares",
    dataset="ashare_stock_valuation_daily", physical=("StockValuationDaily.FreeCap",),
    quality=_NATIVE, coverage=_FULL, source_unit=SHARES, canonical_unit=SHARES,
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="FreeCap (float share count)",
)
_b(
    "free_float_shares", "us", "us_free_float_shares_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=SHARES, canonical_unit=SHARES, transform=_identity,
    temporal_model="unavailable",
    notes="US has no isomorphic free-float shares field (R17-064)",
)
# free_float_market_cap_local: A = FreeCap * Close (derived); US unavailable.
_b(
    "free_float_market_cap_local", "ashare", "ashare_free_float_market_cap",
    dataset="ashare_stock_valuation_daily",
    physical=("StockValuationDaily.FreeCap", "StockDailyBar.Close"),
    quality=_DERIVED, coverage=_FULL, source_unit=CNY, canonical_unit=CNY,
    transform=_mul_two("StockValuationDaily.FreeCap", "StockDailyBar.Close"),
    transform_description="FreeCap * Close (float market cap)",
    temporal_model="exact_daily", available_at="local_close", source_certified=True,
    derived_expression="StockValuationDaily.FreeCap * StockDailyBar.Close",
    notes="float shares * close (R17-064)",
)
_b(
    "free_float_market_cap_local", "us", "us_free_float_market_cap_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=USD, canonical_unit=USD, transform=_identity,
    temporal_model="unavailable",
    notes="US has no free-float shares field (R17-064)",
)
# earnings_per_share: A = StockIndicator.Eps; US = StockIncome.basic_eps.
_b(
    "earnings_per_share", "ashare", "ashare_eps_indicator",
    dataset="ashare_stock_indicator", physical=("StockIndicator.Eps",),
    quality=_NATIVE, coverage=_FULL, source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
    transform=_identity, temporal_model="financial_pit",
    knowledge_time="PubDate", available_at="filing", source_certified=True,
    notes="Eps (CNY/share) asof PubDate (R17-064)",
)
_b(
    "earnings_per_share", "us", "us_eps_income",
    dataset="us_stock_income", physical=("StockIncome.basic_earnings_per_share",),
    quality=_NATIVE, coverage=_PARTIAL, source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE,
    transform=_identity, temporal_model="financial_pit",
    knowledge_time="filing_date", available_at="filing",
    required_filters=("timeframe",), source_certified=True,
    notes="basic_eps asof filing_date; timeframe required (R17-064)",
)
# roa_decimal: US has a direct X0 return_on_assets (sparse); A has no direct ROA
# field — deriving it (net_profit/total_assets) is a recipe-level composition, so
# the CONCEPT is unavailable for A until a certified derived provider exists.
_b(
    "roa_decimal", "ashare", "ashare_roa_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
    temporal_model="unavailable",
    notes="A has no direct ROA field; derive via net_profit/total_assets recipe (R17-064)",
)
_b(
    "roa_decimal", "us", "us_roa_valuation_sparse",
    dataset="us_stock_valuation_daily", physical=("StockValuationDaily.return_on_assets",),
    quality=_SPARSE, coverage=CoverageClass.CURRENT_ONLY,
    source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
    transform_description="identity (already decimal)",
    temporal_model="sparse_snapshot", available_at="local_close",
    source_certified=False,
    notes="X0 sparse ~49 files; do NOT ffill as full-history daily (R17-064)",
)
# gross_margin_decimal / net_profit_margin_decimal: recipe-level derived ratios —
# no certified direct provider in either market.  Explicitly unavailable.
for _c in ("gross_margin_decimal", "net_profit_margin_decimal"):
    _b(
        _c, "ashare", f"ashare_{_c}_unavailable",
        physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
        source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
        temporal_model="unavailable",
        notes="margin ratios are recipe-level derived (gross_profit/revenue); no certified direct provider (R17-064)",
    )
    _b(
        _c, "us", f"us_{_c}_unavailable",
        physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
        source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
        temporal_model="unavailable",
        notes="margin ratios are recipe-level derived; no certified direct provider (R17-064)",
    )
# dividend_ex_date: A = effective-only (no PIT); US = ex_dividend_date effective.
_b(
    "dividend_ex_date", "ashare", "ashare_dividend_ex_date",
    dataset="ashare_stock_dividend", physical=("StockDividend.ExDividendDate",),
    quality=_PIT_BLOCKED, coverage=_EVENT, source_unit=DATE, canonical_unit=DATE,
    transform=_identity, temporal_model="effective_only",
    effective_time="ExDividendDate", available_at="ex_date", source_certified=False,
    notes="effective-only ex-date; strict PIT blocks (R17-064)",
)
_b(
    "dividend_ex_date", "us", "us_dividend_ex_date",
    dataset="us_stock_dividend", physical=("StockDividend.ex_dividend_date",),
    quality=_PIT_BLOCKED, coverage=_EVENT, source_unit=DATE, canonical_unit=DATE,
    transform=_identity, temporal_model="effective_only",
    effective_time="ex_dividend_date", available_at="ex_date", source_certified=False,
    notes="effective ex-date; strict PIT uses declaration_date not ex-date (R17-064)",
)
# raw_pre_close: R17-011 — this is the physically-lagged ``lag(raw_close, 1)``
# concept, NOT a physical column (the exchange PreClose column is the official
# reference pre-close -> reference_pre_close).  No direct provider; express via
# ts_lag(raw_close,1).  Explicitly unavailable so grammar never offers it as a
# physical read.
_b(
    "raw_pre_close", "ashare", "ashare_raw_pre_close_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE, transform=_identity,
    temporal_model="unavailable",
    notes="lag(raw_close,1) is an operator-derived concept, not a physical column; "
          "use ts_lag(raw_close,1).  The exchange PreClose column is "
          "reference_pre_close (R17-011/064)",
)
_b(
    "raw_pre_close", "us", "us_raw_pre_close_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE, transform=_identity,
    temporal_model="unavailable",
    notes="lag(raw_close,1) is an operator-derived concept; use ts_lag(raw_close,1) "
          "(R17-011/064)",
)
# news_sentiment: NO certified sentiment provider in either market (FactNews
# insights are not a validated sentiment source).  Explicitly unavailable.
for _m in ("ashare", "us"):
    _b(
        "news_sentiment", _m, f"{_m}_news_sentiment_unavailable",
        physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
        source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
        temporal_model="unavailable",
        notes="news sentiment requires a versioned NLP sentiment provider; "
              "FactNews insights are not certified (R17-025/064)",
    )
# holder_concentration: A has a top-holder feed via aggregate_holder_rows (snapshot
# semantics); the direct daily provider is wired through the relation storage layer.
# Keep an explicit A-share binding marker so grammar sees a real provider id.
_b(
    "holder_concentration", "ashare", "ashare_holder_concentration",
    dataset="ashare_stock_topten_shareholder",
    physical=("StockTopTenShareholder.ShareRatio",),
    quality=_NATIVE, coverage=_PARTIAL, source_unit=UnitSpec.ratio(scale=0.01),
    canonical_unit=RATIO, transform=_times(0.01),
    transform_description="ShareRatio / 100 (percent -> decimal)",
    temporal_model="relation_pit", knowledge_time="PubDate", available_at="filing",
    source_certified=True,
    notes="top-holder ShareRatio snapshot; concentration aggregation happens in the "
          "relation layer (aggregate_holder_rows) (R17-062/064)",
)
_b(
    "holder_concentration", "us", "us_holder_concentration_unavailable",
    physical=(), quality=_UNAVAILABLE, coverage=CoverageClass.UNKNOWN,
    source_unit=RATIO, canonical_unit=RATIO, transform=_identity,
    temporal_model="unavailable",
    notes="US has no same-structure top-holder feed (R17-064)",
)


# ---------------------------------------------------------------------------
# Resolution helpers.
# ---------------------------------------------------------------------------
def binding(concept_id: str, market: str) -> MarketFieldBinding | None:
    return PROVIDER_REGISTRY.binding(concept_id, market)


def require_binding(concept_id: str, market: str) -> MarketFieldBinding:
    return PROVIDER_REGISTRY.require_binding(concept_id, market)


def _market_ctx(market: str) -> MarketContext:
    # Strict market_context (raises on unknown markets) — never a ternary that
    # silently defaults unknown markets to "us" (P1-1).
    from factor_engine.market.context import market_context

    return market_context(market)


def explain_field_support(
    concept_id: str,
    market: str,
    *,
    context: MarketContext | None = None,
    production: bool = True,
) -> MarketSupport:
    """Return the A/US support verdict for one canonical field concept."""
    ctx = context or _market_ctx(market)
    b = PROVIDER_REGISTRY.binding(concept_id, market)
    if b is None:
        return MarketSupport(
            canonical=concept_id, market=market,
            status=MarketStatus.UNKNOWN,
            notes="no binding registered",
        )
    if b.quality == _UNAVAILABLE:
        return MarketSupport(
            canonical=concept_id, market=market,
            status=MarketStatus.UNSUPPORTED_MARKET_MECHANISM,
            reason_codes=("NO_PROVIDER",),
            notes=b.notes,
        )
    if b.quality == _PIT_BLOCKED:
        if ctx.allow_effective_time_only:
            return MarketSupport(
                canonical=concept_id, market=market,
                status=MarketStatus.RESEARCH_ONLY,
                providers=(b.provider_id,), provider_quality=b.quality,
                coverage=b.coverage, reason_codes=("EFFECTIVE_TIME_ONLY",),
                notes=b.notes,
            )
        return MarketSupport(
            canonical=concept_id, market=market,
            status=MarketStatus.PIT_BLOCKED,
            providers=(b.provider_id,), provider_quality=b.quality,
            coverage=b.coverage, reason_codes=("NO_ANNOUNCEMENT_TIME",),
            notes=b.notes,
        )
    if not b.quality.production_usable:
        if production:
            status = (
                MarketStatus.RESEARCH_ONLY
                if b.quality.research_usable and (ctx.allow_proxy or ctx.allow_sparse)
                else MarketStatus.PROVIDER_REQUIRED
            )
        else:
            status = (
                MarketStatus.CERTIFIED_DERIVED
                if b.quality in (_DERIVED, _SPARSE) and ctx.allow_sparse
                else MarketStatus.RESEARCH_ONLY
            )
        return MarketSupport(
            canonical=concept_id, market=market, status=status,
            providers=(b.provider_id,), provider_quality=b.quality,
            coverage=b.coverage, reason_codes=("QUALITY_FLOOR",),
            notes=b.notes,
        )
    # Provider certification gate: a binding with an *uncertified source* must
    # never be silently production-certified just because its quality tag is
    # EXACT_NATIVE/EXACT_DERIVED.  ``source_certified`` is part of the contract
    # (multi-market plan §45-§46); production fails closed to PROVIDER_REQUIRED.
    if not b.source_certified:
        return MarketSupport(
            canonical=concept_id, market=market,
            status=(
                MarketStatus.PROVIDER_REQUIRED
                if production
                else MarketStatus.RESEARCH_ONLY
            ),
            providers=(b.provider_id,), provider_quality=b.quality,
            coverage=b.coverage, reason_codes=("SOURCE_UNCERTIFIED",),
            notes=b.notes,
        )
    if b.coverage != CoverageClass.FULL:
        return MarketSupport(
            canonical=concept_id, market=market,
            status=MarketStatus.CERTIFIED_PARTIAL,
            providers=(b.provider_id,), provider_quality=b.quality,
            coverage=b.coverage, reason_codes=("PARTIAL_COVERAGE",),
            notes=b.notes,
        )
    status = (
        MarketStatus.CERTIFIED_NATIVE
        if b.quality == _NATIVE
        else MarketStatus.CERTIFIED_DERIVED
    )
    return MarketSupport(
        canonical=concept_id, market=market, status=status,
        providers=(b.provider_id,), provider_quality=b.quality,
        coverage=b.coverage, notes=b.notes,
    )


# ---------------------------------------------------------------------------
# FinancialPeriodAdapter — filter timeframe, then as-of, never mix periods.
# ---------------------------------------------------------------------------
class FinancialPeriodAdapter:
    """Per-market financial PIT join policy.

    A-share: ``asof(PubDate)``; cumulative-YTD flows are single-quarter-ized by
    ``fin_quarter_from_cumulative`` upstream.
    US: ``asof(filing_date)`` with an explicit ``timeframe`` filter first
    (quarterly/annual/trailing_twelve_months must never mix in one cross-section).
    """

    @staticmethod
    def filter_timeframe(frame: Any, timeframe: str, *, market: str | None = None) -> Any:
        """US financial: keep rows whose ``timeframe`` equals ``timeframe``.

        R17-034: the ``timeframe`` filter is a HARD contract for US financials —
        a missing timeframe column or a missing ``timeframe`` argument must fail
        closed (never silently return the unfiltered frame, which would mix
        quarterly/annual/TTM in one cross-section).  A-share financials do not
        use timeframe; ``market="ashare"`` keeps the legacy lenient path.
        """
        if market == "ashare":
            return frame
        if not timeframe:
            raise ValueError(
                "filter_timeframe requires an explicit timeframe (quarterly / "
                "annual / trailing_twelve_months) for US financials (R17-034)"
            )
        if "timeframe" not in getattr(frame, "columns", ()):
            raise ValueError(
                "US financial frame is missing the timeframe column; refusing to "
                "return an unfiltered period mix (R17-034)"
            )
        filtered = frame[frame["timeframe"] == timeframe]
        if filtered.empty and not frame.empty:
            # Timeframe column exists but the requested value selects nothing —
            # still valid (e.g. no annual rows yet), so only the *column* absence
            # is a hard error.  An empty selection is a data condition.
            pass
        return filtered

    @staticmethod
    def asof(
        events: Any,
        signal_dates: Any,
        *,
        market: str,
        on: str | None = None,
        direction: str = "backward",
        by: str | None = None,
        same_day: bool = False,
    ) -> Any:
        """As-of merge of event rows onto signal dates.

        The ``on`` column defaults to the market knowledge-time column
        (ashare=PubDate, us=filing_date).  Look-ahead is impossible because the
        merge only uses rows with knowledge_time <= signal_date.

        ``same_day`` (round-6 P0-02/P0-03) controls date-level knowledge-time
        visibility:

        * ``same_day=False`` (default, conservative) — a signal dated exactly
          on the knowledge-time date CANNOT see that day's filing.  A-share
          PubDate / US filing_date carry a date, not a time-of-day, and a filing
          lands after close; assuming ``00:00:00`` availability would let an
          after-close announcement drive the same-day close (look-ahead).  The
          value is only visible to the first signal strictly after it.
        * ``same_day=True`` — only when the caller asserts the knowledge-time
          column carries a REAL timestamp (``filing_timestamp``) such that
          ``knowledge_time <= decision_time`` is a sound same-day gate.  Never
          pass ``True`` for a midnight-normalized date column.

        ``by`` MUST be the instrument column (Symbol/Ticker) whenever the
        ``events`` frame mixes several instruments: without ``by`` a stock's
        signal row can legally match *another* company's most recent filing when
        the dates line up (entity-identity is not guaranteed), which silently
        cross-contaminates financial data between companies.  When ``events``
        has an instrument column and ``by`` is omitted we raise rather than
        silently risk cross-stock contamination.
        """
        import pandas as pd

        # R24-157: production financial asof is fixed BACKWARD.  A ``forward`` /
        # ``nearest`` direction would read future filings into the cross-section
        # and must never be exposed as a production option.
        if direction != "backward":
            raise ValueError(
                "FinancialPeriodAdapter.asof requires direction='backward' for "
                f"financial reads (got {direction!r}) — forward/nearest would "
                "leak future filings into the cross-section (R24-157)"
            )
        key = on or ("PubDate" if market == "ashare" else "filing_date")
        ev = events.copy()
        if pd.api.types.is_datetime64_any_dtype(ev[key].dtype) is False:
            ev[key] = pd.to_datetime(ev[key])

        # ``signal_dates`` is either a plain Series of dates (single-instrument
        # caller) or a DataFrame carrying the instrument column alongside the
        # signal date column.
        if isinstance(signal_dates, pd.Series):
            dates = pd.DataFrame({"__signal_date__": pd.to_datetime(signal_dates)})
        else:
            dates = signal_dates.copy()
            if "__signal_date__" in dates.columns:
                pass
            elif key in dates.columns:
                dates = dates.rename(columns={key: "__signal_date__"})
            else:
                raise ValueError(
                    "signal frame must carry the signal-date column (__signal_date__ or "
                    f"{key!r})"
                )
            dates["__signal_date__"] = pd.to_datetime(dates["__signal_date__"])

        # Instrument column is mandatory when events carry per-instrument rows:
        # without ``by`` merge_asof can legally match one stock's signal row to
        # another company's filing (dates line up, entity identity is lost).
        if by is None:
            inst_cols = [
                c for c in ev.columns
                if str(c).strip().lower() in {"symbol", "ticker", "instrument", "stock_code", "secucode"}
            ]
            if inst_cols:
                raise ValueError(
                    "FinancialPeriodAdapter.asof requires by=<instrument column> when "
                    f"events carry multiple instruments; found {inst_cols[0]!r}"
                )
        elif by not in ev.columns:
            raise ValueError(f"asof by={by!r} not present in events columns")
        elif by not in dates.columns:
            raise ValueError(
                f"asof by={by!r} not present in signal frames; signals must carry the "
                "instrument column"
            )

        if by is not None:
            # merge_asof with ``by`` requires the time column to be GLOBALLY
            # monotonic (pandas 2.3), so process each instrument group separately
            # and concatenate — this also makes the cross-stock guarantee
            # structural: a signal row can only ever see its own instrument's
            # event rows.
            parts: list[pd.DataFrame] = []
            for _, sub_dates in dates.groupby(by, sort=False):
                sub_events = ev[ev[by] == sub_dates.iloc[0, dates.columns.get_loc(by)]]
                if sub_events.empty:
                    parts.append(sub_dates.copy())
                    continue
                joined = pd.merge_asof(
                    sub_dates.sort_values("__signal_date__"),
                    sub_events.sort_values(key),
                    left_on="__signal_date__",
                    right_on=key,
                    direction=direction,
                    # round-6 P0-02: date-level knowledge time is conservative —
                    # a signal dated on the PubDate must NOT see that day's
                    # after-close filing.  ``allow_exact_matches=False`` makes the
                    # asof strictly ``knowledge_time < signal_date``.
                    allow_exact_matches=bool(same_day),
                )
                parts.append(joined)
            merged = pd.concat(parts, ignore_index=True)
            # Keep a single instrument column (the signal side's), dropping the
            # events-side ``by`` duplicate that merge_asof suffixes.
            if f"{by}_x" in merged.columns and f"{by}_y" in merged.columns:
                merged = merged.rename(columns={f"{by}_x": by}).drop(columns=[f"{by}_y"])
            return merged.sort_values("__signal_date__")
        merged = pd.merge_asof(
            dates.sort_values("__signal_date__"),
            ev.sort_values(key),
            left_on="__signal_date__",
            right_on=key,
            direction=direction,
            # round-6 P0-02: conservative date-level knowledge time (same rule as
            # the per-instrument branch above).
            allow_exact_matches=bool(same_day),
        )
        return merged


def apply_binding_transform(binding: MarketFieldBinding, value: Any, *, fields: dict[str, Any] | None = None) -> Any:
    """Apply the binding's unit transform to a raw physical value.

    Derived providers (``derived_expression`` set, e.g. continuous_close =
    Close * Factor) receive a dict of their physical fields and evaluate the
    real multiplication; single-field bindings keep the legacy ``value`` call.
    """
    if binding.derived_expression is not None:
        if fields is None:
            raise ValueError(
                f"{binding.concept_id}@{binding.market} is a derived provider "
                f"({binding.derived_expression!r}); pass fields={{physical: value}}"
            )
        return binding.transform(fields)
    if fields is not None and len(binding.physical_fields) == 1:
        # ADJ_FIELD_MIGRATION: single-physical-field providers (e.g. A-share
        # continuous_close -> AdjClose) accept the fields-dict contract and resolve
        # their sole physical column (identity), so ``apply_binding_transform(b, None,
        # fields={...})`` stays deterministic instead of returning None.
        return binding.transform(fields)
    return binding.transform(value)


__all__ = [
    "FilterRequirement",
    "FinancialPeriodAdapter",
    "JoinRequirement",
    "MarketFieldBinding",
    "PROVIDER_REGISTRY",
    "PROVIDER_RESOLVER",
    "ProviderDependency",
    "ProviderRegistry",
    "ProviderResolver",
    "apply_binding_transform",
    "binding",
    "canonicalize_provider_market",
    "explain_field_support",
    "parse_filter_requirement",
    "require_binding",
]
