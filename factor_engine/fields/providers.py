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

from market.capabilities import (
    CoverageClass,
    MarketStatus,
    MarketSupport,
    ProviderQuality,
)
from market.context import ASHARE_CONTEXT, US_CONTEXT, MarketContext

from .units_v2 import (
    CNY,
    CNY_PER_SHARE,
    RATIO,
    SHARES,
    USD,
    USD_PER_SHARE,
    UnitSpec,
)

Transform = Callable[[Any], Any]


def _identity(x: Any) -> Any:
    return x


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


# ---------------------------------------------------------------------------
# Structured filter requirement (P1-14): ``required_filters`` was a tuple of
# strings ("IndustrySource", "currency=USD", "timeframe") that could not be
# executed or hashed.  It is now a machine-executable contract.
# ---------------------------------------------------------------------------
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
            "required_filters": [f.to_dict() for f in self.required_filters],
            "source_certified": self.source_certified,
            "derived_expression": self.derived_expression,
            "coverage_gate": self.coverage_gate,
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


PROVIDER_REGISTRY = ProviderRegistry()

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
    dataset="ashare_stock_daily", physical=("StockDailyBar.Return",),
    quality=_NATIVE, coverage=_FULL,
    source_unit=UnitSpec.ratio(scale=0.0001), canonical_unit=RATIO,
    transform=_times(0.0001), transform_description="x * 0.0001 (bp -> decimal)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="Return = (Close/PreClose - 1) * 10000, verified error=0",
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
for _raw, _phys in (
    ("raw_open", "Open"), ("raw_high", "High"), ("raw_low", "Low"),
    ("raw_close", "Close"), ("raw_pre_close", "PreClose"), ("raw_vwap", "Vwap"),
):
    _b(
        _raw, "ashare", f"ashare_raw_{_phys.lower()}",
        dataset="ashare_stock_daily", physical=(f"StockDailyBar.{_phys}",),
        quality=_NATIVE, coverage=_FULL,
        source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
        transform=_identity, temporal_model="exact_daily", available_at="local_close",
        source_certified=True,
    )
for _raw, _phys in (
    ("raw_open", "Open"), ("raw_high", "High"), ("raw_low", "Low"),
    ("raw_close", "Close"), ("raw_pre_close", "PreClose"), ("raw_vwap", "VWAP"),
):
    _b(
        _raw, "us", f"us_raw_{_phys.lower()}",
        dataset="us_stock_daily", physical=(f"StockDailyBar.{_phys}",),
        quality=_NATIVE, coverage=_FULL,
        source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE,
        transform=_identity, temporal_model="exact_daily", available_at="local_close",
        source_certified=True,
    )

# --- continuous (backward-adjusted) prices --------------------------------
_b(
    "continuous_close", "ashare", "ashare_continuous_close",
    dataset="ashare_stock_daily", physical=("StockDailyBar.Close", "StockDailyBar.Factor"),
    quality=_DERIVED, coverage=_FULL,
    source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
    transform=_mul_two("StockDailyBar.Close", "StockDailyBar.Factor"),
    transform_description="Close * Factor (backward multiplier, verified)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
    derived_expression="StockDailyBar.Close * StockDailyBar.Factor",
    notes="Factor is a backward cumulative multiplier; adjusted = raw * Factor",
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
    notes="clamp flag (adj_factor > 1e6) -> use returns for long windows",
)

# P0-010: continuous_open/high/low/vwap — the raw OHLC siblings of
# continuous_close multiplied by the SAME backward adjustment factor.  Before
# these bindings, cross-day open/high/low/vwap trend operators could only pull
# raw (unadjusted) OHLC, so long-window price path factors were split/distortion
# contaminated.
for _canon, _bar in (
    ("continuous_open", "Open"),
    ("continuous_high", "High"),
    ("continuous_low", "Low"),
    ("continuous_vwap", "Vwap"),
):
    _b(
        _canon, "ashare", f"ashare_{_canon}",
        dataset="ashare_stock_daily",
        physical=(f"StockDailyBar.{_bar}", "StockDailyBar.Factor"),
        quality=_DERIVED, coverage=_FULL,
        source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
        transform=_mul_two(f"StockDailyBar.{_bar}", "StockDailyBar.Factor"),
        transform_description=f"{_bar} * Factor (backward multiplier)",
        temporal_model="exact_daily", available_at="local_close",
        source_certified=True,
        derived_expression=f"StockDailyBar.{_bar} * StockDailyBar.Factor",
        notes="same backward Factor as continuous_close",
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
        notes="same backward AdjFactor as continuous_close",
    )
for _raw, _phys in (("raw_volume_shares", "Volume"),):
    _b(
        _raw, "ashare", "ashare_raw_volume",
        dataset="ashare_stock_daily", physical=(f"StockDailyBar.{_phys}",),
        quality=_NATIVE, coverage=_FULL,
        source_unit=SHARES, canonical_unit=SHARES,
        transform=_identity, transform_description="identity (raw shares; no adjustment applied)",
        temporal_model="exact_daily", available_at="local_close",
        source_certified=True, notes="generic volume is raw shares; NEVER divided by any adjustment factor (split-adjusted volume is a separate concept)",
    )
    _b(
        _raw, "us", "us_raw_volume",
        dataset="us_stock_daily", physical=(f"StockDailyBar.{_phys}",),
        quality=_NATIVE, coverage=_FULL,
        source_unit=SHARES, canonical_unit=SHARES,
        transform=_identity, temporal_model="exact_daily", available_at="local_close",
        source_certified=True,
    )

# --- amount_local -----------------------------------------------------------
_b(
    "amount_local", "ashare", "ashare_amount",
    dataset="ashare_stock_daily", physical=("StockDailyBar.Amount",),
    quality=_NATIVE, coverage=_FULL, source_unit=CNY, canonical_unit=CNY,
    transform=_identity, temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="Amount ~= Vwap * Volume (rel err ~2e-5)",
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
    dataset="us_stock_shares_snapshot",
    physical=("TickerSharesSnapshot.weighted_shares_outstanding", "StockDailyBar.Close"),
    quality=_DERIVED, coverage=CoverageClass.PARTIAL, source_unit=USD, canonical_unit=USD,
    transform=_mul_two("TickerSharesSnapshot.weighted_shares_outstanding", "StockDailyBar.Close"),
    transform_description="Close * weighted_shares_outstanding",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True,
    derived_expression="TickerSharesSnapshot.weighted_shares_outstanding * StockDailyBar.Close",
    coverage_gate=0.42,
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
for _concept, _phys, _prov in (
    ("upper_price_limit", "HighLimit", "ashare_high_limit"),
    ("lower_price_limit", "LowLimit", "ashare_low_limit"),
):
    _b(
        _concept, "ashare", _prov,
        dataset="ashare_stock_daily", physical=(f"StockDailyBar.{_phys}",),
        quality=_NATIVE, coverage=_FULL,
        source_unit=CNY_PER_SHARE, canonical_unit=CNY_PER_SHARE,
        transform=_identity, temporal_model="exact_daily", available_at="local_open",
        source_certified=True, notes="RAW official limit price; never adjusted",
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
    dataset="ashare_stock_daily", physical=("StockDailyBar.IsSuspend",),
    quality=_NATIVE, coverage=_FULL, source_unit=UnitSpec(dimension="boolean"),
    canonical_unit=UnitSpec(dimension="boolean"),
    transform=_not_bool,
    transform_description="NOT IsSuspend (not_suspended_state; known-suspended->0, known-not-suspended->1, NaN->NaN; does NOT claim listing membership)",
    temporal_model="exact_daily", available_at="local_open",
    source_certified=True, notes="IsSuspend lives in StockDailyBar; transform negates it preserving NaN (unknown suspension stays unknown); no listing/public-status mask is included",
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
    required_filters=(
        FilterRequirement(field="universe_daily", operator="present", required=True),
        FilterRequirement(field="StockDailyBar", operator="present", required=True),
    ),
    notes="real boolean from CS list type + finite DailyBar close (universe_daily membership declared via required_filters); is_ticker_halt is far too sparse to be an IsSuspend equivalent",
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
    dataset="us_stock_index_components", physical=("StockIndicesComponents.IndexName",),
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
}
_FIN_US = {
    "operating_revenue": ("StockIncome.revenue", "us_stock_income"),
    "net_profit": ("StockIncome.net_income_loss_attributable_common_shareholders", "us_stock_income"),
    "operating_cash_flow": ("StockCashFlow.net_cash_from_operating_activities", "us_stock_cashflow"),
    "total_assets": ("StockBalance.total_assets", "us_stock_balance"),
    "total_liabilities": ("StockBalance.total_liabilities", "us_stock_balance"),
    "equity": ("StockBalance.total_equity", "us_stock_balance"),
}
for _concept, (_phys_a, _ds_a) in _FIN_A.items():
    _b(
        _concept, "ashare", f"ashare_{_concept}",
        dataset=_ds_a, physical=(_phys_a,),
        quality=_NATIVE, coverage=_FULL, source_unit=CNY, canonical_unit=CNY,
        transform=_identity, temporal_model="financial_pit",
        knowledge_time="PubDate", available_at="filing",
        source_certified=True, notes="asof(PubDate); flow fields are cumulative YTD",
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
    from market.context import market_context

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
    def filter_timeframe(frame: Any, timeframe: str) -> Any:
        """US-only: keep rows whose ``timeframe`` equals ``timeframe``."""
        if "timeframe" in getattr(frame, "columns", ()):
            return frame[frame["timeframe"] == timeframe]
        return frame

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
    return binding.transform(value)


__all__ = [
    "FilterRequirement",
    "FinancialPeriodAdapter",
    "MarketFieldBinding",
    "PROVIDER_REGISTRY",
    "ProviderRegistry",
    "apply_binding_transform",
    "binding",
    "explain_field_support",
    "parse_filter_requirement",
    "require_binding",
]
