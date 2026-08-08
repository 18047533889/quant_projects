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


def _mul_factor(x: Any) -> Any:
    """continuous_close = raw_close * adjustment factor (two-column input)."""
    return x


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
    required_filters: tuple[str, ...] = ()
    source_certified: bool = False
    notes: str = ""
    transform_description: str = "identity"

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
            "required_filters": list(self.required_filters),
            "source_certified": self.source_certified,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------
class ProviderRegistry:
    """Deterministic (concept_id, market) -> binding registry."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str], MarketFieldBinding] = {}

    def register(self, binding: MarketFieldBinding, *, replace: bool = False) -> None:
        key = (binding.concept_id.strip(), binding.market.strip().lower())
        if key in self._bindings and not replace:
            raise ValueError(f"binding already registered: {binding.concept_id}@{binding.market}")
        self._bindings[key] = binding

    def binding(self, concept_id: str, market: str) -> MarketFieldBinding | None:
        return self._bindings.get((concept_id.strip(), market.strip().lower()))

    def require_binding(self, concept_id: str, market: str) -> MarketFieldBinding:
        result = self.binding(concept_id, market)
        if result is None:
            raise KeyError(f"no binding for concept {concept_id!r} @ {market!r}")
        return result

    def for_market(self, market: str) -> tuple[MarketFieldBinding, ...]:
        m = market.strip().lower()
        return tuple(
            sorted(
                (b for b in self._bindings.values() if b.market == m),
                key=lambda b: b.concept_id,
            )
        )

    def for_concept(self, concept_id: str) -> tuple[MarketFieldBinding, ...]:
        c = concept_id.strip()
        return tuple(
            sorted(
                (b for b in self._bindings.values() if b.concept_id == c),
                key=lambda b: b.market,
            )
        )

    def concepts(self) -> tuple[str, ...]:
        return tuple(sorted({k[0] for k in self._bindings}))

    def markets(self) -> tuple[str, ...]:
        return tuple(sorted({k[1] for k in self._bindings}))

    def to_dict(self) -> dict[str, Any]:
        return {"bindings": [b.to_dict() for b in self._bindings.values()]}


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
        required_filters=tuple(required_filters),
        source_certified=source_certified,
        notes=notes,
        transform_description=transform_description,
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
    transform=_mul_factor, transform_description="Close * Factor (backward multiplier, verified)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="Factor is a backward cumulative multiplier; adjusted = raw * Factor",
)
_b(
    "continuous_close", "us", "us_continuous_close",
    dataset="us_stock_daily", physical=("StockDailyBar.Close", "StockDailyBar.AdjFactor"),
    quality=_DERIVED, coverage=_FULL,
    source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE,
    transform=_mul_factor, transform_description="Close * AdjFactor (backward multiplier)",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="clamp flag (adj_factor > 1e6) -> use returns for long windows",
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
    transform=_mul_factor, transform_description="Close * weighted_shares_outstanding",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=True, notes="~42% coverage of StockDailyBar tickers; NOT the X0 sparse market_cap",
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
    transform=_identity, transform_description="not IsSuspend (tradable = listed AND not suspended)",
    temporal_model="exact_daily", available_at="local_open",
    source_certified=True, notes="IsSuspend lives in StockDailyBar",
)
_b(
    "tradability_state", "us", "us_tradability_universe",
    dataset="us_stock_list", physical=("StockList.type",),
    quality=ProviderQuality.SEMANTIC_EQUIVALENT, coverage=CoverageClass.PARTIAL,
    source_unit=UnitSpec(dimension="boolean"), canonical_unit=UnitSpec(dimension="boolean"),
    transform=_identity, transform_description="CS universe + valid price/volume",
    temporal_model="exact_daily", available_at="local_close",
    source_certified=False, notes="is_ticker_halt is extremely sparse; cannot serve as IsSuspend equivalent",
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
    dataset="us_stock_dividend", physical=("StockDividend.cash_amount",),
    quality=_NATIVE, coverage=_PARTIAL,
    source_unit=USD_PER_SHARE, canonical_unit=USD_PER_SHARE,
    transform=_identity, temporal_model="financial_pit",
    knowledge_time="declaration_date", available_at="declaration",
    required_filters=("currency=USD",),
    source_certified=True,
    notes="declaration_date PIT (0.51% null); ~12.2% non-USD excluded (no COS FX table); ex_date may be future",
)

# --- financial statements (amounts, local currency) ---------------------------
_FIN_A = {
    "operating_revenue": ("StockIncome.OperatingRevenue",),
    "net_profit": ("StockIncome.NetProfit",),
    "operating_cash_flow": ("StockCashFlow.NetOperateCashFlow",),
    "total_assets": ("StockBalance.TotalAssets",),
    "total_liabilities": ("StockBalance.TotalLiability",),
    "equity": ("StockBalance.EquitiesParentCompanyOwners",),
}
_FIN_US = {
    "operating_revenue": ("StockIncome.revenue",),
    "net_profit": ("StockIncome.net_income_loss_attributable_common_shareholders",),
    "operating_cash_flow": ("StockCashFlow.net_cash_from_operating_activities",),
    "total_assets": ("StockBalance.total_assets",),
    "total_liabilities": ("StockBalance.total_liabilities",),
    "equity": ("StockBalance.total_equity",),
}
for _concept, _phys_a in _FIN_A.items():
    _b(
        _concept, "ashare", f"ashare_{_concept}",
        dataset="ashare_stock_daily", physical=_phys_a,
        quality=_NATIVE, coverage=_FULL, source_unit=CNY, canonical_unit=CNY,
        transform=_identity, temporal_model="financial_pit",
        knowledge_time="PubDate", available_at="filing",
        source_certified=True, notes="asof(PubDate); flow fields are cumulative YTD",
    )
for _concept, _phys_us in _FIN_US.items():
    _b(
        _concept, "us", f"us_{_concept}",
        dataset="us_stock_daily", physical=_phys_us,
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
    return ASHARE_CONTEXT if market == "ashare" else US_CONTEXT


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
    ) -> Any:
        """As-of merge of event rows onto signal dates.

        The ``on`` column defaults to the market knowledge-time column
        (ashare=PubDate, us=filing_date).  Look-ahead is impossible because the
        merge only uses rows with knowledge_time <= signal_date.
        """
        import pandas as pd

        key = on or ("PubDate" if market == "ashare" else "filing_date")
        ev = events.copy()
        if pd.api.types.is_datetime64_any_dtype(ev[key].dtype) is False:
            ev[key] = pd.to_datetime(ev[key])
        dates = pd.Series(pd.to_datetime(signal_dates), name="__signal_date__")
        ev = ev.sort_values(key)
        merged = pd.merge_asof(
            dates.to_frame(),
            ev,
            left_on="__signal_date__",
            right_on=key,
            direction=direction,
        )
        return merged


def apply_binding_transform(binding: MarketFieldBinding, value: Any) -> Any:
    """Apply the binding's unit transform to a raw physical value."""
    return binding.transform(value)


__all__ = [
    "FinancialPeriodAdapter",
    "MarketFieldBinding",
    "PROVIDER_REGISTRY",
    "ProviderRegistry",
    "apply_binding_transform",
    "binding",
    "explain_field_support",
    "require_binding",
]
