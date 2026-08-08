"""Market capability / provider quality / coverage vocabulary.

Design (per the multi-market plan §8, §45-§46, §49):

- ``MarketCapability`` is a *capability*, not a market name.  Operators declare
  ``required_capabilities``; the capability resolver maps each market to the set
  of capabilities it currently provides.  A future HK market that also has daily
  price limits simply adds a capability set — no operator edit.
- ``ProviderQuality`` distinguishes exact/proxy/sparse/unavailable so that
  "concept exists" != "production-usable".
- ``CoverageClass`` describes historical coverage independent of quality.
- ``MarketStatus`` is the per-operator, per-market verdict produced by the
  capability resolver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MarketCapability(str, Enum):
    DAILY_OHLCV = "DAILY_OHLCV"
    ADJUSTED_PRICE = "ADJUSTED_PRICE"
    DAILY_MARKET_CAP = "DAILY_MARKET_CAP"
    DAILY_SHARES = "DAILY_SHARES"
    FREE_FLOAT_SHARES = "FREE_FLOAT_SHARES"
    DAILY_TURNOVER = "DAILY_TURNOVER"
    INDUSTRY_CLASSIFICATION = "INDUSTRY_CLASSIFICATION"
    TRADABILITY_STATUS = "TRADABILITY_STATUS"
    DAILY_PRICE_LIMITS = "DAILY_PRICE_LIMITS"
    FULL_MINUTE_OHLCV = "FULL_MINUTE_OHLCV"
    TICK_TRADES = "TICK_TRADES"
    L1_QUOTES = "L1_QUOTES"
    L2_ORDERBOOK = "L2_ORDERBOOK"
    INDEX_MEMBERSHIP = "INDEX_MEMBERSHIP"
    INDEX_WEIGHTS = "INDEX_WEIGHTS"
    FINANCIAL_PIT = "FINANCIAL_PIT"
    FINANCIAL_QUARTERLY = "FINANCIAL_QUARTERLY"
    FINANCIAL_TTM = "FINANCIAL_TTM"
    DIVIDEND_ANNOUNCEMENT = "DIVIDEND_ANNOUNCEMENT"
    DIVIDEND_EFFECTIVE_EVENT = "DIVIDEND_EFFECTIVE_EVENT"
    TOP_HOLDERS = "TOP_HOLDERS"
    HOLDER_PLEDGE = "HOLDER_PLEDGE"
    NEWS = "NEWS"
    SECURITY_MASTER = "SECURITY_MASTER"
    EARLY_CLOSE_CALENDAR = "EARLY_CLOSE_CALENDAR"
    FX = "FX"


class ProviderQuality(str, Enum):
    """Quality of a market field binding (ordered from best to worst)."""

    EXACT_NATIVE = "EXACT_NATIVE"
    EXACT_DERIVED = "EXACT_DERIVED"
    SEMANTIC_EQUIVALENT = "SEMANTIC_EQUIVALENT"
    PROXY_RESEARCH = "PROXY_RESEARCH"
    SPARSE = "SPARSE"
    PIT_BLOCKED = "PIT_BLOCKED"
    SOURCE_UNCERTIFIED = "SOURCE_UNCERTIFIED"
    UNAVAILABLE = "UNAVAILABLE"

    @property
    def production_usable(self) -> bool:
        return self in (
            ProviderQuality.EXACT_NATIVE,
            ProviderQuality.EXACT_DERIVED,
        )

    @property
    def research_usable(self) -> bool:
        return self in (
            ProviderQuality.EXACT_NATIVE,
            ProviderQuality.EXACT_DERIVED,
            ProviderQuality.SEMANTIC_EQUIVALENT,
            ProviderQuality.PROXY_RESEARCH,
            ProviderQuality.SPARSE,
        )


class CoverageClass(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    SPARSE = "SPARSE"
    EVENT_ONLY = "EVENT_ONLY"
    CURRENT_ONLY = "CURRENT_ONLY"
    UNKNOWN = "UNKNOWN"


class MarketStatus(str, Enum):
    """Per-operator per-market verdict from the capability resolver."""

    CERTIFIED_NATIVE = "CERTIFIED_NATIVE"
    CERTIFIED_DERIVED = "CERTIFIED_DERIVED"
    PROVIDER_REQUIRED = "PROVIDER_REQUIRED"
    UNSUPPORTED_MARKET_MECHANISM = "UNSUPPORTED_MARKET_MECHANISM"
    PIT_BLOCKED = "PIT_BLOCKED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    UNKNOWN = "UNKNOWN"

    @property
    def is_supported(self) -> bool:
        return self in (
            MarketStatus.CERTIFIED_NATIVE,
            MarketStatus.CERTIFIED_DERIVED,
        )

    @property
    def is_production_supported(self) -> bool:
        return self in (
            MarketStatus.CERTIFIED_NATIVE,
            MarketStatus.CERTIFIED_DERIVED,
        )


# ---------------------------------------------------------------------------
# Per-market capability sets (the current snapshot; sourced from the two COS
# data dictionaries, 2026-08-08).
#
# A-share: daily OHLCV + official HighLimit/LowLimit + turnover + industry +
# tradability(IsSuspend) + full minute bars + index membership & weights +
# financial PIT(PubDate) + effective-only dividend + top holders + pledge.
#   Not present: tick/L2/news/FX/announcement-PIT dividends.
#
# US: daily OHLCV + adjusted price + market cap(shares snapshot, partial) +
# financial PIT(filing_date) + dividend announcement(declaration_date) + news +
# security master + early-close calendar + minute/tick (massive external,
# provider-dependent).  Not present: price limits, turnover, industry (EMPTY),
# tradability (halt too sparse to be a stable IsSuspend), index weights,
# top holders, pledge, FX.
# ---------------------------------------------------------------------------
ASHARE_CAPABILITIES: frozenset[MarketCapability] = frozenset(
    {
        MarketCapability.DAILY_OHLCV,
        MarketCapability.ADJUSTED_PRICE,
        MarketCapability.DAILY_MARKET_CAP,
        MarketCapability.DAILY_SHARES,
        MarketCapability.FREE_FLOAT_SHARES,
        MarketCapability.DAILY_TURNOVER,
        MarketCapability.INDUSTRY_CLASSIFICATION,
        MarketCapability.TRADABILITY_STATUS,
        MarketCapability.DAILY_PRICE_LIMITS,
        MarketCapability.FULL_MINUTE_OHLCV,
        MarketCapability.INDEX_MEMBERSHIP,
        MarketCapability.INDEX_WEIGHTS,
        MarketCapability.FINANCIAL_PIT,
        MarketCapability.FINANCIAL_QUARTERLY,
        MarketCapability.FINANCIAL_TTM,
        MarketCapability.DIVIDEND_EFFECTIVE_EVENT,
        MarketCapability.TOP_HOLDERS,
        MarketCapability.HOLDER_PLEDGE,
        MarketCapability.SECURITY_MASTER,
    }
)

US_CAPABILITIES: frozenset[MarketCapability] = frozenset(
    {
        MarketCapability.DAILY_OHLCV,
        MarketCapability.ADJUSTED_PRICE,
        MarketCapability.DAILY_MARKET_CAP,
        MarketCapability.DAILY_SHARES,
        MarketCapability.FINANCIAL_PIT,
        MarketCapability.FINANCIAL_QUARTERLY,
        MarketCapability.FINANCIAL_TTM,
        MarketCapability.DIVIDEND_ANNOUNCEMENT,
        MarketCapability.NEWS,
        MarketCapability.SECURITY_MASTER,
        MarketCapability.EARLY_CLOSE_CALENDAR,
        MarketCapability.INDEX_MEMBERSHIP,
        # Massive-SIP capabilities are *provider-dependent*: certified only after
        # full coverage/PIT/session validation.  Keep them OUT of the production
        # set until then so minute/tick operators stay SOURCE_UNCERTIFIED.
    }
)

_CAPABILITY_SETS = {
    "ashare": ASHARE_CAPABILITIES,
    "us": US_CAPABILITIES,
}


def capabilities_for(market: str) -> frozenset[MarketCapability]:
    key = str(market).strip().lower()
    if key == "a_share" or key == "cn" or key == "china":
        key = "ashare"
    if key in ("usa",):
        key = "us"
    try:
        return _CAPABILITY_SETS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no capability set for market {market!r}") from exc


@dataclass(frozen=True)
class MarketSupport:
    """A resolved per-market support verdict for one canonical operator.

    Mirrors the ``operator_market_capabilities.json`` row shape.
    """

    canonical: str
    market: str
    status: MarketStatus
    required_capabilities: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()
    provider_quality: ProviderQuality = ProviderQuality.UNAVAILABLE
    coverage: CoverageClass = CoverageClass.UNKNOWN
    depends_on_inputs: bool = False
    notes: str = ""

    @property
    def is_supported(self) -> bool:
        return self.status.is_supported

    @property
    def is_production_supported(self) -> bool:
        return self.status.is_production_supported

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "market": self.market,
            "status": self.status.value,
            "required_capabilities": list(self.required_capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "reason_codes": list(self.reason_codes),
            "providers": list(self.providers),
            "provider_quality": self.provider_quality.value,
            "coverage": self.coverage.value,
            "depends_on_inputs": self.depends_on_inputs,
            "notes": self.notes,
        }


__all__ = [
    "ASHARE_CAPABILITIES",
    "US_CAPABILITIES",
    "CoverageClass",
    "MarketCapability",
    "MarketStatus",
    "MarketSupport",
    "ProviderQuality",
    "capabilities_for",
]
