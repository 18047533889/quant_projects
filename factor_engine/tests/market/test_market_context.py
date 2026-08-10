# -*- coding: utf-8 -*-
"""MarketContext + InstrumentKey namespace tests."""
from __future__ import annotations

import pytest

from market import (
    ASHARE_CONTEXT,
    US_CONTEXT,
    InstrumentKey,
    MarketContext,
    capabilities_for,
    market_context,
    qualified_instrument,
)
from market.capabilities import (
    MarketCapability,
    MarketStatus,
    ProviderQuality,
    MarketSupport,
)


def test_market_contexts_have_correct_contracts() -> None:
    assert ASHARE_CONTEXT.market == "ashare"
    assert ASHARE_CONTEXT.currency == "CNY"
    assert ASHARE_CONTEXT.timezone == "Asia/Shanghai"
    assert US_CONTEXT.market == "us"
    assert US_CONTEXT.currency == "USD"
    assert US_CONTEXT.timezone == "America/New_York"
    assert market_context("ashare") is ASHARE_CONTEXT
    assert market_context("us") is US_CONTEXT


def test_market_context_rejects_unknown_market() -> None:
    with pytest.raises(ValueError):
        MarketContext(
            market="hk", currency="HKD", timezone="Asia/Hong_Kong",
            calendar_id="HK", session_id="HK",
        )


def test_research_profile_opens_gates() -> None:
    research = US_CONTEXT.as_research()
    assert research.provider_profile == "research"
    assert research.allow_proxy is True
    assert research.allow_sparse is True
    # production stays closed
    assert US_CONTEXT.allow_proxy is False
    assert US_CONTEXT.allow_sparse is False
    # R17-050: as_research must NOT auto-open effective-time-only (non-PIT ex-date
    # semantics is a dangerous explicit flag, false even in research).
    assert research.allow_effective_time_only is False
    assert US_CONTEXT.with_non_pit_effective_time().allow_effective_time_only is True


def test_capabilities_snapshot() -> None:
    a = capabilities_for("ashare")
    u = capabilities_for("us")
    assert MarketCapability.DAILY_PRICE_LIMITS in a
    assert MarketCapability.DAILY_PRICE_LIMITS not in u
    assert MarketCapability.NEWS in u
    assert MarketCapability.NEWS not in a
    assert MarketCapability.INDUSTRY_CLASSIFICATION in a
    assert MarketCapability.INDUSTRY_CLASSIFICATION not in u
    assert MarketCapability.FINANCIAL_PIT in a
    assert MarketCapability.FINANCIAL_PIT in u


def test_provider_quality_ordering() -> None:
    assert ProviderQuality.EXACT_NATIVE.production_usable
    assert ProviderQuality.EXACT_DERIVED.production_usable
    assert not ProviderQuality.PROXY_RESEARCH.production_usable
    assert ProviderQuality.PROXY_RESEARCH.research_usable
    assert ProviderQuality.UNAVAILABLE.research_usable is False


def test_instrument_key_namespace() -> None:
    a = InstrumentKey(market="ashare", instrument="000001.SZ")
    u = InstrumentKey(market="us", instrument="000001.SZ")  # same code, different market
    assert a != u
    assert str(a) == "ashare:000001.SZ"
    assert str(u) == "us:000001.SZ"
    assert InstrumentKey.parse("us:AAPL") == InstrumentKey("us", "AAPL")
    assert qualified_instrument("us", "BRK.B") == "us:BRK.B"
    with pytest.raises(ValueError):
        InstrumentKey.parse("AAPL")  # missing market namespace


def test_market_support_serialization() -> None:
    support = MarketSupport(
        canonical="ashare_limit_up_touch",
        market="us",
        status=MarketStatus.UNSUPPORTED_MARKET_MECHANISM,
        reason_codes=("MARKET_MECHANISM",),
        required_capabilities=("DAILY_PRICE_LIMITS",),
    )
    d = support.to_dict()
    assert d["status"] == "UNSUPPORTED_MARKET_MECHANISM"
    assert d["reason_codes"] == ["MARKET_MECHANISM"]
    assert support.is_supported is False
