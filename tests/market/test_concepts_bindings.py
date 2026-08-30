# -*- coding: utf-8 -*-
"""Canonical concepts + per-market provider bindings."""
from __future__ import annotations

import pytest

from factor_engine.fields import MULTI_MARKET_FIELD_REGISTRY
from factor_engine.fields.concepts import concept_alias_map, get_concept, list_concepts, require_concept
from factor_engine.fields.providers import (
    PROVIDER_REGISTRY,
    explain_field_support,
    require_binding,
)
from factor_engine.market import CoverageClass, MarketStatus, ProviderQuality


def test_concept_alias_map() -> None:
    aliases = concept_alias_map()
    assert aliases["close"] == "continuous_close"
    assert aliases["ret"] == "return_decimal"
    assert aliases["volume"] == "continuous_volume_shares"
    assert aliases["roe"] == "roe_decimal"
    assert aliases["high_limit"] == "upper_price_limit"


def test_core_concepts_registered() -> None:
    names = {c.concept_id for c in list_concepts()}
    for concept in (
        "return_decimal", "raw_close", "continuous_close", "raw_volume_shares",
        "amount_local", "market_cap_local", "turnover_ratio_decimal",
        "upper_price_limit", "lower_price_limit", "roe_decimal",
        "industry_group", "tradability_state", "index_member", "index_weight",
        "cash_dividend_per_share", "operating_revenue", "net_profit",
    ):
        assert concept in names, concept


def test_return_decimal_both_markets_exact() -> None:
    a = require_binding("return_decimal", "ashare")
    u = require_binding("return_decimal", "us")
    assert a.transform(200.0) == pytest.approx(0.02)  # 200 bp -> 0.02
    assert u.transform(0.02) == pytest.approx(0.02)   # US Ret identity
    assert a.transform_description.startswith("x * 0.0001")
    assert u.transform_description.startswith("identity")


def test_volume_never_scale_by_factor() -> None:
    # ADJ_FIELD_MIGRATION: A-share raw_volume_shares is DISABLED (raw vendor
    # volume is not a mineable factor input); the adjusted share count is
    # continuous_volume_shares = Volume / Factor.  US raw_volume_shares stays.
    vol = require_binding("raw_volume_shares", "us")
    assert vol.transform_description.startswith("identity")


def test_continuous_close_backward_multiplier() -> None:
    a = require_binding("continuous_close", "ashare")
    u = require_binding("continuous_close", "us")
    # ADJ_FIELD_MIGRATION: A-share reads the precomputed AdjClose (identity);
    # US keeps the Close * AdjFactor derived multiplication.
    assert "AdjClose" in a.transform_description
    assert "Close * AdjFactor" in u.transform_description


def test_ashare_raw_price_concepts_disabled() -> None:
    # ADJ_FIELD_MIGRATION: A-share raw unadjusted OHLC/volume providers are
    # DISABLED — raw price-volume must not reach the factor surface.
    from factor_engine.market import MarketStatus

    for concept in ("raw_close", "raw_open", "raw_vwap", "raw_volume_shares"):
        a = explain_field_support(concept, "ashare")
        assert a.status == MarketStatus.UNSUPPORTED_MARKET_MECHANISM, concept
        u = explain_field_support(concept, "us")
        assert u.status != MarketStatus.UNSUPPORTED_MARKET_MECHANISM, concept


def test_ashare_continuous_reads_adj_dataset() -> None:
    # ADJ_FIELD_MIGRATION: A-share continuous_* bindings point at the adj table.
    for concept in ("continuous_close", "continuous_open", "continuous_vwap",
                    "continuous_volume_shares", "return_decimal", "amount_local"):
        b = require_binding(concept, "ashare")
        assert b.dataset == "ashare_stock_daily_adj", (concept, b.dataset)


def test_price_limits_ashare_only() -> None:
    a = explain_field_support("upper_price_limit", "ashare")
    u = explain_field_support("upper_price_limit", "us")
    assert a.status == MarketStatus.CERTIFIED_NATIVE
    assert u.status == MarketStatus.UNSUPPORTED_MARKET_MECHANISM


def test_industry_us_provider_required() -> None:
    u = explain_field_support("industry_group", "us")
    assert u.status == MarketStatus.UNSUPPORTED_MARKET_MECHANISM


def test_dividend_pit_blocked_ashare_allowed_research() -> None:
    from factor_engine.market.context import ASHARE_CONTEXT

    a_prod = explain_field_support("cash_dividend_per_share", "ashare")
    assert a_prod.status == MarketStatus.PIT_BLOCKED
    # R17-050: ``as_research()`` does NOT open effective-time-only — a bare
    # research profile still leaves A-share ex-date-only dividends PIT_BLOCKED.
    a_research = explain_field_support(
        "cash_dividend_per_share", "ashare",
        context=ASHARE_CONTEXT.as_research(),
    )
    assert a_research.status == MarketStatus.PIT_BLOCKED
    # Only the EXPLICIT dangerous flag opts into effective-time-only semantics.
    a_explicit = explain_field_support(
        "cash_dividend_per_share", "ashare",
        context=ASHARE_CONTEXT.with_non_pit_effective_time(),
    )
    assert a_explicit.status == MarketStatus.RESEARCH_ONLY

    # US dividend: EXACT_NATIVE quality but PARTIAL coverage (0.51% null,
    # ~12.2% non-USD excluded) -> CERTIFIED_PARTIAL, not full-market native.
    us = explain_field_support("cash_dividend_per_share", "us")
    assert us.status == MarketStatus.CERTIFIED_PARTIAL


def test_market_cap_us_derived_partial() -> None:
    # US market cap is derived from a ~42%-coverage shares snapshot: quality is
    # EXACT_DERIVED but coverage is PARTIAL, so it is CERTIFIED_PARTIAL (never
    # silently treated as a whole-market CERTIFIED_DERIVED in production).
    us = explain_field_support("market_cap_local", "us")
    assert us.status == MarketStatus.CERTIFIED_PARTIAL
    assert us.provider_quality == ProviderQuality.EXACT_DERIVED
    assert us.coverage == CoverageClass.PARTIAL
    assert not us.is_production_supported
    a = explain_field_support("market_cap_local", "ashare")
    assert a.status == MarketStatus.CERTIFIED_NATIVE
    assert a.is_production_supported


def test_multi_market_registry_isolates_table_collisions() -> None:
    # Same canonical name, different physical tables per market.
    a_spec = MULTI_MARKET_FIELD_REGISTRY.require_field("ashare", "market_cap")
    us_spec = MULTI_MARKET_FIELD_REGISTRY.require_field("us", "market_cap")
    assert a_spec.table == "StockValuationDaily"
    assert a_spec.source_name == "MarketCap"
    assert us_spec.source_name == "market_cap"
    assert a_spec.unit == "CNY"
    assert us_spec.unit == "USD"
