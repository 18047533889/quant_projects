"""Market-aware semantics for FactorEngine (A股 + US).

Layering (per FACTOR_ENGINE_MULTI_MARKET_PLAN.md):

    Physical Data
        ↓  Market Adapter / Provider
    Canonical Field Concept
        ↓  Generic Operator
    Market Capability + Expression Gate

This package owns the market dimension only: ``MarketContext``, market
capabilities, instrument-key namespace, and the capability resolver.  It never
depends on operator internals so it can be imported by the field layer, the
DSL/IR, and the operator registry without cycles.
"""
from __future__ import annotations

from .capabilities import (
    ASHARE_CAPABILITIES,
    US_CAPABILITIES,
    CoverageClass,
    MarketCapability,
    MarketStatus,
    MarketSupport,
    ProviderQuality,
    capabilities_for,
)
from .context import ASHARE_CONTEXT, US_CONTEXT, Market, MarketContext
from .instrument import InstrumentKey, qualified_instrument

__all__ = [
    "ASHARE_CAPABILITIES",
    "ASHARE_CONTEXT",
    "US_CAPABILITIES",
    "US_CONTEXT",
    "CoverageClass",
    "InstrumentKey",
    "Market",
    "MarketCapability",
    "MarketContext",
    "MarketStatus",
    "MarketSupport",
    "ProviderQuality",
    "capabilities_for",
    "qualified_instrument",
]
