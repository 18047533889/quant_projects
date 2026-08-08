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
from .context import (
    ASHARE_CONTEXT,
    US_CONTEXT,
    Market,
    MarketContext,
    known_markets,
    market_context,
)
from .instrument import InstrumentKey, qualified_instrument
from .session import (
    ASHARE_SESSION,
    US_SESSION,
    SessionSegment,
    SessionSpec,
    session_for,
)

# Lazy import of the capability resolver (it imports cleaned_operators helpers,
# which are heavy).  Avoid importing it at module scope so the lightweight market
# package stays importable by the field layer without triggering operator load.
def operator_support(*args, **kwargs):
    from .capability_resolver import operator_support as _fn

    return _fn(*args, **kwargs)


def explain_expression_support(*args, **kwargs):
    from .capability_resolver import explain_expression_support as _fn

    return _fn(*args, **kwargs)


def explain_operator_support(*args, **kwargs):
    from .capability_resolver import explain_operator_support as _fn

    return _fn(*args, **kwargs)


def build_search_grammar(*args, **kwargs):
    from .capability_resolver import build_search_grammar as _fn

    return _fn(*args, **kwargs)


def build_market_operator_manifest(*args, **kwargs):
    from .capability_resolver import build_market_operator_manifest as _fn

    return _fn(*args, **kwargs)


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
    "ASHARE_SESSION",
    "US_SESSION",
    "SessionSegment",
    "SessionSpec",
    "session_for",
    "build_market_operator_manifest",
    "build_search_grammar",
    "capabilities_for",
    "explain_expression_support",
    "explain_operator_support",
    "known_markets",
    "market_context",
    "operator_support",
    "qualified_instrument",
]
