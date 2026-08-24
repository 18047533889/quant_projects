# -*- coding: utf-8 -*-
"""R24-132..134 + R24-205..209: the semantic identity digest changes with every
dimension that changes the economic definition, and excludes debug/cost/tags."""
from __future__ import annotations

import pytest

from factor_engine.ir.types import (
    PeriodDuration,
    SemanticIdentityDigest,
    SemanticType,
    semantic_type_of,
)


def _digest(**attrs):
    return SemanticIdentityDigest.from_attrs(attrs).value


def test_same_text_different_market_differs() -> None:
    # R24-205: fin_growth(revenue) under A YTD vs US quarterly vs US TTM differ.
    assert _digest(market="ashare", flow_semantics="cumulative_ytd_flow") != _digest(
        market="us", flow_semantics="quarterly"
    )


def test_same_text_different_universe_differs() -> None:
    # R24-206: rank(roe) under CSI300 vs All-A differs.
    assert _digest(universe_id="csi300") != _digest(universe_id="all_a")


def test_same_text_different_availability_policy_differs() -> None:
    # R24-207: PubDate next-session vs exact same-day differ.
    assert _digest(availability_policy="next_trading_day") != _digest(
        availability_policy="same_day"
    )


def test_same_text_different_revision_policy_differs() -> None:
    # R24-208: first_available vs latest_available_asof differ.
    assert _digest(revision_policy="first_available") != _digest(
        revision_policy="latest_available_asof"
    )


def test_same_text_different_group_fallback_differs() -> None:
    # R24-209: missing group → nan vs global demean differ.
    assert _digest(group_fallback="nan") != _digest(group_fallback="global_demean")


def test_debug_dimensions_excluded() -> None:
    a = _digest(market="us", description="debug note", cost_estimate=1.0)
    b = _digest(market="us")
    assert a == b


def test_period_duration_members() -> None:
    assert {p.value for p in PeriodDuration} >= {
        "quarter", "half_year", "annual", "ttm", "ytd", "irregular", "unknown",
    }


def test_new_semantic_types_exist() -> None:
    assert SemanticType.FINANCIAL_ANNUAL_FLOW.value == "FinancialAnnualFlow"
    assert SemanticType.OUT_OF_COVERAGE.value == "OutOfCoverage"
    assert semantic_type_of(flow_semantics="annual_flow") == SemanticType.FINANCIAL_ANNUAL_FLOW
