# -*- coding: utf-8 -*-
"""R40 #131/#132/#137/#138/#220: FieldSpec flow-semantics matrix, MarketContext
annualization/extra guard, and production semantic-kind gating."""
from __future__ import annotations

import pytest
from fields.spec import FieldSpec, semantic_kind_of_field
from market.context import ASHARE_CONTEXT, MarketContext, US_CONTEXT


class TestFlowSemantics:
    def _spec(self, grain):
        return FieldSpec(name="f", table="t", source_name="f", grain=grain)

    def test_annual_flow_semantics_binding(self):
        spec = self._spec("flow_annual")
        assert spec.flow_semantics == "annual_flow"
        # explicit declaration is also honored and bound by ir.types
        from ir.types import semantic_type_of, SemanticType

        kind = semantic_type_of(flow_semantics="annual_flow")
        assert kind == SemanticType.FINANCIAL_ANNUAL_FLOW

    def test_flow_semantics_full_period_duration_coverage(self):
        cases = {
            "flow_ytd": "cumulative_ytd_flow",
            "flow_annual": "annual_flow",
            "flow_quarter": "single_period_flow",
            "flow_single_period": "single_period_flow",
            "balance": "stock",
            "flow_ttm": "ttm_flow",
            "flow": "single_period_flow",
        }
        for grain, expected in cases.items():
            spec = self._spec(grain)
            assert spec.flow_semantics == expected, f"grain={grain!r} -> {spec.flow_semantics!r}"
        # annual balance sheet is a stock, not a flow
        assert self._spec("balance_annual").flow_semantics == "stock"
        # pure price grain stays untyped (None)
        assert self._spec("instrument_time").flow_semantics is None

    def test_explicit_declaration_wins(self):
        spec = FieldSpec(name="f", table="t", source_name="f",
                         grain="flow_ytd", flow_semantics="annual_flow")
        assert spec.flow_semantics == "annual_flow"


class TestMarketAnnualization:
    def test_annualization_from_real_calendar_not_252(self):
        ctx = MarketContext(
            market="us", currency="USD", timezone="America/New_York",
            calendar_id="US_EQUITY", session_id="US_REGULAR",
            annualization_basis="calendar",
            resolved_trading_days_per_year=248,
        )
        assert ctx.trading_days_per_year == 248
        assert ctx.annualization_factor == 248.0
        assert ctx.annualization_factor != 252.0

    def test_default_252_when_no_calendar_resolved(self):
        assert ASHARE_CONTEXT.trading_days_per_year == 252
        assert US_CONTEXT.annualization_factor == pytest.approx(252.0)


class TestMarketExtraGuard:
    def test_extra_disallows_semantic_keys(self):
        with pytest.raises(ValueError, match="must not shadow semantic fields"):
            MarketContext(
                market="us", currency="USD", timezone="America/New_York",
                calendar_id="US_EQUITY", session_id="US_REGULAR",
                extra={"market": "hidden"},
            )
        with pytest.raises(ValueError, match="calendar_id"):
            MarketContext(
                market="us", currency="USD", timezone="America/New_York",
                calendar_id="US_EQUITY", session_id="US_REGULAR",
                extra={"calendar_id": "SSE"},
            )

    def test_extra_allows_non_semantic_keys(self):
        ctx = MarketContext(
            market="us", currency="USD", timezone="America/New_York",
            calendar_id="US_EQUITY", session_id="US_REGULAR",
            extra={"note": "ok"},
        )
        assert ctx.extra == {"note": "ok"}


class TestProductionSemanticKind:
    def test_production_rejects_name_based_semantic_kind_fallback(self):
        # "universe_mask" is only resolvable via the name map -> production None.
        assert semantic_kind_of_field("universe_mask", production=True) is None
        assert semantic_kind_of_field("group_id", production=True) is None

    def test_research_keeps_name_fallback(self):
        assert semantic_kind_of_field("universe_mask", production=False) == "MaskBool"
        assert semantic_kind_of_field("group_id", production=False) == "GroupKey"

    def test_explicit_semantic_kind_wins_in_production(self):
        spec = FieldSpec(name="v", table="t", source_name="v", semantic_kind="NonNegativeActivity")
        assert semantic_kind_of_field(spec, production=True) == "NonNegativeActivity"
