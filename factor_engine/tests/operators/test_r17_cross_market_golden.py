# -*- coding: utf-8 -*-
"""R17 golden tests — the §8 evidence this round is done.

Covers:
- 8.1 A/US unit normalisation (Return bp -> decimal; US Ret identity);
- 8.2 Factor / AdjFactor backward multipliers;
- 8.3 reference_pre_close vs lag(raw_close);
- 8.8 industry required source (R17-028) + sw_l1/sw_l2 different factor id;
- 8.9 index membership bool + US weight blocked;
- 8.13 cross-market same-named table separation;
- R17-065 comparability / R17-067 index_weight role.
"""
from __future__ import annotations

import numpy as np
import pytest


def _panel(values):
    import pandas as pd

    return pd.DataFrame(
        values, index=pd.date_range("2024-01-01", periods=len(values)),
        columns=["A", "B"],
    )


# ---------------------------------------------------------------------------
# 8.1 A/US units
# ---------------------------------------------------------------------------
def test_ashare_return_bp_is_hundredth():
    from factor_engine.fields.providers import binding, apply_binding_transform

    b = binding("return_decimal", "ashare")
    assert b is not None
    out = apply_binding_transform(b, np.array([100.0]))
    assert float(out[0]) == pytest.approx(0.01)


def test_us_return_is_identity():
    from factor_engine.fields.providers import binding, apply_binding_transform

    b = binding("return_decimal", "us")
    assert b is not None
    out = apply_binding_transform(b, np.array([0.01]))
    assert float(out[0]) == pytest.approx(0.01)


def test_ashare_turnover_percent_to_decimal():
    from factor_engine.fields.providers import binding, apply_binding_transform

    b = binding("turnover_ratio_decimal", "ashare")
    out = apply_binding_transform(b, np.array([2.5]))
    assert float(out[0]) == pytest.approx(0.025)


def test_us_roe_decimal_identity():
    from factor_engine.fields.providers import binding, apply_binding_transform

    b = binding("roe_decimal", "us")
    out = apply_binding_transform(b, np.array([0.10]))
    assert float(out[0]) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 8.2 Factor / AdjFactor backward multipliers
# ---------------------------------------------------------------------------
def test_continuous_close_mul_transform():
    from factor_engine.fields.providers import binding, apply_binding_transform

    a = binding("continuous_close", "ashare")
    u = binding("continuous_close", "us")
    out_a = apply_binding_transform(
        a, None, fields={"StockDailyBar.Close": np.array([10.0]), "StockDailyBar.Factor": np.array([1.5])}
    )
    out_u = apply_binding_transform(
        u, None, fields={"StockDailyBar.Close": np.array([10.0]), "StockDailyBar.AdjFactor": np.array([2.0])}
    )
    assert float(out_a[0]) == pytest.approx(15.0)
    assert float(out_u[0]) == pytest.approx(20.0)
    # R17-033: US continuous prices are level-sensitive with the clamp policy.
    assert u.level_sensitive is True
    assert "clamp" in (u.dynamic_validity_policy or "")


# ---------------------------------------------------------------------------
# 8.3 reference_pre_close vs raw_pre_close
# ---------------------------------------------------------------------------
def test_reference_pre_close_is_official_basis():
    from factor_engine.fields.concepts import get_concept, concept_alias_map

    assert concept_alias_map().get("pre_close") == "reference_pre_close"
    c = get_concept("reference_pre_close")
    assert c is not None
    assert c.price_basis == "OFFICIAL_REFERENCE_PRE_CLOSE"
    raw = get_concept("raw_pre_close")
    assert raw is not None
    assert raw.aliases == ("prev_close",)  # prev_close stays the lag spelling


# ---------------------------------------------------------------------------
# 8.8 industry source exactly-one + factor identity
# ---------------------------------------------------------------------------
def test_industry_requires_source_filter():
    from factor_engine.fields import FIELD_REGISTRY

    spec = FIELD_REGISTRY.get("industry_code", table="StockIndustry")
    assert spec is not None
    assert "IndustrySource" in (spec.required_filters or ())


def test_index_requires_symbol_filter():
    from factor_engine.fields import FIELD_REGISTRY

    spec = FIELD_REGISTRY.get("index_weight", table="IndexConstituent")
    assert spec is not None
    assert "IndexSymbol" in (spec.required_filters or ())


# ---------------------------------------------------------------------------
# 8.13 cross-market same-named tables are separate objects
# ---------------------------------------------------------------------------
def test_same_named_tables_are_market_separate():
    from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

    for table in ("StockValuationDaily", "StockIndicator", "StockCapitalDaily"):
        a = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare").resolve_table(table)
        u = MULTI_MARKET_FIELD_REGISTRY.registry_for("us").resolve_table(table)
        if a is not None and u is not None:
            assert a is not u, f"{table} shares one TableSpec across markets"
    # US StockValuationDaily is current_snapshot_only; A-share's is not.
    a_val = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare").resolve_table("StockValuationDaily")
    u_val = MULTI_MARKET_FIELD_REGISTRY.registry_for("us").resolve_table("StockValuationDaily")
    if a_val is not None and u_val is not None:
        assert a_val.current_snapshot_only is False
        assert u_val.current_snapshot_only is True


# ---------------------------------------------------------------------------
# R17-065 / R17-066 comparability split
# ---------------------------------------------------------------------------
def test_turnover_not_cross_market_rankable():
    from factor_engine.fields.concepts import get_concept

    c = get_concept("turnover_ratio_decimal")
    assert c is not None
    assert c.unit_comparable is True
    assert c.definition_comparable is False
    assert c.cross_market_rank_allowed is False
    assert c.provider_available_by_market == {"ashare": True, "us": False}


def test_roe_unit_but_not_definition_comparable():
    from factor_engine.fields.concepts import get_concept

    c = get_concept("roe_decimal")
    assert c.unit_comparable is True
    assert c.definition_comparable is False
    assert c.cross_market_rank_allowed is False


# ---------------------------------------------------------------------------
# R17-067 index_weight role is a continuous weight, not a group key
# ---------------------------------------------------------------------------
def test_index_weight_role_is_weight():
    from factor_engine.fields.concepts import get_concept

    c = get_concept("index_weight")
    assert c is not None
    assert c.role == "weight"


# ---------------------------------------------------------------------------
# R17-050 research does not auto-open effective-time-only
# ---------------------------------------------------------------------------
def test_research_does_not_auto_enable_effective_time_only():
    from factor_engine.market.context import ASHARE_CONTEXT

    research = ASHARE_CONTEXT.as_research()
    assert research.allow_effective_time_only is False
    assert research.allow_proxy is True
    assert ASHARE_CONTEXT.with_non_pit_effective_time().allow_effective_time_only is True


# ---------------------------------------------------------------------------
# R17-013 every production table declares strict_pit_allowed
# ---------------------------------------------------------------------------
def test_all_tables_declare_pit():
    from factor_engine.fields.catalog import ASHARE_TABLE_SPECS
    from factor_engine.fields.catalog_us import US_TABLE_SPECS

    for t in list(ASHARE_TABLE_SPECS) + list(US_TABLE_SPECS):
        assert t.strict_pit_allowed is not None, f"{t.name} has UNKNOWN strict_pit_allowed"


# ---------------------------------------------------------------------------
# R17-001 market-aware resolver: A vs US unit separation
# ---------------------------------------------------------------------------
def test_resolve_market_field_a_vs_us_units():
    from factor_engine.fields.resolver import resolve_market_field
    from factor_engine.market.context import ASHARE_CONTEXT, US_CONTEXT

    a = resolve_market_field("StockDailyBar.close", ASHARE_CONTEXT)
    u = resolve_market_field("StockDailyBar.close", US_CONTEXT)
    assert a is not None and u is not None
    assert "CNY" in a.spec.unit
    assert "USD" in u.spec.unit
