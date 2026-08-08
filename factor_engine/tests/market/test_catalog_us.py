# -*- coding: utf-8 -*-
"""US field catalog — unit/PIT facts from the US massive data dictionary."""
from __future__ import annotations

import pytest

from fields import US_FIELD_REGISTRY
from fields.catalog_us import US_FIELD_SPECS, US_TABLE_SPECS
from fields.registry import FieldRegistry


@pytest.fixture(scope="module", autouse=True)
def _reg():
    return FieldRegistry(US_FIELD_SPECS, US_TABLE_SPECS)


def test_us_ret_is_decimal() -> None:
    ret = US_FIELD_REGISTRY.require("ret", table="StockDailyBar")
    assert ret.source_name == "Ret"
    # decimal: no /10000, no /100
    assert ret.canonical_unit == "ratio"
    assert ret.scale_to_canonical == pytest.approx(1.0)
    assert "decimal" in ret.metadata.get("unit_note", "")


def test_us_adj_factor_backward_multiplier() -> None:
    adj = US_FIELD_REGISTRY.require("adj_factor")
    assert adj.metadata.get("direction") == "backward_multiplier"


def test_us_raw_prices_usd_per_share() -> None:
    for name in ("open", "high", "low", "close"):
        spec = US_FIELD_REGISTRY.require(name, table="StockDailyBar")
        assert spec.unit == "USD/share"
        assert spec.metadata.get("adjusted") is False


def test_us_financial_pit_on_filing_date() -> None:
    table = US_FIELD_REGISTRY.resolve_table("StockIncome")
    assert table.join_policy == "financial_pit"
    assert table.knowledge_time_column == "filing_date"
    assert table.period_id_column == "period_end"
    assert "timeframe" in table.required_parameters
    filing = US_FIELD_REGISTRY.require("filing_date")
    assert filing.role == "knowledge_time"


def test_us_valuation_sparse_flagged() -> None:
    val = US_FIELD_REGISTRY.resolve_table("StockValuationDaily")
    assert val.current_snapshot_only is True
    assert val.strict_pit_allowed is False


def test_us_dividend_declaration_pit() -> None:
    table = US_FIELD_REGISTRY.resolve_table("StockDividend")
    assert table.knowledge_time_column == "declaration_date"
    assert table.effective_time_column == "ex_dividend_date"
    div = US_FIELD_REGISTRY.require("cash_amount")
    assert div.unit == "USD"
    assert "currency" in div.metadata.get("note", "")


def test_us_financial_ratios_decimal() -> None:
    for name in ("return_on_equity", "return_on_assets", "dividend_yield"):
        spec = US_FIELD_REGISTRY.require(name)
        assert spec.canonical_unit == "ratio"
        assert spec.scale_to_canonical == pytest.approx(1.0)
        assert "decimal" in spec.metadata.get("unit_note", "")


def test_us_market_cap_derived_source() -> None:
    mc = US_FIELD_REGISTRY.require("market_cap")
    assert mc.unit == "USD"
    assert "TickerSharesSnapshot" in mc.metadata.get("preferred_source", "")


def test_us_no_industry_or_status() -> None:
    # StockIndustry / StockStatus are EMPTY (0 rows) in the US dictionary
    assert US_FIELD_REGISTRY.resolve_table("StockIndustry") is None
    assert US_FIELD_REGISTRY.resolve_table("StockStatus") is None


def test_us_catalog_registers_cleanly() -> None:
    assert len(US_FIELD_REGISTRY.tables()) == 16
    assert len(US_FIELD_REGISTRY.fields()) >= 70
