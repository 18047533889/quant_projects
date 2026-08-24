# -*- coding: utf-8 -*-
"""R23-041..047: flow_type is compile-time semantic context, NOT a searchable
parameter; and A-share YTD fields bind to CumulativeYTDFlow at the catalog level.
"""
from __future__ import annotations

from factor_engine.fields.catalog import ASHARE_FIELD_SPECS, ASHARE_TABLE_SPECS
from factor_engine.cleaned_operators.registry import OperatorRegistry

_FLOW_TYPE_OPS = (
    "fin_pct_change",
    "fin_log_change",
    "fin_growth",
    "fin_cagr",
    "fin_growth_acceleration",
    "fin_growth_volatility",
    "fin_growth_stability",
    "fin_growth_persistence",
    "fin_cash_earnings_gap",
    "fin_accrual_ratio",
    "fin_cash_conversion",
)


def test_flow_type_is_not_a_searchable_parameter():
    # R23-041: a miner must not be able to freely pick flow_type to bypass the
    # A-share YTD grain.  flow_type must NOT appear in any ParamSpec (search).
    for name in _FLOW_TYPE_OPS:
        op = OperatorRegistry._operators.get(name, {}).get("pandas_numpy")
        if op is None:
            continue
        specs = getattr(op.metadata, "param_specs", None) or {}
        assert "flow_type" not in specs, f"{name}: flow_type must not be searchable"


def test_ashare_income_fields_bind_to_cumulative_ytd_flow():
    # R23-045: A-share StockIncome/StockCashFlow flow_ytd -> CumulativeYTDFlow.
    for f in ASHARE_FIELD_SPECS:
        if f.table == "StockIncome" and f.name in ("net_profit", "operating_profit", "total_operating_revenue"):
            assert f.flow_semantics == "cumulative_ytd_flow", f.name


def test_ashare_balance_fields_bind_to_stock():
    for f in ASHARE_FIELD_SPECS:
        if f.table == "StockBalance" and f.name == "total_assets":
            assert f.flow_semantics == "stock", f.name


def test_ashare_financial_tables_grain_is_flow_ytd():
    for t in ASHARE_TABLE_SPECS:
        if t.name in ("StockIncome", "StockCashFlow"):
            fields = [f for f in ASHARE_FIELD_SPECS if f.table == t.name]
            flows = {f.name: f.flow_semantics for f in fields if f.flow_semantics}
            assert flows, t.name
            # every flow-ytd field must be declared cumulative YTD
            for f in fields:
                if "ytd" in f.grain:
                    assert f.flow_semantics == "cumulative_ytd_flow", f"{t.name}.{f.name}"
