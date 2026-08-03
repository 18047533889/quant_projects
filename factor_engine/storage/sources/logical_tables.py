# -*- coding: utf-8 -*-
"""Authoritative A-share LQTP logical-table to data_access mapping.

Keep policy beside the mapping so a newly mapped table cannot silently inherit an
unsafe alignment rule.  ``dataset=None`` denotes a table supplied by the anchor
or by a dedicated resolver rather than data_access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

JoinPolicy = Literal[
    "anchor", "exact", "exact_date", "asof_backward", "financial_pit",
    "relation_pit", "effective_only", "minute_session", "special",
]


@dataclass(frozen=True)
class LogicalTableContract:
    dataset: str | None
    join_policy: JoinPolicy
    required_parameter: str | None = None


ASHARE_LOGICAL_TABLES: dict[str, LogicalTableContract] = {
    "DailyBar": LogicalTableContract(None, "anchor"),
    "StockDailyBar": LogicalTableContract(None, "anchor"),
    "StockMinuteBar": LogicalTableContract("ashare_stock_minute", "minute_session"),
    "MinuteBar": LogicalTableContract("ashare_stock_minute", "minute_session"),
    "StockValuationDaily": LogicalTableContract("ashare_stock_valuation_daily", "exact"),
    # Historical alias retained for old formulas.
    "SizeDaily": LogicalTableContract("ashare_stock_valuation_daily", "exact"),
    "StockCapitalDaily": LogicalTableContract("ashare_stock_capital_daily", "exact"),
    "TurnoverBaseDaily": LogicalTableContract(None, "exact"),
    "StockIndicator": LogicalTableContract("ashare_stock_indicator", "financial_pit"),
    "StockBalance": LogicalTableContract("ashare_stock_balance", "financial_pit"),
    "StockIncome": LogicalTableContract("ashare_stock_income", "financial_pit"),
    "StockCashFlow": LogicalTableContract("ashare_stock_cashflow", "financial_pit"),
    "StockDividend": LogicalTableContract("ashare_stock_dividend", "effective_only"),
    "StockTopTenShareholder": LogicalTableContract("ashare_stock_topten_shareholder", "relation_pit"),
    "StockTopTenFloatShareholder": LogicalTableContract("ashare_stock_topten_float_shareholder", "relation_pit"),
    "StockIndustry": LogicalTableContract("ashare_stock_industry", "asof_backward", "IndustrySource"),
    # Historical alias retained for old formulas.
    "IndustryDaily": LogicalTableContract("ashare_stock_industry", "asof_backward", "IndustrySource"),
    "StockStatus": LogicalTableContract("ashare_stock_status", "asof_backward"),
    "StockList": LogicalTableContract("ashare_stock_list", "asof_backward"),
    "IndexConstituent": LogicalTableContract("ashare_index_constituent", "exact", "IndexSymbol"),
    "BenchmarkIndexDailyBar": LogicalTableContract("ashare_index_daily", "exact_date", "IndexSymbol"),
    "IndexDailyBar": LogicalTableContract("ashare_index_daily", "exact_date", "IndexSymbol"),
    "EtfDailyBar": LogicalTableContract("ashare_etf_daily", "exact"),
    "ETFDailyBar": LogicalTableContract("ashare_etf_daily", "exact"),
    "ETFList": LogicalTableContract("ashare_etf_list", "asof_backward"),
    "IndexList": LogicalTableContract("ashare_index_list", "asof_backward"),
    "Calendar": LogicalTableContract("ashare_calendar", "exact_date"),
    "Intermediate": LogicalTableContract(None, "special"),
    "DerivedField": LogicalTableContract(None, "special"),
}

TABLE_DATASETS: dict[str, str] = {
    table: contract.dataset
    for table, contract in ASHARE_LOGICAL_TABLES.items()
    if contract.dataset is not None
}


def logical_table_contract(table: str) -> LogicalTableContract:
    try:
        return ASHARE_LOGICAL_TABLES[str(table)]
    except KeyError as exc:
        raise KeyError(f"unknown A-share logical table {table!r}") from exc
