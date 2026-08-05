# -*- coding: utf-8 -*-
"""A-share COS contracts."""
from .cos_contract import _c

_INDUSTRY = ("sw_l1", "sw_l2", "sw_l3", "zjw", "jq_l1", "jq_l2")
D = "daily_parquet"

ASHARE_COS_CONTRACTS = {
    "ashare_calendar": _c("ashare_calendar", "ashare", "STATIC", "read_full", "TradeDate"),
    "ashare_stock_daily": _c("ashare_stock_daily", "ashare", "D1", "equi", "Symbol", return_column="Return", return_scale=1 / 10000, adjustment_column="Factor", adjustment_convention="forward_vendor_factor", storage_layout=D),
    "ashare_stock_minute": _c("ashare_stock_minute", "ashare", "MINUTE", "equi", "Symbol", storage_layout=D),
    "ashare_stock_list": _c("ashare_stock_list", "ashare", "D1", "equi", "Symbol", storage_layout=D),
    "ashare_stock_status": _c("ashare_stock_status", "ashare", "S1", "equi", "Symbol", storage_layout=D),
    "ashare_stock_industry": _c("ashare_stock_industry", "ashare", "D1", "equi", "Symbol", required_panel_filters=("IndustrySource",), allowed_filter_values=(("IndustrySource", _INDUSTRY),), storage_layout=D),
    "ashare_stock_valuation_daily": _c("ashare_stock_valuation_daily", "ashare", "D1", "equi", "Symbol", storage_layout=D),
    "ashare_stock_capital_daily": _c("ashare_stock_capital_daily", "ashare", "S1", "equi", "Symbol", storage_layout=D),
    "ashare_etf_daily": _c("ashare_etf_daily", "ashare", "D1", "equi", "Symbol", storage_layout=D),
    "ashare_etf_list": _c("ashare_etf_list", "ashare", "D1", "equi", "Symbol", storage_layout=D),
    "ashare_index_daily": _c("ashare_index_daily", "ashare", "D1", "equi", "Symbol", return_column="Return", return_scale=1 / 10000, storage_layout=D),
    "ashare_index_list": _c("ashare_index_list", "ashare", "D1", "equi", "Symbol", storage_layout=D),
    "ashare_index_constituent": _c("ashare_index_constituent", "ashare", "D1", "equi", "Symbol", required_panel_filters=("IndexSymbol",), storage_layout=D),
    "ashare_universe_daily": _c("ashare_universe_daily", "ashare", "D1", "equi", "Symbol"),
    "ashare_stock_topten_shareholder": _c("ashare_stock_topten_shareholder", "ashare", "S1", "equi", "Symbol", storage_layout=D),
    "ashare_stock_topten_float_shareholder": _c("ashare_stock_topten_float_shareholder", "ashare", "S1", "equi", "Symbol", storage_layout=D),
}

for _name in ("balance", "income", "cashflow", "indicator"):
    key = f"ashare_stock_{_name}"
    ASHARE_COS_CONTRACTS[key] = _c(key, "ashare", "E1", "asof", "Symbol", availability_column="PubDate", period_column="ReportPeriodEndDate", revision_columns=("UpdateTime",), availability_must_follow_period=True, storage_layout=D)

ASHARE_COS_CONTRACTS["ashare_stock_dividend"] = _c(
    "ashare_stock_dividend", "ashare", "E1", "event", "Symbol",
    pit="effective_time_only", event_column="ExDividendDate", period_column="ExDividendDate",
    revision_columns=("UpdateTime",), storage_layout="event_files",
    note="No reliable announcement timestamp; explicit effective-date use only.",
)
