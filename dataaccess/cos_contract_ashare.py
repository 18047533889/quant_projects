# -*- coding: utf-8 -*-
"""A-share COS contracts."""
from .cos_contract import _c

_INDUSTRY = ("sw_l1", "sw_l2", "sw_l3", "zjw", "jq_l1", "jq_l2")
D = "daily_parquet"

ASHARE_COS_CONTRACTS = {
    "ashare_calendar": _c("ashare_calendar", "ashare", "STATIC", "read_full", "TradeDate"),
    # 字典 §1.1/C19（2026-08-08 实证）：A股 Factor 是「后复权累积因子」，
    # 后复权价 = Close × Factor（旧「前复权=Close/Factor」已废止）；与美股
    # AdjFactor 同为乘法后复权但基期/事件覆盖不同，禁止当同一列混用。
    "ashare_stock_daily": _c("ashare_stock_daily", "ashare", "D1", "equi", "Symbol", return_column="Return", return_scale=1 / 10000, adjustment_column="Factor", adjustment_convention="backward_vendor_factor", storage_layout=D),
    "ashare_stock_minute": _c("ashare_stock_minute", "ashare", "MINUTE", "equi", "Symbol", storage_layout=D),
    # 字典：StockList/Status/Industry/TopTen/ETFList/IndexList/IndexConstituent
    # 含周末自然日文件；行情/估值才主要是交易日。→ calendar_domain="calendar_day"。
    "ashare_stock_list": _c("ashare_stock_list", "ashare", "D1", "equi", "Symbol", calendar_domain="calendar_day", storage_layout=D),
    "ashare_stock_status": _c("ashare_stock_status", "ashare", "S1", "equi", "Symbol", calendar_domain="calendar_day", storage_layout=D),
    "ashare_stock_industry": _c("ashare_stock_industry", "ashare", "D1", "equi", "Symbol", required_panel_filters=("IndustrySource",), required_dimension_filters=("IndustrySource",), allowed_filter_values=(("IndustrySource", _INDUSTRY),), cardinality="one_to_many", unique_key=("TradeDate", "Symbol", "IndustrySource"), calendar_domain="calendar_day", storage_layout=D),
    "ashare_stock_valuation_daily": _c("ashare_stock_valuation_daily", "ashare", "D1", "equi", "Symbol", storage_layout=D),
    "ashare_stock_capital_daily": _c("ashare_stock_capital_daily", "ashare", "S1", "equi", "Symbol", storage_layout=D),
    "ashare_etf_daily": _c("ashare_etf_daily", "ashare", "D1", "equi", "Symbol", storage_layout=D),
    "ashare_etf_list": _c("ashare_etf_list", "ashare", "D1", "equi", "Symbol", calendar_domain="calendar_day", storage_layout=D),
    "ashare_index_daily": _c("ashare_index_daily", "ashare", "D1", "equi", "Symbol", return_column="Return", return_scale=1 / 10000, storage_layout=D),
    "ashare_index_list": _c("ashare_index_list", "ashare", "D1", "equi", "Symbol", calendar_domain="calendar_day", storage_layout=D),
    "ashare_index_constituent": _c("ashare_index_constituent", "ashare", "D1", "equi", "Symbol", required_panel_filters=("IndexSymbol",), required_dimension_filters=("IndexSymbol",), cardinality="one_to_many", unique_key=("IndexSymbol", "TradeDate", "Symbol"), calendar_domain="calendar_day", storage_layout=D),
    "ashare_universe_daily": _c("ashare_universe_daily", "ashare", "D1", "equi", "Symbol"),
    "ashare_stock_topten_shareholder": _c("ashare_stock_topten_shareholder", "ashare", "S1", "equi", "Symbol", calendar_domain="calendar_day", cardinality="one_to_many", unique_key=("TradeDate", "Symbol", "Rank"), storage_layout=D),
    "ashare_stock_topten_float_shareholder": _c("ashare_stock_topten_float_shareholder", "ashare", "S1", "equi", "Symbol", calendar_domain="calendar_day", cardinality="one_to_many", unique_key=("TradeDate", "Symbol", "Rank"), storage_layout=D),
}

for _name in ("balance", "income", "cashflow", "indicator"):
    key = f"ashare_stock_{_name}"
    # R24 P0-PIT4 §13：UpdateTime 只是 dedup_tiebreaker（供应商 freshness），
    # **不是** historical revision availability。knowledge_time=PubDate；
    # pit_fidelity=knowledge_date_pit（COS 无历史 revision vintage，不能承诺
    # full bitemporal revision-vintage PIT）。time_representation=date_label——
    # 不允许把 naive 00:00 当 UTC instant 转时区提前一天。
    ASHARE_COS_CONTRACTS[key] = _c(
        key, "ashare", "E1", "asof", "Symbol",
        availability_column="PubDate",
        period_column="ReportPeriodEndDate",
        revision_columns=("UpdateTime",),
        availability_must_follow_period=True,
        storage_layout=D,
        time_representation="date_label",
        time_precision="date",
        semantic_timezone="Asia/Shanghai",
        revision_availability_time=None,
        pit_fidelity="knowledge_date_pit",
    )

ASHARE_COS_CONTRACTS["ashare_stock_dividend"] = _c(
    "ashare_stock_dividend", "ashare", "E1", "event", "Symbol",
    pit="effective_time_only", event_column="ExDividendDate", period_column="ExDividendDate",
    revision_columns=("UpdateTime",), storage_layout="event_files",
    note="No reliable announcement timestamp; explicit effective-date use only.",
)
