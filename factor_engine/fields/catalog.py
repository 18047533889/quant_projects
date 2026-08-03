"""Built-in A-share daily and fundamental field catalog."""
from __future__ import annotations

from .spec import FieldSpec, TableSpec
from .units import (
    UNIT_BOOLEAN, UNIT_CNY, UNIT_DATE, UNIT_DATETIME, UNIT_DIMENSIONLESS,
    UNIT_IDENTIFIER, UNIT_RATIO, UNIT_SHARE, UNIT_TEXT,
)


def _table(name, dataset, *, time="TradeDate", instrument="Symbol", domain="auxiliary", frequency="daily", aliases=()):
    return TableSpec(
        name=name, dataset=dataset, time_column=time, instrument_column=instrument,
        domain=domain, frequency=frequency, aliases=tuple(aliases),
    )


# The 20 contracts mirror the published LQTP A-share inventory.  Minute is
# represented only as a source-table contract here; minute behavior belongs to
# the separate intraday layer and is intentionally not implemented in fields.
ASHARE_TABLE_SPECS: tuple[TableSpec, ...] = (
    _table("StockDailyBar", "ashare_stock_daily", domain="price_volume", aliases=("DailyBar",)),
    _table("StockMinuteBar", "ashare_stock_minute", time="QuoteTime", domain="price_volume", frequency="minute", aliases=("MinuteBar",)),
    _table("StockValuationDaily", "ashare_stock_valuation_daily", domain="valuation"),
    _table("StockCapitalDaily", "ashare_stock_capital_daily", domain="capital"),
    _table("StockIndicator", "ashare_stock_indicator", time="ReportDate", domain="fundamental"),
    _table("StockBalance", "ashare_stock_balance", time="ReportDate", domain="fundamental"),
    _table("StockIncome", "ashare_stock_income", time="ReportDate", domain="fundamental"),
    _table("StockCashFlow", "ashare_stock_cashflow", time="ReportDate", domain="fundamental", aliases=("StockCashflow",)),
    _table("StockDividend", "ashare_stock_dividend", time="ExDate", domain="corporate_action"),
    _table("IndexDailyBar", "ashare_index_daily", instrument="IndexCode", domain="index", aliases=("BenchmarkIndexDailyBar",)),
    _table("IndexConstituent", "ashare_index_constituent", instrument="IndexCode", domain="index"),
    _table("EtfDailyBar", "ashare_etf_daily", domain="etf", aliases=("ETFDailyBar",)),
    _table("Calendar", "ashare_calendar", time="Date", instrument=None, domain="calendar"),
    _table("StockList", "ashare_stock_list", time=None, instrument=None, domain="reference"),
    _table("EtfList", "ashare_etf_list", time=None, instrument=None, domain="reference", aliases=("ETFList",)),
    _table("IndexList", "ashare_index_list", time=None, instrument=None, domain="reference"),
    _table("StockIndustry", "ashare_stock_industry", time=None, domain="classification"),
    _table("StockStatus", "ashare_stock_status", domain="status"),
    _table("StockTopTenShareholder", "ashare_stock_topten_shareholder", time=None, domain="ownership"),
    _table("StockTopTenFloatShareholder", "ashare_stock_topten_float_shareholder", time=None, domain="ownership"),
)


def _f(name, table, source, *, dtype="float64", unit=UNIT_DIMENSIONLESS, aliases=(), role="feature", description="", adjustment=None):
    dataset = next((item.dataset for item in ASHARE_TABLE_SPECS if item.name == table), None)
    return FieldSpec(
        name=name, table=table, source_name=source, dtype=dtype, unit=unit,
        aliases=tuple(aliases), role=role, description=description,
        dataset=dataset, adjustment=adjustment,
    )


# Core fields are intentionally explicit.  Wider financial dictionaries can be
# registered by clients without changing this stable foundation.
ASHARE_FIELD_SPECS: tuple[FieldSpec, ...] = (
    _f("trade_date", "StockDailyBar", "TradeDate", dtype="date", unit=UNIT_DATE, role="time", aliases=("date",)),
    _f("symbol", "StockDailyBar", "Symbol", dtype="string", unit=UNIT_IDENTIFIER, role="instrument", aliases=("ticker",)),
    _f("open", "StockDailyBar", "Open", unit=UNIT_CNY, adjustment="multiply:Factor"),
    _f("high", "StockDailyBar", "High", unit=UNIT_CNY, adjustment="multiply:Factor"),
    _f("low", "StockDailyBar", "Low", unit=UNIT_CNY, adjustment="multiply:Factor"),
    _f("close", "StockDailyBar", "Close", unit=UNIT_CNY, adjustment="multiply:Factor"),
    _f("pre_close", "StockDailyBar", "PreClose", unit=UNIT_CNY, aliases=("prev_close",), adjustment="multiply:Factor"),
    _f("volume", "StockDailyBar", "Volume", unit=UNIT_SHARE, adjustment="divide:Factor"),
    _f("amount", "StockDailyBar", "Amount", unit=UNIT_CNY, aliases=("turnover_value",)),
    _f("ret", "StockDailyBar", "Return", unit=UNIT_RATIO, aliases=("return", "returns")),
    _f("adj_factor", "StockDailyBar", "Factor", unit=UNIT_RATIO, aliases=("factor",)),
    _f("vwap", "StockDailyBar", "Vwap", unit=UNIT_CNY, adjustment="multiply:Factor"),
    _f("high_limit", "StockDailyBar", "HighLimit", unit=UNIT_CNY, adjustment="multiply:Factor"),
    _f("low_limit", "StockDailyBar", "LowLimit", unit=UNIT_CNY, adjustment="multiply:Factor"),
    _f("is_suspend", "StockDailyBar", "IsSuspend", dtype="bool", unit=UNIT_BOOLEAN),
    _f("update_time", "StockDailyBar", "UpdateTime", dtype="datetime", unit=UNIT_DATETIME, role="knowledge_time"),

    _f("market_cap", "StockValuationDaily", "MarketCap", unit=UNIT_CNY, aliases=("mkt_cap",)),
    _f("circulating_market_cap", "StockValuationDaily", "CirculatingMarketCap", unit=UNIT_CNY, aliases=("float_market_cap",)),
    _f("pe_ratio", "StockValuationDaily", "PeRatio", aliases=("pe",)),
    _f("pb_ratio", "StockValuationDaily", "PbRatio", aliases=("pb",)),
    _f("ps_ratio", "StockValuationDaily", "PsRatio", aliases=("ps",)),
    _f("pcf_ratio", "StockValuationDaily", "PcfRatio", aliases=("pcf",)),
    _f("turnover_ratio", "StockValuationDaily", "TurnoverRatio", unit=UNIT_RATIO, aliases=("turnover",)),
    _f("dividend_yield", "StockValuationDaily", "DividendRatio", unit=UNIT_RATIO, aliases=("dividend_ratio",)),
    _f("free_market_cap", "StockValuationDaily", "FreeMarketCap", unit=UNIT_CNY),

    _f("total_capital", "StockCapitalDaily", "TotalCapital", unit=UNIT_SHARE, aliases=("total_shares",)),
    _f("circulating_capital", "StockCapitalDaily", "CirculatingCapital", unit=UNIT_SHARE, aliases=("float_shares",)),
    _f("eps", "StockIndicator", "Eps", unit=UNIT_CNY),
    _f("roe", "StockIndicator", "Roe", unit=UNIT_RATIO),
    _f("report_period_end_date", "StockIndicator", "ReportPeriodEndDate", dtype="date", unit=UNIT_DATE, role="report_time", aliases=("report_date",)),

    _f("total_operating_revenue", "StockIncome", "TotalOperatingRevenue", unit=UNIT_CNY),
    _f("operating_revenue", "StockIncome", "OperatingRevenue", unit=UNIT_CNY, aliases=("revenue",)),
    _f("net_profit", "StockIncome", "NetProfit", unit=UNIT_CNY),
    _f("total_assets", "StockBalance", "TotalAssets", unit=UNIT_CNY),
    _f("total_liabilities", "StockBalance", "TotalLiabilities", unit=UNIT_CNY),
    _f("shareholders_equity", "StockBalance", "ShareholdersEquity", unit=UNIT_CNY, aliases=("equity",)),
    _f("operating_cash_flow", "StockCashFlow", "NetCashFlowFromOperatingActivities", unit=UNIT_CNY, aliases=("ocf",)),
    _f("cash_dividend", "StockDividend", "CashDividend", unit=UNIT_CNY),

    _f("industry_code", "StockIndustry", "IndustryCode", dtype="string", unit=UNIT_IDENTIFIER, role="group_key"),
    _f("industry_name", "StockIndustry", "IndustryName", dtype="string", unit=UNIT_TEXT, role="label"),
    _f("listed_state", "StockStatus", "ListedState", dtype="string", unit=UNIT_TEXT, role="status"),
    _f("index_weight", "IndexConstituent", "Weight", unit=UNIT_RATIO, aliases=("weight",)),
)


__all__ = ["ASHARE_FIELD_SPECS", "ASHARE_TABLE_SPECS"]
