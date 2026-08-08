# -*- coding: utf-8 -*-
"""US COS contracts."""
from .cos_contract import _c

D = "daily_parquet"
_TIMEFRAME = ("quarterly", "annual", "trailing_twelve_months")

US_COS_CONTRACTS = {
    "us_calendar": _c("us_calendar", "us", "STATIC", "read_full", "trade_date"),
    "us_stock_daily": _c("us_stock_daily", "us", "D1", "equi", "Ticker", return_column="Ret", return_scale=1, adjustment_column="AdjFactor", adjustment_convention="backward_vendor_factor", storage_layout=D),
    "us_stock_list": _c("us_stock_list", "us", "D1", "equi", "Symbol", calendar_domain="calendar_day", storage_layout=D),
    "us_etf_daily": _c("us_etf_daily", "us", "D1", "equi", "Ticker", storage_layout=D),
    "us_etf_list": _c("us_etf_list", "us", "STATIC", "read_full", "ticker"),
    "us_security_master": _c("us_security_master", "us", "STATIC", "read_full", "ticker"),
    "us_ticker_map": _c("us_ticker_map", "us", "STATIC", "read_full", "ticker"),
    "us_stock_indices_components": _c("us_stock_indices_components", "us", "D1", "equi", "Symbol", storage_layout=D),
    "us_universe_daily": _c("us_universe_daily", "us", "D1", "equi", "ticker"),
    "us_adj_factor": _c("us_adj_factor", "us", "D1", "equi", "ticker", adjustment_column="adj_factor", adjustment_convention="backward_clean_factor"),
    "us_adj_factor_clamped": _c("us_adj_factor_clamped", "us", "D1", "equi", "ticker"),
    "us_is_early_close": _c("us_is_early_close", "us", "STATIC", "read_full", "date"),
    "us_is_ticker_halt_minute": _c("us_is_ticker_halt_minute", "us", "MINUTE", "equi", "ticker"),
    "us_stocks_sip_day_aggs": _c("us_stocks_sip_day_aggs", "us", "D1", "equi", "ticker"),
}

for _name in ("balance", "income", "cashflow"):
    key = f"us_stock_{_name}"
    US_COS_CONTRACTS[key] = _c(key, "us", "E2", "asof", "ticker", availability_column="filing_date", period_column="period_end", required_event_filters=("timeframe",), allowed_filter_values=(("timeframe", _TIMEFRAME),), availability_must_follow_period=True, storage_layout="period_files")

US_COS_CONTRACTS.update({
    "us_stock_dividend": _c("us_stock_dividend", "us", "E2", "event", "ticker", pit="strict", availability_column="declaration_date", event_column="ex_dividend_date", period_column="ex_dividend_date", event_id_columns=("id",), storage_layout="event_files", note="Knowledge-time = declaration_date (~0.51% null; rows missing it fail closed, never ex-date fallback); effective/event date = ex_dividend_date (secondary)."),
    # StockCapitalDaily 目录混放两种 schema（C13）：{date}.parquet=拆分事件，
    # shares_{date}.parquet=稀疏 PIT 股本。拆成两个数据集，各自 storage_layout
    # 归递归布局（全量 sync + complete marker），绝不按决策日枚举文件名。
    "us_stock_capital_split": _c("us_stock_capital_split", "us", "E2", "event", "ticker", pit="effective_time_only", event_column="execution_date", period_column="execution_date", event_id_columns=("id",), storage_layout="event_files", note="Split/adjustment events ({date}.parquet), not a daily share-count snapshot."),
    "us_stock_capital_shares": _c("us_stock_capital_shares", "us", "E2", "asof", "Ticker", availability_column="TradeDate", period_column="TradeDate", revision_columns=(), storage_layout="period_files", cardinality="one_to_many", unique_key=("Ticker", "TradeDate", "id"), note="Sparse PIT shares (shares_{date}.parquet); same-day same-ticker can repeat, dedup needed. Prefer us_ticker_shares_snapshot for daily exposure."),
    "us_stock_indicator": _c("us_stock_indicator", "us", "X0", "refuse_panel", "ticker", panel="sparse", pit="unsupported", storage_layout="sparse_files"),
    "us_stock_valuation_daily": _c("us_stock_valuation_daily", "us", "X0", "refuse_panel", "ticker", panel="sparse", pit="unsupported", storage_layout="sparse_files"),
    "us_ticker_shares_snapshot": _c("us_ticker_shares_snapshot", "us", "D1", "equi", "ticker", storage_layout=D, note="Daily shares snapshot; preferred market-cap exposure = weighted_shares_outstanding x Close."),
    "us_security_master_daily_snap": _c("us_security_master_daily_snap", "us", "D1", "equi", "ticker", storage_layout=D),
    # 字典把 FactNews 明确标为 RAW_EVENT（PIT=published_utc）。新闻事件与财务
    # 事件不同：entity_list explode、event_id、publication 时间戳、事件窗口、
    # source confidence——不再复用 E2 标签。
    "us_fact_news": _c("us_fact_news", "us", "RAW_EVENT", "event", "ticker", pit="strict", availability_column="published_utc", storage_layout="event_files", cardinality="one_to_many", note="FactNews clean news; PIT=published_utc; TradeDate is partition only."),
    "us_ticker_alias": _c("us_ticker_alias", "us", "EMPTY", "refuse", "ticker", pit="unsupported"),
    "us_stock_status": _c("us_stock_status", "us", "EMPTY", "refuse", "Symbol", pit="unsupported"),
    "us_stock_industry": _c("us_stock_industry", "us", "EMPTY", "refuse", "Symbol", pit="unsupported"),
})
