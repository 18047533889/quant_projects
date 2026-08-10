"""US-equity market field catalog (typed FieldSpec/TableSpec).

Sourced from ``/home/shw/COS_us_massive_data_dictionary.md`` (2026-08-08).
Unit conventions verified against that dictionary:
- ``Ret`` is decimal (0.01 == 1%); do NOT /10000.
- ``AdjFactor`` is a backward cumulative multiplier; adjusted = Close * AdjFactor.
- ``dividend_yield`` / ``return_on_equity`` / ``return_on_assets`` are decimal.
- Financial statements are PIT on ``filing_date``; must filter ``timeframe`` first.
- ``market_cap`` derived from ``TickerSharesSnapshot.weighted_shares_outstanding``
  (~42% coverage) rather than the X0 sparse valuation table.
- US has NO industry table, NO status table, NO index weight (all EMPTY).
"""
from __future__ import annotations

from .spec import FieldSpec, TableSpec

# ---------------------------------------------------------------------------
# Unit spellings (kept as stable catalog strings; v2 UnitSpec mapping lives in
# ``fields/units_v2.py``).
# ---------------------------------------------------------------------------
_UNIT_USD = "USD"
_UNIT_USD_PER_SHARE = "USD/share"
_UNIT_SHARE = "share"
_UNIT_RATIO = "ratio"
_UNIT_BOOLEAN = "boolean"
_UNIT_DATE = "date"
_UNIT_DATETIME = "datetime"
_UNIT_TEXT = "text"
_UNIT_IDENTIFIER = "identifier"


def _table(
    name,
    dataset,
    *,
    time="TradeDate",
    instrument="Ticker",
    domain="auxiliary",
    frequency="daily",
    aliases=(),
    table_kind="panel",
    join_policy="exact",
    required_parameters=(),
    knowledge=None,
    effective=None,
    period=None,
    revision=None,
    timezone=None,
    session_calendar=None,
    cardinality="many_to_one",
    strict_pit_allowed=None,
    current_snapshot_only=False,
    metadata=None,
):
    # R17-013: helper default UNKNOWN (None); every production-readable US table
    # declares PIT eligibility explicitly.
    return TableSpec(
        name=name,
        dataset=dataset,
        time_column=time,
        instrument_column=instrument,
        domain=domain,
        frequency=frequency,
        aliases=tuple(aliases),
        table_kind=table_kind,
        join_policy=join_policy,
        required_parameters=tuple(required_parameters),
        knowledge_time_column=knowledge,
        effective_time_column=effective,
        period_id_column=period,
        revision_column=revision,
        timezone=timezone,
        session_calendar=session_calendar,
        cardinality=cardinality,
        strict_pit_allowed=strict_pit_allowed,
        current_snapshot_only=current_snapshot_only,
        metadata=dict(metadata or {}),
    )


US_TABLE_SPECS: tuple[TableSpec, ...] = (
    # R17-013: every production-readable US table declares strict_pit_allowed
    # explicitly (+ reason in metadata); R17-018: dataset names align with the
    # DataAccess registry.
    _table("StockDailyBar", "us_stock_daily", domain="price_volume", instrument="Ticker",
           strict_pit_allowed=True, metadata={"pit_reason": "exact_daily D1; PIT-safe"}),
    _table("StockList", "us_stock_list", instrument="Symbol", domain="reference", join_policy="exact",
           strict_pit_allowed=True, metadata={"pit_reason": "exact full-history; PIT-safe"}),
    _table(
        "SecurityMaster", "us_security_master_daily_snap", instrument="Ticker",
        domain="reference", table_kind="static", join_policy="exact",
        strict_pit_allowed=True,
        metadata={"pit_reason": "static reference for stable identifiers only; "
                                "historical attributes come from DailySnap (R17-068)"},
    ),
    _table(
        "TickerSharesSnapshot", "us_ticker_shares_snapshot", instrument="ticker",
        domain="capital", join_policy="exact", strict_pit_allowed=True,
        metadata={"pit_reason": "shares snapshot; exact equi-join; PIT-safe"},
    ),
    # R17-018: dataset name aligns with the DataAccess registry
    # (us_stock_indices_components, not us_stock_index_components).
    _table(
        "StockIndicesComponents", "us_stock_indices_components", instrument="Symbol",
        domain="index", table_kind="relation", join_policy="exact",
        required_parameters=("IndexName",), cardinality="one_to_many",
        strict_pit_allowed=True,
        metadata={"pit_reason": "membership relation; IndexName required; PIT-safe"},
    ),
    # R17-019: DataAccess split the dual-schema US StockCapitalDaily into two
    # datasets: ``us_stock_capital_split`` ({date}.parquet split/adjustment
    # events) and ``us_stock_capital_shares`` (shares_{date}.parquet PIT shares).
    # FE mirrors that split — a single reader/glob MUST NOT mix the two schemas.
    _table(
        "USStockCapitalSplitEvent", "us_stock_capital_split", instrument="ticker",
        domain="capital", table_kind="event", join_policy="exact",
        time="TradeDate", effective="execution_date", strict_pit_allowed=False,
        metadata={"pit_reason": "split/adjustment EVENT rows (effective execution_date); "
                                "partition key TradeDate matches DataAccess (R17-019)"},
    ),
    _table(
        "USTickerSharesPITEvent", "us_stock_capital_shares", instrument="ticker",
        domain="capital", table_kind="event", join_policy="financial_pit",
        knowledge="filing_date", period="filing_date", strict_pit_allowed=True,
        metadata={"pit_reason": "sparse shares PIT event source; asof(filing_date) (R17-019)"},
    ),
    # US financial statements: PIT on filing_date; period = period_end;
    # timeframe (quarterly/annual/trailing_twelve_months) MUST be filtered first.
    _table(
        "StockBalance", "us_stock_balance", time="filing_date", instrument="ticker",
        domain="fundamental", table_kind="financial_event", join_policy="financial_pit",
        knowledge="filing_date", period="period_end", required_parameters=("timeframe",),
        strict_pit_allowed=True,
        metadata={"pit_reason": "financial_pit on filing_date; timeframe required; PIT-safe"},
    ),
    _table(
        "StockIncome", "us_stock_income", time="filing_date", instrument="ticker",
        domain="fundamental", table_kind="financial_event", join_policy="financial_pit",
        knowledge="filing_date", period="period_end", required_parameters=("timeframe",),
        strict_pit_allowed=True,
        metadata={"pit_reason": "financial_pit on filing_date; timeframe required; PIT-safe"},
    ),
    _table(
        "StockCashFlow", "us_stock_cashflow", time="filing_date", instrument="ticker",
        domain="fundamental", table_kind="financial_event", join_policy="financial_pit",
        knowledge="filing_date", period="period_end", required_parameters=("timeframe",),
        strict_pit_allowed=True,
        metadata={"pit_reason": "financial_pit on filing_date; timeframe required; PIT-safe"},
    ),
    # R17-020: US StockDividend has NO TradeDate and NO period_end.  knowledge =
    # declaration_date (PIT anchor), effective = ex_dividend_date (may be future;
    # clip in backtest), partition/effective key = ex_dividend_date.
    # ``asof_backward`` is used (not ``financial_pit``) because a dividend is an
    # EVENT table — it has no report-period selection; the registry forbids
    # ``financial_pit`` without a period_id_column.
    _table(
        "StockDividend", "us_stock_dividend", time="ex_dividend_date", instrument="ticker",
        domain="corporate_action", table_kind="event", join_policy="asof_backward",
        knowledge="declaration_date", effective="ex_dividend_date", period=None,
        strict_pit_allowed=True,
        metadata={"pit_reason": "PIT on declaration_date; effective ex_dividend_date may be "
                                "future (clip); currency filter required (R17-020/021)"},
    ),
    # X0 sparse: only ~49 calendar files (2026-05-12..2026-07-27). NOT a full
    # history daily panel. current_snapshot_only forces strict_pit_allowed=False.
    _table(
        "StockValuationDaily", "us_stock_valuation_daily", instrument="ticker",
        domain="valuation", join_policy="exact", current_snapshot_only=True,
        strict_pit_allowed=False,
        metadata={"pit_reason": "X0 sparse current snapshot only; not historical"},
    ),
    _table(
        "StockIndicator", "us_stock_indicator", instrument="ticker",
        domain="fundamental", join_policy="exact", current_snapshot_only=True,
        strict_pit_allowed=False,
        metadata={"pit_reason": "X0 sparse current snapshot only; not historical"},
    ),
    # R17-022/023/024: FactNews published_utc is tz-aware UTC (not America/New_York);
    # ``tickers`` is a list<string> needing an explode adapter (NewsEventAdapter);
    # PIT is published_utc only; timezone conversion happens only when mapping to a
    # US session/decision timestamp.
    _table(
        "FactNews", "us_fact_news", time="published_utc", instrument="ticker",
        domain="alternative", table_kind="event", join_policy="exact",
        knowledge="published_utc", strict_pit_allowed=True,
        metadata={
            "pit_reason": "PIT on published_utc (tz-aware UTC); after-close -> next session",
            "timezone": "UTC",
            "instrument_kind": "list<explode>",  # tickers is a list; NewsEventAdapter explodes
        },
    ),
    _table("Calendar", "us_calendar", time="trade_date", instrument=None,
           domain="calendar", table_kind="calendar",
           strict_pit_allowed=True, metadata={"pit_reason": "calendar reference; PIT-safe"}),
    # R17-018: DataAccess registry name is us_is_early_close.
    _table("EarlyClose", "us_is_early_close", time="date", instrument=None,
           domain="calendar", table_kind="calendar",
           strict_pit_allowed=True, metadata={"pit_reason": "calendar reference; PIT-safe"}),
    _table("UniverseDaily", "us_universe_daily", time="trade_date", instrument="ticker",
           domain="reference", join_policy="exact",
           strict_pit_allowed=True, metadata={"pit_reason": "exact daily universe; PIT-safe"}),
)

_TABLE_BY_NAME = {item.name: item for item in US_TABLE_SPECS}


def _value_kind(dtype: str, unit: str, role: str) -> str:
    if role in {"time", "knowledge_time", "effective_time", "period_id", "ingestion_time"}:
        return "datetime" if dtype == "datetime" else "date"
    if role in {"instrument", "identifier"}:
        return "identifier"
    if role in {"label", "group_key", "status"} or dtype == "string":
        return "category"
    if dtype == "bool":
        return "boolean"
    if unit in (_UNIT_USD, _UNIT_USD_PER_SHARE):
        return "amount"
    if unit == _UNIT_RATIO:
        return "ratio"
    return "numeric"


def _f(
    name,
    table,
    source,
    *,
    dtype="float64",
    unit=_UNIT_RATIO,
    source_unit=None,
    aliases=(),
    role="feature",
    description="",
    adjustment=None,
    temporal_model=None,
    cardinality=None,
    strict_pit_allowed=None,
    mining_allowed=True,
    grain="instrument_time",
    allowed_operator_families=(),
    required_filters=None,
    metadata=None,
    current_snapshot_only=None,
):
    table_spec = _TABLE_BY_NAME[table]
    # R17-014: table-level required parameters (timeframe / IndexName / ...) are
    # the authoritative filter contract; every mineable field inherits them.
    effective_required = (
        tuple(required_filters)
        if required_filters is not None
        else table_spec.required_parameters
    )
    return FieldSpec(
        name=name,
        table=table,
        source_name=source,
        dtype=dtype,
        unit=unit,
        source_unit=source_unit,
        frequency=table_spec.frequency,
        aliases=tuple(aliases),
        role=role,
        description=description,
        dataset=table_spec.dataset,
        adjustment=adjustment,
        domain=table_spec.domain,
        value_kind=_value_kind(dtype, unit, role),
        temporal_model=temporal_model or table_spec.join_policy,
        grain=grain,
        cardinality=cardinality or table_spec.cardinality,
        time_column=table_spec.time_column,
        instrument_column=table_spec.instrument_column,
        knowledge_time_column=table_spec.knowledge_time_column,
        effective_time_column=table_spec.effective_time_column,
        period_id_column=table_spec.period_id_column,
        strict_pit_allowed=(
            table_spec.strict_pit_allowed
            if strict_pit_allowed is None
            else bool(strict_pit_allowed)
        ),
        mining_allowed=mining_allowed,
        allowed_operator_families=tuple(allowed_operator_families),
        required_filters=effective_required,
        metadata=dict(metadata or {}),
    )


US_FIELD_SPECS: tuple[FieldSpec, ...] = (
    # --- StockDailyBar -----------------------------------------------------
    _f("trade_date", "StockDailyBar", "TradeDate", dtype="date", unit=_UNIT_DATE,
       role="time", aliases=("date",)),
    _f("symbol", "StockDailyBar", "Ticker", dtype="string", unit=_UNIT_IDENTIFIER,
       role="instrument", aliases=("ticker",)),
    _f("open", "StockDailyBar", "Open", unit=_UNIT_USD_PER_SHARE,
       metadata={"adjusted": False, "adjustment_status": "raw"}),
    _f("high", "StockDailyBar", "High", unit=_UNIT_USD_PER_SHARE,
       metadata={"adjusted": False, "adjustment_status": "raw"}),
    _f("low", "StockDailyBar", "Low", unit=_UNIT_USD_PER_SHARE,
       metadata={"adjusted": False, "adjustment_status": "raw"}),
    _f("close", "StockDailyBar", "Close", unit=_UNIT_USD_PER_SHARE,
       metadata={"adjusted": False, "adjustment_status": "raw"}),
    _f("pre_close", "StockDailyBar", "PreClose", unit=_UNIT_USD_PER_SHARE,
       aliases=("prev_close",), metadata={"adjusted": True, "adjustment_status": "factor_adjusted"}),
    _f("volume", "StockDailyBar", "Volume", unit=_UNIT_SHARE,
       metadata={"adjusted": False, "adjustment_status": "raw"}),
    # US Ret is DECIMAL. Do NOT /10000 (A-share Return is bp).
    _f("ret", "StockDailyBar", "Ret", unit=_UNIT_RATIO, aliases=("return", "returns"),
       metadata={"unit_note": "decimal; A-share Return is bp (divide by 10000)"}),
    _f("ret_intra", "StockDailyBar", "Ret_Intra", unit=_UNIT_RATIO, aliases=("intraday_return",)),
    _f("ret_overnight", "StockDailyBar", "Ret_Overnight", unit=_UNIT_RATIO, aliases=("overnight_return",)),
    _f("high_low_ratio", "StockDailyBar", "High_Low_Ratio", unit=_UNIT_RATIO),
    _f("upper_shadow_ratio", "StockDailyBar", "Upper_Shadow_Ratio", unit=_UNIT_RATIO),
    _f("vwap_close_dist", "StockDailyBar", "Vwap_Close_Dist", unit=_UNIT_RATIO),
    _f("adj_factor", "StockDailyBar", "AdjFactor", unit=_UNIT_RATIO, aliases=("factor",),
       metadata={"direction": "backward_multiplier", "note": "adjusted = raw * adj_factor"}),
    _f("vwap", "StockDailyBar", "VWAP", unit=_UNIT_USD_PER_SHARE, aliases=("VWAP",),
       metadata={"adjusted": False, "adjustment_status": "raw"}),
    _f("amount", "StockDailyBar", "Amount", unit=_UNIT_USD, aliases=("turnover_value",),
       metadata={"identity": "Amount == VWAP * Volume when both non-null (~96.65% rows)"}),

    # --- StockList (universe snapshot, type=CS) ----------------------------
    _f("name", "StockList", "name", dtype="string", unit=_UNIT_TEXT, role="label", mining_allowed=False),
    _f("security_type", "StockList", "type", dtype="string", unit=_UNIT_TEXT, role="label", mining_allowed=False),
    _f("market", "StockList", "market", dtype="string", unit=_UNIT_TEXT, role="label", mining_allowed=False),
    _f("delisted_utc", "StockList", "delisted_utc", dtype="datetime", unit=_UNIT_DATETIME, role="ingestion_time", mining_allowed=False),

    # --- SecurityMaster ----------------------------------------------------
    _f("cik", "SecurityMaster", "cik", dtype="string", unit=_UNIT_IDENTIFIER, role="identifier", mining_allowed=False),
    _f("composite_figi", "SecurityMaster", "composite_figi", dtype="string", unit=_UNIT_IDENTIFIER, role="identifier", mining_allowed=False),
    _f("is_adr_asset", "SecurityMaster", "is_adr_asset", dtype="bool", unit=_UNIT_BOOLEAN, role="status", mining_allowed=False),

    # --- TickerSharesSnapshot (recommended US market cap source) -----------
    _f("weighted_shares_outstanding", "TickerSharesSnapshot", "weighted_shares_outstanding",
       unit=_UNIT_SHARE, aliases=("shares_out",), metadata={"coverage": "~42% of StockDailyBar tickers", "source": "shares_pit_derived"}),
    _f("share_class_shares_outstanding", "TickerSharesSnapshot", "share_class_shares_outstanding", unit=_UNIT_SHARE),

    # --- StockIndicesComponents (membership only; NO weight) ---------------
    _f("index_name", "StockIndicesComponents", "IndexName", dtype="string", unit=_UNIT_TEXT,
       role="identifier", aliases=("index",), mining_allowed=False),
    _f("index_member", "StockIndicesComponents", "Symbol", dtype="string", unit=_UNIT_IDENTIFIER,
       role="identifier", mining_allowed=False),

    # --- R17-019: US capital split events vs shares PIT vs snapshot --------
    # Split/adjustment events live in ``us_stock_capital_split`` (event rows).
    _f("split_from", "USStockCapitalSplitEvent", "split_from", unit=_UNIT_RATIO, mining_allowed=False),
    _f("split_to", "USStockCapitalSplitEvent", "split_to", unit=_UNIT_RATIO, mining_allowed=False),
    _f("adjustment_type", "USStockCapitalSplitEvent", "adjustment_type", dtype="string",
       unit=_UNIT_TEXT, role="label", aliases=("split_type",), mining_allowed=False),
    _f("execution_date", "USStockCapitalSplitEvent", "execution_date", dtype="date", unit=_UNIT_DATE,
       role="effective_time", mining_allowed=False),
    # Sparse PIT shares live in ``us_stock_capital_shares`` (asof filing_date).
    _f("pit_basic_shares_outstanding", "USTickerSharesPITEvent", "pit_basic_shares_outstanding",
       unit=_UNIT_SHARE, aliases=("basic_shares_pit",), mining_allowed=False),
    _f("pit_diluted_shares_outstanding", "USTickerSharesPITEvent", "pit_diluted_shares_outstanding",
       unit=_UNIT_SHARE, aliases=("diluted_shares_pit",), mining_allowed=False),

    # --- Financial statements (PIT on filing_date, timeframe filtered) -----
    _f("filing_date", "StockIncome", "filing_date", dtype="date", unit=_UNIT_DATE,
       role="knowledge_time", mining_allowed=False),
    _f("timeframe", "StockIncome", "timeframe", dtype="string", unit=_UNIT_TEXT,
       role="label", aliases=("reporting_timeframe",), mining_allowed=False),
    _f("revenue", "StockIncome", "revenue", unit=_UNIT_USD, aliases=("operating_revenue",)),
    _f("cost_of_revenue", "StockIncome", "cost_of_revenue", unit=_UNIT_USD),
    _f("gross_profit", "StockIncome", "gross_profit", unit=_UNIT_USD),
    _f("operating_income", "StockIncome", "operating_income", unit=_UNIT_USD, aliases=("operating_profit",)),
    _f("net_income_common", "StockIncome", "net_income_loss_attributable_common_shareholders",
       unit=_UNIT_USD, aliases=("net_income", "net_profit")),
    _f("basic_eps", "StockIncome", "basic_earnings_per_share", unit=_UNIT_USD_PER_SHARE, aliases=("eps",)),
    _f("diluted_eps", "StockIncome", "diluted_earnings_per_share", unit=_UNIT_USD_PER_SHARE),
    _f("research_development", "StockIncome", "research_development", unit=_UNIT_USD, aliases=("rd_expenses",)),
    _f("total_assets", "StockBalance", "total_assets", unit=_UNIT_USD),
    _f("total_liabilities", "StockBalance", "total_liabilities", unit=_UNIT_USD),
    _f("total_equity", "StockBalance", "total_equity", unit=_UNIT_USD, aliases=("equity",)),
    _f("equity_attributable_to_parent", "StockBalance", "total_equity_attributable_to_parent", unit=_UNIT_USD),
    _f("total_current_assets", "StockBalance", "total_current_assets", unit=_UNIT_USD, aliases=("current_assets",)),
    _f("total_current_liabilities", "StockBalance", "total_current_liabilities", unit=_UNIT_USD, aliases=("current_liabilities",)),
    _f("cash_and_equivalents", "StockBalance", "cash_and_equivalents", unit=_UNIT_USD, aliases=("cash",)),
    _f("receivables", "StockBalance", "receivables", unit=_UNIT_USD, aliases=("account_receivable",)),
    _f("inventories", "StockBalance", "inventories", unit=_UNIT_USD, aliases=("inventory",)),
    _f("accounts_payable", "StockBalance", "accounts_payable", unit=_UNIT_USD),
    _f("goodwill", "StockBalance", "goodwill", unit=_UNIT_USD),
    _f("retained_earnings", "StockBalance", "retained_earnings_deficit", unit=_UNIT_USD),
    _f("net_cash_from_operating_activities", "StockCashFlow", "net_cash_from_operating_activities",
       unit=_UNIT_USD, aliases=("operating_cash_flow", "ocf")),
    _f("net_cash_from_investing_activities", "StockCashFlow", "net_cash_from_investing_activities",
       unit=_UNIT_USD, aliases=("investing_cash_flow",)),
    _f("net_cash_from_financing_activities", "StockCashFlow", "net_cash_from_financing_activities",
       unit=_UNIT_USD, aliases=("financing_cash_flow",)),
    _f("purchase_of_ppe", "StockCashFlow", "purchase_of_property_plant_and_equipment",
       unit=_UNIT_USD, aliases=("capex",)),

    # --- StockDividend (declaration_date PIT; currency varies) -------------
    _f("declaration_date", "StockDividend", "declaration_date", dtype="date", unit=_UNIT_DATE,
       role="knowledge_time", mining_allowed=False),
    _f("ex_dividend_date", "StockDividend", "ex_dividend_date", dtype="date", unit=_UNIT_DATE,
       role="effective_time", mining_allowed=False),
    # R17-021: cash_amount is *per-share in the declared currency*, NOT a bare
    # USD amount.  Only when currency==USD may it map to cash_dividend_per_share@us.
    _f("cash_amount", "StockDividend", "cash_amount", unit=_UNIT_USD_PER_SHARE,
       aliases=("cash_dividend_per_share_declared_currency",),
       metadata={"note": "amount per share in declared currency; currency varies ~12.2% "
                         "non-USD; USD-only usable without FX provider (R17-021)"}),
    _f("dividend_currency", "StockDividend", "currency", dtype="string", unit=_UNIT_TEXT,
       role="label", aliases=("currency",), mining_allowed=False),
    _f("dividend_frequency", "StockDividend", "frequency", dtype="int64", unit="dimensionless",
       role="label", mining_allowed=False),

    # --- StockValuationDaily / StockIndicator (X0 sparse) ------------------
    _f("market_cap", "StockValuationDaily", "market_cap", unit=_UNIT_USD, aliases=("mkt_cap",),
       metadata={"model": "X0 sparse (~49 files)", "coverage": "~16% null on latest sample", "preferred_source": "TickerSharesSnapshot*Close"}),
    _f("price_to_earnings", "StockValuationDaily", "price_to_earnings", unit=_UNIT_RATIO, aliases=("pe_ratio", "pe")),
    _f("price_to_book", "StockValuationDaily", "price_to_book", unit=_UNIT_RATIO, aliases=("pb_ratio", "pb")),
    _f("price_to_sales", "StockValuationDaily", "price_to_sales", unit=_UNIT_RATIO, aliases=("ps_ratio", "ps")),
    # US dividend_yield / ROE / ROA are DECIMAL (do NOT /100).
    _f("dividend_yield", "StockValuationDaily", "dividend_yield", unit=_UNIT_RATIO,
       aliases=("dividend_ratio",), metadata={"unit_note": "decimal; A-share DividendRatio is % (divide by 100)"}),
    _f("return_on_equity", "StockValuationDaily", "return_on_equity", unit=_UNIT_RATIO,
       aliases=("roe",), metadata={"unit_note": "decimal; A-share Roe is % (divide by 100)"}),
    _f("return_on_assets", "StockValuationDaily", "return_on_assets", unit=_UNIT_RATIO,
       aliases=("roa",), metadata={"unit_note": "decimal; A-share Roa is % (divide by 100)"}),

    # --- FactNews (R17-022/023/024) -----------------------------------------
    _f("published_utc", "FactNews", "published_utc", dtype="datetime", unit=_UNIT_DATETIME,
       role="knowledge_time", mining_allowed=False,
       metadata={"timezone": "UTC", "note": "tz-aware UTC; convert to US session "
                                            "time only for decision mapping"}),
    _f("news_tickers", "FactNews", "tickers", dtype="string", unit=_UNIT_TEXT,
       role="identifier", mining_allowed=False,
       metadata={"note": "list<string>; NewsEventAdapter must explode before "
                         "per-ticker daily factor construction (R17-023)"}),
    # R17-022: the real FactNews physical column is ``title`` (not ``headline``).
    _f("news_title", "FactNews", "title", dtype="string", unit=_UNIT_TEXT,
       role="label", aliases=("headline",), mining_allowed=False,
       metadata={"note": "physical column is title; headline is an alias only (R17-022)"}),
    # News article id — kept for article-level dedupe across exploded tickers.
    _f("news_article_id", "FactNews", "id", dtype="string", unit=_UNIT_TEXT,
       role="identifier", mining_allowed=False),
)

# Fields that exist in the US dictionary but are EMPTY / unusable — used by the
# capability resolver to mark US status (never silently "borrow" a similar field).
US_UNAVAILABLE_PHYSICAL = frozenset(
    {
        "StockStatus.*",        # 0 rows
        "StockIndustry.*",      # 0 rows
        "TickerAlias.*",        # 0 rows
        "StockValuationDaily.Volatility_20d",  # 100% null
    }
)


def us_registry_fields() -> tuple[FieldSpec, ...]:
    return US_FIELD_SPECS


__all__ = ["US_FIELD_SPECS", "US_TABLE_SPECS"]
