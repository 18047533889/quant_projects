"""Built-in A-share daily, fiscal, relation, and reference field catalog."""
from __future__ import annotations

from .spec import FieldSpec, TableSpec
from .units import (
    UNIT_BASIS_POINT,
    UNIT_BOOLEAN,
    UNIT_CNY,
    UNIT_DATE,
    UNIT_DATETIME,
    UNIT_DIMENSIONLESS,
    UNIT_IDENTIFIER,
    UNIT_PERCENT,
    UNIT_RATIO,
    UNIT_SHARE,
    UNIT_TEXT,
)


def _table(
    name,
    dataset,
    *,
    time="TradeDate",
    instrument="Symbol",
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
    dedupe_keys=(),
    revision_order=(),
    current_snapshot_only=False,
    cardinality="many_to_one",
    strict_pit_allowed=True,
):
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
        dedupe_keys=tuple(dedupe_keys),
        revision_order=tuple(revision_order),
        current_snapshot_only=current_snapshot_only,
        cardinality=cardinality,
        strict_pit_allowed=strict_pit_allowed,
    )


ASHARE_TABLE_SPECS: tuple[TableSpec, ...] = (
    _table("StockDailyBar", "ashare_stock_daily", domain="price_volume"),
    _table(
        "StockMinuteBar", "ashare_stock_minute", time="QuoteTime",
        domain="price_volume", frequency="minute", table_kind="minute_session",
        join_policy="minute_session", timezone="Asia/Shanghai",
        session_calendar="ashare",
    ),
    _table("StockValuationDaily", "ashare_stock_valuation_daily", domain="valuation"),
    _table("StockCapitalDaily", "ashare_stock_capital_daily", domain="capital", join_policy="state_asof"),
    _table(
        "StockIndicator", "ashare_stock_indicator", time="PubDate", domain="fundamental",
        table_kind="financial_event", join_policy="financial_pit", knowledge="PubDate",
        period="ReportPeriodEndDate", revision="UpdateTime",
    ),
    _table(
        "StockBalance", "ashare_stock_balance", time="PubDate", domain="fundamental",
        table_kind="financial_event", join_policy="financial_pit", knowledge="PubDate",
        period="ReportPeriodEndDate", revision="UpdateTime",
    ),
    _table(
        "StockIncome", "ashare_stock_income", time="PubDate", domain="fundamental",
        table_kind="financial_event", join_policy="financial_pit", knowledge="PubDate",
        period="ReportPeriodEndDate", revision="UpdateTime",
    ),
    _table(
        "StockCashFlow", "ashare_stock_cashflow", time="PubDate", domain="fundamental",
        aliases=("StockCashflow",), table_kind="financial_event",
        join_policy="financial_pit", knowledge="PubDate", period="ReportPeriodEndDate",
        revision="UpdateTime",
    ),
    _table(
        "StockDividend", "ashare_stock_dividend", time="PubDate", domain="corporate_action",
        table_kind="effective_event", join_policy="effective_only", knowledge="PubDate",
        effective="ExDate", strict_pit_allowed=False,
    ),
    _table(
        "IndexDailyBar", "ashare_index_daily", instrument="IndexSymbol", domain="index",
        aliases=("BenchmarkIndexDailyBar",), join_policy="exact_date",
        required_parameters=("IndexSymbol",),
    ),
    _table(
        "IndexConstituent", "ashare_index_constituent", domain="index",
        table_kind="relation", join_policy="exact", required_parameters=("IndexSymbol",),
        cardinality="one_to_many",
    ),
    _table("EtfDailyBar", "ashare_etf_daily", domain="etf", aliases=("ETFDailyBar",)),
    _table("Calendar", "ashare_calendar", time="Date", instrument=None, domain="calendar", table_kind="calendar"),
    _table("StockList", "ashare_stock_list", time=None, instrument=None, domain="reference", current_snapshot_only=True),
    _table("EtfList", "ashare_etf_list", time=None, instrument=None, domain="reference", aliases=("ETFList",), current_snapshot_only=True),
    _table("IndexList", "ashare_index_list", time=None, instrument="IndexSymbol", domain="reference", current_snapshot_only=True),
    _table(
        "StockIndustry", "ashare_stock_industry", time=None, domain="classification",
        table_kind="relation", join_policy="state_asof", required_parameters=("IndustrySource",),
        current_snapshot_only=True,
    ),
    _table("StockStatus", "ashare_stock_status", domain="status", join_policy="state_asof"),
    _table(
        "StockTopTenShareholder", "ashare_stock_topten_shareholder", time="PubDate",
        domain="ownership", table_kind="relation", join_policy="relation_pit",
        knowledge="PubDate", period="ReportPeriodEndDate", cardinality="one_to_many",
    ),
    _table(
        "StockTopTenFloatShareholder", "ashare_stock_topten_float_shareholder",
        time="PubDate", domain="ownership", table_kind="relation",
        join_policy="relation_pit", knowledge="PubDate", period="ReportPeriodEndDate",
        cardinality="one_to_many",
    ),
)

_TABLE_BY_NAME = {item.name: item for item in ASHARE_TABLE_SPECS}


def _value_kind(dtype: str, unit: str, role: str) -> str:
    if role in {"time", "knowledge_time", "effective_time", "period_id"}:
        return "datetime" if dtype == "datetime" else "date"
    if role in {"instrument", "identifier"}:
        return "identifier"
    if role in {"label", "group_key", "status"} or dtype == "string":
        return "category"
    if dtype == "bool":
        return "boolean"
    if unit == UNIT_CNY:
        return "amount"
    if unit in {UNIT_RATIO, UNIT_PERCENT, UNIT_BASIS_POINT}:
        return "ratio"
    return "numeric"


def _f(
    name,
    table,
    source,
    *,
    dtype="float64",
    unit=UNIT_DIMENSIONLESS,
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
):
    table_spec = _TABLE_BY_NAME[table]
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
    )


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
    _f("ret", "StockDailyBar", "Return", unit=UNIT_RATIO, source_unit=UNIT_BASIS_POINT, aliases=("return", "returns")),
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
    _f("turnover_ratio", "StockValuationDaily", "TurnoverRatio", unit=UNIT_RATIO, source_unit=UNIT_PERCENT, aliases=("turnover",)),
    _f("dividend_yield", "StockValuationDaily", "DividendRatio", unit=UNIT_RATIO, source_unit=UNIT_PERCENT, aliases=("dividend_ratio",)),
    _f("free_market_cap", "StockValuationDaily", "FreeMarketCap", unit=UNIT_CNY),

    _f("total_capital", "StockCapitalDaily", "TotalCapital", unit=UNIT_SHARE, aliases=("total_shares",)),
    _f("circulating_capital", "StockCapitalDaily", "CirculatingCapital", unit=UNIT_SHARE, aliases=("float_shares",)),
    _f("eps", "StockIndicator", "Eps", unit=UNIT_CNY),
    _f("roe", "StockIndicator", "Roe", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("pub_date", "StockIndicator", "PubDate", dtype="date", unit=UNIT_DATE, role="knowledge_time", mining_allowed=False),
    _f("report_period_end_date", "StockIndicator", "ReportPeriodEndDate", dtype="date", unit=UNIT_DATE, role="period_id", aliases=("report_date",), mining_allowed=False),

    # StockIncome 物理列名与 COS parquet 逐列核对（2026-08）。
    _f("total_operating_revenue", "StockIncome", "TotalOperatingRevenue", unit=UNIT_CNY, grain="flow_ytd"),
    _f("operating_revenue", "StockIncome", "OperatingRevenue", unit=UNIT_CNY, aliases=("revenue",), grain="flow_ytd"),
    _f("operating_cost", "StockIncome", "OperatingCost", unit=UNIT_CNY, aliases=("cost_of_goods_sold", "cogs"), grain="flow_ytd"),
    _f("operating_profit", "StockIncome", "OperatingProfit", unit=UNIT_CNY, aliases=("operating_income",), grain="flow_ytd"),
    _f("total_profit", "StockIncome", "TotalProfit", unit=UNIT_CNY, grain="flow_ytd"),
    _f("income_tax_expense", "StockIncome", "IncomeTaxExpense", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_profit", "StockIncome", "NetProfit", unit=UNIT_CNY, aliases=("net_income",), grain="flow_ytd"),
    _f("np_parent_company_owners", "StockIncome", "NpParentCompanyOwners", unit=UNIT_CNY, grain="flow_ytd"),
    _f("selling_expense", "StockIncome", "SaleExpense", unit=UNIT_CNY, grain="flow_ytd"),
    _f("administration_expense", "StockIncome", "AdministrationExpense", unit=UNIT_CNY, grain="flow_ytd"),
    _f("financial_expense", "StockIncome", "FinancialExpense", unit=UNIT_CNY, aliases=("interest_expense",), grain="flow_ytd"),
    _f("rd_expenses", "StockIncome", "RdExpenses", unit=UNIT_CNY, aliases=("research_development", "rd"), grain="flow_ytd"),

    # StockBalance 物理列名与 COS parquet 逐列核对（2026-08）。
    _f("total_assets", "StockBalance", "TotalAssets", unit=UNIT_CNY, grain="balance"),
    _f("total_liabilities", "StockBalance", "TotalLiability", unit=UNIT_CNY, grain="balance"),
    _f("shareholders_equity", "StockBalance", "EquitiesParentCompanyOwners", unit=UNIT_CNY, aliases=("equity",), grain="balance"),
    _f("total_owner_equities", "StockBalance", "TotalOwnerEquities", unit=UNIT_CNY, aliases=("total_equity",), grain="balance"),
    _f("current_assets", "StockBalance", "TotalCurrentAssets", unit=UNIT_CNY, grain="balance"),
    _f("current_liabilities", "StockBalance", "TotalCurrentLiability", unit=UNIT_CNY, grain="balance"),
    _f("cash_equivalents", "StockBalance", "CashEquivalents", unit=UNIT_CNY, aliases=("cash",), grain="balance"),
    _f("trading_assets", "StockBalance", "TradingAssets", unit=UNIT_CNY, grain="balance"),
    _f("bill_receivable", "StockBalance", "BillReceivable", unit=UNIT_CNY, grain="balance"),
    _f("account_receivable", "StockBalance", "AccountReceivable", unit=UNIT_CNY, aliases=("receivables", "accounts_receivable"), grain="balance"),
    _f("receivable_fin", "StockBalance", "ReceivableFin", unit=UNIT_CNY, grain="balance"),
    _f("prepayments", "StockBalance", "AdvancePayment", unit=UNIT_CNY, grain="balance"),
    _f("inventory", "StockBalance", "Inventories", unit=UNIT_CNY, aliases=("inventories",), grain="balance"),
    _f("contract_assets", "StockBalance", "ContractAssets", unit=UNIT_CNY, grain="balance"),
    _f("fixed_assets", "StockBalance", "FixedAssets", unit=UNIT_CNY, grain="balance"),
    _f("construction_in_progress", "StockBalance", "ConstruInProcess", unit=UNIT_CNY, grain="balance"),
    _f("intangible_assets", "StockBalance", "IntangibleAssets", unit=UNIT_CNY, grain="balance"),
    _f("usufruct_assets", "StockBalance", "UsufructAssets", unit=UNIT_CNY, grain="balance"),
    _f("goodwill", "StockBalance", "GoodWill", unit=UNIT_CNY, grain="balance"),
    _f("long_term_equity_investment", "StockBalance", "LongtermEquityInvest", unit=UNIT_CNY, grain="balance"),
    _f("deferred_tax_assets", "StockBalance", "DeferredTaxAssets", unit=UNIT_CNY, grain="balance"),
    _f("shortterm_loan", "StockBalance", "ShorttermLoan", unit=UNIT_CNY, aliases=("short_debt",), grain="balance"),
    _f("notes_payable", "StockBalance", "NotesPayable", unit=UNIT_CNY, grain="balance"),
    _f("accounts_payable", "StockBalance", "AccountsPayable", unit=UNIT_CNY, grain="balance"),
    _f("contract_liabilities", "StockBalance", "ContractLiability", unit=UNIT_CNY, grain="balance"),
    _f("employee_payable", "StockBalance", "SalariesPayable", unit=UNIT_CNY, grain="balance"),
    _f("taxes_payable", "StockBalance", "TaxsPayable", unit=UNIT_CNY, grain="balance"),
    _f("longterm_loan", "StockBalance", "LongtermLoan", unit=UNIT_CNY, aliases=("long_debt",), grain="balance"),
    _f("bonds_payable", "StockBalance", "BondsPayable", unit=UNIT_CNY, grain="balance"),
    _f("lease_liability", "StockBalance", "LeaseLiability", unit=UNIT_CNY, grain="balance"),
    _f("deferred_tax_liabilities", "StockBalance", "DeferredTaxLiability", unit=UNIT_CNY, grain="balance"),
    _f("minority_interest", "StockBalance", "MinorityInterests", unit=UNIT_CNY, grain="balance"),

    # StockCashFlow 物理列名与 COS parquet 逐列核对（2026-08）。
    _f("operating_cash_flow", "StockCashFlow", "NetOperateCashFlow", unit=UNIT_CNY, aliases=("ocf",), grain="flow_ytd"),
    _f("investing_cash_flow", "StockCashFlow", "NetInvestCashFlow", unit=UNIT_CNY, grain="flow_ytd"),
    _f("financing_cash_flow", "StockCashFlow", "NetFinanceCashFlow", unit=UNIT_CNY, grain="flow_ytd"),
    _f("fix_intan_other_asset_acquis_cash", "StockCashFlow", "FixIntanOtherAssetAcquiCash", unit=UNIT_CNY, aliases=("capex",), grain="flow_ytd"),
    _f("cash_dividend", "StockDividend", "CashDividend", unit=UNIT_CNY, temporal_model="effective_only", strict_pit_allowed=False),

    # StockIndicator 物理列名与 COS parquet 逐列核对（2026-08）。
    _f("adjusted_profit", "StockIndicator", "AdjustedProfit", unit=UNIT_CNY, grain="flow_ytd"),

    _f("pe_ratio_lyr", "StockValuationDaily", "PeRatioLyr", aliases=("pe_lyr",)),
    _f("pcf_ratio_2", "StockValuationDaily", "PcfRatio2", aliases=("pcf2",)),

    _f("industry_code", "StockIndustry", "IndustryCode", dtype="string", unit=UNIT_IDENTIFIER, role="group_key", mining_allowed=False),
    _f("industry_name", "StockIndustry", "IndustryName", dtype="string", unit=UNIT_TEXT, role="label", mining_allowed=False),
    _f("public_status", "StockStatus", "PublicStatus", dtype="string", unit=UNIT_TEXT, role="status", aliases=("listed_state",), mining_allowed=False),
    _f("index_symbol", "IndexConstituent", "IndexSymbol", dtype="string", unit=UNIT_IDENTIFIER, role="identifier", aliases=("index_code",), mining_allowed=False),
    _f("index_weight", "IndexConstituent", "Weight", unit=UNIT_RATIO, source_unit=UNIT_PERCENT, aliases=("weight",), cardinality="one_to_many", mining_allowed=False),
    _f("share_ratio", "StockTopTenShareholder", "ShareRatio", unit=UNIT_RATIO, source_unit=UNIT_PERCENT, cardinality="one_to_many", mining_allowed=False),
)


__all__ = ["ASHARE_FIELD_SPECS", "ASHARE_TABLE_SPECS"]
