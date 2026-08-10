"""Built-in A-share daily, fiscal, relation, and reference field catalog."""
from __future__ import annotations

import enum

from .spec import FieldSpec, TableSpec
from .units import (
    UNIT_BASIS_POINT,
    UNIT_BOOLEAN,
    UNIT_CNY,
    UNIT_CNY_PER_SHARE,
    UNIT_DATE,
    UNIT_DATETIME,
    UNIT_DIMENSIONLESS,
    UNIT_IDENTIFIER,
    UNIT_PERCENT,
    UNIT_RATIO,
    UNIT_SHARE,
    UNIT_SHARE_RATIO,
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
    strict_pit_allowed=None,
    metadata=None,
):
    # R17-013: the helper default is UNKNOWN (None), matching the TableSpec
    # default.  A table whose PIT eligibility was never declared must NOT
    # silently become PIT-safe; every production-readable table declares
    # True/False explicitly (+ reason in metadata).
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
        metadata=dict(metadata or {}),
    )


ASHARE_TABLE_SPECS: tuple[TableSpec, ...] = (
    # R17-013: every production-readable table declares strict_pit_allowed
    # explicitly (never the UNKNOWN default); reason lives in metadata.
    _table("StockDailyBar", "ashare_stock_daily", domain="price_volume",
           strict_pit_allowed=True,
           metadata={"pit_reason": "exact_daily D1 panel; PIT-safe"}),
    _table(
        "StockMinuteBar", "ashare_stock_minute", time="QuoteTime",
        domain="price_volume", frequency="minute", table_kind="minute_session",
        join_policy="minute_session", timezone="Asia/Shanghai",
        session_calendar="ashare", strict_pit_allowed=True,
        # Round-7 WS-E #293: the A-share COS minute mirror labels the one-minute
        # bar with its END timestamp (09:31 = the 09:30-09:31 bar, last 15:00).
        # Declared here so the runtime session slotting never guesses from a
        # 09:30/13:00 heuristic.
        metadata={"bar_timestamp_role": "bar_end", "bar_timestamp_convention": "bar_end",
                  "pit_reason": "exact minute_session; PIT-safe"},
    ),
    _table("StockValuationDaily", "ashare_stock_valuation_daily", domain="valuation",
           strict_pit_allowed=True,
           metadata={"pit_reason": "exact_daily D1; PIT-safe"}),
    _table("StockCapitalDaily", "ashare_stock_capital_daily", domain="capital",
           join_policy="state_asof", strict_pit_allowed=True,
           metadata={"pit_reason": "S1 snapshot with explicit max_staleness policy (R17-071); exact when fresh"}),
    # Financial tables: knowledge = PubDate (announcement day), period =
    # ReportPeriodEndDate.  revision = UpdateTime is the COS write timestamp used
    # ONLY as a deterministic in-snapshot tiebreaker (never as a historical
    # revision-effective time — re-syncs rewrite UpdateTime; COS lqtp dict §4.21).
    _table(
        "StockIndicator", "ashare_stock_indicator", time="PubDate", domain="fundamental",
        table_kind="financial_event", join_policy="financial_pit", knowledge="PubDate",
        period="ReportPeriodEndDate", revision="UpdateTime", strict_pit_allowed=True,
        metadata={"pit_reason": "financial_pit on PubDate; PIT-safe"},
    ),
    _table(
        "StockBalance", "ashare_stock_balance", time="PubDate", domain="fundamental",
        table_kind="financial_event", join_policy="financial_pit", knowledge="PubDate",
        period="ReportPeriodEndDate", revision="UpdateTime", strict_pit_allowed=True,
        metadata={"pit_reason": "financial_pit on PubDate; PIT-safe"},
    ),
    _table(
        "StockIncome", "ashare_stock_income", time="PubDate", domain="fundamental",
        table_kind="financial_event", join_policy="financial_pit", knowledge="PubDate",
        period="ReportPeriodEndDate", revision="UpdateTime", strict_pit_allowed=True,
        metadata={"pit_reason": "financial_pit on PubDate; PIT-safe"},
    ),
    _table(
        "StockCashFlow", "ashare_stock_cashflow", time="PubDate", domain="fundamental",
        aliases=("StockCashflow",), table_kind="financial_event",
        join_policy="financial_pit", knowledge="PubDate", period="ReportPeriodEndDate",
        revision="UpdateTime", strict_pit_allowed=True,
        metadata={"pit_reason": "financial_pit on PubDate; PIT-safe"},
    ),
    # StockDividend physical schema (COS lqtp dict §4.17): TradeDate/RightRegDate/
    # ExDividendDate/CashDividend/StockDividend/StockTransfer/UpdateTime.  There is
    # NO PubDate and NO announcement time; the partition date == ExDividendDate.
    # Use the effective date as both the time key and the only available event time.
    _table(
        "StockDividend", "ashare_stock_dividend", time="TradeDate", domain="corporate_action",
        table_kind="effective_event", join_policy="effective_only", knowledge=None,
        effective="ExDividendDate", strict_pit_allowed=False,
        metadata={"pit_reason": "no announcement time; effective-only (ex-date)"},
    ),
    # IndexDailyBar keys on (TradeDate, Symbol) — Symbol, not IndexSymbol.  Index
    # selection must use instrument_filter (e.g. ["000300.SH"]), not a parameter.
    _table(
        "IndexDailyBar", "ashare_index_daily", instrument="Symbol", domain="index",
        aliases=("BenchmarkIndexDailyBar",), join_policy="exact_date",
        strict_pit_allowed=True, metadata={"pit_reason": "exact_date; PIT-safe"},
    ),
    _table(
        "IndexConstituent", "ashare_index_constituent", domain="index",
        table_kind="relation", join_policy="exact", required_parameters=("IndexSymbol",),
        cardinality="one_to_many", strict_pit_allowed=True,
        metadata={"pit_reason": "exact_daily relation; PIT-safe; IndexSymbol required"},
    ),
    _table("EtfDailyBar", "ashare_etf_daily", domain="etf", aliases=("ETFDailyBar",),
           strict_pit_allowed=True, metadata={"pit_reason": "exact_daily; PIT-safe"}),
    # Calendar stores natural days in TradeDate (COS lqtp dict §Calendar), not "Date".
    _table("Calendar", "ashare_calendar", time="TradeDate", instrument=None, domain="calendar", table_kind="calendar",
           strict_pit_allowed=True, metadata={"pit_reason": "calendar reference; PIT-safe"}),
    # List tables are D1 natural-day snapshots with full history (incl. delisted
    # symbols).  Marking them current_snapshot_only forced strict_pit_allowed=False
    # and allowed survivorship-biased backfill; with full history they are exact
    # equi-join PIT-safe tables.
    _table("StockList", "ashare_stock_list", time="TradeDate", instrument="Symbol", domain="reference", join_policy="exact",
           strict_pit_allowed=True, metadata={"pit_reason": "exact full-history; PIT-safe"}),
    _table("EtfList", "ashare_etf_list", time="TradeDate", instrument="Symbol", domain="reference", aliases=("ETFList",), join_policy="exact",
           strict_pit_allowed=True, metadata={"pit_reason": "exact full-history; PIT-safe"}),
    _table("IndexList", "ashare_index_list", time="TradeDate", instrument="Symbol", domain="reference", join_policy="exact",
           strict_pit_allowed=True, metadata={"pit_reason": "exact full-history; PIT-safe"}),
    # StockIndustry is a natural-day snapshot per (TradeDate, Symbol, IndustrySource);
    # with full history an exact equi join on TradeDate is PIT-safe and avoids the
    # "current industry viewed into the past" lookahead.  Single source still required.
    _table(
        "StockIndustry", "ashare_stock_industry", time="TradeDate", domain="classification",
        table_kind="relation", join_policy="exact", required_parameters=("IndustrySource",),
        strict_pit_allowed=True,
        metadata={"pit_reason": "exact_daily relation; PIT-safe; IndustrySource required"},
    ),
    _table("StockStatus", "ashare_stock_status", domain="status", join_policy="state_asof",
           strict_pit_allowed=True,
           metadata={"pit_reason": "S1 daily snapshot; exact join when fresh, carry age recorded (R17-072)"}),
    # R17-062: A-share TopTen* are S1 snapshots keyed by (TradeDate, Symbol, Rank)
    # with PubDate/ReportPeriod metadata.  time_column = TradeDate matches the
    # DataAccess partition key; knowledge = PubDate is the visibility date.
    _table(
        "StockTopTenShareholder", "ashare_stock_topten_shareholder", time="TradeDate",
        domain="ownership", table_kind="relation", join_policy="relation_pit",
        knowledge="PubDate", period="ReportPeriodEndDate", cardinality="one_to_many",
        strict_pit_allowed=True,
        metadata={"pit_reason": "S1 snapshot keyed by TradeDate; relation_pit on "
                                "PubDate visibility; PIT-safe (R17-062)"},
    ),
    _table(
        "StockTopTenFloatShareholder", "ashare_stock_topten_float_shareholder",
        time="TradeDate", domain="ownership", table_kind="relation",
        join_policy="relation_pit", knowledge="PubDate", period="ReportPeriodEndDate",
        cardinality="one_to_many", strict_pit_allowed=True,
        metadata={"pit_reason": "S1 snapshot keyed by TradeDate; relation_pit on "
                                "PubDate visibility; PIT-safe (R17-062)"},
    ),
)

_TABLE_BY_NAME = {item.name: item for item in ASHARE_TABLE_SPECS}


class FieldRole(enum.Enum):
    """Typed field role vocabulary (round-7 WS-E #284).

    ``role`` remains a plain ``str`` on :class:`~fields.spec.FieldSpec` for
    backward compatibility; the catalog validates every declared role against
    this enum at load time so a typo like ``"knowledge_tiem"`` fails the catalog
    instead of silently degrading mining/PIT semantics.
    """

    FEATURE = "feature"
    TIME = "time"
    INSTRUMENT = "instrument"
    LABEL = "label"
    IDENTIFIER = "identifier"
    GROUP_KEY = "group_key"
    STATUS = "status"
    KNOWLEDGE_TIME = "knowledge_time"
    EFFECTIVE_TIME = "effective_time"
    PERIOD_ID = "period_id"
    INGESTION_TIME = "ingestion_time"

    @classmethod
    def values(cls) -> set[str]:
        return {item.value for item in cls}


#: Additional role spellings accepted from the DataAccess catalog (semantic
#: fields may use a slightly wider vocabulary).
_EXTRA_ACCEPTED_ROLES = frozenset({"event_time", "revision_id"})


def validate_field_role(role: str) -> str:
    """Validate a field ``role`` spelling; raise on unknown values.

    Raises ``ValueError`` for unknown roles (e.g. a typo such as
    ``"knowledge_tiem"``) so a broken catalog cannot silently change mining /
    PIT semantics.  Returns the canonical role string on success.
    """
    text = str(role).strip().lower()
    if not text:
        raise ValueError("field role must be non-empty")
    if text in FieldRole.values() or text in _EXTRA_ACCEPTED_ROLES:
        return text
    raise ValueError(
        f"unknown field role {role!r}; expected one of "
        f"{sorted(FieldRole.values())}"
    )


def _value_kind(dtype: str, unit: str, role: str) -> str:
    if role in {"time", "knowledge_time", "effective_time", "period_id", "ingestion_time"}:
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
    price_basis=None,
    flow_semantics=None,
    required_filters=None,
    metadata=None,
):
    table_spec = _TABLE_BY_NAME[table]
    role = validate_field_role(role)
    # R17-014: table-level required parameters (IndustrySource / IndexSymbol /
    # timeframe) are the authoritative filter contract — every mineable field
    # inherits them as required_filters.  A field-level override wins.
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
        price_basis=price_basis,
        flow_semantics=flow_semantics,
        required_filters=effective_required,
        metadata=dict(metadata or {}),
    )


ASHARE_FIELD_SPECS: tuple[FieldSpec, ...] = (
    _f("trade_date", "StockDailyBar", "TradeDate", dtype="date", unit=UNIT_DATE, role="time", aliases=("date",)),
    _f("symbol", "StockDailyBar", "Symbol", dtype="string", unit=UNIT_IDENTIFIER, role="instrument", aliases=("ticker",)),
    # R17-010: the A-share COS contract VERIFIES Factor is a backward cumulative
    # multiplier (continuous_price = raw_price * Factor); raw OHLC/VWAP/limit are
    # RAW basis (unadjusted); continuous prices are exposed via canonical derived
    # concepts (continuous_close = Close * Factor).  "unverified" metadata is gone.
    # R17-012: price fields are CNY per share (dimension CNY/share), not a bare
    # CNY amount; scale_to_canonical stays 1.0 (CNY and CNY/share share the
    # numeric scale — the dimension is what differs).
    _f("open", "StockDailyBar", "Open", unit=UNIT_CNY_PER_SHARE, price_basis="RAW",
       metadata={"adjusted": False, "adjustment_status": "raw", "price_basis": "RAW"}),
    _f("high", "StockDailyBar", "High", unit=UNIT_CNY_PER_SHARE, price_basis="RAW",
       metadata={"adjusted": False, "adjustment_status": "raw", "price_basis": "RAW"}),
    _f("low", "StockDailyBar", "Low", unit=UNIT_CNY_PER_SHARE, price_basis="RAW",
       metadata={"adjusted": False, "adjustment_status": "raw", "price_basis": "RAW"}),
    _f("close", "StockDailyBar", "Close", unit=UNIT_CNY_PER_SHARE, price_basis="RAW",
       metadata={"adjusted": False, "adjustment_status": "raw", "price_basis": "RAW"}),
    _f("pre_close", "StockDailyBar", "PreClose", unit=UNIT_CNY_PER_SHARE, aliases=("prev_close",),
       price_basis="OFFICIAL_REFERENCE_PRE_CLOSE",
       metadata={"adjusted": True, "adjustment_status": "official_reference_pre_close",
                 "price_basis": "OFFICIAL_REFERENCE_PRE_CLOSE", "note": "R17-011 official reference pre-close; not lag(raw_close,1)"}),
    _f("volume", "StockDailyBar", "Volume", unit=UNIT_SHARE, metadata={"adjusted": False, "adjustment_status": "raw"}),
    _f("amount", "StockDailyBar", "Amount", unit=UNIT_CNY, aliases=("turnover_value",)),
    _f("ret", "StockDailyBar", "Return", unit=UNIT_RATIO, source_unit=UNIT_BASIS_POINT, aliases=("return", "returns")),
    _f("adj_factor", "StockDailyBar", "Factor", unit=UNIT_RATIO, aliases=("factor",),
       metadata={"direction": "backward_multiplier",
                 "note": "continuous_price = raw_price * Factor (verified; R17-010)"}),
    _f("vwap", "StockDailyBar", "Vwap", unit=UNIT_CNY_PER_SHARE, price_basis="RAW",
       metadata={"adjusted": False, "adjustment_status": "raw", "price_basis": "RAW"}),
    _f("high_limit", "StockDailyBar", "HighLimit", unit=UNIT_CNY_PER_SHARE, price_basis="RAW_OFFICIAL_LIMIT",
       metadata={"adjusted": False, "adjustment_status": "raw_official_limit", "price_basis": "RAW_OFFICIAL_LIMIT"}),
    _f("low_limit", "StockDailyBar", "LowLimit", unit=UNIT_CNY_PER_SHARE, price_basis="RAW_OFFICIAL_LIMIT",
       metadata={"adjusted": False, "adjustment_status": "raw_official_limit", "price_basis": "RAW_OFFICIAL_LIMIT"}),
    _f("is_suspend", "StockDailyBar", "IsSuspend", dtype="bool", unit=UNIT_BOOLEAN),
    # UpdateTime is the vendor/pipeline write time to COS (freshness only), never a
    # market/announcement/revision-knowable instant — role ingestion_time, not knowledge_time.
    _f("update_time", "StockDailyBar", "UpdateTime", dtype="datetime", unit=UNIT_DATETIME, role="ingestion_time"),

    _f("market_cap", "StockValuationDaily", "MarketCap", unit=UNIT_CNY, aliases=("mkt_cap",)),
    _f("circulating_market_cap", "StockValuationDaily", "CirculatingMarketCap", unit=UNIT_CNY, aliases=("float_market_cap",)),
    _f("pe_ratio", "StockValuationDaily", "PeRatio", aliases=("pe",)),
    _f("pb_ratio", "StockValuationDaily", "PbRatio", aliases=("pb",)),
    _f("ps_ratio", "StockValuationDaily", "PsRatio", aliases=("ps",)),
    _f("pcf_ratio", "StockValuationDaily", "PcfRatio", aliases=("pcf",)),
    _f("turnover_ratio", "StockValuationDaily", "TurnoverRatio", unit=UNIT_RATIO, source_unit=UNIT_PERCENT, aliases=("turnover",)),
    _f("dividend_yield", "StockValuationDaily", "DividendRatio", unit=UNIT_RATIO, source_unit=UNIT_PERCENT, aliases=("dividend_ratio",)),
    _f("free_market_cap", "StockValuationDaily", "FreeMarketCap", unit=UNIT_CNY),
    # Remaining StockValuationDaily caps (COS lqtp dict §StockValuationDaily):
    # Capitalization/CirculatingCap/FreeCap/ACap are share counts, AMarketCap is CNY.
    _f("capitalization", "StockValuationDaily", "Capitalization", unit=UNIT_SHARE, aliases=("total_shares_val",)),
    _f("circulating_cap", "StockValuationDaily", "CirculatingCap", unit=UNIT_SHARE, aliases=("float_shares_val",)),
    _f("free_cap", "StockValuationDaily", "FreeCap", unit=UNIT_SHARE, aliases=("free_float_shares",)),
    _f("a_cap", "StockValuationDaily", "ACap", unit=UNIT_SHARE),
    _f("a_market_cap", "StockValuationDaily", "AMarketCap", unit=UNIT_CNY),

    _f("total_capital", "StockCapitalDaily", "TotalCapital", unit=UNIT_SHARE, aliases=("total_shares",)),
    _f("circulating_capital", "StockCapitalDaily", "CirculatingCapital", unit=UNIT_SHARE, aliases=("float_shares",)),
    # R17-012: EPS is CNY per share (per-share dimension), not a bare CNY amount.
    _f("eps", "StockIndicator", "Eps", unit=UNIT_CNY_PER_SHARE),
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
    # StockDividend effective-only fields (COS lqtp dict §4.17): no announcement PIT.
    # R17-012: CashDividend is CNY per share; StockDividend/StockTransfer are
    # dimensionless per-share RATIOS (proportions), not share counts.
    _f("cash_dividend", "StockDividend", "CashDividend", unit=UNIT_CNY_PER_SHARE, temporal_model="effective_only", strict_pit_allowed=False),
    _f("stock_dividend", "StockDividend", "StockDividend", unit=UNIT_SHARE_RATIO, temporal_model="effective_only", strict_pit_allowed=False),
    _f("stock_transfer", "StockDividend", "StockTransfer", unit=UNIT_SHARE_RATIO, temporal_model="effective_only", strict_pit_allowed=False),
    _f("right_reg_date", "StockDividend", "RightRegDate", dtype="date", unit=UNIT_DATE, temporal_model="effective_only", mining_allowed=False),

    # StockIndicator 物理列名与 COS parquet 逐列核对（2026-08）。
    _f("adjusted_profit", "StockIndicator", "AdjustedProfit", unit=UNIT_CNY, grain="flow_ytd"),

    _f("pe_ratio_lyr", "StockValuationDaily", "PeRatioLyr", aliases=("pe_lyr",)),
    _f("pcf_ratio_2", "StockValuationDaily", "PcfRatio2", aliases=("pcf2",)),

    _f("industry_code", "StockIndustry", "IndustryCode", dtype="string", unit=UNIT_IDENTIFIER, role="group_key", mining_allowed=False),
    _f("industry_name", "StockIndustry", "IndustryName", dtype="string", unit=UNIT_TEXT, role="label", mining_allowed=False),
    _f("public_status", "StockStatus", "PublicStatus", dtype="string", unit=UNIT_TEXT, role="status", aliases=("listed_state",), mining_allowed=False),
    _f("change_date", "StockStatus", "ChangeDate", dtype="date", unit=UNIT_DATE, mining_allowed=False),
    _f("change_type", "StockStatus", "ChangeType", dtype="string", unit=UNIT_TEXT, role="label", mining_allowed=False),
    _f("public_status_code", "StockStatus", "PublicStatusCode", dtype="string", unit=UNIT_IDENTIFIER, mining_allowed=False),
    _f("index_symbol", "IndexConstituent", "IndexSymbol", dtype="string", unit=UNIT_IDENTIFIER, role="identifier", aliases=("index_code",), mining_allowed=False),
    _f("index_weight", "IndexConstituent", "Weight", unit=UNIT_RATIO, source_unit=UNIT_PERCENT, aliases=("weight",), cardinality="one_to_many", mining_allowed=False),
    # Top-ten shareholder fields are one-to-many per (TradeDate, Symbol); aggregated
    # before any mining.  ShareRatio is a percentage in the source (÷100 to ratio).
    _f("share_ratio", "StockTopTenShareholder", "ShareRatio", unit=UNIT_RATIO, source_unit=UNIT_PERCENT, cardinality="one_to_many", mining_allowed=False),
    _f("shareholder_id", "StockTopTenShareholder", "ShareholderId", dtype="string", unit=UNIT_IDENTIFIER, role="identifier", cardinality="one_to_many", mining_allowed=False),
    _f("shareholder_rank", "StockTopTenShareholder", "ShareholderRank", dtype="float64", unit=UNIT_RATIO, role="group_key", cardinality="one_to_many", mining_allowed=False),
    _f("shareholder_name", "StockTopTenShareholder", "ShareholderName", dtype="string", unit=UNIT_TEXT, role="label", cardinality="one_to_many", mining_allowed=False),
    _f("shareholder_class", "StockTopTenShareholder", "ShareholderClass", dtype="string", unit=UNIT_TEXT, role="label", cardinality="one_to_many", mining_allowed=False),
    _f("shares_nature", "StockTopTenShareholder", "SharesNature", dtype="string", unit=UNIT_TEXT, role="label", cardinality="one_to_many", mining_allowed=False),
    _f("share_number", "StockTopTenShareholder", "ShareNumber", unit=UNIT_SHARE, cardinality="one_to_many", mining_allowed=False),
    _f("share_pledge", "StockTopTenShareholder", "SharePledge", unit=UNIT_SHARE, cardinality="one_to_many", mining_allowed=False),
    _f("share_freeze", "StockTopTenShareholder", "ShareFreeze", unit=UNIT_SHARE, cardinality="one_to_many", mining_allowed=False),
    # Report snapshot identity for StockTopTenShareholder (§6.8): concentration
    # trends advance by distinct report period, not by ffilled trading rows.
    _f("report_period_end_date", "StockTopTenShareholder", "ReportPeriodEndDate",
        dtype="date", unit=UNIT_DATE, role="period_id", mining_allowed=False),

    # ----------------------------------------------------------------------
    # StockMinuteBar 字段目录（review §10.1）：source 层与字段层必须共享同一
    # 语义系统。分钟价格未复权，名称用 minute_* 避免与日线裸名冲突。
    # ----------------------------------------------------------------------
    _f("quote_time", "StockMinuteBar", "QuoteTime", dtype="datetime", unit=UNIT_DATETIME, role="time", mining_allowed=False),
    _f("minute_open", "StockMinuteBar", "Open", unit=UNIT_CNY, aliases=("m_open",), metadata={"adjusted": False, "adjustment_status": "unverified"}),
    _f("minute_high", "StockMinuteBar", "High", unit=UNIT_CNY, aliases=("m_high",), metadata={"adjusted": False, "adjustment_status": "unverified"}),
    _f("minute_low", "StockMinuteBar", "Low", unit=UNIT_CNY, aliases=("m_low",), metadata={"adjusted": False, "adjustment_status": "unverified"}),
    _f("minute_close", "StockMinuteBar", "Close", unit=UNIT_CNY, aliases=("m_close",), metadata={"adjusted": False, "adjustment_status": "unverified"}),
    _f("minute_volume", "StockMinuteBar", "Volume", unit=UNIT_SHARE, aliases=("m_volume",)),
    _f("minute_amount", "StockMinuteBar", "Amount", unit=UNIT_CNY, aliases=("m_amount",)),
    _f("minute_vwap", "StockMinuteBar", "Vwap", unit=UNIT_CNY, aliases=("m_vwap",), metadata={"adjusted": False, "adjustment_status": "unverified"}),

    # ----------------------------------------------------------------------
    # IndexDailyBar / EtfDailyBar 字段目录（review §4.1）。IndexDailyBar 用于
    # 基准/相对指标，IndexConstituent 权重另列；名称用 index_* / etf_*。
    # ----------------------------------------------------------------------
    _f("index_open", "IndexDailyBar", "Open", unit=UNIT_CNY, aliases=("index_o",), mining_allowed=False),
    _f("index_high", "IndexDailyBar", "High", unit=UNIT_CNY, aliases=("index_h",), mining_allowed=False),
    _f("index_low", "IndexDailyBar", "Low", unit=UNIT_CNY, aliases=("index_l",), mining_allowed=False),
    _f("index_close", "IndexDailyBar", "Close", unit=UNIT_CNY, aliases=("index_c",), mining_allowed=False),
    _f("index_pre_close", "IndexDailyBar", "PreClose", unit=UNIT_CNY, mining_allowed=False),
    _f("index_volume", "IndexDailyBar", "Volume", unit=UNIT_SHARE, mining_allowed=False),
    _f("index_amount", "IndexDailyBar", "Amount", unit=UNIT_CNY, mining_allowed=False),

    _f("etf_open", "EtfDailyBar", "Open", unit=UNIT_CNY, aliases=("etf_o",)),
    _f("etf_high", "EtfDailyBar", "High", unit=UNIT_CNY, aliases=("etf_h",)),
    _f("etf_low", "EtfDailyBar", "Low", unit=UNIT_CNY, aliases=("etf_l",)),
    _f("etf_close", "EtfDailyBar", "Close", unit=UNIT_CNY, aliases=("etf_c",)),
    _f("etf_pre_close", "EtfDailyBar", "PreClose", unit=UNIT_CNY),
    _f("etf_volume", "EtfDailyBar", "Volume", unit=UNIT_SHARE),
    _f("etf_amount", "EtfDailyBar", "Amount", unit=UNIT_CNY),

    # ----------------------------------------------------------------------
    # 财务表级限定的 PubDate / ReportPeriodEndDate / UpdateTime
    # （review §4.1）：knowledge=PIT 可用日、period=报告期、ingestion=写入时间。
    # 裸名 pub_date / report_period_end_date / update_time 因多表共享会歧义，
    # 必须用 StockBalance.pub_date 等限定名解析（review §4.3）。
    # ----------------------------------------------------------------------
    _f("pub_date", "StockBalance", "PubDate", dtype="date", unit=UNIT_DATE, role="knowledge_time", mining_allowed=False),
    _f("report_period_end_date", "StockBalance", "ReportPeriodEndDate", dtype="date", unit=UNIT_DATE, role="period_id", mining_allowed=False),
    _f("update_time", "StockBalance", "UpdateTime", dtype="datetime", unit=UNIT_DATETIME, role="ingestion_time", mining_allowed=False),
    _f("pub_date", "StockIncome", "PubDate", dtype="date", unit=UNIT_DATE, role="knowledge_time", mining_allowed=False),
    _f("report_period_end_date", "StockIncome", "ReportPeriodEndDate", dtype="date", unit=UNIT_DATE, role="period_id", mining_allowed=False),
    _f("update_time", "StockIncome", "UpdateTime", dtype="datetime", unit=UNIT_DATETIME, role="ingestion_time", mining_allowed=False),
    _f("pub_date", "StockCashFlow", "PubDate", dtype="date", unit=UNIT_DATE, role="knowledge_time", mining_allowed=False),
    _f("report_period_end_date", "StockCashFlow", "ReportPeriodEndDate", dtype="date", unit=UNIT_DATE, role="period_id", mining_allowed=False),
    _f("update_time", "StockCashFlow", "UpdateTime", dtype="datetime", unit=UNIT_DATETIME, role="ingestion_time", mining_allowed=False),

    # ----------------------------------------------------------------------
    # StockTopTenFloatShareholder 独立字段（review §4.1）：物理列与
    # StockTopTenShareholder 一致，名称用 float_* 前缀避免裸名歧义。
    # ----------------------------------------------------------------------
    _f("float_share_ratio", "StockTopTenFloatShareholder", "ShareRatio", unit=UNIT_RATIO, source_unit=UNIT_PERCENT, cardinality="one_to_many", mining_allowed=False),
    _f("float_shareholder_id", "StockTopTenFloatShareholder", "ShareholderId", dtype="string", unit=UNIT_IDENTIFIER, role="identifier", cardinality="one_to_many", mining_allowed=False),
    _f("float_shareholder_rank", "StockTopTenFloatShareholder", "ShareholderRank", dtype="float64", unit=UNIT_RATIO, role="group_key", cardinality="one_to_many", mining_allowed=False),
    _f("float_shareholder_name", "StockTopTenFloatShareholder", "ShareholderName", dtype="string", unit=UNIT_TEXT, role="label", cardinality="one_to_many", mining_allowed=False),
    _f("float_shareholder_class", "StockTopTenFloatShareholder", "ShareholderClass", dtype="string", unit=UNIT_TEXT, role="label", cardinality="one_to_many", mining_allowed=False),
    _f("float_shares_nature", "StockTopTenFloatShareholder", "SharesNature", dtype="string", unit=UNIT_TEXT, role="label", cardinality="one_to_many", mining_allowed=False),
    _f("float_share_number", "StockTopTenFloatShareholder", "ShareNumber", unit=UNIT_SHARE, cardinality="one_to_many", mining_allowed=False),
    _f("float_share_pledge", "StockTopTenFloatShareholder", "SharePledge", unit=UNIT_SHARE, cardinality="one_to_many", mining_allowed=False),
    _f("float_share_freeze", "StockTopTenFloatShareholder", "ShareFreeze", unit=UNIT_SHARE, cardinality="one_to_many", mining_allowed=False),

    # ----------------------------------------------------------------------
    # COS lqtp 财务四表全量值字段（字典 §5，schema 已实测）(audit §1.8)。
    # StockIncome/StockCashFlow 为年初至报告期累计（flow_ytd），StockBalance 为
    # 时点存量（balance），StockIndicator 比率为百分数（÷100 得小数）。
    # ----------------------------------------------------------------------
    _f("interest_income", "StockIncome", "InterestIncome", unit=UNIT_CNY, grain="flow_ytd"),
    _f("premiums_earned", "StockIncome", "PremiumsEarned", unit=UNIT_CNY, grain="flow_ytd"),
    _f("commission_income", "StockIncome", "CommissionIncome", unit=UNIT_CNY, grain="flow_ytd"),
    _f("total_operating_cost", "StockIncome", "TotalOperatingCost", unit=UNIT_CNY, grain="flow_ytd"),
    _f("interest_expense", "StockIncome", "InterestExpense", unit=UNIT_CNY, grain="flow_ytd"),
    _f("commission_expense", "StockIncome", "CommissionExpense", unit=UNIT_CNY, grain="flow_ytd"),
    _f("refunded_premiums", "StockIncome", "RefundedPremiums", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_pay_insurance_claims", "StockIncome", "NetPayInsuranceClaims", unit=UNIT_CNY, grain="flow_ytd"),
    _f("withdraw_insurance_contract_reserve", "StockIncome", "WithdrawInsuranceContractReserve", unit=UNIT_CNY, grain="flow_ytd"),
    _f("policy_dividend_payout", "StockIncome", "PolicyDividendPayout", unit=UNIT_CNY, grain="flow_ytd"),
    _f("reinsurance_cost", "StockIncome", "ReinsuranceCost", unit=UNIT_CNY, grain="flow_ytd"),
    _f("operating_tax_surcharges", "StockIncome", "OperatingTaxSurcharges", unit=UNIT_CNY, grain="flow_ytd"),
    _f("asset_impairment_loss", "StockIncome", "AssetImpairmentLoss", unit=UNIT_CNY, grain="flow_ytd"),
    _f("fair_value_variable_income", "StockIncome", "FairValueVariableIncome", unit=UNIT_CNY, grain="flow_ytd"),
    _f("investment_income", "StockIncome", "InvestmentIncome", unit=UNIT_CNY, grain="flow_ytd"),
    _f("invest_income_associates", "StockIncome", "InvestIncomeAssociates", unit=UNIT_CNY, grain="flow_ytd"),
    _f("exchange_income", "StockIncome", "ExchangeIncome", unit=UNIT_CNY, grain="flow_ytd"),
    _f("non_operating_revenue", "StockIncome", "NonOperatingRevenue", unit=UNIT_CNY, grain="flow_ytd"),
    _f("non_operating_expense", "StockIncome", "NonOperatingExpense", unit=UNIT_CNY, grain="flow_ytd"),
    _f("disposal_loss_non_current_liability", "StockIncome", "DisposalLossNonCurrentLiability", unit=UNIT_CNY, grain="flow_ytd"),
    _f("minority_profit", "StockIncome", "MinorityProfit", unit=UNIT_CNY, grain="flow_ytd"),
    _f("basic_eps", "StockIncome", "BasicEps", unit=UNIT_CNY, grain="flow_ytd"),
    _f("diluted_eps", "StockIncome", "DilutedEps", unit=UNIT_CNY, grain="flow_ytd"),
    _f("other_composite_income", "StockIncome", "OtherCompositeIncome", unit=UNIT_CNY, grain="flow_ytd"),
    _f("total_composite_income", "StockIncome", "TotalCompositeIncome", unit=UNIT_CNY, grain="flow_ytd"),
    _f("ci_parent_company_owners", "StockIncome", "CiParentCompanyOwners", unit=UNIT_CNY, grain="flow_ytd"),
    _f("ci_minority_owners", "StockIncome", "CiMinorityOwners", unit=UNIT_CNY, grain="flow_ytd"),
    _f("asset_deal_income", "StockIncome", "AssetDealIncome", unit=UNIT_CNY, grain="flow_ytd"),
    _f("sust_operate_net_profit", "StockIncome", "SustOperateNetProfit", unit=UNIT_CNY, grain="flow_ytd"),
    _f("discon_operate_net_profit", "StockIncome", "DisconOperateNetProfit", unit=UNIT_CNY, grain="flow_ytd"),
    _f("credit_impairment_loss", "StockIncome", "CreditImpairmentLoss", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_open_hedge_income", "StockIncome", "NetOpenHedgeIncome", unit=UNIT_CNY, grain="flow_ytd"),
    _f("interest_cost_fin", "StockIncome", "InterestCostFin", unit=UNIT_CNY, grain="flow_ytd"),
    _f("interest_income_fin", "StockIncome", "InterestIncomeFin", unit=UNIT_CNY, grain="flow_ytd"),
    _f("other_earnings", "StockIncome", "OtherEarnings", unit=UNIT_CNY, grain="flow_ytd"),
    _f("other_composite_income_mino_at", "StockIncome", "OtherCompositeIncomeMinoAt", unit=UNIT_CNY, grain="flow_ytd"),
    _f("goods_sale_and_service_render_cash", "StockCashFlow", "GoodsSaleAndServiceRenderCash", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_deposit_increase", "StockCashFlow", "NetDepositIncrease", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_borrowing_from_central_bank", "StockCashFlow", "NetBorrowingFromCentralBank", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_borrowing_from_finance_co", "StockCashFlow", "NetBorrowingFromFinanceCo", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_original_insurance_cash", "StockCashFlow", "NetOriginalInsuranceCash", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_cash_received_from_reinsurance_business", "StockCashFlow", "NetCashReceivedFromReinsuranceBusiness", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_insurer_deposit_investment", "StockCashFlow", "NetInsurerDepositInvestment", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_deal_trading_assets", "StockCashFlow", "NetDealTradingAssets", unit=UNIT_CNY, grain="flow_ytd"),
    _f("interest_and_commission_cashin", "StockCashFlow", "InterestAndCommissionCashin", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_increase_in_placements", "StockCashFlow", "NetIncreaseInPlacements", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_buyback", "StockCashFlow", "NetBuyback", unit=UNIT_CNY, grain="flow_ytd"),
    _f("tax_levy_refund", "StockCashFlow", "TaxLevyRefund", unit=UNIT_CNY, grain="flow_ytd"),
    _f("other_cashin_related_operate", "StockCashFlow", "OtherCashinRelatedOperate", unit=UNIT_CNY, grain="flow_ytd"),
    _f("subtotal_operate_cash_inflow", "StockCashFlow", "SubtotalOperateCashInflow", unit=UNIT_CNY, grain="flow_ytd"),
    _f("goods_and_services_cash_paid", "StockCashFlow", "GoodsAndServicesCashPaid", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_loan_and_advance_increase", "StockCashFlow", "NetLoanAndAdvanceIncrease", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_deposit_in_cb_and_ib", "StockCashFlow", "NetDepositInCbAndIb", unit=UNIT_CNY, grain="flow_ytd"),
    _f("original_compensation_paid", "StockCashFlow", "OriginalCompensationPaid", unit=UNIT_CNY, grain="flow_ytd"),
    _f("handling_charges_and_commission", "StockCashFlow", "HandlingChargesAndCommission", unit=UNIT_CNY, grain="flow_ytd"),
    _f("policy_dividend_cash_paid", "StockCashFlow", "PolicyDividendCashPaid", unit=UNIT_CNY, grain="flow_ytd"),
    _f("staff_behalf_paid", "StockCashFlow", "StaffBehalfPaid", unit=UNIT_CNY, grain="flow_ytd"),
    _f("tax_payments", "StockCashFlow", "TaxPayments", unit=UNIT_CNY, grain="flow_ytd"),
    _f("other_operate_cash_paid", "StockCashFlow", "OtherOperateCashPaid", unit=UNIT_CNY, grain="flow_ytd"),
    _f("subtotal_operate_cash_outflow", "StockCashFlow", "SubtotalOperateCashOutflow", unit=UNIT_CNY, grain="flow_ytd"),
    _f("invest_withdrawal_cash", "StockCashFlow", "InvestWithdrawalCash", unit=UNIT_CNY, grain="flow_ytd"),
    _f("invest_proceeds", "StockCashFlow", "InvestProceeds", unit=UNIT_CNY, grain="flow_ytd"),
    _f("fix_intan_other_asset_dispo_cash", "StockCashFlow", "FixIntanOtherAssetDispoCash", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_cash_deal_subcompany", "StockCashFlow", "NetCashDealSubcompany", unit=UNIT_CNY, grain="flow_ytd"),
    _f("other_cash_from_invest_act", "StockCashFlow", "OtherCashFromInvestAct", unit=UNIT_CNY, grain="flow_ytd"),
    _f("subtotal_invest_cash_inflow", "StockCashFlow", "SubtotalInvestCashInflow", unit=UNIT_CNY, grain="flow_ytd"),
    _f("invest_cash_paid", "StockCashFlow", "InvestCashPaid", unit=UNIT_CNY, grain="flow_ytd"),
    _f("impawned_loan_net_increase", "StockCashFlow", "ImpawnedLoanNetIncrease", unit=UNIT_CNY, grain="flow_ytd"),
    _f("net_cash_from_sub_company", "StockCashFlow", "NetCashFromSubCompany", unit=UNIT_CNY, grain="flow_ytd"),
    _f("other_cash_to_invest_act", "StockCashFlow", "OtherCashToInvestAct", unit=UNIT_CNY, grain="flow_ytd"),
    _f("subtotal_invest_cash_outflow", "StockCashFlow", "SubtotalInvestCashOutflow", unit=UNIT_CNY, grain="flow_ytd"),
    _f("cash_from_invest", "StockCashFlow", "CashFromInvest", unit=UNIT_CNY, grain="flow_ytd"),
    _f("cash_from_mino_s_invest_sub", "StockCashFlow", "CashFromMinoSInvestSub", unit=UNIT_CNY, grain="flow_ytd"),
    _f("cash_from_borrowing", "StockCashFlow", "CashFromBorrowing", unit=UNIT_CNY, grain="flow_ytd"),
    _f("cash_from_bonds_issue", "StockCashFlow", "CashFromBondsIssue", unit=UNIT_CNY, grain="flow_ytd"),
    _f("other_finance_act_cash", "StockCashFlow", "OtherFinanceActCash", unit=UNIT_CNY, grain="flow_ytd"),
    _f("subtotal_finance_cash_inflow", "StockCashFlow", "SubtotalFinanceCashInflow", unit=UNIT_CNY, grain="flow_ytd"),
    _f("borrowing_repayment", "StockCashFlow", "BorrowingRepayment", unit=UNIT_CNY, grain="flow_ytd"),
    _f("dividend_interest_payment", "StockCashFlow", "DividendInterestPayment", unit=UNIT_CNY, grain="flow_ytd"),
    _f("proceeds_from_sub_to_mino_s", "StockCashFlow", "ProceedsFromSubToMinoS", unit=UNIT_CNY, grain="flow_ytd"),
    _f("other_finance_act_payment", "StockCashFlow", "OtherFinanceActPayment", unit=UNIT_CNY, grain="flow_ytd"),
    _f("subtotal_finance_cash_outflow", "StockCashFlow", "SubtotalFinanceCashOutflow", unit=UNIT_CNY, grain="flow_ytd"),
    _f("exchange_rate_change_effect", "StockCashFlow", "ExchangeRateChangeEffect", unit=UNIT_CNY, grain="flow_ytd"),
    _f("cash_equivalent_increase", "StockCashFlow", "CashEquivalentIncrease", unit=UNIT_CNY, grain="flow_ytd"),
    _f("cash_equivalents_at_beginning", "StockCashFlow", "CashEquivalentsAtBeginning", unit=UNIT_CNY, grain="flow_ytd"),
    _f("cash_and_equivalents_at_end", "StockCashFlow", "CashAndEquivalentsAtEnd", unit=UNIT_CNY, grain="flow_ytd"),
    _f("settlement_provi", "StockBalance", "SettlementProvi", unit=UNIT_CNY, grain="balance"),
    _f("lend_capital", "StockBalance", "LendCapital", unit=UNIT_CNY, grain="balance"),
    _f("insurance_receivables", "StockBalance", "InsuranceReceivables", unit=UNIT_CNY, grain="balance"),
    _f("reinsurance_receivables", "StockBalance", "ReinsuranceReceivables", unit=UNIT_CNY, grain="balance"),
    _f("reinsurance_contract_reserves_receivable", "StockBalance", "ReinsuranceContractReservesReceivable", unit=UNIT_CNY, grain="balance"),
    _f("interest_receivable", "StockBalance", "InterestReceivable", unit=UNIT_CNY, grain="balance"),
    _f("dividend_receivable", "StockBalance", "DividendReceivable", unit=UNIT_CNY, grain="balance"),
    _f("other_receivable", "StockBalance", "OtherReceivable", unit=UNIT_CNY, grain="balance"),
    _f("bought_sellback_assets", "StockBalance", "BoughtSellbackAssets", unit=UNIT_CNY, grain="balance"),
    _f("non_current_asset_in_one_year", "StockBalance", "NonCurrentAssetInOneYear", unit=UNIT_CNY, grain="balance"),
    _f("other_current_assets", "StockBalance", "OtherCurrentAssets", unit=UNIT_CNY, grain="balance"),
    _f("loan_and_advance", "StockBalance", "LoanAndAdvance", unit=UNIT_CNY, grain="balance"),
    _f("hold_for_sale_assets", "StockBalance", "HoldForSaleAssets", unit=UNIT_CNY, grain="balance"),
    _f("hold_to_maturity_investments", "StockBalance", "HoldToMaturityInvestments", unit=UNIT_CNY, grain="balance"),
    _f("longterm_receivable_account", "StockBalance", "LongtermReceivableAccount", unit=UNIT_CNY, grain="balance"),
    _f("investment_property", "StockBalance", "InvestmentProperty", unit=UNIT_CNY, grain="balance"),
    _f("construction_materials", "StockBalance", "ConstructionMaterials", unit=UNIT_CNY, grain="balance"),
    _f("fixed_assets_liquidation", "StockBalance", "FixedAssetsLiquidation", unit=UNIT_CNY, grain="balance"),
    _f("biological_assets", "StockBalance", "BiologicalAssets", unit=UNIT_CNY, grain="balance"),
    _f("oil_gas_assets", "StockBalance", "OilGasAssets", unit=UNIT_CNY, grain="balance"),
    _f("development_expenditure", "StockBalance", "DevelopmentExpenditure", unit=UNIT_CNY, grain="balance"),
    _f("long_deferred_expense", "StockBalance", "LongDeferredExpense", unit=UNIT_CNY, grain="balance"),
    _f("other_non_current_assets", "StockBalance", "OtherNonCurrentAssets", unit=UNIT_CNY, grain="balance"),
    _f("total_non_current_assets", "StockBalance", "TotalNonCurrentAssets", unit=UNIT_CNY, grain="balance"),
    _f("borrowing_from_centralbank", "StockBalance", "BorrowingFromCentralbank", unit=UNIT_CNY, grain="balance"),
    _f("deposit_in_interbank", "StockBalance", "DepositInInterbank", unit=UNIT_CNY, grain="balance"),
    _f("borrowing_capital", "StockBalance", "BorrowingCapital", unit=UNIT_CNY, grain="balance"),
    _f("trading_liability", "StockBalance", "TradingLiability", unit=UNIT_CNY, grain="balance"),
    _f("advance_peceipts", "StockBalance", "AdvancePeceipts", unit=UNIT_CNY, grain="balance"),
    _f("sold_buyback_secu_proceeds", "StockBalance", "SoldBuybackSecuProceeds", unit=UNIT_CNY, grain="balance"),
    _f("commission_payable", "StockBalance", "CommissionPayable", unit=UNIT_CNY, grain="balance"),
    _f("interest_payable", "StockBalance", "InterestPayable", unit=UNIT_CNY, grain="balance"),
    _f("dividend_payable", "StockBalance", "DividendPayable", unit=UNIT_CNY, grain="balance"),
    _f("other_payable", "StockBalance", "OtherPayable", unit=UNIT_CNY, grain="balance"),
    _f("reinsurance_payables", "StockBalance", "ReinsurancePayables", unit=UNIT_CNY, grain="balance"),
    _f("insurance_contract_reserves", "StockBalance", "InsuranceContractReserves", unit=UNIT_CNY, grain="balance"),
    _f("proxy_secu_proceeds", "StockBalance", "ProxySecuProceeds", unit=UNIT_CNY, grain="balance"),
    _f("receivings_from_vicariously_sold_securities", "StockBalance", "ReceivingsFromVicariouslySoldSecurities", unit=UNIT_CNY, grain="balance"),
    _f("non_current_liability_in_one_year", "StockBalance", "NonCurrentLiabilityInOneYear", unit=UNIT_CNY, grain="balance"),
    _f("other_current_liability", "StockBalance", "OtherCurrentLiability", unit=UNIT_CNY, grain="balance"),
    _f("longterm_account_payable", "StockBalance", "LongtermAccountPayable", unit=UNIT_CNY, grain="balance"),
    _f("specific_account_payable", "StockBalance", "SpecificAccountPayable", unit=UNIT_CNY, grain="balance"),
    _f("estimate_liability", "StockBalance", "EstimateLiability", unit=UNIT_CNY, grain="balance"),
    _f("other_non_current_liability", "StockBalance", "OtherNonCurrentLiability", unit=UNIT_CNY, grain="balance"),
    _f("total_non_current_liability", "StockBalance", "TotalNonCurrentLiability", unit=UNIT_CNY, grain="balance"),
    _f("paidin_capital", "StockBalance", "PaidinCapital", unit=UNIT_CNY, grain="balance"),
    _f("capital_reserve_fund", "StockBalance", "CapitalReserveFund", unit=UNIT_CNY, grain="balance"),
    _f("treasury_stock", "StockBalance", "TreasuryStock", unit=UNIT_CNY, grain="balance"),
    _f("specific_reserves", "StockBalance", "SpecificReserves", unit=UNIT_CNY, grain="balance"),
    _f("surplus_reserve_fund", "StockBalance", "SurplusReserveFund", unit=UNIT_CNY, grain="balance"),
    _f("ordinary_risk_reserve_fund", "StockBalance", "OrdinaryRiskReserveFund", unit=UNIT_CNY, grain="balance"),
    _f("retained_profit", "StockBalance", "RetainedProfit", unit=UNIT_CNY, grain="balance"),
    _f("foreign_currency_report_conv_diff", "StockBalance", "ForeignCurrencyReportConvDiff", unit=UNIT_CNY, grain="balance"),
    _f("total_sheet_owner_equities", "StockBalance", "TotalSheetOwnerEquities", unit=UNIT_CNY, grain="balance"),
    _f("other_comprehensive_income", "StockBalance", "OtherComprehensiveIncome", unit=UNIT_CNY, grain="balance"),
    _f("deferred_earning", "StockBalance", "DeferredEarning", unit=UNIT_CNY, grain="balance"),
    _f("loan_and_advance_current_assets", "StockBalance", "LoanAndAdvanceCurrentAssets", unit=UNIT_CNY, grain="balance"),
    _f("derivative_financial_asset", "StockBalance", "DerivativeFinancialAsset", unit=UNIT_CNY, grain="balance"),
    _f("hold_sale_asset", "StockBalance", "HoldSaleAsset", unit=UNIT_CNY, grain="balance"),
    _f("loan_and_advance_noncurrent_assets", "StockBalance", "LoanAndAdvanceNoncurrentAssets", unit=UNIT_CNY, grain="balance"),
    _f("derivative_financial_liability", "StockBalance", "DerivativeFinancialLiability", unit=UNIT_CNY, grain="balance"),
    _f("hold_sale_liability", "StockBalance", "HoldSaleLiability", unit=UNIT_CNY, grain="balance"),
    _f("estimate_liability_current", "StockBalance", "EstimateLiabilityCurrent", unit=UNIT_CNY, grain="balance"),
    _f("deferred_earning_current", "StockBalance", "DeferredEarningCurrent", unit=UNIT_CNY, grain="balance"),
    _f("preferred_shares_noncurrent", "StockBalance", "PreferredSharesNoncurrent", unit=UNIT_CNY, grain="balance"),
    _f("pepertual_liability_noncurrent", "StockBalance", "PepertualLiabilityNoncurrent", unit=UNIT_CNY, grain="balance"),
    _f("longterm_salaries_payable", "StockBalance", "LongtermSalariesPayable", unit=UNIT_CNY, grain="balance"),
    _f("other_equity_tools", "StockBalance", "OtherEquityTools", unit=UNIT_CNY, grain="balance"),
    _f("preferred_shares_equity", "StockBalance", "PreferredSharesEquity", unit=UNIT_CNY, grain="balance"),
    _f("pepertual_liability_equity", "StockBalance", "PepertualLiabilityEquity", unit=UNIT_CNY, grain="balance"),
    _f("bond_invest", "StockBalance", "BondInvest", unit=UNIT_CNY, grain="balance"),
    _f("other_bond_invest", "StockBalance", "OtherBondInvest", unit=UNIT_CNY, grain="balance"),
    _f("other_equity_tools_invest", "StockBalance", "OtherEquityToolsInvest", unit=UNIT_CNY, grain="balance"),
    _f("other_non_current_financial_assets", "StockBalance", "OtherNonCurrentFinancialAssets", unit=UNIT_CNY, grain="balance"),
    # StockIndicator OperatingProfit is the same canonical name as the StockIncome
    # flow; the indicator-table ratio variant is dropped to keep the bare name
    # unambiguous (resolve via StockIndicator.operating_profit if needed).
    _f("value_change_profit", "StockIndicator", "ValueChangeProfit", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_return", "StockIndicator", "IncReturn", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("roa", "StockIndicator", "Roa", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("net_profit_margin", "StockIndicator", "NetProfitMargin", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("gross_profit_margin", "StockIndicator", "GrossProfitMargin", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("expense_to_total_revenue", "StockIndicator", "ExpenseToTotalRevenue", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("operation_profit_to_total_revenue", "StockIndicator", "OperationProfitToTotalRevenue", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("net_profit_to_total_revenue", "StockIndicator", "NetProfitToTotalRevenue", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("operating_expense_to_total_revenue", "StockIndicator", "OperatingExpenseToTotalRevenue", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("ga_expense_to_total_revenue", "StockIndicator", "GaExpenseToTotalRevenue", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("financing_expense_to_total_revenue", "StockIndicator", "FinancingExpenseToTotalRevenue", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("operating_profit_to_profit", "StockIndicator", "OperatingProfitToProfit", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("invesment_profit_to_profit", "StockIndicator", "InvesmentProfitToProfit", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("adjusted_profit_to_profit", "StockIndicator", "AdjustedProfitToProfit", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("goods_sale_and_service_to_revenue", "StockIndicator", "GoodsSaleAndServiceToRevenue", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("ocf_to_revenue", "StockIndicator", "OcfToRevenue", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("ocf_to_operating_profit", "StockIndicator", "OcfToOperatingProfit", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_total_revenue_year_on_year", "StockIndicator", "IncTotalRevenueYearOnYear", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_total_revenue_annual", "StockIndicator", "IncTotalRevenueAnnual", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_revenue_year_on_year", "StockIndicator", "IncRevenueYearOnYear", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_revenue_annual", "StockIndicator", "IncRevenueAnnual", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_operation_profit_year_on_year", "StockIndicator", "IncOperationProfitYearOnYear", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_operation_profit_annual", "StockIndicator", "IncOperationProfitAnnual", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_net_profit_year_on_year", "StockIndicator", "IncNetProfitYearOnYear", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_net_profit_annual", "StockIndicator", "IncNetProfitAnnual", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_net_profit_to_shareholders_year_on_year", "StockIndicator", "IncNetProfitToShareholdersYearOnYear", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
    _f("inc_net_profit_to_shareholders_annual", "StockIndicator", "IncNetProfitToShareholdersAnnual", unit=UNIT_RATIO, source_unit=UNIT_PERCENT),
)


__all__ = [
    "ASHARE_FIELD_SPECS",
    "ASHARE_TABLE_SPECS",
    "FieldRole",
    "validate_field_role",
]
