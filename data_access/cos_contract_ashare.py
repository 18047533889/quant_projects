# -*- coding: utf-8 -*-
"""A-share COS contracts."""
import dataclasses
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
    # R45 P0-ASHARE：StockList/Status/Industry 覆盖起始、自然日 cadence（字典 §2 核实 2016-01-01 起）。
    "ashare_stock_list": _c("ashare_stock_list", "ashare", "D1", "equi", "Symbol", calendar_domain="calendar_day", storage_layout=D, coverage_start="2016-01-01", expected_cadence="daily"),
    "ashare_stock_status": _c("ashare_stock_status", "ashare", "S1", "equi", "Symbol", calendar_domain="calendar_day", storage_layout=D, coverage_start="2016-01-01", expected_cadence="daily"),
    "ashare_stock_industry": _c("ashare_stock_industry", "ashare", "D1", "equi", "Symbol", required_panel_filters=("IndustrySource",), required_dimension_filters=("IndustrySource",), allowed_filter_values=(("IndustrySource", _INDUSTRY),), cardinality="one_to_many", unique_key=("TradeDate", "Symbol", "IndustrySource"), calendar_domain="calendar_day", storage_layout=D, coverage_start="2016-01-01", expected_cadence="daily"),
    # R45 P0-ASHARE：StockValuationDaily 按交易日（字典核实 2016-01-04 起），全历史 D1。
    "ashare_stock_valuation_daily": _c("ashare_stock_valuation_daily", "ashare", "D1", "equi", "Symbol", storage_layout=D, coverage_start="2016-01-04", expected_cadence="daily"),
    # R45 P0-ASHARE：StockCapitalDaily 为 S1 日终股本快照，**非**拆股事件表（区别于美股）。
    # 字典实测最末日约 2026-06-14，明显滞后于行情（2026-07-31）——必须携带 freshness/
    # staleness 门槛，禁止把当日股本当 S1-ready 同日使用（ChangeDate 相对 TradeDate 可滞后
    # 中位 ~155 天）。max_staleness=45d 提示查询前校验滞后。
    "ashare_stock_capital_daily": _c("ashare_stock_capital_daily", "ashare", "S1", "equi", "Symbol", storage_layout=D, coverage_start="2016-01-01", expected_cadence="daily", max_staleness="45d"),
    "ashare_etf_daily": _c("ashare_etf_daily", "ashare", "D1", "equi", "Symbol", storage_layout=D),
    "ashare_etf_list": _c("ashare_etf_list", "ashare", "D1", "equi", "Symbol", calendar_domain="calendar_day", storage_layout=D),
    "ashare_index_daily": _c("ashare_index_daily", "ashare", "D1", "equi", "Symbol", return_column="Return", return_scale=1 / 10000, storage_layout=D),
    "ashare_index_list": _c("ashare_index_list", "ashare", "D1", "equi", "Symbol", calendar_domain="calendar_day", storage_layout=D),
    "ashare_index_constituent": _c("ashare_index_constituent", "ashare", "D1", "equi", "Symbol", required_panel_filters=("IndexSymbol",), required_dimension_filters=("IndexSymbol",), cardinality="one_to_many", unique_key=("IndexSymbol", "TradeDate", "Symbol"), calendar_domain="calendar_day", storage_layout=D),
    "ashare_universe_daily": _c("ashare_universe_daily", "ashare", "D1", "equi", "Symbol", source_class="derived_dataset", calendar_domain="calendar_day", expected_cadence="daily"),
    # R45 P0-ASHARE：TopTen 是 S1 自然日快照，unique_key 用 ShareholderRank（字典 §4.15/§5
    # `StockTopTen*` 粒度 = 一股票×一快照日×一名股东，1–10 名次），**不是** Rank。
    # PIT vintage 注意：TopTen 快照含 ReportPeriodEndDate + PubDate，但 COS 无历史 revision
    # vintage；仅当能以 (PubDate, ReportPeriodEndDate) 证明某 TradeDate 快照「当日真可知」时
    # 才能生产采用（shareholder alpha）；否则保持 DATA_GATED（见 R45 注释块）。
    "ashare_stock_topten_shareholder": _c("ashare_stock_topten_shareholder", "ashare", "S1", "equi", "Symbol", calendar_domain="calendar_day", cardinality="one_to_many", unique_key=("TradeDate", "Symbol", "ShareholderRank"), storage_layout=D, coverage_start="2016-01-01", expected_cadence="daily"),
    "ashare_stock_topten_float_shareholder": _c("ashare_stock_topten_float_shareholder", "ashare", "S1", "equi", "Symbol", calendar_domain="calendar_day", cardinality="one_to_many", unique_key=("TradeDate", "Symbol", "ShareholderRank"), storage_layout=D, coverage_start="2016-01-01", expected_cadence="daily"),
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
        # R45 P0-ASHARE：财报事件表按公告日（PubDate）落文件，2003-04-22 起覆盖；
        # 披露节奏=event_driven（常规年报/中报/季报，偶发修订公告）。
        coverage_start="2003-04-22",
        expected_cadence="event_driven",
        max_staleness="1d",  # 文件日期 = PubDate；同一天公告即当日可用，无滞后
        missing_partition_semantics="warn",
    )

ASHARE_COS_CONTRACTS["ashare_stock_dividend"] = _c(
    "ashare_stock_dividend", "ashare", "E1", "event", "Symbol",
    pit="effective_time_only", event_column="ExDividendDate", period_column="ExDividendDate",
    revision_columns=("UpdateTime",), storage_layout="event_files",
    note="No reliable announcement timestamp; explicit effective-date use only.",
)

# ---- R45 P0-ASHARE：TopTen S1 历史 PIT vintage 生产门槛 ----
# StockTopTenShareholder/TopTenFloatShareholder 是 S1 自然日快照（unique_key 用
# ShareholderRank）。快照行含 ReportPeriodEndDate + PubDate，但 COS **无**历史 revision
# vintage：字典明示 UpdateTime 只是供应商 freshness，不能当作市场可知修订时点。
#
# 因此，S1 TopTen 快照若要作为 shareholder-alpha 信号**生产采用**，必须能证明其「真 PIT
# vintage」——即 TradeDate 当日快照内容确实就是当时可知的（由 PubDate ≤ TradeDate 且该
# 报告期公告已披露证明）。在 PIT vintage 未证明前，该 S1 只能保守为 DATA_GATED 使用
# （DATA_GATED = 数据已落盘/可读，但语义上未获得生产采用资格），禁止直接进入生产
# alpha 流水线。此守卫是保守声明，不改变 S1 分类本身。
_TOP_TEN_PIT_VINTAGE_REQUIREMENT = (
    "TopTen S1 snapshot may only be production-admitted for shareholder alpha once true "
    "PIT vintage is proven: what was knowable at TradeDate given (PubDate, ReportPeriodEndDate). "
    "Until proven, keep DATA_GATED."
)
for _tt in ("ashare_stock_topten_shareholder", "ashare_stock_topten_float_shareholder"):
    _c0 = ASHARE_COS_CONTRACTS[_tt]
    ASHARE_COS_CONTRACTS[_tt] = dataclasses.replace(
        _c0,
        note=(_c0.note + " " + _TOP_TEN_PIT_VINTAGE_REQUIREMENT).strip(),
    )
del _tt, _c0, _TOP_TEN_PIT_VINTAGE_REQUIREMENT
