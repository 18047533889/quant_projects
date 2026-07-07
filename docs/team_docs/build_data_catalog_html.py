#!/usr/bin/env python3
"""Build full 26-source data catalog HTML (tables only, no JSON blocks)."""
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
ROOT = "/home/yluel/share/projects/massive_parquet"

OVERVIEW_ROWS = [
    ("1", "行情", "aggregate_bars/daily_market_summary", "23", "~4754万", "2004–2026", "按年 1 文件/年", "日×ticker", "10", "拆股复权", "✅", "✅", "—"),
    ("2", "行情", "us_stocks_sip/day_aggs_v1", "5657", "~6400万", "2003–2026", "按日 1 文件", "日×ticker", "8", "不复权", "✅", "✅", "生产锚点推荐"),
    ("3", "行情", "us_stocks_sip/minute_aggs_v1", "5657", "~95亿", "2003–2026", "按日 1 文件", "1min×ticker", "8", "L1不复权/L2须复权", "✅", "✅", "算子见§3.2"),
    ("4", "行情", "us_stocks_sip/quotes_v1", "5657", "万亿级", "2003–2026", "按日", "逐笔报价", "14", "不复权", "❌", "❌", "~7.1TB"),
    ("5", "行情", "us_stocks_sip/trades_v1", "5657", "万亿级", "2003–2026", "按日", "逐笔成交", "13", "不复权", "❌", "❌", "~2.9TB"),
    ("6", "基本面", "fundamentals/balance_sheet", "18", "~26.8万", "2010–2031", "按申报年", "报告期×CIK", "38", "—", "✅", "✅", "⚠️2024仅1万行"),
    ("7", "基本面", "fundamentals/income_statement", "17", "~46.9万", "2010+", "按申报年", "报告期×CIK", "34", "—", "✅", "✅", "⚠️2024仅1万行"),
    ("8", "基本面", "fundamentals/cash_flow_statement", "17", "~46.8万", "2010+", "按申报年", "报告期×CIK", "32", "—", "✅", "✅", "⚠️2024仅1万行"),
    ("9", "基本面", "fundamentals/financials_ratios", "1", "5199", "快照", "单文件", "日×ticker", "23", "—", "✅", "✅", "非全历史panel"),
    ("10", "基本面", "fundamentals/short_interest", "10", "~312万", "2017–2026", "按年", "半月×ticker", "5", "—", "✅", "✅", "⚠️2024仅1万行"),
    ("11", "基本面", "fundamentals/short_volume", "3", "~413万", "2024+", "按年", "日×ticker", "15", "—", "✅", "✅", "⚠️2024仅1万行"),
    ("12", "基本面", "fundamentals/stocks_floats", "1", "6370", "快照", "单文件", "事件×ticker", "4", "—", "✅", "✅", "—"),
    ("13", "公司行动", "corporate_actions/dividends", "28", "~4.8万", "2000–2027", "按年", "分红事件", "12", "含调整因子", "✅", "✅", "⚠️2003+每年2k"),
    ("14", "公司行动", "corporate_actions/splits", "39", "~2.6万", "1978–2026", "按年", "拆股事件", "7", "含调整因子", "✅", "✅", "—"),
    ("15", "公司行动", "corporate_actions/ipos", "1", "6185", "全量", "单文件", "IPO事件", "20", "—", "✅", "✅", "—"),
    ("16", "申报", "filing/risk_factors", "12", "~2.4万", "2015–2026", "按年", "风险条目", "7", "—", "✅", "✅", "⚠️每年2k"),
    ("17", "申报", "filing/sec_edgar_index", "1", "10000", "快照", "单文件", "申报索引", "7", "—", "✅", "✅", "分页快照"),
    ("18", "申报", "filing/risk_categories", "1", "140", "字典", "单文件", "分类", "6", "—", "✅", "✅", "无ticker"),
    ("19", "申报", "filing/10k_sections", "0", "0", "—", "—", "—", "—", "—", "—", "—", "未下载"),
    ("20", "申报", "filing/8k_text", "0", "0", "—", "—", "—", "—", "—", "—", "—", "未下载"),
    ("21", "新闻", "news/news", "1", "2000", "快照?", "单文件", "新闻×ticker", "12+", "—", "✅", "✅", "非全量"),
    ("22", "元数据", "tickers/all_tickers", "1", "2000", "快照?", "单文件", "证券", "12", "—", "✅", "✅", "非全宇宙"),
    ("23", "元数据", "tickers/ticker_types", "1", "25", "字典", "单文件", "类型码", "4", "—", "✅", "✅", "—"),
    ("24", "参考", "market_operations/exchanges", "1", "52", "字典", "单文件", "交易所", "7", "—", "✅", "✅", "—"),
    ("25", "参考", "market_operations/market_holidays", "1", "24", "2026–27", "单文件", "休市日", "4", "—", "✅", "✅", "历史不足"),
    ("26", "参考", "market_operations/condition_codes", "1", "130", "字典", "单文件", "条件码", "6", "—", "✅", "✅", "tick过滤用"),
]


def tbl(headers, rows):
    h = "<table class=\"table-schema\"><thead><tr>" + "".join(f"<th>{x}</th>" for x in headers) + "</tr></thead><tbody>"
    for row in rows:
        h += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return h + "</tbody></table>"


def meta_table(m):
    return tbl(
        ["项", "内容"],
        [
            ("相对路径（raw）", f"<code>raw_massive_data/{m['rel']}</code>"),
            ("绝对路径", f"<code>{ROOT}/raw_massive_data/{m['rel']}</code>"),
            ("cleaned 镜像", f"<code>{ROOT}/cleaned_massive_data/{m['rel_clean']}</code>"),
            ("文件数 / 命名", m["files"]),
            ("频率 / 一行代表", m["grain"]),
            ("列数（raw）", f"<strong>{m['ncols']}</strong>"),
            ("复权 / 时间轴", m["adj_time"]),
            ("已清洗", m["cleaned"]),
            ("数据红线", m.get("warn", "—")),
        ],
    )


def sample_table(cols, rows):
    return tbl(cols, rows)


def dataset_block(m):
    schema = m.get("schema") or [
        (f"<code>{c}</code>", "—", "见样例列") for c in m.get("sample_cols", [])[:8]
    ]
    cols_schema = tbl(["字段", "类型", "含义"], schema)
    samp = sample_table(m["sample_cols"], m["sample_rows"])
    extra = ""
    if m.get("all_cols"):
        extra = f"<p><strong>全部列名（{len(m['all_cols'])} 列）：</strong></p>" + tbl(
            ["#", "列名"], [[str(i + 1), f"<code>{c}</code>"] for i, c in enumerate(m["all_cols"])]
        )
    cleaned_note = ""
    if m.get("cleaned_cols"):
        cleaned_note = "<p><strong>cleaned 追加标准列（前 3 列示意）：</strong></p>" + sample_table(
            m["cleaned_cols"], m["cleaned_rows"]
        )
    return f"""
          <details class="sample-fold">
            <summary>{m['title']}</summary>
            <div class="fold-body">
              {meta_table(m)}
              <h4>字段字典</h4>
              {cols_schema}
              {extra}
              <h4>样例数据（前 3 行，表格展示）</h4>
              <p class="section-desc">样例文件：<code>{m.get('sample_file','见上表命名')}</code></p>
              {samp}
              {cleaned_note}
            </div>
          </details>"""


# Dataset definitions (abbreviated schema where huge)
DATASETS = {
    "4.1 行情": [
        {
            "title": "4.1.1 REST 日 K · daily_market_summary",
            "rel": "aggregate_bars/daily_market_summary/",
            "rel_clean": "aggregate_bars/daily_market_summary/",
            "files": "<code>daily_market_summary_{YYYY}.parquet</code>（23 个，2004–2026）",
            "grain": "每个交易日 × 每个 ticker 一行",
            "ncols": "10",
            "adj_time": "拆股复权（API adjusted=true）；align_time←trade_date",
            "cleaned": "✅",
            "warn": "与 SIP 日 K 不可混用",
            "sample_file": "daily_market_summary_2004.parquet",
            "schema": [
                ("<code>T</code>", "string", "ticker"),
                ("<code>o/c/h/l</code>", "float", "开/收/高/低"),
                ("<code>v</code>", "float", "成交量"),
                ("<code>vw</code>", "float", "VWAP"),
                ("<code>t</code>", "int64", "窗口起始 Unix 毫秒"),
                ("<code>n</code>", "float", "成交笔数"),
                ("<code>trade_date</code>", "string", "交易日 YYYY-MM-DD"),
            ],
            "sample_cols": ["T", "trade_date", "o", "h", "l", "c", "v", "n"],
            "sample_rows": [
                ["HRVE", "2004-01-02", "3.56", "3.64", "3.56", "3.64", "175", "4"],
                ["PAS", "2004-01-02", "17.1", "17.2", "17.01", "17.12", "166400", "442"],
                ["SNBC", "2004-01-02", "131.75", "136.8", "130", "130.75", "3734.2", "114"],
            ],
        },
        {
            "title": "4.1.2 SIP 日 K · day_aggs_v1（生产锚点）",
            "rel": "us_stocks_sip/day_aggs_v1/{YYYY}/{MM}/",
            "rel_clean": "us_stocks_sip/day_aggs_v1/{YYYY}/{MM}/",
            "files": "<code>{YYYY-MM-DD}.parquet</code>（5657 日文件）",
            "grain": "每个交易日 × 每个 ticker；window_start=当日 04:00 UTC（纳秒）",
            "ncols": "8",
            "adj_time": "不复权；cleaned align_time←window_start",
            "cleaned": "✅",
            "sample_file": "2003/09/2003-09-10.parquet",
            "schema": [
                ("<code>ticker</code>", "float/str", "证券代码"),
                ("<code>open/high/low/close</code>", "float", "OHLC"),
                ("<code>volume</code>", "float", "成交量"),
                ("<code>window_start</code>", "float", "纳秒时间戳"),
                ("<code>transactions</code>", "float", "成交笔数"),
            ],
            "sample_cols": ["ticker", "open", "high", "low", "close", "volume", "transactions"],
            "sample_rows": [
                ["A", "25.4", "25.58", "24.41", "24.49", "2869700", "2301"],
                ["AA", "28.2", "28.7", "27.85", "27.92", "3543400", "3011"],
                ["AAp", "75.0", "75.5", "72.65", "73.44", "550", "6"],
            ],
            "cleaned_cols": ["source", "frequency", "ticker", "align_time"],
            "cleaned_rows": [
                ["us_stocks_sip/day_aggs_v1", "daily", "A", "2024-06-03T04:00:00Z"],
                ["us_stocks_sip/day_aggs_v1", "daily", "AA", "2024-06-03T04:00:00Z"],
                ["us_stocks_sip/day_aggs_v1", "daily", "AAA", "2024-06-03T04:00:00Z"],
            ],
        },
        {
            "title": "4.1.3 SIP 1 分钟 K · minute_aggs_v1",
            "rel": "us_stocks_sip/minute_aggs_v1/{YYYY}/{MM}/",
            "rel_clean": "us_stocks_sip/minute_aggs_v1/{YYYY}/{MM}/",
            "files": "与日 K 相同按日分区；2024-06-03 约 152 万行/日",
            "grain": "每 1 分钟 × ticker；含盘前盘后",
            "ncols": "8（同 day）",
            "adj_time": "不复权",
            "cleaned": "✅",
            "warn": "sum(minute.vol)≠day.vol",
            "sample_file": "2003/09/2003-09-10.parquet",
            "schema": [
                ("<code>ticker</code>", "string", "代码"),
                ("<code>open/close/high/low</code>", "float", "该分钟 OHLC"),
                ("<code>volume</code>", "float", "该分钟成交量"),
                ("<code>window_start</code>", "float", "该分钟起始纳秒"),
                ("<code>transactions</code>", "float", "笔数"),
            ],
            "sample_cols": ["ticker", "open", "close", "volume", "window_start(ns示意)"],
            "sample_rows": [
                ["A", "25.4", "25.4", "47000", "1063200600…"],
                ["A", "25.39", "25.39", "12000", "1063200660…"],
                ["A", "25.38", "25.38", "8000", "1063200720…"],
            ],
        },
        {
            "title": "4.1.4 SIP 逐笔报价 · quotes_v1",
            "rel": "us_stocks_sip/quotes_v1/{YYYY}/{MM}/",
            "rel_clean": "（默认未清洗，仅测试）",
            "files": "按日；体量约 7.1 TB",
            "grain": "每条 NBBO 报价更新",
            "ncols": "14",
            "adj_time": "不复权；多时间戳列（sip/participant/trf）",
            "cleaned": "❌",
            "sample_file": "2003/09/2003-09-10.parquet",
            "schema": [
                ("<code>ticker</code>", "string", "标的"),
                ("<code>bid/ask_price</code>", "float", "买卖价（0=无报价）"),
                ("<code>bid/ask_size</code>", "float", "买卖量"),
                ("<code>conditions</code>", "varies", "条件码，常 null"),
                ("<code>sip_timestamp</code>", "int", "纳秒"),
            ],
            "all_cols": [
                "ticker", "bid_exchange", "bid_price", "bid_size", "ask_exchange", "ask_price", "ask_size",
                "conditions", "indicators", "participant_timestamp", "sequence_number", "sip_timestamp", "tape", "trf_timestamp",
            ],
            "sample_cols": ["ticker", "bid_price", "ask_price", "bid_size", "ask_size", "conditions"],
            "sample_rows": [
                ["A", "0", "0", "0", "0", "null"],
                ["A", "0", "0", "0", "0", "null"],
                ["A", "0", "0", "0", "0", "null"],
            ],
        },
        {
            "title": "4.1.5 SIP 逐笔成交 · trades_v1",
            "rel": "us_stocks_sip/trades_v1/{YYYY}/{MM}/",
            "rel_clean": "（默认未清洗）",
            "files": "按日；~2.9 TB；2024-06-03 约 8064 万行",
            "grain": "每笔成交一行",
            "ncols": "13",
            "adj_time": "不复权；conditions 约 24% null",
            "cleaned": "❌",
            "schema": [
                ("<code>ticker</code>", "string", "标的"),
                ("<code>price</code>", "float", "成交价"),
                ("<code>size</code>", "float", "成交量"),
                ("<code>exchange</code>", "float", "交易所 ID"),
                ("<code>conditions</code>", "varies", "成交条件"),
            ],
            "all_cols": [
                "ticker", "conditions", "correction", "exchange", "id", "participant_timestamp",
                "price", "sequence_number", "sip_timestamp", "size", "tape", "trf_id", "trf_timestamp",
            ],
            "sample_cols": ["ticker", "price", "size", "exchange", "conditions"],
            "sample_rows": [
                ["A", "25.4", "100", "10", "null"],
                ["A", "25.4", "100", "10", "null"],
                ["A", "25.39", "200", "10", "null"],
            ],
        },
    ],
}


def build_overview():
    rows = []
    for r in OVERVIEW_ROWS:
        rows.append([
            r[0], r[1], f"<code>{r[2]}</code>",
            r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12],
        ])
    return tbl(
        ["#", "大类", "相对路径", "文件数", "总行(约)", "时间范围", "分区", "粒度", "列", "复权", "cleaned", "启用", "备注"],
        rows,
    )


def add_fundamentals(d):
    BS_COLS = "accounts_payable, accrued_and_other_current_liabilities, accumulated_other_comprehensive_income, additional_paid_in_capital, cash_and_equivalents, cik, commitments_and_contingencies, common_stock, debt_current, deferred_revenue_current, filing_date, fiscal_quarter, fiscal_year, goodwill, intangible_assets_net, inventories, long_term_debt_and_capital_lease_obligations, noncontrolling_interest, other_assets, other_current_assets, other_equity, other_noncurrent_liabilities, period_end, preferred_stock, property_plant_equipment_net, receivables, retained_earnings_deficit, short_term_investments, tickers, timeframe, total_assets, total_current_assets, total_current_liabilities, total_equity, total_equity_attributable_to_parent, total_liabilities, total_liabilities_and_equity, treasury_stock".split(", ")
    d["4.2 基本面"] = [
        {
            "title": "4.2.1 资产负债表 · balance_sheet",
            "rel": "fundamentals/balance_sheet/",
            "rel_clean": "fundamentals/balance_sheet/",
            "files": "<code>balance_sheet_{YYYY}.parquet</code>（18 文件）",
            "grain": "1 份报表 / 报告期；tickers[]→cleaned explode",
            "ncols": "38",
            "adj_time": "PiT：align_time=filing_date（非 period_end）",
            "cleaned": "✅",
            "warn": "2024 仅 10000 行",
            "sample_file": "balance_sheet_2010.parquet",
            "schema": [
                ("<code>cik</code>", "string", "SEC 公司 ID"),
                ("<code>tickers</code>", "string[]", "关联代码（raw）"),
                ("<code>filing_date</code>", "date", "申报日（PiT 轴）"),
                ("<code>period_end</code>", "date", "会计期末"),
                ("<code>timeframe</code>", "string", "quarterly/annual"),
                ("<code>total_assets</code>", "float", "总资产"),
                ("<code>total_equity</code>", "float", "总权益"),
            ],
            "all_cols": BS_COLS,
            "sample_cols": ["ticker", "filing_date", "period_end", "timeframe", "total_assets", "total_equity", "cash_and_equivalents"],
            "sample_rows": [
                ["WMT", "2010-06-04", "2009-04-30", "quarterly", "162090000000", "63914000000", "6578000000"],
                ["TGT", "2010-05-28", "2009-05-02", "quarterly", "44212000000", "14119000000", "1371000000"],
                ["GPS", "2010-06-08", "2009-05-02", "quarterly", "7221000000", "4540000000", "1708000000"],
            ],
        },
        {
            "title": "4.2.2 利润表 · income_statement",
            "rel": "fundamentals/income_statement/",
            "rel_clean": "fundamentals/income_statement/",
            "files": "<code>income_statement_{YYYY}.parquet</code>（17 文件）",
            "grain": "1 份利润表 / 报告期",
            "ncols": "34",
            "adj_time": "align_time=filing_date",
            "cleaned": "✅",
            "warn": "2024 仅 10000 行",
            "schema": [
                ("<code>revenue</code>", "float", "营收"),
                ("<code>operating_income</code>", "float", "营业利润"),
                ("<code>basic_earnings_per_share</code>", "float", "EPS"),
                ("<code>ebitda</code>", "float", "EBITDA"),
            ],
            "all_cols": "basic_earnings_per_share, basic_shares_outstanding, cik, consolidated_net_income_loss, cost_of_revenue, depreciation_depletion_amortization, diluted_earnings_per_share, diluted_shares_outstanding, discontinued_operations, ebitda, equity_in_affiliates, extraordinary_items, filing_date, fiscal_quarter, fiscal_year, gross_profit, income_before_income_taxes, income_taxes, interest_expense, interest_income, net_income_loss_attributable_common_shareholders, noncontrolling_interest, operating_income, other_income_expense, other_operating_expenses, period_end, preferred_stock_dividends_declared, research_development, revenue, selling_general_administrative, tickers, timeframe, total_operating_expenses, total_other_income_expense".split(", "),
            "sample_cols": ["ticker", "filing_date", "period_end", "revenue", "operating_income", "basic_eps"],
            "sample_rows": [
                ["JNJ", "2010-05-10", "2009-03-29", "15026000000", "4649000000", "1.27"],
                ["NVDA", "2010-05-21", "2009-04-26", "664231000", "-230965000", "-0.01"],
                ["WMT", "2010-06-04", "2009-04-30", "94242000000", "5217000000", "0.26"],
            ],
        },
        {
            "title": "4.2.3 现金流量表 · cash_flow_statement",
            "rel": "fundamentals/cash_flow_statement/",
            "rel_clean": "fundamentals/cash_flow_statement/",
            "files": "17 个按年文件",
            "grain": "1 份现金流 / 报告期",
            "ncols": "32",
            "adj_time": "align_time=filing_date",
            "cleaned": "✅",
            "schema": [
                ("<code>net_cash_from_operating_activities</code>", "float", "经营现金流"),
                ("<code>net_cash_from_investing_activities</code>", "float", "投资现金流"),
                ("<code>net_cash_from_financing_activities</code>", "float", "筹资现金流"),
                ("<code>dividends</code>", "float", "分红支出"),
            ],
            "all_cols": "cash_from_operating_activities_continuing_operations, change_in_cash_and_equivalents, change_in_other_operating_assets_and_liabilities_net, cik, depreciation_depletion_and_amortization, dividends, effect_of_currency_exchange_rate, filing_date, fiscal_quarter, fiscal_year, income_loss_from_discontinued_operations, long_term_debt_issuances_repayments, net_cash_from_financing_activities, net_cash_from_financing_activities_continuing_operations, net_cash_from_financing_activities_discontinued_operations, net_cash_from_investing_activities, net_cash_from_investing_activities_continuing_operations, net_cash_from_investing_activities_discontinued_operations, net_cash_from_operating_activities, net_cash_from_operating_activities_discontinued_operations, net_income, noncontrolling_interests, other_cash_adjustments, other_financing_activities, other_investing_activities, other_operating_activities, period_end, purchase_of_property_plant_and_equipment, sale_of_property_plant_and_equipment, short_term_debt_issuances_repayments, tickers, timeframe".split(", "),
            "sample_cols": ["ticker", "filing_date", "CFO", "CFI", "CFF", "dividends"],
            "sample_rows": [
                ["JNJ", "2010-05-10", "2827000000", "-1076000000", "132000000", "-1273000000"],
                ["NVDA", "2010-05-21", "142128000", "-8905000", "-38637000", "—"],
                ["WMT", "2010-06-04", "3571000000", "-2683000000", "-1503000000", "-1067000000"],
            ],
        },
        {
            "title": "4.2.4 财务比率 · financials_ratios",
            "rel": "fundamentals/financials_ratios/",
            "rel_clean": "fundamentals/financials_ratios/",
            "files": "<code>financials_ratios_all.parquet</code>（单文件 5199 行）",
            "grain": "每个 ticker 每个自然日一行（供应商截面，非全历史）",
            "ncols": "23",
            "adj_time": "align_time=date",
            "cleaned": "✅",
            "schema": [
                ("<code>price_to_earnings</code>", "float", "PE（亏损常 null）"),
                ("<code>debt_to_equity</code>", "float", "杠杆"),
                ("<code>market_cap</code>", "float", "市值"),
            ],
            "all_cols": "average_volume, cash, cik, current, date, debt_to_equity, dividend_yield, earnings_per_share, enterprise_value, ev_to_ebitda, ev_to_sales, free_cash_flow, market_cap, price, price_to_book, price_to_cash_flow, price_to_earnings, price_to_free_cash_flow, price_to_sales, quick, return_on_assets, return_on_equity, ticker".split(", "),
            "sample_cols": ["ticker", "date", "price", "market_cap", "PE", "PB", "ROE"],
            "sample_rows": [
                ["A", "2026-03-09", "116.64", "32962734255", "25.55", "4.77", "0.1867"],
                ["AA", "2026-03-09", "61.16", "16136438621", "13.95", "2.64", "0.1891"],
                ["AAGR", "2026-03-09", "0.01", "578668", "null", "-0.02", "2.2506"],
            ],
        },
        {
            "title": "4.2.5 做空余额 · short_interest",
            "rel": "fundamentals/short_interest/",
            "rel_clean": "fundamentals/short_interest/",
            "files": "按年 2017–2026",
            "grain": "半月 settlement × ticker",
            "ncols": "5",
            "adj_time": "align_time=settlement_date",
            "cleaned": "✅",
            "warn": "2024 仅 10000 行",
            "schema": [
                ("<code>short_interest</code>", "int/float", "做空股数"),
                ("<code>days_to_cover</code>", "float", "回补天数"),
            ],
            "sample_cols": ["ticker", "settlement_date", "short_interest", "avg_daily_volume", "days_to_cover"],
            "sample_rows": [
                ["A", "2017-12-29", "4197300", "1234014", "3.4"],
                ["AA", "2017-12-29", "12689077", "4267200", "2.97"],
                ["AAALF", "2017-12-29", "13823", "0", "999.99"],
            ],
        },
        {
            "title": "4.2.6 每日做空量 · short_volume",
            "rel": "fundamentals/short_volume/",
            "rel_clean": "fundamentals/short_volume/",
            "files": "2024+ 按年",
            "grain": "日 × ticker",
            "ncols": "15",
            "adj_time": "align_time=date",
            "cleaned": "✅",
            "schema": [
                ("<code>short_volume</code>", "float", "做空成交量"),
                ("<code>short_volume_ratio</code>", "float", "占比"),
                ("<code>total_volume</code>", "float", "总成交量"),
            ],
            "sample_cols": ["ticker", "date", "short_volume", "ratio%", "total_volume"],
            "sample_rows": [
                ["A", "2024-02-06", "476664", "70.1", "679938"],
                ["AA", "2024-02-06", "1129527", "30.34", "3723314"],
                ["AAA", "2024-02-06", "1049", "57.61", "1821"],
            ],
        },
        {
            "title": "4.2.7 流通股 · stocks_floats",
            "rel": "fundamentals/stocks_floats/",
            "rel_clean": "fundamentals/stocks_floats/",
            "files": "<code>stocks_floats_all.parquet</code>",
            "grain": "事件频；effective_date",
            "ncols": "4",
            "adj_time": "align_time=effective_date",
            "cleaned": "✅",
            "schema": [
                ("<code>free_float</code>", "float", "自由流通股数"),
                ("<code>free_float_percent</code>", "float", "占比"),
            ],
            "sample_cols": ["ticker", "free_float", "free_float_percent", "effective_date"],
            "sample_rows": [
                ["A", "282591673", "99.7", "2026-01-08"],
                ["AA", "258394311", "99.8", "2026-01-29"],
                ["AABVF", "133442193", "83.04", "2026-01-02"],
            ],
        },
    ]


def add_corp_filing_news_meta(d):
    d["4.3 公司行动"] = [
        {
            "title": "4.3.1 分红 · dividends",
            "rel": "corporate_actions/dividends/",
            "rel_clean": "corporate_actions/dividends/",
            "files": "<code>dividends_{YYYY}.parquet</code>（28 文件）",
            "grain": "一次分红事件",
            "ncols": "12",
            "adj_time": "align_time=ex_dividend_date；含 historical_adjustment_factor",
            "cleaned": "✅",
            "warn": "2003+ 每年恰好 2000 行",
            "schema": [
                ("<code>ex_dividend_date</code>", "date", "除息日"),
                ("<code>cash_amount</code>", "float", "每股现金"),
                ("<code>historical_adjustment_factor</code>", "float", "复权因子（可 null）"),
            ],
            "sample_cols": ["ticker", "ex_dividend_date", "cash_amount", "currency", "pay_date"],
            "sample_rows": [
                ["CCU", "2002-12-31", "0.0039", "USD", "2003-01-17"],
                ["MCPO.Y", "2002-12-30", "0.55555", "USD", "2003-04-02"],
                ["CAVB", "2002-12-27", "0.05", "USD", "2003-01-10"],
            ],
        },
        {
            "title": "4.3.2 拆股 · splits",
            "rel": "corporate_actions/splits/",
            "rel_clean": "corporate_actions/splits/",
            "files": "39 个按年文件 1978–2026",
            "grain": "一次拆股事件",
            "ncols": "7",
            "adj_time": "execution_date；historical_adjustment_factor",
            "cleaned": "✅",
            "sample_cols": ["ticker", "execution_date", "split_from", "split_to", "adj_factor"],
            "sample_rows": [
                ["AMD", "1983-08-22", "1", "2", "0.25"],
                ["TER", "1983-08-01", "1", "2", "0.125"],
                ["INTC", "1983-07-01", "1", "2", "0.010417"],
            ],
        },
        {
            "title": "4.3.3 IPO · ipos",
            "rel": "corporate_actions/ipos/",
            "rel_clean": "corporate_actions/ipos/",
            "files": "<code>ipos_all.parquet</code>",
            "grain": "一条 IPO 记录",
            "ncols": "20",
            "adj_time": "listing_date / announced_date",
            "cleaned": "✅",
            "sample_cols": ["ticker", "issuer_name", "ipo_status", "announced_date", "listing_date", "issue_price"],
            "sample_rows": [
                ["LHI", "Living Homeopathy…", "pending", "2025-05-23", "null", "5.0"],
                ["ACGCU", "ACP Holdings…", "pending", "2026-03-06", "null", "10.0"],
                ["QREDU", "QuasarEdge…", "pending", "2026-03-05", "null", "10.0"],
            ],
        },
    ]
    d["4.4 SEC 申报"] = [
        {
            "title": "4.4.1 风险因子文本 · risk_factors",
            "rel": "filing/risk_factors/",
            "rel_clean": "filing/risk_factors/",
            "files": "按年 2015–2026；每年约 2000 行",
            "grain": "1 条风险描述 / 申报",
            "ncols": "7",
            "adj_time": "filing_date",
            "cleaned": "✅",
            "sample_cols": ["ticker", "filing_date", "primary_category", "supporting_text(截断)"],
            "sample_rows": [
                ["IBCP", "2026-03-06", "technology_and_information", "Emerging digital assets…"],
                ["IBCP", "2026-03-06", "regulatory_and_compliance", "Changes in regulation…"],
                ["IBCP", "2026-03-06", "cybersecurity…", "third party vendors…"],
            ],
        },
        {
            "title": "4.4.2 SEC 索引 · sec_edgar_index",
            "rel": "filing/sec_edgar_index/",
            "rel_clean": "filing/sec_edgar_index/",
            "files": "<code>sec_edgar_index_all.parquet</code>（10000 行快照）",
            "grain": "一份 EDGAR 申报",
            "ncols": "7",
            "adj_time": "filing_date（可 null）；分钟 PiT 用 acceptance（PREPROC-022）",
            "cleaned": "✅",
            "warn": "ticker ~48% 空",
            "sample_cols": ["ticker", "cik", "form_type", "filing_date", "accession_number"],
            "sample_rows": [
                ["HOC", "0000048039", "null", "null", "0001047469-02-001871"],
                ["SOBI", "0000934860", "null", "null", "0000927089-02-000022"],
                ["null", "0000062741", "S-4", "null", "0000950131-02-002678"],
            ],
        },
        {
            "title": "4.4.3 风险分类字典 · risk_categories",
            "rel": "filing/risk_categories/",
            "rel_clean": "filing/risk_categories/",
            "files": "140 行字典",
            "grain": "三级分类条目",
            "ncols": "6",
            "adj_time": "无 ticker",
            "cleaned": "✅",
            "sample_cols": ["primary", "secondary", "tertiary", "description(截断)"],
            "sample_rows": [
                ["governance…", "organizational…", "performance…", "Risk from inadequate…"],
                ["governance…", "organizational…", "communication…", "Risk from poor internal…"],
                ["governance…", "organizational…", "structure…", "Risk from inadequate structure…"],
            ],
        },
        {
            "title": "4.4.4 10-K 章节全文 · 10k_sections（未下载）",
            "rel": "filing/10k_sections/",
            "rel_clean": "—",
            "files": "目录存在，<strong>0</strong> 个 parquet",
            "grain": "预期：1 章节 × 1 申报",
            "ncols": "—",
            "adj_time": "预期 filing_date / accession",
            "cleaned": "—",
            "warn": "download_all_history 默认未纳入",
            "schema": [("—", "—", "见 FEAT-006 单独下载")],
            "sample_cols": ["状态", "说明"],
            "sample_rows": [["未下载", "NLP/长文本需单独任务"]],
        },
        {
            "title": "4.4.5 8-K 全文 · 8k_text（未下载）",
            "rel": "filing/8k_text/",
            "rel_clean": "—",
            "files": "目录存在，<strong>0</strong> 个 parquet",
            "grain": "预期：1 份 8-K 文本",
            "ncols": "—",
            "adj_time": "预期 filing_date",
            "cleaned": "—",
            "warn": "同上",
            "schema": [("—", "—", "见 FEAT-006")],
            "sample_cols": ["状态", "说明"],
            "sample_rows": [["未下载", "事件驱动 NLP 需单独任务"]],
        },
    ]
    d["4.5 新闻"] = [
        {
            "title": "4.5.1 新闻 · news",
            "rel": "news/news/",
            "rel_clean": "news/news/",
            "files": "<code>news_all.parquet</code>（2000 行）",
            "grain": "1 条新闻；cleaned：1 新闻 × 1 ticker",
            "ncols": "12+（含嵌套 publisher/insights）",
            "adj_time": "published_utc 秒级；事件研究映射下一交易日",
            "cleaned": "✅",
            "warn": "非全量历史",
            "schema": [
                ("<code>id</code>", "UUID", "主键"),
                ("<code>title</code>", "string", "标题"),
                ("<code>published_utc</code>", "ISO UTC", "发表时刻"),
                ("<code>tickers</code>", "string[]", "关联标的"),
                ("<code>insights[].sentiment</code>", "string", "情感"),
            ],
            "sample_cols": ["published_utc", "tickers", "title(截断)", "sentiment"],
            "sample_rows": [
                ["2026-03-10T14:00:00Z", "SYANY", "Staff elected to AL Sydbank…", "neutral"],
                ["2026-03-10T14:00:00Z", "SYANY", "Medarbejdervalg til AL…", "neutral"],
                ["2026-03-10T13:40:26Z", "FTCI", "FTC Solar 1-GW Deal…", "positive"],
            ],
        },
    ]
    d["4.6 元数据/参考"] = [
        {
            "title": "4.6.1 证券主数据 · all_tickers",
            "rel": "tickers/all_tickers/",
            "rel_clean": "tickers/all_tickers/",
            "files": "<code>all_tickers_all.parquet</code>（2000 行快照）",
            "grain": "1 个 ticker 一条",
            "ncols": "12",
            "adj_time": "last_updated_utc",
            "cleaned": "✅",
            "warn": "非全历史宇宙",
            "schema": [
                ("<code>ticker</code>", "string", "代码"),
                ("<code>cik</code>", "string", "CIK"),
                ("<code>composite_figi</code>", "string", "FIGI（实体 ID）"),
                ("<code>active</code>", "bool", "是否活跃"),
            ],
            "sample_cols": ["ticker", "name", "type", "exchange", "active", "cik"],
            "sample_rows": [
                ["A", "Agilent Technologies Inc.", "CS", "XNYS", "true", "0001090872"],
                ["AA", "Alcoa Corporation", "CS", "XNYS", "true", "0001675149"],
                ["AAA", "Alternative Access… ETF", "ETF", "ARCX", "true", "0001776878"],
            ],
        },
        {
            "title": "4.6.2 证券类型 · ticker_types",
            "rel": "tickers/ticker_types/",
            "rel_clean": "tickers/ticker_types/",
            "files": "25 行字典",
            "grain": "类型码",
            "ncols": "4",
            "adj_time": "—",
            "cleaned": "✅",
            "sample_cols": ["code", "description", "asset_class"],
            "sample_rows": [["CS", "Common Stock", "stocks"], ["PFD", "Preferred Stock", "stocks"], ["WARRANT", "Warrant", "stocks"]],
        },
        {
            "title": "4.6.3 交易所 · exchanges",
            "rel": "market_operations/exchanges/",
            "rel_clean": "market_operations/exchanges/",
            "files": "52 行",
            "grain": "交易所",
            "ncols": "7",
            "adj_time": "—",
            "cleaned": "✅",
            "sample_cols": ["id", "name", "mic", "asset_class"],
            "sample_rows": [["1", "NYSE American", "XASE", "stocks"], ["2", "Nasdaq BX", "XBOS", "stocks"], ["3", "NYSE National", "XCIS", "stocks"]],
        },
        {
            "title": "4.6.4 节假日 · market_holidays",
            "rel": "market_operations/market_holidays/",
            "rel_clean": "market_operations/market_holidays/",
            "files": "24 行（仅 2026–2027）",
            "grain": "休市日 × 交易所",
            "ncols": "4",
            "adj_time": "date；配合 ENG-011 提前收盘",
            "cleaned": "✅",
            "warn": "历史休市需从 SIP 反推 𝒯",
            "sample_cols": ["date", "exchange", "name", "status"],
            "sample_rows": [["2026-04-03", "NYSE", "Good Friday", "closed"], ["2026-04-03", "NASDAQ", "Good Friday", "closed"], ["2026-05-25", "NASDAQ", "Memorial Day", "closed"]],
        },
        {
            "title": "4.6.5 条件码 · condition_codes",
            "rel": "market_operations/condition_codes/",
            "rel_clean": "market_operations/condition_codes/",
            "files": "130 行",
            "grain": "成交/报价条件字典",
            "ncols": "6",
            "adj_time": "—",
            "cleaned": "✅",
            "schema": [("<code>id</code>", "int", "条件 ID"), ("<code>name</code>", "string", "名称"), ("<code>type</code>", "string", "类型")],
            "sample_cols": ["id", "type", "name"],
            "sample_rows": [["0", "regular", "Regular Trade"], ["1", "buy_or_sell_side", "Sell Side"], ["2", "buy_or_sell_side", "Buy Side"]],
        },
    ]


def build_catalog_html():
    d = dict(DATASETS)
    add_fundamentals(d)
    add_corp_filing_news_meta(d)

    parts = []
    for cat, items in [
        ("4.1 行情", d["4.1 行情"]),
        ("4.2 基本面", d["4.2 基本面"]),
        ("4.3 公司行动", d["4.3 公司行动"]),
        ("4.4 SEC 申报", d["4.4 SEC 申报"]),
        ("4.5 新闻", d["4.5 新闻"]),
        ("4.6 元数据/参考", d["4.6 元数据/参考"]),
    ]:
        inner = "\n".join(dataset_block(m) for m in items)
        parts.append(f"""
      <details class="fold">
        <summary><strong>{cat}</strong>（{len(items)} 个数据集）</summary>
        <div class="fold-body">{inner}</div>
      </details>""")

    return f"""
    <section id="sec-data-catalog">
      <h2><span class="sec-badge">数据</span> 全库 26 源数据形态手册（仅看本页即可）</h2>
      <p class="section-desc">
        根目录 <code>{ROOT}/</code> · 扫描截止 <strong>2026-03-05</strong>。
        下文<strong>全部用表格</strong>展示路径、分区、频率、列与样例行，无需打开 Parquet 或姊妹 MD。
      </p>

      <div class="box-info">
        <strong>两层目录：</strong><br>
        Layer 0 <code>raw_massive_data/</code>（~10 TB，22,806 文件）— 供应商原始字段<br>
        Layer 1 <code>cleaned_massive_data/</code>（~243 GB）— 追加 <code>ticker</code>、<code>align_time</code> 等 11 列标准列，<strong>不改 OHLCV 数值</strong>
      </div>

      <h3>一、26 源总览（路径 · 文件数 · 频率 · 列数）</h3>
      {build_overview()}

      <h3>二、两套日 K 不可混用（复权对照）</h3>
      {tbl(["数据集", "文件夹", "复权", "时间字段", "适用"], [
        ["REST 日线", "<code>aggregate_bars/daily_market_summary/</code>", "拆股复权", "<code>trade_date</code>", "长周期动量/均线"],
        ["SIP 日 K", "<code>us_stocks_sip/day_aggs_v1/</code>", "不复权", "<code>window_start</code>→04:00 UTC", "生产锚点、需自行复权"],
      ])}

      <h3>三、cleaned 层统一追加列（所有已清洗源）</h3>
      {tbl(["列名", "含义"], [
        ["<code>source</code>", "数据源路径标识"],
        ["<code>dataset_type</code>", "如 market_bar / financial_statement"],
        ["<code>frequency</code>", "daily / minute / quarterly_annual 等"],
        ["<code>ticker</code>", "大写、去空格；报表由 tickers[] explode"],
        ["<code>align_time</code>", "UTC 对齐时刻（PiT / join 键）"],
        ["<code>primary_key</code>", "去重主键哈希"],
        ["<code>align_time_source_column</code>", "如 filing_date / window_start"],
      ])}

      <h3>四、逐源详解（点击分类展开 → 再点各数据集）</h3>
      {"".join(parts)}

      <div class="box-warn">
        <strong>完整性红线：</strong>2024 三大报表与 short_* 各约 1 万行；dividends 2003+ 每年 2000 行；news/all_tickers 各 2000 行快照 — 详见 <a href="#sec-quality">§2b</a>、<a href="#sec-p0">§2 P0</a>。
      </div>
    </section>
"""


def patch_html(html: str) -> str:
    import re

    catalog = build_catalog_html()

    pat = re.compile(
        r'<details class="fold" id="sec-head-samples">.*?</details>\s*',
        re.DOTALL,
    )
    new_fold = f'''<details class="fold" id="sec-head-samples">
        <summary>📋 快速入口：26 源总览表（点击展开摘要）</summary>
        <div class="fold-body">
          <p>完整形态手册（路径+列+样例表格）见下方 <a href="#sec-data-catalog"><strong>§数据 全库 26 源手册</strong></a>。</p>
          {build_overview()}
        </div>
      </details>

{catalog}
'''
    if "sec-data-catalog" in html:
        if "<!DOCTYPE" not in html[:200]:
            raise SystemExit("HTML shell missing — run restore_html_shell.py first")
        pat_cat = re.compile(
            r'<section id="sec-data-catalog">.*?</section>',
            re.DOTALL,
        )
        html = pat_cat.sub(catalog.strip() + "\n\n", html, count=1)
        # refresh quick fold overview only
        if pat.search(html):
            html = pat.sub(new_fold, html, count=1)
    elif pat.search(html):
        html = pat.sub(new_fold, html, count=1)
    elif "sec-data-catalog" not in html:
        html = html.replace(
            '<section id="sec-ok">',
            new_fold + "\n    <section id=\"sec-ok\">",
            1,
        )

    # 删除 catalog 与 sec-ok 之间遗留的 JSON 样例折叠块
    orphan = re.compile(
        r'(</section>\s*)((?:<details class="sample-fold">.*?</details>\s*)+)(?:<p><a href="Massive原始数据报告\.md">.*?</p>\s*)?(?:</div>\s*</details>\s*)?(<section id="sec-ok">)',
        re.DOTALL,
    )
    html = orphan.sub(r"\1\3", html, count=1)

    for old_ver in (
        "版本 v3.2 HTML 终审完整版（12a–12e）",
        "版本 v3.1",
    ):
        html = html.replace(old_ver, "版本 v3.3 HTML 全库数据形态手册")
    if "版本 v3.3 HTML 全库数据形态手册" not in html:
        html = html.replace(
            "Massive 数据治理与改进行动清单",
            "Massive 数据治理与改进行动清单",
            1,
        )
    return html


def main():
    html = HTML.read_text(encoding="utf-8")
    html = patch_html(html)
    HTML.write_text(html, encoding="utf-8")
    print("catalog injected, lines:", html.count("\n") + 1)
    print("sec-data-catalog:", "sec-data-catalog" in html)


if __name__ == "__main__":
    main()
