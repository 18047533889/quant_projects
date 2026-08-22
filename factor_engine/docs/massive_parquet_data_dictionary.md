# Massive Parquet 数据字典（含字段含义）

- 生成时间: 2026-03-13T13:32:45.280925Z
- 数据根目录: `~/quant_projects/data/us_stock/massive_parquet`
- 字段释义文档根目录: `~/quant_projects/docs/massive_rest_api`
- 数据集数量: **24**
- Parquet 文件总数: **22806**
- 读取错误数: **0**（已自动跳过异常文件）
- 已索引文档字段数: **236**

> 说明：字段含义优先来自 `rest_api_doc` 的 Response Attributes；中文释义列已统一转换为中文表述。

> **第 5 版更改-shw**：本文档仅描述 **Parquet 字段与数据集**，不包含因子算子定义。因子侧算子扩展与 DSL 约定见 [`operators_semantics.md`](operators_semantics.md)；版本级代码变更见 [`changelog_shw.md`](changelog_shw.md)。
>
> **第 7 版更改-shw**：算子库新增 **清洗/技术指标/上下文/group_*** 等与数据列组合方式无关。

## 数据集总览

| 数据集 | 文件数 | 字段数 | 可读 | 年份分区 |
|---|---:|---:|:---:|---|
| `aggregate_bars/daily_market_summary` | 23 | 10 | Y | - |
| `corporate_actions/dividends` | 28 | 9 | Y | - |
| `corporate_actions/ipos` | 1 | 20 | Y | - |
| `corporate_actions/splits` | 39 | 7 | Y | - |
| `filing/risk_categories` | 1 | 5 | Y | - |
| `filing/risk_factors` | 12 | 7 | Y | - |
| `filing/sec_edgar_index` | 1 | 7 | Y | - |
| `fundamentals/balance_sheet` | 18 | 38 | Y | - |
| `fundamentals/cash_flow_statement` | 17 | 32 | Y | - |
| `fundamentals/financials_ratios` | 1 | 23 | Y | - |
| `fundamentals/income_statement` | 17 | 34 | Y | - |
| `fundamentals/short_interest` | 10 | 5 | Y | - |
| `fundamentals/short_volume` | 3 | 15 | Y | - |
| `fundamentals/stocks_floats` | 1 | 4 | Y | - |
| `market_operations/condition_codes` | 1 | 11 | Y | - |
| `market_operations/exchanges` | 1 | 10 | Y | - |
| `market_operations/market_holidays` | 1 | 6 | Y | - |
| `news/news` | 1 | 12 | Y | - |
| `tickers/all_tickers` | 1 | 12 | Y | - |
| `tickers/ticker_types` | 1 | 4 | Y | - |
| `us_stocks_sip/day_aggs_v1` | 5657 | 8 | Y | 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, ... |
| `us_stocks_sip/minute_aggs_v1` | 5657 | 8 | Y | 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, ... |
| `us_stocks_sip/quotes_v1` | 5657 | 14 | Y | 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, ... |
| `us_stocks_sip/trades_v1` | 5657 | 13 | Y | 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, ... |

## aggregate_bars/daily_market_summary

- 文件数: 23
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/aggregate_bars/daily_market_summary/daily_market_summary_2004.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `T` | `object` | The exchange symbol that this item is traded under. | 交易标的代码。 | aggregate_bars/daily_market_summary.md |
| `v` | `float64` | The trading volume of the symbol in the given time period. | 成交量。 | aggregate_bars/custom_bars.md<br>aggregate_bars/daily_market_summary.md<br>aggregate_bars/previous_day_bar.md |
| `vw` | `float64` | The volume weighted average price. | 成交量加权平均价（VWAP）。 | aggregate_bars/custom_bars.md<br>aggregate_bars/daily_market_summary.md<br>aggregate_bars/previous_day_bar.md |
| `o` | `float64` | The open price for the symbol in the given time period. | 开盘价。 | aggregate_bars/custom_bars.md<br>aggregate_bars/daily_market_summary.md<br>aggregate_bars/previous_day_bar.md |
| `c` | `float64` | The close price for the symbol in the given time period. | 收盘价。 | aggregate_bars/custom_bars.md<br>aggregate_bars/daily_market_summary.md<br>aggregate_bars/previous_day_bar.md |
| `h` | `float64` | The highest price for the symbol in the given time period. | 最高价。 | aggregate_bars/custom_bars.md<br>aggregate_bars/daily_market_summary.md<br>aggregate_bars/previous_day_bar.md |
| `l` | `float64` | The lowest price for the symbol in the given time period. | 最低价。 | aggregate_bars/custom_bars.md<br>aggregate_bars/daily_market_summary.md<br>aggregate_bars/previous_day_bar.md |
| `t` | `int64` | The Unix millisecond timestamp for the start of the aggregate window. | 时间戳（聚合窗口时间）。 | aggregate_bars/custom_bars.md<br>aggregate_bars/daily_market_summary.md<br>aggregate_bars/previous_day_bar.md |
| `n` | `float64` | The number of transactions in the aggregate window. | 成交笔数。 | aggregate_bars/custom_bars.md<br>aggregate_bars/daily_market_summary.md<br>aggregate_bars/previous_day_bar.md |
| `trade_date` | `object` | 交易日期（通常为 YYYY-MM-DD）。 | 交易日期。 | aggregate_bars/daily_market_summary.md |

### 样本记录（前 3 条）

```json
[
  {
    "T": "HRVE",
    "v": 175.0,
    "vw": 3.5944,
    "o": 3.56,
    "c": 3.64,
    "h": 3.64,
    "l": 3.56,
    "t": 1073077200000,
    "n": 4.0,
    "trade_date": "2004-01-02"
  },
  {
    "T": "PAS",
    "v": 166400.0,
    "vw": 17.1101,
    "o": 17.1,
    "c": 17.12,
    "h": 17.2,
    "l": 17.01,
    "t": 1073077200000,
    "n": 442.0,
    "trade_date": "2004-01-02"
  },
  {
    "T": "SNBC",
    "v": 3734.2,
    "vw": 133.2745,
    "o": 131.75,
    "c": 130.75,
    "h": 136.8,
    "l": 130.0,
    "t": 1073077200000,
    "n": 114.0,
    "trade_date": "2004-01-02"
  }
]
```

## corporate_actions/dividends

- 文件数: 28
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/corporate_actions/dividends/dividends_2000.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `id` | `object` | Unique identifier for each dividend record | 记录唯一标识。 | corporate_actions/dividends.md<br>corporate_actions/splits.md<br>market_operations/condition_codes.md<br>market_operations/exchanges.md<br>news/news.md |
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `record_date` | `object` | Date when shareholders must be on record to be eligible for the dividend payment | 股权登记日。 | corporate_actions/dividends.md |
| `pay_date` | `object` | Date when the dividend payment is distributed to shareholders | 分红派发日。 | corporate_actions/dividends.md |
| `ex_dividend_date` | `object` | Date when the stock begins trading without the dividend value | 除息日（含息转为除息交易的日期）。 | corporate_actions/dividends.md |
| `frequency` | `int64` | How many times per year this dividend is expected to occur. A value of 0 means the distribution is non-recurring or irregular (e.g., special, supplemental, or a one-off dividend). Other possible values include 1 (annual), 2 (semi-annual), 3 (trimester), 4 (quarterly), 12 (monthly), 24 (bi-monthly), 52 (weekly), 104 (bi-weekly), and 365 (daily) depending on the issuer's declared or inferred payout cadence. | 分红频率（每年分红次数，如 1=年、4=季、12=月）。 | corporate_actions/dividends.md |
| `cash_amount` | `float64` | Original dividend amount per share in the specified currency | 每股现金分红金额。 | corporate_actions/dividends.md |
| `currency` | `object` | Currency code for the dividend payment (e.g., USD, CAD) | 币种。 | corporate_actions/dividends.md |
| `distribution_type` | `object` | Classification describing the nature of this dividend's recurrence pattern: recurring (paid on a regular schedule), special (one-time or commemorative), supplemental (extra beyond the regular schedule), irregular (unpredictable or non-recurring), unknown (cannot be classified from available data) | 分红类型（经常性、特别分红、补充分红、不规则等）。 | corporate_actions/dividends.md |

### 样本记录（前 3 条）

```json
[
  {
    "id": "E4a7d4e17e772232caf90d14c98574c1e7b15c49c94246678bd28b603e0501954",
    "ticker": "CHVKF",
    "record_date": "2000-08-16",
    "pay_date": "2000-08-18",
    "ex_dividend_date": "2000-08-15",
    "frequency": 1,
    "cash_amount": 0.15,
    "currency": "CNY",
    "distribution_type": "recurring"
  }
]
```

## corporate_actions/ipos

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/corporate_actions/ipos/ipos_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `last_updated` | `object` | The date when the IPO event was last modified. | 最后更新时间。 | corporate_actions/IPOS.md<br>snapshots/unified_snapshot.md |
| `announced_date` | `object` | The date when the IPO event was announced. | 公告日期。 | corporate_actions/IPOS.md |
| `issuer_name` | `object` | Name of issuer. | 发行人名称。 | corporate_actions/IPOS.md<br>filing/10-K_Sections.md<br>filing/sec_edgar_index.md |
| `currency_code` | `object` | Underlying currency of the security. | 币种代码。 | corporate_actions/IPOS.md |
| `final_issue_price` | `float64` | The price set by the company and its underwriters before the IPO goes live. | 最终发行价。 | corporate_actions/IPOS.md |
| `max_shares_offered` | `float64` | The upper limit of the shares that the company is offering to investors. | 最高发行股数。 | corporate_actions/IPOS.md |
| `lowest_offer_price` | `float64` | The lowest price within the IPO price range that the company is willing to offer its shares to investors. | 发行区间下限价格。 | corporate_actions/IPOS.md |
| `highest_offer_price` | `float64` | The highest price within the IPO price range that the company might use to price the shares. | 发行区间上限价格。 | corporate_actions/IPOS.md |
| `total_offer_size` | `float64` | The total amount raised by the company for IPO. | 总募资规模。 | corporate_actions/IPOS.md |
| `primary_exchange` | `object` | Market Identifier Code (MIC) of the primary exchange where the security is listed. The Market Identifier Code (MIC) (ISO 10383) is a unique identification code used to identify securities trading exchanges, regulated and non-regulated trading markets. | 主上市交易所（MIC 代码）。 | corporate_actions/IPOS.md<br>tickers/all_tickers.md |
| `shares_outstanding` | `float64` | The total number of shares that the company has issued and are held by investors. | 已发行在外股份数。 | corporate_actions/IPOS.md |
| `security_type` | `object` | The classification of the stock. For example, "CS" stands for Common Stock. | 证券类型。 | corporate_actions/IPOS.md |
| `lot_size` | `float64` | The minimum number of shares that can be bought or sold in a single transaction. | 最小交易单位（每手股数）。 | corporate_actions/IPOS.md |
| `security_description` | `object` | Description of the security. | 证券描述。 | corporate_actions/IPOS.md |
| `ipo_status` | `object` | The status of the IPO event. IPO events start out as status "rumor" or "pending". On listing day, the status changes to "new". After the listing day, the status changes to "history".  The status "direct_listing_process" corresponds to a type of offering where, instead of going through all the IPO processes, the company decides to list its shares directly on an exchange, without using an investment bank or other intermediaries. This is called a direct listing, direct placement, or direct public offering (DPO). | 状态。 | corporate_actions/IPOS.md |
| `us_code` | `object` | This is a unique nine-character alphanumeric code that identifies a North American financial security for the purposes of facilitating clearing and settlement of trades. | 北美证券识别码（九位字母数字编码）。 | corporate_actions/IPOS.md |
| `isin` | `object` | International Securities Identification Number. This is a unique twelve-digit code that is assigned to every security issuance in the world. | 国际证券识别码（ISIN）。 | corporate_actions/IPOS.md |
| `min_shares_offered` | `float64` | The lower limit of shares that the company is willing to sell in the IPO. | 最低发行股数。 | corporate_actions/IPOS.md |
| `listing_date` | `object` | First trading date for the newly listed entity. | 上市首日。 | corporate_actions/IPOS.md |

### 样本记录（前 3 条）

```json
[
  {
    "ticker": "LHI",
    "last_updated": "2026-03-09",
    "announced_date": "2025-05-23",
    "issuer_name": "Living Homeopathy International Ltd.",
    "currency_code": "USD",
    "final_issue_price": 5.0,
    "max_shares_offered": 3750000.0,
    "lowest_offer_price": 4.0,
    "highest_offer_price": 6.0,
    "total_offer_size": 18750000.0,
    "primary_exchange": "XNAS",
    "shares_outstanding": 21750000.0,
    "security_type": "CS",
    "lot_size": 100.0,
    "security_description": "Ordinary Shares - Class A",
    "ipo_status": "pending",
    "us_code": null,
    "isin": null,
    "min_shares_offered": null,
    "listing_date": null
  },
  {
    "ticker": "ACGCU",
    "last_updated": "2026-03-09",
    "announced_date": "2026-03-06",
    "issuer_name": "ACP Holdings Acquisition Corp.",
    "currency_code": "USD",
    "final_issue_price": 10.0,
    "max_shares_offered": 20000000.0,
    "lowest_offer_price": 10.0,
    "highest_offer_price": 10.0,
    "total_offer_size": 200000000.0,
    "primary_exchange": "XNAS",
    "shares_outstanding": 20390000.0,
    "security_type": "SP",
    "lot_size": 100.0,
    "security_description": "Units 1 Ord Cls A  1/3 War",
    "ipo_status": "pending",
    "us_code": null,
    "isin": null,
    "min_shares_offered": null,
    "listing_date": null
  },
  {
    "ticker": "QREDU",
    "last_updated": "2026-03-09",
    "announced_date": "2026-03-05",
    "issuer_name": "QuasarEdge Acquisition Corp",
    "currency_code": "USD",
    "final_issue_price": 10.0,
    "max_shares_offered": 10000000.0,
    "lowest_offer_price": 10.0,
    "highest_offer_price": 10.0,
    "total_offer_size": 100000000.0,
    "primary_exchange": "XNAS",
    "shares_outstanding": null,
    "security_type": "SP",
    "lot_size": 100.0,
    "security_description": "Units 1 Ord Shs  1 Rts",
    "ipo_status": "pending",
    "us_code": null,
    "isin": null,
    "min_shares_offered": null,
    "listing_date": null
  }
]
```

## corporate_actions/splits

- 文件数: 39
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/corporate_actions/splits/splits_1978.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `id` | `object` | Unique identifier for each dividend record | 记录唯一标识。 | corporate_actions/dividends.md<br>corporate_actions/splits.md<br>market_operations/condition_codes.md<br>market_operations/exchanges.md<br>news/news.md |
| `execution_date` | `object` | Date when the stock split was applied and shares adjusted | 执行日期。 | corporate_actions/splits.md |
| `split_from` | `float64` | Denominator of the split ratio (old shares) | 拆股比例分母（旧股数）。 | corporate_actions/splits.md |
| `split_to` | `float64` | Numerator of the split ratio (new shares) | 拆股比例分子（新股数）。 | corporate_actions/splits.md |
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `adjustment_type` | `object` | Classification of the share-change event. Possible values include: forward_split (share count increases), reverse_split (share count decreases), stock_dividend (shares issued as a dividend) | 调整类型（前复权拆股/反向拆股/股票股利等）。 | corporate_actions/splits.md |
| `historical_adjustment_factor` | `float64` | Cumulative adjustment factor used to offset dividend effects on historical prices. To adjust a historical price for dividends: for a price on date D, find the first dividend whose `ex_dividend_date` is after date D and multiply the price by that dividend's `historical_adjustment_factor`. | 历史调整因子（用于将历史价格按当前股本口径调整）。 | corporate_actions/dividends.md<br>corporate_actions/splits.md |

### 样本记录（前 3 条）

```json
[
  {
    "id": "Pef962e8ce572df20933cdaac3a2d2d25e3a77cc910f2c84dbe66235bc780538b",
    "execution_date": "1978-10-25",
    "split_from": 2.0,
    "split_to": 3.0,
    "ticker": "AMD",
    "adjustment_type": "forward_split",
    "historical_adjustment_factor": 0.037037
  }
]
```

## filing/risk_categories

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/filing/risk_categories/risk_categories_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `primary_category` | `object` | Top-level risk category | 一级风险分类。 | filing/risk_categories.md<br>filing/risk_factors.md |
| `secondary_category` | `object` | Mid-level risk category | 二级风险分类。 | filing/risk_categories.md<br>filing/risk_factors.md |
| `tertiary_category` | `object` | Most specific risk classification | 三级风险分类。 | filing/risk_categories.md<br>filing/risk_factors.md |
| `description` | `object` | Detailed explanation of what this risk category encompasses, including specific examples and potential impacts | 描述。 | filing/risk_categories.md<br>market_operations/condition_codes.md<br>news/news.md<br>tickers/ticker_types.md |
| `taxonomy` | `float64` | Version identifier (e.g., '1.0', '1.1') for the taxonomy | 分类体系版本。 | filing/risk_categories.md |

### 样本记录（前 3 条）

```json
[
  {
    "primary_category": "governance_and_stakeholder",
    "secondary_category": "organizational_and_management",
    "tertiary_category": "performance_management_and_accountability",
    "description": "Risk from inadequate performance management systems, unclear accountability structures, or ineffective measurement and incentive systems that could affect employee performance, goal achievement, and organizational effectiveness.",
    "taxonomy": 1.0
  },
  {
    "primary_category": "governance_and_stakeholder",
    "secondary_category": "organizational_and_management",
    "tertiary_category": "communication_and_coordination",
    "description": "Risk from poor internal communication, lack of coordination between departments or business units, or information silos that could affect operational efficiency, strategic execution, and organizational alignment and performance.",
    "taxonomy": 1.0
  },
  {
    "primary_category": "governance_and_stakeholder",
    "secondary_category": "organizational_and_management",
    "tertiary_category": "organizational_structure_and_reporting",
    "description": "Risk from inadequate organizational structure, unclear reporting relationships, or ineffective management hierarchies that could create communication problems, decision-making delays, accountability gaps, and operational inefficiencies.",
    "taxonomy": 1.0
  }
]
```

## filing/risk_factors

- 文件数: 12
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/filing/risk_factors/risk_factors_2015.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `cik` | `object` | SEC Central Index Key (CIK) identifying the filing entity. | SEC 中央索引编号（CIK）。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>fundamentals/ratios.md<br>tickers/all_tickers.md |
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `primary_category` | `object` | Top-level risk category | 一级风险分类。 | filing/risk_categories.md<br>filing/risk_factors.md |
| `secondary_category` | `object` | Mid-level risk category | 二级风险分类。 | filing/risk_categories.md<br>filing/risk_factors.md |
| `tertiary_category` | `object` | Most specific risk classification | 三级风险分类。 | filing/risk_categories.md<br>filing/risk_factors.md |
| `filing_date` | `object` | Date when the filing was submitted to the SEC (formatted as YYYY-MM-DD). | 申报/提交日期。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `supporting_text` | `object` | Snippet of text to support the given label | 支持该标签的文本片段。 | filing/risk_factors.md |

### 样本记录（前 3 条）

```json
[
  {
    "cik": "0000039311",
    "ticker": "IBCP",
    "primary_category": "technology_and_information",
    "secondary_category": "digital_transformation_and_innovation",
    "tertiary_category": "technology_obsolescence_and_evolution",
    "filing_date": "2026-03-06",
    "supporting_text": "Emerging digital assets and technologies may disrupt our business and adversely affect our results. Rapid innovation in financial technology - including stablecoins and other digital assets, distributed ledger technologies, real-time payment networks, embedded banking, and artificial intelligence - may change how consumers and businesses store value, make payments, access credit, and obtain financial services."
  },
  {
    "cik": "0000039311",
    "ticker": "IBCP",
    "primary_category": "regulatory_and_compliance",
    "secondary_category": "industry_regulation",
    "tertiary_category": "regulatory_compliance_and_changes",
    "filing_date": "2026-03-06",
    "supporting_text": "Changes in regulation or oversight may have a material adverse impact on our operations. We are subject to extensive regulation, supervision and examination by the Federal Reserve, the FDIC, the Michigan DIFS, the SEC and other regulatory bodies. Such regulation and supervision govern the activities in which we may engage."
  },
  {
    "cik": "0000039311",
    "ticker": "IBCP",
    "primary_category": "technology_and_information",
    "secondary_category": "cybersecurity_and_data_protection",
    "tertiary_category": "third_party_data_security_and_vendors",
    "filing_date": "2026-03-06",
    "supporting_text": "We are also susceptible to cybersecurity risks faced by third party vendors on which we rely for components of our business infrastructure and data processing and which often have access to the sensitive data of our customers."
  }
]
```

## filing/sec_edgar_index

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/filing/sec_edgar_index/sec_edgar_index_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `cik` | `object` | SEC Central Index Key (CIK) identifying the filing entity. | SEC 中央索引编号（CIK）。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>fundamentals/ratios.md<br>tickers/all_tickers.md |
| `issuer_name` | `object` | Name of issuer. | 发行人名称。 | corporate_actions/IPOS.md<br>filing/10-K_Sections.md<br>filing/sec_edgar_index.md |
| `filing_url` | `object` | Direct URL to the filing on the SEC EDGAR website. | 链接。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/sec_edgar_index.md |
| `accession_number` | `object` | SEC accession number uniquely identifying the filing (e.g., '0000320193-24-000123'). | 该字段中文释义已补充。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/sec_edgar_index.md |
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `form_type` | `object` | SEC form type (e.g., '10-K', '10-Q', '8-K', 'S-1', '4', etc.). | SEC 表单类型（如 10-K/10-Q/8-K/S-1 等）。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/sec_edgar_index.md |
| `filing_date` | `object` | Date when the filing was submitted to the SEC (formatted as YYYY-MM-DD). | 申报/提交日期。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |

### 样本记录（前 3 条）

```json
[
  {
    "cik": "0000048039",
    "issuer_name": "HOLLY CORP",
    "filing_url": "https://www.sec.gov/Archives/edgar/data/48039/0001047469-02-001871.txt",
    "accession_number": "0001047469-02-001871",
    "ticker": "HOC",
    "form_type": null,
    "filing_date": null
  },
  {
    "cik": "0000934860",
    "issuer_name": "SOBIESKI BANCORP INC",
    "filing_url": "https://www.sec.gov/Archives/edgar/data/934860/0000927089-02-000022.txt",
    "accession_number": "0000927089-02-000022",
    "ticker": "SOBI",
    "form_type": null,
    "filing_date": null
  },
  {
    "cik": "0000062741",
    "issuer_name": "MARSHALL & ILSLEY CORP/WI/",
    "filing_url": "https://www.sec.gov/Archives/edgar/data/62741/0000950131-02-002678.txt",
    "accession_number": "0000950131-02-002678",
    "ticker": null,
    "form_type": "S-4",
    "filing_date": null
  }
]
```

## fundamentals/balance_sheet

- 文件数: 18
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/fundamentals/balance_sheet/balance_sheet_2010.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `accounts_payable` | `float64` | Amounts owed to suppliers and vendors for goods and services purchased on credit. | 应付账款。 | fundamentals/balance_sheets.md |
| `accrued_and_other_current_liabilities` | `float64` | Current liabilities not classified elsewhere, including accrued expenses, taxes payable, and other obligations due within one year. | 应计及其他流动负债。 | fundamentals/balance_sheets.md |
| `accumulated_other_comprehensive_income` | `float64` | Cumulative gains and losses that bypass the income statement, including foreign currency translation adjustments and unrealized gains/losses on securities. | 累计其他综合收益。 | fundamentals/balance_sheets.md |
| `additional_paid_in_capital` | `float64` | Amount received from shareholders in excess of the par or stated value of shares issued. | 资本公积（超面值实收资本）。 | fundamentals/balance_sheets.md |
| `cash_and_equivalents` | `float64` | Cash on hand and short-term, highly liquid investments that are readily convertible to known amounts of cash. | 现金及现金等价物。 | fundamentals/balance_sheets.md |
| `cik` | `object` | SEC Central Index Key (CIK) identifying the filing entity. | SEC 中央索引编号（CIK）。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>fundamentals/ratios.md<br>tickers/all_tickers.md |
| `commitments_and_contingencies` | `float64` | Disclosed amount related to contractual commitments and potential liabilities that may arise from uncertain future events. | 承诺与或有事项。 | fundamentals/balance_sheets.md |
| `common_stock` | `float64` | Par or stated value of common shares outstanding representing basic ownership in the company. | 普通股（账面口径）。 | fundamentals/balance_sheets.md |
| `debt_current` | `float64` | Short-term borrowings and the current portion of long-term debt due within one year. | 一年内到期债务/短期债务。 | fundamentals/balance_sheets.md |
| `deferred_revenue_current` | `float64` | Customer payments received in advance for goods or services to be delivered within one year. | 流动递延收入。 | fundamentals/balance_sheets.md |
| `filing_date` | `object` | Date when the filing was submitted to the SEC (formatted as YYYY-MM-DD). | 申报/提交日期。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `fiscal_quarter` | `int64` | The fiscal quarter number (1, 2, 3, or 4) for the reporting period. | 财季。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `fiscal_year` | `int64` | The fiscal year for the reporting period. | 财年。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `goodwill` | `float64` | Intangible asset representing the excess of purchase price over fair value of net assets acquired in business combinations. | 商誉。 | fundamentals/balance_sheets.md |
| `intangible_assets_net` | `float64` | Intangible assets other than goodwill, including patents, trademarks, and customer relationships, net of accumulated amortization. | 无形资产净额。 | fundamentals/balance_sheets.md |
| `inventories` | `float64` | Raw materials, work-in-process, and finished goods held for sale in the ordinary course of business. | 存货。 | fundamentals/balance_sheets.md |
| `long_term_debt_and_capital_lease_obligations` | `float64` | Long-term borrowings and capital lease obligations with maturities greater than one year. | 长期债务及资本租赁义务。 | fundamentals/balance_sheets.md |
| `noncontrolling_interest` | `float64` | Equity in consolidated subsidiaries not owned by the parent company, representing minority shareholders' ownership. | 少数股东权益（非控股权益）。 | fundamentals/balance_sheets.md<br>fundamentals/income_statement.md |
| `other_assets` | `float64` | Non-current assets not classified elsewhere, including long-term investments, deferred tax assets, and other long-term assets. | 其他非流动资产。 | fundamentals/balance_sheets.md |
| `other_current_assets` | `float64` | Current assets not classified elsewhere, including prepaid expenses, taxes receivable, and other assets expected to be converted to cash within one year. | 其他流动资产。 | fundamentals/balance_sheets.md |
| `other_equity` | `float64` | Equity components not classified elsewhere in shareholders' equity. | 其他权益项目。 | fundamentals/balance_sheets.md |
| `other_noncurrent_liabilities` | `float64` | Non-current liabilities not classified elsewhere, including deferred tax liabilities, pension obligations, and other long-term liabilities. | 其他非流动负债。 | fundamentals/balance_sheets.md |
| `period_end` | `object` | The last date of the reporting period, representing the specific point in time when the balance sheet snapshot was taken. | 报告期结束日期。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `preferred_stock` | `float64` | Par or stated value of preferred shares outstanding with preferential rights over common stock. | 优先股（账面口径）。 | fundamentals/balance_sheets.md |
| `property_plant_equipment_net` | `float64` | Tangible fixed assets used in operations, reported net of accumulated depreciation. | 固定资产净额（PP&E）。 | fundamentals/balance_sheets.md |
| `receivables` | `float64` | Amounts owed to the company by customers and other parties, primarily accounts receivable, net of allowances for doubtful accounts. | 应收款项净额。 | fundamentals/balance_sheets.md |
| `retained_earnings_deficit` | `float64` | Cumulative net income earned by the company less dividends paid to shareholders since inception. | 留存收益/累计亏损。 | fundamentals/balance_sheets.md |
| `short_term_investments` | `float64` | Marketable securities and other investments with maturities of one year or less that are not classified as cash equivalents. | 短期投资。 | fundamentals/balance_sheets.md |
| `tickers` | `object` | A list of ticker symbols under which the company is listed. Multiple symbols may indicate different share classes for the same company. | 证券代码列表（可能包含多股类别）。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>news/news.md |
| `timeframe` | `object` | The reporting period type. Possible values include: quarterly, annual. | 报告周期类型（季度/年度等）。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>snapshots/unified_snapshot.md |
| `total_assets` | `float64` | Sum of all current and non-current assets representing everything the company owns or controls. | 总资产。 | fundamentals/balance_sheets.md |
| `total_current_assets` | `float64` | Sum of all current assets expected to be converted to cash, sold, or consumed within one year. | 流动资产合计。 | fundamentals/balance_sheets.md |
| `total_current_liabilities` | `float64` | Sum of all liabilities expected to be settled within one year. | 流动负债合计。 | fundamentals/balance_sheets.md |
| `total_equity` | `float64` | Sum of all equity components representing shareholders' total ownership interest in the company. | 股东权益合计。 | fundamentals/balance_sheets.md |
| `total_equity_attributable_to_parent` | `float64` | Total shareholders' equity attributable to the parent company, excluding noncontrolling interests. | 归属于母公司股东权益。 | fundamentals/balance_sheets.md |
| `total_liabilities` | `float64` | Sum of all current and non-current liabilities representing everything the company owes. | 总负债。 | fundamentals/balance_sheets.md |
| `total_liabilities_and_equity` | `float64` | Sum of total liabilities and total equity, which should equal total assets per the fundamental accounting equation. | 负债和股东权益总计。 | fundamentals/balance_sheets.md |
| `treasury_stock` | `float64` | Cost of the company's own shares that have been repurchased and are held in treasury, typically reported as a negative value. | 库存股。 | fundamentals/balance_sheets.md |

### 样本记录（前 3 条）

```json
[
  {
    "accounts_payable": 28541000000.0,
    "accrued_and_other_current_liabilities": 20307000000.0,
    "accumulated_other_comprehensive_income": -3373000000.0,
    "additional_paid_in_capital": null,
    "cash_and_equivalents": 6578000000.0,
    "cik": "0000104169",
    "commitments_and_contingencies": 277000000.0,
    "common_stock": 4048000000.0,
    "debt_current": 7506000000.0,
    "deferred_revenue_current": null,
    "filing_date": "2010-06-04",
    "fiscal_quarter": 1,
    "fiscal_year": 2010,
    "goodwill": 14882000000.0,
    "intangible_assets_net": null,
    "inventories": 34391000000.0,
    "long_term_debt_and_capital_lease_obligations": 35665000000.0,
    "noncontrolling_interest": 1683000000.0,
    "other_assets": 3358000000.0,
    "other_current_assets": 3421000000.0,
    "other_equity": 0.0,
    "other_noncurrent_liabilities": 5835000000.0,
    "period_end": "2009-04-30",
    "preferred_stock": null,
    "property_plant_equipment_net": 96104000000.0,
    "receivables": 3356000000.0,
    "retained_earnings_deficit": 61556000000.0,
    "short_term_investments": null,
    "tickers": [
      "WMT"
    ],
    "timeframe": "quarterly",
    "total_assets": 162090000000.0,
    "total_current_assets": 47746000000.0,
    "total_current_liabilities": 56354000000.0,
    "total_equity": 63914000000.0,
    "total_equity_attributable_to_parent": 62231000000.0,
    "total_liabilities": 97899000000.0,
    "total_liabilities_and_equity": 162090000000.0,
    "treasury_stock": null
  },
  {
    "accounts_payable": 6004000000.0,
    "accrued_and_other_current_liabilities": 2990000000.0,
    "accumulated_other_comprehensive_income": -553000000.0,
    "additional_paid_in_capital": 2788000000.0,
    "cash_and_equivalents": 1371000000.0,
    "cik": "0000027419",
    "commitments_and_contingencies": null,
    "common_stock": 63000000.0,
    "debt_current": 1255000000.0,
    "deferred_revenue_current": null,
    "filing_date": "2010-05-28",
    "fiscal_quarter": 1,
    "fiscal_year": 2010,
    "goodwill": null,
    "intangible_assets_net": null,
    "inventories": 6993000000.0,
    "long_term_debt_and_capital_lease_obligations": 17514000000.0,
    "noncontrolling_interest": null,
    "other_assets": 861000000.0,
    "other_current_assets": 1735000000.0,
    "other_equity": 0.0,
    "other_noncurrent_liabilities": 2330000000.0,
    "period_end": "2009-05-02",
    "preferred_stock": null,
    "property_plant_equipment_net": 25800000000.0,
    "receivables": 7452000000.0,
    "retained_earnings_deficit": 11821000000.0,
    "short_term_investments": null,
    "tickers": [
      "TGT"
    ],
    "timeframe": "quarterly",
    "total_assets": 44212000000.0,
    "total_current_assets": 17551000000.0,
    "total_current_liabilities": 10249000000.0,
    "total_equity": 14119000000.0,
    "total_equity_attributable_to_parent": 14119000000.0,
    "total_liabilities": 30093000000.0,
    "total_liabilities_and_equity": 44212000000.0,
    "treasury_stock": null
  },
  {
    "accounts_payable": 812000000.0,
    "accrued_and_other_current_liabilities": 873000000.0,
    "accumulated_other_comprehensive_income": 116000000.0,
    "additional_paid_in_capital": 2893000000.0,
    "cash_and_equivalents": 1708000000.0,
    "cik": "0000039911",
    "commitments_and_contingencies": null,
    "common_stock": 55000000.0,
    "debt_current": null,
    "deferred_revenue_current": null,
    "filing_date": "2010-06-08",
    "fiscal_quarter": 1,
    "fiscal_year": 2010,
    "goodwill": null,
    "intangible_assets_net": null,
    "inventories": 1393000000.0,
    "long_term_debt_and_capital_lease_obligations": null,
    "noncontrolling_interest": null,
    "other_assets": 632000000.0,
    "other_current_assets": 647000000.0,
    "other_equity": 0.0,
    "other_noncurrent_liabilities": 996000000.0,
    "period_end": "2009-05-02",
    "preferred_stock": null,
    "property_plant_equipment_net": 2820000000.0,
    "receivables": 0.0,
    "retained_earnings_deficit": 10103000000.0,
    "short_term_investments": null,
    "tickers": [
      "GPS"
    ],
    "timeframe": "quarterly",
    "total_assets": 7221000000.0,
    "total_current_assets": 3748000000.0,
    "total_current_liabilities": 1685000000.0,
    "total_equity": 4540000000.0,
    "total_equity_attributable_to_parent": 4540000000.0,
    "total_liabilities": 2681000000.0,
    "total_liabilities_and_equity": 7221000000.0,
    "treasury_stock": -8627000000.0
  }
]
```

## fundamentals/cash_flow_statement

- 文件数: 17
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/fundamentals/cash_flow_statement/cash_flow_statement_2010.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `cash_from_operating_activities_continuing_operations` | `float64` | Cash generated from continuing business operations before discontinued operations. | 持续经营活动产生的经营现金流。 | fundamentals/cashflow_statement.md |
| `change_in_cash_and_equivalents` | `float64` | Net change in cash and cash equivalents during the period, representing the sum of operating, investing, and financing cash flows plus currency effects. | 现金及现金等价物净变动。 | fundamentals/cashflow_statement.md |
| `change_in_other_operating_assets_and_liabilities_net` | `float64` | Net change in working capital components including accounts receivable, inventory, accounts payable, and other operating items. | 其他经营性资产负债净变动。 | fundamentals/cashflow_statement.md |
| `cik` | `object` | SEC Central Index Key (CIK) identifying the filing entity. | SEC 中央索引编号（CIK）。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>fundamentals/ratios.md<br>tickers/all_tickers.md |
| `depreciation_depletion_and_amortization` | `float64` | Non-cash charges for the reduction in value of tangible and intangible assets over time. | 折旧、折耗与摊销。 | fundamentals/cashflow_statement.md |
| `dividends` | `float64` | Cash payments to shareholders in the form of dividends, typically reported as negative values. | 股利支付现金流。 | fundamentals/cashflow_statement.md |
| `effect_of_currency_exchange_rate` | `float64` | Impact of foreign exchange rate changes on cash and cash equivalents denominated in foreign currencies. | 汇率变动对现金的影响。 | fundamentals/cashflow_statement.md |
| `filing_date` | `object` | Date when the filing was submitted to the SEC (formatted as YYYY-MM-DD). | 申报/提交日期。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `fiscal_quarter` | `int64` | The fiscal quarter number (1, 2, 3, or 4) for the reporting period. | 财季。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `fiscal_year` | `int64` | The fiscal year for the reporting period. | 财年。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `income_loss_from_discontinued_operations` | `float64` | After-tax income or loss from business operations that have been discontinued. | 终止经营损益。 | fundamentals/cashflow_statement.md |
| `long_term_debt_issuances_repayments` | `float64` | Net cash flows from issuing or repaying long-term debt obligations. | 长期债务发行与偿还净额。 | fundamentals/cashflow_statement.md |
| `net_cash_from_financing_activities` | `float64` | Total cash generated or used by financing activities, including debt issuance, debt repayment, dividends, and share transactions. | 筹资活动现金流净额。 | fundamentals/cashflow_statement.md |
| `net_cash_from_financing_activities_continuing_operations` | `float64` | Cash flows from financing activities of continuing operations before discontinued operations. | 持续经营筹资现金流净额。 | fundamentals/cashflow_statement.md |
| `net_cash_from_financing_activities_discontinued_operations` | `float64` | Cash flows from financing activities of discontinued business segments. | 终止经营筹资现金流净额。 | fundamentals/cashflow_statement.md |
| `net_cash_from_investing_activities` | `float64` | Total cash generated or used by investing activities, including capital expenditures, acquisitions, and asset sales. | 投资活动现金流净额。 | fundamentals/cashflow_statement.md |
| `net_cash_from_investing_activities_continuing_operations` | `float64` | Cash flows from investing activities of continuing operations before discontinued operations. | 持续经营投资现金流净额。 | fundamentals/cashflow_statement.md |
| `net_cash_from_investing_activities_discontinued_operations` | `float64` | Cash flows from investing activities of discontinued business segments. | 终止经营投资现金流净额。 | fundamentals/cashflow_statement.md |
| `net_cash_from_operating_activities` | `float64` | Total cash generated or used by operating activities, representing cash flow from core business operations. | 经营活动现金流净额。 | fundamentals/cashflow_statement.md |
| `net_cash_from_operating_activities_discontinued_operations` | `float64` | Cash flows from operating activities of discontinued business segments. | 终止经营经营现金流净额。 | fundamentals/cashflow_statement.md |
| `net_income` | `float64` | Net income used as the starting point for operating cash flow calculations. | 净利润。 | fundamentals/cashflow_statement.md |
| `noncontrolling_interests` | `float64` | Cash flows related to minority shareholders in consolidated subsidiaries. | 少数股东相关现金流项目。 | fundamentals/cashflow_statement.md |
| `other_cash_adjustments` | `float64` | Other miscellaneous adjustments to cash flows not classified elsewhere. | 其他现金调整项。 | fundamentals/cashflow_statement.md |
| `other_financing_activities` | `float64` | Cash flows from financing activities not classified elsewhere, including share repurchases and other equity transactions. | 其他筹资活动现金流。 | fundamentals/cashflow_statement.md |
| `other_investing_activities` | `float64` | Cash flows from investing activities not classified elsewhere, including acquisitions, divestitures, and investments. | 其他投资活动现金流。 | fundamentals/cashflow_statement.md |
| `other_operating_activities` | `float64` | Other adjustments to reconcile net income to operating cash flow not classified elsewhere. | 其他经营活动现金流。 | fundamentals/cashflow_statement.md |
| `period_end` | `object` | The last date of the reporting period, representing the specific point in time when the balance sheet snapshot was taken. | 报告期结束日期。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `purchase_of_property_plant_and_equipment` | `float64` | Cash outflows for capital expenditures on fixed assets, typically reported as negative values. | 购建固定资产支出。 | fundamentals/cashflow_statement.md |
| `sale_of_property_plant_and_equipment` | `float64` | Cash inflows from disposing of fixed assets, typically reported as positive values. | 处置固定资产收入。 | fundamentals/cashflow_statement.md |
| `short_term_debt_issuances_repayments` | `float64` | Net cash flows from issuing or repaying short-term debt obligations. | 短期债务发行与偿还净额。 | fundamentals/cashflow_statement.md |
| `tickers` | `object` | A list of ticker symbols under which the company is listed. Multiple symbols may indicate different share classes for the same company. | 证券代码列表（可能包含多股类别）。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>news/news.md |
| `timeframe` | `object` | The reporting period type. Possible values include: quarterly, annual. | 报告周期类型（季度/年度等）。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>snapshots/unified_snapshot.md |

### 样本记录（前 3 条）

```json
[
  {
    "cash_from_operating_activities_continuing_operations": 2827000000.0,
    "change_in_cash_and_equivalents": 1821000000.0,
    "change_in_other_operating_assets_and_liabilities_net": -2749000000.0,
    "cik": "0000200406",
    "depreciation_depletion_and_amortization": 676000000.0,
    "dividends": -1273000000.0,
    "effect_of_currency_exchange_rate": -62000000.0,
    "filing_date": "2010-05-10",
    "fiscal_quarter": 1,
    "fiscal_year": 2010,
    "income_loss_from_discontinued_operations": null,
    "long_term_debt_issuances_repayments": -7000000.0,
    "net_cash_from_financing_activities": 132000000.0,
    "net_cash_from_financing_activities_continuing_operations": 132000000.0,
    "net_cash_from_financing_activities_discontinued_operations": null,
    "net_cash_from_investing_activities": -1076000000.0,
    "net_cash_from_investing_activities_continuing_operations": -1076000000.0,
    "net_cash_from_investing_activities_discontinued_operations": null,
    "net_cash_from_operating_activities": 2827000000.0,
    "net_cash_from_operating_activities_discontinued_operations": null,
    "net_income": 3507000000.0,
    "noncontrolling_interests": null,
    "other_cash_adjustments": null,
    "other_financing_activities": -807000000.0,
    "other_investing_activities": -647000000.0,
    "other_operating_activities": 1393000000.0,
    "period_end": "2009-03-29",
    "purchase_of_property_plant_and_equipment": -435000000.0,
    "sale_of_property_plant_and_equipment": 6000000.0,
    "short_term_debt_issuances_repayments": 2219000000.0,
    "tickers": [
      "JNJ"
    ],
    "timeframe": "quarterly"
  },
  {
    "cash_from_operating_activities_continuing_operations": 142128000.0,
    "change_in_cash_and_equivalents": 94586000.0,
    "change_in_other_operating_assets_and_liabilities_net": 144547000.0,
    "cik": "0001045810",
    "depreciation_depletion_and_amortization": 50658000.0,
    "dividends": 0.0,
    "effect_of_currency_exchange_rate": null,
    "filing_date": "2010-05-21",
    "fiscal_quarter": 1,
    "fiscal_year": 2010,
    "income_loss_from_discontinued_operations": null,
    "long_term_debt_issuances_repayments": -222000.0,
    "net_cash_from_financing_activities": -38637000.0,
    "net_cash_from_financing_activities_continuing_operations": -38637000.0,
    "net_cash_from_financing_activities_discontinued_operations": null,
    "net_cash_from_investing_activities": -8905000.0,
    "net_cash_from_investing_activities_continuing_operations": -8905000.0,
    "net_cash_from_investing_activities_discontinued_operations": null,
    "net_cash_from_operating_activities": 142128000.0,
    "net_cash_from_operating_activities_discontinued_operations": null,
    "net_income": -201338000.0,
    "noncontrolling_interests": null,
    "other_cash_adjustments": null,
    "other_financing_activities": -38415000.0,
    "other_investing_activities": 11872000.0,
    "other_operating_activities": 148261000.0,
    "period_end": "2009-04-26",
    "purchase_of_property_plant_and_equipment": -20777000.0,
    "sale_of_property_plant_and_equipment": null,
    "short_term_debt_issuances_repayments": null,
    "tickers": [
      "NVDA"
    ],
    "timeframe": "quarterly"
  },
  {
    "cash_from_operating_activities_continuing_operations": 3571000000.0,
    "change_in_cash_and_equivalents": -697000000.0,
    "change_in_other_operating_assets_and_liabilities_net": -1084000000.0,
    "cik": "0000104169",
    "depreciation_depletion_and_amortization": 1700000000.0,
    "dividends": -1067000000.0,
    "effect_of_currency_exchange_rate": -82000000.0,
    "filing_date": "2010-06-04",
    "fiscal_quarter": 1,
    "fiscal_year": 2010,
    "income_loss_from_discontinued_operations": 8000000.0,
    "long_term_debt_issuances_repayments": 1390000000.0,
    "net_cash_from_financing_activities": -1503000000.0,
    "net_cash_from_financing_activities_continuing_operations": -1503000000.0,
    "net_cash_from_financing_activities_discontinued_operations": null,
    "net_cash_from_investing_activities": -2683000000.0,
    "net_cash_from_investing_activities_continuing_operations": -2683000000.0,
    "net_cash_from_investing_activities_discontinued_operations": null,
    "net_cash_from_operating_activities": 3571000000.0,
    "net_cash_from_operating_activities_discontinued_operations": null,
    "net_income": 3139000000.0,
    "noncontrolling_interests": -436000000.0,
    "other_cash_adjustments": null,
    "other_financing_activities": -1124000000.0,
    "other_investing_activities": -208000000.0,
    "other_operating_activities": -192000000.0,
    "period_end": "2009-04-30",
    "purchase_of_property_plant_and_equipment": -2607000000.0,
    "sale_of_property_plant_and_equipment": 132000000.0,
    "short_term_debt_issuances_repayments": -266000000.0,
    "tickers": [
      "WMT"
    ],
    "timeframe": "quarterly"
  }
]
```

## fundamentals/financials_ratios

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/fundamentals/financials_ratios/financials_ratios_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `average_volume` | `float64` | Average trading volume over the last 30 trading days, providing context for liquidity. | 近 30 个交易日平均成交量。 | fundamentals/ratios.md |
| `cash` | `float64` | Cash ratio, calculated as cash and cash equivalents divided by current liabilities, measuring the most liquid form of liquidity coverage. | 现金比率。 | fundamentals/ratios.md |
| `cik` | `object` | SEC Central Index Key (CIK) identifying the filing entity. | SEC 中央索引编号（CIK）。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>fundamentals/ratios.md<br>tickers/all_tickers.md |
| `current` | `float64` | Current ratio, calculated as total current assets divided by total current liabilities, measuring short-term liquidity. | 流动比率。 | fundamentals/ratios.md |
| `date` | `object` | Date for which the ratios are calculated, representing the trading date with available price data. | 日期。 | fundamentals/ratios.md<br>fundamentals/short_volume.md |
| `debt_to_equity` | `float64` | Debt-to-equity ratio, calculated as total debt (current debt plus long-term debt) divided by total shareholders' equity, measuring financial leverage. | 资产负债率（债务/权益）。 | fundamentals/ratios.md |
| `dividend_yield` | `float64` | Dividend yield, calculated as annual dividends per share divided by stock price, measuring the income return on investment. | 股息率。 | fundamentals/ratios.md |
| `earnings_per_share` | `float64` | Earnings per share, calculated as net income available to common shareholders divided by weighted shares outstanding. | 每股收益（EPS）。 | fundamentals/ratios.md |
| `enterprise_value` | `float64` | Enterprise value, calculated as market capitalization plus total debt minus cash and cash equivalents, representing total company value. | 企业价值（EV）。 | fundamentals/ratios.md |
| `ev_to_ebitda` | `float64` | Enterprise value to EBITDA ratio, calculated as enterprise value divided by EBITDA, measuring company valuation relative to earnings before interest, taxes, depreciation, and amortization. | EBITDA。 | fundamentals/ratios.md |
| `ev_to_sales` | `float64` | Enterprise value to sales ratio, calculated as enterprise value divided by revenue, measuring company valuation relative to sales. | EV/销售额。 | fundamentals/ratios.md |
| `free_cash_flow` | `float64` | Free cash flow, calculated as operating cash flow minus capital expenditures (purchase of property, plant, and equipment). | 自由现金流。 | fundamentals/ratios.md |
| `market_cap` | `float64` | Market capitalization, calculated as stock price multiplied by total shares outstanding. | 市值。 | fundamentals/ratios.md |
| `price` | `float64` | Stock price used in ratio calculations, typically the closing price for the given date. | 价格。 | fundamentals/ratios.md |
| `price_to_book` | `float64` | Price-to-book ratio, calculated as stock price divided by book value per share, comparing market value to book value. | 市净率（P/B）。 | fundamentals/ratios.md |
| `price_to_cash_flow` | `float64` | Price-to-cash-flow ratio, calculated as stock price divided by operating cash flow per share. Only calculated when operating cash flow per share is positive. | 市现率（P/CF）。 | fundamentals/ratios.md |
| `price_to_earnings` | `float64` | Price-to-earnings ratio, calculated as stock price divided by earnings per share. Only calculated when earnings per share is positive. | 市盈率（P/E）。 | fundamentals/ratios.md |
| `price_to_free_cash_flow` | `float64` | Price-to-free-cash-flow ratio, calculated as stock price divided by free cash flow per share. Only calculated when free cash flow per share is positive. | 市自由现金流率（P/FCF）。 | fundamentals/ratios.md |
| `price_to_sales` | `float64` | Price-to-sales ratio, calculated as stock price divided by revenue per share, measuring valuation relative to sales. | 市销率（P/S）。 | fundamentals/ratios.md |
| `quick` | `float64` | Quick ratio (acid-test ratio), calculated as (current assets minus inventories) divided by current liabilities, measuring immediate liquidity. | 速动比率。 | fundamentals/ratios.md |
| `return_on_assets` | `float64` | Return on assets ratio, calculated as net income divided by total assets, measuring how efficiently a company uses its assets to generate profit. | 总资产收益率（ROA）。 | fundamentals/ratios.md |
| `return_on_equity` | `float64` | Return on equity ratio, calculated as net income divided by total shareholders' equity, measuring profitability relative to shareholders' equity. | 净资产收益率（ROE）。 | fundamentals/ratios.md |
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |

### 样本记录（前 3 条）

```json
[
  {
    "average_volume": 2471277.0,
    "cash": 0.79,
    "cik": "0001090872",
    "current": 2.07,
    "date": "2026-03-09",
    "debt_to_equity": 0.49,
    "dividend_yield": 0.0086,
    "earnings_per_share": 4.56,
    "enterprise_value": 34558734255.0,
    "ev_to_ebitda": 20.74,
    "ev_to_sales": 4.89,
    "free_cash_flow": 993000000.0,
    "market_cap": 32962734255.0,
    "price": 116.64,
    "price_to_book": 4.77,
    "price_to_cash_flow": 23.61,
    "price_to_earnings": 25.55,
    "price_to_free_cash_flow": 33.2,
    "price_to_sales": 4.67,
    "quick": 1.59,
    "return_on_assets": 0.1007,
    "return_on_equity": 0.1867,
    "ticker": "A"
  },
  {
    "average_volume": 7525461.0,
    "cash": 0.42,
    "cik": "0001675149",
    "current": 1.44,
    "date": "2026-03-09",
    "debt_to_equity": 0.4,
    "dividend_yield": 0.0065,
    "earnings_per_share": 4.39,
    "enterprise_value": 16978438621.0,
    "ev_to_ebitda": 9.18,
    "ev_to_sales": 1.32,
    "free_cash_flow": 567000000.0,
    "market_cap": 16136438621.0,
    "price": 61.16,
    "price_to_book": 2.64,
    "price_to_cash_flow": 13.62,
    "price_to_earnings": 13.95,
    "price_to_free_cash_flow": 28.46,
    "price_to_sales": 1.26,
    "quick": 0.87,
    "return_on_assets": 0.0717,
    "return_on_equity": 0.1891,
    "ticker": "AA"
  },
  {
    "average_volume": 712.0,
    "cash": 0.0,
    "cik": "0001848898",
    "current": 0.11,
    "date": "2026-03-09",
    "debt_to_equity": -0.13,
    "dividend_yield": 0.0,
    "earnings_per_share": -0.95,
    "enterprise_value": 3648512.0,
    "ev_to_ebitda": -0.07,
    "ev_to_sales": 1.69,
    "free_cash_flow": -5999846.0,
    "market_cap": 578668.0,
    "price": 0.01,
    "price_to_book": -0.02,
    "price_to_cash_flow": null,
    "price_to_earnings": null,
    "price_to_free_cash_flow": null,
    "price_to_sales": 0.27,
    "quick": 0.1,
    "return_on_assets": -3.1491,
    "return_on_equity": 2.2506,
    "ticker": "AAGR"
  }
]
```

## fundamentals/income_statement

- 文件数: 17
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/fundamentals/income_statement/income_statement_2010.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `basic_earnings_per_share` | `float64` | Earnings per share calculated using the weighted average number of basic shares outstanding. For TTM records, recalculated as TTM net income divided by average basic shares outstanding over the four quarters. | 基本每股收益。 | fundamentals/income_statement.md |
| `basic_shares_outstanding` | `float64` | Weighted average number of common shares outstanding during the period, used in basic EPS calculation. For TTM records, represents the average over the four most recent quarters. | 加权平均基本在外股数。 | fundamentals/income_statement.md |
| `cik` | `object` | SEC Central Index Key (CIK) identifying the filing entity. | SEC 中央索引编号（CIK）。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>fundamentals/ratios.md<br>tickers/all_tickers.md |
| `consolidated_net_income_loss` | `float64` | Total net income or loss for the consolidated entity including all subsidiaries. | 合并净利润/净亏损。 | fundamentals/income_statement.md |
| `cost_of_revenue` | `float64` | Direct costs attributable to the production of goods or services sold, also known as cost of goods sold (COGS). | 营业成本。 | fundamentals/income_statement.md |
| `depreciation_depletion_amortization` | `float64` | Non-cash expenses representing the allocation of asset costs over their useful lives. | 折旧、折耗与摊销。 | fundamentals/income_statement.md |
| `diluted_earnings_per_share` | `float64` | Earnings per share calculated using diluted shares outstanding, including the effect of potentially dilutive securities. For TTM records, recalculated as TTM net income divided by average diluted shares outstanding over the four quarters. | 稀释每股收益。 | fundamentals/income_statement.md |
| `diluted_shares_outstanding` | `float64` | Weighted average number of shares outstanding including the dilutive effect of stock options, warrants, and convertible securities. For TTM records, represents the average over the four most recent quarters. | 加权平均稀释在外股数。 | fundamentals/income_statement.md |
| `discontinued_operations` | `float64` | After-tax results from business segments that have been or will be disposed of. | 终止经营项目。 | fundamentals/income_statement.md |
| `ebitda` | `float64` | Earnings before interest, taxes, depreciation, and amortization, a measure of operating performance. | 税息折旧及摊销前利润（EBITDA）。 | fundamentals/income_statement.md |
| `equity_in_affiliates` | `float64` | The company's share of income or losses from equity method investments in affiliated companies. | 联营/合营企业权益法损益。 | fundamentals/income_statement.md |
| `extraordinary_items` | `float64` | Unusual and infrequent gains or losses that are both unusual in nature and infrequent in occurrence. | 非常项目。 | fundamentals/income_statement.md |
| `filing_date` | `object` | Date when the filing was submitted to the SEC (formatted as YYYY-MM-DD). | 申报/提交日期。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `fiscal_quarter` | `int64` | The fiscal quarter number (1, 2, 3, or 4) for the reporting period. | 财季。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `fiscal_year` | `int64` | The fiscal year for the reporting period. | 财年。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `gross_profit` | `float64` | Revenue minus cost of revenue, representing profit before operating expenses. | 毛利润。 | fundamentals/income_statement.md |
| `income_before_income_taxes` | `float64` | Pre-tax income calculated as operating income plus total other income/expense. | 税前利润。 | fundamentals/income_statement.md |
| `income_taxes` | `float64` | Income tax expense or benefit for the period. | 所得税费用。 | fundamentals/income_statement.md |
| `interest_expense` | `float64` | Cost of borrowed funds, including interest on debt and other financing obligations. | 利息费用。 | fundamentals/income_statement.md |
| `interest_income` | `float64` | Income earned from interest-bearing investments and cash equivalents. | 利息收入。 | fundamentals/income_statement.md |
| `net_income_loss_attributable_common_shareholders` | `float64` | Net income or loss available to common shareholders after preferred dividends and noncontrolling interests. | 归属于普通股股东净利润/净亏损。 | fundamentals/income_statement.md |
| `noncontrolling_interest` | `float64` | Equity in consolidated subsidiaries not owned by the parent company, representing minority shareholders' ownership. | 少数股东权益（非控股权益）。 | fundamentals/balance_sheets.md<br>fundamentals/income_statement.md |
| `operating_income` | `float64` | Income from operations calculated as gross profit minus total operating expenses, excluding non-operating items. | 营业利润。 | fundamentals/income_statement.md |
| `other_income_expense` | `float64` | Non-operating income and expenses not related to the company's core business operations. | 其他收益/费用。 | fundamentals/income_statement.md |
| `other_operating_expenses` | `float64` | Operating expenses not classified in the main expense categories. | 其他经营。 | fundamentals/income_statement.md |
| `period_end` | `object` | The last date of the reporting period, representing the specific point in time when the balance sheet snapshot was taken. | 报告期结束日期。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md |
| `preferred_stock_dividends_declared` | `float64` | Dividends declared on preferred stock during the period. | 优先股。 | fundamentals/income_statement.md |
| `research_development` | `float64` | Expenses incurred for research and development activities to create new products or improve existing ones. | 该字段中文释义已补充。 | fundamentals/income_statement.md |
| `revenue` | `float64` | Total revenue or net sales for the period, representing the company's gross income from operations. | 营业收入。 | fundamentals/income_statement.md |
| `selling_general_administrative` | `float64` | Expenses related to selling products and general administrative costs not directly tied to production. | 该字段中文释义已补充。 | fundamentals/income_statement.md |
| `tickers` | `object` | A list of ticker symbols under which the company is listed. Multiple symbols may indicate different share classes for the same company. | 证券代码列表（可能包含多股类别）。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>news/news.md |
| `timeframe` | `object` | The reporting period type. Possible values include: quarterly, annual. | 报告周期类型（季度/年度等）。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>snapshots/unified_snapshot.md |
| `total_operating_expenses` | `float64` | Sum of all operating expenses including cost of revenue, SG&A, R&D, depreciation, and other operating expenses. | 总经营。 | fundamentals/income_statement.md |
| `total_other_income_expense` | `float64` | Net total of all non-operating income and expenses including interest income, interest expense, and other items. | 总其他收益。 | fundamentals/income_statement.md |

### 样本记录（前 3 条）

```json
[
  {
    "basic_earnings_per_share": 1.27,
    "basic_shares_outstanding": 2765900000.0,
    "cik": "0000200406",
    "consolidated_net_income_loss": 3507000000.0,
    "cost_of_revenue": 4251000000.0,
    "depreciation_depletion_amortization": null,
    "diluted_earnings_per_share": 1.26,
    "diluted_shares_outstanding": 2789800000.0,
    "discontinued_operations": null,
    "ebitda": 5325000000.0,
    "equity_in_affiliates": null,
    "extraordinary_items": null,
    "filing_date": "2010-05-10",
    "fiscal_quarter": 1,
    "fiscal_year": 2010,
    "gross_profit": 10775000000.0,
    "income_before_income_taxes": 4643000000.0,
    "income_taxes": 1136000000.0,
    "interest_expense": -106000000.0,
    "interest_income": 25000000.0,
    "net_income_loss_attributable_common_shareholders": 3507000000.0,
    "noncontrolling_interest": null,
    "operating_income": 4649000000.0,
    "other_income_expense": 75000000.0,
    "other_operating_expenses": 0.0,
    "period_end": "2009-03-29",
    "preferred_stock_dividends_declared": null,
    "research_development": 1518000000.0,
    "revenue": 15026000000.0,
    "selling_general_administrative": 4608000000.0,
    "tickers": [
      "JNJ"
    ],
    "timeframe": "quarterly",
    "total_operating_expenses": 6126000000.0,
    "total_other_income_expense": -6000000.0
  },
  {
    "basic_earnings_per_share": -0.01,
    "basic_shares_outstanding": 21692280000.0,
    "cik": "0001045810",
    "consolidated_net_income_loss": -201338000.0,
    "cost_of_revenue": 474535000.0,
    "depreciation_depletion_amortization": null,
    "diluted_earnings_per_share": -0.01,
    "diluted_shares_outstanding": 21692280000.0,
    "discontinued_operations": null,
    "ebitda": -180307000.0,
    "equity_in_affiliates": null,
    "extraordinary_items": null,
    "filing_date": "2010-05-21",
    "fiscal_quarter": 1,
    "fiscal_year": 2010,
    "gross_profit": 189696000.0,
    "income_before_income_taxes": -224821000.0,
    "income_taxes": -23483000.0,
    "interest_expense": null,
    "interest_income": 6124000.0,
    "net_income_loss_attributable_common_shareholders": -201338000.0,
    "noncontrolling_interest": null,
    "operating_income": -230965000.0,
    "other_income_expense": 20000.0,
    "other_operating_expenses": 0.0,
    "period_end": "2009-04-26",
    "preferred_stock_dividends_declared": null,
    "research_development": 301797000.0,
    "revenue": 664231000.0,
    "selling_general_administrative": 118864000.0,
    "tickers": [
      "NVDA"
    ],
    "timeframe": "quarterly",
    "total_operating_expenses": 420661000.0,
    "total_other_income_expense": 6144000.0
  },
  {
    "basic_earnings_per_share": 0.26,
    "basic_shares_outstanding": 11760000000.0,
    "cik": "0000104169",
    "consolidated_net_income_loss": 3139000000.0,
    "cost_of_revenue": 70388000000.0,
    "depreciation_depletion_amortization": null,
    "diluted_earnings_per_share": 0.26,
    "diluted_shares_outstanding": 11790000000.0,
    "discontinued_operations": -8000000.0,
    "ebitda": 6917000000.0,
    "equity_in_affiliates": null,
    "extraordinary_items": null,
    "filing_date": "2010-06-04",
    "fiscal_quarter": 1,
    "fiscal_year": 2010,
    "gross_profit": 23854000000.0,
    "income_before_income_taxes": 4750000000.0,
    "income_taxes": 1603000000.0,
    "interest_expense": -518000000.0,
    "interest_income": 51000000.0,
    "net_income_loss_attributable_common_shareholders": 3022000000.0,
    "noncontrolling_interest": -117000000.0,
    "operating_income": 5217000000.0,
    "other_income_expense": 0.0,
    "other_operating_expenses": 0.0,
    "period_end": "2009-04-30",
    "preferred_stock_dividends_declared": null,
    "research_development": null,
    "revenue": 94242000000.0,
    "selling_general_administrative": 18637000000.0,
    "tickers": [
      "WMT"
    ],
    "timeframe": "quarterly",
    "total_operating_expenses": 18637000000.0,
    "total_other_income_expense": -467000000.0
  }
]
```

## fundamentals/short_interest

- 文件数: 10
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/fundamentals/short_interest/short_interest_2017.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `avg_daily_volume` | `int64` | The average daily trading volume for the stock over a specified period, typically used to contextualize short interest. | 平均日成交量。 | fundamentals/short_interest.md |
| `days_to_cover` | `float64` | Calculated as short_interest divided by avg_daily_volume, representing the estimated number of days it would take to cover all short positions based on average trading volume. | 回补天数（空头覆盖天数）。 | fundamentals/short_interest.md |
| `settlement_date` | `object` | The date (formatted as YYYY-MM-DD) on which the short interest data is considered settled, typically based on exchange reporting schedules. | 结算日期。 | fundamentals/short_interest.md |
| `short_interest` | `int64` | The total number of shares that have been sold short but have not yet been covered or closed out. | 空头持仓量。 | fundamentals/short_interest.md |
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |

### 样本记录（前 3 条）

```json
[
  {
    "avg_daily_volume": 1234014,
    "days_to_cover": 3.4,
    "settlement_date": "2017-12-29",
    "short_interest": 4197300,
    "ticker": "A"
  },
  {
    "avg_daily_volume": 4267200,
    "days_to_cover": 2.97,
    "settlement_date": "2017-12-29",
    "short_interest": 12689077,
    "ticker": "AA"
  },
  {
    "avg_daily_volume": 0,
    "days_to_cover": 999.99,
    "settlement_date": "2017-12-29",
    "short_interest": 13823,
    "ticker": "AAALF"
  }
]
```

## fundamentals/short_volume

- 文件数: 3
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/fundamentals/short_volume/short_volume_2024.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `adf_short_volume` | `int64` | Short volume reported via the Alternative Display Facility (ADF), excluding exempt volume. | ADF 渠道卖空成交量。 | fundamentals/short_volume.md |
| `adf_short_volume_exempt` | `int64` | Short volume reported via ADF that was marked as exempt. | 短期成交量。 | fundamentals/short_volume.md |
| `date` | `object` | Date for which the ratios are calculated, representing the trading date with available price data. | 日期。 | fundamentals/ratios.md<br>fundamentals/short_volume.md |
| `exempt_volume` | `float64` | Portion of short volume that was marked as exempt from regulation SHO. | 豁免卖空成交量。 | fundamentals/short_volume.md |
| `nasdaq_carteret_short_volume` | `int64` | Short volume reported from Nasdaq's Carteret facility, excluding exempt volume. | 短期成交量。 | fundamentals/short_volume.md |
| `nasdaq_carteret_short_volume_exempt` | `int64` | Short volume from Nasdaq Carteret that was marked as exempt. | 短期成交量。 | fundamentals/short_volume.md |
| `nasdaq_chicago_short_volume` | `int64` | Short volume reported from Nasdaq's Chicago facility, excluding exempt volume. | 短期成交量。 | fundamentals/short_volume.md |
| `nasdaq_chicago_short_volume_exempt` | `int64` | Short volume from Nasdaq Chicago that was marked as exempt. | 短期成交量。 | fundamentals/short_volume.md |
| `non_exempt_volume` | `float64` | Portion of short volume that was not exempt from regulation SHO (i.e., short_volume - exempt_volume). | 非豁免卖空成交量。 | fundamentals/short_volume.md |
| `nyse_short_volume` | `int64` | Short volume reported from NYSE facilities, excluding exempt volume. | NYSE 渠道卖空成交量。 | fundamentals/short_volume.md |
| `nyse_short_volume_exempt` | `int64` | Short volume from NYSE facilities that was marked as exempt. | 短期成交量。 | fundamentals/short_volume.md |
| `short_volume` | `float64` | Total number of shares sold short across all venues for the ticker on the given date. | 卖空成交量。 | fundamentals/short_volume.md |
| `short_volume_ratio` | `float64` | The percentage of total volume that was sold short. Calculated as (short_volume / total_volume) * 100. | 卖空成交占比。 | fundamentals/short_volume.md |
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `total_volume` | `float64` | Total reported volume across all venues for the ticker on the given date. | 总成交量。 | fundamentals/short_volume.md |

### 样本记录（前 3 条）

```json
[
  {
    "adf_short_volume": 0,
    "adf_short_volume_exempt": 0,
    "date": "2024-02-06",
    "exempt_volume": 866.0,
    "nasdaq_carteret_short_volume": 473707,
    "nasdaq_carteret_short_volume_exempt": 866,
    "nasdaq_chicago_short_volume": 130,
    "nasdaq_chicago_short_volume_exempt": 0,
    "non_exempt_volume": 475798.0,
    "nyse_short_volume": 2827,
    "nyse_short_volume_exempt": 0,
    "short_volume": 476664.0,
    "short_volume_ratio": 70.1,
    "ticker": "A",
    "total_volume": 679938.0
  },
  {
    "adf_short_volume": 0,
    "adf_short_volume_exempt": 0,
    "date": "2024-02-06",
    "exempt_volume": 3758.0,
    "nasdaq_carteret_short_volume": 1070799,
    "nasdaq_carteret_short_volume_exempt": 3758,
    "nasdaq_chicago_short_volume": 1861,
    "nasdaq_chicago_short_volume_exempt": 0,
    "non_exempt_volume": 1125769.0,
    "nyse_short_volume": 56867,
    "nyse_short_volume_exempt": 0,
    "short_volume": 1129527.0,
    "short_volume_ratio": 30.34,
    "ticker": "AA",
    "total_volume": 3723314.0
  },
  {
    "adf_short_volume": 0,
    "adf_short_volume_exempt": 0,
    "date": "2024-02-06",
    "exempt_volume": 0.0,
    "nasdaq_carteret_short_volume": 1049,
    "nasdaq_carteret_short_volume_exempt": 0,
    "nasdaq_chicago_short_volume": 0,
    "nasdaq_chicago_short_volume_exempt": 0,
    "non_exempt_volume": 1049.0,
    "nyse_short_volume": 0,
    "nyse_short_volume_exempt": 0,
    "short_volume": 1049.0,
    "short_volume_ratio": 57.61,
    "ticker": "AAA",
    "total_volume": 1821.0
  }
]
```

## fundamentals/stocks_floats

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/fundamentals/stocks_floats/stocks_floats_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `free_float` | `int64` | Number of shares freely tradable in the market. Free float shares represent the portion of a company's outstanding shares that is freely tradable in the market, excluding any holdings considered strategic, controlling, or long term. This excludes insiders, directors, founders, 5 percent plus shareholders, cross holdings, government stakes except pensions, restricted or locked up shares, employee plans, and any entities with board influence, leaving only shares that are genuinely available for public trading. | 自由流通股本。 | fundamentals/float.md |
| `effective_date` | `object` | The effective date of the free float measurement. | 生效日期。 | fundamentals/float.md |
| `free_float_percent` | `float64` | Percentage of total shares outstanding that are available for public trading, rounded to two decimal places. | 自由流通股本占比。 | fundamentals/float.md |

### 样本记录（前 3 条）

```json
[
  {
    "ticker": "A",
    "free_float": 282591673,
    "effective_date": "2026-01-08",
    "free_float_percent": 99.7
  },
  {
    "ticker": "AA",
    "free_float": 258394311,
    "effective_date": "2026-01-29",
    "free_float_percent": 99.8
  },
  {
    "ticker": "AABVF",
    "free_float": 133442193,
    "effective_date": "2026-01-02",
    "free_float_percent": 83.04
  }
]
```

## market_operations/condition_codes

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/market_operations/condition_codes/condition_codes_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `id` | `int64` | Unique identifier for each dividend record | 记录唯一标识。 | corporate_actions/dividends.md<br>corporate_actions/splits.md<br>market_operations/condition_codes.md<br>market_operations/exchanges.md<br>news/news.md |
| `type` | `object` | An identifier for a collection of related conditions. | 类型。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md |
| `name` | `object` | The name of this condition. | 名称。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md |
| `asset_class` | `object` | An identifier for a group of similar financial instruments. | 资产类别。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>tickers/ticker_types.md |
| `data_types` | `object` | Data types that this condition applies to. | 数据类型列表。 | market_operations/condition_codes.md |
| `description` | `object` | Detailed explanation of what this risk category encompasses, including specific examples and potential impacts | 描述。 | filing/risk_categories.md<br>market_operations/condition_codes.md<br>news/news.md<br>tickers/ticker_types.md |
| `abbreviation` | `object` | A commonly-used abbreviation for this condition. | 缩写。 | market_operations/condition_codes.md |
| `sip_mapping` | `object` | A comprehensive mapping that translates condition codes from individual SIPs (CTA, OPRA, UTP) to a unified code used by Massive. This facilitates consistent interpretation and application of market data conditions across different data streams, ensuring that users can accurately apply these conditions to their data analysis and reporting. | SIP 条件码映射关系。 | market_operations/condition_codes.md |
| `update_rules` | `object` | A list of aggregation rules. | 更新规则。 | market_operations/condition_codes.md |
| `legacy` | `object` | If true, this condition is from an old version of the SIPs' specs and no longer is used. Other conditions may or may not reuse the same symbol as this one. | 是否为旧版遗留条件。 | market_operations/condition_codes.md |
| `exchange` | `float64` | If present, mapping this condition from a Massive code to a SIP symbol depends on this attribute. In other words, data with this condition attached comes exclusively from the given exchange. | 交易所代码。 | market_operations/condition_codes.md |

### 样本记录（前 3 条）

```json
[
  {
    "id": 0,
    "type": "regular",
    "name": "Regular Trade",
    "asset_class": "crypto",
    "data_types": [
      "trade"
    ],
    "description": null,
    "abbreviation": null,
    "sip_mapping": null,
    "update_rules": null,
    "legacy": null,
    "exchange": null
  },
  {
    "id": 1,
    "type": "buy_or_sell_side",
    "name": "Sell Side",
    "asset_class": "crypto",
    "data_types": [
      "trade"
    ],
    "description": "The asset was sold at the prevailing best bid price on an exchange.",
    "abbreviation": null,
    "sip_mapping": null,
    "update_rules": null,
    "legacy": null,
    "exchange": null
  },
  {
    "id": 2,
    "type": "buy_or_sell_side",
    "name": "Buy Side",
    "asset_class": "crypto",
    "data_types": [
      "trade"
    ],
    "description": "The asset was bought at the prevailing best ask price on an exchange.",
    "abbreviation": null,
    "sip_mapping": null,
    "update_rules": null,
    "legacy": null,
    "exchange": null
  }
]
```

## market_operations/exchanges

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/market_operations/exchanges/exchanges_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `id` | `int64` | Unique identifier for each dividend record | 记录唯一标识。 | corporate_actions/dividends.md<br>corporate_actions/splits.md<br>market_operations/condition_codes.md<br>market_operations/exchanges.md<br>news/news.md |
| `type` | `object` | An identifier for a collection of related conditions. | 类型。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md |
| `asset_class` | `object` | An identifier for a group of similar financial instruments. | 资产类别。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>tickers/ticker_types.md |
| `locale` | `object` | An identifier for a geographical location. | 市场区域。 | market_operations/exchanges.md<br>tickers/all_tickers.md<br>tickers/ticker_types.md |
| `name` | `object` | The name of this condition. | 名称。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md |
| `acronym` | `object` | A commonly used abbreviation for this exchange. | 简称。 | market_operations/exchanges.md |
| `mic` | `object` | The Market Identifier Code of this exchange (see ISO 10383). | 交易所 MIC 代码。 | market_operations/exchanges.md |
| `operating_mic` | `object` | The MIC of the entity that operates this exchange. | 运营方 MIC 代码。 | market_operations/exchanges.md |
| `participant_id` | `object` | The ID used by SIP's to represent this exchange. | 参与方标识。 | market_operations/exchanges.md |
| `url` | `object` | A link to this exchange's website, if one exists. | 链接地址。 | market_operations/exchanges.md |

### 样本记录（前 3 条）

```json
[
  {
    "id": 1,
    "type": "exchange",
    "asset_class": "stocks",
    "locale": "us",
    "name": "NYSE American, LLC",
    "acronym": "AMEX",
    "mic": "XASE",
    "operating_mic": "XNYS",
    "participant_id": "A",
    "url": "https://www.nyse.com/markets/nyse-american"
  },
  {
    "id": 2,
    "type": "exchange",
    "asset_class": "stocks",
    "locale": "us",
    "name": "Nasdaq OMX BX, Inc.",
    "acronym": null,
    "mic": "XBOS",
    "operating_mic": "XNAS",
    "participant_id": "B",
    "url": "https://www.nasdaq.com/solutions/nasdaq-bx-stock-market"
  },
  {
    "id": 3,
    "type": "exchange",
    "asset_class": "stocks",
    "locale": "us",
    "name": "NYSE National, Inc.",
    "acronym": "NSX",
    "mic": "XCIS",
    "operating_mic": "XNYS",
    "participant_id": "C",
    "url": "https://www.nyse.com/markets/nyse-national"
  }
]
```

## market_operations/market_holidays

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/market_operations/market_holidays/market_holidays_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `date` | `object` | Date for which the ratios are calculated, representing the trading date with available price data. | 日期。 | fundamentals/ratios.md<br>fundamentals/short_volume.md |
| `exchange` | `object` | If present, mapping this condition from a Massive code to a SIP symbol depends on this attribute. In other words, data with this condition attached comes exclusively from the given exchange. | 交易所代码。 | market_operations/condition_codes.md |
| `name` | `object` | The name of this condition. | 名称。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md |
| `status` | `object` | 交易日历接口状态字段。 | 状态字段。 | market_operations/market_holidays.md |
| `close` | `object` | 收盘价。 | 收盘价。 | aggregate_bars/custom_bars.md |
| `open` | `object` | 开盘价。 | 开盘价。 | aggregate_bars/custom_bars.md |

### 样本记录（前 3 条）

```json
[
  {
    "date": "2026-04-03",
    "exchange": "NYSE",
    "name": "Good Friday",
    "status": "closed",
    "close": null,
    "open": null
  },
  {
    "date": "2026-04-03",
    "exchange": "NASDAQ",
    "name": "Good Friday",
    "status": "closed",
    "close": null,
    "open": null
  },
  {
    "date": "2026-05-25",
    "exchange": "NASDAQ",
    "name": "Memorial Day",
    "status": "closed",
    "close": null,
    "open": null
  }
]
```

## news/news

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/news/news/news_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `id` | `object` | Unique identifier for each dividend record | 记录唯一标识。 | corporate_actions/dividends.md<br>corporate_actions/splits.md<br>market_operations/condition_codes.md<br>market_operations/exchanges.md<br>news/news.md |
| `publisher` | `object` | Details the source of the news article, including the publisher's name, logo, and homepage URLs. This information helps users identify and access the original source of news content. | 发布方信息对象。 | news/news.md |
| `title` | `object` | The title of the news article. | 标题。 | news/news.md |
| `author` | `object` | The article's author. | 作者。 | news/news.md |
| `published_utc` | `object` | The UTC date and time when the article was published, formatted in RFC3339 standard (e.g. YYYY-MM-DDTHH:MM:SSZ). | 发布时间（UTC）。 | news/news.md |
| `article_url` | `object` | A link to the news article. | 新闻原文链接。 | news/news.md |
| `tickers` | `object` | A list of ticker symbols under which the company is listed. Multiple symbols may indicate different share classes for the same company. | 证券代码列表（可能包含多股类别）。 | fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>news/news.md |
| `image_url` | `object` | The article's image URL. | 图片链接。 | news/news.md |
| `description` | `object` | Detailed explanation of what this risk category encompasses, including specific examples and potential impacts | 描述。 | filing/risk_categories.md<br>market_operations/condition_codes.md<br>news/news.md<br>tickers/ticker_types.md |
| `keywords` | `object` | The keywords associated with the article (which will vary depending on the publishing source). | 关键词。 | news/news.md |
| `insights` | `object` | The insights related to the article. | 洞察/情绪分析信息。 | news/news.md |
| `amp_url` | `object` | The mobile friendly Accelerated Mobile Page (AMP) URL. | AMP 移动链接。 | news/news.md |

### 样本记录（前 3 条）

```json
[
  {
    "id": "b23004da0fc7e0335ae892129d256c0b7a769c8f606c98b2c1a03b9b53c73224",
    "publisher": {
      "favicon_url": "https://s3.massive.com/public/assets/news/favicons/globenewswire.ico",
      "homepage_url": "https://www.globenewswire.com",
      "logo_url": "https://s3.massive.com/public/assets/news/logos/globenewswire.svg",
      "name": "GlobeNewswire Inc."
    },
    "title": "Staff elected to AL Sydbank’s Board of Directors",
    "author": "Al Sydbank A/S",
    "published_utc": "2026-03-10T14:00:00Z",
    "article_url": "https://www.globenewswire.com/news-release/2026/03/10/3253008/0/en/Staff-elected-to-AL-Sydbank-s-Board-of-Directors.html",
    "tickers": [
      "SYANY"
    ],
    "image_url": "https://ml-eu.globenewswire.com/Resource/Download/5ac16b5f-7ad3-4db9-87b5-d247b2423966",
    "description": "AL Sydbank A/S announced the election of six staff members to its Board of Directors for a four-year term, effective after the AGM on 19 March 2026. The elected representatives include account managers, wealth advisory executives, and members of Finansforbundet union. Four substitute members were also appointed.",
    "keywords": [
      "board of directors",
      "staff election",
      "corporate governance",
      "employee representation",
      "Finansforbundet"
    ],
    "insights": [
      {
        "sentiment": "neutral",
        "sentiment_reasoning": "The announcement is a routine corporate governance matter regarding staff election to the board. It is factual and procedural in nature with no indication of positive or negative business developments. The inclusion of employee representatives in board governance is standard practice and does not signal any material change in company performance or strategy.",
        "ticker": "SYANY"
      }
    ],
    "amp_url": null
  },
  {
    "id": "6a47028f98f7e0156c8651e1a62d8e49e70ac3430c4669feec6817c6c277da3f",
    "publisher": {
      "favicon_url": "https://s3.massive.com/public/assets/news/favicons/globenewswire.ico",
      "homepage_url": "https://www.globenewswire.com",
      "logo_url": "https://s3.massive.com/public/assets/news/logos/globenewswire.svg",
      "name": "GlobeNewswire Inc."
    },
    "title": "Medarbejdervalg til AL Sydbanks bestyrelse",
    "author": "Al Sydbank A/S",
    "published_utc": "2026-03-10T14:00:00Z",
    "article_url": "https://www.globenewswire.com/news-release/2026/03/10/3253008/0/da/Medarbejdervalg-til-AL-Sydbanks-bestyrelse.html",
    "tickers": [
      "SYANY"
    ],
    "image_url": "https://ml-eu.globenewswire.com/Resource/Download/5ac16b5f-7ad3-4db9-87b5-d247b2423966",
    "description": "AL Sydbank A/S held an employee election for its board of directors, selecting 6 employees for a 4-year term. The elected members include business advisors, union representatives, and a wealth management director. Four alternates were also chosen. The newly elected board members will take office following the bank's ordinary general meeting on March 19, 2026.",
    "keywords": [
      "board election",
      "employee representatives",
      "corporate governance",
      "Finansforbundet",
      "4-year term"
    ],
    "insights": [
      {
        "sentiment": "neutral",
        "sentiment_reasoning": "The announcement is a routine corporate governance matter regarding employee board elections. It contains no information about financial performance, strategic changes, or market conditions that would indicate positive or negative sentiment. It is a standard procedural disclosure.",
        "ticker": "SYANY"
      }
    ],
    "amp_url": null
  },
  {
    "id": "3187435f047f07460e51c6878637911b5ec39b0c950e0308ba3484d45192812b",
    "publisher": {
      "favicon_url": "https://s3.massive.com/public/assets/news/favicons/benzinga.ico",
      "homepage_url": "https://www.benzinga.com/",
      "logo_url": "https://s3.massive.com/public/assets/news/logos/benzinga.svg",
      "name": "Benzinga"
    },
    "title": "FTC Solar Lands 1-Gigawatt Deal Expansion With Strata Clean Energy",
    "author": "Akanksha Bakshi",
    "published_utc": "2026-03-10T13:40:26Z",
    "article_url": "https://www.benzinga.com/markets/small-cap/26/03/51157223/ftc-solar-lands-1-gigawatt-deal-expansion-with-strata-clean-energy?utm_source=benzinga_taxonomy&utm_medium=rss_feed_free&utm_content=taxonomy_rss&utm_campaign=channel",
    "tickers": [
      "FTCI"
    ],
    "image_url": "https://cdn.benzinga.com/files/images/story/2026/03/10/Aerial-View-Of-Solar-Panel--Photovoltaic.jpeg?width=1200&height=800&fit=crop",
    "description": "FTC Solar (NASDAQ: FTCI) announced a five-year, 1-gigawatt expansion to its solar tracker supply agreement with Strata Clean Energy. The deal extends their partnership after completing the original 500 MW agreement ahead of schedule, with the first project under the expanded agreement expected to start in H2 2027. Despite the positive business news, FTCI shares were down 1.33% to $4.45 on Tuesday.",
    "keywords": [
      "solar trackers",
      "supply agreement",
      "clean energy",
      "partnership expansion",
      "gigawatt capacity"
    ],
    "insights": [
      {
        "sentiment": "positive",
        "sentiment_reasoning": "The company secured a significant 1-gigawatt contract expansion with an existing customer, demonstrating strong market demand for its solar tracking technology and customer satisfaction. The deal represents meaningful business growth and validates the company's product quality and innovation.",
        "ticker": "FTCI"
      }
    ],
    "amp_url": null
  }
]
```

## tickers/all_tickers

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/tickers/all_tickers/all_tickers_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `ticker` | `object` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `name` | `object` | The name of this condition. | 名称。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md |
| `market` | `object` | The market type of the asset. | 市场类别。 | tickers/all_tickers.md |
| `locale` | `object` | An identifier for a geographical location. | 市场区域。 | market_operations/exchanges.md<br>tickers/all_tickers.md<br>tickers/ticker_types.md |
| `primary_exchange` | `object` | Market Identifier Code (MIC) of the primary exchange where the security is listed. The Market Identifier Code (MIC) (ISO 10383) is a unique identification code used to identify securities trading exchanges, regulated and non-regulated trading markets. | 主上市交易所（MIC 代码）。 | corporate_actions/IPOS.md<br>tickers/all_tickers.md |
| `type` | `object` | An identifier for a collection of related conditions. | 类型。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md |
| `active` | `bool` | Whether or not the asset is actively traded. False means the asset has been delisted. | 是否活跃交易。 | tickers/all_tickers.md |
| `currency_name` | `object` | The name of the currency that this asset is traded with. | 交易货币名称。 | tickers/all_tickers.md |
| `cik` | `object` | SEC Central Index Key (CIK) identifying the filing entity. | SEC 中央索引编号（CIK）。 | filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/balance_sheets.md<br>fundamentals/cashflow_statement.md<br>fundamentals/income_statement.md<br>fundamentals/ratios.md<br>tickers/all_tickers.md |
| `composite_figi` | `object` | The composite OpenFIGI number for this ticker. Find more information [here](https://www.openfigi.com/about/figi) | 复合 FIGI 代码。 | tickers/all_tickers.md |
| `share_class_figi` | `object` | The share Class OpenFIGI number for this ticker. Find more information [here](https://www.openfigi.com/about/figi) | 股份类别 FIGI 代码。 | tickers/all_tickers.md |
| `last_updated_utc` | `object` | The information is accurate up to this time. | 最后更新时间（UTC）。 | tickers/all_tickers.md |

### 样本记录（前 3 条）

```json
[
  {
    "ticker": "A",
    "name": "Agilent Technologies Inc.",
    "market": "stocks",
    "locale": "us",
    "primary_exchange": "XNYS",
    "type": "CS",
    "active": true,
    "currency_name": "usd",
    "cik": "0001090872",
    "composite_figi": "BBG000C2V3D6",
    "share_class_figi": "BBG001SCTQY4",
    "last_updated_utc": "2026-03-10T06:11:22.456718773Z"
  },
  {
    "ticker": "AA",
    "name": "Alcoa Corporation",
    "market": "stocks",
    "locale": "us",
    "primary_exchange": "XNYS",
    "type": "CS",
    "active": true,
    "currency_name": "usd",
    "cik": "0001675149",
    "composite_figi": "BBG00B3T3HD3",
    "share_class_figi": "BBG00B3T3HF1",
    "last_updated_utc": "2026-03-10T06:11:22.456719364Z"
  },
  {
    "ticker": "AAA",
    "name": "Alternative Access First Priority CLO Bond ETF",
    "market": "stocks",
    "locale": "us",
    "primary_exchange": "ARCX",
    "type": "ETF",
    "active": true,
    "currency_name": "usd",
    "cik": "0001776878",
    "composite_figi": "BBG01B0JRCS6",
    "share_class_figi": "BBG01B0JRCT5",
    "last_updated_utc": "2026-03-10T06:11:22.456719565Z"
  }
]
```

## tickers/ticker_types

- 文件数: 1
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/tickers/ticker_types/ticker_types_all.parquet`
- 年份分区: -

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `code` | `object` | A code used by Massive to refer to this ticker type. | 类型代码。 | tickers/ticker_types.md |
| `description` | `object` | Detailed explanation of what this risk category encompasses, including specific examples and potential impacts | 描述。 | filing/risk_categories.md<br>market_operations/condition_codes.md<br>news/news.md<br>tickers/ticker_types.md |
| `asset_class` | `object` | An identifier for a group of similar financial instruments. | 资产类别。 | market_operations/condition_codes.md<br>market_operations/exchanges.md<br>tickers/ticker_types.md |
| `locale` | `object` | An identifier for a geographical location. | 市场区域。 | market_operations/exchanges.md<br>tickers/all_tickers.md<br>tickers/ticker_types.md |

### 样本记录（前 3 条）

```json
[
  {
    "code": "CS",
    "description": "Common Stock",
    "asset_class": "stocks",
    "locale": "us"
  },
  {
    "code": "PFD",
    "description": "Preferred Stock",
    "asset_class": "stocks",
    "locale": "us"
  },
  {
    "code": "WARRANT",
    "description": "Warrant",
    "asset_class": "stocks",
    "locale": "us"
  }
]
```

## us_stocks_sip/day_aggs_v1

- 文件数: 5657
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/us_stocks_sip/day_aggs_v1/2003/09/2003-09-10.parquet`
- 年份分区: 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `ticker` | `string` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `volume` | `float64` | 成交量。 | 成交量。 | aggregate_bars/custom_bars.md |
| `open` | `float64` | 开盘价。 | 开盘价。 | aggregate_bars/custom_bars.md |
| `close` | `float64` | 收盘价。 | 收盘价。 | aggregate_bars/custom_bars.md |
| `high` | `float64` | 最高价。 | 最高价。 | aggregate_bars/custom_bars.md |
| `low` | `float64` | 最低价。 | 最低价。 | aggregate_bars/custom_bars.md |
| `window_start` | `float64` | K 线窗口起始时间（Unix 纳秒时间戳）。 | 窗口起始时间戳。 | aggregate_bars/custom_bars.md |
| `transactions` | `float64` | 成交笔数。 | 成交笔数。 | aggregate_bars/custom_bars.md |

### 样本记录（前 3 条）

```json
[
  {
    "ticker": "A",
    "volume": 2869700.0,
    "open": 25.4,
    "close": 24.49,
    "high": 25.58,
    "low": 24.41,
    "window_start": 1.0631664e+18,
    "transactions": 2301.0
  },
  {
    "ticker": "AA",
    "volume": 3543400.0,
    "open": 28.2,
    "close": 27.92,
    "high": 28.7,
    "low": 27.85,
    "window_start": 1.0631664e+18,
    "transactions": 3011.0
  },
  {
    "ticker": "AAp",
    "volume": 550.0,
    "open": 75.0,
    "close": 73.44,
    "high": 75.5,
    "low": 72.65,
    "window_start": 1.0631664e+18,
    "transactions": 6.0
  }
]
```

## us_stocks_sip/minute_aggs_v1

- 文件数: 5657
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/us_stocks_sip/minute_aggs_v1/2003/09/2003-09-10.parquet`
- 年份分区: 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `ticker` | `string` | The ticker symbol of the IPO event. | 证券代码。 | corporate_actions/IPOS.md<br>corporate_actions/dividends.md<br>corporate_actions/splits.md<br>filing/10-K_Sections.md<br>filing/8-K_Text.md<br>filing/risk_factors.md<br>filing/sec_edgar_index.md<br>fundamentals/float.md<br>fundamentals/ratios.md<br>fundamentals/short_interest.md<br>fundamentals/short_volume.md<br>snapshots/unified_snapshot.md<br>tickers/all_tickers.md<br>tickers/related_tickers.md |
| `volume` | `float64` | 成交量。 | 成交量。 | aggregate_bars/custom_bars.md |
| `open` | `float64` | 开盘价。 | 开盘价。 | aggregate_bars/custom_bars.md |
| `close` | `float64` | 收盘价。 | 收盘价。 | aggregate_bars/custom_bars.md |
| `high` | `float64` | 最高价。 | 最高价。 | aggregate_bars/custom_bars.md |
| `low` | `float64` | 最低价。 | 最低价。 | aggregate_bars/custom_bars.md |
| `window_start` | `float64` | K 线窗口起始时间（Unix 纳秒时间戳）。 | 窗口起始时间戳。 | aggregate_bars/custom_bars.md |
| `transactions` | `float64` | 成交笔数。 | 成交笔数。 | aggregate_bars/custom_bars.md |

### 样本记录（前 3 条）

```json
[
  {
    "ticker": "A",
    "volume": 47000.0,
    "open": 25.4,
    "close": 25.4,
    "high": 25.4,
    "low": 25.4,
    "window_start": 1.0632006e+18,
    "transactions": 18.0
  },
  {
    "ticker": "A",
    "volume": 13100.0,
    "open": 25.4,
    "close": 25.43,
    "high": 25.43,
    "low": 25.3,
    "window_start": 1.06320066e+18,
    "transactions": 16.0
  },
  {
    "ticker": "A",
    "volume": 2800.0,
    "open": 25.42,
    "close": 25.45,
    "high": 25.45,
    "low": 25.42,
    "window_start": 1.06320072e+18,
    "transactions": 6.0
  }
]
```

## us_stocks_sip/quotes_v1

- 文件数: 5657
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/us_stocks_sip/quotes_v1/2003/09/2003-09-10.parquet`
- 年份分区: 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `ticker` | `string` | 证券代码。 | 证券代码。 | tickers/all_tickers.md<br>snapshots/unified_snapshot.md |
| `ask_exchange` | `float64` | 卖一报价交易所代码。 | 卖一报价交易所代码。 | snapshots/unified_snapshot.md |
| `ask_price` | `float64` | 卖一价格（ask）。 | 卖一报价价格。 | snapshots/unified_snapshot.md |
| `ask_size` | `float64` | 卖一报单量。 | 卖一报单量。 | snapshots/unified_snapshot.md |
| `bid_exchange` | `float64` | 买一报价交易所代码。 | 买一报价交易所代码。 | snapshots/unified_snapshot.md |
| `bid_price` | `float64` | 买一价格（bid）。 | 买一报价价格。 | snapshots/unified_snapshot.md |
| `bid_size` | `float64` | 买一报单量。 | 买一报单量。 | snapshots/unified_snapshot.md |
| `conditions` | `string` | 成交/报价条件码列表或编码串，可用于过滤异常成交与统计口径控制。 | 成交/报价条件码。 | market_operations/condition_codes.md<br>snapshots/unified_snapshot.md |
| `indicators` | `float64` | 行情附加标记位（indicator flags）。 | 行情附加标志位。 | market_operations/condition_codes.md |
| `participant_timestamp` | `float64` | 参与方（交易所/撮合源）时间戳，通常为纳秒级。 | 参与方时间戳。 | snapshots/unified_snapshot.md |
| `sequence_number` | `float64` | 行情消息序列号，用于排序与去重。 | 消息序列号。 | snapshots/unified_snapshot.md |
| `sip_timestamp` | `float64` | SIP 汇聚时间戳，通常为纳秒级。 | SIP 汇聚时间戳。 | snapshots/unified_snapshot.md<br>market_operations/condition_codes.md |
| `tape` | `float64` | Tape 代码（A/B/C）对应不同证券信息处理器分区。 | Tape 分区代码（A/B/C）。 | market_operations/condition_codes.md |
| `trf_timestamp` | `float64` | 交易报告设施（TRF）时间戳。 | TRF 时间戳。 | market_operations/condition_codes.md |

### 样本记录（前 3 条）

```json
[
  {
    "ticker": "A",
    "ask_exchange": 0.0,
    "ask_price": 0.0,
    "ask_size": 0.0,
    "bid_exchange": 11.0,
    "bid_price": 0.01,
    "bid_size": 1.0,
    "conditions": null,
    "indicators": null,
    "participant_timestamp": 0.0,
    "sequence_number": 1405.0,
    "sip_timestamp": 1.063195276e+18,
    "tape": 1.0,
    "trf_timestamp": 0.0
  },
  {
    "ticker": "A",
    "ask_exchange": 0.0,
    "ask_price": 0.0,
    "ask_size": 0.0,
    "bid_exchange": 11.0,
    "bid_price": 0.01,
    "bid_size": 2.0,
    "conditions": null,
    "indicators": null,
    "participant_timestamp": 0.0,
    "sequence_number": 1431.0,
    "sip_timestamp": 1.063195411e+18,
    "tape": 1.0,
    "trf_timestamp": 0.0
  },
  {
    "ticker": "A",
    "ask_exchange": 11.0,
    "ask_price": 26.0,
    "ask_size": 2.0,
    "bid_exchange": 9.0,
    "bid_price": 25.25,
    "bid_size": 119.0,
    "conditions": null,
    "indicators": null,
    "participant_timestamp": 0.0,
    "sequence_number": 15138.0,
    "sip_timestamp": 1.063200655e+18,
    "tape": 1.0,
    "trf_timestamp": 0.0
  }
]
```

## us_stocks_sip/trades_v1

- 文件数: 5657
- 可读: 是
- 样本文件: `~/quant_projects/data/us_stock/massive_parquet/us_stocks_sip/trades_v1/2003/09/2003-09-10.parquet`
- 年份分区: 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026

### 字段定义

| 字段名 | 类型 | 含义 | 中文释义 | 释义来源 |
|---|---|---|---|---|
| `ticker` | `string` | 证券代码。 | 证券代码。 | tickers/all_tickers.md<br>snapshots/unified_snapshot.md |
| `conditions` | `string` | 成交/报价条件码列表或编码串，可用于过滤异常成交与统计口径控制。 | 成交/报价条件码。 | market_operations/condition_codes.md<br>snapshots/unified_snapshot.md |
| `correction` | `float64` | 成交更正标记（trade correction indicator）。 | 成交更正标记。 | market_operations/condition_codes.md |
| `exchange` | `float64` | 成交发生的交易所代码。 | 交易所代码。 | market_operations/exchanges.md<br>snapshots/unified_snapshot.md |
| `id` | `float64` | 成交记录标识（trade id）。 | 记录唯一标识。 | snapshots/unified_snapshot.md |
| `participant_timestamp` | `float64` | 参与方（交易所/撮合源）时间戳，通常为纳秒级。 | 参与方时间戳。 | snapshots/unified_snapshot.md |
| `price` | `float64` | 成交价格（trade price）。 | 价格。 | snapshots/unified_snapshot.md |
| `sequence_number` | `float64` | 行情消息序列号，用于排序与去重。 | 消息序列号。 | snapshots/unified_snapshot.md |
| `sip_timestamp` | `float64` | SIP 汇聚时间戳，通常为纳秒级。 | SIP 汇聚时间戳。 | snapshots/unified_snapshot.md<br>market_operations/condition_codes.md |
| `size` | `float64` | 成交数量（trade size）。 | 成交数量。 | snapshots/unified_snapshot.md |
| `tape` | `float64` | Tape 代码（A/B/C）对应不同证券信息处理器分区。 | Tape 分区代码（A/B/C）。 | market_operations/condition_codes.md |
| `trf_id` | `float64` | 交易报告设施标识（TRF ID）。 | TRF 标识。 | market_operations/condition_codes.md |
| `trf_timestamp` | `float64` | 交易报告设施（TRF）时间戳。 | TRF 时间戳。 | market_operations/condition_codes.md |

### 样本记录（前 3 条）

```json
[
  {
    "ticker": "A",
    "conditions": null,
    "correction": 0.0,
    "exchange": 10.0,
    "id": null,
    "participant_timestamp": 0.0,
    "price": 25.4,
    "sequence_number": 1929946249800780.0,
    "sip_timestamp": 1.0632006498e+18,
    "size": 42700.0,
    "tape": 1.0,
    "trf_id": 0.0,
    "trf_timestamp": 0.0
  },
  {
    "ticker": "A",
    "conditions": null,
    "correction": 0.0,
    "exchange": 9.0,
    "id": null,
    "participant_timestamp": 0.0,
    "price": 25.4,
    "sequence_number": 1929946251320893.0,
    "sip_timestamp": 1.06320065132e+18,
    "size": 500.0,
    "tape": 1.0,
    "trf_id": 0.0,
    "trf_timestamp": 0.0
  },
  {
    "ticker": "A",
    "conditions": null,
    "correction": 0.0,
    "exchange": 9.0,
    "id": null,
    "participant_timestamp": 0.0,
    "price": 25.4,
    "sequence_number": 1929946251325765.0,
    "sip_timestamp": 1.063200651325e+18,
    "size": 500.0,
    "tape": 1.0,
    "trf_id": 0.0,
    "trf_timestamp": 0.0
  }
]
```