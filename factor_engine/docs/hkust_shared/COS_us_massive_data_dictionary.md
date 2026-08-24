# COS 美股数据字典（massive_data）— AI 完整版

```yaml
doc_role: machine_readable_us_equity_data_dictionary
audience: AI_agents_adapting_factor_engine_multi_market
purpose: 完整理解美股 COS 清洗数据的布局、字段、单位、PIT/join，并对照 A股以改造 factor_engine
bucket: qs-cold
cos_root: cos://qs-cold/clean_data/us_stock/massive_data/
aux_root: cos://qs-cold/clean_data/{adj_factor,is_adj_factor_clamped,is_early_close,is_ticker_halt,universe_daily}/
format: Parquet
tables: 26
fields_sampled: 311
currency: USD
instrument_id: Ticker
generated: 2026-08-08T00:23:52+0800
schema_verified: live_clean-cos-ro_parquet_pyarrow_2024-06-03_and_filings_2024-03-31_X0_2026-05-14
coverage_verified: clean-cos-ro_ls_2026-08-08
ashare_companion: /home/shw/COS_ashare_lqtp_data_dictionary.md
updated: 2026-08-08T02:00:00+0800
cross_market_section: CROSS_MARKET_BRIDGE
sole_deliverables: [COS_ashare_lqtp_data_dictionary.md, COS_us_massive_data_dictionary.md]
merged_from: [COS_clean_data_catalog.md, COS_field_registry.json]
news_section: NEWS_RAWDATA
machine_registry_section: MACHINE_FIELD_REGISTRY
schema_rechecked: 2026-08-08_unit_matrix_C16_C18
p1_p2_closed: C24_2026-08-08
gaps_filled: C23_C26_full_2026-08-08
second_audit: C26_div_enums_shares_grain_aux_samples
```

## 0. AI 阅读协议

```text
STEP1 读 §1 HARD_RULES（Ret 小数 / AdjFactor 后复权 / 财务 filing_date）
STEP1b 新闻因子 → NEWS_RAWDATA 节（raw_data，非 clean）+ FactNews 表节
STEP2 §2 TABLE_INDEX 定位表与 model
STEP3 D1 equi；E2 asof(filing_date)；X0/EMPTY 拒绝当全市场面板
STEP4 §4 A股对照 → 决定字段映射与算子是否 both/ashare-only/us-only
STEP5 §5 逐字段字典
STEP6 §6 checklist + §8/实测
STEP7 跨市场必读文末 CROSS_MARKET_BRIDGE（与 A股字典同一份对照表）
STEP8 写收益/股息/ROE 因子前必读 Bridge **§C16**（单位归一；禁止对美股再 /100）
```

## 1. HARD_RULES

### 1.1 口径

| 项目 | 美股规则 | 对比 A股 |
|------|----------|----------|
| 标的 | `Ticker`=`AAPL`/`BRK.B` | A股 `Symbol`=`000001.SZ` |
| 日期 | 常 `timestamp[ns]` | A股 `date32` |
| 收益 | `Ret` **小数** | `Return` **bp=/10000** |
| 复权 | `AdjFactor` **后复权**（Close×AdjFactor） | `Factor` **后复权**（Close×Factor；旧文档误称前复权已更正） |
| 货币 | USD | CNY |
| 财务文件名 | **period_end** | **PubDate** |
| 财务 PIT | **filing_date** | **PubDate** |
| 估值日面板 | Valuation/Indicator **X0 稀疏(~49日)** | Valuation **D1 全历史** |
| 行业/状态 | Industry/Status **EMPTY** | 完整 D1/S1 |


### 1.1b 单位总表（美股 · 2026-08-08 实测）

> 与 A 股对照/归一见文末 **CROSS_MARKET_BRIDGE §C16–C18**。

| 字段族 | 存盘单位 | 是否再转换 | 证据 |
|--------|----------|------------|------|
| `Open/High/Low/Close/PreClose/VWAP` | USD/股（未复权） | 否 | |
| `Volume` | 股（double） | 否 | |
| `Amount` | USD | 否 | 非空时 **=`VWAP×Volume`（精确）**；与 VWAP 同步缺失约 3.35% |
| `Ret` / `Ret_Intra` / `Ret_Overnight` | **小数** | **禁止 /10000** | `Ret==Close/PreClose-1` |
| `High_Low_Ratio` | 小数 | 否 | **=`(H-L)/Low`**（不是 /Close） |
| `Upper_Shadow_Ratio` / `Vwap_Close_Dist` | 小数 | 否 | `Vwap_Close_Dist==(VWAP-Close)/Close` |
| `AdjFactor` / `adj_factor` | 后复权因子 | 调整价≈价×因子 | 与日线列一致；hive `adj_factor/date=YYYY-MM-DD/data.parquet` |
| `is_adj_factor_clamped` | bool | 截断标记 | 与 `AdjFactor` 併用 |
| `dividend_yield` | **小数** | **禁止 /100** | >0 中位≈0.023 |
| `return_on_equity` / `return_on_assets` | **小数** | **禁止 /100** | ROE 中位≈0.054 |
| `price_to_*` / `ev_to_*` / `debt_to_equity` / `current`/`quick` | 倍数或比率 | 否 | PE 中位≈21 |
| `market_cap` / `enterprise_value` / `free_cash_flow` | USD | 否 | Valuation **X0** |
| 财务三表金额字段 | USD 绝对额 | 否 | 先 **filter `timeframe`** |
| `basic_earnings_per_share` 等 | USD/股 | 否 | |
| `cash_amount`（分红） | 面值货币/股 | 看 `currency` | 不全是 USD；`frequency` 12/4/2/1 |
| `Weight` | **不存在** | — | Components 仅 IndexName+Symbol |
| `TradeDate` 等 | 常 `timestamp[ns]` | → date | 与 A `date32` 对齐前先归一 |

### 1.1c 财务宽表单位缺省规则

- `StockBalance` / `StockIncome` / `StockCashFlow` 中未另行标注的 **double 金额科目一律为 USD 绝对额**（不是千/百万；以落盘数值为准）。
- `*_per_share` / EPS → **USD/股**；`*shares_outstanding*` → **股**。
- 必须先 **filter `timeframe`**，再用 `filing_date` 做 asof。

### 1.2 MUST / MUST_NOT

```text
MUST:
  - Ret/Ret_Intra 已是小数，禁止 /10000 或 /100
  - dividend_yield / return_on_equity / return_on_assets 已是小数，禁止再 /100（A股对应字段才是 %）
  - 财报 asof(filing_date)；读入后先 filter timeframe（quarterly|annual|trailing_twelve_months）
  - TradeDate/timestamp 归一 date；用 massive_data
  - 日频市值优先 Close*TickerSharesSnapshot.weighted_shares_outstanding
  - Capital 目录按文件名分流：{date}.parquet=拆分事件；shares_{date}=稀疏PIT股本
MUST_NOT:
  - Valuation/Indicator 当全历史截面
  - 用 EMPTY 行业/状态
  - 文件名 period_end 当 PIT
  - 把 A股 Return bp / TurnoverRatio% 公式套到美股
  - 混读 StockCapitalDaily 两种 schema
  - 用 massive_data_temp
```

### 1.3 访问

```bash
clean-cos-ro ls cos://qs-cold/clean_data/us_stock/massive_data/
clean-cos-ro cp cos://qs-cold/clean_data/us_stock/massive_data/StockDailyBar/2024-06-03.parquet ./
```

## 2. TABLE_INDEX

| 表 | model | 字段数 | 文件数 | 跨度 | join |
|----|-------|------:|------:|------|------|
| `Calendar` | `STATIC` | 2 | 1 | full→full | read_full |
| `StockDailyBar` | `D1` | 17 | 5760 | 2003-09-10→2026-08-03 | equi(date,Ticker) |
| `StockList` | `D1` | 9 | 5760 | 2003-09-10→2026-08-03 | equi(date,Symbol≈Ticker) |
| `ETFDailyBar` | `D1` | 17 | 5760 | 2003-09-10→2026-08-03 | equi(date,Ticker) |
| `ETFList` | `STATIC` | 14 | 1 | full→full | read_full |
| `SecurityMaster` | `STATIC` | 16 | 1 | full→full | read_full |
| `SecurityMasterDailySnap` | `D1` | 17 | 5765 | 2003-09-10→2026-08-07 | equi(date,Ticker) |
| `TickerMap` | `STATIC` | 4 | 1 | full→full | read_full |
| `TickerAlias` | `EMPTY` | 2 | 1 | full→full | DO_NOT_USE |
| `TickerSharesSnapshot` | `D1` | 8 | 4099 | 2010-04-23→2026-08-07 | equi(date,ticker) |
| `StockIndicesComponents` | `D1` | 3 | 6508 | 1932-05-26→2026-08-03 | equi(date,Symbol) |
| `StockBalance` | `E2` | 38 | 1941 | 2009-04-30→2026-05-10 | asof(filing_date) |
| `StockIncome` | `E2` | 34 | 1979 | 2009-03-29→2026-05-10 | asof(filing_date) |
| `StockCashFlow` | `E2` | 32 | 1953 | 2009-03-29→2026-05-10 | asof(filing_date) |
| `StockDividend` | `E2` | 13 | 6166 | 2000-08-15→2027-11-12 | event(ex_dividend_date) |
| `StockCapitalDaily` | `E2` **双schema** | 4或8 | 9824 | 2000-03-31→2026-09-29 | split事件=`{date}.parquet`；PIT=`shares_{date}.parquet`；日频股本改用 SharesSnapshot |
| `StockValuationDaily` | `X0` | 24 | 49 | 2026-05-12→2026-07-27 | CHECK nrows; 禁止当全历史面板 |
| `StockIndicator` | `X0` | 23 | 49 | 2026-05-12→2026-07-27 | CHECK nrows |
| `StockStatus` | `EMPTY` | 4 | 1 | full→full | DO_NOT_USE |
| `StockIndustry` | `EMPTY` | 6 | 1 | full→full | DO_NOT_USE |
| `FactNews` | `RAW_EVENT` | 16 | 2198 | 2016-06-22→2026-06-12 | explode(tickers); PIT=published_utc |
| `adj_factor` | `D1` | 2 | hive | 2003-09-10→~2026 | equi(date,ticker) |
| `is_adj_factor_clamped` | `D1` | 2 | hive | 2003-09-10→~2026 | equi(date,ticker) |
| `is_early_close` | `STATIC` | 2 | 1 | full→full | read_full |
| `universe_daily` | `D1` | 2 | year-hive | 2003→~2026 | equi(date,ticker) |
| `is_ticker_halt` | `MINUTE` | ? | hive | ?→2026-06-11 | minute；按日检查文件是否存在 |

## 3. 目录树

```text
cos://qs-cold/clean_data/us_stock/massive_data/
├── Calendar/full.parquet
├── StockDailyBar|ETFDailyBar|StockList|SecurityMasterDailySnap|TickerSharesSnapshot|StockIndicesComponents|FactNews/{YYYY-MM-DD}.parquet
├── ETFList|SecurityMaster|TickerMap|TickerAlias|StockStatus|StockIndustry/full.parquet
├── StockBalance|Income|CashFlow/{period_end}.parquet
├── StockDividend/{ex_div_date}.parquet
├── StockCapitalDaily/shares_{YYYY-MM-DD}.parquet
└── StockValuationDaily|StockIndicator/{YYYY-MM-DD}.parquet   # X0
cos://qs-cold/clean_data/adj_factor|is_adj_factor_clamped|is_early_close|is_ticker_halt|universe_daily/...
```

## 4. A股对照与 factor_engine 改造要点

> 下面是摘要；**完整字段级对照、同名陷阱、仅单边字段、adapter 映射**见文末 **CROSS_MARKET_BRIDGE**（与 A股字典同步维护）。

| 概念 | A股 | 美股 | 算子影响 |
|------|----|----|----------|
| 标的键 | Symbol | Ticker | 全部数据源适配器 |
| 收益 | Return bp | Ret 小数 | 收益/IC/回测单位 |
| 复权 | Factor 前 | AdjFactor 后 | 复权价宏/技术指标入口 |
| VWAP | Vwap | VWAP | 列名映射 |
| 行业中性 | ✓ Industry | ✗ EMPTY | industry_neutralize 默认 ashare-only |
| 市值中性 | Valuation D1 | shares×Close 或稀疏 Valuation | size_neutralize 美股改口径 |
| 涨跌停 | 有 | 无 | limit_* ashare-only |
| 财务 asof | PubDate | filing_date | fiscal 算子时间键 |
| 指数权重 | Weight% | 常无 | 加权中性不对称 |

### 4.1 算子市场能力矩阵

| 能力 | ashare | us | both | 备注 |
|------|:------:|:--:|:----:|------|
| ts_*/OHLC 技术指标 | ✓ | ✓ | ✓ | 复权入口分叉 |
| rank/zscore/cs_* | ✓ | ✓ | ✓ | 需 CS 宇宙 |
| industry_neutralize | ✓ | ✗ | | US 缺行业表 |
| size_neutralize | ✓ | △ | | US 用 SharesSnapshot×Close |
| industry_size_neutralize | ✓ | ✗ | | 依赖行业 |
| limit_up/down | ✓ | ✗ | | |
| fin_*/fiscal | ✓ | ✓ | ✓ | asof 键不同 |
| 分红事件 | ✓ | ✓ | ✓ | |

## 5. FIELD_DICTS

### `Calendar`
- **vs_ashare**: A股 `Calendar.IsTradeDay`（PascalCase）；本表 `is_trading_day`。跨市场日期对齐须 **inner join 双方交易日**（见 Bridge C23.9），勿假设同一自然日两边都开市。

- **model**: `STATIC`
- **purpose**: 美股交易日历
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/Calendar/full.parquet`
- **coverage**: files=1, full → full
- **join**: read_full
- **sample_rows / n_fields**: 8311 / 2

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `trade_date` | `date32[day]` | 交易日（辅助表/宇宙对齐键） | date32[day] | 0.0 |  |
| 2 | `is_trading_day` | `bool` | 是否美股交易日 | bool | 0.0 |  |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "is_trading_day": [
    [
      "True",
      5937
    ],
    [
      "False",
      2374
    ]
  ]
}
```
</details>

<details><summary>sample_row</summary>

```json
{
  "trade_date": "2003-09-10",
  "is_trading_day": true
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "trade_date": "2003-09-10",
    "is_trading_day": true
  },
  {
    "trade_date": "2015-01-25",
    "is_trading_day": false
  },
  {
    "trade_date": "2026-06-11",
    "is_trading_day": true
  }
]
```
</details>

### `StockDailyBar`

- **model**: `D1`
- **purpose**: 美股股票日线 OHLCV+收益/复权派生
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockDailyBar/{YYYY-MM-DD}.parquet`
- **coverage**: files=5760, 2003-09-10 → 2026-08-03
- **join**: equi(date,Ticker)
- **sample_rows / n_fields**: 10561 / 17
- **vs_ashare**: 映射 `Ticker←Symbol`，`Ret←Return/10000`，`AdjFactor≠Factor`，`VWAP←Vwap`；无 HighLimit/LowLimit。详见 CROSS_MARKET_BRIDGE。
- **note_null_PreClose**: 样本日 `PreClose`/`Ret` 缺失约 1.92%；缺失代码多为 `W`/`WS`/`U`/`R` 等权证与单元，很多不在 StockList(CS)。
- **恒等式（已验证）**: `Ret=Close/PreClose-1`；`Ret_Intra=(Close-Open)/Open`；`Amount=VWAP*Volume`（非空时）；`Vwap_Close_Dist=(VWAP-Close)/Close`；`Upper_Shadow_Ratio=(High-max(Open,Close))/(High-Low)`；`High_Low_Ratio=(High-Low)/Low`；`AdjFactor`≡辅助表 `adj_factor`。

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `Ticker` | `string` | 美股代码（无交易所后缀，如 AAPL、BRK.B） | string | 0.0 | 对应 A股 `Symbol`（带 .SH/.SZ）；禁止直接 concat。 |
| 2 | `TradeDate` | `timestamp[ns]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | timestamp[ns] | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |
| 3 | `Open` | `double` | 未复权开盘价（USD/股） | USD/股 | 0.0 |  |
| 4 | `High` | `double` | 未复权最高价（USD/股） | USD/股 | 0.0 |  |
| 5 | `Low` | `double` | 未复权最低价（USD/股） | USD/股 | 0.0 |  |
| 6 | `Close` | `double` | 未复权收盘价（USD/股） | USD/股 | 0.0 |  |
| 7 | `Volume` | `double` | 成交量（股，double） | 股 | 0.0 |  |
| 8 | `AdjFactor` | `double` | 后复权累积因子（调整价≈原价×AdjFactor） | 无量纲 | 0.0 | A股 Factor 同样是后复权乘数（Close×Factor）；与 AdjFactor 公式同类但仍禁止混列。若 clamped=True 慎用绝对复权价。 |
| 9 | `PreClose` | `double` | 因子调整后的前收（供 Ret） | USD/股 | 1.92 |  |
| 10 | `Ret` | `double` | 日收益率（小数，0.01=1%） | 小数 | 1.92 | A股对应 Return 是 bp；美股 Ret 不要 /10000。 |
| 11 | `Ret_Intra` | `double` | 日内收益率 ≈(Close-Open)/Open | 小数 | 0.0 |  |
| 12 | `Ret_Overnight` | `double` | 隔夜收益率（开盘相对前收） | 小数 | 1.92 | **=`(Open-PreClose)/PreClose`**（已验证）。 |
| 13 | `High_Low_Ratio` | `double` | 振幅比 (High−Low)/Low | 小数 | 0.0 | **精确** `(High-Low)/Low`（2026-08-08 验证 match=100%）。不是 /Close，也不是 /PreClose。 |
| 14 | `Upper_Shadow_Ratio` | `double` | 上影线占比 | 小数 | 0.0 | **=`(High-max(Open,Close))/(High-Low)`**；H=L 时为 0。 |
| 15 | `VWAP` | `double` | 成交量加权均价 VWAP（USD/股） | USD/股 | 3.35 | A股列名是 `Vwap`（仅大小写不同）。 |
| 16 | `Amount` | `double` | 成交金额（USD） | USD | 3.35 |  |
| 17 | `Vwap_Close_Dist` | `double` | VWAP 相对收盘偏离 (VWAP-Close)/Close | 小数 | 3.35 |  |

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `Open` | 0.0 | 0.0039 | 22.7 | 102.11359029930877 | 625734.85 |
| `High` | 0.0 | 0.0039 | 22.83 | 113.4156794538396 | 741971.39 |
| `Low` | 0.0 | 0.0038 | 22.41 | 100.89751150099421 | 620700.0 |
| `Close` | 0.0 | 0.0038 | 22.6 | 102.33609700643878 | 631110.1 |
| `Volume` | 0.0 | 0.0 | 72819.0 | 1109606.9596629108 | 298463243.0 |
| `AdjFactor` | 0.0 | 2.204585537918872e-14 | 1.0264466239983743 | 2.2648285615649786 | 6210.622537928616 |
| `PreClose` | 1.92 | 0.0039 | 23.015050000000002 | 103.94088092172399 | 627400.0 |
| `Ret` | 1.92 | -0.9738372093023255 | 0.0 | 0.011086141367533565 | 109.3359173126615 |
| `Ret_Intra` | 0.0 | -0.5928816734311583 | -0.002137146371463894 | -0.005066188898959726 | 3.085021587512455 |
| `Ret_Overnight` | 1.92 | -0.9714285714285714 | 0.0030806420947584456 | 0.021838053920201345 | 162.56589147286823 |
| `High_Low_Ratio` | 0.0 | 0.0 | 0.021775361924162295 | 0.05852180904616352 | 71.61538461538461 |
| `Upper_Shadow_Ratio` | 0.0 | 0.0 | 0.09876543209876362 | 0.18658324225432268 | 1.0 |
| `VWAP` | 3.35 | 0.0038172764172025356 | 22.019872759591614 | 104.07549928325675 | 633555.0116582763 |
| `Amount` | 3.35 | 1.16 | 1058431.458118011 | 58011410.41982826 | 49821714839.15259 |
| `Vwap_Close_Dist` | 3.35 | -0.396 | -0.00022143504812199843 | 0.0010883442398591256 | 0.946969696969697 |

</details>

<details><summary>sample_row</summary>

```json
{
  "Ticker": "AAPL",
  "TradeDate": "2024-06-03 00:00:00",
  "Open": 192.9,
  "High": 194.99,
  "Low": 192.52,
  "Close": 194.03,
  "Volume": 50080539.0,
  "AdjFactor": 132.58263088104235,
  "PreClose": 192.25,
  "Ret": 0.009258777633289972,
  "Ret_Intra": 0.005857957490927879,
  "Ret_Overnight": 0.0033810143042913854,
  "High_Low_Ratio": 0.012829835861209116,
  "Upper_Shadow_Ratio": 0.3886639676113394,
  "VWAP": 193.77785438231206,
  "Amount": 9704499393.7297,
  "Vwap_Close_Dist": -0.0012995187223003857
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "Ticker": "A",
    "TradeDate": "2024-06-03 00:00:00",
    "Open": 130.0,
    "High": 132.58,
    "Low": 130.0,
    "Close": 131.4,
    "Volume": 3114841.0,
    "AdjFactor": 1.5236417823895225,
    "PreClose": 130.41,
    "Ret": 0.007591442374051072,
    "Ret_Intra": 0.010769230769230864,
    "Ret_Overnight": -0.003143930680162499,
    "High_Low_Ratio": 0.019846153846153847,
    "Upper_Shadow_Ratio": 0.45736434108527174,
    "VWAP": 131.74857485571732,
    "Amount": 410375862.6521574,
    "Vwap_Close_Dist": 0.0026527766797359575
  },
  {
    "Ticker": "JPM",
    "TradeDate": "2024-06-03 00:00:00",
    "Open": 202.31,
    "High": 202.42,
    "Low": 199.19,
    "Close": 201.82,
    "Volume": 6444309.0,
    "AdjFactor": 1.7596532916384435,
    "PreClose": 202.63,
    "Ret": -0.003997433746236956,
    "Ret_Intra": -0.002422025604270739,
    "Ret_Overnight": -0.0015792330849331293,
    "High_Low_Ratio": 0.016215673477584236,
    "Upper_Shadow_Ratio": 0.034055727554175096,
    "VWAP": 200.9711334074382,
    "Amount": 1295120083.7577546,
    "Vwap_Close_Dist": -0.004206057836496857
  },
  {
    "Ticker": "ZZZ",
    "TradeDate": "2024-06-03 00:00:00",
    "Open": 25.35,
    "High": 25.35,
    "Low": 24.99,
    "Close": 25.0756,
    "Volume": 13928.0,
    "AdjFactor": 1.000475899691558,
    "PreClose": 24.8794,
    "Ret": 0.00788604226790035,
    "Ret_Intra": -0.01082445759368833,
    "Ret_Overnight": 0.01891524715226245,
    "High_Low_Ratio": 0.014405762304922076,
    "Upper_Shadow_Ratio": 0.0,
    "VWAP": 25.075082724698163,
    "Amount": 349245.752189596,
    "Vwap_Close_Dist": -2.062863109308921e-05
  }
]
```
</details>

### `StockList`

- **note_type**: 多日抽查（2016/2020/2024-06/2024-12）`type` **均为 CS**；ETF/权证不在此表。

- **model**: `D1`
- **purpose**: 每日证券清单（含 type）
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockList/{YYYY-MM-DD}.parquet`
- **coverage**: files=5760, 2003-09-10 → 2026-08-03
- **join**: equi(date,Symbol≈Ticker)
- **sample_rows / n_fields**: 5213 / 9

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `TradeDate` | `timestamp[ns]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | timestamp[ns] | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |
| 2 | `Symbol` | `string` | 代码别名列（美股语义同 Ticker，≠A股 Symbol） | string | 0.0 | 美股 Symbol≠A股 Symbol。 |
| 3 | `name` | `string` | 证券英文名称 | string | 0.0 |  |
| 4 | `type` | `string` | 证券类型（CS/ETF/ETN/ETP/ADRC 等） | string | 0.0 |  |
| 5 | `market` | `string` | 市场（通常 stocks） | string | 0.0 |  |
| 6 | `locale` | `string` | 地区（us） | string | 0.0 |  |
| 7 | `last_updated_utc` | `string` | 上游更新时间（ISO） | string | 0.0 |  |
| 8 | `delisted_utc` | `string` | 退市时间（ISO，未退市为空） | string | 80.4 | 样本缺失 80.4%. |
| 9 | `UpdateTime` | `timestamp[us, tz=UTC]` | 清洗管线产出时间（UTC） | timestamp[us, tz=UTC] | 0.0 |  |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "type": [
    [
      "CS",
      5213
    ]
  ],
  "market": [
    [
      "stocks",
      5213
    ]
  ],
  "locale": [
    [
      "us",
      5213
    ]
  ]
}
```
</details>

<details><summary>sample_row</summary>

```json
{
  "TradeDate": "2024-06-03 00:00:00",
  "Symbol": "AAPL",
  "name": "Apple Inc.",
  "type": "CS",
  "market": "stocks",
  "locale": "us",
  "last_updated_utc": "2026-08-06T06:11:44.671893635Z",
  "delisted_utc": null,
  "UpdateTime": "2026-08-06 23:44:41.632998+00:00"
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "TradeDate": "2024-06-03 00:00:00",
    "Symbol": "A",
    "name": "Agilent Technologies Inc.",
    "type": "CS",
    "market": "stocks",
    "locale": "us",
    "last_updated_utc": "2026-08-06T06:11:44.671887854Z",
    "delisted_utc": null,
    "UpdateTime": "2026-08-06 23:44:41.632998+00:00"
  },
  {
    "TradeDate": "2024-06-03 00:00:00",
    "Symbol": "KLAC",
    "name": "KLA Corporation Common Stock",
    "type": "CS",
    "market": "stocks",
    "locale": "us",
    "last_updated_utc": "2026-08-06T06:11:44.672677862Z",
    "delisted_utc": null,
    "UpdateTime": "2026-08-06 23:44:41.632998+00:00"
  },
  {
    "TradeDate": "2024-06-03 00:00:00",
    "Symbol": "ZYXI",
    "name": "ZYNEX INC",
    "type": "CS",
    "market": "stocks",
    "locale": "us",
    "last_updated_utc": "2025-12-25T07:06:19.28938229Z",
    "delisted_utc": "2025-12-24T00:00:00Z",
    "UpdateTime": "2026-08-06 23:44:41.632998+00:00"
  }
]
```
</details>

### `ETFDailyBar`
- **vs_ashare**: 价量映射同 StockDailyBar；A股 ETF 日线列集合=股票日线（含 Return bp / Factor / 涨跌停）。美股 ETF 无涨跌停列。

- **model**: `D1`
- **purpose**: ETF 日线
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/ETFDailyBar/{YYYY-MM-DD}.parquet`
- **coverage**: files=5760, 2003-09-10 → 2026-08-03
- **join**: equi(date,Ticker)
- **sample_rows / n_fields**: 3422 / 17

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `Ticker` | `string` | 美股代码（无交易所后缀，如 AAPL、BRK.B） | string | 0.0 |  |
| 2 | `TradeDate` | `timestamp[ns]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | timestamp[ns] | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |
| 3 | `Open` | `double` | 未复权开盘价（USD/股） | USD/股 | 0.0 |  |
| 4 | `High` | `double` | 未复权最高价（USD/股） | USD/股 | 0.0 |  |
| 5 | `Low` | `double` | 未复权最低价（USD/股） | USD/股 | 0.0 |  |
| 6 | `Close` | `double` | 未复权收盘价（USD/股） | USD/股 | 0.0 |  |
| 7 | `Volume` | `double` | 成交量（股，double） | 股 | 0.0 |  |
| 8 | `AdjFactor` | `double` | 后复权累积因子（调整价≈原价×AdjFactor） | 无量纲 | 0.0 | A股 Factor 同样是后复权乘数（Close×Factor）；与 AdjFactor 公式同类但仍禁止混列。若 clamped=True 慎用绝对复权价。 |
| 9 | `PreClose` | `double` | 因子调整后的前收（供 Ret） | USD/股 | 0.32 |  |
| 10 | `Ret` | `double` | 日收益率（小数，0.01=1%） | 小数 | 0.32 | A股对应 Return 是 bp；美股 Ret 不要 /10000。 |
| 11 | `Ret_Intra` | `double` | 日内收益率 ≈(Close-Open)/Open | 小数 | 0.0 |  |
| 12 | `Ret_Overnight` | `double` | 隔夜收益率（开盘相对前收） | 小数 | 0.32 | **=`(Open-PreClose)/PreClose`**（已验证）。 |
| 13 | `High_Low_Ratio` | `double` | 振幅比 (High−Low)/Low | 小数 | 0.0 | **精确** `(High-Low)/Low`（2026-08-08 验证 match=100%）。不是 /Close，也不是 /PreClose。 |
| 14 | `Upper_Shadow_Ratio` | `double` | 上影线占比 | 小数 | 0.0 | **=`(High-max(Open,Close))/(High-Low)`**；H=L 时为 0。 |
| 15 | `VWAP` | `double` | 成交量加权均价 VWAP（USD/股） | USD/股 | 9.64 |  |
| 16 | `Amount` | `double` | 成交金额（USD） | USD | 9.64 |  |
| 17 | `Vwap_Close_Dist` | `double` | VWAP 相对收盘偏离 (VWAP-Close)/Close | 小数 | 9.64 |  |

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `Open` | 0.0 | 1.19 | 32.16 | 44.41325777323203 | 859.5684 |
| `High` | 0.0 | 1.19 | 32.275000000000006 | 44.527914815897134 | 859.5684 |
| `Low` | 0.0 | 1.17 | 31.978749999999998 | 43.9757963471654 | 859.5684 |
| `Close` | 0.0 | 1.185 | 32.11505 | 44.24448039158388 | 859.5684 |
| `Volume` | 0.0 | 0.0 | 14472.5 | 593526.7632963179 | 134053601.0 |
| `AdjFactor` | 0.0 | 3.3333333333333353e-10 | 1.0654220662553162 | 1.600938294174214 | 197.3638102401573 |
| `PreClose` | 0.32 | 1.19 | 32.1961 | 44.324959073761356 | 854.4581 |
| `Ret` | 0.32 | -0.31597975415762825 | 0.00030213080083352217 | -0.0002903456528293706 | 0.09606909924202367 |
| `Ret_Intra` | 0.0 | -0.2603596559812352 | -0.0005423115791933775 | -0.0026230633762868057 | 0.11482987876417683 |
| `Ret_Overnight` | 0.32 | -0.10001231895535256 | 0.0017602404482015555 | 0.0023587875806192742 | 0.0836013071895425 |
| `High_Low_Ratio` | 0.0 | 0.0 | 0.007518345748818045 | 0.011391654512303081 | 0.39324618736383443 |
| `Upper_Shadow_Ratio` | 0.0 | 0.0 | 0.0 | 0.12299593654037744 | 1.0 |
| `VWAP` | 9.64 | 1.185705683528186 | 33.1430401973347 | 45.1734664406254 | 542.7421092753517 |
| `Amount` | 9.64 | 1388.0 | 697582.9891601757 | 47486315.207573086 | 24660462569.156555 |
| `Vwap_Close_Dist` | 9.64 | -0.05683223185629549 | -0.0005038865645921042 | -0.00020192270038781819 | 0.07703639189734779 |

</details>

<details><summary>sample_row</summary>

```json
{
  "Ticker": "QQQ",
  "TradeDate": "2024-06-03 00:00:00",
  "Open": 454.57,
  "High": 455.58,
  "Low": 447.9,
  "Close": 453.13,
  "Volume": 32997467.0,
  "AdjFactor": 1.1303111511083284,
  "PreClose": 450.71,
  "Ret": 0.005369306205764257,
  "Ret_Intra": -0.003167828937237438,
  "Ret_Overnight": 0.008564265270351257,
  "High_Low_Ratio": 0.01714668452779633,
  "Upper_Shadow_Ratio": 0.13151041666666535,
  "VWAP": 451.9754072725854,
  "Amount": 14914043586.288696,
  "Vwap_Close_Dist": -0.002548038592489088
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "Ticker": "AAA",
    "TradeDate": "2024-06-03 00:00:00",
    "Open": 25.04,
    "High": 25.15,
    "Low": 25.04,
    "Close": 25.05,
    "Volume": 6900.0,
    "AdjFactor": 1.2604388822443302,
    "PreClose": 24.89196,
    "Ret": 0.00634903800263209,
    "Ret_Intra": 0.0003993610223642641,
    "Ret_Overnight": 0.005947301859716925,
    "High_Low_Ratio": 0.004392971246006461,
    "Upper_Shadow_Ratio": 0.9090909090908944,
    "VWAP": 25.07722852750256,
    "Amount": 173032.87683976765,
    "Vwap_Close_Dist": 0.0010869671657707247
  },
  {
    "Ticker": "JAVA",
    "TradeDate": "2024-06-03 00:00:00",
    "Open": 59.95,
    "High": 59.95,
    "Low": 59.0,
    "Close": 59.4,
    "Volume": 68862.0,
    "AdjFactor": 0.25956405260269705,
    "PreClose": 59.81,
    "Ret": -0.0068550409630496745,
    "Ret_Intra": -0.00917431192660556,
    "Ret_Overnight": 0.002340745694699997,
    "High_Low_Ratio": 0.016101694915254372,
    "Upper_Shadow_Ratio": 0.0,
    "VWAP": 59.48306565168784,
    "Amount": 4096122.866906528,
    "Vwap_Close_Dist": 0.001398411644576436
  },
  {
    "Ticker": "ZZZ",
    "TradeDate": "2024-06-03 00:00:00",
    "Open": 25.35,
    "High": 25.35,
    "Low": 24.99,
    "Close": 25.0756,
    "Volume": 13928.0,
    "AdjFactor": 1.000475899691558,
    "PreClose": 24.8794,
    "Ret": 0.00788604226790035,
    "Ret_Intra": -0.01082445759368833,
    "Ret_Overnight": 0.01891524715226245,
    "High_Low_Ratio": 0.014405762304922076,
    "Upper_Shadow_Ratio": 0.0,
    "VWAP": 25.075082724698163,
    "Amount": 349245.752189596,
    "Vwap_Close_Dist": -2.062863109308921e-05
  }
]
```
</details>

### `ETFList`
- **vs_ashare**: A股有 `ETFList`（中文名/上市日等）；美股本表偏 ETP 产品元数据。股票 CS 不在此表。

- **model**: `STATIC`
- **purpose**: ETF/ETP 静态清单
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/ETFList/full.parquet`
- **coverage**: files=1, full → full
- **join**: read_full
- **sample_rows / n_fields**: 7190 / 14

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `name` | `string` | 证券英文名称 | string | 0.0 |  |
| 3 | `market` | `string` | 市场（通常 stocks） | string | 0.0 |  |
| 4 | `locale` | `string` | 地区（us） | string | 0.0 |  |
| 5 | `primary_exchange` | `string` | 主交易所 MIC（XNAS/XNYS/...） | string | 0.0 |  |
| 6 | `type` | `string` | 证券类型（CS/ETF/ETN/ETP/ADRC 等） | string | 0.0 |  |
| 7 | `active` | `bool` | 是否活跃 | bool | 0.0 |  |
| 8 | `currency_name` | `string` | 币种名（usd） | string | 0.0 |  |
| 9 | `cik` | `string` | SEC 发行人 CIK | string | 33.62 | 样本缺失 33.62%. |
| 10 | `composite_figi` | `string` | Composite FIGI | string | 11.46 |  |
| 11 | `share_class_figi` | `string` | Share Class FIGI | string | 11.46 |  |
| 12 | `last_updated_utc` | `string` | 上游更新时间（ISO） | string | 0.0 |  |
| 13 | `delisted_utc` | `string` | 退市时间（ISO，未退市为空） | string | 76.11 | 样本缺失 76.11%. |
| 14 | `TradeDate` | `string` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | string | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "market": [
    [
      "stocks",
      7190
    ]
  ],
  "locale": [
    [
      "us",
      7190
    ]
  ],
  "primary_exchange": [
    [
      "ARCX",
      3568
    ],
    [
      "BATS",
      1998
    ],
    [
      "XNAS",
      1545
    ],
    [
      "XNYS",
      79
    ]
  ],
  "type": [
    [
      "ETF",
      6940
    ],
    [
      "ETN",
      250
    ]
  ],
  "active": [
    [
      "True",
      5390
    ],
    [
      "False",
      1800
    ]
  ],
  "currency_name": [
    [
      "usd",
      7190
    ]
  ],
  "TradeDate": [
    [
      "2026-08-07",
      7190
    ]
  ]
}
```
</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "QQQ",
  "name": "Invesco QQQ Trust, Series 1",
  "market": "stocks",
  "locale": "us",
  "primary_exchange": "XNAS",
  "type": "ETF",
  "active": true,
  "currency_name": "usd",
  "cik": "0001067839",
  "composite_figi": "BBG000BSWKH7",
  "share_class_figi": "BBG001S9GN63",
  "last_updated_utc": "2026-08-06T06:11:44.67302022Z",
  "delisted_utc": null,
  "TradeDate": "2026-08-07"
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "PALU",
    "name": "Direxion Shares ETF Trust Direxion Daily PANW Bull 2X ETF",
    "market": "stocks",
    "locale": "us",
    "primary_exchange": "XNAS",
    "type": "ETF",
    "active": true,
    "currency_name": "usd",
    "cik": null,
    "composite_figi": "BBG01SQ60PF3",
    "share_class_figi": "BBG01SQ60Q98",
    "last_updated_utc": "2026-08-06T06:11:44.672925428Z",
    "delisted_utc": null,
    "TradeDate": "2026-08-07"
  },
  {
    "ticker": "BETZ",
    "name": "Roundhill Sports Betting & iGaming ETF",
    "market": "stocks",
    "locale": "us",
    "primary_exchange": "ARCX",
    "type": "ETF",
    "active": true,
    "currency_name": "usd",
    "cik": "0001776878",
    "composite_figi": "BBG00V7866V2",
    "share_class_figi": "BBG00V7867P7",
    "last_updated_utc": "2026-08-06T06:11:44.672057319Z",
    "delisted_utc": null,
    "TradeDate": "2026-08-07"
  },
  {
    "ticker": "FOVL",
    "name": "iShares Focused Value Factor ETF",
    "market": "stocks",
    "locale": "us",
    "primary_exchange": "ARCX",
    "type": "ETF",
    "active": false,
    "currency_name": "usd",
    "cik": "0001100663",
    "composite_figi": "BBG00NNQMF71",
    "share_class_figi": "BBG00NNQMFZ0",
    "last_updated_utc": "2025-08-20T06:05:24.370341917Z",
    "delisted_utc": "2025-08-19T00:00:00Z",
    "TradeDate": "2026-08-07"
  }
]
```
</details>

### `SecurityMaster`

- **note_type**: 当前 full 样本 `type` **几乎全为 CS**（与 StockList 一致）；更丰富的 ETP 元数据见 ETFList。
- **vs_snap**: 相对 DailySnap **多** `start_date`/`end_date`，**无** `approx_mode`/`ticker_cur`/`snap_date`。详见 Bridge §C23.3。

- **model**: `STATIC`
- **purpose**: 证券主数据当前快照
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/SecurityMaster/full.parquet`
- **coverage**: files=1, full → full
- **join**: read_full
- **sample_rows / n_fields**: 11882 / 16

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `name` | `string` | 证券英文名称 | string | 0.0 |  |
| 3 | `type` | `string` | 证券类型（CS/ETF/ETN/ETP/ADRC 等） | string | 0.0 |  |
| 4 | `market` | `string` | 市场（通常 stocks） | string | 0.0 |  |
| 5 | `locale` | `string` | 地区（us） | string | 0.0 |  |
| 6 | `is_adr_asset` | `int64` | 是否 ADR 资产 | string | 0.0 | 主数据属性 |
| 7 | `start_date` | `timestamp[ns]` | 生效/起始日 | timestamp[ns] | 100.0 | 样本缺失 100.0%. |
| 8 | `end_date` | `timestamp[ns]` | 上市/有效区间结束日（退市或失效） | timestamp[ns] | 100.0 | 样本缺失 100.0%. |
| 9 | `primary_exchange` | `string` | 主交易所 MIC（XNAS/XNYS/...） | string | 0.0 |  |
| 10 | `cik` | `string` | SEC 发行人 CIK | string | 2.54 |  |
| 11 | `composite_figi` | `string` | Composite FIGI | string | 40.63 | 样本缺失 40.63%. |
| 12 | `share_class_figi` | `string` | Share Class FIGI | string | 40.63 | 样本缺失 40.63%. |
| 13 | `security_id` | `string` | 内部证券 ID | string | 0.0 |  |
| 14 | `issuer_id` | `string` | 发行人 ID | string | 0.0 |  |
| 15 | `listing_id` | `string` | 上市 ID | string | 0.0 |  |
| 16 | `id_source` | `string` | ID 来源 | string | 0.0 |  |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "type": [
    [
      "CS",
      11882
    ]
  ],
  "market": [
    [
      "stocks",
      11882
    ]
  ],
  "locale": [
    [
      "us",
      11882
    ]
  ],
  "primary_exchange": [
    [
      "XNAS",
      6560
    ],
    [
      "XNYS",
      4432
    ],
    [
      "XASE",
      884
    ],
    [
      "BATS",
      4
    ],
    [
      "ARCX",
      2
    ]
  ],
  "id_source": [
    [
      "share_class_figi",
      7054
    ],
    [
      "ticker_provisional",
      4828
    ]
  ]
}
```
</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "NVDA",
  "name": "Nvidia Corp",
  "type": "CS",
  "market": "stocks",
  "locale": "us",
  "is_adr_asset": 0,
  "start_date": "NaT",
  "end_date": "NaT",
  "primary_exchange": "XNAS",
  "cik": "0001045810",
  "composite_figi": "BBG000BBJQV0",
  "share_class_figi": "BBG001S5TZJ6",
  "security_id": "BBG001S5TZJ6",
  "issuer_id": "0001045810",
  "listing_id": "BBG001S5TZJ6",
  "id_source": "share_class_figi"
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "A",
    "name": "Agilent Technologies Inc.",
    "type": "CS",
    "market": "stocks",
    "locale": "us",
    "is_adr_asset": 0,
    "start_date": "NaT",
    "end_date": "NaT",
    "primary_exchange": "XNYS",
    "cik": "0001090872",
    "composite_figi": "BBG000C2V3D6",
    "share_class_figi": "BBG001SCTQY4",
    "security_id": "BBG001SCTQY4",
    "issuer_id": "0001090872",
    "listing_id": "BBG001SCTQY4",
    "id_source": "share_class_figi"
  },
  {
    "ticker": "PGH",
    "name": "Pengrowth Energy Corporation",
    "type": "CS",
    "market": "stocks",
    "locale": "us",
    "is_adr_asset": 0,
    "start_date": "NaT",
    "end_date": "NaT",
    "primary_exchange": "XNYS",
    "cik": "0001088166",
    "composite_figi": "BBG005KYNVM3",
    "share_class_figi": "BBG001SL8LZ7",
    "security_id": "BBG001SL8LZ7",
    "issuer_id": "0001088166",
    "listing_id": "BBG001SL8LZ7",
    "id_source": "share_class_figi"
  },
  {
    "ticker": "FORA",
    "name": "Forian Inc. Common Stock",
    "type": "CS",
    "market": "stocks",
    "locale": "us",
    "is_adr_asset": 0,
    "start_date": "NaT",
    "end_date": "NaT",
    "primary_exchange": "XNAS",
    "cik": "0001829280",
    "composite_figi": "BBG00Y9CR7V0",
    "share_class_figi": "BBG00Y9CR7W9",
    "security_id": "BBG00Y9CR7W9",
    "issuer_id": "0001829280",
    "listing_id": "BBG00Y9CR7W9",
    "id_source": "share_class_figi"
  }
]
```
</details>

### `SecurityMasterDailySnap`
- **vs_ashare**: A股无 SecurityMaster；用 `StockList`/`StockStatus`。本表相对 STATIC master 的字段差集见 Bridge **§C23.3**。

- **vs_master**: 相对 STATIC master **多** `approx_mode`（样本=`stable_id_presence_v1`）、`ticker_cur`、`snap_date`；**无** listing `start_date`/`end_date`。type 样本全 CS。

- **model**: `D1`
- **purpose**: 证券主数据日快照（更 PIT）
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/SecurityMasterDailySnap/{YYYY-MM-DD}.parquet`
- **coverage**: files=5765, 2003-09-10 → 2026-08-07
- **join**: equi(date,Ticker)
- **sample_rows / n_fields**: 5284 / 17

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `security_id` | `string` | 内部证券 ID | string | 0.0 |  |
| 3 | `approx_mode` | `string` | ID 近似/稳定模式（样本恒 stable_id_presence_v1） | string | 0.0 |  |
| 4 | `ticker_cur` | `string` | 快照日当前 ticker | string | 0.0 |  |
| 5 | `name` | `string` | 证券英文名称 | string | 0.0 |  |
| 6 | `type` | `string` | 证券类型（CS/ETF/ETN/ETP/ADRC 等） | string | 0.0 |  |
| 7 | `market` | `string` | 市场（通常 stocks） | string | 0.0 |  |
| 8 | `locale` | `string` | 地区（us） | string | 0.0 |  |
| 9 | `is_adr_asset` | `int64` | 是否 ADR 资产 | string | 0.0 | 主数据属性 |
| 10 | `primary_exchange` | `string` | 主交易所 MIC（XNAS/XNYS/...） | string | 0.0 |  |
| 11 | `cik` | `string` | SEC 发行人 CIK | string | 0.15 |  |
| 12 | `composite_figi` | `string` | Composite FIGI | string | 15.59 |  |
| 13 | `share_class_figi` | `string` | Share Class FIGI | string | 15.59 |  |
| 14 | `issuer_id` | `string` | 发行人 ID | string | 0.0 |  |
| 15 | `listing_id` | `string` | 上市 ID | string | 0.0 |  |
| 16 | `id_source` | `string` | ID 来源 | string | 0.0 |  |
| 17 | `snap_date` | `string` | SecurityMaster 日快照对应交易日 | string | 0.0 |  |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "approx_mode": [
    [
      "stable_id_presence_v1",
      5284
    ]
  ],
  "type": [
    [
      "CS",
      5284
    ]
  ],
  "market": [
    [
      "stocks",
      5284
    ]
  ],
  "locale": [
    [
      "us",
      5284
    ]
  ],
  "primary_exchange": [
    [
      "XNAS",
      3267
    ],
    [
      "XNYS",
      1785
    ],
    [
      "XASE",
      229
    ],
    [
      "BATS",
      3
    ]
  ],
  "id_source": [
    [
      "share_class_figi",
      4460
    ],
    [
      "ticker_provisional",
      824
    ]
  ],
  "snap_date": [
    [
      "2024-06-03",
      5284
    ]
  ]
}
```
</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "AAPL",
  "security_id": "BBG001S5N8V8",
  "approx_mode": "stable_id_presence_v1",
  "ticker_cur": "AAPL",
  "name": "Apple Inc.",
  "type": "CS",
  "market": "stocks",
  "locale": "us",
  "is_adr_asset": 0,
  "primary_exchange": "XNAS",
  "cik": "0000320193",
  "composite_figi": "BBG000B9XRY4",
  "share_class_figi": "BBG001S5N8V8",
  "issuer_id": "0000320193",
  "listing_id": "BBG001S5N8V8",
  "id_source": "share_class_figi",
  "snap_date": "2024-06-03"
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "A",
    "security_id": "BBG001SCTQY4",
    "approx_mode": "stable_id_presence_v1",
    "ticker_cur": "A",
    "name": "Agilent Technologies Inc.",
    "type": "CS",
    "market": "stocks",
    "locale": "us",
    "is_adr_asset": 0,
    "primary_exchange": "XNYS",
    "cik": "0001090872",
    "composite_figi": "BBG000C2V3D6",
    "share_class_figi": "BBG001SCTQY4",
    "issuer_id": "0001090872",
    "listing_id": "BBG001SCTQY4",
    "id_source": "share_class_figi",
    "snap_date": "2024-06-03"
  },
  {
    "ticker": "FOXF",
    "security_id": "BBG004T7VS71",
    "approx_mode": "stable_id_presence_v1",
    "ticker_cur": "FOXF",
    "name": "Fox Factory Holding Corp. Common Stock",
    "type": "CS",
    "market": "stocks",
    "locale": "us",
    "is_adr_asset": 0,
    "primary_exchange": "XNAS",
    "cik": "0001424929",
    "composite_figi": "BBG004T7VS53",
    "share_class_figi": "BBG004T7VS71",
    "issuer_id": "0001424929",
    "listing_id": "BBG004T7VS71",
    "id_source": "share_class_figi",
    "snap_date": "2024-06-03"
  },
  {
    "ticker": "NEWTG",
    "security_id": "TICKER|NEWTG",
    "approx_mode": "stable_id_presence_v1",
    "ticker_cur": "NEWTG",
    "name": "NewtekOne, Inc. 8.50% Fixed Rate Senior Notes due 2029",
    "type": "CS",
    "market": "stocks",
    "locale": "us",
    "is_adr_asset": 0,
    "primary_exchange": "XNAS",
    "cik": "0001587987",
    "composite_figi": null,
    "share_class_figi": null,
    "issuer_id": "0001587987",
    "listing_id": "TICKER|NEWTG",
    "id_source": "ticker_provisional",
    "snap_date": "2024-06-03"
  }
]
```
</details>

### `TickerMap`
- **vs_ashare**: A股无对等图；跨市场代码映射需自建，禁止 Symbol↔Ticker 直接 equi。

- **model**: `STATIC`
- **purpose**: Ticker 映射表
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/TickerMap/full.parquet`
- **coverage**: files=1, full → full
- **join**: read_full
- **sample_rows / n_fields**: 14163 / 4

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `permanent_id` | `string` | 永久 ID | string | 0.0 |  |
| 3 | `effective_date` | `string` | 公司行动/调整生效日 | string | 0.0 |  |
| 4 | `event_type` | `string` | 事件类型 | string | 0.0 |  |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "event_type": [
    [
      "UNKNOWN",
      14163
    ]
  ]
}
```
</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "AAPL",
  "permanent_id": "AAPL",
  "effective_date": "2003-09-10",
  "event_type": "UNKNOWN"
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "AAOI",
    "permanent_id": "AAOI",
    "effective_date": "2013-09-26",
    "event_type": "UNKNOWN"
  },
  {
    "ticker": "KELYB",
    "permanent_id": "KELYB",
    "effective_date": "2004-11-08",
    "event_type": "UNKNOWN"
  },
  {
    "ticker": "ZYME",
    "permanent_id": "ZYME",
    "effective_date": "2022-10-13",
    "event_type": "UNKNOWN"
  }
]
```
</details>

### `TickerAlias`
- **vs_ashare**: 本表 **EMPTY**；A股无别名表。勿用。

- **model**: `EMPTY`
- **purpose**: Ticker 别名占位
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/TickerAlias/full.parquet`
- **coverage**: files=1, full → full
- **join**: DO_NOT_USE
- **sample_rows / n_fields**: 0 / 2
- **warning**: model=EMPTY — 禁止默认当作全市场历史面板。

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `null` | 美股代码 | string | nan | EMPTY 表勿用 |
| 2 | `ticker_normalized` | `null` | 归一化 ticker（EMPTY） | string | nan | EMPTY 表勿用 |

<details><summary>sample_row</summary>

```json
{}
```
</details>

### `TickerSharesSnapshot`
- **vs_ashare**: A股市值用 `StockValuationDaily.MarketCap` 或 Close×Capital；美股市值优先 `weighted_shares_outstanding×Close`（样本覆盖~42%）。

- **model**: `D1`
- **note_coverage**: 2024-06-03 与 StockDailyBar ticker 左连接，`weighted_shares_outstanding` 非空约 **42%**；做 size_neutralize 时要接受覆盖缺口或再合并其他源。选用顺序见 Bridge **§C23.2**。
- **purpose**: 股本日快照（市值中性推荐）
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/TickerSharesSnapshot/{YYYY-MM-DD}.parquet`
- **coverage**: files=4099, 2010-04-23 → 2026-08-07
- **join**: equi(date,ticker)
- **sample_rows / n_fields**: 6834 / 8

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `cik` | `string` | SEC 发行人 CIK | string | 0.0 |  |
| 3 | `share_class_figi` | `string` | Share Class FIGI | string | 24.44 |  |
| 4 | `share_class_shares_outstanding` | `double` | 该股份类别流通/在外股本 | 股 | 0.0 | 股本股数 |
| 5 | `weighted_shares_outstanding` | `double` | 加权平均股本 | 股 | 0.0 | 股本股数 |
| 6 | `TradeDate` | `string` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | string | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |
| 7 | `source` | `string` | 数据来源标记 | string | 0.0 |  |
| 8 | `error` | `null` | 错误信息（如有） | string | 100.0 | 样本缺失 100.0%. |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "TradeDate": [
    [
      "2024-06-03",
      6834
    ]
  ],
  "source": [
    [
      "shares_pit_derived",
      6834
    ]
  ]
}
```
</details>

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `share_class_shares_outstanding` | 0.0 | 0.00022436324668141596 | 35513004.875 | 177389029.60934612 | 266999724000.0 |
| `weighted_shares_outstanding` | 0.0 | 0.00022436324668141596 | 35937500.0 | 180052683.16447297 | 266999724000.0 |

</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "AAPL",
  "cik": "0000320193",
  "share_class_figi": "BBG001S5N8V8",
  "share_class_shares_outstanding": 16014696250.0,
  "weighted_shares_outstanding": 16097697500.0,
  "TradeDate": "2024-06-03",
  "source": "shares_pit_derived",
  "error": null
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "VATE",
    "cik": "0001006837",
    "share_class_figi": "BBG001S16908",
    "share_class_shares_outstanding": 7770000.0,
    "weighted_shares_outstanding": 7770000.0,
    "TradeDate": "2024-06-03",
    "source": "shares_pit_derived",
    "error": null
  },
  {
    "ticker": "ECOR",
    "cik": "0001560258",
    "share_class_figi": "BBG006R93QQ8",
    "share_class_shares_outstanding": 4616000.0,
    "weighted_shares_outstanding": 4616000.0,
    "TradeDate": "2024-06-03",
    "source": "shares_pit_derived",
    "error": null
  },
  {
    "ticker": "ZURA",
    "cik": "0001855644",
    "share_class_figi": null,
    "share_class_shares_outstanding": 1065134.5,
    "weighted_shares_outstanding": 1065134.5,
    "TradeDate": "2024-06-03",
    "source": "shares_pit_derived",
    "error": null
  }
]
```
</details>

### `StockIndicesComponents`
- **vs_ashare**: A股 `IndexConstituent` 有 `Weight`（%）；美股成分常 **无 Weight**，勿抄权重中性逻辑。

- **model**: `D1`
- **purpose**: 指数成分（通常无 Weight）
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockIndicesComponents/{YYYY-MM-DD}.parquet`
- **coverage**: files=6508, 1932-05-26 → 2026-08-03
- **join**: equi(date,Symbol)
- **sample_rows / n_fields**: 651 / 3

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `TradeDate` | `timestamp[ns]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | timestamp[ns] | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |
| 2 | `IndexName` | `string` | 指数名称 | string | 0.0 |  |
| 3 | `Symbol` | `string` | 代码别名列（美股语义同 Ticker，≠A股 Symbol） | string | 0.0 | 美股 Symbol≠A股 Symbol。 |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "IndexName": [
    [
      "S&P 500",
      520
    ],
    [
      "Nasdaq-100",
      104
    ],
    [
      "Dow Jones Industrial Average",
      27
    ]
  ]
}
```
</details>

<details><summary>sample_row</summary>

```json
{
  "TradeDate": "2024-06-03 00:00:00",
  "IndexName": "S&P 500",
  "Symbol": "AAPL"
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "TradeDate": "2024-06-03 00:00:00",
    "IndexName": "S&P 500",
    "Symbol": "A"
  },
  {
    "TradeDate": "2024-06-03 00:00:00",
    "IndexName": "S&P 500",
    "Symbol": "MNST"
  },
  {
    "TradeDate": "2024-06-03 00:00:00",
    "IndexName": "Dow Jones Industrial Average",
    "Symbol": "WMT"
  }
]
```
</details>

### `StockBalance`
- **note_timeframe**: 先 filter `timeframe`；金额 USD；PIT=`filing_date`；文件名=`period_end`。详见 CROSS_MARKET_BRIDGE §C17。

- **model**: `E2`
- **purpose**: 资产负债表
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockBalance/{period_end}.parquet`
- **coverage**: files=1941, 2009-04-30 → 2026-05-10
- **join**: asof(filing_date)
- **sample_rows / n_fields**: 5 / 38（**E2 事件文件正常偏少**，不是全市场截面；同日 Income 可达上百行视 timeframe）
- **vs_ashare**: A股 E1 文件名/PIT=`PubDate`；本表文件名=`period_end`，PIT=`filing_date`。

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `cik` | `string` | SEC 发行人 CIK | string | 0.0 |  |
| 3 | `period_end` | `string` | 报告期期末（常作文件名；≠PIT） | string | 0.0 | 文件名常用此字段；PIT 必须用 filing_date。 |
| 4 | `filing_date` | `timestamp[ns]` | 申报/公告可知日（财务 asof 键） | timestamp[ns] | 0.0 | 财务 asof 键。 |
| 5 | `fiscal_quarter` | `int64` | 财季编号（1–4；配合 fiscal_year） | 季度序号 1-4 | 0.0 | 元数据；非金额 |
| 6 | `fiscal_year` | `int64` | 财年（整数年） | 年 | 0.0 | 元数据；非金额 |
| 7 | `timeframe` | `string` | 报告时间框 | enum string | 0.0 | **必筛**：`quarterly` / `annual` / `trailing_twelve_months`。 |
| 8 | `cash_and_equivalents` | `double` | 现金及等价物 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 9 | `receivables` | `double` | 应收账款 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 10 | `inventories` | `double` | 存货（资产负债表，USD） | USD | 100.0 | 样本缺失 100.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 11 | `other_current_assets` | `double` | 其他流动资产 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 12 | `total_current_assets` | `double` | 流动资产合计 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 13 | `property_plant_equipment_net` | `double` | 固定资产净额 PP&E | USD | 60.0 | 样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 14 | `goodwill` | `double` | 商誉（资产负债表，USD） | USD | 60.0 | 样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 15 | `other_assets` | `double` | 其他资产 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 16 | `total_assets` | `double` | 总资产（USD） | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 17 | `accounts_payable` | `double` | 应付账款 | USD | 60.0 | 样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 18 | `debt_current` | `double` | 流动负债中的有息债 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 19 | `accrued_and_other_current_liabilities` | `double` | 应计及其他流动负债 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 20 | `total_current_liabilities` | `double` | 流动负债合计 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 21 | `long_term_debt_and_capital_lease_obligations` | `double` | 长期债务及融资租赁 | USD | 60.0 | 样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 22 | `other_noncurrent_liabilities` | `double` | 其他非流动负债 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 23 | `total_liabilities` | `double` | 总负债（USD） | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 24 | `commitments_and_contingencies` | `double` | 承诺与或有事项 | USD | 40.0 | 样本缺失 40.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 25 | `common_stock` | `double` | 普通股股本 | USD | 20.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 26 | `accumulated_other_comprehensive_income` | `double` | 累计其他综合收益 AOCI | USD | 60.0 | 样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 27 | `retained_earnings_deficit` | `double` | 留存收益/累计亏损 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 28 | `other_equity` | `double` | 其他权益 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 29 | `total_equity_attributable_to_parent` | `double` | 归母权益 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 30 | `noncontrolling_interest` | `double` | 少数股东损益/权益 | USD | 60.0 | 样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 31 | `total_equity` | `double` | 股东权益合计 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 32 | `total_liabilities_and_equity` | `double` | 负债和权益总计 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 33 | `additional_paid_in_capital` | `double` | 资本公积 APIC | USD | 80.0 | 样本缺失 80.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 34 | `treasury_stock` | `double` | 库存股（常为负值或绝对值约定，USD） | USD | 80.0 | 样本缺失 80.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 35 | `short_term_investments` | `double` | 短期投资 | USD | 100.0 | 样本缺失 100.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 36 | `intangible_assets_net` | `double` | 无形资产净额 | USD | 60.0 | 样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 37 | `preferred_stock` | `double` | 优先股权益（USD） | USD | 100.0 | 样本缺失 100.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 38 | `deferred_revenue_current` | `double` | 递延收入（流动） | USD | 60.0 | 样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "ticker": [
    [
      "CMCA",
      1
    ],
    [
      "CMCAU",
      1
    ],
    [
      "CMCAW",
      1
    ],
    [
      "KD",
      1
    ],
    [
      "GV",
      1
    ]
  ],
  "cik": [
    [
      "0001865248",
      3
    ],
    [
      "0001867072",
      1
    ],
    [
      "0001892274",
      1
    ]
  ],
  "period_end": [
    [
      "2024-03-31",
      5
    ]
  ],
  "timeframe": [
    [
      "quarterly",
      5
    ]
  ]
}
```
</details>

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `cash_and_equivalents` | 0.0 | 184733.0 | 184733.0 | 310835021.8 | 1553000000.0 |
| `accumulated_other_comprehensive_income` | 60.0 | -1145000000.0 | -572546983.5 | -572546983.5 | -93967.0 |
| `retained_earnings_deficit` | 0.0 | -2319000000.0 | -3133127.0 | -465652638.0 | 136191.0 |
| `deferred_revenue_current` | 60.0 | 968676.0 | 412984338.0 | 412984338.0 | 825000000.0 |

</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "CMCA",
  "cik": "0001865248",
  "period_end": "2024-03-31",
  "filing_date": "2026-02-05 00:00:00",
  "fiscal_quarter": 4,
  "fiscal_year": 2024,
  "timeframe": "quarterly",
  "cash_and_equivalents": 184733.0,
  "receivables": 0.0,
  "inventories": null,
  "other_current_assets": 0.0,
  "total_current_assets": 184733.0,
  "property_plant_equipment_net": null,
  "goodwill": null,
  "other_assets": 13483034.0,
  "total_assets": 13667767.0,
  "accounts_payable": null,
  "debt_current": 1360000.0,
  "accrued_and_other_current_liabilities": 797285.0,
  "total_current_liabilities": 2157285.0,
  "long_term_debt_and_capital_lease_obligations": null,
  "other_noncurrent_liabilities": 1160000.0,
  "total_liabilities": 3317285.0,
  "commitments_and_contingencies": 13483034.0,
  "common_stock": 575.0,
  "accumulated_other_comprehensive_income": null,
  "retained_earnings_deficit": -3133127.0,
  "other_equity": 0.0,
  "total_equity_attributable_to_parent": -3132552.0,
  "noncontrolling_interest": null,
  "total_equity": -3132552.0,
  "total_liabilities_and_equity": 13667767.0,
  "additional_paid_in_capital": null,
  "treasury_stock": null,
  "short_term_investments": null,
  "intangible_assets_net": null,
  "preferred_stock": null,
  "deferred_revenue_current": null
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "CMCA",
    "cik": "0001865248",
    "period_end": "2024-03-31",
    "filing_date": "2026-02-05 00:00:00",
    "fiscal_quarter": 4,
    "fiscal_year": 2024,
    "timeframe": "quarterly",
    "cash_and_equivalents": 184733.0,
    "receivables": 0.0,
    "inventories": null,
    "other_current_assets": 0.0,
    "total_current_assets": 184733.0,
    "property_plant_equipment_net": null,
    "goodwill": null,
    "other_assets": 13483034.0,
    "total_assets": 13667767.0,
    "accounts_payable": null,
    "debt_current": 1360000.0,
    "accrued_and_other_current_liabilities": 797285.0,
    "total_current_liabilities": 2157285.0,
    "long_term_debt_and_capital_lease_obligations": null,
    "other_noncurrent_liabilities": 1160000.0,
    "total_liabilities": 3317285.0,
    "commitments_and_contingencies": 13483034.0,
    "common_stock": 575.0,
    "accumulated_other_comprehensive_income": null,
    "retained_earnings_deficit": -3133127.0,
    "other_equity": 0.0,
    "total_equity_attributable_to_parent": -3132552.0,
    "noncontrolling_interest": null,
    "total_equity": -3132552.0,
    "total_liabilities_and_equity": 13667767.0,
    "additional_paid_in_capital": null,
    "treasury_stock": null,
    "short_term_investments": null,
    "intangible_assets_net": null,
    "preferred_stock": null,
    "deferred_revenue_current": null
  },
  {
    "ticker": "CMCAW",
    "cik": "0001865248",
    "period_end": "2024-03-31",
    "filing_date": "2026-02-05 00:00:00",
    "fiscal_quarter": 4,
    "fiscal_year": 2024,
    "timeframe": "quarterly",
    "cash_and_equivalents": 184733.0,
    "receivables": 0.0,
    "inventories": null,
    "other_current_assets": 0.0,
    "total_current_assets": 184733.0,
    "property_plant_equipment_net": null,
    "goodwill": null,
    "other_assets": 13483034.0,
    "total_assets": 13667767.0,
    "accounts_payable": null,
    "debt_current": 1360000.0,
    "accrued_and_other_current_liabilities": 797285.0,
    "total_current_liabilities": 2157285.0,
    "long_term_debt_and_capital_lease_obligations": null,
    "other_noncurrent_liabilities": 1160000.0,
    "total_liabilities": 3317285.0,
    "commitments_and_contingencies": 13483034.0,
    "common_stock": 575.0,
    "accumulated_other_comprehensive_income": null,
    "retained_earnings_deficit": -3133127.0,
    "other_equity": 0.0,
    "total_equity_attributable_to_parent": -3132552.0,
    "noncontrolling_interest": null,
    "total_equity": -3132552.0,
    "total_liabilities_and_equity": 13667767.0,
    "additional_paid_in_capital": null,
    "treasury_stock": null,
    "short_term_investments": null,
    "intangible_assets_net": null,
    "preferred_stock": null,
    "deferred_revenue_current": null
  },
  {
    "ticker": "GV",
    "cik": "0001892274",
    "period_end": "2024-03-31",
    "filing_date": "2026-01-28 00:00:00",
    "fiscal_quarter": 4,
    "fiscal_year": 2024,
    "timeframe": "quarterly",
    "cash_and_equivalents": 620910.0,
    "receivables": 20472.0,
    "inventories": null,
    "other_current_assets": 1375957.0,
    "total_current_assets": 2017339.0,
    "property_plant_equipment_net": 83581322.0,
    "goodwill": 950959.0,
    "other_assets": 299551.0,
    "total_assets": 87859701.0,
    "accounts_payable": 1187480.0,
    "debt_current": 62912133.0,
    "accrued_and_other_current_liabilities": 4612024.0,
    "total_current_liabilities": 69680313.0,
    "long_term_debt_and_capital_lease_obligations": 252476.0,
    "other_noncurrent_liabilities": 180978.0,
    "total_liabilities": 70113767.0,
    "commitments_and_contingencies": null,
    "common_stock": null,
    "accumulated_other_comprehensive_income": -93967.0,
    "retained_earnings_deficit": 136191.0,
    "other_equity": 0.0,
    "total_equity_attributable_to_parent": 17761979.0,
    "noncontrolling_interest": -16045.0,
    "total_equity": 17745934.0,
    "total_liabilities_and_equity": 87859701.0,
    "additional_paid_in_capital": 17719755.0,
    "treasury_stock": null,
    "short_term_investments": null,
    "intangible_assets_net": 933642.0,
    "preferred_stock": null,
    "deferred_revenue_current": 968676.0
  }
]
```
</details>

### `StockIncome`
- **vs_ashare**: A股 E1 文件名/PIT=`PubDate`、金额 CNY、无 `timeframe`；本表文件名=`period_end`、PIT=`filing_date`、须 filter `timeframe`、金额 USD。

- **note_timeframe**: 同一 `period_end` 文件内 `timeframe` 可混有 `quarterly` / `annual` / `trailing_twelve_months`。做截面或 asof **必须先选一种 timeframe**，否则把 TTM 与单季净利拼在一起。金额单位 USD。A股无此列。

- **model**: `E2`
- **purpose**: 利润表
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockIncome/{period_end}.parquet`
- **coverage**: files=1979, 2009-03-29 → 2026-05-10
- **join**: asof(filing_date)
- **sample_rows / n_fields**: 102 / 34

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `cik` | `string` | SEC 发行人 CIK | string | 0.0 |  |
| 3 | `period_end` | `string` | 报告期期末（常作文件名；≠PIT） | string | 0.0 | 文件名常用此字段；PIT 必须用 filing_date。 |
| 4 | `filing_date` | `timestamp[ns]` | 申报/公告可知日（财务 asof 键） | timestamp[ns] | 0.0 | 财务 asof 键。 |
| 5 | `fiscal_quarter` | `int64` | 财季编号（1–4；配合 fiscal_year） | 季度序号 1-4 | 0.0 | 元数据；非金额 |
| 6 | `fiscal_year` | `int64` | 财年（整数年） | 年 | 0.0 | 元数据；非金额 |
| 7 | `timeframe` | `string` | 报告时间框 | enum string | 0.0 | **必筛**：`quarterly` / `annual` / `trailing_twelve_months`。 |
| 8 | `revenue` | `double` | 营业收入 | USD | 23.53 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 9 | `cost_of_revenue` | `double` | 营业成本 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 10 | `gross_profit` | `double` | 毛利（利润表，USD） | USD | 23.53 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 11 | `selling_general_administrative` | `double` | 销售及管理费用 SG&A | USD | 14.71 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 12 | `research_development` | `double` | 研发费用 R&D | USD | 69.61 | 样本缺失 69.61%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 13 | `depreciation_depletion_amortization` | `double` | 折旧/折耗/摊销 | USD | 68.63 | 样本缺失 68.63%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 14 | `other_operating_expenses` | `double` | 其他营业费用 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 15 | `total_operating_expenses` | `double` | 营业费用合计 | USD | 4.9 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 16 | `operating_income` | `double` | 营业利润 | USD | 4.9 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 17 | `interest_expense` | `double` | 利息支出 | USD | 32.35 | 样本缺失 32.35%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 18 | `other_income_expense` | `double` | 其他收支 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 19 | `total_other_income_expense` | `double` | 其他收支合计 | USD | 6.86 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 20 | `income_before_income_taxes` | `double` | 税前利润 | USD | 4.9 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 21 | `income_taxes` | `double` | 所得税费用（USD） | USD | 15.69 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 22 | `consolidated_net_income_loss` | `double` | 合并净利润 | timestamp/date | 4.9 | 事件/报告日；非 PIT 时勿与交易日 equi 混用 |
| 23 | `noncontrolling_interest` | `double` | 少数股东损益/权益 | USD | 79.41 | 样本缺失 79.41%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 24 | `net_income_loss_attributable_common_shareholders` | `double` | 归母普通股净利润 | USD | 4.9 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 25 | `basic_earnings_per_share` | `double` | 基本每股收益 | USD/股 | 4.9 |  |
| 26 | `diluted_earnings_per_share` | `double` | 稀释每股收益 | USD/股 | 4.9 |  |
| 27 | `basic_shares_outstanding` | `double` | 基本股本 | 股 | 0.0 | 股本股数 |
| 28 | `diluted_shares_outstanding` | `double` | 稀释股本 | 股 | 0.0 | 股本股数 |
| 29 | `ebitda` | `double` | EBITDA | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 30 | `discontinued_operations` | `double` | 终止经营 | USD | 91.18 | 样本缺失 91.18%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 31 | `interest_income` | `double` | 利息收入 | USD | 40.2 | 样本缺失 40.2%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 32 | `preferred_stock_dividends_declared` | `double` | 优先股股息 | USD | 95.1 | 样本缺失 95.1%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 33 | `equity_in_affiliates` | `double` | 联营收益 | USD | 91.18 | 样本缺失 91.18%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 34 | `extraordinary_items` | `double` | 非经常项目 | USD | 96.08 | 样本缺失 96.08%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "period_end": [
    [
      "2024-03-31",
      102
    ]
  ],
  "timeframe": [
    [
      "trailing_twelve_months",
      55
    ],
    [
      "quarterly",
      36
    ],
    [
      "annual",
      11
    ]
  ]
}
```
</details>

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `revenue` | 23.53 | 7655001.0 | 629906000.0 | 5992875724.576923 | 308951000000.0 |
| `cost_of_revenue` | 0.0 | 0.0 | 117765000.0 | 3720759071.4117646 | 296123000000.0 |
| `operating_income` | 4.9 | -2713100000.0 | 18523000.0 | 165779118.13402063 | 5639803840.0 |
| `other_income_expense` | 0.0 | -159865000.0 | 0.0 | 16927363.598039217 | 463128000.0 |
| `total_other_income_expense` | 6.86 | -1788107200.0 | -2981000.0 | -17883400.36842105 | 583433000.0 |
| `income_before_income_taxes` | 4.9 | -2744600000.0 | 5574000.0 | 148264447.6701031 | 3851696640.0 |
| `income_taxes` | 15.69 | -206000000.0 | 2496500.0 | 44619818.95348837 | 1050118080.0 |
| `consolidated_net_income_loss` | 4.9 | -2903000000.0 | 2829161.0 | 110269884.90721649 | 3160000000.0 |
| `net_income_loss_attributable_common_shareholders` | 4.9 | -2903000000.0 | 2829161.0 | 102521326.06185567 | 3002000000.0 |
| `basic_earnings_per_share` | 4.9 | -47.2 | 0.24 | 0.6051546391752578 | 22.54 |
| `diluted_earnings_per_share` | 4.9 | -47.2 | 0.23 | 0.576082474226804 | 22.39 |
| `basic_shares_outstanding` | 0.0 | 483024.76 | 41784255.5 | 347289002.9432353 | 20182000000.0 |
| `diluted_shares_outstanding` | 0.0 | 483024.76 | 42823270.0 | 351941348.9873529 | 20359000000.0 |
| `interest_income` | 40.2 | -103600000.0 | 518293.0 | 36039597.59016393 | 828224000.0 |

</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "UHAE.Q",
  "cik": "0000004457",
  "period_end": "2024-03-31",
  "filing_date": "2026-05-27 00:00:00",
  "fiscal_quarter": 4,
  "fiscal_year": 2024,
  "timeframe": "quarterly",
  "revenue": 629906000.0,
  "cost_of_revenue": 278206000.0,
  "gross_profit": 351700000.0,
  "selling_general_administrative": null,
  "research_development": null,
  "depreciation_depletion_amortization": 663931000.0,
  "other_operating_expenses": 381904000.0,
  "total_operating_expenses": 857004000.0,
  "operating_income": -505304000.0,
  "interest_expense": -64184000.0,
  "other_income_expense": 463128000.0,
  "total_other_income_expense": 518965000.0,
  "income_before_income_taxes": 13661000.0,
  "income_taxes": 14524000.0,
  "consolidated_net_income_loss": -863000.0,
  "noncontrolling_interest": null,
  "net_income_loss_attributable_common_shareholders": -9687000.0,
  "basic_earnings_per_share": -0.05,
  "diluted_earnings_per_share": -0.05,
  "basic_shares_outstanding": 196077880.0,
  "diluted_shares_outstanding": 196077880.0,
  "ebitda": -500092000.0,
  "discontinued_operations": null,
  "interest_income": 120021000.0,
  "preferred_stock_dividends_declared": null,
  "equity_in_affiliates": null,
  "extraordinary_items": null
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "UHAE.Q",
    "cik": "0000004457",
    "period_end": "2024-03-31",
    "filing_date": "2026-05-27 00:00:00",
    "fiscal_quarter": 4,
    "fiscal_year": 2024,
    "timeframe": "quarterly",
    "revenue": 629906000.0,
    "cost_of_revenue": 278206000.0,
    "gross_profit": 351700000.0,
    "selling_general_administrative": null,
    "research_development": null,
    "depreciation_depletion_amortization": 663931000.0,
    "other_operating_expenses": 381904000.0,
    "total_operating_expenses": 857004000.0,
    "operating_income": -505304000.0,
    "interest_expense": -64184000.0,
    "other_income_expense": 463128000.0,
    "total_other_income_expense": 518965000.0,
    "income_before_income_taxes": 13661000.0,
    "income_taxes": 14524000.0,
    "consolidated_net_income_loss": -863000.0,
    "noncontrolling_interest": null,
    "net_income_loss_attributable_common_shareholders": -9687000.0,
    "basic_earnings_per_share": -0.05,
    "diluted_earnings_per_share": -0.05,
    "basic_shares_outstanding": 196077880.0,
    "diluted_shares_outstanding": 196077880.0,
    "ebitda": -500092000.0,
    "discontinued_operations": null,
    "interest_income": 120021000.0,
    "preferred_stock_dividends_declared": null,
    "equity_in_affiliates": null,
    "extraordinary_items": null
  },
  {
    "ticker": "HLNE",
    "cik": "0001433642",
    "period_end": "2024-03-31",
    "filing_date": "2026-05-21 00:00:00",
    "fiscal_quarter": 4,
    "fiscal_year": 2024,
    "timeframe": "trailing_twelve_months",
    "revenue": 553842000.0,
    "cost_of_revenue": 0.0,
    "gross_profit": 553842000.0,
    "selling_general_administrative": 308025000.0,
    "research_development": null,
    "depreciation_depletion_amortization": null,
    "other_operating_expenses": 0.0,
    "total_operating_expenses": 308025000.0,
    "operating_income": 245817000.0,
    "interest_expense": -11175000.0,
    "other_income_expense": 37009000.0,
    "total_other_income_expense": 35842000.0,
    "income_before_income_taxes": 281659000.0,
    "income_taxes": 54455000.0,
    "consolidated_net_income_loss": 227204000.0,
    "noncontrolling_interest": -86346000.0,
    "net_income_loss_attributable_common_shareholders": 140858000.0,
    "basic_earnings_per_share": 3.72,
    "diluted_earnings_per_share": 3.69,
    "basic_shares_outstanding": 37755052.0,
    "diluted_shares_outstanding": 45803110.0,
    "ebitda": 251591000.0,
    "discontinued_operations": null,
    "interest_income": 46499000.0,
    "preferred_stock_dividends_declared": null,
    "equity_in_affiliates": null,
    "extraordinary_items": null
  },
  {
    "ticker": "ARM",
    "cik": "0001973239",
    "period_end": "2024-03-31",
    "filing_date": "2026-05-26 00:00:00",
    "fiscal_quarter": 4,
    "fiscal_year": 2024,
    "timeframe": "quarterly",
    "revenue": null,
    "cost_of_revenue": 0.0,
    "gross_profit": null,
    "selling_general_administrative": null,
    "research_development": null,
    "depreciation_depletion_amortization": null,
    "other_operating_expenses": 0.0,
    "total_operating_expenses": null,
    "operating_income": null,
    "interest_expense": null,
    "other_income_expense": 0.0,
    "total_other_income_expense": null,
    "income_before_income_taxes": null,
    "income_taxes": null,
    "consolidated_net_income_loss": null,
    "noncontrolling_interest": null,
    "net_income_loss_attributable_common_shareholders": null,
    "basic_earnings_per_share": null,
    "diluted_earnings_per_share": null,
    "basic_shares_outstanding": 1027000000.0,
    "diluted_shares_outstanding": 1044000000.0,
    "ebitda": 0.0,
    "discontinued_operations": null,
    "interest_income": null,
    "preferred_stock_dividends_declared": null,
    "equity_in_affiliates": null,
    "extraordinary_items": null
  }
]
```
</details>

### `StockCashFlow`
- **vs_ashare**: 同 Income：A=`PubDate`/CNY；US=`filing_date`/`timeframe`/USD。A股现金流多为累计口径需差分；US 依 timeframe 选择。
- **note_timeframe**: 先 filter `timeframe`；金额 USD；PIT=`filing_date`；文件名=`period_end`。详见 CROSS_MARKET_BRIDGE §C17。

- **model**: `E2`
- **purpose**: 现金流量表
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockCashFlow/{period_end}.parquet`
- **coverage**: files=1953, 2009-03-29 → 2026-05-10
- **join**: asof(filing_date)
- **sample_rows / n_fields**: 101 / 32

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `cik` | `string` | SEC 发行人 CIK | string | 0.0 |  |
| 3 | `period_end` | `string` | 报告期期末（常作文件名；≠PIT） | string | 0.0 | 文件名常用此字段；PIT 必须用 filing_date。 |
| 4 | `filing_date` | `timestamp[ns]` | 申报/公告可知日（财务 asof 键） | timestamp[ns] | 0.0 | 财务 asof 键。 |
| 5 | `fiscal_quarter` | `int64` | 财季编号（1–4；配合 fiscal_year） | 季度序号 1-4 | 0.0 | 元数据；非金额 |
| 6 | `fiscal_year` | `int64` | 财年（整数年） | 年 | 0.0 | 元数据；非金额 |
| 7 | `timeframe` | `string` | 报告时间框 | enum string | 0.0 | **必筛**：`quarterly` / `annual` / `trailing_twelve_months`。 |
| 8 | `net_income` | `double` | 净利润（USD） | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 9 | `depreciation_depletion_and_amortization` | `double` | 折旧折耗摊销 | USD | 19.8 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 10 | `other_operating_activities` | `double` | 其他经营活动 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 11 | `change_in_other_operating_assets_and_liabilities_net` | `double` | 营运资本变动净额 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 12 | `cash_from_operating_activities_continuing_operations` | `double` | 持续经营经营现金流 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 13 | `net_cash_from_operating_activities` | `double` | 经营活动现金流量净额 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 14 | `purchase_of_property_plant_and_equipment` | `double` | 购建固定资产（CapEx 流出） | USD | 23.76 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 15 | `sale_of_property_plant_and_equipment` | `double` | 处置固定资产 | USD | 61.39 | 样本缺失 61.39%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 16 | `other_investing_activities` | `double` | 其他投资活动 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 17 | `net_cash_from_investing_activities_continuing_operations` | `double` | 持续经营投资现金流 | USD | 3.96 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 18 | `net_cash_from_investing_activities` | `double` | 投资活动现金流量净额 | USD | 3.96 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 19 | `long_term_debt_issuances_repayments` | `double` | 长期债务发行/偿还 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 20 | `dividends` | `double` | 支付股利 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 21 | `other_financing_activities` | `double` | 其他筹资活动 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 22 | `net_cash_from_financing_activities_continuing_operations` | `double` | 持续经营筹资现金流 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 23 | `net_cash_from_financing_activities` | `double` | 筹资活动现金流量净额 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 24 | `effect_of_currency_exchange_rate` | `double` | 汇率变动对现金影响 | USD | 44.55 | 样本缺失 44.55%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 25 | `change_in_cash_and_equivalents` | `double` | 现金及等价物净增加额 | USD | 0.0 | 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 26 | `short_term_debt_issuances_repayments` | `double` | 短期债务发行/偿还 | USD | 78.22 | 样本缺失 78.22%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 27 | `net_cash_from_operating_activities_discontinued_operations` | `double` | 终止经营-经营现金流 | USD | 93.07 | 样本缺失 93.07%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 28 | `net_cash_from_investing_activities_discontinued_operations` | `double` | 终止经营-投资现金流 | USD | 95.05 | 样本缺失 95.05%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 29 | `noncontrolling_interests` | `double` | 少数股东相关 | USD | 91.09 | 样本缺失 91.09%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 30 | `net_cash_from_financing_activities_discontinued_operations` | `double` | 终止经营-筹资现金流 | USD | 99.01 | 样本缺失 99.01%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 31 | `other_cash_adjustments` | `double` | 其他现金调整 | USD | 97.03 | 样本缺失 97.03%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |
| 32 | `income_loss_from_discontinued_operations` | `double` | 终止经营损益 | USD | 93.07 | 样本缺失 93.07%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "period_end": [
    [
      "2024-03-31",
      101
    ]
  ],
  "timeframe": [
    [
      "annual",
      75
    ],
    [
      "trailing_twelve_months",
      23
    ],
    [
      "quarterly",
      3
    ]
  ]
}
```
</details>

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `net_income` | 0.0 | -3744200000.0 | 51588000.0 | 318048618.8811881 | 9877984028.0 |
| `cash_from_operating_activities_continuing_operations` | 0.0 | -1064362000.0 | 67198000.0 | 639609782.3762376 | 25285296047.0 |
| `net_cash_from_operating_activities` | 0.0 | -1064362000.0 | 67198000.0 | 640081010.0990099 | 25285296047.0 |
| `net_cash_from_investing_activities_continuing_operations` | 3.96 | -9605231680.0 | -40670000.0 | -257804371.95876288 | 5203623000.0 |
| `net_cash_from_investing_activities` | 3.96 | -9476336320.0 | -40670000.0 | -256317481.64948454 | 5203623000.0 |
| `net_cash_from_financing_activities_continuing_operations` | 0.0 | -14989520876.0 | -29300000.0 | -261181327.2079208 | 1674572000.0 |
| `net_cash_from_financing_activities` | 0.0 | -14989520876.0 | -29300000.0 | -260947139.0891089 | 1674572000.0 |
| `change_in_cash_and_equivalents` | 0.0 | -820000000.0 | -596.0 | 137565715.7920792 | 7881393806.0 |
| `net_cash_from_operating_activities_discontinued_operations` | 93.07 | 190000.0 | 2386000.0 | 6799142.857142857 | 14346000.0 |
| `net_cash_from_investing_activities_discontinued_operations` | 95.05 | -156000.0 | 5163000.0 | 28845672.0 | 128895360.0 |
| `net_cash_from_financing_activities_discontinued_operations` | 99.01 | 23653000.0 | 23653000.0 | 23653000.0 | 23653000.0 |
| `other_cash_adjustments` | 97.03 | -3406000.0 | 1125000.0 | -385333.3333333333 | 1125000.0 |
| `income_loss_from_discontinued_operations` | 93.07 | -20655000.0 | -14604000.0 | 37835571.428571425 | 165553000.0 |

</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "UHAE.Q",
  "cik": "0000004457",
  "period_end": "2024-03-31",
  "filing_date": "2026-05-27 00:00:00",
  "fiscal_quarter": 4,
  "fiscal_year": 2024,
  "timeframe": "annual",
  "net_income": 628707000.0,
  "depreciation_depletion_and_amortization": 817889000.0,
  "other_operating_activities": 67804000.0,
  "change_in_other_operating_assets_and_liabilities_net": -61644000.0,
  "cash_from_operating_activities_continuing_operations": 1452756000.0,
  "net_cash_from_operating_activities": 1452756000.0,
  "purchase_of_property_plant_and_equipment": -2992898000.0,
  "sale_of_property_plant_and_equipment": 739178000.0,
  "other_investing_activities": 204364000.0,
  "net_cash_from_investing_activities_continuing_operations": -2046373000.0,
  "net_cash_from_investing_activities": -2046373000.0,
  "long_term_debt_issuances_repayments": 161347000.0,
  "dividends": 0.0,
  "other_financing_activities": -94814000.0,
  "net_cash_from_financing_activities_continuing_operations": 66533000.0,
  "net_cash_from_financing_activities": 66533000.0,
  "effect_of_currency_exchange_rate": 1104000.0,
  "change_in_cash_and_equivalents": -525980000.0,
  "short_term_debt_issuances_repayments": null,
  "net_cash_from_operating_activities_discontinued_operations": null,
  "net_cash_from_investing_activities_discontinued_operations": null,
  "noncontrolling_interests": null,
  "net_cash_from_financing_activities_discontinued_operations": null,
  "other_cash_adjustments": null,
  "income_loss_from_discontinued_operations": null
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "UHAE.Q",
    "cik": "0000004457",
    "period_end": "2024-03-31",
    "filing_date": "2026-05-27 00:00:00",
    "fiscal_quarter": 4,
    "fiscal_year": 2024,
    "timeframe": "annual",
    "net_income": 628707000.0,
    "depreciation_depletion_and_amortization": 817889000.0,
    "other_operating_activities": 67804000.0,
    "change_in_other_operating_assets_and_liabilities_net": -61644000.0,
    "cash_from_operating_activities_continuing_operations": 1452756000.0,
    "net_cash_from_operating_activities": 1452756000.0,
    "purchase_of_property_plant_and_equipment": -2992898000.0,
    "sale_of_property_plant_and_equipment": 739178000.0,
    "other_investing_activities": 204364000.0,
    "net_cash_from_investing_activities_continuing_operations": -2046373000.0,
    "net_cash_from_investing_activities": -2046373000.0,
    "long_term_debt_issuances_repayments": 161347000.0,
    "dividends": 0.0,
    "other_financing_activities": -94814000.0,
    "net_cash_from_financing_activities_continuing_operations": 66533000.0,
    "net_cash_from_financing_activities": 66533000.0,
    "effect_of_currency_exchange_rate": 1104000.0,
    "change_in_cash_and_equivalents": -525980000.0,
    "short_term_debt_issuances_repayments": null,
    "net_cash_from_operating_activities_discontinued_operations": null,
    "net_cash_from_investing_activities_discontinued_operations": null,
    "noncontrolling_interests": null,
    "net_cash_from_financing_activities_discontinued_operations": null,
    "other_cash_adjustments": null,
    "income_loss_from_discontinued_operations": null
  },
  {
    "ticker": "OESX",
    "cik": "0001409375",
    "period_end": "2024-03-31",
    "filing_date": "2026-06-04 00:00:00",
    "fiscal_quarter": 4,
    "fiscal_year": 2024,
    "timeframe": "annual",
    "net_income": -11671000.0,
    "depreciation_depletion_and_amortization": 2590000.0,
    "other_operating_activities": 1736000.0,
    "change_in_other_operating_assets_and_liabilities_net": -3309000.0,
    "cash_from_operating_activities_continuing_operations": -10092000.0,
    "net_cash_from_operating_activities": -10092000.0,
    "purchase_of_property_plant_and_equipment": -837000.0,
    "sale_of_property_plant_and_equipment": 106000.0,
    "other_investing_activities": 0.0,
    "net_cash_from_investing_activities_continuing_operations": -731000.0,
    "net_cash_from_investing_activities": -731000.0,
    "long_term_debt_issuances_repayments": -15000.0,
    "dividends": 0.0,
    "other_financing_activities": 1000.0,
    "net_cash_from_financing_activities_continuing_operations": -14000.0,
    "net_cash_from_financing_activities": -14000.0,
    "effect_of_currency_exchange_rate": null,
    "change_in_cash_and_equivalents": -10837000.0,
    "short_term_debt_issuances_repayments": null,
    "net_cash_from_operating_activities_discontinued_operations": null,
    "net_cash_from_investing_activities_discontinued_operations": null,
    "noncontrolling_interests": null,
    "net_cash_from_financing_activities_discontinued_operations": null,
    "other_cash_adjustments": null,
    "income_loss_from_discontinued_operations": null
  },
  {
    "ticker": "ARM",
    "cik": "0001973239",
    "period_end": "2024-03-31",
    "filing_date": "2026-05-26 00:00:00",
    "fiscal_quarter": 4,
    "fiscal_year": 2024,
    "timeframe": "annual",
    "net_income": 306000000.0,
    "depreciation_depletion_and_amortization": 162000000.0,
    "other_operating_activities": 817000000.0,
    "change_in_other_operating_assets_and_liabilities_net": -195000000.0,
    "cash_from_operating_activities_continuing_operations": 1090000000.0,
    "net_cash_from_operating_activities": 1090000000.0,
    "purchase_of_property_plant_and_equipment": -92000000.0,
    "sale_of_property_plant_and_equipment": null,
    "other_investing_activities": -424000000.0,
    "net_cash_from_investing_activities_continuing_operations": -516000000.0,
    "net_cash_from_investing_activities": -516000000.0,
    "long_term_debt_issuances_repayments": 0.0,
    "dividends": 0.0,
    "other_financing_activities": -208000000.0,
    "net_cash_from_financing_activities_continuing_operations": -208000000.0,
    "net_cash_from_financing_activities": -208000000.0,
    "effect_of_currency_exchange_rate": 3000000.0,
    "change_in_cash_and_equivalents": 369000000.0,
    "short_term_debt_issuances_repayments": null,
    "net_cash_from_operating_activities_discontinued_operations": null,
    "net_cash_from_investing_activities_discontinued_operations": null,
    "noncontrolling_interests": null,
    "net_cash_from_financing_activities_discontinued_operations": null,
    "other_cash_adjustments": null,
    "income_loss_from_discontinued_operations": null
  }
]
```
</details>

### `StockDividend`

- **note_div_units**: `cash_amount` 单位随 `currency`（USD 为主，可有 HKD/EUR/…）；`frequency` 常见 12=月、4=季、2=半年、1=年；存在未来 `ex_dividend_date` 文件，回测需截断。A股无 frequency/currency 列，现金红利为 CNY/股。
- **fx_rule**: 非 USD 分红进美元因子前须乘 FX；**COS 无 FX 表** → 默认 **剔除 currency≠USD** 或标记 unavailable，禁止静默当美元。
- **currency_mix（2024 五日抽查 1595 行）**: USD≈87.8%，非 USD≈**12.2%**（CAD/GBP/HKD/EUR/…）。
- **enums（2024-06-03）**: `distribution_type`∈{recurring,irregular,special}；`frequency` 分布例 12:422 / 4:111 / 2:34 / 1:17 / 0:3。

- **model**: `E2`
- **purpose**: 分红/分配事件
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockDividend/{ex_div_date}.parquet`
- **coverage**: files=6166, 2000-08-15 → 2027-11-12
- **join**: event(ex_dividend_date)
- **sample_rows / n_fields**: 587 / 13

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `id` | `string` | 记录 ID | string | 0.0 |  |
| 2 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 3 | `record_date` | `string` | 股权登记日 | string | 0.0 |  |
| 4 | `pay_date` | `timestamp[ns]` | 现金分红派息日 | timestamp[ns] | 0.0 |  |
| 5 | `declaration_date` | `string` | 分红宣告日（PIT 可用；缺则勿臆造） | string | 0.51 |  |
| 6 | `ex_dividend_date` | `string` | 除权除息日 | string | 0.0 | 可含未来日期，回测需截断。 |
| 7 | `frequency` | `int64` | 分红频率 | 次/年编码 | 0.0 | 12≈月 4≈季 2≈半年 1≈年 0=未知/特殊 |
| 8 | `cash_amount` | `double` | 每股现金分红（币种见 currency） | 随 currency | 0.0 | **不是恒为 USD**；非 USD 样本常见 split_adjusted 为空 |
| 9 | `currency` | `string` | 分红币种（ISO）；非 USD 禁止当美元金额 | string | 0.0 |  |
| 10 | `distribution_type` | `string` | 分配类型 | string | 0.0 | 实盘枚举：`recurring` / `irregular` / `special`（2024-06-03：584/2/1） |
| 11 | `historical_adjustment_factor` | `double` | 历史复权调整因子 | 无量纲 | 3.58 | 调整乘数 |
| 12 | `split_adjusted_cash_amount` | `double` | 拆股调整后每股分红 | 随 currency | 3.58 | 与 cash_amount 同币种；非 USD 时常为 null |
| 13 | `TradeDate` | `timestamp[ns]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | timestamp[ns] | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "record_date": [
    [
      "2024-06-03",
      587
    ]
  ],
  "ex_dividend_date": [
    [
      "2024-06-03",
      587
    ]
  ],
  "currency": [
    [
      "USD",
      566
    ],
    [
      "HKD",
      8
    ],
    [
      "EUR",
      6
    ],
    [
      "CAD",
      4
    ],
    [
      "SEK",
      1
    ],
    [
      "ILS",
      1
    ],
    [
      "PHP",
      1
    ]
  ],
  "distribution_type": [
    [
      "recurring",
      584
    ],
    [
      "irregular",
      2
    ],
    [
      "special",
      1
    ]
  ]
}
```
</details>

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `cash_amount` | 0.0 | 0.006 | 0.191764 | 0.2791125269812606 | 3.15 |
| `historical_adjustment_factor` | 3.58 | 0.000377 | 0.9111225000000001 | 0.8907222155477031 | 1.0 |
| `split_adjusted_cash_amount` | 3.58 | 0.0101858 | 0.19 | 0.2733275589010601 | 3.15 |

</details>

<details><summary>sample_row</summary>

```json
{
  "id": "E0025223f2cf9a51f78a188e5cdc787d3d9824ebc103695b18d1a70f340d1060d",
  "ticker": "VBFC",
  "record_date": "2024-06-03",
  "pay_date": "2024-06-10 00:00:00",
  "declaration_date": "2024-05-21",
  "ex_dividend_date": "2024-06-03",
  "frequency": 4,
  "cash_amount": 0.18,
  "currency": "USD",
  "distribution_type": "recurring",
  "historical_adjustment_factor": 0.987685,
  "split_adjusted_cash_amount": 0.18,
  "TradeDate": "2024-06-03 00:00:00"
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "id": "E0025223f2cf9a51f78a188e5cdc787d3d9824ebc103695b18d1a70f340d1060d",
    "ticker": "VBFC",
    "record_date": "2024-06-03",
    "pay_date": "2024-06-10 00:00:00",
    "declaration_date": "2024-05-21",
    "ex_dividend_date": "2024-06-03",
    "frequency": 4,
    "cash_amount": 0.18,
    "currency": "USD",
    "distribution_type": "recurring",
    "historical_adjustment_factor": 0.987685,
    "split_adjusted_cash_amount": 0.18,
    "TradeDate": "2024-06-03 00:00:00"
  },
  {
    "id": "E776386ce979054235546e804148497ab6207bc78fbadaa62a7f76cfe4e3acbab",
    "ticker": "SFpD",
    "record_date": "2024-06-03",
    "pay_date": "2024-06-17 00:00:00",
    "declaration_date": "2024-05-08",
    "ex_dividend_date": "2024-06-03",
    "frequency": 4,
    "cash_amount": 0.28125,
    "currency": "USD",
    "distribution_type": "recurring",
    "historical_adjustment_factor": 0.867592,
    "split_adjusted_cash_amount": 0.28125,
    "TradeDate": "2024-06-03 00:00:00"
  },
  {
    "id": "Eff5685192c1224f53f36f935599448e22b90db88604f2622bc41c0e12ed75de5",
    "ticker": "IGSB",
    "record_date": "2024-06-03",
    "pay_date": "2024-06-07 00:00:00",
    "declaration_date": "2024-05-31",
    "ex_dividend_date": "2024-06-03",
    "frequency": 12,
    "cash_amount": 0.174378,
    "currency": "USD",
    "distribution_type": "recurring",
    "historical_adjustment_factor": 0.912753,
    "split_adjusted_cash_amount": 0.174378,
    "TradeDate": "2024-06-03 00:00:00"
  }
]
```
</details>

### `StockCapitalDaily`

- **model**: `E2`（**同目录双 schema，禁止混读**）
- **purpose**: 拆分/调整事件 + 稀疏 PIT 股本（都不是 A 股那种全市场日频股本快照）
- **cos**:
  - 拆分事件：`.../StockCapitalDaily/{YYYY-MM-DD}.parquet`
  - PIT 股本：`.../StockCapitalDaily/shares_{YYYY-MM-DD}.parquet`
- **coverage**: files≈9824（其中 `shares_*`≈3966），跨度约 2000-03-31 → 2026-09-29（含未来日文件）
- **join**: 事件按 `execution_date`/`TradeDate`；PIT shares 勿当全市场面板
- **sample_rows / n_fields**: shares 样本≈51/4；split 样本≈1–3/8
- **vs_ashare**: A股 `StockCapitalDaily` = S1 日频 `TotalCapital/CirculatingCapital` 快照。美股日频股本请用 **`TickerSharesSnapshot`**（与 DailyBar 样本日交集约 42%）。见 CROSS_MARKET_BRIDGE §C13。
- **warning**: 用 glob 读整个目录会把两种 schema 拼炸；必须按文件名前缀分流。
- **adjustment_type 枚举（实盘）**: `forward_split` / `reverse_split` / `stock_dividend`（多文件汇总未见其它值）。
- **shares 选用**: 日频股本优先 `TickerSharesSnapshot`；见 Bridge §C23.2。
- **shares grain**: 同日同 `Ticker` **可多行**（例 2024-06-03 `A` 两行不同 pit 股本）；勿假定主键唯一。需业务规则去重（取 max/最新）或改用 SharesSnapshot。
- **split 样本日**: 2024-06-03 共 6 事件（reverse_split=5, stock_dividend=1）；`shares_` 同日仅 51 行。

#### A. `shares_YYYY-MM-DD.parquet`（PIT 股本，稀疏）

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `Ticker` | `string` | 美股代码（无交易所后缀，如 AAPL、BRK.B） | string | 0.0 |  |
| 2 | `TradeDate` | `timestamp[ns]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | timestamp[ns] | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |
| 3 | `pit_basic_shares_outstanding` | `double` | PIT 基本股本（股） | 股 | 27.45 | 单日文件标的很少；非全市场。 |
| 4 | `pit_diluted_shares_outstanding` | `double` | PIT 稀释股本（股） | 股 | 27.45 | 与 basic 同文件；同日同 Ticker 可多行，需去重。 |

#### B. `YYYY-MM-DD.parquet`（拆分/调整事件）

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `id` | `string` | 事件 ID | string | 0.0 |  |
| 2 | `execution_date` | `timestamp[ns]` | 拆分执行日 | timestamp[ns] | 0.0 |  |
| 3 | `split_from` | `double` | 拆分前股数基准（from） | 无量纲 | 0.0 | reverse 时常 > split_to |
| 4 | `split_to` | `double` | 拆分后股数基准（to）；比率=to/from | 无量纲 | 0.0 |  |
| 5 | `ticker` | `string` | 标的（小写列名） | string | 0.0 | 与 shares 文件的 `Ticker` 大小写不同 |
| 6 | `adjustment_type` | `string` | 如 forward_split / reverse_split / stock_dividend | string | 0.0 |  |
| 7 | `historical_adjustment_factor` | `double` | 历史调整因子 | 无量纲 | 0.0 | 与日线 `AdjFactor` 不是同一列 |
| 8 | `TradeDate` | `timestamp[ns]` | 常等于 execution_date | timestamp[ns] | 0.0 |  |


<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "Ticker": [
    [
      "BARK",
      6
    ],
    [
      "BARK.WS",
      6
    ],
    [
      "A",
      3
    ],
    [
      "GDST",
      3
    ],
    [
      "GDSTR",
      3
    ],
    [
      "GDSTU",
      3
    ],
    [
      "GDSTW",
      3
    ],
    [
      "HQY",
      3
    ],
    [
      "IMAQ",
      3
    ],
    [
      "IMAQR",
      3
    ],
    [
      "IMAQU",
      3
    ],
    [
      "IMAQW",
      3
    ],
    [
      "OPGN",
      3
    ],
    [
      "PATH",
      3
    ],
    [
      "SAIC",
      3
    ]
  ]
}
```
</details>

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `pit_basic_shares_outstanding` | 27.45 | 236933.42500000002 | 7810080.050000001 | 58668821.52432433 | 557878000.0 |
| `pit_diluted_shares_outstanding` | 27.45 | 236933.42500000002 | 7810080.050000001 | 58790063.37364864 | 557878000.0 |

</details>

<details><summary>sample_row</summary>

```json
{
  "Ticker": "A",
  "TradeDate": "2024-06-03 00:00:00",
  "pit_basic_shares_outstanding": 296000000.0,
  "pit_diluted_shares_outstanding": 297000000.0
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "Ticker": "A",
    "TradeDate": "2024-06-03 00:00:00",
    "pit_basic_shares_outstanding": 296000000.0,
    "pit_diluted_shares_outstanding": 297000000.0
  },
  {
    "Ticker": "GDSTW",
    "TradeDate": "2024-06-03 00:00:00",
    "pit_basic_shares_outstanding": 2822187.5,
    "pit_diluted_shares_outstanding": 2822187.5
  },
  {
    "Ticker": "SAIC",
    "TradeDate": "2024-06-03 00:00:00",
    "pit_basic_shares_outstanding": null,
    "pit_diluted_shares_outstanding": null
  }
]
```
</details>

### `StockValuationDaily`

- **model**: `X0`
- **purpose**: 估值指标（稀疏试点）
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockValuationDaily/{YYYY-MM-DD}.parquet`
- **coverage**: files=49, 2026-05-12 → 2026-07-27
- **join**: CHECK nrows; 禁止当全历史面板
- **sample_rows / n_fields**: ~4848 / 24（2026-07-27 单日；日历仅 ~49 个文件）
- **warning**: model=X0 — **文件日极少**，禁止当全历史面板；单日内 ticker 可以很多。
- **vs_ashare**: A股同名表是 **D1 全历史**；比率单位也不同（见 CROSS_MARKET_BRIDGE §C10）。市值默认仍用 SharesSnapshot×Close。

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `cik` | `string` | SEC 发行人 CIK | string | 0.0 |  |
| 3 | `price` | `double` | 价格（USD/股） | USD/股 | 0.0 |  |
| 4 | `average_volume` | `double` | 平均成交量 | 股 | 0.0 |  |
| 5 | `market_cap` | `double` | 总市值（USD） | USD | ~16 | **X0=日历稀疏**（非必然高缺失）。2026-07-27 样本 null≈16%。勿当全历史；默认市值源仍推 SharesSnapshot×Close。 |
| 6 | `earnings_per_share` | `double` | 每股收益 EPS（USD/股） | USD/股 | 80.0 | 样本缺失 80.0%. |
| 7 | `price_to_earnings` | `double` | 市盈率 PE | 倍数 | 80.0 | 样本缺失 80.0%. 表级 X0 稀疏。 |
| 8 | `price_to_book` | `double` | 市净率 PB | 倍数 | 80.0 | 样本缺失 80.0%. |
| 9 | `price_to_sales` | `double` | 市销率 PS | 倍数 | 80.0 | 样本缺失 80.0%. |
| 10 | `price_to_cash_flow` | `double` | 价格/现金流 | 倍数 | 80.0 | 样本缺失 80.0%. |
| 11 | `price_to_free_cash_flow` | `double` | 价格/自由现金流 | 倍数 | 80.0 | 样本缺失 80.0%. |
| 12 | `dividend_yield` | `double` | 股息率（小数，0.02=2%；禁止再/100） | **小数倾向** | ~17 | 实测 >0 中位≈0.023（≈2.3%）。A股 `DividendRatio` 是 **%**。禁止直接比。 |
| 13 | `return_on_assets` | `double` | ROA（小数；禁止再/100） | **小数倾向** | ~0 | 中位≈0.007；A股 Indicator.`Roa` 为 **%**。 |
| 14 | `return_on_equity` | `double` | ROE（小数；禁止再/100） | **小数倾向** | ~0 | 中位≈0.054；A股 Indicator.`Roe` 为 **%**。 |
| 15 | `debt_to_equity` | `double` | 资产负债/权益比 D/E | 倍数 | 0.0 |  |
| 16 | `current` | `double` | 流动比率 Current Ratio | 倍数 | 0.0 |  |
| 17 | `quick` | `double` | 速动比率 Quick Ratio | 倍数 | 0.0 |  |
| 18 | `cash` | `double` | 现金比率（非现金余额） | 比率 | ~0 | p50≈0.43，与 current/quick 相关≈0.5；**禁止当 USD 现金余额**。 |
| 19 | `ev_to_sales` | `double` | EV/Sales | 倍数 | 80.0 | 样本缺失 80.0%. |
| 20 | `ev_to_ebitda` | `double` | EV/EBITDA | 倍数 | 80.0 | 样本缺失 80.0%. |
| 21 | `enterprise_value` | `double` | 企业价值 EV（USD） | USD | 80.0 | 样本缺失 80.0%. |
| 22 | `free_cash_flow` | `double` | 自由现金流（USD） | USD | 20.0 |  |
| 23 | `TradeDate` | `timestamp[ns]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | timestamp[ns] | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 表级 X0 稀疏。 |
| 24 | `Volatility_20d` | `double` | 20 日波动率 | 不可用/全空 | 100 | **抽查多日 100% null → 当前当作不可用字段**，勿进因子。 |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "ticker": [
    [
      "AWINW",
      1
    ],
    [
      "CCLDO",
      1
    ],
    [
      "MPX",
      1
    ],
    [
      "PRZM",
      1
    ],
    [
      "TLOG",
      1
    ]
  ],
  "cik": [
    [
      "0001855631",
      1
    ],
    [
      "0001582982",
      1
    ],
    [
      "0001129155",
      1
    ],
    [
      "0001077370",
      1
    ],
    [
      "0001361248",
      1
    ]
  ]
}
```
</details>

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `price` | 0.0 | 0.0 | 0.0 | 7.1240000000000006 | 27.44 |
| `average_volume` | 0.0 | 500.0 | 23743.0 | 21014.4 | 47668.0 |
| `market_cap` | 80.0 | 280037376.0 | 280037376.0 | 280037376.0 | 280037376.0 |
| `earnings_per_share` | 80.0 | 0.19 | 0.19 | 0.19 | 0.19 |
| `price_to_earnings` | 80.0 | 42.4 | 42.4 | 42.4 | 42.4 |
| `price_to_book` | 80.0 | 2.37 | 2.37 | 2.37 | 2.37 |
| `price_to_sales` | 80.0 | 1.11 | 1.11 | 1.11 | 1.11 |
| `price_to_cash_flow` | 80.0 | 18.99 | 18.99 | 18.99 | 18.99 |
| `price_to_free_cash_flow` | 80.0 | 21.87 | 21.87 | 21.87 | 21.87 |
| `dividend_yield` | 80.0 | 0.0701 | 0.0701 | 0.0701 | 0.0701 |
| `return_on_assets` | 0.0 | -9.2193 | -0.7558 | -2.1665400000000004 | 0.0497 |
| `return_on_equity` | 0.0 | 0.0559 | 1.156 | 5.5157 | 24.9766 |
| `debt_to_equity` | 0.0 | -6.23 | -0.48 | -1.538 | 0.02 |
| `current` | 0.0 | 0.2 | 0.48 | 1.1620000000000001 | 3.84 |
| `quick` | 0.0 | 0.2 | 0.48 | 0.784 | 1.97 |
| `cash` | 0.0 | 0.03 | 0.14 | 0.45200000000000007 | 1.55 |
| `ev_to_sales` | 80.0 | 0.93 | 0.93 | 0.93 | 0.93 |
| `ev_to_ebitda` | 80.0 | 13.95 | 13.95 | 13.95 | 13.95 |
| `enterprise_value` | 80.0 | 234238376.0 | 234238376.0 | 234238376.0 | 234238376.0 |
| `free_cash_flow` | 20.0 | -27207942.0 | 4941092.5 | 1291810.75 | 22493000.0 |

</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "AWINW",
  "cik": "0001855631",
  "price": 0.0,
  "average_volume": 800.0,
  "market_cap": null,
  "earnings_per_share": null,
  "price_to_earnings": null,
  "price_to_book": null,
  "price_to_sales": null,
  "price_to_cash_flow": null,
  "price_to_free_cash_flow": null,
  "dividend_yield": null,
  "return_on_assets": -9.2193,
  "return_on_equity": 1.156,
  "debt_to_equity": -0.48,
  "current": 0.2,
  "quick": 0.2,
  "cash": 0.03,
  "ev_to_sales": null,
  "ev_to_ebitda": null,
  "enterprise_value": null,
  "free_cash_flow": -2922815.0,
  "TradeDate": "2026-05-14 00:00:00",
  "Volatility_20d": null
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "AWINW",
    "cik": "0001855631",
    "price": 0.0,
    "average_volume": 800.0,
    "market_cap": null,
    "earnings_per_share": null,
    "price_to_earnings": null,
    "price_to_book": null,
    "price_to_sales": null,
    "price_to_cash_flow": null,
    "price_to_free_cash_flow": null,
    "dividend_yield": null,
    "return_on_assets": -9.2193,
    "return_on_equity": 1.156,
    "debt_to_equity": -0.48,
    "current": 0.2,
    "quick": 0.2,
    "cash": 0.03,
    "ev_to_sales": null,
    "ev_to_ebitda": null,
    "enterprise_value": null,
    "free_cash_flow": -2922815.0,
    "TradeDate": "2026-05-14 00:00:00",
    "Volatility_20d": null
  },
  {
    "ticker": "MPX",
    "cik": "0001129155",
    "price": 8.18,
    "average_volume": 47668.0,
    "market_cap": 280037376.0,
    "earnings_per_share": 0.19,
    "price_to_earnings": 42.4,
    "price_to_book": 2.37,
    "price_to_sales": 1.11,
    "price_to_cash_flow": 18.99,
    "price_to_free_cash_flow": 21.87,
    "dividend_yield": 0.0701,
    "return_on_assets": 0.0443,
    "return_on_equity": 0.0559,
    "debt_to_equity": 0.0,
    "current": 3.84,
    "quick": 1.97,
    "cash": 1.55,
    "ev_to_sales": 0.93,
    "ev_to_ebitda": 13.95,
    "enterprise_value": 234238376.0,
    "free_cash_flow": 12805000.0,
    "TradeDate": "2026-05-14 00:00:00",
    "Volatility_20d": null
  },
  {
    "ticker": "TLOG",
    "cik": "0001361248",
    "price": 0.0,
    "average_volume": 500.0,
    "market_cap": null,
    "earnings_per_share": null,
    "price_to_earnings": null,
    "price_to_book": null,
    "price_to_sales": null,
    "price_to_cash_flow": null,
    "price_to_free_cash_flow": null,
    "dividend_yield": null,
    "return_on_assets": -0.9516,
    "return_on_equity": 1.3161,
    "debt_to_equity": -1.0,
    "current": 0.48,
    "quick": 0.48,
    "cash": 0.44,
    "ev_to_sales": null,
    "ev_to_ebitda": null,
    "enterprise_value": null,
    "free_cash_flow": -27207942.0,
    "TradeDate": "2026-05-14 00:00:00",
    "Volatility_20d": null
  }
]
```
</details>

### `StockIndicator`
- **vs_ashare**: A股 Indicator 随公告、比率多为**百分数**；美股 Indicator 为 **X0 稀疏**，ROE/ROA/`dividend_yield` 已是**小数**，且比 Valuation 少 `Volatility_20d`。

- **model**: `X0`
- **purpose**: 稀疏估值指标克隆（名字像财务，内容不是 A 股 Indicator）
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockIndicator/{YYYY-MM-DD}.parquet`
- **coverage**: files=49, 2026-05-12 → 2026-07-27
- **join**: CHECK nrows
- **sample_rows / n_fields**: ~4848 / 23（2026-07-27；无 `Volatility_20d`）
- **warning**: model=X0 — 禁止默认当作全市场历史面板。
- **vs_ashare_indicator**: **不是** A股那种财报比率 E1 表；内容≈Valuation 克隆。
- **diff_vs_valuation**: 比 Valuation **少且仅少** `Volatility_20d`（该字段当前全空不可用）。其余估值字段同名同义。

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `cik` | `string` | SEC 发行人 CIK | string | 0.0 |  |
| 3 | `TradeDate` | `timestamp[ns]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | timestamp[ns] | 0.0 | timestamp[ns] 需转 date 再与 A股 date32 对齐。 表级 X0 稀疏。 |
| 4 | `price` | `double` | 价格（USD/股） | USD/股 | 0.0 |  |
| 5 | `average_volume` | `double` | 平均成交量 | 股 | 0.0 |  |
| 6 | `market_cap` | `double` | 总市值（USD） | USD | 80.0 | 在 Valuation/Indicator 稀疏表中出现时不能当全市场日面板。 样本缺失 80.0%. 表级 X0 稀疏。 |
| 7 | `earnings_per_share` | `double` | 每股收益 EPS（USD/股） | USD/股 | 80.0 | 样本缺失 80.0%. |
| 8 | `price_to_earnings` | `double` | 市盈率 PE | 倍数 | 80.0 | 样本缺失 80.0%. 表级 X0 稀疏。 |
| 9 | `price_to_book` | `double` | 市净率 PB | 倍数 | 80.0 | 样本缺失 80.0%. |
| 10 | `price_to_sales` | `double` | 市销率 PS | 倍数 | 80.0 | 样本缺失 80.0%. |
| 11 | `price_to_cash_flow` | `double` | 价格/现金流 | 倍数 | 80.0 | 样本缺失 80.0%. |
| 12 | `price_to_free_cash_flow` | `double` | 价格/自由现金流 | 倍数 | 80.0 | 样本缺失 80.0%. |
| 13 | `dividend_yield` | `double` | 股息率（小数，0.02=2%；禁止再/100） | **小数** | ~17 | 与 Valuation 同口径；**禁止 /100**。A股 `DividendRatio` 才是 %。 |
| 14 | `return_on_assets` | `double` | ROA（小数；禁止再/100） | **小数** | ~0 | **禁止 /100**；A股 `Roa` 为 %。 |
| 15 | `return_on_equity` | `double` | ROE（小数；禁止再/100） | **小数** | ~0 | **禁止 /100**；A股 `Roe` 为 %。 |
| 16 | `debt_to_equity` | `double` | 资产负债/权益比 D/E | 倍数 | 0.0 |  |
| 17 | `current` | `double` | 流动比率 Current Ratio | 倍数 | 0.0 |  |
| 18 | `quick` | `double` | 速动比率 Quick Ratio | 倍数 | 0.0 |  |
| 19 | `cash` | `double` | 现金比率（非现金余额） | 比率 | 0.0 | 与 Valuation 同字段；p50 量级约 0.1–0.5；**禁止当 USD 现金余额**。 |
| 20 | `ev_to_sales` | `double` | EV/Sales | 倍数 | 80.0 | 样本缺失 80.0%. |
| 21 | `ev_to_ebitda` | `double` | EV/EBITDA | 倍数 | 80.0 | 样本缺失 80.0%. |
| 22 | `enterprise_value` | `double` | 企业价值 EV（USD） | USD | 80.0 | 样本缺失 80.0%. |
| 23 | `free_cash_flow` | `double` | 自由现金流（USD） | USD | 20.0 |  |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "ticker": [
    [
      "AWINW",
      1
    ],
    [
      "CCLDO",
      1
    ],
    [
      "MPX",
      1
    ],
    [
      "PRZM",
      1
    ],
    [
      "TLOG",
      1
    ]
  ],
  "cik": [
    [
      "0001855631",
      1
    ],
    [
      "0001582982",
      1
    ],
    [
      "0001129155",
      1
    ],
    [
      "0001077370",
      1
    ],
    [
      "0001361248",
      1
    ]
  ]
}
```
</details>

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `price` | 0.0 | 0.0 | 0.0 | 7.1240000000000006 | 27.44 |
| `average_volume` | 0.0 | 500.0 | 23743.0 | 21014.4 | 47668.0 |
| `market_cap` | 80.0 | 280037376.0 | 280037376.0 | 280037376.0 | 280037376.0 |
| `earnings_per_share` | 80.0 | 0.19 | 0.19 | 0.19 | 0.19 |
| `price_to_earnings` | 80.0 | 42.4 | 42.4 | 42.4 | 42.4 |
| `price_to_book` | 80.0 | 2.37 | 2.37 | 2.37 | 2.37 |
| `price_to_sales` | 80.0 | 1.11 | 1.11 | 1.11 | 1.11 |
| `price_to_cash_flow` | 80.0 | 18.99 | 18.99 | 18.99 | 18.99 |
| `price_to_free_cash_flow` | 80.0 | 21.87 | 21.87 | 21.87 | 21.87 |
| `dividend_yield` | 80.0 | 0.0701 | 0.0701 | 0.0701 | 0.0701 |
| `return_on_assets` | 0.0 | -9.2193 | -0.7558 | -2.1665400000000004 | 0.0497 |
| `return_on_equity` | 0.0 | 0.0559 | 1.156 | 5.5157 | 24.9766 |
| `debt_to_equity` | 0.0 | -6.23 | -0.48 | -1.538 | 0.02 |
| `current` | 0.0 | 0.2 | 0.48 | 1.1620000000000001 | 3.84 |
| `quick` | 0.0 | 0.2 | 0.48 | 0.784 | 1.97 |
| `cash` | 0.0 | 0.03 | 0.14 | 0.45200000000000007 | 1.55 |
| `ev_to_sales` | 80.0 | 0.93 | 0.93 | 0.93 | 0.93 |
| `ev_to_ebitda` | 80.0 | 13.95 | 13.95 | 13.95 | 13.95 |
| `enterprise_value` | 80.0 | 234238376.0 | 234238376.0 | 234238376.0 | 234238376.0 |
| `free_cash_flow` | 20.0 | -27207942.0 | 4941092.5 | 1291810.75 | 22493000.0 |

</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "AWINW",
  "cik": "0001855631",
  "TradeDate": "2026-05-14 00:00:00",
  "price": 0.0,
  "average_volume": 800.0,
  "market_cap": null,
  "earnings_per_share": null,
  "price_to_earnings": null,
  "price_to_book": null,
  "price_to_sales": null,
  "price_to_cash_flow": null,
  "price_to_free_cash_flow": null,
  "dividend_yield": null,
  "return_on_assets": -9.2193,
  "return_on_equity": 1.156,
  "debt_to_equity": -0.48,
  "current": 0.2,
  "quick": 0.2,
  "cash": 0.03,
  "ev_to_sales": null,
  "ev_to_ebitda": null,
  "enterprise_value": null,
  "free_cash_flow": -2922815.0
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "AWINW",
    "cik": "0001855631",
    "TradeDate": "2026-05-14 00:00:00",
    "price": 0.0,
    "average_volume": 800.0,
    "market_cap": null,
    "earnings_per_share": null,
    "price_to_earnings": null,
    "price_to_book": null,
    "price_to_sales": null,
    "price_to_cash_flow": null,
    "price_to_free_cash_flow": null,
    "dividend_yield": null,
    "return_on_assets": -9.2193,
    "return_on_equity": 1.156,
    "debt_to_equity": -0.48,
    "current": 0.2,
    "quick": 0.2,
    "cash": 0.03,
    "ev_to_sales": null,
    "ev_to_ebitda": null,
    "enterprise_value": null,
    "free_cash_flow": -2922815.0
  },
  {
    "ticker": "MPX",
    "cik": "0001129155",
    "TradeDate": "2026-05-14 00:00:00",
    "price": 8.18,
    "average_volume": 47668.0,
    "market_cap": 280037376.0,
    "earnings_per_share": 0.19,
    "price_to_earnings": 42.4,
    "price_to_book": 2.37,
    "price_to_sales": 1.11,
    "price_to_cash_flow": 18.99,
    "price_to_free_cash_flow": 21.87,
    "dividend_yield": 0.0701,
    "return_on_assets": 0.0443,
    "return_on_equity": 0.0559,
    "debt_to_equity": 0.0,
    "current": 3.84,
    "quick": 1.97,
    "cash": 1.55,
    "ev_to_sales": 0.93,
    "ev_to_ebitda": 13.95,
    "enterprise_value": 234238376.0,
    "free_cash_flow": 12805000.0
  },
  {
    "ticker": "TLOG",
    "cik": "0001361248",
    "TradeDate": "2026-05-14 00:00:00",
    "price": 0.0,
    "average_volume": 500.0,
    "market_cap": null,
    "earnings_per_share": null,
    "price_to_earnings": null,
    "price_to_book": null,
    "price_to_sales": null,
    "price_to_cash_flow": null,
    "price_to_free_cash_flow": null,
    "dividend_yield": null,
    "return_on_assets": -0.9516,
    "return_on_equity": 1.3161,
    "debt_to_equity": -1.0,
    "current": 0.48,
    "quick": 0.48,
    "cash": 0.44,
    "ev_to_sales": null,
    "ev_to_ebitda": null,
    "enterprise_value": null,
    "free_cash_flow": -27207942.0
  }
]
```
</details>

### `StockStatus`

- **model**: `EMPTY`
- **purpose**: 状态占位
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockStatus/full.parquet`
- **coverage**: files=1, full → full
- **join**: DO_NOT_USE
- **sample_rows / n_fields**: 0 / 4
- **warning**: model=EMPTY — 禁止默认当作全市场历史面板。
- **vs_ashare**: A股 S1 可用；本表 **EMPTY**。

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `TradeDate` | `date32[day]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | date32[day] | nan | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |
| 2 | `Symbol` | `string` | 代码别名列（美股语义同 Ticker，≠A股 Symbol） | string | nan | 美股 Symbol≠A股 Symbol。 |
| 3 | `is_active` | `bool` | 是否活跃（EMPTY 表） | bool | nan |  |
| 4 | `UpdateTime` | `timestamp[ns, tz=UTC]` | 清洗管线产出时间（UTC） | timestamp[ns, tz=UTC] | nan |  |

<details><summary>sample_row</summary>

```json
{}
```
</details>

### `StockIndustry`

- **model**: `EMPTY`
- **purpose**: 行业占位
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/StockIndustry/full.parquet`
- **coverage**: files=1, full → full
- **join**: DO_NOT_USE
- **sample_rows / n_fields**: 0 / 6
- **warning**: model=EMPTY — 禁止默认当作全市场历史面板。
- **vs_ashare**: A股有完整多源行业；本表 **EMPTY** → `industry_neutralize` fail-closed。

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `TradeDate` | `date32[day]` | 交易日（常为 timestamp[ns] 当日0点，建议归一成 date） | date32[day] | nan | timestamp[ns] 需转 date 再与 A股 date32 对齐。 |
| 2 | `Symbol` | `string` | 代码别名列（美股语义同 Ticker，≠A股 Symbol） | string | nan | 美股 Symbol≠A股 Symbol。 |
| 3 | `IndustrySource` | `string` | 行业源（EMPTY） | string | nan |  |
| 4 | `IndustryCode` | `string` | 行业代码（EMPTY） | string | nan |  |
| 5 | `IndustryName` | `string` | 行业名（EMPTY） | string | nan |  |
| 6 | `UpdateTime` | `timestamp[ns, tz=UTC]` | 清洗管线产出时间（UTC） | timestamp[ns, tz=UTC] | nan |  |

<details><summary>sample_row</summary>

```json
{}
```
</details>

### `FactNews`
- **vs_ashare**: A股 clean **无**对等新闻表；PIT 用 `published_utc`。字段差集见 Bridge §C23.4。

- **note_news**: clean 层英文新闻；PIT 对齐用 **`published_utc`**（不是 TradeDate 文件名 alone）。`tickers` 为数组可多标的。更完整 raw 规则见 两份核心字典 §4 NEWS。A股 clean **无**对等新闻表。

- **model**: `RAW_EVENT`
- **purpose**: 新闻/事实摘要
- **cos**: `cos://qs-cold/clean_data/us_stock/massive_data/FactNews/{YYYY-MM-DD}.parquet`
- **coverage**: files=2198, 2016-06-22 → 2026-06-12
- **join**: explode(tickers); PIT=published_utc
- **sample_rows / n_fields**: 800 / 16

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `id` | `string` | 记录 ID | string | 0.0 |  |
| 2 | `title` | `string` | 新闻标题 | string | 0.0 |  |
| 3 | `author` | `string` | 新闻作者/来源署名 | string | 0.0 |  |
| 4 | `published_utc` | `timestamp[ns, tz=UTC]` | 发布时间 UTC（新闻 PIT） | timestamp[ns, tz=UTC] | 0.0 | **唯一推荐 PIT 键**；回测用 published_utc≤信号时点。 |
| 5 | `article_url` | `string` | 文章 URL | string | 0.0 |  |
| 6 | `tickers` | `list<element: string>` | 关联 ticker 列表（需 explode） | string | 0.0 | 数组/列表，需 explode 后对齐。 |
| 7 | `image_url` | `string` | 配图 URL | string | 0.0 |  |
| 8 | `description` | `string` | 摘要/描述 | string | 0.5 |  |
| 9 | `keywords` | `list<element: string>` | 新闻关键词（常大量缺失） | string | 51.12 | 样本缺失 51.12%. |
| 10 | `insights` | `null` | 洞察/标签结构 | JSON/null | 100 | 样本常全空；不要依赖。 |
| 11 | `publisher.name` | `string` | 出版方名称 | string | 0.0 |  |
| 12 | `publisher.homepage_url` | `string` | 出版方主页 | string | 0.0 |  |
| 13 | `publisher.logo_url` | `string` | 出版方 logo | string | 0.0 |  |
| 14 | `publisher.favicon_url` | `string` | 出版方 favicon | string | 0.0 |  |
| 15 | `amp_url` | `string` | AMP URL | string | 10.88 |  |
| 16 | `TradeDate` | `timestamp[ns]` | 文件分区日（常为日历日 0 点） | timestamp[ns] | 0.0 | **不是**新闻可知时点；PIT 用 published_utc。同一 TradeDate 文件可含多日 published_utc。 |

<details><summary>enums（样本文件 value_counts）</summary>

```json
{
  "publisher.name": [
    [
      "Zacks Investment Research",
      398
    ],
    [
      "GlobeNewswire Inc.",
      261
    ],
    [
      "The Motley Fool",
      68
    ],
    [
      "Benzinga",
      62
    ],
    [
      "Seeking Alpha",
      11
    ]
  ],
  "publisher.homepage_url": [
    [
      "https://www.zacks.com/",
      398
    ],
    [
      "https://www.globenewswire.com",
      261
    ],
    [
      "https://www.fool.com/",
      68
    ],
    [
      "https://www.benzinga.com/",
      62
    ],
    [
      "https://seekingalpha.com/",
      11
    ]
  ],
  "publisher.logo_url": [
    [
      "https://s3.massive.com/public/assets/news/logos/zacks.png",
      398
    ],
    [
      "https://s3.massive.com/public/assets/news/logos/globenewswire.svg",
      261
    ],
    [
      "https://s3.massive.com/public/assets/news/logos/themotleyfool.svg",
      68
    ],
    [
      "https://s3.massive.com/public/assets/news/logos/benzinga.svg",
      62
    ],
    [
      "https://s3.massive.com/public/assets/news/logos/seekingalpha.svg",
      11
    ]
  ],
  "publisher.favicon_url": [
    [
      "https://s3.massive.com/public/assets/news/favicons/zacks.ico",
      398
    ],
    [
      "https://s3.massive.com/public/assets/news/favicons/globenewswire.ico",
      261
    ],
    [
      "https://s3.massive.com/public/assets/news/favicons/themotleyfool.ico",
      68
    ],
    [
      "https://s3.massive.com/public/assets/news/favicons/benzinga.ico",
      62
    ],
    [
      "https://s3.massive.com/public/assets/news/favicons/seekingalpha.ico",
      11
    ]
  ]
}
```
</details>

<details><summary>sample_row</summary>

```json
{
  "id": "mcHyP7hjhzKY2mg3bmUCFYM7CSdmBhsfdAXH3ecL2GY",
  "title": "ADC Therapeutics gewährt neuen Mitarbeitern Zuteilungen im Rahmen des Anreizplans",
  "author": "ADC Therapeutics SA",
  "published_utc": "2024-06-03 22:39:00+00:00",
  "article_url": "https://www.globenewswire.com/news-release/2024/06/03/2892674/0/de/ADC-Therapeutics-gew%C3%A4hrt-neuen-Mitarbeitern-Zuteilungen-im-Rahmen-des-Anreizplans.html",
  "tickers": [
    "ADCT"
  ],
  "image_url": "https://ml-eu.globenewswire.com/Resource/Download/94cb96a4-3fa4-43b9-8c66-abbceccced56",
  "description": "LAUSANNE, Schweiz, June  04, 2024  (GLOBE NEWSWIRE) -- ADC Therapeutics SA (NYSE: ADCT) hat heute die Zuteilung von Optionen zum Kauf von insgesamt 109.800 Stammaktien des Unternehmens an drei neue Mitarbeiter am 3. Juni 2024 (jeweils eine „Zuteilung“) bekanntgegeben. Mit den Zuteilungen wurde den Mitarbeitern ein wesentlicher Anreiz für ihre Beschäftigung angeboten. Die Zuteilungen wurden vom Vergütungsausschuss des Unternehmensvorstands im Rahmen des Anreizplans des Unternehmens genehmigt. Damit sollen die Empfänger zu Spitzenleistungen motiviert und belohnt werden und einen wesentlichen Beitrag zum Erfolg des Unternehmens leisten. Die Bewilligung der Zuteilungen erfolgte unter Berufung auf die Ausnahmeregelung für Beschäftigungsanreize gemäß Regel 303A.08 des NYSE Listed Company Manual. Das Unternehmen veröffentlicht diese Pressemitteilung in Übereinstimmung mit Regel 303A.08. Die Zuteilungen werden zu 25 % am ersten Jahrestag des Zuteilungsdatums und zu 1/48 der Gesamtzahl der den Zuteilungen unterliegenden Aktien an jedem darauffolgenden monatlichen Jahrestag des Zuteilungsdatums unverfallbar und können ausgeübt werden, sodass die gesamte Zuteilung am vierten Jahrestag des Zuteilungsdatums unverfallbar wird, sofern das Beschäftigungsverhältnis mit dem Unternehmen fortgesetzt wird.",
  "keywords": [
    "Insider's Buy/Sell"
  ],
  "insights": null,
  "publisher.name": "GlobeNewswire Inc.",
  "publisher.homepage_url": "https://www.globenewswire.com",
  "publisher.logo_url": "https://s3.massive.com/public/assets/news/logos/globenewswire.svg",
  "publisher.favicon_url": "https://s3.massive.com/public/assets/news/favicons/globenewswire.ico",
  "amp_url": "https://www.globenewswire.com/news-release/2024/06/03/2892674/0/de/ADC-Therapeutics-gew%C3%A4hrt-neuen-Mitarbeitern-Zuteilungen-im-Rahmen-des-Anreizplans.html",
  "TradeDate": "2024-06-03 00:00:00"
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "id": "mcHyP7hjhzKY2mg3bmUCFYM7CSdmBhsfdAXH3ecL2GY",
    "title": "ADC Therapeutics gewährt neuen Mitarbeitern Zuteilungen im Rahmen des Anreizplans",
    "author": "ADC Therapeutics SA",
    "published_utc": "2024-06-03 22:39:00+00:00",
    "article_url": "https://www.globenewswire.com/news-release/2024/06/03/2892674/0/de/ADC-Therapeutics-gew%C3%A4hrt-neuen-Mitarbeitern-Zuteilungen-im-Rahmen-des-Anreizplans.html",
    "tickers": [
      "ADCT"
    ],
    "image_url": "https://ml-eu.globenewswire.com/Resource/Download/94cb96a4-3fa4-43b9-8c66-abbceccced56",
    "description": "LAUSANNE, Schweiz, June  04, 2024  (GLOBE NEWSWIRE) -- ADC Therapeutics SA (NYSE: ADCT) hat heute die Zuteilung von Optionen zum Kauf von insgesamt 109.800 Stammaktien des Unternehmens an drei neue Mitarbeiter am 3. Juni 2024 (jeweils eine „Zuteilung“) bekanntgegeben. Mit den Zuteilungen wurde den Mitarbeitern ein wesentlicher Anreiz für ihre Beschäftigung angeboten. Die Zuteilungen wurden vom Vergütungsausschuss des Unternehmensvorstands im Rahmen des Anreizplans des Unternehmens genehmigt. Damit sollen die Empfänger zu Spitzenleistungen motiviert und belohnt werden und einen wesentlichen Beitrag zum Erfolg des Unternehmens leisten. Die Bewilligung der Zuteilungen erfolgte unter Berufung auf die Ausnahmeregelung für Beschäftigungsanreize gemäß Regel 303A.08 des NYSE Listed Company Manual. Das Unternehmen veröffentlicht diese Pressemitteilung in Übereinstimmung mit Regel 303A.08. Die Zuteilungen werden zu 25 % am ersten Jahrestag des Zuteilungsdatums und zu 1/48 der Gesamtzahl der den Zuteilungen unterliegenden Aktien an jedem darauffolgenden monatlichen Jahrestag des Zuteilungsdatums unverfallbar und können ausgeübt werden, sodass die gesamte Zuteilung am vierten Jahrestag des Zuteilungsdatums unverfallbar wird, sofern das Beschäftigungsverhältnis mit dem Unternehmen fortgesetzt wird.",
    "keywords": [
      "Insider's Buy/Sell"
    ],
    "insights": null,
    "publisher.name": "GlobeNewswire Inc.",
    "publisher.homepage_url": "https://www.globenewswire.com",
    "publisher.logo_url": "https://s3.massive.com/public/assets/news/logos/globenewswire.svg",
    "publisher.favicon_url": "https://s3.massive.com/public/assets/news/favicons/globenewswire.ico",
    "amp_url": "https://www.globenewswire.com/news-release/2024/06/03/2892674/0/de/ADC-Therapeutics-gew%C3%A4hrt-neuen-Mitarbeitern-Zuteilungen-im-Rahmen-des-Anreizplans.html",
    "TradeDate": "2024-06-03 00:00:00"
  },
  {
    "id": "gI0z776DoBR4t-bR2L3Ru61sM8z3CxLlzwFcJ7HvoMU",
    "title": "Why W.W. Grainger (GWW) is a Top Growth Stock for the Long-Term",
    "author": "Zacks Equity Research",
    "published_utc": "2024-06-03 13:45:11+00:00",
    "article_url": "https://www.zacks.com/stock/news/2282685/why-ww-grainger-gww-is-a-top-growth-stock-for-the-long-term",
    "tickers": [
      "GWW"
    ],
    "image_url": "https://staticx-tuner.zacks.com/images/default_article_images/default27.jpg",
    "description": "Wondering how to pick strong, market-beating stocks for your investment portfolio? Look no further than the Zacks Style Scores.",
    "keywords": null,
    "insights": null,
    "publisher.name": "Zacks Investment Research",
    "publisher.homepage_url": "https://www.zacks.com/",
    "publisher.logo_url": "https://s3.massive.com/public/assets/news/logos/zacks.png",
    "publisher.favicon_url": "https://s3.massive.com/public/assets/news/favicons/zacks.ico",
    "amp_url": "https://www.zacks.com/amp/stock/news/2282685/why-ww-grainger-gww-is-a-top-growth-stock-for-the-long-term",
    "TradeDate": "2024-06-03 00:00:00"
  },
  {
    "id": "X_oB_t5-nlpyEzxMYfZr7rxQ0YK572GGtJbH6sSs5Bo",
    "title": "HUTCHMED Highlights Publication of Phase III FRUTIGA Results in Nature Medicine",
    "author": "HUTCHMED (China) Limited",
    "published_utc": "2024-06-03 00:00:00+00:00",
    "article_url": "https://www.globenewswire.com/news-release/2024/06/03/2891869/0/en/HUTCHMED-Highlights-Publication-of-Phase-III-FRUTIGA-Results-in-Nature-Medicine.html",
    "tickers": [
      "HCM",
      "HCM"
    ],
    "image_url": "https://ml.globenewswire.com/Resource/Download/085957ad-3bec-49ee-b13e-39961b83c7be",
    "description": "Updated subgroup efficacy and quality of life data were also presented on June 1 at ASCO 2024 Updated subgroup efficacy and quality of life data were also presented on June 1 at ASCO 2024",
    "keywords": [
      "Health",
      "Clinical Study",
      "Calendar of Events"
    ],
    "insights": null,
    "publisher.name": "GlobeNewswire Inc.",
    "publisher.homepage_url": "https://www.globenewswire.com",
    "publisher.logo_url": "https://s3.massive.com/public/assets/news/logos/globenewswire.svg",
    "publisher.favicon_url": "https://s3.massive.com/public/assets/news/favicons/globenewswire.ico",
    "amp_url": "https://www.globenewswire.com/news-release/2024/06/03/2891869/0/en/HUTCHMED-Highlights-Publication-of-Phase-III-FRUTIGA-Results-in-Nature-Medicine.html",
    "TradeDate": "2024-06-03 00:00:00"
  }
]
```
</details>

### `adj_factor`

- **vs_ashare**: A股复权因子嵌在日线 `Factor`（Close×Factor）；美股独立 hive，且 ≡ 日线 `AdjFactor`。
- **model**: `D1`
- **purpose**: 后复权因子源（与 StockDailyBar.AdjFactor 同源）
- **cos**: `cos://qs-cold/clean_data/adj_factor/date=YYYY-MM-DD/data.parquet`
- **coverage**: hive；交易日约 5,725；总行数约 4,900 万；2003-09-10 → ~2026
- **join**: equi(date,ticker)
- **sample_rows / n_fields**: 10561 / 2
- **formula（structure.md）**:
  - 上市首日 `adj_factor=1.0`，之后正向累积
  - 拆股倍数 `multiplier = split_to / split_from`
  - 分红倍数 `multiplier = close_prev / (close_prev - cash_amount)`
  - **调整价 = 原始价 × adj_factor(date)**（后复权）

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0.0 |  |
| 2 | `adj_factor` | `double` | 后复权累积因子 | 无量纲 | 0.0 | ≡ DailyBar.AdjFactor；调整价=原价×本值。若同 ticker 任一日 >1e6 则见 is_adj_factor_clamped |

<details><summary>numeric_stats（样本）</summary>

| field | null% | min | p50 | mean | max |
|-------|------:|-----|-----|------|-----|
| `adj_factor` | 0.0 | 2.204585537918872e-14 | 1.0264466239983743 | 2.2648285615649786 | 6210.622537928616 |

</details>

<details><summary>sample_row</summary>

```json
{
  "ticker": "AAPL",
  "adj_factor": 132.58263088104235
}
```
</details>

<details><summary>more_sample_rows</summary>

```json
[
  {
    "ticker": "A",
    "adj_factor": 1.5236417823895225
  },
  {
    "ticker": "JPM",
    "adj_factor": 1.7596532916384435
  },
  {
    "ticker": "ZZZ",
    "adj_factor": 1.000475899691558
  }
]
```
</details>

### `is_adj_factor_clamped`

- **model**: `D1`
- **purpose**: 复权因子浮点戒律/夹紧标记（PREPRO C-053）
- **cos**: `cos://qs-cold/clean_data/is_adj_factor_clamped/date=YYYY-MM-DD/data.parquet`
- **coverage**: hive 按日；与 StockDailyBar 同日 ticker 对齐；总行数约与 adj_factor 同量级
- **join**: equi(date,ticker)
- **vs_ashare**: A股无对等夹紧表；A 用 Factor 时亦勿对极端拆细股盲目用绝对复权价做长窗口。
- **note_rate**: 长历史抽查 True 率约 0~0.013%（2010–2020）；2022–2025 多个样本日为 0。罕见但非零。
- **clamp 规则（structure.md 实盘契约）**:
  - 阈值：`adj_factor > 10^6`
  - **按 ticker**：若该 ticker **任一日期**超过阈值，则其**全部交易日** `is_adj_factor_clamped=True`
  - structure 记载 clamped tickers 例：`TOT`（1 只）
  - True 时：长窗口时序算子应优先用 **收益率 / log-return**，勿依赖绝对复权价格水平

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0 |  |
| 2 | `is_adj_factor_clamped` | `bool` | 复权因子是否被夹紧（>1e6 触发，ticker 全历史打标） | bool | 0 | True 时勿用绝对复权价做长窗口；改用收益/log-return |

<details><summary>sample_row（2020-01-02，全日 True 仅 1）</summary>

```json
{"ticker": "TOT", "is_adj_factor_clamped": true}
```
</details>


### `is_early_close`

- **model**: `STATIC_FLAG`
- **purpose**: 美股提前收盘日标记（半日市）
- **cos**: `cos://qs-cold/clean_data/is_early_close/data.parquet`（**单文件全历史**，不是 date= hive）
- **coverage**: rows≈5937；其中 `is_early_close=True`≈51
- **join**: equi(`date`→trade_date)
- **vs_ashare**: A股无对等半日市表

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `date` | `string` | 交易日 YYYY-MM-DD | date string | 0 | 与日线 TradeDate 对齐前转 date |
| 2 | `is_early_close` | `bool` | 是否提前收盘 | bool | 0 | True 例：感恩节次日、7/3、平安夜。半日市成交量/波动因子需特殊处理 |

<details><summary>sample_row</summary>

```json
{"date": "2003-11-28", "is_early_close": true}
```

True 日例：2003-11-28、2003-12-24、2004-11-26、2006-07-03…（共 51）
</details>


### `universe_daily`

- **model**: `D1_YEAR_FILE`
- **purpose**: 美股研究宇宙（日×ticker）
- **cos**: `cos://qs-cold/clean_data/universe_daily/year=YYYY/data.parquet`
- **coverage**: 例 2024 年文件 rows≈2.66M，dates≈252，tickers≈12642
- **join**: equi(trade_date,ticker)；常与 StockList(CS) 求交
- **vs_ashare**: A股多用 Status+List 过滤；无同构 year 文件

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `trade_date` | `date/timestamp` | 交易日（辅助表/宇宙对齐键） | date | 0 |  |
| 2 | `ticker` | `string` | 宇宙内代码 | string | 0 | 可能宽于 CS StockList，使用前过滤 |

<details><summary>sample_row</summary>

```json
{"trade_date": "2024-01-02T00:00:00", "ticker": "A"}
```

2024 文件：2,664,185 行；252 日；12,642 tickers。
</details>


### `is_ticker_halt`
- **vs_ashare**: **不可**当 A股 `IsSuspend` 稳定替代（覆盖极稀）。A停牌看 DailyBar.IsSuspend。

- **model**: `MINUTE_SPARSE`
- **purpose**: 分钟级 halt 标记
- **cos**: `cos://qs-cold/clean_data/is_ticker_halt/minute/date=YYYY-MM-DD/data.parquet`
- **coverage**: **极稀**（曾见仅个别日期如 2026-06-11）；不可假设全日历史存在
- **join**: (ticker, timestamp)；tz=`America/New_York`；约 09:30–16:00 共 391 个分钟点
- **warning**: 全市场 True 率可能很高（大量非活跃代码）；AAPL/MSFT 样本日可为 0。**不能**稳定替代 A股 `IsSuspend`。

| # | name | type | 中文说明 | unit | null% | caution |
|--:|------|------|----------|------|------:|---------|
| 1 | `ticker` | `string` | 美股代码 | string | 0 |  |
| 2 | `timestamp` | `timestamp[ns, America/New_York]` | 分钟时间 | ET | 0 | tz-aware；勿当 UTC |
| 3 | `is_ticker_halt` | `bool` | 该分钟是否 halt | bool | 0 | 先限制 CS 宇宙再解释 |


## 6. CHECKLIST

```text
[ ] massive_data 而非 temp
[ ] Ret 小数；无 /10000
[ ] AdjFactor 后复权；检查 clamped
[ ] TradeDate→date
[ ] type 过滤 CS/ETF
[ ] 不用 EMPTY 行业/状态
[ ] 财报 asof filing_date；已 filter timeframe
[ ] Valuation/Indicator 已确认非全历史；市值用 SharesSnapshot×Close
[ ] 分红截断未来 ex_div；非 USD currency 已剔除或 FX
[ ] High_Low_Ratio 用 /Low；勿用 Volatility_20d
[ ] Capital：split 文件 ≠ shares_ 文件；日频股本优先 TickerSharesSnapshot
[ ] 半日市查 is_early_close；halt 表不可当稳定 IsSuspend
```

> 另见 **UNIT_ZERO_BIAS_CHECKLIST** 与 Bridge **§C16**。

## 7. data_access 提示

```text
US_MASSIVE_COS_PREFIX=cos://qs-cold/clean_data/us_stock/massive_data
US_CLEAN_COS_PREFIX=cos://qs-cold/clean_data
本地: ~/quant_projects/data/us_stock/massive_data/
市值中性推荐: Close * TickerSharesSnapshot.weighted_shares_outstanding
```

## 8. 实测校验

- Ret≈Close/PreClose-1 median_abs_err=0.00e+00, p99=0.00e+00, n=10358
- 2024-06-03 StockDailyBar rows=10561, tickers=10561
- AdjFactor all>0=True, median=1.0264
- StockList.type 分布: CS=5213
- StockIncome 样本 period_end 文件 rows=102（E2 事件文件，不是全市场日截面）
- StockValuationDaily 样本 rows=5 ← 稀疏铁证

## 9. 给 AI 的任务卡

```text
请同时阅读（两份字典 + 各自文末 CROSS_MARKET_BRIDGE，内容应对齐）:
  /home/shw/COS_ashare_lqtp_data_dictionary.md
  /home/shw/COS_us_massive_data_dictionary.md
任务: 改造 factor_engine
  1. market 维度 ashare|us
  2. 按 CROSS_MARKET_BRIDGE §C2/C3/C7 做字段 adapter
  3. 按 §C4/C5/C8 标注算子 both|ashare-only|us-only|needs-provider
  4. industry_* 美股 fail-closed 或 IndustryProvider
  5. size_* 美股默认 Close×TickerSharesSnapshot.shares
  6. 财务 asof：ashare=PubDate；us=filing_date
  7. 同名表 StockValuationDaily/Capital/Industry 分市场实现，禁止共用
  8. 单元测试：Return/Ret 单位、Factor/AdjFactor 复权方向、禁止跨市场静默 concat
```

## 10. 相关非主表路径

| 路径 | 用途 |
|------|------|
| us_stock/universe/ | 宇宙发布与监控 |
| us_stock/universe_input/ | SecurityIdMapPIT / SharesPIT |
| raw_data/reference/news/ | 原始新闻（raw-cos） |
| massive_data_temp/ | 禁止研究 |

生成于 2026-08-08T00:23:52+0800，样本缓存 `/tmp/us_cos_dict_866322/files`。


## UNIT_ZERO_BIAS_CHECKLIST（提交因子代码前）

```text
[ ] A股 Return 已 /10000；美股 Ret 未再除
[ ] A股 TurnoverRatio/DividendRatio/Roe/Roa/*Margin/Inc*/ShareRatio/Weight 需要小数时已 /100
[ ] 美股 dividend_yield / return_on_equity / return_on_assets 未 /100
[ ] 未把 A股 Factor 与美股 AdjFactor 共用同一复权函数
[ ] 美股 High_Low_Ratio 用 /Low（已验证）；不要写成 /Close 或照抄 A 股自算 /PreClose
[ ] 美股财报已 filter timeframe；asof 键是 filing_date
[ ] 未把 US Valuation/Indicator 当全历史；市值用 SharesSnapshot×Close
[ ] 未混读 StockCapitalDaily 的 split 与 shares_ 文件
[ ] 停牌用 DailyBar.IsSuspend；ST 用 Status.PublicStatus
[ ] 分钟 QuoteTime 按 UTC+8 理解；分钟无 Return 列
[ ] 指数权重：A股 Weight% ；美股无 Weight
[ ] 货币：A=CNY，US=USD（分红看 currency）
```

## NEWS_RAWDATA（自 catalog 并入；非 clean_data）

> 本节省自原 两份核心字典 §4 + §5 RAWDATA。美股新闻权威源在 **raw_data**；clean 层另有 `FactNews`（见上文表节）。A股 clean **无**对等新闻表。

### JOIN 摘要（原 catalog §3.4）

### 3.4 US news (from RAWDATA) — see §4 for full rules
```text
# COS raw (authoritative on bucket; via raw-cos):
news = read("cos://qs-cold/raw_data/reference/news/year={YYYY}/data.parquet")
events = news.explode("tickers").rename(columns={"tickers": "ticker"})
# map published_utc → signal_trade_date (after US close → next session); see §4.8

# Or local derived daily counts (NOT on clean COS):
left join fact_news_daily on (trade_date, ticker); fillna news_count_1d = 0
```

---

## 4. NEWS — detailed `raw_data` guide (NOT clean_data)

> **一句话**：新闻在 COS 的 **`raw_data`**，不在 `clean_data`；用 **`raw-cos`** 读；粒度为「一篇文章一行」；要做因子必须先 `explode(tickers)`，并用 `published_utc` 做 PIT。

```yaml
layer: RAWDATA                    # ← 不是 clean
bucket: qs-cold
cos_root: cos://qs-cold/raw_data/reference/news/
partition: year={YYYY}/data.parquet   # Hive-style year= ; one parquet per year
access_cli: raw-cos               # clean-cos-ro → HTTP 403 AccessDenied
in_clean_data: false
ashare_news: false                # A股 LQTP clean/raw 本路径均无中文新闻
source_api: Massive (Polygon) /v2/reference/news
grain: 1 row = 1 article
ticker_col: tickers               # list<string>, NOT a scalar Ticker
time_model: RAW_EVENT
pit_field: published_utc          # timestamp[ns, tz=UTC]
related_but_not_news: cos://qs-cold/raw_data/filings/
```

### 4.0 AI reading order for news

```text
1) Confirm you need news → you are leaving clean_data
2) Use raw-cos (never clean-cos-ro)
3) Read year=YYYY/data.parquet (prefer ≥2021; 2016–2020 nearly empty on COS)
4) Check completeness: counts OK; insights/keywords publisher-dependent
5) Check sources: publisher mix drifts across years (Zacks drops after 2024)
6) Understand duplicates: id unique; template titles ≠ dup; cross-publisher = multi-source
7) explode → (id, ticker) event rows; apply PIT
8) Optional: local news_events / fact_news_daily (filter batch_id; sentiment may be empty)
```

### 4.1 Scope boundary — where news is / is not

| location | news? | detail |
|----------|-------|--------|
| `clean_data/ashare/lqtp_data/*` | **NO** | A股清洗表无新闻；也无等价中文新闻表 |
| `clean_data/us_stock/massive_data/*`（约 18 张） | **NO** | 价量/财报/名单等；**不含** news |
| **`raw_data/reference/news/`** | **YES** | 本文主路径；Massive 聚合新闻 |
| `raw_data/filings/` | NO (filings) | 8-K/10-Q 等公告正文；**不要**当 news |
| `raw_data/reference/{tickers,exchanges,...}` | NO | 参考主数据，不是新闻 |
| local `.../tmp/reference/news/` | YES (cache) | 可能按 `year=/month=` 更碎、更密 |
| local `.../materialized_panel/news_events/` | derived | ticker 展开 + 去重 + PIT 字段 |
| local `.../materialized_panel/fact_news_daily/` | derived | `(trade_date,ticker)` 日频计数 |

**权限实测**：`clean-cos-ro ls cos://qs-cold/raw_data/reference/news/` → **403 AccessDenied**；必须 `raw-cos`。

### 4.2 COS 目录结构与逐年清单（已核实）

```text
cos://qs-cold/raw_data/reference/news/
├── _SUCCESS
├── year=2016/data.parquet
├── year=2017/data.parquet
├── ...
└── year=2026/data.parquet
```

每个 `year=YYYY/` 下通常还有空目录标记与 `_SUCCESS`；**真正数据文件是 `data.parquet`**（一年一个文件，不是按交易日切）。

| year | file size | articles (rows) | published_utc range (approx) | AI note |
|-----:|----------:|----------------:|------------------------------|---------|
| 2016 | 63 KB | **22** | 2016-06 → 2016-11 | 几乎空 |
| 2017 | 13 KB | **5** | sparse | 几乎空 |
| 2018 | 14 KB | **14** | sparse | 几乎空 |
| 2019 | 23 KB | **45** | sparse | 几乎空 |
| 2020 | 44 KB | **201** | 2020-01 → 2020-12 | 仍极稀 |
| 2021 | 42 MB | **132,928** | dense | 可用起点 |
| 2022 | 67 MB | **211,764** | dense | |
| 2023 | 71 MB | **221,651** | dense | |
| 2024 | 55 MB | **145,756** | 2024-01-01 → 2024-12-31 | 全字段统计基准年 |
| 2025 | 31 MB | **58,601** | partial | 未满年 / 同步截止 |
| 2026 | 25 MB | **32,193** | YTD as-of sync | 未满年 |

```bash
raw-cos ls  cos://qs-cold/raw_data/reference/news/
raw-cos ls  cos://qs-cold/raw_data/reference/news/year=2024/
raw-cos cp  cos://qs-cold/raw_data/reference/news/year=2024/data.parquet ./news_2024.parquet
```

**COS vs 本地缓存**：本地 `/srv/quant/data/infra/data/tmp/reference/news/` 下某些 parquet（路径即使含 `month=`）可能是**跨年全量缓存**（合计约 80 万行量级）。实测其中 2021–2025 行数与 COS 年文件一致，2024 的 `id` 集合与 COS **完全重合**；2026 YTD 可能因同步时刻与 COS 差几百～上千行。写流水线须写明数据源是 COS 还是 local。

### 4.3 数据粒度与基数（必须先懂再 join）

| concept | meaning |
|---------|---------|
| 一行 | **一篇文章**（不是一个 ticker-日） |
| `id` | 文章唯一；2024 COS：`nunique(id)=nrows` |
| `tickers` | `list<string>`，一篇可挂 **1～195** 个 ticker（2024） |
| explode 后 | 2024：145,756 文 → **324,823** 事件行；**11,299** 个 distinct ticker |
| 与日面板关系 | 事件稀疏：多数 `(trade_date, ticker)` **没有**新闻行 |

`tickers` 长度（2024 COS）：

| stat | value |
|------|------:|
| min / p25 / median / p75 / max | 1 / 1 / **1** / 2 / **195** |
| mean | ≈ 2.23 |
| empty list | **0%**（该年样本） |

极端例：标题含宽基/主题时，`tickers` 可同时挂 `SPY,QQQ,DIA` 与大量 ETF（最多 195）——**explode 后会放大头部宽基/ETF 计数**，做「公司特异新闻」因子时建议过滤指数/ETF 或按 publisher/标题规则清洗。

Ticker 形态注意：
- 美股代码：`AAPL`、`BRK.B`、`BRK.A`（有点号）
- `GOOG` 与 `GOOGL` **同时存在**（2024：2369 vs 3028）——合并需显式映射，不能默认其一
- **绝不**等于 A股 `000001.SZ`

### 4.4 字段字典（RAW parquet，15 列，已用 pyarrow 核实）

| # | name | Arrow type | 含义 | 空值/覆盖（2024） | 用法与陷阱 |
|--:|------|------------|------|-------------------|------------|
| 1 | `id` | `string` | 文章唯一 ID（哈希样字符串） | 非空；与行 1:1 | 去重主键；**不同 publisher 报道同一事实 → 不同 id** |
| 2 | `title` | `string` | 标题 | 非空 | NLP / 去重指纹输入 |
| 3 | `author` | `string` | 作者 | 几乎都有（空极少） | 弱信号 |
| 4 | `published_utc` | `timestamp[ns, tz=UTC]` | 发布时间 | 非空 | **唯一可靠 PIT 时钟**；见 §4.8 |
| 5 | `article_url` | `string` | 原文链接 | 非空 | 外链；回测勿依赖实时可打开 |
| 6 | `tickers` | `list<element: string>` | 关联标的数组 | 2024 均非空 list | **必须 explode**；宽基文章会爆炸行数 |
| 7 | `image_url` | `string` | 配图 URL | 2024 样本非空 | 因子通常不用 |
| 8 | `description` | `string` | 摘要/导语（**不是全文**） | ≈0.6% 空 | 短文本特征；勿当完整正文 |
| 9 | `keywords` | `list<string>` | 关键词 | ≈**42% 空**；非空中位长度 2 | 稀疏 |
| 10 | `insights` | `list<struct<sentiment:string, sentiment_reasoning:string, ticker:string>>` | 按 ticker 的情感结构 | 仅 ≈**21.2%** 行非空 | 见 §4.5；**多数文章无情感** |
| 11 | `publisher.name` | `string` | 来源名 | 非空 | 多源；见 §4.6 |
| 12 | `publisher.homepage_url` | `string` | 来源主页 | | 元数据 |
| 13 | `publisher.logo_url` | `string` | logo | | 元数据 |
| 14 | `publisher.favicon_url` | `string` | favicon | | 元数据 |
| 15 | `amp_url` | `string` | AMP 链接 | ≈**37% 空** | 可忽略 |

**`insights` 元素结构示例**：
```json
{
  "ticker": "X",
  "sentiment": "positive",
  "sentiment_reasoning": "The news of a potential acquisition by Nippon Steel led to a sharp increase in U.S. Steel's stock price..."
}
```
注意：`insights` 是数组，一篇文章可对**多个** ticker 各给一条 insight；应与 `tickers` explode 后按 `ticker` 对齐，不能假设 `insights[0]` 对应该文唯一标的。

**完整样例行（2024-12-31，Motley Fool）**：
```json
{
  "id": "40253806f8ea9e6c3ace1489dc4e60ccce273abf556db5f4ca2ba3d45204f805",
  "title": "Why United States Steel Stock Crushed the Market Today",
  "author": "Eric Volkman",
  "published_utc": "2024-12-31T23:49:50Z",
  "article_url": "https://www.fool.com/investing/2024/12/31/...",
  "tickers": ["X"],
  "description": "Nippon Steel has proposed a new offer to acquire U.S. Steel...",
  "keywords": ["U.S. Steel", "Nippon Steel", "..."],
  "insights": [{"ticker": "X", "sentiment": "positive", "sentiment_reasoning": "..."}],
  "publisher.name": "The Motley Fool",
  "amp_url": null
}
```

### 4.5 字段完整性矩阵（什么是「全的」、什么常缺）

> 判断标准（2024 COS 全量 145,756 篇实测）：**全** = 缺失 <1%；**基本全** = 1–10%；**半缺** = 10–50%；**严重缺** = >50%。

| 字段 | 完整度 | 2024 缺失率 | 说明 |
|------|--------|------------:|------|
| `id` | **全** | 0% | 年内无重复 id |
| `title` | **全** | 0% | |
| `published_utc` | **全** | 0% | PIT 可用 |
| `article_url` | **全** | 0% | 几乎 1 URL↔1 id（仅 11 个 URL 对应多 id） |
| `tickers` | **全** | 0% 空数组 | 至少 1 个 ticker |
| `publisher.name` 及 homepage/logo/favicon | **全** | 0% | |
| `image_url` | **全** | 0% | |
| `author` | **全** | blank ≈0.002% | 可忽略 |
| `description` | **基本全** | na ≈**0.59%** | 摘要级，不是全文 |
| `amp_url` | **半缺** | na ≈**36.9%** | 因子可忽略 |
| `keywords` | **半缺→严重偏斜** | 空 ≈**42.2%** | **强依赖 publisher**，见下表 |
| `insights`（情感） | **严重缺** | 空 ≈**78.8%**（有值仅 ≈21.2%） | **强依赖 publisher**；多数文章无情感 |

**按 publisher 的「可选字段」覆盖率（2024 COS）——决定你能不能用 keywords/sentiment：**

| publisher.name | n | `insights` 有值率 | `keywords` 有值率 | 判读 |
|----------------|--:|------------------:|------------------:|------|
| Zacks Investment Research | 57,169 | **1.5%** | **1.5%** | 量最大，但情感/关键词几乎没有 |
| GlobeNewswire Inc. | 38,520 | 33.5% | **100%** | PR；关键词全；情感约 1/3 |
| The Motley Fool | 25,420 | 36.2% | **100%** | |
| Benzinga | 16,427 | 30.5% | **100%** | |
| Investing.com | 3,134 | **87.5%** | 87.5% | 情感最全的一批 |
| MarketWatch | 2,489 | 4.7% | 4.7% | 几乎无 insights/keywords |
| Seeking Alpha | 2,421 | **0%** | **0%** | 只有标题/摘要等主字段 |
| PennyStocks | 108 | 0% | 100% | |
| Invezz | 68 | 100% | 100% | 样本少 |

**派生层情感也不保证全：**
- 本地 `fact_news_daily.news_sentiment_last`：样本月（2024-06，`20260615_zhangjiayin_001`）可出现 **100% NaN**
- 本地 `news_events.sentiment_score`：同样可能大面积 NaN  
→ **「有新闻行」≠「有情感」**；缺测不要当中性 0。

情感标签映射（仅当 `insights` 非空时）：
```text
positive|bullish → +1
neutral|hold     →  0
negative|bearish → -1
mixed|NA         → NaN（或单独桶）
missing insights → NaN（不要填 0）
```

**内容层「全/缺」总结（给因子选型）：**

| 你想要的特征 | 是否可依赖为「全」 | 建议 |
|--------------|-------------------|------|
| 文章是否存在 / 计数 | **可**（主字段全） | `news_count_1d` 主特征 |
| 发布时间 PIT | **可** | `published_utc` |
| 多 ticker 关联 | **可**（需 explode） | 注意宽基放大 |
| 摘要文本 `description` | **基本可** | 非全文 NLP |
| 关键词 | **不可默认全** | 按 publisher 过滤或接受稀疏 |
| 情感 insights | **不可默认全** | 仅子集；或做 missing 指示 |
| 全文 / 中文 A 股新闻 / 法定公告 | **无** | 换数据源 / `filings` |

### 4.6 数据源（publisher）清单与时间漂移

**上游**：Massive（Polygon）`/v2/reference/news` 的多源聚合，不是单一通讯社。

**COS 中出现过的 `publisher.name`（并集）**：

| publisher | 典型角色 | 2021 | 2024 | 2025 | 2026 YTD | 稳定性 |
|-----------|----------|-----:|-----:|-----:|---------:|--------|
| Zacks Investment Research | 模板化投研/选股文 | 58,969 | **57,169** | **11** | **0** | 2025 起在 COS 中几乎消失 |
| GlobeNewswire Inc. | 公司 PR/通稿 | 19,783 | 38,520 | **27,620** | 12,910 | 稳定主力 |
| The Motley Fool | 评论/投教 | 28,513 | 25,420 | 18,377 | 10,023 | 稳定 |
| Benzinga | 快讯 | 11,988 | 16,427 | 9,604 | 7,552 | 稳定 |
| Investing.com | 媒体 | 1,855 | 3,134 | 2,925 | 1,707 | 稳定偏少 |
| MarketWatch | 媒体 | 6,067 | 2,489 | 63 | 1 | **近年骤降** |
| Seeking Alpha | 社区/早餐 | 3,589 | 2,421 | 0* | 0* | 2025+ COS 未见 |
| Invezz / PennyStocks / Quartz / TheStreet | 长尾 | 有 | 极少 | 极少 | 少 | 长尾 |

\*以对应年 COS 文件为准；本地缓存 publisher 集合可更宽（见下）。

**关键结论：**
1. **数据源 = 上表这些 publisher**，由 Massive 聚合进同一 schema。  
2. **源组成会随年份变**：2024 以 Zacks 为第一大源；**2025–2026 COS 几乎无 Zacks**，条数结构突变——跨年比「每日新闻强度」前必须按 publisher 归一或分段。  
3. GlobeNewswire（PR）与 Zacks/Motley（评论）**经济含义不同**，混加 `news_count` 会把通稿当「信息冲击」。  
4. 本地全量缓存文件中还可见 `Quartz`、`TheStreet` 等；COS 近年以表内主源为主。

### 4.7 重复新闻：分类、实测、怎么处理

> 先分清四种「重复」，不要混为一谈。

| 类型 | 是否同一篇文章？ | 2024 COS 实测 | 因子含义 | 建议处理 |
|------|------------------|---------------|----------|----------|
| **A. 完全重复 id** | 是 | **0**（`id` 唯一） | 文件内无字面重复行 | 按 `id` 去重即可（通常不必要） |
| **B. 同 URL 多 id** | 基本是同链异常 | 仅 **11** 个 URL | 极罕见 | 可按 `article_url` 再 drop_duplicates |
| **C. 同标题模板、不同股票**（伪重复） | **否** | 约 **8.75%** 行 title 非唯一；`title+publisher` 重复组 6,560 个，涉及 19,101 行；最大同标题 244 次 | 几乎全是 **Zacks 填空模板**（如 “Are You a Momentum Investor? This 1 Stock Could Be the Perfect Pick”）换 ticker | 计公司新闻时：**保留**（不同标的）；若做标题去重会误杀 |
| **D. 跨 publisher 同标题** | 常为同一题材多源 | **207** 个 title 出现在 ≥2 个 publisher | 真·多源报道 | 事件去重用 SimHash/标题；或按 publisher 只留优先级 |
| **E. 同 ticker 同日多 publisher** | 多为多源/多篇 | **13.45%** 的 (ticker,UTC日) 有 ≥2 个 publisher；热门股（如 NVDA）多数交易日都有 3–4 源 | 「一天多篇」很常见，**不是存储错误** | 用 count 特征时接受；要「事件数」需近重复抑制 |
| **F. 近重复正文/摘要** | 部分是 | `description` 前 120 字重复组 5,688 个；最大组 1,367（模板摘要） | 与 C 同类 | 模板文可按 fingerprint 压 |

**本地 `news_events` 去重（PREPROC-034）：**
- 字段 `text_fingerprint` + `news_dup_suppressed`
- 样本月近重复抑制率 ≈ **1.0%**（`news_dup_suppressed` 均值 ≈0.01）
- 同一 `id` explode 到多 ticker **不是**重复新闻（均值约 2.1 行/ id）

**实操规则：**
```text
计数「文章篇数」     → 按 id 去重后 count（或 explode 前 count）
计数「ticker 曝光」   → explode 后 count（一篇多 ticker 算多次曝光）
计数「独立事件」     → 需要近重复抑制（fingerprint / SimHash）；否则会高估
跨源同一事实         → 预期行为；按研究设计选择 keep-all / 每源保留 / 去重
Zacks 同标题不同股   → 不是重复，禁止只按 title drop
```

**例子（真多源）：** 同一标题可同时被 Benzinga + GlobeNewswire（或 Investing.com）发布，`id` 不同、`published_utc` 接近、`tickers` 相同 → 两条都在库里。

**例子（伪重复）：** Zacks 244 篇同标题 “Are You a Momentum Investor?…” 对应 **231** 个不同 ticker → 模板营销文，不是 244 次复制粘贴同一篇。

### 4.8 覆盖完整性（时间 / 标的 / 内容边界）

| 维度 | 全？ | 证据与含义 |
|------|------|------------|
| 日历 2016–2020 | **否（严重缺失）** | 合计仅约 287 篇；不可做长历史 |
| 日历 2021–2024 | **相对全（Massive 子集内）** | 年文件 13万～22万篇；2024 自然日 **0 天完全无文**（含周末也有稿，但周末更少） |
| 日历 2025–2026 | **部分 / 漂移** | 行数少于 2023；且 publisher 结构变（Zacks 近消失） |
| 美股全 CS 宇宙 | **否** | 对照单日 `StockList` CS≈3474：2024 全年有新闻的仅约 **49%**（≈1695）；其余股票该年无任何关联新闻 |
| 新闻里的 ticker | **超集混杂** | 2024 explode 后 11,299 distinct ticker，大量不在当日 CS 列表（ETF、其它份额、曾用代码等）；事件行约 **8.9%** 落在 `ETFList` |
| 全文 | **否** | 仅 `title`+`description` |
| A 股中文新闻 | **否** | |
| 法定披露 | **否** | 见 `raw_data/filings/` |
| 实时盘中 | **否**（COS 批落盘） | |
| COS vs 本地历史年 | **2024 及更早可一致** | 本地某全量缓存文件中 2024 行数/ id 与 COS **完全一致**；2026 YTD 可能差几百～上千（同步时刻不同） |

### 4.9 更新频率 / 是否实时

| 问题 | 答案 |
|------|------|
| Massive API 本身 | 接近准实时拉取能力 |
| COS `year=*/data.parquet` | **批量落地**的年文件；观测到统一 `LAST MODIFIED` 批次（例 2026-06-12），**不是**盘中持续 append |
| 能否当盘中直播流？ | **不能**（就 COS 形态而言） |
| 适合场景 | 日频因子、事件研究、历史回测 |
| 本地缓存 | 可为全历史单文件或按月；仍应视为批处理 |

### 4.10 时间与 PIT（因子最容易写错的地方）

**时钟**
- 字段：`published_utc`，类型带 **`tz=UTC`**
- 美东交易时段粗换算：常规 RTH ≈ UTC 14:30–21:00（冬令）或 13:30–20:00（夏令）——**不要硬编码单一时差**，用 `America/New_York` 转换

**2024 COS 按 UTC hour 的发文分布（篇）**：高峰约 UTC 12–16（美东午前～盘中），UTC 20–22 仍有大量（接近/盘后）。

**推荐 PIT 映射（与本地 PREPROC-034 / `news_events` 一致的思路）**

```text
knowledge_ts_utc  := published_utc
knowledge_date    := date(published_utc in America/New_York)   # 或 UTC date，需固定一种并文档化
# 若发布时间 ≥ 美东 16:00（正则收盘）→ 信号归到下一交易日
# 实测 news_events：bump 发生在 UTC hour ∈ {21,22,23}，约 7% 行被推到次日
signal_trade_date := next_US_session(knowledge_ts) if after_close else session_of(knowledge_ts)
# 周末/节假日发文 → 落到下一交易日
```

`news_events` 样本（2024-06，batch `20260615_zhangjiayin_001`）：
- ≈ **7.1%** 行 `signal_trade_date > knowledge_date`
- bumped 的 UTC hour 几乎全是 **21/22/23**
- 周末也有发文（周六日合计不可忽视）→ 必须用交易日历对齐

**错误写法**
```python
# WRONG: 用 UTC 日期当交易日直接 equi-join
daily.merge(news.assign(TradeDate=news.published_utc.dt.date), on=["TradeDate","Ticker"])

# WRONG: 不 explode，把 list 当标量 join
daily.merge(news, left_on="Ticker", right_on="tickers")
```

**正确思路**
```python
ev = news.explode("tickers").rename(columns={"tickers": "ticker"})
ev["signal_trade_date"] = map_to_signal_trade_date(ev["published_utc"], us_calendar)
# then groupby(['signal_trade_date','ticker']) → counts / last sentiment
# left-join to daily panel; missing → news_count=0, sentiment=NaN
```

### 4.11 量级（便于设阈值 / 估算力）

**COS year=2024 文章级**

| metric | value |
|--------|------:|
| articles | 145,756 |
| explode 后事件行 | 324,823 |
| distinct tickers | 11,299 |
| 按 `published_utc` 自然日的每日篇数 median / p90 / max / min | **205** / 913 / **1374** / 5 |

**explode 后 2024 被提及最多的 ticker（事件行）**

| ticker | event rows |
|--------|----------:|
| NVDA | 6472 |
| AMZN | 3680 |
| TSLA | 3531 |
| MSFT | 3481 |
| AAPL | 3471 |
| GOOGL | 3028 |
| GOOG | 2369 |
| META | 2342 |
| … | … |
| SPY / QQQ | 亦很高（宽基噪音） |

**本地 `fact_news_daily`（派生，2024 全年某 batch）参考**：每日 ∑`news_count_1d` 中位约 **500**；当日有新闻的 ticker 数中位约 **367**；头部仍是 NVDA/AMZN/…  
（派生层计数规则含去重/交易日映射，故与「按 UTC 自然日数文章」的 205 不是同一指标。）

### 4.12 本地派生层（可选；不是 clean COS）

#### A) `news_events`（事件级）

路径：`/srv/quant/data/infra/data/tmp/materialized_panel/news_events/year=YYYY/month=MM/part-<batch_id>.parquet`

| 字段 | 含义 |
|------|------|
| 原始列 | 与 raw 同名列大部分保留（`id,title,...,publisher.*`） |
| `ticker` | **已标量**（从 `tickers` explode） |
| `sentiment_score` | 数值情感；**常大量 NaN** |
| `knowledge_ts_utc` | 通常 = `published_utc` |
| `knowledge_date` | 知识日期 |
| `signal_trade_date` | 映射后的信号交易日 |
| `text_fingerprint` | 文本指纹（去重用） |
| `news_dup_suppressed` | 1=近重复抑制；样本月均值 ≈ **0.01**（约 1%） |
| `system_insert_time` | 管线写入时间 |
| `batch_id` | 如 `20260615_zhangjiayin_001`；**目录可多 batch 并存，读取必须过滤** |

去重思路（文档记载）：SimHash + 24h 窗口 Jaccard；被抑制行 `news_dup_suppressed=1`。

#### B) `fact_news_daily`（日频稀疏面板）

路径：`.../fact_news_daily/year=YYYY/month=MM/part-<batch_id>.parquet`

| 字段 | 类型 | 含义 |
|------|------|------|
| `trade_date` | timestamp/date | 信号交易日 |
| `ticker` | string | 美股代码 |
| `news_count_1d` | int | 当日该 ticker 计入的新闻条数（已按管线规则） |
| `news_sentiment_last` | float | 当日最后一条情感；常 NaN |
| `batch_id` | string | 必须筛选单一 batch |

性质：
- **稀疏**：无新闻的 `(date,ticker)` **没有行** → 对日面板 `left join` 后 `news_count_1d.fillna(0)`
- 单 ticker 日计数中位多为 1；极端可到数十（样本月 max 43）
- **不是** COS clean 的一部分；机器若无该路径，应直接从 raw 聚合

### 4.13 因子引擎配方

```text
推荐路径 A（只有 COS raw）：
  raw year files → explode tickers → PIT map → groupby(signal_trade_date, ticker)
       → news_count_1d, has_news, optional sentiment_last
  left join US StockDailyBar on (TradeDate, Ticker)
  fillna count=0; sentiment keep NaN

推荐路径 B（有本地物化）：
  fact_news_daily [filter batch_id] ⋈ daily
  fillna news_count_1d=0

可选：
  按 publisher.name 分层计数（PR vs 评论）
  过滤 SPY/QQQ/DIA 及 ETF 列表，降宽基噪音
  用 insights 时先 drop 无 insights 行或单独 missing 指示变量
```

### 4.14 没有什么 / 常见误解

| 误解 | 正解 |
|------|------|
| clean_data 里找 news | 没有；在 **raw_data** |
| `clean-cos-ro` 能读新闻 | **不能**（403）；用 **raw-cos** |
| 每个交易日全市场都有新闻行 | 否；稀疏；约一半 CS 股票整年都可能无新闻 |
| `tickers` 可直接当 join 键 | 否；先 **explode** |
| 2016–2020 COS 可做长历史 | 否；合计仅数百行量级 |
| insights / keywords 总是有 | 否；insights 约 21%；且 **按 publisher 极不均匀**（Seeking Alpha/Zacks 几乎无） |
| 有 `fact_news_daily` 就有情感 | 否；`news_sentiment_last` 可整月 NaN |
| 新闻 = 法定公告 | 否；公告在 `raw_data/filings/` |
| 含 A股中文新闻 / 全文 | 否；仅英文摘要级 |
| 同标题就是重复文章 | **不一定**：Zacks 模板同标题对不同股票是正常的；跨 publisher 同标题才是多源 |
| 同 ticker 同日多篇是数据错误 | **否**：约 13% ticker-日有多 publisher，属预期 |
| `id` 会在年内重复 | 2024 实测 **不会** |
| 2024 与 2025+ 新闻强度可直接比 | **慎**：publisher 结构漂移（Zacks 近消失） |
| COS 与本地历史年必不一致 | 历史年可完全一致；YTD 可能差同步时刻 |
| 条数 = 独立经济事件数 | 否；多源 + 模板文会高估事件数 |

### 4.15 MUST / MUST_NOT（新闻专用）

```text
MUST:
  - 使用 raw-cos 访问 cos://qs-cold/raw_data/reference/news/
  - 明确标注数据层 = RAWDATA（非 clean）
  - explode(tickers) 后再与美股 Ticker 对齐
  - PIT 用 published_utc，并处理盘后/周末 → 下一交易日
  - 先分清字段全/缺：计数可用；insights/keywords 不可默认全
  - 跨年对比时检查 publisher 组成（尤其 Zacks 是否消失）
  - 区分：id 去重 / 模板同标题 / 跨源同题 / 近重复抑制
  - 多 batch 物化表先 filter batch_id
  - 缺失情感用 NaN，不要填 0 当中性

MUST_NOT:
  - 在 clean_data 下查找新闻表
  - 用 clean-cos-ro 读 raw news
  - 与 A股 Symbol 混 join
  - 把 year=2016..2020 当完整历史
  - 把 list 型 tickers 直接 equi-join
  - 只按 title drop_duplicates（会误杀 Zacks 模板多股票）
  - 把「一天多 publisher」当成脏数据删光
  - 把 news 与 filings 混为一谈
  - 假设盘中实时 COS 更新 SLA
  - 假设全 CS 宇宙每天都有新闻覆盖
```

---

### FIELD_DICT — RAW parquet

### [RAWDATA] `reference/news` ⚠️ not clean_data

- **model**: `RAW_EVENT`
- **layer**: `raw_data` (access: `raw-cos` only; `clean-cos-ro` → 403)
- **purpose**: Massive 美股新闻原文（多 publisher）；**完整细则见 §4**
- **cos**: `cos://qs-cold/raw_data/reference/news/year={YYYY}/data.parquet`
- **n_fields / sample_rows**: 15 / 145756 (year=2024 COS)
- **note**: 1行=1文；`tickers` 为 list 需 explode；PIT=`published_utc`；insights 仅≈21%有；2016–2020 COS 几乎空；本地月缓存可能远密于 COS 年文件。

| # | name | type | zh | unit | desc | caution |
|--:|------|------|----|------|------|---------|
| 1 | `id` | `string` | 文章唯一 ID | string | Massive 文章 ID。 | 多源重复报道 ≠ 同一 id。 |
| 2 | `title` | `string` | 标题 | string | | |
| 3 | `author` | `string` | 作者 | string | 可空。 | |
| 4 | `published_utc` | `timestamp[ns, tz=UTC]` | 发布时间 | UTC | PIT 主字段。 | 收盘后映射下一交易日。 |
| 5 | `article_url` | `string` | 原文 URL | string | | |
| 6 | `tickers` | `list<string>` | 关联 ticker 数组 | list | 需 explode。 | 仅美股代码。 |
| 7 | `image_url` | `string` | 配图 | string | 可空。 | |
| 8 | `description` | `string` | 摘要 | string | 非全文。 | |
| 9 | `keywords` | `list<string>` | 关键词 | list | 可空。 | |
| 10 | `insights` | `list<struct<sentiment:string,sentiment_reasoning:string,ticker:string>>` | 情感结构 | list | 覆盖不全（2024≈21%）。 | 勿默认有情感。 |
| 11 | `publisher.name` | `string` | 来源名 | string | Zacks/GlobeNewswire/Motley Fool/… | |
| 12 | `publisher.homepage_url` | `string` | 来源主页 | string | | |
| 13 | `publisher.logo_url` | `string` | logo | string | | |
| 14 | `publisher.favicon_url` | `string` | favicon | string | | |
| 15 | `amp_url` | `string` | AMP | string | 常空。 | |

---


## CROSS_MARKET_BRIDGE（A股 ↔ 美股，双向必读）

> 配对文档：
> - A股：`/home/shw/COS_ashare_lqtp_data_dictionary.md`
> - 美股：`/home/shw/COS_us_massive_data_dictionary.md`
>
> 本节回答：哪些字段完全一样、哪些同名不同义、哪些只有一边有、factor_engine 怎么分市场接线。

### C0. 一句话总则

```text
1) 同名不一定同义（最危险）
2) 概念相同通常列名不同（Symbol/Ticker, Return/Ret, Factor/AdjFactor, Vwap/VWAP）
3) 表同名也不一定同 model（StockValuationDaily: A股D1全历史 vs 美股X0稀疏）
4) 算子能否跨市场，先看「字段是否存在 + 单位/复权/PIT 是否兼容」
```

### C1. 表级对照

| 概念/表 | A股表 | A股 model | 美股表 | 美股 model | 关系 |
|---------|------|-----------|--------|------------|------|
| 交易日历 | Calendar | STATIC | Calendar | STATIC | 概念同；字段名不同（IsTradeDay vs is_trading_day） |
| 股票日线 | StockDailyBar | D1 | StockDailyBar | D1 | 概念同；收益/复权/日期类型不同 |
| 股票清单 | StockList | D1 | StockList | D1 | 概念同；美股多 type/locale，A股多交易所后缀语义 |
| ETF日线 | ETFDailyBar | D1 | ETFDailyBar | D1 | 同构于各自 StockDailyBar |
| ETF清单 | ETFList | D1(按日) | ETFList | STATIC(full) | 布局不同 |
| 状态 | StockStatus | S1 | StockStatus | **EMPTY** | 仅 A股可用 |
| 行业 | StockIndustry | D1 多源 | StockIndustry | **EMPTY** | 仅 A股可用 → industry_neutralize |
| 估值日面板 | StockValuationDaily | **D1 全历史** | StockValuationDaily | **X0 ~49日** | **禁止当同一能力** |
| 财务指标 | StockIndicator | E1(按PubDate) | StockIndicator | **X0 稀疏** | 能力不对称 |
| 资产负债表 | StockBalance | E1 文件名=PubDate | StockBalance | E2 文件名=period_end | PIT键不同 |
| 利润表 | StockIncome | E1 | StockIncome | E2 | 同上 |
| 现金流 | StockCashFlow | E1 | StockCashFlow | E2 | 同上 |
| 分红 | StockDividend | E1/effective | StockDividend | E2 | 都可事件对齐；美股含未来日 |
| 股本 | StockCapitalDaily | **S1 日快照** | StockCapitalDaily | **E2 事件/PIT股** | 语义不同 |
| 指数日线 | IndexDailyBar | D1 | （无对等表） | — | A股 only |
| 指数清单 | IndexList | D1 | （无对等） | — | A股 only |
| 指数成分 | IndexConstituent | D1 + Weight% | StockIndicesComponents | D1 **无Weight** | 权重能力不对称 |
| 分钟线 | StockMinuteBar | MINUTE | （本字典无全量分钟） | — | A股强；美股仅 halt 分钟标记 |
| 十大股东 | StockTopTen* | S1 | （无） | — | A股 only |
| 复权辅助 | （在日线 Factor） | — | adj_factor / is_adj_factor_clamped | D1 | 美股独立辅助表 |
| 新闻 | （clean 无） | — | FactNews + raw news | RAW_EVENT | 美股 only（clean层） |
| 证券主数据 | （散落 List/Status） | — | SecurityMaster(+DailySnap) | STATIC/D1 | 美股更完整 |
| 股本日快照 | Capital/Valuation | — | TickerSharesSnapshot | D1 | 美股市值中性推荐源 |
| 研究宇宙 | （平台侧） | — | universe_daily | D1 | 美股辅助 |

### C2. 字段：可安全共享语义（同概念，可写适配层后复用算子）

这些字段在两边都存在或可直接映射后，用于 **价量时序/截面** 算子（`ts_*`、OHLC 技术指标、`rank/zscore` 等）。仍需经 **market adapter** 统一列名。

| 逻辑字段 | A股列名 | 美股列名 | 说明 |
|----------|---------|----------|------|
| instrument | Symbol | Ticker（少数表 Symbol/ticker） | 必须 adapter，不可直接 concat |
| trade_date | TradeDate (date32) | TradeDate (timestamp[ns])→date | 必须归一 |
| open/high/low/close | Open/High/Low/Close | 同名 | 未复权价；货币不同但算子可共享 |
| volume | Volume (uint64) | Volume (double) | 类型宽容即可 |
| amount | Amount (CNY) | Amount (USD) | 货币不同；比率类算子通常无感 |
| vwap | Vwap | VWAP | **仅大小写**；映射后可共享 |

### C3. 字段：同概念但严禁混用（最重要）

| 逻辑概念 | A股 | 美股 | 差异 | 错误后果 |
|----------|-----|------|------|----------|
| 日收益 | `Return` **bp**，用前 `/10000` | `Ret` **小数**，不要 `/10000` | 单位差 10000 倍 | IC/回测数量级全错 |
| 复权因子 | `Factor`：**后复权价=Close×Factor** | `AdjFactor`：**Close×AdjFactor** | 同为乘法后复权；基期/事件覆盖不同 | 仍禁止当同一列；旧文档「前复权/Close÷Factor」已更正 |
| 复权价公式 | `Close×Factor`；前复权再 `/Factor_asof` | `Close×AdjFactor` | 公式同类 | 仍分市场实现，禁止静默混用 |
| 财务可知时点 | `PubDate`；**文件名=PubDate** | `filing_date`；**文件名=period_end** | PIT 键与文件切分不同 | 前视偏差 / join 空 |
| 报告期末 | `ReportPeriodEndDate` | `period_end` | 两边都不是默认 PIT | 用错成 equi-join |
| 标的 ID | `Symbol` 带 `.SH/.SZ` | `Ticker` 无交易所后缀 | 编码空间不同 | 静默错配 |
| 市值 | `MarketCap` 在 Valuation **D1 全历史** | `market_cap` 主要在 **X0 稀疏表** | 可用性不同 | 美股 size_neutralize 空窗 |
| 换手/比率单位 | TurnoverRatio 等常为 **%** | US 比率字段需逐项确认 | 百分数 vs 小数 | 阈值错一个数量级 |
| VWAP 列名 | `Vwap` | `VWAP` | 大小写 | select 丢列 |
| 指数权重 | `Weight` 百分数，和≈100 | Components **常无 Weight** | 能力缺失 | 加权中性失败 |
| Capital 语义 | 每日股本**快照** S1 | 事件/PIT 股本 E2 + SharesSnapshot | 表同名语义不同 | 股本时间轴错 |

### C4. 仅 A股有（美股无对等或 EMPTY）

| A股字段/表能力 | 用途 | 对 factor_engine |
|----------------|------|------------------|
| `StockIndustry` + `IndustrySource`（sw_l1 等） | 行业中性 | `industry_neutralize` / 双中性默认 **ashare-only** |
| `StockStatus`（PublicStatus/IsSuspend/ST） | 停牌与风险股过滤 | 美股改用 halt/宇宙规则 |
| `HighLimit` / `LowLimit` | 涨跌停 | `limit_up/down` **ashare-only** |
| `IndexDailyBar` / `IndexList` | 指数行情与清单 | 基准对接分市场 |
| `IndexConstituent.Weight` | 指数加权 | 美股权重需另源 |
| `StockTopTenShareholder*` | 持股集中度 | ashare-only 因子 |
| `StockMinuteBar` 全量 | 分钟因子 | 美股分钟能力不对等 |
| `TurnoverRatio` 全日频估值面板 | 换手 | 美股无同构 D1 换手面板 |
| `Factor`（后复权乘数） | 复权 | 公式类似 AdjFactor，仍不可当同一列 |

### C5. 仅美股有（A股无对等）

| 美股字段/表能力 | 用途 | 对 factor_engine |
|-----------------|------|------------------|
| `AdjFactor` + `adj_factor` + `is_adj_factor_clamped` | 后复权与截断标记 | us 复权链路 |
| `Ret` / `Ret_Intra` / `Ret_Overnight` | 已算好的小数收益分解 | ashare 需自算或从 Return 转换 |
| `SecurityMaster` / `SecurityMasterDailySnap` | 主数据/PIT 属性 | 类型、交易所、FIGI/CIK |
| `TickerSharesSnapshot` | 日频股本 | **推荐** `size_neutralize` 暴露：`Close×shares` |
| `FactNews` + raw news | 新闻事件 | ashare clean 无新闻 |
| `is_early_close` | 半日市 | 会话长度处理 |
| `is_ticker_halt`（分钟） | halt | 替代 A股 IsSuspend 的部分场景 |
| `universe_daily` | 研究宇宙 | 与 StockList 交叉 |
| `TickerMap` / FIGI / CIK | 跨源对齐 | |
| ETFList STATIC 丰富属性 | ETP 元数据 | |

### C6. 「同名字段」清单（最危险，默认 fail-closed）

下列名字在两边都可能出现，但**不能当同一列 concat / 共用常量**：

| 同名 | A股含义 | 美股含义 | 建议 |
|------|---------|----------|------|
| `Symbol` | `000001.SZ` | 有时=Ticker=`AAPL` | 永远带 market 前缀或改名 `instrument` |
| `TradeDate` | date32 交易日 | timestamp[ns] | 先 `.astype('date')` |
| `Open/High/Low/Close/Volume/Amount` | CNY 口径 | USD 口径 | 算子可共享，货币敏感统计要分市场 |
| `StockValuationDaily`（表名） | 全历史 D1 | 稀疏 X0 | 代码里分 `ashare_valuation` / `us_valuation_sparse` |
| `StockIndicator`（表名） | 财报事件 E1 | 稀疏 X0 | 同上 |
| `StockCapitalDaily`（表名） | S1 股本快照 | **双文件**：split 事件 + shares_ PIT（见 §C13） | 分函数；日频股本用 SharesSnapshot |
| `StockIndustry`/`StockStatus`（表名） | 可用 | EMPTY | 美股调用直接报错 |

### C7. 推荐字段映射（factor_engine adapter）

```text
# 统一逻辑层（示例）
instrument:
  ashare: Symbol
  us:     Ticker   # StockList 的 Symbol 先 rename→Ticker

trade_date:
  ashare: TradeDate                 # date32
  us:     TradeDate.dt.date         # from timestamp[ns]

raw_close/open/high/low/volume/amount:
  both: same physical names after instrument/date normalize

vwap:
  ashare: Vwap
  us:     VWAP

decimal_return:
  ashare: Return / 10000
  us:     Ret

adjust_factor:
  ashare: Factor       # backward: adj = Close * Factor; forward = that / Factor_asof
  us:     AdjFactor    # backward: adj = Close * AdjFactor
  NOTE: both multiply; still DO NOT concat columns or assume identical base dates

market_cap_exposure:
  ashare: StockValuationDaily.MarketCap
  us:     Close * TickerSharesSnapshot.weighted_shares_outstanding
          # fallback sparse: StockValuationDaily.market_cap (X0 only)

industry_group:
  ashare: StockIndustry.IndustryCode where IndustrySource='sw_l1'
  us:     UNSUPPORTED unless IndustryProvider plugged in

filing_pit_time:
  ashare: PubDate
  us:     filing_date
```

### C8. 算子市场能力（执行层）

| 算子族 | ashare | us | both | 条件 |
|--------|:------:|:--:|:----:|------|
| `ts_*` / OHLC 技术指标 | ✓ | ✓ | ✓ | 先走 adapter；复权宏分市场 |
| `rank`/`zscore`/`cs_*`/`winsorize` | ✓ | ✓ | ✓ | 宇宙过滤分市场 |
| `group_neutralize`（显式 group） | ✓ | ✓ | ✓ | 美股需自备 group 列 |
| `industry_neutralize` | ✓ | ✗ | | US Industry EMPTY |
| `size_neutralize` | ✓ | △ | | US 用 shares×price |
| `industry_size_neutralize` | ✓ | ✗ | | 依赖行业 |
| `limit_up`/`limit_down`/`tradable_state` | ✓ | ✗ | | |
| `fin_*` / fiscal | ✓ | ✓ | ✓ | asof 键分市场 |
| 新闻事件算子 | ✗ | ✓ | | |
| 分钟算子 | ✓ | △ | | US 分钟不全 |

### C9. 自检十题（跨市场）

1. `Return=-550` 和 `Ret=-0.055` 是否同一收益？ → **是（约）**；单位处理不同  
2. 能否把 A股与美股 `Close` 直接 concat 做统一截面？ → **不能**（货币/标的空间）  
3. 美股能否直接调用 `industry_neutralize(x)`？ → **不能**（无行业表）  
4. 美股 `size_neutralize` 默认读 Valuation.MarketCap？ → **危险**；应用 SharesSnapshot  
5. 财务文件日期能否 equi-join 交易日？ → **两边都不能**  
6. `Factor` 能否当 `AdjFactor` 用？ → **不能直接当同一列**（虽同为 Close×因子 后复权）  
7. `StockValuationDaily` 两边是否同等？ → **否**（D1 vs X0）  
8. `Symbol` 在美股 StockList 是什么？ → **Ticker 语义**，不是 `000001.SZ`  
9. 指数成分加权中性两边都能做吗？ → **A股可以；美股默认缺 Weight**  
10. 新闻因子？ → **仅美股 clean/raw 有**
11. 美股 `dividend_yield=0.02` 与 A股 `DividendRatio=2.0`？ → **可能同是 2%**；单位不同勿直接比  
12. 美股 `StockIndicator.Roe`？ → **没有**；该表是估值克隆，不是 A股财务比率表  
13. 读 `StockCapitalDaily/2024-01-02.parquet` 能当股本面板吗？ → **不能**；那是 split 事件文件  
14. `TickerSharesSnapshot` 是否覆盖全部 DailyBar？ → **否**（样本日约四成）
### C10. 估值/比率字段对照（单位陷阱 · 2026-08-08 复核）

> 同概念映射后仍可能因 **%/小数** 差一个数量级。阈值、winsor、中性化前必须分市场归一。

| 逻辑概念 | A股字段（表） | A股单位（契约） | 美股字段（表） | 美股单位（实测倾向） | 能否直接混用 |
|----------|---------------|-----------------|----------------|----------------------|--------------|
| 总市值 | `MarketCap`（Valuation **D1**） | CNY | `market_cap`（Valuation **X0**）或 `Close×weighted_shares_outstanding` | USD | **否**（货币+可用性） |
| 流通市值 | `CirculatingMarketCap` | CNY | （无同构） | — | A only |
| 换手率 | `TurnoverRatio` | **百分数 %** | （无同构；仅有 `average_volume`） | — | A only |
| PE | `PeRatio` / `PeRatioLyr` | 倍数 | `price_to_earnings` | 倍数 | 概念可映射；面板可用性不同 |
| PB | `PbRatio` | 倍数 | `price_to_book` | 倍数 | 同上 |
| PS | `PsRatio` | 倍数 | `price_to_sales` | 倍数 | 同上 |
| PCF | `PcfRatio`/`PcfRatio2` | 倍数 | `price_to_cash_flow` / `price_to_free_cash_flow` | 倍数 | 口径可能不同 |
| 股息率 | `DividendRatio` | **百分数 %**（例 0.54≈0.54%） | `dividend_yield` | **小数倾向**（>0 中位≈0.023≈2.3%） | **否**，先统一成小数 |
| ROE | `Roe`（Indicator **E1**） | **百分数 %** | `return_on_equity`（Valuation/Indicator **X0**） | **小数倾向**（中位≈0.054） | **否** |
| ROA | `Roa`（Indicator E1） | **百分数 %** | `return_on_assets`（X0） | **小数倾向** | **否** |
| EV / EV倍数 | （无完整同构） | — | `enterprise_value`, `ev_to_*` | USD / 倍数 | US only |
| 波动 | （可自算） | — | `Volatility_20d`（仅 Valuation，样本常空） | 视定义 | 勿假设有值 |

**X0 澄清**：美股 Valuation/Indicator 是「**日历文件极少（~49 天）**」，不是「单文件内 80% 行空」。2026-07-27 单日文件可有 ~4800 ticker；仍**禁止**当全历史日面板。

### C11. 财务报表概念映射（列名几乎全不同）

> 不要用字符串相似度自动 rename。下面是**概念级**映射，会计口径/合并范围仍可能不同。

#### Balance

| 概念 | A股 | 美股 |
|------|-----|------|
| 现金及等价物 | `CashEquivalents` | `cash_and_equivalents` |
| 应收 | `AccountReceivable` 等 | `receivables`（汇总） |
| 存货 | `Inventories` | `inventories` |
| 流动资产合计 | `TotalCurrentAssets` | `total_current_assets` |
| 商誉 | `GoodWill` | `goodwill` |
| 资产合计 | `TotalAssets` | `total_assets` |
| 应付账款 | `AccountsPayable` | `accounts_payable` |
| 流动负债合计 | `TotalCurrentLiability` | `total_current_liabilities` |
| 长期借款/租赁债 | `LongtermLoan` / `LeaseLiability` 等 | `long_term_debt_and_capital_lease_obligations`（汇总） |
| 负债合计 | `TotalLiability` | `total_liabilities` |
| 股本/实收资本 | `PaidinCapital` | `common_stock`（+ preferred 等） |
| 留存收益 | `RetainedProfit` | `retained_earnings_deficit` |
| 归母权益 | `EquitiesParentCompanyOwners` | `total_equity_attributable_to_parent` |
| 少数股东权益 | `MinorityInterests` | `noncontrolling_interest` |
| 权益合计 | `TotalOwnerEquities` | `total_equity` |
| PIT 键 | `PubDate` | `filing_date` |
| 报告期 | `ReportPeriodEndDate` | `period_end` |
| 文件名 | =PubDate | =period_end |

#### Income

| 概念 | A股 | 美股 |
|------|-----|------|
| 营收 | `OperatingRevenue` / `TotalOperatingRevenue` | `revenue` |
| 营业成本 | `OperatingCost` | `cost_of_revenue` |
| 毛利 | （可自算） | `gross_profit` |
| 销售/管理费用 | `SaleExpense` / `AdministrationExpense` | `selling_general_administrative`（常合并） |
| 研发 | `RdExpenses` | `research_development` |
| 营业利润 | `OperatingProfit` | `operating_income` |
| 利润总额 | `TotalProfit` | `income_before_income_taxes` |
| 净利润（含少数） | `NetProfit` | `consolidated_net_income_loss` |
| 归母净利 | `NpParentCompanyOwners` | `net_income_loss_attributable_common_shareholders` |
| 基本/稀释 EPS | `BasicEps` / `DilutedEps` | `basic_earnings_per_share` / `diluted_earnings_per_share` |
| EBITDA | （常需自算） | `ebitda` |

#### CashFlow

| 概念 | A股 | 美股 |
|------|-----|------|
| 经营现金流净额 | `NetOperateCashFlow` | `net_cash_from_operating_activities` |
| 购建固定资产等（≈CAPEX） | `FixIntanOtherAssetAcquiCash` | `purchase_of_property_plant_and_equipment` |
| 投资现金流净额 | `NetInvestCashFlow` | `net_cash_from_investing_activities` |
| 筹资现金流净额 | `NetFinanceCashFlow` | `net_cash_from_financing_activities` |
| 现金净增加 | `CashEquivalentIncrease` | `change_in_cash_and_equivalents` |
| 分红付现 | `DividendInterestPayment`（含息） | `dividends` |

### C12. 分红 / 日历 / 指数成分

| 概念 | A股 | 美股 | 注意 |
|------|-----|------|------|
| 除权日 | `ExDividendDate` | `ex_dividend_date` | 美股文件可含**未来**除权日 |
| 股权登记日 | `RightRegDate` | `record_date` | |
| 现金红利 | `CashDividend`（CNY/股） | `cash_amount`（**随 currency**，另有 `split_adjusted_cash_amount`） | 美股非 USD≈12%；禁止默认当美元 |
| 送股/转增 | `StockDividend` / `StockTransfer` | 拆分多在 Capital **split 事件**；分红 `distribution_type` | 结构不同 |
| 日历字段 | `TradeDate`,`IsTradeDay` | `trade_date`,`is_trading_day` | **蛇形命名** |
| 指数成分 | `IndexConstituent` + `Weight`% | `StockIndicesComponents`（`IndexName`,`Symbol`）**无 Weight** | 加权能力不对称 |
| 指数名 | `IndexSymbol`=`000300.SH` | `IndexName`=`S&P 500` 等 | 编码空间不同 |

### C13. 美股 `StockCapitalDaily` 双文件类型（易踩坑）

同一目录混放两种 Parquet，**schema 完全不同**：

| 文件名模式 | 角色 | 主键/要点 | 是否等于 A股 Capital |
|------------|------|-----------|----------------------|
| `{YYYY-MM-DD}.parquet` | **拆分/调整事件** | `ticker,execution_date,split_from,split_to,adjustment_type,historical_adjustment_factor` | **否** |
| `shares_{YYYY-MM-DD}.parquet` | **稀疏 PIT 股本** | `Ticker,pit_basic/diluted`；单日行数极少；**同日同 Ticker 可多行**需去重 | **否**（A股是全市场 S1 快照） |

日频市值/股本暴露：**优先** `TickerSharesSnapshot`（样本日 2024-06-03：与 StockDailyBar ticker 交集覆盖约 **42%**，仍远好于 Capital shares 文件）。

### C14. `StockIndicator`：同名不同物种

| | A股 StockIndicator | 美股 StockIndicator |
|--|--------------------|---------------------|
| model | **E1** 财报公告事件 | **X0** 稀疏日文件 |
| 内容 | ROE/ROA/利润率/同比增速等财务比率 | 几乎是 Valuation 克隆（缺 `Volatility_20d`） |
| PIT | `PubDate` | 无财报 PIT 语义；且日历覆盖极窄 |
| 映射 | 对应美股应去 **StockIncome/Balance asof** 或自算，**不要**去美股 Indicator 找 `Roe` | |

### C15. 2026-08-08 实盘复核摘要

```text
OK: 抽查 A/US 关键表 live schema 与 FIELD_DICTS 列名一致（DailyBar/三表财务/Valuation/Dividend/Calendar/Components/Shares 等）
OK: Ret ≡ Close/PreClose-1；Return/10000 ≡ Close/PreClose-1（误差中位 0）
NEW: 文档补强 — 比率 % vs 小数；财务概念映射；Capital 双 schema；Indicator 物种差异；Shares 覆盖率
KEEP: Industry/Status US EMPTY；Valuation/Indicator X0（49 files）；Components 无 Weight
```
### C16. 单位归一化总表（挖因子前必做 · 防偏差最重要）

> 目标：所有跨市场/共享算子输入，统一成 **小数收益 / 小数比率 / 原始价格金额（分市场货币）**。  
> **禁止**对美股再 `/100` 或 `/10000`；**禁止**对 A 股 `Return` 不转换就当收益。

#### C16.1 收益 / 复权 / 价量

| 逻辑量 | A股字段 | A股存盘 | → 小数/可用公式 | 美股字段 | 美股存盘 | → 小数/可用公式 | 混用？ |
|--------|---------|---------|-----------------|----------|----------|-----------------|--------|
| 日收益 | `Return` | **bp** | `Return/10000` | `Ret` | **小数** | `Ret`（不要再除） | 转换后可比概念 |
| 日内收益 | （自算 `(C-O)/O`） | — | 小数 | `Ret_Intra` | 小数 | 已是 `(C-O)/O` | |
| 隔夜收益 | （自算） | — | 小数 | `Ret_Overnight` | 小数 | 已算好 | |
| 振幅 | （自算，常 `(H-L)/PreClose`） | — | 小数 | `High_Low_Ratio` | 小数 | **=`(H-L)/Low`**（已验证） | 分母不同：A自算常用 PreClose，美股字段用 Low |
| 复权因子 | `Factor` | **后复权乘数** | `Close×Factor`（已验证） | `AdjFactor` / `adj_factor` | 后复权乘数 | `Close×AdjFactor` | 公式同类，勿混列 |
| VWAP | `Vwap` | CNY | 原样 | `VWAP` | USD | 原样 | 列名+货币 |
| 成交额 | `Amount` | CNY | 原样 | `Amount` | USD | 非空时 `Amount=VWAP*Volume` 精确；约 3.35% 行 Amount 与 VWAP 同为 null | |
| 涨跌停 | `HighLimit`/`LowLimit` | CNY价 | 主板约±10%，创业/科创±20%，ST 常±5% | **无** | — | — | ashare-only |

#### C16.2 百分数 %（A股常见） vs 小数（美股常见）

| 逻辑量 | A股 | A股单位 | 转小数 | 美股 | 美股单位 | 转小数 | 2026-08-08 证据 |
|--------|-----|---------|--------|------|----------|--------|-----------------|
| 换手率 | `TurnoverRatio` | **%** | `/100` | （无同构；`average_volume` 不是换手） | — | — | A: ≈Volume/CirculatingCap×100，误差~0 |
| 股息率 | `DividendRatio` | **%** | `/100` | `dividend_yield` | **小数** | 不除 | A p50≈0.54(=0.54%)；US>0 p50≈0.023(=2.3%) |
| ROE | `Roe` | **%** | `/100` | `return_on_equity` | **小数** | 不除 | A p50≈0.6(=0.6%，常为报告期未年化)；US p50≈0.054 |
| ROA | `Roa` | **%** | `/100` | `return_on_assets` | **小数** | 不除 | 同上 |
| 净利率/毛利率等 | `*Margin` 等 Indicator | **%** | `/100` | （无同构 E1 表） | — | 用财报自算则为小数 | A GrossProfitMargin p50≈22 |
| 同比/环比增速 | `Inc*YearOnYear`/`Inc*Annual` | **%** | `/100` | （无） | — | — | 例 34.14=34.14% |
| 费用/收入比 | `ExpenseToTotalRevenue` 等 | **%** | `/100` | （无） | — | — | p50 可接近 100 |
| 持股比例 | `ShareRatio` | **%** | `/100` | （无十大股东表） | — | — | Top10 合计中位≈87%，可>100（勿假设=100） |
| 指数权重 | `Weight` | **%** | `/100`（若要权重和=1） | **无 Weight** | — | — | 沪深300 ΣWeight≈100.00 |

#### C16.3 倍数类（两边多为「倍」，仍勿 concat）

| 逻辑量 | A股 | 美股 | 备注 |
|--------|-----|------|------|
| PE | `PeRatio` / `PeRatioLyr` | `price_to_earnings` | 倍数；美股仅 X0 日历 |
| PB | `PbRatio` | `price_to_book` | 倍数 |
| PS | `PsRatio` | `price_to_sales` | 倍数 |
| PCF | `PcfRatio*` | `price_to_cash_flow` / `price_to_free_cash_flow` | 口径可能不同 |
| D/E、流动/速动 | （可自算） | `debt_to_equity` / `current` / `quick` / `cash` | 美股 X0；`cash` 语义偏比率 |

#### C16.4 factor_engine 推荐归一伪代码

```python
def to_decimal_return(df, market):
    if market == "ashare":
        return df["Return"] / 10000.0
    if market == "us":
        return df["Ret"]  # already decimal

def to_decimal_ratio(series, market, kind):
    """kind: turnover|div_yield|roe|roa|margin|yoy|weight|share_ratio"""
    if market == "ashare":
        return series / 100.0   # ALL listed kinds are percent-encoded
    if market == "us":
        # div_yield / roe / roa already decimal on Valuation X0
        return series

def adjusted_close(df, market, *, mode="backward", factor_asof=None):
    if market == "ashare":
        px = df["Close"] * df["Factor"]          # verified backward
        if mode == "forward":
            return px / factor_asof               # Factor on asof date per symbol
        return px
    if market == "us":
        return df["Close"] * df["AdjFactor"]     # backward

# NEVER:
#   us_ret = us["Ret"] / 10000
#   ashare_roe_decimal = ashare["Roe"]          # missing /100
#   mixed = concat(ashare_roe, us_roe)          # without unit normalize + market tag
```

### C17. 更多结构差异（单位以外同样会写错因子）

| 主题 | A股 | 美股 | 因子风险 |
|------|-----|------|----------|
| 停牌 | `StockDailyBar.IsSuspend`（**不在** StockStatus） | 无同构；看 `is_ticker_halt` / 宇宙 | 过滤条件抄错表 |
| 上市状态 | `StockStatus.PublicStatus`（正常上市/ST/*ST/终止上市…） | `StockList.type` 等；Status EMPTY | |
| 财务 timeframe | 无此列；靠报告期日期区分 | **`timeframe`∈{quarterly, annual, trailing_twelve_months}** 同文件混存 | **必须先 filter**，否则 TTM 与季度混截面 |
| 财务金额 | CNY 绝对额 | USD 绝对额（`revenue` 等） | 货币 |
| EPS | `BasicEps` CNY/股 | `basic_earnings_per_share` USD/股 | |
| 分红频率 | 事件制，无 frequency | `frequency`：12≈月、4≈季、2≈半年、1≈年 | |
| 分红货币 | CNY | `currency` 以 USD 为主，可有 HKD/EUR/… | 别当全是美元 |
| 分钟线 | 240 根/日；`QuoteTime` **存 UTC**（01:31Z=09:31 上海）；**无 Return 列** | 无全量分钟 | 勿用 UpdateTime；勿找分钟 Return |
| 指数日线 | `IndexDailyBar`，`Return` 仍为 **bp** | 无对等表 | |
| 辅助复权 | 无独立表 | `adj_factor/date=YYYY-MM-DD/data.parquet` 与日线 `AdjFactor` **一致**；`is_adj_factor_clamped` | hive 分区 |
| StockList | 股票名单 | 样本日仅 `type=CS`；ETF 在 ETF* 表 | 勿在 StockList 找 ETF |
| Capital | 单一 S1 schema | 双 schema（§C13） | |
| Indicator | E1 财务比率 % | X0 估值克隆，ROE 已是小数 | 同名最坑 |

### C18. 已验证恒等式（可当单测）

```text
A 2024-06-03:
  Return/10000 == Close/PreClose - 1                 # err med=0
  TurnoverRatio ≈ Volume/CirculatingCap*100          # err med≈0
  MarketCap ≈ Capitalization*Close                   # err≈0
  Amount ≈ Vwap*Volume                               # rel err ~1e-5
  Index 000300.SH Σ Weight ≈ 100.003
  HighLimit/PreClose 中位：主板 1.10，创业/科创 1.20
  IndexDailyBar.Return 同样是 bp

US 2024-06-03 / Valuation 2026-07-27:
  Ret == Close/PreClose - 1                          # err med=0
  Ret_Intra == (Close-Open)/Open
  Ret_Overnight == (Open-PreClose)/PreClose
  High_Low_Ratio == (High-Low)/Low                   # NOT /Close （旧文档曾写错）
  Upper_Shadow_Ratio == (High-max(Open,Close))/(High-Low)  # H=L → 0
  Vwap_Close_Dist == (VWAP-Close)/Close
  Amount == VWAP*Volume                              # 两者非空时精确；约3.35%同为null
  DailyBar.AdjFactor == adj_factor.adj_factor        # 当日全相等
  ETFDailyBar.Ret 同样是小数
```

### C19. 复权口径更正（推翻旧「A股前复权」说法）

```text
旧错误（已废止）:
  - 「A股 Factor = 前复权」
  - 「前复权价 ≈ Close / Factor」
  - 「与美股 AdjFactor 方向相反」

新实证（2026-08-08，15 个除权/送转事件）:
  - |Δ(Close×Factor) − (Close/PreClose−1)| 中位 ≈ 0
  - |Δ(Close/Factor) − day_ret| 中位 ≈ 6%（除权日 Close/Factor 断裂）
  - 结论: 后复权价 = Close × Factor
  - 前复权价（派生）= Close × Factor / Factor_asof
  - 美股: Close × AdjFactor（同为后复权乘法）
  - 跨市场: 公式同类 ≠ 可以混列
```
### C20. 2026-08-08 二次审计：已确认 / 已更正

```text
已再验证为正确:
  A Return bp; TurnoverRatio%; DividendRatio% 量级; Weight% Σ≈100
  A Factor 后复权 Close×Factor; MarketCap≈Close×Capitalization
  A NPM≈NetProfit/OpRev×100; Roa≈NetProfit/TotalAssets×100（中位误差很小）
  A Roe 为 % 口径（≈NpParent/归母权益×100，中位接近；存在极端值拉低相关）
  A IsSuspend∈DailyBar; Minute 无 Return; QuoteTime UTC; 240 根
  US Ret/Ret_Intra/Ret_Overnight/Vwap_Close_Dist 恒等式
  US dividend_yield & ROE/ROA 小数; Industry/Status EMPTY; Components 无 Weight
  US Amount==VWAP×Volume（两者皆非空时 100%）
  US Upper_Shadow_Ratio == (High-max(Open,Close))/(High-Low)（H=L 时为 0）
  US Capital 双 schema; Income/CF timeframe 混存; Valuation 49 files

本次更正（旧文档写错/写满）:
  1) High_Low_Ratio = (High-Low)/Low   ← 不是 (High-Low)/Low
  2) Amount=VWAP×Volume 仅在非空时精确；约 3.35% 行两者同为 null
  3) （此前）Factor 后复权 Close×Factor ← 已在 C19 更正
```
### C21. 三次全量审计清单落地（2026-08-08）

#### 仍成立的关键恒等式（再确认）
```text
A: Return/10000; TurnoverRatio=Vol/CircCap*100; MarketCap=Close*Capitalization
   CirculatingMarketCap=Close*CirculatingCap; FreeMarketCap=Close*FreeCap
   AMarketCap=Close*ACap; Factor后复权 Close*Factor; Weight_000300 Σ≈100
   NPM≈NetProfit/OpRev*100; Roa≈NetProfit/TotalAssets*100
   现金流: NetOperate+NetInvest+NetFinance(+FX) ≈ CashEquivalentIncrease（样本中位误差~0）
   TotalAssets == TotalSheetOwnerEquities
   涨跌停: 主板 High/Pre≈1.10 Low/Pre≈0.90; 创业/科创 1.20/0.80
   ST/*ST: High/Pre≈1.05 Low/Pre≈0.95（约±5%）
US: Ret/Ret_Intra/Ret_Overnight/High_Low=(H-L)/Low/UpperShadow/Amount(非空)/AdjFactor≡aux
```

#### 本轮新补事实
| 项 | 结论 |
|----|------|
| US `Volatility_20d` | 抽查 2026-05-12 / 06-22 / 07-27 **全日 100% null** → **当前不可用** |
| US `StockList.type` | 2016-06-03 / 2020-06-03 / 2024-06-03 / 2024-12-31 **均为仅 CS**；ETF 在 ETF* 表 |
| US `PreClose`/`Ret` 缺失 | 样本日约 1.92%；缺失 ticker 多为权证/单元后缀（`W`/`U`/`WS`/`R` 等），不全在 StockList |
| US `StockBalance` 单文件仅数行 | **正常**（E2 按 period_end 的事件文件，不是全市场截面） |
| US `cash`（Valuation） | 量级像比率（p50≈0.43），与 current/quick 相关≈0.50；**勿当现金余额 USD** |
| A `see` ST 涨跌停 | 实测 ST/*ST 中位约 ±5%（以 HighLimit/LowLimit 为准） |
### C22. 细节补全（see_desc / 质押 / 新闻）

```text
US:
  - FIELD_DICTS 原 ~97 处 see_desc → 已改为 USD / 股 / USD/股 / 日期 / 枚举（剩余 0 处单位占位）
  - 新增 §1.1c 财务宽表缺省单位规则
  - FactNews: PIT=published_utc；TradeDate 仅分区；insights 常空
  - StockList/SecurityMaster 抽查以 CS 为主；ETF 走 ETF* 表
A:
  - TopTen: SharePledge/Freeze=股；ShareRatio=%；§4.16 枚举已含性质/股东类别/变动原因/质押非零率
```
### C23. 全量缺口补齐（2026-08-08）

#### C23.1 美股辅助表（路径已核实）

| 表 | 真实路径 | schema | 要点 |
|----|----------|--------|------|
| `is_early_close` | `cos://qs-cold/clean_data/is_early_close/data.parquet`（**单文件**，非按日 hive） | `date:string`, `is_early_close:bool` | ~5937 行；True≈51 日（感恩节次日/七月三日/平安夜等）。半日市勿当全日成交量/波动基准 |
| `is_adj_factor_clamped` | `.../is_adj_factor_clamped/date=YYYY-MM-DD/data.parquet` | `ticker`, `is_adj_factor_clamped` | 阈值 **adj_factor>1e6**；任一日期超限则该 ticker **全历史** True（例 TOT）。True 率约 0~0.013%。True→用收益/log-return，勿用绝对复权价 |
| `universe_daily` | `.../universe_daily/year=YYYY/data.parquet` | `trade_date`, `ticker` | 2024 年文件：~2.66M 行，252 日 × 最多~12.6k ticker。研究宇宙；与 StockList(CS) 交叉 |
| `is_ticker_halt` | `.../is_ticker_halt/minute/date=YYYY-MM-DD/data.parquet` | `ticker`, `timestamp`(**America/New_York**), `is_ticker_halt` | **覆盖极稀**（listing 所见几乎只有个别日期如 2026-06-11）。全日约 391 分钟网格。AAPL/MSFT 样本日 True=0；全市场 True 率可很高（大量非活跃代码）。**不可**当 A股 IsSuspend 的稳定替代 |

`adj_factor` 累积（structure.md）：上市日=1；拆股 `split_to/split_from`；分红 `close_prev/(close_prev-cash_amount)`；调整价=原价×adj_factor。

#### C23.2 美股市本 / Capital 决策树

```text
日频市值/股本暴露（推荐）:
  1) TickerSharesSnapshot.weighted_shares_outstanding × Close
  2) 若缺失 → share_class_shares_outstanding
  3) 勿把 StockCapitalDaily/shares_* 当全市场面板（单日行数极少）
  4) StockCapitalDaily/{date}.parquet = 拆分事件 only（adjustment_type∈{forward_split, reverse_split, stock_dividend}）
  5) Valuation.market_cap 仅 X0 稀疏日可用，不作默认
```

`adjustment_type` 实盘枚举（多文件汇总）：`reverse_split`、`forward_split`、`stock_dividend`（未见其它取值）。

#### C23.3 SecurityMaster vs DailySnap

| | SecurityMaster (STATIC full) | SecurityMasterDailySnap (D1) |
|--|------------------------------|------------------------------|
| 独有 | `start_date`, `end_date` | `approx_mode`, `ticker_cur`, `snap_date` |
| 共有 | ticker/name/type/market/locale/is_adr/exch/cik/figi/ids… | 同左（无起止日） |
| type | 样本全 CS | 样本全 CS |
| approx_mode | — | 样本恒为 `stable_id_presence_v1` |

#### C23.4 FactNews vs raw news

- clean `FactNews` 字段 = catalog raw 新闻字段 **+ `TradeDate` 分区列**。
- raw-only 差集：**无**（clean 已覆盖 raw 15 列）。
- PIT 仍用 `published_utc`；`insights` 常全空；`keywords` 缺失可过半。

#### C23.5 A股 IndexList `.CSI` / `.SH`

- IndexList 后缀分布例：CSI≈772，SZ≈352，SH≈192。
- **同一 6 位代码可并存 `.CSI` 与 `.SH`**（如 `000001.CSI` 与 `000001.SH`），名称略异但样本日 Close/Return **可完全相同**。
- IndexConstituent 的 `IndexSymbol` 以 `.SH/.SZ` 为主，也有 `.CSI`（如 `932000.CSI`）。
- **规则**：成分/权重对齐用 Constituent 里的 `IndexSymbol`；行情用同一后缀；禁止把 `.CSI` 与 `.SH` 当两个指数加权。

#### C23.6 A股分红 PIT 推荐默认

```text
default_policy = reject_as_strict_pit   # 无 PubDate
if research.allow_ex_date_impact:
    align on ExDividendDate, label=effective_time_only, document lookahead risk
never invent PubDate from UpdateTime
```

#### C23.7 财报累计 → 单季（A股）伪代码

```python
# 中国季报多为「年初至报告期累计」
# period_end: Q1=03-31, H1=06-30, Q3=09-30, FY=12-31
# 单季值 = 本期累计 - 上期累计（同 Symbol, 同 fiscal year）
# 同一 ReportPeriodEndDate 多 PubDate：取 PubDate<=信号日的最新修订后再差分

def to_single_quarter(cum_df, value_col, asof_pubdate=None):
    df = cum_df.copy()
    if asof_pubdate is not None:
        df = df[df["PubDate"] <= asof_pubdate]
    df = df.sort_values(["Symbol", "ReportPeriodEndDate", "PubDate"])
    # 每 (Symbol, period_end) 取最新修订
    df = df.groupby(["Symbol", "ReportPeriodEndDate"], as_index=False).tail(1)
    df["fy"] = df["ReportPeriodEndDate"].dt.year
    df["mmdd"] = df["ReportPeriodEndDate"].dt.strftime("%m-%d")
    order = {"03-31": 1, "06-30": 2, "09-30": 3, "12-31": 4}
    df["qord"] = df["mmdd"].map(order)
    df = df.dropna(subset=["qord"]).sort_values(["Symbol", "fy", "qord"])
    df["prev"] = df.groupby(["Symbol", "fy"])[value_col].shift(1)
    df["single"] = df[value_col].where(df["qord"] == 1, df[value_col] - df["prev"])
    # 差分炸裂（符号反常且 |single| > 5*|cum|）→ NaN
    bad = (df["qord"] > 1) & df["single"].notna() & (df["single"].abs() > 5 * df[value_col].abs().clip(lower=1))
    df.loc[bad, "single"] = float("nan")
    return df
```

#### C23.8 A股 Roe winsorize 建议

样本公告日分位（%）：p01≈−114，p05≈−21，p50≈0.6，p95≈4.7，p99≈17。  
推荐截面：`winsorize(Roe, 0.05, 0.95)` 或 MAD；勿用未截断 Roe 做中性化。

#### C23.9 跨市场日历 / FX / 半日市

```text
- COS 无 FX 表 → 禁止把 CNY 与 USD 金额/市值直接 concat 做统一截面
- 交易日历分市场：A=Calendar.IsTradeDay；US=Calendar.is_trading_day
- 对齐多市场日期：inner join 双方交易日，或按研究日历 asof；不要假设同一自然日两边都开市
- 美国半日市：is_early_close=True 时，成交量/振幅/分钟因子需降权或剔除
- 美股分钟时间戳：halt 表为 America/New_York；A股分钟 QuoteTime 为 UTC（+8=上海）
- 夏令时：美东时间钟点会变；用 tz-aware timestamp，勿写死 UTC 偏移
```

#### C23.10 其它字段差集

| 对比 | 结论 |
|------|------|
| A ETFDailyBar vs StockDailyBar | **列集合完全一致**（同 Return bp / Factor / 涨跌停） |
| A TopTen vs FloatTopTen | Top 多 `SharePledge`/`ShareFreeze`；其余同构 |
| US Valuation vs Indicator | Valuation 多 `Volatility_20d`（当前全空不可用） |

#### C23.11 美股分红 currency（实盘抽查）

2024 五个样本除息日合计 1595 行：`USD`≈87.8%，其余约 **12.2%** 非美元（CAD/GBP/HKD/EUR/ZAR/THB/IDR/DKK/…）。  
**规则**：拼入美元因子截面前，非 `USD` 行要么乘 FX（**COS 无 FX 表**），要么 **剔除 / 标 unavailable**；禁止把 `cash_amount` 当美元。

#### C24. P1/P2 缺口关闭清单（2026-08-08）

| # | 原缺口 | 状态 | 落点 |
|--:|--------|------|------|
| 1 | 辅助表路径/字段 | DONE | US 表节 + C23.1 |
| 2 | clamp 长历史率 | DONE | C23.1 / is_adj_factor_clamped.note_rate |
| 3 | SM vs DailySnap 差集 | DONE | C23.3 |
| 4 | adjustment_type 枚举 | DONE | C23.2 / Capital |
| 5 | FactNews vs raw | DONE | C23.4 |
| 6 | A 分红 PIT 默认 | DONE | C23.6 / §4.17 |
| 7 | 累计→单季伪代码 | DONE | C23.7 / §4.18 |
| 8 | IndexList CSI/SH | DONE | C23.5 |
| 9 | FX/日历菜谱 | DONE | C23.9 |
| 10 | 半日市/夏令时 | DONE | C23.9 / is_early_close |
| 11 | 机器可读 registry | DONE | 各核心字典文末 MACHINE_FIELD_REGISTRY |
| 12 | 短中文说明加厚 | DONE | 本轮字段中文扩写 |
| 13 | 股本决策树 | DONE | C23.2 |
| 14 | Val vs Indicator | DONE | C23.10 |
| 15 | 分红非 USD | DONE | C23.11 |
| 16 | Roe winsor 分位 | DONE | C23.8 |
| 17 | ETF/TopTen 差集 | DONE | C23.10 |
| 18 | catalog 并入核心字典 | DONE | catalog/JSON 已合并进 A+US 两文件后删除 |

#### C25. 第二轮审计补齐（2026-08-08）

| 项 | 结论 |
|----|------|
| `adj_factor` 公式 | 拆股 `to/from`；分红 `close_prev/(close_prev-cash)`；调整价=原价×因子（structure.md） |
| `is_adj_factor_clamped` | 阈值 1e6；ticker 级全历史打标；例 TOT；True→禁绝对复权价长窗口 |
| US 表 `vs_ashare` | Calendar/ETF*/Ticker*/Income/CashFlow/Indicator/FactNews/adj_factor/halt 等已补交叉指针 |
| 分红 `cash_amount` | 单位随 `currency`；2024 抽查非 USD≈12.2% |
| 假阳性 | 「前复权=Close×Factor」实为 `Close×Factor/Factor_asof` 正确公式，非错误 |

#### C26. 第三轮审计补齐（2026-08-08）

| 项 | 结论 |
|----|------|
| 分红 `distribution_type` | 实盘 `recurring`/`irregular`/`special` |
| 分红 `cash_amount`/`split_adjusted` | 随 currency；Bridge C12 已改正「恒 USD」表述 |
| Capital `shares_` | 同日同 Ticker 可多行；优先 SharesSnapshot |
| 辅助表 sample | early_close / universe_daily / clamp(TOT) 已补样本 |

#### C27. 交付收敛（2026-08-08）

最终只维护两份核心 MD：
- `/home/shw/COS_ashare_lqtp_data_dictionary.md`
- `/home/shw/COS_us_massive_data_dictionary.md`

原 `COS_clean_data_catalog.md`（新闻专章等）与 `COS_field_registry.json` 已分别并入：美股 `NEWS_RAWDATA` + 双方文末 `MACHINE_FIELD_REGISTRY`。本 Bridge 节在两文件中保持一致。

## MACHINE_FIELD_REGISTRY（机器可读；本市场）

```json
{
  "doc": "COS_us_massive_data_dictionary",
  "market": "us",
  "version": "2026-08-08",
  "hard_rules": {
    "return": "decimal",
    "adj": "Close*AdjFactor",
    "id": "Ticker",
    "asof": "filing_date"
  },
  "tables": [
    {
      "table": "Calendar",
      "model": "STATIC",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/Calendar/full.parquet",
      "n_fields": 2,
      "fields": [
        {
          "ord": 1,
          "name": "trade_date",
          "type": "date32[day]",
          "zh": "交易日（辅助表/宇宙对齐键）",
          "unit": "date32[day]",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "is_trading_day",
          "type": "bool",
          "zh": "是否美股交易日",
          "unit": "bool",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "StockDailyBar",
      "model": "D1",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockDailyBar/{YYYY-MM-DD}.parquet",
      "n_fields": 17,
      "fields": [
        {
          "ord": 1,
          "name": "Ticker",
          "type": "string",
          "zh": "美股代码（无交易所后缀，如 AAPL、BRK.B）",
          "unit": "string",
          "caution": "对应 A股 `Symbol`（带 .SH/.SZ）；禁止直接 concat。",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "timestamp[ns]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "Open",
          "type": "double",
          "zh": "未复权开盘价（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "High",
          "type": "double",
          "zh": "未复权最高价（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "Low",
          "type": "double",
          "zh": "未复权最低价（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "Close",
          "type": "double",
          "zh": "未复权收盘价（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "Volume",
          "type": "double",
          "zh": "成交量（股，double）",
          "unit": "股",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "AdjFactor",
          "type": "double",
          "zh": "后复权累积因子（调整价≈原价×AdjFactor）",
          "unit": "无量纲",
          "caution": "A股 Factor 同样是后复权乘数（Close×Factor）；与 AdjFactor 公式同类但仍禁止混列。若 clamped=True 慎用绝对复权价。",
          "null_pct": "0.0"
        },
        {
          "ord": 9,
          "name": "PreClose",
          "type": "double",
          "zh": "因子调整后的前收（供 Ret）",
          "unit": "USD/股",
          "null_pct": "1.92"
        },
        {
          "ord": 10,
          "name": "Ret",
          "type": "double",
          "zh": "日收益率（小数，0.01=1%）",
          "unit": "小数",
          "caution": "A股对应 Return 是 bp；美股 Ret 不要 /10000。",
          "null_pct": "1.92"
        },
        {
          "ord": 11,
          "name": "Ret_Intra",
          "type": "double",
          "zh": "日内收益率 ≈(Close-Open)/Open",
          "unit": "小数",
          "null_pct": "0.0"
        },
        {
          "ord": 12,
          "name": "Ret_Overnight",
          "type": "double",
          "zh": "隔夜收益率（开盘相对前收）",
          "unit": "小数",
          "caution": "**=`(Open-PreClose)/PreClose`**（已验证）。",
          "null_pct": "1.92"
        },
        {
          "ord": 13,
          "name": "High_Low_Ratio",
          "type": "double",
          "zh": "振幅比 (High−Low)/Low",
          "unit": "小数",
          "caution": "**精确** `(High-Low)/Low`（2026-08-08 验证 match=100%）。不是 /Close，也不是 /PreClose。",
          "null_pct": "0.0"
        },
        {
          "ord": 14,
          "name": "Upper_Shadow_Ratio",
          "type": "double",
          "zh": "上影线占比",
          "unit": "小数",
          "caution": "**=`(High-max(Open,Close))/(High-Low)`**；H=L 时为 0。",
          "null_pct": "0.0"
        },
        {
          "ord": 15,
          "name": "VWAP",
          "type": "double",
          "zh": "成交量加权均价 VWAP（USD/股）",
          "unit": "USD/股",
          "caution": "A股列名是 `Vwap`（仅大小写不同）。",
          "null_pct": "3.35"
        },
        {
          "ord": 16,
          "name": "Amount",
          "type": "double",
          "zh": "成交金额（USD）",
          "unit": "USD",
          "null_pct": "3.35"
        },
        {
          "ord": 17,
          "name": "Vwap_Close_Dist",
          "type": "double",
          "zh": "VWAP 相对收盘偏离 (VWAP-Close)/Close",
          "unit": "小数",
          "null_pct": "3.35"
        }
      ]
    },
    {
      "table": "StockList",
      "model": "D1",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockList/{YYYY-MM-DD}.parquet",
      "n_fields": 9,
      "fields": [
        {
          "ord": 1,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "timestamp[ns]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "Symbol",
          "type": "string",
          "zh": "代码别名列（美股语义同 Ticker，≠A股 Symbol）",
          "unit": "string",
          "caution": "美股 Symbol≠A股 Symbol。",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "name",
          "type": "string",
          "zh": "证券英文名称",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "type",
          "type": "string",
          "zh": "证券类型（CS/ETF/ETN/ETP/ADRC 等）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "market",
          "type": "string",
          "zh": "市场（通常 stocks）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "locale",
          "type": "string",
          "zh": "地区（us）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "last_updated_utc",
          "type": "string",
          "zh": "上游更新时间（ISO）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "delisted_utc",
          "type": "string",
          "zh": "退市时间（ISO，未退市为空）",
          "unit": "string",
          "caution": "样本缺失 80.4%.",
          "null_pct": "80.4"
        },
        {
          "ord": 9,
          "name": "UpdateTime",
          "type": "timestamp[us, tz=UTC]",
          "zh": "清洗管线产出时间（UTC）",
          "unit": "timestamp[us, tz=UTC]",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "ETFDailyBar",
      "model": "D1",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/ETFDailyBar/{YYYY-MM-DD}.parquet",
      "n_fields": 17,
      "fields": [
        {
          "ord": 1,
          "name": "Ticker",
          "type": "string",
          "zh": "美股代码（无交易所后缀，如 AAPL、BRK.B）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "timestamp[ns]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "Open",
          "type": "double",
          "zh": "未复权开盘价（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "High",
          "type": "double",
          "zh": "未复权最高价（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "Low",
          "type": "double",
          "zh": "未复权最低价（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "Close",
          "type": "double",
          "zh": "未复权收盘价（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "Volume",
          "type": "double",
          "zh": "成交量（股，double）",
          "unit": "股",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "AdjFactor",
          "type": "double",
          "zh": "后复权累积因子（调整价≈原价×AdjFactor）",
          "unit": "无量纲",
          "caution": "A股 Factor 同样是后复权乘数（Close×Factor）；与 AdjFactor 公式同类但仍禁止混列。若 clamped=True 慎用绝对复权价。",
          "null_pct": "0.0"
        },
        {
          "ord": 9,
          "name": "PreClose",
          "type": "double",
          "zh": "因子调整后的前收（供 Ret）",
          "unit": "USD/股",
          "null_pct": "0.32"
        },
        {
          "ord": 10,
          "name": "Ret",
          "type": "double",
          "zh": "日收益率（小数，0.01=1%）",
          "unit": "小数",
          "caution": "A股对应 Return 是 bp；美股 Ret 不要 /10000。",
          "null_pct": "0.32"
        },
        {
          "ord": 11,
          "name": "Ret_Intra",
          "type": "double",
          "zh": "日内收益率 ≈(Close-Open)/Open",
          "unit": "小数",
          "null_pct": "0.0"
        },
        {
          "ord": 12,
          "name": "Ret_Overnight",
          "type": "double",
          "zh": "隔夜收益率（开盘相对前收）",
          "unit": "小数",
          "caution": "**=`(Open-PreClose)/PreClose`**（已验证）。",
          "null_pct": "0.32"
        },
        {
          "ord": 13,
          "name": "High_Low_Ratio",
          "type": "double",
          "zh": "振幅比 (High−Low)/Low",
          "unit": "小数",
          "caution": "**精确** `(High-Low)/Low`（2026-08-08 验证 match=100%）。不是 /Close，也不是 /PreClose。",
          "null_pct": "0.0"
        },
        {
          "ord": 14,
          "name": "Upper_Shadow_Ratio",
          "type": "double",
          "zh": "上影线占比",
          "unit": "小数",
          "caution": "**=`(High-max(Open,Close))/(High-Low)`**；H=L 时为 0。",
          "null_pct": "0.0"
        },
        {
          "ord": 15,
          "name": "VWAP",
          "type": "double",
          "zh": "成交量加权均价 VWAP（USD/股）",
          "unit": "USD/股",
          "null_pct": "9.64"
        },
        {
          "ord": 16,
          "name": "Amount",
          "type": "double",
          "zh": "成交金额（USD）",
          "unit": "USD",
          "null_pct": "9.64"
        },
        {
          "ord": 17,
          "name": "Vwap_Close_Dist",
          "type": "double",
          "zh": "VWAP 相对收盘偏离 (VWAP-Close)/Close",
          "unit": "小数",
          "null_pct": "9.64"
        }
      ]
    },
    {
      "table": "ETFList",
      "model": "STATIC",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/ETFList/full.parquet",
      "n_fields": 14,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "name",
          "type": "string",
          "zh": "证券英文名称",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "market",
          "type": "string",
          "zh": "市场（通常 stocks）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "locale",
          "type": "string",
          "zh": "地区（us）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "primary_exchange",
          "type": "string",
          "zh": "主交易所 MIC（XNAS/XNYS/...）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "type",
          "type": "string",
          "zh": "证券类型（CS/ETF/ETN/ETP/ADRC 等）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "active",
          "type": "bool",
          "zh": "是否活跃",
          "unit": "bool",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "currency_name",
          "type": "string",
          "zh": "币种名（usd）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 9,
          "name": "cik",
          "type": "string",
          "zh": "SEC 发行人 CIK",
          "unit": "string",
          "caution": "样本缺失 33.62%.",
          "null_pct": "33.62"
        },
        {
          "ord": 10,
          "name": "composite_figi",
          "type": "string",
          "zh": "Composite FIGI",
          "unit": "string",
          "null_pct": "11.46"
        },
        {
          "ord": 11,
          "name": "share_class_figi",
          "type": "string",
          "zh": "Share Class FIGI",
          "unit": "string",
          "null_pct": "11.46"
        },
        {
          "ord": 12,
          "name": "last_updated_utc",
          "type": "string",
          "zh": "上游更新时间（ISO）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 13,
          "name": "delisted_utc",
          "type": "string",
          "zh": "退市时间（ISO，未退市为空）",
          "unit": "string",
          "caution": "样本缺失 76.11%.",
          "null_pct": "76.11"
        },
        {
          "ord": 14,
          "name": "TradeDate",
          "type": "string",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "string",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "SecurityMaster",
      "model": "STATIC",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/SecurityMaster/full.parquet",
      "n_fields": 16,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "name",
          "type": "string",
          "zh": "证券英文名称",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "type",
          "type": "string",
          "zh": "证券类型（CS/ETF/ETN/ETP/ADRC 等）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "market",
          "type": "string",
          "zh": "市场（通常 stocks）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "locale",
          "type": "string",
          "zh": "地区（us）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "is_adr_asset",
          "type": "int64",
          "zh": "是否 ADR 资产",
          "unit": "string",
          "caution": "主数据属性",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "start_date",
          "type": "timestamp[ns]",
          "zh": "生效/起始日",
          "unit": "timestamp[ns]",
          "caution": "样本缺失 100.0%.",
          "null_pct": "100.0"
        },
        {
          "ord": 8,
          "name": "end_date",
          "type": "timestamp[ns]",
          "zh": "上市/有效区间结束日（退市或失效）",
          "unit": "timestamp[ns]",
          "caution": "样本缺失 100.0%.",
          "null_pct": "100.0"
        },
        {
          "ord": 9,
          "name": "primary_exchange",
          "type": "string",
          "zh": "主交易所 MIC（XNAS/XNYS/...）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 10,
          "name": "cik",
          "type": "string",
          "zh": "SEC 发行人 CIK",
          "unit": "string",
          "null_pct": "2.54"
        },
        {
          "ord": 11,
          "name": "composite_figi",
          "type": "string",
          "zh": "Composite FIGI",
          "unit": "string",
          "caution": "样本缺失 40.63%.",
          "null_pct": "40.63"
        },
        {
          "ord": 12,
          "name": "share_class_figi",
          "type": "string",
          "zh": "Share Class FIGI",
          "unit": "string",
          "caution": "样本缺失 40.63%.",
          "null_pct": "40.63"
        },
        {
          "ord": 13,
          "name": "security_id",
          "type": "string",
          "zh": "内部证券 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 14,
          "name": "issuer_id",
          "type": "string",
          "zh": "发行人 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 15,
          "name": "listing_id",
          "type": "string",
          "zh": "上市 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 16,
          "name": "id_source",
          "type": "string",
          "zh": "ID 来源",
          "unit": "string",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "SecurityMasterDailySnap",
      "model": "D1",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/SecurityMasterDailySnap/{YYYY-MM-DD}.parquet",
      "n_fields": 17,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "security_id",
          "type": "string",
          "zh": "内部证券 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "approx_mode",
          "type": "string",
          "zh": "ID 近似/稳定模式（样本恒 stable_id_presence_v1）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "ticker_cur",
          "type": "string",
          "zh": "快照日当前 ticker",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "name",
          "type": "string",
          "zh": "证券英文名称",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "type",
          "type": "string",
          "zh": "证券类型（CS/ETF/ETN/ETP/ADRC 等）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "market",
          "type": "string",
          "zh": "市场（通常 stocks）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "locale",
          "type": "string",
          "zh": "地区（us）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 9,
          "name": "is_adr_asset",
          "type": "int64",
          "zh": "是否 ADR 资产",
          "unit": "string",
          "caution": "主数据属性",
          "null_pct": "0.0"
        },
        {
          "ord": 10,
          "name": "primary_exchange",
          "type": "string",
          "zh": "主交易所 MIC（XNAS/XNYS/...）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 11,
          "name": "cik",
          "type": "string",
          "zh": "SEC 发行人 CIK",
          "unit": "string",
          "null_pct": "0.15"
        },
        {
          "ord": 12,
          "name": "composite_figi",
          "type": "string",
          "zh": "Composite FIGI",
          "unit": "string",
          "null_pct": "15.59"
        },
        {
          "ord": 13,
          "name": "share_class_figi",
          "type": "string",
          "zh": "Share Class FIGI",
          "unit": "string",
          "null_pct": "15.59"
        },
        {
          "ord": 14,
          "name": "issuer_id",
          "type": "string",
          "zh": "发行人 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 15,
          "name": "listing_id",
          "type": "string",
          "zh": "上市 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 16,
          "name": "id_source",
          "type": "string",
          "zh": "ID 来源",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 17,
          "name": "snap_date",
          "type": "string",
          "zh": "SecurityMaster 日快照对应交易日",
          "unit": "string",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "TickerMap",
      "model": "STATIC",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/TickerMap/full.parquet",
      "n_fields": 4,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "permanent_id",
          "type": "string",
          "zh": "永久 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "effective_date",
          "type": "string",
          "zh": "公司行动/调整生效日",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "event_type",
          "type": "string",
          "zh": "事件类型",
          "unit": "string",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "TickerAlias",
      "model": "EMPTY",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/TickerAlias/full.parquet",
      "n_fields": 2,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "null",
          "zh": "美股代码",
          "unit": "string",
          "caution": "EMPTY 表勿用",
          "null_pct": "nan"
        },
        {
          "ord": 2,
          "name": "ticker_normalized",
          "type": "null",
          "zh": "归一化 ticker（EMPTY）",
          "unit": "string",
          "caution": "EMPTY 表勿用",
          "null_pct": "nan"
        }
      ]
    },
    {
      "table": "TickerSharesSnapshot",
      "model": "D1",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/TickerSharesSnapshot/{YYYY-MM-DD}.parquet",
      "n_fields": 8,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "cik",
          "type": "string",
          "zh": "SEC 发行人 CIK",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "share_class_figi",
          "type": "string",
          "zh": "Share Class FIGI",
          "unit": "string",
          "null_pct": "24.44"
        },
        {
          "ord": 4,
          "name": "share_class_shares_outstanding",
          "type": "double",
          "zh": "该股份类别流通/在外股本",
          "unit": "股",
          "caution": "股本股数",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "weighted_shares_outstanding",
          "type": "double",
          "zh": "加权平均股本",
          "unit": "股",
          "caution": "股本股数",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "TradeDate",
          "type": "string",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "string",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "source",
          "type": "string",
          "zh": "数据来源标记",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "error",
          "type": "null",
          "zh": "错误信息（如有）",
          "unit": "string",
          "caution": "样本缺失 100.0%.",
          "null_pct": "100.0"
        }
      ]
    },
    {
      "table": "StockIndicesComponents",
      "model": "D1",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockIndicesComponents/{YYYY-MM-DD}.parquet",
      "n_fields": 3,
      "fields": [
        {
          "ord": 1,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "timestamp[ns]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "IndexName",
          "type": "string",
          "zh": "指数名称",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "Symbol",
          "type": "string",
          "zh": "代码别名列（美股语义同 Ticker，≠A股 Symbol）",
          "unit": "string",
          "caution": "美股 Symbol≠A股 Symbol。",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "StockBalance",
      "model": "E2",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockBalance/{period_end}.parquet",
      "n_fields": 38,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "cik",
          "type": "string",
          "zh": "SEC 发行人 CIK",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "period_end",
          "type": "string",
          "zh": "报告期期末（常作文件名；≠PIT）",
          "unit": "string",
          "caution": "文件名常用此字段；PIT 必须用 filing_date。",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "filing_date",
          "type": "timestamp[ns]",
          "zh": "申报/公告可知日（财务 asof 键）",
          "unit": "timestamp[ns]",
          "caution": "财务 asof 键。",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "fiscal_quarter",
          "type": "int64",
          "zh": "财季编号（1–4；配合 fiscal_year）",
          "unit": "季度序号 1-4",
          "caution": "元数据；非金额",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "fiscal_year",
          "type": "int64",
          "zh": "财年（整数年）",
          "unit": "年",
          "caution": "元数据；非金额",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "timeframe",
          "type": "string",
          "zh": "报告时间框",
          "unit": "enum string",
          "caution": "**必筛**：`quarterly` / `annual` / `trailing_twelve_months`。",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "cash_and_equivalents",
          "type": "double",
          "zh": "现金及等价物",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 9,
          "name": "receivables",
          "type": "double",
          "zh": "应收账款",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 10,
          "name": "inventories",
          "type": "double",
          "zh": "存货（资产负债表，USD）",
          "unit": "USD",
          "caution": "样本缺失 100.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "100.0"
        },
        {
          "ord": 11,
          "name": "other_current_assets",
          "type": "double",
          "zh": "其他流动资产",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 12,
          "name": "total_current_assets",
          "type": "double",
          "zh": "流动资产合计",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 13,
          "name": "property_plant_equipment_net",
          "type": "double",
          "zh": "固定资产净额 PP&E",
          "unit": "USD",
          "caution": "样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "60.0"
        },
        {
          "ord": 14,
          "name": "goodwill",
          "type": "double",
          "zh": "商誉（资产负债表，USD）",
          "unit": "USD",
          "caution": "样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "60.0"
        },
        {
          "ord": 15,
          "name": "other_assets",
          "type": "double",
          "zh": "其他资产",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 16,
          "name": "total_assets",
          "type": "double",
          "zh": "总资产（USD）",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 17,
          "name": "accounts_payable",
          "type": "double",
          "zh": "应付账款",
          "unit": "USD",
          "caution": "样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "60.0"
        },
        {
          "ord": 18,
          "name": "debt_current",
          "type": "double",
          "zh": "流动负债中的有息债",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 19,
          "name": "accrued_and_other_current_liabilities",
          "type": "double",
          "zh": "应计及其他流动负债",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 20,
          "name": "total_current_liabilities",
          "type": "double",
          "zh": "流动负债合计",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 21,
          "name": "long_term_debt_and_capital_lease_obligations",
          "type": "double",
          "zh": "长期债务及融资租赁",
          "unit": "USD",
          "caution": "样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "60.0"
        },
        {
          "ord": 22,
          "name": "other_noncurrent_liabilities",
          "type": "double",
          "zh": "其他非流动负债",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 23,
          "name": "total_liabilities",
          "type": "double",
          "zh": "总负债（USD）",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 24,
          "name": "commitments_and_contingencies",
          "type": "double",
          "zh": "承诺与或有事项",
          "unit": "USD",
          "caution": "样本缺失 40.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "40.0"
        },
        {
          "ord": 25,
          "name": "common_stock",
          "type": "double",
          "zh": "普通股股本",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "20.0"
        },
        {
          "ord": 26,
          "name": "accumulated_other_comprehensive_income",
          "type": "double",
          "zh": "累计其他综合收益 AOCI",
          "unit": "USD",
          "caution": "样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "60.0"
        },
        {
          "ord": 27,
          "name": "retained_earnings_deficit",
          "type": "double",
          "zh": "留存收益/累计亏损",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 28,
          "name": "other_equity",
          "type": "double",
          "zh": "其他权益",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 29,
          "name": "total_equity_attributable_to_parent",
          "type": "double",
          "zh": "归母权益",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 30,
          "name": "noncontrolling_interest",
          "type": "double",
          "zh": "少数股东损益/权益",
          "unit": "USD",
          "caution": "样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "60.0"
        },
        {
          "ord": 31,
          "name": "total_equity",
          "type": "double",
          "zh": "股东权益合计",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 32,
          "name": "total_liabilities_and_equity",
          "type": "double",
          "zh": "负债和权益总计",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 33,
          "name": "additional_paid_in_capital",
          "type": "double",
          "zh": "资本公积 APIC",
          "unit": "USD",
          "caution": "样本缺失 80.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "80.0"
        },
        {
          "ord": 34,
          "name": "treasury_stock",
          "type": "double",
          "zh": "库存股（常为负值或绝对值约定，USD）",
          "unit": "USD",
          "caution": "样本缺失 80.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "80.0"
        },
        {
          "ord": 35,
          "name": "short_term_investments",
          "type": "double",
          "zh": "短期投资",
          "unit": "USD",
          "caution": "样本缺失 100.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "100.0"
        },
        {
          "ord": 36,
          "name": "intangible_assets_net",
          "type": "double",
          "zh": "无形资产净额",
          "unit": "USD",
          "caution": "样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "60.0"
        },
        {
          "ord": 37,
          "name": "preferred_stock",
          "type": "double",
          "zh": "优先股权益（USD）",
          "unit": "USD",
          "caution": "样本缺失 100.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "100.0"
        },
        {
          "ord": 38,
          "name": "deferred_revenue_current",
          "type": "double",
          "zh": "递延收入（流动）",
          "unit": "USD",
          "caution": "样本缺失 60.0%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "60.0"
        }
      ]
    },
    {
      "table": "StockIncome",
      "model": "E2",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockIncome/{period_end}.parquet",
      "n_fields": 34,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "cik",
          "type": "string",
          "zh": "SEC 发行人 CIK",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "period_end",
          "type": "string",
          "zh": "报告期期末（常作文件名；≠PIT）",
          "unit": "string",
          "caution": "文件名常用此字段；PIT 必须用 filing_date。",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "filing_date",
          "type": "timestamp[ns]",
          "zh": "申报/公告可知日（财务 asof 键）",
          "unit": "timestamp[ns]",
          "caution": "财务 asof 键。",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "fiscal_quarter",
          "type": "int64",
          "zh": "财季编号（1–4；配合 fiscal_year）",
          "unit": "季度序号 1-4",
          "caution": "元数据；非金额",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "fiscal_year",
          "type": "int64",
          "zh": "财年（整数年）",
          "unit": "年",
          "caution": "元数据；非金额",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "timeframe",
          "type": "string",
          "zh": "报告时间框",
          "unit": "enum string",
          "caution": "**必筛**：`quarterly` / `annual` / `trailing_twelve_months`。",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "revenue",
          "type": "double",
          "zh": "营业收入",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "23.53"
        },
        {
          "ord": 9,
          "name": "cost_of_revenue",
          "type": "double",
          "zh": "营业成本",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 10,
          "name": "gross_profit",
          "type": "double",
          "zh": "毛利（利润表，USD）",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "23.53"
        },
        {
          "ord": 11,
          "name": "selling_general_administrative",
          "type": "double",
          "zh": "销售及管理费用 SG&A",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "14.71"
        },
        {
          "ord": 12,
          "name": "research_development",
          "type": "double",
          "zh": "研发费用 R&D",
          "unit": "USD",
          "caution": "样本缺失 69.61%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "69.61"
        },
        {
          "ord": 13,
          "name": "depreciation_depletion_amortization",
          "type": "double",
          "zh": "折旧/折耗/摊销",
          "unit": "USD",
          "caution": "样本缺失 68.63%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "68.63"
        },
        {
          "ord": 14,
          "name": "other_operating_expenses",
          "type": "double",
          "zh": "其他营业费用",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 15,
          "name": "total_operating_expenses",
          "type": "double",
          "zh": "营业费用合计",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "4.9"
        },
        {
          "ord": 16,
          "name": "operating_income",
          "type": "double",
          "zh": "营业利润",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "4.9"
        },
        {
          "ord": 17,
          "name": "interest_expense",
          "type": "double",
          "zh": "利息支出",
          "unit": "USD",
          "caution": "样本缺失 32.35%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "32.35"
        },
        {
          "ord": 18,
          "name": "other_income_expense",
          "type": "double",
          "zh": "其他收支",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 19,
          "name": "total_other_income_expense",
          "type": "double",
          "zh": "其他收支合计",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "6.86"
        },
        {
          "ord": 20,
          "name": "income_before_income_taxes",
          "type": "double",
          "zh": "税前利润",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "4.9"
        },
        {
          "ord": 21,
          "name": "income_taxes",
          "type": "double",
          "zh": "所得税费用（USD）",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "15.69"
        },
        {
          "ord": 22,
          "name": "consolidated_net_income_loss",
          "type": "double",
          "zh": "合并净利润",
          "unit": "timestamp/date",
          "caution": "事件/报告日；非 PIT 时勿与交易日 equi 混用",
          "null_pct": "4.9"
        },
        {
          "ord": 23,
          "name": "noncontrolling_interest",
          "type": "double",
          "zh": "少数股东损益/权益",
          "unit": "USD",
          "caution": "样本缺失 79.41%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "79.41"
        },
        {
          "ord": 24,
          "name": "net_income_loss_attributable_common_shareholders",
          "type": "double",
          "zh": "归母普通股净利润",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "4.9"
        },
        {
          "ord": 25,
          "name": "basic_earnings_per_share",
          "type": "double",
          "zh": "基本每股收益",
          "unit": "USD/股",
          "null_pct": "4.9"
        },
        {
          "ord": 26,
          "name": "diluted_earnings_per_share",
          "type": "double",
          "zh": "稀释每股收益",
          "unit": "USD/股",
          "null_pct": "4.9"
        },
        {
          "ord": 27,
          "name": "basic_shares_outstanding",
          "type": "double",
          "zh": "基本股本",
          "unit": "股",
          "caution": "股本股数",
          "null_pct": "0.0"
        },
        {
          "ord": 28,
          "name": "diluted_shares_outstanding",
          "type": "double",
          "zh": "稀释股本",
          "unit": "股",
          "caution": "股本股数",
          "null_pct": "0.0"
        },
        {
          "ord": 29,
          "name": "ebitda",
          "type": "double",
          "zh": "EBITDA",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 30,
          "name": "discontinued_operations",
          "type": "double",
          "zh": "终止经营",
          "unit": "USD",
          "caution": "样本缺失 91.18%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "91.18"
        },
        {
          "ord": 31,
          "name": "interest_income",
          "type": "double",
          "zh": "利息收入",
          "unit": "USD",
          "caution": "样本缺失 40.2%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "40.2"
        },
        {
          "ord": 32,
          "name": "preferred_stock_dividends_declared",
          "type": "double",
          "zh": "优先股股息",
          "unit": "USD",
          "caution": "样本缺失 95.1%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "95.1"
        },
        {
          "ord": 33,
          "name": "equity_in_affiliates",
          "type": "double",
          "zh": "联营收益",
          "unit": "USD",
          "caution": "样本缺失 91.18%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "91.18"
        },
        {
          "ord": 34,
          "name": "extraordinary_items",
          "type": "double",
          "zh": "非经常项目",
          "unit": "USD",
          "caution": "样本缺失 96.08%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "96.08"
        }
      ]
    },
    {
      "table": "StockCashFlow",
      "model": "E2",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockCashFlow/{period_end}.parquet",
      "n_fields": 32,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "cik",
          "type": "string",
          "zh": "SEC 发行人 CIK",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "period_end",
          "type": "string",
          "zh": "报告期期末（常作文件名；≠PIT）",
          "unit": "string",
          "caution": "文件名常用此字段；PIT 必须用 filing_date。",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "filing_date",
          "type": "timestamp[ns]",
          "zh": "申报/公告可知日（财务 asof 键）",
          "unit": "timestamp[ns]",
          "caution": "财务 asof 键。",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "fiscal_quarter",
          "type": "int64",
          "zh": "财季编号（1–4；配合 fiscal_year）",
          "unit": "季度序号 1-4",
          "caution": "元数据；非金额",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "fiscal_year",
          "type": "int64",
          "zh": "财年（整数年）",
          "unit": "年",
          "caution": "元数据；非金额",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "timeframe",
          "type": "string",
          "zh": "报告时间框",
          "unit": "enum string",
          "caution": "**必筛**：`quarterly` / `annual` / `trailing_twelve_months`。",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "net_income",
          "type": "double",
          "zh": "净利润（USD）",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 9,
          "name": "depreciation_depletion_and_amortization",
          "type": "double",
          "zh": "折旧折耗摊销",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "19.8"
        },
        {
          "ord": 10,
          "name": "other_operating_activities",
          "type": "double",
          "zh": "其他经营活动",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 11,
          "name": "change_in_other_operating_assets_and_liabilities_net",
          "type": "double",
          "zh": "营运资本变动净额",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 12,
          "name": "cash_from_operating_activities_continuing_operations",
          "type": "double",
          "zh": "持续经营经营现金流",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 13,
          "name": "net_cash_from_operating_activities",
          "type": "double",
          "zh": "经营活动现金流量净额",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 14,
          "name": "purchase_of_property_plant_and_equipment",
          "type": "double",
          "zh": "购建固定资产（CapEx 流出）",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "23.76"
        },
        {
          "ord": 15,
          "name": "sale_of_property_plant_and_equipment",
          "type": "double",
          "zh": "处置固定资产",
          "unit": "USD",
          "caution": "样本缺失 61.39%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "61.39"
        },
        {
          "ord": 16,
          "name": "other_investing_activities",
          "type": "double",
          "zh": "其他投资活动",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 17,
          "name": "net_cash_from_investing_activities_continuing_operations",
          "type": "double",
          "zh": "持续经营投资现金流",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "3.96"
        },
        {
          "ord": 18,
          "name": "net_cash_from_investing_activities",
          "type": "double",
          "zh": "投资活动现金流量净额",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "3.96"
        },
        {
          "ord": 19,
          "name": "long_term_debt_issuances_repayments",
          "type": "double",
          "zh": "长期债务发行/偿还",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 20,
          "name": "dividends",
          "type": "double",
          "zh": "支付股利",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 21,
          "name": "other_financing_activities",
          "type": "double",
          "zh": "其他筹资活动",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 22,
          "name": "net_cash_from_financing_activities_continuing_operations",
          "type": "double",
          "zh": "持续经营筹资现金流",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 23,
          "name": "net_cash_from_financing_activities",
          "type": "double",
          "zh": "筹资活动现金流量净额",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 24,
          "name": "effect_of_currency_exchange_rate",
          "type": "double",
          "zh": "汇率变动对现金影响",
          "unit": "USD",
          "caution": "样本缺失 44.55%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "44.55"
        },
        {
          "ord": 25,
          "name": "change_in_cash_and_equivalents",
          "type": "double",
          "zh": "现金及等价物净增加额",
          "unit": "USD",
          "caution": "报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "0.0"
        },
        {
          "ord": 26,
          "name": "short_term_debt_issuances_repayments",
          "type": "double",
          "zh": "短期债务发行/偿还",
          "unit": "USD",
          "caution": "样本缺失 78.22%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "78.22"
        },
        {
          "ord": 27,
          "name": "net_cash_from_operating_activities_discontinued_operations",
          "type": "double",
          "zh": "终止经营-经营现金流",
          "unit": "USD",
          "caution": "样本缺失 93.07%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "93.07"
        },
        {
          "ord": 28,
          "name": "net_cash_from_investing_activities_discontinued_operations",
          "type": "double",
          "zh": "终止经营-投资现金流",
          "unit": "USD",
          "caution": "样本缺失 95.05%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "95.05"
        },
        {
          "ord": 29,
          "name": "noncontrolling_interests",
          "type": "double",
          "zh": "少数股东相关",
          "unit": "USD",
          "caution": "样本缺失 91.09%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "91.09"
        },
        {
          "ord": 30,
          "name": "net_cash_from_financing_activities_discontinued_operations",
          "type": "double",
          "zh": "终止经营-筹资现金流",
          "unit": "USD",
          "caution": "样本缺失 99.01%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "99.01"
        },
        {
          "ord": 31,
          "name": "other_cash_adjustments",
          "type": "double",
          "zh": "其他现金调整",
          "unit": "USD",
          "caution": "样本缺失 97.03%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "97.03"
        },
        {
          "ord": 32,
          "name": "income_loss_from_discontinued_operations",
          "type": "double",
          "zh": "终止经营损益",
          "unit": "USD",
          "caution": "样本缺失 93.07%. 报表绝对金额（美元）；NaN=未披露勿填0；先 filter timeframe",
          "null_pct": "93.07"
        }
      ]
    },
    {
      "table": "StockDividend",
      "model": "E2",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockDividend/{ex_div_date}.parquet",
      "n_fields": 13,
      "fields": [
        {
          "ord": 1,
          "name": "id",
          "type": "string",
          "zh": "记录 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "record_date",
          "type": "string",
          "zh": "股权登记日",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "pay_date",
          "type": "timestamp[ns]",
          "zh": "现金分红派息日",
          "unit": "timestamp[ns]",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "declaration_date",
          "type": "string",
          "zh": "分红宣告日（PIT 可用；缺则勿臆造）",
          "unit": "string",
          "null_pct": "0.51"
        },
        {
          "ord": 6,
          "name": "ex_dividend_date",
          "type": "string",
          "zh": "除权除息日",
          "unit": "string",
          "caution": "可含未来日期，回测需截断。",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "frequency",
          "type": "int64",
          "zh": "分红频率",
          "unit": "次/年编码",
          "caution": "12≈月 4≈季 2≈半年 1≈年 0=未知/特殊",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "cash_amount",
          "type": "double",
          "zh": "每股现金分红（币种见 currency）",
          "unit": "随 currency",
          "caution": "**不是恒为 USD**；非 USD 样本常见 split_adjusted 为空",
          "null_pct": "0.0"
        },
        {
          "ord": 9,
          "name": "currency",
          "type": "string",
          "zh": "分红币种（ISO）；非 USD 禁止当美元金额",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 10,
          "name": "distribution_type",
          "type": "string",
          "zh": "分配类型",
          "unit": "string",
          "caution": "实盘枚举：`recurring` / `irregular` / `special`（2024-06-03：584/2/1）",
          "null_pct": "0.0"
        },
        {
          "ord": 11,
          "name": "historical_adjustment_factor",
          "type": "double",
          "zh": "历史复权调整因子",
          "unit": "无量纲",
          "caution": "调整乘数",
          "null_pct": "3.58"
        },
        {
          "ord": 12,
          "name": "split_adjusted_cash_amount",
          "type": "double",
          "zh": "拆股调整后每股分红",
          "unit": "随 currency",
          "caution": "与 cash_amount 同币种；非 USD 时常为 null",
          "null_pct": "3.58"
        },
        {
          "ord": 13,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "timestamp[ns]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "StockCapitalDaily",
      "model": "E2",
      "cos": null,
      "n_fields": 12,
      "fields": [
        {
          "ord": 1,
          "name": "Ticker",
          "type": "string",
          "zh": "美股代码（无交易所后缀，如 AAPL、BRK.B）",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "timestamp[ns]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "pit_basic_shares_outstanding",
          "type": "double",
          "zh": "PIT 基本股本（股）",
          "unit": "股",
          "caution": "单日文件标的很少；非全市场。",
          "null_pct": "27.45"
        },
        {
          "ord": 4,
          "name": "pit_diluted_shares_outstanding",
          "type": "double",
          "zh": "PIT 稀释股本（股）",
          "unit": "股",
          "caution": "与 basic 同文件；同日同 Ticker 可多行，需去重。",
          "null_pct": "27.45"
        },
        {
          "ord": 1,
          "name": "id",
          "type": "string",
          "zh": "事件 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "execution_date",
          "type": "timestamp[ns]",
          "zh": "拆分执行日",
          "unit": "timestamp[ns]",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "split_from",
          "type": "double",
          "zh": "拆分前股数基准（from）",
          "unit": "无量纲",
          "caution": "reverse 时常 > split_to",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "split_to",
          "type": "double",
          "zh": "拆分后股数基准（to）；比率=to/from",
          "unit": "无量纲",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "ticker",
          "type": "string",
          "zh": "标的（小写列名）",
          "unit": "string",
          "caution": "与 shares 文件的 `Ticker` 大小写不同",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "adjustment_type",
          "type": "string",
          "zh": "如 forward_split / reverse_split / stock_dividend",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "historical_adjustment_factor",
          "type": "double",
          "zh": "历史调整因子",
          "unit": "无量纲",
          "caution": "与日线 `AdjFactor` 不是同一列",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "常等于 execution_date",
          "unit": "timestamp[ns]",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "StockValuationDaily",
      "model": "X0",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockValuationDaily/{YYYY-MM-DD}.parquet",
      "n_fields": 24,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "cik",
          "type": "string",
          "zh": "SEC 发行人 CIK",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "price",
          "type": "double",
          "zh": "价格（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "average_volume",
          "type": "double",
          "zh": "平均成交量",
          "unit": "股",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "market_cap",
          "type": "double",
          "zh": "总市值（USD）",
          "unit": "USD",
          "caution": "**X0=日历稀疏**（非必然高缺失）。2026-07-27 样本 null≈16%。勿当全历史；默认市值源仍推 SharesSnapshot×Close。",
          "null_pct": "~16"
        },
        {
          "ord": 6,
          "name": "earnings_per_share",
          "type": "double",
          "zh": "每股收益 EPS（USD/股）",
          "unit": "USD/股",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 7,
          "name": "price_to_earnings",
          "type": "double",
          "zh": "市盈率 PE",
          "unit": "倍数",
          "caution": "样本缺失 80.0%. 表级 X0 稀疏。",
          "null_pct": "80.0"
        },
        {
          "ord": 8,
          "name": "price_to_book",
          "type": "double",
          "zh": "市净率 PB",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 9,
          "name": "price_to_sales",
          "type": "double",
          "zh": "市销率 PS",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 10,
          "name": "price_to_cash_flow",
          "type": "double",
          "zh": "价格/现金流",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 11,
          "name": "price_to_free_cash_flow",
          "type": "double",
          "zh": "价格/自由现金流",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 12,
          "name": "dividend_yield",
          "type": "double",
          "zh": "股息率（小数，0.02=2%；禁止再/100）",
          "unit": "**小数倾向**",
          "caution": "实测 >0 中位≈0.023（≈2.3%）。A股 `DividendRatio` 是 **%**。禁止直接比。",
          "null_pct": "~17"
        },
        {
          "ord": 13,
          "name": "return_on_assets",
          "type": "double",
          "zh": "ROA（小数；禁止再/100）",
          "unit": "**小数倾向**",
          "caution": "中位≈0.007；A股 Indicator.`Roa` 为 **%**。",
          "null_pct": "~0"
        },
        {
          "ord": 14,
          "name": "return_on_equity",
          "type": "double",
          "zh": "ROE（小数；禁止再/100）",
          "unit": "**小数倾向**",
          "caution": "中位≈0.054；A股 Indicator.`Roe` 为 **%**。",
          "null_pct": "~0"
        },
        {
          "ord": 15,
          "name": "debt_to_equity",
          "type": "double",
          "zh": "资产负债/权益比 D/E",
          "unit": "倍数",
          "null_pct": "0.0"
        },
        {
          "ord": 16,
          "name": "current",
          "type": "double",
          "zh": "流动比率 Current Ratio",
          "unit": "倍数",
          "null_pct": "0.0"
        },
        {
          "ord": 17,
          "name": "quick",
          "type": "double",
          "zh": "速动比率 Quick Ratio",
          "unit": "倍数",
          "null_pct": "0.0"
        },
        {
          "ord": 18,
          "name": "cash",
          "type": "double",
          "zh": "现金比率（非现金余额）",
          "unit": "比率",
          "caution": "p50≈0.43，与 current/quick 相关≈0.5；**禁止当 USD 现金余额**。",
          "null_pct": "~0"
        },
        {
          "ord": 19,
          "name": "ev_to_sales",
          "type": "double",
          "zh": "EV/Sales",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 20,
          "name": "ev_to_ebitda",
          "type": "double",
          "zh": "EV/EBITDA",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 21,
          "name": "enterprise_value",
          "type": "double",
          "zh": "企业价值 EV（USD）",
          "unit": "USD",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 22,
          "name": "free_cash_flow",
          "type": "double",
          "zh": "自由现金流（USD）",
          "unit": "USD",
          "null_pct": "20.0"
        },
        {
          "ord": 23,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "timestamp[ns]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。 表级 X0 稀疏。",
          "null_pct": "0.0"
        },
        {
          "ord": 24,
          "name": "Volatility_20d",
          "type": "double",
          "zh": "20 日波动率",
          "unit": "不可用/全空",
          "caution": "**抽查多日 100% null → 当前当作不可用字段**，勿进因子。",
          "null_pct": "100"
        }
      ]
    },
    {
      "table": "StockIndicator",
      "model": "X0",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockIndicator/{YYYY-MM-DD}.parquet",
      "n_fields": 23,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "cik",
          "type": "string",
          "zh": "SEC 发行人 CIK",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "timestamp[ns]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。 表级 X0 稀疏。",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "price",
          "type": "double",
          "zh": "价格（USD/股）",
          "unit": "USD/股",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "average_volume",
          "type": "double",
          "zh": "平均成交量",
          "unit": "股",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "market_cap",
          "type": "double",
          "zh": "总市值（USD）",
          "unit": "USD",
          "caution": "在 Valuation/Indicator 稀疏表中出现时不能当全市场日面板。 样本缺失 80.0%. 表级 X0 稀疏。",
          "null_pct": "80.0"
        },
        {
          "ord": 7,
          "name": "earnings_per_share",
          "type": "double",
          "zh": "每股收益 EPS（USD/股）",
          "unit": "USD/股",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 8,
          "name": "price_to_earnings",
          "type": "double",
          "zh": "市盈率 PE",
          "unit": "倍数",
          "caution": "样本缺失 80.0%. 表级 X0 稀疏。",
          "null_pct": "80.0"
        },
        {
          "ord": 9,
          "name": "price_to_book",
          "type": "double",
          "zh": "市净率 PB",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 10,
          "name": "price_to_sales",
          "type": "double",
          "zh": "市销率 PS",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 11,
          "name": "price_to_cash_flow",
          "type": "double",
          "zh": "价格/现金流",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 12,
          "name": "price_to_free_cash_flow",
          "type": "double",
          "zh": "价格/自由现金流",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 13,
          "name": "dividend_yield",
          "type": "double",
          "zh": "股息率（小数，0.02=2%；禁止再/100）",
          "unit": "**小数**",
          "caution": "与 Valuation 同口径；**禁止 /100**。A股 `DividendRatio` 才是 %。",
          "null_pct": "~17"
        },
        {
          "ord": 14,
          "name": "return_on_assets",
          "type": "double",
          "zh": "ROA（小数；禁止再/100）",
          "unit": "**小数**",
          "caution": "**禁止 /100**；A股 `Roa` 为 %。",
          "null_pct": "~0"
        },
        {
          "ord": 15,
          "name": "return_on_equity",
          "type": "double",
          "zh": "ROE（小数；禁止再/100）",
          "unit": "**小数**",
          "caution": "**禁止 /100**；A股 `Roe` 为 %。",
          "null_pct": "~0"
        },
        {
          "ord": 16,
          "name": "debt_to_equity",
          "type": "double",
          "zh": "资产负债/权益比 D/E",
          "unit": "倍数",
          "null_pct": "0.0"
        },
        {
          "ord": 17,
          "name": "current",
          "type": "double",
          "zh": "流动比率 Current Ratio",
          "unit": "倍数",
          "null_pct": "0.0"
        },
        {
          "ord": 18,
          "name": "quick",
          "type": "double",
          "zh": "速动比率 Quick Ratio",
          "unit": "倍数",
          "null_pct": "0.0"
        },
        {
          "ord": 19,
          "name": "cash",
          "type": "double",
          "zh": "现金比率（非现金余额）",
          "unit": "比率",
          "caution": "与 Valuation 同字段；p50 量级约 0.1–0.5；**禁止当 USD 现金余额**。",
          "null_pct": "0.0"
        },
        {
          "ord": 20,
          "name": "ev_to_sales",
          "type": "double",
          "zh": "EV/Sales",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 21,
          "name": "ev_to_ebitda",
          "type": "double",
          "zh": "EV/EBITDA",
          "unit": "倍数",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 22,
          "name": "enterprise_value",
          "type": "double",
          "zh": "企业价值 EV（USD）",
          "unit": "USD",
          "caution": "样本缺失 80.0%.",
          "null_pct": "80.0"
        },
        {
          "ord": 23,
          "name": "free_cash_flow",
          "type": "double",
          "zh": "自由现金流（USD）",
          "unit": "USD",
          "null_pct": "20.0"
        }
      ]
    },
    {
      "table": "StockStatus",
      "model": "EMPTY",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockStatus/full.parquet",
      "n_fields": 4,
      "fields": [
        {
          "ord": 1,
          "name": "TradeDate",
          "type": "date32[day]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "date32[day]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "nan"
        },
        {
          "ord": 2,
          "name": "Symbol",
          "type": "string",
          "zh": "代码别名列（美股语义同 Ticker，≠A股 Symbol）",
          "unit": "string",
          "caution": "美股 Symbol≠A股 Symbol。",
          "null_pct": "nan"
        },
        {
          "ord": 3,
          "name": "is_active",
          "type": "bool",
          "zh": "是否活跃（EMPTY 表）",
          "unit": "bool",
          "null_pct": "nan"
        },
        {
          "ord": 4,
          "name": "UpdateTime",
          "type": "timestamp[ns, tz=UTC]",
          "zh": "清洗管线产出时间（UTC）",
          "unit": "timestamp[ns, tz=UTC]",
          "null_pct": "nan"
        }
      ]
    },
    {
      "table": "StockIndustry",
      "model": "EMPTY",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/StockIndustry/full.parquet",
      "n_fields": 6,
      "fields": [
        {
          "ord": 1,
          "name": "TradeDate",
          "type": "date32[day]",
          "zh": "交易日（常为 timestamp[ns] 当日0点，建议归一成 date）",
          "unit": "date32[day]",
          "caution": "timestamp[ns] 需转 date 再与 A股 date32 对齐。",
          "null_pct": "nan"
        },
        {
          "ord": 2,
          "name": "Symbol",
          "type": "string",
          "zh": "代码别名列（美股语义同 Ticker，≠A股 Symbol）",
          "unit": "string",
          "caution": "美股 Symbol≠A股 Symbol。",
          "null_pct": "nan"
        },
        {
          "ord": 3,
          "name": "IndustrySource",
          "type": "string",
          "zh": "行业源（EMPTY）",
          "unit": "string",
          "null_pct": "nan"
        },
        {
          "ord": 4,
          "name": "IndustryCode",
          "type": "string",
          "zh": "行业代码（EMPTY）",
          "unit": "string",
          "null_pct": "nan"
        },
        {
          "ord": 5,
          "name": "IndustryName",
          "type": "string",
          "zh": "行业名（EMPTY）",
          "unit": "string",
          "null_pct": "nan"
        },
        {
          "ord": 6,
          "name": "UpdateTime",
          "type": "timestamp[ns, tz=UTC]",
          "zh": "清洗管线产出时间（UTC）",
          "unit": "timestamp[ns, tz=UTC]",
          "null_pct": "nan"
        }
      ]
    },
    {
      "table": "FactNews",
      "model": "RAW_EVENT",
      "cos": "cos://qs-cold/clean_data/us_stock/massive_data/FactNews/{YYYY-MM-DD}.parquet",
      "n_fields": 16,
      "fields": [
        {
          "ord": 1,
          "name": "id",
          "type": "string",
          "zh": "记录 ID",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "title",
          "type": "string",
          "zh": "新闻标题",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 3,
          "name": "author",
          "type": "string",
          "zh": "新闻作者/来源署名",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 4,
          "name": "published_utc",
          "type": "timestamp[ns, tz=UTC]",
          "zh": "发布时间 UTC（新闻 PIT）",
          "unit": "timestamp[ns, tz=UTC]",
          "caution": "**唯一推荐 PIT 键**；回测用 published_utc≤信号时点。",
          "null_pct": "0.0"
        },
        {
          "ord": 5,
          "name": "article_url",
          "type": "string",
          "zh": "文章 URL",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 6,
          "name": "tickers",
          "type": "list<element: string>",
          "zh": "关联 ticker 列表（需 explode）",
          "unit": "string",
          "caution": "数组/列表，需 explode 后对齐。",
          "null_pct": "0.0"
        },
        {
          "ord": 7,
          "name": "image_url",
          "type": "string",
          "zh": "配图 URL",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 8,
          "name": "description",
          "type": "string",
          "zh": "摘要/描述",
          "unit": "string",
          "null_pct": "0.5"
        },
        {
          "ord": 9,
          "name": "keywords",
          "type": "list<element: string>",
          "zh": "新闻关键词（常大量缺失）",
          "unit": "string",
          "caution": "样本缺失 51.12%.",
          "null_pct": "51.12"
        },
        {
          "ord": 10,
          "name": "insights",
          "type": "null",
          "zh": "洞察/标签结构",
          "unit": "JSON/null",
          "caution": "样本常全空；不要依赖。",
          "null_pct": "100"
        },
        {
          "ord": 11,
          "name": "publisher.name",
          "type": "string",
          "zh": "出版方名称",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 12,
          "name": "publisher.homepage_url",
          "type": "string",
          "zh": "出版方主页",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 13,
          "name": "publisher.logo_url",
          "type": "string",
          "zh": "出版方 logo",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 14,
          "name": "publisher.favicon_url",
          "type": "string",
          "zh": "出版方 favicon",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 15,
          "name": "amp_url",
          "type": "string",
          "zh": "AMP URL",
          "unit": "string",
          "null_pct": "10.88"
        },
        {
          "ord": 16,
          "name": "TradeDate",
          "type": "timestamp[ns]",
          "zh": "文件分区日（常为日历日 0 点）",
          "unit": "timestamp[ns]",
          "caution": "**不是**新闻可知时点；PIT 用 published_utc。同一 TradeDate 文件可含多日 published_utc。",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "adj_factor",
      "model": "D1",
      "cos": "cos://qs-cold/clean_data/adj_factor/date=YYYY-MM-DD/data.parquet",
      "n_fields": 2,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0.0"
        },
        {
          "ord": 2,
          "name": "adj_factor",
          "type": "double",
          "zh": "后复权累积因子",
          "unit": "无量纲",
          "caution": "≡ DailyBar.AdjFactor；调整价=原价×本值。若同 ticker 任一日 >1e6 则见 is_adj_factor_clamped",
          "null_pct": "0.0"
        }
      ]
    },
    {
      "table": "is_adj_factor_clamped",
      "model": "D1",
      "cos": "cos://qs-cold/clean_data/is_adj_factor_clamped/date=YYYY-MM-DD/data.parquet",
      "n_fields": 2,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0"
        },
        {
          "ord": 2,
          "name": "is_adj_factor_clamped",
          "type": "bool",
          "zh": "复权因子是否被夹紧（>1e6 触发，ticker 全历史打标）",
          "unit": "bool",
          "caution": "True 时勿用绝对复权价做长窗口；改用收益/log-return",
          "null_pct": "0"
        }
      ]
    },
    {
      "table": "is_early_close",
      "model": "STATIC_FLAG",
      "cos": "cos://qs-cold/clean_data/is_early_close/data.parquet",
      "n_fields": 2,
      "fields": [
        {
          "ord": 1,
          "name": "date",
          "type": "string",
          "zh": "交易日 YYYY-MM-DD",
          "unit": "date string",
          "caution": "与日线 TradeDate 对齐前转 date",
          "null_pct": "0"
        },
        {
          "ord": 2,
          "name": "is_early_close",
          "type": "bool",
          "zh": "是否提前收盘",
          "unit": "bool",
          "caution": "True 例：感恩节次日、7/3、平安夜。半日市成交量/波动因子需特殊处理",
          "null_pct": "0"
        }
      ]
    },
    {
      "table": "universe_daily",
      "model": "D1_YEAR_FILE",
      "cos": "cos://qs-cold/clean_data/universe_daily/year=YYYY/data.parquet",
      "n_fields": 2,
      "fields": [
        {
          "ord": 1,
          "name": "trade_date",
          "type": "date/timestamp",
          "zh": "交易日（辅助表/宇宙对齐键）",
          "unit": "date",
          "null_pct": "0"
        },
        {
          "ord": 2,
          "name": "ticker",
          "type": "string",
          "zh": "宇宙内代码",
          "unit": "string",
          "caution": "可能宽于 CS StockList，使用前过滤",
          "null_pct": "0"
        }
      ]
    },
    {
      "table": "is_ticker_halt",
      "model": "MINUTE_SPARSE",
      "cos": "cos://qs-cold/clean_data/is_ticker_halt/minute/date=YYYY-MM-DD/data.parquet",
      "n_fields": 3,
      "fields": [
        {
          "ord": 1,
          "name": "ticker",
          "type": "string",
          "zh": "美股代码",
          "unit": "string",
          "null_pct": "0"
        },
        {
          "ord": 2,
          "name": "timestamp",
          "type": "timestamp[ns, America/New_York]",
          "zh": "分钟时间",
          "unit": "ET",
          "caution": "tz-aware；勿当 UTC",
          "null_pct": "0"
        },
        {
          "ord": 3,
          "name": "is_ticker_halt",
          "type": "bool",
          "zh": "该分钟是否 halt",
          "unit": "bool",
          "caution": "先限制 CS 宇宙再解释",
          "null_pct": "0"
        }
      ]
    }
  ]
}
```

---

*本文档为美股核心数据字典（已并入原 catalog 新闻专章 NEWS_RAWDATA）；与 `COS_ashare_lqtp_data_dictionary.md` 文末 CROSS_MARKET_BRIDGE 同步维护。*
