# Factor Engine - LLM Prompt Guide

> **第 5 版更改-shw**：因子引擎已扩展 WorldQuant BRAIN 风格算子分类；DSL 中逻辑函数须用 `and_` / `or_` / `not_`；算子语义与占位说明见同目录 [`operators_semantics.md`](operators_semantics.md)，按版本变更见 [`changelog_shw.md`](changelog_shw.md)。本 Prompt 若与上述文档冲突，以 `operators_semantics.md` 与当前 `api/operator_registry` 为准，并建议逐步同步下文「Supported Operators」列表。
>
> **第 31 版更改-shw**：**cleaned_operators 全量接入** — 白名单 = `build_dsl_allowlist()`；runtime 在 `cleaned_operators/`。以 [`operators_semantics.md`](operators_semantics.md) 与 [`算子与导入教程.md`](算子与导入教程.md) 为准。

You are an AI assistant helping to write factor definitions and configurations for a quantitative factor engine. Below is the complete specification of the system.

---

## 1. System Overview

**Factor Engine** is a quantitative computing framework that:
- Parses factor expressions written in Python-like DSL (`parse_expr` + `build_dsl_allowlist()`)
- All operator **runtime** lives in **`cleaned_operators/`** (not `api/operators/`, removed in v31)
- Compiles `CleanedCall` expression trees into execution plans
- Executes via `PandasBackend` → `cleaned_bridge`
- Returns factor values as MultiIndex Series: `(timestamp, instrument) → factor_value`

**Output Format**: Long table with columns:
- `timestamp`: Time point
- `instrument`: Security/ticker symbol
- `factor_value`: Computed factor value (float, may contain NaN)

---

## 2. Supported Data Sources

### Data Source Types

| Type | Usage | Key Features |
|---|---|---|
| `parquet_kline` | K-line data (日K线、分钟线等) | Fields: `close, volume, transactions, price, size, bid_price, ask_price` Timestamp unit: nanoseconds (ns) |
| `multi_parquet` | Generic multi-file parquet (财务数据、融券数据等) | Flexible instrument/timestamp columns Can be partitioned across multiple files Supports date range filtering |
| `parquet` | Single/simple directory parquet | Generic fallback |

### Core Requirements for All Data Sources

- **Data Shape**: MultiIndex Series with `(timestamp, instrument)` index
  - timestamp: date or datetime (normalized to midnight UTC via `dt.normalize()`)
  - instrument: ticker symbol (string)
- **Value Index**: Column values at each (timestamp, instrument) point
- **Missing Data**: NaN values allowed (dropna occurs in result phase)

---

## 3. Available Datasets (24 Total)

### **Fundamentals (7 datasets)**

All located in `/massive_parquet/fundamentals/`

#### 3.1 `fundamentals/balance_sheet`
**Type**: `multi_parquet`
**Timestamp Column**: `period_end`
**Instrument Column**: `tickers` (list format)
**Available Fields** (38 total):
- `total_assets` — 总资产 (Total assets)
- `total_liabilities` — 总负债 (Total liabilities)
- `total_equity` — 股东权益 (Shareholders' equity)
- `total_equity_attributable_to_parent` — 归属于母公司股东权益
- `cash_and_equivalents` — 现金及现金等价物 (Cash and equivalents)
- `receivables` — 应收款 (Receivables)
- `inventories` — 存货 (Inventories)
- `total_current_assets` — 流动资产合计 (Current assets)
- `property_plant_equipment_net` — 固定资产净值 (PP&E net)
- `goodwill` — 商誉 (Goodwill)
- `intangible_assets_net` — 无形资产净值 (Intangible assets)
- `total_current_liabilities` — 流动负债合计 (Current liabilities)
- `accounts_payable` — 应付账款 (Accounts payable)
- `debt_current` — 短期债务 (Current debt)
- `long_term_debt_and_capital_lease_obligations` — 长期债务
- `deferred_revenue_current` — 递延收入流动部分
- `other_noncurrent_liabilities` — 其他非流动负债
- `common_stock` — 普通股股本 (Common stock par value)
- `additional_paid_in_capital` — 资本公积
- `retained_earnings_deficit` — 留存收益
- `treasury_stock` — 库存股 (Treasury stock)
- `accumulated_other_comprehensive_income` — 累计其他综合收益
- `noncontrolling_interest` — 少数股东权益
- `other_equity` — 其他权益
- *(+14 more fields)*

**Example Factor**:
```python
# Asset quality: rank by total assets
rank(col("total_assets"))

# Leverage ratio: long-term debt / total assets
col("long_term_debt_and_capital_lease_obligations") / col("total_assets")
```

---

#### 3.2 `fundamentals/cash_flow_statement`
**Type**: `multi_parquet`
**Timestamp Column**: `period_end`
**Instrument Column**: `tickers`
**Sample Fields** (32 total):
- `net_cash_from_operating_activities` — 经营活动现金流 (Operating cash flow)
- `net_cash_from_investing_activities` — 投资活动现金流 (Investing cash flow)
- `net_cash_from_financing_activities` — 融资活动现金流 (Financing cash flow)
- `net_income_loss` — 净利润 (Net income)
- `depreciation_and_amortization` — 折旧摊销 (Depreciation & amortization)
- *(+27 more fields)*

**Example Factor**:
```python
# Operating efficiency: operating cash flow quality
zscore(col("net_cash_from_operating_activities"))
```

---

#### 3.3 `fundamentals/financials_ratios`
**Type**: `multi_parquet`
**Timestamp Column**: `date`
**Instrument Column**: `ticker`
**Sample Fields** (23 total):
- `price_to_earnings` — 市盈率 (P/E ratio)
- `price_to_book` — 市净率 (P/B ratio)
- `price_to_sales` — 市销率 (P/S ratio)
- `price_to_cash_flow` — 市现率 (P/CF ratio)
- `earnings_per_share` — 每股收益 (EPS)
- `book_value_per_share` — 每股净资产 (BVPS)
- `return_on_equity` — 净资产收益率 (ROE)
- `return_on_assets` — 资产收益率 (ROA)
- `debt_to_equity` — 债权比 (Debt/Equity)
- `current_ratio` — 流动比率 (Current ratio)
- `quick_ratio` — 速动比率 (Quick ratio)
- *(+12 more ratios)*

**Example Factor**:
```python
# Value factor: low P/E ranking
rank(col("price_to_earnings"))

# Profitability momentum: ROE with 3-period moving average
ts_mean(col("return_on_equity"), 3)
```

---

#### 3.4 `fundamentals/income_statement`
**Type**: `multi_parquet`
**Timestamp Column**: `period_end`
**Instrument Column**: `tickers`
**Sample Fields** (34 total):
- `revenue` — 营收 (Revenue)
- `operating_income` — 营业利润 (Operating income)
- `net_income` — 净利润 (Net income)
- `net_income_loss_attributable_common_shareholders` — 归属普通股股东的净利润
- `cost_of_revenue` — 成本 (Cost of revenue)
- `operating_expenses` — 营业费用 (Operating expenses)
- `research_and_development` — 研发支出 (R&D)
- `selling_general_and_administrative` — 销售管理费用 (SG&A)
- `income_tax_expense` — 所得税支出 (Income tax expense)
- *(+25 more fields)*

**Example Factor**:
```python
# Growth factor: revenue trend
ts_mean(col("revenue"), 4)

# Profitability: net margin
col("net_income") / col("revenue")
```

---

#### 3.5 `fundamentals/short_interest`
**Type**: `multi_parquet`
**Timestamp Column**: `settlement_date`
**Instrument Column**: `ticker`
**Sample Fields** (5 total):
- `short_interest` — 融券数量 (Short interest volume)
- `short_volume_ratio` — 融券比例 (Short volume ratio)
- `days_to_cover` — 回补天数 (Days to cover)
- `avg_daily_volume` — 日均成交量 (Avg daily volume)
- `settlement_date` — 结算日 (Settlement date)

**Example Factor**:
```python
# Sentiment from short pressure
rank(col("days_to_cover"))

# Short interest intensity
zscore(col("short_volume_ratio"))
```

---

#### 3.6 `fundamentals/short_volume`
**Type**: `multi_parquet`
**Timestamp Column**: `date`
**Instrument Column**: `ticker`
**Sample Fields** (15 total):
- `short_volume` — 融券成交量 (Short trade volume)
- `total_volume` — 总成交量 (Total volume)
- `short_volume_ratio` — 融券比例 (Short volume ratio)
- `date` — 交易日期 (Trading date)
- *(+11 more fields)*

**Example Factor**:
```python
# Daily short pressure
col("short_volume_ratio")
```

---

#### 3.7 `fundamentals/stocks_floats`
**Type**: `multi_parquet`
**Timestamp Column**: `effective_date`
**Instrument Column**: `ticker`
**Sample Fields** (4 total):
- `free_float` — 自由流通股数 (Free float)
- `free_float_percent` — 自由流通比例 (Free float %)
- `outstanding_shares` — 已发行股数 (Outstanding shares)
- `effective_date` — 生效日期 (Effective date)

**Example Factor**:
```python
# Liquidity constraint
zscore(col("free_float_percent"))
```

---

### **US Stocks SIP (4 datasets)**

All located in `/massive_parquet/us_stocks_sip/`

#### 3.8 `us_stocks_sip/day_aggs_v1`
**Type**: `multi_parquet`
**Timestamp Column**: `window_start`
**Instrument Column**: `ticker`
**Timestamp Unit**: `ns` (nanoseconds)
**Sample Fields** (8 total):
- `open` — 开盘价 (Open price)
- `close` — 收盘价 (Close price)
- `high` — 最高价 (High price)
- `low` — 最低价 (Low price)
- `volume` — 成交量 (Volume)
- `transactions` — 成交笔数 (Transaction count)
- `vwap` — 成交量加权平均价 (Volume-weighted avg price)
- `window_start` — 窗口开始时间 (Window start)

**Time Range**: 2003-2025+ (daily bars)

**Example Factor**:
```python
# Momentum: 3-day close momentum
rank(ts_mean(col("close"), 3))

# Volatility: high-low range
col("high") - col("low")
```

---

#### 3.9 `us_stocks_sip/minute_aggs_v1`
**Type**: `multi_parquet`
**Timestamp Column**: `window_start`
**Instrument Column**: `ticker`
**Timestamp Unit**: `ns`
**Sample Fields**: Same as day_aggs (with minute granularity)

**Time Range**: 2003-2025+ (minute bars)

**Example Factor**:
```python
# Short-term momentum: 5-min moving average rank
rank(ts_mean(col("close"), 5))
```

---

#### 3.10 `us_stocks_sip/quotes_v1`
**Type**: `multi_parquet`
**Timestamp Column**: `sip_timestamp`
**Instrument Column**: `ticker`
**Timestamp Unit**: `ns`
**Sample Fields** (14 total):
- `bid_price` — 买价 (Bid price)
- `ask_price` — 卖价 (Ask price)
- `bid_size` — 买单量 (Bid size)
- `ask_size` — 卖单量 (Ask size)
- `sip_timestamp` — 报价时间戳
- *(+9 more fields)*

**Example Factor**:
```python
# Bid-ask spread: liquidity
col("ask_price") - col("bid_price")

# Mid-price momentum
rank((col("bid_price") + col("ask_price")) / 2)
```

---

#### 3.11 `us_stocks_sip/trades_v1`
**Type**: `multi_parquet`
**Timestamp Column**: `sip_timestamp`
**Instrument Column**: `ticker`
**Timestamp Unit**: `ns`
**Sample Fields** (13 total):
- `price` — 成交价 (Trade price)
- `size` — 成交量 (Trade size)
- `exchange` — 交易所 (Exchange)
- `sip_timestamp` — 时间戳 (Timestamp)
- *(+9 more fields)*

**Example Factor**:
```python
# Trade price momentum
rank(col("price"))

# Volume intensity
zscore(col("size"))
```

---

### **Other Datasets (13 datasets)**

Additional datasets available (for reference, fields not detailed here):
- `aggregate_bars/daily_market_summary` (10 fields)
- `corporate_actions/dividends` (9 fields)
- `corporate_actions/ipos` (20 fields)
- `corporate_actions/splits` (7 fields)
- `filing/risk_categories` (5 fields)
- `filing/risk_factors` (7 fields)
- `filing/sec_edgar_index` (7 fields)
- `market_operations/condition_codes` (11 fields)
- `market_operations/exchanges` (10 fields)
- `market_operations/market_holidays` (6 fields)
- `news/news` (12 fields)
- `tickers/all_tickers` (12 fields)
- `tickers/ticker_types` (4 fields)

---

## 4. Configuration File Format (YAML)

All factor definitions must follow this YAML structure:

```yaml
factor:
  name: <factor_identifier>                # Unique factor name (snake_case, e.g., "rank_earnings_per_share")
  expr: <dsl_expression>                   # DSL expression (see section 5 for syntax)
  freq: <frequency>                        # Time frequency (e.g., "1d", "1min", "1h")
  universe: <universe_name>                # Optional: "equities", "etf", etc.
  description: <description>               # Brief description in English or Chinese

data_source:
  type: <source_type>                      # "parquet_kline", "multi_parquet", or "parquet"
  root: <root_path>                        # Path to data directory
  timestamp_col: <timestamp_column>        # Column name for timestamp (e.g., "period_end", "window_start", "sip_timestamp")
  instrument_col: <instrument_column>      # Column name for ticker/security (e.g., "ticker", "tickers")
  max_files: <max_file_count>              # Max number of files to load per column (e.g., 3)
  timestamp_unit: "ns"                     # (Optional) Timestamp unit: "ns" for nanoseconds (default: infer from data)
  # start_date: "YYYY-MM-DD"                # (Optional) Row-level filter: only load data >= this date
  # end_date:   "YYYY-MM-DD"                # (Optional) Row-level filter: only load data <= this date

backend:
  type: pandas                             # Currently only "pandas" supported

engine:
  enable_cache: true                       # Enable column-level caching
```

### Configuration Examples

**Example 1: K-line Momentum Factor**
```yaml
factor:
  name: day_aggs_rank_ts_mean_close_3
  expr: rank(ts_mean(col("close"), 3))
  freq: 1d
  description: Daily close price 3-period momentum rank

data_source:
  type: multi_parquet
  root: /massive_parquet/us_stocks_sip/day_aggs_v1
  timestamp_col: window_start
  instrument_col: ticker
  max_files: 3
  timestamp_unit: "ns"
  # start_date: "2024-01-01"
  # end_date:   "2024-12-31"

backend:
  type: pandas

engine:
  enable_cache: true
```

**Example 2: Fundamentals Value Factor**
```yaml
factor:
  name: financials_ratios_rank_pe
  expr: rank(col("price_to_earnings"))
  freq: 1d
  description: Price-to-earnings cross-sectional rank (value signal)

data_source:
  type: multi_parquet
  root: /massive_parquet/fundamentals/financials_ratios
  timestamp_col: date
  instrument_col: ticker
  max_files: 3

backend:
  type: pandas

engine:
  enable_cache: true
```

**Example 3: Balance Sheet Quality Factor**
```yaml
factor:
  name: balance_sheet_leverage_ratio
  expr: col("long_term_debt_and_capital_lease_obligations") / col("total_assets")
  freq: 1d
  description: Long-term debt intensity - lower is better

data_source:
  type: multi_parquet
  root: /massive_parquet/fundamentals/balance_sheet
  timestamp_col: period_end
  instrument_col: tickers
  max_files: 3

backend:
  type: pandas

engine:
  enable_cache: true
```

---

## 5. Operators Dictionary

All available operators for building factor expressions. Can be combined with Python arithmetic (`+ - * /`).

### 5.1 Data Access

#### `col(name: str) -> Expr`
**Purpose**: Reference a column from the data source
**Example**:
```python
col("close")  # Get the close price column
col("total_assets")  # Get total assets column
```

---

### 5.2 Cross-sectional (Panel) Operators

At each timestamp, compute statistic across all instruments.

#### `rank(x: Expr) -> Expr`
**Purpose**: Cross-sectional percentile rank [0, 1] at each timestamp
**Algorithm**: At each timestep, `rank(column) = percentile rank in [0, 1]`
**Example**:
```python
rank(col("close"))  # Rank all closes at each date (0=lowest, 1=highest)
rank(col("price_to_earnings"))  # Valuation rank (0=expensive, 1=cheap)
```

#### `zscore(x: Expr) -> Expr`
**Purpose**: Cross-sectional standardization (mean 0, std 1) at each timestamp
**Algorithm**: `(x - mean) / std` computed across instruments at each date
**Example**:
```python
zscore(col("return_on_equity"))  # ROE standardized within each period
zscore(col("bid_price") - col("ask_price"))  # Spread standardization
```

---

### 5.3 Time-series (Temporal) Operators

At each (timestamp, instrument), compute statistic across time.

#### `ts_mean(x: Expr, window: int, min_periods: int = None) -> Expr`
**Purpose**: Rolling window mean along time dimension per instrument
**Parameters**:
- `x`: Input expression
- `window`: Window size (e.g., 3, 5, 20)
- `min_periods`: Minimum observations required (default: `window`, i.e., NaN if < window observations)

**Example**:
```python
ts_mean(col("close"), 3)  # 3-period rolling average
ts_mean(col("volume"), 20)  # 20-day average volume
```

#### `ts_std(x: Expr, window: int, min_periods: int = None) -> Expr`
**Purpose**: Rolling window standard deviation per instrument
**Parameters**: Same as `ts_mean`

**Example**:
```python
ts_std(col("close"), 20)  # 20-day rolling volatility
zscore(ts_std(col("volume"), 10))  # Volatility cross-section
```

#### `delay(x: Expr, periods: int) -> Expr`
**Purpose**: Time lag (shift periods backward in time)
**Parameters**:
- `x`: Input expression
- `periods`: Number of periods to lag (positive = backward in time)

**Example**:
```python
col("close") - delay(col("close"), 1)  # Price change (today vs yesterday)
col("price") / delay(col("price"), 5)  # 5-period price momentum
```

### 5.4 Group, cleaning, technical, context（第 31 版 — 以白名单为准）

> **Authority**: Only use names in `build_dsl_allowlist()` / [`dsl_operators_reference.md`](dsl_operators_reference.md).  
> Many names from v10–19 docs (`pasteurize`, `bucket`, `ts_donchian`, `ts_sma`, …) are **NOT** in the allowlist until migrated back to `cleaned_operators`.

**Group (in allowlist)**  
`group_rank(x, g)`, `group_neutralize(x, g)`, `group_zscore(x, g)`, `group_mean(x, weight, g)`  
**Not in allowlist**: `group_scale`, `group_backfill`

**Cleaning (in allowlist)**  
`protected_div`, `protected_log`, `protected_sqrt`, `nan_to_num`, `fillna`, `ffill`, `bfill`, `coalesce`, `winsorize`  
**Not in allowlist**: `pasteurize`, `tail`

**Technical (in allowlist — use these names)**  
Moving averages: **`SMA(x,d)`**, **`EMA(x,d)`** — not `ts_sma` / `ts_ema`  
`ts_rsi`, `ts_macd`, `ts_atr`, `ts_bbands`, `ts_adx`, `ts_adxr`, `ts_aroon`, `ts_obv`, `ts_mom`, `ts_roc`, `ts_trix`, `ts_cci`, `ts_stoch`, `ts_stochf`, `ts_willr`, `ts_kama`, `ts_skew`, `ts_kurt`  
**Not in allowlist**: `ts_donchian`, `ts_keltner`, `ts_natr`, `ts_ad`, `ts_sar`, `ts_mfi`, `ts_ppo`, … — see [`dsl_operators_reference.md`](dsl_operators_reference.md) §4

**Context / cross-section (in allowlist)**  
`neutralize(x, y)` — OLS residual with intercept  
**Not in allowlist**: `orthogonalize`, `change_instrument`

**Signals (in allowlist)**  
`trade_when(trigger, alpha, exit_)`, `hump_decay(x, hump)` — not legacy `hump`  
**Not in allowlist**: `bucket`, `ts_step`

**Never generate (not in allowlist)**  
`vec_avg`, `vec_sum`, all `*_stub` names (~79 in catalog). `STUB_IR_OPS` is empty; `parse_expr` rejects unknown names.

---

### 5.5 Arithmetic Operators

**Supported**: `+`, `-`, `*`, `/`, `sin(x)`, `cos(x)`, `exp(x)`（DSL：`sin` / `cos` / `exp`；`exp` 对齐华泰图表 9/11 `Exp(X)`）

**Broadcast Behavior**:
- Series + Series → element-wise on matching index
- Numeric constant (broadcasts automatically via `Literal` conversion)

**Example**:
```python
col("high") - col("low")  # Intraday range
col("net_income") / col("revenue")  # Profit margin
(col("bid_price") + col("ask_price")) / 2  # Mid-price
col("volume") / (col("avg_daily_volume") + 1e-9)  # Relative volume (safe division)
```

---

## 6. Complete Expression Examples

### Example 1: Momentum Factor
```python
# 3-day close moving average rank
rank(ts_mean(col("close"), 3))
```

### Example 2: Relative Value Factor
```python
# P/E ratio standardized within each period
zscore(col("price_to_earnings"))
```

### Example 3: Quality Factor
```python
# ROE trend
ts_mean(col("return_on_equity"), 4)
```

### Example 4: Liquidity Factor
```python
# Volume intensity: volume relative to average, standardized
zscore(col("volume") / ts_mean(col("volume"), 20))
```

### Example 5: Micro-structure Factor
```python
# Bid-ask spread, standardized
zscore(col("ask_price") - col("bid_price"))
```

### Example 6: Composite Factor
```python
# Growth × Quality composite: Revenue growth rank + ROE rank
rank(ts_mean(col("revenue"), 3)) + rank(col("return_on_equity"))
```

### Example 7: Safety Factor
```python
# Low leverage: rank by low debt/assets ratio (ascending order = safer)
rank(col("total_liabilities") / col("total_assets"))
```

---

## 7. Key Constraints & Notes

1. **Expression Type**: All expressions must evaluate to `Expr` objects. Literals are auto-converted.

2. **Index Assumptions**:
   - All data must be loadable as MultiIndex Series: `(timestamp, instrument) → value`
   - Timestamp is normalized to midnight UTC
   - Instrument is a string (ticker symbol)

3. **Missing Data**:
   - NaN values in intermediate results are preserved
   - In cross-sectional ops (`rank`, `zscore`), NaN rows are skipped
   - In time-series ops (`ts_mean`, `ts_std`), NaN handling depends on Pandas defaults

4. **Time Window Requirements**:
   - Window-based operators implicitly set "lookback" requirement
   - `ts_mean(..., window=3)` → requires ≥ 3 periods of history
   - First valid value appears at period `window`

5. **Performance**:
   - `max_files` parameter limits I/O (e.g., `max_files=3` loads only first 3 files per column)
   - `enable_cache: true` 时，`PandasBackend` 对 **结构相同的子表达式** 做内存结果缓存（子树缓存 MVP）；列数据仍由 `DataSource.load_column` 拉取

---

## 8. Execution Output Format

After running a factor configuration:

```python
result = engine.run(factor)
# Returns: {
#     "factor": Factor object,
#     "analysis": {lookback, has_ts_op, has_cs_op, referenced_columns},
#     "plan": Compiled execution plan,
#     "result": pd.Series (MultiIndex: (timestamp, instrument) → factor_value)
# }
```

**Example DataFrame View**:
```
timestamp   instrument  factor_value
2024-01-01  AAPL        0.523
2024-01-01  MSFT        0.891
2024-01-02  AAPL        0.345
2024-01-02  MSFT        0.712
...
```

---

## 9. Quick Reference

| Task | Example |
|---|---|
| Load a field | `col("close")` |
| Rank across instruments | `rank(col("close"))` |
| Standardize across instruments | `zscore(col("volume"))` |
| Rolling average (5 periods) | `ts_mean(col("price"), 5)` |
| Rolling volatility (20 periods) | `ts_std(col("close"), 20)` |
| Lag by 1 period | `delay(col("price"), 1)` |
| Price momentum | `col("close") / delay(col("close"), 5)` |
| Profit margin | `col("net_income") / col("revenue")` |
| Value rank | `rank(col("price_to_earnings"))` |
| Quality rank | `rank(col("return_on_equity"))` |

---

**Version**: 1.0  
**Last Updated**: 2026-03-26  
**Status**: Production
