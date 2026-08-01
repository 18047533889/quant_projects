# FactorEngine 原生冷启动因子库

本目录是面向因子挖掘、遗传搜索、LLM/Agent 迭代和 AutoFactorEvaluation 的冷启动母库。它与 `gtja191/`、`week2_pv_factors/` 并列，不修改历史公式包，也不把候选公式塞进评估框架内部。

## 当前规模

| 市场 | 层级 | 解析 surface | 数量 | 定位 |
|---|---|---|---:|---|
| A股 | daily | `daily` | 924 | OHLCV、可选增强字段及PIT财务期种子 |
| 美股 | daily | `daily` | 1112 | A股daily能力，加隔夜/日内、短售和盘口字段 |
| A股 | extended | `compat` | 1412 | 技术指标、尾部统计、条件滚动、回归及财务诊断 |
| 美股 | extended | `compat` | 1492 | A股extended能力，加自由流通股本换手候选 |
| A股 | research | `compat_research` | 308 | 市场相对、状态型、高阶矩、衰减及日内研究算子 |
| 美股 | research | `compat_research` | 308 | 独立美股research目录 |

合计 **5556** 条。两个市场的公共逻辑会分别保留，因为字段合同、股票池和横截面不同；每个市场内部公式严格去重。GTJA185与Week2历史包通过AST结构哈希排除精确重复。

## 算子覆盖

当前FactorEngine公开authoring canonical共207个：

- `daily`：86/86；
- `extended`：101/101；
- `research`：20/20；
- 总覆盖：207/207。

这里的“全部使用”指全部公开canonical，而不是重复使用每个别名。以下非authoring实现不会作为冷启动目标：

- `constant`、`identity`、`protected_div`：内部primitive；
- `cube`：legacy别名，使用`power(x, 3)`或`signed_power`替代。

## 分层设计

### Surface层

- `daily`：FactorEngine审核后的daily canonical。第一轮常规冷启动优先使用。
- `extended`：daily与extended组合，适合二阶段变异、技术指标和稳健统计扩展。
- `research`：市场相对、状态型、实验性及日内研究算子，不代表生产准入。

### 字段可用层

- `core`：日频核心价量字段；
- `derived`：美股隔夜/日内等已派生字段；
- `enriched`：市值、换手、交易所、短售等增强字段；
- `microstructure`：bid/ask及报价深度；
- `capitalization`：自由流通股本等可选口径；
- `fundamental`：PIT对齐的财务报表、报告期和披露时间；
- `intraday`：分钟或更细粒度的价格、成交量和交易日字段。

抽样时应传入真实 `available_fields`。字段不满足的因子会在抽样前被过滤，不允许运行时临时补造。

## 新增逻辑

本轮除覆盖新增算子外，还增加了可解释复合结构：

- 收入、利润、资产、权益和现金流的报告期增长、CAGR、TTM与单季转换；
- TTM利润率、ROA、ROE、现金转化率、增长质量、资本开支和研发强度；
- 财报修订、稳定性、陈旧度及“新鲜度加权修订”；
- 滚动市场Beta、CAPM残差动量、特异波动、特异偏度、协偏度及尾部Beta；
- 残差动量/特异波动、Beta偏离×CVaR；
- Rank correlation、历史CVaR、高阶矩、二次趋势曲率和非线性价量残差；
- expanding rank、hump decay、Chinese SMA及条件持有信号；
- 分组衰减和日内累计VWAP偏离；
- 对 `tan/sec/csc/cot/sinh/cosh/arg` 仅在有界且避开奇点的输入上构造实验候选，并标记 `production_default=False`。

## 使用

### 加载目录

```python
from factor_cold_start import load_catalog

ashare_daily = load_catalog("ashare", "daily")
us_extended = load_catalog("us", "extended")
ashare_research = load_catalog("ashare", "research")
```

### 按真实字段分层抽样

```python
from factor_cold_start import sample_factors

batch = sample_factors(
    market="us",
    surface="research",
    size=64,
    seed="experiment-002",
    available_fields={
        "open", "high", "low", "close", "pre_close",
        "volume", "amount", "ret", "vwap", "adj_factor",
        "exchange",
    },
    availability_tiers=("core", "enriched"),
    max_per_family=8,
)
```

### 财务种子

财务期公式必须使用PIT对齐的报告期、披露时间和修订版本数据：

```python
fundamental_batch = sample_factors(
    market="ashare",
    surface="daily",
    size=32,
    seed="fundamental-round-1",
    available_fields={
        "fiscal_year", "fiscal_quarter", "effective_date", "date",
        "revenue", "net_income", "operating_income", "gross_profit",
        "total_assets", "total_equity", "free_cash_flow",
        "net_cash_from_operating_activities",
        "research_development", "interest_expense", "inventories",
        "receivables", "purchase_of_property_plant_and_equipment",
    },
    availability_tiers=("fundamental",),
)
```

### CLI

```bash
python factor_cold_start/scripts/sample_batch.py \
  --market ashare --surface research --size 64 --seed round-2

python factor_cold_start/scripts/validate_catalog.py --all
python factor_cold_start/scripts/report_coverage.py
python factor_cold_start/scripts/build_catalog.py
```

### AutoFactorEvaluation

可用provider：

- `load_ashare_daily_pack`
- `load_us_daily_pack`
- `load_ashare_extended_pack`
- `load_us_extended_pack`
- `load_ashare_research_pack`
- `load_us_research_pack`
- `load_pack`：读取`FACTOR_COLD_START_MARKET`和`FACTOR_COLD_START_SURFACE`

示例：

```bash
PYTHONPATH=.:factor_engine:AutoFactorEvaluation-RECONSTRUCT \
python -m evaluation.batch \
  --provider factor_cold_start.autofactor.provider:load_ashare_research_pack \
  --output-dir output/factor_evaluation \
  --synthetic
```

## 验证

`validate_catalog.py --all`会检查：

- 六个目录的全部公式均能由当前FactorEngine对应surface解析；
- 每个市场在daily、extended、research层覆盖该surface全部canonical；
- 字段存在于canonical manifest并符合市场与availability tier；
- 不存在负lag、lead、backfill等显式未来信息路径；
- A股与美股字段不串用；
- 与GTJA185、Week2不发生精确结构重复；
- 覆盖报告可确定性重建。

完整统计见 [`reports/coverage.md`](reports/coverage.md)。
