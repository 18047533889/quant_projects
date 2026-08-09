# FactorEngine R17：基于最新代码的独立增量全量审计与一次性整改提示词

> **用途**：本文件直接作为代码整改 AI 的执行提示词使用。  
> **审计基线**：`quant_projects/main` 最新可见提交 `5dc49e2af6aaa8ab2bc14fe5968049cb074e7df9`（2026-08-09）。  
> **重要边界**：这是一个**新的、独立的整改文档**。不要与 R16 或更早任何整改文档合并，不要重复输出以前已经提出并正在并行整改的旧问题。若本文件指出的缺陷与旧问题表面相似，但这里给出了**当前 HEAD 下新的具体证据、跨市场冲突或新的调用链后果**，仍按本文件整改。  
> **工作位置**：直接修改服务器/本地工作区中的代码即可；不要把“提交 GitHub、开 PR、合并 GitHub”作为任务目标。GitHub 仅用于我此次审计查看当前代码。  
> **执行方式**：不要分批让我确认，不要只修一部分后停下。读完整个文件后一次性完成全部整改、全量回归、全量 operator 审计、文档/manifest/evidence 重生与最终验收。

---

## 0. 这轮整改的核心目标

这轮不是继续“多加几个算子”，而是把 FactorEngine 从当前的“**A 股 legacy 语义层 + additive 美股语义层**”彻底收敛为一个真正的**单一权威、多市场、可证明、可追溯**的因子执行系统。

当前代码已经做了大量整改，但最新代码仍存在一类更隐蔽的问题：

1. **同一个字段/表/算子在不同层有不止一个语义真值源**；
2. A 股 legacy `FIELD_REGISTRY` 仍在若干关键路径里充当默认真值，从而污染 US；
3. DataAccess 已更新的数据集拆分/命名与 FactorEngine catalog/provider 发生漂移；
4. provider 声明了 required filters / PIT / coverage，但 expression support、field plan、runtime 不一定真正执行这些约束；
5. A/US 的“相同概念”实际上有不同的会计期间、复权、交易状态、会话、新闻、股本和 universe 语义；
6. 当前 operator market manifest 主要证明“算子注册了、市场标签有值”，还不能证明“该算子在这个市场的**实际输入链路**可执行且语义正确”；
7. 需要对**当前 registry 中的每一个 canonical operator**重新做机器化、逐项的 A/US 可执行审计，而不是继续靠 prefix 或手写小集合兜底。

最终目标不是“测试绿”本身，而是：

> **任何一个 formula，从 DSL → field concept → provider → source planning → PIT/filters/universe/session → typed IR → lowering → pandas/polars/duckdb → cache/evidence，都只能存在一套一致的市场语义；没有任何路径能绕过它。**

---

# 1. 数据真值：本轮整改必须以这两套真实数据契约为准

不要凭通用金融常识臆造字段，不要因为算子想要某字段就假设数据有。

## 1.1 A 股真实约束

必须按当前 COS A 股字典处理：

- instrument：`Symbol`，带 `.SH/.SZ`；
- `Return` 是 **bp**，进入通用收益语义必须 `/10000`；
- `Factor` 是**后复权累积乘数**：`continuous_price = raw_price * Factor`；
- `TurnoverRatio`、`DividendRatio`、ROE/ROA/利润率、`ShareRatio`、指数 `Weight` 是 `%`，进入 ratio 语义时 `/100`；
- `StockBalance/Income/CashFlow/Indicator` 的 PIT 是 `PubDate`，不是 `ReportPeriodEndDate`；
- 同一 `PubDate` 可能存在多个报告期，必须有显式 period-selection；
- `StockDividend` 没有可靠公告时间，只有 `ExDividendDate` 的 effective-time 语义，strict PIT 默认禁止；
- `StockIndustry` 一股一天约有多套行业源，必须选择且只选择一个 `IndustrySource`；
- `StockCapitalDaily` 是 S1 股本状态，但覆盖末端可能落后行情；不得无限向未来 carry；
- `StockTopTenShareholder/FloatShareholder` 是 **S1 snapshot**，不是简单按 `PubDate` 的财报事件表；一行是快照日×股票×股东；
- A 股分钟线 `QuoteTime` 存 UTC，但本地会话标签是约 09:31–11:30 / 13:01–15:00，共 240 根，且当前数据 catalog 已声明为 **bar_end**；
- A 股没有 clean 新闻源。

## 1.2 美股真实约束

必须按当前 COS US 字典处理：

- instrument：行情主键 `Ticker`；部分表物理列叫 `Symbol` 或 `ticker`，必须 adapter 统一；
- `Ret` / `Ret_Intra` / `Ret_Overnight` 已是**小数收益**，禁止再 `/10000` 或 `/100`；
- `AdjFactor` 是后复权乘数：`raw_price * AdjFactor`；
- `dividend_yield` / `return_on_equity` / `return_on_assets` 已是小数；
- 财务表 PIT 是 `filing_date`，文件分区 `period_end` 绝不能当知识时点；
- 财务必须先 filter **一个明确 timeframe**，禁止 quarterly / annual / TTM 混在同一截面；
- `StockValuationDaily` / `StockIndicator` 是 X0 稀疏 snapshot，不能伪装全历史 D1；
- `StockIndustry` / `StockStatus` 当前 EMPTY，不能借用 A 股同名表语义；
- 历史日频市值优先 `TickerSharesSnapshot.weighted_shares_outstanding * Close`；
- `StockCapitalDaily` 是**双 schema**，而 DataAccess 已拆分：split events 与 shares PIT 必须分别读；
- `StockIndicesComponents` 有 membership，没有 `Weight`；
- `StockDividend.cash_amount` 币种随 `currency` 变化，不是天然 USD；无 FX 时默认 USD-only；
- 新闻 PIT 用 `published_utc`，`tickers` 是 list，需要 explode；当前 FactNews 的 `insights` 常不可用，不能把“有新闻”直接当“有 sentiment provider”；
- `is_early_close` 全历史约只有几十个 true 日，不是“每年几十天”；
- `is_ticker_halt` 是稀疏分钟 flag，不能替代 A 股 `IsSuspend`；
- `is_adj_factor_clamped` 为真时，长窗口不得无条件依赖绝对后复权价格水平。

---

# 2. 总体整改原则

下面每一条都要执行，不要把任何一条降级成“后续建议”。

1. **市场上下文必须显式贯穿所有层**。production 不允许裸 `resolve_field(name)` 自动落到 A 股。
2. **一个概念只允许一个语义权威入口**。DataAccess catalog、FE FieldSpec、MarketFieldBinding 之间不能各自重新解释单位/PIT/coverage。
3. **物理字段不等于 canonical concept**。生产公式凡能 canonicalize 的字段，必须先过 canonical provider；raw physical column 只允许显式 research escape hatch。
4. **required filter 是执行契约，不是备注**。缺 `IndustrySource`、`IndexSymbol/IndexName`、`timeframe`、currency filter 等必须 fail closed。
5. **provider coverage 是 universe×time-window 条件变量**，不能只存一个静态 `0.42`。
6. **PIT eligibility、coverage eligibility、source certification、provider quality、session availability 都必须同时成立**，不能只检查其中一层。
7. **Research 模式不能自动等价于“允许未来信息”**。研究模式可以放宽 performance/coverage，但非 PIT 语义必须另开显式 dangerous flag。
8. **算子市场支持不能只靠名字前缀**。最终以 typed input contracts + provider capability graph + grain/session requirement 为准。
9. **每一个 canonical operator 必须在当前 registry 动态枚举审计**，不要硬编码“1293/1311”这种会随代码变化的数量。
10. **所有缓存、factor identity、evidence identity、lineage 都必须包含 market + provider/source contract + universe/filter/session 等会改变经济语义的参数**。

---

# 3. 当前 HEAD 已确认的新问题与具体整改

以下条目是本轮 R17 的主体。编号只是便于验收，**不代表优先级**；全部一次性完成。

## R17-001：legacy `FIELD_REGISTRY` 仍固定为 A 股，形成双真值系统

### 当前问题

`factor_engine/fields/__init__.py` 仍把：

```python
FIELD_REGISTRY = FieldRegistry(ASHARE_FIELD_SPECS, ASHARE_TABLE_SPECS)
DEFAULT_FIELD_REGISTRY = FIELD_REGISTRY
```

作为 legacy 默认，同时 multi-market registry 只是 additive layer。

这会导致：

- parser / analyzer / storage / helpers 里任何裸 `from fields import resolve_field` 都隐式 A 股；
- US 公式可能被 A 股 field aliases、price basis、flow semantics 命中；
- 同一 US formula 在 capability resolver 与 IR analyzer 里得到不同语义；
- cache/evidence 可能因为走不同 resolver 得到不同 factor identity。

### 必须整改

建立一个且只有一个 production field-resolution API，例如：

```python
resolve_field(
    market_context,
    logical_or_physical_name,
    table=None,
    source_context=None,
) -> ResolvedMarketField
```

要求：

- production 必须传 `MarketContext`；
- 不允许 production 自动 fallback 到 `ashare`；
- legacy A 股 API 只保留兼容包装，但内部必须显式调用 `ASHARE_CONTEXT`；
- IR、planner、storage、mining、source preflight、manifest、evidence 全部迁到同一 resolver；
- 增加 AST/static audit，禁止 production 目录出现裸 `FIELD_REGISTRY.resolve*` / `resolve_field(name)`。

---

## R17-002：`field_plan._table_current_snapshot_only()` 会用 A 股表语义解释 US 同名表

### 当前问题

`storage/sources/field_plan.py` 的 `_table_current_snapshot_only(table)` 直接导入 `fields.FIELD_REGISTRY`。

因此：

- US `StockValuationDaily` 本应 `current_snapshot_only=True`，但 A 股同名表是完整 D1；
- US `StockIndicator` 本应 X0 snapshot，但 A 股同名表是财报 E1；
- US `StockCapitalDaily` 与 A 股完全不是同一种数据结构。

这属于结构性跨市场污染。

### 必须整改

`NormalizedFieldPlan` 必须携带 `market`，表级语义只允许：

```python
MULTI_MARKET_FIELD_REGISTRY.registry_for(market).resolve_table(table)
```

或者直接把 resolved `TableSpec` 放进 plan，后续禁止再仅凭 table name 二次查表。

加 golden：

- A `StockValuationDaily.current_snapshot_only == False`
- US `StockValuationDaily.current_snapshot_only == True`
- A `StockIndicator` financial_event
- US `StockIndicator` current_snapshot
- 两市场同名表绝不共享 TableSpec object/hash。

---

## R17-003：`NormalizedFieldPlan.universe` 在两条构建路径含义不一致

### 当前问题

`plan_from_field_spec()` 当前把：

```python
universe = spec.domain
```

而 `plan_from_catalog_field()` 把：

```python
universe = field.market
```

于是同一字段走 registry path 时 universe 可能是 `valuation/fundamental`，走 catalog path 则是 `ashare/us`。

### 必须整改

拆成至少：

```text
market
universe_id / universe_policy
semantic_domain
```

严禁一个字段同时承载这三种含义。

所有 factor/cache/evidence key 用明确字段，不得依赖 overloaded `universe`。

---

## R17-004：financial missing 被错误解释成 `NO_EVENT`

### 当前问题

`field_plan.py` 把：

```text
financial_event
financial_pit
sparse_snapshot
partial_history
current_snapshot
```

纳入 `NO_EVENT` 语义。

这会把“某财务字段未披露/缺失”错误解释成“今天没有事件”，进而可能被 event count / zero fill / mask 运算错误消费。

更严重的是 `current_snapshot` 已在 `_NO_EVENT_COVERAGE` 提前返回，因此后面 `current_snapshot -> NOT_APPLICABLE` 分支事实上不可达。

### 必须整改

把 missing semantics 拆开：

- **event occurrence table 没有行** → `NO_EVENT`
- **某 event row 存在但某财务 numeric field NaN** → `UNKNOWN`
- **current snapshot 不覆盖历史日期** → `NOT_APPLICABLE` 或 `OUT_OF_COVERAGE`
- **field 对当前证券结构上不存在** → `NOT_APPLICABLE`
- **停牌/无 session** → `NOT_TRADING`
- 只有明确结构零才能 `STRUCTURAL_ZERO`

不得再通过 `financial_pit` 这个 temporal model 就把 numeric NaN 全部解释成 no event。

---

## R17-005：PIT eligibility 的 tri-state 在 field plan 中被 bool 化

### 当前问题

FieldSpec/TableSpec 已试图把 `strict_pit_allowed=None` 表示 UNKNOWN，但 `NormalizedFieldPlan` 又使用：

```python
strict_pit_allowed: bool = True
bool(getattr(spec, "strict_pit_allowed", True))
```

这样会把“未知契约”压成真假值，并且 raw plan 默认 True。

### 必须整改

全链路保持三态：

```text
ALLOWED / BLOCKED / UNKNOWN
```

production：UNKNOWN 必须 fail closed。  
research：UNKNOWN 可以显式 opt-in，但 lineage 必须记录。

---

## R17-006：coverage contract 只按 logical field 名称注册，跨市场会冲突

### 当前问题

`data_access_source.py`：

```python
_COVERAGE_CONTRACTS: dict[str, HistoricalCoverageContract]
```

key 只有 `contract.field`。

例如：

- `market_cap` A 股完整；
- `market_cap` US 来源覆盖明显不同；
- 同一个 concept 甚至可有多个 provider chain。

后注册者可能覆盖前一个市场/来源的 coverage contract。

### 必须整改

coverage identity 至少：

```text
(market, concept_id, provider_id, dataset, source_version, universe_id)
```

coverage 查询必须再带 requested `[start,end]` 和 target universe。

---

## R17-007：没有 coverage contract 时 `assert_historical_coverage` 直接通过

### 当前问题

当前逻辑允许 `contract is None -> return`。

对 production partial-history / current-only / derived-partial provider，这等于“没有证据 = 合格”。

### 必须整改

production 规则：

- FULL provider：可由数据契约明确声明，不一定逐次扫描；
- PARTIAL/SPARSE/CURRENT_ONLY provider：必须存在 coverage evidence；
- contract 缺失 → `COVERAGE_EVIDENCE_MISSING`，fail closed；
- research 可继续，但必须 warning + lineage。

---

## R17-008：coverage 必须基于“目标可投资 universe”，不能固定用全 DailyBar ticker 比例

### 当前问题

US `TickerSharesSnapshot` 的静态 coverage 约 42% 只是一个粗指标，而 US DailyBar 包含远多于 StockList CS 的 symbol 类型。

### 必须整改

coverage 应计算：

```text
covered_valid_provider_rows
/
(target_universe ∩ valid_market_session ∩ instrument_type_policy)
```

逐日、逐 requested window、逐 universe policy 计算。

US 默认 universe 至少明确：

```text
StockList(type == CS) ∩ universe_daily
```

不要用全 Raw DailyBar 做 size coverage denominator。

---

## R17-009：同一 canonical concept 可能被重复做单位归一

### 当前问题

现在至少有三层可能做 scale：

1. DataAccess SemanticFieldCatalog；
2. FactorEngine FieldSpec / NormalizedFieldPlan；
3. MarketFieldBinding.transform。

在 A 股 `Return`、ROE、TurnoverRatio 这类字段上尤其危险。

### 必须整改

建立“**unit normalization ownership**”单一规则：

- source boundary 返回 raw 还是 canonical，必须显式标记；
- provider 不得对已经 canonical 的值再 scale；
- plan/evidence 记录 `source_unit -> canonical_unit -> normalization_applied_by`；
- 不允许靠字段名猜是否已经归一。

必须做 sentinel tests：

```text
A Return raw=100 bp      -> canonical return_decimal = 0.01，恰好一次
US Ret raw=0.01          -> canonical return_decimal = 0.01，零次缩放
A TurnoverRatio raw=2.5% -> canonical = 0.025
US ROE raw=0.10          -> canonical = 0.10
```

三 backend、eager/lazy、DataAccess direct/provider path 均一致。

---

## R17-010：A 股 legacy catalog 仍把 Factor 方向写成 `unverified`

### 当前问题

`fields/catalog.py` 仍有 stale metadata/comment：

```text
Factor direction = unverified
price adjustment_status = unverified
```

而当前数据契约已经验证 `raw * Factor`。

最新 IR analyzer 会先 consult legacy FieldSpec，因此 stale metadata 不只是文档问题。

### 必须整改

- `Factor`: backward cumulative multiplier；
- raw OHLC/VWAP：`price_basis=RAW`；
- continuous OHLC/VWAP：通过 canonical derived concepts 提供；
- 删除所有“Factor 未验证”的旧注释与 metadata；
- catalog/providers/DataAccess contract 三处 hash 一致性测试。

---

## R17-011：A/US `PreClose` 不应被建模成普通 `raw_pre_close`

### 当前问题

两市场 `PreClose` 都更接近“官方参考前收/公司行为调整后的前收”，而不是简单 `lag(raw_close,1)`。

A 股字典明确说明除权日 `PreClose` 已用于无公司行为跳点的 Return。

### 必须整改

新增独立 canonical semantic：

```text
reference_pre_close
price_basis = OFFICIAL_REFERENCE_PRE_CLOSE
```

明确区分：

- `reference_pre_close`
- `lag(raw_close, 1)`
- `lag(continuous_close, 1)`

所有 overnight / gap / limit-relative 算子逐个检查应使用哪一种，不能混用。

---

## R17-012：A 股 catalog 的 EPS / cash dividend / stock dividend 单位错误

### 当前问题

至少检查并修正：

- `Eps`：CNY/share，不是 CNY absolute；
- `CashDividend`：CNY/share；
- `StockDividend`：每股比例，dimensionless ratio；
- `StockTransfer`：每股比例，dimensionless ratio；
- price fields 应是 CNY/share，不只是 CNY。

### 必须整改

legacy unit system 与 units_v2 统一维度表达，不允许“价格”和“金额”共用同一个 `CNY` 维度。

新增 dimensional tests：

```text
cash_dividend / raw_close -> ratio
EPS / raw_close -> dimensionless earnings yield style ratio
stock_transfer + 1 -> valid
stock_transfer + cash_amount -> invalid
```

---

## R17-013：catalog helper 默认 `strict_pit_allowed=True` 破坏 fail-closed 设计

### 当前问题

虽然 `TableSpec` 默认已改为 UNKNOWN/None，但 A/US `_table()` helper 仍默认 True。

新增表如果忘记声明，就会自动变成 PIT-safe。

### 必须整改

helper 默认改 `None`；所有生产可读 table 必须显式声明：

```text
strict_pit_allowed=True/False + reason
```

CI：任何 table/field 未声明 PIT 状态不得进入 production/mining manifest。

---

## R17-014：TableSpec.required_parameters 没有可靠下沉到 FieldSpec.required_filters

### 当前问题

例如：

- `StockIndustry` 必须 `IndustrySource`；
- `IndexConstituent` 必须 `IndexSymbol`；
- US financial 必须 `timeframe`。

但 `_f()` 并不会自动把 table-level required parameters 带入每个 FieldSpec。

### 必须整改

required filter 的权威应在 TableSpec/ProviderBinding 的一处定义，然后 FieldPlan 继承 resolved effective requirements。

CI：

```text
if table.required_parameters != empty:
    every mineable field must expose same effective filter requirements
```

---

## R17-015：required filter 的结构信息不够，普通字符串不能证明“选择了什么”

### 当前问题

当前 provider 常写：

```python
required_filters=("IndustrySource",)
required_filters=("IndexName",)
required_filters=("timeframe",)
```

这只证明“有一个字段名”，不能表达：

- allowed values；
- exactly one；
- filter 是 column predicate 还是 join dependency；
- 默认值；
- factor identity 是否必须包含 filter value。

### 必须整改

使用结构化 contract，例如：

```python
FilterRequirement(
    field="IndustrySource",
    operator="in",
    allowed=("sw_l1","sw_l2","sw_l3","zjw","jq_l1","jq_l2"),
    exactly_one=True,
)
```

US financial：

```text
timeframe ∈ {quarterly, annual, trailing_twelve_months}
exactly_one = True
```

index：

```text
IndexSymbol/IndexName exactly_one unless operator explicitly supports multi-index relation
```

filter 的**实际取值**必须进入 factor hash / lineage / evidence。

---

## R17-016：`universe_daily present` 被错误建模成 filter，而不是跨数据集依赖

### 当前问题

US tradability binding 把 `universe_daily` 放在 `required_filters`，但它是另一个数据集/关系，不是当前表的一列 filter。

### 必须整改

拆出：

```text
FilterRequirement
JoinRequirement
DependencyRequirement
AvailabilityRequirement
```

provider planner 必须真正 materialize 这些依赖，而不是在 metadata 里写“present”就算满足。

---

## R17-017：`MarketFieldBinding.dataset` 只有一个 dataset，无法正确表示多数据集 derived provider

### 当前问题

US market cap：

```text
TickerSharesSnapshot.weighted_shares_outstanding * StockDailyBar.Close
```

US tradability：

```text
StockList.type + DailyBar.Close + universe_daily
```

但 binding 只有一个 `dataset=` 主字段；`physical_fields` 又跨表。

### 必须整改

derived provider 必须改成明确 dependency graph：

```python
ProviderDependency(
    dataset=...,
    table=...,
    field=...,
    join_keys=...,
    temporal_join=...,
    required_filters=...,
)
```

planner 根据 graph 生成 source plan。禁止“dataset 指 shares、physical_fields 又偷偷引用 DailyBar”这种半声明状态。

---

## R17-018：FE 与 DataAccess 的 US dataset 名称发生漂移

### 当前问题

DataAccess 当前整改已经把/新增：

```text
us_stock_capital_split
us_stock_capital_shares
us_ticker_shares_snapshot
us_security_master_daily_snap
us_fact_news
```

但 FactorEngine 当前 catalog/provider 仍可看到：

```text
us_stock_capital_daily
us_stock_shares_snapshot
```

等旧命名/旧结构。

### 必须整改

不要手工再维护第二套 dataset registry。

做一个自动 consistency audit：

```text
for every FE TableSpec.dataset / ProviderDependency.dataset:
    assert dataset exists in current DataAccess registry
    assert time_column matches
    assert instrument_column matches
    assert every physical field exists
    assert required filters compatible
    assert schema fingerprint compatible
```

任何漂移 CI 直接失败。

---

## R17-019：US `StockCapitalDaily` 在 FE 仍被注册成一个双 schema 逻辑表

### 当前问题

DataAccess 已经正确拆开，但 `catalog_us.py` 仍把一个 `StockCapitalDaily` 逻辑表同时承载 split 与 PIT shares 字段。

### 必须整改

FE 也拆为至少：

```text
USStockCapitalSplitEvent
USTickerSharesPITEvent
USTickerSharesSnapshotDaily
```

并映射真实 DataAccess datasets。

原则：

- 日频 shares / size 首选 `TickerSharesSnapshot`；
- `shares_*.parquet` 只作稀疏 PIT fallback/事件源；
- `{date}.parquet` 只作 split/corporate-action event；
- 严禁一个 reader/glob 混两种 schema；
- 同日同 ticker 多 shares 记录必须有 dedupe/revision policy。

---

## R17-020：US `StockDividend` TableSpec 使用了不存在的 `TradeDate` / `period_end`

### 当前问题

当前 `catalog_us.py` 可见：

```python
time="TradeDate"
period="period_end"
```

但真实 US StockDividend 字段是：

```text
ticker
record_date
pay_date
declaration_date
ex_dividend_date
frequency
cash_amount
currency
...
```

没有该表意义上的 `TradeDate`、也没有 `period_end`。

### 必须整改

TableSpec：

```text
knowledge_time = declaration_date
effective_time = ex_dividend_date
partition/effective key = ex_dividend_date
instrument = ticker
period_id = None
```

`declaration_date`/`ex_dividend_date` string 先显式 parse 为 date/timestamp。

---

## R17-021：US `cash_amount` 的物理单位建模错误

### 当前问题

catalog 把 `cash_amount` 当 `_UNIT_USD`，但真实语义是：

```text
amount per share in declared currency
```

且约有非 USD。

### 必须整改

raw concept：

```text
dividend_cash_amount_declared_currency_per_share
```

只有满足 `currency == USD` 才能映射到：

```text
cash_dividend_per_share@us USD/share
```

若未来有 FX provider，则另做 `cash_dividend_usd_per_share` 的 derived provider，并把 FX observation time 纳入 PIT。

---

## R17-022：US FactNews catalog 的物理字段名存在错误

### 当前问题

当前 catalog 可见 `news_headline -> headline`，而真实 FactNews 字段为 `title`。

### 必须整改

- 修正为 `title`；
- 自动 schema audit，禁止 catalog 再出现不存在 physical field；
- 不要为了兼容静默 alias 一个不存在物理列。

---

## R17-023：US FactNews instrument 不是标量 ticker，必须有 explode adapter

### 当前问题

真实 `tickers` 是 `list<string>`；当前 TableSpec 却以标量 `ticker` 作为 instrument 语义。

### 必须整改

引入 `NewsEventAdapter`：

1. `explode(tickers)`；
2. rename exploded ticker → canonical instrument；
3. 保留 article `id`；
4. PIT 只用 `published_utc`；
5. publisher/article dedupe 与 ticker explode 分开；
6. after-close article → next eligible session when constructing daily factor；
7. 同一 article 关联多 ticker 时，per-ticker count 可各计一次，但 global article aggregate 要按 article id 去重。

---

## R17-024：FactNews 的 source timezone 标错

### 当前问题

`published_utc` 是 tz-aware UTC，但 US TableSpec 当前写 `timezone="America/New_York"`。

### 必须整改

source timestamp contract：UTC。  
只有进行 `published_utc -> US session/decision timestamp` 映射时才转换 `America/New_York`。

CI 覆盖 DST 切换与 after-close。

---

## R17-025：`NEWS` capability 过粗，当前数据并不天然提供 sentiment

### 当前问题

当前 market capability 把 US 标为 `NEWS`，operator family 又包括：

```text
news_sentiment
news_sentiment_momentum
news_negative_ratio
```

但当前 FactNews `insights` 常不可用；有 title/description 不等于已有 sentiment 数值。

### 必须整改

拆能力：

```text
NEWS_EVENT
NEWS_TEXT
NEWS_SENTIMENT
NEWS_ENTITY_LINK
```

当前 US：

- event/text 可有 provider；
- sentiment 必须 `PROVIDER_REQUIRED`，除非你们真的接入一个已版本化 NLP sentiment provider；
- sentiment provider 必须记录 model id/version/text source/language/preprocessing version。

A 股当前全部新闻相关能力默认 unavailable/provider_required。

---

## R17-026：`index_member` provider 返回的是字符串标识，不是 bool membership

### 当前问题

A/US `index_member` 当前绑定到：

```text
IndexConstituent.IndexSymbol
StockIndicesComponents.IndexName
```

并使用 identity transform，但 canonical unit/role 却是 boolean。

### 必须整改

index membership 必须通过：

```text
filter exact index id -> relation membership -> left join target universe -> present=1 / absent=0
```

注意只有在**index relation 当日 coverage 完整**时 absent 才可结构化为 0，否则是 UNKNOWN。

---

## R17-027：A 股 `index_weight` 缺 `IndexSymbol` required filter

### 当前问题

一张 `IndexConstituent` 同一天可包含多个指数。只读取 Weight 而没有 index id 会重复/串指数。

### 必须整改

`index_weight@ashare` 强制：

```text
IndexSymbol exactly_one
Weight / 100 -> ratio
```

filter value 进入 factor identity。

US `index_weight` 保持 unavailable，不能拿 equal-weight membership 冒充。

---

## R17-028：`industry_group@ashare` 没有编码合法 IndustrySource 与 exact-one

### 当前问题

只要求“有 IndustrySource”，不足以阻止：

- 传多源；
- 拼错源；
- 不同源结果共享 factor hash。

### 必须整改

allowed：

```text
sw_l1, sw_l2, sw_l3, zjw, jq_l1, jq_l2
```

exactly one；默认策略如果采用 `sw_l1`，必须显式成为 source policy，不允许隐藏默认。

US industry 继续 provider required。

---

## R17-029：A/US `tradability_state` 不是同一个经济概念

### 当前问题

A 当前是：

```text
NOT IsSuspend
```

US 当前是：

```text
StockList.type == CS AND finite Close
```

二者语义完全不同，却映射到同一 canonical `tradability_state`。

### 必须整改

拆成至少：

```text
listing_membership
common_equity_membership
not_suspended_state
valid_daily_bar_state
research_universe_membership
tradable_eod_mask (composite)
```

A 股 composite 可再结合 `PublicStatus`、IsSuspend、目标策略是否排 ST 等 policy。  
US composite 至少结合 `StockList(CS)` + `universe_daily` + valid bar；`is_ticker_halt` 因覆盖极稀不得当稳定日频条件。

---

## R17-030：US tradability 当前声明了 universe dependency，但 transform 没真正消费

### 当前问题

provider metadata 说需要 `universe_daily`，实际 transform 只看 `type + Close`。

### 必须整改

如果 canonical 定义包含 universe membership，执行 graph 必须真的 join/use 它；否则不要把它写进 required filter 伪装满足。

unknown universe membership 不能默认 hard-zero 后又声称“完整 tradable mask”。

---

## R17-031：US market-cap provider 的数据集命名、跨表依赖和 fallback chain 都要重做

### 当前问题

当前 primary provider：

```text
weighted_shares_outstanding * Close
```

思路正确，但实现仍存在：

- dataset 名漂移；
- binding 单 dataset 无法描述跨表；
- static 42% coverage 过粗；
- provider chain 是否真正执行 fallback 不明确。

### 必须整改

推荐 chain：

1. `TickerSharesSnapshot.weighted_shares_outstanding * Close`
2. 如果业务语义接受，可使用 `share_class_shares_outstanding * Close`，但 quality/coverage 独立记录；
3. X0 `Valuation.market_cap` 只允许 current/snapshot 场景，不用于历史 backfill；
4. 任何更稀疏 shares event 只能 explicit research fallback。

provider selection 要按 date/universe 动态记录，并进入 lineage。

---

## R17-032：provider chain 注册 ≠ provider chain 真正执行

### 当前问题

`ProviderRegistry.binding()` 返回 chain 的第一个 provider。仅仅存在 `bindings()` 并不能证明 runtime 会在 primary 缺数据/不满足 coverage 时选择 secondary。

### 必须整改

实现唯一 `ProviderResolver`：

```text
candidate chain
-> market/context eligibility
-> source certification
-> required filters
-> PIT
-> requested-window coverage
-> universe coverage
-> currency/session constraints
-> select first eligible provider
```

每次选择记录：provider id、reject reasons、chosen reason。

---

## R17-033：US adjusted-price provider 没把 `is_adj_factor_clamped` 变成执行约束

### 当前问题

`continuous_close@us` 只是 notes 提醒“clamped 时长窗改用 return”，但 provider quality 仍是 `EXACT_DERIVED + FULL`。

### 必须整改

增加 dynamic validity policy：

```text
if is_adj_factor_clamped(ticker):
    absolute continuous-price level providers are not certified for long-history level-based operators
```

可用策略：

- return/log-return based path；
- 或以局部重新基准化方式避免极端累积 factor，但必须有数学证明；
- operator contract 声明是否“level-sensitive”。

A 股没有对应 clamp table，也至少加 finite/overflow/extreme-factor sanity gate。

---

## R17-034：`FinancialPeriodAdapter.filter_timeframe()` 在列不存在时 fail-open

### 当前问题

当前逻辑大意：

```python
if "timeframe" in columns:
    filter
return frame
```

US 财务要求 timeframe 是硬契约；列不存在不应悄悄原样返回。

### 必须整改

US financial production：

- timeframe column 缺失 → hard error；
- timeframe 未指定 → hard error；
- 值不合法 / 多值混合 → hard error；
- A 股 path 不要求 timeframe，但使用另一套 fiscal period semantics。

---

## R17-035：财务 asof 没有把 period-selection 变成强制契约

### 当前问题

A 股同一个 `PubDate` 可有多个 `ReportPeriodEndDate`；US 同一 filing/ticker 也可能存在多个 period/revision。

仅按 knowledge date merge_asof 会依赖行顺序或最后一行，无法保证经济语义。

### 必须整改

明确 `FinancialPeriodPolicy`，至少包含：

```text
timeframe
period_selection
revision_selection
same_day_visibility_policy
flow_conversion_policy
```

A 股例如：

```text
latest_period_known_asof_decision_time
```

US：

```text
exact timeframe + latest filing revision for selected period
```

period policy 必须进 factor identity。

---

## R17-036：canonical financial flow semantics 仍以 A 股 cumulative-YTD 为默认，US provider 没真正 override

### 当前问题

`fields/concepts.py` 对 revenue/net_profit/OCF 的 `flow_semantics` 是 A 股 cumulative YTD；注释写“US binding 根据 timeframe override”，但当前 `MarketFieldBinding` 没有可见的 `flow_semantics` override 字段。

### 风险

US quarterly / TTM 数据可能被错误 quarterize，或者 typed IR 把 TTM 当 YTD。

### 必须整改

flow semantics 必须是 resolved-provider 输出，而不是 concept 的单一静态属性。

例如：

```text
A Income quarterly source row: cumulative_ytd_flow
US timeframe=quarterly: single_period_flow
US timeframe=annual: annual_flow
US timeframe=trailing_twelve_months: ttm_flow
Balance: stock
```

任何 `quarter_from_cumulative`、growth、TTM、YoY 算子必须读取 resolved flow semantics。

---

## R17-037：最新 `ir/analyzer.py` 仍通过 A 股 legacy resolver 判断 price basis / flow semantics

### 当前问题

最新 HEAD 中 `_price_basis_of_field()`、`_flow_semantics_of_field()` 会：

```python
from fields import resolve_field
```

该默认 resolver 仍是 A 股 legacy registry。

### 后果

US `revenue`、`close`、`pre_close` 等在 typed IR 阶段可能被 A 股 alias/FieldSpec 命中，直接污染：

- `semantic_kind`
- `price_basis`
- `flow_semantics`
- typed input gate
- hash/evidence

### 必须整改

IR construction 必须携带 `MarketContext` / resolved provider semantic type。

禁止 analyzer 自己再次按字段名查一个全局 registry。

建议 IR leaf semantic attrs 直接来自 resolved source plan：

```text
market
concept_id
provider_id
semantic_kind
price_basis
flow_semantics
canonical_unit
grain
knowledge_time_policy
coverage_id
```

---

## R17-038：expression support 绕开了 provider 的完整 eligibility gate

### 当前问题

`market/capability_resolver._check_column()` 当前主要检查：

- binding 是否存在；
- quality 是否 production usable；
- coverage_gate。

但没有统一执行：

- `source_certified`
- `required_filters`
- strict PIT
- current snapshot only
- currency filter
- effective-time-only permission
- knowledge-time availability
- provider chain selection

### 必须整改

expression support 不要自己写第二套判断。

必须统一调用：

```text
ProviderResolver.resolve(..., mode=compile_preflight)
```

compile 与 runtime 使用同一 eligibility authority，只允许 runtime 再做动态数据覆盖校验。

---

## R17-039：physical-field fallback 可以绕过 canonical provider / unit / PIT 语义

### 当前问题

如果 column 没映射到 concept，capability resolver 会尝试 per-market physical field registry。

这样 production 用户可能直接写 physical column，绕过：

- canonical unit transform；
- provider quality；
- required filters；
- derived field逻辑。

### 必须整改

production：

1. 如果 physical field 能映射 canonical concept → 强制 canonicalize；
2. 如果是未建模 raw physical field → 默认拒绝；
3. 只有 explicit `research_raw_column(...)` 才能用，且 typed semantic UNKNOWN、不可自动 production certify。

---

## R17-040：`explain_expression_support()` 写死 `surface="daily"`

### 当前问题

字符串公式检查时固定 daily surface，无法正确检查 minute/intraday operator 的 grain/session。

### 必须整改

所有 compile support API 参数至少包含：

```text
market_context
surface/frequency
decision_timestamp
universe_policy
source_profile
```

minute formula 不得通过 daily parser 偷过 grain gate。

---

## R17-041：`operator_support(canonical, market, context=...)` 没强制 context.market 与 market 一致

### 必须整改

若同时保留二者参数：

```python
assert canonicalize_market(market) == context.market
```

更好：只传 context，不重复传 market string。

---

## R17-042：production allowlist 为空时仍可能失去 gate 语义

### 当前问题

逻辑存在：

```python
if prod and name not in prod:
```

如果成功加载的 allowlist 恰好为空，整个检查条件为 false。

### 必须整改

“成功加载空 allowlist”应表示 0 个 operator certified，不是“不限制”。

`None/Unavailable` 与 empty set 语义严格区分。

---

## R17-043：market operator manifest 仍不足以证明“每个 operator 真正审过”

### 当前问题

fallback generic operator 即使没有 explicit contract，也可因为不是 UNKNOWN 而视为 reviewed。

### 必须整改

新 manifest 每个 canonical 必须输出其**解析后的输入约束**，而不仅是 `contract_origin`：

```text
required input semantic kinds
allowed price basis
flow semantics constraints
required grain
required session
required market capabilities
required canonical concepts
required filters
currency sensitivity
universe sensitivity
coverage sensitivity
PIT sensitivity
backend certified paths
```

generic math op 可以是 “input-dependent”，但不能是“没有具体输入约束说明”。

---

## R17-044：search grammar 仍是“独立 operator 列表 + 独立 concept 列表”，无法阻止非法组合

### 当前问题

`build_search_grammar()` 返回 allowed operators 和 concepts 的平面集合。

这无法表达：

```text
这个 operator 的第 1 个参数只能 PriceContinuous
这个 operator 需要 EventBool
这个 operator 不能接 LocalMoney 做跨市场 rank
这个 operator 需要 A 股 limit price
```

### 必须整改

构建 typed grammar graph：

```text
Operator -> ParameterSlot -> allowed semantic kinds/concepts -> market/provider availability
```

AlphaProbe/AlphaMiner 只从 legal edges 采样，尽量在生成前剪枝，而不是生成后报错。

---

## R17-045：A/US market capability 目前是静态 bool，无法表达“有能力但 coverage 不足”

### 当前问题

US capability set 包含 `DAILY_MARKET_CAP`、`DAILY_SHARES`，但其真实 provider coverage 并不是完整全市场。

### 必须整改

capability 变成 provider-backed capability resolution：

```text
capability
provider id
quality
coverage class
requested-window coverage
universe coverage
PIT status
source certification
```

不要用一个全局 frozenset 直接宣告“US 有 DAILY_MARKET_CAP”。

---

## R17-046：A 股 `FINANCIAL_TTM` 不应被当无条件 native capability

### 当前问题

A 股原始财务 flow 多为 cumulative YTD；TTM 是派生，需要足够连续 fiscal history 与正确 single-quarter conversion。

### 必须整改

区分：

```text
FINANCIAL_PIT_STATEMENTS
QUARTER_FLOW_DERIVABLE
TTM_DERIVABLE
TTM_NATIVE
```

A 股 TTM 必须在 period continuity/coverage 满足时动态可用。

---

## R17-047：`TRADABILITY_STATUS` capability 过粗

### 当前问题

A `IsSuspend`、PublicStatus、ST、listing membership 是不同状态；US 又缺等价 Status。

### 必须整改

拆 capability：

```text
SUSPENSION_STATUS
LISTING_STATUS
COMMON_EQUITY_MEMBERSHIP
RESEARCH_UNIVERSE
INTRADAY_HALT
```

算子只依赖自己真正需要的能力。

---

## R17-048：market canonicalization 规则不统一

### 当前问题

`capabilities_for()` 接受 `cn/china/usa` 等别名，而 `MarketContext` 只接受 `ashare/us`。

### 必须整改

做唯一 `canonicalize_market_id()`，入口最前层执行一次；内部对象只保存 canonical market id。

不得某层接受 `china`，另一层报错。

---

## R17-049：MarketContext annualization 固定 252，不是真正 calendar-aware

### 当前问题

`trading_days_per_year` 两市场都硬返回 252。

### 必须整改

annualization operator 应使用：

- resolved market calendar；
- requested window 的 session count；
- 或明确参数化 annualization basis。

对 minute volatility 更不能简单拿 252，不同 session length/early close 要有明确策略。

---

## R17-050：`as_research()` 自动打开 non-PIT effective-only 是危险的

### 当前问题

当前 `as_research()` 会同时：

```text
allow_proxy=True
allow_sparse=True
allow_effective_time_only=True
```

这会让“研究模式”自动允许 A 股 ex-date-only dividend 语义进入历史研究。

### 必须整改

分离：

```text
research_profile
allow_sparse
allow_proxy
allow_non_pit_effective_time_only  # dangerous explicit flag
```

最后一个默认必须 False，即使 research 也 False。

---

## R17-051：decision_timestamp 必须强制 tz-aware，并用于跨市场可见性

### 必须整改

- naive timestamp production 直接拒绝；
- A/US 同一个自然日不能默认互相看到“当天收盘数据”；
- 做跨市场 signal 时按绝对时间判断哪一个 local session 已闭市；
- factor identity/lineage 记录 decision policy。

---

## R17-052：MarketContext 缺少 source/data snapshot lineage

### 必须整改

至少加入或关联：

```text
data_snapshot_id / source_snapshot_time
provider_profile_version
field_registry_hash
market_registry_hash
universe_policy_hash
calendar_version
```

否则同一 formula 在数据契约更新后可能复用旧 cache/evidence。

---

## R17-053：A 股 minute bar 是 bar_end，但 `ASHARE_SESSION` 仍继承 bar_start 默认

### 当前问题

A 股 TableSpec 已明确 `bar_timestamp_role=bar_end`，而 `market/session.py` 的 `SessionSpec` 默认 `bar_convention="bar_start"`，ASHARE_SESSION 没覆盖。

### 必须整改

统一：

```text
A: bar_end, labels 09:31...11:30 / 13:01...15:00
US future full minute OHLCV provider: explicit provider-specific convention
```

所有 slot/id/first-hit/last-hit/time-of-day operator 用同一 session adapter。

---

## R17-054：US early-close `for_date()` 实际没有缩短 session

### 当前问题

当前 early-close branch 返回几乎相同 segments/slot_count，只在 notes 上标记。

### 必须整改

CalendarProvider 对每个 early-close date 提供实际 close time，SessionSpec 当天产生正确：

```text
segments
slot_count
normalized_session_position
expected_bar_count
```

volume/volatility/session profile 算子才能正确 normalize。

---

## R17-055：US session 文档写成 `~51 days/year`，真实是全历史约几十个 true 日

### 必须整改

删除错误描述，测试不要以错误年频率构造假数据。

---

## R17-056：有 US session spec 不代表有 US full minute OHLCV capability

### 当前问题

当前 cleaned US 数据的 `is_ticker_halt` 是稀疏 flag，不是完整 minute OHLCV。

### 必须整改

严格区分：

```text
SESSION_CALENDAR_AVAILABLE
FULL_MINUTE_OHLCV_AVAILABLE
INTRADAY_HALT_FLAG_AVAILABLE
```

当前 US 的 `minute_*/intraday_*/micro_*` 需要完整 OHLCV 的算子一律 provider required，除非未来明确接入并认证完整分钟源。

---

## R17-057：operator market fallback 漏掉 `minute_` prefix

### 当前问题

`operator_market.py` 的 `_MINUTE_PREFIXES` 当前主要是：

```text
intraday_, intra_, session_, micro_
```

而 capability resolver 自己又把 `minute_` 认作 mechanism prefix。

### 必须整改

不要再靠两个文件各维护一套 prefix。

建立 operator metadata：

```text
required_grain=minute
session_dependency=True
```

所有 `minute_*` 动态由 metadata 分类。

---

## R17-058：几个 `intra_limit_*` 只声明了 DAILY_PRICE_LIMITS，漏 FULL_MINUTE_OHLCV

### 当前问题

以下至少重新核验：

```text
intra_limit_duration
intra_limit_first_hit_time
intra_limit_reopen_count
```

它们是分钟级 limit behavior，不可能只靠 daily HighLimit/LowLimit 算出来。

### 必须整改

要求：

```text
DAILY_PRICE_LIMITS + FULL_MINUTE_OHLCV
required_grain=minute
intrinsic_market=ashare
```

并检查所有其它 `intra_*limit*` / `intraday_*limit*` 同类算子。

---

## R17-059：news operator family 必须按“事件”与“sentiment”拆分

逐个检查当前注册的：

```text
news_volume
news_event_age
news_coverage_ratio
news_sentiment
news_sentiment_momentum
news_negative_ratio
news_surprise
```

要求：

- volume/age/coverage：可基于 event feed；
- sentiment/negative：必须 sentiment provider；
- surprise 要明确 surprise 的 baseline 与数值来源，不得因为有 title 就认证；
- A 股全部 provider-required。

---

## R17-060：行业算子不能只看 `INDUSTRY_CLASSIFICATION` capability，必须绑定具体 IndustrySource

逐个检查：

```text
industry_neutralize
industry_rank
industry_peer_mean
industry_peer_median
industry_peer_std
industry_zscore
industry_size_neutralize
industry_size_residual
industry_rolling_pca_loading
intra_industry_lead_lag_ex_self
ts_industry_liquidity_beta
```

A 股每一个结果都必须记录 `IndustrySource`。  
US 当前 provider-required。

如果用户选择 sw_l1 与 sw_l2，必须是两个不同 factor id。

---

## R17-061：index operator 必须把 index id 作为语义参数

逐个检查：

```text
index_weight
index_weight_change
index_weight_gap_to_free_float
index_weighted_peer_mean
index_weighted_neutralize
index_weighted_zscore
以及所有 index_member/index_peer 类算子
```

要求：

- index id 必须显式参数；
- A Weight `/100`；
- US membership 可用、Weight 不可用；
- multi-index operator 若支持，必须显式输出 relation grain，不允许行重复后偷偷聚合。

---

## R17-062：holder family 的数据时间模型要改为 snapshot，不要把 raw top-ten 当普通 relation_pit

### 当前问题

A 股 TopTen* 数据字典模型是 S1 snapshot `(TradeDate, Symbol, Rank)`，内部带 ReportPeriod/PubDate 等 metadata。

### 必须整改

holder adapter 先建立“某日可见 holder snapshot”，再做 holder operators。

逐个检查所有：

```text
holder_concentration*
holder_count_change_rate
holder_entry_share / exit_share / net_entry_share
holder_id_matched_*
holder_rank_stability
holder_class_entropy / holder_nature_entropy
holder_shareholder_overlap_ratio
holder_shareholder_network_centrality
holder_peer_return_breadth
holder_pledge_*
holder_freeze_*
holder_locked_share_ratio
```

单位：

- `ShareRatio` % -> decimal；
- `SharePledge/ShareFreeze` 是 shares；
- FloatTopTen 没有 pledge/freeze；
- top10 ShareRatio 加总不保证恰好 100%，不要强行 renormalize 后冒充实际 ownership share。

US 当前 no-equivalent provider。

---

## R17-063：`holder_concentration` canonical concept 已声明，但 provider layer 不完整

### 必须整改

如果保留 canonical concept：

- A 实现明确 snapshot aggregation provider；
- US unavailable/provider-required；
- 如果不用 concept，删除 dead concept，避免 grammar 暴露一个永远解析不了的 concept。

同类 dead concept 见下一条。

---

## R17-064：concept registry 中存在多项“声明了但没有完整 market binding”的 dead concepts

当前至少全量检查：

```text
free_float_market_cap_local
total_shares
free_float_shares
roa_decimal
gross_margin_decimal
net_profit_margin_decimal
earnings_per_share
dividend_ex_date
news_sentiment
holder_concentration
```

### 必须整改

对每个 concept 二选一：

1. 有真实数据与可执行 provider → 补齐 A/US binding、unit/PIT/coverage；
2. 没有真实 provider → 明确 unavailable/provider_required，不进入 production grammar；
3. 如果没有任何存在意义 → 删除概念与 aliases。

不能出现“concept registry 说存在，但 provider/runtime 永远拿不到”的半成品。

---

## R17-065：`turnover_ratio_decimal.cross_market_comparable=True` 与 US unavailable 自相矛盾

### 必须整改

`cross_market_comparable` 不应该仅由数学单位决定。

建议拆为：

```text
unit_comparable
definition_comparable
provider_available_by_market
cross_market_rank_allowed
```

US 当前无等价 D1 turnover，则不能把 turnover concept 标成实际可跨市场比较。

---

## R17-066：ROE 等“单位一致”不代表 accounting definition 真正一致

### 必须整改

A ROE（%→decimal）与 US X0/derived ROE 在期间口径、年化、分母/归属口径上可能不同。

不要用一个 `cross_market_comparable=True` 就允许 A+US 混合截面 rank。

对 fundamental ratio 增加：

```text
accounting_definition_id
period_basis
annualization_basis
equity_attribution_basis
```

跨市场比较只有 explicit comparator policy 才允许。

---

## R17-067：`index_weight` 被标为 group key，语义角色错误

### 当前问题

index weight 是连续 ratio，不是 categorical group id。

### 必须整改

增加/使用 `ROLE_WEIGHT` 或普通 FEATURE + weight semantic kind。

防止 index weight 被 group-by / categorical path 处理。

---

## R17-068：US SecurityMaster static 与 DailySnap 的 PIT 用途要严格分开

### 当前问题

static SecurityMaster 可能包含当前属性；历史 universe/属性不能默认用 current master 回填。

### 必须整改

- stable identifier mapping 可使用明确不会随时间变化的 static 字段；
- 上市状态、类型、交易所、delist 等历史属性优先 DailySnap；
- `delisted_utc` 不应标成 `ingestion_time`，它是市场实体状态/effective reference time；
- historical selection 不得由 current master 造成 survivorship bias。

---

## R17-069：US `InstrumentKey(market,ticker)` 只能防跨市场重名，不能解决 ticker change/reuse

### 必须整改

对需要跨多年拼接财务/新闻/公司主体的场景，引入 stable security identity：

```text
market + stable_security_id(FIGI/CIK/内部security_id) + effective ticker mapping
```

交易输出仍可用当日 ticker，但 cache/lineage 至少要能证明 ticker mapping 的有效期。

TickerMap/SecurityMasterDailySnap 应用于这条链路。

---

## R17-070：跨市场 monetary field 不允许直接混合 rank/zscore

### 当前问题

`amount_local`、`market_cap_local`、财务绝对金额分别是 CNY/USD。

### 必须整改

如果 A/US 分开跑，没问题；如果 formula 明确构造 joint cross-market panel：

- local money 不允许直接 `cs_rank/zscore/regression`；
- 必须先 FX normalize 或转换成 market-local dimensionless ratio；
- FX provider 本身必须 PIT-safe；
- factor engine 应能识别“同一截面含多个 currency domains”。

---

## R17-071：A 股 `StockCapitalDaily` 不允许无限 state-asof carry

### 当前问题

该表覆盖末端可能落后行情，而且 ChangeDate 本身也可能很旧。

### 必须整改

对 A shares provider：

- 优先同日 exact S1 snapshot；
- 若允许 carry，显式 `max_staleness_days`；
- 计算 `share_data_age`；
- 超过 cutoff → UNKNOWN / provider unavailable；
- size/turnover/free-float 因子 coverage 同步下降。

---

## R17-072：A 股 StockStatus 是否 state-asof 也要改成数据真实粒度驱动

### 必须整改

当前数据自然日有 S1 snapshot。优先按真实日快照 exact join。

只有明确缺日并有 policy 时才 carry，carry age 必须记录。

不能因为 TableSpec 写了 state_asof 就默认无限 forward-fill 状态。

---

## R17-073：financial same-day 可见性策略应从“硬编码假设”升级为 policy

### 当前问题

当前 `FinancialPeriodAdapter.asof(... same_day=False)` 对 date-only knowledge time 采取保守 next-day，是安全的，但它是一个具体策略，不应藏在 adapter 里成为 universal truth。

### 必须整改

定义：

```text
DATE_ONLY_CONSERVATIVE_NEXT_SESSION
TIMESTAMP_EXACT
EXCHANGE_PUBLISH_TIME
```

策略进入 factor identity/evidence。

对当前只有日期的 A PubDate、US declaration/某些 filing date，默认 conservative 是合理的；但必须显式。

---

## R17-074：financial asof 后应保留原 signal row identity / order

### 必须整改

当前 per-instrument concat/sort 容易改变原始 row 顺序。

加入 `__anchor_row_id__`，merge 后恢复原 anchor 顺序；重复 signal rows 也必须稳定。

这不仅是工程洁癖，跨 backend parity/缓存 hash 需要确定性。

---

## R17-075：US X0 Valuation/Indicator 必须在 field-level 也明确 `mining_allowed`/coverage

### 当前问题

TableSpec 标了 current snapshot / strict PIT false，但若 field-level 默认 mineable，某些 grammar/source path 仍可能把它作为候选。

### 必须整改

X0 fields production historical mining 明确：

```text
mining_allowed=False for historical automatic mining
usage=current_snapshot_research only
coverage=CURRENT_ONLY
```

单日 current snapshot research 若允许，必须显式 mode。

---

## R17-076：US `Volatility_20d` 当前数据实际不可用，任何 catalog/recipe 不得认证

### 必须整改

如果物理字段仍在 schema 但实测长期 100% null：

- `provider_quality=UNAVAILABLE/SPARSE_INVALID`
- grammar 不暴露；
- 不能因为 schema 存列就判 capability 有效。

---

## R17-077：schema existence 与 semantic availability 必须分开

### 必须整改

对每个 field/provider 保存：

```text
schema_exists
non_null_coverage
market_coverage
time_coverage
semantic_certified
production_usable
```

例如 US `Volatility_20d`：schema_exists=True，但 semantic availability=False。

---

## R17-078：A 股财务 `NetProfit` 与“归母净利润”不能混为一个 concept

### 当前问题

A `NetProfit` 含少数股东口径，而 `NpParentCompanyOwners` 才是归母；US current binding 用 common shareholders net income。

### 必须整改

canonical 至少拆：

```text
net_income_total
net_income_attributable_to_parent/common_shareholders
```

跨市场盈利质量、ROE、PE-like derived factors 要选择真正同义口径。

---

## R17-079：`equity` canonical 的归属口径也要与 net income 对齐

### 必须整改

A 当前 provider 用 parent equity，US 当前 `total_equity` 可能包含 noncontrolling。

如果 numerator 是 common/parent net income，denominator 也应选 parent/common equity。

对 ROE derived provider 添加 accounting compatibility test。

---

## R17-080：所有 growth/YoY/TTM/fiscal operators 必须使用 fiscal period，不得按交易日 lag 替代

### 必须整改

机器扫描所有名字/metadata 属于 fiscal/growth/quarter/TTM 的算子：

- A：以 `ReportPeriodEndDate` + PubDate knowledge；
- US：`period_end` + filing_date + timeframe；
- 同比应 fiscal period matching；
- 不允许 `ts_lag(x,252)` 冒充财务 YoY；
- revision/period gap 必须显式处理。

---

## R17-081：minute/session operator 的 expected bar count 不能写死跨市场

### 必须整改

所有：

```text
first_hit_time
last_hit_time
duration
reopen_count
intraday profile
volume curve
session entropy
minute autocorr
microstructure aggregation
```

读取 `SessionSpec.expected_bar_count(date, provider)`，不要在 kernel 内写 240/390/391。

A bar-end；US future provider convention 单独声明；US early close 动态缩短。

---

## R17-082：US halt flag 391 分钟点不能与 future 390 bar-start OHLCV 静默 outer/equi join

### 必须整改

若未来同时使用 halt 与 OHLCV：

- 明确 timestamp convention；
- 以绝对 timestamp join；
- 不允许按 slot number 直接假设一一对应；
- 391 point flag 的 end-point semantics 必须单独验证。

---

## R17-083：A 股 limit operators 必须统一用 RAW official limit/reference basis

### 必须整改

全量扫描 `ashare_limit_*`, `limit_up_*`, `limit_down_*`, `intra_limit_*`, `intraday_*limit*`：

- HighLimit/LowLimit 是 raw official limit；
- compare price 也必须 compatible raw basis；
- 不允许 `continuous_close` 与 raw limit 直接比较；
- ST/board limit 不要硬编码 5/10/20%，以当日 HighLimit/LowLimit 为主；
- suspension/invalid bar 先 mask。

---

## R17-084：turnover/chip/free-float family 的 US 语义继续 fail closed，禁止 volume proxy

逐个检查：

```text
average_turnover
abnormal_turnover
turnover_volatility
turnover_autocorr
return_per_turnover
turnover_shock
turnover_acceleration
price_turnover_divergence
return_turnover_beta
turnover_adjusted_volatility
turnover_momentum
turnover_zscore

ts_turnover_reference_price
ts_turnover_cost_dispersion
ts_turnover_profit_share
ts_turnover_holding_age
ts_turnover_near_cost_mass
ts_turnover_cost_quantile_distance
ts_turnover_cost_entropy
ts_turnover_cost_mode_distance
ts_turnover_cost_skew
ts_turnover_age_dispersion

free_float_turnover
free_float_ratio
free_float_share_ratio
free_to_circulating_ratio
market_cap_free_cap_gap
float_share_ratio
true_turnover_rate
real_turnover_rate
```

没有同义 US provider 时，不得用 volume/market cap 猜一个“差不多”的字段。

---

## R17-085：长窗口 absolute continuous-price 算子必须声明 `level_sensitive`

### 必须整改

对以下类算子自动扫描：

```text
spectral/wavelet on price level
trend slope on adjusted level
path geometry
price centroid/reference price
pattern recognition on level
long-range distance metrics
```

metadata 增加：

```text
level_sensitive=True/False
supports_return_substitution=True/False
```

US clamped ticker 时据此决定 provider eligibility。

---

## R17-086：跨 backend 不能只测 numeric parity，还要测 semantic-plan parity

### 必须整改

Pandas/Polars/DuckDB 每个 production operator 需要比：

1. canonical operator；
2. canonical params；
3. field/provider plan；
4. unit transforms；
5. PIT/filter/universe/session masks；
6. null semantics；
7. numeric output。

如果三个 backend 数值都错得一样，单纯 numeric parity 检查不出来。

---

## R17-087：cache/evidence key 必须包含所有改变经济定义的上下文

至少包含：

```text
market
canonical formula AST
canonical parameter values
provider ids + provider contract versions
field registry hash / market registry hash
data snapshot id
universe policy + universe id
IndustrySource / IndexSymbol / IndexName / timeframe / currency filter
period-selection policy
same-day visibility policy
session/calendar version
price basis
flow semantics
coverage policy
```

否则同公式换 source/filter 仍命中旧 cache。

---

## R17-088：`source_certified=True` 不能只是静态布尔常量

### 必须整改

source certification 至少关联：

```text
dataset registry version
schema fingerprint
unit evidence
PIT evidence
coverage evidence
last validated date
provider code version
```

schema/contract 变化必须自动失效旧 certification。

---

## R17-089：production field support 必须使用同一个四/多层 gate

建议最终唯一 gate：

```text
MarketContext
∧ ConceptSpec
∧ ProviderBinding
∧ Source/DataAccess contract
∧ Table/Field PIT
∧ Required filters
∧ Requested-window coverage
∧ Universe membership
∧ Session/calendar
∧ Currency/price basis/flow semantics
∧ Operator input contract
∧ Backend certification
```

任何模块不得再复制一份“简化版判断”。

---

## R17-090：DataAccess 与 FactorEngine schema/contract 必须自动生成 drift report

新建类似：

```text
scripts/audit_fe_dataaccess_contract_drift.py
```

输出至少：

```text
unknown_dataset
unknown_physical_field
time_column_mismatch
instrument_column_mismatch
unit_mismatch
temporal_model_mismatch
required_filter_mismatch
current_snapshot_mismatch
coverage_class_mismatch
market_mismatch
```

CI 必须 0 unresolved。

---

# 4. 对当前每一个 canonical operator 做真正的逐个审计

这部分非常重要。不要只修上面点名的算子。

## 4.1 不要硬编码 operator 总数

运行当前代码：

```python
all_ops = sorted(OperatorRegistry.list_canonical())
```

以运行时实际数量为准。

对**每一个 canonical**生成审计行，并且对应 aliases 也要检查参数和 dispatch 是否落到同一个 canonical。

## 4.2 必须生成新的 R17 operator 审计产物

生成：

```text
factor_engine/docs/R17_OPERATOR_CROSS_MARKET_AUDIT.json
factor_engine/docs/R17_OPERATOR_CROSS_MARKET_AUDIT.csv
factor_engine/docs/R17_OPERATOR_CROSS_MARKET_AUDIT.md
```

每个 canonical 一行，至少包含：

```text
canonical
aliases
implementation module/class
family/category
surface
input arity
parameter names
parameter roles
parameter domains
required semantic kinds per positional input
allowed price basis per input
flow semantics constraints
required grain
required session
level_sensitive
currency_sensitive
cross_section_universe_sensitive
market mechanism dependency
ashare status
us status
ashare required capabilities
us required capabilities
ashare required canonical concepts
us required canonical concepts
ashare chosen provider(s)
us chosen provider(s)
provider quality
coverage class
coverage gate
required filters and values
PIT knowledge time
effective time
period policy
session policy
null/missing semantics
pandas support
polars support
duckdb support
backend parity evidence
production certification
reason if blocked
```

### 硬性要求

- 不允许 UNKNOWN；
- 不允许“fallback_default”后什么都没说明；
- `INPUT_DEPENDENT` 必须列出真正的 typed input requirements；
- market-specific operator 必须有明确 blocked reason；
- field-dependent operator 必须能追到最终 physical provider；
- 每个 production-certified operator 必须有可执行 source path。

---

# 5. 按算子族执行的新增审计规则

这里是“每个算子都要检查”的规则，不是只检查名称里恰好有关键词的几个。

## 5.1 elementwise / arithmetic

逐个检查：

- unit algebra；
- money/price/share/ratio 不可乱加减；
- A+US joint panel monetary fields需要 FX or block；
- NaN/inf/zero denominator policy；
- bool/group/status 不可当 numeric alpha 输入；
- derived semantic kind 是否正确传播。

## 5.2 time-series rolling

逐个检查：

- required history；
- finite/contiguous semantics；
- raw split-sensitive price 是否允许；
- continuous-price provider 是否受 clamp；
- missing row vs non-trading day；
- calendar/session frequency；
- warmup 与 backend 一致。

## 5.3 return decomposition / gap / overnight

逐个检查：

- A Return bp 已 canonicalize；
- US Ret decimal；
- `reference_pre_close` 与 `lag(raw_close)` 分开；
- open/close 必须共享可比较 price basis；
- corporate action day golden；
- same-day signal availability。

## 5.4 cross-sectional rank / zscore / normalize / winsorize

逐个检查：

- universe mask；
- listing/common equity policy；
- suspension/invalid bar；
- partial provider coverage selection bias；
- group size；
- A/US joint panel currency block；
- NaN 不得变 0；
- cross-market definition comparability。

## 5.5 neutralization / regression

逐个检查：

- A industry source exact-one；
- US industry provider-required；
- US size exposure coverage；
- parent/common accounting basis；
- design matrix missing mask；
- rank deficiency；
- residual output alignment；
- universe filter进入 factor identity。

## 5.6 fundamental/fiscal

逐个检查：

- A PubDate / ReportPeriod；
- US filing_date / period_end / timeframe；
- period-selection；
- revisions；
- cumulative YTD vs single quarter vs TTM；
- restatement；
- missing numeric UNKNOWN；
- same-day availability policy；
- parent/common attribution。

## 5.7 event operators

逐个检查：

- “没事件”与“字段缺失”分开；
- event timestamp；
- event effective vs knowledge time；
- future scheduled ex-date 不得提前消费；
- event dedupe；
- event lag/horizon calendar semantics。

## 5.8 dividend operators

A：strict PIT 默认 block effective-only。  
US：declaration_date PIT，currency filter，future ex-date clip。

任何“dividend yield”与“cash dividend event”不要混为一个概念。

## 5.9 intraday/minute/microstructure

A：240 bar_end session。  
US：当前无 certified full minute OHLCV，provider required。  
未来 US provider：390/early-close + provider-specific bar convention。

逐个检查 normalized session position、lunch break、early close、halt、missing minute。

## 5.10 price-limit

A only，使用 official raw limit price；US 不映射 LULD/halt。

## 5.11 industry

A 需要 IndustrySource；US provider required。

## 5.12 index

membership 和 weight 分开；A Weight%，US 无 Weight；index id required。

## 5.13 holder / pledge / freeze

A only；snapshot；ShareRatio%、pledge/freeze shares；FloatTopTen 没 pledge/freeze。

## 5.14 news

US event/text 与 sentiment capability 分开；A 当前 provider required。

## 5.15 capital/share/turnover

A S1 / Valuation shares 与 US TickerSharesSnapshot/split events 完全分开。

## 5.16 spectral/wavelet/path geometry/pattern

长窗口 level-sensitive operator 必须处理 adjusted factor clamp；raw price corporate-action discontinuity 默认不允许进入这类 operator，除非 operator 明确只看单日形态且数学上与 scale-invariant。

## 5.17 relation/network operator

检查 relation grain、duplicate edge、snapshot time、entity identity、one-to-many aggregation，禁止 relation 行直接广播到 daily panel 产生行数爆炸。

---

# 6. 对“每个算子具体怎么改”的机器化规则

代码 AI 不要只生成 audit 文档而不改代码。对 R17 audit 每一行执行以下规则：

## 6.1 如果 operator 的数学实现正确，但 market contract 不完整

修改：

- `OperatorMetadata` / market contract；
- required semantic kinds；
- required grain/session/capability；
- provider dependency；
- tests；
- manifests。

不要为了让它在 US 可用而造 proxy 字段。

## 6.2 如果 operator 的 input semantic type 错

修改 operator signature / typed IR contract，而不是在 kernel 里偷偷转换。

## 6.3 如果 operator 依赖一个不存在 provider 的 concept

- 当前市场标 `PROVIDER_REQUIRED/UNSUPPORTED`；
- grammar 移除；
- production compile fail closed；
- 不删除另一个市场真正有用的算子。

## 6.4 如果 operator 只在特定 source filter 下成立

把 filter 变成 operator/factor identity 的显式语义参数，不允许 reader 暗中默认。

## 6.5 如果 operator 的数学结果依赖 session/universe/period policy

policy 一律进入 semantic hash。

## 6.6 如果 operator 的物理计算可共享但经济定义不同

共享底层 numeric primitive；上层保留不同 semantic wrapper，不要为了“代码复用”强行合并经济定义。

---

# 7. 新增自动化审计脚本

至少新增/完善以下机器化审计；文件名可调整，但职责不能少。

## 7.1 `audit_fe_dataaccess_contract_drift.py`

见 R17-090。

## 7.2 `audit_cross_market_field_resolution.py`

遍历所有 aliases/qualified physical fields，验证：

- A resolve 永远只进 A registry/provider；
- US resolve 永远只进 US；
- 同名表不同语义；
- physical→concept 映射无歧义；
- production 无 bare A fallback。

## 7.3 `audit_provider_execution_graph.py`

每个 provider：

- 所有 dependency dataset 存在；
- physical columns 存在；
- join keys/time model 可执行；
- filters 可执行；
- transform input 完整；
- provider chain 能真的 resolve。

## 7.4 `audit_unit_normalization_once.py`

对所有 source_unit != canonical_unit 的字段做 sentinel，保证 scale 恰好一次。

## 7.5 `audit_pit_and_period_policy.py`

遍历 financial/event/dividend/news/holder/capital source：

- knowledge time；
- effective time；
- period/revision；
- same-day policy；
- future event；
- strict PIT。

## 7.6 `audit_operator_cross_market_matrix.py`

动态生成 R17 全 canonical 审计产物。

## 7.7 `audit_semantic_hash_completeness.py`

随机改变：

```text
market
IndustrySource
IndexSymbol
timeframe
currency filter
universe
provider
period policy
session policy
```

断言 semantic factor hash 必须变化；只改变不影响数学语义的 hash-equivalent 参数才允许不变。

---

# 8. 必须补的关键 golden tests

这些 tests 是本轮是否完成的核心证据。

## 8.1 A/US 单位

```text
A Return 100 bp -> 0.01
US Ret 0.01 -> 0.01
A Roe 12.5% -> 0.125
US ROE 0.125 -> 0.125
A Weight 3.2% -> 0.032
```

## 8.2 Factor / AdjFactor

```text
A continuous_close == Close * Factor
US continuous_close == Close * AdjFactor
```

并加入 US clamped instrument policy test。

## 8.3 PreClose

构造 corporate-action day：

- `reference_pre_close != lag(raw_close)`；
- return/gap operator 用各自正确 basis。

## 8.4 A financial PIT

同一 Symbol：

- 多 PubDate；
- 同 PubDate 多 ReportPeriod；
- revision；
- 信号日边界。

断言不跨股票串线、不未来、不随机选 period。

## 8.5 US financial PIT

同一 ticker 同时含 quarterly/annual/TTM；不指定 timeframe 必须报错。

## 8.6 A dividend

strict PIT 下使用 ExDividendDate-only provider 必须报 blocked。

## 8.7 US dividend

- declaration_date 后才可见；
- non-USD 无 FX 时 blocked；
- future ex-date 不提前进 event factor。

## 8.8 Industry

- 不传 IndustrySource → production 报错；
- 传两个 source → 报错；
- sw_l1 与 sw_l2 factor hash 不同。

## 8.9 Index

- 不传 IndexSymbol/Name → 报错；
- membership transform 输出 bool；
- US weight operator blocked。

## 8.10 Capital

- US split event reader 绝不能读 shares schema；
- shares reader 绝不能读 split schema；
- FE dataset names 与 DataAccess current registry 对齐。

## 8.11 News

- `tickers` explode；
- `published_utc` UTC；
- after close -> next session；
- `news_sentiment` 无 sentiment provider 时 compile blocked；
- `news_volume` 可在 event provider 下通过。

## 8.12 Session

A：09:31 bar_end -> slot 1；11:30 slot 120；13:01 slot 121；15:00 slot 240。  
US future provider：bar-start 09:30..15:59；16:00 不属于 390 bars。  
US early close：slot_count 动态缩短。

## 8.13 Cross-market resolver

同名：

```text
StockValuationDaily
StockIndicator
StockCapitalDaily
StockIndustry
StockStatus
Return/Ret
Factor/AdjFactor
```

必须在 A/US 得到完全不同且正确的 Table/Field/Provider contracts。

## 8.14 Coverage

US size：用 investable CS universe 的 requested-window coverage；低于 policy → production blocked / restricted-universe explicit opt-in。

## 8.15 Backend semantic parity

相同 formula 在 pandas/polars/duckdb 的：

- chosen provider；
- filters；
- unit transform；
- mask；
- PIT anchor；
- numeric output；

全部一致。

---

# 9. 必须重新检查但不要盲目“增加”的 operator metadata

对全部 canonical 扫描以下字段是否真正声明，而不是 name heuristic：

```text
ParamSpec / ParamRole
input semantic kinds
output semantic kind
required history
required grain
required market capability
price basis
flow semantics
currency sensitivity
level sensitivity
missing policy
backend support
production evidence
```

任何 production operator 如果关键语义仍靠：

```text
名字包含 window/lag/minute/industry/holder/...
```

来推断，继续迁移成 metadata。

prefix 只可作为 audit fallback，不可作为最终经济契约。

---

# 10. 防止“整改后又生成新旁路”的架构约束

## 10.1 Field resolution

最终只允许：

```text
MarketContext -> MarketFieldResolver -> Concept -> ProviderResolver -> SourcePlan
```

禁止 production 其它模块自行读取 `catalog.py/catalog_us.py/providers.py` 后拼自己的语义。

## 10.2 Unit normalization

只允许一个 normalization owner。

## 10.3 PIT

只允许一个 temporal eligibility engine。

## 10.4 Coverage

只允许一个 coverage engine。

## 10.5 Market support

operator support 必须由：

```text
operator contract + resolved input providers + market context
```

共同决定。

## 10.6 Hash / evidence

semantic identity 的字段集中定义，任何新增影响经济语义的 context 字段必须自动进入 hash schema。

---

# 11. 全量回归要求

不要只跑本轮新增 tests。

至少完成：

1. FactorEngine 全量 test suite；
2. DataAccess 与 FactorEngine integration tests；
3. pandas/polars/duckdb 全 backend parity；
4. SQL pushdown 与 fallback parity；
5. full operator registry load；
6. alias/canonical consistency；
7. production allowlist/evidence rebuild；
8. operator catalog/manifest rebuild；
9. cross-market field manifest rebuild；
10. R17 新 audit 全部 0 unresolved；
11. cold-start / mining grammar 能在 A/US 分别生成，只包含真实可执行 input/operator；
12. deterministic hash 重跑一致。

任何“测试失败因为并发会话在改，所以先忽略”不算最终完成。最终交付时工作区在你控制范围内的永久测试必须全绿。

---

# 12. 最终 Definition of Done

本轮只有在以下全部满足时才算完成：

- [ ] production 不再有任何隐式 A 股 `FIELD_REGISTRY` fallback；
- [ ] FE 与 DataAccess dataset/schema 无 drift；
- [ ] US Capital split/shares/snapshot 三类语义彻底分离；
- [ ] US Dividend TableSpec 与真实 schema 完全一致；
- [ ] FactNews title/tickers/published_utc 语义正确；
- [ ] News event 与 sentiment capability 分开；
- [ ] industry/index/timeframe/currency required filters 真正执行；
- [ ] index membership 真正输出 bool；
- [ ] tradability concept 不再把 A IsSuspend 与 US CS+Close 混成一个语义；
- [ ] US market cap provider graph、coverage、fallback 可执行；
- [ ] financial period/timeframe/flow semantics 是 provider-resolved；
- [ ] latest IR analyzer 不再从 A legacy resolver 给 US 打语义标签；
- [ ] missing financial != NO_EVENT；
- [ ] coverage contract market/provider/universe/window aware；
- [ ] unit normalization exactly once；
- [ ] A minute bar-end 与 US session/early-close 正确；
- [ ] US 没有 full minute provider 时 minute OHLCV operator 全部 blocked/provider-required；
- [ ]所有 price-limit/industry/index/holder/news/fiscal/capital operator family 逐个完成真实数据契约审计；
- [ ] 当前 registry **每一个 canonical operator** 都出现在 R17 operator audit，0 unknown/unreviewed；
- [ ] 每个 production-certified operator 都能追溯到具体 A/US physical provider 与合法 source plan；
- [ ] 所有 semantic context 都进入 factor/evidence identity；
- [ ] pandas/polars/duckdb 不只是数值一致，semantic plan 也一致；
- [ ] 所有永久回归测试全绿；
- [ ] 重生成 catalog / manifest / evidence 后无 stale artifacts。

---

# 13. 代码 AI 的最终执行指令

现在请你直接在服务器/本地工作区中执行，不要只回复分析：

1. 先读取当前 `factor_engine`、`dataaccess`、两套市场数据字典以及当前生成的 manifests/evidence；
2. 逐条落实 R17-001 ～ R17-090；
3. 动态枚举当前所有 canonical operators，生成 R17 全 operator cross-market audit；
4. 对 audit 中发现的每一个新增问题继续修代码，不要只记录；
5. 修完后再重新跑 audit，直到 0 unresolved；
6. 全量跑 tests；
7. 重生成所有受影响的 operator catalog、market manifest、field manifest、backend evidence、recipe evidence、production allowlist/certification artifacts；
8. 删除临时 debug/一次性修复脚本，只保留永久有价值的 audit/CI；
9. 最终给我一个简洁的结果汇总：改了哪些模块、当前 operator 数、A/US 各自 production/research/provider-required 数量、全量测试结果、audit 0 unresolved 的证据。

**不要把任务停在“我发现还有这些问题”这一步。你要根据本文件把发现的问题继续改完，并通过机器化全量审计证明没有剩余未分类项。**
