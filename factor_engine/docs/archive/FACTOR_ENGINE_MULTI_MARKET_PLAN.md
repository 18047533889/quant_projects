# FactorEngine A股 + 美股统一多市场架构改造方案（执行计划）

> 依据：用户提供的《FactorEngine A股+美股统一多市场架构最终改造方案》（122 节）、
> `/home/shw/COS_ashare_lqtp_data_dictionary.md`、`/home/shw/COS_us_massive_data_dictionary.md`。
> 本文件是**可执行计划**：每一节对应具体文件/函数，执行顺序从 P0 到 P2。

## 0. 现状审计（2026-08-08）

已具备的底子（无需重建）：
- `fields/spec.py` 已有 `FieldSpec/TableSpec`：`source_unit/canonical_unit/scale_to_canonical`、
  `knowledge_time/effective_time/period_id`、`grain`、`cardinality`、`strict_pit_allowed`、`required_filters`。
- `fields/units.py`：`ratio/percent/basis_point/CNY/share` 的归一化（`1.0 == 100%`）。
- `docs/canonical_data_fields.json` 已含 `markets.us_stock` 与 `markets.ashare` 双市场物理表描述，
  由 `scripts/build_canonical_fields.py` 生成。
- `api/mining_integration.py` 已有 `export_dsl_allowlist_json(market=us/ashare)`、
  `default_us_stocks_sip_day_aggs_data_source_config()` 等美股数据源预设。
- `cleaned_operators/`：1293 个 canonical，`OperatorRegistry`，`OperatorSpec` 生产契约，
  daily surface（`DAILY_FACTOR_MIGRATED`），DSL 白名单。

**必须修的问题（用户 P0，进入新架构前先定死）：**
1. `fields/__init__.py` 的 `FIELD_REGISTRY` 只由 `ASHARE_FIELD_SPECS/ASHARE_TABLE_SPECS` 创建 —— 不是 multi-market。
2. `units.py` 无 USD、无带币种的价格类型。
3. `build_canonical_fields.py::LQTP_ALIASES` 仍把 `volume → Volume / Factor`、`open/high/low/close → * Factor`、
   `high_limit/low_limit → * Factor`、`ret → Return`（未 /10000）当 canonical 别名 —— 这是旧 LQTP 语义，
   **不得进入新的 canonical semantic layer**。
4. `docs/field_manifest.json` 是 188 字段扁平并集 —— 只能用于搜索，不能作 market eligibility 权威。

## 1. 分层原则

```
Physical Data
   ↓  Market Adapter / Provider
Canonical Field Concept
   ↓  Generic Operator
Market Capability + Expression Gate
```

数学算子（`ts_mean/ts_corr/RSI/...`）不知道 `Return/Ret/PubDate/filing_date/CNY/USD/HighLimit`。
这些在进入算子之前，由 Provider 统一。

## 2. 文件落地

```
factor_engine/
  market/
    __init__.py
    context.py            # P0-A  MarketContext + ashare_ctx/us_ctx
    capabilities.py       # P0-A  MarketCapability / ProviderQuality / CoverageClass / MarketStatus
    instrument.py         # P0-A  InstrumentKey(market, instrument)
    session.py            # P1     SessionSpec (ashare 240 slots / us DST·early-close)
    capability_resolver.py# P1     explain_field/operator/expression_support
  fields/
    units_v2.py           # P0-B  币种感知 UnitSpec（additive，不动 units.py 旧常量）
    concepts.py           # P0-C  FieldConceptSpec + 概念注册表
    catalog_us.py         # P0-C  US FieldSpec/TableSpec
    providers.py          # P0-C  MarketFieldBinding + provider 注册表 + 单位/PIT 适配
    market_registry.py    # P0-C  MultiMarketFieldRegistry
  cleaned_operators/
    operator_market.py    # P1     算子市场能力声明 + price_basis + 解析
  scripts/
    build_market_field_manifest.py       # P2 → docs/market_field_manifest.json
    build_operator_market_capabilities.py# P2 → docs/operator_market_capabilities.json
    audit_cross_market_semantics.py      # P2  CI 断言
  docs/
    market_field_manifest.json
    operator_market_capabilities.json
    operator_market_matrix.md
  tests/
    market/
      test_market_context.py
      test_units_currency.py
      test_catalog_us.py
      test_concepts_bindings.py
      test_capability_resolver.py
      test_operator_market_manifest.py
      test_cross_market_golden.py
```

所有新增模块**自包含注册**（模块尾 `_register_*()`），避免改共享文件。对共享文件的改动仅限：
`fields/__init__.py`（追加导出 multi-market registry，不动 `FIELD_REGISTRY` 旧名）、
`fields/units.py`（追加 USD/price 常量，additive）。

## 3. P0-A：市场语义基础

### 3.1 `market/context.py`
```python
Market = Literal["ashare", "us"]
@dataclass(frozen=True)
class MarketContext:
    market: Market
    currency: str            # CNY / USD
    timezone: str            # Asia/Shanghai / America/New_York
    calendar_id: str
    session_id: str
    decision_timestamp: pd.Timestamp | None = None
    strict_pit: bool = True
    provider_profile: str = "production"   # production / research
    allow_proxy: bool = False
    allow_sparse: bool = False
    allow_effective_time_only: bool = False
ASHARE_CONTEXT / US_CONTEXT 两个单例。
```
禁止 `if ".SH" in symbol` 猜市场；调用者必须显式传 market。

### 3.2 `market/capabilities.py`
- `MarketCapability(Enum)`：`DAILY_OHLCV / ADJUSTED_PRICE / DAILY_MARKET_CAP / DAILY_SHARES /
  FREE_FLOAT_SHARES / DAILY_TURNOVER / INDUSTRY_CLASSIFICATION / TRADABILITY_STATUS /
  DAILY_PRICE_LIMITS / FULL_MINUTE_OHLCV / TICK_TRADES / L1_QUOTES / L2_ORDERBOOK /
  INDEX_MEMBERSHIP / INDEX_WEIGHTS / FINANCIAL_PIT / FINANCIAL_QUARTERLY / FINANCIAL_TTM /
  DIVIDEND_ANNOUNCEMENT / DIVIDEND_EFFECTIVE_EVENT / TOP_HOLDERS / HOLDER_PLEDGE /
  NEWS / SECURITY_MASTER / EARLY_CLOSE_CALENDAR / FX`
- `ProviderQuality(Enum)`：`EXACT_NATIVE / EXACT_DERIVED / SEMANTIC_EQUIVALENT /
  PROXY_RESEARCH / SPARSE / PIT_BLOCKED / SOURCE_UNCERTIFIED / UNAVAILABLE`
- `CoverageClass(Enum)`：`FULL / PARTIAL / SPARSE / EVENT_ONLY / CURRENT_ONLY / UNKNOWN`
- `MarketStatus(Enum)`：`CERTIFIED_NATIVE / CERTIFIED_DERIVED / PROVIDER_REQUIRED /
  UNSUPPORTED_MARKET_MECHANISM / PIT_BLOCKED / RESEARCH_ONLY`
- `ASHARE_CAPABILITIES / US_CAPABILITIES` 每市场能力集合（对 1107 canonical 全体生效）。

### 3.3 `market/instrument.py`
```python
@dataclass(frozen=True)
class InstrumentKey:
    market: str
    instrument: str
    # → "ashare:000001.SZ", "us:AAPL"
```
所有 cache/factor-lake/lineage 的 key 升级为 `(market, instrument, trade_date)`。

## 4. P0-B：单位系统 + 修正旧转换

### 4.1 `fields/units_v2.py`（additive）
`UnitSpec(dimension, currency?, denominator?)`，支持：
```python
UnitSpec("ratio")
UnitSpec("money", currency="CNY")   # 不能与 currency="USD" 直接相加
UnitSpec("price", currency="CNY", denominator="share")
UnitSpec("price", currency="USD", denominator="share")
UnitSpec("count")                   # shares
UnitSpec("boolean") / ("date") / ("datetime")
```
带 `is_compatible_with(other)` / `dimension_of()`；跨币种 money/price 相加 → raise。
`percent/bp` 只作 source unit；进 canonical 层一律 ratio。

### 4.2 修正旧转换（概念命名 + 绑定，不破坏旧名）
`fields/concepts.py` 定义概念并给 alias：
| 概念 | A provider | A transform | US provider | US transform |
|---|---|---|---|---|
| `return_decimal` | `StockDailyBar.Return` | `x * 0.0001` | `StockDailyBar.Ret` | identity |
| `raw_close` | `Close` | identity | `Close` | identity |
| `continuous_close` | `Close * Factor` | backward multiplier | `Close * AdjFactor` | backward multiplier |
| `raw_volume_shares` | `Volume` | identity（**禁止 /Factor**） | `Volume` | identity |
| `raw_high_limit` / `raw_low_limit` | `HighLimit` / `LowLimit` | identity（**复权价禁用**） | — | `UNAVAILABLE`（美股无日线静态涨跌停） |
| `turnover_ratio_decimal` | `TurnoverRatio` | `x / 100` | provider-dependent | — |
| `market_cap_local` | `MarketCap` | identity | `Close * weighted_shares`（derived） | — |
| `industry_group` | `StockIndustry` | identity | `PROVIDER_REQUIRED`（GICS/NAICS 未接） | — |

`scripts/build_canonical_fields.py::LQTP_ALIASES` 的旧语义仅作 legacy 文档；`fields/concepts.py`
的 binding 才是新 canonical 语义真相。禁止用 `if market == "ashare"` 写死算子。

## 5. P0-C：概念 + 双市场 catalog + provider

- `fields/concepts.py`：`FieldConceptSpec(concept_id, domain, value_kind, canonical_unit,
  frequency, grain, role, cross_market_comparable, market_local_only, allowed_operator_families)`
  覆盖：价量（raw/continuous 分列）、股本、估值、财务、行业、可交易状态、涨跌停、指数、股息、新闻、股东。
- `fields/catalog_us.py`：US `TableSpec`（StockDailyBar/StockMinuteBar(StockDailyBar minute)/StockList/
  SecurityMaster/DimCalendar/FactReturnsDaily/PanelDaily/DimSecurityMaster/StockIndicesComponents/
  fundamentals 财务/StockDividend(declaration_date)/FactNews/short_interest）与 `FieldSpec`
  （Ret=ratio decimal、AdjFactor、volume、market_cap、filing_date、declaration_date、cash_amount+currency）。
- `fields/providers.py`：`MarketFieldBinding(concept_id, market, provider_id, dataset,
  physical_fields, transform, quality, coverage, source_unit, canonical_unit, temporal_model,
  knowledge_time, effective_time, available_at, required_filters, source_certified)`。
  注册 `return_decimal / raw_* / continuous_* / volume / amount / market_cap / turnover /
  industry / tradability / price_limits / financial_pit / dividend / news / holder` 等核心绑定。
- `fields/market_registry.py`：`MultiMarketFieldRegistry`，`resolve(market, concept)` →
  binding；`explain_field_support(concept, market)` → status + reason。
- **FinancialPeriodAdapter**（`fields/providers.py`）：A `PubDate` asof / US `filing_date` asof；
  先 filter timeframe 再 asof，绝不允许季度/年报/TTM 混截面。财务算子市场能力逐 dependency 判定。

## 6. P1：算子市场契约 + CapabilityResolver

### 6.1 `cleaned_operators/operator_market.py`
给 `OperatorMetadata` 追加（additive，默认安全值）：
```python
market_capability: tuple[str, ...] = ()        # required MarketCapability 名
intrinsic_markets: tuple[str, ...] = ("ashare", "us")
price_basis: str = "EITHER"                     # RAW / CONTINUOUS / RAW_OFFICIAL_LIMIT
accounting_period: str | None = None
cross_market_comparable: bool = True
output_unit_rule: str | None = None             # "unit(x)" / "ratio" / "unit(y)/unit(x)"
```
`DAILY_PRICE_LIMITS` 一族算子：`intrinsic_markets=("ashare",)`, `price_basis="RAW_OFFICIAL_LIMIT"`。
`group_*`（显式 group 参数）vs `industry_*`（隐式 resolve industry_group）分开判能力。

### 6.2 `market/capability_resolver.py`
- `operator_support(op, market, production=True)`：generic math/TS/CS → both；
  capability 要求缺 → `PROVIDER_REQUIRED` + missing capabilities；市场机制算子 → `UNSUPPORTED_MARKET_MECHANISM`。
- `explain_expression_support(expr, market, production=True)`：对 DSL AST 逐节点传播，
  返回 `{"supported": bool, "failed_nodes": [{node, reason}]}`；compile-time 拒绝。
- `build_search_grammar(market)`：A 股 grammar（price limits/industry/shareholders/minute/index weights/
  A-share state）、美股 grammar（news/SecurityMaster/short/early-close）、共享 grammar。
- 支持级别含 runtime coverage：`provider_quality_floor`（production 最低 `EXACT_DERIVED`，
  proxy/sparse → research-only + warning）。

## 7. P2：全量 manifest + CI + golden tests

### 7.1 `scripts/build_market_field_manifest.py`
生成 `docs/market_field_manifest.json`：每 concept → `{ashare:{status,dataset,physical,transform,
source_unit,coverage,pit}, us:{...}}`（即 spec §79）。

### 7.2 `scripts/build_operator_market_capabilities.py`
对 **全部 canonical**（`OperatorRegistry.list_canonical()`，1293 个）生成
`docs/operator_market_capabilities.json`：
- generic math/TS/CS/技术指标 → auto both；
- field-dependent → 从 child provider 传播；
- 市场机制算子（limit/ST/ashare 特有）→ 手动 capability 声明；
- `UNKNOWN/NOT_REVIEWED/DEFAULT_BOTH` = 0。

### 7.3 CI 门槛
```python
set(OperatorRegistry.list_canonical()) == set(operator_market_capabilities_manifest)
```
缺 market contract 的算子注册 → CI fail。另生成 `docs/operator_market_matrix.md` 页面。

### 7.4 golden / fail tests（`tests/market/`）
- **Unit**：A `Return/10000 == 0.02`，US `Ret == 0.02`；A `Roe/100`，US ROE identity；
  A `DividendRatio/100`，US dividend_yield identity。
- **Adjustment**：A `continuous_close == Close*Factor`；US `== Close*AdjFactor`。
- **Market Cap**：A `MarketCap ≈ Close*Capitalization`；US `Close*weighted_shares`。
- **PIT**：A finance 用 `PubDate`，US 用 `filing_date`。
- **Table collision**：`StockValuationDaily/StockIndicator/StockCapitalDaily/StockIndustry/StockStatus`
  同名跨市场不得拿错 contract。
- **Fail**：A `Return` 未 /10000、US `Ret/10000`、CNY MarketCap + USD MarketCap、
  US `industry_neutralize` 无 provider、US `limit_up_touch`、US `top_holder_concentration`、
  strict PIT A dividend、US sparse valuation 冒充 full D1 —— 全部必须失败。
- **Cross-market identity**：同一 decimal return 0.02 经 adapter 后 A==US==0.02，
  喂 `ts_mean/ts_std/RSI/ts_rank` 输出数值一致（市场差异完全封装在 adapter）。

## 8. 验收标准

```
canonical 覆盖：1293 / 1293
not_reviewed: 0
unknown market: 0
CI 通过：全量（新 market 测试 + 既有回归）
```

## 9. 执行顺序（本次一次性执行）

1. P0-A `market/` 包（context/capabilities/instrument）＋单测
2. P0-B `units_v2.py` ＋单测
3. P0-C `concepts.py` → `catalog_us.py` → `providers.py` → `market_registry.py` ＋单测
4. P1 `operator_market.py` + `capability_resolver.py` ＋单测
5. P2 两个 build 脚本 + CI 断言 + `operator_market_matrix.md`
6. golden/fail 跨市场测试全绿
7. `fields/__init__.py`/`units.py` 最小 additive 接线（追加导出）

## 10. 执行状态（2026-08-08 已完成）

- [x] P0-A：`market/context.py`（MarketContext + ASHARE/US 单例）、`market/capabilities.py`
  （MarketCapability 26 项 / ProviderQuality 8 级 / CoverageClass 6 级 / MarketStatus 7 级 /
  每市场能力集合）、`market/instrument.py`（InstrumentKey 市场命名空间）、
  `market/session.py`（A 240-bar 连续 / US DST·early-close-aware）。
- [x] P0-B：`fields/units_v2.py`（币种感知 UnitSpec：money CNY/USD、price CNY·USD/share，
  跨币种算术禁止；percent/bp 仅作 source unit，canonical 层统一 ratio）。
- [x] P0-C：`fields/concepts.py`（41 个 canonical 概念 + legacy 别名映射）、
  `fields/catalog_us.py`（16 张 US 表 / 70+ 字段，单位与 PIT 语义按 COS US 字典核验）、
  `fields/providers.py`（MarketFieldBinding + ProviderRegistry + FinancialPeriodAdapter +
  explain_field_support）、`fields/market_registry.py`（MultiMarketFieldRegistry，A/US 独立
  FieldRegistry，同名表隔离）。
- [x] P1：`cleaned_operators/operator_market.py`（price-limit/industry/holder/index-weight/news
  契约注册）、`market/capability_resolver.py`（operator_support / explain_operator_support /
  explain_expression_support / build_search_grammar / build_market_operator_manifest；
  生产 fail-closed + 编译期拒绝）。
- [x] P2：`scripts/build_market_field_manifest.py` → `docs/market_field_manifest.json`（41 概念）、
  `scripts/build_operator_market_capabilities.py` → `docs/operator_market_capabilities.json`
  （1311 canonical，UNKNOWN=0 / NOT_REVIEWED=0）+ `docs/operator_market_matrix.md`、
  `scripts/audit_cross_market_semantics.py`（CI：registry==manifest）。
- [x] 接线：`fields/__init__.py` 追加导出 multi-market 层（legacy `FIELD_REGISTRY` 不动）。
- [x] 测试：`tests/market/` 72 项全绿（含 golden unit/adjustment/cross-market identity/
  fail/PIT/table-collision/session 测试）。

**已知非本次回归**（并发会话在改）：
- `tests/fields/test_field_catalog_v2.py::test_data_access_normalizes_registered_units_without_filling_nan`
  —— 外部 `data_access` 包（2026-08-08 被并发会话更新）将 `turnover_ratio` 标为
  `is_scale_applicable=True`，该测试直接调 `_normalize_contract_columns` 绕过 `store.read()`
  导致 stale。
- `tests/planner/test_sql_io.py` 3-4 项 —— 并发会话正修改 `backend/sql_pushdown/emitter.py`
  与 `backend/sql_tiers.py`，`ts_sharpe`/prefetch 相关断言随其进行中编辑波动（两次运行失败数
  从 4→3，证明是进行中状态）。

## 11. 第三轮复审修复单执行状态（2026-08-08 已执行）

第三方复审（1293 canonical / 12 P0 + P1/P2）全部按单修复：

**P0（12 项全清）**：
- P0-1 `FinancialPeriodAdapter.asof` 加 `by=` instrument 列，逐股分组 merge_asof，
  **跨股票财报串线结构性杜绝**（pandas 2.3 by-asof 需时间列全局单调，改逐组拼接）。
- P0-2 financial concept 的 binding dataset 改为真实报表库（A:
  `ashare_stock_income/balance/cashflow`；US: `us_stock_income/balance/cashflow`）。
- P0-3 derived provider 真正可执行：`continuous_close = Close*Factor`、US market_cap
  = `weighted_shares*Close` 用 `_mul_two` 表达式（原为 identity 假乘法）。
- P0-4 `_production_allowlist` 加载失败 → `ProductionCertificationUnavailable`（fail-closed，
  不再静默返回空集绕过 gate）。
- P0-5 `group_tail_lead_score` 补 `f <= r`（去掉未来函数）+ 前缀不变性测试。
- P0-6 `cs_weighted_percentile_rank` unsort 修正（`unsorted[order]=rank_out`，手算 golden）。
- P0-7 `ts_score_rank_weighted_mean` 权重对齐（`w*tv` 原序，手算 golden 24.2857）。
- P0-8 Corwin-Schultz β/γ 换正（纯 spread 构造 golden 区分新旧公式）。
- P0-9/10 TE/CTE：参数可行性验证（window-lag < min_transitions → raise），NaN-gap 不再
  先压缩再 lag（原时间轴先 lag 后 mask）。
- P0-11 lazy/eager 单位 parity：lazy scan 路径（store.scan 不做归一化）对 catalog 覆盖
  字段补乘 scale。
- P0-12 `_diagram_w1` Hungarian 改用 extended-diagram 标准构造（非零非法边），
  `a=[(1,3)],b=[(1,3),(10,20)] → 2.5` golden。

**P1 已修（本会话）**：P1-1 严格 `market_context`（去 ternary fallback）、P1-3
`source_certified` 进 production gate、P1-4 `CERTIFIED_PARTIAL`（<FULL 覆盖率不再冒充
CERTIFIED_DERIVED）、P1-5 provider chain（`bindings()` 有序链）、P1-6
`LOCAL_MONEY/LOCAL_PRICE_PER_SHARE` + `resolve_unit`（概念层不再硬编码 CNY）、P1-7
`build_search_grammar` 用 `explain_field_support`、P1-8 manifest `contract_origin`
（explicit/typed/fallback 诚实记账）、P1-9 `INPUT_DEPENDENT`（generic 不再伪装
CERTIFIED_NATIVE）、P1-14 结构化 `FilterRequirement`、P1-16 tz-aware
`slot_for_timestamp`、P1-17 US 390-bar bar-start 边界（16:00→None）、P1-18
`session_for_date`、P1-25/27 输出单位修正、P1-28/29 pagerank 改名
`group_signal_attraction_share` + damping 移出搜索面、P1-32/33 feature_geometry 冗余
window 移除、P1-42/45 平均 tie rank、P1-43 `cs_knn_tangent_residual` 移除无效 target、
P1-44 copula global_state 标签、P2-1 wavelet 取最近 m 点、P2-3 标准 level-2 signature、
P2-5 Haar MODWT dilation、P2-7 expectile_beta 单位 `unit(y)/unit(x)`。
（P1-26/31、P0-005、P1-010 等由并发会话同步修复。）

**已核验不在当前树 / 无需改**：`ts_feature_eigen_gap`/`ts_local_linear_intercept`/
`cs_linear_locally_coherent_beta_trimmed`/`ts_directional_change_extent`/`update_clock_density`/
`ts_event_interval_count|ordinal` 在当前 main 不存在对应算子（DC 族是 `ts_dc_*`、
event_interval 族是 `event_interval_*`、update_clock 族是 `update_*`，其对应语义已正确）。

**测试**：`tests/market/` 78 项通过（新增 financial asof 4 项 + golden 6 项 + session
修复），operator 数学/治理/SQL-geometry 全绿。`docs/operator_market_capabilities.json`
(1311 canonical, unknown=0, not_reviewed=0) + `docs/market_field_manifest.json` (41 concepts)
已重生成，`scripts/audit_cross_market_semantics.py` CI OK。

**第三轮非回归**（并发会话 in-flight）：`tests/operators/test_recipe_*` 19 项 ——
`backend/recipe_evidence` 的 evidence hash 与当前 `emitter.py`/`operator_policy.py`
(并发会话正在编辑) 不匹配 → `evidence_artifact_valid()=False`（即 P1-46 evidence 过期，
由持有 emitter/evidence 层的会话收尾）；`test_data_access_normalizes...` 同前。
