# FactorEngine 新增算子开发总规范
## 直接执行版｜A 股因子挖掘｜R47 候选池 → Current Main Gap Closure

**基线日期：2026-08-11**  
**目标仓库：`18047533889/quant_projects`**  
**代码真值：GitHub 最新 `main`，不是 R47 ZIP 内的 operator snapshot。**  
**本次任务范围：FactorEngine 新增/补齐算子及必要的 DataAccess 契约适配；不得凭空增加 A 股原始字段。**

---

# 0. 给代码 AI 的执行指令

请把本文件当作**直接开发任务书**，不要只输出分析、建议、TODO 或伪代码。

执行顺序必须是：

1. 拉取并锁定最新 `main` HEAD。
2. 对 current main 做 operator preflight，先识别已有/等价/可组合/真正缺口。
3. 只实现 `TRUE_GAP`，不得重复造轮子。
4. 先做语义正确的 Pandas reference，再按本文件 backend policy 实现 Polars / DuckDB。
5. 补齐 DSL、registry、surface、PIT、lookback、docs、evidence、tests。
6. 运行完整相关测试与因果性测试。
7. 对所有本轮实现算子生成最终实现清单、backend matrix、测试结果和 remaining blockers。
8. 不得因为某个复杂算子难实现，就停在“建议后续实现”；应继续完成能完成的 P0/P1 真缺口，并把真正受数据契约阻塞的项明确标记 `BLOCKED_BY_DATA_CONTRACT`。
9. **不要为了数量实现重复算子。** 如果旧候选已经被 current main 的 canonical 或组合表达式覆盖，应记录映射并跳过开发。
10. 本任务不允许创建不存在于用户 A 股数据字典中的原始字段。

---

# 1. 三个事实源及优先级

## 1.1 代码真值：GitHub `main`

本文件生成时观察到的最新 main HEAD 为：

`8e9893b562f80e083c4baed2e0a15eb4e040cdc9`

但执行本任务时仍必须重新 `git pull` 并记录新的 `git rev-parse HEAD`。若 HEAD 已变化，以执行时 HEAD 为准。

Current-main 自动生成 operator catalog 在本文件生成时已显示：

- canonical 总数：**1362**
- `surface=daily`：**1185**
- `research`：92
- `unsafe`：7
- `legacy`：1

因此 R47 包中的 966 active-operator contract **只能作为历史快照**，不能拿它判断“现在有没有这个算子”。

必须优先检查：

- `factor_engine/cleaned_operators/docs/operators_catalog.md`
- `factor_engine/docs/dsl_allowlist.json`
- `factor_engine/docs/dsl_operators_reference.md`
- `factor_engine/cleaned_operators/registry.py`
- `factor_engine/cleaned_operators/operator_surface.py`
- `factor_engine/cleaned_operators/production_hardening.py`
- `factor_engine/cleaned_operators/_aliases.py`
- `factor_engine/backend/polars_expr_emitter.py`
- `factor_engine/backend/sql_pushdown/emitter.py`
- `factor_engine/docs/sql_pushdown_coverage.md`
- `factor_engine/api/operator_registry.py`

## 1.2 字段真值：用户上传的 A 股 COS 数据字典

字段只能来自用户现有 A 股数据。当前字典覆盖 **20 张表**：

`Calendar`, `StockDailyBar`, `StockMinuteBar`, `StockList`, `StockStatus`, `StockIndustry`, `StockBalance`, `StockIncome`, `StockCashFlow`, `StockIndicator`, `StockValuationDaily`, `StockCapitalDaily`, `StockDividend`, `StockTopTenShareholder`, `StockTopTenFloatShareholder`, `ETFDailyBar`, `ETFList`, `IndexDailyBar`, `IndexList`, `IndexConstituent`

各表物理字段数（按当前字典解析）：

| Table | Physical fields |
|---|---:|
| `Calendar` | 3 |
| `StockDailyBar` | 16 |
| `StockMinuteBar` | 11 |
| `StockList` | 6 |
| `StockStatus` | 12 |
| `StockIndustry` | 6 |
| `StockBalance` | 111 |
| `StockIncome` | 53 |
| `StockCashFlow` | 59 |
| `StockIndicator` | 36 |
| `StockValuationDaily` | 19 |
| `StockCapitalDaily` | 7 |
| `StockDividend` | 8 |
| `StockTopTenShareholder` | 18 |
| `StockTopTenFloatShareholder` | 16 |
| `ETFDailyBar` | 16 |
| `ETFList` | 6 |
| `IndexDailyBar` | 11 |
| `IndexList` | 6 |
| `IndexConstituent` | 5 |

**绝对禁止：**

- 不得因为某个算子“需要”就创造 `bid/ask/order_book/tick/trade_direction/news/analyst_consensus` 等字段。
- A 股当前 clean 数据**没有 Tick / Level2**。
- A 股当前 clean 数据**没有中文新闻/公告全文表**。
- `StockMinuteBar` 只有分钟 OHLCV + `Amount` + `Vwap`；不能把 bar proxy 写成真实订单流。
- 不得把 `UpdateTime` 当可交易信息时间。
- 不得把财报 `ReportPeriodEndDate` 当公告可知时间。
- 不得把分红 `ExDividendDate` 当严格公告 PIT 时间。
- 不得把财务 NaN 填 0。
- 不得假设北交所 `.BJ` 已在当前 A 股面板中可用。

## 1.3 候选真值：R41–R47 156 个 future-operator recommendations

R47 ZIP 累计保存 **156** 个候选算子工程建议。它们是“研究候选池”，不是“现在全部缺失”。

本文件后面完整保留 156 个候选及原始工程规格。执行时必须先做 current-main preflight。

---

# 2. A 股字段与时序硬约束

## 2.1 时间模型

| 数据类型 | 表 | 时间语义 | 正确对齐 |
|---|---|---|---|
| D1 | StockDailyBar / Valuation / Industry / ETF / Index | 交易日截面 | exact/equi |
| S1 | StockStatus / StockCapitalDaily / TopTen* 等 | 状态/快照 | 按契约 exact/state-ready；不得乱做 row shift |
| E1 财务 | Balance / Income / CashFlow / Indicator | `PubDate` 才是 knowledge time | PIT backward-asof |
| E1 effective | Dividend | `ExDividendDate` 是有效日，不是可靠公告日 | 仅 effective-time 场景 |
| MINUTE | StockMinuteBar | `QuoteTime` UTC | 严格 session/slot 顺序 |
| STATIC | Calendar | 日历维表 | 全读/显式 join |

## 2.2 单位

必须遵守 DataAccess SemanticFieldCatalog 的 canonical normalization：

- A 股 `Return` 原始单位是 **bp**，canonical decimal return = `Return / 10000`。
- `TurnoverRatio`、`DividendRatio`、ROE/ROA/利润率、`ShareRatio`、指数 `Weight` 原始多为 `%`，进入通用算子前按契约归一。
- `PeRatio/PbRatio/PsRatio/Pcf*` 是倍数，不除 100。
- `Factor` 为累积后复权乘数：**`adj_close_backward = Close * Factor`**。
- 分钟表没有 `Factor`；不要凭空生成分钟复权字段。
- `QuoteTime` 为 UTC；A 股本地时钟 = UTC+8。

**禁止双重缩放。**  
如果 DataAccess 已通过 SemanticFieldCatalog 把 `return_bp` 转成 decimal，FactorEngine 算子内部不得再次 `/10000`。

## 2.3 财务 PIT

Current DataAccess 已具备：

- `knowledge_time: PubDate`
- `period_time/effective_time: ReportPeriodEndDate`
- `join_policy: pit_asof_backward`
- `period_selection: latest_period`
- financial-event revision handling

因此：

1. **不要在 FactorEngine 内重新发明一套 report asof join。**
2. 旧候选 `report_asof` 默认应判定为 `ARCHITECTURE_SUPERSEDED`，除非 current DataAccess 仍存在无法满足的明确语义缺口。
3. 财务 period operator 必须基于真实 `ReportPeriodEndDate` / period identity，不得对 daily forward-filled 财务值做 `shift(63)` 伪装季度 lag。
4. 中国利润表/现金流表多为 YTD 累计口径；单季度必须用明确的 cumulative→single-quarter 逻辑。
5. 同一 `PubDate` 可能同时披露多个 `ReportPeriodEndDate`，必须先执行 period-selection policy。
6. 后续修订只能从修订可知时点之后影响结果，不能 retroactively rewrite 历史决策行。

---

# 3. Current-main 已观察到的高碰撞区：不得重复开发

下面不是完整清单，只是已经确认的典型冲突。代码 AI 必须自动完成全量 preflight。

## 3.1 财务 period / revision

current main 已有大量：

- `period_lag`
- `period_change`
- `quarter_from_cumulative`
- `fin_pct_change`
- `fin_qoq`
- `fin_yoy`
- `fin_ttm_cumulative`
- `fin_ttm_quarterly`
- `fin_revision_delta`
- `fin_revision_pct`
- `fin_revision_count`
- `fin_revision_direction`
- `fin_revision_magnitude`
- `fin_restated_flag`
- `fin_staleness`
- `fundamental_staleness`
- 多个 `fiscal_*` 质量、趋势、回归、方向一致性算子

因此 R41 的 `fiscal_lag/fiscal_delta/fiscal_pct_change/fiscal_acceleration/financial_snapshot_lag` 很可能已被 exact 或 semantic equivalent 覆盖。**不得因为名字不同重复实现。**

## 3.2 股东

current main 已有：

- concentration / concentration slope / acceleration
- holder entry / exit / churn
- ID-matched overlap / churn
- pledge / freeze ratio
- class / nature entropy
- rank stability / rank migration
- common-holding peer return
- shareholder network centrality

所以新的股东因子优先在 DSL 里组合，除非确实缺一个高复用 primitive。

## 3.3 指数

current main 已有：

- `index_member`
- `index_entry_exit_event`
- `index_membership_age`
- `index_weight`
- `index_weight_change`
- `index_reconstitution_churn`
- `index_weight_gap_to_free_float`

不要把“指数纳入冲击”“指数调仓压力”这种单一研究假设直接做成新 operator；优先由现有 primitive 组合成 factor。

## 3.4 A 股涨跌停 / 日内

current main 已有至少：

- `intra_limit_first_hit_time`
- `intra_limit_duration`
- `intra_limit_reopen_count`
- `ashare_limit_*` 大量日频状态/计数
- `intra_vwap_cross_count`
- `intra_vwap_above_ratio`
- `intra_time_above_vwap`
- `intra_vwap_path_slope`
- `intra_vwap_path_curvature`
- `intra_vwap_reversion_speed`
- realized variance / semivariance / quarticity / jump / drawdown / path-efficiency 等大量日内算子

所以新增应集中在**当前组合能力难表达的事件对齐、路径响应、状态转移和复杂结构 primitive**。

---

# 4. Preflight：开发前必须生成 gap classification

对本文件后面的 156 个候选 + 本文件新增补充候选逐一分类。

输出：

`factor_engine/evidence/operator_gap_preflight_<HEAD>.csv`

至少包含：

| column | meaning |
|---|---|
| candidate | 候选名 |
| candidate_signature | 候选签名 |
| current_exact | 是否 exact canonical 已存在 |
| current_alias | 是否只是 alias |
| semantic_equivalent | 是否有不同名但相同语义 |
| composable | 是否可由现有 daily ops 无损组合 |
| current_surface | daily/research/extended/... |
| pandas | current backend status |
| polars | current backend status |
| duckdb_sql | current backend status |
| field_legal | 是否只依赖现有字段 |
| pit_legal | 是否满足 PIT |
| disposition | 最终处置 |
| replacement | existing canonical / composition |
| reason | 详细说明 |

`disposition` 只能取：

- `EXISTING_EXACT`
- `EXISTING_ALIAS`
- `EXISTING_EQUIVALENT`
- `COMPOSABLE_NO_NEW_OPERATOR`
- `TRUE_GAP_IMPLEMENT`
- `RESEARCH_ONLY_IMPLEMENT`
- `BLOCKED_BY_DATA_CONTRACT`
- `REJECT_FIELD_MISMATCH`
- `REJECT_LOOKAHEAD`
- `ARCHITECTURE_SUPERSEDED`

**只有 `TRUE_GAP_IMPLEMENT` 和经明确批准的 `RESEARCH_ONLY_IMPLEMENT` 可以写新 runtime。**

---

# 5. 新算子的设计原则

## 5.1 优先“搜索空间 primitive”，不是“一个研报一个 monolithic operator”

FactorEngine 的价值是扩大 Alpha 搜索空间。

优先：

- reusable reducer
- event alignment
- state transition
- ex-self aggregation
- period-aware transformation
- causal rolling statistic
- intraday path primitive

谨慎：

- 一个 operator 内塞完整交易策略
- 一个 operator 内塞研究论文所有步骤
- 用一个复杂黑盒直接输出 Alpha
- 只能服务一条 factor formula 的特殊函数

如果一个候选可以拆成 2–4 个通用 primitive，并让 AlphaProbe/LLM 重新组合，**优先拆 primitive**。

## 5.2 输出契约

新 daily operator 默认必须：

- 每股票、每交易日一个标量；
- shape preserving / 明确 daily aggregation；
- 不返回 list/object/model artifact；
- 不输出未来标签；
- 数值为 float/bool/category 但 factor 根节点原则上应可转连续强度；
- 缺失语义明确；
- `min_periods` 明确；
- lookback 明确；
- lag 明确；
- surface 明确；
- PIT 明确；
- deterministic。

纯 0/1 / {{-1,0,1}} 状态算子可存在，但主要用于 gate/condition/weight，不应为了“多一个 Alpha”把状态本身重复登记成大量 factor。

---

# 6. 三后端实现策略：不要追求“形式上的全覆盖”

FactorEngine 当前主后端按本任务统一理解为：

1. **Pandas / pandas_numpy**：语义 reference。
2. **Polars**：生产加速，必须 expression-native 或正式受控 kernel。
3. **DuckDB SQL**：适合可 SQL 化的长表/窗口/截面/分组子树。

## 6.1 总原则

### Pandas：所有新 operator 都必须有

Pandas 是 golden semantics。  
复杂 operator 也必须先写一个清晰、可验证的 Pandas/NumPy reference。

但：

- 不允许股票×日期双重 Python nested loop 作为最终大面板实现；
- 可使用 NumPy/Numba；
- 输出 dtype 不得偷偷变 object；
- index/columns 必须保持一致；
- 不得排序后忘记恢复原 alignment。

### Polars：daily production 基本必须有

对于准备进入 `surface=daily` 的 operator，原则上必须实现真正的 Polars path。

**禁止：**

- Polars backend 内 `.to_pandas()`
- `.to_numpy()` 后调用同一个 Pandas kernel却声称“Polars 已实现”
- `rolling_map` 伪 native
- 为了过 coverage 门禁给一个性能更差的 wrapper

Current production hardening 会剥离这类假 Polars backend。

### DuckDB：只给适合 SQL 的 operator

强烈建议实现 DuckDB 的类型：

- elementwise
- safe math
- finite-window rolling
- conditional rolling
- event age/count
- rank/zscore/quantile
- group aggregate
- ex-self mean/weighted mean
- simple OLS/WLS
- period lag/change
- deterministic categorical gate

**不建议实现 DuckDB**：

- HMM/HSMM
- autoencoder
- topology / persistent homology
- wavelet scattering
- functional regression
- dynamic graph
- optimal transport heavy solver
- nonlinear recovery curve fitting
- dynamic programming segmentation
- complex model refit

这些强行 SQL 化通常只会带来 UDF/Python materialization，破坏 pushdown。

## 6.2 Backend 等级

| Tier | 类型 | Pandas | Polars | DuckDB |
|---|---|:---:|:---:|:---:|
| A | elementwise / rolling / group / period / simple event | MUST | MUST native | MUST if exact SQL semantics feasible |
| B | intraday session/path reducers / event response / chip profile | MUST | MUST | optional；通常不做 |
| C | panel graph / robust multivariate / low-rank | MUST | SHOULD | no，除非自然 SQL |
| D | HMM / AE / topology / wavelet / neural / functional model | MUST reference | preprocessing/controlled kernel only | NO |

## 6.3 SQL emitter ≠ production certified

当前 `sql_pushdown_coverage.md` 已登记 **354 个 emitter implementations**，但生成报告仍显示：

- DuckDB parity verified = 0
- DuckDB production safe = 0

因此本次新增/整改必须区分：

1. SQL 能编译；
2. SQL 能执行；
3. 与 Pandas golden parity；
4. NaN/Inf edge parity；
5. production-safe evidence。

不要只把 operator 加进 emitter mapping 就宣布三后端完成。

---

# 7. Pandas / Polars / DuckDB 常见坑与解决方案

## 7.1 NaN / NULL / Inf

### 问题
Pandas/NumPy 的 `NaN`、Polars `null/NaN`、DuckDB `NULL/NaN` 行为不同。

### 规则
- 每个 operator 明确“finite-only / null-propagating / skipna / pairwise”。
- 不得用 `fillna(0)` 解决 parity。
- `Inf` 与 `NaN` 分开测。
- pair statistic 只使用双方同时 finite 的 observation。
- `min_periods` 计 **有效样本数**，不是窗口总行数。

## 7.2 rolling 边界

禁止：

- centered rolling
- negative shift
- `bfill`
- 用未来窗口
- 删除 NaN 后再 positional shift，造成时间压缩

DuckDB 窗口必须使用显式：

`PARTITION BY instrument ORDER BY time ROWS BETWEEN ... PRECEDING AND CURRENT ROW`

并单独处理有效样本计数。

## 7.3 标准差

默认 `ts_std` 语义是 sample std，`ddof=1`。  
所有需要 std 的新增算子必须显式写 `ddof`，不能让 backend 默认值决定结果。

## 7.4 rank / quantile ties

必须固定：

- tie method
- percentile denominator
- all-equal cross-section behavior
- single valid name behavior
- NaN ranking behavior

## 7.5 ex-self

市场/行业/指数/peer 聚合若语义要求 ex-self：

- 必须严格扣除自己；
- weighted aggregate 要同时扣除 self numerator 和 denominator；
- 缺失 self 不得改变其余人的 denominator 错误；
- 不得 O(N²) 物化全相关矩阵，能 total-minus-self 就用 O(NT)。

## 7.6 财务 forward-filled panel

最大的坑：

`daily_financial.shift(63)` **不是上一季度**。

必须：

- 使用真实 period key；
- distinct period sequence；
- PIT vintage；
- period operator；
- 再 broadcast 到 daily。

## 7.7 分钟缺口

A 股分钟：

- 正常日约 240 根；
- 09:31–11:30 / 13:01–15:00；
- 午休不是“缺失数据”；
- 缺失 minute 不能把后面的 bar positional 左移；
- 同时钟 lag 必须按 `(session_id, slot_id)` exact match；
- 跨午休不能插值；
- suspension/partial day 默认 fail closed / NaN。

## 7.8 EOD 可交易时点

使用完整 t 日分钟路径的 daily factor：

**earliest tradable = t+1 trading session**。

不得在 t 日收盘前的回测里假装整天分钟数据已经知道。

若以后支持 prefix intraday decision，必须单独创建 prefix contract，不得复用 full-session operator。

## 7.9 stateful/model operators

- fit window 必须截止 `t-1` 或显式 `fit_lag>=1`；
- model artifact 要 deterministic；
- random seed 固定；
- label 使用必须显式 horizon/purge；
- 不能用未来 return 训练后给历史打分；
- chunk/shard 重跑结果必须与 full-history 一致；
- label permutation / future poison 测试必须通过。

---

# 8. current-main 代码接入要求

新增 operator 不只是写一个函数。

至少检查/修改：

1. `cleaned_operators/...` 正式实现；
2. `@register_operator(...)` canonical；
3. `_aliases.py`（仅确有兼容需要）；
4. operator surface classification；
5. PIT / lookback / lag / min_periods metadata；
6. API / DSL callshape；
7. Polars native adapter；
8. SQL emitter（若 Tier A 且可行）；
9. operator docs semantics；
10. generated operator catalog；
11. generated DSL allowlist；
12. production-hardening evidence；
13. parity tests；
14. causality/future-poison tests；
15. package-wide tests。

**严禁只写 stub。**  
`*_stub` 未注册不算实现；catalog 有名字但 runtime 不可执行也不算实现。

---

# 9. DataAccess 与 FactorEngine 的边界

## 9.1 DataAccess 应负责

- field resolution；
- physical dataset selection；
- exact / PIT-asof join；
- unit normalization；
- schema / dtype / grain；
- snapshot；
- calendar；
- multi-dataset join；
- revision-aware field materialization。

## 9.2 FactorEngine operator 应负责

- 已正确对齐输入上的数学/统计变换；
- rolling / cross-section / group / state / event path；
- daily scalar output；
- factor DAG expression。

不要让 FactorEngine operator 自己去拼 parquet 路径或重新实现 DataAccess join。

## 9.3 `fin_schema_gate` 特别处理

R47 建议的：

`fin_schema_gate(schema_name, x, statement_schema=None, strict=True)`

语义本身合理，但用户上传的 A 股字段字典中**没有** `financial_statement_schema` 字段；本文件检查到的 current DataAccess `semantic_fields.yaml` 也未观察到同名 semantic field。

所以：

- **不得创造这个字段。**
- 不得用“某些银行字段非空”反推 schema。
- 不得简单用 `IndustryName` 冒充 audited statement schema。
- 可以先实现 runtime API，但必须保持 `BLOCKED_BY_DATA_CONTRACT` / 非 daily admission。
- 只有当 DataAccess 已存在可验证、effective-dated 的真实 statement-schema metadata 时才允许解锁。
- 如果没有真实 metadata，R47 的 32 个 schema-gated candidates 继续保持 gated。

---

# 10. 实现优先级重排

旧 R41–R47 priority 是研究阶段的 priority；current main 已变化，所以执行时要重新排序。

## Wave 0 — 必做：current-main preflight

先完成 156 个候选的 exact/equivalent/composable/gap classification。

没有 Wave 0 结果，不得批量新增。

## Wave 1 — P0：高复用 primitive / A 股核心结构

若 preflight 判定为 TRUE_GAP，优先实现：

### 10.1 通用分钟/event primitive
- `intra_slice_mask_reduce`
- `intra_slice_mask_pair_reduce`
- `intra_multiresolution_resample_reduce`
- `intra_same_slot_zscore`
- `same_clock_lag`
- `intra_session_segment_reduce`
- `intra_session_boundary_jump`

这些 primitive 一旦实现，应让更多“早盘/午后/尾盘/异常分钟/状态条件”因子通过 DSL 组合完成，而不是继续新增几十个硬编码 segment operator。

### 10.2 冲击—试盘—吸收—整理
- `intra_impulse_event_detector`
- `intra_post_impulse_response`
- `intra_probe_outcome_score`
- `intra_supply_absorption_score`
- `intra_consolidation_quality`
- `intra_response_curve_features`
- `intra_liquidity_resilience_curve_fit`

要求事件定义、overlap/refractory、尾盘 censoring、next-session availability 全部固定。

### 10.3 volume-at-price / chip
- `intra_volume_at_price_profile`
- `intra_volume_profile_peak_geometry`
- `intra_volume_profile_supply_structure`
- `intra_volume_profile_value_area`
- `turnover_chip_distribution`
- `turnover_chip_distribution_transport`
- `turnover_chip_age_cost_surface`
- `turnover_chip_overhang_surface`

注意：只能使用分钟 OHLCV/Vwap/Amount 和日频 turnover/capital 等现有字段。  
**不能伪装成真实订单簿筹码。** 名称/文档要明确它是 bar-derived estimated cost/volume distribution。

### 10.4 A 股特殊状态
- `intra_limit_pre_hit_pressure_profile`
- `suspension_restart_response`
- `trading_calendar_mask`

但 current main 已有大量 limit/suspension primitives，必须先判断是否能组合，不重复造综合指标。

### 10.5 历史 signal reliability
- `ts_lagged_predictability_score`

这是 meta-signal，重点是 horizon alignment 与 label leakage。  
只有使用**已实现且已经实现收益的历史时点**才允许进入 t 的 score。

## Wave 2 — P1：panel / cross-domain /复杂但仍生产可控

按 preflight 选择：

- `panel_ewm_beta_ex_self`
- `panel_async_beta_ex_self`
- `panel_cmra_ex_self`
- `panel_day_night_beta_gap`
- `panel_apm_residual_tstat`
- `intra_comovement_curve_ex_self`
- `panel_intraday_low_rank_residual_ex_self`
- `panel_similarity_crowding_score`
- `panel_cluster_risk_score`
- `panel_peer_graph_aggregate`
- `fundamental_cash_flow_duration`
- `fiscal_cost_stickiness_score`
- `fundamental_latent_balance_sheet_factor`
- `fundamental_working_capital_financing_state`
- `fundamental_cost_stickiness_panel`

其中 panel graph / low-rank 需要避免 N² 大矩阵常驻内存，必要时 sparse/top-k/block。

## Wave 3 — research-first

以下不应为了“daily operator 数量”直接进入 public daily DSL：

- `cs_predictability_mosaic_score`
- `intra_functional_autoencoder_score`
- `intra_hmm_posterior_entropy`
- `intra_function_on_function_anomaly_response`
- persistent homology / topology
- wavelet scattering / EMD-Hilbert
- neural CDE / contrastive embedding
- HMM/HSMM
- dynamic graph
- diffusion map
- functional regression
- heavy optimal transport
- bicoherence / nonlinear spectral model
- model-refit / learned embedding family

默认：

- `surface=research` 或 `extended`
- Pandas/NumPy golden
- Polars 仅 session packing / preprocessing
- DuckDB 不做
- 有 deterministic artifact、causality、chunk parity、resource budget 后再申请 daily promotion

---

# 11. 本文件新增的补充 operator 建议

下面不是为了凑数量，而是从“让 LLM 因子挖掘能组合更多逻辑”的角度补充的 generic primitives。  
**执行时同样先做 current-main exact/equivalent preflight。**

## 11.1 `intra_event_window_reduce` — P0

**Signature**

`intra_event_window_reduce(x, event_mask, pre=5, post=15, reducer="mean", event_select="first", overlap="refractory", min_obs=None)`

**Purpose**

通用分钟事件对齐窗口 reducer。避免每一种“冲击后 10 分钟均值/波动/成交量”都新增专用 operator。

**Semantics**

对当日 `event_mask=True` 的事件分钟，构造 `[event-pre, event+post]` 的固定 slot window；按 event selection policy 选择事件，再输出 daily scalar。

`reducer` 建议支持：

- mean
- sum
- std
- max
- min
- last
- slope
- signed_area

**Null / censor**

- 缺 pre/post window → 按 `censor=drop|null` 显式处理；
- lunch gap 不跨越；
- end-of-day truncated event 默认 drop；
- overlap 由 refractory slots 去重；
- 无事件 → NaN，不是 0。

**Backend**

- Pandas MUST
- Polars MUST
- DuckDB OPTIONAL；如果 long-table window SQL 很自然可以做，否则不要硬做

**Tests**

first/last/strongest event、overlap、午休、尾盘 censor、缺分钟、多个股票、future poison、Pandas/Polars parity。

---

## 11.2 `intra_event_pre_post_contrast` — P0

**Signature**

`intra_event_pre_post_contrast(x, event_mask, pre=10, post=10, metric="mean_diff", event_select="first", min_obs=5)`

**Outputs/metric**

- mean_diff
- median_diff
- vol_ratio
- slope_diff
- range_ratio
- activity_ratio

用于通用“事件前后响应”，让 probe/absorption/reversal 等 factor 在 DSL 中组合。

**Backend**

Pandas + Polars MUST；DuckDB optional。

---

## 11.3 `ts_event_decay_kernel` — P1

**Signature**

`ts_event_decay_kernel(event, half_life=5.0, window=60, amplitude=None, normalize=False)`

**Semantics**

只使用过去事件的指数/其他固定 kernel memory：

`score_t = Σ_{{s<=t}} amplitude_s * exp(-(t-s)/half_life)`

没有 future event。NaN event 不当 0，需明确 valid-event semantics。

**Backend**

Pandas / Polars / DuckDB 均建议实现。  
这是典型 Tier-A operator。

---

## 11.4 `intra_state_transition_entropy` — P1

**Signature**

`intra_state_transition_entropy(state, min_slots=30, normalize=True, include_self=True)`

输入为已有条件/gate 生成的离散 state，输出当日状态转移 entropy。

**注意**

- state missing 会 break transition，不允许跨缺口连接；
- lunch boundary 默认 break；
- 只有同一连续 session segment 内相邻 slot 才形成 transition；
- categorical code 顺序不得影响结果。

**Backend**

Pandas + Polars MUST；DuckDB 可选。

---

## 11.5 `intra_state_dwell_stats` — P1

**Signature**

`intra_state_dwell_stats(state, target_state=None, output="mean|max|cv|last|share", min_slots=20)`

通用状态驻留长度/持续性 primitive。

可与价格/量状态条件组合，而不需要为每种状态新增一个 streak operator。

**Backend**

Pandas + Polars MUST；DuckDB optional。

---

## 11.6 `cs_robust_mahalanobis_score` — P1/P2

**Signature**

`cs_robust_mahalanobis_score(*features, shrinkage=0.1, center="median", scale="mad", min_names=50)`

每日横截面多维异常强度，无 return label。

**Rules**

- robust center/scale 只使用当日横截面；
- covariance 必须 shrinkage/regularized；
- listwise missing 默认；
- singular matrix 返回 NaN 或稳定 shrinkage 结果；
- deterministic；
- 不做 full-sample fit。

**Backend**

Pandas MUST；Polars preprocessing + deterministic numeric kernel SHOULD；DuckDB 不建议。

**Preflight**

如果 current main 已有等价 robust multivariate anomaly operator，则不新增。

---

## 11.7 `intra_piecewise_linear_path_features` — P2

**Signature**

`intra_piecewise_linear_path_features(path, max_segments=4, penalty="bic", output="break_count|slope_shift|last_slope|fit_gain")`

对完整日内 path 做**确定性、无监督、同日完成后的**分段线性结构提取。

- 不使用未来天；
- full-session output 只能用于下一交易日；
- 分段点不得跨午休；
- DP/PELT 等实现必须 deterministic。

Pandas + Polars/controlled kernel；不做 DuckDB。

---

## 11.8 `ts_online_change_point_score` — P2 / research-first

**Signature**

`ts_online_change_point_score(x, method="cusum|bocpd", window=252, hazard=0.01, min_periods=60)`

只使用到 t 为止的数据输出 regime-break intensity。

- CUSUM 可以 daily；
- BOCPD 默认 research，除非 state checkpoint、chunk parity、determinism 完成；
- 不得用全样本均值/方差初始化造成 lookahead。

---

# 12. 明确“不建议新增”的 operator 类型

下面即使“能写”，也优先留在 factor expression，不做新 operator：

1. `fundamental_disclosure_latency = PubDate - ReportPeriodEndDate`  
   → date diff/已有 staleness primitive 可表达。

2. `index_rebalance_pressure = index_weight_change * turnover/liquidity`  
   → 组合因子，不是 primitive。

3. `holder_persistence_weighted_concentration`  
   → current holder ops 可组合。

4. `valuation_quality_combo`、`growth_value_combo`、`fundamental_market_confirmation`  
   → 都应由 DSL 组合。

5. 只改 window/threshold 的算子复制品。  
   例如 `ts_mean_5/ts_mean_10` 绝对禁止。

6. 只把已有 operator 改名字的“新算子”。

7. 需要不存在的 Level2/orderbook/news 字段的 operator。

---

# 13. Operator 参数与 callshape 规范

每个新 canonical 必须定义：

- positional vs keyword args；
- 参数类型；
- 合法区间；
- default；
- 是否允许动态 expression 参数；
- output dtype；
- lookback；
- min_periods；
- lag；
- scope；
- surface；
- PIT safe；
- backend support。

参数非法时 fail fast，不要 silent clamp。

例如：

- `window >= 1`
- `min_periods >= 1 and <= window`（若语义如此）
- `horizon >= 1`
- `states >= 2`
- `half_life > 0`
- `ddof >= 0`
- `quantile ∈ [0,1]`

string enum 必须 whitelist，不要任意执行字符串。

---

# 14. 复杂 operator 的性能要求

## 14.1 禁止 N² 常驻矩阵

市场/行业/peer operator：

- 能 aggregate-minus-self 就 O(NT)；
- correlation graph 用 top-k/sparse；
- rolling covariance 分块；
- 不要每个日期创建完整 5000×5000 dense matrix 再丢掉。

## 14.2 分钟算子

A 股单日约 5000 股票 × 240 bars，属于百万级行。

必须：

- column pruning；
- date pruning；
- Polars lazy；
- session packing 批量化；
- 不得 per-stock 反复扫描全日 long table；
- event detection 尽量 vectorized；
- reuse shared intermediate（returns、activity zscore、slot id 等）；
- CSE 能共享就不要每个 operator 重算。

## 14.3 stateful model

必须受 ResourceBroker / memory budget / thread budget 管理。  
不要模型内部自己开无限线程。

---

# 15. 测试标准

每个 TRUE_GAP operator 至少包含以下测试中的适用项。

## 15.1 Golden case

手工可算小样本，验证公式。

## 15.2 Future poison

把 t+1 以后全部输入改成极端值：

`output[:t]` 必须逐位不变。

## 15.3 Current-target poison

如果 operator 使用 historical realized return：

修改未实现的未来 target 不得改变当前 score。

## 15.4 NaN gap

在窗口中插入 NaN，确保：

- 不时间压缩；
- min_periods 正确；
- 不 bfill；
- 不跨 gap 连接状态。

## 15.5 Inf

`+inf/-inf` 单独测试。

## 15.6 Alignment

乱序输入后恢复排序，结果按 key 不变。

## 15.7 Duplicate key

相同 `(time,instrument)` 重复行必须 fail 或按正式 contract 去重，不能 silent nondeterministic。

## 15.8 Cross-backend

Pandas vs Polars / DuckDB 按 operator risk 设置明确 `rtol/atol`。

## 15.9 Chunk / shard parity

full history 与：

- date chunks
- instrument shards
- warmup + main window

结果一致。

## 15.10 Ex-self leakage

复制一个超大股票或修改目标股票值，验证 peer aggregate 确实排除 self。

## 15.11 Session

- 09:31 first slot
- 11:30
- lunch
- 13:01
- 15:00
- missing minute
- suspension
- truncated day

## 15.12 PIT revision

财务：

- first release
- later revision
- two report periods same PubDate
- revision after decision date

历史结果不得被 later revision 回写。

## 15.13 Determinism

random/model operator 连续运行两次 bit/tolerance stable。

---

# 16. 三后端验收矩阵

每个本轮实现算子最终生成：

`factor_engine/evidence/new_operator_backend_matrix_<HEAD>.csv`

字段：

| operator | pandas | polars_native | polars_long | duckdb_emitter | duckdb_exec | parity | surface | prod_safe | reason |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|

**不要把“不适合 SQL”标成失败。**  
应写 `NOT_APPLICABLE_COMPLEX_KERNEL`。

但是 Tier A 若 DuckDB 明显可实现却不做，必须写原因。

---

# 17. Surface promotion 规则

## daily

只有满足：

- causal scalar daily output；
- 字段合法；
- PIT 合法；
- runtime + backend evidence；
- deterministic；
- no-lookahead；
- production hardening；

才进入 daily。

## research

- model fitting
- matrix/decomposition
- topology
- signal processing
- heavy graph
- learned embedding
- global diagnostics

默认 research。

**不能为了 AlphaProbe 能调用，就偷偷把 research 加进 daily allowlist。**

如果 AlphaProbe 需要研究 operator，应扩展明确的 research experiment path，不破坏 production DSL。

---

# 18. 文档与生成物

本次代码修改完成后必须更新/生成：

- `docs/operators_catalog.md`
- `docs/dsl_allowlist.json`
- `docs/dsl_operators_reference.md`
- semantics docs
- backend evidence
- gap preflight CSV
- backend matrix CSV
- test report
- changed canonical list
- alias mapping
- blocked list

并核验：

- no duplicate canonical；
- no aliases shadowing canonical；
- no unclassified operator；
- no stub pretending implementation；
- no research operator leaked to daily；
- no unsupported field；
- no negative shift / bfill / future window。

---

# 19. 最终交付格式

代码 AI 完成后，不要只说“已完成”。

必须报告：

## A. Baseline
- final main/branch HEAD
- preflight candidate count
- exact existing
- equivalent/composable
- true gaps
- blocked/rejected

## B. Implemented
逐 operator：

- canonical
- signature
- category
- surface
- Pandas
- Polars
- DuckDB
- PIT
- tests

## C. Skipped
逐 operator：

- existing replacement 或 composition
- 为什么不新增

## D. Blocked
- 缺哪个**真实**数据契约
- 为什么不能凭空补

## E. Tests
- test command
- passed/failed/skipped
- parity
- causality
- chunk/shard

## F. Remaining risk
只列真实剩余风险，不列泛泛“建议未来优化”。

---

# 20. R41–R47 156 个候选算子总清单

**注意：这是 preflight 输入，不是“156 个全部再实现一遍”的指令。**

{inventory_md}

---

# 21. 原始 R41–R47 工程规格全文

下面保留原始工程规格，防止实现 AI 丢失候选的签名、null policy、PIT、backend 和测试细节。

总计：**156 个候选 operator specs**。



---

## 附录来源：`r41_operator_engineering_specs.md`

> 以下是 R41–R47 原始工程规格，作为候选语义档案保留。**若与本文件前文的 current-main 规则、字段硬约束、backend policy 或 preflight 结果冲突，以前文为准。**

# R41 FactorEngine Operator Engineering Specifications

本文件用于直接交给实现 FactorEngine 的 AI/工程师。所有算子都必须保持 Pandas / Polars / DuckDB 三后端语义一致；时间相关算子默认按 `stock_id, decision_time` 排序，并遵守 no-lookahead。

## 总体实现要求

- **PIT**：任何财务/事件/图关系的估计只能使用决策时点已经可用的数据。
- **确定性**：随机算法（如 Isolation Forest）必须通过 date+seed 固定随机性。
- **缺失值**：不得静默填 0；每个算子按下述 null policy。
- **跨后端**：新增算子必须增加 golden-case 单测与随机 property test；浮点误差容忍度写死在测试中。
- **性能**：panel/graph 算子不得 O(N²) 物化完整矩阵，优先 aggregate-minus-self、sparse top-k、分块计算。

## fiscal_acceleration
- **建议签名**：fiscal_acceleration(x, basis="quarter", mode="growth_delta|level_second_diff", periods=1)
- **定义/算法**：Quality/earnings acceleration. growth_delta = fiscal_pct_change(x)_t - fiscal_pct_change(x)_{t-1}; level_second_diff = Δx_t-Δx_{t-1}.
- **参数**：basis=quarter|annual; mode=growth_delta|level_second_diff; periods>=1; percentage-change denominator policy inherited from fiscal_pct_change.
- **Null policy**：Null until 3 required snapshots exist.
- **排序/对齐**：Evaluate on PIT fiscal snapshot sequence. growth_delta compares two consecutive PIT growth observations; level_second_diff compares consecutive fiscal deltas.
- **三后端实现要求**：Compose fiscal_pct_change/fiscal_delta/fiscal_lag from one canonical snapshot engine; no independent daily-row shift implementation.
- **PIT/no-lookahead**：Requires only fiscal snapshots available by decision_time. Acceleration becomes visible when the newest required report vintage becomes available.
- **最低测试集**：At least 4 hand-calculated report snapshots; acceleration after sign changes; zero/negative base; missing quarter; revision published later; backend parity.
- **可解锁方向**：Quality acceleration, earnings acceleration, margin acceleration, cash profitability acceleration.
- **已预生成待解锁因子**：5 个
- **来源**：QUALITY_ACCEL_2024|PKU_CHINA_ANOMALIES_469
- **优先级**：P0
- **字段门槛**：NONE

## fiscal_delta
- **建议签名**：fiscal_delta(x, periods=1, basis="quarter|annual")
- **定义/算法**：Point-in-time difference x_t - fiscal_lag(x,N).
- **参数**：periods>=1; basis=quarter|annual. Endpoint selection is by distinct fiscal snapshots, not trading-row offsets.
- **Null policy**：Null if either endpoint missing.
- **排序/对齐**：Use fiscal_lag(x,periods,basis) on PIT-deduplicated snapshots; compute x_t-x_lag and broadcast the result forward only after x_t is available.
- **三后端实现要求**：Prefer expression fusion on top of fiscal_lag so all backends share the same snapshot selection; float64 result and identical null propagation.
- **PIT/no-lookahead**：Both current and lagged fiscal endpoints must have availability_time<=decision_time; later revisions never rewrite older decision rows.
- **最低测试集**：Quarterly/annual sequences; irregular/missing report period; revision vintage; current or lag null; zero/negative values; cross-backend bit/tolerance parity.
- **可解锁方向**：Accrual/investment/growth, cash-holding change, debt/equity issuance change.
- **已预生成待解锁因子**：1 个
- **来源**：PKU_CHINA_ANOMALIES_469|CASH_CHANGE|GLOBAL_Q_199_2025
- **优先级**：P0
- **字段门槛**：NONE

## fiscal_lag
- **建议签名**：fiscal_lag(x, periods=1, basis="quarter|annual", report_key=None)
- **定义/算法**：Lag by distinct fiscal observations, not trading rows. For each as-of date use only latest report known by that date, then step backward N fiscal observations.
- **参数**：periods>=1; basis; optional report_key
- **Null policy**：Null until N prior fiscal observations exist; preserve null if prior observation null.
- **排序/对齐**：Per stock ordered by report period + availability timestamp; daily rows receive last-known lagged report value.
- **三后端实现要求**：Vectorized Polars group/state implementation; DuckDB window over deduplicated report snapshots; Pandas semantic parity.
- **PIT/no-lookahead**：Never access report whose availability timestamp is after decision time. Revisions become new vintages; no retroactive overwrite.
- **最低测试集**：Synthetic irregular reports; restatement vintage; missing quarter; annual/quarter boundary; cross-backend equality.
- **可解锁方向**：Asset growth, Beneish components, cash change, inventory/receivables growth, quality growth, R&D capital inputs.
- **已预生成待解锁因子**：0 个
- **来源**：PKU_CHINA_ANOMALIES_469|GLOBAL_Q_199_2025|ACCOUNTING_QUALITY_2026
- **优先级**：P0
- **字段门槛**：NONE

## fiscal_pct_change
- **建议签名**：fiscal_pct_change(x, periods=1, basis="quarter|annual", eps=1e-12, zero_policy="null")
- **定义/算法**：PIT-safe fiscal percentage change (x_t-x_lag)/abs(x_lag), with explicit denominator policy.
- **参数**：periods; basis; eps; zero_policy=null|signed_eps
- **Null policy**：Null if endpoints missing or denominator invalid under zero_policy.
- **排序/对齐**：Fiscal snapshot ordering.
- **三后端实现要求**：Reuse fiscal_lag; exact backend parity on zero/negative denominator handling.
- **PIT/no-lookahead**：No future fiscal observation.
- **最低测试集**：Zero/negative denominator; missing; restatement; parity.
- **可解锁方向**：Asset/sales/inventory/debt/R&D growth and China anomaly definitions.
- **已预生成待解锁因子**：24 个
- **来源**：PKU_CHINA_ANOMALIES_469|GLOBAL_Q_199_2025
- **优先级**：P0
- **字段门槛**：NONE

## intra_market_profile_corr_ex_self
- **建议签名**：intra_market_profile_corr_ex_self(minute_x, transform="log1p", weighting="equal", min_names=50)
- **定义/算法**：Correlation of a stock intraday activity profile with contemporaneous ex-self market profile, producing daily stock value.
- **参数**：transform=identity|log1p|zscore; weighting=equal|market_cap; min_names>=2; min_slots; slot_standardization=none|cross_sectional_z; exclude_self=true.
- **Null policy**：At a slot, omit names with null minute_x; null day output if valid slots<min_slots or market profile names<min_names for too many slots. No zero fill for suspended/missing bars.
- **排序/对齐**：For each date/slot compute market aggregate excluding the target stock using total-minus-self algebra; form target and ex-self market profile vectors over ordered slots; compute Pearson/Spearman correlation as configured.
- **三后端实现要求**：O(N*T) aggregate-minus-self, never N² pair matrix. Market-cap weights must subtract self numerator/denominator exactly. Shared correlation min-periods and zero-variance policy.
- **PIT/no-lookahead**：Full-day correlation is next-session feature by default. If prefix_mode is later implemented, only slots <= decision minute may enter.
- **最低测试集**：Single/too-few names; dominant large-cap self exclusion; missing slots; zero-variance profile; log1p zeros; equal/cap weighting; backend parity.
- **可解锁方向**：方正“水中行舟” amount market-following/commonality signals.
- **已预生成待解锁因子**：2 个
- **来源**：FOUNDER_MARKET_FOLLOW_2023
- **优先级**：P0
- **字段门槛**：NONE

## report_asof
- **建议签名**：report_asof(value, availability_time, report_period=None, revision_id=None, decision_time=None, policy="latest_known")
- **定义/算法**：Point-in-time as-of materialization. For each stock and decision_time, select the latest observation whose availability_time<=decision_time. If multiple vintages exist for the same report_period, keep only the newest revision that was already available by decision_time; never retroactively overwrite history.
- **参数**：availability_time; optional report_period/revision_id; decision_time defaults to row decision timestamp; policy=latest_known|first_release; optional max_age.
- **Null policy**：Return null before the first available observation; preserve a selected vintage null rather than silently backfilling from a later revision. max_age breach -> null.
- **排序/对齐**：Partition by stock; sort events by availability_time then revision_id; perform backward as-of join to decision_time. Repeated daily rows carry only the last-known vintage.
- **三后端实现要求**：Polars: join_asof on stock + availability_time with explicit sortedness; DuckDB: ASOF JOIN or lateral max(availability_time) constrained <= decision_time; Pandas: merge_asof. Golden fixture must be identical across backends.
- **PIT/no-lookahead**：Hard condition availability_time<=decision_time. A restatement published later may affect only decision rows on/after its own availability_time.
- **最低测试集**：Initial report, missing report, two revisions of one period, same-timestamp tie, out-of-order input, max_age, first_release vs latest_known, and Pandas/Polars/DuckDB equality.
- **可解锁方向**：Strict PIT materialization for financial/event factors when announcement/revision timestamps become available.
- **已预生成待解锁因子**：0 个
- **来源**：OSAP_UNIVERSE|JKP_GLOBAL
- **优先级**：P0
- **字段门槛**：REQUIRES_AVAILABILITY_TIMESTAMP_METADATA

## same_calendar_month_return
- **建议签名**：same_calendar_month_return(ret, years_back=(1,), aggregate="mean", exclude_recent_months=0)
- **定义/算法**：Return in the same calendar month in prior years, with exact calendar alignment rather than fixed 252-day approximation.
- **参数**：years_back tuple; aggregate; exclude_recent_months
- **Null policy**：Null if no matched historical month.
- **排序/对齐**：Map trading dates to calendar month; aggregate monthly return then join prior-year same month.
- **三后端实现要求**：Calendar-aware Polars/DuckDB implementation.
- **PIT/no-lookahead**：Uses only prior completed matched months.
- **最低测试集**：Leap years; suspension; month-end missing; parity.
- **可解锁方向**：A-share/global seasonality signals t-12, t-24/36/48/60 averages.
- **已预生成待解锁因子**：3 个
- **来源**：PKU_CHINA_ANOMALIES_469
- **优先级**：P0
- **字段门槛**：NONE

## same_clock_lag
- **建议签名**：same_clock_lag(x, sessions=1, slot_key="minute")
- **定义/算法**：For each intraday slot, return value from same slot N prior trading sessions.
- **参数**：sessions>=1; slot_key=minute|bar_index; missing_slot_policy=null|previous_session_exact_only; session_calendar required.
- **Null policy**：Null when prior session/slot unavailable; never nearest-forward fill.
- **排序/对齐**：Create canonical session index and intraday slot id. Join (stock,session_idx-sessions,slot_id) exactly; do not use raw row shift because missing bars would misalign clock time.
- **三后端实现要求**：Polars keyed self-join or group shift after densifying session-slot key; DuckDB self-join/window over explicit slot key; Pandas MultiIndex reference. Output dtype preserved.
- **PIT/no-lookahead**：Only same slot from strictly prior sessions. No nearest future slot and no same-session forward fill.
- **最低测试集**：Lunch break; missing minute; half-day; suspension; daylight/timezone normalization if foreign test data; sessions 1/5; exact slot mismatch; backend parity.
- **可解锁方向**：Exact Heston same-clock continuation/reversal features.
- **已预生成待解锁因子**：4 个
- **来源**：INTRADAY_CROSS_SECTION
- **优先级**：P0
- **字段门槛**：NONE

## cn_sma
- **建议签名**：cn_sma(x, n, m, init="first_valid", reset_on_gap=True)
- **定义/算法**：Chinese recursive SMA: y_t=(m*x_t+(n-m)*y_{t-1})/n with explicit initialization.
- **参数**：n integer >0; m with 0<m<=n; init=first_valid|zero; reset_on_gap bool.
- **Null policy**：Null before first valid input. With reset_on_gap=true any null breaks state and next valid initializes anew; false preserves state but emits null at missing row.
- **排序/对齐**：Per stock chronological scan; recurrence y_t=(m*x_t+(n-m)*y_{t-1})/n. Initialization must be explicit and identical across backends.
- **三后端实现要求**：One stateful recurrence reference. Polars group scan/plugin; DuckDB UDF/recursive CTE; Pandas loop. Float64 fixed update order.
- **PIT/no-lookahead**：Only x_t and y_{t-1}; historical recursion.
- **最低测试集**：Hand sequence; m=n; m=1; first-valid vs zero init; single/multiple gaps with reset modes; long sequence numerical stability; backend parity.
- **可解锁方向**：GTJA191 SMA formulas.
- **已预生成待解锁因子**：2 个
- **来源**：GTJA_191
- **优先级**：P1
- **字段门槛**：NONE

## cs_isolation_forest_score
- **建议签名**：cs_isolation_forest_score(*features, n_trees=100, subsample=256, seed=0, standardize="robust")
- **定义/算法**：Cross-sectional unsupervised Isolation Forest anomaly score computed independently each date from available firm features. No return labels.
- **参数**：n_trees>=1; subsample=min(requested,N); seed; standardize=robust|zscore|none; missing=listwise|median; min_names; contamination affects labels only, not raw score.
- **Null policy**：Default listwise: stocks with any missing feature -> null. median mode uses same-date feature median and records imputation mask; if names<min_names -> all null.
- **排序/对齐**：Independently fit one forest for each decision-date cross-section after same-date preprocessing. Deterministic sampling seed = hash(global_seed,date). Preserve security-id order only for output alignment, not sampling bias.
- **三后端实现要求**：Prefer a shared backend-neutral tree builder producing serialized split specs applied identically by all backends. Score = normalized expected path length; deterministic tie/sampling. Tolerance documented.
- **PIT/no-lookahead**：Fit uses only same-date PIT features; no future dates, no return labels, no whole-sample standardization.
- **最低测试集**：Tiny N; duplicates; one clear outlier; missing modes; feature scale invariance under robust standardization; seed reproducibility; tree serialization; backend score parity.
- **可解锁方向**：Exact multidimensional fundamental unusualness score.
- **已预生成待解锁因子**：3 个
- **来源**：ISOFOREST_FUND_2026
- **优先级**：P1
- **字段门槛**：NONE

## event_window_return_asof
- **建议签名**：event_window_return_asof(close, event_time, pre_sessions=0, post_sessions=1, benchmark=None, mode="compound", execution="next_bar")
- **定义/算法**：PIT-safe event return aligned to an explicit event timestamp. Map event_time to first tradable bar/session under exchange calendar and execution policy; compound returns over [event-pre_sessions, event+post_sessions]. Optional benchmark produces abnormal return using matched window.
- **参数**：pre_sessions>=0; post_sessions>=0; benchmark optional; mode=compound|sum_log; execution=next_bar|same_session_if_before_cutoff; cutoff_time; suspension_policy.
- **Null policy**：Null if event_time missing, window cannot be fully observed as of decision_time, or security is not tradable and suspension_policy cannot resolve. Never infer event time from report period.
- **排序/对齐**：Event table keyed by stock/event_time; exchange-calendar session mapping; output becomes available only after the requested event window has completed.
- **三后端实现要求**：Separate event alignment primitive from return aggregation. Use deterministic trading-calendar table in all backends; unit-test timestamp/session boundary behavior.
- **PIT/no-lookahead**：At decision_time, require event_time and every bar used in the window <= decision_time. post_sessions>0 necessarily delays factor availability until that post window is complete.
- **最低测试集**：Before/after-close event; weekend/holiday; suspension; event at close cutoff; pre/post windows; benchmark abnormal return; multiple events; backend parity.
- **可解锁方向**：PEAD/announcement jump/post-event drift only if explicit event timestamp field is added.
- **已预生成待解锁因子**：0 个
- **来源**：HT_PEAD_JUMP_2026
- **优先级**：P1
- **字段门槛**：REQUIRES_EVENT_TIMESTAMP_FIELD

## financial_snapshot_lag
- **建议签名**：financial_snapshot_lag(x, periods=1, basis="quarter|annual", availability_time=None, report_period=None)
- **定义/算法**：Compatibility wrapper for fiscal_lag: lag distinct PIT financial snapshots, not repeated trading-day rows. Deduplicate same report period/vintage and step backward by report observations.
- **参数**：periods>=1; basis; availability_time/report_period metadata. Prefer implementing as alias to fiscal_lag rather than a second independent algorithm.
- **Null policy**：Exactly inherit fiscal_lag: null until required prior snapshot exists; selected snapshot null remains null.
- **排序/对齐**：Partition stock; order distinct report_period snapshots by PIT availability; map result to decision-date rows by as-of join.
- **三后端实现要求**：Implement once in fiscal_lag and expose this name as tested alias. Alias output must be bit-identical across all backends.
- **PIT/no-lookahead**：Same fiscal_lag PIT contract; later revisions never back-propagate.
- **最低测试集**：Alias equality with fiscal_lag on quarterly/annual/revision/missing-period fixtures; backend parity.
- **可解锁方向**：Legacy formula compatibility for asset/debt/inventory/cash-flow changes.
- **已预生成待解锁因子**：0 个
- **来源**：ALPHA_ARCH_FACTOR_LIBRARY|OSAP_UNIVERSE|JKP_GLOBAL
- **优先级**：P1
- **字段门槛**：NONE

## fiscal_capital_stock
- **建议签名**：fiscal_capital_stock(flow, depreciation_rate, basis="annual", init="scaled_first")
- **定义/算法**：Perpetual-inventory capital stock K_t=(1-d)K_{t-1}+flow_t on fiscal snapshots, for R&D/organizational capital.
- **参数**：0<=depreciation_rate<1; basis=annual|quarter; init=scaled_first|first_flow|zero; scaled_first uses flow_0/(g+d) only when a supplied/estimated positive growth g is valid; allow_negative_flow=false by default.
- **Null policy**：Before valid initialization -> null. Missing flow defaults to null and freezes no state unless missing_policy=carry_decay is explicitly selected. Negative flow null unless allow_negative_flow=true.
- **排序/对齐**：Run recurrence only on distinct PIT fiscal snapshots in report-period order; output becomes available from each snapshot publication date and is then as-of carried to daily rows.
- **三后端实现要求**：Stateful float64 scan per stock on snapshot table. Same initialization/missing-policy code path across Pandas/Polars/DuckDB UDF; no daily-row recurrence.
- **PIT/no-lookahead**：K_t uses only K_{t-1} and flow_t known by the current report availability. Later revisions create a new vintage path only from that revision date forward.
- **最低测试集**：First-flow and scaled-first init; d=0/high d; missing flow; negative flow policy; irregular periods; revision; annual/quarterly; backend parity.
- **可解锁方向**：R&D capital-to-assets/market, intangible capital growth.
- **已预生成待解锁因子**：3 个
- **来源**：PKU_CHINA_ANOMALIES_469|GLOBAL_Q_199_2025
- **优先级**：P1
- **字段门槛**：NONE

## fiscal_rolling_regression
- **建议签名**：fiscal_rolling_regression(y, x1, x2=None, periods=8, basis="quarter", output="resid_std|beta|r2")
- **定义/算法**：Rolling regression over fiscal observations; primary use is Dechow-Dichev-style accrual quality against cash flows.
- **参数**：periods>=3; basis=quarter|annual; output=resid_std|beta|r2|intercept; min_periods; add_intercept=true; ridge_eps only when condition number exceeds threshold.
- **Null policy**：Listwise deletion; null if insufficient rank/observations.
- **排序/对齐**：Construct one row per PIT fiscal snapshot per stock. Regress trailing y on contemporaneous/explicitly-lagged x columns over the last periods snapshots; broadcast output after current report availability.
- **三后端实现要求**：Use one numerically specified least-squares routine: centered normal equations with ridge fallback or QR. Document ddof and residual variance denominator; backend outputs tolerance <=1e-8.
- **PIT/no-lookahead**：Every y/x observation in regression must be from a report vintage available by decision_time. Any lead variable is forbidden; lag variables must call fiscal_lag.
- **最低测试集**：Known linear relation; intercept/no-intercept; singular/near-singular design; missing rows/listwise deletion; insufficient rank; revision; backend parity.
- **可解锁方向**：Dechow-Dichev accrual quality, fiscal beta/quality stability.
- **已预生成待解锁因子**：1 个
- **来源**：ACCOUNTING_QUALITY_2026
- **优先级**：P1
- **字段门槛**：NONE

## fiscal_rolling_std
- **建议签名**：fiscal_rolling_std(x, periods=5, basis="annual", min_periods=3)
- **定义/算法**：Rolling variability across distinct fiscal observations.
- **参数**：periods>=2; basis=quarter|annual; min_periods between 2 and periods; ddof default=1; optional robust=false|mad.
- **Null policy**：Null until min_periods valid PIT fiscal observations exist. Do not treat repeated daily carry-forward values as separate observations.
- **排序/对齐**：Deduplicate to distinct report snapshots per stock, calculate trailing window ending at current snapshot, then as-of broadcast to daily decision rows.
- **三后端实现要求**：Polars rolling_std on snapshot table; DuckDB STDDEV_SAMP window ROWS periods-1 PRECEDING; Pandas rolling reference. For robust mode share median/MAD definition.
- **PIT/no-lookahead**：Window contains only report vintages available by current decision time; revisions enter from their publication date forward.
- **最低测试集**：Constant sequence; missing periods; ddof 0/1; min_periods; revisions; daily carry-forward not counted; annual/quarterly; backend parity.
- **可解锁方向**：Earnings variability, margin stability, cash-flow stability, Mohanram-style stability.
- **已预生成待解锁因子**：4 个
- **来源**：MSCI_QUALITY|MOHANRAM_GSCORE|PKU_CHINA_ANOMALIES_469
- **优先级**：P1
- **字段门槛**：NONE

## intraday_value_at_extreme_state
- **建议签名**：intraday_value_at_extreme_state(value, state, extreme="max|max_abs|min", transform_state="identity", tie="last", normalize_state=None)
- **定义/算法**：Within each stock-day, identify the minute index where a state series is extreme, then return/aggregate another value series at that exact index. Examples: return-at-max-volume, volatility-at-largest-price-drop, volume-at-largest-absolute-return. Supports deterministic tie handling.
- **参数**：extreme=max|min|max_abs; transform_state; tie=first|last; normalize_state optional intraday zscore/rank; optional neighborhood radius with mean/sum aggregator.
- **Null policy**：Ignore state-null slots when locating extreme; if no valid state -> null. If chosen value is null -> null unless neighborhood aggregation explicitly configured.
- **排序/对齐**：Exact minute slot alignment within stock/session. For ties choose first/last by chronological slot; never sort by value without retaining slot key.
- **三后端实现要求**：Polars arg_max/arg_min over per-day arrays or struct; DuckDB arg_max(value,state) with explicit tie key; Pandas idxmax/idxmin reference. Define same max_abs semantics and tie resolution.
- **PIT/no-lookahead**：Full-session extreme is a close-of-day feature and is available for the next session by default; intraday use before session end is prohibited unless a prefix-only mode is explicitly added.
- **最低测试集**：Ties; all-null; value-null at selected slot; max_abs; first/last; irregular/missing slots; neighborhood radius; backend parity.
- **可解锁方向**：Special-moment intraday hypotheses using existing minute OHLCV/amount without tick/order-count fields.
- **已预生成待解锁因子**：3 个
- **来源**：GS_INTRADAY_SPECIAL
- **优先级**：P1
- **字段门槛**：NONE

## panel_peer_graph_aggregate
- **建议签名**：panel_peer_graph_aggregate(signal, edge_signal, lookback=60, lag=1, top_k=10, metric="corr", weighting="softmax_abs", exclude_self=True)
- **定义/算法**：Build rolling directed/undirected stock peer graph from historical relation metric and aggregate lagged peer signals.
- **参数**：lookback>=10; lag>=1; top_k>=1; metric=corr|leadlag (transfer_entropy only after separate implementation); weighting=equal|softmax_abs|signed_abs; exclude_self=true; min_overlap.
- **Null policy**：Null if edge metric lacks min_overlap or fewer than min_neighbors valid peer signals. Missing peer signals are excluded and weights renormalized.
- **排序/对齐**：At t estimate edges using observations ending at t-lag. Rank neighbors deterministically by |edge| then security_id. Aggregate peer signal dated <=t-lag; never current t peer signal unless explicitly delayed before call.
- **三后端实现要求**：Blockwise/sparse top-k computation; do not materialize full dense N×N over long panels. Cache edges by date/lookback. Same correlation and tie rules across backends.
- **PIT/no-lookahead**：Both graph estimation sample and peer signal endpoint must be <=t-lag. lag=0 prohibited in production.
- **最低测试集**：Known 4-stock correlation graph; tie ranking; disconnected/new listing; universe entry/exit; negative edge weighting; missing peer; changing universe; backend parity.
- **可解锁方向**：Peer Index/PDI, cross-stock diffusion, network momentum.
- **已预生成待解锁因子**：10 个
- **来源**：PEER_EFFECTS_2026
- **优先级**：P1
- **字段门槛**：NONE

## price_delay_score
- **建议签名**：price_delay_score(ret, market_ret, window=252, lags=4, output="delay_ratio")
- **定义/算法**：Hou-Moskowitz style price-delay: incremental explanatory power of lagged market returns relative to contemporaneous-only regression.
- **参数**：window>=60; lags>=1; output=delay_ratio|r2_full|r2_now; market_ret_source=provided|cs_ex_self; min_periods; ridge_eps.
- **Null policy**：Null if regression lacks min_periods or variance/rank. Ex-self market return null if eligible cross-section too small.
- **排序/对齐**：For each stock fit rolling regression r_i,t on contemporaneous market return only and on contemporaneous plus 1..lags lagged market returns over trailing window ending at t; delay_ratio=max(0,1-R2_now/R2_full) with explicit R2_full<=eps -> null.
- **三后端实现要求**：Reuse stable rolling regression core; if market_ret_source=cs_ex_self compute aggregate-minus-self market return. Identical intercept/R²/df conventions.
- **PIT/no-lookahead**：Regressions end at current/previous completed return depending execution convention; no lead market returns. Use next-session if today close return is part of input.
- **最低测试集**：Synthetic no-delay vs one-lag process; collinearity; R2_full≈0; ex-self market; missing; window/lags; backend parity.
- **可解锁方向**：Price delay anomaly / information diffusion.
- **已预生成待解锁因子**：1 个
- **来源**：PKU_CHINA_ANOMALIES_469
- **优先级**：P1
- **字段门槛**：NONE

## winsorized_ratio
- **建议签名**：winsorized_ratio(num, den, denom_floor=1e-12, zero_policy="null", winsor="cs_mad", limit=5.0, preserve_sign=True)
- **定义/算法**：Stable accounting/valuation ratio. First validate denominator; compute num/den with explicit sign/zero policy; then winsorize the ratio cross-sectionally by decision date using robust median-MAD (default) or percentile caps. Do not winsorize num and den separately because that changes economic meaning.
- **参数**：denom_floor; zero_policy=null|signed_floor; winsor=none|cs_mad|percentile; limit for MAD or lower/upper percentile; preserve_sign.
- **Null policy**：Null if num/den null, or |den|<denom_floor under zero_policy=null. MAD=0 leaves finite ratios unchanged except infinities which are null.
- **排序/对齐**：Ratio is per stock/date; winsorization is within same-date eligible cross-section only, after PIT materialization of inputs.
- **三后端实现要求**：Reuse shared safe_divide + cs robust winsor primitive if available. Exact cap thresholds and finite/infinite handling identical in all backends.
- **PIT/no-lookahead**：No time lookahead; cross-sectional operation uses only same decision-date values.
- **最低测试集**：Positive/negative/zero/tiny denominators; infinities; MAD=0; outlier; small cross-section; percentile mode; backend parity.
- **可解锁方向**：Robust profitability, valuation, leverage, accrual and issuance ratios from anomaly libraries.
- **已预生成待解锁因子**：2 个
- **来源**：PYANOMALY|OSAP_UNIVERSE
- **优先级**：P1
- **字段门槛**：NONE

## HMA
- **建议签名**：HMA(x, window=16, rounding="floor")
- **定义/算法**：Hull MA = WMA(2*WMA(x,n/2)-WMA(x,n),sqrt(n)).
- **参数**：window>=2; n2=max(1,round_rule(window/2)); ns=max(1,round_rule(sqrt(window))); rounding=floor|round but default floor fixed in catalog.
- **Null policy**：Null until all component WMAs are valid under WMA strict policy. No ad-hoc fill.
- **排序/对齐**：Compute raw_t=2*WMA(x,n2)-WMA(x,window), then HMA=WMA(raw,ns), all trailing per stock.
- **三后端实现要求**：Compose the canonical WMA operator; do not implement a separate weighted routine. Parameter rounding centralized.
- **PIT/no-lookahead**：Historical trailing WMAs only.
- **最低测试集**：Windows 2/9/16/25; n/2 and sqrt rounding; constant/ramp sequence; inherited nulls; equality with explicit WMA composition; backend parity.
- **可解锁方向**：Low-lag trend factors.
- **已预生成待解锁因子**：1 个
- **来源**：PANDAS_TA
- **优先级**：P2
- **字段门槛**：NONE

## KAMA
- **建议签名**：KAMA(x, er_window=10, fast=2, slow=30, init="sma")
- **定义/算法**：Kaufman adaptive moving average using efficiency ratio to adapt smoothing constant.
- **参数**：er_window>=2; fast>=1; slow>fast; init=sma|first_valid. fastSC=2/(fast+1), slowSC=2/(slow+1).
- **Null policy**：Null until ER window and initialization valid. A hard missing gap resets state by default; no implicit zero.
- **排序/对齐**：ER_t=|x_t-x_{t-er}|/sum_{j=t-er+1}^t |Δx_j|; SC_t=(ER*(fastSC-slowSC)+slowSC)^2; KAMA_t=KAMA_{t-1}+SC_t*(x_t-KAMA_{t-1}).
- **三后端实现要求**：Stateful recurrence with rolling ER; define denominator zero => ER=0. Same init/reset/update order across backends.
- **PIT/no-lookahead**：All terms <=t; trailing only.
- **最低测试集**：Flat series ER=0; monotone trend ER≈1; choppy series; zero denominator; init modes; gap reset; backend parity.
- **可解锁方向**：Adaptive trend factors.
- **已预生成待解锁因子**：1 个
- **来源**：TA_LIB|PANDAS_TA
- **优先级**：P2
- **字段门槛**：NONE

## QQE
- **建议签名**：QQE(x, length=14, smooth=5, factor=4.236, ma_mode="ema", drift=1, output="line|basis|long|short|trend")
- **定义/算法**：Quantitative Qualitative Estimation. Compute RSI(length), smooth it by MA(smooth), take abs drift change of smoothed RSI, double-EMA that range with Wilder length 2*length-1, multiply by factor to form adaptive bands, then recursively trail long/short bands and switch trend when smoothed RSI crosses the prior opposite band.
- **参数**：length>1; smooth>0; factor>0; ma_mode initially ema only for parity; drift>=1; output selector.
- **Null policy**：Null until RSI, smoothing and double-range warmup are available. Do not fill gaps with zero; reset recursive band state after a configurable hard gap (default reset).
- **排序/对齐**：Per stock chronological stateful scan. At t, band/trend state may use only current inputs and state carried from <=t-1.
- **三后端实现要求**：Implement one canonical state machine shared by backends. Polars via map_groups/stateful scan or native expression plugin; Pandas reference loop; DuckDB via recursive/state UDF or pre-registered scalar aggregate. Numerical tolerance 1e-10 to reference.
- **PIT/no-lookahead**：Pure trailing recursion; no centered windows or future bars. Daily close/vwap signal used at next eligible decision boundary unless execution policy explicitly allows close auction.
- **最低测试集**：Reference sequence against published PandasTA-style QQE; flat price; monotone rise/fall; gap/reset; band crossing; warmup; output=line/basis/long/short/trend; cross-backend parity.
- **可解锁方向**：QQE momentum/trend-strength factors and cross-sectional combinations with valuation/quality states.
- **已预生成待解锁因子**：3 个
- **来源**：PANDAS_TA
- **优先级**：P2
- **字段门槛**：NONE

## RSX
- **建议签名**：RSX(x, length=14, output="value")
- **定义/算法**：Relative Strength Xtra/Jurik-inspired low-lag RSI variant. Use the published multi-stage recursive smoothing recurrence: scale price, recursively smooth signed one-step change through three cascaded two-pole stages; run the same cascade on absolute change; form 50*(smoothed_signed/smoothed_abs+1), clamp to [0,100], and emit 50 during initialization/degenerate denominator.
- **参数**：length>1. Core smoothing coefficient alpha=3/(length+2), beta=1-alpha; warmup threshold max(length-1,5).
- **Null policy**：Null for the first length-1 observations; initialization value 50 after warmup if denominator <=1e-10. A hard missing gap resets recursive state by default.
- **排序/对齐**：Per stock chronological state machine. Keep all recurrence states in float64 and update in fixed documented order.
- **三后端实现要求**：Pandas implementation is golden reference; Polars and DuckDB must implement exactly the same recurrence/state update order, not substitute RSI/EMA approximations. Tolerance <=1e-10 on deterministic fixtures.
- **PIT/no-lookahead**：Historical recurrence only; no future values.
- **最低测试集**：Published recurrence fixture; flat/monotone/oscillating sequence; length 5/14/30; denominator near zero; gaps/reset; clamping; backend parity.
- **可解锁方向**：Low-noise momentum/velocity factors and RSX divergence/interactions.
- **已预生成待解锁因子**：2 个
- **来源**：PANDAS_TA
- **优先级**：P2
- **字段门槛**：NONE

## WMA
- **建议签名**：WMA(x, window, min_periods=None, normalize=True)
- **定义/算法**：Linear weighted moving average weights 1..window, newest largest.
- **参数**：window>=2; min_periods defaults window; weights oldest..newest =1..window; normalize must be true for standard WMA.
- **Null policy**：Default strict: null until window valid non-null observations. Optional partial mode reindexes weights to surviving chronological observations and renormalizes; mode must be explicit.
- **排序/对齐**：Per stock trailing window ending at t; newest observation receives largest weight window.
- **三后端实现要求**：Rolling fixed-weight dot product in float64; DuckDB may use generated lag terms for small windows or UDF; no reversed weight order. Shared partial-null semantics.
- **PIT/no-lookahead**：Trailing window only; no centered weights.
- **最低测试集**：Window 2/3 hand calculation; monotone series confirms newest-largest orientation; null strict/partial; window>series; backend parity.
- **可解锁方向**：Technical factor families from public TA libraries.
- **已预生成待解锁因子**：1 个
- **来源**：TA_LIB|PANDAS_TA
- **优先级**：P2
- **字段门槛**：NONE

## cs_factor_bucket_return
- **建议签名**：cs_factor_bucket_return(signal, ret, q=10, weighting="equal", lag=1, output="long_short|bucket")
- **定义/算法**：Lagged realized return of portfolios formed on a characteristic, usable as state input for factor momentum / characteristic payoff dynamics.
- **参数**：q>=2; weighting=equal|market_cap; lag>=1; output=long_short|bucket; bucket_id optional; min_names_per_bucket; tie_method=stable_rank.
- **Null policy**：Null if cross-section cannot form requested buckets with min names. Missing signal excluded from formation; missing realized return excluded with weights renormalized.
- **排序/对齐**：Form buckets on signal at formation date t-lag using stable security-id tiebreak, observe completed forward period return that is fully known by t, aggregate bucket returns, then broadcast only at t or later.
- **三后端实现要求**：Shared quantile/rank bucket assignment and weight normalization; no backend-native ntile differences without golden matching.
- **PIT/no-lookahead**：The realized payoff used as a feature must correspond to a period completed before current decision time. Current/future target returns forbidden.
- **最低测试集**：Many ties; tiny universe; missing returns; cap weights; q=2/10; bucket/long-short outputs; lag boundary; backend parity.
- **可解锁方向**：Factor momentum / characteristic-payoff momentum directions.
- **已预生成待解锁因子**：4 个
- **来源**：FACTOR_MOMENTUM|FACTOR_TREE_2025
- **优先级**：P2
- **字段门槛**：NONE

## pastor_stambaugh_beta
- **建议签名**：pastor_stambaugh_beta(ret, volume, amount, window=252, innovation_window=60)
- **定义/算法**：Stock exposure to market-wide liquidity innovations using a documented Pastor-Stambaugh-style daily liquidity proxy constructed from available price/volume/amount.
- **参数**：window>=60; innovation_window>=20; market_weighting=equal|cap; liquidity_proxy=return_reversal_per_value; min_names; min_periods.
- **Null policy**：Stock-day liquidity proxy null when amount/volume invalid. Market liquidity null if names<min_names. Beta null if regression history<min_periods or innovation variance≈0.
- **排序/对齐**：Step1 per stock-day compute signed next-day/current reversal scaled by amount using only documented lagged construction; Step2 aggregate market liquidity; Step3 estimate expected market liquidity from trailing AR terms and innovation; Step4 rolling regress stock returns or stock liquidity on lagged market-liquidity innovation according to chosen beta definition.
- **三后端实现要求**：Treat as a composite operator with exposed intermediate diagnostics. Panel aggregation + time-series regression must use same weighting, lag and AR convention across backends.
- **PIT/no-lookahead**：Critical: any return needed to measure day-t liquidity makes that liquidity observation available only after that return completes; beta feature at t must use innovations fully observed before decision time.
- **最低测试集**：Zero amount; all same return; known AR liquidity series; innovation lag; cap/equal market aggregation; insufficient history; backend parity.
- **可解锁方向**：Pastor-Stambaugh beta family in PKU frictions library.
- **已预生成待解锁因子**：1 个
- **来源**：PKU_CHINA_ANOMALIES_469
- **优先级**：P2
- **字段门槛**：NONE

## same_calendar_day_mean
- **建议签名**：same_calendar_day_mean(ret, years_back=5, tolerance_sessions=2)
- **定义/算法**：Average return around same calendar day-of-year across prior years, with nearest-trading-session tolerance.
- **参数**：years_back>=1; tolerance_sessions>=0; aggregate=mean|median|ewm; optional min_matches; matching_anchor=month_day|day_of_year (default month_day).
- **Null policy**：Null if fewer than min_matches valid historical matched sessions. Never substitute a future session when tolerance search is configured prior_only.
- **排序/对齐**：For each decision date, build target month/day, search each prior year within ±tolerance_sessions on canonical A-share trading calendar, choose deterministic nearest session (tie -> earlier), then aggregate matched returns.
- **三后端实现要求**：Use a shared trading-calendar dimension and keyed joins. Polars/DuckDB/Pandas must use the same leap-day rule, nearest-session tie rule and aggregation order.
- **PIT/no-lookahead**：Only sessions from years strictly earlier than the decision year are eligible. No current/future-year observations.
- **最低测试集**：Feb-29; Lunar/New-Year holiday displacement; two equidistant sessions; suspended stock on matched date; min_matches; mean/median; backend parity.
- **可解锁方向**：Daily seasonality and recurring calendar effects.
- **已预生成待解锁因子**：2 个
- **来源**：RETURN_SEASONALITY
- **优先级**：P2
- **字段门槛**：NONE

## ALMA
- **建议签名**：ALMA(x, window=10, offset=0.85, sigma=6.0, min_periods=None)
- **定义/算法**：Arnaud Legoux MA: Gaussian weights centered at offset*(window-1), normalized.
- **参数**：window>=2; offset in [0,1]; sigma>0; m=offset*(window-1); s=window/sigma; weights_i=exp(-(i-m)^2/(2s^2)), i oldest..newest, normalized to sum 1.
- **Null policy**：Default strict null until full valid window. Optional partial mode drops null slots and renormalizes their original weights; must be explicit.
- **排序/对齐**：Per stock trailing window, index i=0 oldest and window-1 newest. Offset therefore controls weight center toward recent data when >0.5.
- **三后端实现要求**：Precompute normalized weight vector per parameter tuple; rolling dot product float64; identical old/new orientation across backends.
- **PIT/no-lookahead**：Trailing window only.
- **最低测试集**：Weight sum=1; offset 0/0.5/1 orientation; constant series; impulse location; null strict/partial; backend parity.
- **可解锁方向**：Smooth trend/price-distance factors.
- **已预生成待解锁因子**：1 个
- **来源**：PANDAS_TA
- **优先级**：P3
- **字段门槛**：NONE

## CoppockCurve
- **建议签名**：CoppockCurve(close, roc1=14, roc2=11, wma=10, roc_mode="pct")
- **定义/算法**：WMA of sum of two rate-of-change series.
- **参数**：roc1,roc2>=1; wma>=2; roc_mode=pct|log. Standard formula WMA(ROC_roc1+ROC_roc2,wma).
- **Null policy**：Null until both ROC lookbacks and WMA window are valid; invalid/nonpositive lag close under selected ROC mode -> null.
- **排序/对齐**：Per stock compute trailing ROC ending t, sum two ROC series, then canonical WMA with newest-largest weights.
- **三后端实现要求**：Compose canonical return/ROC and WMA operators; exact percent definition (close/lag-1) fixed.
- **PIT/no-lookahead**：All close values <=t; trailing only.
- **最低测试集**：Hand-computed ROC+WMA; constant price; ramp; zero lag price; differing lookbacks; backend parity.
- **可解锁方向**：Long-horizon momentum cycle.
- **已预生成待解锁因子**：1 个
- **来源**：PANDAS_TA
- **优先级**：P3
- **字段门槛**：NONE

## ElderRay
- **建议签名**：ElderRay(high, low, close, ema=13, output="bull|bear|spread", ema_init="sma")
- **定义/算法**：Bull power=high-EMA(close); bear power=low-EMA(close).
- **参数**：ema>=2; output=bull|bear|spread|both; bull=high-EMA(close), bear=low-EMA(close), spread=bull-bear=high-low.
- **Null policy**：Null until EMA initialization valid and required high/low present. No fill across hard gaps unless EMA policy explicitly carries state.
- **排序/对齐**：Per stock chronological EMA of close; evaluate bull/bear on same bar high/low against that trailing EMA.
- **三后端实现要求**：Compose canonical EMA with fixed alpha/init; do not use backend-native EMA defaults if they differ. Return float64.
- **PIT/no-lookahead**：Uses current completed bar and historical EMA only; next-session feature under close-based decision convention.
- **最低测试集**：Constant bar; rising/falling sequence; bull/bear signs; spread identity; EMA init/gap; backend parity.
- **可解锁方向**：Trend-pressure factors.
- **已预生成待解锁因子**：2 个
- **来源**：PANDAS_TA
- **优先级**：P3
- **字段门槛**：NONE

## FisherTransform
- **建议签名**：FisherTransform(high, low, window=9, smooth=0.33, signal_smooth=0.5, clip=0.999)
- **定义/算法**：Normalize median price in rolling range to (-.999,.999), then 0.5*ln((1+x)/(1-x)); recursive smoothing documented.
- **参数**：window>=2; 0<smooth<=1; 0<signal_smooth<=1; 0<clip<1; source=(high+low)/2.
- **Null policy**：Null until rolling high/low range valid. Zero range -> normalized input 0; null OHLC resets/propagates per state policy, never divide by zero.
- **排序/对齐**：Compute rolling min/max of source; raw=2*((source-min)/(max-min)-0.5); recursively smooth z_t=smooth*raw+(1-smooth)*z_{t-1}; clip z; fisher_t=0.5*ln((1+z)/(1-z)); optional signal is lag/smoothed fisher.
- **三后端实现要求**：Rolling extrema + stateful recursion in float64. Same source definition, clip, init=0 and update order across backends.
- **PIT/no-lookahead**：Rolling range and recursion end at t; no future high/low.
- **最低测试集**：Flat range; single spike; clip boundary; constant trend; state reset gap; signal output if exposed; backend parity.
- **可解锁方向**：Fisher reversal/trend factors.
- **已预生成待解锁因子**：1 个
- **来源**：TA_LIB|PANDAS_TA
- **优先级**：P3
- **字段门槛**：NONE

---

## 附录来源：`r42_operator_engineering_specs.md`

> 以下是 R41–R47 原始工程规格，作为候选语义档案保留。**若与本文件前文的 current-main 规则、字段硬约束、backend policy 或 preflight 结果冲突，以前文为准。**

# R42 新增算子工程规格与待解锁因子

本文件可直接作为 FactorEngine 新算子开发输入。R41 已有建议仍保留在 `reports/r41_operator_gap_recommendations.csv`；本文件只展开 R42 新增项。

## 全局工程约束

1. **字段合同不扩张**：所有算子仅消费当前 276 个已审字段或由这些字段在算子内部生成的中间量。
2. **三后端一致**：Pandas / Polars / DuckDB 的窗口端点、空值、排序、ddof、回归截距与阈值完全一致；Polars/DuckDB 生产路径不得转 Pandas。
3. **PIT**：任何同日需要 `LEAD`/后邻 bar 的事件分类默认只允许收盘后生成、下一交易日使用；panel 模型必须使用成熟标签并做 horizon embargo。
4. **ex-self**：市场/同业聚合必须排除目标股票自身。
5. **可审计**：实现后登记签名、最小历史、cost tier、PIT 标签、backend evidence 与 golden tests。

## 1. `intra_same_slot_zscore`（P0）

- **签名**：`intra_same_slot_zscore(x, history_days=20, ddof=1, min_history=10)`
- **数学/算法定义**：For each stock-day-minute slot, z-score current x against only the same clock slot in prior trading days: (x_t - mean_prior_slot)/std_prior_slot. Returns a minute series aligned to current day.
- **参数**：history_days>1; ddof in {0,1}; min_history>=2
- **空值与异常值**：If fewer than min_history valid prior observations or prior std<=eps, return null for that bar; never replace with zero.
- **排序/时间对齐**：Group strictly by security and exchange-local clock slot; prior-day history only; lunch break/non-trading slots are not forward-filled.
- **Pandas / Polars / DuckDB**：Pandas groupby/rolling shift(1); Polars group_by_dynamic or slot-key rolling with explicit shift; DuckDB window frame ending 1 PRECEDING. Numerical tolerance 1e-10.
- **PIT / 防未来函数**：Current-day bar must not enter its reference mean/std. Same-slot history ends at previous trading day.
- **最低测试**：synthetic constant slots=>null; injected spike sign; day-shift leakage test; missing-slot test; 3-backend parity
- **解锁方向**：volume/amount peak-ridge-valley state classification
- **来源**：KYSEC_VOLUME_PRV_2025
- **字段门槛**：`minute_volume|minute_amount`
- **已预建候选因子**：**0**

## 2. `intra_neighbor_event_class`（P0）

- **签名**：`intra_neighbor_event_class(event_mask, radius=1, isolated_code=1, clustered_code=2)`
- **数学/算法定义**：Classify each true event bar using adjacent bars within same trading session: isolated if no true event within +/-radius; clustered if at least one neighbour is true; false bars return 0.
- **参数**：radius>=1; integer output codes
- **空值与异常值**：Null event_mask propagates null at that bar; session-edge absent neighbours count as false, not null.
- **排序/时间对齐**：Never cross security/day/session boundary or lunch break; ordering by timestamp ascending.
- **Pandas / Polars / DuckDB**：Use shift/join per security-session in all backends; avoid global shift.
- **PIT / 防未来函数**：Only contemporaneous and past/future bars within the already completed historical day are used to create the EOD feature. For same-day live use, future-neighbour classification is not allowed until session close.
- **最低测试**：single isolated; two adjacent events; lunch boundary; day boundary; parity
- **解锁方向**：exact volume peak/ridge classification
- **来源**：KYSEC_VOLUME_PRV_2025
- **字段门槛**：`minute_volume|minute_amount`
- **已预建候选因子**：**0**

## 3. `intra_state_count`（P0）

- **签名**：`intra_state_count(state, target, window_days=1)`
- **数学/算法定义**：Count minute bars whose discrete state equals target over the current day or trailing completed days.
- **参数**：target scalar/list; window_days>=1
- **空值与异常值**：Null state bars excluded from denominator/count. Return null only if no valid bars in required window.
- **排序/时间对齐**：Per security, chronological complete-day aggregation.
- **Pandas / Polars / DuckDB**：Conditional count + rolling daily sum in all three backends.
- **PIT / 防未来函数**：For window_days>1 include current day only at EOD; no future dates.
- **最低测试**：hand-count fixture; missing bars; rolling-day boundary; parity
- **解锁方向**：peak/ridge/valley minute-count factors
- **来源**：KYSEC_VOLUME_PRV_2025|KYSEC_PRICE_PRV_2026
- **字段门槛**：`minute_close|minute_volume`
- **已预建候选因子**：**36**

## 4. `intra_state_sum`（P0）

- **签名**：`intra_state_sum(x, state, target, window_days=1)`
- **数学/算法定义**：Sum x over bars in target state, then optionally sum daily state totals over trailing completed days.
- **参数**：target; window_days>=1
- **空值与异常值**：Ignore x when state not target; within target, valid x summed; null if target has no valid x.
- **排序/时间对齐**：Security/day/session safe.
- **Pandas / Polars / DuckDB**：CASE WHEN + daily aggregate + trailing row window.
- **PIT / 防未来函数**：No future day/bar beyond decision timestamp.
- **最低测试**：manual sums; no-event; missing x; parity
- **解锁方向**：state return/amount totals
- **来源**：KYSEC_VOLUME_PRV_2025|KYSEC_PRICE_PRV_2026
- **字段门槛**：`minute_close|minute_amount|minute_volume`
- **已预建候选因子**：**54**

## 5. `intra_state_vwap`（P0）

- **签名**：`intra_state_vwap(price, volume, state, target, window_days=1)`
- **数学/算法定义**：VWAP of bars in target state: sum(price*volume)/sum(volume), optionally across trailing days.
- **参数**：target; window_days>=1; eps=1e-12
- **空值与异常值**：Require positive finite state volume; zero/invalid denominator => null.
- **排序/时间对齐**：State and price/volume must share exact bar alignment.
- **Pandas / Polars / DuckDB**：Numerator/denominator conditional aggregates; same formula all backends.
- **PIT / 防未来函数**：Only available bars; trailing window ends at decision day.
- **最低测试**：constant price; zero volume; state subset; parity
- **解锁方向**：valley/peak/ridge relative VWAP factors
- **来源**：KYSEC_VOLUME_PRV_2025|KYSEC_PRICE_PRV_2026
- **字段门槛**：`minute_close|minute_volume`
- **已预建候选因子**：**36**

## 6. `intra_state_interval_moment`（P0）

- **签名**：`intra_state_interval_moment(state, target, moment="std", window_days=20, min_events=4)`
- **数学/算法定义**：Collect time gaps (in trading minutes) between consecutive target-state events within each day; aggregate trailing gaps and return std/skew/kurtosis.
- **参数**：moment in {std,skew,kurtosis}; window_days>=1; min_events>=4
- **空值与异常值**：Days with <2 target events contribute no gap; insufficient pooled gaps => null.
- **排序/时间对齐**：No overnight or lunch-gap duration is counted as continuous trading minutes; reset gap chain per session/day.
- **Pandas / Polars / DuckDB**：Generate event timestamps, lag within day/session, convert to trading-minute gap; rolling moment.
- **PIT / 防未来函数**：Trailing completed observations only.
- **最低测试**：known gap vector moments; session reset; sparse events; parity
- **解锁方向**：peak/ridge interval dispersion/skew/kurtosis factors
- **来源**：KYSEC_VOLUME_PRV_2025|KYSEC_PRICE_PRV_2026
- **字段门槛**：`minute_close|minute_volume`
- **已预建候选因子**：**42**

## 7. `intra_state_follow_ratio`（P0）

- **签名**：`intra_state_follow_ratio(x, state, target, lead_bars=1, window_days=20)`
- **数学/算法定义**：For each target event with an in-session lead bar, compute aggregate sum(x_lead)/sum(x_event) over trailing days.
- **参数**：lead_bars>=1; window_days>=1; eps
- **空值与异常值**：Drop target events lacking an in-session lead; zero denominator => null.
- **排序/时间对齐**：Lead never crosses lunch/day/security boundary.
- **Pandas / Polars / DuckDB**：Self-join by bar index+lead or partitioned LEAD; rolling aggregate.
- **PIT / 防未来函数**：Historical EOD factor may use next bar of past events. A current live event is ineligible until its lead bar completes.
- **最低测试**：event at session end; zero denom; hand ratio; parity
- **解锁方向**：follow-amount ratio after volume bursts/jumps
- **来源**：KYSEC_VOLUME_PRV_2025|KYSEC_PRICE_PRV_2026
- **字段门槛**：`minute_amount|minute_volume`
- **已预建候选因子**：**6**

## 8. `intra_state_follow_beta`（P1）

- **签名**：`intra_state_follow_beta(x, state, target, lead_bars=1, window_days=20, min_events=8)`
- **数学/算法定义**：OLS slope of x at lead bar on x at target event across trailing target events. Include intercept; return slope only.
- **参数**：lead_bars; window_days; min_events; eps
- **空值与异常值**：Pairwise complete events only; insufficient variance/events => null.
- **排序/时间对齐**：Lead/session rules identical to intra_state_follow_ratio.
- **Pandas / Polars / DuckDB**：Use stable centered covariance/variance formula in all backends.
- **PIT / 防未来函数**：No current event until lead available; no future day.
- **最低测试**：known linear relation beta; constant x=>null; session boundary; parity
- **解锁方向**：follow-amount beta factor
- **来源**：KYSEC_VOLUME_PRV_2025
- **字段门槛**：`minute_amount`
- **已预建候选因子**：**6**

## 9. `intra_state_follow_corr`（P0）

- **签名**：`intra_state_follow_corr(x, state, target, lead_bars=1, window_days=20, min_events=8)`
- **数学/算法定义**：Pearson correlation between x at target events and x lead_bars later.
- **参数**：lead_bars; window_days; min_events
- **空值与异常值**：Pairwise complete; zero variance => null.
- **排序/时间对齐**：Same partition/session rules as follow beta.
- **Pandas / Polars / DuckDB**：Centered covariance/std; deterministic.
- **PIT / 防未来函数**：Historical lead only.
- **最低测试**：perfect corr; anti-corr; zero variance; boundary; parity
- **解锁方向**：jump/volume follow correlation factors
- **来源**：KYSEC_VOLUME_PRV_2025|KYSEC_PRICE_PRV_2026
- **字段门槛**：`minute_amount|minute_volume`
- **已预建候选因子**：**12**

## 10. `intra_state_pair_same_slot_corr`（P1）

- **签名**：`intra_state_pair_same_slot_corr(state_a, target_a, state_b, target_b, window_days=20, min_slots=8)`
- **数学/算法定义**：For each clock slot, count occurrences of state A and state B over trailing days; return cross-slot Pearson correlation of the two count vectors.
- **参数**：window_days>=2; min_slots>=3
- **空值与异常值**：Slots lacking observations are excluded pairwise; zero variance => null.
- **排序/时间对齐**：Clock-slot mapping exchange-local; no mixing across securities.
- **Pandas / Polars / DuckDB**：Pivot/aggregate by slot then correlation; DuckDB via grouped stats.
- **PIT / 防未来函数**：Trailing window ends at current EOD; no future dates.
- **最低测试**：identical states=>1; disjoint pattern negative; missing slots; parity
- **解锁方向**：peak-vs-ridge same-slot correlation
- **来源**：KYSEC_VOLUME_PRV_2025
- **字段门槛**：`minute_volume`
- **已预建候选因子**：**6**

## 11. `intra_range_gap_flag`（P0）

- **签名**：`intra_range_gap_flag(high, low, event_mask, neighbor_bars=1)`
- **数学/算法定义**：For each event bar, compare the immediately pre-event and post-event price ranges. Return 1 if ranges do not overlap (strict gap), 0 if they overlap/touch.
- **参数**：neighbor_bars>=1; strict=True/False optional
- **空值与异常值**：If either neighbouring range missing or crosses a session boundary => null.
- **排序/时间对齐**：Pre/post neighbours must be same stock/day/session.
- **Pandas / Polars / DuckDB**：Partitioned LAG/LEAD with range-overlap predicate.
- **PIT / 防未来函数**：Post-event neighbour makes current-day live output unavailable until that bar completes; EOD historical safe.
- **最低测试**：overlap/touch/gap fixtures; session boundary; parity
- **解锁方向**：price-jump gap/no-gap classification
- **来源**：KYSEC_PRICE_PRV_2026
- **字段门槛**：`minute_high|minute_low`
- **已预建候选因子**：**0**

## 12. `intra_volume_peak_ridge_valley_state`（P0）

- **签名**：`intra_volume_peak_ridge_valley_state(volume, history_days=20, z_threshold=1.0, radius=1)`
- **数学/算法定义**：Exact public-style state engine: abnormal/burst bar if current same-slot volume exceeds prior same-slot mean + z_threshold*prior same-slot std; mild bars=valley; burst bars isolated from burst neighbours=peak; burst bars adjacent to another burst=ridge. Return categorical 0 valley/1 peak/2 ridge.
- **参数**：history_days; z_threshold; radius
- **空值与异常值**：Requires valid prior same-slot stats; otherwise state=null.
- **排序/时间对齐**：Same-slot benchmark uses previous trading days only; neighbour classification stays inside session/day.
- **Pandas / Polars / DuckDB**：Compose same-slot stats + event classification in vectorized backend-native implementation; no Python UDF in Polars/DuckDB production.
- **PIT / 防未来函数**：Benchmark strictly shifted; current-day future neighbour only permitted for EOD output.
- **最低测试**：public worked example; isolated/clustered bursts; shifted benchmark leakage; parity
- **解锁方向**：all 20 volume peak/ridge/valley public factors
- **来源**：KYSEC_VOLUME_PRV_2025
- **字段门槛**：`minute_volume`
- **已预建候选因子**：**132**

## 13. `intra_price_peak_ridge_valley_state`（P0）

- **签名**：`intra_price_peak_ridge_valley_state(open, high, low, close, history_days=20, jump_z=1.0, radius=1)`
- **数学/算法定义**：State engine for public price-jump peak/ridge/valley framework. Detect jump bars from minute amplitude relative to prior historical scale; non-jump=valley. For jump bars classify local neighbour jump sentiment and pre/post range gap; map non-high-local-sentiment & no-gap jumps to peak and non-low-local-sentiment & gap jumps to ridge; retain auxiliary class metadata if requested.
- **参数**：history_days; jump_z; radius; return_detail=False
- **空值与异常值**：Missing benchmark or neighbour ranges => null state; no silent fallback.
- **排序/时间对齐**：Per stock/day/session; historical benchmark shifted; price ranges use aligned adjacent bars.
- **Pandas / Polars / DuckDB**：Backend-native amplitude, shifted historical scale, neighbour states and gap predicate.
- **PIT / 防未来函数**：No benchmark leakage; EOD only for post-neighbour classifications.
- **最低测试**：classification truth table; boundary cases; historical shift; parity
- **解锁方向**：17 public price peak/ridge/valley factors
- **来源**：KYSEC_PRICE_PRV_2026
- **字段门槛**：`minute_open|minute_high|minute_low|minute_close`
- **已预建候选因子**：**66**

## 14. `intra_smart_money_vwap_ratio`（P0）

- **签名**：`intra_smart_money_vwap_ratio(close, volume, lookback_days=10, beta=0.25, cumulative_volume_cut=0.20)`
- **数学/算法定义**：Exact smart-money selection primitive. Compute minute return r; score S=|r|/volume^beta (beta configurable; public implementation commonly uses 0.25). Pool trailing minute bars, sort descending S with deterministic tie-break timestamp, select bars until cumulative selected volume reaches cut*total volume; compute VWAP_smart/VWAP_all.
- **参数**：lookback_days>=1; beta>=0; cumulative_volume_cut in (0,1); tie_policy="timestamp"
- **空值与异常值**：volume<=0 or invalid bars excluded; require positive total and smart volumes; otherwise null.
- **排序/时间对齐**：Pooling is per security over trailing completed bars; sorting deterministic. Current day may be included only after decision cutoff/EOD.
- **Pandas / Polars / DuckDB**：Pandas stable sort; Polars sort + cum_sum; DuckDB ORDER BY S DESC,timestamp with cumulative SUM window.
- **PIT / 防未来函数**：No future bars; at EOD window ends current session, next-open signal is safe.
- **最低测试**：hand-ranked 6-bar fixture; cutoff crossing; ties; beta cases; zero volume; 3-backend parity
- **解锁方向**：exact smart-money Q factor and beta/cut sensitivity family
- **来源**：KYSEC_SMART_MONEY_2_2020|KYSEC_SMART_FORMULA_PUBLIC
- **字段门槛**：`minute_close|minute_volume`
- **已预建候选因子**：**18**

## 15. `intra_smart_money_fcm_score`（P2）

- **签名**：`intra_smart_money_fcm_score(close, volume, amount, lookback_days=10, n_clusters=2, m=2.0)`
- **数学/算法定义**：Fuzzy-c-means smart-money state score using only price/volume/amount-derived standardized minute features. Fit FCM on trailing bars; identify smart cluster deterministically as lower price-impact / higher activity cluster; output membership-weighted smart VWAP ratio or mean membership.
- **参数**：lookback_days; n_clusters>=2; fuzzifier m>1; max_iter; tol; seed fixed
- **空值与异常值**：Rows missing required features excluded; insufficient rows=>null.
- **排序/时间对齐**：Per security trailing historical window; deterministic initialization/seed and cluster relabeling.
- **Pandas / Polars / DuckDB**：Same mathematical FCM iteration across backends; production may use shared numeric kernel but outputs must be backend-equivalent.
- **PIT / 防未来函数**：Fit only on information available through decision cutoff; no future fit.
- **最低测试**：cluster permutation invariance; deterministic seed; synthetic two-cluster recovery; parity tolerance
- **解锁方向**：FCM smart-money candidates from A-share academic evidence
- **来源**：SMART_MONEY_ASHARE_2022
- **字段门槛**：`minute_close|minute_volume|minute_amount`
- **已预建候选因子**：**6**

## 16. `panel_apm_residual_tstat`（P1）

- **签名**：`panel_apm_residual_tstat(open, pre_close, minute_close, lookback_days=20, afternoon="afternoon", neutralize_momentum=True)`
- **数学/算法定义**：Compute ex-self market return for overnight and afternoon sessions, run stock session returns on corresponding ex-self market returns over trailing days, obtain residuals, form delta=resid_night-resid_afternoon, return mean(delta)/(std(delta)/sqrt(N)). Optional final cross-sectional residualization against trailing momentum.
- **参数**：lookback_days>=10; afternoon segment; weighting="equal|cap"; neutralize_momentum
- **空值与异常值**：Need min observations and nonzero residual std; else null.
- **排序/时间对齐**：Market proxy excludes target stock; all regressions shifted/trailing; afternoon session calendar aware.
- **Pandas / Polars / DuckDB**：Panel aggregate + per-stock rolling OLS; Polars/DuckDB implementations must avoid self-inclusion via sum/count subtraction.
- **PIT / 防未来函数**：No target-day future if computed before afternoon close; default EOD/next-open.
- **最低测试**：ex-self market test; known residual delta; std zero; stock-universe entry/exit; parity
- **解锁方向**：exact APM-style statistic without requiring external index field
- **来源**：KYSEC_TRADING_BEHAVIOR_2026
- **字段门槛**：`open|pre_close|minute_close`
- **已预建候选因子**：**6**

## 17. `panel_day_night_beta_gap`（P1）

- **签名**：`panel_day_night_beta_gap(open, pre_close, minute_close, lookback_days=60, weighting="equal")`
- **数学/算法定义**：Estimate beta of stock overnight returns to ex-self overnight market returns and beta of stock intraday/day returns to ex-self day market returns over same trailing window; return beta_day-beta_night.
- **参数**：lookback_days; weighting; min_obs
- **空值与异常值**：Insufficient market/stock variance=>null.
- **排序/时间对齐**：Ex-self market at each date; same universe rules for day/night.
- **Pandas / Polars / DuckDB**：Panel rolling covariance/variance; ex-self aggregate computed without O(N^2) loops.
- **PIT / 防未来函数**：Trailing data only; current day only after close.
- **最低测试**：known beta panel; self-exclusion; missing days; parity
- **解锁方向**：day-night beta gap factor
- **来源**：DAY_NIGHT_DELAY_2024
- **字段门槛**：`open|pre_close|minute_close`
- **已预建候选因子**：**6**

## 18. `panel_async_beta_ex_self`（P1）

- **签名**：`panel_async_beta_ex_self(ret_like, lookback_days=60, max_lag=5, weighting="equal")`
- **数学/算法定义**：Estimate distributed-lag response of each stock return to ex-self market return using lags 0..max_lag. Return sum of lagged-market coefficients excluding contemporaneous beta, or optionally ratio lagged/total response.
- **参数**：lookback_days; max_lag>=1; output="lag_sum|lag_ratio"
- **空值与异常值**：Require full-rank regression and min observations; else null.
- **排序/时间对齐**：Market return ex-self; lag convention market_{t-k} predicts stock_t.
- **Pandas / Polars / DuckDB**：Panel ex-self aggregate plus rolling multivariate OLS.
- **PIT / 防未来函数**：Only past/contemporaneous market information; for predictive feature use coefficients estimated through t-1 or EOD t for next session.
- **最低测试**：synthetic delayed-beta data; no-delay zero; self-exclusion; parity
- **解锁方向**：asynchronous information-delay factors
- **来源**：DAY_NIGHT_DELAY_2024
- **字段门槛**：`ret|open|pre_close`
- **已预建候选因子**：**6**

## 19. `intra_jump_wavelet_morphology`（P1）

- **签名**：`intra_jump_wavelet_morphology(close, event_z=3.0, pre_bars=30, post_bars=30, wavelet="db4", feature="vol_asymmetry")`
- **数学/算法定义**：Event-centred wavelet representation around detected intraday jumps. Return daily aggregate of one morphology feature: pre/post volatility-energy asymmetry, local trend, local mean-reversion, or multiscale energy vector projection.
- **参数**：event_z; pre_bars; post_bars; wavelet; feature in {vol_asymmetry,trend,reversion,energy_pc1}
- **空值与异常值**：Events lacking full pre/post in-session window excluded; no events=>null.
- **排序/时间对齐**：Windows never cross lunch/day/session boundaries.
- **Pandas / Polars / DuckDB**：Deterministic DWT kernel shared/equivalent across backends; daily aggregation mean/median configurable.
- **PIT / 防未来函数**：Post-event window means same-day output becomes available only after post_bars pass; default EOD/next-open.
- **最低测试**：synthetic exogenous/endogenous shapes; no-event; boundary; wavelet energy conservation; parity
- **解锁方向**：wavelet jump-class factors
- **来源**：PNAS_WAVELET_JUMPS_2025
- **字段门槛**：`minute_close`
- **已预建候选因子**：**24**

## 20. `intra_cojump_breadth_ex_self`（P1）

- **签名**：`intra_cojump_breadth_ex_self(close, event_z=3.0, tolerance_bars=0, weighting="equal")`
- **数学/算法定义**：At each stock jump bar, compute fraction or weighted fraction of other stocks simultaneously classified as jump events; aggregate per stock-day (mean/max/event-weighted).
- **参数**：event_z; tolerance_bars; weighting; aggregate
- **空值与异常值**：If universe peers below minimum or no target events=>null.
- **排序/时间对齐**：Exact exchange timestamp alignment; target stock excluded.
- **Pandas / Polars / DuckDB**：Compute cross-sectional event count per timestamp then subtract own event; O(NT), not pairwise O(N^2).
- **PIT / 防未来函数**：Uses only contemporaneous event states for EOD/next-open feature.
- **最低测试**：single stock; all cojump; self exclusion; timestamp tolerance; parity
- **解锁方向**：cojump endogenous-contagion candidates
- **来源**：PNAS_WAVELET_JUMPS_2025
- **字段门槛**：`minute_close`
- **已预建候选因子**：**6**

## 21. `cs_topological_anomaly_score`（P2）

- **签名**：`cs_topological_anomaly_score(*features, window=60, method="ballmapper", n_landmarks=32, k=8)`
- **数学/算法定义**：Exploratory cross-sectional topology score. Standardize features using trailing-only parameters, build a BallMapper/landmark cover or deterministic graph of current cross-section, measure each stock local structural deviation from peer/topology expectation; return continuous anomaly score.
- **参数**：window; method; n_landmarks; k; seed fixed
- **空值与异常值**：Rows missing too many features=>null; no imputation with future cross-section.
- **排序/时间对齐**：At date t graph uses only date-t cross-section plus trailing scaling fitted through t-1 where predictive.
- **Pandas / Polars / DuckDB**：Deterministic landmark selection (fixed seed/farthest-point); backend may call shared numeric kernel but must return identical stock ordering/scores.
- **PIT / 防未来函数**：No future dates; if scaling uses current cross-section it is contemporaneous cross-sectional only, acceptable for next-session signal.
- **最低测试**：toy manifold with outlier; permutation invariance; seed determinism; missing features; parity
- **解锁方向**：topological anomaly candidate factors
- **来源**：TOPO_ANOMALY_INTRADAY_2026
- **字段门槛**：`close|ret|turnover_ratio|minute_volume|minute_close`
- **已预建候选因子**：**6**

## 22. `panel_predictability_mosaic_score`（P2）

- **签名**：`panel_predictability_mosaic_score(target_ret, *state_features, train_window=756, min_leaf=200, depth=3)`
- **数学/算法定义**：Walk-forward interpretable Panel Tree-style score that partitions historical stock-date observations by state features and estimates within-leaf predictive strength. Output for current stock-date is lagged leaf predictability score or a gated base signal, never the future return itself.
- **参数**：train_window; min_leaf; depth; retrain_freq; seed
- **空值与异常值**：Leaf with insufficient OOS history=>null/neutral according explicit option; default null.
- **排序/时间对齐**：Training labels end before prediction date; cross-sectional state at prediction date may route observation into a pre-trained tree.
- **Pandas / Polars / DuckDB**：Deterministic rolling tree implementation or shared kernel; serialize splits/leaf stats for audit.
- **PIT / 防未来函数**：Hard embargo: labels used for tree fitting must mature before fit date; purge horizon overlap.
- **最低测试**：label-shift leakage; synthetic state-dependent predictability; retrain determinism; OOS embargo; parity
- **解锁方向**：mosaic-state factor gating
- **来源**：NBER_MOSAICS_2026
- **字段门槛**：`ret|inc_net_profit_year_on_year|pe_ratio|turnover_ratio`
- **已预建候选因子**：**6**

## 23. `panel_factor_pocket_strength`（P1）

- **签名**：`panel_factor_pocket_strength(signal, forward_horizon=20, train_window=756, min_obs=252, decay=0.97)`
- **数学/算法定义**：Estimate whether a given signal is currently inside a historical pocket of positive/negative predictive efficacy using only matured past cross-sectional IC/portfolio-spread observations; return signed persistence/strength score.
- **参数**：forward_horizon; train_window; min_obs; decay; method="ewma|changepoint"
- **空值与异常值**：Insufficient matured history=>null.
- **排序/时间对齐**：Daily cross-sectional signal; realized forward returns enter efficacy history only after horizon fully matures.
- **Pandas / Polars / DuckDB**：Compute IC/spread history with strict label maturity then EWMA/regime score.
- **PIT / 防未来函数**：Mandatory horizon embargo; never use overlapping unmatured forward returns.
- **最低测试**：no-skill signal; regime switch; embargo unit test; parity
- **解锁方向**：factor-timing/pockets candidates
- **来源**：POCKETS_FACTOR_PRICING
- **字段门槛**：`ret|close`
- **已预建候选因子**：**24**

## 24. `intra_price_shape_cosine_match`（P1）

- **签名**：`intra_price_shape_cosine_match(close, template, segment="late", normalize="demean_l2")`
- **数学/算法定义**：Measure cosine similarity of normalized intraday return/price path in a chosen segment to a supplied deterministic template (e.g., chase-up, chase-down, V-shape). Output daily similarity.
- **参数**：template fixed vector/id; segment; normalize; min_bars
- **空值与异常值**：Insufficient bars=>null; zero-norm path/template=>null.
- **排序/时间对齐**：Resample/align to fixed within-session slots; never bridge lunch.
- **Pandas / Polars / DuckDB**：Vectorized dot/norm in all backends; template registry versioned.
- **PIT / 防未来函数**：Uses completed same-day segment only; next-open default.
- **最低测试**：identical template=>1; inverse=>-1; scale invariance; missing slot; parity
- **解锁方向**：chasing/killing intraday shape factors
- **来源**：KYSEC_TRADING_BEHAVIOR_2026
- **字段门槛**：`minute_close`
- **已预建候选因子**：**24**

## 25. `intra_distribution_moment`（P1）

- **签名**：`intra_distribution_moment(x, weight=None, moment="skew", normalize_share=False)`
- **数学/算法定义**：Compute within-day distribution moment of minute x or its normalized share: skewness, kurtosis, coefficient of variation. Optional weights.
- **参数**：moment in {skew,kurtosis,cv}; normalize_share; min_bars
- **空值与异常值**：Need sufficient finite bars; mean near zero for CV=>null.
- **排序/时间对齐**：Per security/day/session; no cross-day mixing.
- **Pandas / Polars / DuckDB**：Stable two-pass moments or equivalent; same ddof and bias convention all backends.
- **PIT / 防未来函数**：Contemporaneous completed-day aggregation only.
- **最低测试**：normal/symmetric samples; known skew/kurt; zero mean CV; parity
- **解锁方向**：volume-share skew/kurtosis and amount dispersion public HF factors
- **来源**：KYSEC_SMART_FORMULA_PUBLIC
- **字段门槛**：`minute_volume|minute_amount`
- **已预建候选因子**：**36**

---

## 附录来源：`r43_operator_engineering_specs.md`

> 以下是 R41–R47 原始工程规格，作为候选语义档案保留。**若与本文件前文的 current-main 规则、字段硬约束、backend policy 或 preflight 结果冲突，以前文为准。**

# R43 新增算子工程规格与待解锁因子

本文件可直接交给 FactorEngine 开发 AI/工程师实现。R42 及更早建议仍保留；这里只展开 R43 新增算子。

## 全局约束

1. **不扩原始字段合同**：只能消费当前审定字段或由其在算子内部生成的中间量。
2. **三后端一致**：Pandas/Polars/DuckDB 的窗口端点、session 切分、排序、分位数、ddof、回归截距和空值规则必须一致；生产 Polars/DuckDB 禁止退回 Pandas。
3. **PIT**：禁止负 lag/lead 读取未来分钟；需要整日样本的算子默认 EOD 计算、下一交易日使用。
4. **Panel ex-self**：市场/peer 聚合排除目标股票自身。
5. **确定性**：KNN/聚类/分位数 ties 使用稳定 security-id/timestamp 规则。
6. **登记与测试**：实现后登记 callshape、cost tier、min history、PIT label、backend evidence 与 golden tests。

## 1. `intra_slice_mask_reduce` — P0

- **签名**：`intra_slice_mask_reduce(x, mask_field, window="All", slice=None, mask_side="high", mask_q=0.7, reducer="mean", min_bars=10)`
- **数学/算法定义**：Select an intraday interval centered at normalized slice (or trailing window when slice=None), compute within-slice empirical percentile ranks of mask_field, retain bars satisfying high/low quantile rule, then reduce x to one daily scalar. Reducers: mean/std/sum/skew/kurtosis/median/slope/last_minus_first/positive_share.
- **参数**：window integer bars or All; slice in [0,1] or None; mask_side high|low; mask_q in (0,1); reducer enum; min_bars
- **空值/异常值**：Pairwise finite x/mask required. If selected slice has <min_bars before mask, or retained bars below reducer minimum, return null; never backfill from outside slice.
- **排序与时间对齐**：Partition strictly stock/day/session; normalized slice uses continuous trading-minute index excluding lunch. Quantile ties use deterministic average-rank then timestamp stable tie handling.
- **三后端实现**：Pandas vectorized groupby; Polars native window/filter/aggregate; DuckDB CUME_DIST/PERCENT_RANK + filtered aggregate. No Python UDF in production.
- **PIT / 防未来**：Daily EOD output may use all completed bars. For intraday decision time, slice end must be <= decision timestamp. No negative lead parameter exists.
- **最低测试**：slice boundaries; lunch exclusion; high/low mask; ties; empty mask; reducer fixtures; partial-day PIT; 3-backend parity
- **解锁方向**：Huatai minute parameterized univariate factors
- **来源**：HTSC_INTRADAY_PARAM_2026
- **字段门槛**：`minute_open|minute_high|minute_low|minute_close|minute_volume|minute_amount|minute_vwap`
- **R43 已预建候选因子数**：48

## 2. `intra_slice_mask_pair_reduce` — P0

- **签名**：`intra_slice_mask_pair_reduce(x, y, mask_field, window="All", slice=None, mask_side="high", mask_q=0.7, y_lag=0, reducer="corr", min_pairs=10)`
- **数学/算法定义**：Apply the same slice+quantile mask, align y lagged by non-negative trading bars within session, then reduce x/y pairs using corr/cov/slope/intercept/r2/distance_corr/euclidean/cosine. y_lag>0 means y is shifted backward so only older y is paired with current x.
- **参数**：same slice/mask args; y_lag>=0 only; reducer enum; min_pairs
- **空值/异常值**：Pairwise finite after lag/mask. Zero variance for corr/slope/r2 => null. Euclidean/cosine require scale option documented.
- **排序与时间对齐**：Lag never crosses lunch/day/security boundary. Stable timestamp alignment after filtering; mask is determined before dropping lag-induced missing y.
- **三后端实现**：Backend-native partitioned LAG plus filtered sufficient statistics; distance-correlation may use shared numeric kernel but identical ordering/output.
- **PIT / 防未来**：Negative y_lag is prohibited in FactorEngine predictive mode even though some research-search templates allow leads. This removes an easy future-leak path.
- **最低测试**：perfect linear pair; zero variance; lag boundary; mask tie; negative-lag rejection; parity
- **解锁方向**：Huatai minute parameterized bivariate factors
- **来源**：HTSC_INTRADAY_PARAM_2026
- **字段门槛**：`minute_open|minute_high|minute_low|minute_close|minute_volume|minute_amount|minute_vwap`
- **R43 已预建候选因子数**：64

## 3. `intra_multiresolution_resample_reduce` — P0

- **签名**：`intra_multiresolution_resample_reduce(x, bar_minutes=10, lookback_days=10, reducer="mean", session_split=True, min_coverage=0.8)`
- **数学/算法定义**：Resample 1-minute x into non-overlapping k-minute bars within each trading session, compute a per-day summary at that resolution, then aggregate over trailing completed days. Designed for 1m/10m/30m multi-resolution feature families.
- **参数**：bar_minutes positive divisor/compatible grid; lookback_days>=1; reducer mean/std/slope/skew/kurt/last_minus_first; session_split; min_coverage
- **空值/异常值**：Incomplete resample buckets may be dropped; day below min_coverage is excluded; insufficient valid days => null.
- **排序与时间对齐**：Buckets restart at morning and afternoon session openings; never create a bar spanning lunch or dates. Trailing-day aggregate chronological.
- **三后端实现**：Pandas resample with session keys; Polars group_by_dynamic per session; DuckDB integer slot bucketing + grouped aggregates. Identical bucket anchor.
- **PIT / 防未来**：Only completed bars/days as of decision timestamp. EOD/next-open default for full-day features.
- **最低测试**：1m->10m known fixture; lunch boundary; missing bars; 10d/30d rolling; parity
- **解锁方向**：Guohai 1m/10m/30m multi-resolution factors
- **来源**：GUOHAI_MULTISOURCE_2026
- **字段门槛**：`minute_open|minute_high|minute_low|minute_close|minute_volume|minute_amount|minute_vwap`
- **R43 已预建候选因子数**：16

## 4. `panel_intraday_ordered_reduce` — P1

- **签名**：`panel_intraday_ordered_reduce(x, bar_minutes=15, first="time", time_reducer="mean", cs_transform="rank", lookback_days=10)`
- **数学/算法定义**：Implement the two non-commuting feature orders highlighted in public multi-source research: time→cross-section means first aggregate each stock intraday then transform cross-sectionally; cross-section→time transforms each timestamp cross-section first then aggregates within stock/day. Return daily scalar.
- **参数**：bar_minutes; first in {time,cross_section}; time_reducer; cs_transform in {rank,zscore,demean}; lookback_days optional
- **空值/异常值**：Cross-section transform requires minimum universe size; time aggregation requires minimum bar coverage. Nulls excluded pairwise with explicit denominator.
- **排序与时间对齐**：Exact timestamp alignment across stocks for cross-section-first mode; resampled bars anchored identically by exchange session.
- **三后端实现**：Polars/DuckDB must avoid materializing N×N objects; group by timestamp for cs transform then stock/day for time reduction. Pandas reference implementation only for testing.
- **PIT / 防未来**：No future timestamps. Cross-section at t uses only contemporaneous stocks; trailing history ends at current completed day.
- **最低测试**：non-commutativity fixture; permutation invariance; universe minimum; missing timestamp; parity
- **解锁方向**：Guohai ts→cs and cs→ts 15m feature families
- **来源**：GUOHAI_MULTISOURCE_2026
- **字段门槛**：`minute_open|minute_high|minute_low|minute_close|minute_volume|minute_amount`
- **R43 已预建候选因子数**：6

## 5. `ts_ewm_std` — P0

- **签名**：`ts_ewm_std(x, window=252, half_life=42, min_periods=126, ddof=0)`
- **数学/算法定义**：Exponentially weighted trailing standard deviation with weights w_age=2^(-age/half_life), truncated to window. Compute weighted mean and weighted second central moment using finite past/current observations.
- **参数**：window>1; half_life>0; min_periods>=2; ddof convention explicit
- **空值/异常值**：If valid observations <min_periods or effective denominator <=eps, return null. Do not zero-fill missing returns.
- **排序与时间对齐**：Per security chronological trailing window including current x; no forward observations.
- **三后端实现**：Implement identical normalized exponential weights. DuckDB can use explicit weighted sums with row-number age; Polars rolling expression/kernel; numerical tolerance specified.
- **PIT / 防未来**：Pure backward-looking trailing operator; current observation allowed for next-session signal.
- **最低测试**：constant series=>0; known weighted vector; missing observations; scale property; parity
- **解锁方向**：CNE5-style DASTD and general weighted volatility
- **来源**：MSCI_CNE5_DESCRIPTORS
- **字段门槛**：`ret|close`
- **R43 已预建候选因子数**：1

## 6. `panel_ewm_beta_ex_self` — P0

- **签名**：`panel_ewm_beta_ex_self(ret, weight, window=252, half_life=63, min_periods=126)`
- **数学/算法定义**：Construct contemporaneous ex-self weighted market return for each stock/date, then estimate exponentially weighted beta of stock return on that ex-self market return over trailing window with intercept. Return beta.
- **参数**：window; half_life; min_periods; weight must be non-negative market-cap-like field
- **空值/异常值**：Dates with insufficient peers or denominator weight<=0 yield null market proxy; beta null if predictor variance<=eps or history insufficient.
- **排序与时间对齐**：Market return excludes target stock exactly. Weight at date t must be PIT/current. Regression window chronological per stock.
- **三后端实现**：Efficient O(NT): compute total weighted sum and subtract own numerator/weight; then weighted covariance/variance. Same formulas all backends.
- **PIT / 防未来**：No future returns; current date allowed only for EOD/next-open signal. No target self-contamination.
- **最低测试**：self-exclusion two-stock fixture; known beta; weight-zero; missing date; parity
- **解锁方向**：CNE5-style beta / residual-volatility support without external market series
- **来源**：MSCI_CNE5_DESCRIPTORS
- **字段门槛**：`ret|market_cap|free_market_cap`
- **R43 已预建候选因子数**：1

## 7. `panel_cmra_ex_self` — P1

- **签名**：`panel_cmra_ex_self(ret, weight, months=12, days_per_month=21, min_months=6)`
- **数学/算法定义**：Build ex-self market return and stock-minus-market excess return. Compound/log-sum each fixed 21-trading-day block over trailing months; form cumulative sequence Z(T), and return max(Z)-min(Z) across months.
- **参数**：months>=2; days_per_month>=5; min_months>=2
- **空值/异常值**：Months with inadequate valid observations omitted only under documented coverage threshold; insufficient valid months=>null.
- **排序与时间对齐**：Month here is fixed trading-day block as in Barra descriptor methodology, not natural calendar. Blocks are backward from evaluation date and do not look forward.
- **三后端实现**：Use cumulative log1p excess returns; identical fixed-block indexing across backends.
- **PIT / 防未来**：Backward-only ex-self market. No risk-free field is invented; this is explicitly an A-share ex-self adaptation, not exact vendor replication.
- **最低测试**：constant excess=>0; monotone path; missing block; self-exclusion; parity
- **解锁方向**：CNE5 CMRA-style cumulative range adaptation
- **来源**：MSCI_CNE5_DESCRIPTORS
- **字段门槛**：`ret|market_cap|free_market_cap`
- **R43 已预建候选因子数**：1

## 8. `intra_idiosyncratic_semivariance_balance_ex_self` — P0

- **签名**：`intra_idiosyncratic_semivariance_balance_ex_self(close, weight, min_bars=30)`
- **数学/算法定义**：At each minute timestamp construct ex-self weighted market minute return, regress/estimate stock market loading using only trailing matured history or same-day pre-specified historical beta, form idiosyncratic minute returns, then compute (sum positive residual^2 - sum negative residual^2)/(sum positive^2+sum negative^2).
- **参数**：min_bars; beta_source trailing_days default 20; weight market-cap-like; eps
- **空值/异常值**：Insufficient peer universe/beta history/bars=>null. Zero total idiosyncratic variation=>null.
- **排序与时间对齐**：Timestamp exact, ex-self market; beta fitted from strictly prior completed days by default, then frozen during current day.
- **三后端实现**：O(NT) ex-self market construction plus per-stock residual aggregation; no per-stock N×N loop.
- **PIT / 防未来**：Current-day residuals use beta estimated through prior day, avoiding same-day fitted-residual lookahead. EOD/next-open signal.
- **最低测试**：market-clone stock=>near-zero idio var; positive/negative residual fixtures; self exclusion; frozen-beta test; parity
- **解锁方向**：good/bad idiosyncratic volatility factors
- **来源**：GOOD_BAD_IDIO_VOL_2024
- **字段门槛**：`minute_close|market_cap|free_market_cap`
- **R43 已预建候选因子数**：8

## 9. `panel_similarity_crowding_score` — P2

- **签名**：`panel_similarity_crowding_score(*features, k=20, window=20, metric="robust_euclidean", volume_field=turnover_ratio)`
- **数学/算法定义**：Standardize current cross-section robustly, find k nearest peers, and score crowding as high local density combined with synchronized recent turnover/return activity. Return continuous stock-level score; not a proprietary MSCI formula.
- **参数**：k; metric; window; feature scaling; activity weighting; min_universe
- **空值/异常值**：Rows missing too many features=>null. Require min universe > k. Robust scale zero=>drop that dimension.
- **排序与时间对齐**：Current-date features only plus trailing activity ending today. Peer search excludes self.
- **三后端实现**：Use deterministic approximate/exact KNN with fixed tie break by security id; Polars/DuckDB may call common compiled kernel but not Pandas fallback.
- **PIT / 防未来**：Contemporaneous cross-section acceptable for next-session signal; trailing history only.
- **最低测试**：duplicate points; isolated point; permutation invariance; self exclusion; tie determinism; parity
- **解锁方向**：generic crowding proxies inspired by next-generation China risk-model directions
- **来源**：MSCI_CHINA_NEXTGEN_2025
- **字段门槛**：`ret|close|turnover_ratio|market_cap|roe|pb_ratio`
- **R43 已预建候选因子数**：3

## 10. `panel_cluster_risk_score` — P2

- **签名**：`panel_cluster_risk_score(*features, k=20, return_window=20, method="knn_cluster")`
- **数学/算法定义**：Form deterministic similarity clusters from current characteristics, compute each stock cluster-level recent return/volatility concentration and own deviation, and return a continuous cluster-risk/crowding exposure. Explicit public proxy, not vendor replication.
- **参数**：k; return_window; method; min_cluster; standardize
- **空值/异常值**：Small/degenerate clusters=>null; missing feature dimensions handled by predeclared minimum coverage, never imputed from future data.
- **排序与时间对齐**：Cluster membership at date t from date-t cross-section; return statistics use trailing dates through t. Self can be excluded from cluster aggregate via option default True.
- **三后端实现**：Deterministic graph components/community approximation with common kernel; stable security-id tie break.
- **PIT / 防未来**：No forward returns or future cluster membership.
- **最低测试**：two-cluster synthetic; self exclusion; cluster permutation; singleton; parity
- **解锁方向**：similar-company cluster/crowding risk factors
- **来源**：MSCI_CHINA_NEXTGEN_2025
- **字段门槛**：`ret|close|turnover_ratio|market_cap|roe|pb_ratio`
- **R43 已预建候选因子数**：1

## 11. `intra_functional_beta_profile_ex_self` — P1

- **签名**：`intra_functional_beta_profile_ex_self(minute_close, weight_field=None, lookback_days=60, slot_key="minute", min_days=30, reducer="slope")`
- **数学/算法定义**：Estimate an ex-self market intraday return curve for each clock slot, then estimate each stock beta separately by slot over trailing completed sessions. Reduce the historical beta curve across slots to one daily scalar such as mean, slope, std, open_close_gap or curvature.
- **参数**：lookback_days>=20; slot_key=minute|bar_index; min_days; reducer in {mean,slope,std,open_close_gap,curvature}; optional non-negative market-cap-like weights.
- **空值/异常值**：Require finite stock and ex-self market returns per slot; slots below min_days are null; no interpolation across lunch/session boundaries by default.
- **排序与时间对齐**：Create stable session/slot IDs. For decision date t, beta estimation uses completed sessions <=t; if current session is included, output is EOD-only. Reducer operates on the fixed slot-beta curve.
- **三后端实现**：Pandas reference groupby implementation; Polars native grouped rolling covariance/variance by stock+slot; DuckDB window aggregates. Identical ex-self market formula, session calendar and reducer formulas.
- **PIT / 防未来**：No future minute slots. Default availability EOD and next-session use. For intraday use, require explicit partial-session mode that only includes observed slots and prior-session beta estimates.
- **最低测试**：Synthetic constant-beta panel; known morning-to-afternoon beta slope; zero market variance; missing slots; one-stock edge; ex-self test; backend parity; no future-slot test.
- **解锁方向**：Functional-CAPM intraday beta level/slope/dispersion and liquidity/volatility interactions.
- **来源**：FUNCTIONAL_CAPM_2025
- **字段门槛**：`minute_close|market_cap|free_market_cap`
- **R43 已预建候选因子数**：8

---

## 附录来源：`R44_OPERATOR_ENGINEERING_SPECS.md`

> 以下是 R41–R47 原始工程规格，作为候选语义档案保留。**若与本文件前文的 current-main 规则、字段硬约束、backend policy 或 preflight 结果冲突，以前文为准。**

# R44 New Operator Engineering Specifications

These specifications are intentionally self-contained so an implementation agent can add the operator without reopening the source-research phase. Promotion of dependent factors remains gated on PIT and cross-backend parity tests.

## `fiscal_cost_stickiness_score`
**Signature:** `fiscal_cost_stickiness_score(cost, sales, periods=8, basis="quarter", output="stickiness|up_elasticity|down_elasticity|asymmetry", min_periods=6, eps=1e-12)`

**Semantics.** On PIT-deduplicated fiscal observations, compute Δlog(cost) and Δlog(sales). Estimate separate cost elasticities for sales-increase and sales-decrease observations; output up/down elasticity and their asymmetry. A sticky-cost convention is up_elasticity - down_elasticity, so larger values mean costs decrease less when sales fall than they rise when sales grow.

**Parameters.** periods>=6; basis quarter|annual; output enum; min_periods>=4; eps>0; optional robust=true using Huber weighting.

**Null policy.** Null if insufficient valid up/down observations or nonpositive cost/sales after eps policy; do not silently coerce missing reports to zero.

**Ordering/alignment.** Per stock, order fiscal vintages by fiscal period and availability timestamp; deduplicate to the latest vintage available at each decision time; regress only historical fiscal endpoints.

**Backend requirements.** Polars: grouped state/window over deduplicated fiscal panel; Pandas parity; DuckDB may use grouped sufficient statistics or a registered deterministic aggregate. Outputs Float64 with identical min-period behavior.

**PIT/no-lookahead.** Every fiscal observation entering the regression must have availability_time<=decision_time. Restatements create new vintages and never rewrite past decision rows.

**Minimum tests.** Synthetic symmetric elasticity=0 asymmetry; known sticky/anti-sticky paths; missing quarter; negative/zero inputs; restatement vintage; cross-backend tolerance <=1e-10.

**Unlocks.** Cost-growth underreaction, SG&A stickiness, operating leverage conditioned on asymmetric cost adjustment.

## `fundamental_cash_flow_duration`
**Signature:** `fundamental_cash_flow_duration(earnings, book_equity, market_cap, roe=None, growth=None, horizon=10, terminal_growth=0.0, discount_floor=1e-6, output="duration|near_cash_share|terminal_share")`

**Semantics.** Estimate firm-level cash-flow duration as the present-value weighted average timing of expected equity cash flows using a transparent clean-surplus/earnings persistence approximation. Forecast finite-horizon cash flows from only as-of accounting inputs; terminal value is explicit. Return duration plus near-term/terminal PV shares. This is a research-inspired implementation target and must document the exact forecasting equation used; do not label it an exact Weber replication unless the published specification is matched.

**Parameters.** horizon 3..20; terminal_growth; optional roe/growth; output enum; forecasting_mode="constant_roe|mean_revert_roe"; mean_reversion parameter when selected.

**Null policy.** Null for nonpositive market cap/book equity when required or infeasible discount/terminal denominator; no clipping that changes economics without an explicit flag.

**Ordering/alignment.** Accounting inputs are PIT snapshots; market cap is contemporaneous at decision time. Forecast recursion runs forward only mathematically from known inputs, never reading realized future fields.

**Backend requirements.** Shared pure function for forecast recursion; Polars map-groups or vectorized closed form where possible; Pandas/DuckDB semantic parity.

**PIT/no-lookahead.** No future realized earnings/cash flows. All forecast parameters fixed or estimated from strictly historical cross-sections with an explicit lag.

**Minimum tests.** Closed-form constant-ROE cases; zero/negative growth; terminal-value dominance; PIT restatement; numerical stability; backend parity.

**Unlocks.** Cash-flow duration, near-term cash-flow weight, duration×value, duration×profitability.

## `intra_round_price_clustering_share`
**Signature:** `intra_round_price_clustering_share(price, lattice="1.0|0.5|0.1|adaptive", tolerance_ticks=0.25, window="All", output="share|excess_share|run_length", min_bars=30)`

**Semantics.** For each stock-day, map each intraday price to the nearest round-price lattice. Mark a bar clustered when distance<=tolerance_ticks*tick_size. share is clustered-bar fraction; excess_share subtracts the expected share under the day’s empirical price range/tick grid; run_length summarizes persistence near lattice points. adaptive evaluates approved price-level-dependent lattices.

**Parameters.** lattice numeric/adaptive; tolerance_ticks>=0; window session selector; output enum; min_bars.

**Null policy.** Null when tick size/range cannot be determined or fewer than min_bars valid prices; duplicate timestamps resolved by existing intraday ordering policy.

**Ordering/alignment.** Use exchange-session timestamps and actual tick-size convention valid for that security/date. Aggregate only bars available by day end for next-session factors.

**Backend requirements.** Polars grouped intraday reduce; Pandas parity; DuckDB list/window or pre-aggregation UDF. Must share lattice rounding convention exactly.

**PIT/no-lookahead.** No future bars relative to the declared decision time. For EOD daily factor, full same-day intraday path is allowed only for next-session execution.

**Minimum tests.** Exact lattice examples; boundary tolerance; changing price range; suspended day; one-price limit day; adaptive lattice; backend parity.

**Unlocks.** Intraday price clustering, round-number anchoring persistence, clustering×volatility/turnover.

## `intra_round_price_barrier_response`
**Signature:** `intra_round_price_barrier_response(price, lattice="1.0|0.5|0.1|adaptive", lookback_days=20, tolerance_ticks=1.0, output="cross_rate|bounce_rate|magnet_strength|asymmetry")`

**Semantics.** Identify approaches to round-price lattice points and classify whether price crosses, bounces, or accelerates toward the barrier. Compute historical event rates using only completed events. magnet_strength compares approach speed/probability to matched non-round control levels; asymmetry separates upward versus downward approaches.

**Parameters.** lattice; lookback_days>=5; tolerance_ticks; output enum; min_events>=5; control_mode="within_day_matched".

**Null policy.** Null if too few events; exclude price-limit-pinned intervals by default with optional include_limits flag.

**Ordering/alignment.** Events are ordered by intraday timestamps within each day and accumulated over prior completed sessions only.

**Backend requirements.** State-machine implementation shared across backends; Polars preferred; DuckDB recursive/list UDF acceptable only with deterministic ordering and parity tests.

**PIT/no-lookahead.** At decision date t, response statistics use events ending no later than t; if used intraday, only events whose outcome is already observed may enter.

**Minimum tests.** Synthetic crossing/bounce sequences; upward/downward symmetry; no-event days; limit-pinned day; tick-size changes; backend parity.

**Unlocks.** Round-number natural resistance, magnet/bounce behavior, barrier-conditioned trend/reversal factors.

## `intra_session_segment_reduce`
**Signature:** `intra_session_segment_reduce(x, segment="open30|morning|pre_lunch30|post_lunch30|afternoon|close30", reducer="sum|mean|std|last_minus_first|positive_share", min_bars=5)`

**Semantics.** Exchange-calendar-aware reduce over named A-share session segments. Lunch break is never treated as a continuous bar interval. Segment boundaries use official trading calendar/session templates for each date.

**Parameters.** segment enum or explicit normalized session interval; reducer enum; min_bars; include_auction=false by default.

**Null policy.** Null when target segment has fewer than min_bars; suspended/absent bars remain missing rather than zero.

**Ordering/alignment.** Partition by stock/date then session-aware timestamp. Morning and afternoon are distinct sessions around the lunch break.

**Backend requirements.** Polars grouped slice/reduce; Pandas parity; DuckDB timestamp predicates with calendar table.

**PIT/no-lookahead.** EOD outputs use only current-day completed segment and are next-session eligible; intraday use cannot reference a later segment.

**Minimum tests.** Full day, half day, missing post-lunch, suspension, auction exclusion, DST-not-applicable A-share calendar, backend parity.

**Unlocks.** Opening/closing/lunch effects, session-specific return/volume/volatility, T+1 intraday delayed-feedback localization.

## `intra_session_boundary_jump`
**Signature:** `intra_session_boundary_jump(price, boundary="open|lunch_restart|close", pre_bars=5, post_bars=5, output="gap|normalized_gap|recovery|volume_jump")`

**Semantics.** Measure price/volume discontinuity and short-horizon recovery around a named exchange-session boundary. For lunch_restart, compare last valid morning bar with first valid afternoon bar; recovery uses only subsequent observed post-boundary bars.

**Parameters.** boundary enum; pre_bars/post_bars>=1; output enum; normalization="pre_price|daily_range|pre_vol".

**Null policy.** Null if required pre/post observations are absent; do not bridge suspensions or missing sessions.

**Ordering/alignment.** Strict chronological pairing around official session boundary within stock/day.

**Backend requirements.** Shared boundary locator; vectorized grouped joins in Polars; equivalent Pandas/DuckDB implementation.

**PIT/no-lookahead.** Recovery output is EOD/after-post-window only; never expose future post-boundary bars to earlier decision times.

**Minimum tests.** Known gap/recovery paths; no afternoon session; flat prices; limit lock; backend parity.

**Unlocks.** Lunch-restart repricing, opening-lock-in discount decay, close-session discontinuity.

## `trading_calendar_mask`
**Signature:** `trading_calendar_mask(feature="day_of_week|month_end|quarter_end|pre_holiday|post_holiday", value=None, calendar="SSE_SZSE")`

**Semantics.** Return a boolean/numeric calendar state from the exchange trading calendar attached to the observation timestamp. day_of_week uses trading date, not row count; pre/post holiday refers to adjacent exchange sessions, not civil days.

**Parameters.** feature enum; optional value for day_of_week; calendar identifier; holiday_window=1 by default.

**Null policy.** Null only when timestamp/calendar mapping is unavailable; otherwise deterministic.

**Ordering/alignment.** Calendar metadata join before expression evaluation; identical date semantics across backends/time zones.

**Backend requirements.** Polars/Pandas join against the same exchange-calendar table; DuckDB uses the same persisted calendar relation; output dtype and holiday adjacency must be identical.

**PIT/no-lookahead.** Calendar feature itself is known ex ante. It may condition only information otherwise available at the decision time.

**Minimum tests.** Known 2025/2026 holidays; Thursday/Friday mapping; month/quarter end; timezone; backend parity.

**Unlocks.** T+1 day-of-week asymmetry, holiday/session seasonality and calendar-conditioned factor strength.

## `suspension_restart_response`
**Signature:** `suspension_restart_response(open, close, pre_close, is_suspend, lookback_sessions=20, output="restart_gap|restart_intraday|episode_length|historical_recovery")`

**Semantics.** Detect the first active session after a suspension episode. Report restart gap, restart intraday return, prior suspension length, or historical recovery conditional on completed previous restart events.

**Parameters.** lookback_sessions>=5 for historical_recovery; output enum; min_events=3.

**Null policy.** Null outside restart days for event outputs; episode_length may be zero outside suspensions/restarts; historical output null if too few past events.

**Ordering/alignment.** Per stock order sessions; an episode is one or more consecutive suspended trading sessions followed by an active session.

**Backend requirements.** Polars run-length/event state; Pandas parity; DuckDB islands-and-gaps plus lag/window.

**PIT/no-lookahead.** Current restart gap/intraday is EOD next-session eligible; historical_recovery uses only earlier completed restart episodes.

**Minimum tests.** 1/5/20-day suspension episodes; missing calendar sessions; delisting edge; restart at price limit; backend parity.

**Unlocks.** Post-suspension repricing, information accumulation during halt, restart liquidity stress.

## `intra_limit_pre_hit_pressure_profile`
**Signature:** `intra_limit_pre_hit_pressure_profile(price, volume, high_limit, low_limit, side="up|down", pre_window=30, output="return_accel|volume_accel|distance_decay|path_efficiency", require_hit=True)`

**Semantics.** For a completed limit-hit event, summarize the path in the bars immediately preceding first hit: return acceleration, volume acceleration, distance-to-limit decay, and path efficiency. When require_hit=False, evaluate nearest approach by EOD as a state variable.

**Parameters.** side; pre_window>=5; output enum; require_hit; min_bars.

**Null policy.** Null when no valid hit under require_hit=True or insufficient pre-hit bars; one-price open-at-limit events have explicit zero-length path flag and null acceleration.

**Ordering/alignment.** First-hit timestamp determined within current stock-day; only bars at or before first hit enter pre-hit profile.

**Backend requirements.** Polars grouped event locator; Pandas parity; DuckDB windows with first-hit timestamp.

**PIT/no-lookahead.** Completed EOD factor is next-session eligible. Intraday use allowed only after hit and may not inspect bars after decision time.

**Minimum tests.** Gradual magnet path; sudden jump-to-limit; no hit; open-at-limit; reopen/re-hit; up/down parity; backend parity.

**Unlocks.** Pre-hit dynamics, cooling-off versus magnet state, next-day continuation/reversal conditioning.

---

## 附录来源：`R45_OPERATOR_ENGINEERING_SPECS.md`

> 以下是 R41–R47 原始工程规格，作为候选语义档案保留。**若与本文件前文的 current-main 规则、字段硬约束、backend policy 或 preflight 结果冲突，以前文为准。**

# R45 New Operator Engineering Specifications

All proposed operators must return one scalar per stock/day (unless a documented intermediate vector is internal only). Intraday paths use completed A-share sessions; outputs intended for stock selection are earliest usable next trading session unless an explicit timestamp-safe intraday mode is separately implemented. Terms such as probe, absorption, chip, leader and supply are observable/model-implied proxies, not claims about trader identity.

## `intra_impulse_event_detector`

**Signature:** `intra_impulse_event_detector(price, volume=None, amount=None, event='up|down|both', threshold='robust_z|vol_scaled|quantile', z=3.0, min_bars=1, merge_gap=2, output='count|strength|max_strength|first_time|last_time|duration')`

**Semantics:** Detect contiguous intraday price impulses after volatility normalization. Compute one-bar/log returns, robust or rolling scale, mark threshold exceedances, merge events separated by <= merge_gap, and summarize event magnitude/duration/timing. Volume/amount can weight strength but cannot determine event direction by themselves.

**Parameters:** threshold method; z>0; min_bars>=1; merge_gap>=0; output selector.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Shock/probe event families and event-conditioned response operators.

**Priority/source:** P0; R45_BEHAVIORAL_MICROSTRUCTURE

## `intra_post_impulse_response`

**Signature:** `intra_post_impulse_response(price, volume, amount, direction='up|down', threshold='robust_z', z=3.0, horizon=30, anchor='event_end', output='retention|giveback|max_drawdown|vol_ratio|volume_ratio|amount_ratio|vwap_hold|low_slope|recovery_half_life')`

**Semantics:** For each detected impulse, anchor at event end and measure the subsequent supply/demand response over a fixed forward-within-session horizon. Aggregate multiple historical-in-session events without using bars after EOD; outputs quantify retention, giveback, volatility compression, participation decay and recovery.

**Parameters:** z; horizon minutes/bars; anchor; output; aggregation=max_strength_weighted|last_event|mean.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Post-shock retention, giveback, volatility compression, volume dry-up, VWAP holding.

**Priority/source:** P0; R45_BEHAVIORAL_MICROSTRUCTURE

## `intra_probe_outcome_score`

**Signature:** `intra_probe_outcome_score(price, volume, amount, direction='up|down', z=3.0, probe_horizon=10, response_horizon=30, weights=None, output='score|failure|holding|second_push')`

**Semantics:** Composite observable proxy for a short price probe followed by natural market response. Standardize impulse strength, post-event giveback, VWAP holding, downside excursion, volume/amount participation and second push. Do not label trader identity; score means favorable/unfavorable observable supply response.

**Parameters:** probe/response horizons; z; optional fixed weights summing to one; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Probe success/failure and second-push readiness factors.

**Priority/source:** P0; R45_BEHAVIORAL_MICROSTRUCTURE

## `intra_supply_absorption_score`

**Signature:** `intra_supply_absorption_score(price, volume, amount, event='up_impulse|down_impulse|all', horizon=30, price_scale='atr|realized_vol', output='absorption|price_per_amount|volume_no_drop|downside_resilience')`

**Semantics:** Measure how much trading activity is absorbed with limited adverse price movement after an event. Core statistic compares signed/absolute price displacement to volume or amount and downside excursion; high absorption means substantial activity with small adverse move, not knowledge of buy/sell identity.

**Parameters:** event; horizon; scaling; robust winsorization; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** High-volume-no-drop, downside resilience and absorption efficiency.

**Priority/source:** P0; R45_BEHAVIORAL_MICROSTRUCTURE

## `intra_consolidation_quality`

**Signature:** `intra_consolidation_quality(price, volume, amount, trigger='impulse|breakout|auto', trigger_z=3.0, horizon=30, output='tightness|level|vol_compression|volume_dryup|rising_floor|breakout_readiness')`

**Semantics:** After a trigger, characterize consolidation location, range tightness, realized-volatility compression, volume/amount decay, low-price slope and close position. Trigger detection and response window are fixed ex ante.

**Parameters:** trigger; z; horizon; output; optional normalization by pre-trigger scale.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** High-tight flag, rising-floor consolidation and re-breakout readiness.

**Priority/source:** P0; R45_BEHAVIORAL_MICROSTRUCTURE

## `intra_response_curve_features`

**Signature:** `intra_response_curve_features(price, activity, trigger='impulse', horizon=30, curve='price|drawdown|volatility|activity', output='slope|curvature|auc|half_life|monotonicity|change_count')`

**Semantics:** Construct an event-time response curve following detected events and return geometrical summaries. Curvature uses a fixed polynomial/local finite-difference convention; half-life is first time the response decays to 50% of peak displacement.

**Parameters:** trigger; horizon; curve; output; event aggregation.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Shock response curvature, area and decay-profile factors.

**Priority/source:** P1; R45_BEHAVIORAL_MICROSTRUCTURE

## `intra_volume_at_price_profile`

**Signature:** `intra_volume_at_price_profile(price, volume, amount=None, bins=64, weighting='volume|amount|time', price_basis='close|vwap|ohlc_typical', normalize=True, output='entropy|skew|kurtosis|poc_price|value_area_width|tail_mass|dip|concentration')`

**Semantics:** Project intraday bars from time to price bins and construct a normalized price-acceptance distribution. POC is maximum mass bin; value area is the minimal/contiguous convention documented to contain target mass (default 70%); entropy/moments refer to the normalized profile. Bar-based profile is an approximation, not tick-level inventory.

**Parameters:** bins>=8; weighting; price basis; target value-area mass; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** TPO/Volume Profile shape factors: POC, entropy, value area, tails, multimodality.

**Priority/source:** P0; TPO_ENTROPY_2026

## `intra_volume_profile_peak_geometry`

**Signature:** `intra_volume_profile_peak_geometry(price, volume, amount=None, bins=64, smooth=2, min_prominence=0.05, output='peak_count|top_peak_mass|second_peak_mass|peak_ratio|peak_distance|top_peak_width|nearest_peak_distance|valley_depth')`

**Semantics:** On a normalized volume-at-price profile, smooth with a fixed symmetric kernel, identify local maxima using prominence, and summarize peak/valley geometry. Distances are normalized by daily price range or current price as documented.

**Parameters:** bins; smooth bandwidth; prominence; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Chip/acceptance peak geometry and valley/vacuum directions.

**Priority/source:** P0; TPO_ENTROPY_2026

## `intra_volume_profile_supply_structure`

**Signature:** `intra_volume_profile_supply_structure(price, volume, current_price=None, bins=64, decay=20.0, output='overhead_mass|near_overhead_mass|under_price_mass|supply_vacuum|nearest_upper_peak|nearest_lower_peak|distance_weighted_overhang')`

**Semantics:** Interpret the bar-based volume-at-price profile relative to current/end price. Sum mass above/below price and distance-weight overhead mass using exp(-decay*relative_distance). Supply-vacuum is low profile mass between current price and next material upper peak. This is historical price-acceptance supply proxy, not true shareholder positions.

**Parameters:** bins; decay>0; near band; material peak threshold; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Overhead supply, supply vacuum and profit-taking proxy factors.

**Priority/source:** P0; R45_MARKET_PROFILE

## `turnover_chip_distribution`

**Signature:** `turnover_chip_distribution(close, turnover, window=120, decay='turnover_survival', bins=128, output='entropy|mode|mean|std|skew|profit_mass|loss_mass|near_mass|upper_mass')`

**Semantics:** Reconstruct a model-implied historical cost distribution from daily turnover. Each day contributes mass at its price/cost proxy and prior mass survives according to (1-turnover); normalize surviving masses. It is an inferred cost-basis model, not account-level holdings.

**Parameters:** window; bins; survival rule; output; turnover clipping [0,1].

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Longer-horizon chip distribution beyond current scalar cost operators.

**Priority/source:** P0; R45_CHIP_PROXY

## `turnover_chip_distribution_transport`

**Signature:** `turnover_chip_distribution_transport(close, turnover, window=120, lag=1, bins=128, output='wasserstein|jsd|centroid_shift|mode_shift|upper_mass_shift')`

**Semantics:** Build PIT chip distributions at t and t-lag with the same grid, then quantify migration using 1D Wasserstein distance, Jensen-Shannon divergence and location/mass shifts.

**Parameters:** window; lag>=1; bins; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Chip migration velocity, distribution restructuring and cost-basis turnover.

**Priority/source:** P0; R45_CHIP_PROXY

## `turnover_chip_age_cost_surface`

**Signature:** `turnover_chip_age_cost_surface(close, turnover, window=120, price_bins=64, age_bins=16, output='surface_entropy|age_price_mi|young_profit_mass|old_overhang_mass|young_overhang_mass|cost_age_slope')`

**Semantics:** Maintain surviving turnover-implied mass on a 2D age x cost grid. Age advances one trading day; new turnover mass enters age zero; existing mass decays by turnover survival. Outputs summarize entropy, mutual information, age-conditioned profitable/overhead mass and cost-age gradient.

**Parameters:** window; price_bins; age_bins; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Two-dimensional chip-age/cost mechanisms and young-vs-old supply pressure.

**Priority/source:** P1; R45_CHIP_PROXY

## `intra_log_signature_features`

**Signature:** `intra_log_signature_features(x, y=None, z=None, depth=3, lead_lag=True, normalize='zscore|range', output='norm|area|level2_norm|level3_norm|coordinate', coordinate=None)`

**Semantics:** Construct a time-augmented/lead-lag path from one to three intraday channels, normalize using same-day past-complete path scale, compute truncated log-signature up to depth, and return scalar coordinates/norms. Lead-lag lift is explicitly defined to retain quadratic variation information.

**Parameters:** depth 2-4; lead_lag; normalization; output/coordinate.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Intraday rough-path geometry and multichannel price-volume path interactions.

**Priority/source:** P1; PATH_SIGNATURE_A_SHARE_2026

## `intra_functional_pca_shape`

**Signature:** `intra_functional_pca_shape(x, history_days=60, grid=48, n_components=5, component=1, output='score|reconstruction_error|subspace_distance')`

**Semantics:** Resample each completed session to a fixed intraday grid, standardize using past-session statistics, fit FPCA/PCA basis only on prior sessions, project current path, and return score or residual distance.

**Parameters:** history_days; grid; n_components; component; output; rolling/expanding basis.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Functional data path-shape coordinates and deformation factors.

**Priority/source:** P1; FUNCTIONAL_MOTIF_2023

## `intra_functional_motif_score`

**Signature:** `intra_functional_motif_score(x, history_days=120, grid=48, k=8, alignment='dtw|local', output='nearest_distance|motif_frequency|motif_age|cluster_distance|forecast_dispersion')`

**Semantics:** Resample completed sessions, compare to strictly prior sessions using functional distance with optional local/DTW alignment, form motif neighborhoods/clusters, and summarize similarity, frequency, age and historical post-motif dispersion without using future outcomes in the feature itself.

**Parameters:** history_days; grid; k; alignment; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Functional motif recurrence and path analog factors.

**Priority/source:** P1; FUNCTIONAL_MOTIF_2023

## `intra_shapelet_match`

**Signature:** `intra_shapelet_match(x, model_id, output='best_distance|best_similarity|shapelet_id|margin')`

**Semantics:** Match current normalized path/subpaths against a frozen library of discriminative or unsupervised shapelets stored as a versioned model artifact. If shapelets are learned with labels, model_id must refer to a walk-forward training artifact ending before decision date.

**Parameters:** model_id; normalization in artifact; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Learned price-pattern/shapelet factors without hand-coded chart patterns.

**Priority/source:** P2; SHAPELET_FINANCE_2025

## `intra_rqa_features`

**Signature:** `intra_rqa_features(x, embedding_dim=3, delay=1, recurrence_rate_target=0.05, min_diag=2, min_vert=2, output='rr|det|lam|trapping|divergence|diag_entropy')`

**Semantics:** Time-delay embed the intraday path, choose recurrence threshold to hit a fixed recurrence-rate target (or explicit epsilon), build recurrence matrix excluding main diagonal, and compute standard RQA statistics.

**Parameters:** embedding_dim; delay; RR target; line minima; Theiler window; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Intraday recurrence stability, determinism, laminarity and trapping factors.

**Priority/source:** P1; RQA_INTRADAY_2023

## `intra_persistent_homology_features`

**Signature:** `intra_persistent_homology_features(x, embedding_dim=3, delay=1, maxdim=1, output='entropy|total_persistence|h0_count|h1_count|max_lifetime|birth_dispersion')`

**Semantics:** Takens-embed the intraday series, compute Vietoris-Rips persistent homology on the completed-session point cloud with deterministic filtration cap, and summarize finite barcodes.

**Parameters:** embedding_dim; delay; maxdim<=1 default; filtration cap; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Intraday topological state shape and persistence-complexity factors.

**Priority/source:** P1; TDA_CHANGEPOINT_2025

## `intra_topological_anomaly_score`

**Signature:** `intra_topological_anomaly_score(x, history_days=60, embedding_dim=3, delay=1, metric='wasserstein|bottleneck', output='distance|zscore|rank')`

**Semantics:** Compute current-session persistence diagram and compare with strictly prior session diagrams using diagram Wasserstein/bottleneck distance; standardize using historical distances only.

**Parameters:** history_days; embedding; diagram metric; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Topological regime-change / anomaly factors.

**Priority/source:** P1; TDA_CHANGEPOINT_2025

## `intra_optimal_transport_profile_shift`

**Signature:** `intra_optimal_transport_profile_shift(x, history_days=20, grid=48, representation='time_mass|price_mass|joint_time_price', metric='wasserstein1|sinkhorn', output='distance|signed_centroid|transport_cost')`

**Semantics:** Map current and historical sessions to probability measures over time, price or joint time-price grid and compute 1D Wasserstein or entropy-regularized Sinkhorn transport. Historical reference is past sessions only.

**Parameters:** history_days; grid; representation; metric; Sinkhorn epsilon; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Timing/price-acceptance distribution migration beyond scalar JSD.

**Priority/source:** P1; R45_OPTIMAL_TRANSPORT

## `intra_dmd_koopman_features`

**Signature:** `intra_dmd_koopman_features(x, grid=48, delay_dim=6, rank=4, output='growth|frequency|mode_concentration|reconstruction_error|spectral_radius')`

**Semantics:** Resample completed intraday path, build Hankel delay coordinates, estimate truncated DMD/Koopman linear operator by SVD, and summarize eigenvalue growth/frequency, modal concentration and residual.

**Parameters:** grid; delay_dim; rank; eigenvalue stability convention; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Intraday dynamic-mode/Koopman state factors.

**Priority/source:** P1; DMD_FINANCE_PUBLIC

## `intra_kalman_latent_price`

**Signature:** `intra_kalman_latent_price(price, model='local_level|local_linear_trend', q=None, r=None, estimation='past_em|robust_fixed', history_days=60, output='latent_close|trend|noise_var|innovation_z|filter_gap|forecast_return')`

**Semantics:** State-space decomposition of observed intraday price into latent efficient-price state and microstructure noise. Hyperparameters are fixed or estimated from prior sessions only. forecast_return can use filtered state dynamics but never realized future price.

**Parameters:** model; q/r; estimation; history_days; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** A-share Kalman smoothing, latent price, noise and innovation daily factors.

**Priority/source:** P0; KALMAN_A_SHARE_2025

## `intra_state_space_volume_components`

**Signature:** `intra_state_space_volume_components(volume, amount=None, history_days=60, model='local_level|dynamic_factor', output='latent_activity|innovation|persistent_share|transitory_share|signal_to_noise')`

**Semantics:** Decompose intraday trading activity into persistent/information-like state and transitory/noise-like innovations under a clearly specified linear Gaussian state-space model. Names describe statistical components, not causal informed trading.

**Parameters:** history_days; model; output; fixed/rolling parameter estimation.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Persistent vs transitory volume components and signal-to-noise factors.

**Priority/source:** P1; STATE_SPACE_VOLUME_INFO_2018

## `intra_hmm_state_features`

**Signature:** `intra_hmm_state_features(x, y=None, states=3, history_days=120, emission='gaussian|student', output='state_prob|entropy|persistence|transition_surprise|max_prob', state=None)`

**Semantics:** Fit an HMM on strictly prior completed sessions/bars or use a frozen artifact, then filter current session sequentially. Return end-of-session state probabilities, entropy, transition surprise and persistence. No smoothing with future sessions.

**Parameters:** states; history; emission; output; state selector; walk-forward fit cadence.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Latent intraday volatility/liquidity regime factors.

**Priority/source:** P1; HMM_REGIME_PUBLIC

## `intra_hsmm_duration_features`

**Signature:** `intra_hsmm_duration_features(x, states=3, history_days=120, duration='poisson|negative_binomial', output='state_prob|expected_remaining_duration|duration_surprise|switch_hazard', state=None)`

**Semantics:** Hidden semi-Markov extension with explicit state-duration distribution. Filter current path sequentially and summarize remaining duration/switch hazard of current latent regime.

**Parameters:** states; history; duration law; output; fit cadence.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Regime-duration and state-exhaustion factors.

**Priority/source:** P2; R45_COMPLEX_DIRECTION_RESEARCH

## `intra_change_point_sequence_features`

**Signature:** `intra_change_point_sequence_features(x, method='pelt|bocpd|cusum', penalty='bic', min_segment=5, output='count|last_age|mean_spacing|max_shift|direction_balance|segment_entropy')`

**Semantics:** Detect structural breaks in level/mean/variance of normalized intraday path with a deterministic offline-on-completed-session algorithm, or sequential BOCPD. Summarize number, spacing, last-age, magnitudes and direction.

**Parameters:** method; penalty/hazard; min_segment; target mean|variance|both; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Impulse-to-consolidation switching, regime sequence and structural-break geometry.

**Priority/source:** P0; TDA_CHANGEPOINT_2025

## `intra_hawkes_event_features`

**Signature:** `intra_hawkes_event_features(x, event='abs_return|volume_burst|amount_burst', threshold='quantile', q=0.95, kernel='exp|powerlaw', history_days=20, output='branching_ratio|baseline|decay|end_intensity|cluster_ratio|half_life')`

**Semantics:** Convert completed-session threshold events into point process timestamps and fit a stable univariate Hawkes model on past+current history with parameters constrained to branching ratio <1. Return self-excitation and intensity summaries.

**Parameters:** event; threshold/q; kernel; history; optimizer bounds; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Event clustering/self-excitation, isolated spike versus ridge, rough-flow directions.

**Priority/source:** P1; ROUGH_MARKET_2026

## `intra_multifractal_spectrum`

**Signature:** `intra_multifractal_spectrum(x, method='mfdfa', q_grid=(-4,-2,0,2,4), scales=(4,8,16,32), output='width|asymmetry|curvature|h_q1|h_q4|delta_alpha')`

**Semantics:** Compute multifractal detrended fluctuation analysis on a normalized intraday return/activity path over fixed scales and q values; regress log fluctuation against log scale and derive spectrum summary.

**Parameters:** method; q_grid; scales; polynomial detrending order; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Intraday multifractal roughness, heterogeneity and scaling asymmetry.

**Priority/source:** P1; MULTIFRACTAL_CHINA_PUBLIC

## `intra_wavelet_scattering_features`

**Signature:** `intra_wavelet_scattering_features(x, J=5, Q=8, order=2, output='energy|entropy|order2_ratio|scale_slope|coordinate', coordinate=None)`

**Semantics:** Compute fixed wavelet-scattering transform of normalized completed-session path; aggregate first/second-order scattering coefficients into invariant multiscale summaries. Filter bank is fixed ex ante.

**Parameters:** J; Q; order; padding; output/coordinate.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Stable multiscale nonlinear path geometry beyond ordinary wavelet energy.

**Priority/source:** P2; WAVELET_SCATTERING_2022

## `intra_emd_hilbert_huang_features`

**Signature:** `intra_emd_hilbert_huang_features(x, method='eemd|ceemdan', max_imfs=8, ensemble=50, output='imf_energy_ratio|instant_freq_centroid|hilbert_entropy|trend_energy|mode_count')`

**Semantics:** Apply EEMD/CEEMDAN with deterministic seed and fixed ensemble noise, obtain intrinsic mode functions, Hilbert instantaneous amplitudes/frequencies, and summarize scale energy/frequency organization.

**Parameters:** method; max_imfs; ensemble; seed; stopping rule; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Adaptive nonlinear multi-scale decomposition and instantaneous-frequency factors.

**Priority/source:** P2; R45_COMPLEX_DIRECTION_RESEARCH

## `intra_visibility_graph_features`

**Signature:** `intra_visibility_graph_features(x, graph='horizontal|natural', output='degree_entropy|clustering|assortativity|forward_backward_asymmetry|motif_entropy|spectral_gap')`

**Semantics:** Map normalized intraday path to natural or horizontal visibility graph using canonical visibility rule, then return deterministic graph statistics. Forward/backward asymmetry compares graph statistics under time reversal.

**Parameters:** graph type; optional max_nodes/resample grid; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Intraday temporal-network geometry and time-arrow factors.

**Priority/source:** P1; VISIBILITY_GRAPH_PUBLIC

## `intra_information_flow_features`

**Signature:** `intra_information_flow_features(x, y, estimator='discrete|knn', lags=(1,2,3,5), output='te_peak|te_lag|effective_te|conditional_mi|directional_asymmetry')`

**Semantics:** Estimate nonlinear lagged information flow between two intraday channels using strictly contemporaneous/past observations within the completed session. Directional asymmetry = flow x->y minus y->x under identical estimator settings.

**Parameters:** estimator; lags; bins/k; bias correction/permutation count; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Price-volume nonlinear causal-direction proxies and information-flow asymmetry.

**Priority/source:** P1; TRANSFER_ENTROPY_PUBLIC

## `intra_covariance_manifold_shift`

**Signature:** `intra_covariance_manifold_shift(x, y, z=None, history_days=20, grid=24, metric='affine|bures|logeuclidean', output='distance|logdet_shift|condition_shift')`

**Semantics:** Build SPD covariance matrices from multichannel intraday features, regularize eigenvalues, and compare current covariance structure with past reference on the SPD manifold via affine-invariant/Bures/log-Euclidean metrics.

**Parameters:** history; grid; metric; shrinkage epsilon; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Correlation/covariance geometry and dependence-regime shifts.

**Priority/source:** P2; R45_COMPLEX_DIRECTION_RESEARCH

## `intra_diffusion_map_state`

**Signature:** `intra_diffusion_map_state(x, y=None, history_days=120, grid=48, n_components=3, epsilon='median', output='coord|novelty|local_density|transition_speed', component=1)`

**Semantics:** Represent completed historical sessions as normalized path vectors, construct kernel graph using past sessions only, compute diffusion-map embedding, Nyström-project current session and summarize coordinates/density/novelty.

**Parameters:** history; grid; components; kernel epsilon; output/component.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Nonlinear manifold state and path-regime coordinates.

**Priority/source:** P2; MANIFOLD_LEARNING_PUBLIC

## `intra_common_trading_intensity`

**Signature:** `intra_common_trading_intensity(volume, weight=None, n_slots=48, history_days=60, output='loading|idio_activity|common_shock_exposure|residual_energy|lead_lag')`

**Semantics:** Estimate a cross-sectional common intraday trading-intensity factor from normalized stock volume profiles at each time slot using lagged/available universe only. For each stock return loading, residual activity and exposure to current common shocks; use ex-self construction where feasible to avoid mechanical self-inclusion.

**Parameters:** n_slots; history; weighting; robust normalization; output; ex_self flag default True.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Common Trading Intensity, systematic-liquidity shocks and idiosyncratic activity.

**Priority/source:** P1; COMMON_TRADING_INTENSITY_2026

## `intra_price_efficiency_state_space`

**Signature:** `intra_price_efficiency_state_space(price, market_price=None, history_days=60, output='pricing_error_std|error_persistence|innovation_ratio|efficiency_score|common_error_loading')`

**Semantics:** Use a local state-space model to decompose observed minute price into efficient state and transitory pricing error; optional market channel decomposes common/idiosyncratic error. Efficiency score is inverse standardized transitory error persistence/variance.

**Parameters:** history; local level/trend choice; optional market; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Cross-sectional price-efficiency, transient mispricing and common pricing-error factors.

**Priority/source:** P1; PRICE_EFFICIENCY_STATE_SPACE_2026

## `intra_neural_cde_embedding`

**Signature:** `intra_neural_cde_embedding(x, y=None, z=None, model_id=None, output='embedding_norm|coordinate|reconstruction_error|state_change', coordinate=None)`

**Semantics:** Evaluate a frozen Neural Controlled Differential Equation encoder on a time-augmented normalized intraday path. Training is outside FactorEngine; model_id must identify a versioned walk-forward artifact trained only on dates preceding the decision date. FactorEngine performs deterministic inference only.

**Parameters:** model_id required; interpolation convention; output/coordinate; deterministic device/seed policy.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Learned continuous-time path embeddings for complex nonlinear shape discovery.

**Priority/source:** P2; R45_COMPLEX_DIRECTION_RESEARCH

## `intra_contrastive_path_embedding`

**Signature:** `intra_contrastive_path_embedding(x, y=None, model_id=None, output='embedding_norm|coordinate|prototype_distance|novelty', coordinate=None)`

**Semantics:** Deterministic inference from a frozen self-supervised/contrastive path encoder trained on strictly historical sessions. Return scalar coordinates or distances to frozen prototypes; no online label use.

**Parameters:** model_id; normalization/artifact metadata; output/coordinate.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Self-supervised motif/state discovery without hand-specified pattern taxonomy.

**Priority/source:** P2; R45_COMPLEX_DIRECTION_RESEARCH

## `intra_matrix_profile_session_features`

**Signature:** `intra_matrix_profile_session_features(x, history_days=120, subsequence=12, normalize='z', output='discord|motif_distance|motif_frequency|motif_age|neighbor_dispersion')`

**Semantics:** Resample each completed intraday path to a fixed grid; compare current subsequences against strictly prior sessions with a z-normalized matrix-profile search. Exclude trivial self matches and same-day future observations. Return discord/motif statistics aggregated to one stock-day scalar.

**Parameters:** history_days>=20; subsequence>=4; grid and exclusion zone fixed; output selector.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Session-level Matrix Profile motifs/discords and repeated intraday archetypes.

**Priority/source:** P1; MATRIX_PROFILE_FINANCE_2021

## `intra_dtw_archetype_features`

**Signature:** `intra_dtw_archetype_features(x, history_days=120, k=8, band=0.1, output='nearest_distance|archetype_id|prototype_margin|historical_forward_dispersion')`

**Semantics:** Normalize completed-session paths and cluster strictly historical sessions into k medoids under constrained Dynamic Time Warping. Current session is assigned to frozen/current walk-forward archetypes; forward_dispersion uses only historical outcomes attached to prior archetype instances and never current/future labels.

**Parameters:** history_days; k; Sakoe-Chiba band; normalization; fit cadence; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** DTW pattern representation, including A-share short-term morphology families.

**Priority/source:** P1; DTW_A_SHARE_2020

## `intra_local_conditional_entropy`

**Signature:** `intra_local_conditional_entropy(x, alphabet=5, word=4, history_days=60, output='local_entropy|continuation_max_prob|word_frequency|surprise')`

**Semantics:** Discretize normalized intraday increments using fixed/lagged quantile thresholds, form words of length m, and estimate empirical continuation distributions from strictly past comparable words. Return local conditional entropy, maximum continuation probability, frequency or surprise.

**Parameters:** alphabet; word length; threshold policy; smoothing pseudocount; minimum matches; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Pockets of local order/predictability hidden by average entropy.

**Priority/source:** P1; LOCAL_PREDICTABILITY_2000

## `intra_business_time_deformation`

**Signature:** `intra_business_time_deformation(price, activity, buckets=32, clock='volume|amount|volatility', output='clock_gap|path_distance|roughness_gap|efficiency_gap|timing_wasserstein')`

**Semantics:** Construct equal-business-time buckets from cumulative activity and compare the price path with equal-clock-time sampling. Outputs quantify how much the trading clock is deformed, including path/roughness/efficiency differences and Wasserstein distance between activity-time and clock-time locations.

**Parameters:** clock; buckets; activity monotonicity; zero-activity handling; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Time deformation / volume clock / business-time state as an independent microstructure direction.

**Priority/source:** P1; BUSINESS_TIME_PUBLIC

## `intra_quantile_dependence_features`

**Signature:** `intra_quantile_dependence_features(x, y=None, qx=0.9, qy=0.9, lags=(1,2,3,5), output='quantilogram|cross_quantilogram|peak_lag|asymmetry|spectral_concentration')`

**Semantics:** Build quantile-hit indicators on the completed session or past rolling intraday history and estimate auto/cross quantilograms. Thresholds must be determined from lagged or past-complete history; return lag-specific, peak-lag, asymmetry or spectral concentration statistics.

**Parameters:** quantiles; lags; threshold window; min hits; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Intraday directional predictability in different distribution regions, not mean correlation.

**Priority/source:** P1; QUANTILOGRAM_2007|CROSS_QUANTILOGRAM_2016

## `intra_kramers_moyal_dynamics`

**Signature:** `intra_kramers_moyal_dynamics(x, bins=12, lag=1, output='drift_at_zero|restoring_strength|diffusion_at_zero|diffusion_slope|stability_ratio')`

**Semantics:** Estimate conditional first and second Kramers-Moyal coefficients from intraday increments/state bins. Fit local drift D1 and diffusion D2 with robust minimum-bin counts; restoring strength is negative local derivative of drift around equilibrium.

**Parameters:** bins; lag; state normalization; min_bin_count; local regression; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Local restoring force versus stochastic diffusion in the intraday path.

**Priority/source:** P1; KRAMERS_MOYAL_FINANCE_2007

## `intra_extreme_event_interval_memory`

**Signature:** `intra_extreme_event_interval_memory(x, threshold='quantile', q=0.95, event='abs_return|volume|amount', output='mean_interval|cv|lag1_memory|long_memory_slope|cluster_index')`

**Semantics:** Detect intraday extreme events using lagged/fixed thresholds, compute inter-event durations within and across completed sessions, and summarize interval dispersion and dependence. Do not bridge lunch/overnight unless mode explicitly requests trading-time durations.

**Parameters:** event; q; lookback; session-boundary mode; minimum events; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Memory/scaling of volatility or activity return intervals and event clustering.

**Priority/source:** P1; EXTREME_INTERVAL_MEMORY_PUBLIC

## `intra_signature_lead_lag_network`

**Signature:** `intra_signature_lead_lag_network(price, universe_weight=None, depth=2, history_days=40, output='out_strength|in_strength|net_lead|pagerank|leader_exposure')`

**Semantics:** For each stock pair compute an ordered level-2 path-signature/Levy-area directional score from strictly historical synchronized returns; sparsify statistically or by fixed top-k and return stock-level directed-network features. Ex-self construction required for leader_exposure.

**Parameters:** depth fixed at 2 initially; history; sparsification/FDR; weights; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Signature-based directed lead-lag networks and leader/follower roles.

**Priority/source:** P1; LEAD_LAG_SIGNATURE_2026

## `intra_validated_lead_lag_network`

**Signature:** `intra_validated_lead_lag_network(price, lags=(1,2,3,5), history_days=40, fdr=0.05, output='out_degree|in_degree|net_flow|motif_role|lead_strength|lag_strength')`

**Semantics:** Build pairwise lagged-return links from synchronized intraday bars using strictly historical sessions. Validate links with permutation/bootstrap null and Benjamini-Hochberg FDR before computing directed graph features per stock.

**Parameters:** lags; history; FDR; bootstrap/permutation count; minimum observations; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Statistically validated intraday lead-lag network, separate from simple market beta/lead-lag scalar.

**Priority/source:** P1; INTRADAY_LEAD_LAG_NETWORK_2014

## `intra_dynamic_stock_graph_features`

**Signature:** `intra_dynamic_stock_graph_features(price, history_days=20, metric='corr|mutual_info|partial_corr', threshold='topk|fdr', output='degree|eigenvector|clustering|community_bridge|graph_surprise')`

**Semantics:** Construct a walk-forward cross-stock dependence graph from intraday return profiles and return node-specific network structure. graph_surprise compares current completed-session graph role to prior reference; no future dates in graph estimation.

**Parameters:** history; metric; threshold; shrinkage for partial corr; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Dynamic cross-stock network centrality, bridge status and structural surprise.

**Priority/source:** P2; FINANCIAL_NETWORK_PUBLIC

## `intra_tensor_common_mode`

**Signature:** `intra_tensor_common_mode(price, volume=None, amount=None, history_days=20, rank=5, output='stock_loading|residual_energy|mode_concentration|loading_shift|reconstruction_error')`

**Semantics:** Build a rolling stock x intraday-slot x channel tensor from strictly completed sessions, apply deterministic CP/Tucker low-rank decomposition with sign/order identification, and return each stock's loading/residual/common-mode statistics.

**Parameters:** history; grid; rank; CP/Tucker choice; initialization/seed; component ordering; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Low-rank common intraday modes and stock-specific deviation from market-wide tensor structure.

**Priority/source:** P2; TENSOR_NETWORK_2017

## `intra_topological_peer_anomaly`

**Signature:** `intra_topological_peer_anomaly(price, history_days=40, embedding_dim=3, delay=1, peer_k=20, output='peer_distance|persistence_residual|topological_z|state_novelty')`

**Semantics:** For each stock/day compute delay-embedding persistence features from completed intraday path, then compare with a lagged peer/cross-sectional reference. Peer groups are based on lagged features/industry when available; never use future returns.

**Parameters:** history; embedding_dim; delay; peer_k; topological feature vector; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Cross-sectional topological anomaly: unusual path topology relative to market/peers.

**Priority/source:** P2; TOPOLOGICAL_CROSS_SECTION_2026

## `intra_critical_transition_score`

**Signature:** `intra_critical_transition_score(x, history_days=60, outputs='recurrence|topology|autocorr|variance', output='score|recurrence_component|topology_component|critical_slowing')`

**Semantics:** Combine pre-specified early-warning components from recurrence statistics, persistence change and critical-slowing proxies. All scaling parameters are estimated from prior sessions; return both components and a fixed-weight or historically frozen composite.

**Parameters:** history; component set; fixed weights or frozen historical weights; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Critical transition / local instability direction distinct from ordinary volatility level.

**Priority/source:** P2; RECURRENCE_EWS_2024|TDA_FINANCE_PUBLIC

## `intra_realized_measure_state_vector`

**Signature:** `intra_realized_measure_state_vector(price, output='semivariance_balance|jump_fraction|quarticity_ratio|shape_distance|realized_state_norm')`

**Semantics:** Compute a standardized vector of realized variance, up/down semivariances, bipower variation, signed jumps, skewness, kurtosis and quarticity from completed-session returns. Outputs are predefined nonlinear coordinates/distances, not a fitted predictive model.

**Parameters:** sampling frequency; microstructure-robust return convention; truncation/winsorization; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Joint realized-measure state rather than isolated volatility/jump statistics.

**Priority/source:** P0; GOOD_BAD_VOLATILITY_2019|REALIZED_MOMENTS_2022|SIGNED_JUMP_CHINA_2017

## `intra_symbolic_dynamics_features`

**Signature:** `intra_symbolic_dynamics_features(x, alphabet=5, order=3, output='lz_complexity|permutation_entropy|transition_entropy|forbidden_ratio|irreversibility|predictability_bound')`

**Semantics:** Discretize completed intraday increments and compute multiple symbolic-dynamics statistics under one consistent alphabet/ordinal convention. Predictability bound is derived from entropy rate and alphabet size; thresholds are fixed or lagged-history based.

**Parameters:** alphabet; order; delay; threshold policy; pseudocount; output.

**Null policy:** Return null when the session/history is insufficient; never coerce missing bars to zero unless explicitly specified.

**Ordering/alignment:** Per stock, sort by trade_date then minute timestamp; respect the A-share lunch break and session boundaries; deterministic tie-breaking by timestamp.

**Backends:** Define one numerical reference implementation; Polars is primary vectorized/stateful backend, Pandas parity is required, DuckDB uses window/list UDF or pre-aggregation with identical float/null semantics.

**PIT/no-lookahead:** Use only observations timestamped at or before the end-of-day decision time. Any fitted state/model/basis must be expanding or rolling on past data only; no same-day future bars before their timestamp and no future labels.

**Minimum tests:** Synthetic path with known result; irregular/missing minutes; flat/zero-volume day; extreme jump day; lunch boundary; deterministic repeatability; Pandas/Polars/DuckDB tolerance parity; explicit no-lookahead test.

**Unlocks:** Intraday symbolic order, time arrow and entropy-bound predictability.

**Priority/source:** P1; LZ_PREDICTABILITY_CHINA_2020|PERMUTATION_DYNAMICS_PUBLIC

---

## 附录来源：`R46_OPERATOR_ENGINEERING_SPECS.md`

> 以下是 R41–R47 原始工程规格，作为候选语义档案保留。**若与本文件前文的 current-main 规则、字段硬约束、backend policy 或 preflight 结果冲突，以前文为准。**

# R46 Operator Engineering Specifications

These operators are proposed from the R46 public-research and mechanism-expansion pass. They are specification-first: implement the semantics below before promoting the pre-created factors.

## `intra_eod_reversal_decomposition`
**Signature:** `intra_eod_reversal_decomposition(price, volume=None, amount=None, window_minutes=30, baseline='prior_window|morning', output='pressure|reversal|retention|participation_adjusted')`

**Semantics.** Decompose the final trading-window move into transient pressure, retained information, and participation-adjusted reversal using only same-day bars up to the close. Compare the close window with an immediately preceding or morning baseline; output one stock-day scalar.

**Parameters.** window_minutes>=5; baseline; robust scaling; output selector

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_volume_shock_state`
**Signature:** `intra_volume_shock_state(volume, amount=None, forecast_profile='historical_slot', z=3.0, output='shock|late_shock|persistence|concentration')`

**Semantics.** Estimate expected volume by historical same-slot profile using past sessions only, then measure standardized same-day volume surprise, its timing, persistence, and concentration.

**Parameters.** lookback_days>=20; z>0; slot definition; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_comovement_curve_ex_self`
**Signature:** `intra_comovement_curve_ex_self(price, weight, bins=12, output='slope|dispersion|open_close_gap|curvature')`

**Semantics.** For each stock and session, estimate ex-self market correlation/beta on successive intraday bins and summarize the curve. Weighting must exclude the focal stock from the benchmark.

**Parameters.** bins>=4; weight; robust estimator; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_price_volume_cross_wavelet`
**Signature:** `intra_price_volume_cross_wavelet(price, activity, scales=(2,4,8,16,32), output='coherence|phase|lead_lag|scale_concentration')`

**Semantics.** Cross-wavelet transform of price returns and volume/amount changes; summarize scale-specific coherence, phase lead-lag and concentration.

**Parameters.** causal/session-local wavelet; scales; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_bicoherence_features`
**Signature:** `intra_bicoherence_features(price, output='max|mean|phase_coupling|triad_concentration')`

**Semantics.** Estimate normalized bispectrum/bicoherence of intraday returns to capture nonlinear phase coupling across frequencies that ordinary power spectra miss.

**Parameters.** minimum bars; frequency grid; shrinkage; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_frequency_granger_price_volume`
**Signature:** `intra_frequency_granger_price_volume(price, activity, bands=((2,8),(8,32),(32,120)), output='activity_to_price|price_to_activity|net')`

**Semantics.** Estimate causal price-volume information flow by frequency band using a stable VAR/spectral-Granger representation.

**Parameters.** past-only within session; lag order fixed or BIC from past; bands; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 3

## `intra_range_competition_profile`
**Signature:** `intra_range_competition_profile(high, low, close, volume=None, bins=12, output='competition|acceptance|rejection|rotation')`

**Semantics.** Represent intraday bullish-bearish competition through range expansion, repeated boundary tests, close acceptance, and optional volume participation; distinguish wide efficient discovery from noisy rotation.

**Parameters.** bins; robust ATR scaling; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_absorption_curve_area`
**Signature:** `intra_absorption_curve_area(price, activity, event_threshold=3.0, horizon=30, output='area|half_life|convexity|adverse_area')`

**Semantics.** After objective price impulses, build the adverse-price-move per unit activity curve and summarize area, recovery half-life and convexity. Does not infer trader identity.

**Parameters.** event threshold; horizon; aggregation; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_volume_profile_value_area`
**Signature:** `intra_volume_profile_value_area(price, volume, amount=None, price_bins=50, value_area=0.7, output='poc_distance|value_area_width|upper_overhang|lower_support|vacuum_width')`

**Semantics.** Construct same-day volume-at-price distribution, point of control and value area; quantify current-price distance, overhead supply mass and low-volume vacuum width.

**Parameters.** price_bins>=20; value_area in (0,1); output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 5

## `turnover_chip_overhang_surface`
**Signature:** `turnover_chip_overhang_surface(close, turnover, lookback=120, decay='survival', bands=(0.02,0.05,0.10), output='near_overhang|profit_supply|vacuum|peak_pressure')`

**Semantics.** Infer historical cost-distribution survival from turnover and prices, then compute distance-weighted overhead supply, profit-taking supply, cost vacuum and peak pressure.

**Parameters.** lookback; survival convention; bands; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `fundamental_latent_balance_sheet_factor`
**Signature:** `fundamental_latent_balance_sheet_factor(*fields, lookback_periods=20, method='pca|robust_pca', component=1, output='score|residual|loading_stability')`

**Semantics.** Across selected PIT financial-statement changes, estimate rolling past-only latent balance-sheet financing/investment components and return firm scores or reconstruction residuals.

**Parameters.** field list fixed in recipe; fiscal alignment; rolling training strictly past; component; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 3

## `fundamental_working_capital_financing_state`
**Signature:** `fundamental_working_capital_financing_state(receivables, inventory, payables, short_debt, long_debt, ocf, assets, output='collateral|supplier_finance|mismatch|refinancing_pressure')`

**Semantics.** Composite but decomposable working-capital financing state based on receivables/inventory collateral, supplier financing, explicit debt, and operating cash support.

**Parameters.** PIT fiscal values; robust denominator floors; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `fundamental_cost_stickiness_panel`
**Signature:** `fundamental_cost_stickiness_panel(cost, sales, periods=8, output='stickiness|anti_stickiness|elasticity_down|elasticity_up')`

**Semantics.** Estimate asymmetric cost response to sales increases versus decreases over historical fiscal observations, with no future periods.

**Parameters.** periods>=6; minimum up/down observations; log-change handling; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_multiscale_state_residence`
**Signature:** `intra_multiscale_state_residence(price, activity=None, scales=(5,15,30,60), states=5, output='persistence|switch_rate|metastability|scale_disagreement')`

**Semantics.** Discretize standardized path state at multiple horizons and measure residence time, switching and disagreement across scales.

**Parameters.** scales; state quantization learned from past; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_visibility_motif_transition`
**Signature:** `intra_visibility_motif_transition(price, motif_size=4, output='entropy|dominant_transition|irreversibility|rare_motif')`

**Semantics.** Build horizontal/natural visibility graph motifs over the intraday path and summarize motif transition probabilities and time irreversibility.

**Parameters.** motif_size; graph convention; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_recurrence_network_features`
**Signature:** `intra_recurrence_network_features(price, embedding_dim=3, delay=1, target_rr=0.05, output='clustering|transitivity|path_length|assortativity')`

**Semantics.** Convert delay-embedded intraday path to a recurrence network at fixed recurrence rate and extract graph topology beyond standard RQA line statistics.

**Parameters.** embedding; delay; target_rr; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_transfer_operator_metastability`
**Signature:** `intra_transfer_operator_metastability(price, activity=None, bins=8, lag=1, output='second_eigenvalue|metastability|committor_gap|mixing_time')`

**Semantics.** Estimate a regularized empirical transfer/Perron-Frobenius operator for intraday state transitions and summarize slow metastable modes.

**Parameters.** bins; lag; shrinkage; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_local_drift_diffusion_surface`
**Signature:** `intra_local_drift_diffusion_surface(price, activity=None, state_bins=8, dt=1, output='drift_slope|diffusion_slope|restoring_strength|instability')`

**Semantics.** Estimate Kramers-Moyal local drift and diffusion conditional on price/return state, providing restoring-force and instability measures.

**Parameters.** state bins; regularization; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `intra_session_ot_map`
**Signature:** `intra_session_ot_map(price, activity, reference='past_profile', bins=30, output='wasserstein|mass_upshift|mass_downshift|transport_asymmetry')`

**Semantics.** Optimal-transport distance and directional mass movement between today’s price-activity distribution and a past-only expected session profile.

**Parameters.** reference lookback; bins; ground metric; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

## `panel_intraday_low_rank_residual_ex_self`
**Signature:** `panel_intraday_low_rank_residual_ex_self(price, weight, rank=5, bins=12, output='residual_energy|loading|common_share|idiosyncratic_shape')`

**Semantics.** Construct a past-only low-rank factorization of cross-stock intraday return curves and return focal-stock residual/common structure using ex-self benchmark safeguards.

**Parameters.** rank; bins; expanding/rolling basis; ex-self; output

**Null policy.** Return null when required session/fiscal history is insufficient; never convert missing observations to zero unless the operator definition explicitly models zero activity.

**Ordering/alignment.** Sort deterministically by stock/date/time. Respect A-share session boundaries and lunch break. Financial data must be announcement-PIT/as-of aligned.

**Backend requirements.** Define one reference numerical algorithm. Polars is primary; Pandas parity required. DuckDB may use list/UDF or pre-aggregation but must match reference null, ordering, tie and floating semantics.

**PIT / no-lookahead.** All fitted profiles, bases, thresholds, latent states and peer benchmarks use only data available at the decision timestamp. EOD daily factors may use the completed current session and are earliest tradable NEXT_TRADING_SESSION. Cross-stock benchmarks must be ex-self where specified.

**Minimum tests.** Synthetic known-result path; flat/zero-activity day; missing/irregular bars; lunch boundary; extreme event; deterministic repeatability; insufficient-history null; PIT shift test; Pandas/Polars/DuckDB parity; ex-self leakage test where applicable.

**Pre-created factors:** 4

---

## 附录来源：`r47_operator_engineering_specs.md`

> 以下是 R41–R47 原始工程规格，作为候选语义档案保留。**若与本文件前文的 current-main 规则、字段硬约束、backend policy 或 preflight 结果冲突，以前文为准。**

# R47 Operator Engineering Specifications

These are R47 additions beyond the inherited R46 recommendation set. Active factors never call these until implementation and contract tests pass.

## `fin_schema_gate`
- **Signature:** `fin_schema_gate(schema_name, x, statement_schema=None, strict=True)`
- **Semantics:** Return x only for instruments whose audited financial-statement schema matches schema_name; otherwise NaN. The schema comes from DataAccess metadata, never inferred from which fields happen to be non-null.
- **Parameters:** schema_name: bank|insurance|financial_services|oil_gas|agriculture|general; x: panel expression; statement_schema: runtime metadata panel/string; strict: missing schema fails closed.
- **Null policy:** Unknown/missing schema -> NaN; x NaN remains NaN. Never coerce unknown schema to general.
- **Ordering/alignment:** Exact timestamp×instrument alignment between x and schema metadata. No row-order inference.
- **Backend requirements:** Pandas/Polars/DuckDB must implement identical categorical gate; schema metadata should be dictionary/categorical encoded.
- **PIT / lookahead:** Schema classification must be known as of decision time; historical reclassification applies only from its effective/publication timestamp.
- **Minimum tests:** Schema match/mismatch/unknown; historical schema change; alignment mismatch; all-NaN industry field; cross-backend parity.
- **Unlocks:** Bank/insurance/broker/oil-gas/agriculture field families currently excluded from universal active pool.
- **Priority:** P0
- **Field gate:** REQUIRES_FINANCIAL_STATEMENT_SCHEMA_METADATA
- **Pre-created R47 candidates:** 32

## `ts_lagged_predictability_score`
- **Signature:** `ts_lagged_predictability_score(signal, realized_return, window=252, horizon=1, method='rankic', min_periods=80)`
- **Semantics:** At date t estimate historical predictive skill using pairs signal[s-horizon] versus realized_return[s] for s<=t only; emit rolling rank-IC/IC/t-stat style skill as a causal meta-signal.
- **Parameters:** window; horizon>=1; method rankic|pearson|sign; min_periods. Pairing is explicit signal_{s-h}->return_s.
- **Null policy:** Pairwise finite observations only; below min_periods -> NaN; gaps never shift/reconnect across timestamps.
- **Ordering/alignment:** Strict date/instrument alignment; compute per instrument through time; horizon pairing uses calendar panel shift, not positional compression after NaNs.
- **Backend requirements:** Pandas/Polars streaming rolling pair-stat; DuckDB window implementation where possible.
- **PIT / lookahead:** No current/future target: at t only returns already realized by t and signals dated <=t-horizon may enter. Must fail tests that substitute signal_t with return_{t+1}.
- **Minimum tests:** Synthetic predictive/noise signals; horizon pairing; NaN gaps; no-lookahead perturbation; chunk/full-history parity; backend parity.
- **Unlocks:** State-dependent predictability/mosaic reliability weighting and historical signal-quality meta-factors.
- **Priority:** P0
- **Field gate:** PASS_EXISTING_FIELD_CONTRACT
- **Pre-created R47 candidates:** 5

## `cs_predictability_mosaic_score`
- **Signature:** `cs_predictability_mosaic_score(base_signal, state_features..., history_window=504, min_history=160, clusters=6, lag=1)`
- **Semantics:** Causally learn state partitions from lagged historical feature/return pairs, assign current stock-state to a learned mosaic, and output historical out-of-sample predictive skill of base_signal in that mosaic.
- **Parameters:** base_signal plus 2-6 state features; history_window; min_history; clusters; lag>=1. Deterministic tree/partition seed.
- **Null policy:** Insufficient cluster history -> NaN; unknown state feature -> NaN/fallback only if explicitly configured.
- **Ordering/alignment:** Cross-section per date plus historical rolling training. Training set ends before current target horizon.
- **Backend requirements:** Pandas reference; Polars preprocessing plus deterministic sklearn-equivalent kernel acceptable only with exact evidence; DuckDB may materialize features but model scoring is Python/Polars research-fastpath.
- **PIT / lookahead:** Purged historical fit only. Current/future return labels are prohibited. Refit schedule and lag must be explicit.
- **Minimum tests:** Leakage perturbation; deterministic partitions; sparse-cluster fallback; microcap/NaN stress; rolling refit reproducibility.
- **Unlocks:** Mosaic-style state-dependent predictability factors using volume, valuation, liquidity, volatility and accounting state.
- **Priority:** P1
- **Field gate:** PASS_EXISTING_FIELD_CONTRACT
- **Pre-created R47 candidates:** 3

## `intra_functional_autoencoder_score`
- **Signature:** `intra_functional_autoencoder_score(path, history_days=120, latent_dim=4, fit_lag=1, output='reconstruction_error')`
- **Semantics:** Normalize each completed intraday path to a fixed session grid; fit a low-dimensional autoencoder/PCA-like nonlinear manifold only on prior sessions; score current path reconstruction error or latent coordinates.
- **Parameters:** history_days; latent_dim; fit_lag>=1; output reconstruction_error|latent_1..k; deterministic seed and fixed architecture.
- **Null policy:** Insufficient minute coverage or training days -> NaN; no interpolation across lunch/session gaps unless grid policy explicitly permits.
- **Ordering/alignment:** Per instrument or pooled model with stock-id neutralization; fixed official minute slots.
- **Backend requirements:** Reference numpy/PyTorch optional research kernel; production Polars handles session packing; model artifact versioned and deterministic.
- **PIT / lookahead:** Current day can be scored only after session close for next-session use; training excludes current day and all future sessions.
- **Minimum tests:** Known manifold synthetic paths; anomaly injection; missing-slot policy; refit leakage; deterministic serialization.
- **Unlocks:** Latent intraday path anomaly, nonlinear motif and manifold-distance factors.
- **Priority:** P2
- **Field gate:** PASS_MINUTE_OHLCV
- **Pre-created R47 candidates:** 4

## `intra_hmm_posterior_entropy`
- **Signature:** `intra_hmm_posterior_entropy(path_features..., history_days=120, states=4, fit_lag=1, output='entropy')`
- **Semantics:** Fit a causal HMM/HSMM to prior-session intraday feature sequences and score the completed current session by posterior state entropy, dominant-state occupancy, switching intensity or state surprise.
- **Parameters:** 2-5 path features; states; history_days; fit_lag; output entropy|switch_rate|dominant_share|surprise.
- **Null policy:** Insufficient training/session observations -> NaN; missing segments break sequence rather than bridge states.
- **Ordering/alignment:** Official minute order; lunch break/session boundary explicit.
- **Backend requirements:** Deterministic reference implementation; Polars session packing; model fitting may remain controlled Python kernel with versioned parameters.
- **PIT / lookahead:** Fit through t-1 only; t session score available after close for t+1 decisions.
- **Minimum tests:** Regime-switch synthetic data; no-leak fit; gap segmentation; deterministic labels/permutation-invariant outputs.
- **Unlocks:** Intraday latent regime maturity/switching and state-transition factors.
- **Priority:** P2
- **Field gate:** PASS_MINUTE_OHLCV
- **Pre-created R47 candidates:** 4

## `intra_liquidity_resilience_curve_fit`
- **Signature:** `intra_liquidity_resilience_curve_fit(price, activity, shock_threshold=2.5, horizon=30, output='half_life')`
- **Semantics:** Detect intraday price/activity shocks and fit post-shock recovery/decay curves. Outputs residual fraction, exponential half-life, asymptote, recovery slope or fit quality, aggregated across causal within-day events.
- **Parameters:** shock_threshold; horizon; output half_life|residual|asymptote|slope|r2; tie and overlapping-event policy.
- **Null policy:** No valid shock -> NaN; truncated end-of-session events excluded or explicitly censored.
- **Ordering/alignment:** Strict minute slots; activity aligned to price; overlapping shocks use deterministic refractory rule.
- **Backend requirements:** Vectorized Polars session kernel preferred; Pandas/numpy reference; DuckDB not required for nonlinear fit.
- **PIT / lookahead:** Uses only completed same-day path and is deployable next session; no post-close/future bars.
- **Minimum tests:** Synthetic exponential recovery; no recovery; multiple shocks; censoring; missing minute; cross-backend tolerance.
- **Unlocks:** Price-probe success, supply absorption, post-impulse consolidation and resilience-curve factors.
- **Priority:** P0
- **Field gate:** PASS_MINUTE_OHLCV
- **Pre-created R47 candidates:** 5

## `intra_function_on_function_anomaly_response`
- **Signature:** `intra_function_on_function_anomaly_response(path, anomaly_path, history_days=120, basis_dim=6, fit_lag=1, output='response_norm')`
- **Semantics:** Fit a penalized function-on-function regression on prior sessions linking an intraday anomaly/state curve to later intraday return curve; score current completed anomaly curve by expected response norm, sign, timing centroid or reversal point.
- **Parameters:** history_days; basis_dim; fit_lag>=1; regularization; output response_norm|signed_response|centroid|reversal_time.
- **Null policy:** Insufficient functional coverage/training -> NaN; basis fitted only on historical sessions.
- **Ordering/alignment:** Fixed minute grid with explicit session breaks; basis and regression artifact versioned.
- **Backend requirements:** Research/production boundary explicit; Polars for grid/basis inputs, deterministic numerical kernel for penalized functional regression.
- **PIT / lookahead:** Training targets end before current decision date; current day functional score only predicts next session unless an earlier truncated path contract is separately certified.
- **Minimum tests:** Synthetic functional response; reversal timing; leakage perturbation; basis stability; missing-grid behavior.
- **Unlocks:** Topological/latent anomaly history to future-return-curve response and intraday functional transmission.
- **Priority:** P3
- **Field gate:** PASS_MINUTE_OHLCV
- **Pre-created R47 candidates:** 4

---

# 22. 最终硬性检查清单

在宣布完成前逐项打勾：

- [ ] 已重新读取执行时最新 `main`，没有按 R47 旧 snapshot 开发。
- [ ] 已对 156 + 新补充候选做 preflight。
- [ ] 没有重复 current canonical。
- [ ] 没有只换名字的重复 operator。
- [ ] 没有能无损组合却硬新增的 composite operator。
- [ ] 没有新增用户不存在的 A 股 raw field。
- [ ] 没有 Level2/Tick/orderbook/news 幻觉。
- [ ] 没有用 UpdateTime 作为交易信号。
- [ ] 没有用 ReportPeriodEndDate 冒充公告时间。
- [ ] 没有把财务 NaN 填 0。
- [ ] 没有把 daily forward-filled 财务 row shift 当 fiscal lag。
- [ ] 没有 current/future target leakage。
- [ ] 没有 negative shift / bfill / centered rolling。
- [ ] full-session minute factor 明确 next-session availability。
- [ ] ex-self 真正排除了 self。
- [ ] Pandas golden 完成。
- [ ] daily operator 的 Polars 路径是真 native/正式 kernel，不是 pandas bridge。
- [ ] DuckDB 只实现适合 SQL 的 operator。
- [ ] DuckDB emitter 与 parity/production certification 分开。
- [ ] 所有适用 operator 通过 future-poison。
- [ ] 所有适用 operator 通过 NaN/Inf edge tests。
- [ ] 所有适用 operator 通过 chunk/shard parity。
- [ ] registry/alias/surface/allowlist/docs/evidence 已更新。
- [ ] 无 unclassified / stub-as-implementation / research→daily 泄漏。
- [ ] 最终报告列清楚 implemented/skipped/blocked。

**任务结束标准不是“写完代码”，而是：语义、PIT、字段、backend、测试、catalog、evidence 全部闭环。**
