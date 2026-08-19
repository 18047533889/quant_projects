# FactorEngine R24：跨源语义传播、关系状态、IR 身份、快照缓存与 Mining/Label 闭环独立增量审计整改提示词

> **用途**：直接交给负责 FactorEngine 整改的代码 AI，在服务器真实工作区执行。  
> **性质**：R23 之后的**独立增量轮**，不要把 R17–R23 内容复制/合并进来。  
> **本轮目标**：专门解决 R23 深挖后继续发现的“旁路与中间层语义丢失”问题——即底层/算子本身越来越严格，但在 Relation、PIT helper、IR、SourceRef、Planner、Composite cache、Mining API、Label API 等链路上仍可能把语义丢掉、降级、猜测或绕过。  
> **GitHub 当前可见基线**：`b947c690a119c6fe42fe51b7fd2f9b5f0c746807`。  
> **重要**：服务器如果已有更新，以服务器真实 `HEAD + dirty tree` 为准；先检查现状，不要机械按本文修改已经修好的代码。  
> **硬要求**：本轮结束后，不允许存在“核心路径严格，但 convenience/legacy/default API 能绕开”的情况。

---

# 0. 本轮为什么必须单独做

R23 重点解决：

```text
基本面双时态
knowledge time
period time
revision vintage
YTD/Quarter/TTM
same-day availability
financial bundle
逐 canonical PIT certificate
```

本轮继续向外扩以后，发现新的核心问题不是“某一个财务算子还算错”，而是：

```text
正确语义进入某层
→ 中间层把它丢掉
→ 下游又靠默认值/字符串/启发式猜
→ 最终生产入口重新放行
```

典型链路：

```text
FieldSpec 有 flow_semantics
→ NormalizedFieldPlan 丢掉
→ mining search space 不导出
→ generator 不知道
→ Analyzer 某个入口 production=False
→ 非法公式仍被视为可投递
```

以及：

```text
Field 有完整 AvailabilityExpr
→ operator IR 只传播 label 字符串
→ plan/hash 又排除 semantic_attrs
→ cache/materialization identity 与真实语义脱钩
```

所以 R24 要解决的是：

> **Semantic Continuity：语义从 source 到 mining、从 field 到 cache、从 event 到 state，不能在任何一层断掉。**

---

# 1. R24 最终新增硬指标

## R24-001

定义：

```text
SEMANTIC_CONTINUITY_CLOSED
```

只有当：

```text
FieldSpec
→ ProviderBinding
→ SourceRef
→ NormalizedFieldPlan
→ Schema
→ IR
→ Plan
→ CacheIdentity
→ MaterializationIdentity
→ MiningSearchSpace
→ RuntimeAdmission
```

关键语义全链不丢失才为 true。

## R24-002

关键语义至少包括：

```text
market
dataset/provider
concept_id
field_id
unit
semantic_kind
price_basis
flow_semantics
frequency
cardinality
universe_id

temporal_model
availability_precision
availability_expr
knowledge_time
effective_time
period_id
timeframe
revision_policy
source_vintage
snapshot_id
source_version

missing_semantic
coverage_scope
applicability
```

## R24-003

新增最终硬旗标：

```text
RELATION_STATE_PIT_CLOSED=true/false
IR_AVAILABILITY_CLOSED=true/false
SEMANTIC_IDENTITY_CLOSED=true/false
SNAPSHOT_CACHE_COHERENT=true/false
MINING_MODE_AUTHORITY_CLOSED=true/false
LABEL_FEATURE_SEPARATION_CLOSED=true/false
LEGACY_BYPASS_CLOSED=true/false
```

---

# 2. Relation operator：所有多 panel 输入必须 strict axes

当前 relation 模块存在两套不一致哲学：

```text
_stack_panels()
→ exact axes fail closed

relation_category_share()
→ value.reindex_like(category)
```

以及若干函数直接：

```text
panel1.to_numpy()
panel2.to_numpy()
```

做位置运算。

## R24-004

禁止所有 public relation operator自行：

```text
reindex_like
reindex
align with outer/inner
position-only numpy pairing
```

除非 operator经济定义明确就是 reindex transform。

## R24-005

统一使用：

```python
strict_relation_align(*panels)
```

检查：

```text
DataFrame type
index exact equality
columns exact equality
unique index
unique columns
timezone equality
instrument identity
```

## R24-006

重点逐个查：

```text
relation_category_share
relation_peer_weighted_mean_ex_self
relation_weighted_change
event_cumulative_return_past
event_abnormal_return_past
event_since_last_*
index_weight_change
index_membership_age
所有 relation/event 多 panel canonical
```

## R24-007

新增 mutation test：

```text
随机打乱 secondary.columns
```

结果必须：

```text
raise alignment error
```

而不是正常出数。

---

# 3. `relation_category_share` 输入语义错误

当前公式本质：

```text
value_of_category
/
sum(value over observed categories)
```

这只有在 value 是非负权重/活动量时天然叫“share”。

## R24-008

输入必须限制：

```text
NonNegativeWeight
NonNegativeActivity
NonNegativeAmount
```

## R24-009

若需要 signed value：

另建：

```text
relation_category_signed_contribution
```

其输出允许：

```text
<0
>1
```

不要再叫 share。

## R24-010

增加：

```text
output_domain=bounded_0_1
```

到真正的 category share。

---

# 4. Relation HHI：当前存在两个不同经济定义

当前至少存在：

```text
source aggregation:
Σ(raw company ownership ratio_i²)

operator relation_hhi:
先按 observed top-k subtotal 归一化，再 Σ share_i²
```

## R24-011

这两个不是同一个 HHI。

分别命名：

```text
holder_company_ownership_hhi
holder_observed_topk_hhi
```

## R24-012

第一个：

```text
权重分母 = 公司总股本
```

## R24-013

第二个：

```text
权重分母 = observed top-k subtotal
```

## R24-014

禁止 alias 到同一个 canonical。

---

# 5. Relation missing holder rows 不能默认 0

当前部分 HHI/entropy/topk：

```text
nan_to_num(0)
nansum
```

## R24-015

新增：

```text
HolderRankMissingSemantic
```

至少：

```text
STRUCTURAL_ZERO
NOT_REPORTED
SOURCE_MISSING
OUTSIDE_TOP_K
UNKNOWN
```

## R24-016

只有：

```text
STRUCTURAL_ZERO / OUTSIDE_TOP_K
```

允许按 0。

## R24-017

`NOT_REPORTED/SOURCE_MISSING/UNKNOWN`：

必须：

```text
NaN / coverage fail
```

## R24-018

输出同时可生成：

```text
known_rank_count
known_ownership_ratio
snapshot_coverage
```

供 mining gate。

---

# 6. Holder 原始单位禁止“看数值猜百分比”

当前 `aggregate_holder_rows()` 存在：

```text
max(abs(ratio)) > 1
→ /100
else
→ identity
```

## R24-019

彻底删除 data-value heuristic。

## R24-020

单位只允许来自：

```text
FieldSpec.source_unit
ProviderBinding.source_unit
SourceContract
```

## R24-021

测试：

```text
raw percent values = [0.3, 0.8]
```

必须正确解释：

```text
0.3% / 0.8%
```

不能变成：

```text
30% / 80%
```

## R24-022

测试混合错误输入：

```text
[0.008, 2.0]
```

必须 source-contract error，而不是整体缩放。

---

# 7. Holder entity identity 不得静默退到 name

当前候选：

```text
holder_entity_id
entity_id
holder_name
```

## R24-023

需要区分：

```text
EntityStableId
EntityDisplayName
```

## R24-024

涉及跨期实体追踪的 canonical：

```text
holder_churn
holder_entry_exit
network
centrality
ownership persistence
holder turnover
```

必须要求：

```text
EntityStableId
```

## R24-025

只有静态同 snapshot 聚合：

```text
concentration
topk sum
```

才可在明确声明下使用 display name fallback。

## R24-026

同名不同主体测试：

```text
holder_name identical
entity_id different
```

不得 merge。

---

# 8. TopTen 官方 rank 优先，不要重新按 amount 发明 rank

A股源已有：

```text
ShareholderRank
```

## R24-027

官方 TopTen snapshot：

默认按：

```text
source ShareholderRank
```

选择 1..10。

## R24-028

如果 source rank与 amount sorting冲突：

记录 DQ warning。

## R24-029

不能静默重排后声称还是“官方前十大”。

---

# 9. Relation snapshot revision 必须 bitemporal

当前 `relation_snapshot_change` 有类似：

```text
sort available_at
drop_duplicates(instrument,snapshot_id,keep=last)
```

风险：

```text
旧 snapshot 后续被 revision
→ 初始 vintage 被删除
→ 历史回放看到未来修订
```

## R24-030

保留：

```text
snapshot_id
available_at
revision_id/vintage_id
```

三维 identity。

## R24-031

选择规则：

```text
decision_time
→ visible vintages
→ specific snapshot
→ latest visible revision for that snapshot
```

## R24-032

禁止全样本先 keep-last。

---

# 10. Relation snapshot staleness

当前：

```text
max_age_days=None
```

可以永久 ffill。

## R24-033

所有 snapshot relation output必须携带：

```text
snapshot_available_at
snapshot_age
snapshot_period
snapshot_vintage
```

## R24-034

production mining默认必须有：

```text
max_snapshot_age
```

或显式：

```text
allow_unbounded_staleness=True
```

## R24-035

unbounded不能是默认。

---

# 11. Index membership age：provider gap 不能伪造重新纳入

当前：

```text
membership NaN
→ last_entry=-1
```

恢复后仍 member：

```text
age重新从0
```

## R24-036

维护：

```text
last_confirmed_membership_state
last_confirmed_entry
unknown_gap_start
```

## R24-037

gap恢复且 membership 与之前一致：

不能 reset entry age。

## R24-038

若 gap内是否退出无法证明：

输出：

```text
age=NaN
```

直到状态重新得到足够证明，

或者提供：

```text
lower_bound_age
```

另一个 typed output。

---

# 12. PIT helper 自己存在 selector semantic divergence

`pit_asof_join`：

```text
latest available_at row
```

`select_visible_row_bundles`：

```text
all visible
→ latest report period
→ latest revision of that period
```

## R24-039

这两个不是同一语义。

## R24-040

拆成明确 API：

```text
select_latest_event_by_knowledge_time
select_latest_fiscal_period_state
select_specific_period_vintage
select_revision_event_stream
```

## R24-041

禁止一个 generic `pit_asof_join` 同时承担所有 event/financial/relation semantics。

---

# 13. Old-period restatement 对 generic asof 的污染

例：

```text
Q3 已可见
今天重述 Q1
```

generic：

```text
latest available_at
→ Q1
```

普通 latest financial state错误倒退。

## R24-042

增加 golden：

```text
latest period remains Q3
updated period is Q1
```

---

# 14. `_period_mask` 不能用 calendar 12/31 判断 annual

当前 annual selector若按：

```text
month=12 day=31
```

## R24-043

US非12月财政年度会选错。

## R24-044

annual/quarterly selector必须优先用：

```text
timeframe
fiscal_year
fiscal_quarter
```

## R24-045

53-week/transition period同样不能靠日期。

---

# 15. Availability precision 禁止通过 00:00:00 猜

当前某 same-day precision逻辑：

```text
timestamp time == midnight
→ date-only
```

## R24-046

新增：

```python
AvailabilityPrecision:
    DATE
    TIMESTAMP_SECOND
    TIMESTAMP_MILLISECOND
    TIMESTAMP_NANOSECOND
    SESSION_LABEL
    UNKNOWN
```

## R24-047

precision只能来自 source metadata/schema。

## R24-048

禁止通过数据值猜 precision。

---

# 16. Four-layer PIT gate 默认不能 `True`

当前若调用方省略 table/dataset/operator evidence，

部分 API仍默认：

```text
True
```

## R24-049

改 tri-state：

```text
PROVEN_TRUE
PROVEN_FALSE
UNKNOWN
```

## R24-050

production：

```text
UNKNOWN → reject
```

## R24-051

任何 omitted layer：

不能当 PASS。

---

# 17. Financial loader production mode没有真正传到底

当前 `load_financial_row_bundle()` 调 selector时未显式：

```text
production=True
market_calendar
market_timezone
```

## R24-052

production path必须从顶层传：

```text
ExecutionContext
```

不能 helper自己默认 research。

## R24-053

所有 temporal helper签名禁止：

```text
production=False
```

作为生产可达默认。

推荐：

```text
execution_context.required
```

---

# 18. Financial loader 丢 knowledge_at / market_visible_at

底层生成：

```text
knowledge_at
market_visible_at
```

上层返回时又丢掉。

## R24-054

所有 PIT projection必须保留：

```text
raw knowledge_at
market_visible_at
decision_at
```

## R24-055

不要只保留shift后的：

```text
available_at
```

否则：

```text
same-day proof
age
revision latency
debug lineage
```

都不可恢复。

---

# 19. Fundamental period selector约束逻辑要做 set containment

当前存在：

```text
required=annual_only
requested=latest_visible_period
→ 被放行
```

## R24-056

定义 selector lattice。

例如：

```text
annual_only ⊂ latest_visible_period
quarterly_only ⊂ latest_visible_period
specific_timeframe ⊂ latest_visible_period
```

## R24-057

调用方只能：

```text
保持或收紧
```

不能从 narrow contract放宽。

---

# 20. Fundamental field contract不能只注册 A股

## R24-058

把：

```text
ASHARE_FIELD_SPECS
US_FIELD_SPECS
```

统一注册进：

```text
market-scoped financial field contract registry
```

## R24-059

key：

```text
market + field_id
```

不要裸 field name。

---

# 21. Physical source_name 不能绕过 canonical contract

## R24-060

所有 source field resolution最后必须落到：

```text
qualified FieldId / ConceptId
```

## R24-061

禁止：

```text
FUNDAMENTAL_FIELD_CONTRACTS.get(raw_name)
not found
→ treat as ad-hoc
```

---

# 22. NormalizedFieldPlan 正在丢 temporal metadata

当前 plan没有完整携带：

```text
temporal_model
period_id
timeframe
availability precision
vintage
```

## R24-062

扩展：

```python
NormalizedFieldPlan
```

至少加入：

```text
market
concept_id
field_id
temporal_model
knowledge_time_column
effective_time_column
period_id_column
revision_columns
availability_precision
availability_expr
timeframe
flow_semantics
source_vintage
snapshot_policy
missing_semantic
```

---

# 23. `_resolve_table_spec` blanket except 不能 fail-open

当前类似：

```python
except Exception:
    return None
```

随后某些逻辑：

```text
None
→ current_snapshot_only=False
```

## R24-063

区分：

```text
TABLE_NOT_REGISTERED
REGISTRY_ERROR
AMBIGUOUS
```

## R24-064

production：

```text
REGISTRY_ERROR / AMBIGUOUS → hard fail
```

---

# 24. MissingSemantic 缺 `OUT_OF_COVERAGE`

当前：

```text
current_snapshot historical absence
```

可能被归：

```text
NOT_APPLICABLE
```

## R24-065

新增：

```text
OUT_OF_COVERAGE
```

## R24-066

区分：

```text
NOT_APPLICABLE
= 经济上不适用

OUT_OF_COVERAGE
= source没有该时间/标的历史
```

## R24-067

历史 mining遇：

```text
OUT_OF_COVERAGE
→ source block
```

不是普通 NaN处理。

---

# 25. Coverage contract identity过粗

当前 coverage registry只按：

```text
field name
```

## R24-068

改：

```text
market
dataset
provider
field_id/concept_id
timeframe
universe_id
source_version
```

联合 key。

## R24-069

A revenue 与 US revenue不得共享。

## R24-070

US quarterly 与 TTM不得共享。

---

# 26. Coverage overall gate不应盖过 requested window

当前：

```text
先全历史 overall coverage
再 window-specific coverage
```

## R24-071

如果用户只请求：

```text
2025–2026
```

不能因为 2010年代 coverage低直接否决，

除非 policy明确要求 full-history minimum。

## R24-072

分开：

```text
requested_window_coverage
full_history_coverage
```

---

# 27. Coverage by stock必须按当前 universe算

当前 stock quantile如果基于 contract全体股票：

## R24-073

对于：

```text
CSI300
small-cap universe
US S&P500
```

verdict可能不对应本次 campaign。

## R24-074

coverage evidence必须带：

```text
universe_id
instrument_filter_digest
```

---

# 28. snapshot_now_only 不能由调用方 bool 自证

当前：

```text
snapshot_now_only=True
```

可绕 current_snapshot历史 gate。

## R24-075

source必须提供：

```text
snapshot_valid_at
snapshot_id
snapshot_created_at
```

## R24-076

runtime自行验证：

```text
requested decision range
⊆ snapshot validity
```

## R24-077

caller boolean只能表达 intent，不能表达 proof。

---

# 29. `_window_is_historical` 不应以机器 today 为权威

## R24-078

历史/当前判断应该相对：

```text
source snapshot validity
decision context
```

不是：

```text
datetime.now()
```

---

# 30. Composite child 没 snapshot token时不能 verified=True

当前如果 child没有 token：

可能形成：

```text
version=None
verified=True
```

前后：

```text
None == None
```

被视为一致。

## R24-079

生产：

```text
no snapshot token
→ unverifiable
```

## R24-080

可接受替代证明：

```text
immutable content hash
transaction id
MVCC version
file manifest digest
dataset snapshot id
```

---

# 31. Composite cache manifest 有 read-epoch 错配风险

危险顺序：

```text
read A
barrier verifies A
_record_snapshot_manifest(refresh=True)
source advances B
manifest recorded B
cache data still A
```

下一次：

```text
B == B
→ old A cache survives
```

## R24-081

manifest必须记录：

```text
the exact epoch verified for the read
```

不能读后再 refresh换 epoch。

## R24-082

如果 read后 source epoch变化：

```text
invalidate + retry
```

## R24-083

增加并发 deterministic test：

```text
A read
A→B exactly between verify and manifest record
```

不能产生：

```text
data=A, manifest=B
```

---

# 32. PIT-sensitive source识别不能靠 `.dataset`

当前若 source无：

```text
dataset
```

可能直接认为：

```text
not PIT-sensitive
```

## R24-084

定义统一：

```python
DataSource.temporal_contract()
```

返回：

```text
TemporalSensitivity
SnapshotCapability
JoinCapability
```

## R24-085

所有 DataSource必须实现。

---

# 33. Composite asof 只适合普通 timestamp source

Composite generic asof只有：

```text
timestamp
instrument
```

无法表达：

```text
knowledge
effective
period
revision
```

## R24-086

PIT-sensitive source继续禁止 generic composite asof。

## R24-087

但检测必须来自：

```text
temporal_contract
```

而不是 dataset name blacklist。

---

# 34. Cross-sectional neutralize 缺 group时不能静默换定义

当前某路径：

```text
group absent/all NaN
→ global demean
```

## R24-088

production默认：

```text
missing_group_policy="nan"
```

或：

```text
"error"
```

## R24-089

`global_demean` 只允许显式配置。

## R24-090

policy必须进：

```text
ParamSpec
factor identity
cache key
```

---

# 35. CrossSectionalUniverseContract

所有：

```text
rank
zscore
percentile
mean
std
demean
neutralize
group stats
cs regression
```

必须声明当前横截面是谁。

## R24-091

新增：

```python
CrossSectionalUniverseContract
```

字段：

```text
universe_id
tradability_filter
missing_field_policy
suspension_policy
new_listing_policy
group_missing_policy
min_valid_names
weighting
```

## R24-092

横截面输出 identity必须含 universe digest。

---

# 36. row_beta / row_corr 等必须 exact universe

## R24-093

multi-panel cross-sectional：

```text
y
x
weights
groups
```

先 strict align，

不得靠 Pandas交集自动修。

---

# 37. group operator hidden fallback_policy

多个 group operator：

```text
fallback_policy
```

并不总在：

```text
param_names / ParamSpec
```

里。

## R24-094

所有行为参数必须显式进入 contract。

## R24-095

strict enum：

```text
nan
error
global
keep_original
```

## R24-096

未知字符串：

```text
raise
```

不能落到默认 else branch。

---

# 38. group operator仍有 reindex

## R24-097

重点：

```text
group_ts_decay_linear
group_demean
group_mean
group_sum
group_rank_weighted_value
```

统一 strict axes。

---

# 39. `group_demean(group=None)` 不应是 Direct typed definition

legacy可以：

```text
group=None → global demean
```

## R24-098

但 DirectUse canonical：

```text
group_neutralize/group_demean
```

必须要求：

```text
GroupKey
```

## R24-099

无group的行为放 compatibility alias。

---

# 40. implementation-boundary 参数 strictness仍不统一

重点已发现：

```text
event_refractory cooldown
index membership windows
group_ts_decay window
survival history_window/min_completed
label horizon
PIT max_age
```

还有：

```text
max(... int(...))
```

## R24-100

静态扫描所有 public kernel：

```regex
int\(
max\([^,\n]+,\s*int\(
min\([^,\n]+,\s*int\(
```

## R24-101

每一个 public scalar都必须通过统一 strict validator，

不能假设上层永远调用 ParamSpec。

---

# 41. Stateful gap semantics不能统一 reset

目前很多：

```text
NaN
→ reset/rebaseline state
```

## R24-102

区分：

```text
UNKNOWN_OBSERVATION
PROVIDER_GAP
NOT_TRADING
NOT_APPLICABLE
EXPLICIT_RESET
```

## R24-103

不同 state machine选择不同：

```text
BREAK
CENSOR
CARRY_WITH_UNKNOWN
RESET
```

## R24-104

missing policy必须进入 factor identity。

---

# 42. Survival operator有 left censor bias

若样本第一行：

```text
state=active
```

真实 run可能早已开始。

当前后续结束时会把该 run作为完整 completed episode。

## R24-105

初始 active run标：

```text
left_censored=True
```

## R24-106

不得进入：

```text
completed duration distribution
```

除非观察到真实 episode start。

---

# 43. Survival gap应考虑 censor，不只是 drop run

## R24-107

provider gap中：

```text
run length unknown
```

## R24-108

可输出：

```text
age_lower_bound
age_exact_known
```

而不是恢复后简单从1开始。

---

# 44. Survival inactive=0 与 not-applicable混淆

当前：

```text
inactive
→ percentile=0
hazard=0
residual=0
```

## R24-109

明确 output contract：

```text
inactive_policy
```

推荐：

```text
NaN + active_mask
```

或拆：

```text
value
applicable
```

---

# 45. IR SemanticType 仍表达不完整

当前已有：

```text
FinancialStock
FinancialSinglePeriodFlow
FinancialCumulativeYTDFlow
FinancialTTMFlow
```

缺：

```text
FinancialAnnualFlow
FinancialRatio
FinancialRate
FiscalPeriodId
KnowledgeTimestamp
EffectiveTimestamp
RevisionTimestamp
ComponentStack
ApplicabilityMask
CoverageMask
OutOfCoverage
```

## R24-110

扩展类型系统。

---

# 46. `single_period_flow` 不足以同时表示 quarter 和 annual

## R24-111

新增 duration dimension：

```text
PeriodDuration
```

至少：

```text
quarter
half_year
annual
ttm
ytd
irregular
unknown
```

## R24-112

US annual不能仅标为普通 single-period后与季度数据无差别组合。

---

# 47. Typed input contract覆盖率太低

当前 `OPERATOR_INPUT_TYPE_CONTRACTS` 只列极少 operator。

## R24-113

动态计算：

```text
typed_contract_coverage =
operators_with_panel_inputs_and_semantic_constraints
/
retained_public_operators_requiring_typed_constraints
```

## R24-114

目标：

```text
100%
```

不是所有 generic math都必须限制，

但所有经济上有特定输入含义的 operator必须限制。

---

# 48. Production unknown semantic kind已经有 fail-closed，但入口没有打开

Analyzer支持：

```text
strict_unknown=True in production
```

这是对的。

## R24-115

所有 production入口必须真的：

```text
Analyzer(production=True, market=...)
```

否则这个 gate形同虚设。

---

# 49. FilingDate 的 alias fallback存在硬 PIT 漏洞

当前 `FilingDate.resolve()` 可能尝试：

```text
filing_date
announcement_date
filingdate
report_date
self.reference
```

默认 `self.reference` 还是：

```text
ReportPeriodEndDate
```

## R24-116

删除：

```text
report_date
ReportPeriodEndDate
```

作为 filing knowledge-time fallback。

## R24-117

找不到 filing timestamp/date：

```text
UNKNOWN
```

## R24-118

绝不：

```text
period_time → knowledge_time
```

---

# 50. NextTradingOpen/Day 不得拿 decision time补未知 PubDate

当前 row缺 publication reference时：

可能 fallback：

```text
decision_context.asof/date
```

## R24-119

删除该 fallback。

## R24-120

knowledge reference未知：

```text
fail closed
```

---

# 51. Calendar unavailable不能用硬编码交易时间

当前有默认：

```text
09:30
14:57
15:00
11:30
```

## R24-121

production：

```text
calendar/session schedule unavailable
→ UnknownAvailability / error
```

## R24-122

research可显式：

```text
allow_calendar_fallback=True
```

但不得默认。

---

# 52. 早收市 / 不同市场

## R24-123

测试：

```text
US early close 13:00
A股常规15:00
minute lunch break
different timezone
```

## R24-124

不能一套固定 wall-clock。

---

# 53. `_localize` timezone失败不能返回 naive

当前异常：

```text
except:
    return dt
```

## R24-125

改：

```text
raise TemporalResolutionError
```

production fail closed。

---

# 54. AvailabilityExpr必须贯穿整个 IR DAG，不能退回 label

当前已有 `MaxAvailability`，

但 analyzer operator propagation仍主要保存：

```text
available_at string label
```

而未保持完整 child expression DAG。

## R24-126

operator semantic：

```text
availability_expr = latest_availability_expr(child_exprs)
available_at = availability_expr.label # compatibility only
```

## R24-127

runtime/PIT admission只用：

```text
availability_expr
```

禁止用 label做最终时点判断。

---

# 55. 旧财报 + 今日行情 availability golden

输入：

```text
financial PubDate = 2025-01-01
today close = 2026-08-10 15:00
```

## R24-128

root：

```text
available = 2026-08-10 15:00
```

而不是因 `PubDate` descriptor静态 lateness较大就选 financial descriptor。

---

# 56. `Schema.pit_safe=True` 默认违背 fail-closed

## R24-129

改：

```text
pit_safe: bool | None = None
```

## R24-130

DEFAULT schema：

```text
UNKNOWN
```

research raw compatibility例外显式授权。

---

# 57. `semantic_attrs` 不能整体排除 factor identity

当前：

```text
IRNode.semantic_attrs compare=False/hash=False
PlanNode.semantic_attrs compare=False
```

## R24-131

现在 semantic_attrs包含：

```text
market-specific semantics
flow semantics
price basis
source vintage
universe
availability
```

这些已经改变经济定义。

## R24-132

不要直接 hash 全 dict。

新增稳定：

```python
SemanticIdentityDigest
```

---

# 58. SemanticIdentityDigest

## R24-133

至少：

```text
market
concept/field identity
provider/dataset
unit semantic
price basis
flow semantic
period duration
timeframe
universe
availability policy
knowledge-time policy
revision policy
source vintage/snapshot
missing policy
group fallback
```

## R24-134

不应包含：

```text
debug notes
description
cost estimate
display tags
```

---

# 59. Semantic digest进入所有身份

## R24-135

必须进入：

```text
IR canonical identity
Plan cache key
CSE key where semantically relevant
FactorId
MaterializationId
Checkpoint key
Incremental state key
Cold-start dedup key
Mining candidate dedup
```

---

# 60. SourceRef v1装不下当前语义

当前 SourceRef identity缺：

```text
market
provider
dataset
catalog hash
timeframe
availability policy
vintage
```

## R24-136

设计：

```text
SourceRef v2
```

## R24-137

兼容读 v1，

production新生成只发 v2。

---

# 61. SourceRef v2至少包含

## R24-138

```text
market
concept_id/field_id
table
physical_field
provider_id
dataset
required filters
timeframe
transform
transform params
temporal policy digest
catalog hash
provider contract hash
dialect version
```

---

# 62. SourceRef decode production strictness不能丢

当前 decode重建默认：

```text
production=False
```

## R24-139

新增：

```python
decode_source_ref(name, context=ExecutionContext)
```

## R24-140

production decode：

```text
unknown transform
unknown param
old unapproved dialect
→ reject
```

---

# 63. minute SourceTransform参数还不够严格

当前：

```text
minute_at.hhmm = any string
minute_range.start/end = any string
```

## R24-141

验证：

```text
HH:MM syntax
00<=HH<=23
00<=MM<=59
```

## R24-142

再由 session calendar验证：

```text
inside session
not lunch break
start <= end
```

## R24-143

minute `index/period`关系也要 relational validation。

---

# 64. Provider continuous price certification与 FieldSpec冲突

当前 provider层：

```text
A Factor backward multiplier verified
source_certified=True
```

而某 FieldSpec仍：

```text
direction=unverified
adjustment_status=unverified
```

## R24-144

单一权威：

```text
AdjustmentFactorCertificate
```

## R24-145

FieldSpec/ProviderBinding只能引用该证书，

不能各自写相反状态。

---

# 65. Backward adjustment factor的 vintage PIT要单独证明

即使：

```text
adjusted = raw * factor
```

公式正确，

也要证明今天读取的历史 factor没有包含未来 corporate actions。

## R24-146

新增：

```text
adjustment_factor_vintage_pit
```

## R24-147

若 vendor历史 back-adjusted factor会随未来拆分/分红重算：

```text
continuous historical level
```

不得作为严格 PIT raw feature。

---

# 66. Return因子与continuous level分开认证

## R24-148

某些 return ratio可能对统一 scale不敏感，

但：

```text
price level
distance
support/resistance
cross-sectional adjusted price
```

可能受未来 normalization影响。

逐 family认证。

---

# 67. raw_open/raw_pre_close availability不应统一 local_close

## R24-149

字段级：

```text
pre_close
→ previous session known / before open

open
→ after open auction

high/low/close/vwap/volume/amount
→ end of aggregation window
```

## R24-150

不要批量循环赋同一个 available_at。

---

# 68. Concept role错误：index_weight不是 GroupKey

## R24-151

`index_weight`：

```text
NonNegativeWeight
```

不是：

```text
GroupKey
```

## R24-152

`index_member`：

应：

```text
MaskBool / MembershipBool
```

而不是普通 status string。

---

# 69. cross-market comparable不能只看 unit

例如：

```text
ROE
margin
valuation
```

## R24-153

`cross_market_comparable=True`需要同时证明：

```text
economic definition
accounting attribution
timeframe
annualization
currency/dimensionless
source vintage
```

## R24-154

否则改：

```text
comparable_with_normalization
```

---

# 70. FinancialPeriodAdapter.filter_timeframe fail-open

当前：

```text
if timeframe column absent
→ return frame unchanged
```

## R24-155

US financial production：

```text
missing timeframe column
→ raise
```

## R24-156

timeframe严格 enum。

---

# 71. Financial asof方向必须固定 backward

## R24-157

production：

```text
direction != backward
→ reject
```

不能暴露：

```text
nearest
forward
```

---

# 72. Production DSL validator实际上没有启 production Analyzer

当前：

```python
validate_production_dsl()
→ Analyzer().lower(...)
```

## R24-158

修成：

```python
Analyzer(
    production=True,
    market=resolved_market,
).lower(..., production=True)
```

## R24-159

`market`变成 required参数。

---

# 73. production DSL bare-field检查也不能用 legacy A registry

当前：

```text
FIELD_REGISTRY
```

## R24-160

改：

```text
MULTI_MARKET_FIELD_REGISTRY.registry_for(market)
```

---

# 74. validate manifest必须传 market进 production validator

## R24-161

所有：

```text
validate_manifest_for_execution
validate_formula_in_mining_allowlist
fastpath validator
```

统一 market/context authority。

---

# 75. default typed mining search space缺 market

当前：

```text
fields=None
→ legacy FIELD_REGISTRY
```

## R24-162

新增 required：

```text
market
```

## R24-163

字段来自：

```text
MULTI_MARKET_FIELD_REGISTRY.registry_for(market)
```

---

# 76. typed mining field metadata太少

当前主要导：

```text
name
frequency
domain
cardinality
unit
temporal_model
strict_pit
```

## R24-164

必须再导：

```text
market
concept_id
field_id
semantic_kind
price_basis
flow_semantics
period duration
knowledge model
availability_expr
availability precision
source vintage
timeframe
coverage
applicability
```

---

# 77. typed mining operator inputs仍有旧 `inputs=params`

R22已指出过该问题，本轮只验收，不重新设计。

## R24-165

确认最终：

```text
inputs == panel/data inputs only
scalar_parameters separate
```

---

# 78. mining role异常不能 `except Exception: pass`

R22已指出，本轮作为 legacy bypass gate。

## R24-166

生产 search-space生成过程中：

```text
role resolution failure
→ generation failure
```

---

# 79. default A-share universe preset schema漂移

当前 preset使用类似：

```text
StockStatus.ListedState
```

而当前 catalog物理字段为：

```text
PublicStatus
```

## R24-167

禁止 preset硬编码 physical字段名。

## R24-168

preset必须：

```text
concept/provider resolver
```

生成。

---

# 80. A-share universe preset描述“停牌”但没真正用 IsSuspend

## R24-169

将 universe/tradability拆清楚：

```text
listing status
ST/risk status
suspension
daily price availability
index membership
```

## R24-170

不要一个 `status` source名字暗示全部过滤已经完成。

---

# 81. Legacy fundamental mining presets必须关旁路

当前还暴露：

```text
financials_ratios
fundamentals_balance_sheet
fundamentals_cash_flow_statement
fundamentals_income_statement
```

## R24-171

每个先拿：

```text
TemporalSourceCertificate
```

## R24-172

没有：

```text
knowledge time
period
timeframe
revision/vintage
```

就：

```text
research only
```

不能进入 production default registry。

---

# 82. Default mining presets只能由 certified registry生成

## R24-173

不要手写一批 dict继续漂移。

建立：

```python
build_certified_mining_preset(
    market,
    concepts,
    context,
)
```

---

# 83. Label layer：forward target和 feature DSL必须分开

当前：

```text
build_forward_return_series()
```

是真 forward return，

但：

```text
label_formula = ts_pct(close,horizon)
```

更像 backward pct-change。

## R24-174

不要让普通 feature DSL承担标签未来语义。

新增：

```text
LabelExpr
LabelIR
```

---

# 84. LabelIR允许受控未来，FeatureIR禁止

## R24-175

LabelIR专门允许：

```text
ForwardReturn
ForwardExcessReturn
ForwardResidualReturn
```

## R24-176

FeatureIR任何 forward op：

```text
forbidden
```

---

# 85. default label formula必须与真实 builder同义

## R24-177

不能一个 config写：

```text
ts_pct(close,h)
```

runtime却用：

```text
close[t+h]/close[t]-1
```

## R24-178

唯一 authority。

---

# 86. Label raw close可能被公司行动污染

## R24-179

默认 target优先：

```text
return_decimal
```

聚合：

```text
Π(1+r)-1
```

或已证明 PIT-safe的 return series。

## R24-180

raw close forward ratio只有在：

```text
corporate-action-safe
```

才允许。

---

# 87. Label horizon参数 strict

## R24-181

删除：

```text
max(1,int(horizon))
```

使用 strict positive int。

---

# 88. `gap_bars` 不能转成 calendar Timedelta

当前：

```text
pd.Timedelta(days=gap_bars)
```

## R24-182

使用：

```text
TradingCalendar.shift_sessions
BarIndex.shift
```

## R24-183

分钟：

```text
bar count
```

不是 day。

---

# 89. Label alignment必须按 sample，不是只看全局 max/min

## R24-184

构造训练样本 `(t,label_t+h)` 时逐行验证：

```text
feature_available_at(t) < label_interval_start/end
```

以及：

```text
train split embargo/purge
```

---

# 90. Label decision time也要明确

## R24-185

例如：

```text
close-to-close 5d label
```

feature若用 t close，

label起点是：

```text
t close
```

是否需要 gap=0/1必须由策略定义，

不是写死一个整数。

---

# 91. Label delisting/suspension处理

## R24-186

forward horizon中：

```text
delist
suspend
missing close
corporate action
```

不能简单 shift跳过。

## R24-187

需要：

```text
label_missing_policy
terminal_return_policy
```

---

# 92. `SourceVintageSpec` 目前信息不足

当前只含：

```text
knowledge_time
revision_time
version_id
snapshot_id
```

## R24-188

扩：

```text
market
provider
dataset
timeframe
period identity
availability precision
policy hash
```

或者：

```text
SourceTemporalIdentity
```

包裹它。

---

# 93. `revision_columns` 不能自动猜“第一个=revision_time，最后一个=version_id”

## R24-189

FieldSpec应该明确：

```text
revision_time_column
revision_id_column
revision_sequence_column
```

不要靠 tuple位置解释。

---

# 94. AvailabilityExpr 的 effective date 与 knowledge date不要混成一个抽象

目前：

```text
ExDate
RecordDate
PaymentDate
EffectiveDate
```

都继承 AvailabilityExpr。

## R24-190

区分：

```text
KnowledgeAvailabilityExpr
EconomicEffectiveExpr
```

## R24-191

ex-date本身不是“什么时候市场知道 ex-date”的时间。

如果 source有 declaration：

```text
knowledge=declaration
effective=ex-date
```

---

# 95. Corporate action事件需要双时间

## R24-192

typed event：

```text
announced_event
effective_event
```

两个 grammar slot。

---

# 96. Cross-sectional current universe与 data missing分开

## R24-193

一个股票：

```text
in universe but field missing
```

和：

```text
not in universe
```

不能都是 NaN后随便 rank skip。

## R24-194

显式：

```text
universe_mask
field_validity_mask
```

---

# 97. Rank因子必须记录样本数

## R24-195

最低：

```text
min_cross_section
```

## R24-196

极小截面：

```text
rank
zscore
quantile
regression
```

fail/mask。

---

# 98. Group sample size

## R24-197

行业只有：

```text
1–2 stocks
```

group zscore/regression不应输出看似正常数。

每个 group operator声明：

```text
min_group_size
```

---

# 99. Weight semantics

所有 weighted operator：

```text
weight
```

必须：

```text
finite
>=0
sum>0
```

除非明确 signed weights。

## R24-198

typed：

```text
NonNegativeWeight
SignedWeight
```

分开。

---

# 100. EventBool 与 SignedEvent分开

当前部分 event kernel：

```text
non-zero = event
```

## R24-199

如果输入 contract是 EventBool：

只允许：

```text
0,1,NaN
```

## R24-200

如果要多方向事件：

```text
SignedEvent {-1,0,1,NaN}
```

单独 type。

---

# 101. Stateful initial condition必须进入 factor identity

例如：

```text
state_latch initial_state
```

已是 param。

但：

```text
missing policy
left censor policy
warm-start state
```

同样影响结果。

## R24-201

都必须 identity化。

---

# 102. Stateful chunk/full-run parity

## R24-202

针对：

```text
latch
hold
refractory
membership age
survival
state duration
```

做：

```text
full run
vs
chunk + serialized state
```

parity。

没有 checkpoint support：

```text
required_full_history
```

保持诚实。

---

# 103. Temporal state correction

如果历史 source correction在 t之后可知：

## R24-203

state recomputation只从：

```text
correction knowledge time
```

开始，

不能从 economic event period写回。

---

# 104. Adjustment factor correction同理

## R24-204

未来 corporate action引起 adjustment series重算时，

不能让 materialized historical feature在过去日期被静默覆盖。

---

# 105. Cache/semantic identity real collision tests

## R24-205

构造同文本公式：

```text
fin_growth(revenue)
```

上下文：

```text
A YTD
US quarterly
US TTM
```

最终：

```text
semantic digest
factor id
cache key
```

必须不同。

---

# 106. 同文本不同 universe

## R24-206

```text
rank(roe)
CSI300
All-A
```

factor semantic identity不同。

---

# 107. 同文本不同 same-day policy

## R24-207

```text
PubDate next-session
exact timestamp same-day
```

identity不同。

---

# 108. 同文本不同 revision policy

## R24-208

```text
first_available
latest_available_asof
```

identity不同。

---

# 109. 同文本不同 group fallback

## R24-209

```text
missing group → nan
missing group → global demean
```

identity不同。

---

# 110. Semantic identity不参与数学 CSE时要非常小心

某些纯数学相同子树可共享 computation，

但：

```text
source-vintage / universe / availability不同
```

不能跨语义 context共享 materialized result。

## R24-210

拆：

```text
StructuralMathHash
SemanticExecutionHash
```

---

# 111. Plan optimizer不能删除 semantic barrier

## R24-211

任何：

```text
source conversion
PIT projection
quarterization
currency normalization
universe mask
```

即使数值表达式看似 identity，

不能被普通 algebra rewrite消除。

---

# 112. Error handling：不能 silent `except Exception: pass`

本轮重点静态扫描：

```text
ir
fields
storage/sources
api/mining
planner
relation
group
stateful
label
```

## R24-212

分类：

```text
expected optional dependency
expected research fallback
semantic error
unknown bug
```

## R24-213

semantic error production必须向上抛。

---

# 113. Default/mining helper必须与生产核心走同一 resolver

## R24-214

禁止 helper自己：

```text
hardcode physical field
hardcode join
hardcode availability
hardcode market
```

## R24-215

全部通过：

```text
FieldConcept
ProviderBinding
TemporalContract
```

生成。

---

# 114. Static audit：legacy A resolver不能到 production

已有 R17思想，

本轮重点扫描新增旁路：

```text
Analyzer()
FIELD_REGISTRY
resolve_field()
```

## R24-216

production模块出现：

```text
legacy resolve
```

直接 gate fail。

---

# 115. Static audit：semantic metadata降级成字符串

## R24-217

扫描：

```text
availability_expr → label → no expr
source_vintage → str
field_id → bare field name
market context → omitted
```

每处列整改。

---

# 116. Static audit：source metadata看数值猜

禁止：

```text
max(value)>1 → percent
midnight → date-only
column name contains xxx → semantic
```

## R24-218

所有 semantic inference必须：

```text
declared metadata
```

---

# 117. Static audit：hidden behavioral kwargs

查：

```text
fallback_policy
missing_policy
revision_policy
same_day
direction
normalize
production
strict
```

## R24-219

只要改变结果：

必须：

```text
declared Param/PolicySpec
identity
validation
```

---

# 118. Static audit：current snapshot bypass

查：

```text
snapshot_only
snapshot_now_only
current_only
allow_sparse
allow_proxy
```

## R24-220

所有 bypass必须有 source proof。

---

# 119. Static audit：time/bar混用

查：

```text
Timedelta(days=bars)
rows=days
calendar day as trading day
```

## R24-221

统一：

```text
bars
sessions
calendar duration
fiscal periods
events
episodes
```

五种 clock类型。

---

# 120. ClockType系统

## R24-222

定义：

```text
BarClock
SessionClock
CalendarClock
FiscalPeriodClock
EventClock
EpisodeClock
```

## R24-223

window/history/age参数绑定 clock type。

---

# 121. `max_age_days` 名字也要诚实

如果实际按：

```text
decision rows
```

不能叫 days。

改：

```text
max_age_sessions
```

或真实 calendar days。

---

# 122. Real source invariants

## R24-224

A relation真实数据检查：

```text
ShareRatio unit
official rank
duplicate snapshot period
same holder id/name mapping
PubDate ordering
```

## R24-225

US：

```text
fiscal timeframe
early close
snapshot validity
shares snapshot duplicate
```

---

# 123. Corpus-level semantic audits

不是只跑 unit tests。

## R24-226

自动扫描所有 current canonical：

```text
multi-panel without alignment guard
public kernel with int coercion
behavioral kwargs missing ParamSpec
typed-required operator without input contract
stateful operator without missing policy
cross-sectional operator without universe policy
```

---

# 124. Source-level audits

## R24-227

自动扫描：

```text
source contract missing temporal model
PIT source missing knowledge clock
current-only source missing validity interval
relation source missing snapshot/vintage identity
derived source missing lineage
```

---

# 125. IR-level audits

## R24-228

自动扫描：

```text
semantic attrs not propagated
availability expr lost
mixed semantic allowed through constrained operator
unknown type entering production
legacy market resolver in production
```

---

# 126. Cache-level audits

## R24-229

自动扫描：

```text
cache key excludes semantic digest
manifest epoch differs from read epoch
snapshot unknown treated verified
```

---

# 127. Mining-level audits

## R24-230

自动扫描：

```text
production validator using research Analyzer
market missing
legacy A field catalog in US
typed search fields missing semantic kinds
```

---

# 128. Label-level audits

## R24-231

自动扫描：

```text
forward label formula mismatch
raw price corporate-action contamination
bar/day confusion
feature/label overlap
```

---

# 129. 新 artifacts

## R24-232

生成：

```text
factor_engine/docs/R24_SEMANTIC_CONTINUITY_MATRIX.csv
factor_engine/docs/R24_SEMANTIC_CONTINUITY_MATRIX.json
factor_engine/docs/R24_SEMANTIC_CONTINUITY_MATRIX.md
```

每一 semantic field列出：

```text
FieldSpec
Provider
SourceRef
FieldPlan
Schema
IR
Plan
Cache
Mining
Materialization
```

是否保留。

---

# 130. Relation artifacts

## R24-233

```text
factor_engine/docs/R24_RELATION_STATE_AUDIT.csv
factor_engine/docs/R24_RELATION_VINTAGE_MATRIX.json
factor_engine/docs/R24_RELATION_MISSING_SEMANTICS.json
```

---

# 131. Identity artifacts

## R24-234

```text
factor_engine/docs/R24_SEMANTIC_IDENTITY_SPEC.md
factor_engine/docs/R24_SOURCE_REF_V2_SPEC.md
factor_engine/docs/R24_FACTOR_ID_COLLISION_TESTS.json
```

---

# 132. PIT infrastructure artifacts

## R24-235

```text
factor_engine/docs/R24_PIT_SELECTOR_MATRIX.md
factor_engine/docs/R24_AVAILABILITY_EXPR_AUDIT.csv
factor_engine/docs/R24_CLOCK_TYPE_MATRIX.json
```

---

# 133. Mining/Label artifacts

## R24-236

```text
factor_engine/docs/R24_MINING_MODE_AUTHORITY_AUDIT.md
factor_engine/docs/R24_TYPED_SEARCH_SPACE_AUDIT.json
factor_engine/docs/R24_LABEL_IR_SPEC.md
factor_engine/docs/R24_LEGACY_PRESET_AUDIT.csv
```

---

# 134. Snapshot artifacts

## R24-237

```text
factor_engine/docs/R24_SNAPSHOT_COHERENCE_AUDIT.md
factor_engine/docs/R24_CACHE_EPOCH_TEST_REPORT.json
```

---

# 135. 新 test suites

## R24-238

至少：

```text
tests/relation/test_relation_strict_alignment.py
tests/relation/test_holder_ratio_unit_contract.py
tests/relation/test_holder_snapshot_revision_vintage.py
tests/relation/test_index_membership_gap_recovery.py
tests/relation/test_hhi_semantic_split.py

tests/pit/test_selector_semantic_modes.py
tests/pit/test_filingdate_never_falls_back_to_period.py
tests/pit/test_unknown_pubdate_does_not_use_decision_time.py
tests/pit/test_availability_precision_metadata.py

tests/ir/test_availability_expr_survives_operator_dag.py
tests/ir/test_semantic_identity_digest.py
tests/ir/test_schema_pit_unknown_default.py
tests/ir/test_typed_contract_coverage.py

tests/source/test_source_ref_v2_identity.py
tests/source/test_source_ref_production_decode.py
tests/source/test_minute_transform_clock_validation.py

tests/storage/test_snapshot_unknown_not_verified.py
tests/storage/test_snapshot_manifest_exact_read_epoch.py
tests/storage/test_current_snapshot_validity_interval.py

tests/mining/test_production_validator_uses_production_analyzer.py
tests/mining/test_us_search_space_uses_us_registry.py
tests/mining/test_market_required_for_production.py
tests/mining/test_legacy_preset_temporal_certificate.py

tests/label/test_label_ir_forward_return.py
tests/label/test_gap_bars_use_bar_clock.py
tests/label/test_corporate_action_safe_target.py
tests/label/test_feature_label_temporal_separation.py

tests/stateful/test_left_censored_episode.py
tests/stateful/test_provider_gap_censoring.py
```

---

# 136. Cache epoch race test

## R24-239

模拟：

```text
snapshot=A
read A
verify A
source advances B
manifest record
```

最终只允许：

```text
retry
or manifest=A
```

不允许：

```text
cache data=A
manifest=B
```

---

# 137. IR availability expression test

## R24-240

构造：

```text
old filing child
current close child
```

operator root必须持有：

```text
MaxAvailability(children=[...])
```

而不是只：

```text
"filing"
```

---

# 138. SourceRef cross-market collision

## R24-241

构造：

```text
A StockIncome.revenue
US StockIncome.revenue
```

v2 encoded identity必须不同。

---

# 139. Market search-space golden

## R24-242

A search space：

```text
A fields only
```

US：

```text
US fields only
```

hash不同。

---

# 140. Production mode authority golden

## R24-243

一个：

```text
semantic_kind unknown
```

但 operator slot constrained 的 formula：

```text
research validate = allowed/warn
production validate = reject
```

---

# 141. Filing period fallback golden

## R24-244

row：

```text
ReportPeriodEndDate=2024-03-31
filing_date missing
```

`FilingDate.resolve()`：

```text
must fail
```

绝不能：

```text
2024-03-31
```

---

# 142. NextTradingOpen unknown knowledge golden

## R24-245

row没有 PubDate：

```text
decision_context=2026-08-10
```

结果：

```text
UNKNOWN/error
```

绝不能：

```text
2026-08-11 open
```

---

# 143. Group fallback typo golden

## R24-246

```text
fallback_policy="nna"
```

必须：

```text
ValueError
```

不能变 global behavior。

---

# 144. Survival censor golden

## R24-247

dataset start：

```text
active, active, active, inactive
```

第一 run：

```text
left-censored
```

不得加入 completed distribution。

---

# 145. Relation holder unit golden

## R24-248

raw percent：

```text
0.3
0.8
```

FieldSpec says percent：

canonical：

```text
0.003
0.008
```

---

# 146. Label golden

## R24-249

价格：

```text
t close 100
t+5 close 110
```

label：

```text
0.10
```

与配置formula语义完全一致。

---

# 147. Label corporate action golden

拆股：

```text
100 → 50
```

无经济损失：

label不得：

```text
-50%
```

---

# 148. Legacy bypass static gate

## R24-250

production import graph中出现：

```text
legacy resolve_field
FIELD_REGISTRY
generic Analyzer()
generic asof financial
uncertified preset
```

输出 blocking report。

---

# 149. 当前已经确认的重点 concrete findings checklist

## R24-251

确认/整改：

```text
relation_category_share reindex_like
```

## R24-252

```text
relation positional numpy multi-panel without exact axes
```

## R24-253

```text
index_membership_age gap resets entry age
```

## R24-254

```text
relation HHI/entropy/topk missing-as-zero ambiguity
```

## R24-255

```text
holder ratio unit inference from data max
```

## R24-256

```text
holder name fallback identity
```

## R24-257

```text
TopTen official rank re-sorted by amount
```

## R24-258

```text
relation_snapshot_change keep-last destroys historical vintage
```

## R24-259

```text
unbounded relation snapshot ffill
```

## R24-260

```text
generic pit_asof latest-event semantics != latest-period-state semantics
```

## R24-261

```text
calendar-quarter annual selector
```

## R24-262

```text
availability precision guessed from midnight
```

## R24-263

```text
four-layer PIT omitted evidence defaults true
```

## R24-264

```text
financial loader production/context not passed
```

## R24-265

```text
financial loader drops knowledge_at / market_visible_at
```

## R24-266

```text
period selector widening bug
```

## R24-267

```text
financial contracts A-only/bare-name keyed
```

## R24-268

```text
NormalizedFieldPlan temporal metadata lost
```

## R24-269

```text
current snapshot missing wrongly mapped to NOT_APPLICABLE
```

## R24-270

```text
coverage contract field-only key collision
```

## R24-271

```text
coverage overall gate vs requested window mismatch
```

## R24-272

```text
coverage by stock ignores requested universe
```

## R24-273

```text
snapshot_now_only caller bypass
```

## R24-274

```text
Composite no snapshot token treated verified
```

## R24-275

```text
Composite cache data epoch != recorded manifest epoch race
```

## R24-276

```text
PIT-sensitive detection depends on dataset attr/heuristics
```

## R24-277

```text
neutralize missing group silently becomes global demean
```

## R24-278

```text
group reindex and hidden fallback policy
```

## R24-279

```text
stateful left censor / gap semantics
```

## R24-280

```text
FilingDate period fallback
```

## R24-281

```text
NextTradingOpen decision-time fallback for unknown publication
```

## R24-282

```text
hard-coded session wall clocks
```

## R24-283

```text
timezone localization failure returns naive datetime
```

## R24-284

```text
AvailabilityExpr DAG degraded to label
```

## R24-285

```text
Schema PIT default true
```

## R24-286

```text
semantic_attrs excluded from IR/Plan identity
```

## R24-287

```text
SourceRef missing market/provider/temporal identity
```

## R24-288

```text
SourceRef production strictness lost on decode
```

## R24-289

```text
minute transform time syntax/session not validated
```

## R24-290

```text
continuous price provider certification conflicts with field catalog
```

## R24-291

```text
adjustment factor historical vintage PIT unproven
```

## R24-292

```text
index_weight typed as GroupKey
```

## R24-293

```text
FinancialPeriodAdapter missing timeframe fail-open
```

## R24-294

```text
production validator uses Analyzer() research/default market
```

## R24-295

```text
typed mining search space uses legacy A FIELD_REGISTRY and has no market
```

## R24-296

```text
default A universe preset physical schema/description drift
```

## R24-297

```text
legacy fundamental presets lack full temporal contract
```

## R24-298

```text
label formula backward/forward semantic mismatch
```

## R24-299

```text
label raw close corporate action risk
```

## R24-300

```text
gap_bars implemented as calendar days
```

---

# 150. 新 hard gates

## R24-301

必须为0：

```text
RELATION_MULTI_PANEL_SILENT_REINDEX
RELATION_POSITIONAL_AXIS_MISMATCH
RELATION_PERCENT_UNIT_HEURISTIC
RELATION_VINTAGE_KEEP_LAST_LOOKAHEAD
RELATION_STALE_SNAPSHOT_UNBOUNDED
RELATION_ENTITY_NAME_IDENTITY_FALLBACK_FOR_TRACKING

PIT_PERIOD_USED_AS_FILING_FALLBACK
PIT_DECISION_TIME_USED_AS_UNKNOWN_KNOWLEDGE
PIT_AVAILABILITY_PRECISION_GUESSED_FROM_VALUE
PIT_OMITTED_LAYER_ASSUMED_TRUE

FIELD_PLAN_TEMPORAL_METADATA_LOST
CURRENT_SNAPSHOT_MARKED_NOT_APPLICABLE
COVERAGE_CONTRACT_CROSS_CONTEXT_COLLISION

SNAPSHOT_NO_TOKEN_MARKED_VERIFIED
CACHE_DATA_MANIFEST_EPOCH_MISMATCH

GROUP_MISSING_SILENT_DEFINITION_CHANGE
BEHAVIORAL_POLICY_NOT_IN_IDENTITY

IR_AVAILABILITY_EXPR_DROPPED
IR_SEMANTIC_IDENTITY_EXCLUDED
SCHEMA_UNKNOWN_PIT_DEFAULT_SAFE

SOURCE_REF_CROSS_MARKET_COLLISION
SOURCE_REF_PRODUCTION_DECODE_PERMISSIVE

PRODUCTION_VALIDATOR_RESEARCH_ANALYZER
PRODUCTION_VALIDATOR_MARKET_UNKNOWN
US_MINING_SPACE_USING_ASHARE_REGISTRY

LABEL_CONFIG_FORWARD_SEMANTIC_MISMATCH
LABEL_BAR_CALENDAR_DAY_CONFUSION
LABEL_RAW_CORPORATE_ACTION_CONTAMINATION

ADJUSTMENT_FACTOR_VINTAGE_PIT_UNPROVEN
```

---

# 151. Final report

## R24-302

输出真实计数：

```text
Relation canonicals audited:
Relation multi-panel strict:
Relation vintage certified:
Relation snapshot-age covered:

Sources audited:
Temporal contracts complete:
Current snapshot sources:
Unverifiable snapshot sources:

IR nodes/semantic dimensions audited:
AvailabilityExpr preserved:
Semantic digest collisions:

SourceRefs v1 found:
SourceRefs v2 migrated:
Production-invalid v1 refs:

Mining validators:
Production-mode verified:
Market-context verified:
Typed field-space A count:
Typed field-space US count:

Label pipelines:
Forward target certified:
Corporate-action safe:
Bar-clock safe:

Legacy presets:
Certified:
Research-only:
Deleted:

Snapshot race tests:
Stateful censor tests:
Full suite:
```

---

# 152. 最终旗标

## R24-303

只有全部 hard gate为0才允许：

```text
SEMANTIC_CONTINUITY_CLOSED=true
RELATION_STATE_PIT_CLOSED=true
IR_AVAILABILITY_CLOSED=true
SEMANTIC_IDENTITY_CLOSED=true
SNAPSHOT_CACHE_COHERENT=true
MINING_MODE_AUTHORITY_CLOSED=true
LABEL_FEATURE_SEPARATION_CLOSED=true
LEGACY_BYPASS_CLOSED=true
```

---

# 153. DoD

## R24-304

本轮真正结束标准：

```text
1. Relation/holder/index状态不会因缺失、reindex、revision而伪造事件；
2. PIT helper不会把 period/effective/decision time误当 knowledge time；
3. Field/Provider里声明的语义不会在 FieldPlan/IR/Plan 中途丢失；
4. AvailabilityExpr是完整 DAG，不是只剩字符串 label；
5. Factor/cache/materialization identity包含真正改变经济定义的 semantic digest；
6. Snapshot manifest精确对应实际读取 epoch；
7. Production validator真的运行 production+market-aware Analyzer；
8. US mining search space不会再使用A股 legacy registry；
9. SourceRef可以独立表达 market/provider/timeframe/temporal identity；
10. Label拥有独立未来语义，不污染 Feature DSL，也不会被raw corporate action欺骗；
11. legacy/default convenience API无法绕开新核心合同。
```

---

# 154. 执行指令

## R24-305

现在直接在服务器真实工作区执行：

```text
inspect HEAD/dirty
→ dynamic discover
→ fix code
→ add tests
→ run real data
→ regenerate artifacts
→ run full suite
→ run static gates
→ report exact remaining blockers
```

不要只修本文列出的函数。

要搜索同类模式：

```text
reindex
reindex_like
to_numpy multi-panel
drop_duplicates keep=last
nan_to_num
nansum
int(...)
max(...int(...))
except Exception
FIELD_REGISTRY
Analyzer()
source_certified
available_at label
semantic_attrs compare=False
snapshot_now_only
current_only
asof_backward
fallback_policy
missing_policy
```

每个命中都判断：

```text
是否属于同一个 semantic continuity / PIT / identity 风险
```

并一次性清掉。

---

# 155. 最终原则

FactorEngine 现在真正危险的已经不是“某个 SMA 写错”这种问题，而是：

```text
语义在 A 层是正确的，
经过 B/C/D 层以后被简化，
最后 E 层又用一个默认值把它当安全。
```

本轮必须做到：

```text
No semantic downgrade.
No temporal guessing.
No implicit market.
No implicit universe.
No silent alignment.
No unproven snapshot.
No semantic cache collision.
No production-via-research-validator.
No label/feature future-semantics confusion.
```

**只要其中任意一条还存在，就不能说“每个算子已经可以在自动化挖掘里被稳定、正确、PIT-safe 地使用”。**
