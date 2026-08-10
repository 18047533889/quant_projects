# FactorEngine R25：全算子真实可用性、证据可信度、参数域、真实 Recipe 与最终验收闭环整改提示词

> 用途：直接交给负责 FactorEngine 的代码 AI，在服务器真实工作区一次性完成最终整改与验收。  
> 性质：R17–R23 之后的**独立验收/整改轮**；R24 仍是并行必做依赖，本文件不复制 R24 内容。  
> 当前 GitHub 审计基线：`7b15a5e7a7a769734f7d1e003a6b0867c1ce88c9`。执行时先读取服务器真实 `HEAD + dirty tree`，以后者为准。  
> 禁止硬编码历史 canonical 数量。所有数量、集合、字段、证据均动态发现。

---

# 0. 本轮目标：不要再用“标签变绿”定义可用

用户所说的“算子可用”不是：

```text
registered=True
status=production
pit_safe tag存在
mining_visible=True
有 role
有 smoke_recipe_id
有 default_input_recipe 字符串
synthetic DataFrame 上能跑
```

真正可用必须至少同时满足：

```text
数学定义正确
输入语义/单位/grain/轴正确
输出语义/domain正确
参数合法域真实可执行
searchable 参数有效且非 dead
history/warmup/state topology正确
missing/NaN/Inf/zero/tie边界正确
真实 source/provider 可解析
真实 market/context 可用
PIT/knowledge/effective/period/vintage正确
真实 recipe 可 compile + load + execute
至少一个 correctness-certified backend
自动挖掘 grammar 中位置合法
所有证据绑定当前代码快照
```

任何一项不满足，不允许靠改 metadata、tag、status 或 artifact JSON 解决。

---

# 1. 当前 GitHub main 不能证明“全算子已可用”

## R25-001

当前 committed R23 artifact 仍显示大致：

```text
Total canonicals: 1430
SUPPORTING_ONLY: 1377
CERTIFIED: 29
CERTIFIED_CONTEXTUAL: 21
RESEARCH_TOOL: 3
```

先在服务器按当前 HEAD 重新生成，不要把历史数字当常量。

## R25-002

当前 R23 remediation 中主要 blocker 仍包括：

```text
experimental lifecycle: not production-certified
PIT11_EXPECTATION_POST_EVENT_LEAK
PIT18_REVISION_EVENT_UNPROVEN
```

禁止简单将这些状态批量改成 certified。

## R25-003

当前 committed R19 audit 仍记录：

```text
约 1430 rows
约 648 canonicals with blockers
```

先真实重跑 R19；如果当前代码已修，应以 fresh artifact 证明 blocker=0，而不是忽略旧报告。

---

# 2. 已确认当前代码仍存在旧问题，不是全部 R17–R23 都真正落地

## R25-004 Public kernel 参数仍有 local coercion

当前代码仍可见类似：

```python
int(max_lookback)
str(side).lower()
```

例如 A-share state-machine family。

要求全仓扫描 public operator implementation：

```text
int(...)
float(...)
str(...)
bool(...)
max(... int(...))
min(... int(...))
```

区分安全的“已验证后使用”与非法的“实现层静默纠正输入”。

## R25-005

非法：

```text
3.7 -> int -> 3
"3" -> int -> 3
True -> int -> 1
拼错 enum -> lower -> 默认 branch
负值 -> max(0,x) -> 偷偷修正
```

必须 implementation boundary strict fail。

## R25-006

所有行为参数必须 strict enum：

```text
side
direction
method
mode
policy
fallback_policy
missing_policy
revision_policy
```

未知值直接 raise。

---

# 3. R22 的“ALL_RETAINED_OPERATORS_MINING_USABLE”目前是结构闭环，不是真实闭环

## R25-007

现有 R22 closure 主要检查：

```text
role resolved
AST position exists
output domain非空
source context非空
smoke recipe metadata存在
grammar reachable
deny conflict
```

它没有充分要求：

```text
production_admitted
R19 math blocker=0
R23/R24 PIT blocker=0
真实 recipe执行
真实 provider解析
searchable 参数域认证
parameter injectivity
真实 backend parity
```

因此 R22 的 true 不能作为最终可用证书。

## R25-008

新增唯一最终 authority：

```text
GenuineUsabilityCertificate
```

R18/R19/R22/R23/R24 都只是它的子证据。

---

# 4. 定义 GenuineUsabilityCertificate

## R25-009

每个 retained canonical × market/context 至少生成：

```text
canonical
semantic_version
role
disposition
terminal_allowed
allowed_ast_positions

math_definition_certified
input_contract_certified
output_contract_certified
parameter_region_certified
parameter_injectivity_certified
history_contract_certified
missing_numeric_policy_certified

runtime_reference_certified
edge_cases_certified
backend_execution_certified

source_recipe_certified
source_contract_certified
market_context_certified
pit_certified
revision_vintage_certified_if_required
pre_event_snapshot_certified_if_required

real_recipe_compiles
real_recipe_executes
real_recipe_output_valid
real_recipe_coverage_valid

genuine_usable
production_admitted
blockers[]
evidence_ids[]

head
canonical_set_digest
field_catalog_digest
provider_digest
parameter_region_digest
source_context_digest
```

## R25-010

一个 active operator/context 只有上述所有适用项 PASS 才能：

```text
genuine_usable=True
```

不适用项必须明确 `N/A`，禁止空字符串或 UNKNOWN 被当 PASS。

---

# 5. 当前 factor certifier 存在“过度授证”

## R25-011

当前 `certify_factor_operator_evidence.py` 在 synthetic runtime audit 通过后，会为每个通过 canonical 直接写：

```text
semantic_golden_verified=True
temporal_prefix_verified=True
source_contract_verified=True
```

但 runtime audit 实际主要证明：

```text
能执行
轴基本正确
重复执行确定
prefix invariance
```

这不能证明数学语义 golden，更不能证明真实 DataAccess/source PIT。

## R25-012

拆证据：synthetic runtime audit 只允许授予：

```text
runtime_execution_verified
shape_verified
determinism_verified
prefix_kernel_causality_verified
```

## R25-013

`semantic_golden_verified` 必须来自独立：

```text
数学/经济公式
reference implementation
known-value golden
metamorphic properties
独立 test fixture
```

## R25-014

`source_contract_verified` 必须来自：

```text
FieldSpec
ProviderBinding
DataAccess contract
真实 schema/field/filter
真实 source read/join
knowledge/effective/period/vintage tests
```

## R25-015

禁止用“参数名匹配 + axes不变”证明 source contract。

---

# 6. six-gate 架构可保留，但每门证据必须独立

## R25-016

保留：

```text
implementation
semantic
temporal
source-PIT
edge-case
backend
```

## R25-017

对应 authority：

```text
implementation -> reference runtime执行
semantic -> 独立数学/经济 golden
temporal -> prefix/history/state/decision-time
source-PIT -> 真实 provider + PIT/vintage
edge-case -> hostile fixtures
backend -> 至少一个 evidence-backed生产 backend
```

## R25-018

任何 gate 不得由另一 gate“背书补齐”。

---

# 7. 当前 factor evidence 对当前 HEAD 已陈旧

## R25-019

当前 `factor_operator_verified.json` 绑定旧 commit `0b659eca...`，而审计时 main 为 `7b15a5e7...`，期间 operator/IR/planner/source/runtime/api/field 均有变化。

## R25-020

不要现在直接重跑旧 certifier“刷新绿灯”。先修 certifier 的过度授证，再最后生成 evidence。

## R25-021

最终 evidence 必须绑定当前 clean snapshot 的全部执行 TCB。

---

# 8. Artifact canonical-set 目前不一致

## R25-022

当前历史 artifact 中出现：

```text
R18 registered ≈1426
R22 rows ≈1433
R23 rows ≈1430
```

还有 R18 artifact 曾记录 dirty fingerprint。

## R25-023

建立单一：

```text
R25_CANONICAL_SNAPSHOT.json
```

字段：

```text
HEAD
canonical_set_digest
count
canonical list
surface
role
disposition
implementation source
backend set
```

## R25-024

所有最终 auditor 都消费该 snapshot，不再各自用略不同的筛选规则。

## R25-025

最终所有 artifact 必须共享：

```text
HEAD
canonical_set_digest
operator_registry_digest
field_catalog_digest
provider_registry_digest
```

---

# 9. R18/R22 smoke recipe 现在不是真正的全算子 smoke

## R25-026

现有 direct-use smoke 更多验证 registry/input metadata，不能作为真实可执行证明。

## R25-027

修复 `DIRECT_WITHOUT_SMOKE_RECIPE` 一类 detector，必须真正检查：

```text
smoke_recipe_ids
real recipe object
compile result
execute result
```

而不是重复检查 input contract 非空。

## R25-028

测试 `smoke_recipes_cover_all_direct_ops` 不能只检查：

```text
有 input_slots / scalar params
```

必须验证每 retained direct op 至少一条真实可执行 recipe。

---

# 10. Default recipe 当前存在明确不真实的绑定

## R25-029

当前 explicit recipe 可见：

```text
ts_average_volume.volume -> continuous_volume
dollar_volume.volume -> continuous_volume
```

而标准 concept 应核对为 `raw_volume_shares`，且 volume 不应随价格复权因子调整。

## R25-030

全量修正 volume recipe：

```text
volume -> raw_volume_shares
```

除非存在经过明确定义和验证的独立 synthetic adjusted-volume concept。

## R25-031

禁止 generic fallback：

```text
没有 recipe -> 第一个 data input = continuous_close
```

作为 production recipe authority。

## R25-032

禁止当前这类泛化：

```text
weight/weights -> continuous_close
sid1..sid10 -> continuous_close
p1..p7 -> continuous_close
s1..s10 -> continuous_close
```

## R25-033

`sid*` 需要 `EntityStableId`；`weight` 需要 `NonNegativeWeight` 或 `SignedWeight`；匿名多输入 slot 必须由 per-canonical contract决定。

---

# 11. 建立 Real Recipe Resolution Matrix

## R25-034

每 canonical × market × recipe × slot 记录：

```text
required semantic kind
required unit
required grain
required cardinality
concept_id
provider_id
dataset
physical fields
required filters
resolved
loadable
coverage
PIT status
```

## R25-035

active genuine operator 至少一条 recipe：

```text
所有 slot resolved
provider真实存在
required filter真实可执行
数据真实可读
```

## R25-036

不允许“recipe 字典非空”直接 PASS。

---

# 12. Recipe 必须真实执行

## R25-037

A-share 与 US 分 context 做 real-data smoke；允许使用小窗口/小股票集，但数据必须来自真实 provider，不是随机 synthetic panel。

## R25-038

记录：

```text
market
dataset/date range
instrument count
rows read
backend
runtime
finite ratio
nonconstant ratio
quantiles
output domain pass
exception
```

## R25-039

role-specific output：

```text
Condition -> {0,1,NaN}
SignedEvent -> {-1,0,1,NaN}
State -> declared state domain
Count -> nonnegative integer where finite
Probability -> [0,1]
Corr -> [-1,1]
```

---

# 13. Parameter injectivity 当前没有闭环

## R25-040

当前 DirectUse 中 `parameter_injectivity_passed=False` 被硬编码，必须改成真实证书。

## R25-041

新增 `ParameterInjectivityCertificate`：

```text
canonical
parameter
certified domain
probe values
factor identity changed
output changed
injective/effective
reason
```

## R25-042

若参数改变但合法数据下输出始终不变：

```text
DEAD_PARAMETER
```

处理：删除参数、固定参数或修实现，禁止 miner 继续搜索。

---

# 14. 当前 parameter evidence 只证明 default 值，不能支持自动搜索

## R25-043

当前 factor certifier 默认 `certified_parameter_domain={"bounds":["default"]}`；primitive evidence 也大量是 default-only 执行域。

## R25-044

miner 实际搜索：

```text
window
lag
q
threshold
bins
rank
alpha
min_periods
...
```

所以 default-only 证书不能证明 searchable parameter 可用。

## R25-045

建立 `CertifiedParameterRegion`：

```text
dtype
bounds
choices
coupled constraints
boundary cases
representative cases
property-based cases
```

## R25-046

关系约束必须执行：

```text
fast < slow
min_periods <= window
lag < window
q in legal interval
rank <= embedding dimension
bins >= minimum
```

## R25-047

miner 暴露域必须满足：

```text
search_domain ⊆ certified_parameter_region
```

---

# 15. R19 数学 dynamic coverage 必须覆盖全部适用 canonical

## R25-048

当前 dynamic/reference/metamorphic audit 主要覆盖 hardcoded core subset；最终不允许用小 subset 的动态测试证明全部 1400+。

## R25-049

每 retained operator 获得：

```text
EXECUTED
N/A_WITH_REASON
BLOCKED_CONTEXTUAL
```

禁止 `NOT_RUN/SKIPPED_UNKNOWN` 出现在 genuine surface。

## R25-050

每个 canonical 必须有独立 math definition / family-exact definition，不能只凭名字或 category生成证书。

---

# 16. Metamorphic properties 必须 per-canonical

## R25-051

不要给所有 operator 统一套 scale/translation/permutation property。

## R25-052

例：

```text
rank -> order/row permutation equivariance
corr -> translation invariant, positive-scale invariant, symmetry, [-1,1]
variance -> translation invariant, quadratic scale
zscore -> translation invariant, positive-scale invariant
count -> integer nonnegative
```

## R25-053

不适用必须 N/A，不允许错误 metamorphic property 造成假 blocker或假 pass。

---

# 17. Backend differential audit 目前覆盖太小

## R25-054

现有 differential MODE_TABLE 只覆盖少量 core operator；不能证明全部 claimed optimized backend。

## R25-055

不要求每个 operator同时有 Pandas/Polars/DuckDB。用户接受单后端生产。

## R25-056

但每个 operator 声称的：

```text
production backend
preferred backend
native backend
```

必须真实 differential against reference。

## R25-057

若只有 Pandas，且正确：

```text
genuine_usable=True
production_backend=pandas_numpy
```

完全允许。

## R25-058

若 registry 有 Polars object 但实际 fallback：

```text
polars_native_certified=False
```

## R25-059

backend parity 必须覆盖 certified parameter region 的 representative/boundary samples，不只 default window。

---

# 18. R23 generator 当前更多是枚举/分类，不是深度逐算子 proof

## R25-060

现有 R23 generator 多个 issue 字段可直接写空，classification 主要依据 tags/lifecycle/backend count。

## R25-061

改成四态：

```text
PASS
FAIL
BLOCKED
N/A
```

每项带 `evidence_id`。

## R25-062

禁止：

```text
math_issue=""
```

被解释为 math PASS。

## R25-063

`test_all_canonical_semantic_pit_certificate` 不能只测行数/status enum；对 genuine candidate 必须要求所有必需字段都有明确 proof state。

---

# 19. R23 Temporal Source Certificate 有跨市场 key 覆盖风险

## R25-064

当前 artifact 按裸 `table.name` 做 key，A/US 同名表（StockDailyBar、StockIncome、StockBalance、StockCashFlow 等）可能相互覆盖。

## R25-065

改 key：

```text
market::dataset::table
```

例如：

```text
ashare::ashare_stock_income::StockIncome
us::us_stock_income::StockIncome
```

## R25-066

最终 artifact 必须同时保留 A 与 US，且测试 set equality。

---

# 20. `strict_pit_allowed=true` 与 `knowledge_time_resolution=UNPROVEN` 不得矛盾

## R25-067

对 exact observed daily 数据，如果 knowledge-time概念不适用：

```text
N/A_EXACT_OBSERVATION
```

而不是 UNPROVEN。

## R25-068

对于 snapshot/relation/event/fundamental，如果关键时钟 UNPROVEN：

```text
production block
```

不能 pit_reason 同时写 PIT-safe。

---

# 21. 财报 revision vintage 仍未闭环

## R25-069

当前 US StockBalance/Income/CashFlow source certificate 明确存在：

```text
announcement_pit_certified=true
revision_vintage_pit_certified=false
restatement_risk=true
```

## R25-070

普通 announcement-PIT factor 与 revision-sensitive factor分开认证。

## R25-071

revision family：

```text
fin_revision_*
fin_restated_flag
revision age/count/magnitude
```

在真实 historical vintage source不存在时：

```text
BLOCKED_NO_DATA / BLOCKED_CONTEXT
```

从 production-grade mining surface移除。

## R25-072

不要因为 synthetic daily panel能模拟 revision就授予 source证书。

---

# 22. Expectation / surprise source 仍是 contextual blocker

## R25-073

当前测试主要证明 operator 声明：

```text
requires:PreEventExpectationSnapshot
requires:ConsensusVintageSource
```

这不是证明 source已经存在。

## R25-074

没有 historical consensus vintage 时 expectation revision family必须 block。

## R25-075

没有严格 pre-event expectation snapshot时 surprise/beat/miss family不得 production admit。

---

# 23. FinancialStatementBundleIdentity 需要真实 integration test

## R25-076

当前 bundle test主要是 axes 对齐和简单 ratio，不足以证明：

```text
same fiscal period
compatible timeframe
compatible vintage
knowledge visible
```

## R25-077

增加真实/构造 event-source integration：

```text
Q3 balance + Q2 income -> fail
quarterly income + TTM cashflow -> fail unless exact operator semantics allows
old vintage + new vintage incompatible -> fail
```

---

# 24. `fin_component_score` 目前仍是结构性 bug，只是被隔离

## R25-078

当前 implementation 仍把 DataFrame columns 当 components，然后把一行 score repeat 回所有列。

标准 FE wide panel：

```text
columns = instruments
```

所以实现仍会把股票当组件。

## R25-079

必须真正重构，不允许只保留 `supporting_only` tag作为“已修”。

## R25-080 方案 A

多 panel：

```text
fin_component_score(component_1, component_2, ...)
```

每个 component 都是 date×instrument。

## R25-081 方案 B

显式 `ComponentStack(component × date × instrument)` 类型。

## R25-082

普通 date×instrument DataFrame传入 ComponentStack-only接口必须 hard fail。

---

# 25. Input slot semantic typing 当前仍过泛

## R25-083

大量 slot 只有 `allowed_semantic_kinds=("field",)`，不足以阻止错误 recipe。

## R25-084

对经济含义明确的 slot至少细化：

```text
price -> PriceRaw/PriceContinuous
volume -> NonNegativeActivity
ret -> ReturnDecimal
condition -> ConditionBool
event -> EventBool/SignedEvent
group -> GroupKey
weight -> NonNegativeWeight/SignedWeight
entity_id -> EntityStableId
period_id -> FiscalPeriodId
```

## R25-085

generic arithmetic可宽，但单位兼容仍要检查。

---

# 26. output_value_domain 不能由 suffix heuristic作最终 authority

## R25-086

当前存在类似：

```text
*_corr -> bounded_0_1
```

但相关系数合法域为 `[-1,1]`。

## R25-087

`*_r2` 也不能统一 `[0,1]`：OOS/某些定义可负。

## R25-088

建立显式 `OutputDomainSpec`：

```text
REAL
NONNEGATIVE
POSITIVE
ZERO_ONE
NEG_ONE_ONE
INTEGER_NONNEGATIVE
BOOLEAN
SIGNED_EVENT
CATEGORY
PROBABILITY
PRICE_LEVEL
RETURN
RATIO_SIGNED
COUNT
DURATION
DATE
IDENTIFIER
```

支持 lower/upper/open/closed/integer/nullability。

## R25-089

suffix heuristic最多用于 candidate suggestion，production certificate必须显式审计。

---

# 27. Output domain直接参与 grammar合法性

## R25-090

例如：

```text
log
sqrt
inverse
asin
acos
```

只允许接受 compatible domain。

## R25-091

这不是“加标签”，而是减少自动挖掘的大量无效表达式。

---

# 28. Default mining surface区分 exploration 与 genuine production-grade

## R25-092

允许保留：

```text
research_all
direct_structural
```

## R25-093

新增：

```text
genuine_usable
genuine_usable_high_cost
production_fastpath
```

## R25-094

正式自动因子挖掘默认使用：

```text
genuine_usable + budgeted genuine_usable_high_cost
```

而不是所有 mining_visible/direct-all-context。

---

# 29. 高成本不是不可用；单后端也不是不可用

## R25-095

wavelet/DMD/spectral/topological/kernel等只要正确，可放 high-cost lane。

## R25-096

只有 Pandas但正确，也可以 genuine usable。

## R25-097

runtime策略：

```text
correctness-certified backends
-> 选择当前最快 eligible backend
```

---

# 30. FinalAdmissionResolver 必须消费所有子证据

## R25-098

当前 admission 逻辑没有完整消费：

```text
R19 blocker
R23/R24 PIT blocker
recipe failure
parameter region failure
injectivity failure
```

## R25-099

建立统一 `FinalAdmissionResolver`，输入：

```text
R19 math certificate
R23/R24 source-temporal certificate
R25 input/output contract
R25 parameter region/injectivity
R25 real recipe
backend evidence
edge evidence
market capability
DirectUse role
```

## R25-100

只有该 resolver可写最终：

```text
genuine_usable
production_admitted
```

---

# 31. Role 与 usability 保持正交

## R25-101

最终“所有 retained operator genuinely usable”含义是：

```text
每个 retained useful operator
在自己声明的 role/context里真正可用
```

不是所有都 terminal alpha。

## R25-102

角色：

```text
ALPHA -> terminal/intermediate
HIGH_COST_ALPHA -> budget lane
INTERMEDIATE -> composition
STATE -> state position
CONDITION -> gate
EVENT -> event position
GROUP/GLOBAL -> context
SOURCE_TRANSFORM -> source preprocessing
```

---

# 32. Context-specific usability

## R25-103

至少输出：

```text
ashare_daily
ashare_minute_to_daily
us_daily
us_minute_to_daily
```

适用状态。

## R25-104

市场机制不存在不是 failure：

```text
A-share limit operator @ US -> UNSUPPORTED_MARKET_MECHANISM
```

即可。

## R25-105

source availability必须验证：

```text
provider exists
dataset exists
fields exist
filters resolvable
coverage满足 policy
PIT certificate通过
```

---

# 33. Coverage 与 effective sample 是 usability 的一部分

## R25-106

区分：

```text
mathematically usable
contextually usable
default-mining eligible
```

## R25-107

部分 coverage 可 `CERTIFIED_PARTIAL`，不必删除，但 miner必须看到真实 joint coverage。

## R25-108

多字段 recipe以真实 joint finite coverage 为准，不是单字段 coverage最大值。

## R25-109

已有 `min_effective_samples/min_event_count` 不能只做 metadata，runtime/miner应真正 gate。

---

# 34. Advanced statistical operators要检查真实输出质量

## R25-110

对 entropy/MI/topological/spectral/DMD/GARCH等 real smoke报告：

```text
finite_output_ratio
unique_output_count
variance
min effective sample
runtime
```

## R25-111

真实 A-share/US 数据上几乎全 NaN 的 operator不能默认进入普通搜索空间，除非 context/coverage明确说明。

---

# 35. History / state / incremental

## R25-112

每 parameter sample 验证：

```text
planner requested lookback >= kernel实际需要
```

## R25-113

执行：

```text
full history
vs prefetch(warmup)+target window
```

parity。

## R25-114

stateful有 checkpoint：

```text
full vs chunk+restore parity
```

无 checkpoint：诚实 `required_full_history`。

---

# 36. Audit script不能再“测一小部分、全局绿”

## R25-115

每 auditor输出：

```text
eligible_count
executed_count
N/A_count
blocked_count
error_count
coverage_ratio
```

## R25-116

release/genuine pass要求对适用集合：

```text
executed + N/A + blocked == eligible
UNKNOWN/SKIPPED == 0
```

## R25-117

`load_all`失败可以生成 diagnostic artifact，但正式 artifact必须 `INCOMPLETE` 且 release fail。

---

# 37. Bulk promotion list不能再是认证 authority

## R25-118

大型 `PROMOTED_OUT_OF_EXPERIMENTAL` 之类手工集合只能记录 intended disposition。

## R25-119

最终必须：

```text
certificate pass -> promotion
```

不能：

```text
promotion list -> certificate pass
```

---

# 38. Edge/hostile fixtures全量化

## R25-120

适用 operator覆盖：

```text
NaN block
all NaN
Inf/-Inf
zero
zero denominator
constant
ties
tiny sample
exact minimum sample
below minimum
misaligned axes
```

## R25-121

condition：

```text
0,1,NaN合法
0.2,-1,2,Inf非法
```

## R25-122

group：

```text
missing group
single member
tiny group
all same group
```

---

# 39. Cross-market unit normalization real golden

## R25-123

至少验证：

```text
A Return bp -> decimal
US Ret already decimal
percent -> ratio
CNY/USD local money不可直接跨市场数值比较
shares count
```

## R25-124

确保 normalization exactly once。

---

# 40. 多输入 recipe防退化

## R25-125

检查：

```text
x==y
所有 component同一 source
benchmark==own return
group/weight误用signal
```

若会让经济定义失效，默认 recipe不得如此绑定。

## R25-126

benchmark family必须有真实 benchmark provider；A/US分别验证。

---

# 41. Fundamental recipe逐 slot对齐真实字段

## R25-127

遍历所有 fundamental input：

```text
short_term_debt
tax_payable
receivables
inventory
depreciation
...
```

必须解析到实际 A/US concept/provider。

## R25-128

不存在时不能 fallback close。

## R25-129

`fin_total_operating_accruals` 等重新按当前数据逐 market审：若 A无数据、US有数据，应 context-specific，不要沿用过时全局 DELETE_NO_DATA。

---

# 42. A-share minute real smoke

## R25-130

至少覆盖：

```text
09:31
11:30
午休无bar
13:01
15:00
缺bar
停牌
涨跌停
```

## R25-131

US intraday若当前无 provider，诚实 context blocked，不伪造。

---

# 43. Decision-time availability也进入 usability

## R25-132

一个 EOD causal factor可 genuine usable，但必须声明：

```text
usable_after=CLOSE
```

不能被开盘策略使用。

## R25-133

建议 certificate记录：

```text
OPEN
INTRADAY
MORNING_END
CLOSE
NEXT_OPEN
```

适用 decision points。

---

# 44. Evidence/artifact false-green防护

## R25-134

核心依赖失败时 auditor必须 exit non-zero。

## R25-135

禁止 catch exception -> 写 partial rows -> exit 0。

## R25-136

最终 artifact不能用空字符串代表 PASS。

---

# 45. 新 artifacts

## R25-137

生成：

```text
factor_engine/docs/R25_CANONICAL_SNAPSHOT.json
factor_engine/docs/R25_GENUINE_USABILITY_MATRIX.csv
factor_engine/docs/R25_GENUINE_USABILITY_MATRIX.json
factor_engine/docs/R25_GENUINE_USABILITY_MATRIX.md
```

## R25-138

证据可信度：

```text
factor_engine/docs/R25_EVIDENCE_TRUST_AUDIT.json
factor_engine/docs/R25_EVIDENCE_TRUST_AUDIT.md
```

## R25-139

参数：

```text
factor_engine/docs/R25_PARAMETER_REGION_CERTIFICATES.json
factor_engine/docs/R25_PARAMETER_INJECTIVITY_AUDIT.csv
```

## R25-140

recipe：

```text
factor_engine/docs/R25_REAL_RECIPE_AUDIT.csv
factor_engine/docs/R25_REAL_RECIPE_AUDIT.json
factor_engine/docs/R25_REAL_DATA_SMOKE_REPORT.md
```

## R25-141

语义：

```text
factor_engine/docs/R25_OUTPUT_DOMAIN_MATRIX.json
factor_engine/docs/R25_INPUT_SLOT_SEMANTIC_MATRIX.json
```

## R25-142

backend：

```text
factor_engine/docs/R25_BACKEND_CLAIM_MATRIX.csv
factor_engine/docs/R25_BACKEND_PARAMETER_PARITY.json
```

## R25-143

source/context：

```text
factor_engine/docs/R25_MARKET_SOURCE_RECIPE_MATRIX.json
factor_engine/docs/R25_SOURCE_PIT_CONTEXT_MATRIX.json
```

## R25-144

最终：

```text
factor_engine/docs/R25_FINAL_BLOCKERS.json
factor_engine/docs/R25_FINAL_BLOCKERS.md
factor_engine/docs/R25_FINAL_ACCEPTANCE_REPORT.md
```

---

# 46. 必加测试

## R25-145 证书独立性

```text
test_runtime_smoke_does_not_grant_semantic_golden
test_runtime_smoke_does_not_grant_source_pit
```

## R25-146 错公式但能跑

构造能跑/causal/shape正确但数学名字不符 fixture：

```text
runtime PASS
semantic FAIL
genuine FAIL
```

## R25-147 source缺PIT

synthetic能跑但真实 provider无 knowledge proof：

```text
source FAIL
genuine FAIL
```

## R25-148 参数域

```text
default正常
boundary出错
```

必须 parameter_region FAIL。

## R25-149 dead param

改变 searchable param 输出无变化 -> injectivity FAIL。

## R25-150 invalid concept

`continuous_volume` 等不存在 concept -> recipe FAIL。

## R25-151 slot type

`sid*` 解析到 Price、weight解析到 Price -> hard fail。

## R25-152 output domain

corr=-0.8 必须合法；不能 ZERO_ONE。

## R25-153 cross-market source cert

A/US同名 table证书同时存在且不覆盖。

## R25-154 revision

`revision_vintage_pit_certified=False` -> revision-sensitive op production_admitted=False。

## R25-155 expectation

无 consensus vintage/pre-event proof -> corresponding op production_admitted=False。

## R25-156 component score

普通 date×instrument DataFrame不得被解释成 ComponentStack；重构后 multi-component output逐股票正确。

## R25-157 real recipe

所有 genuine operator/context至少一条真实 compile+execute recipe。

## R25-158 backend

单 Pandas correctness可以 PASS；claimed native fallback不能算 native PASS。

## R25-159 artifact snapshot

所有 final artifacts HEAD/digest一致。

---

# 47. 最终统一 runner

## R25-160

新增：

```text
scripts/audit_r25_genuine_usability.py
```

执行顺序：

```text
1. assert/read exact workspace snapshot
2. load/finalize registry
3. build canonical snapshot
4. rerun R19 current math
5. consume fresh R23/R24 temporal/source
6. certify input/output semantics
7. certify parameter regions
8. certify injectivity
9. resolve+execute real recipes
10. run semantic golden/metamorphic
11. run hostile edge fixtures
12. verify claimed backends
13. resolve market/source contexts
14. build GenuineUsabilityCertificate
15. export genuine mining manifests
16. run final hard gates
```

---

# 48. R24 是最终依赖，不能跳过

## R25-161

R24 尚未执行时，不允许宣布以下已经闭环：

```text
relation/shareholder/index state PIT
IR semantic continuity
AvailabilityExpr identity
snapshot/cache epoch
SourceRef v2
production/mining mode authority
label separation
legacy bypass
```

## R25-162

执行 AI 应同时读取并执行 R24 与 R25；本文件不复制 R24 内容。

## R25-163

最终只有：

```text
R24 hard flags all true
AND
R25 hard flags all true
```

才允许系统总绿。

---

# 49. R25 hard blockers

## R25-164

最终必须为0：

```text
R19_CURRENT_MATH_BLOCKERS
R23_CURRENT_SEMANTIC_PIT_BLOCKERS_FOR_ACTIVE_CONTEXT

UNCERTIFIED_SEMANTIC_GOLDEN
UNCERTIFIED_PARAMETER_REGION
DEAD_SEARCHABLE_PARAMETER
UNTESTED_SEARCHABLE_PARAMETER

UNRESOLVED_REAL_RECIPE
UNEXECUTED_REAL_RECIPE
RECIPE_SEMANTIC_TYPE_MISMATCH
RECIPE_NONEXISTENT_CONCEPT
RECIPE_PROVIDER_UNAVAILABLE
RECIPE_PIT_UNCERTIFIED

OUTPUT_DOMAIN_HEURISTIC_ONLY
INPUT_SLOT_GENERIC_WHERE_TYPED_REQUIRED

SOURCE_CERT_CROSS_MARKET_COLLISION
SOURCE_CERT_UNPROVEN_BUT_MARKED_SAFE
REVISION_OPERATOR_WITHOUT_VINTAGE_SOURCE
EXPECTATION_OPERATOR_WITHOUT_VINTAGE_SOURCE

COMPONENT_SCORE_AXIS_DEFECT

FACTOR_EVIDENCE_STALE
FACTOR_EVIDENCE_GATE_OVERCLAIM
ARTIFACT_CANONICAL_SET_MISMATCH
ARTIFACT_DIRTY_UNPINNED

CLAIMED_NATIVE_BACKEND_UNTESTED
CLAIMED_NATIVE_BACKEND_FALLBACK_ONLY

GENUINE_USABILITY_UNKNOWN
```

---

# 50. 当前已确认 concrete findings，执行 AI 必须逐项复核

## R25-165

```text
R23 committed matrix：绝大多数仍 SUPPORTING_ONLY
```

## R25-166

```text
R19 committed matrix：仍有大量 blockers
```

## R25-167

```text
A-share state-machine仍有 local int/str coercion
```

## R25-168

```text
R18 persisted artifact来自旧/dirty fingerprint
```

## R25-169

```text
R18/R22/R23 canonical counts不一致
```

## R25-170

```text
R22 mining-usable gate未要求 production_admitted / R19 / R23 / real recipe
```

## R25-171

```text
R18/R22 smoke recipe更多是结构 metadata，不是真实 source执行
```

## R25-172

```text
DirectUse parameter_injectivity_passed hardcoded False
```

## R25-173

```text
default recipe存在 continuous_volume
```

## R25-174

```text
generic recipe用 continuous_close填未知 data slot，包括不合理的 ID/weight类
```

## R25-175

```text
output_value_domain依赖 suffix heuristic，*_corr 域错误
```

## R25-176

```text
production-grade mining与 direct-all-context / uncertified candidate边界尚未最终闭合
```

## R25-177

```text
R23 generator空 issue字段不能证明 PASS
```

## R25-178

```text
R23 all-canonical test主要验证行数/status enum
```

## R25-179

```text
R23 temporal source cert用裸 table key，A/US同名表存在覆盖风险
```

## R25-180

```text
部分 source cert存在 strict_pit_allowed=true + knowledge_time_resolution=UNPROVEN
```

## R25-181

```text
US financial revision_vintage_pit_certified=false
```

## R25-182

```text
revision/expectation tests主要声明“仍需要某 source”，不是 source已解决
```

## R25-183

```text
FinancialBundle测试不足以证明 fiscal period/vintage/timeframe bundle
```

## R25-184

```text
fin_component_score实现轴缺陷仍在，只是被 supporting_only/experimental 隔离
```

## R25-185

```text
factor_operator_verified绑定旧 commit，对当前 TCB stale
```

## R25-186

```text
factor certifier把 synthetic runtime smoke过度授予 semantic/source证书
```

## R25-187

```text
audit_all_factor_production依赖大量 hand-maintained synthetic parameter/panel heuristic
```

## R25-188

```text
factor/primitive parameter execution evidence主要 default-only，不能覆盖 miner完整 searchable domain
```

## R25-189

```text
R19 dynamic math coverage不是全 canonical
```

## R25-190

```text
backend differential MODE_TABLE只覆盖小 subset
```

## R25-191

```text
final admission尚未统一消费 math/PIT/recipe/parameter全部 blocker
```

## R25-192

```text
大型人工 promotion set不能作为长期 certification authority
```

---

# 51. 最终 flags

## R25-193

只有全部 hard gate 通过才允许：

```text
R25_CURRENT_R19_BLOCKERS_ZERO=true
R25_SEMANTIC_EVIDENCE_INDEPENDENT=true
R25_SOURCE_EVIDENCE_INDEPENDENT=true
R25_PARAMETER_REGION_CLOSED=true
R25_PARAMETER_INJECTIVITY_CLOSED=true
R25_REAL_RECIPE_CLOSED=true
R25_REAL_DATA_SMOKE_CLOSED=true
R25_OUTPUT_DOMAIN_CLOSED=true
R25_INPUT_SLOT_TYPING_CLOSED=true
R25_BACKEND_CLAIMS_CLOSED=true
R25_SOURCE_CONTEXT_PIT_CLOSED=true
R25_FUNDAMENTAL_CONTEXT_CLOSED=true
R25_COMPONENT_SCORE_AXIS_CLOSED=true
R25_EVIDENCE_FRESH=true
R25_ARTIFACT_SNAPSHOT_COHERENT=true
R25_FINAL_ADMISSION_SINGLE_AUTHORITY=true
R25_ALL_GENUINE_OPERATORS_REACHABLE=true
R25_ALL_RETAINED_OPERATORS_DISPOSITIONED=true
R25_ALL_RETAINED_OPERATORS_GENUINELY_USABLE_IN_DECLARED_ROLE=true
```

---

# 52. 最后一条 flag 的含义

## R25-194

不是：

```text
所有 retained operator 都能作为 terminal alpha
```

而是：

```text
所有值得保留的 operator
都在自己合法 role / market / source / decision context 中真正可用；
不能真实使用的，则已经诚实地 BLOCK / RESEARCH / INTERNAL / DELETE。
```

---

# 53. 最终报告必须给精确数字

## R25-195

`R25_FINAL_ACCEPTANCE_REPORT.md` 第一屏直接给：

```text
Current canonical count:
Retained public:
Deleted:
Internal:
Research tools:
Blocked no-data/context:

Genuine usable total:
- Alpha:
- High-cost alpha:
- Intermediate:
- State:
- Condition:
- Event:
- Group/global:
- Source transform:

A-share genuine:
US genuine:

Math PASS/FAIL:
Parameter PASS/FAIL:
Real recipe PASS/FAIL:
PIT PASS/FAIL:
Backend PASS/FAIL:
Real-data smoke PASS/FAIL:
```

## R25-196

如果仍有失败，逐个列：

```text
canonical
market/context
blocker
source file
exact remediation
after-fix disposition
```

禁止只写 `remaining=12`。

---

# 54. 禁止的整改方式

## R25-197

禁止：

```text
bulk set production_certified
bulk set semantic_golden_verified
bulk set source_contract_verified
bulk clear blockers
直接改 JSON/CSV 让 artifact变绿
```

## R25-198

artifact只能由修好的代码与测试重新生成。

## R25-199

最终检查 diff：如果主要改动只是 status/tags/docs/artifact，而没有 implementation/test/source/evidence对应变化，则判 FAIL。

---

# 55. 最终执行顺序

## R25-200

不要停在分析，直接在服务器完成：

```text
inspect HEAD/dirty
reproduce current blockers
执行 R24整改
执行 R25实现整改
重新跑 R19/R23 current audits
修真实 recipe
修参数域/injectivity
修 output/input semantics
修 source/context/PIT gate
修 component score
修 certifier过度授证
扩全量 dynamic/backend tests
real-data smoke
targeted tests
full suite
最后重新认证 evidence
重新生成所有 final artifacts
fresh-process evidence validation
final acceptance
```

---

# 56. 最终 DoD

## R25-201

本轮真正完成必须同时满足：

```text
1. active genuine surface 当前 R19 blocker=0；
2. active context 当前 R23/R24 source/PIT blocker=0；
3. 每个 retained operator有准确 role/disposition；
4. 每个 genuine operator有独立 mathematical semantic evidence；
5. 每个 searchable parameter只开放经过执行认证的 region；
6. dead searchable parameter=0；
7. 每个 genuine operator至少一条真实可解析/加载/执行 recipe；
8. recipe不再依赖 generic continuous_close 填洞；
9. 每个 source-sensitive operator有 market/context-specific PIT proof；
10. revision/expectation无真实 vintage source时诚实 block；
11. fin_component_score真正修轴；
12. output domain显式正确，不靠 suffix；
13. 至少一个 correctness-certified backend；
14. claimed native backend有真实 differential evidence；
15. final artifacts来自同一 clean HEAD/canonical digest；
16. production-grade mining只消费 genuine certificate；
17. high-cost正确算子保留在专属 lane；
18. state/event/condition/intermediate在合法 role里真实可用；
19. no-data/Research/Internal/Delete均明确 disposition；
20. 不存在任何“标签变绿但实现/证据没变”的假完成。
```

---

# 57. R24 + R25 联合最终门槛

## R25-202

只有：

```text
R24_ALL_HARD_FLAGS=true
AND
R25_ALL_HARD_FLAGS=true
```

才能最终输出：

```text
FACTOR_ENGINE_OPERATOR_SYSTEM_PRODUCTION_READY=true
```

---

# 58. 最终原则

## R25-203

FactorEngine 的“可用”以后只能定义为：

```text
一个 operator 在明确 market/source/decision context 下，
使用经过认证的输入和参数域，
按照被独立证明的数学/PIT/缺失值语义，
由至少一个经过 correctness 认证的 backend 稳定执行，
并能被 typed mining grammar 在合法 AST role 中真实调用。
```

只有这样，才叫：

```text
GENUINELY_USABLE
```
