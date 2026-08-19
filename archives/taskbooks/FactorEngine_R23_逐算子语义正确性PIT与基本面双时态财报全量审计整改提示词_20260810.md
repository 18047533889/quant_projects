# FactorEngine R23：逐算子语义正确性、PIT 与基本面双时态财报全量审计整改提示词

> **用途**：把本文件直接交给负责 FactorEngine 整改的代码 AI，在服务器真实工作区执行。  
> **本轮性质**：独立新一轮，不与 R17–R22 合并。  
> **本轮重点**：不是再讨论“一个 operator 是否 Direct Use”，而是对 **每一个当前 canonical 的定义、时间语义、输入语义、数据可知性与实际运行结果逐项重新做 semantic/PIT certification**。  
> **尤其重点**：Fundamental / Financial / Expectation / Revision / Valuation / Shareholder / Corporate Action。  
> **最新 GitHub 可见基线**：`b947c690a119c6fe42fe51b7fd2f9b5f0c746807`。服务器工作区如果更新，以真实 `HEAD + dirty tree` 为准。  
> **数据字典基线**：
>
> - A股：`3fe57785-f68a-4df8-b06e-c8bc44d1ae68.md`
> - 美股：`86b37d75-6854-4cad-ac26-5f4b024d4c63.md`
>
> **硬要求**：不要只修本文件点名的 operator。必须 `load_all()`，动态枚举服务器当前全部 canonical，逐个生成证书；任何 canonical 不能 `skip / except: continue / unknown forever`。

---

# 1. 先回答最重要的问题

**目前不能宣称“每个算子的 PIT、数学与基本面语义都已经全部解决”。**

R17–R22 已经大幅处理：

- 字段目录；
- provider/source；
- Direct Use；
- 数学统计；
- 参数；
- backend；
- compiler；
- service；
- mining integration；
- Research operator promotion；

但当前可见代码仍然存在一层更深的问题：

```text
表是 PIT-safe
≠ 字段组合是 PIT-safe

字段分别是 PIT-safe
≠ 两个字段 join 以后 PIT-safe

operator 本身不看未来
≠ operator 输入的“当前值”真的是当时可知的值

ReportPeriod 正确
≠ KnowledgeTime 正确

PubDate 正确
≠ 同一天任意交易时点都能使用

latest_available revision
≠ historical vintage PIT

flow grain metadata存在
≠ operator实际强制使用了正确 grain

fiscal-period aware
≠ missing quarter / irregular fiscal period / revision backfill 全部正确

production_certified
≠ 本轮逐 operator 的经济定义和实际数据 binding 已经被证明
```

本轮就是把这一层全部封死。

---

# 2. 最终目标：每个 canonical 一张 `OperatorSemanticPITCertificate`

## R23-001

动态枚举：

```python
from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

load_all()
all_canonicals = sorted(OperatorRegistry._catalog)
```

一个都不遗漏。

## R23-002

每个 canonical 最终生成：

```python
OperatorSemanticPITCertificate
```

至少包含：

```text
canonical
implementation_source
actual_backend_sources
operator_role
direct_use_status

mathematical_definition
economic_interpretation

data_inputs
input_semantic_types
input_units
input_grains
input_cardinalities

output_semantic_type
output_unit
output_domain

market_support
frequency_contract

knowledge_time_contract
effective_time_contract
period_time_contract
revision_time_contract
decision_time_contract

flow_semantics
period_topology_policy
missing_topology_policy

same_day_availability_policy
cross_source_join_policy
bundle_identity_policy
revision_vintage_policy

required_history
warmup
incremental_semantics

parameter_contract
parameter_injectivity

reference_backend
backend_parity

pit_tests
math_goldens
metamorphic_tests
real_data_smoke

certificate_status
blockers
remediation
```

## R23-003

最终状态只能：

```text
CERTIFIED
CERTIFIED_CONTEXTUAL
CERTIFIED_HIGH_COST
SUPPORTING_ONLY
RESEARCH_TOOL
INTERNAL_ONLY
DELETE
BLOCKED_NO_DATA
```

禁止：

```text
UNKNOWN
MAYBE
TODO
PENDING_WITHOUT_ACTION
SKIPPED
```

---

# 3. 建立真正的 PIT 时钟模型

基本面 PIT 不允许再只用一个 `pit_safe=True`。

## R23-004

每个数据输入必须区分至少以下时间：

```text
event_time
effective_time
period_time
knowledge_time
revision_time
ingestion_time
decision_time
```

## R23-005

定义：

```text
period_time
= 这个财务数字描述哪一个会计期间
```

例如：

```text
A股 ReportPeriodEndDate
US period_end / fiscal_year + fiscal_quarter
```

它不是可知时间。

## R23-006

定义：

```text
knowledge_time
= 市场最早可以知道该 observation 的时间
```

A股财报：

```text
PubDate
```

美股财报：

```text
filing_date
```

## R23-007

定义：

```text
revision_time
= 某个已披露 report period 的新 vintage 真正变为可知的时间
```

不能把：

```text
vendor UpdateTime
ingestion timestamp
sync timestamp
```

当 revision_time。

## R23-008

定义：

```text
decision_time
= 策略真正计算并允许使用信息的时点
```

例如：

```text
previous_close
today_open
10:00
today_close
after_close
next_session_open
```

## R23-009

最终 PIT proof：

```text
knowledge_time <= decision_time
AND
revision_time <= decision_time
AND
effective-time constraints satisfied
AND
no future source snapshot
```

---

# 4. 日期级 PubDate / filing_date 的 same-day lookahead 必须解决

这是基本面最容易被忽略的一类。

## R23-010

A股 `PubDate` 在当前数据中是 `date`，不是可靠的交易所公告 timestamp。

所以：

```text
PubDate == signal_date
```

不能自动等价于：

```text
开盘前已经知道
```

## R23-011

如果没有真实公告时分秒，production strict PIT 必须采用明确政策。

推荐：

```text
DATE_ONLY_KNOWLEDGE_TIME
→ usable_from_next_trading_session
```

即：

```text
PubDate = 2026-04-25
→ 默认从下一个交易日开盘以后使用
```

除非 source 能证明 announcement timestamp。

## R23-012

不能用：

```python
PubDate <= TradeDate
```

就允许当天开盘信号使用。

## R23-013

美股 `filing_date` 数据也要验证 timestamp 精度。

如果实际上全部：

```text
00:00:00
```

只是日期占位，而不是 SEC submission timestamp，

同样执行：

```text
next-session conservative policy
```

## R23-014

引入：

```python
KnowledgeTimeResolutionPolicy
```

枚举：

```text
EXACT_TIMESTAMP
DATE_ONLY_NEXT_SESSION
DATE_ONLY_AFTER_CLOSE
SESSION_CLOSE_AVAILABLE
EFFECTIVE_DATE_ONLY
UNPROVEN
```

## R23-015

任何 fundamental operator 的 PIT certificate 必须携带这个 policy。

---

# 5. A股财务四表：PubDate 只能解决 announcement PIT，不能自动解决 vintage PIT

当前 A股目录已经正确区分：

```text
PubDate = knowledge time
ReportPeriodEndDate = report period
UpdateTime = ingestion/sync
```

但还有一个关键问题。

## R23-016

必须证明 COS 历史文件保存的是：

```text
当时发布时的历史 vintage
```

而不是：

```text
今天重新同步得到的“最新修订值”写回历史 PubDate 文件
```

## R23-017

如果历史文件会被 vendor restatement overwrite：

即使：

```text
PubDate <= decision_date
```

仍然可能有经典 **revision lookahead**。

例如：

```text
2024Q1 初始收入 = 100
2025-03-01 公司重述 2024Q1 = 80

2024-06-01 回测
如果读取今天 COS 里的 80
→ 看到了 2025-03-01 才知道的重述
```

## R23-018

因此必须新增独立证书：

```text
announcement_pit_certified
revision_vintage_pit_certified
```

不能合成一个 `pit_safe=True`。

## R23-019

若 DataAccess 无法证明 immutable vintage history：

```text
revision_vintage_pit_certified=False
```

并且：

```text
revision-sensitive operators production blocked
```

## R23-020

普通财务 factor 也必须记录：

```text
restatement_risk = true/false
```

不能把 restatement 风险只留给 `fin_revision_*`。

---

# 6. A股 `UpdateTime` 绝不能作为财报 revision clock

## R23-021

当前字段语义已经说明：

```text
UpdateTime = vendor / pipeline write time
```

它只可用于：

```text
freshness
ingestion monitoring
dedupe within one immutable snapshot
```

## R23-022

禁止：

```text
UpdateTime <= decision_time
```

作为市场可知性证明。

## R23-023

禁止使用今天重新同步得到的 UpdateTime 推断过去某天发生了 revision。

---

# 7. 同一 PubDate 多 ReportPeriod：不能按 `(PubDate, Symbol)` 粗暴去重

A股数据字典明确允许：

```text
同一公告日披露多个 ReportPeriodEndDate
```

## R23-024

财务 event identity 至少需要：

```text
Symbol
PubDate
ReportPeriodEndDate
statement/source
```

revision 场景还需要：

```text
revision/vintage identity
```

## R23-025

审计所有 DataAccess/dedupe/pivot：

禁止仅按：

```text
(Symbol, PubDate)
```

选一行。

## R23-026

否则会出现：

```text
同一天年报 + 一季报
→ 一个报告期被静默删掉
```

然后：

```text
TTM
QoQ
YoY
trend
persistence
streak
revision
```

全部被污染。

## R23-027

生成真实数据测试：

```text
same instrument
same PubDate
two different ReportPeriodEndDate
```

必须都进入 fiscal event state。

---

# 8. 一个旧报告期被重述时，`current visible period` 不得倒退

## R23-028

维护两个不同概念：

```text
latest_visible_period
updated_period
```

不能混成同一个 `period_id`。

例：

```text
已知 2025Q3
今天公司重述 2025Q1
```

今天的：

```text
updated_period = 2025Q1
latest_visible_period = 2025Q3
```

## R23-029

如果 source projection把当前行 period_id设成 2025Q1，

很多 `_walk_periods` operator会把“当前报告期”理解成 Q1。

这会错误改变：

```text
lag
TTM
current ratio
trend
persistence
streak
latest factor value
```

## R23-030

引入：

```python
FundamentalPeriodState
```

至少：

```text
latest_visible_period
updated_period
visible_value_by_period
updated_fields
filing_time
vintage_id
```

## R23-031

revision event更新：

```text
visible_value_by_period[old_period]
```

但 current terminal factor通常应继续以：

```text
latest_visible_period
```

为 anchor，除非 canonical 明确是在测 revision 本身。

---

# 9. Cross-statement bundle PIT：财务多表不能独立 asof 后随便拼

## R23-032

一个 factor如果同时使用：

```text
StockBalance
StockIncome
StockCashFlow
```

不能只证明：

```text
每个字段 individually PubDate <= decision_time
```

## R23-033

必须证明：

```text
report_period匹配
filing bundle兼容
vintage兼容
flow timeframe兼容
```

## R23-034

建立：

```python
FinancialStatementBundleIdentity
```

字段：

```text
instrument
fiscal_period
filing/announcement id
knowledge_time
revision/vintage id
timeframe
statement_family
```

## R23-035

典型错误：

```text
Balance = 2025Q2 最新可见
Income = 2025Q1 最新可见
CashFlow = 2025Q2 最新可见

直接 current_ratio / accrual / quality composition
```

静态 algebra 不看未来，但经济语义已经错。

## R23-036

多表 operator默认要求：

```text
same fiscal_period
```

如果某种经济定义允许 asynchronous latest-known values，必须显式声明。

---

# 10. `period_id.reindex(...)` 不应该成为财务 operator 的自动修复机制

当前多个 fundamental kernel仍有：

```python
period_id.reindex(index=x.index, columns=x.columns)
```

以及 secondary panel reindex。

## R23-037

production 下所有 fundamental multi-input operator：

```text
axis mismatch -> hard fail
```

## R23-038

不允许 kernel自行：

```text
reindex
sort
drop duplicates
fill
align by position
```

这些属于 source/IR contract。

## R23-039

具体整改：

```text
fundamental/transforms_v2.py::_walk_periods
fundamental/quality_v2.py::_walk_two
fundamental/transforms_repairs_v2.py::_revision_masks
fundamental/transforms_repairs_v2.py::_revision_complete
fundamental/transforms_repairs_v2.py::fin_days_since_update
```

统一调用：

```python
strict_panel_align(...)
```

## R23-040

alignment证书必须包括：

```text
index equality
column equality
uniqueness
timezone
instrument identity
decision timestamps
```

---

# 11. Flow semantics 不能继续依赖 optional `flow_type`

当前架构已有：

```text
CumulativeYTDFlow
SinglePeriodFlow
TTMFlow
AnnualFlow
Stock
```

这是正确方向。

但：

```python
flow_type=None
```

会跳过很多 gate。

## R23-041

`flow_type` 不能是普通用户 scalar/search parameter。

## R23-042

它应该由：

```text
FieldSpec.flow_semantics
SourceRef
timeframe
IR SemanticType
```

自动绑定。

## R23-043

矿工不能自己随机选择：

```text
flow_type="SinglePeriodFlow"
```

去绕过 A股 YTD 字段的真实 grain。

## R23-044

把：

```text
flow_type
```

从：

```text
searchable parameter
```

迁成：

```text
compile-time semantic context
```

## R23-045

A股：

```text
StockIncome/StockCashFlow flow_ytd
→ CumulativeYTDFlow
```

## R23-046

US：

```text
timeframe=quarterly
→ SinglePeriodFlow（需 source contract证明）
timeframe=annual
→ AnnualFlow
timeframe=trailing_twelve_months
→ TTMFlow
StockBalance
→ Stock
```

## R23-047

如果 source 无法证明某个 timeframe 的含义：

```text
flow semantic UNKNOWN
→ fail closed
```

---

# 12. 美股 `timeframe` 必须成为数据 identity，不只是 query optional filter

美股字典明确：

```text
同一 period_end 可以同时有
quarterly
annual
trailing_twelve_months
```

## R23-048

任何 US financial read 在未指定 timeframe时：

```text
production hard fail
```

## R23-049

不能：

```text
把 quarterly revenue
和 trailing_twelve_months OCF
拼成一个 ratio
```

## R23-050

`timeframe` 必须进入：

```text
SourceRef identity
cache key
factor semantic identity
data snapshot identity
bundle identity
```

## R23-051

审计 `required_parameters=("timeframe",)` 是否真正沿：

```text
TableSpec
→ FieldSpec
→ resolver
→ SourceRef
→ DataAccess query
→ cache
```

全链传播。

---

# 13. 当前 US catalog 的 `strict_pit_allowed` 显式声明要全量修

最新可见代码 `_table()` 默认：

```python
strict_pit_allowed=None
```

但多个 US production-readable table没有显式传值。

## R23-052

动态审计：

```python
for table in US_TABLE_SPECS:
    assert table.strict_pit_allowed is not None
```

production-readable全部不得 UNKNOWN。

## R23-053

尤其核对：

```text
StockDailyBar
StockList
SecurityMaster
TickerSharesSnapshot
StockIndicesComponents
StockCapitalDaily
StockBalance
StockIncome
StockCashFlow
StockDividend
FactNews
Calendar
EarlyClose
UniverseDaily
```

## R23-054

不是全部简单设 True。

逐表决定：

```text
True
False
Contextual
CurrentSnapshotOnly
```

## R23-055

E2财务：

```text
filing_date PIT + timeframe required
```

证明完整后才 True。

## R23-056

SecurityMaster如果只有 current static snapshot：

```text
历史 PIT 不得 True
```

## R23-057

shares snapshot需要证明 observation date/knowledge time，不能只因为名字有 PIT 就自动 True。

---

# 14. 美股 period_end 不是 PIT，文件名更不是 PIT

## R23-058

任何代码：

```text
use filename period_end as asof timestamp
```

直接 release blocker。

## R23-059

US：

```text
period_end = accounting period
filing_date = knowledge
```

## R23-060

真实 regression test：

```text
period_end=2024-03-31
filing_date=2026-05-27

decision=2025-01-01
→ 必须不可见
```

---

# 15. Fiscal calendar：不能假设所有公司都是自然季度

## R23-061

US字段有：

```text
fiscal_quarter
fiscal_year
timeframe
```

这些必须进入 fiscal identity。

## R23-062

不要只从：

```text
period_end.month
```

推 Q1/Q2/Q3/Q4 语义。

## R23-063

例如：

```text
period_end=2024-03-31
fiscal_quarter=4
```

是完全可能的。

## R23-064

建立：

```text
FiscalPeriodKey
= fiscal_year + fiscal_quarter + timeframe
```

优先于 calendar-quarter guess。

## R23-065

`period_ordinal(period_end)` 只能作为：

```text
chronological ordering fallback
```

不能代表真实 fiscal quarter label。

## R23-066

`quarter_from_cumulative` 必须用真实：

```text
fiscal_quarter
```

当前没有 fiscal_quarter时 fail-closed的方向保持。

---

# 16. 53-week / irregular / transition periods

## R23-067

测试：

```text
53-week FY
short transition period
long transition period
fiscal year change
IPO partial quarter
missing filing
semiannual filer
```

## R23-068

任何需要：

```text
QoQ
YoY
CAGR
TTM
AR persistence
volatility
smoothness
```

的 operator必须声明：

```text
period adjacency policy
duration policy
```

## R23-069

不能因为 ordinal差1就自动假设经济 duration完全一致。

---

# 17. `require_consecutive=False` 必须按 operator逐个重新审

“最后 N 个 visible reports”只适合某些 distribution statistic。

## R23-070

适合允许缺期：

```text
historical percentile
historical robust distribution
possibly median/MAD
```

## R23-071

通常不适合：

```text
AR(1)
persistence
smoothness
trend
growth acceleration
growth change
volatility of period changes
TTM
moving average balance
streak
sign-change rate
```

## R23-072

每个 period operator新增：

```python
PeriodTopologyPolicy
```

枚举：

```text
CONSECUTIVE_REQUIRED
VISIBLE_EVENT_WINDOW
SAME_FISCAL_SLOT
DURATION_NORMALIZED
REVISION_EVENT_CLOCK
```

---

# 18. 当前 quality_v2 的 persistence/smoothness/volatility要专项修

## R23-073

当前：

```text
fin_earnings_cash_gap_volatility
fin_earnings_smoothness
fin_earnings_persistence
fin_cashflow_persistence
fin_margin_persistence
```

需要重新验证 missing fiscal periods。

## R23-074

`fin_*_persistence` AR(1)：

```text
Q1 -> Q3
```

不能当：

```text
one-step lag
```

## R23-075

默认：

```text
CONSECUTIVE_REQUIRED
```

## R23-076

如果要支持 event-clock persistence：

新建诚实名：

```text
*_visible_report_persistence
```

不要混在同一 canonical。

---

# 19. Fundamental 内仍存在 `int(periods)` 路径，必须统一参数权威

## R23-077

扫描所有：

```text
int(periods)
int(window)
max(..., int(...))
```

## R23-078

当前重点：

```text
fundamental/quality_v2.py
fundamental/accruals_scores.py
report_timing.py
```

## R23-079

全部改成统一：

```text
strict_positive_int
strict_nonnegative_int
```

## R23-080

必须拒绝：

```text
True
3.7
nan
inf
"3.7"
```

---

# 20. `fin_component_score` 的 axis contract 要重做

当前标准 FactorEngine panel：

```text
rows = time
columns = instruments
```

但 `fin_component_score` 当前把 DataFrame columns当 components。

## R23-081

这在标准 panel语义下等于：

```text
把股票当 component
```

然后把横截面 score复制回股票列。

## R23-082

禁止这种隐式形状。

## R23-083

推荐重设计为：

```python
fin_component_score(
    component1,
    component2,
    ...
)
```

每个 component都是：

```text
date x instrument panel
```

## R23-084

或者新增：

```text
ComponentStack semantic type
```

显式维度：

```text
component x date x instrument
```

不能复用普通 DataFrame宽表。

## R23-085

在修好前：

```text
SUPPORTING_ONLY / BLOCK
```

不能自动挖。

---

# 21. Revision operator：从 daily panel变化推 revision 不足够

当前：

```text
fin_revision_delta
fin_revision_pct
fin_revision_direction
fin_revision_count
fin_revision_magnitude
fin_restated_flag
```

很多仍通过：

```text
same period_id
+ value changed vs previous daily observation
```

识别 revision。

## R23-086

这只有在 source能够提供真正 historical vintage state时才成立。

## R23-087

如果 COS当前历史是重同步后的 final value：

```text
revision event根本不可恢复
```

## R23-088

因此这些 operator必须要求：

```text
RevisionEventSource
```

或：

```text
BitemporalFundamentalSource
```

## R23-089

不能把：

```text
daily panel value changes
```

作为独立 revision proof。

## R23-090

`report_revision_magnitude` 当前“source提供 prev_x”方向正确。

继续统一所有 revision family到同一个：

```text
RevisionPair
```

语义。

---

# 22. Revision pair 的标准结构

## R23-091

定义：

```python
RevisionPair:
    instrument
    report_period
    old_value
    new_value
    old_available_at
    new_available_at
    revision_available_at
    vintage_old
    vintage_new
```

## R23-092

必须：

```text
same report period
new_available_at > old_available_at
new_available_at <= decision_time
```

## R23-093

`UpdateTime` 不得代替 `revision_available_at`。

---

# 23. `fin_days_since_update / fin_staleness` 的左截断问题

当前第一次观察到完整值时：

```text
age=0
```

但如果 backtest sample从中途开始，这个财报可能早已公布几十天。

## R23-094

first observation != new report event。

## R23-095

必须从：

```text
knowledge_time / PubDate / filing_date
```

计算真实 age。

## R23-096

如果无法知道 sample开始前的 publication time：

```text
left_censored=True
```

输出：

```text
NaN
```

直到第一次真正 observed update。

## R23-097

`fin_days_since_update` 更名/定义拆分：

```text
days_since_publication
days_since_any_revision
days_since_latest_report_period
```

避免混义。

---

# 24. Expectation / Surprise：最危险的是 cross-source temporal leakage

## R23-098

`fin_surprise(actual, expected, ...)` 不能只做：

```python
actual - expected
```

## R23-099

必须要求：

```text
expected snapshot available strictly before actual publication
```

推荐：

```text
expected.available_at < actual.available_at
```

## R23-100

如果 actual与expected都是日频 ffill：

```text
same TradeDate
```

可能已经把 earnings公布后 analyst consensus revision拿来计算 surprise。

## R23-101

实现：

```python
PreEventExpectationSnapshot
```

在 source layer冻结：

```text
last estimate strictly before actual knowledge time
```

## R23-102

禁止 operator自行：

```text
expected.shift(1 trading row)
```

替代真正 event-time snapshot。

---

# 25. Expectation revision 本身也需要 vintage proof

## R23-103

要证明 expectation source真的保留：

```text
historical consensus vintages
```

而不是今天回填的历史 final consensus。

## R23-104

否则：

```text
fin_expectation_revision_count
fin_expectation_revision_magnitude
fin_days_since_expectation_revision
```

全部不能 production certify。

---

# 26. `fin_days_since_expectation_revision` 的 left-censor / cap 语义

如果第一条观察没有历史：

不能默认：

```text
age=max_days
```

代表“很久没修正”。

## R23-105

无历史是：

```text
unknown
```

不是：

```text
stale
```

## R23-106

新增：

```text
history_observed
left_censored
```

gate。

---

# 27. Surprise 标准化必须 prior-only

## R23-107

任何：

```text
surprise_zscore
surprise_percentile
```

审计：

```text
current event是否进入自己的reference distribution
```

## R23-108

默认 factor版本：

```text
strict prior-only
```

## R23-109

inclusive版本若保留：

```text
名字必须明确 *_inclusive
```

并单独 DirectUse status。

---

# 28. A股累计流量：所有 Income/CashFlow operator必须先决定 YTD / quarter / TTM

A股：

```text
StockIncome
StockCashFlow
```

为：

```text
flow_ytd
```

## R23-110

不能把：

```text
Q2 YTD / Q1 YTD - 1
```

叫：

```text
QoQ
quarter growth
```

## R23-111

需要单季：

```text
quarter_from_cumulative
```

## R23-112

需要 TTM：

```text
ttm_from_cumulative
```

## R23-113

每个 fundamental factor必须在 canonical contract声明：

```text
accepted flow semantics
```

---

# 29. 静态 flow/flow ratio不一定错，但必须诚实声明口径

例如：

```text
operating_profit / revenue
impairment / revenue
OCF / revenue
```

如果两边都是同一 YTD：

数学上可以是：

```text
YTD margin
```

## R23-114

不能无差别把所有 YTD ratio删掉。

## R23-115

但 canonical必须明确：

```text
same-grain ratio
```

并禁止：

```text
numerator YTD
denominator quarterly
```

## R23-116

可提供：

```text
YTD
single-quarter
TTM
annual
```

不同 typed recipe，不要让矿工混。

---

# 30. Flow / Stock ratio必须声明 horizon

例如：

```text
capex / avg_assets
borrowings cash / avg_assets
impairment / total_assets
R&D / assets
```

## R23-117

如果 flow是 YTD：

结果是：

```text
YTD intensity
```

而不是自动的 annualized intensity。

## R23-118

如果 canonical经济意义想表达 annual intensity：

先转：

```text
TTM/AnnualFlow
```

## R23-119

重点逐个审：

```text
fin_borrowing_intensity
fin_debt_repayment_intensity
fin_capex_intensity
fin_acquisition_cash_intensity
fin_goodwill_risk_score impairment term
financing cash flow / assets families
```

---

# 31. `fin_cash_burn_runway` 必须在类型层要求 Annualized/TTM OCF

当前文档已经说：

```text
annualized_ocf
```

但名字不等于类型证明。

## R23-120

输入 semantic：

```text
TTMFlow | AnnualizedFlow
```

## R23-121

拒绝：

```text
CumulativeYTDFlow
SinglePeriodFlow
```

除非先显式 annualize。

---

# 32. Divergence family 不能用一个“same flow grain”规则覆盖 Stock vs Flow

例如：

```text
receivable growth - sales growth
inventory growth - sales growth
```

其中：

```text
receivable/inventory = Stock
sales = Flow
```

## R23-122

它们经济上可以组合，但要求：

```text
stock growth over same fiscal interval
vs
single-period/TTM sales growth over same fiscal interval
```

## R23-123

所以当前这种：

```text
_require_same_flow_grain(..., 2)
```

如果真的要求两个输入同 grain，会过度拒绝；

如果 `flow_type=None` 又会完全不检查。

## R23-124

改成 per-slot semantic contract：

```text
slot A: Stock
slot B: SinglePeriodFlow | TTMFlow
comparison_interval: same fiscal interval
```

## R23-125

分别审：

```text
fin_receivable_sales_divergence
fin_inventory_sales_divergence
fin_cash_sales_divergence
fin_expense_sales_divergence
```

---

# 33. Fundamental static ratio也必须同 report-period bundle

比如：

```text
current_ratio
quick_ratio
debt_to_equity
goodwill_intensity
lease_intensity
deferred_tax_gap
```

## R23-126

如果输入都来自同一 Balance statement：

要求：

```text
same report period
same filing vintage
```

## R23-127

不能只是 daily index相同。

---

# 34. `operating_margin` 等 flow ratio必须同 timeframe

## R23-128

例如 US：

```text
operating_income timeframe=quarterly
revenue timeframe=TTM
```

即便同 filing_date也不合法。

## R23-129

ratio operator的 input contract必须传播：

```text
FinancialBundleIdentity
```

---

# 35. Piotroski family专项

## R23-130

逐个证明：

```text
ROA
OCF
NetProfit
Leverage
CurrentRatio
TotalCapital
GrossMargin
AssetTurnover
```

的 input recipe。

## R23-131

尤其：

```text
OCF > NetProfit
```

要比较同一 normalization basis。

不要：

```text
raw CNY OCF
vs ratio NetProfit
```

## R23-132

input metadata写 `"rate"` 不够，default input binding必须真实构造对应 rate。

## R23-133

A股 OCF/利润如果基于 YTD：

Piotroski所需 period semantics必须统一转换。

## R23-134

applicability mask：

```text
non-financial
```

必须来自 PIT industry/classification。

不能用今天行业分类回填历史。

---

# 36. Fundamental strength cross-sectional score专项

## R23-135

同一交易日不同股票可能：

```text
A 已披露 Q3
B 还停留 Q2
C 只到 Q1
```

## R23-136

直接横截面 rank是可定义的“latest-known”因子，但要显式控制 staleness。

## R23-137

至少提供：

```text
max_days_since_publication
max_fiscal_period_age
min_component_coverage
```

## R23-138

不要把：

```text
新鲜 Q3
和陈旧 Q1
```

不加标记地当完全同质观测。

---

# 37. 财务适用性：银行、保险、券商不能套普通工业企业公式

## R23-139

必须审：

```text
current ratio
quick ratio
working-capital accruals
Altman
Piotroski
asset turnover
inventory metrics
receivable metrics
debt metrics
```

## R23-140

`fin_applicability_mask`必须：

```text
PIT classification
```

## R23-141

A股 IndustrySource：

使用当时 TradeDate的 industry snapshot。

## R23-142

美股目前无可靠行业表时：

不能假造 non-financial mask。

需要：

```text
source-backed security/industry classification
```

否则对应 composite score在 US context blocked。

---

# 38. Valuation daily“日频存在”不代表 denominator PIT安全

## R23-143

PE/PB/PS/PCF是：

```text
price
/
vendor fundamental denominator
```

## R23-144

如果 vendor后来重述财报并回填历史 PE：

历史 daily valuation factor会发生 revision lookahead。

## R23-145

因此 `StockValuationDaily exact_daily` 不足以单独证明：

```text
derived ratio historical PIT
```

## R23-146

给 derived vendor fields新增：

```text
derived_lineage_pit_certified
fundamental_vintage_policy
```

## R23-147

如果不能证明：

优先自己用 PIT financial denominator + daily price构造：

```text
earnings_yield
book_to_price
sales_yield
cashflow_yield
```

---

# 39. US current-snapshot-only valuation/indicator必须彻底禁止历史 mining

当前：

```text
StockValuationDaily
StockIndicator
```

US是 sparse/current snapshot。

## R23-148

任何历史 backtest：

```text
strict_pit_allowed=False
```

保持。

## R23-149

不得为了 Direct Use coverage把它们 forward/backward fill成历史。

---

# 40. US market cap source也要确认 shares snapshot PIT

## R23-150

`TickerSharesSnapshot` coverage有限。

要核对：

```text
observation timestamp
publication timestamp
snapshot date
duplicate rows
share class
split adjustment
```

## R23-151

`market_cap = close * weighted_shares_outstanding`

必须保证 shares：

```text
available_at <= decision_time
```

---

# 41. Corporate actions / dividend 的 PIT

A股 StockDividend：

```text
没有 announcement time
只有 ex/effective date
```

## R23-152

所以只能：

```text
effective_only
```

## R23-153

禁止在 ex-date之前做：

```text
future dividend known
```

## R23-154

美股 dividend：

有：

```text
declaration_date
ex_dividend_date
```

且 ex-date可能未来。

## R23-155

pre-event factor必须：

```text
declaration_date <= decision_time
```

## R23-156

effective execution adjustment：

```text
ex_dividend_date
```

## R23-157

cash currency非USD时：

禁止直接合并/排名，除非 PIT FX provider。

---

# 42. Shareholder / ownership PIT

A股 TopTen shareholder：

```text
PubDate knowledge
ReportPeriodEndDate period
one-to-many
```

## R23-158

任何：

```text
holder concentration
holder churn
pledge
freeze
network
```

必须先：

```text
aggregate one-to-many at one report vintage
```

再投影到日频。

## R23-159

不能：

```text
先 forward-fill raw shareholder rows
再 aggregate
```

因为 entity set会跨 vintage混合。

## R23-160

snapshot identity：

```text
Symbol
ReportPeriodEndDate
PubDate/vintage
```

---

# 43. Shareholder entity ID稳定性

## R23-161

如果 `ShareholderId` vendor会变化：

network/churn不能靠名字或临时ID直接跨期匹配。

## R23-162

需要：

```text
stable entity identity proof
```

否则：

```text
holder_entry/exit
churn
network centrality
```

可能是假事件。

---

# 44. Index constituent PIT

## R23-163

指数成分因子必须区分：

```text
announcement date
effective date
trade date snapshot
```

## R23-164

如果数据只有 effective daily membership：

只能从 effective date使用。

不能预知未来调入调出。

---

# 45. Industry PIT

## R23-165

行业中性化/group operators：

必须使用：

```text
decision-date industry
```

## R23-166

不能用：

```text
current/latest industry classification
```

回填历史。

---

# 46. Universe/listing PIT

## R23-167

全量检查：

```text
delisted stocks
IPO date
suspension
ST/risk status
index membership
security master
```

## R23-168

任何全历史 universe必须避免：

```text
today surviving symbols only
```

---

# 47. Price/volume operator也要按 decision time认证

PIT不只基本面。

## R23-169

daily OHLC：

```text
today close/high/low/volume
```

只能在：

```text
session close之后
```

使用。

## R23-170

如果策略是：

```text
today open下单
```

today full-day bar是未来数据。

## R23-171

Factor identity必须带：

```text
decision_time
```

不是只带 `TradeDate`。

---

# 48. A股 minute bar

当前语义：

```text
QuoteTime = bar_end
```

## R23-172

09:31 bar：

只在：

```text
>=09:31
```

可用。

## R23-173

不能把 bar-end数据分配到09:30决策。

---

# 49. Intraday → daily operator

## R23-174

每个 session aggregation必须声明：

```text
required_session_cutoff
```

例如：

```text
EOD full-session factor
```

不能给开盘策略。

## R23-175

partial-session factor：

只读 cutoff之前bar。

---

# 50. Return / corporate-action basis

## R23-176

所有：

```text
return
momentum
volatility
overnight
gap
breakout
candlestick
```

要证明 price basis一致。

## R23-177

A股 factor direction当前仍要以实测复权规则为准。

在未证明前：

```text
raw + corporate-action contamination
```

必须标出。

---

# 51. Cross-sectional operator PIT

## R23-178

`cs_*` 在时间上不看未来，也可能有 cross-sectional information leak。

## R23-179

一个 t日横截面必须只含：

```text
t时点真实 universe
```

## R23-180

不能用：

```text
未来上市股票
已退市被删股票
未来指数成分
```

---

# 52. Group operator PIT

## R23-181

group membership：

```text
group_t
```

必须 PIT。

## R23-182

group mean/rank/zscore：

不能使用未来行业映射。

---

# 53. Neutralization PIT

## R23-183

industry/size neutralize：

```text
industry_t
size_t
```

都必须 decision-time可知。

## R23-184

size如果用：

```text
today close * shares
```

只能close后。

开盘信号要：

```text
previous close
或 open-available market cap
```

---

# 54. Benchmark-related operator

## R23-185

benchmark return必须与 stock return：

```text
same interval
same price basis
same availability
```

## R23-186

`ex_self` benchmark：

必须在当前 universe动态计算，不能用未来完整样本。

---

# 55. Event operator PIT

## R23-187

任何：

```text
event_age
event_count
event_spacing
event_response
marked_event
```

必须使用：

```text
event knowledge time
```

而不是 event的 economic period。

## R23-188

若事件表带未来 effective dates：

event不可提前出现。

---

# 56. State machine operator

## R23-189

state递推必须只有过去/current合法输入。

## R23-190

checkpoint恢复必须携带：

```text
state as of last processed decision time
```

不能恢复未来checkpoint到历史segment。

---

# 57. Rolling/statistical operator

## R23-191

统一检查：

```text
window inclusive/exclusive
current observation role
missing topology
min periods
ties
Inf
zero variance
```

R19整改视为并行，但本轮逐 canonical验证实际结果。

---

# 58. Regression operator

## R23-192

分：

```text
in-sample diagnostic
prior fit
forecast
forecast error
```

不能混名。

## R23-193

任何 factor版 regression：

训练数据终点必须：

```text
<= t-1
```

除非定义明确允许 current predictor而 target不参与fit。

---

# 59. Cross-sectional regression

## R23-194

t日 cross-sectional neutralization：

允许使用同一 t日所有 stocks的 t日 exposures，

但 exposures本身必须在decision time可知。

---

# 60. Advanced nonlinear/spectral/topology operator

## R23-195

这些通常没有 fundamental PIT，但仍要证明：

```text
trailing-only
no centered window
no future padding
no full-series normalization
no future-derived embedding scale
```

## R23-196

DMD/HVG/RQA/SSA/multifractal等：

normalization parameters必须由当前窗口内数据决定。

---

# 61. SourceTransform

## R23-197

`ffill` 是 PIT-safe 只有：

```text
source value确实在过去已知
```

## R23-198

`ffill financial`还要保留：

```text
original knowledge_time
period_id
vintage_id
age
```

不能只留下value。

---

# 62. Imputation

## R23-199

cross-sectional mean/median fill：

只能用当前合法 universe可见值。

## R23-200

time-series fill：

不能跨 future observation。

---

# 63. Missing financial values不能默认0

## R23-201

财报 NaN通常：

```text
not disclosed
not applicable
missing vendor coverage
```

不是：

```text
economic zero
```

## R23-202

逐 operator检查：

```text
fillna(0)
np.nansum
skipna
```

是否有业务证明。

---

# 64. `row_sum_skipna` 类要区分“缺项允许”与“缺项不可比较”

## R23-203

例如：

```text
资产组成合计
```

某些缺项不能自动当0。

## R23-204

要求每个 sum operator声明：

```text
min_count
missing_component_policy
applicability_policy
```

---

# 65. Financial component score缺失策略

## R23-205

“missing component ignored”可能会让不同 coverage股票得分不可比。

## R23-206

如果保留 partial score：

名字必须：

```text
partial
normalized_partial
observed_count
```

## R23-207

full score：

必须完整组件。

---

# 66. Derived financial input不能只靠字段名证明语义

例如：

```text
annualized_ocf
avg_assets
gross_margin
roa
leverage
```

## R23-208

这些必须是：

```text
typed derived node
```

携带 provenance。

## R23-209

矿工不能把任意 field alias成 `avg_assets` 就过类型检查。

---

# 67. DefaultInputRecipe 对 fundamental尤其要严格

## R23-210

每个 fundamental operator默认 recipe必须完整写出：

```text
source table
field concept
flow conversion
period alignment
normalization
applicability
```

## R23-211

不是：

```text
x -> revenue
```

这么简单。

---

# 68. A/US 同一个 canonical的 binding可以不同，但经济定义必须一致

## R23-212

例如：

```text
revenue growth
```

A股：

```text
YTD -> quarterize -> growth
```

US：

```text
timeframe=quarterly -> growth
```

## R23-213

最终 canonical output economic definition仍是：

```text
single-quarter revenue growth
```

---

# 69. 跨市场财务字段不能按“名字像”绑定

## R23-214

要求：

```text
concept_id
economic definition
unit
flow grain
period coverage
```

都匹配。

---

# 70. Currency

## R23-215

raw：

```text
CNY amount
USD amount
```

不能共同 cross-market rank。

## R23-216

dimensionless ratio可以跨市场，但仍要 accounting definition一致。

---

# 71. US inventory 100% null等字段不得进入默认 recipe

数据字典有些字段 coverage极低/全空。

## R23-217

operator source availability必须基于真实 coverage。

## R23-218

例如 US：

```text
inventory
```

若当前 dataset实际不可用：

相关 operator在 US：

```text
context blocked
```

不是自动借A股字段。

---

# 72. Financial source freshness vs PIT

## R23-219

freshness与PIT分开。

```text
source迟到
```

不代表未来函数；

但会影响 live trading。

## R23-220

每个 fundamental source记录：

```text
expected publication lag
data pipeline lag
max freshness lag
```

---

# 73. PIT cache identity

## R23-221

财务 cache key至少包含：

```text
dataset snapshot
decision time
timeframe
period selection
revision policy
knowledge-time policy
source vintage generation
```

## R23-222

不能：

```text
same field + same date
```

复用不同 PIT policy的结果。

---

# 74. Materialization

## R23-223

materialized fundamental factor必须保存：

```text
available_at
source knowledge-time digest
source vintage id
financial bundle id
```

## R23-224

否则 factor lake里只有：

```text
timestamp,value
```

无法证明 PIT。

---

# 75. Incremental restatement propagation

## R23-225

如果今天修订：

```text
2024Q1
```

哪些 factor date需要重算取决于：

```text
revision knowledge time
```

不是 period_end。

## R23-226

历史 factor在 revision announcement之前不能改变。

## R23-227

从 revision available_at之后：

所有依赖该历史period的：

```text
TTM
growth
trend
persistence
score
```

需要按 dependency lineage更新。

---

# 76. 增量重算不能把重述写回 revision之前

## R23-228

建立：

```text
BitemporalInvalidationRange
```

例如：

```text
economic_period = 2024Q1
knowledge_change_time = 2025-03-01

recompute validity:
[2025-03-01, +∞)
not [2024-03-31, +∞)
```

---

# 77. `first_available` vs `latest_available`

## R23-229

两个 revision policy都合法，但含义不同。

## R23-230

必须进入 factor identity/hash。

## R23-231

不能两个 policy共用 cache/materialization。

---

# 78. `latest_available` 的真实含义

## R23-232

必须是：

```text
latest available AS OF decision time
```

不是：

```text
latest vintage known today
```

---

# 79. Financial operator actual input lineage

## R23-233

每个运行中的 fundamental operator生成：

```text
InputTemporalLineage
```

列：

```text
field
period_id
knowledge_time
revision_time
vintage
flow_semantics
timeframe
```

---

# 80. 逐 operator PIT differential test

## R23-234

对每个 operator构造：

```text
base dataset
future append dataset
```

要求：

```text
outputs before cutoff unchanged
```

---

# 81. Future filing randomization

## R23-235

随机修改：

```text
future PubDate filing values
future revisions
future analyst expectations
```

历史 output必须 bitwise/NaN-mask一致。

---

# 82. ReportPeriod randomization

## R23-236

修改未来 period metadata不能影响历史。

---

# 83. Revision scenario golden tests

## R23-237

至少覆盖：

```text
new period
same-period revision
old-period restatement
same-day two periods
same-day two revisions
missing period
provider gap
late filing
out-of-order filing
```

---

# 84. A股 YTD golden sequence

## R23-238

构造：

```text
Q1 YTD=10
Q2 YTD=25
Q3 YTD=36
FY YTD=50
```

应：

```text
quarters = 10,15,11,14
TTM = 50
```

## R23-239

再加入 Q1 later revision：

```text
10 -> 8
```

revision之前历史不变；

revision之后 downstream quarter/TTM按 policy更新。

---

# 85. US timeframe golden

## R23-240

同 period_end构造：

```text
quarterly=10
annual=40
TTM=38
```

任何 operator只能读选定 timeframe。

---

# 86. Same-day knowledge test

## R23-241

date-only PubDate：

```text
decision = same-day 09:30
```

必须不可用。

## R23-242

next trading session：

可用。

---

# 87. Cross-source surprise golden

## R23-243

```text
09:00 expected = 100
16:00 actual = 120
17:00 analyst consensus revised = 118
```

surprise必须：

```text
120 - 100
```

不能：

```text
120 - 118
```

---

# 88. Current-period-vs-restated-period golden

## R23-244

已知：

```text
Q1,Q2,Q3
```

今天重述 Q1。

普通：

```text
latest earnings
TTM
trend
```

current period仍 Q3。

revision factor则识别 updated period=Q1。

---

# 89. `fundamental/ops.py` 与 `fiscal_strict.py` 的 override chain必须收口

当前旧模块仍注册：

```text
quarter_from_cumulative
ttm_from_quarterly
ttm_from_cumulative
yoy_by_period
```

到 `period_helpers`，

后续 strict模块再 replace。

## R23-245

这条链太危险。

## R23-246

最终公共 canonical只注册一次逻辑 implementation authority。

## R23-247

旧 `period_helpers`：

```text
compat/private
```

不得先占 canonical再被覆盖。

## R23-248

增加：

```text
import-order permutation test
```

无论模块加载顺序：

```text
canonical implementation hash/source
```

完全一致。

---

# 90. `period_helpers` 旧逻辑不得进入 production

## R23-249

特别是旧逻辑：

```text
row i-1
last 4 finite rows
fallback i-4
```

在 daily forward-filled fundamental panel上错误。

## R23-250

检查 registry最终真实backend source必须指向：

```text
fiscal_strict
```

或已证明严格 replacement。

---

# 91. Fundamental operator metadata自己写 `pit_safe` 不算证据

## R23-251

当前多个模块 metadata仍带：

```text
pit_safe
causal
```

## R23-252

这些 tag只作为描述，不得授信。

## R23-253

真正证书来自：

```text
operator semantic certificate
source PIT certificate
temporal differential tests
```

---

# 92. Per-canonical audit dimensions：所有算子都跑

下面不是 fundamental专属。

## R23-254

数学：

```text
definition
reference formula
domain
edge cases
```

## R23-255

输入：

```text
arity
semantic role
units
grain
cardinality
```

## R23-256

时间：

```text
knowledge/effective/period/decision/revision
```

## R23-257

缺失：

```text
NaN
Inf
gap
warmup
left censor
```

## R23-258

参数：

```text
strict type
range
relational constraints
injectivity
```

## R23-259

输出：

```text
shape
unit
domain
variation
terminal role
```

## R23-260

backend：

```text
reference
parity
fallback
```

## R23-261

incremental：

```text
warmup
checkpoint
restatement
correction
```

## R23-262

PIT：

```text
prefix invariance
future randomization
decision time
source knowledge
```

---

# 93. 自动生成“每个算子还有什么问题”的表

## R23-263

生成：

```text
R23_PER_CANONICAL_AUDIT.csv
```

每行一个 canonical。

列至少：

```text
canonical
category
module
source
direct_status

math_issue
input_issue
unit_issue
grain_issue
pit_issue
same_day_issue
revision_issue
period_issue
missing_issue
parameter_issue
backend_issue
incremental_issue
cross_market_issue

severity
concrete_remediation
test_to_add
final_status
```

## R23-264

必须：

```text
row count == current canonical count
```

---

# 94. 基本面单独生成更细表

## R23-265

```text
R23_FUNDAMENTAL_OPERATOR_AUDIT.csv
```

额外列：

```text
accepted_flow_semantics
actual_default_binding_flow
period_identity
knowledge_clock
revision_clock
bundle_requirement
timeframe_requirement
fiscal_calendar_requirement
left_censor_policy
restatement_policy
applicability
staleness_policy
```

---

# 95. 每个 fundamental operator必须回答 12 个问题

## R23-266

1. 输入是 Stock / YTD / Quarter / TTM / Annual 中哪个？
2. A股默认字段真实是什么 grain？
3. US默认 timeframe是什么？
4. report period怎么确定？
5. knowledge time怎么确定？
6. same-day什么时候可用？
7. revision如何处理？
8. old-period restatement如何处理？
9. missing quarter能否跳过？
10. 多表输入是否同bundle？
11. sample起点是否left-censored？
12. output到底代表 YTD、quarter、TTM还是latest-known？

任何一项回答不清：

```text
不能 production certify
```

---

# 96. 对 fundamental family逐类跑

## R23-267 Transform family

至少包括动态发现的：

```text
fin_lag
fin_diff
fin_pct_change
fin_log_change
fin_qoq
fin_yoy
fin_ttm
fin_average_balance
fin_growth
fin_cagr
fin_growth_acceleration
fin_growth_change
fin_growth_volatility
fin_growth_stability
fin_growth_persistence
fin_std
fin_mad
fin_cv
fin_stability
fin_range
fin_zscore_history
fin_percentile_history
fin_trend_*
fin_monotonicity
fin_*streak
fin_sign_change_count
```

逐个检查：

```text
period topology
flow semantics
prior/inclusive
revision
```

---

# 97. Period conversion family

## R23-268

```text
period_lag
period_change
period_average
period_cagr
quarter_from_cumulative
ttm_from_quarterly
ttm_from_cumulative
yoy_by_period
```

重点：

```text
strict implementation authority
fiscal quarter
timeframe
consecutive periods
revision policy
```

---

# 98. Quality/accrual family

## R23-269

逐动态 canonical：

```text
working capital accrual
total operating accrual
delta NOA
earnings-cash gap
smoothness
persistence
core/noncore earnings
comprehensive income
OCI
minority/discontinued
```

重点：

```text
stock/flow mixing
YTD
multi-statement bundle
missing period
```

---

# 99. Revenue/asset quality family

## R23-270

```text
receivable-sales
inventory-sales
cash-sales
expense-sales
contract asset/liability
```

重点：

```text
per-slot grain contract
same comparison interval
```

---

# 100. Financing / capex / R&D family

## R23-271

逐个确认：

```text
flow horizon
annualization
sign convention
cash-flow sign
stock denominator
```

---

# 101. Composite score family

## R23-272

```text
Piotroski
Altman
fundamental strength
component score
```

重点：

```text
input construction
applicability
coverage
cross-sectional staleness
axis semantics
```

---

# 102. Expectation/revision family

## R23-273

重点：

```text
pre-event snapshot
consensus vintage
actual knowledge time
revision event
same target period
```

---

# 103. Valuation family

## R23-274

重点：

```text
derived vendor denominator PIT
current-snapshot-only source
price decision time
fundamental vintage
```

---

# 104. Shareholder/relation family

## R23-275

重点：

```text
one-to-many snapshot
entity identity
PubDate
period_id
aggregation before ffill
```

---

# 105. Event/corporate-action family

## R23-276

重点：

```text
knowledge vs effective
future dated events
currency
```

---

# 106. 不要只用 synthetic tests

## R23-277

fundamental至少抽真实数据：

```text
A股 20–50 stocks
多个财报期
含同日多period
含缺失字段
```

## R23-278

US抽：

```text
non-calendar fiscal year
multiple timeframe
late filing
sparse fields
```

---

# 107. 真实数据做 bitemporal replay

## R23-279

对一个decision date：

只允许加载：

```text
当时可知vintages
```

再与今天完整数据切片结果比较。

如果不同：

必须证明 difference来自合法后续revision，而不是 historical leakage。

---

# 108. DataAccess 与 FactorEngine联合证书

## R23-280

Fundamental production certification不能只在 FactorEngine单边完成。

最终证书依赖：

```text
DataAccess source temporal certificate
+
FactorEngine operator temporal certificate
```

---

# 109. SourceRef必须携带 temporal contract digest

## R23-281

包含：

```text
knowledge column
period column
revision policy
timeframe
same-day policy
vintage policy
```

---

# 110. Formula identity必须包括财务口径

## R23-282

两个表面一样的公式：

```text
fin_growth(revenue)
```

如果一个输入是：

```text
quarterly
```

另一个：

```text
TTM
```

不是同一个 factor。

---

# 111. 矿工必须生成 typed financial AST

## R23-283

例如：

```text
fin_growth(CumulativeYTDFlow)
```

compile-time直接拒绝。

## R23-284

正确：

```text
fin_growth(
  quarter_from_cumulative(CumulativeYTDFlow)
)
```

---

# 112. Financial conversion不算普通 alpha depth

## R23-285

像：

```text
quarterize
TTM conversion
PIT alignment
currency normalization
```

属于 semantic/source preprocessing。

不要浪费alpha tree depth。

---

# 113. 冷启动库也必须重新 PIT type-check

## R23-286

所有 fundamental seed：

```text
compile semantic
source context
PIT
flow grain
timeframe
```

重新验证。

## R23-287

旧 formula如果隐式对 YTD做 QoQ：

迁移或删除。

---

# 114. Cross-market cold-start

## R23-288

同一个 seed如果 A/US定义不能同构：

```text
market-specific recipe
```

而不是强行一份表达式。

---

# 115. Machine audit：禁止“标签说PIT所以PASS”

## R23-289

PIT PASS只能来自：

```text
declared temporal contract
+
source evidence
+
differential temporal tests
+
real data replay
```

---

# 116. 生成真正的 blocker codes

## R23-290

至少：

```text
PIT01_KNOWLEDGE_TIME_UNKNOWN
PIT02_SAME_DAY_AMBIGUOUS
PIT03_PERIOD_USED_AS_KNOWLEDGE
PIT04_INGESTION_USED_AS_REVISION
PIT05_VINTAGE_HISTORY_UNPROVEN
PIT06_CROSS_SOURCE_BUNDLE_MISMATCH
PIT07_TIMEFRAME_MISMATCH
PIT08_FLOW_GRAIN_MISMATCH
PIT09_PERIOD_GAP_BRIDGED
PIT10_CURRENT_PERIOD_REGRESSION
PIT11_EXPECTATION_POST_EVENT_LEAK
PIT12_CURRENT_SNAPSHOT_BACKFILL
PIT13_UNIVERSE_SURVIVORSHIP
PIT14_GROUP_CLASSIFICATION_LOOKAHEAD
PIT15_EFFECTIVE_EVENT_PREKNOWN
PIT16_CORPORATE_ACTION_FUTURE
PIT17_LEFT_CENSOR_BIAS
PIT18_REVISION_EVENT_UNPROVEN
PIT19_DERIVED_VENDOR_RATIO_VINTAGE_UNKNOWN
PIT20_DECISION_TIME_UNDECLARED
```

---

# 117. 当前已确认需要重点修/验证的 concrete code points

## R23-291

`fields/catalog_us.py`

问题：

```text
strict_pit_allowed默认None后，
多个 production-readable table仍未显式声明。
```

整改：

```text
逐表 explicit temporal eligibility
```

---

## R23-292

`fundamental/transforms_v2.py::_walk_periods`

问题：

```text
silently reindex period_id
```

整改：

```text
strict align
```

---

## R23-293

`fundamental/quality_v2.py::_walk_two`

问题：

```text
silently reindex period_id + secondary
```

整改：

```text
strict align + bundle proof
```

---

## R23-294

`fundamental/quality_v2.py`

问题：

```text
部分 periods = max(... int(periods))
```

整改：

```text
strict parameter authority
```

---

## R23-295

`fundamental/accruals_scores.py`

问题：

```text
_growth/_delta直接int(periods)
```

整改：

```text
strict int
```

---

## R23-296

`fundamental/accruals_scores.py::_divergence`

问题：

```text
same-flow-grain模型不能表达 Stock growth vs Flow growth
```

整改：

```text
per-slot grain contract
```

---

## R23-297

`fundamental/transforms_repairs_v2.py revision family`

问题：

```text
daily state change != proven historical revision
```

整改：

```text
RevisionPair / bitemporal source
```

---

## R23-298

`fin_days_since_update`

问题：

```text
first sample observation age=0 left-censor bias
```

整改：

```text
knowledge-time derived age
```

---

## R23-299

`expectation_v2 surprise family`

问题：

```text
actual/expected arithmetic本身不能证明 expected是pre-event snapshot
```

整改：

```text
PreEventExpectationSnapshot
```

---

## R23-300

`fundamental/component_score.py`

问题：

```text
standard FactorEngine panel columns=instruments，
实现把columns当components
```

整改：

```text
multi-panel component contract / ComponentStack
```

---

## R23-301

`fundamental/ops.py` + `fiscal_strict.py`

问题：

```text
同 canonical先旧实现注册、后strict replace
```

整改：

```text
single registration authority + import-order invariant
```

---

## R23-302

A股 financial source

问题：

```text
PubDate解决announcement PIT，
但UpdateTime不能证明historical revision vintage
```

整改：

```text
separate announcement PIT / vintage PIT certificates
```

---

# 118. Per-operator remediation规则

## R23-303

如果数学正确但 PIT source缺：

```text
BLOCKED_CONTEXTUAL
```

不要删数学能力。

## R23-304

如果定义本身错：

```text
FIX_MATH
```

## R23-305

如果名字错：

```text
RENAME + compatibility alias
```

## R23-306

如果只有旧in-sample版本：

```text
ResearchTool + causal sibling
```

## R23-307

如果数据永远无法支持：

```text
DELETE_NO_DATA
```

---

# 119. 所有 fundamental canonical最终至少一个合法 market recipe

## R23-308

例如：

```text
A-only
US-only
A+US with different source binding
```

必须明确。

---

# 120. 无合法 context不能伪装 Direct

## R23-309

```text
math direct
source unavailable
```

可以保留 capability，

但 default production miner不得生成。

---

# 121. Final PIT certificate hard gates

## R23-310

全部必须0：

```text
CERTIFIED_OPERATOR_WITH_UNKNOWN_DECISION_TIME
CERTIFIED_OPERATOR_WITH_UNKNOWN_KNOWLEDGE_TIME
CERTIFIED_FINANCIAL_WITH_PERIOD_AS_KNOWLEDGE
CERTIFIED_FINANCIAL_WITH_INGESTION_AS_REVISION
CERTIFIED_FINANCIAL_WITH_UNPROVEN_VINTAGE
CERTIFIED_MULTI_STATEMENT_WITHOUT_BUNDLE_POLICY
CERTIFIED_US_FINANCIAL_WITHOUT_TIMEFRAME
CERTIFIED_A_YTD_GROWTH_WITHOUT_CONVERSION
CERTIFIED_FLOW_OPERATOR_WITH_OPTIONAL_UNBOUND_FLOW_SEMANTICS
CERTIFIED_PERIOD_OPERATOR_BRIDGING_ILLEGAL_GAP
CERTIFIED_REVISION_OPERATOR_WITHOUT_REVISION_EVENT_SOURCE
CERTIFIED_SURPRISE_WITHOUT_PRE_EVENT_EXPECTATION
CERTIFIED_SAME_DAY_DATE_ONLY_WITHOUT_AVAILABILITY_POLICY
CERTIFIED_CURRENT_SNAPSHOT_HISTORY_BACKFILL
CERTIFIED_FUNDAMENTAL_WITH_LEFT_CENSOR_MISLABELED_AS_AGE_ZERO
CERTIFIED_OPERATOR_WITH_SILENT_PANEL_REINDEX
```

---

# 122. 全局 per-canonical gates

## R23-311

全部必须0：

```text
CANONICAL_WITHOUT_SEMANTIC_PIT_CERTIFICATE
CANONICAL_AUDIT_SKIPPED
CANONICAL_AUDIT_EXCEPTION_SWALLOWED
CANONICAL_WITH_UNKNOWN_INPUT_SEMANTIC
CANONICAL_WITH_UNKNOWN_OUTPUT_SEMANTIC
CANONICAL_WITH_PARAMETER_COERCION
CANONICAL_WITH_UNPROVEN_TERMINAL_ROLE
CANONICAL_WITH_BACKEND_SEMANTIC_DRIFT
```

---

# 123. Artifacts

## R23-312

至少生成：

```text
factor_engine/docs/R23_PER_CANONICAL_AUDIT.csv
factor_engine/docs/R23_PER_CANONICAL_AUDIT.json
factor_engine/docs/R23_PER_CANONICAL_AUDIT.md
```

## R23-313

```text
factor_engine/docs/R23_FUNDAMENTAL_OPERATOR_AUDIT.csv
factor_engine/docs/R23_FUNDAMENTAL_OPERATOR_AUDIT.json
factor_engine/docs/R23_FUNDAMENTAL_OPERATOR_AUDIT.md
```

## R23-314

```text
factor_engine/docs/R23_TEMPORAL_SOURCE_CERTIFICATES.json
factor_engine/docs/R23_FINANCIAL_BUNDLE_CONTRACTS.json
factor_engine/docs/R23_FLOW_SEMANTICS_MATRIX.json
factor_engine/docs/R23_REVISION_VINTAGE_MATRIX.json
factor_engine/docs/R23_SAME_DAY_AVAILABILITY_MATRIX.json
```

## R23-315

```text
factor_engine/docs/R23_OPERATOR_REMEDIATION_PLAN.json
factor_engine/docs/R23_OPERATOR_REMEDIATION_PLAN.md
```

---

# 124. Test suites

## R23-316

增加：

```text
tests/operators/test_all_canonical_semantic_pit_certificate.py
tests/fundamental/test_knowledge_vs_period_time.py
tests/fundamental/test_same_day_availability.py
tests/fundamental/test_revision_vintage_pit.py
tests/fundamental/test_old_period_restatement.py
tests/fundamental/test_same_day_multi_period_filing.py
tests/fundamental/test_flow_semantics_binding.py
tests/fundamental/test_us_timeframe_identity.py
tests/fundamental/test_financial_bundle_identity.py
tests/fundamental/test_left_censor_staleness.py
tests/fundamental/test_expectation_pre_event_snapshot.py
tests/fundamental/test_fiscal_gap_topology.py
tests/fundamental/test_import_order_strict_period_ops.py
```

---

# 125. Real data audit

## R23-317

从 A股字典和实际数据抽：

```text
StockIncome
StockBalance
StockCashFlow
StockIndicator
StockTopTenShareholder
StockValuationDaily
```

## R23-318

US抽：

```text
StockIncome
StockBalance
StockCashFlow
StockDividend
TickerSharesSnapshot
```

---

# 126. 不要把 current GitHub green当证书

## R23-319

最终以服务器真实：

```text
HEAD
dirty diff
generated manifests
tests
```

为准。

---

# 127. 执行顺序

## R23-320

按这个顺序一次性执行：

1. 读取 server HEAD/dirty；
2. `load_all()`；
3. 枚举全部 canonical；
4. 枚举全部 A/US TableSpec/FieldSpec；
5. 建 temporal source matrix；
6. 修 US strict_pit explicit declarations；
7. 建 decision-time/knowledge-time policy；
8. 建 date-only next-session policy；
9. 审历史 vintage/restatement；
10. 建 FinancialStatementBundleIdentity；
11. 建 FundamentalPeriodState；
12. flow semantics改为compile-time binding；
13. timeframe进入 SourceRef/cache/factor identity；
14. strict alignment替代财务kernel reindex；
15. strict params替代所有 int coercion；
16. 修 component_score axis；
17. 修 revision source contract；
18. 修 staleness left censor；
19. 修 expectation pre-event snapshot；
20. 修 A股 YTD default recipes；
21. 修 US fiscal/timeframe recipes；
22. 修 divergence per-slot grains；
23. 修 static ratios bundle/timeframe；
24. 修 valuation derived-vintage PIT；
25. 修 shareholder/relation snapshot PIT；
26. 修 corporate action effective/knowledge clocks；
27. 对其它所有 operator做 per-canonical PIT certificate；
28. temporal differential tests；
29. real data replay；
30. backend parity；
31. incremental/revision invalidation；
32. 重生 direct mining catalog/cold-start；
33. 跑所有 tests；
34. 再动态枚举，确保0 skip；
35. 输出最终 closure report。

---

# 128. 最终汇报

## R23-321

必须汇报真实数：

```text
Total canonicals:
Audited canonicals:
Certified:
Certified contextual:
High-cost certified:
Supporting only:
Research tools:
Internal:
Deleted:
Blocked no data:

Fundamental canonicals:
Fundamental certified:
Fundamental PIT blocked:
Vintage PIT unproven:
Same-day ambiguity:
Flow-grain mismatch:
Timeframe mismatch:
Bundle mismatch:
Left-censor issues:
Revision-source issues:
Expectation pre-event issues:
Silent reindex issues:
Parameter coercion issues:

Temporal tests passed:
Real-data replay passed:
Backend parity passed:
Full tests:
```

## R23-322

最后必须：

```text
ALL_CANONICALS_SEMANTICALLY_AUDITED=true/false
FUNDAMENTAL_PIT_CLOSED=true/false
REVISION_VINTAGE_PIT_CLOSED=true/false
```

只有所有硬 gate为0才可写 true。

---

# 129. 这一轮最重要的原则

**基本面 PIT 必须从“一个日期字段”升级成“真正双时态/多时钟数据模型”。**

一个财务数字至少要回答：

```text
它描述哪个期间？
最初什么时候市场知道？
后来什么时候被修订？
当前策略什么时候做决策？
今天看到的这个值，是不是当年真的已经知道？
```

一个财务 operator还必须回答：

```text
输入是YTD还是单季？
是Stock还是Flow？
多个statement是否同report bundle？
missing quarter能不能跨？
旧期重述是否会让current period倒退？
预期是不是财报公布前的预期？
同一天PubDate在开盘时是否真的已经知道？
```

任何一个问题答不上来，都不能只靠 `pit_safe` tag宣布 production-safe。

---

# 130. 最终指令

现在直接在服务器工作区执行。

**不要只回复“已经检查”。**

必须：

```text
逐 canonical
逐 field
逐 source
逐 market
逐 fundamental family
```

生成机器可读证书、修代码、加测试、跑真实数据。

本轮结束的真正标准不是：

```text
测试没报错
```

而是：

```text
每一个 retained operator 的数学定义、输入语义、时间语义、PIT、
财报口径、revision/vintage、市场上下文都有可证明的机器契约。
```

尤其 fundamental：

```text
ReportPeriod != KnowledgeTime
KnowledgeTime != IngestionTime
Initial Filing != Revision Vintage
YTD != Quarter
Quarter != TTM
Same Date != Same Decision Time
Individually PIT != Cross-source PIT
Latest Today != Latest Available Then
```

**把这些全部做到机器可证明，才能真正说 FactorEngine 的“每个算子，尤其基本面算子”可以放心交给自动化因子挖掘。**
