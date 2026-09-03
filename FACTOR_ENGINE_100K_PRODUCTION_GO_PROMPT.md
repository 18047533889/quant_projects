# FactorEngine × DataAccess：10 万因子生产落值前最终审计与 Go 执行 Prompt

> **用途**：直接交给 Claude Code / AI Cloud Code / Codex 类代码 Agent。  
> **目标**：持续使用 subagents 对 `quant_projects` 中的 `factor_engine` 与 `data_access` 做正确性、算子覆盖、分钟频性能、多后端、并发与生产门禁整改，直到满足本文的 Go-Live 条件。  
> **仓库**：`https://github.com/18047533889/quant_projects`  
> **审计基线**：以执行时 `main` 最新 HEAD 和本地 working tree 为唯一事实来源；本文审计时观察到的 HEAD 为 `7628674b95348cc3227881546a7d21ed8cd50881`。如果 HEAD 已变化，必须重新生成所有 inventory/evidence/benchmark，不得复用旧结论。  
> **最终业务目标**：FactorEngine 对 Agent 暴露一套**真正可以直接安全使用**的算子集合，并以尽可能高的吞吐批量计算约 100,000 个因子，而不是让 Agent 看到“注册了很多、但语义/后端/性能/证据不确定”的算子。

---

# 0. 最高级指令：不要把“能 import / 有实现 / 单测过”当成“Agent 可直接用”

这次工作的判定标准不是：

- 算子出现在 registry；
- Pandas 能跑；
- 有一个 unit test；
- README 写了多少算子；
- 某个历史 evidence 曾经 PASS；
- 单因子跑出了一个 Series。

真正的 `AgentDirectOperator` 必须同时满足：

1. **注册正确**：canonical / alias / surface / lifecycle / execution_kind 无冲突；
2. **字段语义正确**：输入字段的单位、频率、时间模型、grain/cardinality、PIT 语义正确；
3. **参数正确**：所有窗口、min_periods、ddof、adjust、tie method、方向等参数有明确 contract；
4. **数学正确**：存在可信 oracle / invariant / reference；
5. **因果正确**：prefix causality / no-lookahead / PIT 检查通过；
6. **边界正确**：NaN、inf、0 分母、常数列、短样本、停牌、缺 bar、奇异矩阵等行为明确；
7. **跨后端一致**：Agent 实际可能路由到的 backend 与 Pandas canonical semantics 在容差内一致；
8. **分块一致**：full-run == chunked/incremental-run（考虑 warmup/state carry）；
9. **序列化一致**：DSL parse → IR → lower → execute 与 Python/reference 路径一致；
10. **性能达标**：没有明显 Python per-row/per-group 热循环；高成本算子有 cost class；
11. **生产证据绑定当前源码**：evidence 必须绑定当前 source/tree hash；
12. **可观测**：执行后能知道 backend、fallback、scan bytes、cache/CSE/fusion hit、wall time；
13. **失败关闭**：任何关键证据缺失时，不允许向 Agent 宣称“可直接生产使用”。

最终只向 Agent 暴露类似：

```python
AgentDirectOperatorSet(
    terminal=True,
    production_admitted=True,
    current_source_verified=True,
    field_contract_verified=True,
    backend_verified=True,
)
```

的集合。

**绝对禁止**简单把全部 canonical operator 全开放给 Agent。

---

# 1. 开始工作前必须做的事实重建

先执行，不要修改代码：

```bash
git status --short
git rev-parse HEAD
git log -20 --oneline --decorate

cd factor_engine
python -m cleaned_operators.audit
# 或仓库当前实际存在的 operator audit / registry audit 入口

pytest -q
```

同时输出并保存：

```text
artifacts/preflight/
├── git_head.txt
├── working_tree_status.txt
├── operator_inventory.json
├── alias_inventory.json
├── surface_inventory.json
├── direct_use_matrix.json
├── backend_matrix.json
├── field_registry.json
├── source_contract_matrix.json
├── test_baseline.txt
└── benchmark_baseline.json
```

必须实时统计，而不是引用 README 中的静态数字：

- canonical operator count；
- DSL allowlist count；
- daily / extended / research / unsafe / legacy；
- production / experimental / research / deprecated；
- primitive / composite / stateful / external kernel；
- terminal direct-use 数；
- production-admitted direct-use 数；
- 各 backend implemented / selectable / parity / evidence / production admitted；
- intraday operator 数；
- intraday vectorized operator 数；
- scalar fallback intraday operator 数；
- source-backed operator 数。

## 当前已观察到的风险信号

审计时 README 声明约：

```text
canonical operators ≈ 1737
DSL allowlist ≈ 1421
daily ≈ 1238
extended ≈ 480
production lifecycle ≈ 137
experimental lifecycle ≈ 1593
```

但 `docs/BACKEND_COVERAGE.md` 仍来自旧 HEAD，记录的是约 1624 个 canonical，且旧矩阵中的 `production_admitted=0`。因此：

> **后台覆盖报告、README、registry 当前状态并不处于同一个版本事实面。**

第一项工作就是重新生成所有权威报告。

---

# 2. 使用现有 Claude/subagent 体系，不要单 Agent 串行硬改

仓库已经存在：

```text
.claude/skills/factor-engine-rules/SKILL.md
.claude/loop/start_fe.md
.claude/agents/coordinator-fe.md
```

请沿用其 finder → dispatcher → writer → tester → reviewer → integration 的循环，但增加以下专门 subagents。

## 2.1 必须并行派出的 subagents

### A. `operator-correctness-auditor`

职责：

- 枚举全部 Agent 候选算子；
- 为每个 canonical 算子确定数学定义；
- 建立 oracle / invariant；
- 检查参数和边界；
- 检查 alias 等价；
- 检查 composite lowering parity；
- 输出逐算子 correctness matrix。

### B. `backend-parity-auditor`

职责：

- Pandas canonical vs Polars panel；
- Pandas canonical vs Polars long；
- Pandas canonical vs DuckDB SQL；
- 只有真正 runtime-wired 的 backend 才算；
- 生成 per-operator parity evidence；
- 检查 backend fallback 是否静默改变语义。

### C. `intraday-performance-profiler`

职责：

- 定位分钟因子 10–20 分钟的时间构成；
- scan/decode/sessionize/pivot/groupby/operator/write 分阶段 profiling；
- 检查 vector kernel 绑定；
- 修复 vector benchmark；
- 把适合的分钟算子转为批量 Polars/DuckDB/NumPy/Numba；
- 建 feature bundle 与 materialization。

### D. `data-access-io-profiler`

职责：

- COS/local mirror/Parquet scan；
- Arrow/Polars/DuckDB read path；
- predicate/column pushdown；
- partition pruning；
- cache；
- duplicate reads；
- concurrent remote reads；
- materialized minute/daily feature reads；
- 输出 IO flame/profile。

### E. `multiworker-resource-auditor`

职责：

- CPU oversubscription；
- DuckDB threads；
- Polars threads；
- BLAS/OpenMP threads；
- process vs thread；
- RAM admission；
- DA remote concurrency；
- cross-process governor；
- worker scaling 1/2/4/8 benchmark。

### F. `batch-dag-cse-auditor`

职责：

- 100k roots 的 parse/compile/graph memory；
- structural hash dedup；
- CSE hit rate；
- native fusion hit rate；
- read-wave reuse；
- microbatch scheduling overhead；
- result streaming；
- CSE node lifetime/release。

### G. `field-pit-auditor`

职责：

严格按 A 股数据契约审：

- Return bp；
- percent fields；
- E1 财务 PubDate；
- YTD flow；
- dividend effective-only；
- industry single source；
- relation grain；
- index constituent grain；
- minute UTC/session；
- UpdateTime 非 PIT；
- actual HighLimit/LowLimit；
- 无 Level2/Tick 时 proxy 命名。

### H. `benchmark-regression-guardian`

职责：

建立永久 benchmark，不允许优化后语义或性能回退。

### I. `integration-reviewer`

只做 review，不写主要实现：

- 检查 subagent 修改冲突；
- 查 shortcut / silent fallback；
- 查“为了通过测试而改测试”；
- 查旧 evidence 被错误继承；
- 最终 Go/No-Go。

---

# 3. A 股数据语义：这是不可违背的硬约束

以下不是建议，是 FactorEngine / DataAccess 的输入契约。

## 3.1 日频收益与百分比

```text
StockDailyBar.Return:
    raw unit = bp
    decimal_return = Return / 10000

StockValuationDaily.TurnoverRatio:
    raw unit = %
    decimal = /100

StockValuationDaily.DividendRatio:
    raw unit = %
    decimal = /100

StockIndicator:
    Roe/Roa/margins/growth = %
    decimal = /100

TopTen.ShareRatio:
    raw unit = %
    decimal = /100

IndexConstituent.Weight:
    raw unit = %
    decimal = /100
```

所有读路径必须统一：

```text
raw physical field
→ FieldSpec
→ canonical unit normalization
→ cache
→ operator
```

禁止：

- Pandas 路径归一化，但 lazy/Polars 路径不归一化；
- alias `ret` 绕过物理字段 `Return` 的 `/10000`；
- operator 自己猜单位；
- 不同 backend 使用不同单位。

必须写跨路径测试：

```text
read direct
prefetch
lazy scan
Polars long
DuckDB
source ref
batch read
```

得到相同 canonical value。

---

## 3.2 财务 PIT

四张表：

```text
StockBalance
StockIncome
StockCashFlow
StockIndicator
```

全部：

```text
knowledge time = PubDate
period id = ReportPeriodEndDate
```

禁止使用：

```text
ReportPeriodEndDate 作为可知时点
TradeDate equi-join 到日线
UpdateTime 作为公告时点
```

同一个 `PubDate` 可以同时出现多个报告期。

正确策略必须是显式：

```text
latest_visible_period
annual_only
quarterly_only
fixed_period
```

对于 production 默认：

```text
latest_visible_period
```

并处理旧报告期 revision：

```text
先选择决策时点可见记录
→ 再选最大 ReportPeriodEndDate
→ 再选该 period 最新合法 revision
```

不是简单“最近一次 PubDate”。

如果 PubDate 只有日期没有公告具体时刻，必须定义 decision-time policy。日频收盘后信号与开盘前信号不能使用同一 availability assumption。

建议：

```text
daily_close_decision:
    same_day PubDate allowed only if business convention明确

daily_open_decision:
    date-only PubDate default effective next trading day
```

必须在 lineage 保存 policy。

---

## 3.3 中国财务累计口径

利润表和现金流中：

```text
Q1 = Q1 cumulative
H1 = Q1+Q2 cumulative
Q3 = Q1+Q2+Q3 cumulative
FY = full year
```

FieldSpec 必须区分：

```text
flow_ytd
flow_quarterly
balance
ratio
per_share
```

`flow_ytd` 不能直接进入普通：

```text
ts_delta
ts_pct_change
ts_sum
rolling_mean
```

除非算子明确支持 cumulative semantics。

应优先使用：

```text
fin_quarter_from_cumulative
fin_ttm_cumulative
fin_yoy
fin_qoq
```

并验证缺季度、revision、年报切换。

---

## 3.4 D1 / S1 / E1 / effective / MINUTE

必须统一 source contract dispatcher。

```text
D1:
    exact/equi

S1:
    state-ready exact daily snapshot
    不应随意再次 asof，除非源合同明确要求

E1 financial:
    PIT asof PubDate

E1 effective:
    explicit effective-time only
    strict PIT 默认拒绝

MINUTE:
    QuoteTime axis
```

不能让不同入口各自 if/else 实现一套时间逻辑。

---

## 3.5 行业

`StockIndustry` 每股每天约 6 个 IndustrySource。

任何 group/neutralization 前必须显式选：

```text
sw_l1
sw_l2
sw_l3
zjw
jq_l1
jq_l2
```

默认推荐 `sw_l1`，但默认值必须可追踪。

未指定 source：

```text
production => fail
```

不能让 join 后行数 ×6。

---

## 3.6 股东关系

TopTen grain：

```text
TradeDate × Symbol × ShareholderRank/ShareholderId
```

不能直接 pivot 为一股一日单值。

必须先 relation aggregate：

```text
top1/top3/top5/top10
hhi
entropy
institution ratio
pledge/freeze
entity churn
new/exit entity
```

任何变化：

```text
rel_lag_by_snapshot
```

而不是日频前填后的 `.shift(1)`。

---

## 3.7 指数

`IndexConstituent` grain：

```text
TradeDate × IndexSymbol × Symbol
```

必须强制显式 `IndexSymbol`。

Weight `/100`。

Benchmark 日线按日期精确广播，不做错误 backward asof。

---

## 3.8 分红

`StockDividend` 无可靠 announcement PIT。

严格生产：

```text
strict_pit => reject
```

只有 explicit:

```text
allow_effective_time=True
```

时，才能按 `ExDividendDate` 做研究事件。

---

## 3.9 分钟

原始：

```text
QuoteTime = UTC
```

本地 A 股 session：

```text
09:31–11:30
13:01–15:00
240 bars/day mode
```

必须：

```text
UTC
→ Asia/Shanghai
→ trade_date
→ session_id
```

rolling key 必须至少包含：

```text
Symbol
TradeDate
session_id
```

禁止：

- 跨午休 rolling；
- 跨股票 rolling；
- 跨交易日 pct_change；
- 假定存在 09:30 或 13:00 bar。

数据没有 Tick / Level2，因此：

```text
VPIN
Kyle lambda
spread
order imbalance
```

如果只由 OHLCV 推导，必须加 `_proxy`，不能冒充真实微观结构。

涨跌停使用实际 `HighLimit/LowLimit`，不要优先用代码规则推算。

---

# 4. Agent 可直接用算子：逐算子全面审计

不要人工打开 1700 个文件逐个拍脑袋判断。建立**自动化矩阵 + family-specific oracle**。

最终输出：

```text
artifacts/operator_audit/current_head/
├── operator_matrix.parquet
├── operator_matrix.csv
├── failures/
├── parity/
├── oracles/
├── causality/
├── parameters/
├── performance/
└── direct_agent_allowlist.json
```

每个 canonical 一行，至少有：

```text
canonical
aliases
surface
lifecycle
execution_kind
terminal_or_intermediate
direct_use_status
agent_visible
production_admitted

pandas_impl
polars_panel_impl
polars_long_impl
duckdb_impl

math_oracle
oracle_pass
deterministic_pass
shape_pass
index_pass
dtype_pass
nan_edge_pass
parameter_pass
prefix_causality_pass
pit_pass
chunk_full_pass
incremental_full_pass
backend_parity_pass
lowering_parity_pass
alias_parity_pass

lookback_contract
warmup_contract
frequency_contract
unit_contract
cardinality_contract
source_contract

cold_runtime_ms
warm_runtime_ms
peak_rss_mb
cost_class

source_hash
evidence_hash
tested_head
```

只要 Agent-visible terminal operator 任意关键列为空或 false：

```text
agent_visible = false
```

---

# 5. 算子 family-specific 正确性检查

## 5.1 Arithmetic / transform

检查：

```text
add/sub/mul/div
safe_div
signed_log
log/log1p
sqrt/power
clip
abs/sign
where
```

重点：

- denominator=0；
- `0/0`；
- ±inf；
- signed power；
- negative input；
- NaN propagation；
- dtype；
- Unit inference。

---

## 5.2 Rank / cross-sectional

检查：

```text
cs_rank
percentile
zscore
winsor
mad
group_rank
group_zscore
neutralize
```

必须固定：

```text
tie method
rank range
NaN policy
min cross-section
ddof
zero std
```

同一日输入行顺序打乱，结果必须不变。

---

## 5.3 Rolling / TS

检查：

```text
delay
delta
pct_change
rolling sum/mean/std/min/max/median
rank
corr/cov
beta
argmax/argmin
```

必须验证：

```text
window
min_periods
ddof
pairwise NaN
sort order
duplicate timestamp
warmup
chunk boundary
```

随机 prefix truncation 后，历史输出不得变化。

---

## 5.4 EWM / decay

必须固定：

```text
adjust
ignore_na
alpha/span/halflife
initialization
normalization
weight direction
```

Pandas/Polars/SQL 不得各自使用不同 EWM 定义。

---

## 5.5 Corr / regression / neutralization

检查：

- intercept 是否存在；
- rank deficiency；
- zero variance；
- singular matrix；
- min sample；
- weighted regression；
- residual definition；
- rolling beta；
- market/industry neutralization；
- NaN pair alignment。

---

## 5.6 Technical indicators

对：

```text
ATR
RSI
MACD
ADX
CCI
OBV
MFI
Bollinger
stochastic
candlestick/pattern
```

每个家族必须有公式 reference/oracle。

不允许：

- Pandas 用一种标准，Polars 用另一种初始化；
- TA-Lib 结果被误当成唯一正确答案而不记录其 warmup 定义；
- 形态识别使用未来 K 线。

---

## 5.7 Stateful operators

必须额外验证：

```text
full run == 2 chunks == N chunks
```

包括：

- state serialization；
- state carry；
- restart；
- checkpoint resume；
- missing partition；
- empty chunk。

---

## 5.8 Composite / recipe

必须测试：

```text
reference Python implementation
==
lowered primitive graph
==
backend execution
```

特别查：

- Pandas callable 做了 domain guard，但 lowering 没做；
- composite 参数在 lowering 中遗漏；
- alias 调错 canonical；
- unit normalization重复或遗漏。

---

## 5.9 Financial

除普通数值测试外，必须使用 synthetic reporting calendar：

```text
Q1
H1
Q3
FY
revision
same PubDate multiple periods
missing quarter
late filing
```

检查：

```text
fin_lag
fin_qoq
fin_yoy
fin_quarter_from_cumulative
fin_ttm_cumulative
fin_growth
fin_cagr
fin_trend
fin_average_balance
```

禁止按交易日 shift。

---

## 5.10 Relation / shareholder

检查：

```text
ratio percent→decimal
HHI
entropy
top-k
duplicates
same entity duplicate rows
snapshot changes
entity new/exit
```

---

## 5.11 Intraday

每个算子至少测：

- 240 bars；
- 少 1 bar；
- 午休；
- 停牌空日；
- 单股票；
- 多股票；
- unsorted rows；
- duplicate minute；
- all zero volume；
- NaN amount；
- limit up/down；
- DST 不相关但 timezone must be explicit；
- UTC conversion；
- first/last N minutes。

---

# 6. 当前分钟频 10–20 分钟：先修这些明确问题

## 6.1 `daily_agg` 的 fast path 覆盖太少

当前 intraday `_core.py` 的模式大致是：

```python
if feature has __vec__:
    run vector kernel
else:
    group date×symbol
    Python scalar path
```

但是当前 `perf_vec_kernels.py` 实际只绑定了极少几个 feature。

已观察到：

```text
vec_time_efficiency
vec_twap_deviation
vec_intraday_vwap
vec_volume_cv
```

存在 vector kernel，但 `bind_whitelist_kernels()` 实际主要只绑定：

```text
time_efficiency
twap_deviation
volume_cv
```

`vec_intraday_vwap` 并没有形成完整绑定。

这意味着大量分钟特征仍可能走：

```text
date × symbol group
→ Python callable
→ scalar aggregation
```

对于 5000 股票 × 250 日 × 多年会极慢。

### 必须整改

自动生成：

```text
intraday_vector_coverage.json
```

列：

```text
canonical
scalar_impl
vector_impl
vector_bound
vector_equivalence
speedup
fallback_reason
```

要求：

- 所有已写 vector kernel 必须要么绑定，要么删除；
- 不能存在“实现了但从不执行”的 kernel；
- eligible 高频分钟特征优先全部 vectorize；
- scalar fallback 必须有 telemetry。

---

## 6.2 当前 vector benchmark 本身有 bug，不能相信 speedup

`perf_vec_bench.py` 当前 benchmark 的关键模式是类似：

```python
daily_agg(frame, lambda v, t: fn(v, t))
```

但 `__vec__` 属性挂在 `fn` 上，而 lambda 没有 `__vec__`。

所以：

```text
所谓 vector benchmark
很可能仍在跑 scalar path
```

必须立即修。

正确 benchmark：

```python
daily_agg(frame, fn_with_vec_attached)
```

vs：

```python
scalar_fn = clone_without_vec(fn)
daily_agg(frame, scalar_fn)
```

并增加 assertion：

```text
vector_path_counter > 0
scalar_fallback_counter == 0
```

否则 benchmark fail。

---

## 6.3 vector binding 不能 `except Exception: pass`

当前 import/bind 逻辑存在 broad catch。

性能加速如果因为：

- Polars import；
- circular import；
- feature rename；
- attribute failure；

失效，不能悄悄退回慢路径。

改为：

```text
research:
    warning + metric

production performance profile:
    required vector set missing => fail launch benchmark
```

增加：

```text
intraday_vector_kernels_bound
intraday_vector_bind_failures
intraday_scalar_fallback_count
```

---

# 7. 分钟频架构：不要“一个因子扫一次分钟数据”

这是 10 万因子性能最重要的原则。

错误模式：

```text
factor 1
  read minute parquet
  sessionize
  groupby
  aggregate

factor 2
  read same minute parquet
  sessionize
  groupby
  aggregate

...

factor N
```

正确模式：

```text
MinuteFeatureBundle
    ↓
一次 scan / partition
    ↓
一次 timezone/sessionize
    ↓
一次 shared base columns:
        minute_return
        abs_return
        log_return
        typical_price
        vwap
        minute_index
        session flags
    ↓
同时计算几百/几千 intraday features
    ↓
daily materialized feature block
    ↓
10万表达式复用
```

---

# 8. 建立 `IntradayFeatureCompiler`

统一现在可能并存的多套实现：

```text
runtime/intraday_aggregator.py
storage/sources/intraday_feature_extension.py
storage/sources/intraday_feature_runtime_v2.py
LQTPLogicalDataSource minute path
runtime/ashare_intraday.py
cleaned_operators/intraday
```

最终只保留一个权威核心：

```python
IntradayFeatureCompiler
```

其他入口只能 adapter 到它。

接口示例：

```python
bundle = IntradayFeatureCompiler(
    fields=["Open","High","Low","Close","Volume","Amount","Vwap"],
    timezone="Asia/Shanghai",
    session="ashare_regular",
)

out = bundle.compute_many(
    features=[...],
    dates=...,
    universe=...,
)
```

必须做到：

1. 一次 scan；
2. 一次 partition prune；
3. 一次 column projection；
4. 一次 timezone conversion；
5. 一次 session key；
6. shared intermediate；
7. vectorized multiple aggregation；
8. streaming daily block；
9. 可 materialize；
10. lineage 保存 input snapshot hash。

---

# 9. 分钟频建议优先 vectorize 的家族

不是所有 1000+ 分钟特征都需要独立 kernel。先实现 reusable primitives。

## 9.1 Base primitives

```text
minute_ret
minute_log_ret
abs_ret
positive_ret
negative_ret
range
true_range_proxy
typical_price
vwap_dev
volume_share
amount_share
bar_index
session_index
```

## 9.2 Group aggregations

```text
sum
mean
std
var
min
max
first
last
quantile
skew
kurt
argmax
argmin
count
count_if
```

## 9.3 Reusable sufficient statistics

一次聚合：

```text
Σr
Σr²
Σr³
Σr⁴
Σv
Σv²
Σrv
Σ|r|v
Σamount
max drawdown auxiliaries
```

很多特征可从这些统计量组合出来，无需再次扫 240 bars。

## 9.4 Window masks

预计算：

```text
first_5m
first_15m
first_30m
last_5m
last_15m
last_30m
morning
afternoon
14:30-close
```

不要每个 feature 重建 clock mask。

---

# 10. 多后端：明确“谁负责什么”，不要盲目全后端覆盖

推荐职责：

## Pandas/NumPy

作为：

```text
canonical semantic oracle
rare fallback
small-data reference
```

不应承担 100k 因子主吞吐。

## Polars long

优先：

- 分钟长表；
- group aggregation；
- rolling；
- native expressions；
- column projection；
- predicate pushdown；
- multi-root expression。

禁止 hot path：

```text
.to_pandas()
map_elements(Python)
per-group Python UDF
```

## DuckDB

优先：

- Parquet direct scan；
- joins；
- financial PIT/asof；
- relation aggregation；
- index/industry filtering；
- SQL pushdown；
- large group aggregation。

## Hybrid

只允许：

```text
planner明确 region
证据支持
observable fallback
```

不能 silent logical reroute。

---

# 11. 当前调度器与 HybridExecutor 有一处必须 benchmark/修正的冲突

`HybridExecutor` 的设计意图是：

```text
pandas_numpy / research_python:
    GIL-bound → process

Polars / DuckDB / native:
    native/GIL-releasing → thread
```

但是当前 adaptive scheduler 多个 `executor.submit(...)` 调用显式：

```text
prefer="thread"
```

包括普通 task、fusion、microbatch 等路径。

这会导致：

> `HybridExecutor` 的 backend classifier 在这些路径上被覆盖。

结果：

- Pandas CPU-bound 任务仍进 thread pool；
- Python GIL 下多线程扩展差；
- `max_workers` 增加不等于 CPU throughput 增加。

### 不要直接粗暴改成 process

Process pool 有：

- pickle；
- DataFrame copy；
- COW；
- IPC；
- worker warmup；

成本。

### 正确做法

做 A/B benchmark：

```text
A: scheduler forced thread
B: backend auto classify
C: all thread but hot kernels native
D: process for Pandas roots
```

对：

```text
1e6 / 5e6 / 20e6 rows
10 / 100 / 1000 roots
```

测：

```text
wall
CPU utilization
RSS
IPC bytes
serialization_ms
throughput
```

然后：

- 如果 Pandas process 明显收益，取消强制 thread；
- 如果 native conversion 更优，优先消灭 Pandas hot path；
- 最终 scheduler 不能无条件覆盖 executor classifier。

---

# 12. HybridExecutor 需要额外修正

## 12.1 Process pool 创建失败不能静默降 thread

如果 process pool 因环境问题失败：

```text
必须记录
process_pool_available = false
failure_reason
```

production benchmark 不能继续假装混合执行正常。

## 12.2 Nested thread 限制

Process worker 中：

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

或者基于明确资源计划设置。

Polars/DuckDB 线程也必须与 worker budget 配套。

---

# 13. 多 worker：当前不能直接 4/8 worker 放大

这是一个 P0 性能/稳定性问题。

FactorEngine 的 HostResourceCoordinator 与 DataAccess GlobalResourceGovernor 目前主要是：

```text
process-level authority
```

多个独立 OS worker：

```text
worker 1 认为自己有 64 core / 256GB
worker 2 也认为自己有 64 core / 256GB
worker 3 ...
```

则：

- worker_count × duckdb_threads 爆 CPU；
- remote scan concurrency 放大；
- RAM reservation 分别成立但总和超 host；
- local disk/COS cache 被重复读；
- 每个 worker 都认为有独占资源。

DataAccess 代码本身已经明确：

> 多进程部署若没有 shared limiter，需要单 worker contract 或外部共享资源治理。

---

# 14. 推荐 worker 部署方案

## 方案 A：默认推荐

```text
一台机器：
    1 个 FactorEngine 主进程
    内部 AdaptiveBatchScheduler
    内部 Thread/Process/Native hybrid
```

优点：

- CSE 跨全部 roots；
- source read reuse 最大；
- HostResourceCoordinator 真正有效；
- 共享缓存；
- 无重复分钟 scan。

对于单机 10 万因子，这是第一选择。

---

## 方案 B：必须多进程时

每个 worker 必须显式资源分区。

例如 64 CPU / 256GB：

```text
4 workers:

worker cpu = 16
worker RAM budget <= 50~55GB
DuckDB threads <= 16
Polars threads <= 16
BLAS threads = 1
remote concurrency = total budget / 4
```

最好：

```text
cpuset / cgroup
memory.max
```

强制隔离。

禁止四个 worker 都看到 64 CPU 后各开 64 thread。

---

## 方案 C：真正企业级

实现 Host Local Resource Arbiter：

```text
Redis
Postgres advisory/reservation
Unix domain daemon
shared-memory semaphore
```

管理全机：

```text
CPU tokens
RAM reservations
DuckDB thread tokens
remote IO tokens
COS download tokens
cache build locks
```

FactorEngine 与 DataAccess 所有进程从同一个 authority 申请。

---

# 15. 多 worker 不要按“因子”随机切分

错误：

```text
worker1 factor 1-25000
worker2 factor 25001-50000
...
```

如果这 4 组都需要同样分钟数据，就会：

```text
同一 Parquet × 4 扫描
同一 sessionize × 4
同一 base intermediate × 4
```

CSE 也被进程边界切碎。

更好的 shard：

```text
先物化 shared source feature blocks
↓
再按 expression DAG / backend / date partition 分工
```

若必须 outer shard：

- 优先按日期 partition；
- shared daily/minute feature store；
- 每 worker 使用已物化块；
- 因子 DAG 在 shard 内复用；
- 最终 concatenate by date。

但 cross-sectional operator 不能任意按 instrument shard，除非有 global-state stage。

---

# 16. 10 万因子绝不能 10 万次 `run()`

强制：

```python
engine.run_many(
    factors,
    enable_cse=True,
    ...
)
```

或当前最优 batch API。

禁止：

```python
for factor in factors:
    engine.run(factor)
```

因为后者重复：

- parse；
- compile；
- source resolve；
- read；
- warmup；
- intermediate；
- write。

---

# 17. 100k roots 前增加 formula canonicalization

执行前：

```text
DSL parse
→ canonical AST
→ constant normalize
→ alias canonicalize
→ commutative normalization（仅数学允许时）
→ structural hash
→ exact dedup
→ subtree fingerprint
```

输出：

```text
requested_roots
unique_roots
duplicate_roots
unique_subtrees
CSE_reuse_ratio
```

十万个挖掘因子中经常存在大量结构共享。

不做 canonicalization 就浪费算力。

---

# 18. CSE：必须量化命中率

当前已经有 `enable_cse=True`，但不能只看“开关打开”。

增加 telemetry：

```text
cse_candidates
cse_unique_nodes
cse_shared_nodes
cse_reuse_edges
cse_hit_ratio
cse_memory_saved_estimate
cse_compute_saved_estimate
cse_nodes_released
```

并 benchmark：

```text
CSE off
CSE on
```

若 CSE 导致 graph memory 过大，需要：

```text
bounded CSE scope
subtree materialization
campaign sharding
```

而不是直接关掉。

---

# 19. Native fusion：必须看真实 hit rate

已有 native fusion planner 不等于实际执行了 fusion。

增加：

```text
fusion_groups_planned
fusion_groups_executed
fusion_roots
avg_roots_per_group
fusion_compile_ms
fusion_execute_ms
native_fusion_fallback_count
fallback_reason
```

对于 100k roots：

```text
planned >> executed
```

或 fallback 高，说明 fusion 只是“代码存在”，没有吞吐价值。

---

# 20. 100k root cap 本身需要压力测试

Adaptive scheduler 当前 root cap 约 100,000。

用户恰好准备运行约 100k。

不要直接在设计上限边缘生产运行。

测试：

```text
5k
10k
20k
50k
80k
100k
```

记录：

```text
parse_ms
compile_ms
planner_ms
graph_nodes
scheduler_queue_peak
RSS_peak
CSE_state_mb
result_buffer_mb
time_to_first_result
total_wall
```

如果 100k graph 本身成本高，采用：

```text
5k~20k roots/shard
```

但 shard 之前要先完成 shared base feature materialization，以减少跨 shard read duplication。

---

# 21. Result memory：100k × 5000 × 多年不允许一次常驻 RAM

必须 streaming sink。

禁止：

```text
dict[factor_name] = full_history_dataframe
```

在全部 roots 结束后一次写。

正确：

```text
root done
→ validate
→ encode
→ partition write
→ release root buffer
→ release CSE refcount
```

输出 Factor Lake partition：

```text
factor_id
date partition
universe/version
formula_hash
source_snapshot
engine_head
backend
```

---

# 22. DataAccess IO 优化清单

## 22.1 一定使用列裁剪

分钟 11 列，很多特征只需：

```text
Close, Volume, Amount
```

不能总读全部列。

建立 factor dependency → source projection。

## 22.2 Partition pruning

所有请求必须把日期下推到 Parquet partition。

禁止：

```text
scan all history → pandas filter
```

## 22.3 Arrow first

DataAccess 已提供 Arrow/Polars/DuckDB 路径。

分钟大量 IO 时优先：

```text
Arrow table / lazy scan
```

不要先构造巨大 Pandas。

## 22.4 Predicate pushdown

```text
Symbol
IndexSymbol
IndustrySource
date
```

尽可能 source 层过滤。

## 22.5 Read once

加入 read identity：

```text
dataset
snapshot/version
date range
symbol set
columns
filters
```

相同 read identity 在同一 batch/read-wave 复用。

## 22.6 Local SSD cache

远程 COS 场景：

```text
content-addressed partition cache
```

并发 worker 下载同一文件：

```text
single-flight / file lock
```

禁止重复下载。

---

# 23. DataAccess minute aggregation

优先使用/强化现有：

```text
aggregate_minute_to_daily
aggregate_minute_bundle
materialized daily minute feature
```

但必须验证它们最终走的是：

```text
Polars/Arrow/DuckDB batch path
```

不是内部循环每个 feature 重新 Pandas groupby。

建立 bundle benchmark：

```text
1 feature
10 features
100 features
500 features
1000 features
```

正确架构的 wall time 应远低于线性增长。

---

# 24. 分钟性能 Benchmark Protocol

A股典型一天约：

```text
~1.2M minute rows
```

至少跑以下规模。

## Dataset tiers

```text
T0: 1 day
T1: 5 days
T2: 20 days
T3: 60 days
T4: 250 days
```

代表股票：

```text
500
2000
all A-share
```

## Root tiers

```text
1
10
100
1000
```

再做 5k/10k factor campaign 的 materialized-day test。

## Backend

```text
pandas_numpy
polars_long
duckdb_sql
hybrid
```

如果某 backend 本来不适合该算子，不强行测“可用”，但要标明 reason。

## Cache

分别：

```text
cold cache
warm cache
```

## Workers

```text
1
2
4
8
```

## 指标

```text
wall_time
CPU_user/system
CPU_utilization
peak_RSS
read_bytes
remote_bytes
parquet_files
rows_scanned
rows/sec
features/sec
roots/sec
time_to_first_result
cache_hit
CSE_hit
fusion_hit
scalar_fallback
spill_bytes
write_bytes
```

---

# 25. 必须修复 perf benchmark 的可信度

所有 benchmark 必须：

1. assertion 确认实际走到目标 backend/path；
2. 禁用结果缓存或明确分冷/热缓存；
3. 记录 git HEAD；
4. 记录机器 CPU/RAM；
5. 记录 env；
6. 重复 >= 3 次，报告 median/p95；
7. correctness parity 先过，再比较速度；
8. benchmark 代码本身写 unit test。

任何 benchmark 如果无法证明：

```text
target path actually executed
```

一律视为无效。

---

# 26. PerfConfig 一个需要检查/修复的具体问题

`PerfConfig.from_env()` 读取多项：

```text
FACTOR_ENGINE_SCHEDULER
FACTOR_ENGINE_RESOURCE_PROFILE
FACTOR_ENGINE_COEXIST
FACTOR_ENGINE_NATIVE_FUSION
```

但当前 env cache signature 列表需要核对是否包含它们。

如果缺失，会出现：

```text
环境变量改了
但 PerfConfig cache 未失效
```

导致 benchmark/生产运行使用旧配置。

必须：

- 所有读取的 env key 都进入 cache key；
- 写测试逐项 mutate env；
- 确认配置 reload。

---

# 27. Polars hot-path 审计

自动 grep/AST 检查 FactorEngine production Polars 路径：

```text
.to_pandas(
.map_elements(
.map_groups(
.apply(
.iter_rows(
for row in
collect().to_
```

每一个命中分类：

```text
allowed boundary
performance bug
semantic fallback
research only
```

生产高频 hot path 禁止：

```text
Polars → Pandas → Polars
```

---

# 28. Pandas hot-path 审计

自动定位：

```text
groupby(...).apply
rolling.apply Python lambda
for symbol in
for date in
DataFrame.copy()
pivot/unstack repeated
sort_values repeated
concat in loop
```

对 direct-use high-frequency operator 改成：

```text
NumPy vectorized
Numba
Polars
DuckDB
```

但 Pandas reference 保留作 oracle。

---

# 29. DuckDB 审计

检查：

- 每个 factor 独立 connection？
- 每次重复 `read_parquet`？
- threads 失控？
- temp directory/spill？
- SQL CTE 是否复用？
- projection/predicate 下推？
- query compile overhead？
- 多 worker 是否抢同一 temp disk？

推荐：

```text
worker-local long-lived connection
explicit thread budget
explicit temp directory
shared file cache
```

避免 100k query one-by-one。

---

# 30. 算子覆盖：不要追求“1737 越多越好”

最终要产出：

```text
AgentDirect
AgentDirectHighCost
StateOnly
ConditionOnly
Intermediate
SourceTransform
RecipeOnly
ResearchOnly
Unsafe
Deprecated
```

Agent 生成 terminal factor 时默认只采：

```text
AgentDirect
AgentDirectHighCost（受 cost budget）
```

Intermediate 可以作为 expression child，但不能作为 terminal factor。

Condition/Event/GlobalState 应有明确 grammar。

---

# 31. Agent 搜索空间必须结构化

不要：

```text
all fields × all operators
```

应使用：

```text
PriceExpr
ReturnExpr
RatioExpr
VolumeExpr
AmountExpr
FinancialBalanceExpr
FinancialFlowYTDExpr
FinancialGrowthExpr
CategoryExpr
GroupExpr
BenchmarkExpr
IntradayExpr
RelationExpr
ConditionExpr
```

OperatorSignature 决定输入输出类型。

---

# 32. `typed_v2` 不能只是 tag

如果当前仍通过：

```text
"signature:..."
"domain:..."
"unit:..."
"cost:..."
```

字符串 tag 描述类型，必须迁移为真实结构：

```python
@dataclass(frozen=True)
class OperatorSignature:
    inputs: tuple[InputSpec, ...]
    output: OutputSpec
    parameter_domains: dict
    lookback_rule: ...
    frequency_rule: ...
    unit_rule: ...
    cardinality_rule: ...
    cost_rule: ...
```

Validator 必须真正拒绝：

```text
log(category)
ts_mean(text)
price + market_cap
minute + daily
flow_ytd + ordinary pct_change
one_to_many relation + rolling
```

---

# 33. Lookback / warmup 必须成为算子契约

不能长期在 Analyzer 大量维护：

```text
_FIXED_LAGS
_WINDOW_PARAM_NAMES
if canonical == ...
```

逐步迁移：

```text
OperatorSignature.lookback_rule
OperatorSignature.warmup_rule
```

组合表达式由 Analyzer 组合 child contracts。

必须自动检测：

```text
registry新增operator
但没有lookback rule
```

如果是 time-series operator：

```text
production => fail
```

---

# 34. 当前 operator catalog 中 `lookback=null` 要核查

不要假设 null 就是 0。

逐项区分：

```text
true zero-lookback
lookback computed from params
lookback metadata missing
stateful dynamic lookback
```

生成：

```text
lookback_audit.json
```

任何 TS direct-use 算子无法计算 lookback：

```text
Agent-visible = false
```

---

# 35. backend coverage 必须在当前 HEAD 重生成

当前旧 coverage 不能作为上线证据。

必须：

```text
current HEAD
current registry
current FieldRegistry
current backend code
```

重建：

```text
BACKEND_COVERAGE.md
backend_coverage.json
direct_use_matrix.json
```

旧 artifact：

```text
head mismatch => stale
```

不得参与生产 admission。

---

# 36. backend parity 容差不能统一一个数字

按 family：

```text
arithmetic: tight
rank: exact / equivalent tie semantics
rolling std/corr: numerical tolerance
regression: relative tolerance
spectral/fractal: family-specific
```

NaN mask 也必须一致。

不能只比较 non-NaN 数值。

---

# 37. Alias 审计

对全部 alias：

```text
alias formula == canonical formula
alias parameter mapping == canonical
alias backend path == canonical
```

查：

- alias collision；
- deprecated alias；
- alias 指向不存在；
- alias 参数顺序错；
- LQTP compatibility alias 与 canonical 语义漂移。

---

# 38. DSL round-trip

对全部 Agent-visible operator 自动生成合法 sample formula：

```text
serialize
→ parse
→ IR
→ serialize canonical
→ parse
→ execute
```

要求 structural hash 一致。

---

# 39. 数据源与 Operator correctness 联动

任何 source-backed operator 不能只测纯 Series kernel。

还要测：

```text
FieldRef
→ source resolve
→ time align
→ unit normalize
→ kernel
```

否则 kernel 正确、数据错，最后仍然白算。

---

# 40. Factor output DQ

每个 factor 完成后自动计算：

```text
coverage
NaN ratio
inf ratio
constant ratio
unique ratio
cross-sectional count
extreme magnitude
date coverage
symbol coverage
```

出现：

```text
>99.9% NaN
全常数
全inf
量纲爆炸
日期错位
```

立即 quarantine，不要继续进评估。

---

# 41. 性能 cost class

每个 operator 明确：

```text
O(1) elementwise
O(T) rolling
O(T log T) rank
O(N×T) cross-section
O(regression)
source-transform
intraday-scan
relation-aggregate
high-cost spectral
```

给 cost：

```text
1 / 2 / 3 / 5 / 8 / 13 / 21
```

Agent 搜索时限制 expression total cost。

否则 AlphaProbe 很容易生成大量数学上合法、计算上不可承受的 10 万因子。

---

# 42. Agent 侧加入“可计算性先验”

采样 operator 时权重不仅看 novelty：

```text
weight ∝
    production_confidence
    backend_speed
    field_availability
    1 / compute_cost
    novelty
```

高成本算子设置 campaign budget，例如：

```text
high_cost_root_ratio <= 5%
intraday_heavy_ratio <= 10%
relation_heavy_ratio <= 5%
```

数值按 benchmark 调整，而不是硬编码后永远不变。

---

# 43. 10 万因子建议分层物化

## L0 Raw canonical fields

归一化字段。

## L1 Shared base transforms

```text
returns
log returns
ranges
turnover decimal
market cap transforms
industry group
benchmark return
```

## L2 Expensive source features

```text
intraday→daily
financial PIT standardized quarterly/TTM
shareholder relation aggregate
index membership
```

## L3 Common subtrees

高频子树 materialization。

## L4 Final factor roots

按 partition streaming 写。

不要每个 final factor 从 raw COS 重走全部 pipeline。

---

# 44. Materialization 需要 lineage

每块记录：

```text
source snapshot
FieldRegistry hash
source contract hash
feature definition hash
engine source hash
timezone/session policy
PIT policy
unit policy
```

任何一项变化，缓存失效。

---

# 45. 多 worker cache single-flight

同一个 materialization key：

```text
只有一个 worker build
其他 wait/reuse
```

用：

```text
file lock
Redis lock
DB advisory lock
```

避免分钟特征被 N worker 同时重建。

---

# 46. 数据访问和计算 overlap

可以 pipeline：

```text
read wave k+1
    与
compute wave k
```

但必须受：

```text
memory budget
remote IO budget
```

控制。

不要无限 prefetch。

---

# 47. Instrument chunk 的语义风险

TS-only：

```text
可以按 symbol shard
```

CS/group operator：

```text
必须完整日截面
```

不能为了内存按 instrument chunk 后各自 cs_rank，再 concat。

Planner 必须知道：

```text
TS local
CS global
group global
benchmark broadcast
```

并设置 barrier/global stage。

---

# 48. Date chunk warmup

时间分块时：

```text
requested chunk start
← extend by exact lookback/warmup
→ compute
→ trim output
```

不能固定“多读 100 天”。

Financial report-period lookback 不能简单转换成 `80 * periods` 交易日，应该按事件/report period 预取。

---

# 49. Stateful chunking

Stateful operator：

- state carry 优先；
- 或 chunk overlap + deterministic trim；
- 必须 full/chunk parity。

---

# 50. 因子写出策略

Factor lake 建议：

```text
factor_id
formula_hash
date
symbol
value
engine_version
source_snapshot
```

列式 partition：

```text
date / factor_bucket
```

或按你们下游读取方式 benchmark 后决定。

重点：

- 避免 100k 小文件；
- batch write；
- compression；
- manifest；
- atomic commit。

---

# 51. 不要把单因子耗时当唯一性能指标

分钟频最重要的是：

```text
batch throughput
```

如果第一次 scan 300M rows 需要时间，但同一 scan 能算 500 个 features，则平均每 factor 成本很低。

最终 KPI：

```text
roots/sec
features per source scan
scan reuse ratio
CPU utilization
IO utilization
cost per 1000 factors
```

而不是单纯“一个 factor 多少秒”。

---

# 52. 生产 Benchmark 的目标形式

不要预设一个没有硬件依据的绝对秒数。

要求至少：

```text
eligible intraday vector kernel:
    vector path 显著快于 scalar；
    若 speedup < 2x，解释为何仍保留；
    对典型聚合优先目标 >=5x，但以真实硬件 benchmark 决定

minute bundle:
    feature count 10→100 不应接近10倍 wall；
    100→1000 应展示高复用

worker scaling:
    1→2→4 worker 有明确 throughput curve
    出现拐点后禁止继续加 worker
```

---

# 53. 资源自动调优

在启动一次 production campaign 前跑 calibration：

```text
CPU
RAM
local disk bandwidth
COS/network throughput
Parquet decode throughput
Polars thread scaling
DuckDB thread scaling
```

然后生成：

```json
{
  "fe_workers": ...,
  "duckdb_threads": ...,
  "polars_threads": ...,
  "remote_concurrency": ...,
  "read_wave_size": ...,
  "instrument_chunk_size": ...,
  "memory_budget": ...
}
```

不要用固定所有机器通用参数。

---

# 54. BLAS/OpenMP oversubscription audit

启动 worker 前打印：

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
POLARS_MAX_THREADS
DuckDB threads
FE worker count
```

计算：

```text
effective possible threads
```

超 CPU budget：

```text
launch warning/fail
```

---

# 55. DataAccess 资源治理必须跨进程闭环

当前进程级 GlobalResourceGovernor 不够。

短期：

```text
单 FE 主进程
```

或 worker 明确分 quota。

中期：

实现跨进程：

```text
HostResourceAuthority
```

统一：

```text
active query
scan bytes
remote concurrency
RAM reservation
DuckDB slots
cache build
```

---

# 56. 失败重试不能造成重复重扫

Adaptive scheduler OOM/retry 时：

- 已完成 read wave 复用；
- 已物化 base block 复用；
- 不应从 COS 重新扫全历史；
- 重试记录 retry reason；
- 降 shard size。

---

# 57. OOM 测试

专门压力测试：

```text
memory budget:
    8GB
    16GB
    32GB
```

观察：

```text
spill
replan
result release
CSE release
```

不能 OOM kill。

---

# 58. 生产可观测性必须增加

每个 campaign 输出：

```text
operator counts
backend distribution
source scan stats
CSE stats
fusion stats
vector stats
fallback stats
worker utilization
memory high-water
spill
write throughput
top 50 slow operators
top 50 slow source reads
```

生成：

```text
performance_report.html/json
```

---

# 59. Slow operator 自动 quarantine

如果 direct-use operator 在标准 benchmark：

```text
比同 family median 慢 > X 倍
```

先标记：

```text
DIRECT_ALPHA_HIGH_COST
```

或 temporary deny。

X 由 benchmark policy 配置，例如 5×/10×，不要写死到 kernel。

---

# 60. 增加 flame/profile pipeline

支持：

```text
py-spy
cProfile
scalene（如环境允许）
Polars explain
DuckDB EXPLAIN ANALYZE
```

至少能区分：

```text
IO
decode
join
pivot
groupby
Python
backend
write
```

---

# 61. CI 分层

## PR fast CI

```text
registry
syntax/import
unit
selected parity
property tests
field contract
DSL
```

## Nightly

```text
all operator parity
all prefix causality
backend matrix
minute equivalence
package install
medium benchmark
```

## Release / 100k launch

```text
full evidence rebuild
large benchmark
multiworker benchmark
100k dry run
```

---

# 62. 当前 HEAD 的 CI 不能简单当作全绿

执行时必须重新检查：

```bash
gh run list --commit $(git rev-parse HEAD)
```

只有 FactorEngine/DataAccess 相关 required checks：

```text
green
```

才能 release。

其他无关 Web build failure 可以分类，但不能因此跳过 FE 自己的 required CI。

---

# 63. Evidence 版本绑定

生产 evidence 必须：

```text
source tree hash
operator implementation hash
backend emitter hash
FieldRegistry hash
source contract hash
oracle hash
test vector hash
```

任一变化：

```text
evidence stale
```

禁止旧 evidence 自动继承。

---

# 64. `production_admitted` 不能永远为 0，也不能一次性全开

最终流程：

```text
audit
→ pass
→ promote operator subset
→ regenerate direct use
```

先形成一个高质量核心 Agent 集，例如：

```text
核心数学
核心TS
核心CS
核心group
核心financial
核心intraday
核心A股
```

再逐步扩大。

不要为了“1737 看起来很全”把 experimental 全标 production。

---

# 65. 建立 Agent Operator API

给 Agent 一个机器可读 endpoint/file：

```json
{
  "canonical": "ts_mean",
  "terminal": true,
  "production_admitted": true,
  "inputs": [...],
  "params": {...},
  "output_type": "...",
  "frequency": ["daily"],
  "cost": 2,
  "backends": ["polars_long", "duckdb", "pandas_oracle"],
  "examples": [...],
  "forbidden_fields": [...],
  "notes": ...
}
```

Agent 不应该自己读 Python registry 猜。

---

# 66. 参数采样也要安全

对窗口：

```text
allowed range
recommended range
cost effect
```

例如：

```text
window 2..252
```

不能 Agent 随机生成：

```text
window=100000
```

造成爆算力。

高频分钟窗口也设合法 session-aware domain。

---

# 67. 检查“算子覆盖很多但重复”

做 semantic fingerprint：

```text
canonical formula
operator DAG
invariance
parameter transform
```

识别：

- exact duplicate；
- alias disguised duplicate；
- monotonic transform duplicate；
- only rename；
- obsolete recipe。

重复算子不必全部暴露 Agent。

---

# 68. 算子命名清理

明确：

```text
proxy
raw
normalized
annualized
daily
intraday
```

例如：

```text
signed_volume_imbalance_proxy
intraday_amihud_proxy
realized_vol
realized_vol_annualized
limit_up_close
```

避免名称比数学语义更强。

---

# 69. A股特色算子覆盖检查

至少确认 Agent 有正确、已认证版本：

## Trading state

```text
limit_up_touch
limit_down_touch
limit_up_close
limit_down_close
one_word_limit_up/down
limit reopen
limit streak
days since limit
suspension
days since resume
listing age
ST/*ST
delisting
```

## Liquidity/capital

```text
free float ratio
free float turnover
circulating turnover
capital change
float adjusted Amihud
```

## Valuation

```text
earnings/book/sales/cashflow yield
historical valuation percentile/z
peer valuation
industry neutral
```

## Financial

```text
period-aware lag/yoy/qoq/ttm
quality
growth
accrual
cash conversion
profitability
leverage
turnover
```

## Shareholder

```text
top-k concentration
HHI
entropy
institution ratio
pledge/freeze
churn
```

## Benchmark

```text
excess return
beta
downside beta
residual return
residual momentum
idio vol
tracking error
```

## Intraday

```text
return path
volume distribution
realized moments
VWAP structure
limit behavior
```

---

# 70. 但“增加算子”排在正确性和性能后

优先级：

```text
P0 correctness
P0 source/PIT/unit
P0 batch performance
P0 multiworker safety
P1 backend parity
P1 direct Agent gate
P1 coverage gaps
P2 exotic operators
```

---

# 71. 10 万因子正式运行前必须做 dry-run ladder

## Stage A

```text
100 factors
```

核语义。

## Stage B

```text
1000
```

看 CSE/IO。

## Stage C

```text
5000
```

看 scheduler/RSS。

## Stage D

```text
20k
```

看 write/cache。

## Stage E

```text
50k
```

看 long-run stability。

## Stage F

```text
100k
```

正式。

每一级比较：

```text
output checksum
DQ
throughput
memory
failures
```

任何非线性恶化先查原因。

---

# 72. 10 万因子的生产 checkpoint

支持：

```text
campaign manifest
completed root IDs
failed root IDs
source snapshot
factor hash
partition completion
```

断点重启不得重算全部。

---

# 73. 失败分类

```text
INVALID_FORMULA
FIELD_CONTRACT
PIT_VIOLATION
BACKEND_PARITY
NUMERIC
DATA_MISSING
OOM
TIMEOUT
IO
WRITE
INTERNAL
```

不要全部捕获成 Exception + continue。

---

# 74. 100k factor结果必须可重现

记录：

```text
git HEAD
Python/package lock
FieldRegistry hash
operator registry hash
source snapshot
universe
date range
decision time
PIT policy
backend plan
seed
```

同一环境重复：

```text
sample checksum identical/tolerance equal
```

---

# 75. 不允许的“优化”

禁止以下方式伪造速度：

```text
减少min_periods
改NaN policy
偷偷fillna(0)
用近似但不改名
跳过PIT
减少股票
减少日期
改成float32而无误差评估
跳过backend parity
直接缓存旧错误结果
```

---

# 76. float32 优化规则

可以研究：

```text
minute raw volume/price intermediate float32
```

但必须：

- oracle float64；
- error bound；
- downstream rank stability；
- no overflow；
- financial金额慎用；
- 不允许全局一刀切。

---

# 77. 排序和 index 成本

检查所有 hot path 是否反复：

```text
sort_values(["date","symbol"])
set_index
reset_index
pivot
unstack
stack
```

一个 source block 尽量只完成一次 canonical ordering。

---

# 78. Symbol encoding

大量长表可以 benchmark：

```text
categorical
dictionary encoded Arrow
integer instrument id
```

减少内存和 groupby 成本。

但对外 lineage 保留原 Symbol。

---

# 79. 日期编码

分钟内部：

```text
trade_date_id
minute_slot 0..239
instrument_id
```

可极大简化 group/window。

建立标准 slot axis：

```text
0..119 morning
120..239 afternoon
```

但必须与真实 09:31/13:01 标签严格映射。

---

# 80. 分钟矩阵化

对于 dense 240 bar 特征，可 benchmark：

```text
[D, N, 240]
```

或 block:

```text
[B instruments, 240]
```

NumPy/Numba kernel。

与 Polars long 做对比，按 feature family 选。

不要先验认为某一种一定最快。

---

# 81. 分钟数据缺 bar 不能强行 dense 后补 0

dense slot matrix：

```text
missing mask
```

必须保留。

Volume missing 与 volume=0 语义不同。

---

# 82. 停牌

停牌日：

- 日频 `IsSuspend=True`；
- minute 可能无 bar。

任何 intraday feature：

```text
NaN
```

或明确 neutral policy，不能虚构 0 path。

---

# 83. Daily vs intraday output

明确两套 surface：

```text
minute_native
intraday_to_daily
```

不要混用频率。

Agent formula compiler 必须拒绝未经 resample/aggregate 的：

```text
minute_expr + daily_expr
```

---

# 84. SourceRef Polars support

当前 SourceRef 若仍存在 Polars long `NotImplementedError`，应按优先级补：

1. exact daily；
2. benchmark；
3. state；
4. financial；
5. relation；
6. intraday materialized。

否则多域因子会大量退 Pandas。

---

# 85. Fallback 必须显式

任何：

```text
Polars unsupported
→ Pandas
```

必须计数：

```text
fallback_count
fallback_operator
fallback_reason
rows
time
```

100k campaign 后能列出“导致 80% 时间的 fallback”。

---

# 86. Top Slow Report

每次 batch 自动生成：

```text
top slow operators
top slow DAG nodes
top slow source reads
top memory nodes
top repeated scans
top fallback
```

优化按真实占比，不靠猜。

---

# 87. 编译性能

100k expression 本身 parse/IR 可能贵。

优化：

- parse cache by formula hash；
- operator lookup cache；
- intern canonical strings；
- batch IR build；
- structural hashing；
- avoid deep recursion overhead；
- precompile common recipes。

建立：

```text
compile roots/sec
```

指标。

---

# 88. Formula length/depth guard

Agent 生成：

```text
max_nodes
max_depth
max_lookback
max_cost
```

避免极深 AST 导致：

- recursion；
- compile explosion；
- CSE graph explosion。

---

# 89. 控制不同算法的 campaign

AlphaProbe / AlphaMiner / CogAlpha 不应直接共享一个无限搜索 surface。

配置：

```text
daily_core
financial
intraday
cross_domain
experimental
```

分别预算。

---

# 90. AlphaProbe `evaluate_many(enable_cse=True)` 要验证真的调用 batch path

增加 integration test：

```text
mock/read counter
```

100 个共享底层字段因子：

```text
read count << 100
```

而不是只是参数名传递了 `enable_cse=True`。

---

# 91. DataAccess QueryBudget/deadline

100k campaign 不应给每个小 query 独立无限 deadline。

用 campaign root budget + child deadline：

- IO；
- computation；
- write；

避免 deadline context 泄漏。

近期 DA 已经修过相关 leak，继续加并发 test。

---

# 92. Cache key 正确性

不能只用：

```text
dataset+date
```

还要：

```text
columns
filters
unit policy
PIT policy
source snapshot
FieldRegistry
IndustrySource
IndexSymbol
```

否则缓存会返回不同语义结果。

---

# 93. Financial materialization

强烈建议先物化标准财务层：

```text
quarterly standalone
TTM
latest visible period
days since report
```

然后 10 万 formula 基于标准 layer 搜索，避免每个因子重复做 PIT join。

---

# 94. Shareholder materialization

同理：

```text
daily snapshot aggregate
```

预计算后再给 Agent。

---

# 95. Intraday materialization

如果大多数 10 万因子是日频输出：

> **不要让每个 factor run 触碰 raw minute COS。**

先把可复用 intraday primitives/features daily 化。

只有真正 `minute_native` campaign 才直接访问 raw minute。

---

# 96. 磁盘布局 benchmark

物化后的 daily feature blocks：

比较：

```text
wide Parquet
long factor-value
factor bucket
Zarr/Arrow IPC（如需要）
```

按下游读取模式选。

不预设。

---

# 97. Small file problem

100k 因子不能：

```text
100k × 每天一个小 parquet
```

控制 file size，批量写。

---

# 98. Exactly-once / atomic commit

写：

```text
tmp
→ validate
→ atomic rename/manifest commit
```

任务 crash 不得产生“看似完整”的半成品。

---

# 99. Launch Gate：算子正确性

正式 100k 前：

```text
Agent-visible terminal operator:
    100% registry valid
    100% math/oracle evidence
    100% prefix causality
    100% field contract
    100% required backend parity
    100% lookback/warmup contract
```

高成本可慢，但不能“不知道对不对”。

---

# 100. Launch Gate：分钟性能

必须：

```text
vector benchmark fixed
vector coverage report
scalar fallback report
minute bundle benchmark
cold/warm cache benchmark
1/2/4/8 worker scaling
```

当前“一个分钟因子10–20分钟”必须被 profiling 分解，不能只靠猜。

---

# 101. Launch Gate：多 worker

必须证明：

```text
host total CPU <= budget
host total RAM <= budget
DuckDB threads <= CPU
Polars threads controlled
remote concurrency controlled
cross-process cache single-flight
```

否则 production 默认单主进程。

---

# 102. Launch Gate：Backend

当前 HEAD 重新生成：

```text
backend coverage
parity evidence
runtime wiring
production admitted
```

旧 HEAD 报告作废。

---

# 103. Launch Gate：100k stress

必须通过：

```text
100k graph
or approved sharded equivalent
```

没有内存爆炸、scheduler starvation、无限 retry、writer backlog。

---

# 104. Launch Gate：Data correctness

A股 contract 必须全部专项测试：

```text
Return /10000
all % /100
PubDate PIT
ReportPeriodEndDate period-only
YTD financial
IndustrySource
IndexSymbol
ShareRatio
Weight
Minute UTC
HighLimit/LowLimit
UpdateTime not PIT
Dividend effective-only
```

---

# 105. 最终输出给用户的文件

完成整改后必须生成：

```text
FINAL_FACTOR_ENGINE_PRODUCTION_REPORT.md
AGENT_DIRECT_OPERATORS.json
AGENT_DIRECT_OPERATORS.md
OPERATOR_CORRECTNESS_MATRIX.parquet
BACKEND_COVERAGE_CURRENT_HEAD.md
INTRADAY_VECTOR_COVERAGE.md
PERFORMANCE_BENCHMARK.md
MULTIWORKER_SCALING.md
DATA_ACCESS_IO_REPORT.md
100K_DRY_RUN_REPORT.md
GO_NO_GO.md
```

---

# 106. `GO_NO_GO.md` 必须只有两种结论

## GO

只有所有 P0 gate 关闭。

列：

```text
HEAD
direct operator count
certified count
backend
benchmark hardware
recommended workers
recommended threads
100k expected execution mode
known P1/P2 limitations
```

## NO-GO

列出具体 blocker。

不能写：

```text
“基本可以”
“应该没问题”
“看起来能用”
```

---

# 107. Claude Code / AI Cloud Code 的循环执行方式

请进入持续 loop：

```text
while P0/P1 launch blockers exist:
    coordinator-fe
        → spawn independent audit subagents
        → consolidate findings
        → dispatcher split non-overlapping tasks
        → writer subagents implement
        → tester subagents independently test
        → benchmark guardian measure
        → reviewer inspect
        → integration test
        → regenerate evidence
        → update queue/status
```

不要因为：

- 单次 context 长；
- 某个测试慢；
- 某个算子多；

而提前停止。

可以分批，但必须更新 durable queue。

---

# 108. 修改纪律

1. 不允许 destructive git；
2. 不修改用户无关 working tree；
3. 每一改动先 reproduce；
4. 小批次 commit；
5. 测试失败不能删除测试；
6. 旧 API 需要兼容时加 shim；
7. 每个性能改动同时有 equivalence test；
8. 每个 correctness 改动有 regression；
9. evidence 必须重新生成；
10. reviewer 不能是主要代码 writer。

---

# 109. 优先级清单

## P0-A：100k 算值前必须先做

- 当前 HEAD operator direct-use full audit；
- stale backend coverage 重建；
- Agent allowlist 只暴露 certified terminal；
- A股 unit/PIT/source contract；
- minute benchmark bug；
- intraday vector binding coverage；
- run_many/CSE path；
- streaming sink；
- multi-process governor；
- worker oversubscription；
- backend fallback observability；
- 100k dry-run。

## P0-B：性能

- minute bundle；
- shared base materialization；
- Polars/DuckDB pushdown；
- scheduler thread/process conflict；
- CSE/fusion telemetry；
- read reuse；
- local cache；
- result memory release。

## P1

- 更多 direct operator promotion；
- SourceRef Polars；
- relation/financial materialization；
- minute_native；
- package/service hardening；
- more exotic ops。

## P2

- 新奇高成本数学算子；
- 超复杂 spectral/fractal；
- 未有数据支撑的 Level2 指标。

---

# 110. 当前审计最关键的 12 个待办

如果只能先做一批，按下面顺序：

1. **重新生成当前 HEAD 的 DirectUseMatrix / backend coverage / evidence。**
2. **把 Agent 可见集合改成 current-head production-admitted terminal operators，fail closed。**
3. **修 `perf_vec_bench.py`，确保 benchmark 真正触发 `__vec__`。**
4. **统计并扩大 intraday vector kernel 实际绑定覆盖；删除 silent bind failure。**
5. **把分钟特征改成一次 scan 的 `IntradayFeatureCompiler.compute_many()`。**
6. **确认所有 100k 计算走 `run_many(enable_cse=True)`，禁止逐因子 `run()`。**
7. **增加 CSE/fusion/read-wave/backend fallback telemetry。**
8. **A/B 修 scheduler 强制 thread 与 HybridExecutor classifier 冲突。**
9. **多 worker 默认改为单主进程内部并发；若多进程则实现共享资源配额。**
10. **把 A股 Return/%/PIT/minute/session/industry/index/relation contract 做成不可绕过的 production gate。**
11. **建立 100/1k/5k/20k/50k/100k dry-run ladder。**
12. **所有 P0 关闭后再开始正式落 10 万因子。**

---

# 111. 最后的业务原则

用户现在的目标不是继续“把 FactorEngine 做得看起来功能多”。

目标是：

> **把 FactorEngine 变成一个 Agent 可以不用猜、直接安全调用的生产因子计算系统。**

因此最优先的是：

```text
正确
> 可验证
> 可复现
> 可观测
> 批量吞吐
> 算子数量
```

10 万因子最危险的不是“跑慢一点”。

最危险的是：

```text
10万因子已经计算完成
然后才发现：
    Return单位错
    财务PIT错
    某个backend语义漂移
    intraday跨session
    关系数据shift错
    fallback改变结果
```

所以本轮必须同时完成：

```text
Operator correctness closure
+
Data semantic closure
+
Batch performance closure
+
Multiworker resource closure
```

完成后才输出 `GO`。
