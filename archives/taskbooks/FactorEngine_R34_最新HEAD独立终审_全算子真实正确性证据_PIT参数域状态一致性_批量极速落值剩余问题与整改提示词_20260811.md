# FactorEngine R34｜最新 HEAD 独立终审：全算子真实正确性证据、PIT/参数域/状态一致性、批量极速落值剩余问题与整改提示词

> **审计基线**
>
> - Repository: `18047533889/quant_projects`
> - Branch: `main`
> - Fresh audited HEAD: `8449d9c253c55308f3e406a15739e984387e9691`
> - Previous audit baseline: `499078505edff30b1268b60862dbcaab2f9ab31e`
> - 本轮确认：`8449d9...` 已包含用户此前要求的 R28/R30/R31/R32 大量整改。
> - **R33 尚未执行完毕，本文件原则上不重复 R33 已经提出的 Unified Query Graph、DataAccess source authority、真实 ReadWave、DuckDB-first cost-driven routing、FactorBlock、COW generation、multi-root、PreparedBatchReadSession 等架构整改。**
> - R34 的任务是：在“除了 R33 都已经改完”的前提下，重新审计最新 HEAD，找出仍然阻碍“所有算子真实正确 + PIT 无隐患 + 一批因子最快且准确落值”的剩余问题。

---

# 0. 执行整改 AI 的总目标

这次不要再把“测试很多”“1436 个 canonical 能执行”“hard gate 全绿”当作最终完成标准。

真正的完成标准是：

```text
A. Correctness
每一个保留的 production operator，在其公开允许的参数域、输入类型、市场、频率、
PIT/source contract 下，数学/统计/金融语义正确。

B. Causality
任何 production factor 都不能使用未来可见数据；
field availability、decision time、effective time、revision、universe、
fiscal period、session、snapshot、source revision 全部正确。

C. Backend equivalence
Pandas reference / Polars / DuckDB / specialized fast kernel，
只有在“该 canonical + 该参数域 + 该 execution variant”真正通过差分证据时，
才允许 production router 使用。

D. Batch correctness
run_many / fusion / CSE / read reuse / cache / multi-root / block compute
不能改变单因子参考结果。

E. Throughput
在 A-D 已认证成立的候选物理计划中，才选择 end-to-end 最快的执行路径，
最终优化指标是 TimeToDurableCommit。
```

最终原则：

> **FASTEST AMONG CERTIFIED-CORRECT PLANS**

---

# 1. 禁止继续出现的“假完成”方式

以下行为一律视为 R34 FAIL：

```text
1. hard gate 直接写 True；
2. “文件存在”就算功能完成；
3. grep / source contains 就算并发、事务、PIT、回滚验证；
4. surface=daily 就推导 pit_safe；
5. status=production 就推导 semantic correct；
6. default 参数测过，就开放整个参数域；
7. Pandas 跑通就推导 Polars / DuckDB 正确；
8. synthetic prefix test 就推导真实 DataAccess source PIT 正确；
9. expected exception 被 broad except 吞掉后把 canonical 记成 pass；
10. all-NaN 只要不报错就算通过；
11. 旧 SHA evidence 给最新 HEAD 背书；
12. 用 audit script 修改 metadata 后再测自己；
13. 通过放大 tolerance 掩盖 backend drift；
14. 为了速度牺牲 null/tie/min_periods/PIT/grain 语义；
15. R33 fast path 没做 reference equivalence 就直接 production。
```

---

# 2. 本轮最重要的结论

当前代码最大的问题已经不是“完全没治理”，而是：

> **治理框架和 evidence 很多，但部分 evidence 并没有独立地证明它声称证明的事情。**

当前可见的典型情况：

```text
1436/1436 execution coverage
≠
1436/1436 semantic correctness

latest_status=passed
≠
production certified

semantic_golden_verified=true in an old artifact
≠
current certifier can reproduce that proof

certified_parameter_domain=default
≠
all positive windows production-safe

needs_contract 非空
≠
source/PIT 已闭环
```

R34 因此把“证据真实性”放到比继续加 operator、更高的优先级。

---

# 3. P0 总览

当前必须首先闭合的 P0：

```text
P0-001  所有旧 evidence 与最新 HEAD 不一致，必须全量重新生成
P0-002  禁止 hardcoded True / presence-only hard gate
P0-003  R28 execution smoke 不能再叫“全算子正确”
P0-004  建立 Per-Canonical Correctness Ledger
P0-005  Daily / factor evidence authority 当前存在结构断层
P0-006  Semantic Golden 必须真正独立
P0-007  Golden 不能由被测 kernel 自己生成
P0-008  Parameter Domain Certification 存在 default-only overclaim
P0-009  参数边界/随机域测试不足
P0-010  production certification 必须 parameter-aware
P0-011  Typed Operator Signature 必须成为唯一输入/参数权威
P0-012  DataAccess dependency 必须由 typed signature 驱动
P0-013  Daily migration manifest 旧 seed 不是严格 version-bound
P0-014  AuthoringTier 与 Certification 彻底分离
P0-015  structural pit_safe 与 verified pit_safe 必须拆开
P0-016  source_contract_verified 必须来自真实 DataAccess contract
P0-017  Source/Universe audit 的 needs_contract 必须 release-blocking
P0-018  AvailabilityClock 的 open 语义错误/过度保守
P0-019  unknown market/frequency production fail-open
P0-020  bars_per_day 静态常数不能作为 production 时间真值
P0-021  Financial validator 硬编码 ASHARE_CONTEXT
P0-022  财报 period 不能只靠固定 periods_per_year=4
P0-023  财报 restatement/revision PIT golden
P0-024  holder/index/event/relation source-level PIT golden
P0-025  ModelTiming 缺失 contract 不能自动生成 production trust
P0-026  predictive/descriptive 不能靠 canonical 名字判断
P0-027  predictive model future perturbation
P0-028  multi-horizon label maturity
P0-029  Stateful 不能只靠手工列表分类
P0-030  Universal Chunk/Checkpoint parity
P0-031  Incremental append parity
P0-032  Historical correction parity
P0-033  mutation tests 必须真正 mutate production implementation
P0-034  static lookahead review 不能绑定旧 file:line
P0-035  Edge certification 不能默认绑 DuckDB
P0-036  EdgeContract 必须覆盖全部 production canonical
P0-037  shape-changing grain transform 的 gate 目前语义不统一
P0-038  多套 contract model 应收敛为单一事实源
P0-039  Production DataEvent partial-publish escape hatch 必须移除
P0-040  最新 HEAD 必须有真实 CI required-check 证据
```

---

# 4. R34-P0-001｜所有旧 evidence 必须在当前 HEAD 全量失效并重新生成

当前仓库里的多个验收产物绑定较早提交，而当前 main 是：

```text
8449d9c253c55308f3e406a15739e984387e9691
```

已确认至少存在：

```text
R28 final acceptance -> 更早 SHA
R30 HEAD/evidence     -> 更早 SHA
R31 final acceptance -> 更早 SHA
R32 HEAD/evidence     -> 更早 SHA
factor_operator_verified.json -> 2026-08-09 的旧 commit
```

而 `8449d9...` 又改动了会改变语义/执行行为的路径：

```text
api/
backend/
cleaned_operators/
planner/
runtime/
storage/
service/
```

这些恰好属于 factor evidence TCB。

## 整改

统一 evidence header：

```python
EvidenceHeader(
    commit_sha,
    dirty_tree_hash,
    operator_registry_hash,
    operator_surface_hash,
    operator_semantic_hash,
    field_catalog_hash,
    dataaccess_contract_hash,
    planner_hash,
    backend_hash,
    runtime_hash,
    test_source_hash,
    golden_source_hash,
    dependency_lock_hash,
)
```

任何一个不匹配：

```text
evidence invalid
production certification false
```

禁止旧 artifact “继续沿用”。

## Hard Gate

```text
R34_ALL_PRODUCTION_EVIDENCE_CURRENT_HEAD
R34_ZERO_STALE_R28_ARTIFACTS
R34_ZERO_STALE_R30_ARTIFACTS
R34_ZERO_STALE_R31_ARTIFACTS
R34_ZERO_STALE_R32_ARTIFACTS
R34_ZERO_STALE_FACTOR_OPERATOR_EVIDENCE
R34_ZERO_STALE_PRIMITIVE_EVIDENCE
```

---

# 5. R34-P0-002｜Hard Gate 禁止直接写 True

已确认当前 audit/generator 中仍存在：

```python
gates["..."] = True
```

或逻辑等价的：

```python
_gate(..., True, "covered elsewhere")
```

例如“每个 current canonical reviewed”等。

这不是真 hard gate。

## 统一 GateResult

```python
@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: Literal["PASS","FAIL","NOT_RUN","N/A"]
    executed_cases: int
    failed_cases: int
    evidence_files: tuple[str, ...]
    evidence_hashes: tuple[str, ...]
    commit_sha: str
    details: dict
```

只有：

```text
executed_cases > 0
+
artifact current-head bound
+
可重复计算
```

才能 PASS。

---

# 6. R34-P0-003｜R28 “全 canonical execute”只允许叫 smoke coverage

当前 `tests/operators/r28/test_all_canonicals_execute.py` 的实际能力主要是：

```text
能否找到 runtime
能否调用
输出 axes 是否合理
是否 deterministic
是否至少有部分 finite
```

但当前还有大量 family 被放入 contract-rejected 路径；
遇到某些 expected contract exception 后可以直接 return。

因此：

```text
1436 canonical executed
```

不能解释成：

```text
1436 formula verified correct
```

## 重新定义测试层级

```text
T0_REGISTRATION
T1_EXECUTION
T2_AXES_GRAIN
T3_SEMANTIC_GOLDEN
T4_TEMPORAL_PREFIX
T5_FUTURE_PERTURBATION
T6_SOURCE_PIT
T7_EDGE_DOMAIN
T8_PARAMETER_DOMAIN
T9_BACKEND_PARITY
T10_STATEFUL_INCREMENTAL_PARITY
T11_BATCH_PARITY
T12_OPTIMIZER_PARITY
```

每个 canonical 明确：

```text
PASS / FAIL / PENDING / N/A
```

---

# 7. R34-P0-004｜建立真正的 Per-Canonical Correctness Ledger

生成：

```text
docs/evidence/r34/R34_CANONICAL_CORRECTNESS_LEDGER.parquet
docs/evidence/r34/R34_CANONICAL_CORRECTNESS_LEDGER.csv
```

字段至少：

```text
canonical
surface
role
semantic_version
implementation_hash

runtime_exec
axes_grain
semantic_golden
prefix_causality
future_perturbation
source_pit
edge_nan
edge_inf
edge_zero
edge_domain
parameter_boundary
parameter_randomized
stateful_chunk
incremental_append
batch_parity
optimizer_parity

pandas_reference
polars_parity
duckdb_parity

certified_parameter_domain
certified_markets
certified_frequencies
certified_sources
certified_backends
production_verdict
remaining_risk
```

production verdict 必须是实际证据 AND，不允许由 status/surface推导。

---

# 8. R34-P0-005｜Daily / Factor Evidence Authority 当前存在结构断层

这是本轮确认的最关键代码级问题之一。

当前：

```text
Daily implementation evidence -> primitive_verified
non-Daily implementation evidence -> factor_operator_verified
```

但：

```text
semantic_golden_verified
temporal_prefix_verified
source_contract_verified
```

的 overlay 统一从 `factor_operator_verified` 查。

与此同时，最新 `certify_factor_operator_evidence.py` 又只给：

```text
factor_production_targets() - DAILY_CANONICALS
```

生成 record。

这意味着 fresh regenerate 后：

```text
Daily:
implementation = primitive evidence
semantic/source = factor evidence record missing
```

可能全部 fail closed。

## 重构

建立统一：

```python
OperatorEvidenceStore
```

key：

```text
canonical
semantic_version
implementation_hash
backend
parameter_domain_hash
source_contract_hash
```

gate：

```text
implementation
semantic
temporal
source_pit
edge
backend
stateful
```

Primitive / Factor evidence 只是测试 provider，不再是两套平行 authority。

---

# 9. R34-P0-006｜Semantic Golden 必须独立

最新 certifier 已经正确承认：

```text
synthetic runtime audit
只证明 execution / shape / determinism / prefix causality
不证明 math semantic golden
不证明真实 DataAccess source contract
```

下一步必须真的补 independent oracle。

## Golden 来源

### 数学原语

独立小实现：

```text
add/divide/rank/zscore/rolling mean/std/corr/cov
```

### 经典技术指标

```text
手算小样本
+
TA-Lib/独立 textbook recurrence（测试环境）
```

### 回归/统计/模型

```text
NumPy/SciPy/sklearn/statsmodels 等独立 oracle
```

### 自定义量化算子

至少：

```text
hand-calculable cases
metamorphic properties
economic invariants
```

不能只“当前函数重复计算两遍”。

---

# 10. R34-P0-007｜Golden Data 不允许由 Production Kernel 自己生成

禁止：

```python
expected = operator.calculate(x)
save(expected)
```

然后以后：

```python
assert operator.calculate(x) == expected
```

这只能证明代码没改。

每个 golden record 必须带：

```text
oracle_name
oracle_version
oracle_source_hash
formula_reference
fixture_hash
semantic_version
```

---

# 11. R34-P0-008｜Parameter Domain Certification 存在 default-only overclaim

这是本轮另一个最高优先级问题。

当前 primitive evidence 里很多 operator 明确记录：

```json
"certified_parameter_domain": {
  "bounds": ["default"]
}
```

最新 factor certifier默认同样：

```json
{"bounds":["default"]}
```

但 production signature 却允许：

```text
window=任意正整数
min_periods=1..window
ddof=0/1
lag=...
q=...
threshold=...
```

自动因子挖掘恰恰会大量探索非默认参数。

所以当前逻辑存在：

```text
tested(default)
→
allowed(broad domain)
```

## 必须改成

```python
CanonicalCertification(
    canonical,
    semantic_version,
    parameter_domain,
    backend,
)
```

生产调用：

```text
bound params ∈ certified domain
```

才允许。

---

# 12. R34-P0-009｜参数边界测试矩阵

### window

```text
1
2
minimum valid
default-1
default
default+1
5
20
60
120
252
large operational max
```

### lag

```text
0（如允许）
1
window-1
window
window+1
```

### q/quantile

```text
0
ε
0.01
0.5
0.99
1
outside
```

### ddof

```text
0
1
2 reject
```

### fast/slow

```text
fast < slow
fast == slow
fast > slow
```

### min_periods

```text
1
window
window+1 reject
```

### PCA/k/components

```text
1
rank
feature_dim
feature_dim+1 reject
```

---

# 13. R34-P0-010｜Parameter Property Testing

推荐 Hypothesis 或 deterministic generator：

```python
for seed in FIXED_SEED_CORPUS:
    params = sample_valid_domain(contract)
    oracle = reference(...)
    backend = candidate(...)
    assert_semantic_equivalence(oracle, backend)
```

每个 production canonical：

```text
boundary cases
random valid cases
invalid rejection cases
```

都记录 coverage。

---

# 14. R34-P0-011｜Typed Operator Signature 必须成为唯一权威

当前仍同时存在：

```text
panel_params
input_fields
signature introspection
_PANEL_NAMES
_SCALAR_THRESHOLD_NAMES
_WINDOW_NAMES
special-case maps
```

已确认部分 fallback 会根据参数名字判断 panel/scalar；
而例如：

```text
high
low
beta
sigma
```

这些名字本身根本不能证明 scalar/panel角色。

## 最终形式

```python
OperatorSignature(
    panels=(
        PanelParam("high", dtype=float, grain="daily"),
        PanelParam("low", dtype=float, grain="daily"),
        PanelParam("close", dtype=float, grain="daily"),
    ),
    scalars=(
        ScalarParam("window", dtype=int, min=1),
    ),
    relations=(...)
)
```

Production runtime：

```text
missing typed signature = FAIL
```

不得再 fallback name guessing。

---

# 15. R34-P0-012｜Typed Signature 直接驱动 DataAccess Dependency Manifest

DataAccess取数不能依靠：

```text
function introspection
parameter name guessing
synthetic fixture mapping
```

必须：

```text
OperatorSignature
→ Analyzer
→ PlanDependencyManifest
→ DataAccess DataRequest
```

这样同时解决：

```text
漏字段
多读字段
错 source
cache identity
batch union
ScanCost
```

这是 correctness + performance共同优化。

---

# 16. R34-P0-013｜Daily Migration Manifest 旧 seed 并不真正 version-bound

当前大量：

```python
review_id = "seed-2026-08-manual"
semantic_hash = ""
```

名字叫 reviewed manifest，但旧 seed 没有真实 semantic hash。

## 必须补：

```text
review_id
semantic_hash
implementation_hash
source_contract_hash
review_artifact_hash
test_case_ids
```

禁止空 hash 的 migrated daily operator进入最终 production authoring surface。

---

# 17. R34-P0-014｜Surface / Certification / Backend / ParamDomain 四维彻底分开

例如：

```text
ADX:
authoring_tier = daily
semantic_certification = pending
pandas_backend = certified
duckdb_backend = pending
param_domain = default-only
```

这是合法状态。

不允许：

```text
daily => production
production => all backends
canonical certified => all params
```

---

# 18. R34-P0-015｜Structural Causality 与 Production PIT 认证分开

当前 hardening/policy中仍存在：

```text
daily
elementwise
cs
group
causal tag
```

自动生成 structural pit-safe candidate 的机制。

工程 bootstrap可以保留，但必须改名：

```python
structurally_causal_candidate
```

真正：

```python
production_pit_safe =
    temporal_verified
    and source_pit_verified
```

生产 admission不得读 structural candidate 当最终证据。

---

# 19. R34-P0-016｜Source Contract 必须由真实 DataAccess 证明

synthetic panel prefix test不能证明：

```text
announcement_time
effective_time
provider binding
revision
snapshot
universe
```

每个 source-sensitive operator生成：

```python
SourceProof(
    canonical,
    logical_fields,
    dataset_ids,
    field_versions,
    provider_bindings,
    availability_times,
    effective_time_rules,
    decision_clock,
    universe_semantics,
    source_snapshot,
    source_contract_hash,
)
```

---

# 20. R34-P0-017｜`needs_contract` 必须真正阻断 release

当前 Source/Universe/PIT audit可以出现：

```text
needs_contract = [...]
violations = []
```

这不是 closure。

production target：

```text
needs_contract 非空
=> FAIL
```

除非明确：

```text
source_insensitive = true
```

并有理由。

---

# 21. R34-P0-018｜AvailabilityClock 的 `open` 语义需要修

当前 generic：

```text
open
```

和：

```text
high
low
close
volume
amount
vwap
```

被放在 session-end-known 集合。

这虽然不产生未来函数，但会错误阻止：

```text
open后即可计算的 same-session factor
```

## 正确做法

由 DataAccess FieldSpec声明 knowledge time：

```text
open -> opening auction / first completed bar
high/low -> evolving intraday, final at session end
close -> session end
volume/amount -> evolving, final at session end
```

operator availability：

```text
max(input knowledge times)
```

---

# 22. R34-P0-019｜Unknown Market/Frequency 必须 Production Fail-Closed

当前存在默认：

```text
market missing -> US
unknown market -> US
freq missing -> 1d
unknown freq -> 1d
```

这对 production不安全。

例如 minute source metadata丢失：

```text
1d fallback
→ warmup/history under-read
→ wrong output
```

## Production

```text
missing market -> error
missing frequency -> error
unsupported frequency -> error
missing session calendar -> error
```

Research可显式 fallback，但必须 warning。

---

# 23. R34-P0-020｜bars_per_day 静态常数不是 production truth

静态：

```text
A-share 240
US 390
```

无法表达：

```text
half-day
auction
special interruption
custom bars
calendar changes
```

真实 source-window planning必须来自：

```text
SessionCalendar + DataAccess trading calendar
```

---

# 24. R34-P0-021｜Financial Validator 硬编码 ASHARE_CONTEXT

当前 `check_financial_grain_contract()` 明确使用：

```text
ASHARE_CONTEXT
```

用户系统又要求未来 US。

## 必须改成

```python
check_financial_grain_contract(
    formula,
    market_context,
    data_snapshot,
)
```

市场来自 execution context。

---

# 25. R34-P0-022｜财报 Period 不能只靠固定每年 4 期

固定：

```text
periods_per_year = 4
```

无法正确覆盖：

```text
annual
semiannual
irregular fiscal year
changed fiscal year-end
IPO partial period
53-week year
```

需要：

```text
FiscalPeriodId
FiscalYear
FiscalQuarter
PeriodStart
PeriodEnd
ReportType
```

---

# 26. R34-P0-023｜财报 Restatement / Revision PIT Golden

构造：

```text
2024Q1 original at t1
2024Q1 restated at t2
2024Q2 report at t3
```

验证：

```text
decision_time < t2 -> original
decision_time >= t2 -> restated
```

并覆盖：

```text
TTM
QoQ
YoY
average balance
growth
revision factors
```

---

# 27. R34-P0-024｜Holder / Index / Event / Relation 各自做 Source-Level PIT Golden

## Holder

```text
snapshot_date
disclosure_date
holder identity
reordered top-N
entry/exit
revision
```

## Index

```text
announcement_date
effective_date
old/new weight
entry/exit
```

## Event

```text
announcement
effective
cancel/revise
```

## Relation

```text
edge valid_from
valid_to
revision
entity identity
```

---

# 28. R34-P0-025｜ModelTiming 缺失显式 contract 不得 production

当前 model timing对缺失 explicit contract会生成默认 contract。

这会让：

```text
contracts_missing
```

天然难暴露。

Production要求：

```text
model-like canonical
+ no reviewed explicit ModelTimingContract
=> FAIL
```

自动生成 contract只能：

```text
research hint
```

---

# 29. R34-P0-026｜Predictive / Descriptive 不得靠名字猜

不要再用：

```text
_forecast
_innovation
_next_
prior_
oos
```

推断最终语义。

每个 model显式：

```python
ModelTimingContract(
    label_definition,
    horizon,
    feature_cutoff,
    fit_cutoff,
    embargo,
    refit_schedule,
    prediction_time,
)
```

---

# 30. R34-P0-027｜真正的 Future Perturbation

固定：

```text
input[:t]
```

随机改变：

```text
input[t+1:]
labels[t+1:]
```

必须：

```text
output[:t] unchanged
```

并针对 label：

```text
label_t
label_{t-1}
label_{t-H}
```

测试 maturity边界。

---

# 31. R34-P0-028｜Multi-Horizon Label Maturity

至少：

```text
H = 1, 2, 5, 10, 20
```

验证：

```text
label for row s
only mature at s+H
```

---

# 32. R34-P0-029｜Stateful 分类不能只靠人工 canonical 集合

当前：

```text
FULL_HISTORY_REPLAY_CANONICALS
SEGMENTED_EXECUTION_CANONICALS
STATEFUL_CHECKPOINTS
```

仍大量手工维护。

新 operator如果漏登记，会被错误当 bounded history。

## 加行为检测

```text
full(x)
vs
run(x[:split]) + run(x[split:])
```

无 restore时不同：

```text
stateful = true
```

---

# 33. R34-P0-030｜Universal Chunk / Checkpoint Parity

每个 production TS operator：

```text
random split
multiple chunks
1-row chunk
missing gap
session boundary
```

比较：

```text
FULL == CHUNKED
```

stateful有 checkpoint时：

```text
FULL == CHECKPOINT_RESUME
```

---

# 34. R34-P0-031｜Incremental Append Parity

比较：

```text
full [1..T]
```

与：

```text
full [1..T-1]
+ append T
```

以及多日：

```text
append T..T+N
```

必须一致。

---

# 35. R34-P0-032｜Historical Correction Parity

DataAccess历史修正：

```text
t-30 source revision
```

触发依赖闭包增量重算后：

```text
incremental rebuild
==
full rebuild
```

R33 ChangeImpactDAG要以这个 correctness gate为前置。

---

# 36. R34-P0-033｜Mutation Testing 必须真正 Mutation Production Kernel

当前高风险 mutation test主要通过：

```text
本地 ref函数
本地 mutant函数
证明两者不同
```

这不能证明真实 production tests能杀掉真实 bug。

## 实际注入 mutation

```text
SHIFT_MINUS_ONE
SHIFT_PLUS_ONE
WINDOW_OFF_BY_ONE
MIN_PERIODS_ONE
DDOF_FLIP
NAN_TO_ZERO
INF_CLIP
BACKFILL
CURRENT_MISSING_BACKOFF
REVERSE_TIE
GROUP_INCLUDE_SELF
ASOF_FORWARD
ASOF_NEAREST
LABEL_MATURITY_MINUS_ONE
STATE_RESET
NO_SESSION_RESET
WRONG_PERIOD_OFFSET
FLOW_STOCK_SWAP
WRONG_ADJUSTMENT_BASIS
```

高风险 family设置 mutation score门槛。

---

# 37. R34-P0-034｜Static Lookahead Review 不能再用 file:line 当身份

当前 scanner review key类似：

```text
(file, line, hazard_pattern)
```

并且注释还绑定早期 HEAD。

代码加几行就会漂移。

## 改成

```text
file
symbol
AST fingerprint
source hash
hazard type
```

symbol source hash变化：

```text
review invalid
```

---

# 38. R34-P0-035｜Edge Certification 不得默认绑 DuckDB

当前 canonical semantic certification调用 edge gate时未显式传 backend；
而 edge gate默认：

```text
backend="duckdb"
```

这对 Pandas-first Extended operator逻辑不成立。

## 拆成两层

### Canonical edge semantics

由：

```text
independent oracle / Pandas reference
```

证明：

```text
NaN/Inf/zero/domain-invalid 应该怎么表现
```

### Backend edge parity

分别：

```text
Pandas
Polars
DuckDB
```

验证是否遵守 canonical edge semantics。

---

# 39. R34-P0-036｜EdgeContract 覆盖全部 Production Canonical

目前已经实现：

```text
undeclared => production fail-closed
```

这是对的。

下一步不要长期靠少数：

```text
NAN_REQUIRED
INF_REQUIRED
EDGE_IMMUNE
```

手工表。

每个 Typed OperatorSignature内置：

```python
EdgeContract(
    nan=...,
    pos_inf=...,
    neg_inf=...,
    zero=...,
    domain_invalid=...,
)
```

最终：

```text
production target undeclared edge contract = 0
```

---

# 40. R34-P0-037｜合法 Grain Transform 在多个 Gate 中处理不一致

当前一处 production admission已经允许：

```text
shape_preserving=False
但 input_grain != output_grain 明确声明
```

这是正确的。

但另外一些 hardening/shape check仍会：

```text
shape_preserving=False => error
```

runtime audit对 minute-source又有单独例外。

这三套规则不统一。

## 统一模型

```python
OutputShapeContract(
    input_grain,
    output_grain,
    preserves_index,
    preserves_columns,
    aggregation_keys,
)
```

允许：

```text
minute -> daily
event table -> entity-date panel
snapshot -> daily
```

禁止：

```text
silent axis drop
silent reindex
```

---

# 41. R34-P0-038｜多套 Contract 事实源必须收敛

当前有：

```text
OperatorPolicy
OperatorSpec
ProductionSignature
PandasFirstSignature
Catalog
Surface
DirectUse
BackendCapability
SemanticCertification
EdgeContract
ModelTimingContract
```

很多重复事实。

最终建议：

```text
CanonicalContract
├─ TypedSignature
├─ SemanticContract
├─ TemporalContract
├─ SourceContract
├─ EdgeContract
├─ GrainContract
├─ StateContract
└─ BackendCertifications
```

其他模块只能 derive，不再维护自己的 canonical hardcoded list。

---

# 42. R34-P0-039｜Production DataEvent Partial Publish Escape Hatch 必须删除

当前 production policy自己已经承认：

```text
逐 factor publish 不是 visibility transaction
f1 success, f2 fail
=> mixed published state
```

默认会拒绝，这是对的。

但仍有：

```text
DATA_EVENT_PRODUCTION_AUTO_PUBLISH=1
```

可启用实验性路径。

对于严格 production：

```text
这个 env bypass 必须不存在
```

统一：

```text
GenerationTransaction
stage all
validate all
commit once
```

---

# 43. R34-P0-040｜最新 HEAD 必须有真正 CI Required Checks

当前审计通过 GitHub connector没有拿到能为 `8449d9...` 背书的 combined status/workflow run证据。

不要写“CI一定没有”，而是：

```text
当前没有可验证的 latest-head CI proof
```

## 建议 Required Checks

```text
fe-unit
fe-all-operators
fe-semantic-golden
fe-parameter-domain
fe-pit-source
fe-model-timing
fe-stateful
fe-backend-parity
fe-batch-parity
fe-r33-integration
fe-dataaccess-integration
evidence-current-head
```

main protection强制全部通过。

---

# 44. R34-P1-041｜Coverage 必须从“有测试”升级成“需要的证据维度齐全”

当前历史 coverage 中大量 canonical 是：

```text
test_count = 1
execution_test_count = 1
semantic_test_count = 0
prefix_test_count = 0
future_perturbation_test_count = 0
parameter_test_count = 0
backend_parity_test_count = 0
source_pit_test_count = 0
chunk_test_count = 0
latest_status = passed
```

这只能说明 execution smoke pass。

release report必须按 RiskProfile计算每个 canonical所需证据矩阵。

---

# 45. R34-P1-042｜`PENDING` 与 `PASS` 不能混写

Dashboard必须分：

```text
execution_status
semantic_status
source_status
backend_status
production_certification
```

禁止一个：

```text
latest_status=passed
```

覆盖所有维度。

---

# 46. R34-P1-043｜所有 Production 参数必须由 ParamSpec 驱动

当前 pandas-first validator虽然优先读取 ParamSpec，但无声明时仍 fallback：

```text
_PANEL_NAMES
_WINDOW_NAMES
special-case sets
```

最终：

```text
production canonical missing complete ParamSpec
=> FAIL
```

legacy/research才允许 heuristic fallback。

---

# 47. R34-P1-044｜Relational Parameter Constraints 统一进入 Contract

不仅要单参数：

```text
window > 0
```

还要：

```text
fast < slow
short < medium < long
min_periods <= window
lag < window
k <= window
0 < acceleration <= maximum
components <= feature_dim
```

统一存：

```text
relational_specs
```

parser、runtime、backend emitter、evidence共用。

---

# 48. R34-P1-045｜Backend Certification 必须 Parameter-Aware

不要：

```text
DUCKDB_SQL_PARITY_VERIFIED contains "ts_mean"
```

就认为：

```text
ts_mean(any legal window)
```

都可以。

要：

```text
canonical = ts_mean
backend = duckdb
semantic_version = ...
certified_window_domain = ...
certified_min_periods_relation = ...
null_policy = ...
```

---

# 49. R34-P1-046｜Backend Certification 必须 Execution-Variant-Aware

同一个 canonical可能走：

```text
direct
fused
nested
multi-root
shared-window
hybrid
```

每个路径都可能单独出 bug。

认证单位至少：

```text
canonical
backend
parameter_domain
execution_variant
semantic_version
```

---

# 50. R34-P1-047｜Optimizer Differential Testing

R33会进一步加强：

```text
CSE
fusion
multi-root
pushdown
window coalescing
hybrid partition
```

所有优化必须：

```text
NO_OPT_REFERENCE
==
OPTIMIZED
```

每条 rewrite有：

```text
rewrite_id
semantic proof
randomized regression
```

---

# 51. R34-P1-048｜CSE Identity 必须包含所有语义维度

CSE key至少：

```text
operator semantic hash
bound params
input identities
source snapshot
market
frequency
adjustment basis
universe
decision clock
null policy
grain
session completeness
```

属性只差一个：

```text
不得共享
```

---

# 52. R34-P1-049｜Batch Order Invariance

同一 factors：

```text
[f1,f2,f3]
[f3,f1,f2]
[f2,f3,f1]
```

必须得到相同 per-factor结果和相同 semantic lineage。

用于抓：

```text
mutable cache contamination
state leak
shared buffer mutation
writer order bug
```

---

# 53. R34-P1-050｜Batch vs Independent Parity

对 heterogeneous batch：

```text
run_many(factors)
```

必须等于：

```text
{f: run_one(f) for f in factors}
```

相同：

```text
source snapshot
universe
market
params
decision time
```

---

# 54. R34-P1-051｜Duplicate Factor / Alias Isolation

测试：

```text
[f1, f1]
alias(f1)
same IR but different logical factor name
```

允许：

```text
CSE复用物理计算
```

但不允许：

```text
output identity
lineage
materialization path
```

串掉。

---

# 55. R34-P1-052｜Source Snapshot Isolation

同一 batch故意放：

```text
snapshot A
snapshot B
```

或：

```text
market A
market B
frequency A
frequency B
```

表达式相同也不能错误共享 source/CSE/cache。

---

# 56. R34-P1-053｜Cache Poisoning Tests

顺序：

```text
old snapshot
→ new snapshot
```

以及：

```text
A-share
→ US
```

必须证明：

```text
new result != stale cache hit
```

---

# 57. R34-P1-054｜Cross-Section Tie Semantics 全量固定

对：

```text
rank
pct_rank
quantile
top-k
winsorize
group_rank
```

显式定义：

```text
tie method
stable order
NaN handling
singleton group
all-equal group
minimum group size
```

三后端一致。

---

# 58. R34-P1-055｜Universe 是 Factor 数学输入

例如：

```text
rank(close)
```

在不同 universe结果不同。

所以：

```text
universe identity
```

必须进入：

```text
factor execution identity
CSE identity
cache
lineage
```

---

# 59. R34-P1-056｜Group Operator 的 Missing / Self-Exclusion 语义

覆盖：

```text
unknown group
null group
singleton
all same
group changes over time
exclude_self
```

尤其：

```text
peer mean
peer beta
industry neutralization
```

---

# 60. R34-P1-057｜Backend Parity 不只比较有限数值

必须比较：

```text
shape
index
columns
grain
dtype class
null mask
NaN mask
+Inf mask
-Inf mask
finite values
```

finite部分再：

```text
rtol / atol
```

---

# 61. R34-P1-058｜Rank-Like Factor 增加排序语义指标

额外：

```text
per-date rank correlation
top-k overlap
tie-group parity
sign parity
```

防止微小数值差导致组合完全不同。

---

# 62. R34-P1-059｜Regression / Model Design Matrix Contract

统一：

```text
intercept
standardization
missing filtering
weighting
rank deficiency
condition number
minimum obs
regularization
solver tolerance
```

不能各 backend使用各自默认。

---

# 63. R34-P1-060｜所有 Iterative Solver Non-Convergence Fail-Closed

对：

```text
ElasticNet
robust regression
iterative optimization
state estimation
root finding
```

统一：

```text
converged=False
=> NaN / explicit error
```

不能返回最后一次 iterate冒充正常结果。

---

# 64. R34-P1-061｜Recursive Technical Indicator Warm-Start Golden

重点：

```text
EMA
Wilder
RSI
ATR
ADX
MACD
KAMA
PSAR
Supertrend
```

验证：

```text
initial seed
first valid position
warmup
restart
missing segment
```

---

# 65. R34-P1-062｜Minute→Daily Operator 使用真实 Session Hostile Fixture

不仅 240完整bars。

必须覆盖：

```text
missing first bar
missing last bar
lunch gap
suspension
limit lock
duplicate minute
out-of-order minute
partial live session
early close
```

---

# 66. R34-P1-063｜Partial Session 与 Final Session Identity 分开

实时 10:30得到的 daily aggregate候选：

```text
partial
```

15:00：

```text
final
```

不能共享同一个 cache/materialization identity。

加入：

```text
session_completeness
source_high_water_mark
```

---

# 67. R34-P1-064｜Production Deployment 必须显式指定 Run Mode

库默认 research可以接受。

但 production service entrypoint：

```text
如果没有明确 production config
=> refuse to start
```

避免环境漏配导致 guard被关闭。

---

# 68. R34-P1-065｜Cost-Aware Routing 不应由可选 Env 决定

R33完成后：

```text
cost-aware AUTO routing
```

应该是默认生产行为。

可选：

```text
strict fastpath admission
```

但不应：

```text
不开 env就完全不做 cost planning
```

---

# 69. R34-P1-066｜Production Hot Path Python UDF = 0

验收：

```text
DataFrame.apply(axis=1)
Python map_elements
DuckDB Python UDF
per-row Python loop
```

在 native fast lane必须为 0。

特殊算子走：

```text
explicit specialized kernel lane
```

---

# 70. R34-P1-067｜编译 Typed ABI 减少 1000 因子重复 introspection

在：

```text
registry freeze
```

时生成 immutable：

```python
CompiledOperatorABI(
    typed_inputs,
    scalar_specs,
    relational_constraints,
    grains,
    units,
    backend_caps,
)
```

运行时不要反复：

```text
inspect.signature
catalog dict walking
name-role guessing
```

---

# 71. R34-P1-068｜Compile-Time Contract Cache

key：

```text
registry_version
semantic_hash
```

value：

```text
typed ABI
source deps
backend eligibility
validators
```

对 AlphaProbe等连续批次收益很大。

---

# 72. R34-P1-069｜Parameter-Sweep Multi-Window Kernel

自动挖掘中仍会出现：

```text
ts_mean(close,5)
ts_mean(close,10)
ts_mean(close,20)
ts_mean(close,60)
```

对：

```text
sum
mean
std
min/max
corr/cov
```

可构建 multi-window共享状态。

但必须先完成：

```text
parameter-domain certification
```

---

# 73. R34-P1-070｜技术指标 Family Multi-Output Kernel

天然共享：

```text
MACD line/signal/hist
PPO line/signal/hist
PVO line/signal/hist
DMI+/DMI-/DX/ADX
Keltner mid/upper/lower/position
```

内部一次算 shared recurrence，
对外保留 canonical identity。

Hard Gate：

```text
family kernel outputs == independent references
```

---

# 74. R34-P1-071｜Cross-Section Multi-Output Matrix Kernel

R33的 block compute应继续深入：

```text
rank
zscore
winsorize
group neutralization
multi-RHS regression
```

在同 date block执行。

R34要求：

```text
block kernel == per-factor reference
```

---

# 75. R34-P1-072｜Router 只能在“已认证参数域”中比速度

例：

```text
DuckDB ts_mean certified window<=252
Polars certified window<=4096
call window=1000
```

DuckDB不得因为 canonical status是 production-safe就参与。

接口：

```python
backend_eligible(
    canonical,
    bound_params,
    semantic_version,
    source_context,
    execution_variant,
)
```

---

# 76. R34-P1-073｜Runtime Shadow Validation

新 backend/新 optimizer上线初期：

```text
配置小比例同步 shadow
fast backend
vs
canonical reference
```

发现 mismatch：

```text
quarantine
```

---

# 77. R34-P1-074｜Backend Automatic Quarantine

维护：

```python
BackendHealth(
    canonical,
    backend,
    semantic_version,
    parameter_bucket,
    last_failure,
    parity_error,
    quarantined,
)
```

runtime shadow drift立即：

```text
production_safe -> quarantined
```

---

# 78. R34-P1-075｜Backend Promotion 经 Canary

生命周期：

```text
implemented
→ parity_verified
→ canary
→ production_safe
```

不要：

```text
test name存在
→ production_safe
```

---

# 79. R34-P1-076｜Materialized Factor Content-Level Checksum

partition metadata：

```text
row_count
date range
instrument cardinality
null ratio
finite ratio
content checksum
factor semantic id
source snapshot id
```

不只是“文件成功写出”。

---

# 80. R34-P1-077｜Read-After-Write Golden

publish前：

```text
write staging
→ close
→ reopen through DataAccess
→ compare
→ commit generation
```

---

# 81. R34-P1-078｜Batch Write Failure Injection

模拟：

```text
writer crash
disk full
network failure
N-th partition exception
manifest write fail
pointer flip fail
```

要求：

```text
old generation完整可见
new generation完全不可见
```

---

# 82. R34-P1-079｜DataEvent 与普通 Batch 使用同一 GenerationTransaction

不要维护：

```text
普通 factor transaction
+
DataEvent special publish
```

统一 visibility model。

---

# 83. R34-P1-080｜最终 Report 输出真实 Certification Counts

必须输出：

```text
total canonicals
daily / extended / research / unsafe / internal

execution passed
semantic certified
temporal certified
source PIT certified
parameter-domain certified
edge certified
production certified

pandas certified
polars certified
duckdb certified
```

---

# 84. R34-P1-081｜按业务 Family 输出 Coverage

至少：

```text
price-volume
technical
fundamental
valuation
shareholder
event
index
relation
cross-sectional
stateful
intraday
model/statistical
```

---

# 85. R34-P1-082｜Risk Level 决定 Test Depth

### LOW

```text
simple elementwise
```

### MEDIUM

```text
rolling stats
cross-section
group
```

### HIGH

```text
fundamental PIT
holder
event
index
stateful
model
minute→daily
advanced nonlinear/statistical
```

HIGH需要更多：

```text
golden
mutation
future perturb
source PIT
chunk
backend differential
```

---

# 86. R34-P1-083｜自动生成 RiskProfile

```python
RiskProfile(
    stateful,
    source_dependent,
    multi_panel,
    model_based,
    grain_changing,
    cross_sectional,
    parameter_complexity,
    numerical_instability,
    backend_count,
)
```

据此生成 required evidence matrix。

---

# 87. R34-P1-084｜PASS / FAIL / PENDING / N/A / NOT_RUN 分开

这五个状态必须明确。

特别：

```text
不需要测试
```

与：

```text
还没测试
```

不能相同。

---

# 88. R34-P1-085｜Production Tests 禁止 Broad Exception 吞错

审计：

```python
except Exception:
    pass
```

```python
except Exception:
    return
```

在 production certification test中默认违规。

Expected rejection必须：

```python
pytest.raises(ExpectedContractError)
```

---

# 89. R34-P1-086｜Expected Rejection 一等化

记录：

```text
case_status = EXPECTED_REJECTION_PASS
reason
exception_type
```

不能仅“没崩测试”。

---

# 90. R34-P1-087｜All-NaN 不再默认合法

只有：

```text
fixture不满足 min obs / domain
```

时 all-NaN才是 expected。

否则：

```text
all-NaN => FAIL
```

---

# 91. R34-P1-088｜Finite Coverage 按 Family 定阈值

例如：

```text
elementwise hostile fixture >95%
rolling after warmup >90%
rare event domain-specific
```

不要统一“有一个 finite即可”。

---

# 92. R34-P1-089｜Machine-Readable Semantic Spec

每个 canonical至少：

```text
formula/spec
input meaning
output meaning
units
grain
edge semantics
window semantics
```

测试/文档同源。

---

# 93. R34-P1-090｜Semantic Version 自动治理

如果：

```text
implementation change
导致 certified fixture output改变
```

必须：

```text
semantic version bump
```

否则 FactorId/cache/history会混语义。

---

# 94. R34-P1-091｜Backend-Only Change 失效该 Backend Evidence

语义不变：

```text
不必 bump logical semantic version
```

但：

```text
backend implementation hash change
=> backend certification stale
```

---

# 95. R34-P1-092｜Planner Rewrite Change 失效 Execution Variant Evidence

例如：

```text
multi-root fusion改动
```

要失效：

```text
fused execution evidence
```

即使单算子 kernel没变。

---

# 96. R34-P1-093｜DataAccess Version 进入 Source Certification

FE source proof绑定：

```text
DA semantic version
field registry hash
provider binding hash
PIT join implementation hash
snapshot implementation hash
```

---

# 97. R34-P1-094｜FE 主 CI 必须主动跑 DA Integration

不能：

```text
DA tests green
+
FE synthetic tests green
```

就推导 FE×DA correct。

新增：

```text
tests/integration/fe_dataaccess/
```

---

# 98. R34-P1-095｜Tiny Deterministic DataAccess Fixtures

用真实 parquet/registry建立：

```text
daily price
minute price
fundamental
index constituent
holder snapshot
events
```

真正走：

```text
DataAccess
→ FE
→ materialization
→ DataAccess readback
```

---

# 99. R34-P1-096｜Mixed-Source End-to-End Golden Batch

至少一个 50+ factor batch：

```text
daily price
fundamental
event
holder
minute-derived
cross-section/group
stateful
```

与 reference pipeline比较。

---

# 100. R34-P1-097｜真正的 Concurrency Determinism

不要只：

```text
同一 Pandas rolling算两遍
```

而要：

```text
workers=1
workers=2
workers=4
workers=N
DuckDB threads变化
```

结果一致。

---

# 101. R34-P1-098｜Nested Threading Test

组合：

```text
FE workers
DuckDB threads
Polars pool
BLAS/OpenMP
writer compression workers
```

验证：

```text
no deadlock
no oversubscription-induced wrong result
determinism
```

---

# 102. R34-P1-099｜Race Destructive Tests

并发：

```text
same factor
same partition
same cache key
same job state
```

最终只能看到一个正确一致状态。

---

# 103. R34-P1-100｜Failure 不得污染下一次 Run

第一次故意：

```text
planner/cache/writer failure
```

第二次 clean rerun：

```text
result == fresh-process reference
```

---

# 104. R34-P1-101｜Batch Execution Identity 必须完整

logical FactorId可以不包含 backend，
但 physical execution proof必须：

```text
FactorSemanticId
BoundParameters
SourceSnapshot
Market
Frequency
Universe
DecisionClock
SessionCompleteness
Backend
ExecutionVariant
```

---

# 105. R34-P1-102｜所有 Fallback 要有原因

记录：

```text
unsupported canonical
unsupported parameter domain
source-context mismatch
memory budget
backend quarantine
backend unhealthy
```

用于后续 native coverage优化。

---

# 106. R34-P1-103｜Native Coverage 按真实 Mining Usage 排序

统计：

```text
canonical usage_count
parameter distribution
average rows
runtime share
fallback rate
conversion penalty
```

优先优化：

```text
high usage × high runtime/fallback cost
```

而不是按 catalogue顺序。

---

# 107. R34-P1-104｜Eligibility First, Cost Second

最终 router：

```text
先 correctness eligibility
再 cost optimization
```

核心：

```python
eligible_physical_regions(
    canonical_contract,
    bound_params,
    source_context,
)
```

之后才：

```python
argmin(total_cost)
```

---

# 108. R34-P1-105｜Cost Router 记录 Prediction Error

每次真实 run：

```text
predicted_ms
actual_ms
absolute_error
percentage_error
```

按：

```text
backend
workload class
hardware
```

统计。

如果 MAPE持续高：

```text
calibration unhealthy
```

---

# 109. R34-P1-106｜性能 Benchmark 只统计 Correctness-Qualified Run

一个 run只有：

```text
semantic parity PASS
PIT PASS
readback PASS
```

之后其 TimeToDurableCommit才进入速度排名。

---

# 110. R34-P1-107｜Performance Mix 必须真实

不能只测 trivial factors。

至少：

```text
simple
rolling
cross-sectional
fundamental
intraday
stateful
mixed
```

---

# 111. R34-P1-108｜Performance Threshold 用相对回归

绑定：

```text
current hardware
current dependency versions
current dataset fixture
```

报告：

```text
vs baseline regression %
```

而不是随意固定 10000 factors/s。

---

# 112. R34-P1-109｜保护 Small-Batch Overhead

R33 AUTO执行模式完成后：

```text
20 cheap factors
```

AUTO不得显著慢于 best direct mode。

---

# 113. R34-P1-110｜Large Batch 证明 Scan Reuse 真发生

记录：

```text
factor count
unique logical data demands
physical scans
bytes scanned
```

1000 factors不能退化成1000次 scan。

具体物理优化由 R33执行。

---

# 114. R34-P1-111｜BatchExecutionProof

每个生产 batch持久化：

```python
BatchExecutionProof(
    factor_ids,
    query_graph_hash,
    source_snapshot,
    source_scans,
    backend_regions,
    conversion_edges,
    dq_result,
    materialization_generation,
    commit_id,
)
```

---

# 115. R34-P1-112｜高风险 Operator 可做上线 Shadow Recompute

对：

```text
new operator
recently changed operator
high-risk source dependent operator
```

前 N次：

```text
fast path
+
reference path
```

同步比较。

---

# 116. R34-P1-113｜Evidence Generator 必须 Read-Only

认证脚本不得在运行过程中：

```text
patch pit_safe
patch status
patch backend safe
```

然后再测自己。

测试输入 contract必须在运行前固定。

---

# 117. R34-P1-114｜Audit Oracle 与 Production Implementation 解耦

例如 rolling golden不能调用生产：

```text
_rolling_core
```

来计算 expected。

否则同一个 bug两边一起存在。

---

# 118. R34-P1-115｜Clean Checkout Certification

release evidence只允许：

```text
exact SHA
git status clean
fresh venv/container
dependency lock
```

artifact记录：

```text
container/image digest
dependency lock digest
```

---

# 119. R34-P1-116｜Runtime Dependency Versions 是 Numeric Evidence 的组成

至少：

```text
numpy
pandas
polars
duckdb
pyarrow
scipy
```

版本 family不兼容：

```text
numeric evidence stale
```

---

# 120. R34-P1-117｜Parameter Domain 分 Semantic Bounds / Operational Bounds

例如：

```text
window > 0
```

数学合法，
但：

```text
window=1e9
```

production不可接受。

分别：

```text
semantic_valid_domain
operational_certified_domain
```

---

# 121. R34-P1-118｜Mining Surface 直接暴露 Certified Operator Schema

AlphaProbe/FactorMiner等应读取：

```text
operator role
input types
units
grains
certified params
source requirements
native backend availability
cost class
```

减少生成非法或必失败表达式。

---

# 122. R34-P1-119｜Mining Scalar Params 类型化

如：

```text
window=20.3
window="20"
```

compile阶段直接 reject，
不要到 runtime crash。

---

# 123. R34-P1-120｜Typed Formula Canonicalization

如果 contract是 int：

```text
20
20.0
```

如双方都合法且语义等价，可规范成相同 IR。

```text
"20"
```

非法。

---

# 124. R34-P1-121｜Factor Dedup 使用 Typed IR Semantic Hash

不能只靠字符串。

需要消除：

```text
alias
whitespace
keyword order
```

造成的重复计算。

---

# 125. R34-P1-122｜Constant Folding 必须 IEEE/Null-Safe

例如：

```text
0 * NaN
```

不能只按代数规则变：

```text
0
```

optimizer rewrite必须尊重 canonical null/Inf语义。

---

# 126. R34-P1-123｜Where/Boolean 三值逻辑统一

Pandas / Polars / SQL对：

```text
condition NULL
NaN
invalid unselected branch
```

可能不同。

显式：

```text
ThreeValuedLogicContract
```

---

# 127. R34-P1-124｜Division-by-Zero Policy 固定

统一定义：

```text
divide
safe_div_null
protected_div
ratio
```

在：

```text
0/0
x/0
NaN/0
Inf/x
```

的行为。

---

# 128. R34-P1-125｜Quantile Interpolation 固定

Pandas / DuckDB / Polars默认 quantile method可能不同。

每个 quantile-like op：

```text
method/interpolation
```

必须成为 semantic contract。

---

# 129. R34-P1-126｜Variance/Std/Cov DDOF 固定

后端实现显式指定：

```text
ddof
```

不能依赖 backend默认。

---

# 130. R34-P1-127｜Rolling Endpoint 语义固定

明确：

```text
includes_current_bar
closed endpoint
lag placement
min_periods
```

---

# 131. R34-P1-128｜Time Window / Bar Window 严格区分

```text
20 bars
```

与：

```text
20 calendar days
```

必须有不同类型。

---

# 132. R34-P1-129｜Suspension 下 Rolling Window 语义

A股：

```text
20 actual observations
```

还是：

```text
20 exchange sessions including suspended date
```

必须 operator/window contract化。

---

# 133. R34-P1-130｜Price Adjustment Basis 是 Source Contract

DataAccess field明确：

```text
raw
forward-adjusted
back-adjusted
total-return
```

FactorEngine不能看到 `close` 就假设唯一含义。

---

# 134. R34-P1-131｜跨市场 Currency / Unit

未来 US/多市场：

```text
market cap
revenue
price
```

必须携带：

```text
currency
scale
unit
```

---

# 135. R34-P1-132｜单位转换必须显式

例如：

```text
万元 / 元
千股 / 股
```

禁止 silent implicit conversion。

---

# 136. R34-P1-133｜Synthetic Fixture 不能把所有输入都 Float 化

真实输入还有：

```text
bool
nullable int
category
string IDs
timestamps
```

Typed fixture必须覆盖。

---

# 137. R34-P1-134｜Holder / Entity ID 使用真实 ID Fixture

测试：

```text
string/integer identity
duplicate
missing
reordered slot
```

不能用 float signal代替 ID。

---

# 138. R34-P1-135｜Categorical Group Backend Parity

覆盖：

```text
null group
dictionary order
category encoding
```

---

# 139. R34-P1-136｜Timestamp / Timezone Parity

测试：

```text
UTC
Asia/Shanghai
America/New_York
DST
```

decision clock明确。

---

# 140. R34-P1-137｜A股/美股 Session Clock 分开认证

禁止把 A股 session assumption默认套给 US。

---

# 141. R34-P1-138｜Half-Day / Special Trading Day Fixture

US half day、特殊休市必须真正由 calendar生成。

---

# 142. R34-P1-139｜Live Incomplete Bar 不能当 Completed Bar

实时分钟数据必须：

```text
bar_complete
```

进入 source availability。

---

# 143. R34-P1-140｜Source Snapshot 包含 High-Water Mark

同 generation下持续 append新分钟：

```text
10:00 snapshot
11:00 snapshot
```

必须不同。

---

# 144. R34-P1-141｜Production Source Cache 包含 Decision Time

同交易日不同决策时点可见数据不同。

cache key：

```text
decision_time
source high-water mark
```

---

# 145. R34-P1-142｜Hostile Numeric Fixture

不能只：

```text
trend + sinusoid
```

新增：

```text
constant
alternating
step
spike
heavy-tail
all-zero
near-zero
NaN block
Inf
ties
outliers
regime switch
```

---

# 146. R34-P1-143｜每个 Family 专项 Hostile Fixture

技术：

```text
flat/gap/reversal
```

统计：

```text
collinearity/rank deficient
```

event：

```text
no event/simultaneous
```

fundamental：

```text
restatement/missing quarter
```

---

# 147. R34-P1-144｜Deterministic Seed Corpus

randomized property tests：

```text
固定多组 seeds
```

失败 seed进入永久 regression corpus。

---

# 148. R34-P1-145｜Fuzz Failure 最小化并固化

保存：

```text
minimal input
params
semantic hash
backend
```

---

# 149. R34-P1-146｜Fuzz 深度按 RiskProfile 分配

高风险更多 seeds，
控制 CI成本。

---

# 150. R34-P1-147｜PR / Nightly / Release 分层

### PR

```text
changed-impact tests
core golden
critical PIT
```

### Nightly

```text
all canonicals
param fuzz
all backends
chunk/incremental
FE×DA
performance
```

### Release

```text
fresh full evidence regeneration
```

---

# 151. R34-P1-148｜Change Impact 同时驱动 Evidence Re-test

底层 rolling core变化：

```text
自动找所有依赖 canonical
```

重新认证。

---

# 152. R34-P1-149｜Evidence Dependency DAG

```text
source file
→ kernel
→ canonical
→ execution variant
→ test
→ evidence
→ production certification
```

---

# 153. R34-P1-150｜Semantic Config 也进入 Hash

不仅 Python：

```text
field registry
market calendar
operator catalog
default params
```

都能改变行为。

---

# 154. R34-P1-151｜Evidence Generation 可复现

同 SHA + 同 env：

```text
除 timestamp外
artifact内容/hash稳定
```

---

# 155. R34-P1-152｜Audit Script 自己也必须有 Tests

当前已经看到：

```text
hardcoded true
presence-only gate
```

说明 audit code也是 critical code。

给 audit script写：

```text
negative controls
```

---

# 156. R34-P1-153｜Hard-Gate Mutation Test

故意：

```text
future shift
missing source contract
stale HEAD
partial publish
backend parity corruption
```

验证对应 release gate一定 FAIL。

---

# 157. R34-P1-154｜Evidence Producer / Validator / Release Evaluator 分层

不要一个脚本：

```text
生成
修改
验证
汇报
```

全部自己完成。

---

# 158. R34-P1-155｜Read-Only External Validator

提供：

```bash
python scripts/verify_release_evidence.py
```

它绝不能修改任何 production metadata/artifact。

---

# 159. R34-P1-156｜Final Report 绑定 CI 结果

写：

```text
commit
required checks
workflow/run IDs
test counts
artifact hashes
```

---

# 160. R34-P1-157｜没有 Blocker=0 不得写 All Fixed

最终：

```text
R34 P0 remaining = 0
R33 required gates = PASS
R34 required gates = PASS
```

才可：

```text
PRODUCTION READY
```

---

# 161. R34-P1-158｜R33 与 R34 不是两套平行工程

R33：

```text
FAST
```

R34：

```text
CORRECT / TRUSTWORTHY
```

最终必须：

```text
FAST AND CORRECT
```

---

# 162. R34-P1-159｜每个 R33 Performance Optimization 都挂 Reference Equivalence Gate

包括：

```text
DuckDB fusion
Polars fusion
ReadWave
CSE
window sharing
FactorBlock
COW writer
```

---

# 163. R34-P1-160｜保留 Canonical Reference

即使 production最终主要跑 DuckDB：

```text
reference implementation
```

仍必须存在。

它可以慢，但必须明显正确、可审计。

---

# 164. R34-P1-161｜Reference 不承担生产吞吐

分工：

```text
Reference = correctness oracle
Fast backend = production execution
```

---

# 165. R34-P1-162｜复杂 Operator 提供 Spec Implementation

例如：

```text
rolling regression
PCA
state machine
```

可以维护：

```text
slow spec implementation
fast production kernel
```

测试两者一致。

---

# 166. R34-P1-163｜Expression-Type Operator 尽量从 Semantic AST 生成多后端实现

对于简单组合型 operator：

```text
semantic AST
→ Pandas
→ Polars
→ SQL
```

减少三份手写逻辑漂移。

复杂 kernel例外。

---

# 167. R34-P1-164｜SQL Coverage 扩张优先自动可生成族

这样可以提高 DuckDB native coverage，
同时降低手写 emitter bug。

---

# 168. R34-P1-165｜SQL Emitter 明确 NULL Semantics

必须显式：

```sql
CASE
NULLIF
isfinite-like logic
```

不能假设 SQL NULL == Pandas NaN。

---

# 169. R34-P1-166｜DuckDB NaN / NULL / Inf 专项

所有 SQL production operator覆盖：

```text
NaN
NULL
+Inf
-Inf
```

---

# 170. R34-P1-167｜Polars NaN / Null 专项

覆盖：

```text
is_nan
is_null
fill
rolling
rank
```

---

# 171. R34-P1-168｜禁止 Silent Storage Dtype Downcast

```text
float64 -> float32
int64 -> int32
```

必须有：

```text
storage contract
numeric error bound
readback proof
```

---

# 172. R34-P1-169｜Materialization Dtype 按 Factor Contract

不要所有 factor强制 float。

例如：

```text
state
boolean
count
category
```

应保留合理类型。

---

# 173. R34-P1-170｜FactorBlock 必须保留 Validity Bitmap

不能为了矩阵速度：

```text
NaN -> 0
```

---

# 174. R34-P1-171｜Missing Semantics 可追溯

至少区分统计：

```text
warmup missing
source unavailable
outside universe
domain invalid
```

---

# 175. R34-P1-172｜Unknown Data 不得默认为 0

尤其：

```text
fundamental
event
holder
relation
```

unknown != zero。

---

# 176. R34-P1-173｜Event No-Event / Unknown-Event 区分

```text
known no event = 0
source unavailable = null
```

---

# 177. R34-P1-174｜Outside Universe / Suspended / Missing 分开

不能所有情况都只有一个无解释 NaN。

---

# 178. R34-P1-175｜Block DQ 保留 Per-Factor Stats

R33 FactorBlock可向量化 DQ，
但输出：

```text
null ratio
finite ratio
inf ratio
coverage
outlier stats
```

仍按 factor可见。

---

# 179. R34-P1-176｜DQ 不得二次全量 Scan

尽量复用：

```text
compute masks
Arrow stats
block reductions
```

---

# 180. R34-P1-177｜Materialization Generation 记录 Evidence Version

metadata包含：

```text
factor semantic version
operator evidence version
source snapshot
engine version
```

---

# 181. R34-P1-178｜Semantic Version 变化不得覆盖旧历史

必须：

```text
new logical factor version
```

或显式：

```text
migration + rebuild
```

---

# 182. R34-P1-179｜历史 Factor Store Semantic Migration Tooling

能回答：

```text
哪个 operator semantic update
影响哪些 materialized factors / partitions
```

---

# 183. R34-P1-180｜“所有算子正确”最终定义

只有满足：

```text
1. every production canonical typed signature
2. independent semantic golden
3. temporal proof
4. source-sensitive op has DA PIT proof
5. certified parameter domain
6. used backend certified for exact domain/variant
7. stateful full/chunk/incremental parity or explicit full replay
8. optimized batch == reference
9. readback == computed
10. all evidence current HEAD
11. CI required checks green
12. R33 performance gates green
```

才允许向外写：

```text
all production operators are certified
```

---

# 184. 建议新增测试目录

```text
factor_engine/tests/operators/r34/
    test_evidence_current_head.py
    test_no_hardcoded_pass_gates.py
    test_gate_negative_controls.py

    test_all_semantic_goldens.py
    test_parameter_domains.py
    test_future_perturbation.py
    test_source_pit_contracts.py
    test_availability_clock.py

    test_model_timing_explicit.py
    test_model_horizon_maturity.py

    test_stateful_chunk_parity.py
    test_checkpoint_restore_parity.py
    test_incremental_append_parity.py
    test_historical_correction_parity.py

    test_backend_parameter_parity.py
    test_backend_edges.py

    test_optimizer_differential.py
    test_batch_independent_parity.py
    test_batch_order_invariance.py
    test_cache_isolation.py

    test_fe_dataaccess_golden.py
    test_materialization_readback.py
    test_generation_atomic_failure.py
```

---

# 185. 建议新增 Evidence 目录

```text
factor_engine/docs/evidence/r34/
```

至少：

```text
R34_HEAD.json
R34_REQUIRED_GATES.json
R34_CANONICAL_CORRECTNESS_LEDGER.csv
R34_CANONICAL_CORRECTNESS_LEDGER.parquet

R34_PARAMETER_DOMAIN_COVERAGE.csv
R34_SEMANTIC_GOLDEN_COVERAGE.csv
R34_TEMPORAL_CAUSALITY_COVERAGE.csv
R34_SOURCE_PIT_COVERAGE.csv
R34_EDGE_COVERAGE.csv
R34_STATEFUL_PARITY.csv
R34_BACKEND_PARAMETER_PARITY.csv
R34_BATCH_REFERENCE_PARITY.csv
R34_OPTIMIZER_REFERENCE_PARITY.csv

R34_EVIDENCE_FRESHNESS.json
R34_AUDIT_NEGATIVE_CONTROL.json
R34_FE_DA_GOLDEN_RESULTS.json
R34_MATERIALIZATION_READBACK.json
R34_FINAL_ACCEPTANCE_REPORT.md
```

---

# 186. R34 Hard Gates

## Evidence Truth

```text
R34_CURRENT_HEAD_BOUND
R34_CLEAN_TREE_CERTIFICATION
R34_ALL_ARTIFACTS_CURRENT_HEAD
R34_ZERO_HARDCODED_TRUE_GATES
R34_ZERO_PRESENCE_ONLY_HARD_GATES
R34_ZERO_SOURCE_STRING_ONLY_HARD_GATES
R34_AUDIT_NEGATIVE_CONTROL_PASS
```

## Canonical Correctness

```text
R34_ALL_PRODUCTION_CANONICALS_TYPED_SIGNATURE
R34_ALL_PRODUCTION_CANONICALS_SEMANTIC_GOLDEN
R34_ALL_PRODUCTION_CANONICALS_TEMPORAL_PROOF
R34_ALL_SOURCE_DEPENDENT_CANONICALS_SOURCE_PIT_PROOF
R34_ALL_PRODUCTION_CANONICALS_EDGE_DECLARED
R34_ALL_PRODUCTION_CANONICALS_REQUIRED_EDGE_VERIFIED
```

## Parameter Domain

```text
R34_ZERO_DEFAULT_ONLY_DOMAIN_OVERCLAIM
R34_PRODUCTION_PARAM_DOMAIN_SUBSET_OF_CERTIFIED_DOMAIN
R34_ALL_RELATIONAL_PARAM_CONSTRAINTS_TESTED
R34_INVALID_PARAM_REJECTION_PASS
```

## Model

```text
R34_ZERO_GENERATED_MODEL_CONTRACT_PRODUCTION_ADMISSION
R34_ALL_PREDICTIVE_MODELS_EXPLICIT_TIMING
R34_ALL_PREDICTIVE_MODELS_FUTURE_PERTURBATION_PASS
R34_MULTI_HORIZON_LABEL_MATURITY_PASS
```

## Stateful

```text
R34_ALL_STATEFUL_BEHAVIOR_CLASSIFIED
R34_ALL_SEGMENTED_OPS_CHUNK_PARITY
R34_ALL_CHECKPOINT_OPS_RESTORE_PARITY
R34_ALL_INCREMENTAL_OPS_APPEND_PARITY
R34_HISTORICAL_CORRECTION_PARITY
```

## Backend

```text
R34_BACKEND_CERTIFICATION_PARAMETER_AWARE
R34_BACKEND_CERTIFICATION_EXECUTION_VARIANT_AWARE
R34_PANDAS_REFERENCE_GOLDEN_PASS
R34_POLARS_CERTIFIED_REGION_PARITY_PASS
R34_DUCKDB_CERTIFIED_REGION_PARITY_PASS
R34_BACKEND_NULL_NAN_INF_MASK_PARITY
```

## Batch

```text
R34_RUN_MANY_EQUALS_RUN_ONE
R34_BATCH_ORDER_INVARIANCE
R34_DUPLICATE_FACTOR_ISOLATION
R34_CSE_SOURCE_SNAPSHOT_ISOLATION
R34_CACHE_POISONING_ZERO
R34_WORKER_COUNT_NUMERIC_PARITY
```

## DataAccess

```text
R34_REAL_DA_SOURCE_PIT_GOLDEN
R34_FINANCIAL_RESTATEMENT_GOLDEN
R34_INDEX_ANNOUNCE_EFFECTIVE_GOLDEN
R34_HOLDER_SNAPSHOT_GOLDEN
R34_EVENT_EFFECTIVE_GOLDEN
R34_MINUTE_SESSION_GOLDEN
```

## Materialization

```text
R34_FACTORBLOCK_DQ_PASS
R34_READ_AFTER_WRITE_PARITY
R34_PARTIAL_GENERATION_VISIBILITY_ZERO
R34_DATAEVENT_ATOMIC_BATCH_PUBLISH
R34_WRITE_FAILURE_ROLLBACK_PASS
```

## R33 Integration

```text
R34_R33_FAST_PLAN_ONLY_AFTER_CERTIFICATION
R34_R33_OPTIMIZED_EQUALS_REFERENCE
R34_R33_TIME_TO_DURABLE_COMMIT_BENCH_CORRECTNESS_PASS
```

---

# 187. 实施顺序

## Phase 0｜冻结 Current Truth

```text
1. checkout 8449d9...
2. clean tree
3. 所有旧 evidence 标 stale
4. 旧报告禁止进入 release
```

## Phase 1｜先修 Evidence Framework

```text
5. GateResult
6. negative-control tests
7. current-head binding
8. 移除 hardcoded True
9. 移除 presence/string pseudo-gates
```

## Phase 2｜统一 Certification Authority

```text
10. OperatorEvidenceStore
11. Daily/Extended统一 per-gate evidence
12. primitive/factor artifact降级为 test provider
13. 修 fresh-regenerate Daily evidence断层
```

## Phase 3｜Typed Signature

```text
14. 全 production operator typed signature
15. 删除 production name heuristics
16. relational constraints
17. DataAccess dependency直接来自 typed signature
```

## Phase 4｜Parameter Domain

```text
18. boundary cases
19. randomized property
20. invalid rejection
21. production domain ⊆ certified domain
```

## Phase 5｜Semantic Golden

```text
22. elementwise
23. rolling
24. technical
25. cross-section
26. statistics/models
27. custom operators
```

## Phase 6｜PIT / Source

```text
28. availability
29. market/freq fail-closed
30. fundamental
31. holder
32. event
33. index
34. relation
35. minute
```

## Phase 7｜Stateful / Incremental

```text
36. behavior detection
37. chunk
38. checkpoint
39. append
40. historical correction
```

## Phase 8｜Backend

```text
41. Pandas reference
42. Polars parameter-domain parity
43. DuckDB parameter-domain parity
44. edges
45. execution variants
```

## Phase 9｜Batch Correctness

```text
46. run_many/run_one
47. order invariance
48. CSE/cache isolation
49. concurrency
50. optimizer differential
```

## Phase 10｜R33 Performance Architecture

完成 R33，并确保每个 fast region先过 R34 eligibility。

## Phase 11｜Materialization

```text
block DQ
readback
atomic generation
destructive failures
```

## Phase 12｜Release

```text
fresh environment
fresh evidence
current HEAD CI
R33 PASS
R34 PASS
```

---

# 188. 当前 HEAD 已确认、执行 AI 必须先复核的具体矛盾

```text
[ ] R28 final report SHA != current HEAD
[ ] R30 evidence SHA != current HEAD
[ ] R31 acceptance SHA != current HEAD
[ ] R32 evidence SHA != current HEAD
[ ] factor_operator_verified.json旧于当前 TCB

[ ] R28 generator有 hardcoded True gates
[ ] R30 every-current-canonical-reviewed有 hardcoded True
[ ] R31存在 hardcoded/presence pass
[ ] R32存在 source-string/presence pseudo-gates

[ ] R28 canonical coverage大量只有 execution=1，其余证据=0
[ ] current factor_operator_verified 与 current certifier语义不一致
[ ] fresh factor certifier只给 non-Daily records
[ ] overlay对 Daily semantic/source仍读 factor evidence

[ ] primitive certified parameter domain大量 default-only
[ ] factor certifier参数域默认 default-only
[ ] production signature却允许更宽 domain

[ ] needs_contract可非空但 violations为空
[ ] model timing missing contract自动生成
[ ] open被当 session-end-known
[ ] unknown market/freq fail-open
[ ] financial grain validator硬编码 ASHARE_CONTEXT
[ ] migration manifest seed semantic_hash为空
[ ] stateful coverage依赖 manual list
[ ] mutation tests未真实 mutate production kernel
[ ] static lookahead reviews依赖旧 HEAD line numbers
[ ] edge semantic certification默认 backend=duckdb
[ ] grain-transform gate规则互相不一致
[ ] production DataEvent仍有 non-transactional env escape hatch
```

---

# 189. 这轮不要重复制造同义模块

已经有基础的地方：

```text
ResourceBroker
FactorId
lineage
catalog
stateful contract
backend capability
operator spec
semantic certification
```

本轮优先：

```text
统一事实源
补真正证据
删除循环证明
```

而不是再建：

```text
OperatorSpecV2
PolicyV3
CertificationV4
```

造成更多漂移。

---

# 190. Definition of Done

R34完成后，系统必须能准确回答：

### Q1：这个 operator公式为什么是对的？

```text
→ independent semantic golden
```

### Q2：window=252也认证了吗？

```text
→ certified parameter domain
```

### Q3：这个基本面 factor有没有未来数据？

```text
→ DataAccess source/PIT proof
```

### Q4：DuckDB为什么有资格跑？

```text
→ exact canonical + params + execution variant backend parity
```

### Q5：1000 factors batch和一个个算相同吗？

```text
→ batch/reference parity
```

### Q6：增量与全量相同吗？

```text
→ incremental/full parity
```

### Q7：落盘读回来相同吗？

```text
→ read-after-write proof
```

### Q8：证据是当前代码的吗？

```text
→ evidence HEAD == runtime HEAD
```

### Q9：快路径有没有改变结果？

```text
→ optimized == reference
```

### Q10：最后到底快不快？

```text
→ correctness-qualified TimeToDurableCommit
```

---

# 191. Trust-First Fast Engine 最终结构

```text
Canonical Semantic Spec
        │
Typed Operator Signature
        │
 ┌──────┼────────┬───────────┐
 │      │        │           │
Math   Time    Source/PIT   Edge
Golden Proof    DA Proof   Contract
 │      │        │           │
 └──────┴────────┴─────┬─────┘
                       │
              Parameter Domain
                       │
             Canonical Certification
                       │
          ┌────────────┴────────────┐
          │                         │
   Backend Certification      Stateful Proof
   Pandas/Polars/DuckDB      Full/Chunk/Incremental
          │                         │
          └────────────┬────────────┘
                       │
                 Eligible Plans
                       │
                 R33 Cost Model
                       │
          Fastest Certified Execution
                       │
                FactorBlock + DQ
                       │
              Atomic Durable Commit
```

---

# 192. 最终交付要求

执行整改 AI 不得只交一个 markdown报告。

必须交：

```text
1. code changes
2. tests
3. R34 evidence directory
4. latest HEAD binding
5. canonical correctness ledger
6. parameter-domain coverage
7. source/PIT coverage
8. backend certification matrix
9. stateful/incremental matrix
10. batch/reference parity
11. optimizer/reference parity
12. FE×DA integration goldens
13. materialization readback/failure evidence
14. R33 integration status
15. current-head benchmark
16. final remaining-risk list
```

任何 P0仍未闭合：

```text
FINAL VERDICT = NOT PRODUCTION READY
```

---

# 193. 最终统一验收命令建议

```bash
python -m scripts.verify_operator_contracts
python -m scripts.verify_semantic_goldens
python -m scripts.verify_parameter_domains
python -m scripts.verify_temporal_causality
python -m scripts.verify_source_pit
python -m scripts.verify_stateful_parity
python -m scripts.verify_backend_parity
python -m scripts.verify_batch_parity
python -m scripts.verify_optimizer_parity
python -m scripts.verify_materialization
python -m scripts.verify_r33_performance
python -m scripts.verify_release_evidence
```

最后：

```bash
python -m scripts.factor_engine_release_gate
```

只有所有 required gate真实 PASS：

```text
exit 0
```

否则非零。

---

# 194. 第一批最高优先级执行清单

先做：

```text
1. stale evidence全部失效
2. hardcoded/presence pseudo-gates清理
3. Daily/factor evidence authority修复
4. typed signature全覆盖
5. parameter-domain overclaim修复
6. independent semantic goldens
7. real DataAccess PIT source proof
8. market/frequency fail-closed
9. financial cross-market context修复
10. explicit model timing
11. universal stateful chunk/incremental
12. real mutation testing
13. backend-aware edge certification
14. grain-transform统一 contract
15. production atomic publish
16. current HEAD CI
```

然后：

```text
17. backend parameter-domain parity
18. optimizer differential
19. run_many/run_one parity
20. cache/CSE/snapshot isolation
21. FE×DA mixed-source golden
22. read-after-write/failure injection
23. R33性能架构
```

---

# 195. 最终禁止事项

```text
DO NOT:
- 把 R34变成“再多一些 True gates”
- 没证据的 operator强推 production
- 为了 DuckDB方便改 canonical semantics
- 为了性能改变 NaN/tie/min_periods/PIT
- default-only测试后开放无限参数域
- skip高风险 family换绿 CI
- 用旧 evidence给最新 HEAD背书
- 把 research/generated timing contract当 production proof
```

---

# 196. 最终一句执行目标

> **把 FactorEngine 从“功能多、测试多、治理多”收敛成：每一个 production operator 的公式、参数域、PIT、source、state、backend execution variant 都有不可自证的独立证据；然后 R33 的统一物理执行层只在这些已认证正确的计划里选择 end-to-end 最快方案，并通过 DataAccess 原子、准确、尽可能快速地完成一批因子的计算与落值。**
