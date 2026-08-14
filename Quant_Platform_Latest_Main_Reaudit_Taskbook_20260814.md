# Quant Platform 最新 main 再审计与第二轮整改任务书
## Snapshot `57dba6684d5a0f2f92be2b1cf6aefc4efccaf7dd`
**Date:** 2026-08-14

> 这是最新 `main` 的增量审计。不要继续机械执行旧 `AUTONOMOUS_MASTER_QUEUE.md`。先把旧任务逐项重新标记为 `CLOSED_VERIFIED / PARTIALLY_FIXED / STILL_OPEN / SUPERSEDED / NEW_REGRESSION`。任何 FINAL/VERIFIED/PRODUCTION READY 报告都不能单独作为关闭依据，只认当前代码、可复现 case、regression test、独立 reviewer、integration/clean-wheel/runtime proof。

---

# 1. 本轮已经明显修好的内容

这些不应从零重写，只做独立回归证明：

- DataAccess I/O checker：violation 现在会 exit 1。
- scanner SyntaxError/内部失败：现在记录 `ScanError` 并 exit 2。
- FactorEngine 局部新增 `PhysicalIOAuthorityPolicy` 与 typed exemption。
- BackendKind / ExecutionKind / CapabilityLevel 已集中到 `backend/contracts`。
- Numba quantile 已取消 `fastmath=True`。
- FactorEngine modeling 新增 `EmbargoSpec`、`ApplicationWindow`、gap/purge 等结构。
- Evidence hash 与 source/emitter/semantic 校验已有加强。
- 大量 placeholder operator 已降级为 `research_only`，至少不再默认生产放行。

但“新增结构”不等于生产 contract 闭环。下面才是当前开放问题。

---

# 2. 当前最高优先级结论

当前必须优先处理：

1. q capability 会产生**假 production-ready**。
2. q 仍有两套手工 capability authority。
3. q compiler 仍有 unreachable/suspicious lowering。
4. q backend 仍自己把 whole logical plan 当单 region。
5. q executor 有明确 fan-in input 丢失、并发 workspace 冲突风险。
6. q NULL / symbol / nullable bool / integer null 语义未闭环。
7. Backend 主 capability registry 仍没真正纳入 q。
8. Polars `ts_zscore` 仍有已知 parity failure。
9. 16 个 Operator ABI violation 最新报告明确仍未修。
10. 678 个 operator 降 research_only 后，冷启动 strict-active 库尚未重新认证。
11. FactorEngine Modeling 与独立 `modeling` package 已形成 split-brain。
12. ModelArtifact OOS 防泄漏仍可绕过。
13. QuantEvaluator 默认 top-bottom spread 方向反了。
14. Quantile shape ABI / tie policy 没贯穿 evaluator identity。
15. FactorAssets 的 DataAccess adapter、Campaign、MultiFidelity 仍大量半成品。
16. 最新 gate 没统一进入 root-level CI。
17. Evidence 仍需防止“只改 SHA/报告”替代真实重新认证。

---

# 3. P0 — q capability 仍不能生产

## Q2-P0-001 `production_safe` 两套定义冲突
`q_capability_evidence.py` 中 `QCapabilityEvidence.production_safe` 正确定义为：

```text
declared_native
AND lowering_exists
AND compile_pass
AND runtime_pass
AND parity_pass
```

但 `get_q_production_safe_ops()` 实际只检查：

```text
declared_native AND lowering_exists
```

当前 compile/runtime/parity 又仍是 False/TODO。

**修：** production safe 集只能读取 `ev.production_safe`，禁止复制弱逻辑。

Gate：
```text
Q_PRODUCTION_SAFE_SINGLE_DEFINITION
```

## Q2-P0-002 `production_ready` 判定过弱
当前 report 基本：
```text
native_without_lowering == 0 → production_ready=True
```
这只能证明 declaration/lowering 对齐，不能证明 runtime/parity/PIT/null/time。

拆成：
```text
DECLARATION_CLOSED
LOWERING_CLOSED
COMPILE_CERTIFIED
RUNTIME_CERTIFIED
PARITY_CERTIFIED
NULL_TIME_CERTIFIED
PIT_CERTIFIED
PRODUCTION_CERTIFIED
```

## Q2-P0-003 `verify_q_capability_evidence.py` 会错误输出 “Q backend is PRODUCTION READY”
修掉错误 exit-0/文案。只有全部 mandatory gate PASS 才可输出 production-ready。

## Q2-P0-004 旧 `_PHASE1_NATIVE_OPS` 仍存在
`q_capability.py` 仍有完整 `_PHASE1_NATIVE_OPS`；新 evidence 文件又复制 `get_declared_native_ops()`。

现在实际是：
```text
旧 manual set
+ 新 manual set
+ evidence wrapper
```
不是 single authority。

建立唯一：
```text
QPhysicalImplementationRegistry
```
字段至少：
```text
canonical
lowering_id
parameter_domain_id
compile_evidence
runtime_evidence
parity_evidence
implementation_hash
q_version_range
pykx_version_range
```

## Q2-P0-005 QCompiler 仍消费旧 capability
`QCompiler.can_compile_operator()` 仍走旧 `get_q_capability().supports_native()`。

Gate：
```text
Q_COMPILER_ADMISSION_USES_EVIDENCE_AUTHORITY_ONLY
```

---

# 4. P0 — q compiler lowering

## Q2-P0-006 generic lambda 会截获 special lowering
`compile_operator()` 先处理 `q_func.startswith("{")`，后面才处理 `ts_beta/wma/clip/...`。这些 special branch 可能根本不可达。

改成：
```text
Q_LOWERING_REGISTRY[canonical] -> exactly one lowering function
```

Gate：
```text
Q_EXACTLY_ONE_REACHABLE_LOWERING_PER_CANONICAL
```

## Q2-P0-007 `ts_quantile/cs_quantile` declaration 与 map 不闭环
若 native=True 但 `_operator_map` 无 entry，会先 `q_func is None → raise`，后面的 quantile branch不可达。

Gate：
```text
Q_DECLARED_IMPL_REACHABLE
```

## Q2-P0-008 `ts_beta` 未证明是全序列 rolling beta
必须逐 t 验证：
\[
\beta_t=\frac{Cov(x_{t-W+1:t},y_{t-W+1:t})}{Var(y_{t-W+1:t})}
\]
不能只取最后 `window#x/window#y`。

## Q2-P0-009 `ts_corr/ts_cov`
真实 q runtime 与 Pandas oracle 验证：
```text
window=1/2/5
window>history
min_periods
NaN
Inf
constant
two instruments
missing sessions
```

## Q2-P0-010 `lag(periods>1)` 真实 q 语义认证
重点 periods=2/5、instrument boundary、null。

## Q2-P0-011 rolling median lowering 里的 `w` 必须真正绑定
当前 lambda 使用 `w`，不能依赖隐式自由变量。

## Q2-P0-012 q rank/tie 语义
冻结：
```text
tie policy
NULL order
NaN order
Inf order
percentile denominator
```

## Q2-P0-013 q round 对负数/半值
测试：
```text
-1.5 -0.5 0.5 1.5
```
与 canonical round 完全一致。

## Q2-P0-014 backend 不得自己拥有默认 window/span
`params.get("window",20)` 等必须来自 canonical ParamSpec。

Gate：
```text
Q_ZERO_BACKEND_OWNED_PARAM_DEFAULTS
```

---

# 5. P0 — q 没真正接 Physical Region

## Q2-P0-015 QBackend 仍接任意 PlanNode
代码仍明确写着 whole logical tree → one q region 的 simplified implementation。

生产必须：
```text
Canonical DAG
→ PhysicalBackendRegion
→ QCompiler
→ QExecutor
```

## Q2-P0-016 missing base data 返回 empty DataFrame
改 typed `DataUnavailableError`。

## Q2-P0-017 output schema 仍猜第一列/empty Series
建立 `QOutputContract`：
```text
timestamp
instrument
value
dtype
row_count
grain
sort order
duplicate key
nullability
```
schema mismatch hard fail。

## Q2-P0-018 runtime 任何异常都可能 fallback Pandas
不能把 q semantic/compiler/schema/PIT bug 变成成功的 Pandas run。

fallback 只能在 planning 前选择 certified alternative；运行中 contract/semantic failure hard fail。

## Q2-P0-019 q backend singleton 仍有 research→production 污染
第一次 `fallback=True, production=False` 创建后，后续 production 调用仍可能拿同一个实例。

Gate：
```text
Q_RESEARCH_PRODUCTION_INSTANCE_ISOLATION
```

---

# 6. P0 — q Executor 新确认 fan-in bug

## Q2-P0-020 `execute_batch_regions()` 会丢 next region 其他 inputs
当前若 previous output 被下一 region 使用，会：
```python
current_input = {previous_output: resident_handle}
```
从而丢掉原始 base inputs / group labels / second predecessor。

必须：
```python
current_input = dict(base_inputs)
current_input[previous_output] = resident_handle
```
并支持多个 predecessor resident handles。

必测：
```text
A ─┐
   ├→ C
B ─┘
```
以及 diamond DAG。

## Q2-P0-021 q workspace 全局变量名有并发覆盖
同一 global `kx.q` workspace 中写：
```text
input_table
result
node ids
```
两个 task 并发会互相覆盖。

建立：
```text
ExecutionNamespace(run_id, region_id)
```
或独立 session/process。

## Q2-P0-022 QExecutor 实际 q execution 没有 connection-level lock
availability lock 不等于执行 lock。必须明确 PyKX/q thread-safety contract。

## Q2-P0-023 resident handle stale check 只有 `id(q)`
增加：
```text
process_generation
session_generation
connection_generation
runtime_version
schema_hash
snapshot identity
PIT identity
```

## Q2-P1-024 resident table 无 liveness/refcount/delete
建立 producer/consumer count、last-use release、failure cleanup、session teardown。

## Q2-P1-025 resident metadata 失败不能 UNKNOWN→0
`len/nbytes` 失败若影响 resource governance，不得 success with rows=0/bytes=0。

---

# 7. P0 — q type/null/time

## Q2-P0-026 object NULL 被转 empty string
禁止：
```python
fillna("").astype(str)
```
必须区分：
```text
None / pd.NA / "" / empty symbol
```

## Q2-P0-027 short/int/long null sentinel 分开处理
当前基本只识别 int null。

## Q2-P0-028 nullable boolean 保留 True/False/NULL 三态

## Q2-P0-029 object dtype 不得默认全部 SYMBOL
改 schema-driven conversion。

## Q2-P0-030 zero-copy fallback 不能 catch 所有 Exception
只允许明确 zero-copy unavailable 时重试；dtype/semantic/timestamp/schema error hard fail。

## Q2-P0-031 timestamp/date/timespan/timezone round-trip
至少：
```text
Asia/Shanghai
America/New_York
UTC
DST transition
NaT
ns timestamp
date
timespan
```

---

# 8. P0 — q Process Governance

## Q2-P0-032 VERSION_MISMATCH enum 有但没真正检查
建立 q/PyKX supported version ranges。

## Q2-P0-033 availability 永久缓存
q 崩溃/重启后 `_checked=True` 可能继续返回旧 AVAILABLE。加入 health generation/reconnect invalidation。

## Q2-P0-034 license check broad exception
区分：
```text
LICENSE_MISSING
PROCESS_FAILED
TRANSPORT_ERROR
QUERY_ERROR
VERSION_MISMATCH
```

---

# 9. P0 — Backend capability 仍未统一

## BE2-P0-001 main `BackendName` 仍不含 q
`BackendKind` 有 `Q_KDB`，但 `operator_capability.BackendName` 仍只有 Pandas/Polars/DuckDB/ClickHouse。

## BE2-P0-002 `PhysicalImplementationSpec.backend` 仍是 str
改为 `BackendKind`，避免 `"q"/"q_kdb"/"duckdb"/"duckdb_sql"` 漂移。

## BE2-P0-003 `is_production_eligible()` 过度放行
现在基本“不是 unsupported/delegate 就 True”，`REFERENCE/SQL_PYTHON_UDF` 等都可能被放行。

重命名为：
```text
is_structurally_native_candidate()
```
真正 production admission 由 evidence authority 完成。

## BE2-P0-004 Polars `get_physical_spec()` broad except → None
必须区分：
```text
MISSING
INVALID
INFRASTRUCTURE_ERROR
```

## BE2-P0-005 `capability_quality()` 默认 research heuristic
拆：
```text
production_capability_quality()
research_capability_diagnostics()
```

## BE2-P0-006 capability_quality 对 pandas/sql 不能仅按 backend name 直接报 native
生产 report 必须读 registry + evidence。

## BE2-P0-007 `_emitter_source_hash()` 仍可 `"unknown"`
production evidence hash unavailable → typed hard failure。

## BE2-P0-008 SQL production signature import error 仍 silently pass
不能把 infrastructure failure 当“无 signature”。

## BE2-P0-009 `_sql_emitter_ok()` broad exception → False
区分：
```text
UNSUPPORTED / COMPILE_REJECTED / COMPILER_BUG / REGISTRY_FAILURE / INFRA_FAILURE
```

---

# 10. P0 — Polars 已知 parity failure

## POL2-P0-001 `ts_zscore` 仍有 `inf != nan` mismatch
当前仓库自己的 Polars parity final report 明确仍失败，所以：
```text
POLARS_PARITY_CERTIFIED = NO
```

必须记录单点：
```text
raw
rolling mean
rolling std
finite mask
zero std mask
min_periods
Pandas
Polars
DuckDB
```

## POL2-P0-002 不要用 `np.isclose(std,0)` 随便消 mismatch
先在 NumericSemantics 定义 zero threshold，再各 backend 实现。

## POL2-P1-003 扫整个 zscore family
```text
ts_zscore
cs_zscore
group_zscore
robust_zscore
```

---

# 11. P0 — Operator ABI：16 个明确 violation 最新仍未修

最新 `ABI_VIOLATIONS_FINAL_MANIFEST_2026-08-14.md` 明确：

```text
16 violations
Fixes: deferred
```

不要重新讨论是否存在，直接修。

## ABI2-P0-001 `return_decomp.py` 4 项最高优先级
涉及：
```text
overnight_return
open_close_return
open_to_vwap_return
vwap_to_close_return
```

问题：
```text
open vs open_px
missing price_basis
```

收益分解是核心算子，先修 metadata/kernel ABI，再做 adjusted/raw price policy parity。

## ABI2-P0-002 `ts_regression_slope`
phantom 参数：
```text
lag
retval
min_periods
add_intercept
```
以 canonical contract 为单一 authority，不能只是删 metadata 让 validator 绿。

## ABI2-P1-003 `ts_signature_mahalanobis_anomaly`
补/修 `depth` contract。

## ABI2-P1-004 12 个 `**kwargs` 空 ABI
主要：
```text
time_semantic_ops
panel_batch1
fiscal_event_ops
```
production-facing operator 应显式 signature；ParamSpec/文档/kernel 必须一致。

## ABI2-P0-005 ABI validator 进入 root CI
至少验证：
```text
param_names
param order
ParamSpec
kernel signature
defaults
types
active_when
```

---

# 12. P0 — 678 个 placeholder operator 降 research_only 后必须重认证

“标 research_only”是止血，不是实现完成。

## OP2-P0-001 重算 production operator surface
输出：
```text
before production count
after production count
demoted 678 family distribution
```

## OP2-P0-002 重算冷启动 strict-active 因子库
对每个 strict-active factor：
```text
factor DAG
→ transitive operator dependencies
→ current production admission
```
任何依赖新 `research_only` canonical：
```text
不得继续 strict-active
```

## OP2-P0-003 DirectUse/mining catalog 不能继续暴露 demoted op
检查：
```text
AlphaProbe
AlphaMiner
CogAlpha
cold-start generation
DirectUse manifest
```

## OP2-P0-004 research_only 与旧 production evidence 冲突 gate
```text
research_only=True
AND production_safe evidence
→ FAIL
```

## OP2-P1-005 为 678 个 operator 做 disposition map
```text
IMPLEMENT_LATER
REPLACE_WITH_X
MERGE_DUPLICATE
PERMANENT_RESEARCH
DELETE
```

---

# 13. P0 — placeholder audit script 自己仍有覆盖漏洞

## OP2-P0-006 SyntaxError → warning + return []
`audit_placeholder_operators.py` 仍：
```text
SyntaxError → WARNING → []
```
这是 fail-open。改 infrastructure failure / non-zero exit。

## OP2-P1-007 decorator detection 太窄
目前主要识别：
```python
@register_operator(...)
```
需要考虑 module-qualified/decorator aliases。

## OP2-P1-008 删除 `/home/shw/...` fallback
审计工具也必须 checkout-location independent / clean-wheel safe。

## OP2-P1-009 `if "test" in str(pyfile)` skip 太粗
改真实 path segment / configured test roots。

---

# 14. P0 — FactorEngine Modeling semantic identity

## FM2-P0-001 `_semantic_hash()` 仍 `default=str`
correctness identity 禁止 unknown object → `str()`。

统一使用严格 canonical identity encoder。

## FM2-P0-002 mapping key `str(k)` 会碰撞
例如：
```python
{1:"x"}
{"1":"x"}
```
不能同 identity。

## FM2-P0-003 `LabelContract.semantic_hash` 漏关键字段
至少补：
```text
origin_time
start_time_rule
end_time_rule
entry_price_basis
exit_price_basis
```
以及未来所有 semantic field。

Gate：
```text
LABEL_CONTRACT_ALL_SEMANTIC_FIELDS_HASHED
```

## FM2-P0-004 `EmbargoSpec.days` 与 `horizon_bars` 单位混用
不能：
```text
calendar days < trading bars
```
拆：
```text
embargo_bars
embargo_calendar_days + CalendarIdentity
```

---

# 15. P0 — Walk-forward 仍有时序 contract 问题

## FM2-P0-005 `gap_days` 文档与实现单位矛盾
文档说 full calendar days，代码却：
```python
val_start_pos = pos + gap_days
```
这是 date positions / trading bars。

同时 `check_fold_order()` 又用 timestamp `.days` 校验 calendar days。

拆字段：
```text
gap_bars
gap_calendar_days
```

## FM2-P0-006 validation_bars=0 时 gap 完全忽略
当前：
```text
test_start_pos = pos
```
所以 train→test 没 gap。

## FM2-P1-007 fold inadequate 后 advance cadence 不一致
有时 `pos += stride`，有时 `pos += step_bars`。统一 retrain cadence。

## FM2-P0-008 production purge 必须要求 ExchangeSessionCalendar
不能从 panel 内仅存日期猜 `horizon_bars`，因为停牌/缺失/session gap 会破坏 bar distance。

## FM2-P0-009 purge 没发生不能只 warning
若 contract 推导应有 overlap，但 zero rows purged：
```text
PurgeValidationError
```

## FM2-P0-010 `purge_policy="purge_bars"` 不应绕开 label interval purge
默认应：
```text
label_interval_purge + embargo
```
不是互斥二选一。

## FM2-P0-011 test-only fold audit `purge_before_boundary`
没有 validation 时也要避免 train label maturity 跨 test start。

---

# 16. P0 — ModelArtifact OOS 防泄漏仍可绕过

## FM2-P0-012 `FrozenPreprocessing.transform(application_window)` 仍是 placeholder `pass`
必须真正检查 observation dates/fit cutoff，或者删除这个假接口并只由更高层 enforce。

## FM2-P0-013 公共 `ModelArtifact.predict()` 无时间检查
调用者可直接：
```python
artifact.predict(X_oos)
```
绕过 `predict_oos/ApplicationWindow`。

生产 API 应只暴露：
```text
predict_asof / predict_oos
```
无时钟的 predict 仅 internal training/test helper。

## FM2-P0-014 `predict_oos(... dates=None)` 仍可跳过日期 proof
production `dates` 必须 required。

## FM2-P0-015 ApplicationWindow 没与 artifact cutoff/available_at 绑定
必须：
```text
window.start >= artifact_available_at
all dates >= activation_at
```

---

# 17. P0 — ModelArtifact identity/time

## FM2-P0-016 时间字段不能用字符串 lexical comparison
全部升级 typed timezone-aware `TradingTimestamp` 或统一 UTC Timestamp。

## FM2-P0-017 production identity 字段不能默认为 ""
例如：
```text
feature_schema_hash
data_source_hash
universe_hash
fit_code_commit
fit_code_component_hash
```
production artifact 缺任意关键 identity 应构造失败。

## FM2-P0-018 lineage 不能只存 `decision_clock_id/label_contract_id`
绑定：
```text
DecisionClock semantic hash
LabelContract semantic hash
```

## FM2-P0-019 lineage 补全
至少：
```text
DataSnapshotIdentity
UniverseSnapshotIdentity
SemanticCatalogIdentity
CalendarIdentity
FeatureSetArtifact
PreprocessArtifact
LabelContractHash
DecisionClockHash
CodeBuildIdentity
```

## FM2-P0-020 cache_key 不应再手写一个比 lineage 弱的 identity
建议：
```text
cache_key = lineage_hash + frozen_model_content_hash
```

## FM2-P0-021 `artifact_id` 与 lineage/content 建立 invariant
优先 content-addressed。

## FM2-P0-022 `artifact_available_at >= fit_completed_at`
现在还要加入这个顺序约束。

## FM2-P0-023 `refit_used_validation` consistency
若 True：
```text
final_fit_end
validation_end
selection_train_end
```
必须逻辑一致。

---

# 18. P0 — 两套 Modeling split-brain

仓库同时：
```text
factor_engine/modeling/*
modeling/modeling/*
```

最近 P0 主要修前者，独立 Modeling package 仍旧。

## MODEL2-P0-001 standalone SplitSpec.gap_days 仍未 enforce
## MODEL2-P0-002 standalone Modeling 没 label interval purge/embargo 一级 contract
## MODEL2-P0-003 anchored OOF 仍按 rolling 要求 train_start 增长
anchored 应固定 anchor start、扩展 train_end。
## MODEL2-P0-004 `get_fit_window()` 丢 universe_ref/data_snapshot_ref
## MODEL2-P0-005 ModelReadyData producer_version 仍硬编码且 provenance 太弱
## MODEL2-P0-006 立即做唯一 authority 决策

二选一：
```text
A. standalone Modeling 是唯一 authority，FE modeling 仅 adapter
B. FE modeling 是唯一 authority，standalone Modeling 迁移/下线
```

Gate：
```text
ONE_MODEL_CONTRACT_AUTHORITY
```

---

# 19. P0 — QuantEvaluator top-bottom spread 方向明确错误

## QE2-P0-001 默认方向反了
当前 quantile assignment：
```text
bin 0 = 最低 factor value
bin n-1 = 最高 factor value
```

但 `compute_top_bottom_spread()` 默认：
```python
top_q=0
bottom_q=-1
```

实际得到：
\[
LowReturn-HighReturn
\]

不是：
\[
HighReturn-LowReturn
\]

可能直接翻转：
```text
factor orientation
long-short return
Sharpe sign
admission/ranking
```

统一 orientation：
```text
LOWEST_BIN=0
HIGHEST_BIN=n_quantiles-1
```
默认：
```python
top_q=-1
bottom_q=0
```

并检查历史报告/缓存是否受影响。

---

# 20. P0 — Quantile shape/tie ABI

## QE2-P0-002 `assign_quantiles_batch()` 仍 `.squeeze()`
内部 ABI 应固定：
```text
(T,N,F)
```
不要因 F/T/N=1 随机降维。

## QE2-P0-003 Numba 同样 `.squeeze()`

## QE2-P0-004 quantile return public API 没暴露 tie method
`compute_quantile_returns/optimized/fast` 都要把：
```text
QuantileTiePolicy
```
贯穿：
```text
MetricSpec
compute
cache/evaluation identity
report
```

## QE2-P0-005 AVERAGE/FIRST enum 暴露但不可实现
要么实现，要么 production enum 只保留 MIN/MAX。

## QE2-P0-006 NumPy/Numba percentile interpolation exact property tests
至少 N=2...5000、ties、odd/even、float32/64。

## QE2-P0-007 显式冻结 percentile method
例如：
```text
method="linear"
```
并进入 metric semantic identity。

## QE2-P0-008 参数 validation
```text
n_quantiles>=2
min_assets>=1
valid ndim
labels aligned
finite asset rules
```

## QE2-P1-009 heavy ties 空 bin policy
定义：
```text
allow empty / insufficient / rank-based equal-mass / merge?
```
不能默默继续。

---

# 21. P0 — FactorAssets DataAccess adapter 仍是半成品

## FA2-P0-001 正式 import contract clean-wheel 验证
当前 adapter 使用：
```python
from dataaccess import ...
```
而 DataAccess 正式 package 架构是 `data_access`。不要靠 monorepo path 偶然成功。

## FA2-P0-002 不要 catch TypeError 当 dependency missing
只 catch 真正 import/dependency exception。

## FA2-P0-003 availability 拿到 handle 不能直接 True
至少：
```text
execute/materialize
result status
snapshot
PIT/asof
factor identity
```

## FA2-P0-004 broad except → False/None 拆 typed errors
```text
NOT_FOUND
UNAVAILABLE_ASOF
INFRASTRUCTURE_ERROR
PERMISSION
PIT_REJECTED
SCHEMA_ERROR
```

## FA2-P0-005 catalog entry 不能 fabricated
当前“read 成功→basic dict”不是 catalog。

## FA2-P0-006 `list_available_factors()` hardcoded `()`
明确 placeholder，必须接正式 catalog API。

## FA2-P1-007 `date.today()` 不是 market/asof authority
改 explicit asof + market calendar。

## FA2-P0-008 `factor_assets[adapters]` dependency 不完整
目前只声明 QE。建议：
```text
adapters-da
adapters-fe
adapters-qe
full
```

---

# 22. P0 — CampaignCoordinator

## FA2-P0-009 `max_duration_s` 不会自动推进
`elapsed_s` 没绑定 monotonic clock。

## FA2-P0-010 budget 无 atomic reservation
并发 subagents 会超预算。
增加：
```text
reserve_evaluation
commit_evaluation
release_evaluation
```

## FA2-P0-011 objective 永远 higher-is-better
增加：
```text
ObjectiveSpec(direction=max|min)
```

## FA2-P0-012 NaN/Inf metric fail-closed

## FA2-P0-013 complete/cancel 完整 state-transition enforcement

## FA2-P1-014 campaign persistence/checkpoint/resume

---

# 23. P0 — MultiFidelity

## FA2-P0-015 `confidence_threshold` 未使用
## FA2-P0-016 `max_candidates_per_tier` 未 enforce
## FA2-P0-017 NaN metric 可绕过 threshold
## FA2-P0-018 NaN stability/survival 同理
## FA2-P0-019 买不起 L0 仍返回 L0
应返回 `NO_AFFORDABLE_TIER`。
## FA2-P0-020 sample/universe fraction 无 deterministic PIT-safe sampling
禁止 random row sampling。

---

# 24. Carry-forward：FactorAssets Plateau / Pareto / Ledger

本轮没有看到实质闭环，继续 OPEN：

- `no neighbor evidence` 不能变 perfect robustness。
- perturbation 必须服从 ParamSpec/type/bounds/active_when。
- Pareto 不能默认所有 objective maximize。
- missing/NaN objective 不能静默 -Inf。
- Pareto distance 必须真正 normalization。
- test split seal 必须真正 enforcement。
- test 参与任何 adaptive search 必须永久 contamination。
- FactorAssets 不应自行重算 FE canonical identity；应消费 FE 签发 artifact。

---

# 25. P0 — Evidence / Certification

## EV2-P0-001 `current_commit_sha()` 依赖 runtime git checkout
clean wheel / container / production image 可能没有 `.git`。

构建时冻结：
```text
ExecutionBuildIdentity
SCM revision
wheel hash
```

## EV2-P0-002 commit ancestor + 部分 source hash 不足以覆盖 implementation closure
operator 依赖：
```text
numeric_semantics
helpers
bridges
shared kernels
registry
type adapters
```
这些改了，operator 自身 source file 不一定改。

建立：
```text
ImplementationClosureHash
```
覆盖 transitive implementation dependency。

## EV2-P0-003 不能编辑 stale evidence SHA 代替重新 certification
规范：
```text
execute certification suite
→ capture cases/results
→ regenerate artifact
```
禁止：
```text
old results + new SHA binding = new certification
```

## EV2-P1-004 runtime library identity
只比较 major/minor 是否足够需要明确 contract；优先绑定 environment/lock artifact hash。

## EV2-P0-005 required backend 版本 `missing/unknown`
production evidence 直接 invalid。

## EV2-P0-006 evidence 必须绑定 parameter-domain identity
一个 operator implementation 不代表所有 parameter combinations 都认证。

## EV2-P0-007 q evidence 单独建立真实 artifact
不能复用 declaration report 充当 runtime evidence。

---

# 26. P0 — 报告状态与真实代码矛盾

仓库报告仍有类似：
```text
FactorAssets Ready for Production
Modeling Ready for Production
QE Core Ready for Production
All P0 correctness resolved
```

但当前代码仍有本任务书明确 P0。

## META2-P0-001
最终状态报告改为 machine-generated / gate-backed：

```text
CERTIFIED
PARTIALLY_VERIFIED
OPEN
BLOCKED
UNKNOWN
```

禁止人工“Mission Accomplished”覆盖真实 gate。

## META2-P1-002
旧 `AUTONOMOUS_MASTER_QUEUE.md` 已过时：
- 部分 ARCH 任务代码已修；
- 部分新问题没进入 queue；
- 仍显示大量 unassigned/0 active。

ChiefCoordinator 首项工作就是重建 queue，而不是继续旧表。

---

# 27. P0 — Root CI / Pre-commit 没接新核心 gate

当前 root `.pre-commit-config.yaml` 主要：
```text
black
isort
flake8
mypy
```

FactorEngine nested pre-commit 虽有 DataAccess I/O checker，但从 repo root 安装 pre-commit 通常不会自动使用 nested config。

## CI2-P0-001 建立 root-level production workflow
建议：
```text
.github/workflows/quant-platform-production-gates.yml
```

至少跑：

```text
DataAccess physical I/O boundary
Operator ABI
placeholder/research_only admission
evidence integrity
Polars parity
DuckDB parity
q capability truth
q compile/runtime parity when runtime available
model purge/leakage
quantile correctness
batch-stream evaluator equivalence
clean-wheel
cold-start operator dependency integrity
```

## CI2-P0-002 q weak verifier 修完之前不能进 mandatory “production ready” gate

## CI2-P0-003 ABI 16 violations 清零后 mandatory

## CI2-P0-004 Polars `ts_zscore` 当前 known fail 不得被 allow-failure 永久掩盖

## CI2-P1-005 nightly：
```text
large property matrix
fault injection
soak
performance regression
```

---

# 28. P0 — I/O boundary 仍需全仓统一

FactorEngine local checker本身已改善，但 whole repo 仍有：

```text
root .data_access_allowlist.yaml
root check_data_access_allowlist.py
factor_engine PhysicalIOAuthorityPolicy
factor_engine local checker
```

## IO2-P0-001
确定全仓唯一 physical I/O policy authority。

如果因为职责需要两域：
```text
DataSourceReadAuthority
FactorResultPersistenceAuthority
```
也必须由同一个 policy schema 生成，不要两个独立 allowlist。

## IO2-P0-002
FactorEngine `SOURCE_READ` exemption 要继续收缩。
原始/市场数据读取最终应走 DataAccess；FactorEngine 只保留合法 factor result persistence / adapter boundary。

## IO2-P1-003
local checker `_get_call_repr` 对 alias/nested attribute 检测能力弱，统一采用更完整 AST binding analysis。

## IO2-P1-004
automatic exemption 使用真实 glob/path segment，不要 substring。

---

# 29. P0 — Packaging / clean-wheel

## PKG2-P0-001 实际 clean-wheel 不只 Modeling
必须真实跑：
```text
FactorEngine
DataAccess
FactorAssets
QuantEvaluator
FactorOptimizer
FactorPreprocess
Modeling
```

要求：
```text
build wheel
clean venv
install
cd 到源码树之外
public import
minimal real execution
```

## PKG2-P0-002 FactorEngine 没 q/PyKX install profile
如果 q 是正式 backend，增加：
```text
q / production-q runtime profile
```
明确 q/PyKX supported version、license preflight、runtime availability。

## PKG2-P0-003 production profile 不应让 FactorEngine 无 DataAccess 也“看起来完整”
建议 install profiles：
```text
research-minimal
production
production-q
```

## PKG2-P0-004 删除生产/审计工具绝对 home path、sys.path source injection

## PKG2-P1-005 FactorAssets adapter extras 真正声明 DataAccess/FE/QE integrations

---

# 30. DataAccess carry-forward：本轮没有理由自动关闭

这 13 commits 主要在 FE/Q/model/QE/FA；DataAccess 上轮尚未完成的证据仍要保留 OPEN/NEEDS_PROOF。

## DA2-P0-001 correctness identity caller audit
所有：
```text
64-bit
non-strict
hash_cache_key
```
caller 分类：
```text
ephemeral performance
correctness/result reuse
```
后一类全部 strict >=128-bit。

## DA2-P1-002 timezone canonical identity
同一 instant：
```text
2026-08-14 10:00+08
2026-08-14 02:00Z
```
是否同 identity，冻结为 canonical UTC instant。

## DA2-P1-003 list/tuple、set/frozenset identity
明确类型是否 semantic-equivalent，不能偶然由 serializer 决定。

## DA2-P0-004 DataSnapshotIdentity 与 ExecutionBuildIdentity 分离
代码 commit 变化不等于 data snapshot 变化。

## DA2-P0-005 legacy bucket API production caller audit
生产只走 validated layout policy。

## DA2-P0-006 PIT poison suite
至少：
```text
financial restatement
after-close announcement
future universe member
future extreme value
revision vintage
corporate action
```

## DA2-P0-007 Atomic publish fault injection
```text
disk full
rename fail
manifest fail
concurrent reader
concurrent writer
partial partition
```

---

# 31. 冷启动因子库重新认证

最新一次 demote 678 operators 后必须跑：

```text
FactorLibraryRevalidation
```

流程：
```text
factor
→ canonical DAG
→ transitive operator dependencies
→ current production admission
→ field/DataAccess PIT availability
→ backend evidence
→ strict-active / staged / invalid
```

输出至少：

```text
strict-active before
strict-active after
newly invalid count
affected factor families
replacement candidates
affected microclusters
affected macroclusters
backend availability impact
```

不能继续沿用 demotion 前的 strict-active 数量。

---

# 32. Production Truth Matrix

新增：

```text
PRODUCTION_TRUTH_MATRIX.md
```

不要再只有 `supported=True`。

每项至少四层：

| Capability | Declared | Implemented | Runtime Proven | Production Admitted |
|---|---:|---:|---:|---:|
| q ts_mean | ? | ? | ? | ? |
| q ts_beta | yes | suspect | no | no |
| Polars ts_zscore | yes | yes | FAIL | NO |
| return_decomp | yes | ABI mismatch | no | no |

再加：

```text
PIT Proven
Identity Complete
Evidence Fresh
```

---

# 33. 新 hard gates

```text
Q_PRODUCTION_SAFE_SINGLE_DEFINITION
Q_ZERO_DUPLICATE_CAPABILITY_AUTHORITIES
Q_COMPILE_RUNTIME_PARITY_REQUIRED
Q_COMPILER_ADMISSION_USES_EVIDENCE_AUTHORITY_ONLY
Q_EXACTLY_ONE_REACHABLE_LOWERING_PER_CANONICAL
Q_ZERO_BACKEND_OWNED_PARAM_DEFAULTS
Q_RESEARCH_PRODUCTION_INSTANCE_ISOLATION
Q_OUTPUT_SCHEMA_EXACT
Q_FANIN_INPUT_PRESERVATION
Q_WORKSPACE_NAMESPACE_ISOLATION
Q_NULL_STRING_SEPARATION
Q_NULLABLE_BOOL_PARITY
Q_SESSION_GENERATION_VALIDATION

BACKEND_Q_INCLUDED_IN_PRIMARY_REGISTRY
BACKEND_PHYSICAL_SPEC_TYPED_BACKEND
BACKEND_SPEC_ERROR_NOT_MISSING
SQL_EMITTER_HASH_REQUIRED

POLARS_ZERO_KNOWN_PARITY_FAILURES

OPERATOR_ABI_ZERO_VIOLATIONS
RESEARCH_ONLY_ZERO_PRODUCTION_EVIDENCE
COLD_START_ZERO_RESEARCH_ONLY_DEPENDENCIES

LABEL_CONTRACT_FULL_SEMANTIC_HASH
GAP_UNIT_EXPLICIT
PRODUCTION_PURGE_REQUIRES_CALENDAR
MODEL_PUBLIC_OOS_ZERO_UNCHECKED_PREDICT
MODEL_APPLICATION_DATES_REQUIRED
MODEL_ARTIFACT_IDENTITY_COMPLETE
ONE_MODEL_CONTRACT_AUTHORITY

QE_QUANTILE_SHAPE_STABLE
QE_QUANTILE_TIE_POLICY_IN_IDENTITY
QE_TOP_BOTTOM_ORIENTATION_CORRECT
QE_NUMPY_NUMBA_EXACT_PARITY

FA_DA_ADAPTER_REAL_INTEGRATION
CAMPAIGN_MONOTONIC_BUDGET
CAMPAIGN_ATOMIC_RESERVATION
MULTIFIDELITY_ZERO_UNUSED_CRITERIA
TEST_SPLIT_SEAL_ENFORCED

EVIDENCE_REGENERATED_NOT_REBOUND
EVIDENCE_IMPLEMENTATION_CLOSURE_HASH
ROOT_PRODUCTION_GATES_MANDATORY
ALL_PRODUCTION_PACKAGES_CLEAN_WHEEL
```

---

# 34. 首轮 Subagent 分配

## ChiefCoordinator
先：
```text
重建 AUTONOMOUS_MASTER_QUEUE
更新 AGENT_STATUS
更新 PRODUCTION_TRUTH_MATRIX
更新 CERTIFICATION_MATRIX
```

## Agent-Q-Truth
```text
Q2-P0-001~005
Q2-P0-015~019
```

## Agent-Q-Compiler
```text
Q2-P0-006~014
```

## Agent-Q-Runtime
```text
Q2-P0-020~034
```

## Agent-Backend
```text
BE2-P0-*
POL2-P0-*
```

## Agent-Operator
```text
ABI2-*
OP2-*
cold-start revalidation
```

## Agent-Modeling-FE
```text
FM2-*
```

## Agent-Modeling-Authority
```text
MODEL2-*
```

## Agent-QE
```text
QE2-*
```

## Agent-FA
```text
FA2-*
```

## Agent-Evidence-CI
```text
EV2-*
CI2-*
IO2-*
PKG2-*
META2-*
```

## IndependentReviewer
不能让实现 Agent 自己判 PASS。

---

# 35. 最优先前 20 项

立刻执行顺序：

```text
01 QE2-P0-001        top-bottom spread sign
02 Q2-P0-001         production_safe 单一定义
03 Q2-P0-002         production_ready 判定
04 Q2-P0-003         假认证 script
05 Q2-P0-004/005     双 q capability authority
06 Q2-P0-020         fan-in input 丢失
07 Q2-P0-021/022     q workspace/thread isolation
08 Q2-P0-015         PhysicalBackendRegion
09 Q2-P0-016/017     missing data/output schema hard fail
10 Q2-P0-018/019     runtime fallback/singleton
11 ABI2-P0-001       return_decomp ABI
12 POL2-P0-001       ts_zscore parity
13 FM2-P0-003        LabelContract semantic hash
14 FM2-P0-005/006    gap unit/test gap
15 FM2-P0-008~011    purge/calendar/test boundary
16 FM2-P0-012~015    OOS predict bypass
17 FM2-P0-016~023    artifact identity/time
18 MODEL2-P0-006     one Modeling authority
19 FA2-P0-001~006    DataAccess adapter
20 CI2-P0-001        root production gates
```

完成后继续剩余项，不能停。

---

# 36. 禁止的“修法”

禁止：

```text
把更多 operator 直接 research_only 就宣告功能完成
xfail/skip failing parity
修改 expected value 配合错误实现
q 失败后 silently Pandas fallback
broad except → unsupported
只改 FINAL_REPORT
只加 dataclass 不接 consumer
missing identity 用 ""
用 warning 代替 PIT/leakage fail
clean-wheel 里 sys.path 插源码
为了 gate 绿缩小 scanner coverage
```

---

# 37. 持续 Audit Miner

显式任务清完以后继续扫：

```text
except Exception
pass
TODO/FIXME
NotImplemented
return [] / {} / None / 0
default=str
repr/str
warnings.warn
global singleton
sys.path
/home/
production_ready
research_only
is_production_eligible
supported=True
fallback
```

这轮重点识别一种模式：

```text
Contract exists
BUT consumer not wired
```

它现在是全仓最常见风险。

---

# 38. 当前建议生产状态

在上述问题清理前：

```text
DataAccess:
  PARTIALLY_VERIFIED

FactorEngine Pandas reference:
  PARTIALLY_VERIFIED

Polars:
  PARTIALLY_VERIFIED
  KNOWN_PARITY_FAILURE = ts_zscore

DuckDB:
  PARTIALLY_VERIFIED
  continue systematic property/integration parity

q/K:
  NOT_PRODUCTION_CERTIFIED

FactorAssets:
  NOT_PRODUCTION_CERTIFIED

FactorEngine Modeling:
  NOT_PRODUCTION_CERTIFIED

Standalone Modeling:
  NOT_PRODUCTION_CERTIFIED

QuantEvaluator:
  NOT_PRODUCTION_CERTIFIED
  KNOWN_ORIENTATION_BUG = top_bottom_spread

Cold-start factor library:
  REVALIDATION_REQUIRED after 678 operator demotions

Overall:
  PARTIALLY_VERIFIED
```

禁止再写：
```text
All P0 resolved
Core platform production-ready
```

---

# 39. 最终验收问题

某域只有能回答下面问题才允许 `CERTIFIED`：

```text
capability 谁声明？
implementation 在哪里？
runtime 真跑过？
和 canonical oracle 一致？
parameter domain 一致？
null/NaN/Inf 一致？
PIT 一致？
data snapshot 是哪个？
universe 当时有效？
model artifact 什么时候真正可用？
有无 public API 绕过 OOS？
quantile 0 是最低还是最高？
long-short spread orientation 是否冻结？
evidence 是重跑生成还是手改 SHA？
root CI 真执行 gate？
clean wheel 离开源码树还能运行？
```

回答不了：
```text
UNKNOWN
```
不是：
```text
PASS
```
