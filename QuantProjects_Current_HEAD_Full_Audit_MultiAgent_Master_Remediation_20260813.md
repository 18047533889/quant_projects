# QuantProjects 当前 HEAD 全量审计 + 大规模 Multi-Agent 最终整改总任务书

> 仓库：`18047533889/quant_projects`  
> 外部审计基线：`main@ddb03749b7ff85e63b633770c43dfbbb7562af19`  
> 日期：2026-08-13  
> 目标：将 FactorEngine / DataAccess / Modeling / Filter / MultiBackend 收敛为可审计、PIT-safe、可直接使用、多后端语义一致、证据绑定当前 HEAD 的企业级量化基础设施。

---

# 0. 给执行 AI 的总指令

你接手的是一个已经被多个 AI / subagents 高频修改过的大型量化 monorepo。不要相信文件名中的 `FINAL`、`STATUS`、`PRODUCTION_READY`、注释中的 `PASS`、旧验收报告或过去 agent 的口头结论。**代码、真实执行测试、FINAL HEAD evidence 才是事实。**

本任务不是“修下面列出的几个 bug 即结束”，而是：

1. 重新读取真实最新 `main`；
2. 将本文问题逐项分类为 `OPEN / PARTIAL / FIXED / NOT_APPLICABLE / NOT_PROVEN`；
3. 主动继续审计，发现本文遗漏的问题；
4. 大量使用 subagents 并行修复；
5. 每个 agent 独立 worktree / branch / ownership；
6. 按 merge wave 集成；
7. 在 FINAL MERGED HEAD 上重新执行全量验证；
8. evidence 必须重新生成并绑定 FINAL HEAD；
9. 未真实执行的测试只能写 `NOT_RUN`；
10. 最终目标是 correctness + PIT + semantic parity + production operability + evidence truthfulness。

---

# 1. 不可改变的量化语义底线

- 主要场景：股票横截面 / 日频多因子，同时支持分钟→日频。
- 严格 Point-in-Time（PIT）。
- 禁止未来函数、未来数据、未来 revision、未来 universe。
- 同日截面只能使用 DecisionClock 下已经可得的信息。
- 财务数据必须尊重真实公告时间 / revision vintage / source snapshot。
- 因子评估收益标签默认语义：**VWAP-to-VWAP**。
- canonical operator 名称必须对应真实数学含义。
- `mutual_information` 不能实际返回 `corr²`。
- `fractional_difference` 不能实际返回普通一阶差分加权。
- `Lyapunov / Markov / Kramers-Moyal / PCA / first-passage` 不能用 volatility/rolling mean proxy 冒充。
- 无法正确实现时宁可 `QUARANTINED / RESEARCH_ONLY / NOT_IMPLEMENTED`，不能伪装 Production。
- Production fail-closed。
- Backend 不拥有 canonical 语义。

目标架构：

```text
Canonical Semantics
    ↓
Logical IR
    ↓
Physical Implementation Registry
    ↓
Backend Capability + Evidence
    ↓
Region Planner
    ↓
Representation / Residency
    ↓
Cost / Resource Plan
    ↓
Pandas / Polars / DuckDB / ClickHouse / q
```

Pandas/NumPy reference implementation应作为数学 oracle；其他 backend必须证明 parity。

---

# 2. 当前基线与 CI

外部审计 HEAD：

`ddb03749b7ff85e63b633770c43dfbbb7562af19`

开始执行必须重新：

```bash
git fetch --all --prune
git status
git rev-parse HEAD
git log -1 --oneline
```

如果 HEAD 已变化，所有判断重新验证。

## P0-001 — 当前 GitHub Actions 不是 Green

当前 `ddb037...` 存在 6 个 push workflow run。外部已确认至少：

- AutoFactorEvaluation Platform Integration = failure
- GTJA191 Production Gate = failure
- Operator Surface Gate = failure
- FactorEngine Contract Hardening = failure

因此 CURRENT HEAD 不可声明 CI PASS。

第一波 agent必须：
- 拉取全部 workflow run / jobs / steps / logs；
- 区分 workflow 配置、runner、dependency、permission、import、syntax、test、generated artifact stale；
- 建 `CI_FAILURE_LEDGER.jsonl`；
- 每个失败明确 owner；
- FINAL HEAD 所有 required workflows重跑。

外部 infra无法执行时是 `INFRA_BLOCKED / NOT_RUN`，不是 PASS。

---

# 3. Operator / Polars Native 最高风险

## P0-002 — Polars Native Import / ABI 不一致

当前：

`factor_engine/cleaned_operators/polars_native/technical_final.py`

向 `cleaned_operators.base_polars.OperatorMetadata` 传入：

```python
timing_kind="INTRADAY_BARS"
production_ready=True
```

但当前 `base_polars.OperatorMetadata` 没有这些字段。

必须新增硬门：

```text
FULL_REPO_PARSE
FULL_REPO_COMPILE
ALL_REQUIRED_MODULE_IMPORT
ALL_POLARS_NATIVE_MODULE_IMPORT
REGISTRY_BOOTSTRAP
PUBLIC_API_IMPORT
```

仅 py_compile 不够。

不要为了让 import 通过而随意在 `base_polars` 再复制一套 semantic metadata；最终 semantic contract应只有一个 authority。

## P0-003 — 大量正式 canonical 仍是 Placeholder

`polars_native/ts_advanced_batch5.py` 当前仍明确写复杂算法包含 skeleton/TODO。

已确认典型包括：

- `ts_feature_mode_share`
- `ts_fir_lowpass_causal`
- `ts_first_passage_bias`
- `ts_first_passage_conditional_time`
- `ts_fisher_information_shift`
- `ts_forbidden_ordinal_pattern_ratio`
- `ts_fractional_difference`
- `ts_fractional_difference_discarded_weight_mass`
- `ts_generalized_hurst_exponent`
- `ts_generalized_hurst_spread_q1_q4`
- `ts_lag_of_peak_corr`
- `ts_lagged_mutual_information`
- `ts_lempel_ziv_complexity`
- `ts_level_shift_score`
- 以及同目录/其他 backend中的全部 TODO / Placeholder / Proxy / Skeleton / Approximation。

处理规则：

1. canonical有明确数学定义 → 真正实现；
2. 暂不能正确实现 → `RESEARCH_ONLY / QUARANTINED / NOT_IMPLEMENTED`；
3. Production registry不可路由；
4. mining grammar不可抽样；
5. operator catalog明确不可直接用；
6. capability不得把 proxy计作 production coverage。

建立全仓 Operator Authenticity Audit，扫描：

```text
TODO
FIXME
placeholder
proxy
skeleton
approximation
simplified
not implemented
for now
temporary
```

## P0-004 — 明确未来函数：`ts_leverage_effect`

当前 Polars 实现含：

```python
ret = feature.diff()
fut_vol = (ret.shift(-1) ** 2).rolling_mean(window)
```

同时标 `pit_safe`。

必须立即移出 production，并重新定义：
- production historical leverage effect只能用当前及历史已观测数据；
- 如果定义本身依赖 future volatility，则应属于 research/evaluation label diagnostic，而不是 production factor operator。

全仓建立静态 future scanner：

```text
shift(-N)
lead
future
next_
iloc[i+1]
index + 1
center=True
filtfilt
sosfiltfilt
two-sided
full-sample PCA/normalization/decomposition
```

再做动态 prefix/future poison：

```text
X[0:T] -> y_old
append future sentinel -> y_new
assert y_old[:T] == y_new[:T]
```

NaN mask也必须一致。

## P0-005 — 技术指标 canonical 数学意义需逐一重审

例如当前 `ElderRay`：

```python
bull = high - ema
bear = low - ema
elder_ray = bull - bear
```

代数上等于 `high-low`，EMA完全消失。

所有 technical / candlestick / pattern 算子都要：
- 对 authoritative definition；
- 确认输出语义；
- 初始化/Wilder/EMA/min_periods/missing版本化；
- Pandas/Polars parity；
- 不要因为类名正确就认为算法正确。

---

# 4. Backend Capability / Physical Authority

## P0-006 — Capability duplicate部分修了，但新 ExecutionKind duplicate出现

当前 `backend/capability_registry.py` 已降为 deprecated facade，这是正确方向。

但：
- `backend/operator_capability.py` 定义一套 `ExecutionKind`
- `backend/polars_backend_kind.py` 又定义另一套 `ExecutionKind`

必须只保留一个 authoritative：

```text
PhysicalImplementationSpec
ExecutionKind
BackendKind
CapabilityLevel
```

建议独立放：
`backend/contracts/physical_implementation.py`

## P0-007 — Production PhysicalImplementationSpec必须显式完整

production不能长期靠：
- `inspect.getsource()`
- `"pl." in body`
- `"to_pandas()" in body`
- source/module字符串

猜 physical kind。

每个 production implementation显式声明：

```text
canonical
backend
implementation_id/version/hash
execution_kind
input_representation
output_representation
supports_lazy
supports_fusion
supports_streaming
materializes_full_panel
materialization_barrier
stateful
checkpointable
state_schema_version
requires_sorted
requires_partitioning
supports_nulls/nan/inf/min_periods/broadcast
parameter_domain_hash
semantic_contract_hash
backend_evidence_hash
```

缺失 spec：
`production capability = NOT_ELIGIBLE`。

ExecutionKind 至少区分：

```text
PANDAS_REFERENCE
POLARS_NATIVE_EXPR
POLARS_NATIVE_GROUP
POLARS_NATIVE_STREAMING
POLARS_NUMPY_KERNEL
POLARS_NUMBA_KERNEL
POLARS_PYTHON_UDF
POLARS_PANDAS_DELEGATE
DUCKDB_NATIVE_SQL
CLICKHOUSE_NATIVE_SQL
Q_NATIVE
UNSUPPORTED
```

NumPy kernel不等于可融合的 Polars expression。

---

# 5. q/kdb+ 后端必须从“平行世界”并入主 Runtime

## P0-008 — 主 BackendCapabilityRegistry仍未真正支持 q

当前虽然有 `BackendKind.Q_KDB`，但：
- `BackendName` 不含 q；
- q query最终仍 not implemented；
- production eligible backends不含 q；
- flat capability export不含 q；
- get_best_backend不含 q；
- BackendRouter RequestedBackend不含 q。

与此同时 q有自己的：
`q_capability.py / q_compiler.py / q_executor.py / q_process_manager.py / q_adapter.py`

必须并入同一个：
`PhysicalImplementationSpec → BackendCapabilityRegistry → RegionPlanner → Cost → Representation → Residency → Resource → Telemetry → Evidence`。

## P0-009 — q capability hardcoded NATIVE集合与 compiler实际 lowering漂移

`q_capability.py` 声称大量算子 NATIVE，但 compiler并没有完整 mapping。

删除 parallel hardcoded truth。

q production capability必须来自：

```text
physical spec
+ lowering exists
+ bound-parameter compile
+ semantic parity evidence
+ runtime/license evidence
```

## P0-010 — q compiler 多个映射当前不能证明等价

停止扩大 q coverage，先认证现有算子。

重点：

### lag
N>1使用明确认证的 q N-lag primitive/wrapper，验证 group/session/missing。

### rolling
`mavg/mdev/mmin/mmax/mcount` 默认 warmup/null/ddof不一定等于 FactorEngine，必须 wrapper + parity。

### ts_corr / ts_cov
当前形态接近 `window cor[x;y]`，但 aggregate correlation不等于 rolling series。必须真正 rolling或移出 capability。

### ts_beta
不能只对末尾 window slice算一次，必须逐时点 rolling。

### median/product/var/rank/zscore
确认不是 aggregate、自由变量、错误 window binding。

### rank ties
明确 average/min/max/dense/percentile denominator，与 canonical一致。

## P0-011 — q residency仍会把每个中间 Region下载成 Pandas

当前 execute_region无论是否要 resident handle都会：

```python
q_result = q(plan.output_table)
output_df = type_adapter.q_to_pandas(q_result)
```

所以虽然避免了下一 Region重复 upload，但中间 Region仍 q→Python。

正确目标：

```text
Q A -> Q B -> Q C
```

中间不 materialize Python；只有 backend boundary/final output才下载。

新增硬测试：

```text
intermediate_q_to_python_bytes == 0
final_q_to_python_count == 1
```

## P0-012 — q resident handle不足以支撑真实 DAG

当前 handle需扩展：
- backend
- process_id/session_id/runtime_generation
- adapter_version
- schema_hash
- semantic_contract_hash
- source_snapshot/PIT/DecisionClock identity
- ownership/refcount/lifetime
- table generation
- immutable/mutable
- created_at/last consumer

支持：
- stale handle fail
- process restart
- fan-out/fan-in
- multi-input region
- branch reuse
- cleanup/drop
- cancellation
- OOM

不要只支持线性 A→B→C demo。

## P0-013 — q residency测试主要是 MagicMock，不证明真实 q

保留 mock control-flow tests，同时新增真实 q integration：
- real compile
- real execution
- real alias/residency
- conversion
- null/time
- parity
- transfer telemetry
- process restart

q runtime/license不可用：
`Q_REAL_RUNTIME_INTEGRATION=NOT_RUN` 且 `Q_PRODUCTION_SAFE=False`。

## P0-014 — q adapter/runtime authority重复

当前至少：
- `backend/q_backend_governance.py`
- `backend/q_backend/q_adapter.py`
- `backend/q_backend/q_process_manager.py`

重复 QTypeAdapter / version / runtime info / null semantics。

收敛职责：
- runtime contract：version/license/session
- type adapter：type/null/time conversion
- process manager：lifecycle
- main capability：production eligibility

旧 governance最终只可做 compatibility facade。

## P0-015 — q null/type conversion不完整

当前 integer null handler只按一个 int sentinel处理，未按 byte/short/int/long分 dtype。

同时修：
- chained assignment
- object→symbol把 missing和empty混合
- timezone/date/timestamp/timespan/NaT
- nullable boolean/string
- symbol categorical semantics

无法无损表示 → fail-closed。

## P0-016 — PyKX API必须按真实安装版本 smoke-test

不要凭记忆使用：
`zero_copy=True / qsql.fromtable / qsql.totable`。

运行时：
`inspect.signature + real conversion smoke tests`。

zero-copy只是优化，不是 correctness。

---

# 6. MultiBackend Runtime 中的“假完成”风险

## P0-017 — BatchGlobalOptimizer报告 APPLIED，但实际上 return原 DAG

当前 `_apply_optimization()` 明确是：
`For now, return dag unchanged`，
但上层仍累加 applied_count 和 savings。

必须改为：
- before/after structural hash不同；
- semantic equivalence通过；
- 才能 APPLIED。
否则 `DETECTED_NOT_APPLIED`，不得计 savings。

## P0-018 — Streaming Planner对未知算子 fail-open

当前逻辑相当于：
“只要不在 materialization blacklist，就默认 streaming”。

生产必须只读 `PhysicalImplementationSpec.supports_streaming`。
unknown = false。

## P0-019 — Streaming scheduler可能破坏拓扑顺序

当前将原 topo order拆成：
`streaming + other + materialization`，
可能把 consumer放在 dependency之前。

改为 priority-aware Kahn topological scheduling：
只在 ready queue内部选择优先级。

Property gate：

```text
for every edge u -> v:
position[u] < position[v]
```

## P0-020 — broad exception返回空 plan/空 optimizer结果

区分：
- NoOpportunity
- Unsupported
- InfrastructureFailure
- SemanticFailure
- ResourceFailure

生产 semantic/planner错误不可静默变空结果。

## P0-021 — 文档“Production-Ready”与实现不一致

`runtime/multibackend/IMPLEMENTATION_SUMMARY.md`声称 Production-Ready / no placeholders，
实际存在 return unchanged scaffolding。

以后 STATUS / FINAL / SUMMARY全部由 machine-readable evidence生成，禁止手写成为 authority。

---

# 7. Polars 后端专项

## P1-022 — Thread model主要方向已修，清理旧概念

当前已正确停止 runtime `pl.Config.set_thread_count()`。

后续：
- ResourceBroker控制 concurrent Polars jobs；
- `POLARS_MAX_THREADS` 在 import前设；
- 需要不同配置用 spawn worker process；
- `optimal_polars_threads` 若无法实际约束 region线程数，应重命名/移除避免误导。

## P1-023 — Polars Lazy Region

目标：
`FactorEngine Region → one LazyFrame graph → one collect at boundary`

operator级 collect必须：
`materialization_barrier=True`
并进入 cost model。

## P1-024 — Production Polars显式 spec覆盖率

最终：
`POLARS_PRODUCTION_WITHOUT_PHYSICAL_SPEC = 0`

---

# 8. DuckDB / ClickHouse SQL

## P0-025 — “能 compile SQL”不等于 production-safe

必须：
`compile + execute + reference parity + null/missing + bound params + PIT + dialect evidence`

## P0-026 — DuckDB / ClickHouse独立认证

每个：
`canonical × bound params × dialect`
独立 evidence。

## P0-027 — ClickHouse integration可整套 skip

当前真实 CH integration使用 `skipif(not _ch_available())`。

建立：
- DuckDB mandatory CI；
- ClickHouse service container / dedicated runner / nightly真实环境；
- 无 infra则 NOT_RUN，不是 PASS。

当前仅 `len>0/notna.any()`不够；必须与 Pandas reference数值对比。

---

# 9. Operator Direct-Use Certification

建立：

```python
OperatorOperationalStatus:
    DIRECT_USE
    RESEARCH_ONLY
    QUARANTINED
    NOT_IMPLEMENTED
    DEPRECATED
```

成为 DIRECT_USE必须满足：

1. 数学定义明确；
2. canonical名与实现一致；
3. reference implementation存在；
4. 参数域明确；
5. 无 silent dtype coercion；
6. missing policy明确；
7. warmup/min_periods明确；
8. unit/value semantics明确；
9. timing/grain明确；
10. PIT/future poison PASS；
11. deterministic；
12. stateful时 checkpoint/resume PASS；
13. input alignment PASS；
14. edge cases PASS；
15. 至少一个 production-safe backend；
16. evidence绑定 FINAL HEAD；
17. mining/export catalog不误标。

生成：
- `docs/operator_direct_use_ledger.json`
- `docs/operator_direct_use_ledger.csv`
- `docs/operator_direct_use_report.md`

Markdown从 JSON生成。

---

# 10. 自动 Operator Test Matrix

每个 production canonical至少测试：

## 数据
- random finite
- positive price
- returns
- noisy trend
- monotone
- constant
- zero variance
- all zero
- short history

## Missing
- leading/trailing/internal NaN
- long gap
- whole instrument missing
- whole date missing

## Extreme
- inf/-inf
- huge/tiny values
- zero denominator

## Cross-sectional
- ties
- duplicate
- 1/2 stocks
- group of one
- missing group
- shuffled instrument order

## Time-series
- window boundary
- min_periods
- irregular trading sessions
- suspension gaps

## Params
- min/max/default
- invalid
- bool as int
- non-integral float
- NaN/Inf params
- unknown kwarg

## PIT
- future append poison
- revision poison
- universe poison
- source-vintage replay

---

# 11. Stateful Operator Certification

metadata写 `checkpointable=True` 不算证据。

要求真实：
- StateSchema
- serialize_state
- deserialize_state
- resume
- state_schema_version

测试：
`full_batch == chunk1 + checkpoint + chunk2`

覆盖 arbitrary split / NaN / warmup / state change / multi-instrument。

---

# 12. Filter Layer

## P1-028 — RobustEMA missing policy显式化
当前 missing会 hold previous output。纳入统一 `MissingPolicy`：

```text
BREAK_RESET
OUTPUT_NAN_HOLD_STATE
OUTPUT_NAN_ADVANCE_STATE
HOLD_OUTPUT
PREDICT_ONLY
FAIL_CLOSED
```

## P1-029 — SuperSmoother gap语义
当前 NaN不更新 state，下一 valid继续 gap前状态。明确：
`PHYSICAL_BAR_TIME / OBSERVATION_TIME / RESET_AFTER_GAP`

## P1-030 — warmup contract与实际输出一致
不要文档写 period warmup而前两点已经输出。

分开：
`availability_lag_bars` 与 `response_delay_bars`。

## P1-031 — rank helper不得 silent O(N²) fallback
生产用内建 vectorized rank / scipy mandatory；不能在 5000股票时悄悄退化。

## P1-032 — unit metadata
rank percentile = dimensionless/rank_percentile。
一般平滑：`unit(output)=unit(input)`。

## P1-033 — zero scale不是 numerical epsilon决定经济行为
增加 `ZeroScalePolicy`。

## P1-034 — quantile hysteresis missing不能隐式 EXIT
由 MissingPolicy决定。

同时继续检查：
- cost-aware filter量纲；
- missing cost fail-closed；
- Filter层不替代最终 Portfolio turnover optimizer；
- filtered model artifact必须绑定 filter chain spec/hash/state。

---

# 13. Modeling

## P0-035 — `_maturity_cutoff` 当前仍是假 fail-closed

当前注释说无法扩展未来 calendar时 fail-closed，但实现会返回 `final_fit_end`。

对 horizon>0，这会低估 label maturity时间。

正确：
- `raise LabelMaturityUnavailable`
或
- artifact `UNAVAILABLE_UNTIL_CALENDAR_EXTENDS`

绝不能返回 anchor假装成熟。

## P0-036 — Artifact时间字段拆开

```text
final_fit_anchor_end
label_maturity_cutoff
fit_completed_at
artifact_available_at
activation_at
```

约束：
`artifact_available_at >= max(label_maturity_cutoff, fit_completed_at)`
`activation_at >= artifact_available_at`

## P0-037 — Model evidence stale

当前 `MODEL_CURRENT_HEAD.json` 仍绑定旧 `854bdc...`，而审计 HEAD为 `ddb037...`；
且 `final_direct_use_ready_count=0`、`runtime_versions={}`。

所有 model evidence FINAL HEAD重生成。

优先把现有：
PCR / PLS / ElasticNet / Regime / MoE
生命周期闭环，不要先继续加模型 zoo。

必测：
- date-group split
- label-interval purge
- embargo
- train-only preprocessing
- validation search
- hidden final holdout
- per-date cross-sectional IC
- sample adequacy
- frozen artifact
- artifact-as-of replay
- immutable catalog
- drift/revocation

---

# 14. DataAccess

## P0-038 — 现有 R32 “FINAL”实际上自己写非最终验收

当前报告正文写：
- 中期进度报告（非最终验收）
- P0 60/112
- P1 0/150
并绑定旧 HEAD。

不能因为文件名 FINAL就视为完成。

## P0-039 — vacuous tests仍存在

`dataaccess/tests/unit/test_fe_da_p1_fixes.py` 仍有：
- `Can't easily test`
- `Would need a real DataAccessSource`
- `This would raise`
但没真正触发/断言。

全部修。

用 `monkeypatch.setenv/delenv`，避免污染全局 env。

## P0/P1-040 — 重新核验旧报告未闭环项

逐项按当前代码分类：
- DQ异常 fail-open？
- Coverage自然日 vs trading sessions？
- ExperimentSnapshot identity？
- changed_time_range timing？
- multi-column change impact？
- FactorSourcePlan typed binding？
- dependency extraction fail-closed？
- FactorBatchPlan vs ReadWavePlanner？
- backend actual counter vs estimate？
- metadata write authorization？
- 所有旧 P1。

不允许从旧报告直接复制 OPEN，也不允许无证据写 FIXED。

---

# 15. FE × DA 边界

DataAccess：
`source/PIT/snapshot/vintage/schema/physical data authority`

FactorEngine：
`canonical semantics / DAG / computation`

生产 source dependency使用 typed binding，例如：

```text
ColumnSourceBinding
dataset
field
market
provider
timeframe
temporal_policy
snapshot
availability
```

dependency extraction失败 → fail-closed。

---

# 16. Evidence Truthfulness

每个 evidence绑定：

```text
git_sha
tree_hash
registry_digest
semantic_contract_digest
physical_implementation_digest
backend_capability_digest
python/numpy/pandas/polars/duckdb/pyarrow
clickhouse server/client when tested
pykx/q version/license family when tested
```

禁止 evidence自己证明自己：
真实执行 → raw result → certifier → evidence → report。

STATUS / FINAL / SUMMARY文档由 evidence生成。

---

# 17. 建立项目级 Agent Constitution / Skill

当前外部审计未发现根目录 `AGENTS.md`，也未发现 `.claude/` project skill。

第一阶段创建：

```text
AGENTS.md
docs/agent/PROJECT_CONSTITUTION.md
docs/agent/OWNERSHIP.yaml
docs/agent/ISSUE_LEDGER.jsonl
docs/agent/AGENT_MANIFEST.schema.json
docs/agent/MERGE_WAVES.md
docs/agent/FINAL_GATE_MATRIX.json
```

如果执行平台支持 Claude Code Skill，再建：

```text
.claude/skills/quant-projects-enterprise-remediation/SKILL.md
```

Skill/Constitution必须包含：
- PIT/no future
- canonical真实性
- VWAP-to-VWAP标签
- DA source authority
- Pandas reference oracle
- backend semantic parity
- production fail-closed
- evidence current-head
- ownership规则
- merge waves
- test gates
- 禁止全仓 regex数值改写
- 未跑测试不得 PASS
- 不得存在双 authority
- placeholder不可冒充实现

所有 subagent启动时先读。

---

# 18. 30–60 个 Subagents 的并行方案

不要因为怕冲突而只开少量 agent。目标是**大量但可控并行**。

每个 agent必须：
- pinned base SHA
- 独立 branch
- 独立 git worktree
- allowed_files
- forbidden_files
- owned authority
- targeted tests
- agent manifest

`modified_files ⊆ allowed_files`，否则拒绝 merge。

建议约 55 个 workstreams：

## A — Baseline / Audit
A01 HEAD/dependencies  
A02 6个 CI failures  
A03 parse/compile  
A04 imports/public API  
A05 registry/duplicate authority  
A06 future scanner  
A07 placeholder scanner  
A08 stale evidence scanner  

## B — Pandas Reference families
B01 arithmetic  
B02 rolling TS  
B03 cross-sectional  
B04 group  
B05 robust/statistics  
B06 technical  
B07 state/event  
B08 intraday  
B09 fundamental/PIT  
B10 nonlinear/complexity  

## C — Polars
C01 metadata/ABI  
C02 physical spec migration  
C03 basic expr  
C04 rolling  
C05 CS/group  
C06 technical  
C07 intraday  
C08 advanced placeholders  
C09 stateful  
C10 Lazy region/fusion  
C11 parity/performance  

## D — SQL
D01 DuckDB basic  
D02 DuckDB rolling  
D03 DuckDB CS/group  
D04 DuckDB edge/null  
D05 ClickHouse basic  
D06 ClickHouse rolling  
D07 ClickHouse CS/group  
D08 CH real CI  
D09 SQL bound-param evidence  

## E — q
E01 capability integration  
E02 runtime/type authority  
E03 arithmetic/basic  
E04 lag/rolling  
E05 std/var/cov/corr/beta  
E06 rank/CS/group  
E07 null/time/adapter  
E08 residency DAG  
E09 process/lifetime/resource  
E10 real q integration/evidence  

## F — Runtime
F01 unified ExecutionKind  
F02 Region Planner authority  
F03 representation authority  
F04 resident value graph  
F05 transfer optimizer  
F06 CSE  
F07 streaming scheduler  
F08 memory/liveness/spill  
F09 cost model  
F10 typed errors/retry  
F11 telemetry/explain plan  

## G — DA / Model / Filter
G01 DA vacuous tests  
G02 DA PIT/snapshot  
G03 DA DQ/coverage/change  
G04 FE×DA binding  
G05 model label maturity  
G06 model artifact lifecycle  
G07 model evidence  
G08 filter contracts  
G09 filter checkpoint  
G10 filter evidence  

## V — Independent verification
V01 operator authenticity  
V02 PIT/future  
V03 backend parity  
V04 state/checkpoint  
V05 CI/evidence truthfulness  

Orchestrator不要承担大量 leaf coding。

---

# 19. Shared Authority 文件必须 SINGLE OWNER

至少：

```text
cleaned_operators/registry.py
cleaned_operators/base.py
cleaned_operators/base_polars.py
cleaned_operators/__init__.py
backend/operator_capability.py
backend/contracts/*
backend/backend_router.py
planner/physical contracts
representation/residency contracts
DataAccess PIT/runtime authority
Model artifact core contracts
FilterContract
```

leaf agent需要改 shared authority时提交 `CONTRACT_CHANGE_REQUEST`，由 owner统一改。

---

# 20. 禁止全仓正则批量改 Numerical Code

禁止 global sed/perl/regex 修改：
- np.where
- np.divide
- int(...)
- parentheses
- valid.sum
- NaN/Inf masks
- rolling/shift/lag/rank/ddof/min_periods

机械 transformation必须 AST-aware + 单 family + compile + targeted tests + diff review。

---

# 21. Merge Waves

Wave 0：agent governance / baseline / CI ledger  
Wave 1：leaf operator repairs  
Wave 2：shared semantic/physical contracts  
Wave 3：backend + planner/residency  
Wave 4：DA/Model/Filter  
Wave 5：full integration/performance/PIT  
Wave 6：evidence + generated docs  

每 wave后：
- compile
- imports
- registry
- targeted integration
- duplicate-authority scan

失败自动 bisect本 wave commits。

---

# 22. Agent Manifest

每个 agent结束输出：

```json
{
  "task_id": "",
  "base_sha": "",
  "commit_sha": "",
  "owned_files": [],
  "modified_files": [],
  "canonicals_touched": [],
  "backends_touched": [],
  "semantic_contracts_touched": [],
  "tests_run": [],
  "tests_passed": [],
  "tests_failed": [],
  "tests_not_run": [],
  "evidence_generated": [],
  "known_limitations": [],
  "remaining_todos": [],
  "status": "IMPLEMENTED|PARSE_VERIFIED|IMPORT_VERIFIED|UNIT_VERIFIED|PARITY_VERIFIED|PIT_VERIFIED|INTEGRATED|NOT_PROVEN"
}
```

Subagent禁止写 “100% complete / production ready / all fixed”。
只有 FINAL independent verifier可写 `PRODUCTION_CERTIFIED`。

---

# 23. Typed Backend Errors

禁止按错误字符串分类。

建立：

```text
BackendError
├── BackendUnavailableError
├── BackendTransportError
├── BackendTimeoutError
├── BackendOOMError
├── BackendUnsupportedError
├── BackendSemanticMismatchError
├── BackendDataContractError
├── BackendLicenseError
├── BackendStaleHandleError
└── BackendInfrastructureError
```

semantic/PIT mismatch绝不 fallback后继续。

---

# 24. Cost Model

任何：
- 1,000,000 rows/sec default
- 1.2 memory multiplier
- estimated speedup

没有真实 calibrated evidence时：
`routing_basis=HEURISTIC`。

Production优先 conservative或环境匹配 measured baseline。

measured baseline绑定：
- implementation hash
- backend version
- CPU/core/RAM/storage
- data shape
- operator params
- representation

---

# 25. Performance

Correctness后测试真实规模：

```text
5k stocks × 1y
5k × 5y
5k × 10y
realistic minute panel
100 factors
1k factors
10k candidate DAG
```

测：
- wall time
- CPU
- peak RSS
- bytes read
- bytes transferred
- materialization count
- backend switches
- CSE hit
- resident reuse
- spill
- throughput

防止 “Polars” 实际 pandas roundtrip。

---

# 26. FINAL HARD GATES

FINAL merged HEAD必须 machine-readable 输出：

```text
FINAL_HEAD_PINNED
FULL_REPO_PARSE
FULL_REPO_COMPILE
REQUIRED_IMPORTS
POLARS_NATIVE_IMPORTS
REGISTRY_BOOTSTRAP
NO_DUPLICATE_CANONICAL_AUTHORITY
NO_DUPLICATE_EXECUTION_KIND_AUTHORITY
NO_PRODUCTION_PLACEHOLDER
NO_PRODUCTION_FUTURE_ACCESS
OPERATOR_DIRECT_USE_LEDGER_COMPLETE

PANDAS_REFERENCE_SEMANTICS
POLARS_PHYSICAL_SPEC_COMPLETE
POLARS_REFERENCE_PARITY
POLARS_NO_FAKE_NATIVE
POLARS_LAZY_REGION

DUCKDB_REFERENCE_PARITY
CLICKHOUSE_REFERENCE_PARITY
CLICKHOUSE_REAL_INTEGRATION

Q_MAIN_CAPABILITY_INTEGRATED
Q_CANONICAL_IR_ONLY
Q_SEMANTIC_AUTHORITY_ZERO
Q_REFERENCE_PARITY
Q_NULL_TIME_PARITY
Q_PIT_PARITY
Q_INTERMEDIATE_DOWNLOAD_ZERO
Q_RESIDENT_HANDLE_VALIDATION
Q_REAL_RUNTIME_INTEGRATION

REGION_PLANNER_SINGLE_AUTHORITY
REPRESENTATION_SINGLE_AUTHORITY
RESIDENCY_SINGLE_AUTHORITY
COST_MODEL_TRUTHFUL
STREAMING_TOPOLOGICAL_VALIDITY

DA_PIT
DA_SNAPSHOT
DA_DQ
DA_CHANGE_IMPACT
DA_FAULT_INJECTION
DA_VACUOUS_TESTS_ZERO

MODEL_LABEL_MATURITY_FAIL_CLOSED
MODEL_ARTIFACT_TIME_CONTRACT
MODEL_CURRENT_HEAD_EVIDENCE

FILTER_CONTRACT_COMPLETE
FILTER_STATE_CHECKPOINT
FILTER_CURRENT_HEAD_EVIDENCE

GITHUB_ACTIONS_REQUIRED_GREEN
EVIDENCE_FINAL_HEAD
DOCS_GENERATED_FROM_EVIDENCE
```

没执行 = NOT_RUN，不得 PASS。

---

# 27. Definition of Done

## Operator
- 每个 registered canonical有 operational status；
- production canonical无假实现；
- 无未来访问；
- params/missing/warmup/unit/timing明确；
- direct-use ledger完整。

## Backend
- Pandas oracle稳定；
- Polars真实分类；
- DuckDB parity；
- ClickHouse真实测试或明确 NOT_RUN；
- q进入主 capability/planner；
- q中间 region零无意义下载；
- backend切换最小化。

## Runtime
- planner拓扑正确；
- unknown fail-closed；
- optimizer不虚报；
- resource/cost/telemetry真实。

## Data
- PIT/source/snapshot/vintage由 DA唯一 authority；
- destructive/fault tests。

## Model
- label maturity fail-closed；
- artifact availability时间正确；
- evidence current-head。

## Filter
- missing/gap/state/checkpoint真实；
- Filter不替代最终 Portfolio optimizer；
- evidence current-head。

## CI / Evidence
- FINAL HEAD required CI green；
- evidence FINAL HEAD；
- docs从 evidence生成。

---

# 28. Orchestrator执行节奏

不要先写长报告然后不改代码。

```text
AUDIT
→ ISSUE LEDGER
→ OWNERSHIP
→ SPAWN AGENTS
→ LEAF FIXES
→ MERGE WAVE
→ INTEGRATION
→ ACTIVE RE-AUDIT
→ NEXT WAVE
→ FINAL VERIFIERS
→ EVIDENCE
```

审计和修复并行。

发现新的严重问题：
直接增加 issue / agent，不用等用户再次提示。

资源不足时优先 correctness / PIT / semantic authority，不要用 placeholder赶进度。

---

# 29. 可直接发给新 AI 的总 Prompt

你现在接管 `18047533889/quant_projects` 的企业级最终整改。

请先完整阅读本任务书，然后：

1. 重新确认最新 main SHA；
2. 创建 `AGENTS.md`、project skill/constitution、ownership、issue ledger；
3. 将本任务书所有 issue按当前代码重新分类；
4. 主动继续全仓代码审计，本文不是问题上限；
5. 根据文件 ownership启动尽可能多的独立 subagents，目标 30–60 个 workstreams；每个使用独立 branch + git worktree，不共享 mutable working tree；
6. 优先解决 P0 correctness / PIT / fake implementation / backend authority / current CI；
7. 多 agent可以很多，但 shared authority必须 single owner；
8. 每 wave集成、跑 integration gate，再进入下一 wave；
9. 每次 merge后继续 re-audit；
10. 不能停止在“测试大部分通过”；
11. 所有不能直接使用的 registered operator必须处理：能明确实现的真正实现，无法证明正确的 quarantine/research-only，并从 production/mining surface剔除；
12. Pandas / Polars / DuckDB / ClickHouse / q 全部建立真实语义 parity与能力 evidence；
13. q必须正式并入主 BackendCapability/RegionPlanner，不能是平行世界；
14. q residency必须做到中间 region不回 Pandas；
15. MultiBackend optimizer/planner不能虚报 applied，也不能 unknown fail-open；
16. DataAccess、Model、Filter全部重新对 FINAL HEAD验收；
17. 不要把失败测试 skip掉来过 gate；
18. 不要修改测试去迎合错误输出；
19. 不要把 placeholder改名后继续冒充；
20. 不要 broad exception吞掉 production semantic错误；
21. 不要将 NOT_RUN写成 PASS；
22. 不要只更新文档不改实现；
23. 不要为了 backend coverage把 pandas delegate称为 native；
24. FINAL HEAD由独立 verifier agents重新验收；
25. evidence全部绑定 FINAL MERGED HEAD。

最终交付：

- FINAL HEAD SHA
- 完整 issue ledger
- fixed/open/partial/not-applicable数量
- operator direct-use ledger
- backend capability matrix
- CI workflow结果
- current-head evidence
- NOT_RUN / external infra blockers
- architecture authority map
- verifier machine-readable final gate matrix

直到 Definition of Done满足，任务不算完成。
