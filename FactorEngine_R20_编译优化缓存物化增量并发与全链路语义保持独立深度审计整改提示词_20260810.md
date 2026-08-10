# FactorEngine R20：编译、优化、CSE、缓存、物化、增量、并发、资源与全链路语义保持独立深度审计整改提示词

> **用途**：本文件直接作为新的代码整改 AI 执行提示词。  
> **独立性要求**：这是一个**全新的、独立的 R20**。不要与 R17 / R18 / R19 或任何更早文档合并。  
> **前提假设**：R17 的多市场 / Field / Provider / PIT / Session 问题、R18 的算子 DirectUse / Mining / 删除冗余问题、R19 的算子数学统计定义 / 参数绑定 / 后端数学 parity 问题，均视为正在由其他并行 AI 修复。本轮不重复这些问题本身。  
> **当前 GitHub 可见基线**：`quant_projects/main` 最新可见提交为 `48aec8d20825fd50086ace0fc06b26a5f7273e49`，提交信息：`Sync FE operator/param hardening, R17 audit prompt, and OPEN_WEEKLY report.`。如果服务器工作区更新，以**服务器真实 HEAD**为最终审计对象。  
> **工作方式**：直接修改服务器 / 本地代码，不要把“提交 GitHub / 开 PR / 合并 GitHub”当任务目标。GitHub 只是本轮审计参考。  
> **执行方式**：读完整份文档后，一次性完成所有整改、机器审计、回归测试、artifact 重生和验收。不要修几条就停，不要分优先级让我确认。  
> **动态原则**：不得硬编码当前 canonical 数量。必须动态 `load_all()` 后枚举 `OperatorRegistry.list_canonical()`。

---

# 0. R20 的核心主题

R19 已经下沉到 operator mathematical definition、parameter semantics、missing / Inf / tie / ddof、history topology 和 Pandas / Numba / Polars / DuckDB 数值一致性。R20 再往下一层：

```text
DSL / Expr
→ Analyzer / Typed IR
→ Logical Plan
→ Parameter validation
→ Composite lowering
→ Optimizer / Fastpath rewrite
→ Structural CSE
→ Rolling semantic CSE
→ Physical SQL lowering
→ Backend routing
→ Runtime / Parallel batch
→ Cache
→ Warmup / Incremental
→ Materialization
→ Catalog / Factor Identity / Checkpoint
→ Factor Lake downstream reuse
```

本轮要解决：一个原本定义正确的 factor，经过编译、rewrite、CSE、SQL partial materialization、缓存、并发、增量和落盘以后，会不会悄悄变成另一个 factor。

最终要求：

> **任何语义等价 rewrite 必须可证明等价；semantic attrs 必须跨所有 compiler/runtime boundary 完整保存；CSE/cache identity 必须与真正的 execution semantic identity 一致；增量/物化结果必须等于完整重算。**

---

# 1. 统一全链路语义载体

新增 `ExecutionSemanticEnvelope`，但不要把它做成新的平行真值源；它只是现有 typed semantic contract 的 compiler/runtime carrier。至少携带：

```text
market
universe_id
universe_membership_hash
frequency
input_grain
output_grain
calendar_id
timezone
timestamp_convention
available_at
decision_time_policy
same_session_usable

semantic_kind
unit
currency
price_basis
adjustment_basis
flow_semantics

source_contract_hash
source_dependency_hash
source_scope_hash
field_contract_hash

operator_semantic_hash
numeric_semantics_hash
math_semantics_hash
lowering_semantics_hash
optimizer_semantics_hash

history_contract_hash
missing_topology_policy
storage_precision_policy
```

IRNode、PlanNode、plan_ref、materialized_series、SQL subtree、CSE shared node、factor identity、cache namespace、materialization lineage只能引用/携带同一个规范语义 envelope 的 digest。

---

# 2. 当前代码已确认：Rolling CSE 会丢 `semantic_attrs`

## R20-001
普通 structural CSE 已经会给 `plan_ref` 复制 representative 的 `semantic_attrs`；但 `planner/rolling_cse.py` 当前新建 `plan_ref` 时没有 `semantic_attrs`。

**整改**：rolling CSE 创建 `plan_ref` 必须继承被共享子树的 output semantic contract。

## R20-002
rolling CSE 重建普通节点时也没有复制 `semantic_attrs`。

**整改**：所有 planner rewrite 统一通过 `rewrite_node()` / `PlanNodeRewriter`，默认保留 `semantic_attrs`、`node_id` 和 provenance。

## R20-003
`plan_ref` 不能只是 `sid`，还要能验证 `sid` 指向节点的 output semantic digest 与引用节点声明一致。

## R20-004
新增 release invariant：

```text
SEMANTIC_ATTRS_LOST_BY_ROLLING_CSE == 0
```

自动比较 CSE 前、structural CSE 后、rolling CSE 后的 output semantic digest。

---

# 3. Rolling CSE 的语义等价 key 本身不安全

## R20-005
当前 rolling semantic key 主要根据 canonical、第一/第二个底层 column、window/attrs 识别共享，但 `_first_col_ref()` 只是 DFS 找第一个列节点。

这会让以下不同子树过度接近：

```text
ts_mean(close,20)
ts_mean(log(close),20)
ts_mean(close / ts_mean(close,5),20)
```

它们都可能被抽成 `first_col_ref=close`。

**整改**：key 必须包含每个 panel input subtree 的完整 structural semantic key。

## R20-006
二元 rolling 也不能只看两边“第一个 column”。例如：

```text
ts_corr(log(close), volume, 20)
ts_corr(close, volume, 20)
```

不能共享。

## R20-007
建议 key：

```python
{
  "canonical": canonical,
  "input_semantic_keys": [structural_key(x) for x in panel_inputs],
  "canonical_params": normalized_params,
  "output_semantic_digest": ...,
}
```

## R20-008
如果无法正式证明 rolling semantic CSE 比普通 structural CSE 多出来的等价规则是 sound，允许直接删除这一优化。性能不能优先于正确性。

## R20-009
新增 adversarial CSE corpus：

```text
same raw column + different transform
same field name + different source/provider
same input + different price basis
same input + different availability
same window + different min_periods
same window + different ddof
same window + different missing policy
```

全部不得误共享。

---

# 4. Rolling CSE / Rolling Cache 仍维护手工真值表

## R20-010
`ROLLING_OPS` 是手工集合。

## R20-011
`ROLLING_OP_CANONICAL` 是另一套手工 alias map。

## R20-012
`_window_from_attrs()` 又自己猜 `window/d/period/n/lag/periods`。

**整改**：统一从 registry canonical alias、OperatorMetadata、ParamSpec、history semantics 派生。

## R20-013
当前 `ROLLING_OPS` 甚至出现重复 `ts_std`。set 虽会去重，但说明这张表没有机器治理。最终：

```text
MANUAL_ROLLING_OPERATOR_LIST == 0
```

---

# 5. SQL Physical Lowering 也会丢 semantic attrs

## R20-014
`planner/sql_lowerer.py` 用裸 `PlanNode(op="materialized_series", attrs={"sid":...})` 替换 SQL subtree，没有 output semantic attrs。

## R20-015
递归重建非 SQL 节点时也只复制 op/inputs/attrs，丢 semantic_attrs/node_id。

## R20-016
因此 logical plan 语义正确，不代表 partial SQL 后 residual Python plan 仍有语义。

## R20-017
`materialized_series` 必须继承 extracted subtree 的 unit/grain/available_at/source identity/semantic kind/price basis。

## R20-018
新增：

```text
SEMANTIC_ATTRS_LOST_BY_SQL_LOWERING == 0
```

---

# 6. Composite lowering 本身没有完整 semantic propagation

## R20-019
`planner/lowerings/_helpers.py` 的 `literal/delay/ts_mean/ts_std/ts_min/ts_max/ts_delta/ts_ema/ts_pct/binop/unary/cum_sum/fillna_const/protected_div/safe_div/coalesce` 都创建裸 PlanNode。

## R20-020
`lower_composite_operators()` lowering 前会复制原 node semantics，但 lowering 返回新的 bare DAG 后不会恢复原 composite 的 output semantic contract。

## R20-021
不能简单把 composite semantics 复制给所有 primitive children。每个 primitive child 必须根据自己的 typed operator contract推导 semantics；lowered root再证明与 original composite output等价。

## R20-022
建立 `SemanticPlanBuilder`，production lowering 禁止直接裸 `PlanNode(...)` 改写已有节点。

## R20-023
AST static audit：planner/lowerings 内所有 bare PlanNode 创建点分类；真正 leaf creation可允许，rewrite必须 semantic-aware。

---

# 7. Composite lowering hash 不完整

## R20-024
当前 `_lowering_hash(fn)` 主要看 module、qualname、`co_code`。

## R20-025
Python 常数一般存在 `co_consts`。把 `100.0` 改成 `1.0` 可能改变语义但不改变 opcode骨架。

## R20-026
lowering hash至少覆盖 `co_consts/co_names/defaults/kwdefaults/closure/source AST/referenced helper semantic hashes`。

## R20-027
推荐 `LoweringSemanticIdentity = hash(canonical, normalized AST, lowering contract, helper dependency hashes, compiler version)`。

---

# 8. Fastpath rewrite 必须有严格等价证书

## R20-028
当前 z-score pattern rewrite 会把 `(x-ts_mean(x,w))/ts_std(x,w)` 重写到 `ts_zscore(x,w)`。

## R20-029
rewrite层携带的 min_periods/null_policy/nan_policy/includes_current_bar/ddof/zero_std_policy 与当前 direct `ts_zscore` metadata/实现并不完全一致。

## R20-030
generic divide 的 zero-denominator 语义也可能与 ts_zscore 的 zero-std 处理不同，常数窗口就是最直接的反例 fixture。

## R20-031
建立 `RewriteEquivalenceCertificate`：

```text
source pattern
target operator
required preconditions
parameter mapping
missing semantics
zero semantics
history equality
availability equality
unit equality
grain equality
```

## R20-032
没有完整 preconditions 的 rule 禁止 production rewrite。

## R20-033
所有 rewrite 做开/关 differential，比较值、NaN/Inf topology、index、grain、semantic digest。

## R20-034
log-return等 matcher还要有“规则可达性”正/负/boundary fixture，避免维护永远匹配不到的规则。

## R20-035
新增：

```text
UNREACHABLE_OPTIMIZER_REWRITE_RULES == 0
```

---

# 9. Optimizer 整体 differential

## R20-036
当前 pipeline 包含 literal fold、parameter validation、composite lowering、再次 fold、fastpath rewrite、parameter canonicalization。任何一步都可能变义。

## R20-037
建立 `OptimizerDifferentialAudit`：

```text
reference_plan = no semantic rewrite
optimized_plan = full optimizer
```

同数据执行比较。

## R20-038
比较不仅 allclose，还包括 NaN/Inf topology、index、columns、grain、availability、semantic attrs、history、source dependencies。

## R20-039
所有 composite parameter branch与边界值都覆盖。

---

# 10. PIT/Temporal proof 要覆盖优化后的计划

## R20-040
当前 compile主要在 Analyzer IR 后先做 PIT audit，再 Lowerer/Optimizer。

## R20-041
即使原始 IR PIT-safe，lowering/rewrite仍可能引入不同 availability/history/grain/same-session semantics。

## R20-042
建立 `RewriteTemporalProof`，证明：

```text
new dependencies are causal
available_at not earlier than allowed
no future shift
no session-close signal fed to intraday decision
history requirement not understated
```

## R20-043
production final plan必须 `POST_OPTIMIZATION_TEMPORAL_PROOF_PASS`。

---

# 11. Scoped-universe classifier 不能 fail-open

## R20-044
`_plan_has_cross_sectional_ops()` 遇 registry lookup异常时当前会继续 fallback。

## R20-045
production下 unresolved scope必须是 `UNKNOWN_SCOPE`，不是“不是横截面”。

## R20-046
market/grain/session/PIT/cross-section/source resolver内部异常统一 fail closed。

---

# 12. Data source config canonicalization 仍有两套规则

## R20-047
`storage/data_scope.py::_jsonable` 对 unsupported object直接 TypeError，是正确的 fail-closed方向。

## R20-048
engine CSE scope `_stable_config` 仍可以 `str(value)` 兜底。

## R20-049
所以 cache scope 与 CSE scope对同一个 source config可能得到不同 identity。

## R20-050
arbitrary object str可能带内存地址，或者把不同 config压成同字符串。

## R20-051
建立唯一 `canonicalize_execution_config()`，供 CSE/cache/lineage/identity/snapshot/checkpoint复用。

---

# 13. FactorExecutionScope / Cache Scope / FactorIdentity 必须来自同一 identity

## R20-052
当前至少有 `FactorExecutionScope`、`DataExecutionScope/ExecutionCacheNamespace`、`FactorSemanticIdentity` 三套结构。

## R20-053
允许保留不同 dataclass，但它们必须是同一个 `ExecutionSemanticIdentityV2` 的 projection，而不是独立推断。

## R20-054
强 invariant：

```text
same ExecutionSemanticIdentity
→ same CSE scope
→ same cache namespace
→ same checkpoint identity
→ same materialization factor_version
```

## R20-055
任何实际执行语义变化，至少一个 identity component必须改变。

---

# 14. FactorExecutionScope 默认 market 仍是 A

## R20-056
`FactorExecutionScope` dataclass 当前默认 `market="A"`，但 engine实际已经专门让 market未知时保持空串，避免 US被默认A。

## R20-057
默认改为 UNKNOWN/空，不允许 helper/manual DAG重新引入 A股默认污染。

---

# 15. CSE scope 未绑定 compile 后真实 secondary SourceRef dependency

## R20-058
`_scope_from_factor` source scope主要取 factor hint + anchor data source config。

## R20-059
它没有直接把 `source_dependency_hash(plan)` 纳入 FactorExecutionScope。

## R20-060
两个 factor同 anchor/freq/market/universe但使用不同 secondary source时仍可能同 CSE group；rolling CSE弱key进一步放大风险。

## R20-061
CSE scope增加 source_dependency_hash，最好直接由统一 ExecutionSemanticIdentity投影。

---

# 16. Scoped universe 不能只靠 instrument_filter 非空

## R20-062
当前非空 instrument_filter可以让 source被认为 scoped。

## R20-063
但 `factor.universe=CSI300` + 任意100只股票filter同样非空。

## R20-064
增加 `universe_membership_hash`，实际 resolved membership进入 execution scope/cache/CSE/factor identity/lineage。

## R20-065
命名 universe必须按 as-of date解析 membership，不能只字符串比较。

## R20-066
`*_ALL` whole-market判断必须同时验证 market family一致。

---

# 17. unresolved operator contract 不能进入 production CSE

## R20-067
plan hash遇 operator semantic contract lookup失败会产生 `semantic_version="unregistered"`。

## R20-068
persistent cache已经部分避免 unresolved写磁盘，但 CSE仍可能消费这种 deterministic key。

## R20-069
production CSE前强制 `ALL_OPERATOR_CONTRACTS_RESOLVED=True`。

## R20-070
registry bootstrap错误不能让所有节点都落入同一个“unregistered语义”。

---

# 18. field contract unavailable 不能成为公共 hash值

## R20-071
column field catalog hash失败时不能统一成 `"unavailable"` 后继续 production CSE/cache。

## R20-072
production必须 fail；research可以显式 unresolved且禁止 persistent reuse。

---

# 19. semantic attrs / factor identity 禁止 repr/default=str fallback

## R20-073
plan semantic digest里 unsupported object不能 `repr()`。

## R20-074
这会带 memory address / unstable ordering。

## R20-075
semantic attrs必须 typed JSON schema。

## R20-076
`FactorSemanticIdentity._stable_hash` 也不应继续 `default=str`。

## R20-077
当前字段虽然多为字符串，但 Identity V2扩展后必须提前收口。

## R20-078
统一 typed canonical identity serializer。

---

# 20. Typed IR identity 分层

## R20-079
`compute_ir_hash` 主要序列化 op/attrs/inputs，不完整覆盖 semantic_attrs。

## R20-080
需要独立 `TypedIRSemanticHash`。

## R20-081
至少清晰区分：

```text
SourceExpressionHash
TypedIRStructuralHash
TypedIRSemanticHash
LogicalPlanHash
OptimizedPlanHash
PhysicalPlanHash
ExecutionSemanticIdentity
```

## R20-082
不要再把所有东西都叫 ast_hash。

---

# 21. Lineage 当前的 lowered_plan_hash 并非 optimized execution plan

## R20-083
lineage_service 当前会从 analysis.ir重新 `Lowerer().to_logical_plan()` 后做 structural_key。

## R20-084
它没有记录 optimizer/composite/CSE/physical SQL后的真实计划，却容易被名字理解成实际执行plan。

## R20-085
lineage分别记录 typed_ir/logical/optimized/execution/physical hashes。

---

# 22. Lineage 必须记录真实 backend route

## R20-086
engine backend class叫 SQL/Polars，不代表实际每个 subtree真的 native执行。

## R20-087
materialization lineage写入 backend_path_summary、per-operator routes、fallback events、native/fallback counts。

---

# 23. PhysicalPlan contract太薄

## R20-088
当前 PhysicalNode主要是 kind/plan/sid/children。

## R20-089
缺 output semantic digest/native backend/fallback policy/source snapshot/input/output grain/availability。

## R20-090
增加 `PhysicalExecutionContract`。

---

# 24. malformed SourceRef 不得被 SQL detector吞

## R20-091
`_contains_source_ref()` 当前 catch所有异常。

## R20-092
prefix存在但 payload损坏不能当普通 column。

## R20-093
三态：NOT_SOURCE_REF / VALID_SOURCE_REF / MALFORMED_SOURCE_REF。第三种 production hard fail。

---

# 25. materialized_series 必须是 typed boundary

## R20-094
它不能只是 sid→Series。

## R20-095
下游必须知道 daily/minute/event/fiscal/session、available_at、unit、price basis等。

---

# 26. Downsample validation 不仅是“行数变少”

## R20-096
现有检查主要是 DatetimeIndex、unique、same columns、rows<=input。

## R20-097
这不能证明输出真的是合法日频。

## R20-098
检查 trading session映射、output label convention、calendar、timezone、declared grain、available_at。

## R20-099
rows fewer不能自动等价于合法 downsample。

---

# 27. Operator result normalization 的 Index分支统一

## R20-100
`_normalize_operator_result` 已有 `_target_index(template)`。

## R20-101
1D ndarray分支不能再直接使用 `template.index`，因为 template也可能本身就是 `pd.Index`。

## R20-102
所有分支统一使用 canonical target index。

---

# 28. Backend bridge double validation

## R20-103
bridge先 `validate_operator_call()`，SeriesOperator.calculate又做一次 validation。

## R20-104
如果 validator完全纯，只是浪费；如果 validate_params有状态/side effect，会执行两次。

## R20-105
审计全部 validate_params purity，最终一条调用只验证一次，或明确 pure/idempotent。

## R20-106
自定义 override calculate的 operator仍必须经过唯一 central gate。

---

# 29. Polars lazy scan 不是线程安全的 per-call配置

## R20-107
当前 lazy execute会修改 shared data source的 `_lazy_scan/read_auto`，甚至 inner对象。

## R20-108
run_many_parallel使用 threading，不同 worker local ctx仍共享同一个 engine data_source。

## R20-109
存在典型 restore interleaving race，最终 source可能残留错误模式。

## R20-110
禁止“修改共享source然后恢复”的设计。改 immutable read options、per-worker clone、process isolation或thread-local wrapper。

## R20-111
Polars restore exception当前被无条件吞掉。

## R20-112
production restore失败必须 abort/poison source，不能继续。

## R20-113
research至少 warning + runtime corrupted-state marker。

---

# 30. ExecutionContext 仍把 execution-local wrapper挂到 shared inner source

## R20-114
ExecutionContext会 `setattr(inner, "_factor_engine_lqtp_wrapper", wrapper)`。

## R20-115
并发context会互相覆盖这个 pointer。

## R20-116
lineage_service又可能从这个 pointer寻找 logical wrapper，因此某execution lineage可能读到另一个 execution wrapper。

## R20-117
execution-local wrapper只存在ctx，不挂shared inner。

## R20-118
增加 N线程不同 SourceRef并发 lineage隔离测试。

---

# 31. Global MemoryGovernor / Cache 并发合同

## R20-119
MemoryGovernor `_usage/evictions/throttles` 为共享 mutable state。

## R20-120
run_many_parallel(threading)下必须同步。

## R20-121
reserve/release/accounting/evict/release_all原子化。

## R20-122
CacheManager with_scope可以共享 backing store。

## R20-123
并发 get/set/evict/clear必须锁。

## R20-124
dict mutation、byte counter、governor accounting必须同一个原子 critical section。

---

# 32. `check_admit()` 存在 ghost accounting

## R20-125
当前超预算时会 `reserve("__probe__", size)`。

## R20-126
reserve成功后真实增加 usage，但只是 admission check，没有实际存对象。

## R20-127
实现无副作用 `can_admit()`，必要时evict但不得留下 `__probe__` accounting。

---

# 33. warmup pressure stage逻辑冲突

## R20-128
`check_pre_warmup()` 当前允许 `("normal","stop_warmup")`。

## R20-129
文档却说 stop_warmup及以上不允许。

## R20-130
修 stage/action语义，新增 threshold boundary tests。

## R20-131
每个 threshold 做 `-epsilon/exact/+epsilon` 行为验证。

---

# 34. Parallel run resource contract 当前可 fail-open

## R20-132
run_many_parallel构建 resource plan/scope的 try异常可以直接 `resource_scope=None`。

## R20-133
production这意味着资源治理坏了反而无约束继续跑。

## R20-134
production resource plan resolution failure必须 hard fail。

---

# 35. ResourceScope strictness依赖 ambient env

## R20-135
apply/restore PRAGMA错误当前看 `FACTOR_ENGINE_RUN_MODE` env。

## R20-136
FactorEngine显式 run_mode=production并不要求同步这个env。

## R20-137
ExecutionResourceScope必须显式接收真实 run_mode/strict，不再从环境猜。

---

# 36. CPU requested slots 未 clamp hard quota

## R20-138
memory limit逻辑会 `min(requested, hard_limit)`。

## R20-139
CPU slots却 explicit/env优先直接return。

## R20-140
重构为 `hard_cpu_limit` + requested clamp，容器2 CPU / explicit64最终不能>2。

---

# 37. Process-global resource env并发安全

## R20-141
ExecutionResourceScope修改 DUCKDB/POLARS等 process-global env。

## R20-142
两个并发engine request enter/exit会错误恢复 previous值。

## R20-143
使用 process-level lock、per-worker process或connection-local settings。

## R20-144
Polars import后 POLARS_MAX_THREADS不保证生效。

## R20-145
telemetry区分 requested threads与effective threads，不可虚假标已生效。

---

# 38. LRU overwrite不是MRU

## R20-146
普通 dict对已有key重新赋值不会改变插入位置。

## R20-147
cache eviction却用第一项作为LRU。

## R20-148
overwrite先pop再insert，或统一 OrderedDict.move_to_end。

## R20-149
PanelCache已有OrderedDict，也要在existing set后move_to_end。

---

# 39. Cache accounting强 invariant

## R20-150
`cache._bytes == sum(actual resident sizes)`。

## R20-151
`governor usage[layer] == sum(all registered resident stores)`。

## R20-152
多线程几千次 set/get/overwrite/evict/clear/scope stress后不得漂移/负数。

---

# 40. Persistent cache schema与空轴

## R20-153
save写了 schema_hash，但load没有真正验证actual schema。

## R20-154
读parquet后重新计算并比较。

## R20-155
schema mismatch quarantine/cache miss，并有telemetry。

## R20-156
schema hash加入 index levels/names/dtypes/timezone/categorical/column order。

## R20-157
缓存格式本身加 schema_version。

## R20-158
空 Series round-trip当前会返回普通 empty Series。

## R20-159
原 MultiIndex轴语义丢失。

## R20-160
根据meta重建empty index schema。

---

# 41. Persistent cache并发generation publication

## R20-161
单writer interrupt可checksum fail-closed，但多writer同key可能交错生成payload/meta。

## R20-162
这会造成持续miss或generation不一致。

## R20-163
使用per-key lock或generation directory + atomic pointer。

---

# 42. Persistent cache namespace必须覆盖 compiler semantics

## R20-164
namespace已有operator+field hash及scoped key，这是基础。

## R20-165
继续验证 numeric semantics/lowering semantics/optimizer compiler semantic version最终进入persistent identity。

---

# 43. Structural NaN literal 与 Plan Hash 直接冲突

## R20-166
lowering helper明确允许 `literal(float("nan"))`。

## R20-167
earnings_yield/book_to_price会构造 `where(cond,ratio,NaN)`。

## R20-168
plan hash却明确拒绝 NaN/Inf literal。

## R20-169
因此合法 lowered factor可能“不开cache能跑，一做CSE/hash就失败”。

**整改**：Plan-level missing使用 typed NULL，不再用 IEEE NaN literal。

## R20-170
建立 LiteralKind：NUMBER / NULL / BOOL / STRING。

## R20-171
missing branch是NULL。

## R20-172
backend分别映射 Pandas NaN / Polars null / SQL NULL。

## R20-173
Plan hash对typed NULL稳定编码。

## R20-174
constant folding必须识别typed null。

## R20-175
不能对所有literal无脑 `float(value)`。

---

# 44. Scalar parameter default不能用 truthiness

## R20-176
MACD lowering存在类似 `_literal_input(...) or default`。

## R20-177
显式0会被解释为“没提供”并换成default。

## R20-178
即使上层最终会reject，lowering也不能把invalid explicit与missing混淆。

## R20-179
全库扫描 `param or default`，对合法0/False/空串用 MISSING/None判断。

---

# 45. Composite contract panel/scalar arity混杂

## R20-180
当前 deps + min_inputs需要probe层自己猜panel数量。

## R20-181
这不是稳定的typed contract。

## R20-182
改成 panel_arity/scalar_params/context_inputs/branch_params/optional_inputs。

## R20-183
production composite全部explicit，无 signature/name heuristic fallback。

---

# 46. Composite依赖声明与实际代码读取一致性

## R20-184
例如 MACD_line contract包含 signal依赖，但 line本身不消费 signal。

## R20-185
自动AST审计 declared deps == actual dependency reads。

## R20-186
新增：

```text
UNUSED_LOWERING_DECLARED_PARAMS == 0
UNDECLARED_LOWERING_PARAM_READS == 0
```

---

# 47. Composite lowering必须 total

## R20-187
很多 lowering遇 invalid params/input会 `return node`。

## R20-188
如果metadata有缺口，同 canonical就可能一部分走lowered primitives、一部分走direct implementation。

## R20-189
production：legal domain始终lower；illegal domain parameter error；不能silent no-op lowering。

---

# 48. Branch certification名称与实际覆盖一致

## R20-190
当前所谓 certified_for_all_branches 实质是有限representative grid。

## R20-191
改成 certified_for_declared_branch_coverage。

## R20-192
再用 AST path coverage + boundaries + relational constraints证明代码分支覆盖。

---

# 49. Direct-vs-lowered differential

## R20-193
如果 composite同时有direct implementation，全部legal params执行 direct vs lowered。

## R20-194
不等价就只能选一个production authority。

---

# 50. A股 limit-state lowering的 tolerance单位必须明确

## R20-195
当前生成 `threshold = limit_price - tolerance`，默认值形似0.005。

## R20-196
必须从ParamSpec/semantic unit确认它是 absolute price tolerance还是relative ratio。

## R20-197
如果是ratio，应使用比例形式；如果是price，unit明确为currency。

## R20-198
metadata没有tolerance unit就是production contract incomplete。

## R20-199
所有 generated plan lowering后重新做 unit/grain/availability checker。

## R20-200
不能因为原 composite typed-safe就假设generated primitive DAG自动safe。


# 51. Materialization precision 是 factor definition 的一部分

## R20-201
Parquet materializer 当前默认 `value_dtype="float32"`，live engine多数计算是float64。

## R20-202
如果 materialized factor会再次作为下游输入，float32版本并不严格等于live版本。

## R20-203
高风险场景：

```text
small regression residual
near-zero alpha
near-threshold condition
rank/tie
recursive state
long compounding
```

## R20-204
storage precision policy进入 FactorSemanticIdentity/factor_version/lineage。

## R20-205
production默认建议float64，除非该factor有明确float32 quantization certificate。

## R20-206
做 `live → materialize → reload` differential，记录 max abs/rel error、rank changes、sign flips、threshold flips。

---

# 52. resolved snapshot 与 row metadata必须一致

## R20-207
materializer已能从显式data_snapshot_id或lineage extra解析 `source_snapshot`。

## R20-208
但 row-level MaterializeMetadata不能仍只写显式参数。

## R20-209
统一使用 resolved source_snapshot，保证 rows/catalog/checkpoint看到同一snapshot identity。

---

# 53. Identity computation exception不能吞

## R20-210
materializer fallback identity路径当前 generic exception可降级成 checkpoint_identity=None。

## R20-211
production下“identity不可合法构造”和“identity代码抛异常”必须区分。

## R20-212
定义 `IdentityUnavailable` 与 `IdentityComputationFailed`；后者production hard fail。

---

# 54. FactorSemanticIdentity V2

## R20-213
增加：

```text
numeric_semantics_hash
mathematical_semantics_hash
composite_lowering_hash
optimizer_rewrite_hash
physical_lowering_hash
history_contract_hash
storage_precision_policy
```

## R20-214
optimizer/lowering规则改变后旧cache/checkpoint/evidence必须失效。

## R20-215
identity不能只看root price_basis。

## R20-216
最终dimensionless factor仍可能依赖raw/adjusted/multi-currency inputs。

## R20-217
加入整个typed IR的 `input_semantic_dependency_digest`。

## R20-218
full factor definition保存完整 ExecutionSemanticIdentity digest。

## R20-219
restore/rebuild不能只凭same expression/ast hash判为同factor。

---

# 55. Catalog strict mode不能fail-open

## R20-220
catalog `_resolve_strict()` 如果 production-mode resolver异常，不能返回False。

## R20-221
production critical catalog decode显式 `strict=True`。

## R20-222
unexpected run-mode resolution error直接fail closed。

---

# 56. FactorCatalog线程模型

## R20-223
SQLite connection默认 `check_same_thread=True`。

## R20-224
如果一个ParquetMaterializer/FactorCatalog被worker线程共享，会抛SQLite thread错误。

## R20-225
不要只是设置 `check_same_thread=False`。使用per-thread connection、writer queue或明确lock。

---

# 57. 物化事务与恢复

## R20-226
测试两个factor并发写不同partition。

## R20-227
测试同factor并发写同partition。

## R20-228
physical parquet成功/catalog commit失败。

## R20-229
catalog成功/publish失败。

所有情况要有明确 journal/recovery state。

---

# 58. all-NaN recompute不能留下旧有限值

## R20-230
research/plain materialize存在all-NaN且未force tombstone时skip write的兼容行为。

## R20-231
如果这是已有窗口的重新计算，旧finite会残留。

## R20-232
materialize API显式区分 `append/upsert/replace_window/recompute_window`。

## R20-233
recompute_window中 new NaN必须覆盖 old finite。

---

# 59. Incremental correctness唯一标准：等于 full recompute

## R20-234
对所有 incremental-safe/checkpoint-safe/bounded-history operator建立：

```text
full_result
incremental_result
checkpoint_restore_result
```

三路 differential。

## R20-235
覆盖 append、late correction、delete、restatement、universe exit、split/dividend、new listing、suspension、source missing recovery。

## R20-236
SourceRef历史修订必须反向传播到所有依赖factor，不只处理新增rows。

---

# 60. Batch warmup不能改变 factor的semantic history anchor

## R20-237
run_many共享union load window只是I/O优化。

## R20-238
expanding/recursive factor的history anchor仍是自身declared anchor，不能被同batch其它factor拉早后改变结果。

## R20-239
warmup/history resolver异常production不能默认为short/no warmup。

## R20-240
任何 unresolved history requirement production fail closed。

---

# 61. Parallel worker只是部分隔离

## R20-241
worker local ctx会复制runtime/shared dict/panel cache，这是正确方向。

## R20-242
但仍共享 backend instance、data_source、global governor、process environment、registries。

## R20-243
每个共享对象标注 immutable/locked/thread-local/process-local contract。

## R20-244
backend route telemetry必须root-local。

## R20-245
不能让共享 `last_operator_backend_route` 覆盖其它factor telemetry。

---

# 62. Production fallback要在执行前阻断

## R20-246
fully_sql计划实际full SQL失败后，当前可能继续partial/python fallback。

## R20-247
如果production fallback policy=error，应在fallback执行前失败。

## R20-248
不能先把未授权路线算完，最后runtime gate再说“不允许”。

## R20-249
fallback reason绑定具体 sid/canonical/exception/dialect/query，不从共享last reason推断。

## R20-250
每个subtree structured fallback event。

## R20-251
Polars Expr fallback同样结构化。

## R20-252
新增：

```text
NATIVE_CERTIFIED_BUT_FALLBACK_EXECUTED == 0
```

---

# 63. SourceRef identity要绑定“实际resolved依赖”

## R20-253
runtime logical source可能根据market/provider/coverage选择真正source。

## R20-254
记录 syntactic requested dependency + runtime resolved dependency。

## R20-255
最终execution identity优先resolved dependency manifest。

## R20-256
lineage不再从shared inner source的“最后一个wrapper”推断本execution依赖。

---

# 64. Snapshot identity层次化

## R20-257
snapshot至少包括：

```text
anchor snapshot
secondary source snapshots
provider versions
field catalog
universe membership snapshot
```

## R20-258
随机抽查row metadata、catalog、run lineage、checkpoint全部一致。

## R20-259
full 256-bit identity digest持久化；16 hex只作为展示/紧凑版本。

---

# 65. Rewrite provenance

## R20-260
planner rewrite重建node时不能丢node_id/provenance。

## R20-261
记录 source_node_ids、rewrite_rule_id、lowering_source_canonical。

## R20-262
planner AST audit所有 bare PlanNode rewrite。

## R20-263
真正创建source/literal leaf可以裸建，其余必须semantic-aware。

---

# 66. PlanNode文档与真实hash语义统一

## R20-264
文档中“semantic_attrs不参与hash”如果指dataclass equality，要明确。

## R20-265
FactorEngine structural semantic key实际上必须包含semantic digest，避免未来开发误删。

---

# 67. Plan serialization round-trip

## R20-266
任何plan JSON/debug/checkpoint serialization都验证：

```text
serialize
deserialize
structural_key unchanged
semantic digest unchanged
```

---

# 68. Credentials redaction与真实dataset identity分离

## R20-267
secret不应进入factor identity，这是正确的。

## R20-268
但不同credential若看到不同entitlement，必须由resolved dataset/provider snapshot区分，而不是credential hash。

## R20-269
`hash_data_source_config`与`compute_data_scope`共享同一个canonicalizer。

---

# 69. Ephemeral data scope禁止持久化

## R20-270
无法稳定描述source时的 `ephemeral:{id(data_source)}` 只允许same-process memory cache。

## R20-271
禁止进入persistent cache/factor identity/materialization catalog/checkpoint。

## R20-272
persistent path发现ephemeral identity直接fail。

---

# 70. Unknown / N/A / Explicit scope值分开

## R20-273
空字符串不能同时代表unknown、not applicable、default。

## R20-274
使用typed `UNKNOWN / NOT_APPLICABLE / EXPLICIT(value)`。

---

# 71. Physical SQL identity绑定dialect/compiler

## R20-275
DuckDB与ClickHouse对null/date/quantile等可能有不同实现。

## R20-276
physical execution SID加入dialect semantic version。

## R20-277
SQL emitter/compiler semantic version变化使旧physical cache失效。

---

# 72. SQL/Python/Polars canonical ordering

## R20-278
fully Python、partial SQL、fully SQL必须输出同一timestamp/instrument canonical ordering。

## R20-279
不能依赖SQL engine默认row order。

## R20-280
cache/materialization reload也保持canonical ordering。

---

# 73. Materialization upsert deterministic tie policy

## R20-281
同一个 `(datetime, asset)` 多条记录不能只靠concat当前顺序决定last。

## R20-282
定义 generation_id/calc_time/sequence的确定性优先级。

## R20-283
同generation同时出现valid和tombstone应视为upstream invariant violation。

---

# 74. value dtype cast前后双DQ

## R20-284
float64 finite cast float32后可能overflow/underflow。

## R20-285
DQ增加post-cast检查和quantization report。

## R20-286
统计float32新制造的rank ties/sign flips/threshold flips。

---

# 75. 下游读取materialized factor必须校验版本

## R20-287
不能只按factor_id拿“当前文件”。

## R20-288
consumer可声明 expected semantic identity/version；不匹配fail。

---

# 76. Retired factor代际隔离

## R20-289
integration test：

```text
delete factor
recreate same id without rebuild -> reject
explicit rebuild -> no old partitions mixed
```

---

# 77. Catalog + physical publish journal

## R20-290
显式状态：

```text
prepared
physical_written
catalog_committed
published
watermark_advanced
```

## R20-291
多target写入只有全部required target成功才推进watermark。

---

# 78. Factor dependency不能只存column name

## R20-292
同名Close可能属于不同source/market/basis。

## R20-293
dependency使用field_id/provider/source semantic identity。

## R20-294
event incremental impact analysis不能只依赖referenced_columns。

---

# 79. Snapshot变化必须使cache失效

## R20-295
long-running process里source data更新，如果source scope不变，旧persistent result不能继续命中。

## R20-296
真正data_snapshot_id/generation进入execution namespace。

## R20-297
source object config不等于source snapshot。

## R20-298
无snapshot保证时，cache lifetime最多one execution。

---

# 80. process cached catalog hash热重载

## R20-299
类似 `_field_catalog_version()` 的 lru_cache如果支持hot reload必须主动失效。

## R20-300
若production不支持hot reload，明确freeze registry/catalog。

## R20-301
测试registry/catalog reload不会让旧hash继续使用。

---

# 81. Context version字段必须真正接线

## R20-302
ExecutionContext.registry_version要确认被赋真实版本且进入相关lineage/cache；否则删掉dead field。

## R20-303
evidence_version同理，必须与backend router实际消费的evidence generation一致。

---

# 82. Semantic identity 与 execution provenance分开

## R20-304
backend route通常不应改变factor semantic identity，前提backend被证明数学等价。

## R20-305
具体backend、fallback、线程数、query count属于RunExecutionProvenance。

## R20-306
如果两个backend只能tolerance-equivalent而不是declared semantic-equivalent，要明确equivalence class。

---

# 83. Execution CSE不能使用mining equivalence

## R20-307
factor semantic/mining dedup与exact execution reuse是不同层。

## R20-308
rank-equivalent/monotonic-equivalent只能用于mining去重，绝不能用于runtime CSE/cache。

## R20-309
plan/cache float canonicalization只有ParamSpec明确declared equivalence时才能合并。

---

# 84. Timestamp convention进入scope

## R20-310
date/datetime/UTC ns/local date不是同一execution contract。

## R20-311
timezone normalization必须发生在cache identity之前。

## R20-312
materialized factor持久化timestamp convention。

---

# 85. `available_at` 必须进入factor lake

## R20-313
daily factor由完整日内数据计算后，只有date字段会让下游误以为当日开盘就可用。

## R20-314
持久化 `available_at=session_close/next_open/report_publish_time/...`。

## R20-315
下游DataAccess读取factor时mechanically enforce available_at。

## R20-316
factor lake自身作为一等Provider，具备PIT/coverage/snapshot/unit/grain/availability/identity。

## R20-317
available_at semantic digest进入cache identity。

---

# 86. Shared CSE DAG引用完整性

## R20-318
every plan_ref sid exists。

## R20-319
every shared sid consumer count准确。

## R20-320
no orphan shared node / no premature release。

## R20-321
shared A依赖shared B时，B生命周期包含A materialization。

## R20-322
refcount不能只扫描roots而忽略shared-to-shared依赖。

## R20-323
shared nodes materialization必须拓扑排序，不依赖dict insertion order。

## R20-324
lazy shared compile失败要rollback半生成state。

---

# 87. Runtime telemetry merge

## R20-325
worker local runtime events结构化merge回batch。

## R20-326
禁止last-write-wins覆盖其它factor telemetry。

---

# 88. Cache layer按生命周期分类

## R20-327
execution-local cache需要execution isolation。

## R20-328
persistent semantic cache不能带随机execution_id。

## R20-329
明确三类：

```text
execution-local
snapshot-local
semantic-persistent
```

不同层使用不同identity schema。

---

# 89. Panel cache correctness + performance

## R20-330
完整content hash是O(N)，要有hash_time telemetry。

## R20-331
记录panel_cache_hash_time/unstack_time/hit_rate。

## R20-332
若需要优化，可用immutable source token，但不能退回id(series)之类不安全key。

## R20-333
object dtype hash使用repr仍可能不稳定。

## R20-334
production panel values/axis限制可接受dtype，unsupported object fail。

---

# 90. Instrument ordering

## R20-335
明确canonical instrument ordering或source-order preservation policy。

## R20-336
cross-sectional factor经过CSE/cache/materialize reload仍应满足column permutation equivariance。

---

# 91. SQL snapshot isolation

## R20-337
一个factor/batch内多SQL query不能看到不同source snapshot。

## R20-338
production batch使用immutable snapshot或transaction/snapshot isolation。

## R20-339
partial SQL多个subtree与Python fallback读取同一数据世代。

---

# 92. Resource settings execution-scoped

## R20-340
DuckDB threads/memory/temp不能跨run泄漏。

## R20-341
spill path按execution隔离。

## R20-342
spill max不能超过实际可用disk还声称安全。

## R20-343
spill cleanup失败有telemetry/quarantine。

## R20-344
memory budget包含SQL materialized、shared_result_cache、worker results、final result，不只cache。

## R20-345
parallel `result_policy=return`检查总batch result budget。

## R20-346
sink callback线程模型明确，不偷偷在多个worker并发调用非thread-safe sink。

## R20-347
parallel结果输出顺序deterministic。

## R20-348
serial vs parallel result和semantic lineage等价，只有timing/run_id/scheduling telemetry可不同。

---

# 93. Global/random/environment determinism

## R20-349
production DAG禁止stochastic side-effect operator。

## R20-350
扫描global numpy/random state。

## R20-351
identity跨process/locale/machine稳定。

## R20-352
library版本变化通过pin semantic policy处理，而不是依赖library默认。

## R20-353
supported dependency versions跑golden differential。

---

# 94. Lowering读取canonical bound，不再猜literal位置

## R20-354
private `_literal_input`等不作为production semantic authority。

## R20-355
lowering接收canonical `BoundPlanParameters`。

## R20-356
最终canonical plan最好统一：panel inputs在children、scalar params在attrs。

## R20-357
若保留positional scalar literal，必须有typed slot metadata。

## R20-358
parameter normalization发生在CSE前。

## R20-359
CSE只消费normalized plan。

## R20-360
SQL lowerer同样只消费normalized plan。

---

# 95. Final plan禁止legacy spelling

## R20-361
所有op canonical、所有param canonical。

## R20-362
invariants：

```text
LEGACY_OPERATOR_ALIAS_IN_FINAL_PLAN == 0
LEGACY_PARAM_ALIAS_IN_FINAL_PLAN == 0
```

---

# 96. Compiled plan schema/version

## R20-363
保存/reuse的plan有 `plan_schema_version/compiler_semantics_version`。

## R20-364
precompiled plan compiler版本不兼容则recompile/reject。

## R20-365
AnalysisResult与plan绑定，不允许plan A + analysis B。

## R20-366
保存 `analysis.typed_ir_hash ↔ optimized_plan.source_ir_hash` binding digest。

---

# 97. Source dependencies post-optimizer audit

## R20-367
optimizer后重算source dependency manifest并与前层diff。

## R20-368
generated plan referenced columns也要重新审。

## R20-369
optimizer/lowering不能创建裸column(name)绕过field/source identity。

## R20-370
`__probe_*` synthetic column只能在audit。

## R20-371
probe structural evidence不能冒充真实source certification。

## R20-372
artifact每条evidence标 static/synthetic/real-sample/production。

## R20-373
real-sample evidence绑定snapshot。

---

# 98. Evidence invalidation

## R20-374
operator/lowering/numeric semantics/field/provider/backend emitter任一hash改变，相关evidence自动stale。

## R20-375
manifest generator先验证dependency hashes，不因为旧evidence文件存在就通过。

---

# 99. Artifact generation必须同一代

## R20-376
operator catalog/evidence/mining manifest不能各自由不同HEAD生成。

## R20-377
generation manifest记录 generation_id、all component hashes、created_at、source HEAD/source semantic hash。

## R20-378
runtime production启动验证artifact generation一致。

## R20-379
多AI并行改代码场景尤其禁止mixed generation。

## R20-380
artifact build期间检测source变化，变化则abort/restart。

---

# 100. Registry/load/reset determinism

## R20-381
随机允许的module import order后，canonical registry/aliases/metadata/lowerings/semantics一致。

## R20-382
duplicate replacement不依赖import order。

## R20-383
`_CLEANED_LOADED/_LOWERINGS_LOADED/_DEDUPE_APPLIED`等mutable loaded flag有统一reset。

## R20-384
reset不能半清registry导致test-order dependency。

## R20-385
pytest随机顺序至少两轮。

## R20-386
subprocess clean-start hash完全一致。

## R20-387
fork/spawn future compatibility审计。

---

# 101. Persistent identity稳定性

## R20-388
所有persistent artifacts扫描memory address模式：`0x...`, `<object at`, `<function ... at`。

## R20-389
semantic identity不含current timestamp。

## R20-390
absolute local path只有真正改变数据identity时才进入semantic identity。

## R20-391
logical dataset snapshot与physical mount path分离。

## R20-392
factor lake换机器/mount后semantic factor identity仍可识别。

---

# 102. Cache corruption可观测与自愈

## R20-393
checksum/schema失败记录 persistent_cache_corruption_count。

## R20-394
坏文件quarantine/delete，不能每次都重新读大parquet失败。

## R20-395
cache metadata schema version显式。

## R20-396
factor lake schema version显式。

---

# 103. Stateful checkpoint compatibility

## R20-397
checkpoint schema version绑定operator checkpoint semantic hash。

## R20-398
restore验证state shape/dtype/version。

## R20-399
incremental partition fingerprint包含storage precision。

## R20-400
partition fingerprint包含output availability policy。

---

# 104. Universe/corporate-action invalidation

## R20-401
股票退出universe触发旧值tombstone。

## R20-402
universe membership本身PIT/snapshot化，不用今天成分重算历史。

## R20-403
adjustment factor历史revision触发所有adjusted-price依赖重算。

## R20-404
修改t时，影响范围按DAG history反向传播，例如window20影响t...t+19。

## R20-405
recursive operator revision通常影响revision point之后全部结果。

## R20-406
checkpoint必须选择earliest affected point之前的clean checkpoint。

## R20-407
late correction测试覆盖composite + stateful。

---

# 105. Factor-to-factor dependencies

## R20-408
materialized factor作为SourceRef也参与invalidations。

## R20-409
factor dependency graph cycle detection。

## R20-410
alias/self-reference cycle也检测。

## R20-411
dependency graph identity使用canonical factor semantic identity，不只是factor_id。

## R20-412
rebuild pin upstream factor_version。

---

# 106. Code provenance

## R20-413
git commit属于provenance，不直接作为semantic identity。

## R20-414
取不到git commit显式UNKNOWN。

## R20-415
服务器常有未上传工作区修改，建议记录 `source_tree_semantic_hash` / dirty state。

## R20-416
最终报告输出 server HEAD、working-tree dirty、source semantic hash，不假设GitHub HEAD就是真实执行版本。

---

# 107. 新建全链路 semantic preservation audit

## R20-417
创建：

```text
factor_engine/scripts/audit_execution_semantic_preservation.py
```

## R20-418
逐层抓：

```text
IR
logical
lowered
optimized
structural CSE
rolling CSE
physical SQL
runtime
materialized reload
```

每层记录 structural hash、semantic digest、source dependency hash、history hash、availability、grain、unit、price basis、universe hash。

## R20-419
语义等价步骤要求digest invariant。

## R20-420
合法semantic transform必须有declared transform certificate。

---

# 108. Compiler differential corpus

## R20-421
至少覆盖：

```text
pure elementwise
rolling
cross-sectional
group neutralize
fundamental PIT
technical composite
A-share event
intraday->daily
stateful recursive
SourceRef
factor-lake input
multi-source
```

## R20-422
每类factor跑：

```text
single
run_many no CSE
structural CSE
rolling CSE
parallel
Pandas
Polars
Polars lazy
SQL full
SQL partial
persistent cache cold
persistent cache warm
materialize reload
incremental
```

## R20-423
生成统一execution path matrix。

---

# 109. R20 compiler/runtime blocker体系

## R20-424
至少增加：

```text
C01_SEMANTIC_ATTRS_LOST
C02_UNSAFE_CSE_EQUIVALENCE
C03_UNRESOLVED_OPERATOR_CONTRACT
C04_UNSTABLE_EXECUTION_SCOPE
C05_SOURCE_DEPENDENCY_SCOPE_MISSING
C06_UNIVERSE_MEMBERSHIP_MISMATCH
C07_REWRITE_EQUIVALENCE_UNPROVEN
C08_POST_OPT_TEMPORAL_PROOF_MISSING
C09_COMPOSITE_SEMANTIC_LOSS
C10_COMPOSITE_DEPENDENCY_DRIFT
C11_MALFORMED_SOURCE_REF_SWALLOWED
C12_PHYSICAL_PLAN_SEMANTIC_LOSS
C13_NATIVE_FALLBACK_POLICY_VIOLATION
C14_SHARED_DATASOURCE_RACE
C15_RESOURCE_SCOPE_FAIL_OPEN
C16_MEMORY_ACCOUNTING_DRIFT
C17_ADMISSION_PROBE_ACCOUNTING_LEAK
C18_CACHE_SCHEMA_MISMATCH
C19_CACHE_EMPTY_AXIS_LOSS
C20_NONATOMIC_CACHE_GENERATION
C21_STORAGE_PRECISION_IDENTITY_MISSING
C22_MATERIALIZE_IDENTITY_FAIL_OPEN
C23_STALE_VALUE_NOT_TOMBSTONED
C24_INCREMENTAL_FULL_RECOMPUTE_MISMATCH
C25_ARTIFACT_GENERATION_MIXED
C26_PERSISTENT_IDENTITY_UNSTABLE
C27_TYPED_LITERAL_CONTRACT_CONFLICT
C28_LOWERING_HASH_INCOMPLETE
C29_LEGACY_ALIAS_IN_FINAL_PLAN
C30_PLAN_ANALYSIS_BINDING_MISMATCH
```

---

# 110. Rewrite registry与CSE equivalence policy

## R20-425
所有 optimizer rewrite/composite lowering/physical rewrite登记 rule_id、source pattern、target、transform type、proof tests、version/hash。

## R20-426
没证书的rewrite不进production。

## R20-427
CSE equivalence分：

```text
STRUCTURAL_EXACT
DECLARED_EXECUTION_EQUIVALENT
MINING_EQUIVALENT
RANK_EQUIVALENT
```

## R20-428
runtime CSE只允许前两类。

## R20-429
mining semantic dedup永不流入runtime CSE。

---

# 111. Final-plan validator

## R20-430
production execute前验证：

```text
all ops canonical
all params canonical
all operator contracts resolved
all semantic contracts resolved
all SourceRefs valid
all plan_refs bound
all materialized_refs typed
all history contracts resolved
all temporal proofs valid
no legacy aliases
no research-only nodes
```

## R20-431
precompiled plan走同一个validator。

## R20-432
SQL physical root走physical final validator。

---

# 112. Execution identity finalization

## R20-433
materialization前，在真实backend route发生以后finalize execution provenance。

## R20-434
FactorSemanticIdentity描述factor是什么；RunExecutionProvenance描述这次怎么计算。

## R20-435
同factor不同backend可共享semantic identity，前提backend被证明semantic equivalent。

## R20-436
backend不等价时必须成为不同semantic variant，不能硬归为同factor。

---

# 113. 全库 exception swallowing audit

## R20-437
扫描：

```text
except Exception: pass
except Exception: return default
except Exception: return False
except Exception: return None
```

## R20-438
分类 cosmetic / performance fallback / research fallback / semantic-security-production gate。

## R20-439
semantic/security/production gate禁止silent swallow。

## R20-440
best-effort函数不能决定cache identity、market、universe、PIT、resource、dependency。

---

# 114. Ambient environment dependency audit

## R20-441
扫描所有 `os.environ/getenv`，凡改变数学/执行/安全语义，必须进入显式context。

## R20-442
production run_mode统一以ExecutionContext/engine run_mode为准。

---

# 115. Global singleton audit

## R20-443
registry/governor/data store/catalog/backend state明确concurrency/reset contract。

## R20-444
operator execution禁止hidden mutable global state。

---

# 116. Determinism矩阵

## R20-445
同code/snapshot/factor/scope连续执行10次，value hash、NaN mask、index、semantic identity一致。

## R20-446
serial vs parallel。

## R20-447
cold cache vs warm cache。

## R20-448
CSE off vs on。

## R20-449
SQL off vs on。

## R20-450
materialize reload（按declared storage tolerance）。

## R20-451
incremental vs full。

## R20-452
production evidence必须包含这些actual-engine path，不只isolated operator smoke。

---

# 117. R20 必须生成的 artifacts

## R20-453

```text
factor_engine/docs/R20_EXECUTION_SEMANTIC_AUDIT.md
factor_engine/docs/R20_EXECUTION_SEMANTIC_AUDIT.json
factor_engine/docs/R20_EXECUTION_SEMANTIC_AUDIT.csv

factor_engine/docs/R20_REWRITE_EQUIVALENCE_MATRIX.md
factor_engine/docs/R20_REWRITE_EQUIVALENCE_MATRIX.json

factor_engine/docs/R20_CSE_EQUIVALENCE_AUDIT.md
factor_engine/docs/R20_CSE_EQUIVALENCE_AUDIT.json

factor_engine/docs/R20_CACHE_IDENTITY_AUDIT.md
factor_engine/docs/R20_CACHE_IDENTITY_AUDIT.json

factor_engine/docs/R20_MATERIALIZATION_ROUNDTRIP_AUDIT.md
factor_engine/docs/R20_MATERIALIZATION_ROUNDTRIP_AUDIT.json

factor_engine/docs/R20_INCREMENTAL_EQUIVALENCE_AUDIT.md
factor_engine/docs/R20_INCREMENTAL_EQUIVALENCE_AUDIT.json

factor_engine/docs/R20_CONCURRENCY_DETERMINISM_AUDIT.md
factor_engine/docs/R20_CONCURRENCY_DETERMINISM_AUDIT.json

factor_engine/docs/R20_RESOURCE_ACCOUNTING_AUDIT.md
factor_engine/docs/R20_RESOURCE_ACCOUNTING_AUDIT.json

factor_engine/docs/R20_EXECUTION_PATH_MATRIX.csv
factor_engine/docs/R20_EXECUTION_PATH_MATRIX.json
factor_engine/docs/R20_EXECUTION_PATH_MATRIX.md
```

---

# 118. R20 execution matrix 每个 canonical/composite的字段

## R20-454

```text
canonical
execution_kind
scope
statefulness

logical_semantic_digest
optimized_semantic_digest
post_cse_semantic_digest
physical_semantic_digest

rewrite_rules_applied
lowering_hash
optimizer_hash

source_dependency_hash
universe_membership_hash

cse_exact_safe
rolling_cse_safe

native_backends
native_route_pass
fallback_route_pass

cache_cold_hash
cache_warm_hash
serial_hash
parallel_hash
full_hash
incremental_hash
materialized_reload_hash

semantic_loss_detected
identity_drift_detected
concurrency_drift_detected
cache_drift_detected
fix_action
```

## R20-455
不适用必须给原因，不能“skip/unknown/not tested”。

---

# 119. 质量门不看canonical数量

## R20-456
删除 `>=N operators` 类型质量门。R20看的是 execution path coverage=100%。

---

# 120. 已确认缺陷必须加 targeted regression tests

## R20-457
rolling CSE前后semantic digest一致。

## R20-458
`ts_mean(log(close),20)`绝不能与`ts_mean(close,20)` rolling CSE共享。

## R20-459
SQL materialized_series semantic attrs等于extracted subtree output semantics。

## R20-460
earnings_yield经过compile/CSE/hash/cache/execute都能处理typed missing literal。

## R20-461
调用MemoryGovernor admission probe 1000次后无 `__probe__` ghost usage。

## R20-462
stage=stop_warmup时不再继续warmup。

## R20-463
cgroup2CPU/explicit64最终effective<=2。

## R20-464
并发Polars lazy/non-lazy最终source state等于initial，结果deterministic。

## R20-465
persistent cache empty MultiIndex Series round-trip轴完全一致。

## R20-466
live/materialized reload输出quantization报告。

## R20-467
lineage-derived snapshot真实写入row metadata/catalog。

## R20-468
仅修改lowering constant时lowering semantic hash变化、evidence stale、cache invalidated。

## R20-469
unsupported source config object production hard fail，不stringify。

## R20-470
secondary SourceRef不同的factor不得跨dependency CSE共享。

---

# 121. 最终静态机器审计

## R20-471
AST/grep至少扫描：

```text
bare PlanNode rewrite
except Exception pass
default=str
repr fallback in identity
str(object) fallback in identity
os.environ production safety branch
manual operator set
manual alias map
manual window-name inference
param or default
shared data_source mutation
float32 silent cast
NaN structural literal
untyped materialized_series
semantic_attrs omitted
```

---

# 122. 最终动态机器审计

## R20-472

```text
compile determinism
optimizer equivalence
structural CSE equivalence
rolling CSE equivalence
physical lowering equivalence
backend path equivalence
cache cold/warm
parallel/serial
materialization roundtrip
incremental/full
checkpoint/full
```

所有retained production-capable路径都要有机器结论。

---

# 123. Definition of Done

下面全部满足才算R20完成：

- [ ] rolling CSE不再丢semantic_attrs；
- [ ] rolling CSE不再用first-column heuristic判定不同input subtree等价；
- [ ] manual rolling operator/window truth sources被清理；
- [ ] SQL physical lowering全程保留semantics；
- [ ] composite lowering使用semantic-aware builder；
- [ ] composite root与original output semantic可证明一致；
- [ ] lowering hash覆盖constants/defaults/helper dependencies；
- [ ] fastpath rewrite全部有equivalence certificate；
- [ ] zscore rewrite严格等价或禁用；
- [ ] rewrite规则可达性100%有测试；
- [ ] optimizer before/after differential全绿；
- [ ] optimized final plan通过post-optimization temporal proof；
- [ ] scoped-universe classification内部错误fail closed；
- [ ] source config canonicalizer统一；
- [ ] CSE/cache/factor identity来自同一ExecutionSemanticIdentity；
- [ ] FactorExecutionScope.market不再默认A；
- [ ] CSE scope包含真实secondary source dependencies；
- [ ] universe membership hash接通；
- [ ] unresolved operator contract production禁止CSE/cache；
- [ ] field catalog unavailable production fail；
- [ ] semantic attrs不再repr fallback；
- [ ] factor identity不再default=str；
- [ ] TypedIR semantic hash存在；
- [ ] lineage区分logical/optimized/physical/execution hashes；
- [ ] PhysicalPlan具有可审计execution contract；
- [ ] malformed SourceRef不再被SQL detector吞；
- [ ] materialized_series带完整语义；
- [ ] downsample timestamp/session/grain验证完整；
- [ ] 1D result Index normalization修复；
- [ ] operator call validation不再危险重复；
- [ ] Polars lazy不再修改共享data source；
- [ ] restore failure production hard fail；
- [ ] underlying source不再挂execution-local wrapper pointer；
- [ ] logical-source并发不串execution；
- [ ] MemoryGovernor thread-safe；
- [ ] CacheManager/PanelCache accounting atomic；
- [ ] check_admit无ghost accounting；
- [ ] check_pre_warmup与stage语义一致；
- [ ] resource plan production不silent disable；
- [ ] resource strictness使用真实run_mode；
- [ ] CPU requested slots clamp hard quota；
- [ ] process-global resource scope并发安全；
- [ ] Polars effective threads不虚假报告；
- [ ] LRU overwrite真正MRU；
- [ ] cache bytes/governor accounting invariant全绿；
- [ ] persistent cache schema hash真实验证；
- [ ] empty cached Series保留轴；
- [ ] concurrent persistent writes generation-safe；
- [ ] NaN structural literal收敛到typed NULL；
- [ ] constant folding支持typed literal；
- [ ] scalar default不使用truthiness；
- [ ] composite panel/scalar arity明确；
- [ ] declared composite deps与actual reads一致；
- [ ] legal composite branch全部lower，illegal直接reject；
- [ ] branch certification名称与实际覆盖一致；
- [ ] direct-vs-lowered differential全绿；
- [ ] A股limit tolerance单位显式；
- [ ] generated plan重新做unit/grain/availability checker；
- [ ] storage precision进入factor identity；
- [ ] live/materialized reload parity有证据；
- [ ] resolved snapshot写入row metadata；
- [ ] production identity computation异常不再吞；
- [ ] FactorSemanticIdentity V2包含compiler/numeric/math/storage semantics；
- [ ] full input semantic dependency digest接通；
- [ ] catalog strict mode不fail-open；
- [ ] FactorCatalog线程模型明确；
- [ ] materialize physical+catalog transaction恢复测试通过；
- [ ] recompute-window NaN不会留下stale finite；
- [ ] incremental==full recompute；
- [ ] SourceRef revision反向失效依赖；
- [ ] batch warmup不改变factor history anchor；
- [ ] worker共享对象并发合同明确；
- [ ] backend route telemetry root-local；
- [ ] native failure不会执行未授权fallback；
- [ ] runtime resolved SourceRef dependency进入identity；
- [ ] row/catalog/checkpoint identity一致；
- [ ] full identity digest持久化；
- [ ] rewrite provenance完整；
- [ ] no bare rewrite PlanNode；
- [ ] plan serialization round-trip semantics；
- [ ] ephemeral data scope不进入persistent artifact；
- [ ] physical SQL identity绑定dialect/compiler；
- [ ] SQL多subtree共享同一source snapshot；
- [ ] spill/resource accounting完整；
- [ ] serial/parallel deterministic；
- [ ] CSE off/on deterministic；
- [ ] cache cold/warm deterministic；
- [ ] SQL off/on deterministic；
- [ ] materialize reload deterministic；
- [ ] incremental/full deterministic；
- [ ] artifact generation原子且同一source generation；
- [ ] registry load-order deterministic；
- [ ] no memory address进入persistent identity；
- [ ] final production plan无legacy aliases；
- [ ] precompiled plan/analysis/compiler版本绑定；
- [ ] final execution semantic preservation audit覆盖100%。

---

# 124. 最终执行指令

请代码整改 AI 现在直接：

1. 读取服务器真实 HEAD、dirty状态与source semantic hash；
2. `load_all()` 动态枚举全部当前 canonical；
3. 建当前 compiler/runtime semantic path baseline；
4. 先修R20已确认的明确缺陷；
5. 建统一ExecutionSemanticIdentity/semantic carrier；
6. 统一planner rewrite constructor；
7. 修structural/rolling CSE；
8. 修composite lowering semantics/hash/contracts；
9. 修optimizer rewrite proof；
10. 修SQL physical lowering；
11. 修universe/source dependency identity；
12. 修cache/data_scope/factor identity统一；
13. 修Polars/data-source并发；
14. 修MemoryGovernor/resource scope；
15. 修cache LRU/persistence/schema；
16. 修typed NULL literal；
17. 修materialization precision/snapshot/identity；
18. 修incremental invalidation/full parity；
19. 修lineage/catalog/factor-version；
20. 建execution path matrix；
21. 跑 serial/parallel、CSE on/off、cache cold/warm、SQL on/off、Pandas/Polars、materialize reload、incremental/full全矩阵；
22. 重生production evidence/catalog/manifests；
23. 保证所有artifact来自同一source snapshot/generation；
24. 输出本R20要求的全部审计文件；
25. 最终打印：
   - server HEAD/source semantic hash；
   - current canonical count；
   - R20 confirmed bugs fixed count；
   - semantic-loss instances found/fixed；
   - unsafe CSE pairs found/fixed；
   - rewrite rules audited/fixed/disabled；
   - composite contracts audited/fixed；
   - cache identity defects fixed；
   - concurrency defects fixed；
   - resource-accounting defects fixed；
   - materialization/incremental mismatches fixed；
   - artifact generation consistency；
   - remaining R20 blocker count；
   - full test summary。

---

# 125. 最终原则

最终不是证明“代码能跑”，而是证明：

> **同一个 factor，从 DSL 到 IR、从 IR 到 optimized plan、从 plan 到 CSE/SQL/backend、从 runtime 到 cache、从 cache 到 materialization、从 full 到 incremental，在任何合法执行路径上都保持同一个可追溯、可证明、可复现的经济与时间语义。**

只要某一层仍存在：

```text
丢语义
猜语义
吞异常
使用第二套identity
静默fallback
共享错误cache
并发污染
增量不等于全量
落盘精度改变却不改identity
```

就不算生产化完成。
