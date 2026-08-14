# LOOP ENGINEERING V2 — 6 小时“百亿量化私募级”夜间持续硬化任务书

**Repository:** `18047533889/quant_projects`  
**重新审计基准:** committed `main` = `d78ed761b7d098d27e3acf916f3961d75096b3b3`  
**日期:** 2026-08-14  
**执行窗口:** 连续 6 小时  
**V1 文件保留:** `LOOP_ENGINEERING_INSTITUTIONAL_HARDENING_20260814.md` 不删除、不覆盖。  
**本文件性质:** V2 是在 V1 全部有效原则之上的增量再审计 + 旧状态修订 + 6 小时可直接执行的完整任务书。

---

# 0. 先读：这次任务的真正目标

本轮不是“再修 4 个 bug、再跑 687 个测试、再写一个 FINAL 报告”。

本轮目标是把当前平台推进到：

```text
INSTITUTIONAL_PRODUCTION_CANDIDATE
```

而不是凭单元测试数量宣称：

```text
PRODUCTION_CERTIFIED
```

真正的 `PRODUCTION_CERTIFIED` 仍需要在本轮之后经过：

```text
全量 A 股历史数据 replay
shadow run
paper trading
live canary
数据延迟/断流/修订演练
交易日历/停牌/涨跌停真实场景验证
故障恢复演练
持续若干交易日的 streaming health
```

六小时内必须做到的是：

1. 把所有当前已知的 P0/P1 correctness / leakage / packaging / evidence 问题压到 0；
2. 建立以后很难再次出现“报告说好了、main 实际坏了”的验证真值链；
3. 把暂时实现不了的研究功能明确 `RESEARCH_ONLY / OFFLINE_ONLY / SAFE_BY_GATING`，绝不假装 production；
4. 用独立 reviewer、exact SHA、clean wheel、differential test、causal prefix invariance、fault injection 来证明，而不是靠“pytest 全绿”；
5. 六小时没有用完时必须继续换审计方法找新问题，不能提前 55 分钟宣布结束。

---

# 1. 对刚刚 55 分钟 Loop Engineering 成果的重新定级

刚刚会话的 4 项工作有价值，但不能按原报告全部视为“P0 已完全关闭”。

## 1.1 REG-001 — 测试注册表污染

### 当前状态

```text
CLOSED_FOR_TEST_ISOLATION
但不是 CLOSED_FOR_PRODUCTION_REGISTRY_ARCHITECTURE
```

已经增加：

```text
quant_evaluator/tests/conftest.py
factor_preprocess/tests/conftest.py
factor_optimizer/tests/conftest.py
```

通过 autouse fixture 重置 global registry，确实能消除测试顺序污染。

### 但新的问题是

测试现在直接操作：

```python
registry._global_registry = ...
transforms._default_registry = None
```

这证明“测试可以清干净”，不证明真实服务中的：

```text
多线程
多任务
多 campaign
plugin 动态注册
worker fork
long-running process
```

不会互相污染。

### 新任务 REG-002

建立正式 production registry contract：

```text
ScopedRegistry / RegistryContext
registry.freeze()
registry_hash
generation_id
thread/process ownership
explicit dependency injection
```

生产模式：

```text
初始化完成 -> freeze -> 禁止运行中无审计修改
```

registry hash 必须进入 `RunManifest`。

---

## 1.2 CACHE-001 — FactorAssets similarity cache 2 元组 → 6 元组

### 当前状态

```text
PARTIALLY_FIXED
```

6 元组：

```text
factor_id_a
factor_id_b
method
universe_ref
period_start
period_end
```

比原来正确很多。

### 但不能关闭到机构级

仍缺：

```text
factor definition/version identity
factor value snapshot identity
data snapshot identity
calendar identity
validity/sample policy
metric implementation hash
null/tie/numerical policy
QE evidence identity
```

同一个 factor ID、同一个 universe、同一个期间，在数据回填/财报修订/代码升级后仍可能拿到旧 similarity。

更具体地说，当前 `find_similar()` 没有 universe/period 参数，却会扫描混合上下文 cache；同一 pair 在多个 universe / period 下的结果可能被混在一起返回。

### 新任务 CACHE-FA-002

FactorAssets 不应该 hash raw values；应把 cache key 升级成：

```text
SimilarityEvidenceKey(
    factor_identity_a,
    factor_identity_b,
    factor_value_snapshot_ref_a,
    factor_value_snapshot_ref_b,
    evaluation_scope_id,
    universe_ref,
    period_ref,
    method,
    metric_impl_hash,
    numerical_policy_id
)
```

---

## 1.3 ERR-001 — singular matrix

### 当前状态

```text
PARTIALLY_FIXED
```

rank deficient / LinAlgError 现在有显式 warning + NaN，比 silent result 好。

### 新发现 ERR-002

当前 `modeling/exposure/__init__.py` 在：

```text
有效样本不足
n_valid < K+1
```

时会直接：

```python
return y
```

即返回**原始未中性化因子**。

这是比 singular matrix 更危险的 silent semantic fallback：

```text
调用方以为得到 exposure-neutralized factor
实际得到 raw factor
```

### 必须修改

不要返回裸 ndarray 表示“成功”。

建立：

```python
NeutralizationResult(
    values,
    valid_mask,
    status,
    n_obs,
    rank,
    condition_number,
    exposure_names,
    method,
    warnings,
    failure_reason
)
```

`INSUFFICIENT_OBSERVATIONS` 必须显式。

另外 weighted OLS 不要构造：

```python
W = np.diag(weights)
```

这是 O(N²) 内存。

改为：

```text
Xw = X * sqrt(w)[:, None]
yw = y * sqrt(w)
```

---

## 1.4 TIME-002 — created_at=data_end

### 当前状态

```text
REOPENED_AS_PROVENANCE_BUG
```

“为了 deterministic，把 `created_at` 改成 `data_end`”不是机构级做法。

它混淆了：

```text
数据结束时间
artifact 真实生产时间
```

### 正确设计

分成：

```text
data_start
data_end
fit_start
fit_end
produced_at_utc       # 真实 wall-clock 审计时间
artifact_content_hash # deterministic
artifact_id           # 由 deterministic identity 导出
run_id
source_commit
config_hash
snapshot_hash
```

**确定性应该靠内容寻址，不应该伪造时间戳。**

---

## 1.5 DATA-001 — NaN 传播策略

不要只决定“零方差返回 NaN 还是 0”。

升级成统一：

```text
NumericValidityPolicy
```

至少覆盖：

```text
NaN
+Inf
-Inf
pd.NA
None
zero denominator
near-zero denominator
zero variance
near-zero variance
constant cross-section
all ties
insufficient observations
not disclosed
not applicable
suspension
limit-up/limit-down
untradable
missing label
missing exposure
```

并明确：

```text
INVALID
UNAVAILABLE
UNTRADABLE
MISSING_DISCLOSURE
NOT_APPLICABLE
WARMUP
NUMERICAL_FAILURE
```

不能全部压成一个 NaN 后忘记原因。

---

# 2. 这次重新审计发现的 STOP-THE-LINE P0

下面是今晚必须优先处理的真正生产阻断项。

---

# P0-A — q_executor 当前 committed main 被删成了“只剩 PASS 常量”

这是本轮最明显的验证系统失效实例。

当前：

```text
factor_engine/backend/q_backend/q_executor.py
```

实际上只剩：

```python
Q_FAN_IN_INPUT_PRESERVATION = "PASS"
Q_WORKSPACE_ISOLATED = "PASS"
Q_CONNECTION_THREAD_SAFE = "PASS"
Q_RESIDENT_HANDLE_BOUNDED_LIFETIME = "PASS"
```

而：

```text
factor_engine/backend/q_backend/__init__.py
```

仍然 import：

```python
QExecutor
```

`q_backend.py` 仍 import：

```python
QExecutionFallbackPolicy
get_q_executor
```

因此当前 exact committed SHA 的 q backend import surface 已经是 broken。

## 今晚必须做

1. 从最后一个已验证的真实 executor 版本恢复/重建完整实现；
2. 禁止通过“写 PASS 常量”表示 gate 通过；
3. 所有 gate 都必须是 executable proof；
4. 增加 recursive import 测试；
5. 增加：
   - fan-in DAG；
   - diamond DAG；
   - concurrent workspace；
   - connection lock；
   - stale handle；
   - cleanup；
   - process restart；
   - exception cleanup；
   - cancellation；
6. commit 后 fresh worktree 重新验证。

## 硬规则

代码里以后禁止：

```python
SOME_PRODUCTION_GATE = "PASS"
```

作为 production evidence。

必须变成：

```python
GateResult(
    gate_id=...,
    passed=runtime_proof(...),
    evidence_ref=...
)
```

---

# P0-B — q capability “单一权威”目前仍是假闭环

当前已经新建：

```text
QPhysicalImplementationRegistry
```

方向对。

但当前 registry 最后仍是：

```python
# TODO: Populate from evidence ledger
```

也就是 production registry 实际为空。

与此同时：

```text
get_declared_native_ops()
```

仍是一个大型手工 set。

旧 `_PHASE1_NATIVE_OPS` 仍存在。

更严重的是所谓：

```text
Q_CAPABILITY_SINGLE_AUTHORITY
```

gate 遇到旧 authority 时只是：

```python
pass
```

然后返回：

```text
PASS
```

这是典型的 **tautological gate（自证式门禁）**。

## 今晚修改

只能保留一个真 authority：

```text
QPhysicalImplementationRegistry
```

每一个 production op 必须有：

```text
canonical_op
lowering_id
parameter_domain_id
implementation_hash
q_version
pykx_version
compile_artifact
runtime_artifact
parity_artifact
null_semantics_artifact
time_semantics_artifact
PIT_scope
certified_at_commit
```

然后：

```text
compiler admission
backend routing
production report
CI gate
```

全部读取这同一个 registry。

任何旧 list 只能做 migration input，不能做 admission authority。

---

# P0-C — q compiler 仍有 unreachable lowering / 错误 lowering

当前 compiler 仍然有：

```python
if q_func.startswith("{"):
    ...
```

放在 special lowering 前面。

因此：

### ts_beta

map 中：

```text
ts_beta -> lambda
```

会被 generic lambda 抢先处理。

后面的 rolling-window `ts_beta` branch 实际不可达。

### wma

同样 map 是 lambda，后面的 window branch不可达。

### clip

map 是 3-input lambda，但 generic branch可能以 1 input 调用，后面的 lower/upper param branch不可达。

### ts_quantile / cs_quantile

有 special branch，但不在 `_operator_map`，所以：

```text
can_compile_operator -> False
```

根本走不到 special branch。

### 还有 backend-owned defaults

当前还有：

```text
window=20
span=20
clip ±999999
fillna 0
```

backend 不应该拥有 canonical default。

## 今晚方案

不要继续 if/elif compiler。

改成：

```python
Q_LOWERING_REGISTRY = {
    canonical_op: QLoweringSpec(
        lowering_fn=...,
        param_schema_ref=...,
        semantic_version=...,
        implementation_hash=...,
    )
}
```

硬门禁：

```text
每个 production canonical op:
exactly one reachable lowering
zero backend-owned defaults
all parameter domains tested
```

---

# P0-D — q backend 仍未真正接 PhysicalBackendRegion

当前 `QBackend.execute()` 自己明确 warning：

```text
receiving whole logical tree (simplified)
Production requires PhysicalBackendRegion
```

这就说明它不能被标记为 production。

今晚二选一：

## 方案 1：做完

```text
Canonical DAG
  ↓
PhysicalPlanner
  ↓
PhysicalBackendRegion
  ↓
QCompiler
  ↓
QExecutor
```

## 方案 2：时间不够

把整个 q backend production capability 强制：

```text
RESEARCH_ONLY / DISABLED_FOR_PRODUCTION
```

但绝不能留：

```text
partially implemented + can be selected by production planner
```

另外当前 `get_q_backend()` 是全局 singleton：

第一次：

```python
get_q_backend(fallback_to_pandas=True, production_mode=False)
```

之后再：

```python
get_q_backend(fallback_to_pandas=False, production_mode=True)
```

仍可能拿同一个实例。

必须解决 research/production instance pollution。

---

# P0-E — FactorPreprocess 出现系统性未来函数，比 DATA-001 更急

这是本轮重新审计最重要的新发现之一。

很多文件 docstring 都写：

```text
causal
no future leakage
shift(1)
```

但代码实际上仍会泄漏。

**shift(1) 并不自动等于 causal。**

---

## P0-E1 — rolling 跨股票污染

当前：

```python
values.groupby(asset)[value].shift(1).rolling(...)
```

shift 后没有再次 groupby。

rolling 对整个拼接后的 Series 做窗口。

于是：

```text
股票 A 的尾部
会进入
股票 B 的开头窗口
```

受影响至少：

```text
rolling_mean
rolling_std
rolling_zscore
ewma
volatility_scale
realized_volatility
garch_inspired_volatility
```

这属于明确 correctness bug。

### 必测 metamorphic test

构造两只股票：

```text
A: 正常数值 + 最后一天 1e12
B: 正常数值
```

修改 A，不允许 B 的任何结果变化。

再把 asset block 顺序 A/B 与 B/A 互换，按 key 对齐后的结果必须完全一致。

---

## P0-E2 — HP filter 是 two-sided，shift(1) 也不 causal

当前 HP：

```text
先 shift(1)
再对整个历史 sample 一次性解 HP optimization
```

早期时间点的 trend 会使用后面的 lagged 数据。

这是典型 future leakage。

### 今晚最安全方案

直接将 HP：

```text
OFFLINE_ONLY
```

从 production model-input registry 移除。

后续如果真要用：

```text
expanding endpoint HP
state-space local linear trend
Kalman filter
one-sided real-time estimate
```

---

## P0-E3 — STL 也是 full-sample two-sided

当前：

```text
shift(1)
STL(full lagged series).fit()
```

LOESS/STL 对历史早期点使用后续样本。

仍然泄漏。

并且：

```python
except Exception:
    return all NaN
```

还会静默吞数值/实现错误。

### 处理

今晚先：

```text
STL -> OFFLINE_ONLY
```

不进入 production `causal_safe` registry。

---

## P0-E4 — Wavelet full-sample reconstruction 不是 causal

当前：

```text
shift
wavedec(full series)
waverec(full series)
symmetric padding
```

早期输出依赖后续观测。

denoise 的 threshold 也用整段样本估计。

### 处理

今晚：

```text
wavelet_decompose / wavelet_smooth / wavelet_denoise
-> RESEARCH_ONLY or OFFLINE_ONLY
```

以后需要 causal DWT / rolling endpoint implementation。

---

## P0-E5 — bandpass_filter 明确用了 filtfilt

代码自己写：

```text
filtfilt is acausal but we already shifted data by 1
```

这个判断是错误的。

`filtfilt` 使用整个 shifted series 的前后数据。

shift 1 天不可能消掉几十/几百天的 future dependency。

### 处理

禁止 production。

改 one-sided：

```text
sosfilt + state
```

或者干脆 research-only。

---

## P0-E6 — Christiano-Fitzgerald 路径也泄漏

虽然最后 window 是 backward-looking，

但前面：

```python
np.polyfit(x, y, 1)
```

是用整段 full lagged sample 拟合趋势。

早期 t 仍得到未来拟合参数。

---

## P0-E7 — linear interpolation 本质使用未来值

当前：

```text
linear_interpolate
time_weighted_interpolate
```

通过 gap 后面的未来观察去补 gap 前的空值。

这不允许出现在实时预测 feature pipeline。

### 分类

```text
OFFLINE_DATA_REPAIR
```

而不是：

```text
CAUSAL_MODEL_FEATURE
```

---

## P0-E8 — median/mean imputation 用全样本

当前：

```python
groupby(asset).transform("median")
groupby(asset).transform("mean")
```

用的是整段历史，包括未来。

### 正确

要么：

```text
expanding lagged statistic
```

要么：

```text
train-fitted imputer state
```

再应用到 validation/test。

---

## P0-E9 — correlation regime 使用未来分位数

`detect_correlation_regime()` 虽然 rolling correlation 本身只看过去，

但 regime boundaries：

```python
np.nanquantile(avg_corr, percentiles)
```

对整段时间一次性计算。

因此早期 regime label 使用未来的 regime distribution。

### 正确

```text
train-fitted boundaries
或
expanding/rolling historical boundaries shifted 1
```

---

# 3. 必须新增一个统一的 CAUSALITY CERTIFICATION FRAMEWORK

不要继续靠：

```python
causal_safe=True
```

这种手写 boolean。

当前 TransformRegistry 默认：

```text
causal_safe = True
```

这是危险默认值。

改为：

```text
UNVERIFIED
```

定义：

```python
CausalityClass:
    STATELESS_CROSS_SECTIONAL
    PAST_ONLY
    FITTED_TRAIN_ONLY
    ONLINE_STATEFUL
    OFFLINE_ONLY
    RESEARCH_ONLY
    UNVERIFIED
```

每个 transform production admission 必须关联：

```text
implementation_hash
causal_prefix_invariance_test_ref
asset_isolation_test_ref
fit_apply_test_ref
snapshot_contract_ref
```

---

# 4. 今晚新增一个最重要的通用未来函数测试：Prefix Invariance

对任何声称 causal 的 transform `F`：

给定序列：

```text
x[0:t]
```

计算：

```text
y1 = F(x)[0:t]
```

然后把未来：

```text
x[t+1:]
```

替换成：

```text
极大数
极小数
随机值
NaN
不同 regime
```

重新计算：

```text
y2 = F(x_modified)[0:t]
```

必须：

```text
y1 == y2
```

在声明 tolerance 内严格成立。

这一个测试可以自动扫：

```text
rolling
EWMA
volatility
decomposition
imputation
regime
neutralization state
feature selection
PCA
model preprocessing
```

命名：

```text
CAUSAL_PREFIX_INVARIANCE
```

以后 production transform 没过它，一律不能 `causal_safe`。

---

# 5. P0-F — QE multiple-testing 当前有明确算法 bug

当前：

```text
quant_evaluator/metrics/multiple_testing.py
```

Benjamini-Hochberg 里：

```python
original_idx = valid_indices[sort_idx]
...
adjusted_flat[original_idx[sort_idx]] = adjusted_sorted
```

`original_idx` 已经按照 `sort_idx` 排过一次。

又套一次：

```text
original_idx[sort_idx]
```

会把结果映射到错误 hypothesis。

reject mask 同样存在重复排序问题。

Holm 也有类似 mapping 问题。

另外 Holm adjusted p-value 的累计处理也值得直接和权威参考实现做 differential test。

`compute_fdr()` 当前注释说：

```text
FDR = E[V/R]
```

但实现用：

```text
mean rejected p-value × n_tests / n_rejections
```

这不是一般意义下可观测的“empirical FDR”权威估计，不应作为 production research-control 核心指标。

## 今晚必须做

所有 multiple-testing 实现逐样本对比：

```python
statsmodels.stats.multitest.multipletests
```

覆盖：

```text
Bonferroni
Holm
BH/FDR
Sidak
NaN
duplicate p
p=0
p=1
random permutations
```

增加 permutation invariance：

输入 p-values 顺序打乱，恢复原 order 后 adjusted p / reject 必须一致。

这是 research integrity P0。

---

# 6. P0-G — FactorAssets CompositeGate 会把错误 metric 送进错误 gate

当前 `CompositeGate` 构造器虽然有：

```text
metric_bindings
```

但 `evaluate_all()` 实现基本没使用它。

它会：

1. 根据字符串包含关系猜；
2. 猜不到再做 prefix 猜；
3. 最后拿“第一个 available metric”试。

这意味着：

```text
turnover gate
```

理论上可以拿到：

```text
IC
coverage
别的 metric
```

然后照样产生 PASS/FAIL。

这是 admission 语义破坏。

## 今晚必须改为

生产模式只允许：

```text
explicit metric ID binding
exact evidence schema
metric unit/direction validation
```

禁止 heuristic fallback。

缺 metric：

```text
ERROR / FAIL_CLOSED
```

不是拿第一个 metric。

### 另外

SelectionPolicy 现在是：

```text
fixed gates
+ max_similarity threshold
```

长期不应把 factor admission 简化成固定加权/固定阈值体系。

正确层次：

```text
Hard Safety Gates
↓
Evidence Profile
↓
Conditional Novelty
↓
Pareto / contextual / budgeted selection
```

---

# 7. P0-H — QE cache 的真正 P0 仍然完全没解决

注意：

刚才修的是：

```text
factor_assets similarity cache
```

不是：

```text
quant_evaluator runtime cache
```

当前 QE `Evaluator` full-batch cache key 依然基本是：

```text
metric_id
factor_ids
```

chunk 也主要是：

```text
IDs + slices
```

缺少：

```text
factor values/value_hash
factor validity
label values/hash
label validity
label timing
metric params
metric implementation hash
universe
evaluation scope
split
data snapshot
calendar
backend
numeric policy
```

这会造成 silent stale metric。

## 今晚设计

```python
EvaluationCacheIdentityV2(
    metric_id,
    metric_version,
    metric_impl_hash,
    metric_config_hash,
    factor_batch_hash,
    label_bundle_hash,
    evaluation_scope_id,
    split_plan_id,
    universe_id,
    data_snapshot_id,
    calendar_id,
    numerical_policy_id,
    backend_identity,
)
```

缺关键 identity：

```text
production cache disabled
```

不能自动弱化 key。

---

# 8. P0-I — QE chunking/streaming 数学等价性仍未解决

当前 chunk reducer 仍有：

```text
IC        -> concatenate axis 0
coverage  -> mean(chunk_values)
summary   -> take first chunk
other arr -> concatenate axis 0
```

这不能对所有 metric 成立。

## 典型错误

### RankIC / quantile

必须看完整同日 cross-section。

不能按 asset 分块后各算各的再合并。

### rolling metrics

time chunk 要有：

```text
warmup overlap
或
explicit state carry
```

### factor chunk

输出 factor axis 必须精确还原。

### coverage

不能平均各 chunk coverage。

正确：

```text
sum(valid_count) / sum(total_count)
```

### summary

不能拿第一个 chunk 当全局 summary。

## 今晚实现

每个 metric 注册：

```python
MetricPartitionSemantics(
    legal_partition_axes,
    requires_full_cross_section,
    requires_time_order,
    warmup,
    stateful,
    merge_operator,
    merge_associative,
    output_axes,
)
```

缺声明：

```text
production default = NOT PARTITIONABLE
```

硬测试：

```text
full == every legal chunk plan
full == streaming
full == parallel
cache on == cache off
```

---

# 9. P0-J — QE chunk 会丢坐标与 provenance

FactorBatch 原本有：

```text
AxisRef.values
context_refs
value_hash
```

但 chunk extraction 没有可靠完整传播这些 identity。

对于量化：

```text
shape=(T,N,F)
```

完全相同，不代表股票顺序相同。

必须携带：

```text
exact time coordinates
exact asset coordinates
factor coordinates
parent batch hash
chunk hash
universe ref
snapshot ref
PIT/asof ref
```

同时：

```text
FactorBatch
LabelBundle
```

虽然 dataclass frozen，但内部：

```text
numpy arrays
dict metadata/context_refs
```

仍然是 mutable。

机构级 identity object 应：

```text
copy + write-protect ndarray
MappingProxyType / immutable mapping
```

否则创建 hash 后对象还可被改。

---

# 10. P0-K — LabelBundle 的“strict timing”仍然不 strict

当前 LabelBundle docstring 说 strict timing，

但真实 contract 仍没有完整实现：

```text
signal_available_time
execution_time required
array length alignment
timezone canonicalization
decision <= signal <= execution <= label_start <= label_end
```

今晚必须冻结 label contract。

对 A 股尤其要明确：

```text
decision_time
signal_available_time
first_executable_time
label_start
label_end
exchange calendar
buyable/sellable mask
```

“收盘后形成信号”不允许默认按当日 close 成交。

---

# 11. P0-L — ModelArtifact OOS 仍有 bypass

最新代码增加了：

```text
predict_oos(application_window=...)
```

这是进步。

但同时还保留：

```python
ModelArtifact.predict(X)
```

该方法：

```text
不需要 ApplicationWindow
不需要 dates
直接 preprocessing.transform
```

所以任何调用者仍可以拿未来/历史任意 X 调 `predict()` 绕过 OOS gate。

而 `FrozenPreprocessing.transform(application_window=...)` 当前里面的 temporal validation 还是：

```python
pass
```

也就是说真正 enforcement 主要在上层一个 API 路径。

## 今晚必须做

production artifact：

```text
predict()
```

不能成为无时间语义 public API。

方案：

```python
predict_in_sample(...)
predict_oos(..., application_window, dates REQUIRED)
```

或者：

```python
predict(..., prediction_context REQUIRED)
```

所有 production path 都要上下文。

`dates` 不应 optional。

必须验证：

```text
date >= artifact activation
date >= label maturity / availability rules
schema hash
feature order
snapshot
universe
```

---

# 12. P0-M — Modeling gap_days 修复仍是 partial

最新代码已经开始 enforce train→validation gap，这是进步。

但需要继续审：

```text
train -> validation
validation -> test
train -> test（没有 validation 时）
```

都必须有统一规则。

当前 gap 还是 calendar day 思维。

A 股日频更合理的是：

```text
trading-session gap
```

由 market calendar authority 提供。

不要本地自己猜：

```text
gap_days + 1
```

必须先定义 canonical semantics：

```text
gap=0
gap=1
gap=5
```

到底允许哪些 anchor。

更重要的是：

```text
purge by label interval
```

不是只看 date gap。

如果训练样本 t 的：

```text
label_end
```

落进 validation/test，仍然泄漏。

---

# 13. P0-N — FactorAssets assembly/campaign 仍是 broken public package surface

当前 `assembly/__init__.py` 仍 import：

```text
set_builder
selection_policy
diversification
```

而当前 committed tree 中对应文件不存在。

`campaigns/__init__.py` 仍 import：

```text
CampaignSpec
CampaignStatus
split_ledger
```

实际 `campaign_coordinator.py` 定义的是：

```text
CampaignConfig
Campaign
CampaignState
```

并且 `split_ledger.py` 不存在。

为什么之前 clean-wheel 还能 PASS？

因为 `factor_assets/scripts/wheel_clean_install_smoke.py` 只 import：

```text
factor_assets root
FactorAsset
AssetRepository
LifecycleState
LineageRef
```

它根本没递归 import subpackages。

## 今晚立即添加

```python
pkgutil.walk_packages(...)
```

递归 import production modules。

再维护：

```text
EXPECTED_PUBLIC_MODULES.json
```

验证：

```text
source expected modules
==
wheel contained modules
==
importable modules
```

---

# 14. P0-O — 当前 latest SHA 没有 CI status proof

重新查询最新 committed SHA 时：

```text
combined status = empty
workflow runs = empty
```

无论原因是 workflow path trigger、GitHub App 可见性还是 CI 没跑，

结论都是：

```text
当前 commit 没有可消费的 root production gate evidence
```

今晚必须建立一个统一 root gate：

```text
.github/workflows/platform-release-gate.yml
```

或者至少服务器上的：

```text
scripts/platform_release_gate.py
make institutional-gate
```

覆盖：

```text
dataaccess
factor_engine
quant_evaluator
factor_preprocess
factor_assets
factor_optimizer
modeling
```

---

# 15. P0-P — multiple package authority 出现 split-brain

当前代码已经出现：

```text
factor_engine/modeling
modeling/
```

以及：

```text
factor_assets/optimizer
factor_optimizer/
```

还有：

```text
modeling preprocessing
factor_preprocess/
```

这不是靠写一份 `AUTHORITY.md` 就能真正解决的。

## 今晚建立 Architecture Authority Map

推荐：

```text
DataAccess
    唯一拥有 data/PIT/calendar/snapshot/universe data semantics

FactorEngine
    唯一拥有 factor DSL/IR/operator semantics/physical factor execution

QuantEvaluator
    唯一拥有 metrics/evaluation evidence

FactorPreprocess
    唯一拥有 pre-model feature transforms

FactorOptimizer
    唯一拥有 mutation/search/optimization algorithms

FactorAssets
    唯一拥有 identity/registry/lifecycle/novelty/admission/aggregation governance

Modeling
    唯一拥有 model training/CV/OOS prediction/model artifact
```

因此长期建议：

```text
factor_engine/modeling -> migrate/deprecate to standalone modeling
factor_assets/optimizer -> orchestration adapter only，不再自己拥有 Pareto/search 算法
modeling preprocess -> 调 factor_preprocess，不复制实现
```

如果今晚没时间迁移：

至少建立 import-boundary gate，明确一处 canonical authority，另外一处只 adapter/re-export，禁止双写业务逻辑。

---

# 16. P1 — FactorAssets repository/SeenIndex 目前只是内存实现

当前：

```text
AssetRepository = in-memory
SeenIndex = in-memory
```

代码自己甚至写：

```text
NO SQLite, NO file I/O
```

这对单元测试可以。

对：

```text
100,000+ factors
多组员
多进程
断点恢复
审计
权限
全局去重
```

不够。

## 夜间能做的 v1

建立：

```python
AssetRepositoryBackend protocol
SeenIndexBackend protocol
```

然后提供：

```text
InMemoryBackend       # tests
SQLiteWALBackend      # single-server durable v1
```

要求：

```text
UNIQUE canonical_hash
UNIQUE factor_id
transactional lifecycle event
append-only event table
optimistic version
WAL
crash recovery
migration schema version
```

后续扩成：

```text
PostgreSQL / central service
```

对于“算法负责人不能看到全因子”：

`GlobalSeenIndex` 最适合变成 central service：

```text
只返回 seen/similarity/novelty result
不暴露其他组公式
```

---

# 17. P1 — FactorAssets representative/aggregation 还不够机构级

## EQUAL_WEIGHT 名称与实现不一致

文档说：

```text
equal representation per subfamily
```

实际只是：

```text
按输入顺序等距抽样
```

根本没用 subfamily assignment。

## MAX_IC 丢方向

当前：

```python
abs(ic)
```

选出负 IC 后只保存 magnitude。

必须显式：

```text
raw_ic
orientation
oriented_ic
```

## MIN_CORRELATION 不能保证 selected set 彼此低相关

当前是每个 factor 对整个 family 算 average corr 再排序。

改成：

```text
greedy max-min
facility location
cluster medoid
mRMR
DPP（可选）
```

## 更好的 production selection

目标不是：

```text
max IC
```

而是：

```text
predictive evidence
stability
incremental information
cost/turnover
capacity
exposure
fragility
complexity
diversity
```

做 Pareto / budgeted selection。

---

# 18. P1 — SelectionPolicy 不应把“相关性 > 0.7”直接等价于删除

高相关不一定是冗余。

要区分：

```text
signal corr
rank corr
PnL corr
tail overlap
exposure similarity
horizon similarity
regime similarity
conditional/residual information
```

对于候选 factor：

```text
f_new = projection(existing) + residual
```

评估 residual：

```text
residual IC
conditional IC
incremental OOS utility
```

高相关 factor 优先：

```text
SHADOWED
```

而不是直接物理删除。

这样以后 regime 改变还可以重新激活。

---

# 19. P1 — IC/RankIC 与统计推断还要建立更严谨的 reference system

当前 daily IC reference 逻辑整体可用，但大规模实现还是双 Python loop：

```text
T × F
```

对 100k factors 不够。

今晚 correctness 修好后，做：

```text
block factor matrix
NumPy/Numba
shared ranks
shared valid masks
Arrow/contiguous buffer
```

但禁止为了快改变：

```text
tie
NaN
min_assets
ddof
rank denominator
```

统计推断要区分：

```text
naive t-test
HAC/Newey-West
block bootstrap
stationary bootstrap
multiple-testing correction
```

不能所有 IC p-value 都当独立同分布。

---

# 20. P1 — IC stability 当前定义需要重新审视

当前 `compute_ic_stability()`：

```text
rolling window 分成前半/后半
再把两个 half 的 IC 序列直接 corr
```

有两个问题：

1. 两半的时间点不是同一个时点，逐位 correlation 的经济意义有限；
2. 如果两半 NaN 数不同，过滤后长度可能不同。

更合理的 stability evidence：

```text
rolling mean IC
rolling ICIR
sign survival
rank of factor performance across regimes
year-by-year dispersion
worst-window IC
change-point
half-life
subsample consistency
```

保留旧指标可以，但不要把它叫唯一 “stability”。

---

# 21. P1 — Regime weighting 的 semantic fallback 不严谨

`regime_adaptive_weights()` 当前：

未知 regime：

```text
直接保留原 factor values
```

这会让：

```text
已加权 signal
突然切回 raw scale
```

应由显式 policy 控制：

```text
ERROR
NA
LAST_KNOWN_REGIME
GLOBAL_FALLBACK_WEIGHTS
EQUAL_WEIGHT
```

并记录 fallback event。

另外名为：

```text
method="sharpe"
```

实际使用：

```text
correlation / factor std
```

不是真正 Sharpe。

需要改名或重定义。

---

# 22. P1 — silent broad exception 全仓扫描

本轮已经发现：

```text
STL
wavelet
bandpass
CF
regularized neutralization
```

多处：

```python
except Exception:
    return NaN
```

这会把：

```text
programming bug
schema bug
dtype bug
numerical failure
insufficient data
optional dependency failure
```

全部混成一个 NaN。

今晚做静态扫描：

```text
except Exception
except:
return None
return empty DataFrame
return all NaN
fallback
pass
TODO
stub
mock
toy
```

Critical path 必须改成 typed failure taxonomy。

---

# 23. 统一 Failure Taxonomy

至少：

```text
ContractViolation
DataUnavailable
InsufficientObservations
NumericalFailure
RankDeficient
SchemaMismatch
SnapshotMismatch
TimingViolation
PITViolation
BackendUnavailable
BackendSemanticMismatch
DependencyUnavailable
BudgetExceeded
Timeout
Cancellation
InfrastructureFailure
ResearchOnlyCapability
ProductionGateFailure
```

“结果 NaN”不是 error taxonomy。

---

# 24. V1 的 Verification Truth Chain 原则继续保留，并升级

下面规则全部保留：

```text
DISCOVERED
-> REPRODUCED
-> FAILING_TEST_ADDED
-> FIXED_LOCAL
-> DOMAIN_TEST_PASS
-> CROSS_PACKAGE_PASS
-> COMMITTED
-> EXACT_SHA_VERIFIED
-> RED_TEAM_APPROVED
-> CLOSED_VERIFIED
```

并新增：

```text
MERGE_CONFLICT_REVALIDATED
WHEEL_VERIFIED
CAUSALITY_VERIFIED
```

任何 issue commit 后代码变化：

```text
相关 evidence 自动 STALE
```

---

# 25. 所有 subagent 必须用隔离 worktree / 严格 file ownership

你们之前已经发生过：

```text
一个并发 session 删除另一个 session 的 27+ 文件
```

这次 q_executor 又出现疑似类似 regression。

今晚禁止多个 agent 直接在同一 working tree 随便改。

推荐：

```text
/worktrees/agent-qe
/worktrees/agent-fp
/worktrees/agent-fa
/worktrees/agent-q
/worktrees/agent-model
...
```

每个 agent：

```text
独立 branch/worktree
只改 owner scope
```

Coordinator 才能 merge。

如果环境不允许 worktree：

必须至少：

```text
non-overlapping write scopes
shared files only coordinator edits
```

共享：

```text
pyproject
root CI
root registry
package __init__
version
global config
```

只能 Integrator/Coordinator 改。

---

# 26. 今晚建议 12 个 Subagents

不是都写代码。分为 owner 与攻击者。

---

## Agent 0 — Chief Coordinator / Merge Guardian

只负责：

```text
任务账本
优先级
file ownership
合并
exact SHA
re-open
最终 capability matrix
```

不大面积修业务实现。

---

## Agent 1 — Verification / Packaging / CI

只攻击：

```text
假 PASS
stale report
broken wheel
dangling imports
missing files
dirty tree certification
CI blind spots
```

第一件事：

```text
复现 q_executor broken import
复现 FA assembly/campaign broken import
```

---

## Agent 2 — FactorPreprocess Causality Owner

今晚权重最高之一。

负责：

```text
rolling cross-asset contamination
volatility cross-asset contamination
HP/STL/wavelet/bandpass/CF gating
interpolation leakage
imputation leakage
regime future thresholds
causality registry
prefix invariance
asset isolation
```

---

## Agent 3 — QuantEvaluator Correctness Owner

负责：

```text
QE cache V2
chunk semantics
streaming equivalence
coordinates
LabelBundle timing
multiple-testing bugs
quantile parity
IC performance correctness
```

---

## Agent 4 — q/K Backend Owner

负责：

```text
restore q_executor
physical region
compiler lowering registry
capability authority
q process/thread/state
dtype/null/time
```

如没有真实 q runtime：

必须将 q：

```text
production disabled
```

而不是 mock PASS。

---

## Agent 5 — FactorEngine Core/Backend Parity Owner

负责非 q：

```text
Pandas/Polars/DuckDB
operator ABI
warmup
min_periods
ddof
NaN/Inf
group boundaries
backend evidence
cold-start strict-active recertification
```

---

## Agent 6 — FactorAssets Governance Owner

负责：

```text
assembly/campaign package repair
CompositeGate explicit binding
selection policy
similarity context
representative selection
durable repository/SeenIndex
lifecycle
```

---

## Agent 7 — FactorOptimizer / Research Contamination Owner

当前 FO 仍有 stub。

负责：

```text
production gate = RESEARCH_ONLY until real
external SplitPlan
train/validation/test permissions
sealed test
real legality validator
objective direction
failure taxonomy
determinism
checkpoint/resume
multi-fidelity promotion
```

---

## Agent 8 — Modeling / OOS Owner

负责：

```text
predict() bypass
predict_oos
ApplicationWindow
gap
purge
embargo
artifact identity
created_at semantics
fitted preprocessing
authority split
```

---

## Agent 9 — Data/PIT/Universe Red Team

尽量只写 tests/repros。

主动构造：

```text
财报公告晚于 report period
revision
IPO
退市
指数成分未来加入
ST
停牌
涨跌停
行业变更
corporate action
missing session
buyable/sellable difference
signal after close
```

---

## Agent 10 — Systems / Concurrency / Fault Injection

负责：

```text
registry race
SQLite/WAL
cache race
partial write
process kill
OOM
worker cancellation
q session restart
file descriptor
retry/idempotency
```

---

## Agent 11 — Independent Institutional Red Team

禁止改 production code。

只做：

```text
复现
反例
property tests
metamorphic tests
fresh SHA
review
REOPEN
```

Owner 不能给自己 final approval。

---

# 27. 资源治理

之前已出现过 OOM / exit 137。

继续使用：

```bash
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

不要：

```text
pytest -n auto
```

建议：

```text
最多 2 个 memory-heavy agents 并行
其他静态审计/轻测试 agent 可以并行
```

每个 benchmark 记录：

```text
peak RSS
wall time
CPU time
temporary allocation
```

---

# 28. 6 小时准确执行时间表

## 00:00–00:20 — 冻结基线

Coordinator：

```text
记录 exact SHA
确认 working tree
建立 tasks.jsonl
建立 gate registry
建立 agent ownership
```

Verification Agent：

```text
recursive import
compileall
wheel manifests
public module manifest
```

**此阶段必须先暴露 q_executor / FA package 问题。**

---

## 00:20–01:20 — Stop-the-line Wave 1

### q Agent

```text
恢复 q_executor
禁止 hardcoded PASS
修 q import
production q 暂时 fail closed
```

### FP Agent

```text
rolling/volatility asset isolation
把 HP/STL/wavelet/bandpass/interpolation 等先 production-gate 掉
建立 Prefix Invariance test harness
```

### QE Agent

```text
multiple-testing mapping
QE cache false-hit repro
LabelBundle timing
```

### FA Agent

```text
assembly/campaign dangling imports
CompositeGate metric binding
```

所有问题先有 failing test 再修。

---

## 01:20–02:30 — Stop-the-line Wave 2

### QE

实现：

```text
EvaluationCacheIdentityV2
MetricPartitionSemantics
coordinate preservation
full/chunk parity
```

### Modeling

实现：

```text
no raw predict bypass
dates/context required
created_at vs content hash
gap/purge/embargo
```

### FP

继续：

```text
causality class
imputation fit state
regime historical threshold
```

### q

继续：

```text
single evidence authority
lowering reachability
```

---

## 02:30–03:30 — Architecture Integrity Wave

处理：

```text
package authority split-brain
FactorAssets durable metadata backend
FO production gating
registry freeze/hash
RunManifest
```

建立统一：

```python
RunManifest
```

字段至少：

```text
run_id
commit_sha
code_component_hashes
dependency_lock_hash
data_snapshot_id
universe_id
calendar_id
evaluation_scope_id
split_plan_id
label_spec_id
numeric_policy_id
registry_hashes
seed
backend versions
created_at_utc
```

---

## 03:30–04:30 — Adversarial / Metamorphic Wave

不继续普通 happy-path tests。

跑：

```text
future suffix perturbation
asset permutation
asset block contamination
factor permutation
time chunk permutation
cache same IDs/different values
cache same values/different labels
duplicate ties
NaN/Inf
zero variance
near-zero variance
window edge
min_periods
label horizon overlap
split overlap
revision
universe changes
```

要求找到新的问题继续修。

---

## 04:30–05:10 — Systems / Fault / Scale Wave

运行：

```text
kill worker
cancel run
partial persistence
concurrent registry write
stale cache
corrupt artifact
missing optional dependency
q unavailable
low memory
```

同时小规模 benchmark：

```text
QE: F=100/1000/5000/10000
similarity sparse pipeline
FactorAssets registry 100k metadata
```

正确性不通过的模块禁止做性能优化。

---

## 05:10–05:40 — Clean Distribution / Integration

对每个 production package：

```text
build wheel
fresh venv
wheel-only install
remove monorepo path
recursive production import
public API smoke
contract integration
```

然后：

```text
DA -> FE -> FP -> QE -> FA
```

做最小端到端 toy-data pipeline。

不是 mock return。

---

## 05:40–06:00 — Independent Release Red Team

Agent 11：

```text
fresh exact SHA
不读取 owner 的结论
自己重新跑
```

输出：

```text
PRODUCTION_CANDIDATE
SAFE_BY_GATING
RESEARCH_ONLY
BROKEN
```

按 package/module 分类。

**不允许整个 repo 一个布尔值 READY/NOT READY。**

如果还有 P0：

```text
不写“圆满完成”
直接继续修到时间结束
```

---

# 29. 这次必须改变“Loop 提前结束”行为

刚刚任务 55 分钟结束，还剩 2 小时却停了。

今晚主 Coordinator 要写死：

```text
IF queue empty:
    switch audit methodology
    do NOT terminate
```

方法轮换：

```text
structural
numerical
temporal
PIT
metamorphic
property-based
differential
concurrency
fault injection
packaging
performance
research-integrity
```

连续同一种 audit 没问题不代表无问题。

---

# 30. 新增自动化 Red-Team Generators

建议加入 Hypothesis/property tests。

## FactorPreprocess

自动生成：

```text
多 asset
随机长度
随机 NaN
随机 extreme
随机 future suffix
```

性质：

```text
asset isolation
prefix causality
permutation invariance
```

## QE

性质：

```text
RankIC(a*x+b,y) == RankIC(x,y), a>0
RankIC(-x,y) == -RankIC(x,y)
factor order permutation -> output reorder only
asset order permutation -> same metric
cache off == on
chunk == full
```

## Multiple testing

```text
permutation of hypotheses does not change hypothesis-level adjusted p after realign
reference == statsmodels
```

## Backend parity

```text
Pandas == Polars == DuckDB
```

仅对 certified parameter domain。

---

# 31. Root Release Gate 要真正进入 CI

统一 workflow 建议：

```text
platform-release-gate.yml
```

Jobs：

```text
static-contract
package-build
wheel-isolation
recursive-import
unit-reference
causality
qe-differential
backend-parity
PIT
research-contamination
fault-smoke
integration
release-manifest
```

注意：

GitHub CI 是补充。

服务器上的 canonical：

```text
python scripts/institutional_release_gate.py
```

也必须能一次运行。

---

# 32. VerificationManifest V2

每次验证必须生成：

```json
{
  "commit_sha": "...",
  "working_tree_clean": true,
  "python": "...",
  "os": "...",
  "dependency_lock_hash": "...",
  "wheel_hashes": {},
  "module_manifest_hash": "...",
  "data_snapshot_refs": [],
  "commands": [],
  "junit_hashes": [],
  "benchmark_hashes": [],
  "gate_results": {},
  "reviewer": "...",
  "created_at_utc": "..."
}
```

报告从 JSON 生成。

禁止：

```text
人工先写 PASS
再想办法让代码配上报告
```

---

# 33. Production Capability 不再是一个布尔值

每个 module：

```python
CapabilityState:
    PRODUCTION_CERTIFIED
    PRODUCTION_CANDIDATE
    SAFE_BY_GATING
    RESEARCH_ONLY
    OFFLINE_ONLY
    UNVERIFIED
    BROKEN
```

例如今晚开始时合理状态应类似：

```text
q executor                 BROKEN
q compiler/capability      RESEARCH_ONLY
HP/STL/wavelet/bandpass    OFFLINE_ONLY
factor_optimizer split     RESEARCH_ONLY
FA assembly/campaign       BROKEN
QE basic IC                CANDIDATE
```

修完后逐个升级。

---

# 34. 因子研究层还要补的“百亿私募级”控制

工程正确之后，继续建设：

```text
SearchCampaign ledger
TrialLedger
negative controls
null alphas
family-aware FDR
hierarchical FDR
online FDR
DSR
PBO/CSCV
SPA / Reality Check
specification robustness cube
```

六小时内不要求全部开发完。

但至少：

```text
每一个 trial
每一次参数尝试
每一次 LLM proposal
每一次手工尝试
```

都要进入统一 campaign history。

不能“失败 trial 不记账”。

---

# 35. Test Split 必须物理封印

FactorOptimizer 当前 split contracts 仍是 stub。

最终目标不是：

```python
if split == "test":
    don't use
```

而是：

search process 的 object graph 中根本没有 test values。

设计：

```text
SearchDataView = train + validation only
SealedTestHandle = opaque capability
```

只有：

```text
session.freeze()
```

之后才能由另一个 service/evaluator 解封一次。

所有：

```text
plateau
Pareto
LLM proposal
selection
early stopping
mutation
```

都不可能读取 test metric。

---

# 36. A 股专项 Production Gates

由于你们是 A 股日频多因子，今晚若有数据 adapter 可以做 smoke，必须补：

```text
ST / *ST
suspension
limit-up
limit-down
IPO age
delisted names
board
CSI300/500/1000 history constituent
microcap exclusion
liquidity tiers
```

并明确：

```text
buyable
sellable
```

不能一个 `tradable=True/False`。

例：

```text
涨停：可能 sellable but not buyable
跌停：可能 buyable but not sellable
```

实际规则还需与交易执行口径一致。

---

# 37. 时间语义统一

平台统一五个概念：

```text
observation_time
available_at
decision_time
signal_available_time
first_executable_time
```

财务数据另有：

```text
report_period
publication_time
revision_time
```

任何 feature 必须能回答：

```text
这个值在 decision_time 当时真的知道吗？
```

---

# 38. 数据修订与 Snapshot

同一 2024 年财报在 2026 年回看可能已经被修订。

所以所有 evaluation 必须绑定：

```text
data_snapshot_id
asof/vintage
```

不允许 cache / evaluation 只写：

```text
2018-2026
```

而不说明版本。

---

# 39. Factor Identity 必须分开

至少：

```text
DefinitionIdentity
ValueIdentity
EvaluationIdentity
```

同一个 formula：

```text
close/lag(close)-1
```

definition 相同。

但：

```text
不同 data snapshot
不同 universe
不同 PIT correction
```

值不同。

不能只用一个 factor_id 代表全部身份。

---

# 40. Streaming 实盘健康

你之前要求最新一天 streaming 评估。

今晚如果核心正确性有余力，建立最小：

```text
DailyFactorHealth
```

可在 label 未成熟时算：

```text
coverage
NaN/Inf
distribution drift
rank turnover
exposure drift
breadth
cluster drift
```

T+1/T+5 label 成熟后再补：

```text
daily IC
rolling IC
tail spread
```

禁止未成熟 label 偷看。

---

# 41. 性能目标：先建立基线，不瞎写 SLA

今晚 benchmark：

```text
100 factors
1,000
5,000
10,000
```

100k 如果内存允许做 synthetic smoke。

记录：

```text
RankIC
rank
quantile
coverage
turnover
horizon decay
similarity ANN
```

输出：

```text
factor×date×asset throughput
peak RSS
```

然后再决定正式 SLA。

不要 agent 自己拍脑袋写：

```text
100k factors < 1 sec
```

---

# 42. FactorAssets similarity 不要做 O(K²)

最终：

```text
fingerprint
↓
ANN top M
↓
exact similarity
↓
sparse graph
↓
community
```

fingerprint 可以：

```text
projected rank signal
top-K MinHash
compressed factor-PnL
horizon IC profile
exposure profile
```

不要 100k 全 pair correlation。

---

# 43. Aggregation 不要只做一个 Super Alpha

ModelInputFactorSet 应保留多 representation：

```text
selected raw
family alpha
cluster alpha
residual alpha
latent component
optional global meta-alpha
health/exposure metadata
```

不同 model 可选不同输入。

---

# 44. Security / 权限

团队不同算法负责人看不到全部因子时：

Python wheel 不能提供真正源码保密。

高敏能力：

```text
GlobalSeenIndex
global dedup
hidden proprietary factor library
```

最终放：

```text
service/API
或 compiled extension
```

但这属于后续架构，今晚先保证权限 contract 与数据最小暴露。

---

# 45. 不允许的“修复方式”

今晚任何 agent 不得：

```text
删除 failing test
加 skip/xfail 掩盖 production issue
增大 tolerance 掩盖 semantic mismatch
catch Exception -> return None/NaN/empty
runtime failure -> silent Pandas fallback
把 stub 改名 production
写 PASS 常量
只改 README
只改 evidence SHA
只写新的 FINAL report
mock 真实 integration
使用 monorepo PYTHONPATH 冒充 wheel isolation
 owner 自己 final approve
```

---

# 46. 每个 P0 的 Definition of Done

例如一个 P0 只有同时满足：

```text
1 reproduction
2 failing regression test
3 root cause
4 fix
5 narrow pass
6 package pass
7 cross-package pass
8 exact SHA commit
9 clean worktree
10 clean wheel
11 adversarial test
12 independent reviewer
```

才能 `CLOSED_VERIFIED`。

---

# 47. 今晚优先级队列

严格顺序如下。

## Tier 0 — 立即停线

```text
T0-01 q_executor broken committed import
T0-02 FP cross-asset rolling contamination
T0-03 FP acausal decompositions/interpolation/regime
T0-04 QE multiple-testing mapping/algorithm
T0-05 FA CompositeGate wrong metric binding
T0-06 QE cache silent false hit
T0-07 QE label timing
T0-08 QE chunk mathematical invalidity
T0-09 ModelArtifact predict bypass
T0-10 FA assembly/campaign broken package surface
```

## Tier 1 — 机构级正确性

```text
T1-01 q capability true authority
T1-02 q lowering reachability
T1-03 q physical regions
T1-04 modeling purge/embargo/gap
T1-05 durable FactorAssets metadata
T1-06 runtime registry scope/freeze
T1-07 package authority split-brain
T1-08 exact-sha VerificationManifest
T1-09 root release CI
T1-10 NumericValidityPolicy
```

## Tier 2 — 研究质量/扩展

```text
T2-01 FO sealed test
T2-02 campaign trial ledger
T2-03 advanced selection/Pareto
T2-04 ANN sparse similarity
T2-05 streaming daily health
T2-06 performance
```

---

# 48. 不要被“687/687”迷惑

687 tests passing 只证明：

```text
被收集到的 687 个测试
在当时那个 working tree
按那个环境
通过了
```

它不证明：

```text
所有 production modules 可 import
所有文件进 wheel
all causal
no stale cache
no split contamination
no hidden future data
q runtime exists
latest commit CI passed
```

所以以后报告必须同时写：

```text
tests collected
test manifest hash
modules covered
gates covered
commit SHA
```

---

# 49. 最终夜间输出不要再写“Mission Accomplished”

最终只输出事实：

```text
Exact SHA:
Open P0:
Open P1:
Closed verified:
Production candidates:
Research only:
Broken:
Wheel gates:
PIT gates:
Causality gates:
QE differential gates:
Backend parity:
Fault injection:
Performance baseline:
Known residual risks:
```

如果 Open P0 > 0：

直接：

```text
NOT PRODUCTION CANDIDATE
```

不要庆祝性语言。

---

# 50. 6 小时退出条件

只有满足：

```text
OPEN P0 = 0
OPEN P1 = 0（或明确 user-approved risk acceptance）
all production package wheels PASS
recursive imports PASS
no dangling public exports
QE cache adversarial PASS
QE chunk/full/stream legal parity PASS
Label timing PASS
FP causal prefix PASS
FP asset isolation PASS
multiple-testing reference parity PASS
Model OOS no bypass PASS
PIT smoke PASS
backend mandatory parity PASS
root exact-SHA gate PASS
independent reviewer APPROVED
```

才允许：

```text
PRODUCTION_CANDIDATE
```

如果六小时到了仍不满足：

输出剩余 blockers。

**绝不通过改 wording 强行达标。**

---

# 51. Coordinator 可直接复制使用的 6 小时主 Prompt

```text
You are the chief coordinator of a 6-hour autonomous institutional quantitative
research-platform hardening campaign for the repository quant_projects.

The target is not “more code” and not “more green unit tests.” The target is a
verifiable INSTITUTIONAL_PRODUCTION_CANDIDATE for an A-share daily multi-factor
quant platform.

Audit the exact committed main SHA first. Never trust prior FINAL/DONE/READY
reports without re-verifying current code.

CRITICAL CURRENT FINDINGS THAT MUST BE REPRODUCED FIRST:

1. factor_engine/backend/q_backend/q_executor.py on current committed main appears
   to have lost its real implementation and contains only hard-coded PASS constants,
   while other q modules still import QExecutor/QExecutionFallbackPolicy/get_q_executor.
   Treat as stop-the-line P0.

2. q capability is not truly evidence-driven yet. QPhysicalImplementationRegistry
   is not populated from a real evidence ledger, manual capability sets still exist,
   and at least one “single authority” gate can return PASS without proving authority.

3. q compiler still has generic-lambda branches before special lowerings, creating
   unreachable/incorrect ts_beta, wma, clip and quantile paths. Eliminate backend-owned
   defaults and enforce exactly one reachable lowering per canonical operator.

4. FactorPreprocess has likely future leakage and cross-asset contamination:
   groupby().shift().rolling() without regrouping; full-sample HP/STL/wavelet;
   sosfiltfilt; full-sample interpolation/imputation; future-derived correlation-regime
   thresholds. Implement CAUSAL_PREFIX_INVARIANCE and ASSET_ISOLATION tests immediately.
   Gate unsafe transforms OFFLINE_ONLY/RESEARCH_ONLY instead of pretending they are causal.

5. quant_evaluator multiple_testing has suspected wrong index remapping in BH/Holm.
   Differential-test every method against statsmodels and test permutation invariance.

6. factor_assets CompositeGate accepts metric_bindings but does not enforce them and
   may heuristically evaluate a gate against the wrong metric. Production gates must
   use explicit typed metric IDs and fail closed when evidence is missing.

7. QE cache identity is still semantically incomplete. Bind factor-value identity,
   labels, validity, metric config/implementation, universe, split, snapshot, calendar,
   numeric policy, and backend identity.

8. QE chunk aggregation is not mathematically valid for all metrics. Introduce
   MetricPartitionSemantics and prove full == legal chunked == streaming == parallel.

9. LabelBundle does not yet enforce the complete timing chain. Implement explicit
   signal_available_time/first execution semantics and strict alignment/order.

10. ModelArtifact still exposes predict(X) without an OOS ApplicationWindow, while
    predict_oos is only one guarded path. Eliminate all production OOS bypasses.

11. factor_assets assembly/campaign public imports are still inconsistent/missing and
    the clean-wheel smoke only imports root/core modules. Add recursive wheel import and
    a public module manifest.

12. The latest committed SHA currently has no root CI status evidence visible. Build a
    root release gate that covers every production package.

Use isolated worktrees or non-overlapping write scopes. One coordinator/integrator owns
shared files. Domain owners cannot approve their own fixes.

Required issue lifecycle:

DISCOVERED
-> REPRODUCED
-> FAILING_TEST_ADDED
-> FIXED_LOCAL
-> DOMAIN_TEST_PASS
-> CROSS_PACKAGE_PASS
-> COMMITTED
-> EXACT_SHA_VERIFIED
-> WHEEL_VERIFIED
-> RED_TEAM_APPROVED
-> CLOSED_VERIFIED

If post-commit verification fails, automatically REOPEN.

Do not stop early when the current queue is empty. Switch audit methodology:
structural -> numerical -> temporal -> PIT -> metamorphic -> property-based ->
differential -> concurrency -> fault injection -> packaging -> performance ->
research integrity.

Do not use any of the following to manufacture a pass:
skip/xfail, weakened tolerances, broad-exception-to-NaN, silent fallback, mock evidence,
hard-coded PASS constants, README-only fixes, or owner self-approval.

For any transform claiming causality, require:
- causal prefix invariance
- asset isolation
- permutation invariance where applicable
- explicit fit/apply state if train-fitted

For any evaluation metric, require:
- trusted reference implementation
- edge cases
- cache-off/cache-on equivalence
- full/chunk/stream/parallel equivalence for explicitly legal partitions

For all production artifacts, bind:
commit SHA, code hash, config hash, data snapshot, universe, calendar, split, label spec,
numeric policy, registry generations, dependency versions and random seed.

At the end of six hours, do not write “Mission Accomplished.” Produce a machine-verifiable
capability matrix for every major module:
PRODUCTION_CANDIDATE / SAFE_BY_GATING / RESEARCH_ONLY / OFFLINE_ONLY / BROKEN.

Only declare the whole pre-model platform PRODUCTION_CANDIDATE if all mandatory P0/P1
release gates pass on the exact committed SHA and an independent red-team reviewer approves.
```

---

# 52. 最后一个架构判断

今晚最重要的不是继续增加：

```text
新模型
新算法
新指标
新 factor transform
```

而是先把你们已经写出来的能力做成：

```text
语义可证明
时间可证明
PIT 可证明
坐标可证明
缓存可证明
commit 可证明
安装可证明
故障可恢复
研究过程可审计
```

这批问题一旦关掉，你们平台会从：

```text
“AI 写了很多功能”
```

真正跨到：

```text
“机构研究基础设施”
```

下一阶段再扩模型、组合优化、实时监控、执行系统，才不会建立在不稳定地基上。
