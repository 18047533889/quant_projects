# FactorEngine 模型层与 Filter Layer 当前 HEAD 最终补充整改任务书
## —— 基于 `main@ea02121edf2a9dd19b067dfe88f20d483cc9c0fb` 的剩余问题终审、实现修正、生产证据闭环

> **仓库**：`18047533889/quant_projects`  
> **审计基线**：`main@ea02121edf2a9dd19b067dfe88f20d483cc9c0fb`  
> **日期**：2026-08-12  
> **范围**：FactorEngine `modeling/` + model-like operators + 新增 `signal_filter` / Filter Layer + 二者的生产集成与 evidence。  
> **性质**：这是“当前最新 HEAD 剩余问题”的执行任务书，不是重新推翻已经完成的模型大改。  
> **最高原则**：开始执行时必须重新读取最新 `main`；如果后续提交已经真正修复某项，标记 `ALREADY_FIXED` 并用代码/测试/证据证明，不得重复造第二套 abstraction。

---

# 0. 最高级执行指令

执行 AI 必须采用：

```text
READ LATEST MAIN
      ↓
建立当前问题 ledger
      ↓
每项判定：
OPEN
PARTIAL
ALREADY_FIXED
NOT_APPLICABLE
      ↓
先修 P0 correctness / PIT / state / identity
      ↓
再修 P1 contract / performance / governance
      ↓
执行真实测试
      ↓
重新生成 evidence
      ↓
只在最终代码 SHA 上做 acceptance
```

禁止：

```text
只改文档
只改测试
只改 hard-gate 字符串
为了 PASS 把错误行为写进 oracle
重复实现已经存在的 canonical
把 production 问题降级成 warning
except Exception -> PASS
未执行 evidence -> PASS
旧 SHA evidence 冒充 CURRENT_HEAD
```

---

# 1. 当前已经真正修好的内容：不要重复推翻

本轮重新检查已经确认以下核心方向已有实质代码修正。

除非最新 HEAD 又回归，否则不要重复重写：

## 1.1 PLS

现有实现已经使用：

\[
\beta = W(P^\top W)^{-1}q
\]

并恢复到 raw feature coefficient。

不要退回旧的 `W.T @ P` 公式。

---

## 1.2 ElasticNet

当前实现已体现：

\[
\frac{1}{2N}\|y-X\beta\|^2
+\alpha l_1\|\beta\|_1
+\frac{\alpha(1-l_1)}{2}\|\beta\|_2^2
\]

并包含：

```text
raw-space coefficient restoration
sample-count invariant alpha semantics
non-convergence fail closed
```

不要回退。

---

## 1.3 Model Evaluation

当前已经将主 RankIC 改为：

```text
per-date cross-sectional Spearman
→ aggregate across dates
```

同时保留 pooled correlation 作为单独诊断。

Turnover 已使用真实 `security_id` 而不是每日局部 row index。

不要重新引入 pooled RankIC 作为默认模型选择指标。

---

## 1.4 Trainer governance

当前代码已经有：

```text
validation-backed hyperparameter selection
feature availability fail-closed option
sample weight wiring
actual transformed-finite cohort
trainer governance
```

本轮只补剩余问题。

---

# 2. 当前总体结论

当前模型层已从“架构和数学大修期”进入：

```text
代码语义精修
+
Filter Layer 成熟化
+
逐 canonical 生产认证
+
最终 CURRENT_HEAD evidence closure
```

但当前不能宣称生产收官。

原因包括：

```text
MODEL_CURRENT_HEAD.json 仍绑定旧 SHA 6fd71595...
当前 main 已经是 ea02121e...

final_direct_use_ready_count 仍为 0

direct-use model:
parameter-domain evidence 仍 FAIL
oracle / causality / missing / unit / optimized parity 等仍有 NOT_RUN
```

因此本轮目标不是“继续加模型数量”，而是：

```text
修完 Filter 新实现
+
修完模型/Filter 交界
+
跑完真实 per-canonical evidence
+
让生产准入建立在真实 CURRENT_HEAD 上
```

---

# 3. P0 总表

| ID | 问题 | 严重度 |
|---|---|---|
| MF-P0-001 | Model evidence 已再次 stale | P0 |
| MF-P0-002 | final_direct_use_ready 仍为 0，不能伪装闭环 | P0 |
| MF-P0-003 | direct-use model parameter-domain evidence 0/88 | P0 |
| MF-P0-004 | `_maturity_cutoff` 无未来 calendar 时 fail-open 到过早日期 | P0 |
| MF-P0-005 | Artifact `training_cutoff` 文档语义与 label maturity 语义冲突 | P0 |
| FL-P0-001 | AdaptiveDeadband 当前 delta 参与自己的阈值估计 | P0 |
| FL-P0-002 | RobustEMA warmup 以 `scale_floor` 裁正常 innovation，启动近冻结 | P0 |
| FL-P0-003 | KAMA `min_periods` 死参数 | P0 |
| FL-P0-004 | KAMA 对 int 参数仍有 float→int silent coercion | P0 |
| FL-P0-005 | Local-linear 在缺失窗口压缩时间轴 | P0 |
| FL-P0-006 | Butterworth cutoff period 归一化频率口径错误/歧义 | P0 |
| FL-P0-007 | Cost-aware deadband 单位不闭合 | P0 |
| FL-P0-008 | Cost-aware slew 公式单位不闭合且默认可能几乎失效 | P0 |
| FL-P0-009 | invalid/missing cost 当前 fail-open | P0 |
| FL-P0-010 | RankDeadband 的“held rank”与“held raw value”语义混杂 | P0 |
| FL-P0-011 | Filter unit metadata 大量固定成 dimensionless/level | P0 |
| FL-P0-012 | Filter certified parameter 文档与 runtime/ParamSpec 不一致 | P0 |
| FL-P0-013 | Filter `checkpointable=True` 可能只是声明，runtime state 未证明可恢复 | P0 |
| FL-P0-014 | Rank percentile helper 为 O(T×N²)，A 股截面规模不可接受 | P0 |
| FL-P0-015 | Filter contract 将 availability lag 与真实 signal-response lag 混为一谈 | P0 |
| FL-P0-016 | 多个 filter warmup 是参数依赖，但 contract 用静态 int/0 | P0 |
| FL-P0-017 | Filter missing policy 未成为统一 contract | P0 |
| FL-P0-018 | Production model 使用 filtered score 时证据/artifact identity 未完整绑定 | P0 |

---

# 4. MF-P0-001 — 最终 Model Evidence 必须重新绑定真正 CURRENT HEAD

当前基线：

```text
repository HEAD = ea02121edf2a...
```

但现有：

```text
factor_engine/docs/evidence/model_operators/MODEL_CURRENT_HEAD.json
```

仍绑定：

```text
6fd715953558...
```

因此旧：

```text
MODEL_CURRENT_HEAD_EVIDENCE_FRESH = PASS
```

在当前 HEAD 上已经不能成立。

## 修法

最终 evidence generator 必须在**所有代码改完后的 final SHA** 运行：

```python
repo_head = git_rev_parse("HEAD")
evidence_head = generated.commit_sha

assert evidence_head == repo_head
```

如果不同：

```text
FAIL
```

不得：

```text
PASS with stale evidence
```

## 进一步要求

建议记录：

```text
repository_head_sha
modeling_component_tree_hash
cleaned_model_operator_tree_hash
filter_component_tree_hash
evidence_generator_hash
```

---

# 5. MF-P0-002 — `final_direct_use_ready=0` 必须真实解决，而不是只改数字

当前 evidence 仍显示：

```text
final_direct_use_ready_count = 0
```

只有在全部必要 evidence PASS 后才允许：

```text
final_direct_use_ready = true
```

至少包括：

```text
typed inputs
parameter domain
oracle
causality
missing/gap
unit
optimized/reference parity
state contract if stateful
time-shard/checkpoint parity if stateful
negative control where applicable
```

---

# 6. MF-P0-003 — 88 个 direct-use model 的 Parameter Domain 必须真正执行

对每个 canonical 建 parameter-domain matrix：

```text
valid default
valid certified low
valid certified high

below min
above max
wrong dtype
bool-as-int
nan
inf

invalid relational combinations
```

验收：

```text
88/88 executed
88/88 contract-consistent
```

---

# 7. MF-P0-004 — `_maturity_cutoff` 缺未来交易日时必须 FAIL CLOSED

如果 horizon > 0 且：

```text
calendar 无法覆盖 maturity target
```

则：

```text
raise LabelMaturityUnavailableError
```

或返回：

```text
UNAVAILABLE
```

禁止自动生成 production artifact。

---

# 8. MF-P0-005 — 区分三个以上时间概念，不要继续混成 `training_cutoff`

建议正式拆分：

```text
final_fit_anchor_end
label_maturity_cutoff
fit_completed_at
artifact_available_at
activation_at
```

定义：

```text
artifact_available_at
>= max(label_maturity_cutoff, fit_completed_at)

activation_at
>= artifact_available_at
```

---

# 9. FL-P0-001 — 修复 AdaptiveDeadband 自包含阈值

正确顺序：

```python
scale_t = robust_scale(delta_history)
band_t = band_mult * scale_t

update_or_hold_current()

delta_history.append(delta_t)
```

Hard Gate：

```text
FILTER_ADAPTIVE_THRESHOLD_EXCLUDES_CURRENT_INNOVATION
```

---

# 10. FL-P0-002 — RobustEMA warmup 不能用接近零的 scale 裁掉正常变化

Warmup 阶段应使用：

```text
ordinary EMA
```

或：

```text
pass-through
```

成熟后才使用 robust innovation clipping。

建议新增：

```text
min_scale_obs
warmup_policy
```

---

# 11. FL-P0-003 — KAMA `min_periods` 必须删掉或真的生效

合法方案只有两个：

```text
A. 删除无效参数
B. 明确定义 runtime 语义并真正使用
```

禁止保留 dead parameter。

---

# 12. FL-P0-004 — KAMA 禁止 silent float→int coercion

对：

```text
er_window
fast_period
slow_period
min_periods
```

使用 strict binder：

```text
bool reject
float reject
int accept
```

runtime kernel 不再 `int(...)` 静默强转。

---

# 13. FL-P0-005 — Local Linear Smoother 不能在缺失时压缩时间轴

应使用真实 trailing offsets：

```python
offsets = np.arange(window_length)
mask = finite(window_data)

xpos = offsets[mask]
y = window_data[mask]
```

而不是：

```python
np.arange(len(finite_values))
```

---

# 14. FL-P0-006 — Butterworth cutoff period 必须有唯一数学定义

推荐：

```python
fs = 1.0
cutoff_hz = 1.0 / cutoff_period
butter(order, cutoff_hz, fs=1.0, ...)
```

并增加 frequency-response oracle。

---

# 15. FL-P0-007 — CostAwareDeadband 必须 unit-compatible

建立：

```text
CostAwareSignalContract
```

至少区分：

```text
EXPECTED_RETURN_SPACE
RANK_SPACE
ZSCORE_SPACE
TARGET_WEIGHT_SPACE
```

Production 优先只允许同单位空间。

---

# 16. FL-P0-008 — CostAwareSlew 当前公式必须重做

不再使用：

\[
limit_t = slew\_mult / (cost_t+\epsilon)
\]

推荐：

\[
limit_t=
\frac{base\_limit}{1+k c_t^{norm}}
\]

或者在 target-weight space 使用 liquidity capacity。

---

# 17. FL-P0-009 — Missing/Invalid Cost 不允许 fail-open

默认应：

```text
FAIL_CLOSED
NO_UPDATE
OUTPUT_NAN
```

不能默认 zero-cost / unlimited update。

---

# 18. FL-P0-010 — RankDeadband 输出语义拆分

推荐：

```text
state_rank_deadband
→ held_rank_pct
```

如果还需要保存 raw：

```text
state_rank_triggered_raw_hold
```

作为另一 canonical。

---

# 19. FL-P0-011 — Filter Output Unit 必须传播输入单位

大部分平滑器：

```text
output unit = input unit
```

Rank filter：

```text
dimensionless percentile
```

State filter：

```text
state/binary
```

Cost-aware：

```text
显式单位兼容
```

---

# 20. FL-P0-012 — Certified Parameter Grid 必须由机器强制

例如：

```text
Hampel n_sigma ∈ {2.5,3,4}
RollingMedian window ∈ {3,5,7}
Butterworth order ∈ {2,3,4}
```

Production miner 只能看到 certified values。

---

# 21. FL-P0-013 — `checkpointable=True` 必须对应真实 checkpoint API

每个 stateful filter 定义：

```text
StateSchema
serialize_state()
deserialize_state()
update_one()
resume()
```

Hard Gate：

```text
FILTER_CHECKPOINTABLE_MEANS_RESUME_PARITY_PASS
```

---

# 22. FL-P0-014 — `_cs_rank_pct` O(T×N²) 必须优化

无 group：

```text
一次 rankdata/argsort per date
```

有 group：

```text
一次 group partition + rank per group
```

复杂度目标：

```text
O(T × N log N)
```

---

# 23. FL-P0-015 — 拆分 Availability Lag 与 Signal Response Lag

改成：

```text
availability_lag_bars
response_delay_contract
```

Response evidence：

```text
step_response_50pct_bars
step_response_90pct_bars
group_delay
```

---

# 24. FL-P0-016 — Warmup 必须支持参数依赖

新建：

```text
WarmupContract
```

支持：

```text
EXACT_ROWS(param)
CONTIGUOUS_ROWS(param)
RECURSIVE_SEED
STATISTICAL_MATURITY
```

---

# 25. FL-P0-017 — Missing Policy 纳入 FilterContract

建议：

```text
BREAK_RESET
OUTPUT_NAN_HOLD_STATE
OUTPUT_NAN_ADVANCE_STATE
HOLD_OUTPUT
PREDICT_ONLY
FAIL_CLOSED
```

每个 filter 必须显式声明。

---

# 26. FL-P0-018 — Filtered Model Score 必须成为 Artifact 的正式组成

Artifact manifest 增加：

```text
filter_chain_spec
filter_chain_hash
filter_semantic_versions
filter_parameter_policy
filter_state_schema_version
```

同时保存：

```text
raw_prediction
filtered_prediction
```

线上使用 filtered score，则 admission 必须评估 filtered score。

---

# 27. FilterContract 当前还需要扩展

建议最终字段：

```text
role
causal
current_observation_role
reference_cutoff

stateful
checkpoint_schema
time_shard_policy

warmup_contract
missing_input_policy
gap_time_policy

input_unit_contract
output_unit_rule
scale_policy

availability_lag_bars
response_delay_evidence_required

jump_policy
turnover_control

parameter_policy
production_surface
```

---

# 28. Filter Role 需要补充家族

建议新增：

```text
COST_AWARE_CONTROL
CONFIDENCE_AWARE_CONTROL
CHANGE_POINT_AWARE
CROSS_SECTIONAL_STABILIZER
```

---

# 29. AdaptiveDeadband 参数角色需要修正

建议：

```text
scale_window → SUPPORT_POLICY / ESTIMATOR_HORIZON
scale_method → ESTIMATOR_POLICY
band_mult → TURNOVER_POLICY
```

---

# 30. Rank/Quantile group 输入不要强制 float

必须支持：

```text
int
string
categorical
```

并绑定 PIT membership vintage。

---

# 31. Rank Percentile tie policy 必须版本化

建立：

```text
RankConvention
```

包括：

```text
tie_method
denominator
range convention
missing policy
group policy
```

---

# 32. Quantile Hysteresis Missing State 不应隐式 reset

缺失时必须显式选择：

```text
RESET_OUT
HOLD_STATE_OUTPUT_NAN
HOLD_STATE_HOLD_OUTPUT
FAIL_CLOSED
```

---

# 33. Confidence 输入禁止 silent clamp

Production 对：

```text
confidence < 0
confidence > 1
```

应 fail closed，不要默认 clip。

---

# 34. Uncertainty <= 0 不应默认“无约束更新”

应：

```text
FAIL_CLOSED / OUTPUT_NAN / HOLD_STATE
```

---

# 35. Hampel 在 MAD=0 时的 `scale_floor` 语义要改

纯 numerical epsilon 不应决定经济 jump。

增加：

```text
zero_scale_policy
```

例如：

```text
RECENT_RANGE_FLOOR
DELAYED_CONFIRMATION
BYPASS_IF_ZERO_SCALE
```

---

# 36. Hampel `replacement="median"` 建议 research-only

Production 默认：

```text
clip
```

`median` 只保留 research / DataQuality confirmed spike 场景。

---

# 37. Median3 的 Jump Policy 与 Lag Contract 要修

应拆：

```text
availability lag = 0
response lag = evidence measured
```

并使用更准确 jump policy，例如：

```text
MEDIAN_REJECTION
```

---

# 38. RollingMedian 参数只允许奇数窗口

Production 建议只允许：

```text
3 / 5 / 7
```

---

# 39. SuperSmoother ParamSpec 与状态契约补齐

建议：

```text
period ∈ {5,10,20,40}
```

并完成：

```text
state schema
missing-gap
response evidence
```

---

# 40. SuperSmoother / Butterworth 初始化瞬态必须认证

比较：

```text
steady-state zi
first-value zi
zero-state
```

选择 production policy，并报告 startup transient bars。

---

# 41. Butterworth missing input “freeze state” 需要明确 gap-time 语义

明确：

```text
COMPRESSED_OBSERVATION_TIME
PHYSICAL_BAR_TIME
RESET_ON_GAP
```

不要隐式 freeze 无限久。

---

# 42. RobustEMA 的 innovation buffer 语义必须版本化

明确使用：

```text
RAW_INNOVATION_SCALE
```

还是：

```text
CLIPPED_INNOVATION_SCALE
```

---

# 43. AdaptiveDeadband scale 需要 zero-scale policy

避免历史 delta 全零时：

```text
band=0
```

导致微小数值噪声频繁更新。

---

# 44. AdaptiveSlew warmup pass-through 需要 boundary evidence

比较：

```text
PASS_THROUGH_THEN_ENABLE
SEED_SCALE_EARLY
GRADUAL_ENABLE
```

选择一个认证 policy。

---

# 45. L1 Turnover Prox 需要 scale-aware variant

可新增：

```text
state_adaptive_l1_turnover_prox
```

例如：

\[
\lambda_t = k \times MAD(\Delta x)
\]

或在 rank/target space 运行。

---

# 46. L2 Partial Adjustment 与 EMA 同构，不能算新机制多样性

标记：

```text
semantic_equivalence_family = EMA_PARTIAL_ADJUSTMENT
```

---

# 47. Filter semantic dedup 必须进入 mining governance

建立：

```text
filter_semantic_family
parameter_normalized_filter_identity
```

---

# 48. Filter Grammar 必须限制嵌套深度

Production：

```text
despike <= 1
smooth <= 1
rank/normalize <= 1
turnover-control <= 2
```

---

# 49. Filter Composition Order 建立 canonical policy

默认：

```text
DESPIKE
→ LOW_PASS / ADAPTIVE_LOW_PASS
→ CS NORMALIZE/RANK
→ HYSTERESIS
→ RATE/COST CONTROL
```

---

# 50. 新 Filter 不应自动视为 Production Direct-Use

必须区分：

```text
RESEARCH
EXTENDED
PRODUCTION_CERTIFIED
```

---

# 51. FilterEvidence 必须单独建立

建议目录：

```text
factor_engine/docs/evidence/filter_layer/
```

输出：

```text
FILTER_CURRENT_HEAD.json
FILTER_CANONICAL_LEDGER.csv
FILTER_HARD_GATES.json
FILTER_RESPONSE_EVIDENCE.json
FILTER_TURNOVER_EVIDENCE.parquet
FILTER_BACKEND_PARITY.json
FILTER_FINAL_ACCEPTANCE_REPORT.md
```

---

# 52. Filter Evidence：因果负控

对 t 之后输入 poison：

```text
<= t 输出完全不变
```

必须重跑 authoritative runtime。

---

# 53. Filter Evidence：Step Response

记录：

```text
first response bar
50% response bars
90% response bars
settling time
overshoot
```

---

# 54. Filter Evidence：Impulse / Spike Response

记录：

```text
peak attenuation
residual tail
recovery bars
```

---

# 55. Filter Evidence：Regime Jump Preservation

确保真实永久 level shift 不被长期压回旧值。

---

# 56. Filter Evidence：Trend Preservation

测试：

```text
linear trend
accelerating trend
```

记录：

```text
slope retention
endpoint bias
turning-point delay
```

---

# 57. Filter Evidence：Missing Gap

测试：

```text
single NaN
3-bar gap
20-bar gap
```

检查：

```text
output
state
resume
physical-time semantics
```

---

# 58. Filter Evidence：Scale Equivariance / Unit

对保持单位的 filter 检查：

```text
filter(10x) ≈ 10 filter(x)
```

---

# 59. Filter Evidence：Turnover Reduction

真实 A 股样本至少报告：

```text
raw rank autocorrelation
filtered rank autocorrelation

raw top-decile turnover
filtered top-decile turnover

raw target L1 change
filtered target L1 change
```

---

# 60. Filter Evidence：Alpha Retention

报告：

```text
raw mean daily RankIC
filtered mean daily RankIC

raw RankICIR
filtered RankICIR

IC retention
sign consistency
```

sign flip：

```text
FAIL
```

---

# 61. Filter Evidence：成本后表现

报告：

```text
gross spread
turnover
assumed cost
net spread
net Sharpe
```

---

# 62. Filter Evidence：Pareto Frontier

先满足：

```text
IC retention >= threshold
lag <= threshold
coverage >= threshold
```

再比较：

```text
turnover reduction
net performance
```

---

# 63. Filter 参数搜索必须走 Validation

所有滤波参数进入：

```text
SearchExposureLedger
```

不能在 final test 上选择。

---

# 64. Filter 参数不能制造冷启动库爆炸

identity 分为：

```text
economic_factor_id
conditioning_policy_id
```

---

# 65. Filter 应采用两阶段搜索

```text
Stage A:
挖 raw economic factor

Stage B:
对通过 basic gate 的 factor
试少数 certified filter policies
```

---

# 66. Filter State 必须进入 Incremental/Revision Replay

历史 source revision 后：

```text
从最早受影响 state 开始重放
```

---

# 67. Stateful Filter 的 ChangeImpact 需要 transitive tail

递归状态受历史变更影响时，后续状态要继续传播，不能只扩 rolling window。

---

# 68. Filter Backends：不要做伪 Polars

```text
stateless rolling/rank
→ genuine Polars expression where suitable

recursive filter
→ shared optimized array/state kernel
```

禁止 pandas bridge 冒充 Polars native。

---

# 69. P0 性能优化建议

## Hampel / RollingMedian

尽量使用：

```text
shift(1)
rolling median
rolling robust scale
```

减少 Python 三重循环。

## LocalLinear

使用 rolling sufficient statistics：

```text
Σx
Σt
Σt²
Σtx
n
```

做到近 O(T×N)。

## AdaptiveDeadband

用 deque/ring buffer，不用 list `pop(0)`。

## Rank

必须 O(N log N) / date。

---

# 70. Benchmark 尺度

至少：

```text
252 × 1,000
1,260 × 3,000
2,520 × 5,000
```

记录：

```text
wall time
peak memory
rows×stocks/sec
```

---

# 71. Model Evidence 仍需跑完整 Oracle

每个 direct-use canonical 都要有：

```text
oracle
equivalence
invariant
```

不能只靠 family-level test。

---

# 72. Optimized vs Reference parity 不能 NOT_RUN

生产 backend 都必须比较：

```text
NaN mask
finite values
state progression
edge cases
```

---

# 73. Stateful model/filter 都必须做 batch/chunk parity

包括：

```text
Kalman
recursive filters
stateful events
```

---

# 74. Model Negative Controls 必须在最终 HEAD 重跑

至少：

```text
future poison
label shuffle
scaler poison
universe poison
timestamp poison
```

---

# 75. `block_aware_ic` 不得冒充独立统计推断

简单 H-day block aggregation 只能做 reporting。

统计推断至少支持：

```text
HAC/Newey-West
stationary/block bootstrap
non-overlapping anchor subsample
```

---

# 76. Validation/Test Adequacy 最终也要进入 evidence

每 fold 报：

```text
validation valid dates
median stocks/date
label coverage
test valid dates
test coverage
```

---

# 77. Production CI 当前未被证明

最终不得写：

```text
CURRENT_HEAD CI PASS
```

除非真的获得 CI evidence。

---

# 78. Acceptance Report 旧 SHA 文本必须全部再生成

所有：

```text
CURRENT_HEAD
FINAL
ACCEPTANCE
```

报告必须自动绑定 final SHA。

---

# 79. Evidence generator 本身需要自证

记录：

```text
generator code hash
runtime versions
timestamp
input ledger hash
test commands
```

---

# 80. Filter 与 Model Semantic Registry 要打通

统一描述：

```text
raw predictor semantic
filter chain semantic
final score semantic
```

---

# 81. PredictionOutputContract 增加 filtered 状态

建议：

```text
raw_score_semantic
final_score_semantic
filter_chain_hash
unit
rank_or_absolute
higher_is_better
calibrated
```

---

# 82. Filter State 与新 Model Artifact 切换策略

支持：

```text
RESET
WARM_START_IF_COMPATIBLE
BRIDGE
```

---

# 83. Model Retrain Boundary 加 filtered-score discontinuity test

比较：

```text
rank correlation
scale ratio
mean shift
top-k overlap
```

---

# 84. Cost-aware Filter 与 Portfolio 层职责边界

FactorEngine 负责：

```text
stable signal
rank-deadband signal
cost-aware signal-change indicator
```

最终：

```text
target-weight turnover
impact constraints
no-trade region
```

仍应归 Portfolio/Optimizer。

---

# 85. FactorAssembly 本轮只建立接口，不要硬塞进 cleaned_operators

预留：

```text
feature_set_id
factor_set_hash
```

但不要把千/万因子聚类治理塞进 Filter operator。

---

# 86. 新 Hard Gates

建议新增并真正执行：

```text
MODEL_CURRENT_HEAD_EVIDENCE_EXACT_SHA
MODEL_DIRECT_USE_READY_DERIVED_FROM_EXECUTED_EVIDENCE
MODEL_ALL_DIRECT_USE_PARAMETER_DOMAIN_PASS
MODEL_LABEL_MATURITY_CALENDAR_FAIL_CLOSED
MODEL_ARTIFACT_TIME_FIELDS_SEMANTICALLY_DISTINCT

FILTER_ADAPTIVE_THRESHOLD_EXCLUDES_CURRENT_INNOVATION
FILTER_ROBUST_EMA_WARMUP_NOT_EPSILON_FROZEN
FILTER_ZERO_DEAD_PARAMETERS
FILTER_ZERO_SILENT_INTEGER_COERCION

FILTER_LOCAL_LINEAR_PRESERVES_PHYSICAL_BAR_OFFSETS
FILTER_BUTTERWORTH_CUTOFF_PERIOD_ORACLE_PASS

FILTER_COST_AWARE_UNITS_COMPATIBLE
FILTER_COST_MISSING_FAILS_CLOSED
FILTER_COST_SLEW_DIMENSIONALLY_VALID

FILTER_RANK_DEADBAND_OUTPUT_SEMANTIC_UNAMBIGUOUS
FILTER_ALL_OUTPUT_UNITS_DERIVED_CORRECTLY
FILTER_CERTIFIED_GRIDS_MACHINE_ENFORCED

FILTER_ALL_STATEFUL_HAVE_REAL_STATE_SCHEMA
FILTER_ALL_CHECKPOINTABLE_RESUME_PARITY_PASS
FILTER_RANK_COMPLEXITY_SUBQUADRATIC

FILTER_AVAILABILITY_LAG_SEPARATE_FROM_RESPONSE_DELAY
FILTER_ALL_WARMUP_PARAMETER_DEPENDENT_WHERE_REQUIRED
FILTER_ALL_HAVE_MISSING_POLICY

FILTER_PRODUCTION_MODEL_BINDS_FILTER_CHAIN_IDENTITY
FILTER_PRODUCTION_FILTERED_SCORE_HAS_OOS_EVIDENCE

FILTER_ZERO_PARAMETER_CLONE_COUNTED_AS_ECONOMIC_DIVERSITY
FILTER_PRODUCTION_NESTING_DEPTH_BOUNDED
FILTER_PRODUCTION_COMPOSITION_ORDER_CERTIFIED

FILTER_ALL_PRODUCTION_HAVE_RESPONSE_EVIDENCE
FILTER_ALL_PRODUCTION_HAVE_ALPHA_RETENTION_EVIDENCE
FILTER_ALL_TURNOVER_FILTERS_HAVE_TURNOVER_EVIDENCE

FILTER_ALL_OPTIMIZED_PATHS_REFERENCE_PARITY
FILTER_NO_FAKE_POLARS_BACKEND
```

---

# 87. 必须新增/强化测试清单

```text
test_adaptive_deadband_excludes_current_delta_from_scale
test_robust_ema_warmup_uses_unclipped_or_ordinary_ema
test_kama_min_periods_not_dead_or_removed
test_kama_rejects_float_integer_params
test_local_linear_missing_preserves_offsets
test_butterworth_cutoff_period_frequency_response
test_cost_deadband_rejects_unit_mismatch
test_cost_slew_dimensionless_formula
test_invalid_cost_fails_closed
test_rank_deadband_outputs_held_rank
test_filter_units_preserve_input
test_certified_grid_rejects_unapproved_values

test_robust_ema_checkpoint_resume_parity
test_kama_checkpoint_resume_parity
test_butterworth_checkpoint_resume_parity
test_adaptive_deadband_checkpoint_resume_parity
test_rank_deadband_checkpoint_resume_parity
test_adaptive_slew_checkpoint_resume_parity
test_l1_prox_checkpoint_resume_parity
test_l2_partial_checkpoint_resume_parity

test_filter_single_gap_policy
test_filter_long_gap_policy
test_quantile_hysteresis_missing_does_not_implicit_exit
test_confidence_invalid_policy
test_uncertainty_invalid_policy

test_filter_impulse_response
test_filter_permanent_step_response
test_filter_linear_trend_preservation
test_filter_regime_jump_preservation
test_filter_v_reversal_response

test_rank_deadband_no_quadratic_peer_loop
benchmark_rank_5000_stocks
benchmark_filter_core_2520x5000

test_model_current_head_exact_sha
test_maturity_cutoff_requires_future_calendar
test_artifact_not_available_before_label_maturity
test_direct_use_parameter_domain_all_pass
test_direct_use_oracle_all_executed
test_direct_use_causality_all_executed
test_direct_use_unit_all_executed
test_stateful_chunk_parity_all_executed
test_optimized_reference_parity_all_executed
```

---

# 88. 推荐执行顺序

## Phase 1

先修：

```text
模型时间语义
maturity cutoff
evidence freshness
```

## Phase 2

修 Filter P0：

```text
AdaptiveDeadband
RobustEMA
KAMA
LocalLinear
Butterworth
CostAware
RankDeadband
```

## Phase 3

重构：

```text
FilterContract
unit
warmup
missing
lag
state schema
parameter policy
```

## Phase 4

完成：

```text
checkpoint
incremental
revision replay
chunk parity
```

## Phase 5

性能：

```text
rank O(N²) → O(N log N)
rolling kernels
local-linear sufficient stats
ring buffers
```

## Phase 6

模型/Filter 生产集成：

```text
filter chain artifact identity
filtered score evaluation
deployment state migration
```

## Phase 7

全量 evidence：

```text
model per-canonical
filter per-canonical
CURRENT_HEAD acceptance
```

---

# 89. P0 完成标准

不得宣称 production ready，除非：

```text
[ ] maturity cutoff fail-closed
[ ] adaptive deadband self-threshold 修复
[ ] robust EMA warmup 修复
[ ] KAMA dead/silent params 修复
[ ] local-linear physical time 修复
[ ] Butterworth cutoff oracle 通过
[ ] cost-aware unit/invalid policy 修复
[ ] rank deadband semantic 拆清
[ ] unit propagation 正确
[ ] certified parameter grid 强制
[ ] checkpoint 真实可恢复
[ ] rank O(N²) 消除
[ ] missing/warmup/lag contract 完成
[ ] final filtered model identity/evidence 完成
```

---

# 90. 最终 Evidence Definition of Done

Model：

```text
[ ] MODEL_CURRENT_HEAD = final repository SHA
[ ] direct-use denominator 正确
[ ] parameter domain 全执行
[ ] oracle 全执行
[ ] causality 全执行
[ ] missing 全执行
[ ] unit 全执行
[ ] state parity 全执行
[ ] optimized/reference parity 全执行
[ ] negative controls 全执行
[ ] final_direct_use_ready 由真实证据推导
```

Filter：

```text
[ ] FILTER_CURRENT_HEAD = final repository SHA
[ ] 每个 production filter contract 完整
[ ] 每个 stateful filter resume parity
[ ] 每个 production filter unit 正确
[ ] 每个 production filter parameter domain certified
[ ] causal future-poison
[ ] missing-gap
[ ] step response
[ ] impulse response
[ ] jump preservation
[ ] alpha retention
[ ] turnover evidence
[ ] backend parity
```

---

# 91. 最终 Acceptance Report 必须诚实

只允许：

```text
PASS
FAIL
NOT_RUN
NOT_APPLICABLE
```

禁止：

```text
“结构上等于 PASS”
“预计可以”
“代码看起来支持”
“taskbook 已完成所以 PASS”
```

---

# 92. 最终建议目录

建议：

```text
factor_engine/
    signal_filter/
        contracts.py
        state.py
        evidence.py
        evaluation.py
```

或暂时继续放：

```text
cleaned_operators/filter_*.py
```

但共享契约必须收口。

---

# 93. 本轮完成前不要再大规模增加新模型/滤波器

优先解决：

```text
正确性
状态恢复
PIT
性能
evidence
```

不是数量。

---

# 94. Production Filter Bank 建议先冻结

```text
ts_hampel_filter_causal
ts_median3_causal
ts_robust_ema
ts_kama
ts_super_smoother
ts_butterworth_lowpass_causal
ts_causal_local_linear_smoother

state_adaptive_deadband
state_rank_deadband
state_quantile_hysteresis
state_adaptive_slew_limit

state_l1_turnover_prox
state_l2_partial_adjustment

state_confidence_weighted_ema
state_uncertainty_deadband
```

Cost-aware 两个只有 unit contract 修好后再 production。

---

# 95. 给 coding agent 的最终完成指令

不要做到：

```text
pytest 一部分绿
```

就停止。

必须做到：

```text
1. 最新 main 全量读取
2. 所有 OPEN/PARTIAL 问题修复
3. 重复 abstraction 清理
4. 真实 runtime path 测试
5. state resume parity
6. model per-canonical evidence
7. filter per-canonical evidence
8. 性能 benchmark
9. final SHA evidence regeneration
10. acceptance report
11. 最终 ledger 只剩：
    PASS / intentional research-only / genuinely NOT_APPLICABLE
```

如果还有：

```text
FAIL
NOT_RUN
stale SHA
fake checkpoint
uncertified production filter
```

不得宣布整改完成。

---

# 96. 最终目标

最终系统必须能机器化回答：

```text
这个输出用的哪个模型？
哪个 filter chain？
每个参数是什么？
单位是什么？
当前 bar 可不可用？
state 从哪里恢复？
缺失时发生什么？
真实 jump 会不会被抹掉？
换手到底降了多少？
IC 保留多少？
旧/新 artifact 怎么切换？
当前 evidence 是否真的属于这个 HEAD？
```

这些都能回答，模型层与 Filter Layer 才算真正收口。
