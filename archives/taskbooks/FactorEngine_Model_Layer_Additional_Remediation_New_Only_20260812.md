# FactorEngine 模型层新增补充整改任务书
## —— 仅包含前序审计未提出的新问题；禁止重复整改已发送问题

> **仓库**：`18047533889/quant_projects`  
> **审计基线**：本轮审计时 `main` 最新可见提交为 `4b1577fce14c097fba48619885b38d6c596242eb`。  
> **重要要求**：执行 AI 开始工作前必须重新读取最新 `main`，不得假定上述 SHA 仍是最新。  
> **用途**：本文件是“新增补充整改项”，不是前两份模型整改文档的重写版。  
> **执行原则**：本文件只处理此前没有明确提出过的问题；前序问题已经分批发送给其他 AI 并行整改，禁止重复开工、重复改语义、重复创建 canonical。

---

# 0. 本文件明确排除的前序问题

以下问题此前已经提出过，**本轮不要重复整改**。如果当前 HEAD 已有并行改动，只需要避免冲突。

包括但不限于：

- PLS `P'W / W'P` 系数公式问题；
- ElasticNet 标准化空间系数还原、alpha 目标函数尺度、non-convergence fail-closed；
- `train_plus_validation` artifact `training_cutoff / available_at` 未来函数；
- PCR/PLS/ElasticNet 默认 SampleAdequacy contract family 映射；
- SampleAdequacy 已定义字段未 wiring；
- free-parameter count 旧实现问题；
- `(stock,date)` 唯一性与 `stock_col` 基础校验；
- 无 validation 时 train→test purge 的旧问题；
- Markov TimingKind；
- First Passage 参数 clamp / unit / scale horizon / doc；
- Kalman checkpoint、q/r 尺度、warmup；
- GARCH/HAR 已讨论问题；
- AR / rolling regression window/min-history；
- `ts_ar_coefficient` / `ts_regression_slope` lane；
- legacy PCR/PLS/ENet/Regime/MoE lane；
- DMD/Hankel/SSA/Matrix Profile 已讨论治理；
- KNN DecisionClock / universe；
- TE surrogate/search policy；
- Geyer IMS/IPS；
- event_fano closure/runtime；
- state_episode / directional-change checkpoint；
- beta-break unit；
- Kramers–Moyal unit；
- energy U-stat clipping；
- duplicate feature geometry；
- Mahalanobis absolute EPS；
- cross-spectrum window maturity；
- multifractal asymmetry naming；
- rough-vol search/window；
- conditional TE DecisionClock；
- quantile-dynamics repaint；
- binned-response tie ordering；
- group membership vintage；
- MI 参数依赖输出 unit；
- intrinsic-dimension / topology 已讨论的 ParamSpec、Theiler、partial warmup、2-window maturity；
- spectral duplicate canonical / price-vs-return heuristic；
- estimator-resolution 默认搜索治理；
- 当前 model evidence stale / `0/300 final_direct_use_ready` / per-canonical evidence NOT_RUN；
- 以及此前两轮文档已经明确列过的数学、PIT、lane、stateful、typed-unit、silent-clamp 问题。

本轮 AI 不应重新解释上述问题，不应再次创建一套平行 contract。

---

# 1. 本轮新增问题总览

本轮新增问题主要集中在此前容易漏掉的八个区域：

1. **模型选择指标本身是否正确**；
2. **负控测试是不是真的能抓泄漏，而不是看起来有测试**；
3. **Walk-forward OOS 如何拼接成唯一历史模拟序列**；
4. **百万级 pooled panel 每一行到底应该如何加权**；
5. **样本 telemetry 是否和实际进入 learner.fit 的 cohort 完全一致**；
6. **Artifact 从“模型文件”升级为真正可部署、可撤销、可回滚、可过期的生产资产**；
7. **旧 Artifact 是否能用原来的代码语义重放**；
8. **DecisionClock / LabelContract 是否真正成为机器可比较的时间语义，而不是字符串说明**。

下面逐项整改。

---

# 2. NEW-M-301 — 禁止 Predictive Learner 在无 Validation 时使用训练集内 IC 选择超参数

## 当前问题

当前 `modeling/trainer.py` 存在正式路径：

```python
if validation_ds is None:
    pred_train = learner.predict(frozen, X_tr)
    rank_ic = _rank_ic(y_tr, pred_train)
```

随后该分数仍进入 `select_best_validation(...)`。

也就是说，`validation_ds=None` 且 `hyperparam_grid` 有多个候选时，系统会用 **in-sample train IC** 选择超参数。

## 风险

复杂度越高越容易提高训练集表现，容易形成：

```text
无 validation
→ 训练集择参
→ 复杂模型偏好
→ OOS 过拟合
```

## 必须整改

Production predictive mode：

```text
如果 hyperparam_grid 有 >1 个候选
    validation_ds 必须存在
否则
    hard fail
```

允许无 validation 的唯一情况应是 `FIXED_HYPERPARAM_MODE`：只有一个已认证参数集，完全不做 selection。

建议新增：

```python
class HyperparamSelectionMode(Enum):
    VALIDATION_SELECT = "validation_select"
    FIXED_CERTIFIED = "fixed_certified"
```

### Hard Gate

```text
MODEL_ZERO_INSAMPLE_HYPERPARAM_SELECTION
```

---

# 3. NEW-M-302 — Validation evaluator 异常不能 silent fallback 到 pooled Spearman

`trainer._evaluate_validation()` 当前存在 broad `except Exception: pass`。真实 cross-sectional evaluator 若发生 import、shape、date alignment、runtime 错误，系统仍可能保留 fallback pooled Spearman 继续择参。

Production 应：

```text
cross-sectional evaluator failure
→ hard fail
```

Research 如需 fallback，必须显式 `allow_fallback=True`，并记录：

```text
metric_source = fallback_pooled
evaluation_degraded = true
```

### Hard Gate

```text
MODEL_ZERO_SILENT_EVALUATION_FALLBACK
```

---

# 4. NEW-M-303 — 当前模型选择的 `rank_ic` 实际仍可能是 pooled Rank IC，而不是 mean daily IC

`modeling/evaluation.py` 虽然已经计算 per-date IC，但 `cross_sectional_ic()` 的 `rank_ic` 仍来自 pooled stock-date Spearman。

对于 A 股横截面模型，默认模型选择应使用：

\[
MeanDailyRankIC=rac1T\sum_t IC_t
\]

而不是把所有年份和股票混在一起做一个 pooled rank correlation。

必须拆字段：

```text
mean_daily_rank_ic
pooled_rank_ic
daily_ic_std
daily_icir
n_valid_ic_dates
```

Trainer 默认 objective：

```text
mean_daily_rank_ic
```

### Hard Gate

```text
MODEL_SELECTION_USES_DATEWISE_CROSS_SECTIONAL_OBJECTIVE
```

---

# 5. NEW-M-304 — Turnover 当前比较的是“每日局部行号”，不是股票身份

当前 `_turnover()` 使用每个日期内部 `argsort` 后的局部位置集合。两个日期的 `7` 并不代表同一只股票。

这会让 turnover 计算产生结构性错误。

必须让 `evaluate_predictions()` 接收 row-aligned `stock_ids`，并使用真实证券 ID 构造 top-set。

增加测试：每日 DataFrame 行顺序完全打乱，但相同股票 prediction 不变，turnover 必须完全不变。

### Hard Gate

```text
MODEL_TURNOVER_IDENTITY_INVARIANT
```

---

# 6. NEW-M-305 — `block_aware_ic` 不能声称把 H-day overlapping labels 变成独立 evidence

当前是把连续 H 天的 daily IC 平均成 block，但相邻 block 的边界附近 forward-H labels 仍可能共享未来价格路径。

因此要把：

```text
performance reporting
```

和：

```text
statistical inference
```

拆开。

绩效可以继续保留每日 IC；推断应使用：

- non-overlapping anchor subsampling；
- HAC/Newey-West；
- block bootstrap / stationary bootstrap。

禁止再把简单 H-day average block 数量直接解释成独立样本数。

---

# 7. NEW-M-306 — ICIR 定义需要 versioned

当前 `mean(IC)/std(IC)` 没有年化。建议拆成：

```text
daily_ic_mean
daily_ic_std
daily_ic_mean_over_sd
annualized_icir
```

年化必须明确 annualization factor，以及 overlapping-label adjustment。

---

# 8. NEW-M-307 — Model Selection metric direction 不能靠字符串猜

当前逻辑近似：

```python
maximize = objective not in ("mse", "rmse")
```

将来 `mae`、turnover、drawdown、prediction_error 等都可能被错误 maximize。

必须建立 `MetricSpec`：

```python
MetricSpec(
    name="mean_daily_rank_ic",
    direction="maximize",
    aggregation="per_date_then_mean",
)
```

### Hard Gate

```text
MODEL_ALL_SELECTION_METRICS_HAVE_DIRECTION_CONTRACT
```

---

# 9. NEW-M-308 — Parameter Stability 已有函数，但没有进入 artifact admission

Trainer 选出 best 后直接 refit/artifact，没有要求 `neighborhood_stability` 真正 PASS。

对 `MODEL_COMPLEXITY` 可搜索参数，应在 artifact production admission 之前强制执行 stability gate。失败时降为 research 或选择 plateau-center。

---

# 10. NEW-M-309 — `neighborhood_stability()` 对离散非等距 grid 定义错误

当前检查 best±1，但 approved grid 可能是：

```text
2, 3, 5, 8
```

best=5 时真正邻居应是 3 和 8，而不是不存在的 4/6。

应按 certified ordered grid 的相邻位置定义邻居，并正确处理：

- negative score；
- minimize objective；
- score≈0；
- multi-dimensional hyperparameter grids。

---

# 11. NEW-M-310 — `parameter_stability_report()` 实际把 best 与所有候选中的最差点比较

这不是 local parameter stability。应该报告：

```text
local neighbors
plateau width
local slope
sign consistency
rank stability
```

不能用远端极端参数的差表现来否定局部平台稳定性。

---

# 12. NEW-M-311 — `label_shuffle_control` 的 leakage 判定逻辑方向不正确

当前逻辑会把“真实标签模型明显优于 shuffled-label 模型”判作 suspected leakage，但正常有信号的模型本来就应如此。

正确负控：

```text
Train:
    shuffled labels
Evaluation:
    untouched strict OOS
Expected:
    OOS IC ≈ 0
```

只有 shuffled-label model 在严格 OOS 上仍持续显著，才应怀疑 leakage / evaluator bug。

### Hard Gate

```text
MODEL_LABEL_SHUFFLE_OOS_NULL_CONTROL_VALID
```

---

# 13. NEW-M-312 — `future_poison` 当前没有重跑训练流水线，容易成为 vacuous control

当前思路是先训练 artifact，再修改未来 row，然后继续用同一个 frozen artifact 算历史输入。过去 prediction 当然不会变。

真正 future poison 必须：

```text
Pipeline A:
raw source → feature build → dataset → fit

Pipeline B:
poison cutoff 之后 raw source
→ 完整重跑 feature build → dataset → fit

Compare:
<=cutoff dataset hash
preprocessing
selected params
artifact params
past predictions
```

---

# 14. NEW-M-313 — `scaler_poison` 同样必须重跑 authoritative preprocessing pipeline

不能“训练后改未来，再看同一个 scaler hash”。

必须在 poisoned raw source 上重新完整构建 train preprocessing，然后比较 train-cutoff 以前的 frozen state。

---

# 15. NEW-M-314 — Negative controls 不能只跑 synthetic helper path

当前很多负控使用 tiny PCR / identity preprocessing / synthetic panel。它们只能证明 helper 安全，不能证明真实：

```text
train_model
DataAccess PIT
preprocessing
artifact resolver
DSL scoring
```

安全。

必须分：

```text
UNIT_NEGATIVE_CONTROL
PRODUCTION_PATH_NEGATIVE_CONTROL
```

生产负控不能 fallback 到 helper。

---

# 16. NEW-M-315 — Vacuous control 不能返回 PASS

例如没有 future rows 可 poison，不应：

```text
PASS: control vacuous
```

应返回：

```text
NOT_RUN / INVALID_FIXTURE
```

每个负控必须同时输出：

```text
exercised
mutation_effect_verified
pass
```

---

# 17. NEW-M-316 — 每类负控都必须有“已知坏实现会失败”的 Mutation Test

只证明好实现 PASS 不够。

需要 known-bad fixtures，例如：

```text
full-sample scaler
future-label fit
future artifact resolver
random-time split
```

负控必须能把这些坏实现打红。

---

# 18. NEW-M-317 — 当前 Feature Ablation 实际是 occlusion/permutation，不是真正 retrained ablation

把某特征填均值或 shuffle 后继续用同一 frozen model，应该叫：

```text
feature_occlusion_importance
feature_permutation_importance
```

真正 ablation 应：

```text
删除 feature
→ 重新训练
→ validation/OOS 对比
```

应改名并另实现 `retrained_feature_ablation`。

---

# 19. NEW-M-318 — Feature shuffle 应支持 within-date / block shuffle

全局 shuffle 会混合年份、regime 和横截面。

对于横截面模型优先：

```text
within-date shuffle
```

对于时序机制：

```text
block shuffle
```

不要用 global shuffle 解释所有 feature importance。

---

# 20. NEW-M-319 — Walk-forward 默认 Test windows 大量重叠，必须定义唯一 OOS 拼接规则

默认：

```text
test_bars=126
step_bars=21
retrain_every_bars=21
```

相邻 test window 高度重叠。如果直接 concat，同一个日期会被多个 artifact 重复预测、重复计权。

必须增加：

```python
class OOSStitchPolicy(Enum):
    ACTIVE_UNTIL_NEXT_RETRAIN = ...
    LATEST_LEGAL_ARTIFACT = ...
```

推荐 simulated-live：

```text
artifact_k active in
[activation_k, activation_{k+1})
```

每个日期只能对应一个 artifact。

### Hard Gate

```text
MODEL_OOS_HISTORY_HAS_UNIQUE_DATE_ARTIFACT_MAPPING
```

---

# 21. NEW-M-320 — Validation/Test 自身也需要 SampleAdequacy

当前主要检查 training adequacy，但 validation/test 可能有效日期极少、横截面很小、label coverage 很低却仍用于择参/评价。

新增：

```text
min_validation_dates
min_validation_effective_dates
min_validation_median_stocks
min_validation_label_coverage

min_test_dates
min_test_effective_dates
min_test_median_stocks
min_test_label_coverage

min_valid_daily_ic_dates
```

---

# 22. NEW-M-321 — Purge/Embargo 必须基于交易所 Session Calendar，而不是“数据里出现过的日期”

如果真实交易日因数据源事故整天缺失，unique-date ordinal 会改变 H-bar 时间距离。

必须以 ExchangeSessionCalendar 为 authority。数据缺 session 应被视为 data-quality failure，不能改变 label maturity/Purge 时间代数。

---

# 23. NEW-M-322 — `date_coverage` 应以 expected exchange sessions 为分母

不能使用自然日跨度，否则周末和法定假日都会降低 coverage。

正确：

\[
coverage=
rac{observed\ expected\ sessions}
{expected\ exchange\ sessions}
\]

---

# 24. NEW-M-323 — Mature Label 不能等同于 `label.notna()`

完整历史表里 forward-H label 即便非 NaN，在某个历史 training cutoff 当时可能尚未成熟。

必须根据：

```text
anchor session
LabelContract interval
training cutoff
ExchangeSessionCalendar
```

机械判断 maturity。

---

# 25. NEW-M-324 — SampleTelemetry 必须用同一 cohort mask 逐层 narrowing

现在 independent counts 可能导致 `mature_label_obs > finite_obs`。

应维护：

```text
raw_mask
finite_input_mask
mature_label_mask
purge_mask
support_mask
final_fit_mask
```

并保证：

```text
mask_next = mask_prev & new_condition
```

---

# 26. NEW-M-325 — Pooled Panel 需要明确 SampleWeightPolicy，不能默认让股票更多的日期权重更大

当前 `weights=None` 相当于每个 stock-date 等权。随着 A 股股票数量增长，后期日期自然获得更多 loss weight。

至少支持：

```text
EQUAL_ROW
EQUAL_DATE
EQUAL_DATE_THEN_STOCK
TIME_DECAY_EQUAL_DATE
```

对横截面模型建议优先采用/研究 `EQUAL_DATE`：

\[
\sum_i w_{i,t}=1
\]

---

# 27. NEW-M-326 — `decay_weighted_expanding` 的 decay helper 已存在，但 trainer 需要真正使用

如果 preset 声称 decay-weighted expanding，runtime fit、artifact manifest、evidence 必须都反映真实 weight policy。

---

# 28. NEW-M-327 — PCR/PLS/ElasticNet 的 weights 参数不能“接受但忽略”

如果暂未支持 weighted fit：

```python
if weights is not None:
    raise NotImplementedError
```

比 silent ignore 更安全。

### Hard Gate

```text
MODEL_ZERO_IGNORED_DECLARED_SAMPLE_WEIGHTS
```

---

# 29. NEW-M-328 — Production PanelDataset 禁止自动把所有非 date/stock/label 列当 feature

真实 panel 还可能有：

```text
regime_state
industry_id
sample_weight
available_at
source_vintage
label_end
tradable
universe_flag
```

自动 feature inference 可能直接引入 context leakage。

Production 必须显式 FeatureSchema；auto-infer 只允许 research convenience。

---

# 30. NEW-M-329 — Cross-sectional coverage 应基于每个日期的 PIT eligible universe

不能用整个训练期出现过的所有股票作为统一分母。

正确：

\[
coverage_t=
rac{observed\ eligible_t}
{PIT\ eligible\ universe_t}
\]

并报告 median/p10/min。

---

# 31. NEW-M-330 — `effective_date_count` 不能简单等于 unique dates

定义有效日期至少要求：

```text
finite stocks >= threshold
label coverage >= threshold
universe coverage >= threshold
```

---

# 32. NEW-M-331 — Sample telemetry 必须与真正进入 learner.fit 的 cohort 一致

当前 raw telemetry 可能把 feature-missing rows 判为非 finite，但 preprocessing impute 后这些 row 仍可能进入 fit。

Artifact 应保存：

```text
n_actual_fit_rows
n_imputed_rows
imputed_fraction_per_feature
```

SampleAdequacy 应基于实际 final fit cohort，而不是理论 raw panel count。

---

# 33. NEW-M-332 — `label_coverage` 使用 `isfinite`，不能只用 `.notna()`

`+inf/-inf` 不能算 covered。建议同时报告：

```text
nan_fraction
inf_fraction
```

---

# 34. NEW-M-333 — Predictor 通过 monkey-patch learner.fit 做 guard，并发不安全

Production scoring 不能每次：

```text
临时替换 learner.fit
→ predict
→ 恢复
```

同一个 artifact 多线程共享会 race。

应在类型层面拆：

```text
TrainableLearner
FrozenPredictor
```

FrozenPredictor 根本不暴露 fit。

---

# 35. NEW-M-334 — Direct `artifact.predict()` 是 as-of 检查旁路

`artifact.predict(X)` 不需要 asof/training_cutoff/DecisionClock。

Production public API 应要求：

```python
ScoringContext(
    asof,
    market_session,
    decision_clock,
    source_snapshot,
    feature_schema,
)
```

裸 `artifact._predict_unchecked()` 只能 internal/test。

### Hard Gate

```text
MODEL_PRODUCTION_SCORING_REQUIRES_ASOF_CONTEXT
```

---

# 36. NEW-M-335 — Resolver 不能自动选择“最新训练完成模型”，必须选择“已激活模型”

真实生产状态应区分：

```text
CANDIDATE
VALIDATED
STAGED
ACTIVE
RETIRED
REVOKED
```

新训练 artifact 不应因 cutoff 更新就自动替换线上模型。

历史 replay 必须按 activation interval，而不是按 training cutoff 自动切换。

---

# 37. NEW-M-336 — Production resolver 必须要求 Certification

Artifact admission 至少包括：

```text
production_certified
certification_evidence_hash
approval_id
certified_at
certification_code_sha
```

Research resolver 可读未认证 artifact；production resolver 不可读。

---

# 38. NEW-M-337 — ArtifactStore 持久化后，Resolver 需要可恢复索引

进程重启后不能因为 in-memory registry 清空导致“磁盘有模型但 resolver 查不到”。

需要 ArtifactCatalog/Index，启动加载或 lazy load。

---

# 39. NEW-M-338 — Artifact 文件写入必须 atomic durable

使用：

```text
temp
→ write
→ flush
→ fsync
→ checksum
→ atomic rename
→ catalog commit
```

不能直接 `open(...,"w")` 后一次 JSON dump。

---

# 40. NEW-M-339 — `_default_resolver` 进程级全局变量会污染多策略/多回测上下文

Production 应显式 dependency injection 或使用 context-local resolver，而不是 process-global mutable singleton。

---

# 41. NEW-M-340 — `model_version` 不应参与普通字符串排序决定上线模型

版本是 identity，不是 deployment authority。使用 `activation_epoch/deployment_sequence` 决定 active artifact。

---

# 42. NEW-M-341 — 历史 Artifact 必须绑定原 predictor ABI，不能用“当前同名 learner”解释

当前 load 基于 learner name 获取当前 registry class。如果实现升级但名称不变，旧 artifact 会被新代码执行。

需要：

```text
(model_name, semantic_version, predictor_abi_version)
```

老 artifact 只能由兼容 ABI 加载，否则 migration/fail。

### Hard Gate

```text
MODEL_HISTORICAL_ARTIFACT_PREDICTOR_ABI_REPRODUCIBLE
```

---

# 43. NEW-M-342 — Artifact lineage identity 应包含历史合法区间/部署字段

建议纳入：

```text
training_cutoff
available_at
activation_at
retired_at
certification_hash
```

改变这些字段会改变 artifact 的历史合法性，因此不能保持同 lineage identity。

---

# 44. NEW-M-343 — 时间字段不能长期使用裸字符串/Any 比较

建立 typed：

```python
TradingTimestamp(
    session_date,
    session_phase,
    timezone,
)
```

session phase 至少：

```text
PRE_OPEN
OPEN
INTRADAY
VWAP_COMPLETE
CLOSE
AFTER_CLOSE
```

才能机械比较 same-day timing。

---

# 45. NEW-M-344 — Artifact 必须保存完整训练政策 hash

新增：

```text
walk_forward_spec_hash
sample_contract_hash
parameter_search_policy_hash
preprocessing_spec_hash
sample_weight_policy_hash
selection_metric_spec_hash
oos_stitch_policy_hash
```

只存 model hyperparameters 不足以复现训练决策。

---

# 46. NEW-M-345 — ModelRegistry record 应 immutable

`get()` 不应返回可直接修改的内部 dict。使用 frozen dataclass / MappingProxy，并返回 immutable snapshot。

---

# 47. NEW-M-346 — ModelRegistry 应按 semantic version / predictor ABI 版本化

不能永远只以 `model_name` 为 key。推荐：

```python
ModelKey(
    model_name,
    semantic_version,
    predictor_abi_version,
)
```

---

# 48. NEW-M-347 — APPROVED_SEARCH_SPACES 必须在 trainer 边界强制执行

`train_model()` 当前可接收任意 caller-provided hyperparam_grid。

必须：

```text
candidate param 已知
role 已登记
searchable/tune_inside_validation 合法
值在 certified_values/bounds
```

未知或越界直接 hard fail。

### Hard Gate

```text
MODEL_TRAINER_ENFORCES_PARAMETER_SEARCH_POLICY
```

---

# 49. NEW-M-348 — Unknown feature availability 必须 fail closed

当前未登记 feature availability 时，不应默认视为 decision time 可用。

Production：

```text
unknown availability
→ reject
```

Research：

```text
UNKNOWN/UNCERTIFIED
```

但不得 PIT-certified。

### Hard Gate

```text
MODEL_ZERO_UNKNOWN_FEATURE_AVAILABILITY_FAIL_OPEN
```

---

# 50. NEW-M-349 — LabelContract identity 不能只靠 `label_name`

至少把：

```text
horizon_bars
return_basis
entry_price_basis
exit_price_basis
availability_time_rule
overlapping
purge_by_interval
embargo_bars
```

纳入 `LabelContract.semantic_hash()`。

---

# 51. NEW-M-350 — DecisionClock identity 不能只靠 `decision_at`

必须把：

```text
feature_available_at
label_available_at
score_written_at
execution_at
```

一起进入 `DecisionClock.semantic_hash()`。

---

# 52. NEW-M-351 — `LabelContract.overlapping` 应由 label interval/anchor frequency 推导或校验

H=20、daily anchors 通常是 overlapping。不能完全依靠调用方手填 bool。

---

# 53. NEW-M-352 — Timing authority 不应长期使用自由字符串

将：

```text
t_close
t+H close
next session VWAP
```

升级为 typed session-time algebra：

```text
SessionOffset
SessionPhase
AvailabilityRule
LabelIntervalRule
ExecutionRule
```

文本描述从 typed rule 自动生成。

---

# 54. NEW-M-353 — Predictive Model 需要正式 PredictionOutputContract

至少：

```text
semantic
unit
label_horizon
price_basis
expected_range
higher_is_better
cross_sectional_or_absolute
calibrated
```

避免所有模型输出都只是 generic float。

---

# 55. NEW-M-354 — Prediction 输出必须做 shape/alignment/finite-policy hard gate

验证：

```text
len(pred)==len(input)
pred.ndim==1
row identity preserved
dtype allowed
unexpected inf rejected
NaN reason/status defined
```

建议输出 `PredictionBatch(values,row_ids,artifact_id,asof)`。

---

# 56. NEW-M-355 — 模型上线需要 canary / activation / rollback 生命周期

建议：

```text
TRAINED → VALIDATED → STAGED → CANARY → ACTIVE → RETIRED/REVOKED
```

Promotion 可检查：

```text
OOS evidence
coverage
old/new correlation
prediction scale
risk exposure
```

---

# 57. NEW-M-356 — Retrain boundary 应增加旧/新 Artifact discontinuity test

在激活日前用同一 feature panel 比较：

```text
rank correlation
Pearson correlation
mean shift
std ratio
top-k overlap
sign flip ratio
```

异常跳变阻止自动 promotion。

---

# 58. NEW-M-357 — 明确 Raw Score 是否跨 Artifact 可比

如果只用于每日 rank，声明 raw magnitude 不跨 artifact 比较；如果是 expected return，则需要 PIT-safe calibration contract。

---

# 59. NEW-M-358 — Resolver 必须有 stale-model 最大年龄约束

增加：

```text
max_artifact_age_sessions
expected_retrain_frequency
staleness_status
```

重训失败后不能无限期静默用旧模型。

---

# 60. NEW-M-359 — Artifact 需要 Revocation 机制

记录：

```text
revoked_at
reason
affected_interval
replacement_artifact
```

生产 resolver 不得继续返回 REVOKED artifact。

---

# 61. NEW-M-360 — Drift Report 应升级成可用于生产监控的 contract

补充：

```text
PSI
KS
Wasserstein
prediction rank drift
coefficient drift
IC decay
coverage drift
regime/expert occupancy drift
missingness drift
```

并明确哪些只是 monitor、哪些会触发 retrain/promotion block。

---

# 62. NEW-M-361 — Model monitoring 的 realised performance 同样必须 PIT

H=20 的模型今天 performance 直到未来 20 sessions 后才成熟。Monitoring record 必须有 `metric_available_at`。

---

# 63. NEW-M-362 — Regime/MoE gate 需要明确 transform space contract

即使 train/predict 用同一个 gating feature，如果 train boundary 在 raw gate 上拟合、predict 却在标准化/裁剪后的 gate 上路由，也会不一致。

保存：

```text
gating_feature_transform_id
gating_space
```

---

# 64. NEW-M-363 — Soft gating 的 temperature/geometry 是隐藏模型政策

当前 inverse-distance+softmax 的 sharpness 会影响输出。应固定并 semantic-versioned，或显式 `gate_temperature`，但不要让 miner 大范围搜索。

---

# 65. NEW-M-364 — Regime quantile boundaries 在 ties 下可能重复，应在 fit 前直接判 infeasible

检查：

```text
effective_unique_boundaries
effective_regime_count
```

必须等于 requested regime count，否则参数 infeasible。

---

# 66. NEW-M-365 — 训练必须有 deterministic row-order contract

保存：

```text
training_sort_policy=(date, stock)
dataset_order_hash
```

为未来 SGD/NN/GBDT/online learner 做准备。

---

# 67. NEW-M-366 — 仅保存 random_seed 不足以复现随机模型

增加：

```text
library versions
BLAS
device
determinism flags
thread count
runtime_environment_hash
```

---

# 68. NEW-M-367 — 增加 Model Fit Fingerprint

由：

```text
frozen params hash
preprocessing hash
training cohort hash
weights hash
feature schema hash
label hash
code hash
```

组成。

同输入同代码应得到同 fingerprint。

---

# 69. NEW-M-368 — Trainer 不得把所有 `Exception` 都当成“候选拟合失败”

仅捕获预期异常：

```text
ParameterDomainError
SampleAdequacyError
NumericalConvergenceError
ExpectedDegeneracyError
```

未知异常必须抛出，不能换下一个 hyperparameter 继续。

### Hard Gate

```text
MODEL_ZERO_UNKNOWN_FIT_EXCEPTION_DOWNGRADED_TO_CANDIDATE_REJECTION
```

---

# 70. NEW-M-369 — Evaluation 也必须区分“指标不可计算”和“代码错误”

constant prediction 可以返回 metric NaN + reason；shape mismatch/date alignment bug 必须 hard fail。

---

# 71. NEW-M-370 — 每个 hyperparameter candidate 需要稳定 identity

记录：

```text
candidate_id
parameter_hash
feature_bundle_hash
training_spec_hash
```

---

# 72. NEW-M-371 — Selection tie-break 必须 deterministic，优先简单模型

两个候选 score 相同/几乎相同时，推荐：

```text
更低 complexity
→ 更强 regularization
→ 更低 cost
→ stable lexical identity
```

不能由 grid 输入顺序决定。

---

# 73. NEW-M-372 — Selection 需要 uncertainty，而不是只比较点估计

对候选 daily IC difference 做 paired/block bootstrap CI。差异不显著时优先更简单模型。

---

# 74. NEW-M-373 — Model family competition 也需要 multiple-testing/complexity governance

PCR/PLS/ENet/Regime/MoE 如果都在同一 validation 反复竞争，必须记录 family trial multiplicity。复杂 family 需要更高 promotion threshold。

---

# 75. NEW-M-374 — Final Holdout exposure ledger 应不可逆

一旦某段 holdout 被请求过 metric，记录：

```text
exposed_at
requester
candidate_generation_version
```

后续不得再称为 untouched holdout。

---

# 76. NEW-M-375 — OOS Prediction Ledger 每行必须绑定 artifact

至少：

```text
date
stock
prediction
artifact_id
model_version
training_cutoff
available_at
feature_schema_hash
source_snapshot_hash
```

---

# 77. NEW-M-376 — Prediction 还必须绑定 Feature Snapshot identity

同一 artifact 输入不同 revision feature 会得到不同结果，必须保存 `feature_snapshot_id/hash`。

---

# 78. NEW-M-377 — Prediction cache 必须绑定 artifact_id + feature snapshot

不能只用 model_name/date。

---

# 79. NEW-M-378 — Retrain job 应幂等

相同：

```text
model semantic
training cutoff
snapshot
policy
code
```

重复训练应得到相同 fit fingerprint 或识别“artifact already exists”。

---

# 80. NEW-M-379 — Artifact ID 不宜完全随机 UUID

建议 content-addressed artifact ID；另用 `run_id` 表示某次训练执行。

---

# 81. NEW-M-380 — 区分 TrainingRun 与 ModelArtifact

TrainingRun 保存机器、日志、retry、时间；Artifact 是 immutable content output。多次 run 可以得到同一个 artifact。

---

# 82. NEW-M-381 — 资源调度变化不应改变确定性模型结果

增加：

```text
single-thread vs multi-thread
batch size
shard order
```

metamorphic parity。

---

# 83. NEW-M-382 — 随机模型未来要做 Seed Stability，不只固定一个 seed

多 seed OOS 极不稳定的模型应保持 research-only。

---

# 84. NEW-M-383 — 所有 predictive families 应统一记录 design condition telemetry

包括：

```text
effective rank
condition number
near-zero variance
duplicate columns
collinearity summary
```

---

# 85. NEW-M-384 — 模型训练前还需要 feature-value-level exact duplicate 检查

不同 canonical 在当前训练 cohort 上可能完全相同。Exact duplicate feature columns 默认 reject，避免无意义增加自由度。

---

# 86. NEW-M-385 — Coverage 应是 selection hard constraint

高 IC 但覆盖 15% 股票的模型不能直接赢过覆盖 95% 的模型。

先：

```text
coverage >= threshold
```

再比较 IC。

---

# 87. NEW-M-386 — Turnover/Coverage/Latency 建议作为 constraints，而不是任意加权 objective

推荐：

```text
Hard constraints
+
one primary objective
```

更容易审计与稳定。

---

# 88. NEW-M-387 — 增加 Model Cost Contract

记录：

```text
fit time
peak memory
artifact size
prediction latency
rows/sec
```

避免 miner 选出无法按计划重训的模型。

---

# 89. NEW-M-388 — Retrain 前必须绑定 DataQualityCertificate

到了 scheduled retrain date，不代表数据完整。Feature snapshot/universe/source quality 不达标时禁止训练或 promotion。

---

# 90. NEW-M-389 — Validation/Test 数据质量也必须 gating

坏数据月不能驱动 hyperparameter selection。

---

# 91. NEW-M-390 — Feature Bundle 需要冻结 semantic version，不只是 feature 名称

保存：

```text
canonical
operator semantic version
params
normalized AST hash
source snapshot
```

历史 artifact 不得被新语义同名 factor 重新解释。

---

# 92. NEW-M-391 — 记录 Factor Search × Model Search 的联合 trial 数

LLM 先搜公式、再对每个公式搜模型参数，总试验数是乘法级。SearchExposureLedger 必须记录 joint trial multiplicity。

---

# 93. NEW-M-392 — 给各 Model Family 设置 candidate budget

复杂 family 参数更多，不能因为“彩票多”而更容易中奖。

---

# 94. NEW-M-393 — 提前建立 EnsembleArtifact contract

未来 PCR+ENet+Regime ensemble 时，ensemble 权重本身也需要 train/validation/PIT/availability/artifact lineage。不要直接在 DSL 随意平均后当 production model。

---

# 95. NEW-M-394 — Model Fallback Policy 必须统一

模型当天无法打分时明确：

```text
NO_SIGNAL
USE_PREVIOUS_ACTIVE_ARTIFACT
USE_BASELINE_MODEL
ZERO_SCORE
```

每种行为进入 semantic/deployment policy。

---

# 96. NEW-M-395 — Prediction NaN 与 Model Failure 必须区分

增加 `PredictionStatus`：

```text
OK
MISSING_FEATURE
OUT_OF_UNIVERSE
NO_ACTIVE_ARTIFACT
NUMERICAL_FAILURE
CLOCK_VIOLATION
```

---

# 97. NEW-M-396 — Retrain failure 后继续旧模型必须进入 DEGRADED_STALE 状态

不能静默继续。

---

# 98. NEW-M-397 — RETIRED/REVOKED Artifact 不能物理删除

历史 replay 仍需要它和撤销原因。

---

# 99. NEW-M-398 — Artifact 保存训练 cohort 摘要

至少：

```text
n_dates
n_stocks
n_rows
industry coverage
cap distribution
missingness
label distribution
feature summary
```

---

# 100. NEW-M-399 — 模型间比较必须报告 common OOS cohort

不同模型 coverage 不同，不能只比较各自 native cohort IC。

---

# 101. NEW-M-400 — Common cohort 也要防 intersection shrinkage

比较很多模型时全模型交集可能很小。报告 pairwise common cohort / benchmark common cohort / minimum coverage。

---

# 102. NEW-M-401 — 增加 Baseline Model Contract

复杂模型至少对比 zero/simple linear/ridge baseline。

---

# 103. NEW-M-402 — 复杂模型必须通过 Incremental Value Gate

Regime 对比同 feature pooled linear；MoE 对比同 feature single model。OOS 无稳定增益则不 promotion。

---

# 104. NEW-M-403 — Model OOS evidence 增加已知风格暴露报告

至少：

```text
size
industry
beta
liquidity
```

不是要求全部中性，而是知道模型学到了什么。

---

# 105. NEW-M-404 — TrainingObjective 必须区分 raw-return prediction 与 cross-sectional ranking

支持并版本化：

```text
RAW_RETURN_MSE
DATE_DEMEANED_RETURN_MSE
DATE_ZSCORE_RETURN_MSE
RANK_SURROGATE
```

不能共用一个模糊“预测模型”身份。

---

# 106. NEW-M-405 — Label preprocessing 独立建 `LabelTransformContract`

如 y winsor、demean、vol-normalize、rank transform，都要进入 artifact identity。

---

# 107. NEW-M-406 — Label cross-sectional transform 也必须用历史 PIT universe

不能用未来清洗后 stock set 回填历史 rank/zscore。

---

# 108. NEW-M-407 — 未来出现 stock embedding/fixed-effect 时，必须单独评估 unseen-stock OOS

防止模型仅记忆股票 identity。

---

# 109. NEW-M-408 — 行业/概念 categorical vocabulary 也要 PIT/versioned

保存 vocabulary、unknown-category policy、membership vintage。

---

# 110. NEW-M-409 — 增加 Stock-Level Leakage Control

做 leave-stocks-out / unseen-stock 负控。

---

# 111. NEW-M-410 — 增加 within-date label shuffle 与 date-block permutation 两类负控

分别检验横截面和时序信号。

---

# 112. NEW-M-411 — 增加 Feature Timestamp Poison

不只改 feature value，还把同一 value 的 `available_at` 推迟到 decision 后，确保 pipeline 真拒绝。

---

# 113. NEW-M-412 — 增加 Source Revision Replay 测试

历史 revision 只能影响 revision 生效后重训的 artifacts。

---

# 114. NEW-M-413 — 建 Model Artifact Dependency DAG

Artifact 指向 FeatureArtifact/DatasetSnapshot/UniverseSnapshot/LabelSnapshot/TrainingRun/CodeComponent。

---

# 115. NEW-M-414 — 支持依赖传播 invalidation

某 factor semantic bug 修复后，能自动定位所有受影响 model artifacts。

---

# 116. NEW-M-415 — Artifact schema migration 必须显式版本和 converter

不能靠缺字段默认值偷偷赋予旧 artifact 新语义。

---

# 117. NEW-M-416 — FrozenModel serialization 需要 per-learner schema version

`dict[str,Any]` 长期不足以约束字段、dtype、shape。增加 `FrozenModelSchemaVersion` 和 validator。

---

# 118. NEW-M-417 — Prediction benchmark 按真实股票规模

至少：

```text
1k / 3k / 5k / 10k rows
```

覆盖 preprocessing+resolution+predict。

---

# 119. NEW-M-418 — Retraining benchmark 按真实百万级 pooled panel

至少：

```text
100k / 500k / 1m / 5m rows
```

记录 fit time/peak memory/scaling。

---

# 120. NEW-M-419 — 审计模型训练中的大矩阵复制次数

pandas→numpy→preprocessing→learner 可能多份 X copy。数学语义稳定后再优化 memory layout / zero-copy。

---

# 121. NEW-M-420 — 最终锁死 Training Pipeline 与 FactorEngine Runtime 职责边界

```text
Training Pipeline:
dataset build
train/validation/test
selection
artifact production
certification

FactorEngine Runtime:
feature computation
legal artifact resolution
frozen scoring
```

Runtime 不应重新出现 fit/select/retrain。

---

# 122. 本轮新增 Hard Gates 汇总

建议加入：

```text
MODEL_ZERO_INSAMPLE_HYPERPARAM_SELECTION
MODEL_ZERO_SILENT_EVALUATION_FALLBACK
MODEL_SELECTION_USES_DATEWISE_CROSS_SECTIONAL_OBJECTIVE
MODEL_TURNOVER_IDENTITY_INVARIANT
MODEL_ALL_SELECTION_METRICS_HAVE_DIRECTION_CONTRACT
MODEL_COMPLEXITY_SELECTION_REQUIRES_STABILITY_GATE

MODEL_LABEL_SHUFFLE_OOS_NULL_CONTROL_VALID
MODEL_NEGATIVE_CONTROLS_REQUIRE_EXERCISED_MUTATION
MODEL_NEGATIVE_CONTROLS_KNOWN_BAD_VARIANTS_FAIL

MODEL_OOS_HISTORY_HAS_UNIQUE_DATE_ARTIFACT_MAPPING
MODEL_VALIDATION_AND_TEST_HAVE_ADEQUACY_GATES
MODEL_PURGE_USES_EXCHANGE_SESSION_CALENDAR
MODEL_LABEL_MATURITY_USES_SESSION_CALENDAR
MODEL_SAMPLE_TELEMETRY_MATCHES_ACTUAL_FIT_COHORT

MODEL_ZERO_IGNORED_DECLARED_SAMPLE_WEIGHTS
MODEL_ALL_POOLED_LEARNERS_HAVE_SAMPLE_WEIGHT_POLICY
MODEL_PRODUCTION_FEATURE_SCHEMA_EXPLICIT

MODEL_PRODUCTION_SCORING_REQUIRES_ASOF_CONTEXT
MODEL_RESOLVER_USES_ACTIVE_DEPLOYMENT_NOT_LATEST_FIT
MODEL_PRODUCTION_ARTIFACT_REQUIRES_CERTIFICATION
MODEL_ARTIFACT_STORE_ATOMIC_DURABLE
MODEL_HISTORICAL_ARTIFACT_PREDICTOR_ABI_REPRODUCIBLE

MODEL_TRAINER_ENFORCES_PARAMETER_SEARCH_POLICY
MODEL_ZERO_UNKNOWN_FEATURE_AVAILABILITY_FAIL_OPEN
MODEL_LABEL_CONTRACT_IDENTITY_COMPLETE
MODEL_DECISION_CLOCK_IDENTITY_COMPLETE

MODEL_PREDICTION_OUTPUT_CONTRACT_REQUIRED
MODEL_PREDICTION_ROW_ALIGNMENT_VERIFIED
MODEL_ARTIFACT_STALENESS_POLICY_REQUIRED
MODEL_ARTIFACT_REVOCATION_SUPPORTED

MODEL_TRAINING_UNKNOWN_EXCEPTIONS_FAIL_LOUD
MODEL_SELECTION_TIEBREAK_DETERMINISTIC
MODEL_OOS_LEDGER_BINDS_ARTIFACT_PER_ROW
MODEL_ARTIFACT_BINDS_FEATURE_SNAPSHOT
MODEL_RETRAIN_IDEMPOTENT_BY_FINGERPRINT

MODEL_COMMON_OOS_COHORT_REPORTED
MODEL_COMPLEXITY_REQUIRES_INCREMENTAL_VALUE
MODEL_TRAINING_OBJECTIVE_VERSIONED
MODEL_LABEL_TRANSFORM_VERSIONED
MODEL_ARTIFACT_DEPENDENCY_DAG_COMPLETE
```

---

# 123. 建议优先级

## P0

优先：

```text
NEW-M-301~304
NEW-M-311~316
NEW-M-319
NEW-M-321
NEW-M-323~324
NEW-M-331
NEW-M-333~336
NEW-M-341
NEW-M-347~350
```

这些直接关系：

- 训练集择参；
- 评价指标正确性；
- 负控是否真实；
- OOS 是否重复；
- 交易时间；
- 真实训练 cohort；
- production scoring 旁路；
- 模型部署；
- 历史可复现；
- PIT fail-open。

## P1

随后处理 selection stability、样本权重、artifact lifecycle、typed timing、output contract、deployment safety、search exposure。

## P2

最后做监控、schema migration、大规模 benchmark、性能/内存优化。

---

# 124. 推荐执行顺序

1. **Evaluation Correctness**：NEW-M-301~310  
2. **Negative Control Truthfulness**：NEW-M-311~318  
3. **Walk-forward / Calendar / OOS Stitch**：NEW-M-319~324  
4. **Sample Weight / Cohort Truth**：NEW-M-325~332  
5. **Artifact Deployment Lifecycle**：NEW-M-333~359  
6. **Monitoring / Reproducibility**：NEW-M-360~367  
7. **Error / Selection Governance**：NEW-M-368~374  
8. **Prediction Ledger / Idempotency**：NEW-M-375~382  
9. **Model Comparison / Objective / PIT Extensions**：NEW-M-383~416  
10. **Scale / Performance / Responsibility Boundary**：NEW-M-417~420

---

# 125. 必须新增的关键测试

```text
test_no_validation_grid_search_rejected
test_validation_evaluator_exception_fails_closed
test_model_selection_uses_mean_daily_ic_not_pooled
test_turnover_invariant_to_daily_row_permutation
test_overlapping_oos_dates_resolve_to_one_artifact
test_purge_calendar_invariant_to_missing_source_date
test_label_maturity_not_equal_non_nan
test_sample_telemetry_matches_actual_fit_rows
test_equal_date_weighting
test_declared_weights_not_silently_ignored
test_predictor_thread_safety
test_direct_future_artifact_scoring_rejected_in_production
test_unpromoted_artifact_not_resolved
test_revoked_artifact_not_resolved
test_stale_artifact_policy
test_old_artifact_requires_matching_predictor_abi
test_artifact_write_atomicity
test_unknown_feature_availability_rejected
test_label_contract_hash_changes_with_horizon
test_decision_clock_hash_changes_with_execution
test_parameter_grid_policy_enforced_at_trainer
test_label_shuffle_null_on_strict_oos
test_future_poison_rebuilds_authoritative_pipeline
test_negative_control_known_bad_pipeline_fails
test_feature_schema_auto_inference_rejected_in_production
test_selection_tie_prefers_simpler_model
test_common_oos_cohort_model_comparison
test_retrain_boundary_prediction_shift_report
test_training_run_idempotent_fingerprint
```

---

# 126. Definition of Done

```text
[ ] 无 validation 的 predictive grid search 被禁止
[ ] 选模使用真正 date-wise cross-sectional objective
[ ] turnover 使用 stock identity
[ ] evaluation 不再 silent fallback
[ ] metric direction 有 contract
[ ] parameter stability 真正进入 admission

[ ] label-shuffle 改成 strict OOS null
[ ] future/scaler poison 重跑 authoritative pipeline
[ ] vacuous control 不得 PASS
[ ] known-bad mutation test 必须 FAIL

[ ] OOS 每 date 唯一 artifact
[ ] purge/maturity 使用 exchange session calendar
[ ] validation/test 有 adequacy
[ ] sample telemetry 与实际 fit cohort 一致

[ ] pooled panel 有 SampleWeightPolicy
[ ] PCR/PLS/ENet 不再 silent ignore weights
[ ] production feature schema 显式
[ ] coverage 使用 PIT eligible universe

[ ] production scoring 必须带 as-of context
[ ] artifact 有 certify/stage/activate/retire/revoke
[ ] stale model 有 policy
[ ] resolver 按 deployment timeline
[ ] store atomic durable
[ ] restart 可恢复 catalog

[ ] historical artifact 绑定 predictor ABI
[ ] artifact schema migration 明确
[ ] fit fingerprint 可复现
[ ] content identity 与 TrainingRun 分离

[ ] unknown feature availability fail closed
[ ] LabelContract 有 semantic hash
[ ] DecisionClock 有 semantic hash
[ ] session timing typed

[ ] PredictionOutputContract 完成
[ ] row alignment/status 完成
[ ] retrain boundary discontinuity 有 gate

[ ] trainer enforce ParameterSearchPolicy
[ ] selection tie-break deterministic
[ ] joint factor×model trials 有 exposure ledger
[ ] final holdout exposure 不可伪装

[ ] 模型比较报告 common OOS cohort
[ ] 复杂模型证明 incremental value
[ ] TrainingObjective / LabelTransform versioned

[ ] 大规模训练/评分 benchmark 完成
```

---

# 127. 最终要求

这轮整改不要继续以“模型能 fit、能 predict”为标准。

真正 production-ready 必须同时能回答：

```text
为什么选中这个模型？
用的是什么 validation 指标？
指标有没有算错？
负控是否真的能抓未来函数？
OOS 每一天到底是哪一个 artifact 负责？
每个日期在训练 loss 里权重是多少？
模型什么时候训练、什么时候批准、什么时候上线？
重训失败怎么办？
模型过期怎么办？
模型发现错误后怎么撤销？
历史 artifact 是否还能由原 ABI 重放？
未知字段 availability 会不会被默认放行？
这一行 prediction 能否追到唯一 artifact + feature snapshot？
```

上述问题全部能机器化回答，模型层才算真正进入企业级生产状态。
