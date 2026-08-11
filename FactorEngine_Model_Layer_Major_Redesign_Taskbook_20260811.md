# FactorEngine 模型层整体重构执行任务书
## —— 从“短窗口模型算子”升级为“企业级 PIT Predictive Modeling + Rolling Statistical Estimator 双体系”

> **适用仓库**：`18047533889/quant_projects`  
> **主要范围**：`factor_engine/cleaned_operators/`、模型 contract/timing/lane、模型训练与 artifact 生命周期、因子挖掘搜索空间、回测/验证接口  
> **用途**：本文件不是建议清单，而是可直接交给 Coding AI / Agent 执行的整体模型层整改任务书。  
> **执行要求**：开始修改前必须重新读取最新 `main`，逐条对照本文件；已经修复的条目标记 `ALREADY_FIXED` 并用当前 HEAD 测试证明，不得凭注释宣称完成。

---

# 0. 本轮不是“小修”，而是模型层架构重构

当前 FactorEngine 把大量不同性质的东西统一包装成 `Operator`：

- rolling PCA / AR / regression beta；
- Kalman / GARCH / Markov 状态估计；
- KNN / Mahalanobis / Matrix Profile；
- PCR / PLS / ElasticNet；
- Regime-conditioned forecast；
- Mixture-of-Experts；
- DMD / SSA / RQA / TE / HSIC 等研究估计器。

这种统一 registry 对 DSL 很方便，但统计含义完全不同。

尤其当前 supervised predictive models 仍然偏向：

```text
每只股票
×
短 rolling window
×
现场 fit
×
当前 row score
```

这可以作为研究原型，但不能继续作为企业级 predictive model 的最终训练范式。

本轮整改后的顶层原则必须是：

```text
FactorEngine Model Layer
    ├── A. Rolling / Local Statistical Estimators
    ├── B. Recursive Stateful Estimators
    ├── C. Same-time Cross-sectional Models
    ├── D. Predictive Supervised Learners
    └── E. Research Structural Estimators
```

其中：

```text
A/B/C/E
    仍然可以表现为 daily factor operator

D. Predictive Supervised Learners
    必须拥有独立的 fit / validate / freeze / predict / retrain / artifact 生命周期
    不能再只靠 SeriesOperator.calculate() 在 120 日窗口里临时训练
```

---

# 1. 总体目标架构

## 1.1 新增统一模型分类

建议建立：

```python
class ModelExecutionClass(Enum):
    LOCAL_ROLLING_ESTIMATOR = "local_rolling_estimator"
    RECURSIVE_STATE_ESTIMATOR = "recursive_state_estimator"
    SAME_TIME_CROSS_SECTIONAL = "same_time_cross_sectional"
    PREDICTIVE_SUPERVISED = "predictive_supervised"
    RESEARCH_STRUCTURAL = "research_structural"
```

### LOCAL_ROLLING_ESTIMATOR

典型：

```text
rolling AR beta
rolling OLS beta
rolling PCA loading/residual
HAR local state
GARCH local volatility state
mean-reversion half-life
local covariance/correlation
```

这里：

```text
window = 模型/因子定义的一部分
```

60/120/252 日并没有问题。

它的目的不是训练一个长期 ML predictor，而是估计“近期状态”。

### RECURSIVE_STATE_ESTIMATOR

典型：

```text
Kalman
CUSUM state
state latch
recursive regime state
```

关键 contract：

```text
initialization
state update
missing update
checkpoint
revision replay
time-shard legality
```

### SAME_TIME_CROSS_SECTIONAL

典型：

```text
cs_knn_local_linear_residual
cross-sectional regression / neutralization
same-date peer model
```

关键不是“过去几年训练”，而是：

```text
date t 的 peer universe
date t 的 feature/target availability
self exclusion
cross-sectional sample adequacy
```

### PREDICTIVE_SUPERVISED

典型：

```text
PCR
PLS
ElasticNet
Regime-conditioned predictor
Mixture-of-Experts
未来 RandomForest / LightGBM / NN / Transformer 等
```

这类必须改造成：

```text
multi-year panel training
+ validation
+ OOS test
+ walk-forward
+ purge/embargo
+ scheduled retraining
+ frozen artifact prediction
```

### RESEARCH_STRUCTURAL

典型：

```text
DMD
SSA
RQA
Lyapunov
TE
HSIC
Kernel Granger
复杂 entropy / topology
```

可以保留作为研究型 alpha primitive，但：

```text
research_only / expensive
```

不自动等同于 production predictive model。

---

# 2. 第一项核心大改：把 Predictive Supervised Model 从短窗口 operator 中拆出来

当前类似：

```text
panel_rolling_pcr_forecast
panel_rolling_pls_forecast
panel_rolling_elastic_net_forecast
panel_regime_conditioned_forecast
panel_mixture_of_experts_score
```

必须停止把“per-stock 120 日 rolling fit”作为唯一生产定义。

## 2.1 保留现有实现，但降级为 Legacy / Research Local Variant

不要直接删除，避免破坏 DSL 和历史因子。

建议标记：

```text
panel_rolling_pcr_forecast
    -> LEGACY_LOCAL_ROLLING / RESEARCH

panel_rolling_pls_forecast
    -> LEGACY_LOCAL_ROLLING / RESEARCH

panel_rolling_elastic_net_forecast
    -> LEGACY_LOCAL_ROLLING / RESEARCH

panel_regime_conditioned_forecast
    -> LEGACY_LOCAL_ROLLING / RESEARCH

panel_mixture_of_experts_score
    -> LEGACY_LOCAL_ROLLING / RESEARCH
```

如果允许改 canonical，则新建：

```text
ts_local_pcr_forecast
ts_local_pls_forecast
ts_local_elastic_net_forecast
ts_local_regime_forecast
ts_local_moe_forecast
```

旧名称做 compat alias。

不要把旧行为静默替换成完全不同的 pooled panel 模型，否则历史 factor identity 会改变。

## 2.2 新增真正的 Predictive Model API

建议新建：

```text
factor_engine/modeling/
    __init__.py
    model_spec.py
    training_spec.py
    timing.py
    sample_policy.py
    split.py
    trainer.py
    predictor.py
    artifact.py
    registry.py
    hyperparams.py
    selection.py
    evaluation.py
    leakage_guard.py
    walk_forward.py
    diagnostics.py
    evidence.py

factor_engine/modeling/learners/
    pcr.py
    pls.py
    elastic_net.py
    regime.py
    mixture_of_experts.py
```

预测模型生命周期必须支持：

```python
artifact = trainer.fit(
    model_spec,
    train_dataset,
    validation_dataset,
)

artifact.save(...)

prediction = predictor.predict(
    artifact,
    asof_features,
)
```

禁止 production score path 在预测时重新：

```python
model.fit(...)
```

---

# 3. Predictive Model 正式 Training / Validation / Test 体系

## 3.1 不允许随机 KFold

时间序列 / panel 数据禁止：

```python
KFold(shuffle=True)
train_test_split(shuffle=True)
```

也禁止 stock-date random row split。

必须按 **日期边界** 分割。

## 3.2 建立 WalkForwardSpec

建议：

```python
@dataclass(frozen=True)
class WalkForwardSpec:
    train_lookback_bars: int | None
    train_expanding: bool
    validation_bars: int
    test_bars: int
    step_bars: int
    retrain_every_bars: int

    purge_policy: str
    embargo_bars: int

    min_train_dates: int
    min_train_stocks: int
    min_train_obs: int
```

## 3.3 建议的 A 股默认 preset

不是硬编码真理，而是企业级默认起点。

### Linear / PCR / PLS / ElasticNet

建议：

```text
train_lookback:
    756 ~ 1260 trading days
    约 3–5 年

validation:
    126 ~ 252 trading days
    约 6–12 个月

test fold:
    126 ~ 252 trading days

retrain:
    20 ~ 60 trading days
    月度 / 季度
```

### Regime / Mixture-of-Experts

由于需要分 regime / expert，建议训练期更长：

```text
train_lookback:
    1000 ~ 1500 trading days
    或 expanding history with decay

validation:
    >= 252 trading days

retrain:
    20 ~ 60 trading days
```

不要继续默认：

```text
120 日窗口
3 regimes
每 regime 8 条就训练
```

## 3.4 Expanding vs Rolling

都支持：

### Rolling

```text
过去 3/5 年训练
旧数据滚出
```

### Expanding

```text
从固定起点一直扩张
```

### Decay-weighted expanding

保留更长历史，但旧数据降低权重。

建议支持：

```text
ROLLING
EXPANDING
DECAY_WEIGHTED_EXPANDING
```

Regime / MoE 可优先考虑 `DECAY_WEIGHTED_EXPANDING`。

---

# 4. Panel Training：不要把“每只股票 120 天”当正式预测训练集

真正 supervised model 建议使用 pooled panel：

```text
(stock, date) observations
```

例如：

```text
X_{i,t}
    -> y_{i,t+H}
```

训练集可以达到：

```text
数百/上千交易日 × 数千股票
```

即百万级 raw rows。

## 4.1 训练集统计必须同时记录

不要只记录：

```text
n_rows
```

必须：

```text
n_stock_date_obs
n_unique_dates
n_unique_stocks
n_industries
median_stocks_per_date
min_stocks_per_date
cross_sectional_coverage
label_coverage
effective_date_count
```

因为：

```text
4,000 stocks × 1,000 dates
```

不等于 4,000,000 个独立样本。

同日股票高度相关。

## 4.2 Split 必须以 date 为 authority

同一个 date：

```text
不能部分股票 Train
部分股票 Validation
```

默认所有 date t stock rows 同属于一个 split。

---

# 5. SampleAdequacyContract：替代 scattered `if n < 8/10/25`

新建统一：

```python
@dataclass(frozen=True)
class SampleAdequacyContract:
    min_raw_obs: int
    min_effective_obs: int
    min_unique_dates: int
    min_unique_stocks: int
    min_obs_per_parameter: float

    min_regime_obs: int | None = None
    min_expert_obs: int | None = None
    min_state_transitions: int | None = None
    min_cross_section_peers: int | None = None

    max_missing_fraction: float | None = None
    min_date_coverage: float | None = None
```

## 5.1 必须区分 raw N 与 effective N

输出 telemetry：

```text
raw_obs
finite_obs
mature_label_obs
post_purge_obs
post_regime_obs
effective_obs
free_parameter_count
obs_per_parameter
```

## 5.2 Local estimator 和 Predictive learner 使用不同门槛

### Local rolling estimator

门槛决定：

```text
当前局部估计是否可信
```

可用：

```text
60 / 120 / 252
```

等 window。

### Predictive learner

训练期按：

```text
multi-year dates
× broad cross section
```

治理。

不能再认为：

```text
min_train_obs = 10
```

就满足 production。

---

# 6. Regime / MoE 样本量必须重做

## 6.1 Regime

新 contract：

```text
min_total_dates
min_total_obs
min_obs_per_regime
min_dates_per_regime
min_stocks_per_regime
```

并建立：

```text
n_regimes × min_regime_support <= effective training capacity
```

compile-time / fit-time 双 gate。

## 6.2 MoE

每个 expert 都必须：

```text
有足够训练 observations
有足够 unique dates
design full rank
condition number pass
```

不能：

```text
5 observations
4 predictors + intercept
```

就认为一个 expert 可用。

建议企业级默认：

```text
min_expert_unique_dates >= 60
min_expert_obs >= 1000 pooled stock-date rows
```

具体值可以根据实际 panel 规模配置，但不能恢复成个位数门槛。

## 6.3 不允许 partial expert silently survive

如果请求：

```text
n_experts=3
```

但只有 1 个 expert 足够样本：

```text
不能悄悄变成单模型
```

应：

```text
fail closed
或 explicit fallback policy
```

fallback 必须进入 semantic identity。

---

# 7. Label Contract 必须升级

建议：

```python
@dataclass(frozen=True)
class LabelContract:
    label_name: str
    origin_time: str
    start_time_rule: str
    end_time_rule: str
    availability_time_rule: str

    horizon_bars: int
    overlapping: bool

    return_basis: str
    entry_price_basis: str
    exit_price_basis: str

    purge_by_interval: bool = True
    embargo_bars: int = 0
```

---

# 8. 对本项目，VWAP-to-VWAP 必须成为 Label Authority

例如：

```text
label_t(H)
=
VWAP_{t+H} / VWAP_t - 1
```

必须明确：

```text
entry VWAP 什么时候完成
exit VWAP 什么时候完成
label 什么时候真正成熟
```

不能只写：

```text
label_horizon=H
```

而忽略价格 basis。

---

# 9. Purge 不要简单只写 `purge_bars=H`

更正确：

对于训练样本 s：

```text
label interval = [entry_time(s), exit_time(s)]
```

若其 label interval 与 validation/test 开始区间重叠：

```text
该训练样本必须 purge
```

建立：

```python
label_interval(sample)
```

由 interval overlap 判断。

---

# 10. Embargo

validation/test fold 后可配置：

```text
embargo_bars
```

用于控制高重叠 label / event window 的边界污染。

---

# 11. DecisionClock：隐性未来函数的核心防线

新建：

```python
@dataclass(frozen=True)
class DecisionClock:
    decision_at: str
    feature_available_at: dict[str, str]
    label_available_at: str
    score_written_at: str
    execution_at: str
```

## 11.1 必须支持至少两种 A 股日频场景

### AFTER_CLOSE_TO_NEXT_VWAP

```text
t 日全部收盘数据已知
factor_t 生成
t+1 VWAP 执行
```

### BEFORE_SAME_DAY_VWAP

如果以后存在：

```text
t 日 VWAP 执行
```

则所有 feature 必须在相应执行前可获得。

## 11.2 Same-day target 必须被 availability gate 拦截

如果某 cross-sectional model 使用：

```text
target_t = 当日完整收益
```

那它不能产出：

```text
t 日 earlier VWAP
```

可交易因子。

---

# 12. PIT 数据来源统一交给 DataAccess

FactorEngine 不应自己猜：

```text
财报发布日期
revision
行业成员
ST 状态
指数成分
复权
```

DataAccess 必须提供：

```text
source_identity
asof_timestamp
available_at
revision_vintage
universe_vintage
price_basis
corporate_action_vintage
```

Model artifact 必须保存这些 identity。

---

# 13. 必须防的隐性数据泄漏

## 13.1 Label maturity leakage

未来 H 日收益未成熟不能进入训练。

## 13.2 Scaler leakage

禁止：

```python
scaler.fit(all_history)
```

必须：

```text
fit on train only
```

Validation/Test：

```text
transform only
```

## 13.3 Imputer leakage

均值、中位数、winsor boundary、missing fill model 都只能 train fit。

## 13.4 PCA / dimensionality reduction leakage

禁止：

```text
full-history PCA
-> rolling/predictive model
```

Predictive preprocessing PCA：

```text
fit on train only
freeze for val/test
```

## 13.5 Feature selection leakage

不能用全部历史 IC 选 feature 再回测全部历史。

每 fold：

```text
feature selection only on train/validation
```

## 13.6 Hyperparameter leakage

不能在未来 test 期选：

```text
alpha
l1_ratio
n_components
n_regimes
n_experts
```

## 13.7 Model-selection leakage

不能先看完整历史后决定哪个模型最好，再回填整个历史。

## 13.8 Universe / survivorship leakage

禁止当前存活股票名单回填历史。

## 13.9 Fundamental revision leakage

财务数据必须使用当时可见 revision vintage。

## 13.10 Corporate-action / price adjustment leakage

复权 basis 必须明确 PIT。

## 13.11 Same-day availability leakage

close/full-day VWAP/全天成交量不能驱动同日更早执行。

## 13.12 Cache leakage

缓存 key 必须包含：

```text
asof
source vintage
training cutoff
split id
artifact version
```

禁止未来训练 artifact 被历史调用复用。

## 13.13 Regime-boundary leakage

Regime quantile/cluster/HMM boundary 必须只在允许的 train history 中拟合。

## 13.14 Neutralization leakage

行业/市值中性化所需统计量必须符合当日 as-of universe，不得用未来全样本系数。

## 13.15 Target normalization leakage

若 y 做 rank/zscore/winsorization，边界必须按训练/当日合法横截面定义，不能全历史 fit。

---

# 14. Hyperparameter 体系全面整改

建立：

```python
class ParamRole(Enum):
    ECONOMIC_HORIZON
    MODEL_COMPLEXITY
    REGULARIZATION
    ESTIMATOR_RESOLUTION
    NUMERICAL_POLICY
    DATA_POLICY
    TIMING_POLICY
```

---

# 15. 各类参数是否允许搜索

## ECONOMIC_HORIZON

例如：

```text
window
label_horizon
```

可搜索，但用少量 meaningful presets。

## MODEL_COMPLEXITY

例如：

```text
AR order
PCA n_components
PLS components
n_regimes
n_experts
DMD rank
```

可有限搜索，必须通过 sample adequacy + stability。

## REGULARIZATION

例如：

```text
ElasticNet alpha
l1_ratio
Ridge alpha
Kalman q/r
```

推荐 validation-only tuning 或固定 audited presets。

## ESTIMATOR_RESOLUTION

例如：

```text
TE bins
RQA epsilon
KNN k
DMD embedding dim
SSA rank
```

通常：

```text
searchable=False
```

或小型 reviewed preset。

## NUMERICAL_POLICY

例如：

```text
tol
max_iter
pinv_rcond
EPS
solver
optimizer stopping criteria
```

必须：

```text
searchable=False
```

## DATA_POLICY

例如：

```text
universe
price basis
PIT revision policy
missing policy
```

不能被 FactorMiner 随便搜索。

---

# 16. 新增 SearchPolicy

```python
@dataclass(frozen=True)
class ParameterSearchPolicy:
    role: ParamRole
    searchable: bool
    certified_values: tuple | None
    continuous_bounds: tuple | None
    tune_inside_validation: bool
    contributes_to_factor_identity: bool
    robustness_only: bool
```

---

# 17. 禁止“参数多 = 因子多”

搜索层必须做：

```text
parameter-normalized dedup
```

非法 fractional integer 直接 reject。

---

# 18. Multiple Testing / Search Overfitting

LLM/AlphaProbe/CogAlpha 不断提出模型，本身就是 adaptive search。

必须记录：

```text
how many candidates were tested
which dataset was exposed
which metric was observed
which model family
which parameter trials
```

## 18.1 Candidate Budget

每 family 设置：

```text
max_candidates_per_round
max_parameter_variants
max_feature_sets
max_expensive_models
```

## 18.2 Final holdout 不给搜索算法看

建议四层：

```text
Train
Validation
Research OOS
Final Holdout / Shadow
```

搜索 agent 不得持续读取 Final Holdout。

## 18.3 Test set 被反复查看后就不再是 Test

系统要记录：

```text
dataset_exposure_ledger
```

---

# 19. Walk-Forward 完整训练流程

建议实现：

```python
for fold in walk_forward_splits:
    train = build_train_dataset(fold)
    validation = build_validation_dataset(fold)

    train = purge_unmatured_and_overlap(train, validation)

    preprocessing.fit(train)
    X_train = preprocessing.transform(train)
    X_val = preprocessing.transform(validation)

    for hyperparams in approved_search_space:
        model.fit(X_train, y_train)
        val_score = evaluate(validation)

    best = select_by_validation_only()

    artifact = refit_on_allowed_train_plus_validation_if_policy_allows(best)

    test_prediction = artifact.predict(test_features)

    freeze_test_result()
```

是否 train+validation refit 后测 test 必须固定 policy。

---

# 20. 推荐 Nested Walk-Forward

研究阶段：

```text
Outer fold:
    产生真正 OOS test

Inner fold:
    hyperparameter / model selection
```

避免直接拿 outer test 选参数。

---

# 21. ModelArtifact 必须是一等公民

新增：

```python
@dataclass(frozen=True)
class ModelArtifactManifest:
    model_name: str
    model_version: str
    artifact_id: str

    train_start: str
    train_end: str
    validation_start: str | None
    validation_end: str | None

    decision_clock_id: str
    label_contract_id: str
    feature_schema_hash: str
    data_source_hash: str
    universe_hash: str

    hyperparameters: dict
    preprocessing_state_hash: str

    fit_code_commit: str
    fit_code_component_hash: str

    random_seed: int | None
    solver_version: str | None
```

---

# 22. 线上预测禁止训练

Production scoring：

```text
load frozen artifact
validate as-of compatibility
transform using frozen preprocessing
predict
```

禁止：

```text
score request
-> fit current model
```

---

# 23. 定期重训

建议：

```text
PCR / PLS / ElasticNet
    monthly / quarterly

Regime / MoE
    monthly
```

真正触发机制由生产调度系统负责，FactorEngine 保留 deterministic retraining policy / artifact lookup。

---

# 24. 模型重训不等于每天 rolling refit

把两个概念拆开：

```text
local statistical operator
    可每日滚动重算

predictive learner
    artifact 可在 20/60 日内冻结
```

---

# 25. 过拟合治理

模型越复杂，准入门越高。

建立：

```text
effective_parameter_count
sample_parameter_ratio
model_complexity_score
```

## 25.1 Complexity-aware gate

MoE / 多 regime 比简单 linear model 要更长样本和更强 OOS evidence。

## 25.2 Parameter Stability Test

不要只接受：

```text
argmax Sharpe
```

要求邻域参数性能方向稳定，而不是 razor-thin optimum。

## 25.3 Feature Ablation

复杂 model score 必须做：

```text
remove one feature
remove feature family
shuffle one feature
```

## 25.4 Label Shuffle Negative Control

随机打乱历史 label：

```text
predictive performance 应消失
```

若仍高，优先怀疑 leakage / evaluation bug。

---

# 26. 评估体系不能只看 model loss

预测模型需要同时报告：

```text
MSE / MAE
Rank IC
Pearson IC
ICIR
IC positive ratio
long-short spread
turnover
coverage
industry exposure
size exposure
subperiod stability
year-by-year
bull/bear
large/small cap
liquidity bucket
```

---

# 27. 横截面评估按 date 聚合

优先：

```text
IC_t across stocks
then
mean(IC_t)
std(IC_t)
ICIR
```

不要把所有 stock-date 混成一个总体相关系数。

---

# 28. Model Selection metric 必须 versioned

例如：

```text
validation objective =
0.5 * RankIC
+ 0.3 * ICIR
- 0.2 * turnover_penalty
```

必须进入 artifact/evidence。

---

# 29. Regime / MoE 的额外防过拟合

必须监控：

```text
regime occupancy
expert occupancy
gate entropy
dominant expert ratio
regime transition rate
per-regime OOS IC
per-expert sample count
```

---

# 30. Model Drift

新增：

```text
feature distribution drift
prediction distribution drift
IC drift
coverage drift
regime occupancy drift
coefficient drift
```

Drift 只用于监控 / retrain policy，不允许反向污染历史参数选择。

---

# 31. Preprocessing 必须成为 artifact 的组成部分

需要冻结：

```text
winsorization bounds
normalization mean/std
industry-neutralization settings
PCA projection
imputer
feature order
feature dtype
```

预测时不能重新用未来数据 fit。

---

# 32. Cross-sectional preprocessing 必须定义 as-of

每日 cross-sectional z-score 可以使用同一 date 横截面，前提是这些 peer feature 在 decision time 都已可见。

---

# 33. 当前 `ModelOperatorContract` 必须重构

建议：

```python
@dataclass(frozen=True)
class ModelOperatorContract:
    canonical: str
    execution_class: ModelExecutionClass
    semantic_role: str

    timing_contract_id: str
    sample_contract_id: str
    parameter_policy_id: str
    input_contract_id: str

    training_lifecycle: str
    artifact_required: bool

    stateful: bool
    checkpoint_supported: bool

    cost_class: str
    default_searchable: bool
```

---

# 34. 当前 `ModelTimingContract` 必须升级为 Rich Timing

保留 backward compatibility，但新增：

```python
@dataclass(frozen=True)
class RichModelTiming:
    timing_kind: str

    decision_time: str
    feature_cutoff_rule: str

    train_feature_cutoff_rule: str | None
    train_label_maturity_rule: str | None

    reference_cutoff_rule: str | None
    query_time_rule: str | None

    cross_section_time_rule: str | None
    self_exclusion: bool

    state_update_rule: str | None

    score_time_rule: str
    execution_time_rule: str

    label_contract_id: str | None
```

---

# 35. 至少支持七类 TimingKind

```text
SELF_FIT_DESCRIPTIVE
PRIOR_FIT_PREDICTIVE
PRIOR_REFERENCE_CURRENT_QUERY
SAME_TIME_CROSS_SECTIONAL
RECURSIVE_CAUSAL_FILTER
FORWARD_LABEL_SUPERVISED
MATURED_HISTORICAL_OUTCOME
```

---

# 36. PCR 重构建议

### Local legacy

保留短 rolling per-stock PCR，标 research/local。

### New predictive PCR

训练：

```text
pooled panel
multi-year
train-only scaling
train-only PCA
validation select n_components
```

参数：

```text
train_lookback
n_components
label_horizon
retrain_frequency
feature set
```

`n_components` 属于 `MODEL_COMPLEXITY`。

---

# 37. PLS 重构建议

同 PCR。

额外：

```text
n_components <= effective rank
```

每 fold 用 validation 选。

---

# 38. ElasticNet 重构建议

参数：

```text
alpha
l1_ratio
```

不直接生成海量 factor variants。

推荐：

```text
small log-spaced alpha grid
small l1_ratio grid
inner validation only
```

例如：

```text
alpha:
    [1e-4, 1e-3, 1e-2, 1e-1]

l1_ratio:
    [0.0, 0.25, 0.5, 0.75, 1.0]
```

具体 grid 必须可配置并经过参数域认证。

---

# 39. Regime Model 重构建议

Regime definition 必须仅用：

```text
train history / current available market_state
```

支持：

```text
hard regime
soft regime
```

但不同 semantic identity。

---

# 40. Mixture-of-Experts 重构建议

需要真正定义：

```text
expert models
gating model
gating feature
expert support
expert fallback
```

当前短窗口 quantile-gated local regressions 可以保留为 `simple local MoE`，但不要作为最终 production MoE 定义。

---

# 41. Kalman 参数治理

当前：

```text
q
r
q_level
q_trend
```

应分类：

```text
REGULARIZATION / STATE_NOISE_POLICY
```

不是经济 alpha 参数。

建议：

```text
固定 audited presets
或 historical-only likelihood / EM estimation
```

如果自动估 q/r，必须严格 train cutoff 内估。

---

# 42. GARCH 参数治理

alpha/beta/gamma 内部 MLE 的方向优于让 FactorMiner 随便搜参数。

需要：

```text
MLE only on allowed history
convergence fail-closed
minimum window
state fit telemetry
```

`window` 是 horizon，可有限搜索。

---

# 43. HAR 参数治理

HAR local volatility model 可以继续作为 rolling estimator。

如果未来变成真正 predictive learner，也可 artifact 化，但需要保持 next-RV 与 current surprise timing 区分。

---

# 44. AR / Regression

### Rolling beta / AR coeff

继续 local estimator。

### Predictive forecast

可以保留 prior rolling forecast；如需要长期稳定 predictor，再提供 artifact-backed variant。

---

# 45. PCA

PCA 有两种角色：

### Rolling cross-sectional PCA

当前共同因子结构，保留 operator。

### Predictive preprocessing PCA

必须：

```text
fit train
freeze val/test
```

二者不能混。

---

# 46. KNN / Local Regression

必须区分：

```text
same-time peer geometry
vs
historical predictive learner
```

`cs_knn_local_linear_residual` 属于前者，不需要强行多年 Train/Val/Test，但必须有 same-day peer sample adequacy、PIT universe、target availability、自身排除。

---

# 47. DMD / RQA / TE 等不要假装 supervised model

仍可做 rolling research factor。

重点治理：

```text
window
estimator resolution
sample adequacy
PIT
numerics
cost
```

---

# 48. 新增 Model Registry

```python
ModelRegistry.register(
    model_name="predictive_elastic_net",
    execution_class=PREDICTIVE_SUPERVISED,
    learner=...,
    training_spec=...,
    parameter_policy=...,
)
```

Operator registry 不再是 model lifecycle 的唯一 authority。

---

# 49. Operator 与 Model Artifact 的连接

Factor DSL 最终可以引用模型 score，但运行时必须解析到合法 artifact：

```text
model_score("elastic_net_v3")
```

FactorEngine 负责：

```text
读取该 date 合法 artifact
score
返回 stock×date series
```

而不是现场训练。

---

# 50. Artifact selection 必须 as-of

例如模型每月重训：

```text
2025-05-15
只能加载 training_cutoff <= 2025-05-15 且 available_at <= 2025-05-15 的最新 artifact
```

禁止加载未来训练 artifact 回算历史。

---

# 51. 历史回测必须重建 historical artifacts

Walk-forward backtest：

```text
每个 fold / retrain point 产生独立 artifact
```

历史预测必须使用当时可用 artifact。

禁止：

```text
2026 最终模型
回填 2019–2025
```

---

# 52. Model Artifact Cache Key

至少：

```text
model_name
model_version
train_cutoff
validation_cutoff
feature_schema_hash
label_contract_hash
hyperparams_hash
dataset_vintage_hash
universe_hash
code_hash
```

---

# 53. 随机性

未来随机模型必须保存：

```text
seed
library version
determinism policy
```

---

# 54. Final Holdout / Shadow

建议：

```text
Research OOS
!=
Final Holdout
```

LLM/研究流程不断看到 OOS 后，OOS 也会被过拟合。

---

# 55. Search-agent 数据权限

AlphaProbe / CogAlpha / 其他 LLM miner 应只获取允许的 development metrics，不得持续读取 final holdout 详细表现。

---

# 56. Factor Candidate Ledger

每个由模型产生的因子记录：

```text
candidate_id
model_family
artifact_family
feature_set
parameter_set
train_range
validation_range
OOS_range
number_of_search_attempts_before_selection
selection_metric
```

---

# 57. 防止“看测试集后继续改”

建立：

```text
dataset_exposure_count
```

Final holdout 一旦被查看，后续改动必须视为新的 research cycle。

---

# 58. 新增模型测试体系

必须包括：

```text
unit tests
oracle tests
synthetic recovery
future poison
label poison
scaler poison
hyperparameter poison
universe poison
revision poison
walk-forward split tests
purge overlap tests
artifact as-of tests
historical artifact replay
batch/single parity
backend/reference parity
```

---

# 59. Walk-forward split 单测

测试：

```text
train_end < validation_start
validation_end < test_start
```

并验证 purge 后：

```text
所有 train label_end < validation_start
```

---

# 60. Hyperparameter leak 单测

修改 validation/test 未来数据后：

```text
train-fit scaler
train feature selector
train PCA
train artifact
```

不得改变。

---

# 61. Artifact future poison

修改未来数据：

```text
artifact trained at t
```

的 hash / params / predictions <= t 不得改变。

---

# 62. Same-day execution poison

对 `AFTER_CLOSE_TO_NEXT_VWAP` 允许 close。

对 `BEFORE_SAME_DAY_VWAP`，close/full-day volume 应被编译或 timing gate 拒绝。

---

# 63. Final Model Ledger

新增：

```text
MODEL_FAMILY_LEDGER.parquet
MODEL_ARTIFACT_LEDGER.parquet
MODEL_WALK_FORWARD_LEDGER.parquet
MODEL_SEARCH_EXPOSURE_LEDGER.parquet
```

---

# 64. Hard Gates

新增：

```text
MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH
MODEL_ALL_PREDICTIVE_LEARNERS_ARTIFACT_BACKED
MODEL_ALL_PREDICTIVE_LEARNERS_HAVE_WALK_FORWARD_SPEC
MODEL_ALL_PREDICTIVE_LEARNERS_HAVE_VALIDATION
MODEL_ALL_PREDICTIVE_LEARNERS_PURGE_LABEL_OVERLAP
MODEL_ZERO_RANDOM_TIME_SPLIT
MODEL_ZERO_FULL_SAMPLE_SCALER
MODEL_ZERO_FULL_SAMPLE_PCA_PREPROCESS
MODEL_ZERO_FULL_SAMPLE_FEATURE_SELECTION
MODEL_ZERO_TEST_DRIVEN_HYPERPARAM_SELECTION
MODEL_ZERO_FUTURE_ARTIFACT_BACKFILL
MODEL_ALL_ARTIFACTS_ASOF_RESOLVED
MODEL_ALL_MODELS_HAVE_SAMPLE_ADEQUACY_CONTRACT
MODEL_ALL_SEARCHABLE_PARAMS_HAVE_SEARCH_POLICY
MODEL_ZERO_NUMERICAL_POLICY_SEARCHABLE
MODEL_ZERO_DATA_POLICY_SEARCHABLE_BY_FACTOR_MINER
MODEL_REGIME_ALL_COMPONENTS_HAVE_SUPPORT
MODEL_MOE_ALL_ACTIVE_EXPERTS_HAVE_SUPPORT
MODEL_FINAL_HOLDOUT_NOT_EXPOSED_TO_SEARCH
MODEL_CURRENT_HEAD_EVIDENCE_FRESH
```

---

# 65. 迁移策略：不要一次打断旧 DSL

## Phase A

新增：

```text
modeling/
RichTiming
SampleAdequacy
SearchPolicy
WalkForward
Artifact
```

不改旧结果。

## Phase B

旧 supervised rolling models：

```text
标记 legacy_local_predictive
default_searchable=False
research_only=True
```

## Phase C

实现：

```text
predictive PCR
predictive PLS
predictive ElasticNet
```

artifact-backed。

## Phase D

实现：

```text
predictive Regime
predictive MoE
```

## Phase E

Factor DSL 接入 as-of artifact score。

## Phase F

历史 walk-forward 重新生成 factor outputs。

---

# 66. 不要直接把旧 `min_train_obs=10` 改成一个很大的数字就结束

这是错误整改。

因为没有解决：

- validation；
- test；
- hyperparameter lifecycle；
- artifact；
- scheduled retraining；
- final holdout；
- search exposure governance。

---

# 67. 旧 Local Rolling Predictive Variant 仍要加强样本门槛

因为这些 research 版本仍会被保留。

建议：

```text
PCR / PLS / ENet local:
    min effective obs >= max(60, 10 × free_params)

Regime local:
    min regime obs >= max(30, 10 × free_params)

MoE local:
    min expert obs >= max(30, 10 × free_params)
```

这里只是 research safety floor。

生产 predictive learner 使用多年 panel。

---

# 68. 默认推荐配置

可提供 preset：

```yaml
predictive_linear_default:
  train_mode: rolling
  train_lookback_bars: 1000
  validation_bars: 252
  test_bars: 126
  retrain_every_bars: 20
  purge_by_label_interval: true
  embargo_bars: 5

predictive_regime_default:
  train_mode: decay_weighted_expanding
  min_train_bars: 1000
  validation_bars: 252
  test_bars: 126
  retrain_every_bars: 20
  purge_by_label_interval: true
  embargo_bars: 5
```

这是默认 preset，不是硬编码唯一值。

---

# 69. 参数推荐表

| Model | 参数 | Role | 默认是否给 Factor Miner 搜 |
|---|---|---|---|
| PCR | train_lookback | horizon | 少量 preset |
| PCR | n_components | complexity | 有限 |
| PCR | label_horizon | economic horizon | 可以 |
| PLS | n_components | complexity | 有限 |
| ENet | alpha | regularization | validation 内调 |
| ENet | l1_ratio | regularization | validation 内调 |
| Regime | n_regimes | complexity | 极小 grid |
| Regime | market_state | feature/economic choice | 可以，但视为 feature choice |
| MoE | n_experts | complexity | 极小 grid |
| Kalman | q/r | estimator/state | 默认不直接挖 |
| GARCH | window | horizon | 少量 preset |
| AR | order | complexity | 1–5 等小范围 |
| AR | window | horizon | 少量 preset |
| DMD | rank/dim/delay | estimator resolution | research preset |
| RQA | dim/delay/eps | estimator resolution | research preset |
| TE | bins | estimator resolution | 默认不搜 |
| KNN | k | estimator resolution | 小 preset |
| SSA | components | estimator resolution | 小 preset |

---

# 70. 模型选择稳定性

每个选中的 hyperparameter 要检查邻域性能。

例如最佳 `n_components=3`，则至少检查 2/3/4，不接受只有单点异常优秀、相邻参数全部崩溃的 razor-thin optimum。

---

# 71. OOS stability gate

建议至少看多个 outer folds，例如：

```text
>= 3 个 outer folds
```

不能单个测试年份优秀就 production。

---

# 72. Cross-regime stability

模型至少报告：

```text
bull
bear
high vol
low vol
large cap
small cap
liquid
illiquid
```

不是要求全部为正，而是识别收益依赖。

---

# 73. 训练样本权重

支持：

```text
uniform
time_decay
cross_section_equal_date
```

Pooled panel 默认应考虑 `equal-date weighting`，避免后期股票数量更多导致自然更高权重。

---

# 74. Cross-sectional dependence

百万 panel rows 不能假装 IID。

验证统计建议按 date cluster；训练可考虑 date-balanced sampling / weighting。

---

# 75. Label overlap 对统计显著性的影响

即使训练允许 overlapping labels，评估必须 block-aware / purged。

H=20 的 20 个高度重叠日收益不能当 20 个完全独立 evidence。

---

# 76. Feature lineage

模型 feature 不能只记 `x1/x2/x3`。

Artifact 保存：

```text
factor canonical
parameters
source columns
PIT policy
operator semantic version
```

---

# 77. Model semantic version

以下变化都必须 bump：

```text
training split
label basis
PIT rule
preprocessing
feature set
solver policy
model architecture
regime routing
fallback behavior
```

---

# 78. Model artifact 不应由 FactorEngine 任意隐式重训

Artifact 由 training pipeline / scheduler 生成；FactorEngine 运行时消费。

---

# 79. 性能方案

Predictive models 改 artifact 后通常会更快：

旧：

```text
stock × date × fit
```

新：

```text
每月/每季一次 pooled fit
+
每天 vectorized predict
```

---

# 80. Local estimator 性能仍走 shared state

继续：

```text
PCAState
GARCHFitState
HankelState
KNNGraphState
MarkovState
```

不受 predictive artifact 重构影响。

---

# 81. 最终 Production Model Contract

任何真正 predictive model 在 production 前必须同时拥有：

```text
ModelSpec
TrainingSpec
WalkForwardSpec
LabelContract
DecisionClock
SampleAdequacyContract
ParameterSearchPolicy
PreprocessingSpec
ArtifactManifest
EvaluationEvidence
```

缺任一：

```text
NOT_PRODUCTION_READY
```

---

# 82. 执行 AI 的具体顺序

## Phase 0：Current HEAD Inventory

重新生成：

```text
model canonicals
true predictive models
local models
stateful models
research models
```

不要再把所有 model-like 直接理解成 ML models。

## Phase 1：Contract Architecture

先建：

```text
ModelExecutionClass
RichModelTiming
SampleAdequacyContract
ParameterSearchPolicy
LabelContract
DecisionClock
```

## Phase 2：Legacy Classification

把现有 PCR/PLS/ENet/Regime/MoE 标记 legacy local / research-only / default-search excluded。

## Phase 3：Modeling Package

实现：

```text
split
walk_forward
trainer
artifact
predictor
registry
evaluation
```

## Phase 4：Predictive Linear Family

先做：

```text
PCR
PLS
ElasticNet
```

## Phase 5：Regime / MoE

在 linear family 的 walk-forward artifact 体系稳定后再迁移。

## Phase 6：Factor DSL Integration

让 model score 读取 as-of artifact。

## Phase 7：Leakage Tests

完成 label/scaler/feature selection/hyperparams/artifact as-of/universe/revision/execution clock 负控。

## Phase 8：Walk-forward Evidence

生成多 folds OOS。

## Phase 9：Performance

模型训练复用、向量化预测、artifact cache。

## Phase 10：Hard Gates + Current HEAD Evidence

全绿后才宣称完成。

---

# 83. 最终交付物

必须交：

```text
MODEL_ARCHITECTURE_MIGRATION_REPORT.md
MODEL_CLASSIFICATION_LEDGER.csv
MODEL_SAMPLE_ADEQUACY_LEDGER.csv
MODEL_PARAM_SEARCH_POLICY.csv
MODEL_WALK_FORWARD_SPLITS.csv
MODEL_ARTIFACT_LEDGER.csv
MODEL_DATA_EXPOSURE_LEDGER.csv
MODEL_LEAKAGE_NEGATIVE_CONTROLS.json
MODEL_PIT_TEST_RESULTS.json
MODEL_OOS_EVALUATION.parquet
MODEL_HARD_GATES.json
MODEL_FINAL_ACCEPTANCE_REPORT.md
```

---

# 84. Definition of Done

只有以下全部满足，模型层“大改造”才完成：

```text
[ ] local estimator 与 predictive learner 架构拆开
[ ] PCR/PLS/ENet/Regime/MoE 不再只有短窗口 production 定义
[ ] predictive learner 支持 multi-year panel training
[ ] train/validation/test 按日期分割
[ ] 支持 rolling / expanding walk-forward
[ ] overlapping label interval purge 正确
[ ] validation 只用于选择，test 不参与 tuning
[ ] final holdout 不暴露给 search agent
[ ] preprocessing 全部 train-only fit
[ ] feature selection train/validation-only
[ ] hyperparameter tuning train/validation-only
[ ] model choice 不使用未来 test
[ ] DataAccess 提供 PIT source/revision/universe authority
[ ] DecisionClock 能验证 VWAP-to-VWAP 可交易时间
[ ] historical backtest 使用 historical as-of artifacts
[ ] production score path 不做 fit
[ ] artifact version/hash/source/feature/label 全可追溯
[ ] 所有模型有 SampleAdequacyContract
[ ] 所有参数有 ParamRole / SearchPolicy
[ ] numerical policy 不进入因子搜索
[ ] estimator-resolution 搜索空间受限
[ ] regime/expert support 足够
[ ] 多个 walk-forward OOS folds 通过
[ ] parameter stability 通过
[ ] label shuffle / future poison / scaler poison 等负控通过
[ ] current HEAD evidence freshness PASS
```

---

# 85. 最终设计原则

FactorEngine 模型层以后不要再以：

```text
“这个算子能算出一个值”
```

作为完成标准。

真正标准应该是：

```text
这个模型属于哪种模型？
它什么时候 fit？
它为什么需要这么多样本？
它的训练集覆盖几年？
Validation 怎么选参数？
Test 有没有被看过？
Label 什么时候成熟？
Scaler/PCA/feature selection 有没有穿越？
当前 factor 什么时候可交易？
它用的是哪一个 historical artifact？
artifact 当时是否真的存在？
这个参数是经济参数还是估计器参数？
FactorMiner 有资格搜索它吗？
最终 OOS 是否跨多个 walk-forward fold 稳定？
```

最终目标：

```text
Rolling/State Model
    -> 快速、稳定、局部状态估计、严格 PIT

Predictive Model
    -> 多年 Panel
    -> Walk-Forward
    -> Train / Validation / Test
    -> Purge / Embargo
    -> Hyperparameter Governance
    -> Frozen Artifact
    -> As-of Prediction
    -> OOS Evidence

Factor Mining
    -> 只在正确、受约束、有统计意义的模型空间中搜索
```

这才是适合量化私募企业级 FactorEngine 的模型基础设施。
