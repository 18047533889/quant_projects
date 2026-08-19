# FactorEngine 模型算子全量终审与整改任务书
## —— 282 个 Model-like Canonical 的 Timing、语义、数学、状态、参数域、性能与生产准入统一整改

> **用途**：本文件可直接作为整改 AI / Coding Agent 的执行任务书。  
> **审计对象**：`18047533889/quant_projects` → `factor_engine/`  
> **审计基线**：2026-08-11 当前 `main`，审计时最近 HEAD：`8e9893b562f80e083c4baed2e0a15eb4e040cdc9`。  
> **注意**：执行整改时必须重新读取最新 `main`，不得假定上述 SHA 仍是最新版本。  
> **目标**：不是把所有研究模型强行变成 production，而是让每个 retained model-like canonical 都拥有唯一、可执行、可验证的 timing / role / input / state / parameter / missing / unit / performance contract；所有 direct-use 模型在当前 HEAD 上有完整证据；所有 diagnostic/research/expensive 模型不能误入默认挖掘空间。

---

# 0. 先纠正概念：282 个 model-like ≠ 282 个独立机器学习模型

当前 R35 inventory 曾统计约 **282 个 model-like canonical**。这个集合是“需要模型级审计的高风险拟合/状态/复杂统计量集合”，并不等于 282 个独立 ML 模型。它同时包含：

- Panel PCA / PCR / PLS / ElasticNet；
- Rolling OLS / Huber / Ridge / Quantile / Expectile；
- AR / mean-reversion / variance-ratio；
- Kalman；
- GARCH / GJR / HAR；
- DMD / Hankel / SSA；
- Matrix Profile / sequence anomaly；
- KNN / local regression / local manifold；
- HSIC / Kernel-Granger / Transfer Entropy；
- Markov / regime / state dynamics；
- First Passage；
- Recurrence / RQA；
- Lyapunov；
- entropy / complexity / spectral / wavelet；
- path signature / Mahalanobis anomaly；
- change-point / GLR / Pettitt；
- 以及一批只是因为名称/category 命中 heuristic 而进入 model-like audit set 的统计量。

本轮整改必须拆成：

```text
MODEL_LIKE_AUDIT_SET
    = 所有需要 timing/state/fit/estimator/numerical 审计的 canonical

TRUE_MODEL_SET
    = 真正存在拟合、状态估计、reference model、监督训练或预测结构的 canonical
```

以后不要再把 `model_like_count=282` 对外描述成“282 个机器学习模型”。

---

# 1. 不可违反的全局原则

## 1.1 PIT / Decision Clock 是最高优先级

每个模型都必须明确：

```text
decision_time
feature_available_at
label_origin
label_mature_at
reference/fit_cutoff
query/score_time
execution_time
```

不得仅靠 `fit_cutoff_offset=0/1` 粗略描述所有模型。必须能表达 A 股日频研究的真实 Decision Clock，尤其与 VWAP-to-VWAP 口径兼容：

```text
A. t 日完整数据/收盘后生成 factor_t，t+1 VWAP 执行
B. t 日 VWAP 前生成 factor_t，t 日 VWAP 执行
```

两种时钟下可用信息不同，所有 supervised / same-day cross-section 模型必须做 availability 验证。

## 1.2 “无未来函数”不等于“预测模型”

必须区分：

```text
SELF_FIT_DESCRIPTIVE
PRIOR_FIT_PREDICTIVE
PRIOR_REFERENCE_CURRENT_QUERY
SAME_TIME_CROSS_SECTIONAL
RECURSIVE_CAUSAL_FILTER
MATURED_HISTORICAL_OUTCOME
```

例如窗口包含 t 后拟合 AR，再输出 t 的 fitted value，是 causal descriptive，不是 OOS forecast。

## 1.3 禁止 silent parameter coercion

以下模式全面审计并消除：

```python
window = max(2, int(window))
lag = max(1, int(lag))
m = max(3, int(m))
tau = max(1, int(tau))
```

原因：`lag=0 / 0.9 / 1` 可能编译成同一个真实因子，却保留不同 AST。所有用户/搜索参数必须 strict type + finite + range + relational feasibility；非法参数直接 reject。

## 1.4 Window / Min History / Min Effective Obs 必须分开

统一概念：

```text
window                正式/最大 estimator horizon
min_history           允许首次估计的最小物理历史长度
min_effective_obs     缺失/transition/regime/peer 等筛选后的最小有效样本
```

不得 `window=120` 但模型只用 2/5/10 条历史就无提示输出。

## 1.5 Estimator 参数与 Alpha 参数拆开

PCA components、KNN k、ridge、TE bins、embedding_dim、DMD rank、SSA rank、RQA epsilon 等大多是 estimator-resolution / numerical-policy，不应自动被 AlphaProbe/CogAlpha/FactorMiner 当作新的经济机制方向。默认应 `searchable=False` 或只进 audited preset / robustness lane。

## 1.6 PIT-safe ≠ Alpha-appropriate

建议 role：

```text
ALPHA_PREDICTIVE
ALPHA_REFERENCE_QUERY
STATE_CONDITION_EVENT
MODEL_FEATURE_SCORE
DIAGNOSTIC_DESCRIPTIVE
RESEARCH_ONLY
EXPENSIVE_RESEARCH
DELETE_TOMBSTONE
```

---

# 2. 全局治理与证据体系问题

## M-001【P0】R35 evidence 不是当前 HEAD 证明

历史 R35 evidence 绑定旧 SHA，不能自动证明当前 main。所有新工件必须绑定：

```text
commit_sha
source_component_hashes
fixture_hashes
semantic_version
runtime_versions
generated_at
```

硬门：`bound_sha == current_HEAD`，否则 `STALE`。负控必须覆盖：旧 SHA、kernel 改动、timing 改动、lane 改动、numerical-policy 改动。

## M-002【P0】旧 inventory 与 hard gate 曾存在逻辑冲突

曾出现 `FAST_NATIVE_ALPHA/EXPENSIVE_CERTIFIED_ALPHA` 但 `explicit_timing=False`，同时报告声称 `ALL_MODEL_DIRECT_USE_EXPLICIT_TIMING=TRUE`。重新定义 gate：

```python
for canonical in actual_registry:
    if direct_production_model_like(canonical):
        assert model_timing_contract_is_explicit(canonical)
```

不接受 generated family default 作为生产 timing。

## M-003【P0】Generated timing 只能是 research hint

`_default_contract_for()` 可保留用于 inventory/research diagnostics，但 production admission 必须要求 authored reviewed contract。

## M-004【P0】Timing ontology 不够丰富

建议新增：

```python
class TimingKind(Enum):
    SELF_FIT_DESCRIPTIVE = ...
    PRIOR_FIT_PREDICTIVE = ...
    PRIOR_REFERENCE_CURRENT_QUERY = ...
    SAME_TIME_CROSS_SECTIONAL = ...
    RECURSIVE_CAUSAL_FILTER = ...
    MATURED_HISTORICAL_OUTCOME = ...
```

每类明确 estimation/reference/query/self inclusion/peer inclusion/label maturity/state update/output timestamp。

## M-005【P1】model-like heuristic 只负责召回，不能决定生产语义

`MODEL_LIKE_HINTS` 只用于 audit recall；lane/timing/statefulness/production obligation 必须来自 typed semantic contract。

## M-006【P0】Explicit contract registry 不得保留 dead canonical

建立 gate：`MODEL_TIMING_CONTRACTS.keys()`、`MODEL_OPERATOR_CONTRACTS.keys()`、`MODEL_LANE_EXPLICIT.keys()` 必须都映射 live canonical、明确 compat alias 或 tombstone。清理历史 KNN/HSIC/Granger 等旧名称残留。

## M-007【P0】Lane override 不得覆盖真实 semantic role

不要让硬编码名字把 implementation/semantic 上的 diagnostic 强行改成 `EXPENSIVE_CERTIFIED_ALPHA`。推荐单一方向：

```text
SemanticRole -> TimingKind -> Surface -> Certification -> Lane
```

## M-008【P0】Statefulness 建立唯一 authority

所有 Kalman/CUSUM/latch/episode/recursive state 等统一声明：

```text
stateful
state_schema_version
checkpointable
time_shard_safe
reset_semantics
missing_update_semantics
revision_replay_semantics
```

## M-009【P0】Direct-use canonical 必须逐 canonical 有 oracle obligation

共享 kernel 可以共享数学 oracle，但每个 canonical 必须证明自身 output mapping、timing、role、parameter binding。

## M-010【P0】Causality poison 扩展到所有模型

至少覆盖：future value poison、current-row poison、unmatured label poison、scaler poison、hyperparameter poison、universe poison、peer target poison、missing-gap poison、revision poison。

---

# 3. Panel PCA 家族

涉及 `panel_rolling_pca_loading/resid/explained_ratio/resid_vol/resid_momentum`、`industry_rolling_pca_loading`、intraday PCA variants。

## M-020【P0】PCA `window` 没有真正落实 min/full history

新增 `requested_window/min_history/min_coverage_ratio/absolute_min_obs`。生产默认建议 `min_history=requested_window`。若允许 expanding warmup，必须显式 `warmup_policy` 并进入 semantic identity。

必测：`window=120` 时 full-history 模式 row<119 全 NaN，row=119 才首次 eligible；再单测缺失 coverage。

## M-021【P0】Reconstruction PCA rank policy 被误复用于 PCR

Reconstruction 为避免 full-rank residual=0 可用 `k<=p-1`；PCR 应允许 `k<=p`。拆 `PCAReconstructionPolicy` 与 `PCARegressionPolicy`，或 `rank_policy=` 参数但必须属于 internal semantic policy。

## M-022【P0】PCR 单 feature contract 与 kernel 不一致

若 contract 允许 min feature=1，则 p=1 必须退化成合法标准化线性回归；否则 contract 改 min=2。推荐支持 p=1。

## M-023【P1】`panel_rolling_pca_explained_ratio` 命名错误

实际是 `1-Var(resid_i)/Var(x_i)` 的 per-stock commonality。改 canonical 为 `panel_rolling_pca_commonality`，旧名 compat alias，禁止双重 mining。

## M-024【P1】PCA loading 跨时间 sign orientation 仍不充分

单窗口 deterministic sign 不保证时间稳定。PC1 建议用 fit-window historical equal-weight market factor 做 orientation anchor；PC2+ 若无稳定经济 anchor，优先提供 sign-invariant 输出或留 research。

## M-025【P1】Industry PCA membership 语义必须 contract 化

明确是 `CURRENT_ASOF_MEMBERSHIP_APPLIED_TO_HISTORY` 还是 `HISTORICAL_VINTAGE_MEMBERSHIP_PER_ROW`。

## M-026【P2】Industry PCA 重复 fit

按 `date × unique industry` fit 一次并 broadcast，禁止 `date × stock × same-industry-fit`。

## M-027【P1】PCA downstream rolling warmup 需独立定义

`PCA window=120 + residual-vol window=120` 的首次输出时间必须明确，不能 residual 只有 10 个观测就悄悄变成短窗口统计。

---

# 4. Panel PCR / PLS / ElasticNet / Regime / MoE

## M-030【P1】“panel” 实际是 per-symbol rolling time-series model

文档/catalog 必须写清，不是 pooled stock×date model。若未来需要 pooled panel，另起 canonical。

## M-031【P0】Label maturity 必须由 exact parameters 实例化

不能 contract 固定 H=1、runtime 支持 H=1/3/5/20。实现 `instantiate_timing(canonical, params)`。

## M-032【P0/P1】Supervised loop 是否多 lag 一日必须用 Decision Clock 判定

基于 feature availability 和 label maturity 计算 latest usable origin，禁止只写 `t-fit_lag-label_horizon`。建立 H=1/3/5 + 收盘后/盘前两个时钟的 golden tests。

## M-033【P0】Regime / MoE 必须要求至少一个真正 predictor

`market_state` 是 routing/state variable，不能同时作为唯一 predictor 满足 feature_count。要求 `market_state required` 且 `predictive_feature_count>=1`。

## M-034【P0】MoE supervised timing 不得靠 `_forecast` 后缀

应由 `contract.label_param is not None` 判断 supervised；`panel_mixture_of_experts_score` 也必须拿到 FeatureLabelTiming。

## M-035【P1】Regime 文档承认 current state 用于当前 routing

正确描述：历史估计边界/专家参数，`market_state_t` 选择当前 regime。

## M-036【P1】Regime/MoE 输出 fit-quality telemetry

至少 effective_train_obs、effective_regime_obs、active_expert_count、condition_number、convergence、gate_entropy、expert_weight_max。

## M-037【P2】PLS/ENet/MoE 进入 Budget Lane

不要进入默认海量随机 grammar；cheap primitives 默认，model score 单独 expensive scheduler。

---

# 5. AR / Mean Reversion / Variance Ratio

## M-040【P0】`ts_ar_coefficient` timing mismatch

若实现包含 t：`fit_cutoff=0` 且 descriptive；严格 prior alpha 用 `ts_ar_prior_coeff`。不要两者混。

## M-041【P1】Legacy `ts_ar_forecast` 名称误导

若本质是 in-sample fitted value，compat alias 到 `ts_ar_fitted_value`，不作为独立 mining candidate。

## M-042【P1】Legacy `ts_ar_innovation` 名称误导

若是 self-fit residual，alias 到 `ts_ar_in_sample_resid`；真正 prior innovation 用 `ts_ar_prior_innovation`。

## M-043【P0】AR family authored timing 全覆盖

fitted/in-sample/prior/innovation-z/coeff-stability 全部 authored，不允许同 family 一半 generated。

## M-044【P1】Mean-reversion half-life 输入语义过宽

优先接受 spread/residual/stationary-level/log-relative-price；raw trending price 至少 warning/research gate。

## M-045【P1】Lo-MacKinlay / VR 文档 sign 统一

`VR>1` 对应正序列相关/趋势倾向，`VR<1` 对应负序列相关/均值回复倾向。扫描全部说明，防止方向写反。

---

# 6. OLS / Multi Regression / Huber / Ridge / Quantile / Expectile

## M-050【P0】所有 in-sample regression 明确 diagnostic role

例如 `ts_multi_regression_coeff/resid/r2`、`ts_huber_regression_coeff`、`ts_quantile_regression_coeff/slope`、`ts_expectile_regression_coeff`，若 fit 包含 t，必须 `DIAGNOSTIC_DESCRIPTIVE`，不得因 PIT-safe 进入 predictive alpha lane。

## M-051【P0】Prior regression variants 全部 explicit timing

包括 multi/huber/ridge/quantile/expectile prior variants。

## M-052【P1】`ts_multi_regression_coeff_stability` role 重审

若完全来自 prior fits，可作为 causal model-state alpha，不应仅因名称压成 diagnostic。

## M-053【P1】No-intercept R² convention 唯一化

明确 centered/uncentered；建议 no-intercept 用 uncentered SST=`sum(y²)`，若不改也必须在 semantic version/documentation 固定定义。

## M-054【P1】Adjusted R² 与 intercept convention/DOF 配套

不要无论 design 是否带 intercept 都固定 `(n-1)`。

## M-055【P1】Huber delta versioned

固定 `delta=1.345` 则作为 numerical policy 进入 identity；或有限 certified choices，禁止自由无边界搜索。

## M-056【P1】Ridge alpha 语义统一

不同 family 中暴露/固定 alpha 的差异必须明确；固定值进入 semantic identity。

## M-057【P1】新增 `ts_quantile_beta_spread_prior`

现有 self-fit spread 保持 descriptive；prior 版严格 fit through t-1。

## M-058【P1】Expectile beta spread 同理

## M-059【P2】Quantile LP 为 expensive budget lane

每股×每日×窗口的 HiGHS 不能默认大规模 grammar 滥用。

## M-060【P0】迭代模型 convergence 必须可区分失败原因

ElasticNet、Huber、Expectile、GARCH 等内部 telemetry 至少区分 converged/non-converged/singular/insufficient-sample/invalid-params，而不是全变成无原因 NaN。

---

# 7. Kalman 家族

## M-070【P0】所有 Kalman canonical 都是 stateful

`level/trend/beta/beta_change/beta_uncertainty/innovation_z` 全部 `stateful=True`，并有 checkpoint/time-shard/revision-replay obligation。

## M-071【P0/P1】q/r 绝对尺度不可泛化

同一 q/r 用于 return、price、turnover、amount 代表不同 smoothing。优先增加 dimensionless 模式：`q=cq*scale², r=cr*scale²` 或以 `q/r` 为核心；若保留 absolute q/r，typed unit 必须参与参数域。

## M-072【P1】Kalman Beta 是 through-origin

明确 `y=beta*x+eps`，不是 alpha+beta；可新增二维 state `ts_kalman_alpha_beta`，但不能把现有 beta 描述成含截距动态回归。

## M-073【P1】Kalman Beta typed input 约束

更适合 return-vs-return / excess-return-vs-market；不要自动允许 price-vs-volume 等无意义组合。

## M-074【P1】Warmup contiguity 明确

finite-pair warmup 与 contiguous warmup 二选一并进入 contract。

## M-075【P0】Numba 只放已 parity kernel

Kalman level 若 parity 已证可 dispatch；trend/beta 若仍有漂移必须 reference fallback，禁止为了 Numba 覆盖率强行放行。

---

# 8. GARCH / GJR / HAR

## M-080【防回归 P0】GARCH 参数拟合严格 prior

保持：params fit through t-1；`h_t` 不吃 `r_t`；shock=`r_t/sqrt(h_t)`；next-vol 仅在 t->t+1 update 使用 `r_t`。增加 mutation test 防回归。

## M-081【P0】`ts_gjr_leverage` 必须明确 timing

确认真实 kernel fit-through-t 还是 t-1；前者 descriptive，后者才可 alpha candidate。不得 explicit_timing=False 同时列 expensive alpha。

## M-082【P1】return unit contract 是 authority

price-vs-return runtime heuristic 只能 warning/telemetry，不能代替 typed input。

## M-083【P2】GARCH/GJR family shared fit state

建立 `GARCHFitState/GJRFitState`，一次优化派生 persistence/shock/surprise/next-vol/leverage，禁止多个 canonical 重复 MLE。

## M-084【P1】GARCH missing-gap policy 显式

A 股停牌/缺失时选择 strict-contiguous / state-propagate / fail-closed，不得隐式 drop gaps。

## M-085【P0/P1】HAR 以 FeatureLabelTiming 为权威

`X_s -> RV_{s+1}` 的成熟语义不能只靠单整数 fit_cutoff。

## M-086【P1】HAR window/min-train-obs 拆开

## M-087【P1】HAR-from-return 明确是 daily squared-return RV proxy

真实分钟 RV 存在时优先 minute→daily realized variance。

## M-088【P1】HAR legacy duplicate aliases 去重

---

# 9. DMD / Hankel / SSA

## M-090【P1】DMD research surface 与 lane 一致

如果模块明确 Research/default-search excluded，就不要用容易误解为 production 的 `EXPENSIVE_CERTIFIED_ALPHA`。可改成 `EXPENSIVE_RESEARCH_MODEL`，并明确 “certified-correct ≠ production/default-searchable”。

## M-091【P1】Generic DMD 降级

既已有 level/return typed variants，generic `ts_dmd_*` 应 compat/research，不进正式搜索。

## M-092【P1】DMD physical-mode count data-dependent feasibility

`top_k` 参数合法但共轭合并后 physical modes 不足时 fail-closed是对的，但必须 telemetry：physical_mode_count/rank/top_k/failure_reason。

## M-093【P2】DMDState 共享 Hankel/SVD/eig/modes/energy

## M-094【P1】SSA 当前是 self-fit structural residual

明确 `SELF_FIT_DESCRIPTIVE`，不要描述成 OOS anomaly。

## M-095【建议新增】`ts_ssa_prior_reconstruction_error`

用历史 through t-1 拟合低秩 subspace，当前/最新 segment 作为 query。这是新机制，不是参数变体。

## M-096【P1】Hankel/SSA production strict-contiguous 与 interpolate research 完全隔离

interpolation 必须进入 semantic identity。

## M-097【P2】HankelState 共享 effective-rank/singular-gap/SSA decomposition

---

# 10. Matrix Profile / Sequence Anomaly

## M-100【P0/P1】Timing 改为 PRIOR_REFERENCE_CURRENT_QUERY

当前 subsequence Q_t 与严格历史 subsequences 比较，self-match excluded。不能简单写 fit_cutoff=0 descriptive。

## M-101【P0】禁止 `m=max(3,int(m))`

严格 `m: int>=3`。

## M-102【P1】禁止 `history_window<=0 -> expanding history`

生产必须 finite positive integer；若需要 expanding，另起显式 research mode/canonical。

## M-103【P1】Discord/motif duplicate 只留一个 canonical

相同数学量必须 compat alias，不得双重 mining。

## M-104【P1】所有 profile 派生量明确 historical-only reference

## M-105【P2】O(W²) family 进入 expensive lane

---

# 11. Cross-sectional KNN / Local Regression / Manifold

## M-110【P0】`cs_knn_local_linear_residual` timing 当前语义错误

真实是 date-t peer cross-section fit，self excluded，peer target_t 用于拟合，不是 fit through t-1。新增 `SameTimeCrossSectionTiming(as_of=0,self_excluded=True,peer_feature_available=0,peer_target_available=0)`。

## M-111【P0】绑定 target availability / VWAP Decision Clock

若 target_t 收盘后才可知，则这个 factor_t 不可回测同日 VWAP。建立 same-day availability golden tests。

## M-112【P1】Dynamic KNN 文档 2–4 features 与 API 固定 3 不一致

推荐真正支持 optional f1..f4，min=2 max=4；否则文档改 exactly 3。

## M-113【P0/P1】KNN lag/k 等参数 strict

禁止 `max/int` cast。

## M-114【P1】tie-inclusive KNN 的 k 不是 exact neighbor count

文档改为 kth-distance radius / minimum-k neighbours / boundary ties included。

## M-115【P1】KNN estimator params 默认不搜索

k/ridge/tangent_dim 等进入 preset/robustness lane。

## M-116【P1】Universe PIT 进入 KNN identity

as-of universe、tradable mask、listed/delisted、ST/停牌 policy 都要进入 source/semantic identity。

## M-117【P2】KNNGraphState 共享 peer mean/retention/Dirichlet/local regression/local gradient

---

# 12. Markov / State Dynamics

## M-120【P1】Markov `window=120` 早期 expanding 语义

增加 min_history/min_transitions/min_state_support；若允许 expanding warmup，明确 policy。

## M-121【P1】Discretization policy versioned

quantile edges、tie handling、Jeffreys pseudo-count、empty-state behavior 改动必须改变 semantic identity/evidence hash。

## M-122【P1】Current state 仅 query，historical transition strictly prior

使用 PRIOR_REFERENCE_CURRENT_QUERY contract：edges/P/D1/D2 <=t-1，x_t 只映射 current state。

## M-123【P1】stationary vs empirical distribution 持续分离

stationary surprisal/entropy production/committor/MFPT 按数学定义用正确 distribution。

## M-124【P1】Pseudo-count 不得制造虚假 reachability

MFPT/committor 用 observed support graph 判 reachability。

## M-125【P2】MarkovState family shared

一次生成 edges/state/P/N_obs/pi/D1/D2/centers/support。

---

# 13. First Passage / Barrier Dynamics

## M-130【P0/P1】真正 enforce `unit(scale)==unit(x)`

算法 `x ± barrier*scale` 要求同单位。`log(close)+return-vol` 可合法；`raw close(CNY)+dimensionless return-vol` 不合法。必须 typed relational unit constraint，不能只靠文案。

## M-131【P1】Bias 文档与实现 non-hit anchor 统一

若 fully-observed non-hit contributes 0，则 bias 是 direction×speed×hit-probability；所有 doc/catalog 同步。

## M-132【P1】First Passage 用 MATURED_HISTORICAL_OUTCOME timing

anchor s 只有 `s+horizon<=t` 才可参与 t 输出。

## M-133【P1】scale horizon 是 semantic identity

1-day vol 与 20-day vol 是不同 barrier definition。

## M-134【P2】FirstPassageState 共享 bias/hit-probability/conditional-time 扫描

---

# 14. State Density / Mahalanobis / Reference-query Anomaly

## M-140【P1】统一 ReferenceQueryModel contract

覆盖 state_density、vector_state_mahalanobis/local_density、PCA residual、matrix-profile novelty、signature Mahalanobis：reference <=t-1，current query=t，query excluded from reference。

## M-141【P1】State Density min_periods/window 拆分

`window=60,min_periods=5` 必须明确是 expanding warmup 还是 full-window estimator。

## M-142【P1】Mahalanobis sample floor 与 effective dimension 绑定

保留 drop constant dims + N>=c*p，并输出 p_effective/N_effective/condition telemetry。

---

# 15. Local Lyapunov / Nonlinear Dynamics

## M-150【P0/P1】NaN 压缩后 forward horizon 不再代表物理 bar

当前若 drop interior NaN 后再按 `i+k` 测 divergence，则 k 跨越的真实交易日不固定。推荐生产改 strict trailing contiguous finite；或保留 sparse 模式但回归 `log distance` 对 physical elapsed bars，Theiler 与 horizon 使用同一 physical clock。

## M-151【P0】Lyapunov 所有参数 strict

window/tau/embedding_dim/horizon/min_anchors 禁止 silent clamp。

## M-152【P1】不要把 Lyapunov 输出宣称为“确定性混沌证明”

只描述 local nonlinear divergence feature。

---

# 16. Recurrence / RQA

## M-160【P1】同 family window maturity policy 统一

recurrence_rate 与 diagonal-entropy/trapping/divergence 不应对 `window=40` 有完全不同的有效历史理解而无说明。统一 window/min_effective_fraction/min_periods family policy。

## M-161【P2】顶部文档与 epsilon 实现同步

若真实是 `eps_fraction*robust_scale*sqrt(dim)`，不能继续写 pairwise-distance quantile。

## M-162【P1】dim/delay/eps_fraction/min_line 大多是 estimator resolution

默认限制 preset，不自由搜索。

## M-163【P2】Pairwise RQA O(M²) 进入 expensive lane

---

# 17. Transfer Entropy / Information Theory

## M-170【P1】TE bins/min_cells_ratio 默认不自由搜索

标 `ParamRole.ESTIMATOR_RESOLUTION` 还不够，默认 `searchable=False` 或 preset。

## M-171【P1】window×bins×state-space feasibility compile-time enforce

usable transitions 必须 >= k×effective cells，不能 runtime 全 NaN。

## M-172【P1/P2】Effective TE surrogate policy 相对 window 稳定

固定 offsets 使不同 window 的 surrogate count 变化。建议用 window fractions，或把 surrogate_count/policy digest 进入 semantic identity。

## M-173【P1】Surrogate missing-mask 与 real cohort 一致

保留固定 NaN mask、只重排 finite source values，并建立 regression test。

## M-174【P2】TE family 共享 bins/transition triples

---

# 18. HSIC / Kernel Granger / Nonlinear Dependence

## M-180【P0】清理历史 canonical contract 残留

所有 explicit timing 名称必须映射 live canonical/alias/tombstone。

## M-181【P1】Kernel bandwidth/regularization 属于 estimator policy

默认不自由搜索。

## M-182【P1】OOS Granger 明确 blocked train/test timing

至少 train_end/test_start/test_end/scaler scope/bandwidth selection scope。

## M-183【P1】Residualized HSIC 的 nuisance residualizer 有独立 timing obligation

---

# 19. Path Signature / Signature Mahalanobis

## M-190【P1】Signature Mahalanobis 是 prior reference + current query

历史 signature 用于 scaler/mean/covariance；当前 signature 只 query，更新 timing。

## M-191【P1】跨通道单位说明

历史 robust-standardization 后得到的是 standardized-signature geometry，不是 raw physical signature distance。

## M-192【P1】Depth 固定则从 canonical search signature 移除

compat API 可接受 depth=2，但不应作为搜索参数。

## M-193【P2】Overlapping signature history 用 effective sample size gate，并 telemetry effective_history_count

---

# 20. Change Point / GLR / Pettitt

## M-200【P1】Change-point 是 structural state，不是 next-return predictor

role 用 STATE_CONDITION_EVENT / DIAGNOSTIC_STRUCTURE。

## M-201【P1】GLR cap/scan penalty/noise floor versioned

`_GLR_MAX_LLR/_GLR_SCAN_PENALTY_COEF/_GLR_MEASURE_NOISE_MULT` 改动必须改变 definition identity。

## M-202【P1】Sign convention 全 family 统一

positive=post-regime higher，negative=post-regime lower。

## M-203【P1】Trailing-contiguous/full-window 规则稳定

gap 后不能悄悄变短模型。

---

# 21. Entropy / Complexity / Spectral / Wavelet / Multifractal

这些多数应从 TRUE_MODEL_SET 拆出，但保留 MODEL_LIKE_AUDIT_SET。

## M-210【P1】不要强加 supervised label obligation

分类成 STATISTICAL_ESTIMATOR / COMPLEXITY / SPECTRAL / NONLINEAR_STRUCTURE，各自 test obligations。

## M-211【P1】Estimator sample floor 随状态空间复杂度增长

例如 permutation entropy 要求 effective patterns >= c×order!。

## M-212【P1】Wavelet dead parameter region 清零

若 transform 需要 power-of-two，window 只允许真实不同的 powers-of-two，不允许 65/80/100 全落 64。

## M-213【P1】Spectral/DMD frequency unit 写 per-bar

分钟与日频不能混成无单位 ratio。

## M-214【P2】SpectrumState 共享 FFT/spectrum 派生量

---

# 22. 全模型 Missing Value 统一整改

每个 canonical 必须机器可读地选择：

```text
STRICT_CONTIGUOUS
PAIRWISE_FINITE
DROP_INTERIOR_PRESERVE_PHYSICAL_TIME
PREDICT_ONLY_ON_MISSING
INTERPOLATE_RESEARCH_ONLY
FAIL_CURRENT_ROW_MISSING
```

## M-220【P0】Current-row missing policy

current-query factor 当前行缺失通常必须 NaN，禁止回退昨天窗口输出 stale factor。

## M-221【P0/P1】禁止压缩 NaN 后重定义 lag/horizon

重点审 Lyapunov/TE/Markov/RQA/DMD/Hankel/Matrix Profile。

---

# 23. Typed Input / Unit 全模型整改

每个模型必须定义 input semantic/input unit/input relation/output unit。典型：GARCH→return，HAR→variance/RV，Kalman beta→兼容 return pair，First Passage→scale same unit as x，regression beta→unit(y)/unit(x)。

## M-230【P0/P1】Regression coefficient output unit 真实传播

## M-231【P1】raw residual 与 z-score output unit 分开

---

# 24. Parameter Domain 全模型整改

建立 exact-call certification key：

```text
canonical
semantic_version
backend
execution_variant
source_context
parameter_point
dtype
grain
input_semantic
timing_mode
```

不能 `canonical passed once -> all params certified`。

## M-240【P0】全仓扫描 silent clamp

所有用户/搜索参数上的 `int/max/min` coercion 逐个审。

## M-241【P1】Relational feasibility compile-time prune

DMD、SSA、TE、RQA、GLR、signature effective-history 等全部在 bind/search 阶段拒绝 guaranteed-NaN 参数区。

---

# 25. Stateful / Incremental / Time Sharding

所有 stateful 模型必须证明：

```text
full recompute
incremental row-by-row
checkpoint restore
legal time-shard + checkpoint handoff
```

## M-250【P0】未证明 checkpoint 的 stateful 模型禁止 time shard

Asset shard 可以；time shard 必须有 state handoff 或整段顺序执行。

## M-251【P0】历史 revision 触发 state replay

历史 t0 被修订，stateful 输出通常需 `[t0,∞)` replay，不能只按普通 rolling window 局部重算。

---

# 26. Performance / Shared Intermediate

优先减少重复工作，再谈线程。建立或完善：

```text
PCAState
RollingLinearState
GARCHFitState
GJRFitState
HARDesignState
KalmanState
HankelState
DMDState
KNNGraphState
MarkovState
FirstPassageState
SpectrumState
RQAState
SignatureHistoryState
```

CSE key 必须包含 all semantic params/source identity/universe/price basis/revision vintage/timing mode。

---

# 27. 必须生成的 Canonical Ledger

生成 `MODEL_CANONICAL_LEDGER.parquet/.csv`，至少字段：

```text
canonical
family
true_model_type
semantic_role
surface
lane
timing_kind
explicit_timing
feature_inputs
label_input
label_horizon
stateful
checkpointable
missing_policy
window_semantics
min_history
min_effective_obs
input_units
output_unit
parameter_domain_certified
oracle_required
oracle_pass
causality_pass
future_poison_pass
label_poison_pass
missing_gap_pass
unit_contract_pass
backend_parity_pass
batch_single_parity_pass
optimized_reference_parity_pass
performance_lane
default_searchable
final_direct_use_ready
failure_reasons
evidence_sha
```

`final_direct_use_ready` 必须由子 gate 推导，不允许人工填 True。

---

# 28. 必须建立的测试矩阵

## 28.1 Independent Oracle

至少 family 级独立数学 oracle，且每 canonical 验证 output mapping：PCA/PCR/PLS/ElasticNet/OLS/Huber/Ridge/Quantile/Expectile/AR/Kalman level-trend-beta/GARCH/GJR/HAR/DMD/Hankel/SSA/Matrix Profile/KNN local/Mahalanobis/Markov/First Passage/Lyapunov/RQA/TE/HSIC/Kernel Granger/Change Point。

## 28.2 Synthetic Recovery

种植已知结构：AR(1)、GARCH(1,1)、Kalman local level、dynamic beta、latent PCA、piecewise regime shift、known Markov chain、sinusoid DMD/spectral、repeating motif、known KNN manifold、first-passage drift、known TE direction。

## 28.3 Causality Poison

- future rows >t 改成 NaN/extreme/random，输出 <=t 不变；
- current-row poison 根据 TimingKind 应按预期变化；
- unmatured label poison 不得影响 fit；
- KNN peer/self poison 验证 self-exclusion；
- future universe/member poison 不得影响过去；
- scaler/hyperparameter selection poison 单独测。

## 28.4 Missing-gap

连续窗口、中间 1 NaN、连续 5 NaN、current NaN、长停牌 gap。

## 28.5 Parameter mutation

每整数参数测试 `1/1.0/1.5/0/-1/NaN/Inf/True`。

## 28.6 Semantic mutation

故意引入以下错误，测试必须杀掉：

```text
AR prior include current
PCA prior include current
Matrix Profile allow self-match
First Passage allow s+H>t
KNN include self
GARCH fit current shock
TE lag after NaN compression
```

---

# 29. 生产准入 Hard Gates

至少：

```text
MODEL_ZERO_DEAD_EXPLICIT_CONTRACT_KEYS
MODEL_ZERO_UNCLASSIFIED_TRUE_MODELS
MODEL_ALL_DIRECT_USE_HAVE_EXPLICIT_TIMING_KIND
MODEL_ALL_DIRECT_USE_HAVE_TYPED_INPUTS
MODEL_ALL_DIRECT_USE_HAVE_PARAMETER_DOMAIN
MODEL_ALL_DIRECT_USE_HAVE_ORACLE
MODEL_ALL_DIRECT_USE_HAVE_CAUSALITY_EVIDENCE
MODEL_ALL_DIRECT_USE_HAVE_MISSING_POLICY_EVIDENCE
MODEL_ALL_DIRECT_USE_HAVE_UNIT_EVIDENCE
MODEL_ALL_STATEFUL_HAVE_STATE_CONTRACT
MODEL_ALL_STATEFUL_TIME_SHARD_SAFE_OR_FORBIDDEN
MODEL_ALL_OPTIMIZED_PATHS_REFERENCE_PARITY
MODEL_ZERO_INSAMPLE_DIAGNOSTIC_IN_PREDICTIVE_LANE
MODEL_ZERO_GENERATED_TIMING_USED_FOR_PRODUCTION
MODEL_ZERO_SILENT_PARAMETER_CLAMP
MODEL_ZERO_DUPLICATE_MINING_CANONICAL_ALIASES
MODEL_CURRENT_HEAD_EVIDENCE_FRESH
MODEL_NEGATIVE_CONTROLS_ALL_FIRE
```

---

# 30. 文件级整改顺序

第一批先修 ontology / role / lane / production gate：

```text
factor_engine/cleaned_operators/model_timing.py
factor_engine/cleaned_operators/model_contract.py
factor_engine/cleaned_operators/model_lane.py
factor_engine/cleaned_operators/operator_surface.py
```

第二批核心数学模型：

```text
cross_section/panel_model.py
cross_section/pca_state.py
ts_model/ar_meanrev.py
regression_models.py
ts_model/dynamic_regression.py
ts_model/state_space.py
ts_model/volatility.py
```

第三批高级模型/统计状态：

```text
dmd.py
hankel.py
ts_model/sequence_anomaly.py
candle_state_space.py
dynamic_knn.py
cross_section_local.py
markov_dynamics.py
state_geometry.py
first_passage.py
local_lyapunov.py
recurrence_analysis.py
advanced_information.py
research_transform.py
glr_change.py
```

第四批：shared family states、fast path、Numba、batch CSE、performance router。

---

# 31. P0 优先整改清单

1. 当前 HEAD 重新生成 model inventory/evidence；
2. direct-use model 无 explicit timing 全部补齐；
3. 建立六类 TimingKind；
4. `ts_ar_coefficient` timing/role；
5. in-sample Huber/Quantile/Expectile 移出 predictive alpha lane；
6. PCR rank policy 与 1-feature；
7. PCA full/min-history；
8. Regime/MoE 至少一个 predictor；
9. MoE supervised timing 不准靠名字后缀；
10. Kalman 全 family stateful；
11. `ts_gjr_leverage` timing；
12. KNN local same-time cross-section timing；
13. Matrix Profile/Signature/State anomaly 改 reference-query timing；
14. Local Lyapunov physical-time horizon；
15. 全 model 参数 silent clamp 清零；
16. First-Passage unit relational constraint；
17. contract dead names 清理；
18. stateful time-shard/checkpoint fail-closed；
19. direct-use canonical 全量 oracle/causality ledger；
20. evidence current-HEAD + negative control。

---

# 32. P1 语义与搜索空间清单

1. PCA explained_ratio → commonality；
2. PCA sign orientation；
3. industry membership basis；
4. supervised Decision Clock；
5. regression R² convention；
6. Huber delta / Ridge alpha versioned；
7. quantile/expectile spread prior variants；
8. Kalman q/r scale-safe；
9. Kalman beta through-origin 明示；
10. HAR RV proxy 明示；
11. DMD generic canonical 降级；
12. SSA prior anomaly；
13. KNN 2–4 feature contract；
14. KNN kth-radius 语义；
15. Markov min-history；
16. First Passage non-hit 文档；
17. RQA family window policy；
18. TE estimator params 不自由搜索；
19. Effective TE surrogate policy；
20. change-point role/sign；
21. entropy/complexity 从 TRUE_MODEL_SET 拆出。

---

# 33. P2 性能清单

优先级：shared fit/decomposition > rolling sufficient statistics > Numba recurrence loops > BLAS/LAPACK shared matrix ops > budget lanes > CSE > batch execution。不要无脑 multiprocessing。

---

# 34. 明确禁止的“修复”

1. 不得为了 gate 全绿把 research 模型强行 production；
2. 不得给 generated timing 简单加 `reviewed=True`；
3. 不得用更多 `max/min/int` 来“消灭异常”；
4. 不得只改 doc 不改 runtime，也不得只改 runtime 不同步 contract/lane/tests/evidence/search grammar；
5. 共享数学后仍要保留 self-fit/prior-fit/level/return 等 semantic identity；
6. 不得把 NaN 全部 drop 后继续假装 lag/horizon 仍是原物理时间；
7. 不得用 presence-only 测试证明 correctness；
8. 不得因为“有 timing object”就宣称 timing 已审计；
9. 不得因为一个 family 代表算子通过就把整个 family 全标 PASS。

---

# 35. 最终交付物

```text
1. MODEL_CURRENT_INVENTORY.csv/parquet
2. MODEL_CANONICAL_LEDGER.csv/parquet
3. MODEL_TIMING_LEDGER.csv
4. MODEL_ROLE_LANE_LEDGER.csv
5. MODEL_PARAMETER_DOMAIN_LEDGER
6. MODEL_ORACLE_RESULTS
7. MODEL_CAUSALITY_RESULTS
8. MODEL_MISSING_POLICY_RESULTS
9. MODEL_STATEFUL_CHECKPOINT_RESULTS
10. MODEL_BACKEND_PARITY_RESULTS
11. MODEL_OPTIMIZED_REFERENCE_PARITY_RESULTS
12. MODEL_NEGATIVE_CONTROL_RESULTS
13. MODEL_CURRENT_HEAD.json
14. MODEL_FINAL_HARD_GATES.json
15. MODEL_FINAL_ACCEPTANCE_REPORT.md
```

---

# 36. 每个 canonical 最终只能落三种状态

```text
A. DIRECT_USE_CERTIFIED
B. RESEARCH_OR_DIAGNOSTIC_RETAINED
C. DELETE/TOMBSTONE
```

禁止 unknown / sort-of-production / probably-safe。

---

# 37. Definition of Done

只有以下全部完成，才可以回答“模型部分处理完了”：

```text
[ ] 当前 HEAD model-like inventory 已重新生成
[ ] TRUE_MODEL_SET 与 statistical-estimator 集合已拆分
[ ] 每个 direct-use model 有 explicit authored timing
[ ] timing ontology 能表达 self-fit/prior/reference-query/cross-section/filter/matured-outcome
[ ] 所有 in-sample diagnostic 不再冒充 predictive alpha
[ ] 所有 stateful canonical 有 state contract
[ ] 所有参数无 silent clamp
[ ] 所有 direct-use model 有 typed input/unit
[ ] 所有 direct-use model 参数域 exact-call certified
[ ] 每个 direct-use canonical 有 oracle obligation
[ ] 每个时间模型有 future perturbation
[ ] supervised model 有 label maturity poison
[ ] cross-sectional model 有 as-of/universe/self-exclusion测试
[ ] reference-query model 明确 current query 不进入 reference
[ ] missing/gap semantics 全 family 明确
[ ] batch == single
[ ] optimized == reference
[ ] fast path 仅在 parity 后启用
[ ] stateful full/incremental/checkpoint 等价或明确禁止 time-shard
[ ] aliases 不重复进入 mining
[ ] estimator-resolution 参数不污染默认搜索空间
[ ] current HEAD evidence freshness PASS
[ ] negative controls 全部能把错误实现打红
[ ] full regression suite PASS
```

---

# 38. 执行 AI 的工作方式

```text
Phase 0
    读取最新 main
    重建 inventory
    对本文件每条 issue 标 OPEN / ALREADY_FIXED / PARTIAL / NOT_APPLICABLE

Phase 1
    修 timing / role / lane / contract ontology

Phase 2
    修 P0 数学和参数语义

Phase 3
    修 stateful / missing / unit / Decision Clock

Phase 4
    修 search-space / alias / estimator-resolution

Phase 5
    family shared state + performance

Phase 6
    全量 oracle / causality / parameter / missing / backend / batch / mutation

Phase 7
    重新生成 current-HEAD evidence

Phase 8
    hard gates 全通过后才允许宣称完成
```

对已经被后续代码修好的问题，不要重复破坏正确实现；必须用当前 HEAD 行为测试证明 `ALREADY_FIXED`。无法证明的问题不得写 PASS，标 `NOT_PROVEN/NOT_RUN`。

---

# 39. 最终原则

FactorEngine 模型层的竞争力不是“282 个名字”，而是：

```text
282 个 model-like/statistical operators
    ↓
严格语义分类
    ↓
正确 PIT / Decision Clock
    ↓
正确参数域与单位
    ↓
严格生产准入
    ↓
共享高性能执行
    ↓
让 AlphaProbe / CogAlpha / 其他自动挖掘算法只在“正确且有意义”的模型搜索空间里搜索
```

最终必须做到：对于任意一个进入搜索空间的模型，系统能回答——它是什么、什么时候拟合、什么时候可用、当前行是否参与拟合、输入是什么单位、输出是什么含义、是否 stateful、如何处理缺失、哪些参数是 estimator resolution、是否昂贵、当前参数点在哪条 evidence 上被证明正确。
