# FactorEngine R22：剩余算子全量 Direct-Use 化——Research 迁移、因果改写与“所有可用算子都能参与挖掘”的最终整改提示词

> **用途**：本文件直接交给代码整改 AI，在服务器/本地当前 `factor_engine` 工作区执行。  
> **独立性**：这是一个新的独立 R22。不要与 R17/R18/R19/R20/R21 合并。  
> **前提**：R17–R21 已提出的问题视为其它 AI 正在并行处理。本轮只围绕一个目标：**把剩余不能真正被 AlphaProbe / AlphaMiner / FactorMiner / CogAlpha / AlphaSage / EvoAlpha / AlphaCFG / QuantaAlpha 使用的 operator 全部收口。**  
> **最新 GitHub 可见基线**：`b947c690a119c6fe42fe51b7fd2f9b5f0c746807`。服务器本地若更新，以真实 HEAD/dirty tree 为准。  
> **工作方式**：不要只写报告；不要只改下面点名的 operator；必须 `load_all()` 后动态枚举当前所有 canonical，再根据本文件规则逐个处理。  
> **最终目标不是“所有算子都能作为最终因子 terminal”**，而是：  
>
> ```text
> 除了真正应该删除/隔离的 operator，
> 每一个保留的 factor/operator capability
> 都必须能被自动挖掘 grammar 合法调用。
> ```
>
> 也就是说：
>
> ```text
> Alpha       可以 terminal / intermediate
> State       可以 gate / interaction
> Condition   可以 condition
> Event       可以 event/mask
> GroupState  可以 group context
> GlobalState 可以 market context
> Intermediate可以参与组合
> SourceTransform 可以在数据处理节点使用
> ControlFlow 可以 where/trade_when/gate
> HighCost    可以在受预算约束的深度搜索 lane 使用
> ```
>
> **Research 不能继续成为“暂时不知道怎么处理，所以先不让矿工看”的垃圾桶。**

---

# 0. 本轮 Definition of Done

最终必须满足：

```text
PUBLIC_RETAINED_CANONICAL_NOT_MINING_VISIBLE = 0

FACTOR_SHAPED_RESEARCH_SURFACE = 0
UNRESOLVED_DIRECT_ROLE = 0
DIRECT_ROLE_BUT_LEGACY_ALLOWLIST_BLOCKED = 0
DIRECT_ROLE_BUT_PRODUCTION_DENIED_CONFLICT = 0
DIRECT_ROLE_BUT_NO_GRAMMAR_POSITION = 0
DIRECT_ROLE_BUT_NO_LEGAL_INPUT_BINDING = 0
DIRECT_ROLE_BUT_NO_SMOKE_RECIPE = 0

RESEARCH_FACTOR_WITH_CAUSAL_REWRITE_AVAILABLE = 0
BENCHMARK_SELF_INCLUSION_OPERATOR_WITHOUT_EX_SELF_OR_EXPLICIT_BENCHMARK = 0
IN_SAMPLE_DIAGNOSTIC_EXPOSED_TO_DEFAULT_MINING = 0
HIGH_COST_FACTOR_EXCLUDED_ONLY_BECAUSE_IT_IS_HIGH_COST = 0

PUBLIC_OPERATOR_THAT_ALWAYS_RAISES = 0
PUBLIC_NO_DATA_OPERATOR = 0
PUBLIC_EXACT_DUPLICATE = 0
PUBLIC_NONCAUSAL_OPERATOR = 0
PUBLIC_RANDOM_OPERATOR = 0
```

允许长期不进入 factor-mining 的，只剩：

```text
1. 真正未来函数 / lookahead
2. 真随机生成器
3. 精确语义重复且已有 canonical replacement
4. 纯研究诊断，且没有合理的 causal factor 版本
5. 当前 A/US 现有字段无法支持，且不能由现有字段合理推导
6. 原始矩阵/算法 helper，不是 panel factor operator
7. compatibility/legacy alias
```

---

# 1. 先纠正一个核心概念：Direct Use != Terminal

## R22-001
当前 `mining/direct_use.py` 的 `DirectUseStatus` 已经区分：

```text
DIRECT_ALPHA
DIRECT_ALPHA_HIGH_COST
DIRECT_STATE
DIRECT_CONDITION
DIRECT_EVENT
DIRECT_GROUP_STATE
DIRECT_GLOBAL_STATE
DIRECT_INTERMEDIATE
DIRECT_SOURCE_TRANSFORM
DIRECT_RECIPE
DIRECT_CONTROL_FLOW
```

这个方向是正确的。

## R22-002
但当前 `build_direct_use_operator()` 中：

```python
directly_usable = (
    contract.status in _DIRECT_TERMINAL_STATUSES
    and production_certified
    ...
)
```

导致：

```text
DIRECT_STATE
DIRECT_CONDITION
DIRECT_EVENT
DIRECT_GROUP_STATE
DIRECT_GLOBAL_STATE
DIRECT_INTERMEDIATE
DIRECT_SOURCE_TRANSFORM
DIRECT_CONTROL_FLOW
```

从 `directly_usable` 字段看永远是假。

这会持续制造“这些算子不能直接用”的认知混乱。

## R22-003
废止单一 `directly_usable` 作为总判据，拆成：

```python
mining_visible: bool
composition_usable: bool
terminal_usable: bool
production_admitted: bool
context_admitted: bool
```

## R22-004
定义：

```text
mining_visible
= retained public capability
∧ semantic role resolved
∧ not delete/research-tool/internal-only
```

## R22-005
定义：

```text
composition_usable
= mining_visible
∧ has legal AST position
∧ input/output contract complete
```

## R22-006
定义：

```text
terminal_usable
= composition_usable
∧ status ∈ {DIRECT_ALPHA, DIRECT_ALPHA_HIGH_COST, DIRECT_RECIPE}
```

## R22-007
定义：

```text
production_admitted
= production certification + source/context + execution/cost gates
```

## R22-008
一个 State/Event/Intermediate 可以：

```text
mining_visible=True
composition_usable=True
terminal_usable=False
```

这正是我们要的。

---

# 2. “Research” 与 “未认证”必须彻底拆开

## R22-009
当前 `operator_catalog.assign_mining_role_ex()`：

```text
surface == research -> MiningRole.RESEARCH
```

这是本轮要重点改的结构性问题。

## R22-010
Research surface 是 authoring/lifecycle 状态，不应该自动决定 operator 的语义角色。

一个：

```text
causal
deterministic
panel-shaped
有经济/统计含义
只是尚未完成 production evidence
```

的 operator，应该是：

```text
DirectUseRole = ALPHA/HIGH_COST/INTERMEDIATE/STATE/...
AdmissionState = PENDING_CERTIFICATION
```

而不是：

```text
MiningRole = RESEARCH
```

## R22-011
彻底正交化：

```text
AuthoringTier
DirectUseRole
AdmissionState
ProductionCertification
BackendCapability
CostLane
```

## R22-012
`surface=research` 不再自动把 factor-shaped operator排除出 mining ontology。

## R22-013
真正的 ResearchTool 只放：

```text
ResearchToolRegistry
```

而不是留在公共 Factor DSL registry。

---

# 3. 当前代码本身存在“Research surface 应该为空”与动态 Research 注册的矛盾

## R22-014
`operator_surface.py` 顶部已经写：

```text
research is intentionally empty for factor-shaped operators.
Anything that returns a factor panel and remains in the DSL must be production hardened.
```

## R22-015
但当前又动态注册了一批 factor-shaped research canonical。

必须把这个架构原则真正执行到底：

```text
factor-shaped + useful
-> direct role + pending/certified

pure diagnostics
-> ResearchToolRegistry

obsolete/broken
-> delete/migration
```

---

# 4. 当前 mining integration 仍没有真正消费 DirectUse authority

## R22-016
当前 `api/mining_integration.py::default_mining_operator_allowlist()` 仍主要读取：

```text
backend.fastpath_allowlists
research_allowlist
production_allowlist
production_fastpath_allowlist
```

## R22-017
这意味着一个 operator：

```text
数学正确
production-certified
DirectUse role正确
但只有 Pandas backend
```

可能仅因为“不在 production_fastpath”就不进入默认 mining。

这与你们“单 backend production 也允许，引擎运行时选最快 backend”的目标冲突。

## R22-018
默认 mining authority 改为：

```python
get_direct_use_mining_operators(context, admission="eligible")
```

而不是 fastpath allowlist。

## R22-019
Fastpath只作为：

```text
execution preference
performance lane
cost discount
```

不能决定 operator 是否有资格被挖。

## R22-020
新的 mining tiers：

```text
direct_standard
direct_high_cost
direct_all_context
research_tools  # 不给自动因子矿工
```

不要再用：

```text
research / production / production_fastpath
```

来混淆语义准入和 backend速度。

---

# 5. 当前 typed mining config 仍把 scalar params 塞进 inputs

## R22-021
当前代码已经：

```python
scalar_params, panel_inputs = split_scalar_panel_params(...)
```

但仍写：

```python
"inputs": params
"panel_inputs": panel_inputs
"scalar_parameters": scalar_params
```

## R22-022
整改：

```text
inputs == data_inputs only
```

## R22-023
最终 signature schema：

```json
{
  "data_inputs": [],
  "scalar_parameters": [],
  "condition_inputs": [],
  "event_inputs": [],
  "group_inputs": [],
  "context_inputs": [],
  "source_inputs": []
}
```

## R22-024
不要再保留含糊的 `params -> inputs` 兼容行为给新矿工。

---

# 6. 当前 terminal_allowed 又回到了负面排除

## R22-025
当前 mining integration 有：

```python
terminal_allowed = role.value not in {
    "state", "condition", "event", "group_state", "global_state"
}
```

## R22-026
这会让：

```text
internal
diagnostic
research
legacy
denied
unresolved
source_transform
```

如果误入 allowlist，再次成为 terminal。

## R22-027
只允许：

```python
terminal_allowed = direct_use_contract.terminal_usable
```

或正向集合。

---

# 7. mining integration 中 `except Exception: pass` 必须清零

## R22-028
role/direct-use/signature resolution失败：

```text
production/direct mining hard fail
```

不能生成缺 role 的 operator signature继续挖。

---

# 8. 所有保留 public canonical 必须“矿工可见”

## R22-029
建立：

```python
PublicMiningDisposition
```

仅允许：

```text
DIRECT_VISIBLE
MOVED_RESEARCH_TOOL
MOVED_INTERNAL
DELETED_REPLACED
DELETED_NONCAUSAL
DELETED_NO_DATA
DELETED_DUPLICATE
DELETED_OBSOLETE
```

## R22-030
禁止：

```text
PUBLIC_BUT_HIDDEN
PUBLIC_RESEARCH
PUBLIC_PENDING_FOREVER
PUBLIC_UNKNOWN
```

---

# 9. Research-surface：当前已确认的 factor-shaped operators

下面是当前最新可见代码中明确发现的 Research factor families。

整改 AI 还必须动态枚举，下面不是全量硬编码替代。

---

# 10. `research_transform` 三个算子：全部有机会进入 Direct Use

## R22-031 `ts_wavelet_lowpass_reconstruct`

当前特点：

```text
strict trailing
causal
deterministic
Haar
window只允许 32/64/128/256
cross-param feasibility已声明
```

它不是“不能用”，只是**输出更适合做 intermediate，而不是裸 terminal alpha**。

### 改法

迁移：

```text
RESEARCH_TOOL/RESEARCH
→ DIRECT_INTERMEDIATE
```

### 输入契约

```text
x: continuous numeric panel
unit: preserved from input
axis: ts/per-instrument
```

### 输出

```text
output_unit = same_as_input
output_semantic = smoothed_component
```

### 合法 parent

优先允许：

```text
subtract(x, lowpass)
safe_div(x-lowpass, scale)
ts_delta(lowpass)
ts_time_slope(lowpass)
cs_rank(...)
cs_zscore(...)
```

### 不允许

默认裸 terminal：

```text
false
```

因为输入若是 price level，lowpass本身仍是price level。

### 自动组合模板

为矿工暴露 recipe：

```text
wavelet_residual = x - ts_wavelet_lowpass_reconstruct(x,w,l)
wavelet_residual_z = wavelet_residual / ts_std(x,w)
wavelet_trend_slope = ts_time_slope(lowpass,w2)
wavelet_level_distance = (x-lowpass)/abs(lowpass)
```

这样这个算子真正进入搜索空间。

---

## R22-032 `ts_signature_mahalanobis_anomaly`

当前实现已经具备：

```text
causal
strict trailing
3-channel path signature
history-only robust scaling
effective sample gate
fixed depth=2
cross-param feasibility
deterministic
```

这不是 ResearchTool。

迁移：

```text
→ DIRECT_ALPHA_HIGH_COST
```

### 输入 slot必须显式

```text
f1: feature_continuous
f2: feature_continuous
f3: feature_continuous
```

不是任意字符串/标量。

### 默认 input recipe

至少提供真实可用组合：

```text
(return_decimal, volume_zscore, intraday_range_ratio)
(return_decimal, turnover_zscore, volatility)
(momentum, liquidity, volatility)
```

只能使用当前数据字段/可构造因子。

### output

```text
unit: dimensionless anomaly distance
domain: >=0 or NaN
terminal_allowed: true
cost lane: high_cost
```

### 参数

```text
path_window
history_window
```

进入 reviewed coarse grid。

`depth` 保持 fixed，不进入search。

---

## R22-033 `ts_persistence_birth_dispersion`

当前：

```text
causal
deterministic
Rips H1
dim>=2
tau/window validated
ratio output
```

迁移：

```text
→ DIRECT_ALPHA_HIGH_COST
```

### 输入限制

不要允许任意 raw accounting level。

优先：

```text
return
log_return
normalized price path
volatility-normalized path
```

### output

```text
dimensionless topology dispersion
terminal true
high_cost lane
```

### search grid

dim/tau是 estimator resolution，使用coarse/fixed grid。

---

# 11. DMD：不要把整个 family 永久放 Research

当前 `dmd.py` 注册：

```text
ts_dmd_dominant_growth_rate
ts_dmd_dominant_frequency
ts_dmd_mode_concentration

ts_dmd_level_dominant_growth_rate
ts_dmd_level_dominant_frequency
ts_dmd_level_mode_concentration

ts_dmd_return_dominant_growth_rate
ts_dmd_return_dominant_frequency
ts_dmd_return_mode_concentration
```

并统一 `union_research()`。

这是过度保守。

---

## R22-034
三个 generic/untyped：

```text
ts_dmd_dominant_*
```

语义含糊：

```text
raw level vs return dynamics不同
```

不要直接矿。

处理：

```text
DEPRECATED_COMPAT_ALIAS / MOVE_INTERNAL
```

指向 typed variant，不作为独立搜索 canonical。

---

## R22-035
三个 `ts_dmd_level_*`：

迁移：

```text
→ DIRECT_ALPHA_HIGH_COST
```

输入：

```text
continuous_close / log_price_level
```

不要喂 arbitrary accounting ratio。

---

## R22-036
三个 `ts_dmd_return_*`：

迁移：

```text
→ DIRECT_ALPHA_HIGH_COST
```

输入：

```text
return_decimal
log_return
residual_return
```

---

## R22-037
DMD的：

```text
rank
dim
delay
top_k
```

是 estimator resolution，不应该大范围搜索。

保留 small reviewed grids。

`window` 才是主要 horizon。

---

## R22-038
DMD已有 relational feasibility：

```text
window-(dim-1)*delay >= rank+2
rank <= dim
top_k <= rank
```

矿工必须直接读取这些约束，不靠 runtime 出NaN后再淘汰。

---

## R22-039
DMD无需因为 Polars只是 bridge就永远 Research。

若：

```text
pandas_numpy reference backend
```

通过 production numerical/temporal/source evidence，

则允许：

```text
single-backend DIRECT_ALPHA_HIGH_COST
```

runtime再选择是否加速。

---

# 12. research_spectral：六个算子逐个处理

---

## R22-040 `ts_bicoherence_top_decile_mean`

当前统计量虽比raw max稳，但仍存在 estimator-size bias。

不要删除。

建议：

```text
DIRECT_INTERMEDIATE
```

或：

```text
DIRECT_ALPHA_HIGH_COST
terminal=false by default
```

更推荐作为 intermediate/base statistic。

### 矿工优先使用其 bias-corrected sibling：

```text
ts_bicoherence_top_decile_excess
```

---

## R22-041 `ts_bicoherence_top_decile_excess`

已有 deterministic phase-surrogate null subtraction。

迁移：

```text
→ DIRECT_ALPHA_HIGH_COST
```

### 参数

```text
window: horizon/coarse
n_segments: estimator_resolution/coarse
n_surrogates: fixed or very small coarse grid
```

禁止 miner把 n_surrogates 当 alpha经济参数大范围调优。

---

## R22-042 `ts_kernel_granger_score`

当前不是 training-residual cheating，而是 blocked OOS predictive improvement。

迁移：

```text
→ DIRECT_ALPHA_HIGH_COST
```

### slot role

```text
y = target series
x = predictor series
```

不能当作无序 binary operator。

### grammar

允许：

```text
predictor <- alpha/intermediate field
target <- return/residual return/volatility target
```

不要自动交换 x/y去制造重复。

---

## R22-043 `ts_residualized_hsic`

当前名字已经诚实表达 residualized HSIC，不冒充严格 KCI。

迁移：

```text
→ DIRECT_ALPHA_HIGH_COST
```

### slot

```text
x, y: tested variables
z: conditioning variable
```

z必须是 condition/context continuous panel，不是scalar。

### parameter

`purge_gap` 作为 estimator-control，固定/小grid。

---

## R22-044 `ts_bds_statistic`

迁移：

```text
→ DIRECT_ALPHA_HIGH_COST
```

输出是非线性/序列依赖强度。

`embedding_dim/distance_multiplier` coarse grid。

---

## R22-045 `ts_rolling_sr_gaussian_mean_shift_score`

这是 rolling change-point intensity，不是未来函数。

迁移：

```text
→ DIRECT_ALPHA
```

或成本预算紧时 `DIRECT_ALPHA_HIGH_COST`。

### side

```text
up/down
```

是有限枚举，可搜索。

### output

连续 change evidence，可 terminal。

---

# 13. `sin/cos/asin/acos` 不应简单扔 ResearchTool

当前 `direct_use.py` 把：

```text
sin
cos
asin
acos
```

归 RESEARCH_TOOL。

这个判断过粗。

---

## R22-046
不允许：

```text
sin(arbitrary raw price)
cos(arbitrary balance-sheet amount)
```

这种无语义搜索。

但这不代表这些数学能力不能进入挖掘。

---

## R22-047
方案A（推荐）：typed math canonical

```text
sin_phase(x)
cos_phase(x)
asin_bounded(x)
acos_bounded(x)
```

---

## R22-048
`sin_phase/cos_phase` 输入语义：

```text
phase_radians
cyclic_position
normalized angular state
```

---

## R22-049
`asin/acos`：

```text
input range [-1,1]
```

仅允许：

```text
correlation
bounded ratio
normalized state
```

---

## R22-050
原始 generic：

```text
sin/cos/asin/acos
```

移 internal/compat，不作为矿工 public canonical。

typed variants：

```text
DIRECT_INTERMEDIATE
```

---

## R22-051
如果不想新增名字，也可以给原 canonical增加严格 SemanticType/Domain contract，并设：

```text
DIRECT_INTERMEDIATE
```

但必须防 arbitrary numeric input。

---

# 14. in-sample regression/model diagnostics：不要直接矿旧定义，做 causal rewrite

当前已明确 diagnostic family包括：

```text
ts_multi_regression_coeff
ts_multi_regression_resid
ts_multi_regression_resid_z
ts_multi_regression_r2

ts_huber_regression_coeff
ts_huber_regression_resid
ts_huber_regression_resid_z

ts_ridge_regression_coeff
ts_ridge_regression_resid
ts_ridge_regression_resid_z

ts_ar_forecast
ts_ar_innovation
ts_ar_innovation_z

ts_expectile_regression_coeff
ts_expectile_regression_resid

ts_quantile_regression_coeff
ts_quantile_regression_resid
```

这些当前被视作 in-sample diagnostics。

---

## R22-052
旧 in-sample canonical：

```text
MOVE_RESEARCH_TOOL
```

不进默认 mining。

---

## R22-053
为每个 family 建 causal mineable sibling：

```text
*_prior_coeff
*_forecast
*_forecast_error
*_forecast_error_z
*_prior_r2
```

实际只保留真正有意义的变体，不机械复制所有后缀。

---

## R22-054
`prior` 的严格定义：

t日输出的model参数只允许用：

```text
<= t-1
```

数据拟合。

---

## R22-055
`forecast_error`：

```text
model fit on past
evaluate current observation
```

不能把当前 y同时用于拟合再称“surprise”。

---

## R22-056
这些 causal sibling：

```text
DIRECT_ALPHA
```

模型昂贵的：

```text
DIRECT_ALPHA_HIGH_COST
```

---

# 15. `ts_poly2_coeff / ts_poly2_resid`

当前 direct-use显式判 ResearchTool，production denied也包含。

---

## R22-057
不要强行把旧 in-sample定义变成 direct。

---

## R22-058
新增/确认：

```text
ts_poly2_prior_coeff
ts_poly2_forecast_error
ts_poly2_forecast_error_z
```

---

## R22-059
旧 canonical：

```text
ResearchTool / deprecated alias
```

如果没有任何用户依赖，可删除 public。

---

# 16. Benchmark / market self-inclusion family全部改成明确可挖版本

当前 benchmark-only：

```text
cs_beta_to_market
cs_alpha_to_market
rolling_beta_to_market
```

---

## R22-060
不能因为 self-inclusion bias就永久废掉“beta/alpha to market”能力。

---

## R22-061
优先两条合法路线：

### 路线A
显式外部 benchmark：

```text
beta(ret, benchmark_ret)
alpha(ret, benchmark_ret)
```

### 路线B
cross-section内部benchmark：

```text
market_ex_self
```

---

## R22-062
建立/确认：

```text
cs_beta_to_market_ex_self
cs_alpha_to_market_ex_self
rolling_beta_to_benchmark
```

---

## R22-063
旧 ambiguous名字：

```text
legacy / ResearchTool
```

或者语义固定为 ex-self并做version migration，不能静默改定义。

---

# 17. 当前 production denied 与 DirectUse 存在明确冲突

当前最新代码中 `PRODUCTION_DENIED_CANONICALS` 仍含一批其实已经被 DirectUse判为有价值的 operator。

必须机器审计：

```text
DIRECT_* ∩ PRODUCTION_DENIED
```

---

## R22-064
当前已确认重点冲突包括：

```text
tail_beta
residual_momentum_capm
coskewness_to_market
idio_vol
idio_skew
rank_corr
intraday_vwap_deviation
digital_count
```

另有其它项必须动态枚举。

---

## R22-065
对于：

```text
tail_beta
residual_momentum_capm
coskewness_to_market
idio_vol
idio_skew
```

`direct_use.py` 已明确把它们视为 benchmark-required alpha。

只要 R17/R19 的：

```text
benchmark provider
PIT
math
parameter
backend evidence
```

完成，

就从静态 production deny移除。

---

## R22-066
不要再用永久手写 deny list维护“暂时没有 evidence”的 operator。

拆成：

```text
PERMANENTLY_FORBIDDEN
ADMISSION_PENDING
CONTEXT_BLOCKED
```

---

# 18. `rank_corr`

## R22-067
当前 direct-use认为它是合法 cross-sectional dependency alpha。

R19已指出数学/时间语义要整改。

### 本轮原则

R19修好后：

```text
rank_corr -> DIRECT_ALPHA
```

并从 `PRODUCTION_DENIED` 删除。

不要因为旧bug永久封禁这项能力。

---

# 19. `digital_count`

## R22-068
它更像：

```text
DIRECT_STATE / DIRECT_CONDITION statistic
```

而不是普通terminal alpha。

---

## R22-069
从 production deny移除后：

```text
mining_visible=True
terminal_usable=False
```

放进 state/gate grammar。

---

# 20. `intraday_vwap_deviation`

当前存在明显真值冲突：

```text
direct_use.py:
A-share has certified minute source -> direct

production_hardening.py:
SOURCE_BLOCKED_CANONICALS contains intraday_vwap_deviation

PRODUCTION_DENIED also contains it
```

---

## R22-070
把 source eligibility 改成 contextual：

```text
A-share minute source available -> DIRECT_ALPHA
US minute source unavailable -> context blocked
```

---

## R22-071
禁止全局：

```text
SOURCE_BLOCKED_CANONICALS
```

把一个“某市场不可用”的 operator变成“两市场都不可用”。

---

## R22-072
ContextualDirectUse：

```text
canonical
market
frequency
available_sources
session_calendar
```

共同决定 admission。

---

# 21. `fin_total_operating_accruals`

当前确认缺可靠 depreciation/amortization source。

---

## R22-073
不要为了“全部能用”虚构字段。

---

## R22-074
旧 canonical：

```text
DELETE_NO_DATA
```

直到真实source存在。

---

## R22-075
但是要检查当前字段能否支持另一个**定义不同且诚实命名**的 accrual factor，例如：

```text
cashflow_accrual = net_income - operating_cash_flow
```

仅在字段真实存在时新增。

---

## R22-076
不能拿近似字段冒充 `total_operating_accruals` 原定义。

---

# 22. Shareholder Research 两个旧算子

当前：

```text
holder_concentration_change
holder_count_change_rate
```

在 research surface。

而实现本身直接 raise ValueError。

---

## R22-077
**public operator不能保留一个“调用就报错”的 canonical。**

---

## R22-078 `holder_concentration_change`

当前错误说明：

```text
必须基于 distinct relation snapshots
```

如果现有 relation source已经可提供：

```text
snapshot concentration
previous snapshot concentration
```

则实现新的：

```text
holder_concentration_snapshot_change
```

并通过 PIT projection到日频。

新算子：

```text
DIRECT_ALPHA / FUNDAMENTAL_PIT lane
```

旧名字：

```text
deprecated alias/migration
```

---

## R22-079 `holder_count_change_rate`

当前 TopTen rows不是总股东户数。

若现有数据没有真实 total shareholder count：

```text
DELETE_NO_DATA
```

不要拿 Top10 entity count伪造。

---

# 23. `micro_bvc_vpin`

当前代码明确：

```text
minute bars are not ticks
BV-C estimation error
VPIN literature contested
research-only
```

---

## R22-080
不强行promotion。

这是允许保留 ResearchTool 的少数例子之一。

---

## R22-081
但 public mining surface不要保留它。

可用替代能力：

```text
intraday_bvc_imbalance
intraday_impact_beta
intraday_impact_asymmetry
intraday_return_wasserstein_shift
```

应全部直接可挖。

---

## R22-082
未来若接入真正 tick/trade classification source，再重新审 VPIN，不提前放行。

---

# 24. Raw matrix/math primitives：不需要“作为factor terminal”，但能力不能浪费

例如：

```text
fft
ifft
wavelet
convolve
correlate
mat_inverse
eig
svd
pca
norm*
```

---

## R22-083
这些 generic raw array primitives：

```text
MOVE_INTERNAL
```

不是 public factor canonical。

---

## R22-084
但对应 factor-shaped typed wrappers：

```text
ts_spectral_*
ts_wavelet_*
ts_hankel_*
ts_ssa_*
ts_dmd_*
cs_pca_like / factor model wrapper
```

只要安全，应 direct-use。

---

## R22-085
最终审计按**能力覆盖**而不是“raw helper也一定terminal”判断。

---

# 25. Random/future/noncausal 永久不救

以下类不允许为了提高 direct coverage而放行：

```text
Lead
next
bfill
causal_bfill  # 如果语义本质仍利用未来
fillna_interpolate
shuffle
sample
rand_*
```

---

## R22-086
从 public Factor registry移除。

---

## R22-087
如测试需要，放：

```text
research utility / test helper
```

不要给 DSL/miner。

---

# 26. Price-level intermediate必须被矿工“真正用上”

当前 DirectUse已经把大量：

```text
rolling_vwap
ATR_WILDER
donchian_*
Keltner*
ichimoku*
KAMA/DEMA/TEMA
PSAR
pivot/support/resistance
candle body/range/gap
...
```

判为 DIRECT_INTERMEDIATE。

但仅分类不够。

---

## R22-088
矿工grammar必须允许：

```text
alpha child <- DIRECT_INTERMEDIATE
```

---

## R22-089
每个 level intermediate至少提供一组 legal composition templates：

```text
distance
ratio
normalized deviation
slope
break strength
relative rank
```

---

## R22-090
intermediate不能因为 `terminal_usable=False` 被整个矿工过滤掉。

---

# 27. State / Condition / Event 同样必须实际进入搜索

## R22-091
这些不是“不能用”。

它们用于：

```text
where(condition, alpha1, alpha2)
alpha * state_strength
event-conditioned ts_mean
trade_when
regime interaction
```

---

## R22-092
Grammar必须能生成：

```text
DIRECT_CONDITION -> condition slot
DIRECT_EVENT -> condition/mask slot
DIRECT_STATE -> gate/interaction
DIRECT_GROUP_STATE -> group interaction
DIRECT_GLOBAL_STATE -> market regime interaction
```

---

## R22-093
最终 direct coverage统计不能只数terminal alpha。

同时报告：

```text
terminal operators
composition-only operators
gating/context operators
source transforms
```

---

# 28. SourceTransform 不消耗 alpha tree depth

## R22-094
fill/impute/source-side transforms能参与构造数据输入。

---

## R22-095
但是：

```text
source transform node
```

不要占普通 alpha complexity depth，否则矿工会因为预处理步骤变“更复杂”而不公平惩罚。

---

# 29. ControlFlow 必须是一等 grammar 节点

## R22-096
`where/trade_when/gate` 等不要当普通numeric operator。

---

## R22-097
定义：

```text
condition
true_branch
false_branch
state/event trigger
```

typed slots。

---

# 30. High-cost != 不可挖

## R22-098
以下类：

```text
DMD
HSIC
kernel Granger
topology
matrix profile
spectral
multifractal
Markov/KM
recurrence
advanced tail
```

只要因果和数值正确：

```text
DIRECT_ALPHA_HIGH_COST
```

---

## R22-099
通过：

```text
family quota
max_cost
low prior
coarse estimator grids
```

控制搜索成本。

不要直接 Research ban。

---

# 31. Full-history operator也应可挖

## R22-100
如果：

```text
因子有价值
数学正确
PIT正确
只是无checkpoint
```

则：

```text
DIRECT_ALPHA_HIGH_COST
execution_model=full_history
```

---

## R22-101
不要把“无法增量”误等于“无法挖”。

---

## R22-102
miner可以在小样本/预筛阶段评估 high-cost，再对入选factor做完整验证。

---

# 32. `experimental` 不应该自动改变 terminal legality

当前 direct_use default逻辑会因为：

```text
lifecycle_status == experimental
or not production_certified
```

把 DIRECT_ALPHA 的 `terminal_allowed=False`。

这是职责混淆。

---

## R22-103
语义上一个factor是否能terminal：

由：

```text
DirectUseRole
```

决定。

---

## R22-104
是否“现在能进入production mining”：

由：

```text
AdmissionState / Certification
```

决定。

---

## R22-105
因此：

```text
DIRECT_ALPHA + PENDING_CERTIFICATION
terminal_semantically_legal=True
production_admitted=False
```

而不是改 terminal role。

---

# 33. 引入 `RemediationPath`

## R22-106
所有非eligible operator必须有：

```python
RemediationPath
```

枚举：

```text
CERTIFY_AS_IS
PROMOTE_HIGH_COST
MOVE_TO_INTERMEDIATE
MOVE_TO_STATE
MOVE_TO_CONDITION
MOVE_TO_EVENT
MOVE_TO_CONTEXT
ADD_TYPED_VARIANT
ADD_CAUSAL_PRIOR_VARIANT
ADD_EX_SELF_VARIANT
ADD_EXPLICIT_BENCHMARK_INPUT
ADD_SOURCE_RECIPE
ADD_GRAIN_PROJECTION
ADD_CHECKPOINT
FIX_MATH
FIX_PARAM_SPACE
FIX_OUTPUT_ROLE
REPLACE_DUPLICATE
MOVE_RESEARCH_TOOL
MOVE_INTERNAL
DELETE_NO_DATA
DELETE_NONCAUSAL
DELETE_OBSOLETE
```

---

# 34. 非eligible不能只写 blocker，要写“怎么变eligible”

## R22-107
每个 canonical输出：

```text
current_status
current_role
current_surface
current_blockers
target_status
target_role
remediation_path
files_to_change
tests_to_add
expected_mining_lane
```

---

# 35. 建 `OperatorPromotionMatrix`

## R22-108
机器产物字段：

```text
canonical
current_surface
current_direct_status
current_mining_role
current_admission
production_denied
source_blocked
diagnostic
benchmark_only
hidden_from_default_mining
target_disposition
target_direct_status
target_ast_positions
target_terminal
market_contexts
required_source_recipe
required_code_change
replacement
delete_reason
```

---

# 36. 先动态枚举所有 Research

## R22-109
不要只读静态：

```text
RESEARCH_ONLY_CANONICALS
```

因为 modules会动态：

```text
extend_research_only
union_research
```

---

## R22-110
必须：

```python
load_all()
surface == research
```

之后得到真实全量。

---

# 37. 再动态枚举“虽然不是 Research，但矿工仍用不了”的

## R22-111
集合：

```text
all_registered
- direct_mining_visible
- delete/internal/research-tool justified
```

必须为空。

---

## R22-112
特别扫描：

```text
hidden_from_default_mining
diagnostic_only
benchmark_only
compatibility_only
production_denied
source_blocked
experimental
uncertified
full_history
high_cost
no_cost_contract
no_source_recipe
no_smoke_recipe
no_input_semantics
```

---

# 38. `hidden_from_default_mining` 不允许长期作为黑箱

## R22-113
每个 hidden operator必须归因：

```text
pure diagnostic
temporary certification pending
too expensive
semantic ambiguity
no data
```

---

## R22-114
如果只是：

```text
too expensive
```

改 HighCost。

---

## R22-115
如果只是：

```text
experimental
```

改 pending certification，不隐藏语义角色。

---

# 39. Production denied列表必须缩成真正禁止项

## R22-116
最终 `PRODUCTION_DENIED_CANONICALS` 不应混：

```text
永久禁止
暂未认证
source context missing
benchmark source missing
high cost
```

---

## R22-117
保留永久/明确语义不适用。

其余变 AdmissionState。

---

# 40. Source blocked同样必须 contextual

## R22-118
一个operator只要至少一个 market/source context可执行：

```text
不能全局 DELETE_NO_DATA
```

---

## R22-119
例如：

```text
A可用、US不可用
```

应该：

```text
market_support = ashare
```

不是全局blocked。

---

# 41. DefaultInputRecipe 覆盖必须从少数扩到100%

当前 `_DEFAULT_INPUT_RECIPE` 只有很少一批例子。

---

## R22-120
所有 retained direct operator至少一个真实 input recipe。

---

## R22-121
recipe不是硬编码physical column，而是 canonical concepts：

```text
continuous_close
return_decimal
volume
turnover
benchmark_return
industry_group
...
```

由R17 resolver映射市场实际source。

---

# 42. Multi-input advanced operator必须有slot semantics

## R22-122
不能再靠：

```text
x/y/z/f1/f2/f3
```

推断含义。

---

## R22-123
至少声明：

```text
predictor
target
conditioner
benchmark
scale
group
event
price
return
volume
```

---

# 43. Output semantic必须完整

## R22-124
当前 `DirectUseOperator.output_value_domain` 还是空字符串。

---

## R22-125
补：

```text
continuous_signed
continuous_nonnegative
bounded_0_1
boolean
ternary_event
category
price_level
return
ratio
zscore
rank
count
duration
frequency
growth_rate
```

---

# 44. Output domain决定合法 parent

## R22-126
例：

```text
boolean -> where/if/count_if
category -> interaction/group, 不直接 arithmetic rank id
price_level -> normalize/distance/delta
bounded ratio -> rank/zscore/logit(optional)
count/duration -> normalize/rank
```

---

# 45. Search parameter不是越多越好

## R22-127
对 research转direct的 advanced operators：

```text
estimator resolution
numerical knob
regularization
surrogate count
embedding dimension
```

默认：

```text
fixed/coarse
```

---

## R22-128
真正 horizon/threshold/economic parameter才 full search。

---

# 46. 参数可辨识性

## R22-129
对每个 promoted operator跑 parameter-injectivity。

---

## R22-130
如果两个合法参数区域输出总相同：

```text
删除死参数
收窄choices
改 relational spec
```

---

# 47. Advanced operator non-degeneracy

## R22-131
不能只“执行不报错”。

每个 promotion需要：

```text
finite coverage
cross-sectional variation
time variation
expected output domain
parameter sensitivity
```

---

# 48. Direct promotion的最低证据

## R22-132
`CERTIFY_AS_IS` 至少：

```text
math golden
prefix invariance
missing topology
default execution
param boundary
source recipe
one real/synthetic market smoke
backend reference evidence
output role evidence
```

---

# 49. Single backend可以合格

## R22-133
不要再要求每个direct operator都有：

```text
Pandas + Polars + SQL
```

才可挖。

---

## R22-134
如果只有：

```text
pandas_numpy
```

但：

```text
正确
稳定
成本可接受
有evidence
```

可以：

```text
production_admitted=True
preferred_backend=pandas_numpy
```

---

# 50. Backend performance只影响cost lane

## R22-135
慢backend：

```text
提高cost
降低search prior
```

而不是 semantic deny。

---

# 51. Research operator promotion表：当前明确建议

按最新可见代码，先按下面迁移；整改AI还要动态扩充。

| Canonical/family | 当前 | 目标 |
|---|---|---|
| ts_wavelet_lowpass_reconstruct | Research | DIRECT_INTERMEDIATE |
| ts_signature_mahalanobis_anomaly | Research | DIRECT_ALPHA_HIGH_COST |
| ts_persistence_birth_dispersion | Research | DIRECT_ALPHA_HIGH_COST |
| ts_dmd_level_* | Research | DIRECT_ALPHA_HIGH_COST |
| ts_dmd_return_* | Research | DIRECT_ALPHA_HIGH_COST |
| generic ts_dmd_* | Research | Legacy/Internal alias to typed variants |
| ts_bicoherence_top_decile_mean | Research | DIRECT_INTERMEDIATE / low-prior high-cost |
| ts_bicoherence_top_decile_excess | Research | DIRECT_ALPHA_HIGH_COST |
| ts_kernel_granger_score | Research | DIRECT_ALPHA_HIGH_COST |
| ts_residualized_hsic | Research | DIRECT_ALPHA_HIGH_COST |
| ts_bds_statistic | Research | DIRECT_ALPHA_HIGH_COST |
| ts_rolling_sr_gaussian_mean_shift_score | Research | DIRECT_ALPHA |
| sin/cos | ResearchTool verdict | typed DIRECT_INTERMEDIATE |
| asin/acos | ResearchTool verdict | bounded-input DIRECT_INTERMEDIATE |
| ts_poly2_coeff/resid | diagnostic | old stays ResearchTool; causal siblings DIRECT |
| in-sample regression diagnostics | diagnostic | causal prior/forecast siblings DIRECT |
| benchmark self-inclusion forms | diagnostic | ex-self / explicit benchmark siblings DIRECT |
| holder_concentration_change | deprecated/raises | snapshot-aware replacement DIRECT |
| holder_count_change_rate | deprecated/raises | DELETE_NO_DATA unless real total-holder field exists |
| micro_bvc_vpin | research only | keep ResearchTool; direct alternatives used |
| fin_total_operating_accruals | no data | DELETE_NO_DATA / honest alternative factor only |
| random/future ops | denied | DELETE/MOVE TEST TOOL |
| raw matrix primitives | non-factor | MOVE_INTERNAL; factor-shaped wrappers DIRECT |

---

# 52. Direct-use promotion不等于默认高权重搜索

## R22-136
每个 direct operator增加：

```text
search_prior
family_budget
cost_budget
```

---

## R22-137
例如：

```text
basic ts: prior 1.0
technical: 0.8
advanced nonlinear: 0.2
DMD/topology: 0.1
```

具体数值可配置，不要硬编码经济结论。

---

# 53. Search diversity quota

## R22-138
避免数百高级算子淹没基础算子。

按：

```text
economic_effect_family
semantic_redundancy_group
cost_lane
source_domain
```

设quota。

---

# 54. Research promotion后cold-start也要能用

## R22-139
所有新direct operator至少：

```text
一条 cold-start exemplar
或一条 grammar generation recipe
```

---

## R22-140
不要人工给每个参数组合造factor，只给不同语义结构。

---

# 55. Direct operator必须在实际矿工入口可达

## R22-141
做 reachability test：

```python
for op in direct_visible:
    assert grammar_can_generate_expression_containing(op)
```

---

## R22-142
State/Event/Intermediate也做相应slot reachability，不只是terminal。

---

# 56. 不能只验证“allowlist含名字”

## R22-143
必须真实：

```text
generate legal AST
serialize DSL
parse
analyze
compile
execute
```

---

# 57. 迁移后的 Research surface

## R22-144
最终 Factor DSL：

```text
factor-shaped research canonical = 0
```

---

## R22-145
ResearchToolRegistry可以保留：

```text
diagnostics
methodology research
non-factor analytics
```

但自动因子矿工不读。

---

# 58. ResearchTool如果存在 causal sibling，必须记录replacement

## R22-146
例如：

```text
ts_huber_regression_resid
-> ts_huber_forecast_error
```

---

## R22-147
Research manifest字段：

```text
reason_not_mineable
recommended_factor_replacement
```

---

# 59. 删除 public Research“永远等待以后再改”的状态

## R22-148
任何 research factor：

```text
要么本轮promotion
要么给明确 causal replacement
要么move research tool
要么delete
```

没有 pending forever。

---

# 60. 全量 promote-or-delete 逻辑升级

## R22-149

对每个 canonical：

```python
if permanently_noncausal_or_random:
    delete_public()
elif exact_duplicate:
    migrate_to_replacement()
elif no_data_all_supported_markets:
    delete_no_data()
elif raw_internal_capability:
    move_internal()
elif pure_in_sample_diagnostic:
    create_or_confirm_causal_sibling()
    move_research_tool(old)
elif semantic_input_ambiguous:
    split_typed_variants()
elif factor_shaped_causal:
    assign_direct_role()
    certify_or_pending_admission()
elif useful_nonterminal:
    assign_state_event_condition_intermediate_lane()
else:
    fail_audit()
```

---

# 61. 一个重要验收原则：不要用“目前未认证”当删除理由

## R22-150
`production_certified=False`：

只说明：

```text
还没通过evidence
```

不说明：

```text
这个operator不值得挖
```

---

# 62. 也不要用“暂时backend慢”当Research理由

## R22-151
慢：

```text
high_cost
```

不是：

```text
research
```

---

# 63. 也不要用“参数复杂”当Research理由

## R22-152
参数复杂：

```text
fixed/coarse/relational constrained
```

---

# 64. 也不要用“高级统计”当Research理由

## R22-153
高级统计是否mineable由：

```text
causality
sample contract
numerical stability
computational budget
```

决定。

不是名字高级就ban。

---

# 65. 只有纯diagnostic才留ResearchTool

## R22-154
典型：

```text
same-window fit coefficient for diagnosis
in-sample residual quality
benchmark report metric
debug-only internals
```

---

# 66. “算子全能用”最终指标

## R22-155
最终报告不要只给：

```text
direct alpha count
```

而要给：

```text
registered canonical count
public retained count
mining_visible count
terminal_usable count
composition_only count
state/condition/event count
context count
source-transform count
high-cost count
research-tool count
internal count
deleted count
```

---

## R22-156
关键比率：

```text
Mining Coverage
= mining_visible / public_retained
```

必须：

```text
100%
```

---

# 67. ResearchTool不算 public_retained

## R22-157
如果一个 operator被决定为纯ResearchTool：

从：

```text
public Factor DSL registry
```

迁出。

这样不会用“ResearchTool太多”拖低FactorEngine direct coverage。

---

# 68. Replacement coverage

## R22-158
所有被移ResearchTool/deleted但数学能力有价值的：

```text
必须有 replacement
```

或明确：

```text
NO_SAFE_REPLACEMENT
```

---

# 69. Current static surface cleanup

## R22-159
最终：

```text
RESEARCH_ONLY_CANONICALS
```

不再承载 factor-shaped operator。

可为空，或只保留compat transitional alias但不进Factor DSL。

---

# 70. Production deny cleanup

## R22-160
最终机器生成：

```text
DIRECT_ROLE_PRODUCTION_DENY_CONFLICT.csv
```

必须为空。

---

# 71. Source blocked cleanup

## R22-161
生成：

```text
OPERATOR_CONTEXT_AVAILABILITY.csv
```

每个 direct op至少一个：

```text
market + source + frequency
```

context可用。

---

# 72. No-context direct operator

## R22-162
如果：

```text
0 supported context
```

不得标Direct。

只能：

```text
DELETE_NO_DATA
ResearchTool
future backlog outside public registry
```

---

# 73. Smoke recipe 100%

## R22-163
每个 mining-visible operator至少一条合法 smoke recipe。

---

## R22-164
State/Event等smoke要验证role，不要求像alpha一样非constant。

---

# 74. Advanced promoted operators的真实数据smoke

## R22-165
DMD/signature/topology/kernel Granger等不能只synthetic。

至少抽一小段真实可用市场数据：

```text
compile
execute
coverage
domain
runtime
```

---

# 75. Promotion artifacts

## R22-166
生成：

```text
factor_engine/docs/R22_OPERATOR_PROMOTION_MATRIX.csv
factor_engine/docs/R22_OPERATOR_PROMOTION_MATRIX.json
factor_engine/docs/R22_OPERATOR_PROMOTION_MATRIX.md
```

---

## R22-167
生成：

```text
factor_engine/docs/R22_RESEARCH_MIGRATION_PLAN.md
factor_engine/docs/R22_RESEARCH_MIGRATION_PLAN.json
```

---

## R22-168
生成：

```text
factor_engine/docs/R22_CAUSAL_REPLACEMENT_MAP.json
factor_engine/docs/R22_CAUSAL_REPLACEMENT_MAP.md
```

---

## R22-169
生成：

```text
factor_engine/docs/R22_DIRECT_GRAMMAR_REACHABILITY.csv
factor_engine/docs/R22_DIRECT_GRAMMAR_REACHABILITY.json
```

---

## R22-170
生成：

```text
factor_engine/docs/R22_CONTEXT_AVAILABILITY_MATRIX.csv
```

---

## R22-171
生成：

```text
factor_engine/build/mining/direct_operator_catalog_v2.json
factor_engine/build/mining/direct_search_space_v2.json
factor_engine/build/mining/high_cost_lane.json
factor_engine/build/mining/research_tools_only.json
```

---

# 76. 硬性 CI gate

## R22-172
以下全部必须0：

```text
PUBLIC_RETAINED_NOT_MINING_VISIBLE
FACTOR_SHAPED_RESEARCH_OPERATOR
DIRECT_WITHOUT_ROLE
DIRECT_WITHOUT_AST_POSITION
DIRECT_WITHOUT_INPUT_SLOT_TYPES
DIRECT_WITHOUT_OUTPUT_DOMAIN
DIRECT_WITHOUT_SOURCE_CONTEXT
DIRECT_WITHOUT_SMOKE_RECIPE
DIRECT_NOT_GRAMMAR_REACHABLE
DIRECT_ROLE_PRODUCTION_DENY_CONFLICT
DIRECT_ROLE_HIDDEN_FROM_MINING
PUBLIC_OPERATOR_ALWAYS_RAISES
PUBLIC_DIAGNOSTIC_IN_SAMPLE
PUBLIC_BENCHMARK_SELF_INCLUDED
PUBLIC_NO_DATA
PUBLIC_NONCAUSAL
PUBLIC_RANDOM
PUBLIC_EXACT_DUPLICATE
PUBLIC_LEGACY_GHOST
CAUSAL_REPLACEMENT_MISSING
```

---

# 77. 对 current mining integration 的明确修改

## R22-173
`default_mining_operator_allowlist()` 不再从 backend allowlist决定。

---

## R22-174
`default_typed_mining_search_space_config()`直接遍历：

```text
DirectUse catalog
```

---

## R22-175
signature中的：

```text
inputs
```

等于：

```text
data_inputs
```

---

## R22-176
不再：

```text
except Exception: pass
```

role resolution。

---

## R22-177
terminal只读DirectUse正向authority。

---

## R22-178
输出完整：

```text
status
lane
roles
input slots
output domain
market/source contexts
cost
search grades
default recipe
```

---

# 78. 推荐新的 mining signature

## R22-179

```json
{
  "canonical": "ts_kernel_granger_score",
  "direct_use_status": "direct_alpha_high_cost",
  "mining_visible": true,
  "composition_usable": true,
  "terminal_usable": true,
  "lane": "alpha_high_cost",
  "data_inputs": [
    {"name": "y", "semantic_role": "target", "unit": ["return", "ratio", "normalized"]},
    {"name": "x", "semantic_role": "predictor", "unit": ["return", "ratio", "normalized"]}
  ],
  "scalar_parameters": [
    {"name": "window", "search_grade": "coarse"},
    {"name": "lag", "search_grade": "coarse"}
  ],
  "output_semantic": "predictive_improvement_score",
  "output_domain": "continuous_signed",
  "cost": 9,
  "terminal_allowed": true
}
```

---

# 79. 因子生成器必须理解非terminal节点

## R22-180
AlphaProbe等不要只得到一张“函数名list”。

必须得到typed grammar。

---

## R22-181
表达式生成流程：

```text
choose terminal role
→ recursively choose compatible children
→ choose field/source-compatible inputs
→ choose legal params
→ cost check
→ compile-time semantic type check
→ execute
```

---

# 80. promotion不是让搜索空间失控

## R22-182
Research转direct后，用：

```text
lane
prior
quota
cost
param grid
```

控制规模。

不要通过永久ban控制。

---

# 81. Final promotion decisions 必须一次性完成

## R22-183
对动态枚举出来的每一个：

```text
surface=research
role=research
diagnostic
hidden
production denied
source blocked
uncertified direct
```

逐个给最终 verdict。

---

## R22-184
不能只做我上面点名的20多个。

---

# 82. 对“真正不能用”的定义

## R22-185
最终只允许四类真正不进入Factor mining：

### A. 非因果/随机
永久删除public。

### B. 纯diagnostic
移ResearchTool；有causal sibling。

### C. 无数据
删除public；不能编造字段。

### D. internal raw capability
移internal；factor-shaped wrapper direct。

---

# 83. 任何其它 retained operator都应 mining-visible

## R22-186
如果审计AI认为一个 operator既：

```text
有数学意义
有合理factor composition用途
因果
数据可得
```

但仍决定：

```text
not mining-visible
```

必须视为审计失败，除非给出明确技术证明。

---

# 84. 最终执行顺序

1. 读取服务器真实 HEAD / dirty fingerprint；
2. `load_all()`；
3. 动态导出所有 canonical；
4. 导出当前 surface/direct role/admission/production deny/source blocked；
5. 建 current non-mining-visible集合；
6. 逐个归因；
7. 先修 `mining_visible/composition_usable/terminal_usable` 模型；
8. 让 mining integration只读DirectUse；
9. 修typed input schema和terminal authority；
10. 清理 Research authoring与DirectUse role耦合；
11. promote research_transform；
12. promote typed DMD；
13. promote research_spectral；
14. typed化sin/cos/asin/acos；
15. causal rewrite regression/poly2 diagnostics；
16. benchmark explicit/ex-self化；
17. 修 production-denied/direct冲突；
18. 修 contextual source availability；
19. 处理 holder/no-data；
20. 保留真正 ResearchTool/internal/delete项；
21. 给全部 retained op default input recipe；
22. 给全部 retained op output semantic/domain；
23. 给全部 direct op smoke；
24. grammar reachability全量测试；
25. real-data spot smoke；
26. 重生 DirectUse/mining manifests；
27. cold-start改为DirectUse grammar；
28. 全量测试；
29. 再次动态枚举；
30. `Mining Coverage == 100%` 后才结束。

---

# 85. 最终汇报格式

最终只汇报真实数字：

```text
Current canonical count:
Public retained:
Mining-visible:
Terminal-usable:
Composition-only:
State/Condition/Event:
Context:
Source transforms:
High-cost:
Moved ResearchTool:
Moved Internal:
Deleted duplicate:
Deleted noncausal/random:
Deleted no-data:
Replaced by causal sibling:
Research factor-shaped remaining:
Direct/production-denied conflicts:
Grammar unreachable:
Smoke failed:
Mining Coverage:
```

最后一行：

```text
ALL_RETAINED_OPERATORS_MINING_USABLE=true/false
```

只有：

```text
Mining Coverage = 100%
Research factor-shaped remaining = 0
Direct/deny conflicts = 0
Grammar unreachable = 0
```

才能写 `true`。

---

# 86. 核心原则

不要把“所有算子都能用”错误理解成：

```text
所有算子都能裸着当最终factor
```

真正正确的目标是：

```text
所有有价值的能力都能被矿工合法访问。
```

最终应该形成：

```text
Alpha：直接挖
HighCost：受预算直接挖
State/Event/Condition：作为条件/状态直接挖
Intermediate：作为组合节点直接挖
SourceTransform：作为数据处理节点直接挖
ControlFlow：作为规则节点直接挖

ResearchTool：只剩纯诊断
Internal：只剩底层能力
Delete：只剩真正错误/重复/无数据/非因果
```

**不要再让“Research”成为“以后再说”的状态。能因果化就因果化，能typed就typed，能进high-cost lane就进high-cost lane，能作为intermediate/state/event就放到正确grammar位置。最终 public FactorEngine 里留下来的每个 canonical，都必须对自动挖掘算法有明确、真实、可执行的用途。**
