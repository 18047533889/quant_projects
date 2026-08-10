# FactorEngine R18：全算子“直接可用化”与删冗独立审计整改提示词

> **用途**：把本文件单独交给代码整改 AI，让它直接读取当前服务器工作区中的 `factor_engine` 并执行整改。  
> **文档关系**：这是一个**全新的、独立的 R18 文档**，严禁与 R17、R16 或任何更早的整改文档合并。  
> **前提假设**：R17-001～R17-090 以及此前我已经给出的整改项，都视为已经由其它并行 AI 处理完成。本轮只处理**此前没有完整解决的“每一个算子能不能真正直接拿去挖因子”问题**。  
> **审计基线**：当前 GitHub `main` 可见最新基线 `5dc49e2af6aaa8ab2bc14fe5968049cb074e7df9`（2026-08-09）。服务器本地若已比 GitHub 更新，以你实际读取到的服务器 HEAD/工作区为准，并在报告中写明真实 commit/dirty fingerprint。  
> **执行位置**：直接修改服务器/本地工作区代码。**不要把提交 GitHub、开 PR、合并 GitHub 作为本轮任务。**  
> **执行方式**：不要只写分析报告，不要做完几十个算子就停。必须动态枚举当前所有 canonical operators，一个一个审，直到每个 retained canonical 都有明确的直接使用路径；真正无意义、无数据、重复、危险的 canonical 应删除/移出公共挖掘层。

---

# 0. 本轮的目标与 R17 完全不同

R17 主要解决的是 A 股/美股数据契约、field/provider/PIT/coverage/session/market semantics、typed IR、DataAccess/FactorEngine contract drift 等。本轮不重复。

R18 要解决的是更靠近 AlphaProbe / AlphaMiner / EvoAlpha / 冷启动库实际使用的一层：

> **FactorEngine 里留下来的每一个 public canonical，都必须有理由存在，而且矿工必须知道它该怎么用。**

不能再把“注册了/有实现/在 daily surface/production_certified”直接等同于“适合自动挖因子”。

本轮结束后应满足：

```text
所有 retained canonical
=
有明确数学意义
∧ 有明确挖掘角色
∧ 有明确输入语义
∧ 有合法参数域
∧ 有真实 source recipe
∧ 能 compile
∧ 能 execute
∧ 能产生符合角色预期的非退化输出
∧ 能被正确 mining grammar 调用
∧ 有明确 terminal/non-terminal 规则
∧ 有成本/资源契约
∧ 有逐算子 smoke evidence
```

---

# 1. “直接可用”重新定义

新建机器可读 `DirectUseStatus`：

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

RESEARCH_TOOL

DELETE_DUPLICATE
DELETE_NO_DATA
DELETE_NONCAUSAL
DELETE_MATH_DEFECT
DELETE_USELESS
DELETE_OBSOLETE
```

最终 public factor/mining 层不允许 `UNRESOLVED / UNKNOWN / PENDING_FOREVER / MAYBE_ALPHA / TEMPORARILY_DAILY`。

不是 terminal alpha 不等于没用：状态、条件、event、level、source transform 都可保留，但必须处于正确 AST 位置。

真正删除只针对：
- future/random/non-causal；
- exact semantic duplicate；
- A/US 当前都无真实 source 且无明确接入计划；
- 纯诊断工具（移 ResearchToolRegistry）；
- 没有合理 factor composition 用途的任意数学噪声；
- 数学定义错误/输出长期结构退化且无法修复。

不得因为“单因子 IC 弱”删除有独特组合价值的 primitive。

---

# 2. 当前最新代码新发现：真实 mining 入口仍没有收口

## R18-001：`mining/operator_catalog.py` 号称唯一准入层，但 typed mining 入口仍绕过它

`factor_engine/api/mining_integration.py` 中真实的：
- `default_mining_operator_allowlist`
- `default_mining_search_space_config`
- `default_typed_mining_search_space_config`
- `validate_formula_in_mining_allowlist`

仍主要读取旧 `backend.fastpath_allowlists`，没有真正以 `get_mining_operators()` / direct-use catalog 为唯一 authority。

**整改：**
自动挖掘只允许走：

```python
get_direct_use_mining_operators(context)
```

旧 allowlist 只能表示 backend execution capability，不能决定矿工能看什么。

## R18-002：typed search-space 把所有 params 又放回 `inputs`

当前虽然算出：

```python
scalar_params, panel_inputs = split_scalar_panel_params(...)
```

但又写：

```python
"inputs": params,
"panel_inputs": panel_inputs,
"scalar_parameters": scalar_params,
```

整改成：

```text
data_inputs
scalar_parameters
context_inputs
group_inputs
event_inputs
```

`inputs` 若保留，只能等于 data inputs，绝不能包含 window/lag/threshold 等 scalar knob。

## R18-003：typed mining integration 又重新使用负面排除式 terminal 判定

当前类似：

```python
terminal_allowed = role.value not in {
  "state","condition","event","group_state","global_state"
}
```

会让 INTERNAL/DIAGNOSTIC/RESEARCH/LEGACY/DENIED/UNRESOLVED/SOURCE_TRANSFORM 在误入旧 allowlist 时重新成为 terminal。

整改：全系统只允许一个正向 authority：

```python
terminal_allowed = role in TERMINAL_ROLES
```

## R18-004：mining integration 对 role resolution `except Exception: pass`

production/direct mining 中 metadata/role resolution 失败必须 hard fail，不得继续输出一个缺 role 的 operator signature。

## R18-005：standard mining manifest 默认不应是 `pending`

将：

```text
direct_mining_manifest.json
remediation_backlog_manifest.json
research_operator_manifest.json
```

彻底分开。AlphaProbe/AlphaMiner 默认只读 direct manifest。

## R18-006：cold-start validation 当前主要证明“在 manifest 里”，不是“真的能运行”

升级为：

```text
resolve canonical
→ bind legal inputs
→ build legal params
→ compile
→ execute
→ validate output role
```

每个 retained operator 至少一条真实 smoke recipe。

---

# 3. role classifier 仍把 daily/extended surface 错等于 ALPHA

## R18-007：explicit role authority 不完整

当前 role resolver 对 explicit role 的完整覆盖不足。新增唯一 `direct_use_role` 或等价的显式 metadata。所有 retained canonical 必须显式声明，最终 role 不依赖名字/prefix/surface猜测。

## R18-008：彻底废止 `surface == daily/extended -> ALPHA`

四维完全正交：

```text
AuthoringTier
DirectUseRole
ProductionCertification
BackendCapability
```

Authoring surface 只说明可引用，不说明能否 terminal。

---

# 4. 必须重新分类的具体 operator 族

## R18-009：布尔/比较/missingness 不能作为普通 ALPHA terminal

逐个检查：

```text
and_ or_ not_
eq ne gt ge lt le
is_finite is_infinite is_null is_not_null is_nan
```

默认目标：`DIRECT_CONDITION`。允许 condition/mask/gate，禁止普通 standalone numeric alpha terminal。

## R18-010：cross-sectional aggregate 的 global broadcast 不能当 stock alpha

逐个检查：

```text
cs_count
cs_mean
cs_std
cs_sum
cs_mad
cs_quantile
cs_valid_count
cs_coverage_ratio
cs_hartigan_dip
```

若同日全股票相同：`DIRECT_GLOBAL_STATE`，terminal false。

机器检测：
```python
nunique_across_stocks_by_date
```

## R18-011：group aggregate 必须按真实输出 cardinality 判 role

逐个检查：

```text
group_count
group_max
group_mean
group_min
group_std
group_sum
group_weighted_mean
group_valid_count
group_distribution_js_divergence
group_feature_spectral_gap
group_feature_second_mode_localization
```

组内广播 → `DIRECT_GROUP_STATE`。  
个股相对组差异，如 `group_rank/group_zscore/group_neutralize/group_percentile` → `DIRECT_ALPHA`。

## R18-012：bucket/categorical output 不能按连续强弱直接 rank

```text
cs_bucket
cs_bucket_fixed
cs_bucket_historical
candlestick_pattern
```

若 category id → `DIRECT_STATE`，只用于 category condition/interaction，不把编号本身当连续强度。

## R18-013：K 线二值 pattern / breakout 重新分 EVENT/STATE

至少：
```text
cdl_doji
cdl_hammer
cdl_inverted_hammer
cdl_shooting_star
cdl_marubozu
cdl_spinning_top
cdl_engulfing
cdl_inside_bar
cdl_outside_bar
cdl_dragonfly_doji
cdl_gravestone_doji
cdl_hanging_man
cdl_harami
cdl_harami_cross
cdl_piercing
cdl_dark_cloud_cover
cdl_morning_star
cdl_evening_star
cdl_three_white_soldiers
cdl_three_black_crows
cdl_tweezer_top
cdl_tweezer_bottom

ts_breakout_high
ts_breakdown_low
ts_new_high
ts_new_low
ts_resistance_break
ts_support_break
```

若输出 0/1/NaN 或 {-1,0,1} → EVENT/STATE；连续 strength 才 ALPHA。必须实际看 dtype/unique/value-domain，而非名字猜。

## R18-014：A 股 limit 家族按事件/强度/统计量拆 role

布尔 touch/close/failed/open-limit → EVENT/STATE。  
distance/density/asymmetry/days-since/streak 等连续统计才进入 ALPHA/STATE statistic lane。

## R18-015：`fin_applicability_mask/index_member/index_entry_exit_event`

预期：
```text
fin_applicability_mask -> DIRECT_CONDITION
index_member -> DIRECT_STATE
index_entry_exit_event -> DIRECT_EVENT
```

数值 age/decay 可保留 alpha/state statistic。

---

# 5. Price-level / raw-level 算子：保留，但多数改成 DIRECT_INTERMEDIATE

## R18-016：price level 不能默认 terminal

至少逐个检查：
```text
rolling_vwap
true_range
ATR_WILDER

donchian_upper
donchian_lower
donchian_mid

KeltnerMid
KeltnerUpper
KeltnerLower

ichimoku_tenkan
ichimoku_kijun
ichimoku_senkou_a
ichimoku_senkou_b

KAMA
DEMA
TEMA
PSAR
Supertrend  # 若输出 level

ts_prev_high
ts_prev_low
ts_last_pivot_high
ts_last_pivot_low
ts_nth_pivot_high
ts_nth_pivot_low
ts_resistance_level
ts_support_level

candle_body
candle_abs_body
candle_range
candle_upper_shadow
candle_lower_shadow
candle_gap

ts_swing_amplitude
ts_channel_width
ts_consolidation_width
```

output unit 为 price 且不 scale-invariant → `DIRECT_INTERMEDIATE`, terminal false。

## R18-017：归一化/相对版本可保留 DIRECT_ALPHA

例如：
```text
vwap_deviation
NATR
KeltnerPosition
ichimoku_cloud_position
candle_gap_pct
candle_gap_atr
ts_swing_amplitude_pct
ts_swing_amplitude_atr
ts_channel_width_pct
ts_channel_width_atr
ts_distance_to_high
ts_distance_to_low
ts_distance_to_resistance
ts_distance_to_support
```

逐个验证 dimensionless/relative semantics。

## R18-018：为每个 level intermediate 自动提供 composition templates

例如：
```text
(price-level)/scale
distance-to-level
break-strength
level-slope/scale
```

确保 intermediate 不是死代码，而是真能参与矿工组合。

---

# 6. Missing/source transform 不占 alpha 搜索分支

## R18-019

至少：
```text
fillna_const
coalesce
cs_fill_mean
cs_fill_median
cs_impute_mean
cs_impute_median
group_impute_median
ffill_limit
ts_ffill_limited
```

设为 `DIRECT_SOURCE_TRANSFORM`，只在数据处理/missing slot出现，不能 terminal。missing policy 默认不是 alpha 搜索维度。

---

# 7. 任意数学函数重新审“是否值得自动搜索”

## R18-020

重新审：
```text
sin cos acos asin atan atan2
floor ceil round fix
cbrt sqrt_abs square signed_power power
lerp flex_max flex_min
```

不要因为数学合法就进入自动 factor grammar。

处理规则：
- 被 recipe需要 → internal/intermediate；
- 有明确金融/统计用途 → direct；
- 只是制造周期/离散跳变/搜索噪声 → 移出默认 mining，必要时 ResearchTool/internal；
- 不机械删除代码。

## R18-021：单调变换防“假多样性”

给：
```text
exp log sqrt signed_log sigmoid tanh rank scale normalize unitize
```

声明 `MonotonicTransformClass`。

Rank-IC/search 时对严格 monotonic 等价表达做 semantic dedup，不把 x/exp(x)/sigmoid(x)/rank(x)都当新的独立能力。

---

# 8. 现有 duplicate / dead-param audit 不足以证明可用

## R18-022：semantic duplicate detector 不能用“同参数/同单位/同 grain”判重复

必须多层证明：
- implementation/AST/lowering semantic fingerprint；
- randomized differential tests；
- 多参数 grid；
- 多种 input archetype。

分类：
```text
EXACT_EQUIVALENT
MONOTONIC_EQUIVALENT
SCALE_EQUIVALENT
ALIAS_ONLY
RELATED_NOT_DUPLICATE
```

只有 EXACT_EQUIVALENT 自动合并 canonical。

## R18-023：dead-param detector 必须覆盖 100% retained searchable params

现有采样约几十个、单一 Gaussian、单 panel 调用、异常 silent skip 都不够。

每个 retained mineable operator 的每个 searchable parameter至少测：
```text
default
low
mid
high
valid choices
boundary valid
```

changing param 必须改变 output 或 execution semantics。否则修/删 param。

## R18-024：dead-param hash 必须保留 NaN mask/index/shape

不能先删 non-finite 再 hash，否则两份不同 NaN 拓扑可能误判相同。

## R18-025：多输入 operator 的 parameter injectivity probe 要实际提供所有 panel inputs

不能只 `operator.calculate(x, **kwargs)` 后异常就跳过。

## R18-026：ParamSpec `MISSING` sentinel 不得被当 default runtime value

audit generator必须与 production parameter authority完全一致。

## R18-027：choices alternate 必须选合法 choice，而不是拿 min/max拼一个可能不存在的值。

---

# 9. 参数“合法”不等于“值得搜索”

## R18-028

每个 parameter 声明：
```text
full
coarse
fixed
excluded
```

full 只给经济维度/真正 horizon/threshold。  
bins、n_segments、embedding_dim、max_iter、epsilon、kernel bandwidth、Theiler window、n_components 等 estimator knob → coarse/fixed/excluded。

---

# 10. input semantic types 与 source closure 必须逐 operator 实锤

## R18-029：`input_semantic_types` 不能返回 field name

每个 positional slot使用 `InputSlotSpec`：
```text
parameter
allowed_semantic_kinds
allowed_units
allowed_roles
cardinality
axis semantics
```

retained mining operator不允许 unknown/field-name冒充 semantic type。

## R18-030：`required_sources` 不能绝大多数 fallback `daily_bar`

逐 operator构造真实 source closure：
```text
daily
minute
benchmark
industry/group
market_cap
fundamental
holder
index
event
daily+minute
daily+fundamental
daily+benchmark
relation+return
...
```

每个 direct-use operator至少一个 executable source recipe。

## R18-031：`mining_eligible` 升级为 contextual direct usability

必须统一回答：
```text
role legal?
AST position legal?
source recipe exists?
market/frequency supported?
params legal?
evidence current?
cost budget?
```
然后执行 smoke。

## R18-032：fundamental-period operator 要有标准日频 PIT projection

自然 grain 是 fundamental period，但最终挖 daily factor。必须有标准 `GrainProjection`，不能让每个因子自己猜 ffill/asof。

---

# 11. Reviewed migration 不能再是假“批量 review”

## R18-033

当前大量 `DAILY_FACTOR_MIGRATED` 通过统一：
```text
review_id=seed-2026-08-manual
semantic_hash=""
```
进入 reviewed manifest。

最终 retained direct operator必须有：
```text
review_id
non-empty semantic_hash
direct_use_status
terminal_allowed
smoke_evidence_id
```

seed+empty hash 只能算 migration backlog。

---

# 12. 当前测试哲学需要反转：不保数量，只保质量

## R18-034：删除 `len(records)>=1000` / `manifest count>=1000`

替换为：
```text
ALL_RETAINED_HAVE_DIRECT_STATUS
ALL_DIRECT_HAVE_SMOKE_RECIPE
ALL_DIRECT_SMOKE_EXECUTE
NO_DUPLICATE
NO_DEAD_PARAM
NO_GHOST
NO_UNRESOLVED
NO_INVALID_TERMINAL_ROLE
```

canonical 数减少完全允许。

## R18-035：删除 `US operators <= A-share operators` 这种错误集合假设

R17 已处理市场语义；本轮测试层必须允许：
```text
A-only
US-only
both
provider-required
```

不再用数量/集合包含关系代表市场能力正确。

## R18-036：现有 admission test 只验证 role 有值不够

必须新增 direct execution + role-output validation。

---

# 13. stale surface / ghost canonical 全清

## R18-037

当前 surface 仍可看到历史 `state_since_reduce`，而其它代码已明确其已退休并拆为：
```text
state_since_sum
state_since_mean
state_since_count
state_since_last
```

新增 invariant：
```text
surface_names - registry_canonicals == ∅
registry_public - declared_direct/status == ∅
```

retired name不能留在 surface/manifest/docs/evidence/cold-start。

---

# 14. `PRODUCTION_DENIED` 与 authoring surface 的冲突逐个清账

## R18-038：风险因子类能修就不要长期 deny

重新审：
```text
downside_beta
tail_beta
residual_momentum_capm
coskewness_to_market
idio_vol
idio_skew
rolling_beta_to_market
```

`rolling_beta_to_market`若有 self-inclusion，改 explicit benchmark/ex-self 后新语义 direct usable；旧有问题 canonical 删除/alias。

其余有明确量化价值，只要问题是 benchmark/PIT/min-samples，应修到 `DIRECT_ALPHA/HIGH_COST`，不永久 deny。

## R18-039：基本面常用比率应做 audited recipes

至少：
```text
operating_margin
current_ratio
quick_ratio
debt_to_equity
```

在 R17 financial provider完成后实现严格 recipe，明确行业适用性/分母/会计口径，转 `DIRECT_RECIPE/DIRECT_ALPHA`。

## R18-040：legacy `quarter/ttm/yoy/avg2` 收口

若已有：
```text
quarter_from_cumulative
ttm_from_cumulative
ttm_from_quarterly
yoy_by_period
period_average
```
旧名 DELETE_OBSOLETE 或 alias，不再第二搜索分支。

## R18-041：`rank_corr/ts_poly2_coeff/ts_poly2_resid/digital_count`

不能长期“surface存在+hard deny”。逐个归：
- direct；
- research tool；
- obsolete/duplicate。

## R18-042：`intraday_vwap_deviation` 按市场 context direct use

A 有 certified minute source则 A direct；US没有则仅 US blocked。不要全局 source-blocked。

## R18-043：`fin_total_operating_accruals`

若 R17后两市场仍无可靠 depreciation/amortization source且无已接provider → 从 public factor层 `DELETE_NO_DATA`，只留backlog/docs。

## R18-044：`fin_revision_* / fin_expectation_* / fin_surprise*` 逐个证明当前真实 source

每一个 operator：
```text
required concepts
A provider?
US provider?
```
两个市场都无 → DELETE_NO_DATA / Research backlog，不占 public mining。

---

# 15. full-history recursive operators：全部给最终状态

## R18-045

逐个检查当前 full-history replay 家族，包括至少：
```text
expanding_rank
hump_decay
KAMA
Supertrend
SupertrendDirection
PSAR
DMI_plus
DMI_minus
DX
NATR
PPO
PPO_signal
PPO_hist
PVO
PVO_signal
PVO_hist
KeltnerMid
KeltnerUpper
KeltnerLower
KeltnerPosition
TSI
TSI_signal
DEMA
TEMA
ChaikinOscillator
ForceIndex
ts_sma_cn
state_latch
state_hold
state_slew_limit
state_deadband
state_ewm_if
event_refractory
cross_event
directional_change_state
directional_change_extent
state_since_trend_tstat
ts_cumulative_deviation_score
ts_threshold_cycle_period
ts_threshold_cycle_asymmetry
state_episode_mfe
state_episode_mae
state_episode_efficiency
state_episode_retrace_ratio
state_episode_excursion_balance
ts_interval_nesting_depth
candle_gap_atr
```

三选一：
1. 值得高频：补 checkpoint/restore；
2. 数学有价值但高成本：`DIRECT_*_HIGH_COST` + reliable full-history replay；
3. 不值得维护：ResearchTool/delete。

最终不能以 pending收尾。

## R18-046：`expanding_rank` 特别检查 sample-start dependence

保留必须固定 anchor并进入 factor identity；否则移出默认 mining，推荐 bounded rolling alternative。

---

# 16. 高级统计/非线性算子：不是越多越好

## R18-047

重点逐个审核：
```text
ts_markov_*
ts_km_*
ts_first_passage_*
ts_extremal_index
ts_mean_excess_slope
ts_gpd_shape_pwm
ts_transfer_entropy_*
ts_mmd_rbf_shift
ts_hsic
ts_conditional_mutual_information
ts_recurrence_*
ts_hvg_*
ts_matrix_profile_*
ts_spectral_*
ts_hankel_*
ts_ssa_*
ts_generalized_hurst_*
ts_multifractal_*
ts_delay_intrinsic_dimension
ts_persistence_entropy_*
ts_hartigan_dip
cs_hartigan_dip
```

每个增加 effective sample contract：
```text
min_effective_samples
min_unique_values
min_state_occupancy
min_event_count
max_missing_ratio
parameter-dependent sample floor
```

## R18-048：高级 estimator knobs只能 coarse/fixed

embedding/bins/epsilon/states/components/bandwidth等禁止自由连续搜索。

## R18-049：实际 benchmark决定 high-cost lane

至少在适当规模下记录：
```text
wall time
peak memory
scaling
preferred backend
```

## R18-050：不以单次IC差删高级 primitive

删除依据数学/估计/数据/重复/无composition价值，不以单因子IC。

---

# 17. 按 role 做真实 output validation

## R18-051：DIRECT_ALPHA/HIGH_COST 横截面信息量 gate

真实 A/US representative panel：
```text
finite coverage
cross-sectional std
nunique
days with useful dispersion
dominant-value share
NaN/Inf rate
```

长期同日全股票相同不能 stock-alpha terminal。

## R18-052：EVENT/CONDITION/STATE 用事件标准

检测：
```text
event frequency
transition frequency
always-zero/one rate
spell length
coverage
```

## R18-053：GROUP_STATE/GLOBAL_STATE验证 cardinality

GROUP_STATE：组内一致、组间可不同。  
GLOBAL_STATE：同日全市场一致、跨日可变。

## R18-054：DIRECT_INTERMEDIATE必须有合理 composition

每个至少一条 `recommended_compositions`；完全无组合用途再评 DELETE_USELESS。

---

# 18. 条件与 control-flow 真实接入 grammar

## R18-055：所有 `*_if` 声明 condition slot

至少：
```text
ts_mean_if
ts_sum_if
ts_std_if
ts_min_if
ts_max_if
ts_quantile_if
ts_corr_if
ts_beta_if
ts_regression_resid_if
ts_cov_if
ts_rank_if
```

condition slot只能接受 Condition/Event/State Bool。

## R18-056：`where/trade_when` 设为 control-flow，不是独立 alpha

新增 `DIRECT_CONTROL_FLOW` 或明确 `DIRECT_INTERMEDIATE`，只在 AST内部出现。

## R18-057：mask/state arithmetic语义

允许 `state*alpha` 等明确组合；禁止 `sqrt(group_id)`, `rank(category_id)` 等无意义组合。

---

# 19. domain-sensitive math必须直接可安全用

## R18-058

逐个：
```text
log log10 log2 sqrt inverse divide power signed_power asin acos exp
```

声明：
```text
valid domain
invalid behavior
overflow
zero
negative
NaN
```

grammar尽量不生成明显 domain-invalid AST。

## R18-059：divide/safe_div_null/inverse/reciprocal/protected_div 收口

一套 public canonical semantics，其余 alias/internal。denominator null/epsilon policy明确。

## R18-060：scale/normalize/unitize/zscore/rank exact/axis audit

重复合并，axis模糊修成明确 ts/cs/group semantic。

## R18-061：flex_min/flex_max/minimum/maximum exact duplicate audit

若仅实现差异，一个 canonical + backend/policy，不多个搜索 branch。

## R18-062：row_sum_skipna等 axis-specific operator声明 axis semantics

DirectUse matrix必须包含：
```text
elementwise/time_series/cross_section/group/relation/session/global
```

---

# 20. pivot/pattern/technical具体可用性

## R18-063：confirmed pivot必须在确认日可见

`ts_confirmed_pivot_* / ts_last_pivot_* / ts_nth_pivot_*` 不允许 backdate到中心峰值日供矿工使用。metadata记录 timestamp semantics。

## R18-064：所有 pivot-dependent pattern重新 prefix-invariance

double-top/head-shoulder/triangle/wedge/channel/flag/cup/retest/123 等全部真实验证。二值 confirmed pattern → EVENT；连续 score → ALPHA。

## R18-065：技术指标“line/level/position/direction”分 role

例如：
```text
MACD_line/signal/hist
PPO/signal/hist
PVO/signal/hist
TSI/signal
Keltner level/position
Ichimoku level/position
Supertrend level/direction
```

dimensionless oscillator/spread → alpha；raw level → intermediate；direction → state。

## R18-066：蜡烛 raw geometry vs relative geometry

raw price unit body/range/shadow/gap → intermediate。  
ratio/pct/ATR-normalized/strength → alpha。

---

# 21. activity/size-sensitive输出明确暴露类型

## R18-067

如：
```text
dollar_volume
ts_average_volume
adv
rolling_obv
rolling_pvt
```

逐个标：
```text
scale_sensitive
size_sensitive
liquidity_exposure
```

可作为 alpha，但矿工应优先提供 rank/zscore/relative组合，而不是假装 scale-neutral。

---

# 22. 为搜索多样性建立 family，而不是堆名字

## R18-068：每个 DIRECT_ALPHA 有 `economic_effect_family`

至少支持：
```text
trend
mean_reversion
volatility
liquidity
volume
price_volume_interaction
range
gap
tail
distribution_shape
autocorrelation
breakout
pattern
fundamental_value
fundamental_quality
fundamental_growth
fundamental_revision
ownership
index_flow
event
intraday_microstructure
regime
nonlinear_dependence
spectral
geometry
```

## R18-069：增加 `semantic_redundancy_group`

用于 family budget/cold-start多样性，不因为高相关就粗暴删除。

---

# 23. 让每个 operator “拿来就能跑”：default input binder

## R18-070

每个 operator 声明 `default_input_recipe`，例如：

```text
ts_log_return:
  x = continuous_close

parkinson_vol:
  high = continuous_high
  low = continuous_low

amihud_illiquidity:
  ret = return_decimal
  amount = amount_local

size_neutralize:
  x = factor
  size = log(market_cap)

industry_neutralize:
  x = factor
  group = industry_group
```

具体市场字段仍调用 R17 provider resolver，本轮只负责 operator-level binding recipe。

---

# 24. 每个 retained canonical 至少一条真实 smoke recipe

## R18-071：新建 `OperatorSmokeRecipe`

字段：
```text
canonical
market
role
input_bindings
parameter_values
required_context
expected_output_kind
expected_grain
expected_cardinality
minimum_coverage
expected_value_domain
```

A/US都支持最好各一条。

## R18-072：smoke必须真正执行

流程：
```text
resolve source
compile
execute pandas reference
validate output
execute certified fast backend
compare
```

## R18-073：fixture不能只有 Gaussian

准备：
```text
TREND
MEAN_REVERTING
VOL_CLUSTER
JUMP
GAP
SPARSE
MISSING_BLOCK
CONSTANT
OUTLIER
BINARY_EVENT
GROUP_PANEL
MARKET_BENCHMARK
FUNDAMENTAL_EVENT
MINUTE_SESSION
```
并配真实 A/US 小样本。

---

# 25. 参数空间真正可直接搜

## R18-074：default params必须合法、有意义、执行通过

不允许“存在但默认调用报错”。

## R18-075：合法参数域抽样全部可运行

min/p25/mid/p75/max/choices，加 relational constraint，检测 all-NaN/Inf/zero-variance/runtime explosion，必要时收窄 ParamSpec。

## R18-076：multi-input operator做 input permutation negative tests

price/volume、high/low、ret/benchmark、x/weight、x/group等错误互换必须 compile fail。

## R18-077：multi-panel axis alignment真实测试

禁止 silent universe intersection/outer join，除非显式 contract。

---

# 26. backend目标是“至少一条可靠直接执行路径”，不是强迫三份手写实现

## R18-078

DirectUse matrix列：
```text
reference_backend
certified_fast_backend
fallback_backend
```

复杂算子可 production fallback reference backend，前提成本/lane明确。

## R18-079：允许 HIGH_COST完成状态，不允许长期 pending

full-history可靠就直接 high-cost；不可靠就修/删。

---

# 27. 搜索空间按 role/lane真正路由

## R18-080

至少：
```text
alpha_direct
alpha_high_cost
condition
state
event
group_state
global_state
intermediate
control_flow
fundamental_pit
intraday_eod
```

不同 lane独立 budget。SOURCE_TRANSFORM不占 alpha tree depth。semantic complexity cost与runtime cost分开。

---

# 28. cost contract逐 operator闭环

## R18-081

所有 retained operator：
```text
cost_contract_declared=True
```

至少记录：
```text
time complexity class
memory class
runtime benchmark
preferred backend
full-history requirement
```

不再靠 category/name fallback。

---

# 29. “完全无用”结构性判断，不拍脑袋

## R18-082

对每个 candidate计算：
```text
direct role exists?
unique mathematical capability?
source exists?
non-degenerate output?
reasonable composition exists?
parameter injective?
not exact duplicate?
cost acceptable?
```

满足多数否定才 `DELETE_USELESS`。

---

# 30. 删除必须干净，兼容由 migration承担

## R18-083：删除 canonical 后清所有 ghost reference

检查：
```text
registry
aliases
surface
migration manifest
production hardening
market contract
mining catalog
cold-start
recipes
docs
evidence
tests
backend allowlist
```

## R18-084：alias只能指 retained canonical，alias chain尽量 flatten

## R18-085：旧公式兼容用 `operator_migration_map.json`

exact duplicate/obsolete old name自动rewrite到新 canonical，不让旧 canonical继续占搜索 branch。

---

# 31. primitive vs recipe重新审

## R18-086

可由已有 primitives稳定组合的复杂 operator优先 `DIRECT_RECIPE`。  
若 profiling证明 fused执行显著收益，可保留 fused backend，但 mining只暴露一个 canonical representation。

## R18-087：建立 `CanonicalRepresentationPolicy`

```text
preferred_mining_representation
manual_aliases
fused_execution_equivalents
```

执行优化不制造搜索多样性。

---

# 32. family budget，而非过度删除

## R18-088

高相关但非完全相同的 range-vol/robust-scale/oscillator/entropy/pattern等不要因为相关性删，放同 `semantic_redundancy_group` 做 quota。

## R18-089：candlestick/chart-pattern family设置每表达式事件数量 budget。

## R18-090：nonlinear/statistical family设置 complexity tax。

---

# 33. 每个 retained operator必须回答“为什么留、怎么用、不能怎么用”

## R18-091：`retention_reason`

例如：
```text
unique family
control-flow primitive
level building block
industry-standard indicator
robust alternative
market-specific mechanism
nonlinear dependence
fundamental transform
```

## R18-092：`recommended_usage`

```text
terminal
intermediate
condition
interaction
normalizer
state gate
event gate
group context
global context
```

## R18-093：`forbidden_usage`

例如：
```text
rolling_vwap: no standalone cross-sectional terminal
cs_mean: no stock-level terminal
cdl_hammer: no continuous-rank treatment without event encoding
```

## R18-094：description lint

description必须与真实 role/unit/semantics一致。

---

# 34. admission matrix补真实 detectors

## R18-095

当前 blocker vocabulary 中以下必须真正实现 detector，不只是有名字：

```text
B24 LOW_EMPIRICAL_COVERAGE
B25 INSUFFICIENT_EFFECTIVE_SAMPLE
B26 PARAM_SPACE_UNSAFE
B27 DEAD_PARAMETER
B28 SEMANTIC_DUPLICATE
B29 MATH_DEFINITION_DEFECT
B30 UNIT_CONTRACT_DEFECT
B31 AXIS_CONTRACT_DEFECT
B32 MISSING_TIME_TOPOLOGY_DEFECT
```

新增：
```text
B36 DEFAULT_INVOCATION_FAIL
B37 OUTPUT_ROLE_MISMATCH
B38 TERMINAL_DEGENERATE
B39 NO_COMPOSITION_PATH
B40 NO_DIRECT_SOURCE_RECIPE
B41 PARAMETER_INJECTIVITY_FAIL
B42 GHOST_CANONICAL
B43 EXACT_DUPLICATE
B44 UNBOUNDED_RUNTIME
B45 INVALID_DEFAULT_INPUT_BINDING
```

最终 applicable detector coverage=100%。

---

# 35. 新的全算子 DirectUse Matrix

## R18-096

生成：
```text
factor_engine/docs/R18_DIRECT_USE_MATRIX.json
factor_engine/docs/R18_DIRECT_USE_MATRIX.csv
factor_engine/docs/R18_DIRECT_USE_MATRIX.md
```

每个 runtime registered canonical一行，至少包含：

```text
canonical
aliases
module
class
authoring_tier

direct_use_status
mining_role
terminal_allowed
allowed_ast_positions

retention_reason
delete_reason
replacement

economic_effect_family
semantic_redundancy_group

input_slots
input_semantic_kinds
input_units
input_cardinality
input_axis_semantics

output_semantic_kind
output_unit
output_cardinality
output_value_domain

default_input_recipe
smoke_recipe_ids

supported_markets
source_recipes

default_params
searchable_params
search_grade_by_param
parameter_injectivity_passed

stateful
execution_model
checkpoint_supported
full_history_replay_allowed

runtime_cost
memory_cost
preferred_backend
reference_backend

default_compile_passed
default_execution_passed
synthetic_suite_passed
real_sample_passed

finite_coverage
cross_section_dispersion
event_rate
state_transition_rate
role_validation_passed

duplicate_class
semantic_hash

production_certified
directly_usable
```

每一行 `direct_use_status` 非空。

---

# 36. 动态枚举，不保历史数量

## R18-097

```python
load_all()
all_ops = sorted(OperatorRegistry.list_canonical())
```

整改前/中/后各枚举一次。

报告：
```text
before count
after count
deleted
alias-created
moved research
moved internal
direct alpha
direct high cost
state/event/condition
intermediate
source transform
```

不硬编码 1409/1300/1000。

---

# 37. 每个 retained operator生成最小可运行示例

## R18-098

生成：
```text
factor_engine/docs/R18_OPERATOR_SMOKE_RECIPES.json
```

包括 alpha/condition/intermediate/control-flow 等真实推荐组合。

alias也要 parse/compile/execute并与 canonical等价。

---

# 38. 文档和注册制度

## R18-099

每个 public operator documentation必须回答：
1. 数学做什么？
2. 输入是什么？
3. 输出是什么？
4. mining role是什么？
5. 推荐怎么组合？
6. 哪些市场/source可直接用？

## R18-100

未来任何新 canonical 注册时必须同时提供：
```text
direct_use_status
input slots
output semantic
retention reason
smoke recipe
param roles
cost contract
source recipe
```
缺一项注册失败。

---

# 39. 进一步的删冗规则

## R18-101：搜索 `_v2/_new/_alt/_fast/_exact/_safe/_robust`

不是见到就删，但每一对都做 semantic differential audit。仅实现路径不同 → 一个 canonical + backend。

## R18-102：只改默认参数不能形成两个 canonical

window=20/60等必须一个 operator+parameter。

## R18-103：不同 backend不能形成多个 semantic canonical。

## R18-104：不同 NaN policy默认不能形成不同 alpha canonical。

## R18-105：不同 estimator solver但目标语义相同，应一个 canonical + policy/backend。

---

# 40. technical/pattern family 保留但控制搜索预算

## R18-106

MACD/PPO/PVO/TSI等 line/signal/hist并非 exact duplicate，可保留，但分别明确 role/unit/input，并设 family budget。

## R18-107

candlestick几十种事件可保留，但 `semantic family=candlestick_event`，限制每 expression数量。

## R18-108

chart pattern同理，causal detector正确后保留，明确 event vs score、cost、budget。

---

# 41. direct-use evidence不以IC为标准

## R18-109

Evidence分：
```text
math
causality
source
execution
role
nondegeneracy
cost
```

不是“IC>0.02才叫算子有效”。

---

# 42. 给 AlphaProbe 的真正 direct catalog / grammar

## R18-110

生成：
```text
factor_engine/build/mining/direct_mining_catalog.json
```

只含 DIRECT_*，带 role/slots/composition/param/cost/family/market/source recipe。

## R18-111

生成 role-aware grammar：
```text
AlphaNode
ConditionNode
StateNode
EventNode
IntermediateNode
GroupStateNode
GlobalStateNode
ControlFlowNode
```

## R18-112

最终 invariant：
```text
DIRECT_OPERATOR_NOT_REACHABLE_IN_GRAMMAR == ∅
```

reachable不等于terminal。

## R18-113

注册了但非 internal/research/source-transform、又永远不可达 grammar、也无 manual DSL用途的 ghost capability删除。

---

# 43. 新永久审计脚本

## R18-114

至少新增/完善：

```text
scripts/audit_all_operators_direct_use.py
scripts/audit_operator_parameter_injectivity.py
scripts/audit_operator_output_role.py
scripts/audit_operator_semantic_duplicates.py
scripts/audit_operator_runtime_cost.py
scripts/audit_operator_source_recipes.py
scripts/audit_operator_grammar_reachability.py
scripts/audit_surface_registry_ghosts.py
scripts/export_direct_mining_catalog.py
```

`audit_all_operators_direct_use.py` 对每个 canonical只能 PASS/MOVE/DELETE/FAIL，不能 SKIP。

---

# 44. 100% coverage hard gates

## R18-115

Parameter injectivity：
```text
untested retained searchable params = 0
dead params = 0
```
或死参数已移除 search space。

## R18-116

Output-role auditor：
100% direct operators都有实际 output role evidence。

## R18-117

Semantic duplicate auditor只自动合并证明 exact equivalent 的。

## R18-118

Source recipe auditor：
每个 direct operator至少一个 supported context；0 context → DELETE_NO_DATA/ResearchTool。

## R18-119

Grammar auditor：
每个 role在正确 AST位置可达。

## R18-120

Surface/registry ghost auditor双向检查 surface/registry/migration/evidence/alias/recipe。

---

# 45. cold-start 与自动挖掘最终目标

## R18-121

cold-start不再追求“覆盖所有 registered canonical”，而是：
```text
覆盖所有 relevant semantic families
只使用 direct-use operators
没有 exact duplicate representation
没有 unsupported role
```

## R18-122

冷启动多样性基于：
```text
economic family
source domain
semantic redundancy group
role
horizon
```
而不是 operator数量。

## R18-123

AlphaProbe/AlphaMiner直接消费 role：
- condition slot只取 condition/event/state；
- alpha child取 alpha/intermediate；
- global context取 global state；
- terminal只取合法 terminal roles。

---

# 46. 最终 DirectUseStatus 决策，一个都不能跳

## R18-124

对 `all_ops` 循环：

```python
for canonical in all_ops:
    math = audit_math(canonical)
    role = resolve_direct_use_role(canonical)
    inputs = resolve_input_contract(canonical)
    sources = resolve_source_recipes(canonical)
    params = audit_param_space(canonical)
    duplicates = audit_semantic_duplicates(canonical)
    runtime = run_smoke_suite(canonical)
    output = audit_output_semantics(canonical)
    cost = benchmark_cost(canonical)

    verdict = decide(...)
```

异常不能 `except: continue`，必须 `AUDIT_FAILED` 并阻止 closure。

## R18-125

每个 canonical最终只落：
```text
KEEP_DIRECT_ALPHA
KEEP_DIRECT_HIGH_COST
KEEP_DIRECT_STATE/GATE/EVENT
KEEP_DIRECT_INTERMEDIATE
KEEP_DIRECT_SOURCE_TRANSFORM
KEEP_DIRECT_RECIPE
MOVE_RESEARCH_TOOL
MOVE_INTERNAL
DELETE_AND_ALIAS
DELETE
```

没有“以后再看”。

---

# 47. “完全无用挖因子的算子”判断标准

## R18-126

满足多数：
```text
无独特数学能力
无独特经济/市场语义
无合理 non-terminal角色
无source
输出长期结构退化
参数不影响输出
已有 exact replacement
runtime极高且无表达增量
纯诊断
```
才 DELETE_USELESS。

“IC低”不是删除条件。

---

# 48. DirectUse 硬性 CI

## R18-127

最终必须全部为空：

```text
REGISTERED_PUBLIC_WITHOUT_DIRECT_STATUS
DIRECT_WITHOUT_INPUT_CONTRACT
DIRECT_WITHOUT_SMOKE_RECIPE
DIRECT_DEFAULT_EXECUTION_FAILURE
DIRECT_WITH_DEAD_SEARCHABLE_PARAM
DIRECT_WITH_UNKNOWN_OUTPUT_SEMANTIC
ALPHA_TERMINAL_WITH_GLOBAL_CONSTANT_OUTPUT
ALPHA_TERMINAL_WITH_GROUP_CONSTANT_OUTPUT
CONDITION_AS_ALPHA_TERMINAL
CATEGORY_AS_CONTINUOUS_ALPHA
INTERMEDIATE_LEVEL_AS_DEFAULT_TERMINAL
SOURCE_TRANSFORM_AS_ALPHA_TERMINAL
EXACT_SEMANTIC_DUPLICATE_PUBLIC_CANONICALS
GHOST_SURFACE_CANONICALS
ALIASES_TO_DELETED_CANONICALS
PUBLIC_NO_DATA_CANONICALS
PUBLIC_NONCAUSAL_CANONICALS
DEFAULT_MINING_MANIFEST_CONTAINS_PENDING
DEFAULT_MINING_MANIFEST_CONTAINS_RESEARCH
DIRECT_OPERATOR_NOT_REACHABLE_IN_GRAMMAR
```

---

# 49. 改测试：质量导向

## R18-128

删除 minimum operator count tests。

改为 parameterized：
```python
@pytest.mark.parametrize("canonical", all_direct_canonicals)
def test_direct_operator_smoke(canonical):
    ...
```

读取 R18 smoke recipes，未来新增 operator自动进入 gate。

---

# 50. 最终产物

## R18-129

至少生成：

```text
factor_engine/docs/R18_DIRECT_USE_MATRIX.json
factor_engine/docs/R18_DIRECT_USE_MATRIX.csv
factor_engine/docs/R18_DIRECT_USE_MATRIX.md

factor_engine/docs/R18_OPERATOR_SMOKE_RECIPES.json

factor_engine/docs/R18_DELETE_PLAN.json
factor_engine/docs/R18_DELETE_PLAN.md

factor_engine/docs/operator_migration_map.json

factor_engine/build/mining/direct_mining_catalog.json
factor_engine/build/mining/direct_mining_manifest.json
factor_engine/build/mining/research_operator_manifest.json
factor_engine/build/mining/remediation_backlog_manifest.json
```

---

# 51. 全量回归

## R18-130

完成：
1. FactorEngine全 tests；
2. DataAccess integration；
3. all direct operator smoke；
4. all aliases；
5. all searchable param injectivity；
6. semantic duplicates；
7. output role；
8. grammar reachability；
9. ghost refs；
10. backend/evidence重生；
11. cold-start direct-only；
12. deterministic hashes。

任何 permanent test failure不能以“别的并行 session 在改”作为最终完成理由。

---

# 52. Definition of Done

全部满足才结束：

- [ ] 动态枚举所有 current canonical，0 skip；
- [ ] 每个 canonical有最终 DirectUseStatus；
- [ ] retained public无 UNKNOWN/unresolved/vague pending；
- [ ] 默认 mining入口只读 direct-use authority；
- [ ] `api/mining_integration.py`不再绕过 direct catalog；
- [ ] data inputs/scalar params/context inputs彻底分开；
- [ ] terminal判定只有一个正向 authority；
- [ ] bool/category/state/group/global/intermediate不再误当普通 alpha terminal；
- [ ] price-level building blocks进入 intermediate；
- [ ] missing/source transforms退出 alpha search branch；
- [ ] arbitrary math完成 direct/internal/research/delete审计；
- [ ] exact duplicates合并；
- [ ] dead searchable params=0；
- [ ] retained operator defaults合法；
- [ ] retained operator input-slot semantics完整；
- [ ] retained direct operator smoke recipe覆盖100%；
- [ ] smoke compile+execute覆盖100%；
- [ ] terminal alpha有非退化输出证据；
- [ ] event/state/group/global输出符合角色；
- [ ] 每个 direct op至少一个真实 market/source context；
- [ ] full-history useful op已checkpoint或明确high-cost full replay；
- [ ] advanced estimator有effective-sample/search-role/cost contract；
- [ ] production denied与direct role无矛盾；
- [ ] no-data public canonical=0；
- [ ] ghost surface=0；
- [ ] alias to deleted=0；
- [ ] direct mining manifest pending=0；
- [ ] cold-start只使用direct operators；
- [ ] minimum-count tests删除；
- [ ] US⊆A股错误测试删除；
- [ ] R18 DirectUse Matrix覆盖100%；
- [ ] 全量 tests全绿；
- [ ] artifacts/evidence按最终代码重生。

---

# 53. 最终执行指令

现在直接在服务器工作区执行，不要只回复分析：

1. 读取当前真实 HEAD/dirty state；
2. 把 R17及更早整改视为已经完成，**不要重复那些旧整改项**；
3. `load_all()` 动态枚举所有当前 canonical；
4. 生成整改前 baseline；
5. 按本文件对每一个 operator审：
   - math；
   - role；
   - input slots；
   - output semantics；
   - params；
   - source recipe；
   - smoke；
   - non-degeneracy；
   - cost；
   - duplicate；
   - retention value；
6. 每个 operator必须给最终 verdict；
7. 能修到 direct 的全部修；
8. 有价值 state/intermediate/source-transform放正确 role，不误删；
9. 真正无用、无数据、重复、危险、obsolete 的 canonical删除/迁移；
10. 修 mining integration，让 AlphaProbe/AlphaMiner默认只消费 direct catalog；
11. 重建 direct-use grammar；
12. 生成 R18 DirectUse Matrix / Smoke Recipes / Delete Plan / Migration Map / Direct Mining Catalog；
13. 跑所有 direct operator smoke；
14. 跑全量 tests；
15. 重生 backend/evidence/manifest/cold-start；
16. 最后再动态枚举一次确认无 ghost；
17. 最终简洁汇总：
   - 改前/改后 operator 数；
   - 删除各类型数量；
   - 各 DirectUseStatus 数量；
   - smoke pass 数；
   - dead param 数；
   - exact duplicate 数；
   - unresolved 数；
   - 全量测试结果。

**不要以 FactorEngine 算子越多越好为目标。目标是：表达空间足够大，但留下来的每一个 canonical 都清楚、独特、可执行、可组合，并且 AlphaProbe/AlphaMiner 真正知道该如何直接使用它。**
