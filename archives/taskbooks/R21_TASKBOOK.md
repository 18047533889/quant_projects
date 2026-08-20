# FactorEngine R21 — Operator Rehabilitation × Backend Coverage × Production Certification 全量整改

> 原文由用户提供（2026-08-20），逐字保存。AI 不要改写本文件；执行状态记在
> `factor_engine/evidence/r2/` 各 YAML 与 `LOOP_ENGINEERING_STATUS.md`。
> 基线 HEAD（用户提供时）：`724cf286c567512efeefb2e24622eb0bb665680d` —
> 开始任何工作前必须重新读取当前 main HEAD，不得默认本 SHA 仍最新。

当前代码基线：
- Repo: /home/shw/quant_projects
- GitHub: 18047533889/quant_projects
- 开始工作前必须重新读取当前 main HEAD；不得默认本文记录的 SHA 仍是最新。
- 本任务重点只看 FactorEngine，DataAccess 仅在 PIT/source contract/backend pushdown 确实需要时做最小接口修改。
- 不要为了完成数量而简单把 direct_intermediate / research_tool / event / state 的 flag 改成 DIRECT_ALPHA。
- 核心目标是：凡是有真实经济信息、可以改造成 causal / dimensionless / stock-specific / terminal-safe 因子的现有算子，都尽量真正改好；确实不适合 terminal 的原始 primitive 保留为 intermediate/internal，并增加正确的 mineable sibling。
- 所有“实现完成”和“生产可用”必须分开。CLOSED_LOCAL ≠ PRODUCTION_CERTIFIED。
- 不得用 implementation 自己复制一遍形成所谓 independent oracle。
- 不得用另一个可能共享同一 bug 的 backend 自证正确。
- tests NOT_RUN 必须明确写 NOT_RUN。
- production certification 必须绑定 current source SHA / semantic hash / PhysicalImplementationID。
- 禁止 placeholder / TODO / approximation 冒充 canonical production math。
- 修改 numerical kernel 禁止全局 regex/sed 批量替换，逐函数改并逐函数验证。

======================================================================
一、先修当前已经确认的 P0 数学问题
======================================================================

P0-1. beta_residual_z / beta_divergence_pct 漏掉 OLS intercept

文件：
factor_engine/cleaned_operators/technical/group_state_v1.py

当前 prior-window market model 实际拟合：
r = alpha + beta * market + eps

历史 residual std 计算中实际上隐含了 alpha：
alpha_hat = mean(r) - beta_hat * mean(market)

但 signal row 当前代码却使用：
residual_t = r_t - beta_hat * market_t

这是错误的。

必须改为：
alpha_hat = mean(r_prior) - beta_hat * mean(m_prior)
residual_t = r_t - (alpha_hat + beta_hat * m_t)

_prior_beta_stats() 应返回至少：
(alpha, beta, residual_std)

然后：
beta_residual_z =
[r_t - (alpha + beta*m_t)] / prior_residual_std

beta_divergence_pct =
[r_t - (alpha + beta*m_t)] / normalization

现有 test oracle 也复制了相同错误，必须重写 independent oracle。

必须加确定性反例：
r_t = 0.01 + 1.5 * market_t
无噪声情况下正确 out-of-sample residual 应接近 0。
旧实现会残留约 0.01，必须确保新测试能杀死旧实现。

同时重新思考 beta_divergence_pct 的 denominator。
当前除以 |market_return_t| 在 market≈0 时非常不稳定。
至少比较三种定义并保留经济语义最合理的：
A. residual / |market_t|
B. residual / prior residual volatility
C. residual / stock prior volatility

A 可以保留为 divergence ratio，但必须设置 market-return floor 或直接 NaN；
B 就是 beta_residual_z；
可以考虑新增：
beta_residual_vol_pct = residual / prior stock volatility

---------------------------------------------------------------------

P0-2. 独立 Red-Team 全面检查“实现与 oracle 同错”

以后所有新 Rehabilitation operator 必须至少有一个：
- hand-derived scalar/window oracle；
- scipy/statsmodels/numpy.linalg 等独立数学 oracle；
- brute-force Python loop oracle；

但 oracle 不允许调用 production helper。

重点复核：
beta_residual_z
beta_divergence_pct
event_cluster_score
state_transition_surprise
vpin_pct
reg_forecast_error_pct
reg_slope_tstat
spectral_energy_ratio
wavelet_detail_energy_ratio
ex_self_zscore
ex_self_mad_z
sr_touch_count

======================================================================
二、继续把 price-level / intermediate 算子真正改造成 Direct Alpha
======================================================================

原始 level 保持 INTERMEDIATE，不直接 promotion。
新增 sibling 才进入 Direct Alpha。

---------------------------------------------------------------------
2.1 ATR / True Range 第二轮
---------------------------------------------------------------------

现有第一轮已有：
atr_pct
atr_zscore
atr_percentile
true_range_pct
true_range_surprise
true_range_zscore
atr_short_long_ratio
atr_acceleration

继续新增：

atr_surprise_prior
= atr_pct_t / mean(atr_pct[t-W:t-1]) - 1

true_range_surprise_prior
= tr_pct_t / mean(tr_pct[t-W:t-1]) - 1

atr_term_structure_log
= log(ATR_short / ATR_long)

atr_regime_strength
= zscore(log(ATR_short/ATR_long))

atr_vol_ratio
= ATR_pct / realized_volatility

atr_vol_divergence
= zscore(ATR_pct) - zscore(realized_vol)

atr_persistence
atr_percentile_change
atr_acceleration_z
range_vol_divergence

重点：
“surprise”应提供严格 prior baseline 版本，而不是全部都用包含当前值的 mean 自我稀释。

---------------------------------------------------------------------
2.2 EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA 第二轮
---------------------------------------------------------------------

现有：
*_distance_pct
ma_slope_pct
ema_crossover

继续补统一 MovingAverageDerived family：

ma_distance_atr
ma_distance_z
ma_distance_percentile

ma_slope(close, ma_window, slope_horizon)
不要只固定 diff(1)。

ma_slope_vol_scaled
ma_acceleration
ma_curvature
ma_slope_stability

ma_short_long_spread_pct
ma_short_long_spread_atr
ma_cross_strength
ma_cross_age
ma_cross_density
ma_cross_persistence

price_ma_reclaim_event
price_ma_reclaim_age
price_ma_rejection_strength

KAMA 特别增加：
kama_efficiency_adjusted_distance
kama_adaptation_speed
kama_adaptation_change
kama_trend_quality

ema_crossover 当前 {-1,0,+1} 更像 STATE/SignedEvent。
不要仅仅把它当连续 Direct Alpha。
保留：
ema_crossover -> STATE/SIGNED_EVENT

新增真正连续终端：
ema_cross_spread_pct
ema_cross_spread_atr
ema_cross_strength
ema_cross_age
ema_cross_persistence

---------------------------------------------------------------------
2.3 Keltner 第二轮
---------------------------------------------------------------------

现有：
KeltnerPosition
keltner_width_pct
keltner_compression
keltner_breakout_strength

继续增加：
keltner_width_z
keltner_width_percentile
keltner_expansion

keltner_upper_distance_pct
keltner_lower_distance_pct
keltner_upper_distance_atr
keltner_lower_distance_atr

keltner_breakout_age
keltner_breakout_persistence
keltner_reentry_age
keltner_failed_breakout_strength
keltner_boundary_dwell
keltner_squeeze_release

---------------------------------------------------------------------
2.4 Donchian 第二轮
---------------------------------------------------------------------

现有：
donchian_width_pct
donchian_channel_position
donchian_breakout_up/down

继续增加：
donchian_width_z
donchian_width_percentile
donchian_compression
donchian_expansion

donchian_breakout_age
donchian_breakout_persistence
donchian_retest_age
donchian_retest_strength

donchian_failed_breakout_strength
donchian_false_break_density

donchian_upper_distance_pct
donchian_lower_distance_pct

不要重复已有数学等价 channel canonical。

---------------------------------------------------------------------
2.5 PSAR 第二轮
---------------------------------------------------------------------

现有：
psar_direction
psar_distance_pct
psar_flip
psar_days_since_flip

继续：
psar_distance_atr
psar_distance_z

psar_flip_density
psar_flip_rate_change

psar_state_duration_pct
psar_direction_persistence

psar_trend_confirmation
psar_price_divergence
psar_reversal_pressure

psar_flip 是 SignedEvent，不应仅仅作为 ordinary continuous alpha。

---------------------------------------------------------------------
2.6 Supertrend 第二轮
---------------------------------------------------------------------

现有：
supertrend_direction
supertrend_distance_pct
supertrend_flip
supertrend_days_since_flip

继续：
supertrend_distance_atr
supertrend_distance_z

supertrend_flip_density
supertrend_state_duration_pct
supertrend_direction_persistence

supertrend_confirmation_strength

supertrend_failed_flip
supertrend_reentry_strength
supertrend_break_strength

---------------------------------------------------------------------
2.7 Ichimoku 第二轮
---------------------------------------------------------------------

继续形成完整 Direct Alpha family：

ichimoku_tenkan_kijun_spread_pct
ichimoku_tenkan_kijun_spread_atr

ichimoku_tk_cross_event
ichimoku_tk_cross_age
ichimoku_tk_cross_strength

ichimoku_cloud_thickness_pct
ichimoku_cloud_thickness_atr
ichimoku_cloud_thickness_z

ichimoku_cloud_distance_pct
ichimoku_cloud_distance_atr

ichimoku_cloud_twist_event
ichimoku_cloud_twist_age
ichimoku_cloud_twist_strength

ichimoku_cloud_breakout_strength
ichimoku_cloud_breakout_age
ichimoku_cloud_state_duration

硬约束：
Senkou 的“向未来绘图 displacement”绝不能移动因子 timestamp。
t 的因子值只能由 <=t 可知数据生成。
display offset 与 information timestamp 必须完全分离。

---------------------------------------------------------------------
2.8 Rolling VWAP 第二轮
---------------------------------------------------------------------

继续：
rolling_vwap_distance_pct
rolling_vwap_distance_atr
rolling_vwap_distance_z
rolling_vwap_percentile

rolling_vwap_slope
rolling_vwap_acceleration

vwap_cross_event
vwap_cross_age

vwap_reclaim_strength
vwap_rejection_strength

price_vwap_divergence
volume_vwap_confirmation

---------------------------------------------------------------------
2.9 Bollinger / band-level family
---------------------------------------------------------------------

如果 raw upper/mid/lower 仍在 intermediate：
不要直接 promotion。

保证有：
bollinger_pct_b
bollinger_width

再补：
bollinger_width_z
bollinger_width_percentile
bollinger_compression
bollinger_expansion

bollinger_upper_distance
bollinger_lower_distance

bollinger_breakout_strength
bollinger_breakout_age
bollinger_reentry_strength
bollinger_failed_breakout

bollinger_mid_distance_pct
bollinger_mid_slope

======================================================================
三、真正修 Structural Support / Resistance / Pivot，而不是只绕开
======================================================================

当前新增 sr_distance_pct / sr_touch_count 使用 prior median typical-price pivot，
这本身合理，但并没有真正解决原：
ts_support_level
ts_resistance_level
ts_last_pivot_*
ts_nth_pivot_*
等结构型 pivot confirmation 的问题。

必须给 structural pivot 引入“双时间”：

event_time
= pivot 实际发生的 bar

available_at
= 经过 right_window 确认后市场参与者真正知道它是 pivot 的时间

例如：
pivot 在 t
right_window=3
则最早 available_at=t+3

绝对禁止把 t+3 才确认的信息回填到 t 用来交易。

在此基础上增加：

causal_pivot_high
causal_pivot_low

causal_support_level
causal_resistance_level

distance_to_support_pct
distance_to_resistance_pct
nt_cluster_duration
event_interarrival_cv
event_interarrival_z
event_post_return_past
event_post_volume_response
event_post_vol_response

SignedEvent：
signed_event_rate
positive_event_rate
negative_event_rate

positive_event_age
negative_event_age

signed_event_decay
event_direction_imbalance
event_direction_persistence
event_flip_density

CategoricalEvent：
category_age
category_frequency
category_transition_rate
category_transition_surprise

特别注意：
event_recency_z 需要 >=3 个事件，因此对稀疏事件经常全 NaN。
必须额外有简单稳健的 event_age / event_decay，
不能依赖 recency_z 作为唯一 recency 表达。

======================================================================
六、State → Direct Alpha 完整化
======================================================================

已有：
state_dwell_pct
state_transition_surprise

继续：
state_age
state_episode_age
state_episode_duration

state_persistence
state_transition_count
state_transition_rate

state_flip_age
state_flip_density

state_entropy
state_instability

state_confidence
state_change_magnitude

state_duration_z
state_duration_percentile

state_conditional_return_prior
state_conditional_vol_prior

注意：
state_dwell_pct != 当前 episode age。
两者都需要保留，不能混成一个定义。

======================================================================
七、Global State → stock-specific Alpha
======================================================================

全市场每天同一个值不能直接横截面挖因子。

例如：
market breadth
market volatility regime
market liquidity regime
index state
macro state

必须通过股票 prior exposure 转成 stock-specific signal：

market_state_beta_prior
market_state_sensitivity

market_state_residual
market_state_residual_z

market_state_divergence

market_state_sensitivity_change
market_state_beta_break

market_state_conditional_return_prior
market_state_conditional_vol_prior

stock_breadth_divergence
stock_breadth_sensitivity
breadth_beta_break

regime_exposure
regime_exposure_change

所有 exposure/beta 估计必须严格 <=t-1。

======================================================================
八、Group State → stock-relative Alpha
======================================================================

行业/group aggregate 本身同组股票一样，不能直接 terminal。

继续增加：

group_ex_self_gap
group_ex_self_z
group_ex_self_rank
group_ex_self_mad_z

within_group_rank_pct
within_group_dispersion_position

group_beta_prior
group_residual_z
group_divergence

relative_strength_group_pct
relative_strength_group_z

group_leader_laggard_score
group_rotation_sensitivity

group_state_conditional_return_prior

group_dispersion_exposure
group_dispersion_divergence

避免和现有 ex_self_* exact duplicate。
每次新增前先做 semantic duplicate audit。

======================================================================
九、Regression / Model-style Research Tool 完整 causalization
======================================================================

原则：

IN-SAMPLE fit / residual 不要直接 promotion。

统一转换成：
strict-prior fit
forecast
forecast error
forecast error z
coefficient stability
parameter break

---------------------------------------------------------------------
9.1 AR
---------------------------------------------------------------------

保留 research：
ts_ar_fitted_value
ts_ar_in_sample_resid
legacy ts_ar_forecast / innovation

production mining 使用：
ts_ar_prior_forecast
ts_ar_prior_innovation
ts_ar_prior_innovation_z
ts_ar_prior_coeff
ts_ar_coeff_stability

继续补：
ts_ar_coeff_change
ts_ar_regime_break
ts_ar_predictability_prior
ts_ar_forecast_surprise

---------------------------------------------------------------------
9.2 Ridge
---------------------------------------------------------------------

完整补：
ts_ridge_regression_coeff_prior
ts_ridge_intercept_prior
ts_ridge_r2_prior

ts_ridge_forecast
ts_ridge_forecast_error
ts_ridge_forecast_error_z

ts_ridge_coeff_stability
ts_ridge_coeff_change

ts_ridge_vs_ols_beta_gap
ts_ridge_shrinkage_effect

generic ridge ABI 必须唯一：
y, x / predictors, window, alpha, fit_intercept, min_periods

禁止同一个 canonical 同时拥有 trend-ridge 和 y~x ridge 两种 ABI。

---------------------------------------------------------------------
9.3 Quantile Regression
---------------------------------------------------------------------

这是 P0/P1 高价值 family：

ts_quantile_beta_prior
ts_quantile_intercept_prior

ts_quantile_forecast
ts_quantile_forecast_error
ts_quantile_forecast_error_z

ts_quantile_coeff_stability
ts_quantile_coeff_change

ts_quantile_beta_spread_prior
= beta(q_high)-beta(q_low)

ts_quantile_beta_curvature

ts_quantile_up_down_asymmetry

它后面可以直接服务 Distributional Model / tail alpha。

---------------------------------------------------------------------
9.4 Expectile Regression
---------------------------------------------------------------------

补：
expectile_coeff_prior
expectile_forecast_error
expectile_forecast_error_z

expectile_upper_lower_beta_spread
expectile_tail_asymmetry
expectile_coeff_stability
expectile_coeff_break

---------------------------------------------------------------------
9.5 Huber
---------------------------------------------------------------------

补：
huber_coeff_stability
huber_coeff_change
huber_forecast_surprise

huber_vs_ols_beta_gap
huber_vs_ols_residual_gap
outlier_pressure

Huber vs OLS 的差本身可以表示近期关系是否被极端样本驱动。

---------------------------------------------------------------------
9.6 Multi Regression
---------------------------------------------------------------------

继续：
multi_regression_r2_change
multi_regression_coeff_break
multi_regression_model_decay
multi_regression_forecast_surprise

---------------------------------------------------------------------
9.7 Polynomial
---------------------------------------------------------------------

继续：
poly2_curvature_prior
poly2_curvature_change
poly2_linear_vs_quadratic_gain
poly2_forecast_break

======================================================================
十、Mean Reversion / OU family 必须真正解决
======================================================================

当前：
ts_mean_reversion_half_life
仍是 experimental/research，且要求 stationary/spread/residual input。

不要把 raw price half-life 直接 promotion。

增加 strict-prior family：

ts_mean_reversion_half_life_prior

ts_ou_phi_prior
ts_ou_kappa_prior
ts_ou_equilibrium_prior

ts_ou_equilibrium_distance
ts_ou_equilibrium_distance_z

ts_ou_speed_change
ts_ou_half_life_change

ts_mean_reversion_strength_prior
ts_mean_reversion_break_score

输入类型必须显式标识：
SPREAD
RESIDUAL
STATIONARY_SERIES

不要只靠 runtime warning 猜 raw price 是否 trending。

======================================================================
十一、Rolling statistical intermediate 的 Rehabilitation
======================================================================

很多数学算子本身可以 composition，但裸 terminal 经济语义不够。

不要把以下 raw levels 全部 promotion：
ts_mean
ts_median
ts_min
ts_max
ts_std(price)
ts_var(price)
ts_cov
rolling quantile level
raw regression slope with units

生成可用 sibling：

rolling_level_distance_pct
rolling_level_distance_z

rolling_position
= (x-min)/(max-min)

rolling_range_pct
rolling_range_z

rolling_quantile_position
rolling_interquantile_range_pct
rolling_quantile_asymmetry

covariance_normalized
= cov/(std_x*std_y)
即 correlation

standardized_beta
beta_tstat

variance_ratio
short_long_vol_ratio

argmax/argmin：
不要只返回 raw index；
重点输出：
argmax_age
argmin_age
distance_to_recent_high
distance_to_recent_low
extreme_age_spread

======================================================================
十二、PCA / Latent Structure Rehabilitation
======================================================================

当前这块仍未完整解决。

rolling PCA eigenvector 有 ± sign ambiguity。

严禁裸 signed loading 因 sign flip 产生假信号。

至少增加：

pca_loading_abs
pca_loading_squared
pca_loading_rank

pca_loading_stability
pca_loading_change_abs

pca_loading_concentration

pca_subspace_angle
pca_subspace_instability

pca_loading_regime_shift

如果确实要 signed loading：
必须 deterministic sign anchoring：
dot(v_t, v_{t-1}) < 0 时 v_t *= -1

然后才允许：
pca_loading_signed_stable

Reconstruction family：
pca_reconstruction_error
pca_reconstruction_error_z
pca_reconstruction_error_rank
pca_reconstruction_error_change
pca_anomaly_score

======================================================================
十三、Spectral / Wavelet / DMD 第二轮
======================================================================

已有第一轮 ratio。

继续增加：

spectral_low_high_ratio
spectral_entropy

spectral_centroid
spectral_centroid_change

spectral_low_frequency_excess_prior
spectral_regime_shift

wavelet_high_low_energy_ratio
wavelet_energy_imbalance
wavelet_frequency_shift

wavelet_lowpass_residual
wavelet_lowpass_residual_z

wavelet_reconstruction_ratio

DMD 必须使用 typed canonical：
ts_dmd_level_*
ts_dmd_return_*

不要复活 untyped alias。

可新增：
dmd_frequency_shift
dmd_growth_shift
dmd_mode_stability
dmd_reconstruction_error
dmd_regime_break

======================================================================
十四、Intraday → Daily Rehabilitation
==========================

当前：
sign(delta_close) * volume

严格说更像 sign-volume classification proxy，
不应不加说明地宣称为完整 probabilistic BVC。

如果继续用 bar-sign estimator：
建议命名诚实：
sign_volume_imbalance_pct
approx_vpin_sign

如果 DataAccess 有真正 minute data：
实现：
intra_bvc_buy_volume_share
intra_bvc_sell_volume_share

intra_bvc_signed_imbalance

intra_vpin
intra_vpin_z
intra_vpin_percentile

intra_vpin_change
intra_vpin_acceleration
intra_vpin_persistence
intra_vpin_tail

equal-volume bucket 必须跨完整 session 正确处理 bucket 边界，
禁止跨交易日连接一个 bucket。

======================================================================
十六、Fundamental / Fiscal Level → Direct Alpha
======================================================================

凡是 raw accounting level 不要直接 terminal。

重点形成：

YoY growth
QoQ growth
TTM growth

growth acceleration
growth stability

margin
margin change
margin zscore

ROA / ROE / ROIC
return-on-capital change

cash conversion
OCF / NI
FCF / NI

working-capital accrual
balance-sheet accrual

asset growth
inventory growth
receivable growth
payable growth

sales vs inventory divergence
sales vs receivable divergence

earnings quality
revenue quality

fundamental trend slope
fundamental trend tstat
fundamental trend r2

standardized surprise
revision surprise

days_since_report
days_since_revision
fundamental_staleness

任何季度/TTM 算子：
必须使用真正的 available_at / announcement timestamp / revision vintage，
不能按 fiscal period end 当可用日。

DATA 不足的：
holder_count_change_rate
holder_concentration_change
fin_total_operating_accruals
等继续 DATA_GATED / DELETE_NO_DATA；
禁止使用错误 provider 代理冒充。

======================================================================
十七、Index / Holdings / Corporate Event Rehabilitation
======================================================================

index_member 本身是 STATE。

增加：
index_entry_age
index_exit_age
index_weight_change
index_weight_z

index_rebalance_pressure
index_entry_exit_density

membership_duration
membership_stability

holder snapshots 有真实数据后：
holder_hhi
holder_hhi_change

top1/top5/top10 concentration
concentration_change

holder_entry_intensity
holder_exit_intensity
holder_churn
holder_persistence

corporate action：
split/dividend/buyback/issuance/unlock 等不要只有 event，
增加：
event_age
event_size_pct
event_decay
event_cluster
post_event_volume_response
post_event_vol_response

======================================================================
十八、Raw math primitive 永远不要硬 promotion
======================================================================

以下保持 internal/intermediate：

constant
identity
arg
protected_div

raw sin/cos/tan
sinh/cosh
sec/csc/cot

coalesce
ffill
source transforms

raw ceil/floor/round/fix

generic:
square
power
sqrt_abs
flex_min
flex_max

但可以作为 composition primitive。

phase 类如需变成 alpha，要新增：
phase_angle
phase_velocity
phase_acceleration
phase_reversal
phase_coherence
phase_dispersion

离散/bucket：
bucket_transition
bucket_age
bucket_persistence
bucket_transition_rate

而不是把 floor/round 本身当 alpha。

======================================================================
十九、Backend 架构：先修“统计口径 split-brain”
======================================================================

当前仓库对 backend coverage 至少出现三套互相冲突数字：

backend/README.md：
Polars 325
SQL 58

docs/sql_pushdown_coverage.md：
SQL emitter 354
DuckDB parity 0
DuckDB production safe 0

evidence/r2/BACKEND_GAP_MATRIX.yaml：
pandas 1362
polars 1361
sql 375
backend_passed 0
implementation_certified 0

必须删除人工维护 backend count 的权威地位。

建立唯一 MACHINE-GENERATED：

PHYSICAL_IMPLEMENTATION_MATRIX

每行：
canonical
registry_slot
physical_backend

implementation_id

execution_kind

implemented
selectable

spec_complete

oracle_passed
edge_passed

parity_passed

production_evidence_passed

production_admitted

source_hash
semantic_hash
parameter_domain_hash

current_head_sha

所有 README / docs 数量必须从该矩阵生成。

======================================================================
二十、Pandas / NumPy backend
======================================================================

定位：
REFERENCE BACKEND / correctness oracle candidate

不是主要性能 backend。

目标：
所有 production canonical 至少有一个稳定 Pandas/NumPy reference。

但 reference 不能自动等同数学真值。

对于：
regression
tail
entropy
spectral
robust statistics
PCA
VPIN
必须再有 independent mathematical oracle。

不要为了性能在 Pandas reference 路径做复杂隐式 shortcut。

======================================================================
二十一、Polars backend：P0 主力补全
======================================================================

当前 Polars 应成为日频大部分 TS/CS/group 的主要内存列式 backend。

优先补真正 native：

A. elementwise
add/sub/mul/div
abs/sign/log/exp
clip/where
safe-div

B. simple rolling
ts_sum
ts_mean
ts_min
ts_max
ts_std
ts_var
ts_median
ts_quantile

lag
delta
pct_change

ts_argmax
ts_argmin

ts_count_if
ts_sum_if
ts_mean_if

C. cross-sectional
cs_rank
cs_percentile
cs_demean
cs_zscore
cs_mad_z

D. group
group_rank
group_percentile
group_demean
group_zscore
group_mean
group_std

E. rehab simple families
atr_pct
true_range_pct
MA distance
Donchian width/position
Keltner width/position
candle ratios
consolidation
VWAP distance
event-rate/state simple rolling

但硬性要求：

禁止：
Python per-row callback
rolling_map Python callback
map_elements Python
Pandas fallback

然后还标记 POLARS_NATIVE。

如果使用 NumPy kernel：
execution_kind 必须明确：
POLARS_NUMPY_KERNEL
而不是 POLARS_NATIVE_EXPR。

---------------------------------------------------------------------

P0：重新审计 ts_ewm_corr / ts_ewm_cov

当前 Polars 大致：
x_mean_t = EWM(x)
residual_i = x_i - EWMMean_i
再：
EWM(residual_x * residual_y)

这不应未经证明地认定等于 canonical EW covariance。

必须用标准递推或与 Pandas ewm.cov/corr 严格定义对齐。

明确：
alpha/span
adjust
bias
min_periods
ignore_nulls

pairwise finite mask
NaN
Inf

warmup

然后 property test：
ewm_corr(x,x)=1 on valid nonzero-variance period
ewm_cov(x,x)=ewm_var(x)
translation invariance
scale transform

所有新 R20 DirectAlpha：
至少判断：
适合 native Polars 的就补 PhysicalImplementationSpec；
不适合的不要为了覆盖率硬塞。

======================================================================
二十二、DuckDB SQL backend：这是当前最高 ROI 后端补强之一
======================================================================

当前已经有大量 emitter，但 DuckDB parity verified=0 / production safe=0。

下一阶段不要继续只追 emitter 数量。

第一目标：
让第一批真正成为 PRODUCTION_SAFE。

优先认证：

Tier 1:
column/literal
add/sub/mul/div
abs/sign/log/exp
clip
where

lag
delta
return

ts_sum
ts_mean
ts_min
ts_max
ts_count
ts_std
ts_var

ts_rank/quantile（先固定 tie/NULL semantics）

CS:
cs_rank
cs_percentile
cs_mean
cs_std
cs_demean
cs_zscore

Group:
group_mean
group_std
group_rank
group_zscore
group_demean

Tier 2:
ts_corr
ts_cov
simple beta/regression

volatility family

Donchian
Bollinger
simple candle
VWAP

Tier 3:
Fundamental/PIT
这是 DuckDB ROI 最高的部分：

DataAccess Parquet scan
→ PIT as-of
→ fiscal/revision filter
→ universe join
→ group/industry join
→ factor calculation

尽量保持 DuckDB resident，不在每个 operator 后转 Pandas。

必须单独有：
DuckDB semantic certification
ClickHouse semantic certification

严禁：
DuckDB PASS 自动等于 ClickHouse PASS。

======================================================================
二十三、ClickHouse backend
======================================================================

只在：
真实数据 resident in ClickHouse
大表扫描/聚合/窗口 pushdown
确实有明显数据移动收益

时重点扩。

第一阶段只认证：
elementwise
lag
simple rolling
CS/group aggregations
PIT-compatible joins
simple fundamentals

ClickHouse 不继承 DuckDB certificate。

timestamp/null/window/frame/rank tie semantics 都独立测试。

======================================================================
二十四、Numba：建议增加为 Physical Kernel 层，不要盲目增加一个空壳 backend
======================================================================

现在 backend README 明确说旧 numba_kernels 已不是主路径。

但大量算子非常适合 Numba：

PSAR
Supertrend
KAMA
Wilder recursive family
ADX/RSI Wilder

QQE
RSX

Kalman
alpha-beta filter

directional change

state machines

event age/duration
VPIN volume buckets

robust rolling statistics

entropy
first passage
motif
fractal

高级 regression rolling loops

不要简单恢复旧 numba_kernels.py。

建立新的 physical implementation identity。

推荐二选一：

方案 A：
新增 ExecutionKind.NUMBA_CPU_KERNEL，
Numba 是 CPU physical kernel，
不单独做 user-facing Backend。

方案 B：
如果 Planner 确实需要在 Numba 与 Polars/DuckDB 之间做 per-region cost assignment，
再正式增加 backend：
numba_cpu

但不能半套。

如果增加 numba长期 resident
CPU↔GPU transfer 占比可控

======================================================================
二十七、Arrow 应作为 interchange contract，不是“又一个 backend”
======================================================================

规范 region boundary：

Arrow Table / RecordBatch
NumPy contiguous buffer
Polars Arrow zero-copy
DuckDB Arrow scan

尽量避免：
Pandas stack/unstack
object dtype
Python row object
pickle

每个 RegionPlan 记录：
input bytes
output bytes
conversion bytes
materialization count
transfer time

======================================================================
二十八、Backend Planner 必须以 PhysicalImplementation 为选择单位
======================================================================

不能只看：
canonical × backend

必须看：
PhysicalImplementationID

ID 至少绑定：

canonical
backend
execution_kind

implementation source hash
emitter identity

parameter-domain hash
semantic-contract hash

runtime version family
dialect

production evidence hash

同一个 canonical 的：
Pandas reference
Polars native
Numba kernel
DuckDB SQL
ClickHouse SQL
q lowering

是六个不同 physical implementations，
不能共用一个 certification。

======================================================================
二十九、所有 Rehabilitation 算子都要经过统一 Certification
======================================================================

每个新 Direct Alpha 至少：

1. Semantic
明确：
input semantic
output unit
scale sensitivity
causality
window semantics
warmup
min_periods
missing policy

2. ParamSpec
bool-as-int reject
float-as-int reject
NaN/Inf reject
min/default/max
relational parameter constraints

3. PIT
future poison
prefix invariance

4. Numeric
NaN
±Inf
zero denominator
constant panel
ties
extreme magnitude
1e9 level + tiny dispersion

5. Economic invariance
该 scale-invariant 的测试：
x * 10 后输出相同

rank 类：
monotonic transform invariance

CS：
asset permutation equivariance

6. Independent oracle

7. Backend parity

8. Production evidence

然后才：
production_certified=True
production_admitted=True

======================================================================
三十、特别建立 Operator Rehabilitation Rule Engine
======================================================================

不要以后再一个个人工发明。

正式定义 SemanticType → allowed derivations：

PRICE_LEVEL:
pct_distance
atr_distance
zscore
percentile
slope
acceleration
curvature

BAND_LEVEL:
position
width
distance
compression
expansion
breakout
retest

EVENT_BOOL:
rate
age
streak
decay
cluster
past_response

SIGNED_EVENT:
positive_rate
negative_rate
signed_decay
direction_imbalance
flip_density

STATE:
age
duration
persistence
transition
entropy
confidence

GLOBAL_STATE:
prior_beta
sensitivity
residual
divergence
conditional_signal

GROUP_STATE:
ex_self_gap
ex_self_z
within_group_rank
group_beta
relative_strength

IN_SAMPLE_MODEL:
prior_fit
forecast
forecast_error
forecast_error_z
coeff_stability

SPECTRAL_COMPONENT:
ratio
excess
change
residual
regime_shift

LATENT_LOADING:
abs
square
rank
sign_stability
subspace_angle

SOURCE_UPDATE:
staleness
days_since_update
revision_count
revision_magnitude

然后 FactorEngine 可以自动提示：
raw intermediate X 存在哪些 legal mineable siblings。

但不要自动生成重复 canonical；
生成前跑 semantic redundancy/dedup。

======================================================================
三十一、重新生成 Current-HEAD Truth
======================================================================

完成上述阶段后，必须从 current runtime 自动生成：

A. DIRECT_USE_MATRIX
每个 canonical：
DirectUseStatus
mining_visible
composition_usable
terminal_usable
production_admitted
directly_usable

B. REHABILITATION_MATRIX
原算子
问题类型
new sibling
replacement
status

C. PHYSICAL_IMPLEMENTATION_MATRIX
见前文。

D. BACKEND_GAP_MATRIX
Pandas
Polars
Numba
DuckDB
ClickHouse
q

不要再使用旧 R18：
1528
1197
62
29

作为 current truth。

不要再允许：
backend README 一套数字
SQL coverage 一套数字
evidence matrix 又一套数字。

======================================================================
三十二、执行优先级
======================================================================

P0：
1. 修 beta residual 漏 intercept + 错误 oracle
2. 修/认证 Polars EWM corr/cov
3. 重新生成 current DirectUse / backend truth matrix
4. 给 R20 第一轮已有 Rehabilitation 算子跑 production certification
5. 先认证一批 Polars production physical implementation
6. 先认证一批 DuckDB production physical implementation
7. EventBool/SignedEvent/CategoricalEvent typed contract
8. Event age/decay + State episode-age
9. Support/Resistance available_at causal rebuild
10. Regression Ridge/Quantile/OU causal family

P1：
11. MA/ATR/Keltner/Donchian/PSAR/Supertrend/Ichimoku/VWAP 第二轮
12. Global/Group State stock-specific family
13. PCA/latent rehabilitation
14. Intraday→Daily
15. Fundamental/fiscal rehabilitation
16. new Numba physical kernel layer
17. spectral/wavelet/DMD second round

P2：
18. ClickHouse wider certification
19. q core certification
20. phase/bucket advanced derivations

当前不要：
为了数字增加无意义 Direct Alpha
为了覆盖率硬塞 q
为了“多 backend”仓促上 GPU
把 internal math primitive promotion
复制完全等价 canonical
用 proxy data 冒充缺失字段

======================================================================
三十三、开发工作方式
======================================================================

建议并行角色：

Writer-A：
Operator Rehabilitation

Writer-B：
Polars + Numba physical implementations

Writer-C：
DuckDB/SQL physical implementations

Math Red-Team：
只读，独立推公式和反例，禁止复用 implementation helper

Certification Reviewer：
只负责 current SHA evidence / physical ID / parity / production gate

如果多个 Writer 会修改：
indicators_v2.py
operator_surface.py
mining/direct_use.py

禁止同时无锁写同一文件。

使用 source write lease：
同一 authority file 同时仅一个 writer。

每一个 atomic slice：

reproduce/failing test
→ implementation
→ focused test
→ independent oracle
→ prefix/future poison
→ backend parity
→ independent review
→ integration
→ regenerate evidence

不得出现：
Agent failed/cancelled 就假定没改代码。
每个失败 Agent 后检查 worktree/source diff。

关键 source 文件增加：
exported symbol inventory
AST parse
minimum structural integrity
防止再次出现 q_executor 类似 partial overwrite/truncation。

======================================================================
三十四、最终验收标准
======================================================================

最终不要只告诉我：
“新增了 N 个算子”。

必须输出：

current HEAD SHA

当前 canonical 总数

当前：
direct_alpha
direct_alpha_high_cost
intermediate
research
event
state
global_state
group_state
internal/delete

本轮：
多少 raw intermediate 得到 mineable sibling
多少 research operator 得到 causal replacement
多少 event/state 得到 numeric Direct Alpha
多少 duplicate 被跳过
多少 data-gated

以及：

Pandas:
implemented / certified

Polars:
implemented / parity / production-certified

Numba:
implemented / parity / production-certified

DuckDB:
implemented / parity / production-certified

ClickHouse:
implemented / parity / production-certified

q:
implemented / runtime-tested / production-certified

最后必须给：

PRODUCTION_ADMITTED_DIRECT_MINING_COUNT

而不是只给 terminal_usable 数量。

任何 gate 未运行：
明确 NOT_RUN。

只有 current HEAD + exact PhysicalImplementationID + current evidence
全部匹配，才允许写 CERTIFIED。
