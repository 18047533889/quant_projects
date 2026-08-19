# FactorEngine 滤波器家族扩展方案 V2
## —— 在 RobustEMA / KAMA / Hampel / Deadband / Slew 之外补齐真正不同机制

> 本文件只补充上一份滤波专项没有充分展开的“不同滤波器家族”。  
> 不重复创建已有/已建议 canonical；开工前必须重新检查最新 main 的 registry 与 alias。  
> 目标仍是：量价/因子/模型分数去毛刺、保留真实变化、降低无意义 rank 抖动与换手。

---

# 1. 现有/已讨论能力不要重复造轮子

当前/前序已包括：

```text
ts_ema
state_deadband
state_slew_limit
state_latch
state_hold
state_ewm_if
event_refractory
hysteresis family

Kalman
Haar low-pass
HMA
ALMA
RSX
QQE

已建议：
ts_hampel_filter_causal
ts_robust_ema
ts_kama
ts_super_smoother
state_adaptive_deadband
state_adaptive_slew_limit
state_rank_deadband
state_quantile_hysteresis
ts_one_euro_filter
ts_alpha_beta_filter
ts_robust_kalman_level
```

本轮扩展应优先补“机制真正不同”的家族。

---

# 2. 总体分类

建议 Filter Layer 最终覆盖：

```text
A. Robust Spike Filters
B. Classical Causal Digital Low-pass
C. Edge/Jump-Preserving Trend Filters
D. Online Adaptive Filters
E. State-Space / Robust State Filters
F. Multiscale / Frequency Filters
G. Change-Point-Aware Filters
H. Cross-Sectional Shrinkage Filters
I. Turnover / Cost-Aware Filters
J. Confidence / Liquidity-Aware Filters
```

---

# 3. A 类：Robust Spike Filters

## A1. `ts_median3_causal`

```text
median(x_t, x_{t-1}, x_{t-2})
```

专门处理单点孤立 spike。

优点：

```text
O(T)
极快
无参数或参数很少
对孤立毛刺极有效
```

缺点：

```text
对真实单日 jump 也可能衰减
```

因此定位：

```text
light despike
```

而不是通用趋势 filter。

---

## A2. `ts_rolling_median_causal`

短窗口 rolling median：

```text
3 / 5 / 7
```

比 mean 更抗 outlier。

不建议 window 很大，否则 lag 明显。

---

## A3. `ts_trimmed_mean_causal`

窗口内删除上下尾部若干比例，再 mean。

例如：

```text
trim = 10% / 20%
```

相较 median：

```text
保留更多幅度信息
```

适合 volume/turnover/microstructure 类厚尾信号。

---

## A4. `ts_winsorized_mean_causal`

窗口内使用历史分位点 winsorize 后再 mean。

必须严格 trailing/PIT。

---

## A5. `ts_tukey_biweight_filter`

使用 redescending M-estimator。

相较 Huber/clip：

```text
极端异常点权重可以下降到接近 0
```

风险：

```text
容易把真实 regime jump 当成异常
```

建议 Research/Extended，不作为默认主链。

---

# 4. B 类：Classical Causal Digital Low-pass

这是目前 FactorEngine 明显可以补的一整类。

## B1. `ts_butterworth_lowpass_causal`

使用 causal IIR Butterworth。

参数建议：

```text
cutoff_period
order = 2 / 3 / 4
```

不要暴露任意 filter coefficient。

生产实现应采用：

```text
SOS second-order sections
recursive state
checkpoint
```

禁止使用 forward-backward zero-phase 版本，因为历史回测中 forward-backward 会使用未来数据。

---

## B2. `ts_bessel_lowpass_causal`

Bessel 低通的价值不在“更强平滑”，而在：

```text
更好的时域响应 / 更小的形状失真
```

适合希望保留 trend/event shape 的 signal。

建议 Research→Extended 后认证。

---

## B3. `ts_fir_lowpass_causal`

有限冲击响应 FIR：

```text
固定 trailing coefficients
```

优点：

```text
没有递归数值漂移
state 只需要最近 L-1 个输入
很好做跨后端 parity
```

缺点：

```text
同等平滑强度下 lag 往往更大
```

适合低频日因子而不是快速因子。

---

## B4. `ts_kaiser_fir_lowpass_causal`

如果需要更严格控制：

```text
passband / stopband
```

可用 Kaiser-window FIR 作为 Research filter。

不要让 FactorMiner 搜任意 taps。

---

# 5. C 类：Edge / Jump-Preserving Trend Filters

普通低通最大的缺点：

```text
真实 step/jump 也被抹平
```

这一类专门解决“平滑但保边”。

## C1. `ts_l1_trend_filter_trailing`

在 trailing window 求：

\[
\min_z
\frac12 \|x-z\|_2^2
+
\lambda \|D^{(2)}z\|_1
\]

输出当前 endpoint。

特点：

```text
趋势段平滑
允许 slope 发生稀疏变化
比普通均线更保留转折
```

注意：

```text
只能 trailing window
不能 full-history 双向重估后回填过去
```

建议 Research/Extended。

---

## C2. `ts_total_variation_filter_trailing`

TV denoising：

\[
\min_z
\frac12 \|x-z\|_2^2
+
\lambda \|Dz\|_1
\]

更倾向：

```text
piecewise constant
```

特别适合：

```text
状态/level 信号
```

不适合所有连续 alpha。

---

## C3. `ts_causal_local_linear_smoother`

只用 trailing window 做 local-linear fit，取 endpoint fitted value。

比普通 moving average：

```text
对斜率趋势的滞后更小
```

应明确：

```text
one-sided local polynomial
```

不是 centered LOESS。

---

## C4. `ts_causal_savgol_endpoint`

Savitzky-Golay 思路但只用 trailing points，对 endpoint 做多项式拟合。

严格禁止 centered SG 进入 production。

参数：

```text
window
polyorder
```

小网格认证。

---

# 6. D 类：Online Adaptive Filters

## D1. `ts_vidya`

Volatility Index Dynamic Average。

核心：

```text
根据近期趋势/波动状态动态调整 alpha
```

与 KAMA 不同机制，可作为另一种 adaptive low-pass。

---

## D2. `ts_mcginley_dynamic`

通过当前价格/信号相对上一状态的比率自适应调整更新速度。

适合：

```text
趋势速度变化明显的价格/信号
```

建议 Extended。

---

## D3. `ts_variable_forgetting_ema`

不是固定 half-life，而是：

\[
\alpha_t=f(noise_t, trend_t)
\]

建议只提供少数已定义 policy：

```text
volatility_adaptive
efficiency_adaptive
confidence_adaptive
```

不能让用户直接传任意函数。

---

## D4. `ts_nlms_filter`

Normalized LMS 在线自适应滤波。

可用于：

```text
从 noisy signal 中在线学习 short-memory predictor
```

然后输出：

```text
predicted level
innovation
filtered level
```

适合研究价格/成交量局部线性结构。

需要严格限制：

```text
tap length
learning rate
```

---

## D5. `ts_rls_filter`

Recursive Least Squares。

相较 NLMS：

```text
收敛更快
可以使用 forgetting factor
```

但复杂度和 state 更高。

适合：

```text
局部动态回归型 filter
```

不默认进入 AlphaMiner。

---

# 7. E 类：State-Space / Robust State Filters

## E1. `ts_h_infinity_level_filter`

H∞ filter 的定位：

```text
对模型误设/不确定性比标准 Kalman 更保守
```

适合：

```text
measurement/model uncertainty 较强
```

Research only 起步。

---

## E2. `ts_alpha_beta_gamma_filter`

现有 alpha-beta filter 可以进一步扩到：

```text
level
velocity
acceleration
```

适合：

```text
趋势加速度类 signal
```

但日频金融数据容易过拟合，默认 Research。

---

## E3. `ts_student_t_kalman_filter`

把 observation noise 从 Gaussian 改成厚尾/robust update。

实际工程可先实现：

```text
Student-t inspired innovation weight
```

避免异常冲击过度改变 state。

比单纯 clip 更连续。

---

## E4. `ts_adaptive_noise_kalman`

q/r 不作为固定搜索参数，而由严格过去 innovation statistics 在线缓慢更新。

需要：

```text
NoiseAdaptationContract
```

防止重新引入尺度/未来函数问题。

---

## E5. Particle Filter

可以支持：

```text
nonlinear/non-Gaussian state
```

但：

```text
成本高
随机性高
参数多
```

只建议 Research，不进入默认 FactorEngine 搜索。

---

# 8. F 类：Multiscale / Frequency Filters

## F1. `ts_wavelet_shrinkage_trailing`

不同于当前 low-pass reconstruction：

```text
不是简单删除高频层
```

而是：

```text
对 wavelet detail coefficients 做 threshold/shrinkage
```

可保留部分高频真实结构。

需要严格 trailing window。

---

## F2. `ts_modwt_denoise_trailing`

MODWT 多尺度分解：

```text
保留指定尺度
抑制最细噪声尺度
```

Research。

要明确：

```text
rolling decomposition 会随窗口变化
```

不能把历史 decomposition 当 immutable state。

---

## F3. `ts_ssa_denoise_trailing`

利用 trailing Hankel/SVD：

```text
保留主要 singular components
```

输出当前 endpoint reconstruction。

当前已有 SSA/Hankel family 时优先复用 shared state，不要再造独立 SVD。

Research/Extended。

---

## F4. `ts_spectral_lowpass_trailing`

FFT-domain low-pass 只在 trailing window 上执行。

但窗口 endpoint 容易有：

```text
spectral leakage
ringing
```

所以不建议作为 P0。

---

# 9. 不建议 Production 的双向/重绘型滤波

以下只能 offline diagnostic 或明确 research：

```text
filtfilt / sosfiltfilt
centered Savitzky-Golay
centered LOESS
two-sided HP filter
full-sample spline smoothing
full-sample EMD/CEEMDAN reconstruction
full-sample Wiener smoother
Kalman smoother (RTS backward smoothing)
```

原因：

```text
它们使用 future observations 或回溯修正历史 state
```

不满足实时 causal factor 语义。

---

# 10. G 类：Change-Point-Aware Filters

这是我认为非常值得补、而且和普通 KAMA 不同的一大类。

## G1. `state_change_point_adaptive_ema`

平常：

```text
low alpha
强平滑
```

检测到：

```text
CUSUM / GLR / Pettitt / change score
```

时：

```text
暂时提高 alpha
快速跟上新 regime
```

因此解决：

```text
“平时低换手，但真实结构变化时不要慢半拍”
```

---

## G2. `state_filter_reset_on_break`

如果 change-point 强度超过阈值：

```text
直接 reset smoother state 到 current observation
```

适合：

```text
permanent level shift
```

但 threshold 必须非常保守。

---

## G3. `state_gain_scheduler`

输入：

```text
signal
state/regime indicator
```

根据市场状态选择：

```text
fast gain
normal gain
slow gain
```

例如：

```text
趋势强 → fast
震荡 → slow
高成本/低流动性 → slow
```

这比把 filter parameters 固定全年更合理。

---

# 11. H 类：Cross-Sectional Shrinkage Filters

这是传统“时序滤波”之外非常值得加的一层。

## H1. `cs_shrink_to_market_mean`

\[
y_{i,t}
=
(1-\lambda)x_{i,t}
+
\lambda \bar{x}_t
\]

适合：

```text
高噪声 raw cross-sectional score
```

但会压缩截面 dispersion，因此必须看 RankIC/coverage。

---

## H2. `cs_shrink_to_group_mean`

向行业/概念/PIT group mean 做 shrinkage：

\[
y_{i,t}
=
(1-\lambda)x_{i,t}
+
\lambda \bar{x}_{g(i),t}
\]

用途：

```text
抑制个股孤立噪声
保留行业相对结构
```

必须使用 PIT membership。

---

## H3. `cs_empirical_bayes_shrinkage`

根据：

```text
个股信号估计不确定度
group dispersion
```

自动决定 shrink strength。

比固定 λ 更合理。

Research/Extended。

---

## H4. `cs_peer_graph_smooth`

基于：

```text
行业
风格
收益相似度
供应链/关系图
```

做 graph Laplacian smoothing：

\[
\min_z \|z-x\|^2 + \lambda z^\top L z
\]

非常适合：

```text
多股票 noisy cross-sectional signal
```

但必须：

```text
PIT graph
严格控制过平滑
```

Research only 起步。

---

# 12. I 类：直接面向换手/成本的 Filter

这类往往比“继续平滑 raw factor”更直接。

## I1. `state_l1_turnover_prox`

求：

\[
y_t
=
\arg\min_y
\frac12(y-x_t)^2
+
\lambda |y-y_{t-1}|
\]

解本质上对应：

```text
围绕旧状态的 soft-threshold/no-trade band
```

这是 `deadband` 的优化论解释版。

建议直接做正式 Turnover Filter。

---

## I2. `state_l2_partial_adjustment`

\[
y_t=
\frac{x_t+\lambda y_{t-1}}{1+\lambda}
\]

等价于一种 partial adjustment。

比 absolute slew 更平滑。

---

## I3. `state_elastic_turnover_prox`

同时：

\[
\lambda_1|y-y_{t-1}|
+
\lambda_2(y-y_{t-1})^2
\]

得到：

```text
小变化不动
大变化渐进调整
```

很符合量化组合信号。

---

## I4. `state_cost_aware_deadband`

最值得增加之一。

定义 band：

\[
band_{i,t}
=
k \times expected\_trading\_cost_{i,t}
\]

只有：

```text
预期 signal improvement
```

超过成本阈值才更新。

输入成本代理可以来自：

```text
spread
Amihud
ADV
impact proxy
```

这会把“降换手”直接与真实交易成本挂钩。

---

## I5. `state_cost_aware_slew`

流动性越差：

```text
允许日变化越小
```

流动性越好：

```text
允许更快更新
```

比固定 rank slew 更贴近实盘。

---

# 13. J 类：Confidence / Uncertainty-Aware Filters

## J1. `state_confidence_weighted_ema`

模型如果输出：

```text
score + confidence
```

定义：

\[
\alpha_t
=
\alpha_{min}
+
confidence_t(\alpha_{max}-\alpha_{min})
\]

高置信度：

```text
快速更新
```

低置信度：

```text
更多平滑
```

很适合模型输出层。

---

## J2. `state_uncertainty_deadband`

如果：

```text
|new_score-old_score|
```

没有超过：

```text
预测不确定度
```

就不更新。

例如：

\[
|x_t-y_{t-1}| < k\sigma_t
\Rightarrow y_t=y_{t-1}
\]

非常适合 ensemble/model uncertainty。

---

# 14. 既有状态控制算子应该纳入统一 Filter Layer

不要重复造：

```text
state_ewm_if
event_refractory
hysteresis
state_latch
state_hold
state_deadband
state_slew_limit
```

而应给它们补：

```text
FilterRole
FilterContract
TurnoverRole
state checkpoint
search policy
```

使它们成为统一 Signal Conditioning Layer 的一部分。

---

# 15. 建议的优先级

## P0：非常值得加入

```text
ts_median3_causal
ts_butterworth_lowpass_causal
ts_causal_local_linear_smoother

state_change_point_adaptive_ema

state_l1_turnover_prox
state_l2_partial_adjustment
state_cost_aware_deadband
state_cost_aware_slew

state_confidence_weighted_ema
state_uncertainty_deadband
```

加上上一份已经建议的：

```text
Hampel
RobustEMA
KAMA
SuperSmoother
AdaptiveDeadband
RankDeadband
AdaptiveSlew
```

基本能覆盖绝大多数实用场景。

---

## P1：值得 Research→Extended

```text
Bessel causal lowpass
FIR lowpass
causal Savitzky-Golay endpoint
VIDYA
McGinley
NLMS
RLS
L1 trend filter
TV filter
wavelet shrinkage
SSA denoise
group shrinkage
empirical-Bayes shrinkage
```

---

## P2：Research only

```text
H-infinity
Student-t Kalman
adaptive-noise Kalman
particle filter
graph Laplacian smoothing
MODWT denoise
spectral lowpass
EMD/CEEMDAN-style decompositions
```

---

# 16. 不建议为了“滤波器数量”盲目扩张

最终 FilterEngine 不应该变成：

```text
100 个不同名字的均线
```

真正需要的是“机制覆盖”。

推荐核心机制数量：

```text
Robust/despike        3~5
Causal low-pass       3~4
Adaptive              3~4
State-space           2~4
Jump-aware            2~3
Cross-sectional       2~4
Turnover/cost-aware   4~6
Confidence-aware      2~3
```

大约 25~35 个真正不同机制已经很完整。

---

# 17. 最推荐的最终生产 Filter Bank

如果只保留一套高价值 production bank：

```text
1  causal Hampel
2  median3
3  RobustEMA
4  KAMA
5  SuperSmoother
6  causal Butterworth
7  causal local-linear endpoint

8  adaptive deadband
9  rank deadband
10 adaptive slew
11 quantile hysteresis

12 change-point adaptive EMA
13 L1 turnover prox
14 cost-aware deadband
15 cost-aware slew

16 confidence-weighted EMA
17 uncertainty deadband

18 Kalman level
19 robust Kalman（成熟后）
```

其余留 Research。

---

# 18. 推荐的使用逻辑

## 高频毛刺明显的量价因子

```text
Median3 / Hampel
→ RobustEMA
```

## 普通日频横截面因子

```text
KAMA / SuperSmoother
→ CS Rank
→ RankDeadband
```

## 很重视换手的组合

```text
CS Rank
→ CostAwareDeadband
→ L1TurnoverProx / AdaptiveSlew
```

## 市场经常突然变 regime

```text
RobustEMA
+
ChangePointGainScheduler
```

## 模型 score 有 confidence/uncertainty

```text
ConfidenceWeightedEMA
→ UncertaintyDeadband
```

---

# 19. Filter Search Governance

滤波器不能成为新的参数海洋。

每个 family：

```text
2~4 certified presets
```

优先。

禁止：

```text
任意 order
任意 cutoff
任意 λ
任意 nesting
```

Filter search 应是：

```text
Raw factor
→ 基本 alpha gate
→ 少量 certified FilterBank
→ validation selection
→ untouched OOS
```

---

# 20. 最重要的新增 Hard Gates

```text
FILTER_ZERO_NONCAUSAL_PRODUCTION_FILTER

FILTER_DIGITAL_IIR_USES_CAUSAL_FORWARD_PATH_ONLY

FILTER_ALL_IIR_HAVE_CHECKPOINT_STATE

FILTER_ALL_TREND_FILTERS_TRAILING_ENDPOINT_ONLY

FILTER_ALL_CHANGE_AWARE_FILTERS_HAVE_JUMP_RESPONSE_TEST

FILTER_ALL_COST_AWARE_FILTERS_BIND_PIT_COST_PROXY

FILTER_ALL_CONFIDENCE_FILTERS_BIND_UNCERTAINTY_SOURCE

FILTER_CROSS_SECTIONAL_SHRINKAGE_USES_PIT_UNIVERSE

FILTER_GRAPH_SMOOTHING_USES_PIT_GRAPH

FILTER_ZERO_FILTER_FAMILY_PARAMETER_EXPLOSION

FILTER_ECONOMIC_FACTOR_ID_SEPARATE_FROM_FILTER_POLICY_ID

FILTER_PRODUCTION_BANK_MECHANISM_DEDUP_PASS
```

---

# 21. 最终判断

Filter Layer 不需要追求“滤波器越多越好”。

应该追求：

```text
不同噪声结构
→ 有不同机制解决

孤立毛刺
→ robust/median

连续高频抖动
→ low-pass/adaptive

真实 regime jump
→ change-aware

横截面个股孤立噪声
→ shrinkage

无意义调仓
→ hysteresis/prox

交易成本高
→ cost-aware

模型置信度低
→ uncertainty-aware
```

这样才是一个真正适合 A 股日频横截面 FactorEngine 的完整 Filter Bank。
