# FactorEngine 信号滤波与降换手层专项整改任务书

> 仓库：`18047533889/quant_projects`  
> 范围：FactorEngine / Model Layer / Signal Conditioning  
> 原则：只处理滤波、信号平滑、去毛刺、换手控制及其模型治理，不重复此前已发送整改项。开工前重新读取最新 `main`。

## 1. 目标

这里的“滤波”不能等于“把曲线弄平”。真正目标是：

`去毛刺 → 抑制无意义高频摆动 → 尽量保留真实趋势/拐点/持续跳变 → 稳定横截面 rank → 降低无意义调仓和换手 → 提高成本后净收益。`

不能无差别平滑原始 OHLCV。涨跌停、公告跳空、放量突破等可能是真实 alpha；坏 tick、错误复权、source corruption 应由 DataAccess/DataQuality 处理。Filter Layer 只处理信号噪声和估计噪声。

## 2. 当前已有能力

当前代码已经有：

- `ts_ema`
- `state_slew_limit`
- `state_deadband`
- `state_latch`
- `state_hold`
- Kalman level/trend/beta
- causal Haar wavelet low-pass
- HMA / ALMA / RSX / QQE

其中 `state_deadband` 与 `state_slew_limit` 已经非常接近“降换手”，但现在主要是绝对尺度阈值，无法跨不同因子自然复用；现有 Kalman 与 wavelet 更偏模型/研究组件，也还不是专门的 signal-conditioning 生产层。

## 3. 正式建议的 Filter Pipeline

```text
RAW FACTOR / MODEL SCORE
    ↓
[1] Robust Despike
    ↓
[2] Adaptive Low-pass
    ↓
[3] Cross-sectional Rank / Normalization（可选）
    ↓
[4] Hysteresis / Deadband
    ↓
[5] Slew / Partial Adjustment
    ↓
FILTERED TRADABLE SIGNAL
```

不是每条链都要五层。默认允许：

```text
Despike only
Smooth only
Deadband only
Despike + Smooth
Smooth + RankDeadband
Despike + KAMA + RankDeadband + Slew
```

## 4. P0：新增 `ts_hampel_filter_causal`

用途：去掉单日/短时量价毛刺。

严格使用过去窗口估计参考，不让当前点参与自己的阈值：

\[
m_t = median(x_{t-w:t-1})
\]

\[
s_t = 1.4826 \cdot median(|x-m_t|)
\]

若：

\[
|x_t-m_t| > k s_t
\]

推荐不是直接删掉，而是 clip：

\[
x_t^* = m_t + clip(x_t-m_t,-ks_t,ks_t)
\]

这样仍保留跳变方向，只抑制异常幅度。

参数：

```text
window
n_sigma
replacement = clip / median
scale_floor
```

`scale_floor` 为 numerical policy，不可搜索；`n_sigma` 只允许少量认证值，如 2.5/3/4。

## 5. P0：新增 `ts_robust_ema`

普通 EMA 会被单个大毛刺明显拉走。建议在 innovation 层做稳健截断：

\[
e_t=x_t-y_{t-1}
\]

由历史 innovation 的 MAD 得到：

\[
s_t=MAD(e_{t-w:t-1})
\]

\[
e_t^*=clip(e_t,-ks_t,ks_t)
\]

\[
y_t=y_{t-1}+lpha e_t^*
\]

优点是：比 “Hampel→EMA” 更低延迟，而且极端 observation 不会直接把递归 state 拉走。

## 6. P0：新增 `ts_kama`

KAMA 这类自适应平滑非常适合当前需求。

\[
ER_t=
rac{|x_t-x_{t-w}|}
{\sum_{j=1}^{w}|x_{t-j+1}-x_{t-j}|+\epsilon}
\]

当 ER 高：变化单向、趋势清晰，filter 应快速跟随。  
当 ER 低：来回抖动、噪声高，filter 自动加强平滑。

建议认证参数只保留小网格，例如：

```text
er_window = 10 / 20 / 40
fast_period = 2 / 3
slow_period = 20 / 40 / 60
```

不允许 LLM 任意连续搜索。

## 7. P0：新增通用 `ts_super_smoother`

当前技术指标内部已有 two-pole low-pass 思路，但应该抽象为真正 generic canonical。

目标：

```text
比 SMA/EMA 更强地衰减高频噪声
同时减少普通长均线的滞后
```

它是递归 IIR，应标记：

```text
stateful=True
checkpointable=True
time_shard_safe=False without checkpoint
```

## 8. P0：新增 `state_adaptive_deadband`

现有 `state_deadband(x, band)` 的最大问题是 `band` 直接使用原始单位。

建议改为新增 canonical，而不是修改旧语义：

\[
band_t = k \cdot scale_t
\]

其中 `scale_t` 推荐：

```text
rolling MAD of delta(signal)
```

也可支持：

```text
rolling std of delta(signal)
cross-sectional dispersion
```

这样收益因子、量比、成交额、模型 score 都可以用统一的 dimensionless `band_mult`。

## 9. P0：新增 `state_adaptive_slew_limit`

现有 `state_slew_limit` 继续保留。

新增：

\[
limit_t=k\cdot scale_t
\]

\[
y_t=y_{t-1}+clip(x_t-y_{t-1},-limit_t,+limit_t)
\]

用途是限制单日 signal/target 改变量，直接减少大幅权重调整。

## 10. P0：新增 `state_rank_deadband`

这是最推荐给 A 股横截面选股因子的新增算子。

真正影响换手的经常不是 raw factor 从 `1.23→1.24`，而是横截面 rank 从 `82%→79%`，导致股票频繁穿过组合阈值。

建议：

```text
raw factor
→ cs_rank_pct
→ state_rank_deadband
```

例如：

```text
band_pct = 0.03 / 0.05 / 0.10
```

如果今日 rank 相对上次已生效 rank 的变化小于 5 个百分点，则不更新。

这种在 rank space 做迟滞，比 raw-value deadband 跨因子更稳定。

## 11. P0：新增 `state_quantile_hysteresis`

例如多头组合：

```text
进入：rank >= 90%
退出：rank < 80%
```

不是同一阈值进出。

这样股票在 82%~92% 之间来回波动时不会每天进出组合。

适合作为 selection/portfolio-interface 算子，不一定作为 continuous cold-start factor。

## 12. P1：可增加 `ts_one_euro_filter`

低变化速度时加强平滑，高变化速度时提高 cutoff 快速跟随。它与 KAMA 同属“平滑 vs 延迟”平衡器，但机制不同。

建议 Research/Extended 起步，参数小网格治理。

## 13. P1：可增加 `ts_alpha_beta_filter`

简单 local-level + trend 递归 filter，比完整 Kalman 更轻、更快、更容易大规模用于 FactorMiner。

## 14. P1：可增加 `ts_robust_kalman_level`

对 Kalman innovation 做 Huber/clip，避免单点 observation 把 state 拉走。

但必须等 Kalman 的 q/r scale、state checkpoint 等前序整改稳定后再生产准入。

## 15. 不建议直接加 centered Savitzky-Golay

标准 centered SG 会使用未来数据。

如果未来实现，只允许：

```text
trailing causal polynomial smoother
```

并必须明确不是 centered SG。

## 16. Wavelet Low-pass 的定位

现有 causal Haar low-pass 保持 Research Transform 更合适。

它适合多尺度研究，但不建议作为默认降换手 filter，因为参数/延迟/计算结构不如 RobustEMA/KAMA/Deadband 直观。

## 17. 建立正式 `FilterRole`

```python
class FilterRole(Enum):
    DESPIKE = "despike"
    LOW_PASS = "low_pass"
    ADAPTIVE_LOW_PASS = "adaptive_low_pass"
    HYSTERESIS = "hysteresis"
    RATE_LIMIT = "rate_limit"
    STATE_HOLD = "state_hold"
    JUMP_DETECTOR = "jump_detector"
    SIGNAL_NORMALIZER = "signal_normalizer"
```

每个 operator 只有一个 primary role。

## 18. 建立 `FilterContract`

至少包括：

```text
role
causal
uses_current_observation
reference_cutoff

stateful
checkpointable
time_shard_safe

input_unit_policy
scale_normalized

warmup
missing_policy

lag_class
expected_lag

jump_policy
turnover_control

searchable_parameters
non_searchable_parameters
```

## 19. 去毛刺不等于消灭真实 Jump

在 t 时刻看见一个大变化，仅凭过去无法确定它是：

```text
错误毛刺
还是新的真实 regime jump
```

因此 production causal filter 默认应该：

```text
downweight
clip innovation
adaptive follow
```

而不是“只要大就删”。

如果要识别“单日 spike 后第二天完全回撤”，必须等 t+1，天然有确认延迟，不得回填成 t 时刻信号。

## 20. `JumpPreservationContract`

建议至少：

```text
PRESERVE_ALL_FINITE_JUMPS
ROBUST_CLIP_INNOVATION
DELAYED_CONFIRMATION
```

Price/return 类信号默认不得 hard-delete extreme，除非 DataQuality 已确认坏点。

## 21. 阈值优先 dimensionless

不要大量出现：

```text
band=0.01
limit=0.02
```

优先：

```text
MAD multiples
vol multiples
percentile rank points
```

例如：

```text
deadband = 0.5 × rolling_MAD(delta(signal))
slew = 1.0 × rolling_MAD(delta(signal))
rank_deadband = 5 percentile points
```

## 22. 最推荐的默认组合

### 轻度去毛刺

```text
raw factor
→ causal Hampel clip
```

### 稳健低延迟

```text
raw factor
→ RobustEMA(span≈5~10)
```

### 自适应平滑

```text
raw factor
→ KAMA
```

### 横截面直接降换手

```text
raw factor
→ CS Rank
→ Rank Deadband
```

### 更强降换手

```text
raw factor
→ RobustEMA / KAMA
→ CS Rank
→ Rank Deadband
→ Adaptive Slew
```

## 23. Filter 评价不能只看“平不平”

必须至少看三组指标。

Alpha preservation：

```text
RankIC
RankICIR
long-short return
long-short Sharpe
decay
```

Turnover：

```text
factor rank autocorrelation
quantile turnover
top-k turnover
target L1 change
portfolio turnover proxy
```

Lag：

```text
cross-correlation lag
event-response delay
shock capture ratio
turning-point delay
```

## 24. 推荐 Admission 方式

不要单一加权总分。

建议：

```text
IC retention >= 90%~95%
turnover reduction >= 15%~20%
lag <= 1~2 bars
coverage unchanged
```

先满足 gate，再比较成本后净收益。

## 25. 定义 `ICRetention`

\[
ICRetention=
rac{|IC_{filtered}|}{|IC_{raw}|+\epsilon}
\]

同时必须报告 raw/filtered sign，若 sign flip 应 hard fail。

## 26. 定义 `TurnoverReduction`

\[
TR=
1-rac{Turnover_{filtered}}{Turnover_{raw}+\epsilon}
\]

同时报告绝对值和相对值。

## 27. 定义 `RankStabilityGain`

\[
Gain=
RankCorr(f_t,f_{t-1})_{filtered}
-
RankCorr(f_t,f_{t-1})_{raw}
\]

## 28. 必须增加 Event Preservation Test

至少构造：

```text
A. 单日 spike 后立即恢复
B. 永久 level shift
C. 持续 trend
D. V 型反转
E. regime jump
```

要求：

```text
单日噪声 spike：显著衰减
永久 level shift：不能长期压回旧值
持续 trend：不能迟滞过大
regime jump：有限 bars 内追上
```

## 29. 递归 filter 要有 Step/Impulse Response Evidence

EMA、KAMA、SuperSmoother、RobustEMA、Kalman、Slew 都输出：

```text
rise time
settling time
overshoot
impulse suppression
```

用于衡量平滑与延迟。

## 30. Filter Chain 也是 Stateful DAG

例如：

```text
RobustEMA → RankDeadband → Slew
```

多个递归 state 必须一起 checkpoint。

实盘/模拟盘重启后不能因 filter state 丢失造成 signal 跳变。

## 31. Source revision 要 replay filter state

历史输入被 revision 后，应从 revision date 起重放递归 filter state。

不能改历史数据却继续使用旧 state。

## 32. Filter 参数同样只能 Validation 选择

`span / band / slew / clip_sigma / fast / slow` 均不能在 final test 上调。

它们也必须进入 SearchExposureLedger。

## 33. 防止滤波器制造参数克隆爆炸

原因子如果再乘：

```text
8 spans × 5 bands × 4 slew × 4 clip
```

就会变成大量参数型“伪多样性”。

每类 filter 默认只保留 2~4 个 certified presets。

## 34. Filter 参数角色

```text
window/span           → ECONOMIC_HORIZON
fast/slow             → ADAPTATION_POLICY
clip_sigma            → ROBUSTNESS_POLICY
deadband/slew         → TURNOVER_POLICY
eps/tol               → NUMERICAL_POLICY
```

NUMERICAL_POLICY 不允许搜索。

## 35. Filtered factor 与经济因子 identity 分离

例如：

```text
momentum
EMA5(momentum)
EMA10(momentum)
EMA20(momentum)
```

不能当作 4 个独立经济机制。

建议：

```text
economic_factor_id
conditioning_policy_id
```

分别记录。

## 36. LLM Factor Mining 推荐两阶段

Stage A：

```text
先挖 raw economic factor
```

Stage B：

```text
只对通过基本 IC gate 的 factor
尝试少量 certified filter policies
```

不要允许 AST 一开始就任意多层滤波嵌套。

## 37. 限制 nesting depth

默认：

```text
despike <= 1
smooth <= 1
turnover_control <= 2
```

禁止：

```text
EMA(EMA(KAMA(HMA(...))))
```

## 38. 规范 Filter Composition Order

推荐默认：

```text
DESPIKE
→ LOW_PASS
→ NORMALIZE/RANK
→ HYSTERESIS
→ RATE_LIMIT
```

其他顺序先留 Research。

## 39. 三层稳定控制

FactorEngine 建议支持：

```text
RAW_SIGNAL_FILTER
RANK_FILTER
TARGET_FILTER
```

其中横截面选股最推荐 Rank Filter。

真正最终 portfolio turnover 仍受组合优化、风险约束、停牌、涨跌停等影响；Portfolio 层还应单独有 turnover penalty / no-trade region / transaction-cost-aware optimizer。

## 40. 模型输出也可以滤波，但必须成为正式 production identity

如果：

```text
model_score
→ RobustEMA
→ RankDeadband
```

则 `raw model + filter chain` 整体才是 production predictor identity。

禁止：

```text
研究回测 raw model
实盘偷偷加 EMA
```

然后仍引用 raw model 的历史绩效。

## 41. 模型准入要评估最终 filtered score

保存：

```text
raw model OOS
filtered model OOS
```

如果线上使用 filtered score，准入 RankIC/turnover/net Sharpe 必须来自 filtered output。

## 42. 新旧模型切换时 Filter State 需要 Migration Contract

至少：

```text
RESET
WARM_START_IF_SCALE_COMPATIBLE
BRIDGE
```

Rank-space filter 更适合 warm start；raw expected-return filter 要先检查新旧模型 score scale 是否兼容。

## 43. 模型层新增 Filter-specific 问题

```text
F-M-001
研究/验证用 raw score，实盘用 filtered score，research-production mismatch。

F-M-002
filter state 没进入 artifact/checkpoint identity。

F-M-003
filter parameter tuning 没进入 SearchExposureLedger。

F-M-004
模型 admission 没有 turnover / net-after-cost gate。

F-M-005
模型没有 lag / jump-preservation evidence。

F-M-006
model output contract 未说明是否允许 temporal filtering。

F-M-007
filter chain 改变后仍复用旧 evidence/cache。

F-M-008
没有同时保存 raw prediction / filtered prediction。

F-M-009
model retrain 与 filter reset 边界没定义。

F-M-010
新 artifact 激活时 filter state migration 没有 contract。
```

## 44. 推荐 Hard Gates

```text
FILTER_ALL_PRODUCTION_FILTERS_CAUSAL
FILTER_ZERO_CENTERED_FUTURE_SMOOTHING
FILTER_ALL_STATEFUL_HAVE_CHECKPOINT
FILTER_ALL_FILTERS_HAVE_MISSING_POLICY
FILTER_ALL_FILTERS_HAVE_WARMUP_POLICY
FILTER_ALL_FILTERS_HAVE_SCALE_POLICY
FILTER_ALL_DESPIKE_HAVE_JUMP_POLICY

FILTER_ZERO_RAW_OHLCV_SILENT_REPLACEMENT
FILTER_ALL_TURNOVER_CONTROLS_DIMENSIONLESS_OR_SCALE_AWARE

FILTER_ROBUST_EMA_AVAILABLE
FILTER_KAMA_AVAILABLE
FILTER_SUPERSMOOTHER_AVAILABLE
FILTER_RANK_DEADBAND_AVAILABLE
FILTER_ADAPTIVE_SLEW_AVAILABLE

FILTER_ALL_FILTERS_HAVE_LAG_EVIDENCE
FILTER_ALL_FILTERS_HAVE_EVENT_PRESERVATION_TEST
FILTER_ALL_FILTERS_HAVE_TURNOVER_EVIDENCE
FILTER_ALL_FILTERS_HAVE_ALPHA_RETENTION_EVIDENCE

FILTER_ZERO_PARAMETER_CLONE_COUNTED_AS_NEW_MECHANISM
FILTER_GRAMMAR_MAX_DESPIKE_DEPTH_ONE
FILTER_GRAMMAR_MAX_SMOOTH_DEPTH_ONE
FILTER_PRODUCTION_CHAIN_ORDER_CERTIFIED

FILTER_PARAMETERS_VALIDATION_ONLY
FILTER_FINAL_TEST_UNTOUCHED
FILTER_MODEL_SCORE_EVALUATED_AFTER_PRODUCTION_FILTER
FILTER_STATE_REPLAY_ON_SOURCE_REVISION
FILTER_CROSS_BACKEND_PARITY
```

## 45. P0 实施优先级

第一批只做最有价值的：

```text
1. ts_hampel_filter_causal
2. ts_robust_ema
3. ts_kama
4. ts_super_smoother
5. state_adaptive_deadband
6. state_adaptive_slew_limit
7. state_rank_deadband
8. state_quantile_hysteresis

+ FilterContract
+ FilterChainSpec
+ FilterEvaluationReport
+ state checkpoint
+ validation-only filter parameter selection
```

不要第一批一次性加几十种滤波器。

## 46. 最推荐的核心六个

如果只做最精简版本：

```text
ts_hampel_filter_causal
ts_robust_ema
ts_kama
state_adaptive_deadband
state_rank_deadband
state_adaptive_slew_limit
```

这六个已经覆盖：

```text
去毛刺
低延迟平滑
自适应噪声控制
raw signal 稳定
rank 稳定
换手限制
```

## 47. 对 A 股日频横截面量价因子的默认实践

推荐：

```text
RAW FACTOR
    ↓
RobustEMA / KAMA
    ↓
CS Rank
    ↓
Rank Deadband
```

如果换手仍高，再：

```text
Adaptive Slew
```

而不是一开始把 raw factor 平得非常重。

## 48. 成功标准

一个好的 filter 不是“看起来很平”，而应满足：

```text
RankIC：小幅下降 / 持平 / 提升
RankICIR：持平或提高
Rank autocorrelation：提高
Top-decile turnover：明显下降
Portfolio turnover proxy：下降
Net Sharpe after cost：提升
Signal lag：可接受
真实 jump：没有被抹掉
```

例如：

```text
turnover -30%
IC -5%
net Sharpe +20%
```

可能是非常成功的 filter。

而：

```text
turnover -50%
IC -50%
```

不是成功。

## 49. Definition of Done

```text
[ ] FilterRole / FilterContract 建立
[ ] DataQuality 与 Signal Filter 严格分层
[ ] 不再无差别平滑 raw OHLCV

[ ] causal Hampel 完成
[ ] Robust EMA 完成
[ ] KAMA 完成
[ ] generic SuperSmoother 完成

[ ] Adaptive Deadband 完成
[ ] Adaptive Slew 完成
[ ] Rank Deadband 完成
[ ] Quantile Hysteresis 完成

[ ] 阈值优先 dimensionless / scale-aware

[ ] 所有 recursive filter 可 checkpoint
[ ] source revision 可 replay
[ ] Pandas/Polars/optimized path parity

[ ] Filter composition order 受治理
[ ] nesting depth 受限
[ ] 参数克隆不计作独立经济机制
[ ] LLM grammar 将 filter 作为 optional conditioning wrapper

[ ] 每个 filter 有 alpha-retention evidence
[ ] 每个 filter 有 turnover evidence
[ ] 每个 filter 有 lag evidence
[ ] 每个 despike 有 jump-preservation evidence

[ ] Filter 参数只在 validation 选择
[ ] final test untouched

[ ] 模型实盘使用 filtered score 时，准入按 filtered score 评价
[ ] filter state 进入 production checkpoint
[ ] model artifact 切换时 filter-state migration 有 contract

[ ] 最终报告同时展示 gross alpha / turnover / cost / net alpha
```

## 50. 最终原则

Filter Layer 的目标是：

```text
“不要因为没有意义的抖动频繁换仓”
```

而不是：

```text
“让所有信号都变平”
```

最优先的设计原则：

```text
噪声时慢
趋势时快
小变化不交易
大变化能跟上
真实跳变不抹掉
状态可重放
参数不过拟合
成本后净收益更高
```

对 A 股横截面因子，推荐优先顺序：

```text
Robust innovation control
→ Adaptive smoothing
→ Rank-space hysteresis
→ Rate limiting
```
