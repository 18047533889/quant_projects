# FactorEngine R28：全算子逐项审计、模型 Walk-Forward / PIT 安全、禁用清理与逐算子测试证据闭环

> **执行基线（本文件生成时）**  
> Repository: `18047533889/quant_projects`  
> Target: `factor_engine/`  
> Baseline HEAD: `37008c7ff4eff3823480443b6325577eae394250`  
> Date: `2026-08-10`
>
> **本轮目标不是抽样审计。**  
> 必须对当前 HEAD 中 **每一个 registered canonical operator** 建立可追溯处置结论、真实测试和当前 HEAD 绑定的机器证据。任何算子不得因为“名字看起来没问题”“以前审过”“有 status/tag”“能 import”而跳过。
>
> 本文是独立整改文档。R24/R25/R26/R27 的语义、真实可用性、数值正确性、资源性能要求继续有效；R28 重点补齐：
>
> 1. **全 canonical 无遗漏盘点**
> 2. **永久无用 / 随机 / 未来函数 / 非因果算子的物理清理**
> 3. **模型类算子的 Walk-Forward / PIT / Label Maturity / Purge-Embargo 安全**
> 4. **每个 retained canonical 至少一个真实执行测试**
> 5. **高风险算子必须有因果性 / 未来扰动 / 前缀不变性等专项测试**
> 6. **测试结果与审计证据必须作为 GitHub 可见 tracked artifacts 提交**
> 7. **证据必须绑定当前 Git SHA + canonical-set digest，旧证据不得冒充当前证据**

---

# 一、先明确当前事实：现在不能说“1362 个算子都已经逐个检查完了”

当前 `cleaned_operators/docs/operators_catalog.json` 显示：

```text
canonical_count = 1362

daily       = 1185
extended    = 74
research    = 92
unsafe      = 7
legacy      = 1
internal    = 3
unclassified= 0
```

但当前仓库内已有旧审计报告却分别写：

```text
R19_OPERATOR_MATH_AUDIT.md
total rows = 1430
blockers   = 648

R23_PER_CANONICAL_AUDIT.md
Total canonicals = 1436
CERTIFIED = 28
CERTIFIED_CONTEXTUAL = 21
SUPPORTING_ONLY = 1384
RESEARCH_TOOL = 3
```

所以现状至少存在：

```text
current catalog canonical set       = 1362
R19 evidence canonical set          = 1430
R23 evidence canonical set          = 1436
```

这三者不一致。

因此：

```text
不能用 R19 / R23 的旧表证明当前 HEAD 的每一个 canonical 都已经审过。
```

R28 必须重新从 **fresh process + current HEAD** 枚举 canonical set，所有审计和测试都绑定这一个集合。

---

# 二、R28 的最终回答标准

完成后，以下问题必须能由机器证据直接回答，而不是靠口头判断：

```text
Q1. 当前一共有多少 canonical？
Q2. 每个 canonical 在哪个源文件、哪个 class/function 中实现？
Q3. 每个 canonical 最终用途是什么？
Q4. 哪些可用于生产挖因子？
Q5. 哪些只能做 state / condition / event / intermediate？
Q6. 哪些只能 research？
Q7. 哪些只应 internal？
Q8. 哪些必须彻底删除？
Q9. 哪些是永久禁止 tombstone？
Q10. 是否还有随机数因子？
Q11. 是否还有 future function？
Q12. 是否还有 bfill / centered rolling / two-sided smoother？
Q13. 是否还有全样本 scaler/PCA/model fit？
Q14. 所有模型是否严格 walk-forward？
Q15. 所有 supervised model 的 label 是否成熟后才进入训练？
Q16. 每个 retained canonical 是否至少有一个真实执行测试？
Q17. 每个 high-risk canonical 是否有专项因果测试？
Q18. 测试结果是否与当前 HEAD 一致？
Q19. 测试结果是否在 GitHub 可见？
Q20. 是否存在“代码有算子但 evidence 没有”或反过来的情况？
```

任何一个回答不出来：

```text
R28 不通过。
```

---

# 三、禁止再以“surface”代替真实安全结论

当前体系已经区分：

```text
AuthoringTier
ProductionCertification
BackendCapability
MiningRole
DirectUseStatus
```

这是正确方向。

但 R28 强制：

```text
daily / extended
≠
production safe

registered
≠
usable

implemented
≠
correct

pit_safe tag
≠
PIT proof

diagnostic_only tag
≠
矿工真的看不到

research_only set
≠
所有 alias / consumer 都绝对拿不到
```

最终准入只能消费：

```text
current-code-backed test/evidence
```

---

# 四、每个 canonical 必须有唯一最终 disposition

不再允许大量使用含义模糊的：

```text
SUPPORTING_ONLY
PENDING_FOREVER
MAYBE
UNKNOWN
```

每个当前 canonical 必须唯一归入：

```text
PRODUCTION_FACTOR
PRODUCTION_HIGH_COST
CONTEXTUAL_FACTOR
STATE_CONDITION_EVENT
INTERMEDIATE_TRANSFORM
RESEARCH_DIAGNOSTIC
INTERNAL_HELPER
BLOCKED_NO_DATA
BLOCKED_CONTEXT
PERMANENTLY_FORBIDDEN_TOMBSTONE
DELETE_DUPLICATE
DELETE_NONCAUSAL
DELETE_MATH_DEFECT
DELETE_USELESS
DELETE_OBSOLETE
```

并给出：

```text
reason
replacement
test_ids
evidence_ids
```

---

# 五、随机数 / 未来函数 / 非因果算子：不是“打标签”，而是彻底清理公共路径

当前 `PERMANENTLY_FORBIDDEN_CANONICALS` 已包含：

```text
Lead
next

bfill
causal_bfill
fillna_interpolate
interpolate

shuffle
sample

rand_exp
rand_lognormal
rand_normal
rand_poisson
rand_uniform

dropna
constant

norm
norm_l1
norm_linf
```

这说明方向是对的。

但是 R28 要进一步证明：

```text
这些名字不是仅仅“status=denied”，
而是已经无法作为生产/挖掘 operator 被发现或执行。
```

---

# 六、永久禁止算子的正确存在形式

对于：

```text
Lead
next
bfill
shuffle
sample
rand_*
```

推荐最终状态：

```text
只保留 Tombstone / Deny Registry
```

用于：

```text
旧表达式解析时给出明确错误
旧配置迁移时给出 replacement / delete reason
```

不要再保留：

```text
public runtime implementation
production registry entry
daily/extended authoring surface
mining grammar
DirectUse recipe
backend emitter
SQL emitter
Polars emitter
public alias
```

---

# 七、永久禁止算子硬测试

新增：

```text
test_permanently_forbidden_not_registered_for_runtime
test_permanently_forbidden_not_in_daily_allowlist
test_permanently_forbidden_not_in_extended_allowlist
test_permanently_forbidden_not_in_mining_catalog
test_permanently_forbidden_not_in_direct_use
test_permanently_forbidden_not_in_backend_emitters
test_permanently_forbidden_alias_cannot_escape
test_legacy_formula_gets_explicit_forbidden_error
```

---

# 八、随机因子：生产中必须为 0

要求：

```text
PUBLIC_RANDOM_FACTOR_TERMINALS == 0
MINING_RANDOM_OPERATORS == 0
```

静态扫描：

```text
np.random
numpy.random
random.
default_rng
RandomState
choice
shuffle
permutation
sample
```

---

# 九、随机方法要区分两种

## A. 随机因子原语

例如：

```text
rand_uniform()
shuffle(x)
sample(x)
```

最终：

```text
永久删除 / tombstone
```

---

## B. 研究统计量内部的固定 seed surrogate

例如：

```text
phase-randomized surrogate null
```

如果：

```text
固定 seed
不依赖 global RNG state
结果完全 reproducible
仅 research
```

可保留：

```text
RESEARCH_DIAGNOSTIC
```

但不能：

```text
直接进入 production default mining
```

---

# 十、固定 seed 也必须测试

```text
same input + same params -> bitwise/reproducible
global np.random state changes -> output unchanged
parallel order changes -> output unchanged
process restart -> output unchanged
```

---

# 十一、当前 unsafe surface 也必须逐个重新问“是否有必要存在”

当前 unsafe 7 个：

```text
arg
tan
cot
sec
csc
cosh
sinh
```

R28 不自动把它们当有价值。

对每个问：

```text
量化因子表达里是否有真实、稳定、可解释的用途？
是否存在大量奇点/overflow？
是否只是为了数学函数全集而加入？
是否有现有 recipe/论文/研报实际使用？
是否可以由更安全变换替代？
```

如果只是：

```text
“数学库里有，所以 FactorEngine 也有”
```

应该：

```text
DELETE_USELESS
```

或：

```text
INTERNAL_HELPER / RESEARCH
```

不应占生产搜索空间。

---

# 十二、原始矩阵/线性代数 primitive 不应是 factor terminal

当前 deny 中已有：

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
```

这类可以作为：

```text
内部实现能力
```

但不应该：

```text
直接作为 daily factor terminal
```

正确做法是暴露：

```text
明确经济/统计语义的 trailing statistic
```

例如：

```text
rolling PCA residual
spectral concentration
wavelet energy ratio
```

而不是：

```text
pca(x)
svd(x)
fft(x)
```

---

# 十三、全库 Future-Lookahead Static Scan

新增脚本：

```text
factor_engine/scripts/audit_r28_lookahead_static.py
```

扫描所有：

```text
factor_engine/cleaned_operators/**/*.py
factor_engine/research_operators/**/*.py
factor_engine/research_tools/**/*.py
factor_engine/backend/**/*.py
factor_engine/planner/**/*.py
factor_engine/mining/**/*.py
```

---

# 十四、必须扫描的明确 future patterns

至少：

```text
.shift(-1)
.shift(-k)

Lead
lead
next

bfill
backfill

center=True
rolling(..., center=True)

iloc[row + 1]
iloc[t + 1]
values[row + 1]

[row + 1:]
[t + 1:]

future
forward_value
forward_feature

filtfilt
sosfiltfilt

symmetric convolution
two-sided smoothing

interpolate()
linear interpolation using future endpoint
```

---

# 十五、模型类额外 static patterns

扫描：

```text
.fit(
.fit_transform(
StandardScaler
RobustScaler
MinMaxScaler
QuantileTransformer
PowerTransformer

PCA(
TruncatedSVD(
KernelPCA(

KFold(
StratifiedKFold(
cross_val_score
train_test_split

shuffle=True

HMM
forward_backward
smoother
RTS
KalmanSmoother

np.mean(full_series)
np.std(full_series)
np.quantile(full_series)
cov(full_full)
corrcoef(full_full)
```

重点不是禁止 API 本身，而是确认：

```text
在计算 output[t] 时，fit 数据没有超过 t 的时间。
```

---

# 十六、必须扫描“全样本标准化再滚动”的泄漏模式

错误示例：

```python
z = (x - x.mean()) / x.std()

for t:
    fit(z[:t])
```

虽然：

```text
model fit只用过去
```

但：

```text
x.mean()/x.std()
```

已经看了未来。

必须改成：

```text
每个 fit window 内独立 scaler
```

或者：

```text
expanding scaler 截止 t-1
```

---

# 十七、必须扫描“全样本 PCA 再滚动”的泄漏模式

错误：

```text
full sample PCA
→ transform all dates
→ rolling model
```

正确：

```text
for decision t:
    fit PCA on historical training window
    transform t using historical PCA
```

---

# 十八、必须扫描超参数选择泄漏

错误：

```text
全样本 CV 找 alpha
然后回填历史 factor。
```

正确：

```text
每个 decision t
只在 t 之前的数据中选择超参数
```

或者：

```text
固定预先声明参数
```

---

# 十九、模型型算子建立单独 Model Causality Gate

任何名称 / category / source 命中：

```text
model
regression
forecast
innovation
predict
PCA
PLS
elastic
ridge
huber
quantile
expectile
AR
GARCH
Kalman
state_space
Markov
HMM
DMD
SSA
Hankel
Granger
HSIC
kernel
KNN
Lyapunov
RQA
recurrence
matrix_profile
anomaly
autoencoder
mixture
expert
regime
```

都必须走：

```text
MODEL_CAUSALITY_GATE
```

---

# 二十、模型安全的统一时间定义

对于 output at decision time `t`：

定义：

```text
decision_time = t
feature_cutoff = t
fit_cutoff = t-1
```

如果是纯 descriptive state：

```text
fit_cutoff 可为 t
```

但必须明确：

```text
它是 in-sample descriptive state
不是 forecast / innovation
```

---

# 二十一、预测型模型统一要求

若算子声称：

```text
forecast
prediction
innovation
forecast_error
out_of_sample
surprise
```

必须：

```text
model fit data <= t-1
scaler fit data <= t-1
PCA fit data <= t-1
kernel bandwidth fit data <= t-1
residual sigma fit data <= t-1
hyperparameter selection data <= t-1
```

当前 `(x_t, y_t)`：

```text
只能用于 scoring
不能用于 training
```

---

# 二十二、in-sample 模型不要冒充 alpha forecast

当前代码已经有不少：

```text
*_in_sample_resid
*_fitted_value
```

这类并不使用未来数据，

但：

```text
当前样本参与了拟合
```

所以：

```text
不是 out-of-sample innovation。
```

处置：

```text
可作为 RESEARCH_DIAGNOSTIC
或明确 descriptive STATE
```

但默认：

```text
不要进入自动 alpha terminal 搜索。
```

---

# 二十三、Legacy 误导名称应进一步收口

例如当前 AR 家族保留：

```text
ts_ar_forecast
ts_ar_innovation
```

实际 legacy kernel：

```text
fit_lag = 0
```

是：

```text
in-sample fitted value / residual
```

即使 metadata 写清楚，

名称仍容易被：

```text
人
LLM
自动 mining consumer
```

误用。

R28 建议：

```text
旧名称只保留 deprecated compat alias
且 alias 永远不可 mining
```

真正公开：

```text
ts_ar_prior_forecast
ts_ar_prior_innovation
```

---

# 二十四、动态回归家族要求

重点：

```text
ts_multi_regression_*
ts_huber_regression_*
ts_ridge_regression_*
ts_quantile_*
ts_expectile_*
```

必须区分：

```text
in_sample
prior
predictive
forecast_error
```

---

# 二十五、模型前缀不变性：PIT 最强基础测试之一

对每个 PIT-sensitive operator：

构造完整序列：

```text
x[0:T]
```

先算：

```text
full = op(x[0:T])
```

再对多个 cutoff `t`：

```text
prefix = op(x[0:t+1])
```

要求：

```text
prefix[t] == full[t]
```

允许数值 tolerance，

但：

```text
NaN mask必须一致。
```

---

# 二十六、Future Perturbation Test

这个比普通 prefix test更强。

先：

```text
base = op(x)
```

然后对 `t` 之后的所有数据：

```text
x[t+1:] = 巨大值 / 随机值 / 反号 / NaN
```

重新计算：

```text
changed = op(x_perturbed)
```

必须：

```text
base[:t+1] == changed[:t+1]
```

如果变化：

```text
明确 future leakage。
```

---

# 二十七、对模型类默认做多个 future perturbation

至少：

```text
future constant shift
future scale ×100
future sign flip
future random permutation
future NaN
future outliers
```

---

# 二十八、Supervised Model 不能只说“caller 已经 lag 好 label”

这是 R28 的一个硬原则。

如果算子接受：

```text
label y
```

不能依赖：

```text
调用方自己保证 label 没泄漏。
```

必须在 typed contract 中区分：

```text
FeaturePanel
LabelPanel
```

---

# 二十九、LabelContract

新增：

```python
@dataclass(frozen=True)
class LabelContract:
    horizon: int
    origin_time: ...
    maturity_time: ...
    availability_time: ...
    overlapping: bool
    purge_bars: int
    embargo_bars: int
```

---

# 三十、Forward Return Label

如果：

```text
y_t = return(t -> t+H)
```

这个 label 在：

```text
t+H
```

之后才成熟。

因此 decision time `T` 的训练集最多使用：

```text
label origin <= T-H
```

不是：

```text
origin <= T-1。
```

---

# 三十一、当前 panel supervised model 必须重新验证

重点：

```text
panel_rolling_pcr_forecast
panel_rolling_pls_forecast
panel_rolling_elastic_net_forecast
panel_regime_conditioned_forecast
panel_mixture_of_experts_score
```

当前实现已经有：

```text
label_horizon
```

和：

```text
训练尾部排除未成熟 label
```

这是正确方向。

但它们当前 metadata 仍然：

```text
pit_safe=False
research-only
```

R28 不要直接升 production。

必须先有：

```text
typed LabelContract
label maturity tests
walk-forward tests
purge/embargo tests
real mining exclusion tests
```

---

# 三十二、Overlapping labels 必须 purge / embargo

例如：

```text
5日 forward return
```

相邻训练 label：

```text
共享未来价格区间
```

模型验证时不能用普通：

```text
random KFold
```

至少：

```text
purged walk-forward
```

必要时：

```text
embargo。
```

---

# 三十三、禁止普通 shuffled CV 用于时序模型

以下在 production model operator中：

```text
KFold(shuffle=True)
train_test_split(shuffle=True)
random split
```

一律禁止。

---

# 三十四、推荐验证结构

```text
train [0 ................ a]
purge
validation [b ......... c]

然后时间向前滚
```

或：

```text
expanding walk-forward
rolling walk-forward
```

---

# 三十五、Scaler 必须属于模型 pipeline

训练：

```text
fit scaler on train
fit model on scaled train
```

预测：

```text
transform current using historical scaler
```

禁止：

```text
提前 scale 全历史。
```

---

# 三十六、PCA / PCR / PLS

PCA 型模型是最常见隐性前视源之一。

必须测试：

```text
PCA mean
PCA std
loading
component count
sign orientation
```

全部只由：

```text
历史训练窗
```

决定。

---

# 三十七、Rolling PCA 当前方向

当前 `panel_model.py` 已默认：

```text
fit_lag = 1
```

即：

```text
训练 [t-W, t-1]
评价 t
```

这是正确方向。

R28 仍需：

```text
未来扰动
prefix invariance
universe missing
stock reordering
chunk boundary
```

专项验证。

---

# 三十八、PCA sign 不得依赖未来窗口

如果通过：

```text
与下一窗口 loadings 对齐 sign
```

就是未来泄漏。

可以：

```text
当前窗口 stateless deterministic orientation
```

或：

```text
只与上一窗口对齐 + checkpoint。
```

---

# 三十九、Kalman：Filter 可以，Smoother 默认不可以

单向：

```text
Kalman filter
```

在时序因子中可以 PIT-safe。

双向：

```text
RTS smoother
Kalman smoother
forward-backward smoothing
```

会用未来观测修正过去状态。

production factor：

```text
禁止。
```

---

# 四十、HMM / Markov 同理

允许：

```text
filtered probability
P(state_t | observations <= t)
```

不允许：

```text
smoothed probability
P(state_t | observations <= T)
```

如果使用：

```text
forward-backward
```

只能：

```text
research/offline diagnostic。
```

---

# 四十一、Kalman 当前家族专项测试

重点：

```text
ts_kalman_level
ts_kalman_innovation_z
ts_kalman_trend
ts_kalman_beta
ts_kalman_beta_change
ts_kalman_beta_uncertainty
```

要求：

```text
prefix invariance
future perturbation
missing predict-only covariance growth
full == segmented checkpoint
parameter q/r fixed
current observation innovation timing
```

---

# 四十二、GARCH / GJR / HAR

模型风险点：

```text
参数是否全样本估计
当前 shock 是否进入自身 volatility denominator
optimizer 是否失败后仍返回最后迭代
初始化 variance 是否含当前 observation
return/price semantic 是否搞错
```

---

# 四十三、当前 GARCH 正确方向

当前代码已经针对：

```text
standardized shock
```

使用：

```text
fit_seg = seg[:-1]
```

避免当前 return进入自身参数拟合。

R28 继续验证：

```text
真正 runtime每个 t 都满足
```

---

# 四十四、GARCH 输入类型不要靠统计启发式作为唯一门禁

当前还存在类似：

```text
sd(level) / sd(diff)
```

来判断是不是 price level。

这个可以作为：

```text
DQ warning
```

但生产语义应来自：

```text
typed FieldConcept / UnitContract
```

不能靠数据形状猜。

---

# 四十五、DMD / SSA / Hankel / spectral 属于高风险模型类

默认：

```text
RESEARCH / HIGH_COST
```

在真正通过：

```text
数值稳定性
window semantics
prefix invariance
missing topology
parameter feasibility
```

之前不要升生产。

---

# 四十六、当前 DMD 仍存在需要阻断认证的数值路径

当前 HEAD 可见：

```python
rho = np.abs(eig_vals) ** 2
```

可能在进入 log 之前 overflow。

以及：

```python
np.log(abs_b ** 2)
```

可能在进入 log 之前 underflow。

正确做法：

```text
log_rho = 2 * log(abs(lambda))
log_b2  = 2 * log(abs(b))
```

并显式处理：

```text
zero amplitude -> -inf
all energy -inf -> fail closed
```

这属于 R26 已识别类问题。

R28 的原则：

```text
如果 current HEAD 仍未修，
必须继续 BLOCKED，
不能因为 research-only 就忽略。
```

---

# 四十七、Wavelet / FFT factor 不是天然未来函数

如果：

```text
只对 trailing window [t-W+1,t]
```

做变换，

它可以是 causal。

但是：

```text
raw FFT primitive
two-sided filtering
full-sample spectral transform
```

不能作为生产 factor。

---

# 四十八、滤波器必须区分 causal filter vs zero-phase filter

允许：

```text
lfilter
one-sided EMA
causal convolution
```

禁止：

```text
filtfilt
zero-phase
symmetric centered filter
```

---

# 四十九、Matrix Profile / Sequence Anomaly

必须明确：

```text
query subsequence只能与历史 subsequence 比较
不能与未来 subsequence比较
```

exclusion zone：

```text
也不能让 query自己匹配自己。
```

---

# 五十、Path Signature 当前发现一个具体 bug

当前：

```text
_trailing_contiguous_xy()
```

在当前行 joint input 非 finite 时：

```text
先向前退 end
然后找到上一段 finite run
```

因此当前时点缺失：

```text
仍可能用旧历史段计算一个当前输出。
```

这不应发生。

---

# 五十一、Path Signature 修复

应改为：

```python
if n == 0 or not valid[-1]:
    return None
```

然后只取：

```text
以 current row t 结尾的 joint finite run。
```

---

# 五十二、Path Signature scale 必须基于同一个 joint run

当前 depth2 norm 的 scale 也必须：

```text
不能单独在带 NaN 的 x/y suffix 上 median/std
```

正确：

```text
先取得 joint trailing run
再在同一 joint run计算 robust scale
再算 signature
```

---

# 五十三、Path Signature 测试

新增：

```text
current x missing -> output[t] NaN
current y missing -> output[t] NaN
interior joint gap -> pre-gap points never bridged
future values changed -> output<=t unchanged
x/y columns reordered -> fail or correctly align
leadlag sign synthetic golden
translation invariance
scale normalization golden
```

---

# 五十四、Kernel Granger

必须证明：

```text
training block
test block
scaler
kernel bandwidth
lambda
```

全部仅从：

```text
training data
```

得到。

当前 blocked OOS 方向正确，

但仍需自动 future perturbation测试。

---

# 五十五、Residualized HSIC

需要：

```text
blocked cross-fitting
purge gap
training-only kernel hyperparameters
```

测试：

```text
future block改变不能影响过去输出。
```

---

# 五十六、研究 surrogate operator

例如 phase surrogate：

固定 seed并不等于：

```text
production mining适用。
```

继续：

```text
RESEARCH_DIAGNOSTIC
```

除非另有完整统计稳定性证据。

---

# 五十七、Expectile / Quantile Regression

必须区分：

```text
conditional quantile fit
expectile IRLS
quantile hit state
```

名称必须与真实数学对象一致。

对 predictive statistic：

```text
fit <= t-1
```

对 descriptive trailing regression：

```text
可以含 t
但不能冒充 forecast。
```

---

# 五十八、KNN / local model

动态 KNN / peer model 的常见泄漏：

```text
用全历史标准差归一化距离
用当前以后数据选邻居特征尺度
用未来 index membership
用未来行业分类
```

全部需要：

```text
as-of membership
historical scaler
decision-time cohort
```

---

# 五十九、Cross-sectional 模型的 PIT 不只看时间窗

在 `t` 的 cross-section：

```text
只能使用 t 时点真实可知的股票池
```

不能：

```text
用未来成分股名单
用事后退市过滤
用今天知道的历史成分回填
```

---

# 六十、Universe / membership 必须 as-of

所有：

```text
index_member
industry
concept
group
peer
relation
```

在 `t`：

```text
membership(t)
```

而不是：

```text
current membership retroactively applied。
```

---

# 六十一、Fundamental 模型

所有：

```text
expectation
revision
forecast
quality
growth
trend
```

必须使用：

```text
knowledge-time PIT
```

不能：

```text
按 report period直接回填
```

---

# 六十二、财务模型的“全样本标准差”尤其危险

例如：

```text
historical zscore
revision zscore
trend standardization
```

只允许：

```text
历史已知 filing/revision vintage
```

进入窗口。

---

# 六十三、分析师预期 / consensus

必须使用：

```text
historical consensus vintage
```

不能拿：

```text
今天数据库里的最终 consensus历史回填。
```

如果数据源不能证明 vintage：

```text
BLOCKED_NO_DATA / RESEARCH
```

而不是：

```text
假装 PIT。
```

---

# 六十四、Event Response learned model

常见泄漏：

```text
用未来事件结果学习历史 response
然后回填到过去。
```

要求：

```text
每个 t 的 response template
只能从 t 之前已完成的历史事件中学习。
```

---

# 六十五、State model initialization

禁止：

```text
用整段序列 mean/std/variance 初始化 t=0 state
```

允许：

```text
固定先验
第一可见观测
历史 warmup
```

---

# 六十六、Full-history operator 不等于 future leak

有些 causal recursive operator：

```text
需要从序列起点一路递推
```

可以 PIT-safe。

但必须标：

```text
FULL_HISTORY
```

且：

```text
incremental用 checkpoint
```

不能：

```text
随便从中途截断重新初始化。
```

---

# 六十七、Chunk Invariance

对 stateful/model operator：

```text
full-run
```

必须等于：

```text
chunk1
save checkpoint
chunk2 resume
```

否则：

```text
不能生产增量。
```

---

# 六十八、Prefix Invariance 与 Chunk Invariance 是两个不同测试

```text
Prefix invariance
→ 证明没有未来数据影响过去

Chunk/checkpoint invariance
→ 证明增量执行与全历史一致
```

两个都需要。

---

# 六十九、Current-row missing 统一语义

一般时序 factor：

```text
当前 required observation缺失
→ output[t] = NaN
```

除非算子明确定义：

```text
state prediction through missing observation
```

例如：

```text
Kalman predict-only state
```

此时必须：

```text
metadata明确
专项测试
```

不能所有 operator各自随便决定。

---

# 七十、禁止 stale substitution

扫描：

```text
while current invalid:
    t -= 1
```

如果最终把：

```text
旧值
```

写到当前 timestamp，

默认属于：

```text
STALE_SUBSTITUTION_HAZARD
```

除非明确定义：

```text
ffill/state carry-forward
```

---

# 七十一、Missing-gap compression

禁止默认：

```python
vals = x[np.isfinite(x)]
```

然后把 gap两边：

```text
重新当相邻时间点。
```

时序模型默认应：

```text
trailing contiguous
physical clock
或明确 event-time semantics。
```

---

# 七十二、每个算子至少一个测试：这是 R28 硬要求

最终：

```text
RETAINED_CANONICAL_WITH_ZERO_TESTS == 0
```

这里“测试”不是：

```text
catalog里有一行
import成功
metadata存在
```

而是：

```text
真正调用 operator runtime。
```

---

# 七十三、最基础 Per-Canonical Execution Test

新增：

```text
factor_engine/tests/operators/r28/test_all_canonicals_execute.py
```

由当前 registry动态参数化：

```python
@pytest.mark.parametrize("canonical", CURRENT_CANONICALS, ids=...)
def test_canonical_executes(canonical):
    ...
```

---

# 七十四、这个测试必须真实执行，不是结构 smoke

每个 canonical：

```text
构造合法 synthetic fixture
选择合法默认参数
实际调用 reference backend
检查：

返回类型
shape
index
dtype
至少在可观测区间出现非 NaN（除非 disposition明确是 negative test）
无异常
无 inf
```

---

# 七十五、不能给 guaranteed-NaN operator“测试通过”

如果：

```text
默认参数
+
合法输入
```

输出：

```text
全 NaN
```

默认：

```text
TEST_FAIL
```

除非：

```text
该算子本来就是 negative/denied tombstone。
```

---

# 七十六、默认参数必须可运行

对于 retained factor operator：

```text
DEFAULT_VALID_CONFIG_EXECUTES == true
```

默认参数不能：

```text
自身违反 min sample
自身违反 window/rank
自身 guaranteed NaN。
```

---

# 七十七、每个参数化算子至少边界测试

至少：

```text
default
one interior valid
one invalid
```

如果有 relational constraints：

```text
边界合法
边界非法
```

---

# 七十八、测试不是都要求一模一样数量

最低：

```text
每 retained canonical >= 1 real execution test
```

但高风险族需要更强。

---

# 七十九、风险分级

## LOW

```text
elementwise math
simple boolean
basic arithmetic
```

至少：

```text
execution
NaN policy
domain boundary
```

---

## MEDIUM

```text
rolling
cross-section
group
technical
fundamental transforms
```

至少：

```text
execution
prefix
missing
parameter
golden/metamorphic
```

---

## HIGH

```text
regression
PCA
Kalman
GARCH
DMD
entropy
kernel
stateful
intraday clock
event-response
fundamental PIT
relation model
```

至少：

```text
execution
prefix invariance
future perturbation
parameter boundaries
missing topology
reference/golden
chunk/backend/source as applicable
```

---

# 八十、测试覆盖矩阵必须按 canonical，而不是按文件

不要：

```text
“这个模块有20个测试，所以里面80个算子应该都测到了。”
```

必须：

```text
canonical -> test node ids
```

---

# 八十一、R28_CANONICAL_TEST_COVERAGE.csv

字段：

```text
canonical
source_file
source_symbol
surface
production_certification
mining_role
disposition

test_count
execution_test_count
semantic_test_count
prefix_test_count
future_perturbation_test_count
missing_test_count
parameter_test_count
backend_parity_test_count
source_pit_test_count
chunk_test_count

latest_status
failure_count
```

---

# 八十二、Pytest node id也要保存

例如：

```text
tests/operators/r28/test_all_canonicals_execute.py::test_canonical_executes[ts_mean]
```

证据中直接列：

```text
canonical -> node_ids
```

---

# 八十三、测试结果必须是当前代码跑出来的

Artifact header：

```json
{
  "git_sha": "...",
  "canonical_set_digest": "...",
  "test_suite_digest": "...",
  "python_version": "...",
  "platform": "...",
  "numpy_version": "...",
  "pandas_version": "...",
  "polars_version": "...",
  "duckdb_version": "...",
  "generated_at": "..."
}
```

---

# 八十四、证据不能复用旧 HEAD

如果：

```text
artifact.git_sha != git rev-parse HEAD
```

直接：

```text
STALE
```

CI fail。

---

# 八十五、Canonical-set digest

fresh process：

```text
load_all
sorted canonical list
sha256
```

所有 R28 artifact必须：

```text
同一个 canonical_set_digest。
```

---

# 八十六、当前旧 evidence 数量不一致必须自动检测

新增：

```text
scripts/audit_r28_artifact_coherence.py
```

发现：

```text
catalog count != per-canonical evidence count
```

立即 fail。

---

# 八十七、GitHub-visible evidence 目录

固定：

```text
factor_engine/docs/evidence/r28/
```

不要放：

```text
output/
logs/
.pytest_cache/
htmlcov/
```

因为这些现有 `.gitignore` 会忽略。

---

# 八十八、必须提交的 tracked artifacts

```text
factor_engine/docs/evidence/r28/
├── R28_OPERATOR_INVENTORY.csv
├── R28_OPERATOR_INVENTORY.json
├── R28_CANONICAL_TEST_COVERAGE.csv
├── R28_CANONICAL_TEST_COVERAGE.json
├── R28_PER_CANONICAL_TEST_RESULTS.csv
├── R28_PER_CANONICAL_TEST_RESULTS.json
├── R28_FORBIDDEN_OPERATOR_AUDIT.json
├── R28_STATIC_LOOKAHEAD_SCAN.csv
├── R28_STATIC_LOOKAHEAD_SCAN.json
├── R28_MODEL_CAUSALITY_MATRIX.csv
├── R28_MODEL_CAUSALITY_MATRIX.json
├── R28_MODEL_WALK_FORWARD_RESULTS.csv
├── R28_MODEL_WALK_FORWARD_RESULTS.json
├── R28_PREFIX_INVARIANCE_RESULTS.csv
├── R28_PREFIX_INVARIANCE_RESULTS.json
├── R28_FUTURE_PERTURBATION_RESULTS.csv
├── R28_FUTURE_PERTURBATION_RESULTS.json
├── R28_MISSING_TOPOLOGY_RESULTS.csv
├── R28_PARAMETER_DOMAIN_RESULTS.csv
├── R28_BACKEND_PARITY_SUMMARY.csv
├── R28_SOURCE_PIT_SMOKE_RESULTS.csv
├── R28_PYTEST_JUNIT.xml
├── R28_PYTEST_SUMMARY.json
├── R28_ARTIFACT_MANIFEST.json
└── R28_FINAL_ACCEPTANCE_REPORT.md
```

---

# 八十九、测试结果不要提交巨大原始日志

GitHub里需要：

```text
可审计
可定位
体积可控
```

所以：

```text
JUnit XML
per-canonical JSON/CSV
summary
failure digest
```

即可。

---

# 九十、失败 traceback

每个失败行记录：

```text
exception_type
message
traceback_digest
traceback_head
```

不要把：

```text
几百MB raw log
```

直接提交。

---

# 九十一、`.gitignore` 强验证

新增：

```text
scripts/audit_r28_evidence_tracking.py
```

对所有 expected artifact：

```bash
git check-ignore <path>
```

必须：

```text
NOT ignored
```

---

# 九十二、还要验证已 tracked

在 final acceptance：

```bash
git ls-files --error-unmatch factor_engine/docs/evidence/r28/<file>
```

每个必须成功。

---

# 九十三、不要提交 `.coverage`

当前 `.gitignore` 已忽略：

```text
.coverage
htmlcov/
```

没关系。

R28 关注：

```text
operator semantic/test coverage
```

而不是单纯 Python line coverage。

---

# 九十四、如果需要 line coverage

可生成：

```text
coverage JSON
```

再提炼为：

```text
R28_OPERATOR_LINE_COVERAGE.json
```

放 evidence目录。

不要依赖：

```text
htmlcov。
```

---

# 九十五、测试目录

新增：

```text
factor_engine/tests/operators/r28/
```

至少：

```text
test_all_canonicals_execute.py
test_all_canonicals_have_tests.py
test_forbidden_operator_surface.py
test_no_random_factor_terminals.py
test_no_future_operator_terminals.py
test_static_lookahead_patterns.py

test_model_prefix_invariance.py
test_model_future_perturbation.py
test_model_fit_cutoff.py
test_model_scaler_fit_cutoff.py
test_model_label_maturity.py
test_model_purged_walk_forward.py
test_model_no_bidirectional_smoothing.py
test_model_chunk_checkpoint_equivalence.py

test_current_missing_fail_closed.py
test_missing_gap_topology.py

test_alias_surface_leak.py
test_research_not_mineable.py
test_diagnostic_not_mineable.py
test_internal_not_mineable.py

test_parameter_domain.py
test_backend_claims.py
test_source_context_pit.py

test_path_signature_time_semantics.py
test_dmd_numeric_log_path.py
test_garch_causality.py
test_kalman_filter_causality.py
test_panel_model_walk_forward.py
test_ar_regression_walk_forward.py
test_kernel_granger_oos.py
test_hsic_crossfit.py
test_matrix_profile_history.py
```

---

# 九十六、不要让 generated test 掩盖人工语义测试

`test_all_canonicals_execute`：

```text
只解决“没有任何测试”的底线。
```

它不能替代：

```text
数学/市场语义测试。
```

---

# 九十七、Golden / Reference 测试

高风险模型：

```text
用小型手算或 independent reference
```

不要：

```text
拿同一个实现算出来的值做 expected。
```

---

# 九十八、模型测试应使用 synthetic DGP

例如 AR：

```text
x_t = phi*x_{t-1} + eps
```

验证：

```text
prior coefficient接近真实 phi
future perturbation不影响过去
```

---

# 九十九、GARCH synthetic

模拟：

```text
已知 GARCH(1,1)
```

验证：

```text
variance recursion timing
shock denominator
parameter stationarity
```

---

# 一百、Kalman synthetic

已知：

```text
latent level + noise
```

验证：

```text
filter one-sided
missing predict step
future改变不影响过去 filtered state
```

---

# 一百零一、PCA synthetic

构造：

```text
latent factor + idiosyncratic noise
```

验证：

```text
prior loading
prior residual
current row不参与 fit
```

---

# 一百零二、Supervised synthetic

构造：

```text
x_t
label_t = future H return
```

人工在未来段加入：

```text
巨大冲击
```

验证：

```text
在 label成熟之前
模型绝对不能看到它。
```

---

# 一百零三、Purge test

H=5：

```text
decision t
```

训练中：

```text
origin > t-5
```

全部：

```text
禁止。
```

---

# 一百零四、Alias 测试

一个 canonical可能通过：

```text
legacy alias
compat alias
DSL alias
```

重新漏入 mining。

所以：

```text
每个 alias也要解析最终 canonical
```

验证：

```text
final policy不能比 canonical更宽。
```

---

# 一百零五、Policy monotonicity

必须：

```text
alias_safety <= canonical_safety
```

不允许：

```text
canonical research
alias daily
```

或：

```text
canonical denied
alias mining-visible。
```

---

# 一百零六、Deprecated alias不能独立搜索

比如：

```text
motif vs discord同一个实现
GARCH legacy forecast alias
AR legacy aliases
```

不能让 mining：

```text
把同一个因子当两个独立候选。
```

---

# 一百零七、重复算子全量审计

计算：

```text
semantic hash
source implementation hash
parameter contract
output contract
```

发现 exact duplicates：

```text
DELETE_DUPLICATE
```

保留：

```text
canonical + compatibility alias。
```

---

# 一百零八、不要因为参数不同就当“不同算子”

同一个 kernel：

```text
window=5
window=10
```

是：

```text
同一 canonical参数空间。
```

不是：

```text
两个 operator。
```

---

# 一百零九、Useless operator 审核标准

以下任一成立应优先删除：

```text
永远全 NaN
合法默认参数不可运行
永远常数
输出只广播同一值且无 state用途
没有任何量化/统计解释
与另一个 canonical exact duplicate
只有随机/未来信息
只有 raw matrix utility
无法满足 PIT
无真实数据源
名称与实现严重不一致且不值得修
```

---

# 一百一十、不能“为了数量”保留算子

目标不是：

```text
1362 → 1500
```

而是：

```text
保留的每一个都有真实用途。
```

如果最后：

```text
1362 → 1180
```

但全部更干净，

是好结果。

---

# 一百一十一、模型类 high-cost 不等于 useless

例如：

```text
GARCH
DMD
HSIC
kernel Granger
PCA
```

如果：

```text
量化语义合理
PIT安全
测试通过
```

可以：

```text
RESEARCH / HIGH_COST lane
```

不要因为慢就删。

---

# 一百一十二、ResearchTool 的定义要严格

真正 research：

```text
统计检验
诊断
矩阵输出
offline model analysis
assumption-heavy estimator
```

而不是：

```text
“还没测试好所以先扔 research”
```

“没测好”应该：

```text
BLOCKED
```

---

# 一百一十三、Internal 的定义

Internal：

```text
raw helper
intermediate kernel
backend utility
source transform
```

不能：

```text
被普通 DSL直接 author。
```

---

# 一百一十四、Mining Consumer Integration Audit

必须真实测试：

```text
AlphaProbe
AlphaMiner
FactorMiner
CogAlpha
AlphaSage
EvoAlpha
AlphaCFG
QuantaAlpha
```

---

# 一百一十五、消费者必须只读同一个 authority

禁止：

```text
A算法读 DAILY_CANONICALS
B算法读 OperatorRegistry
C算法读 backend allowlist
D算法读 DirectUse
```

最终：

```text
一个 production mining admission authority。
```

---

# 一百一十六、负面 consumer test

对：

```text
Lead
next
rand_uniform
bfill
diagnostic in-sample model
research-only Kalman
research-only GARCH
DMD research
unsafe trig
```

逐个：

```text
所有 default miner discovery API
必须返回不存在。
```

---

# 一百一十七、Positive consumer test

对真正 production operator：

```text
ts_mean
ts_std
rank
zscore
...
```

确认：

```text
至少一个合法 grammar slot可达。
```

---

# 一百一十八、STATE / CONDITION / EVENT

这些不一定 terminal。

要测试：

```text
legal position可达
terminal位置不可达。
```

---

# 一百一十九、Model role建议

模型类不要一刀切 terminal。

例如：

```text
vol regime estimate
state probability
change score
```

可能应该：

```text
STATE / CONDITION
```

而不是：

```text
ALPHA terminal。
```

---

# 一百二十、模型输出必须明确“预测什么时点”

Metadata新增：

```text
fit_cutoff
score_time
forecast_horizon
label_horizon
availability_lag
in_sample
```

---

# 一百二十一、ModelTimingContract

建议：

```python
@dataclass(frozen=True)
class ModelTimingContract:
    model_kind: str

    fit_cutoff_offset: int
    score_offset: int
    forecast_horizon: int

    label_horizon: int | None
    purge_bars: int
    embargo_bars: int

    scaler_fit_cutoff_offset: int
    hyperparam_fit_cutoff_offset: int

    state_filtering: str  # filter / smoother / none
```

---

# 一百二十二、模型类没有 TimingContract 不准 production

```text
MODEL_OPERATOR_WITHOUT_TIMING_CONTRACT == 0
```

---

# 一百二十三、未来信息测试不能靠字符串判断

Static scan只发现：

```text
明显模式。
```

最终必须：

```text
行为测试。
```

---

# 一百二十四、黑箱 Causality Oracle

对任意 operator：

```python
def future_perturbation_oracle(op, inputs, cutoff):
    base = op(inputs)
    modified = perturb_after(inputs, cutoff)
    changed = op(modified)
    assert_equal(base[:cutoff+1], changed[:cutoff+1])
```

---

# 一百二十五、Causality Oracle 自动覆盖

对所有：

```text
time-series
model
stateful
intraday→daily
fundamental PIT
event
```

自动参数化。

---

# 一百二十六、Cross-section oracle

未来日期改变：

```text
不能影响过去 cross-section。
```

同日：

```text
股票顺序 permutation
```

必须：

```text
output按股票重排后相同。
```

---

# 一百二十七、Group oracle

组标签顺序改变：

```text
不应影响输出。
```

但：

```text
membership改变
```

只能影响对应 as-of时点。

---

# 一百二十八、Column permutation

大量模型依赖矩阵。

必须：

```text
instrument column permutation
```

后：

```text
结果相应 permutation
```

不能：

```text
股票 A/B 静默错配。
```

---

# 一百二十九、Multi-input alignment

任何多输入模型：

```text
y
x1
x2
benchmark
market_state
group
```

必须：

```text
严格 index/column alignment
```

或通过 typed adapter明确 reindex。

不能：

```text
to_numpy后按位置盲配。
```

---

# 一百三十、参数 silently cast

全库扫描：

```text
int(window)
int(rank)
int(k)
str(side)
bool(...)
```

如果外层 ParamSpec没有严格绑定：

```text
3.9 → 3
```

会让不同参数搜索结果变成同一个因子。

必须：

```text
boundary严格拒绝非法类型。
```

---

# 一百三十一、**kwargs swallowing

扫描：

```python
def calc(..., **_):
```

确认：

```text
未知参数不会被悄悄吞掉。
```

public operator调用：

```text
unknown kwarg -> error
```

---

# 一百三十二、模型参数 feasibility

例如：

```text
rank <= dim
q < window
n_components < active rank
label_horizon < available training window
purge_gap feasible
```

搜索空间和 runtime：

```text
同一份 constraint authority。
```

---

# 一百三十三、当前 model source逐类清单

R28 至少完整遍历以下 module family：

```text
cleaned_operators/regression_models.py

cleaned_operators/ts_model/
  dynamic_regression.py
  ar_meanrev.py
  state_space.py
  volatility.py
  complexity.py
  wavelet_spectral.py
  sequence_anomaly.py
  path_signature.py

cleaned_operators/cross_section/
  panel_model.py
  robust_cs.py
  peer_ops.py

cleaned_operators/
  advanced_information.py
  advanced_quantile_dynamics.py
  advanced_expectile.py
  conditional_dependence.py

  markov_dynamics.py
  state_geometry.py
  first_passage.py
  event_response.py
  distribution_break.py
  dynamic_knn.py
  local_lyapunov.py
  recurrence_analysis.py

  hankel.py
  spectral.py
  spectral_ext.py
  cross_spectrum.py
  dmd.py
  research_spectral.py

  hvg_ext.py
  rqa_ext.py
  glr_change.py
  multifractal.py
  multifractal_asym.py
  intrinsic_dimension.py

  cs_state_ops.py
  research_transform.py

fundamental/
  expectation_v2.py
  revision / trend / quality / forecast相关
```

---

# 一百三十四、不要只审名字含 model 的

很多 model-like算子名字可能没有：

```text
model
```

比如：

```text
entropy
state persistence
local stability
change score
```

所以还要按：

```text
implementation primitives
```

扫描：

```text
lstsq
pinv
svd
eig
optimize
kernel matrix
quantile fit
polyfit
covariance eigendecomposition
nearest neighbors
```

---

# 一百三十五、全 canonical Source Symbol Inventory

R28 inventory每行至少：

```text
canonical
aliases
source_file
source_symbol
category
surface
status

role
terminal_allowed
production_certified

deterministic
pit_safe_claim
stateful

input_grain
output_grain
input_units
output_unit

params
default_params

backends

test_count
final_disposition
blockers
```

---

# 一百三十六、Current catalog `pit_safe=false` 不得被忽略

当前 catalog里仍存在：

```text
surface=daily
pit_safe=false
```

的条目。

R28必须逐个解释：

```text
A. catalog stale
B. operator确实不 PIT
C. policy字段生成错误
D. daily authoring与certification正交
```

不能：

```text
一律批量改 true。
```

---

# 一百三十七、PIT evidence必须来自行为

最终：

```text
pit_safe_claim
```

和：

```text
prefix/future test
```

不一致：

```text
行为证据优先。
```

---

# 一百三十八、测试失败不得通过改 status掩盖

禁止：

```text
test fail
→ 把 production 改 research
→ 宣称修复。
```

如果算子仍有量化价值：

```text
修代码。
```

只有：

```text
本来就是真 research utility
```

才保留 research。

---

# 一百三十九、删除算子必须删干净

DELETE后检查：

```text
implementation
registry
surface
catalog
DSL
aliases
mining
recipe
backend emitter
docs reference
cold-start recipes
tests
```

只保留：

```text
migration tombstone
```

如果需要兼容。

---

# 一百四十、Cold-start / Recipe 清理

被删除/禁用 operator：

```text
不能残留在冷启动库
不能残留 default recipe
不能残留 AI prompt grammar
```

---

# 一百四十一、Operator existence gate

对所有 recipe：

```text
每个 operator必须 resolve 到 retained canonical。
```

---

# 一百四十二、每个 retained operator至少一个“合法组合”

对于：

```text
STATE
EVENT
CONDITION
INTERMEDIATE
```

即使不能 terminal，

也要有：

```text
合法 AST composition test。
```

---

# 一百四十三、ProductionFactor真实 recipe smoke

真正 terminal：

```text
Source → operator DAG → output
```

至少跑一个真实/fixture recipe。

---

# 一百四十四、Source-dependent operator

例如：

```text
fundamental
shareholder
index
relation
intraday
```

测试中必须带：

```text
对应真实 source contract。
```

不能只喂：

```text
随便一个同名 DataFrame列
```

就宣称 production usable。

---

# 一百四十五、无数据字段就别硬留

如果现有 A/US 数据都没有 source：

```text
BLOCKED_NO_DATA
```

如果未来也没有计划：

```text
DELETE_NO_DATA。
```

---

# 一百四十六、A股/美股 Context Matrix

每 canonical：

```text
ashare
us
```

分别 disposition。

不要：

```text
A股可用
→ 自动 global可用。
```

---

# 一百四十七、Intraday model

分钟模型还要额外：

```text
physical session clock
missing slot
timezone
auction
early close
```

这些沿用 R26 gate。

---

# 一百四十八、测试 evidence不要假绿

以下不算 semantic pass：

```text
function returned
shape matched
没有 exception
```

还必须：

```text
value semantics有可验证性质。
```

---

# 一百四十九、Metamorphic tests

可以大量自动覆盖：

```text
scale invariance
translation invariance
sign symmetry
permutation equivariance
bounded output
monotonicity
zero-input behavior
constant-input behavior
```

按 operator contract选择。

---

# 一百五十、Numeric finite tests

极端输入：

```text
1e-300
1e300
near-zero denominator
constant series
near-singular design
```

不能：

```text
silent inf
overflow到错误 finite值。
```

---

# 一百五十一、Model solver

optimizer：

```text
success=False
```

必须：

```text
NaN/fail closed
```

不能：

```text
拿最后 iterate当结果。
```

---

# 一百五十二、随机优化算法

如果某 estimator内部使用：

```text
randomized SVD
random init
stochastic optimizer
```

必须：

```text
fixed deterministic seed
或改 deterministic solver
```

否则：

```text
research only / delete。
```

---

# 一百五十三、Reproducibility

所有 production retained：

```text
same input
same params
same code SHA
```

必须：

```text
repeatable。
```

---

# 一百五十四、Thread/process order independence

并行执行后：

```text
结果不能依赖 task执行顺序。
```

这也属于 R27/R28交叉验收。

---

# 一百五十五、Current evidence结果提交策略

不要每次手工复制测试结果。

新增：

```text
scripts/generate_r28_evidence.py
```

顺序：

```text
1. fresh process enumerate canonicals
2. run test suite
3. parse JUnit
4. map test node IDs -> canonical
5. generate all matrices
6. bind git SHA
7. validate artifacts
8. final report
```

---

# 一百五十六、Artifacts必须最后生成

如果：

```text
生成 evidence
之后又改代码
```

evidence立即 stale。

因此：

```text
代码完成
→ tests完成
→ 最后生成 evidence
→ fresh process validate
```

---

# 一百五十七、Artifact manifest

`R28_ARTIFACT_MANIFEST.json`：

```text
path
sha256
git_sha
canonical_set_digest
generated_at
generator_version
```

---

# 一百五十八、Evidence 自身 CI

新增：

```text
test_r28_evidence_current_head
test_r28_evidence_all_canonicals_present
test_r28_evidence_no_extra_deleted_canonicals
test_r28_all_retained_have_test
test_r28_all_required_files_tracked
```

---

# 一百五十九、Current HEAD变化

如果执行本文件时 HEAD 已不是：

```text
37008c7...
```

不要硬按1362。

必须：

```text
重新 fresh enumerate
```

并在最终报告中：

```text
写真实 HEAD和真实 count。
```

---

# 一百六十、R28_OPERATOR_INVENTORY完整性

硬 gate：

```text
set(inventory canonical)
==
set(OperatorRegistry canonicals)
==
set(current catalog canonicals)
```

允许 catalog derived artifact重新生成。

---

# 一百六十一、Aliases不计 canonical

不要把：

```text
alias
```

当新的测试缺口 canonical。

但 alias需要：

```text
policy leak test。
```

---

# 一百六十二、Research registry单独盘点

除 production OperatorRegistry外：

```text
ResearchToolRegistry
```

也要列清楚。

但最终矩阵分别：

```text
factor canonical
research tool
```

不要混成一个数。

---

# 一百六十三、Unsafe registry

unsafe只能：

```text
explicit opt-in。
```

默认 discovery：

```text
0。
```

---

# 一百六十四、Model audit自动分类器

脚本根据：

```text
source path
category
tags
AST calls
canonical name
```

标：

```text
is_model_like
risk_level
```

---

# 一百六十五、Model Causality Matrix字段

```text
canonical
model_family

descriptive_or_predictive

fit_cutoff
feature_cutoff
score_time

label_used
label_horizon
label_maturity_enforced
purge
embargo

scaler_scope
pca_scope
hyperparam_scope

filter_or_smoother

prefix_invariance
future_perturbation
walk_forward
chunk_equivalence

final_model_pit_status
blocker
```

---

# 一百六十六、对“模型有当前样本参与训练”不要简单叫 lookahead

要精确区分：

```text
in-sample descriptive
vs
out-of-sample predictive
```

前者：

```text
没有未来信息
```

但：

```text
不能当真正的创新/预测误差。
```

所以风险是：

```text
semantic misuse
```

而不一定是：

```text
future leakage。
```

---

# 一百六十七、真正 future leak

例如：

```text
scaler用全样本
PCA用全样本
模型用t+1
label未成熟
smoother用未来
center rolling
```

必须：

```text
DELETE_NONCAUSAL
或重写 causal。
```

---

# 一百六十八、能够 causal rewrite 的优先改

例如：

```text
full sample scaler
→ rolling/expanding historical scaler

full sample PCA
→ rolling PCA t-1

smoother
→ filter

bfill
→ NaN / causal ffill（如果语义允许）

current-fit forecast
→ prior-window forecast
```

---

# 一百六十九、改不了的直接踢

如果算法定义本身必须：

```text
看未来
```

例如真正：

```text
centered two-sided feature
future extrema confirmation with output backdated to pivot date
offline smoother
```

不能：

```text
硬改标签。
```

直接：

```text
RESEARCH_DIAGNOSTIC
或 DELETE_NONCAUSAL。
```

---

# 一百七十、Confirmed pivot特别注意

如果：

```text
需要右侧 N bars确认 pivot
```

可以有两种合法语义：

### A
在 pivot发生日回填信号：

```text
未来函数
禁止。
```

### B
在 N bars以后确认日输出：

```text
causal delayed event
允许。
```

必须：

```text
timestamp是 confirmation time
```

---

# 一百七十一、ZigZag / swing 同类

任何：

```text
用未来价格确认拐点
```

必须：

```text
确认时点输出
```

不能：

```text
回写拐点原时点。
```

---

# 一百七十二、Pattern recognition

图形形态如果：

```text
只用 trailing completed bars
```

可以。

如果：

```text
模式终点以后数据参与确认
然后回填到模式起点
```

不可以。

---

# 一百七十三、Model training clock必须文档化

每个模型类 operator：

docstring至少写：

```text
At output time t:
training rows = [...]
scoring row = [...]
labels matured through = [...]
```

---

# 一百七十四、测试通过后自动生成这段 timing summary

尽量：

```text
从 ModelTimingContract生成 docs
```

减少漂移。

---

# 一百七十五、测试不允许 xfail逃避

对 production target：

```text
xfail
skip
```

默认：

```text
不算 pass。
```

---

# 一百七十六、optional dependency

如果：

```text
TA-Lib
SciPy
```

可选，

至少：

```text
reference fallback path
```

必须有测试。

---

# 一百七十七、只有 optional backend的 operator

如果环境没依赖：

```text
明确 BLOCKED_OPTIONAL_DEP
```

不能：

```text
无测试却算通过。
```

---

# 一百七十八、三后端不要求全有

继续遵守：

```text
至少一个正确 backend
即可 usable。
```

但：

```text
凡是宣称某 backend支持
就必须有 backend parity证据。
```

---

# 一百七十九、Backend claim > evidence

如果 catalog写：

```text
pandas
polars
sql
```

实际只有：

```text
pandas测试
```

那：

```text
Polars/SQL claim必须降级。
```

---

# 一百八十、Canonical baseline test使用 reference backend

优先：

```text
pandas/numpy semantic reference
```

然后 native parity。

---

# 一百八十一、测试 fixture生成器

新增：

```text
tests/operators/r28/fixture_factory.py
```

根据 metadata：

```text
price
return
volume
amount
high
low
open
close
group
event
fundamental
minute
```

生成：

```text
语义合理数据。
```

---

# 一百八十二、不要所有参数都喂 close

这是旧 smoke最容易假绿的问题。

例如：

```text
volume input
```

不能：

```text
拿close代替。
```

---

# 一百八十三、Model fixture必须贴近模型输入

GARCH：

```text
return
```

PCA：

```text
cross-sectional return matrix
```

fundamental：

```text
PIT filing series
```

event：

```text
sparse event process
```

---

# 一百八十四、NaN fixtures

每个 high-risk operator至少：

```text
leading NaN
interior gap
current NaN
long suspension
```

---

# 一百八十五、Panel model cross-sectional missing

测试：

```text
一只股票停牌
```

不能：

```text
让整个截面全部 NaN。
```

除非模型数学上必须。

---

# 一百八十六、Model Feature/Label separation

mining grammar：

```text
LabelPanel
```

绝不能作为普通：

```text
FeatureExpr
```

输入。

---

# 一百八十七、Forward-return label不能被 LLM自动构成 factor input

即使：

```text
数据系统里有未来收益标签
```

它只能：

```text
training/evaluation subsystem
```

不能：

```text
FactorEngine production feature source。
```

---

# 一百八十八、Target leakage scanner

对字段名 / semantic type扫描：

```text
future_return
forward_return
target
label
next_return
lead_return
```

进入 feature DAG：

```text
hard fail。
```

---

# 一百八十九、但不能只靠名字

typed source contract必须：

```text
is_label=True
```

才是权威。

---

# 一百九十、R28 测试结果报告按 family总结

最终报告包括：

```text
Basic math
Time series
Cross-sectional
Group
Technical
Candle/pattern
A-share rules
Intraday
Fundamental
Shareholder
Relation
Event
Model/regression
Stateful
Spectral/topology
Research
Unsafe
Legacy/Internal
```

---

# 一百九十一、报告必须同时给 count

每组：

```text
total
production
high_cost
state/intermediate
research
forbidden
deleted
blocked
tests_passed
tests_failed
```

---

# 一百九十二、不要只给总体 99%

如果：

```text
1361/1362 pass
```

那个 1 个如果是：

```text
future leak
```

仍然：

```text
整体不通过。
```

---

# 一百九十三、Hard blocker优先级

### P0

```text
future leak
label leak
random factor terminal
wrong current-time semantics
model future normalization
unsafe alias leak
operator with no test
stale evidence
```

### P1

```text
parameter domains
missing topology
backend claim drift
numerical instability
test semantics weak
```

### P2

```text
docs
naming
performance
```

---

# 一百九十四、测试覆盖不是“代码行覆盖率”

R28关注：

```text
semantic behavior coverage。
```

一个 operator 100% line coverage：

```text
也可能前视。
```

---

# 一百九十五、Current path_signature bug应作为 R28 P0

编号建议：

```text
R28-P0-001
```

内容：

```text
current-row missing can fall back to previous finite run
```

---

# 一百九十六、Current evidence mismatch作为 R28 P0

```text
R28-P0-002
```

```text
catalog=1362
R19=1430
R23=1436
```

---

# 一百九十七、Forbidden physical removal audit作为 R28 P0

```text
R28-P0-003
```

不能只证明：

```text
policy deny。
```

还要证明：

```text
runtime/mining/backend无入口。
```

---

# 一百九十八、Per-canonical zero-test audit作为 R28 P0

```text
R28-P0-004
```

现有 repo evidence没有证明：

```text
1362当前 canonical每个>=1测试。
```

必须新生成。

---

# 一百九十九、Model causality universal gate作为 R28 P0

```text
R28-P0-005
```

所有模型类：

```text
prefix + future perturbation。
```

---

# 二百、Label maturity gate作为 R28 P0

```text
R28-P0-006
```

Supervised models必须：

```text
typed label maturity。
```

---

# 二百零一、Diagnostic mining exclusion作为 R28 P0

```text
R28-P0-007
```

metadata写：

```text
diagnostic_only
```

不够。

所有 consumer discovery真实测试：

```text
0 exposure。
```

---

# 二百零二、Research mining exclusion

```text
R28-P0-008
```

同理。

---

# 二百零三、Random research reproducibility

```text
R28-P1-009
```

固定 seed surrogate：

```text
不受 global RNG影响。
```

---

# 二百零四、Model CV scanner

```text
R28-P0-010
```

production model：

```text
random CV count = 0。
```

---

# 二百零五、Smoother scanner

```text
R28-P0-011
```

production factor：

```text
two-sided smoother count = 0。
```

---

# 二百零六、Full-sample fit scanner

```text
R28-P0-012
```

任何：

```text
全样本 fit后回填历史
```

count：

```text
0。
```

---

# 二百零七、Current model-family coverage

最终必须明确：

```text
MODEL_CANONICALS_TOTAL
MODEL_CANONICALS_TESTED
MODEL_CAUSALITY_PASS
MODEL_RESEARCH_ONLY
MODEL_BLOCKED
MODEL_DELETED
```

---

# 二百零八、不要把“research-only”当作免检

即使 research：

```text
也至少要知道它数学上算的是什么。
```

因为：

```text
以后可能有人手工调用。
```

但 production-level PIT gate可以：

```text
not applicable
```

只要明确：

```text
不进入 factor mining。
```

---

# 二百零九、Research未来函数可以保留吗？

如果用途是：

```text
offline retrospective diagnostic
```

可以保留在：

```text
ResearchToolRegistry
```

但必须：

```text
名字明确
explicit opt-in
绝不出现在factor surface。
```

---

# 二百一十、如果根本没用就删

不要为了：

```text
“研究可能有用”
```

把明显垃圾留着。

---

# 二百一十一、最终 public surface应该更干净

理想：

```text
Production authoring
Contextual/High-cost
Research tools
Internal helpers
Forbidden tombstones
```

边界清晰。

---

# 二百一十二、代码 AI 执行顺序

必须按：

```text
Phase 1  Fresh Inventory
Phase 2  Forbidden/Dead Cleanup
Phase 3  Static Lookahead Scan
Phase 4  Model Causality Audit
Phase 5  Per-Canonical Test Generation
Phase 6  Family Semantic Tests
Phase 7  Mining Consumer Negative Tests
Phase 8  Source/Backend/Parameter Tests
Phase 9  Full Suite
Phase 10 Evidence Generation
Phase 11 Fresh-process Acceptance
```

---

# 二百一十三、Phase 1：Fresh Inventory

运行：

```text
fresh Python process
ensure_cleaned_loaded
enumerate registry
```

记录：

```text
actual HEAD
actual canonical count
actual aliases
actual surfaces
```

---

# 二百一十四、Phase 2：Forbidden Cleanup

对：

```text
PERMANENTLY_FORBIDDEN
unsafe
legacy
duplicate
no-data
```

逐个决定：

```text
delete / tombstone / research / internal。
```

---

# 二百一十五、Phase 3：Static Scan

结果写：

```text
R28_STATIC_LOOKAHEAD_SCAN.csv
```

每条：

```text
file
line
symbol
pattern
canonical(s)
severity
review_status
resolution
```

---

# 二百一十六、Static scan不能留未处置项

最终：

```text
UNREVIEWED_STATIC_HAZARD == 0
```

---

# 二百一十七、Phase 4：Model audit

对每个 model-like canonical：

```text
写 ModelTimingContract。
```

---

# 二百一十八、Phase 5：Per-canonical tests

自动生成：

```text
每 canonical至少1个真实 node id。
```

---

# 二百一十九、Phase 6：Family semantic tests

高风险族：

```text
专项 golden + metamorphic。
```

---

# 二百二十、Phase 7：Consumer tests

所有挖掘算法：

```text
production discovery
```

不得暴露：

```text
forbidden/research/diagnostic/internal。
```

---

# 二百二十一、Phase 8：Source/backend/parameter

按声明测试，

不声明的 backend：

```text
不用强求。
```

---

# 二百二十二、Phase 9：Full suite

建议：

```bash
pytest factor_engine/tests/operators/r28 -q --junitxml=...
```

以及：

```text
现有 factor_engine完整回归。
```

---

# 二百二十三、Phase 10：Evidence

只在：

```text
所有代码最终稳定
```

后生成。

---

# 二百二十四、Phase 11：Fresh process

重新：

```text
import
enumerate
validate digest
validate artifacts
```

防止：

```text
旧进程 registry状态污染。
```

---

# 二百二十五、Final Acceptance Report必须列失败项

如果仍有 blocker：

```text
明确列 canonical
```

不要：

```text
写“大部分通过”。
```

---

# 二百二十六、R28 Hard Gates

最终必须全部 TRUE：

```text
R28_CURRENT_HEAD_BOUND
R28_CANONICAL_SET_COHERENT
R28_EVERY_CANONICAL_DISPOSITIONED
R28_EVERY_RETAINED_CANONICAL_HAS_REAL_TEST
R28_ZERO_STALE_CANONICAL_EVIDENCE

R28_PERMANENTLY_FORBIDDEN_RUNTIME_SURFACE_ZERO
R28_RANDOM_FACTOR_TERMINALS_ZERO
R28_FUTURE_FUNCTION_TERMINALS_ZERO
R28_NONCAUSAL_FILL_TERMINALS_ZERO
R28_UNSAFE_ALIAS_ESCAPE_ZERO

R28_STATIC_LOOKAHEAD_HAZARDS_REVIEWED
R28_FULL_SAMPLE_MODEL_FITS_ZERO
R28_CENTERED_TWO_SIDED_MODEL_FEATURES_ZERO
R28_RANDOM_TIME_SERIES_CV_ZERO
R28_BIDIRECTIONAL_SMOOTHER_FACTOR_OUTPUT_ZERO

R28_ALL_MODEL_CANONICALS_TIMING_CONTRACTED
R28_ALL_PREDICTIVE_MODELS_FIT_THROUGH_T_MINUS_1
R28_ALL_SUPERVISED_LABELS_MATURITY_SAFE
R28_ALL_OVERLAPPING_LABEL_MODELS_PURGED
R28_ALL_MODEL_SCALERS_TRAIN_ONLY
R28_ALL_MODEL_PCA_TRAIN_ONLY
R28_ALL_MODEL_HYPERPARAM_SELECTION_PIT_SAFE

R28_MODEL_PREFIX_INVARIANCE_PASS
R28_MODEL_FUTURE_PERTURBATION_PASS
R28_STATEFUL_CHUNK_EQUIVALENCE_PASS

R28_CURRENT_MISSING_SEMANTICS_CLOSED
R28_PATH_SIGNATURE_STALE_SUBSTITUTION_CLOSED

R28_DIAGNOSTIC_DEFAULT_MINING_EXPOSURE_ZERO
R28_RESEARCH_DEFAULT_MINING_EXPOSURE_ZERO
R28_INTERNAL_DEFAULT_MINING_EXPOSURE_ZERO

R28_ALL_DECLARED_BACKEND_CLAIMS_EVIDENCED
R28_ALL_PARAMETER_DEFAULTS_FEASIBLE
R28_ALL_SOURCE_DEPENDENT_OPS_CONTEXT_TESTED

R28_TEST_RESULTS_GITHUB_TRACKED
R28_EVIDENCE_FILES_NOT_GITIGNORED
R28_EVIDENCE_GIT_SHA_CURRENT
R28_EVIDENCE_CANONICAL_DIGEST_CURRENT

R28_HARD_BLOCKERS_ZERO
```

---

# 二百二十七、最终 Production Ready条件

R28 不替代此前轮次。

最终：

```text
R24 semantic / state / identity / PIT
AND
R25 genuine usability / source / recipe / evidence
AND
R26 hidden numeric / time-topology correctness
AND
R27 large-batch execution performance
AND
R28 full canonical / model causality / test evidence
```

全部 hard flags通过，

才能：

```text
FACTOR_ENGINE_OPERATOR_SYSTEM_PRODUCTION_READY = true
```

---

# 二百二十八、给代码 AI 的最后要求

收到本文件后：

```text
不要只写分析报告。
不要只改 metadata。
不要只改 surface。
不要只把问题算子改成 research。
不要只生成一个漂亮 CSV。
```

必须：

```text
真正扫 current registry
真正扫 source code
真正修 implementation
真正删 useless/noncausal/random public operators
真正跑每个 canonical
真正跑模型 causality tests
真正跑 mining consumer tests
真正生成 current-head evidence
真正把 evidence提交到 GitHub-visible路径
```

---

# 二百二十九、最关键的质量标准

最终我们不是要一个：

```text
“算子很多的 FactorEngine”
```

而是一个：

```text
每一个留下来的算子，
都知道为什么存在，
都知道在量化场景中怎么使用，
都知道它的时间语义，
都知道它是不是可做 terminal，
都知道它的模型训练截止在哪里，
都能由测试证明不会偷看未来，
而且 GitHub 上能直接看到这份证明。
```

---

# 二百三十、最终机器必须能输出类似结果

```text
Git SHA: ...
Current canonicals: N

Production factor: ...
Production high-cost: ...
Contextual: ...
State/condition/event: ...
Intermediate: ...
Research diagnostic: ...
Internal: ...
Forbidden tombstone: ...
Deleted: ...
Blocked: ...

Retained canonicals with >=1 test: N / N
Zero-test canonicals: 0

Model-like canonicals: M
Model timing contracts: M / M
Prefix invariance pass: M / M applicable
Future perturbation pass: M / M applicable
Label maturity failures: 0
Random CV findings: 0
Bidirectional smoother findings: 0

Random public operators: 0
Future public operators: 0
Forbidden alias escapes: 0

Tracked R28 evidence files: all present
Stale evidence: 0

R28_HARD_BLOCKERS_ZERO = true
```

只有这样才算本轮完成。
