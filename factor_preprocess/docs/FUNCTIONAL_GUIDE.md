# factor_preprocess 中文功能与算法指南

本文以当前源码、默认注册表和测试为准，说明已实现功能、数学口径、输入输出、因果边界与能力限制。它不把规划中的功能写成现有能力。逐项签名与源码定位见 [API 参考](API_REFERENCE.md)，当前环境的注册元数据与参数表见 [变换目录](TRANSFORM_CATALOG.md)。

## 1. 定位与工作流

本包位于因子计算、因子评估之后，模型训练之前。它不计算原始因子、不评价 alpha、不训练模型；职责是把“时间 × 资产”的因子面板转换成受治理的模型特征。

典型流程：

1. 固定数据快照、时间轴、资产宇宙、因子顺序和 PIT 暴露。
2. 从训练集生成 FactorProfileArtifact，结合已有处理谱系建立合法 TreatmentSearchSpace。
3. 选择 PolicyPreset 或构造有序 TreatmentRecipe。
4. 只在训练集拟合有状态步骤，冻结 FittedState。
5. 用同一状态应用到验证、测试或生产。
6. 产出 FeatureBundle、FeatureManifest、处理谱系和内容哈希。
7. 如模型需要特殊表示，另建 FeatureRepresentationArtifact，不覆盖 canonical factor asset。

默认 TransformRegistry 的数量取决于可选依赖：当前 server-c 环境注册 **40 个**；安装 PyWavelets 且可选模块成功导入时，会额外注册 3 个小波变换，总数为 **43 个**。因此调用方应查询 registry 快照，不能把固定数字当作跨环境合同。

## 2. 输入与输出

截面族接受 NumPy 数组，默认沿最后一轴处理，二维数组通常是日期 × 资产。时序、波动、缺失、新鲜度和中性化族主要接受 pandas 长表，默认列为 asset_id、date、value，收益列默认 return。每个资产内部必须按时间单调递增；实现不会偷偷排序。

严格滞后变换在时点 t 只使用不晚于 t−1 的历史。横截面变换使用当期资产截面，是否可交易仍须由上游 available-at 和决策时钟证明。

## 3. 截面变换

实现与边界测试：[cross_sectional.py](../factor_preprocess/transforms/cross_sectional.py)、[变换回归测试](../tests/transforms/test_review_regressions.py)。

### cs_rank

对有限值排序，支持 average、min、max、dense、ordinal 五种并列规则。pct=False 返回从 1 开始的秩；pct=True 使用：

```math
p_i=\frac{r_i-1}{n-1}
```

n 为有限值数。只有一个有限值时返回 0.5；NaN 原位保留。

### cs_zscore、cs_demean、cs_scale

```math
z_i=\frac{x_i-\bar{x}}{s},\qquad
d_i=x_i-\bar{x},\qquad
q_i=x_i\frac{s_{target}}{s}
```

cs_zscore 和 cs_scale 默认 ddof=1。zscore 遇零标准差时在有限位置返回 constant_value，默认 0；scale 遇常数截面保留原值。demean 只去均值，不改变尺度。

### cs_winsor

```math
y_i=\min\!\left(q_u,\max(q_l,x_i)\right)
```

默认 lower=0.01、upper=0.99，使用 NumPy 线性分位插值；要求 0≤lower＜upper≤1。

## 4. 因果滚动与平滑

rolling_mean 与 trailing_sma 都返回过去 window 个轴位置、排除当前值的均值；rolling_std 返回同一窗口的标准差。默认 min_periods=window。rolling_zscore 用过去窗口统计量标准化当前值：

```math
z_t=\frac{x_t-\bar{x}_{t-1,w}}{s_{t-1,w}}
```

trailing_median 返回过去窗口中位数。

ewma 先 shift(1)，再执行 adjust=False 的递归：

```math
y_t=(1-a)y_{t-1}+a x_{t-1},\qquad
a=1-2^{-1/h}
```

halflife 必须大于 0，min_periods 默认 1。

robust_ewma 先用滞后序列固定 10 期、至少 2 个值的均值和标准差做因果截尾，再做 EWMA；winsor_std 默认 4。非正 winsor_std 会被收缩成极小正数。

`kama`（KAMA）的效率比与递归为：

```math
ER_t=\frac{|x_t-x_{t-e}|}{\sum_{j=t-e+1}^{t}|x_j-x_{j-1}|}
```

```math
a_t=\left[ER_t(a_f-a_s)+a_s\right]^2,\qquad
y_t=y_{t-1}+a_t(x_t-y_{t-1})
```

默认 period_fast=2、period_slow=30、period_er=10，并整体滞后一期。use_current=True 只有在外部决策时钟已经证明当前值可用时才安全。

one_sided_iir_lowpass 使用：

```math
y_t=(1-a)y_{t-1}+a x_{t-1}
```

alpha 必须在 (0,1]；滞后 NaN 会重置状态。

kalman_local_level 使用局部水平模型：

```math
l_t=l_{t-1}+w_t,\qquad x_t=l_t+v_t
```

process_noise 可为 0，measurement_noise 必须大于 0；滤波消费滞后观测，缺失会重置状态。

event_decay 使用滞后事件序列：

```math
a=\min\!\left(1,\frac{\ln 2}{h}\right),\qquad
y_t=(1-a)y_{t-1}+a x_{t-1}
```

它不是“只保存最近非零脉冲”；有限的 0 也参与递归。NaN 会清空状态并重置 warmup。

## 5. 波动率

实现：[volatility.py](../factor_preprocess/transforms/volatility.py)。这里的 `annualization_factor` 是直接乘数；若日频标准差要年化，调用方应显式传入平方根 252，而不是 252。

realized_volatility 对过去 window 期计算标准差并乘 annualization_factor：

```math
v_t=A\,s(x_{t-w},\ldots,x_{t-1})
```

volatility_scale 和 volatility_scale_returns 用滞后波动缩放当前值：

```math
y_t=x_t\frac{v_{target}}{v_t}
```

前者目标波动默认 1，后者默认 0.01；零波动返回 NaN。ewma_volatility 存在于模块，但未注册为默认 transform ID。

## 6. 缺失与新鲜度

实现：[missingness.py](../factor_preprocess/transforms/missingness.py)、[freshness.py](../factor_preprocess/transforms/freshness.py)。

- forward_fill：逐资产传播最近有效值；max_lag=None 不限长度，正整数限制连续填充期数。
- missing_indicator：当前缺失为 1，否则 0。应在填充前生成，才能保存原始缺失事实。
- missing_run_length：当前连续缺失段长度，遇有效值归零。
- missing_rate：过去 window 期、排除当前值的缺失比例。
- impute_with_fallback：先 forward fill，再用 fallback；全样本 mean/median 会消费未来分布，因此该注册项是 RESEARCH_ONLY。
- days_since_update：距最近非缺失值的日历天数；当前有效为 0，从未有效为 NaN。
- observation_age：current date 减 observation date 的日历天数；负数意味着 PIT 违规，函数本身不会替治理层拒绝。
- freshness_score：从未有效返回 0，否则：

```math
f_t=2^{-age_t/h}
```

- stale_data_indicator：age 严格大于 max_days 时为 1；从未有效保留 NaN。
- freshness_aware_fill：最近值按年龄衰减：

```math
y_t=x_{last}\,2^{-lag_t/h}
```

linear_interpolate 和 time_weighted_interpolate 虽存在于模块，但未注册，且会利用未来端点填内部空洞，不能视为生产因果能力。

## 7. 中性化与暴露

实现与诊断：[ols.py](../factor_preprocess/neutralization/ols.py)、[regularized.py](../factor_preprocess/neutralization/regularized.py)、[诊断测试](../tests/test_v8_regularized_diagnostics.py)。

ols_neutralize 按日期对齐因子与暴露，在共同有限资产上拟合：

```math
x_t=a_t{\bf 1}+Z_t b_t+e_t
```

输出残差 e；默认含截距，min_observations=10。industry_neutral、size_neutral、dual_neutral 是同一内核的语义别名，**不会自动选择暴露列**，调用者必须传入语义匹配的 exposure 矩阵。

compute_exposures 返回逐日暴露诊断，不等于中性化残差。

NeutralizationSpec 可声明 OLS、RIDGE、HUBER 等方法、暴露集合、标准化、条件数策略与 PIT 身份。默认 registry 直接暴露的是 OLS；regularized 模块的 ridge/huber 是 helper，不是额外默认 transform ID。秩亏和病态矩阵应依策略 fail、warn 或 regularize，并保存 NeutralizationDiagnostics。

## 8. Regime 与离线分解

默认 registry 只有 detect_correlation_regime，使用全样本相关结构，标记为 RESEARCH_ONLY，输出 RegimeState 而非逐行 Series。regime 子包另有 detect_variance_regime、CausalRegimeDetector、fit_regime_weights、regime_adaptive_weights、fit_regime_switching 和 regime_switching_transform；是否能生产使用仍须检查 fit/apply、状态身份与 admission。

以下 5 项在所有环境均注册为 OFFLINE_ONLY、非 prefix-invariant：

- bandpass_filter：全序列带通；
- extract_cycle：按周期区间提取循环；
- christiano_fitzgerald_filter：CF 滤波；
- hp_filter：HP 单输出；
- hp_decompose：返回 cycle 与 trend。

若安装 PyWavelets，还会以同样的 OFFLINE_ONLY 约束注册 wavelet_decompose、wavelet_smooth、wavelet_denoise；当前环境未安装该可选能力，所以当前变换目录中没有这 3 项。

它们不能进入验证、测试或生产 apply 路径。

## 9. 六个内置策略

| 名称 | 等级 | 实际步骤 |
|---|---|---|
| minimal | research | rank percentile |
| cs_only | research | winsor 1%/99% → rank percentile → zscore |
| causal_basic | staging | fill 5 → winsor 2.5%/97.5% → EWMA 20 → rank → zscore |
| production_full | production | fill 3 → missing indicator → winsor → EWMA 20 → vol scale 60/20 → OLS neutralize → rank |
| returns_preprocessing | staging | winsor → return vol scale 60 → zscore |
| research_full | research | fill 10 → missing rate 60 → winsor → rolling zscore 120/60 → rank |

PolicyPreset.validate 检查注册存在性、参数绑定、生产 admission、因果安全和 step_id 唯一性。相同 transform 可重复，但 step_id 必须不同。policy_identity 覆盖步骤顺序、参数、版本、实现哈希和 numeric policy。

production_full 的 missing_indicator 是独立通道；简单“上一输出覆盖下一输入”的循环不能表达这种 fan-out。Preset 是受治理的步骤声明，不是完整 DAG 执行器。

## 10. Fit / Apply 无泄漏纪律

合同与对抗测试：[fit_apply.py](../factor_preprocess/contracts/fit_apply.py)、[fit/apply guards](../tests/contracts/test_fit_apply_guards.py)、[leakage properties](../tests/contracts/test_leakage_properties.py)。

TreatmentFitApplyDeclaration 声明 causality class、fit scope、fit split、state identity 和 application scope。硬规则：

- 验证、测试、生产只能使用 TRAIN 拟合状态；
- full-sample research 状态只能用于研究全样本；
- offline-only 不能用于 evaluation-valid split；
- apply as-of 不得早于 fit window 结束；
- 状态绑定数据快照、宇宙和特征顺序，任一不匹配均 fail closed；
- 标签知识时间不得晚于 apply as-of。

assert_prefix_invariant 会分别对完整序列和前缀重算；未来数据追加后历史输出改变，即判定泄漏。FittedState 记录 transform、版本、fit 窗口、状态类型和 provenance。不要在验证集重新拟合 scaler、regime 阈值或任何冻结状态。

## 11. 资格引擎与优化器接口

TreatmentEligibilityEngine 读取训练集 FactorProfileArtifact 和 ExistingTreatmentSignature，返回合法 TreatmentSearchSpace：

- RAW:noop 永远保留，以识别“处理后反而变差”；
- EVENT 偏向 event_decay，禁止长窗口平滑；
- FUNDAMENTAL、SPARSE_UPDATE 可用 freshness_aware_fill；
- 已有 temporal smoothing 会把平滑预算减半；
- raw_turnover 低于 0.1 会进一步收窄平滑上界；
- 已做行业中性化会剪掉重复 industry/dual neutralization；
- 候选名称最终必须在 canonical registry 可解析。

搜索范围包括 SMA/EWMA 约 3–60、event decay 1–5、freshness max lag 1–20、winsor 下界 0.5%–5% 和上界 95%–99.5%。当前 KAMA 搜索参数名 er_window、fast_span、slow_span 与执行签名 period_er、period_fast、period_slow 不同，集成层必须显式映射。

## 12. FeatureBundle 与模型表示

FeatureBundle 支持 NF 和 TNF 布局，绑定 AxisRef、ChannelRef、值数组和 FeatureManifest。Manifest 要求特征轴连续、无重叠、无洞；对象深度不可变并带内容身份。

四种表示策略：

- TREE_TABULAR：允许 raw、rank、clipped，zscore 可选；
- LINEAR：要求 robust outlier handling，并要求 train-fitted zscore 或 robust zscore 至少一种；
- NEURAL_TABULAR：要求 robust zscore 或 rank-gaussian 语义、missing channel 和稳定 clipping；
- SEQUENCE_MODEL：要求训练集拟合缩放和因果时序归一化。

模型专用 FeatureRepresentationArtifact 必须引用 canonical factor，写入器拒绝覆盖 canonical asset。non_inferior 守卫用于阻止模型强制处理明显破坏 alpha。

## 13. 示例

查询并校验预设：

```python
from factor_preprocess.registry.policies import get_default_policy_registry
from factor_preprocess.registry.transforms import create_default_registry

registry = create_default_registry()
registry_snapshot = registry.seal()
policy = get_default_policy_registry().get("causal_basic")
valid, errors = policy.validate(registry)
assert valid, errors

for step in policy.steps:
    metadata = registry.get(step.name)
    print(step.step_id, metadata.version, dict(step.parameters))
```

运行严格滞后 EWMA：

```python
import pandas as pd
from factor_preprocess.registry.transforms import get_default_registry

frame = pd.DataFrame({
    "asset_id": ["A", "A", "A", "B", "B", "B"],
    "date": pd.to_datetime([
        "2026-01-01", "2026-01-02", "2026-01-03",
        "2026-01-01", "2026-01-02", "2026-01-03",
    ]),
    "value": [1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
})

ewma = get_default_registry().get_execution("ewma")
result = ewma(frame, halflife=3, min_periods=1)
assert pd.isna(result.iloc[0])
assert result.iloc[1] == 1.0
```

截面链：

```python
import numpy as np
from factor_preprocess.registry.transforms import get_default_registry

registry = get_default_registry()
raw = np.array([[1.0, 2.0, 100.0, np.nan]])
winsor = registry.get_research_reference_function("cs_winsor", allow_research=True)
rank = registry.get_research_reference_function("cs_rank", allow_research=True)
zscore = registry.get_research_reference_function("cs_zscore", allow_research=True)
clipped = winsor(raw, lower=0.05, upper=0.95)
ranked = rank(clipped, pct=True)
scaled = zscore(ranked, ddof=1)
```

上例明确是本地 parity / 教学研究路径。`cs_winsor` 与 `cs_rank` 的正式执行路由到 FactorEngine，`cs_zscore` 保留 FP 原生路径；生产环境应安装匹配的 FE adapter 并使用 `get_execution`。对 FE 路由项，当 FE authority 不可用时，`get_execution` 会 fail closed，不会静默退回本地 reference kernel。

构造可寻址 recipe（声明）与单函数调用（执行）是两件事。recipe 保存步骤语义、实现引用、顺序和身份；正式执行还必须提供批准的 execution context，不能把下面的声明误当成已经完成 materialization：

```python
from factor_preprocess import FitBoundary, RecipeStep, TreatmentRecipe
from factor_preprocess.registry.transforms import create_default_registry

registry = create_default_registry()
registry_snapshot = registry.seal()
steps = (
    RecipeStep(
        step_id="winsor_raw",
        semantic_transform_id="WINSOR",
        implementation_ref="cs_winsor",
        stage="outlier",
        parameters={"lower": 0.01, "upper": 0.99},
    ),
    RecipeStep(
        step_id="rank_after_winsor",
        semantic_transform_id="RANK",
        implementation_ref="cs_rank",
        stage="representation",
        parameters={"pct": True},
    ),
)

recipe = TreatmentRecipe(
    recipe_id="demo:factor:preprocess:v1",
    source_factor_definition_ref="factor-def:demo:v1",
    source_factor_value_ref="factor-values:train:snapshot-001",
    ordered_steps=steps,
    fit_boundary=FitBoundary.EXPANDING,
    registry_snapshot_identity=registry_snapshot,
)
```

若只验证单个无状态 kernel，可用 `get_execution`；若要执行整个 recipe，应通过 `get_recipe_execution` 接入显式 execution context。研究环境也必须显式 `allow_research=True`，不能退回 `get_function` 旁路。需要 exposures、universe 或 fitted state 的步骤还必须由上层 wiring 提供对应制品引用；函数签名示例只是 standalone 调用，不替代这些治理绑定。

## 14. 能力边界

- 默认 registry 是执行真相；模块中的 helper 不一定能按 transform ID 调用。
- 正式执行使用 get_execution；get_function 是已退役的不受治理旁路，会直接拒绝。仅做 parity 研究时，才显式调用 get_research_reference_function 并传 allow_research=True。
- 当前 server-c 环境为 40 项；安装 PyWavelets 的环境可为 43 项。以 registry 快照为准，不要硬编码数量。
- 中性化别名不自动选择 exposure。
- 严格滞后平滑首行 NaN 是安全设计。
- forward_fill 无上限会无限传播陈旧值；生产通常应限制并输出 freshness channel。
- impute_with_fallback 的全样本统计模式、相关 regime 和所有离线分解不能用于生产。
- PolicyPreset 是步骤身份，不是处理多通道、exposure 和 universe 的完整编排器。
- 线性/神经模型的 train-fitted scaling 由 fit/apply 与 representation 合同治理，不等于逐日无状态 cs_zscore。

上线前至少验证输入排序、轴身份、PIT 暴露、prefix invariance、训练状态复用、NaN/Inf numeric policy、CPU/加速后端 parity、FeatureManifest 连续性，以及生产策略不存在 RESEARCH_ONLY/OFFLINE_ONLY 步骤。
