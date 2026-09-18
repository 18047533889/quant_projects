# 默认变换目录与策略预设

本目录来自当前环境的 `create_default_registry()` / `create_default_policies()`。共 40 项；当前环境 PyWavelets 未安装，三个 wavelet 函数未注册；源码采用条件注册，安装可选依赖并重新构造默认注册表后才会出现。所有默认项均 `requires_fit=False`、`fit_kind=stateless`；“历史”仅表示本次调用读取按资产隔离的过去输入。DataFrame 时间变换要求每资产日期单调，不会静默排序。公共列名默认 `asset_id/date/value`（收益列 `return`）。

## 全目录

| 注册名 | 参数（除公共列名） | 轴/历史；准入；路径 | 实际公式与边界 |
|---|---|---|---|
| `cs_rank` | `axis=-1, method="average", pct=False` | 截面/否；生产；FE `rank` | 有限值排名、NaN 原位；默认并列平均。百分位 $`z_i=(r_i-1)/(n-1)`$，单元素 0.5。 |
| `cs_zscore` | `axis=-1, ddof=1, constant_value=0` | 截面/否；生产；FP | $`z_i=(x_i-\bar x)/s`$ 零或不可用标准差时有限位写常数；与 FE 单元素规则不同。 |
| `cs_demean` | `axis=-1` | 截面/否；生产；FE `cs_demean` | $`z_i=x_i-\bar x`$；均值忽略 NaN。 |
| `cs_winsor` | `lower=.01, upper=.99, axis=-1` | 截面/否；生产；FE `winsorize` | $`z_i=\min\{q_u,\max(q_l,x_i)\}`$ 线性分位插值；要求 $`0\le l\lt u\le1`$。 |
| `cs_scale` | `axis=-1, target_std=1, ddof=1` | 截面/否；生产；FP | $`z_i=x_i\,target\_std/s`$ 不中心化；零标准差保留原值。 |
| `rolling_mean` | `window, min_periods=None` | 时间/严格滞后；生产；FP | None=window；$`z_t=\mathrm{mean}(x_{t-w:t-1})`$，有限数不足为 NaN。 |
| `rolling_std` | `window, min_periods=None, ddof=1` | 时间/严格滞后；生产；FP | $`z_t=\mathrm{std}_{ddof}(x_{t-w:t-1})`$。 |
| `rolling_zscore` | 同上 | 时间/历史；生产；FP | $`z_t=(x_t-\bar x_{t-1,w})/s_{t-1,w}`$ 当前值仅进分子，零标准差 NaN。 |
| `ewma` | `halflife, min_periods=1` | 时间/严格滞后；生产；FP | 对 shift(1) 做 `ewm(adjust=False)`：$`z_t=(1-\alpha)z_{t-1}+\alpha x_{t-1},\ \alpha=1-e^{-\ln2/h}`$。 |
| `trailing_sma` | `window, min_periods=None` | 时间/严格滞后；生产；FP | lagged 窗口均值；None=window。 |
| `trailing_median` | 同上 | 时间/严格滞后；生产；FP | $`z_t=\mathrm{median}(x_{t-w:t-1})`$。 |
| `robust_ewma` | `halflife, winsor_std=4, min_periods=1` | 时间/严格滞后；生产；FP | lagged 值先按 10 期、至少2点的 $`\mu\pm cs`$ 夹断，再 EWMA；非正 c 钳为 $`10^{-12}`$。 |
| `kama` | `period_fast=2, period_slow=30, period_er=10, use_current=False` | 时间/默认严格滞后；生产；FP | $`ER=\lvert x_t-x_{t-p}\rvert/\sum\lvert\Delta x\rvert,\ SC=[ER(a_f-a_s)+a_s]^2,\ K_t=K_{t-1}+SC(x_t-K_{t-1})`$ 平路径 ER=0；缺失效率保持前值。`use_current=True` 需外部 DecisionClock 证明。 |
| `one_sided_iir_lowpass` | `alpha` | 时间/严格滞后；生产；FP | $`z_t=(1-\alpha)z_{t-1}+\alpha x_{t-1},\quad0\lt\alpha\le1`$。 |
| `kalman_local_level` | `process_noise, measurement_noise` | 时间/严格滞后；生产；FP | $`P^-=P+q,\ K=P^-/(P^-+r),\ m=m^-+K(x_{t-1}-m^-)`$；lagged NaN 输出 NaN，并把水平与协方差状态都重置；下一有限值重新播种，绝非“只预测”。 |
| `event_decay` | `halflife, min_periods=1` | 时间/严格滞后；生产；FP | $`\alpha=\min(1,\ln2/h),\ z_t=(1-\alpha)z_{t-1}+\alpha x_{t-1}`$ lagged NaN 重置，后续有限值重播种。 |
| `volatility_scale` | `window, min_periods=None, ddof=1, target_vol=1` | 时间/历史；生产；FP | $`s_t=\mathrm{std}(x_{t-w:t-1}),\ z_t=x_t target\_vol/s_t`$ None=window；零波动 NaN。 |
| `volatility_scale_returns` | 同上，`target_vol=.01` | 时间/历史；生产；FP | 同式，输入契约为 returns。 |
| `realized_volatility` | `window, min_periods=None, ddof=1, annualization_factor=1` | 时间/历史；生产；FP | $`z_t=annualization\_factor\,\mathrm{std}(x_{t-w:t-1})`$。 |
| `forward_fill` | `max_lag=None` | 时间/历史；生产；FE `ffill_limit` | 最近有限值前填；None 无界，整数最多填连续 max_lag 个空位。 |
| `missing_indicator` | 无 | 当前行/否；生产；FP | $`z_t=\mathbf1[x_t\ \mathrm{missing}]`$。 |
| `missing_run_length` | 无 | 时间/历史；生产；FP | 当前连续缺失段累计长度；有限值归零。 |
| `missing_rate` | `window, min_periods=1` | 时间/严格滞后；生产；FP | $`z_t=\mathrm{mean}_{j=t-w}^{t-1}\mathbf1[x_j\ \mathrm{missing}]`$。 |
| `impute_with_fallback` | `fallback_strategy="forward_fill", max_lag=None` | 时间/全样本；RESEARCH_ONLY；FP | None 表示无界前填，正整数表示有界前填，0 跳过前填；随后可选 `forward_fill/zero/median/mean`，其中 median/mean 是每资产全样本统计，非因果。 |
| `detect_correlation_regime` | `window, n_regimes=2, percentiles=None, min_periods=None, value_cols=None` | 时间/全样本；RESEARCH_ONLY；FP | 滚动相关结构加全样本边界分 regime；返回 RegimeState，非数值 Series、非前缀不变。 |
| `days_since_update` | 无 | 时间/历史；生产；FP | $`z_t=(date_t-last\_valid\_date_t)/(1\ \mathrm{day})`$ 当前有限为0，从未有效 NaN。 |
| `observation_age` | `observation_date_col, current_date_col` | 当前行/否；生产；FP | $`z=current\_date-observation\_date`$，整数日；日期缺失则 NaN。 |
| `freshness_score` | `halflife_days` | 时间/历史；生产；FP | $`z_t=e^{-\ln2\,age_t/h}`$ 从未有效填0。 |
| `stale_data_indicator` | `max_days` | 时间/历史；生产；FP | $`z_t=\mathbf1[age_t\gt max\_days]`$ 从未有效 NaN；max_days 非负。 |
| `freshness_aware_fill` | `max_lag=None, decay_halflife=10` | 时间/历史；生产；FP | 当前有限原样；缺失时 $`z_t=x_{last}e^{-\ln2\,age/h}`$，超界/无过去值 NaN。 |
| `ols_neutralize` | `exposures, min_observations=10, add_intercept=True` | 截面/否；生产；FE `cs_neutralize` | 逐日共同有限支持：$`\hat\beta=(X^TX)^+X^Ty,\ e=y-X\hat\beta`$ 不足门槛整日 NaN。 |
| `industry_neutral` | 同上 | 截面/否；生产；FE | 与 OLS 同 kernel 的语义别名；调用者须提供行业暴露，名称不自动筛列。 |
| `size_neutral` | 同上 | 截面/否；生产；FE | 同上，调用者须提供 size 暴露。 |
| `dual_neutral` | 同上 | 截面/否；生产；FE | 同上，调用者须提供行业与 size 暴露。 |
| `compute_exposures` | `residuals, exposures` | 截面/否；生产；FP | 每日加截距 OLS：$`\hat\gamma_t=([\mathbf1,X_t]^T[\mathbf1,X_t])^+[\mathbf1,X_t]^Ty_t`$ 仅共同有限行，要求有效数严格大于 exposure 列数；输出 intercept 与各 `column_coef`，不是 residual transform。 |
| `bandpass_filter` | `low_freq, high_freq, order=4` | 时间/全样本；OFFLINE_ONLY；FP | Butterworth 带通加零相位全序列滤波；未来影响历史。 |
| `extract_cycle` | `low_period, high_period, order=4` | 时间/全样本；OFFLINE_ONLY；FP | 周期换为带通频率后调用 bandpass。 |
| `christiano_fitzgerald_filter` | `low_period, high_period, drift=True` | 时间/全样本；OFFLINE_ONLY；FP | statsmodels CF 双边近似理想带通，可去漂移。 |
| `hp_filter` | `lambda_param=1600` | 时间/全样本；OFFLINE_ONLY；FP | $`\min_\tau\sum(x_t-\tau_t)^2+\lambda\sum(\tau_{t+1}-2\tau_t+\tau_{t-1})^2`$ 返回 trend，非前缀不变。 |
| `hp_decompose` | `lambda_param=1600` | 时间/全样本；OFFLINE_ONLY；FP | 返回 $`(cycle,trend),\quad cycle=x-trend`$。 |
| `wavelet_decompose` | `wavelet="db4", level=None, mode="symmetric"` | 时间/全 lagged 样本；条件注册、OFFLINE_ONLY；FP | 每资产先 `shift(1)`、删除 NaN，对有限序列做 `pywt.wavedec`；None 取 $`\min(\mathrm{max\_level},5)`$。分别保留 approximation 或单层 detail 系数再 `waverec`，输出字典 `a{level}` 与各 `d{i}`。长度不足、分解失败或 level 小于 1 时返回 NaN 通道。重构使用整个可用 lagged 序列，故非前缀不变。 |
| `wavelet_smooth` | `wavelet="db4", level=1` | 时间/全 lagged 样本；条件注册、OFFLINE_ONLY；FP | 调用 `wavelet_decompose` 并返回 approximation 通道 $`a_{level}`$；若指定键不存在则取首个 approximation，否则全 NaN。 |
| `wavelet_denoise` | `wavelet="db4", level=None, threshold_mode="soft", threshold_scale=1` | 时间/全 lagged 样本；条件注册、OFFLINE_ONLY；FP | 对 lagged 有限序列分解，以最细 detail 的 MAD 估计 $`\sigma=\mathrm{median}(\lvert d-\mathrm{median}(d)\rvert)/0.6745`$，阈值 $`\lambda=threshold\_scale\,\sigma\sqrt{2\ln n}`$；保留 approximation，对 details 做 soft/hard threshold 后全序列重构。少于 4 点或失败为 NaN。 |

FE_OPERATOR 共 8 项：`cs_rank`、`cs_demean`、`cs_winsor`、`forward_fill`、`ols_neutralize` 及三个 neutral 别名；基础 40 项中的其余 32 项为 FP_NATIVE。三个 wavelet 项也始终在本目录列明，但仅在可选 PyWavelets 依赖使函数对象非 None 时才作为 FP_NATIVE/OFFLINE_ONLY 注册，因此运行时总数可能是 40 或 43。

## 默认策略

### `cs_only`（research）

`cs_winsor(.01,.99) → cs_rank(pct=True) → cs_zscore(ddof=1)`。纯截面；rank 后 z-score 会再次标准化百分位。

### `causal_basic`（staging）

`forward_fill(max_lag=5) → cs_winsor(.025,.975) → ewma(halflife=20) → cs_rank(pct=True) → cs_zscore(ddof=1)`。EWMA 严格滞后，会引入一期间隔。

### `production_full`（production）

`forward_fill(max_lag=3) → missing_indicator() → cs_winsor(.01,.99) → ewma(halflife=20,min_periods=10) → volatility_scale(window=60,min_periods=20) → ols_neutralize(skip_if_missing=False) → cs_rank(pct=True)`。

要求 universe/exposure；缺 exposure 不跳过。顺序为 missingness → outlier → temporal/risk scaling → neutralization → representation。missing indicator 是独立缺失通道，执行器必须遵守多通道 recipe，不能将 0/1 当主因子值覆盖。

### `returns_preprocessing`（staging）

`cs_winsor(.01,.99) → volatility_scale_returns(window=60) → cs_zscore(ddof=1)`。要求 returns；第二步默认 min_periods=60、target_vol=.01，按资产历史而非截面缩放。

### `minimal`（research）

`cs_rank(pct=True)`。仅适用于已清洁因子；NaN 保留、并列平均、单元素0.5。

### `research_full`（research）

`forward_fill(max_lag=10) → missing_rate(window=60) → cs_winsor(.025,.975) → rolling_zscore(window=120,min_periods=60) → cs_rank(pct=True)`。

missing rate 是诊断通道；执行器须遵守 recipe 通道契约，不能误把它串行覆盖原因子。rolling z-score 的统计严格滞后。

## 注意

- `causal_safe=True` 不保证任意参数安全；`kama(use_current=True)` 需要外部时钟证明。
- RESEARCH_ONLY/OFFLINE_ONLY 不能进入生产策略。
- rolling/volatility 的 `min_periods=None` 解析为 window，不是 1。
- neutral 语义别名不会自动选择 exposure 列。

## 实现来源

- [默认 TransformRegistry](../factor_preprocess/registry/transforms.py)：注册、准入、语义元数据与 FE/FP 路由。
- [默认 PolicyRegistry](../factor_preprocess/registry/policies.py)：六个预设的顺序、覆盖参数与 level。
- [截面实现](../factor_preprocess/transforms/cross_sectional.py)、[滚动实现](../factor_preprocess/transforms/rolling.py)、[平滑实现](../factor_preprocess/transforms/smoothing.py)。
- [波动率实现](../factor_preprocess/transforms/volatility.py)、[缺失实现](../factor_preprocess/transforms/missingness.py)、[新鲜度实现](../factor_preprocess/transforms/freshness.py)。
- [事件衰减](../factor_preprocess/transforms/event_decay.py)、[freshness-aware fill](../factor_preprocess/transforms/treatment_variants.py)、[OLS 中性化](../factor_preprocess/neutralization/ols.py)。
- [分解实现](../factor_preprocess/transforms/decomposition/) 与 [regime detector](../factor_preprocess/regime/detector.py)。
