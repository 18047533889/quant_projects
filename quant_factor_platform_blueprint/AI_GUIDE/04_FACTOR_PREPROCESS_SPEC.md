# 04 — FactorPreprocess Development Specification

## 4.1 定位

`factor_preprocess` = 模型输入前的 Factor Representation Engine。

它只决定：已选择的因子用什么数值表示交给下游模型。

不负责：

- 原始因子计算（FE）
- 因子有效性判断（QE）
- 因子自动优化（FO）
- 因子资产准入/聚类（FA）
- 最终模型训练

## 4.2 主要 Legacy 来源

优先迁移：

- `AlphaPurifier.py`
- `APr_utils.py`
- `toolkit/cross_sectional.py`
- `Exposures.py` 中与 neutralization/decomposition 相关的纯算法

Legacy 实现必须拆除旧 Database/YAML/path/report/pipeline 外壳。

## 4.3 两种核心 Protocol

### StatelessTransform

```python
transform(values, context) -> values
```

典型：当日 rank、当日 zscore、MAD winsor、demean、已知 exposure 的当日 OLS residual（如果系数仅当日横截面求解且使用当时已知 exposure）。

### FittedTransform

```python
state = fit(train_values, train_context)
out = transform(test_values, state, test_context)
```

典型：PCA/ICA、Box-Cox 参数、learned scaler、supervised transform、time-series learned coefficient。

`FittedState` 必须存：

- fit_start/end
- feature IDs/order
- transform version/config
- train universe definition
- learned params hash

禁止 `fit_transform(full_sample)` 后再切 train/test。

## 4.4 Production Core

第一版生产方法：

- cross-sectional rank
- zscore
- robust zscore (median/MAD)
- quantile/MAD winsorization
- rank gaussianization
- demean
- OLS neutralization
- ridge neutralization
- rolling robust scale（causal）
- volatility scaling（causal）
- missing indicator
- freshness/age channel

高级/实验：Huber/Theil-Sen/Yeo-Johnson/Box-Cox/PCA/ICA/RF/GBDT neutralization/RANSAC 等，默认关闭。

## 4.5 Neutralization

不允许一个全局 `neutralize=True`。

支持：

- raw
- industry residual
- size residual
- liquidity residual
- industry+size(+liquidity) residual
- exposure component
- soft neutralization

Soft neutralization：

`f_alpha = f - alpha * projection`, alpha ∈ [0,1]

Neutralization policy 的选择必须基于 QE 的 neutralization survival / exposure evidence；FP 不自行重新计算 IC 决策。

## 4.6 Multi-channel Representation

一个 factor 可输出多个 channel：

- raw
- rank
- zscore
- residuals
- exposure components
- missing mask
- freshness/age

不要把所有信号强行压成同一 residual channel。

## 4.7 Model-Oriented Policy（只准备，不训练）

可提供 preset：

- `linear_ready`: rank/residual/低共线表示
- `tree_ready`: raw+rank+exposure+missingness
- `neural_ready`: normalized+residual+missing+freshness

这些只是 representation policy，不导入 sklearn/torch 模型训练器。

## 4.8 Missingness

基本面缺失不能默认填 0。

支持：

- leave missing + missing indicator
- cross-sectional robust imputation（如政策允许）
- historical causal imputation
- fitted imputer（必须 fold-local）

任何 imputation 都要可审计，并在 output metadata 标注。

## 4.9 Performance

Reference：Pandas/NumPy；Fast：NumPy/Numba/Polars。

优先批量多因子：

- rank/zscore/winsor 可以对 factor block 向量化
- neutralization 若 exposure matrix 相同，应复用分解/设计矩阵中间量
- 不为每个 factor 重新构建 industry dummy matrix

## 4.10 Output

`FeatureBundle`：

- feature matrix
- feature IDs
- source factor IDs
- channel/type
- transform pipeline
- fitted state refs
- availability/timing metadata
- missingness metadata

## 4.11 Tests

必须：

- stateless determinism
- fitted train/test boundary poison test
- expanding training window consistency
- reference/fast parity
- batch/chunk parity
- same-day permutation invariance（适用 transform）
- no future influence：修改未来数据不得影响过去 transformed output
- neutralization residual exposure sanity
- extraction test
