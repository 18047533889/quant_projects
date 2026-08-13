# 17 — FactorPreprocess Policy Catalog

## Core Stateless

- rank
- zscore
- robust zscore
- MAD/quantile winsor
- rank gaussian
- demean
- one-day cross-sectional residualization using available exposure

## Core Causal Temporal

- rolling robust scale
- EWMA scale
- volatility scale
- freshness decay/channel

## Fitted Unsupervised

- PCA/ICA (advanced)
- transform parameter estimation
- imputation state

所有 fitted state fold-local。

## Policy Resolver

输入：FactorSet metadata + desired downstream model family + QE recommendations。

输出：PreprocessingPolicy，不重新训练模型、不重新筛因子。

## Representation Examples

LinearReady：rank + selected residual + low-collinearity channels。

TreeReady：raw + rank + exposure + missingness。

NeuralReady：normalized + residual + missing/freshness + family metadata。

## Forbidden Defaults

- 全部 fillna(0)
- 全部 100% industry+size neutralization
- 全样本 PCA/scaler
- 使用未来行业/指数成分做历史 transform
