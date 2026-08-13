# 21 — Legacy Function Mining Checklist

本清单要求 AI 不只看目录名，而要真正读取实现、tests、call sites。

## QuantEvaluator Miner

搜索：`FactorAnalyzer`, `Exposures`, `daily_ic`, `rank_ic`, `quantile`, `turnover`, `sharpe`, `drawdown`, `hac`, `purged`, `forward_return`。

对每个 symbol 记录：

- formula semantics
- data layout assumption
- NaN/tie behavior
- sign/return convention
- annualization
- groupby loops / vectorization
- implicit future shift
- I/O coupling
- tests/callers

重点发现：旧实现之间 forward return shift 方向是否一致，hard-coded symbol/date names，Python 条件 bug，隐式未来数据。

## FactorPreprocess Miner

搜索：winsorize, zscore, rank_gauss, neutralize, ridge, lasso, huber, rolling, ewma, boxcox, yeojohnson, PCA, ICA, impute。

每个方法标：

- stateless / fitted
- fit parameters
- uses labels?
- time direction
- cross-sectional vs time-series
- production/advanced/research tier
- dependency weight

## FactorOptimizer Miner

搜索：registry, TransformSpec, parameter bounds, hash, complexity, candidate, verifier, agent loop, mutation/evolution。

重点判断：

- 是否只是 metadata 可复用
- 是否重复 FE operators
- complexity 是否 regex placeholder
- tool/agent loop 是否值得只借 orchestration pattern

## FactorAssets Miner

搜索：admission, catalog, sqlite, decision, evaluation run, status, dedup, fingerprint, assetization, lineage, candidate_id, campaign_id。

重点：

- 哪些表/字段可以保留思想
- 哪些 metric columns 应改 EvidenceRef
- 哪些固定 threshold 应去 policy config
- 是否物理 delete
- 是否有 schema migration

## Corpus Miner

GTJA/Week2/其他 factor packs：

- formula count
- manifests
- canonical DSL compatibility
- required fields/operators
- duplicate families
- test coverage
- runtime cost distribution

输出可重复的 `corpus_manifest.json`，但不要修改公式来让测试通过。
