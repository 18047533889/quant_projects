# 12 — Implementation Roadmap

## Phase 0 — Audit & Freeze

输出：

- LOCAL_REPO_SNAPSHOT.md
- MIGRATION_MATRIX.md
- PACKAGE_BOUNDARY.md
- CONTRACT_FREEZE.md
- FILE_OWNERSHIP.md

不完成不得进入 Phase 1。

## Phase 1 — QuantEvaluator Foundation

P0：contracts/API/registry/shared intermediates/reference IC/coverage/quantile/turnover。

Acceptance：单独安装；old-vs-new golden；100/1000 factor benchmark。

## Phase 2 — FactorPreprocess Foundation

迁：rank/zscore/winsor/OLS/ridge/robust transforms；stateless/fitted protocol。

Acceptance：future-poison + fold boundary + parity。

## Phase 3 — FactorAssets Registry/Identity

迁 SQLite/catalog 思路；canonical identity adapter；seen index；lifecycle events。

Acceptance：no raw values stored；state/history tests。

## Phase 4 — FactorOptimizer Grammar/Search

registry + typed mutations + FE/QE adapters + complexity/plateau + basic multi-fidelity。

Acceptance：invalid mutation rejected；search budget/lineage tests。

## Phase 5 — QE Advanced Evidence

A-share robustness, exposure, decay, shape, HAC/bootstrap, conditional novelty primitives。

## Phase 6 — FA Scale Layer

fingerprints/ANN/sparse graph/families/shadow/aggregation。

Acceptance：10k benchmark；100k synthetic design benchmark；ANN recall small exact reference。

## Phase 7 — FP Representation

multi-channel, neutralization policy, FactorSet -> FeatureBundle。

## Phase 8 — Research Control & Review Skill

campaign/trial ledger；library-level review skill；搜索边际收益/失败历史。

## Phase 9 — Full Integration

DA/FE -> QE -> optional FO loop -> FA -> FP。

真实 A 股小窗 + GTJA/Week2 + external corpus smoke。

## Phase 10 — Hardening

- math/leakage/performance/boundary/simplification/license audit
- extraction test
- docs
- legacy import zero
- archive old platform shells

## Priority Rule

第一版宁愿把 40 个核心 metrics 做得极快、极准，也不要为了“200 个指标”让 runtime 架构复杂、不可维护。Metric Catalog 可以先注册 capability/status，逐批实现。
