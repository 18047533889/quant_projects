# 06 — FactorAssets Development Specification

## 6.1 定位

`factor_assets` = Factor Asset Governance & Assembly Engine。

核心问题：数千到十万因子如何身份化、去重、判断增量信息、组织成 family/graph、筛选、聚合、生命周期治理。

## 6.2 Legacy 来源

优先迁移思想/结构：

- `factor_admission/catalog.py`：decision ledger/catalog/status 分离
- Gateway Candidate：origin/campaign/stage metadata
- exact expression hash / seen history 思想
- Assetization 的 FactorCandidate/FactorAsset 元数据
- factor-pool-standard 中仍有价值的 domain taxonomy

删除：

- 固定 `IC > x` 的万能准入
- 第二套 metrics
- PhysicalPlan
- 大 factor matrix storage
- stub semantic/vector dedup

## 6.3 Core Data Model

### FactorAsset

至少包含：

- factor_id
- definition_ref/canonical_repr/hash
- identity metadata
- origin/campaign/parents
- semantic/data/structural families
- complexity profile/ref
- latest evidence ref
- relation/membership refs
- lifecycle state
- shadow reason
- timestamps/version

**不嵌入全历史 factor values。**

### EvidenceRef

只保存 QE bundle ID/ref + selected summary，避免复制 200 个指标真值到 SQLite 列。

## 6.4 Registry Storage

第一版：Repository interface + SQLite WAL backend。

表建议：

- `factor_asset`
- `factor_lineage`
- `factor_state_event`
- `factor_decision`
- `factor_evidence_ref`
- `factor_relation`
- `factor_membership`
- `factor_shadow`
- `factor_fingerprint`
- `search_provenance_ref`

数据库 migrations 必须版本化；写入事务化；状态变化 append event，不只 overwrite current state。

## 6.5 Lifecycle

建议状态：

- DISCOVERED
- COMPILE_VALIDATED
- VALUE_VALIDATED
- EVALUATED
- ADMITTED
- ACTIVE_CORE
- ACTIVE_TACTICAL
- REGIME_CONDITIONAL
- SHADOWED
- WEAKENING
- DORMANT
- QUARANTINED
- REJECTED
- RETIRED
- PRODUCTION_CERTIFIED（可作为 certification flag 而非主状态，实施时二选一保持简洁）

状态机必须定义合法 transition；不允许物理删除历史因子来表示“淘汰”。

## 6.6 Identity & Seen Index

Stage 1：FE 提供 canonical AST hash（若可用）；FA 不另写完整 DSL parser。

Stage 2：参数 canonicalization / sign-orientation metadata。

Stage 3：structural fingerprint。

Stage 4：behavioral similarity。

GlobalSeenIndex 保存 active/shadowed/rejected/retired/failed 历史，避免自动 miner 重复试验。

可支持 service adapter：算法组只提交 fingerprint，返回 duplicate/neighbor/family/decision hints，不泄露其他团队公式。

## 6.7 Admission

不要单 Score。

流程：

1. Hard Safety Gates：非法/常数/coverage 极差/证据缺失等。
2. Evidence Profile：QE bundle。
3. Conditional Novelty。
4. Multi-objective/Pareto + budget/context decision。

固定阈值只允许作为某个 policy/config 的 gate，不得写死在核心代码。

## 6.8 Conditional Novelty Boundary

统计计算归 QE；FA 负责 orchestration/decision。

典型：

- candidate vs current pool linear residual IC
- nonlinear predictability residual
- portfolio PnL residual
- conditional IC / incremental OOS utility

FA 先用 ANN shortlist 找邻居，再请求 QE 对 shortlist 做 expensive exact metrics，避免全 O(K²)。

## 6.9 Multi-view Similarity

Views：

- structural/genealogy
- rank signal similarity
- PnL similarity
- top/bottom overlap
- exposure similarity
- horizon/decay vector
- regime vector
- turnover/liquidity profile

不要过早合成一个永久距离；保存 view-level evidence。

## 6.10 10k–100k Scaling

禁止全 pair matrix。

```text
Factor -> compact fingerprint (64–256D or hashes)
       -> ANN/LSH shortlist
       -> top M neighbors (50–200)
       -> exact QE similarity/novelty
       -> sparse graph
       -> community/family analysis
```

ANN index backend 先抽象；第一版可用 HNSW/FAISS-like 可选实现，但不能让核心强依赖特定大型服务。

## 6.11 Multiple Membership

一个 factor 不只有 cluster_id。

允许：

- structural family
- data family
- economic family
- behavioral/statistical cluster
- model-utility cluster（未来）

支持 soft membership；动态 graph 记录 split/merge/persistence，而不是每次聚类覆盖历史。

## 6.12 Shadow, Not Delete

高相关/低残余价值因子状态 `SHADOWED`，记录：

- shadowed_by
- reason
- exact metrics
- activation/revival policy

如果代表因子弱化或 regime 变化，可重新激活 shadow。

## 6.13 Aggregation

Production Core：

- representative factor
- mean rank
- median rank
- trimmed rank ensemble
- stability-weighted
- shrinkage-weighted

高级：PLS/sparse PLS/supervised compression 可作为 plugin/plan；不要第一版做所有 latent model。

FA 负责选择成员/权重/aggregation spec；如果需要 raw value matrix，可通过 ValueProvider 获取，不自行建设 factor lake。

## 6.14 Output

`FactorSetArtifact`：

- selected raw factors
- family/cluster alphas
- residual signals refs
- aggregation specs
- family/membership metadata
- evidence summaries
- allowed preprocessing policies

不要默认压成单一 Super Alpha。

## 6.15 Tests

- registry transaction/migration
- state transition property tests
- exact hash repeatability
- no physical delete
- ANN recall benchmark against small exact subset
- sparse graph scale test
- shadow/revival tests
- no raw-value persistence policy
- QE adapter mocked tests
- package extraction
