# 16 — Factor Asset Lifecycle, Similarity Graph & Assembly

## Identity Layers

1. exact ID/hash
2. canonical AST/parameters
3. structural fingerprint
4. behavioral fingerprint
5. conditional novelty

不要把 4/5 和 1/2 混成一个“duplicate=true”。

## Fingerprint Candidates

低成本 compact representation：

- sampled cross-sectional rank sketches
- TopK MinHash
- compressed PnL vector
- horizon IC vector
- exposure vector
- turnover/liquidity vector
- operator/domain genealogy

ANN 只负责 recall shortlist，不做最终去重裁决。

## Sparse Graph

Edge 存 view-level evidence：

- relation type
- weight/vector
- evaluation period
- confidence
- created_at

动态更新时保留历史 graph version/cluster lineage。

## Family Strategy

至少区分：structural/data/economic/behavioral。不要让单 `cluster_id` 成为所有概念真值。

## Assembly

第一版：

- representative selection
- mean/median/trimmed rank
- stability/shrinkage weight
- keep residual signals

目标不是“10万删到500”，而是识别独立信息维度并构造 300–1000 左右稳定 ModelInput 候选（具体数量由证据/模型端决定，不写死）。
