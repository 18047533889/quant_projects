# factor_assets — 因子资产注册表

因子资产库：**身份 / 注册 / 去重 / 相似度 / 聚类 / 生命周期 / 入库治理**。
它是"平台上存在哪些因子、彼此如何关联、处于什么状态"的单一事实源。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/factor_assets (私有)

## 它拥有什么

- **身份（identity）** — 从因子表达式得到 canonical 身份（`FactorIdentityProvider`），
  确定性 `create_factor_id(canonical_hash)`，`FactorValueIdentity` / `FactorDefinitionIdentity`。
- **注册表（registry）** — `AssetRepository`（append-only、身份不可变），可选
  `SQLiteLifecycleRepository`；快照与迁移。
- **去重（seen index）** — `SeenIndex`（exact）+ `PersistentSeenIndex`（sqlite），
  判断"这个因子是否已入库过"。
- **相似度（similarity）** — `QEPairwiseSimilarity`（经 quant_evaluator）、
  ANN 后端（faiss / annoy）、`SimilarityArtifact`。
- **聚类（clustering）** — `LeidenClustering`（igraph/leidenalg，仅认证图生产级）+
  hierarchical/modularity 回退；`incremental_assign` 增量聚类版本；`certification`（图内容哈希）。
- **生命周期（lifecycle）** — `LifecycleState`：REGISTERED → EVALUATED → APPROVED →
  PRODUCTION_READY → DEPRECATED / RETIRED，状态机 + 迁移守卫。
- **入库治理（library）** — `PromotionGate` / `RollbackGate`（基于证据的晋升决策）。
- **组装与聚合（assembly/aggregation）** — `FactorSetAssembler`、复合聚合
  （horizon/orientation/regime）、家族代表选择。
- **战役（campaigns）** — 研究战役治理 / 预算 / 协调器。

## 安装与第一步

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_assets.git
cd factor_assets
pip install -e .                     # core
pip install -e ".[adapters]"         # + quant_evaluator / factor_engine / data_access
```

```python
from factor_assets.registry.factory import create_repository
from factor_assets.identity.canonical import FactorIdentityProvider, create_factor_id

repo = create_repository()
identity = FactorIdentityProvider().get_full_identity(expr_node)  # 来自 factor_engine
factor_id = create_factor_id(identity.canonical_hash)
repo.register(factor_id, metadata={...})
```

适配器（可选、lazy import、protocol-based）：`QEEvidenceProvider`（bundle → EvidenceRef）、
`FEIdentityProvider`（Expr → identity）、`DAFactorValueReader` / `DACatalogReader`（protocol）。
详见 `docs/archive/ADAPTERS_IMPLEMENTATION.md`。

## 目录

```
contracts/     FactorAsset, FactorAdmissionArtifact, FactorSetArtifact,
               SimilarityArtifact, TreatmentSelectionArtifact, lifecycle, lineage
registry/      repository, lifecycle, sqlite_repository, factory, snapshots
identity/      canonical (hash), adapters
seen_index/    exact + persistent (sqlite) 去重
similarity/    exact (QE), ANN (faiss/annoy)
clustering/    families (Leiden/modularity/hierarchical), incremental, certification, lineage
library/       promotion / rollback 门
lifecycle/     状态机
assembly/      FactorSetAssembler
aggregation/   复合评估、代表选择
novelty/       条件新颖度 / 残差-IC 新颖度（adapters/residual_novelty.py）
selection/     选择策略 + 门
campaigns/     战役协调器 / 预算 / 治理
adapters/      factor_engine / quant_evaluator / data_access（可选）
graph/         稀疏相关图、边过滤
optimizer/     pareto / plateau / multifidelity / typed_mutation 辅助
docs/          文档 + archive
tests/         63 个测试文件
```

## 硬性规则

- Append-only：已入库因子不得静默修改；lineage 保留。
- Core 从不 import 适配器；适配器互不 import。
- 证据只**引用**不复制 —— `EvidenceRef` 指向 QE bundle。
- 生产聚类要求认证图（graph content hash 强制）。

## 依赖与接口（谁 import 谁）

- **依赖（可选适配器）**：`factor_engine`（表达式身份）、`quant_evaluator`（证据）、
  `data_access`（因子值/目录读 protocol）。
- **被谁调用**：`quant_platform`（cluster/library DTO 映射到 FA artifact）、
  `factor_optimizer`（把入选 treatment 提交入库）。

## 相关仓库

- **factor_engine** — canonical 身份来源（表达式系统）
- **quant_evaluator** — 证据来源（`QEEvidenceProvider`）
- **factor_optimizer** — 上游搜索产出 treatment 候选
- **quant_platform** — 平台侧 cluster/library DTO 映射
