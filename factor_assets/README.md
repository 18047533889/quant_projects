# factor_assets — 因子资产注册表（身份/去重/聚类/生命周期/入库治理）

因子资产库：**身份 / 注册 / 去重 / 相似度 / 聚类 / 生命周期 / 入库治理**。
它是"平台上存在哪些因子、彼此如何关联、处于什么状态"的**单一事实源**。

**定位:** 企业级 A 股横截面日频多因子量化项目的**因子资产层** —— factor_optimizer
选出 treatment 候选，本库决定它能否入库、入到哪个簇、处于什么状态；平台前端展示的
因子目录/聚类/库全部以本库为准。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/factor_assets (私有)

---

## 它是什么 / 不是什么

**做什么：** 因子身份铸造（canonical hash → factor_id）、append-only 注册、三层去重、
QE 证据相似度 + ANN 近邻、Leiden 生产聚类（认证图）、六态生命周期状态机、
PromotionGate 入库门、复合聚合与家族代表选择。

**不做什么：** 不算因子值（factor_engine）、不算指标（quant_evaluator）、不做搜索寻优
（factor_optimizer）—— 证据只**引用**不复制（`EvidenceRef` 指向 QE bundle）。

## 功能全清单

### 身份（identity/）
- `FactorIdentityProvider.get_full_identity(expr_node)` — 从 factor_engine 表达式得到
  canonical 身份（AST 归一化 SHA-256）
- `create_factor_id(canonical_hash)` — 确定性 factor_id 铸造；身份**不可变**
- `FactorValueIdentity` / `FactorDefinitionIdentity` — 定义级 vs 值级身份分离

### 注册表（registry/）
- `create_repository(backend)` — **memory / sqlite / production** 三种后端
- `AssetRepository` — append-only 注册表：`register` / `commit_transition`
  （**乐观锁** + `decision_id` 幂等重放）、快照与迁移
- `SQLiteLifecycleRepository` — 持久化生命周期

### 去重（seen_index/）
- `SeenIndex` — 精确（in-memory）
- `PersistentSeenIndex` — sqlite 持久化，跨进程判断"这个因子是否已入库过"

### 相似度（similarity/）
- `QEPairwiseSimilarity` — 走 quant_evaluator 算真实相关；**UNKNOWN ≠ dissimilar**
  （评估不出来时绝不当成"不相似"，fail-closed 语义）
- ANN 后端：**Faiss EXACT_FLAT**（诚实精确检索）/ **Annoy**（APPROXIMATE，声明近似）
- `SimilarityArtifact` — 相似度结果落盘契约

### 聚类（clustering/）— 生产级 Leiden only
- `LeidenClustering`（igraph/leidenalg）— **生产唯一允许的聚类算法**
- hierarchical / modularity — 仅研究回退
- `enforce_certified_graph` — **fail-closed 认证门**：图内容哈希不匹配直接拒绝聚类，
  防止"聚类结果与图不对应"
- `incremental_assign` — 增量聚类（`UNKNOWN_AFFINITY_FLOOR=0.25`：未知亲和度按 0.25
  下限处理，新因子保守分簇，不打破既有簇）
- lineage — 聚类版本血缘

### 生命周期（lifecycle/）— 六态状态机
```
REGISTERED → EVALUATED → APPROVED → PRODUCTION_READY → DEPRECATED / RETIRED
```
状态机 + 迁移守卫：非法跃迁直接拒绝，每次迁移都要 evidence。

### 入库治理（library/）
- **PromotionGate** — 基于证据的晋升决策，fail-closed 规则：
  - `min_rank_ic = 0.02`（不达标 → REJECT）
  - 与已入库因子重复度 > 0.7 → **REJECT**
  - 证据未测（unmeasured）→ REVIEW（不自动放行）
  - **price_convention 必须是 vwap_to_vwap** —— 其他口径直接防御性拒绝
- `RollbackGate` — 生产因子回滚决策

### 组装与聚合（assembly/ aggregation/）
- `FactorSetAssembler` — 把入选因子组装成 FeatureSet（对接 quant_platform 特征集 DTO）
- 复合聚合：horizon / orientation / regime 维度
- 家族代表选择 — 同簇选代表，避免冗余入库

### 新颖度（novelty/）
- 条件新颖度 / 残差-IC 新颖度（`adapters/residual_novelty.py`）：剔除已知因子解释部分后
  剩余 IC 才算真新颖

### 战役治理（campaigns/）
研究战役协调器 / 预算管控 / 治理。

## 安装与第一步

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_assets.git
cd factor_assets
pip install -e .                     # core（零重依赖）
pip install -e ".[adapters]"         # + quant_evaluator / factor_engine / data_access
```

```python
from factor_assets.registry.factory import create_repository
from factor_assets.identity.canonical import FactorIdentityProvider, create_factor_id

repo = create_repository("sqlite")                       # memory / sqlite / production
identity = FactorIdentityProvider().get_full_identity(expr_node)   # 来自 factor_engine
factor_id = create_factor_id(identity.canonical_hash)
repo.register(factor_id, metadata={...})
```

适配器（可选、lazy import、protocol-based、互不 import）：
`QEEvidenceProvider`（QE bundle → EvidenceRef）、`FEIdentityProvider`（Expr → identity）、
`DAFactorValueReader` / `DACatalogReader`（data_access 读 protocol）。
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
lifecycle/     六态状态机
assembly/      FactorSetAssembler
aggregation/   复合评估、家族代表选择
novelty/       条件新颖度 / 残差-IC 新颖度
selection/     选择策略 + 门
campaigns/     战役协调器 / 预算 / 治理
adapters/      factor_engine / quant_evaluator / data_access（可选）
graph/         稀疏相关图、边过滤
optimizer/     pareto / plateau / multifidelity / typed_mutation 辅助
tests/         63 个测试文件
```

## 硬性规则

- **Append-only**：已入库因子不得静默修改；lineage 永远保留
- Core 从不 import 适配器；适配器互不 import（protocol 解耦）
- 证据只**引用**不复制 —— `EvidenceRef` 指向 QE bundle，QE 是唯一评估权威
- 生产聚类**必须**认证图（graph content hash 强制，Leiden only）
- 入库门 fail-closed：证据缺失/口径不对 → REVIEW/REJECT，绝不默认放行

## 依赖与接口（谁 import 谁）

- **依赖（可选适配器）**：`factor_engine`（表达式身份）、`quant_evaluator`（证据）、
  `data_access`（因子值/目录读 protocol）。
- **被谁调用**：`factor_optimizer`（入选 treatment 提交入库）、
  `quant_platform`（cluster/library DTO 映射到 FA artifact）、
  `alphaprobe`（挖掘候选投递 candidate_pool 后由本库治理）。

## 相关仓库

- **factor_engine** — canonical 身份来源（表达式系统）
- **quant_evaluator** — 证据来源（QEEvidenceProvider）
- **factor_optimizer** — 上游搜索产出 treatment 候选
- **quant_platform** — 平台侧 cluster/library DTO 映射
