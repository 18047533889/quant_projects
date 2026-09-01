# DLIB-FA 实施说明（Factor_Assets 权威性边界清理）

| DLIB | 议题 | 状态 | 落地 |
|------|------|------|------|
| DLIB-FA-001 | FA 拥有「第二个 optimizer」— Pareto / MultiFidelity / Plateau / Mutation / Sealed Test | **完成** | `optimizer/FA_OPTIMIZER_MIGRATION_MATRIX.md` 矩阵；`optimizer/__init__.py` RESEARCH_ONLY；`multifidelity.py`/`plateau.py`/`frozen_candidate.py` 弃用警告；`pareto.py` 确认仅组装用途；`typed_mutation.py` 确认仅引用适配 |
| DLIB-FA-002 | FA 拥有 campaign 搜索控制 | **完成** | `ResearchCampaignGovernance`（治理元数据，无搜索控制字段）；legacy `CampaignCoordinator` RESEARCH_ONLY |
| DLIB-FA-003 | 顶部 API 暴露降级两用户 | **完成** | `factor_assets/__init__.py` 优先导出 `FactorSetArtifact` 等五件套规范工件；legacy `FactorSet` 保留为别名/视图 |
| DLIB-FA-004 | 相似性 0 值 / 未测对 vs 测得的低对 | **完成** | `SIMILARITY_VIEW_KEYS` 显式版本化（新增 pearson_corr/kendall_tau/quantile_overlap/regime 等）；`SimilarityViewRegistry`（防 typo fail-closed）；`EdgeAffinityPolicy`（多视图融合，UNKNOWN→None 非 0） |
| DLIB-FA-006 | Fingerprint + ANN 索引工件 | **完成** | `contracts/fingerprint.py`：`SimilarityFingerprintArtifact`、`ANNIndexArtifact`（EXACT_FLAT vs APPROXIMATE 显式） |
| DLIB-FA-007 | 图完整度类 only ANN-Refined 或 Exact 可入 Leiden | **完成** | `contracts/cluster_governance.py`：`GraphCompletenessClass`、`UnknownEdgeSemantics`、`SimilarityGraphArtifact.is_certified` |
| DLIB-FA-008 | 深度不可变 artifact | **完成** | `ClusterArtifact.assignments` 快照为 MappingProxyType；`FamilyLineage.members/roots/leaves/relations` 为 frozenset/tuple |
| DLIB-FA-009 | ClusterSetVersion 一成不变，每次刷新新版本 | **完成** | `ClusterSetVersionArtifact`、`LogicalCluster`（CL_PV_MOM_017 稳定 id，与不稳定 label 解耦）、`ClusterVersionArtifact`、`ClusterVersionMatcher`（UNCHANGED/MIGRATED/SPLIT/MERGED/NEW/DISSOLVED） |
| DLIB-FA-010 | 增量分配从不就地改动 ClusterSetVersion | **完成** | `IncrementalClusterAssignment` + `IncrementalAssignmentKind`（ASSIGNED/AMBIGUOUS/BRIDGE/SINGLETON/OUTLIER/PENDING_GLOBAL_REFRESH） |
| DLIB-FA-011 | 不算出的分辨率选择（非 max-modularity） | **完成** | `ClusterResolutionSelector`（STABILITY_KEYS 驱动 + modularity 仅平局决定） |
| DLIB-FA-012 | 库定义 / 成员 / 版本 | **完成** | `contracts/library_governance.py`：`FactorLibraryDefinition`、`FactorLibraryMembership`（assembly_score 有限）、`FactorLibraryVersionArtifact`（cluster_set_version_ref 必填）、`NewLibraryProposal` |
| DLIB-FA-013 | 类型化组装证据 + 丢失质量为 NOT_ELIGIBLE | **完成** | `contracts/assembly_evidence.py`：`AssemblyEvidence`、`LibraryCandidateEvidence`、`EvidenceMaturity`、`AssemblyPolicy`（替换 MMR 魔数 0.5，见 assembly/engine.py `_diverse_mmr_rank`） |
| DLIB-FA-014 | MAX_IC 绑定 EvaluationArtifactRef 非裸浮点 | **完成** | `RepresentativeSelection.ic_evidence_refs` + `ic_window_ref`/`ic_snapshot_ref`/`ic_split_ref` |
| DLIB-FA-015 | IC_WEIGHTED / INV_VAR 权重训练分割绑定 | **完成** | `AggregationFitArtifact`（fit_split_ref 必填） |
| DLIB-FA-016 | 消失 0.0 相似度必须为刻意记录 | **完成** | `SimilarityResult.measurement_status` + `value_is_admissible`；`SimilarityMeasurementStatus` 枚举 |
| DLIB-FA-017 | ESM 风格直接导出，无命名空间解析 | **完成** | `FactorValueIdentity` 增加 `status`（FactorValueStatus 枚举）+ sample_ratio/confidence/window_ref/method_version；顶层/contracts 直接导出 |
| DLIB-FA-018 | 复合因子评估（vwap→vwap 收益口径） | **完成** | `aggregation/composite.py`：CCCC 组合（Horizon/Orientation/Regime 映射）+ weight-only 会计诊断，明确非回测权威（QuantPlatform compositor 目标） |

## 权威性边界（完成清理后）

- **FO 拥有**：Pareto（治疗/搜索）、MultiFidelity、Plateau（搜索停止）、Mutation（执行）、Search、Sealed Test——FA 均为消费者/引用适配器。
- **FA 拥有**：`pareto.py` 组装专用 dominance 比较器；治理/生命周期账本；规范工件（FactorSetArtifact/AdmissionArtifact/SimilarityArtifact/TreatmentSelectionArtifact/ClusterSetVersion/库版本）。
- **FA 不拥有**（仍缺实现，标记为平台目标）：复合因子长/短回测（QuantPlatform compositor）、IC 原始计算（QE）、解析器/规范哈希（FE）。

## 变更文件清单

新增 `factor_assets/`:
- `contracts/_canonical.py`
- `contracts/cluster_governance.py`
- `contracts/library_governance.py`
- `contracts/assembly_evidence.py`
- `contracts/fingerprint.py`
- `optimizer/FA_OPTIMIZER_MIGRATION_MATRIX.md`
- `aggregation/composite.py`

修改 `factor_assets/`:
- `__init__.py`
- `contracts/__init__.py`、`contracts/similarity.py`、`contracts/factor_set.py`（未直接改但为规范工件文档主体）
- `identity/canonical.py`
- `clustering/families.py`、`clustering/lineage.py`
- `assembly/engine.py`
- `aggregation/__init__.py`、`aggregation/specs.py`、`aggregation/representatives.py`
- `similarity/exact.py`、`similarity/__init__.py`
- `campaigns/__init__.py`、`campaigns/campaign_coordinator.py`、`campaigns/ledger_adapter.py`
- `optimizer/__init__.py`、`optimizer/multifidelity.py`、`optimizer/plateau.py`、`optimizer/frozen_candidate.py`

新增测试 `factor_assets/tests/`:
- `test_dlib_fa_001_optimizer_authority.py`（6 测试）
- `test_dlib_fa_002_003_campaign_api.py`（5 测试）
- `test_dlib_fa_004_011_cluster_governance.py`（15 测试）
- `test_dlib_fa_004_017_identity.py`（4 测试）
- `test_dlib_fa_012_016_library_evidence.py`（12 测试）
- `test_dlib_fa_018_composite.py`（7 测试）

**验证：`880 passed, 43 skipped` 全绿。**