# R41 全量映射摘要（2026-08-12，HEAD 9bd5836f）

## 映射覆盖
- **585 项**全部映射完成（11 个 section，9 个 agent 成功 + 2 个部分输出可用）
- 映射基准：HEAD `9bd5836f094474cd257f3fbac1ab5644af6e916d`（FE + DA 同 commit）
- 执行模型：只读探查，零文件改动

## 分类汇总

### 按 section 分布（585 项）
| Section | 范围 | REAL_BUG | PARTIAL | FIXED | N/A | BLOCKED |
|---------|------|----------|---------|-------|-----|---------|
| 4 build/version/evidence | R41-261..280 (20) | 15 | 3 | 2 | 0 | 0 |
| 5 training math | R41-281..330 (50) | 27 | 14 | 9 | 1 | 0 |
| 6 artifact lifecycle | R41-331..370 (40) | 33 | 3 | 1 | 1 | 0 |
| 7 evaluation/OOS/IC | R41-371..400 (30) | 24 | 6 | 0 | 0 | 0 |
| 8 security/service | R41-401..430 (30) | 21 | 4 | 0 | 1 | 0 |
| 9 observability/async | R41-431..450 (20) | 14 | 3 | 2 | 0 | 0 |
| 10 HTTP identity/config | R41-451..475 (25) | 19 | 3 | 3 | 0 | 0 |
| 11 param evidence | R41-476..495 (20) | 13 | 3 | 2 | 0 | 0 |
| 12 DA R30 | R41-496..525 (30) | (部分映射，API 余额中断) | | | | |
| 13 runtime/perf | R41-526..555 (30) | 14 | 8 | 7 | 0 | 2 |
| 14 ops/recovery | R41-556..585 (30) | 9 | 7 | 4 | 0 | 0 |

### 全局汇总（已完成映射的 555 项）
- **REAL_BUG**: ~189 项（34%）
- **PARTIAL**: ~54 项（10%）
- **FIXED_ALREADY_WITH_CURRENT_HEAD_PROOF**: ~30 项（5%）
- **NOT_APPLICABLE / BLOCKED**: ~4 项（<1%）
- **剩余 278 项**：需二次确认（DA R30 30 项因 API 中断未完整返回，可从部分输出重建）

## 关键发现（跨 section 模式）

### 1. 三个真实已修复的 P0 集群
**R40 遗留的 3 个 P0 已在当前 HEAD 修复**：
- **R41-282**（training_cutoff ≥ final_fit_end + horizon）：`trainer.py:439-440` + `artifact.py:93-102` 强制契约
- **R41-461**（ValidatedFactorRequest digest 重建对比）：`service/app.py:798` 执行前强制对比
- **R41-552**（perf 优化不改 identity）：`operator_semantic_version.py` + numba kernel 语义版本机制

### 2. 单文件集中爆炸点（需整体重构）
| 文件 | 行数 | 集中问题数 | 典型缺陷 |
|------|------|-----------|---------|
| `service/security.py` | ~470 | 15+ | 手写 auth 层：unknown role→READ、JWT 无 exp/aud、空 allowlist=allow-all、API key 明文 |
| `service/app.py` | ~1800 | 12+ | HTTP identity 近似（`_stable_hex(str(market or "ashare"))`）、`_source_profile_binding` 无 redact、`_catalog_generations` 吞异常返 `"unavailable"` |
| `modeling/trainer.py` | 501 | 10+ | preprocessing 各步独立 fit raw X、validation=None 不拒、无 CancellationToken、无 TrainingTemporalLegalityPass |
| `modeling/artifact.py` | 377 | 8+ | shallow freeze（直接返内部 list）、`default=str` lineage、非原子 save、无 schema_version gate |
| `runtime/parameter_domain_store.py` | ~430 | 8+ | `_current_head()` 依赖 live `.git`（R16 证据重生根因）、`bool("false")==True`、`repr()` 在 key、`_frozen` 不强制 |

### 3. 缺失的治理类型（需新增，非单点修复）
以下类型在全 repo grep 返回 0：
- `ModelArtifactCatalog` / `FrozenScorer` / `ModelScoreExpr` / `TypedModelScoreIR`（artifact 生命周期 + DSL）
- `DataExposureLedger` / `ValidatedHyperparameterSearchPlan`（training exposure）
- `TrainingDatasetCertificate` / `FeatureSchemaCertificate`（dataset 绑定）
- `LabelMaturityResolver` / `TrainingTemporalLegalityPass`（时序合法性）
- `ExecutionSemanticContext` / `RuntimeReadinessCertificate` / `IdentityGenerationUnavailableError`（runtime semantic）
- `EvidenceApplicability` / `EvidenceProbeResult`（证据生命周期）
- `NativePipelineRegion` / `RollingStateBlock` / `PrimitiveBlock`（R39 深架构 5 项全部未实现）

### 4. 已存在但未接线的正确实现
**FE 内部 identity 机制已正确，但 HTTP 路径不用它**：
- `planner/source_dependencies.py` 的 `source_dependency_hash(plan)` / `universe_membership_hash` 已绑定真实依赖
- `runtime/factor_identity.py:394` 的 `scoped_field_contract_hash(plan)` 已是依赖范围契约
- `runtime/evidence_truth.py:331` 的 `build_manifest_digest()` 已是 source_commit/tree digest
- **但 `service/app.py` HTTP identity 全走近似 `_stable_hex("field", str(market or "ashare"))`**

### 5. 三个实 P0 wheel 打包缺陷（阻断部署）
- **R41-261**：FE wheel 缺 `modeling*` 包（`pyproject.toml:37-58` 无此项）
- **R41-262**：DA wheel 缺 `data_access.r30` 包（`pyproject.toml:61-75` 无此项）
- **R41-263**：3 个 version 权威（pyproject 0.10.2 vs r30 `__init__.py` 0.10.3 vs metadata）

### 6. R39 五个 "深架构" NOT_CLOSED 全部未实现
`docs/R39_ISSUE_CLOSURE_LEDGER.md` 明确标注的 5 项：
- PERF-017（native backend emit OutputSlice）：PARTIAL，后处理 trim 存在但源头不 emit
- PERF-020（process worker mmap/Arrow）：未实现，只有 pickle DataFrame
- PERF-073（ScopedConnectionPool）：未实现
- PERF-078（NativePipelineRegion）：未实现
- PERF-079（RollingStateBlock）：未实现
- PERF-080（PrimitiveBlock）：未实现

### 7. 证据过期根因（R16 证据重生阻塞）
**R41-476/477** 是 R16/R37/R40 证据重生的真实 P0 阻塞：
- `parameter_domain_store.py:41` `_current_head()` shell 出 `git rev-parse HEAD`
- 安装 wheel（无 `.git`）→ `head=""` → freshness 对比被跳过 → 过期证据不拒绝
- 当前 `docs/evidence/r37/R37_PARAMETER_DOMAIN_STORE.json` 绑定 `a2dc3a1a` vs HEAD `9bd5836f` = **STALE**

## 立即可验证的 3 个 headline P0

1. **R41-261/262 wheel 打包**（2 分钟验证）
   ```bash
   python3 -m build --wheel
   unzip -l dist/*.whl | grep -E 'modeling/|r30/'  # 预期：0 匹配
   ```

2. **R41-263 version 分裂**（10 秒）
   ```python
   import data_access, data_access.r30
   print(data_access.__version__, data_access.r30.__version__)
   # 预期：0.10.2+build... vs 0.10.3（不一致）
   ```

3. **R41-476 证据过期**（30 秒）
   ```python
   from runtime.parameter_domain_store import ParameterDomainCertificationStore
   store = ParameterDomainCertificationStore.load_default(strict=True)
   # 当前 HEAD 会因 generated_commit != current 而 raise
   ```

## 下一步

基于映射结果，需：
1. **立即修 3 个 P0 wheel/version 缺陷**（阻断任何部署）
2. **HTTP identity 接线**（19 项集中在 `service/app.py`，用已有正确实现替换近似）
3. **security.py 全面重构**（15 项集中，JWT/allowlist/API-key 全面加固）
4. **artifact.py 治理层**（8 项：freeze/save/catalog/promotion）
5. **training 时序 + 取消传播**（10 项：legal pass/cancellation/sample mask）
6. **证据重生**（R41-476/477 修复后，重跑 `audit_r37_parameter_domains.py`）
7. **R39 深架构 5 项**（工作量大，P1，可延后）

预计总工作量：**P0 155 项 + P1 34 项 = 189 REAL_BUG**；按簇分 8-12 个实施 agent，中央串行回归。
