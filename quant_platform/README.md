# quant_platform — 平台集成层（纯合同 DTO + auth/RBAC + 编排骨架）

量化研究平台的**薄集成层**：纯 stdlib DTO 合同（24 个模块，`__all__` **135 项**）、
auth/RBAC、outbox/worker/orchestrator 骨架、存储适配器、Observability 与报告导出。
它是域包的**平台侧边界** —— 域包不得 import 它。

**定位:** 企业级 A 股横截面日频多因子量化项目的**平台合同层** —— 13 个库的 DTO 在这里
统一成 135 项 frozen dataclass，前端类型镜像它，全平台"类型即合同"。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/quant_platform (私有)

> ⚠️ 状态：**DRAFT / WIP** —— Control 与 Metadata 平面未建（无 Postgres、无 workflow runtime、
> 无统一 registry）。合同 **未 V1_FROZEN**，待跨包 reconcile（`[RECONCILE]` 标记遍布 docstring）。

---

## 它是什么 / 不是什么

**做什么：** 只做"合同 + 适配 + 编排骨架"：跨库 DTO 合同、认证/授权、事务型 outbox、
作业/工作流状态机、COS 存储适配、候选流水线编排、健康/指标/审计。

**不做什么：** **禁止重算量化逻辑** —— 不算 RankIC、不落第二套因子湖、不做第二套元数据注册表。
域权威一律留在域包（factor_engine / quant_evaluator / factor_assets ...）。

## 拥有什么（功能全清单）

### 合同层 `app/contracts/`（24 模块，纯 stdlib frozen dataclasses + Protocol）

PURE-DTO 硬性规则：**禁止 fastapi/sqlalchemy/pydantic 进 contracts**（`tests/test_smoke.py`
断言零第三方依赖可 import）。主要 DTO（`__all__` 135 项）：

| 族 | DTO |
|---|---|
| **制品** | `ArtifactRef`（content_hash=对象字节 sha256，发布方报告；15 种 ARTIFACT_TYPES）、`DailySnapshot`/`SnapshotManifest`+`verify_snapshot`（防篡改校验） |
| **事件/作业** | `EventEnvelope`（统一信封，payload 深冻结；19 种 EVENT_TYPE）、`JobSpec`/`JobRecord`/`JobAttempt`/`JobResult`（四件套，JobStatus 10 态 + ErrorClass 10 类含 `retryable`） |
| **工作流** | `WorkflowSpec`/`WorkflowRun`/`WorkflowStatus`+`WorkflowBackend`（Protocol，Temporal 未接） |
| **因子治理** | `FactorCandidateManifest`（spec §10.2）、`LifecycleState`（8 态）、`QRPPipelineStage`（11 态）、`HealthState`（6 态）、`IdentityRef`/`FactorDefinitionRef`/`FactorValueRef`/`EvaluationRef`/`TreatmentRef`（身份**携带引用**，平台永不铸造身份） |
| **准入** | `AdmissionRequest`/`AdmissionVerdict`/`AdmissionAuthority`/`RefuseAdmission`（准入委托 seam） |
| **聚类/库** | `ClusterVersion`/`ClusterSetVersion`/`LogicalCluster`/`ClusterMembership`/`ClusterLineageEdge`/`SimilarityGraphVersion`/`FactorLibraryVersion`/`LibraryMembership`（分层聚类模型） |
| **特征集** | `FeatureSetArtifact`/`FeatureSetVersion`/`FeatureMemberRef`/`FeatureSetDiff`（8 类 diff）/`RetrainPolicy`/`ModelRetrainRequiredEvent` |
| **授权** | `Permission`(26)/`Role`(5)/`Team`(5)/`SecurityClassification`(5)/`Grant`/`ResourceScope`/`HumanPrincipal`/`WorkloadPrincipal` |
| **时序/证据** | `TimingContract`（5 个时间字段）+ `EvidenceStatus`（含 **LABEL_NOT_MATURE**，永不全零填充） |
| **回测/模型** | `BacktestRequest`/`BacktestArtifactRef`/`ModelDatasetRequest`/`ModelTrainingRequest`/`ModelArtifactRef`+`BacktestProvider` |
| **版本链** | `ModelVersion`/`LabelDefinition`/`DataSnapshot`/`SplitPlan`/`ModelWeight`（四层版本链） |
| **基础** | `_contenthash`（canonical_str/content_hash 语义身份哈希，拒绝 NaN/Inf/naive datetime） |

### DB schema `app/db/schema.py`
- `schema.py`（DDL）+ `SqliteDb`（可运行默认后端）+ `PostgresDb`/`PostgresDialect`（适配器 seam，
  psycopg2 缺装抛 `PostgresBackendUnavailable`）。
- DDL 为 **PostgreSQL 兼容**（TEXT 代替 JSONB/UUID），SQLite 可直接跑。
- **实表 31 张**（注意：`SCHEMA_DDL` tuple 收集 31 张，另有 `factor_library_versions`/
  `factor_library_members` 两张表的 CREATE 已定义但未纳入 `create_schema()` 应用集合）：
  principals、human_users、workload_principals、teams、team_members、roles、permissions、
  artifacts、artifact_lineage、jobs、job_attempts、job_results、workflow_runs、outbox_events、
  inbox_events、factor_candidates、similarity_graph_versions、cluster_set_versions、
  logical_clusters、cluster_versions、cluster_memberships、cluster_lineage、factor_libraries、
  factor_library_versions、factor_library_members、feature_sets、feature_set_versions、
  feature_set_members、production_pointers、sessions、audit_logs。

### 安全 `app/security/`
- **认证**：session-token，HMAC-SHA256 签名的 HTTP-only cookie `qrp_session`
  （samesite=lax，8h TTL，格式 `<session_id>.<principal_id>.<expiry>.<hex_hmac>`）。
- **密码**：scrypt（n=2^14/r=8/p=1，常量时间比较）；会话行存 sessions 表，支持撤销。
- **RBAC**：`Authorization = grant(principal, permission, resource_scope)`，无"角色可读一切"裸规则；
  26 Permission / 5 角色（MEMBER/LEAD/CORE/ADMIN/SERVICE）/ 5 团队 / 5 级安全分类
  （PUBLIC_METADATA→PRODUCTION_ONLY），每权限有最低分类门槛（如 `factor:read_formula` 需 ≥CONFIDENTIAL_ALPHA）；
  10 个服务工作负载最小权限矩阵（`WORKLOAD_LEAST_PRIVILEGE` + COS 前缀作用域）。
- **审计**：`FormulaAccessGuard` 每次调用（允许/拒绝）都记 audit_logs（spec §23，403 而非 200+null）。

### worker/outbox `app/worker/` + `app/outbox.py`
- **JobRunner**（in-memory）：按 idempotency_key 幂等执行，成功→SUCCEEDED，
  `JobError(ErrorClass.retryable)`→FAILED_RETRYABLE，否则 FAILED_TERMINAL，attempt 累积。
- **事务型 outbox**：事件行与状态更新**同一事务**写入；后台 claim 并发守卫
  （单条 `UPDATE...WHERE status='pending'`，SQLite BEGIN IMMEDIATE / PG 行锁）；
  生命周期 pending→claimed→sent / 失败回 pending 重试 / dead_letter；
  inbox 按 idempotency_key 去重，四态 RECEIVED/PROCESSING/DONE/FAILED + 死信。
- **WorkerLoop**：in-memory publish pump，claim-once 保证同 event_id 至多投递一次，
  PublishTimeoutError 支持 in-flight 租约。
- **WorkflowLifecycle**：PENDING→RUNNING→终态（SUCCEEDED/FAILED/CANCELED/TIMED_OUT）严格转移守卫。

### 编排 `app/orchestrator.py`
candidate → evaluation → admission 全流水线骨架。

### 候选 `app/candidate/`
`ingest.py` + `candidate.py`（归一化/对账/批量指纹）。

### 存储 `app/storage/`
- `COSArtifactPublisher`：字节→COS→HEAD 校验→ArtifactRef；data_access 不可导入时
  `COS_AVAILABLE=False` + `COSUnavailableError`（延迟构建 COSObjectStore）。
- `DataAccessStorageAdapter`（`app/adapters/data_access_storage.py`）：实现 ArtifactStoragePort，
  两阶段 `ObjectStoreGenerationPublisher`（payload→manifest→CURRENT.json 指针最后翻转，
  读侧永远只见完整代）；大对象 multipart 路由；安全上下文路由（ContextVar 凭证）。
- `ArtifactRegistry`：sqlite/memory 统一，按 (content_hash, artifact_type) 幂等。
- `LocalArtifactCacheImpl`：磁盘 LRU，store 校验和 fail-closed，命中率指标。
- `access_guard`：ClassificationAccessGate + FormulaAccessGuardImpl（审计落库）。

### Observability `app/observability/`
`health.py` + `metrics.py` + `collectors.py`。

### 报告导出 `app/export/`
`report_artifact.py`（报告导出制品，QRP-P10）。

### API `app/api/app.py`
FastAPI `create_app()`（注入 Db + server secret），4 条路由：
`POST /auth/login`、`POST /auth/logout`、`GET /auth/me`、`GET /health`。无领域逻辑。

## 已知缺口（见 `docs/GAP_ANALYSIS.md` / `docs/CURRENT_ARCHITECTURE.md`）

- **Control Plane**（workflow runtime、生产 outbox、retry/promotion）— 未建。
- **Metadata Plane**（Postgres 统一 registry、factor catalog）— 未建。
- `research_platform` 旧双 artifact authority 待迁移。

## 设计规则（硬性）

- 平台**禁止重算量化逻辑**；域权威留在域包。
- `quant_platform.app.contracts` 零第三方依赖可 import（`tests/test_smoke.py` 断言）。
- 平台 ↔ 域翻译发生在 worker/adapters：`Platform DTO ↔ Domain Native Contract`
  （如 `FactorCandidateManifest` ↔ FA treatment artifact、`BacktestRequest` ↔
  vectorbt_qs `BacktestRequest`、model DTO ↔ modeling artifact）。

## 依赖与接口（谁 import 谁）

- **依赖**：`data_access`（`app/adapters/data_access_storage.py`、
  `data_access.read.object_store` 等，guarded import，缺装可降级）。
- **被谁调用**：`platform_web`（前端类型即 contracts DTO）。

## 安装与测试

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/quant_platform.git
cd quant_platform
pip install -e .
# 测试须从 monorepo 根跑（否则 quant_platform 包不可导入，4 个收集 ERROR）：
cd /repo-root && python3 -m pytest quant_platform/tests/ -q   # 367 passed + 8 skipped
```

## 相关仓库

- **所有域包** — 本层是它们的平台侧边界
- **platform_web** — 前端 types 镜像 `app/contracts/*`
- **data_access** — 存储适配器来源
