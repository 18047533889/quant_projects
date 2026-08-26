
# Quant Research Platform：从因子发现到模型前全链路自动化平台实施总规范

> 文档用途：直接交给 AI / Claude Code / 多 Agent 工程团队执行。
>
> 目标：把现有“量化研究与生产报告中心”的静态 HTML 报告体系，升级为企业级、可登录、可权限控制、可自动发现 COS 新因子、可驱动 FactorEngine / DataAccess / QuantEvaluator / FactorPreprocess / FactorOptimizer / FactorAssets 的 **Quant Research Platform**。平台必须覆盖从“因子产生”到“模型正式训练前 FeatureSet 形成”的全链路，并为后续模型训练、回测、组合优化接入保留标准边界。
>
> 本文是实施规范，不是概念方案。除非本地代码与本文存在客观冲突，否则按本文执行；若冲突，先通过代码审计确认真实现状，再在不破坏本文核心架构原则的前提下做兼容迁移。

---

# 0. 执行前硬规则

## 0.1 本地工作树是唯一开发真源

用户本次给出的本地目标路径为：

```text
/home/sunhaiwei/quant_projects
```

现有报告中心目标目录为：

```text
/home/sunhaiwei/quant_projects/量化研究与生产报告中心
```

AI 开始前必须执行并记录：

```bash
pwd
realpath .
git rev-parse --show-toplevel
git status --short
git rev-parse HEAD
```

如果真实 repo root 与上述路径不一致：

1. 以当前实际工作树为准；
2. 不要因此停止；
3. 在实施记录里写明实际 root；
4. GitHub 只用于参考/推送，不允许把 GitHub snapshot 当成本地修改目标。

禁止为了“回到干净状态”执行高风险命令：

```text
git reset --hard
git clean -fd
git checkout -- .
git restore .
git stash
```

除非用户明确授权。

## 0.2 先审计，后修改；但不是“只写计划不实现”

第一阶段必须扫描：

```text
quant_evaluator/
factor_preprocess/
factor_optimizer/
factor_assets/
factor_engine/
dataaccess/
factor_mining/
量化研究与生产报告中心/
现有 pyproject / Docker / CI / scripts / service / registry / reporting
```

必须识别已有能力，优先复用，禁止重复造同义模块。

审计完成后立即进入实施，不要只输出 architecture plan 然后停止。

## 0.3 不允许为了平台把现有独立库揉成一个大包

最终形态是：

```text
一个 monorepo
+
多个可独立安装/测试/发布的 domain package
+
一个负责串联它们的 platform/control-plane
```

以下库继续保持独立边界：

```text
dataaccess
factor_engine
quant_evaluator
factor_preprocess
factor_optimizer
factor_assets
factor_mining
modeling（若已有）
```

平台只能通过 adapter / protocol / typed request / typed artifact 与这些库连接。

禁止：

```python
# 错误示例
platform.web.routes.factor -> 直接调用 factor_engine 内部 private function
platform.api -> 直接 import quant_evaluator.metrics.xxx 然后现场重算 RankIC
factor_assets -> 强依赖 FastAPI / React / Temporal
```

## 0.4 每个阶段都必须有测试和可回滚性

禁止一次性重写全部仓库后再测试。

每个 phase 必须：

```text
实现
→ unit test
→ integration test
→ migration test
→ regression test
→ 更新文档
→ 再进入下一 phase
```

---

# 1. 现状判断与这次工程的真正目标

当前“量化研究与生产报告中心”已经有正确的雏形：

- 指标字典；
- 标签字典；
- 图表字典；
- 评分体系；
- 因子/模型/组合优化报告；
- EXP / NEXP 两套视图；
- EvaluationBundle 作为报告输入；
- Registry 是真源、HTML 是 View；
- Streaming / Atomic Publish / Asset Lifecycle 等概念。

问题是它现在仍然属于 **static report center**：

```text
一堆 HTML
+
按日报/周报/部署报告组织信息
```

用户真正需要的是：

```text
Quant Research Platform
=
Research Control Plane
+
Factor Asset Platform
+
Workflow Orchestrator
+
Metadata Registry
+
Web UI
+
Permission / Audit / Governance
```

因此本次改造不是“HTML V2”，而是新增一个正式平台层。

---

# 2. 最终目标范围

本阶段最终需要自动化到：

```text
LLM / Human / Factor Mining
        ↓
Factor Candidate
        ↓
COS Ingestion
        ↓
Validation / Identity / Dedup
        ↓
DataAccess + FactorEngine Materialization
        ↓
Factor Value Artifact 写 COS
        ↓
QuantEvaluator Raw Evaluation
        ↓
FactorPreprocess / FactorOptimizer Treatment Search
        ↓
QuantEvaluator Treated Evaluation
        ↓
FactorAssets Novelty / Admission
        ↓
Similarity Graph
        ↓
Incremental Cluster Assignment
        ↓
Periodic Global Reclustering
        ↓
Factor Library Candidate
        ↓
Factor Library Version Promotion
        ↓
FeatureSet Version
        ↓
Model Retrain Required Event
```

本工程需要把模型前的全部流程打通。

**本阶段不要求把生产模型训练和实盘执行完整实现进平台核心。**

但必须预留正式 contract：

```text
FeatureSetArtifact
ModelDatasetRequest
ModelTrainingRequest
ModelArtifactRef
BacktestRequest
BacktestArtifactRef
```

确保后续无需重构平台即可继续接模型、回测、组合优化。

---

# 3. 总体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                     Quant Web Platform                      │
│                                                             │
│ Dashboard / Factors / Clusters / Libraries / Campaigns      │
│ FeatureSets / Models / Jobs / Data / Health / Standards     │
│ Users / Roles / Audit / Exports                             │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTPS / REST / WebSocket/SSE
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        Platform API                         │
│                                                             │
│ Auth / RBAC / Search / Query / Metadata / Presigned Access  │
│ No heavy quant computation inside request process           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       Control Plane                         │
│                                                             │
│ Event Ingestion / Workflow / Scheduler / Retry / State      │
│ Idempotency / Outbox / Reconciliation / Promotion Policy    │
└───────────┬───────────┬───────────┬───────────┬─────────────┘
            │           │           │           │
            ▼           ▼           ▼           ▼
        FE Worker    QE Worker    FP/FO Worker  FA Worker
            │           │           │           │
            └───────────┴───────────┴───────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         Data Plane                          │
│                                                             │
│ PostgreSQL                         COS Object Storage         │
│ metadata / jobs / versions         immutable large artifacts │
│ permissions / lineage              factor values / matrices  │
│ cluster / library registry         evaluation / model blobs  │
└─────────────────────────────────────────────────────────────┘
```

## 3.1 三个平面必须分开

### Control Plane

负责：

- 什么时候跑；
- 谁跑；
- 当前跑到哪一步；
- 失败重试；
- 版本 promotion；
- 依赖关系；
- 状态机。

### Metadata Plane

负责：

- 有哪些 factor；
- 当前版本是什么；
- 哪个 cluster；
- 哪个 library；
- artifact 在 COS 哪里；
- 当前 health；
- 哪个 job 产生；
- 谁看过公式。

### Data Plane

负责大体量数据：

- factor values；
- large evaluation series；
- feature matrices；
- backtest series；
- model binaries；
- large logs；
- chart data artifacts。

平台查询“有多少因子”时绝不能扫 COS。

平台查询必须访问 Metadata Registry。

---

# 4. 推荐仓库结构

不要立即机械搬目录，先适配当前结构。目标可逐步演化为：

```text
quant_projects/
│
├── dataaccess/
├── factor_engine/
├── quant_evaluator/
├── factor_preprocess/
├── factor_optimizer/
├── factor_assets/
├── factor_mining/
├── modeling/
│
├── platform/
│   ├── README.md
│   ├── pyproject.toml                 # platform backend package
│   │
│   ├── app/
│   │   ├── api/
│   │   │   ├── main.py
│   │   │   ├── dependencies.py
│   │   │   └── routes/
│   │   │       ├── auth.py
│   │   │       ├── dashboard.py
│   │   │       ├── factors.py
│   │   │       ├── evaluations.py
│   │   │       ├── clusters.py
│   │   │       ├── libraries.py
│   │   │       ├── feature_sets.py
│   │   │       ├── jobs.py
│   │   │       ├── artifacts.py
│   │   │       ├── standards.py
│   │   │       ├── exports.py
│   │   │       └── admin.py
│   │   │
│   │   ├── auth/
│   │   ├── contracts/
│   │   ├── domain_views/
│   │   ├── registry/
│   │   ├── persistence/
│   │   ├── storage/
│   │   ├── workflows/
│   │   ├── workers/
│   │   ├── adapters/
│   │   ├── services/
│   │   ├── events/
│   │   ├── observability/
│   │   └── settings/
│   │
│   ├── migrations/
│   ├── tests/
│   ├── docker/
│   └── scripts/
│
├── platform_web/
│   ├── package.json
│   ├── src/
│   │   ├── app/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── features/
│   │   ├── api/
│   │   ├── auth/
│   │   ├── charts/
│   │   └── types/
│   └── tests/
│
├── integration_tests/
│   └── platform_pipeline/
│
├── legacy_reports/
│   └── 量化研究与生产报告中心/
│
└── deployment/
    ├── docker-compose.yml
    ├── nginx/
    └── env.example
```

如果当前报告中心目录名真实存在尾随空格，必须用安全的 `git mv` 修复，不要通过复制后删除造成历史丢失。

---

# 5. 依赖规则

建立 CI 级别依赖约束。

允许：

```text
platform workers -> factor_engine
platform workers -> quant_evaluator
platform workers -> factor_preprocess
platform workers -> factor_optimizer
platform workers -> factor_assets
platform workers -> dataaccess
```

禁止：

```text
factor_engine -> platform
quant_evaluator -> platform
factor_assets -> platform
factor_preprocess -> FastAPI
factor_optimizer -> Temporal
```

Web 前端只能访问 Platform API。

浏览器绝不能持有：

- COS SecretKey；
- 数据库密码；
- FE worker credentials；
- 公式解密密钥。

---

# 6. Integration Contract 设计

不要新建一个无限膨胀的 `common/utils/core`。

平台层可以存在一个很薄的 integration DTO 层：

```text
platform/app/contracts/
```

它的职责只有：

- 外部 API DTO；
- workflow request；
- ArtifactRef；
- EventEnvelope；
- domain adapter protocol。

现有 domain package 不需要 import 它。

worker adapter 负责：

```text
Platform DTO
    ↕
Domain Native Contract
```

---

# 7. Artifact 是整个系统的核心抽象

## 7.1 禁止跨模块直接传物理文件路径

错误：

```python
qe.evaluate('/cos/factor/F001.parquet')
```

正确：

```python
qe_worker.evaluate(FactorValueArtifactRef(...))
```

## 7.2 ArtifactRef 最少字段

实现强类型 immutable DTO：

```python
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    schema_version: str
    content_hash: str
    storage_uri: str
    size_bytes: int
    created_at: datetime
    producer_type: str
    producer_version: str
    snapshot_id: str | None
```

至少支持 artifact type：

```text
FACTOR_CANDIDATE
FACTOR_DEFINITION
FACTOR_VALUE
EVALUATION_BUNDLE
TREATMENT_SELECTION
SIMILARITY_FINGERPRINT
SIMILARITY_GRAPH
CLUSTER_VERSION
CLUSTER_ASSIGNMENT
FACTOR_LIBRARY_VERSION
FEATURE_SET
MODEL_DATASET
MODEL
BACKTEST
REPORT_EXPORT
```

## 7.3 Artifact 必须 immutable

同一 `artifact_id/content_hash` 对应的数据不能原地覆盖。

更新意味着产生新 artifact。

## 7.4 Artifact 注册必须采用两阶段提交

```text
write temp object
→ checksum
→ upload/rename/finalize COS object
→ verify object metadata/hash
→ insert artifact metadata DB
→ publish event
```

不能先把数据库写成 READY，再慢慢上传文件。

---

# 8. Identity 与 Version 体系

这是本次工程最容易出错的地方之一。

禁止用一个 `factor_version=v2` 表达全部变化。

必须至少分：

```text
FactorDefinitionIdentity
FactorValueIdentity
EvaluationIdentity
TreatmentIdentity
SimilarityGraphIdentity
ClusterVersionIdentity
FactorLibraryVersionIdentity
FeatureSetIdentity
ModelDatasetIdentity
ModelIdentity
```

## 8.1 FactorDefinitionIdentity

由以下语义 canonicalize 后 hash：

```text
formula / AST
operator semantics version
frequency
market
input schema requirements
parameter values
calculation semantics
```

显示名称不进入 identity。

## 8.2 FactorValueIdentity

建议：

```text
FactorDefinitionIdentity
+
DataSnapshotIdentity
+
UniverseIdentity
+
CalculationSpecIdentity
+
TimingSemanticsIdentity
```

## 8.3 EvaluationIdentity

```text
FactorValueIdentity
+
EvaluationPolicyIdentity
+
LabelDefinitionIdentity
+
EvaluationProfileIdentity
```

## 8.4 TreatmentIdentity

```text
source_factor_value_id
+
ordered preprocessing recipe
+
fit boundary
+
fit state content hash
+
neutralization schema
```

## 8.5 ClusterVersionIdentity

```text
member universe snapshot
+
similarity graph version
+
clustering algorithm/version
+
parameters
+
seed
+
policy hash
```

## 8.6 FactorLibraryVersionIdentity

```text
logical_library_id
+
ordered/normalized factor membership identities
+
selection policy
+
cluster version
+
evidence snapshot
```

## 8.7 FeatureSetIdentity

不能只存 factor_id。

每个 member 至少绑定：

```text
factor_definition_id
factor_value semantics
selected treatment
orientation
feature channel name
required timing
```

---

# 9. Factor 生命周期状态机

数据库中状态不可使用任意字符串。

建议主 lifecycle：

```text
DISCOVERED
VALIDATING
VALIDATED
COMPILED
MATERIALIZING
MATERIALIZED
RAW_EVALUATING
RAW_EVALUATED
TREATMENT_SEARCHING
TREATMENT_SELECTED
TREATED_EVALUATING
TREATED_EVALUATED
NOVELTY_CHECKING
ADMISSION_REVIEW
CLUSTER_PENDING
CLUSTERED
LIBRARY_CANDIDATE
SHADOW
PRODUCTION_ELIGIBLE
PRODUCTION
DEGRADED
RETIRED
QUARANTINED
```

执行状态另建 JobStatus，不要和生命周期混在一起：

```text
PENDING
RUNNING
SUCCEEDED
FAILED_RETRYABLE
FAILED_TERMINAL
CANCEL_REQUESTED
CANCELLED
BLOCKED_DATA
BLOCKED_DEPENDENCY
TIMED_OUT
```

HealthState 再单独：

```text
UNKNOWN
HEALTHY
WATCH
DEGRADED
CRITICAL
STALE
```

即：

```text
LifecycleState != JobStatus != HealthState
```

---

# 10. COS Ingestion 协议

## 10.1 不允许“看见一个 .py 就认为是完整因子”

Factor Candidate 必须使用 manifest protocol。

建议对象布局：

```text
/candidates/
  market=ashare/
    source=llm/
      date=2026-08-26/
        candidate_id=FC_xxx/
          manifest.json
          factor_spec.json
          lineage.json        # optional
          evidence.json       # optional
          _READY
```

只有 `_READY` 存在且 manifest checksum 验证通过，Ingestion Service 才受理。

## 10.2 manifest.json 最少字段

```json
{
  "schema_version": "1",
  "candidate_id": "FC_...",
  "submitted_at": "...",
  "submitted_by": "llm_agent_x",
  "generator_type": "LLM_AGENT",
  "generator_version": "...",
  "market": "ASHARE",
  "frequency": "1D",
  "formula_language": "FACTOR_ENGINE_DSL",
  "factor_spec_uri": "...",
  "factor_spec_sha256": "...",
  "parent_factor_ids": [],
  "required_fields": [],
  "semantic_family_hint": "PRICE_VOLUME",
  "campaign_id": "...",
  "attempt_id": "..."
}
```

## 10.3 Ingestion 必须双通道

主通道：

```text
COS Object Event
→ ingestion event
```

补偿通道：

```text
periodic reconciliation scanner
```

Reconciliation 不允许每次全桶无脑 list。

至少维护：

```text
prefix partition
last_modified cursor
object etag
manifest hash
last_seen_at
```

必须做到“事件漏了也最终一致”。

---

# 11. 事件系统

定义统一 EventEnvelope：

```python
class EventEnvelope:
    event_id: str
    event_type: str
    schema_version: str
    occurred_at: datetime
    producer: str
    aggregate_type: str
    aggregate_id: str
    correlation_id: str
    causation_id: str | None
    payload: dict
```

核心 event：

```text
FactorCandidateDiscovered
FactorCandidateValidated
FactorDefinitionRegistered
FactorMaterializationRequested
FactorMaterialized
RawEvaluationCompleted
TreatmentSearchCompleted
TreatedEvaluationCompleted
FactorAdmissionDecided
FactorClusterAssigned
ClusterVersionPublished
FactorLibraryCandidateCreated
FactorLibraryPromoted
FeatureSetCreated
FeatureSetSemanticChanged
ModelRetrainRequired
FactorHealthChanged
ArtifactPublished
JobFailed
```

## 11.1 必须实现 Transactional Outbox

数据库状态更新与 event publication 不允许存在双写不一致。

流程：

```text
DB transaction:
  update state
  insert outbox event
commit

outbox publisher:
  publish
  mark delivered
```

这样可以避免：

```text
数据库已经 MATERIALIZED
但 FactorMaterialized event 丢失
```

---

# 12. Workflow Runtime

推荐 Temporal，但必须通过内部抽象隔离。

定义：

```python
class WorkflowBackend(Protocol):
    def start(...): ...
    def signal(...): ...
    def cancel(...): ...
    def status(...): ...
```

实现：

```text
TemporalWorkflowBackend
```

测试可有：

```text
InMemoryWorkflowBackend
```

不要让业务模块到处出现 Temporal-specific decorators。

## 12.1 一个 factor workflow 示例

```text
IngestFactorWorkflow
  1 validate_candidate
  2 register_definition
  3 request_data_snapshot
  4 compile_factor
  5 materialize_factor
  6 run_l0_quality
  7 run_l1_raw_evaluation
  8 treatment_search
  9 treated_evaluation
  10 novelty_admission
  11 incremental_cluster_assignment
  12 library_candidate_update
  13 maybe_trigger_promotion
```

## 12.2 所有 activity 必须具备

```text
idempotency key
retry policy
timeout
heartbeat
resource class
input artifact refs
output artifact refs
structured error code
```

---

# 13. Pipeline 详细阶段

# Stage 0：Candidate Validation

检查：

- manifest schema；
- hash；
- formula syntax；
- operator whitelist；
- required fields；
- market/frequency；
- obvious future leakage；
- duplicate factor definition；
- candidate provenance；
- generator campaign；
- payload completeness。

失败区分：

```text
INVALID_SCHEMA
UNKNOWN_OPERATOR
MISSING_DATA_FIELD
DUPLICATE_DEFINITION
UNSUPPORTED_FREQUENCY
LEAKAGE_RISK
CORRUPT_UPLOAD
```

Duplicate Definition 不必重新跑全链路。

应链接到已有 FactorDefinition，并记录：

```text
candidate -> existing definition
```

以便统计不同挖掘算法“重复生产率”。

# Stage 1：Factor Definition Registration

建立：

```text
Factor
FactorDefinitionVersion
GeneratorLineage
CampaignMembership
```

不要把 expression name 当 FactorIdentity。

# Stage 2：Data Resolution

DataAccess 提供：

```text
DataSnapshotRef
UniverseRef
FieldResolution
TimingPolicy
```

必须 fail closed：

- 字段缺失不能偷偷替代；
- PIT 不满足不能降级为 current data；
- universe/snapshot 不明确不能进入生产 evaluation。

# Stage 3：FactorEngine Compile / Certification

FactorEngine 输出至少：

```text
compile result
resolved operator graph
required lookback
backend plan
capability certification
formula identity
```

生产 pipeline 中未认证算子：

```text
BLOCKED / RESEARCH_ONLY
```

不能自动 fallback 到语义未知实现然后继续 promotion。

# Stage 4：Factor Materialization

执行：

```text
FactorDefinition
+
DataSnapshot
+
Universe
→
FactorValueArtifact
```

要求：

- incremental 可用；
- batch 多因子共享数据读取；
- 按 data scope 分组；
- 支持 cache hit；
- 原子发布；
- 产生 row/date/asset statistics；
- 验证 finite/missing/schema。

# Stage 5：QE L0 Data Quality Gate

所有因子必须跑，成本低。

至少：

```text
coverage
valid assets
NaN/Inf
unique ratio
tie ratio
zero ratio
constant-day ratio
staleness
outlier diagnostics
missing pattern
```

明显失败因子直接：

```text
QUARANTINED / REJECTED
```

但保留所有 evidence 和 trial lineage。

# Stage 6：QE L1 Statistical Evaluation

建议覆盖：

```text
RankIC / PearsonIC
IC mean/std/IR
HAC t / p
positive IC ratio
rolling IC
recent vs full
horizon decay
quantile monotonicity
Top-Bottom spread
stability slices
exposure
neutralized comparison
tradability proxy
multiple-testing evidence
```

不能只用 `RankIC > 0.015` 作为单硬阈值。

# Stage 7：Treatment Search

由 FactorPreprocess + FactorOptimizer 承担。

可搜索：

```text
raw
winsorization
robust scaling
rank transform
EWMA variants
causal smoothing
industry neutralization
size neutralization
combined neutralization
residualization
approved causal transforms
```

硬规则：

- 所有 transform 必须 causal 或 train-fitted；
- 禁止 full-sample leakage；
- 不允许 test information 决定 treatment；
- search trial ledger 全量保留；
- failed trials 也记入 multiple-testing count。

输出：

```text
TreatmentSelectionArtifact
```

必须记录：

```text
raw profile
candidate treatments
trial results
selection policy
winner
runner-ups
failed trials
fit boundary
```

# Stage 8：Treated QE

再跑正式 evaluation。

平台必须能在一个 factor 详情页同时看到：

```text
Raw
vs
Selected Treatment
```

防止处理把原始信号缺陷隐藏。

# Stage 9：Novelty / Admission

FactorAssets 负责。

禁止：

```text
corr > threshold => permanent reject
```

应使用：

```text
exact formula identity
→ exact result identity
→ multi-view similarity
→ residual/incremental novelty
```

高相关但存在 incremental alpha：

```text
SHADOW / ALTERNATIVE / COMPLEMENTARY
```

而不是直接删除。

# Stage 10：Incremental Similarity

新因子无需与全部 10 万因子做全矩阵 exact correlation。

```text
fingerprint
→ ANN candidate neighbor shortlist
→ exact multi-view similarity
→ sparse graph update
```

Multi-view similarity 至少考虑：

```text
rank correlation
Pearson correlation
PnL correlation
quantile overlap
top/bottom overlap
residual relationship
regime-conditional similarity
```

# Stage 11：Incremental Cluster Assignment

状态：

```text
ASSIGNED
AMBIGUOUS
BRIDGE
SINGLETON
OUTLIER
PENDING_GLOBAL_REFRESH
```

新因子分配应有置信度与 evidence。

# Stage 12：Global Cluster Refresh

不要每新增因子全量重聚类。

建议：

```text
Daily: incremental assignment
Weekly/Monthly: global graph refresh + Leiden
Triggered: graph drift threshold exceeded
```

small-K research mode 可层次聚类；large-K production 推荐 sparse graph + mature Leiden。

禁止：

- sparse graph missing edge 当 corr=0；
- arbitrary `1-|corr|` 配 Ward；
- 丢掉 isolated node；
- 把 Leiden 原始整数 label 当永久 cluster ID。

# Stage 13：Factor Library Candidate

Cluster 与 Library 必须分开。

Cluster 是“相似性结构”。

Library 是“面向某种建模用途的治理资产集合”。

一个 Factor 可以：

```text
属于一个主 cluster
但属于多个 library
```

# Stage 14：FeatureSet Promotion

Library promotion 后形成 immutable FeatureSet。

只有 FeatureSet semantic change 才应触发模型重训事件。

---

# 14. Cluster Version 与稳定 Logical Cluster ID

聚类输出不能直接展示：

```text
cluster 18
```

必须分：

```text
algorithm_cluster_label
logical_cluster_id
cluster_version_id
```

每次 global clustering 后，对上一版本做 matching。

匹配 evidence：

```text
member Jaccard
representative overlap
weighted graph overlap
centroid/fingerprint similarity
```

转换类型：

```text
UNCHANGED
MIGRATED
SPLIT
MERGED
NEW
DISSOLVED
```

逻辑 ID 示例：

```text
CL_PV_MOM_0017
```

ClusterVersion 不可原地更新。

平台要能展示：

```text
v43 -> v44
CL_017 split into CL_017 + CL_083
```

---

# 15. Factor Library 设计

不要只做一个“低相关库”。

建议支持多个 logical library：

```text
CORE_LOW_REDUNDANCY
PRICE_VOLUME_CORE
FUNDAMENTAL_CORE
EVENT_ALPHA
LOW_TURNOVER
HIGH_CAPACITY
REGIME_BULL
REGIME_BEAR
TREE_MODEL_FEATURES
NEURAL_MODEL_FEATURES
RESEARCH_SHADOW
```

LibraryDefinition：

```text
library_id
purpose
eligibility policy
selection policy
cluster diversity policy
capacity policy
max factors
min evidence
promotion cadence
```

LibraryVersion：

```text
library_version_id
logical_library_id
cluster_version_id
members
policy_hash
evidence_snapshot
created_at
status
content_hash
```

member 不只是 factor_id，至少：

```text
factor_definition_id
selected treatment_id
orientation
cluster_id
representative_of
similarity_ref
health_state_ref
assembly_score
selection_rank
```

---

# 16. Library 增量更新与 Promotion

新因子通过 admission 后，不要立即污染 production library。

流程：

```text
eligible factor
→ candidate library overlay
→ shadow evaluation
→ promotion window
→ create immutable new LibraryVersion
→ FeatureSet diff
```

Promotion Policy 可配置：

```text
weekly schedule
minimum new qualified factors
minimum utility improvement
maximum allowed turnover of library members
cluster coverage target
minimum robustness
health gates
```

生产 library 永远 immutable。

更新只能：

```text
v102 PRODUCTION
v103 CANDIDATE
v103 SHADOW
v103 APPROVED
v103 PRODUCTION
```

并保留 rollback 到 v102。

---

# 17. FeatureSet 设计

FactorLibrary 不是直接给模型的最终 X schema。

新增：

```text
FeatureSetArtifact
```

字段：

```text
feature_set_id
feature_set_version
source_library_versions
ordered_feature_manifest
feature name
factor_definition_ref
treatment_ref
orientation
channel/schema
timing semantics
snapshot compatibility
content_hash
```

一定要保持 **ordered feature identity**。

因为：

```text
同一批 factor，列顺序不同
```

对于部分模型和序列化接口都可能产生实质问题。

---

# 18. 模型重训触发规则

不要每新增一个因子重训。

先实现：

```text
FeatureSetDiff
```

分类：

```text
METADATA_ONLY
EVIDENCE_ONLY
FEATURE_MEMBERSHIP_CHANGE
FEATURE_TRANSFORM_CHANGE
FEATURE_ORIENTATION_CHANGE
FEATURE_SCHEMA_CHANGE
LABEL_CHANGE
DATA_REVISION
```

默认：

```text
METADATA_ONLY -> 不重训
EVIDENCE_ONLY -> 不重训

FEATURE_* -> MODEL_RETRAIN_REQUIRED
LABEL_CHANGE -> MODEL_RETRAIN_REQUIRED
重大 DATA_REVISION -> MODEL_RETRAIN_REQUIRED
```

模型层收到的是事件，而不是平台直接 import 模型训练代码。

---

# 19. PostgreSQL Metadata Schema

优先 PostgreSQL，SQLite 只允许 local test/dev adapter，不作为生产真源。

至少建立：

```text
users
roles
permissions
user_roles

audit_logs

artifacts
artifact_lineage

factor_candidates
factors
factor_definitions
factor_generator_lineage

factor_value_versions

evaluation_runs
evaluation_summaries

treatment_runs
treatment_selections

similarity_graph_versions
similarity_edges_or_partitions

cluster_versions
clusters
cluster_memberships
cluster_lineage

factor_libraries
factor_library_versions
factor_library_members

feature_sets
feature_set_members

jobs
job_attempts
workflow_runs

inbox_events
outbox_events

data_snapshots
universes

report_exports
```

## 19.1 数据库不存大矩阵

不存：

```text
全部 factor daily values
全部 similarity dense matrix
全部 backtest curve
全部 quantile time series
```

这些存 COS。

DB 只存：

```text
summary + ArtifactRef
```

## 19.2 索引必须提前设计

典型查询索引：

```text
factors(status, created_at)
factor_definitions(content_hash)
evaluation_summaries(factor_id, evaluation_profile, created_at)
cluster_memberships(cluster_version_id, factor_id)
factor_library_members(library_version_id, factor_id)
jobs(status, created_at)
artifacts(content_hash)
audit_logs(user_id, created_at)
```

Factor Catalog 搜索需要：

```text
factor id/name/family/tags/status/grade/health/cluster/library/generator/campaign
```

不要页面每次 join 20 张表；为 catalog 建 read model / materialized view。

---

# 20. CQRS 风格的 Read Model

平台页面查询和 pipeline 写操作访问模式不同。

建议建立轻量 read model：

```text
factor_catalog_view
cluster_catalog_view
library_catalog_view
job_dashboard_view
```

例如 `factor_catalog_view` 直接包含：

```text
factor_id
name
family
lifecycle
health
grade
rank_ic
icir
turnover
selected_treatment
cluster_id
production_library_count
created_at
updated_at
```

这样 10 万因子分页/筛选不会每次动态拼超复杂 SQL。

read model 通过 event 或事务后刷新。

---

# 21. COS 数据湖布局

不要产生“每个 factor × 每个 date 一个小文件”。

推荐逻辑布局：

```text
/canonical/
  factor_definitions/
  factor_values/
  evaluations/
  treatments/
  similarity/
  clusters/
  libraries/
  feature_sets/
  models/
  backtests/
  exports/
```

Factor values 根据访问模式做分区。

建议同时支持：

## Canonical factor-major

适合单因子评估/详情：

```text
factor_values/
  market=ashare/
  freq=1d/
  factor_bucket=0012/
  factor_id=F_xxx/
  snapshot=...
  part-....parquet
```

## Derived feature-block

适合模型一次读几百/几千因子：

```text
feature_blocks/
  feature_set=FS_102/
  date_bucket=2026Q3/
  block=0001.parquet
```

FeatureBlock 是 derived cache，可重建，不是 identity 真源。

---

# 22. 本地磁盘策略

1.6TB 本地盘不能作为长期真源。

实现统一：

```text
ObjectStore
LocalArtifactCache
```

LocalArtifactCache：

```text
max bytes
LRU eviction
checksum validation
partial download temp suffix
file lock
cache metrics
```

worker 流程：

```text
ArtifactRef
→ cache lookup by content_hash
→ if miss: download
→ verify hash
→ use
```

执行输出：

```text
local temp
→ fsync/close
→ hash
→ COS upload
→ verify
→ DB register
```

本地 cache 随时可以删除并重建。

---

# 23. API 设计

使用 FastAPI 合理，但 route 只做：

```text
auth
permission
request validation
query/command dispatch
response serialization
```

禁止 route 内运行重计算。

主要 API：

```text
POST   /api/v1/auth/login
POST   /api/v1/auth/logout
GET    /api/v1/me

GET    /api/v1/dashboard

GET    /api/v1/factors
GET    /api/v1/factors/{factor_id}
GET    /api/v1/factors/{factor_id}/evaluations
GET    /api/v1/factors/{factor_id}/lineage
GET    /api/v1/factors/{factor_id}/versions
GET    /api/v1/factors/{factor_id}/cluster
GET    /api/v1/factors/{factor_id}/libraries
GET    /api/v1/factors/{factor_id}/formula
POST   /api/v1/factors/{factor_id}/reprocess

GET    /api/v1/clusters
GET    /api/v1/clusters/{logical_cluster_id}
GET    /api/v1/cluster-versions

GET    /api/v1/libraries
GET    /api/v1/libraries/{library_id}
GET    /api/v1/libraries/{library_id}/versions
POST   /api/v1/libraries/{library_id}/promote
POST   /api/v1/libraries/{library_id}/rollback

GET    /api/v1/feature-sets
GET    /api/v1/feature-sets/{id}
GET    /api/v1/feature-sets/{id}/diff/{other_id}

GET    /api/v1/jobs
GET    /api/v1/jobs/{job_id}
POST   /api/v1/jobs/{job_id}/retry
POST   /api/v1/jobs/{job_id}/cancel

GET    /api/v1/artifacts/{id}
POST   /api/v1/artifacts/{id}/download-token

GET    /api/v1/standards/metrics
GET    /api/v1/standards/tags
GET    /api/v1/standards/charts
GET    /api/v1/standards/policies

POST   /api/v1/exports
GET    /api/v1/exports/{id}
```

列表 API 必须支持：

```text
cursor pagination
filter
sort
search
field projection
```

10 万因子不要 offset 到第 90000 行，优先 keyset/cursor pagination。

---

# 24. 权限与因子 IP 安全

这是量化私募平台的 P0，不是后补功能。

## 24.1 RBAC + Resource Permission

角色建议：

```text
VIEWER
RESEARCHER
CORE_RESEARCHER
OPS
ADMIN
```

权限不要只按页面。

要按能力：

```text
factor:read_summary
factor:read_evidence
factor:read_formula
factor:read_values
factor:download_values
factor:reprocess
cluster:read
library:promote
job:retry
user:manage
standards:edit
```

## 24.2 Formula 不能前端隐藏

无 `factor:read_formula` 时：

```text
GET /factors/F001/formula -> 403
```

Factor summary API 本身也不能包含 formula 字段。

## 24.3 Raw Factor Values

不允许浏览器直接获得永久 COS URL。

流程：

```text
permission check
→ audit
→ short-lived signed URL / streamed download
```

设置：

```text
short expiry
content disposition
optional IP restriction
rate limit
```

## 24.4 Audit Log

以下操作强制审计：

```text
查看公式
下载 factor values
下载 FeatureSet
library promotion
rollback
retry production job
user/permission change
standards policy change
```

记录：

```text
user
resource
action
timestamp
request id
source ip
result
reason
```

## 24.5 Authentication

第一阶段可：

```text
用户名密码
+
Argon2id password hash
+
HTTP-only secure cookie/session 或短时 access token
```

架构上预留 OIDC/Keycloak。

生产必须 HTTPS。

---

# 25. Web UI 信息架构

原报告中心的“日报/周报”不再是一级导航。

一级导航推荐：

```text
Overview

Research
  Factors
  Campaigns
  Treatments
  Clusters
  Libraries

Models
  Feature Sets
  Models          # 可先只读/占位接正式 registry

Portfolio
  Backtests       # contract + future adapter
  Optimizers

Production
  Live Health
  Streaming
  Alerts

Platform
  Jobs
  Data Snapshots
  Artifacts
  Standards
  Exports
  Audit
  Users & Roles
```

---

# 26. Dashboard

首页必须是“系统运营态势”，不是报告目录。

KPI：

```text
Total Factors
New Today / 7D
Processing
Validated
Shadow
Production Eligible
Production
Degraded
Quarantined
```

Pipeline funnel：

```text
Generated
→ Valid
→ Compiled
→ Materialized
→ QE Passed
→ Novel
→ Clustered
→ Library Candidate
→ Promoted
```

再展示：

```text
job backlog
failure rate
factor throughput
COS ingest lag
cluster drift
library version changes
recent promotions
health alerts
```

---

# 27. Factor Catalog

`/factors` 必须是平台最重要页面之一。

列建议：

```text
Factor ID
Name
Family
Generator
Campaign
Lifecycle
Health
Grade
RankIC
ICIR
Coverage
Turnover
Selected Treatment
Cluster
Libraries
Created
Last Evaluated
```

Filter：

```text
时间
family
generator
campaign
lifecycle
health
grade
RankIC range
ICIR range
cluster
library
treatment
market
frequency
```

支持 saved views，例如：

```text
今日新增有效因子
S/S+ 且低相关
待 cluster 因子
Health degraded
LLM campaign 2026-08-26
```

---

# 28. Factor Detail

Tabs：

```text
Overview
Performance
IC & Decay
Quantiles
Robustness
Exposure
Tradability
Treatment
Similarity
Cluster
Libraries
Lineage
Versions
Artifacts
```

有公式权限的人额外：

```text
Formula
Raw Values
Generator Detail
```

页面数据全部来自已有 artifact / registry。

禁止 React/FastAPI 在页面请求时重新计算指标。

---

# 29. Clusters 页面

必须能查看：

```text
current cluster version
logical cluster count
singleton count
bridge count
cluster size distribution
cross-cluster similarity
cluster drift
```

单 cluster 页面：

```text
members
representatives
quality distribution
similarity graph
internal correlation
external nearest clusters
cluster lineage
library usage
```

需要明确展示：

```text
logical_cluster_id
algorithm label
cluster_version
```

---

# 30. Libraries 页面

必须展示：

```text
logical library
production version
candidate version
member count
cluster coverage
redundancy
expected utility
turnover/capacity profile
health
```

支持版本 diff：

```text
v102 → v103
+ factor
- factor
changed treatment
changed orientation
cluster coverage change
```

Promotion 必须有权限、审计和 rollback。

---

# 31. Jobs / Pipeline Operations 页面

因子上传后立即显示，不等全部结束。

Factor detail 可看到：

```text
Current Stage
Progress
Start Time
Attempts
Worker
Input Artifacts
Output Artifacts
Error Code
```

Jobs 页面：

```text
queued/running/failed
filter by stage
filter by campaign
filter by factor
retry
cancel
```

进度推送可用：

```text
SSE 或 WebSocket
```

若只是单向 job update，优先 SSE，简单稳定。

---

# 32. 原 HTML 报告中心如何迁移

不要删除价值，也不要继续让它当主系统。

迁移：

```text
现有 metric registry
→ Platform Standards

现有 tag registry
→ Platform Standards

现有 chart registry
→ Platform Standards

现有 scorecard
→ Platform Standards / Policies

现有 HTML reports
→ Export Templates / Legacy Reports
```

原来的：

```text
FOP-DLY
FOP-WKL
FAR-*
SML-*
PFO-*
```

以后由平台按 saved query + EvaluationBundle 生成。

例如：

```text
用户选择过去7天 + factor production
→ Export Weekly Factor Report
→ HTML/PDF artifact
```

必须继续遵守：

```text
Report Builder only renders artifacts
Never recompute metric in report layer
```

EXP/NEXP 两目录最终废弃为逻辑权限，不再复制两套页面。

---

# 33. QuantEvaluator 改造要求

先审计已有内容，不重复实现。

目标：

1. EvaluationArtifact / EvaluationBundle 成为唯一事实；
2. metric registry 只有一个 authority；
3. typed dependency，不使用模糊 magic string；
4. reporting 只读 artifact；
5. summary 支持 platform read model；
6. large series/matrix 使用 ArtifactRef；
7. provenance immutable；
8. content hash 包含数据语义。

平台至少需要 QE 提供：

```text
factor evaluation summary
metric artifacts
chart specs/data refs
grade
health evidence
data quality
rolling health updates
```

如果目前存在两个 MetricSpec/catalog authority，统一为一个 canonical registry，并提供 migration compatibility layer。

RankIC / PearsonIC 依赖链必须语义分离，不能 generic `ic_std` 在不同 IC 类型间串错。

---

# 34. FactorPreprocess 改造要求

重点：

```text
TreatmentRecipe
FittedState
FeatureBundle
FeatureManifest
Causality certification
```

所有生产 transform 必须：

```text
causal
or train-fitted then apply
```

property tests：

```text
Prefix Invariance
Asset Isolation
Fit/Apply Boundary
```

生产禁用或 fail closed：

```text
full-sample HP
full-sample STL
filtfilt
full-sample interpolation
future-aware imputation
```

如果目前存在两套 preprocessing policy schema，要统一 authority，旧 schema 只兼容。

每个处理步骤使用唯一 `step_id`，不能只靠 transform 名称。

---

# 35. FactorOptimizer 改造要求

FO 在这里主要负责 treatment/search orchestration，而不是控制 Web。

必须支持：

```text
sealed test
no test object in search callback
purge + embargo
trial ledger
multi-fidelity
failed trial logging
multiple testing
```

Factor treatment search 不必机械复制 model rolling WF；但时间边界必须正确。

生产 callback 不允许获得 test 数据对象。

---

# 36. FactorAssets 改造要求

FactorAssets 是本次平台化最需要增强的 domain 包。

它继续拥有：

```text
Factor Identity
Asset Registry
Lifecycle
Evidence References
Novelty
Admission
Similarity
Clustering
Assembly
Library governance
```

新增/正式化 contract：

```text
ResultIdentity
SimilarityFingerprintArtifact
SimilarityGraphArtifact
IncrementalClusterAssignment
ClusterVersionArtifact
ClusterLineageArtifact
FactorLibraryDefinition
FactorLibraryVersionArtifact
FactorLibraryMembership
FeatureSetCandidate
PromotionDecision
```

特别注意：

```text
FactorSet / FactorSetArtifact
```

只保留一个 canonical production artifact 语义；旧类型兼容但不得形成双真源。

Assembly 不能只是 family round-robin + recency。

生产 assembly 应考虑：

```text
quality
stability
robustness
novelty
tradability
cluster diversity
health
capacity
```

---

# 37. FactorEngine 改造边界

不要为了平台重写 FE。

只补平台需要的 adapter contract：

```text
validate definition
compile
resolve required fields
materialize
incremental materialize
batch materialize
return Artifact metadata
```

FE 仍可独立运行。

平台 worker 负责：

```text
Platform request
→ FE native API
→ result
→ FactorValueArtifact publish
```

不要让 FE 依赖 PostgreSQL platform schema。

---

# 38. DataAccess 改造边界

DataAccess 保持：

```text
canonical PIT data semantics
calendar
snapshot
universe
field registry
```

平台需要标准 adapter 获取：

```text
DataSnapshotRef
UniverseRef
FieldAvailability
DataFreshness
```

严禁平台自己重新实现一套 PIT/asof 逻辑。

---

# 39. A 股必须纳入统一 timing contract

平台中所有 evaluation/materialization/feature set 都统一字段：

```text
decision_time
signal_available_time
first_executable_time
label_start_time
label_end_time
```

A 股 production evidence 至少考虑：

```text
ST / *ST
停牌
不同板块/时期涨跌停制度
IPO / delist
复权与公司行动
买入可交易 / 卖出可交易区别
T+1
收盘后信号时间
公告发布日期 / 修订日期
PIT industry/universe
```

这些语义必须来自 DataAccess / execution contract，不允许页面或 QE 自己随意猜。

---

# 40. Streaming 与增量更新

系统要区分两个时钟：

```text
Signal available clock
Label matured clock
```

T 日信号完成可以马上更新：

```text
coverage
missing
exposure
rank retention
runtime
health
```

标签未成熟时：

```text
LABEL_NOT_MATURE
```

不能填 0。

等 T+n label 成熟，再增量更新：

```text
daily RankIC
quantile outcome
rolling IC
health
```

缺失 evidence 状态必须区分：

```text
NOT_COMPUTED
UNAVAILABLE
INVALID_EVIDENCE
LABEL_NOT_MATURE
```

绝不 zero-fill。

---

# 41. Backtest / Simulation 的接口预留

本阶段不把完整回测引擎合并进 QE。

定义 Protocol：

```text
BacktestProvider
```

输入：

```text
SignalArtifact
BacktestRequest
ExecutionPolicy
```

输出：

```text
BacktestArtifactRef
```

QE 再评估 BacktestArtifact。

严格区分：

```text
qe.probe.*
```

和：

```text
bt.*
```

不要把 cheap probe 当正式 execution backtest。

---

# 42. 幂等性设计

所有 heavy stage 必须计算 idempotency key。

例如 materialization：

```text
hash(
  factor_definition_id,
  data_snapshot_id,
  universe_id,
  calculation_spec_id
)
```

QE：

```text
hash(
  factor_value_id,
  evaluation_policy_id,
  label_id,
  profile_id
)
```

Treatment：

```text
hash(
  source_evidence,
  search_policy,
  split_plan
)
```

如果已存在合法 artifact：

```text
CACHE_HIT
```

而不是再算一次。

---

# 43. Retry 与错误分类

所有 worker error 分类：

```text
RetryableInfrastructureError
RetryableStorageError
RetryableDatabaseError
DataUnavailableError
InvalidInputError
SemanticContractError
CapabilityError
NumericalFailure
ResourceExceededError
CancellationError
```

Retryable：指数退避 + jitter。

Semantic/invalid：不重试，直接 terminal。

必须设置 retry budget，不能无限重试。

---

# 44. 资源调度

每个 job 声明：

```text
cpu
memory
io class
estimated factor count
estimated row count
priority
```

建议 queue/resource class：

```text
light
io_heavy
cpu_heavy
memory_heavy
long_running
```

因子 materialization 应优先按相同 DataSnapshot / field scope batching，减少重复 IO。

10 万因子规模下，优化 IO 往往比单个算子微优化更重要。

---

# 45. 观测与 SLO

至少接：

```text
OpenTelemetry
Prometheus
Grafana
structured logs
```

核心 metric：

```text
candidate_ingest_total
candidate_ingest_lag
factor_materialize_rate
factor_materialize_latency
qe_rate
qe_failure_rate
treatment_search_latency
cluster_assignment_latency
workflow_backlog
workflow_failure_rate
cos_read_bytes
cos_write_bytes
local_cache_hit_rate
postgres_query_latency
api_p95
artifact_corruption_count
```

建议初始 SLO：

```text
API read p95 < 500ms（不含 artifact download）
Factor catalog first page < 1s
Object event eventual ingestion < 5min
No silent workflow loss
No unauthorized formula response
Artifact checksum mismatch = critical
```

---

# 46. 安全设计

除 RBAC 外：

- Secrets 使用 environment/secret manager，不能 commit；
- DB 最小权限账号；
- COS worker 与 browser 权限分开；
- Signed URL 短时；
- CORS 明确白名单；
- CSRF 防护（若 cookie auth）；
- API rate limit；
- SQL injection 依赖 ORM/parameterized query；
- artifact path 不能允许 `../`；
- 上传 manifest 限制大小；
- HTML export 防 XSS；
- formula display 做 escaping；
- audit log append-only；
- admin action 二次确认；
- production promotion 可选双人审批接口。

---

# 47. 数据库迁移

使用 Alembic 或当前项目已有等价 migration 工具。

原则：

```text
expand
→ backfill
→ switch reads/writes
→ contract
```

不要直接 destructive migration。

尤其现有 SQLite registry/JSON registry 迁 PostgreSQL 时：

```text
保留 old ID
保留 content hash
记录 migrated_from
校验 count/hash
```

迁移前后生成 reconciliation report。

---

# 48. 报告标准 Registry 的真源统一

现在静态目录里的：

```text
standards-policy.json
metric-registry.json
tag-registry.json
chart-registry.json
```

不要同时在 DB、Python、JSON 维护三套人工真源。

选择一个 canonical source。

推荐：

```text
version-controlled machine-readable registry files
→ startup/CI validate
→ sync read model into DB
```

平台 UI 是 viewer/editor（如果以后允许）。

任何 policy 修改：

```text
policy version bump
content hash
change log
audit
```

---

# 49. 前端技术要求

建议：

```text
React
TypeScript
Vite
TanStack Query
TanStack Table
React Router
ECharts/Plotly（按现有 chart stack 统一）
```

避免在前端写领域逻辑。

TypeScript 类型应从 OpenAPI 自动生成或至少 CI 校验。

关键页面必须：

```text
loading
empty
permission denied
partial evidence
error
stale data
```

状态完整。

不要把“没有数据”渲染成 0。

---

# 50. Platform API 技术要求

建议：

```text
FastAPI
Pydantic v2
SQLAlchemy 2.x
Alembic
PostgreSQL
```

所有 endpoint：

```text
request_id
structured logging
permission check
schema validation
consistent error model
```

统一错误响应：

```json
{
  "error": {
    "code": "FACTOR_NOT_FOUND",
    "message": "...",
    "request_id": "...",
    "retryable": false
  }
}
```

---

# 51. 部署策略

第一阶段不强制 Kubernetes。

推荐：

```text
Docker Compose
```

服务：

```text
platform-api
platform-web
postgres
temporal-server（若自建）
temporal-worker
nginx/traefik
prometheus
grafana
```

量化 heavy worker 可以运行在独立服务器，不一定全部放一个 compose。

通过 Temporal/task queue 连接。

后续机器增加再迁 K8s。

---

# 52. CI / Release Gate

每个 package 保持独立测试。

平台 CI 至少：

```text
ruff/formatter
mypy/pyright（按现有标准）
pytest
frontend lint
tsc
frontend unit test
build frontend
migration test
API OpenAPI compatibility
security dependency scan
```

更重要的是 clean install：

```text
build wheel
fresh venv outside repo
pip install wheel
recursive import
contract tests
```

防止 monorepo PYTHONPATH 掩盖 packaging 错误。

---

# 53. Integration Test Matrix

至少实现：

## 53.1 Happy path

```text
fake candidate
→ ingest
→ FE adapter fake/materialize small fixture
→ QE
→ treatment
→ FA
→ cluster
→ library candidate
→ FeatureSet
```

## 53.2 Duplicate

同一 definition 上传两次：

```text
不能重算整个 pipeline
必须关联同 definition
```

## 53.3 Event lost

人为不发送 event：

```text
reconciliation scanner 最终发现
```

## 53.4 Worker crash

materialization 中断：

```text
retry
无重复 artifact
```

## 53.5 DB commit / event failure

验证 outbox 能恢复。

## 53.6 COS corrupt

hash mismatch：

```text
artifact 不注册 READY
critical alert
```

## 53.7 Permission

Viewer：

```text
summary 200
formula 403
values download 403
```

CoreResearcher：按授权成功。

## 53.8 Version rollback

library v103 promotion 后 rollback v102：

```text
历史 v103 不删除
active pointer 回 v102
```

## 53.9 Cluster split/merge

验证 logical cluster lineage。

## 53.10 Missing evidence

前端显示：

```text
UNAVAILABLE
```

而不是 0。

---

# 54. 性能与规模测试

生成 synthetic metadata：

```text
100k factors
500 microclusters
50 macroclusters
1m+ artifact refs
大量 historical jobs
```

验证：

```text
factor catalog query
filter
search
cluster members
library diff
job dashboard
```

不要真的生成 100k × 全历史 factor values 做 CI。

大数据 benchmark 独立 nightly/offline。

---

# 55. 常见失败模式与预防

| 风险 | 后果 | 必须采取的设计 |
|---|---|---|
| Web route 直接跑 FE/QE | API 卡死 | command → workflow → worker |
| COS 当 metadata DB | 全桶扫描、慢 | PostgreSQL registry |
| 每个日期一个 factor 文件 | object explosion | bucket/partition |
| 每个新因子全量 recluster | 算力爆炸 | incremental + periodic global |
| Leiden label 当永久 ID | cluster identity 漂移 | logical cluster + lineage |
| Cluster=Library | 概念混乱 | cluster 与 library 分离 |
| Library 原地改 | 模型无法复现 | immutable version |
| 新因子立即触发 retrain | 模型频繁抖动 | promotion cadence + FeatureSet diff |
| Formula 前端 hide | IP 泄露 | backend field-level RBAC |
| EXP/NEXP 两套数据 | 双真源 | one snapshot + permission |
| 报告重算 metric | 数字不一致 | render artifact only |
| Event/DB 双写 | workflow 丢失 | transactional outbox |
| Retry 不幂等 | 重复 artifact | content identity + idempotency |
| 缺失 metric 填 0 | 错误决策 | typed evidence status |
| Test 进入 search object graph | 泄漏 | physical sealed test |
| Full-sample preprocess | 未来函数 | causality certification |
| PostgreSQL 存大矩阵 | DB 崩 | COS artifact |
| Browser 拿 COS key | 安全事故 | signed URL/server proxy |
| 本地磁盘做真源 | 空间耗尽 | COS authority + local cache |
| 巨型 common package | 强耦合 | domain owners + adapters |

---

# 56. 实施阶段

不要把下面 Phase 并行乱改同一核心 contract。

## Phase 0 — Repository Audit & Contract Freeze

必须产出：

```text
CURRENT_ARCHITECTURE.md
GAP_ANALYSIS.md
PLATFORM_CONTRACTS.md
```

确认：

- 哪些 artifact 已存在；
- 哪些 registry 已存在；
- QE reporting 当前结构；
- FA clustering/assembly 当前能力；
- FP/FO contract；
- FE/DA adapter 能力；
- 现有 HTML machine registries。

然后冻结 v1 contracts。

## Phase 1 — Platform Skeleton

实现：

```text
platform backend
platform_web
PostgreSQL
Alembic
Auth
RBAC
Factor Catalog read model
basic Dashboard
```

暂时可以导入 fixture metadata。

验收：浏览器登录后看到真实 Registry Factor 列表。

## Phase 2 — Artifact Registry + COS

实现：

```text
ObjectStore protocol
COS adapter
LocalArtifactCache
Artifact registry
checksums
atomic publish
signed access
```

验收：factor value/evaluation artifact 可登记/查询/授权下载。

## Phase 3 — Candidate Ingestion

实现：

```text
manifest protocol
_READY
object event adapter
reconciliation scanner
candidate registry
```

验收：COS 新 candidate 自动出现在 UI。

## Phase 4 — Workflow Automation

接：

```text
FE
DA
QE
FP
FO
FA
```

实现 job/status/progress/retry。

验收：小 fixture 因子全自动跑完。

## Phase 5 — Clustering & Library Governance

实现：

```text
incremental assignment
cluster versions
logical cluster lineage
library versions
promotion/rollback
```

验收：新 factor 自动进入 provisional cluster，周期 global refresh 可生成新版本且 lineage 正确。

## Phase 6 — FeatureSet / Model Boundary

实现：

```text
FeatureSetArtifact
FeatureSetDiff
ModelRetrainRequired
```

不强制实现完整模型训练。

## Phase 7 — Report Center Migration

迁移：

```text
Standards
Saved Views
Exports
legacy reports
```

停止维护 EXP/NEXP duplicated pages。

## Phase 8 — Production Hardening

完成：

```text
observability
backup
restore drill
security hardening
load test
failure injection
runbook
```

---

# 57. 多 Agent 开发分工建议

可以并行，但 contract 先冻结。

建议：

```text
Agent A: Platform contracts + DB schema
Agent B: FastAPI/RBAC/API
Agent C: React platform
Agent D: ObjectStore/COS/cache/artifact registry
Agent E: Workflow/Temporal/event/outbox
Agent F: QE integration/reporting migration
Agent G: FA clustering/library/versioning
Agent H: FP/FO treatment integration
Agent I: FE/DA adapter
Agent J: Integration/chaos/security tests
```

规则：

- A 先定 contract；
- 多 Agent 不同时改同一个 contract 文件；
- 每个 agent 提交前跑自身 test；
- 最后由独立 reviewer 做 cross-package contract review；
- reviewer 不能只看 markdown completion report，必须看代码和测试结果。

---

# 58. 必须生成的工程文档

最终至少：

```text
platform/README.md
platform/docs/ARCHITECTURE.md
platform/docs/CONTRACTS.md
platform/docs/STATE_MACHINES.md
platform/docs/EVENTS.md
platform/docs/STORAGE_LAYOUT.md
platform/docs/RBAC.md
platform/docs/OPERATIONS.md
platform/docs/FAILURE_RECOVERY.md
platform/docs/MIGRATION_FROM_REPORT_CENTER.md
platform/docs/DEPLOYMENT.md
platform/docs/API.md
```

不要生成几十份重复“完成报告”污染根目录。

最终 status 只保留少量 machine-verifiable evidence。

---

# 59. Definition of Done

这次工程不能以“页面能打开”作为完成。

P0 Done 必须同时满足：

1. 用户通过浏览器 URL 登录；
2. 不同账号权限不同；
3. 无权限用户 API 层拿不到 formula/value；
4. 10 万量级 metadata 下 Factor Catalog 可分页筛选；
5. 新 FactorCandidate 上传 COS 后可被自动发现；
6. 新 factor 立即进入平台并显示 processing stage；
7. FE/DA/QE/FP/FO/FA 通过 worker adapter 串联；
8. heavy task 不在 API process 执行；
9. Artifact 全部可追踪 lineage/content hash；
10. 大数据 COS 存储，本地仅 cache；
11. workflow idempotent；
12. event 丢失可 reconciliation；
13. DB/event 使用 outbox 保证最终一致；
14. Factor cluster 支持 incremental assignment；
15. global clustering 有 version 与 logical lineage；
16. Cluster 和 FactorLibrary 明确分离；
17. Library immutable version + promotion + rollback；
18. FeatureSet immutable version；
19. FeatureSet semantic diff 可发 ModelRetrainRequired；
20. 原 HTML 报告变成 Standards/Export/Legacy，而不是系统真源；
21. Report Builder 不重新计算指标；
22. 缺失 evidence 不填 0；
23. 所有核心 action 有 audit log；
24. 集成测试覆盖正常、重复、失败、重试、权限、回滚；
25. fresh wheel/install test 通过；
26. 有部署、恢复、备份、故障 runbook。

---

# 60. AI 最终执行要求

AI 看到本文后不要仅仅回答“方案很好”或再写一份概念设计。

执行流程必须是：

```text
1. 审计当前本地代码
2. 对照本文建立 gap list
3. 锁定 canonical contracts
4. 分阶段实现
5. 每阶段运行测试
6. 修复 regression
7. 完成数据迁移/兼容
8. 完成 UI
9. 完成端到端自动化
10. 独立 review
11. 生成 verification evidence
```

如果已有实现与本文同义：

```text
复用 / 加固 / 迁移
```

不要复制一份新实现形成第二真源。

如果现有接口较差但已有生产调用：

```text
新增 canonical API
+
兼容 adapter
+
逐步迁移旧调用
```

不要暴力删旧接口导致全仓 regression。

任何“Production Ready”结论必须有：

```text
exact local git SHA
working tree status
test commands
test outputs
migration status
known limitations
```

Markdown 自述不能作为唯一证据。

---

# 61. 最终架构一句话

最终系统应该形成：

```text
Factor Generation
→ Candidate Lake
→ Data/Factor Materialization
→ Evaluation
→ Treatment Search
→ Asset Admission
→ Similarity Graph
→ Incremental/Global Clustering
→ Versioned Factor Libraries
→ Versioned FeatureSets
→ Model Retrain Event
```

同时：

```text
Web Platform = 可视化 + 权限 + 查询 + 管理 + 触发 + 审计
Control Plane = 自动化编排 + 状态机 + 重试 + 版本治理
Domain Packages = 真正的量化计算与语义
PostgreSQL = Metadata Truth
COS = Large Artifact Truth
Local Disk = Cache
```

不要让任何一层越权成为另一层的第二真源。

这条原则比某个具体框架、某张页面、某个数据库表更重要。
