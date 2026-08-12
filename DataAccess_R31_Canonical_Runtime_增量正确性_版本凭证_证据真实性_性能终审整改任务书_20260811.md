# DataAccess R31 — Canonical Runtime 收敛、版本自动化、增量正确性、云凭证、证据真实性与性能终审整改任务书

> 仓库：`https://github.com/18047533889/quant_projects`  
> 审计基线：`main@5f44df633ee4d1763a5d2d1cdeb55eb579f1e4e5`  
> 基线提交：`Sync R30 DA maturity/read-perf, FE R39/materialize work, and R39/R30/operator-dev prompts.`  
> 日期：2026-08-11  
> 前序：R24–R30 DataAccess；R37–R39 FactorEngine 资源、批量读取、物化与执行整改  
> 本轮性质：**R30 之后的二阶审计、canonical runtime 收敛与新问题修复。**

---

## 0. 给 coding agent 的最高级指令

本文件不是让你再发明一套 DataAccess。

执行前必须先读取**执行时最新 main**，逐项核验本文件中的问题是否仍存在：

```text
CURRENT HEAD VERIFY
        ↓
仍存在            → FIX
后续提交已真正修复 → ALREADY CLOSED
已有更成熟 canonical 实现 → MIGRATE / DELETE DUPLICATE
无法证明           → 不得宣称 CLOSED
```

### 硬约束

1. 禁止为了完成 checklist 再新建第三套同义 abstraction。
2. 禁止只改测试、不改真正 public runtime path。
3. 禁止为了性能关闭 PIT、snapshot、security、schema/semantic gate。
4. 禁止 FactorEngine production 路径重新裸读 parquet/COS。
5. 禁止 DataAccess 反向承担 FactorEngine operator/alpha 逻辑。
6. A股与美股必须继续保留真实：
   - 单位；
   - currency；
   - calendar；
   - filing/PubDate；
   - financial flow semantics；
   - revision fidelity；
   - instrument identity。
7. `except Exception -> pass/None/PASS` 在安全、PIT、身份、变更传播、DQ、durable lineage 等边界默认禁止。
8. 性能证据必须来自**真实 runtime counter**，不得把 planner 的估算字段命名成 actual physical scan count。
9. 当前 FE 已有 `HostResourceCoordinator / ResourceBroker / ReadWavePlanner / SourceBinding / PhysicalFactorDAG` 等能力，DataAccess 必须与它们收敛，不建立互相竞争的第二套主权威。
10. 本轮最终目标：**一个数据事实只定义一次，一个请求只走一条治理链，一批因子共享的数据只物理准备一次。**

---

## 1. 当前总诊断

R30 已经新增大量高级能力：

```text
dataaccess/r30/
    adapter.py
    cache_hierarchy.py
    calendar_snapshot.py
    change_impact.py
    concepts.py
    coverage_service.py
    data_change.py
    data_quality.py
    execution_lease.py
    experiment_snapshot.py
    lineage.py
    metrics.py
    partition_index.py
    policy.py
    query_trace.py
    session.py
    ...
```

FactorEngine 同时已有：

```text
factor_source_plan.py
factor_batch_plan.py
field_request_coalescer.py
batch_data_request.py
read_wave_planner.py
source_binding.py
projected_column_footprint.py
physical_factor_dag.py
...
```

当前最大风险已经从“缺能力”变成：

```text
Canonical DataAccess Core
        +
R30 Additive Layer
        +
FE R30 Planner
        +
FE R39 Canonical Planner
```

部分语义重复、部分只是 facade、部分证据只证明 IR 而非真实执行。

### R31 最终必须只剩一套权威

```text
Canonical Identity Encoder
Canonical DataReadSession
Canonical QueryTrace/Telemetry
Canonical Metadata Plane
Canonical Coverage Provider
Canonical Resource Lease
Canonical Semantic/Concept/Unit
Canonical Factor Source Binding
Canonical Batch Read Planner
Canonical Change Impact Contract
Canonical Snapshot / Execution Identity
```

---

## 2. 当前 CI / Evidence 状态

审计基线：

```text
5f44df633ee4d1763a5d2d1cdeb55eb579f1e4e5
```

当前 GitHub connector 返回：

```text
combined status = []
workflow runs = []
```

所以基线状态只能写：

```text
CURRENT-HEAD CI EVIDENCE = NOT PROVEN
```

最终报告禁止把：
```text
本地 pytest
旧 SHA benchmark
planner estimate
```
写成：
```text
current final SHA production evidence
```

---

# 3. P0 总表

| ID | 问题 | 类别 |
|---|---|---|
| R31-P0-001 | `data_access.r30` 未进入 wheel | Packaging |
| R31-P0-002 | R30 additive layer 未收敛 canonical runtime | Architecture |
| R31-P0-003 | base package version 仍需人工改 | Versioning |
| R31-P0-004 | `stable_digest` 把 tuple/list 顺序抹掉 | Identity |
| R31-P0-005 | SourceSnapshot 混入 build SHA 且 rebuild identity 算法不一致 | Reproducibility |
| R31-P0-006 | build SHA 依赖 runtime git | Release |
| R31-P0-007 | SourceBlockKey 缺 projection/instrument/params 等身份 | Cache correctness |
| R31-P0-008 | SourceBlock refcount 无 release | Lifecycle |
| R31-P0-009 | `DataReadSession` 与 `R30ReadSession` 双轨 | Architecture |
| R31-P0-010 | CostModel 三态未真实执行 | Performance |
| R31-P0-011 | FactorSourcePlan 丢 concept→dataset binding | FE/DA correctness |
| R31-P0-012 | BatchPlan 把全部字段复制给每个 dataset | Planner bug |
| R31-P0-013 | Dependency extraction fail-open | Correctness |
| R31-P0-014 | FactorBatchPlan 与 ReadWavePlanner 双 planner | Architecture |
| R31-P0-015 | 1000/10000 factor “scan 数”是估算，不是实际执行 | Evidence |
| R31-P0-016 | QueryTrace 未接 canonical ReadPipeline | Observability |
| R31-P0-017 | PartitionIndex 热路径可能 rebuild manifest | Read/write boundary |
| R31-P0-018 | PartitionIndex freshness 与 unknown-byte 语义不稳 | Metadata |
| R31-P0-019 | UnitType 乘除法代数错误 | Semantic |
| R31-P0-020 | Concept/Unit 未进入 canonical SemanticField | Duplicate truth |
| R31-P0-021 | Remote HEAD 漏 STS session token | Cloud |
| R31-P0-022 | Remote HEAD client/metadata cache scope 不正确 | Cloud/perf |
| R31-P0-023 | STS `expires_at` 未形成 CredentialLease | Cloud |
| R31-P0-024 | EnvCredentialProvider 可跨 family 混拼 | Security |
| R31-P0-025 | revision availability 用 mtime clamp 伪造历史 revision | PIT |
| R31-P0-026 | ChangeImpact 对 rolling factor 的传播方向错误 | Incremental |
| R31-P0-027 | ChangeImpact 用自然日而非 trading bars | Incremental |
| R31-P0-028 | 横截面算子影响未扩散全截面 | Incremental |
| R31-P0-029 | min/max instrument 被当 exact changed set | Incremental |
| R31-P0-030 | `changed_columns=()` 语义歧义 | Incremental |
| R31-P0-031 | ExperimentSnapshot 无法证明 source 时仍生成稳定占位 | Reproducibility |
| R31-P0-032 | Calendar freeze 失败被吞 | PIT |
| R31-P0-033 | CoverageService coverage 口径错误 | Governance |
| R31-P0-034 | DQ checker 异常直接 PASS | Data quality |
| R31-P0-035 | ExecutionLease child release 不归还 parent | Resource |
| R31-P0-036 | LineageStore DuckDB 模式仍全量镜像 RAM | Memory |
| R31-P0-037 | durable lineage 失败静默降级 memory | Audit |
| R31-P0-038 | Metrics 把 result bytes 当 scan bytes / 假 histogram | Observability |
| R31-P0-039 | Unknown format 默认冒充 Parquet capability | Adapter |
| R31-P0-040 | 最终 SHA / wheel / benchmark / CI evidence 未闭环 | Release |

---


# R31-P0-001 — 把 R30 能力真正打进 distribution，并最终迁出版本型 runtime package

当前 `pyproject.toml` 使用显式 packages / package-dir，但没有 `data_access.r30`。

这会造成：

```text
repo source import 成功
wheel install 后 import data_access.r30 失败
```

### 立即修复

过渡期至少加入：

```toml
"data_access.r30"
"data_access.r30" = "r30"
```

并让 wheel inventory 动态发现所有生产包。

### 更重要的最终目标

R31 不应该长期保留：

```text
data_access.r30
```

作为第二套生产 runtime。

对每个模块建立 migration ledger：

```text
MIGRATE_TO_CANONICAL
KEEP_COMPAT_ALIAS
DELETE_DUPLICATE
DOC/EVIDENCE_ONLY
```

### 验收

```bash
python -m build
pip install dist/*.whl
cd /tmp
python -c "import data_access"
python -c "import data_access.r30"
```

迁移完成后再把 `r30` 降为 compatibility layer。


# R31-P0-002 — R30 additive layer 收敛成 canonical runtime

以下模块不能长期只作为“additive facade”：

```text
r30.session
r30.query_trace
r30.partition_index
r30.coverage_service
r30.concepts
r30.execution_lease
r30.policy
r30.lineage
r30.data_quality
r30.experiment_snapshot
```

### 迁移目标

```text
r30.session
    → read/read_session.py + runtime/read_session_context.py

r30.query_trace
    → runtime/read_pipeline.py + read/telemetry.py

r30.partition_index
    → read/metadata_plane.py + read/partition_planner.py

r30.coverage_service
    → canonical read/coverage service

r30.concepts
    → semantic_catalog / RuntimeDatasetContract

r30.execution_lease
    → runtime/resource_governor.py

r30.policy
    → security/policy.py + execution_context.py

r30.lineage
    → canonical lineage/audit layer

r30.data_quality
    → quality/ certification path
```

### 删除重复真相

迁移 parity 通过后：
```text
旧 facade → delegate canonical
再 deprecate
最后删除
```

主文档不得再要求用户理解“Core API”和“R30 API”两套。


# R31-P0-003 — 版本完全自动化：Git tag / SCM 驱动 base version

当前真实状态：

```text
pyproject base version = 0.10.2（静态）
build SHA = 自动
```

所以：
```text
0.10.2 → 0.10.3
```
仍需人工修改。

### 推荐

使用 `setuptools-scm` 或等价稳定方案。

Tag：
```text
dataaccess-v0.11.0
```

自动构建：
```text
0.11.0
```

非 tag commit：
```text
0.11.1.devN+g<sha>
```

### 不要自动“猜” breaking level

以下仍由 release policy/tag/PR label决定：

```text
major/minor/patch
API_VERSION
CONTRACT_SCHEMA_VERSION
REGISTRY_SCHEMA_VERSION
SEMANTIC_SCHEMA_VERSION
STORAGE_FORMAT_VERSION
```

可以使用：
```text
release:patch
release:minor
release:major
```
标签辅助 release workflow。


# R31-P0-004 — 建立唯一 CanonicalIdentityEncoder，修复顺序身份碰撞

当前 digest 对：

```text
list
tuple
set
frozenset
```

全部排序。

因此：
```text
(A,B) == (B,A)
```
在 digest 层发生碰撞。

### 正确 canonicalization

```text
tuple/list        → preserve order
set/frozenset     → sort
mapping           → key sort
enum              → type + value
date/datetime     → timezone-aware canonical ISO
bytes             → deterministic binary encoding
dataclass         → canonical field order
```

### 禁止

production identity fallback：
```python
repr(obj)
```
未知类型应该：
```text
UncanonicalizableIdentityError
```

### 统一替换

审计并逐步替换：

```text
r30._shared.stable_digest
factor_source_plan.stable_digest
cache key
snapshot id
policy digest
calendar digest
experiment snapshot
plan digest
```

最终所有 identity 走一个 encoder。


# R31-P0-005 — 拆开 SourceSnapshot 与 ExecutionIdentity

现在部分 `DataSnapshot` identity 混入 build SHA，而 snapshot rebuild 路径又没有完全采用同一算法。

更根本的问题：

```text
SourceSnapshot回答：数据是什么？
ExecutionIdentity回答：用什么代码/语义执行？
```

### 新模型

```python
SourceSnapshotIdentity(
    dataset,
    source_generation,
    objects/content_digest,
    semantic_data_version,
    relevant_params,
)

ExecutionIdentity(
    package_version,
    build_sha,
    semantic_execution_version,
    backend_version,
    contract_compiler_version,
)

ExperimentDataSnapshot(
    source_snapshots,
    calendar_snapshot,
    universe_snapshot,
    semantic_contract,
    execution_identity,
)
```

### 好处

改 README/无关代码：
```text
不应清空所有 data cache
```

改 PIT compiler/operator semantics：
```text
应改变 execution/result identity
```

### 必须修

`build_data_snapshot()` 和 `rebuild_snapshot_files()` 采用完全同一 SourceSnapshot identity 算法。


# R31-P0-006 — build metadata 在构建时固化，不依赖 production runtime 的 `.git`

当前：
```text
env
→ git rev-parse HEAD
→ None
```

wheel 部署时可能无 `.git`。

### 改法

build wheel 时生成：

```text
data_access/_build_info.py
```

包含：

```text
PACKAGE_VERSION
GIT_SHA
GIT_DIRTY
BUILD_TIME_UTC
BUILD_ID
```

runtime 只读它。

### production

build metadata 缺失：
```text
startup hard fail
```

不要默默 `build_sha=None`。


# R31-P0-007 — SourceBlockKey 从 PreparedRead 自动生成，禁止 caller 手工拼身份

当前 source block key 没有天然完整绑定真实读请求。

必须包含：

```text
dataset
runtime_contract fingerprint
exact source snapshot
projection
instrument scope
predicate IR digest
canonical params
time range
PIT policy
timeframe
aggregation recipe
price basis
security digest
credential scope
provider/source version
representation
```

### 实现

```python
SourceBlockIdentity.from_prepared_read(prepared)
```

不要：
```python
source_block_key(... caller manually supplies fields ...)
```

### destructive tests

```text
Close vs Volume                    → different
instrument A vs B                  → different
quarterly vs TTM                   → different
raw vs adjusted                    → different
principal/security scope different→ different
```


# R31-P0-008 — SourceBlockCache refcount 必须有真实 lease/release

当前 `get()` 会：
```text
refcount += 1
```
但没有成对 decrement。

### 方案 A（推荐）

```python
with source_block_cache.acquire(key) as block:
    ...
```

`SourceBlockLease.close()`：
```text
refcount -= 1
```

eviction：
```text
refcount > 0 → pinned / cannot evict
```

### 方案 B

如果暂不做 lease：
```text
删除 refcount 字段
```
不要保留假生命周期。

### Session exit

必须：
```text
release leases
clear session-only cache
release memory accounting
```


# R31-P0-009 — 合并 R30ReadSession 与 canonical DataReadSession

最终只允许：

```python
DataReadSession
```

统一持有：

```text
execution/security ContextVar
calendar freeze
resolution cache
prepared cache
source-block cache
experiment snapshot
query trace root
resource envelope
singleflight registry
```

`R30ReadSession`：
```text
deprecated alias → delete after migration
```

不要让 FE/用户选择“应该开哪个 session”。


# R31-P0-010 — CostModel 的 materialize / relation / rescan 三态必须真实执行

当前“relation”主要存在于模型描述，实际执行常常先 Arrow 物化。

### 真三态

#### materialize
```text
ArrowBlockRef
```

#### relation
```text
DuckDBRelationRef / backend relation reference
```
不先 Arrow materialize。

#### rescan
```text
不缓存
```

### planner 提前提供

```text
expected_reuse_count
ScanCost
remote/local
estimated bytes
available memory
```

第一次 scan 前决定，不要等第三次 miss 才发现该缓存。

### 验收

1000 consumers 共享 source：
```text
first scan 前即可选择 materialize
```


# R31-P0-011 — FactorSourcePlan 改为 typed Concept→Source Binding

当前：

```text
leaf_concepts = (...)
source_datasets = (...)
```

两个独立集合无法表达：

```text
price.close       → ashare_stock_daily
financial.revenue → ashare_stock_income
```

### 复用 FE 已有 typed binding

优先使用：

```text
ColumnSourceBinding
SourceScopeId
```

FactorSourcePlan 保存：

```python
bindings: tuple[ColumnSourceBinding, ...]
```

每个 binding 至少含：

```text
concept/column identity
dataset
market
provider
timeframe
frequency
grain
price basis
revision/source version
```

`leaf_concepts/source_datasets` 只做 derived summary。


# R31-P0-012 — 修复 BatchPlan 把全部字段复制给每个 dataset 的逻辑错误

当前逻辑相当于：

```text
datasets=[daily,income]
fields=[price.close,financial.revenue]

→ daily  [price.close,financial.revenue]
→ income [price.close,financial.revenue]
```

这是错误 source request。

### 正确

先：
```text
binding → per-dataset exact field set
```

再分组。

### destructive test

因子：
```text
price.close + financial.revenue
```

必须：
```text
daily 只读 price.close
income 只读 financial.revenue
```


# R31-P0-013 — Factor dependency extraction production fail-closed

当前 manifest import/build/parse 异常可能直接：
```text
None / skip
```

### run-mode policy

```text
interactive_research
→ degraded evidence + warning

automated_research
→ hard fail

production
→ hard fail
```

新增：
```python
DependencyExtractionEvidence(
    authoritative,
    missing_bindings,
    parser_version,
    source_manifest_digest,
)
```

任何 leaf dependency 未解析：
```text
production cannot execute
```


# R31-P0-014 — 合并 FactorBatchDataPlan 与 canonical BatchDataRequest/ReadWavePlanner

不要长期存在：

```text
FactorBatchDataPlan + FieldRequestCoalescer
```

另一边又有：

```text
BatchDataRequest + ReadWavePlanner
```

### 最终主链

```text
FactorSourcePlan
       ↓
canonical BatchDataRequest
       ↓
ReadWavePlanner
       ↓
PhysicalFactorDAG
```

`FactorBatchDataPlan` 可保留为：
```text
explain/report view
```
不再独立承担 execution planner。


# R31-P0-015 — 1000/10000 factor scan 证据必须来自实际 backend execution

当前某些 trace：
```text
physical_scan_count = source_group_count
```
只是 planner 估算。

当前 benchmark 仍可以逐 factor：
```python
session.read(...).to_arrow()
```

### 重做

真实：

```text
1000 expressions
→ FactorSourcePlan
→ BatchDataRequest
→ ReadWavePlanner
→ DataReadSession
→ backend actual execution
```

记录：

```text
planned_scan_count
actual_scan_count
actual_object_count
actual_scan_bytes
source_block_count
consumer_count
```

### 命名规则

估算只能叫：
```text
planned_scan_count
```

只有 backend/runtime counter 才叫：
```text
actual_physical_scan_count
```


# R31-P0-016 — QueryTrace 进入 canonical ReadPipeline

现在 `run_traced_read()` 是另一个 wrapper。

最终：

```text
store.read()
store.scan()
read_joined()
HTTP
FE
```

都自然进入同一 trace hook。

### ReadPipeline stage hooks

```text
auth
contract
resolution
metadata
snapshot
calendar
resource_wait
backend_wait
backend_execute
normalize
join
serialize
post_verify
```

使用：
```text
current QueryTrace ContextVar
```

没有 active trace 时：
```text
near-zero-cost no-op
```


# R31-P0-017 — PartitionIndex read path 禁止主动 build manifest

Reader 不应因为查询：
```text
隐式触发 manifest rebuild/write
```

### 正确边界

Publisher/ingestion：
```text
build + publish metadata index
```

Reader：
```text
read-only lookup
```

缺失时按 mutation owner：

```text
IMMUTABLE / EXTERNAL_VERSIONED
→ production fail or explicit certified fallback

EXTERNAL_MUTABLE
→ refresh contract
```

绝不“读的时候顺便 rebuild”。


# R31-P0-018 — Partition metadata freshness 绑定 authoritative source generation

cache key 至少：

```text
dataset
params
source_generation
manifest_digest
contract fingerprint
```

不能只靠本地 built epoch。

### unknown byte size

当前 fallback 若不知道 bytes：
```text
byte_size=0
```
是错误。

改：
```text
byte_size=None
```

Cost planner 对 None 使用：
```text
conservative estimate
```
不能把未知对象当零成本。


# R31-P0-019 — 修 UnitType 乘除法

错误规则：
```text
Money × Shares → Price
```

正确：
```text
Money / Shares → Price
Price × Shares → Money
```

### 实现

分开：
```python
multiply_result()
divide_result()
```

禁止在 multiply 中把 Shares 偷偷解释成倒数。

### 进一步

`Return × Price` 可以暂时映射 Price，但文档应说明它表示 scaled price/price delta；不要把“维度兼容”误当成“金融语义一定合理”。


# R31-P0-020 — ConceptId / UnitType 并入 canonical SemanticField compiler

不要长期：

```text
旧 SemanticField
+
r30 ConceptId/UnitType
```

### 最终 SemanticField

```text
concept_id
unit_type
market
grain
PIT
physical mapping
coverage
mining eligibility
```

生成 migration ledger：

```text
logical_name
dataset
physical_field
ConceptId
UnitType
status
```

production/mining：
```text
unmapped  → fail
ambiguous → fail
```


# R31-P0-021 — Remote HEAD 完整传 STS session token

DuckDB S3 已支持 token，但 snapshot HEAD 的 boto client 必须同样传：

```python
aws_session_token=creds.session_token
```

### 更好

Snapshot module 不直接创建 boto client。

统一：
```python
CredentialScopedS3ClientFactory
```

输入完整：
```text
access
secret
session_token
endpoint
region
ssl
credential scope
```


# R31-P0-022 — S3 client pool 与 remote metadata cache scope

每个 object HEAD 都新建 boto client，规模大时开销很高。

### Client pool key

```text
endpoint
region
credential_scope_id
access-key fingerprint
session-token fingerprint
```

### Remote metadata cache key

至少：

```text
URI
endpoint
region
credential_scope_id
security/principal scope
```

### Cache policy

```text
TTL
max_entries
LRU
negative-result short TTL
```

不能使用无界 process dict。


# R31-P0-023 — CredentialLease / STS 刷新生命周期

已有 `expires_at` 但还不够。

### 定义

```python
CredentialLease(
    credential_generation,
    scope_id,
    valid_until,
    refresh_after,
)
```

查询开始：

```text
expires_at < now + expected_duration + safety_skew
→ refresh or fail
```

长 stream/pinned read：
```text
credential validity 必须覆盖 execution
```

credential refresh 不应在正在执行的共享 connection 中间乱替换。


# R31-P0-024 — EnvCredentialProvider 按 credential family 原子解析

当前 access/secret/token 分别从 COS/AWS/S3 找第一个，存在：

```text
COS access + AWS secret
```

被拼接的风险。

### 正确

依次尝试完整 family：

```text
COS_SECRET_ID + COS_SECRET_KEY + COS_SESSION_TOKEN

AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_SESSION_TOKEN

S3_ACCESS_KEY_ID + S3_SECRET_ACCESS_KEY + S3_SESSION_TOKEN
```

某 family 配了一半：
```text
reject malformed family
```

不能混到下一 family。


# R31-P0-025 — 重写 revision availability：严禁 mtime clamp 伪造历史 vintage

错误语义示例：

```text
物理当前文件 mtime = 6/1
decision_time = 5/15
→ latest revision date 被钳成 5/15
```

这凭空创造了一个不存在的历史 revision。

### 正确模型

#### TRUE_VENDOR_VINTAGE
```text
latest vintage where knowledge_time <= decision_time
```

#### INGESTION_VINTAGE
使用 ingestion archive。

#### 只有当前文件
只能判断：
```text
当前版本是否在 decision_time 前存在
```

不能反推历史内容。

正式使用：
```text
RevisionFidelity:
NONE
KNOWLEDGE_DATE
INGESTION_VINTAGE
TRUE_VENDOR_VINTAGE
```

unknown fidelity：
```text
production conservative
```


# R31-P0-026 — ChangeImpact 对 rolling factor 必须向未来传播

当前思路类似：

```text
input[t] changed
→ recompute t-lookback ... t
```

对 rolling 算子不正确。

例如：
```text
MA20[t]=x[t-19:t]
```

x[t] 变化会影响：
```text
MA20[t ... t+19]
```

### FE operator dependency trait

```python
DependencyImpact(
    backward_input_horizon,
    forward_output_horizon,
    axis_impact,
)
```

DataAccess 只给：
```text
source change window
```

FactorEngine 沿 DAG 正向传播到 output。


# R31-P0-027 — ChangeImpact 使用 trading bars/session，不是自然日

`ts_mean(x,60)` 的 60 是：
```text
60 trading bars
```
不是：
```text
60 natural days
```

禁止：
```python
timedelta(days=60)
```

### 实现

优先在 FE axis index 上传播。

如需日期：
```text
MarketCalendar.shift_sessions()
```

A股/US 分别使用对应 calendar snapshot。


# R31-P0-028 — 横截面算子的变化影响必须扩散整个截面

一只股票数据变了：

```text
rank
zscore
neutralize
group rank
cross-sectional regression
PCA/panel model
```

同一天其他股票结果也可能变化。

### 新/复用 trait

```text
AxisImpact:
ELEMENTWISE
TIME_FORWARD
CROSS_SECTION_ALL
GROUP_CROSS_SECTION
GLOBAL_STATE
```

例：

```text
abs          → ELEMENTWISE
ts_mean      → TIME_FORWARD
rank         → CROSS_SECTION_ALL
industry_rank→ GROUP_CROSS_SECTION
panel PCA    → GLOBAL_STATE
```

ChangeImpact 按 operator DAG 传播。


# R31-P0-029 — Instrument change scope 用 typed scope，禁止 min/max 冒充 exact set

manifest：

```text
min_symbol=000001
max_symbol=688999
```

只说明范围，不代表只有这两只股票变化。

### 定义

```text
ExactInstrumentSet
InstrumentRange
PartitionWide
UniverseWide
UnknownAll
```

没有 row-level delta：
```text
不能输出 ExactSet{min,max}
```

FE 对 Range/PartitionWide 做保守 impact。


# R31-P0-030 — Column change scope 必须区分 unknown / exact / no-change

空 tuple 不能同时表示：

```text
没有字段变化
不知道哪列变化
schema没变但值变了
```

### 定义

```text
ExactColumns(...)
AllColumnsUnknown
SchemaOnly(...)
NoColumnChange
```

普通 file revision 无列级 delta：
```text
AllColumnsUnknown
```

`to_fe_input()` 禁止只取第一个 changed column。


# R31-P0-031 — ExperimentDataSnapshot 无 authoritative source 时不得生成伪稳定 identity

当前 source identity 取不到时，不应 fallback：

```text
stable_digest(dataset_name)
```

这会把“完全不知道内容”的 source 伪装成可复现。

### production

```text
required dataset source identity unknown
→ ExperimentSnapshotUnavailable
```

### research degraded

允许：
```text
authoritative=False
unknown_sources=[...]
```

但：
```text
不可 production publish
```


# R31-P0-032 — Calendar freeze production fail-closed

当前 freeze 过程中：

```text
lock_calendars() exception
→ pass
```

不可以。

### production / automated research

```text
freeze failure → fail
required market calendar missing → fail
calendar snapshot build failure → fail
```

interactive research 可以 degraded，但必须显式：
```text
authoritative=False
```


# R31-P0-033 — CoverageService 重做 coverage 口径

当前 coverage 有几个系统性问题。

### A. by_year

不能：
```text
year rows / all-history rows
```

应该：
```text
valid expected cells in year / expected cells in year
```

### B. multi-day file

一个 file 的总 row_count 不能完整复制到覆盖区间内的每一天。

### C. daily coverage

股票日频 expected days 必须来自：
```text
MarketCalendar trading sessions
```
不是自然日。

### D. field/concept coverage

`concept=` 不能只是标签。

必须检查：
```text
field exists
valid/non-null ratio
first/last valid
```

### E. instrument coverage

min/max instrument 不能形成精确 by_instrument coverage。

### 新模型

```python
CoverageCube(
    expected_sessions,
    expected_instruments,
    field_presence,
    valid_cells,
    by_date,
    by_instrument,
    by_year,
    authority,
)
```

Authority：
```text
EXACT
MANIFEST_EXACT
APPROXIMATE
UNKNOWN
```


# R31-P0-034 — DataQuality checker 崩溃不得返回 PASS

当前多个 checker：

```python
except Exception:
    return PASS
```

这是 fail-open。

### 新状态

```text
PASS
WARN
BLOCK
UNKNOWN
CHECK_FAILED
```

production certification：

```text
CHECK_FAILED != PASS
UNKNOWN != PASS
```

### 语义化字段定位

不要硬编码：
```text
high/low/volume/report_period/publish_time
```

通过：
```text
SemanticField / ConceptId
```
定位 A股/US physical fields。

### missing partition

按：
```text
MarketCalendar expected sessions
```
而不是自然日跨度。


# R31-P0-035 — ExecutionLease child release 归还 parent 资源

当前：

```text
parent request_child(60)
→ parent remaining -= 60
child.release()
→ parent 仍少 60
```

长 job 会“越跑资源越少”。

### 修法

child 保存：
```text
parent reference
allocation
lease token
```

release exactly-once：
```text
parent.remaining += allocation
```

需要 lock。

### nested child

所有资源 key 都初始化 0，不能只保存 request 里出现的 key。

### 统一字段名

统一为：

```text
memory_bytes
scan_bytes
remote_slots
duckdb_slots
temp_disk_bytes
spill_bytes
absolute_deadline
```


# R31-P0-036 — LineageStore DuckDB 模式不得同时永久保存全部 `_rows`

当前每次 record：
```text
DuckDB INSERT
+
self._rows.append()
```

长驻服务会无界增长。

### DuckDB mode

```text
不维护全量 memory mirror
count()        → SELECT COUNT(*)
search()       → SQL
to_parquet()   → COPY/SELECT
```

### memory mode

才使用：
```text
_rows
```

同时修复：
```text
进程重启后 DB 有历史行，但 _rows 空导致 count/to_parquet 错
```


# R31-P0-037 — Lineage durable storage 失败不可静默降级

生产审计需要明确模式：

```text
lineage_mode:
required_durable
best_effort
disabled
```

### required_durable

路径不可写 / DuckDB 打不开：
```text
startup hard fail
```

不能：
```text
silent memory fallback
```

### 另外修

- shared connection 写入线程安全；
- request principal 使用 ExecutionContext；
- `extra` 走 audit sanitizer；
- source_snapshot 不得 fallback 到 build_sha/dataset name。


# R31-P0-038 — Metrics 变成真实 Counter/Gauge/Histogram

当前存在：

```text
result_bytes → scan_bytes
```

语义错误。

必须分：

```text
bytes_scanned
bytes_returned
bytes_remote
bytes_cache
```

### Histogram

当前只有：
```text
count/sum/min/max/last
```
却输出 histogram 外观。

要么：
```text
真实 OTel Histogram
```

要么：
```text
固定 Prometheus buckets
```

### 类型

```text
Counter
Gauge
Histogram
```

不要全部 TYPE counter。

### 目标

真正支持：
```text
P50/P95/P99
```


# R31-P0-039 — SourceAdapter unknown format 必须 fail-closed

当前 unknown format 可能：

```text
matrix.get(fmt, parquet_default)
```

获得 Parquet capabilities。

### production

```text
unknown format
→ UnsupportedFormat
```

### scan

`build_from_clause` 失败：
```text
AdapterPlanError
```
不能：
```text
from_clause=None → 继续
```

### 能力分层

不要把以下混成一个 bool：
```text
format capability
storage capability
execution backend capability
table transaction capability
```


# R31-P0-040 — Final SHA / wheel / tests / benchmark / CI evidence 必须同一事实

最终 release 顺序：

```text
final code committed
↓
resolve exact SHA + SCM version
↓
clean clone
↓
build wheel
↓
install wheel outside repo
↓
full tests
↓
actual FE batch benchmark
↓
production-like benchmark
↓
generate evidence
↓
GitHub Actions
```

所有 evidence：
```text
exact same final SHA
```

禁止：
```text
dirty/pre-final benchmark
→ 后续 commit
→ 报告继续叫 final evidence
```


# R31-P1-001 — PolicyManifest deep immutable，外部 digest 必须验证

当前内部仍是 mutable dict。构建时 deep-freeze；任何 supplied digest 必须与 computed digest 相等，否则拒绝。Production policy 不得允许 caller 用任意字符串覆盖真实内容摘要。


# R31-P1-002 — security scope 不可异常后返回共享 None

security digest 计算失败时，production 必须 fail。Research degraded 也必须生成唯一 untrusted scope，不能把多个 principal 都落进同一个 `None` cache scope。


# R31-P1-003 — QueryTrace 错误结构化并统一 redaction

定义 `ErrorObservation(error_type,error_code,stage,retryable,sanitized_message)`；不要从字符串 `split(':')` 猜异常类型。所有 message 在进入 metrics/trace 前先脱敏。


# R31-P1-004 — Metrics cardinality policy

Prometheus label 禁止 request_id/factor_id/instrument/raw path/user email 等高基数值。只允许 market/backend/dataset_class/status/error_type/run_mode 等受控标签。细节进 Trace/Lineage。


# R31-P1-005 — Coverage/DQ/PartitionIndex/ChangeSet 共用一个 MetadataPlane

不要每个服务自己 load/build manifest。Canonical `DatasetMetadataPlane` 一次绑定同一 source generation，提供 objects/schema/coverage/DQ/partition/change tokens。


# R31-P1-006 — Source change fingerprint 不可用 files/rows/bytes 摘要冒充 authoritative

统计摘要只能 approximate。Production change detection 必须优先 source generation、manifest digest、exact object identity。


# R31-P1-007 — 引入 typed ChangeSnapshot

替换 `before: str|Mapping|None` 无限 duck-typing。定义 `ChangeSnapshot(identity,generation,objects,schema,time_axes,authority)`，DataChangeDetector 比较两个 typed snapshot。


# R31-P1-008 — DataChangeSet 区分 data/knowledge/effective/period 时间轴

不要把文件 `min_time/max_time` 自动叫 knowledge window。增加 `changed_data_time_range / changed_knowledge_time_range / changed_effective_time_range / changed_period_range`。


# R31-P1-009 — FinancialRevisionIdentity

财务 revision change 至少携带 instrument、period_end、knowledge timestamp、revision/vintage identity、changed field scope、revision fidelity。


# R31-P1-010 — Corporate Action 支持 backward impact

split/adjustment factor 变化可能改写历史 adjusted series。ChangeImpact trait 允许 FORWARD/BACKWARD/BIDIRECTIONAL/POINT_ONLY。


# R31-P1-011 — Universe change 传播到整个横截面

`universe_change` 对 rank/neutralize/portfolio exposure 等应产生 affected-session 的 CROSS_SECTION_ALL，而不是只处理某些 instrument。


# R31-P1-012 — Calendar change 使 PIT-sensitive plan/materialization identity 失效

calendar snapshot 变化可能改变 rolling bar、next_session availability、minute aggregation。必须进入 PIT-sensitive FactorSourcePlan/Experiment identity。


# R31-P1-013 — Benchmark `_last_year_range()` 不再依赖 limit=1

使用 manifest max_time 或 `MAX(TradeDate)`；物理文件顺序不能决定 benchmark 时间窗。


# R31-P1-014 — Synthetic 财务 fixture 永远保证 PubDate >= period_end

不要把跨年 Q4 PubDate 强行 clamp 到当年年末。生成 end_year+1 或移除最后不完整 period。


# R31-P1-015 — 性能 benchmark 与 PIT correctness benchmark 分离

吞吐 synthetic 可以工作日近似；PIT correctness 必须使用固定 golden calendar，覆盖春节/国庆/A股午休/US DST/early close。


# R31-P1-016 — Remote HEAD typed errors + bounded retry

区分 403/Auth、404/NotFound、429/5xx/timeout transient、CredentialExpired。Transient 可 retry+jitter，但受 absolute deadline/remote budget 限制。


# R31-P1-017 — Remote negative cache 按失败类型设 TTL

NotFound短 TTL；Transient极短或不缓存；Auth绑定 credential generation；Success正常 TTL。不要所有失败都缓存成 None。


# R31-P1-018 — S3 credential fingerprint 加 token/scope/generation

连接配置 identity 加 session token hash、credential_scope_id、credential generation/expiry generation，保证 STS rotation 后刷新。


# R31-P1-019 — Credential rotation 不在活跃 query 中替换共享 connection

按 credential-scope generation 建 pool；旧 query 用旧 lease 完成，新 query 用新 generation。


# R31-P1-020 — DataReadSession source-block singleflight

同一 session 两线程同 key miss 只能一个 producer scan，其他等待同一 future。失败向所有 waiter 传播，失败结果不缓存。


# R31-P1-021 — PreparedRead singleflight

同一 source 同时被很多 factor prepare 时，只编译/resolve/snapshot 一次。


# R31-P1-022 — 缓存区分 logical identity 与 representation identity

Arrow block、DuckDB relation、Polars lazy object 是不同 representation。Logical source 相同不代表能从同一个 representation key 直接取。


# R31-P1-023 — 共享 source block 必须 immutable/read-only

canonical cache 只存 immutable Arrow/read-only relation。需要 mutable consumer 时 copy-on-write。


# R31-P1-024 — Batch compatibility 全部 typed，禁止空字符串默认

使用 PITPolicy.UNKNOWN、UniverseScope、SnapshotIdentity、SecurityScopeId 等 typed values。Production UNKNOWN 不得 merge。


# R31-P1-025 — Aggregation compatibility 使用完整 ordered recipe digest

不能只取 `aggs[0]`。所有 aggregation steps、window、session、missing/halt policy 都进入 digest。


# R31-P1-026 — Join contract 进入 batch compatibility

不同 ASOF policy、tolerance、join key、grain、revision policy 不得合并成同一 source block。


# R31-P1-027 — Provider/source version 进入 SourceScopeId

相同 dataset 名来自 Wind/Massive/其他 provider 或不同 source version 时不能互相复用。


# R31-P1-028 — Multi-market batch 使用 MarketScope，不取第一个 market

A股+US batch 标记 MultiMarket，并强制 CalendarAlignment/CurrencyPolicy/CrossSectionPolicy。


# R31-P1-029 — Coverage 使用 grain-specific expectation

daily 用 expected trading sessions；financial 用 expected periods/filings；event 用 event model；不要所有数据统一自然日格子。


# R31-P1-030 — Coverage authority 明确 EXACT/MANIFEST_EXACT/APPROXIMATE/UNKNOWN

严格 mining gate 默认不接受 UNKNOWN；approximate 不能对外表现成精确覆盖率。


# R31-P1-031 — Field-level coverage 是一等结果

dataset files齐全不等于 ROE/Revenue 等字段有值。记录 field valid ratio、first/last valid、by-date/by-instrument non-null coverage。


# R31-P1-032 — DQ checker 先通过 Concept/SemanticField 归一化列角色

不要因 physical column 大小写/别名不同而静默跳过 OHLC/PIT/currency check。


# R31-P1-033 — DQ 分 publish-time full / query-time light / periodic deep

大分钟数据不要每 query 全表 DQ。查询热路径消费已有 source certification；深 DQ 在 ingestion/publish/定期任务。


# R31-P1-034 — DQ certification 绑定 exact source snapshot

DQResult 加 source_snapshot_id、contract_digest、DQ_RULESET_VERSION；source变更自动失效旧 certification。


# R31-P1-035 — Lineage retention / partition / compaction

正式运行后 lineage 数据会大。提供 retention、按日期/job partition、compaction、archive/index，不让单表无限 append。


# R31-P1-036 — Audit 与 Lineage 明确责任边界

Audit=security/compliance；Lineage=data/result provenance。共享 ExecutionIdentity/SourceIdentity/PrincipalIdentity，不复制两套所有字段。


# R31-P1-037 — ExperimentDataSnapshot deep immutable

frozen dataclass里的 dict仍可变。改 FrozenMap/tuple entries/MappingProxy，构造后内容不可变。


# R31-P1-038 — ExperimentSnapshot production completeness gate

required datasets、calendar、universe(when required)、semantic/registry/execution/security identity必须完整；缺一项不能生成 authoritative snapshot。


# R31-P1-039 — Schema/API/Storage 版本增加 reader/writer compatibility metadata

增加 min_reader_version/min_writer_version/migration_from/backward_compatible/forward_compatible；storage manifest升级必须有migration test。


# R31-P1-040 — 关键 identity 不只用 16 hex

16 hex=64-bit，长期大量 artifact/cache identity偏短。关键 source/experiment/policy/security identity 建议 32 hex/128-bit 或 full SHA256。


# R31-P1-041 — 禁止 production identity fallback 到 repr()

自定义 repr可能含地址/非稳定字段。未知类型 production应抛 UncanonicalizableIdentityError。


# R31-P1-042 — Backend/Source capabilities 必须有实际 certification

不能只硬编码 DuckDB/Polars 支持。记录 backend version + feature test evidence，Planner只消费 certified capabilities。


# R31-P1-043 — CSV production schema 必须显式

禁止生产依赖 sample auto-infer。CSV contract明确 schema/delimiter/encoding/header/null/timestamp。


# R31-P1-044 — CSV/Arrow IPC 也必须有 SourceSnapshot identity

所有格式都需要 size/mtime/checksum/version；snapshot capability不应等价于“是不是Parquet”。


# R31-P1-045 — Format/Storage/Backend capability 三层拆分

Parquet format、S3 storage、DuckDB engine的 snapshot/transaction/remote能力不是同一个层次。Planner组合三层能力。


# R31-P1-046 — Startup gate 覆盖 R31 invariants

校验 build_info、version compatibility、calendar freeze、durable lineage、policy manifest、credential lease、source adapter certification、schema migration availability。


# R31-P1-047 — FE↔DA Resource Envelope 字段统一

统一 memory_bytes/scan_bytes/remote_request_budget/remote_concurrency/duckdb_concurrency/temp_disk_bytes/spill_bytes/absolute_deadline。


# R31-P1-048 — ExecutionLease 并发分配必须原子

request_child/release/remaining用锁；并发线程不能同时看到同一 remaining 后重复超配。


# R31-P1-049 — 父取消向所有 child/remote/backend 传播

复用FE Cancellation/Deadline机制，父job cancel后DuckDB、remote、spill、child lease全部及时取消。


# R31-P1-050 — 全请求只有一个 absolute deadline

HTTP/QueryBudget/Lease/DuckDB/remote不能各有独立完整timeout。统一 DeadlineContext，所有阶段只取 remaining。


# R31-P1-051 — 真实 FE integration benchmark

benchmark从 expression compile 开始，经过 SourcePlan/ReadWave/DA/operator/materialize，而不是单独for-loop调用DA。


# R31-P1-052 — 明确 cold/warm 层级

区分 OS cache warm、DuckDB warm、DataReadSession warm、mirror warm、S3 metadata client warm，分别报告。


# R31-P1-053 — Synthetic benchmark 固定 seed + fixture fingerprint

evidence记录random seed、schema、row count、fixture hash；否则两次benchmark不可比较。


# R31-P1-054 — benchmark 主操作只用 public API

private registry/internal paths只用于fixture setup，不作为被测用户路径。


# R31-P1-055 — benchmark source groups 必须由真实 FE planner 产生

不要脚本用列组合手工算source_group_count。计划必须来自production planner。


# R31-P1-056 — 保留真实 COS production-like benchmark

CI synthetic之外，定期在真实A股/COS服务器跑cold/warm/profile benchmark。


# R31-P1-057 — ServerPerformanceProfile

记录CPU/cores/NUMA/RAM/disk/network/COS region/DuckDB threads/FE workers，performance baseline绑定profile。


# R31-P1-058 — FactorArtifact 绑定 ExperimentSnapshot/SourcePlan/ReadWave/Execution identity

物化后可明确复现“数据、代码、plan、calendar、universe”。


# R31-P1-059 — CSE只跨兼容 scope

不同security/PIT/universe/provider/price basis绝不因为字段相同就合并。


# R31-P1-060 — Factor identity 与 DataDemand identity 分离

两个factor不同但数据需求相同，应有不同FactorPlanIdentity、相同DataDemandIdentity，利于CSE。


# R31-P1-061 — Dependency unresolved 禁止 default dataset fallback

不能用字符串 `default` 假装source已解析。Production hard fail。


# R31-P1-062 — Grain 与 Timeframe 分离

`(date,instrument,period)`是grain；quarterly/TTM是timeframe。不要互相fallback。


# R31-P1-063 — PIT缺失用 UNKNOWN，不是 none

none只能表示明确不需要PIT；unknown表示没有证明。Production unknown不得直接执行。


# R31-P1-064 — PriceBasis 从 SemanticField/Concept 推导

adjusted_close等concept不能因planner没填就默认raw。


# R31-P1-065 — Join结果必须做 ResultGrain certification

join后检查1:1/N:1、duplicate keys、row explosion，FactorEngine只消费certified panel grain。


# R31-P1-066 — Backend switching只改变 ExecutionIdentity

同一logical source在DuckDB/Polars parity通过时SourceSnapshot相同，ExecutionIdentity记录backend/version。


# R31-P1-067 — Result cache 不因任意 build commit 全失效

引入 SemanticExecutionVersion。README变更不应清结果cache；PIT/compiler/operator semantic变更才应失效相关结果。


# R31-P1-068 — Metadata cache 与 result cache 的 invalidation 分层

metadata由source generation控制；result由source+semantic execution控制。


# R31-P1-069 — Current-state docs 只描述 canonical runtime

R31完成后R30文档进入archive/history，不再作为新开发者理解主架构的入口。


# R31-P2-001 — 中期标准化 package layout 到 src/data_access

当前非标准 physical-dir→dotted-package 映射导致 package list 漏项风险。

中期可迁移：

```text
dataaccess/src/data_access/
    core/
    read/
    ...
```

然后标准 package discovery。

但这是大搬迁：
```text
不要在R31前半段做
```
先完成 correctness/canonical convergence，再单独 migration。


# R31-P2-002 — MetadataStore Provider，为未来多服务器留接口

未来多server时：

```text
manifest
partition index
coverage
DQ certification
lineage
```

不能只靠各机本地文件。

先定义：
```python
MetadataStore
```
当前 local implementation，未来可换 object store/DB。


# R31-P2-003 — Distributed singleflight 只预留协议

未来多server同时请求同一cold COS block可以共享materialization claim。

当前不要上重型分布式系统，只定义：
```text
MaterializationClaimProvider
```
默认 local。


# R31-P2-004 — Hybrid historical + live snapshot

未来模拟盘/实盘：

```text
HistoricalGeneration
+
LiveSequenceCheckpoint
```

组成：
```text
HybridSourceSnapshot
```

ChangeSet也支持：
```text
sequence_offset
```

当前先定义接口与identity，不必一次实现完整流系统。


# 120. 必须新增的 destructive tests

## T-R31-ID-001：序列顺序

```text
digest((A,B)) != digest((B,A))
digest({A,B}) == digest({B,A})
```

## T-R31-CACHE-001：SourceBlock identity

```text
Close vs Volume → different
instrument A vs B → different
quarterly vs TTM → different
raw vs adjusted → different
security scope A vs B → different
```

## T-R31-CRED-001：STS HEAD

只给：
```text
access + secret + session_token
```
fake S3 才允许 HEAD success。

漏 token test 必须 fail。

## T-R31-CRED-002：credential family

```text
COS_ID + AWS_SECRET
→ reject
```

## T-R31-PIT-001：revision

当前 file mtime 晚于 decision time：
```text
不得返回 fabricated decision-date revision
```

## T-R31-IMPACT-001：rolling forward

```text
x[t] changed
MA20 → t..t+19 outputs affected
```

## T-R31-IMPACT-002：trading bars

跨周末/节假日：
```text
20 bars != 20 natural days
```

## T-R31-IMPACT-003：cross-section

一个 symbol ROE 改变：
```text
rank(ROE) → 当日整个 universe affected
```

## T-R31-IMPACT-004：instrument scope

只有 min/max metadata：
```text
不得输出 ExactSet{min,max}
```

## T-R31-COVERAGE-001

完整A股交易日数据：
```text
daily overall coverage = 1.0（容差内）
```

不能被周末拖低。

## T-R31-DQ-001

checker人为抛异常：
```text
result != PASS
```

## T-R31-LEASE-001

```text
parent 100
child 60
child release
parent remaining = 100
```

## T-R31-LINEAGE-001

```text
required_durable + unwritable path
→ hard fail
```

## T-R31-METRIC-001

```text
bytes_scanned != bytes_returned
```
分别记录。

## T-R31-ADAPTER-001

```text
format="magic"
→ UnsupportedFormat
```

## T-R31-SNAPSHOT-001

unknown required source：
```text
production ExperimentSnapshot build fails
```

---

# 121. 真正的 FE×DA 批量执行验收

构造至少 1000 factors：

```text
400 price-only
300 price-volume
200 fundamental
100 minute-derived
```

完整走：

```text
Expression compile
→ FactorSourcePlan
→ typed SourceBinding
→ BatchDataRequest
→ ReadWavePlanner
→ DataReadSession
→ backend
→ operator DAG
```

输出：

```text
factor_count
unique concepts
planned source groups
planned scans
actual scans
actual objects
actual scan bytes
source block cache hits
source block producers
consumer count
DA wall time
FE compute time
peak RSS
```

验收核心：

```text
actual scan count ≈ compatible source groups
actual scan count << factor count
```

---

# 122. ChangeImpact trait 测试矩阵

至少覆盖：

```text
ELEMENTWISE
TS_WINDOW
CS_RANK
GROUP_NEUTRALIZE
PANEL_MODEL
CORPORATE_ACTION
UNIVERSE_CHANGE
CALENDAR_CHANGE
```

每个测试：

```text
affected dates
affected instruments
affected group/universe
forward/backward horizon
```

---

# 123. 永久 A/US PIT/Revision tests

A股：

```text
PubDate
ReportPeriodEndDate
YTD flow
RevisionFidelity
Universe PIT
```

US：

```text
filing_date
period_end
timeframe
USD
date-label availability
StockCapital split/shares
```

禁止任何测试依赖：
```text
file mtime == historical knowledge time
```

---

# 124. Evidence Truth 契约

每份 evidence 必须写：

```text
code_sha
package_version
build_id
fixture_hash
machine_profile
planner_digest
actual_counter_source
run_timestamp
```

如果某值是 estimate：

```text
planned_*
estimated_*
```

绝不命名：
```text
actual_*
physical_*
```

---

# 125. 静态审计脚本

新增/扩展：

```text
audit_r30_parallel_runtime.py
audit_identity_encoders.py
audit_dataaccess_r30_imports.py
audit_fe_source_binding.py
audit_change_impact_traits.py
audit_dq_failopen.py
audit_lineage_durability.py
audit_version_sources.py
audit_cache_identity.py
audit_metric_semantics.py
```

---

# 126. `except Exception` 专项审计

### 可 best-effort

```text
optional metrics export
debug telemetry
non-critical docs
```

### 默认不得吞

```text
PIT
security
snapshot
canonical identity
dependency extraction
change impact
credential
durable lineage
DQ certification
policy
version compatibility
```

所有吞异常点必须有：
```text
明确 rationale
run-mode distinction
test
```

---

# 127. Identity 专项审计

repo-wide 搜索：

```text
hash(
repr(
json.dumps(
sha256(
stable_digest
fingerprint
cache_key
snapshot_id
```

每个 identity 分类成：

```text
SourceIdentity
ExecutionIdentity
SecurityIdentity
PlanIdentity
CacheIdentity
ArtifactIdentity
```

不能同一个 build SHA/registry hash 随意混入所有身份。

---

# 128. 性能专项审计

搜索：

```text
glob
stat
ParquetFile
boto3.client
head_object
to_pandas
tempfile parquet
build_manifest
```

判断它是否发生在：

```text
per factor
```

应该尽量变成：

```text
per session
per source group
per generation
```

---

# 129. R31 实施阶段

## Phase A — Correctness first

```text
CanonicalIdentityEncoder
SourceBlock identity
typed SourceBinding
revision PIT
ChangeImpact
STS credentials
DQ fail-open
ExecutionLease
```

## Phase B — Canonical convergence

```text
R30ReadSession → DataReadSession
QueryTrace → ReadPipeline
PartitionIndex → MetadataPlane
Concept/Unit → SemanticField
FactorBatchPlan → BatchDataRequest/ReadWavePlanner
```

## Phase C — Performance truth

```text
real batch execution
source/prepared singleflight
S3 client pool
session source-block reuse
```

## Phase D — Governance/Operations

```text
Coverage
DQ certification
Lineage durability
Metrics
Policy
ExperimentSnapshot
```

## Phase E — Version/Release

```text
SCM tag version
build_info
wheel
CI
final-SHA evidence
```

---

# 130. 最优先的 15 项

如果任务过大，至少先：

```text
1  CanonicalIdentityEncoder
2  SourceBlockKey from PreparedRead
3  Concept→Dataset typed SourceBinding
4  FactorBatchPlan/ReadWavePlanner 收敛
5  真实 1000/10000 factor scan CSE
6  Revision vintage/PIT 修正
7  ChangeImpact forward trading-bar propagation
8  Cross-sectional AxisImpact
9  STS token/client pool/CredentialLease
10 DQ fail-open 修正
11 Coverage trading-session/field coverage 修正
12 ExecutionLease parent resource return
13 Lineage durable + no full-memory mirror
14 SourceSnapshot vs ExecutionIdentity 拆分
15 SCM/tag version + final-SHA CI/evidence
```

---

# 131. 最终目标架构

```text
┌─────────────────────────┐
│       FactorEngine      │
│ expression/operator DAG │
└────────────┬────────────┘
             │ typed SourceBindings
             ▼
┌─────────────────────────┐
│    FactorSourcePlan     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   BatchDataRequest      │
│   ReadWavePlanner       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    DataReadSession      │
│ prepared/source CSE     │
│ ExperimentSnapshot      │
│ Resource/Trace          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Canonical ReadPipeline  │
│ Contract/PIT/Snapshot   │
│ Metadata/Security       │
└────────────┬────────────┘
             │
      ┌──────┼──────┐
      ▼      ▼      ▼
   DuckDB  Polars  Arrow
      └──────┼──────┘
             ▼
   Governed Source Blocks
             ▼
      Factor Execution
```

---

# 132. 最终 Closure Report

必须生成：

```text
DATAACCESS_R31_FINAL_ACCEPTANCE_REPORT.md
```

至少包含：

## Baseline

```text
start SHA
final SHA
SCM/package version
build id
clean git status
```

## Closure Ledger

每项：

```text
DONE
ALREADY CLOSED
MIGRATED
DELETED_DUPLICATE
DEFERRED
BLOCKED
```

## Canonical convergence

明确：

```text
哪些 r30 模块迁入 canonical
哪些仅 alias
哪些删除
```

## Version

```text
tag/scm test
wheel version
build_info
```

## Correctness

```text
PIT
revision
ChangeImpact
unit
coverage
DQ
security
snapshot
```

## Batch performance

至少：

```text
100
1000
10000 factors
```

真实记录：

```text
planned groups
actual scans
actual bytes
source block reuse
DA time
FE time
peak RSS
```

## Cloud

```text
STS HEAD
credential rotation
client reuse
expiry handling
```

## CI

必须是 final SHA。

若没有：
```text
NO CURRENT-HEAD CI EVIDENCE
```

---

# 133. Freeze 条件

只有以下同时成立，才允许：

```text
DATAACCESS CORE FREEZE = YES
```

1. R31 所有 P0 关闭；
2. `r30` 不再是第二套 production runtime；
3. wheel 包含全部生产模块；
4. SCM/version/build_info 一致；
5. revision/PIT destructive tests 通过；
6. ChangeImpact 时间、bar、横截面传播正确；
7. 真实 FE batch CSE 有 actual scan evidence；
8. STS/credential lifecycle 正确；
9. Coverage/DQ 不再错口径/fail-open；
10. lineage durability contract 明确；
11. current final SHA 有 GitHub CI evidence；
12. benchmark/evidence 与 final SHA 完全一致。

---

# 134. 最终一句

R31 不应该继续把 DataAccess 变得更“大”。

真正应该做的是：

> **删除平行真相、修正增量/PIT/身份/凭证这些二阶高风险问题，把 R30 已经开发出来的高级能力收敛进唯一 canonical runtime，并用真实 FactorEngine 批量执行和 final-SHA evidence 证明它真的工作。**

最终成功标准不是：

```text
有多少类
```

而是：

```text
一个数据事实只定义一次
一个请求只经过一条治理链
一批因子只准备一次共享数据
一个版本/快照/证据都能精确复现
```
