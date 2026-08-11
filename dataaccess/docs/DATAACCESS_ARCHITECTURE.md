# DataAccess 系统架构（current-state）

> 本文档只描述**当前系统**（`data_access` 包，版本 `__version__`，构建 SHA `__build_sha__`）。
> 不讲历史修复过程。历史决策过程见 `docs/R2*.md` 报告。
> 目录：`dataaccess/`（仓库根），包名 `data_access`。唯一对外入口 `data_access.get_store()`。

---

## 1. 总览

DataAccess 是团队统一的**数据读写层**：把「数据集物理位置/格式/时间语义」登记在
`config/datasets.yaml`（单一真源），业务代码只写数据集名。读路径从登记表出发，
编译出不可变的执行契约，经过安全/预算/快照治理后落到三种执行后端
（DuckDB / Polars / Arrow）。写路径只允许写入 `namespaced`/`staging` 数据集，
`published` 必须走 publish 原子晋升。

```
config/datasets.yaml ──► DatasetRegistry（登记表）
                              │
                              ▼
                 ContractIR（统一契约视图：registry + COS 契约 + 语义字段）
                              │
        DataRequest / ReadPlan / PhysicalPlan（逻辑需求 → 物理计划 DAG）
                              │
                     DataAccessStore（治理 + 调度 + 执行）
        ┌───────────────┼───────────────┬────────────────┐
        ▼               ▼               ▼                ▼
   read/           runtime/         security/        snapshot/
（读路径模块）   （预算/管线/缓存）  （身份/授权）    （快照解析/校验）
        └───────────────┼───────────────┬────────────────┘
                        ▼               ▼
                  三后端：DuckDB / Polars / Arrow（pyarrow）
```

---

## 2. 分层架构

### 层 1 — Dataset Registry（登记表）

| 文件 | 职责 | 关键类 |
|---|---|---|
| `config/datasets.yaml` | 数据集单一真源（路径/格式/分区/权限/治理字段） | — |
| `registry/loader.py` | 解析 YAML → 对象模型；strict key 校验；env/namespace 展开 | `DatasetRegistry`, `StaticDataset`, `ParametricDataset`, `Dataset` |
| `registry/paths.py` | 路径 canonicalize / env 展开 / namespace 解析 / 路径白名单 | `PathAuthorizer` |
| `registry/schema_validation.py` | declared schema 校验（首访自检） | `check_schema`, `enforce_schema_or_raise` |
| `registry/params_validation.py` | parametric 数据集参数硬校验 | `ParamSpec`, `validate_params` |
| `registry/layout_policy.py` | hive/bucket 布局策略解析 | `LayoutPolicy` |
| `core/storage.py` | 存储后端抽象（local/s3/cos/http/cli/clickhouse）→ `StorageSpec` | `StorageSpec` |

数据集对象携带的治理字段（current-state）：

- `access_mode`：`published`（只读）/ `namespaced`（个人隔离）/ `staging`（发布前暂存）；
- `mutation_owner`：`dataaccess` / `external_versioned` / `external_mutable` / `immutable`
  —— 决定 manifest 新鲜度语义（见 §3.3）；
- `generation_pointer` / `generation_required`：generation 指针模型（`factor_matrix`）；
- `schema_migrations`：跨 epoch schema 演进审批（approved 才放行）；
- `storage`（`StorageSpec`）/ `partitioning`（`PartitionSpec`）/ `format`（`FormatSpec`）。

### 层 2 — Contract IR（统一契约视图）

| 文件 | 职责 | 关键类 |
|---|---|---|
| `read/contract_ir.py` | 把 registry + COS 契约 + 语义字段三套声明编译成一个统一视图 | `ContractIR`, `ContractIRDataset`, `TemporalAxes`, `ContractCompiler` |
| `contract/runtime_contract.py` | 运行时数据集契约（v2 字段） | `RuntimeDatasetContract` |
| `contract/physical_partition.py` | 物理分区规格 | `PhysicalPartitionSpec` |
| `contract/temporal_axis.py` | 时间轴规格 | `TemporalAxisSpec` |
| `contract/filters.py`, `file_selector.py` | 过滤需求 / 文件选择 | `FilterRequirement` |

### 层 3 — DataRequest / ReadPlan / PhysicalPlan

| 文件 | 职责 | 关键类 |
|---|---|---|
| `read/data_request.py` | 一次逻辑数据需求；`plan()` 产出 `ReadPlan` | `DataRequest`, `ReadPlan` |
| `read/physical_plan.py` | 编译成物理节点 DAG（scan→filter→temporal join→agg→normalize→project） | `PlanNode`, `build_physical_plan` |
| `read/predicate.py`, `predicate_ast.py` | 谓词编译 / AST | `Predicate`, `compile_predicate` |
| `read/key_policy.py` | `(timestamp, instrument)` 键策略 | `KeyPolicy`, `resolve_key_policy` |

### 层 4 — DataAccessStore（对外唯一入口）

`store.py` 定义 `DataAccessStore`（进程单例，`get_store()` / `reset_store()`）。

- **读**：`read_arrow` / `read_frame` / `read`（auto 路由）/ `read_auto` / `read_uri` /
  `read_joined` / `read_factors` / `read_cached` / `load_columns` / `scan` /
  `scan_polars` / `sql_relation` / `read_asof`；
- **写**：`write_arrow`（overwrite/append，只允许 namespaced/staging）、`upsert`、
  `publish_from_staging`、`delete_rows`；
- **规划/契约**：`prepare_read` → `execute_prepared_read`（唯一执行链）、`plan`、
  `resolve_fields`、`contract_ir` / `contract_ir_fingerprint`、`coverage`、
  `build_dataset_manifest`。

所有 public 读路径都走 **prepare → execute** 两段式：先产出不可变的
`PreparedRead`（已决议的 contract + security + physical scope + budget），再执行。

### 层 5 — 三后端

| 后端 | 入口 | 机制 |
|---|---|---|
| DuckDB | `core/engine.py`（`DuckDBEngine`，进程单例共享 buffer pool）；SQL 由 `read/formats.py` 的 `FormatAdapter` 生成 | 主执行引擎；`read_parquet(?)` 表函数 + hive/union 选项 |
| Polars | `read/scan_handle.py` `ScanHandle`；`read/relation_handle.py` `RelationHandle`；`store.scan_polars` | 受控扫描；production 下 collect 必经 budget + snapshot 校验 |
| Arrow | `store.read_arrow` / `read_handle.py` `ReadHandle` | pyarrow Table 直出；`read_frame`/`to_pandas` 是终端输出转换（读管线结束后才物化） |

`read/read_auto_router.py` + `read/scan_cost.py` 按成本估算（行数/字节/远程/引擎启动）路由
engine + result_mode（arrow / stream / polars / lazy）。

---

## 3. 治理

### 3.1 安全（security）

| 文件 | 职责 | 关键类 |
|---|---|---|
| `security/principal.py` | 身份 + 动作最小集合（`dataset:read` / `factor:read` / …）；权限取交集 | `DataPrincipal`, `AccessPolicy`, `ACTION_*` |
| `security/policy.py` | 逻辑授权：authorize_dataset / authorize_uri；deny 不 fallback | `DatasetAuthorizer` |
| `security/credentials.py` | CredentialProvider 契约；production 禁自动解析 `~/.cos.yaml` | `CredentialProvider`, `CredentialMaterial` |
| `security/execution_context.py` | request-scoped 嵌套执行上下文（ContextVar） | `DataAccessExecutionContext`, `execution_scope` |
| `security/governed_frame.py` | 可信 provenance 读句柄 | `GovernedFrame`, `ExecutionEnvironmentIdentity` |
| `security/run_mode.py`, `runtime/mode_identity.py` | 运行模式 + request-scoped run-mode 单一权威 | `RunMode`, `RuntimeModeIdentity` |

授权链：`Principal ∩ AccessPolicy ∩ Registry 路径边界 ∩ IAM/STS ∩ 本地 cache 所有权`，
任何一层 deny 即 deny。

### 3.2 PIT / Snapshot / Manifest / Generation / COW

| 文件 | 职责 | 关键类 |
|---|---|---|
| `read/session_calendar.py` | 交易所 session / 交易日历（时区感知）；`available_from` 编译 | `MarketSession`, `MarketCalendar` |
| `read/pit_event_index.py` | 美股财务 PIT 物理索引（filing_date 精确裁剪） | `PITEventIndex`, `PITEventRecord`, `build_pit_event_index` |
| `snapshot/resolver.py` | SourceSnapshot 解析链（latest / pin / fail_if_changed） | `SourceSnapshotResolver` |
| `snapshot/source_snapshot.py` | 解析后的精确物理对象集 | `ResolvedObject`, `ResolvedSourceSnapshot` |
| `snapshot/verifier.py` | 执行前/后统一快照验证 | `SnapshotVerifier` |
| `read/manifest.py` | `_manifest.parquet` 文件级 min/max 元数据（时间/标的裁剪） | `build_manifest_for_dataset`, `prune_by_time`, `prune_by_instruments` |
| `write/generation.py` | generation_pointer 数据集的原子代写（写全代 → 原子 flip manifest） | `write_generation`, `delete_generation` |
| `write/publish_manifest.py` | publish 后原子 manifest（审计/回滚/对账） | `MANIFEST_NAME` |

- **Manifest**：读路径只在 manifest 存在且新鲜时用它裁剪文件（计数比对判断过期），
  构建是显式操作（`store.build_dataset_manifest`）。
- **Generation/COW**：`generation_pointer: true` 的数据集读路径只解析
  `manifest.json` 的 `generation` 指针指向那一代目录，写路径写完整不可变代后原子
  flip 单指针（copy-on-write 语义）。`generation_required: true` 时 manifest 缺失
  → fail-closed，禁止 legacy fallback。
- **PIT**：`session_calendar`（日频）与 `pit_event_index`（美股财务按 filing_date）
  为两层 PIT 物化；`read_asof` / `read_joined` 消费。

### 3.3 QueryBudget / Resource Governor / SchemaEpoch / Semantic Contract

| 文件 | 职责 | 关键类 |
|---|---|---|
| `read/query_budget.py` | 读路径预算（行数/字节/耗时/列/时间窗）；hard 拦截 | `QueryBudget`, `resolve_query_budget`, `is_strict_semantics` |
| `runtime/resource_governor.py` | 进程级最小全局 governor（并发查询/内存/扫描字节/远程并发） | `GlobalResourceGovernor`, `ResourceReservation`, `duckdb_slot` |
| `runtime/cache_manager.py` | 统一 cache lifecycle（pin/GC/quota/TTL/隔离） | `CacheManager`, `CacheEntry` |
| `runtime/startup_gate.py` | production 启动门（registry/契约审计/凭证/日历权威/快照可达） | `StartupGateResult` |
| `read/schema_epoch.py` | 逐物理 object 读 footer → schema 指纹 → epoch 分组；跨 epoch 兼容门 | `SchemaEpoch` 校验逻辑 |
| `read/semantic_catalog.py` | 字段语义单一事实源（dtype/frequency/grain/单位/时间语义/PIT·join） | `SemanticFieldCatalog`, `SemanticField`, `UnitSpec` |
| `read/coverage.py` | 数据集覆盖/完整性/陈旧度（declared vs observed） | `coverage` 判定 |
| `read/metadata_plane.py` | sidecar/推导结果统一查询入口 | `metadata_plane` |
| `read/query_cache.py` | 查询结果 LRU + TTL（默认关闭） | `QueryResultCache` |

---

## 4. 会话（DataReadSession + R30 层）

### 4.1 DataReadSession

`read/read_session.py` 定义 `DataReadSession`（job 级读会话）：

- 绑定 request-scoped 执行上下文（principal/authorizer/credential/run_mode），
  嵌套读自动继承请求级授权；
- job 首读即冻结 calendar 世界（PIT 语义跨因子稳定）；
- **resolution 缓存**注入：`(dataset, params, time_range, instruments) → (paths, files)`，
  同一批因子重复读同一 dataset 时 prepare ONCE。

resolution 缓存在 **ContextVar**（`runtime/read_session_context.py`）承载，并发
session 互不覆盖；Store `_resolution_cache` 仅向后兼容显示。

### 4.2 R30 成熟度层（`data_access.r30`）

`r30/__init__.py` 声明 R30 全维度成熟度层命名空间（additive，不改既有文件）：

- `r30/_shared.py`：R30 共享工具 —— `stable_digest` / `stable_digest_full`（确定性摘要）、
  `security_scope`（折叠执行上下文授权范围）、版本治理常量
  （`API_VERSION` / `CONTRACT_SCHEMA_VERSION` / `REGISTRY_SCHEMA_VERSION` /
  `SEMANTIC_SCHEMA_VERSION` / `STORAGE_FORMAT_VERSION`）与 `version_gate()`。

声明的子模块（`concepts`/`specs`/`calendar_snapshot`/`universe_snapshot`/
`experiment_snapshot`/`coverage_service`/`partition_index`/`query_trace`/`session`/
`data_change`/`change_impact`/`adapter`/`cache_hierarchy`/`execution_lease`/
`lineage`/`policy`/`data_quality`/`mining_profile`/`artifact_meta`/`training`/
`multi_asset`/`distributed`/`api_surface`）为按需惰性加载的扩展点；当前具体落地的
「会话级 prepared/source-block 复用」机制在以下既有模块：

| 文件 | 职责 | 关键类 |
|---|---|---|
| `runtime/prepared_read.py` | immutable executable plan（已决议的全部状态） | `PreparedRead`, `PredicateConstraint` |
| `runtime/read_pipeline.py` | 统一读执行链（snapshot resolve → budget → governor admit → verify → execute → verify） | `ReadPipeline` |
| `runtime/read_session_context.py` | request-scoped resolution cache（ContextVar） | `get_resolution_cache` / `set_resolution_cache` |

---

## 5. 目录速查（module map）

| 目录/文件 | 职责 |
|---|---|
| `core/` | 引擎（`engine.py`）、存储抽象（`storage.py`）、原子（`atomic.py`）、审计（`audit.py`）、重试（`retry.py`）、命名空间（`namespace.py`）、DuckDB 配置/能力 |
| `read/` | 读路径：契约/请求/计划/谓词/预算/PIT/日历/manifest/coverage/schema_epoch/语义字段/三后端句柄 |
| `runtime/` | 执行链：prepared/read_pipeline/资源 governor/缓存 manager/startup gate/mode identity |
| `security/` | 身份/授权/凭证/执行上下文/provenance |
| `snapshot/` | 快照解析/校验（local + remote exact objects） |
| `write/` | 写路径：generation 原子代写 / upsert / publish / mutation_lock / publish_manifest |
| `registry/` | 登记表解析/路径/参数/schema/布局 |
| `contract/` | typed contract IR 构件（runtime/physical/temporal/filters/file_selector） |
| `r30/` | R30 成熟度层命名空间（当前 `_shared` 已落地） |
| `quality/` | 企业级数据质量契约检查（Arrow 层，不绕开 DataAccess） |
| `service/` | HTTP 读数服务（服务端 + 轻量客户端） |
| `cos/`, `cos_contract*.py`, `cos_*_runtime.py` | COS mirror/remote 与面板/事件 PIT 契约运行时 |
| `clickhouse/` | ClickHouse 执行后端 |
| `config/` | `datasets.yaml`（登记表单一真源）、`semantic_fields.yaml`（字段语义单一真源） |
| `benchmarks/`, `tests/` | 基准 workload 与测试 |

---

## 6. 版本与身份

- 包版本 `data_access.__version__` = `full_version(_base_version)`，含
  `build_sha`（R29-P0 #207：SCM/commit 驱动版本，`data_access._build_meta`）。
- `DataAccessStore.contract_ir_fingerprint()` / `registry_fingerprint()` /
  `semantic_catalog.fingerprint()`：部署间可对比的语义版本。
- `DataReadSession` 与 `PreparedRead` 携带 `security_digest`；快照/缓存按
  principal/security_scope 隔离。
